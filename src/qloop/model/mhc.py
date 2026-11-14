"""Loop-back bridge built on the mHC (manifold-constrained hyper-connection) idea.

Why: the core's output residual (after physical layer `core_end-1`) has a much larger norm
and a different distribution than what layer `prelude_end` saw in pre-training. Feeding it
straight back ("hard-wiring") saturates DeltaNet decay gates and confuses MoE routers.

mHC's key property is *non-expansiveness*: residual mixing matrices are projected onto the
Birkhoff polytope (doubly-stochastic, via Sinkhorn-Knopp), so every mixed stream is a convex
combination of input streams and can never be amplified. At a single loop-back point the
read-out is one row of such a matrix, i.e. a point on the simplex. The bridge therefore forms

    x_r = sum_k w_k(r) * S_k  +  e_step(r)  [+ zero-init low-rank adapter]
    S   = [ h_{r-1},  anchor,  h_{r-1} * rms(anchor)/rms(h_{r-1}) ]   w(r) in simplex

with `anchor` = prelude output (input injection, as in recurrent-depth models) and the third
stream a norm-matched copy that directly cancels the "residual norm shock".

Exactness at step 0 is NOT provided here; it comes from the zero-initialised per-pass output
gate in `loop.py` (x + 0 * delta == x bit-exactly), so the bridge may start anywhere.

`sinkhorn_project` + `BirkhoffMix` are the full n-stream building blocks for the optional
per-layer mHC residual (see docs/LOOP_ARCHITECTURE.md, "phase 2").
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from ._dtype import upcast


def sinkhorn_project(logits: torch.Tensor, iters: int = 20, eps: float = 1e-8) -> torch.Tensor:
    """Differentiable projection of exp(logits) onto (approximately) doubly-stochastic matrices."""
    m = torch.exp(upcast(logits) - upcast(logits).amax(dim=(-1, -2), keepdim=True))
    for _ in range(iters):
        m = m / (m.sum(dim=-1, keepdim=True) + eps)
        m = m / (m.sum(dim=-2, keepdim=True) + eps)
    return m


class BirkhoffMix(nn.Module):
    """H = (1-lam) I + lam * Sinkhorn(W). Convex in the Birkhoff polytope; lam=0 => exact identity.

    lam is a *projected* parameter (clamped into [0, 1] after each optimizer step by the
    trainer, see `project_constrained_`), so its gradient is non-zero at lam=0.
    """

    def __init__(self, n: int, iters: int = 20):
        super().__init__()
        self.n, self.iters = n, iters
        self.logits = nn.Parameter(torch.zeros(n, n))
        self.lam = nn.Parameter(torch.zeros(()))
        self.lam._qloop_project = (0.0, 1.0)  # type: ignore[attr-defined]

    def matrix(self) -> torch.Tensor:
        eye = torch.eye(self.n, device=self.logits.device)
        return (1 - self.lam) * eye + self.lam * sinkhorn_project(self.logits, self.iters)

    def forward(self, streams: torch.Tensor) -> torch.Tensor:  # streams [n, ...]
        h = self.matrix().to(streams.dtype)
        return torch.einsum("ij,j...->i...", h, streams)


def _rms(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return upcast(x).pow(2).mean(-1, keepdim=True).add(eps).sqrt()


V3_BRIDGE_WEIGHTS: tuple[float, float, float] = (0.05, 0.85, 0.10)


class LoopBridge(nn.Module):
    """Convex (simplex / one-row-mHC) mix of [h, anchor, norm-matched h] + step embedding + adapter.

    Supports two low-rank adapter modes when `rank > 0`:
      * `"concat"` (legacy): shared `2*hidden -> rank -> hidden` over `[h_prev, anchor]` with zero `up`.
      * `"subspace"` (v3 `ReEntryBridge`, Fixes A1/A2/A3): per-pass `nn.RMSNorm(hidden)` on `h_prev` ->
        orthogonal-init `down(hidden -> rank)` -> `SiLU` -> normal-init `up(rank -> hidden)` (`std=up_init_std`),
        scaled by `_rms(anchor) / sqrt(hidden)`. Step-0 bit-identity is guaranteed by the zero-initialised
        pass-exit gate in `loop.py`, while non-zero `up` ensures immediate step-1 gradients into `down`/`norms`.
    """

    STREAMS = ("h_prev", "anchor", "h_prev_rms_matched")

    def __init__(
        self,
        hidden: int,
        n_extra_passes: int,
        init_weights: tuple[float, ...] = (0.6, 0.2, 0.2),
        rank: int = 0,
        step_embedding: bool = True,
        adapter_mode: str = "concat",
        up_init_std: float = 0.01,
    ):
        super().__init__()
        if len(init_weights) != 3 or min(init_weights) <= 0:
            raise ValueError(f"bridge init_weights must be 3 positive numbers, got {init_weights}")
        if adapter_mode not in ("concat", "subspace"):
            raise ValueError(f"bridge adapter_mode must be 'concat' or 'subspace', got {adapter_mode!r}")
        if rank < 0:
            raise ValueError(f"bridge rank must be >= 0, got {rank}")
        if up_init_std < 0:
            raise ValueError(f"bridge up_init_std must be >= 0, got {up_init_std}")
        if n_extra_passes < 1:
            raise ValueError(f"n_extra_passes must be >= 1, got {n_extra_passes}")
        self.hidden = int(hidden)
        self.n_extra_passes = int(n_extra_passes)
        self.rank = int(rank)
        self.adapter_mode = adapter_mode
        self.up_init_std = float(up_init_std)
        s = sum(init_weights)
        logits = torch.tensor([math.log(w / s) for w in init_weights])
        self.mix_logits = nn.Parameter(logits.repeat(self.n_extra_passes, 1))
        self.step_emb = nn.Parameter(torch.zeros(self.n_extra_passes, self.hidden)) if step_embedding else None
        if self.rank > 0:
            if adapter_mode == "concat":
                self.down = nn.Linear(2 * self.hidden, self.rank, bias=False)
                self.up = nn.Linear(self.rank, self.hidden, bias=False)
                nn.init.zeros_(self.up.weight)
            else:
                self.norms = nn.ModuleList([nn.RMSNorm(self.hidden, eps=1e-6) for _ in range(self.n_extra_passes)])
                self.downs = nn.ModuleList(
                    [nn.Linear(self.hidden, self.rank, bias=False) for _ in range(self.n_extra_passes)]
                )
                self.ups = nn.ModuleList(
                    [nn.Linear(self.rank, self.hidden, bias=False) for _ in range(self.n_extra_passes)]
                )
                for i in range(self.n_extra_passes):
                    nn.init.orthogonal_(self.downs[i].weight)
                    if self.up_init_std == 0.0:
                        nn.init.zeros_(self.ups[i].weight)
                    else:
                        nn.init.normal_(self.ups[i].weight, mean=0.0, std=self.up_init_std)

    def weights(self, extra_pass: int) -> torch.Tensor:
        return F.softmax(upcast(self.mix_logits[extra_pass]), dim=-1)

    def forward(self, h: torch.Tensor, anchor: torch.Tensor, extra_pass: int) -> torch.Tensor:
        w = self.weights(extra_pass)
        rms_anc = _rms(anchor)
        matched = upcast(h) * (rms_anc / _rms(h))
        x = w[0] * upcast(h) + w[1] * upcast(anchor) + w[2] * matched
        if self.step_emb is not None:
            x = x + upcast(self.step_emb[extra_pass])
        if self.rank > 0:
            if self.adapter_mode == "concat":
                cat_in = torch.cat([upcast(h), upcast(anchor)], dim=-1).to(self.down.weight.dtype)
                delta = self.up(F.silu(self.down(cat_in)).to(self.up.weight.dtype))
                x = x + upcast(delta)
            else:
                norm_mod = self.norms[extra_pass]
                down_mod = self.downs[extra_pass]
                up_mod = self.ups[extra_pass]
                u_hat = norm_mod(upcast(h).to(norm_mod.weight.dtype))
                delta = up_mod(F.silu(down_mod(u_hat.to(down_mod.weight.dtype))).to(up_mod.weight.dtype))
                x = x + (rms_anc / math.sqrt(self.hidden)) * upcast(delta)
        return x.to(h.dtype)


class ReEntryBridge(LoopBridge):
    """v3 Subspace-Aligned `ReEntryBridge` preset (`init_weights=(0.05, 0.85, 0.10)`, `adapter_mode='subspace'`)."""

    def __init__(
        self,
        hidden: int,
        n_extra_passes: int,
        init_weights: tuple[float, ...] = V3_BRIDGE_WEIGHTS,
        rank: int = 256,
        step_embedding: bool = True,
        up_init_std: float = 0.01,
    ):
        super().__init__(
            hidden=hidden,
            n_extra_passes=n_extra_passes,
            init_weights=init_weights,
            rank=rank,
            step_embedding=step_embedding,
            adapter_mode="subspace",
            up_init_std=up_init_std,
        )
