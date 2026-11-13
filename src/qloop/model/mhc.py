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


class LoopBridge(nn.Module):
    """Convex (simplex / one-row-mHC) mix of [h, anchor, norm-matched h] + step embedding."""

    STREAMS = ("h_prev", "anchor", "h_prev_rms_matched")

    def __init__(self, hidden: int, n_extra_passes: int, init_weights=(0.6, 0.2, 0.2),
                 rank: int = 0, step_embedding: bool = True):
        super().__init__()
        if len(init_weights) != 3 or min(init_weights) <= 0:
            raise ValueError(f"bridge init_weights must be 3 positive numbers, got {init_weights}")
        s = sum(init_weights)
        logits = torch.tensor([math.log(w / s) for w in init_weights])
        self.mix_logits = nn.Parameter(logits.repeat(n_extra_passes, 1))
        self.step_emb = nn.Parameter(torch.zeros(n_extra_passes, hidden)) if step_embedding else None
        self.rank = rank
        if rank > 0:
            self.down = nn.Linear(2 * hidden, rank, bias=False)
            self.up = nn.Linear(rank, hidden, bias=False)
            nn.init.zeros_(self.up.weight)

    def weights(self, extra_pass: int) -> torch.Tensor:
        return F.softmax(upcast(self.mix_logits[extra_pass]), dim=-1)

    def forward(self, h: torch.Tensor, anchor: torch.Tensor, extra_pass: int) -> torch.Tensor:
        w = self.weights(extra_pass)
        matched = upcast(h) * (_rms(anchor) / _rms(h))
        x = w[0] * upcast(h) + w[1] * upcast(anchor) + w[2] * matched
        if self.step_emb is not None:
            x = x + upcast(self.step_emb[extra_pass])
        if self.rank > 0:
            x = x + self.up(F.silu(self.down(torch.cat([h, anchor], dim=-1))).to(self.up.weight.dtype)).to(x.dtype)
        return x.to(h.dtype)
