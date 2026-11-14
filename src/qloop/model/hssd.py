"""Horizon-Split Surprise-Gated Delta Memory (HSSD v3.1) retrofit for linear-attention layers.

Implements Track B (`HSSD v3.1`) of `Qwen3.6-35B-A6B-LoopHSSD` (`docs/ARCHITECTURE_SPEC.md` §5,
`docs/RESEARCH_ESSAY.md` §4 Theorems 1 & 2), incorporating all external audit hardenings:

  * **Fix B1 (Strictly Disjoint Horizon Partition):** For 1-based token `t` in 0-based chunk
    `n = (t - 1) // C`, sets `k_far = max(0, n - W)` (`W = window_chunks`, default 8) and
    `far_cutoff = k_far * C`. Far-horizon memory reads exclusively from `S_boundary[k_far]`
    covering `I_far = [1, far_cutoff]`, while exact causal local SWA covers
    `I_loc = [far_cutoff + 1, t]` (`|I_loc| in [W*C + 1, (W + 1)*C]` for `t > W*C`, `[1, t]`
    for `t <= W*C`), guaranteeing zero double-counting and zero gap.
  * **Fix B5 (`PeakBoltzmannGate`):** Hybrid log-partition + Gumbel extreme-value branch gate
    `stat = 0.5 * (LSE(z) - log(N_t)) + 0.5 * (max(z) - sqrt(2 * log(max(N_t, 2))))`,
    `p_hit = sigmoid(kappa_h * (stat - tau_h))`, blending `o = p_hit * o_loc + (1 - p_hit) * o_far`.
  * **Theorem 1 (Low-Rank 2D Phase Rotation & `fp64` Phase Accumulator):** Pure linear low-rank
    phase step `theta_t = pi * tanh(W_up(W_down(x_t)))` (`theta_up` zero-init), accumulated in
    `torch.float64` (`Theta_t = cumsum(theta_t) mod 2*pi`, `< 2e-6 rad` drift at `1M` steps),
    counter-rotating `(q, k)` via `R(-Theta_t)` with over-relaxed erase `b_t in (0, 2)^{d_k}`
    and linear-`beta` write scaling `sqrt(w_t)`.
  * **Fixes H2, B3, B4 & 4.2 (Multi-Page Routed Chunk-Start Root-Surprise `s = g^2`):** Evaluates
    relative key-value prediction error against the top-`k` write-routed pages of `boundary_ring[-1]`
    (`S_{nC}`) in parallel across all tokens of chunk `n` before the WY triangular solve, with
    smooth gradients at `g -> 0` and bit-exact `S_t == S_{t-1}` at `g = 0`.
  * **Fixes B2, B6, 3A-3E, 4.1 (Factored Decay, Actual Sparse STE Write Budget, Compact Cache):**
    Factored `(gamma_page_scale[M], gamma_chan_pair[B, T, H_v, d_k // 2])` decay (`16x` smaller),
    actual sparse Top-`k` write-mass duty with Straight-Through Estimation (`pi_ste`), compact
    `HSSDCacheState` (`__torch_function__`, `HSSDHybridCache`, `state_dtype="bfloat16"` option),
    packed-document isolation (`cu_seqlens` / `seq_idx`), and 2D padding logical-clock invariance.

Clean-room Apache-2.0 / MIT PyTorch implementation.
"""
from __future__ import annotations

import functools
import importlib.util
import logging
import math
import sys
from collections.abc import Sequence
from dataclasses import dataclass, fields
from typing import Any, Callable

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from ..debug import require
from ._dtype import upcast
from .gdn2 import _infer_gdn_dims, l2norm, logistic_shift
from .introspect import LINEAR, attention_module, find_text_model, inspect_model, layer_type_of

try:
    from transformers.cache_utils import Cache as _HFCacheBase  # type: ignore[import-not-found]
except ImportError:
    _HFCacheBase = object  # type: ignore[assignment,misc]

log = logging.getLogger("qloop")

DEFAULT_MU_MAX: tuple[float, ...] = (0.0025, 0.0025, 0.025, 0.025, 0.25, 0.25, 0.25, 0.25)

_VALID_STATE_DTYPES: dict[str, torch.dtype | None] = {
    "auto": None,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "float16": torch.float16,
    "fp16": torch.float16,
    "float32": torch.float32,
    "fp32": torch.float32,
    "float64": torch.float64,
    "fp64": torch.float64,
}
_VALID_SURPRISE_MODES = ("routed", "page0")


def has_fla() -> bool:
    """Return True if optional MIT `flash-linear-attention` (`fla`) package is importable."""
    return importlib.util.find_spec("fla") is not None


def _in_backward() -> bool:
    fn = getattr(torch._C, "_current_graph_task_id", None)
    return fn is not None and fn() != -1


def _is_compiling() -> bool:
    compiler = getattr(torch, "compiler", None)
    fn = getattr(compiler, "is_compiling", None) if compiler is not None else None
    return bool(fn()) if callable(fn) else False


def _compact(t: torch.Tensor) -> torch.Tensor:
    """Ensure `t` is contiguous and owns compact backing storage (`nbytes == numel * element_size`, offset 0)."""
    if not t.is_contiguous():
        return t.contiguous()
    if t.numel() == 0:
        if int(t.untyped_storage().nbytes()) != 0 or int(t.storage_offset()) != 0:
            return t.clone()
        return t
    if int(t.untyped_storage().nbytes()) != int(t.numel() * t.element_size()) or int(t.storage_offset()) != 0:
        return t.clone().contiguous()
    return t


def _is_hssd_state_duck(obj: Any) -> bool:
    return (
        not isinstance(obj, torch.Tensor)
        and hasattr(obj, "page_states")
        and hasattr(obj, "boundary_ring")
        and callable(getattr(obj, "_iter_tensors", None))
    )


def _ensure_dynamo_mark_static_address_compat() -> None:
    """Make `torch._dynamo.mark_static_address` accept `HSSDCacheState` / `_HSSDCacheState` (`transformers>=5.20`)."""
    dynamo = getattr(torch, "_dynamo", None)
    orig_fn = getattr(dynamo, "mark_static_address", None) if dynamo is not None else None
    if not callable(orig_fn) or getattr(orig_fn, "_qloop_hssd_aware", False):
        return

    @functools.wraps(orig_fn)
    def _hssd_mark_static_address(t: Any, *args: Any, **kwargs: Any) -> Any:
        if _is_hssd_state_duck(t):
            seen: set[int] = set()
            for sub_t in t._iter_tensors():
                if isinstance(sub_t, torch.Tensor) and id(sub_t) not in seen:
                    seen.add(id(sub_t))
                    orig_fn(sub_t, *args, **kwargs)
            return None
        return orig_fn(t, *args, **kwargs)

    forbid = getattr(dynamo, "forbid_in_graph", None)
    wrapped = forbid(_hssd_mark_static_address) if callable(forbid) else _hssd_mark_static_address
    wrapped._qloop_hssd_aware = True  # type: ignore[attr-defined]
    wrapped._qloop_orig_mark_static_address = orig_fn  # type: ignore[attr-defined]
    dynamo.mark_static_address = wrapped  # type: ignore[attr-defined]


_ensure_dynamo_mark_static_address_compat()


def _resolve_state_dtype(state_dtype: str | torch.dtype | None, acc_dt: torch.dtype) -> torch.dtype:
    if state_dtype is None:
        return acc_dt
    if isinstance(state_dtype, torch.dtype):
        return state_dtype
    key = str(state_dtype).replace("torch.", "").lower()
    if key not in _VALID_STATE_DTYPES:
        raise ValueError(f"unknown hssd.state_dtype {state_dtype!r}; valid: {sorted(_VALID_STATE_DTYPES)}")
    resolved = _VALID_STATE_DTYPES[key]
    return acc_dt if resolved is None else resolved


# ============================================================================ config

@dataclass
class HSSDConfig:
    rank: int = 64
    layers: list[int] | None = None
    chunk_size: int = 64
    window_chunks: int = 8
    num_pages: int = 8
    k_pages: int = 2
    top_k_pages: int | None = None
    segment_chunks: int = 16
    checkpoint_segments: bool = True
    gate_logit_cap: float = 8.0
    decay_log_cap: float = 2.0
    kappa_init: float = 2.0
    tau_init: float = 2.2
    surprise_slope_init: float = 3.0
    surprise_thresh_init: float = 0.85
    budget_weight: float = 1.0
    mu_max: tuple[float, ...] = DEFAULT_MU_MAX
    mu_min_ratio: float = 0.2
    decay_gamma_min: float = 1e-4
    decay_gamma_max: float = 5e-2
    theta_low_rank: bool = True
    state_dtype: str = "auto"
    paired_b_gate: bool = False
    use_swa: bool = True
    use_phase: bool = True
    use_surprise: bool = True
    surprise_mode: str = "routed"

    def __post_init__(self) -> None:
        if self.top_k_pages is not None:
            self.k_pages = int(self.top_k_pages)
        else:
            self.top_k_pages = int(self.k_pages)

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> HSSDConfig:
        clean = {k: v for k, v in dict(d or {}).items() if k != "enabled"}
        if "mu_max" in clean and isinstance(clean["mu_max"], list):
            clean["mu_max"] = tuple(float(x) for x in clean["mu_max"])
        known = {f.name for f in fields(cls)}
        bad = set(clean) - known
        if bad:
            raise ValueError(f"unknown hssd.* keys {sorted(bad)}; valid: {sorted(known)}")
        return cls(**clean)

    def resolved_mu_max(self) -> tuple[float, ...]:
        if len(self.mu_max) == self.num_pages:
            return tuple(float(x) for x in self.mu_max)
        if self.mu_max != DEFAULT_MU_MAX:
            raise ValueError(f"hssd.mu_max length {len(self.mu_max)} != num_pages {self.num_pages}")
        if self.num_pages == 1:
            return (0.25,)
        lo, hi = math.log(0.0025), math.log(0.25)
        return tuple(math.exp(lo + (hi - lo) * i / (self.num_pages - 1)) for i in range(self.num_pages))

    def validate(self) -> None:
        if self.rank < 1:
            raise ValueError(f"hssd.rank must be >= 1, got {self.rank}")
        if self.chunk_size < 1:
            raise ValueError(f"hssd.chunk_size must be >= 1, got {self.chunk_size}")
        if self.window_chunks < 0:
            raise ValueError(f"hssd.window_chunks must be >= 0, got {self.window_chunks}")
        if self.num_pages < 1:
            raise ValueError(f"hssd.num_pages must be >= 1, got {self.num_pages}")
        if not 1 <= self.k_pages <= self.num_pages:
            raise ValueError(f"hssd.k_pages must be in [1, {self.num_pages}], got {self.k_pages}")
        if self.segment_chunks < 1:
            raise ValueError(f"hssd.segment_chunks must be >= 1, got {self.segment_chunks}")
        if self.gate_logit_cap <= 0 or self.decay_log_cap <= 0:
            raise ValueError("hssd.gate_logit_cap and hssd.decay_log_cap must be > 0")
        if self.kappa_init <= 0 or self.surprise_slope_init <= 0:
            raise ValueError("hssd.kappa_init and hssd.surprise_slope_init must be > 0")
        if not 0.0 <= self.mu_min_ratio <= 1.0:
            raise ValueError(f"hssd.mu_min_ratio must be in [0, 1], got {self.mu_min_ratio}")
        if self.decay_gamma_min <= 0 or self.decay_gamma_max < self.decay_gamma_min:
            raise ValueError("hssd decay_gamma bounds must satisfy 0 < min <= max")
        sdt_key = str(self.state_dtype).replace("torch.", "").lower()
        if sdt_key not in _VALID_STATE_DTYPES:
            raise ValueError(f"hssd.state_dtype must be one of {sorted(_VALID_STATE_DTYPES)}, got {self.state_dtype!r}")
        if self.surprise_mode not in _VALID_SURPRISE_MODES:
            raise ValueError(
                f"hssd.surprise_mode must be one of {_VALID_SURPRISE_MODES}, got {self.surprise_mode!r}"
            )
        _ = self.resolved_mu_max()


# ============================================================================ math primitives

def _safe_sqrt(x: torch.Tensor) -> torch.Tensor:
    """Singularity-free `sqrt(max(x, 0))` with exact 0 value and finite 0 derivative at `x <= 0`."""
    pos = x > 0.0
    safe = torch.where(pos, x, torch.ones_like(x))
    return torch.where(pos, torch.sqrt(safe), torch.zeros_like(x))


def _safe_norm(x: torch.Tensor, dim: int = -1, keepdim: bool = False) -> torch.Tensor:
    """L2 norm with finite 0 derivative at exact zero vectors."""
    return _safe_sqrt((x * x).sum(dim=dim, keepdim=keepdim))


def _rms_norm_scaled(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    xf = upcast(x)
    normed = xf * torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + eps)
    return normed * upcast(weight)


def apply_2d_rotation(x: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    """Block-diagonal 2D phase rotation `R(theta)` on last dim `d = 2 * (d // 2)` (Theorem 1).

    Evaluates `cos`/`sin` at `theta`'s precision (`float64` phase accumulator) and casts only the
    bounded `[-1, 1]` trig factors to `x.dtype`.
    """
    require(x.shape[-1] % 2 == 0, "apply_2d_rotation requires an even last dimension", x=x)
    pairs = x.reshape(*x.shape[:-1], x.shape[-1] // 2, 2)
    cos_t = torch.cos(theta).to(dtype=x.dtype)
    sin_t = torch.sin(theta).to(dtype=x.dtype)
    x0 = pairs[..., 0]
    x1 = pairs[..., 1]
    out0 = x0 * cos_t - x1 * sin_t
    out1 = x0 * sin_t + x1 * cos_t
    return torch.stack([out0, out1], dim=-1).reshape_as(x)


def disjoint_horizon_bounds(t_1based: int, chunk_size: int = 64, window_chunks: int = 8) -> tuple[int, int, int]:
    """Return `(chunk_idx_n, k_far, far_cutoff)` for 1-based token index `t_1based` (Fix B1).

    `I_far = [1, far_cutoff]` and `I_loc = [far_cutoff + 1, t_1based]` partition `[1, t_1based]`
    with zero overlap and zero gap.
    """
    require(t_1based >= 1, "t_1based must be >= 1", t_1based=t_1based)
    n = (int(t_1based) - 1) // int(chunk_size)
    k_far = max(0, n - int(window_chunks))
    far_cutoff = k_far * int(chunk_size)
    return n, k_far, far_cutoff


class PeakBoltzmannGate(nn.Module):
    """Peak-augmented Boltzmann local-hit branch gate (`Fix B5`).

    Computes:
        stat = 0.5 * (LSE(scores) - log(N_t)) + 0.5 * (max(scores) - sqrt(2 * log(max(N_t, 2))))
        p_hit = sigmoid(kappa_h * (stat - tau_h))
    """

    def __init__(self, num_heads: int = 1, kappa_init: float = 2.0, tau_init: float = 2.2):
        super().__init__()
        self.num_heads = int(num_heads)
        self.log_kappa = nn.Parameter(torch.full((self.num_heads,), math.log(float(kappa_init))))
        self.tau = nn.Parameter(torch.full((self.num_heads,), float(tau_init)))

    def forward(self, scores: torch.Tensor, valid_mask: torch.Tensor | None = None) -> torch.Tensor:
        sf = upcast(scores)
        if valid_mask is None:
            n_val = float(sf.shape[-1])
            lse = torch.logsumexp(sf, dim=-1)
            max_sc = sf.max(dim=-1).values
            log_n = sf.new_tensor(math.log(max(n_val, 1.0)))
            gumbel = sf.new_tensor(math.sqrt(2.0 * math.log(max(n_val, 2.0))))
            has_valid: torch.Tensor | None = None
        else:
            vm = valid_mask.to(dtype=torch.bool).expand_as(sf)
            counts = vm.sum(dim=-1)
            has_valid = counts > 0
            safe_vm = torch.where(has_valid.unsqueeze(-1), vm, torch.ones_like(vm))
            masked = sf.masked_fill(~safe_vm, float("-inf"))
            lse = torch.logsumexp(masked, dim=-1)
            max_sc = masked.max(dim=-1).values
            n_eff = counts.clamp_min(1).to(sf.dtype)
            log_n = torch.log(n_eff)
            gumbel = torch.sqrt(2.0 * torch.log(n_eff.clamp_min(2.0)))

        stat = 0.5 * (lse - log_n) + 0.5 * (max_sc - gumbel)
        kappa = upcast(self.log_kappa).exp().to(stat.dtype)
        tau = upcast(self.tau).to(stat.dtype)
        if self.num_heads == 1 or stat.ndim == 0:
            kappa_b = kappa[0]
            tau_b = tau[0]
        elif stat.ndim >= 2 and stat.shape[1] == self.num_heads:
            shape = [1, self.num_heads] + [1] * (stat.ndim - 2)
            kappa_b = kappa.view(*shape)
            tau_b = tau.view(*shape)
        elif stat.shape[-1] == self.num_heads:
            kappa_b = kappa
            tau_b = tau
        else:
            raise ValueError(f"PeakBoltzmannGate(num_heads={self.num_heads}) incompatible with stat shape {stat.shape}")

        p_hit = torch.sigmoid(kappa_b * (stat - tau_b))
        if has_valid is not None:
            p_hit = torch.where(has_valid, p_hit, torch.zeros_like(p_hit))
        return p_hit.to(scores.dtype)


# ============================================================================ O(1) cache state & hybrid cache

class HSSDCacheState:
    """Fixed-footprint `O(1)` incremental decode cache for one `(layer, pass)` HSSD slot.

    Maintains compact, view-free tensors:
      * `step_count`: `[B]` (`torch.long`) per-sequence logical token count (unaffected by padding).
      * `page_states`: `[B, M, H_v, d_k, d_v]` live multi-page state after `step_count` tokens.
      * `theta_cum`: `[B, H_v, d_k // 2]` (`torch.float64` ALWAYS) cumulative 2D phase `Theta_t mod 2*pi`.
      * `boundary_ring`: length `W + 1` tuple of `[B, M, H_v, d_k, d_v]` boundary snapshots where
        `boundary_ring[0] == S_boundary[max(0, n - W)]` and `boundary_ring[-1] == S_{nC}` across all
        `M` pages (exposing `s_chunk_start == boundary_ring[-1][:, 0]` without duplicate storage).
      * `swa_k_ring`, `swa_v_ring`, `swa_valid_ring`: circular buffer of size `(W + 1) * C` (`9 * C`).

    Implements the Hugging Face Transformers tensor-cache lifecycle (`__torch_function__` for
    `zeros_like`, `empty_like`, `clone`, `index_select`, `cat`, plus `.copy_()`, `state[beam_idx]`,
    `.reorder_cache()`, `.shape`, `.size()`, `.ndim`, `.numel()`, `.device`, `.dtype`).
    """

    def __init__(
        self,
        step_count: int | torch.Tensor,
        chunk_size: int,
        window_chunks: int,
        page_states: torch.Tensor,
        theta_cum: torch.Tensor,
        boundary_ring: tuple[torch.Tensor, ...],
        swa_k_ring: torch.Tensor,
        swa_v_ring: torch.Tensor,
        swa_valid_ring: torch.Tensor,
        *,
        s_chunk_start: torch.Tensor | None = None,
    ) -> None:
        del s_chunk_start
        p_states = _compact(page_states)
        bsz = int(p_states.shape[0])
        dev = p_states.device
        if isinstance(step_count, torch.Tensor):
            if step_count.ndim == 0:
                sc_t = torch.full((bsz,), int(step_count.item()), device=dev, dtype=torch.long)
            else:
                require(
                    step_count.shape == (bsz,),
                    f"HSSDCacheState.step_count shape {tuple(step_count.shape)} != ({bsz},)",
                )
                sc_t = _compact(step_count.to(device=dev, dtype=torch.long))
        else:
            sc_t = torch.full((bsz,), int(step_count), device=dev, dtype=torch.long)

        th64 = _compact(theta_cum.to(device=dev, dtype=torch.float64))
        b_ring = tuple(
            p_states if (t is page_states) else _compact(t.to(device=dev, dtype=p_states.dtype))
            for t in boundary_ring
        )
        require(
            len(b_ring) == int(window_chunks) + 1,
            f"boundary_ring length {len(b_ring)} != window_chunks + 1 ({int(window_chunks) + 1})",
        )
        self.step_count: torch.Tensor = sc_t
        self.chunk_size: int = int(chunk_size)
        self.window_chunks: int = int(window_chunks)
        self.page_states: torch.Tensor = p_states
        self.theta_cum: torch.Tensor = th64
        self.boundary_ring: tuple[torch.Tensor, ...] = b_ring
        self.swa_k_ring: torch.Tensor = _compact(swa_k_ring.to(device=dev, dtype=p_states.dtype))
        self.swa_v_ring: torch.Tensor = _compact(swa_v_ring.to(device=dev, dtype=p_states.dtype))
        self.swa_valid_ring: torch.Tensor = _compact(swa_valid_ring.to(device=dev, dtype=torch.bool))

    def __repr__(self) -> str:
        return (
            f"HSSDCacheState(shape={tuple(self.shape)}, dtype={self.dtype}, "
            f"theta_dtype={self.theta_cum.dtype}, step_count={self.step_count.tolist()}, "
            f"chunk_size={self.chunk_size}, window_chunks={self.window_chunks})"
        )

    @property
    def s_chunk_start(self) -> torch.Tensor:
        """Primary Page-0 chunk-start state `S_{nC}^{(0)}` (`[B, H_v, d_k, d_v]`)."""
        return self.boundary_ring[-1][:, 0]

    @property
    def s_chunk_start_bank(self) -> torch.Tensor:
        """Full `M`-page chunk-start state bank `S_{nC}` (`[B, M, H_v, d_k, d_v]`)."""
        return self.boundary_ring[-1]

    @property
    def step_count_int(self) -> int:
        if self.step_count.numel() == 0:
            return 0
        if self.step_count.numel() > 1:
            require(
                bool((self.step_count == self.step_count[0]).all()),
                "step_count_int requires uniform step_count across batch",
                step_count=self.step_count,
            )
        return int(self.step_count[0].item())

    @property
    def max_swa_len(self) -> int:
        return (self.window_chunks + 1) * self.chunk_size

    @property
    def shape(self) -> torch.Size:
        return self.page_states.shape

    def size(self, dim: int | None = None) -> torch.Size | int:
        return self.page_states.size() if dim is None else self.page_states.size(dim)

    @property
    def ndim(self) -> int:
        return self.page_states.ndim

    def dim(self) -> int:
        return self.page_states.dim()

    def numel(self) -> int:
        return self.page_states.numel()

    def __len__(self) -> int:
        return int(self.page_states.shape[0])

    @property
    def dtype(self) -> torch.dtype:
        return self.page_states.dtype

    @property
    def device(self) -> torch.device:
        return self.page_states.device

    def _iter_tensors(self) -> tuple[torch.Tensor, ...]:
        return (
            self.step_count,
            self.page_states,
            self.theta_cum,
            *self.boundary_ring,
            self.swa_k_ring,
            self.swa_v_ring,
            self.swa_valid_ring,
        )

    def unique_storage_bytes(self) -> int:
        """Total unique backing-storage bytes across all tensors in this state (deduplicated by `data_ptr`)."""
        seen: set[int] = set()
        total = 0
        for t in self._iter_tensors():
            if t.numel() <= 0:
                continue
            st = t.untyped_storage()
            ptr = int(st.data_ptr())
            if ptr not in seen:
                seen.add(ptr)
                total += int(st.nbytes())
        return total

    def has_compact_storage(self) -> bool:
        """Return True iff every tensor is contiguous and owns exact-sized backing storage (zero view leaks)."""
        for t in self._iter_tensors():
            if not t.is_contiguous():
                return False
            if int(t.untyped_storage().nbytes()) != int(t.numel() * t.element_size()) or int(t.storage_offset()) != 0:
                return False
        return True

    def is_contiguous(self) -> bool:
        """Return True iff every tensor in `_iter_tensors()` is contiguous."""
        return all(t.is_contiguous() for t in self._iter_tensors())

    def contiguous(self) -> HSSDCacheState:
        """Return `self` if `has_compact_storage()` else a compacted copy."""
        if self.has_compact_storage():
            return self
        memo: dict[tuple[int, int, tuple[int, ...], tuple[int, ...]], torch.Tensor] = {}

        def _cp(t: torch.Tensor) -> torch.Tensor:
            if t.numel() > 0:
                key = (
                    int(t.untyped_storage().data_ptr()),
                    int(t.storage_offset()),
                    tuple(t.shape),
                    tuple(t.stride()),
                )
                if key in memo:
                    return memo[key]
                out_t = _compact(t)
                memo[key] = out_t
                return out_t
            return _compact(t)

        return HSSDCacheState(
            step_count=_compact(self.step_count),
            chunk_size=self.chunk_size,
            window_chunks=self.window_chunks,
            page_states=_cp(self.page_states),
            theta_cum=_compact(self.theta_cum),
            boundary_ring=tuple(_cp(t) for t in self.boundary_ring),
            swa_k_ring=_compact(self.swa_k_ring),
            swa_v_ring=_compact(self.swa_v_ring),
            swa_valid_ring=_compact(self.swa_valid_ring),
        )

    def compact(self) -> HSSDCacheState:
        return self.contiguous()

    def zero_(self) -> HSSDCacheState:
        """In-place zero all state buffers, unaliasing shared ring slots while preserving distinct static addresses."""
        seen_ptrs: set[int] = (
            {int(self.page_states.untyped_storage().data_ptr())} if self.page_states.numel() > 0 else set()
        )
        fixed_ring: list[torch.Tensor] = []
        unaliased = False
        for ring_t in self.boundary_ring:
            cur_t = ring_t
            if cur_t.numel() > 0:
                ptr = int(cur_t.untyped_storage().data_ptr())
                if ptr in seen_ptrs:
                    cur_t = torch.zeros_like(cur_t).contiguous()
                    ptr = int(cur_t.untyped_storage().data_ptr())
                    unaliased = True
                else:
                    cur_t.zero_()
                seen_ptrs.add(ptr)
            else:
                cur_t.zero_()
            fixed_ring.append(cur_t)
        if unaliased:
            self.boundary_ring = tuple(fixed_ring)

        self.step_count.zero_()
        self.page_states.zero_()
        self.theta_cum.zero_()
        self.swa_k_ring.zero_()
        self.swa_v_ring.zero_()
        self.swa_valid_ring.zero_()
        return self

    @classmethod
    def zeros(
        cls,
        batch_size: int,
        num_pages: int,
        num_v_heads: int,
        head_k_dim: int,
        head_v_dim: int,
        *,
        chunk_size: int = 64,
        window_chunks: int = 8,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
        distinct_ring_buffers: bool = False,
    ) -> HSSDCacheState:
        p_states = torch.zeros(batch_size, num_pages, num_v_heads, head_k_dim, head_v_dim, device=device, dtype=dtype)
        theta = torch.zeros(batch_size, num_v_heads, head_k_dim // 2, device=device, dtype=torch.float64)
        if distinct_ring_buffers:
            b_ring = tuple(torch.zeros_like(p_states) for _ in range(window_chunks + 1))
        else:
            b_ring = tuple(p_states for _ in range(window_chunks + 1))
        max_win = (window_chunks + 1) * chunk_size
        k_ring = torch.zeros(batch_size, max_win, num_v_heads, head_k_dim, device=device, dtype=dtype)
        v_ring = torch.zeros(batch_size, max_win, num_v_heads, head_v_dim, device=device, dtype=dtype)
        val_ring = torch.zeros(batch_size, max_win, device=device, dtype=torch.bool)
        steps = torch.zeros(batch_size, device=device, dtype=torch.long)
        return cls(
            step_count=steps,
            chunk_size=int(chunk_size),
            window_chunks=int(window_chunks),
            page_states=p_states,
            theta_cum=theta,
            boundary_ring=b_ring,
            swa_k_ring=k_ring,
            swa_v_ring=v_ring,
            swa_valid_ring=val_ring,
        )

    @classmethod
    def empty_like_state(
        cls,
        other: HSSDCacheState,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> HSSDCacheState:
        dev = other.device if device is None else torch.device(device)
        dt = other.dtype if dtype is None else dtype
        bsz, mp, hv, dk, dv = other.page_states.shape
        p_states = torch.empty(bsz, mp, hv, dk, dv, device=dev, dtype=dt)
        theta = torch.empty(bsz, hv, dk // 2, device=dev, dtype=torch.float64)
        b_ring = tuple(torch.empty_like(p_states) for _ in range(other.window_chunks + 1))
        max_win = other.max_swa_len
        k_ring = torch.empty(bsz, max_win, hv, dk, device=dev, dtype=dt)
        v_ring = torch.empty(bsz, max_win, hv, dv, device=dev, dtype=dt)
        val_ring = torch.empty(bsz, max_win, device=dev, dtype=torch.bool)
        steps = torch.empty(bsz, device=dev, dtype=torch.long)
        return cls(
            step_count=steps,
            chunk_size=other.chunk_size,
            window_chunks=other.window_chunks,
            page_states=p_states,
            theta_cum=theta,
            boundary_ring=b_ring,
            swa_k_ring=k_ring,
            swa_v_ring=v_ring,
            swa_valid_ring=val_ring,
        )

    @classmethod
    def from_tensor(
        cls,
        state_tensor: torch.Tensor,
        *,
        num_pages: int = 8,
        chunk_size: int = 64,
        window_chunks: int = 8,
    ) -> HSSDCacheState:
        st = upcast(state_tensor)
        if st.ndim == 4:
            bsz, hv, dk, dv = st.shape
            p_states = st.unsqueeze(1).expand(bsz, num_pages, hv, dk, dv).clone().contiguous()
        elif st.ndim == 5:
            bsz, mp, hv, dk, dv = st.shape
            require(mp == num_pages, f"expected {num_pages} state pages, got {mp}", state=st)
            p_states = st.clone().contiguous()
        else:
            raise ValueError(f"initial_state tensor must be 4D or 5D, got shape {tuple(st.shape)}")
        base = cls.zeros(
            bsz,
            num_pages,
            hv,
            dk,
            dv,
            chunk_size=chunk_size,
            window_chunks=window_chunks,
            device=p_states.device,
            dtype=p_states.dtype,
        )
        base.page_states = p_states
        base.boundary_ring = tuple(p_states for _ in range(window_chunks + 1))
        return base

    def clone(self) -> HSSDCacheState:
        memo: dict[int, torch.Tensor] = {}

        def _cl(t: torch.Tensor) -> torch.Tensor:
            ptr = int(t.untyped_storage().data_ptr()) if t.numel() > 0 else id(t)
            if ptr not in memo:
                memo[ptr] = t.clone().contiguous()
            return memo[ptr]

        return HSSDCacheState(
            step_count=self.step_count.clone().contiguous(),
            chunk_size=self.chunk_size,
            window_chunks=self.window_chunks,
            page_states=_cl(self.page_states),
            theta_cum=self.theta_cum.clone().contiguous(),
            boundary_ring=tuple(_cl(t) for t in self.boundary_ring),
            swa_k_ring=self.swa_k_ring.clone().contiguous(),
            swa_v_ring=self.swa_v_ring.clone().contiguous(),
            swa_valid_ring=self.swa_valid_ring.clone().contiguous(),
        )

    def detach(self) -> HSSDCacheState:
        memo: dict[int, torch.Tensor] = {}

        def _dt(t: torch.Tensor) -> torch.Tensor:
            ptr = int(t.untyped_storage().data_ptr()) if t.numel() > 0 else id(t)
            if ptr not in memo:
                memo[ptr] = _compact(t.detach())
            return memo[ptr]

        return HSSDCacheState(
            step_count=_compact(self.step_count.detach()),
            chunk_size=self.chunk_size,
            window_chunks=self.window_chunks,
            page_states=_dt(self.page_states),
            theta_cum=_compact(self.theta_cum.detach()),
            boundary_ring=tuple(_dt(t) for t in self.boundary_ring),
            swa_k_ring=_compact(self.swa_k_ring.detach()),
            swa_v_ring=_compact(self.swa_v_ring.detach()),
            swa_valid_ring=_compact(self.swa_valid_ring.detach()),
        )

    def to(self, *args: Any, **kwargs: Any) -> HSSDCacheState:
        memo: dict[int, torch.Tensor] = {}
        non_blocking = bool(kwargs.get("non_blocking", False))
        p_states = _compact(self.page_states.to(*args, **kwargs))
        if self.page_states.numel() > 0:
            memo[int(self.page_states.untyped_storage().data_ptr())] = p_states
        dev = p_states.device
        dt = p_states.dtype

        def _conv(t: torch.Tensor) -> torch.Tensor:
            if t.numel() > 0:
                ptr = int(t.untyped_storage().data_ptr())
                if ptr in memo:
                    return memo[ptr]
                out_t = _compact(t.to(device=dev, dtype=dt, non_blocking=non_blocking))
                memo[ptr] = out_t
                return out_t
            return _compact(t.to(device=dev, dtype=dt, non_blocking=non_blocking))

        return HSSDCacheState(
            step_count=_compact(self.step_count.to(device=dev, dtype=torch.long, non_blocking=non_blocking)),
            chunk_size=self.chunk_size,
            window_chunks=self.window_chunks,
            page_states=p_states,
            theta_cum=_compact(self.theta_cum.to(device=dev, dtype=torch.float64, non_blocking=non_blocking)),
            boundary_ring=tuple(_conv(t) for t in self.boundary_ring),
            swa_k_ring=_compact(self.swa_k_ring.to(device=dev, dtype=dt, non_blocking=non_blocking)),
            swa_v_ring=_compact(self.swa_v_ring.to(device=dev, dtype=dt, non_blocking=non_blocking)),
            swa_valid_ring=_compact(self.swa_valid_ring.to(device=dev, dtype=torch.bool, non_blocking=non_blocking)),
        )

    def copy_(self, other: Any) -> HSSDCacheState:
        require(
            isinstance(other, HSSDCacheState) or _is_hssd_state_duck(other),
            f"copy_ expects HSSDCacheState, got {type(other).__name__}",
        )
        require(
            self.chunk_size == int(other.chunk_size) and self.window_chunks == int(other.window_chunks),
            "HSSDCacheState.copy_ chunk_size/window_chunks mismatch",
            dst=(self.chunk_size, self.window_chunks),
            src=(other.chunk_size, other.window_chunks),
        )
        # Unalias destination boundary_ring slots if `self` was created via `HSSDCacheState.zeros()`
        seen_ptrs: set[int] = (
            {int(self.page_states.untyped_storage().data_ptr())} if self.page_states.numel() > 0 else set()
        )
        fixed_ring: list[torch.Tensor] = []
        unaliased = False
        for ring_t in self.boundary_ring:
            cur_t = ring_t
            if cur_t.numel() > 0:
                ptr = int(cur_t.untyped_storage().data_ptr())
                if ptr in seen_ptrs:
                    cur_t = cur_t.clone().contiguous()
                    ptr = int(cur_t.untyped_storage().data_ptr())
                    unaliased = True
                seen_ptrs.add(ptr)
            fixed_ring.append(cur_t)
        if unaliased:
            self.boundary_ring = tuple(fixed_ring)

        self.step_count.copy_(other.step_count)
        self.page_states.copy_(other.page_states)
        self.theta_cum.copy_(other.theta_cum)
        for dst, src in zip(self.boundary_ring, other.boundary_ring):
            dst.copy_(src)
        self.swa_k_ring.copy_(other.swa_k_ring)
        self.swa_v_ring.copy_(other.swa_v_ring)
        self.swa_valid_ring.copy_(other.swa_valid_ring)
        return self

    def index_select(self, dim: int, index: torch.Tensor) -> HSSDCacheState:
        d = dim if dim >= 0 else dim + self.ndim
        if d != 0:
            raise ValueError(f"HSSDCacheState.index_select only supports batch dim=0, got dim={dim}")
        idx = index.to(device=self.device, dtype=torch.long)
        memo: dict[int, torch.Tensor] = {}

        def _sel(t: torch.Tensor) -> torch.Tensor:
            if t.numel() > 0:
                ptr = int(t.untyped_storage().data_ptr())
                if ptr in memo:
                    return memo[ptr]
                out_t = _compact(t.index_select(0, idx))
                memo[ptr] = out_t
                return out_t
            return _compact(t.index_select(0, idx))

        return HSSDCacheState(
            step_count=_compact(self.step_count.index_select(0, idx)),
            chunk_size=self.chunk_size,
            window_chunks=self.window_chunks,
            page_states=_sel(self.page_states),
            theta_cum=_compact(self.theta_cum.index_select(0, idx)),
            boundary_ring=tuple(_sel(t) for t in self.boundary_ring),
            swa_k_ring=_compact(self.swa_k_ring.index_select(0, idx)),
            swa_v_ring=_compact(self.swa_v_ring.index_select(0, idx)),
            swa_valid_ring=_compact(self.swa_valid_ring.index_select(0, idx)),
        )

    def __getitem__(self, idx: Any) -> HSSDCacheState:
        if isinstance(idx, tuple):
            rest = idx[1:]
            if any(x is not Ellipsis and x != slice(None) for x in rest):
                raise IndexError("HSSDCacheState indexing only supports batch dimension 0")
            idx = idx[0]
        if isinstance(idx, int):
            bsz = len(self)
            norm_i = idx if idx >= 0 else bsz + idx
            if not 0 <= norm_i < bsz:
                raise IndexError(f"batch index {idx} out of range for batch size {bsz}")
            return self.index_select(0, torch.tensor([norm_i], device=self.device, dtype=torch.long))
        if isinstance(idx, slice):
            indices = torch.arange(len(self), device=self.device, dtype=torch.long)[idx]
            return self.index_select(0, indices)
        if isinstance(idx, (list, tuple)):
            return self.index_select(0, torch.tensor(idx, device=self.device, dtype=torch.long))
        if isinstance(idx, torch.Tensor):
            if idx.dtype == torch.bool:
                require(idx.ndim == 1 and idx.shape[0] == len(self), "boolean batch mask must have shape [B]")
                indices = idx.to(device=self.device).nonzero(as_tuple=False).squeeze(-1)
                return self.index_select(0, indices)
            return self.index_select(0, idx)
        raise TypeError(f"Unsupported HSSDCacheState index type: {type(idx).__name__}")

    def reorder_cache(self, beam_idx: torch.LongTensor) -> HSSDCacheState:
        return self.index_select(0, beam_idx)

    def repeat_interleave(
        self,
        repeats: int | torch.Tensor,
        dim: int | None = 0,
        *,
        output_size: int | None = None,
    ) -> HSSDCacheState:
        if dim is None:
            raise ValueError("HSSDCacheState.repeat_interleave requires explicit batch dim=0, got dim=None")
        d = dim if dim >= 0 else dim + self.ndim
        if d != 0:
            raise ValueError(f"HSSDCacheState.repeat_interleave only supports batch dim=0, got dim={dim}")
        rep = repeats.to(device=self.device) if isinstance(repeats, torch.Tensor) else repeats
        memo: dict[int, torch.Tensor] = {}

        def _ri(t: torch.Tensor) -> torch.Tensor:
            if t.numel() > 0:
                ptr = int(t.untyped_storage().data_ptr())
                if ptr in memo:
                    return memo[ptr]
                out_t = _compact(t.repeat_interleave(rep, dim=0, output_size=output_size))
                memo[ptr] = out_t
                return out_t
            return _compact(t.repeat_interleave(rep, dim=0, output_size=output_size))

        return HSSDCacheState(
            step_count=_compact(self.step_count.repeat_interleave(rep, dim=0, output_size=output_size)),
            chunk_size=self.chunk_size,
            window_chunks=self.window_chunks,
            page_states=_ri(self.page_states),
            theta_cum=_compact(self.theta_cum.repeat_interleave(rep, dim=0, output_size=output_size)),
            boundary_ring=tuple(_ri(t) for t in self.boundary_ring),
            swa_k_ring=_compact(self.swa_k_ring.repeat_interleave(rep, dim=0, output_size=output_size)),
            swa_v_ring=_compact(self.swa_v_ring.repeat_interleave(rep, dim=0, output_size=output_size)),
            swa_valid_ring=_compact(self.swa_valid_ring.repeat_interleave(rep, dim=0, output_size=output_size)),
        )

    @classmethod
    def __torch_function__(
        cls,
        func: Callable[..., Any],
        types: tuple[type, ...],
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> Any:
        del types
        kw = kwargs or {}
        if func is torch.zeros_like:
            st: HSSDCacheState = args[0]
            dev = st.device if kw.get("device") is None else torch.device(kw["device"])
            dt = st.dtype if kw.get("dtype") is None else kw["dtype"]
            bsz, mp, hv, dk, dv = st.page_states.shape
            return cls.zeros(
                bsz,
                mp,
                hv,
                dk,
                dv,
                chunk_size=st.chunk_size,
                window_chunks=st.window_chunks,
                device=dev,
                dtype=dt,
                distinct_ring_buffers=True,
            )
        if func is torch.empty_like:
            st = args[0]
            return cls.empty_like_state(st, device=kw.get("device"), dtype=kw.get("dtype"))
        if func is torch.clone:
            st = args[0]
            return st.clone()
        if func is torch.detach:
            st = args[0]
            return st.detach()
        if func is torch.index_select:
            st = args[0]
            dim = kw["dim"] if "dim" in kw else args[1]
            index = kw["index"] if "index" in kw else args[2]
            return st.index_select(int(dim), index)
        if func is torch.repeat_interleave:
            st = args[0]
            repeats = kw["repeats"] if "repeats" in kw else args[1]
            dim = kw.get("dim", args[2] if len(args) >= 3 else None)
            return st.repeat_interleave(repeats, dim=dim, output_size=kw.get("output_size"))
        if func is torch.cat:
            states: Sequence[HSSDCacheState] = args[0]
            dim = int(args[1] if len(args) >= 2 else kw.get("dim", 0))
            if dim not in (0, -5):
                raise ValueError(f"torch.cat on HSSDCacheState only supports batch dim=0, got dim={dim}")
            if not states or not all(isinstance(s, cls) or _is_hssd_state_duck(s) for s in states):
                return NotImplemented
            ref = states[0]
            for s in states[1:]:
                require(
                    int(s.chunk_size) == ref.chunk_size and int(s.window_chunks) == ref.window_chunks,
                    "torch.cat HSSDCacheState chunk_size/window_chunks mismatch",
                )
            return cls(
                step_count=torch.cat([s.step_count for s in states], dim=0),
                chunk_size=ref.chunk_size,
                window_chunks=ref.window_chunks,
                page_states=torch.cat([s.page_states for s in states], dim=0),
                theta_cum=torch.cat([s.theta_cum for s in states], dim=0),
                boundary_ring=tuple(
                    torch.cat([s.boundary_ring[j] for s in states], dim=0)
                    for j in range(ref.window_chunks + 1)
                ),
                swa_k_ring=torch.cat([s.swa_k_ring for s in states], dim=0),
                swa_v_ring=torch.cat([s.swa_v_ring for s in states], dim=0),
                swa_valid_ring=torch.cat([s.swa_valid_ring for s in states], dim=0),
            )
        return NotImplemented

    def state_equal(self, other: Any, *, atol: float = 1e-5, rtol: float = 1e-4) -> bool:
        if not (isinstance(other, HSSDCacheState) or _is_hssd_state_duck(other)):
            return False
        if self.chunk_size != int(other.chunk_size) or self.window_chunks != int(other.window_chunks):
            return False
        osc = (
            other.step_count.to(device=self.device, dtype=torch.long).reshape(-1)
            if isinstance(other.step_count, torch.Tensor)
            else torch.full(self.step_count.shape, int(other.step_count), device=self.device, dtype=torch.long)
        )
        if self.step_count.shape != osc.shape or not bool(torch.equal(self.step_count, osc)):
            return False
        cmp_dt = (
            torch.float64
            if torch.float64 in (self.page_states.dtype, other.page_states.dtype)
            else torch.float32
        )
        if self.page_states.shape != other.page_states.shape or not bool(
            torch.allclose(
                self.page_states.to(cmp_dt),
                other.page_states.to(device=self.device, dtype=cmp_dt),
                atol=atol,
                rtol=rtol,
            )
        ):
            return False
        oth_th = other.theta_cum.to(device=self.device, dtype=torch.float64)
        if self.theta_cum.shape != oth_th.shape or not bool(
            torch.allclose(self.theta_cum, oth_th, atol=atol, rtol=rtol)
        ):
            return False
        if len(self.boundary_ring) != len(other.boundary_ring):
            return False
        for a, b in zip(self.boundary_ring, other.boundary_ring):
            if a.shape != b.shape or not bool(
                torch.allclose(a.to(cmp_dt), b.to(device=self.device, dtype=cmp_dt), atol=atol, rtol=rtol)
            ):
                return False
        oth_val = other.swa_valid_ring.to(device=self.device, dtype=torch.bool)
        if self.swa_valid_ring.shape != oth_val.shape or not bool(torch.equal(self.swa_valid_ring, oth_val)):
            return False
        if self.swa_k_ring.shape != other.swa_k_ring.shape or self.swa_v_ring.shape != other.swa_v_ring.shape:
            return False
        if bool(self.swa_valid_ring.any()):
            vm = self.swa_valid_ring.unsqueeze(-1).unsqueeze(-1)
            k_a = torch.where(vm, self.swa_k_ring.to(cmp_dt), torch.zeros_like(self.swa_k_ring, dtype=cmp_dt))
            k_b = torch.where(
                vm,
                other.swa_k_ring.to(device=self.device, dtype=cmp_dt),
                torch.zeros_like(self.swa_k_ring, dtype=cmp_dt),
            )
            if not bool(torch.allclose(k_a, k_b, atol=atol, rtol=rtol)):
                return False
            v_a = torch.where(vm, self.swa_v_ring.to(cmp_dt), torch.zeros_like(self.swa_v_ring, dtype=cmp_dt))
            v_b = torch.where(
                vm,
                other.swa_v_ring.to(device=self.device, dtype=cmp_dt),
                torch.zeros_like(self.swa_v_ring, dtype=cmp_dt),
            )
            if not bool(torch.allclose(v_a, v_b, atol=atol, rtol=rtol)):
                return False
        return True


class _GrowingSlotList(list):  # type: ignore[type-arg]
    """List that auto-extends with `None` when indexed at a virtual loop slot >= current length."""

    def _ensure(self, idx: int) -> None:
        if idx >= len(self):
            self.extend([None] * (idx + 1 - len(self)))

    def __getitem__(self, idx: Any) -> Any:
        if isinstance(idx, int) and idx >= 0:
            self._ensure(idx)
        return super().__getitem__(idx)

    def __setitem__(self, idx: Any, value: Any) -> None:
        if isinstance(idx, int) and idx >= 0:
            self._ensure(idx)
        super().__setitem__(idx, value)


class _HybridHasPreviousState:
    """Dual-mode `has_previous_state`: truthy as a property (`HF 5.2-5.5`) and callable `(layer_idx)` (`HF 5.20+`)."""

    def __init__(self, cache: HSSDHybridCache) -> None:
        self._cache = cache

    def __bool__(self) -> bool:
        return self()

    def __call__(self, layer_idx: int | None = None, state_idx: int | None = None) -> bool:
        del state_idx
        c = self._cache
        if layer_idx is not None:
            cs = c.conv_states[layer_idx] if layer_idx < len(c.conv_states) else None
            rs = c.recurrent_states[layer_idx] if layer_idx < len(c.recurrent_states) else None
            return cs is not None or rs is not None
        last_lin = getattr(c, "last_linear_layer", None)
        if isinstance(last_lin, int) and 0 <= last_lin < len(c.conv_states):
            return c.conv_states[last_lin] is not None or c.recurrent_states[last_lin] is not None
        return any(x is not None for x in c.conv_states) or any(x is not None for x in c.recurrent_states)


class HSSDHybridCache(_HFCacheBase):  # type: ignore[misc]
    """Hugging Face `Cache`-compatible hybrid KV + `HSSDCacheState` container (`Bug 3D`).

    Supports full-attention KV caching (`update`), causal short-conv state slots (`conv_states`),
    independent per-(layer, loop-pass) `HSSDCacheState` slots (`recurrent_states`), beam-search
    `reorder_cache`, `reset`, `get_seq_length`, and deduplicated `unique_storage_bytes()`.
    """

    def __init__(
        self,
        config: Any = None,
        *,
        num_layers: int | None = None,
        layer_types: Sequence[str] | None = None,
    ) -> None:
        if _HFCacheBase is not object:
            try:
                super().__init__()
            except TypeError:
                pass
        if layer_types is not None:
            ltypes = [str(x) for x in layer_types]
        elif config is not None and getattr(config, "layer_types", None):
            ltypes = [str(x) for x in config.layer_types]
        else:
            n_cfg = int(num_layers or getattr(config, "num_hidden_layers", 0) or 0)
            ltypes = ["linear_attention"] * n_cfg
        n = int(num_layers) if num_layers is not None else len(ltypes)
        if len(ltypes) < n:
            ltypes.extend(["linear_attention"] * (n - len(ltypes)))
        self._initial_slots = n
        self.layer_types: list[str] = ltypes
        self.key_cache: _GrowingSlotList = _GrowingSlotList([None] * n)
        self.value_cache: _GrowingSlotList = _GrowingSlotList([None] * n)
        self.conv_states: _GrowingSlotList = _GrowingSlotList([None] * n)
        self.recurrent_states: _GrowingSlotList = _GrowingSlotList([None] * n)
        self.first_full: int | None = (
            self.layer_types.index("full_attention") if "full_attention" in self.layer_types else None
        )
        lin_indices = [idx for idx, lt in enumerate(self.layer_types) if lt == "linear_attention"]
        self.last_linear_layer: int | None = lin_indices[-1] if lin_indices else None

    @property
    def has_previous_state(self) -> _HybridHasPreviousState:
        return _HybridHasPreviousState(self)

    def is_recording_past(self, layer_idx: int = 0) -> bool:
        del layer_idx
        return False

    def __len__(self) -> int:
        return max(len(self.key_cache), len(self.recurrent_states))

    def __getitem__(self, layer_idx: int) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del cache_kwargs
        if key_states.shape[-2] == 0:
            kc = self.key_cache[layer_idx]
            vc = self.value_cache[layer_idx]
            if kc is not None and vc is not None:
                return kc, vc
        if self.key_cache[layer_idx] is None:
            self.key_cache[layer_idx] = _compact(key_states)
            self.value_cache[layer_idx] = _compact(value_states)
        else:
            self.key_cache[layer_idx] = _compact(torch.cat([self.key_cache[layer_idx], key_states], dim=-2))
            self.value_cache[layer_idx] = _compact(torch.cat([self.value_cache[layer_idx], value_states], dim=-2))
        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def get_seq_length(self, layer_idx: int | None = None) -> int:
        if layer_idx is not None:
            kc = self.key_cache[layer_idx] if layer_idx < len(self.key_cache) else None
            if isinstance(kc, torch.Tensor):
                return int(kc.shape[-2])
            rs = self.recurrent_states[layer_idx] if layer_idx < len(self.recurrent_states) else None
            if isinstance(rs, HSSDCacheState):
                return int(rs.step_count.max().item()) if rs.step_count.numel() > 0 else 0
            return 0
        if self.first_full is not None and self.first_full < len(self.key_cache):
            kc0 = self.key_cache[self.first_full]
            if isinstance(kc0, torch.Tensor):
                return int(kc0.shape[-2])
        for kc in self.key_cache:
            if isinstance(kc, torch.Tensor):
                return int(kc.shape[-2])
        for rs in self.recurrent_states:
            if isinstance(rs, HSSDCacheState) and rs.step_count.numel() > 0:
                return int(rs.step_count.max().item())
        return 0

    def get_max_cache_shape(self) -> int | None:
        return None

    def reorder_cache(self, beam_idx: torch.LongTensor) -> None:
        for seq in (self.key_cache, self.value_cache, self.conv_states):
            for i, item in enumerate(seq):
                if isinstance(item, torch.Tensor):
                    seq[i] = _compact(item.index_select(0, beam_idx.to(device=item.device, dtype=torch.long)))
        for i, st in enumerate(self.recurrent_states):
            if isinstance(st, HSSDCacheState) or _is_hssd_state_duck(st):
                self.recurrent_states[i] = st.reorder_cache(beam_idx)
            elif isinstance(st, torch.Tensor):
                self.recurrent_states[i] = _compact(
                    st.index_select(0, beam_idx.to(device=st.device, dtype=torch.long))
                )

    def reset(self) -> None:
        n = max(self._initial_slots, len(self))
        self.key_cache[:] = [None] * n
        self.value_cache[:] = [None] * n
        self.conv_states[:] = [None] * n
        self.recurrent_states[:] = [None] * n

    def slot_index(self, layer_idx: int, *, pass_idx: int = 0, stride: int | None = None) -> int:
        if pass_idx < 0:
            raise ValueError(f"pass_idx must be >= 0, got {pass_idx}")
        base = max(int(stride) if stride is not None else self._initial_slots, 1)
        return int(layer_idx) + int(pass_idx) * base

    def update_conv_state(
        self,
        conv_states: torch.Tensor,
        layer_idx: int,
        state_idx: int = 0,
        *,
        conv_kernel_size: int | None = None,
        pass_idx: int = 0,
        stride: int | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        del kwargs
        require(state_idx == 0, f"HSSDHybridCache only supports state_idx=0, got {state_idx}")
        idx = self.slot_index(layer_idx, pass_idx=pass_idx, stride=stride)
        prev = self.conv_states[idx]
        if prev is None:
            full_conv = conv_states
            if conv_kernel_size is not None and full_conv.shape[-1] < int(conv_kernel_size):
                full_conv = F.pad(full_conv, (int(conv_kernel_size) - full_conv.shape[-1], 0), value=0)
        else:
            full_conv = torch.cat([prev, conv_states], dim=-1)
        keep_w = int(conv_kernel_size) if conv_kernel_size is not None else full_conv.shape[-1]
        self.conv_states[idx] = _compact(full_conv[..., -keep_w:])
        return full_conv

    def get_conv_state(
        self,
        layer_idx: int,
        state_idx: int = 0,
        *,
        pass_idx: int = 0,
        stride: int | None = None,
    ) -> torch.Tensor | None:
        require(state_idx == 0, f"HSSDHybridCache only supports state_idx=0, got {state_idx}")
        idx = self.slot_index(layer_idx, pass_idx=pass_idx, stride=stride)
        return self.conv_states[idx] if idx < len(self.conv_states) else None

    def update_recurrent_state(
        self,
        state: HSSDCacheState | torch.Tensor | None,
        layer_idx: int,
        state_idx: int = 0,
        *,
        pass_idx: int = 0,
        stride: int | None = None,
        **kwargs: Any,
    ) -> HSSDCacheState | torch.Tensor | None:
        del kwargs
        require(state_idx == 0, f"HSSDHybridCache only supports state_idx=0, got {state_idx}")
        idx = self.slot_index(layer_idx, pass_idx=pass_idx, stride=stride)
        self.recurrent_states[idx] = state
        return state

    def get_recurrent_state(
        self,
        layer_idx: int,
        state_idx: int = 0,
        *,
        pass_idx: int = 0,
        stride: int | None = None,
    ) -> HSSDCacheState | torch.Tensor | None:
        require(state_idx == 0, f"HSSDHybridCache only supports state_idx=0, got {state_idx}")
        idx = self.slot_index(layer_idx, pass_idx=pass_idx, stride=stride)
        return self.recurrent_states[idx] if idx < len(self.recurrent_states) else None

    def unique_storage_bytes(self) -> int:
        seen: set[int] = set()
        total = 0

        def _acc(t: torch.Tensor) -> None:
            nonlocal total
            if t.numel() <= 0:
                return
            st = t.untyped_storage()
            ptr = int(st.data_ptr())
            if ptr not in seen:
                seen.add(ptr)
                total += int(st.nbytes())

        for seq in (self.key_cache, self.value_cache, self.conv_states):
            for item in seq:
                if isinstance(item, torch.Tensor):
                    _acc(item)
        for rs in self.recurrent_states:
            if isinstance(rs, HSSDCacheState) or _is_hssd_state_duck(rs):
                for t in rs._iter_tensors():
                    _acc(t)
            elif isinstance(rs, torch.Tensor):
                _acc(rs)
        return total


# ============================================================================ core HSSD kernels

def _prepare_qkv_unscaled(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    use_qk_l2norm: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    q, k, v = upcast(query), upcast(key), upcast(value)
    hv = v.shape[2]
    if q.shape[2] != hv:
        require(hv % q.shape[2] == 0, "num_v_heads must be divisible by num_q_heads", q=q, v=v)
        q = q.repeat_interleave(hv // q.shape[2], dim=2)
    if k.shape[2] != hv:
        require(hv % k.shape[2] == 0, "num_v_heads must be divisible by num_k_heads", k=k, v=v)
        k = k.repeat_interleave(hv // k.shape[2], dim=2)
    if use_qk_l2norm:
        q = l2norm(q, dim=-1, eps=1e-6)
        k = l2norm(k, dim=-1, eps=1e-6)
    return q, k, v


def _coerce_init_state(
    initial_state: HSSDCacheState | torch.Tensor | None,
    *,
    bsz: int,
    num_pages: int,
    hv: int,
    dk: int,
    dv: int,
    chunk_size: int,
    window_chunks: int,
    device: torch.device,
    dtype: torch.dtype,
) -> HSSDCacheState:
    if initial_state is None:
        return HSSDCacheState.zeros(
            bsz,
            num_pages,
            hv,
            dk,
            dv,
            chunk_size=chunk_size,
            window_chunks=window_chunks,
            device=device,
            dtype=dtype,
        )
    if isinstance(initial_state, HSSDCacheState):
        require(
            initial_state.chunk_size == chunk_size and initial_state.window_chunks == window_chunks,
            "HSSDCacheState chunk_size/window_chunks mismatch",
            got=(initial_state.chunk_size, initial_state.window_chunks),
            want=(chunk_size, window_chunks),
        )
        return initial_state.to(device=device, dtype=dtype)
    if _is_hssd_state_duck(initial_state):
        require(
            int(initial_state.chunk_size) == chunk_size and int(initial_state.window_chunks) == window_chunks,
            "HSSDCacheState chunk_size/window_chunks mismatch",
            got=(initial_state.chunk_size, initial_state.window_chunks),
            want=(chunk_size, window_chunks),
        )
        return HSSDCacheState(
            step_count=initial_state.step_count,
            chunk_size=chunk_size,
            window_chunks=window_chunks,
            page_states=initial_state.page_states.to(device=device, dtype=dtype),
            theta_cum=initial_state.theta_cum,
            boundary_ring=tuple(t.to(device=device, dtype=dtype) for t in initial_state.boundary_ring),
            swa_k_ring=initial_state.swa_k_ring.to(device=device, dtype=dtype),
            swa_v_ring=initial_state.swa_v_ring.to(device=device, dtype=dtype),
            swa_valid_ring=initial_state.swa_valid_ring,
        )
    if isinstance(initial_state, torch.Tensor):
        return HSSDCacheState.from_tensor(
            initial_state,
            num_pages=num_pages,
            chunk_size=chunk_size,
            window_chunks=window_chunks,
        ).to(device=device, dtype=dtype)
    raise TypeError(f"Unsupported initial_state type: {type(initial_state).__name__}")


def _cast_state_dtype(
    state: HSSDCacheState,
    state_dtype: str | torch.dtype | None,
    acc_dt: torch.dtype,
) -> HSSDCacheState:
    target_dt = _resolve_state_dtype(state_dtype, acc_dt)
    if state.dtype == target_dt and state.theta_cum.dtype == torch.float64 and state.has_compact_storage():
        return state
    return state.to(device=state.device, dtype=target_dt)


def _normalize_gamma_spec(
    gamma: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
    *,
    num_pages: int,
    dk: int,
    acc_dt: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor | None, torch.Tensor]:
    """Normalize `gamma` without materializing 5D `[B, T, M, H_v, d_k]`!

    Returns `(page_scale [M] | None, chan_or_full)`:
      * When `page_scale is not None`, `chan_or_full` is 4D `[B, T, H_v, d_k // 2 or d_k]`.
      * When `page_scale is None`, `chan_or_full` is legacy 5D `[B, T, M, H_v, d_k // 2 or d_k]`.
    """
    if isinstance(gamma, tuple):
        require(len(gamma) == 2, "factored gamma tuple must be (gamma_page_scale, gamma_chan_pair)")
        ps, chan = gamma
        ps_f = upcast(ps).to(device=device, dtype=acc_dt).clamp_min(0.0)
        require(ps_f.shape == (num_pages,), f"gamma_page_scale shape {tuple(ps_f.shape)} != ({num_pages},)")
        ch_f = upcast(chan).to(device=device, dtype=acc_dt).clamp_min(0.0)
        require(
            ch_f.ndim == 4 and ch_f.shape[-1] in (dk // 2, dk),
            "gamma_chan_pair must have shape [B, T, H_v, d_k // 2] or [B, T, H_v, d_k]",
            gamma_chan_pair=ch_f,
        )
        return ps_f, ch_f

    gf = upcast(gamma).to(device=device, dtype=acc_dt).clamp_min(0.0)
    if gf.ndim == 4:
        require(gf.shape[-1] in (dk // 2, dk), "4D gamma last dim must be d_k // 2 or d_k", gamma=gf)
        ones_p = torch.ones(num_pages, device=device, dtype=acc_dt)
        return ones_p, gf
    require(
        gf.ndim == 5 and gf.shape[2] == num_pages and gf.shape[-1] in (dk // 2, dk),
        "gamma must be factored (ps[M], chan[B,T,H,D//2]) or Tensor [B,T,H,D] / [B,T,M,H,D]",
        gamma=gf,
    )
    return None, gf


def _slice_gamma_input(
    gamma: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
    *,
    batch_sl: slice | None = None,
    time_sl: slice | torch.Tensor | None = None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    def _sl_tensor(t: torch.Tensor) -> torch.Tensor:
        out = t if batch_sl is None else t[batch_sl]
        if time_sl is None:
            return out
        if isinstance(time_sl, torch.Tensor):
            return out.index_select(1, time_sl)
        return out[:, time_sl]

    if isinstance(gamma, tuple):
        return (gamma[0], _sl_tensor(gamma[1]))
    return _sl_tensor(gamma)


def _materialize_chunk_gamma(
    page_scale: torch.Tensor | None,
    chan_c: torch.Tensor,
    *,
    num_pages: int,
    dk: int,
) -> torch.Tensor:
    """Expand `gamma` to `[B, c_len, M, H_v, d_k]` ONLY inside a single `c_len <= C` chunk."""
    if chan_c.shape[-1] == dk // 2:
        chan_c = chan_c.repeat_interleave(2, dim=-1)
    if page_scale is None:
        return chan_c
    return page_scale.view(1, 1, num_pages, 1, 1) * chan_c.unsqueeze(2)


def _wy_kda_state_update(
    page_states: torch.Tensor,
    k_eff: torch.Tensor,
    v_eff: torch.Tensor,
    log_a: torch.Tensor,
) -> torch.Tensor:
    """Exact overflow-free, zero-host-sync chunk-parallel WY state transition for symmetric DPLR KDA.

    Uses block-factored non-positive cumulative log-decays (`b_sz = 8`): because `log_a <= 0`,
    `cum_log_a` is monotonically non-increasing along time, so every diagonal `i > j` difference
    AND every off-diagonal block split `(g[i] - g[b_end]) + (g[b_end] - g[j])` (`i > b_end >= j`)
    is unconditionally `<= 0.0` (never overflowing `exp` even for `span = 10,000`, with zero
    `bool()` / `.item()` host synchronizations).

    Shapes:
        page_states: [B, M, H_v, d_k, d_v]
        k_eff:       [B, M, H_v, c_len, d_k]
        v_eff:       [B, M, H_v, c_len, d_v]
        log_a:       [B, M, H_v, c_len, d_k]  (<= 0)
    """
    log_a_neg = log_a.clamp(max=0.0)
    c_len = k_eff.shape[-2]
    if c_len == 1:
        dec = log_a_neg.squeeze(-2).exp().unsqueeze(-1) * page_states
        ke = k_eff.squeeze(-2)
        ve = v_eff.squeeze(-2)
        mem = (ke.unsqueeze(-2) @ dec).squeeze(-2)
        return dec + ke.unsqueeze(-1) * (ve - mem).unsqueeze(-2)

    cum_log_a = torch.cumsum(log_a_neg, dim=-2)
    b_sz = 8
    n_blk = (c_len + b_sz - 1) // b_sz
    row_blocks: list[torch.Tensor] = []

    for bi in range(n_blk):
        i0 = bi * b_sz
        i1 = min(c_len, (bi + 1) * b_sz)
        bi_len = i1 - i0
        k_i = k_eff[..., i0:i1, :]
        g_i = cum_log_a[..., i0:i1, :]

        col_parts: list[torch.Tensor] = []
        for bj in range(bi):
            j0 = bj * b_sz
            j1 = (bj + 1) * b_sz
            g_piv = cum_log_a[..., j1 - 1 : j1, :]
            # i >= i0 >= j1 > j1 - 1 => g_i - g_piv <= 0; j <= j1 - 1 => g_piv - g_j <= 0
            lhs = k_i * (g_i - g_piv).exp()
            rhs_k = k_eff[..., j0:j1, :] * (g_piv - cum_log_a[..., j0:j1, :]).exp()
            col_parts.append(lhs @ rhs_k.transpose(-1, -2))

        idx_d = torch.arange(bi_len, device=k_eff.device)
        strict_lo = idx_d[:, None] > idx_d[None, :]
        diff_d = g_i[..., :, None, :] - g_i[..., None, :, :]
        safe_d = torch.where(strict_lo.unsqueeze(-1), diff_d, torch.zeros_like(diff_d))
        diag_blk = (k_i[..., :, None, :] * k_i[..., None, :, :] * safe_d.exp()).sum(dim=-1)
        diag_blk = torch.where(strict_lo, diag_blk, torch.zeros_like(diag_blk))
        col_parts.append(diag_blk)

        if i1 < c_len:
            z_pad = k_eff.new_zeros(*k_eff.shape[:-2], bi_len, c_len - i1)
            col_parts.append(z_pad)

        row_blocks.append(col_parts[0] if len(col_parts) == 1 else torch.cat(col_parts, dim=-1))

    l_mat = row_blocks[0] if len(row_blocks) == 1 else torch.cat(row_blocks, dim=-2)

    gamma_from_0 = cum_log_a.exp()
    rhs = v_eff - (gamma_from_0 * k_eff) @ page_states
    eye = torch.eye(c_len, dtype=k_eff.dtype, device=k_eff.device)
    u_wy = torch.linalg.solve_triangular(eye + l_mat, rhs, upper=False, unitriangular=True)

    gamma_to_end = (cum_log_a[..., -1:, :] - cum_log_a).exp()
    d_chunk = cum_log_a[..., -1, :].exp().unsqueeze(-1)
    return d_chunk * page_states + (gamma_to_end * k_eff).transpose(-1, -2) @ u_wy


def _routed_bank_readout(
    vec_c: torch.Tensor,
    state_bank: torch.Tensor,
    top_idx: torch.Tensor,
    top_w: torch.Tensor,
) -> torch.Tensor:
    """Read `vec_c` (`[B, c_len, H_v, d_k]`) from `top_idx` (`[B, c_len, k]`) of `state_bank`.

    For decode / short spans (`c_len <= 4`) when `k < M`, gathers ONLY the `k` active pages before
    matmul (`4x` fewer state bytes & FLOPs for `k=2, M=8`).
    """
    bsz, c_len, hv, dk = vec_c.shape
    num_pages = state_bank.shape[1]
    dv = state_bank.shape[-1]
    k_sel = top_idx.shape[-1]

    if k_sel < num_pages and c_len == 1:
        g_idx = top_idx[:, 0, :, None, None, None].expand(bsz, k_sel, hv, dk, dv)
        s_sel = torch.gather(state_bank, dim=1, index=g_idx)               # [B, k, H_v, d_k, d_v]
        v_q = vec_c.permute(0, 2, 1, 3).unsqueeze(1)                       # [B, 1, H_v, 1, d_k]
        picked_1 = (v_q @ s_sel).squeeze(-2)                               # [B, k, H_v, d_v]
        return (picked_1 * top_w[:, 0, :, None, None]).sum(dim=1, keepdim=True)

    if k_sel < num_pages and c_len <= 4:
        flat_idx = top_idx.reshape(bsz, c_len * k_sel, 1, 1, 1).expand(bsz, c_len * k_sel, hv, dk, dv)
        s_sel = torch.gather(state_bank, dim=1, index=flat_idx).view(bsz, c_len, k_sel, hv, dk, dv)
        v_q = vec_c.unsqueeze(2).unsqueeze(-2)                             # [B, c_len, 1, H_v, 1, d_k]
        picked_s = (v_q @ s_sel).squeeze(-2)                               # [B, c_len, k, H_v, d_v]
        return (picked_s * top_w.unsqueeze(-1).unsqueeze(-1)).sum(dim=2)

    v_all = vec_c.permute(0, 2, 1, 3).unsqueeze(1)                         # [B, 1, H_v, c_len, d_k]
    all_out = (v_all @ state_bank).permute(0, 3, 1, 2, 4)                  # [B, c_len, M, H_v, d_v]
    gather_idx = top_idx.unsqueeze(-1).unsqueeze(-1).expand(bsz, c_len, k_sel, hv, dv)
    picked = torch.gather(all_out, dim=2, index=gather_idx)
    return (picked * top_w.unsqueeze(-1).unsqueeze(-1)).sum(dim=2)


def _far_readout(
    q_hat_c: torch.Tensor,
    s_far: torch.Tensor,
    read_logits_c: torch.Tensor,
    top_k_pages: int,
) -> torch.Tensor:
    """Top-`k` page-routed readout from `S_boundary[k_far]` (`[B, c_len, H_v, d_v]`)."""
    pi_read = F.softmax(read_logits_c, dim=-1)
    top_vals, top_idx = torch.topk(pi_read, k=top_k_pages, dim=-1)
    top_w = top_vals / top_vals.sum(dim=-1, keepdim=True).clamp_min(1e-30)
    return _routed_bank_readout(q_hat_c, s_far, top_idx, top_w)


def _compute_root_surprise_and_routing(
    k_hat_c: torch.Tensor,
    v_c: torch.Tensor,
    s_chunk_start_bank: torch.Tensor,
    sbias_c: torch.Tensor,
    surprise_slope: torch.Tensor,
    surprise_thresh: torch.Tensor,
    write_logits_c: torch.Tensor,
    top_k_pages: int,
    wmask_c: torch.Tensor | None,
    g_override_c: torch.Tensor | None,
    *,
    use_surprise: bool = True,
    surprise_mode: str = "routed",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute `(g_page, u_sum, valid_toks, top_idx, g_active)` with actual sparse STE duty (`Bug 3C`).

    Returns:
      * `g_page`: `[B, c_len, M, H_v]` full-bank root write gates
      * `u_sum`: `[M]` actual sparse Top-`k` write-mass sum with STE router gradient (`pi_ste * g^2`)
      * `valid_toks`: `[]` scalar valid token count
      * `top_idx`: `[B, c_len, k]` selected write page indices
      * `g_active`: `[B, c_len, k, H_v]` root write gates on the `k` selected pages (for `O(k)` decode)
    """
    bsz, c_len, hv, _dk = k_hat_c.shape
    pi_write = F.softmax(write_logits_c, dim=-1)
    top_vals, top_idx = torch.topk(pi_write, k=top_k_pages, dim=-1)
    top_probs = top_vals / top_vals.sum(dim=-1, keepdim=True).clamp_min(1e-30)

    if g_override_c is not None:
        g_root = (
            g_override_c.squeeze(-1)
            if (g_override_c.ndim == 4 and g_override_c.shape[-1] == 1)
            else g_override_c
        )
        if g_root.shape != (bsz, c_len, hv):
            g_root = g_root.expand(bsz, c_len, hv)
    elif not use_surprise:
        g_root = torch.ones(bsz, c_len, hv, device=k_hat_c.device, dtype=k_hat_c.dtype)
    else:
        if surprise_mode == "routed" and s_chunk_start_bank.ndim == 5:
            pred_v = _routed_bank_readout(k_hat_c, s_chunk_start_bank, top_idx, top_probs)
        else:
            s0 = s_chunk_start_bank[:, 0] if s_chunk_start_bank.ndim == 5 else s_chunk_start_bank
            pred_v = (k_hat_c.transpose(1, 2) @ s0).transpose(1, 2)
        eps_err = _safe_norm(v_c - pred_v, dim=-1) / (_safe_norm(v_c, dim=-1) + 1e-6)
        slope = surprise_slope.view(1, 1, hv)
        thresh = surprise_thresh.view(1, 1, hv)
        sb_eff = sbias_c.squeeze(-1) if (sbias_c.ndim == 4 and sbias_c.shape[-1] == 1) else sbias_c
        g_root = torch.sigmoid(slope * (eps_err - thresh) + sb_eff)

    if wmask_c is not None:
        g_root = g_root * wmask_c.unsqueeze(-1)
        valid_toks = wmask_c.sum()
    else:
        valid_toks = g_root.new_tensor(float(bsz * c_len))

    # Bug 3C: Actual sparse Top-k write probability + Straight-Through Estimator into `write_logits`
    pi_actual = torch.zeros_like(pi_write).scatter(-1, top_idx, top_probs)
    pi_ste = pi_actual + (pi_write - pi_write.detach())
    g_sq_mean = (g_root * g_root).mean(dim=-1, keepdim=True)
    u_sum = (pi_ste * g_sq_mean).sum(dim=(0, 1))

    top_roots = torch.sqrt(top_probs.clamp_min(1e-30))
    g_active = top_roots.unsqueeze(-1) * g_root.unsqueeze(2)
    write_root = torch.zeros_like(pi_write).scatter(-1, top_idx, top_roots)
    g_page = write_root.unsqueeze(-1) * g_root.unsqueeze(2)
    return g_page, u_sum, valid_toks, top_idx, g_active


def compute_write_budget_loss(
    duty_vec: torch.Tensor,
    mu_max: torch.Tensor | Sequence[float],
    mu_min_ratio: float = 0.2,
    budget_weight: float = 1.0,
) -> torch.Tensor:
    """Two-sided cumulative sparse write-mass budget regularizer (`Fixes B2 & 3C`)."""
    if isinstance(mu_max, torch.Tensor):
        target_max = mu_max.to(device=duty_vec.device, dtype=duty_vec.dtype)
    else:
        target_max = torch.tensor(list(mu_max), device=duty_vec.device, dtype=duty_vec.dtype)
    target_min = float(mu_min_ratio) * target_max
    over = F.relu(duty_vec - target_max)
    under = F.relu(target_min - duty_vec)
    return float(budget_weight) * (over * over + under * under).sum()


def _validate_packed_boundaries(
    *,
    bsz: int,
    seq_len: int,
    initial_state: Any,
    cu_seqlens: Any,
    seq_idx: torch.Tensor | None,
) -> list[list[tuple[int, int]]] | None:
    """Validate `cu_seqlens` / `seq_idx` (`Bug 3E`) and return per-row document spans `[s0, s1)`."""
    if cu_seqlens is None and seq_idx is None:
        return None
    if cu_seqlens is not None and seq_idx is not None:
        raise ValueError("Specify at most one of `cu_seqlens` or `seq_idx`, not both")

    if cu_seqlens is not None:
        if bsz != 1:
            raise ValueError(f"cu_seqlens requires batch size B == 1, got B == {bsz}")
        if initial_state is not None:
            raise ValueError("cu_seqlens cannot be combined with a non-None initial_state")
        if isinstance(cu_seqlens, torch.Tensor):
            if cu_seqlens.ndim != 1:
                raise ValueError(f"cu_seqlens tensor must be 1D, got shape {tuple(cu_seqlens.shape)}")
            cu = [int(x) for x in cu_seqlens.detach().cpu().tolist()]
        elif isinstance(cu_seqlens, Sequence):
            cu = [int(x) for x in cu_seqlens]
        else:
            raise ValueError(f"Unsupported cu_seqlens type: {type(cu_seqlens).__name__}")
        if len(cu) < 2:
            raise ValueError(f"cu_seqlens must have length >= 2, got {cu}")
        if cu[0] != 0 or cu[-1] != seq_len:
            raise ValueError(f"cu_seqlens must start at 0 and end at seq_len={seq_len}, got {cu}")
        for i in range(len(cu) - 1):
            if cu[i] > cu[i + 1]:
                raise ValueError(f"cu_seqlens must be monotonic non-decreasing, got {cu}")
        spans = [(cu[i], cu[i + 1]) for i in range(len(cu) - 1) if cu[i + 1] > cu[i]]
        return [spans]

    assert seq_idx is not None
    if not isinstance(seq_idx, torch.Tensor) or seq_idx.ndim != 2 or seq_idx.shape != (bsz, seq_len):
        got = tuple(seq_idx.shape) if isinstance(seq_idx, torch.Tensor) else type(seq_idx).__name__
        raise ValueError(f"seq_idx must be a 2D Tensor of shape {(bsz, seq_len)}, got {got}")
    if seq_idx.is_floating_point() or seq_idx.is_complex():
        raise ValueError(f"seq_idx must be an integer Tensor, got dtype {seq_idx.dtype}")
    s_cpu = seq_idx.detach().cpu()
    if seq_len > 0 and bool((s_cpu < 0).any()):
        raise ValueError("seq_idx entries must be non-negative")
    if seq_len > 1 and bool((s_cpu[:, 1:] < s_cpu[:, :-1]).any()):
        raise ValueError("seq_idx must be monotonic non-decreasing along dim=1")
    if initial_state is not None and seq_len > 0 and bool((s_cpu[:, 0] != s_cpu[:, -1]).any()):
        raise ValueError("multi-document seq_idx cannot be combined with a non-None initial_state")

    row_spans: list[list[tuple[int, int]]] = []
    for b in range(bsz):
        if seq_len == 0:
            row_spans.append([])
            continue
        _vals, counts = torch.unique_consecutive(s_cpu[b], return_counts=True)
        cur = 0
        b_list: list[tuple[int, int]] = []
        for cnt in counts.tolist():
            nxt = cur + int(cnt)
            b_list.append((cur, nxt))
            cur = nxt
        row_spans.append(b_list)
    return row_spans


def recurrent_hssd_step(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    gamma: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
    b_gate: torch.Tensor,
    w_gate: torch.Tensor,
    theta: torch.Tensor,
    surprise_bias: torch.Tensor,
    surprise_slope: torch.Tensor,
    surprise_thresh: torch.Tensor,
    write_logits: torch.Tensor,
    read_logits: torch.Tensor,
    q_swa: torch.Tensor,
    k_swa: torch.Tensor,
    v_swa: torch.Tensor,
    state: HSSDCacheState,
    *,
    boltzmann_gate: PeakBoltzmannGate | None = None,
    top_k_pages: int = 2,
    write_mask: torch.Tensor | None = None,
    pad_mask: torch.Tensor | None = None,
    g_override: torch.Tensor | None = None,
    use_qk_l2norm_in_kernel: bool = False,
    scale: float | None = None,
    state_dtype: str | torch.dtype = "auto",
    use_swa: bool = True,
    use_phase: bool = True,
    use_surprise: bool = True,
    surprise_mode: str = "routed",
    return_duty_sum: bool = False,
) -> tuple[torch.Tensor, HSSDCacheState] | tuple[torch.Tensor, HSSDCacheState, torch.Tensor, torch.Tensor]:
    """Single-token `O(1)`-memory HSSD v3.1 autoregressive decode step (`O(k)` active-page update).

    Accepts `[B, 1, ...]` (returning `[B, 1, H_v, d_v]`) or `[B, ...]` (returning `[B, H_v, d_v]`).
    """
    squeeze_time = query.ndim == 3
    if squeeze_time:
        query = query.unsqueeze(1)
        key = key.unsqueeze(1)
        value = value.unsqueeze(1)
        gamma = (gamma[0], gamma[1].unsqueeze(1)) if isinstance(gamma, tuple) else gamma.unsqueeze(1)
        b_gate = b_gate.unsqueeze(1)
        w_gate = w_gate.unsqueeze(1)
        theta = theta.unsqueeze(1)
        surprise_bias = surprise_bias.unsqueeze(1)
        write_logits = write_logits.unsqueeze(1)
        read_logits = read_logits.unsqueeze(1)
        q_swa = q_swa.unsqueeze(1)
        k_swa = k_swa.unsqueeze(1)
        v_swa = v_swa.unsqueeze(1)
        if g_override is not None:
            g_override = g_override.unsqueeze(1)
    require(query.shape[1] == 1, "recurrent_hssd_step expects T == 1", query=query)

    init_dtype = query.dtype
    bsz = query.shape[0]
    hv = value.shape[2]
    dk = key.shape[-1]
    dv = value.shape[-1]
    num_pages = state.page_states.shape[1]
    acc_dt = torch.float64 if init_dtype == torch.float64 else torch.float32

    pm_bool: torch.Tensor | None = None
    has_pad_zeros = False
    if pad_mask is not None:
        pm_bool = pad_mask.reshape(bsz, 1).to(device=query.device) > 0.5
        has_pad_zeros = not bool(pm_bool.all())
    inhomogeneous_steps = bsz > 1 and not bool((state.step_count == state.step_count[0]).all())

    if has_pad_zeros or inhomogeneous_steps:
        row_outs: list[torch.Tensor] = []
        row_states: list[HSSDCacheState] = []
        u_acc = query.new_zeros(num_pages, dtype=acc_dt)
        cnt_acc = query.new_zeros((), dtype=acc_dt)
        for b in range(bsz):
            st_b = state[b : b + 1]
            if pm_bool is not None and not bool(pm_bool[b, 0]):
                row_outs.append(query.new_zeros(1, 1, hv, dv, dtype=init_dtype))
                row_states.append(_cast_state_dtype(st_b, state_dtype, acc_dt))
                continue
            sl_b = slice(b, b + 1)
            o_b, nxt_b, u_b, c_b = recurrent_hssd_step(
                query[sl_b],
                key[sl_b],
                value[sl_b],
                _slice_gamma_input(gamma, batch_sl=sl_b),
                b_gate[sl_b],
                w_gate[sl_b],
                theta[sl_b],
                surprise_bias[sl_b],
                surprise_slope,
                surprise_thresh,
                write_logits[sl_b],
                read_logits[sl_b],
                q_swa[sl_b],
                k_swa[sl_b],
                v_swa[sl_b],
                st_b,
                boltzmann_gate=boltzmann_gate,
                top_k_pages=top_k_pages,
                write_mask=None if write_mask is None else write_mask.reshape(bsz, 1)[sl_b],
                pad_mask=None,
                g_override=None if g_override is None else g_override[sl_b],
                use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
                scale=scale,
                state_dtype=state_dtype,
                use_swa=use_swa,
                use_phase=use_phase,
                use_surprise=use_surprise,
                surprise_mode=surprise_mode,
                return_duty_sum=True,
            )
            row_outs.append(o_b)
            row_states.append(nxt_b)
            u_acc = u_acc + u_b
            cnt_acc = cnt_acc + c_b
        out_cat = torch.cat(row_outs, dim=0)
        st_cat = torch.cat(row_states, dim=0)
        out_ret = out_cat.squeeze(1) if squeeze_time else out_cat
        if return_duty_sum:
            return out_ret, st_cat, u_acc, cnt_acc
        return out_ret, st_cat

    q, k, v = _prepare_qkv_unscaled(query, key, value, use_qk_l2norm_in_kernel)
    st = state.to(device=q.device, dtype=acc_dt)
    ps_gamma, ch_gamma = _normalize_gamma_spec(
        gamma, num_pages=num_pages, dk=dk, acc_dt=acc_dt, device=q.device
    )

    bf = upcast(b_gate).to(acc_dt)
    wf = upcast(w_gate).to(acc_dt)
    sb = upcast(surprise_bias).to(acc_dt)
    slp = upcast(surprise_slope).to(acc_dt)
    thr = upcast(surprise_thresh).to(acc_dt)
    wl = upcast(write_logits).to(acc_dt)
    rl = upcast(read_logits).to(acc_dt)
    qs = upcast(q_swa).to(acc_dt)
    ks = upcast(k_swa).to(acc_dt)
    vs = upcast(v_swa).to(acc_dt)
    go = None if g_override is None else upcast(g_override).to(acc_dt)

    th64 = theta.to(dtype=torch.float64)
    if write_mask is not None:
        wm = write_mask.reshape(bsz, 1).to(device=q.device, dtype=acc_dt).clamp(0.0, 1.0)
        valid_tok = wm.squeeze(1) > 0.5
        th64 = th64 * wm.to(dtype=torch.float64).unsqueeze(-1).unsqueeze(-1)
    else:
        wm = None
        valid_tok = torch.ones(bsz, device=q.device, dtype=torch.bool)
    if not use_phase:
        th64 = torch.zeros_like(th64)

    idx0 = int(st.step_count[0].item())
    c_size = st.chunk_size
    w_chunks = st.window_chunks
    s_cstart_bank = st.boundary_ring[-1]

    theta_cum = _compact(torch.remainder(st.theta_cum + th64.squeeze(1), 2.0 * math.pi))
    sc = (dk ** -0.5) if scale is None else float(scale)
    q_hat = apply_2d_rotation(q, -theta_cum.unsqueeze(1)) if use_phase else q
    if sc != 1.0:
        q_hat = q_hat * sc
    k_hat = apply_2d_rotation(k, -theta_cum.unsqueeze(1)) if use_phase else k

    # 1. Disjoint far-horizon readout from S_boundary[max(0, n - W)] == st.boundary_ring[0]
    o_far = _far_readout(q_hat, st.boundary_ring[0], rl, top_k_pages)

    # 2. Update O(1) circular SWA ring buffer at slot = idx0 % ((W + 1) * C) and attend over I_loc
    max_win = st.max_swa_len
    slot = idx0 % max_win
    slot_idx = torch.tensor([slot], device=q.device, dtype=torch.long)
    swa_k_ring = _compact(st.swa_k_ring.index_copy(1, slot_idx, ks))
    swa_v_ring = _compact(st.swa_v_ring.index_copy(1, slot_idx, vs))
    swa_valid_ring = _compact(st.swa_valid_ring.index_copy(1, slot_idx, valid_tok.unsqueeze(1)))

    if use_swa:
        _n_chunk, _k_far, far_cutoff = disjoint_horizon_bounds(idx0 + 1, c_size, w_chunks)
        win_indices = torch.arange(far_cutoff, idx0 + 1, device=q.device) % max_win
        k_win = swa_k_ring.index_select(1, win_indices).transpose(1, 2)       # [B, H_v, N_t, d_k]
        v_win = swa_v_ring.index_select(1, win_indices).transpose(1, 2)       # [B, H_v, N_t, d_v]
        val_win = swa_valid_ring.index_select(1, win_indices)[:, None, None, :]  # [B, 1, 1, N_t]

        q_loc = qs.transpose(1, 2)                                            # [B, H_v, 1, d_k]
        scores = (q_loc @ k_win.transpose(-1, -2)) * (dk ** -0.5)             # [B, H_v, 1, N_t]
        has_valid = val_win.any(dim=-1)
        safe_val = torch.where(has_valid.unsqueeze(-1), val_win, torch.ones_like(val_win))
        masked_sc = scores.masked_fill(~safe_val, float("-inf"))
        probs = torch.softmax(masked_sc, dim=-1)
        probs = torch.where(has_valid.unsqueeze(-1), probs, torch.zeros_like(probs))
        o_loc = (probs @ v_win).transpose(1, 2)                               # [B, 1, H_v, d_v]

        if boltzmann_gate is not None:
            p_hit = boltzmann_gate(scores, valid_mask=val_win).transpose(1, 2).unsqueeze(-1)
        else:
            default_gate = PeakBoltzmannGate(1).to(device=q.device, dtype=acc_dt)
            p_hit = default_gate(scores, valid_mask=val_win).transpose(1, 2).unsqueeze(-1)

        out = p_hit * o_loc + (1.0 - p_hit) * o_far
    else:
        out = o_far

    # 3. Routed chunk-start root-surprise `s = g^2` & `O(k)` active-page state update (4x speedup at k=2, M=8)
    _g_page, u_sum, valid_cnt, top_idx, g_active = _compute_root_surprise_and_routing(
        k_hat,
        v,
        s_cstart_bank,
        sb,
        slp,
        thr,
        wl,
        top_k_pages,
        wm,
        go,
        use_surprise=use_surprise,
        surprise_mode=surprise_mode,
    )
    sqrt_b = _safe_sqrt(bf[:, 0])
    sqrt_w = _safe_sqrt(wf[:, 0])
    g_sel = g_active[:, 0]                                                    # [B, k, H_v]
    t_idx = top_idx[:, 0]                                                     # [B, k]
    p_idx = t_idx[:, :, None, None, None].expand(bsz, top_k_pages, hv, dk, dv)
    s_sel = torch.gather(st.page_states, dim=1, index=p_idx)                  # [B, k, H_v, d_k, d_v]

    if ps_gamma is not None:
        ch0 = ch_gamma[:, 0]
        if ch0.shape[-1] == dk // 2:
            ch0 = ch0.repeat_interleave(2, dim=-1)
        ps_sel = ps_gamma.index_select(0, t_idx.reshape(-1)).view(bsz, top_k_pages, 1, 1)
        gam_sel = ps_sel * ch0.unsqueeze(1)                                   # [B, k, H_v, d_k]
    else:
        ch5 = ch_gamma[:, 0]
        if ch5.shape[-1] == dk // 2:
            ch5 = ch5.repeat_interleave(2, dim=-1)
        g_idx_gam = t_idx[:, :, None, None].expand(bsz, top_k_pages, hv, dk)
        gam_sel = torch.gather(ch5, dim=1, index=g_idx_gam)

    dec_sel = (-(g_sel * g_sel).unsqueeze(-1) * gam_sel).clamp(max=0.0).exp().unsqueeze(-1) * s_sel
    ke_sel = g_sel.unsqueeze(-1) * (sqrt_b * k_hat[:, 0]).unsqueeze(1)
    ve_sel = g_sel.unsqueeze(-1) * (sqrt_w * v[:, 0]).unsqueeze(1)
    mem_sel = (ke_sel.unsqueeze(-2) @ dec_sel).squeeze(-2)
    updated_sel = dec_sel + ke_sel.unsqueeze(-1) * (ve_sel - mem_sel).unsqueeze(-2)
    new_pages = _compact(st.page_states.scatter(1, p_idx, updated_sel))

    new_step_int = idx0 + 1
    new_step_vec = st.step_count + 1
    new_b_ring = (*st.boundary_ring[1:], new_pages) if new_step_int % c_size == 0 else st.boundary_ring

    next_state = _cast_state_dtype(
        HSSDCacheState(
            step_count=new_step_vec,
            chunk_size=c_size,
            window_chunks=w_chunks,
            page_states=new_pages,
            theta_cum=theta_cum,
            boundary_ring=new_b_ring,
            swa_k_ring=swa_k_ring,
            swa_v_ring=swa_v_ring,
            swa_valid_ring=swa_valid_ring,
        ),
        state_dtype,
        acc_dt,
    )
    out_cast = out.squeeze(1).to(init_dtype) if squeeze_time else out.to(init_dtype)
    if return_duty_sum:
        return out_cast, next_state, u_sum, valid_cnt
    return out_cast, next_state


def _recurrent_hssd_raw(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    gamma: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
    b_gate: torch.Tensor,
    w_gate: torch.Tensor,
    theta: torch.Tensor,
    surprise_bias: torch.Tensor,
    surprise_slope: torch.Tensor,
    surprise_thresh: torch.Tensor,
    write_logits: torch.Tensor,
    read_logits: torch.Tensor,
    q_swa: torch.Tensor,
    k_swa: torch.Tensor,
    v_swa: torch.Tensor,
    *,
    boltzmann_gate: PeakBoltzmannGate | None,
    chunk_size: int,
    window_chunks: int,
    num_pages: int,
    top_k_pages: int,
    initial_state: HSSDCacheState | torch.Tensor | None,
    output_final_state: bool,
    write_mask: torch.Tensor | None,
    pad_mask: torch.Tensor | None,
    cu_seqlens: Any,
    seq_idx: torch.Tensor | None,
    g_override: torch.Tensor | None,
    use_qk_l2norm_in_kernel: bool,
    scale: float | None,
    state_dtype: str | torch.dtype,
    use_swa: bool,
    use_phase: bool,
    use_surprise: bool,
    surprise_mode: str,
) -> tuple[torch.Tensor, HSSDCacheState | None, torch.Tensor, torch.Tensor]:
    bsz, seq_len, hv, dv = value.shape
    dk = key.shape[-1]
    init_dtype = query.dtype
    acc_dt = torch.float64 if init_dtype == torch.float64 else torch.float32

    doc_spans = _validate_packed_boundaries(
        bsz=bsz, seq_len=seq_len, initial_state=initial_state, cu_seqlens=cu_seqlens, seq_idx=seq_idx
    )
    if doc_spans is not None:
        row_outs: list[torch.Tensor] = []
        row_states: list[HSSDCacheState] = []
        u_tot = query.new_zeros(num_pages, dtype=acc_dt)
        cnt_tot = query.new_zeros((), dtype=acc_dt)
        for b, spans_b in enumerate(doc_spans):
            sl_b = slice(b, b + 1)
            doc_outs: list[torch.Tensor] = []
            last_st: HSSDCacheState | None = None
            for s0, s1 in spans_b:
                sl_t = slice(s0, s1)
                o_d, st_d, u_d, c_d = _recurrent_hssd_raw(
                    query[sl_b, sl_t],
                    key[sl_b, sl_t],
                    value[sl_b, sl_t],
                    _slice_gamma_input(gamma, batch_sl=sl_b, time_sl=sl_t),
                    b_gate[sl_b, sl_t],
                    w_gate[sl_b, sl_t],
                    theta[sl_b, sl_t],
                    surprise_bias[sl_b, sl_t],
                    surprise_slope,
                    surprise_thresh,
                    write_logits[sl_b, sl_t],
                    read_logits[sl_b, sl_t],
                    q_swa[sl_b, sl_t],
                    k_swa[sl_b, sl_t],
                    v_swa[sl_b, sl_t],
                    boltzmann_gate=boltzmann_gate,
                    chunk_size=chunk_size,
                    window_chunks=window_chunks,
                    num_pages=num_pages,
                    top_k_pages=top_k_pages,
                    initial_state=None,
                    output_final_state=output_final_state,
                    write_mask=None if write_mask is None else write_mask[sl_b, sl_t],
                    pad_mask=None if pad_mask is None else pad_mask[sl_b, sl_t],
                    cu_seqlens=None,
                    seq_idx=None,
                    g_override=None if g_override is None else g_override[sl_b, sl_t],
                    use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
                    scale=scale,
                    state_dtype=state_dtype,
                    use_swa=use_swa,
                    use_phase=use_phase,
                    use_surprise=use_surprise,
                    surprise_mode=surprise_mode,
                )
                doc_outs.append(o_d)
                last_st = st_d
                u_tot = u_tot + u_d
                cnt_tot = cnt_tot + c_d
            row_outs.append(
                torch.cat(doc_outs, dim=1)
                if doc_outs
                else query.new_zeros(1, 0, hv, dv, dtype=init_dtype)
            )
            if output_final_state:
                if last_st is None:
                    last_st = _cast_state_dtype(
                        HSSDCacheState.zeros(
                            1, num_pages, hv, dk, dv,
                            chunk_size=chunk_size, window_chunks=window_chunks,
                            device=query.device, dtype=acc_dt,
                        ),
                        state_dtype,
                        acc_dt,
                    )
                row_states.append(last_st)
        out_all = row_outs[0] if len(row_outs) == 1 else torch.cat(row_outs, dim=0)
        st_all = (
            (row_states[0] if len(row_states) == 1 else torch.cat(row_states, dim=0))
            if output_final_state
            else None
        )
        return out_all, st_all, u_tot, cnt_tot

    state = _coerce_init_state(
        initial_state,
        bsz=bsz,
        num_pages=num_pages,
        hv=hv,
        dk=dk,
        dv=dv,
        chunk_size=chunk_size,
        window_chunks=window_chunks,
        device=query.device,
        dtype=acc_dt,
    )

    pm_bool: torch.Tensor | None = None
    has_pad_zeros = False
    if pad_mask is not None:
        pm_bool = pad_mask.reshape(bsz, seq_len).to(device=query.device) > 0.5
        has_pad_zeros = not bool(pm_bool.all())
    inhomogeneous_steps = bsz > 1 and not bool((state.step_count == state.step_count[0]).all())

    if has_pad_zeros or inhomogeneous_steps:
        row_outs_p: list[torch.Tensor] = []
        row_states_p: list[HSSDCacheState] = []
        u_tot_p = query.new_zeros(num_pages, dtype=acc_dt)
        cnt_tot_p = query.new_zeros((), dtype=acc_dt)
        for b in range(bsz):
            st_b = state[b : b + 1]
            sl_b = slice(b, b + 1)
            idx_b = pm_bool[b].nonzero(as_tuple=False).squeeze(-1) if pm_bool is not None else None
            if idx_b is not None and idx_b.numel() == 0:
                row_outs_p.append(query.new_zeros(1, seq_len, hv, dv, dtype=init_dtype))
                if output_final_state:
                    row_states_p.append(_cast_state_dtype(st_b, state_dtype, acc_dt))
                continue
            tsl: slice | torch.Tensor | None = (
                idx_b if (idx_b is not None and idx_b.numel() < seq_len) else None
            )
            q_b = query[sl_b] if tsl is None else query[sl_b].index_select(1, idx_b)  # type: ignore[arg-type]
            k_b = key[sl_b] if tsl is None else key[sl_b].index_select(1, idx_b)  # type: ignore[arg-type]
            v_b = value[sl_b] if tsl is None else value[sl_b].index_select(1, idx_b)  # type: ignore[arg-type]
            gam_b = _slice_gamma_input(gamma, batch_sl=sl_b, time_sl=tsl)
            bg_b = b_gate[sl_b] if tsl is None else b_gate[sl_b].index_select(1, idx_b)  # type: ignore[arg-type]
            wg_b = w_gate[sl_b] if tsl is None else w_gate[sl_b].index_select(1, idx_b)  # type: ignore[arg-type]
            th_b = theta[sl_b] if tsl is None else theta[sl_b].index_select(1, idx_b)  # type: ignore[arg-type]
            sb_b = surprise_bias[sl_b] if tsl is None else surprise_bias[sl_b].index_select(1, idx_b)  # type: ignore[arg-type]
            wl_b = write_logits[sl_b] if tsl is None else write_logits[sl_b].index_select(1, idx_b)  # type: ignore[arg-type]
            rl_b = read_logits[sl_b] if tsl is None else read_logits[sl_b].index_select(1, idx_b)  # type: ignore[arg-type]
            qs_b = q_swa[sl_b] if tsl is None else q_swa[sl_b].index_select(1, idx_b)  # type: ignore[arg-type]
            ks_b = k_swa[sl_b] if tsl is None else k_swa[sl_b].index_select(1, idx_b)  # type: ignore[arg-type]
            vs_b = v_swa[sl_b] if tsl is None else v_swa[sl_b].index_select(1, idx_b)  # type: ignore[arg-type]
            wm_b = None
            if write_mask is not None:
                wm_row = write_mask.reshape(bsz, seq_len)[sl_b]
                wm_b = wm_row if tsl is None else wm_row.index_select(1, idx_b)  # type: ignore[arg-type]
            go_b = None
            if g_override is not None:
                go_b = g_override[sl_b] if tsl is None else g_override[sl_b].index_select(1, idx_b)  # type: ignore[arg-type]

            o_valid, st_next_b, u_b, c_b = _recurrent_hssd_raw(
                q_b,
                k_b,
                v_b,
                gam_b,
                bg_b,
                wg_b,
                th_b,
                sb_b,
                surprise_slope,
                surprise_thresh,
                wl_b,
                rl_b,
                qs_b,
                ks_b,
                vs_b,
                boltzmann_gate=boltzmann_gate,
                chunk_size=chunk_size,
                window_chunks=window_chunks,
                num_pages=num_pages,
                top_k_pages=top_k_pages,
                initial_state=st_b,
                output_final_state=output_final_state,
                write_mask=wm_b,
                pad_mask=None,
                cu_seqlens=None,
                seq_idx=None,
                g_override=go_b,
                use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
                scale=scale,
                state_dtype=state_dtype,
                use_swa=use_swa,
                use_phase=use_phase,
                use_surprise=use_surprise,
                surprise_mode=surprise_mode,
            )
            if tsl is None:
                row_outs_p.append(o_valid)
            else:
                assert idx_b is not None
                n_val = int(idx_b.numel())
                idx_exp = idx_b.view(1, n_val, 1, 1).expand(1, n_val, hv, dv)
                out_row = o_valid.new_zeros(1, seq_len, hv, dv).scatter(1, idx_exp, o_valid)
                row_outs_p.append(out_row)
            if output_final_state:
                assert st_next_b is not None
                row_states_p.append(st_next_b)
            u_tot_p = u_tot_p + u_b
            cnt_tot_p = cnt_tot_p + c_b

        out_p = row_outs_p[0] if len(row_outs_p) == 1 else torch.cat(row_outs_p, dim=0)
        st_p = (
            (row_states_p[0] if len(row_states_p) == 1 else torch.cat(row_states_p, dim=0))
            if output_final_state
            else None
        )
        return out_p, st_p, u_tot_p, cnt_tot_p

    outs: list[torch.Tensor] = []
    u_total = query.new_zeros(num_pages, dtype=acc_dt)
    tok_total = query.new_zeros((), dtype=acc_dt)

    for i in range(seq_len):
        sl_i = slice(i, i + 1)
        wm_i = None if write_mask is None else write_mask[:, sl_i]
        go_i = None if g_override is None else g_override[:, sl_i]
        o_i, state, u_i, cnt_i = recurrent_hssd_step(
            query[:, sl_i],
            key[:, sl_i],
            value[:, sl_i],
            _slice_gamma_input(gamma, time_sl=sl_i),
            b_gate[:, sl_i],
            w_gate[:, sl_i],
            theta[:, sl_i],
            surprise_bias[:, sl_i],
            surprise_slope,
            surprise_thresh,
            write_logits[:, sl_i],
            read_logits[:, sl_i],
            q_swa[:, sl_i],
            k_swa[:, sl_i],
            v_swa[:, sl_i],
            state,
            boltzmann_gate=boltzmann_gate,
            top_k_pages=top_k_pages,
            write_mask=wm_i,
            pad_mask=None,
            g_override=go_i,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            scale=scale,
            state_dtype="auto",
            use_swa=use_swa,
            use_phase=use_phase,
            use_surprise=use_surprise,
            surprise_mode=surprise_mode,
            return_duty_sum=True,
        )
        outs.append(o_i)
        u_total = u_total + u_i
        tok_total = tok_total + cnt_i

    out = torch.cat(outs, dim=1) if outs else query.new_zeros(bsz, 0, hv, dv)
    final_st = _cast_state_dtype(state, state_dtype, acc_dt) if output_final_state else None
    return out, final_st, u_total, tok_total


def recurrent_hssd(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    gamma: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
    b_gate: torch.Tensor,
    w_gate: torch.Tensor,
    theta: torch.Tensor,
    surprise_bias: torch.Tensor,
    surprise_slope: torch.Tensor,
    surprise_thresh: torch.Tensor,
    write_logits: torch.Tensor,
    read_logits: torch.Tensor,
    q_swa: torch.Tensor,
    k_swa: torch.Tensor,
    v_swa: torch.Tensor,
    *,
    boltzmann_gate: PeakBoltzmannGate | None = None,
    chunk_size: int = 64,
    window_chunks: int = 8,
    num_pages: int = 8,
    top_k_pages: int = 2,
    initial_state: HSSDCacheState | torch.Tensor | None = None,
    output_final_state: bool = False,
    write_mask: torch.Tensor | None = None,
    pad_mask: torch.Tensor | None = None,
    cu_seqlens: Any = None,
    seq_idx: torch.Tensor | None = None,
    g_override: torch.Tensor | None = None,
    use_qk_l2norm_in_kernel: bool = False,
    scale: float | None = None,
    state_dtype: str | torch.dtype = "auto",
    use_swa: bool = True,
    use_phase: bool = True,
    use_surprise: bool = True,
    surprise_mode: str = "routed",
    return_aux: bool = False,
) -> tuple[torch.Tensor, HSSDCacheState | None] | tuple[torch.Tensor, HSSDCacheState | None, torch.Tensor]:
    """Token-by-token sequential HSSD v3.1 reference execution over `T` steps via `recurrent_hssd_step`."""
    out, final_st, u_total, tok_total = _recurrent_hssd_raw(
        query,
        key,
        value,
        gamma,
        b_gate,
        w_gate,
        theta,
        surprise_bias,
        surprise_slope,
        surprise_thresh,
        write_logits,
        read_logits,
        q_swa,
        k_swa,
        v_swa,
        boltzmann_gate=boltzmann_gate,
        chunk_size=chunk_size,
        window_chunks=window_chunks,
        num_pages=num_pages,
        top_k_pages=top_k_pages,
        initial_state=initial_state,
        output_final_state=output_final_state,
        write_mask=write_mask,
        pad_mask=pad_mask,
        cu_seqlens=cu_seqlens,
        seq_idx=seq_idx,
        g_override=g_override,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        scale=scale,
        state_dtype=state_dtype,
        use_swa=use_swa,
        use_phase=use_phase,
        use_surprise=use_surprise,
        surprise_mode=surprise_mode,
    )
    if return_aux:
        return out, final_st, u_total / tok_total.clamp_min(1.0)
    return out, final_st


def _chunk_slices_for_span(step_offset: int, seq_len: int, chunk_size: int) -> list[tuple[int, int]]:
    """Partition local `[0, seq_len)` along global chunk boundaries `g // chunk_size`."""
    slices: list[tuple[int, int]] = []
    cur = 0
    while cur < seq_len:
        g_cur = step_offset + cur
        rem = chunk_size - (g_cur % chunk_size)
        nxt = min(seq_len, cur + rem)
        slices.append((cur, nxt))
        cur = nxt
    return slices


def _run_chunk_group(
    chunk_bounds: list[tuple[int, int]],
    *,
    step_offset: int,
    swa_base_global: int,
    chunk_size: int,
    window_chunks: int,
    num_pages: int,
    top_k_pages: int,
    use_swa: bool,
    use_surprise: bool,
    surprise_mode: str,
    q_hat: torch.Tensor,
    k_hat: torch.Tensor,
    v: torch.Tensor,
    gamma_page_scale: torch.Tensor | None,
    gamma_chan_seg: torch.Tensor,
    bf: torch.Tensor,
    wf: torch.Tensor,
    sb: torch.Tensor,
    slp: torch.Tensor,
    thr: torch.Tensor,
    wl: torch.Tensor,
    rl: torch.Tensor,
    qs: torch.Tensor,
    all_ks: torch.Tensor,
    all_vs: torch.Tensor,
    all_valid_swa: torch.Tensor,
    wm: torch.Tensor | None,
    go: torch.Tensor | None,
    gate_log_kappa: torch.Tensor,
    gate_tau: torch.Tensor,
    page_states: torch.Tensor,
    boundary_ring: tuple[torch.Tensor, ...],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, tuple[torch.Tensor, ...]]:
    """Execute a contiguous group of global-aligned chunks (pure functional for checkpointing)."""
    dk = k_hat.shape[-1]
    hv = k_hat.shape[2]
    outs: list[torch.Tensor] = []
    u_acc = q_hat.new_zeros(wl.shape[-1])
    tok_acc = q_hat.new_zeros(())

    kappa = gate_log_kappa.exp()
    kappa_b = kappa[0] if kappa.numel() == 1 else kappa.view(1, hv, 1)
    tau_b = gate_tau[0] if gate_tau.numel() == 1 else gate_tau.view(1, hv, 1)

    p_cur = page_states
    ring_cur = boundary_ring

    for l0, l1 in chunk_bounds:
        g0 = step_offset + l0
        g1 = step_offset + l1

        q_c = q_hat[:, l0:l1]
        k_c = k_hat[:, l0:l1]
        v_c = v[:, l0:l1]
        rl_c = rl[:, l0:l1]
        wl_c = wl[:, l0:l1]
        sb_c = sb[:, l0:l1]
        wm_c = None if wm is None else wm[:, l0:l1]
        go_c = None if go is None else go[:, l0:l1]

        # 1. Far-horizon readout strictly from S_boundary[max(0, n - W)] == ring_cur[0]
        o_far = _far_readout(q_c, ring_cur[0], rl_c, top_k_pages)

        # 2. Exact causal local SWA over I_loc(t) = [far_cutoff + 1, t] (0-based [far_cutoff, g1))
        if use_swa:
            _n_idx, _k_far, far_cutoff = disjoint_horizon_bounds(g0 + 1, chunk_size, window_chunks)
            s0 = far_cutoff - swa_base_global
            s1 = g1 - swa_base_global
            k_win = all_ks[:, s0:s1].transpose(1, 2)               # [B, H_v, win_len, d_k]
            v_win = all_vs[:, s0:s1].transpose(1, 2)               # [B, H_v, win_len, d_v]
            val_keys = all_valid_swa[:, s0:s1]                     # [B, win_len]

            q_pos = torch.arange(g0, g1, device=q_c.device)[:, None]
            k_pos = torch.arange(far_cutoff, g1, device=q_c.device)[None, :]
            valid_mask = (k_pos <= q_pos)[None, None, :, :] & val_keys[:, None, None, :]

            q_loc = qs[:, l0:l1].transpose(1, 2)                   # [B, H_v, c_len, d_k]
            scores = (q_loc @ k_win.transpose(-1, -2)) * (dk ** -0.5)
            has_valid = valid_mask.any(dim=-1)
            safe_vm = torch.where(has_valid.unsqueeze(-1), valid_mask, torch.ones_like(valid_mask))
            masked_sc = scores.masked_fill(~safe_vm, float("-inf"))

            probs = torch.softmax(masked_sc, dim=-1)
            probs = torch.where(has_valid.unsqueeze(-1), probs, torch.zeros_like(probs))
            o_loc = (probs @ v_win).transpose(1, 2)

            counts = valid_mask.expand_as(scores).sum(dim=-1)
            has_v_sc = counts > 0
            n_eff = counts.clamp_min(1).to(scores.dtype)
            lse = torch.logsumexp(masked_sc, dim=-1)
            max_sc = masked_sc.max(dim=-1).values
            stat = 0.5 * (lse - torch.log(n_eff)) + 0.5 * (max_sc - torch.sqrt(2.0 * torch.log(n_eff.clamp_min(2.0))))
            p_hit = torch.sigmoid(kappa_b * (stat - tau_b))
            p_hit = torch.where(has_v_sc, p_hit, torch.zeros_like(p_hit)).transpose(1, 2).unsqueeze(-1)
            outs.append(p_hit * o_loc + (1.0 - p_hit) * o_far)
        else:
            outs.append(o_far)

        # 3. Parallel routed chunk-start root-surprise (against ring_cur[-1]) + WY multi-page state update
        g_page, u_sum, cnt_c, _top_idx, _g_act = _compute_root_surprise_and_routing(
            k_c,
            v_c,
            ring_cur[-1],
            sb_c,
            slp,
            thr,
            wl_c,
            top_k_pages,
            wm_c,
            go_c,
            use_surprise=use_surprise,
            surprise_mode=surprise_mode,
        )
        u_acc = u_acc + u_sum
        tok_acc = tok_acc + cnt_c

        # Expand factored gamma across M pages ONLY inside this single chunk [l0:l1] (Bug 3B)
        gamma_c = _materialize_chunk_gamma(
            gamma_page_scale, gamma_chan_seg[:, l0:l1], num_pages=num_pages, dk=dk
        )
        sqrt_b = _safe_sqrt(bf[:, l0:l1])
        sqrt_w = _safe_sqrt(wf[:, l0:l1])
        k_eff = (g_page.unsqueeze(-1) * (sqrt_b * k_c).unsqueeze(2)).permute(0, 2, 3, 1, 4)
        v_eff = (g_page.unsqueeze(-1) * (sqrt_w * v_c).unsqueeze(2)).permute(0, 2, 3, 1, 4)
        log_a = (-(g_page * g_page).unsqueeze(-1) * gamma_c).permute(0, 2, 3, 1, 4)
        p_cur = _wy_kda_state_update(p_cur, k_eff, v_eff, log_a)

        if g1 % chunk_size == 0:
            ring_cur = (*ring_cur[1:], p_cur)

    seg_out = outs[0] if len(outs) == 1 else torch.cat(outs, dim=1)
    return seg_out, p_cur, u_acc, tok_acc, ring_cur


def _chunk_hssd_raw(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    gamma: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
    b_gate: torch.Tensor,
    w_gate: torch.Tensor,
    theta: torch.Tensor,
    surprise_bias: torch.Tensor,
    surprise_slope: torch.Tensor,
    surprise_thresh: torch.Tensor,
    write_logits: torch.Tensor,
    read_logits: torch.Tensor,
    q_swa: torch.Tensor,
    k_swa: torch.Tensor,
    v_swa: torch.Tensor,
    *,
    boltzmann_gate: PeakBoltzmannGate | None,
    chunk_size: int,
    window_chunks: int,
    num_pages: int,
    top_k_pages: int,
    segment_chunks: int,
    checkpoint_segments: bool,
    initial_state: HSSDCacheState | torch.Tensor | None,
    output_final_state: bool,
    write_mask: torch.Tensor | None,
    pad_mask: torch.Tensor | None,
    cu_seqlens: Any,
    seq_idx: torch.Tensor | None,
    g_override: torch.Tensor | None,
    use_qk_l2norm_in_kernel: bool,
    scale: float | None,
    state_dtype: str | torch.dtype,
    use_swa: bool,
    use_phase: bool,
    use_surprise: bool,
    surprise_mode: str,
) -> tuple[torch.Tensor, HSSDCacheState | None, torch.Tensor, torch.Tensor]:
    init_dtype = query.dtype
    bsz, seq_len, hv, dv = value.shape
    dk = key.shape[-1]
    acc_dt = torch.float64 if init_dtype == torch.float64 else torch.float32

    doc_spans = _validate_packed_boundaries(
        bsz=bsz, seq_len=seq_len, initial_state=initial_state, cu_seqlens=cu_seqlens, seq_idx=seq_idx
    )
    if doc_spans is not None:
        row_outs: list[torch.Tensor] = []
        row_states: list[HSSDCacheState] = []
        u_tot = query.new_zeros(num_pages, dtype=acc_dt)
        cnt_tot = query.new_zeros((), dtype=acc_dt)
        for b, spans_b in enumerate(doc_spans):
            sl_b = slice(b, b + 1)
            doc_outs: list[torch.Tensor] = []
            last_st: HSSDCacheState | None = None
            for s0, s1 in spans_b:
                sl_t = slice(s0, s1)
                o_d, st_d, u_d, c_d = _chunk_hssd_raw(
                    query[sl_b, sl_t],
                    key[sl_b, sl_t],
                    value[sl_b, sl_t],
                    _slice_gamma_input(gamma, batch_sl=sl_b, time_sl=sl_t),
                    b_gate[sl_b, sl_t],
                    w_gate[sl_b, sl_t],
                    theta[sl_b, sl_t],
                    surprise_bias[sl_b, sl_t],
                    surprise_slope,
                    surprise_thresh,
                    write_logits[sl_b, sl_t],
                    read_logits[sl_b, sl_t],
                    q_swa[sl_b, sl_t],
                    k_swa[sl_b, sl_t],
                    v_swa[sl_b, sl_t],
                    boltzmann_gate=boltzmann_gate,
                    chunk_size=chunk_size,
                    window_chunks=window_chunks,
                    num_pages=num_pages,
                    top_k_pages=top_k_pages,
                    segment_chunks=segment_chunks,
                    checkpoint_segments=checkpoint_segments,
                    initial_state=None,
                    output_final_state=output_final_state,
                    write_mask=None if write_mask is None else write_mask[sl_b, sl_t],
                    pad_mask=None if pad_mask is None else pad_mask[sl_b, sl_t],
                    cu_seqlens=None,
                    seq_idx=None,
                    g_override=None if g_override is None else g_override[sl_b, sl_t],
                    use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
                    scale=scale,
                    state_dtype=state_dtype,
                    use_swa=use_swa,
                    use_phase=use_phase,
                    use_surprise=use_surprise,
                    surprise_mode=surprise_mode,
                )
                doc_outs.append(o_d)
                last_st = st_d
                u_tot = u_tot + u_d
                cnt_tot = cnt_tot + c_d
            row_outs.append(
                torch.cat(doc_outs, dim=1)
                if doc_outs
                else query.new_zeros(1, 0, hv, dv, dtype=init_dtype)
            )
            if output_final_state:
                if last_st is None:
                    last_st = _cast_state_dtype(
                        HSSDCacheState.zeros(
                            1, num_pages, hv, dk, dv,
                            chunk_size=chunk_size, window_chunks=window_chunks,
                            device=query.device, dtype=acc_dt,
                        ),
                        state_dtype,
                        acc_dt,
                    )
                row_states.append(last_st)
        out_all = row_outs[0] if len(row_outs) == 1 else torch.cat(row_outs, dim=0)
        st_all = (
            (row_states[0] if len(row_states) == 1 else torch.cat(row_states, dim=0))
            if output_final_state
            else None
        )
        return out_all, st_all, u_tot, cnt_tot

    st = _coerce_init_state(
        initial_state,
        bsz=bsz,
        num_pages=num_pages,
        hv=hv,
        dk=dk,
        dv=dv,
        chunk_size=chunk_size,
        window_chunks=window_chunks,
        device=query.device,
        dtype=acc_dt,
    )

    pm_bool: torch.Tensor | None = None
    has_pad_zeros = False
    if pad_mask is not None:
        pm_bool = pad_mask.reshape(bsz, seq_len).to(device=query.device) > 0.5
        has_pad_zeros = not bool(pm_bool.all())
    inhomogeneous_steps = bsz > 1 and not bool((st.step_count == st.step_count[0]).all())

    if has_pad_zeros or inhomogeneous_steps:
        row_outs_p: list[torch.Tensor] = []
        row_states_p: list[HSSDCacheState] = []
        u_tot_p = query.new_zeros(num_pages, dtype=acc_dt)
        cnt_tot_p = query.new_zeros((), dtype=acc_dt)
        for b in range(bsz):
            st_b = st[b : b + 1]
            sl_b = slice(b, b + 1)
            idx_b = pm_bool[b].nonzero(as_tuple=False).squeeze(-1) if pm_bool is not None else None
            if idx_b is not None and idx_b.numel() == 0:
                row_outs_p.append(query.new_zeros(1, seq_len, hv, dv, dtype=init_dtype))
                if output_final_state:
                    row_states_p.append(_cast_state_dtype(st_b, state_dtype, acc_dt))
                continue
            tsl: slice | torch.Tensor | None = (
                idx_b if (idx_b is not None and idx_b.numel() < seq_len) else None
            )
            q_b = query[sl_b] if tsl is None else query[sl_b].index_select(1, idx_b)  # type: ignore[arg-type]
            k_b = key[sl_b] if tsl is None else key[sl_b].index_select(1, idx_b)  # type: ignore[arg-type]
            v_b = value[sl_b] if tsl is None else value[sl_b].index_select(1, idx_b)  # type: ignore[arg-type]
            gam_b = _slice_gamma_input(gamma, batch_sl=sl_b, time_sl=tsl)
            bg_b = b_gate[sl_b] if tsl is None else b_gate[sl_b].index_select(1, idx_b)  # type: ignore[arg-type]
            wg_b = w_gate[sl_b] if tsl is None else w_gate[sl_b].index_select(1, idx_b)  # type: ignore[arg-type]
            th_b = theta[sl_b] if tsl is None else theta[sl_b].index_select(1, idx_b)  # type: ignore[arg-type]
            sb_b = surprise_bias[sl_b] if tsl is None else surprise_bias[sl_b].index_select(1, idx_b)  # type: ignore[arg-type]
            wl_b = write_logits[sl_b] if tsl is None else write_logits[sl_b].index_select(1, idx_b)  # type: ignore[arg-type]
            rl_b = read_logits[sl_b] if tsl is None else read_logits[sl_b].index_select(1, idx_b)  # type: ignore[arg-type]
            qs_b = q_swa[sl_b] if tsl is None else q_swa[sl_b].index_select(1, idx_b)  # type: ignore[arg-type]
            ks_b = k_swa[sl_b] if tsl is None else k_swa[sl_b].index_select(1, idx_b)  # type: ignore[arg-type]
            vs_b = v_swa[sl_b] if tsl is None else v_swa[sl_b].index_select(1, idx_b)  # type: ignore[arg-type]
            wm_b = None
            if write_mask is not None:
                wm_row = write_mask.reshape(bsz, seq_len)[sl_b]
                wm_b = wm_row if tsl is None else wm_row.index_select(1, idx_b)  # type: ignore[arg-type]
            go_b = None
            if g_override is not None:
                go_b = g_override[sl_b] if tsl is None else g_override[sl_b].index_select(1, idx_b)  # type: ignore[arg-type]

            o_valid, st_next_b, u_b, c_b = _chunk_hssd_raw(
                q_b,
                k_b,
                v_b,
                gam_b,
                bg_b,
                wg_b,
                th_b,
                sb_b,
                surprise_slope,
                surprise_thresh,
                wl_b,
                rl_b,
                qs_b,
                ks_b,
                vs_b,
                boltzmann_gate=boltzmann_gate,
                chunk_size=chunk_size,
                window_chunks=window_chunks,
                num_pages=num_pages,
                top_k_pages=top_k_pages,
                segment_chunks=segment_chunks,
                checkpoint_segments=checkpoint_segments,
                initial_state=st_b,
                output_final_state=output_final_state,
                write_mask=wm_b,
                pad_mask=None,
                cu_seqlens=None,
                seq_idx=None,
                g_override=go_b,
                use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
                scale=scale,
                state_dtype=state_dtype,
                use_swa=use_swa,
                use_phase=use_phase,
                use_surprise=use_surprise,
                surprise_mode=surprise_mode,
            )
            if tsl is None:
                row_outs_p.append(o_valid)
            else:
                assert idx_b is not None
                n_val = int(idx_b.numel())
                idx_exp = idx_b.view(1, n_val, 1, 1).expand(1, n_val, hv, dv)
                out_row = o_valid.new_zeros(1, seq_len, hv, dv).scatter(1, idx_exp, o_valid)
                row_outs_p.append(out_row)
            if output_final_state:
                assert st_next_b is not None
                row_states_p.append(st_next_b)
            u_tot_p = u_tot_p + u_b
            cnt_tot_p = cnt_tot_p + c_b

        out_p = row_outs_p[0] if len(row_outs_p) == 1 else torch.cat(row_outs_p, dim=0)
        st_p = (
            (row_states_p[0] if len(row_states_p) == 1 else torch.cat(row_states_p, dim=0))
            if output_final_state
            else None
        )
        return out_p, st_p, u_tot_p, cnt_tot_p

    if seq_len == 0:
        empty_out = query.new_zeros(bsz, 0, hv, dv, dtype=init_dtype)
        final_0 = _cast_state_dtype(st, state_dtype, acc_dt) if output_final_state else None
        return empty_out, final_0, query.new_zeros(num_pages, dtype=acc_dt), query.new_zeros((), dtype=acc_dt)

    if seq_len == 1:
        o_step, st_next, u_sum, cnt_sum = recurrent_hssd_step(
            query,
            key,
            value,
            gamma,
            b_gate,
            w_gate,
            theta,
            surprise_bias,
            surprise_slope,
            surprise_thresh,
            write_logits,
            read_logits,
            q_swa,
            k_swa,
            v_swa,
            st,
            boltzmann_gate=boltzmann_gate,
            top_k_pages=top_k_pages,
            write_mask=write_mask,
            pad_mask=None,
            g_override=g_override,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            scale=scale,
            state_dtype=state_dtype,
            use_swa=use_swa,
            use_phase=use_phase,
            use_surprise=use_surprise,
            surprise_mode=surprise_mode,
            return_duty_sum=True,
        )
        return o_step, (st_next if output_final_state else None), u_sum, cnt_sum

    q, k, v = _prepare_qkv_unscaled(query, key, value, use_qk_l2norm_in_kernel)
    ps_gamma, ch_gamma = _normalize_gamma_spec(
        gamma, num_pages=num_pages, dk=dk, acc_dt=acc_dt, device=q.device
    )
    bf = upcast(b_gate).to(acc_dt)
    wf = upcast(w_gate).to(acc_dt)
    sb = upcast(surprise_bias).to(acc_dt)
    slp = upcast(surprise_slope).to(acc_dt)
    thr = upcast(surprise_thresh).to(acc_dt)
    wl = upcast(write_logits).to(acc_dt)
    rl = upcast(read_logits).to(acc_dt)
    qs = upcast(q_swa).to(acc_dt)
    ks = upcast(k_swa).to(acc_dt)
    vs = upcast(v_swa).to(acc_dt)
    go = None if g_override is None else upcast(g_override).to(acc_dt)

    th64 = theta.to(dtype=torch.float64)
    if write_mask is not None:
        wm = write_mask.reshape(bsz, seq_len).to(device=q.device, dtype=acc_dt).clamp(0.0, 1.0)
        valid_cur = wm > 0.5
        th64 = th64 * wm.to(dtype=torch.float64).unsqueeze(-1).unsqueeze(-1)
    else:
        wm = None
        valid_cur = torch.ones(bsz, seq_len, device=q.device, dtype=torch.bool)
    if not use_phase:
        th64 = torch.zeros_like(th64)

    # Bug 4.3: Accumulate phase strictly in float64 across all chunks; cast only cos/sin in apply_2d_rotation
    theta_cum_all = torch.remainder(
        st.theta_cum.to(dtype=torch.float64).unsqueeze(1) + torch.cumsum(th64, dim=1),
        2.0 * math.pi,
    )
    sc = (dk ** -0.5) if scale is None else float(scale)
    q_hat = apply_2d_rotation(q, -theta_cum_all) if use_phase else q
    if sc != 1.0:
        q_hat = q_hat * sc
    k_hat = apply_2d_rotation(k, -theta_cum_all) if use_phase else k

    step_offset = int(st.step_count[0].item())
    max_win = st.max_swa_len
    _, _, swa_base_global = disjoint_horizon_bounds(step_offset + 1, chunk_size, window_chunks)
    if step_offset > swa_base_global:
        prior_idx = torch.arange(swa_base_global, step_offset, device=q.device) % max_win
        all_ks = torch.cat([st.swa_k_ring.index_select(1, prior_idx), ks], dim=1)
        all_vs = torch.cat([st.swa_v_ring.index_select(1, prior_idx), vs], dim=1)
        all_valid = torch.cat([st.swa_valid_ring.index_select(1, prior_idx), valid_cur], dim=1)
    else:
        all_ks = ks
        all_vs = vs
        all_valid = valid_cur

    if boltzmann_gate is not None:
        gate_log_kappa = upcast(boltzmann_gate.log_kappa).to(acc_dt)
        gate_tau = upcast(boltzmann_gate.tau).to(acc_dt)
    else:
        gate_log_kappa = q.new_full((1,), math.log(2.0), dtype=acc_dt)
        gate_tau = q.new_full((1,), 2.2, dtype=acc_dt)

    all_chunks = _chunk_slices_for_span(step_offset, seq_len, chunk_size)
    seg_step = max(1, int(segment_chunks))
    need_grad = any(
        isinstance(t, torch.Tensor) and t.requires_grad
        for t in (
            q_hat,
            k_hat,
            v,
            ps_gamma,
            ch_gamma,
            bf,
            wf,
            sb,
            slp,
            thr,
            wl,
            rl,
            qs,
            all_ks,
            all_vs,
            gate_log_kappa,
            gate_tau,
            st.page_states,
            go,
        )
    )
    use_ckpt = bool(checkpoint_segments and torch.is_grad_enabled() and need_grad and len(all_chunks) > seg_step)

    p_cur = st.page_states
    ring_cur = st.boundary_ring
    seg_outs: list[torch.Tensor] = []
    u_total = q.new_zeros(num_pages, dtype=acc_dt)
    tok_total = q.new_zeros((), dtype=acc_dt)

    for c_idx in range(0, len(all_chunks), seg_step):
        c_group = all_chunks[c_idx : c_idx + seg_step]
        l_start, l_end = c_group[0][0], c_group[-1][1]
        g_first = step_offset + l_start
        g_last = step_offset + l_end
        _, _, grp_far_cut = disjoint_horizon_bounds(g_first + 1, chunk_size, window_chunks)
        s_lo = grp_far_cut - swa_base_global
        s_hi = g_last - swa_base_global
        rel_bounds = [(a - l_start, b - l_start) for a, b in c_group]

        q_seg = q_hat[:, l_start:l_end]
        k_seg = k_hat[:, l_start:l_end]
        v_seg = v[:, l_start:l_end]
        ch_seg = ch_gamma[:, l_start:l_end]
        bf_seg = bf[:, l_start:l_end]
        wf_seg = wf[:, l_start:l_end]
        sb_seg = sb[:, l_start:l_end]
        wl_seg = wl[:, l_start:l_end]
        rl_seg = rl[:, l_start:l_end]
        qs_seg = qs[:, l_start:l_end]
        ks_seg = all_ks[:, s_lo:s_hi]
        vs_seg = all_vs[:, s_lo:s_hi]
        val_seg = all_valid[:, s_lo:s_hi]
        wm_seg = None if wm is None else wm[:, l_start:l_end]
        go_seg = None if go is None else go[:, l_start:l_end]

        if use_ckpt:
            ps_arg = ps_gamma if ps_gamma is not None else q.new_empty(0)
            has_ps = ps_gamma is not None

            def _ckpt_fn(
                qc: torch.Tensor,
                kc: torch.Tensor,
                vc: torch.Tensor,
                psc: torch.Tensor,
                chc: torch.Tensor,
                bfc: torch.Tensor,
                wfc: torch.Tensor,
                sbc: torch.Tensor,
                slpc: torch.Tensor,
                thrc: torch.Tensor,
                wlc: torch.Tensor,
                rlc: torch.Tensor,
                qsc: torch.Tensor,
                ksc: torch.Tensor,
                vsc: torch.Tensor,
                glk: torch.Tensor,
                gtau: torch.Tensor,
                pst: torch.Tensor,
                *ring_tensors: torch.Tensor,
                _rb: list[tuple[int, int]] = rel_bounds,
                _goff: int = g_first,
                _sbase: int = grp_far_cut,
                _val: torch.Tensor = val_seg,
                _wm: torch.Tensor | None = wm_seg,
                _go: torch.Tensor | None = go_seg,
                _hps: bool = has_ps,
            ):
                o_g, p_n, u_g, tk_g, r_n = _run_chunk_group(
                    _rb,
                    step_offset=_goff,
                    swa_base_global=_sbase,
                    chunk_size=chunk_size,
                    window_chunks=window_chunks,
                    num_pages=num_pages,
                    top_k_pages=top_k_pages,
                    use_swa=use_swa,
                    use_surprise=use_surprise,
                    surprise_mode=surprise_mode,
                    q_hat=qc,
                    k_hat=kc,
                    v=vc,
                    gamma_page_scale=(psc if _hps else None),
                    gamma_chan_seg=chc,
                    bf=bfc,
                    wf=wfc,
                    sb=sbc,
                    slp=slpc,
                    thr=thrc,
                    wl=wlc,
                    rl=rlc,
                    qs=qsc,
                    all_ks=ksc,
                    all_vs=vsc,
                    all_valid_swa=_val,
                    wm=_wm,
                    go=_go,
                    gate_log_kappa=glk,
                    gate_tau=gtau,
                    page_states=pst,
                    boundary_ring=ring_tensors,
                )
                return (o_g, p_n, u_g, tk_g, *r_n)

            ckpt_out = checkpoint(
                _ckpt_fn,
                q_seg,
                k_seg,
                v_seg,
                ps_arg,
                ch_seg,
                bf_seg,
                wf_seg,
                sb_seg,
                slp,
                thr,
                wl_seg,
                rl_seg,
                qs_seg,
                ks_seg,
                vs_seg,
                gate_log_kappa,
                gate_tau,
                p_cur,
                *ring_cur,
                use_reentrant=False,
            )
            o_grp, p_cur, u_grp, tok_grp = ckpt_out[:4]
            ring_cur = tuple(ckpt_out[4:])
        else:
            o_grp, p_cur, u_grp, tok_grp, ring_cur = _run_chunk_group(
                rel_bounds,
                step_offset=g_first,
                swa_base_global=grp_far_cut,
                chunk_size=chunk_size,
                window_chunks=window_chunks,
                num_pages=num_pages,
                top_k_pages=top_k_pages,
                use_swa=use_swa,
                use_surprise=use_surprise,
                surprise_mode=surprise_mode,
                q_hat=q_seg,
                k_hat=k_seg,
                v=v_seg,
                gamma_page_scale=ps_gamma,
                gamma_chan_seg=ch_seg,
                bf=bf_seg,
                wf=wf_seg,
                sb=sb_seg,
                slp=slp,
                thr=thr,
                wl=wl_seg,
                rl=rl_seg,
                qs=qs_seg,
                all_ks=ks_seg,
                all_vs=vs_seg,
                all_valid_swa=val_seg,
                wm=wm_seg,
                go=go_seg,
                gate_log_kappa=gate_log_kappa,
                gate_tau=gate_tau,
                page_states=p_cur,
                boundary_ring=ring_cur,
            )
        seg_outs.append(o_grp)
        u_total = u_total + u_grp
        tok_total = tok_total + tok_grp

    out = (seg_outs[0] if len(seg_outs) == 1 else torch.cat(seg_outs, dim=1)).to(init_dtype)
    final_state: HSSDCacheState | None = None
    if output_final_state:
        end_step = step_offset + seq_len
        keep_from = max(swa_base_global, end_step - max_win)
        rel_from = keep_from - swa_base_global
        ring_slots = torch.arange(keep_from, end_step, device=q.device) % max_win
        new_k_ring = _compact(st.swa_k_ring.index_copy(1, ring_slots, all_ks[:, rel_from:]))
        new_v_ring = _compact(st.swa_v_ring.index_copy(1, ring_slots, all_vs[:, rel_from:]))
        new_val_ring = _compact(st.swa_valid_ring.index_copy(1, ring_slots, all_valid[:, rel_from:]))
        # Bug 3A: Explicitly clone & compact terminal theta_cum so [B, T, H_v, d_k // 2] is freed!
        term_theta = theta_cum_all[:, -1].clone().contiguous()
        final_state = _cast_state_dtype(
            HSSDCacheState(
                step_count=st.step_count + seq_len,
                chunk_size=chunk_size,
                window_chunks=window_chunks,
                page_states=_compact(p_cur),
                theta_cum=term_theta,
                boundary_ring=ring_cur,
                swa_k_ring=new_k_ring,
                swa_v_ring=new_v_ring,
                swa_valid_ring=new_val_ring,
            ),
            state_dtype,
            acc_dt,
        )

    return out, final_state, u_total, tok_total


def chunk_hssd(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    gamma: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
    b_gate: torch.Tensor,
    w_gate: torch.Tensor,
    theta: torch.Tensor,
    surprise_bias: torch.Tensor,
    surprise_slope: torch.Tensor,
    surprise_thresh: torch.Tensor,
    write_logits: torch.Tensor,
    read_logits: torch.Tensor,
    q_swa: torch.Tensor,
    k_swa: torch.Tensor,
    v_swa: torch.Tensor,
    *,
    boltzmann_gate: PeakBoltzmannGate | None = None,
    chunk_size: int = 64,
    window_chunks: int = 8,
    num_pages: int = 8,
    top_k_pages: int = 2,
    segment_chunks: int = 4,
    checkpoint_segments: bool = True,
    initial_state: HSSDCacheState | torch.Tensor | None = None,
    output_final_state: bool = False,
    write_mask: torch.Tensor | None = None,
    pad_mask: torch.Tensor | None = None,
    cu_seqlens: Any = None,
    seq_idx: torch.Tensor | None = None,
    g_override: torch.Tensor | None = None,
    use_qk_l2norm_in_kernel: bool = False,
    scale: float | None = None,
    state_dtype: str | torch.dtype = "auto",
    use_swa: bool = True,
    use_phase: bool = True,
    use_surprise: bool = True,
    surprise_mode: str = "routed",
    return_aux: bool = False,
) -> tuple[torch.Tensor, HSSDCacheState | None] | tuple[torch.Tensor, HSSDCacheState | None, torch.Tensor]:
    """Parallel chunkwise WY + disjoint local SWA HSSD v3.1 kernel (`O(T)` prefill & training)."""
    out, final_state, u_total, tok_total = _chunk_hssd_raw(
        query,
        key,
        value,
        gamma,
        b_gate,
        w_gate,
        theta,
        surprise_bias,
        surprise_slope,
        surprise_thresh,
        write_logits,
        read_logits,
        q_swa,
        k_swa,
        v_swa,
        boltzmann_gate=boltzmann_gate,
        chunk_size=chunk_size,
        window_chunks=window_chunks,
        num_pages=num_pages,
        top_k_pages=top_k_pages,
        segment_chunks=segment_chunks,
        checkpoint_segments=checkpoint_segments,
        initial_state=initial_state,
        output_final_state=output_final_state,
        write_mask=write_mask,
        pad_mask=pad_mask,
        cu_seqlens=cu_seqlens,
        seq_idx=seq_idx,
        g_override=g_override,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        scale=scale,
        state_dtype=state_dtype,
        use_swa=use_swa,
        use_phase=use_phase,
        use_surprise=use_surprise,
        surprise_mode=surprise_mode,
    )
    if return_aux:
        return out, final_state, u_total / tok_total.clamp_min(1.0)
    return out, final_state


# ============================================================================ adapter & retrofit

class HSSDAdapter(nn.Module):
    """Per-layer `HSSD v3.1` module attached to each linear-attention (`GatedDeltaNet`) layer."""

    def __init__(
        self,
        hidden_size: int,
        num_v_heads: int,
        head_k_dim: int,
        head_v_dim: int,
        cfg: HSSDConfig,
        layer_id: int = 0,
    ):
        super().__init__()
        cfg.validate()
        require(head_k_dim % 2 == 0, f"layer {layer_id}: HSSD requires even head_k_dim, got {head_k_dim}")
        self.cfg = cfg
        self.layer_id = int(layer_id)
        self.hidden_size = int(hidden_size)
        self.num_v_heads = int(num_v_heads)
        self.head_k_dim = int(head_k_dim)
        self.head_v_dim = int(head_v_dim)
        self.half_k_dim = self.head_k_dim // 2
        self.num_pages = int(cfg.num_pages)
        eff_rank = max(1, min(int(cfg.rank), self.hidden_size))
        self.eff_rank = eff_rank

        # Channel-wise Erase (over-relaxed (0, 2)), Write ([0, 1]), and paired Decay rates
        self.down = nn.Linear(self.hidden_size, eff_rank, bias=False)
        self.erase_up = nn.Linear(eff_rank, self.num_v_heads * self.head_k_dim, bias=False)
        self.write_up = nn.Linear(eff_rank, self.num_v_heads * self.head_v_dim, bias=False)
        self.decay_up = nn.Linear(eff_rank, self.num_v_heads * self.half_k_dim, bias=False)

        # Theorem 1 & Issue 5: Low-rank 2D phase rotation projector (pure linear down->up, zero-init up)
        if cfg.theta_low_rank:
            self.theta_down: nn.Linear | None = nn.Linear(self.hidden_size, eff_rank, bias=False)
            self.theta_up: nn.Linear | None = nn.Linear(
                eff_rank, self.num_v_heads * self.half_k_dim, bias=False
            )
            self.theta_proj: nn.Linear | None = None
        else:
            self.theta_down = None
            self.theta_up = None
            self.theta_proj = nn.Linear(self.hidden_size, self.num_v_heads * self.half_k_dim, bias=False)

        # Fixes H2, B3, B4: Smooth routed chunk-start root-surprise `s = g^2`
        self.surprise_proj = nn.Linear(self.hidden_size, self.num_v_heads, bias=True)
        self.log_surprise_slope = nn.Parameter(torch.full((self.num_v_heads,), math.log(cfg.surprise_slope_init)))
        self.surprise_thresh = nn.Parameter(torch.full((self.num_v_heads,), cfg.surprise_thresh_init))

        # Fix B2: Tiered M=8 multi-page write/read routers & log-spaced decay priors
        self.write_router = nn.Linear(self.hidden_size, self.num_pages, bias=True)
        self.read_router = nn.Linear(self.num_v_heads * self.head_k_dim, self.num_pages, bias=True)
        self.log_page_gamma = nn.Parameter(torch.zeros(self.num_pages))

        # Fixes B1 & B5: Local sliding-window Q/K/V adapters + PeakBoltzmannGate
        self.swa_q_up = nn.Linear(eff_rank, self.num_v_heads * self.head_k_dim, bias=False)
        self.swa_k_up = nn.Linear(eff_rank, self.num_v_heads * self.head_k_dim, bias=False)
        self.swa_v_up = nn.Linear(eff_rank, self.num_v_heads * self.head_v_dim, bias=False)
        self.swa_q_norm = nn.Parameter(torch.ones(self.head_k_dim))
        self.swa_k_norm = nn.Parameter(torch.ones(self.head_k_dim))
        self.boltzmann_gate = PeakBoltzmannGate(
            self.num_v_heads, kappa_init=cfg.kappa_init, tau_init=cfg.tau_init
        )

        mu_max_vals = cfg.resolved_mu_max()
        self.register_buffer("mu_max", torch.tensor(mu_max_vals, dtype=torch.float32), persistent=False)
        if self.num_pages == 1:
            gamma_prior = torch.tensor([cfg.decay_gamma_min], dtype=torch.float32)
        else:
            gamma_prior = torch.exp(
                torch.linspace(math.log(cfg.decay_gamma_min), math.log(cfg.decay_gamma_max), self.num_pages)
            )
        self.register_buffer("page_gamma_prior", gamma_prior, persistent=False)

        self._reset_parameters()
        self._pending_aux: list[torch.Tensor] = []
        self.last_page_duty: torch.Tensor | None = None
        self.last_dense_router_prob: torch.Tensor | None = None
        self.last_stats: dict[str, float] = {}

    def _reset_parameters(self) -> None:
        std_h = 1.0 / math.sqrt(self.hidden_size)
        std_q = 1.0 / math.sqrt(self.num_v_heads * self.head_k_dim)
        nn.init.normal_(self.down.weight, mean=0.0, std=std_h)
        nn.init.zeros_(self.erase_up.weight)
        nn.init.zeros_(self.write_up.weight)
        nn.init.zeros_(self.decay_up.weight)
        if self.theta_down is not None and self.theta_up is not None:
            nn.init.kaiming_uniform_(self.theta_down.weight, a=math.sqrt(5))
            nn.init.zeros_(self.theta_up.weight)
        elif self.theta_proj is not None:
            nn.init.zeros_(self.theta_proj.weight)
        nn.init.zeros_(self.surprise_proj.weight)
        nn.init.zeros_(self.surprise_proj.bias)
        nn.init.normal_(self.write_router.weight, mean=0.0, std=std_h)
        nn.init.zeros_(self.write_router.bias)
        nn.init.normal_(self.read_router.weight, mean=0.0, std=std_q)
        nn.init.zeros_(self.read_router.bias)
        nn.init.zeros_(self.swa_q_up.weight)
        nn.init.zeros_(self.swa_k_up.weight)
        nn.init.zeros_(self.swa_v_up.weight)

    def clear_aux(self) -> None:
        self._pending_aux.clear()

    def project_inputs(
        self,
        x_in: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        g_base: torch.Tensor,
        beta_base: torch.Tensor,
        *,
        use_qk_l2norm: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Compute HSSD gate, router, phase, and SWA projections with factored decay (`Bug 3B`)."""
        bsz, seq_len, _ = x_in.shape
        hv, dk, dv = self.num_v_heads, self.head_k_dim, self.head_v_dim
        half_dk = self.half_k_dim

        q_prep, k_prep, v_prep = _prepare_qkv_unscaled(query, key, value, use_qk_l2norm)
        xf = upcast(x_in)
        h = F.silu(F.linear(xf, upcast(self.down.weight)))

        cap_g = float(self.cfg.gate_logit_cap)
        cap_d = float(self.cfg.decay_log_cap)
        d_erase = (cap_g * torch.tanh(F.linear(h, upcast(self.erase_up.weight)))).view(bsz, seq_len, hv, dk)
        d_write = (cap_g * torch.tanh(F.linear(h, upcast(self.write_up.weight)))).view(bsz, seq_len, hv, dv)
        d_decay_pair = (cap_d * torch.tanh(F.linear(h, upcast(self.decay_up.weight)))).view(bsz, seq_len, hv, half_dk)

        beta_f = upcast(beta_base).clamp(0.0, 1.0).unsqueeze(-1)
        b_gate = 2.0 * logistic_shift(0.5 * beta_f, d_erase)
        if self.cfg.paired_b_gate:
            b_gate = (
                b_gate.view(bsz, seq_len, hv, half_dk, 2)
                .mean(dim=-1, keepdim=True)
                .expand(bsz, seq_len, hv, half_dk, 2)
                .reshape(bsz, seq_len, hv, dk)
            )
        w_gate = logistic_shift(beta_f, d_write)

        # Bug 3B: Keep decay factored as gamma_page_scale [M] and gamma_chan_pair [B, T, H_v, d_k // 2]
        base_rate = 1.0 + 0.1 * (-upcast(g_base).clamp(max=0.0)).tanh().unsqueeze(-1)
        gamma_chan_pair = base_rate * d_decay_pair.exp()
        gamma_page_scale = upcast(self.page_gamma_prior) * upcast(self.log_page_gamma).clamp(-3.0, 3.0).exp()

        if not self.cfg.use_phase:
            theta = xf.new_zeros(bsz, seq_len, hv, half_dk)
        elif self.theta_down is not None and self.theta_up is not None:
            theta_raw = F.linear(
                F.linear(xf, upcast(self.theta_down.weight)),
                upcast(self.theta_up.weight),
            )
            theta = math.pi * torch.tanh(theta_raw).view(bsz, seq_len, hv, half_dk)
        else:
            assert self.theta_proj is not None
            theta = math.pi * torch.tanh(F.linear(xf, upcast(self.theta_proj.weight))).view(bsz, seq_len, hv, half_dk)

        surprise_bias = F.linear(xf, upcast(self.surprise_proj.weight), upcast(self.surprise_proj.bias))
        write_logits = F.linear(xf, upcast(self.write_router.weight), upcast(self.write_router.bias))
        q_flat = q_prep.reshape(bsz, seq_len, hv * dk)
        read_logits = F.linear(q_flat, upcast(self.read_router.weight), upcast(self.read_router.bias))

        root_dk = math.sqrt(dk)
        dq_swa = F.linear(h, upcast(self.swa_q_up.weight)).view(bsz, seq_len, hv, dk)
        dk_swa = F.linear(h, upcast(self.swa_k_up.weight)).view(bsz, seq_len, hv, dk)
        dv_swa = F.linear(h, upcast(self.swa_v_up.weight)).view(bsz, seq_len, hv, dv)
        q_swa = _rms_norm_scaled(root_dk * q_prep + dq_swa, self.swa_q_norm)
        k_swa = _rms_norm_scaled(root_dk * k_prep + dk_swa, self.swa_k_norm)
        v_swa = v_prep + dv_swa

        return {
            "query": q_prep.to(query.dtype),
            "key": k_prep.to(key.dtype),
            "value": v_prep.to(value.dtype),
            "gamma_page_scale": gamma_page_scale,
            "gamma_chan_pair": gamma_chan_pair,
            "b_gate": b_gate,
            "w_gate": w_gate,
            "theta": theta,
            "surprise_bias": surprise_bias,
            "write_logits": write_logits,
            "read_logits": read_logits,
            "q_swa": q_swa,
            "k_swa": k_swa,
            "v_swa": v_swa,
        }

    def record_duty(
        self,
        duty_vec: torch.Tensor,
        *,
        write_logits: torch.Tensor | None = None,
        pad_mask: torch.Tensor | None = None,
    ) -> None:
        bud_loss = compute_write_budget_loss(
            duty_vec,
            self.mu_max,
            mu_min_ratio=self.cfg.mu_min_ratio,
            budget_weight=self.cfg.budget_weight,
        )
        if _in_backward():
            return
        self._pending_aux.append(bud_loss)
        if _is_compiling():
            return
        with torch.no_grad():
            det = duty_vec.detach()
            self.last_page_duty = det
            dead_cnt = int((det < 1e-6).sum().item())
            if write_logits is not None and write_logits.numel() > 0:
                probs_d = F.softmax(write_logits.detach().float(), dim=-1)
                if pad_mask is not None and pad_mask.shape == probs_d.shape[:2]:
                    pm_f = (pad_mask.detach() > 0.5).float().unsqueeze(-1)
                    dense_vec = (probs_d * pm_f).sum(dim=(0, 1)) / pm_f.sum().clamp_min(1.0)
                else:
                    dense_vec = probs_d.mean(dim=(0, 1))
                self.last_dense_router_prob = dense_vec
                dense_prob_stat = float(dense_vec.min().item())
            else:
                dense_prob_stat = float(det.mean().item()) if det.numel() > 0 else 0.0
            self.last_stats["budget_aux_loss"] = float(bud_loss.detach().item())
            self.last_stats["vault_duty"] = float(det[: min(2, det.numel())].mean().item())
            self.last_stats["scratch_duty"] = float(det[min(4, det.numel() - 1) :].mean().item())
            self.last_stats["dense_router_prob"] = dense_prob_stat
            self.last_stats["dead_page_count"] = float(dead_cnt)


def _extract_hssd_gdr_args(
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    default_chunk: int,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    HSSDCacheState | torch.Tensor | None,
    bool,
    bool,
    float | None,
    int,
    Any,
    torch.Tensor | None,
]:
    g = args[0] if len(args) >= 1 else kwargs.get("g")
    beta = args[1] if len(args) >= 2 else kwargs.get("beta")
    require(
        isinstance(g, torch.Tensor) and isinstance(beta, torch.Tensor),
        "HSSD wrapper expected Tensor `g` and `beta`",
        g=g,
        beta=beta,
    )
    chunk_size = int(kwargs.get("chunk_size", default_chunk))
    initial_state = kwargs.get("initial_state")
    for extra in args[2:]:
        if isinstance(extra, int):
            chunk_size = extra
        elif initial_state is None and (
            isinstance(extra, (HSSDCacheState, torch.Tensor)) or _is_hssd_state_duck(extra)
        ):
            initial_state = extra
    output_final = bool(kwargs.get("output_final_state", False))
    use_l2 = bool(kwargs.get("use_qk_l2norm_in_kernel", False))
    scale = kwargs.get("scale")
    cu_seqlens = kwargs.get("cu_seqlens")
    seq_idx = kwargs.get("seq_idx")
    return g, beta, initial_state, output_final, use_l2, scale, chunk_size, cu_seqlens, seq_idx


def _active_mor_write_mask(text_model: nn.Module, bsz: int, seq_len: int) -> torch.Tensor | None:
    """Inspect `LoopController` (if attached via `apply_loop`) for Pass `r >= 2` MoR `keep` mask."""
    layers = getattr(text_model, "layers", None)
    if not isinstance(layers, nn.ModuleList) or len(layers) == 0:
        return None
    ctl = getattr(layers[0], "_ctl", None)
    if ctl is None:
        return None
    cur = getattr(ctl, "_cur", None)
    if cur is None:
        return None
    ctx, r = cur
    spec = getattr(ctl, "spec", None)
    if r < 2 or spec is None or not (bool(getattr(spec, "mor", False)) and bool(getattr(spec, "write_mask", False))):
        return None
    keep = ctx.keep.get(r)
    if isinstance(keep, torch.Tensor) and keep.shape == (bsz, seq_len):
        return keep
    return None


class HSSDHandle:
    """Manages `HSSDAdapter` modules, forward hooks, aux write-budget losses, and clean removal."""

    def __init__(
        self,
        model: nn.Module,
        cfg: HSSDConfig,
        adapters: dict[int, HSSDAdapter],
        hooks: list[Any],
        restorers: list[Callable[[], None]],
    ):
        self.model = model
        self.cfg = cfg
        self.adapters = adapters
        self._hooks = hooks
        self._restorers = restorers
        self._removed = False

    def trainable_parameters(self) -> list[nn.Parameter]:
        return [p for ad in self.adapters.values() for p in ad.parameters() if p.requires_grad]

    def parameters(self) -> list[nn.Parameter]:
        return [p for ad in self.adapters.values() for p in ad.parameters()]

    def clear_aux(self) -> None:
        for ad in self.adapters.values():
            ad.clear_aux()

    def aux_loss(self, *, clear: bool = True) -> torch.Tensor:
        """Aggregate two-sided cumulative actual-sparse write-mass budget loss across layers & passes."""
        terms: list[torch.Tensor] = []
        for ad in self.adapters.values():
            if ad._pending_aux:
                terms.extend(ad._pending_aux)
                if clear:
                    ad.clear_aux()
        if not terms:
            if self.adapters:
                ref = next(iter(self.adapters.values())).log_surprise_slope
                return ref.new_zeros(())
            return torch.zeros(())
        return torch.stack(terms).mean()

    def page_duties(self) -> dict[int, torch.Tensor]:
        return {
            lid: ad.last_page_duty.clone()
            for lid, ad in self.adapters.items()
            if ad.last_page_duty is not None
        }

    def stats(self) -> dict[str, float]:
        if not self.adapters:
            return {}
        out: dict[str, float] = {}
        stat_keys = (
            "budget_aux_loss",
            "vault_duty",
            "scratch_duty",
            "dense_router_prob",
            "dead_page_count",
        )
        for key in stat_keys:
            vals = [ad.last_stats[key] for ad in self.adapters.values() if key in ad.last_stats]
            if vals:
                out[f"hssd/{key}"] = sum(vals) / len(vals)
        return out

    def uninstall(self) -> None:
        if self._removed:
            return
        self._removed = True
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
        for fn in reversed(self._restorers):
            fn()
        self._restorers.clear()
        if getattr(self.model, "_hssd_handle", None) is self:
            delattr(self.model, "_hssd_handle")
        log.info("HSSD v3.1 uninstalled from %d linear-attention layers", len(self.adapters))

    def remove(self) -> None:
        self.uninstall()


HSSDManager = HSSDHandle


_HSSD_CHUNK_RULE_ATTRS: tuple[str, ...] = ("chunk_gated_delta_rule", "torch_chunk_gated_delta_rule")
_HSSD_REC_RULE_ATTRS: tuple[str, ...] = (
    "recurrent_gated_delta_rule",
    "fused_recurrent_gated_delta_rule",
    "torch_recurrent_gated_delta_rule",
)


def _make_hssd_rule_wrapper(
    adapter: HSSDAdapter,
    ctx_stack: list[tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]],
    *,
    recurrent_only: bool,
    orig_fn: Callable[..., Any] | None = None,
    in_rule_box: list[int] | None = None,
) -> Callable[..., Any]:
    def wrapped(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, *args: Any, **kwargs: Any):
        if in_rule_box is not None and in_rule_box[0] > 0:
            if orig_fn is not None:
                return orig_fn(query, key, value, *args, **kwargs)
            raise RuntimeError(f"HSSD layer {adapter.layer_id}: unexpected re-entrant rule call without orig_fn")
        if in_rule_box is not None:
            in_rule_box[0] += 1
        try:
            require(bool(ctx_stack), f"HSSD layer {adapter.layer_id}: missing captured input hidden_states")
            x_in, mor_wmask, pad_mask = ctx_stack[-1]
            bsz, seq_len, _ = x_in.shape
            g, beta, init_st, out_final, use_l2, scale, _csize, cu_seqlens, seq_idx = _extract_hssd_gdr_args(
                args, kwargs, adapter.cfg.chunk_size
            )

            head_first = bool(kwargs.get("head_first", False)) or (
                seq_len != adapter.num_v_heads and value.shape[1] == adapter.num_v_heads and value.shape[2] == seq_len
            )
            if head_first:
                query, key, value = query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2)
                g, beta = g.transpose(1, 2), beta.transpose(1, 2)
            require(
                value.shape[:2] == (bsz, seq_len),
                f"HSSD layer {adapter.layer_id}: value shape {tuple(value.shape)} disagrees with input "
                f"{(bsz, seq_len)}",
            )

            proj = adapter.project_inputs(x_in, query, key, value, g, beta, use_qk_l2norm=use_l2)
            factored_gamma = (proj["gamma_page_scale"], proj["gamma_chan_pair"])
            c_cfg = adapter.cfg

            if seq_len == 1 and cu_seqlens is None and seq_idx is None:
                st0 = _coerce_init_state(
                    init_st,
                    bsz=bsz,
                    num_pages=c_cfg.num_pages,
                    hv=adapter.num_v_heads,
                    dk=adapter.head_k_dim,
                    dv=adapter.head_v_dim,
                    chunk_size=c_cfg.chunk_size,
                    window_chunks=c_cfg.window_chunks,
                    device=query.device,
                    dtype=torch.float64 if query.dtype == torch.float64 else torch.float32,
                )
                out, st_next, u_sum, cnt_sum = recurrent_hssd_step(
                    proj["query"],
                    proj["key"],
                    proj["value"],
                    factored_gamma,
                    proj["b_gate"],
                    proj["w_gate"],
                    proj["theta"],
                    proj["surprise_bias"],
                    adapter.log_surprise_slope.exp(),
                    adapter.surprise_thresh,
                    proj["write_logits"],
                    proj["read_logits"],
                    proj["q_swa"],
                    proj["k_swa"],
                    proj["v_swa"],
                    st0,
                    boltzmann_gate=adapter.boltzmann_gate,
                    top_k_pages=c_cfg.k_pages,
                    write_mask=mor_wmask,
                    pad_mask=pad_mask,
                    use_qk_l2norm_in_kernel=False,
                    scale=scale,
                    state_dtype=c_cfg.state_dtype,
                    use_swa=c_cfg.use_swa,
                    use_phase=c_cfg.use_phase,
                    use_surprise=c_cfg.use_surprise,
                    surprise_mode=c_cfg.surprise_mode,
                    return_duty_sum=True,
                )
                final_state = st_next if out_final else None
                duty_vec = u_sum / cnt_sum.clamp_min(1.0)
            elif recurrent_only:
                out, final_state, duty_vec = recurrent_hssd(
                    proj["query"],
                    proj["key"],
                    proj["value"],
                    factored_gamma,
                    proj["b_gate"],
                    proj["w_gate"],
                    proj["theta"],
                    proj["surprise_bias"],
                    adapter.log_surprise_slope.exp(),
                    adapter.surprise_thresh,
                    proj["write_logits"],
                    proj["read_logits"],
                    proj["q_swa"],
                    proj["k_swa"],
                    proj["v_swa"],
                    boltzmann_gate=adapter.boltzmann_gate,
                    chunk_size=c_cfg.chunk_size,
                    window_chunks=c_cfg.window_chunks,
                    num_pages=c_cfg.num_pages,
                    top_k_pages=c_cfg.k_pages,
                    initial_state=init_st,
                    output_final_state=out_final,
                    write_mask=mor_wmask,
                    pad_mask=pad_mask,
                    cu_seqlens=cu_seqlens,
                    seq_idx=seq_idx,
                    use_qk_l2norm_in_kernel=False,
                    scale=scale,
                    state_dtype=c_cfg.state_dtype,
                    use_swa=c_cfg.use_swa,
                    use_phase=c_cfg.use_phase,
                    use_surprise=c_cfg.use_surprise,
                    surprise_mode=c_cfg.surprise_mode,
                    return_aux=True,
                )
            else:
                out, final_state, duty_vec = chunk_hssd(
                    proj["query"],
                    proj["key"],
                    proj["value"],
                    factored_gamma,
                    proj["b_gate"],
                    proj["w_gate"],
                    proj["theta"],
                    proj["surprise_bias"],
                    adapter.log_surprise_slope.exp(),
                    adapter.surprise_thresh,
                    proj["write_logits"],
                    proj["read_logits"],
                    proj["q_swa"],
                    proj["k_swa"],
                    proj["v_swa"],
                    boltzmann_gate=adapter.boltzmann_gate,
                    chunk_size=c_cfg.chunk_size,
                    window_chunks=c_cfg.window_chunks,
                    num_pages=c_cfg.num_pages,
                    top_k_pages=c_cfg.k_pages,
                    segment_chunks=c_cfg.segment_chunks,
                    checkpoint_segments=c_cfg.checkpoint_segments,
                    initial_state=init_st,
                    output_final_state=out_final,
                    write_mask=mor_wmask,
                    pad_mask=pad_mask,
                    cu_seqlens=cu_seqlens,
                    seq_idx=seq_idx,
                    use_qk_l2norm_in_kernel=False,
                    scale=scale,
                    state_dtype=c_cfg.state_dtype,
                    use_swa=c_cfg.use_swa,
                    use_phase=c_cfg.use_phase,
                    use_surprise=c_cfg.use_surprise,
                    surprise_mode=c_cfg.surprise_mode,
                    return_aux=True,
                )

            adapter.record_duty(duty_vec, write_logits=proj["write_logits"], pad_mask=pad_mask)
            if head_first:
                out = out.transpose(1, 2)
            return out, final_state
        finally:
            if in_rule_box is not None:
                in_rule_box[0] -= 1

    return wrapped


def _resolve_physical_layers(model: nn.Module) -> tuple[nn.Module, nn.ModuleList, list[str]]:
    """Support installing HSSD either before `apply_loop` (standard) or on an already looped model."""
    tm = find_text_model(model)
    layers = getattr(tm, "layers", None)
    require(isinstance(layers, nn.ModuleList) and len(layers) > 0, "text model has no `.layers`")
    ctl = getattr(layers[0], "_ctl", None)
    if ctl is not None and hasattr(ctl, "_orig") and "layers" in ctl._orig:
        phys_layers = ctl._orig["layers"]
        cfg = getattr(tm, "config", None)
        types = [layer_type_of(layer, i, cfg) for i, layer in enumerate(phys_layers)]
        return tm, phys_layers, types
    parts = inspect_model(model)
    return parts.text_model, parts.layers, parts.layer_types


def apply_hssd(model: nn.Module, cfg: HSSDConfig | dict[str, Any] | None = None) -> HSSDHandle:
    """Retrofit Qwen3.6 / Toy linear-attention layers in-place to `HSSD v3.1`.

    Works seamlessly with `ToyForCausalLM` (`ToyHybridMoEForCausalLM`), HF Qwen3.6 GDN layers,
    `apply_loop` per-pass virtual cache slots, MoR early-exit write masks, 2D `attention_mask`
    padding invariance, and packed `cu_seqlens` / `seq_idx` document boundaries.
    """
    if getattr(model, "_hssd_handle", None) is not None:
        raise RuntimeError("HSSD v3 is already installed on this model; call remove_hssd(model) first")
    _ensure_dynamo_mark_static_address_compat()
    if not isinstance(cfg, HSSDConfig):
        cfg = HSSDConfig.from_dict(cfg)
    cfg.validate()

    text_model, phys_layers, layer_types = _resolve_physical_layers(model)
    lin_indices = [i for i, lt in enumerate(layer_types) if lt == LINEAR]
    target_layers = lin_indices if cfg.layers is None else [int(i) for i in cfg.layers]
    bad = [i for i in target_layers if i not in lin_indices]
    if bad:
        raise ValueError(f"hssd.layers {bad} are not linear_attention layers (linear: {lin_indices})")

    ref = text_model.embed_tokens.weight
    param_dtype = torch.float64 if ref.dtype == torch.float64 else torch.float32
    adapters: dict[int, HSSDAdapter] = {}
    hooks: list[Any] = []
    restorers: list[Callable[[], None]] = []
    patched_mod_fns: set[tuple[str, str]] = set()
    active_attn_stack: list[
        tuple[
            HSSDAdapter,
            list[tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]],
            list[int],
            bool,
            bool,
        ]
    ] = []
    amask_stack: list[torch.Tensor | None] = []

    rule_specs: list[tuple[str, bool]] = (
        [(attr, False) for attr in _HSSD_CHUNK_RULE_ATTRS]
        + [(attr, True) for attr in _HSSD_REC_RULE_ATTRS]
    )

    def _capture_model_amask(_mod: nn.Module, a: tuple[Any, ...], kw: dict[str, Any]) -> None:
        am = kw.get("attention_mask")
        if am is None and len(a) >= 2 and isinstance(a[1], torch.Tensor) and a[1].ndim == 2:
            am = a[1]
        if isinstance(am, torch.Tensor) and am.ndim == 2:
            amask_stack.append(am)
        elif amask_stack:
            amask_stack.append(amask_stack[-1])
        else:
            amask_stack.append(None)

    def _pop_model_amask(_mod: nn.Module, _inp: Any, _out: Any) -> None:
        if amask_stack:
            amask_stack.pop()

    def _clear_model_aux(_mod: nn.Module, _args: tuple[Any, ...]) -> None:
        if not _in_backward():
            for ad in adapters.values():
                ad.clear_aux()

    hooks.append(model.register_forward_pre_hook(_capture_model_amask, with_kwargs=True))
    hooks.append(model.register_forward_hook(_pop_model_amask))
    if text_model is not model:
        hooks.append(text_model.register_forward_pre_hook(_capture_model_amask, with_kwargs=True))
        hooks.append(text_model.register_forward_hook(_pop_model_amask))
    hooks.append(text_model.register_forward_pre_hook(_clear_model_aux))

    for i in target_layers:
        attn = attention_module(phys_layers[i], LINEAR)
        hs, hv, dk, dv = _infer_gdn_dims(attn, i)
        adapter = HSSDAdapter(hs, hv, dk, dv, cfg, i).to(device=ref.device, dtype=param_dtype)
        attn.hssd_adapter = adapter
        adapters[i] = adapter

        ctx_stack: list[tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]] = []
        in_rule_box: list[int] = [0]

        def _del_adapter(a: nn.Module = attn, ad: HSSDAdapter = adapter) -> None:
            if getattr(a, "hssd_adapter", None) is ad:
                delattr(a, "hssd_adapter")

        restorers.append(_del_adapter)

        if "_kernel_repo" in attn.__dict__ and "forward" in attn.__dict__:
            saved_kfwd = attn.__dict__.pop("forward")
            saved_krepo = attn.__dict__.pop("_kernel_repo")

            def _restore_kernel_fwd(
                a: nn.Module = attn,
                kfwd: Any = saved_kfwd,
                krepo: Any = saved_krepo,
            ) -> None:
                a.__dict__["forward"] = kfwd
                a.__dict__["_kernel_repo"] = krepo

            restorers.append(_restore_kernel_fwd)

        wrapped_chunk = False
        wrapped_rec = False
        for attr, rec_only in rule_specs:
            orig_fn = getattr(attn, attr, None)
            if callable(orig_fn):
                if rec_only:
                    wrapped_rec = True
                else:
                    wrapped_chunk = True
                had_instance_attr = attr in attn.__dict__
                setattr(
                    attn,
                    attr,
                    _make_hssd_rule_wrapper(
                        adapter, ctx_stack, recurrent_only=rec_only, orig_fn=orig_fn, in_rule_box=in_rule_box
                    ),
                )

                def _restore_attr(
                    a: nn.Module = attn,
                    name: str = attr,
                    fn: Callable[..., Any] = orig_fn,
                    had: bool = had_instance_attr,
                ) -> None:
                    if had:
                        setattr(a, name, fn)
                    elif name in a.__dict__:
                        delattr(a, name)

                restorers.append(_restore_attr)

        def _make_pre(
            ad_ref: HSSDAdapter,
            st_ref: list[tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]],
            ir_ref: list[int],
            wc_flag: bool,
            wr_flag: bool,
        ):
            def pre_hook(_mod: nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
                x_in = (
                    args[0]
                    if args
                    else (kwargs.get("hidden_states") if "hidden_states" in kwargs else kwargs.get("x"))
                )
                require(isinstance(x_in, torch.Tensor), "HSSD pre_hook could not locate input hidden_states")
                amask = kwargs.get("attention_mask", args[2] if len(args) >= 3 else None)
                if amask is None and amask_stack:
                    amask = amask_stack[-1]
                mor_wmask = _active_mor_write_mask(text_model, x_in.shape[0], x_in.shape[1])
                pad_mask: torch.Tensor | None = None
                ret_modified = False
                if (
                    isinstance(amask, torch.Tensor)
                    and amask.ndim == 2
                    and amask.shape[0] == x_in.shape[0]
                    and amask.shape[1] >= x_in.shape[1]
                ):
                    pad_mask = amask[:, -x_in.shape[1] :]
                    if not bool((pad_mask > 0.5).all()):
                        x_in = x_in * pad_mask.to(dtype=x_in.dtype).unsqueeze(-1)
                        ret_modified = True
                x_proj = x_in * mor_wmask.to(dtype=x_in.dtype).unsqueeze(-1) if mor_wmask is not None else x_in
                st_ref.append((x_proj, mor_wmask, pad_mask))
                active_attn_stack.append((ad_ref, st_ref, ir_ref, wc_flag, wr_flag))
                if ret_modified:
                    if args:
                        return (x_in, *args[1:]), kwargs
                    new_kw = dict(kwargs)
                    if "hidden_states" in new_kw:
                        new_kw["hidden_states"] = x_in
                    else:
                        new_kw["x"] = x_in
                    return args, new_kw
                return None

            return pre_hook

        def _make_post(st_ref: list[tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]]):
            def post_hook(_mod: nn.Module, _inp: Any, _out: Any) -> None:
                if st_ref:
                    st_ref.pop()
                if active_attn_stack:
                    active_attn_stack.pop()

            return post_hook

        hooks.append(
            attn.register_forward_pre_hook(
                _make_pre(adapter, ctx_stack, in_rule_box, wrapped_chunk, wrapped_rec),
                with_kwargs=True,
            )
        )
        hooks.append(attn.register_forward_hook(_make_post(ctx_stack)))

        if not (wrapped_chunk and wrapped_rec):
            mod_name = type(attn).__module__
            host_mod = sys.modules.get(mod_name)
            if host_mod is not None:
                missing_specs: list[tuple[str, bool]] = []
                if not wrapped_chunk:
                    missing_specs.extend((nm, False) for nm in _HSSD_CHUNK_RULE_ATTRS)
                if not wrapped_rec:
                    missing_specs.extend((nm, True) for nm in _HSSD_REC_RULE_ATTRS)
                for fn_name, rec_only in missing_specs:
                    if (mod_name, fn_name) in patched_mod_fns:
                        continue
                    orig_mfn = getattr(host_mod, fn_name, None)
                    if callable(orig_mfn):
                        patched_mod_fns.add((mod_name, fn_name))

                        def _mod_dispatch(
                            query: torch.Tensor,
                            key: torch.Tensor,
                            value: torch.Tensor,
                            *args: Any,
                            _orig: Callable[..., Any] = orig_mfn,
                            _rec: bool = rec_only,
                            **kwargs: Any,
                        ):
                            if not active_attn_stack:
                                return _orig(query, key, value, *args, **kwargs)
                            cur_ad, cur_st, cur_ir, cur_wc, cur_wr = active_attn_stack[-1]
                            if cur_ir[0] > 0 or (_rec and cur_wr) or ((not _rec) and cur_wc):
                                return _orig(query, key, value, *args, **kwargs)
                            return _make_hssd_rule_wrapper(
                                cur_ad, cur_st, recurrent_only=_rec, orig_fn=_orig, in_rule_box=cur_ir
                            )(query, key, value, *args, **kwargs)

                        setattr(host_mod, fn_name, _mod_dispatch)
                        restorers.append(lambda hm=host_mod, n=fn_name, f=orig_mfn: setattr(hm, n, f))

    handle = HSSDHandle(model, cfg, adapters, hooks, restorers)
    object.__setattr__(model, "_hssd_handle", handle)
    n_params = sum(p.numel() for p in handle.parameters())
    log.info(
        "HSSD v3.1 installed on %d linear-attention layers "
        "(rank=%d, low_rank_theta=%s, chunk=%d, window=%d, pages=%d, state_dtype=%s, params=%.2fM)",
        len(adapters),
        cfg.rank,
        cfg.theta_low_rank,
        cfg.chunk_size,
        cfg.window_chunks,
        cfg.num_pages,
        cfg.state_dtype,
        n_params / 1e6,
    )
    return handle


install_hssd = apply_hssd


def remove_hssd(model_or_handle: nn.Module | HSSDHandle) -> None:
    """Cleanly uninstall `HSSD v3` from `model` (or via its `HSSDHandle`)."""
    if isinstance(model_or_handle, HSSDHandle):
        model_or_handle.uninstall()
        return
    handle = getattr(model_or_handle, "_hssd_handle", None)
    if isinstance(handle, HSSDHandle):
        handle.uninstall()
        return
    raise ValueError(f"No active HSSDHandle found on {type(model_or_handle).__name__}")


def profile_hssd_layer(
    *,
    hidden_size: int = 2048,
    num_v_heads: int = 32,
    head_k_dim: int = 128,
    head_v_dim: int = 128,
    seq_len: int = 1024,
    stream_chunk_len: int = 256,
    state_dtype: str = "bfloat16",
    theta_low_rank: bool = True,
    rank: int = 64,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
) -> dict[str, Any]:
    """Re-export of `qloop.model.probes.profile_hssd_layer`."""
    from .probes import profile_hssd_layer as _profile_fn

    return _profile_fn(
        hidden_size=hidden_size,
        num_v_heads=num_v_heads,
        head_k_dim=head_k_dim,
        head_v_dim=head_v_dim,
        seq_len=seq_len,
        stream_chunk_len=stream_chunk_len,
        state_dtype=state_dtype,
        theta_low_rank=theta_low_rank,
        rank=rank,
        device=device,
        dtype=dtype,
    )


def hssd_ablation_matrix() -> dict[str, dict[str, Any]]:
    """Re-export of `qloop.model.probes.hssd_ablation_matrix`."""
    from .probes import hssd_ablation_matrix as _ablation_fn

    return _ablation_fn()


__all__ = [
    "DEFAULT_MU_MAX",
    "HSSDAdapter",
    "HSSDCacheState",
    "HSSDConfig",
    "HSSDHandle",
    "HSSDHybridCache",
    "HSSDManager",
    "PeakBoltzmannGate",
    "apply_2d_rotation",
    "apply_hssd",
    "chunk_hssd",
    "compute_write_budget_loss",
    "disjoint_horizon_bounds",
    "has_fla",
    "hssd_ablation_matrix",
    "install_hssd",
    "profile_hssd_layer",
    "recurrent_hssd",
    "recurrent_hssd_step",
    "remove_hssd",
]
