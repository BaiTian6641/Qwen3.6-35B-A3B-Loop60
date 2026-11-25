"""Distilled Latent Sparse Attention (D-LSA) for the gated full-attention layers.

Pipeline (Latent Sparse Attention, arXiv 2610.42147, D-LSA variant):
  1. index projections  q -> q_idx, k -> k_idx in a learned low-dim latent space (default 64),
     L2-normalised. We project only the *non-rotary* part of each head (Qwen3.6: dims 64..255 of
     256), so clustering is driven by content, not by RoPE phase.
  2. 4:1 pooling of consecutive key latents into "units".
  3. Ward (minimum-variance) agglomerative clustering of the units inside every complete block
     of `block_size` keys into buckets (avg `avg_bucket_keys`, hard cap `max_bucket_keys`),
     compacted to active cluster slots (`compact_active_slots=True`, `D5`) with zero host syncs (`D8`).
     Each bucket stores a normalised centroid and its member key positions (bucket cache).
  4. every query (per KV group by default, or per query head when `per_head_select=True`, `D4`)
     scores bucket centroids (flat search, or a two-level tree search over super-nodes of
     `tree_group` blocks with beam `tree_beam`) and keeps the top `num_select`.
  5. attention runs over   local window  U  sink tokens  U  tightly packed valid keys (`D6`)
     of the selected buckets (duplicates removed), with a standard softmax - only *which* keys
     are read changes.
  6. D-LSA training (`distill` and alternating sparse+KL `hybrid` mode, `D1`): index projections
     learn by minimising KL( A_full(bucket) || softmax(q_idx . c_bucket / tau) ) on sampled
     queries. Once `recall_ready` (`last_recall >= recall_threshold`), `hybrid` mode alternates
     true sparse forward steps with sampled KL supervision so downstream Q/O adapters, norms, and
     loop bridges adapt to the exact sparse key subset used at inference.
"""
from __future__ import annotations

import contextlib
import logging
import math
import weakref
from collections import OrderedDict
from collections.abc import Hashable, Iterator
from dataclasses import dataclass, fields
from typing import Any, Callable

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from ._cache_compat import extract_cache_from_kwargs
from ._dtype import upcast
from .introspect import FULL, attention_module, find_text_model, inspect_model

try:
    import triton
    import triton.language as tl

    _TRITON_AVAILABLE = True
except Exception:  # pragma: no cover - CPU / non-Triton environments
    triton = None  # type: ignore[assignment]
    tl = None  # type: ignore[assignment]
    _TRITON_AVAILABLE = False

log = logging.getLogger("qloop")

LSA_IMPL_NAME = "qloop_lsa"
_VALID_LSA_MODES: tuple[str, ...] = ("off", "dense", "distill", "sparse", "hybrid")
_LSA_FP8_DTYPES: tuple[torch.dtype, ...] = tuple(
    dt
    for dt in (
        getattr(torch, "float8_e4m3fn", None),
        getattr(torch, "float8_e4m3fnuz", None),
        getattr(torch, "float8_e5m2", None),
        getattr(torch, "float8_e5m2fnuz", None),
    )
    if dt is not None
)


def _weakref_or_none(obj: Any) -> weakref.ReferenceType[Any] | None:
    if obj is None:
        return None
    try:
        return weakref.ref(obj)
    except TypeError:
        return None


def _is_sync_restricted() -> bool:
    compiling_fn = getattr(torch.compiler, "is_compiling", None)
    if callable(compiling_fn) and bool(compiling_fn()):
        return True
    if torch.cuda.is_available() and torch.cuda.is_initialized():
        get_mode = getattr(torch.cuda, "get_sync_debug_mode", None)
        if callable(get_mode) and int(get_mode()) != 0:
            return True
    return False


def _ensure_attn_kv_dtype(
    module: nn.Module,
    t: torch.Tensor,
    target_dtype: torch.dtype,
    *,
    is_key: bool,
) -> torch.Tensor:
    if t.dtype in _LSA_FP8_DTYPES or not t.is_floating_point():
        scale_attr = "k_scale" if is_key else "v_scale"
        kvc = getattr(module, "kv_cache", None)
        sc = getattr(kvc, scale_attr, None) if kvc is not None else None
        if not isinstance(sc, torch.Tensor):
            sc = getattr(module, scale_attr, None)
        if isinstance(sc, torch.Tensor):
            sf = sc.to(device=t.device, dtype=torch.float32)
            if sf.numel() == 1:
                sf = sf.reshape(())
            elif sf.ndim == 1 and t.ndim == 4 and sf.shape[0] == t.shape[1]:
                sf = sf.view(1, -1, 1, 1)
            elif sf.ndim < t.ndim:
                while sf.ndim < t.ndim:
                    sf = sf.unsqueeze(-1)
            return (t.to(torch.float32) * sf).to(target_dtype)
        return t.to(target_dtype)
    return t


def _unwrap_pkv_for_cache(pkv: Any) -> tuple[Any | None, bool]:
    """Unwrap `SharedKVView` / `_ReplayKVCacheProxy` to root KV cache without host syncs."""
    cur = pkv
    for _ in range(4):
        if cur is None:
            return None, False
        try:
            pending = object.__getattribute__(cur, "_pending_replay")
        except AttributeError:
            pending = None
        if isinstance(pending, dict):
            if pending:
                return cur, False
            try:
                cur = object.__getattribute__(cur, "_base_cache")
                continue
            except AttributeError:
                break
        try:
            inner = object.__getattribute__(cur, "_cache")
            if inner is not None and inner is not cur:
                cur = inner
                continue
        except AttributeError:
            pass
        break
    return cur, (cur is not None)


# ============================================================================ config


@dataclass
class LSAConfig:
    mode: str = "distill"               # off | dense | distill | sparse | hybrid
    layers: list[int] | None = None     # physical full-attention layers; None = all of them
    index_dim: int = 64
    index_on: str = "nope"              # "nope" (non-rotary dims) | "all"
    rotary_dim: int | None = None       # default: head_dim * partial_rotary_factor
    pool: int = 4
    block_size: int = 512
    avg_bucket_keys: int = 8
    max_bucket_keys: int = 32
    num_select: int = 64
    window: int | None = None           # default: block_size
    sinks: int = 64
    search: str = "flat"                # flat | tree
    tree_group: int = 16
    tree_beam: int = 8
    temperature: float = 0.1
    distill_weight: float = 1.0
    distill_query_frac: float = 0.125
    distill_chunk: int = 16
    query_chunk: int = 256
    detach_inputs: bool = True
    checkpoint_chunks: bool = True
    strict_mask: bool = False
    cache_entries: int = 4
    num_query_passes: int = 1           # 1 = shared query indexer; >1 = per-pass query projections
    per_head_select: bool = False       # D4: per-query-head selection & KL when True
    max_selected_keys: int = 768        # D6: tight packed valid bucket-member cap per query
    compact_active_slots: bool = True   # D5: compact BucketIndex along slot axis (~50% slots)
    recall_threshold: float = 0.70      # D1: recall threshold to unlock hybrid sparse steps
    hybrid_sparse_every: int = 2        # D1: run sparse+KL every N steps in hybrid mode
    hybrid_warmup_steps: int = 0        # D1: optional step ceiling to activate hybrid mode
    recall_check_every: int = 5         # D1: cadence for sampled recall checks
    aux_reduction: str = "mean"         # C2: "mean" | "sum" reduction in LSAManager.aux_loss()

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> LSAConfig:
        d = {k: v for k, v in dict(d or {}).items() if k != "enabled"}
        known = {f.name for f in fields(cls)}
        bad = set(d) - known
        if bad:
            raise ValueError(f"unknown lsa.* keys {sorted(bad)}; valid: {sorted(known)}")
        return cls(**d)

    def resolved_window(self) -> int:
        return self.block_size if self.window is None else self.window

    def slots_per_block(self) -> int:
        """Exact active-slot upper bound per block (`n_clusters` under standard cap, `n` if uncompacted)."""
        n = self.block_size // self.pool
        if not self.compact_active_slots:
            return n
        n_clusters = max(1, self.block_size // self.avg_bucket_keys)
        max_units = max(1, self.max_bucket_keys // self.pool)
        m_cap = 1 if n <= max_units else 2 + (n - max_units - 1) // ((max_units // 2) + 1)
        return min(n, max(n_clusters, m_cap))

    def validate(self) -> None:
        if self.mode not in _VALID_LSA_MODES:
            raise ValueError(f"lsa.mode must be one of {_VALID_LSA_MODES}, got {self.mode!r}")
        if self.search not in ("flat", "tree"):
            raise ValueError(f"lsa.search must be flat|tree, got {self.search!r}")
        if self.num_query_passes < 1:
            raise ValueError(f"lsa.num_query_passes must be >= 1, got {self.num_query_passes}")
        if self.block_size % self.pool:
            raise ValueError("lsa.block_size must be a multiple of lsa.pool")
        if self.avg_bucket_keys % self.pool or self.max_bucket_keys % self.pool:
            raise ValueError("lsa.avg_bucket_keys / max_bucket_keys must be multiples of lsa.pool")
        if not self.pool <= self.avg_bucket_keys <= self.max_bucket_keys <= self.block_size:
            raise ValueError("need pool <= avg_bucket_keys <= max_bucket_keys <= block_size")
        if self.resolved_window() < self.block_size:
            raise ValueError("lsa.window must be >= lsa.block_size (it must cover the unclustered tail)")
        if self.hybrid_sparse_every < 0:
            raise ValueError(f"lsa.hybrid_sparse_every must be >= 0, got {self.hybrid_sparse_every}")
        if self.aux_reduction not in ("mean", "sum"):
            raise ValueError(f"lsa.aux_reduction must be 'mean' or 'sum', got {self.aux_reduction!r}")


# ============================================================================ clustering


def ward_cluster(x: torch.Tensor, n_clusters: int, max_size: int) -> torch.Tensor:
    """Batched Ward agglomerative clustering (zero GPU->CPU host syncs across all merge steps, `D8`).

    x: [N, n, d] points (one independent problem per row). Returns labels [N, n] where each
    point's label is the index of its cluster representative (the smallest member index).
    Merges never create clusters larger than `max_size` points; if the cap blocks further
    merges a row keeps more than `n_clusters` clusters via pure device-side `torch.where` no-ops.
    """
    nrow, n, _ = x.shape
    dev = x.device
    if n_clusters >= n or nrow == 0:
        return torch.arange(n, device=dev).expand(nrow, n).clone()
    x = upcast(x)
    cent = x.clone()
    size = torch.ones(nrow, n, device=dev, dtype=x.dtype)
    active = torch.ones(nrow, n, device=dev, dtype=torch.bool)
    labels = torch.arange(n, device=dev).expand(nrow, n).clone()
    if max_size < 2:
        return labels
    inf = torch.finfo(x.dtype).max
    eye = torch.eye(n, device=dev, dtype=torch.bool)
    dist = 0.5 * torch.cdist(x, x).pow(2)
    dist = dist.masked_fill(eye, inf)
    rows = torch.arange(nrow, device=dev)
    ar_n = torch.arange(n, device=dev)
    zeros_sz = torch.zeros(nrow, device=dev, dtype=x.dtype)
    zeros_bool = torch.zeros(nrow, device=dev, dtype=torch.bool)
    inf_row = torch.full((nrow, n), inf, device=dev, dtype=x.dtype)

    for _ in range(n - n_clusters):
        flat = dist.view(nrow, -1)
        best, idx = flat.min(dim=-1)
        ok = best < inf
        i = idx // n
        j = idx % n
        a = torch.minimum(i, j)
        b_idx = torch.maximum(i, j)

        na = size[rows, a]
        nb_sz = size[rows, b_idx]
        denom = (na + nb_sz).clamp_min(1e-12)
        merged_cent = (na[:, None] * cent[rows, a] + nb_sz[:, None] * cent[rows, b_idx]) / denom[:, None]
        cent[rows, a] = torch.where(ok[:, None], merged_cent, cent[rows, a])

        size[rows, a] = torch.where(ok, na + nb_sz, na)
        size[rows, b_idx] = torch.where(ok, zeros_sz, size[rows, b_idx])
        active[rows, b_idx] = torch.where(ok, zeros_bool, active[rows, b_idx])
        labels = torch.where(ok[:, None] & (labels == b_idx[:, None]), a[:, None], labels)

        # Ward distance of merged cluster `a` to every other active cluster
        sa = size[rows, a][:, None]
        sk = size
        d_new = (sa * sk / (sa + sk).clamp_min(1e-12)) * (cent - cent[rows, a][:, None]).pow(2).sum(-1)
        bad = (~active) | (sa + sk > max_size) | (ar_n[None, :] == a[:, None])
        d_new = d_new.masked_fill(bad, inf)

        upd_a = torch.where(ok[:, None], d_new, dist[rows, a, :])
        dist[rows, a, :] = upd_a
        dist[rows, :, a] = upd_a
        dist[rows, b_idx, :] = torch.where(ok[:, None], inf_row, dist[rows, b_idx, :])
        dist[rows, :, b_idx] = torch.where(ok[:, None], inf_row, dist[rows, :, b_idx])
    return labels


def members_table(labels: torch.Tensor, max_members: int) -> torch.Tensor:
    """labels [N, n] (representative index per point) -> [N, n, max_members] member indices, -1 padded."""
    nrow, n = labels.shape
    dev = labels.device
    if nrow == 0 or max_members <= 0:
        return torch.full((nrow, n, max(0, max_members)), -1, device=dev, dtype=torch.long)
    ar = torch.arange(n, device=dev)
    order = torch.argsort(labels * n + ar, dim=-1)
    sorted_lab = labels.gather(-1, order)
    counts = torch.zeros(nrow, n, device=dev, dtype=torch.long).scatter_add_(1, labels, torch.ones_like(labels))
    offsets = counts.cumsum(-1) - counts
    rank = ar.expand(nrow, n) - offsets.gather(1, sorted_lab)
    # Use a throwaway overflow slot at `max_members` so zero GPU->CPU sync (`counts.max()`) is needed
    out = torch.full((nrow, n, max_members + 1), -1, device=dev, dtype=torch.long)
    rr = torch.arange(nrow, device=dev)[:, None].expand(nrow, n)
    out[rr, sorted_lab, rank.clamp(min=0, max=max_members)] = order
    return out[..., :max_members].contiguous()


# ============================================================================ bucket index


@dataclass
class BucketIndex:
    n_blocks: int
    labels: torch.Tensor                  # [B,G,nb,n]       representative unit per unit
    centroids: torch.Tensor               # [B,G,S,d]        S = nb * slots_per_block (normalised; 0 if inactive)
    active: torch.Tensor                  # [B,G,S] bool
    members: torch.Tensor                 # [B,G,S,max_keys] key positions, -1 padded
    slot_end: torch.Tensor                # [S] exclusive end position of the slot's block
    key2slot: torch.Tensor                # [B,G,nb*block]   slot id of every clustered key
    counts: torch.Tensor | None = None    # [B,G,S] key counts per slot (`0` if inactive)
    block_id: torch.Tensor | None = None  # [S] block index `0..nb-1` of each slot
    slots_per_block: int = 0              # compacted (`~64`) or full (`128`) slots per block
    slot_labels: torch.Tensor | None = None  # [B,G,nb,n] local slot id in 0..spb-1 per unit
    slot_counts: torch.Tensor | None = None  # [B,G,nb,spb] unit counts per slot


@dataclass
class _CacheEntry:
    n_blocks: int
    last_tk: int
    k_ident: tuple[int, int]              # (data_ptr, _version) without host sync
    owner_ref: weakref.ReferenceType[Any] | None
    loop_ctx_ref: weakref.ReferenceType[Any] | None
    labels: torch.Tensor                  # [B,G,nb,n]
    slot_labels: torch.Tensor             # [B,G,nb,n] local slot id in 0..spb-1 per unit
    slot_counts: torch.Tensor             # [B,G,nb,spb] unit counts per slot
    unit_members: torch.Tensor            # [B,G,nb,spb,max_units]
    centroids: torch.Tensor | None        # [B,G,nb,spb,d] kept for no-grad incremental reuse
    wk_version: int = -1                  # self.wk._version for safe centroid reuse across opt steps
    assembled: BucketIndex | None = None  # cached assembled BucketIndex for no-grad full reuse


class _PassQueryProj(nn.Module):
    """Per-pass query latent projection (`[num_heads, din, index_dim]`) for passes 2..R."""

    def __init__(self, init_weight: torch.Tensor):
        super().__init__()
        self.weight = nn.Parameter(init_weight.clone())

    def forward(self, q_slice: torch.Tensor) -> torch.Tensor:
        return torch.matmul(q_slice.to(self.weight.dtype), self.weight)


class LSAIndexer(nn.Module):
    """Trainable index projections + bucket construction for one full-attention layer."""

    def __init__(self, num_heads: int, num_kv_heads: int, head_dim: int, cfg: LSAConfig, layer_id: int):
        super().__init__()
        cfg.validate()
        self.cfg, self.layer_id = cfg, layer_id
        self.h, self.g, self.dh = num_heads, num_kv_heads, head_dim
        self.hg = num_heads // num_kv_heads
        rot = cfg.rotary_dim or 0
        self.lo = rot if cfg.index_on == "nope" else 0
        din = head_dim - self.lo
        if din <= 0:
            raise ValueError(f"index_on=nope but rotary_dim={rot} >= head_dim={head_dim}")
        base = torch.randn(num_kv_heads, din, cfg.index_dim) / math.sqrt(din)
        # Same JL projection for keys and their query heads => q_idx.k_idx ~ cos(q, k) at init
        self.wk = nn.Parameter(base.clone())
        head_init = base.repeat_interleave(self.hg, dim=0)
        if cfg.per_head_select:
            # D4: [num_query_passes, num_q_heads, head_q_dim, index_dim], bit-identical across passes at step 0
            n_passes = max(1, cfg.num_query_passes)
            self.wq = nn.Parameter(head_init.unsqueeze(0).repeat(n_passes, 1, 1, 1).clone())
            self.q_proj_extra = nn.ModuleList()
        else:
            self.wq = nn.Parameter(head_init.clone())
            self.q_proj_extra = nn.ModuleList(
                [_PassQueryProj(self.wq.detach()) for _ in range(max(0, cfg.num_query_passes - 1))]
            )
        self._current_pass: int = 1
        self._cache: OrderedDict[tuple[Any, ...], _CacheEntry] = OrderedDict()
        self._active_pkv: tuple[weakref.ReferenceType[Any] | None, int | None, bool] = (None, None, False)
        self.pending: list[torch.Tensor] = []
        self.last_stats: dict[str, float] = {}
        self.last_selected_buckets: dict[int, torch.Tensor] = {}
        self._step: int = 0
        self._last_recall_step: int = -1
        self._recall_checks: int = 0
        self.last_recall: float = 0.0
        self.recall_ready: bool = bool(cfg.recall_threshold <= 0.0)

    @property
    def W_q_idx(self) -> nn.Parameter:
        return self.wq

    @property
    def W_k_idx(self) -> nn.Parameter:
        return self.wk

    @property
    def current_pass(self) -> int:
        return self._current_pass

    def set_current_pass(self, r: int) -> None:
        """Explicitly set the active query pass index `r >= 1` (used when not inside a VirtualLayer)."""
        if r < 1:
            raise ValueError(f"pass index r must be >= 1, got {r}")
        if self.cfg.num_query_passes > 1 and r > self.cfg.num_query_passes:
            raise ValueError(
                f"pass index r={r} exceeds lsa.num_query_passes={self.cfg.num_query_passes}"
            )
        self._current_pass = int(r)

    def _current_loop_state(self) -> tuple[Any | None, int | None, bool]:
        """Return `(loop_ctx, raw_pass, kv_share)` from the active `LoopController`, if present."""
        tm_ref = getattr(self, "_tm_ref", None)
        tm = tm_ref() if callable(tm_ref) else None
        if tm is not None:
            layers = getattr(tm, "layers", None)
            if layers is not None and len(layers) > 0:
                ctl = getattr(layers[0], "_ctl", None)
                if ctl is not None:
                    kv_share = bool(getattr(getattr(ctl, "spec", None), "kv_share", True))
                    cur = getattr(ctl, "_cur", None)
                    if cur is not None:
                        return cur[0], int(cur[1]), kv_share
                    return None, None, kv_share
        return None, None, True

    def _current_loop_ctx(self) -> tuple[Any | None, int | None]:
        loop_ctx, loop_r, _kv_share = self._current_loop_state()
        return loop_ctx, loop_r

    def _resolve_pass(self, module: nn.Module | None = None, pass_idx: int | None = None) -> int:
        cap = max(1, self.cfg.num_query_passes)
        if pass_idx is not None:
            return 1 if int(pass_idx) <= 1 else min(int(pass_idx), cap)
        _ctx, loop_r = self._current_loop_ctx()
        if loop_r is not None:
            return 1 if loop_r <= 1 else min(loop_r, cap)
        mod_r = getattr(module, "_current_pass", None) if module is not None else None
        if mod_r is not None:
            return 1 if int(mod_r) <= 1 else min(int(mod_r), cap)
        return 1 if self._current_pass <= 1 else min(self._current_pass, cap)

    def active_wq(self, pass_idx: int | None = None, module: nn.Module | None = None) -> nn.Parameter:
        if self.wq.ndim == 4:
            return self.wq
        r = self._resolve_pass(module=module, pass_idx=pass_idx)
        if r <= 1 or self.cfg.num_query_passes <= 1 or len(self.q_proj_extra) == 0:
            return self.wq
        proj: _PassQueryProj = self.q_proj_extra[r - 2]  # type: ignore[assignment]
        return proj.weight

    # --------------------------------------------------------------- latents
    def query_latent(
        self, q: torch.Tensor, pass_idx: int | None = None, module: nn.Module | None = None
    ) -> torch.Tensor:
        """Project `q [B,H,T,D]` -> `[B,G,T,d]` (default) or `[B,H,T,d]` (`per_head_select=True`)."""
        r = self._resolve_pass(module=module, pass_idx=pass_idx)
        q_sl = q[..., self.lo:]
        if self.wq.ndim == 4:
            p_idx = min(max(r - 1, 0), self.wq.shape[0] - 1)
            w_r = self.wq[p_idx]
            z = torch.matmul(q_sl.to(w_r.dtype), w_r)
        elif r <= 1 or self.cfg.num_query_passes <= 1 or len(self.q_proj_extra) == 0:
            z = torch.matmul(q_sl.to(self.wq.dtype), self.wq)
        else:
            z = self.q_proj_extra[r - 2](q_sl)
        z = F.normalize(upcast(z), dim=-1)
        if self.cfg.per_head_select:
            return z
        b, _, t, d = z.shape
        return z.view(b, self.g, self.hg, t, d).sum(2)

    def key_latent(self, k: torch.Tensor) -> torch.Tensor:
        z = torch.matmul(k[..., self.lo:].to(self.wk.dtype), self.wk)
        return F.normalize(upcast(z), dim=-1)

    def q_proj(self, q: torch.Tensor, pass_idx: int | None = None) -> torch.Tensor:
        """Alias for `query_latent` supporting explicit `pass_idx`."""
        return self.query_latent(q, pass_idx=pass_idx)

    def k_proj(self, k: torch.Tensor) -> torch.Tensor:
        """Differentiable key latent projection through `self.wk` (`W_k_idx`)."""
        return self.key_latent(k)

    def select_buckets(
        self,
        q: torch.Tensor,
        index: BucketIndex,
        qpos: torch.Tensor | None = None,
        pass_idx: int | None = None,
    ) -> torch.Tensor:
        """Return `[B, G, Tq, kk]` (or `[B, Hq, Tq, kk]` when `per_head_select=True`) slot indices."""
        tq = q.shape[2]
        if qpos is None:
            tk = index.n_blocks * self.cfg.block_size + self.cfg.resolved_window()
            qpos = torch.arange(tk - tq, tk, device=q.device)
        qz = self.query_latent(q, pass_idx=pass_idx)
        scores = self.slot_scores(qz, index, qpos)
        kk = min(self.cfg.num_select, scores.shape[-1])
        top = scores.topk(kk, dim=-1)
        return torch.where(torch.isfinite(top.values), top.indices, torch.full_like(top.indices, -1))

    select = select_buckets

    # --------------------------------------------------------------- bucket index
    def _cluster_blocks(
        self, zblocks: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Cluster `zblocks [B,G,nb,block,d]` -> `(labels, slot_labels, slot_counts, unit_members)`."""
        cfg = self.cfg
        b, g, nb, _, d = zblocks.shape
        dev = zblocks.device
        n = cfg.block_size // cfg.pool
        spb = cfg.slots_per_block()
        units = F.normalize(zblocks.reshape(b, g, nb, n, cfg.pool, d).mean(4), dim=-1)
        n_clusters = max(1, cfg.block_size // cfg.avg_bucket_keys)
        max_units = cfg.max_bucket_keys // cfg.pool
        flat_labels = ward_cluster(units.reshape(-1, n, d), n_clusters, max_units)
        flat_mem = members_table(flat_labels, max_units)
        labels = flat_labels.view(b, g, nb, n)
        umem_full = flat_mem.view(b, g, nb, n, max_units)
        counts_full = torch.zeros_like(labels).scatter_add_(3, labels, torch.ones_like(labels))

        if spb == n:
            return labels, labels, counts_full, umem_full

        # D5: Compact each block along the slot axis from `n` (`128`) down to `spb` (`~64`) active slots!
        slot_counts, rep_units = counts_full.topk(spb, dim=-1, largest=True, sorted=True)
        rep_to_compact = torch.zeros_like(labels)
        ar_spb = torch.arange(spb, device=dev, dtype=labels.dtype).view(1, 1, 1, spb).expand_as(rep_units)
        rep_to_compact.scatter_(3, rep_units, ar_spb)
        slot_labels = rep_to_compact.gather(3, labels)
        umem_compact = umem_full.gather(3, rep_units.unsqueeze(-1).expand(b, g, nb, spb, max_units))
        umem_compact = torch.where(
            (slot_counts > 0).unsqueeze(-1), umem_compact, torch.full_like(umem_compact, -1)
        )
        return labels, slot_labels, slot_counts, umem_compact

    @staticmethod
    def _centroids(
        unit_lat: torch.Tensor,
        slot_labels: torch.Tensor,
        slot_counts: torch.Tensor,
        spb: int,
    ) -> torch.Tensor:
        """Compute normalised centroids `[B,G,nb,spb,d]` (`0` on inactive slots)."""
        b, g, nb, _n, d = unit_lat.shape
        idx = slot_labels.unsqueeze(-1).expand(b, g, nb, _n, d)
        cent = torch.zeros(b, g, nb, spb, d, device=unit_lat.device, dtype=unit_lat.dtype)
        cent = cent.scatter_add(3, idx, unit_lat)
        cent = F.normalize(cent, dim=-1)
        return cent * (slot_counts > 0).unsqueeze(-1).to(cent.dtype)

    def _resolve_cache_lookup(
        self,
        k: torch.Tensor,
        kin: torch.Tensor,
        nb: int,
        grad_mode: bool,
        cache_key: Hashable | None,
        *,
        masked: bool = False,
    ) -> tuple[
        tuple[Any, ...] | None,
        _CacheEntry | None,
        int,
        weakref.ReferenceType[Any] | None,
        weakref.ReferenceType[Any] | None,
    ]:
        """Resolve zero-sync cache key and safe block-reuse count (`D8`, `QL60-008`)."""
        b, g, tk, _ = k.shape
        dev_s = str(k.device)
        k_ident = (int(kin.data_ptr()), int(kin._version))
        loop_ctx, loop_r, kv_share = self._current_loop_state()
        loop_ref = _weakref_or_none(loop_ctx)
        # Preserve single shared BucketIndex when `kv_share=True`; isolate per pass when `kv_share=False`!
        pass_cache_key = 0 if (kv_share or loop_r is None) else loop_r
        owners = getattr(self, "_owners", None)
        slot_id = int(getattr(owners[0], "layer_idx", self.layer_id)) if owners else self.layer_id

        wk_ver = int(self.wk._version)

        if cache_key is not None:
            lkey: tuple[Any, ...] = ("explicit", cache_key, pass_cache_key, slot_id, b, g, dev_s, masked)
            ent = self._cache.get(lkey)
            reuse = ent.n_blocks if (ent is not None and ent.n_blocks <= nb and ent.wk_version == wk_ver) else 0
            return lkey, ent, reuse, None, loop_ref

        pkv_ref, pkv_id, pkv_ok = self._active_pkv
        pkv_obj = pkv_ref() if pkv_ref is not None else None
        if pkv_ok and (pkv_obj is not None or pkv_ref is None) and pkv_id is not None:
            lkey = ("pkv", pkv_id, self.layer_id, pass_cache_key, slot_id, b, g, dev_s, masked)
            ent = self._cache.get(lkey)
            reuse = 0
            if ent is not None and ent.n_blocks <= nb and ent.wk_version == wk_ver:
                owner_alive = (ent.owner_ref is None) or (ent.owner_ref() is pkv_obj)
                same_loop = (
                    loop_ctx is not None
                    and ent.loop_ctx_ref is not None
                    and ent.loop_ctx_ref() is loop_ctx
                )
                # Reuse prior blocks when tokens were appended (`tk > ent.last_tk`) or within the
                # same loop forward / identical unmutated tensor (`tk == ent.last_tk`). If a caller
                # reset/mutated a preallocated cache in-place at `tk <= ent.last_tk`, rebuild cleanly!
                same_tk_ok = same_loop or (not masked and k_ident == ent.k_ident)
                if owner_alive and (tk > ent.last_tk or (tk == ent.last_tk and same_tk_ok)):
                    reuse = ent.n_blocks
            return lkey, ent, reuse, pkv_ref, loop_ref

        if loop_ctx is not None:
            lkey = ("loop", id(loop_ctx), self.layer_id, pass_cache_key, slot_id, b, g, tk, dev_s, masked)
            ent = self._cache.get(lkey)
            reuse = (
                ent.n_blocks
                if (
                    ent is not None
                    and ent.n_blocks == nb
                    and ent.wk_version == wk_ver
                    and ent.owner_ref is not None
                    and ent.owner_ref() is loop_ctx
                )
                else 0
            )
            return lkey, ent, reuse, loop_ref, loop_ref

        if not grad_mode and not masked:
            lkey = ("tensor", k_ident[0], tuple(kin.shape), tuple(kin.stride()), k_ident[1], dev_s)
            ent = self._cache.get(lkey)
            k_ref = _weakref_or_none(k)
            reuse = (
                ent.n_blocks
                if (
                    ent is not None
                    and ent.n_blocks == nb
                    and ent.wk_version == wk_ver
                    and ent.k_ident == k_ident
                    and ent.owner_ref is not None
                    and ent.owner_ref() is k
                )
                else 0
            )
            return lkey, ent, reuse, k_ref, None

        return None, None, 0, None, None

    def build(
        self,
        k: torch.Tensor,
        cache_key: Hashable | None = None,
        *,
        key_valid_mask: torch.Tensor | None = None,
    ) -> BucketIndex | None:
        """Build or incrementally extend the `BucketIndex` for `k [B,G,Tk,D]` with zero host syncs."""
        cfg = self.cfg
        b, g, tk, _ = k.shape
        bs, n, pool = cfg.block_size, cfg.block_size // cfg.pool, cfg.pool
        spb = cfg.slots_per_block()
        nb = tk // bs
        if nb == 0:
            return None
        kin = k.detach() if cfg.detach_inputs else k
        grad_mode = torch.is_grad_enabled() and (self.wk.requires_grad or kin.requires_grad)
        km: torch.Tensor | None = None
        if key_valid_mask is not None:
            km = key_valid_mask.to(device=k.device, dtype=torch.bool)
            if km.ndim == 1:
                km = km.view(1, 1, -1, 1)
            elif km.ndim == 2:
                km = km.view(km.shape[0], 1, km.shape[1], 1)
            else:
                km = km.reshape(b, 1, tk, 1)

        lkey, ent, reuse, owner_ref, loop_ref = self._resolve_cache_lookup(
            k, kin, nb, grad_mode, cache_key, masked=(km is not None)
        )

        # Fast path: identical block set in no-grad mode -> return cached BucketIndex directly
        # (avoids re-allocating members / key2slot / slot_end across loop passes & ckpt recompute).
        if not grad_mode and reuse == nb and ent is not None and ent.assembled is not None:
            ent.last_tk = tk
            ent.k_ident = (int(kin.data_ptr()), int(kin._version))
            ent.loop_ctx_ref = loop_ref
            if lkey is not None and lkey in self._cache:
                self._cache.move_to_end(lkey)
            return ent.assembled

        def _masked_key_lat(k_slice: torch.Tensor, lo_t: int, hi_t: int) -> torch.Tensor:
            z_out = self.key_latent(k_slice)
            if km is not None:
                z_out = z_out * km[:, :, lo_t:hi_t, :].to(z_out.dtype)
            return z_out

        with torch.no_grad():
            if reuse < nb:
                znew = _masked_key_lat(kin[:, :, reuse * bs : nb * bs], reuse * bs, nb * bs).view(
                    b, g, nb - reuse, bs, -1
                )
                lab_new, slab_new, scnt_new, mem_new = self._cluster_blocks(znew)
            if reuse:
                assert ent is not None
                if reuse == nb:
                    labels = ent.labels
                    slot_labels = ent.slot_labels
                    slot_counts = ent.slot_counts
                    umem = ent.unit_members
                else:
                    labels = torch.cat([ent.labels, lab_new], dim=2)
                    slot_labels = torch.cat([ent.slot_labels, slab_new], dim=2)
                    slot_counts = torch.cat([ent.slot_counts, scnt_new], dim=2)
                    umem = torch.cat([ent.unit_members, mem_new], dim=2)
            else:
                labels, slot_labels, slot_counts, umem = lab_new, slab_new, scnt_new, mem_new

        if grad_mode:
            z = _masked_key_lat(kin[:, :, : nb * bs], 0, nb * bs).view(b, g, nb, n, pool, -1).sum(4)
            cent = self._centroids(z, slot_labels, slot_counts, spb)
        else:
            with torch.no_grad():
                if reuse and ent is not None and ent.centroids is not None:
                    if reuse < nb:
                        zn = (
                            _masked_key_lat(kin[:, :, reuse * bs : nb * bs], reuse * bs, nb * bs)
                            .view(b, g, nb - reuse, n, pool, -1)
                            .sum(4)
                        )
                        cent = torch.cat(
                            [ent.centroids, self._centroids(zn, slab_new, scnt_new, spb)], dim=2
                        )
                    else:
                        cent = ent.centroids
                else:
                    z = _masked_key_lat(kin[:, :, : nb * bs], 0, nb * bs).view(b, g, nb, n, pool, -1).sum(4)
                    cent = self._centroids(z, slot_labels, slot_counts, spb)

        assembled = self._assemble(labels, slot_labels, slot_counts, umem, cent)
        if lkey is not None and cfg.cache_entries > 0:
            self._cache[lkey] = _CacheEntry(
                n_blocks=nb,
                last_tk=tk,
                k_ident=(int(kin.data_ptr()), int(kin._version)),
                owner_ref=owner_ref,
                loop_ctx_ref=loop_ref,
                labels=labels,
                slot_labels=slot_labels,
                slot_counts=slot_counts,
                unit_members=umem,
                centroids=None if grad_mode else cent.detach(),
                wk_version=int(self.wk._version),
                assembled=None if grad_mode else assembled,
            )
            self._cache.move_to_end(lkey)
            while len(self._cache) > cfg.cache_entries:
                self._cache.popitem(last=False)

        return assembled

    bucket_index = build

    def _assemble(
        self,
        labels: torch.Tensor,
        slot_labels: torch.Tensor,
        slot_counts: torch.Tensor,
        umem: torch.Tensor,
        cent: torch.Tensor,
    ) -> BucketIndex:
        cfg = self.cfg
        b, g, nb, _n = labels.shape
        spb = slot_counts.shape[3]
        bs, pool, dev = cfg.block_size, cfg.pool, labels.device
        s_total = nb * spb
        counts_keys = (slot_counts * pool).reshape(b, g, s_total)
        active = (slot_counts > 0).reshape(b, g, s_total)
        blk = torch.arange(nb, device=dev)
        # unit -> key positions
        upos = blk.view(1, 1, nb, 1, 1) * bs + umem * pool
        keys = upos.unsqueeze(-1) + torch.arange(pool, device=dev)
        keys = torch.where(umem.unsqueeze(-1) >= 0, keys, torch.full_like(keys, -1))
        members = keys.reshape(b, g, s_total, -1)
        block_id = torch.arange(s_total, device=dev) // spb
        slot_end = (block_id + 1) * bs
        ar_keys = torch.arange(nb * bs, device=dev)
        unit_of_key = (ar_keys % bs) // pool
        blk_of_key = ar_keys // bs
        key2slot = blk_of_key * spb + slot_labels[:, :, blk_of_key, unit_of_key]
        return BucketIndex(
            n_blocks=nb,
            labels=labels,
            centroids=cent.reshape(b, g, s_total, -1),
            active=active,
            members=members,
            slot_end=slot_end,
            key2slot=key2slot,
            counts=counts_keys,
            block_id=block_id,
            slots_per_block=spb,
            slot_labels=slot_labels,
            slot_counts=slot_counts,
        )

    # --------------------------------------------------------------- selection
    def slot_scores(
        self, qz: torch.Tensor, index: BucketIndex, qpos: torch.Tensor, flat: bool = False
    ) -> torch.Tensor:
        """`qz [B,G_or_Hq,C,d]`, `qpos [C]` -> masked scores `[B,G_or_Hq,C,S]` (`-inf` if ineligible)."""
        cfg = self.cfg
        b, q_heads, c, d = qz.shape
        s_total = index.centroids.shape[2]
        frontier = (qpos // cfg.block_size) * cfg.block_size
        elig = index.active[:, :, None, :] & (
            index.slot_end[None, None, None, :] <= frontier[None, None, :, None]
        )
        cent = index.centroids.to(qz.dtype)
        per_head = cfg.per_head_select or (q_heads == self.h and self.h != self.g)
        if per_head:
            hg = q_heads // self.g
            qz_g = qz.view(b, self.g, hg, c, d)
            s = torch.matmul(qz_g, cent.unsqueeze(2).transpose(-1, -2)) / cfg.temperature
            if cfg.search == "tree" and not flat:
                tmask = self._tree_mask_per_head(qz_g, index, elig, qpos)
                s = s.masked_fill(~tmask, float("-inf"))
            s = s.masked_fill(~elig[:, :, None, :, :], float("-inf"))
            return s.reshape(b, q_heads, c, s_total)

        s = torch.matmul(qz, cent.transpose(-1, -2)) / cfg.temperature
        if cfg.search == "tree" and not flat:
            s = s.masked_fill(~self._tree_mask(qz, index, elig, qpos), float("-inf"))
        return s.masked_fill(~elig, float("-inf"))

    def _tree_super_summaries(
        self, index: BucketIndex, b: int, g: int
    ) -> tuple[torch.Tensor, torch.Tensor, int, int]:
        cfg = self.cfg
        spb = index.slots_per_block or (index.centroids.shape[2] // max(1, index.n_blocks))
        tg = cfg.tree_group
        nb = index.n_blocks
        ng = -(-nb // tg)
        cent = index.centroids.view(b, g, nb, spb, -1) * index.active.view(b, g, nb, spb, 1)
        bsum = cent.sum(3)
        pad = ng * tg - nb
        if pad:
            bsum = F.pad(bsum, (0, 0, 0, pad))
        cs = bsum.view(b, g, ng, tg, -1).cumsum(3)
        full = F.normalize(cs[:, :, :, -1], dim=-1)
        return cs, full, ng, spb

    def _tree_mask(
        self, qz: torch.Tensor, index: BucketIndex, elig: torch.Tensor, qpos: torch.Tensor
    ) -> torch.Tensor:
        """Two-level causal super-node beam search (`per_head_select=False`)."""
        cfg = self.cfg
        b, g, c, s = elig.shape
        tg = cfg.tree_group
        ng = -(-index.n_blocks // tg)
        if ng <= cfg.tree_beam:
            return torch.ones_like(elig)
        cs, full, ng, spb = self._tree_super_summaries(index, b, g)
        fb = qpos // cfg.block_size
        jq, rem = fb // tg, fb % tg
        ss = torch.matmul(qz, full.to(qz.dtype).transpose(-1, -2))
        part = F.normalize(cs[:, :, jq.clamp(max=ng - 1), (rem - 1).clamp(min=0)], dim=-1)
        sp = (qz * part.to(qz.dtype)).sum(-1)
        jn = torch.arange(ng, device=qz.device)
        is_part = jn[None, :] == jq[:, None]
        ss = torch.where(is_part[None, None], sp.unsqueeze(-1), ss)
        valid = (jn[None, :] < jq[:, None]) | (is_part & (rem[:, None] > 0))
        ss = ss.masked_fill(~valid[None, None], float("-inf"))
        top = ss.topk(min(cfg.tree_beam, ng), dim=-1)
        keep_sup = torch.zeros(b, g, c, ng, dtype=torch.bool, device=qz.device)
        keep_sup.scatter_(-1, top.indices, torch.isfinite(top.values))
        return keep_sup.repeat_interleave(tg * spb, dim=-1)[..., :s]

    def _tree_mask_per_head(
        self, qz_g: torch.Tensor, index: BucketIndex, elig: torch.Tensor, qpos: torch.Tensor
    ) -> torch.Tensor:
        """Two-level causal super-node beam search per query head (`[B,G,Hg,C,S]`)."""
        cfg = self.cfg
        b, g, hg, c, _d = qz_g.shape
        s = elig.shape[-1]
        tg = cfg.tree_group
        ng = -(-index.n_blocks // tg)
        if ng <= cfg.tree_beam:
            return torch.ones((b, g, hg, c, s), dtype=torch.bool, device=qz_g.device)
        cs, full, ng, spb = self._tree_super_summaries(index, b, g)
        fb = qpos // cfg.block_size
        jq, rem = fb // tg, fb % tg
        ss = torch.matmul(qz_g, full.to(qz_g.dtype).unsqueeze(2).transpose(-1, -2))
        part = F.normalize(cs[:, :, jq.clamp(max=ng - 1), (rem - 1).clamp(min=0)], dim=-1)
        sp = (qz_g * part.to(qz_g.dtype)[:, :, None, :, :]).sum(-1)
        jn = torch.arange(ng, device=qz_g.device)
        is_part = jn[None, :] == jq[:, None]
        ss = torch.where(is_part[None, None, None], sp.unsqueeze(-1), ss)
        valid = (jn[None, :] < jq[:, None]) | (is_part & (rem[:, None] > 0))
        ss = ss.masked_fill(~valid[None, None, None], float("-inf"))
        top = ss.topk(min(cfg.tree_beam, ng), dim=-1)
        keep_sup = torch.zeros(b, g, hg, c, ng, dtype=torch.bool, device=qz_g.device)
        keep_sup.scatter_(-1, top.indices, torch.isfinite(top.values))
        return keep_sup.repeat_interleave(tg * spb, dim=-1)[..., :s]

    def evaluate_recall(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        *,
        top_k_keys: int = 64,
        qpos: torch.Tensor | None = None,
        pass_idx: int | None = None,
        cache_key: Hashable | None = None,
    ) -> dict[str, Any]:
        """Measure per-head top-k historical key recall and update `last_recall` / `recall_ready`."""
        with torch.no_grad():
            index = self.build(k, cache_key=cache_key)
            if index is None:
                res: dict[str, Any] = {
                    "mean_recall": 1.0,
                    "min_head_recall": 1.0,
                    "per_head_recall": [1.0] * self.h,
                }
            else:
                sel = self.select_buckets(q, index, qpos=qpos, pass_idx=pass_idx)
                res = compute_lsa_recall(
                    q,
                    k,
                    index,
                    sel,
                    top_k_keys=top_k_keys,
                    window=self.cfg.resolved_window(),
                    sinks=self.cfg.sinks,
                    qpos=qpos,
                )
            self._update_recall_state(float(res["mean_recall"]), float(res["min_head_recall"]))
            return res

    def _update_recall_state(self, mean_rec: float, min_rec: float) -> None:
        self.last_recall = float(mean_rec)
        self._last_recall_step = self._step
        self._recall_checks += 1
        self.last_stats["recall"] = self.last_recall
        self.last_stats["min_head_recall"] = float(min_rec)
        if self.last_recall >= self.cfg.recall_threshold or self.cfg.recall_threshold <= 0.0:
            self.recall_ready = True
        mgr_ref = getattr(self, "_mgr_ref", None)
        mgr = mgr_ref() if callable(mgr_ref) else None
        if mgr is not None:
            mgr._refresh_recall_from_indexers()

    # --------------------------------------------------------------- losses
    def pop_losses(self) -> list[torch.Tensor]:
        out, self.pending = self.pending, []
        return out


# ============================================================================ recall metric


@torch.no_grad()
def compute_lsa_recall(
    q: torch.Tensor,
    k: torch.Tensor,
    index: BucketIndex,
    idx: torch.Tensor | LSAIndexer,
    *,
    top_k_keys: int = 64,
    window: int = 512,
    sinks: int = 64,
    scale: float | None = None,
    qpos: torch.Tensor | None = None,
    query_chunk: int = 64,
) -> dict[str, Any]:
    """Compute exact per-head recall of dense top-`top_k_keys` historical keys captured by `idx`.

    Restricts evaluation to historical keys in `[sinks, min(q_t - window + 1, block_frontier(q_t)))`
    where bucket selection operates (excluding always-kept sinks and local window).
    """
    if q.ndim == 5:
        b, g_in, hg_in, tq, dh = q.shape
        q = q.reshape(b, g_in * hg_in, tq, dh)
    b, hq, tq, dh = q.shape
    g, tk = k.shape[1], k.shape[2]
    hg = hq // g
    dev = q.device
    sc = (dh ** -0.5) if scale is None else float(scale)

    sel_slots = idx.select_buckets(q, index, qpos=qpos) if isinstance(idx, LSAIndexer) else idx
    qpos = torch.arange(tk - tq, tk, device=dev) if qpos is None else qpos.to(device=dev)

    # Determine block_size from index (`nb * block_size == key2slot.shape[-1]`)
    nb = max(1, int(index.n_blocks))
    indexed_keys = int(index.key2slot.shape[-1])
    bs = max(1, indexed_keys // nb)

    frontier = ((qpos // bs) * bs).clamp(max=indexed_keys)
    win_lo = qpos - int(window) + 1
    hist_hi = torch.minimum(win_lo, frontier)
    valid_q = hist_hi > int(sinks)
    if not bool(valid_q.any()):
        return {
            "mean_recall": 1.0,
            "min_head_recall": 1.0,
            "per_head_recall": [1.0] * hq,
        }

    # Expand KV-group bucket selection `[B, G, Tq, K_sel]` to `[B, Hq, Tq, K_sel]` if needed
    if sel_slots.shape[1] == g and g != hq:
        sel_h = sel_slots.unsqueeze(2).expand(b, g, hg, tq, sel_slots.shape[-1]).reshape(
            b, hq, tq, sel_slots.shape[-1]
        )
    else:
        sel_h = sel_slots

    valid_q_idx = valid_q.nonzero(as_tuple=True)[0]
    head_rec_sum = torch.zeros(hq, device=dev, dtype=torch.float64)
    total_q_eval = 0

    kf = upcast(k[:, :, : int(hist_hi.max())])
    max_hi = kf.shape[2]
    kpos = torch.arange(max_hi, device=dev)
    k2s_h = (
        index.key2slot[:, :, :max_hi]
        .unsqueeze(2)
        .expand(b, g, hg, max_hi)
        .reshape(b, hq, max_hi)
    )

    for c0 in range(0, int(valid_q_idx.numel()), max(1, int(query_chunk))):
        qi = valid_q_idx[c0 : c0 + max(1, int(query_chunk))]
        c_len = int(qi.numel())
        hi_c = hist_hi[qi]
        qc = upcast(q[:, :, qi]).view(b, g, hg, c_len, dh)
        scores = torch.einsum("bghcd,bgtd->bghct", qc, kf).reshape(b, hq, c_len, max_hi) * sc
        hmask = (kpos[None, :] >= int(sinks)) & (kpos[None, :] < hi_c[:, None])
        scores = scores.masked_fill(~hmask[None, None, :, :], float("-inf"))

        n_hist_c = hmask.sum(dim=-1)  # [c_len], strictly >= 1
        k_req = min(int(top_k_keys), int(n_hist_c.max()))
        top_vals, top_k_idx = scores.topk(k_req, dim=-1)  # [B, Hq, c_len, k_req]
        valid_top = torch.isfinite(top_vals) & (
            torch.arange(k_req, device=dev)[None, :] < n_hist_c[:, None]
        )[None, None, :, :]

        # Map each true top-k key to its bucket slot and check membership in `sel_h`
        true_slots = k2s_h.unsqueeze(2).expand(b, hq, c_len, max_hi).gather(-1, top_k_idx)
        chosen = sel_h[:, :, qi, :]  # [B, Hq, c_len, K_sel]
        hit = (
            (true_slots.unsqueeze(-1) == chosen.unsqueeze(-2)) & (chosen.unsqueeze(-2) >= 0)
        ).any(dim=-1) & valid_top

        denom = valid_top.sum(dim=-1).clamp_min(1).to(torch.float64)
        rec_bc = hit.to(torch.float64).sum(dim=-1) / denom  # [B, Hq, c_len]
        head_rec_sum += rec_bc.sum(dim=(0, 2))
        total_q_eval += b * c_len

    per_head = (head_rec_sum / max(1, total_q_eval)).tolist()
    mean_rec = float(sum(per_head) / max(1, len(per_head)))
    min_rec = float(min(per_head)) if per_head else 1.0
    return {
        "mean_recall": mean_rec,
        "min_head_recall": min_rec,
        "per_head_recall": [float(v) for v in per_head],
    }


# ============================================================================ attention


def _in_backward() -> bool:
    fn = getattr(torch._C, "_current_graph_task_id", None)
    return fn is not None and fn() != -1


def _pack_valid_members(
    mem: torch.Tensor,
    m_b: torch.Tensor,
    *,
    tk: int,
    eff_cap: int,
    sync_free: bool,
    out_dtype: torch.dtype | None = None,
) -> tuple[torch.Tensor, torch.Tensor, int] | None:
    """Pack valid (`m_b`) bucket key indices per query row down to `L_pack <= eff_cap` (`D6`)."""
    dev = mem.device
    kb_n = mem.shape[-1]
    if kb_n == 0 or eff_cap <= 0:
        return None

    n_sel = 0
    if sync_free:
        l_pack = min(kb_n, eff_cap)
    else:
        valid_per_q = m_b.sum(dim=-1)
        max_valid = int(valid_per_q.max())
        if max_valid == 0:
            return None
        l_pack = min(max_valid, eff_cap, kb_n)

    # Use int32 internally on CUDA when indices fit comfortably in int31 (2x faster sort/topk + half HBM).
    use_i32 = dev.type == "cuda" and (tk + l_pack) < (1 << 30) and kb_n < (1 << 30)
    work_dt = torch.int32 if use_i32 else torch.long
    mem_work = mem.to(work_dt) if mem.dtype != work_dt else mem

    # Priority `Kb - j` keeps earliest / highest-scoring selected buckets first when capped!
    ar_kb = torch.arange(kb_n, device=dev, dtype=work_dt)
    prio = torch.where(m_b, kb_n - ar_kb, torch.zeros_like(ar_kb))
    if l_pack < kb_n:
        top_prio, pack_idx = prio.topk(l_pack, dim=-1, largest=True, sorted=True)
        mem_p = mem_work.gather(-1, pack_idx)
        m_p = top_prio > 0
    else:
        mem_p = mem_work
        m_p = m_b

    # Sort packed valid indices ascending for coalesced KV gather & exact deduplication
    sentinel = tk + torch.arange(mem_p.shape[-1], device=dev, dtype=work_dt)
    key_for_sort = torch.where(m_p, mem_p, sentinel)
    key_sorted, _ = key_for_sort.sort(dim=-1)
    if key_sorted.shape[-1] > 1:
        is_dup = F.pad(key_sorted[..., 1:] == key_sorted[..., :-1], (1, 0), value=False)
    else:
        is_dup = torch.zeros_like(key_sorted, dtype=torch.bool)
    m_final = (key_sorted < tk) & (~is_dup)
    mem_final = torch.where(m_final, key_sorted, torch.zeros_like(key_sorted))
    target_dt = out_dtype if out_dtype is not None else torch.long
    if mem_final.dtype != target_dt:
        mem_final = mem_final.to(target_dt)
    if not sync_free:
        n_sel = int(m_final.sum())
    return mem_final, m_final, n_sel


def _effective_member_cap(cfg: LSAConfig, s_slots: int, kk: int, kb_n: int, tk: int) -> int:
    if cfg.max_selected_keys <= 0 or kk >= s_slots:
        return kb_n
    if cfg.num_select * cfg.avg_bucket_keys > cfg.max_selected_keys:
        return min(kb_n, max(cfg.max_selected_keys, min(tk, cfg.num_select * cfg.avg_bucket_keys * 2)))
    return min(kb_n, cfg.max_selected_keys)


_ADDITIVE_MASK_DISALLOW_THRESHOLD = -1e4


def _normalize_caller_mask(
    attention_mask: torch.Tensor,
    *,
    b: int,
    g: int,
    hg: int,
    tq: int,
    tk: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Normalize 1D/2D/3D/4D bool, int, or additive float `attention_mask` to 5D without host syncs.

    Returns `(allow, bias)` each of shape `[B_m, G_m, Hg_m, Tq_m, Tk]` where:
      * `B_m in (1, B)`, `(G_m, Hg_m) in ((1, 1), (G, 1), (G, Hg))`, `Tq_m in (1, Tq)`
      * `allow` is `torch.bool` (`True` = allowed, `False` = masked out)
      * `bias` is `None` for bool/int masks, or upcast additive float bias zeroed on `~allow`
    """
    m = attention_mask
    if m.device != device:
        m = m.to(device=device)
    h = g * hg
    if m.shape[-1] > tk:
        m = m[..., :tk]
    if m.shape[-1] != tk:
        raise ValueError(f"attention_mask last dim ({m.shape[-1]}) must be >= Tk ({tk})")

    if m.ndim == 1:
        m4 = m.view(1, 1, 1, tk)
    elif m.ndim == 2:
        d0 = m.shape[0]
        if d0 == 1:
            m4 = m.view(1, 1, 1, tk)
        elif d0 == b and d0 != tq:
            m4 = m.view(b, 1, 1, tk)
        elif d0 == tq and d0 != b:
            m4 = m.view(1, 1, tq, tk)
        elif d0 == b and d0 == tq:
            # Ambiguous only when B == Tq > 1: float masks follow SDPA `[Tq, Tk]`; bool/int follow HF `[B, Tk]`.
            m4 = m.view(1, 1, tq, tk) if m.is_floating_point() else m.view(b, 1, 1, tk)
        elif d0 == tk and tq < tk:
            m4 = m[tk - tq : tk, :tk].view(1, 1, tq, tk)
        else:
            raise ValueError(
                f"incompatible 2D attention_mask shape {tuple(m.shape)} for B={b}, Tq={tq}, Tk={tk}"
            )
    elif m.ndim == 3:
        d0, d1, _ = m.shape
        if d1 > tq and d1 == tk:
            m = m[:, tk - tq : tk, :]
            d1 = tq
        if d0 in (1, b) and d1 in (1, tq):
            m4 = m.unsqueeze(1)
        elif d0 == h and b == 1 and d1 in (1, tq):
            m4 = m.unsqueeze(0)
        else:
            raise ValueError(
                f"incompatible 3D attention_mask shape {tuple(m.shape)} for B={b}, H={h}, Tq={tq}, Tk={tk}"
            )
    elif m.ndim == 4:
        d0, d1, d2, _ = m.shape
        if d2 > tq and d2 == tk:
            m = m[:, :, tk - tq : tk, :]
            d2 = tq
        if d0 not in (1, b) or d1 not in (1, g, h) or d2 not in (1, tq):
            raise ValueError(
                f"incompatible 4D attention_mask shape {tuple(m.shape)} for B={b}, G={g}, H={h}, Tq={tq}, Tk={tk}"
            )
        m4 = m
    else:
        raise ValueError(f"unsupported attention_mask ndim={m.ndim} (expected 1..4)")

    bm, hm, tqm, _ = m4.shape
    if hm == 1:
        m5 = m4.view(bm, 1, 1, tqm, tk)
    elif hm == g:
        m5 = m4.view(bm, g, 1, tqm, tk)
    else:
        m5 = m4.view(bm, g, hg, tqm, tk)

    if m5.dtype == torch.bool:
        return m5, None
    if not m5.is_floating_point():
        return m5 != 0, None

    mf = upcast(m5)
    allow = torch.isfinite(mf) & (mf > _ADDITIVE_MASK_DISALLOW_THRESHOLD)
    bias = torch.where(allow, mf, torch.zeros_like(mf))
    return allow, bias


def _slice_caller_mask(
    mask5: torch.Tensor | None,
    q_idx: slice | torch.Tensor,
    k_end: int | None = None,
) -> torch.Tensor | None:
    if mask5 is None:
        return None
    sub = mask5 if mask5.shape[3] == 1 else mask5[:, :, :, q_idx, :]
    return sub if k_end is None else sub[..., :k_end]


def _slot_valid_mask(
    allow_nk: torch.Tensor,
    key2slot: torch.Tensor,
    *,
    b: int,
    g: int,
    hg: int,
    c: int,
    s_slots: int,
    per_head: bool,
) -> torch.Tensor:
    """Compute `[B, G, C_m, S]` or `[B, H, C, S]` boolean mask of slots with >= 1 allowed key."""
    nk = key2slot.shape[-1]
    dev = key2slot.device
    if per_head:
        hg_m, c_m = allow_nk.shape[2], allow_nk.shape[3]
        cnt = torch.zeros((b, g, hg_m, c_m, s_slots), device=dev, dtype=torch.int32)
        cnt.scatter_add_(
            4,
            key2slot[:, :, None, None, :].expand(b, g, hg_m, c_m, nk),
            allow_nk.expand(b, g, hg_m, c_m, nk).to(torch.int32),
        )
        return (cnt > 0).expand(b, g, hg, c, s_slots).reshape(b, g * hg, c, s_slots)
    allow_g = allow_nk.any(dim=2)  # [B_m, G_m, C_m, nk]
    c_m = allow_g.shape[2]
    cnt = torch.zeros((b, g, c_m, s_slots), device=dev, dtype=torch.int32)
    cnt.scatter_add_(
        3,
        key2slot.unsqueeze(2).expand(b, g, c_m, nk),
        allow_g.expand(b, g, c_m, nk).to(torch.int32),
    )
    return cnt > 0


def _sparse_chunk(
    q: torch.Tensor,
    qz: torch.Tensor,
    qpos: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    index: BucketIndex | None,
    idx: LSAIndexer,
    scaling: float,
    cur_pass: int = 1,
    q_bounds: tuple[int, int] | None = None,
    record_stats: bool = True,
    caller_allow: torch.Tensor | None = None,
    caller_bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """`q [B,G,Hg,C,D]`, `qz [B,G_or_Hq,C,d]`, `qpos [C]` -> `out [B,G,Hg,C,D]`."""
    cfg = idx.cfg
    w = cfg.resolved_window()
    dev = q.device
    b, g, hg, c, dh = q.shape
    tk = k.shape[2]
    sync_free = _is_sync_restricted()

    if q_bounds is not None:
        q0_pos, q1_pos = q_bounds
        lo, hi = max(0, q0_pos - w + 1), q1_pos + 1
    else:
        lo = max(0, int(qpos[0]) - w + 1) if not sync_free else 0
        hi = int(qpos[-1]) + 1 if not sync_free else tk
    kw, vw = k[:, :, lo:hi], v[:, :, lo:hi]
    kpos = torch.arange(lo, hi, device=dev)
    win_lo = qpos - w + 1
    m_win = (kpos[None] <= qpos[:, None]) & (kpos[None] >= win_lo[:, None])
    ns = min(cfg.sinks, hi)
    spos = torch.arange(ns, device=dev)
    m_sink = (spos[None] < win_lo[:, None]) & (spos[None] <= qpos[:, None])
    s_win = torch.einsum("bghcd,bgld->bghcl", q, kw) * scaling
    m_win_5d = m_win[None, None, None]
    if caller_allow is not None:
        m_win_5d = m_win_5d & caller_allow[..., lo:hi]
        if caller_bias is not None:
            s_win_u = upcast(s_win)
            s_win = s_win_u + caller_bias[..., lo:hi].to(s_win_u.dtype)
    s_parts = [s_win]
    masks = [m_win_5d]
    if ns:
        s_sink = torch.einsum("bghcd,bgld->bghcl", q, k[:, :, :ns]) * scaling
        m_sink_5d = m_sink[None, None, None]
        if caller_allow is not None:
            m_sink_5d = m_sink_5d & caller_allow[..., :ns]
            if caller_bias is not None:
                s_sink_u = upcast(s_sink)
                s_sink = s_sink_u + caller_bias[..., :ns].to(s_sink_u.dtype)
        s_parts.append(s_sink)
        masks.append(m_sink_5d)

    kb = vb = None
    per_head = cfg.per_head_select or (qz.shape[1] == g * hg and hg > 1)
    n_sel = 0
    n_hist_bound = max(0, (q_bounds[1] - w + 1) - ns) if q_bounds is not None else tk
    need_buckets = index is not None and (n_hist_bound > 0 or record_stats)
    if need_buckets:
        assert index is not None
        with torch.no_grad():
            scores = idx.slot_scores(qz.detach(), index, qpos)
            s_slots = scores.shape[-1]
            if caller_allow is not None:
                nk = index.key2slot.shape[-1]
                slot_valid = _slot_valid_mask(
                    caller_allow[..., :nk],
                    index.key2slot,
                    b=b,
                    g=g,
                    hg=hg,
                    c=c,
                    s_slots=s_slots,
                    per_head=per_head,
                )
                scores = scores.masked_fill(~slot_valid, float("-inf"))
            kk = min(cfg.num_select, s_slots)
            top = scores.topk(kk, dim=-1)
            finite_top = torch.isfinite(top.values)
            sel_slots = torch.where(finite_top, top.indices, torch.full_like(top.indices, -1))
            if record_stats and not _in_backward():
                idx.last_selected_buckets[cur_pass] = sel_slots
            packed = None
            if n_hist_bound > 0:
                mk = index.members.shape[-1]
                kb_n = kk * mk
                eff_cap = min(_effective_member_cap(cfg, s_slots, kk, kb_n, tk), n_hist_bound)

                if per_head:
                    top_idx_g = top.indices.view(b, g, hg, c, kk)
                    fin_g = finite_top.view(b, g, hg, c, kk)
                    flat_sel = top_idx_g.reshape(b, g, hg * c * kk, 1).expand(b, g, hg * c * kk, mk)
                    mem = index.members.gather(2, flat_sel).view(b, g, hg, c, kk, mk)
                    mem = torch.where(fin_g.unsqueeze(-1), mem, torch.full_like(mem, -1)).flatten(-2)
                    m_b = (mem >= ns) & (mem < win_lo[None, None, None, :, None])
                    if caller_allow is not None:
                        mem_safe = mem.clamp(min=0, max=tk - 1)
                        mem_allow = caller_allow.expand(b, g, hg, c, tk).gather(-1, mem_safe)
                        m_b = m_b & mem_allow
                else:
                    flat_sel = top.indices.reshape(b, g, c * kk, 1).expand(b, g, c * kk, mk)
                    mem = index.members.gather(2, flat_sel).view(b, g, c, kk, mk)
                    mem = torch.where(finite_top.unsqueeze(-1), mem, torch.full_like(mem, -1)).flatten(-2)
                    m_b = (mem >= ns) & (mem < win_lo[None, None, :, None])
                    if caller_allow is not None:
                        mem_safe = mem.clamp(min=0, max=tk - 1)
                        allow_g = (
                            caller_allow[:, :, 0]
                            if caller_allow.shape[2] == 1
                            else caller_allow.any(dim=2)
                        )
                        mem_allow = allow_g.expand(b, g, c, tk).gather(-1, mem_safe)
                        m_b = m_b & mem_allow

                # Skip GPU->CPU host syncs (`valid_per_q.max()` / `m_final.sum()`) on intermediate chunks
                # when `record_stats=False` — padded entries stay masked (`-inf`), keeping attention exact!
                pack_sync_free = sync_free or (not record_stats)
                packed = _pack_valid_members(mem, m_b, tk=tk, eff_cap=eff_cap, sync_free=pack_sync_free)

        if packed is not None:
            mem_final, m_final, n_sel = packed
            l_pack = mem_final.shape[-1]
            if per_head:
                gidx = mem_final.reshape(b, g, hg * c * l_pack, 1).expand(b, g, hg * c * l_pack, dh)
                kb = k.gather(2, gidx).view(b, g, hg, c, l_pack, dh)
                vb = v.gather(2, gidx).view(b, g, hg, c, l_pack, dh)
                s_b = torch.einsum("bghcd,bghckd->bghck", q, kb) * scaling
                if caller_bias is not None:
                    s_b_u = upcast(s_b)
                    bias_b = caller_bias.expand(b, g, hg, c, tk).gather(-1, mem_final)
                    s_b = s_b_u + bias_b.to(s_b_u.dtype)
                s_parts.append(s_b)
                masks.append(m_final)
            else:
                gidx = mem_final.reshape(b, g, c * l_pack, 1).expand(b, g, c * l_pack, dh)
                kb = k.gather(2, gidx).view(b, g, c, l_pack, dh)
                vb = v.gather(2, gidx).view(b, g, c, l_pack, dh)
                s_b = torch.einsum("bghcd,bgckd->bghck", q, kb) * scaling
                m_b_final = m_final[:, :, None]
                if caller_allow is not None and caller_allow.shape[2] > 1:
                    head_allow = caller_allow.expand(b, g, hg, c, tk).gather(
                        -1, mem_final[:, :, None].expand(b, g, hg, c, l_pack)
                    )
                    m_b_final = m_b_final & head_allow
                if caller_bias is not None:
                    s_b_u = upcast(s_b)
                    if caller_bias.shape[2] == 1:
                        bias_b = caller_bias[:, :, 0].expand(b, g, c, tk).gather(-1, mem_final)[:, :, None]
                    else:
                        bias_b = caller_bias.expand(b, g, hg, c, tk).gather(
                            -1, mem_final[:, :, None].expand(b, g, hg, c, l_pack)
                        )
                    s_b = s_b_u + bias_b.to(s_b_u.dtype)
                s_parts.append(s_b)
                masks.append(m_b_final)

    if caller_allow is None:
        scores_all = torch.cat(
            [upcast(sp).masked_fill(~mk_i, float("-inf")) for sp, mk_i in zip(s_parts, masks)], dim=-1
        )
        attn = torch.softmax(scores_all, dim=-1)
        # Guard rows that might have all -inf (never occurs in causal self-attn since qpos is in window)
        attn = torch.nan_to_num(attn, nan=0.0).to(v.dtype)
    else:
        scores_raw = torch.cat([upcast(sp) for sp in s_parts], dim=-1)
        mask_all = torch.cat([mk_i.expand(sp.shape) for sp, mk_i in zip(s_parts, masks)], dim=-1)
        has_any = mask_all.any(dim=-1, keepdim=True)
        neg_inf = scores_raw.new_full((), float("-inf"))
        zero_s = scores_raw.new_zeros(())
        scores_safe = torch.where(mask_all, scores_raw, torch.where(has_any, neg_inf, zero_s))
        attn = torch.softmax(scores_safe, dim=-1)
        attn = torch.where(has_any, attn, torch.zeros_like(attn)).to(v.dtype)
    sizes = [sp.shape[-1] for sp in s_parts]
    parts = torch.split(attn, sizes, dim=-1)
    out = torch.einsum("bghcl,bgld->bghcd", parts[0], vw)
    pi = 1
    if ns:
        out = out + torch.einsum("bghcl,bgld->bghcd", parts[1], v[:, :, :ns])
        pi = 2
    if kb is not None and vb is not None:
        if per_head:
            out = out + torch.einsum("bghck,bghckd->bghcd", parts[pi], vb)
        else:
            out = out + torch.einsum("bghck,bgckd->bghcd", parts[pi], vb)

    if record_stats and not sync_free and not _in_backward():
        denom_q = b * (g * hg if per_head else g) * max(1, c)
        per_q_bucket = n_sel / max(1, denom_q)
        kpq = float(m_win.sum() + m_sink.sum()) / max(1, c) + per_q_bucket
        idx.last_stats["keys_per_query"] = kpq
        idx.last_stats[f"keys_per_query_p{cur_pass}"] = kpq
    return out


def _distill_kl(
    q: torch.Tensor,
    qz: torch.Tensor,
    qpos: torch.Tensor,
    k: torch.Tensor,
    index: BucketIndex,
    idx: LSAIndexer,
    scaling: float,
    caller_allow: torch.Tensor | None = None,
    caller_bias: torch.Tensor | None = None,
) -> torch.Tensor | None:
    """KL(true bucket mass || indexer distribution) for queries `qpos` (`q [B,G,Hg,C,D]`)."""
    cfg = idx.cfg
    frontier = (qpos // cfg.block_size) * cfg.block_size
    valid = frontier > 0
    if not bool(valid.any()):
        return None
    hi = int(qpos[-1]) + 1
    s_slots = index.centroids.shape[2]
    per_head = cfg.per_head_select or (qz.shape[1] == idx.h and idx.h != idx.g)
    b, g, hg, c, _ = q.shape
    with torch.no_grad():
        s = torch.einsum("bghcd,bgtd->bghct", upcast(q), upcast(k[:, :, :hi])) * scaling
        kpos = torch.arange(hi, device=q.device)
        m_allow = kpos[None, None, None, None] <= qpos[None, None, None, :, None]
        if caller_allow is not None:
            m_allow = m_allow & caller_allow[..., :hi]
            if caller_bias is not None:
                s = s + caller_bias[..., :hi].to(s.dtype)
            has_any = m_allow.any(dim=-1, keepdim=True)
            s = torch.where(m_allow, s, torch.where(has_any, s.new_full((), float("-inf")), s.new_zeros(())))
            a = torch.where(has_any, torch.softmax(s, dim=-1), torch.zeros_like(s))
        else:
            s = s.masked_fill(~m_allow, float("-inf"))
            a = torch.softmax(s, dim=-1)
        nk = min(hi, index.key2slot.shape[-1])
        a = a[..., :nk] * (kpos[:nk][None] < frontier[:, None])[None, None, None]
        slots = index.key2slot[:, :, :nk][:, :, None, None, :].expand(b, g, hg, c, nk)
        mass_h = torch.zeros(b, g, hg, c, s_slots, device=q.device, dtype=a.dtype)
        mass_h = mass_h.scatter_add_(-1, slots, a)
        mass = mass_h.reshape(b, g * hg, c, s_slots) if per_head else mass_h.mean(2)
        tot = mass.sum(-1, keepdim=True)
        target = mass / tot.clamp_min(1e-30)
        ok = valid[None, None, :] & (tot.squeeze(-1) > 0)
    logits = idx.slot_scores(qz, index, qpos, flat=True)
    if caller_allow is not None:
        nk_full = index.key2slot.shape[-1]
        slot_valid = _slot_valid_mask(
            caller_allow[..., :nk_full],
            index.key2slot,
            b=b,
            g=g,
            hg=hg,
            c=c,
            s_slots=s_slots,
            per_head=per_head,
        )
        logits = logits.masked_fill(~slot_valid, float("-inf"))
    has_slot = torch.isfinite(logits).any(dim=-1, keepdim=True)
    logp = torch.log_softmax(torch.where(has_slot, logits, torch.zeros_like(logits)), dim=-1)
    logp = torch.where(target > 0, logp, torch.zeros_like(logp))
    kl = (target * (torch.log(target.clamp_min(1e-30)) - logp)).sum(-1)
    return kl[ok].mean() if bool(ok.any()) else None


# ============================================================================ Triton sparse & distill kernels


if _TRITON_AVAILABLE:

    @triton.jit
    def _lsa_sparse_attn_fwd_kernel(
        Q_ptr,
        K_ptr,
        V_ptr,
        MEM_ptr,
        O_ptr,
        LSE_ptr,
        sq_b,
        sq_g,
        sq_h,
        sq_t,
        sq_d,
        sk_b,
        sk_g,
        sk_t,
        sk_d,
        sv_b,
        sv_g,
        sv_t,
        sv_d,
        sm_b,
        sm_g,
        sm_t,
        sm_l,
        so_b,
        so_g,
        so_h,
        so_t,
        so_d,
        sl_b,
        sl_g,
        sl_h,
        sl_t,
        G,
        TQ,
        TK,
        Q_BASE,
        WIN,
        NS,
        L_PACK,
        SCALE,
        HG: tl.constexpr,
        D_HEAD: tl.constexpr,
        BLOCK_H: tl.constexpr,
        BLOCK_K: tl.constexpr,
        HAS_BUCKETS: tl.constexpr,
    ):
        qi = tl.program_id(0)
        bg = tl.program_id(1)
        b_idx = bg // G
        g_idx = bg % G

        b64 = b_idx.to(tl.int64)
        g64 = g_idx.to(tl.int64)
        qi64 = qi.to(tl.int64)

        offs_h = tl.arange(0, BLOCK_H)
        offs_d = tl.arange(0, D_HEAD)
        offs_k = tl.arange(0, BLOCK_K)
        h_mask = offs_h < HG
        offs_h64 = offs_h.to(tl.int64)
        offs_d64 = offs_d.to(tl.int64)

        q_base = Q_ptr + b64 * sq_b + g64 * sq_g + qi64 * sq_t
        q_tile = tl.load(
            q_base + offs_h64[:, None] * sq_h + offs_d64[None, :] * sq_d,
            mask=h_mask[:, None],
            other=0.0,
        )
        k_base = K_ptr + b64 * sk_b + g64 * sk_g
        v_base = V_ptr + b64 * sv_b + g64 * sv_g

        m_i = tl.full([BLOCK_H], -float("inf"), dtype=tl.float32)
        l_i = tl.zeros([BLOCK_H], dtype=tl.float32)
        acc = tl.zeros([BLOCK_H, D_HEAD], dtype=tl.float32)

        qpos = Q_BASE + qi
        win_lo = qpos - WIN + 1

        # Phase 1: Local window [max(0, win_lo), min(TK, qpos + 1))
        w_start = tl.maximum(0, win_lo)
        w_end = tl.minimum(TK, qpos + 1)
        for k0 in range(w_start, w_end, BLOCK_K):
            k_offs = k0 + offs_k
            k_valid = k_offs < w_end
            k64 = k_offs.to(tl.int64)
            k_tile = tl.load(
                k_base + k64[:, None] * sk_t + offs_d64[None, :] * sk_d,
                mask=k_valid[:, None],
                other=0.0,
            )
            v_tile = tl.load(
                v_base + k64[:, None] * sv_t + offs_d64[None, :] * sv_d,
                mask=k_valid[:, None],
                other=0.0,
            )
            qk = tl.dot(q_tile, tl.trans(k_tile), out_dtype=tl.float32) * SCALE
            qk = tl.where(k_valid[None, :], qk, -float("inf"))
            m_ij = tl.max(qk, axis=1)
            m_new = tl.maximum(m_i, m_ij)
            safe_m_i = tl.where(m_i > -1e30, m_i, 0.0)
            safe_m_new = tl.where(m_new > -1e30, m_new, 0.0)
            alpha = tl.where(m_i > -1e30, tl.exp(safe_m_i - safe_m_new), 0.0)
            p = tl.where(k_valid[None, :], tl.exp(qk - safe_m_new[:, None]), 0.0)
            l_i = l_i * alpha + tl.sum(p, axis=1)
            acc = acc * alpha[:, None] + tl.dot(p.to(v_tile.dtype), v_tile, out_dtype=tl.float32)
            m_i = m_new

        # Phase 2: Sinks [0, min(NS, max(0, min(win_lo, qpos + 1))))
        s_end = tl.minimum(NS, tl.maximum(0, tl.minimum(win_lo, qpos + 1)))
        for k0 in range(0, s_end, BLOCK_K):
            k_offs = k0 + offs_k
            k_valid = k_offs < s_end
            k64 = k_offs.to(tl.int64)
            k_tile = tl.load(
                k_base + k64[:, None] * sk_t + offs_d64[None, :] * sk_d,
                mask=k_valid[:, None],
                other=0.0,
            )
            v_tile = tl.load(
                v_base + k64[:, None] * sv_t + offs_d64[None, :] * sv_d,
                mask=k_valid[:, None],
                other=0.0,
            )
            qk = tl.dot(q_tile, tl.trans(k_tile), out_dtype=tl.float32) * SCALE
            qk = tl.where(k_valid[None, :], qk, -float("inf"))
            m_ij = tl.max(qk, axis=1)
            m_new = tl.maximum(m_i, m_ij)
            safe_m_i = tl.where(m_i > -1e30, m_i, 0.0)
            safe_m_new = tl.where(m_new > -1e30, m_new, 0.0)
            alpha = tl.where(m_i > -1e30, tl.exp(safe_m_i - safe_m_new), 0.0)
            p = tl.where(k_valid[None, :], tl.exp(qk - safe_m_new[:, None]), 0.0)
            l_i = l_i * alpha + tl.sum(p, axis=1)
            acc = acc * alpha[:, None] + tl.dot(p.to(v_tile.dtype), v_tile, out_dtype=tl.float32)
            m_i = m_new

        # Phase 3: Packed deduplicated bucket members (strictly disjoint from window & sinks)
        if HAS_BUCKETS:
            mem_base = MEM_ptr + b64 * sm_b + g64 * sm_g + qi64 * sm_t
            for j0 in range(0, L_PACK, BLOCK_K):
                j_offs = j0 + offs_k
                k_ids = tl.load(
                    mem_base + j_offs.to(tl.int64) * sm_l,
                    mask=j_offs < L_PACK,
                    other=-1,
                )
                valid_b = (k_ids >= 0) & (k_ids < TK)
                if tl.sum(valid_b.to(tl.int32), axis=0) > 0:
                    k_safe = tl.where(valid_b, k_ids, 0).to(tl.int64)
                    k_tile = tl.load(
                        k_base + k_safe[:, None] * sk_t + offs_d64[None, :] * sk_d,
                        mask=valid_b[:, None],
                        other=0.0,
                    )
                    v_tile = tl.load(
                        v_base + k_safe[:, None] * sv_t + offs_d64[None, :] * sv_d,
                        mask=valid_b[:, None],
                        other=0.0,
                    )
                    qk = tl.dot(q_tile, tl.trans(k_tile), out_dtype=tl.float32) * SCALE
                    qk = tl.where(valid_b[None, :], qk, -float("inf"))
                    m_ij = tl.max(qk, axis=1)
                    m_new = tl.maximum(m_i, m_ij)
                    safe_m_i = tl.where(m_i > -1e30, m_i, 0.0)
                    safe_m_new = tl.where(m_new > -1e30, m_new, 0.0)
                    alpha = tl.where(m_i > -1e30, tl.exp(safe_m_i - safe_m_new), 0.0)
                    p = tl.where(valid_b[None, :], tl.exp(qk - safe_m_new[:, None]), 0.0)
                    l_i = l_i * alpha + tl.sum(p, axis=1)
                    acc = acc * alpha[:, None] + tl.dot(p.to(v_tile.dtype), v_tile, out_dtype=tl.float32)
                    m_i = m_new

        inv_l = tl.where(l_i > 0.0, 1.0 / l_i, 0.0)
        o_tile = (acc * inv_l[:, None]).to(q_tile.dtype)
        o_base = O_ptr + b64 * so_b + g64 * so_g + qi64 * so_t
        tl.store(
            o_base + offs_h64[:, None] * so_h + offs_d64[None, :] * so_d,
            o_tile,
            mask=h_mask[:, None],
        )
        lse_val = tl.where(l_i > 0.0, m_i + tl.log(tl.where(l_i > 0.0, l_i, 1.0)), -1e30)
        lse_base = LSE_ptr + b64 * sl_b + g64 * sl_g + qi64 * sl_t
        tl.store(lse_base + offs_h64 * sl_h, lse_val, mask=h_mask)

    @triton.jit
    def _lsa_sparse_bwd_delta_kernel(
        O_ptr,
        DO_ptr,
        DELTA_ptr,
        so_b,
        so_g,
        so_h,
        so_t,
        so_d,
        sdo_b,
        sdo_g,
        sdo_h,
        sdo_t,
        sdo_d,
        sd_b,
        sd_g,
        sd_h,
        sd_t,
        G,
        HG,
        TQ,
        D_HEAD: tl.constexpr,
        BLOCK_T: tl.constexpr,
    ):
        pid_t = tl.program_id(0)
        bgh = tl.program_id(1)
        h_idx = bgh % HG
        bg = bgh // HG
        g_idx = bg % G
        b_idx = bg // G

        b64 = b_idx.to(tl.int64)
        g64 = g_idx.to(tl.int64)
        h64 = h_idx.to(tl.int64)

        offs_t = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
        offs_d = tl.arange(0, D_HEAD)
        t_mask = offs_t < TQ
        t64 = offs_t.to(tl.int64)
        d64 = offs_d.to(tl.int64)

        o_tile = tl.load(
            O_ptr + b64 * so_b + g64 * so_g + h64 * so_h + t64[:, None] * so_t + d64[None, :] * so_d,
            mask=t_mask[:, None],
            other=0.0,
        ).to(tl.float32)
        do_tile = tl.load(
            DO_ptr + b64 * sdo_b + g64 * sdo_g + h64 * sdo_h + t64[:, None] * sdo_t + d64[None, :] * sdo_d,
            mask=t_mask[:, None],
            other=0.0,
        ).to(tl.float32)
        delta = tl.sum(o_tile * do_tile, axis=1)
        tl.store(
            DELTA_ptr + b64 * sd_b + g64 * sd_g + h64 * sd_h + t64 * sd_t,
            delta,
            mask=t_mask,
        )

    @triton.jit
    def _lsa_sparse_attn_bwd_dq_and_buckets_kernel(
        Q_ptr,
        K_ptr,
        V_ptr,
        MEM_ptr,
        DO_ptr,
        LSE_ptr,
        DELTA_ptr,
        DQ_ptr,
        DK_ptr,
        DV_ptr,
        sq_b,
        sq_g,
        sq_h,
        sq_t,
        sq_d,
        sk_b,
        sk_g,
        sk_t,
        sk_d,
        sv_b,
        sv_g,
        sv_t,
        sv_d,
        sm_b,
        sm_g,
        sm_t,
        sm_l,
        sdo_b,
        sdo_g,
        sdo_h,
        sdo_t,
        sdo_d,
        sl_b,
        sl_g,
        sl_h,
        sl_t,
        sdq_b,
        sdq_g,
        sdq_h,
        sdq_t,
        sdq_d,
        sdk_b,
        sdk_g,
        sdk_t,
        sdk_d,
        G,
        TQ,
        TK,
        Q_BASE,
        WIN,
        NS,
        L_PACK,
        SCALE,
        HG: tl.constexpr,
        D_HEAD: tl.constexpr,
        BLOCK_H: tl.constexpr,
        BLOCK_K: tl.constexpr,
        HAS_BUCKETS: tl.constexpr,
        COMPUTE_DQ: tl.constexpr,
        COMPUTE_DKV: tl.constexpr,
    ):
        qi = tl.program_id(0)
        bg = tl.program_id(1)
        b_idx = bg // G
        g_idx = bg % G

        b64 = b_idx.to(tl.int64)
        g64 = g_idx.to(tl.int64)
        qi64 = qi.to(tl.int64)

        offs_h = tl.arange(0, BLOCK_H)
        offs_d = tl.arange(0, D_HEAD)
        offs_k = tl.arange(0, BLOCK_K)
        h_mask = offs_h < HG
        offs_h64 = offs_h.to(tl.int64)
        offs_d64 = offs_d.to(tl.int64)

        q_base = Q_ptr + b64 * sq_b + g64 * sq_g + qi64 * sq_t
        q_tile = tl.load(
            q_base + offs_h64[:, None] * sq_h + offs_d64[None, :] * sq_d,
            mask=h_mask[:, None],
            other=0.0,
        )
        do_base = DO_ptr + b64 * sdo_b + g64 * sdo_g + qi64 * sdo_t
        do_tile = tl.load(
            do_base + offs_h64[:, None] * sdo_h + offs_d64[None, :] * sdo_d,
            mask=h_mask[:, None],
            other=0.0,
        )
        lse_base = LSE_ptr + b64 * sl_b + g64 * sl_g + qi64 * sl_t
        lse_h = tl.load(lse_base + offs_h64 * sl_h, mask=h_mask, other=0.0)
        delta_base = DELTA_ptr + b64 * sl_b + g64 * sl_g + qi64 * sl_t
        delta_h = tl.load(delta_base + offs_h64 * sl_h, mask=h_mask, other=0.0)

        k_base = K_ptr + b64 * sk_b + g64 * sk_g
        v_base = V_ptr + b64 * sv_b + g64 * sv_g

        dq_acc = tl.zeros([BLOCK_H, D_HEAD], dtype=tl.float32)

        qpos = Q_BASE + qi
        win_lo = qpos - WIN + 1

        # Window + sink loops only needed for dQ (key-parallel kernels B2/B3 compute window & sink dK/dV atomic-free)
        if COMPUTE_DQ:
            w_start = tl.maximum(0, win_lo)
            w_end = tl.minimum(TK, qpos + 1)
            for k0 in range(w_start, w_end, BLOCK_K):
                k_offs = k0 + offs_k
                k_valid = k_offs < w_end
                k64 = k_offs.to(tl.int64)
                k_tile = tl.load(
                    k_base + k64[:, None] * sk_t + offs_d64[None, :] * sk_d,
                    mask=k_valid[:, None],
                    other=0.0,
                )
                v_tile = tl.load(
                    v_base + k64[:, None] * sv_t + offs_d64[None, :] * sv_d,
                    mask=k_valid[:, None],
                    other=0.0,
                )
                qk = tl.dot(q_tile, tl.trans(k_tile), out_dtype=tl.float32) * SCALE
                dp = tl.dot(do_tile, tl.trans(v_tile), out_dtype=tl.float32)
                valid_hk = h_mask[:, None] & k_valid[None, :]
                p = tl.where(valid_hk, tl.exp(qk - lse_h[:, None]), 0.0)
                ds = tl.where(valid_hk, p * (dp - delta_h[:, None]), 0.0)
                dq_acc += tl.dot(ds.to(k_tile.dtype), k_tile, out_dtype=tl.float32)

            s_end = tl.minimum(NS, tl.maximum(0, tl.minimum(win_lo, qpos + 1)))
            for k0 in range(0, s_end, BLOCK_K):
                k_offs = k0 + offs_k
                k_valid = k_offs < s_end
                k64 = k_offs.to(tl.int64)
                k_tile = tl.load(
                    k_base + k64[:, None] * sk_t + offs_d64[None, :] * sk_d,
                    mask=k_valid[:, None],
                    other=0.0,
                )
                v_tile = tl.load(
                    v_base + k64[:, None] * sv_t + offs_d64[None, :] * sv_d,
                    mask=k_valid[:, None],
                    other=0.0,
                )
                qk = tl.dot(q_tile, tl.trans(k_tile), out_dtype=tl.float32) * SCALE
                dp = tl.dot(do_tile, tl.trans(v_tile), out_dtype=tl.float32)
                valid_hk = h_mask[:, None] & k_valid[None, :]
                p = tl.where(valid_hk, tl.exp(qk - lse_h[:, None]), 0.0)
                ds = tl.where(valid_hk, p * (dp - delta_h[:, None]), 0.0)
                dq_acc += tl.dot(ds.to(k_tile.dtype), k_tile, out_dtype=tl.float32)

        if HAS_BUCKETS:
            mem_base = MEM_ptr + b64 * sm_b + g64 * sm_g + qi64 * sm_t
            dk_base = DK_ptr + b64 * sdk_b + g64 * sdk_g
            dv_base = DV_ptr + b64 * sdk_b + g64 * sdk_g
            for j0 in range(0, L_PACK, BLOCK_K):
                j_offs = j0 + offs_k
                k_ids = tl.load(
                    mem_base + j_offs.to(tl.int64) * sm_l,
                    mask=j_offs < L_PACK,
                    other=-1,
                )
                valid_b = (k_ids >= 0) & (k_ids < TK)
                if tl.sum(valid_b.to(tl.int32), axis=0) > 0:
                    k_safe = tl.where(valid_b, k_ids, 0).to(tl.int64)
                    k_tile = tl.load(
                        k_base + k_safe[:, None] * sk_t + offs_d64[None, :] * sk_d,
                        mask=valid_b[:, None],
                        other=0.0,
                    )
                    v_tile = tl.load(
                        v_base + k_safe[:, None] * sv_t + offs_d64[None, :] * sv_d,
                        mask=valid_b[:, None],
                        other=0.0,
                    )
                    qk = tl.dot(q_tile, tl.trans(k_tile), out_dtype=tl.float32) * SCALE
                    dp = tl.dot(do_tile, tl.trans(v_tile), out_dtype=tl.float32)
                    valid_hk = h_mask[:, None] & valid_b[None, :]
                    p = tl.where(valid_hk, tl.exp(qk - lse_h[:, None]), 0.0)
                    ds = tl.where(valid_hk, p * (dp - delta_h[:, None]), 0.0)
                    if COMPUTE_DQ:
                        dq_acc += tl.dot(ds.to(k_tile.dtype), k_tile, out_dtype=tl.float32)
                    if COMPUTE_DKV:
                        # Reduce all Hg query heads via Tensor Cores before atomic scatter-add (8x fewer atomics)
                        dv_b = tl.dot(tl.trans(p.to(do_tile.dtype)), do_tile, out_dtype=tl.float32)
                        tl.atomic_add(
                            dv_base + k_safe[:, None] * sdk_t + offs_d64[None, :] * sdk_d,
                            dv_b,
                            mask=valid_b[:, None],
                        )
                        dk_b = tl.dot(tl.trans(ds.to(q_tile.dtype)), q_tile, out_dtype=tl.float32) * SCALE
                        tl.atomic_add(
                            dk_base + k_safe[:, None] * sdk_t + offs_d64[None, :] * sdk_d,
                            dk_b,
                            mask=valid_b[:, None],
                        )

        if COMPUTE_DQ:
            dq_base = DQ_ptr + b64 * sdq_b + g64 * sdq_g + qi64 * sdq_t
            tl.store(
                dq_base + offs_h64[:, None] * sdq_h + offs_d64[None, :] * sdq_d,
                (dq_acc * SCALE).to(q_tile.dtype),
                mask=h_mask[:, None],
            )

    @triton.jit
    def _lsa_sparse_attn_bwd_dkv_win_kernel(
        Q_ptr,
        K_ptr,
        V_ptr,
        DO_ptr,
        LSE_ptr,
        DELTA_ptr,
        DK_ptr,
        DV_ptr,
        sq_b,
        sq_g,
        sq_h,
        sq_t,
        sq_d,
        sk_b,
        sk_g,
        sk_t,
        sk_d,
        sv_b,
        sv_g,
        sv_t,
        sv_d,
        sdo_b,
        sdo_g,
        sdo_h,
        sdo_t,
        sdo_d,
        sl_b,
        sl_g,
        sl_h,
        sl_t,
        sdk_b,
        sdk_g,
        sdk_t,
        sdk_d,
        G,
        TQ,
        K_WIN_START,
        K_WIN_END,
        Q_BASE,
        WIN,
        SCALE,
        HG: tl.constexpr,
        D_HEAD: tl.constexpr,
        BLOCK_K: tl.constexpr,
        BLOCK_Q: tl.constexpr,
    ):
        pid_k = tl.program_id(0)
        bg = tl.program_id(1)
        b_idx = bg // G
        g_idx = bg % G

        b64 = b_idx.to(tl.int64)
        g64 = g_idx.to(tl.int64)

        k0 = K_WIN_START + pid_k * BLOCK_K
        offs_k = k0 + tl.arange(0, BLOCK_K)
        offs_d = tl.arange(0, D_HEAD)
        offs_q_rel = tl.arange(0, BLOCK_Q)
        k_valid = offs_k < K_WIN_END
        k64 = offs_k.to(tl.int64)
        d64 = offs_d.to(tl.int64)

        k_base = K_ptr + b64 * sk_b + g64 * sk_g
        v_base = V_ptr + b64 * sv_b + g64 * sv_g
        k_tile = tl.load(k_base + k64[:, None] * sk_t + d64[None, :] * sk_d, mask=k_valid[:, None], other=0.0)
        v_tile = tl.load(v_base + k64[:, None] * sv_t + d64[None, :] * sv_d, mask=k_valid[:, None], other=0.0)

        dk_acc = tl.zeros([BLOCK_K, D_HEAD], dtype=tl.float32)
        dv_acc = tl.zeros([BLOCK_K, D_HEAD], dtype=tl.float32)

        qi_start = tl.maximum(0, k0 - Q_BASE)
        qi_end = tl.minimum(TQ, k0 + BLOCK_K + WIN - 1 - Q_BASE)

        q_bg = Q_ptr + b64 * sq_b + g64 * sq_g
        do_bg = DO_ptr + b64 * sdo_b + g64 * sdo_g
        lse_bg = LSE_ptr + b64 * sl_b + g64 * sl_g
        del_bg = DELTA_ptr + b64 * sl_b + g64 * sl_g

        for q0 in range(qi_start, qi_end, BLOCK_Q):
            q_offs = q0 + offs_q_rel
            q_valid = q_offs < qi_end
            q64 = q_offs.to(tl.int64)
            qpos = Q_BASE + q_offs
            m_win = (
                q_valid[:, None]
                & k_valid[None, :]
                & (offs_k[None, :] <= qpos[:, None])
                & (offs_k[None, :] >= (qpos[:, None] - WIN + 1))
            )
            zero64 = b64 * 0
            for h in range(HG):
                h64 = zero64 + h
                q_tile = tl.load(
                    q_bg + h64 * sq_h + q64[:, None] * sq_t + d64[None, :] * sq_d,
                    mask=q_valid[:, None],
                    other=0.0,
                )
                do_tile = tl.load(
                    do_bg + h64 * sdo_h + q64[:, None] * sdo_t + d64[None, :] * sdo_d,
                    mask=q_valid[:, None],
                    other=0.0,
                )
                lse_q = tl.load(lse_bg + h64 * sl_h + q64 * sl_t, mask=q_valid, other=0.0)
                del_q = tl.load(del_bg + h64 * sl_h + q64 * sl_t, mask=q_valid, other=0.0)
                qk = tl.dot(q_tile, tl.trans(k_tile), out_dtype=tl.float32) * SCALE
                dp = tl.dot(do_tile, tl.trans(v_tile), out_dtype=tl.float32)
                p = tl.where(m_win, tl.exp(qk - lse_q[:, None]), 0.0)
                ds = tl.where(m_win, p * (dp - del_q[:, None]), 0.0)
                dv_acc += tl.dot(tl.trans(p.to(do_tile.dtype)), do_tile, out_dtype=tl.float32)
                dk_acc += tl.dot(tl.trans(ds.to(q_tile.dtype)), q_tile, out_dtype=tl.float32)

        dk_ptrs = DK_ptr + b64 * sdk_b + g64 * sdk_g + k64[:, None] * sdk_t + d64[None, :] * sdk_d
        dv_ptrs = DV_ptr + b64 * sdk_b + g64 * sdk_g + k64[:, None] * sdk_t + d64[None, :] * sdk_d
        dk_cur = tl.load(dk_ptrs, mask=k_valid[:, None], other=0.0)
        dv_cur = tl.load(dv_ptrs, mask=k_valid[:, None], other=0.0)
        tl.store(dk_ptrs, dk_cur + dk_acc * SCALE, mask=k_valid[:, None])
        tl.store(dv_ptrs, dv_cur + dv_acc, mask=k_valid[:, None])

    @triton.jit
    def _lsa_sparse_attn_bwd_dkv_sink_kernel(
        Q_ptr,
        K_ptr,
        V_ptr,
        DO_ptr,
        LSE_ptr,
        DELTA_ptr,
        DK_ptr,
        DV_ptr,
        sq_b,
        sq_g,
        sq_h,
        sq_t,
        sq_d,
        sk_b,
        sk_g,
        sk_t,
        sk_d,
        sv_b,
        sv_g,
        sv_t,
        sv_d,
        sdo_b,
        sdo_g,
        sdo_h,
        sdo_t,
        sdo_d,
        sl_b,
        sl_g,
        sl_h,
        sl_t,
        sdk_b,
        sdk_g,
        sdk_t,
        sdk_d,
        G,
        TQ,
        NS_EFF,
        Q_BASE,
        WIN,
        SPLIT_Q,
        SCALE,
        HG: tl.constexpr,
        D_HEAD: tl.constexpr,
        BLOCK_K: tl.constexpr,
        BLOCK_Q: tl.constexpr,
    ):
        pid_k = tl.program_id(0)
        pid_s = tl.program_id(1)
        bg = tl.program_id(2)
        b_idx = bg // G
        g_idx = bg % G

        k0 = pid_k * BLOCK_K
        qi_min = tl.maximum(0, k0 + WIN - Q_BASE)
        n_rem = TQ - qi_min
        chunk = tl.maximum(1, (n_rem + SPLIT_Q - 1) // SPLIT_Q)
        s_start = qi_min + pid_s * chunk
        s_end = tl.minimum(TQ, s_start + chunk)
        if (n_rem > 0) & (s_start < s_end):
            b64 = b_idx.to(tl.int64)
            g64 = g_idx.to(tl.int64)
            zero64 = b64 * 0

            offs_k = k0 + tl.arange(0, BLOCK_K)
            offs_d = tl.arange(0, D_HEAD)
            offs_q_rel = tl.arange(0, BLOCK_Q)
            k_valid = offs_k < NS_EFF
            k64 = offs_k.to(tl.int64)
            d64 = offs_d.to(tl.int64)

            k_base = K_ptr + b64 * sk_b + g64 * sk_g
            v_base = V_ptr + b64 * sv_b + g64 * sv_g
            k_tile = tl.load(k_base + k64[:, None] * sk_t + d64[None, :] * sk_d, mask=k_valid[:, None], other=0.0)
            v_tile = tl.load(v_base + k64[:, None] * sv_t + d64[None, :] * sv_d, mask=k_valid[:, None], other=0.0)

            dk_acc = tl.zeros([BLOCK_K, D_HEAD], dtype=tl.float32)
            dv_acc = tl.zeros([BLOCK_K, D_HEAD], dtype=tl.float32)

            q_bg = Q_ptr + b64 * sq_b + g64 * sq_g
            do_bg = DO_ptr + b64 * sdo_b + g64 * sdo_g
            lse_bg = LSE_ptr + b64 * sl_b + g64 * sl_g
            del_bg = DELTA_ptr + b64 * sl_b + g64 * sl_g

            for q0 in range(s_start, s_end, BLOCK_Q):
                q_offs = q0 + offs_q_rel
                q_valid = q_offs < s_end
                q64 = q_offs.to(tl.int64)
                qpos = Q_BASE + q_offs
                m_sink = (
                    q_valid[:, None]
                    & k_valid[None, :]
                    & (offs_k[None, :] < (qpos[:, None] - WIN + 1))
                    & (offs_k[None, :] <= qpos[:, None])
                )
                for h in range(HG):
                    h64 = zero64 + h
                    q_tile = tl.load(
                        q_bg + h64 * sq_h + q64[:, None] * sq_t + d64[None, :] * sq_d,
                        mask=q_valid[:, None],
                        other=0.0,
                    )
                    do_tile = tl.load(
                        do_bg + h64 * sdo_h + q64[:, None] * sdo_t + d64[None, :] * sdo_d,
                        mask=q_valid[:, None],
                        other=0.0,
                    )
                    lse_q = tl.load(lse_bg + h64 * sl_h + q64 * sl_t, mask=q_valid, other=0.0)
                    del_q = tl.load(del_bg + h64 * sl_h + q64 * sl_t, mask=q_valid, other=0.0)
                    qk = tl.dot(q_tile, tl.trans(k_tile), out_dtype=tl.float32) * SCALE
                    dp = tl.dot(do_tile, tl.trans(v_tile), out_dtype=tl.float32)
                    p = tl.where(m_sink, tl.exp(qk - lse_q[:, None]), 0.0)
                    ds = tl.where(m_sink, p * (dp - del_q[:, None]), 0.0)
                    dv_acc += tl.dot(tl.trans(p.to(do_tile.dtype)), do_tile, out_dtype=tl.float32)
                    dk_acc += tl.dot(tl.trans(ds.to(q_tile.dtype)), q_tile, out_dtype=tl.float32)

            dk_ptrs = DK_ptr + b64 * sdk_b + g64 * sdk_g + k64[:, None] * sdk_t + d64[None, :] * sdk_d
            dv_ptrs = DV_ptr + b64 * sdk_b + g64 * sdk_g + k64[:, None] * sdk_t + d64[None, :] * sdk_d
            tl.atomic_add(dk_ptrs, dk_acc * SCALE, mask=k_valid[:, None])
            tl.atomic_add(dv_ptrs, dv_acc, mask=k_valid[:, None])

    @triton.jit
    def _lsa_distill_lse_kernel(
        Q_ptr,
        K_ptr,
        QPOS_ptr,
        LSE_ptr,
        sq_b,
        sq_g,
        sq_h,
        sq_t,
        sq_d,
        sk_b,
        sk_g,
        sk_t,
        sk_d,
        sl_b,
        sl_g,
        sl_h,
        sl_t,
        G,
        HG,
        NQ,
        TK,
        SCALE,
        D_HEAD: tl.constexpr,
        BLOCK_Q: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        pid_q = tl.program_id(0)
        bgh = tl.program_id(1)
        h_idx = bgh % HG
        bg = bgh // HG
        g_idx = bg % G
        b_idx = bg // G

        b64 = b_idx.to(tl.int64)
        g64 = g_idx.to(tl.int64)
        h64 = h_idx.to(tl.int64)

        q0 = pid_q * BLOCK_Q
        offs_q = q0 + tl.arange(0, BLOCK_Q)
        offs_d = tl.arange(0, D_HEAD)
        offs_k = tl.arange(0, BLOCK_K)
        q_valid = offs_q < NQ
        q64 = offs_q.to(tl.int64)
        d64 = offs_d.to(tl.int64)

        qpos = tl.load(QPOS_ptr + q64, mask=q_valid, other=-1)
        qpos_max = tl.max(qpos, axis=0)
        q_tile = tl.load(
            Q_ptr + b64 * sq_b + g64 * sq_g + h64 * sq_h + q64[:, None] * sq_t + d64[None, :] * sq_d,
            mask=q_valid[:, None],
            other=0.0,
        )
        k_base = K_ptr + b64 * sk_b + g64 * sk_g

        m_i = tl.full([BLOCK_Q], -float("inf"), dtype=tl.float32)
        l_i = tl.zeros([BLOCK_Q], dtype=tl.float32)
        k_end = tl.minimum(TK, qpos_max + 1)
        for k0 in range(0, k_end, BLOCK_K):
            k_offs = k0 + offs_k
            k_valid = k_offs < k_end
            k64 = k_offs.to(tl.int64)
            k_tile = tl.load(
                k_base + k64[:, None] * sk_t + d64[None, :] * sk_d,
                mask=k_valid[:, None],
                other=0.0,
            )
            qk = tl.dot(q_tile, tl.trans(k_tile), out_dtype=tl.float32) * SCALE
            causal = q_valid[:, None] & k_valid[None, :] & (k_offs[None, :] <= qpos[:, None])
            qk = tl.where(causal, qk, -float("inf"))
            m_ij = tl.max(qk, axis=1)
            m_new = tl.maximum(m_i, m_ij)
            safe_m_i = tl.where(m_i > -1e30, m_i, 0.0)
            safe_m_new = tl.where(m_new > -1e30, m_new, 0.0)
            alpha = tl.where(m_i > -1e30, tl.exp(safe_m_i - safe_m_new), 0.0)
            p = tl.where(causal, tl.exp(qk - safe_m_new[:, None]), 0.0)
            l_i = l_i * alpha + tl.sum(p, axis=1)
            m_i = m_new

        lse_val = tl.where(l_i > 0.0, m_i + tl.log(tl.where(l_i > 0.0, l_i, 1.0)), -1e30)
        tl.store(
            LSE_ptr + b64 * sl_b + g64 * sl_g + h64 * sl_h + q64 * sl_t,
            lse_val,
            mask=q_valid,
        )

    @triton.jit
    def _lsa_distill_block_mass_kernel(
        Q_ptr,
        K_ptr,
        QPOS_ptr,
        LSE_ptr,
        SLAB_ptr,
        MASS_ptr,
        sq_b,
        sq_g,
        sq_h,
        sq_t,
        sq_d,
        sk_b,
        sk_g,
        sk_t,
        sk_d,
        sl_b,
        sl_g,
        sl_h,
        sl_t,
        ss_b,
        ss_g,
        ss_nb,
        ss_u,
        sm_b,
        sm_g,
        sm_q,
        sm_s,
        G,
        NQ,
        NB,
        BLOCK_SIZE,
        SPB,
        SCALE,
        HG: tl.constexpr,
        D_HEAD: tl.constexpr,
        POOL: tl.constexpr,
        N_UNITS: tl.constexpr,
        BLOCK_Q: tl.constexpr,
        BLOCK_U: tl.constexpr,
        SPB_PAD: tl.constexpr,
    ):
        blk = tl.program_id(0)
        pid_q = tl.program_id(1)
        bg = tl.program_id(2)
        b_idx = bg // G
        g_idx = bg % G

        q0 = pid_q * BLOCK_Q
        offs_q = q0 + tl.arange(0, BLOCK_Q)
        q_valid = offs_q < NQ
        q64 = offs_q.to(tl.int64)
        qpos = tl.load(QPOS_ptr + q64, mask=q_valid, other=0)
        valid_q_blk = q_valid & ((qpos // BLOCK_SIZE) > blk)
        if tl.sum(valid_q_blk.to(tl.int32), axis=0) > 0:
            b64 = b_idx.to(tl.int64)
            g64 = g_idx.to(tl.int64)
            blk64 = blk.to(tl.int64)
            zero64 = b64 * 0

            offs_d = tl.arange(0, D_HEAD)
            d64 = offs_d.to(tl.int64)
            offs_u = tl.arange(0, BLOCK_U)
            offs_k_rel = tl.arange(0, BLOCK_U * POOL)
            offs_s = tl.arange(0, SPB_PAD)

            q_bg = Q_ptr + b64 * sq_b + g64 * sq_g
            lse_bg = LSE_ptr + b64 * sl_b + g64 * sl_g
            k_base = K_ptr + b64 * sk_b + g64 * sk_g
            slab_base = SLAB_ptr + b64 * ss_b + g64 * ss_g + blk64 * ss_nb

            mass_acc = tl.zeros([BLOCK_Q, SPB_PAD], dtype=tl.float32)
            blk_k_base = blk64 * BLOCK_SIZE

            for u0 in range(0, N_UNITS, BLOCK_U):
                u_offs = u0 + offs_u
                u_slots = tl.load(slab_base + u_offs.to(tl.int64) * ss_u)
                one_hot = (u_slots[:, None] == offs_s[None, :]).to(tl.float32)

                k_pos64 = blk_k_base + (u0 * POOL + offs_k_rel).to(tl.int64)
                k_tile = tl.load(k_base + k_pos64[:, None] * sk_t + d64[None, :] * sk_d)

                unit_prob = tl.zeros([BLOCK_Q, BLOCK_U], dtype=tl.float32)
                for h in range(HG):
                    h64 = zero64 + h
                    q_tile = tl.load(
                        q_bg + h64 * sq_h + q64[:, None] * sq_t + d64[None, :] * sq_d,
                        mask=valid_q_blk[:, None],
                        other=0.0,
                    )
                    lse_q = tl.load(lse_bg + h64 * sl_h + q64 * sl_t, mask=valid_q_blk, other=0.0)
                    qk = tl.dot(q_tile, tl.trans(k_tile), out_dtype=tl.float32) * SCALE
                    p = tl.where(valid_q_blk[:, None], tl.exp(qk - lse_q[:, None]), 0.0)
                    if POOL == 1:
                        unit_prob += p
                    else:
                        unit_prob += tl.sum(tl.reshape(p, (BLOCK_Q, BLOCK_U, POOL)), axis=2)

                mass_acc += tl.dot(unit_prob, one_hot, input_precision="ieee")

            inv_hg = 1.0 / HG
            s_mask = offs_s < SPB
            slot_offs64 = blk64 * SPB + offs_s.to(tl.int64)
            m_ptrs = MASS_ptr + b64 * sm_b + g64 * sm_g + q64[:, None] * sm_q + slot_offs64[None, :] * sm_s
            tl.store(m_ptrs, mass_acc * inv_hg, mask=q_valid[:, None] & s_mask[None, :])


class _TritonSparseGatherAttnFn(torch.autograd.Function):
    """Fused zero-materialization Triton Sparse-Gather FlashAttention (`per_head=False`, CUDA bf16/fp16)."""

    @staticmethod
    def forward(  # type: ignore[override]
        ctx: Any,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mem_idx: torch.Tensor,
        q_base_pos: int,
        win: int,
        ns: int,
        scaling: float,
    ) -> torch.Tensor:
        qc = q.contiguous()
        kc = k.contiguous()
        vc = v.contiguous()
        mc = mem_idx.contiguous()
        b, g, hg, tq, dh = qc.shape
        tk = kc.shape[2]
        l_pack = int(mc.shape[-1])
        has_buckets = l_pack > 0
        mem_arg = mc if has_buckets else torch.empty((1,), device=qc.device, dtype=torch.int32)
        sm_b, sm_g, sm_t, sm_l = mc.stride() if has_buckets else (0, 0, 0, 0)

        out = torch.empty_like(qc)
        lse = torch.empty((b, g, hg, tq), device=qc.device, dtype=torch.float32)

        block_h = max(16, triton.next_power_of_2(hg))
        block_k = 32
        grid = (tq, b * g)
        _lsa_sparse_attn_fwd_kernel[grid](
            qc,
            kc,
            vc,
            mem_arg,
            out,
            lse,
            *qc.stride(),
            *kc.stride(),
            *vc.stride(),
            sm_b,
            sm_g,
            sm_t,
            sm_l,
            *out.stride(),
            *lse.stride(),
            g,
            tq,
            tk,
            int(q_base_pos),
            int(win),
            int(ns),
            l_pack,
            float(scaling),
            HG=hg,
            D_HEAD=dh,
            BLOCK_H=block_h,
            BLOCK_K=block_k,
            HAS_BUCKETS=has_buckets,
            num_warps=4,
            num_stages=2,
        )

        ctx.save_for_backward(qc, kc, vc, mc, out, lse)
        ctx.q_base_pos = int(q_base_pos)
        ctx.win = int(win)
        ctx.ns = int(ns)
        ctx.scaling = float(scaling)
        return out

    @staticmethod
    def backward(ctx: Any, grad_out: torch.Tensor) -> tuple[torch.Tensor | None, ...]:  # type: ignore[override]
        qc, kc, vc, mc, out, lse = ctx.saved_tensors
        need_q = bool(ctx.needs_input_grad[0])
        need_k = bool(ctx.needs_input_grad[1])
        need_v = bool(ctx.needs_input_grad[2])
        need_kv = need_k or need_v
        if not (need_q or need_kv):
            return (None,) * 8

        do = grad_out.contiguous()
        b, g, hg, tq, dh = qc.shape
        tk = kc.shape[2]
        q_base_pos: int = ctx.q_base_pos
        win: int = ctx.win
        ns: int = ctx.ns
        scaling: float = ctx.scaling
        l_pack = int(mc.shape[-1])
        has_buckets = l_pack > 0
        mem_arg = mc if has_buckets else torch.empty((1,), device=qc.device, dtype=torch.int32)
        sm_b, sm_g, sm_t, sm_l = mc.stride() if has_buckets else (0, 0, 0, 0)

        delta = torch.empty_like(lse)
        block_t_del = 64
        _lsa_sparse_bwd_delta_kernel[(triton.cdiv(tq, block_t_del), b * g * hg)](
            out,
            do,
            delta,
            *out.stride(),
            *do.stride(),
            *delta.stride(),
            g,
            hg,
            tq,
            D_HEAD=dh,
            BLOCK_T=block_t_del,
            num_warps=4,
        )

        dq = torch.empty_like(qc) if need_q else None
        dq_arg = dq if dq is not None else torch.empty((1,), device=qc.device, dtype=qc.dtype)
        sdq = dq.stride() if dq is not None else (0, 0, 0, 0, 0)

        dk_fp32 = torch.zeros((b, g, tk, dh), device=qc.device, dtype=torch.float32) if need_kv else None
        dv_fp32 = torch.zeros((b, g, tk, dh), device=qc.device, dtype=torch.float32) if need_kv else None
        dk_arg = dk_fp32 if dk_fp32 is not None else torch.empty((1,), device=qc.device, dtype=torch.float32)
        dv_arg = dv_fp32 if dv_fp32 is not None else torch.empty((1,), device=qc.device, dtype=torch.float32)
        sdk = dk_fp32.stride() if dk_fp32 is not None else (0, 0, 0, 0)

        block_h = max(16, triton.next_power_of_2(hg))
        block_k = 32
        if need_q or (need_kv and has_buckets):
            _lsa_sparse_attn_bwd_dq_and_buckets_kernel[(tq, b * g)](
                qc,
                kc,
                vc,
                mem_arg,
                do,
                lse,
                delta,
                dq_arg,
                dk_arg,
                dv_arg,
                *qc.stride(),
                *kc.stride(),
                *vc.stride(),
                sm_b,
                sm_g,
                sm_t,
                sm_l,
                *do.stride(),
                *lse.stride(),
                *sdq,
                *sdk,
                g,
                tq,
                tk,
                q_base_pos,
                win,
                ns,
                l_pack,
                scaling,
                HG=hg,
                D_HEAD=dh,
                BLOCK_H=block_h,
                BLOCK_K=block_k,
                HAS_BUCKETS=has_buckets,
                COMPUTE_DQ=need_q,
                COMPUTE_DKV=need_kv,
                num_warps=4,
                num_stages=1,
            )

        if need_kv:
            assert dk_fp32 is not None and dv_fp32 is not None
            # Atomic-free key-parallel window dK/dV
            k_win_start = max(0, q_base_pos - win + 1)
            k_win_end = min(tk, q_base_pos + tq)
            if k_win_end > k_win_start:
                bk_w = 32
                bq_w = 32
                _lsa_sparse_attn_bwd_dkv_win_kernel[(triton.cdiv(k_win_end - k_win_start, bk_w), b * g)](
                    qc,
                    kc,
                    vc,
                    do,
                    lse,
                    delta,
                    dk_fp32,
                    dv_fp32,
                    *qc.stride(),
                    *kc.stride(),
                    *vc.stride(),
                    *do.stride(),
                    *lse.stride(),
                    *dk_fp32.stride(),
                    g,
                    tq,
                    k_win_start,
                    k_win_end,
                    q_base_pos,
                    win,
                    scaling,
                    HG=hg,
                    D_HEAD=dh,
                    BLOCK_K=bk_w,
                    BLOCK_Q=bq_w,
                    num_warps=8,
                    num_stages=2,
                )

            # Split-Q key-parallel sink dK/dV (32 atomics per sink key vs 32,768)
            ns_eff = min(ns, tk)
            if ns_eff > 0 and (q_base_pos + tq - 1) >= win:
                bk_s = 32
                bq_s = 32 if dh >= 256 else 64
                split_q = min(32, max(1, tq // 256))
                _lsa_sparse_attn_bwd_dkv_sink_kernel[(triton.cdiv(ns_eff, bk_s), split_q, b * g)](
                    qc,
                    kc,
                    vc,
                    do,
                    lse,
                    delta,
                    dk_fp32,
                    dv_fp32,
                    *qc.stride(),
                    *kc.stride(),
                    *vc.stride(),
                    *do.stride(),
                    *lse.stride(),
                    *dk_fp32.stride(),
                    g,
                    tq,
                    ns_eff,
                    q_base_pos,
                    win,
                    split_q,
                    scaling,
                    HG=hg,
                    D_HEAD=dh,
                    BLOCK_K=bk_s,
                    BLOCK_Q=bq_s,
                    num_warps=8,
                    num_stages=2,
                )

        dk_out = dk_fp32.to(kc.dtype) if (need_k and dk_fp32 is not None) else None
        dv_out = dv_fp32.to(vc.dtype) if (need_v and dv_fp32 is not None) else None
        return dq, dk_out, dv_out, None, None, None, None, None


@torch.no_grad()
def _build_sparse_mem_table(
    qz_span: torch.Tensor,
    qpos_span: torch.Tensor,
    index: BucketIndex | None,
    idx: LSAIndexer,
    *,
    tk: int,
    q_base_pos: int,
    cur_pass: int,
    record_tail_len: int,
    record_stats: bool,
) -> torch.Tensor:
    """Build `[B, G, T_span, L_pack]` int32 packed bucket-member table (`-1` = invalid), zero intermediate syncs."""
    cfg = idx.cfg
    b, g, t_span, _ = qz_span.shape
    dev = qz_span.device
    w = cfg.resolved_window()
    ns = min(cfg.sinks, tk)
    sync_free = _is_sync_restricted()

    n_hist_max = max(0, (q_base_pos + t_span - 1 - w + 1) - ns)
    if index is None or t_span == 0:
        if record_stats and not sync_free and not _in_backward() and t_span > 0:
            t_tail = min(t_span, max(1, record_tail_len))
            qp_t = qpos_span[-t_tail:]
            lo_t = max(0, q_base_pos + t_span - t_tail - w + 1)
            hi_t = q_base_pos + t_span
            kpos_t = torch.arange(lo_t, hi_t, device=dev)
            win_lo_t = qp_t - w + 1
            m_win_t = (kpos_t[None] <= qp_t[:, None]) & (kpos_t[None] >= win_lo_t[:, None])
            ns_t = min(cfg.sinks, hi_t)
            spos_t = torch.arange(ns_t, device=dev)
            m_sink_t = (spos_t[None] < win_lo_t[:, None]) & (spos_t[None] <= qp_t[:, None])
            kpq = float(m_win_t.sum() + m_sink_t.sum()) / max(1, t_tail)
            idx.last_stats["keys_per_query"] = kpq
            idx.last_stats[f"keys_per_query_p{cur_pass}"] = kpq
        return torch.empty((b, g, t_span, 0), device=dev, dtype=torch.int32)

    s_slots = index.centroids.shape[2]
    kk = min(cfg.num_select, s_slots)
    mk = index.members.shape[-1]
    kb_n = kk * mk
    eff_cap_max = min(_effective_member_cap(cfg, s_slots, kk, kb_n, tk), n_hist_max) if n_hist_max > 0 else 0
    l_pack_max = min(kb_n, eff_cap_max) if (eff_cap_max > 0 and kb_n > 0) else 0
    mem_signed = (
        torch.full((b, g, t_span, l_pack_max), -1, device=dev, dtype=torch.int32)
        if l_pack_max > 0
        else torch.empty((b, g, t_span, 0), device=dev, dtype=torch.int32)
    )

    sel_chunk = max(cfg.query_chunk, 1024)
    members_i32 = index.members.to(torch.int32) if (dev.type == "cuda" and tk < (1 << 30)) else index.members
    neg_one_i32 = torch.tensor(-1, device=dev, dtype=torch.int32)

    for c0 in range(0, t_span, sel_chunk):
        c1 = min(t_span, c0 + sel_chunk)
        c_len = c1 - c0
        is_last = c1 >= t_span
        n_hist_c = max(0, (q_base_pos + c1 - 1 - w + 1) - ns)
        if n_hist_c <= 0 and not (is_last and record_stats):
            continue

        qpos_c = qpos_span[c0:c1]
        scores = idx.slot_scores(qz_span[:, :, c0:c1], index, qpos_c)
        top = scores.topk(kk, dim=-1)
        finite_top = torch.isfinite(top.values)
        if is_last and record_stats and not _in_backward():
            sel_slots = torch.where(finite_top, top.indices, torch.full_like(top.indices, -1))
            t_tail = min(c_len, max(1, record_tail_len))
            idx.last_selected_buckets[cur_pass] = sel_slots[:, :, -t_tail:, :]

        if n_hist_c > 0 and l_pack_max > 0:
            win_lo_c = qpos_c - w + 1
            eff_cap_c = min(_effective_member_cap(cfg, s_slots, kk, kb_n, tk), n_hist_c)
            flat_sel = top.indices.reshape(b, g, c_len * kk, 1).expand(b, g, c_len * kk, mk)
            mem = members_i32.gather(2, flat_sel).view(b, g, c_len, kk, mk)
            mem = torch.where(finite_top.unsqueeze(-1), mem, torch.full_like(mem, -1)).flatten(-2)
            m_b = (mem >= ns) & (mem < win_lo_c[None, None, :, None])
            packed = _pack_valid_members(
                mem,
                m_b,
                tk=tk,
                eff_cap=eff_cap_c,
                sync_free=True,
                out_dtype=torch.int32,
            )
            if packed is not None:
                mem_final, m_final, _ = packed
                l_c = mem_final.shape[-1]
                mem_signed[:, :, c0:c1, :l_c] = torch.where(m_final, mem_final, neg_one_i32)

    if record_stats and not sync_free and not _in_backward():
        t_tail = min(t_span, max(1, record_tail_len))
        qp_t = qpos_span[-t_tail:]
        lo_t = max(0, q_base_pos + t_span - t_tail - w + 1)
        hi_t = q_base_pos + t_span
        kpos_t = torch.arange(lo_t, hi_t, device=dev)
        win_lo_t = qp_t - w + 1
        m_win_t = (kpos_t[None] <= qp_t[:, None]) & (kpos_t[None] >= win_lo_t[:, None])
        ns_t = min(cfg.sinks, hi_t)
        spos_t = torch.arange(ns_t, device=dev)
        m_sink_t = (spos_t[None] < win_lo_t[:, None]) & (spos_t[None] <= qp_t[:, None])
        n_sel_t = int((mem_signed[:, :, -t_tail:, :] >= 0).sum()) if l_pack_max > 0 else 0
        kpq = float(m_win_t.sum() + m_sink_t.sum()) / max(1, t_tail) + n_sel_t / max(1, b * g * t_tail)
        idx.last_stats["keys_per_query"] = kpq
        idx.last_stats[f"keys_per_query_p{cur_pass}"] = kpq

    return mem_signed


def _run_detached_distill_kl(
    qin: torch.Tensor,
    key: torch.Tensor,
    qpos_all: torch.Tensor,
    index: BucketIndex,
    idx: LSAIndexer,
    active_wq: nn.Parameter,
    scaling: float,
    cur_pass: int,
    caller_allow: torch.Tensor | None = None,
    caller_bias: torch.Tensor | None = None,
) -> None:
    """Fast exact detached-input KL distillation with analytic `(d_qz, d_cent)` and fused Triton teacher mass."""
    cfg = idx.cfg
    b, h, tq, dh = qin.shape
    g, tk = key.shape[1], key.shape[2]
    hg = h // g
    dev = qin.device
    q_base_pos = tk - tq

    # Avoid `torch.nonzero` GPU->CPU sync: `qpos_all[i] = q_base_pos + i` is strictly increasing.
    cand_start = max(0, cfg.block_size - q_base_pos)
    n_cand = max(0, tq - cand_start)
    if n_cand <= 0:
        return

    nq = max(1, int(n_cand * cfg.distill_query_frac))
    pick = (torch.randperm(n_cand, device=dev)[:nq] + cand_start).sort().values
    qpos_pick = qpos_all[pick]
    q_pick = qin[:, :, pick]
    allow_pick = _slice_caller_mask(caller_allow, pick)
    bias_pick = _slice_caller_mask(caller_bias, pick)

    need_wq = bool(active_wq.requires_grad)
    need_wk = bool(idx.wk.requires_grad)
    per_head = bool(cfg.per_head_select)
    s_slots = index.centroids.shape[2]
    spb = index.slots_per_block or (s_slots // max(1, index.n_blocks))
    nb = index.n_blocks

    can_triton_distill = (
        _TRITON_AVAILABLE
        and caller_allow is None
        and q_pick.is_cuda
        and q_pick.dtype in (torch.bfloat16, torch.float16)
        and key.dtype == q_pick.dtype
        and not per_head
        and dh in (32, 64, 128, 256)
        and hg <= 32
        and cfg.pool in (1, 2, 4, 8)
        and (cfg.block_size % (16 * cfg.pool) == 0)
        and index.slot_labels is not None
        and spb <= 128
        and nb > 0
    )

    with torch.autograd.graph.saved_tensors_hooks(lambda x: x.detach(), lambda x: x):
        with torch.no_grad():
            qz_all = idx.query_latent(q_pick, pass_idx=cur_pass)  # [B, G_or_H, Nq, d] (upcast dtype)
            acc_dt = qz_all.dtype
            cent = index.centroids.detach().to(acc_dt)  # [B, G, S, d]
            d_qz_all = torch.zeros_like(qz_all) if need_wq else None
            g_cent_sum = torch.zeros_like(cent) if need_wk else None
            loss_sum_t = torch.zeros((), device=dev, dtype=acc_dt)
            n_chunks_ok_t = torch.zeros((), device=dev, dtype=acc_dt)
            zero_acc = torch.zeros((), device=dev, dtype=acc_dt)

            dc = max(1, int(cfg.distill_chunk))
            if can_triton_distill:
                # Pass 1: exact causal LSE across all sampled queries in one fused Triton launch
                qg_pick = q_pick.view(b, g, hg, nq, dh).contiguous()
                kc = key.detach().contiguous()
                qpos_i32 = qpos_pick.to(torch.int32).contiguous()
                lse_all = torch.empty((b, g, hg, nq), device=dev, dtype=torch.float32)
                bq_lse = 64
                bk_lse = 64
                _lsa_distill_lse_kernel[(triton.cdiv(nq, bq_lse), b * g * hg)](
                    qg_pick,
                    kc,
                    qpos_i32,
                    lse_all,
                    *qg_pick.stride(),
                    *kc.stride(),
                    *lse_all.stride(),
                    g,
                    hg,
                    nq,
                    tk,
                    float(scaling),
                    D_HEAD=dh,
                    BLOCK_Q=bq_lse,
                    BLOCK_K=bk_lse,
                    num_warps=4,
                    num_stages=2,
                )
                assert index.slot_labels is not None
                slab_i32 = index.slot_labels.to(torch.int32).contiguous()
                spb_pad = max(16, triton.next_power_of_2(spb))
                n_units = cfg.block_size // cfg.pool
                # Outer chunk size aligned to `distill_chunk` (bounds `mass`/`logits` to ~33 MiB)
                q_outer = max(dc, (256 // dc) * dc) if dc <= 256 else dc
                inv_temp = 1.0 / float(cfg.temperature)
                for o0 in range(0, nq, q_outer):
                    o1 = min(nq, o0 + q_outer)
                    oc = o1 - o0
                    mass_c = torch.zeros((b, g, oc, s_slots), device=dev, dtype=torch.float32)
                    qg_o = qg_pick[:, :, :, o0:o1]
                    lse_o = lse_all[:, :, :, o0:o1]
                    qp_o = qpos_i32[o0:o1]
                    bq_m = 16
                    _lsa_distill_block_mass_kernel[(nb, triton.cdiv(oc, bq_m), b * g)](
                        qg_o,
                        kc,
                        qp_o,
                        lse_o,
                        slab_i32,
                        mass_c,
                        *qg_o.stride(),
                        *kc.stride(),
                        *lse_o.stride(),
                        *slab_i32.stride(),
                        *mass_c.stride(),
                        g,
                        oc,
                        nb,
                        cfg.block_size,
                        spb,
                        float(scaling),
                        HG=hg,
                        D_HEAD=dh,
                        POOL=cfg.pool,
                        N_UNITS=n_units,
                        BLOCK_Q=bq_m,
                        BLOCK_U=16,
                        SPB_PAD=spb_pad,
                        num_warps=4,
                        num_stages=2,
                    )
                    mass = mass_c.to(acc_dt)
                    tot = mass.sum(-1, keepdim=True)
                    target = mass / tot.clamp_min(1e-30)
                    qpos_o_long = qpos_pick[o0:o1]
                    frontier_o = (qpos_o_long // cfg.block_size) * cfg.block_size
                    elig_o = index.active[:, :, None, :] & (
                        index.slot_end[None, None, None, :] <= frontier_o[None, None, :, None]
                    )
                    ok_o = (frontier_o > 0)[None, None, :] & (tot.squeeze(-1) > 0)
                    qz_o = qz_all[:, :, o0:o1]
                    logits_o = (torch.matmul(qz_o, cent.transpose(-1, -2)) * inv_temp).masked_fill(
                        ~elig_o, float("-inf")
                    )
                    logp_o = torch.log_softmax(logits_o, dim=-1)
                    probs_o = torch.where(elig_o, torch.exp(logp_o), zero_acc)
                    logp_safe = torch.where(target > 0, logp_o, zero_acc)
                    kl_rows = (target * (torch.log(target.clamp_min(1e-30)) - logp_safe)).sum(-1)

                    # Exact chunk-mean weighting matching `for s0 in range(0, nq, distill_chunk)`
                    row_scale = torch.zeros_like(kl_rows)
                    for sub0 in range(0, oc, dc):
                        sub1 = min(oc, sub0 + dc)
                        ok_s = ok_o[:, :, sub0:sub1]
                        cnt_s = ok_s.sum().to(acc_dt)
                        has_s = (cnt_s > 0).to(acc_dt)
                        inv_cnt = has_s / cnt_s.clamp_min(1.0)
                        n_chunks_ok_t = n_chunks_ok_t + has_s
                        loss_sum_t = loss_sum_t + torch.where(ok_s, kl_rows[:, :, sub0:sub1], zero_acc).sum() * inv_cnt
                        row_scale[:, :, sub0:sub1] = torch.where(ok_s, inv_cnt, zero_acc)

                    if need_wq or need_wk:
                        d_logits = torch.where(
                            ok_o.unsqueeze(-1) & elig_o,
                            (probs_o - target) * (row_scale.unsqueeze(-1) * inv_temp),
                            zero_acc,
                        )
                        if d_qz_all is not None:
                            d_qz_all[:, :, o0:o1] = torch.matmul(d_logits, cent)
                        if g_cent_sum is not None:
                            g_cent_sum = g_cent_sum + torch.matmul(d_logits.transpose(-1, -2), qz_o)
                    del mass_c, mass, target, logits_o, logp_o, probs_o
            else:
                # Exact PyTorch fallback (CPU / fp32 / fp64 / per_head=True / non-standard block/head dims):
                # hoists `upcast(key)` ONCE outside the chunk loop and uses exact analytic `(d_qz, d_cent)`.
                hi_max = int(qpos_pick[-1]) + 1
                k_up = upcast(key[:, :, :hi_max].detach())
                kpos_all_hi = torch.arange(hi_max, device=dev)
                qg_pick = q_pick.view(b, g, hg, nq, dh)
                inv_temp = 1.0 / float(cfg.temperature)
                for s0 in range(0, nq, dc):
                    s1 = min(nq, s0 + dc)
                    c_len = s1 - s0
                    qpos_c = qpos_pick[s0:s1]
                    frontier_c = (qpos_c // cfg.block_size) * cfg.block_size
                    hi_c = int(qpos_c[-1]) + 1
                    s_attn = (
                        torch.matmul(
                            upcast(qg_pick[:, :, :, s0:s1]),
                            k_up[:, :, :hi_c].unsqueeze(2).transpose(-1, -2),
                        )
                        * scaling
                    )
                    kpos_c = kpos_all_hi[:hi_c]
                    allow_sub = _slice_caller_mask(allow_pick, slice(s0, s1))
                    bias_sub = _slice_caller_mask(bias_pick, slice(s0, s1))
                    m_allow_c = kpos_c[None, None, None, None] <= qpos_c[None, None, None, :, None]
                    if allow_sub is not None:
                        m_allow_c = m_allow_c & allow_sub[..., :hi_c]
                        if bias_sub is not None:
                            s_attn = s_attn + bias_sub[..., :hi_c].to(s_attn.dtype)
                        has_any_c = m_allow_c.any(dim=-1, keepdim=True)
                        s_attn = torch.where(
                            m_allow_c,
                            s_attn,
                            torch.where(has_any_c, s_attn.new_full((), float("-inf")), s_attn.new_zeros(())),
                        )
                        a = torch.where(has_any_c, torch.softmax(s_attn, dim=-1), torch.zeros_like(s_attn))
                    else:
                        s_attn = s_attn.masked_fill(~m_allow_c, float("-inf"))
                        a = torch.softmax(s_attn, dim=-1)
                    nk = min(hi_c, index.key2slot.shape[-1])
                    a = a[..., :nk] * (kpos_c[:nk][None] < frontier_c[:, None])[None, None, None]
                    slots = index.key2slot[:, :, :nk][:, :, None, None, :].expand(b, g, hg, c_len, nk)
                    mass_h = torch.zeros(b, g, hg, c_len, s_slots, device=dev, dtype=a.dtype)
                    mass_h = mass_h.scatter_add_(-1, slots, a)
                    mass = mass_h.reshape(b, g * hg, c_len, s_slots) if per_head else mass_h.mean(2)
                    tot = mass.sum(-1, keepdim=True)
                    target = mass / tot.clamp_min(1e-30)
                    ok_c = (frontier_c > 0)[None, None, :] & (tot.squeeze(-1) > 0)
                    elig_c = index.active[:, :, None, :] & (
                        index.slot_end[None, None, None, :] <= frontier_c[None, None, :, None]
                    )
                    qz_c = qz_all[:, :, s0:s1]
                    logits_c = idx.slot_scores(qz_c, index, qpos_c, flat=True)
                    elig_cmp = (
                        elig_c[:, :, None, :, :]
                        .expand(b, g, hg, c_len, s_slots)
                        .reshape(b, h, c_len, s_slots)
                        if per_head
                        else elig_c
                    )
                    if allow_sub is not None:
                        nk_full = index.key2slot.shape[-1]
                        slot_valid_c = _slot_valid_mask(
                            allow_sub[..., :nk_full],
                            index.key2slot,
                            b=b,
                            g=g,
                            hg=hg,
                            c=c_len,
                            s_slots=s_slots,
                            per_head=per_head,
                        )
                        elig_cmp = elig_cmp & slot_valid_c
                        logits_c = logits_c.masked_fill(~slot_valid_c, float("-inf"))
                    has_slot_c = torch.isfinite(logits_c).any(dim=-1, keepdim=True)
                    logp_c = torch.log_softmax(torch.where(has_slot_c, logits_c, torch.zeros_like(logits_c)), dim=-1)
                    probs_c = torch.where(elig_cmp, torch.exp(logp_c), zero_acc)
                    logp_safe = torch.where(target > 0, logp_c, zero_acc)
                    kl_rows = (target * (torch.log(target.clamp_min(1e-30)) - logp_safe)).sum(-1)

                    cnt_c = ok_c.sum().to(acc_dt)
                    has_c = (cnt_c > 0).to(acc_dt)
                    inv_cnt = has_c / cnt_c.clamp_min(1.0)
                    n_chunks_ok_t = n_chunks_ok_t + has_c
                    loss_sum_t = loss_sum_t + torch.where(ok_c, kl_rows, zero_acc).sum() * inv_cnt

                    if need_wq or need_wk:
                        row_scale = torch.where(ok_c, inv_cnt, zero_acc)
                        d_logits = torch.where(
                            ok_c.unsqueeze(-1) & elig_cmp,
                            (probs_c - target) * (row_scale.unsqueeze(-1) * inv_temp),
                            zero_acc,
                        )
                        if per_head:
                            d_logits_g = d_logits.view(b, g, hg, c_len, s_slots)
                            if d_qz_all is not None:
                                d_qz_all[:, :, s0:s1] = torch.matmul(d_logits_g, cent.unsqueeze(2)).reshape(
                                    b, h, c_len, -1
                                )
                            if g_cent_sum is not None:
                                qz_cg = qz_c.view(b, g, hg, c_len, -1)
                                g_cent_sum = g_cent_sum + torch.matmul(d_logits_g.transpose(-1, -2), qz_cg).sum(dim=2)
                        else:
                            if d_qz_all is not None:
                                d_qz_all[:, :, s0:s1] = torch.matmul(d_logits, cent)
                            if g_cent_sum is not None:
                                g_cent_sum = g_cent_sum + torch.matmul(d_logits.transpose(-1, -2), qz_c)

        sync_free = _is_sync_restricted()
        if not sync_free:
            n_ok_f, loss_sum_f = torch.stack([n_chunks_ok_t, loss_sum_t]).tolist()
            if n_ok_f <= 0.0:
                return
            inv_n_f = 1.0 / n_ok_f
            loss_val = loss_sum_f * inv_n_f
            inv_n_t = n_chunks_ok_t.new_tensor(inv_n_f)
        else:
            inv_n_t = torch.where(n_chunks_ok_t > 0, 1.0 / n_chunks_ok_t.clamp_min(1.0), zero_acc)
            loss_val_t = loss_sum_t * inv_n_t

        g_wq_avg: torch.Tensor | None = None
        if need_wq and d_qz_all is not None:
            qz_live = idx.query_latent(q_pick, pass_idx=cur_pass)
            (gwq_tot,) = torch.autograd.grad(
                qz_live,
                active_wq,
                grad_outputs=(d_qz_all * inv_n_t).to(qz_live.dtype),
                allow_unused=True,
            )
            if gwq_tot is not None:
                g_wq_avg = gwq_tot.detach()
            del qz_live

        g_wk_avg: torch.Tensor | None = None
        if (
            need_wk
            and g_cent_sum is not None
            and index.slot_labels is not None
            and index.slot_counts is not None
        ):
            g_cent_avg = g_cent_sum * inv_n_t
            bs = cfg.block_size
            pool = cfg.pool
            n_units_k = bs // pool
            kin_det = key.detach()
            km_det = caller_allow.any(dim=(1, 2, 3)).unsqueeze(1).unsqueeze(-1) if caller_allow is not None else None
            g_wk_acc = torch.zeros_like(idx.wk)
            for b0 in range(0, nb, 64):
                b1 = min(nb, b0 + 64)
                nb_c = b1 - b0
                z_raw = idx.key_latent(kin_det[:, :, b0 * bs : b1 * bs])
                if km_det is not None:
                    z_raw = z_raw * km_det[:, :, b0 * bs : b1 * bs, :].to(z_raw.dtype)
                z_b = z_raw.view(b, g, nb_c, n_units_k, pool, -1).sum(4)
                c_b = idx._centroids(
                    z_b,
                    index.slot_labels[:, :, b0:b1],
                    index.slot_counts[:, :, b0:b1],
                    spb,
                ).reshape(b, g, nb_c * spb, -1)
                go_b = g_cent_avg[:, :, b0 * spb : b1 * spb].to(c_b.dtype)
                (gw_b,) = torch.autograd.grad(c_b, idx.wk, grad_outputs=go_b, allow_unused=True)
                if gw_b is not None:
                    g_wk_acc = g_wk_acc + gw_b.detach()
                del z_b, c_b, go_b, gw_b
            g_wk_avg = g_wk_acc
            del g_cent_avg

        surrogate = active_wq.new_tensor(loss_val) if not sync_free else loss_val_t.to(active_wq.dtype)
        if need_wq and g_wq_avg is not None:
            term_q = (active_wq * g_wq_avg.to(active_wq.dtype)).sum()
            surrogate = surrogate + (term_q - term_q.detach())
        if need_wk and g_wk_avg is not None:
            term_k = (idx.wk * g_wk_avg.to(idx.wk.dtype)).sum()
            surrogate = surrogate + (term_k - term_k.detach())
        idx.pending.append(cfg.distill_weight * surrogate)
        if not sync_free:
            idx.last_stats["distill_kl"] = loss_val


def _maybe_sample_recall(
    idx: LSAIndexer,
    q: torch.Tensor,
    k: torch.Tensor,
    qpos_all: torch.Tensor,
    index: BucketIndex | None,
    cur_pass: int,
) -> None:
    """Run lightweight periodic sampled recall check and update `idx.last_recall` / `mgr.last_recall` (`B2`)."""
    cfg = idx.cfg
    win = cfg.resolved_window()
    tk = k.shape[2]
    tq = q.shape[2]
    if (
        index is None
        or cfg.recall_check_every <= 0
        or tk <= cfg.sinks + win
        or _in_backward()
        or _is_sync_restricted()
    ):
        return
    mgr_ref = getattr(idx, "_mgr_ref", None)
    mgr = mgr_ref() if callable(mgr_ref) else None
    step = int(mgr._step) if mgr is not None else int(idx._step)
    check_every = max(1, int(cfg.recall_check_every))
    due = (idx._recall_checks == 0) or (
        step != idx._last_recall_step and (step % check_every) == 0
    )
    if not due:
        return
    q_base_pos = tk - tq
    cand_start = max(0, (cfg.sinks + win) - q_base_pos)
    n_cand = max(0, tq - cand_start)
    if n_cand <= 0:
        return
    with torch.no_grad():
        take = min(16, n_cand)
        stride = max(1, n_cand // take)
        sel_r = torch.arange(cand_start, tq, stride, device=q.device)[:take]
        q_sub = q[:, :, sel_r].detach()
        qpos_sub = qpos_all[sel_r]
        chosen = idx.select_buckets(q_sub, index, qpos=qpos_sub, pass_idx=cur_pass)
        rec = compute_lsa_recall(
            q_sub,
            k.detach(),
            index,
            chosen,
            top_k_keys=64,
            window=win,
            sinks=cfg.sinks,
            qpos=qpos_sub,
        )
        idx._update_recall_state(float(rec["mean_recall"]), float(rec["min_head_recall"]))


def _resolve_hybrid_sparse(
    idx: LSAIndexer,
    q: torch.Tensor,
    k: torch.Tensor,
    qpos_all: torch.Tensor,
    index: BucketIndex | None,
    cur_pass: int,
    *,
    in_eval: bool = False,
) -> bool:
    """Check recall readiness (sampling if needed) and decide whether this `hybrid` step runs sparse."""
    cfg = idx.cfg
    mgr_ref = getattr(idx, "_mgr_ref", None)
    mgr = mgr_ref() if callable(mgr_ref) else None
    step = int(mgr._step) if mgr is not None else int(idx._step)
    ready = bool(mgr.recall_ready) if mgr is not None else bool(idx.recall_ready)
    if cfg.recall_threshold <= 0.0:
        ready = True
        idx.recall_ready = True
        if mgr is not None:
            mgr.recall_ready = True

    tk = k.shape[2]
    win = cfg.resolved_window()
    if (not ready) or cfg.distill_weight > 0.0 or in_eval:
        _maybe_sample_recall(idx, q, k, qpos_all, index, cur_pass)
        ready = bool(mgr.recall_ready) if mgr is not None else bool(idx.recall_ready)

    if not ready and cfg.hybrid_warmup_steps > 0 and step >= cfg.hybrid_warmup_steps:
        ready = True

    if not ready or cfg.hybrid_sparse_every <= 0 or tk <= win:
        return False
    return (step % cfg.hybrid_sparse_every) == 0


class _RingLowerRightCausalFlashAttn(torch.autograd.Function):
    """Exact Ring-FlashAttention for unpadded lower-right causal attention (`Tk = past_len + Tq`, `past_len > 0`).

    Avoids materializing a `[B, 1, Tq, Tk]` dense causal mask (`17-64 GiB` at `512K-1M` context)
    and avoids GQA `repeat_kv` copies by iterating over KV groups with zero-copy `.expand` and
    prefix blocks of `prefix_chunk` keys, using ATen's native FlashAttention forward + exact
    merged-LSE backward.
    """

    @staticmethod
    def forward(  # type: ignore[override]
        ctx: Any,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        past_len: int,
        scale: float,
        prefix_chunk: int = 65536,
    ) -> torch.Tensor:
        b, h, tq, dh = q.shape
        g = k.shape[1]
        hg = h // g
        fwd = torch.ops.aten._scaled_dot_product_flash_attention
        p_step = max(1, int(prefix_chunk))
        outs: list[torch.Tensor] = []
        lses: list[torch.Tensor] = []
        philox_seed: Any = None
        philox_offset: Any = None

        for gi in range(g):
            qg = q[:, gi * hg : (gi + 1) * hg].contiguous()
            o_acc: torch.Tensor | None = None
            lse_acc: torch.Tensor | None = None

            for p0 in range(0, past_len, p_step):
                p1 = min(past_len, p0 + p_step)
                kp = k[:, gi : gi + 1, p0:p1].expand(b, hg, p1 - p0, dh)
                vp = v[:, gi : gi + 1, p0:p1].expand(b, hg, p1 - p0, dh)
                r_p = fwd(qg, kp, vp, 0.0, False, False, scale=scale)
                o_p, lse_p = r_p[0], r_p[1]
                if philox_seed is None:
                    philox_seed, philox_offset = r_p[6], r_p[7]
                if o_acc is None or lse_acc is None:
                    o_acc = o_p.float()
                    lse_acc = lse_p
                else:
                    m = torch.maximum(lse_acc, lse_p)
                    w_old = torch.exp(lse_acc - m)
                    w_new = torch.exp(lse_p - m)
                    denom = w_old + w_new
                    o_acc = (o_acc * w_old.unsqueeze(-1) + o_p.float() * w_new.unsqueeze(-1)) / denom.unsqueeze(-1)
                    lse_acc = m + torch.log(denom)

            ks = k[:, gi : gi + 1, past_len:].expand(b, hg, tq, dh)
            vs = v[:, gi : gi + 1, past_len:].expand(b, hg, tq, dh)
            r_s = fwd(qg, ks, vs, 0.0, True, False, scale=scale)
            o_s, lse_s = r_s[0], r_s[1]
            if philox_seed is None:
                philox_seed, philox_offset = r_s[6], r_s[7]
            if o_acc is None or lse_acc is None:
                o_final = o_s
                lse_final = lse_s
            else:
                m = torch.maximum(lse_acc, lse_s)
                w_old = torch.exp(lse_acc - m)
                w_new = torch.exp(lse_s - m)
                denom = w_old + w_new
                o_final = ((o_acc * w_old.unsqueeze(-1) + o_s.float() * w_new.unsqueeze(-1)) / denom.unsqueeze(-1)).to(
                    q.dtype
                )
                lse_final = m + torch.log(denom)

            outs.append(o_final)
            lses.append(lse_final)

        out = torch.cat(outs, dim=1)
        lse_all = torch.cat(lses, dim=1)
        ctx.save_for_backward(q, k, v, out, lse_all)
        ctx.past_len = past_len
        ctx.scale = scale
        ctx.prefix_chunk = p_step
        ctx.philox_seed = philox_seed
        ctx.philox_offset = philox_offset
        return out

    @staticmethod
    def backward(ctx: Any, grad_out: torch.Tensor) -> tuple[torch.Tensor | None, ...]:  # type: ignore[override]
        q, k, v, out, lse_all = ctx.saved_tensors
        past_len: int = ctx.past_len
        scale: float = ctx.scale
        p_step: int = ctx.prefix_chunk
        b, h, tq, dh = q.shape
        g = k.shape[1]
        hg = h // g
        bwd = torch.ops.aten._scaled_dot_product_flash_attention_backward
        grad_out = grad_out.contiguous()

        need_q = bool(ctx.needs_input_grad[0])
        need_k = bool(ctx.needs_input_grad[1])
        need_v = bool(ctx.needs_input_grad[2])
        dq = torch.empty_like(q) if need_q else None
        dk = torch.empty_like(k) if need_k else None
        dv = torch.empty_like(v) if need_v else None

        for gi in range(g):
            sl_h = slice(gi * hg, (gi + 1) * hg)
            qg = q[:, sl_h].contiguous()
            og = out[:, sl_h].contiguous()
            gog = grad_out[:, sl_h].contiguous()
            lseg = lse_all[:, sl_h].contiguous()

            dq_acc: torch.Tensor | None = None
            for p0 in range(0, past_len, p_step):
                p1 = min(past_len, p0 + p_step)
                kp = k[:, gi : gi + 1, p0:p1].expand(b, hg, p1 - p0, dh)
                vp = v[:, gi : gi + 1, p0:p1].expand(b, hg, p1 - p0, dh)
                dq_p, dk_p, dv_p = bwd(
                    gog,
                    qg,
                    kp,
                    vp,
                    og,
                    lseg,
                    None,
                    None,
                    tq,
                    p1 - p0,
                    0.0,
                    False,
                    ctx.philox_seed,
                    ctx.philox_offset,
                    scale=scale,
                )
                if need_q:
                    dq_acc = dq_p if dq_acc is None else (dq_acc + dq_p)
                if dk is not None:
                    dk[:, gi, p0:p1] = dk_p.sum(dim=1)
                if dv is not None:
                    dv[:, gi, p0:p1] = dv_p.sum(dim=1)
                del dq_p, dk_p, dv_p

            ks = k[:, gi : gi + 1, past_len:].expand(b, hg, tq, dh)
            vs = v[:, gi : gi + 1, past_len:].expand(b, hg, tq, dh)
            dq_s, dk_s, dv_s = bwd(
                gog,
                qg,
                ks,
                vs,
                og,
                lseg,
                None,
                None,
                tq,
                tq,
                0.0,
                True,
                ctx.philox_seed,
                ctx.philox_offset,
                scale=scale,
            )
            if dq is not None:
                dq[:, sl_h] = dq_s if dq_acc is None else (dq_acc + dq_s)
            if dk is not None:
                dk[:, gi, past_len:] = dk_s.sum(dim=1)
            if dv is not None:
                dv[:, gi, past_len:] = dv_s.sum(dim=1)
            del dq_s, dk_s, dv_s, dq_acc

        return dq, dk, dv, None, None, None


def _dense_lower_right_causal_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    scaling: float,
    dropout: float = 0.0,
) -> tuple[torch.Tensor, None]:
    """Exact unpadded lower-right causal attention for `Tq > 1, Tk > Tq` with zero full-matrix mask allocation."""
    _b, h, tq, dh = query.shape
    g, tk = key.shape[1], key.shape[2]
    past_len = tk - tq
    if (
        query.is_cuda
        and query.dtype in (torch.bfloat16, torch.float16)
        and key.dtype == query.dtype
        and value.dtype == query.dtype
        and dropout == 0.0
        and past_len > 0
        and dh <= 256
        and dh % 8 == 0
        and h % g == 0
        and hasattr(torch.ops.aten, "_scaled_dot_product_flash_attention")
        and hasattr(torch.ops.aten, "_scaled_dot_product_flash_attention_backward")
    ):
        out = _RingLowerRightCausalFlashAttn.apply(query, key, value, past_len, float(scaling), 65536)
        return out.transpose(1, 2).contiguous(), None

    # Exact fallback (CPU / fp32 / fp64 / dropout > 0): query-chunked masked SDPA
    k_rep = key.repeat_interleave(h // g, dim=1) if g != h else key
    v_rep = value.repeat_interleave(h // g, dim=1) if g != h else value
    kpos = torch.arange(tk, device=query.device)
    q_step = 512
    parts: list[torch.Tensor] = []
    for q0 in range(0, tq, q_step):
        q1 = min(tq, q0 + q_step)
        qpos_c = torch.arange(past_len + q0, past_len + q1, device=query.device)
        m_c = kpos[None, :] <= qpos_c[:, None]
        parts.append(
            F.scaled_dot_product_attention(
                query[:, :, q0:q1],
                k_rep,
                v_rep,
                attn_mask=m_c,
                dropout_p=dropout,
                scale=scaling,
                is_causal=False,
            )
        )
    out = parts[0] if len(parts) == 1 else torch.cat(parts, dim=2)
    return out.transpose(1, 2).contiguous(), None


def lsa_attention(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    scaling: float | None = None,
    dropout: float = 0.0,
    **kwargs: Any,
) -> tuple[torch.Tensor, None]:
    """HF attention-interface function. `q [B,H,Tq,D]`, `k/v [B,G,Tk,D]` -> `([B,Tq,H,D], None)`."""
    idx: LSAIndexer | None = getattr(module, "lsa_indexer", None)
    dense = _DENSE_FN[0]
    if dense is None:
        raise RuntimeError("lsa_attention called before install_lsa() registered a dense fallback")
    key = _ensure_attn_kv_dtype(module, key, query.dtype, is_key=True)
    value = _ensure_attn_kv_dtype(module, value, query.dtype, is_key=False)
    if idx is None or idx.cfg.mode in ("off", "dense"):
        if attention_mask is None and query.shape[2] > 1 and key.shape[2] > query.shape[2]:
            sc = query.shape[-1] ** -0.5 if scaling is None else scaling
            return _dense_lower_right_causal_attention(query, key, value, scaling=sc, dropout=dropout)
        return dense(module, query, key, value, attention_mask, scaling=scaling, dropout=dropout, **kwargs)

    cfg = idx.cfg
    cache_key = kwargs.pop("cache_key", None)
    if cache_key is None:
        pkv_kw = extract_cache_from_kwargs(kwargs)
        if pkv_kw is not None:
            pkv_root, pkv_ok = _unwrap_pkv_for_cache(pkv_kw)
            if pkv_ok and pkv_root is not None:
                idx._active_pkv = (_weakref_or_none(pkv_root), id(pkv_root), True)

    b, h, tq, dh = query.shape
    g, tk = key.shape[1], key.shape[2]
    hg = h // g
    scaling = dh ** -0.5 if scaling is None else scaling
    qpos_all = torch.arange(tk - tq, tk, device=query.device)
    qin = query.detach() if cfg.detach_inputs else query
    cur_pass = idx._resolve_pass(module=module)
    active_wq = idx.active_wq(pass_idx=cur_pass)

    caller_allow: torch.Tensor | None = None
    caller_bias: torch.Tensor | None = None
    key_valid_mask: torch.Tensor | None = None
    if attention_mask is not None:
        caller_allow, caller_bias = _normalize_caller_mask(
            attention_mask,
            b=b,
            g=g,
            hg=hg,
            tq=tq,
            tk=tk,
            device=query.device,
        )
        key_valid_mask = caller_allow.any(dim=(1, 2, 3))

    want_kl = (
        cfg.distill_weight > 0
        and torch.is_grad_enabled()
        and (active_wq.requires_grad or idx.wk.requires_grad)
        and tk >= cfg.block_size + 1
    )
    need_index = (
        (cfg.mode == "sparse" and tk > cfg.resolved_window())
        or (cfg.mode == "hybrid" and tk > cfg.resolved_window())
        or (want_kl and (not cfg.detach_inputs or not _in_backward()))
    )
    if need_index:
        if cfg.detach_inputs:
            with torch.no_grad():
                index = idx.build(key, cache_key=cache_key, key_valid_mask=key_valid_mask)
        else:
            index = idx.build(key, cache_key=cache_key, key_valid_mask=key_valid_mask)
    else:
        index = None

    in_eval = (not module.training) or (not torch.is_grad_enabled())
    if cfg.mode == "sparse":
        sparse = tk > cfg.resolved_window()
        if sparse and (cfg.distill_weight > 0.0 or in_eval):
            _maybe_sample_recall(idx, qin, key, qpos_all, index, cur_pass)
    elif cfg.mode == "hybrid":
        sparse = _resolve_hybrid_sparse(idx, qin, key, qpos_all, index, cur_pass, in_eval=in_eval)
    else:
        sparse = False

    if sparse and attention_mask is not None and cfg.strict_mask:
        raise RuntimeError(
            "LSA sparse mode does not support padding masks (strict_mask=true); "
            "use sequence packing without padding"
        )

    if want_kl and index is not None:
        if cfg.detach_inputs:
            # Exact zero-checkpoint-unpack, zero-inner-sync KL distillation with analytic `(d_qz, d_cent)`
            # and fused 2-pass Triton teacher bucket-mass kernels on CUDA bf16/fp16.
            if not _in_backward():
                _run_detached_distill_kl(
                    qin,
                    key,
                    qpos_all,
                    index,
                    idx,
                    active_wq,
                    scaling,
                    cur_pass,
                    caller_allow=caller_allow,
                    caller_bias=caller_bias,
                )
        else:
            cand_start = max(0, cfg.block_size - (tk - tq))
            n_cand = max(0, tq - cand_start)
            if n_cand > 0:
                nq = max(1, int(n_cand * cfg.distill_query_frac))
                pick = (torch.randperm(n_cand, device=query.device)[:nq] + cand_start).sort().values
                qg_kl = qin.view(b, g, hg, tq, dh)
                kls = []
                for s0 in range(0, pick.numel(), cfg.distill_chunk):
                    sel = pick[s0 : s0 + cfg.distill_chunk]
                    qz = idx.query_latent(qin[:, :, sel], pass_idx=cur_pass)
                    kl = _distill_kl(
                        qg_kl[:, :, :, sel],
                        qz,
                        qpos_all[sel],
                        key.detach(),
                        index,
                        idx,
                        scaling,
                        caller_allow=_slice_caller_mask(caller_allow, sel),
                        caller_bias=_slice_caller_mask(caller_bias, sel),
                    )
                    if kl is not None:
                        kls.append(kl)
                if kls and not _in_backward():
                    loss = torch.stack(kls).mean()
                    idx.pending.append(cfg.distill_weight * loss)
                    idx.last_stats["distill_kl"] = float(loss.detach())

    if not sparse:
        if not _in_backward():
            idx.last_stats["forward_sparse"] = 0.0
        if attention_mask is None and tq > 1 and tk > tq:
            return _dense_lower_right_causal_attention(query, key, value, scaling=scaling, dropout=dropout)
        return dense(module, query, key, value, attention_mask, scaling=scaling, dropout=dropout, **kwargs)

    if not _in_backward() and not _is_sync_restricted():
        idx.last_stats["forward_sparse"] = 1.0

    qg = query.view(b, g, hg, tq, dh)
    with torch.no_grad():
        qz_all = idx.query_latent(qin.detach(), pass_idx=cur_pass)
    q_base_pos = tk - tq

    per_head = cfg.per_head_select or (qz_all.shape[1] == g * hg and hg > 1)
    can_triton_sparse = (
        _TRITON_AVAILABLE
        and caller_allow is None
        and query.is_cuda
        and query.dtype in (torch.bfloat16, torch.float16)
        and key.dtype == query.dtype
        and value.dtype == query.dtype
        and not per_head
        and dh in (32, 64, 128, 256)
        and hg <= 32
        and tq > 0
        and tk > 0
    )
    if can_triton_sparse:
        w = cfg.resolved_window()
        ns = min(cfg.sinks, tk)
        max_span = 32768
        tail_len = tq - ((tq - 1) // cfg.query_chunk) * cfg.query_chunk
        outs: list[torch.Tensor] = []
        for s0 in range(0, tq, max_span):
            s1 = min(tq, s0 + max_span)
            is_last_span = s1 >= tq
            mem_span = _build_sparse_mem_table(
                qz_all[:, :, s0:s1],
                qpos_all[s0:s1],
                index,
                idx,
                tk=tk,
                q_base_pos=q_base_pos + s0,
                cur_pass=cur_pass,
                record_tail_len=tail_len,
                record_stats=is_last_span,
            )
            o_span = _TritonSparseGatherAttnFn.apply(
                qg[:, :, :, s0:s1],
                key,
                value,
                mem_span,
                q_base_pos + s0,
                w,
                ns,
                float(scaling),
            )
            outs.append(o_span)
        out = outs[0] if len(outs) == 1 else torch.cat(outs, dim=3)
        return out.reshape(b, h, tq, dh).transpose(1, 2).contiguous(), None

    outs = []
    for c0 in range(0, tq, cfg.query_chunk):
        c1 = min(tq, c0 + cfg.query_chunk)
        sl = slice(c0, c1)
        qb = (q_base_pos + c0, q_base_pos + c1 - 1)
        is_last = c1 >= tq
        allow_c = _slice_caller_mask(caller_allow, sl)
        bias_c = _slice_caller_mask(caller_bias, sl)

        def run(
            qc: torch.Tensor,
            kc: torch.Tensor,
            vc: torch.Tensor,
            qzc: torch.Tensor,
            sl: slice = sl,
            p: int = cur_pass,
            q_bounds: tuple[int, int] = qb,
            rec_stats: bool = is_last,
            m_allow: torch.Tensor | None = allow_c,
            m_bias: torch.Tensor | None = bias_c,
        ) -> torch.Tensor:
            return _sparse_chunk(
                qc,
                qzc,
                qpos_all[sl],
                kc,
                vc,
                index,
                idx,
                scaling,
                cur_pass=p,
                q_bounds=q_bounds,
                record_stats=rec_stats,
                caller_allow=m_allow,
                caller_bias=m_bias,
            )

        args = (qg[:, :, :, sl], key, value, qz_all[:, :, sl])
        if cfg.checkpoint_chunks and torch.is_grad_enabled() and (query.requires_grad or key.requires_grad):
            outs.append(checkpoint(run, *args, use_reentrant=False))
        else:
            outs.append(run(*args))
    out = torch.cat(outs, dim=3)
    return out.reshape(b, h, tq, dh).transpose(1, 2).contiguous(), None


_DENSE_FN: list[Callable[..., Any] | None] = [None]


# ============================================================================ install / manage


def apply_yarn(config: Any, target_len: int, original_len: int | None = None) -> dict[str, Any]:
    """Set YaRN RoPE scaling on a (text) config for `target_len` tokens. Call BEFORE model init."""
    params = getattr(config, "rope_parameters", None)
    has_v5_params = isinstance(params, dict) or (
        hasattr(config, "rope_parameters")
        and params is None
        and isinstance(getattr(type(config), "rope_scaling", None), property)
    )

    # Resolve original context length; remain idempotent if YaRN was already applied on `config`.
    existing_orig: int | None = None
    if original_len is None:
        if isinstance(params, dict):
            probe_dicts = (
                [v for v in params.values() if isinstance(v, dict)]
                if (params and all(isinstance(v, dict) for v in params.values()))
                else [params]
            )
        elif not has_v5_params:
            rs = getattr(config, "rope_scaling", None)
            probe_dicts = [rs] if isinstance(rs, dict) else []
        else:
            probe_dicts = []
        for d in probe_dicts:
            raw_orig = d.get("original_max_position_embeddings")
            if (
                str(d.get("rope_type", d.get("type", ""))) == "yarn"
                and isinstance(raw_orig, (int, float))
                and int(raw_orig) > 0
            ):
                existing_orig = int(raw_orig)
                break

    orig = int(original_len or existing_orig or getattr(config, "max_position_embeddings", None) or 262144)
    factor = max(1.0, float(target_len) / float(orig))
    rope: dict[str, Any] = {
        "rope_type": "yarn",
        "factor": factor,
        "original_max_position_embeddings": orig,
    }

    if has_v5_params:
        # Transformers >= 5.x: update `rope_parameters` cleanly while preserving `rope_theta`,
        # `mrope_section`, `mrope_interleaved`, `partial_rotary_factor`, etc. NEVER assign the
        # deprecated `rope_scaling` setter, which overwrites `rope_parameters` and wipes
        # `rope_theta` to `None` in `transformers >= 5.x`.
        base_params = dict(params) if isinstance(params, dict) else {}
        is_nested = bool(base_params) and all(isinstance(v, dict) for v in base_params.values())

        def _merge_v5_entry(entry: dict[str, Any]) -> dict[str, Any]:
            merged = dict(entry)
            merged.pop("type", None)
            merged.update(rope)
            if merged.get("rope_theta") is None:
                cfg_theta = getattr(config, "rope_theta", None)
                if cfg_theta is not None:
                    merged["rope_theta"] = float(cfg_theta)
            if merged.get("partial_rotary_factor") is None:
                cfg_prf = getattr(config, "partial_rotary_factor", None)
                if cfg_prf is not None:
                    merged["partial_rotary_factor"] = float(cfg_prf)
            return merged

        if is_nested:
            target_keys = [
                k for k in base_params if "sliding" not in str(k).lower() and "linear" not in str(k).lower()
            ]
            if not target_keys:
                target_keys = list(base_params.keys())
            target_set = set(target_keys)
            config.rope_parameters = {
                k: (_merge_v5_entry(sub) if k in target_set else dict(sub))
                for k, sub in base_params.items()
            }
        else:
            config.rope_parameters = _merge_v5_entry(base_params)
    else:
        # Legacy Transformers 4.x or ToyConfig without v5 `rope_parameters`:
        # merge into any existing `rope_scaling` dict rather than clobbering sibling fields.
        existing_rs = getattr(config, "rope_scaling", None)
        merged_rs = dict(existing_rs) if isinstance(existing_rs, dict) else {}
        merged_rs.update(rope)
        config.rope_scaling = merged_rs

    config.max_position_embeddings = int(max(int(target_len), orig))
    log.info("YaRN: %d -> %d tokens (factor %.2f)", orig, int(target_len), factor)
    return rope


class LSAManager:
    def __init__(
        self,
        model: nn.Module,
        cfg: LSAConfig,
        indexers: dict[int, LSAIndexer],
        prev_impl: str,
        set_impl: Callable[[str], None],
        hooks: list[Any] | None = None,
    ):
        self.model, self.cfg, self.indexers = model, cfg, indexers
        self._prev_impl, self._set_impl = prev_impl, set_impl
        self._hooks: list[Any] = list(hooks or [])
        self._step: int = 0
        self.last_recall: float = 0.0
        self.recall_ready: bool = bool(cfg.recall_threshold <= 0.0)
        for ix in self.indexers.values():
            object.__setattr__(ix, "_mgr_ref", weakref.ref(self))
            if self.recall_ready:
                ix.recall_ready = True

    def set_step(self, step: int) -> None:
        self._step = int(step)
        for ix in self.indexers.values():
            ix._step = int(step)

    def set_mode(self, mode: str) -> None:
        if mode not in _VALID_LSA_MODES:
            raise ValueError(f"lsa.mode must be one of {_VALID_LSA_MODES}, got {mode!r}")
        self.cfg.mode = mode
        for ix in self.indexers.values():
            ix.cfg.mode = mode
        if mode == "hybrid" and self.cfg.recall_threshold <= 0.0:
            self.recall_ready = True
            for ix in self.indexers.values():
                ix.recall_ready = True

    @contextlib.contextmanager
    def temporary_mode(self, mode: str) -> Iterator[LSAManager]:
        prev = self.cfg.mode
        prev_ix = {lid: ix.cfg.mode for lid, ix in self.indexers.items()}
        self.set_mode(mode)
        try:
            yield self
        finally:
            self.cfg.mode = prev
            for lid, ix in self.indexers.items():
                ix.cfg.mode = prev_ix[lid]

    def _refresh_recall_from_indexers(self) -> None:
        measured = [ix.last_recall for ix in self.indexers.values() if ix._recall_checks > 0]
        if measured:
            self.last_recall = float(sum(measured) / len(measured))
            if self.last_recall >= self.cfg.recall_threshold or self.cfg.recall_threshold <= 0.0:
                self.recall_ready = True
                for ix in self.indexers.values():
                    ix.recall_ready = True

    def evaluate_recall(
        self,
        q: torch.Tensor | None = None,
        k: torch.Tensor | None = None,
        *,
        layer_id: int | None = None,
        top_k_keys: int = 64,
    ) -> dict[str, Any]:
        """Evaluate recall explicitly on `(q, k)` or aggregate latest sampled recall across indexers."""
        if q is not None and k is not None and self.indexers:
            targets = (
                [self.indexers[layer_id]]
                if layer_id is not None
                else list(self.indexers.values())
            )
            per_layer = [ix.evaluate_recall(q, k, top_k_keys=top_k_keys) for ix in targets]
            mean_rec = float(sum(r["mean_recall"] for r in per_layer) / len(per_layer))
            min_rec = float(min(r["min_head_recall"] for r in per_layer))
            n_h = len(per_layer[0]["per_head_recall"])
            per_head = [
                float(sum(r["per_head_recall"][hi] for r in per_layer) / len(per_layer))
                for hi in range(n_h)
            ]
            self.last_recall = mean_rec
            if self.last_recall >= self.cfg.recall_threshold or self.cfg.recall_threshold <= 0.0:
                self.recall_ready = True
                for ix in self.indexers.values():
                    ix.recall_ready = True
            return {
                "mean_recall": mean_rec,
                "min_head_recall": min_rec,
                "per_head_recall": per_head,
                "recall_ready": self.recall_ready,
            }
        self._refresh_recall_from_indexers()
        return {
            "mean_recall": self.last_recall,
            "recall_ready": self.recall_ready,
        }

    def aux_loss(self, reduction: str | None = None) -> torch.Tensor | None:
        losses = [loss for ix in self.indexers.values() for loss in ix.pop_losses()]
        if not losses:
            return None
        red = self.cfg.aux_reduction if reduction is None else str(reduction)
        stacked = torch.stack(losses)
        if red == "mean":
            return stacked.mean()
        if red == "sum":
            return stacked.sum()
        raise ValueError(f"lsa.aux_reduction must be 'mean' or 'sum', got {red!r}")

    def trainable_parameters(self) -> list[nn.Parameter]:
        return [p for ix in self.indexers.values() for p in ix.parameters()]

    def set_current_pass(self, r: int) -> None:
        for ix in self.indexers.values():
            ix.set_current_pass(r)

    def clear_cache(self) -> None:
        for ix in self.indexers.values():
            ix._cache.clear()
            ix._active_pkv = (None, None, False)
            ix.last_selected_buckets.clear()

    def stats(self) -> dict[str, float]:
        out: dict[str, float] = {
            "lsa/last_recall": float(self.last_recall),
            "lsa/recall_ready": 1.0 if self.recall_ready else 0.0,
        }
        for lid, ix in self.indexers.items():
            for k, v in ix.last_stats.items():
                out[f"lsa/{k}/L{lid}"] = v
        return out

    def uninstall(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
        for ix in self.indexers.values():
            for attn in getattr(ix, "_owners", []):
                if getattr(attn, "lsa_indexer", None) is ix:
                    del attn.lsa_indexer
        self._set_impl(self._prev_impl)
        log.info("D-LSA removed; attention implementation restored to %s", self._prev_impl)


def _resolve_dense_and_register(model: nn.Module, text_config: Any) -> tuple[str, Callable[[str], None]]:
    prev = getattr(text_config, "_attn_implementation", None) or "sdpa"
    from . import toy as toy_mod

    if prev in toy_mod.TOY_ATTENTION_FUNCTIONS and getattr(text_config, "model_type", "").startswith("toy"):
        _DENSE_FN[0] = toy_mod.TOY_ATTENTION_FUNCTIONS[prev]
        toy_mod.TOY_ATTENTION_FUNCTIONS[LSA_IMPL_NAME] = lsa_attention

        def set_impl(name: str) -> None:
            text_config._attn_implementation = name

        return prev, set_impl
    try:
        from transformers import AttentionInterface
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    except Exception as e:  # pragma: no cover - requires transformers
        raise RuntimeError(f"LSA on a HF model needs transformers with AttentionInterface: {e!r}") from e
    if prev in ALL_ATTENTION_FUNCTIONS:
        dense_key = prev
    elif "sdpa" in ALL_ATTENTION_FUNCTIONS and ("quant" in prev.lower() or "compressed" in prev.lower()):
        dense_key = "sdpa"
    else:
        raise RuntimeError(
            f"cannot use attn_implementation={prev!r} as LSA dense fallback; load the model with "
            f"attn_implementation='sdpa' (or one of {list(ALL_ATTENTION_FUNCTIONS.keys())})"
        )
    _DENSE_FN[0] = ALL_ATTENTION_FUNCTIONS[dense_key]
    try:
        AttentionInterface.register(LSA_IMPL_NAME, lsa_attention)
    except Exception:
        ALL_ATTENTION_FUNCTIONS[LSA_IMPL_NAME] = lsa_attention
    try:
        from transformers import AttentionMaskInterface
        from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS

        mask_key = prev if prev in ALL_MASK_ATTENTION_FUNCTIONS else dense_key
        if mask_key in ALL_MASK_ATTENTION_FUNCTIONS:
            try:
                AttentionMaskInterface.register(LSA_IMPL_NAME, ALL_MASK_ATTENTION_FUNCTIONS[mask_key])
            except Exception:
                ALL_MASK_ATTENTION_FUNCTIONS[LSA_IMPL_NAME] = ALL_MASK_ATTENTION_FUNCTIONS[mask_key]
    except Exception:
        pass

    def set_impl(name: str) -> None:  # pragma: no cover - requires transformers
        tm = find_text_model(model)
        if hasattr(tm, "set_attn_implementation"):
            try:
                tm.set_attn_implementation(name)
            except Exception as e:
                log.debug("text_model.set_attn_implementation(%s) failed (%r); setting config attribute", name, e)
        targets = [text_config, getattr(tm, "config", None)]
        root_cfg = getattr(model, "config", None)
        if root_cfg is not None and getattr(root_cfg, "text_config", None) is None:
            targets.append(root_cfg)
        tm_layers = getattr(tm, "layers", None)
        if isinstance(tm_layers, (list, tuple, nn.ModuleList)):
            for lyr in tm_layers:
                inner_lyr = getattr(lyr, "inner", lyr)
                s_attn = getattr(inner_lyr, "self_attn", None)
                if s_attn is not None and getattr(s_attn, "config", None) is not None:
                    targets.append(s_attn.config)
        for cfg_obj in {id(c): c for c in targets if c is not None}.values():
            try:
                cfg_obj._attn_implementation = name
            except Exception:
                pass

    return prev, set_impl


def _make_attn_pkv_pre_hook(ix: LSAIndexer) -> Callable[..., None]:
    """Overwrite-only pre-hook recording the active `past_key_values` object id with zero host syncs."""

    def _pre(_mod: nn.Module, args: tuple[Any, ...], kw: dict[str, Any]) -> None:
        pkv = extract_cache_from_kwargs(kw)
        if pkv is None and len(args) >= 4:
            pkv = args[3]
        if pkv is None:
            ix._active_pkv = (None, None, False)
            return
        root_pkv, ok = _unwrap_pkv_for_cache(pkv)
        if ok and root_pkv is not None:
            ix._active_pkv = (_weakref_or_none(root_pkv), id(root_pkv), True)
        else:
            ix._active_pkv = (None, None, False)

    return _pre


def install_lsa(model: nn.Module, cfg: LSAConfig | dict[str, Any]) -> LSAManager:
    """Attach an LSAIndexer to every selected full-attention layer and route attention through LSA.

    Install BEFORE `apply_loop` (the loop wraps layers but keeps the same attention modules).
    """
    if isinstance(cfg, dict):
        cfg = LSAConfig.from_dict(cfg)
    cfg.validate()
    parts = inspect_model(model)
    tc = parts.text_config
    if cfg.rotary_dim is None:
        hd = getattr(tc, "head_dim", None) or tc.hidden_size // tc.num_attention_heads
        raw_rope = getattr(tc, "rope_parameters", None) or {}
        rope = (
            raw_rope.get("full_attention")
            if isinstance(raw_rope, dict) and isinstance(raw_rope.get("full_attention"), dict)
            else raw_rope
        )
        rope_d = rope if isinstance(rope, dict) else {}
        prf = getattr(tc, "partial_rotary_factor", None) or rope_d.get("partial_rotary_factor", 1.0)
        cfg.rotary_dim = int(hd * prf)
    full = parts.full_attention_indices()
    layers = full if cfg.layers is None else list(cfg.layers)
    bad = [i for i in layers if i not in full]
    if bad:
        raise ValueError(f"lsa.layers {bad} are not full-attention layers (full: {full})")
    prev, set_impl = _resolve_dense_and_register(model, tc)
    head_dim = getattr(tc, "head_dim", None) or tc.hidden_size // tc.num_attention_heads
    indexers: dict[int, LSAIndexer] = {}
    hooks: list[Any] = []
    ref = parts.text_model.embed_tokens.weight
    for i in layers:
        attn = attention_module(parts.layers[i], FULL)
        ix = LSAIndexer(tc.num_attention_heads, tc.num_key_value_heads, head_dim, cfg, i)
        ix = ix.to(device=ref.device, dtype=torch.float64 if ref.dtype == torch.float64 else torch.float32)
        attn.lsa_indexer = ix
        object.__setattr__(ix, "_owners", [attn])
        object.__setattr__(ix, "_tm_ref", weakref.ref(parts.text_model))
        hooks.append(attn.register_forward_pre_hook(_make_attn_pkv_pre_hook(ix), with_kwargs=True))
        indexers[i] = ix
    set_impl(LSA_IMPL_NAME)
    log.info(
        "D-LSA installed on layers %s (mode=%s, index_dim=%d on %s dims, block=%d, pool=%d, buckets~%d keys, "
        "select=%d, window=%d, sinks=%d, search=%s, per_head=%s, compact_slots=%s)",
        layers,
        cfg.mode,
        cfg.index_dim,
        cfg.index_on,
        cfg.block_size,
        cfg.pool,
        cfg.avg_bucket_keys,
        cfg.num_select,
        cfg.resolved_window(),
        cfg.sinks,
        cfg.search,
        cfg.per_head_select,
        cfg.compact_active_slots,
    )
    return LSAManager(model, cfg, indexers, prev, set_impl, hooks=hooks)
