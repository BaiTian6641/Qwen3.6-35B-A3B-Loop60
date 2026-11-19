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
from dataclasses import dataclass, fields, replace
from typing import Any, Callable

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from ._dtype import upcast
from .introspect import FULL, attention_module, find_text_model, inspect_model

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
    if torch.cuda.is_available():
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
    recall_check_every: int = 5         # D1: cadence for sampled recall checks before recall_ready

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


class _PassQueryProj(nn.Module):
    """Per-pass query latent projection (`[num_heads, din, index_dim]`) for passes 2..R."""

    def __init__(self, init_weight: torch.Tensor):
        super().__init__()
        self.weight = nn.Parameter(init_weight.clone())

    def forward(self, q_slice: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bhtd,hde->bhte", q_slice.to(self.weight.dtype), self.weight)


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

    def _current_loop_ctx(self) -> tuple[Any | None, int | None]:
        tm_ref = getattr(self, "_tm_ref", None)
        tm = tm_ref() if callable(tm_ref) else None
        if tm is not None:
            layers = getattr(tm, "layers", None)
            if layers is not None and len(layers) > 0:
                ctl = getattr(layers[0], "_ctl", None)
                cur = getattr(ctl, "_cur", None) if ctl is not None else None
                if cur is not None:
                    return cur[0], int(cur[1])
        return None, None

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
            z = torch.einsum("bhtd,hde->bhte", q_sl.to(w_r.dtype), w_r)
        elif r <= 1 or self.cfg.num_query_passes <= 1 or len(self.q_proj_extra) == 0:
            z = torch.einsum("bhtd,hde->bhte", q_sl.to(self.wq.dtype), self.wq)
        else:
            z = self.q_proj_extra[r - 2](q_sl)
        z = F.normalize(upcast(z), dim=-1)
        if self.cfg.per_head_select:
            return z
        b, _, t, d = z.shape
        return z.view(b, self.g, self.hg, t, d).sum(2)

    def key_latent(self, k: torch.Tensor) -> torch.Tensor:
        z = torch.einsum("bgtd,gde->bgte", k[..., self.lo:].to(self.wk.dtype), self.wk)
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
    ) -> tuple[
        tuple[Any, ...] | None,
        _CacheEntry | None,
        int,
        weakref.ReferenceType[Any] | None,
        weakref.ReferenceType[Any] | None,
    ]:
        """Resolve zero-sync cache key and safe block-reuse count (`D8`)."""
        b, g, tk, _ = k.shape
        dev_s = str(k.device)
        k_ident = (int(kin.data_ptr()), int(kin._version))
        loop_ctx, _ = self._current_loop_ctx()
        loop_ref = _weakref_or_none(loop_ctx)

        wk_ver = int(self.wk._version)

        if cache_key is not None:
            lkey: tuple[Any, ...] = ("explicit", cache_key, b, g, dev_s)
            ent = self._cache.get(lkey)
            reuse = ent.n_blocks if (ent is not None and ent.n_blocks <= nb and ent.wk_version == wk_ver) else 0
            return lkey, ent, reuse, None, loop_ref

        pkv_ref, pkv_id, pkv_ok = self._active_pkv
        pkv_obj = pkv_ref() if pkv_ref is not None else None
        if pkv_ok and (pkv_obj is not None or pkv_ref is None) and pkv_id is not None:
            lkey = ("pkv", pkv_id, self.layer_id, b, g, dev_s)
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
                if owner_alive and (tk > ent.last_tk or (tk == ent.last_tk and (same_loop or k_ident == ent.k_ident))):
                    reuse = ent.n_blocks
            return lkey, ent, reuse, pkv_ref, loop_ref

        if loop_ctx is not None:
            lkey = ("loop", id(loop_ctx), self.layer_id, b, g, tk, dev_s)
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

        if not grad_mode:
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

    def build(self, k: torch.Tensor, cache_key: Hashable | None = None) -> BucketIndex | None:
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

        lkey, ent, reuse, owner_ref, loop_ref = self._resolve_cache_lookup(
            k, kin, nb, grad_mode, cache_key
        )

        with torch.no_grad():
            if reuse < nb:
                znew = self.key_latent(kin[:, :, reuse * bs : nb * bs]).view(b, g, nb - reuse, bs, -1)
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
            z = self.key_latent(kin[:, :, : nb * bs]).view(b, g, nb, n, pool, -1).sum(4)
            cent = self._centroids(z, slot_labels, slot_counts, spb)
        else:
            with torch.no_grad():
                if reuse and ent is not None and ent.centroids is not None:
                    if reuse < nb:
                        zn = (
                            self.key_latent(kin[:, :, reuse * bs : nb * bs])
                            .view(b, g, nb - reuse, n, pool, -1)
                            .sum(4)
                        )
                        cent = torch.cat(
                            [ent.centroids, self._centroids(zn, slab_new, scnt_new, spb)], dim=2
                        )
                    else:
                        cent = ent.centroids
                else:
                    z = self.key_latent(kin[:, :, : nb * bs]).view(b, g, nb, n, pool, -1).sum(4)
                    cent = self._centroids(z, slot_labels, slot_counts, spb)

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
            )
            self._cache.move_to_end(lkey)
            while len(self._cache) > cfg.cache_entries:
                self._cache.popitem(last=False)

        return self._assemble(labels, slot_labels, slot_counts, umem, cent)

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
            s = torch.einsum("bghcd,bgsd->bghcs", qz_g, cent) / cfg.temperature
            if cfg.search == "tree" and not flat:
                tmask = self._tree_mask_per_head(qz_g, index, elig, qpos)
                s = s.masked_fill(~tmask, float("-inf"))
            s = s.masked_fill(~elig[:, :, None, :, :], float("-inf"))
            return s.reshape(b, q_heads, c, s_total)

        s = torch.einsum("bgcd,bgsd->bgcs", qz, cent) / cfg.temperature
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
        ss = torch.einsum("bgcd,bgnd->bgcn", qz, full.to(qz.dtype))
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
        ss = torch.einsum("bghcd,bgnd->bghcn", qz_g, full.to(qz_g.dtype))
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

    # Priority `Kb - j` keeps earliest / highest-scoring selected buckets first when capped!
    ar_kb = torch.arange(kb_n, device=dev, dtype=torch.long)
    prio = torch.where(m_b, kb_n - ar_kb, torch.zeros_like(ar_kb))
    if l_pack < kb_n:
        top_prio, pack_idx = prio.topk(l_pack, dim=-1, largest=True, sorted=True)
        mem_p = mem.gather(-1, pack_idx)
        m_p = top_prio > 0
    else:
        mem_p = mem
        m_p = m_b

    # Sort packed valid indices ascending for coalesced KV gather & exact deduplication
    sentinel = tk + torch.arange(mem_p.shape[-1], device=dev, dtype=torch.long)
    key_for_sort = torch.where(m_p, mem_p, sentinel)
    key_sorted, _ = key_for_sort.sort(dim=-1)
    if key_sorted.shape[-1] > 1:
        is_dup = F.pad(key_sorted[..., 1:] == key_sorted[..., :-1], (1, 0), value=False)
    else:
        is_dup = torch.zeros_like(key_sorted, dtype=torch.bool)
    m_final = (key_sorted < tk) & (~is_dup)
    mem_final = torch.where(m_final, key_sorted, torch.zeros_like(key_sorted))
    if not sync_free:
        n_sel = int(m_final.sum())
    return mem_final, m_final, n_sel


def _effective_member_cap(cfg: LSAConfig, s_slots: int, kk: int, kb_n: int, tk: int) -> int:
    if cfg.max_selected_keys <= 0 or kk >= s_slots:
        return kb_n
    if cfg.num_select * cfg.avg_bucket_keys > cfg.max_selected_keys:
        return min(kb_n, max(cfg.max_selected_keys, min(tk, cfg.num_select * cfg.avg_bucket_keys * 2)))
    return min(kb_n, cfg.max_selected_keys)


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
) -> torch.Tensor:
    """`q [B,G,Hg,C,D]`, `qz [B,G_or_Hq,C,d]`, `qpos [C]` -> `out [B,G,Hg,C,D]`."""
    cfg = idx.cfg
    w = cfg.resolved_window()
    dev = q.device
    b, g, hg, c, dh = q.shape
    tk = k.shape[2]
    sync_free = _is_sync_restricted()

    lo, hi = max(0, int(qpos[0]) - w + 1) if not sync_free else 0, int(qpos[-1]) + 1 if not sync_free else tk
    kw, vw = k[:, :, lo:hi], v[:, :, lo:hi]
    kpos = torch.arange(lo, hi, device=dev)
    win_lo = qpos - w + 1
    m_win = (kpos[None] <= qpos[:, None]) & (kpos[None] >= win_lo[:, None])
    ns = min(cfg.sinks, hi)
    spos = torch.arange(ns, device=dev)
    m_sink = (spos[None] < win_lo[:, None]) & (spos[None] <= qpos[:, None])
    s_parts = [torch.einsum("bghcd,bgld->bghcl", q, kw) * scaling]
    masks = [m_win[None, None, None]]
    if ns:
        s_parts.append(torch.einsum("bghcd,bgld->bghcl", q, k[:, :, :ns]) * scaling)
        masks.append(m_sink[None, None, None])

    kb = vb = None
    per_head = cfg.per_head_select or (qz.shape[1] == g * hg and hg > 1)
    n_sel = 0
    if index is not None:
        with torch.no_grad():
            scores = idx.slot_scores(qz.detach(), index, qpos)
            s_slots = scores.shape[-1]
            kk = min(cfg.num_select, s_slots)
            top = scores.topk(kk, dim=-1)
            finite_top = torch.isfinite(top.values)
            sel_slots = torch.where(finite_top, top.indices, torch.full_like(top.indices, -1))
            if not _in_backward():
                idx.last_selected_buckets[cur_pass] = sel_slots
            mk = index.members.shape[-1]
            kb_n = kk * mk
            eff_cap = _effective_member_cap(cfg, s_slots, kk, kb_n, tk)

            if per_head:
                top_idx_g = top.indices.view(b, g, hg, c, kk)
                fin_g = finite_top.view(b, g, hg, c, kk)
                flat_sel = top_idx_g.reshape(b, g, hg * c * kk, 1).expand(b, g, hg * c * kk, mk)
                mem = index.members.gather(2, flat_sel).view(b, g, hg, c, kk, mk)
                mem = torch.where(fin_g.unsqueeze(-1), mem, torch.full_like(mem, -1)).flatten(-2)
                m_b = (mem >= ns) & (mem < win_lo[None, None, None, :, None])
            else:
                flat_sel = top.indices.reshape(b, g, c * kk, 1).expand(b, g, c * kk, mk)
                mem = index.members.gather(2, flat_sel).view(b, g, c, kk, mk)
                mem = torch.where(finite_top.unsqueeze(-1), mem, torch.full_like(mem, -1)).flatten(-2)
                m_b = (mem >= ns) & (mem < win_lo[None, None, :, None])

            packed = _pack_valid_members(mem, m_b, tk=tk, eff_cap=eff_cap, sync_free=sync_free)

        if packed is not None:
            mem_final, m_final, n_sel = packed
            l_pack = mem_final.shape[-1]
            if per_head:
                gidx = mem_final.reshape(b, g, hg * c * l_pack, 1).expand(b, g, hg * c * l_pack, dh)
                kb = k.gather(2, gidx).view(b, g, hg, c, l_pack, dh)
                vb = v.gather(2, gidx).view(b, g, hg, c, l_pack, dh)
                s_parts.append(torch.einsum("bghcd,bghckd->bghck", q, kb) * scaling)
                masks.append(m_final)
            else:
                gidx = mem_final.reshape(b, g, c * l_pack, 1).expand(b, g, c * l_pack, dh)
                kb = k.gather(2, gidx).view(b, g, c, l_pack, dh)
                vb = v.gather(2, gidx).view(b, g, c, l_pack, dh)
                s_parts.append(torch.einsum("bghcd,bgckd->bghck", q, kb) * scaling)
                masks.append(m_final[:, :, None])

    scores_all = torch.cat(
        [upcast(sp).masked_fill(~mk_i, float("-inf")) for sp, mk_i in zip(s_parts, masks)], dim=-1
    )
    attn = torch.softmax(scores_all, dim=-1)
    # Guard rows that might have all -inf (never occurs in causal self-attn since qpos is in window)
    attn = torch.nan_to_num(attn, nan=0.0).to(v.dtype)
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

    if not sync_free and not _in_backward():
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
    with torch.no_grad():
        s = torch.einsum("bghcd,bgtd->bghct", upcast(q), upcast(k[:, :, :hi])) * scaling
        kpos = torch.arange(hi, device=q.device)
        s = s.masked_fill(kpos[None, None, None, None] > qpos[None, None, None, :, None], float("-inf"))
        a = torch.softmax(s, dim=-1)
        nk = min(hi, index.key2slot.shape[-1])
        a = a[..., :nk] * (kpos[:nk][None] < frontier[:, None])[None, None, None]
        b, g, hg, c, _ = a.shape
        slots = index.key2slot[:, :, :nk][:, :, None, None, :].expand(b, g, hg, c, nk)
        mass_h = torch.zeros(b, g, hg, c, s_slots, device=q.device, dtype=a.dtype)
        mass_h = mass_h.scatter_add_(-1, slots, a)
        mass = mass_h.reshape(b, g * hg, c, s_slots) if per_head else mass_h.mean(2)
        tot = mass.sum(-1, keepdim=True)
        target = mass / tot.clamp_min(1e-30)
        ok = valid[None, None, :] & (tot.squeeze(-1) > 0)
    logp = torch.log_softmax(idx.slot_scores(qz, index, qpos, flat=True), dim=-1)
    logp = torch.where(target > 0, logp, torch.zeros_like(logp))
    kl = (target * (torch.log(target.clamp_min(1e-30)) - logp)).sum(-1)
    return kl[ok].mean() if bool(ok.any()) else None


def _resolve_hybrid_sparse(
    idx: LSAIndexer,
    q: torch.Tensor,
    k: torch.Tensor,
    qpos_all: torch.Tensor,
    index: BucketIndex | None,
    cur_pass: int,
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
    if (
        not ready
        and index is not None
        and tk > cfg.sinks + win
        and not _in_backward()
        and not _is_sync_restricted()
    ):
        check_every = max(1, int(cfg.recall_check_every))
        due = (idx._recall_checks == 0) or (
            step != idx._last_recall_step and (step % check_every) == 0
        )
        if due:
            with torch.no_grad():
                cand_r = torch.nonzero(qpos_all >= (cfg.sinks + win)).flatten()
                if cand_r.numel() > 0:
                    take = min(16, int(cand_r.numel()))
                    stride = max(1, int(cand_r.numel()) // take)
                    sel_r = cand_r[::stride][:take]
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
    if cache_key is None and "past_key_values" in kwargs and kwargs["past_key_values"] is not None:
        pkv_root, pkv_ok = _unwrap_pkv_for_cache(kwargs["past_key_values"])
        if pkv_ok and pkv_root is not None:
            idx._active_pkv = (_weakref_or_none(pkv_root), id(pkv_root), True)

    b, h, tq, dh = query.shape
    g, tk = key.shape[1], key.shape[2]
    scaling = dh ** -0.5 if scaling is None else scaling
    qpos_all = torch.arange(tk - tq, tk, device=query.device)
    qin = query.detach() if cfg.detach_inputs else query
    cur_pass = idx._resolve_pass(module=module)
    active_wq = idx.active_wq(pass_idx=cur_pass)

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
                index = idx.build(key, cache_key=cache_key)
        else:
            index = idx.build(key, cache_key=cache_key)
    else:
        index = None

    if cfg.mode == "sparse":
        sparse = tk > cfg.resolved_window()
    elif cfg.mode == "hybrid":
        sparse = _resolve_hybrid_sparse(idx, qin, key, qpos_all, index, cur_pass)
    else:
        sparse = False

    if sparse and attention_mask is not None and cfg.strict_mask:
        raise RuntimeError(
            "LSA sparse mode does not support padding masks (strict_mask=true); "
            "use sequence packing without padding"
        )

    if want_kl and index is not None:
        if cfg.detach_inputs:
            # Exact zero-checkpoint-unpack, O(distill_chunk * S) memory KL distillation:
            # isolates `lsa_kl` from any outer `VirtualLayer` `checkpoint(use_reentrant=False)` frame
            # via identity `saved_tensors_hooks`, computes exact `(dKL/dwq, dKL/dwk)` chunk-by-chunk,
            # and appends an exact first-order surrogate scalar to `idx.pending`.
            if not _in_backward():
                with torch.autograd.graph.saved_tensors_hooks(lambda x: x, lambda x: x):
                    cand = torch.nonzero((qpos_all // cfg.block_size) > 0).flatten()
                    if cand.numel():
                        nq = max(1, int(cand.numel() * cfg.distill_query_frac))
                        pick = cand[torch.randperm(cand.numel(), device=cand.device)[:nq]].sort().values
                        qg_kl = qin.view(b, g, h // g, tq, dh)
                        need_wq = bool(active_wq.requires_grad)
                        need_wk = bool(idx.wk.requires_grad)
                        cent_leaf = index.centroids.detach().requires_grad_(need_wk)
                        kl_index = replace(index, centroids=cent_leaf)
                        g_wq_sum: torch.Tensor | None = None
                        g_cent_sum: torch.Tensor | None = None
                        loss_sum = 0.0
                        n_chunks_ok = 0
                        for s0 in range(0, pick.numel(), cfg.distill_chunk):
                            sel = pick[s0 : s0 + cfg.distill_chunk]
                            qz = idx.query_latent(qin[:, :, sel], pass_idx=cur_pass)
                            kl = _distill_kl(
                                qg_kl[:, :, :, sel], qz, qpos_all[sel], key.detach(), kl_index, idx, scaling
                            )
                            if kl is None:
                                del qz
                                continue
                            loss_sum += float(kl.detach())
                            n_chunks_ok += 1
                            grad_targets: list[torch.Tensor] = []
                            if need_wq:
                                grad_targets.append(active_wq)
                            if need_wk:
                                grad_targets.append(cent_leaf)
                            if grad_targets and kl.requires_grad:
                                grads = torch.autograd.grad(kl, grad_targets, allow_unused=True)
                                gi_t = 0
                                if need_wq:
                                    gwq_c = grads[gi_t]
                                    gi_t += 1
                                    if gwq_c is not None:
                                        g_wq_sum = gwq_c.detach() if g_wq_sum is None else (g_wq_sum + gwq_c.detach())
                                if need_wk:
                                    gct_c = grads[gi_t]
                                    if gct_c is not None:
                                        g_cent_sum = (
                                            gct_c.detach() if g_cent_sum is None else (g_cent_sum + gct_c.detach())
                                        )
                            del qz, kl
                        if n_chunks_ok > 0:
                            inv_n = 1.0 / float(n_chunks_ok)
                            loss_val = loss_sum * inv_n
                            g_wq_avg = (g_wq_sum * inv_n) if g_wq_sum is not None else None
                            g_wk_avg: torch.Tensor | None = None
                            if (
                                need_wk
                                and g_cent_sum is not None
                                and index.slot_labels is not None
                                and index.slot_counts is not None
                            ):
                                g_cent_avg = g_cent_sum * inv_n
                                bs = cfg.block_size
                                pool = cfg.pool
                                n_units = bs // pool
                                spb = index.slots_per_block or (index.centroids.shape[2] // max(1, index.n_blocks))
                                kin_det = key.detach()
                                g_wk_acc = torch.zeros_like(idx.wk)
                                for b0 in range(0, index.n_blocks, 64):
                                    b1 = min(index.n_blocks, b0 + 64)
                                    nb_c = b1 - b0
                                    z_b = (
                                        idx.key_latent(kin_det[:, :, b0 * bs : b1 * bs])
                                        .view(b, g, nb_c, n_units, pool, -1)
                                        .sum(4)
                                    )
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
                            surrogate = active_wq.new_tensor(loss_val)
                            if need_wq and g_wq_avg is not None:
                                term_q = (active_wq * g_wq_avg.to(active_wq.dtype)).sum()
                                surrogate = surrogate + (term_q - term_q.detach())
                            if need_wk and g_wk_avg is not None:
                                term_k = (idx.wk * g_wk_avg.to(idx.wk.dtype)).sum()
                                surrogate = surrogate + (term_k - term_k.detach())
                            idx.pending.append(cfg.distill_weight * surrogate)
                            idx.last_stats["distill_kl"] = loss_val
        else:
            cand = torch.nonzero((qpos_all // cfg.block_size) > 0).flatten()
            if cand.numel():
                nq = max(1, int(cand.numel() * cfg.distill_query_frac))
                pick = cand[torch.randperm(cand.numel(), device=cand.device)[:nq]].sort().values
                qg_kl = qin.view(b, g, h // g, tq, dh)
                kls = []
                for s0 in range(0, pick.numel(), cfg.distill_chunk):
                    sel = pick[s0 : s0 + cfg.distill_chunk]
                    qz = idx.query_latent(qin[:, :, sel], pass_idx=cur_pass)
                    kl = _distill_kl(qg_kl[:, :, :, sel], qz, qpos_all[sel], key.detach(), index, idx, scaling)
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

    qg = query.view(b, g, h // g, tq, dh)
    with torch.no_grad():
        qz_all = idx.query_latent(qin.detach(), pass_idx=cur_pass)
    outs = []
    for c0 in range(0, tq, cfg.query_chunk):
        sl = slice(c0, min(tq, c0 + cfg.query_chunk))

        def run(
            qc: torch.Tensor,
            kc: torch.Tensor,
            vc: torch.Tensor,
            qzc: torch.Tensor,
            sl: slice = sl,
            p: int = cur_pass,
        ) -> torch.Tensor:
            return _sparse_chunk(qc, qzc, qpos_all[sl], kc, vc, index, idx, scaling, cur_pass=p)

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

    def aux_loss(self) -> torch.Tensor | None:
        losses = [loss for ix in self.indexers.values() for loss in ix.pop_losses()]
        return torch.stack(losses).sum() if losses else None

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
        pkv = kw.get("past_key_values")
        if pkv is None:
            pkv = kw.get("past_key_value")
        if pkv is None:
            pkv = kw.get("cache")
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
