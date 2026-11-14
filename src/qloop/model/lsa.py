"""Distilled Latent Sparse Attention (D-LSA) for the gated full-attention layers.

Pipeline (Latent Sparse Attention, arXiv 2610.42147, D-LSA variant):
  1. index projections  q -> q_idx, k -> k_idx in a learned low-dim latent space (default 64),
     L2-normalised. We project only the *non-rotary* part of each head (Qwen3.6: dims 64..255 of
     256), so clustering is driven by content, not by RoPE phase.
  2. 4:1 pooling of consecutive key latents into "units".
  3. Ward (minimum-variance) agglomerative clustering of the units inside every complete block
     of `block_size` keys into buckets (avg `avg_bucket_keys`, hard cap `max_bucket_keys`).
     Each bucket stores a normalised centroid and its member key positions (bucket cache).
  4. every query scores bucket centroids (flat search, or a two-level tree search over
     super-nodes of `tree_group` blocks with beam `tree_beam`) and keeps the top `num_select`.
  5. attention runs over   local window  U  sink tokens  U  keys of the selected buckets
     (duplicates removed), with a standard softmax - only *which* keys are read changes.
  6. D-LSA training: the base model is frozen; only the index projections learn, by minimising
     KL( A_full(bucket) || softmax(q_idx . c_bucket / tau) ) on sampled queries, where
     A_full(bucket) is the true attention mass of each bucket (heads of the KV group averaged).

Engineering choices (documented deviations):
  * selection is **per query**. Sharing a selection across a query block would make the key set
    of token t depend on later queries of its block (non-causal) and break decode equivalence.
  * a bucket is eligible for query t only if its block ends <= floor(t / block) * block; the
    window (>= block_size) therefore always covers every not-yet-clustered key, so attention
    with all buckets selected equals dense attention exactly.
  * clustering is computed once per key tensor and memoised (fingerprint), so with recursive KV
    sharing pass 2/3 reuse pass-1's bucket index (only queries are recomputed). In inference
    (no grad) centroids/members are cached and extended incrementally block by block.
"""
from __future__ import annotations

import logging
import math
import weakref
from collections import OrderedDict
from dataclasses import dataclass, fields
from typing import Any, Callable

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from ._dtype import upcast
from .introspect import FULL, attention_module, find_text_model, inspect_model

log = logging.getLogger("qloop")

LSA_IMPL_NAME = "qloop_lsa"


# ============================================================================ config

@dataclass
class LSAConfig:
    mode: str = "distill"           # dense | distill | sparse
    layers: list[int] | None = None  # physical full-attention layers; None = all of them
    index_dim: int = 64
    index_on: str = "nope"          # "nope" (non-rotary dims) | "all"
    rotary_dim: int | None = None   # default: head_dim * partial_rotary_factor
    pool: int = 4
    block_size: int = 512
    avg_bucket_keys: int = 8
    max_bucket_keys: int = 32
    num_select: int = 64
    window: int | None = None       # default: block_size
    sinks: int = 64
    search: str = "flat"            # flat | tree
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
    num_query_passes: int = 1       # 1 = shared query indexer; >1 (e.g. 3) = per-pass q_proj_extra for passes 2..R

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

    def validate(self) -> None:
        if self.mode not in ("dense", "distill", "sparse"):
            raise ValueError(f"lsa.mode must be dense|distill|sparse, got {self.mode!r}")
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


# ============================================================================ clustering

def ward_cluster(x: torch.Tensor, n_clusters: int, max_size: int) -> torch.Tensor:
    """Batched Ward agglomerative clustering.

    x: [N, n, d] points (one independent problem per row). Returns labels [N, n] where each
    point's label is the index of its cluster representative (the smallest member index).
    Merges never create clusters larger than `max_size` points; if the cap blocks further
    merges a row keeps more than `n_clusters` clusters.
    """
    nrow, n, _ = x.shape
    dev = x.device
    if n_clusters >= n:
        return torch.arange(n, device=dev).expand(nrow, n).clone()
    x = upcast(x)
    cent = x.clone()
    size = torch.ones(nrow, n, device=dev, dtype=x.dtype)
    active = torch.ones(nrow, n, device=dev, dtype=torch.bool)
    labels = torch.arange(n, device=dev).expand(nrow, n).clone()
    inf = torch.finfo(x.dtype).max
    eye = torch.eye(n, device=dev, dtype=torch.bool)
    dist = 0.5 * torch.cdist(x, x).pow(2)
    dist = dist.masked_fill(eye, inf)
    if max_size < 2:
        return labels
    rows = torch.arange(nrow, device=dev)
    for _ in range(n - n_clusters):
        flat = dist.view(nrow, -1)
        best, idx = flat.min(dim=-1)
        ok = best < inf
        if not bool(ok.any()):
            break
        i, j = idx // n, idx % n
        a, b = torch.minimum(i, j), torch.maximum(i, j)
        r, a, b = rows[ok], a[ok], b[ok]
        na, nb = size[r, a], size[r, b]
        cent[r, a] = (na[:, None] * cent[r, a] + nb[:, None] * cent[r, b]) / (na + nb)[:, None]
        size[r, a] = na + nb
        size[r, b] = 0
        active[r, b] = False
        lab = labels[r]
        labels[r] = torch.where(lab == b[:, None], a[:, None], lab)
        # Ward distance of the merged cluster to every other active cluster
        sa = size[r, a][:, None]
        sk = size[r]
        d_new = (sa * sk / (sa + sk).clamp_min(1e-12)) * (cent[r] - cent[r, a][:, None]).pow(2).sum(-1)
        bad = (~active[r]) | (sa + sk > max_size)
        bad[torch.arange(len(r), device=dev), a] = True
        d_new = d_new.masked_fill(bad, inf)
        dist[r, a, :] = d_new
        dist[r, :, a] = d_new
        dist[r, b, :] = inf
        dist[r, :, b] = inf
    return labels


def members_table(labels: torch.Tensor, max_members: int) -> torch.Tensor:
    """labels [N, n] (representative index per point) -> [N, n, max_members] member indices, -1 padded."""
    nrow, n = labels.shape
    dev = labels.device
    ar = torch.arange(n, device=dev)
    order = torch.argsort(labels * n + ar, dim=-1)
    sorted_lab = labels.gather(-1, order)
    counts = torch.zeros(nrow, n, device=dev, dtype=torch.long).scatter_add_(1, labels, torch.ones_like(labels))
    if int(counts.max()) > max_members:
        raise RuntimeError(f"bucket larger than max_members={max_members} (max {int(counts.max())})")
    offsets = counts.cumsum(-1) - counts
    rank = ar.expand(nrow, n) - offsets.gather(1, sorted_lab)
    out = torch.full((nrow, n, max_members), -1, device=dev, dtype=torch.long)
    rr = torch.arange(nrow, device=dev)[:, None].expand(nrow, n)
    out[rr, sorted_lab, rank] = order
    return out


# ============================================================================ bucket index

@dataclass
class BucketIndex:
    n_blocks: int
    labels: torch.Tensor      # [B,G,nb,n]       representative unit per unit
    centroids: torch.Tensor   # [B,G,S,d]        S = nb*n slots (normalised; zeros if inactive)
    active: torch.Tensor      # [B,G,S] bool
    members: torch.Tensor     # [B,G,S,max_keys] key positions, -1 padded
    slot_end: torch.Tensor    # [S] exclusive end position of the slot's block
    key2slot: torch.Tensor    # [B,G,nb*block]   slot id of every clustered key


@dataclass
class _CacheEntry:
    n_blocks: int
    fp_first: tuple[float, float]
    fp_last: tuple[float, float]
    labels: torch.Tensor
    unit_members: torch.Tensor
    centroids: torch.Tensor | None   # only kept for no-grad (inference) use


def _fingerprint(block: torch.Tensor) -> tuple[float, float]:
    f = upcast(block)
    return float(f.sum()), float(f.abs().sum())


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
        # same JL projection for keys and their query heads => q_idx.k_idx ~ cos(q, k) at init
        self.wk = nn.Parameter(base.clone())
        self.wq = nn.Parameter(base.repeat_interleave(self.hg, dim=0).clone())
        # For passes 2..R: copy pass-1 `wq` weights so step-0 bucket selection is bit-identical
        self.q_proj_extra = nn.ModuleList(
            [_PassQueryProj(self.wq.detach()) for _ in range(max(0, cfg.num_query_passes - 1))]
        )
        self._current_pass: int = 1
        self._cache: OrderedDict[tuple, _CacheEntry] = OrderedDict()
        self.pending: list[torch.Tensor] = []
        self.last_stats: dict[str, float] = {}
        self.last_selected_buckets: dict[int, torch.Tensor] = {}

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

    def _resolve_pass(self, module: nn.Module | None = None, pass_idx: int | None = None) -> int:
        cap = max(1, self.cfg.num_query_passes)
        if pass_idx is not None:
            return 1 if int(pass_idx) <= 1 else min(int(pass_idx), cap)
        tm_ref = getattr(self, "_tm_ref", None)
        tm = tm_ref() if callable(tm_ref) else None
        if tm is not None:
            layers = getattr(tm, "layers", None)
            if layers is not None and len(layers) > 0:
                ctl = getattr(layers[0], "_ctl", None)
                cur = getattr(ctl, "_cur", None) if ctl is not None else None
                if cur is not None:
                    r = int(cur[1])
                    return 1 if r <= 1 else min(r, cap)
        mod_r = getattr(module, "_current_pass", None) if module is not None else None
        if mod_r is not None:
            return 1 if int(mod_r) <= 1 else min(int(mod_r), cap)
        return 1 if self._current_pass <= 1 else min(self._current_pass, cap)

    def active_wq(self, pass_idx: int | None = None, module: nn.Module | None = None) -> nn.Parameter:
        r = self._resolve_pass(module=module, pass_idx=pass_idx)
        if r <= 1 or self.cfg.num_query_passes <= 1:
            return self.wq
        proj: _PassQueryProj = self.q_proj_extra[r - 2]  # type: ignore[assignment]
        return proj.weight

    # --------------------------------------------------------------- latents
    def query_latent(
        self, q: torch.Tensor, pass_idx: int | None = None, module: nn.Module | None = None
    ) -> torch.Tensor:
        """q [B,H,T,D] -> group query latent [B,G,T,d] (sum of normalised head latents)."""
        r = self._resolve_pass(module=module, pass_idx=pass_idx)
        q_sl = q[..., self.lo:]
        if r <= 1 or self.cfg.num_query_passes <= 1:
            z = torch.einsum("bhtd,hde->bhte", q_sl.to(self.wq.dtype), self.wq)
        else:
            z = self.q_proj_extra[r - 2](q_sl)
        z = F.normalize(upcast(z), dim=-1)
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
        """Return `[B, G, Tq, kk]` selected bucket slot indices (`-1` where ineligible)."""
        tq = q.shape[2]
        if qpos is None:
            tk = index.n_blocks * self.cfg.block_size + self.cfg.resolved_window()
            qpos = torch.arange(tk - tq, tk, device=q.device)
        qz = self.query_latent(q, pass_idx=pass_idx)
        scores = self.slot_scores(qz, index, qpos)
        kk = min(self.cfg.num_select, scores.shape[-1])
        top = scores.topk(kk, dim=-1)
        return torch.where(torch.isfinite(top.values), top.indices, torch.full_like(top.indices, -1))

    # --------------------------------------------------------------- bucket index
    def _cluster_blocks(self, zblocks: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """zblocks [B,G,nb,block,d] -> labels [B,G,nb,n], unit member table [B,G,nb,n,max_units]."""
        cfg = self.cfg
        b, g, nb, _, d = zblocks.shape
        n = cfg.block_size // cfg.pool
        units = F.normalize(zblocks.reshape(b, g, nb, n, cfg.pool, d).mean(4), dim=-1)
        n_clusters = max(1, cfg.block_size // cfg.avg_bucket_keys)
        max_units = cfg.max_bucket_keys // cfg.pool
        labels = ward_cluster(units.reshape(-1, n, d), n_clusters, max_units)
        mem = members_table(labels, max_units)
        return labels.view(b, g, nb, n), mem.view(b, g, nb, n, max_units)

    def bucket_index(self, k: torch.Tensor) -> BucketIndex | None:
        cfg = self.cfg
        b, g, tk, _ = k.shape
        bs, n, pool = cfg.block_size, cfg.block_size // cfg.pool, cfg.pool
        nb = tk // bs
        if nb == 0:
            return None
        kin = k.detach() if cfg.detach_inputs else k
        grad_mode = torch.is_grad_enabled() and (self.wk.requires_grad or kin.requires_grad)
        fp0 = _fingerprint(kin[:, :, :bs])
        key = (b, g, str(k.device), fp0)
        ent = self._cache.get(key)
        reuse = 0
        if ent is not None and ent.n_blocks <= nb:
            lastb = kin[:, :, (ent.n_blocks - 1) * bs: ent.n_blocks * bs]
            if _fingerprint(lastb) == ent.fp_last:
                reuse = ent.n_blocks
        with torch.no_grad():
            if reuse < nb:
                znew = self.key_latent(kin[:, :, reuse * bs: nb * bs]).view(b, g, nb - reuse, bs, -1)
                lab_new, mem_new = self._cluster_blocks(znew)
            if reuse:
                labels = ent.labels if reuse == nb else torch.cat([ent.labels, lab_new], 2)
                umem = ent.unit_members if reuse == nb else torch.cat([ent.unit_members, mem_new], 2)
            else:
                labels, umem = lab_new, mem_new
        # centroids: differentiable recompute in grad mode, cached/incremental otherwise
        if grad_mode:
            z = self.key_latent(kin[:, :, : nb * bs]).view(b, g, nb, n, pool, -1).sum(4)
            cent = self._centroids(z, labels)
        else:
            with torch.no_grad():
                if reuse and ent.centroids is not None:
                    if reuse < nb:
                        zn = self.key_latent(kin[:, :, reuse * bs: nb * bs]).view(b, g, nb - reuse, n, pool, -1).sum(4)
                        cent = torch.cat([ent.centroids, self._centroids(zn, lab_new)], 2)
                    else:
                        cent = ent.centroids
                else:
                    z = self.key_latent(kin[:, :, : nb * bs]).view(b, g, nb, n, pool, -1).sum(4)
                    cent = self._centroids(z, labels)
        self._cache[key] = _CacheEntry(nb, fp0, _fingerprint(kin[:, :, (nb - 1) * bs: nb * bs]),
                                       labels, umem, None if grad_mode else cent.detach())
        self._cache.move_to_end(key)
        while len(self._cache) > cfg.cache_entries:
            self._cache.popitem(last=False)
        return self._assemble(labels, umem, cent)

    @staticmethod
    def _centroids(unit_lat: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """unit_lat [B,G,nb,n,d] (sum of key latents per unit), labels [B,G,nb,n] -> [B,G,nb,n,d]."""
        idx = labels.unsqueeze(-1).expand_as(unit_lat)
        cent = torch.zeros_like(unit_lat).scatter_add(3, idx, unit_lat)
        return F.normalize(cent, dim=-1)

    def _assemble(self, labels: torch.Tensor, umem: torch.Tensor, cent: torch.Tensor) -> BucketIndex:
        cfg = self.cfg
        b, g, nb, n = labels.shape
        bs, pool, dev = cfg.block_size, cfg.pool, labels.device
        counts = torch.zeros_like(labels).scatter_add_(3, labels, torch.ones_like(labels))
        active = (counts > 0).view(b, g, nb * n)
        blk = torch.arange(nb, device=dev)
        # unit -> key positions
        upos = blk.view(1, 1, nb, 1, 1) * bs + umem * pool                     # [B,G,nb,n,mu]
        keys = upos.unsqueeze(-1) + torch.arange(pool, device=dev)              # [B,G,nb,n,mu,pool]
        keys = torch.where(umem.unsqueeze(-1) >= 0, keys, torch.full_like(keys, -1))
        members = keys.view(b, g, nb * n, -1)
        slot_end = ((torch.arange(nb * n, device=dev) // n) + 1) * bs
        unit_of_key = (torch.arange(nb * bs, device=dev) % bs) // pool
        blk_of_key = torch.arange(nb * bs, device=dev) // bs
        key2slot = blk_of_key * n + labels.view(b, g, nb, n)[:, :, blk_of_key, unit_of_key]
        return BucketIndex(nb, labels, cent.reshape(b, g, nb * n, -1), active, members, slot_end, key2slot)

    # --------------------------------------------------------------- selection
    def slot_scores(self, qz: torch.Tensor, index: BucketIndex, qpos: torch.Tensor, flat: bool = False) -> torch.Tensor:
        """qz [B,G,C,d], qpos [C] -> masked scores [B,G,C,S] (-inf where not eligible)."""
        cfg = self.cfg
        frontier = (qpos // cfg.block_size) * cfg.block_size
        elig = index.active[:, :, None, :] & (index.slot_end[None, None, None, :] <= frontier[None, None, :, None])
        s = torch.einsum("bgcd,bgsd->bgcs", qz, index.centroids.to(qz.dtype)) / cfg.temperature
        if cfg.search == "tree" and not flat:
            s = s.masked_fill(~self._tree_mask(qz, index, elig, qpos), float("-inf"))
        return s.masked_fill(~elig, float("-inf"))

    def _tree_mask(self, qz: torch.Tensor, index: BucketIndex, elig: torch.Tensor, qpos: torch.Tensor) -> torch.Tensor:
        """Two-level search: keep only slots inside the top `tree_beam` super-nodes per query.

        Super-node centroids are *causal per query*: a super-node only aggregates blocks that
        are already eligible for that query (prefix sums inside the super-node), otherwise
        selection would leak information from keys the query may not see yet.
        """
        cfg = self.cfg
        b, g, c, s = elig.shape
        n, tg = cfg.block_size // cfg.pool, cfg.tree_group
        nb = index.n_blocks
        ng = -(-nb // tg)
        if ng <= cfg.tree_beam:
            return torch.ones_like(elig)
        cent = index.centroids.view(b, g, nb, n, -1) * index.active.view(b, g, nb, n, 1)
        bsum = cent.sum(3)                                                               # [B,G,nb,d]
        pad = ng * tg - nb
        if pad:
            bsum = F.pad(bsum, (0, 0, 0, pad))
        cs = bsum.view(b, g, ng, tg, -1).cumsum(3)                                       # prefix within super
        full = F.normalize(cs[:, :, :, -1], dim=-1)                                      # [B,G,ng,d]
        fb = qpos // cfg.block_size                                                      # eligible blocks
        jq, rem = fb // tg, fb % tg
        ss = torch.einsum("bgcd,bgnd->bgcn", qz, full.to(qz.dtype))
        part = F.normalize(cs[:, :, jq.clamp(max=ng - 1), (rem - 1).clamp(min=0)], dim=-1)  # [B,G,C,d]
        sp = (qz * part.to(qz.dtype)).sum(-1)
        jn = torch.arange(ng, device=qz.device)
        is_part = jn[None, :] == jq[:, None]                                             # [C,ng]
        ss = torch.where(is_part[None, None], sp.unsqueeze(-1), ss)
        valid = (jn[None, :] < jq[:, None]) | (is_part & (rem[:, None] > 0))
        ss = ss.masked_fill(~valid[None, None], float("-inf"))
        top = ss.topk(min(cfg.tree_beam, ng), dim=-1)
        keep_sup = torch.zeros(b, g, c, ng, dtype=torch.bool, device=qz.device)
        keep_sup.scatter_(-1, top.indices, torch.isfinite(top.values))
        return keep_sup.repeat_interleave(tg * n, dim=-1)[..., :s]

    # --------------------------------------------------------------- losses
    def pop_losses(self) -> list[torch.Tensor]:
        out, self.pending = self.pending, []
        return out


# ============================================================================ attention

def _in_backward() -> bool:
    fn = getattr(torch._C, "_current_graph_task_id", None)
    return fn is not None and fn() != -1


def _sparse_chunk(q: torch.Tensor, qz: torch.Tensor, qpos: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                  index: BucketIndex | None, idx: LSAIndexer, scaling: float, cur_pass: int = 1) -> torch.Tensor:
    """q [B,G,Hg,C,D], qz [B,G,C,d], qpos [C] -> out [B,G,Hg,C,D]."""
    cfg = idx.cfg
    w = cfg.resolved_window()
    dev = q.device
    lo, hi = max(0, int(qpos[0]) - w + 1), int(qpos[-1]) + 1
    kw, vw = k[:, :, lo:hi], v[:, :, lo:hi]
    kpos = torch.arange(lo, hi, device=dev)
    win_lo = qpos - w + 1
    m_win = (kpos[None] <= qpos[:, None]) & (kpos[None] >= win_lo[:, None])          # [C,Lw]
    ns = min(cfg.sinks, int(qpos[-1]) + 1)
    spos = torch.arange(ns, device=dev)
    m_sink = (spos[None] < win_lo[:, None]) & (spos[None] <= qpos[:, None])           # [C,ns]
    s_parts = [torch.einsum("bghcd,bgld->bghcl", q, kw) * scaling]
    masks = [m_win[None, None, None]]
    if ns:
        s_parts.append(torch.einsum("bghcd,bgld->bghcl", q, k[:, :, :ns]) * scaling)
        masks.append(m_sink[None, None, None])
    kb = vb = None
    n_sel = 0
    if index is not None:
        scores = idx.slot_scores(qz, index, qpos)                                     # [B,G,C,S]
        kk = min(cfg.num_select, scores.shape[-1])
        top = scores.topk(kk, dim=-1)
        sel_slots = torch.where(
            torch.isfinite(top.values.detach()),
            top.indices,
            torch.full_like(top.indices, -1),
        )
        if not _in_backward():
            idx.last_selected_buckets[cur_pass] = sel_slots
        mem = index.members.unsqueeze(2).expand(-1, -1, qz.shape[2], -1, -1)          # [B,G,C,S,mk]
        mem = mem.gather(3, top.indices.unsqueeze(-1).expand(-1, -1, -1, -1, mem.shape[-1]))
        mem = torch.where(torch.isfinite(top.values).unsqueeze(-1), mem, torch.full_like(mem, -1))
        mem = mem.flatten(-2)                                                         # [B,G,C,Kb]
        m_b = (mem >= ns) & (mem < win_lo[None, None, :, None])
        b, g, c, kb_n = mem.shape
        gidx = mem.clamp_min(0).view(b, g, c * kb_n, 1).expand(-1, -1, -1, k.shape[-1])
        kb = k.gather(2, gidx).view(b, g, c, kb_n, -1)
        vb = v.gather(2, gidx).view(b, g, c, kb_n, -1)
        s_parts.append(torch.einsum("bghcd,bgckd->bghck", q, kb) * scaling)
        masks.append(m_b[:, :, None])
        n_sel = int(m_b.sum())
    scores_all = torch.cat([upcast(sp).masked_fill(~mk, float("-inf")) for sp, mk in zip(s_parts, masks)], -1)
    attn = torch.softmax(scores_all, dim=-1).to(v.dtype)
    sizes = [sp.shape[-1] for sp in s_parts]
    parts = torch.split(attn, sizes, dim=-1)
    out = torch.einsum("bghcl,bgld->bghcd", parts[0], vw)
    pi = 1
    if ns:
        out = out + torch.einsum("bghcl,bgld->bghcd", parts[1], v[:, :, :ns])
        pi = 2
    if kb is not None:
        out = out + torch.einsum("bghck,bgckd->bghcd", parts[pi], vb)
    c = len(qpos)
    per_q_bucket = n_sel / max(1, q.shape[0] * q.shape[1] * c)
    kpq = float(m_win.sum() + m_sink.sum()) / c + per_q_bucket
    idx.last_stats["keys_per_query"] = kpq
    idx.last_stats[f"keys_per_query_p{cur_pass}"] = kpq
    return out


def _distill_kl(q: torch.Tensor, qz: torch.Tensor, qpos: torch.Tensor, k: torch.Tensor,
                index: BucketIndex, idx: LSAIndexer, scaling: float) -> torch.Tensor | None:
    """KL(true bucket mass || indexer distribution) for queries qpos (q [B,G,Hg,C,D])."""
    cfg = idx.cfg
    frontier = (qpos // cfg.block_size) * cfg.block_size
    valid = frontier > 0
    if not bool(valid.any()):
        return None
    hi = int(qpos[-1]) + 1
    with torch.no_grad():
        s = torch.einsum("bghcd,bgtd->bghct", upcast(q), upcast(k[:, :, :hi])) * scaling
        kpos = torch.arange(hi, device=q.device)
        s = s.masked_fill(kpos[None, None, None, None] > qpos[None, None, None, :, None], float("-inf"))
        a = torch.softmax(s, dim=-1)                                                   # [B,G,Hg,C,hi]
        nk = min(hi, index.key2slot.shape[-1])
        a = a[..., :nk] * (kpos[:nk][None] < frontier[:, None])[None, None, None]
        b, g, hg, c, _ = a.shape
        slots = index.key2slot[:, :, :nk][:, :, None, None, :].expand(b, g, hg, c, nk)
        mass = torch.zeros(b, g, hg, c, index.centroids.shape[2], device=q.device, dtype=a.dtype)
        mass = mass.scatter_add_(-1, slots, a).mean(2)                                  # [B,G,C,S]
        tot = mass.sum(-1, keepdim=True)
        target = mass / tot.clamp_min(1e-30)
        ok = valid[None, None, :] & (tot.squeeze(-1) > 0)
    logp = torch.log_softmax(idx.slot_scores(qz, index, qpos, flat=True), dim=-1)
    logp = torch.where(target > 0, logp, torch.zeros_like(logp))
    kl = (target * (torch.log(target.clamp_min(1e-30)) - logp)).sum(-1)               # [B,G,C]
    return kl[ok].mean() if bool(ok.any()) else None


def lsa_attention(module: nn.Module, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                  attention_mask: torch.Tensor | None = None, scaling: float | None = None,
                  dropout: float = 0.0, **kwargs: Any) -> tuple[torch.Tensor, None]:
    """HF attention-interface function. q [B,H,Tq,D], k/v [B,G,Tk,D] -> ([B,Tq,H,D], None)."""
    idx: LSAIndexer | None = getattr(module, "lsa_indexer", None)
    dense = _DENSE_FN[0]
    if dense is None:
        raise RuntimeError("lsa_attention called before install_lsa() registered a dense fallback")
    if idx is None or idx.cfg.mode == "dense":
        return dense(module, query, key, value, attention_mask, scaling=scaling, dropout=dropout, **kwargs)
    cfg = idx.cfg
    b, h, tq, dh = query.shape
    g, tk = key.shape[1], key.shape[2]
    scaling = dh ** -0.5 if scaling is None else scaling
    qpos_all = torch.arange(tk - tq, tk, device=query.device)
    qin = query.detach() if cfg.detach_inputs else query
    cur_pass = idx._resolve_pass(module=module)
    active_wq = idx.active_wq(pass_idx=cur_pass)
    # for Tk <= window every key is inside the local window => sparse == dense exactly
    sparse = cfg.mode == "sparse" and tk > cfg.resolved_window()
    if sparse and attention_mask is not None and cfg.strict_mask:
        raise RuntimeError("LSA sparse mode does not support padding masks (strict_mask=true); "
                           "use sequence packing without padding")
    # NOTE: computed identically during checkpoint recomputation (same RNG state is restored), so
    # the saved-tensor structure matches; only the *registration* of the loss is skipped then.
    want_kl = (cfg.distill_weight > 0 and torch.is_grad_enabled()
               and (active_wq.requires_grad or idx.wk.requires_grad)
               and tk >= cfg.block_size + 1)
    index = idx.bucket_index(key) if (sparse or want_kl) else None
    if want_kl and index is not None:
        cand = torch.nonzero((qpos_all // cfg.block_size) > 0).flatten()
        if cand.numel():
            nq = max(1, int(cand.numel() * cfg.distill_query_frac))
            pick = cand[torch.randperm(cand.numel(), device=cand.device)[:nq]].sort().values
            qg = qin.view(b, g, h // g, tq, dh)
            kls = []
            for s0 in range(0, pick.numel(), cfg.distill_chunk):
                sel = pick[s0:s0 + cfg.distill_chunk]
                qz = idx.query_latent(qin[:, :, sel], pass_idx=cur_pass)
                kl = _distill_kl(qg[:, :, :, sel], qz, qpos_all[sel], key.detach(), index, idx, scaling)
                if kl is not None:
                    kls.append(kl)
            if kls and not _in_backward():
                loss = torch.stack(kls).mean()
                idx.pending.append(cfg.distill_weight * loss)
                idx.last_stats["distill_kl"] = float(loss.detach())
    if not sparse:
        return dense(module, query, key, value, attention_mask, scaling=scaling, dropout=dropout, **kwargs)
    qg = query.view(b, g, h // g, tq, dh)
    qz_all = idx.query_latent(qin, pass_idx=cur_pass)
    outs = []
    for c0 in range(0, tq, cfg.query_chunk):
        sl = slice(c0, min(tq, c0 + cfg.query_chunk))

        def run(qc: torch.Tensor, kc: torch.Tensor, vc: torch.Tensor, qzc: torch.Tensor,
                sl: slice = sl, p: int = cur_pass) -> torch.Tensor:
            return _sparse_chunk(qc, qzc, qpos_all[sl], kc, vc, index, idx, scaling, cur_pass=p)

        args = (qg[:, :, :, sl], key, value, qz_all[:, :, sl])
        if cfg.checkpoint_chunks and torch.is_grad_enabled() and (query.requires_grad or key.requires_grad):
            outs.append(checkpoint(run, *args, use_reentrant=False))
        else:
            outs.append(run(*args))
    out = torch.cat(outs, dim=3)                                                       # [B,G,Hg,Tq,D]
    return out.reshape(b, h, tq, dh).transpose(1, 2).contiguous(), None


_DENSE_FN: list[Callable[..., Any] | None] = [None]


# ============================================================================ install / manage

def apply_yarn(config: Any, target_len: int, original_len: int | None = None) -> dict[str, Any]:
    """Set YaRN RoPE scaling on a (text) config for `target_len` tokens. Call BEFORE model init."""
    orig = original_len or getattr(config, "max_position_embeddings", None) or 262144
    factor = max(1.0, float(target_len) / float(orig))
    rope = {"rope_type": "yarn", "factor": factor, "original_max_position_embeddings": int(orig)}
    params = getattr(config, "rope_parameters", None)
    if isinstance(params, dict):
        params = dict(params)
        params.update(rope)
        config.rope_parameters = params
    config.rope_scaling = dict(rope)
    config.max_position_embeddings = int(max(target_len, orig))
    log.info("YaRN: %d -> %d tokens (factor %.2f)", orig, target_len, factor)
    return rope


class LSAManager:
    def __init__(self, model: nn.Module, cfg: LSAConfig, indexers: dict[int, LSAIndexer],
                 prev_impl: str, set_impl: Callable[[str], None]):
        self.model, self.cfg, self.indexers = model, cfg, indexers
        self._prev_impl, self._set_impl = prev_impl, set_impl

    def set_mode(self, mode: str) -> None:
        if mode not in ("dense", "distill", "sparse"):
            raise ValueError(mode)
        self.cfg.mode = mode
        for ix in self.indexers.values():
            ix.cfg.mode = mode

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
            ix.last_selected_buckets.clear()

    def stats(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for lid, ix in self.indexers.items():
            for k, v in ix.last_stats.items():
                out[f"lsa/{k}/L{lid}"] = v
        return out

    def uninstall(self) -> None:
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
    if prev not in ALL_ATTENTION_FUNCTIONS:
        raise RuntimeError(f"cannot use attn_implementation={prev!r} as LSA dense fallback; load the model with "
                           f"attn_implementation='sdpa' (or one of {list(ALL_ATTENTION_FUNCTIONS.keys())})")
    _DENSE_FN[0] = ALL_ATTENTION_FUNCTIONS[prev]
    try:
        AttentionInterface.register(LSA_IMPL_NAME, lsa_attention)
    except Exception:
        ALL_ATTENTION_FUNCTIONS[LSA_IMPL_NAME] = lsa_attention
    try:
        from transformers import AttentionMaskInterface
        from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS

        if prev in ALL_MASK_ATTENTION_FUNCTIONS:
            try:
                AttentionMaskInterface.register(LSA_IMPL_NAME, ALL_MASK_ATTENTION_FUNCTIONS[prev])
            except Exception:
                ALL_MASK_ATTENTION_FUNCTIONS[LSA_IMPL_NAME] = ALL_MASK_ATTENTION_FUNCTIONS[prev]
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
        for cfg in {id(c): c for c in targets if c is not None}.values():
            cfg._attn_implementation = name
    return prev, set_impl


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
        rope = getattr(tc, "rope_parameters", None) or {}
        prf = getattr(tc, "partial_rotary_factor", None) or rope.get("partial_rotary_factor", 1.0)
        cfg.rotary_dim = int(hd * prf)
    full = parts.full_attention_indices()
    layers = full if cfg.layers is None else list(cfg.layers)
    bad = [i for i in layers if i not in full]
    if bad:
        raise ValueError(f"lsa.layers {bad} are not full-attention layers (full: {full})")
    prev, set_impl = _resolve_dense_and_register(model, tc)
    head_dim = getattr(tc, "head_dim", None) or tc.hidden_size // tc.num_attention_heads
    indexers: dict[int, LSAIndexer] = {}
    ref = parts.text_model.embed_tokens.weight
    for i in layers:
        attn = attention_module(parts.layers[i], FULL)
        ix = LSAIndexer(tc.num_attention_heads, tc.num_key_value_heads, head_dim, cfg, i)
        ix = ix.to(device=ref.device, dtype=torch.float64 if ref.dtype == torch.float64 else torch.float32)
        attn.lsa_indexer = ix
        object.__setattr__(ix, "_owners", [attn])
        object.__setattr__(ix, "_tm_ref", weakref.ref(parts.text_model))
        indexers[i] = ix
    set_impl(LSA_IMPL_NAME)
    log.info("D-LSA installed on layers %s (mode=%s, index_dim=%d on %s dims, block=%d, pool=%d, buckets~%d keys, "
             "select=%d, window=%d, sinks=%d, search=%s)", layers, cfg.mode, cfg.index_dim, cfg.index_on,
             cfg.block_size, cfg.pool, cfg.avg_bucket_keys, cfg.num_select, cfg.resolved_window(), cfg.sinks,
             cfg.search)
    return LSAManager(model, cfg, indexers, prev, set_impl)
