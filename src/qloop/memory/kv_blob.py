"""Phase-2A/2B `LSA-NL` Split-Horizon KV-Mem Blobs, Dynamic Latent Memory Graph, and 10B-Token Funnel.

Strictly isolated under `src/qloop/memory/` (zero modifications to active training codepaths) and
reusing the verified partial-RoPE (`pos_rerope.py`) and Block-64 Walsh-Hadamard INT4 (`offload.py`)
primitives directly:
  * **Pre-RoPE Sink-Stripped KV-Mem Blobs (`KVMemBlob`):** strips original `[0:sink_tokens)` (`EPIC`
    rule), applies exact inverse partial-RoPE `R(-(p_s + j) * omega)` strictly on `[0:rotary_dim)`
    via `apply_partial_rope_k` while keeping `[rotary_dim:head_dim)` (`192`-D NoPE) bitwise
    identical, packs K/V via `walsh_hadamard_block(..., block_size=64)` + `pack_group64_int4`
    (`2.8125 MiB` at `10L, 2H, 256D, 512T`), and maintains `1,344 B` GPU-resident coarse headers.
  * **Dynamic Latent Memory Graph (`DynamicLatentMemoryGraph` / `DLMG`):** registers `KVBlobHeader`
    nodes, wires reciprocal top-`m` semantic edges (`A-MEM`), temporal & supersession edges
    (`Zep/Graphiti`), and Hebbian co-access edges, skips `t_invalid` tombstoned blobs, and executes
    damped Personalized PageRank (`HippoRAG 2`) spreading activation.
  * **Split-Horizon Lifelong Store (`SplitHorizonKVBlobStore`):** implements the 4-stage `10B`-token
    lifelong funnel — Salience Cherry-Picking, `EpiCache` cluster compaction, Ebbinghaus ACT-R
    spaced-repetition hardening (`S_b <- 2 S_b`), bi-temporal `SleepGate` supersession, active
    forgetting (`R_b(t) = exp(-(t - t_last) / S_b)`), and Engram graduation selection.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, fields
from typing import Any

import torch
import torch.nn.functional as F

from qloop.model.offload import (
    GROUP_PACKED_BYTES,
    GROUP_SIZE_DEFAULT,
    pack_group64_int4,
    unpack_group64_int4,
    walsh_hadamard_block,
)
from qloop.model.pos_rerope import (
    apply_partial_rope_k,
    build_rope_inv_freq,
    clamped_relative_distance,
)

HEADER_METADATA_TAIL_BYTES = 64  # bi-temporal [t_valid, t_invalid), salience, DLMG CSR ptr, header IDs, LBA


def theoretical_header_wire_bytes(num_layers: int = 10, index_dim: int = 64) -> int:
    """Exact GPU-resident coarse header wire footprint (`L * d_idx * 2 + 64 = 1,344 B` at `10L, 64D`)."""
    if num_layers < 1 or index_dim < 1:
        raise ValueError(f"num_layers and index_dim must be >= 1, got ({num_layers}, {index_dim})")
    return int(num_layers) * int(index_dim) * 2 + HEADER_METADATA_TAIL_BYTES


def theoretical_h64_int4_bytes(
    num_layers: int = 10,
    num_kv_heads: int = 2,
    head_dim: int = 256,
    n_tokens: int = 512,
    *,
    block_size: int = GROUP_SIZE_DEFAULT,
) -> int:
    """Exact Block-64 Hadamard INT4 payload bytes for K + V (`2.8125 MiB` = `2,949,120 B` at 10L, 2H, 256D, 512T)."""
    if num_layers < 1 or num_kv_heads < 1 or head_dim < 1 or n_tokens < 0:
        raise ValueError(
            f"invalid KV blob geometry: L={num_layers}, H={num_kv_heads}, D={head_dim}, T={n_tokens}"
        )
    if block_size != GROUP_SIZE_DEFAULT or head_dim % block_size != 0:
        raise ValueError(f"head_dim={head_dim} must be divisible by block_size={GROUP_SIZE_DEFAULT} (got {block_size})")
    n_groups = int(head_dim) // int(block_size)
    return int(num_layers) * 2 * int(num_kv_heads) * int(n_tokens) * n_groups * GROUP_PACKED_BYTES


def _unit_normalize_rows(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Unit-normalize along last dimension in float32, preserving exact zeros without NaN."""
    xf = x.float()
    norms = xf.norm(dim=-1, keepdim=True)
    return torch.where(norms > eps, xf / norms.clamp(min=eps), torch.zeros_like(xf))


def _derive_coarse_centroid(
    k_pre_rope: torch.Tensor,
    rotary_dim: int,
    index_dim: int,
) -> torch.Tensor:
    """Derive deterministic position-free `[L, index_dim]` `float16` centroids strictly from NoPE `[rotary_dim:]`."""
    k_nope = k_pre_rope[..., rotary_dim:].float()  # [L, H_kv, T, D_nope]
    if k_nope.shape[-1] == 0:
        raise ValueError("cannot derive NoPE coarse centroid when rotary_dim == head_dim")
    layer_mean = k_nope.mean(dim=(1, 2))  # [L, D_nope]
    d_nope = layer_mean.shape[-1]
    if d_nope == index_dim:
        proj = layer_mean
    elif d_nope > index_dim and d_nope % index_dim == 0:
        proj = layer_mean.reshape(layer_mean.shape[0], index_dim, d_nope // index_dim).mean(dim=-1)
    elif d_nope > index_dim:
        proj = F.adaptive_avg_pool1d(layer_mean.unsqueeze(1), index_dim).squeeze(1)
    else:
        pad = layer_mean.new_zeros((layer_mean.shape[0], index_dim - d_nope))
        proj = torch.cat([layer_mean, pad], dim=-1)
    return _unit_normalize_rows(proj).to(torch.float16)


@dataclass
class KVBlobConfig:
    """Configuration for Split-Horizon `LSA-NL` Pre-RoPE `H64-INT4` KV-Mem Blobs and 10B-Token Funnel."""

    num_global_layers: int = 10
    num_kv_heads: int = 2
    head_dim: int = 256
    rotary_dim: int = 64
    blob_tokens: int = 512
    sink_tokens: int = 4
    boundary_header_tokens: int = 4
    index_dim: int = 64
    hadamard_block_size: int = 64
    rope_theta: float = 10_000_000.0
    salience_threshold: float = 0.25
    dedup_cos_threshold: float = 0.90
    min_retention: float = 0.15
    initial_stability: float = 1.0
    hardening_factor: float = 2.0
    max_stability: float = 1e6
    graduation_hits: int = 4
    graduation_min_stability: float = 8.0

    def __post_init__(self) -> None:
        self.validate()

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> KVBlobConfig:
        raw = dict(d or {})
        known = {f.name for f in fields(cls)}
        bad = set(raw) - known
        if bad:
            raise ValueError(f"unknown kv_blob.* keys {sorted(bad)}; valid: {sorted(known)}")
        return cls(**raw)

    def validate(self) -> None:
        if self.num_global_layers < 1 or self.num_kv_heads < 1 or self.head_dim < 1 or self.blob_tokens < 1:
            raise ValueError("num_global_layers, num_kv_heads, head_dim, and blob_tokens must be >= 1")
        if self.rotary_dim < 0 or self.rotary_dim > self.head_dim or self.rotary_dim % 2 != 0:
            raise ValueError(f"invalid rotary_dim={self.rotary_dim} for head_dim={self.head_dim}")
        if self.hadamard_block_size != GROUP_SIZE_DEFAULT or self.head_dim % self.hadamard_block_size != 0:
            raise ValueError(
                f"head_dim={self.head_dim} must be divisible by hadamard_block_size={GROUP_SIZE_DEFAULT} "
                f"(got {self.hadamard_block_size})"
            )
        if self.sink_tokens < 0 or self.boundary_header_tokens < 0 or self.index_dim < 1:
            raise ValueError("sink_tokens, boundary_header_tokens must be >= 0 and index_dim >= 1")
        if not math.isfinite(self.rope_theta) or self.rope_theta <= 0.0:
            raise ValueError(f"rope_theta must be finite and > 0, got {self.rope_theta}")
        if not math.isfinite(self.salience_threshold) or self.salience_threshold < 0.0:
            raise ValueError(f"salience_threshold must be finite and >= 0, got {self.salience_threshold}")
        if not math.isfinite(self.dedup_cos_threshold) or not (-1.0 <= self.dedup_cos_threshold <= 1.0):
            raise ValueError(f"dedup_cos_threshold must be finite and in [-1, 1], got {self.dedup_cos_threshold}")
        if not math.isfinite(self.min_retention) or not (0.0 < self.min_retention <= 1.0):
            raise ValueError(f"min_retention must be finite and in (0, 1], got {self.min_retention}")
        if not math.isfinite(self.initial_stability) or self.initial_stability <= 0.0:
            raise ValueError(f"initial_stability must be finite and > 0, got {self.initial_stability}")
        if not math.isfinite(self.hardening_factor) or self.hardening_factor < 1.0:
            raise ValueError(f"hardening_factor must be finite and >= 1.0, got {self.hardening_factor}")
        if not math.isfinite(self.max_stability) or self.max_stability < self.initial_stability:
            raise ValueError(
                f"max_stability must be finite and >= initial_stability ({self.initial_stability}), "
                f"got {self.max_stability}"
            )
        if self.graduation_hits < 1 or not math.isfinite(self.graduation_min_stability) or (
            self.graduation_min_stability <= 0.0
        ):
            raise ValueError("graduation_hits must be >= 1 and graduation_min_stability must be finite and > 0")

    def header_wire_bytes(self) -> int:
        return theoretical_header_wire_bytes(self.num_global_layers, self.index_dim)

    def theoretical_payload_bytes(self, n_tokens: int | None = None) -> int:
        tok = self.blob_tokens if n_tokens is None else int(n_tokens)
        return theoretical_h64_int4_bytes(
            self.num_global_layers,
            self.num_kv_heads,
            self.head_dim,
            tok,
            block_size=self.hadamard_block_size,
        )


@dataclass
class KVBlobHeader:
    """GPU-resident coarse semantic header + bi-temporal/ACT-R lifecycle metadata (`1,344 B` at 10L, 64D)."""

    blob_id: str
    coarse_centroid: torch.Tensor
    t_created: float = 0.0
    t_last_access: float = 0.0
    t_valid: float = 0.0
    t_invalid: float | None = None
    stability: float = 1.0
    hit_count: int = 0
    salience: float = 0.0
    n_tokens: int = 512
    topic_tags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.blob_id:
            raise ValueError("blob_id must be a non-empty string")
        if self.coarse_centroid.ndim != 2:
            raise ValueError(
                f"coarse_centroid must have shape [L, d_idx], got {tuple(self.coarse_centroid.shape)}"
            )
        if not math.isfinite(self.stability) or self.stability <= 0.0:
            raise ValueError(f"stability must be finite and > 0, got {self.stability}")
        if self.hit_count < 0 or self.n_tokens < 0:
            raise ValueError("hit_count and n_tokens must be >= 0")
        self.coarse_centroid = _unit_normalize_rows(self.coarse_centroid.detach()).to(torch.float16).contiguous()
        self.t_created = float(self.t_created)
        self.t_last_access = float(self.t_last_access)
        self.t_valid = float(self.t_valid)
        self.t_invalid = None if self.t_invalid is None else float(self.t_invalid)
        self.stability = float(self.stability)
        self.hit_count = int(self.hit_count)
        self.salience = float(self.salience)
        self.n_tokens = int(self.n_tokens)
        self.topic_tags = tuple(str(tag) for tag in self.topic_tags)

    def header_wire_bytes(self) -> int:
        """Exact wire byte count: `L * d_idx * 2 + 64` (`1,344 B` for `L=10, d_idx=64`)."""
        n_layers, d_idx = int(self.coarse_centroid.shape[0]), int(self.coarse_centroid.shape[1])
        return theoretical_header_wire_bytes(n_layers, d_idx)

    def is_active(self, now: float | None = None) -> bool:
        """Return True iff the blob is valid and NOT tombstoned at `now` (or currently when `now is None`)."""
        if now is None:
            return self.t_invalid is None
        t = float(now)
        if t < self.t_valid:
            return False
        return self.t_invalid is None or t < self.t_invalid

    def retention(self, now: float) -> float:
        """Ebbinghaus / ACT-R retention probability `R_b(t) = exp(-max(0, t - t_last_access) / S_b)`."""
        dt = max(0.0, float(now) - self.t_last_access)
        return math.exp(-dt / self.stability)

    def mean_centroid(self) -> torch.Tensor:
        """Unit-normalized mean centroid `[d_idx]` (`float32`) across layers for A-MEM / DLMG / EpiCache."""
        mean_vec = self.coarse_centroid.float().mean(dim=0)
        return _unit_normalize_rows(mean_vec)


@dataclass
class KVMemBlob:
    """Packed Pre-RoPE `Block64HadamardINT4` KV-Mem Blob paired with its GPU-resident `KVBlobHeader`."""

    header: KVBlobHeader
    packed_k: torch.Tensor
    packed_v: torch.Tensor
    rotary_dim: int = 64
    hadamard_block_size: int = GROUP_SIZE_DEFAULT
    rope_theta: float = 10_000_000.0

    def __post_init__(self) -> None:
        if self.packed_k.shape != self.packed_v.shape:
            raise ValueError(
                f"packed_k {tuple(self.packed_k.shape)} and packed_v {tuple(self.packed_v.shape)} must match"
            )
        if (
            self.packed_k.ndim != 5
            or self.packed_k.dtype != torch.uint8
            or self.packed_k.shape[-1] != GROUP_PACKED_BYTES
        ):
            raise ValueError(
                f"expected uint8 packed tensors of shape [L, H_kv, T, n_groups, {GROUP_PACKED_BYTES}], "
                f"got {self.packed_k.dtype} {tuple(self.packed_k.shape)}"
            )

    @property
    def blob_id(self) -> str:
        return self.header.blob_id

    @property
    def num_layers(self) -> int:
        return int(self.packed_k.shape[0])

    @property
    def num_kv_heads(self) -> int:
        return int(self.packed_k.shape[1])

    @property
    def n_tokens(self) -> int:
        return int(self.packed_k.shape[2])

    @property
    def head_dim(self) -> int:
        return int(self.packed_k.shape[3]) * int(self.hadamard_block_size)

    def payload_bytes(self) -> int:
        """Exact packed K + V payload bytes (`2,949,120 B = 2.8125 MiB` at `10L, 2H, 256D, 512T`)."""
        return int(
            self.packed_k.numel() * self.packed_k.element_size()
            + self.packed_v.numel() * self.packed_v.element_size()
        )

    def header_wire_bytes(self) -> int:
        return self.header.header_wire_bytes()


def strip_sinks_and_unrotate_k(
    k: torch.Tensor,
    positions: torch.Tensor,
    rotary_dim: int = 64,
    rope_theta: float = 10_000_000.0,
    *,
    sink_tokens: int = 4,
    inv_freq: torch.Tensor | None = None,
    interleaved: bool = False,
) -> torch.Tensor:
    """Strip original `[0:sink_tokens)` (`EPIC` rule) and apply exact inverse partial-RoPE `R(-(p_s + j) * omega)`.

    Reuses `apply_partial_rope_k` and `build_rope_inv_freq` from `src/qloop/model/pos_rerope.py`:
      * `[0:rotary_dim)` undergoes exact inverse rotation `R(-p * omega)` in float64 phase precision.
      * `[rotary_dim:head_dim)` (`192`-D NoPE) is never touched arithmetically and remains bitwise identical.
    """
    if k.ndim < 2:
        raise ValueError(f"expected k with ndim >= 2 ([..., T, D]), got shape {tuple(k.shape)}")
    if sink_tokens < 0:
        raise ValueError(f"sink_tokens must be >= 0, got {sink_tokens}")
    t_len = int(k.shape[-2])
    if sink_tokens > t_len:
        raise ValueError(f"sink_tokens={sink_tokens} exceeds sequence length T={t_len}")

    k_kept = k[..., sink_tokens:, :] if sink_tokens > 0 else k
    if not isinstance(positions, torch.Tensor):
        raise TypeError(f"positions must be a torch.Tensor, got {type(positions).__name__}")
    if positions.ndim < 1:
        raise ValueError(f"positions must have ndim >= 1 ([..., T]), got shape {tuple(positions.shape)}")
    if int(positions.shape[-1]) != t_len:
        raise ValueError(f"positions length ({positions.shape[-1]}) must match k token dim ({t_len})")
    pos_kept = positions[..., sink_tokens:] if sink_tokens > 0 else positions

    if rotary_dim == 0 or k_kept.shape[-2] == 0:
        return k_kept.clone()

    if inv_freq is None:
        inv_freq = build_rope_inv_freq(rotary_dim, rope_theta=rope_theta, device=k.device, dtype=torch.float64)
    neg_pos = -pos_kept.to(device=k.device, dtype=torch.float64)
    return apply_partial_rope_k(
        k_kept,
        neg_pos,
        inv_freq=inv_freq,
        rotary_dim=rotary_dim,
        rope_theta=rope_theta,
        interleaved=interleaved,
    )


def pack_kv_blob(
    k_pre_rope: torch.Tensor,
    v: torch.Tensor,
    *,
    blob_id: str,
    config: KVBlobConfig | None = None,
    coarse_centroid: torch.Tensor | None = None,
    t_created: float = 0.0,
    t_valid: float | None = None,
    t_last_access: float | None = None,
    salience: float = 1.0,
    stability: float | None = None,
    topic_tags: tuple[str, ...] | list[str] = (),
) -> KVMemBlob:
    """Pack Pre-RoPE `[L, H_kv, T, D]` keys and values into a `Block64HadamardINT4` `KVMemBlob`.

    Applies `walsh_hadamard_block(..., block_size=64)` and `pack_group64_int4` from
    `src/qloop/model/offload.py`, preserving exact orthogonal isolation between Group 0 (`[0:64)`
    mRoPE) and Groups `1..3` (`[64:256)` NoPE).
    """
    cfg = config if config is not None else KVBlobConfig()
    if k_pre_rope.shape != v.shape:
        raise ValueError(f"k_pre_rope {tuple(k_pre_rope.shape)} and v {tuple(v.shape)} shapes must match")
    if k_pre_rope.ndim != 4:
        raise ValueError(f"expected [L, H_kv, T, D] tensors, got shape {tuple(k_pre_rope.shape)}")
    n_layers, _n_heads, n_tok, head_dim = k_pre_rope.shape
    if head_dim % cfg.hadamard_block_size != 0:
        raise ValueError(f"head_dim={head_dim} must be divisible by hadamard_block_size={cfg.hadamard_block_size}")
    if cfg.rotary_dim > head_dim:
        raise ValueError(f"config.rotary_dim={cfg.rotary_dim} exceeds head_dim={head_dim}")

    k_h = walsh_hadamard_block(k_pre_rope, block_size=cfg.hadamard_block_size)
    v_h = walsh_hadamard_block(v, block_size=cfg.hadamard_block_size)
    packed_k = pack_group64_int4(k_h, group_size=cfg.hadamard_block_size).contiguous()
    packed_v = pack_group64_int4(v_h, group_size=cfg.hadamard_block_size).contiguous()

    if coarse_centroid is None:
        centroid = _derive_coarse_centroid(k_pre_rope, cfg.rotary_dim, cfg.index_dim)
    else:
        if int(coarse_centroid.shape[0]) != int(n_layers):
            raise ValueError(
                f"coarse_centroid layer dim ({coarse_centroid.shape[0]}) must match k_pre_rope ({n_layers})"
            )
        centroid = coarse_centroid

    created = float(t_created)
    header = KVBlobHeader(
        blob_id=blob_id,
        coarse_centroid=centroid,
        t_created=created,
        t_last_access=created if t_last_access is None else float(t_last_access),
        t_valid=created if t_valid is None else float(t_valid),
        t_invalid=None,
        stability=cfg.initial_stability if stability is None else float(stability),
        hit_count=0,
        salience=float(salience),
        n_tokens=int(n_tok),
        topic_tags=tuple(topic_tags),
    )
    return KVMemBlob(
        header=header,
        packed_k=packed_k,
        packed_v=packed_v,
        rotary_dim=cfg.rotary_dim,
        hadamard_block_size=cfg.hadamard_block_size,
        rope_theta=cfg.rope_theta,
    )


def mount_kv_blob(
    blob: KVMemBlob,
    virtual_positions: torch.Tensor | int,
    *,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | None = None,
    inv_freq: torch.Tensor | None = None,
    interleaved: bool = False,
    clamp_relative_to_q_pos: float | None = None,
    linear_ceil: int = 131_072,
    log_scale: int = 32_768,
    max_pos: int = 240_000,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Unpack `H64-INT4`, inverse Walsh-Hadamard transform, and stamp Partition-B virtual positions.

    Reuses `unpack_group64_int4`, `walsh_hadamard_block`, `apply_partial_rope_k`, and (when
    `clamp_relative_to_q_pos` is provided) `clamped_relative_distance`. Leaves `[rotary_dim:head_dim)`
    (`192`-D NoPE) strictly unmutated (`torch.equal` across arbitrary `virtual_positions`).
    """
    target_dev = blob.packed_k.device if device is None else torch.device(device)
    pk = blob.packed_k.to(device=target_dev)
    pv = blob.packed_v.to(device=target_dev)

    k_h_hat = unpack_group64_int4(pk, group_size=blob.hadamard_block_size, dtype=dtype)
    v_h_hat = unpack_group64_int4(pv, group_size=blob.hadamard_block_size, dtype=dtype)
    k_pre_hat = walsh_hadamard_block(k_h_hat, block_size=blob.hadamard_block_size)
    v_hat = walsh_hadamard_block(v_h_hat, block_size=blob.hadamard_block_size)

    if isinstance(virtual_positions, int):
        vpos = torch.arange(
            virtual_positions,
            virtual_positions + blob.n_tokens,
            device=target_dev,
            dtype=torch.float64,
        )
    elif isinstance(virtual_positions, torch.Tensor):
        vpos = virtual_positions.to(device=target_dev, dtype=torch.float64)
        if vpos.ndim == 0:
            base = float(vpos.item())
            vpos = base + torch.arange(blob.n_tokens, device=target_dev, dtype=torch.float64)
        elif int(vpos.shape[-1]) != blob.n_tokens:
            raise ValueError(f"virtual_positions length ({vpos.shape[-1]}) must match blob.n_tokens ({blob.n_tokens})")
    else:
        raise TypeError(f"virtual_positions must be int or torch.Tensor, got {type(virtual_positions).__name__}")

    if clamp_relative_to_q_pos is not None:
        q_pos = float(clamp_relative_to_q_pos)
        delta = q_pos - vpos
        delta_eff = clamped_relative_distance(delta, linear_ceil=linear_ceil, log_scale=log_scale, max_pos=max_pos)
        vpos = q_pos - delta_eff

    if blob.rotary_dim == 0:
        return k_pre_hat, v_hat

    if inv_freq is None:
        inv_freq = build_rope_inv_freq(
            blob.rotary_dim, rope_theta=blob.rope_theta, device=target_dev, dtype=torch.float64
        )
    k_mounted = apply_partial_rope_k(
        k_pre_hat,
        vpos,
        inv_freq=inv_freq,
        rotary_dim=blob.rotary_dim,
        rope_theta=blob.rope_theta,
        interleaved=interleaved,
    )
    return k_mounted, v_hat


class DynamicLatentMemoryGraph:
    """`DLMG`: `A-MEM` semantic links + `Zep/Graphiti` bi-temporal/supersession + Hebbian + `HippoRAG 2` PPR."""

    def __init__(
        self,
        semantic_top_m: int = 4,
        semantic_min_cos: float = 0.20,
        temporal_weight: float = 0.25,
        hebbian_weight_step: float = 0.50,
        hebbian_max_weight: float = 4.0,
    ) -> None:
        if semantic_top_m < 0:
            raise ValueError(f"semantic_top_m must be >= 0, got {semantic_top_m}")
        if temporal_weight < 0.0 or hebbian_weight_step < 0.0 or hebbian_max_weight < 0.0:
            raise ValueError("edge weights must be >= 0")
        self.semantic_top_m = int(semantic_top_m)
        self.semantic_min_cos = float(semantic_min_cos)
        self.temporal_weight = float(temporal_weight)
        self.hebbian_weight_step = float(hebbian_weight_step)
        self.hebbian_max_weight = float(hebbian_max_weight)

        self.nodes: dict[str, KVBlobHeader] = {}
        self._mean_centroids: dict[str, torch.Tensor] = {}
        self.semantic_edges: dict[str, dict[str, float]] = {}
        self.temporal_edges: dict[str, dict[str, float]] = {}
        self.hebbian_edges: dict[str, dict[str, float]] = {}
        self.supersession_edges: dict[str, str] = {}

    def __len__(self) -> int:
        return len(self.nodes)

    def _ensure_adj(self, blob_id: str) -> None:
        self.semantic_edges.setdefault(blob_id, {})
        self.temporal_edges.setdefault(blob_id, {})
        self.hebbian_edges.setdefault(blob_id, {})

    def register_node(
        self,
        header: KVBlobHeader,
        *,
        link_semantic: bool = True,
        temporal_parent: str | None = None,
        semantic_top_m: int | None = None,
    ) -> list[tuple[str, float]]:
        """Register a `KVBlobHeader`, auto-linking top-`m` reciprocal `A-MEM` edges & optional temporal parent."""
        bid = header.blob_id
        self.nodes[bid] = header
        mean_c = header.mean_centroid().cpu()
        self._mean_centroids[bid] = mean_c
        self._ensure_adj(bid)

        linked: list[tuple[str, float]] = []
        top_m = self.semantic_top_m if semantic_top_m is None else int(semantic_top_m)
        if link_semantic and top_m > 0 and len(self.nodes) > 1:
            candidates: list[tuple[float, str]] = []
            for other_id, other_hdr in self.nodes.items():
                if other_id == bid or not other_hdr.is_active():
                    continue
                other_c = self._mean_centroids[other_id]
                if other_c.shape != mean_c.shape:
                    continue
                cos_val = float(torch.dot(mean_c, other_c).item())
                if cos_val >= self.semantic_min_cos:
                    candidates.append((cos_val, other_id))
            candidates.sort(key=lambda item: item[0], reverse=True)
            for cos_val, other_id in candidates[:top_m]:
                self.add_semantic_edge(bid, other_id, cos_val)
                linked.append((other_id, cos_val))

        if temporal_parent is not None and temporal_parent in self.nodes and temporal_parent != bid:
            self.add_temporal_edge(temporal_parent, bid, self.temporal_weight)

        return linked

    def refresh_centroid(self, blob_id: str) -> None:
        """Recompute cached mean centroid after header mutation (e.g., `EpiCache` cluster merge)."""
        if blob_id not in self.nodes:
            raise KeyError(f"unknown blob_id {blob_id!r}")
        self._mean_centroids[blob_id] = self.nodes[blob_id].mean_centroid().cpu()

    def add_semantic_edge(self, src_id: str, dst_id: str, weight: float) -> None:
        """Add or update a reciprocal (`A-MEM`) semantic edge between two registered nodes."""
        if src_id not in self.nodes or dst_id not in self.nodes:
            raise KeyError(f"both nodes must be registered: {src_id!r}, {dst_id!r}")
        if src_id == dst_id or weight <= 0.0:
            return
        w = float(weight)
        self.semantic_edges[src_id][dst_id] = w
        self.semantic_edges[dst_id][src_id] = w

    def add_temporal_edge(self, src_id: str, dst_id: str, weight: float | None = None) -> None:
        """Add or update a temporal edge (`Zep/Graphiti`) between two registered nodes."""
        if src_id not in self.nodes or dst_id not in self.nodes:
            raise KeyError(f"both nodes must be registered: {src_id!r}, {dst_id!r}")
        if src_id == dst_id:
            return
        w = self.temporal_weight if weight is None else float(weight)
        if w <= 0.0:
            return
        self.temporal_edges[src_id][dst_id] = w
        self.temporal_edges[dst_id][src_id] = w

    def record_supersession(self, old_blob_id: str, new_blob_id: str, now: float) -> None:
        """Stamp `t_invalid = now` on `old_blob_id` and record bi-temporal supersession edge to `new_blob_id`."""
        if old_blob_id not in self.nodes or new_blob_id not in self.nodes:
            raise KeyError(f"both blobs must be registered: {old_blob_id!r}, {new_blob_id!r}")
        if old_blob_id == new_blob_id:
            raise ValueError("a blob cannot supersede itself")
        self.nodes[old_blob_id].t_invalid = float(now)
        self.supersession_edges[old_blob_id] = new_blob_id
        self.add_temporal_edge(old_blob_id, new_blob_id, self.temporal_weight)

    def record_co_access(
        self,
        blob_ids: list[str] | tuple[str, ...],
        weight_step: float | None = None,
    ) -> None:
        """Strengthen symmetric Hebbian co-access edges among co-retrieved active blobs."""
        step = self.hebbian_weight_step if weight_step is None else float(weight_step)
        if step <= 0.0:
            return
        uniq: list[str] = []
        seen: set[str] = set()
        for bid in blob_ids:
            if bid in self.nodes and bid not in seen:
                seen.add(bid)
                uniq.append(bid)
        for i in range(len(uniq)):
            u = uniq[i]
            for j in range(i + 1, len(uniq)):
                v = uniq[j]
                cur = self.hebbian_edges[u].get(v, 0.0)
                nxt = min(self.hebbian_max_weight, cur + step)
                self.hebbian_edges[u][v] = nxt
                self.hebbian_edges[v][u] = nxt

    def remove_node(self, blob_id: str) -> None:
        """Remove a node and all incident edges (including dangling supersession targets) from the graph."""
        if blob_id not in self.nodes:
            return
        self.nodes.pop(blob_id, None)
        self._mean_centroids.pop(blob_id, None)
        self.supersession_edges.pop(blob_id, None)
        dangling = [old_id for old_id, dst_id in self.supersession_edges.items() if dst_id == blob_id]
        for old_id in dangling:
            self.supersession_edges.pop(old_id, None)
        for adj in (self.semantic_edges, self.temporal_edges, self.hebbian_edges):
            nbrs = adj.pop(blob_id, {})
            for nb in nbrs:
                if nb in adj:
                    adj[nb].pop(blob_id, None)

    def merge_node_into(self, dup_id: str, keeper_id: str) -> None:
        """Transfer active incident edges and supersession targets from `dup_id` onto `keeper_id`."""
        if dup_id not in self.nodes or keeper_id not in self.nodes or dup_id == keeper_id:
            return
        for old_id, dst_id in list(self.supersession_edges.items()):
            if dst_id == dup_id:
                if old_id == keeper_id:
                    self.supersession_edges.pop(old_id, None)
                else:
                    self.supersession_edges[old_id] = keeper_id
        for adj in (self.semantic_edges, self.temporal_edges, self.hebbian_edges):
            for nb, w in list(adj.get(dup_id, {}).items()):
                if nb == keeper_id or nb not in self.nodes:
                    continue
                merged_w = max(adj[keeper_id].get(nb, 0.0), w)
                adj[keeper_id][nb] = merged_w
                adj[nb][keeper_id] = merged_w
        self.remove_node(dup_id)
        self.refresh_centroid(keeper_id)

    def _normalize_query(self, query_centroid: torch.Tensor) -> torch.Tensor:
        q = query_centroid.detach().float().cpu()
        if q.ndim == 2:
            q = _unit_normalize_rows(q).mean(dim=0)
        elif q.ndim != 1:
            raise ValueError(f"query_centroid must be 1D [d_idx] or 2D [L, d_idx], got {tuple(q.shape)}")
        return _unit_normalize_rows(q)

    def _active_ids(self, now: float | None = None) -> list[str]:
        return [bid for bid, hdr in self.nodes.items() if hdr.is_active(now)]

    def retrieve_cosine(
        self,
        query_centroid: torch.Tensor,
        top_k: int = 8,
        *,
        now: float | None = None,
        return_scores: bool = False,
    ) -> list[str] | list[tuple[str, float]]:
        """Flat cosine similarity retrieval over active (non-tombstoned) blob headers."""
        if top_k <= 0:
            return []
        active_ids = self._active_ids(now)
        if not active_ids:
            return []
        q = self._normalize_query(query_centroid)
        mat = torch.stack([self._mean_centroids[bid] for bid in active_ids], dim=0)  # [N, d_idx]
        sims = (mat @ q).tolist()
        ranked = sorted(zip(active_ids, sims), key=lambda item: item[1], reverse=True)[:top_k]
        if return_scores:
            return [(bid, float(sc)) for bid, sc in ranked]
        return [bid for bid, _ in ranked]

    def retrieve_ppr(
        self,
        query_centroid: torch.Tensor,
        top_k: int = 8,
        alpha: float = 0.85,
        n_iters: int = 15,
        *,
        now: float | None = None,
        seed_temperature: float = 0.10,
        return_scores: bool = False,
    ) -> list[str] | list[tuple[str, float]]:
        """Damped Personalized PageRank (`HippoRAG 2`) spreading activation over active nodes.

        Computes `pi^{t+1} = (1 - alpha) * p + alpha * (M^T pi^t + dangling_mass * p)` strictly
        over active (`is_active(now)`) nodes and edges, skipping all `t_invalid` tombstoned blobs.
        """
        if top_k <= 0:
            return []
        if not (0.0 <= alpha <= 1.0):
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")
        if n_iters < 0:
            raise ValueError(f"n_iters must be >= 0, got {n_iters}")
        if seed_temperature <= 0.0:
            raise ValueError(f"seed_temperature must be > 0, got {seed_temperature}")

        active_ids = self._active_ids(now)
        n = len(active_ids)
        if n == 0:
            return []

        q = self._normalize_query(query_centroid)
        mat = torch.stack([self._mean_centroids[bid] for bid in active_ids], dim=0).to(torch.float64)
        sims = mat @ q.to(torch.float64)  # [N]
        p = torch.softmax(sims / float(seed_temperature), dim=0)  # [N], sums to 1.0

        id_to_idx = {bid: idx for idx, bid in enumerate(active_ids)}
        adj_w = torch.zeros((n, n), dtype=torch.float64)
        for u_idx, u_id in enumerate(active_ids):
            for edge_map in (self.semantic_edges, self.temporal_edges, self.hebbian_edges):
                for v_id, w in edge_map.get(u_id, {}).items():
                    v_idx = id_to_idx.get(v_id)
                    if v_idx is not None and v_idx != u_idx and w > 0.0:
                        adj_w[u_idx, v_idx] += float(w)

        row_sums = adj_w.sum(dim=-1)  # [N]
        dangling = row_sums <= 0.0
        trans = torch.zeros_like(adj_w)
        non_dangling = ~dangling
        if bool(non_dangling.any()):
            trans[non_dangling] = adj_w[non_dangling] / row_sums[non_dangling].unsqueeze(-1)

        pi = p.clone()
        trans_t = trans.T
        for _ in range(n_iters):
            dangling_mass = pi[dangling].sum() if bool(dangling.any()) else 0.0
            pi = (1.0 - alpha) * p + alpha * (trans_t @ pi + dangling_mass * p)

        scores = pi.tolist()
        ranked = sorted(zip(active_ids, scores), key=lambda item: item[1], reverse=True)[:top_k]
        if return_scores:
            return [(bid, float(sc)) for bid, sc in ranked]
        return [bid for bid, _ in ranked]


class SplitHorizonKVBlobStore:
    """Split-Horizon `LSA-NL` Partition-B KV-Mem Blob Store implementing the 4-Stage `10B`-Token Funnel.

    Lifecycle stages:
      1. `admit(blob, salience)`: Salience Cherry-Picking admission gate (`admit` / `reject`).
      2. `compact_duplicates(cos_threshold)`: `EpiCache` cluster deduplication/merging in centroid space.
      3. `record_hit(blob_id, now)`: Ebbinghaus ACT-R spaced-repetition hardening (`S_b *= hardening_factor`).
      4. `supersede(old_blob_id, new_blob_id, now)`: Bi-temporal `[t_valid, t_invalid)` conflict resolution.
      5. `run_forgetting_cycle(now)`: Active forgetting (`R_b(t) = exp(-(t - t_last) / S_b) < min_retention`)
         and garbage collection of expired/tombstoned blobs.
      6. `select_for_graduation()`: Selects hardened episodes (`hit_count >= graduation_hits` and
         `stability >= graduation_min_stability`) for Stage-A/B/C Engram distillation.
      7. `memory_stats()`: Exact payload bytes, GPU header bytes, active/tombstoned/evicted counts.
    """

    def __init__(
        self,
        config: KVBlobConfig | None = None,
        graph: DynamicLatentMemoryGraph | None = None,
    ) -> None:
        self.config = config if config is not None else KVBlobConfig()
        self.config.validate()
        self.graph = graph if graph is not None else DynamicLatentMemoryGraph()
        self.blobs: dict[str, KVMemBlob] = {}
        self._rejected_count: int = 0
        self._merged_count: int = 0
        self._evicted_count: int = 0

    def __len__(self) -> int:
        return len(self.blobs)

    def admit(
        self,
        blob: KVMemBlob,
        salience: float | None = None,
        *,
        temporal_parent: str | None = None,
        link_semantic: bool = True,
    ) -> bool:
        """Stage 1 — Salience Cherry-Picking admission gate (`True` = admitted, `False` = rejected)."""
        score = blob.header.salience if salience is None else float(salience)
        blob.header.salience = float(score)
        if score < self.config.salience_threshold:
            self._rejected_count += 1
            return False
        if blob.blob_id in self.blobs:
            raise ValueError(f"blob_id {blob.blob_id!r} already exists in store")
        self.blobs[blob.blob_id] = blob
        self.graph.register_node(
            blob.header,
            link_semantic=link_semantic,
            temporal_parent=temporal_parent,
        )
        return True

    def compact_duplicates(
        self,
        cos_threshold: float | None = None,
        *,
        now: float | None = None,
    ) -> list[tuple[str, str]]:
        """Stage 2 — `EpiCache`-style cluster deduplication/merging over active blob centroids.

        Returns a list of `(removed_dup_id, canonical_keeper_id)` pairs merged in this pass.
        """
        thresh = self.config.dedup_cos_threshold if cos_threshold is None else float(cos_threshold)
        active = [b for b in self.blobs.values() if b.header.is_active(now)]
        # Keep highest-utility (hits, stability, salience, earliest creation) blob as canonical representative
        active.sort(
            key=lambda b: (b.header.hit_count, b.header.stability, b.header.salience, -b.header.t_created),
            reverse=True,
        )

        keepers: list[KVMemBlob] = []
        keeper_means: list[torch.Tensor] = []
        merged_pairs: list[tuple[str, str]] = []

        for candidate in active:
            cand_mean = candidate.header.mean_centroid().cpu()
            best_idx: int | None = None
            best_cos = -2.0
            for idx, k_mean in enumerate(keeper_means):
                if k_mean.shape != cand_mean.shape:
                    continue
                cos_val = float(torch.dot(cand_mean, k_mean).item())
                if cos_val >= thresh and cos_val > best_cos:
                    best_cos = cos_val
                    best_idx = idx

            if best_idx is None:
                keepers.append(candidate)
                keeper_means.append(cand_mean)
                continue

            keeper = keepers[best_idx]
            kh, dh = keeper.header, candidate.header
            wk = float(1 + kh.hit_count)
            wd = float(1 + dh.hit_count)
            merged_c = (
                wk * kh.coarse_centroid.float() + wd * dh.coarse_centroid.float()
            ) / (wk + wd)
            kh.coarse_centroid = _unit_normalize_rows(merged_c).to(torch.float16).contiguous()
            kh.hit_count += dh.hit_count
            kh.salience = max(kh.salience, dh.salience)
            kh.stability = max(kh.stability, dh.stability)
            kh.t_last_access = max(kh.t_last_access, dh.t_last_access)
            kh.t_valid = min(kh.t_valid, dh.t_valid)
            merged_tags = list(dict.fromkeys([*kh.topic_tags, *dh.topic_tags]))
            kh.topic_tags = tuple(merged_tags)

            keeper_means[best_idx] = kh.mean_centroid().cpu()
            self.graph.merge_node_into(candidate.blob_id, keeper.blob_id)
            self.blobs.pop(candidate.blob_id, None)
            self._merged_count += 1
            merged_pairs.append((candidate.blob_id, keeper.blob_id))

        return merged_pairs

    def record_hit(
        self,
        blob_id: str,
        now: float,
        *,
        co_accessed: list[str] | tuple[str, ...] | None = None,
    ) -> KVBlobHeader:
        """Stage 3a — Ebbinghaus ACT-R spaced-repetition hardening (`S_b *= hardening_factor`)."""
        if blob_id not in self.blobs:
            raise KeyError(f"unknown blob_id {blob_id!r}")
        hdr = self.blobs[blob_id].header
        t_now = float(now)
        if not hdr.is_active(t_now):
            raise ValueError(f"cannot record hit on inactive/tombstoned blob {blob_id!r} at t={t_now}")
        hdr.stability = min(self.config.max_stability, hdr.stability * self.config.hardening_factor)
        hdr.hit_count += 1
        hdr.t_last_access = t_now
        if co_accessed:
            self.graph.record_co_access([blob_id, *co_accessed])
        return hdr

    def supersede(self, old_blob_id: str, new_blob_id: str, now: float) -> None:
        """Stage 3b — Bi-temporal `[t_valid, t_invalid)` conflict resolution (`SleepGate` + `Zep/Graphiti`)."""
        if old_blob_id not in self.blobs or new_blob_id not in self.blobs:
            raise KeyError(f"both blobs must exist in store: {old_blob_id!r}, {new_blob_id!r}")
        t_now = float(now)
        self.graph.record_supersession(old_blob_id, new_blob_id, now=t_now)

    def run_forgetting_cycle(self, now: float, *, evict_tombstoned: bool = True) -> list[str]:
        """Stage 3c — Compute `R_b(t) = exp(-(t - t_last) / S_b)` and evict decayed/tombstoned blobs."""
        t_now = float(now)
        evicted: list[str] = []
        for bid in list(self.blobs):
            hdr = self.blobs[bid].header
            is_tombstoned = hdr.t_invalid is not None and t_now >= hdr.t_invalid
            decayed = hdr.retention(t_now) < self.config.min_retention
            if (evict_tombstoned and is_tombstoned) or decayed:
                self.blobs.pop(bid, None)
                self.graph.remove_node(bid)
                self._evicted_count += 1
                evicted.append(bid)
        return evicted

    def select_for_graduation(self, *, now: float | None = None) -> list[KVMemBlob]:
        """Stage 4 — Return hardened active blobs ready for Stage-A/B/C Engram distillation."""
        ready = [
            blob
            for blob in self.blobs.values()
            if blob.header.is_active(now)
            and blob.header.hit_count >= self.config.graduation_hits
            and blob.header.stability >= self.config.graduation_min_stability
        ]
        ready.sort(
            key=lambda b: (b.header.stability, b.header.hit_count, b.header.salience),
            reverse=True,
        )
        return ready

    def retrieve(
        self,
        query_centroid: torch.Tensor,
        top_k: int = 8,
        alpha: float = 0.85,
        n_iters: int = 15,
        *,
        now: float | None = None,
    ) -> list[KVMemBlob]:
        """Retrieve ranked active `KVMemBlob`s via `DLMG` Personalized PageRank."""
        ids = self.graph.retrieve_ppr(query_centroid, top_k=top_k, alpha=alpha, n_iters=n_iters, now=now)
        return [self.blobs[bid] for bid in ids if bid in self.blobs]

    def mount_blob(
        self,
        blob_id: str,
        virtual_positions: torch.Tensor | int,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Mount a stored `KVMemBlob` into Partition-B virtual positions."""
        if blob_id not in self.blobs:
            raise KeyError(f"unknown blob_id {blob_id!r}")
        return mount_kv_blob(self.blobs[blob_id], virtual_positions, **kwargs)

    def memory_stats(self, *, now: float | None = None) -> dict[str, int]:
        """Exact accounting of payload bytes, GPU header bytes, and active/tombstoned/evicted counts."""
        active_count = 0
        tombstoned_count = 0
        payload_bytes = 0
        active_payload_bytes = 0
        gpu_header_bytes = 0
        active_gpu_header_bytes = 0
        active_tokens = 0

        for blob in self.blobs.values():
            p_bytes = blob.payload_bytes()
            h_bytes = blob.header_wire_bytes()
            payload_bytes += p_bytes
            gpu_header_bytes += h_bytes
            if blob.header.is_active(now):
                active_count += 1
                active_payload_bytes += p_bytes
                active_gpu_header_bytes += h_bytes
                active_tokens += blob.n_tokens
            else:
                tombstoned_count += 1

        return {
            "total_blobs": len(self.blobs),
            "active_blobs": active_count,
            "tombstoned_blobs": tombstoned_count,
            "evicted_blobs": self._evicted_count,
            "rejected_blobs": self._rejected_count,
            "merged_blobs": self._merged_count,
            "payload_bytes": payload_bytes,
            "active_payload_bytes": active_payload_bytes,
            "gpu_header_bytes": gpu_header_bytes,
            "active_gpu_header_bytes": active_gpu_header_bytes,
            "active_tokens": active_tokens,
        }
