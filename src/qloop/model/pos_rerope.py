"""Dual Chunk Attention / ReRoPE smooth relative-distance clamping for partial-mRoPE heads (Fixes C1 & C2).

Qwen3.6-35B-A3B partitions each `256`-d full-attention head into:
  * `[0:64]`   (`rotary_dim = 64`): mRoPE channels trained up to `max_position_embeddings = 262,144`
  * `[64:256]` (`192` channels):    position-free NoPE content channels (also driving D-LSA clustering)

Beyond `linear_ceil = 131,072` tokens, `clamped_relative_distance` smoothly compresses relative
distances via a `C^1`-continuous logarithmic curve capped at `max_pos = 240,000`, keeping all
rotary angles strictly inside the pretrained `[0, 262K]` arc while leaving all `192` NoPE content
channels bit-for-bit untouched (`torch.equal`). Because `[0:64]` is isolated from `[64:256]`,
query-time ReRoPE clamping commutes exactly (`< 1e-12` in fp64) with Block-64 Walsh-Hadamard
(`H_64`) KV compression (`src/qloop/model/offload.py`, Fix C2).
"""
from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any

import torch

from ._dtype import upcast


@dataclass
class ReRoPEConfig:
    """Configuration for smooth DCA / ReRoPE relative-distance clamping on partial-rotary heads."""

    enabled: bool = True
    linear_ceil: int = 131_072
    log_scale: int = 32_768
    max_pos: int = 240_000
    rotary_dim: int = 64
    rope_theta: float = 10_000_000.0
    interleaved: bool = False

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> ReRoPEConfig:
        raw = dict(d or {})
        if "linear_Ceil" in raw and "linear_ceil" not in raw:
            raw["linear_ceil"] = raw.pop("linear_Ceil")
        else:
            raw.pop("linear_Ceil", None)
        known = {f.name for f in fields(cls)}
        bad = set(raw) - known
        if bad:
            raise ValueError(f"unknown rerope.* keys {sorted(bad)}; valid: {sorted(known)}")
        return cls(**raw)

    def validate(self) -> None:
        if self.linear_ceil < 0:
            raise ValueError(f"linear_ceil must be >= 0, got {self.linear_ceil}")
        if self.log_scale <= 0:
            raise ValueError(f"log_scale must be > 0, got {self.log_scale}")
        if self.max_pos < self.linear_ceil:
            raise ValueError(f"max_pos ({self.max_pos}) must be >= linear_ceil ({self.linear_ceil})")
        if self.rotary_dim <= 0 or self.rotary_dim % 2 != 0:
            raise ValueError(f"rotary_dim must be a positive even int, got {self.rotary_dim}")
        if self.rope_theta <= 0:
            raise ValueError(f"rope_theta must be > 0, got {self.rope_theta}")

    def build_inv_freq(
        self,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float64,
    ) -> torch.Tensor:
        self.validate()
        return build_rope_inv_freq(
            rotary_dim=self.rotary_dim,
            rope_theta=self.rope_theta,
            device=device,
            dtype=dtype,
        )


def build_rope_inv_freq(
    rotary_dim: int = 64,
    rope_theta: float = 10_000_000.0,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Return standard RoPE inverse frequencies `[rotary_dim // 2]`."""
    if rotary_dim <= 0 or rotary_dim % 2 != 0:
        raise ValueError(f"rotary_dim must be a positive even int, got {rotary_dim}")
    idx = torch.arange(0, rotary_dim, 2, device=device, dtype=torch.float64)
    inv = 1.0 / (float(rope_theta) ** (idx / float(rotary_dim)))
    return inv.to(dtype=dtype)


def clamped_relative_distance(
    delta: torch.Tensor,
    linear_Ceil: int | float = 131_072,
    log_scale: int | float = 32_768,
    max_pos: int | float = 240_000,
    *,
    linear_ceil: int | float | None = None,
) -> torch.Tensor:
    """Smoothly clamp relative token distance `delta = q_pos - k_pos` into `[-max_pos, max_pos]`.

    For `0 <= delta <= L_c` (`L_c = linear_ceil`):
        `delta_eff = delta`  (exact relative mRoPE inside `131,072` tokens)
    For `delta > L_c`:
        `delta_eff = min(max_pos, L_c + S_c * ln(1 + (delta - L_c) / S_c))`

    The transition at `delta = L_c` is `C^1`-smooth (matching value `L_c` and slope `1.0`).
    Signed negative distances (if passed) are mapped odd-symmetrically via `sign(delta) * f(|delta|)`.
    """
    lc = float(linear_Ceil if linear_ceil is None else linear_ceil)
    sc = float(log_scale)
    mp = float(max_pos)
    if lc < 0:
        raise ValueError(f"linear_ceil must be >= 0, got {lc}")
    if sc <= 0:
        raise ValueError(f"log_scale must be > 0, got {sc}")
    if mp < lc:
        raise ValueError(f"max_pos ({mp}) must be >= linear_ceil ({lc})")

    out_dtype = delta.dtype if delta.is_floating_point() else torch.float64
    d_work = delta.to(torch.float64)
    abs_d = d_work.abs()
    excess = (abs_d - lc).clamp_min(0.0)
    log_part = lc + sc * torch.log1p(excess / sc)
    eff = torch.where(abs_d <= lc, abs_d, log_part).clamp(max=mp)
    signed_eff = torch.where(d_work < 0, -eff, eff)
    return signed_eff.to(out_dtype)


def _rotate_2d_slice(
    x_rot: torch.Tensor,
    cos_half: torch.Tensor,
    sin_half: torch.Tensor,
    *,
    interleaved: bool = False,
) -> torch.Tensor:
    """Apply 2D planar rotations (`cos_half, sin_half` of shape `[..., rot_dim // 2]`) to `x_rot`."""
    work = upcast(x_rot)
    c = cos_half.to(work.dtype)
    s = sin_half.to(work.dtype)
    if interleaved:
        pairs = work.reshape(*work.shape[:-1], -1, 2)
        x0, x1 = pairs[..., 0], pairs[..., 1]
        y0 = x0 * c - x1 * s
        y1 = x0 * s + x1 * c
        return torch.stack([y0, y1], dim=-1).reshape(*torch.broadcast_shapes(work.shape[:-1], c.shape[:-1]), -1)
    x0, x1 = work.chunk(2, dim=-1)
    y0 = x0 * c - x1 * s
    y1 = x1 * c + x0 * s
    return torch.cat([y0, y1], dim=-1)


def _align_pos_for_tensor(pos: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Align position tensor `pos` for broadcasting against `x[..., :rotary_dim]` (`x.ndim >= 2`).

    Handles `[T]` or `[B, T]` against `[B, H, T, D]` or `[B, G, Hg, T, D]`, as well as 1-to-1
    leading shapes `[..., T]` against `[..., T, D]`, and pairwise `[..., T, K]` against `[..., T, 1, D]`.
    """
    target_lead = x.shape[:-1]
    if pos.shape == target_lead:
        return pos
    if x.ndim == 4 and pos.ndim == 2 and pos.shape == (x.shape[0], x.shape[2]):
        return pos[:, None, :]
    if x.ndim == 5 and pos.ndim == 2 and pos.shape == (x.shape[0], x.shape[3]):
        return pos[:, None, None, :]
    return pos


def apply_partial_rope_k(
    k: torch.Tensor,
    k_positions: torch.Tensor | int | float,
    inv_freq: torch.Tensor | None = None,
    rotary_dim: int = 64,
    *,
    rope_theta: float = 10_000_000.0,
    interleaved: bool = False,
) -> torch.Tensor:
    """Apply standard partial RoPE to `k[..., :rotary_dim]`, keeping `k[..., rotary_dim:]` bit-identical."""
    d = k.shape[-1]
    if rotary_dim < 0 or rotary_dim > d or rotary_dim % 2 != 0:
        raise ValueError(f"invalid rotary_dim={rotary_dim} for head_dim={d}")
    if rotary_dim == 0:
        return k
    if inv_freq is None:
        inv_freq = build_rope_inv_freq(rotary_dim, rope_theta=rope_theta, device=k.device, dtype=torch.float64)
    if not isinstance(k_positions, torch.Tensor):
        k_pos = torch.tensor(float(k_positions), device=k.device, dtype=torch.float64)
    else:
        k_pos = k_positions.to(device=k.device, dtype=torch.float64)
    k_pos = _align_pos_for_tensor(k_pos, k)
    freqs = k_pos.unsqueeze(-1) * inv_freq.to(device=k.device, dtype=torch.float64)
    cos_h, sin_h = torch.cos(freqs), torch.sin(freqs)

    k_rot = _rotate_2d_slice(k[..., :rotary_dim], cos_h, sin_h, interleaved=interleaved).to(k.dtype)
    k_nope = k[..., rotary_dim:]
    if k_nope.shape[:-1] != k_rot.shape[:-1]:
        k_nope = k_nope.expand(*k_rot.shape[:-1], k_nope.shape[-1])
    return torch.cat([k_rot, k_nope], dim=-1)


def apply_partial_rerope_q(
    q: torch.Tensor,
    k_positions: torch.Tensor | int | float,
    q_positions: torch.Tensor | int | float,
    inv_freq: torch.Tensor | None = None,
    rotary_dim: int = 64,
    *,
    linear_Ceil: int | float = 131_072,
    linear_ceil: int | float | None = None,
    log_scale: int | float = 32_768,
    max_pos: int | float = 240_000,
    rope_theta: float = 10_000_000.0,
    q_already_rotated: bool = False,
    interleaved: bool = False,
) -> torch.Tensor:
    """Apply smooth ReRoPE / DCA clamped rotary phase strictly to `q[..., :rotary_dim]`.

    Suppose cached keys were rotated at `k_positions`: `k_cached = [R(p_k * w) k_raw_rot, k_nope]`.
    Let `delta = q_positions - k_positions` and `delta_eff = clamped_relative_distance(delta, ...)`:
      * When `q_already_rotated=False` (default, `q` is pre-RoPE `q_raw`):
        rotates `q[..., :rotary_dim]` by `(k_positions + delta_eff) * inv_freq`, so that
        `<q_rerope, k_cached> == <R(delta_eff * w) q_raw_rot, k_raw_rot> + <q_nope, k_nope>`!
        Note: if `k_positions == 0` (unrotated keys), this applies `R(delta_eff * w)` directly.
      * When `q_already_rotated=True` (`q` already has standard `R(q_positions * w)` applied):
        rotates `q[..., :rotary_dim]` by the differential angle `(delta_eff - delta) * inv_freq`,
        yielding the exact same clamped relative inner product (`0` correction for `delta <= L_c`).

    In all cases, `q[..., rotary_dim:]` (`192` NoPE dims in Qwen3.6) is concatenated with zero
    arithmetic operations and remains bit-for-bit identical (`torch.equal`).
    """
    d = q.shape[-1]
    if rotary_dim < 0 or rotary_dim > d or rotary_dim % 2 != 0:
        raise ValueError(f"invalid rotary_dim={rotary_dim} for head_dim={d}")
    if rotary_dim == 0:
        return q
    if inv_freq is None:
        inv_freq = build_rope_inv_freq(rotary_dim, rope_theta=rope_theta, device=q.device, dtype=torch.float64)

    k_pos = (
        torch.tensor(float(k_positions), device=q.device, dtype=torch.float64)
        if not isinstance(k_positions, torch.Tensor)
        else k_positions.to(device=q.device, dtype=torch.float64)
    )
    q_pos = (
        torch.tensor(float(q_positions), device=q.device, dtype=torch.float64)
        if not isinstance(q_positions, torch.Tensor)
        else q_positions.to(device=q.device, dtype=torch.float64)
    )
    k_pos = _align_pos_for_tensor(k_pos, q)
    q_pos = _align_pos_for_tensor(q_pos, q)

    delta = q_pos - k_pos
    lc_val = linear_Ceil if linear_ceil is None else linear_ceil
    delta_eff = clamped_relative_distance(delta, linear_Ceil=lc_val, log_scale=log_scale, max_pos=max_pos)
    eff_pos = (delta_eff - delta) if q_already_rotated else (k_pos + delta_eff)

    freqs = eff_pos.unsqueeze(-1) * inv_freq.to(device=q.device, dtype=torch.float64)
    cos_h, sin_h = torch.cos(freqs), torch.sin(freqs)

    q_rot_out = _rotate_2d_slice(q[..., :rotary_dim], cos_h, sin_h, interleaved=interleaved).to(q.dtype)
    q_nope = q[..., rotary_dim:]
    if q_nope.shape[:-1] != q_rot_out.shape[:-1]:
        q_nope = q_nope.expand(*q_rot_out.shape[:-1], q_nope.shape[-1])
    return torch.cat([q_rot_out, q_nope], dim=-1)
