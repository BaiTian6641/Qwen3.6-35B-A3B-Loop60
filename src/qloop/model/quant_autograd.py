"""Blackwell FP8 (`float8_e4m3fn`) & NVFP4 (`compressed-tensors` E2M1) transient autograd bridge.

In `transformers>=5`, frozen quantized modules (`FineGrainedFP8` / `FP8Linear` / FP8
`Qwen3_5MoeExperts` and `compressed-tensors` `CompressedLinear` / NVFP4 experts) often dispatch to
inference-only kernels that do not propagate input gradients `dL/dx` (or `dL/d(top_k_weights)`)
during `loss.backward()`.

This module provides:
  * Exact, zero-persistent-VRAM transient dequantization helpers (`dequantize_fp8_tensor`,
    `dequantize_nvfp4_tensor`, `auto_dequant_expert_fn`) that never keep dequantized base weights
    resident in GPU memory.
  * `install_quantized_autograd_bridge(model)`: micro-probes each unique quantized module layout on
    2 synthetic tokens for both `(native_fwd_ok, native_grad_ok)` and wraps any module whose native
    forward or backward is unavailable or drops gradients:
      - When `not (torch.is_grad_enabled() and any_input_requires_grad)`: dispatches to the native
        Blackwell FP8/NVFP4 inference kernel if `native_fwd_ok` (dynamically catching runtime kernel
        `ImportError` / `ModuleNotFoundError` / `NotImplementedError` without crashing), and
        otherwise executes a zero-persistent-VRAM transient `bf16` fallback (`_fallback_linear_nograd`
        / `_fallback_moe_nograd`).
      - When `torch.is_grad_enabled()` and an input tensor requires grad: runs memory-zero custom
        `torch.autograd.Function` primitives that transiently dequantize in `forward` and
        re-dequantize in `backward` (and for fused-3D MoE experts, use a single-sync GPU `sort` +
        `bincount` + chunked-3D dequant dispatch touching ONLY active routed experts while computing
        both `dL/dh` and `dL/d(top_k_weights)`).
"""
from __future__ import annotations

import functools
import logging
import math
from collections import Counter
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

log = logging.getLogger("qloop")

FP8_DTYPES: tuple[torch.dtype, ...] = tuple(
    getattr(torch, name)
    for name in ("float8_e4m3fn", "float8_e4m3fnuz", "float8_e5m2", "float8_e5m2fnuz")
    if hasattr(torch, name)
)

# IEEE / NVIDIA FP4 E2M1 magnitude table [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0] + sign bit 0x08.
NVFP4_E2M1_TABLE: tuple[float, ...] = (
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    6.0,
    -0.0,
    -0.5,
    -1.0,
    -1.5,
    -2.0,
    -3.0,
    -4.0,
    -6.0,
)

_E2M1_BOUNDARIES: tuple[float, ...] = (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)
_LUT_CACHE: dict[torch.device, torch.Tensor] = {}
_BOUND_CACHE: dict[torch.device, torch.Tensor] = {}

_PACKED_INT_DTYPES: tuple[torch.dtype, ...] = (
    torch.uint8,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
)

_LINEAR_FP8_SCALE_ATTRS: tuple[str, ...] = (
    "weight_scale_inv",
    "scale_inv",
    "weight_scale",
    "scale",
    "weight_scales",
)
_MOE_GU_FP8_SCALE_ATTRS: tuple[str, ...] = (
    "gate_up_proj_scale_inv",
    "gate_up_proj_weight_scale_inv",
    "gate_up_proj_scale",
    "gate_up_proj_scales",
    "gate_up_scale_inv",
    "gate_up_scale",
    "weight_scale_inv_up",
    "weight_scale_up",
    "scale_inv_up",
    "scale_up",
    "w13_weight_scale_inv",
    "w13_weight_scale",
)
_MOE_DN_FP8_SCALE_ATTRS: tuple[str, ...] = (
    "down_proj_scale_inv",
    "down_proj_weight_scale_inv",
    "down_proj_scale",
    "down_proj_scales",
    "down_scale_inv",
    "down_scale",
    "weight_scale_inv_down",
    "weight_scale_down",
    "scale_inv_down",
    "scale_down",
    "w2_weight_scale_inv",
    "w2_weight_scale",
)
_MOE_CHUNK_EXPERTS: int = 32


def _get_e2m1_lut(device: torch.device) -> torch.Tensor:
    lut = _LUT_CACHE.get(device)
    if lut is None:
        with torch.inference_mode(False), torch.no_grad():
            lut = torch.tensor(NVFP4_E2M1_TABLE, dtype=torch.float32, device=device)
        _LUT_CACHE[device] = lut
    return lut


def _get_e2m1_boundaries(device: torch.device) -> torch.Tensor:
    bnd = _BOUND_CACHE.get(device)
    if bnd is None:
        with torch.inference_mode(False), torch.no_grad():
            bnd = torch.tensor(_E2M1_BOUNDARIES, dtype=torch.float32, device=device)
        _BOUND_CACHE[device] = bnd
    return bnd


def _infer_block_dim(dim_len: int, num_blocks: int, hint: int | None = None) -> int:
    if num_blocks <= 0:
        raise ValueError(f"scale block count must be > 0, got {num_blocks} for dim={dim_len}")
    if hint is not None and hint > 0 and math.ceil(dim_len / hint) == num_blocks:
        return int(hint)
    if num_blocks == 1:
        return dim_len
    if num_blocks == dim_len:
        return 1
    # Standard FineGrainedFP8 block size is 128 (handles non-divisible edge dimensions cleanly)
    if math.ceil(dim_len / 128) == num_blocks:
        return 128
    if dim_len % num_blocks == 0:
        return dim_len // num_blocks
    for cand in (256, 64, 32, 16, 8, 4, 2):
        if math.ceil(dim_len / cand) == num_blocks:
            return cand
    blk = math.ceil(dim_len / num_blocks)
    if math.ceil(dim_len / blk) != num_blocks:
        raise ValueError(f"cannot infer valid block size for dim={dim_len}, scale_blocks={num_blocks}")
    return blk


def _can_infer_block_dim(dim_len: int, num_blocks: int, hint: int | None = None) -> bool:
    try:
        blk = _infer_block_dim(dim_len, num_blocks, hint)
        return math.ceil(dim_len / blk) == num_blocks
    except ValueError:
        return False


def _extract_block_size(mod: nn.Module) -> tuple[int, int] | None:
    for attr in ("weight_block_size", "block_size"):
        val = getattr(mod, attr, None)
        if isinstance(val, (tuple, list)) and len(val) == 2:
            return int(val[0]), int(val[1])
    return None


def _first_tensor_attr(mod: nn.Module, names: tuple[str, ...]) -> tuple[str | None, torch.Tensor | None]:
    for name in names:
        val = getattr(mod, name, None)
        if isinstance(val, torch.Tensor):
            return name, val
    return None, None


def _iter_module_named_tensors(mod: nn.Module) -> list[tuple[str, torch.Tensor]]:
    out: list[tuple[str, torch.Tensor]] = []
    seen: set[str] = set()
    for name, param in mod._parameters.items():
        if isinstance(param, torch.Tensor):
            seen.add(name)
            out.append((name, param))
    for name, buf in mod._buffers.items():
        if isinstance(buf, torch.Tensor) and name not in seen:
            seen.add(name)
            out.append((name, buf))
    for name, val in vars(mod).items():
        if not name.startswith("_") and name not in seen and isinstance(val, torch.Tensor):
            seen.add(name)
            out.append((name, val))
    return out


def _module_storage_summary(mod: nn.Module) -> str:
    pnames = sorted(k for k, v in mod._parameters.items() if v is not None)
    bnames = sorted(k for k, v in mod._buffers.items() if v is not None)
    tnames = sorted(
        k
        for k, v in vars(mod).items()
        if not k.startswith("_") and isinstance(v, torch.Tensor)
    )
    return f"parameters={pnames}, buffers={bnames}, tensor_attrs={tnames}"


def _resolve_linear_fp8_scale(mod: nn.Module, weight: torch.Tensor) -> tuple[str | None, torch.Tensor | None]:
    name, s = _first_tensor_attr(mod, _LINEAR_FP8_SCALE_ATTRS)
    if s is not None:
        return name, s

    m, k = int(weight.shape[0]), int(weight.shape[1])
    bsize = _extract_block_size(mod)
    hint0 = bsize[0] if bsize is not None else None
    hint1 = bsize[1] if bsize is not None else None
    b0_def = hint0 if hint0 is not None else 128
    b1_def = hint1 if hint1 is not None else 128
    exp_s0, exp_s1 = math.ceil(m / b0_def), math.ceil(k / b1_def)
    bias_t = getattr(mod, "bias", None)

    best_score = 0
    best_pair: tuple[str | None, torch.Tensor | None] = (None, None)
    for cname, t in _iter_module_named_tensors(mod):
        if t is weight or t is bias_t:
            continue
        clow = cname.lower()
        if "bias" in clow or not t.is_floating_point() or t.dtype in FP8_DTYPES:
            continue
        has_scale = "scale" in clow
        score = 0
        if t.ndim == 2 and tuple(t.shape) == (exp_s0, exp_s1) and (has_scale or (exp_s0, exp_s1) != (m, k)):
            score = 4 if has_scale else 2
        elif has_scale and (
            t.ndim == 0
            or t.numel() == 1
            or (t.ndim == 1 and (int(t.shape[0]) in (m, k) or _can_infer_block_dim(m, int(t.shape[0]), hint0)))
            or (
                t.ndim == 2
                and (
                    (int(t.shape[0]) in (1, m) and int(t.shape[1]) in (1, k))
                    or (
                        _can_infer_block_dim(m, int(t.shape[0]), hint0)
                        and _can_infer_block_dim(k, int(t.shape[1]), hint1)
                    )
                )
            )
        ):
            score = 3
        if score > best_score:
            best_score = score
            best_pair = (cname, t)
    return best_pair


def _resolve_moe_fp8_scale(
    mod: nn.Module,
    weight_3d: torch.Tensor,
    role: str,
) -> tuple[str | None, torch.Tensor | None]:
    attrs = _MOE_GU_FP8_SCALE_ATTRS if role == "gu" else _MOE_DN_FP8_SCALE_ATTRS
    name, s = _first_tensor_attr(mod, attrs)
    if s is not None:
        return name, s

    e_cnt, m, k = int(weight_3d.shape[0]), int(weight_3d.shape[1]), int(weight_3d.shape[2])
    bsize = _extract_block_size(mod)
    hint0 = bsize[0] if bsize is not None else None
    hint1 = bsize[1] if bsize is not None else None
    b0_def = hint0 if hint0 is not None else 128
    b1_def = hint1 if hint1 is not None else 128
    exp_s0, exp_s1 = math.ceil(m / b0_def), math.ceil(k / b1_def)

    gu_t = getattr(mod, "gate_up_proj", None)
    dn_t = getattr(mod, "down_proj", None)
    pos_tokens = ("gate_up", "up", "w13", "gu") if role == "gu" else ("down", "dn", "w2")
    neg_tokens = ("down", "w2") if role == "gu" else ("gate_up", "up", "w13")

    best_score = 0
    best_pair: tuple[str | None, torch.Tensor | None] = (None, None)
    for cname, t in _iter_module_named_tensors(mod):
        if t is weight_3d or t is gu_t or t is dn_t:
            continue
        clow = cname.lower()
        if "bias" in clow or not t.is_floating_point() or t.dtype in FP8_DTYPES:
            continue
        if any(tok in clow for tok in neg_tokens):
            continue
        role_match = any(tok in clow for tok in pos_tokens)
        has_scale = "scale" in clow
        score = 0
        if (t.ndim == 3 and tuple(t.shape) == (e_cnt, exp_s0, exp_s1)) or (
            t.ndim == 2 and tuple(t.shape) == (e_cnt * exp_s0, exp_s1)
        ):
            score = 5 if role_match else (4 if has_scale else 2)
        elif has_scale or role_match:
            if t.ndim == 3 and int(t.shape[0]) in (1, e_cnt):
                s0, s1 = int(t.shape[1]), int(t.shape[2])
                if (s0 in (1, m) and s1 in (1, k)) or (
                    _can_infer_block_dim(m, s0, hint0) and _can_infer_block_dim(k, s1, hint1)
                ):
                    score = 3 if role_match else 1
            elif t.ndim == 2 and int(t.shape[0]) % e_cnt == 0:
                s0, s1 = int(t.shape[0]) // e_cnt, int(t.shape[1])
                if (s0 in (1, m) and s1 in (1, k)) or (
                    _can_infer_block_dim(m, s0, hint0) and _can_infer_block_dim(k, s1, hint1)
                ):
                    score = 3 if role_match else 1
            elif role_match and has_scale and (
                t.ndim == 0 or t.numel() == 1 or tuple(t.shape) in ((e_cnt,), (e_cnt, m), (e_cnt, k))
            ):
                score = 2
        if score > best_score:
            best_score = score
            best_pair = (cname, t)
    return best_pair


# ============================================================================ dequantization & quantization helpers


def dequantize_fp8_tensor(
    weight: torch.Tensor,
    scale_inv: torch.Tensor | None,
    dtype: torch.dtype = torch.bfloat16,
    *,
    block_size: tuple[int, int] | None = None,
) -> torch.Tensor:
    """Transiently dequantize a 2D `[out, in]` or 3D `[E, out, in]` FP8 tensor into `dtype`.

    Supports scalar, per-channel, and 2D block-scaled (`[ceil(out/B0), ceil(in/B1)]`, e.g.
    `[128, 128]` in `Qwen/Qwen3.6-35B-A3B-FP8`) `scale_inv` / `weight_scale` multipliers,
    slicing partial edge blocks cleanly when dimensions are not exact multiples of block size.
    """
    w = weight.to(torch.float32)
    if scale_inv is None:
        return w.to(dtype=dtype)

    s = scale_inv.to(device=weight.device, dtype=torch.float32)
    hint0 = block_size[0] if block_size is not None else None
    hint1 = block_size[1] if block_size is not None else None

    if w.ndim == 2:
        m, k = w.shape
        if s.ndim == 0 or s.numel() == 1:
            res = w * s.reshape(())
        elif s.ndim == 1:
            if s.shape[0] == m:
                res = w * s[:, None]
            elif s.shape[0] == k:
                res = w * s[None, :]
            else:
                b0 = _infer_block_dim(m, int(s.shape[0]), hint0)
                res = w * s.repeat_interleave(b0, dim=0)[:m, None]
        elif s.ndim == 2:
            s0, s1 = int(s.shape[0]), int(s.shape[1])
            if s0 in (1, m) and s1 in (1, k):
                res = w * s
            else:
                b0 = _infer_block_dim(m, s0, hint0)
                b1 = _infer_block_dim(k, s1, hint1)
                if math.ceil(m / b0) != s0 or math.ceil(k / b1) != s1:
                    raise ValueError(
                        f"FP8 block scale shape {tuple(s.shape)} incompatible with weight {tuple(w.shape)} "
                        f"and block_size=({b0}, {b1})"
                    )
                if m == s0 * b0 and k == s1 * b1:
                    res = (w.reshape(s0, b0, s1, b1) * s.reshape(s0, 1, s1, 1)).reshape(m, k)
                else:
                    s_exp = s if b0 == 1 else s.repeat_interleave(b0, dim=0)[:m, :]
                    s_exp = s_exp if b1 == 1 else s_exp.repeat_interleave(b1, dim=1)[:, :k]
                    res = w * s_exp
        else:
            raise ValueError(f"unsupported scale_inv shape {tuple(s.shape)} for 2D weight {tuple(w.shape)}")
        return res.to(dtype=dtype)

    if w.ndim == 3:
        e_cnt, m, k = w.shape
        if (
            s.ndim == 2
            and e_cnt > 1
            and s.shape not in ((e_cnt, m), (e_cnt, k))
            and int(s.shape[0]) % e_cnt == 0
        ):
            sub_s0 = int(s.shape[0]) // e_cnt
            is_flat_3d = (
                (hint0 is not None and math.ceil(m / hint0) == sub_s0)
                or (math.ceil(m / 128) == sub_s0 and math.ceil(m / 128) != int(s.shape[0]))
                or (not _can_infer_block_dim(m, int(s.shape[0]), hint0) and _can_infer_block_dim(m, sub_s0, hint0))
            )
            if is_flat_3d:
                s = s.reshape(e_cnt, sub_s0, int(s.shape[1]))

        if s.ndim == 0 or s.numel() == 1:
            res = w * s.reshape(())
        elif s.ndim == 1 and s.shape[0] == e_cnt:
            res = w * s.reshape(e_cnt, 1, 1)
        elif s.ndim == 2:
            if s.shape == (e_cnt, m):
                res = w * s[:, :, None]
            elif s.shape == (e_cnt, k):
                res = w * s[:, None, :]
            else:
                s0, s1 = int(s.shape[0]), int(s.shape[1])
                b0 = _infer_block_dim(m, s0, hint0)
                b1 = _infer_block_dim(k, s1, hint1)
                s_exp = s if b0 == 1 else s.repeat_interleave(b0, dim=0)[:m, :]
                s_exp = s_exp if b1 == 1 else s_exp.repeat_interleave(b1, dim=1)[:, :k]
                res = w * s_exp.unsqueeze(0)
        elif s.ndim == 3:
            es, s0, s1 = int(s.shape[0]), int(s.shape[1]), int(s.shape[2])
            if es not in (1, e_cnt):
                raise ValueError(f"3D FP8 scale expert dim {es} != weight expert dim {e_cnt}")
            if s0 in (1, m) and s1 in (1, k):
                res = w * s
            else:
                b0 = _infer_block_dim(m, s0, hint0)
                b1 = _infer_block_dim(k, s1, hint1)
                if math.ceil(m / b0) != s0 or math.ceil(k / b1) != s1:
                    raise ValueError(
                        f"3D FP8 block scale shape {tuple(s.shape)} incompatible with weight {tuple(w.shape)} "
                        f"and block_size=({b0}, {b1})"
                    )
                if m == s0 * b0 and k == s1 * b1:
                    res = (w.reshape(e_cnt, s0, b0, s1, b1) * s.reshape(es, s0, 1, s1, 1)).reshape(e_cnt, m, k)
                else:
                    s_exp = s if b0 == 1 else s.repeat_interleave(b0, dim=1)[:, :m, :]
                    s_exp = s_exp if b1 == 1 else s_exp.repeat_interleave(b1, dim=2)[:, :, :k]
                    res = w * s_exp
        else:
            raise ValueError(f"unsupported scale_inv shape {tuple(s.shape)} for 3D weight {tuple(w.shape)}")
        return res.to(dtype=dtype)

    raise ValueError(f"dequantize_fp8_tensor expects 2D or 3D weight, got shape {tuple(weight.shape)}")


def dequantize_nvfp4_tensor(
    weight_packed: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_global_scale: torch.Tensor | float | None = None,
    dtype: torch.dtype = torch.bfloat16,
    group_size: int = 16,
    *,
    global_scale_mode: str = "divisor",
) -> torch.Tensor:
    """Transiently unpack & dequantize an NVFP4 (`nvfp4-pack-quantized`) tensor into `dtype`.

    Nibble packing: each `uint8` byte stores two FP4 E2M1 codes (`lo = b & 0x0F` at index `2j`,
    `hi = (b >> 4) & 0x0F` at index `2j + 1`), mapped through `NVFP4_E2M1_TABLE`.

    Global scale convention (`global_scale_mode`):
      * `"divisor"` (default, `compressed-tensors` `weight_global_scale` convention):
        `weight_global_scale` stores the quantization scale `(448 * 6) / amax`, so
        `dequant = fp4_val * weight_scale.float() / weight_global_scale.float()`.
      * `"multiplier"` (ModelOpt `weight_scale_2` or pre-inverted runtime scale convention):
        `dequant = fp4_val * weight_scale.float() * weight_global_scale.float()`.
    """
    if group_size <= 0:
        raise ValueError(f"group_size must be > 0, got {group_size}")
    if global_scale_mode not in ("divisor", "multiplier"):
        raise ValueError(f"global_scale_mode must be 'divisor' or 'multiplier', got {global_scale_mode!r}")

    if weight_packed.dtype == torch.uint8:
        u8 = weight_packed
    elif weight_packed.dtype in _PACKED_INT_DTYPES:
        u8 = weight_packed.contiguous().view(torch.uint8)
    else:
        raise TypeError(f"dequantize_nvfp4_tensor expected packed integer tensor, got {weight_packed.dtype}")

    lo = u8 & 0x0F
    hi = (u8 >> 4) & 0x0F
    k_unpacked = int(u8.shape[-1]) * 2
    nibbles = torch.stack((lo, hi), dim=-1).reshape(*u8.shape[:-1], k_unpacked)
    lut = _get_e2m1_lut(u8.device)
    fp4_val = lut[nibbles.long()]

    s = weight_scale.to(device=u8.device, dtype=torch.float32)
    if s.ndim == 0 or s.numel() == 1:
        deq = fp4_val * s.reshape(())
    elif s.shape[:-1] == fp4_val.shape[:-1]:
        n_groups = int(s.shape[-1])
        if n_groups == 1:
            deq = fp4_val * s
        elif n_groups * group_size == k_unpacked:
            deq = (fp4_val.reshape(*s.shape, group_size) * s.unsqueeze(-1)).reshape(fp4_val.shape)
        elif math.ceil(k_unpacked / group_size) == n_groups:
            deq = fp4_val * s.repeat_interleave(group_size, dim=-1)[..., :k_unpacked]
        elif k_unpacked % n_groups == 0:
            eff_gs = k_unpacked // n_groups
            deq = (fp4_val.reshape(*s.shape, eff_gs) * s.unsqueeze(-1)).reshape(fp4_val.shape)
        else:
            raise ValueError(
                f"NVFP4 weight_scale shape {tuple(s.shape)} incompatible with unpacked shape "
                f"{tuple(fp4_val.shape)} (group_size={group_size})"
            )
    elif fp4_val.ndim == 3 and s.ndim == 2 and s.shape == (fp4_val.shape[0], 1):
        deq = fp4_val * s.unsqueeze(-1)
    else:
        raise ValueError(
            f"NVFP4 weight_scale shape {tuple(s.shape)} incompatible with unpacked shape {tuple(fp4_val.shape)}"
        )

    if weight_global_scale is not None:
        if isinstance(weight_global_scale, torch.Tensor):
            gs: torch.Tensor | float = weight_global_scale.to(device=u8.device, dtype=torch.float32)
            if isinstance(gs, torch.Tensor):
                if gs.ndim == 1:
                    if deq.ndim == 3 and gs.shape[0] == deq.shape[0]:
                        gs = gs.reshape(deq.shape[0], 1, 1)
                    elif deq.ndim == 2 and gs.shape[0] == deq.shape[0]:
                        gs = gs.reshape(deq.shape[0], 1)
                elif (
                    deq.ndim == 3
                    and gs.ndim == 2
                    and gs.shape[0] == deq.shape[0]
                    and gs.shape[1] in (1, deq.shape[1])
                ):
                    gs = gs.unsqueeze(-1)
        else:
            gs = float(weight_global_scale)
        deq = deq / gs if global_scale_mode == "divisor" else deq * gs

    return deq.to(dtype=dtype)


def quantize_fp8_blockwise(
    weight: torch.Tensor,
    block_size: tuple[int, int] = (128, 128),
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a 2D `[M, K]` or 3D `[E, M, K]` float tensor to `float8_e4m3fn` + `scale_inv`."""
    if not hasattr(torch, "float8_e4m3fn"):
        raise RuntimeError("PyTorch build lacks torch.float8_e4m3fn")
    fp8_dtype = torch.float8_e4m3fn
    fp8_max = 448.0
    b0, b1 = int(block_size[0]), int(block_size[1])
    wf = weight.detach().to(torch.float32)

    is_2d = wf.ndim == 2
    if is_2d:
        wf = wf.unsqueeze(0)
    elif wf.ndim != 3:
        raise ValueError(f"quantize_fp8_blockwise expects 2D or 3D tensor, got {tuple(weight.shape)}")

    e_cnt, m, k = wf.shape
    s0, s1 = math.ceil(m / b0), math.ceil(k / b1)
    pad_m, pad_k = s0 * b0 - m, s1 * b1 - k
    w_pad = F.pad(wf, (0, pad_k, 0, pad_m)) if (pad_m or pad_k) else wf
    blocks = w_pad.reshape(e_cnt, s0, b0, s1, b1)
    amax = blocks.abs().amax(dim=(2, 4)).clamp_min(1e-12)
    scale_inv = amax / fp8_max
    q_blocks = (blocks / scale_inv.reshape(e_cnt, s0, 1, s1, 1)).clamp(-fp8_max, fp8_max).to(fp8_dtype)
    q_full = q_blocks.reshape(e_cnt, s0 * b0, s1 * b1)[:, :m, :k].contiguous()

    if is_2d:
        return q_full.squeeze(0), scale_inv.squeeze(0).contiguous()
    return q_full, scale_inv.contiguous()


def quantize_nvfp4_tensor(
    weight: torch.Tensor,
    group_size: int = 16,
    *,
    global_scale_mode: str = "divisor",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize a 2D `[M, K]` or 3D `[E, M, K]` float tensor to NVFP4 `(packed, scale, global_scale)`."""
    if not hasattr(torch, "float8_e4m3fn"):
        raise RuntimeError("PyTorch build lacks torch.float8_e4m3fn")
    if weight.shape[-1] % 2 != 0:
        raise ValueError(f"NVFP4 inner dimension must be even, got {weight.shape[-1]}")
    if global_scale_mode not in ("divisor", "multiplier"):
        raise ValueError(f"global_scale_mode must be 'divisor' or 'multiplier', got {global_scale_mode!r}")

    wf = weight.detach().to(torch.float32)
    k = int(wf.shape[-1])
    n_groups = math.ceil(k / group_size)
    pad_k = n_groups * group_size - k
    w_pad = F.pad(wf, (0, pad_k)) if pad_k else wf
    grouped = w_pad.reshape(*wf.shape[:-1], n_groups, group_size)

    amax_global = wf.abs().amax().clamp_min(1e-12)
    divisor_gs = (448.0 * 6.0) / amax_global
    grp_amax = grouped.abs().amax(dim=-1)
    w_scale = (grp_amax * divisor_gs / 6.0).clamp(0.0, 448.0).to(torch.float8_e4m3fn)

    eff_block_scale = (w_scale.to(torch.float32) / divisor_gs).unsqueeze(-1)
    normed = torch.where(eff_block_scale > 0, grouped / eff_block_scale, torch.zeros_like(grouped))

    sign = (normed < 0).to(torch.uint8) << 3
    bnd = _get_e2m1_boundaries(wf.device)
    mag_code = torch.bucketize(normed.abs(), bnd).to(torch.uint8)
    nibbles = (sign | mag_code).reshape(*wf.shape[:-1], n_groups * group_size)[..., :k]
    packed = (nibbles[..., 0::2] | (nibbles[..., 1::2] << 4)).contiguous()

    global_scale_out = divisor_gs if global_scale_mode == "divisor" else (1.0 / divisor_gs)
    return packed, w_scale.contiguous(), global_scale_out.to(torch.float32)


# ============================================================================ module classification & introspection


def _classify_quant_module(mod: nn.Module) -> str | None:
    """Return `'linear_fp8' | 'linear_nvfp4' | 'moe_fp8' | 'moe_nvfp4' | None` for `mod` itself."""
    if isinstance(mod, nn.Embedding):
        return None

    # 1) Fused 3D MoE experts: NVFP4 packed
    gu_p = getattr(mod, "gate_up_proj_packed", None)
    dn_p = getattr(mod, "down_proj_packed", None)
    gu_s = getattr(mod, "gate_up_proj_scale", None)
    dn_s = getattr(mod, "down_proj_scale", None)
    if (
        isinstance(gu_p, torch.Tensor)
        and isinstance(dn_p, torch.Tensor)
        and gu_p.ndim == 3
        and dn_p.ndim == 3
        and gu_p.dtype in _PACKED_INT_DTYPES
        and isinstance(gu_s, torch.Tensor)
        and isinstance(dn_s, torch.Tensor)
    ):
        return "moe_nvfp4"

    # 2) Fused 3D MoE experts: FP8 or uint8-in-gate_up_proj
    gu = getattr(mod, "gate_up_proj", None)
    dn = getattr(mod, "down_proj", None)
    if isinstance(gu, torch.Tensor) and isinstance(dn, torch.Tensor) and gu.ndim == 3 and dn.ndim == 3:
        if gu.dtype in FP8_DTYPES and dn.dtype in FP8_DTYPES:
            return "moe_fp8"
        if gu.dtype in _PACKED_INT_DTYPES and isinstance(gu_s, torch.Tensor) and isinstance(dn_s, torch.Tensor):
            return "moe_nvfp4"

    # 3) 2D Linear: NVFP4 packed (`weight_packed` + `weight_scale`, or uint8 `weight` + `weight_scale`)
    wp = getattr(mod, "weight_packed", None)
    ws = getattr(mod, "weight_scale", None)
    if (
        isinstance(wp, torch.Tensor)
        and wp.ndim == 2
        and wp.dtype in _PACKED_INT_DTYPES
        and isinstance(ws, torch.Tensor)
    ):
        return "linear_nvfp4"

    w = getattr(mod, "weight", None)
    if isinstance(w, torch.Tensor) and w.ndim == 2:
        if w.dtype in FP8_DTYPES:
            return "linear_fp8"
        if w.dtype in _PACKED_INT_DTYPES and isinstance(ws, torch.Tensor):
            return "linear_nvfp4"

    return None


def is_quantized_module(mod: nn.Module) -> bool:
    """Return `True` if `mod` directly holds quantized (`FP8` or `NVFP4`) linear or MoE expert weights."""
    if _classify_quant_module(mod) is not None:
        return True
    if isinstance(mod, nn.ModuleList) and len(mod) > 0:
        return any(is_quantized_module(sub) for sub in mod.modules() if sub is not mod)
    return False


def _extract_raw_quant_config(model: nn.Module) -> dict[str, Any] | None:
    for cfg_owner in (model, getattr(model, "model", None)):
        cfg = getattr(cfg_owner, "config", None)
        if cfg is None:
            continue
        qcfg = getattr(cfg, "quantization_config", None)
        if qcfg is None:
            tcfg = getattr(cfg, "text_config", None)
            qcfg = getattr(tcfg, "quantization_config", None) if tcfg is not None else None
        if qcfg is not None:
            if isinstance(qcfg, dict):
                return dict(qcfg)
            if hasattr(qcfg, "to_dict"):
                return dict(qcfg.to_dict())
            return {k: v for k, v in vars(qcfg).items() if not k.startswith("_")}
    return None


def detect_model_quantization(model: nn.Module) -> dict[str, Any]:
    """Inspect `model` for FP8 / NVFP4 quantized linears, fused MoE experts, and `quantization_config`."""
    seen: set[int] = set()
    counts: Counter[str] = Counter()
    mod_types: Counter[str] = Counter()
    for mod in model.modules():
        if id(mod) in seen:
            continue
        seen.add(id(mod))
        kind = _classify_quant_module(mod)
        if kind is not None:
            counts[kind] += 1
            mod_types[type(mod).__name__] += 1

    qcfg = _extract_raw_quant_config(model)
    qmethod: str | None = None
    qfmt: str | None = None
    if qcfg:
        qmethod = str(qcfg.get("quant_method") or "") or None
        qfmt = str(qcfg.get("format") or qcfg.get("fmt") or "") or None

    has_nvfp4 = (counts["linear_nvfp4"] + counts["moe_nvfp4"]) > 0
    has_fp8 = (counts["linear_fp8"] + counts["moe_fp8"]) > 0
    if qmethod is None:
        if has_nvfp4:
            qmethod = "compressed-tensors"
        elif has_fp8:
            qmethod = "fp8"
    if qfmt is None:
        if has_nvfp4:
            qfmt = "nvfp4-pack-quantized"
        elif has_fp8:
            qfmt = "e4m3"

    is_quant = bool(has_nvfp4 or has_fp8 or qcfg)
    return {
        "is_quantized": is_quant,
        "quant_method": qmethod,
        "format": qfmt,
        "linear_fp8_count": int(counts["linear_fp8"]),
        "linear_nvfp4_count": int(counts["linear_nvfp4"]),
        "moe_fp8_count": int(counts["moe_fp8"]),
        "moe_nvfp4_count": int(counts["moe_nvfp4"]),
        "quantized_module_types": dict(mod_types),
        "has_quantization_config": qcfg is not None,
    }


# ============================================================================ module-level dequantization


def _nvfp4_scale_mode_for_module(mod: nn.Module, gs_attr_name: str | None) -> str:
    calibrated = getattr(mod, "_qloop_nvfp4_scale_mode", None)
    if calibrated in ("divisor", "multiplier"):
        return str(calibrated)
    if gs_attr_name is not None and gs_attr_name.endswith("_scale_2"):
        return "multiplier"
    return "divisor"


def _dequant_linear_weight(
    mod: nn.Module,
    dtype: torch.dtype = torch.bfloat16,
    *,
    nvfp4_mode_override: str | None = None,
) -> torch.Tensor:
    kind = _classify_quant_module(mod)
    if kind == "linear_fp8":
        w = mod.weight
        _, s = _resolve_linear_fp8_scale(mod, w)
        if s is None and not getattr(mod, "_qloop_allow_unscaled_fp8", False):
            raise RuntimeError(
                f"FP8 linear module {type(mod).__name__} has no scale tensor "
                f"({_module_storage_summary(mod)}). Set `_qloop_allow_unscaled_fp8 = True` "
                "only if unscaled float8 is intentional."
            )
        return dequantize_fp8_tensor(w, s, dtype=dtype, block_size=_extract_block_size(mod))

    if kind == "linear_nvfp4":
        wp = getattr(mod, "weight_packed", None)
        if not isinstance(wp, torch.Tensor):
            wp = mod.weight
        ws = mod.weight_scale
        gs_name, gs = _first_tensor_attr(mod, ("weight_global_scale", "global_scale", "weight_scale_2"))
        gsize = int(getattr(mod, "group_size", None) or 16)
        mode = nvfp4_mode_override or _nvfp4_scale_mode_for_module(mod, gs_name)
        return dequantize_nvfp4_tensor(wp, ws, gs, dtype=dtype, group_size=gsize, global_scale_mode=mode)

    w_plain = getattr(mod, "weight", None)
    if isinstance(w_plain, torch.Tensor) and w_plain.ndim == 2:
        return w_plain.to(dtype=dtype)
    raise TypeError(f"cannot dequantize 2D linear weight from module {type(mod).__name__}")


def _linear_in_out_features(mod: nn.Module) -> tuple[int, int]:
    in_f = getattr(mod, "in_features", None)
    out_f = getattr(mod, "out_features", None)
    if isinstance(in_f, int) and isinstance(out_f, int) and in_f > 0 and out_f > 0:
        return in_f, out_f
    kind = _classify_quant_module(mod)
    if kind == "linear_fp8":
        w = mod.weight
        return int(w.shape[1]), int(w.shape[0])
    if kind == "linear_nvfp4":
        wp = getattr(mod, "weight_packed", None)
        if not isinstance(wp, torch.Tensor):
            wp = mod.weight
        nbytes_last = int(wp.shape[1]) * int(wp.element_size())
        return nbytes_last * 2, int(wp.shape[0])
    w = getattr(mod, "weight", None)
    if isinstance(w, torch.Tensor) and w.ndim == 2:
        return int(w.shape[1]), int(w.shape[0])
    raise RuntimeError(f"cannot determine (in_features, out_features) for {type(mod).__name__}")


def _is_expert_pair_transposed(
    experts_mod: nn.Module,
    gu_shape: tuple[int, int],
    dn_shape: tuple[int, int],
) -> bool:
    """Detect whether single-expert `(gate_up, down)` are transposed `[H, 2I]` / `[I, H]`."""
    cfg = getattr(experts_mod, "config", None)
    h_hint = getattr(experts_mod, "hidden_size", None) or getattr(experts_mod, "hidden_dim", None)
    if h_hint is None and cfg is not None:
        h_hint = getattr(cfg, "hidden_size", None)
    i_hint = (
        getattr(experts_mod, "moe_intermediate_size", None)
        or getattr(experts_mod, "intermediate_size", None)
        or getattr(experts_mod, "intermediate_dim", None)
    )
    if i_hint is None and cfg is not None:
        i_hint = getattr(cfg, "moe_intermediate_size", None) or getattr(cfg, "intermediate_size", None)

    # Exact structural check: `[2I, H]` + `[H, I]` and `[H, 2I]` + `[I, H]` are mutually exclusive for all H,I >= 1.
    std_match = gu_shape[1] == dn_shape[0] and gu_shape[0] == 2 * dn_shape[1]
    tr_match = gu_shape[0] == dn_shape[1] and gu_shape[1] == 2 * dn_shape[0]
    if std_match and not tr_match:
        return False
    if tr_match and not std_match:
        return True

    if isinstance(h_hint, int) and isinstance(i_hint, int):
        if gu_shape == (2 * i_hint, h_hint) and dn_shape == (h_hint, i_hint):
            return False
        if gu_shape == (h_hint, 2 * i_hint) and dn_shape == (i_hint, h_hint):
            return True
    if isinstance(h_hint, int):
        if gu_shape[1] == h_hint and dn_shape[0] == h_hint:
            return False
        if gu_shape[0] == h_hint and dn_shape[1] == h_hint:
            return True
    raise ValueError(
        f"cannot orient expert weight shapes gate_up={gu_shape}, down={dn_shape} on {type(experts_mod).__name__}"
    )


def _slice_expert_scale(
    scale: torch.Tensor | None,
    e: int,
    num_experts: int,
    *,
    weight_2d_shape: tuple[int, int] | None = None,
    block_size: tuple[int, int] | None = None,
) -> torch.Tensor | None:
    if scale is None:
        return None
    if scale.ndim == 0 or scale.numel() == 1:
        return scale
    if scale.ndim == 2 and num_experts > 0:
        rows, cols = int(scale.shape[0]), int(scale.shape[1])
        if rows != num_experts and rows % num_experts == 0:
            s0 = rows // num_experts
            return scale[e * s0 : (e + 1) * s0]
        if rows == num_experts and weight_2d_shape is not None:
            m, k = int(weight_2d_shape[0]), int(weight_2d_shape[1])
            if block_size is not None:
                b0, b1 = int(block_size[0]), int(block_size[1])
                if math.ceil(m / b0) == 1 and math.ceil(k / b1) == cols:
                    return scale[e : e + 1]
            if cols != m:
                return scale[e : e + 1]
    if scale.shape[0] == num_experts and scale.ndim in (1, 2, 3):
        return scale[e]
    return scale


def _slice_expert_scale_batch(
    scale: torch.Tensor | None,
    expert_ids: torch.Tensor,
    num_experts: int,
    *,
    weight_2d_shape: tuple[int, int] | None = None,
    block_size: tuple[int, int] | None = None,
) -> torch.Tensor | None:
    if scale is None:
        return None
    if scale.ndim == 0 or scale.numel() == 1:
        return scale
    eids = expert_ids.to(device=scale.device, dtype=torch.long)
    if scale.ndim == 2 and num_experts > 0:
        rows, cols = int(scale.shape[0]), int(scale.shape[1])
        if rows != num_experts and rows % num_experts == 0:
            s0 = rows // num_experts
            return scale.reshape(num_experts, s0, cols).index_select(0, eids)
        if rows == num_experts and weight_2d_shape is not None:
            m, k = int(weight_2d_shape[0]), int(weight_2d_shape[1])
            if block_size is not None:
                b0, b1 = int(block_size[0]), int(block_size[1])
                if math.ceil(m / b0) == 1 and math.ceil(k / b1) == cols:
                    return scale.reshape(num_experts, 1, cols).index_select(0, eids)
            if cols != m:
                return scale.reshape(num_experts, 1, cols).index_select(0, eids)
        if rows == num_experts:
            return scale.index_select(0, eids)
        return scale.unsqueeze(0)
    if scale.shape[0] == num_experts and scale.ndim in (1, 3):
        return scale.index_select(0, eids)
    return scale


def _require_moe_fp8_scales(
    experts_mod: nn.Module,
    gu_s: torch.Tensor | None,
    dn_s: torch.Tensor | None,
) -> None:
    if (gu_s is None or dn_s is None) and not getattr(experts_mod, "_qloop_allow_unscaled_fp8", False):
        missing = []
        if gu_s is None:
            missing.append("gate_up_proj")
        if dn_s is None:
            missing.append("down_proj")
        raise RuntimeError(
            f"FP8 experts module {type(experts_mod).__name__} missing scale tensor(s) for {missing} "
            f"({_module_storage_summary(experts_mod)}). Set `_qloop_allow_unscaled_fp8 = True` "
            "only if unscaled float8 is intentional."
        )


def _dequant_single_expert(
    experts_mod: nn.Module,
    e: int,
    dtype: torch.dtype = torch.float32,
    *,
    nvfp4_mode_override: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract `(gate_up [2I, H], down [H, I])` in `dtype` for expert `e` without touching other experts."""
    kind = _classify_quant_module(experts_mod)

    if kind == "moe_fp8":
        gu_all = experts_mod.gate_up_proj
        dn_all = experts_mod.down_proj
        n_exp = int(gu_all.shape[0])
        _, gu_s_all = _resolve_moe_fp8_scale(experts_mod, gu_all, role="gu")
        _, dn_s_all = _resolve_moe_fp8_scale(experts_mod, dn_all, role="dn")
        _require_moe_fp8_scales(experts_mod, gu_s_all, dn_s_all)
        bsize = _extract_block_size(experts_mod)
        gu_wshape = (int(gu_all.shape[1]), int(gu_all.shape[2]))
        dn_wshape = (int(dn_all.shape[1]), int(dn_all.shape[2]))
        gu_deq = dequantize_fp8_tensor(
            gu_all[e],
            _slice_expert_scale(gu_s_all, e, n_exp, weight_2d_shape=gu_wshape, block_size=bsize),
            dtype=dtype,
            block_size=bsize,
        )
        dn_deq = dequantize_fp8_tensor(
            dn_all[e],
            _slice_expert_scale(dn_s_all, e, n_exp, weight_2d_shape=dn_wshape, block_size=bsize),
            dtype=dtype,
            block_size=bsize,
        )
        if _is_expert_pair_transposed(
            experts_mod,
            (int(gu_deq.shape[0]), int(gu_deq.shape[1])),
            (int(dn_deq.shape[0]), int(dn_deq.shape[1])),
        ):
            gu_deq = gu_deq.t().contiguous()
            dn_deq = dn_deq.t().contiguous()
        return gu_deq, dn_deq

    if kind == "moe_nvfp4":
        gu_p = getattr(experts_mod, "gate_up_proj_packed", None)
        if not isinstance(gu_p, torch.Tensor):
            gu_p = experts_mod.gate_up_proj
        dn_p = getattr(experts_mod, "down_proj_packed", None)
        if not isinstance(dn_p, torch.Tensor):
            dn_p = experts_mod.down_proj
        n_exp = int(gu_p.shape[0])
        gu_s = experts_mod.gate_up_proj_scale
        dn_s = experts_mod.down_proj_scale
        gu_gs_name, gu_gs = _first_tensor_attr(
            experts_mod,
            ("gate_up_proj_global_scale", "gate_up_global_scale", "gate_up_proj_scale_2"),
        )
        dn_gs_name, dn_gs = _first_tensor_attr(
            experts_mod,
            ("down_proj_global_scale", "down_global_scale", "down_proj_scale_2"),
        )
        gsize = int(getattr(experts_mod, "group_size", None) or 16)
        gu_mode = nvfp4_mode_override or _nvfp4_scale_mode_for_module(experts_mod, gu_gs_name)
        dn_mode = nvfp4_mode_override or _nvfp4_scale_mode_for_module(experts_mod, dn_gs_name)
        gu_deq = dequantize_nvfp4_tensor(
            gu_p[e],
            _slice_expert_scale(gu_s, e, n_exp),  # type: ignore[arg-type]
            _slice_expert_scale(gu_gs, e, n_exp),
            dtype=dtype,
            group_size=gsize,
            global_scale_mode=gu_mode,
        )
        dn_deq = dequantize_nvfp4_tensor(
            dn_p[e],
            _slice_expert_scale(dn_s, e, n_exp),  # type: ignore[arg-type]
            _slice_expert_scale(dn_gs, e, n_exp),
            dtype=dtype,
            group_size=gsize,
            global_scale_mode=dn_mode,
        )
        if _is_expert_pair_transposed(
            experts_mod,
            (int(gu_deq.shape[0]), int(gu_deq.shape[1])),
            (int(dn_deq.shape[0]), int(dn_deq.shape[1])),
        ):
            gu_deq = gu_deq.t().contiguous()
            dn_deq = dn_deq.t().contiguous()
        return gu_deq, dn_deq

    # ModuleList per-expert layout (`experts_mod[e]` or `experts_mod.experts[e]`)
    mlist: nn.ModuleList | None = None
    if isinstance(experts_mod, nn.ModuleList):
        mlist = experts_mod
    elif isinstance(getattr(experts_mod, "experts", None), nn.ModuleList):
        mlist = experts_mod.experts  # type: ignore[assignment]
    if mlist is not None:
        exp_e = mlist[int(e)]
        if hasattr(exp_e, "gate_proj") and hasattr(exp_e, "up_proj") and hasattr(exp_e, "down_proj"):
            wg = _dequant_linear_weight(exp_e.gate_proj, dtype=dtype)
            wu = _dequant_linear_weight(exp_e.up_proj, dtype=dtype)
            wd = _dequant_linear_weight(exp_e.down_proj, dtype=dtype)
            return torch.cat((wg, wu), dim=0), wd
        if hasattr(exp_e, "gate_up_proj") and hasattr(exp_e, "down_proj"):
            wgu = _dequant_linear_weight(exp_e.gate_up_proj, dtype=dtype)
            wd = _dequant_linear_weight(exp_e.down_proj, dtype=dtype)
            return wgu, wd

    # Plain float fused 3D fallback
    gu_f = getattr(experts_mod, "gate_up_proj", None)
    dn_f = getattr(experts_mod, "down_proj", None)
    if isinstance(gu_f, torch.Tensor) and isinstance(dn_f, torch.Tensor) and gu_f.ndim == 3 and dn_f.ndim == 3:
        gu_e = gu_f[e].to(dtype=dtype)
        dn_e = dn_f[e].to(dtype=dtype)
        if _is_expert_pair_transposed(
            experts_mod,
            (int(gu_e.shape[0]), int(gu_e.shape[1])),
            (int(dn_e.shape[0]), int(dn_e.shape[1])),
        ):
            return gu_e.t().contiguous(), dn_e.t().contiguous()
        return gu_e, dn_e

    raise TypeError(f"unsupported experts module layout on {type(experts_mod).__name__}")


def _num_experts_of(experts_mod: nn.Module) -> int:
    gu_p = getattr(experts_mod, "gate_up_proj_packed", None)
    if isinstance(gu_p, torch.Tensor) and gu_p.ndim == 3:
        return int(gu_p.shape[0])
    gu = getattr(experts_mod, "gate_up_proj", None)
    if isinstance(gu, torch.Tensor) and gu.ndim == 3:
        return int(gu.shape[0])
    if isinstance(experts_mod, nn.ModuleList):
        return len(experts_mod)
    mlist = getattr(experts_mod, "experts", None)
    if isinstance(mlist, nn.ModuleList):
        return len(mlist)
    ne = getattr(experts_mod, "num_experts", None)
    if isinstance(ne, int) and ne > 0:
        return ne
    raise TypeError(f"cannot determine num_experts on {type(experts_mod).__name__}")


def _dequant_experts_chunk(
    experts_mod: nn.Module,
    expert_ids: list[int],
    dtype: torch.dtype = torch.bfloat16,
    *,
    nvfp4_mode_override: str | None = None,
) -> tuple[torch.Tensor | list[torch.Tensor], torch.Tensor | list[torch.Tensor]]:
    """Batch-dequantize a chunk of `<= 32` active experts `[C, 2I, H]` & `[C, H, I]` in one 3D call."""
    kind = _classify_quant_module(experts_mod)

    if kind == "moe_fp8":
        gu_all = experts_mod.gate_up_proj
        dn_all = experts_mod.down_proj
        n_exp = int(gu_all.shape[0])
        _, gu_s_all = _resolve_moe_fp8_scale(experts_mod, gu_all, role="gu")
        _, dn_s_all = _resolve_moe_fp8_scale(experts_mod, dn_all, role="dn")
        _require_moe_fp8_scales(experts_mod, gu_s_all, dn_s_all)
        bsize = _extract_block_size(experts_mod)
        eids_t = torch.tensor(expert_ids, device=gu_all.device, dtype=torch.long)
        gu_wshape = (int(gu_all.shape[1]), int(gu_all.shape[2]))
        dn_wshape = (int(dn_all.shape[1]), int(dn_all.shape[2]))
        gu_s_sub = _slice_expert_scale_batch(
            gu_s_all,
            eids_t,
            n_exp,
            weight_2d_shape=gu_wshape,
            block_size=bsize,
        )
        dn_s_sub = _slice_expert_scale_batch(
            dn_s_all,
            eids_t,
            n_exp,
            weight_2d_shape=dn_wshape,
            block_size=bsize,
        )
        try:
            gu_chunk = dequantize_fp8_tensor(gu_all.index_select(0, eids_t), gu_s_sub, dtype=dtype, block_size=bsize)
            dn_chunk = dequantize_fp8_tensor(dn_all.index_select(0, eids_t), dn_s_sub, dtype=dtype, block_size=bsize)
            if _is_expert_pair_transposed(
                experts_mod,
                (int(gu_chunk.shape[1]), int(gu_chunk.shape[2])),
                (int(dn_chunk.shape[1]), int(dn_chunk.shape[2])),
            ):
                gu_chunk = gu_chunk.transpose(1, 2).contiguous()
                dn_chunk = dn_chunk.transpose(1, 2).contiguous()
            return gu_chunk, dn_chunk
        except ValueError:
            pass

    elif kind == "moe_nvfp4":
        gu_p = getattr(experts_mod, "gate_up_proj_packed", None)
        if not isinstance(gu_p, torch.Tensor):
            gu_p = experts_mod.gate_up_proj
        dn_p = getattr(experts_mod, "down_proj_packed", None)
        if not isinstance(dn_p, torch.Tensor):
            dn_p = experts_mod.down_proj
        n_exp = int(gu_p.shape[0])
        eids_t = torch.tensor(expert_ids, device=gu_p.device, dtype=torch.long)
        gu_s = experts_mod.gate_up_proj_scale
        dn_s = experts_mod.down_proj_scale
        gu_gs_name, gu_gs = _first_tensor_attr(
            experts_mod,
            ("gate_up_proj_global_scale", "gate_up_global_scale", "gate_up_proj_scale_2"),
        )
        dn_gs_name, dn_gs = _first_tensor_attr(
            experts_mod,
            ("down_proj_global_scale", "down_global_scale", "down_proj_scale_2"),
        )
        gsize = int(getattr(experts_mod, "group_size", None) or 16)
        gu_mode = nvfp4_mode_override or _nvfp4_scale_mode_for_module(experts_mod, gu_gs_name)
        dn_mode = nvfp4_mode_override or _nvfp4_scale_mode_for_module(experts_mod, dn_gs_name)
        gu_s_sub = _slice_expert_scale_batch(gu_s, eids_t, n_exp)
        dn_s_sub = _slice_expert_scale_batch(dn_s, eids_t, n_exp)
        gu_gs_sub = _slice_expert_scale_batch(gu_gs, eids_t, n_exp)
        dn_gs_sub = _slice_expert_scale_batch(dn_gs, eids_t, n_exp)
        try:
            gu_chunk = dequantize_nvfp4_tensor(
                gu_p.index_select(0, eids_t),
                gu_s_sub,  # type: ignore[arg-type]
                gu_gs_sub,
                dtype=dtype,
                group_size=gsize,
                global_scale_mode=gu_mode,
            )
            dn_chunk = dequantize_nvfp4_tensor(
                dn_p.index_select(0, eids_t),
                dn_s_sub,  # type: ignore[arg-type]
                dn_gs_sub,
                dtype=dtype,
                group_size=gsize,
                global_scale_mode=dn_mode,
            )
            if _is_expert_pair_transposed(
                experts_mod,
                (int(gu_chunk.shape[1]), int(gu_chunk.shape[2])),
                (int(dn_chunk.shape[1]), int(dn_chunk.shape[2])),
            ):
                gu_chunk = gu_chunk.transpose(1, 2).contiguous()
                dn_chunk = dn_chunk.transpose(1, 2).contiguous()
            return gu_chunk, dn_chunk
        except ValueError:
            pass

    gu_list: list[torch.Tensor] = []
    dn_list: list[torch.Tensor] = []
    for eid in expert_ids:
        gu_e, dn_e = _dequant_single_expert(
            experts_mod,
            eid,
            dtype=dtype,
            nvfp4_mode_override=nvfp4_mode_override,
        )
        gu_list.append(gu_e)
        dn_list.append(dn_e)
    return gu_list, dn_list


def auto_dequant_expert_fn(experts_mod: nn.Module, e: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Return `(gate_up [2I, H], down [H, I])` in `torch.float32` for single expert `e`."""
    return _dequant_single_expert(experts_mod, int(e), dtype=torch.float32)


# ============================================================================ zero-persistent-VRAM autograd functions


def _any_tensor_requires_grad(args: Any, kwargs: Any) -> bool:
    for item in args:
        if isinstance(item, torch.Tensor) and item.requires_grad:
            return True
        if isinstance(item, (tuple, list)) and _any_tensor_requires_grad(item, ()):
            return True
    if isinstance(kwargs, dict):
        for item in kwargs.values():
            if isinstance(item, torch.Tensor) and item.requires_grad:
                return True
            if isinstance(item, (tuple, list)) and _any_tensor_requires_grad(item, ()):
                return True
    return False


class _QuantLinearAutogradFn(torch.autograd.Function):
    """Memory-zero 2D quantized Linear autograd: saves NO weights and NO activations for frozen W."""

    @staticmethod
    def forward(  # type: ignore[override]
        ctx: Any,
        x: torch.Tensor,
        bias: torch.Tensor | None,
        mod: nn.Module,
    ) -> torch.Tensor:
        ctx.mod = mod
        ctx.has_bias = bias is not None
        ctx.bias_dtype = bias.dtype if bias is not None else None
        w_deq = _dequant_linear_weight(mod, dtype=x.dtype)
        b_cast = bias.to(device=x.device, dtype=x.dtype) if bias is not None else None
        out = F.linear(x, w_deq, b_cast)
        del w_deq
        return out

    @staticmethod
    def backward(ctx: Any, grad_out: torch.Tensor) -> tuple[torch.Tensor | None, ...]:  # type: ignore[override]
        grad_x: torch.Tensor | None = None
        if ctx.needs_input_grad[0]:
            w_deq = _dequant_linear_weight(ctx.mod, dtype=grad_out.dtype)
            grad_x = torch.matmul(grad_out, w_deq)
            del w_deq

        grad_bias: torch.Tensor | None = None
        if ctx.has_bias and ctx.needs_input_grad[1]:
            grad_bias = grad_out.reshape(-1, grad_out.shape[-1]).sum(dim=0).to(dtype=ctx.bias_dtype)
        return grad_x, grad_bias, None


class _QuantSwiGLUExpertFn(torch.autograd.Function):
    """Single-expert transient SwiGLU autograd (saves neither dequantized W nor `[M_e, 2I]` mid-activations)."""

    @staticmethod
    def forward(  # type: ignore[override]
        ctx: Any,
        x_e: torch.Tensor,
        w_e: torch.Tensor,
        experts_mod: nn.Module,
        expert_idx: int,
    ) -> torch.Tensor:
        ctx.save_for_backward(x_e.detach(), w_e.detach())
        ctx.experts_mod = experts_mod
        ctx.expert_idx = int(expert_idx)
        gu_e, dn_e = _dequant_single_expert(experts_mod, int(expert_idx), dtype=x_e.dtype)
        gate, up = F.linear(x_e, gu_e).chunk(2, dim=-1)
        act_fn = getattr(experts_mod, "act_fn", None) or F.silu
        y_e = F.linear(act_fn(gate) * up, dn_e)
        out_e = y_e * w_e.unsqueeze(-1).to(y_e.dtype)
        del gu_e, dn_e, gate, up, y_e
        return out_e

    @staticmethod
    def backward(ctx: Any, grad_out_e: torch.Tensor) -> tuple[torch.Tensor | None, ...]:  # type: ignore[override]
        x_e_saved, w_e_saved = ctx.saved_tensors
        need_x = bool(ctx.needs_input_grad[0])
        need_w = bool(ctx.needs_input_grad[1])
        if not (need_x or need_w):
            return None, None, None, None

        gu_e, dn_e = _dequant_single_expert(ctx.experts_mod, ctx.expert_idx, dtype=x_e_saved.dtype)
        act_fn = getattr(ctx.experts_mod, "act_fn", None) or F.silu
        grad_xe: torch.Tensor | None = None
        if need_x:
            with torch.enable_grad():
                xe = x_e_saved.detach().requires_grad_(True)
                gate, up = F.linear(xe, gu_e).chunk(2, dim=-1)
                y_e = F.linear(act_fn(gate) * up, dn_e)
                grad_y = grad_out_e.to(y_e.dtype) * w_e_saved.unsqueeze(-1).to(y_e.dtype)
                grad_xe = torch.autograd.grad(y_e, xe, grad_y, retain_graph=False)[0].to(x_e_saved.dtype)
                y_det = y_e.detach()
        else:
            gate, up = F.linear(x_e_saved, gu_e).chunk(2, dim=-1)
            y_det = F.linear(act_fn(gate) * up, dn_e)

        grad_we: torch.Tensor | None = None
        if need_w:
            grad_we = (grad_out_e.float() * y_det.float()).sum(dim=-1).to(w_e_saved.dtype)
        del gu_e, dn_e, gate, up, y_det
        return grad_xe, grad_we, None, None


def _build_moe_dispatch_plan(
    top_k_index: torch.Tensor,
    n_exp: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[tuple[int, int, int]]]:
    """Single-sync GPU `sort` + `bincount` MoE dispatch plan (`0` `.nonzero()`, `0` `.item()` loops)."""
    k_slots = int(top_k_index.shape[1]) if top_k_index.ndim >= 2 else 1
    flat_idx = top_k_index.reshape(-1).long()
    if flat_idx.numel() == 0 or k_slots <= 0 or n_exp <= 0:
        empty = flat_idx[:0]
        return empty, empty, empty, []

    valid = (flat_idx >= 0) & (flat_idx < n_exp)
    safe_idx = torch.where(valid, flat_idx, flat_idx.new_full((), n_exp))
    sorted_e, perm = safe_idx.sort(dim=0, stable=True)
    counts = torch.bincount(sorted_e, minlength=n_exp + 1)
    counts_cpu: list[int] = counts.tolist()

    active_spans: list[tuple[int, int, int]] = []
    offset = 0
    for eid in range(n_exp):
        cnt = int(counts_cpu[eid])
        if cnt > 0:
            active_spans.append((eid, offset, cnt))
            offset += cnt

    perm_valid = perm[:offset]
    tok_valid = torch.div(perm_valid, k_slots, rounding_mode="floor")
    slot_valid = perm_valid % k_slots
    return perm_valid, tok_valid, slot_valid, active_spans


def _moe_routed_forward_chunked(
    flat_hidden: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
    experts_mod: nn.Module,
    *,
    chunk_experts: int = _MOE_CHUNK_EXPERTS,
    nvfp4_mode_override: str | None = None,
) -> torch.Tensor:
    out = torch.zeros_like(flat_hidden)
    if flat_hidden.numel() == 0 or top_k_index.numel() == 0:
        return out

    n_exp = _num_experts_of(experts_mod)
    _, tok_valid, slot_valid, active_spans = _build_moe_dispatch_plan(top_k_index, n_exp)
    if not active_spans:
        return out

    act_fn = getattr(experts_mod, "act_fn", None) or F.silu
    x_sorted = flat_hidden.index_select(0, tok_valid)
    w_sorted = top_k_weights[tok_valid, slot_valid]
    c_step = max(1, int(chunk_experts))

    for c0 in range(0, len(active_spans), c_step):
        chunk_spans = active_spans[c0 : c0 + c_step]
        chunk_eids = [eid for eid, _, _ in chunk_spans]
        gu_chunk, dn_chunk = _dequant_experts_chunk(
            experts_mod,
            chunk_eids,
            dtype=flat_hidden.dtype,
            nvfp4_mode_override=nvfp4_mode_override,
        )
        for local_i, (_eid, off, cnt) in enumerate(chunk_spans):
            x_e = x_sorted[off : off + cnt]
            w_e = w_sorted[off : off + cnt]
            gu_e = gu_chunk[local_i].to(device=x_e.device, dtype=x_e.dtype)
            dn_e = dn_chunk[local_i].to(device=x_e.device, dtype=x_e.dtype)
            gate, up = F.linear(x_e, gu_e).chunk(2, dim=-1)
            y_e = F.linear(act_fn(gate) * up, dn_e)
            contrib = y_e * w_e.unsqueeze(-1).to(y_e.dtype)
            out.index_add_(0, tok_valid[off : off + cnt], contrib)
            del gate, up, y_e, contrib
        del gu_chunk, dn_chunk
    return out


class _QuantMoERoutedAutogradFn(torch.autograd.Function):
    """Single-sync sorted + chunked-3D active-expert MoE autograd (`0` `.item()` / `.nonzero()` loops)."""

    @staticmethod
    def forward(  # type: ignore[override]
        ctx: Any,
        flat_hidden: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
        experts_mod: nn.Module,
    ) -> torch.Tensor:
        ctx.save_for_backward(flat_hidden.detach(), top_k_index.detach(), top_k_weights.detach())
        ctx.experts_mod = experts_mod
        return _moe_routed_forward_chunked(flat_hidden, top_k_index, top_k_weights, experts_mod)

    @staticmethod
    def backward(ctx: Any, grad_out: torch.Tensor) -> tuple[torch.Tensor | None, ...]:  # type: ignore[override]
        flat_hidden, top_k_index, top_k_weights = ctx.saved_tensors
        need_h = bool(ctx.needs_input_grad[0])
        need_w = bool(ctx.needs_input_grad[2])
        if not (need_h or need_w):
            return None, None, None, None

        grad_h = torch.zeros_like(flat_hidden) if need_h else None
        grad_w = (
            torch.zeros(top_k_weights.shape, dtype=top_k_weights.dtype, device=top_k_weights.device)
            if need_w
            else None
        )
        if flat_hidden.numel() == 0 or top_k_index.numel() == 0:
            return grad_h, None, grad_w, None

        experts_mod = ctx.experts_mod
        n_exp = _num_experts_of(experts_mod)
        perm_valid, tok_valid, slot_valid, active_spans = _build_moe_dispatch_plan(top_k_index, n_exp)
        if not active_spans:
            return grad_h, None, grad_w, None

        act_fn = getattr(experts_mod, "act_fn", None) or F.silu
        x_sorted = flat_hidden.index_select(0, tok_valid)
        w_sorted = top_k_weights[tok_valid, slot_valid]
        go_sorted = grad_out.index_select(0, tok_valid)
        flat_grad_w = grad_w.view(-1) if grad_w is not None else None

        for c0 in range(0, len(active_spans), _MOE_CHUNK_EXPERTS):
            chunk_spans = active_spans[c0 : c0 + _MOE_CHUNK_EXPERTS]
            chunk_eids = [eid for eid, _, _ in chunk_spans]
            gu_chunk, dn_chunk = _dequant_experts_chunk(experts_mod, chunk_eids, dtype=flat_hidden.dtype)
            for local_i, (_eid, off, cnt) in enumerate(chunk_spans):
                x_saved = x_sorted[off : off + cnt]
                w_saved = w_sorted[off : off + cnt]
                go_e = go_sorted[off : off + cnt]
                gu_e = gu_chunk[local_i].to(device=x_saved.device, dtype=x_saved.dtype)
                dn_e = dn_chunk[local_i].to(device=x_saved.device, dtype=x_saved.dtype)

                if need_h and grad_h is not None:
                    with torch.enable_grad():
                        xe = x_saved.detach().requires_grad_(True)
                        gate, up = F.linear(xe, gu_e).chunk(2, dim=-1)
                        y_e = F.linear(act_fn(gate) * up, dn_e)
                        grad_y = go_e.to(y_e.dtype) * w_saved.unsqueeze(-1).to(y_e.dtype)
                        grad_xe = torch.autograd.grad(y_e, xe, grad_y, retain_graph=False)[0].to(flat_hidden.dtype)
                        y_det = y_e.detach()
                    grad_h.index_add_(0, tok_valid[off : off + cnt], grad_xe)
                else:
                    gate, up = F.linear(x_saved, gu_e).chunk(2, dim=-1)
                    y_det = F.linear(act_fn(gate) * up, dn_e)

                if flat_grad_w is not None:
                    grad_we = (go_e.float() * y_det.float()).sum(dim=-1).to(top_k_weights.dtype)
                    flat_grad_w.index_copy_(0, perm_valid[off : off + cnt], grad_we)
                del gate, up, y_det
            del gu_chunk, dn_chunk

        return grad_h, None, grad_w, None


# ============================================================================ bridge micro-probing & installation


def _first_module_device(mod: nn.Module) -> torch.device:
    for p in mod.parameters():
        return p.device
    for b in mod.buffers():
        return b.device
    for val in vars(mod).values():
        if isinstance(val, torch.Tensor):
            return val.device
    return torch.device("cpu")


def _infer_activation_dtype(model: nn.Module, device: torch.device) -> torch.dtype:
    for mod in model.modules():
        if (
            isinstance(mod, nn.Embedding)
            and mod.weight.is_floating_point()
            and mod.weight.dtype not in FP8_DTYPES
        ):
            return mod.weight.dtype
    for p in model.parameters():
        if p.is_floating_point() and p.dtype not in FP8_DTYPES and p.ndim == 1:
            return p.dtype
    return torch.bfloat16 if device.type == "cuda" else torch.float32


def _parse_moe_forward_args(
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Extract `(hidden_states, top_k_index, top_k_weights)` across HF>=5 & fallback MoE signatures."""
    pos_tensors = [a for a in args if isinstance(a, torch.Tensor)]
    hidden = pos_tensors[0] if pos_tensors else None
    if hidden is None:
        for k in ("hidden_states", "x", "input"):
            if isinstance(kwargs.get(k), torch.Tensor):
                hidden = kwargs[k]
                break
    if hidden is None:
        raise TypeError("MoE experts forward received no hidden_states Tensor")

    rem_tensors = pos_tensors[1:]
    for k, v in kwargs.items():
        if k in ("hidden_states", "x", "input"):
            continue
        if isinstance(v, torch.Tensor):
            rem_tensors.append(v)

    top_idx: torch.Tensor | None = None
    top_w: torch.Tensor | None = None
    for k in ("top_k_index", "selected_experts", "router_indices", "top_idx"):
        v = kwargs.get(k)
        if isinstance(v, torch.Tensor):
            top_idx = v
            break
    for k in ("top_k_weights", "routing_weights", "router_scores", "top_w"):
        v = kwargs.get(k)
        if isinstance(v, torch.Tensor):
            top_w = v
            break

    for t in rem_tensors:
        if top_idx is None and not t.is_floating_point() and not t.is_complex():
            top_idx = t
        elif top_w is None and t.is_floating_point():
            top_w = t

    if top_idx is None or top_w is None:
        raise TypeError(
            f"could not resolve (top_k_index, top_k_weights) from MoE args={[type(a).__name__ for a in args]} "
            f"kwargs={sorted(kwargs)}"
        )
    return hidden, top_idx, top_w


def _moe_dims_for_probe(experts_mod: nn.Module) -> tuple[int, int]:
    """Return `(num_experts, hidden_size)` for a fused-3D quantized `experts_mod`."""
    kind = _classify_quant_module(experts_mod)
    if kind == "moe_nvfp4":
        gu = getattr(experts_mod, "gate_up_proj_packed", None)
        if not isinstance(gu, torch.Tensor):
            gu = experts_mod.gate_up_proj
        dn = getattr(experts_mod, "down_proj_packed", None)
        if not isinstance(dn, torch.Tensor):
            dn = experts_mod.down_proj
        n_exp = int(gu.shape[0])
        gu_shape = (int(gu.shape[1]), int(gu.shape[2]) * int(gu.element_size()) * 2)
        dn_shape = (int(dn.shape[1]), int(dn.shape[2]) * int(dn.element_size()) * 2)
    else:
        gu = experts_mod.gate_up_proj
        dn = experts_mod.down_proj
        n_exp = int(gu.shape[0])
        gu_shape = (int(gu.shape[1]), int(gu.shape[2]))
        dn_shape = (int(dn.shape[1]), int(dn.shape[2]))
    try:
        is_tr = _is_expert_pair_transposed(experts_mod, gu_shape, dn_shape)
        return n_exp, (gu_shape[0] if is_tr else gu_shape[1])
    except ValueError:
        g0, _ = _dequant_single_expert(experts_mod, 0, dtype=torch.float32)
        return n_exp, int(g0.shape[1])


def _calibrate_nvfp4_linear_scale_mode(
    mod: nn.Module,
    orig_forward: Any,
    x_probe: torch.Tensor,
    *,
    native_fwd_ok: bool = True,
) -> None:
    if getattr(mod, "_qloop_nvfp4_scale_mode", None) is not None:
        return
    gs_name, gs = _first_tensor_attr(mod, ("weight_global_scale", "global_scale", "weight_scale_2"))
    if gs is None:
        return
    default_mode = _nvfp4_scale_mode_for_module(mod, gs_name)
    if native_fwd_ok:
        try:
            with torch.no_grad():
                y_nat_raw = orig_forward(x_probe.detach())
                y_nat = (y_nat_raw[0] if isinstance(y_nat_raw, (tuple, list)) else y_nat_raw).float()
                bias = getattr(mod, "bias", None)
                b_f = bias.to(device=x_probe.device, dtype=torch.float32) if isinstance(bias, torch.Tensor) else None
                xf = x_probe.detach().float()
                w_div = _dequant_linear_weight(mod, dtype=torch.float32, nvfp4_mode_override="divisor")
                w_mul = _dequant_linear_weight(mod, dtype=torch.float32, nvfp4_mode_override="multiplier")
                err_div = float((y_nat - F.linear(xf, w_div, b_f)).abs().max().item())
                err_mul = float((y_nat - F.linear(xf, w_mul, b_f)).abs().max().item())
                if math.isfinite(err_div) and math.isfinite(err_mul):
                    if err_mul < 0.1 * err_div:
                        mod._qloop_nvfp4_scale_mode = "multiplier"  # type: ignore[attr-defined]
                        return
                    if err_div < 0.1 * err_mul:
                        mod._qloop_nvfp4_scale_mode = "divisor"  # type: ignore[attr-defined]
                        return
        except Exception:
            pass
    mod._qloop_nvfp4_scale_mode = default_mode  # type: ignore[attr-defined]


def _calibrate_nvfp4_moe_scale_mode(
    mod: nn.Module,
    orig_forward: Any,
    h_probe: torch.Tensor,
    idx_probe: torch.Tensor,
    w_probe: torch.Tensor,
    *,
    native_fwd_ok: bool = True,
) -> None:
    if getattr(mod, "_qloop_nvfp4_scale_mode", None) is not None:
        return
    gs_name, gs = _first_tensor_attr(mod, ("gate_up_proj_global_scale", "gate_up_global_scale", "gate_up_proj_scale_2"))
    if gs is None:
        return
    default_mode = _nvfp4_scale_mode_for_module(mod, gs_name)
    if native_fwd_ok:
        try:
            with torch.no_grad():
                y_nat_raw = orig_forward(h_probe.detach(), idx_probe, w_probe.detach())
                y_nat = (y_nat_raw[0] if isinstance(y_nat_raw, (tuple, list)) else y_nat_raw).float()
                flat_h = h_probe.detach().float().reshape(-1, h_probe.shape[-1])
                idx_2d = idx_probe.reshape(flat_h.shape[0], -1).long()
                w_2d = w_probe.detach().float().reshape(flat_h.shape[0], -1)
                errs: dict[str, float] = {}
                for mode in ("divisor", "multiplier"):
                    out_m = _moe_routed_forward_chunked(
                        flat_h,
                        idx_2d,
                        w_2d,
                        mod,
                        nvfp4_mode_override=mode,
                    ).reshape_as(y_nat)
                    errs[mode] = float((y_nat - out_m).abs().max().item())
                if math.isfinite(errs["divisor"]) and math.isfinite(errs["multiplier"]):
                    if errs["multiplier"] < 0.1 * errs["divisor"]:
                        mod._qloop_nvfp4_scale_mode = "multiplier"  # type: ignore[attr-defined]
                        return
                    if errs["divisor"] < 0.1 * errs["multiplier"]:
                        mod._qloop_nvfp4_scale_mode = "divisor"  # type: ignore[attr-defined]
                        return
        except Exception:
            pass
    mod._qloop_nvfp4_scale_mode = default_mode  # type: ignore[attr-defined]


def _micro_probe_linear(mod: nn.Module, act_dtype: torch.dtype) -> tuple[bool, bool]:
    """Return `(native_fwd_ok, native_grad_ok)` for `mod.forward` on 2 synthetic tokens."""
    dev = _first_module_device(mod)
    if dev.type == "meta":
        return False, False
    in_f, _ = _linear_in_out_features(mod)
    orig_forward = getattr(mod, "_qloop_orig_forward", mod.forward)
    idx = torch.arange(2 * in_f, device=dev, dtype=torch.float32).reshape(2, in_f)
    x_base = (torch.sin(idx * 0.37 + 0.13) * 0.5).to(dtype=act_dtype)

    native_fwd_ok = False
    try:
        with torch.no_grad():
            yp_fwd = orig_forward(x_base.detach().clone())
            y_fwd = yp_fwd[0] if isinstance(yp_fwd, (tuple, list)) else yp_fwd
            native_fwd_ok = bool(isinstance(y_fwd, torch.Tensor) and torch.isfinite(y_fwd).all())
    except Exception:
        native_fwd_ok = False

    if _classify_quant_module(mod) == "linear_nvfp4":
        _calibrate_nvfp4_linear_scale_mode(mod, orig_forward, x_base, native_fwd_ok=native_fwd_ok)

    if not native_fwd_ok:
        return False, False

    try:
        with torch.enable_grad():
            xp = x_base.detach().clone().requires_grad_(True)
            yp = orig_forward(xp)
            y_main = yp[0] if isinstance(yp, (tuple, list)) else yp
            if not isinstance(y_main, torch.Tensor) or not y_main.requires_grad:
                return True, False
            weights = torch.linspace(0.5, 1.5, steps=max(1, y_main.numel()), device=dev, dtype=torch.float32)
            loss = (y_main.float().reshape(-1) * weights).mean()
            loss.backward()
            gx = xp.grad
            grad_ok = bool(
                gx is not None
                and torch.isfinite(gx).all()
                and float(gx.float().abs().max().item()) > 0.0
            )
            return True, grad_ok
    except Exception:
        return True, False


def _micro_probe_moe_experts(mod: nn.Module, act_dtype: torch.dtype) -> tuple[bool, bool]:
    """Return `(native_fwd_ok, native_grad_ok)` for `mod.forward` (`h.grad` & `top_w.grad`) on 2 tokens."""
    dev = _first_module_device(mod)
    if dev.type == "meta":
        return False, False
    n_exp, hidden = _moe_dims_for_probe(mod)
    orig_forward = getattr(mod, "_qloop_orig_forward", mod.forward)
    idx = torch.arange(2 * hidden, device=dev, dtype=torch.float32).reshape(2, hidden)
    h_base = (torch.sin(idx * 0.29 + 0.17) * 0.5).to(dtype=act_dtype)
    k = min(2, max(1, n_exp))
    if k == 1:
        top_idx = torch.zeros((2, 1), device=dev, dtype=torch.long)
        w_base = torch.ones((2, 1), device=dev, dtype=torch.float32)
    else:
        top_idx = torch.tensor([[0, 1 % n_exp], [1 % n_exp, 2 % n_exp]], device=dev, dtype=torch.long)
        w_base = torch.tensor([[0.6, 0.4], [0.7, 0.3]], device=dev, dtype=torch.float32)

    native_fwd_ok = False
    try:
        with torch.no_grad():
            yp_fwd = orig_forward(h_base.detach().clone(), top_idx, w_base.detach().clone())
            y_fwd = yp_fwd[0] if isinstance(yp_fwd, (tuple, list)) else yp_fwd
            native_fwd_ok = bool(isinstance(y_fwd, torch.Tensor) and torch.isfinite(y_fwd).all())
    except Exception:
        native_fwd_ok = False

    if _classify_quant_module(mod) == "moe_nvfp4":
        _calibrate_nvfp4_moe_scale_mode(mod, orig_forward, h_base, top_idx, w_base, native_fwd_ok=native_fwd_ok)

    if not native_fwd_ok:
        return False, False

    try:
        with torch.enable_grad():
            hp = h_base.detach().clone().requires_grad_(True)
            wp = w_base.detach().clone().requires_grad_(True)
            yp = orig_forward(hp, top_idx, wp)
            y_main = yp[0] if isinstance(yp, (tuple, list)) else yp
            if not isinstance(y_main, torch.Tensor) or not y_main.requires_grad:
                return True, False
            weights = torch.linspace(0.5, 1.5, steps=max(1, y_main.numel()), device=dev, dtype=torch.float32)
            loss = (y_main.float().reshape(-1) * weights).mean()
            loss.backward()
            gh, gw = hp.grad, wp.grad
            grad_ok = bool(
                gh is not None
                and gw is not None
                and torch.isfinite(gh).all()
                and torch.isfinite(gw).all()
                and float(gh.float().abs().max().item()) > 0.0
                and float(gw.float().abs().max().item()) > 0.0
            )
            return True, grad_ok
    except Exception:
        return True, False


def _fallback_linear_nograd(
    mod: nn.Module,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> torch.Tensor:
    """Zero-persistent-VRAM transient dequantization `no_grad` fallback in `x.dtype` (`bf16` on GPU)."""
    x = args[0] if args else next(v for v in kwargs.values() if isinstance(v, torch.Tensor))
    bias = getattr(mod, "bias", None)
    if not isinstance(bias, torch.Tensor):
        bias = None
    with torch.no_grad():
        w_deq = _dequant_linear_weight(mod, dtype=x.dtype)
        b_cast = bias.to(device=x.device, dtype=x.dtype) if bias is not None else None
        out = F.linear(x, w_deq, b_cast)
        del w_deq
        return out


def _fallback_moe_nograd(
    mod: nn.Module,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> torch.Tensor:
    """Single-sync sorted + chunked-3D transient dequantization `no_grad` fallback in `hidden.dtype`."""
    hidden, top_idx, top_w = _parse_moe_forward_args(args, kwargs)
    orig_shape = hidden.shape
    with torch.no_grad():
        flat_h = hidden.reshape(-1, orig_shape[-1])
        idx_2d = top_idx.reshape(flat_h.shape[0], -1).long()
        w_2d = top_w.reshape(flat_h.shape[0], -1)
        flat_out = _moe_routed_forward_chunked(flat_h, idx_2d, w_2d, mod)
        return flat_out.reshape(orig_shape)


def _wrap_quantized_linear(
    mod: nn.Module,
    *,
    native_fwd_ok: bool = True,
    warned_classes: set[str] | None = None,
) -> None:
    if getattr(mod, "_qloop_quant_bridge_installed", False):
        return
    in_f, out_f = _linear_in_out_features(mod)
    if not isinstance(getattr(mod, "in_features", None), int):
        mod.in_features = in_f  # type: ignore[attr-defined]
    if not isinstance(getattr(mod, "out_features", None), int):
        mod.out_features = out_f  # type: ignore[attr-defined]

    orig_forward = mod.forward
    warn_set = warned_classes if warned_classes is not None else set()
    mod._qloop_native_fwd_ok = bool(native_fwd_ok)  # type: ignore[attr-defined]

    @functools.wraps(orig_forward)
    def bridged_linear_forward(*args: Any, **kwargs: Any) -> Any:
        if not (torch.is_grad_enabled() and _any_tensor_requires_grad(args, kwargs)):
            if getattr(mod, "_qloop_native_fwd_ok", False):
                try:
                    return orig_forward(*args, **kwargs)
                except (ImportError, ModuleNotFoundError, NotImplementedError) as exc:
                    mod._qloop_native_fwd_ok = False  # type: ignore[attr-defined]
                    cls_name = type(mod).__name__
                    if cls_name not in warn_set:
                        warn_set.add(cls_name)
                        log.warning(
                            "Native quantized linear kernel failed at runtime on %s (%s: %s); "
                            "switching to transient PyTorch fallback.",
                            cls_name,
                            type(exc).__name__,
                            exc,
                        )
            return _fallback_linear_nograd(mod, args, kwargs)
        x = args[0] if args else next(v for v in kwargs.values() if isinstance(v, torch.Tensor))
        bias = getattr(mod, "bias", None)
        if not isinstance(bias, torch.Tensor):
            bias = None
        return _QuantLinearAutogradFn.apply(x, bias, mod)

    mod._qloop_orig_forward = orig_forward  # type: ignore[attr-defined]
    mod._qloop_quant_bridge_installed = True  # type: ignore[attr-defined]
    mod.forward = bridged_linear_forward  # type: ignore[method-assign]


def _wrap_quantized_moe_experts(
    mod: nn.Module,
    *,
    native_fwd_ok: bool = True,
    warned_classes: set[str] | None = None,
) -> None:
    if getattr(mod, "_qloop_quant_bridge_installed", False):
        return
    orig_forward = mod.forward
    warn_set = warned_classes if warned_classes is not None else set()
    mod._qloop_native_fwd_ok = bool(native_fwd_ok)  # type: ignore[attr-defined]

    @functools.wraps(orig_forward)
    def bridged_moe_forward(*args: Any, **kwargs: Any) -> Any:
        if not (torch.is_grad_enabled() and _any_tensor_requires_grad(args, kwargs)):
            if getattr(mod, "_qloop_native_fwd_ok", False):
                try:
                    return orig_forward(*args, **kwargs)
                except (ImportError, ModuleNotFoundError, NotImplementedError) as exc:
                    mod._qloop_native_fwd_ok = False  # type: ignore[attr-defined]
                    cls_name = type(mod).__name__
                    if cls_name not in warn_set:
                        warn_set.add(cls_name)
                        log.warning(
                            "Native quantized MoE kernel failed at runtime on %s (%s: %s); "
                            "switching to transient PyTorch fallback.",
                            cls_name,
                            type(exc).__name__,
                            exc,
                        )
            return _fallback_moe_nograd(mod, args, kwargs)
        hidden, top_idx, top_w = _parse_moe_forward_args(args, kwargs)
        orig_shape = hidden.shape
        flat_h = hidden.reshape(-1, orig_shape[-1])
        idx_2d = top_idx.reshape(flat_h.shape[0], -1).long()
        w_2d = top_w.reshape(flat_h.shape[0], -1)
        flat_out = _QuantMoERoutedAutogradFn.apply(flat_h, idx_2d, w_2d, mod)
        return flat_out.reshape(orig_shape)

    mod._qloop_orig_forward = orig_forward  # type: ignore[attr-defined]
    mod._qloop_quant_bridge_installed = True  # type: ignore[attr-defined]
    mod.forward = bridged_moe_forward  # type: ignore[method-assign]


def _layout_probe_key(mod: nn.Module, kind: str, dev: torch.device) -> tuple[Any, ...]:
    inst_fwd = vars(mod).get("_qloop_orig_forward", vars(mod).get("forward"))
    if kind == "linear_fp8":
        sname, s = _resolve_linear_fp8_scale(mod, mod.weight)
        return (kind, type(mod), inst_fwd, mod.weight.dtype, sname, None if s is None else s.ndim, str(dev))
    if kind == "linear_nvfp4":
        wp = getattr(mod, "weight_packed", None)
        if not isinstance(wp, torch.Tensor):
            wp = mod.weight
        gs_name, _ = _first_tensor_attr(mod, ("weight_global_scale", "global_scale", "weight_scale_2"))
        return (kind, type(mod), inst_fwd, wp.dtype, gs_name, str(dev))
    if kind == "moe_fp8":
        sname, gu_s = _resolve_moe_fp8_scale(mod, mod.gate_up_proj, role="gu")
        return (kind, type(mod), inst_fwd, mod.gate_up_proj.dtype, sname, None if gu_s is None else gu_s.ndim, str(dev))
    gu_p = getattr(mod, "gate_up_proj_packed", None)
    if not isinstance(gu_p, torch.Tensor):
        gu_p = mod.gate_up_proj
    gs_name, _ = _first_tensor_attr(mod, ("gate_up_proj_global_scale", "gate_up_global_scale", "gate_up_proj_scale_2"))
    return (kind, type(mod), inst_fwd, gu_p.dtype, gs_name, str(dev))


def install_quantized_autograd_bridge(model: nn.Module) -> dict[str, Any]:
    """Micro-probe quantized linear & MoE modules and install transient `dL/dx` + `no_grad` bridges.

    Idempotent: already-bridged modules are counted and skipped without double-wrapping.
    Native inference kernels execute at full Blackwell speed under `no_grad()` when `native_fwd_ok`;
    otherwise modules transparently fall back to transient `bf16` Tensor Core execution.
    """
    qmeta = detect_model_quantization(model)
    seen: set[int] = set()
    probe_cache: dict[tuple[Any, ...], tuple[bool, bool]] = {}
    scale_mode_cache: dict[tuple[Any, ...], str] = {}
    warned_classes: set[str] = set()

    bridged_linear = 0
    bridged_moe = 0
    native_fwd_ok_count = 0
    native_grad_ok = 0
    already_bridged = 0

    for mod in model.modules():
        if id(mod) in seen:
            continue
        seen.add(id(mod))
        kind = _classify_quant_module(mod)
        if kind is None:
            continue

        if kind in ("linear_fp8", "linear_nvfp4"):
            in_f, out_f = _linear_in_out_features(mod)
            if not isinstance(getattr(mod, "in_features", None), int):
                mod.in_features = in_f  # type: ignore[attr-defined]
            if not isinstance(getattr(mod, "out_features", None), int):
                mod.out_features = out_f  # type: ignore[attr-defined]

        dev = _first_module_device(mod)
        if getattr(mod, "_qloop_quant_bridge_installed", False):
            already_bridged += 1
            if getattr(mod, "_qloop_probed_device", str(dev)) != str(dev):
                act_dtype = _infer_activation_dtype(model, dev)
                pkey = _layout_probe_key(mod, kind, dev)
                if pkey not in probe_cache:
                    if kind in ("linear_fp8", "linear_nvfp4"):
                        probe_cache[pkey] = _micro_probe_linear(mod, act_dtype)
                    else:
                        probe_cache[pkey] = _micro_probe_moe_experts(mod, act_dtype)
                fwd_ok, _ = probe_cache[pkey]
                mod._qloop_native_fwd_ok = bool(fwd_ok)  # type: ignore[attr-defined]
                mod._qloop_probed_device = str(dev)  # type: ignore[attr-defined]
            if getattr(mod, "_qloop_native_fwd_ok", False):
                native_fwd_ok_count += 1
            continue

        act_dtype = _infer_activation_dtype(model, dev)
        pkey = _layout_probe_key(mod, kind, dev)

        if pkey not in probe_cache:
            if kind in ("linear_fp8", "linear_nvfp4"):
                probe_cache[pkey] = _micro_probe_linear(mod, act_dtype)
            else:
                probe_cache[pkey] = _micro_probe_moe_experts(mod, act_dtype)
            cal_mode = getattr(mod, "_qloop_nvfp4_scale_mode", None)
            if isinstance(cal_mode, str):
                scale_mode_cache[pkey] = cal_mode
        elif pkey in scale_mode_cache and getattr(mod, "_qloop_nvfp4_scale_mode", None) is None:
            mod._qloop_nvfp4_scale_mode = scale_mode_cache[pkey]  # type: ignore[attr-defined]

        fwd_ok, grad_ok = probe_cache[pkey]
        mod._qloop_native_fwd_ok = bool(fwd_ok)  # type: ignore[attr-defined]
        mod._qloop_probed_device = str(dev)  # type: ignore[attr-defined]
        if fwd_ok:
            native_fwd_ok_count += 1
        if grad_ok:
            native_grad_ok += 1
            continue

        if kind in ("linear_fp8", "linear_nvfp4"):
            _wrap_quantized_linear(mod, native_fwd_ok=fwd_ok, warned_classes=warned_classes)
            bridged_linear += 1
        else:
            _wrap_quantized_moe_experts(mod, native_fwd_ok=fwd_ok, warned_classes=warned_classes)
            bridged_moe += 1

    total_quant = (
        qmeta["linear_fp8_count"]
        + qmeta["linear_nvfp4_count"]
        + qmeta["moe_fp8_count"]
        + qmeta["moe_nvfp4_count"]
    )
    return {
        **qmeta,
        "total_quantized_modules": int(total_quant),
        "bridged_linear_count": int(bridged_linear),
        "bridged_moe_count": int(bridged_moe),
        "bridged_total": int(bridged_linear + bridged_moe + already_bridged),
        "already_bridged_count": int(already_bridged),
        "native_fwd_ok_count": int(native_fwd_ok_count),
        "native_grad_ok_count": int(native_grad_ok),
        "probed_layouts": len(probe_cache),
    }


__all__ = [
    "FP8_DTYPES",
    "NVFP4_E2M1_TABLE",
    "auto_dequant_expert_fn",
    "dequantize_fp8_tensor",
    "dequantize_nvfp4_tensor",
    "detect_model_quantization",
    "install_quantized_autograd_bridge",
    "is_quantized_module",
    "quantize_fp8_blockwise",
    "quantize_nvfp4_tensor",
]
