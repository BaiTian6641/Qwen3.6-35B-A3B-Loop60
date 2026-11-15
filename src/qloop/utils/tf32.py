"""TF32 runtime configuration helper for NVIDIA Ampere/Hopper/Blackwell Tensor Cores.

PyTorch defaults `torch.backends.cuda.matmul.allow_tf32` to `False`, forcing every
`float32` matmul inside HSSD's WY chunk kernel, D-LSA Ward clustering, and optimizer
projections onto scalar FP32 CUDA cores instead of Tensor Cores (`~8x` lower throughput
on Blackwell RTX PRO 6000). Enabling TF32 routes `float32` `matmul` / `bmm` onto TF32
Tensor Cores while preserving `fp32` cumulative log-decay, `fp32` triangular solves,
and `fp64` phase angle reduction.
"""
from __future__ import annotations

import logging

import torch

log = logging.getLogger("qloop")

_LAST_TF32_STATE: bool | None = None


def is_tf32_configured() -> bool:
    """Return True if `configure_tf32` has already been invoked in this process."""
    return _LAST_TF32_STATE is not None


def configure_tf32(allow_tf32: bool = True) -> dict[str, bool]:
    """Configure global CUDA/cuDNN TF32 flags and PyTorch `float32` matmul precision.

    Args:
        allow_tf32: When `True`, enables TF32 for `cuda.matmul` and `cudnn` and sets
            `float32` matmul precision to `"high"`. When `False`, disables TF32 and
            restores `"highest"` IEEE-754 `fp32` matmul precision.

    Returns:
        Dictionary of boolean runtime state flags.
    """
    global _LAST_TF32_STATE
    flag = bool(allow_tf32)

    cuda_backend = getattr(torch.backends, "cuda", None)
    matmul_backend = getattr(cuda_backend, "matmul", None) if cuda_backend is not None else None
    if matmul_backend is not None and hasattr(matmul_backend, "allow_tf32"):
        matmul_backend.allow_tf32 = flag

    cudnn_backend = getattr(torch.backends, "cudnn", None)
    if cudnn_backend is not None and hasattr(cudnn_backend, "allow_tf32"):
        cudnn_backend.allow_tf32 = flag

    target_prec = "high" if flag else "highest"
    set_prec = getattr(torch, "set_float32_matmul_precision", None)
    if callable(set_prec):
        set_prec(target_prec)

    get_prec = getattr(torch, "get_float32_matmul_precision", None)
    active_prec = str(get_prec()) if callable(get_prec) else target_prec

    cuda_tf32 = bool(getattr(matmul_backend, "allow_tf32", flag)) if matmul_backend is not None else flag
    cudnn_tf32 = bool(getattr(cudnn_backend, "allow_tf32", flag)) if cudnn_backend is not None else flag
    high_prec = active_prec in ("high", "medium")

    if flag != _LAST_TF32_STATE:
        log.info(
            "TF32 precision configured: allow_tf32=%s (cuda.matmul=%s, cudnn=%s, float32_matmul_precision=%s)",
            flag,
            cuda_tf32,
            cudnn_tf32,
            active_prec,
        )
    _LAST_TF32_STATE = flag

    return {
        "allow_tf32": flag,
        "cuda_matmul_allow_tf32": cuda_tf32,
        "cudnn_allow_tf32": cudnn_tf32,
        "float32_matmul_high": high_prec,
    }
