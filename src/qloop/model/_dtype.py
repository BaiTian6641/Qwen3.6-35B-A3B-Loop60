"""Precision helpers shared by model code."""
from __future__ import annotations

import torch


def upcast(t: torch.Tensor) -> torch.Tensor:
    """bf16/fp16/fp32 -> fp32, fp64 stays fp64 (lets unit tests verify semantics in fp64)."""
    return t if t.dtype == torch.float64 else t.float()
