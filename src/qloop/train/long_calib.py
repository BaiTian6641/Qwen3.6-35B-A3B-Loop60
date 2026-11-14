"""Long-Prefix `no_grad` + Short-Suffix `grad` Read-Side Calibration (`Gate 2a`, Fix C1).

At `1M` context (`1,048,576` tokens), softmax attention log-partition entropy reaches
`ln(10^6) = 13.82`, requiring calibration of read-side attention parameters (`QK-Norm`,
attention output gates, per-pass Q/O LoRA, and `D-LSA` query/key indexers) against realistic
long-prefix KV caches and Ward centroids without materializing prefix activations in autograd.

`prefix_suffix_calibration_step(model, input_ids, prefix_len, chunk_size, loss_fn)`:
  1. Streams `[0 : prefix_len]` in `chunk_size` slices under `torch.no_grad()` (`use_cache=True`)
     through the text decoder (skipping `lm_head` on prefix chunks to save `248K`-vocab HBM),
     populating long-prefix KV caches, `D-LSA` incremental Ward centroids, and linear recurrent
     (`GDN-2` / `HSSD`) states with `O(1)` prefix activation memory.
  2. Detaches the populated prefix cache cleanly (optionally quantizing prefix KV via Tier-1
     `FP8KVCacheQuantizer`), then runs `[prefix_len : T]` WITH gradients enabled.
  3. Evaluates `loss_fn` (+ any `D-LSA` KL distillation / loop auxiliary loss on the suffix) and
     backpropagates cleanly into read-side trainable parameters!
"""
from __future__ import annotations

import inspect
from dataclasses import dataclass, field, fields
from typing import Any, Callable

import torch
import torch.nn.functional as F
from torch import nn

from ..model.introspect import find_text_model
from ..model.offload import FP8KVCacheQuantizer
from .bptt import _detach_nested, _ReplayKVCacheProxy


@dataclass
class LongCalibConfig:
    """Configuration for Gate 2a Long-Prefix (`no_grad`) + Short-Suffix (`grad`) calibration."""

    enabled: bool = True
    prefix_len: int = 524_288
    max_prefix_len: int = 1_015_808
    suffix_len: int = 32_768
    chunk_size: int = 4_096
    fp8_kv_cache: bool = False

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> LongCalibConfig:
        raw = dict(d or {})
        known = {f.name for f in fields(cls)}
        bad = set(raw) - known
        if bad:
            raise ValueError(f"unknown long_calib.* keys {sorted(bad)}; valid: {sorted(known)}")
        return cls(**raw)


@dataclass
class CalibrationStepResult:
    """Result of a single Long-Prefix (`no_grad`) + Short-Suffix (`grad`) calibration step."""

    loss: torch.Tensor
    task_loss: float
    lsa_aux_loss: float
    suffix_logits: torch.Tensor
    prefix_len: int
    suffix_len: int
    num_prefix_chunks: int
    grad_norms: dict[str, float] = field(default_factory=dict)


def _quantize_prefix_cache_fp8_(cache: Any, quantizer: FP8KVCacheQuantizer) -> None:
    """Simulate/apply Tier-1 `FP8 E4M3` precision on populated prefix full-attention KV caches."""
    kc = getattr(cache, "key_cache", None)
    vc = getattr(cache, "value_cache", None)
    if isinstance(kc, list) and isinstance(vc, list):
        for i in range(min(len(kc), len(vc))):
            if isinstance(kc[i], torch.Tensor) and kc[i].numel() > 0:
                kc[i] = quantizer.dequantize(quantizer.quantize(kc[i]), dtype=kc[i].dtype)
            if isinstance(vc[i], torch.Tensor) and vc[i].numel() > 0:
                vc[i] = quantizer.dequantize(quantizer.quantize(vc[i]), dtype=vc[i].dtype)


def _invoke_calib_loss(
    loss_fn: Callable[..., torch.Tensor] | None,
    suffix_logits: torch.Tensor,
    suffix_ids: torch.Tensor,
    suffix_labels: torch.Tensor | None,
) -> torch.Tensor:
    targets = suffix_labels if suffix_labels is not None else suffix_ids
    if loss_fn is None:
        if suffix_logits.shape[1] < 2:
            raise ValueError("default next-token CE requires suffix_len >= 2 when loss_fn is None")
        flat_logits = suffix_logits[:, :-1].reshape(-1, suffix_logits.shape[-1]).float()
        flat_targets = targets[:, 1:].reshape(-1)
        return F.cross_entropy(flat_logits, flat_targets, ignore_index=-100)

    sig = inspect.signature(loss_fn)
    params = [
        p
        for p in sig.parameters.values()
        if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    has_varargs = any(p.kind == inspect.Parameter.VAR_POSITIONAL for p in sig.parameters.values())
    if has_varargs or len(params) >= 2:
        return loss_fn(suffix_logits, targets)
    return loss_fn(suffix_logits)


def prefix_suffix_calibration_step(
    model: nn.Module,
    input_ids: torch.Tensor,
    prefix_len: int,
    chunk_size: int = 4096,
    loss_fn: Callable[..., torch.Tensor] | None = None,
    *,
    labels: torch.Tensor | None = None,
    backward: bool = True,
    lsa_aux_weight: float = 1.0,
    loop_aux_weight: float = 1.0,
    fp8_kv_cache: bool = False,
    fp8_quantizer: FP8KVCacheQuantizer | None = None,
) -> CalibrationStepResult:
    """Stream `[0 : prefix_len]` under `no_grad`, then run `[prefix_len : T]` with `grad` & backward."""
    if fp8_kv_cache and fp8_quantizer is None:
        fp8_quantizer = FP8KVCacheQuantizer()
    root = getattr(model, "model", model) if hasattr(model, "named_trainable") else model
    if input_ids.ndim != 2:
        raise ValueError(f"input_ids must have shape [B, T], got {tuple(input_ids.shape)}")
    total_len = input_ids.shape[1]
    if not 0 <= prefix_len < total_len:
        raise ValueError(f"need 0 <= prefix_len < total_len ({total_len}), got prefix_len={prefix_len}")
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be > 0, got {chunk_size}")

    try:
        tm: nn.Module | None = find_text_model(root)
    except Exception:
        tm = None

    # Clear stale LSA losses prior to streaming
    scan_root = tm if tm is not None else root
    for mod in scan_root.modules():
        lsa_ix = getattr(mod, "lsa_indexer", None)
        if lsa_ix is not None:
            lsa_ix.pop_losses()

    # Phase 1: Stream `[0 : prefix_len]` in `chunk_size` chunks under `torch.no_grad()`
    cache: Any = None
    n_prefix_chunks = 0
    if prefix_len > 0:
        with torch.no_grad():
            for s0 in range(0, prefix_len, chunk_size):
                s1 = min(prefix_len, s0 + chunk_size)
                p_chunk = input_ids[:, s0:s1]
                if tm is not None and tm is not root:
                    p_out = tm(input_ids=p_chunk, past_key_values=cache, use_cache=True)
                else:
                    p_out = root(p_chunk, past_key_values=cache, use_cache=True)
                cache = p_out.past_key_values
                n_prefix_chunks += 1
        if cache is not None:
            _detach_nested(cache)
            if fp8_quantizer is not None:
                with torch.no_grad():
                    _quantize_prefix_cache_fp8_(cache, fp8_quantizer)

    # Phase 2: Run `[prefix_len : total_len]` WITH gradients enabled (checkpoint-safe cache view)
    suffix_ids = input_ids[:, prefix_len:]
    suffix_labels = labels[:, prefix_len:] if labels is not None else None
    suffix_cache = _ReplayKVCacheProxy(cache) if cache is not None else None
    suffix_out = root(suffix_ids, past_key_values=suffix_cache, use_cache=True)
    suffix_logits = suffix_out.logits

    task_loss = _invoke_calib_loss(loss_fn, suffix_logits, suffix_ids, suffix_labels)
    total_loss = task_loss

    lsa_losses = [
        loss_t
        for mod in scan_root.modules()
        if getattr(mod, "lsa_indexer", None) is not None
        for loss_t in mod.lsa_indexer.pop_losses()
    ]
    lsa_val = 0.0
    if lsa_losses and lsa_aux_weight > 0:
        lsa_sum = torch.stack([t.to(total_loss.dtype) for t in lsa_losses]).sum()
        lsa_val = float(lsa_sum.detach())
        total_loss = total_loss + float(lsa_aux_weight) * lsa_sum

    if tm is not None and loop_aux_weight > 0:
        layers = getattr(tm, "layers", None)
        ctl = getattr(layers[0], "_ctl", None) if (layers is not None and len(layers) > 0) else None
        if ctl is not None:
            l_aux = ctl.aux_loss()
            if l_aux is not None:
                total_loss = total_loss + float(loop_aux_weight) * l_aux.to(total_loss.dtype)

    if backward and total_loss.requires_grad:
        total_loss.backward()

    grad_norms: dict[str, float] = {}
    for name, p in root.named_parameters():
        if p.requires_grad and p.grad is not None:
            grad_norms[name] = float(p.grad.detach().float().norm())

    return CalibrationStepResult(
        loss=total_loss.detach(),
        task_loss=float(task_loss.detach()),
        lsa_aux_loss=lsa_val,
        suffix_logits=suffix_logits.detach(),
        prefix_len=int(prefix_len),
        suffix_len=int(total_len - prefix_len),
        num_prefix_chunks=n_prefix_chunks,
        grad_norms=grad_norms,
    )
