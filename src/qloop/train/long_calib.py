"""Long-Prefix `no_grad` + Short-Suffix `grad` Read-Side Calibration (`Gate 2a`, Fixes C1, D2, D3).

At `1M` context (`1,048,576` tokens), softmax attention log-partition entropy reaches
`ln(10^6) = 13.82`, requiring calibration of read-side attention parameters (`QK-Norm`,
attention output gates, per-pass Q/O LoRA, and `D-LSA` query/key indexers) against realistic
long-prefix KV caches and Ward centroids without materializing prefix activations in autograd.

`prefix_suffix_calibration_step(model, input_ids, prefix_len, chunk_size, loss_fn)`:
  1. Streams `[0 : prefix_len]` in `chunk_size` slices under `torch.no_grad()` (`use_cache=True`)
     through the text decoder (skipping `lm_head` on prefix chunks to save `248K`-vocab HBM).
     Once `global_step >= sparse_prefix_after_step` or `lsa_manager.recall_ready` is true (`D3`),
     temporarily switches `lsa_manager` to `prefix_lsa_mode` (`"sparse"`) during prefix streaming
     for sub-quadratic prefix attention, then restores training mode for the suffix.
  2. Detaches the populated prefix cache cleanly (optionally quantizing prefix KV via Tier-1
     `FP8KVCacheQuantizer`), then runs `[prefix_len : T]` WITH gradients enabled to pre-head
     `last_hidden_state` `[B, suffix_len, D]` (`D2`: never materializes `[B, 32768, 248320]`
     full-vocabulary suffix logits).
  3. Evaluates chunked cross-entropy via `qloop.train.loss.lm_loss(..., impl="chunked")` (or custom
     `loss_fn`) + any `D-LSA` KL distillation / loop auxiliary loss on the suffix, and
     backpropagates cleanly into read-side trainable parameters.
"""
from __future__ import annotations

import contextlib
import inspect
from collections.abc import Iterator
from dataclasses import dataclass, field, fields
from typing import Any, Callable

import torch
import torch.nn.functional as F
from torch import nn

from ..model._cache_compat import cache_kwarg_name, extract_output_cache
from ..model.introspect import find_lm_head, find_text_model
from ..model.offload import FP8KVCacheQuantizer, QuantizedFP8KV
from .bptt import _detach_nested, _ReplayKVCacheProxy
from .loss import lm_loss


@dataclass
class LongCalibConfig:
    """Configuration for Gate 2a Long-Prefix (`no_grad`) + Short-Suffix (`grad`) calibration."""

    enabled: bool = True
    prefix_len: int = 524_288
    max_prefix_len: int = 1_015_808
    suffix_len: int = 32_768
    chunk_size: int = 4_096
    fp8_kv_cache: bool = False
    return_logits: bool = False
    prefix_lsa_mode: str = "sparse"
    calib_every_steps: int = 10
    sparse_prefix_after_step: int = 100

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> LongCalibConfig:
        raw = dict(d or {})
        known = {f.name for f in fields(cls)}
        bad = set(raw) - known
        if bad:
            raise ValueError(f"unknown long_calib.* keys {sorted(bad)}; valid: {sorted(known)}")
        return cls(**raw)

    def validate(self) -> None:
        if self.chunk_size <= 0:
            raise ValueError(f"long_calib.chunk_size must be > 0, got {self.chunk_size}")
        if self.suffix_len < 2:
            raise ValueError(f"long_calib.suffix_len must be >= 2, got {self.suffix_len}")
        if self.calib_every_steps < 1:
            raise ValueError(f"long_calib.calib_every_steps must be >= 1, got {self.calib_every_steps}")
        if self.sparse_prefix_after_step < 0:
            raise ValueError(
                f"long_calib.sparse_prefix_after_step must be >= 0, got {self.sparse_prefix_after_step}"
            )
        if self.prefix_lsa_mode not in ("off", "dense", "distill", "sparse", "hybrid"):
            raise ValueError(
                f"long_calib.prefix_lsa_mode must be off|dense|distill|sparse|hybrid, got {self.prefix_lsa_mode!r}"
            )


@dataclass
class CalibrationStepResult:
    """Result of a single Long-Prefix (`no_grad`) + Short-Suffix (`grad`) calibration step."""

    loss: torch.Tensor
    task_loss: float
    lsa_aux_loss: float
    suffix_logits: torch.Tensor | None = None
    prefix_len: int = 0
    suffix_len: int = 0
    num_prefix_chunks: int = 0
    grad_norms: dict[str, float] = field(default_factory=dict)
    prefix_lsa_mode_used: str | None = None


LongCalibStepResult = CalibrationStepResult


def _quantize_fp8_tensor_lean(
    x: torch.Tensor, quantizer: FP8KVCacheQuantizer, chunk_tokens: int = 65536
) -> QuantizedFP8KV:
    """Bit-exact chunked `FP8KVCacheQuantizer.quantize(x)` with zero full-length fp32 temporary."""
    if not quantizer.supports_fp8(x.device) or quantizer.per_token or x.ndim < 3 or x.shape[-2] <= chunk_tokens:
        return quantizer.quantize(x)
    reduce_dims = tuple(range(2, x.ndim))
    amax = x.abs().amax(dim=reduce_dims, keepdim=True).float()
    scale = (amax / quantizer.FP8_MAX).clamp(min=1e-8)
    q_fp8 = torch.empty(x.shape, device=x.device, dtype=torch.float8_e4m3fn)
    t_len = int(x.shape[-2])
    for t0 in range(0, t_len, chunk_tokens):
        t1 = min(t_len, t0 + chunk_tokens)
        q_fp8[..., t0:t1, :] = (x[..., t0:t1, :].float() / scale).clamp(
            -quantizer.FP8_MAX, quantizer.FP8_MAX
        ).to(torch.float8_e4m3fn)
    return QuantizedFP8KV(q_fp8, scale.to(torch.float32), x.dtype, is_fp8_native=True)


def _extract_fp8_prefix_kv_(
    cache: Any, quantizer: FP8KVCacheQuantizer
) -> dict[int, tuple[QuantizedFP8KV, QuantizedFP8KV]]:
    """Quantize populated full-attention prefix KV caches into true 1-byte FP8 E4M3 and release bf16 copies."""
    out: dict[int, tuple[QuantizedFP8KV, QuantizedFP8KV]] = {}
    kc = getattr(cache, "key_cache", None)
    vc = getattr(cache, "value_cache", None)
    if isinstance(kc, list) and isinstance(vc, list):
        for i in range(min(len(kc), len(vc))):
            k_t, v_t = kc[i], vc[i]
            if (
                isinstance(k_t, torch.Tensor)
                and isinstance(v_t, torch.Tensor)
                and k_t.numel() > 0
                and quantizer.supports_fp8(k_t.device)
            ):
                out[i] = (
                    _quantize_fp8_tensor_lean(k_t, quantizer),
                    _quantize_fp8_tensor_lean(v_t, quantizer),
                )
                kc[i] = k_t.new_empty(0)
                vc[i] = v_t.new_empty(0)
        return out
    layers = getattr(cache, "layers", None)
    if isinstance(layers, list):
        for i, lyr in enumerate(layers):
            k_t = getattr(lyr, "keys", None)
            v_t = getattr(lyr, "values", None)
            if (
                isinstance(k_t, torch.Tensor)
                and isinstance(v_t, torch.Tensor)
                and k_t.numel() > 0
                and quantizer.supports_fp8(k_t.device)
            ):
                out[i] = (
                    _quantize_fp8_tensor_lean(k_t, quantizer),
                    _quantize_fp8_tensor_lean(v_t, quantizer),
                )
                lyr.keys = k_t.new_empty(0)
                lyr.values = v_t.new_empty(0)
    return out


def _quantize_prefix_cache_fp8_(cache: Any, quantizer: FP8KVCacheQuantizer) -> None:
    """Simulate/apply Tier-1 `FP8 E4M3` precision on populated prefix full-attention KV caches.

    Supports both legacy `key_cache`/`value_cache` lists (`HSSDHybridCache`, `ToyHybridCache`,
    HF 4.x `DynamicCache`) and `transformers >= 5.x` `cache.layers[i].keys`/`values` layers.
    """
    kc = getattr(cache, "key_cache", None)
    vc = getattr(cache, "value_cache", None)
    if isinstance(kc, list) and isinstance(vc, list):
        for i in range(min(len(kc), len(vc))):
            if isinstance(kc[i], torch.Tensor) and kc[i].numel() > 0:
                kc[i] = quantizer.dequantize(quantizer.quantize(kc[i]), dtype=kc[i].dtype)
            if isinstance(vc[i], torch.Tensor) and vc[i].numel() > 0:
                vc[i] = quantizer.dequantize(quantizer.quantize(vc[i]), dtype=vc[i].dtype)
        return
    layers = getattr(cache, "layers", None)
    if isinstance(layers, list):
        for lyr in layers:
            k_t = getattr(lyr, "keys", None)
            v_t = getattr(lyr, "values", None)
            if isinstance(k_t, torch.Tensor) and k_t.numel() > 0:
                lyr.keys = quantizer.dequantize(quantizer.quantize(k_t), dtype=k_t.dtype)
            if isinstance(v_t, torch.Tensor) and v_t.numel() > 0:
                lyr.values = quantizer.dequantize(quantizer.quantize(v_t), dtype=v_t.dtype)


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


def _chunked_linear_ce_with_bias(
    hidden: torch.Tensor,
    lm_head: nn.Module,
    targets: torch.Tensor,
    *,
    chunk_size: int = 4096,
    ignore_index: int = -100,
) -> torch.Tensor:
    """Fallback chunked CE when a synthetic `lm_head` carries an explicit bias vector."""
    h_shift = hidden[:, :-1].reshape(-1, hidden.shape[-1])
    y_shift = targets[:, 1:].reshape(-1)
    valid = y_shift != ignore_index
    n_tok = valid.sum().clamp_min(1)
    acc_dt = torch.float64 if hidden.dtype == torch.float64 else torch.float32
    total = torch.zeros((), dtype=acc_dt, device=hidden.device)
    for s0 in range(0, h_shift.shape[0], chunk_size):
        s1 = min(s0 + chunk_size, h_shift.shape[0])
        logits_c = lm_head(h_shift[s0:s1]).to(acc_dt)
        total = total + F.cross_entropy(logits_c, y_shift[s0:s1], ignore_index=ignore_index, reduction="sum")
    return total / n_tok.to(total.dtype)


def _resolve_backbone_and_head(root: nn.Module) -> tuple[nn.Module | None, nn.Module | None]:
    """Unwrap `(text_model, lm_head)` while preserving compatibility with custom/toy test models."""
    direct_head = getattr(root, "lm_head", None)
    direct_tm = getattr(root, "model", None)
    if isinstance(direct_head, nn.Module) and isinstance(direct_tm, nn.Module) and direct_tm is not root:
        return direct_tm, direct_head

    tm: nn.Module | None = None
    try:
        tm = find_text_model(root)
    except Exception:
        tm = None

    vocab_sz = getattr(getattr(tm, "config", None), "vocab_size", None) if tm is not None else None
    head = direct_head if isinstance(direct_head, nn.Module) else find_lm_head(root, vocab_sz)
    return tm, head


def _can_bypass_hf_causal_mask(tm: nn.Module | None, lsa_manager: Any | None) -> bool:
    """True when `tm` accepts a prebuilt dict `attention_mask` and `qloop_lsa` handles lower-right causal."""
    if tm is None:
        return False
    cfg = getattr(tm, "config", None)
    if cfg is None or str(getattr(cfg, "model_type", "")).startswith("toy"):
        return False
    if getattr(cfg, "layer_types", None) is None:
        return False
    return lsa_manager is not None or str(getattr(cfg, "_attn_implementation", "")) == "qloop_lsa"


@contextlib.contextmanager
def _prefix_lsa_mode_ctx(lsa_manager: Any, target_mode: str | None) -> Iterator[None]:
    if lsa_manager is None or target_mode is None:
        yield
        return
    temp_fn = getattr(lsa_manager, "temporary_mode", None)
    if callable(temp_fn):
        with temp_fn(target_mode):
            yield
        return
    prev_mode = getattr(getattr(lsa_manager, "cfg", None), "mode", None)
    set_fn = getattr(lsa_manager, "set_mode", None)
    if callable(set_fn) and prev_mode is not None:
        set_fn(target_mode)
        try:
            yield
        finally:
            set_fn(prev_mode)
        return
    yield


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
    return_logits: bool | None = None,
    lsa_manager: Any | None = None,
    global_step: int = 0,
    cfg: LongCalibConfig | None = None,
    prefix_lsa_mode: str | None = None,
    sparse_prefix_after_step: int | None = None,
    ce_chunk_size: int = 4096,
) -> CalibrationStepResult:
    """Stream `[0 : prefix_len]` under `no_grad`, then run `[prefix_len : T]` with `grad` & backward."""
    eff_fp8 = bool(fp8_kv_cache) or (bool(cfg.fp8_kv_cache) if cfg is not None else False)
    if eff_fp8 and fp8_quantizer is None:
        fp8_quantizer = FP8KVCacheQuantizer()

    if hasattr(model, "named_trainable"):
        if lsa_manager is None:
            lsa_manager = getattr(model, "lsa", None)
        root = getattr(model, "model", model)
    else:
        root = model
        if lsa_manager is None:
            lsa_manager = getattr(root, "lsa_manager", None)

    if input_ids.ndim != 2:
        raise ValueError(f"input_ids must have shape [B, T], got {tuple(input_ids.shape)}")
    total_len = input_ids.shape[1]
    if not 0 <= prefix_len < total_len:
        raise ValueError(f"need 0 <= prefix_len < total_len ({total_len}), got prefix_len={prefix_len}")
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be > 0, got {chunk_size}")
    if ce_chunk_size <= 0:
        raise ValueError(f"ce_chunk_size must be > 0, got {ce_chunk_size}")

    tm, lm_head = _resolve_backbone_and_head(root)

    # Clear stale LSA / HSSD aux buffers prior to streaming
    scan_root = tm if tm is not None else root
    for mod in scan_root.modules():
        lsa_ix = getattr(mod, "lsa_indexer", None)
        if lsa_ix is not None:
            lsa_ix.pop_losses()
        hssd_ad = getattr(mod, "hssd_adapter", None)
        if hssd_ad is not None and callable(getattr(hssd_ad, "clear_aux", None)):
            hssd_ad.clear_aux()

    eff_prefix_mode = (
        prefix_lsa_mode
        if prefix_lsa_mode is not None
        else (cfg.prefix_lsa_mode if cfg is not None else "sparse")
    )
    eff_after_step = (
        sparse_prefix_after_step
        if sparse_prefix_after_step is not None
        else (cfg.sparse_prefix_after_step if cfg is not None else 100)
    )
    use_sparse_prefix = lsa_manager is not None and (
        int(global_step) >= int(eff_after_step) or bool(getattr(lsa_manager, "recall_ready", False))
    )
    prefix_mode_used: str | None = None

    # Phase 1: Stream `[0 : prefix_len]` in `chunk_size` chunks under `torch.no_grad()`
    tm_cache_kw = cache_kwarg_name(tm) if tm is not None else "past_key_values"
    root_cache_kw = cache_kwarg_name(root)
    bypass_mask = _can_bypass_hf_causal_mask(tm, lsa_manager)
    no_mask_dict: dict[str, None] = {"full_attention": None, "linear_attention": None}
    cache: Any = None
    n_prefix_chunks = 0
    if prefix_len > 0:
        with _prefix_lsa_mode_ctx(lsa_manager, eff_prefix_mode if use_sparse_prefix else None):
            if lsa_manager is not None:
                prefix_mode_used = getattr(getattr(lsa_manager, "cfg", None), "mode", None)
            with torch.no_grad():
                for s0 in range(0, prefix_len, chunk_size):
                    s1 = min(prefix_len, s0 + chunk_size)
                    p_chunk = input_ids[:, s0:s1]
                    if tm is not None and tm is not root:
                        tm_kw: dict[str, Any] = {
                            "input_ids": p_chunk,
                            tm_cache_kw: cache,
                            "use_cache": True,
                        }
                        if bypass_mask and cache is not None:
                            tm_kw["attention_mask"] = no_mask_dict
                        p_out = tm(**tm_kw)
                    else:
                        p_out = root(p_chunk, **{root_cache_kw: cache, "use_cache": True})
                    cache = extract_output_cache(p_out)
                    n_prefix_chunks += 1
        fp8_prefix_kv: dict[int, tuple[QuantizedFP8KV, QuantizedFP8KV]] = {}
        if cache is not None:
            _detach_nested(cache)
            if fp8_quantizer is not None:
                with torch.no_grad():
                    if hasattr(cache, "update"):
                        fp8_prefix_kv = _extract_fp8_prefix_kv_(cache, fp8_quantizer)
                    if not fp8_prefix_kv:
                        _quantize_prefix_cache_fp8_(cache, fp8_quantizer)
    else:
        fp8_prefix_kv = {}

    # Clear any non-differentiable LSA / HSSD bookkeeping before differentiable suffix
    for mod in scan_root.modules():
        lsa_ix = getattr(mod, "lsa_indexer", None)
        if lsa_ix is not None:
            lsa_ix.pop_losses()
        hssd_ad = getattr(mod, "hssd_adapter", None)
        if hssd_ad is not None and callable(getattr(hssd_ad, "clear_aux", None)):
            hssd_ad.clear_aux()

    # Phase 2: Run `[prefix_len : total_len]` WITH gradients enabled to pre-head hidden states (`D2`)
    suffix_ids = input_ids[:, prefix_len:]
    suffix_labels = labels[:, prefix_len:] if labels is not None else None
    suffix_targets = suffix_labels if suffix_labels is not None else suffix_ids
    suffix_cache = (
        _ReplayKVCacheProxy(cache, readonly_base=True, fp8_prefix_kv=fp8_prefix_kv)
        if (cache is not None and hasattr(cache, "update"))
        else cache
    )
    fp8_prefix_kv.clear()

    can_pre_head = (
        tm is not None
        and tm is not root
        and lm_head is not None
        and isinstance(getattr(lm_head, "weight", None), torch.Tensor)
    )

    suffix_logits_out: torch.Tensor | None = None
    try:
        if can_pre_head:
            assert tm is not None and lm_head is not None
            suf_kw: dict[str, Any] = {
                "input_ids": suffix_ids,
                tm_cache_kw: suffix_cache,
                "use_cache": True,
            }
            if bypass_mask and suffix_cache is not None:
                suf_kw["attention_mask"] = no_mask_dict
            tm_out = tm(**suf_kw)
            suffix_hidden = tm_out.last_hidden_state if hasattr(tm_out, "last_hidden_state") else tm_out[0]
            if loss_fn is None:
                if suffix_hidden.shape[1] < 2:
                    raise ValueError("default next-token CE requires suffix_len >= 2 when loss_fn is None")
                if getattr(lm_head, "bias", None) is None:
                    task_loss, _ = lm_loss(
                        suffix_hidden,
                        lm_head.weight,
                        suffix_targets,
                        impl="chunked",
                        chunk_size=ce_chunk_size,
                        shift=True,
                        ignore_index=-100,
                    )
                else:
                    task_loss = _chunked_linear_ce_with_bias(
                        suffix_hidden,
                        lm_head,
                        suffix_targets,
                        chunk_size=ce_chunk_size,
                        ignore_index=-100,
                    )
                want_logits = (
                    bool(return_logits)
                    if return_logits is not None
                    else (bool(cfg.return_logits) if cfg is not None else False)
                )
                if want_logits:
                    with torch.no_grad():
                        suffix_logits_out = lm_head(suffix_hidden.detach())
            else:
                raw_logits = lm_head(suffix_hidden)
                task_loss = _invoke_calib_loss(loss_fn, raw_logits, suffix_ids, suffix_labels)
                want_logits = (
                    bool(return_logits)
                    if return_logits is not None
                    else (bool(cfg.return_logits) if cfg is not None else True)
                )
                if want_logits:
                    suffix_logits_out = raw_logits.detach()
        else:
            suffix_out = root(suffix_ids, **{root_cache_kw: suffix_cache, "use_cache": True})
            raw_logits = suffix_out.logits if hasattr(suffix_out, "logits") else suffix_out
            task_loss = _invoke_calib_loss(loss_fn, raw_logits, suffix_ids, suffix_labels)
            want_logits = (
                bool(return_logits)
                if return_logits is not None
                else (bool(cfg.return_logits) if cfg is not None else (loss_fn is not None))
            )
            if want_logits:
                suffix_logits_out = raw_logits.detach()

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
        lsa_losses.clear()

        if tm is not None:
            layers = getattr(tm, "layers", None)
            ctl = getattr(layers[0], "_ctl", None) if (layers is not None and len(layers) > 0) else None
            if ctl is not None:
                l_aux = ctl.aux_loss()
                if l_aux is not None and loop_aux_weight > 0:
                    total_loss = total_loss + float(loop_aux_weight) * l_aux.to(total_loss.dtype)
                del l_aux

        # Drop unconsumed HSSD write-budget aux terms BEFORE backward: retaining live `bud_loss`
        # tensors on `adapter._pending_aux` across `total_loss.backward()` pins every linear layer's
        # checkpointed HSSD segment frames (~4.8 GiB/layer at 32K suffix) until backward finishes!
        for mod in scan_root.modules():
            hssd_ad = getattr(mod, "hssd_adapter", None)
            if hssd_ad is not None and callable(getattr(hssd_ad, "clear_aux", None)):
                hssd_ad.clear_aux()

        if backward and total_loss.requires_grad:
            total_loss.backward()
    finally:
        # Release long-prefix caches, loop stashes, and LSA bucket caches so 512K-1M calibration
        # leaves 0 GiB pinned in GPU HBM across subsequent 64K training steps!
        if isinstance(suffix_cache, _ReplayKVCacheProxy):
            suffix_cache.reset_segment()
        for mod in scan_root.modules():
            lsa_ix = getattr(mod, "lsa_indexer", None)
            if lsa_ix is not None:
                lsa_ix.pop_losses()
                lsa_cache = getattr(lsa_ix, "_cache", None)
                if isinstance(lsa_cache, dict):
                    lsa_cache.clear()
            hssd_ad = getattr(mod, "hssd_adapter", None)
            if hssd_ad is not None and callable(getattr(hssd_ad, "clear_aux", None)):
                hssd_ad.clear_aux()
        if tm is not None:
            layers_cl = getattr(tm, "layers", None)
            ctl_cl = (
                getattr(layers_cl[0], "_ctl", None)
                if (layers_cl is not None and len(layers_cl) > 0)
                else None
            )
            if ctl_cl is not None and hasattr(ctl_cl, "ctx"):
                ctx_cl = ctl_cl.ctx
                ctx_cl.anchor = None
                for _attr in (
                    "start",
                    "kv_stash",
                    "pkv_stash",
                    "keep",
                    "coef",
                    "aux",
                    "div_terms",
                    "routing",
                    "router_probs",
                ):
                    _box = getattr(ctx_cl, _attr, None)
                    if isinstance(_box, (dict, list)):
                        _box.clear()

    grad_norms: dict[str, float] = {}
    for name, p in root.named_parameters():
        if p.requires_grad and p.grad is not None:
            grad_norms[name] = float(p.grad.detach().float().norm())

    return CalibrationStepResult(
        loss=total_loss.detach(),
        task_loss=float(task_loss.detach()),
        lsa_aux_loss=lsa_val,
        suffix_logits=suffix_logits_out,
        prefix_len=int(prefix_len),
        suffix_len=int(total_len - prefix_len),
        num_prefix_chunks=n_prefix_chunks,
        grad_norms=grad_norms,
        prefix_lsa_mode_used=prefix_mode_used,
    )
