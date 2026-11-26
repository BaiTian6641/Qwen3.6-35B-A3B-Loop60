"""Shipped Optimized On-Weight Inference Engine for Loop-HSSD v3 / Loop-GDN2 Models.

Supports both `ToyForCausalLM` and `Qwen3.6-35B-A6B-LoopHSSD` Whole Model Packs with:
  * Dynamic test-time pass depth `depth(r: int)` (`r in {1, 2, 3}` -> `40L / 64L / 88L`)
    without reloading weights (usable both as `with engine.depth(r):` and `engine.depth(r)`),
  * **Chunked Prefill** (`prefill_chunk_size = 4096`): streams arbitrarily long prompts
    (`128K - 1M` tokens) chunk-by-chunk through hybrid KV + recurrent states, materializing
    `lm_head` logits **ONLY on the final token** of the prompt,
  * **Quantized `1.0x` SharedKV Cache**: `"native"`, `"fp8"` (`float8_e4m3fn` + `fp16` per-token-head
    scale, ~50% resident full-attn KV VRAM savings), and `"hadamard_int4"` (Block-64 orthogonal
    Walsh-Hadamard packed `uint8` nibbles + `fp16` block scale, ~73% resident full-attn KV VRAM
    savings), while preserving `1.0x` SharedKV across passes `r >= 2`. Note: standard dense
    `update()` currently materializes full dequantized K/V tensors for SDPA compatibility; sparse
    consumers can use `gather_kv(layer_idx, token_indices)` to dequantize only selected tokens,
  * **Causal MoR Early Exit** (`mor_early_exit=True`) & optional **`torch.compile`** decode step
    acceleration (`compile_decode=True`),
  * Live hardware/routing telemetry (`GenerationMetrics`) and `benchmark(...)` across depths.
"""
from __future__ import annotations

import math
import os
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F
from torch import nn

from ..debug import require
from ..model._cache_compat import cache_kwarg_name, extract_output_cache
from ..model.experts import physical_layers
from ..model.introspect import find_lm_head, find_text_model
from ..model.loop import LoopController
from ..model.lsa import LSAManager
from ..model.toy import ToyHybridCache
from ..train.stack import Stack
from .hf_custom_modeling import Qwen36LoopHSSDForCausalLM, _DepthContext

if TYPE_CHECKING:
    from ..export_pack import LoadedModelPack

try:
    from transformers.cache_utils import Cache as _HFCacheBase  # type: ignore[import-not-found]
except ImportError:
    _HFCacheBase = object  # type: ignore[assignment,misc]

try:
    from transformers.cache_utils import CacheLayerMixin as _HFCacheLayerBase  # type: ignore[import-not-found]
except ImportError:
    _HFCacheLayerBase = object  # type: ignore[assignment,misc]


# ============================================================================ configs & telemetry

@dataclass
class InferenceConfig:
    """Configuration for `OptimizedLoopInferenceEngine`."""

    active_passes: int | None = None
    prefill_chunk_size: int = 4096
    kv_cache_quant: str = "native"            # "native" | "fp8" | "hadamard_int4"
    fp8_kv_cache: bool | None = None          # convenience alias (`True` -> `kv_cache_quant="fp8"`)
    compile_decode: bool = False
    compile_backend: str | None = None        # None -> "inductor" on CUDA, "aot_eager" on CPU/MPS
    compile_mode: str = "default"
    mor_early_exit: bool = False
    mor_confidence_threshold: float = 0.95
    temperature: float = 0.0
    top_k: int = 0
    top_p: float = 1.0
    max_new_tokens: int = 128
    stop_token_ids: tuple[int, ...] = ()
    seed: int | None = None

    def __post_init__(self) -> None:
        if self.fp8_kv_cache is True and self.kv_cache_quant == "native":
            self.kv_cache_quant = "fp8"
        elif self.fp8_kv_cache is False and self.kv_cache_quant == "fp8":
            self.kv_cache_quant = "native"
        elif self.fp8_kv_cache is None:
            self.fp8_kv_cache = self.kv_cache_quant == "fp8"

    def validate(self) -> None:
        require(int(self.prefill_chunk_size) >= 1, "prefill_chunk_size must be >= 1")
        valid_q = {"native", "bf16", "fp16", "fp32", "fp8", "hadamard_int4"}
        require(
            self.kv_cache_quant in valid_q,
            f"kv_cache_quant must be one of {sorted(valid_q)}, got {self.kv_cache_quant!r}",
        )
        require(float(self.temperature) >= 0.0, "temperature must be >= 0")
        require(0.0 < float(self.top_p) <= 1.0, "top_p must be in (0, 1]")
        require(int(self.top_k) >= 0, "top_k must be >= 0")


@dataclass
class GenerationMetrics:
    """Live throughput, memory, depth & MoE routing telemetry for a generation run."""

    prompt_tokens: int = 0
    generated_tokens: int = 0
    prefill_ms: float = 0.0
    decode_ms: float = 0.0
    total_ms: float = 0.0
    prefill_tok_s: float = 0.0
    decode_tok_s: float = 0.0
    peak_vram_mb: float = 0.0
    kv_cache_mb: float = 0.0
    recurrent_state_mb: float = 0.0
    active_passes: int = 1
    mean_active_passes: float = 1.0
    effective_layers: int = 0
    unique_experts_per_tok: float = 0.0
    prefill_chunks: int = 1
    kv_cache_quant: str = "native"
    compiled_decode: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def __getitem__(self, key: str) -> Any:
        if hasattr(self, key):
            return getattr(self, key)
        raise KeyError(key)


@dataclass
class GenerationResult:
    """Output of `OptimizedLoopInferenceEngine.generate`, supporting attribute and dict access."""

    text: str
    token_ids: list[int]
    prompt_token_ids: list[int] = field(default_factory=list)
    finish_reason: str = "length"
    stop_token_id: int | None = None
    logits_last: torch.Tensor | None = None
    metrics: GenerationMetrics = field(default_factory=GenerationMetrics)

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "token_ids": list(self.token_ids),
            "prompt_token_ids": list(self.prompt_token_ids),
            "finish_reason": self.finish_reason,
            "stop_token_id": self.stop_token_id,
            "metrics": self.metrics.to_dict(),
        }

    def keys(self) -> tuple[str, ...]:
        return ("text", "token_ids", "prompt_token_ids", "finish_reason", "stop_token_id", "metrics")

    def __getitem__(self, key: str) -> Any:
        if hasattr(self, key):
            return getattr(self, key)
        if hasattr(self.metrics, key):
            return getattr(self.metrics, key)
        raise KeyError(key)


# ============================================================================ FP8 & Block-64 Hadamard INT4 KV cache

_FP8_DTYPE = getattr(torch, "float8_e4m3fn", torch.int8)
_FP8_MAX = 448.0 if hasattr(torch, "float8_e4m3fn") else 127.0


def _largest_pow2_divisor_le_64(dim: int) -> int:
    b = 1
    while b * 2 <= 64 and dim % (b * 2) == 0:
        b *= 2
    return b


def _fwht_pow2(x: torch.Tensor) -> torch.Tensor:
    """Orthonormal Sylvester Fast Walsh-Hadamard Transform along last dim (power-of-2 length)."""
    n = x.shape[-1]
    if n <= 1:
        return x
    h = x
    step = 1
    while step < n:
        h = h.reshape(*x.shape[:-1], n // (2 * step), 2, step)
        a = h[..., 0, :]
        b = h[..., 1, :]
        h = torch.stack([a + b, a - b], dim=-2)
        step *= 2
    return h.reshape(x.shape) * (1.0 / math.sqrt(float(n)))


class _FP8Slot:
    """FP8 (`float8_e4m3fn`) KV slot with geometric buffer pre-allocation and low-VRAM BF16 dequant."""

    __slots__ = ("_len", "_q_buf", "_s_buf")

    def __init__(self, q_data: torch.Tensor, scale: torch.Tensor, *, length: int | None = None) -> None:
        self._q_buf: torch.Tensor = q_data   # [B, G, cap, D] in float8_e4m3fn
        self._s_buf: torch.Tensor = scale    # [B, G, cap, 1] in float16
        self._len: int = int(q_data.shape[-2]) if length is None else int(length)

    @property
    def q_data(self) -> torch.Tensor:
        return self._q_buf[..., : self._len, :]

    @property
    def scale(self) -> torch.Tensor:
        return self._s_buf[..., : self._len, :]

    @property
    def capacity(self) -> int:
        return int(self._q_buf.shape[-2])

    @staticmethod
    def _quantize_chunk(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        xf = x.float()
        amax = xf.abs().amax(dim=-1, keepdim=True).clamp_min(1e-12)
        sc = (amax / _FP8_MAX).to(torch.float16)
        sc_safe = sc.float().clamp_min(1e-12)
        q = (xf / sc_safe).clamp(-_FP8_MAX, _FP8_MAX)
        q_t = q.to(_FP8_DTYPE) if hasattr(torch, "float8_e4m3fn") else q.round().to(torch.int8)
        return q_t, sc

    @staticmethod
    def from_float(x: torch.Tensor) -> _FP8Slot:
        q_t, sc = _FP8Slot._quantize_chunk(x)
        return _FP8Slot(q_data=q_t, scale=sc)

    def _ensure_capacity(self, needed: int) -> None:
        cap = int(self._q_buf.shape[-2])
        if needed <= cap:
            return
        new_cap = max(int(needed), max(cap, 1) * 2)
        q_shape = (*self._q_buf.shape[:-2], new_cap, self._q_buf.shape[-1])
        s_shape = (*self._s_buf.shape[:-2], new_cap, self._s_buf.shape[-1])
        new_q = torch.empty(q_shape, dtype=self._q_buf.dtype, device=self._q_buf.device)
        new_s = torch.empty(s_shape, dtype=self._s_buf.dtype, device=self._s_buf.device)
        if self._len > 0:
            new_q[..., : self._len, :].copy_(self._q_buf[..., : self._len, :])
            new_s[..., : self._len, :].copy_(self._s_buf[..., : self._len, :])
        self._q_buf = new_q
        self._s_buf = new_s

    def append(self, x: torch.Tensor) -> None:
        t_add = int(x.shape[-2])
        if t_add <= 0:
            return
        q_t, sc = _FP8Slot._quantize_chunk(x)
        needed = self._len + t_add
        self._ensure_capacity(needed)
        self._q_buf[..., self._len : needed, :].copy_(q_t)
        self._s_buf[..., self._len : needed, :].copy_(sc)
        self._len = needed

    @staticmethod
    def _dequant_pair(q: torch.Tensor, s: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        if dtype == torch.bfloat16:
            # BF16 shares FP32's exact 8-bit exponent range (zero overflow) while avoiding 2x FP32 temporaries.
            out = q.to(torch.bfloat16)
            s_bf16 = s.to(torch.bfloat16)
            return out * s_bf16 if s_bf16.requires_grad else out.mul_(s_bf16)
        return (q.float() * s.float()).to(dtype)

    def dequantize(self, dtype: torch.dtype) -> torch.Tensor:
        return self._dequant_pair(self.q_data, self.scale, dtype)

    def dequantize_slice(self, token_indices: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        """Dequantize ONLY the requested `token_indices` along the sequence dimension (sparse read)."""
        q_live = self.q_data
        s_live = self.scale
        idx = token_indices.to(device=q_live.device, dtype=torch.long)
        q_sel = q_live.index_select(-2, idx)
        s_sel = s_live.index_select(-2, idx)
        return self._dequant_pair(q_sel, s_sel, dtype)

    @property
    def seq_len(self) -> int:
        return int(self._len)

    def nbytes(self) -> int:
        q = self.q_data
        s = self.scale
        return int(q.numel() * q.element_size() + s.numel() * s.element_size())

    def crop(self, max_length: int) -> None:
        m = max(0, int(max_length))
        self._len = min(self._len, m)

    def reorder_cache(self, beam_idx: torch.Tensor) -> None:
        idx = beam_idx.to(device=self._q_buf.device, dtype=torch.long)
        self._q_buf = self._q_buf[..., : self._len, :].index_select(0, idx).contiguous()
        self._s_buf = self._s_buf[..., : self._len, :].index_select(0, idx).contiguous()

    def batch_repeat_interleave(self, repeats: int) -> None:
        r = int(repeats)
        self._q_buf = self._q_buf[..., : self._len, :].repeat_interleave(r, dim=0).contiguous()
        self._s_buf = self._s_buf[..., : self._len, :].repeat_interleave(r, dim=0).contiguous()


class _HadamardINT4Slot:
    """Block-64 Walsh-Hadamard INT4 KV slot with geometric buffer pre-allocation."""

    __slots__ = ("_len", "_p_buf", "_s_buf", "block_sz", "orig_dim")

    def __init__(
        self,
        packed: torch.Tensor,
        scale: torch.Tensor,
        orig_dim: int,
        block_sz: int,
        *,
        length: int | None = None,
    ) -> None:
        self._p_buf: torch.Tensor = packed  # [B, G, cap, n_blocks, block_sz // 2] in uint8
        self._s_buf: torch.Tensor = scale   # [B, G, cap, n_blocks, 1] in float16
        self._len: int = int(packed.shape[2]) if length is None else int(length)
        self.orig_dim: int = int(orig_dim)
        self.block_sz: int = int(block_sz)

    @property
    def packed(self) -> torch.Tensor:
        return self._p_buf[:, :, : self._len, :, :]

    @property
    def scale(self) -> torch.Tensor:
        return self._s_buf[:, :, : self._len, :, :]

    @property
    def capacity(self) -> int:
        return int(self._p_buf.shape[2])

    @staticmethod
    def _quantize_chunk(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, int, int]:
        b, g, t, d = x.shape
        bsz = max(2, _largest_pow2_divisor_le_64(d if d % 2 == 0 else d + 1))
        pad = (bsz - (d % bsz)) % bsz
        xf = F.pad(x.float(), (0, pad)) if pad else x.float()
        d_pad = xf.shape[-1]
        nb = d_pad // bsz
        rot = _fwht_pow2(xf.view(b, g, t, nb, bsz))
        amax = rot.abs().amax(dim=-1, keepdim=True).clamp_min(1e-12)
        sc = (amax / 7.0).to(torch.float16)
        q_int = (rot / sc.float().clamp_min(1e-12)).round().clamp(-7.0, 7.0).to(torch.int16) + 8
        q_u8 = q_int.to(torch.uint8)
        hi = q_u8[..., 0::2]
        lo = q_u8[..., 1::2]
        pk = (hi << 4) | (lo & 0x0F)
        return pk, sc, d, bsz

    @staticmethod
    def from_float(x: torch.Tensor) -> _HadamardINT4Slot:
        pk, sc, d, bsz = _HadamardINT4Slot._quantize_chunk(x)
        return _HadamardINT4Slot(packed=pk, scale=sc, orig_dim=d, block_sz=bsz)

    def _ensure_capacity(self, needed: int) -> None:
        cap = int(self._p_buf.shape[2])
        if needed <= cap:
            return
        new_cap = max(int(needed), max(cap, 1) * 2)
        p_shape = (
            self._p_buf.shape[0],
            self._p_buf.shape[1],
            new_cap,
            self._p_buf.shape[3],
            self._p_buf.shape[4],
        )
        s_shape = (
            self._s_buf.shape[0],
            self._s_buf.shape[1],
            new_cap,
            self._s_buf.shape[3],
            self._s_buf.shape[4],
        )
        new_p = torch.empty(p_shape, dtype=self._p_buf.dtype, device=self._p_buf.device)
        new_s = torch.empty(s_shape, dtype=self._s_buf.dtype, device=self._s_buf.device)
        if self._len > 0:
            new_p[:, :, : self._len, :, :].copy_(self._p_buf[:, :, : self._len, :, :])
            new_s[:, :, : self._len, :, :].copy_(self._s_buf[:, :, : self._len, :, :])
        self._p_buf = new_p
        self._s_buf = new_s

    def append(self, x: torch.Tensor) -> None:
        t_add = int(x.shape[2])
        if t_add <= 0:
            return
        pk, sc, _, _ = _HadamardINT4Slot._quantize_chunk(x)
        needed = self._len + t_add
        self._ensure_capacity(needed)
        self._p_buf[:, :, self._len : needed, :, :].copy_(pk)
        self._s_buf[:, :, self._len : needed, :, :].copy_(sc)
        self._len = needed

    def _dequantize_parts(self, packed: torch.Tensor, scale: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        b, g, t, nb, half_bsz = packed.shape
        hi = ((packed >> 4) & 0x0F).to(torch.int16) - 8
        lo = (packed & 0x0F).to(torch.int16) - 8
        unpacked = torch.stack([hi, lo], dim=-1).reshape(b, g, t, nb, half_bsz * 2).float()
        rot = unpacked * scale.float()
        rec = _fwht_pow2(rot).reshape(b, g, t, nb * self.block_sz)
        if rec.shape[-1] != self.orig_dim:
            rec = rec[..., : self.orig_dim]
        return rec.to(dtype)

    def dequantize(self, dtype: torch.dtype) -> torch.Tensor:
        return self._dequantize_parts(self.packed, self.scale, dtype)

    def dequantize_slice(self, token_indices: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        """Dequantize ONLY the requested `token_indices` along the sequence dimension (sparse read)."""
        pk_live = self.packed
        sc_live = self.scale
        idx = token_indices.to(device=pk_live.device, dtype=torch.long)
        pk_sel = pk_live.index_select(2, idx)
        sc_sel = sc_live.index_select(2, idx)
        return self._dequantize_parts(pk_sel, sc_sel, dtype)

    @property
    def seq_len(self) -> int:
        return int(self._len)

    def nbytes(self) -> int:
        p = self.packed
        s = self.scale
        return int(p.numel() * p.element_size() + s.numel() * s.element_size())

    def crop(self, max_length: int) -> None:
        m = max(0, int(max_length))
        self._len = min(self._len, m)

    def reorder_cache(self, beam_idx: torch.Tensor) -> None:
        idx = beam_idx.to(device=self._p_buf.device, dtype=torch.long)
        self._p_buf = self._p_buf[:, :, : self._len, :, :].index_select(0, idx).contiguous()
        self._s_buf = self._s_buf[:, :, : self._len, :, :].index_select(0, idx).contiguous()

    def batch_repeat_interleave(self, repeats: int) -> None:
        r = int(repeats)
        self._p_buf = self._p_buf[:, :, : self._len, :, :].repeat_interleave(r, dim=0).contiguous()
        self._s_buf = self._s_buf[:, :, : self._len, :, :].repeat_interleave(r, dim=0).contiguous()


class _QuantizedHasPreviousState:
    """Dual-mode `has_previous_state`: truthy property (`HF 5.2-5.5`) and callable `(layer_idx)` (`HF 5.20+`)."""

    def __init__(self, cache: QuantizedHybridCache) -> None:
        self._cache = cache

    def __bool__(self) -> bool:
        return self()

    def __call__(self, layer_idx: int | None = None, state_idx: int = 0) -> bool:
        del state_idx
        c = self._cache
        if layer_idx is not None:
            idx = int(layer_idx)
            if not (0 <= idx < c.num_layers):
                return False
            return c.conv_states[idx] is not None or c.recurrent_states[idx] is not None
        last_lin = c.last_linear_layer
        if isinstance(last_lin, int) and 0 <= last_lin < c.num_layers:
            return c.conv_states[last_lin] is not None or c.recurrent_states[last_lin] is not None
        return any(x is not None for x in c.conv_states) or any(x is not None for x in c.recurrent_states)


class _SingleSlotStateList:
    """Mutable 1-element write-through proxy backed directly by `cache.<attr>[layer_idx]`."""

    __hash__ = None  # type: ignore[assignment]

    def __init__(self, cache: QuantizedHybridCache, attr: str, layer_idx: int) -> None:
        self._cache = cache
        self._attr = attr
        self._layer_idx = int(layer_idx)

    def _slot_list(self) -> list[Any]:
        return getattr(self._cache, self._attr)

    def __len__(self) -> int:
        return 1

    def __getitem__(self, idx: int) -> Any:
        if idx in (0, -1):
            return self._slot_list()[self._layer_idx]
        raise IndexError(f"single-slot state list index out of range: {idx}")

    def __setitem__(self, idx: int, value: Any) -> None:
        if idx not in (0, -1):
            raise IndexError(f"single-slot state list index out of range: {idx}")
        self._slot_list()[self._layer_idx] = value

    def __iter__(self) -> Any:
        yield self[0]

    def __repr__(self) -> str:
        return f"[{self[0]!r}]"

    @staticmethod
    def _item_eq(a: Any, b: Any) -> bool:
        if a is b:
            return True
        if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
            return a.shape == b.shape and a.dtype == b.dtype and a.device == b.device and bool(torch.equal(a, b))
        if isinstance(a, torch.Tensor) or isinstance(b, torch.Tensor):
            return False
        return bool(a == b)

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, _SingleSlotStateList):
            return self._item_eq(self[0], other[0])
        if isinstance(other, list):
            return len(other) == 1 and self._item_eq(self[0], other[0])
        return NotImplemented


class _LayerHasPreviousState:
    """Per-layer dual-mode `has_previous_state` supporting `bool(x)`, `x[0]`, `x.get(0)`, `0 in x`, and `x()`."""

    def __init__(self, cache: QuantizedHybridCache, layer_idx: int) -> None:
        self._cache = cache
        self._layer_idx = int(layer_idx)

    def __bool__(self) -> bool:
        return bool(self._cache.has_previous_state(self._layer_idx))

    def __call__(self, state_idx: int = 0) -> bool:
        return bool(self._cache.has_previous_state(self._layer_idx, state_idx))

    def __getitem__(self, state_idx: int) -> bool:
        idx = int(state_idx)
        if idx != 0:
            raise KeyError(state_idx)
        return bool(self._cache.has_previous_state(self._layer_idx, idx))

    def get(self, state_idx: int = 0, default: Any = False) -> Any:
        if int(state_idx) == 0:
            return bool(self._cache.has_previous_state(self._layer_idx, 0))
        return default

    def __contains__(self, state_idx: object) -> bool:
        return state_idx == 0


class _QuantizedCacheLayer(_HFCacheLayerBase):  # type: ignore[misc]
    """HF 5.24+ `DynamicLayer` / `LinearAttentionLayer`-compatible per-layer view into `QuantizedHybridCache`."""

    is_compileable = False
    is_sliding = False
    record_past = False

    def __init__(self, cache: QuantizedHybridCache, layer_idx: int, layer_type: str) -> None:
        # Bind backing cache attributes BEFORE super().__init__() so `CacheLayerMixin.__init__`'s
        # `self.keys = None; self.values = None; self.is_initialized = False` succeeds cleanly.
        self._cache = cache
        self._layer_idx = int(layer_idx)
        self.layer_type = str(layer_type)
        self.supports_early_init = self.layer_type == "full_attention"
        if _HFCacheLayerBase is not object:
            try:
                super().__init__()
            except (TypeError, ValueError):
                pass

    @property
    def is_croppable(self) -> bool:
        return self.layer_type == "full_attention"

    @property
    def is_initialized(self) -> bool:
        if self.layer_type == "full_attention":
            return self._cache._kv_seq_len(self._layer_idx) > 0
        return bool(self._cache.has_previous_state(self._layer_idx))

    @is_initialized.setter
    def is_initialized(self, _value: bool) -> None:
        return None

    def lazy_initialization(
        self,
        key_states: torch.Tensor | None = None,
        value_states: torch.Tensor | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        del args, kwargs
        if isinstance(key_states, torch.Tensor):
            self._cache._k_dtypes[self._layer_idx] = key_states.dtype
        if isinstance(value_states, torch.Tensor):
            self._cache._v_dtypes[self._layer_idx] = value_states.dtype

    def get_max_length(self) -> int:
        return -1

    @property
    def keys(self) -> torch.Tensor | None:
        c = self._cache
        idx = self._layer_idx
        if c.mode in ("fp8", "hadamard_int4"):
            sl = c._k_slots[idx]
            return None if sl is None else sl.dequantize(c._k_dtypes[idx])
        return c.key_cache[idx]

    @keys.setter
    def keys(self, val: torch.Tensor | None) -> None:
        c = self._cache
        idx = self._layer_idx
        if val is None:
            c._k_slots[idx] = None
            c.key_cache[idx] = None
            return
        c._k_dtypes[idx] = val.dtype
        if c.mode in ("fp8", "hadamard_int4"):
            c._k_slots[idx] = c._encode_slot(val)
        else:
            c.key_cache[idx] = c._encode_slot(val)

    @property
    def values(self) -> torch.Tensor | None:
        c = self._cache
        idx = self._layer_idx
        if c.mode in ("fp8", "hadamard_int4"):
            sl = c._v_slots[idx]
            return None if sl is None else sl.dequantize(c._v_dtypes[idx])
        return c.value_cache[idx]

    @values.setter
    def values(self, val: torch.Tensor | None) -> None:
        c = self._cache
        idx = self._layer_idx
        if val is None:
            c._v_slots[idx] = None
            c.value_cache[idx] = None
            return
        c._v_dtypes[idx] = val.dtype
        if c.mode in ("fp8", "hadamard_int4"):
            c._v_slots[idx] = c._encode_slot(val)
        else:
            c.value_cache[idx] = c._encode_slot(val)

    @property
    def conv_states(self) -> _SingleSlotStateList:
        return _SingleSlotStateList(self._cache, "conv_states", self._layer_idx)

    @property
    def recurrent_states(self) -> _SingleSlotStateList:
        return _SingleSlotStateList(self._cache, "recurrent_states", self._layer_idx)

    @property
    def has_previous_state(self) -> _LayerHasPreviousState:
        return _LayerHasPreviousState(self._cache, self._layer_idx)

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        cache_kwargs: dict[str, Any] | None = None,
        **extra_kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self._cache.update(
            key_states,
            value_states,
            self._layer_idx,
            cache_kwargs=cache_kwargs,
            **extra_kwargs,
        )

    def update_conv_state(
        self,
        conv_states: torch.Tensor,
        state_idx: int = 0,
        conv_kernel_size: int | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        return self._cache.update_conv_state(
            conv_states,
            self._layer_idx,
            state_idx=state_idx,
            conv_kernel_size=conv_kernel_size,
            **kwargs,
        )

    def get_conv_state(self, state_idx: int = 0) -> torch.Tensor | None:
        return self._cache.get_conv_state(self._layer_idx, state_idx=state_idx)

    def update_recurrent_state(
        self,
        recurrent_states: Any,
        state_idx: int = 0,
        **kwargs: Any,
    ) -> Any:
        return self._cache.update_recurrent_state(recurrent_states, self._layer_idx, state_idx=state_idx, **kwargs)

    def get_recurrent_state(self, state_idx: int = 0) -> Any:
        return self._cache.get_recurrent_state(self._layer_idx, state_idx=state_idx)

    def get_seq_length(self, **extra_kwargs: Any) -> int:
        return self._cache.get_seq_length(self._layer_idx, **extra_kwargs)

    def get_max_cache_shape(self) -> int | None:
        return self._cache.get_max_cache_shape(self._layer_idx)

    def get_mask_sizes(self, cache_position: torch.Tensor | int, **extra_kwargs: Any) -> tuple[int, int]:
        return self._cache.get_mask_sizes(cache_position, self._layer_idx, **extra_kwargs)

    def crop(self, max_length: int) -> None:
        self._cache._crop_layer(self._layer_idx, max_length)

    def reset(self) -> None:
        self._cache._reset_layer(self._layer_idx)

    def reorder_cache(self, beam_idx: torch.Tensor) -> None:
        self._cache._reorder_layer(self._layer_idx, beam_idx)

    def batch_select_indices(self, indices: torch.Tensor) -> None:
        self._cache._reorder_layer(self._layer_idx, indices)

    def batch_repeat_interleave(self, repeats: int) -> None:
        self._cache._repeat_layer(self._layer_idx, repeats)


class QuantizedHybridCache(_HFCacheBase):  # type: ignore[misc]
    """Hybrid KV + recurrent state cache with FP8 / Block-64 Hadamard INT4 full-attn compression.

    Compatible with both legacy Hugging Face `Cache` APIs (`transformers <= 5.23`) and modern
    `transformers >= 5.24` hybrid `Cache(layers=...)` contracts (`layers[idx]`, `conv_states`,
    `recurrent_states`, `has_previous_state`, `get_seq_length`, `get_max_cache_shape`,
    `get_mask_sizes`, `reorder_cache`, `crop`, `reset`, `batch_repeat_interleave`,
    `batch_select_indices`).

    Preserves `1.0x` SharedKV across Sandwich-Loop passes (`r >= 2` pass zero-length `k[..., :0, :]`
    via `SharedKVView` and read the pass-1 quantized slot directly with zero duplicate KV storage).

    Memory note:
      Resident KV storage is compressed in FP8 (`~50%` of BF16/FP16) or Block-64 Hadamard INT4
      (`~27%` of BF16/FP16). However, standard dense `update()` currently materializes the full
      dequantized `[B, G, T, D]` K/V tensors for the active layer on return so stock PyTorch/HF
      attention implementations work unmodified. Sparse consumers (e.g. D-LSA top-K read paths)
      can call `gather_kv(layer_idx, token_indices)` to dequantize ONLY selected token positions.
    """

    def __init__(self, num_layers: int, layer_types: Sequence[str], mode: str = "fp8") -> None:
        self.mode = mode
        self.num_layers = int(num_layers)
        ltypes = [str(x) for x in layer_types]
        if len(ltypes) < self.num_layers:
            ltypes.extend(["full_attention"] * (self.num_layers - len(ltypes)))
        self.layer_types = ltypes[: self.num_layers]
        self._has_full_attention = "full_attention" in self.layer_types
        self.first_full = self.layer_types.index("full_attention") if self._has_full_attention else 0
        lin_indices = [idx for idx, lt in enumerate(self.layer_types) if lt == "linear_attention"]
        self.last_linear_layer: int | None = lin_indices[-1] if lin_indices else None
        self._k_slots: list[Any] = [None] * self.num_layers
        self._v_slots: list[Any] = [None] * self.num_layers
        self._k_dtypes: list[torch.dtype] = [torch.float32] * self.num_layers
        self._v_dtypes: list[torch.dtype] = [torch.float32] * self.num_layers
        self.key_cache: list[torch.Tensor | None] = [None] * self.num_layers
        self.value_cache: list[torch.Tensor | None] = [None] * self.num_layers
        self.conv_states: list[torch.Tensor | None] = [None] * self.num_layers
        self.recurrent_states: list[Any] = [None] * self.num_layers

        layer_wrappers = [
            _QuantizedCacheLayer(self, i, self.layer_types[i]) for i in range(self.num_layers)
        ]
        if _HFCacheBase is not object:
            inited = False
            # Attempt modern transformers 5.24+ Cache(layers=...) signature first
            try:
                super().__init__(layers=layer_wrappers)
                inited = True
            except (TypeError, ValueError):
                pass
            if not inited:
                try:
                    super().__init__(layers=[])
                    inited = True
                except (TypeError, ValueError):
                    pass
            if not inited:
                try:
                    super().__init__()
                except (TypeError, ValueError):
                    pass
        if getattr(self, "layers", None) is not layer_wrappers:
            self.layers: list[_QuantizedCacheLayer] = layer_wrappers

    @property
    def is_compileable(self) -> bool:
        return False

    @property
    def is_sliding(self) -> list[bool]:
        return [False] * self.num_layers

    def is_recording_past(self, layer_idx: int = 0) -> bool:
        del layer_idx
        return False

    @property
    def has_previous_state(self) -> _QuantizedHasPreviousState:
        return _QuantizedHasPreviousState(self)

    @property
    def is_initialized(self) -> bool:
        if any(self._kv_seq_len(i) > 0 for i in range(self.num_layers)):
            return True
        return any(x is not None for x in self.conv_states) or any(x is not None for x in self.recurrent_states)

    @property
    def seen_tokens(self) -> int:
        return self.get_seq_length()

    def __len__(self) -> int:
        return self.num_layers

    def __getitem__(self, layer_idx: int) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        idx = int(layer_idx)
        return self.layers[idx].keys, self.layers[idx].values

    def _encode_slot(self, x: torch.Tensor) -> Any:
        if self.mode == "fp8":
            return _FP8Slot.from_float(x)
        if self.mode == "hadamard_int4":
            return _HadamardINT4Slot.from_float(x)
        if self.mode == "bf16":
            return x.to(torch.bfloat16)
        if self.mode == "fp16":
            return x.to(torch.float16)
        return x

    def _kv_seq_len(self, layer_idx: int) -> int:
        if not (0 <= layer_idx < self.num_layers):
            return 0
        if self.mode in ("fp8", "hadamard_int4"):
            sl = self._k_slots[layer_idx]
            return 0 if sl is None else int(sl.seq_len)
        kc = self.key_cache[layer_idx]
        return 0 if kc is None else int(kc.shape[-2])

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: dict[str, Any] | None = None,
        **extra_kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Append new KV tokens (when `T_new > 0`) and return full layer `(K, V)`.

        NOTE: For quantized modes (`"fp8"`, `"hadamard_int4"`), resident cache storage stays
        compressed while `update()` materializes the full `[B, G, T_total, D]` K/V tensors on
        return to satisfy standard dense/SDPA attention callers. Use `gather_kv()` for sparse
        selected-token dequantization without full-sequence materialization.
        """
        del cache_kwargs, extra_kwargs
        out_dt = key_states.dtype
        v_dt = value_states.dtype
        self._k_dtypes[layer_idx] = out_dt
        self._v_dtypes[layer_idx] = v_dt
        t_new = int(key_states.shape[-2])

        if self.mode in ("fp8", "hadamard_int4"):
            ks = self._k_slots[layer_idx]
            vs = self._v_slots[layer_idx]
            if t_new > 0:
                if ks is None:
                    ks = self._encode_slot(key_states)
                    vs = self._encode_slot(value_states)
                    self._k_slots[layer_idx] = ks
                    self._v_slots[layer_idx] = vs
                else:
                    ks.append(key_states)
                    vs.append(value_states)
            require(ks is not None and vs is not None, f"empty quantized KV slot at layer {layer_idx}")
            return ks.dequantize(out_dt), vs.dequantize(v_dt)

        # Native / explicit float dtype storage
        if t_new > 0:
            k_enc = self._encode_slot(key_states)
            v_enc = self._encode_slot(value_states)
            if self.key_cache[layer_idx] is None:
                self.key_cache[layer_idx] = k_enc
                self.value_cache[layer_idx] = v_enc
            else:
                self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], k_enc], dim=-2)
                self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], v_enc], dim=-2)
        kc = self.key_cache[layer_idx]
        vc = self.value_cache[layer_idx]
        require(kc is not None and vc is not None, f"empty KV slot at layer {layer_idx}")
        assert kc is not None and vc is not None
        return kc.to(out_dt), vc.to(v_dt)

    def gather_kv(
        self,
        layer_idx: int,
        token_indices: torch.Tensor,
        *,
        dtype: torch.dtype | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sparse-only read hook: gather & dequantize ONLY `token_indices` without full KV materialization."""
        idx = int(layer_idx)
        k_dt = dtype if dtype is not None else self._k_dtypes[idx]
        v_dt = dtype if dtype is not None else self._v_dtypes[idx]
        if self.mode in ("fp8", "hadamard_int4"):
            ks = self._k_slots[idx]
            vs = self._v_slots[idx]
            require(ks is not None and vs is not None, f"empty quantized KV slot at layer {idx}")
            return ks.dequantize_slice(token_indices, k_dt), vs.dequantize_slice(token_indices, v_dt)
        kc = self.key_cache[idx]
        vc = self.value_cache[idx]
        require(kc is not None and vc is not None, f"empty KV slot at layer {idx}")
        assert kc is not None and vc is not None
        t_idx = token_indices.to(device=kc.device, dtype=torch.long)
        return kc.index_select(-2, t_idx).to(k_dt), vc.index_select(-2, t_idx).to(v_dt)

    def update_conv_state(
        self,
        conv_states: torch.Tensor,
        layer_idx: int,
        state_idx: int = 0,
        conv_kernel_size: int | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Update short-conv state for a linear-attention layer (HF 5.20-5.24+ hybrid contract)."""
        del kwargs
        require(int(state_idx) == 0, f"QuantizedHybridCache only supports state_idx=0, got {state_idx}")
        idx = int(layer_idx)
        prev = self.conv_states[idx]
        if prev is None:
            full_conv = conv_states
            if conv_kernel_size is not None and full_conv.shape[-1] < int(conv_kernel_size):
                full_conv = F.pad(full_conv, (int(conv_kernel_size) - full_conv.shape[-1], 0), value=0)
        else:
            full_conv = torch.cat([prev, conv_states], dim=-1)
        keep_w = int(conv_kernel_size) if conv_kernel_size is not None else full_conv.shape[-1]
        self.conv_states[idx] = full_conv[..., -keep_w:].contiguous()
        return full_conv

    def get_conv_state(self, layer_idx: int, state_idx: int = 0) -> torch.Tensor | None:
        require(int(state_idx) == 0, f"QuantizedHybridCache only supports state_idx=0, got {state_idx}")
        idx = int(layer_idx)
        return self.conv_states[idx] if 0 <= idx < self.num_layers else None

    def update_recurrent_state(
        self,
        recurrent_states: Any,
        layer_idx: int,
        state_idx: int = 0,
        **kwargs: Any,
    ) -> Any:
        """Update DeltaNet / HSSD recurrent state for a linear-attention layer (HF 5.20-5.24+ hybrid contract)."""
        del kwargs
        require(int(state_idx) == 0, f"QuantizedHybridCache only supports state_idx=0, got {state_idx}")
        idx = int(layer_idx)
        self.recurrent_states[idx] = recurrent_states
        return recurrent_states

    def get_recurrent_state(self, layer_idx: int, state_idx: int = 0) -> Any:
        require(int(state_idx) == 0, f"QuantizedHybridCache only supports state_idx=0, got {state_idx}")
        idx = int(layer_idx)
        return self.recurrent_states[idx] if 0 <= idx < self.num_layers else None

    def get_seq_length(self, layer_idx: int | None = None, **extra_kwargs: Any) -> int:
        del extra_kwargs
        if layer_idx is None:
            if self._has_full_attention:
                seq_l = self._kv_seq_len(self.first_full)
                if seq_l > 0:
                    return seq_l
                for i, lt in enumerate(self.layer_types):
                    if lt == "full_attention":
                        sl = self._kv_seq_len(i)
                        if sl > 0:
                            return sl
            for rs in self.recurrent_states:
                sc = getattr(rs, "step_count", None)
                if isinstance(sc, torch.Tensor) and sc.numel() > 0:
                    return int(sc.max().item())
            return 0

        idx = int(layer_idx)
        if not (0 <= idx < self.num_layers):
            return 0
        sl = self._kv_seq_len(idx)
        if sl > 0:
            return sl
        if self.layer_types[idx] != "full_attention":
            if self._has_full_attention:
                return self._kv_seq_len(self.first_full)
            rs = self.recurrent_states[idx]
            sc = getattr(rs, "step_count", None)
            if isinstance(sc, torch.Tensor) and sc.numel() > 0:
                return int(sc.max().item())
        return 0

    def get_usable_length(self, new_seq_length: int, layer_idx: int | None = None) -> int:
        del new_seq_length
        return self.get_seq_length(layer_idx)

    def get_max_cache_shape(self, layer_idx: int | None = None) -> int | None:
        del layer_idx
        return None

    def get_mask_sizes(
        self,
        cache_position: torch.Tensor | int,
        layer_idx: int = 0,
        **extra_kwargs: Any,
    ) -> tuple[int, int]:
        """Return `(kv_length, kv_offset)` for HF `create_causal_mask` (`transformers 5.x`)."""
        del extra_kwargs
        if isinstance(cache_position, torch.Tensor):
            q_len = int(cache_position.shape[0]) if cache_position.ndim > 0 else 1
        else:
            q_len = int(cache_position)
        return self.get_seq_length(layer_idx) + q_len, 0

    def _crop_layer(self, layer_idx: int, max_length: int) -> None:
        m = max(0, int(max_length))
        if m == 0:
            self._reset_layer(layer_idx)
            return
        ks = self._k_slots[layer_idx]
        if ks is not None:
            ks.crop(m)
        vs = self._v_slots[layer_idx]
        if vs is not None:
            vs.crop(m)
        kt = self.key_cache[layer_idx]
        if isinstance(kt, torch.Tensor) and kt.shape[-2] > m:
            self.key_cache[layer_idx] = kt[..., :m, :].contiguous()
        vt = self.value_cache[layer_idx]
        if isinstance(vt, torch.Tensor) and vt.shape[-2] > m:
            self.value_cache[layer_idx] = vt[..., :m, :].contiguous()

    def crop(self, max_length: int) -> None:
        """Crop cached KV sequences to at most `max_length` tokens (supports negative relative offsets)."""
        cur = self.get_seq_length()
        target = cur - abs(int(max_length)) if int(max_length) < 0 else int(max_length)
        target = max(0, target)
        if target == 0:
            self.reset()
            return
        if cur <= target:
            return
        for i in range(self.num_layers):
            self._crop_layer(i, target)

    def _reorder_layer(self, i: int, beam_idx: torch.Tensor) -> None:
        ks = self._k_slots[i]
        if ks is not None:
            ks.reorder_cache(beam_idx)
        vs = self._v_slots[i]
        if vs is not None:
            vs.reorder_cache(beam_idx)
        kt = self.key_cache[i]
        if isinstance(kt, torch.Tensor):
            self.key_cache[i] = kt.index_select(0, beam_idx.to(device=kt.device, dtype=torch.long)).contiguous()
        vt = self.value_cache[i]
        if isinstance(vt, torch.Tensor):
            self.value_cache[i] = vt.index_select(0, beam_idx.to(device=vt.device, dtype=torch.long)).contiguous()
        cs = self.conv_states[i]
        if isinstance(cs, torch.Tensor):
            self.conv_states[i] = cs.index_select(0, beam_idx.to(device=cs.device, dtype=torch.long)).contiguous()
        rs = self.recurrent_states[i]
        if rs is not None:
            if hasattr(rs, "reorder_cache") and callable(rs.reorder_cache):
                new_rs = rs.reorder_cache(beam_idx)
                if new_rs is not None:
                    self.recurrent_states[i] = new_rs
            elif isinstance(rs, torch.Tensor):
                self.recurrent_states[i] = rs.index_select(
                    0, beam_idx.to(device=rs.device, dtype=torch.long)
                ).contiguous()
            else:
                self.recurrent_states[i] = torch.index_select(rs, 0, beam_idx)

    def reorder_cache(self, beam_idx: torch.LongTensor) -> None:
        """Reorder hybrid quantized KV slots, conv states, and recurrent (`HSSDCacheState`/Tensor) states."""
        for i in range(self.num_layers):
            self._reorder_layer(i, beam_idx)

    def batch_select_indices(self, indices: torch.Tensor) -> None:
        """Select a subset/permutation of batch entries across all hybrid cache layers (HF 5.x)."""
        self.reorder_cache(indices)  # type: ignore[arg-type]

    def _repeat_layer(self, i: int, repeats: int) -> None:
        r = int(repeats)
        ks = self._k_slots[i]
        if ks is not None:
            ks.batch_repeat_interleave(r)
        vs = self._v_slots[i]
        if vs is not None:
            vs.batch_repeat_interleave(r)
        kt = self.key_cache[i]
        if isinstance(kt, torch.Tensor):
            self.key_cache[i] = kt.repeat_interleave(r, dim=0).contiguous()
        vt = self.value_cache[i]
        if isinstance(vt, torch.Tensor):
            self.value_cache[i] = vt.repeat_interleave(r, dim=0).contiguous()
        cs = self.conv_states[i]
        if isinstance(cs, torch.Tensor):
            self.conv_states[i] = cs.repeat_interleave(r, dim=0).contiguous()
        rs = self.recurrent_states[i]
        if rs is not None:
            if hasattr(rs, "repeat_interleave") and callable(rs.repeat_interleave):
                self.recurrent_states[i] = rs.repeat_interleave(r, dim=0)
            elif isinstance(rs, torch.Tensor):
                self.recurrent_states[i] = rs.repeat_interleave(r, dim=0).contiguous()

    def batch_repeat_interleave(self, repeats: int) -> None:
        """Repeat batch entries `repeats` times across all hybrid cache layers (HF 5.x beam search)."""
        for i in range(self.num_layers):
            self._repeat_layer(i, repeats)

    def _reset_layer(self, i: int) -> None:
        self._k_slots[i] = None
        self._v_slots[i] = None
        self.key_cache[i] = None
        self.value_cache[i] = None
        self.conv_states[i] = None
        self.recurrent_states[i] = None

    def reset(self) -> None:
        """Clear all cached KV slots, conv states, and recurrent states across all layers."""
        for i in range(self.num_layers):
            self._reset_layer(i)

    def kv_bytes(self) -> int:
        seen: set[int] = set()
        if self.mode in ("fp8", "hadamard_int4"):
            return _tensor_tree_bytes((*self._k_slots, *self._v_slots), seen)
        return _tensor_tree_bytes((*self.key_cache, *self.value_cache), seen)

    def recurrent_bytes(self) -> int:
        seen: set[int] = set()
        return _tensor_tree_bytes((*self.conv_states, *self.recurrent_states), seen)


def _tensor_tree_bytes(obj: Any, seen_ptrs: set[int]) -> int:
    """Accumulate unique backing-storage bytes across nested tensors, sequences, dicts, and `HSSDCacheState`."""
    if obj is None:
        return 0
    if isinstance(obj, _FP8Slot):
        tot = 0
        for t in (obj.q_data, obj.scale):
            if t.numel() == 0:
                continue
            ptr = int(t.untyped_storage().data_ptr())
            if ptr not in seen_ptrs:
                seen_ptrs.add(ptr)
                tot += int(t.numel() * t.element_size())
        return tot
    if isinstance(obj, _HadamardINT4Slot):
        tot = 0
        for t in (obj.packed, obj.scale):
            if t.numel() == 0:
                continue
            ptr = int(t.untyped_storage().data_ptr())
            if ptr not in seen_ptrs:
                seen_ptrs.add(ptr)
                tot += int(t.numel() * t.element_size())
        return tot
    if isinstance(obj, torch.Tensor):
        if obj.numel() == 0:
            return 0
        st = obj.untyped_storage()
        ptr = int(st.data_ptr())
        if ptr in seen_ptrs:
            return 0
        seen_ptrs.add(ptr)
        return int(st.nbytes())
    if isinstance(obj, (list, tuple)):
        return sum(_tensor_tree_bytes(x, seen_ptrs) for x in obj)
    if isinstance(obj, dict):
        return sum(_tensor_tree_bytes(v, seen_ptrs) for v in obj.values())
    if isinstance(obj, nn.Module):
        return 0
    if hasattr(obj, "_iter_tensors") and callable(obj._iter_tensors):
        return sum(_tensor_tree_bytes(t, seen_ptrs) for t in obj._iter_tensors())
    if hasattr(obj, "__dataclass_fields__"):
        return sum(_tensor_tree_bytes(getattr(obj, f), seen_ptrs) for f in obj.__dataclass_fields__)
    if hasattr(obj, "__dict__"):
        return sum(_tensor_tree_bytes(v, seen_ptrs) for v in vars(obj).values())
    if hasattr(obj, "__slots__"):
        return sum(_tensor_tree_bytes(getattr(obj, s), seen_ptrs) for s in obj.__slots__ if hasattr(obj, s))
    return 0


def _measure_any_cache_bytes(cache: Any) -> tuple[int, int]:
    """Return `(kv_bytes, recurrent_bytes)` for `QuantizedHybridCache`, `ToyHybridCache`, or HF `Cache`."""
    if cache is None:
        return 0, 0
    if isinstance(cache, QuantizedHybridCache):
        return cache.kv_bytes(), cache.recurrent_bytes()

    kv_b = 0
    rec_b = 0
    seen_ptrs: set[int] = set()

    for attr in ("key_cache", "value_cache"):
        seq = getattr(cache, attr, None)
        if isinstance(seq, (list, tuple)):
            kv_b += _tensor_tree_bytes(seq, seen_ptrs)
    for attr in ("conv_states", "recurrent_states"):
        seq = getattr(cache, attr, None)
        if isinstance(seq, (list, tuple)):
            rec_b += _tensor_tree_bytes(seq, seen_ptrs)
    layers = getattr(cache, "layers", None)
    if kv_b == 0 and rec_b == 0 and isinstance(layers, (list, tuple, nn.ModuleList)):
        for lyr in layers:
            for attr in ("keys", "values"):
                val = getattr(lyr, attr, None)
                if val is not None:
                    kv_b += _tensor_tree_bytes(val, seen_ptrs)
            for attr in ("conv_states", "recurrent_states"):
                val = getattr(lyr, attr, None)
                if val is not None:
                    rec_b += _tensor_tree_bytes(val, seen_ptrs)
    return kv_b, rec_b


# ============================================================================ sampling helpers

def _sample_next_token(
    logits_last: torch.Tensor,
    *,
    temperature: float,
    top_k: int,
    top_p: float,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample `[B, 1]` token IDs from `[B, 1, V]` or `[B, V]` logits."""
    scores = logits_last[:, -1, :].float() if logits_last.ndim == 3 else logits_last.float()
    if temperature <= 0.0:
        return scores.argmax(dim=-1, keepdim=True)

    scores = scores / float(temperature)
    vocab = scores.shape[-1]
    if top_k > 0 and top_k < vocab:
        kth = torch.topk(scores, int(top_k), dim=-1).values[..., -1, None]
        scores = scores.masked_fill(scores < kth, float("-inf"))

    if top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(scores, descending=True, dim=-1)
        probs_sorted = F.softmax(sorted_logits, dim=-1)
        cumprobs = torch.cumsum(probs_sorted, dim=-1)
        drop_sorted = cumprobs - probs_sorted > float(top_p)
        drop_sorted[..., 0] = False
        sorted_logits = sorted_logits.masked_fill(drop_sorted, float("-inf"))
        scores = torch.empty_like(scores).scatter(-1, sorted_idx, sorted_logits)

    probs = F.softmax(scores, dim=-1)
    return torch.multinomial(probs, num_samples=1, generator=generator)


# ============================================================================ Inference Engine

class OptimizedLoopInferenceEngine:
    """Production on-weight inference engine for `Qwen3.6-35B-A6B-LoopHSSD` and `ToyForCausalLM`."""

    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any = None,
        cfg: InferenceConfig | None = None,
        *,
        loop_controller: LoopController | None = None,
        lsa_manager: LSAManager | None = None,
    ) -> None:
        self.cfg = cfg if cfg is not None else InferenceConfig()
        self.cfg.validate()

        self.outer_model = model
        if isinstance(model, Qwen36LoopHSSDForCausalLM):
            self.backbone = model.backbone
            self.loop_controller = loop_controller or model.loop_controller
            self.lsa_manager = lsa_manager or model.lsa_manager
        else:
            self.backbone = model
            self.loop_controller = loop_controller or self._detect_loop(model)
            self.lsa_manager = lsa_manager

        self.tokenizer = tokenizer
        self.text_model = find_text_model(self.backbone)
        self.text_config = self.text_model.config
        head = find_lm_head(self.backbone, getattr(self.text_config, "vocab_size", None))
        require(head is not None, "model has no lm_head module")
        assert head is not None
        self.lm_head = head

        phys_pairs, _, _ = physical_layers(self.backbone)
        self.num_physical_layers = len(phys_pairs)
        mtype = str(getattr(self.text_config, "model_type", "") or "")
        self._is_toy = mtype.startswith("toy") or type(self.backbone).__name__.startswith("Toy")

        self.backbone.eval()
        self.outer_model.eval()

        if self.cfg.active_passes is not None:
            self.set_active_passes(int(self.cfg.active_passes))

        self._compiled_lm_head: Any = None
        if self.cfg.compile_decode:
            self._ensure_compiled_decode()

    @staticmethod
    def _detect_loop(backbone: nn.Module) -> LoopController | None:
        tm = find_text_model(backbone)
        layers = getattr(tm, "layers", None)
        if isinstance(layers, nn.ModuleList) and len(layers) > 0:
            ctl = getattr(layers[0], "_ctl", None)
            if isinstance(ctl, LoopController):
                return ctl
        return None

    # ------------------------------------------------------------------ constructors

    @classmethod
    def from_stack(
        cls,
        stack: Stack | LoadedModelPack,
        cfg: InferenceConfig | None = None,
    ) -> OptimizedLoopInferenceEngine:
        """Construct an `OptimizedLoopInferenceEngine` directly from an in-memory `Stack`."""
        hf_mod = getattr(stack, "hf_model", None)
        model = hf_mod if isinstance(hf_mod, nn.Module) else stack.model
        return cls(
            model=model,
            tokenizer=stack.tokenizer,
            cfg=cfg,
            loop_controller=stack.loop,
            lsa_manager=stack.lsa,
        )

    @classmethod
    def from_pretrained(
        cls,
        pack_dir_or_repo: str | os.PathLike[str],
        cfg: InferenceConfig | None = None,
        *,
        token: str | None = None,
        device: str | torch.device | None = None,
        device_map: str | torch.device | None = None,
        dtype: str | torch.dtype | None = "auto",
        revision: str | None = None,
        local_files_only: bool = False,
        api: Any = None,
    ) -> OptimizedLoopInferenceEngine:
        """Load a Whole Model Pack from disk or Hugging Face Hub into an `OptimizedLoopInferenceEngine`."""
        from ..export_pack import load_whole_model_pack

        icfg = cfg if cfg is not None else InferenceConfig()
        pack = load_whole_model_pack(
            pack_dir_or_repo,
            device=device,
            device_map=device_map,
            dtype=dtype,
            active_passes=icfg.active_passes,
            token=token,
            revision=revision,
            local_files_only=local_files_only,
            api=api,
        )
        return cls.from_stack(pack, cfg=icfg)

    # ------------------------------------------------------------------ dynamic depth control

    @property
    def device(self) -> torch.device:
        return self.text_model.embed_tokens.weight.device

    @property
    def active_passes(self) -> int:
        return 1 if self.loop_controller is None else int(self.loop_controller.active_passes)

    @property
    def max_passes(self) -> int:
        return 1 if self.loop_controller is None else int(self.loop_controller.spec.num_passes)

    @property
    def effective_layers(self) -> int:
        if self.loop_controller is None:
            return self.num_physical_layers
        core_len = int(self.loop_controller.spec.core_len)
        return self.num_physical_layers + (self.active_passes - 1) * core_len

    def set_active_passes(self, r: int) -> None:
        """Switch active core pass count (`r in {1, 2, 3}` -> `40L / 64L / 88L`) with zero weight reload."""
        if isinstance(self.outer_model, Qwen36LoopHSSDForCausalLM):
            self.outer_model.set_active_passes(int(r))
            return
        if self.loop_controller is None:
            require(int(r) == 1, f"model is not Loop-wrapped; cannot set active_passes={r}")
            return
        self.loop_controller.set_active_passes(int(r))

    def depth(self, r: int) -> _DepthContext:
        """Set active depth `r` immediately AND return a `with engine.depth(r):` context manager."""
        prev = self.active_passes
        self.set_active_passes(int(r))
        return _DepthContext(self, prev, int(r))

    # ------------------------------------------------------------------ cache & torch.compile helpers

    def create_cache(self, quant_mode: str | None = None) -> Any:
        """Allocate a fresh hybrid KV + recurrent state cache for prefill + decode."""
        mode = quant_mode or self.cfg.kv_cache_quant
        n_virt = int(getattr(self.text_model.config, "num_hidden_layers", self.num_physical_layers))
        ltypes = list(
            getattr(
                self.text_model.config,
                "layer_types",
                ["full_attention"] * n_virt,
            )
        )
        if mode in ("fp8", "hadamard_int4", "bf16", "fp16"):
            return QuantizedHybridCache(num_layers=n_virt, layer_types=ltypes, mode=mode)
        if self._is_toy:
            return ToyHybridCache(self.text_model.config)
        return None

    def _ensure_compiled_decode(self) -> None:
        if self._compiled_lm_head is not None:
            return
        if not hasattr(torch, "compile"):
            self._compiled_lm_head = self.lm_head
            return
        try:
            if hasattr(torch, "_dynamo") and hasattr(torch._dynamo, "config"):
                torch._dynamo.config.allow_unspec_int_on_nn_module = True
                torch._dynamo.config.suppress_errors = True
            backend = self.cfg.compile_backend or ("inductor" if self.device.type == "cuda" else "aot_eager")

            def _step_head(h_last: torch.Tensor) -> torch.Tensor:
                return self.lm_head(h_last)

            self._compiled_lm_head = torch.compile(_step_head, backend=backend, mode=self.cfg.compile_mode)
        except Exception:
            self._compiled_lm_head = self.lm_head

    def _project_last_logits(self, h_last: torch.Tensor, *, is_decode: bool) -> torch.Tensor:
        if is_decode and self.cfg.compile_decode:
            self._ensure_compiled_decode()
            return self._compiled_lm_head(h_last)
        return self.lm_head(h_last)

    # ------------------------------------------------------------------ routing & memory telemetry

    def _model_param_bytes(self) -> int:
        seen: set[int] = set()
        tot = 0
        for p in self.backbone.parameters():
            if p.numel() == 0:
                continue
            ptr = int(p.untyped_storage().data_ptr())
            if ptr not in seen:
                seen.add(ptr)
                tot += int(p.numel() * p.element_size())
        return tot

    def _unique_experts_last_step(self) -> float:
        default_k = float(getattr(self.text_config, "num_experts_per_tok", 0) or 0)
        if self.loop_controller is None:
            return default_k
        routing = self.loop_controller.ctx.routing
        if not routing:
            return default_k
        r_max = self.active_passes
        by_phys: dict[int, list[torch.Tensor]] = {}
        for (phys, r), top_idx in routing.items():
            if 1 <= r <= r_max:
                by_phys.setdefault(phys, []).append(top_idx)
        if not by_phys:
            return default_k

        layer_means: list[float] = []
        for idx_list in by_phys.values():
            base_n = idx_list[0].shape[0]
            aligned = [t for t in idx_list if t.shape[0] == base_n]
            cat = torch.cat(aligned, dim=-1)  # [N_tok, R_act * top_k]
            sorted_cat, _ = torch.sort(cat, dim=-1)
            uniq_per_tok = 1 + (sorted_cat[:, 1:] != sorted_cat[:, :-1]).sum(dim=-1).float()
            layer_means.append(float(uniq_per_tok.mean().item()))
        return sum(layer_means) / len(layer_means)

    # ------------------------------------------------------------------ chunked prefill & causal MoR decode

    def _run_text_model_step(
        self,
        step_ids: torch.Tensor,
        cache: Any,
        *,
        allow_mor_early_exit: bool = False,
    ) -> tuple[Any, Any, int]:
        """Run `text_model` on `step_ids` and return `(tm_out, updated_cache, executed_passes)`."""
        ctl = self.loop_controller
        orig_r = self.active_passes
        executed_r = orig_r
        handles: list[Any] = []

        if (
            allow_mor_early_exit
            and ctl is not None
            and orig_r > 1
            and step_ids.shape[0] == 1
            and step_ids.shape[1] == 1
        ):
            vlayers = getattr(self.text_model, "layers", None)
            if isinstance(vlayers, nn.ModuleList):
                if ctl.adapters.mor_router is not None and ctl.spec.mor:
                    # Causal MoR router decides token depth at pass-1 entry; clamp active_passes right after!
                    for vl in vlayers:
                        slot = getattr(vl, "slot", None)
                        if slot is not None and slot.r == 1 and slot.entry:

                            def _on_p1_entry(_m: nn.Module, _i: Any, _o: Any) -> None:
                                non_zero_r = 1
                                for r_chk in range(2, orig_r + 1):
                                    kp = ctl.ctx.keep.get(r_chk)
                                    if kp is not None and bool((kp > 0).any()):
                                        non_zero_r = r_chk
                                ctl.active_passes = non_zero_r

                            handles.append(vl.register_forward_hook(_on_p1_entry))
                            break
                else:
                    # Confidence-based causal early exit at each core pass exit (1 .. orig_r - 1)
                    thresh = float(self.cfg.mor_confidence_threshold)
                    norm_mod = getattr(self.text_model, "norm", None)
                    for vl in vlayers:
                        slot = getattr(vl, "slot", None)
                        if slot is not None and slot.exit and 1 <= slot.r < orig_r:

                            def _on_pass_exit(
                                _m: nn.Module,
                                _i: Any,
                                out_val: Any,
                                _r: int = int(slot.r),
                            ) -> None:
                                if _r > ctl.active_passes:
                                    return
                                h_cur = out_val[0] if isinstance(out_val, tuple) else out_val
                                h_norm = norm_mod(h_cur) if callable(norm_mod) else h_cur
                                max_p = F.softmax(self.lm_head(h_norm[:, -1:, :]).float(), dim=-1).amax()
                                if float(max_p.item()) >= thresh:
                                    ctl.active_passes = _r

                            handles.append(vl.register_forward_hook(_on_pass_exit))

        try:
            tm_out = self.text_model(
                input_ids=step_ids,
                **{cache_kwarg_name(self.text_model): cache, "use_cache": True},
            )
            executed_r = 1 if ctl is None else int(ctl.active_passes)
        finally:
            for h in handles:
                h.remove()
            if ctl is not None:
                ctl.active_passes = orig_r

        next_cache = extract_output_cache(tm_out)
        if next_cache is None:
            next_cache = cache
        return tm_out, next_cache, executed_r

    @torch.no_grad()
    def prefill(
        self,
        input_ids: torch.Tensor,
        *,
        cache: Any = None,
        chunk_size: int | None = None,
    ) -> tuple[torch.Tensor, Any, int]:
        """Stream `input_ids [B, T]` in `prefill_chunk_size` chunks; materialize logits ONLY on token `T-1`."""
        require(input_ids.ndim == 2 and input_ids.shape[1] >= 1, "input_ids must have shape [B, T >= 1]")
        ids = input_ids.to(self.device)
        seq_len = int(ids.shape[1])
        csize = max(1, int(chunk_size if chunk_size is not None else self.cfg.prefill_chunk_size))

        if cache is None:
            cache = self.create_cache()
        if self.lsa_manager is not None:
            self.lsa_manager.clear_cache()

        n_chunks = 0
        last_hidden_final: torch.Tensor | None = None
        for start in range(0, seq_len, csize):
            end = min(seq_len, start + csize)
            chunk = ids[:, start:end]
            tm_out, cache, _ = self._run_text_model_step(chunk, cache, allow_mor_early_exit=False)
            n_chunks += 1
            if end == seq_len:
                last_hidden_final = tm_out.last_hidden_state[:, -1:, :]
            del tm_out

        assert last_hidden_final is not None
        logits_last = self._project_last_logits(last_hidden_final, is_decode=False)
        return logits_last, cache, n_chunks

    @torch.no_grad()
    def forward_logits(self, input_ids: torch.Tensor, *, depth: int | None = None) -> torch.Tensor:
        """Compute full-sequence logits `[B, T, V]` (convenience helper for eval/logit-equivalence checks)."""
        ids = input_ids.to(self.device)
        if depth is not None:
            prev = self.active_passes
            self.set_active_passes(int(depth))
            try:
                return self.outer_model(input_ids=ids, use_cache=False).logits
            finally:
                self.set_active_passes(prev)
        return self.outer_model(input_ids=ids, use_cache=False).logits

    # ------------------------------------------------------------------ tokenization helpers

    def _encode_prompt(self, prompt: str | Sequence[int] | torch.Tensor) -> torch.Tensor:
        if isinstance(prompt, torch.Tensor):
            ids = prompt.long()
            return ids.unsqueeze(0) if ids.ndim == 1 else ids
        if isinstance(prompt, str):
            require(self.tokenizer is not None, "string prompt passed to generate(), but engine has no tokenizer")
            raw = self.tokenizer.encode(prompt, add_special_tokens=False)
            if hasattr(raw, "tolist"):
                raw = raw.tolist()
            return torch.tensor([int(t) for t in raw], dtype=torch.long, device=self.device).unsqueeze(0)
        return torch.tensor([int(t) for t in prompt], dtype=torch.long, device=self.device).unsqueeze(0)

    def _resolve_stop_ids(self, stop_token_ids: Sequence[int] | None) -> set[int]:
        stops: set[int] = {int(t) for t in self.cfg.stop_token_ids}
        if stop_token_ids is not None:
            stops.update(int(t) for t in stop_token_ids)
        tok_eos = getattr(self.tokenizer, "eos_token_id", None)
        if tok_eos is not None:
            if isinstance(tok_eos, (list, tuple, set)):
                stops.update(int(t) for t in tok_eos)
            else:
                stops.add(int(tok_eos))
        cfg_eos = getattr(self.text_config, "eos_token_id", None)
        if cfg_eos is not None and not stops:
            if isinstance(cfg_eos, (list, tuple, set)):
                stops.update(int(t) for t in cfg_eos)
            else:
                stops.add(int(cfg_eos))
        return stops

    def _sync_device(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    # ------------------------------------------------------------------ autoregressive generation

    @torch.no_grad()
    def generate(
        self,
        prompt: str | Sequence[int] | torch.Tensor,
        *,
        max_new_tokens: int | None = None,
        depth: int | None = None,
        temperature: float | None = None,
        top_k: int | None = None,
        top_p: float | None = None,
        stop_token_ids: Sequence[int] | None = None,
        prefill_chunk_size: int | None = None,
        seed: int | None = None,
        ignore_eos: bool = False,
    ) -> GenerationResult:
        """Run chunked prefill + optimized autoregressive decode with live telemetry."""
        prev_depth = self.active_passes
        if depth is not None:
            self.set_active_passes(int(depth))

        try:
            prompt_ids_2d = self._encode_prompt(prompt).to(self.device)
            require(prompt_ids_2d.shape[0] == 1, "OptimizedLoopInferenceEngine.generate expects batch_size == 1")
            prompt_len = int(prompt_ids_2d.shape[1])

            n_new = int(self.cfg.max_new_tokens if max_new_tokens is None else max_new_tokens)
            temp = float(self.cfg.temperature if temperature is None else temperature)
            tk = int(self.cfg.top_k if top_k is None else top_k)
            tp = float(self.cfg.top_p if top_p is None else top_p)
            stops = set() if ignore_eos else self._resolve_stop_ids(stop_token_ids)
            use_seed = self.cfg.seed if seed is None else seed

            gen_rng: torch.Generator | None = None
            if use_seed is not None and temp > 0.0:
                gen_rng = torch.Generator(device=self.device.type if self.device.type == "cuda" else "cpu")
                gen_rng.manual_seed(int(use_seed))

            if self.device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(self.device)

            self._sync_device()
            t0 = time.perf_counter()
            logits_last, cache, n_chunks = self.prefill(
                prompt_ids_2d,
                chunk_size=prefill_chunk_size,
            )
            self._sync_device()
            t_prefill_end = time.perf_counter()
            prefill_ms = (t_prefill_end - t0) * 1000.0

            expert_counts: list[float] = [self._unique_experts_last_step()]
            pass_counts: list[int] = []
            generated: list[int] = []
            finish_reason = "length"
            hit_stop_id: int | None = None

            if n_new > 0:
                nxt_tok = _sample_next_token(
                    logits_last,
                    temperature=temp,
                    top_k=tk,
                    top_p=tp,
                    generator=gen_rng,
                )
                tok_val = int(nxt_tok.item())
                generated.append(tok_val)
                pass_counts.append(self.active_passes)
                if tok_val in stops:
                    finish_reason = "stop"
                    hit_stop_id = tok_val

                while len(generated) < n_new and finish_reason != "stop":
                    tm_out, cache, exec_r = self._run_text_model_step(
                        nxt_tok.to(self.device),
                        cache,
                        allow_mor_early_exit=bool(self.cfg.mor_early_exit),
                    )
                    h_last = tm_out.last_hidden_state[:, -1:, :]
                    logits_last = self._project_last_logits(h_last, is_decode=True)
                    del tm_out

                    pass_counts.append(exec_r)
                    expert_counts.append(self._unique_experts_last_step())

                    nxt_tok = _sample_next_token(
                        logits_last,
                        temperature=temp,
                        top_k=tk,
                        top_p=tp,
                        generator=gen_rng,
                    )
                    tok_val = int(nxt_tok.item())
                    generated.append(tok_val)
                    if tok_val in stops:
                        finish_reason = "stop"
                        hit_stop_id = tok_val

            self._sync_device()
            t_end = time.perf_counter()
            decode_ms = (t_end - t_prefill_end) * 1000.0
            total_ms = (t_end - t0) * 1000.0

            kv_b, rec_b = _measure_any_cache_bytes(cache)
            mib = float(1024 * 1024)
            if self.device.type == "cuda":
                peak_mb = float(torch.cuda.max_memory_allocated(self.device)) / mib
            else:
                peak_mb = float(self._model_param_bytes() + kv_b + rec_b) / mib

            prefill_s = max(prefill_ms / 1000.0, 1e-9)
            decode_s = max(decode_ms / 1000.0, 1e-9)
            decode_steps = max(len(generated) - 1, 1) if len(generated) > 1 else len(generated)

            metrics = GenerationMetrics(
                prompt_tokens=prompt_len,
                generated_tokens=len(generated),
                prefill_ms=round(prefill_ms, 3),
                decode_ms=round(decode_ms, 3),
                total_ms=round(total_ms, 3),
                prefill_tok_s=round(prompt_len / prefill_s, 2),
                decode_tok_s=round(decode_steps / decode_s, 2) if generated else 0.0,
                peak_vram_mb=round(peak_mb, 3),
                kv_cache_mb=round(kv_b / mib, 4),
                recurrent_state_mb=round(rec_b / mib, 4),
                active_passes=self.active_passes,
                mean_active_passes=round(
                    sum(pass_counts) / len(pass_counts), 3
                ) if pass_counts else float(self.active_passes),
                effective_layers=self.effective_layers,
                unique_experts_per_tok=round(sum(expert_counts) / len(expert_counts), 3),
                prefill_chunks=n_chunks,
                kv_cache_quant=self.cfg.kv_cache_quant,
                compiled_decode=bool(self.cfg.compile_decode),
            )

            text = ""
            if self.tokenizer is not None and hasattr(self.tokenizer, "decode"):
                decode_ids = generated[:-1] if (finish_reason == "stop" and generated) else generated
                tok_len = getattr(self.tokenizer, "vocab_size", None)
                if hasattr(self.tokenizer, "__len__"):
                    tok_len = max(int(tok_len or 0), len(self.tokenizer))
                if isinstance(tok_len, int) and tok_len > 0:
                    decode_ids = [tid for tid in decode_ids if 0 <= tid < tok_len]
                text = str(self.tokenizer.decode(decode_ids, skip_special_tokens=True))

            return GenerationResult(
                text=text,
                token_ids=generated,
                prompt_token_ids=[int(x) for x in prompt_ids_2d[0].tolist()],
                finish_reason=finish_reason,
                stop_token_id=hit_stop_id,
                logits_last=logits_last.detach().cpu(),
                metrics=metrics,
            )
        finally:
            if depth is not None:
                self.set_active_passes(prev_depth)

    # ------------------------------------------------------------------ structured depth benchmark

    @torch.no_grad()
    def benchmark(
        self,
        prompt_len: int = 512,
        gen_len: int = 64,
        depths: Sequence[int] = (1, 2, 3),
    ) -> dict[str, Any]:
        """Benchmark prefill & decode throughput, latency, KV VRAM, and expert diversity across `depths`."""
        require(int(prompt_len) >= 1 and int(gen_len) >= 1, "prompt_len and gen_len must be >= 1")
        vocab = int(getattr(self.text_config, "vocab_size", 256))
        rng = torch.Generator(device="cpu").manual_seed(2026)
        synth_prompt = torch.randint(4, max(8, min(vocab, 250)), (1, int(prompt_len)), generator=rng)

        # Short warmup on depth 1 to stabilize allocator / optional torch.compile cache
        orig_passes = self.active_passes
        try:
            self.set_active_passes(1)
            _ = self.generate(synth_prompt[:, : min(8, int(prompt_len))], max_new_tokens=2, ignore_eos=True)

            per_depth: dict[int, dict[str, Any]] = {}
            for r in depths:
                r_int = int(r)
                if r_int > self.max_passes:
                    continue
                res = self.generate(
                    synth_prompt,
                    max_new_tokens=int(gen_len),
                    depth=r_int,
                    temperature=0.0,
                    ignore_eos=True,
                )
                per_depth[r_int] = res.metrics.to_dict()
        finally:
            self.set_active_passes(orig_passes)

        return {
            **per_depth,
            "depths": per_depth,
            "prompt_len": int(prompt_len),
            "gen_len": int(gen_len),
            "device": str(self.device),
            "kv_cache_quant": self.cfg.kv_cache_quant,
            "prefill_chunk_size": int(self.cfg.prefill_chunk_size),
            "compiled_decode": bool(self.cfg.compile_decode),
        }


__all__ = [
    "GenerationMetrics",
    "GenerationResult",
    "InferenceConfig",
    "OptimizedLoopInferenceEngine",
    "QuantizedHybridCache",
]
