"""Segmented `4 x 32K = 128K` Truncated BPTT with Differentiable K/V Gradient Replay (Fix E3).

Standard Truncated BPTT across `32K`-token segments carries recurrent linear-attention states
(`GDN-2` / `HSSD v3`) forward by value while detaching full-attention KV caches at segment
boundaries, starving `k_proj`, `v_proj`, `k_norm`, and `D-LSA` `W_k_idx` of cross-segment
retrieval gradients.

`SegmentedKVBPTTRunner` closes Review Finding `E3` in `< 2 GiB` extra memory:
  1. Splits `[B, T_total]` into segments of `seg_len` tokens (or `n_segments` equal chunks).
  2. Carries live recurrent/conv/KV caches across segment boundaries, detaching graph history
     after each segment's `.backward()` so peak activation HBM stays `O(seg_len)`.
  3. Caches lightweight pre-attention normalized hidden states `u_seg` (`bf16/fp32/fp64`,
     optionally offloaded to CPU RAM) and rotary `(cos, sin)` slices for full-attention layers.
  4. On Segment `s > 0`, selects the Top-`replay_k` prior-segment token positions per KV head
     (via `D-LSA` latent similarity when installed, or `192-d` NoPE content cosine similarity
     otherwise) and **re-projects only those selected prior hidden states through live `k_proj`,
     `k_norm`, `v_proj`, and `lsa_indexer.k_proj` with `requires_grad=True`**, scattering them
     out-of-place into the KV tensor consumed by Segment `s`'s attention!
"""
from __future__ import annotations

import copy
import inspect
import math
from dataclasses import dataclass, field, fields
from typing import Any, Callable

import torch
import torch.nn.functional as F
from torch import nn

from ..model._dtype import upcast
from ..model.experts import physical_layers
from ..model.introspect import FULL, find_text_model
from .loss import lm_loss


@dataclass
class BPTTConfig:
    """Configuration for `SegmentedKVBPTTRunner` (`4 x 32K = 128K` K/V-replay TBPTT)."""

    enabled: bool = True
    seg_len: int = 32_768
    n_segments: int | None = None
    replay_k: int = 2_048
    replay_enabled: bool = True
    offload_u_to_cpu: bool = True
    offload_prior_hidden: bool | None = None
    replay_indexer_weight: float = 0.1
    query_sim_chunk: int = 256

    def __post_init__(self) -> None:
        if self.offload_prior_hidden is not None:
            self.offload_u_to_cpu = bool(self.offload_prior_hidden)
        else:
            self.offload_prior_hidden = bool(self.offload_u_to_cpu)

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> BPTTConfig:
        raw = dict(d or {})
        known = {f.name for f in fields(cls)}
        bad = set(raw) - known
        if bad:
            raise ValueError(f"unknown bptt.* keys {sorted(bad)}; valid: {sorted(known)}")
        return cls(**raw)

    def resolve_seg_len(self, total_len: int) -> int:
        if total_len <= 0:
            raise ValueError(f"total_len must be > 0, got {total_len}")
        if self.n_segments is not None:
            if self.n_segments < 1:
                raise ValueError(f"n_segments must be >= 1, got {self.n_segments}")
            return max(1, math.ceil(total_len / self.n_segments))
        if self.seg_len < 1:
            raise ValueError(f"seg_len must be >= 1, got {self.seg_len}")
        return int(self.seg_len)


@dataclass
class BPTTStepResult:
    """Summary of a segmented K/V-replay TBPTT step."""

    total_loss: float
    segment_losses: list[float]
    num_segments: int
    replayed_tokens_per_seg: list[int]
    replayed_indices: dict[tuple[int, int], torch.Tensor] = field(default_factory=dict)
    stats: dict[str, float] = field(default_factory=dict)


def _in_backward() -> bool:
    fn = getattr(torch._C, "_current_graph_task_id", None)
    return fn is not None and fn() != -1


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    a, b = x.chunk(2, dim=-1)
    return torch.cat([-b, a], dim=-1)


def _apply_partial_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply `(cos, sin)` (`[B, T, rot_dim]`) to `x` (`[B, H, T, D]`), preserving `[..., rot_dim:]`."""
    r = cos.shape[-1]
    if r == 0:
        return x
    xr, xp = x[..., :r], x[..., r:]
    c = cos[:, None].to(xr.dtype)
    s = sin[:, None].to(xr.dtype)
    return torch.cat([xr * c + _rotate_half(xr) * s, xp], dim=-1)


def _detach_nested(obj: Any, seen: set[int] | None = None) -> Any:
    """Detach all tensors inside an arbitrary hybrid cache object in-place while keeping live values."""
    if seen is None:
        seen = set()
    oid = id(obj)
    if oid in seen:
        return obj
    if isinstance(obj, torch.Tensor):
        return obj.detach()
    seen.add(oid)
    if isinstance(obj, list):
        for i, item in enumerate(obj):
            obj[i] = _detach_nested(item, seen)
        return obj
    if isinstance(obj, tuple):
        return tuple(_detach_nested(x, seen) for x in obj)
    if isinstance(obj, dict):
        for k, val in list(obj.items()):
            obj[k] = _detach_nested(val, seen)
        return obj
    if hasattr(obj, "__dict__"):
        for attr, val in list(vars(obj).items()):
            if attr.startswith("__"):
                continue
            if isinstance(val, (torch.Tensor, list, dict, tuple)) or hasattr(val, "__dict__"):
                try:
                    setattr(obj, attr, _detach_nested(val, seen))
                except Exception:
                    pass
    return obj


def _extract_cached_keys(cache: Any, layer_idx: int) -> torch.Tensor | None:
    """Extract cached key tensor `[B, G, Tk, D]` for `layer_idx` across Toy and HF cache types."""
    kc = getattr(cache, "key_cache", None)
    if isinstance(kc, list) and layer_idx < len(kc) and isinstance(kc[layer_idx], torch.Tensor):
        return kc[layer_idx]
    layers = getattr(cache, "layers", None)
    if isinstance(layers, list) and layer_idx < len(layers):
        keys = getattr(layers[layer_idx], "keys", None)
        if isinstance(keys, torch.Tensor):
            return keys
    return None


def _extract_cached_kv(cache: Any, layer_idx: int) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Extract cached `(key, value)` tensors `[B, G, Tk, D]` for `layer_idx` across Toy and HF caches."""
    kc = getattr(cache, "key_cache", None)
    vc = getattr(cache, "value_cache", None)
    if (
        isinstance(kc, list)
        and isinstance(vc, list)
        and layer_idx < len(kc)
        and layer_idx < len(vc)
        and isinstance(kc[layer_idx], torch.Tensor)
        and isinstance(vc[layer_idx], torch.Tensor)
    ):
        return kc[layer_idx], vc[layer_idx]
    layers = getattr(cache, "layers", None)
    if isinstance(layers, list) and layer_idx < len(layers):
        lyr = layers[layer_idx]
        keys = getattr(lyr, "keys", None)
        vals = getattr(lyr, "values", None)
        if isinstance(keys, torch.Tensor) and isinstance(vals, torch.Tensor):
            return keys, vals
    return None, None


class _SlotStateListProxy:
    """Checkpoint-recomputation-safe list view over `conv_states` / `recurrent_states`."""

    def __init__(self, backing: list[Any]) -> None:
        self._backing = backing
        self._snapshot = [x.detach() if isinstance(x, torch.Tensor) else x for x in backing]

    def __len__(self) -> int:
        return len(self._backing)

    def __getitem__(self, idx: int) -> Any:
        if _in_backward():
            return self._snapshot[idx]
        return self._backing[idx]

    def __setitem__(self, idx: int, val: Any) -> None:
        if _in_backward():
            return
        self._backing[idx] = val.detach() if isinstance(val, torch.Tensor) else val


def _clone_detached_state(obj: Any) -> Any:
    """Return an independent detached clone of a conv/recurrent state (safe against in-place `.copy_`)."""
    if obj is None:
        return None
    if isinstance(obj, torch.Tensor):
        return obj.detach().clone()
    clone_fn = getattr(obj, "clone", None)
    if callable(clone_fn):
        return _detach_nested(clone_fn())
    try:
        return _detach_nested(copy.deepcopy(obj))
    except Exception:
        return _detach_nested(obj)


def _resolve_bptt_backbone_and_head(model: nn.Module) -> tuple[nn.Module, nn.Module | None]:
    tm = find_text_model(model)
    head = getattr(model, "lm_head", None)
    if head is None and callable(getattr(model, "get_output_embeddings", None)):
        head = model.get_output_embeddings()
    if not isinstance(head, nn.Module):
        head = None
    return tm, head


def _can_bypass_bptt_causal_mask(tm: nn.Module) -> bool:
    """Return True when `qloop_lsa` handles lower-right causal masking without HF's O(T_q*T_k) 4D mask."""
    cfg = getattr(tm, "config", None)
    if cfg is None or str(getattr(cfg, "model_type", "")).startswith("toy"):
        return False
    if str(getattr(cfg, "_attn_implementation", "")) != "qloop_lsa":
        return False
    return any(getattr(m, "lsa_indexer", None) is not None for m in tm.modules())


def _chunked_replay_indexer_kl(
    qn_t: torch.Tensor,
    kn_t: torch.Tensor,
    qz_live: torch.Tensor,
    kz_live: torch.Tensor,
    *,
    scale_nope: float,
    temperature: float,
    per_head_q: bool,
    chunk_size: int,
) -> torch.Tensor:
    """Memory-lean chunked KL(softmax(q_nope k_nope^T / sqrt(d)) || softmax(qz kz^T / tau)).

    Computes exact scalar KL and exact first-order analytical gradients `(dKL/dqz, dKL/dkz)` in
    `chunk_size`-token slices (`O(chunk * K_sel)` instead of `O(T_seg * K_sel)`, cutting `17 GiB`
    to `< 32 MiB` at `T_seg=32K, K_sel=2048`) while preserving bit-exact gradients through `qz_live`
    and `kz_live` with zero large tensors saved in checkpoint frames.
    """
    bsz, num_kv_heads, hg, t_seg, _ = qn_t.shape
    inv_temp = 1.0 / float(temperature)
    chunk = max(1, int(chunk_size))
    qz_view = (
        qz_live.view(bsz, num_kv_heads, hg, t_seg, qz_live.shape[-1])
        if per_head_q
        else qz_live
    )
    denom = float(bsz * num_kv_heads * (hg if per_head_q else 1) * max(1, t_seg))

    with torch.no_grad():
        qz_d = upcast(qz_view.detach())
        kz_d = upcast(kz_live.detach())
        g_qz = torch.zeros_like(qz_d)
        g_kz = torch.zeros_like(kz_d)
        kl_sum = qz_d.new_zeros(())
        for c0 in range(0, t_seg, chunk):
            c1 = min(t_seg, c0 + chunk)
            raw_sc = torch.einsum("bghcd,bgkd->bghck", qn_t[:, :, :, c0:c1], kn_t) * scale_nope
            if per_head_q:
                tgt_p = torch.softmax(raw_sc, dim=-1)  # [B, G, Hg, C, K]
                qz_c = qz_d[:, :, :, c0:c1]
                pred_logits = torch.einsum("bghcd,bgkd->bghck", qz_c, kz_d) * inv_temp
                log_pred = torch.log_softmax(pred_logits, dim=-1)
                kl_sum = kl_sum + (tgt_p * (torch.log(tgt_p.clamp_min(1e-30)) - log_pred)).sum()
                d_logits = (log_pred.exp() - tgt_p) * (inv_temp / denom)
                g_qz[:, :, :, c0:c1] = torch.einsum("bghck,bgkd->bghcd", d_logits, kz_d)
                g_kz.add_(torch.einsum("bghck,bghcd->bgkd", d_logits, qz_c))
            else:
                tgt_p = torch.softmax(raw_sc, dim=-1).mean(dim=2)  # [B, G, C, K]
                qz_c = qz_d[:, :, c0:c1]
                pred_logits = torch.einsum("bgcd,bgkd->bgck", qz_c, kz_d) * inv_temp
                log_pred = torch.log_softmax(pred_logits, dim=-1)
                kl_sum = kl_sum + (tgt_p * (torch.log(tgt_p.clamp_min(1e-30)) - log_pred)).sum()
                d_logits = (log_pred.exp() - tgt_p) * (inv_temp / denom)
                g_qz[:, :, c0:c1] = torch.einsum("bgck,bgkd->bgcd", d_logits, kz_d)
                g_kz.add_(torch.einsum("bgck,bgcd->bgkd", d_logits, qz_c))

    kl_val = (kl_sum / denom).to(qz_live.dtype)
    g_qz_cast = g_qz.to(qz_live.dtype).reshape_as(qz_live)
    g_kz_cast = g_kz.to(kz_live.dtype)
    return (
        kl_val
        + (qz_live * g_qz_cast).sum()
        - (qz_live.detach() * g_qz_cast).sum()
        + (kz_live * g_kz_cast).sum()
        - (kz_live.detach() * g_kz_cast).sum()
    )


def _maybe_create_initial_cache(tm: nn.Module) -> Any:
    """Create an empty hybrid KV/recurrent cache when possible so Segment 0 is checkpoint-safe."""
    cfg = getattr(tm, "config", None)
    if cfg is not None and str(getattr(cfg, "model_type", "")).startswith("toy"):
        from ..model.toy import ToyHybridCache

        return ToyHybridCache(cfg)
    try:
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DynamicCache

        if cfg is not None:
            return Qwen3_5DynamicCache(cfg)
    except Exception:
        pass
    try:
        from transformers.cache_utils import DynamicCache

        if cfg is not None:
            return DynamicCache(config=cfg)
    except Exception:
        pass
    return None


def _dequantize_fp8_kv_lean(qkv: Any, dtype: torch.dtype, chunk_tokens: int = 65536) -> torch.Tensor:
    """Chunked dequantization of `QuantizedFP8KV` -> `dtype` without allocating a full fp32 copy."""
    if not getattr(qkv, "is_fp8_native", False):
        return qkv.data.to(dtype)
    q_data: torch.Tensor = qkv.data
    scale: torch.Tensor = qkv.scale.to(device=q_data.device, dtype=torch.float32)
    t_len = int(q_data.shape[-2])
    if t_len <= chunk_tokens or dtype == torch.float64:
        acc_dt = torch.float64 if dtype == torch.float64 else torch.float32
        return (q_data.to(acc_dt) * scale.to(acc_dt)).to(dtype)
    out = torch.empty(q_data.shape, device=q_data.device, dtype=dtype)
    for t0 in range(0, t_len, chunk_tokens):
        t1 = min(t_len, t0 + chunk_tokens)
        out[..., t0:t1, :] = (q_data[..., t0:t1, :].float() * scale).to(dtype)
    return out


class _ReplayKVCacheProxy:
    """Wraps `past_key_values` so `update` injects live re-projected `(k_live, v_live)` out-of-place
    and linear-attention conv/recurrent states are checkpoint-recomputation safe across HF 4.x & 5.x."""

    def __init__(
        self,
        base_cache: Any,
        *,
        readonly_base: bool = False,
        fp8_prefix_kv: dict[int, tuple[Any, Any]] | None = None,
    ) -> None:
        object.__setattr__(self, "_base_cache", base_cache)
        object.__setattr__(self, "_readonly_base", bool(readonly_base))
        object.__setattr__(self, "_detach_recurrent_state", True)
        object.__setattr__(self, "_fp8_prefix_kv", dict(fp8_prefix_kv) if fp8_prefix_kv else {})
        object.__setattr__(self, "_pending_replay", {})
        object.__setattr__(self, "_live_kv_for_seg", {})
        object.__setattr__(self, "_raw_prefix_kv", {})
        object.__setattr__(self, "_conv_snap", {})
        object.__setattr__(self, "_rec_snap", {})
        conv = getattr(base_cache, "conv_states", None)
        rec = getattr(base_cache, "recurrent_states", None)
        object.__setattr__(
            self, "_conv_proxy", _SlotStateListProxy(conv) if isinstance(conv, list) else None
        )
        object.__setattr__(
            self, "_rec_proxy", _SlotStateListProxy(rec) if isinstance(rec, list) else None
        )

    @property
    def base_cache(self) -> Any:
        return object.__getattribute__(self, "_base_cache")

    def get_seq_length(self, layer_idx: int | None = None) -> int:
        fp8_pref: dict[int, tuple[Any, Any]] = object.__getattribute__(self, "_fp8_prefix_kv")
        if fp8_pref:
            if layer_idx is not None and int(layer_idx) in fp8_pref:
                return int(fp8_pref[int(layer_idx)][0].data.shape[-2])
            first_pair = next(iter(fp8_pref.values()))
            return int(first_pair[0].data.shape[-2])
        base = object.__getattribute__(self, "_base_cache")
        fn = getattr(base, "get_seq_length", None)
        if callable(fn):
            if layer_idx is None:
                return int(fn())
            try:
                return int(fn(layer_idx))
            except TypeError:
                return int(fn())
        return 0

    def reset_segment(self) -> None:
        object.__getattribute__(self, "_pending_replay").clear()
        object.__getattribute__(self, "_live_kv_for_seg").clear()
        object.__getattribute__(self, "_raw_prefix_kv").clear()
        object.__getattribute__(self, "_fp8_prefix_kv").clear()
        object.__getattribute__(self, "_conv_snap").clear()
        object.__getattribute__(self, "_rec_snap").clear()

    def get_conv_state(self, layer_idx: int, state_idx: int = 0, **kwargs: Any) -> torch.Tensor | None:
        base = object.__getattribute__(self, "_base_cache")
        conv_snap: dict[tuple[int, int], tuple[bool, torch.Tensor | None]] = object.__getattribute__(
            self, "_conv_snap"
        )
        key = (int(layer_idx), int(state_idx))
        if _in_backward() and key in conv_snap:
            return conv_snap[key][1]
        if not _in_backward() and key not in conv_snap:
            val = base.get_conv_state(layer_idx, state_idx=state_idx, **kwargs)
            conv_snap[key] = (val is not None, _clone_detached_state(val))
        if key in conv_snap:
            return conv_snap[key][1]
        return base.get_conv_state(layer_idx, state_idx=state_idx, **kwargs)

    def update_conv_state(
        self,
        conv_states: torch.Tensor,
        layer_idx: int,
        state_idx: int = 0,
        conv_kernel_size: int | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        base = object.__getattribute__(self, "_base_cache")
        conv_snap: dict[tuple[int, int], tuple[bool, torch.Tensor | None]] = object.__getattribute__(
            self, "_conv_snap"
        )
        key = (int(layer_idx), int(state_idx))
        if not _in_backward():
            if key not in conv_snap:
                prev = (
                    base.get_conv_state(layer_idx, state_idx=state_idx)
                    if hasattr(base, "get_conv_state")
                    else None
                )
                conv_snap[key] = (prev is not None, _clone_detached_state(prev))
            base.update_conv_state(
                conv_states.detach(),
                layer_idx,
                state_idx=state_idx,
                conv_kernel_size=conv_kernel_size,
                **kwargs,
            )
        had_prev, prev_snap = conv_snap.get(key, (False, None))
        if had_prev and isinstance(prev_snap, torch.Tensor):
            return torch.cat([prev_snap.to(device=conv_states.device, dtype=conv_states.dtype), conv_states], dim=-1)
        full_conv = conv_states
        if conv_kernel_size is not None and full_conv.shape[-1] < int(conv_kernel_size):
            full_conv = F.pad(full_conv, (int(conv_kernel_size) - full_conv.shape[-1], 0), value=0)
        return full_conv

    def get_recurrent_state(self, layer_idx: int, state_idx: int = 0, **kwargs: Any) -> Any:
        base = object.__getattribute__(self, "_base_cache")
        rec_snap: dict[tuple[int, int], Any] = object.__getattribute__(self, "_rec_snap")
        key = (int(layer_idx), int(state_idx))
        if _in_backward() and key in rec_snap:
            return rec_snap[key]
        if not _in_backward() and key not in rec_snap:
            val = base.get_recurrent_state(layer_idx, state_idx=state_idx, **kwargs)
            rec_snap[key] = _clone_detached_state(val)
        if key in rec_snap:
            return rec_snap[key]
        return base.get_recurrent_state(layer_idx, state_idx=state_idx, **kwargs)

    def update_recurrent_state(
        self,
        recurrent_states: Any,
        layer_idx: int,
        state_idx: int = 0,
        **kwargs: Any,
    ) -> Any:
        if _in_backward():
            return recurrent_states
        base = object.__getattribute__(self, "_base_cache")
        rec_snap: dict[tuple[int, int], Any] = object.__getattribute__(self, "_rec_snap")
        key = (int(layer_idx), int(state_idx))
        if key not in rec_snap and hasattr(base, "get_recurrent_state"):
            prev = base.get_recurrent_state(layer_idx, state_idx=state_idx)
            rec_snap[key] = _clone_detached_state(prev)
        return base.update_recurrent_state(
            _clone_detached_state(recurrent_states),
            layer_idx,
            state_idx=state_idx,
            **kwargs,
        )

    def set_pending(
        self,
        layer_idx: int,
        indices: torch.Tensor,
        k_live: torch.Tensor,
        v_live: torch.Tensor,
    ) -> None:
        object.__getattribute__(self, "_pending_replay")[int(layer_idx)] = (indices, k_live, v_live)

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        *args: Any,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        base = object.__getattribute__(self, "_base_cache")
        readonly_base: bool = object.__getattribute__(self, "_readonly_base")
        fp8_pref: dict[int, tuple[Any, Any]] = object.__getattribute__(self, "_fp8_prefix_kv")
        live_map: dict[int, tuple[torch.Tensor, torch.Tensor]] = object.__getattribute__(self, "_live_kv_for_seg")
        pending: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = object.__getattribute__(
            self, "_pending_replay"
        )
        raw_pref: dict[int, tuple[torch.Tensor, torch.Tensor]] = object.__getattribute__(self, "_raw_prefix_kv")
        slot = int(layer_idx)
        # SharedKVView on Pass r >= 2 calls update with 0-length slice -> return Pass 1's live (k_ret, v_ret)
        if key_states.shape[-2] == 0 and slot in live_map:
            return live_map[slot]

        t_cur = key_states.shape[-2]
        if readonly_base:
            # Read-only prefix mode (e.g. `long_calib` suffix where `base_cache` is discarded afterward):
            # avoid mutating `base_cache` (saves a full 10-20 GiB duplicate prefix KV copy!) and when
            # `fp8_prefix_kv` holds true 1-byte FP8 E4M3 tensors, dequantize ONLY the active layer on the fly!
            if slot in fp8_pref:
                qk, qv = fp8_pref[slot]
                k_prefix = _dequantize_fp8_kv_lean(qk, key_states.dtype)
                v_prefix = _dequantize_fp8_kv_lean(qv, value_states.dtype)
                if t_cur == 0:
                    return k_prefix, v_prefix
                k_ret = torch.cat([k_prefix, key_states], dim=-2)
                v_ret = torch.cat([v_prefix, value_states], dim=-2)
                return k_ret, v_ret
            if slot not in raw_pref:
                kp, vp = _extract_cached_kv(base, slot)
                if kp is not None and vp is not None:
                    raw_pref[slot] = (kp.detach(), vp.detach())
            if t_cur == 0:
                if slot in raw_pref:
                    return raw_pref[slot]
                return key_states, value_states
        elif not _in_backward():
            k_full, v_full = base.update(key_states.detach(), value_states.detach(), slot, *args, **kwargs)
            if t_cur > 0:
                raw_pref[slot] = (k_full[:, :, :-t_cur, :], v_full[:, :, :-t_cur, :])
            else:
                live_map[slot] = (k_full, v_full)
                return k_full, v_full

        if t_cur > 0 and slot in raw_pref:
            k_prefix, v_prefix = raw_pref[slot]
            rep = pending.pop(slot, None)
            if rep is not None and k_prefix.shape[-2] > 0:
                idx_bgk, k_live, v_live = rep
                idx_exp = idx_bgk.unsqueeze(-1).expand(-1, -1, -1, k_prefix.shape[-1])
                k_prefix = torch.scatter(k_prefix, dim=2, index=idx_exp, src=k_live.to(k_prefix.dtype))
                v_prefix = torch.scatter(v_prefix, dim=2, index=idx_exp, src=v_live.to(v_prefix.dtype))
            k_ret = torch.cat([k_prefix, key_states], dim=-2)
            v_ret = torch.cat([v_prefix, value_states], dim=-2)
        else:
            k_ret, v_ret = key_states, value_states

        if not readonly_base:
            live_map[slot] = (k_ret, v_ret)
        return k_ret, v_ret

    def __getattr__(self, name: str) -> Any:
        if name == "conv_states":
            cp = object.__getattribute__(self, "_conv_proxy")
            if cp is not None:
                return cp
        if name == "recurrent_states":
            rp = object.__getattribute__(self, "_rec_proxy")
            if rp is not None:
                return rp
        return getattr(object.__getattribute__(self, "_base_cache"), name)

    def __setattr__(self, name: str, value: Any) -> None:
        setattr(object.__getattribute__(self, "_base_cache"), name, value)

    def __getitem__(self, idx: Any) -> Any:
        return object.__getattribute__(self, "_base_cache")[idx]


@dataclass
class _PriorSegRecord:
    u: torch.Tensor    # [B, T_seg, hidden_size] detached (CPU or device)
    cos: torch.Tensor  # [B, T_seg, rot_dim] detached
    sin: torch.Tensor  # [B, T_seg, rot_dim] detached


class SegmentedKVBPTTRunner:
    """Runs `n_segments x seg_len` Truncated BPTT with Differentiable Sparse K/V Gradient Replay."""

    def __init__(
        self,
        model: nn.Module,
        seg_len: int = 32_768,
        replay_k: int = 2_048,
        *,
        n_segments: int | None = None,
        replay_enabled: bool = True,
        offload_u_to_cpu: bool = True,
        offload_prior_hidden: bool | None = None,
        replay_indexer_weight: float = 0.1,
        query_sim_chunk: int = 256,
        cfg: BPTTConfig | dict[str, Any] | None = None,
    ) -> None:
        self.model = model
        if isinstance(cfg, dict):
            self.cfg = BPTTConfig.from_dict(cfg)
        elif cfg is not None:
            self.cfg = cfg
        else:
            self.cfg = BPTTConfig(
                seg_len=seg_len,
                n_segments=n_segments,
                replay_k=replay_k,
                replay_enabled=replay_enabled,
                offload_u_to_cpu=offload_u_to_cpu,
                offload_prior_hidden=offload_prior_hidden,
                replay_indexer_weight=replay_indexer_weight,
                query_sim_chunk=query_sim_chunk,
            )

    @staticmethod
    def _extract_query_heads(attn_mod: nn.Module, x_in: torch.Tensor) -> tuple[torch.Tensor, int, int, int]:
        """Project `x_in` (`[B, T, hidden]`) through `q_proj` (+ `q_norm` if present) -> `[B, H, T, D]`."""
        bsz, t_len, _ = x_in.shape
        q_raw = attn_mod.q_proj(x_in)
        k_out = attn_mod.k_proj.out_features
        v_out = attn_mod.v_proj.out_features
        q_norm = getattr(attn_mod, "q_norm", None)
        if q_norm is not None and hasattr(q_norm, "weight"):
            head_dim = int(q_norm.weight.shape[-1])
        else:
            head_dim = int(getattr(attn_mod, "d", None) or getattr(attn_mod, "head_dim", 0))
            if head_dim <= 0:
                kv_h = int(getattr(attn_mod, "kv", None) or getattr(attn_mod, "num_key_value_heads", 1))
                head_dim = k_out // max(kv_h, 1)
        num_kv_heads = k_out // head_dim
        if v_out // head_dim != num_kv_heads:
            raise RuntimeError(f"inconsistent KV head dims: k={k_out}, v={v_out}, head_dim={head_dim}")
        # Gated attention (`q_proj` emits query + gate of size 2 * H * D) vs plain `H * D`
        num_heads = int(getattr(attn_mod, "h", None) or getattr(attn_mod, "num_heads", 0))
        if num_heads <= 0:
            if q_raw.shape[-1] % (2 * head_dim) == 0 and (q_raw.shape[-1] // (2 * head_dim)) >= num_kv_heads:
                num_heads = q_raw.shape[-1] // (2 * head_dim)
            else:
                num_heads = q_raw.shape[-1] // head_dim
        if q_raw.shape[-1] == 2 * num_heads * head_dim:
            q_states, _ = q_raw.view(bsz, t_len, num_heads, 2 * head_dim).chunk(2, dim=-1)
        else:
            q_states = q_raw.view(bsz, t_len, num_heads, head_dim)
        if q_norm is not None:
            q_states = q_norm(q_states)
        return q_states.transpose(1, 2), num_heads, num_kv_heads, head_dim

    def _select_replay_indices(
        self,
        attn_mod: nn.Module,
        q_cur_detached: torch.Tensor,
        k_prior_cached: torch.Tensor,
        rot_dim: int,
        k_sel: int,
    ) -> torch.Tensor:
        """Select Top-`k_sel` prior token indices `[B, G, k_sel]` per KV head using LSA or NoPE similarity."""
        bsz, h_cnt, t_seg, dh = q_cur_detached.shape
        g_cnt, t_prior = k_prior_cached.shape[1], k_prior_cached.shape[2]
        hg = h_cnt // g_cnt
        lsa_ix = getattr(attn_mod, "lsa_indexer", None)
        chunk = max(1, int(self.cfg.query_sim_chunk))

        with torch.no_grad():
            if lsa_ix is not None:
                qz = lsa_ix.query_latent(q_cur_detached)      # [B, G | H, T_seg, d_idx]
                kz = lsa_ix.key_latent(k_prior_cached)        # [B, G, T_prior, d_idx]
                kz_m = kz.to(qz.dtype)
                sim = qz.new_full((bsz, g_cnt, t_prior), float("-inf"))
                if qz.shape[1] != g_cnt:
                    hg_q = qz.shape[1] // g_cnt
                    qz_g = qz.view(bsz, g_cnt, hg_q, t_seg, -1)
                    for c0 in range(0, t_seg, chunk):
                        qc = qz_g[:, :, :, c0 : c0 + chunk]
                        sc = torch.einsum("bghcd,bgtd->bghct", qc, kz_m).amax(dim=(2, 3))
                        sim = torch.maximum(sim, sc)
                else:
                    for c0 in range(0, t_seg, chunk):
                        qc = qz[:, :, c0 : c0 + chunk]
                        sc = torch.einsum("bgcd,bgtd->bgct", qc, kz_m).amax(dim=2)
                        sim = torch.maximum(sim, sc)
            else:
                lo = rot_dim if rot_dim < dh else 0
                scale_nope = 1.0 / math.sqrt(max(1, dh - lo))
                q_nope = upcast(q_cur_detached[..., lo:]).view(bsz, g_cnt, hg, t_seg, -1)
                k_nope = upcast(k_prior_cached[..., lo:])
                sim = q_nope.new_full((bsz, g_cnt, t_prior), float("-inf"))
                for c0 in range(0, t_seg, chunk):
                    qc = q_nope[:, :, :, c0 : c0 + chunk]
                    sc = (torch.einsum("bghcd,bgtd->bghct", qc, k_nope) * scale_nope).amax(dim=(2, 3))
                    sim = torch.maximum(sim, sc)

            top_idx = sim.topk(k_sel, dim=-1).indices
            return top_idx.sort(dim=-1).values

    @staticmethod
    def _invoke_segment_loss(
        loss_fn: Callable[..., torch.Tensor | None] | None,
        seg_idx: int,
        seg_logits: torch.Tensor,
        seg_ids: torch.Tensor,
        seg_targets: torch.Tensor | None,
    ) -> torch.Tensor | None:
        if loss_fn is not None:
            sig = inspect.signature(loss_fn)
            params = [
                p
                for p in sig.parameters.values()
                if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
            ]
            has_varargs = any(p.kind == inspect.Parameter.VAR_POSITIONAL for p in sig.parameters.values())
            if has_varargs or len(params) >= 4:
                return loss_fn(seg_idx, seg_logits, seg_ids, seg_targets)
            if len(params) == 3:
                return loss_fn(seg_idx, seg_logits, seg_targets if seg_targets is not None else seg_ids)
            if len(params) == 2:
                return loss_fn(seg_idx, seg_logits)
            return loss_fn(seg_logits)

        if seg_targets is None or seg_targets.numel() == 0:
            return None
        valid = seg_targets != -100
        if not bool(valid.any()):
            return None
        flat_logits = seg_logits.reshape(-1, seg_logits.shape[-1]).float()
        flat_targets = seg_targets.reshape(-1)
        return F.cross_entropy(flat_logits, flat_targets, ignore_index=-100)

    def run_step(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
        loss_fn: Callable[..., torch.Tensor | None] | None = None,
        *,
        backward: bool = True,
        hssd_aux_weight: float = 0.0,
        lsa_aux_weight: float = 1.0,
        loop_aux_weight: float = 1.0,
        ce_chunk_size: int = 2048,
    ) -> BPTTStepResult:
        """Execute segmented forward (+ per-segment backward with K/V replay) over `input_ids` `[B, T]`."""
        if input_ids.ndim != 2:
            raise ValueError(f"input_ids must have shape [B, T], got {tuple(input_ids.shape)}")
        bsz, t_total = input_ids.shape
        seg_len = self.cfg.resolve_seg_len(t_total)
        n_segs = math.ceil(t_total / seg_len)
        target_full = input_ids if (labels is None and loss_fn is None) else labels

        phys_list, _text_cfg, loop_ctl = physical_layers(self.model)
        full_layers: list[tuple[int, nn.Module]] = []
        for phys_idx, layer_mod in phys_list:
            lt = getattr(layer_mod, "layer_type", None)
            if lt == FULL or (lt is None and hasattr(layer_mod, "self_attn")):
                full_layers.append((phys_idx, layer_mod.self_attn))

        prior_records: dict[int, list[_PriorSegRecord]] = {p: [] for p, _ in full_layers}
        cur_seg_records: dict[int, _PriorSegRecord] = {}
        seg_replay_aux: list[torch.Tensor] = []
        recorded_indices: dict[tuple[int, int], torch.Tensor] = {}
        replayed_counts: list[int] = []
        seg_losses: list[float] = []
        active_seg_state = {"seg_idx": 0}
        proxy_holder: list[_ReplayKVCacheProxy | None] = [None]
        hooks: list[Any] = []

        def make_attn_pre_hook(phys_idx: int):
            def pre_hook(attn_mod: nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
                x_cur: torch.Tensor = args[0] if len(args) > 0 else kwargs["hidden_states"]
                pos_emb = args[1] if len(args) > 1 else kwargs.get("position_embeddings")
                pkv = args[3] if len(args) > 3 else kwargs.get("past_key_values")
                if pos_emb is None:
                    return

                # Skip pass >= 2 when kv_share=True (Pass 1 already computed live replay for this slot)
                if loop_ctl is not None and getattr(loop_ctl, "_cur", None) is not None:
                    _ctx, r_pass = loop_ctl._cur
                    if r_pass >= 2 and getattr(loop_ctl.spec, "kv_share", True):
                        return
                if pkv is not None and type(pkv).__name__ == "SharedKVView":
                    return

                cos_cur, sin_cur = pos_emb
                rot_dim = int(cos_cur.shape[-1])
                s_idx = active_seg_state["seg_idx"]
                cache_slot = int(getattr(attn_mod, "layer_idx", phys_idx))
                proxy = proxy_holder[0]

                if (
                    s_idx > 0
                    and self.cfg.replay_enabled
                    and self.cfg.replay_k > 0
                    and proxy is not None
                    and prior_records[phys_idx]
                    and torch.is_grad_enabled()
                ):
                    k_cached = _extract_cached_keys(proxy.base_cache, cache_slot)
                    t_prior = sum(rec.u.shape[1] for rec in prior_records[phys_idx])
                    if k_cached is not None and k_cached.shape[-2] >= t_prior > 0:
                        k_prior = k_cached[:, :, :t_prior, :].detach()
                        k_sel = min(int(self.cfg.replay_k), t_prior)

                        # Pause LoopController._cur while calling projections on replay tensors
                        saved_cur = getattr(loop_ctl, "_cur", None) if loop_ctl is not None else None
                        if loop_ctl is not None:
                            loop_ctl._cur = None
                        try:
                            with torch.no_grad():
                                q_det, _nh, num_kv_heads, head_dim = self._extract_query_heads(
                                    attn_mod, x_cur.detach()
                                )
                            top_idx = self._select_replay_indices(
                                attn_mod, q_det, k_prior, rot_dim=rot_dim, k_sel=k_sel
                            )
                            if not _in_backward():
                                recorded_indices[(s_idx, phys_idx)] = top_idx.detach().cpu()

                            dev, dt = x_cur.device, x_cur.dtype
                            u_all = torch.cat([r.u for r in prior_records[phys_idx]], dim=1)
                            cos_all = torch.cat([r.cos for r in prior_records[phys_idx]], dim=1)
                            sin_all = torch.cat([r.sin for r in prior_records[phys_idx]], dim=1)

                            # Build per-batch union of selected indices across KV heads for single k/v_proj call
                            union_rows: list[torch.Tensor] = []
                            max_u = 0
                            for b_i in range(bsz):
                                u_b = torch.unique(top_idx[b_i].reshape(-1), sorted=True)
                                union_rows.append(u_b)
                                max_u = max(max_u, int(u_b.numel()))
                            uniq_pad = top_idx.new_zeros((bsz, max_u))
                            for b_i, u_b in enumerate(union_rows):
                                uniq_pad[b_i, : u_b.numel()] = u_b
                                if u_b.numel() < max_u:
                                    uniq_pad[b_i, u_b.numel() :] = u_b[-1]

                            idx_cpu_or_dev = uniq_pad.to(u_all.device)
                            u_sel = u_all.gather(
                                1, idx_cpu_or_dev.unsqueeze(-1).expand(-1, -1, u_all.shape[-1])
                            ).to(device=dev, dtype=dt)
                            cos_sel = cos_all.gather(
                                1, idx_cpu_or_dev.unsqueeze(-1).expand(-1, -1, rot_dim)
                            ).to(device=dev, dtype=cos_cur.dtype)
                            sin_sel = sin_all.gather(
                                1, idx_cpu_or_dev.unsqueeze(-1).expand(-1, -1, rot_dim)
                            ).to(device=dev, dtype=sin_cur.dtype)

                            # Live differentiable re-projection through k_proj, k_norm, and v_proj!
                            k_proj_sel = attn_mod.k_proj(u_sel).view(bsz, max_u, num_kv_heads, head_dim)
                            k_norm = getattr(attn_mod, "k_norm", None)
                            if k_norm is not None:
                                k_proj_sel = k_norm(k_proj_sel)
                            k_union = _apply_partial_rope(k_proj_sel.transpose(1, 2), cos_sel, sin_sel)
                            v_union = attn_mod.v_proj(u_sel).view(bsz, max_u, num_kv_heads, head_dim).transpose(1, 2)

                            local_pos = torch.searchsorted(
                                uniq_pad.unsqueeze(1).expand(-1, num_kv_heads, -1).contiguous(),
                                top_idx.contiguous(),
                            ).clamp(max=max(0, max_u - 1))
                            gather_idx = local_pos.unsqueeze(-1).expand(-1, -1, -1, head_dim)
                            k_live = k_union.gather(2, gather_idx)
                            v_live = v_union.gather(2, gather_idx)

                            # Also re-project selected keys through `lsa_indexer.k_proj` (`W_k_idx`) if present
                            lsa_ix = getattr(attn_mod, "lsa_indexer", None)
                            if (
                                lsa_ix is not None
                                and self.cfg.replay_indexer_weight > 0
                                and (lsa_ix.wk.requires_grad or lsa_ix.wq.requires_grad or k_live.requires_grad)
                            ):
                                kz_live = lsa_ix.k_proj(k_live)  # [B, G, K_sel, d_idx]
                                q_live_for_idx, _, _, _ = self._extract_query_heads(attn_mod, x_cur.detach())
                                qz_live = lsa_ix.query_latent(q_live_for_idx)  # [B, G | H, T_seg, d_idx]
                                per_head_q = qz_live.shape[1] != num_kv_heads
                                lo = lsa_ix.lo
                                hg = q_det.shape[1] // num_kv_heads
                                qn_t = upcast(q_det[..., lo:]).view(bsz, num_kv_heads, hg, -1, head_dim - lo)
                                kn_t = upcast(k_live.detach()[..., lo:])
                                kl_idx = _chunked_replay_indexer_kl(
                                    qn_t,
                                    kn_t,
                                    qz_live,
                                    kz_live,
                                    scale_nope=1.0 / math.sqrt(max(1, head_dim - lo)),
                                    temperature=float(lsa_ix.cfg.temperature),
                                    per_head_q=per_head_q,
                                    chunk_size=self.cfg.query_sim_chunk,
                                )
                                if not _in_backward():
                                    seg_replay_aux.append(float(self.cfg.replay_indexer_weight) * kl_idx)

                            proxy.set_pending(cache_slot, top_idx, k_live, v_live)
                        finally:
                            if loop_ctl is not None:
                                loop_ctl._cur = saved_cur

                if not _in_backward():
                    u_store = x_cur.detach().to("cpu") if self.cfg.offload_u_to_cpu else x_cur.detach()
                    c_store = cos_cur.detach().to("cpu") if self.cfg.offload_u_to_cpu else cos_cur.detach()
                    s_store = sin_cur.detach().to("cpu") if self.cfg.offload_u_to_cpu else sin_cur.detach()
                    cur_seg_records[phys_idx] = _PriorSegRecord(u_store, c_store, s_store)

            return pre_hook

        for phys_idx, attn_mod in full_layers:
            hooks.append(attn_mod.register_forward_pre_hook(make_attn_pre_hook(phys_idx), with_kwargs=True))

        # Clear any stale LSA / HSSD pending losses before starting
        tm, lm_head = _resolve_bptt_backbone_and_head(self.model)
        for mod in tm.modules():
            lsa_ix = getattr(mod, "lsa_indexer", None)
            if lsa_ix is not None:
                lsa_ix.pop_losses()
            hssd_ad = getattr(mod, "hssd_adapter", None)
            if hssd_ad is not None and callable(getattr(hssd_ad, "clear_aux", None)):
                hssd_ad.clear_aux()

        can_pre_head = (
            loss_fn is None
            and tm is not self.model
            and lm_head is not None
            and isinstance(getattr(lm_head, "weight", None), torch.Tensor)
            and getattr(lm_head, "bias", None) is None
            and lm_head.weight.dtype != torch.float64
        )
        bypass_mask = _can_bypass_bptt_causal_mask(tm)
        no_mask_dict: dict[str, None] = {"full_attention": None, "linear_attention": None}

        try:
            base_cache: Any = None
            for s_idx in range(n_segs):
                active_seg_state["seg_idx"] = s_idx
                cur_seg_records.clear()
                seg_replay_aux.clear()

                s0 = s_idx * seg_len
                s1 = min(t_total, (s_idx + 1) * seg_len)
                seg_ids = input_ids[:, s0:s1]

                seg_targets: torch.Tensor | None = None
                if target_full is not None:
                    seg_targets = (
                        target_full[:, s0 + 1 : s1 + 1]
                        if s1 < t_total
                        else target_full[:, s0 + 1 : t_total]
                    )

                if base_cache is None:
                    base_cache = _maybe_create_initial_cache(tm)
                if base_cache is not None:
                    proxy = _ReplayKVCacheProxy(base_cache)
                    proxy_holder[0] = proxy
                    pkv_arg: Any = proxy
                else:
                    proxy_holder[0] = None
                    pkv_arg = None

                if can_pre_head:
                    assert lm_head is not None
                    tm_kw: dict[str, Any] = {
                        "input_ids": seg_ids,
                        "past_key_values": pkv_arg,
                        "use_cache": True,
                    }
                    if bypass_mask and s_idx > 0:
                        tm_kw["attention_mask"] = no_mask_dict
                    tm_out = tm(**tm_kw)
                    base_cache = (
                        proxy_holder[0].base_cache
                        if proxy_holder[0] is not None
                        else getattr(tm_out, "past_key_values", None)
                    )
                    seg_hidden = tm_out.last_hidden_state if hasattr(tm_out, "last_hidden_state") else tm_out[0]
                    hidden_for_loss = seg_hidden if s1 < t_total else seg_hidden[:, :-1]
                    if (
                        seg_targets is not None
                        and seg_targets.numel() > 0
                        and hidden_for_loss.shape[1] == seg_targets.shape[1]
                        and bool((seg_targets != -100).any())
                    ):
                        seg_loss, _ = lm_loss(
                            hidden_for_loss,
                            lm_head.weight,
                            seg_targets,
                            impl="chunked",
                            chunk_size=int(ce_chunk_size),
                            shift=False,
                            ignore_index=-100,
                        )
                    else:
                        seg_loss = None
                else:
                    fwd_kw: dict[str, Any] = {"past_key_values": pkv_arg, "use_cache": True}
                    if bypass_mask and s_idx > 0:
                        fwd_kw["attention_mask"] = no_mask_dict
                    out = self.model(seg_ids, **fwd_kw)
                    base_cache = (
                        proxy_holder[0].base_cache
                        if proxy_holder[0] is not None
                        else getattr(out, "past_key_values", None)
                    )
                    logits = out.logits
                    logits_for_loss = logits if (s1 < t_total or loss_fn is not None) else logits[:, :-1]
                    seg_loss = self._invoke_segment_loss(loss_fn, s_idx, logits_for_loss, seg_ids, seg_targets)

                # Collect per-segment LSA, HSSD, loop & replay indexer auxiliary losses and drop unconsumed
                # HSSD aux terms BEFORE backward so live `bud_loss` tensors never pin checkpoint frames!
                lsa_terms = [
                    loss_t
                    for mod in tm.modules()
                    if getattr(mod, "lsa_indexer", None) is not None
                    for loss_t in mod.lsa_indexer.pop_losses()
                ]
                hssd_terms: list[torch.Tensor] = []
                for mod in tm.modules():
                    hssd_ad = getattr(mod, "hssd_adapter", None)
                    if hssd_ad is not None:
                        pending_h = getattr(hssd_ad, "_pending_aux", None)
                        if pending_h:
                            hssd_terms.extend(pending_h)
                        if callable(getattr(hssd_ad, "clear_aux", None)):
                            hssd_ad.clear_aux()
                loop_aux = loop_ctl.aux_loss() if loop_ctl is not None else None

                total_seg_loss: torch.Tensor | None = seg_loss
                extra_terms: list[torch.Tensor] = list(seg_replay_aux)
                if lsa_terms and lsa_aux_weight > 0:
                    extra_terms.extend(float(lsa_aux_weight) * t for t in lsa_terms)
                if hssd_terms and hssd_aux_weight > 0:
                    extra_terms.append(float(hssd_aux_weight) * torch.stack([t.float() for t in hssd_terms]).mean())
                if loop_aux is not None and loop_aux_weight > 0:
                    extra_terms.append(float(loop_aux_weight) * loop_aux)
                lsa_terms.clear()
                hssd_terms.clear()
                seg_replay_aux.clear()
                del loop_aux

                if total_seg_loss is not None and extra_terms:
                    aux_sum = torch.stack([t.to(total_seg_loss.dtype) for t in extra_terms]).sum()
                    total_seg_loss = total_seg_loss + aux_sum
                extra_terms.clear()

                if total_seg_loss is not None:
                    seg_losses.append(float(total_seg_loss.detach()))
                    if backward and total_seg_loss.requires_grad:
                        (total_seg_loss / float(n_segs)).backward()
                else:
                    seg_losses.append(0.0)

                # Count replayed positions on this segment (from first full-attn layer if any)
                if s_idx > 0 and full_layers and (s_idx, full_layers[0][0]) in recorded_indices:
                    replayed_counts.append(int(recorded_indices[(s_idx, full_layers[0][0])].shape[-1]))
                else:
                    replayed_counts.append(0)

                # Commit pre-attention `u_seg` records and detach carried recurrent/KV states!
                for p_id, rec in cur_seg_records.items():
                    prior_records[p_id].append(rec)
                if proxy_holder[0] is not None:
                    proxy_holder[0].reset_segment()
                if base_cache is not None:
                    _detach_nested(base_cache)

            active_losses = [val for val in seg_losses if val != 0.0]
            mean_loss = sum(active_losses) / len(active_losses) if active_losses else 0.0
            stats = {
                "bptt/num_segments": float(n_segs),
                "bptt/seg_len": float(seg_len),
                "bptt/replay_k": float(self.cfg.replay_k if self.cfg.replay_enabled else 0),
                "bptt/mean_replayed_tokens": (
                    sum(replayed_counts[1:]) / max(1, len(replayed_counts) - 1) if n_segs > 1 else 0.0
                ),
            }
            return BPTTStepResult(
                total_loss=mean_loss,
                segment_losses=seg_losses,
                num_segments=n_segs,
                replayed_tokens_per_seg=replayed_counts,
                replayed_indices=recorded_indices,
                stats=stats,
            )
        finally:
            for h in hooks:
                h.remove()
            prior_records.clear()
            cur_seg_records.clear()

    run = run_step
    __call__ = run_step
