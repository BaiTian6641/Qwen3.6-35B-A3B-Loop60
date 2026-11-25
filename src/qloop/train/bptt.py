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

from ..model._cache_compat import cache_kwarg_name, extract_cache_kwarg, extract_output_cache
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
    offload_prior_hidden: bool | str | None = None
    async_offload: bool = True
    hbm_headroom_gib: float = 8.0
    loss_impl: str = "auto"
    replay_indexer_weight: float = 0.1
    query_sim_chunk: int = 256

    def __post_init__(self) -> None:
        raw = self.offload_prior_hidden
        if isinstance(raw, str):
            norm = raw.strip().lower()
            if norm == "auto":
                self.offload_prior_hidden = "auto"
                self.offload_u_to_cpu = False
            elif norm in ("true", "1", "yes"):
                self.offload_prior_hidden = True
                self.offload_u_to_cpu = True
            elif norm in ("false", "0", "no"):
                self.offload_prior_hidden = False
                self.offload_u_to_cpu = False
            else:
                raise ValueError(
                    f"bptt.offload_prior_hidden must be bool or 'auto', got {raw!r}"
                )
        elif raw is not None:
            self.offload_prior_hidden = bool(raw)
            self.offload_u_to_cpu = bool(raw)
        else:
            self.offload_prior_hidden = bool(self.offload_u_to_cpu)

    def resolve_offload_to_cpu(self, device: torch.device, estimated_bytes: int) -> bool:
        """Decide whether prior-segment hidden states offload to CPU or stay HBM-resident."""
        if self.offload_prior_hidden == "auto":
            if device.type != "cuda" or not torch.cuda.is_available():
                return False
            try:
                free_b, _total_b = torch.cuda.mem_get_info(device)
                cached_free = max(
                    0,
                    int(torch.cuda.memory_reserved(device)) - int(torch.cuda.memory_allocated(device)),
                )
                avail_b = int(free_b) + cached_free
                headroom_b = int(max(0.0, float(self.hbm_headroom_gib)) * (1024 ** 3))
                return bool(estimated_bytes + headroom_b > avail_b)
            except Exception:
                return True
        return bool(self.offload_u_to_cpu)

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
    ce_loss: float = 0.0
    aux_loss: float = 0.0
    segment_ce_losses: list[float] = field(default_factory=list)
    segment_aux_losses: list[float] = field(default_factory=list)
    segment_valid_tokens: list[int] = field(default_factory=list)


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
    detach_fn = getattr(obj, "detach", None)
    if callable(detach_fn) and not isinstance(obj, (type, nn.Module)):
        try:
            detached = detach_fn()
            if detached is not None:
                return detached
        except Exception:
            pass
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
            if (
                isinstance(val, (torch.Tensor, list, dict, tuple))
                or hasattr(val, "__dict__")
                or hasattr(type(val), "__slots__")
                or callable(getattr(val, "detach", None))
            ):
                try:
                    setattr(obj, attr, _detach_nested(val, seen))
                except Exception:
                    pass
    for cls in type(obj).__mro__:
        slots = getattr(cls, "__slots__", ())
        if isinstance(slots, str):
            slots = (slots,)
        for slot_name in slots:
            if slot_name in ("__dict__", "__weakref__") or slot_name.startswith("__"):
                continue
            if not hasattr(obj, slot_name):
                continue
            val = getattr(obj, slot_name)
            if (
                isinstance(val, (torch.Tensor, list, dict, tuple))
                or hasattr(val, "__dict__")
                or hasattr(type(val), "__slots__")
                or callable(getattr(val, "detach", None))
            ):
                try:
                    setattr(obj, slot_name, _detach_nested(val, seen))
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


def _clone_detached_state(obj: Any, *, allow_zero_copy_hssd: bool = False) -> Any:
    """Return an independent detached clone of a conv/recurrent state (safe against in-place `.copy_`)."""
    if obj is None:
        return None
    if isinstance(obj, torch.Tensor):
        return obj.detach().clone()
    if allow_zero_copy_hssd and type(obj).__name__ == "HSSDCacheState":
        detach_fn = getattr(obj, "detach", None)
        if callable(detach_fn):
            return detach_fn()
    clone_fn = getattr(obj, "clone", None)
    if callable(clone_fn):
        return _detach_nested(clone_fn())
    try:
        return _detach_nested(copy.deepcopy(obj))
    except Exception:
        return _detach_nested(obj)


def _call_cache_method(fn: Callable[..., Any], *pos_args: Any, **kw_args: Any) -> Any:
    """Invoke a cache method while filtering optional kwargs unsupported by its HF/Toy signature."""
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return fn(*pos_args, **kw_args)
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return fn(*pos_args, **kw_args)
    filtered = {k: v for k, v in kw_args.items() if k in params}
    return fn(*pos_args, **filtered)


class _SlotStateListProxy:
    """Checkpoint-recomputation-safe list view over `conv_states` / `recurrent_states`.

    Clones plain tensors on snapshot so in-place `.copy_()` mutations during forward never corrupt
    the pre-segment state seen during `torch.utils.checkpoint` backward recomputation.
    """

    @staticmethod
    def _detach_slot(val: Any) -> Any:
        if isinstance(val, torch.Tensor):
            return val.detach()
        detach_fn = getattr(val, "detach", None)
        if callable(detach_fn) and not isinstance(val, (type, nn.Module)):
            return detach_fn()
        return val

    def __init__(
        self,
        backing: list[Any],
        *,
        zero_copy_hssd: bool = False,
        readonly: bool = False,
    ) -> None:
        self._backing = backing
        self._zero_copy_hssd = bool(zero_copy_hssd)
        self._readonly = bool(readonly)
        self._snapshot: list[Any] = [
            _clone_detached_state(x, allow_zero_copy_hssd=self._zero_copy_hssd) for x in backing
        ]
        self._fwd_overlay: dict[int, Any] = {}

    def _sync_backing(self, min_idx: int = -1) -> None:
        target = max(len(self._backing), min_idx + 1)
        if len(self._snapshot) < target:
            for j in range(len(self._snapshot), min(len(self._backing), target)):
                self._snapshot.append(
                    _clone_detached_state(self._backing[j], allow_zero_copy_hssd=self._zero_copy_hssd)
                )
            while len(self._snapshot) < target:
                self._snapshot.append(None)

    def __len__(self) -> int:
        return len(self._backing)

    def extend(self, values: Any) -> None:
        if _in_backward() or self._readonly:
            return
        self._sync_backing()
        vals = list(values)
        self._backing.extend(vals)
        self._snapshot.extend(
            _clone_detached_state(x, allow_zero_copy_hssd=self._zero_copy_hssd) for x in vals
        )

    def _norm_key(self, idx: int) -> int:
        return idx if idx >= 0 else len(self._backing) + idx

    def __getitem__(self, idx: Any) -> Any:
        self._sync_backing(idx if isinstance(idx, int) and idx >= 0 else -1)
        if _in_backward():
            val = self._snapshot[idx]
            if isinstance(idx, slice):
                return [x.clone() if isinstance(x, torch.Tensor) else x for x in val]
            return val.clone() if isinstance(val, torch.Tensor) else val
        if self._readonly:
            if isinstance(idx, int):
                nk = self._norm_key(idx)
                if nk in self._fwd_overlay:
                    return self._fwd_overlay[nk]
                val = self._snapshot[idx]
                if isinstance(val, torch.Tensor):
                    cloned = val.clone()
                    self._fwd_overlay[nk] = cloned
                    return cloned
                return val
            if isinstance(idx, slice):
                rng = range(*idx.indices(len(self._backing)))
                return [self[i] for i in rng]
            return self._snapshot[idx]
        return self._backing[idx]

    def __setitem__(self, idx: Any, val: Any) -> None:
        if _in_backward():
            return
        self._sync_backing(idx if isinstance(idx, int) and idx >= 0 else -1)
        det = self._detach_slot(val)
        if self._readonly:
            if isinstance(idx, int):
                self._fwd_overlay[self._norm_key(idx)] = det
            return
        self._backing[idx] = det

    def reset_snapshot(self) -> None:
        self._snapshot = [
            _clone_detached_state(x, allow_zero_copy_hssd=self._zero_copy_hssd) for x in self._backing
        ]
        self._fwd_overlay.clear()


class _SnapshotFlagContainer:
    """Checkpoint-safe wrapper around `LinearAttentionLayer.has_previous_state` (dict or list)."""

    def __init__(self, backing: Any, *, readonly: bool = False) -> None:
        self._backing = backing
        self._readonly = bool(readonly)
        self._snapshot: Any = None
        self._overlay: dict[Any, bool] = {}
        self.reset_snapshot()

    def reset_snapshot(self) -> None:
        if isinstance(self._backing, dict):
            self._snapshot = {k: bool(v) for k, v in self._backing.items()}
        elif isinstance(self._backing, list):
            self._snapshot = [bool(v) for v in self._backing]
        else:
            self._snapshot = bool(self._backing)
        self._overlay.clear()

    def __bool__(self) -> bool:
        if _in_backward():
            if isinstance(self._snapshot, dict):
                return any(self._snapshot.values())
            if isinstance(self._snapshot, list):
                return any(self._snapshot)
            return bool(self._snapshot)
        if self._readonly and self._overlay:
            return any(self._overlay.values()) or bool(self._snapshot)
        return bool(self._backing)

    def __len__(self) -> int:
        return len(self._backing)

    def __contains__(self, key: Any) -> bool:
        return key in self._backing

    def get(self, key: Any, default: Any = False) -> Any:
        if _in_backward():
            if isinstance(self._snapshot, dict):
                return self._snapshot.get(key, default)
            if isinstance(self._snapshot, list) and isinstance(key, int) and 0 <= key < len(self._snapshot):
                return self._snapshot[key]
            return default
        if self._readonly and key in self._overlay:
            return self._overlay[key]
        if isinstance(self._backing, dict):
            return self._backing.get(key, default)
        if isinstance(self._backing, list) and isinstance(key, int) and 0 <= key < len(self._backing):
            return self._backing[key]
        return default

    def __getitem__(self, key: Any) -> Any:
        if _in_backward():
            if isinstance(self._snapshot, dict):
                return self._snapshot.get(key, False)
            return self._snapshot[key]
        if self._readonly and key in self._overlay:
            return self._overlay[key]
        return self._backing[key]

    def __setitem__(self, key: Any, val: Any) -> None:
        if _in_backward():
            return
        if isinstance(self._snapshot, dict) and key not in self._snapshot:
            self._snapshot[key] = bool(self._backing.get(key, False))
        if self._readonly:
            self._overlay[key] = bool(val)
            return
        self._backing[key] = val


class _ProxyHasPreviousState:
    """Dual-mode checkpoint-safe `has_previous_state` (property bool OR callable `(layer_idx, ...)`)."""

    def __init__(self, proxy: _ReplayKVCacheProxy) -> None:
        self._proxy = proxy

    def __bool__(self) -> bool:
        return self._proxy._query_has_previous_state_global()

    def __call__(self, layer_idx: int | None = None, state_idx: int = 0, **kwargs: Any) -> bool:
        del kwargs
        if layer_idx is None:
            return self._proxy._query_has_previous_state_global()
        return self._proxy._query_has_previous_state_layer(int(layer_idx), int(state_idx))


def _read_layer_has_prev(lyr: Any, state_idx: int = 0) -> bool | None:
    hps = getattr(lyr, "has_previous_state", None)
    if hps is None:
        return None
    if isinstance(hps, dict):
        return bool(hps.get(state_idx, False))
    if isinstance(hps, (list, tuple)):
        return bool(hps[state_idx]) if 0 <= state_idx < len(hps) else False
    if callable(hps):
        try:
            return bool(_call_cache_method(hps, state_idx=state_idx))
        except Exception:
            return bool(hps())
    return bool(hps)


def _read_base_has_previous_state(base: Any, layer_idx: int, state_idx: int = 0) -> bool:
    """Query `has_previous_state` on `base` across HF 5.2-5.24, HSSDHybridCache, and ToyHybridCache."""
    layers = getattr(base, "layers", None)
    if isinstance(layers, list) and 0 <= layer_idx < len(layers):
        lyr = layers[layer_idx]
        flag = _read_layer_has_prev(lyr, state_idx)
        if flag is not None:
            return flag
        l_cs = getattr(lyr, "conv_states", None)
        l_rs = getattr(lyr, "recurrent_states", None)
        if l_cs is not None or l_rs is not None:
            c_ok = isinstance(l_cs, list) and state_idx < len(l_cs) and l_cs[state_idx] is not None
            r_ok = isinstance(l_rs, list) and state_idx < len(l_rs) and l_rs[state_idx] is not None
            return bool(c_ok or r_ok)
        return False
    hps = getattr(base, "has_previous_state", None)
    if callable(hps):
        try:
            return bool(_call_cache_method(hps, layer_idx, state_idx=state_idx))
        except (TypeError, AttributeError):
            try:
                return bool(hps())
            except Exception:
                pass
    if hps is not None and not callable(hps):
        return bool(hps)
    cs = getattr(base, "conv_states", None)
    rs = getattr(base, "recurrent_states", None)
    c_ok = isinstance(cs, list) and 0 <= layer_idx < len(cs) and cs[layer_idx] is not None
    r_ok = isinstance(rs, list) and 0 <= layer_idx < len(rs) and rs[layer_idx] is not None
    return bool(c_ok or r_ok)


def _read_base_conv_state(base: Any, layer_idx: int, state_idx: int = 0) -> tuple[bool, torch.Tensor | None]:
    """Read `(had_prev, conv_state)` from `base` across HF 5.x `layers[i]`, `get_conv_state`, and list caches."""
    layers = getattr(base, "layers", None)
    if isinstance(layers, list) and 0 <= layer_idx < len(layers):
        lyr = layers[layer_idx]
        flag = _read_layer_has_prev(lyr, state_idx)
        if flag is False:
            return False, None
        get_cs_lyr = getattr(lyr, "get_conv_state", None)
        if callable(get_cs_lyr):
            val = _call_cache_method(get_cs_lyr, state_idx=state_idx)
            if isinstance(val, torch.Tensor):
                return True, val
            return False, None
        l_cs = getattr(lyr, "conv_states", None)
        if isinstance(l_cs, list) and 0 <= state_idx < len(l_cs):
            val = l_cs[state_idx]
            if isinstance(val, torch.Tensor):
                return True, val
            return False, None
        if isinstance(l_cs, torch.Tensor):
            return True, l_cs
    get_cs = getattr(base, "get_conv_state", None)
    if callable(get_cs):
        val = _call_cache_method(get_cs, layer_idx, state_idx=state_idx)
        if isinstance(val, torch.Tensor):
            return True, val
        return False, None
    cs = getattr(base, "conv_states", None)
    if isinstance(cs, list) and 0 <= layer_idx < len(cs):
        val = cs[layer_idx]
        if isinstance(val, torch.Tensor):
            return True, val
    return False, None


def _read_base_recurrent_state(base: Any, layer_idx: int, state_idx: int = 0) -> Any:
    """Read `recurrent_state` from `base` across HF 5.x `layers[i]`, `get_recurrent_state`, and list caches."""
    get_rs = getattr(base, "get_recurrent_state", None)
    if callable(get_rs):
        return _call_cache_method(get_rs, layer_idx, state_idx=state_idx)
    layers = getattr(base, "layers", None)
    if isinstance(layers, list) and 0 <= layer_idx < len(layers):
        lyr = layers[layer_idx]
        flag = _read_layer_has_prev(lyr, state_idx)
        if flag is False:
            return None
        get_rs_lyr = getattr(lyr, "get_recurrent_state", None)
        if callable(get_rs_lyr):
            return _call_cache_method(get_rs_lyr, state_idx=state_idx)
        l_rs = getattr(lyr, "recurrent_states", None)
        if isinstance(l_rs, list) and 0 <= state_idx < len(l_rs):
            return l_rs[state_idx]
        if l_rs is not None and not isinstance(l_rs, list):
            return l_rs
    rs = getattr(base, "recurrent_states", None)
    if isinstance(rs, list) and 0 <= layer_idx < len(rs):
        return rs[layer_idx]
    return None


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
        kn_t_tr = kn_t.unsqueeze(2).transpose(-1, -2)
        kz_d_tr = kz_d.unsqueeze(2).transpose(-1, -2) if per_head_q else kz_d.transpose(-1, -2)
        g_qz = torch.zeros_like(qz_d)
        g_kz = torch.zeros_like(kz_d)
        kl_sum = qz_d.new_zeros(())
        for c0 in range(0, t_seg, chunk):
            c1 = min(t_seg, c0 + chunk)
            raw_sc = torch.matmul(qn_t[:, :, :, c0:c1], kn_t_tr) * scale_nope
            if per_head_q:
                tgt_p = torch.softmax(raw_sc, dim=-1)  # [B, G, Hg, C, K]
                qz_c = qz_d[:, :, :, c0:c1]
                pred_logits = torch.matmul(qz_c, kz_d_tr) * inv_temp
                log_pred = torch.log_softmax(pred_logits, dim=-1)
                kl_sum = kl_sum + (tgt_p * (torch.log(tgt_p.clamp_min(1e-30)) - log_pred)).sum()
                d_logits = (log_pred.exp() - tgt_p) * (inv_temp / denom)
                g_qz[:, :, :, c0:c1] = torch.matmul(d_logits, kz_d.unsqueeze(2))
                g_kz.add_(torch.matmul(d_logits.transpose(-1, -2), qz_c).sum(dim=2))
            else:
                tgt_p = torch.softmax(raw_sc, dim=-1).mean(dim=2)  # [B, G, C, K]
                qz_c = qz_d[:, :, c0:c1]
                pred_logits = torch.matmul(qz_c, kz_d_tr) * inv_temp
                log_pred = torch.log_softmax(pred_logits, dim=-1)
                kl_sum = kl_sum + (tgt_p * (torch.log(tgt_p.clamp_min(1e-30)) - log_pred)).sum()
                d_logits = (log_pred.exp() - tgt_p) * (inv_temp / denom)
                g_qz[:, :, c0:c1] = torch.matmul(d_logits, kz_d)
                g_kz.add_(torch.matmul(d_logits.transpose(-1, -2), qz_c))

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
    """Create an empty hybrid KV/recurrent cache when possible so Segment 0 is checkpoint-safe.

    Checks `ToyHybridCache` for toy models, then `Qwen3_5MoeDynamicCache` / `Qwen3_5DynamicCache`
    (HF 5.2-5.19), then unified `DynamicCache(config=cfg)` (HF >= 5.20 / 5.24.0).
    """
    cfg = getattr(tm, "config", None)
    mtype = str(getattr(cfg, "model_type", "") or "").lower() if cfg is not None else ""
    if cfg is not None and mtype.startswith("toy"):
        from ..model.toy import ToyHybridCache

        return ToyHybridCache(cfg)
    if cfg is not None:
        is_moe = (
            ("moe" in mtype)
            or ("moe" in type(tm).__name__.lower())
            or (getattr(cfg, "num_experts", None) is not None)
        )
        candidates = (
            (
                ("transformers.models.qwen3_5_moe.modeling_qwen3_5_moe", "Qwen3_5MoeDynamicCache"),
                ("transformers.models.qwen3_5.modeling_qwen3_5", "Qwen3_5DynamicCache"),
            )
            if is_moe
            else (
                ("transformers.models.qwen3_5.modeling_qwen3_5", "Qwen3_5DynamicCache"),
                ("transformers.models.qwen3_5_moe.modeling_qwen3_5_moe", "Qwen3_5MoeDynamicCache"),
            )
        )
        for mod_path, cls_name in candidates:
            try:
                mod = __import__(mod_path, fromlist=[cls_name])
                cls = getattr(mod, cls_name)
                return cls(cfg)
            except Exception:
                pass
        try:
            from transformers.cache_utils import DynamicCache

            try:
                return DynamicCache(config=cfg)
            except TypeError:
                return DynamicCache()
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


_GUARDED_LAYER_STATE_ATTRS = frozenset(
    {
        "has_previous_state",
        "conv_states",
        "recurrent_states",
        "keys",
        "values",
        "is_conv_states_initialized",
        "is_recurrent_states_initialized",
    }
)


class _ReplayCacheLayerProxy:
    """Checkpoint-guarded view over one `cache.layers[layer_idx]` entry (`transformers >= 5.20 / 5.24`)."""

    def __init__(self, parent: _ReplayKVCacheProxy, base_layer: Any, layer_idx: int) -> None:
        readonly = bool(object.__getattribute__(parent, "_readonly_base"))
        zc_hssd = bool(object.__getattribute__(parent, "_zero_copy_hssd"))
        object.__setattr__(self, "_parent", parent)
        object.__setattr__(self, "_base_layer", base_layer)
        object.__setattr__(self, "_layer_idx", int(layer_idx))
        object.__setattr__(self, "_readonly", readonly)
        object.__setattr__(self, "_state_overlay", {})
        object.__setattr__(self, "_state_snap", {})
        cs = getattr(base_layer, "conv_states", None)
        rs = getattr(base_layer, "recurrent_states", None)
        hps = getattr(base_layer, "has_previous_state", None)
        object.__setattr__(
            self,
            "_conv_proxy",
            _SlotStateListProxy(cs, zero_copy_hssd=False, readonly=readonly) if isinstance(cs, list) else None,
        )
        object.__setattr__(
            self,
            "_rec_proxy",
            _SlotStateListProxy(rs, zero_copy_hssd=zc_hssd, readonly=readonly) if isinstance(rs, list) else None,
        )
        object.__setattr__(
            self,
            "_hps_proxy",
            _SnapshotFlagContainer(hps, readonly=readonly) if isinstance(hps, (dict, list)) else None,
        )
        object.__setattr__(
            self,
            "_hps_bool_snap",
            bool(hps) if isinstance(hps, bool) else None,
        )
        object.__setattr__(self, "_hps_bool_overlay", [None])

    def reset_snapshot(self) -> None:
        cp = object.__getattribute__(self, "_conv_proxy")
        if cp is not None:
            cp.reset_snapshot()
        rp = object.__getattribute__(self, "_rec_proxy")
        if rp is not None:
            rp.reset_snapshot()
        hp = object.__getattribute__(self, "_hps_proxy")
        if hp is not None:
            hp.reset_snapshot()
        hps = getattr(object.__getattribute__(self, "_base_layer"), "has_previous_state", None)
        object.__setattr__(self, "_hps_bool_snap", bool(hps) if isinstance(hps, bool) else None)
        object.__getattribute__(self, "_hps_bool_overlay")[0] = None
        object.__getattribute__(self, "_state_overlay").clear()
        object.__getattribute__(self, "_state_snap").clear()

    @property
    def is_initialized(self) -> bool:
        parent = object.__getattribute__(self, "_parent")
        idx = object.__getattribute__(self, "_layer_idx")
        fp8_pref: dict[int, tuple[Any, Any]] = object.__getattribute__(parent, "_fp8_prefix_kv")
        if idx in fp8_pref:
            return True
        base_lyr = object.__getattribute__(self, "_base_layer")
        return bool(getattr(base_lyr, "is_initialized", False))

    def update(self, key_states: torch.Tensor, value_states: torch.Tensor, *args: Any, **kwargs: Any) -> Any:
        parent = object.__getattribute__(self, "_parent")
        idx = object.__getattribute__(self, "_layer_idx")
        return parent.update(key_states, value_states, idx, *args, **kwargs)

    def get_seq_length(self) -> int:
        parent = object.__getattribute__(self, "_parent")
        idx = object.__getattribute__(self, "_layer_idx")
        return parent.get_seq_length(idx)

    def get_mask_sizes(self, cache_position: torch.Tensor | int) -> tuple[int, int]:
        parent = object.__getattribute__(self, "_parent")
        idx = object.__getattribute__(self, "_layer_idx")
        fp8_pref: dict[int, tuple[Any, Any]] = object.__getattribute__(parent, "_fp8_prefix_kv")
        if fp8_pref:
            q_len = int(cache_position.shape[0]) if isinstance(cache_position, torch.Tensor) else int(cache_position)
            return parent.get_seq_length(idx) + q_len, 0
        base_lyr = object.__getattribute__(self, "_base_layer")
        fn = getattr(base_lyr, "get_mask_sizes", None)
        if callable(fn):
            return fn(cache_position)
        return parent.get_mask_sizes(cache_position, idx)

    def get_conv_state(self, state_idx: int = 0, **kwargs: Any) -> torch.Tensor | None:
        parent = object.__getattribute__(self, "_parent")
        idx = object.__getattribute__(self, "_layer_idx")
        return parent.get_conv_state(idx, state_idx=state_idx, **kwargs)

    def update_conv_state(
        self,
        conv_states: torch.Tensor,
        state_idx: int = 0,
        conv_kernel_size: int | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        parent = object.__getattribute__(self, "_parent")
        idx = object.__getattribute__(self, "_layer_idx")
        return parent.update_conv_state(
            conv_states, idx, state_idx=state_idx, conv_kernel_size=conv_kernel_size, **kwargs
        )

    def get_recurrent_state(self, state_idx: int = 0, **kwargs: Any) -> Any:
        parent = object.__getattribute__(self, "_parent")
        idx = object.__getattribute__(self, "_layer_idx")
        return parent.get_recurrent_state(idx, state_idx=state_idx, **kwargs)

    def update_recurrent_state(self, recurrent_states: Any, state_idx: int = 0, **kwargs: Any) -> Any:
        parent = object.__getattribute__(self, "_parent")
        idx = object.__getattribute__(self, "_layer_idx")
        return parent.update_recurrent_state(recurrent_states, idx, state_idx=state_idx, **kwargs)

    def __getattr__(self, name: str) -> Any:
        if name == "conv_states":
            cp = object.__getattribute__(self, "_conv_proxy")
            if cp is not None:
                return cp
        if name == "recurrent_states":
            rp = object.__getattribute__(self, "_rec_proxy")
            if rp is not None:
                return rp
        if name in ("conv_states", "recurrent_states"):
            base_lyr = object.__getattribute__(self, "_base_layer")
            snap_map: dict[str, Any] = object.__getattribute__(self, "_state_snap")
            ov_map: dict[str, Any] = object.__getattribute__(self, "_state_overlay")
            if _in_backward():
                sv = snap_map[name] if name in snap_map else _clone_detached_state(getattr(base_lyr, name, None))
                return sv.clone() if isinstance(sv, torch.Tensor) else sv
            if name not in snap_map:
                snap_map[name] = _clone_detached_state(getattr(base_lyr, name, None))
            if object.__getattribute__(self, "_readonly"):
                if name not in ov_map:
                    sv = snap_map[name]
                    ov_map[name] = sv.clone() if isinstance(sv, torch.Tensor) else sv
                return ov_map[name]
            return getattr(base_lyr, name)
        if name == "has_previous_state":
            hp = object.__getattribute__(self, "_hps_proxy")
            if hp is not None:
                return hp
            hps_snap: bool | None = object.__getattribute__(self, "_hps_bool_snap")
            if hps_snap is not None:
                if _in_backward():
                    return hps_snap
                if object.__getattribute__(self, "_readonly"):
                    ov = object.__getattribute__(self, "_hps_bool_overlay")[0]
                    if ov is not None:
                        return ov
        if _in_backward():
            snap_map: dict[str, Any] = object.__getattribute__(self, "_state_snap")
            if name in snap_map:
                sv = snap_map[name]
                return sv.clone() if isinstance(sv, torch.Tensor) else sv
        elif object.__getattribute__(self, "_readonly"):
            ov_map: dict[str, Any] = object.__getattribute__(self, "_state_overlay")
            if name in ov_map:
                return ov_map[name]
        return getattr(object.__getattribute__(self, "_base_layer"), name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name in _GUARDED_LAYER_STATE_ATTRS:
            if _in_backward():
                return
            if object.__getattribute__(self, "_readonly"):
                if name == "has_previous_state" and isinstance(value, bool):
                    object.__getattribute__(self, "_hps_bool_overlay")[0] = value
                else:
                    object.__getattribute__(self, "_state_overlay")[name] = value
                return
            if name == "has_previous_state" and isinstance(value, bool):
                if object.__getattribute__(self, "_hps_bool_snap") is None:
                    cur = getattr(object.__getattribute__(self, "_base_layer"), "has_previous_state", False)
                    object.__setattr__(self, "_hps_bool_snap", bool(cur))
            else:
                snap_map: dict[str, Any] = object.__getattribute__(self, "_state_snap")
                if name not in snap_map:
                    cur = getattr(object.__getattribute__(self, "_base_layer"), name, None)
                    snap_map[name] = _clone_detached_state(cur) if isinstance(cur, torch.Tensor) else cur
        setattr(object.__getattribute__(self, "_base_layer"), name, value)


class _ReplayCacheLayersList(list):  # type: ignore[type-arg]
    """`list` subclass exposing `_ReplayCacheLayerProxy` wrappers so `isinstance(cache.layers, list)` stays True."""

    def __init__(self, parent: _ReplayKVCacheProxy, base_layers: list[Any]) -> None:
        super().__init__(_ReplayCacheLayerProxy(parent, lyr, i) for i, lyr in enumerate(base_layers))

    def reset_snapshots(self) -> None:
        for item in self:
            if isinstance(item, _ReplayCacheLayerProxy):
                item.reset_snapshot()


class _ReplayKVCacheProxy:
    """Wraps `past_key_values` / `cache` so `update` injects live re-projected `(k_live, v_live)`
    out-of-place and linear-attention conv/recurrent states + `has_previous_state` are checkpoint-
    recomputation safe across HF 4.x, HF 5.2-5.19, HF 5.20-5.24 (`cache.layers[i]`), HSSD, and Toy."""

    def __init__(
        self,
        base_cache: Any,
        *,
        readonly_base: bool = False,
        fp8_prefix_kv: dict[int, tuple[Any, Any]] | None = None,
        zero_copy_hssd: bool = False,
    ) -> None:
        object.__setattr__(self, "_base_cache", base_cache)
        object.__setattr__(self, "_readonly_base", bool(readonly_base))
        object.__setattr__(self, "_zero_copy_hssd", bool(zero_copy_hssd))
        object.__setattr__(self, "_detach_recurrent_state", True)
        object.__setattr__(self, "_fp8_prefix_kv", dict(fp8_prefix_kv) if fp8_prefix_kv else {})
        object.__setattr__(self, "_pending_replay", {})
        object.__setattr__(self, "_live_kv_for_seg", {})
        object.__setattr__(self, "_raw_prefix_kv", {})
        object.__setattr__(self, "_conv_snap", {})
        object.__setattr__(self, "_rec_snap", {})
        object.__setattr__(self, "_conv_fwd_overlay", {})
        object.__setattr__(self, "_rec_fwd_overlay", {})
        object.__setattr__(self, "_has_prev_snap", {})
        object.__setattr__(self, "_has_prev_global", [None])
        conv = getattr(base_cache, "conv_states", None)
        rec = getattr(base_cache, "recurrent_states", None)
        layers = getattr(base_cache, "layers", None)
        object.__setattr__(
            self,
            "_conv_proxy",
            _SlotStateListProxy(conv, zero_copy_hssd=False, readonly=bool(readonly_base))
            if isinstance(conv, list)
            else None,
        )
        object.__setattr__(
            self,
            "_rec_proxy",
            _SlotStateListProxy(rec, zero_copy_hssd=bool(zero_copy_hssd), readonly=bool(readonly_base))
            if isinstance(rec, list)
            else None,
        )
        object.__setattr__(
            self,
            "_layers_proxy",
            _ReplayCacheLayersList(self, layers) if isinstance(layers, list) else None,
        )

    @property
    def base_cache(self) -> Any:
        return object.__getattribute__(self, "_base_cache")

    @property
    def is_initialized(self) -> bool:
        fp8_pref: dict[int, tuple[Any, Any]] = object.__getattribute__(self, "_fp8_prefix_kv")
        if fp8_pref:
            return True
        base = object.__getattribute__(self, "_base_cache")
        return bool(getattr(base, "is_initialized", False))

    @property
    def has_previous_state(self) -> _ProxyHasPreviousState:
        return _ProxyHasPreviousState(self)

    def _query_has_previous_state_global(self) -> bool:
        box: list[bool | None] = object.__getattribute__(self, "_has_prev_global")
        if box[0] is not None:
            return bool(box[0])
        base = object.__getattribute__(self, "_base_cache")
        hps = getattr(base, "has_previous_state", None)
        if callable(hps):
            try:
                val = bool(hps())
            except TypeError:
                layers = getattr(base, "layers", None)
                if isinstance(layers, list):
                    val = any(_read_base_has_previous_state(base, i) for i in range(len(layers)))
                else:
                    val = False
        elif hps is not None:
            val = bool(hps)
        else:
            cs = getattr(base, "conv_states", None)
            rs = getattr(base, "recurrent_states", None)
            val = bool(
                (isinstance(cs, list) and any(x is not None for x in cs))
                or (isinstance(rs, list) and any(x is not None for x in rs))
            )
        if not _in_backward():
            box[0] = val
        return val

    def _query_has_previous_state_layer(self, layer_idx: int, state_idx: int = 0) -> bool:
        snap: dict[tuple[int, int], bool] = object.__getattribute__(self, "_has_prev_snap")
        key = (int(layer_idx), int(state_idx))
        if key in snap:
            return snap[key]
        base = object.__getattribute__(self, "_base_cache")
        val = _read_base_has_previous_state(base, int(layer_idx), int(state_idx))
        if not _in_backward():
            snap[key] = val
        return val

    def _snapshot_has_prev_before_mutation(self, layer_idx: int, state_idx: int = 0) -> None:
        if _in_backward():
            return
        snap: dict[tuple[int, int], bool] = object.__getattribute__(self, "_has_prev_snap")
        key = (int(layer_idx), int(state_idx))
        if key not in snap:
            base = object.__getattribute__(self, "_base_cache")
            snap[key] = _read_base_has_previous_state(base, int(layer_idx), int(state_idx))
        box: list[bool | None] = object.__getattribute__(self, "_has_prev_global")
        if box[0] is None:
            self._query_has_previous_state_global()

    def __len__(self) -> int:
        base = object.__getattribute__(self, "_base_cache")
        if hasattr(base, "__len__"):
            return len(base)
        layers = getattr(base, "layers", None)
        if isinstance(layers, list):
            return len(layers)
        return 0

    def __bool__(self) -> bool:
        return True

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

    def get_mask_sizes(self, cache_position: torch.Tensor | int, layer_idx: int) -> tuple[int, int]:
        fp8_pref: dict[int, tuple[Any, Any]] = object.__getattribute__(self, "_fp8_prefix_kv")
        if fp8_pref:
            q_len = int(cache_position.shape[0]) if isinstance(cache_position, torch.Tensor) else int(cache_position)
            return self.get_seq_length(layer_idx) + q_len, 0
        base = object.__getattribute__(self, "_base_cache")
        return base.get_mask_sizes(cache_position, layer_idx)

    def reset_segment(self) -> None:
        object.__getattribute__(self, "_pending_replay").clear()
        object.__getattribute__(self, "_live_kv_for_seg").clear()
        object.__getattribute__(self, "_raw_prefix_kv").clear()
        object.__getattribute__(self, "_fp8_prefix_kv").clear()
        object.__getattribute__(self, "_conv_snap").clear()
        object.__getattribute__(self, "_rec_snap").clear()
        object.__getattribute__(self, "_conv_fwd_overlay").clear()
        object.__getattribute__(self, "_rec_fwd_overlay").clear()
        object.__getattribute__(self, "_has_prev_snap").clear()
        object.__getattribute__(self, "_has_prev_global")[0] = None
        cp = object.__getattribute__(self, "_conv_proxy")
        if cp is not None:
            cp.reset_snapshot()
        rp = object.__getattribute__(self, "_rec_proxy")
        if rp is not None:
            rp.reset_snapshot()
        lp = object.__getattribute__(self, "_layers_proxy")
        if lp is not None:
            lp.reset_snapshots()

    def get_conv_state(self, layer_idx: int, state_idx: int = 0, **kwargs: Any) -> torch.Tensor | None:
        del kwargs
        base = object.__getattribute__(self, "_base_cache")
        readonly_base: bool = object.__getattribute__(self, "_readonly_base")
        conv_snap: dict[tuple[int, int], tuple[bool, torch.Tensor | None]] = object.__getattribute__(
            self, "_conv_snap"
        )
        conv_ov: dict[tuple[int, int], torch.Tensor | None] = object.__getattribute__(self, "_conv_fwd_overlay")
        key = (int(layer_idx), int(state_idx))
        if _in_backward():
            if key in conv_snap:
                val = conv_snap[key][1]
                return val.clone() if isinstance(val, torch.Tensor) else val
            _, val = _read_base_conv_state(base, int(layer_idx), int(state_idx))
            return val.detach().clone() if isinstance(val, torch.Tensor) else val
        if key not in conv_snap:
            self._snapshot_has_prev_before_mutation(int(layer_idx), int(state_idx))
            had_prev, val = _read_base_conv_state(base, int(layer_idx), int(state_idx))
            conv_snap[key] = (had_prev, _clone_detached_state(val))
        if readonly_base:
            if key not in conv_ov:
                snap_v = conv_snap[key][1]
                conv_ov[key] = snap_v.clone() if isinstance(snap_v, torch.Tensor) else snap_v
            return conv_ov[key]
        _, live_val = _read_base_conv_state(base, int(layer_idx), int(state_idx))
        return live_val

    def update_conv_state(
        self,
        conv_states: torch.Tensor,
        layer_idx: int,
        state_idx: int = 0,
        conv_kernel_size: int | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        base = object.__getattribute__(self, "_base_cache")
        readonly_base: bool = object.__getattribute__(self, "_readonly_base")
        conv_snap: dict[tuple[int, int], tuple[bool, torch.Tensor | None]] = object.__getattribute__(
            self, "_conv_snap"
        )
        conv_ov: dict[tuple[int, int], torch.Tensor | None] = object.__getattribute__(self, "_conv_fwd_overlay")
        key = (int(layer_idx), int(state_idx))
        if not _in_backward():
            self._snapshot_has_prev_before_mutation(int(layer_idx), int(state_idx))
            if key not in conv_snap:
                had_prev, prev = _read_base_conv_state(base, int(layer_idx), int(state_idx))
                conv_snap[key] = (had_prev, _clone_detached_state(prev))
            if not readonly_base:
                upd_fn = getattr(base, "update_conv_state", None)
                if callable(upd_fn):
                    _call_cache_method(
                        upd_fn,
                        conv_states.detach(),
                        int(layer_idx),
                        state_idx=int(state_idx),
                        conv_kernel_size=conv_kernel_size,
                        **kwargs,
                    )
                else:
                    layers = getattr(base, "layers", None)
                    if isinstance(layers, list) and 0 <= int(layer_idx) < len(layers):
                        lyr_upd = getattr(layers[int(layer_idx)], "update_conv_state", None)
                        if callable(lyr_upd):
                            _call_cache_method(
                                lyr_upd,
                                conv_states.detach(),
                                state_idx=int(state_idx),
                                conv_kernel_size=conv_kernel_size,
                                **kwargs,
                            )
        had_prev, prev_snap = conv_snap.get(key, (False, None))
        if had_prev and isinstance(prev_snap, torch.Tensor):
            full_conv = torch.cat(
                [prev_snap.to(device=conv_states.device, dtype=conv_states.dtype), conv_states], dim=-1
            )
        else:
            full_conv = conv_states
            if conv_kernel_size is not None and full_conv.shape[-1] < int(conv_kernel_size):
                full_conv = F.pad(full_conv, (int(conv_kernel_size) - full_conv.shape[-1], 0), value=0)
        if not _in_backward() and readonly_base:
            keep_w = (
                int(conv_kernel_size)
                if conv_kernel_size is not None
                else (int(prev_snap.shape[-1]) if isinstance(prev_snap, torch.Tensor) else int(full_conv.shape[-1]))
            )
            conv_ov[key] = full_conv.detach()[..., -keep_w:].clone()
        return full_conv

    def get_recurrent_state(self, layer_idx: int, state_idx: int = 0, **kwargs: Any) -> Any:
        del kwargs
        base = object.__getattribute__(self, "_base_cache")
        readonly_base: bool = object.__getattribute__(self, "_readonly_base")
        zc_hssd: bool = object.__getattribute__(self, "_zero_copy_hssd")
        rec_snap: dict[tuple[int, int], Any] = object.__getattribute__(self, "_rec_snap")
        rec_ov: dict[tuple[int, int], Any] = object.__getattribute__(self, "_rec_fwd_overlay")
        key = (int(layer_idx), int(state_idx))
        if _in_backward():
            if key in rec_snap:
                val = rec_snap[key]
                return val.clone() if isinstance(val, torch.Tensor) else val
            val = _read_base_recurrent_state(base, int(layer_idx), int(state_idx))
            return _clone_detached_state(val, allow_zero_copy_hssd=zc_hssd)
        if key not in rec_snap:
            self._snapshot_has_prev_before_mutation(int(layer_idx), int(state_idx))
            val = _read_base_recurrent_state(base, int(layer_idx), int(state_idx))
            rec_snap[key] = _clone_detached_state(val, allow_zero_copy_hssd=zc_hssd)
        if readonly_base:
            if key not in rec_ov:
                rec_ov[key] = _clone_detached_state(rec_snap[key], allow_zero_copy_hssd=zc_hssd)
            return rec_ov[key]
        return _read_base_recurrent_state(base, int(layer_idx), int(state_idx))

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
        readonly_base: bool = object.__getattribute__(self, "_readonly_base")
        zc_hssd: bool = object.__getattribute__(self, "_zero_copy_hssd")
        rec_snap: dict[tuple[int, int], Any] = object.__getattribute__(self, "_rec_snap")
        rec_ov: dict[tuple[int, int], Any] = object.__getattribute__(self, "_rec_fwd_overlay")
        key = (int(layer_idx), int(state_idx))
        self._snapshot_has_prev_before_mutation(int(layer_idx), int(state_idx))
        if key not in rec_snap:
            prev = _read_base_recurrent_state(base, int(layer_idx), int(state_idx))
            rec_snap[key] = _clone_detached_state(prev, allow_zero_copy_hssd=zc_hssd)
        if readonly_base:
            rec_ov[key] = _clone_detached_state(recurrent_states, allow_zero_copy_hssd=zc_hssd)
            return recurrent_states
        det_state = _clone_detached_state(recurrent_states, allow_zero_copy_hssd=zc_hssd)
        upd_fn = getattr(base, "update_recurrent_state", None)
        if callable(upd_fn):
            return _call_cache_method(
                upd_fn,
                det_state,
                int(layer_idx),
                state_idx=int(state_idx),
                **kwargs,
            )
        layers = getattr(base, "layers", None)
        if isinstance(layers, list) and 0 <= int(layer_idx) < len(layers):
            lyr_upd = getattr(layers[int(layer_idx)], "update_recurrent_state", None)
            if callable(lyr_upd):
                return _call_cache_method(lyr_upd, det_state, state_idx=int(state_idx), **kwargs)
        return det_state

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
        t_cur = key_states.shape[-2]
        # SharedKVView on Pass r >= 2 calls update with 0-length slice -> return Pass 1's live (k_ret, v_ret)
        if t_cur == 0:
            if slot in live_map:
                return live_map[slot]
            if not readonly_base and _in_backward():
                if slot in raw_pref:
                    return raw_pref[slot]
                kp, vp = _extract_cached_kv(base, slot)
                if kp is not None and vp is not None:
                    return kp.detach(), vp.detach()
                return key_states, value_states
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
        if name == "layers":
            lp = object.__getattribute__(self, "_layers_proxy")
            if lp is not None:
                return lp
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
    done_ev: Any = None

    def wait_ready(self) -> None:
        if self.done_ev is not None:
            self.done_ev.synchronize()
            self.done_ev = None


def _offload_prior_record(
    u: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    *,
    offload_to_cpu: bool,
    copy_stream: Any | None,
) -> _PriorSegRecord:
    """Capture prior-segment `(u, cos, sin)` either HBM-resident or via async pinned-memory DMA."""
    if not offload_to_cpu:
        return _PriorSegRecord(u.detach(), cos.detach(), sin.detach())
    if copy_stream is not None and u.is_cuda:
        dev = u.device
        u_s = u.detach().contiguous()
        c_s = cos.detach().contiguous()
        s_s = sin.detach().contiguous()
        ready_ev = torch.cuda.Event()
        ready_ev.record(torch.cuda.current_stream(dev))
        u_dst = torch.empty(u_s.shape, dtype=u_s.dtype, pin_memory=True)
        c_dst = torch.empty(c_s.shape, dtype=c_s.dtype, pin_memory=True)
        s_dst = torch.empty(s_s.shape, dtype=s_s.dtype, pin_memory=True)
        with torch.cuda.stream(copy_stream):
            copy_stream.wait_event(ready_ev)
            u_dst.copy_(u_s, non_blocking=True)
            u_s.record_stream(copy_stream)
            if c_s.is_cuda:
                c_dst.copy_(c_s, non_blocking=True)
                c_s.record_stream(copy_stream)
            else:
                c_dst.copy_(c_s)
            if s_s.is_cuda:
                s_dst.copy_(s_s, non_blocking=True)
                s_s.record_stream(copy_stream)
            else:
                s_dst.copy_(s_s)
            done_ev = torch.cuda.Event()
            done_ev.record(copy_stream)
        return _PriorSegRecord(u_dst, c_dst, s_dst, done_ev=done_ev)
    return _PriorSegRecord(u.detach().to("cpu"), cos.detach().to("cpu"), sin.detach().to("cpu"))


def _gather_prior_field(tensors: list[torch.Tensor], idx: torch.Tensor) -> torch.Tensor:
    """Gather `[B, M, D]` along dim=1 from per-segment `[B, T_i, D]` tensors without concatenating them."""
    ref = tensors[0]
    idx_d = idx if idx.device == ref.device else idx.to(ref.device)
    bsz_r = int(ref.shape[0])
    if len(tensors) == 1:
        if bsz_r == 1:
            return ref[0].index_select(0, idx_d[0]).unsqueeze(0)
        return ref.gather(1, idx_d.unsqueeze(-1).expand(-1, -1, ref.shape[-1]))
    out = ref.new_empty((bsz_r, idx_d.shape[1], ref.shape[-1]))
    offset = 0
    if bsz_r == 1 and ref.device.type == "cpu":
        idx_1d = idx_d[0]
        for t in tensors:
            t_len = int(t.shape[1])
            pos = torch.nonzero((idx_1d >= offset) & (idx_1d < offset + t_len), as_tuple=False).flatten()
            if pos.numel() > 0:
                out[0, pos] = t[0].index_select(0, idx_1d.index_select(0, pos) - offset)
            offset += t_len
        return out
    for t in tensors:
        t_len = int(t.shape[1])
        local = (idx_d - offset).clamp(0, t_len - 1)
        if bsz_r == 1:
            gathered = t[0].index_select(0, local[0]).unsqueeze(0)
        else:
            gathered = t.gather(1, local.unsqueeze(-1).expand(-1, -1, t.shape[-1]))
        mask = (idx_d >= offset) & (idx_d < offset + t_len)
        out = torch.where(mask.unsqueeze(-1), gathered, out)
        offset += t_len
    return out


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
        offload_prior_hidden: bool | str | None = None,
        async_offload: bool = True,
        hbm_headroom_gib: float = 8.0,
        loss_impl: str = "auto",
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
                async_offload=async_offload,
                hbm_headroom_gib=hbm_headroom_gib,
                loss_impl=loss_impl,
                replay_indexer_weight=replay_indexer_weight,
                query_sim_chunk=query_sim_chunk,
            )

    @staticmethod
    def _extract_query_heads_split(
        attn_mod: nn.Module, x_in: torch.Tensor
    ) -> tuple[torch.Tensor, nn.Module | None, int, int, int]:
        """Project `x_in` (`[B, T, hidden]`) through `q_proj` -> `(q_pre_norm [B, T, H, D], q_norm, H, G, D)`."""
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
        return q_states, q_norm, num_heads, num_kv_heads, head_dim

    @classmethod
    def _extract_query_heads(cls, attn_mod: nn.Module, x_in: torch.Tensor) -> tuple[torch.Tensor, int, int, int]:
        """Project `x_in` (`[B, T, hidden]`) through `q_proj` (+ `q_norm` if present) -> `[B, H, T, D]`."""
        q_states, q_norm, num_heads, num_kv_heads, head_dim = cls._extract_query_heads_split(attn_mod, x_in)
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
        if q_cur_detached.is_cuda:
            chunk = max(chunk, 512 if t_prior > 65536 else 1024)

        with torch.no_grad():
            if lsa_ix is not None:
                qz = lsa_ix.query_latent(q_cur_detached)      # [B, G | H, T_seg, d_idx]
                kz = lsa_ix.key_latent(k_prior_cached)        # [B, G, T_prior, d_idx]
                kz_m_tr = kz.to(qz.dtype).transpose(-1, -2)
                sim = qz.new_full((bsz, g_cnt, t_prior), float("-inf"))
                if qz.shape[1] != g_cnt:
                    hg_q = qz.shape[1] // g_cnt
                    qz_g = qz.view(bsz, g_cnt, hg_q, t_seg, -1)
                    kz_g_tr = kz_m_tr.unsqueeze(2)
                    for c0 in range(0, t_seg, chunk):
                        qc = qz_g[:, :, :, c0 : c0 + chunk]
                        sc = torch.matmul(qc, kz_g_tr).amax(dim=(2, 3))
                        sim = torch.maximum(sim, sc)
                else:
                    for c0 in range(0, t_seg, chunk):
                        qc = qz[:, :, c0 : c0 + chunk]
                        sc = torch.matmul(qc, kz_m_tr).amax(dim=2)
                        sim = torch.maximum(sim, sc)
            else:
                lo = rot_dim if rot_dim < dh else 0
                scale_nope = 1.0 / math.sqrt(max(1, dh - lo))
                q_nope = upcast(q_cur_detached[..., lo:]).view(bsz, g_cnt, hg, t_seg, -1)
                k_nope_tr = upcast(k_prior_cached[..., lo:]).unsqueeze(2).transpose(-1, -2)
                sim = q_nope.new_full((bsz, g_cnt, t_prior), float("-inf"))
                for c0 in range(0, t_seg, chunk):
                    qc = q_nope[:, :, :, c0 : c0 + chunk]
                    sc = (torch.matmul(qc, k_nope_tr) * scale_nope).amax(dim=(2, 3))
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
        loss_scale: float = 1.0,
    ) -> BPTTStepResult:
        """Execute segmented forward (+ per-segment backward with K/V replay) over `input_ids` `[B, T]`."""
        if input_ids.ndim != 2:
            raise ValueError(f"input_ids must have shape [B, T], got {tuple(input_ids.shape)}")
        bsz, t_total = input_ids.shape
        seg_len = self.cfg.resolve_seg_len(t_total)
        n_segs = math.ceil(t_total / seg_len)
        target_full = input_ids if (labels is None and loss_fn is None) else labels

        phys_list, text_cfg, loop_ctl = physical_layers(self.model)
        full_layers: list[tuple[int, nn.Module]] = []
        for phys_idx, layer_mod in phys_list:
            lt = getattr(layer_mod, "layer_type", None)
            if lt == FULL or (lt is None and hasattr(layer_mod, "self_attn")):
                full_layers.append((phys_idx, layer_mod.self_attn))

        # Resolve HBM-resident vs async pinned-memory CPU offload once per step
        hidden_est = int(getattr(text_cfg, "hidden_size", 2048) or 2048)
        first_p = next(self.model.parameters(), None)
        elem_b = int(first_p.element_size()) if isinstance(first_p, torch.Tensor) else 2
        est_prior_bytes = (
            max(0, n_segs - 1) * len(full_layers) * bsz * seg_len * hidden_est * elem_b * 105 // 100
        )
        offload_to_cpu = self.cfg.resolve_offload_to_cpu(input_ids.device, est_prior_bytes)
        copy_stream = (
            torch.cuda.Stream(device=input_ids.device)
            if (offload_to_cpu and self.cfg.async_offload and input_ids.is_cuda)
            else None
        )

        prior_records: dict[int, list[_PriorSegRecord]] = {p: [] for p, _ in full_layers}
        cur_seg_records: dict[int, _PriorSegRecord] = {}
        seg_replay_aux: list[torch.Tensor] = []
        recorded_indices: dict[tuple[int, int], torch.Tensor] = {}
        replay_meta: dict[tuple[int, int], tuple[torch.Tensor, torch.Tensor, int, int]] = {}
        replayed_counts: list[int] = []
        seg_losses: list[float] = []
        seg_ce_losses: list[float] = []
        seg_aux_losses: list[float] = []
        ce_loss_weighted_sum = 0.0
        aux_loss_weighted_sum = 0.0
        lsa_aux_total = 0.0
        loop_aux_total = 0.0
        hssd_aux_total = 0.0
        replay_aux_total = 0.0
        active_seg_state = {"seg_idx": 0}
        proxy_holder: list[_ReplayKVCacheProxy | None] = [None]
        hooks: list[Any] = []

        tm, lm_head = _resolve_bptt_backbone_and_head(self.model)
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

        # Pre-compute exact valid (!= -100) supervised target token counts per segment in a single
        # batched sync so supervised LM CE is normalized by `valid_tok_s / total_valid_toks` (exact
        # parity with un-segmented full-sequence CE) and all-masked (`valid_tok_s == 0`) segments
        # never waste autograd frames or divide by zero.
        seg_valid_counts: list[int] = [0] * n_segs
        seg_valid_flags: list[bool] = [True] * n_segs
        if target_full is not None:
            cnt_t: list[torch.Tensor] = []
            shape_ok: list[bool] = []
            for chk_i in range(n_segs):
                c0 = chk_i * seg_len
                c1 = min(t_total, (chk_i + 1) * seg_len)
                st_chk = target_full[:, c0 + 1 : c1 + 1] if c1 < t_total else target_full[:, c0 + 1 : t_total]
                exp_len = (c1 - c0) if c1 < t_total else max(0, (c1 - c0) - 1)
                ok_s = bool(st_chk.numel() > 0 and st_chk.shape[1] == exp_len)
                shape_ok.append(ok_s)
                cnt_t.append(
                    (st_chk != -100).sum(dtype=torch.long)
                    if ok_s
                    else torch.zeros((), device=target_full.device, dtype=torch.long)
                )
            raw_counts = torch.stack(cnt_t).tolist() if cnt_t else []
            seg_valid_counts = [int(c) if ok_s else 0 for ok_s, c in zip(shape_ok, raw_counts)]
            if can_pre_head:
                seg_valid_flags = [c > 0 for c in seg_valid_counts]
        elif can_pre_head:
            seg_valid_flags = [False] * n_segs
        total_valid_toks = sum(seg_valid_counts)
        n_active_tok_segs = sum(1 for c in seg_valid_counts if c > 0)
        last_active_seg = max((i for i, ok in enumerate(seg_valid_flags) if ok), default=-1)

        def make_attn_pre_hook(phys_idx: int):
            def pre_hook(attn_mod: nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
                x_cur: torch.Tensor = args[0] if len(args) > 0 else kwargs["hidden_states"]
                pos_emb = args[1] if len(args) > 1 else kwargs.get("position_embeddings")
                pkv_key: str | None = None
                pkv_from_pos = False
                pkv: Any = None
                if len(args) > 3 and args[3] is not None:
                    pkv = args[3]
                    pkv_from_pos = True
                else:
                    pkv_key, pkv = extract_cache_kwarg(kwargs)
                if pos_emb is None:
                    return None

                # Skip pass >= 2 when kv_share=True (Pass 1 already computed live replay for this slot)
                if loop_ctl is not None and getattr(loop_ctl, "_cur", None) is not None:
                    _ctx, r_pass = loop_ctl._cur
                    if r_pass >= 2 and getattr(loop_ctl.spec, "kv_share", True):
                        return None
                if pkv is not None and type(pkv).__name__ == "SharedKVView":
                    try:
                        is_rec = bool(object.__getattribute__(pkv, "_record"))
                    except AttributeError:
                        is_rec = False
                    if not is_rec:
                        return None

                cos_cur, sin_cur = pos_emb
                rot_dim = int(cos_cur.shape[-1])
                s_idx = active_seg_state["seg_idx"]
                cache_slot = int(getattr(attn_mod, "layer_idx", phys_idx))
                proxy = proxy_holder[0]
                in_bwd = _in_backward()

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
                            dev, dt = x_cur.device, x_cur.dtype
                            cached_meta = replay_meta.get((s_idx, phys_idx)) if in_bwd else None
                            q_pre_norm_det: torch.Tensor | None = None
                            q_norm_mod: nn.Module | None = None
                            if cached_meta is not None:
                                top_idx_c, uniq_pad_c, num_kv_heads, head_dim = cached_meta
                                top_idx = top_idx_c if top_idx_c.device == dev else top_idx_c.to(device=dev)
                                uniq_pad = uniq_pad_c if uniq_pad_c.device == dev else uniq_pad_c.to(device=dev)
                                max_u = int(uniq_pad.shape[1])
                                q_det: torch.Tensor | None = None
                            else:
                                with torch.no_grad():
                                    q_pre_norm_det, q_norm_mod, _nh, num_kv_heads, head_dim = (
                                        self._extract_query_heads_split(attn_mod, x_cur.detach())
                                    )
                                    q_det_t = q_norm_mod(q_pre_norm_det) if q_norm_mod is not None else q_pre_norm_det
                                    q_det = q_det_t.transpose(1, 2)
                                top_idx = self._select_replay_indices(
                                    attn_mod, q_det, k_prior, rot_dim=rot_dim, k_sel=k_sel
                                )
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
                                if not in_bwd:
                                    top_det = top_idx.detach()
                                    recorded_indices[(s_idx, phys_idx)] = top_det
                                    # Keep tiny 64 KB replay meta on device (zero forward D2H / backward H2D syncs)
                                    replay_meta[(s_idx, phys_idx)] = (
                                        top_det,
                                        uniq_pad.detach(),
                                        num_kv_heads,
                                        head_dim,
                                    )

                            recs = prior_records[phys_idx]
                            for rec in recs:
                                rec.wait_ready()
                            nb_h2d = bool(dev.type == "cuda" and recs[0].u.is_pinned())
                            u_sel = _gather_prior_field([r.u for r in recs], uniq_pad).to(
                                device=dev, dtype=dt, non_blocking=nb_h2d
                            )
                            cos_sel = _gather_prior_field([r.cos for r in recs], uniq_pad).to(
                                device=dev, dtype=cos_cur.dtype, non_blocking=nb_h2d
                            )
                            sin_sel = _gather_prior_field([r.sin for r in recs], uniq_pad).to(
                                device=dev, dtype=sin_cur.dtype, non_blocking=nb_h2d
                            )

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
                            # (isolate saved tensors from any enclosing checkpoint frame so backward recompute
                            # never re-runs `_chunked_replay_indexer_kl` only to discard it!)
                            lsa_ix = getattr(attn_mod, "lsa_indexer", None)
                            if (
                                not in_bwd
                                and q_det is not None
                                and lsa_ix is not None
                                and self.cfg.replay_indexer_weight > 0
                                and (lsa_ix.wk.requires_grad or lsa_ix.wq.requires_grad or k_live.requires_grad)
                            ):
                                with torch.autograd.graph.saved_tensors_hooks(lambda x: x.detach(), lambda x: x):
                                    kz_live = lsa_ix.k_proj(k_live)  # [B, G, K_sel, d_idx]
                                    q_proj_grad = any(p.requires_grad for p in attn_mod.q_proj.parameters())
                                    q_norm_grad = q_norm_mod is not None and any(
                                        p.requires_grad for p in q_norm_mod.parameters()
                                    )
                                    if q_proj_grad:
                                        q_live_for_idx, _, _, _ = self._extract_query_heads(attn_mod, x_cur.detach())
                                    elif q_norm_grad and q_pre_norm_det is not None and q_norm_mod is not None:
                                        q_live_for_idx = q_norm_mod(q_pre_norm_det).transpose(1, 2)
                                    else:
                                        q_live_for_idx = q_det
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
                                    seg_replay_aux.append(float(self.cfg.replay_indexer_weight) * kl_idx)

                            proxy.set_pending(cache_slot, top_idx, k_live, v_live)
                        finally:
                            if loop_ctl is not None:
                                loop_ctl._cur = saved_cur

                # Segments at or after `last_active_seg` are never replayed by any future active segment — skip offload!
                if not in_bwd and s_idx < last_active_seg:
                    cur_seg_records[phys_idx] = _offload_prior_record(
                        x_cur,
                        cos_cur,
                        sin_cur,
                        offload_to_cpu=offload_to_cpu,
                        copy_stream=copy_stream,
                    )

                if proxy is not None and pkv is proxy.base_cache:
                    if pkv_key is not None:
                        new_kw = dict(kwargs)
                        new_kw[pkv_key] = proxy
                        return args, new_kw
                    if pkv_from_pos:
                        return (*args[:3], proxy, *args[4:]), kwargs
                return None

            return pre_hook

        for phys_idx, attn_mod in full_layers:
            hooks.append(attn_mod.register_forward_pre_hook(make_attn_pre_hook(phys_idx), with_kwargs=True))

        # Clear any stale LSA / HSSD pending losses before starting
        for mod in tm.modules():
            lsa_ix = getattr(mod, "lsa_indexer", None)
            if lsa_ix is not None:
                lsa_ix.pop_losses()
            hssd_ad = getattr(mod, "hssd_adapter", None)
            if hssd_ad is not None and callable(getattr(hssd_ad, "clear_aux", None)):
                hssd_ad.clear_aux()

        tm_cache_kw = cache_kwarg_name(tm)
        model_cache_kw = cache_kwarg_name(self.model)
        base_cache: Any = None
        try:
            for s_idx in range(n_segs):
                active_seg_state["seg_idx"] = s_idx
                cur_seg_records.clear()
                seg_replay_aux.clear()

                # Skip trailing all-(-100) segments (e.g. respect_doc_boundaries trailing padding):
                # no future segment needs their KV/HSSD state and they contribute zero loss.
                if can_pre_head and s_idx > last_active_seg:
                    seg_losses.append(0.0)
                    seg_ce_losses.append(0.0)
                    seg_aux_losses.append(0.0)
                    replayed_counts.append(0)
                    continue

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
                    proxy = _ReplayKVCacheProxy(base_cache, zero_copy_hssd=(s1 - s0 > 1))
                    proxy_holder[0] = proxy
                    pkv_arg: Any = proxy
                else:
                    proxy_holder[0] = None
                    pkv_arg = None

                tm_out: Any = None
                seg_hidden: torch.Tensor | None = None
                hidden_for_loss: torch.Tensor | None = None
                out: Any = None
                logits: torch.Tensor | None = None
                logits_for_loss: torch.Tensor | None = None

                seg_has_targets = seg_valid_flags[s_idx]
                if can_pre_head:
                    assert lm_head is not None
                    tm_kw: dict[str, Any] = {
                        "input_ids": seg_ids,
                        tm_cache_kw: pkv_arg,
                        "use_cache": True,
                    }
                    if bypass_mask and s_idx > 0:
                        tm_kw["attention_mask"] = no_mask_dict
                    if seg_has_targets:
                        tm_out = tm(**tm_kw)
                    else:
                        # Interior all-(-100) segment (e.g. >32K SFT prompt prefix): advance detached
                        # KV/HSSD caches and prior-hidden replay records under `no_grad()` so zero
                        # autograd graph or checkpoint frames are constructed!
                        with torch.no_grad():
                            tm_out = tm(**tm_kw)
                    base_cache = (
                        proxy_holder[0].base_cache
                        if proxy_holder[0] is not None
                        else extract_output_cache(tm_out)
                    )
                    seg_hidden = tm_out.last_hidden_state if hasattr(tm_out, "last_hidden_state") else tm_out[0]
                    hidden_for_loss = seg_hidden if s1 < t_total else seg_hidden[:, :-1]
                    if seg_has_targets and seg_targets is not None:
                        seg_loss, _ = lm_loss(
                            hidden_for_loss,
                            lm_head.weight,
                            seg_targets,
                            impl=self.cfg.loss_impl,
                            chunk_size=int(ce_chunk_size),
                            shift=False,
                            ignore_index=-100,
                        )
                    else:
                        seg_loss = None
                else:
                    fwd_kw: dict[str, Any] = {model_cache_kw: pkv_arg, "use_cache": True}
                    if bypass_mask and s_idx > 0:
                        fwd_kw["attention_mask"] = no_mask_dict
                    out = self.model(seg_ids, **fwd_kw)
                    base_cache = (
                        proxy_holder[0].base_cache
                        if proxy_holder[0] is not None
                        else extract_output_cache(out)
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

                seg_ce_loss: torch.Tensor | None = seg_loss
                acc_dt = seg_ce_loss.dtype if seg_ce_loss is not None else torch.float32
                replay_term = (
                    torch.stack([t.to(acc_dt) for t in seg_replay_aux]).sum()
                    if seg_replay_aux
                    else None
                )
                lsa_term = (
                    float(lsa_aux_weight) * torch.stack([t.to(acc_dt) for t in lsa_terms]).sum()
                    if (lsa_terms and lsa_aux_weight > 0)
                    else None
                )
                hssd_term = (
                    float(hssd_aux_weight) * torch.stack([t.to(acc_dt) for t in hssd_terms]).mean()
                    if (hssd_terms and hssd_aux_weight > 0)
                    else None
                )
                loop_term = (
                    float(loop_aux_weight) * loop_aux.to(acc_dt)
                    if (loop_aux is not None and loop_aux_weight > 0)
                    else None
                )
                aux_parts: list[torch.Tensor] = [
                    t for t in (replay_term, lsa_term, hssd_term, loop_term) if t is not None
                ]
                aux_sum: torch.Tensor | None = torch.stack(aux_parts).sum() if aux_parts else None
                lsa_terms.clear()
                hssd_terms.clear()
                seg_replay_aux.clear()
                aux_parts.clear()
                del loop_aux

                valid_tok_s = seg_valid_counts[s_idx]
                if seg_ce_loss is not None:
                    if total_valid_toks > 0 and valid_tok_s > 0:
                        ce_weight_s = float(valid_tok_s) / float(total_valid_toks)
                        aux_weight_s = 1.0 / float(max(1, n_active_tok_segs))
                    elif loss_fn is None:
                        ce_weight_s = 0.0
                        aux_weight_s = 0.0
                    else:
                        ce_weight_s = 1.0 / float(n_segs)
                        aux_weight_s = 1.0 / float(n_segs)

                    ce_val_s = float(seg_ce_loss.detach())
                    aux_val_s = float(aux_sum.detach()) if aux_sum is not None else 0.0
                    seg_ce_losses.append(ce_val_s)
                    seg_aux_losses.append(aux_val_s)
                    seg_losses.append(ce_val_s + aux_val_s)
                    ce_loss_weighted_sum += ce_weight_s * ce_val_s
                    aux_loss_weighted_sum += aux_weight_s * aux_val_s
                    if replay_term is not None:
                        replay_aux_total += aux_weight_s * float(replay_term.detach())
                    if lsa_term is not None:
                        lsa_aux_total += aux_weight_s * float(lsa_term.detach())
                    if hssd_term is not None:
                        hssd_aux_total += aux_weight_s * float(hssd_term.detach())
                    if loop_term is not None:
                        loop_aux_total += aux_weight_s * float(loop_term.detach())

                    bwd_terms: list[torch.Tensor] = []
                    if ce_weight_s > 0.0:
                        bwd_terms.append(seg_ce_loss * ce_weight_s)
                    if aux_sum is not None and aux_weight_s > 0.0:
                        bwd_terms.append(aux_sum * aux_weight_s)
                    bwd_loss = (
                        bwd_terms[0]
                        if len(bwd_terms) == 1
                        else (torch.stack(bwd_terms).sum() if bwd_terms else None)
                    )
                    bwd_terms.clear()
                    if backward and bwd_loss is not None and bwd_loss.requires_grad:
                        (bwd_loss * float(loss_scale)).backward()
                    bwd_loss = None
                else:
                    seg_ce_losses.append(0.0)
                    seg_aux_losses.append(0.0)
                    seg_losses.append(0.0)

                # Release live segment outputs & LoopContext graph stashes immediately after backward
                # (or skipped backward) so C++-bridged checkpoint frames can never remain pinned!
                tm_out = None
                seg_hidden = None
                hidden_for_loss = None
                out = None
                logits = None
                logits_for_loss = None
                seg_loss = None
                seg_ce_loss = None
                aux_sum = None
                replay_term = None
                lsa_term = None
                hssd_term = None
                loop_term = None

                # Count replayed positions on this segment (from first full-attn layer if any)
                if s_idx > 0 and full_layers and (s_idx, full_layers[0][0]) in recorded_indices:
                    replayed_counts.append(int(recorded_indices[(s_idx, full_layers[0][0])].shape[-1]))
                else:
                    replayed_counts.append(0)

                # Commit pre-attention `u_seg` records and detach carried recurrent/KV states!
                for p_id, rec in cur_seg_records.items():
                    prior_records[p_id].append(rec)
                replay_meta.clear()
                if proxy_holder[0] is not None:
                    proxy_holder[0].reset_segment()
                if base_cache is not None:
                    _detach_nested(base_cache)
                if loop_ctl is not None and callable(getattr(loop_ctl, "release_graph", None)):
                    loop_ctl.release_graph()

            ce_loss_out = ce_loss_weighted_sum
            aux_loss_out = aux_loss_weighted_sum
            total_loss_out = ce_loss_out + aux_loss_out
            recorded_indices_cpu = {
                k: (v.cpu() if v.device.type != "cpu" else v) for k, v in recorded_indices.items()
            }
            stats = {
                "bptt/num_segments": float(n_segs),
                "bptt/seg_len": float(seg_len),
                "bptt/replay_k": float(self.cfg.replay_k if self.cfg.replay_enabled else 0),
                "bptt/offload_to_cpu": 1.0 if offload_to_cpu else 0.0,
                "bptt/mean_replayed_tokens": (
                    sum(replayed_counts[1:]) / max(1, len(replayed_counts) - 1) if n_segs > 1 else 0.0
                ),
                "bptt/ce_loss": float(ce_loss_out),
                "bptt/aux_loss": float(aux_loss_out),
                "bptt/total_loss": float(total_loss_out),
                "bptt/valid_tokens": float(total_valid_toks),
                "bptt/lsa_kl": float(lsa_aux_total),
                "bptt/loop_aux": float(loop_aux_total),
                "bptt/hssd_aux": float(hssd_aux_total),
                "bptt/replay_aux": float(replay_aux_total),
            }
            return BPTTStepResult(
                total_loss=total_loss_out,
                segment_losses=seg_losses,
                num_segments=n_segs,
                replayed_tokens_per_seg=replayed_counts,
                replayed_indices=recorded_indices_cpu,
                stats=stats,
                ce_loss=ce_loss_out,
                aux_loss=aux_loss_out,
                segment_ce_losses=seg_ce_losses,
                segment_aux_losses=seg_aux_losses,
                segment_valid_tokens=list(seg_valid_counts),
            )
        finally:
            if copy_stream is not None:
                copy_stream.synchronize()
            for h in hooks:
                h.remove()
            if proxy_holder[0] is not None:
                proxy_holder[0].reset_segment()
                proxy_holder[0] = None
            if base_cache is not None:
                _detach_nested(base_cache)
            prior_records.clear()
            cur_seg_records.clear()
            replay_meta.clear()
            seg_replay_aux.clear()
            if loop_ctl is not None and callable(getattr(loop_ctl, "release_graph", None)):
                loop_ctl.release_graph()
            for mod in tm.modules():
                lsa_ix = getattr(mod, "lsa_indexer", None)
                if lsa_ix is not None:
                    lsa_ix.pop_losses()
                    lsa_cache = getattr(lsa_ix, "_cache", None)
                    if isinstance(lsa_cache, dict):
                        lsa_cache.clear()
                hssd_ad = getattr(mod, "hssd_adapter", None)
                if hssd_ad is not None and callable(getattr(hssd_ad, "clear_aux", None)):
                    hssd_ad.clear_aux()

    run = run_step
    __call__ = run_step
