"""Encode-Think-Decode loop retrofit for hybrid (Gated DeltaNet + Gated Attention) MoE models.

Physical layers [0, P) = prelude (run once), [P, C) = core (run `num_passes` times),
[C, N) = coda (run once). Default for Qwen3.6-35B-A3B (N=40, 10 macro-blocks of 3 linear +
1 full attention): P=8, C=32, passes=2  ->  8 + 24*2 + 8 = **64 virtual layers**.

How it is wired (no re-implementation of the HF forward):
  * `text_model.layers` is replaced by a ModuleList of `VirtualLayer`s in schedule order and
    `config.num_hidden_layers / layer_types` are patched to the virtual depth. HF's own forward
    loop, mask construction and cache allocation then "just work", and each virtual layer gets
    its **own cache slot** => Gated DeltaNet recurrent/conv states are isolated per
    (physical layer, pass) automatically.
  * Core full-attention layers in pass r>=2 use **recursive KV sharing** (MoR): their k_proj /
    v_proj outputs are replaced by pass-1's (forward hooks) and, with a cache, writes are
    redirected to the pass-1 slot through a zero-length `update` (no KV growth).
  * Pass r>=2 is merged as  h_r = h_{r-1} + alpha_r * m * (Core(bridge(h_{r-1})) - h_{r-1})
    with alpha_r zero-initialised => at step 0 the wrapped model is **bit-identical** to the
    original (x + 0*d == x), and gradients still reach alpha_r.
  * Optional MoR token-choice depth router (causal => usable in decode), with a DeltaNet
    *write mask* so exited tokens neither write the pass-r recurrent state nor enter the
    pass-r short-conv window (keeps train/decode semantics identical).
  * Optional per-step depth sampling (`set_active_passes`) keeps depth-1 (== stock Qwen3.6
    graph, vLLM-servable) valid while training depth-2.
"""
from __future__ import annotations

import functools
import hashlib
import logging
import math
import random
import types
from dataclasses import dataclass, field, fields
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from ._cache_compat import extract_cache_from_kwargs, extract_cache_kwarg
from ._dtype import upcast
from .introspect import FULL, LINEAR, attention_module, find_children, inspect_model, modules_with_layer_idx, router_of
from .mhc import LoopBridge
from .quant_autograd import (
    _LINEAR_FP8_SCALE_ATTRS,
    FP8_DTYPES,
    _dequant_linear_weight,
    _extract_block_size,
    _first_tensor_attr,
    _linear_in_out_features,
    dequantize_fp8_tensor,
    is_quantized_module,
)

log = logging.getLogger("qloop")


def _in_backward() -> bool:
    fn = getattr(torch._C, "_current_graph_task_id", None)
    return fn is not None and fn() != -1


def _looks_like_probs(t: torch.Tensor) -> bool:
    with torch.no_grad():
        f = upcast(t.detach())
        if f.numel() == 0:
            return False
        return bool((f >= 0).all() and ((f.sum(-1) - 1.0).abs() < 1e-3).all())


# ============================================================================ spec / schedule

@dataclass
class LoopSpec:
    prelude_end: int = 8
    core_end: int = 32
    num_passes: int = 2
    target_virtual_layers: int | None = 64
    kv_share: bool = True
    bridge_init: tuple[float, ...] = (0.6, 0.2, 0.2)
    bridge_rank: int = 0
    bridge_mode: str = "concat"
    bridge_up_std: float = 0.01
    step_embedding: bool = True
    gate_init: float = 0.0
    per_pass_router: bool = True
    router_adapter_rank: int = 32
    router_repel_init: float = 0.0
    router_diversity_weight: float = 0.0
    attn_adapter_rank: int = 0
    attn_adapter_alpha: float | None = None
    attn_proj_names: tuple[str, ...] = ("q_proj", "o_proj")
    mor: bool = False
    mor_target_depth: float | None = None
    mor_aux_weight: float = 0.01
    write_mask: bool = True
    checkpoint: bool = False
    collect_routing: bool = False
    kv_proj_names: tuple[str, ...] = ("k_proj", "v_proj")
    write_zero_names: tuple[str, ...] = ("in_proj_qkvz", "in_proj_qkv")
    write_neg_names: tuple[str, ...] = ("in_proj_ba", "in_proj_b", "in_proj_a")
    write_neg_value: float = -30.0
    train_pass_probs: list[float] | None = field(default=None)

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> LoopSpec:
        d = dict(d or {})
        known = {f.name for f in fields(cls)}
        unknown = set(d) - known - {"enabled"}
        if unknown:
            raise ValueError(f"unknown loop.* keys: {sorted(unknown)}; valid: {sorted(known)}")
        for k in ("bridge_init", "kv_proj_names", "attn_proj_names", "write_zero_names", "write_neg_names"):
            if k in d and isinstance(d[k], list):
                d[k] = tuple(d[k])
        d.pop("enabled", None)
        return cls(**d)

    @property
    def core_len(self) -> int:
        return self.core_end - self.prelude_end

    def virtual_layers(self, n_phys: int) -> int:
        return self.prelude_end + self.core_len * self.num_passes + (n_phys - self.core_end)

    def validate(self, layer_types: list[str]) -> list[str]:
        n = len(layer_types)
        if not (0 <= self.prelude_end < self.core_end <= n):
            raise ValueError(f"need 0 <= prelude_end < core_end <= {n}, got {self.prelude_end}, {self.core_end}")
        if self.num_passes < 1:
            raise ValueError("num_passes must be >= 1")
        if self.bridge_mode not in ("concat", "subspace"):
            raise ValueError(f"bridge_mode must be 'concat' or 'subspace', got {self.bridge_mode!r}")
        if self.bridge_up_std < 0:
            raise ValueError(f"bridge_up_std must be >= 0, got {self.bridge_up_std}")
        if self.attn_adapter_rank < 0:
            raise ValueError(f"attn_adapter_rank must be >= 0, got {self.attn_adapter_rank}")
        if self.kv_share and self.attn_adapter_rank > 0 and (set(self.attn_proj_names) & set(self.kv_proj_names)):
            raise ValueError(
                f"attn_proj_names {self.attn_proj_names} overlaps kv_proj_names {self.kv_proj_names} with kv_share=True"
            )
        v = self.virtual_layers(n)
        if self.target_virtual_layers is not None and v != self.target_virtual_layers:
            raise ValueError(
                f"schedule gives {v} virtual layers but target_virtual_layers={self.target_virtual_layers}. "
                f"With N={n}: virtual = N + core_len*(passes-1); e.g. core_len={(self.target_virtual_layers - n)} "
                f"for 2 passes. Set target_virtual_layers=null to skip this check."
            )
        if self.train_pass_probs is not None and len(self.train_pass_probs) != self.num_passes:
            raise ValueError("train_pass_probs needs one probability per pass (index 0 = depth 1)")
        warns = []
        if self.prelude_end > 0 and layer_types[self.prelude_end - 1] != FULL:
            lt = layer_types[self.prelude_end - 1]
            warns.append(f"prelude ends mid macro-block (layer {self.prelude_end - 1} is {lt})")
        if layer_types[self.core_end - 1] != FULL:
            warns.append(f"core ends mid macro-block (layer {self.core_end - 1} is {layer_types[self.core_end - 1]})")
        if FULL not in layer_types[self.prelude_end:self.core_end]:
            warns.append("core contains no full-attention layer")
        return warns


@dataclass(frozen=True)
class Slot:
    vidx: int        # virtual position (== cache slot for DeltaNet)
    phys: int        # physical layer index
    r: int           # 0 = prelude/coda, 1..R = core pass
    entry: bool      # first core layer of pass r
    exit: bool       # last core layer of pass r
    cache_idx: int   # layer_idx given to the attention module (KV-share -> pass-1 slot)
    layer_type: str


def build_schedule(spec: LoopSpec, layer_types: list[str]) -> list[Slot]:
    n, p, c = len(layer_types), spec.prelude_end, spec.core_end
    order: list[tuple[int, int]] = [(i, 0) for i in range(p)]
    for r in range(1, spec.num_passes + 1):
        order += [(i, r) for i in range(p, c)]
    order += [(i, 0) for i in range(c, n)]
    pass1_vidx = {i: i for i in range(p, c)}  # prelude length unchanged => pass-1 vidx == phys
    slots = []
    for v, (i, r) in enumerate(order):
        lt = layer_types[i]
        share = spec.kv_share and r >= 2 and lt == FULL
        slots.append(Slot(v, i, r, r >= 1 and i == p, r >= 1 and i == c - 1,
                          pass1_vidx[i] if share else v, lt))
    return slots


# ============================================================================ runtime state

class LoopContext:
    """Per-forward state. Captured by checkpoint closures, so recompute sees the same state."""

    def __init__(self) -> None:
        self.anchor: torch.Tensor | None = None
        self.start: dict[int, torch.Tensor] = {}
        self.kv_stash: dict[tuple[int, str], Any] = {}
        self.pkv_stash: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        self.keep: dict[int, torch.Tensor] = {}       # pass -> [B,T] {0,1}
        self.coef: dict[int, torch.Tensor] = {}       # pass -> [B,T,1] straight-through MoR coef
        self.aux: dict[str, torch.Tensor] = {}
        self.div_terms: list[torch.Tensor] = []
        self.routing: dict[tuple[int, int], torch.Tensor] = {}
        self.router_probs: dict[tuple[int, int], torch.Tensor] = {}
        self.stats: dict[str, float] = {}

    def release_graph(self, *, clear_routing: bool = False) -> None:
        """Drop all live autograd-tensor references to break C++-bridged checkpoint cycles.

        Under `checkpoint(use_reentrant=False)`, each `VirtualLayer` `_CheckpointFrame` captures
        this `LoopContext` while `anchor`/`start`/`kv_stash`/`pkv_stash`/`keep`/`coef`/`aux`/
        `div_terms` point to live tensors whose C++ `grad_fn` nodes hold `_CheckpointFrame` unpack
        hooks. If `.backward()` is skipped or aborted, Python's cyclic GC cannot traverse C++
        autograd nodes, permanently leaking the checkpoint graph unless these fields are cleared.
        """
        if self.div_terms:
            self.div_terms = [t.detach().float() for t in self.div_terms]
        self.anchor = None
        self.start.clear()
        self.kv_stash.clear()
        self.pkv_stash.clear()
        self.keep.clear()
        self.coef.clear()
        self.aux.clear()
        self.router_probs.clear()
        if clear_routing:
            self.routing.clear()


def _resolve_kv_slot_scale(
    lyr: Any,
    cache: Any,
    layer_idx: int,
    *,
    is_key: bool,
) -> torch.Tensor | None:
    attr_single = "k_scale" if is_key else "v_scale"
    attr_plural = "k_scales" if is_key else "v_scales"
    for holder in (lyr, cache):
        if holder is None:
            continue
        sc = getattr(holder, attr_single, None)
        if isinstance(sc, torch.Tensor):
            return sc
        scs = getattr(holder, attr_plural, None)
        if isinstance(scs, (list, tuple, nn.ParameterList)) and 0 <= layer_idx < len(scs):
            cand = scs[layer_idx]
            if isinstance(cand, torch.Tensor):
                return cand
        elif isinstance(scs, dict) and layer_idx in scs and isinstance(scs[layer_idx], torch.Tensor):
            return scs[layer_idx]
    return None


def _dequant_kv_slot_tensor(
    t: torch.Tensor,
    out_dt: torch.dtype,
    lyr: Any,
    cache: Any,
    layer_idx: int,
    *,
    is_key: bool,
) -> torch.Tensor:
    if t.dtype in FP8_DTYPES or not t.is_floating_point():
        sc = _resolve_kv_slot_scale(lyr, cache, layer_idx, is_key=is_key)
        if isinstance(sc, torch.Tensor):
            sf = sc.to(device=t.device, dtype=torch.float32)
            if sf.numel() == 1:
                sf = sf.reshape(())
            elif sf.ndim == 1 and t.ndim == 4 and sf.shape[0] == t.shape[1]:
                sf = sf.view(1, -1, 1, 1)
            elif sf.ndim < t.ndim:
                while sf.ndim < t.ndim:
                    sf = sf.unsqueeze(-1)
            return (t.to(torch.float32) * sf).to(out_dt)
        return t.to(out_dt)
    return t if t.dtype == out_dt else t.to(out_dt)


class SharedKVView:
    """Cache proxy for pass>=2 full attention: `update` appends nothing, returns pass-1 K/V."""

    def __init__(
        self,
        cache: Any,
        stash: dict[int, tuple[torch.Tensor, torch.Tensor]] | None = None,
        *,
        record: bool = False,
    ):
        object.__setattr__(self, "_cache", cache)
        object.__setattr__(self, "_stash", stash)
        object.__setattr__(self, "_record", bool(record))

    def __getattr__(self, name: str) -> Any:
        return getattr(object.__getattribute__(self, "_cache"), name)

    def __setattr__(self, name: str, value: Any) -> None:
        setattr(object.__getattribute__(self, "_cache"), name, value)

    def __getitem__(self, idx: Any) -> Any:
        return object.__getattribute__(self, "_cache")[idx]

    def __len__(self) -> int:
        return len(object.__getattribute__(self, "_cache"))

    def __bool__(self) -> bool:
        return True

    def update(self, key_states, value_states, layer_idx, *args, **kwargs):
        cache = object.__getattribute__(self, "_cache")
        stash = object.__getattribute__(self, "_stash")
        record = object.__getattribute__(self, "_record")
        lidx = int(layer_idx)
        readonly_cache = bool(getattr(cache, "_readonly_base", False))
        if record:
            res = cache.update(key_states, value_states, layer_idx, *args, **kwargs)
            if stash is not None and isinstance(res, tuple) and len(res) >= 2:
                if readonly_cache:
                    # Stash only the compact current-suffix `(k, v)` (`~64 MiB/layer` at 32K) rather than
                    # pinning the entire concatenated `[prefix + suffix]` tensor (`~2 GiB/layer` at 1M)!
                    stash[lidx] = (key_states, value_states)
                else:
                    stash[lidx] = (res[0], res[1])
            return res
        if stash is not None and lidx in stash:
            if readonly_cache:
                k_suf, v_suf = stash[lidx]
                return cache.update(k_suf, v_suf, layer_idx, *args, **kwargs)
            return stash[lidx]
        if type(cache).__name__ in ("QuantizedHybridCache", "_QuantizedKV"):
            return cache.update(key_states[..., :0, :], value_states[..., :0, :], layer_idx, *args, **kwargs)
        inner_cache = getattr(cache, "cache", None)
        if inner_cache is None:
            inner_cache = getattr(cache, "_cache", None)
        for holder in (cache, inner_cache):
            if holder is None:
                continue
            layers = getattr(holder, "layers", None)
            if isinstance(layers, (list, tuple, nn.ModuleList)) and 0 <= lidx < len(layers):
                lyr = layers[lidx]
                deq_fn = getattr(lyr, "_dequantize", None)
                qk = getattr(lyr, "_quantized_keys", None)
                qv = getattr(lyr, "_quantized_values", None)
                if (
                    callable(deq_fn)
                    and isinstance(qk, torch.Tensor)
                    and isinstance(qv, torch.Tensor)
                    and qk.numel() > 0
                ):
                    return deq_fn(qk).to(key_states.dtype), deq_fn(qv).to(value_states.dtype)
                k_t = getattr(lyr, "keys", None)
                v_t = getattr(lyr, "values", None)
                if isinstance(k_t, torch.Tensor) and isinstance(v_t, torch.Tensor) and k_t.numel() > 0:
                    return (
                        _dequant_kv_slot_tensor(k_t, key_states.dtype, lyr, cache, lidx, is_key=True),
                        _dequant_kv_slot_tensor(v_t, value_states.dtype, lyr, cache, lidx, is_key=False),
                    )
            kc = getattr(holder, "key_cache", None)
            vc = getattr(holder, "value_cache", None)
            if (
                isinstance(kc, list)
                and isinstance(vc, list)
                and 0 <= lidx < len(kc)
                and 0 <= lidx < len(vc)
                and isinstance(kc[lidx], torch.Tensor)
                and isinstance(vc[lidx], torch.Tensor)
                and kc[lidx].numel() > 0
            ):
                return (
                    _dequant_kv_slot_tensor(kc[lidx], key_states.dtype, None, cache, lidx, is_key=True),
                    _dequant_kv_slot_tensor(vc[lidx], value_states.dtype, None, cache, lidx, is_key=False),
                )
        return cache.update(key_states[..., :0, :], value_states[..., :0, :], layer_idx, *args, **kwargs)


class PassRouterAdapters(nn.Module):
    """Per-(extra_pass, core_layer) MoE router adapters + cross-pass expert repulsion.

    Lets looped passes (r >= 2) route to distinct expert subsets from pass 1, unlocking effective
    active parameter scaling beyond the single-pass 3B subset without duplicating expert weights.
    Zero-initialised `up`/`weight`/`bias` (+ default `repel=0`) keep step-0 output bit-identical.
    """

    def __init__(self, hidden: int, num_experts: int, core_len: int, extra_passes: int,
                 rank: int = 32, repel_init: float = 0.0):
        super().__init__()
        self.hidden = int(hidden)
        self.num_experts = int(num_experts)
        self.core_len = int(core_len)
        self.extra_passes = int(extra_passes)
        self.rank = min(int(rank), self.hidden, self.num_experts) if int(rank) > 0 else 0
        if self.rank > 0:
            down = torch.empty(extra_passes, core_len, self.rank, self.hidden)
            for p in range(extra_passes):
                for c in range(core_len):
                    gen = torch.Generator().manual_seed(1729 + p * 997 + c)
                    bound = 1.0 / math.sqrt(self.hidden)
                    down[p, c].uniform_(-bound, bound, generator=gen)
            self.down = nn.Parameter(down)
            self.up = nn.Parameter(torch.zeros(extra_passes, core_len, self.num_experts, self.rank))
        else:
            self.weight = nn.Parameter(torch.zeros(extra_passes, core_len, self.num_experts, self.hidden))
        self.bias = nn.Parameter(torch.zeros(extra_passes, core_len, self.num_experts))
        self.repel = nn.Parameter(torch.full((extra_passes, core_len), float(repel_init)))

    def is_zero(self, pass_idx: int, core_idx: int) -> bool:
        with torch.no_grad():
            w = self.up[pass_idx, core_idx] if self.rank > 0 else self.weight[pass_idx, core_idx]
            b = self.bias[pass_idx, core_idx]
            r = self.repel[pass_idx, core_idx]
            return bool((w == 0).all() and (b == 0).all() and float(r) == 0.0)

    def delta_logits(self, flat_x: torch.Tensor, pass_idx: int, core_idx: int,
                     prev_mask: torch.Tensor | None = None) -> torch.Tensor:
        x = upcast(flat_x).to(self.bias.dtype)
        if self.rank > 0:
            h = F.linear(x, self.down[pass_idx, core_idx])
            d = F.linear(h, self.up[pass_idx, core_idx], self.bias[pass_idx, core_idx])
        else:
            d = F.linear(x, self.weight[pass_idx, core_idx], self.bias[pass_idx, core_idx])
        if prev_mask is not None:
            d = d - self.repel[pass_idx, core_idx] * prev_mask.to(d.dtype)
        return d


def _linear_dims(mod: nn.Module) -> tuple[int, int]:
    return _linear_in_out_features(mod)


class _PassLoRAProj(nn.Module):
    """Per-extra-pass zero-initialised rank-r LoRA delta `scale * (x @ down^T) @ up^T`."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        extra_passes: int,
        rank: int,
        scale: float,
        seed_base: int,
    ):
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.extra_passes = int(extra_passes)
        self.rank = int(rank)
        self.scale = float(scale)
        down = torch.empty(self.extra_passes, self.rank, self.in_features)
        bound = 1.0 / math.sqrt(self.in_features)
        for p in range(self.extra_passes):
            gen = torch.Generator().manual_seed(seed_base + p * 1009)
            down[p].uniform_(-bound, bound, generator=gen)
        self.down = nn.Parameter(down)
        self.up = nn.Parameter(torch.zeros(self.extra_passes, self.out_features, self.rank))

    def is_zero(self, pass_idx: int) -> bool:
        with torch.no_grad():
            return bool((self.up[pass_idx] == 0).all())

    def delta(self, x_in: torch.Tensor, pass_idx: int) -> torch.Tensor:
        x = upcast(x_in).to(self.down.dtype)
        h = F.linear(x, self.down[pass_idx])
        return F.linear(h, self.up[pass_idx]) * self.scale


class PassAttnAdapters(nn.Module):
    """Per-(pass r>=2, core full-attn layer) zero-init LoRA residuals on Q/O projections (Fix A5).

    Keeps `k_proj` and `v_proj` shared with Pass 1 (`kv_share=True` => 1.0x KV cache footprint)
    while letting Passes 2..R specialise query addressing and output mixing independently.
    """

    def __init__(
        self,
        proj_specs: dict[int, dict[str, tuple[int, int]]],
        extra_passes: int,
        rank: int,
        alpha: float | None = None,
    ):
        super().__init__()
        self.extra_passes = int(extra_passes)
        self.rank = int(rank)
        self.scale = float(alpha if alpha is not None else self.rank) / max(self.rank, 1)
        self.layers = nn.ModuleDict()
        for phys in sorted(proj_specs):
            pdict = nn.ModuleDict()
            for j, (pname, (in_f, out_f)) in enumerate(sorted(proj_specs[phys].items())):
                seed = 4099 + phys * 131 + j * 37
                pdict[pname] = _PassLoRAProj(in_f, out_f, self.extra_passes, self.rank, self.scale, seed)
            self.layers[f"L{phys}"] = pdict

    def is_zero(self, phys: int, pname: str, pass_idx: int) -> bool:
        proj: _PassLoRAProj = self.layers[f"L{phys}"][pname]  # type: ignore[assignment]
        return proj.is_zero(pass_idx)

    def delta(self, x_in: torch.Tensor, phys: int, pname: str, pass_idx: int) -> torch.Tensor:
        proj: _PassLoRAProj = self.layers[f"L{phys}"][pname]  # type: ignore[assignment]
        return proj.delta(x_in, pass_idx)

    def up_abs_mean(self, pass_idx: int) -> float:
        vals = []
        for pdict in self.layers.values():
            for proj in pdict.values():
                vals.append(float(proj.up[pass_idx].detach().float().abs().mean()))
        return sum(vals) / max(len(vals), 1)


class LoopAdapters(nn.Module):
    """All new parameters (kept fp32; tiny). Attached as `text_model.loop_adapters`."""

    def __init__(
        self,
        hidden: int,
        spec: LoopSpec,
        num_experts: int | None = None,
        attn_proj_specs: dict[int, dict[str, tuple[int, int]]] | None = None,
    ):
        super().__init__()
        extra = max(spec.num_passes - 1, 1)
        self.gates = nn.Parameter(torch.full((extra, hidden), float(spec.gate_init)))
        self.bridge = LoopBridge(
            hidden,
            extra,
            spec.bridge_init,
            spec.bridge_rank,
            spec.step_embedding,
            adapter_mode=spec.bridge_mode,
            up_init_std=spec.bridge_up_std,
        )
        self.pass_router: PassRouterAdapters | None = None
        if spec.per_pass_router and spec.num_passes > 1 and num_experts is not None and spec.core_len > 0:
            self.pass_router = PassRouterAdapters(
                hidden=hidden,
                num_experts=int(num_experts),
                core_len=spec.core_len,
                extra_passes=extra,
                rank=spec.router_adapter_rank,
                repel_init=spec.router_repel_init,
            )
        self.pass_attn: PassAttnAdapters | None = None
        if spec.attn_adapter_rank > 0 and spec.num_passes > 1 and attn_proj_specs:
            self.pass_attn = PassAttnAdapters(
                proj_specs=attn_proj_specs,
                extra_passes=extra,
                rank=spec.attn_adapter_rank,
                alpha=spec.attn_adapter_alpha,
            )
        self.mor_router: nn.Linear | None = None
        if spec.mor:
            self.mor_router = nn.Linear(hidden, spec.num_passes)
            nn.init.zeros_(self.mor_router.weight)
            with torch.no_grad():  # start with every token taking the full depth
                self.mor_router.bias.zero_()
                self.mor_router.bias[-1] = 2.0

    @property
    def pass_gates(self) -> nn.Parameter:
        return self.gates


# ============================================================================ virtual layer

class VirtualLayer(nn.Module):
    def __init__(self, ctl: LoopController, slot: Slot, inner: nn.Module):
        super().__init__()
        self.inner = inner
        self.layer_type = slot.layer_type
        self.attention_type = slot.layer_type
        object.__setattr__(self, "_ctl", ctl)
        object.__setattr__(self, "slot", slot)
        object.__setattr__(self, "_idx_mods", modules_with_layer_idx(inner))

    def extra_repr(self) -> str:
        s = self.slot
        return f"v={s.vidx} phys={s.phys} pass={s.r} {s.layer_type} cache_idx={s.cache_idx}"

    def forward(self, hidden_states: torch.Tensor, *args: Any, **kwargs: Any) -> Any:
        ctl, s = self._ctl, self.slot
        if s.vidx == 0:
            ctl._begin_forward()
        if s.r > ctl.active_passes:  # depth sampling / test-time depth < trained depth
            return (hidden_states, *([None] * ctl._tuple_extra)) if ctl._tuple_out else hidden_states
        ctx = ctl.ctx
        x = hidden_states
        if s.entry:
            if s.r == 1:
                ctx.anchor = x
                ctx.start[1] = x
                if ctl.spec.mor:
                    ctl._route(ctx, x)
            else:
                ctx.start[s.r] = x
                if ctl.spec.checkpoint and self.training and torch.is_grad_enabled() and ctx.anchor is not None:
                    pass_idx = s.r - 2
                    x = checkpoint(
                        lambda h_in, anc_in, p_i=pass_idx: ctl.adapters.bridge(h_in, anc_in, p_i),
                        x,
                        ctx.anchor,
                        use_reentrant=False,
                    )
                else:
                    x = ctl.adapters.bridge(x, ctx.anchor, s.r - 2)
        pkv_key, pkv = extract_cache_kwarg(kwargs)
        if s.vidx == 0 and pkv is not None:
            ctl._sync_cache_last_linear(pkv)
        if pkv is not None and pkv_key is not None:
            if s.cache_idx != s.vidx:
                pkv_layers = getattr(pkv, "layers", None)
                if isinstance(pkv_layers, (list, tuple, nn.ModuleList)) and 0 <= s.vidx < len(pkv_layers):
                    v_lyr = pkv_layers[s.vidx]
                    if hasattr(v_lyr, "supports_early_init") and getattr(v_lyr, "keys", None) is None:
                        v_lyr.supports_early_init = False
                kwargs = dict(kwargs)
                kwargs[pkv_key] = SharedKVView(pkv, stash=ctx.pkv_stash, record=False)
            elif (
                ctl.spec.kv_share
                and ctl.active_passes > 1
                and s.r == 1
                and s.layer_type == FULL
                and ctl.spec.prelude_end <= s.phys < ctl.spec.core_end
            ):
                kwargs = dict(kwargs)
                kwargs[pkv_key] = SharedKVView(pkv, stash=ctx.pkv_stash, record=True)
        r, idx_mods, cache_idx, inner = s.r, self._idx_mods, s.cache_idx, self.inner
        attn_mod = getattr(inner, "self_attn", None)

        def run(inp: torch.Tensor) -> Any:
            prev = ctl._cur
            ctl._cur = (ctx, r)
            for m in idx_mods:
                m.layer_idx = cache_idx
            lsa_ix = getattr(attn_mod, "lsa_indexer", None) if attn_mod is not None else None
            set_pass = getattr(lsa_ix, "set_current_pass", None) if lsa_ix is not None else None
            if callable(set_pass):
                set_pass(max(r, 1))
            try:
                return inner(inp, *args, **kwargs)
            finally:
                if callable(set_pass):
                    set_pass(1)
                ctl._cur = prev

        if ctl.spec.checkpoint and self.training and torch.is_grad_enabled():
            out = checkpoint(run, x, use_reentrant=False)
        else:
            out = run(x)
        is_tuple = isinstance(out, tuple)
        ctl._tuple_out = is_tuple
        ctl._tuple_extra = (len(out) - 1) if is_tuple else 0
        h = out[0] if is_tuple else out
        if s.exit and r >= 2:
            h = ctl._combine(ctx, h, r)
        return (h, *out[1:]) if is_tuple else h


# ============================================================================ controller

class LoopController:
    """Apply / remove the loop retrofit and expose its knobs, stats and aux losses."""

    def __init__(self, model: nn.Module, spec: LoopSpec):
        parts = inspect_model(model)
        self.parts, self.spec = parts, spec
        for w in spec.validate(parts.layer_types):
            log.warning("loop spec: %s", w)
        self.schedule = build_schedule(spec, parts.layer_types)
        cfg, tm = parts.text_config, parts.text_model
        self._orig = {
            "layers": parts.layers,
            "num_hidden_layers": cfg.num_hidden_layers,
            "layer_types": list(getattr(cfg, "layer_types", parts.layer_types)),
            "layer_idx": [(m, m.layer_idx) for layer in parts.layers for m in modules_with_layer_idx(layer)],
        }
        has_core_routers = any(router_of(parts.layers[i]) is not None for i in range(spec.prelude_end, spec.core_end))
        num_experts = getattr(cfg, "num_experts", None) if has_core_routers else None
        self._num_experts: int | None = int(num_experts) if num_experts is not None else None
        self._top_k: int = int(getattr(cfg, "num_experts_per_tok", 8))
        self._norm_topk: bool = bool(getattr(cfg, "norm_topk_prob", True))
        attn_proj_specs: dict[int, dict[str, tuple[int, int]]] = {}
        self._attn_proj_mods: list[tuple[int, str, nn.Module]] = []
        if spec.attn_adapter_rank > 0 and spec.num_passes > 1:
            for phys, layer in self._core_layers(FULL):
                attn = attention_module(layer, FULL)
                found = find_children(attn, list(spec.attn_proj_names))
                if len(found) != len(spec.attn_proj_names):
                    raise RuntimeError(
                        f"attn_adapter_rank: layer {phys} attention lacks {spec.attn_proj_names}; children: "
                        f"{[n for n, _ in attn.named_children()]}"
                    )
                attn_proj_specs[phys] = {}
                for name, mod in found.items():
                    key = name.replace(".", "__")
                    attn_proj_specs[phys][key] = _linear_dims(mod)
                    self._attn_proj_mods.append((phys, key, mod))
        ref = tm.embed_tokens.weight
        adapter_dtype = torch.float64 if ref.dtype == torch.float64 else torch.float32
        self.adapters = LoopAdapters(
            cfg.hidden_size, spec, num_experts=self._num_experts, attn_proj_specs=attn_proj_specs
        ).to(device=ref.device, dtype=adapter_dtype)
        tm.loop_adapters = self.adapters
        self.active_passes = spec.num_passes
        self.ctx = LoopContext()
        self._cur: tuple[LoopContext, int] | None = None
        self._tuple_out = False
        self._tuple_extra = 0
        self._hooks: list[Any] = []
        self._wrapped_fwds: list[tuple[nn.Module, bool, Any]] = []
        self._is_probs_cache: dict[int, bool] = {}
        tm.layers = nn.ModuleList(VirtualLayer(self, s, parts.layers[s.phys]) for s in self.schedule)
        cfg.num_hidden_layers = len(self.schedule)
        cfg.layer_types = [s.layer_type for s in self.schedule]
        self._register_hooks()
        log.info(
            "loop retrofit: %d physical -> %d virtual layers (prelude %d | core %d x %d | coda %d), "
            "kv_share=%s bridge_mode=%s attn_lora=%s per_pass_router=%s mor=%s checkpoint=%s",
            parts.num_layers, len(self.schedule), spec.prelude_end, spec.core_len, spec.num_passes,
            parts.num_layers - spec.core_end, spec.kv_share, spec.bridge_mode,
            self.adapters.pass_attn is not None, self.adapters.pass_router is not None, spec.mor, spec.checkpoint,
        )

    # ------------------------------------------------------------------ hooks
    def _core_layers(self, layer_type: str) -> list[tuple[int, nn.Module]]:
        lo, hi = self.spec.prelude_end, self.spec.core_end
        return [(i, self._orig["layers"][i]) for i in range(lo, hi) if self.parts.layer_types[i] == layer_type]

    def _last_active_linear_vidx(self) -> int | None:
        for s in reversed(self.schedule):
            if s.layer_type == LINEAR and s.r <= self.active_passes:
                return s.cache_idx
        return None

    def _sync_cache_last_linear(self, pkv: Any) -> None:
        if pkv is None or not isinstance(getattr(pkv, "last_linear_layer", None), int):
            return
        n_virt = len(self.schedule)
        for attr in ("conv_states", "recurrent_states", "key_cache", "value_cache"):
            slots = getattr(pkv, attr, None)
            if isinstance(slots, list) and len(slots) < n_virt:
                slots.extend([None] * (n_virt - len(slots)))
        last_lin = self._last_active_linear_vidx()
        if last_lin is not None and pkv.last_linear_layer != last_lin:
            pkv.last_linear_layer = last_lin

    def _register_hooks(self) -> None:
        spec = self.spec

        def _tm_cache_pre(_mod: nn.Module, _args: tuple[Any, ...], kw: dict[str, Any]) -> None:
            pkv = extract_cache_from_kwargs(kw)
            if pkv is not None:
                self._sync_cache_last_linear(pkv)

        self._hooks.append(self.parts.text_model.register_forward_pre_hook(_tm_cache_pre, with_kwargs=True))
        if spec.kv_share and spec.num_passes > 1:
            for phys, layer in self._core_layers(FULL):
                attn = attention_module(layer, FULL)
                found = find_children(attn, list(spec.kv_proj_names))
                if len(found) != len(spec.kv_proj_names):
                    raise RuntimeError(f"kv_share: layer {phys} attention lacks {spec.kv_proj_names}; children: "
                                       f"{[n for n, _ in attn.named_children()]}")
                for name, mod in found.items():
                    self._hooks.append(mod.register_forward_hook(self._kv_hook(phys, name)))
                kvc_mod = getattr(attn, "kv_cache", None)
                if isinstance(kvc_mod, nn.Module):
                    self._wrap_kv_cache_submodule(phys, kvc_mod)
        if self.adapters.pass_attn is not None:
            for phys, key, mod in self._attn_proj_mods:
                self._hooks.append(mod.register_forward_hook(self._attn_lora_hook(phys, key)))
        if spec.mor and spec.write_mask and spec.num_passes > 1:
            for phys, layer in self._core_layers(LINEAR):
                attn = attention_module(layer, LINEAR)
                zero = find_children(attn, list(spec.write_zero_names))
                neg = find_children(attn, list(spec.write_neg_names))
                if not zero or not neg:
                    raise RuntimeError(f"write_mask: layer {phys} DeltaNet projections not found (want "
                                       f"{spec.write_zero_names} + {spec.write_neg_names}); children: "
                                       f"{[n for n, _ in attn.named_children()]}")
                for mod in zero.values():
                    self._hooks.append(mod.register_forward_hook(self._write_hook(None)))
                for mod in neg.values():
                    self._hooks.append(mod.register_forward_hook(self._write_hook(spec.write_neg_value)))
                gdn2_ad = getattr(attn, "gdn2_adapter", None)
                if isinstance(gdn2_ad, nn.Module) and isinstance(getattr(gdn2_ad, "down", None), nn.Module):
                    self._hooks.append(gdn2_ad.down.register_forward_hook(self._write_hook(None)))
        if self.adapters.pass_router is not None:
            for phys in range(spec.prelude_end, spec.core_end):
                router = router_of(self._orig["layers"][phys])
                if router is not None:
                    # prepend=True guarantees per-pass routing executes BEFORE HotExpertDeltas / scan hooks
                    fn = self._pass_router_hook(phys, phys - spec.prelude_end)
                    self._hooks.append(router.register_forward_hook(fn, prepend=True))
        elif spec.collect_routing or spec.router_diversity_weight > 0:
            k = self._top_k
            e = self._num_experts
            for phys in range(spec.prelude_end, spec.core_end):
                router = router_of(self._orig["layers"][phys])
                if router is not None:
                    self._hooks.append(router.register_forward_hook(self._route_hook(phys, k, e)))

    def _wrap_kv_cache_submodule(self, phys: int, kvc_mod: nn.Module) -> None:
        had_inst_fwd = "forward" in kvc_mod.__dict__
        orig_inst_fwd = kvc_mod.__dict__.get("forward")
        orig_fwd = kvc_mod.forward
        raw_fn = getattr(orig_fwd, "__func__", orig_fwd)

        @functools.wraps(raw_fn)
        def _kvc_fwd(_self: nn.Module, *a: Any, **kw: Any) -> Any:
            cur = self._cur
            if cur is None:
                return orig_fwd(*a, **kw)
            ctx, r = cur
            key = (phys, "__kv_cache_mod__")
            if r == 1:
                res = orig_fwd(*a, **kw)
                ctx.kv_stash[key] = res
                return res
            if key in ctx.kv_stash:
                return ctx.kv_stash[key]
            return orig_fwd(*a, **kw)

        kvc_mod.forward = types.MethodType(_kvc_fwd, kvc_mod)  # type: ignore[method-assign]
        self._wrapped_fwds.append((kvc_mod, had_inst_fwd, orig_inst_fwd))

    def _kv_hook(self, phys: int, name: str):
        def hook(mod: nn.Module, inp: Any, out: torch.Tensor):
            cur = self._cur
            if cur is None:
                return None
            ctx, r = cur
            key = (phys, name)
            if r == 1:
                if self.active_passes > 1:
                    ctx.kv_stash[key] = out
                return None
            stash = ctx.kv_stash.get(key)
            if stash is None or stash.shape != out.shape:
                raise RuntimeError(f"kv_share: missing/mismatched pass-1 {name} for layer {phys} at pass {r} "
                                   f"(stash={None if stash is None else tuple(stash.shape)}, now={tuple(out.shape)})")
            return stash
        return hook

    def _attn_lora_hook(self, phys: int, key: str):
        pa_mod = self.adapters.pass_attn
        assert pa_mod is not None

        def hook(mod: nn.Module, inp: Any, out: torch.Tensor):
            cur = self._cur
            if cur is None or cur[1] < 2:
                return None
            p_idx = cur[1] - 2
            if not torch.is_grad_enabled() and pa_mod.is_zero(phys, key, p_idx):
                return None
            delta = pa_mod.delta(inp[0], phys, key, p_idx)
            return out + delta.to(out.dtype).view_as(out)

        return hook

    def _write_hook(self, neg_value: float | None):
        def hook(mod: nn.Module, inp: Any, out: torch.Tensor):
            cur = self._cur
            if cur is None or cur[1] < 2:
                return None
            keep = cur[0].keep.get(cur[1])
            if keep is None:
                return None
            k = keep.to(out.dtype).unsqueeze(-1)
            return out * k if neg_value is None else out * k + neg_value * (1 - k)
        return hook

    def _pass_router_hook(self, phys: int, core_idx: int):
        pr_mod = self.adapters.pass_router
        assert pr_mod is not None
        e, k, norm_topk = pr_mod.num_experts, self._top_k, self._norm_topk
        div_w = float(self.spec.router_diversity_weight)

        def _extract_full_and_topk(out: Any) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None, bool]:
            outs = out if isinstance(out, (tuple, list)) else (out,)
            tensors = [o for o in outs if isinstance(o, torch.Tensor)]
            full = next((o for o in tensors if o.is_floating_point() and o.shape[-1] == e), None)
            if full is None:
                raise RuntimeError(f"per_pass_router: layer {phys} router output lacks [..., {e}] float tensor")
            if phys not in self._is_probs_cache:
                self._is_probs_cache[phys] = isinstance(out, (tuple, list)) and _looks_like_probs(full)
            is_probs = self._is_probs_cache[phys]
            idx = next((o for o in tensors if not o.is_floating_point() and o.shape[-1] == k), None)
            sc = next((o for o in tensors if o.is_floating_point() and o.shape[-1] == k and o is not full), None)
            return full, sc, idx, is_probs

        def hook(mod: nn.Module, inp: Any, out: Any):
            cur = self._cur
            if cur is None or cur[1] < 1:
                return None
            ctx, r = cur
            x_in = inp[0]
            flat_x = x_in.reshape(-1, x_in.shape[-1])
            full, sc, idx, is_probs = _extract_full_and_topk(out)
            if r == 1:
                probs_1 = upcast(full.reshape(-1, e)) if is_probs else F.softmax(upcast(full.reshape(-1, e)), dim=-1)
                top_1 = idx.detach().reshape(-1, k).long() if (idx is not None and k != e) else (
                    probs_1.detach().topk(k, dim=-1).indices
                )
                if not _in_backward():
                    ctx.routing[(phys, 1)] = top_1
                    if div_w > 0:
                        ctx.router_probs[(phys, 1)] = probs_1.detach()
                return None

            # Pass r >= 2: compute pass-specific router logit shift + cross-pass expert repulsion
            p_idx = r - 2
            if not torch.is_grad_enabled() and pr_mod.is_zero(p_idx, core_idx):
                probs_r = upcast(full.reshape(-1, e)) if is_probs else F.softmax(upcast(full.reshape(-1, e)), dim=-1)
                top_r = idx.detach().reshape(-1, k).long() if (idx is not None and k != e) else (
                    probs_r.detach().topk(k, dim=-1).indices
                )
                ctx.routing[(phys, r)] = top_r
                return None

            prev_mask: torch.Tensor | None = None
            for pr in range(1, r):
                pidx = ctx.routing.get((phys, pr))
                if pidx is not None and pidx.shape[0] == flat_x.shape[0]:
                    if prev_mask is None:
                        prev_mask = flat_x.new_zeros(flat_x.shape[0], e, dtype=torch.float32)
                    prev_mask.scatter_(1, pidx, 1.0)

            delta = pr_mod.delta_logits(flat_x, p_idx, core_idx, prev_mask)

            if not isinstance(out, (tuple, list)):
                new_out = out + delta.to(out.dtype).view_as(out)
                probs_r = F.softmax(upcast(new_out.reshape(-1, e)), dim=-1)
                if not _in_backward():
                    ctx.routing[(phys, r)] = probs_r.detach().topk(k, dim=-1).indices
                    if div_w > 0:
                        ctx.router_probs[(phys, r)] = probs_r.detach()
                if div_w > 0:
                    for pr in range(1, r):
                        p_prev = ctx.router_probs.get((phys, pr))
                        if p_prev is not None and p_prev.shape == probs_r.shape:
                            div_term = (p_prev.to(probs_r.dtype) * probs_r).sum(dim=-1).mean()
                            if not _in_backward():
                                ctx.div_terms.append(div_term)
                return new_out

            # Tuple/list router (e.g. HF>=5 Qwen3_5MoeTopKRouter returning (probs, scores, indices))
            w_mod = getattr(mod, "weight", None)
            b_mod = getattr(mod, "bias", None)
            bias_up = upcast(b_mod) if isinstance(b_mod, torch.Tensor) else None
            acc_dt = torch.float64 if flat_x.dtype == torch.float64 else torch.float32
            base_logits: torch.Tensor | None = None
            is_fp8_w = isinstance(w_mod, torch.Tensor) and (
                w_mod.dtype in FP8_DTYPES or "float8" in str(w_mod.dtype)
            )
            _, s_attr = _first_tensor_attr(mod, _LINEAR_FP8_SCALE_ATTRS)
            if is_quantized_module(mod) or is_fp8_w or (
                isinstance(w_mod, torch.Tensor) and w_mod.ndim == 2 and s_attr is not None
            ):
                w_deq: torch.Tensor | None = None
                try:
                    if is_quantized_module(mod):
                        w_deq = _dequant_linear_weight(mod, dtype=acc_dt)
                    elif isinstance(w_mod, torch.Tensor) and s_attr is not None:
                        w_deq = dequantize_fp8_tensor(
                            w_mod, s_attr, dtype=acc_dt, block_size=_extract_block_size(mod)
                        )
                except (RuntimeError, TypeError, ValueError):
                    w_deq = None
                if isinstance(w_deq, torch.Tensor) and w_deq.shape == (e, flat_x.shape[-1]):
                    x_acc = upcast(flat_x).to(dtype=acc_dt)
                    b_acc = bias_up.to(dtype=acc_dt) if bias_up is not None else None
                    base_logits = F.linear(x_acc, w_deq, b_acc)
            elif (
                isinstance(w_mod, torch.Tensor)
                and w_mod.is_floating_point()
                and w_mod.shape == (e, flat_x.shape[-1])
            ):
                base_logits = F.linear(upcast(flat_x), upcast(w_mod), bias_up)

            if base_logits is None:
                if is_probs:
                    base_logits = upcast(full.reshape(-1, e)).clamp_min(1e-30).log()
                else:
                    base_logits = upcast(full.reshape(-1, e))
            new_logits = base_logits + delta.to(base_logits.dtype)
            probs_r = F.softmax(new_logits, dim=-1)
            top_w, top_idx = probs_r.topk(k, dim=-1)
            if norm_topk:
                top_w = top_w / top_w.sum(-1, keepdim=True)
            if not _in_backward():
                ctx.routing[(phys, r)] = top_idx.detach()
                if div_w > 0:
                    ctx.router_probs[(phys, r)] = probs_r.detach()
            if div_w > 0:
                for pr in range(1, r):
                    p_prev = ctx.router_probs.get((phys, pr))
                    if p_prev is not None and p_prev.shape == probs_r.shape:
                        div_term = (p_prev.to(probs_r.dtype) * probs_r).sum(dim=-1).mean()
                        if not _in_backward():
                            ctx.div_terms.append(div_term)
            new_full = (probs_r if is_probs else new_logits).to(full.dtype).view_as(full)
            rebuilt = []
            for item in out:
                if item is full:
                    rebuilt.append(new_full)
                elif sc is not None and item is sc:
                    rebuilt.append(top_w.to(sc.dtype).view_as(sc))
                elif idx is not None and item is idx:
                    rebuilt.append(top_idx.to(idx.dtype).view_as(idx))
                else:
                    rebuilt.append(item)
            return tuple(rebuilt) if isinstance(out, tuple) else rebuilt

        return hook

    def _route_hook(self, phys: int, k: int, n_experts: int | None):
        div_w = float(self.spec.router_diversity_weight)

        def hook(mod: nn.Module, inp: Any, out: Any):
            cur = self._cur
            if cur is None:
                return None
            ctx, r = cur
            outs = out if isinstance(out, (tuple, list)) else (out,)
            full = next((o for o in outs if isinstance(o, torch.Tensor) and o.is_floating_point() and
                         (n_experts is None or o.shape[-1] == n_experts)), None)
            if full is not None:
                e = full.shape[-1]
                is_probs = isinstance(out, (tuple, list)) and _looks_like_probs(full)
                probs = upcast(full.reshape(-1, e)) if is_probs else F.softmax(upcast(full.reshape(-1, e)), dim=-1)
                if not _in_backward():
                    ctx.routing[(phys, r)] = probs.detach().topk(k, dim=-1).indices
                    if div_w > 0 and r >= 1:
                        ctx.router_probs[(phys, r)] = probs.detach()
                if div_w > 0 and r >= 2:
                    for pr in range(1, r):
                        p_prev = ctx.router_probs.get((phys, pr))
                        if p_prev is not None and p_prev.shape == probs.shape:
                            div_term = (p_prev.to(probs.dtype) * probs).sum(dim=-1).mean()
                            if not _in_backward():
                                ctx.div_terms.append(div_term)
            return None
        return hook

    # ------------------------------------------------------------------ forward-time helpers
    def _begin_forward(self) -> None:
        self.ctx = LoopContext()

    def release_graph(self, *, clear_routing: bool = False) -> None:
        """Release live autograd tensors held in `self.ctx` after backward or when skipping backward."""
        self.ctx.release_graph(clear_routing=clear_routing)

    def _route(self, ctx: LoopContext, x: torch.Tensor) -> None:
        router = self.adapters.mor_router
        assert router is not None
        p = F.softmax(router(upcast(x).to(router.weight.dtype)), dim=-1)                   # [B,T,R]
        depth = p.argmax(-1) + 1                                    # hard, causal, per token
        for r in range(2, self.spec.num_passes + 1):
            keep = (depth >= r).float()
            soft = p[..., r - 1:].sum(-1)
            ctx.keep[r] = keep
            ctx.coef[r] = (keep * (1.0 + soft - soft.detach())).unsqueeze(-1)  # value=keep, grad via soft
        exp_depth = (p * torch.arange(1, p.shape[-1] + 1, device=p.device, dtype=p.dtype)).sum(-1)
        ctx.stats["mor/mean_depth_hard"] = float(depth.clamp(max=self.active_passes).float().mean())
        if self.spec.mor_target_depth is not None:
            ctx.aux["mor_depth"] = self.spec.mor_aux_weight * (exp_depth.mean() - self.spec.mor_target_depth) ** 2

    def _combine(self, ctx: LoopContext, h: torch.Tensor, r: int) -> torch.Tensor:
        start = ctx.start[r]
        coef = self.adapters.gates[r - 2].to(h.dtype)
        if r in ctx.coef:
            coef = coef * ctx.coef[r].to(h.dtype)
        return start + coef * (h - start)

    # ------------------------------------------------------------------ public API
    @property
    def active_virtual_layers(self) -> int:
        return (
            self.spec.prelude_end
            + self.spec.core_len * self.active_passes
            + (self.parts.num_layers - self.spec.core_end)
        )

    def set_active_passes(self, r: int) -> None:
        if not 1 <= r <= self.spec.num_passes:
            raise ValueError(f"active passes must be in [1, {self.spec.num_passes}], got {r}")
        self.active_passes = r

    def set_router_repel(self, value: float) -> None:
        """Set cross-pass expert repulsion logit penalty across all core layers (runtime knob)."""
        if self.adapters.pass_router is None:
            raise RuntimeError("per_pass_router is disabled on this LoopController")
        with torch.no_grad():
            self.adapters.pass_router.repel.fill_(float(value))

    def sample_active_passes(self, rng: random.Random | None = None) -> int:
        probs = self.spec.train_pass_probs
        r = self.spec.num_passes if not probs else (rng or random).choices(
            range(1, self.spec.num_passes + 1), weights=probs)[0]
        self.set_active_passes(r)
        return r

    def aux_loss(self) -> torch.Tensor | None:
        vals = [v.float() for v in self.ctx.aux.values()]
        if self.ctx.div_terms and self.spec.router_diversity_weight > 0:
            div_mean = torch.stack([t.float() for t in self.ctx.div_terms]).mean()
            vals.append(float(self.spec.router_diversity_weight) * div_mean)
        return torch.stack(vals).sum() if vals else None

    def trainable_parameters(self) -> list[nn.Parameter]:
        return list(self.adapters.parameters())

    def routing_overlap(self, pass_a: int = 1, pass_b: int = 2) -> dict[int, float]:
        """Mean per-token Jaccard of top-k expert sets between `pass_a` and `pass_b`, per core layer."""
        out = {}
        for (phys, r), idx2 in self.ctx.routing.items():
            if r != pass_b or (phys, pass_a) not in self.ctx.routing:
                continue
            idx1 = self.ctx.routing[(phys, pass_a)]
            if idx1.shape != idx2.shape:
                continue
            inter = (idx1.unsqueeze(-1) == idx2.unsqueeze(-2)).any(-1).sum(-1).float()
            k = idx1.shape[-1]
            out[phys] = float((inter / (2 * k - inter).clamp_min(1.0)).mean())
        return out

    def reroute_rates(self, pass_r: int = 2) -> dict[int, float]:
        """Fraction of `pass_r` top-k slots routed to experts NOT selected in passes `1..pass_r-1`."""
        out = {}
        for (phys, r), idx_r in self.ctx.routing.items():
            if r != pass_r:
                continue
            prev_sets = [self.ctx.routing[(phys, pr)] for pr in range(1, pass_r) if (phys, pr) in self.ctx.routing]
            if not prev_sets or any(ps.shape != idx_r.shape for ps in prev_sets):
                continue
            prev_cat = torch.cat(prev_sets, dim=-1)
            reused = (idx_r.unsqueeze(-1) == prev_cat.unsqueeze(-2)).any(-1).sum(-1).float()
            k = max(int(idx_r.shape[-1]), 1)
            out[phys] = float(((k - reused) / k).mean())
        return out

    def unique_experts_per_layer(self) -> dict[int, float]:
        """Mean unique experts activated per token across active passes [1..R] for each core layer."""
        if self._num_experts is None:
            return {}
        e = self._num_experts
        out: dict[int, float] = {}
        for phys in range(self.spec.prelude_end, self.spec.core_end):
            sets = [self.ctx.routing[(phys, r)] for r in range(1, self.active_passes + 1)
                    if (phys, r) in self.ctx.routing]
            if not sets:
                continue
            n_tok = sets[0].shape[0]
            if any(s.shape[0] != n_tok for s in sets):
                continue
            union = torch.zeros(n_tok, e, dtype=torch.bool, device=sets[0].device)
            for s in sets:
                union.scatter_(1, s, True)
            out[phys] = float(union.sum(-1).float().mean())
        return out

    def stats(self) -> dict[str, float]:
        s = dict(self.ctx.stats)
        with torch.no_grad():
            for i in range(self.spec.num_passes - 1):
                s[f"loop/gate_abs_mean_p{i + 2}"] = float(self.adapters.gates[i].abs().mean())
                w = self.adapters.bridge.weights(i)
                for j, nm in enumerate(LoopBridge.STREAMS):
                    s[f"loop/bridge_w_{nm}_p{i + 2}"] = float(w[j])
                if self.adapters.pass_attn is not None:
                    s[f"loop/attn_lora_abs_mean_p{i + 2}"] = self.adapters.pass_attn.up_abs_mean(i)
            if self.adapters.pass_router is not None:
                s["loop/router_repel_mean"] = float(self.adapters.pass_router.repel.mean())
            ov = self.routing_overlap(1, 2)
            if ov:
                s["loop/expert_jaccard_p12"] = sum(ov.values()) / len(ov)
            rr = self.reroute_rates(2)
            if rr:
                s["loop/reroute_rate"] = sum(rr.values()) / len(rr)
            if self.active_passes >= 3:
                ov13 = self.routing_overlap(1, 3)
                if ov13:
                    s["loop/expert_jaccard_p13"] = sum(ov13.values()) / len(ov13)
                ov23 = self.routing_overlap(2, 3)
                if ov23:
                    s["loop/expert_jaccard_p23"] = sum(ov23.values()) / len(ov23)
                rr3 = self.reroute_rates(3)
                if rr3:
                    s["loop/reroute_rate_p3"] = sum(rr3.values()) / len(rr3)
            uniq = self.unique_experts_per_layer()
            if uniq:
                mean_u = sum(uniq.values()) / len(uniq)
                s["loop/unique_experts_core"] = mean_u
                n_phys = self.parts.num_layers
                k = max(self._top_k, 1)
                non_core = max(n_phys - self.spec.core_len, 0) * k
                s["loop/effective_expert_mult"] = (non_core + sum(uniq.values())) / max(n_phys * k, 1)
            if self.ctx.div_terms:
                divs = [t.detach().float() for t in self.ctx.div_terms]
                s["loop/router_prob_overlap"] = float(torch.stack(divs).mean())
        s["loop/active_passes"] = float(self.active_passes)
        s["loop/active_virtual_layers"] = float(self.active_virtual_layers)
        return s

    def unwrap(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
        for mod, had_inst_fwd, orig_inst_fwd in reversed(self._wrapped_fwds):
            if had_inst_fwd:
                mod.forward = orig_inst_fwd  # type: ignore[method-assign]
            else:
                mod.__dict__.pop("forward", None)
        self._wrapped_fwds.clear()
        tm, cfg = self.parts.text_model, self.parts.text_config
        tm.layers = self._orig["layers"]
        cfg.num_hidden_layers = self._orig["num_hidden_layers"]
        cfg.layer_types = self._orig["layer_types"]
        for m, idx in self._orig["layer_idx"]:
            m.layer_idx = idx
        if hasattr(tm, "loop_adapters"):
            del tm.loop_adapters
        log.info("loop retrofit removed (adapters kept on controller.adapters)")


_QUANT_EXPERT_DICT_SUFFIXES = ("_packed", "_scale", "_scales", "_scale_inv", "_global_scale", "_scale_2")
_QUANT_EXPERT_DICT_EXACT = frozenset({"gate_up_proj", "down_proj", "weight", "bias"})


def _iter_canonical_params(
    mod: nn.Module,
    prefix: str,
    seen_tensors: set[int],
    seen_mods: set[int] | None = None,
    seen_names: set[str] | None = None,
) -> list[tuple[str, torch.Tensor]]:
    if seen_mods is None:
        seen_mods = set()
    if seen_names is None:
        seen_names = set()
    if id(mod) in seen_mods:
        return []
    seen_mods.add(id(mod))

    out: list[tuple[str, torch.Tensor]] = []
    for pname, p in mod._parameters.items():
        if p is None:
            continue
        full = f"{prefix}.{pname}" if prefix else pname
        if full not in seen_names and id(p) not in seen_tensors:
            seen_names.add(full)
            seen_tensors.add(id(p))
            out.append((full, p))
    for bname, b in mod._buffers.items():
        if b is None:
            continue
        full = f"{prefix}.{bname}" if prefix else bname
        if full not in seen_names and id(b) not in seen_tensors:
            seen_names.add(full)
            seen_tensors.add(id(b))
            out.append((full, b))
    for aname, val in mod.__dict__.items():
        if aname.startswith("_") or aname in mod._parameters or aname in mod._buffers:
            continue
        if not isinstance(val, torch.Tensor):
            continue
        if aname in _QUANT_EXPERT_DICT_EXACT or aname.endswith(_QUANT_EXPERT_DICT_SUFFIXES):
            full = f"{prefix}.{aname}" if prefix else aname
            if full not in seen_names and id(val) not in seen_tensors:
                seen_names.add(full)
                seen_tensors.add(id(val))
                out.append((full, val))
    for cname, child in mod._modules.items():
        if child is None:
            continue
        slot = getattr(child, "slot", None)
        meta = getattr(child, "_meta", None)
        inner = getattr(child, "inner", None)
        if slot is not None and isinstance(inner, nn.Module) and hasattr(slot, "phys"):
            c_prefix = f"{prefix}.{slot.phys}" if prefix else str(slot.phys)
            out.extend(_iter_canonical_params(inner, c_prefix, seen_tensors, seen_mods, seen_names))
        elif isinstance(meta, tuple) and len(meta) >= 2 and isinstance(inner, nn.Module):
            c_prefix = f"{prefix}.{meta[1]}" if prefix else str(meta[1])
            out.extend(_iter_canonical_params(inner, c_prefix, seen_tensors, seen_mods, seen_names))
        else:
            c_prefix = f"{prefix}.{cname}" if prefix else cname
            out.extend(_iter_canonical_params(child, c_prefix, seen_tensors, seen_mods, seen_names))
    return out


_EXPERT_PATH_PARTS = frozenset({"experts", "shared_expert", "shared_experts"})


def _is_expert_param_path(name: str) -> bool:
    parts = name.split(".")
    if "expert_deltas" in parts:
        return False
    for seg in parts:
        if seg in _EXPERT_PATH_PARTS:
            return True
        if seg.startswith(("experts_", "shared_expert_")) and not seg.endswith("_gate"):
            return True
    return False


def sha256_frozen_experts(model: nn.Module) -> str:
    """Compute a deterministic SHA-256 digest over frozen MoE expert parameters and buffers.

    Covers all `experts` and `shared_expert` parameters and quantized buffers with
    `requires_grad == False`, canonicalising `VirtualLayer` wrappers back to physical layer
    indices so the digest is identical before, during, and after `apply_loop`.
    """
    seen_tensors: set[int] = set()
    items = [
        (name, t)
        for name, t in _iter_canonical_params(model, "", seen_tensors)
        if not t.requires_grad and _is_expert_param_path(name)
    ]
    items.sort(key=lambda kv: kv[0])
    digest = hashlib.sha256()
    for name, t in items:
        header = f"{name}|{t.dtype}|{tuple(t.shape)}|".encode()
        digest.update(header)
        raw = t.detach().contiguous().reshape(-1).view(torch.uint8).cpu().numpy().tobytes()
        digest.update(raw)
    return digest.hexdigest()


def apply_loop(model: nn.Module, spec: LoopSpec | dict[str, Any]) -> LoopController:
    if isinstance(spec, dict):
        spec = LoopSpec.from_dict(spec)
    return LoopController(model, spec)
