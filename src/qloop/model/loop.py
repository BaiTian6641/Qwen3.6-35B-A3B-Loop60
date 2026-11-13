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

import logging
import random
from dataclasses import dataclass, field, fields
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from ._dtype import upcast
from .introspect import FULL, LINEAR, attention_module, find_children, inspect_model, modules_with_layer_idx, router_of
from .mhc import LoopBridge

log = logging.getLogger("qloop")


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
    step_embedding: bool = True
    gate_init: float = 0.0
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
        for k in ("bridge_init", "kv_proj_names", "write_zero_names", "write_neg_names"):
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
        self.kv_stash: dict[tuple[int, str], torch.Tensor] = {}
        self.keep: dict[int, torch.Tensor] = {}       # pass -> [B,T] {0,1}
        self.coef: dict[int, torch.Tensor] = {}       # pass -> [B,T,1] straight-through MoR coef
        self.aux: dict[str, torch.Tensor] = {}
        self.routing: dict[tuple[int, int], torch.Tensor] = {}
        self.stats: dict[str, float] = {}


class SharedKVView:
    """Cache proxy for pass>=2 full attention: `update` appends nothing, returns pass-1 K/V."""

    def __init__(self, cache: Any):
        object.__setattr__(self, "_cache", cache)

    def __getattr__(self, name: str) -> Any:
        return getattr(object.__getattribute__(self, "_cache"), name)

    def __setattr__(self, name: str, value: Any) -> None:
        setattr(object.__getattribute__(self, "_cache"), name, value)

    def __getitem__(self, idx: Any) -> Any:
        return object.__getattribute__(self, "_cache")[idx]

    def update(self, key_states, value_states, layer_idx, *args, **kwargs):
        cache = object.__getattribute__(self, "_cache")
        return cache.update(key_states[..., :0, :], value_states[..., :0, :], layer_idx, *args, **kwargs)


class LoopAdapters(nn.Module):
    """All new parameters (kept fp32; tiny). Attached as `text_model.loop_adapters`."""

    def __init__(self, hidden: int, spec: LoopSpec):
        super().__init__()
        extra = max(spec.num_passes - 1, 1)
        self.gates = nn.Parameter(torch.full((extra, hidden), float(spec.gate_init)))
        self.bridge = LoopBridge(hidden, extra, spec.bridge_init, spec.bridge_rank, spec.step_embedding)
        self.mor_router: nn.Linear | None = None
        if spec.mor:
            self.mor_router = nn.Linear(hidden, spec.num_passes)
            nn.init.zeros_(self.mor_router.weight)
            with torch.no_grad():  # start with every token taking the full depth
                self.mor_router.bias.zero_()
                self.mor_router.bias[-1] = 2.0


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
            return (hidden_states,) if ctl._tuple_out else hidden_states
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
                x = ctl.adapters.bridge(x, ctx.anchor, s.r - 2)
        pkv = kwargs.get("past_key_values")
        if pkv is not None and s.cache_idx != s.vidx:
            kwargs = dict(kwargs)
            kwargs["past_key_values"] = SharedKVView(pkv)
        r, idx_mods, cache_idx, inner = s.r, self._idx_mods, s.cache_idx, self.inner

        def run(inp: torch.Tensor) -> Any:
            prev = ctl._cur
            ctl._cur = (ctx, r)
            for m in idx_mods:
                m.layer_idx = cache_idx
            try:
                return inner(inp, *args, **kwargs)
            finally:
                ctl._cur = prev

        if ctl.spec.checkpoint and self.training and torch.is_grad_enabled():
            out = checkpoint(run, x, use_reentrant=False)
        else:
            out = run(x)
        is_tuple = isinstance(out, tuple)
        ctl._tuple_out = is_tuple
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
        ref = tm.embed_tokens.weight
        adapter_dtype = torch.float64 if ref.dtype == torch.float64 else torch.float32
        self.adapters = LoopAdapters(cfg.hidden_size, spec).to(device=ref.device, dtype=adapter_dtype)
        tm.loop_adapters = self.adapters
        self.active_passes = spec.num_passes
        self.ctx = LoopContext()
        self._cur: tuple[LoopContext, int] | None = None
        self._tuple_out = False
        self._hooks: list[Any] = []
        tm.layers = nn.ModuleList(VirtualLayer(self, s, parts.layers[s.phys]) for s in self.schedule)
        cfg.num_hidden_layers = len(self.schedule)
        cfg.layer_types = [s.layer_type for s in self.schedule]
        self._register_hooks()
        log.info("loop retrofit: %d physical -> %d virtual layers (prelude %d | core %d x %d | coda %d), "
                 "kv_share=%s mor=%s checkpoint=%s", parts.num_layers, len(self.schedule), spec.prelude_end,
                 spec.core_len, spec.num_passes, parts.num_layers - spec.core_end, spec.kv_share, spec.mor,
                 spec.checkpoint)

    # ------------------------------------------------------------------ hooks
    def _core_layers(self, layer_type: str) -> list[tuple[int, nn.Module]]:
        lo, hi = self.spec.prelude_end, self.spec.core_end
        return [(i, self._orig["layers"][i]) for i in range(lo, hi) if self.parts.layer_types[i] == layer_type]

    def _register_hooks(self) -> None:
        spec = self.spec
        if spec.kv_share and spec.num_passes > 1:
            for phys, layer in self._core_layers(FULL):
                found = find_children(attention_module(layer, FULL), list(spec.kv_proj_names))
                if len(found) != len(spec.kv_proj_names):
                    raise RuntimeError(f"kv_share: layer {phys} attention lacks {spec.kv_proj_names}; children: "
                                       f"{[n for n, _ in attention_module(layer, FULL).named_children()]}")
                for name, mod in found.items():
                    self._hooks.append(mod.register_forward_hook(self._kv_hook(phys, name)))
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
        if spec.collect_routing:
            k = getattr(self.parts.text_config, "num_experts_per_tok", 8)
            e = getattr(self.parts.text_config, "num_experts", None)
            for phys, layer in [(i, self._orig["layers"][i]) for i in range(spec.prelude_end, spec.core_end)]:
                router = router_of(layer)
                if router is not None:
                    self._hooks.append(router.register_forward_hook(self._route_hook(phys, k, e)))

    def _kv_hook(self, phys: int, name: str):
        def hook(mod: nn.Module, inp: Any, out: torch.Tensor):
            cur = self._cur
            if cur is None:
                return None
            ctx, r = cur
            key = (phys, name)
            if r == 1:
                ctx.kv_stash[key] = out
                return None
            stash = ctx.kv_stash.get(key)
            if stash is None or stash.shape != out.shape:
                raise RuntimeError(f"kv_share: missing/mismatched pass-1 {name} for layer {phys} at pass {r} "
                                   f"(stash={None if stash is None else tuple(stash.shape)}, now={tuple(out.shape)})")
            return stash
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

    def _route_hook(self, phys: int, k: int, n_experts: int | None):
        def hook(mod: nn.Module, inp: Any, out: Any):
            cur = self._cur
            if cur is None:
                return None
            outs = out if isinstance(out, (tuple, list)) else (out,)
            logits = next((o for o in outs if isinstance(o, torch.Tensor) and
                           (n_experts is None or o.shape[-1] == n_experts)), None)
            if logits is not None:
                cur[0].routing[(phys, cur[1])] = logits.detach().topk(k, dim=-1).indices.reshape(-1, k)
            return None
        return hook

    # ------------------------------------------------------------------ forward-time helpers
    def _begin_forward(self) -> None:
        self.ctx = LoopContext()

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
        ctx.stats["mor/mean_depth_hard"] = float(depth.float().mean())
        if self.spec.mor_target_depth is not None:
            ctx.aux["mor_depth"] = self.spec.mor_aux_weight * (exp_depth.mean() - self.spec.mor_target_depth) ** 2

    def _combine(self, ctx: LoopContext, h: torch.Tensor, r: int) -> torch.Tensor:
        start = ctx.start[r]
        coef = self.adapters.gates[r - 2].to(h.dtype)
        if r in ctx.coef:
            coef = coef * ctx.coef[r].to(h.dtype)
        return start + coef * (h - start)

    # ------------------------------------------------------------------ public API
    def set_active_passes(self, r: int) -> None:
        if not 1 <= r <= self.spec.num_passes:
            raise ValueError(f"active passes must be in [1, {self.spec.num_passes}], got {r}")
        self.active_passes = r

    def sample_active_passes(self, rng: random.Random | None = None) -> int:
        probs = self.spec.train_pass_probs
        r = self.spec.num_passes if not probs else (rng or random).choices(
            range(1, self.spec.num_passes + 1), weights=probs)[0]
        self.set_active_passes(r)
        return r

    def aux_loss(self) -> torch.Tensor | None:
        vals = list(self.ctx.aux.values())
        return torch.stack([v.float() for v in vals]).sum() if vals else None

    def trainable_parameters(self) -> list[nn.Parameter]:
        return list(self.adapters.parameters())

    def routing_overlap(self) -> dict[int, float]:
        """Mean per-token Jaccard of top-k expert sets between pass 1 and pass 2, per core layer."""
        out = {}
        for (phys, r), idx2 in self.ctx.routing.items():
            if r != 2 or (phys, 1) not in self.ctx.routing:
                continue
            idx1 = self.ctx.routing[(phys, 1)]
            inter = (idx1.unsqueeze(-1) == idx2.unsqueeze(-2)).any(-1).sum(-1).float()
            k = idx1.shape[-1]
            out[phys] = float((inter / (2 * k - inter)).mean())
        return out

    def stats(self) -> dict[str, float]:
        s = dict(self.ctx.stats)
        with torch.no_grad():
            for i in range(self.spec.num_passes - 1):
                s[f"loop/gate_abs_mean_p{i + 2}"] = float(self.adapters.gates[i].abs().mean())
                w = self.adapters.bridge.weights(i)
                for j, nm in enumerate(LoopBridge.STREAMS):
                    s[f"loop/bridge_w_{nm}_p{i + 2}"] = float(w[j])
        s["loop/active_passes"] = float(self.active_passes)
        return s

    def unwrap(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
        tm, cfg = self.parts.text_model, self.parts.text_config
        tm.layers = self._orig["layers"]
        cfg.num_hidden_layers = self._orig["num_hidden_layers"]
        cfg.layer_types = self._orig["layer_types"]
        for m, idx in self._orig["layer_idx"]:
            m.layer_idx = idx
        if hasattr(tm, "loop_adapters"):
            del tm.loop_adapters
        log.info("loop retrofit removed (adapters kept on controller.adapters)")


def apply_loop(model: nn.Module, spec: LoopSpec | dict[str, Any]) -> LoopController:
    if isinstance(spec, dict):
        spec = LoopSpec.from_dict(spec)
    return LoopController(model, spec)
