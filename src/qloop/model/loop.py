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
import math
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
    step_embedding: bool = True
    gate_init: float = 0.0
    per_pass_router: bool = True
    router_adapter_rank: int = 32
    router_repel_init: float = 0.0
    router_diversity_weight: float = 0.0
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
        self.div_terms: list[torch.Tensor] = []
        self.routing: dict[tuple[int, int], torch.Tensor] = {}
        self.router_probs: dict[tuple[int, int], torch.Tensor] = {}
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


class LoopAdapters(nn.Module):
    """All new parameters (kept fp32; tiny). Attached as `text_model.loop_adapters`."""

    def __init__(self, hidden: int, spec: LoopSpec, num_experts: int | None = None):
        super().__init__()
        extra = max(spec.num_passes - 1, 1)
        self.gates = nn.Parameter(torch.full((extra, hidden), float(spec.gate_init)))
        self.bridge = LoopBridge(hidden, extra, spec.bridge_init, spec.bridge_rank, spec.step_embedding)
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
        has_core_routers = any(router_of(parts.layers[i]) is not None for i in range(spec.prelude_end, spec.core_end))
        num_experts = getattr(cfg, "num_experts", None) if has_core_routers else None
        self._num_experts: int | None = int(num_experts) if num_experts is not None else None
        self._top_k: int = int(getattr(cfg, "num_experts_per_tok", 8))
        self._norm_topk: bool = bool(getattr(cfg, "norm_topk_prob", True))
        ref = tm.embed_tokens.weight
        adapter_dtype = torch.float64 if ref.dtype == torch.float64 else torch.float32
        self.adapters = LoopAdapters(cfg.hidden_size, spec, num_experts=self._num_experts).to(
            device=ref.device, dtype=adapter_dtype
        )
        tm.loop_adapters = self.adapters
        self.active_passes = spec.num_passes
        self.ctx = LoopContext()
        self._cur: tuple[LoopContext, int] | None = None
        self._tuple_out = False
        self._hooks: list[Any] = []
        self._is_probs_cache: dict[int, bool] = {}
        tm.layers = nn.ModuleList(VirtualLayer(self, s, parts.layers[s.phys]) for s in self.schedule)
        cfg.num_hidden_layers = len(self.schedule)
        cfg.layer_types = [s.layer_type for s in self.schedule]
        self._register_hooks()
        log.info("loop retrofit: %d physical -> %d virtual layers (prelude %d | core %d x %d | coda %d), "
                 "kv_share=%s per_pass_router=%s mor=%s checkpoint=%s", parts.num_layers, len(self.schedule),
                 spec.prelude_end, spec.core_len, spec.num_passes, parts.num_layers - spec.core_end, spec.kv_share,
                 self.adapters.pass_router is not None, spec.mor, spec.checkpoint)

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
                if div_w > 0 and (phys, 1) in ctx.router_probs:
                    p1 = ctx.router_probs[(phys, 1)]
                    if p1.shape == probs_r.shape:
                        div_term = (p1.to(probs_r.dtype) * probs_r).sum(dim=-1).mean()
                        if not _in_backward():
                            ctx.div_terms.append(div_term)
                return new_out

            # Tuple/list router (e.g. HF>=5 Qwen3_5MoeTopKRouter returning (probs, scores, indices))
            w_mod = getattr(mod, "weight", None)
            if isinstance(w_mod, torch.Tensor) and w_mod.shape == (e, flat_x.shape[-1]):
                b_mod = getattr(mod, "bias", None)
                bias_up = upcast(b_mod) if isinstance(b_mod, torch.Tensor) else None
                base_logits = F.linear(upcast(flat_x), upcast(w_mod), bias_up)
            elif is_probs:
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
            if div_w > 0 and (phys, 1) in ctx.router_probs:
                p1 = ctx.router_probs[(phys, 1)]
                if p1.shape == probs_r.shape:
                    div_term = (p1.to(probs_r.dtype) * probs_r).sum(dim=-1).mean()
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
                    if div_w > 0 and r == 1:
                        ctx.router_probs[(phys, 1)] = probs.detach()
                if div_w > 0 and r >= 2 and (phys, 1) in ctx.router_probs:
                    p1 = ctx.router_probs[(phys, 1)]
                    if p1.shape == probs.shape:
                        div_term = (p1.to(probs.dtype) * probs).sum(dim=-1).mean()
                        if not _in_backward():
                            ctx.div_terms.append(div_term)
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

    def routing_overlap(self) -> dict[int, float]:
        """Mean per-token Jaccard of top-k expert sets between pass 1 and pass 2, per core layer."""
        out = {}
        for (phys, r), idx2 in self.ctx.routing.items():
            if r != 2 or (phys, 1) not in self.ctx.routing:
                continue
            idx1 = self.ctx.routing[(phys, 1)]
            if idx1.shape != idx2.shape:
                continue
            inter = (idx1.unsqueeze(-1) == idx2.unsqueeze(-2)).any(-1).sum(-1).float()
            k = idx1.shape[-1]
            out[phys] = float((inter / (2 * k - inter).clamp_min(1.0)).mean())
        return out

    def reroute_rates(self) -> dict[int, float]:
        """Fraction of pass-2 top-k slots routed to experts NOT selected in pass 1, per core layer."""
        out = {}
        for (phys, r), idx2 in self.ctx.routing.items():
            if r != 2 or (phys, 1) not in self.ctx.routing:
                continue
            idx1 = self.ctx.routing[(phys, 1)]
            if idx1.shape != idx2.shape:
                continue
            inter = (idx1.unsqueeze(-1) == idx2.unsqueeze(-2)).any(-1).sum(-1).float()
            k = max(int(idx1.shape[-1]), 1)
            out[phys] = float(((k - inter) / k).mean())
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
            if self.adapters.pass_router is not None:
                s["loop/router_repel_mean"] = float(self.adapters.pass_router.repel.mean())
            ov = self.routing_overlap()
            if ov:
                s["loop/expert_jaccard_p12"] = sum(ov.values()) / len(ov)
            rr = self.reroute_rates()
            if rr:
                s["loop/reroute_rate"] = sum(rr.values()) / len(rr)
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
