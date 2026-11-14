"""Expert-specialised fine-tuning (ESFT) primitives for fused-3D-expert MoE models.

* `scan_router_affinity`  - forward hooks on every router (`layer.mlp.gate`) accumulating, per
  (physical layer, loop pass, expert): softmax probability mass, *gate affinity* (ESFT-Gate: the
  top-k renormalised routing weight actually used, 0 when not selected) and top-k selection counts
  (ESFT-Token). Works on plain and loop-wrapped models (the shared physical router runs once per
  pass; the pass index is read from the loop controller's `_cur`).
* `ExpertStats.select_hot` - choose "hot" experts per layer (top fraction / top-k / ESFT
  cumulative-mass threshold), union-over-passes or pass-1 only.
* `HotExpertDeltas`  - zero-initialised trainable weight deltas for the hot experts that never
  touch the frozen fused (possibly quantised) base tensors `experts.gate_up_proj [E, 2I, H]` /
  `experts.down_proj [E, H, I]`. Hooks recompute Qwen's routing from the router output of *this*
  call (so a layer executed twice per forward in the looped model uses fresh routing each time)
  and add, for every token routed to a hot expert e with weight w,

        w * (Expert_{W+dW}(x) - Expert_W(x))

  computed exactly through the factorisation (SwiGLU expert, act = silu)

        gu_b = W_gu x                gu_m = gu_b + dW_gu x
        h_b  = act(g_b) * u_b        h_m  = act(g_m) * u_m
        Expert_{W+dW}(x) - Expert_W(x) = W_d (h_m - h_b) + dW_d h_m

  which is algebraically identical to "merged expert minus base expert" but never materialises a
  merged weight and loses less precision in bf16. Zero deltas give an exact 0 => the model output
  is bit-identical at init.
* `freeze_all_but` - freeze everything except regex-selected parameters (+ explicit extras) and
  return a per-group report; refuses accidental training of the fused expert base tensors.
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any, Callable

import torch
import torch.nn.functional as F
from torch import nn

from qloop.debug import describe, require

from ._dtype import upcast
from .introspect import find_text_model
from .quant_autograd import auto_dequant_expert_fn

log = logging.getLogger("qloop")

DequantFn = Callable[[nn.Module, int], "tuple[torch.Tensor, torch.Tensor]"]


# ============================================================================ model plumbing

def physical_layers(model: nn.Module) -> tuple[list[tuple[int, nn.Module]], Any, Any]:
    """[(physical_idx, decoder_layer)], text config, loop controller (or None).

    Loop-wrapped models (qloop.model.loop) hold `VirtualLayer`s with `.slot.phys` / `.inner`;
    every physical layer is returned once.
    """
    tm = find_text_model(model)
    cfg = getattr(tm, "config", None)
    out: dict[int, nn.Module] = {}
    ctl = None
    for i, layer in enumerate(tm.layers):
        slot, inner = getattr(layer, "slot", None), getattr(layer, "inner", None)
        if slot is not None and isinstance(inner, nn.Module):
            out.setdefault(int(slot.phys), inner)
            ctl = ctl or getattr(layer, "_ctl", None)
        else:
            out.setdefault(i, layer)
    return sorted(out.items()), cfg, ctl


def _moe_of(layer: nn.Module, idx: int) -> nn.Module:
    mlp = getattr(layer, "mlp", None)
    require(mlp is not None and hasattr(mlp, "gate") and hasattr(mlp, "experts"),
            f"layer {idx}: expected a sparse MoE block `mlp` with `.gate` and `.experts`",
            children=[n for n, _ in layer.named_children()])
    return mlp


@dataclass
class RouteInfo:
    probs: torch.Tensor     # [N, E] fp32/fp64 softmax over experts (may carry autograd graph)
    top_w: torch.Tensor     # [N, k] routing weights actually used (renormalised if norm_topk_prob)
    top_idx: torch.Tensor   # [N, k] int64 expert ids


def _looks_like_probs(t: torch.Tensor | None) -> bool:
    """Heuristic: HF>=5 routers return softmax probabilities in the "logits" slot."""
    if t is None:
        return False
    with torch.no_grad():
        f = upcast(t.detach())
        if f.numel() == 0:
            return False
        return bool((f >= 0).all() and ((f.sum(-1) - 1).abs() < 1e-3).all())


def parse_router_output(out: Any, num_experts: int, top_k: int, norm_topk: bool,
                        is_probs: bool | None = None) -> RouteInfo:
    """Turn whatever the router returned into Qwen's routing decision.

    Handles (a) a plain logits tensor `[N, E]` (older HF / toy: `nn.Linear` router) and
    (b) HF>=5 `TopKRouter` tuples `(router_logits_or_probs [N,E], scores [N,k], indices [N,k])`
    (if present, scores/indices are used verbatim - exactly what the experts consume).
    Recomputation follows Qwen: softmax over experts in fp32 -> top-k -> renormalise.
    """
    outs = out if isinstance(out, (tuple, list)) else (out,)
    tensors = [o for o in outs if isinstance(o, torch.Tensor)]
    full = next((o for o in tensors if o.is_floating_point() and o.shape[-1] == num_experts), None)
    require(full is not None, f"router output has no float tensor with last dim == num_experts={num_experts}",
            outputs=[describe(o) for o in outs])
    full = full.reshape(-1, num_experts)
    if is_probs is None:
        is_probs = _looks_like_probs(full)
    probs = upcast(full) if is_probs else F.softmax(upcast(full), dim=-1)
    idx = next((o for o in tensors if not o.is_floating_point() and o.shape[-1] == top_k), None)
    sc = next((o for o in tensors if o.is_floating_point() and o.shape[-1] == top_k and o is not full), None)
    if idx is not None and sc is not None and top_k != num_experts:
        top_w, top_idx = sc.reshape(-1, top_k), idx.reshape(-1, top_k).long()
    else:
        top_w, top_idx = probs.topk(top_k, dim=-1)
        if norm_topk:
            top_w = top_w / top_w.sum(-1, keepdim=True)
    return RouteInfo(probs, top_w, top_idx)


def _moe_dims(cfg: Any, num_experts: int | None, top_k: int | None,
              norm_topk: bool | None) -> tuple[int, int, bool]:
    e = num_experts if num_experts is not None else getattr(cfg, "num_experts", None)
    k = top_k if top_k is not None else getattr(cfg, "num_experts_per_tok", None)
    n = norm_topk if norm_topk is not None else bool(getattr(cfg, "norm_topk_prob", True))
    require(e is not None and k is not None, "num_experts / top_k not given and not found on the text config",
            config=type(cfg).__name__)
    return int(e), int(k), n


# ============================================================================ affinity scan

def _key(layer: int, r: int) -> str:
    return f"{layer}:{r}"


def _unkey(s: str) -> tuple[int, int]:
    a, b = s.split(":")
    return int(a), int(b)


@dataclass
class ExpertStats:
    """Per (layer, pass) router statistics. Pass 0 = prelude/coda or un-looped model."""

    num_experts: int
    top_k: int
    gate_mass: dict[tuple[int, int], torch.Tensor] = field(default_factory=dict)  # [E] sum of used weights
    prob_mass: dict[tuple[int, int], torch.Tensor] = field(default_factory=dict)  # [E] sum of softmax probs
    counts: dict[tuple[int, int], torch.Tensor] = field(default_factory=dict)     # [E] top-k selections
    tokens: dict[tuple[int, int], int] = field(default_factory=dict)

    # ---------------------------------------------------------------- accumulation
    def add(self, layer: int, r: int, route: RouteInfo) -> None:
        key = (layer, r)
        e = self.num_experts
        probs = route.probs.detach()
        dt = torch.float32 if probs.device.type == "mps" else torch.float64
        gate = torch.zeros(probs.shape[0], e, device=probs.device, dtype=dt)
        gate.scatter_add_(1, route.top_idx, route.top_w.detach().to(dt))
        pm = probs.to(dt).sum(0).cpu().double()
        gm = gate.sum(0).cpu().double()
        cnt = torch.bincount(route.top_idx.reshape(-1), minlength=e).cpu().long()
        if key in self.tokens:
            self.prob_mass[key] += pm
            self.gate_mass[key] += gm
            self.counts[key] += cnt
            self.tokens[key] += int(probs.shape[0])
        else:
            self.prob_mass[key], self.gate_mass[key], self.counts[key] = pm, gm, cnt
            self.tokens[key] = int(probs.shape[0])

    def merge(self, other: ExpertStats) -> ExpertStats:
        require(other.num_experts == self.num_experts and other.top_k == self.top_k,
                "cannot merge ExpertStats with different num_experts/top_k")
        for key in other.tokens:
            if key in self.tokens:
                self.prob_mass[key] = self.prob_mass[key] + other.prob_mass[key]
                self.gate_mass[key] = self.gate_mass[key] + other.gate_mass[key]
                self.counts[key] = self.counts[key] + other.counts[key]
                self.tokens[key] += other.tokens[key]
            else:
                self.prob_mass[key] = other.prob_mass[key].clone()
                self.gate_mass[key] = other.gate_mass[key].clone()
                self.counts[key] = other.counts[key].clone()
                self.tokens[key] = other.tokens[key]
        return self

    # ---------------------------------------------------------------- queries
    def layers(self) -> list[int]:
        return sorted({k[0] for k in self.tokens})

    def passes(self, layer: int) -> list[int]:
        return sorted(k[1] for k in self.tokens if k[0] == layer)

    def score(self, layer: int, by: str = "affinity", passes: Iterable[int] | None = None) -> torch.Tensor:
        """Per-expert score for `layer`, summed over `passes` and normalised per token."""
        src = {"affinity": self.gate_mass, "prob": self.prob_mass, "count": self.counts}
        require(by in src, f"by must be one of {sorted(src)}, got {by!r}")
        ps = list(passes) if passes is not None else self.passes(layer)
        keys = [(layer, r) for r in ps if (layer, r) in self.tokens]
        require(bool(keys), f"no statistics recorded for layer {layer} passes {ps}",
                recorded=sorted(self.tokens))
        tot = sum(src[by][k].double() for k in keys)
        ntok = sum(self.tokens[k] for k in keys)
        denom = ntok * (self.top_k if by == "count" else 1)
        return tot / max(denom, 1)

    @staticmethod
    def _pick(score: torch.Tensor, k: int | None, mass: float | None) -> list[int]:
        order = sorted(range(score.numel()), key=lambda i: (-float(score[i]), i))
        if mass is not None:
            share = score / score.sum().clamp_min(1e-30)
            out, cum = [], 0.0
            for i in order:
                out.append(i)
                cum += float(share[i])
                if cum >= mass - 1e-12:
                    break
            return sorted(out)
        assert k is not None
        return sorted(order[: max(0, min(k, score.numel()))])

    def select_hot(self, fraction: float | None = None, per_layer_k: int | dict[int, int] | None = None,
                   mass: float | None = None, by: str = "affinity", per_pass: str = "union",
                   layers: Iterable[int] | None = None) -> dict[int, list[int]]:
        """Hot experts per layer.

        Exactly one of: `fraction` (k = ceil(fraction*E) per layer), `per_layer_k` (int or
        {layer: k}), `mass` (ESFT: smallest set whose score share reaches `mass`, e.g. 0.1).
        `per_pass`: "union" - select per recorded pass and take the union; "pass1" - only the
        first execution (pass 0 for prelude/coda, pass 1 for core layers); "sum" - aggregate all
        passes, then select.
        """
        require(sum(x is not None for x in (fraction, per_layer_k, mass)) == 1,
                "give exactly one of fraction / per_layer_k / mass",
                fraction=fraction, per_layer_k=per_layer_k, mass=mass)
        require(per_pass in ("union", "pass1", "sum"), f"per_pass must be union|pass1|sum, got {per_pass!r}")
        out: dict[int, list[int]] = {}
        for layer in (list(layers) if layers is not None else self.layers()):
            if fraction is not None:
                k: int | None = math.ceil(fraction * self.num_experts)
            elif per_layer_k is not None:
                k = per_layer_k if isinstance(per_layer_k, int) else per_layer_k.get(layer, 0)
            else:
                k = None
            ps = self.passes(layer)
            if per_pass == "pass1":
                groups = [[min(ps)]]
            elif per_pass == "sum":
                groups = [ps]
            else:
                groups = [[r] for r in ps]
            sel: set[int] = set()
            for g in groups:
                sel.update(self._pick(self.score(layer, by, g), k, mass))
            if sel:
                out[layer] = sorted(sel)
        return out

    def summary(self, top_fraction: float = 0.1, by: str = "affinity") -> dict[int, dict[str, float]]:
        """Per layer (all passes summed): normalised entropy, share covered by the top x% experts,
        never-selected experts, tokens, and pass1/pass2 top-k overlap where available."""
        out: dict[int, dict[str, float]] = {}
        k = max(1, math.ceil(top_fraction * self.num_experts))
        for layer in self.layers():
            s = self.score(layer, by)
            p = s / s.sum().clamp_min(1e-30)
            ent = float(-(p * p.clamp_min(1e-30).log()).sum() / math.log(self.num_experts))
            cov = float(p.sort(descending=True).values[:k].sum())
            cnt = sum(self.counts[(layer, r)] for r in self.passes(layer))
            row = {"entropy_norm": ent, f"top{top_fraction:g}_coverage": cov,
                   "dead_experts": float((cnt == 0).sum()),
                   "tokens": float(sum(self.tokens[(layer, r)] for r in self.passes(layer)))}
            ps = self.passes(layer)
            if 1 in ps and 2 in ps:
                a = set(self._pick(self.score(layer, by, [1]), k, None))
                b = set(self._pick(self.score(layer, by, [2]), k, None))
                row["pass12_hot_jaccard"] = len(a & b) / max(len(a | b), 1)
            out[layer] = row
        return out

    def log_summary(self, top_fraction: float = 0.1, by: str = "affinity") -> dict[int, dict[str, float]]:
        summ = self.summary(top_fraction, by)
        if not summ:
            log.warning("expert stats: nothing recorded (did the forward reach the routers?)")
            return summ
        ents = [r["entropy_norm"] for r in summ.values()]
        covs = [r[f"top{top_fraction:g}_coverage"] for r in summ.values()]
        log.info("expert stats (%s): %d layers, E=%d top-%d | entropy_norm mean %.3f [%.3f..%.3f] | "
                 "top-%g%% experts cover %.1f%% of mass (mean)", by, len(summ), self.num_experts, self.top_k,
                 sum(ents) / len(ents), min(ents), max(ents), 100 * top_fraction, 100 * sum(covs) / len(covs))
        for layer, row in summ.items():
            log.debug("  layer %2d: %s", layer, " ".join(f"{k}={v:.3f}" for k, v in row.items()))
        return summ

    # ---------------------------------------------------------------- (de)serialisation
    def to_json(self, path: str | None = None) -> str:
        data = {
            "num_experts": self.num_experts, "top_k": self.top_k,
            "records": {_key(*k): {"tokens": self.tokens[k],
                                   "gate_mass": self.gate_mass[k].tolist(),
                                   "prob_mass": self.prob_mass[k].tolist(),
                                   "counts": self.counts[k].tolist()} for k in sorted(self.tokens)},
        }
        text = json.dumps(data)
        if path is not None:
            with open(path, "w", encoding="utf-8") as f:
                f.write(text)
        return text

    @classmethod
    def from_json(cls, src: str) -> ExpertStats:
        """`src` is a JSON string or a path to a JSON file."""
        text = src
        if not src.lstrip().startswith("{"):
            with open(src, encoding="utf-8") as f:
                text = f.read()
        d = json.loads(text)
        st = cls(int(d["num_experts"]), int(d["top_k"]))
        for ks, rec in d["records"].items():
            k = _unkey(ks)
            st.tokens[k] = int(rec["tokens"])
            st.gate_mass[k] = torch.tensor(rec["gate_mass"], dtype=torch.float64)
            st.prob_mass[k] = torch.tensor(rec["prob_mass"], dtype=torch.float64)
            st.counts[k] = torch.tensor(rec["counts"], dtype=torch.long)
        return st


def _run_batch(model: nn.Module, batch: Any) -> None:
    if isinstance(batch, dict):
        model(**batch)
    elif isinstance(batch, (tuple, list)):
        model(*batch)
    else:
        model(batch)


@torch.no_grad()
def scan_router_affinity(model: nn.Module, batches: Iterable[Any], num_experts: int | None = None,
                         top_k: int | None = None, controller: Any = None, max_batches: int | None = None,
                         norm_topk_prob: bool | None = None) -> ExpertStats:
    """Run `batches` (input_ids tensors, kwargs dicts or arg tuples) and collect router statistics.

    For loop-wrapped models the controller is auto-detected from the virtual layers (or pass
    `controller=`); records are keyed (physical layer, pass) with pass 0 for prelude/coda.
    """
    layers, cfg, ctl_auto = physical_layers(model)
    ctl = controller if controller is not None else ctl_auto
    e, k, norm = _moe_dims(cfg, num_experts, top_k, norm_topk_prob)
    stats = ExpertStats(e, k)
    is_probs: dict[int, bool] = {}
    handles = []

    def make_hook(phys: int):
        def hook(mod: nn.Module, inp: Any, out: Any) -> None:
            cur = getattr(ctl, "_cur", None) if ctl is not None else None
            r = int(cur[1]) if cur is not None else 0
            if phys not in is_probs:
                is_probs[phys] = _looks_like_probs(_first_full(out, e))
            stats.add(phys, r, parse_router_output(out, e, k, norm, is_probs[phys]))
        return hook

    for phys, layer in layers:
        handles.append(_moe_of(layer, phys).gate.register_forward_hook(make_hook(phys)))
    n = 0
    try:
        for batch in batches:
            if max_batches is not None and n >= max_batches:
                break
            _run_batch(model, batch)
            n += 1
    finally:
        for h in handles:
            h.remove()
    log.info("router affinity scan: %d batches, %d (layer, pass) records, E=%d top-%d%s", n, len(stats.tokens),
             e, k, " (loop-wrapped)" if ctl is not None else "")
    return stats


def _first_full(out: Any, e: int) -> torch.Tensor | None:
    outs = out if isinstance(out, (tuple, list)) else (out,)
    return next((o for o in outs if isinstance(o, torch.Tensor) and o.is_floating_point() and o.shape[-1] == e),
                None)


# ============================================================================ hot-expert deltas

_FLOAT = (torch.float16, torch.bfloat16, torch.float32, torch.float64)


def _is_plain_float(w: Any) -> bool:
    return type(w) in (torch.Tensor, nn.Parameter) and w.dtype in _FLOAT


def _experts_are_plain_float(experts: nn.Module) -> bool:
    gu = getattr(experts, "gate_up_proj", None)
    dn = getattr(experts, "down_proj", None)
    return bool(gu is not None and dn is not None and _is_plain_float(gu) and _is_plain_float(dn))


def _first_module_device(mod: nn.Module) -> torch.device:
    for p in mod.parameters():
        return p.device
    for b in mod.buffers():
        return b.device
    return torch.device("cpu")


class _LayerDeltas(nn.Module):
    def __init__(self, experts: list[int], i2: int, h: int, i: int, rank: int, dtype: torch.dtype,
                 device: torch.device):
        super().__init__()
        kw = {"dtype": dtype, "device": device}
        if rank <= 0:
            self.gate_up = nn.ParameterDict({str(e): nn.Parameter(torch.zeros(i2, h, **kw)) for e in experts})
            self.down = nn.ParameterDict({str(e): nn.Parameter(torch.zeros(h, i, **kw)) for e in experts})
        else:
            self.gate_up_A = nn.ParameterDict({str(e): nn.Parameter(torch.empty(rank, h, **kw)) for e in experts})
            self.gate_up_B = nn.ParameterDict({str(e): nn.Parameter(torch.zeros(i2, rank, **kw)) for e in experts})
            self.down_A = nn.ParameterDict({str(e): nn.Parameter(torch.empty(rank, i, **kw)) for e in experts})
            self.down_B = nn.ParameterDict({str(e): nn.Parameter(torch.zeros(h, rank, **kw)) for e in experts})
            for pd in (self.gate_up_A, self.down_A):
                for p in pd.values():
                    nn.init.kaiming_uniform_(p, a=math.sqrt(5))


@dataclass
class _LayerInfo:
    mlp: nn.Module
    experts: nn.Module
    transposed: bool      # base stored as [E, H, 2I] / [E, I, H] instead of [E, 2I, H] / [E, H, I]
    i2: int
    h: int
    i: int


class HotExpertDeltas(nn.Module):
    """Trainable zero-init deltas for hot experts; frozen fused base weights stay untouched.

        deltas = HotExpertDeltas(model, {layer: [expert, ...]}, rank=0)   # full-rank deltas
        deltas.attach()          # hooks; model output is bit-identical until deltas move
        ...train deltas.parameters()...
        deltas.merge_into_base_()   # export (float base only), deltas are reset to zero

    Parameter names: `L{layer}.gate_up.{e}` [2I, H] / `L{layer}.down.{e}` [H, I] (full rank) or
    `L{layer}.{gate_up,down}_{A,B}.{e}` (rank>0, effective delta = scale * B @ A, B zero-init,
    scale = alpha / rank).
    """

    def __init__(self, model: nn.Module, hot: dict[int, list[int]], rank: int = 0, alpha: float | None = None,
                 dtype: torch.dtype | None = None, dequant_fn: DequantFn | None = None,
                 num_experts: int | None = None, top_k: int | None = None, norm_topk_prob: bool | None = None):
        super().__init__()
        layers, cfg, _ = physical_layers(model)
        self.num_experts, self.top_k, self.norm_topk = _moe_dims(cfg, num_experts, top_k, norm_topk_prob)
        self.rank = int(rank)
        self.scale = float(alpha if alpha is not None else max(self.rank, 1)) / max(self.rank, 1)
        self.dequant_fn = dequant_fn
        hidden = getattr(cfg, "hidden_size", None)
        by_idx = dict(layers)
        infos: dict[int, _LayerInfo] = {}
        self.hot: dict[int, list[int]] = {}
        self.deltas = nn.ModuleDict()
        for layer, raw in sorted(hot.items()):
            experts = sorted({int(x) for x in raw})
            if not experts:
                continue
            require(layer in by_idx, f"hot layer {layer} not in model (physical layers 0..{len(layers) - 1})")
            require(all(0 <= x < self.num_experts for x in experts), f"layer {layer}: expert id out of range",
                    experts=experts, num_experts=self.num_experts)
            mlp = _moe_of(by_idx[layer], layer)
            info = self._layer_info(mlp, layer, hidden)
            infos[layer] = info
            ref = mlp.gate.weight if isinstance(getattr(mlp.gate, "weight", None), torch.Tensor) else None
            dev = ref.device if ref is not None else _first_module_device(mlp)
            gu = getattr(info.experts, "gate_up_proj", None)
            ddt = dtype or (gu.dtype if _is_plain_float(gu) else torch.float32)
            self.deltas[f"L{layer}"] = _LayerDeltas(experts, info.i2, info.h, info.i, self.rank, ddt, dev)
            self.hot[layer] = experts
        object.__setattr__(self, "_infos", infos)
        object.__setattr__(self, "_handles", [])
        object.__setattr__(self, "_pending", {})
        object.__setattr__(self, "_is_probs", {})
        n = sum(p.numel() for p in self.parameters())
        log.info("hot-expert deltas: %d layers, %d experts, rank=%s, %.2fM trainable params", len(self.hot),
                 sum(len(v) for v in self.hot.values()), self.rank or "full", n / 1e6)

    # ---------------------------------------------------------------- base weight access
    def _layer_info(self, mlp: nn.Module, layer: int, hidden: int | None) -> _LayerInfo:
        experts = mlp.experts
        if not _experts_are_plain_float(experts):
            if self.dequant_fn is None:
                self.dequant_fn = auto_dequant_expert_fn
            with torch.no_grad():
                g0, d0 = self.dequant_fn(experts, 0)
            require(
                g0.ndim == 2 and d0.ndim == 2 and g0.shape[1] == d0.shape[0] and g0.shape[0] == 2 * d0.shape[1],
                f"layer {layer}: dequant_fn returned incompatible shapes gate_up={tuple(g0.shape)}, "
                f"down={tuple(d0.shape)}",
            )
            return _LayerInfo(mlp, experts, False, int(g0.shape[0]), int(g0.shape[1]), int(d0.shape[1]))
        gu, dn = experts.gate_up_proj, experts.down_proj
        h = hidden or int(dn.shape[1])
        if gu.shape[-1] == h and dn.shape[-2] == h:
            return _LayerInfo(mlp, experts, False, int(gu.shape[-2]), h, int(dn.shape[-1]))
        require(gu.shape[-2] == h and dn.shape[-1] == h,
                f"layer {layer}: cannot match fused expert shapes to hidden={h}", gate_up_proj=gu, down_proj=dn)
        return _LayerInfo(mlp, experts, True, int(gu.shape[-1]), h, int(dn.shape[-2]))

    def base_weights(self, layer: int, e: int) -> tuple[torch.Tensor, torch.Tensor]:
        """(gate_up [2I, H], down [H, I]) of the frozen base expert (dequantised if needed)."""
        info = self._infos[layer]
        if not _experts_are_plain_float(info.experts):
            if self.dequant_fn is None:
                self.dequant_fn = auto_dequant_expert_fn
            return self.dequant_fn(info.experts, e)
        gu, dn = info.experts.gate_up_proj, info.experts.down_proj
        if info.transposed:
            return gu[e].t(), dn[e].t()
        return gu[e], dn[e]

    def delta_weights(self, layer: int, e: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Dense (dW_gate_up [2I, H], dW_down [H, I]) for expert e of `layer`."""
        ld = self.deltas[f"L{layer}"]
        k = str(e)
        if self.rank <= 0:
            return ld.gate_up[k], ld.down[k]
        return (self.scale * (ld.gate_up_B[k] @ ld.gate_up_A[k]),
                self.scale * (ld.down_B[k] @ ld.down_A[k]))

    def _apply_gu(self, ld: _LayerDeltas, k: str, x: torch.Tensor) -> torch.Tensor:
        if self.rank <= 0:
            return F.linear(x, ld.gate_up[k].to(x.dtype))
        return F.linear(F.linear(x, ld.gate_up_A[k].to(x.dtype)), ld.gate_up_B[k].to(x.dtype)) * self.scale

    def _apply_down(self, ld: _LayerDeltas, k: str, x: torch.Tensor) -> torch.Tensor:
        if self.rank <= 0:
            return F.linear(x, ld.down[k].to(x.dtype))
        return F.linear(F.linear(x, ld.down_A[k].to(x.dtype)), ld.down_B[k].to(x.dtype)) * self.scale

    # ---------------------------------------------------------------- hooks
    @property
    def attached(self) -> bool:
        return bool(self._handles)

    def attach(self) -> HotExpertDeltas:
        if self.attached:
            return self
        for layer, info in self._infos.items():
            self._handles.append(info.mlp.register_forward_pre_hook(self._pre_hook(layer)))
            self._handles.append(info.mlp.gate.register_forward_hook(self._gate_hook(layer)))
            self._handles.append(info.mlp.register_forward_hook(self._mlp_hook(layer), with_kwargs=True))
        log.debug("hot-expert deltas attached to %d MoE blocks", len(self._infos))
        return self

    def detach(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()
        self._pending.clear()

    def _pre_hook(self, layer: int):
        def hook(mod: nn.Module, args: Any) -> None:
            self._pending[layer] = None  # routing must come from *this* call
        return hook

    def _gate_hook(self, layer: int):
        def hook(mod: nn.Module, inp: Any, out: Any) -> None:
            if layer not in self._is_probs:
                self._is_probs[layer] = _looks_like_probs(_first_full(out, self.num_experts))
            self._pending[layer] = parse_router_output(out, self.num_experts, self.top_k, self.norm_topk,
                                                       self._is_probs[layer])
        return hook

    def _mlp_hook(self, layer: int):
        def hook(mod: nn.Module, args: Any, kwargs: Any, out: Any) -> Any:
            route = self._pending.pop(layer, None)
            require(route is not None, f"layer {layer}: MoE block ran but its router `mlp.gate` hook did not fire "
                    "(fused routing kernel bypassing the gate module?) - cannot apply expert deltas")
            x = args[0] if args else next(v for v in kwargs.values() if isinstance(v, torch.Tensor))
            main = out[0] if isinstance(out, tuple) else out
            flat = x.reshape(-1, x.shape[-1])
            require(route.top_idx.shape[0] == flat.shape[0], f"layer {layer}: router saw a different token count",
                    top_idx=route.top_idx, x=x)
            delta = self._delta(layer, flat, route)
            if delta is None:
                return None
            new = main + delta.view(main.shape).to(main.dtype)
            return (new, *out[1:]) if isinstance(out, tuple) else new
        return hook

    def _delta(self, layer: int, x: torch.Tensor, route: RouteInfo) -> torch.Tensor | None:
        info = self._infos[layer]
        ld = self.deltas[f"L{layer}"]
        act = getattr(info.experts, "act_fn", None) or F.silu
        acc = None
        for e in self.hot[layer]:
            tok, slot = (route.top_idx == e).nonzero(as_tuple=True)
            if tok.numel() == 0:
                continue
            k = str(e)
            xe = x[tok]
            gu_w, dn_w = self.base_weights(layer, e)
            gu_b = F.linear(xe, gu_w.to(xe.dtype))
            gu_m = gu_b + self._apply_gu(ld, k, xe)
            g_b, u_b = gu_b.chunk(2, dim=-1)
            g_m, u_m = gu_m.chunk(2, dim=-1)
            h_b = act(g_b) * u_b
            h_m = act(g_m) * u_m
            d = F.linear(h_m - h_b, dn_w.to(xe.dtype)) + self._apply_down(ld, k, h_m)
            contrib = d * route.top_w[tok, slot, None].to(d.dtype)
            if acc is None:
                acc = torch.zeros(x.shape, dtype=d.dtype, device=x.device)
            acc = acc.index_add(0, tok, contrib)
        return acc

    # ---------------------------------------------------------------- export / io
    @torch.no_grad()
    def merge_into_base_(self) -> int:
        """Add the deltas into the fused base tensors (float base only) and reset deltas to 0."""
        n = 0
        for layer, experts in self.hot.items():
            info = self._infos[layer]
            gu = getattr(info.experts, "gate_up_proj", None)
            dn = getattr(info.experts, "down_proj", None)
            require(_is_plain_float(gu) and _is_plain_float(dn),
                    f"layer {layer}: cannot merge into non-float (quantised) base weights; export the compact "
                    "delta state_dict and merge on a dequantised copy", gate_up_proj=gu)
            assert gu is not None and dn is not None
            for e in experts:
                dgu, ddn = self.delta_weights(layer, e)
                if info.transposed:
                    gu[e] += dgu.t().to(gu.dtype)
                    dn[e] += ddn.t().to(dn.dtype)
                else:
                    gu[e] += dgu.to(gu.dtype)
                    dn[e] += ddn.to(dn.dtype)
                n += 1
        self.reset_()
        log.info("merged %d hot-expert deltas into the fused base weights (deltas reset to zero)", n)
        return n

    @torch.no_grad()
    def reset_(self) -> None:
        for ld in self.deltas.values():
            for name, pd in ld.named_children():
                if self.rank <= 0 or name.endswith("_B"):
                    for p in pd.values():
                        p.zero_()

    def state_dict_compact(self) -> dict[str, Any]:
        """{"meta": {...}, "deltas": {(layer, expert): {name: tensor}}} (detached CPU clones)."""
        out: dict[tuple[int, int], dict[str, torch.Tensor]] = {}
        for layer, experts in self.hot.items():
            ld = self.deltas[f"L{layer}"]
            for e in experts:
                out[(layer, e)] = {name: pd[str(e)].detach().cpu().clone() for name, pd in ld.named_children()}
        meta = {"rank": self.rank, "scale": self.scale, "num_experts": self.num_experts, "top_k": self.top_k,
                "hot": {int(k): list(v) for k, v in self.hot.items()}}
        return {"meta": meta, "deltas": out}

    @torch.no_grad()
    def load_compact(self, sd: dict[str, Any], strict: bool = True) -> list[tuple[int, int]]:
        """Load a `state_dict_compact()`; returns the (layer, expert) keys that were loaded."""
        meta, deltas = sd.get("meta", {}), sd["deltas"]
        require(int(meta.get("rank", self.rank)) == self.rank, "compact delta rank mismatch",
                saved=meta.get("rank"), current=self.rank)
        mine = {(layer, e) for layer, es in self.hot.items() for e in es}
        theirs = {(int(a), int(b)) for a, b in deltas}
        if strict:
            require(mine == theirs, "compact delta keys differ (strict=True)",
                    missing=sorted(mine - theirs), unexpected=sorted(theirs - mine))
        loaded = []
        for (layer, e), tensors in deltas.items():
            key = (int(layer), int(e))
            if key not in mine:
                continue
            ld = self.deltas[f"L{key[0]}"]
            for name, t in tensors.items():
                p = getattr(ld, name)[str(key[1])]
                require(tuple(p.shape) == tuple(t.shape), f"shape mismatch for {key} {name}", saved=t, current=p)
                p.copy_(t.to(device=p.device, dtype=p.dtype))
            loaded.append(key)
        return sorted(loaded)

    def extra_repr(self) -> str:
        return f"layers={sorted(self.hot)}, experts={sum(len(v) for v in self.hot.values())}, rank={self.rank}"


def attach_hot_expert_deltas(model: nn.Module, hot: dict[int, list[int]], register: bool = True,
                             **kwargs: Any) -> HotExpertDeltas:
    """Build + attach deltas; with `register` they become `text_model.expert_deltas` (so they
    show up in `model.named_parameters()` / state_dict as `...expert_deltas.L{l}...`)."""
    deltas = HotExpertDeltas(model, hot, **kwargs).attach()
    if register:
        find_text_model(model).expert_deltas = deltas
    return deltas


install_hot_experts = attach_hot_expert_deltas


_EXPERT_DELTA_KEY_RE = re.compile(
    r"(?:^|\.)expert_deltas\.deltas\.L(\d+)\.(gate_up|down|gate_up_A|gate_up_B|down_A|down_B)\.(\d+)$"
)


@torch.no_grad()
def fold_expert_deltas_state_into_base_(
    model: nn.Module,
    params: dict[str, torch.Tensor],
    *,
    scale: float = 1.0,
    skip: set[tuple[int, int]] | None = None,
) -> set[tuple[int, int]]:
    """Fold unattached `expert_deltas` tensors from a prior stage directly into float base experts.

    Enables downstream stages (`S3`, narrower `S4`, `eval`, or `load_qloop_from_hub`) to preserve
    earlier hot-expert adaptations without keeping extra `HotExpertDeltas` hooks or duplicating the
    70 GB base checkpoint.
    """
    grouped: dict[tuple[int, int], dict[str, torch.Tensor]] = {}
    for k, t in params.items():
        m = _EXPERT_DELTA_KEY_RE.search(k)
        if m is None:
            continue
        pair = (int(m.group(1)), int(m.group(3)))
        if skip and pair in skip:
            continue
        grouped.setdefault(pair, {})[m.group(2)] = t
    if not grouped:
        return set()

    layers, cfg, _ = physical_layers(model)
    by_idx = dict(layers)
    hidden = getattr(cfg, "hidden_size", None)
    folded: set[tuple[int, int]] = set()
    for (layer, e), parts in sorted(grouped.items()):
        if layer not in by_idx:
            continue
        mlp = _moe_of(by_idx[layer], layer)
        gu = getattr(mlp.experts, "gate_up_proj", None)
        dn = getattr(mlp.experts, "down_proj", None)
        if not (_is_plain_float(gu) and _is_plain_float(dn)):
            log.warning("layer %d: skipping fold of expert %d into non-float base weights", layer, e)
            continue
        assert gu is not None and dn is not None
        if "gate_up" in parts and "down" in parts:
            dgu = parts["gate_up"].to(device=gu.device, dtype=torch.float32)
            ddn = parts["down"].to(device=dn.device, dtype=torch.float32)
        elif {"gate_up_A", "gate_up_B", "down_A", "down_B"} <= set(parts):
            dgu = float(scale) * (
                parts["gate_up_B"].to(device=gu.device, dtype=torch.float32)
                @ parts["gate_up_A"].to(device=gu.device, dtype=torch.float32)
            )
            ddn = float(scale) * (
                parts["down_B"].to(device=dn.device, dtype=torch.float32)
                @ parts["down_A"].to(device=dn.device, dtype=torch.float32)
            )
        else:
            continue
        h = hidden or int(dn.shape[1])
        transposed = not (gu.shape[-1] == h and dn.shape[-2] == h)
        if transposed:
            gu[e] += dgu.t().to(gu.dtype)
            dn[e] += ddn.t().to(dn.dtype)
        else:
            gu[e] += dgu.to(gu.dtype)
            dn[e] += ddn.to(dn.dtype)
        folded.add((layer, e))
    if folded:
        log.info("folded %d prior-stage hot-expert deltas into float base experts", len(folded))
    return folded


# ============================================================================ freezing

PARAM_GROUP_RULES: list[tuple[str, str]] = [
    ("expert_deltas", r"expert_deltas\."),
    ("gdn2_adapters", r"gdn2_adapter\."),
    ("hssd_adapters", r"hssd_adapter\."),
    ("loop_adapters", r"loop_adapters\."),
    ("lsa_indexers", r"(^|\.)(lsa|indexer)[^.]*\.|lsa_indexer"),
    ("router", r"mlp\.gate\.(weight|bias)$"),
    ("shared_expert", r"shared_expert"),
    ("expert_base", r"mlp\.experts\."),
    ("norms", r"(norm[^.]*\.(weight|bias)$)|layernorm"),
    ("deltanet", r"linear_attn\."),
    ("attention", r"self_attn\."),
    ("embeddings", r"embed_tokens|lm_head|wte|embed_positions"),
]


def param_group_of(name: str) -> str:
    for group, pat in PARAM_GROUP_RULES:
        if re.search(pat, name):
            return group
    return "other"


@dataclass
class FreezeReport:
    groups: dict[str, dict[str, int]]
    trainable_names: list[str]
    n_trainable: int
    n_total: int

    def __str__(self) -> str:
        rows = [f"{g:>14}: trainable {v['trainable']:>12,d} ({v['tensors']} tensors) / frozen {v['frozen']:>14,d}"
                for g, v in sorted(self.groups.items())]
        return (f"trainable {self.n_trainable:,d} / {self.n_total:,d} params "
                f"({100 * self.n_trainable / max(self.n_total, 1):.4f}%)\n" + "\n".join(rows))


def freeze_all_but(model: nn.Module, patterns: Iterable[str], extra_params: Iterable[Any] | None = None,
                   allowed_groups: Iterable[str] | None = None, forbid_groups: Iterable[str] = ("expert_base",),
                   strict_patterns: bool = True) -> FreezeReport:
    """Freeze every parameter except those whose name matches one of the `patterns` (re.search)
    plus `extra_params` (by identity; may live outside `model`). `extra_params` items are
    parameters or `(name, parameter)` pairs (e.g. `deltas.named_parameters(prefix="expert_deltas")`);
    unnamed extras outside the model are reported as group "extra".

    Safety: a pattern that matches nothing raises (typo guard, `strict_patterns`); a trainable
    parameter in `forbid_groups` raises (default: the fused expert base tensors - use
    HotExpertDeltas instead); if `allowed_groups` is given, any trainable group outside it raises.
    """
    pats = [re.compile(p) for p in patterns]
    extra: list[tuple[str | None, nn.Parameter]] = []
    for item in extra_params or []:
        if isinstance(item, tuple):
            extra.append((str(item[0]), item[1]))
        else:
            extra.append((None, item))
    extra_ids = {id(p) for _, p in extra}
    hits = dict.fromkeys(range(len(pats)), 0)
    groups: dict[str, dict[str, int]] = {}
    trainable: list[str] = []
    tgroup: dict[str, str] = {}
    seen: set[int] = set()
    n_tr = n_tot = 0

    def account(name: str, p: nn.Parameter, on: bool, group: str | None = None) -> None:
        nonlocal n_tr, n_tot
        grp = group or param_group_of(name)
        g = groups.setdefault(grp, {"trainable": 0, "frozen": 0, "tensors": 0})
        n_tot += p.numel()
        if on:
            g["trainable"] += p.numel()
            g["tensors"] += 1
            n_tr += p.numel()
            trainable.append(name)
            tgroup[name] = grp
        else:
            g["frozen"] += p.numel()

    for name, p in model.named_parameters():
        seen.add(id(p))
        on = id(p) in extra_ids
        for i, pat in enumerate(pats):
            if pat.search(name):
                hits[i] += 1
                on = True
        p.requires_grad_(on)
        account(name, p, on)
    for j, (name, p) in enumerate(extra):
        if id(p) not in seen:
            seen.add(id(p))
            p.requires_grad_(True)
            account(name or f"extra.{j}", p, True, None if name else "extra")
    if strict_patterns:
        dead = [pats[i].pattern for i, c in hits.items() if c == 0]
        require(not dead, f"freeze_all_but: patterns matched no parameter: {dead}")
    bad = sorted({g for g, v in groups.items() if v["trainable"] and g in set(forbid_groups)})
    require(not bad, f"freeze_all_but: forbidden groups would be trainable: {bad}",
            names=[n for n in trainable if tgroup[n] in bad][:10])
    if allowed_groups is not None:
        allowed = set(allowed_groups)
        unexpected = sorted(set(tgroup.values()) - allowed)
        require(not unexpected, f"freeze_all_but: unexpected trainable groups {unexpected} (allowed {sorted(allowed)})",
                names=[n for n in trainable if tgroup[n] in unexpected][:10])
    rep = FreezeReport(groups, trainable, n_tr, n_tot)
    log.info("freeze_all_but:\n%s", rep)
    return rep


# ============================================================================ frozen expert SHA-256 tripwire

_EXPERT_PATH_PARTS = frozenset({"experts", "shared_expert", "shared_experts"})


def _is_expert_tensor_path(name: str) -> bool:
    parts = name.split(".")
    if "expert_deltas" in parts:
        return False
    for seg in parts:
        if seg in _EXPERT_PATH_PARTS:
            return True
        if seg.startswith(("experts_", "shared_expert_")) and not seg.endswith("_gate"):
            return True
    return False


def _iter_canonical_expert_tensors(
    mod: nn.Module,
    prefix: str,
    visited_mods: set[tuple[str, int]],
    seen_names: set[str],
) -> list[tuple[str, torch.Tensor]]:
    visit_key = (prefix, id(mod))
    if visit_key in visited_mods:
        return []
    visited_mods.add(visit_key)

    out: list[tuple[str, torch.Tensor]] = []
    for pname, p in mod._parameters.items():
        if p is None or p.requires_grad:
            continue
        full = f"{prefix}.{pname}" if prefix else pname
        if _is_expert_tensor_path(full) and full not in seen_names:
            seen_names.add(full)
            out.append((full, p))

    for bname, buf in mod._buffers.items():
        if buf is None:
            continue
        full = f"{prefix}.{bname}" if prefix else bname
        if _is_expert_tensor_path(full) and full not in seen_names:
            seen_names.add(full)
            out.append((full, buf))

    for aname, aval in vars(mod).items():
        if aname.startswith("_") or isinstance(aval, nn.Parameter) or not isinstance(aval, torch.Tensor):
            continue
        full = f"{prefix}.{aname}" if prefix else aname
        if _is_expert_tensor_path(full) and full not in seen_names:
            seen_names.add(full)
            out.append((full, aval))

    for cname, child in mod._modules.items():
        if child is None:
            continue
        slot = getattr(child, "slot", None)
        inner = getattr(child, "inner", None)
        if slot is not None and isinstance(inner, nn.Module) and hasattr(slot, "phys"):
            c_prefix = f"{prefix}.{slot.phys}" if prefix else str(slot.phys)
            out.extend(_iter_canonical_expert_tensors(inner, c_prefix, visited_mods, seen_names))
        else:
            c_prefix = f"{prefix}.{cname}" if prefix else cname
            out.extend(_iter_canonical_expert_tensors(child, c_prefix, visited_mods, seen_names))
    return out


def sha256_frozen_experts(model: nn.Module) -> str:
    """Deterministic SHA-256 over frozen MoE expert parameters AND scale/packed buffers.

    Hashes raw bytes (`.detach().contiguous().reshape(-1).view(torch.uint8).cpu().numpy().tobytes()`)
    of all frozen parameters and registered buffers on `mlp.experts` / `shared_expert` (sorted by
    canonical physical path), working seamlessly across `bfloat16`, `float8_e4m3fn`, and `uint8`
    packed `NVFP4` tensors.
    """
    items = _iter_canonical_expert_tensors(model, "", set(), set())
    items.sort(key=lambda kv: kv[0])
    digest = hashlib.sha256()
    for name, t in items:
        header = f"{name}|{t.dtype}|{tuple(t.shape)}|".encode()
        digest.update(header)
        raw = t.detach().contiguous().reshape(-1).view(torch.uint8).cpu().numpy().tobytes()
        digest.update(raw)
    return digest.hexdigest()
