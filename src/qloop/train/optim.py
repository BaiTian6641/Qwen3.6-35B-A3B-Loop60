"""Per-parameter optimizer-in-backward with APOLLO / AdamW parameter groups.

Why in-backward: with a 35B MoE on one 96GB GPU, a full set of gradients for the trainable
parameters must never coexist with the activations of the backward pass. Each trainable
parameter gets its *own* optimizer instance; a `register_post_accumulate_grad_hook` steps it
the moment its gradient is complete and frees `param.grad` immediately.

Usage (identical loop for fused and non-fused modes):

    groups = build_param_groups(model.named_parameters(), cfg.optim)
    opt = FusedStepOptimizer(groups, schedule=WarmupCosine(total_steps=1000, warmup_steps=50))
    for i, mb in enumerate(micro_batches):
        last = (i + 1) % grad_accum == 0
        opt.set_boundary(last)          # non-boundary: grads accumulate in .grad as usual
        (loss(mb) / grad_accum).backward()
        if last:
            opt.step()                  # fused: bookkeeping only (params were stepped in backward)
            opt.zero_grad()

Semantics and caveats
* A leaf's AccumulateGrad node runs ONCE per backward even if the parameter is used several
  times in the forward (looped core layers, tied weights): the engine sums all contributions
  first. The hook therefore sees the complete gradient. If a hook fires twice within one
  boundary backward (reentrant checkpointing, several `.backward()` calls on the boundary
  micro-batch, or a forgotten `opt.step()`), we raise instead of silently double-stepping.
* When a single micro-step legitimately invokes `.backward()` multiple times (e.g. online
  per-layer teacher-forced HSSD warm-up or segmented KV-BPTT), wrap the micro-step in
  `with defer_micro_step(opt):` so hooks let `.grad` accumulate across all per-layer/segment
  backwards and advance `_micro` / trigger the fused update once on context exit.
* Zero per-parameter GPU->CPU host syncs in `_make_hook`: per-parameter gradient norms,
  clipping, non-finite guards, and squared-norm accumulation stay 100% on device during
  backward, syncing to host at most once inside `opt.step()`.
* Global-norm clipping is impossible in-backward (the global norm is known only after the
  last gradient). We offer per-parameter clipping (`max_grad_norm_per_param`) and, optionally,
  clipping by the *previous* step's exact global norm (`global_clip_from_prev`). In non-fused
  mode (`fused=False`), true global-norm clipping via `torch.nn.utils.clip_grad_norm_` and
  batched `torch.optim.AdamW(..., fused=torch.cuda.is_available())` are also supported.
* Non-finite gradients skip that parameter's update (counter + log naming the parameter).
* Parameters with attribute `_qloop_project = (lo, hi)` are clamped after each update.
* `state_dict()` is keyed by parameter NAME, so it survives re-wrapping / re-ordering.

APOLLO: `apollo_torch.APOLLOAdamW` is used when installed (impl="auto"/"apollo_torch");
otherwise `ApolloAdamWFallback`, a faithful re-implementation of APOLLO Algorithm 1
(arXiv 2412.05270) with the same update rule as the reference implementation.
"""
from __future__ import annotations

import contextlib
import copy
import logging
import math
import re
import zlib
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import Any, Callable

import torch
from torch import nn
from torch.optim import Optimizer

from qloop.debug import describe, require

log = logging.getLogger("qloop")

_LOWP = (torch.float16, torch.bfloat16)


def apollo_torch_available() -> bool:
    try:
        import apollo_torch  # noqa: F401
    except Exception:
        return False
    return True


# ============================================================================ built-in optimizers

def _state_dtype(p: torch.Tensor) -> torch.dtype:
    return torch.float64 if p.dtype == torch.float64 else torch.float32


def _work_tensor(p: torch.Tensor, st: dict[str, Any], master_weights: bool) -> torch.Tensor:
    """fp32 master copy for bf16/fp16 params (kept in optimizer state), else the param itself."""
    if master_weights and p.dtype in _LOWP:
        if "master" not in st:
            st["master"] = p.detach().float().clone()
        return st["master"]
    return p


class AdamWLite(Optimizer):
    """Plain-torch AdamW (decoupled weight decay) with optional fp32 master weights.

    For bf16/fp16 parameters with `master_weights=True` the optimizer keeps an fp32 master
    copy plus fp32 moments in `state` and writes the rounded result back into the parameter.
    """

    def __init__(self, params: Iterable[Any], lr: float = 1e-3, betas: tuple[float, float] = (0.9, 0.999),
                 eps: float = 1e-8, weight_decay: float = 0.0, master_weights: bool = True):
        require(lr >= 0 and eps >= 0 and 0 <= betas[0] < 1 and 0 <= betas[1] < 1, "invalid AdamW hyper-parameters",
                lr=lr, betas=betas, eps=eps)
        defaults = {"lr": lr, "betas": tuple(betas), "eps": eps, "weight_decay": weight_decay,
                    "master_weights": master_weights}
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(
        self,
        closure: Callable[[], Any] | None = None,
        *,
        finite_mask: torch.Tensor | None = None,
    ) -> Any:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            b1, b2 = group["betas"]
            lr, wd, eps = group["lr"], group["weight_decay"], group["eps"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                require(not p.grad.is_sparse, "AdamWLite does not support sparse gradients", param=p)
                st = self.state[p]
                sd = _state_dtype(p)
                w = _work_tensor(p, st, group["master_weights"])
                if "exp_avg" not in st:
                    st["step"] = 0
                    st["exp_avg"] = torch.zeros(p.shape, dtype=sd, device=p.device)
                    st["exp_avg_sq"] = torch.zeros(p.shape, dtype=sd, device=p.device)
                g = p.grad.to(sd)
                if finite_mask is not None:
                    fmask = finite_mask.to(device=p.device, dtype=torch.bool)
                    old_w = w.clone()
                    old_ea = st["exp_avg"].clone()
                    old_easq = st["exp_avg_sq"].clone()
                    g = torch.where(fmask, g, torch.zeros_like(g))
                st["step"] += 1
                t = st["step"]
                if wd:
                    w.mul_(1 - lr * wd)
                st["exp_avg"].mul_(b1).add_(g, alpha=1 - b1)
                st["exp_avg_sq"].mul_(b2).addcmul_(g, g, value=1 - b2)
                bc1, bc2 = 1 - b1 ** t, 1 - b2 ** t
                denom = (st["exp_avg_sq"].sqrt() / math.sqrt(bc2)).add_(eps)
                w.addcdiv_(st["exp_avg"].to(w.dtype), denom.to(w.dtype), value=-lr / bc1)
                if finite_mask is not None:
                    torch.where(fmask, w, old_w, out=w)
                    torch.where(fmask, st["exp_avg"], old_ea, out=st["exp_avg"])
                    torch.where(fmask, st["exp_avg_sq"], old_easq, out=st["exp_avg_sq"])
                    if w is not p:
                        p.copy_(torch.where(fmask, w.to(p.dtype), p))
                elif w is not p:
                    p.copy_(w)
        return loss


class ApolloAdamWFallback(Optimizer):
    """APOLLO / APOLLO-Mini (Zhu et al., "APOLLO: SGD-like Memory, AdamW-level Performance",
    arXiv 2412.05270, Algorithm 1), mirroring `apollo_torch.APOLLOAdamW`.

    Param groups with a `rank` key and 2D params use the low-rank path; everything else is
    plain AdamW (bias-corrected, weight decay applied after the update like the reference):

        P  ~ N(0, 1/r), refreshed every `update_proj_gap` steps (seeded, reproducible)
        R  = G P^T  (m >= n, "right")    or   R = P^T G  (m < n, "left")     [proj_type="std"]
        M, V <- Adam moments of R;   R~ = M / (sqrt(V) + eps)
        s  = ||R~[:, j]|| / ||R[:, j]||  per channel of the kept dimension  (scale_type="channel")
           = ||R~|| / ||R||                                                 (scale_type="tensor")
        G~ = G * s;  Fira norm-growth limiter: ||G~_t|| <= gamma * ||G~_{t-1}|| (gamma=1.01)
        W <- W - lr * sqrt(bc2)/bc1 * sqrt(scale) * G~ ;   W <- W - lr * wd * W

    Optimizer memory per (m x n) matrix: 2 * r * max(m, n) (+ the r x min(m,n) projection),
    i.e. SGD-like for APOLLO-Mini (rank=1, scale_type="tensor", scale=128).
    """

    def __init__(self, params: Iterable[Any], lr: float = 1e-3, betas: tuple[float, float] = (0.9, 0.999),
                 eps: float = 1e-6, weight_decay: float = 0.0, correct_bias: bool = True,
                 master_weights: bool = True, seed: int = 0, limiter_gamma: float = 1.01):
        defaults = {"lr": lr, "betas": tuple(betas), "eps": eps, "weight_decay": weight_decay,
                    "correct_bias": correct_bias, "master_weights": master_weights, "seed": int(seed),
                    "limiter_gamma": float(limiter_gamma)}
        super().__init__(params, defaults)

    @staticmethod
    def make_projection(shape: tuple[int, int], rank: int, proj_type: str, seed: int, device: torch.device,
                        dtype: torch.dtype) -> tuple[torch.Tensor, bool]:
        m, n = int(shape[0]), int(shape[1])
        if proj_type == "std":
            right = m >= n
        elif proj_type == "reverse_std":
            right = m < n
        elif proj_type in ("right", "left"):
            right = proj_type == "right"
        else:
            raise ValueError(f"unknown APOLLO proj_type {proj_type!r} (std|reverse_std|right|left)")
        r = max(1, min(int(rank), n if right else m))
        gen = torch.Generator(device="cpu").manual_seed(int(seed) % (2 ** 63 - 1))
        shape_p = (r, n) if right else (m, r)
        proj = torch.randn(shape_p, generator=gen, dtype=torch.float64) / math.sqrt(r)
        if device.type == "cuda":
            proj_cast = proj.to(dtype=dtype)
            try:
                proj_cast = proj_cast.pin_memory()
                return proj_cast.to(device=device, non_blocking=True), right
            except Exception:
                return proj_cast.to(device=device), right
        return proj.to(device=device, dtype=dtype), right

    @torch.no_grad()
    def step(
        self,
        closure: Callable[[], Any] | None = None,
        *,
        finite_mask: torch.Tensor | None = None,
    ) -> Any:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            b1, b2 = group["betas"]
            lr, wd, eps = group["lr"], group["weight_decay"], group["eps"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                st = self.state[p]
                sd = _state_dtype(p)
                w = _work_tensor(p, st, group["master_weights"])
                g = p.grad if p.grad.dtype == sd else p.grad.to(sd)
                old_sgn = (
                    st["scaling_grad_norm"].clone()
                    if (finite_mask is not None and "scaling_grad_norm" in st)
                    else None
                )
                if finite_mask is not None:
                    fmask = finite_mask.to(device=p.device, dtype=torch.bool)
                    old_w = w.clone()
                    g = torch.where(fmask, g, torch.zeros_like(g))
                st["step"] = st.get("step", 0) + 1
                t = st["step"]
                lowrank = bool(group.get("rank")) and p.dim() == 2
                if lowrank:
                    if group.get("proj", "random") != "random":
                        raise ValueError("ApolloAdamWFallback implements proj='random' only (as APOLLO does); "
                                         f"got {group.get('proj')!r} - install apollo-torch for SVD projections")
                    gap = int(group.get("update_proj_gap", 200))
                    if "proj" not in st or (t - 1) % gap == 0:
                        n_ref = st.get("proj_refresh", 0)
                        seed = group["seed"] * 1_000_003 + n_ref
                        st["proj"], st["proj_right"] = self.make_projection(
                            tuple(p.shape), int(group["rank"]), group.get("proj_type", "std"), seed, p.device, sd)
                        st["proj_refresh"] = n_ref + 1
                    proj = st["proj"]
                    r_grad = g @ proj.t() if st["proj_right"] else proj.t() @ g
                else:
                    r_grad = g
                if "exp_avg" not in st:
                    st["exp_avg"] = torch.zeros_like(r_grad)
                    st["exp_avg_sq"] = torch.zeros_like(r_grad)
                exp_avg, exp_avg_sq = st["exp_avg"], st["exp_avg_sq"]
                if finite_mask is not None:
                    old_ea = exp_avg.clone()
                    old_easq = exp_avg_sq.clone()
                exp_avg.mul_(b1).add_(r_grad, alpha=1 - b1)
                exp_avg_sq.mul_(b2).addcmul_(r_grad, r_grad, value=1 - b2)
                denom = exp_avg_sq.sqrt().add_(eps)
                step_size = lr
                if group["correct_bias"]:
                    step_size = lr * math.sqrt(1 - b2 ** t) / (1 - b1 ** t)
                norm_grad = exp_avg / denom
                if lowrank:
                    stype = group.get("scale_type", "tensor")
                    if stype == "channel":
                        norm_dim = 0 if norm_grad.shape[0] < norm_grad.shape[1] else 1
                        factor = torch.linalg.vector_norm(norm_grad, dim=norm_dim) / (
                            torch.linalg.vector_norm(r_grad, dim=norm_dim) + 1e-8)
                        if norm_dim == 1:
                            factor = factor.unsqueeze(1)
                    elif stype == "tensor":
                        factor = torch.linalg.vector_norm(norm_grad) / (torch.linalg.vector_norm(r_grad) + 1e-8)
                    else:
                        raise ValueError(f"unknown APOLLO scale_type {stype!r} (channel|tensor)")
                    scaling_grad = g * factor
                    sg_norm = torch.linalg.vector_norm(scaling_grad)
                    if "scaling_grad_norm" in st:  # Fira norm-growth limiter
                        gamma = group["limiter_gamma"]
                        limiter = torch.clamp(sg_norm / (st["scaling_grad_norm"] + 1e-8), min=gamma) / gamma
                        scaling_grad.div_(limiter)
                        st["scaling_grad_norm"] = sg_norm / limiter
                    else:
                        st["scaling_grad_norm"] = sg_norm
                    scale_mul = math.sqrt(float(group.get("scale", 1.0)))
                    if scale_mul != 1.0:
                        scaling_grad.mul_(scale_mul)
                    update = scaling_grad
                else:
                    update = norm_grad
                upd_w = update if update.dtype == w.dtype else update.to(w.dtype)
                w.add_(upd_w, alpha=-step_size)
                if wd > 0:
                    w.add_(w, alpha=-lr * wd)
                if finite_mask is not None:
                    torch.where(fmask, w, old_w, out=w)
                    torch.where(fmask, exp_avg, old_ea, out=exp_avg)
                    torch.where(fmask, exp_avg_sq, old_easq, out=exp_avg_sq)
                    if lowrank and old_sgn is not None:
                        st["scaling_grad_norm"] = torch.where(fmask, st["scaling_grad_norm"], old_sgn)
                    if w is not p:
                        p.copy_(torch.where(fmask, w.to(p.dtype), p))
                elif w is not p:
                    p.copy_(w)
        return loss


# ============================================================================ param groups

DEFAULT_OPTIM_CFG: dict[str, Any] = {
    "lr": 1e-4,
    "weight_decay": 0.0,
    "betas": [0.9, 0.999],
    "eps": 1e-8,
    "decay_1d": False,             # weight decay on 1D params (norm gains, gates, biases)
    "apollo": {
        "enabled": True,
        "rank": 1,                 # APOLLO-Mini; APOLLO: rank=256, scale_type=channel, scale=1
        "proj": "random",
        "scale_type": "tensor",
        "scale": 128.0,
        "update_proj_gap": 200,
        "proj_type": "std",
        "min_dim": 128,            # smaller matrices -> AdamW
        "eps": 1e-6,
    },
    # regexes (re.search on the parameter name) always routed to AdamW
    "adamw_patterns": [
        r"mlp\.gate\.weight$",
        r"shared_expert_gate",
        r"embed_tokens",
        r"lm_head",
        r"loop_adapters",
        r"hssd_adapter\.",
        r"lsa_indexer\.",
        r"gdn2_adapter\.",
    ],
    # regex -> lr multiplier; first match wins
    "lr_mult": {r"loop_adapters": 10.0, r"gdn2_adapter": 5.0, r"mlp\.gate\.weight$": 0.5},
}


def _plain(cfg: Any) -> dict[str, Any]:
    if cfg is None:
        return {}
    if hasattr(cfg, "to_dict"):
        return cfg.to_dict()
    return copy.deepcopy(dict(cfg))


def _merge(base: dict[str, Any], over: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict) and k != "lr_mult":
            out[k] = _merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def resolve_optim_cfg(cfg: Any = None) -> dict[str, Any]:
    raw = _plain(cfg)
    if isinstance(raw.get("apollo"), dict):
        ap_over = raw["apollo"]
        if "update_Gap" in ap_over and "update_proj_gap" not in ap_over:
            ap_over["update_proj_gap"] = int(ap_over["update_Gap"])
    out = _merge(DEFAULT_OPTIM_CFG, raw)
    if isinstance(out.get("apollo"), dict) and "update_Gap" in out["apollo"]:
        out["apollo"].setdefault("update_proj_gap", int(out["apollo"]["update_Gap"]))
    return out


def build_param_groups(named_params: Iterable[tuple[str, nn.Parameter]], cfg: Any = None) -> list[dict[str, Any]]:
    """Split trainable params into APOLLO (2D, min(shape) >= apollo.min_dim, not excluded) and
    AdamW (everything else) groups, further split by lr multiplier and weight decay.

    Each group dict carries `kind` ("apollo"|"adamw"), `params`, `names`, `lr` (= base lr x
    multiplier), `lr_mult`, `weight_decay`, `betas`, `eps` and, for APOLLO groups, the
    apollo_torch keys `rank, proj, scale_type, scale, update_proj_gap, proj_type`.
    """
    c = resolve_optim_cfg(cfg)
    ap = c["apollo"]
    excl = [re.compile(p) for p in c.get("adamw_patterns") or []]
    mults = [(re.compile(k), float(v)) for k, v in (c.get("lr_mult") or {}).items()]
    buckets: dict[tuple[str, float, float], dict[str, Any]] = {}
    seen: set[int] = set()
    for name, p in named_params:
        if not p.requires_grad or id(p) in seen:
            continue
        seen.add(id(p))
        is_apollo = (bool(ap.get("enabled", True)) and p.dim() == 2 and min(p.shape) >= int(ap.get("min_dim", 128))
                     and not any(r.search(name) for r in excl))
        kind = "apollo" if is_apollo else "adamw"
        mult = next((m for r, m in mults if r.search(name)), 1.0)
        wd = float(c["weight_decay"]) if (p.dim() >= 2 or c.get("decay_1d")) else 0.0
        key = (kind, mult, wd)
        if key not in buckets:
            g: dict[str, Any] = {"group": f"{kind}/lr_x{mult:g}/wd{wd:g}", "kind": kind, "params": [], "names": [],
                                 "lr": float(c["lr"]) * mult, "lr_mult": mult, "weight_decay": wd,
                                 "betas": tuple(float(b) for b in c["betas"]), "eps": float(c["eps"])}
            if is_apollo:
                g.update({"rank": int(ap["rank"]), "proj": str(ap["proj"]), "scale_type": str(ap["scale_type"]),
                          "scale": float(ap["scale"]), "update_proj_gap": int(ap["update_proj_gap"]),
                          "proj_type": str(ap["proj_type"]), "eps": float(ap.get("eps", c["eps"]))})
            buckets[key] = g
        buckets[key]["params"].append(p)
        buckets[key]["names"].append(name)
    groups = list(buckets.values())
    for g in groups:
        n = sum(p.numel() for p in g["params"])
        log.info("param group %-28s %4d tensors %12s params lr=%.3g", g["group"], len(g["params"]), f"{n:,d}", g["lr"])
    return groups


# ============================================================================ schedule

@dataclass
class WarmupCosine:
    """lr factor for the update with 0-based index `step`: linear warmup then cosine to min_ratio."""

    total_steps: int
    warmup_steps: int = 0
    min_ratio: float = 0.1

    def __call__(self, step: int) -> float:
        if self.warmup_steps > 0 and step < self.warmup_steps:
            return (step + 1) / self.warmup_steps
        span = max(1, self.total_steps - self.warmup_steps)
        prog = min(1.0, max(0.0, (step - self.warmup_steps) / span))
        return self.min_ratio + (1 - self.min_ratio) * 0.5 * (1 + math.cos(math.pi * prog))


# ============================================================================ fused optimizer

@dataclass
class _Entry:
    idx: int
    name: str
    param: nn.Parameter
    opt: Optimizer
    base_lr: float
    kind: str


def _copy_state(st: dict[str, Any]) -> dict[str, Any]:
    return {k: (v.detach().clone() if isinstance(v, torch.Tensor) else copy.deepcopy(v)) for k, v in st.items()}


def _move_state_value(val: Any, dev: torch.device, _seen: dict[int, Any] | None = None) -> Any:
    """Recursively copy & move tensors inside optimizer state values (e.g. APOLLO `GradientProjector`)."""
    if isinstance(val, torch.Tensor):
        return val.detach().clone().to(dev)
    if val is None or isinstance(val, (bool, int, float, str, bytes, torch.dtype, torch.device)):
        return val
    if _seen is None:
        _seen = {}
    oid = id(val)
    if oid in _seen:
        return _seen[oid]
    if isinstance(val, dict):
        out_d: dict[Any, Any] = {}
        _seen[oid] = out_d
        for k, v in val.items():
            out_d[k] = _move_state_value(v, dev, _seen)
        return out_d
    if isinstance(val, list):
        out_l: list[Any] = []
        _seen[oid] = out_l
        for v in val:
            out_l.append(_move_state_value(v, dev, _seen))
        return out_l
    if isinstance(val, tuple):
        out_t = tuple(_move_state_value(v, dev, _seen) for v in val)
        _seen[oid] = out_t
        return out_t
    copied = copy.deepcopy(val)
    _seen[oid] = copied
    if hasattr(copied, "__dict__"):
        for attr, attr_val in list(vars(copied).items()):
            moved = _move_state_value(attr_val, dev, _seen)
            if moved is not attr_val:
                try:
                    setattr(copied, attr, moved)
                except Exception:
                    pass
    slots = getattr(type(copied), "__slots__", ())
    if isinstance(slots, str):
        slots = (slots,)
    for slot_name in slots:
        if hasattr(copied, slot_name):
            attr_val = getattr(copied, slot_name)
            moved = _move_state_value(attr_val, dev, _seen)
            if moved is not attr_val:
                try:
                    setattr(copied, slot_name, moved)
                except Exception:
                    pass
    return copied


@contextlib.contextmanager
def defer_micro_step(opt: Any) -> Iterator[None]:
    """Defer per-parameter fused backward updates until the end of a multi-backward micro-step.

    For `FusedStepOptimizer`, delegates to `opt.defer_micro_step()`. For standard
    `torch.optim.Optimizer` instances (or `None`), acts as a zero-overhead no-op context manager.
    """
    defer_fn = getattr(opt, "defer_micro_step", None) if opt is not None else None
    if callable(defer_fn):
        with defer_fn():
            yield
    else:
        yield


class FusedStepOptimizer:
    """One optimizer per parameter, stepped from `register_post_accumulate_grad_hook`.

    Args:
        groups: output of `build_param_groups` (needs `names` per group).
        fused: True = optimizer-in-backward; False = classic `step()` (batched `torch.optim.AdamW`
            when all groups are AdamW/adapter groups with `impl="auto"` or `impl="torch_adamw"`,
            or per-parameter built-in debug equivalence with `impl="builtin"`).
        impl: "auto" (apollo_torch if installed) | "apollo_torch" | "builtin" | "torch_adamw".
        schedule: callable step -> lr factor (e.g. `WarmupCosine`), applied to every group lr.
        max_grad_norm_per_param: clip each parameter's gradient to this L2 norm on device.
        global_clip_from_prev: additionally scale grads by min(1, c / previous_global_norm).
        max_grad_norm: true global L2 norm clip via `torch.nn.utils.clip_grad_norm_` when `fused=False`.
        grad_accum: micro-batches per optimizer step for automatic `_micro` boundary tracking.
        master_weights: fp32 master copies for bf16/fp16 params (built-in optimizers).
        skip_nonfinite: skip a parameter's update when its gradient has NaN/Inf.
    """

    def __init__(
        self,
        groups: list[dict[str, Any]],
        *,
        fused: bool = True,
        impl: str = "auto",
        schedule: Callable[[int], float] | None = None,
        max_grad_norm_per_param: float | None = None,
        global_clip_from_prev: float | None = None,
        max_grad_norm: float | None = None,
        grad_accum: int = 1,
        master_weights: bool = True,
        skip_nonfinite: bool = True,
        seed: int = 0,
        norm_ema_beta: float = 0.9,
        log_every: int = 0,
    ):
        require(
            impl in ("auto", "apollo_torch", "builtin", "torch_adamw"),
            f"impl must be auto|apollo_torch|builtin|torch_adamw, got {impl!r}",
        )
        if impl == "apollo_torch" and not apollo_torch_available():
            raise ImportError("impl='apollo_torch' but `apollo_torch` is not importable: pip install apollo-torch")
        self.use_apollo_torch = impl == "apollo_torch" or (impl == "auto" and apollo_torch_available())
        self.fused, self.schedule = bool(fused), schedule
        self.impl = impl
        self.max_norm, self.global_clip = max_grad_norm_per_param, global_clip_from_prev
        self.max_grad_norm = max_grad_norm
        self.grad_accum = max(1, int(grad_accum))
        self.master_weights, self.skip_nonfinite = master_weights, skip_nonfinite
        self.seed, self.norm_ema_beta, self.log_every = int(seed), float(norm_ema_beta), int(log_every)

        all_adamw = all(g.get("kind", "adamw") == "adamw" for g in groups)
        self._torch_adamw: torch.optim.AdamW | None = None
        if impl == "torch_adamw" and not all_adamw:
            log.warning("impl='torch_adamw' requested with APOLLO groups; routing all groups to torch.optim.AdamW")

        batched_groups: list[dict[str, Any]] = []
        batched_param_ids: set[int] = set()
        if (not self.fused) and impl in ("auto", "torch_adamw") and len(groups) > 0:
            for g in groups:
                kind = g.get("kind", "adamw")
                if impl != "torch_adamw" and kind != "adamw":
                    continue
                b_params = [
                    p
                    for p in g["params"]
                    if impl == "torch_adamw" or not (self.master_weights and p.dtype in _LOWP)
                ]
                if b_params:
                    for p in b_params:
                        batched_param_ids.add(id(p))
                    batched_groups.append(
                        {
                            "params": b_params,
                            "lr": float(g["lr"]),
                            "betas": g["betas"],
                            "eps": float(g["eps"]),
                            "weight_decay": float(g["weight_decay"]),
                        }
                    )
            if batched_groups:
                self._torch_adamw = self._build_batched_torch_adamw(batched_groups)

        self.entries: dict[str, _Entry] = {}
        ids: set[int] = set()
        idx_counter = 0
        for g in groups:
            names = g.get("names")
            require(
                names is not None and len(names) == len(g["params"]),
                "each group needs `names` aligned with `params` (use build_param_groups)",
                group=g.get("group"),
            )
            for name, p in zip(names, g["params"]):
                require(name not in self.entries, f"duplicate parameter name {name!r}")
                require(id(p) not in ids, f"parameter {name!r} appears in more than one group")
                require(p.requires_grad, f"parameter {name!r} does not require grad", param=p)
                ids.add(id(p))
                if self._torch_adamw is not None and id(p) in batched_param_ids:
                    self.entries[name] = _Entry(
                        idx_counter, name, p, self._torch_adamw, float(g["lr"]), "adamw"
                    )
                else:
                    self.entries[name] = _Entry(
                        idx_counter, name, p, self._make_opt(name, p, g), float(g["lr"]), g.get("kind", "adamw")
                    )
                idx_counter += 1

        self.step_count = 0
        self.skipped: dict[str, int] = {}
        self.norm_ema: float | None = None
        self.prev_global_norm: float | None = None
        self.last_global_norm: float | None = None
        self._boundary = True
        self._boundary_explicit = False
        self._atomic_step_aborted = False
        self._defer_depth = 0
        self._micro: list[int] = [0] * len(self.entries)
        self._stepped: set[str] = set()
        self._param_finite: dict[str, torch.Tensor] = {}
        self._step_gnorm_sq: torch.Tensor | None = None
        self._clipped_dev: torch.Tensor | None = None
        self._has_grad_samples = False
        self._clipped = 0
        self._skipped_now = 0
        self._warned_empty = False
        self._hooks: list[Any] = []
        if self.fused:
            for ent in self.entries.values():
                self._hooks.append(ent.param.register_post_accumulate_grad_hook(self._make_hook(ent)))
        kinds: dict[str, int] = {}
        for ent in self.entries.values():
            k = f"{ent.kind}:{type(ent.opt).__name__}"
            kinds[k] = kinds.get(k, 0) + 1
        log.info(
            "FusedStepOptimizer: %d params (%s), fused=%s, grad_accum=%d, per-param clip=%s, "
            "global_clip_from_prev=%s, max_grad_norm=%s",
            len(self.entries),
            ", ".join(f"{k} x{v}" for k, v in sorted(kinds.items())),
            self.fused,
            self.grad_accum,
            self.max_norm,
            self.global_clip,
            self.max_grad_norm,
        )

    @classmethod
    def from_config(
        cls,
        named_params: Iterable[tuple[str, nn.Parameter]],
        cfg: Any = None,
        *,
        grad_accum: int | None = None,
    ) -> FusedStepOptimizer:
        """cfg = optim section: DEFAULT_OPTIM_CFG keys + fused, impl, max_grad_norm_per_param,
        global_clip_from_prev, max_grad_norm, grad_accum, master_weights, total_steps, warmup_steps,
        min_lr_ratio, seed."""
        c = resolve_optim_cfg(cfg)
        sched = None
        if c.get("total_steps"):
            sched = WarmupCosine(
                int(c["total_steps"]),
                int(c.get("warmup_steps", 0)),
                float(c.get("min_lr_ratio", 0.1)),
            )
        eff_accum = int(grad_accum if grad_accum is not None else c.get("grad_accum", 1))
        return cls(
            build_param_groups(named_params, c),
            fused=bool(c.get("fused", True)),
            impl=str(c.get("impl", "auto")),
            schedule=sched,
            max_grad_norm_per_param=c.get("max_grad_norm_per_param"),
            global_clip_from_prev=c.get("global_clip_from_prev"),
            max_grad_norm=c.get("max_grad_norm"),
            grad_accum=eff_accum,
            master_weights=bool(c.get("master_weights", True)),
            seed=int(c.get("seed", 0)),
        )

    # ---------------------------------------------------------------- construction helpers
    def _build_batched_torch_adamw(self, groups: list[dict[str, Any]]) -> torch.optim.AdamW:
        pgs: list[dict[str, Any]] = []
        all_cuda_fp = torch.cuda.is_available()
        for g in groups:
            params = list(g["params"])
            if not params:
                continue
            for p in params:
                if not p.is_cuda or p.dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
                    all_cuda_fp = False
            pgs.append(
                {
                    "params": params,
                    "lr": float(g["lr"]),
                    "_qloop_base_lr": float(g["lr"]),
                    "betas": tuple(float(b) for b in g["betas"]),
                    "eps": float(g["eps"]),
                    "weight_decay": float(g["weight_decay"]),
                }
            )
        if all_cuda_fp and pgs:
            try:
                return torch.optim.AdamW(pgs, fused=True)
            except Exception:
                pass
        return torch.optim.AdamW(pgs, fused=False)

    def _make_opt(self, name: str, p: nn.Parameter, g: dict[str, Any]) -> Optimizer:
        lr, betas, eps, wd = float(g["lr"]), tuple(g["betas"]), float(g["eps"]), float(g["weight_decay"])
        if g.get("kind") == "apollo":
            keys = {k: g[k] for k in ("rank", "proj", "scale_type", "scale", "update_proj_gap", "proj_type")}
            if self.use_apollo_torch:
                from apollo_torch import APOLLOAdamW

                return APOLLOAdamW([{"params": [p], **keys}], lr=lr, betas=betas, eps=eps, weight_decay=wd)
            pseed = (self.seed * 7919 + zlib.crc32(name.encode("utf-8"))) % (2 ** 31)
            return ApolloAdamWFallback(
                [{"params": [p], **keys}],
                lr=lr,
                betas=betas,
                eps=eps,
                weight_decay=wd,
                master_weights=self.master_weights,
                seed=pseed,
            )
        return AdamWLite([p], lr=lr, betas=betas, eps=eps, weight_decay=wd, master_weights=self.master_weights)

    def _should_step_micro(self, idx: int) -> bool:
        if self._boundary_explicit:
            return self._boundary
        return (self._micro[idx] % self.grad_accum) == 0

    def _make_hook(self, ent: _Entry) -> Callable[[torch.Tensor], None]:
        def hook(p: torch.Tensor) -> None:
            if self._defer_depth > 0:
                return  # deferred: keep summing per-layer/segment grads into .grad
            self._micro[ent.idx] += 1
            if not self._should_step_micro(ent.idx):
                return  # gradient accumulation: keep summing into .grad
            if ent.name in self._stepped:
                raise RuntimeError(
                    f"FusedStepOptimizer: parameter '{ent.name}' received a second gradient within one boundary "
                    "step. Causes: reentrant activation checkpointing (use use_reentrant=False), several "
                    ".backward() calls on the boundary micro-batch (wrap in `with opt.defer_micro_step():` or "
                    "sum the losses first), or a missing opt.step() after the previous boundary backward."
                )
            self._step_param(ent)
            p.grad = None

        return hook

    @contextlib.contextmanager
    def defer_micro_step(self) -> Iterator[None]:
        """Defer hook stepping across multiple `.backward()` calls within one micro-batch.

        While active, `_make_hook` lets PyTorch accumulate `.grad` across per-layer or
        per-segment `.backward()` invocations without advancing `_micro[i]`. On exit of the
        outermost `defer_micro_step()`, advances `_micro[i]` by `1` for every parameter with a
        non-None `.grad` and triggers the fused parameter update if at the accumulation boundary.
        """
        self._defer_depth += 1
        exc_raised = False
        try:
            yield
        except BaseException:
            exc_raised = True
            raise
        finally:
            self._defer_depth -= 1
            if self._defer_depth == 0 and not exc_raised:
                for ent in self.entries.values():
                    if ent.param.grad is None:
                        continue
                    self._micro[ent.idx] += 1
                    if self.fused and self._should_step_micro(ent.idx):
                        if ent.name in self._stepped:
                            raise RuntimeError(
                                f"FusedStepOptimizer: parameter '{ent.name}' received a second gradient within "
                                "one boundary step across `defer_micro_step()` blocks without `opt.step()`."
                            )
                        self._step_param(ent)
                        ent.param.grad = None

    # ---------------------------------------------------------------- core update
    @property
    def lr_factor(self) -> float:
        return float(self.schedule(self.step_count)) if self.schedule is not None else 1.0

    def current_lr(self, name: str) -> float:
        return self.entries[name].base_lr * self.lr_factor

    def _accumulate_device_norm(self, ent_name: str, gnorm: torch.Tensor, finite: torch.Tensor) -> None:
        self._param_finite[ent_name] = finite
        g64 = gnorm.to(dtype=torch.float64)
        g2 = torch.where(finite, g64 * g64, torch.zeros((), device=gnorm.device, dtype=torch.float64))
        if self._step_gnorm_sq is None:
            self._step_gnorm_sq = g2
        else:
            self._step_gnorm_sq = self._step_gnorm_sq + g2.to(device=self._step_gnorm_sq.device)

    def _accumulate_device_clip(self, clipped_flag: torch.Tensor) -> None:
        c64 = clipped_flag.to(dtype=torch.float64)
        if self._clipped_dev is None:
            self._clipped_dev = c64
        else:
            self._clipped_dev = self._clipped_dev + c64.to(device=self._clipped_dev.device)

    def _clamp_projected_param(self, p: nn.Parameter, opt: Optimizer) -> None:
        bounds = getattr(p, "_qloop_project", None)
        if bounds is not None:
            lo, hi = bounds
            with torch.no_grad():
                p.clamp_(lo, hi)
                master = opt.state.get(p, {}).get("master")
                if isinstance(master, torch.Tensor):
                    master.clamp_(lo, hi)

    def _step_param(self, ent: _Entry) -> bool:
        """On-device norm, clipping, non-finite guard, and parameter update (ZERO host syncs!)."""
        p = ent.param
        grad = p.grad
        if grad is None:
            return False
        self._stepped.add(ent.name)
        acc = torch.float64 if grad.dtype == torch.float64 else torch.float32
        with torch.no_grad():
            gnorm = torch.linalg.vector_norm(grad.detach(), dtype=acc)
            finite = torch.isfinite(gnorm)
            self._accumulate_device_norm(ent.name, gnorm, finite)

            scale: torch.Tensor | None = None
            if self.max_norm is not None:
                scale = torch.clamp(float(self.max_norm) / (gnorm + 1e-12), max=1.0)
                self._accumulate_device_clip(finite & (gnorm > float(self.max_norm)))
            if self.global_clip is not None and self.prev_global_norm and self.prev_global_norm > self.global_clip:
                gcoef = float(self.global_clip) / float(self.prev_global_norm)
                scale = (scale * gcoef) if scale is not None else torch.full_like(gnorm, gcoef)
            if scale is not None:
                safe_scale = torch.where(finite, scale, torch.ones_like(scale)).to(dtype=grad.dtype)
                grad.mul_(safe_scale)

        lr = ent.base_lr * self.lr_factor
        for g in ent.opt.param_groups:
            g["lr"] = lr

        if self.skip_nonfinite:
            if isinstance(ent.opt, (AdamWLite, ApolloAdamWFallback)):
                ent.opt.step(finite_mask=finite)
            else:
                with torch.no_grad():
                    fmask = finite.to(device=p.device, dtype=torch.bool)
                    p_before = p.detach().clone()
                    st_before = _copy_state(ent.opt.state.get(p, {}))
                    p.grad = torch.where(fmask, grad, torch.zeros_like(grad))
                ent.opt.step()
                with torch.no_grad():
                    p.copy_(torch.where(fmask, p, p_before))
                    cur_st = ent.opt.state.get(p)
                    if isinstance(cur_st, dict) and st_before:
                        for k, v_old in st_before.items():
                            v_cur = cur_st.get(k)
                            if isinstance(v_cur, torch.Tensor) and isinstance(v_old, torch.Tensor):
                                cur_st[k] = torch.where(fmask, v_cur, v_old.to(device=v_cur.device))
        else:
            ent.opt.step()

        self._clamp_projected_param(p, ent.opt)
        return True

    @staticmethod
    def _batched_grad_norms(
        active: list[_Entry],
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], torch.Tensor]:
        """Compute per-param L2 norms and total finite squared norm in fp64 via fused `_foreach_norm`."""
        gnorms: dict[str, torch.Tensor] = {}
        finites: dict[str, torch.Tensor] = {}
        buckets: dict[tuple[torch.device, torch.dtype], list[_Entry]] = {}
        for ent in active:
            grad = ent.param.grad
            assert grad is not None
            buckets.setdefault((grad.device, grad.dtype), []).append(ent)

        ref_dev = active[0].param.device
        step_gnorm_sq: torch.Tensor | None = None
        has_foreach = hasattr(torch, "_foreach_norm")
        for (_dev, dt), ents in buckets.items():
            acc = torch.float64 if dt == torch.float64 else torch.float32
            grads = [e.param.grad.detach() for e in ents if e.param.grad is not None]
            norms_list: list[torch.Tensor] | None = None
            if has_foreach and len(grads) > 1 and not any(g.is_sparse for g in grads):
                try:
                    norms_list = list(torch._foreach_norm(grads, 2.0, dtype=acc))
                except (TypeError, RuntimeError):
                    if dt == acc:
                        try:
                            norms_list = list(torch._foreach_norm(grads, 2.0))
                        except (TypeError, RuntimeError):
                            norms_list = None
            if norms_list is None:
                norms_list = [torch.linalg.vector_norm(g, dtype=acc) for g in grads]

            norm_vec = torch.stack(norms_list)
            fin_vec = torch.isfinite(norm_vec)
            g64 = norm_vec.to(dtype=torch.float64)
            bucket_sq = torch.where(fin_vec, g64 * g64, torch.zeros_like(g64)).sum()
            if step_gnorm_sq is None:
                step_gnorm_sq = bucket_sq.to(device=ref_dev)
            else:
                step_gnorm_sq = step_gnorm_sq + bucket_sq.to(device=ref_dev)
            for e, gn, fn in zip(ents, norms_list, fin_vec.unbind(0)):
                gnorms[e.name] = gn
                finites[e.name] = fn

        assert step_gnorm_sq is not None
        return gnorms, finites, step_gnorm_sq

    @staticmethod
    def _step_batched_adamw_lite(entries: list[_Entry], lrf: float) -> None:
        """Execute multi-tensor `_foreach_*` AdamW updates across `AdamWLite` params (bit-exact `state_dict`)."""
        buckets: dict[tuple[Any, ...], list[tuple[_Entry, torch.Tensor, dict[str, Any]]]] = {}
        fallback: list[_Entry] = []
        has_foreach = all(
            hasattr(torch, fn)
            for fn in (
                "_foreach_mul_",
                "_foreach_add_",
                "_foreach_addcmul_",
                "_foreach_sqrt",
                "_foreach_div_",
                "_foreach_addcdiv_",
                "_foreach_copy_",
            )
        )
        for ent in entries:
            p = ent.param
            grad = p.grad
            if grad is None:
                continue
            if not has_foreach or grad.is_sparse or len(ent.opt.param_groups) != 1:
                fallback.append(ent)
                continue
            pg = ent.opt.param_groups[0]
            lr = ent.base_lr * lrf
            pg["lr"] = lr
            st = ent.opt.state[p]
            sd = _state_dtype(p)
            w = _work_tensor(p, st, pg["master_weights"])
            if "exp_avg" not in st:
                st["step"] = 0
                st["exp_avg"] = torch.zeros(p.shape, dtype=sd, device=p.device)
                st["exp_avg_sq"] = torch.zeros(p.shape, dtype=sd, device=p.device)
            b1, b2 = pg["betas"]
            key = (
                p.device,
                p.dtype,
                sd,
                w.dtype,
                bool(pg["master_weights"]),
                float(lr),
                float(b1),
                float(b2),
                float(pg["eps"]),
                float(pg["weight_decay"]),
                int(st["step"]),
            )
            buckets.setdefault(key, []).append((ent, w, st))

        with torch.no_grad():
            for key, items in buckets.items():
                if len(items) == 1:
                    ent, _w, _st = items[0]
                    ent.opt.step()
                    continue
                _dev, _p_dt, sd, w_dt, _mw, lr, b1, b2, eps, wd, prev_step = key
                t = prev_step + 1
                ps = [ent.param for ent, _w, _st in items]
                ws = [w for _ent, w, _st in items]
                gs = [
                    ent.param.grad if ent.param.grad.dtype == sd else ent.param.grad.to(sd)  # type: ignore[union-attr]
                    for ent, _w, _st in items
                ]
                eas = [st["exp_avg"] for _ent, _w, st in items]
                easqs = [st["exp_avg_sq"] for _ent, _w, st in items]
                for _ent, _w, st in items:
                    st["step"] = t
                if wd:
                    torch._foreach_mul_(ws, 1.0 - lr * wd)
                torch._foreach_mul_(eas, b1)
                torch._foreach_add_(eas, gs, alpha=1.0 - b1)
                torch._foreach_mul_(easqs, b2)
                torch._foreach_addcmul_(easqs, gs, gs, value=1.0 - b2)
                bc1, bc2 = 1.0 - b1 ** t, 1.0 - b2 ** t
                denoms = list(torch._foreach_sqrt(easqs))
                torch._foreach_div_(denoms, math.sqrt(bc2))
                torch._foreach_add_(denoms, eps)
                ea_w = eas if sd == w_dt else [e.to(w_dt) for e in eas]
                den_w = denoms if sd == w_dt else [d.to(w_dt) for d in denoms]
                torch._foreach_addcdiv_(ws, ea_w, den_w, value=-lr / bc1)
                if ws[0] is not ps[0]:
                    torch._foreach_copy_(ps, ws)

        for ent in fallback:
            lr = ent.base_lr * lrf
            for g in ent.opt.param_groups:
                g["lr"] = lr
            ent.opt.step()

    def _step_nonfused(self) -> None:
        """Classic post-backward update with atomic whole-step non-finite guard, true global
        `clip_grad_norm_(max_grad_norm)`, and fused CUDA `torch.optim.AdamW` + batched master-weight / APOLLO."""
        active = [ent for ent in self.entries.values() if ent.param.grad is not None]
        if not active:
            return

        with torch.no_grad():
            gnorms, finites, step_gnorm_sq = self._batched_grad_norms(active)
            for name, fin in finites.items():
                self._param_finite[name] = fin
            if self._step_gnorm_sq is None:
                self._step_gnorm_sq = step_gnorm_sq
            else:
                self._step_gnorm_sq = self._step_gnorm_sq + step_gnorm_sq.to(device=self._step_gnorm_sq.device)

            ref_dev = self._step_gnorm_sq.device
            finite_stack = torch.stack(
                [finites[ent.name].to(device=ref_dev, dtype=torch.bool) for ent in active]
            )
            all_finite_dev = finite_stack.all() & torch.isfinite(self._step_gnorm_sq)

            if self.skip_nonfinite and not bool(all_finite_dev.item()):
                self._atomic_step_aborted = True
                return

            for ent in active:
                self._stepped.add(ent.name)

            if self.max_grad_norm is not None:
                max_n = float(self.max_grad_norm)
                if self.skip_nonfinite:
                    # All grads are verified finite above: reuse the already-computed global L2 norm
                    # (zero duplicate 897-tensor norm pass!) and apply via fused `_foreach_mul_`.
                    tot_t = torch.sqrt(self._step_gnorm_sq.clamp_min(0.0))
                    clip_coef = torch.clamp(max_n / (tot_t + 1e-6), max=1.0)
                    grad_buckets: dict[tuple[torch.device, torch.dtype], list[torch.Tensor]] = {}
                    for ent in active:
                        g_t = ent.param.grad
                        assert g_t is not None
                        grad_buckets.setdefault((g_t.device, g_t.dtype), []).append(g_t)
                    has_fmul = hasattr(torch, "_foreach_mul_")
                    for (g_dev, g_dt), g_list in grad_buckets.items():
                        coef_dt = torch.float64 if g_dt == torch.float64 else torch.float32
                        coef_d = clip_coef.to(device=g_dev, dtype=coef_dt)
                        if has_fmul and len(g_list) > 1 and not any(g.is_sparse for g in g_list):
                            torch._foreach_mul_(g_list, coef_d)
                        else:
                            for g_t in g_list:
                                g_t.mul_(coef_d)
                else:
                    tot = torch.nn.utils.clip_grad_norm_(
                        [ent.param for ent in active], max_norm=max_n
                    )
                    tot_t = tot if isinstance(tot, torch.Tensor) else torch.tensor(float(tot))
                self._accumulate_device_clip(
                    (torch.isfinite(tot_t) & (tot_t > max_n)).to(dtype=torch.float64) * float(len(active))
                )
            else:
                for ent in active:
                    grad = ent.param.grad
                    assert grad is not None
                    gnorm = gnorms[ent.name]
                    finite = finites[ent.name]
                    scale: torch.Tensor | None = None
                    if self.max_norm is not None:
                        scale = torch.clamp(float(self.max_norm) / (gnorm + 1e-12), max=1.0)
                        self._accumulate_device_clip(finite & (gnorm > float(self.max_norm)))
                    if (
                        self.global_clip is not None
                        and self.prev_global_norm
                        and self.prev_global_norm > self.global_clip
                    ):
                        gcoef = float(self.global_clip) / float(self.prev_global_norm)
                        scale = (scale * gcoef) if scale is not None else torch.full_like(gnorm, gcoef)
                    if scale is not None:
                        safe_scale = torch.where(finite, scale, torch.ones_like(scale)).to(dtype=grad.dtype)
                        grad.mul_(safe_scale)

        lrf = self.lr_factor
        if self._torch_adamw is not None and any(ent.opt is self._torch_adamw for ent in active):
            for pg in self._torch_adamw.param_groups:
                pg["lr"] = float(pg.get("_qloop_base_lr", pg["lr"])) * lrf
            self._torch_adamw.step()

        lite_entries: list[_Entry] = []
        other_entries: list[_Entry] = []
        for ent in active:
            if ent.opt is self._torch_adamw:
                continue
            if isinstance(ent.opt, AdamWLite):
                lite_entries.append(ent)
            else:
                other_entries.append(ent)

        if lite_entries:
            self._step_batched_adamw_lite(lite_entries, lrf)

        for ent in other_entries:
            lr = ent.base_lr * lrf
            for g in ent.opt.param_groups:
                g["lr"] = lr
            ent.opt.step()

        with torch.no_grad():
            for ent in active:
                self._clamp_projected_param(ent.param, ent.opt)

    # ---------------------------------------------------------------- public API
    def set_boundary(self, is_last_microbatch: bool) -> None:
        """Call before each micro-batch backward. False: grads accumulate; True: step in backward."""
        self._boundary = bool(is_last_microbatch)
        self._boundary_explicit = True

    def step(self) -> dict[str, float]:
        """Non-fused: update every param with a grad. Fused: post-backward bookkeeping (global
        grad norm, schedule advance, sanity checks). Call once per optimizer step in both modes."""
        if not self.fused:
            self._step_nonfused()
        else:
            if self._boundary_explicit and not self._boundary:
                log.warning("FusedStepOptimizer.step() called while set_boundary(False): nothing was stepped")
            else:
                # params that got gradient only in earlier (non-boundary) micro-batches never fire a hook
                # in the boundary backward (e.g. a hot expert not routed to in the last micro-batch):
                # their accumulated grad is complete now -> step them here, exactly like fused=False.
                late = [e for n, e in self.entries.items() if e.param.grad is not None and n not in self._stepped]
                for ent in late:
                    self._step_param(ent)
                    ent.param.grad = None
                if late:
                    log.debug(
                        "FusedStepOptimizer: stepped %d params outside backward (no grad in the boundary "
                        "micro-batch), e.g. %s",
                        len(late),
                        [e.name for e in late[:3]],
                    )
            if not self._stepped and not self._warned_empty:
                self._warned_empty = True
                log.warning(
                    "FusedStepOptimizer: no parameter was stepped in this backward - did the loss depend "
                    "on the trainable params and was set_boundary(True) called before backward()?"
                )
        return self._finalize()

    def _finalize(self) -> dict[str, float]:
        gnorm = 0.0
        has_finite_norm = False
        if self._param_finite:
            names = list(self._param_finite.keys())
            ref_dev = self._step_gnorm_sq.device if self._step_gnorm_sq is not None else torch.device("cpu")
            finite_vec = torch.stack([self._param_finite[n].to(device=ref_dev, dtype=torch.bool) for n in names])
            skipped_dev = (~finite_vec).sum(dtype=torch.float64)
            sq_dev = (
                self._step_gnorm_sq.to(device=ref_dev, dtype=torch.float64)
                if self._step_gnorm_sq is not None
                else torch.zeros((), device=ref_dev, dtype=torch.float64)
            )
            clip_dev = (
                self._clipped_dev.to(device=ref_dev, dtype=torch.float64)
                if self._clipped_dev is not None
                else torch.zeros((), device=ref_dev, dtype=torch.float64)
            )
            # Single host transfer for the entire optimizer step (zero per-param .item() syncs!).
            packed_cpu = torch.stack([sq_dev, clip_dev, skipped_dev]).to(device="cpu", dtype=torch.float64)
            sq_val, clip_val, skip_val = (float(x) for x in packed_cpu.numpy())
            gnorm = math.sqrt(max(0.0, sq_val)) if math.isfinite(sq_val) else float("inf")
            self._clipped = round(clip_val)
            self._skipped_now = round(skip_val)
            if self._atomic_step_aborted and self._skipped_now == 0:
                self._skipped_now = len(names)
            has_finite_norm = (
                len(names) > 0
                and self._skipped_now == 0
                and not self._atomic_step_aborted
                and math.isfinite(gnorm)
            )

            if self._skipped_now > 0:
                finite_flags_cpu = finite_vec.to(device="cpu").numpy()
                for n, is_fin in zip(names, finite_flags_cpu):
                    if not bool(is_fin):
                        self.skipped[n] = self.skipped.get(n, 0) + 1
                        ent = self.entries[n]
                        if self.skip_nonfinite and not self._atomic_step_aborted:
                            st = ent.opt.state.get(ent.param)
                            if isinstance(st, dict) and "step" in st:
                                st_step = st["step"]
                                if isinstance(st_step, torch.Tensor):
                                    if float(st_step.detach().to("cpu").numpy()) <= 1.0:
                                        st.clear()
                                    else:
                                        st["step"] = st_step - 1
                                elif int(st_step) <= 1:
                                    st.clear()
                                else:
                                    st["step"] = int(st_step) - 1
                        if self._atomic_step_aborted:
                            act = "skipping entire optimizer step atomically (0 params updated)"
                        elif self.skip_nonfinite:
                            act = "skipping its update"
                        else:
                            act = "APPLYING anyway (skip_nonfinite=False)"
                        log.warning(
                            "non-finite gradient for '%s' at step %d: %s - %s (skips for this param: %d)",
                            n,
                            self.step_count,
                            describe(ent.param),
                            act,
                            self.skipped[n],
                        )

        self.last_global_norm = gnorm
        self.prev_global_norm = gnorm if has_finite_norm else self.prev_global_norm
        if has_finite_norm:
            b = self.norm_ema_beta
            self.norm_ema = gnorm if self.norm_ema is None else b * self.norm_ema + (1 - b) * gnorm
        stats = {
            "optim/global_grad_norm": gnorm,
            "optim/grad_norm_ema": float(self.norm_ema or 0.0),
            "optim/lr_factor": self.lr_factor,
            "optim/stepped_params": float(len(self._stepped)),
            "optim/clipped_params": float(self._clipped),
            "optim/skipped_params": float(self._skipped_now),
            "optim/skipped_total": float(sum(self.skipped.values())),
        }
        self.step_count += 1
        if self.log_every and self.step_count % self.log_every == 0:
            log.info(
                "optim step %d: %s",
                self.step_count,
                " ".join(f"{k.split('/')[1]}={v:.4g}" for k, v in stats.items()),
            )
        self._stepped.clear()
        self._param_finite.clear()
        self._step_gnorm_sq = None
        self._clipped_dev = None
        self._clipped = 0
        self._skipped_now = 0
        self._boundary_explicit = False
        self._atomic_step_aborted = False
        return stats

    def zero_grad(self, set_to_none: bool = True) -> None:
        for ent in self.entries.values():
            if ent.param.grad is not None:
                if set_to_none:
                    ent.param.grad = None
                else:
                    ent.param.grad.zero_()

    def remove_hooks(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def state_dict(self) -> dict[str, Any]:
        params = {}
        for name, ent in self.entries.items():
            params[name] = {
                "state": _copy_state(ent.opt.state.get(ent.param, {})),
                "impl": type(ent.opt).__name__,
                "kind": ent.kind,
                "shape": list(ent.param.shape),
            }
        return {
            "version": 1,
            "step_count": self.step_count,
            "skipped": dict(self.skipped),
            "norm_ema": self.norm_ema,
            "prev_global_norm": self.prev_global_norm,
            "params": params,
        }

    def load_state_dict(self, sd: dict[str, Any], strict: bool = True) -> None:
        require(
            int(sd.get("version", 0)) == 1,
            "unsupported FusedStepOptimizer state_dict version",
            version=sd.get("version"),
        )
        saved = sd["params"]
        missing, unexpected = sorted(set(self.entries) - set(saved)), sorted(set(saved) - set(self.entries))
        if strict:
            require(
                not missing and not unexpected,
                "optimizer state_dict names differ (strict=True)",
                missing=missing[:10],
                unexpected=unexpected[:10],
            )
        elif missing or unexpected:
            log.warning(
                "optimizer state: %d missing (fresh state), %d unexpected (ignored)",
                len(missing),
                len(unexpected),
            )
        for name, rec in saved.items():
            ent = self.entries.get(name)
            if ent is None:
                continue
            require(
                list(rec.get("shape", list(ent.param.shape))) == list(ent.param.shape),
                f"optimizer state shape mismatch for {name!r}",
                saved=rec.get("shape"),
                param=ent.param,
            )
            saved_impl = rec.get("impl")
            cur_impl = type(ent.opt).__name__
            dev = ent.param.device
            if saved_impl != cur_impl:
                if (
                    {saved_impl, cur_impl} == {"AdamWLite", "AdamW"}
                    and isinstance(ent.opt, (AdamWLite, torch.optim.AdamW))
                    and "exp_avg" in rec.get("state", {})
                    and "exp_avg_sq" in rec.get("state", {})
                ):
                    st_in = rec["state"]
                    raw_step = st_in.get("step", 0)
                    step_val = (
                        float(raw_step.detach().to("cpu").item())
                        if isinstance(raw_step, torch.Tensor)
                        else float(raw_step)
                    )
                    if isinstance(ent.opt, torch.optim.AdamW):
                        pg_fused = any(
                            bool(pg.get("fused", False) or pg.get("capturable", False))
                            for pg in ent.opt.param_groups
                        )
                        step_dev = dev if pg_fused else torch.device("cpu")
                        ent.opt.state[ent.param] = {
                            "step": torch.tensor(step_val, dtype=torch.float32, device=step_dev),
                            "exp_avg": st_in["exp_avg"].detach().clone().to(device=dev, dtype=ent.param.dtype),
                            "exp_avg_sq": st_in["exp_avg_sq"].detach().clone().to(device=dev, dtype=ent.param.dtype),
                        }
                    else:
                        sd_dt = _state_dtype(ent.param)
                        new_st: dict[str, Any] = {
                            "step": round(step_val),
                            "exp_avg": st_in["exp_avg"].detach().clone().to(device=dev, dtype=sd_dt),
                            "exp_avg_sq": st_in["exp_avg_sq"].detach().clone().to(device=dev, dtype=sd_dt),
                        }
                        if self.master_weights and ent.param.dtype in _LOWP:
                            new_st["master"] = (
                                st_in["master"].detach().clone().to(device=dev, dtype=sd_dt)
                                if "master" in st_in
                                else ent.param.detach().to(device=dev, dtype=sd_dt)
                            )
                        ent.opt.state[ent.param] = new_st
                    continue
                if {saved_impl, cur_impl} == {"AdamWLite", "AdamW"} and not rec.get("state"):
                    ent.opt.state.pop(ent.param, None)
                    continue
                msg = f"optimizer impl changed for {name!r}: saved {saved_impl} vs now {cur_impl}"
                require(not strict, msg + " (state layouts are incompatible)")
                log.warning("%s - starting this param with fresh state", msg)
                continue
            if not rec["state"]:
                ent.opt.state.pop(ent.param, None)
                continue
            new_state: dict[str, Any] = {
                k: _move_state_value(v, dev) for k, v in rec["state"].items()
            }
            if isinstance(ent.opt, torch.optim.AdamW) and isinstance(new_state.get("step"), torch.Tensor):
                pg_fused = any(
                    bool(pg.get("fused", False) or pg.get("capturable", False))
                    for pg in ent.opt.param_groups
                )
                step_dev = dev if pg_fused else torch.device("cpu")
                new_state["step"] = new_state["step"].to(device=step_dev, dtype=torch.float32)
            ent.opt.state[ent.param] = new_state
        self.step_count = int(sd.get("step_count", 0))
        self.skipped = dict(sd.get("skipped", {}))
        self.norm_ema = sd.get("norm_ema")
        self.prev_global_norm = sd.get("prev_global_norm")
        log.info("optimizer state loaded: %d params, step %d", len(saved), self.step_count)
