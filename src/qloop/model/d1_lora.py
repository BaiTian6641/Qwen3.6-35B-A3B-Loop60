"""Sprint 2.2 Parameter-Matched d=1 (40L) LoRA Control (`A2` scientific control gate).

Provides zero-initialised forward-hook LoRA residual projections attached to physical core layers
`[layer_start, layer_end)` (`[12, 36)` on 40L `Qwen3.6-35B-A3B`) so a non-looped `d=1` baseline
matches `s1_loop_a6b`'s `~18.41M` trainable parameter budget under the exact same `LMTrainer` +
`FusedStepOptimizer` + `CheckpointManager` + `sha256_frozen_experts` pipeline.

Key invariants:
  * Never modifies or replaces frozen base linear modules (works seamlessly on BF16 `nn.Linear`
    and frozen Blackwell FP8/NVFP4 quantized linear modules via `_linear_in_out_features`).
  * Exact IEEE-754 Step-0 bit-identity (`out + 0 == out`, `torch.equal`) with zero-initialised `up`
    (`B`) and Kaiming/normal `down` (`A`), while preserving strictly non-zero Step-0 `up.grad`.
  * Zero impact on `sha256_frozen_experts(model)`.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, fields
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from ._dtype import upcast
from .introspect import FULL, LINEAR, ModelParts, attention_module, find_children, inspect_model
from .quant_autograd import _linear_in_out_features

log = logging.getLogger("qloop")

DEFAULT_ATTN_PROJS: tuple[str, ...] = ("q_proj", "o_proj")
DEFAULT_LINEAR_PROJS: tuple[str, ...] = ("in_proj_qkv", "in_proj_qkvz", "out_proj")
_LINEAR_INPUT_PROJ_ALIASES: frozenset[str] = frozenset({"in_proj_qkv", "in_proj_qkvz"})

# Canonical Qwen3.6-35B-A3B (40L) parameter counts matching `s1_loop_a6b` (core [12, 36)):
#   * 40 layers * (input_layernorm 2048 + post_attention_layernorm 2048) = 163,840
#   * 10 full-attention layers * (q_norm 256 + k_norm 256)              =   5,120
#   * Total trainable norm params                                       = 168,960
#   * s1_loop_a6b loop_adapters params                                  = 18,243,606
#   * s1_loop_a6b total trainable params                                = 18,412,566 (~18.41M)
QWEN36_NORM_PARAMS: int = 168_960
S1_LOOP_ADAPTER_PARAMS: int = 18_243_606
S1_LOOP_TOTAL_TRAINABLE_PARAMS: int = S1_LOOP_ADAPTER_PARAMS + QWEN36_NORM_PARAMS


@dataclass
class D1LoRAConfig:
    """Configuration for Sprint 2.2 parameter-matched `d=1` (`40L`) LoRA control."""

    enabled: bool = False
    layer_start: int = 12
    layer_end: int = 36
    rank: int = 46
    alpha: float | None = None
    attn_projs: tuple[str, ...] = DEFAULT_ATTN_PROJS
    linear_projs: tuple[str, ...] = DEFAULT_LINEAR_PROJS
    target_params: int | None = None
    seed: int = 0

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> D1LoRAConfig:
        raw = dict(d or {})
        known = {f.name for f in fields(cls)}
        unknown = set(raw) - known
        if unknown:
            raise ValueError(f"unknown d1_lora.* keys: {sorted(unknown)}; valid: {sorted(known)}")
        for key in ("attn_projs", "linear_projs"):
            if key in raw and isinstance(raw[key], (list, tuple)):
                raw[key] = tuple(str(x) for x in raw[key])
        if "enabled" in raw:
            raw["enabled"] = bool(raw["enabled"])
        for int_key in ("layer_start", "layer_end", "rank", "seed"):
            if int_key in raw and raw[int_key] is not None:
                raw[int_key] = int(raw[int_key])
        if "target_params" in raw and raw["target_params"] is not None:
            raw["target_params"] = int(raw["target_params"])
        if "alpha" in raw and raw["alpha"] is not None:
            raw["alpha"] = float(raw["alpha"])
        return cls(**raw)

    def validate(self, num_layers: int | None = None) -> None:
        if self.layer_start < 0 or self.layer_start >= self.layer_end:
            raise ValueError(
                f"d1_lora requires 0 <= layer_start < layer_end, got [{self.layer_start}, {self.layer_end})"
            )
        if num_layers is not None and self.layer_end > num_layers:
            raise ValueError(
                f"d1_lora layer range [{self.layer_start}, {self.layer_end}) out of bounds "
                f"for model with {num_layers} layers"
            )
        if self.rank < 1:
            raise ValueError(f"d1_lora.rank must be >= 1, got {self.rank}")
        if self.alpha is not None and (not math.isfinite(self.alpha) or self.alpha < 0.0):
            raise ValueError(f"d1_lora.alpha must be a finite non-negative float or None, got {self.alpha!r}")
        if self.target_params is not None and self.target_params <= 0:
            raise ValueError(f"d1_lora.target_params must be > 0 or None, got {self.target_params}")
        if not self.attn_projs and not self.linear_projs:
            raise ValueError("d1_lora requires at least one entry in attn_projs or linear_projs")

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": bool(self.enabled),
            "layer_start": int(self.layer_start),
            "layer_end": int(self.layer_end),
            "rank": int(self.rank),
            "alpha": None if self.alpha is None else float(self.alpha),
            "attn_projs": list(self.attn_projs),
            "linear_projs": list(self.linear_projs),
            "target_params": None if self.target_params is None else int(self.target_params),
            "seed": int(self.seed),
        }


class _LoRAResidualProj(nn.Module):
    """Zero-initialised rank-`r` residual LoRA projection `scale * (x @ down^T) @ up^T`.

    `down` (`A`: `[rank, in_features]`) uses deterministic Kaiming/normal initialisation
    (`std = 1 / sqrt(in_features)`) and `up` (`B`: `[out_features, rank]`) is zero-initialised,
    guaranteeing exact IEEE-754 Step-0 bit-identity (`out + 0 == out`) AND non-zero Step-0 `up.grad`.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int,
        alpha: float | None = None,
        *,
        seed: int = 0,
    ) -> None:
        super().__init__()
        in_f = int(in_features)
        out_f = int(out_features)
        r = int(rank)
        if in_f <= 0 or out_f <= 0 or r <= 0:
            raise ValueError(
                f"_LoRAResidualProj requires positive (in_features, out_features, rank), "
                f"got ({in_f}, {out_f}, {r})"
            )
        self.in_features = in_f
        self.out_features = out_f
        self.rank = r
        eff_alpha = float(r) if alpha is None else float(alpha)
        self.alpha = eff_alpha
        self.scale = eff_alpha / float(r)

        gen = torch.Generator(device="cpu").manual_seed(int(seed) % (2**63 - 1))
        down = torch.empty(r, in_f, dtype=torch.float32)
        down.normal_(mean=0.0, std=1.0 / math.sqrt(float(in_f)), generator=gen)
        self.down = nn.Parameter(down)
        self.up = nn.Parameter(torch.zeros(out_f, r, dtype=torch.float32))

    def is_zero(self) -> bool:
        with torch.no_grad():
            return bool((self.up == 0).all())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_up = upcast(x)
        if x_up.device != self.down.device or x_up.dtype != self.down.dtype:
            x_up = x_up.to(device=self.down.device, dtype=self.down.dtype)
        h = F.linear(x_up, self.down)
        delta = F.linear(h, self.up)
        if self.scale != 1.0:
            delta = delta * self.scale
        return delta


def _module_device(mod: nn.Module, default: torch.device) -> torch.device:
    for p in mod.parameters():
        return p.device
    for b in mod.buffers():
        return b.device
    for val in vars(mod).values():
        if isinstance(val, torch.Tensor):
            return val.device
    return default


def _discover_target_projections(
    parts: ModelParts,
    *,
    layer_start: int,
    layer_end: int,
    attn_projs: tuple[str, ...],
    linear_projs: tuple[str, ...],
) -> list[tuple[int, str, str, nn.Module, int, int]]:
    """Discover `(phys_idx, layer_type, proj_key, module, in_features, out_features)` targets.

    Avoids double-hooking when a linear-attention layer exposes both `in_proj_qkv` and
    `in_proj_qkvz` by selecting only the first matching input projection alias in `linear_projs`.
    """
    n_layers = parts.num_layers
    if layer_start < 0 or layer_start >= layer_end or layer_end > n_layers:
        raise ValueError(
            f"d1_lora layer range [{layer_start}, {layer_end}) out of bounds for model with {n_layers} layers"
        )

    targets: list[tuple[int, str, str, nn.Module, int, int]] = []
    for phys in range(layer_start, layer_end):
        layer = parts.layers[phys]
        ltype = parts.layer_types[phys]
        attn = attention_module(layer, ltype)
        wanted = attn_projs if ltype == FULL else linear_projs
        if not wanted:
            continue

        found_map = find_children(attn, list(wanted))
        by_leaf: dict[str, list[tuple[str, nn.Module]]] = {}
        for full_name, sub in found_map.items():
            leaf = full_name.rsplit(".", 1)[-1]
            by_leaf.setdefault(leaf, []).append((full_name, sub))

        layer_hits: list[tuple[int, str, str, nn.Module, int, int]] = []
        seen_mod_ids: set[int] = set()
        picked_linear_in_proj = False

        for pname in wanted:
            if ltype == LINEAR and pname in _LINEAR_INPUT_PROJ_ALIASES and picked_linear_in_proj:
                continue
            direct = getattr(attn, pname, None)
            cands: list[tuple[str, nn.Module]] = (
                [(pname, direct)] if isinstance(direct, nn.Module) else by_leaf.get(pname, [])
            )
            if not cands:
                continue
            for full_name, sub in cands:
                if id(sub) in seen_mod_ids:
                    continue
                seen_mod_ids.add(id(sub))
                in_f, out_f = _linear_in_out_features(sub)
                key = full_name.replace(".", "__")
                layer_hits.append((phys, ltype, key, sub, in_f, out_f))
                if ltype == LINEAR and pname in _LINEAR_INPUT_PROJ_ALIASES:
                    picked_linear_in_proj = True
                    break

        if not layer_hits:
            raise RuntimeError(
                f"d1_lora: layer {phys} ({ltype}) matched none of {wanted}; "
                f"available children: {[n for n, _ in attn.named_children()]}"
            )
        targets.extend(layer_hits)

    return targets


def compute_qwen36_d1_lora_params(
    rank: int,
    *,
    layer_start: int = 12,
    layer_end: int = 36,
    num_layers: int = 40,
    full_attention_interval: int = 4,
    hidden_size: int = 2048,
    q_proj_out: int = 8192,
    o_proj_in: int = 4096,
    linear_in_proj_out: int | None = None,
    linear_out_proj_in: int = 4096,
    use_qkvz: bool = False,
    head_dim: int = 256,
    include_norms: bool = False,
) -> dict[str, int]:
    """Closed-form parameter accounting for Qwen3.6-35B-A3B `d=1` LoRA control on `[layer_start, layer_end)`.

    Default dimensions match `Qwen/Qwen3.6-35B-A3B` (`40L`, `3:1` GDN/Full-Attn, `hidden=2048`):
      * Full attention (`6` layers in `[12, 36)`): `q_proj [2048->8192]` + `o_proj [4096->2048]`
        = `16,384 * r` / layer -> `98,304 * r`.
      * Linear GDN (`18` layers in `[12, 36)`):
          - `in_proj_qkv [2048->8192]` + `out_proj [4096->2048]` = `16,384 * r` / layer -> `294,912 * r`
            (total `393,216 * r`; `rank=46` -> `18,087,936` LoRA + `168,960` norms = `18.257M`).
          - `in_proj_qkvz [2048->12288]` + `out_proj [4096->2048]` = `20,480 * r` / layer -> `368,640 * r`
            (total `466,944 * r`; `rank=39` -> `18,210,816` LoRA + `168,960` norms = `18.380M`).
    """
    r = int(rank)
    if r < 0:
        raise ValueError(f"rank must be >= 0, got {r}")
    if not (0 <= layer_start < layer_end <= num_layers):
        raise ValueError(
            f"requires 0 <= layer_start < layer_end <= num_layers ({num_layers}), "
            f"got [{layer_start}, {layer_end})"
        )
    if full_attention_interval <= 0:
        raise ValueError(f"full_attention_interval must be > 0, got {full_attention_interval}")

    lin_in_out = int(linear_in_proj_out) if linear_in_proj_out is not None else (12288 if use_qkvz else 8192)
    n_total_full = sum(1 for i in range(num_layers) if (i + 1) % full_attention_interval == 0)
    n_full = sum(1 for i in range(layer_start, layer_end) if (i + 1) % full_attention_interval == 0)
    n_linear = (layer_end - layer_start) - n_full

    full_per_layer = (int(hidden_size) + int(q_proj_out)) + (int(o_proj_in) + int(hidden_size))
    linear_per_layer = (int(hidden_size) + lin_in_out) + (int(linear_out_proj_in) + int(hidden_size))

    full_per_rank = n_full * full_per_layer
    linear_per_rank = n_linear * linear_per_layer
    params_per_rank = full_per_rank + linear_per_rank

    full_lora = full_per_rank * r
    linear_lora = linear_per_rank * r
    lora_params = full_lora + linear_lora
    norm_params = int(num_layers) * 2 * int(hidden_size) + n_total_full * 2 * int(head_dim)
    total_with_norms = lora_params + norm_params

    return {
        "rank": r,
        "n_full_layers": n_full,
        "n_linear_layers": n_linear,
        "params_per_rank": params_per_rank,
        "full_attn_lora_params": full_lora,
        "linear_attn_lora_params": linear_lora,
        "lora_params": lora_params,
        "norm_params": norm_params,
        "total_with_norms": total_with_norms,
        "total_params": total_with_norms if include_norms else lora_params,
    }


def solve_rank_for_target_params(
    model: nn.Module,
    target_params: int,
    *,
    layer_start: int = 12,
    layer_end: int = 36,
    attn_projs: tuple[str, ...] = DEFAULT_ATTN_PROJS,
    linear_projs: tuple[str, ...] = DEFAULT_LINEAR_PROJS,
    min_rank: int = 1,
    max_rank: int | None = None,
    fixed_params: int = 0,
) -> int:
    """Solve for the closest positive integer LoRA rank matching `target_params` on `model`.

    Inspects the actual projection shapes `(in_features, out_features)` in `[layer_start, layer_end)`
    (handling both `in_proj_qkv` and `in_proj_qkvz` layouts as well as BF16 / FP8 / NVFP4 modules)
    and selects the integer rank minimizing `|rank * params_per_rank + fixed_params - target_params|`.
    """
    if int(target_params) <= 0:
        raise ValueError(f"target_params must be > 0, got {target_params}")
    if int(min_rank) < 1:
        raise ValueError(f"min_rank must be >= 1, got {min_rank}")
    if max_rank is not None and int(max_rank) < int(min_rank):
        raise ValueError(f"max_rank ({max_rank}) must be >= min_rank ({min_rank})")

    parts = inspect_model(model)
    targets = _discover_target_projections(
        parts,
        layer_start=int(layer_start),
        layer_end=int(layer_end),
        attn_projs=tuple(attn_projs),
        linear_projs=tuple(linear_projs),
    )
    params_per_rank = sum(in_f + out_f for _, _, _, _, in_f, out_f in targets)
    if params_per_rank <= 0:
        raise RuntimeError("d1_lora: discovered target projections have zero params_per_rank")

    eff_target = max(1, int(target_params) - int(fixed_params))
    lo = max(int(min_rank), eff_target // params_per_rank)
    hi = lo + 1
    best = lo if abs(lo * params_per_rank - eff_target) <= abs(hi * params_per_rank - eff_target) else hi
    if max_rank is not None:
        best = min(best, int(max_rank))
    return int(best)


class D1LoRAManager:
    """Forward-hook manager for parameter-matched `d=1` (`40L`) LoRA control adapters."""

    def __init__(self, model: nn.Module, cfg: D1LoRAConfig) -> None:
        parts = inspect_model(model)
        cfg.validate(num_layers=parts.num_layers)
        tm = parts.text_model
        if getattr(tm, "d1_lora", None) is not None:
            raise RuntimeError("d1_lora is already installed on this model; call remove() first")

        self.parts = parts
        self.cfg = cfg
        self.enabled: bool = True
        self._removed: bool = False
        self._hooks: list[Any] = []

        targets = _discover_target_projections(
            parts,
            layer_start=cfg.layer_start,
            layer_end=cfg.layer_end,
            attn_projs=cfg.attn_projs,
            linear_projs=cfg.linear_projs,
        )
        params_per_rank = sum(in_f + out_f for _, _, _, _, in_f, out_f in targets)

        if cfg.target_params is not None:
            lo = max(1, int(cfg.target_params) // params_per_rank)
            hi = lo + 1
            resolved_rank = (
                lo
                if abs(lo * params_per_rank - int(cfg.target_params))
                <= abs(hi * params_per_rank - int(cfg.target_params))
                else hi
            )
        else:
            resolved_rank = int(cfg.rank)

        self.rank: int = int(resolved_rank)
        self.alpha: float = float(self.rank) if cfg.alpha is None else float(cfg.alpha)
        self.scale: float = self.alpha / float(self.rank)
        self._targets: list[tuple[int, str, str, int, int]] = []

        ref = tm.embed_tokens.weight
        adapter_dtype = torch.float64 if ref.dtype == torch.float64 else torch.float32
        self.adapters = nn.ModuleDict()

        for idx, (phys, ltype, proj_key, mod, in_f, out_f) in enumerate(targets):
            layer_key = f"L{phys}"
            if layer_key not in self.adapters:
                self.adapters[layer_key] = nn.ModuleDict()
            pdict: nn.ModuleDict = self.adapters[layer_key]  # type: ignore[assignment]
            proj_seed = (int(cfg.seed) * 1_000_003 + 7919 + phys * 131 + idx * 37) % (2**63 - 1)
            proj = _LoRAResidualProj(
                in_features=in_f,
                out_features=out_f,
                rank=self.rank,
                alpha=self.alpha,
                seed=proj_seed,
            ).to(device=_module_device(mod, ref.device), dtype=adapter_dtype)
            pdict[proj_key] = proj
            self._targets.append((phys, ltype, proj_key, in_f, out_f))
            self._hooks.append(mod.register_forward_hook(self._make_hook(proj)))

        tm.d1_lora = self.adapters
        log.info(
            "d1_lora installed: layers=[%d, %d) rank=%d alpha=%.4g scale=%.4g hooks=%d params=%d (%.3fM)",
            cfg.layer_start,
            cfg.layer_end,
            self.rank,
            self.alpha,
            self.scale,
            len(self._hooks),
            self.param_count(),
            self.param_count() / 1e6,
        )

    def _make_hook(self, proj: _LoRAResidualProj):
        def hook(_mod: nn.Module, inp: tuple[Any, ...], out: Any) -> Any:
            if not self.enabled:
                return None
            if not inp or not isinstance(inp[0], torch.Tensor):
                return None
            if not torch.is_grad_enabled() and proj.is_zero():
                return None
            x_in = inp[0]
            if isinstance(out, tuple):
                if not out or not isinstance(out[0], torch.Tensor):
                    return None
                base_out = out[0]
                delta = proj(x_in).to(device=base_out.device, dtype=base_out.dtype).view_as(base_out)
                return (base_out + delta, *out[1:])
            if not isinstance(out, torch.Tensor):
                return None
            delta = proj(x_in).to(device=out.device, dtype=out.dtype).view_as(out)
            return out + delta

        return hook

    def set_enabled(self, enabled: bool) -> None:
        self.enabled = bool(enabled)

    def remove(self) -> None:
        if self._removed:
            return
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
        tm = self.parts.text_model
        if getattr(tm, "d1_lora", None) is self.adapters:
            delattr(tm, "d1_lora")
        self.enabled = False
        self._removed = True

    def param_count(self) -> int:
        return sum(p.numel() for p in self.adapters.parameters())

    def stats(self) -> dict[str, float]:
        total_abs = 0.0
        total_el = 0
        for pdict in self.adapters.values():
            if not isinstance(pdict, nn.ModuleDict):
                continue
            for proj in pdict.values():
                if isinstance(proj, _LoRAResidualProj):
                    u = proj.up.detach().float()
                    total_abs += float(u.abs().sum().item())
                    total_el += int(u.numel())
        return {
            "d1_lora/up_abs_mean": total_abs / max(total_el, 1),
            "d1_lora/n_hooks": float(len(self._hooks)),
            "d1_lora/n_params": float(self.param_count()),
        }

    def describe(self) -> dict[str, Any]:
        return {
            **self.cfg.to_dict(),
            "enabled": True,
            "active": bool(self.enabled),
            "resolved_rank": int(self.rank),
            "resolved_alpha": float(self.alpha),
            "scale": float(self.scale),
            "n_hooks": len(self._hooks),
            "n_params": self.param_count(),
            "layers": sorted({phys for phys, *_ in self._targets}),
        }


def install_d1_lora(model: nn.Module, cfg: D1LoRAConfig | dict[str, Any] | None = None) -> D1LoRAManager:
    """Install parameter-matched `d=1` LoRA control hooks onto physical layers `[layer_start, layer_end)`."""
    if cfg is None:
        resolved = D1LoRAConfig(enabled=True)
    elif isinstance(cfg, D1LoRAConfig):
        resolved = cfg
    else:
        resolved = D1LoRAConfig.from_dict(cfg)
    return D1LoRAManager(model, resolved)
