"""Locate the pieces of a (toy or real HF) Qwen3.5/3.6-MoE model the retrofit needs.

All lookups are structural (by shape/attribute), never by hard-coded dotted paths, because
the multimodal `Qwen3_5MoeForConditionalGeneration` nests the text model differently from
`...ForCausalLM`, and transformers moves things between releases. Every failure raises an
error listing what *was* found, so a version drift is diagnosable from the message alone.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from torch import nn

LINEAR, FULL = "linear_attention", "full_attention"


@dataclass
class ModelParts:
    root: nn.Module            # whatever the user loaded
    text_model: nn.Module      # has .layers / .embed_tokens / .norm
    text_config: Any           # config object whose num_hidden_layers/layer_types drive caches
    layers: nn.ModuleList
    lm_head: nn.Module | None
    layer_types: list[str]

    @property
    def num_layers(self) -> int:
        return len(self.layers)

    def full_attention_indices(self) -> list[int]:
        return [i for i, t in enumerate(self.layer_types) if t == FULL]


def find_text_model(root: nn.Module) -> nn.Module:
    cands = []
    for name, mod in root.named_modules():
        layers = getattr(mod, "layers", None)
        if isinstance(layers, nn.ModuleList) and hasattr(mod, "embed_tokens") and len(layers) > 0:
            cands.append((name, mod))
    if not cands:
        seen = [n for n, m in root.named_modules() if isinstance(getattr(m, "layers", None), nn.ModuleList)]
        raise RuntimeError(
            "could not find a text decoder (module with `.layers` ModuleList and `.embed_tokens`). "
            f"Modules with `.layers`: {seen[:10]}"
        )
    # prefer the shallowest match (vision towers also have `.layers` but no embed_tokens)
    cands.sort(key=lambda x: x[0].count("."))
    return cands[0][1]


def find_lm_head(root: nn.Module, vocab_size: int | None) -> nn.Module | None:
    head = getattr(root, "lm_head", None)
    if isinstance(head, nn.Module):
        return head
    for _name, mod in root.named_modules():
        if isinstance(mod, nn.Linear) and vocab_size and mod.out_features == vocab_size:
            return mod
    return None


def layer_type_of(layer: nn.Module, idx: int, config: Any) -> str:
    lt = getattr(layer, "layer_type", None)
    if lt is None:
        types = getattr(config, "layer_types", None)
        if types:
            lt = types[idx]
        elif hasattr(layer, "linear_attn"):
            lt = LINEAR
        elif hasattr(layer, "self_attn"):
            lt = FULL
    if lt not in (LINEAR, FULL):
        raise RuntimeError(f"layer {idx}: cannot determine layer type (got {lt!r}); "
                           f"children={[n for n, _ in layer.named_children()]}")
    return lt


def inspect_model(root: nn.Module) -> ModelParts:
    tm = find_text_model(root)
    cfg = getattr(tm, "config", None)
    if cfg is None:
        raise RuntimeError("text model has no `.config`")
    layers = tm.layers
    n_cfg = getattr(cfg, "num_hidden_layers", None)
    if n_cfg is not None and n_cfg != len(layers):
        raise RuntimeError(
            f"config.num_hidden_layers={n_cfg} but found {len(layers)} layers - is the model already "
            "loop-wrapped? call `controller.unwrap()` first."
        )
    types = [layer_type_of(layer, i, cfg) for i, layer in enumerate(layers)]
    return ModelParts(root, tm, cfg, layers, find_lm_head(root, getattr(cfg, "vocab_size", None)), types)


def attention_module(layer: nn.Module, layer_type: str) -> nn.Module:
    name = "linear_attn" if layer_type == LINEAR else "self_attn"
    mod = getattr(layer, name, None)
    if mod is None:
        raise RuntimeError(f"expected `{name}` on {type(layer).__name__}; children: "
                           f"{[n for n, _ in layer.named_children()]}")
    return mod


def find_children(mod: nn.Module, names: list[str]) -> dict[str, nn.Module]:
    """Direct-or-nested submodules whose *leaf name* is in `names`."""
    out = {}
    for full, sub in mod.named_modules():
        leaf = full.rsplit(".", 1)[-1]
        if leaf in names:
            out[full] = sub
    return out


def modules_with_layer_idx(layer: nn.Module) -> list[nn.Module]:
    return [m for m in layer.modules() if isinstance(getattr(m, "layer_idx", None), int)]


def router_of(layer: nn.Module) -> nn.Module | None:
    mlp = getattr(layer, "mlp", None)
    return getattr(mlp, "gate", None) if mlp is not None else None


def summarize(parts: ModelParts) -> str:
    fa = parts.full_attention_indices()
    n_params = sum(p.numel() for p in parts.root.parameters())
    return (f"{type(parts.root).__name__}: {parts.num_layers} layers "
            f"({parts.num_layers - len(fa)} linear / {len(fa)} full @ {fa}), "
            f"hidden={getattr(parts.text_config, 'hidden_size', '?')}, "
            f"experts={getattr(parts.text_config, 'num_experts', '?')}"
            f"x top{getattr(parts.text_config, 'num_experts_per_tok', '?')}, params={n_params/1e9:.3f}B")
