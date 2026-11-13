"""Assemble the full trainable stack from a config, in the only order that is correct:

    load base (frozen)  ->  D-LSA indexers  ->  hot-expert deltas  ->  stable names
    ->  freeze policy on base params  ->  loop wrap (64 virtual layers)  ->  trainable dict

Stable parameter names are recorded *before* loop wrapping (wrapping renames core/coda layers),
so checkpoints and optimizer state are portable across loop configs.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import nn

from ..debug import register_nonfinite_hooks, require, stage
from ..model.experts import ExpertStats, HotExpertDeltas, attach_hot_expert_deltas, freeze_all_but
from ..model.introspect import ModelParts, inspect_model
from ..model.loading import load_model
from ..model.loop import LoopController, apply_loop
from ..model.lsa import LSAManager, install_lsa
from .checkpoint import stable_names

log = logging.getLogger("qloop")


@dataclass
class Stack:
    model: nn.Module
    tokenizer: Any
    parts: ModelParts
    loop: LoopController | None = None
    lsa: LSAManager | None = None
    deltas: HotExpertDeltas | None = None
    named_trainable: dict[str, nn.Parameter] = field(default_factory=dict)
    load_info: dict[str, Any] = field(default_factory=dict)
    hot_map: dict[int, list[int]] | None = None

    @property
    def device(self) -> torch.device:
        return self.parts.text_model.embed_tokens.weight.device

    @property
    def lm_head_weight(self) -> torch.Tensor:
        require(self.parts.lm_head is not None, "model has no lm_head")
        return self.parts.lm_head.weight

    def hidden(self, input_ids: torch.Tensor, position_ids: torch.Tensor | None = None) -> torch.Tensor:
        kw: dict[str, Any] = {"input_ids": input_ids, "use_cache": False}
        if position_ids is not None:
            kw["position_ids"] = position_ids
        return self.parts.text_model(**kw).last_hidden_state

    def aux_losses(self) -> dict[str, torch.Tensor]:
        out: dict[str, torch.Tensor] = {}
        if self.loop is not None:
            a = self.loop.aux_loss()
            if a is not None:
                out["loop_aux"] = a
        if self.lsa is not None:
            a = self.lsa.aux_loss()
            if a is not None:
                out["lsa_kl"] = a
        return out

    def stats(self) -> dict[str, float]:
        s: dict[str, float] = {}
        if self.loop is not None:
            s.update(self.loop.stats())
        if self.lsa is not None:
            s.update(self.lsa.stats())
        return s

    def describe(self) -> dict[str, Any]:
        return {
            "loop": None if self.loop is None else self.loop.spec.__dict__,
            "lsa": None if self.lsa is None else self.lsa.cfg.__dict__,
            "hot_experts": None if self.hot_map is None else {str(k): v for k, v in self.hot_map.items()},
            "n_trainable": sum(p.numel() for p in self.named_trainable.values()),
        }


def _hot_map(ecfg: dict[str, Any]) -> dict[int, list[int]] | None:
    if ecfg.get("hot_map_path"):
        import json

        with open(ecfg["hot_map_path"], encoding="utf-8") as f:
            raw = json.load(f)
        return {int(k): [int(e) for e in v] for k, v in raw.items()}
    path = ecfg.get("stats_path")
    require(bool(path) and os.path.exists(path), "experts.enabled needs experts.stats_path (run `qloop scan-experts`) "
            "or experts.hot_map_path", stats_path=path)
    stats = ExpertStats.from_json(path)
    sel: dict[str, Any] = {"by": ecfg.get("by", "affinity"), "per_pass": ecfg.get("per_pass", "union")}
    if ecfg.get("per_layer_k") is not None:
        sel["per_layer_k"] = int(ecfg["per_layer_k"])
    elif ecfg.get("mass") is not None:
        sel["mass"] = float(ecfg["mass"])
    else:
        sel["fraction"] = float(ecfg.get("fraction", 0.1))
    return stats.select_hot(**sel)


def build_stack(cfg: dict[str, Any]) -> Stack:
    mcfg = dict(cfg.get("model") or {"name": "toy"})
    with stage("model:load"):
        model, tok, info = load_model(mcfg)
    st = Stack(model, tok, inspect_model(model), load_info=info)

    lcfg = dict(cfg.get("lsa") or {})
    if lcfg.pop("enabled", False):
        with stage("model:lsa"):
            st.lsa = install_lsa(model, lcfg)

    ecfg = dict(cfg.get("experts") or {})
    if ecfg.get("enabled", False):
        with stage("model:expert-deltas"):
            st.hot_map = _hot_map(ecfg)
            n_hot = sum(len(v) for v in st.hot_map.values())
            log.info("hot experts: %d across %d layers", n_hot, len(st.hot_map))
            st.deltas = attach_hot_expert_deltas(model, st.hot_map, rank=int(ecfg.get("rank", 0)),
                                                 alpha=ecfg.get("alpha"))
            ref = st.parts.text_model.embed_tokens.weight
            st.deltas.to(device=ref.device, dtype=torch.float32 if ref.dtype != torch.float64 else ref.dtype)

    names = stable_names(model)
    tcfg = dict(cfg.get("train") or {})
    patterns = list(tcfg.get("trainable") or [])
    if st.lsa is not None and tcfg.get("train_lsa", True):
        patterns.append(r"lsa_indexer\.")
    if st.deltas is not None and tcfg.get("train_expert_deltas", True):
        patterns.append(r"expert_deltas\.")
    with stage("model:freeze"):
        if patterns:
            report = freeze_all_but(model, patterns, allowed_groups=tcfg.get("allowed_groups"),
                                    strict_patterns=bool(tcfg.get("strict_patterns", True)))
            log.info("freeze policy:\n%s", report)
        else:
            for p in model.parameters():
                p.requires_grad_(False)

    loop_cfg = dict(cfg.get("loop") or {})
    if loop_cfg.pop("enabled", False):
        with stage("model:loop"):
            st.loop = apply_loop(model, loop_cfg)
            for n, p in st.loop.adapters.named_parameters():
                names[id(p)] = f"loop_adapters.{n}"
                p.requires_grad_(bool(tcfg.get("train_loop_adapters", True)))

    params = {id(p): p for p in model.parameters()}
    if st.loop is not None:
        params.update({id(p): p for p in st.loop.adapters.parameters()})
    st.named_trainable = {names[i]: p for i, p in params.items() if p.requires_grad}
    require(all(i in names for i in params), "a parameter has no stable name (internal error)")
    n_tr = sum(p.numel() for p in st.named_trainable.values())
    log.info("trainable tensors: %d (%.3fM params)", len(st.named_trainable), n_tr / 1e6)
    if int((cfg.get("debug") or {}).get("level", 1)) >= 2 or (cfg.get("debug") or {}).get("nan_hooks"):
        register_nonfinite_hooks(model)
    return st
