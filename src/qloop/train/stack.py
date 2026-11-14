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
from ..model.experts import (
    ExpertStats,
    HotExpertDeltas,
    attach_hot_expert_deltas,
    fold_expert_deltas_state_into_base_,
    freeze_all_but,
)
from ..model.gdn2 import GDN2Manager, install_gdn2
from ..model.introspect import ModelParts, inspect_model
from ..model.loading import load_model
from ..model.loop import LoopController, apply_loop
from ..model.lsa import LSAManager, install_lsa
from .checkpoint import collect_trainable, stable_names

log = logging.getLogger("qloop")


@dataclass
class Stack:
    model: nn.Module
    tokenizer: Any
    parts: ModelParts
    loop: LoopController | None = None
    gdn2: GDN2Manager | None = None
    lsa: LSAManager | None = None
    deltas: HotExpertDeltas | None = None
    named_trainable: dict[str, nn.Parameter] = field(default_factory=dict)
    named_all: dict[str, nn.Parameter] = field(default_factory=dict)
    carried_params: dict[str, nn.Parameter] = field(default_factory=dict)
    passthrough_tensors: dict[str, torch.Tensor] = field(default_factory=dict)
    carried_gdn2_cfg: dict[str, Any] | None = None
    carried_lsa_cfg: dict[str, Any] | None = None
    _folded_experts: set[tuple[int, int]] = field(default_factory=set)
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
        if self.gdn2 is not None:
            s.update(self.gdn2.stats())
        if self.lsa is not None:
            s.update(self.lsa.stats())
        return s

    def checkpoint_params(self) -> dict[str, torch.Tensor]:
        """Cumulative diff relative to the pristine HF base model (`passthrough + carried + trainable`)."""
        out: dict[str, torch.Tensor] = dict(self.passthrough_tensors)
        for k, p in self.carried_params.items():
            out[k] = p.detach()
        out.update(collect_trainable(self.named_trainable))
        return out

    def restore_checkpoint(
        self,
        saved: dict[str, torch.Tensor],
        *,
        strict: bool = True,
        expert_delta_scale: float = 1.0,
        prior_stack_meta: dict[str, Any] | None = None,
    ) -> dict[str, int]:
        """Restore `saved` across both active trainables and frozen prior-stage adaptations."""
        if prior_stack_meta:
            if self.gdn2 is None and isinstance(prior_stack_meta.get("gdn2"), dict):
                self.carried_gdn2_cfg = dict(prior_stack_meta["gdn2"])
            if self.lsa is None and isinstance(prior_stack_meta.get("lsa"), dict):
                self.carried_lsa_cfg = dict(prior_stack_meta["lsa"])
            if self.deltas is None and prior_stack_meta.get("expert_delta_scale") is not None:
                expert_delta_scale = float(prior_stack_meta["expert_delta_scale"])
        if strict:
            missing = [n for n, p in self.named_trainable.items() if p.requires_grad and n not in saved]
            if missing:
                raise RuntimeError(
                    f"checkpoint mismatch: missing={missing[:10]} (+{max(0, len(missing) - 10)}). "
                    "Did the loop/LSA/expert config change? use strict=false to load partially."
                )
        lookup = self.named_all or self.named_trainable
        n_train = 0
        with torch.no_grad():
            for n, t in saved.items():
                if n in lookup:
                    p = lookup[n]
                    if p.shape != t.shape:
                        raise RuntimeError(
                            f"shape mismatch for {n}: model {tuple(p.shape)} vs ckpt {tuple(t.shape)}"
                        )
                    p.copy_(t.to(p.device, p.dtype))
                    if n in self.named_trainable:
                        n_train += 1
                    else:
                        self.carried_params[n] = p
                else:
                    if strict and not self.named_all:
                        raise RuntimeError(f"checkpoint mismatch: unexpected key {n!r}")
                    self.passthrough_tensors[n] = t.detach().to("cpu")

        folded = fold_expert_deltas_state_into_base_(
            self.model,
            self.passthrough_tensors,
            scale=expert_delta_scale,
            skip=self._folded_experts,
        )
        self._folded_experts |= folded
        return {
            "trainable": n_train,
            "carried": len(self.carried_params),
            "passthrough": len(self.passthrough_tensors),
            "folded_experts": len(folded),
        }

    def describe(self) -> dict[str, Any]:
        return {
            "loop": None if self.loop is None else self.loop.spec.__dict__,
            "gdn2": (self.gdn2.cfg.__dict__ if self.gdn2 is not None else self.carried_gdn2_cfg),
            "lsa": (self.lsa.cfg.__dict__ if self.lsa is not None else self.carried_lsa_cfg),
            "hot_experts": None if self.hot_map is None else {str(k): v for k, v in self.hot_map.items()},
            "expert_delta_scale": None if self.deltas is None else self.deltas.scale,
            "n_trainable": sum(p.numel() for p in self.named_trainable.values()),
        }


def _hot_map(ecfg: dict[str, Any]) -> dict[int, list[int]] | None:
    from ..hub import resolve_run_path

    hm_path = resolve_run_path(ecfg.get("hot_map_path"))
    if hm_path:
        import json

        with open(hm_path, encoding="utf-8") as f:
            raw = json.load(f)
        return {int(k): [int(e) for e in v] for k, v in raw.items()}
    path = resolve_run_path(ecfg.get("stats_path"))
    require(bool(path) and os.path.exists(path), "experts.enabled needs experts.stats_path (run `qloop scan-experts`) "
            "or experts.hot_map_path", stats_path=path)
    assert path is not None
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

    gcfg = dict(cfg.get("gdn2") or {})
    if gcfg.pop("enabled", False):
        with stage("model:gdn2"):
            st.gdn2 = install_gdn2(model, gcfg)

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
    allowed_groups = list(tcfg["allowed_groups"]) if tcfg.get("allowed_groups") is not None else None
    if st.gdn2 is not None and st.gdn2.adapters and tcfg.get("train_gdn2", True):
        patterns.append(r"gdn2_adapter\.")
        if allowed_groups is not None and "gdn2_adapters" not in allowed_groups:
            allowed_groups.append("gdn2_adapters")
    if st.lsa is not None and tcfg.get("train_lsa", True):
        patterns.append(r"lsa_indexer\.")
    if st.deltas is not None and tcfg.get("train_expert_deltas", True):
        patterns.append(r"expert_deltas\.")
    with stage("model:freeze"):
        if patterns:
            report = freeze_all_but(model, patterns, allowed_groups=allowed_groups,
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
    require(all(i in names for i in params), "a parameter has no stable name (internal error)")
    st.named_all = {names[i]: p for i, p in params.items()}
    st.named_trainable = {names[i]: p for i, p in params.items() if p.requires_grad}
    n_tr = sum(p.numel() for p in st.named_trainable.values())
    log.info("trainable tensors: %d (%.3fM params)", len(st.named_trainable), n_tr / 1e6)
    if int((cfg.get("debug") or {}).get("level", 1)) >= 2 or (cfg.get("debug") or {}).get("nan_hooks"):
        register_nonfinite_hooks(model)
    return st
