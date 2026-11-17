"""Regression tests for Phase-1 HSSD distillation OOM fixes (`distill.py` & `s1b_hssd_distill.yaml`)."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest
import torch

from qloop.config import load_config
from qloop.data.tokenizer import ToyTokenizer
from qloop.model.hssd import HSSDConfig, apply_hssd
from qloop.model.introspect import inspect_model
from qloop.model.loop import LoopSpec, apply_loop
from qloop.model.toy import ToyConfig, ToyForCausalLM
from qloop.train.distill import SharedBackboneTeacher, run_layerwise_warmup_microstep
from qloop.train.optim import FusedStepOptimizer
from qloop.train.stack import Stack

ROOT = Path(__file__).resolve().parents[1]


def _make_toy_distill_stack(*, outer_checkpoint_max_tokens: int = 32, checkpoint_segments: bool = True) -> Stack:
    torch.manual_seed(2026)
    tok = ToyTokenizer()
    tcfg = ToyConfig(
        vocab_size=tok.vocab_size,
        hidden_size=32,
        moe_intermediate_size=16,
        shared_expert_intermediate_size=16,
        num_hidden_layers=4,
        layer_types=("linear_attention", "full_attention", "linear_attention", "linear_attention"),
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        linear_num_heads=4,
        linear_head_dim=8,
        num_experts=4,
        num_experts_per_tok=2,
    )
    model = ToyForCausalLM(tcfg)
    hssd = apply_hssd(
        model,
        HSSDConfig(
            rank=8,
            chunk_size=8,
            window_chunks=2,
            num_pages=4,
            k_pages=2,
            segment_chunks=2,
            checkpoint_segments=checkpoint_segments,
            outer_checkpoint_max_tokens=outer_checkpoint_max_tokens,
        ),
    )
    loop = apply_loop(
        model,
        LoopSpec(
            prelude_end=1,
            core_end=3,
            num_passes=2,
            target_virtual_layers=6,
            checkpoint=True,
        ),
    )
    for p in model.parameters():
        p.requires_grad_(False)
    for p in hssd.parameters():
        p.requires_grad_(True)
    return Stack(
        model=model,
        tokenizer=tok,
        parts=inspect_model(model),
        loop=loop,
        hssd=hssd,
        lsa=None,
        deltas=None,
        load_info={"name": "toy"},
        hot_map={},
    )


def test_layerwise_warmup_preserves_checkpoint_segments_above_max_tokens(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify `run_layerwise_warmup_microstep` keeps `checkpoint_segments=True` above `outer_checkpoint_max_tokens`."""
    stack = _make_toy_distill_stack(outer_checkpoint_max_tokens=32, checkpoint_segments=True)
    assert stack.hssd is not None
    shared_teacher = SharedBackboneTeacher(stack, teacher_ckpt=None)
    trainable = [(n, p) for n, p in stack.model.named_parameters() if p.requires_grad]
    opt = FusedStepOptimizer.from_config(
        trainable,
        {"lr": 1e-3, "fused": False, "impl": "builtin", "apollo": {"enabled": False}},
        grad_accum=1,
    )

    import qloop.model.hssd as hssd_mod

    observed_flags: list[bool] = []
    orig_chunk_hssd = hssd_mod.chunk_hssd

    def _spy_chunk_hssd(*args: Any, **kwargs: Any) -> Any:
        observed_flags.append(bool(kwargs.get("checkpoint_segments", True)))
        return orig_chunk_hssd(*args, **kwargs)

    monkeypatch.setattr(hssd_mod, "chunk_hssd", _spy_chunk_hssd)

    # 1. Long sequence: input_ids.numel() = 2 * 32 = 64 > outer_checkpoint_max_tokens (32) -> stays True!
    ids_long = torch.randint(0, 64, (2, 32))
    observed_flags.clear()
    opt.set_boundary(True)
    metrics_long = run_layerwise_warmup_microstep(stack, shared_teacher, opt, ids_long, depth=2, grad_accum=1)
    assert metrics_long["distill/calls"] == 4.0
    assert observed_flags == [True] * 4, f"Expected checkpoint_segments=True for 64 > 32 tokens, got {observed_flags}"
    assert stack.hssd.cfg.checkpoint_segments is True

    # 2. Short sequence: input_ids.numel() = 2 * 16 = 32 <= outer_checkpoint_max_tokens (32) -> skips inner ckpt (False)
    #    and restores orig_ckpt_seg (True) in finally!
    ids_short = torch.randint(0, 64, (2, 16))
    observed_flags.clear()
    opt.set_boundary(True)
    metrics_short = run_layerwise_warmup_microstep(stack, shared_teacher, opt, ids_short, depth=2, grad_accum=1)
    assert metrics_short["distill/calls"] == 4.0
    assert observed_flags == [False] * 4, (
        f"Expected checkpoint_segments=False for 32 <= 32 tokens, got {observed_flags}"
    )
    assert stack.hssd.cfg.checkpoint_segments is True

    # 3. Exception inside backbone still restores orig_ckpt_seg in `finally:`
    def _boom(*_a: Any, **_kw: Any) -> Any:
        raise RuntimeError("synthetic failure")

    monkeypatch.setattr(hssd_mod, "chunk_hssd", _boom)
    with pytest.raises(RuntimeError, match="synthetic failure"):
        run_layerwise_warmup_microstep(stack, shared_teacher, opt, ids_short, depth=2, grad_accum=1)
    assert stack.hssd.cfg.checkpoint_segments is True


def test_s1b_hssd_distill_yaml_micro_batch_1_grad_accum_4_budget() -> None:
    """Verify `s1b_hssd_distill.yaml` uses `micro_batch=1, grad_accum=4` (`65,536` tok/step, `< 80 GiB` on G4)."""
    cfg = load_config(os.path.join(ROOT, "configs", "stages", "s1b_hssd_distill.yaml"))
    assert cfg["gdn2"]["enabled"] is False
    assert cfg["hssd"]["enabled"] is True
    assert cfg["train"]["micro_batch"] == 1
    assert cfg["train"]["grad_accum"] == 4
    assert cfg["data"]["seq_len"] == 16384
    assert cfg["train"]["micro_batch"] * cfg["train"]["grad_accum"] * cfg["data"]["seq_len"] == 65536
