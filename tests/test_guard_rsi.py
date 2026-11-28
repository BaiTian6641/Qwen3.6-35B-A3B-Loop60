"""Comprehensive CPU unit and end-to-end tests for Google GUARD (arXiv:2611.23476) G1, G2, and G3."""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

import pytest
import torch

from qloop.config import load_config
from qloop.eval.generate import Generator, GenOut
from qloop.eval.runners import run_benchmark
from qloop.eval.testing import ByteTokenizer, ScriptedLM
from qloop.model.experts import param_group_of
from qloop.train.rl import GRPOTrainer, RLTask, TaskSampler, _Seq, group_advantages
from qloop.train.sirsi import (
    REGRET_NEWLY_HURT,
    REGRET_PINNED_PASS,
    REGRET_UNSOLVED_CHRONIC,
    REGRET_UNSOLVED_NOVEL,
    TAG_LOOP_DEGENERATION,
    TAG_UNCLOSED_THINK,
    FailureLedger,
    SIRSIConfig,
    SIRSIPathology,
    TaskRegretTracker,
    build_curriculum_sculpt_messages,
    build_hint_messages,
    detect_pathologies,
    evaluate_tier_a_checks,
    unlikelihood_span_loss,
)

ROOT = Path(__file__).resolve().parent.parent


def _guard_toy_cfg(
    *,
    steps: int = 1,
    gate: dict[str, Any] | None = None,
    sirsi: dict[str, Any] | None = None,
    tasks: list[dict[str, Any]] | None = None,
    group_size: int = 2,
    prompts_per_step: int = 1,
    fused: bool = False,
    teacher: dict[str, Any] | None = None,
    overlong: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "seed": 0,
        "stage": "guard-test",
        "model": {"name": "toy", "dtype": "float32", "device_map": "cpu", "toy": {"seed": 0}},
        "loop": {"enabled": True, "prelude_end": 4, "core_end": 12, "num_passes": 2, "target_virtual_layers": 24},
        "train": {
            "steps": steps,
            "ckpt_every": 10,
            "train_loop_adapters": True,
            "trainable": [r"mlp\.gate\.weight$", r"input_layernorm\.weight$"],
            "optim": {
                "lr": 1e-2,
                "fused": fused,
                "max_grad_norm": 0.5,
                "apollo": {"min_dim": 64},
            },
        },
        "rl": {
            "tasks": tasks
            or [
                {
                    "source": "inline",
                    "name": "m",
                    "verifier": "scorer:math",
                    "items": [{"id": "t0", "prompt": "Compute 6 * 7.", "answer": "42"}],
                }
            ],
            "group_size": group_size,
            "prompts_per_step": prompts_per_step,
            "max_new_tokens": 16,
            "temperature": 1.0,
            "micro_batch_seqs": 2,
            "ppo_epochs": 1,
            "overlong": overlong or {"enabled": False},
            "teacher": teacher or {},
            "sirsi": sirsi
            if sirsi is not None
            else {
                "enabled": True,
                "slap_weight": 0.2,
                "slap_eps": 1e-4,
                "slap_max_seqs_per_step": 4,
                "slap_target_groups": ["loop_adapters", "router"],
                "anti_slop_weight": 0.05,
                "failure_ledger": True,
                "regret_enabled": True,
                "atomic_checks_enabled": False,
            },
            "gate": gate or {},
        },
    }


def _fake_bench_summary(
    name: str,
    item_scores: dict[str, float],
    mean_len: float = 100.0,
    *,
    include_items: bool = True,
) -> dict[str, Any]:
    mean_sc = sum(item_scores.values()) / max(len(item_scores), 1)
    depth_entry: dict[str, Any] = {
        "n_items": len(item_scores),
        "mean_score": mean_sc,
        "mean_completion_tokens": mean_len,
    }
    if include_items:
        depth_entry["item_scores"] = dict(item_scores)
    return {"benchmark": name, "depths": {"2": depth_entry}}


# ============================================================================ G1: Item-Level Soft-HURT Gate


def test_g1_run_benchmark_attaches_item_scores(tmp_path: Path) -> None:
    gen = Generator(ScriptedLM.from_text(r"The answer is \boxed{42}."), ByteTokenizer())
    rows = [
        {"problem_idx": 1, "problem": "6*7?", "answer": "42"},
        {"problem_idx": 2, "problem": "40+2?", "answer": "42"},
        {"problem_idx": 3, "problem": "3+4?", "answer": "7"},
    ]
    summ = run_benchmark(
        "aime_2026",
        gen,
        depths=[1],
        max_new_tokens=32,
        out_dir=str(tmp_path),
        offline_items=rows,
    )
    d = summ["depths"]["native"]
    assert "item_scores" in d
    assert d["item_scores"] == {"1": 1.0, "2": 1.0, "3": 0.0}
    disk = json.loads((tmp_path / "aime_2026" / "summary.json").read_text(encoding="utf-8"))
    assert disk["depths"]["native"]["item_scores"] == {"1": 1.0, "2": 1.0, "3": 0.0}


def test_g1_soft_hurt_gate_decisions_clamping_firewall_and_rollback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = str(tmp_path / "g1_soft_hurt")
    secret_prompt = "SECRET_HELD_OUT_PROMPT_NEVER_LEAK_XYZ"
    secret_completion = "SECRET_HELD_OUT_COMPLETION_NEVER_LEAK_ABC"

    gate_cfg = {
        "mode": "soft_hurt",
        "eval_at_step_0": True,
        "every": 1,
        "depth": 2,
        "lambda_hurt": 2.0,
        "min_soft_hurt": 0.0,
        "max_hurt_frac": 0.05,
        "max_bench_drop": 0.025,
        "max_len_growth": 1.5,
        "rollback": True,
        "benchmarks": [
            {"name": "bench_a", "weight": 1.0},
            {"name": "bench_b", "weight": 1.0},
        ],
    }
    tr = GRPOTrainer(_guard_toy_cfg(steps=1, gate=gate_cfg), run_dir)
    try:
        # 10 items per bench -> 20 total items; 1 hurt item => hurt_frac = 0.05 (at boundary)
        base_a = {f"a{i}": (1.0 if i < 6 else 0.0) for i in range(10)}
        base_b = {f"b{i}": (1.0 if i < 6 else 0.0) for i in range(10)}

        # Call 0 (step 0): baseline -> promoted
        # Call 1 (step 1): 2 helped + 1 hurt in a => soft_hurt=0.0 (not > 0!) -> ROLLED BACK
        # Call 2 (step 2): speedup_bonus=1.5 on a0 clamped to 1.0 -> cannot mask a1 drop -> ROLLED BACK
        # Call 3 (step 3): 4 helped in a, 1 hurt in b => bench_b drops 0.10 > 0.025 -> ROLLED BACK
        # Call 4 (step 4): 3 helped in a, 0 hurt => gain=0.15, hurt=0 -> PROMOTED
        # Call 5 (step 5): 4 helped in b + 2 hurt in a => hurt_frac=0.10 > 0.05 -> ROLLED BACK
        # Call 6 (step 6): fallback reading depth_2.jsonl when item_scores absent from summary
        scenarios = [
            {"bench_a": base_a, "bench_b": base_b},
            {"bench_a": {**base_a, "a0": 0.0, "a6": 1.0, "a7": 1.0}, "bench_b": base_b},
            {"bench_a": {**base_a, "a0": 1.5, "a1": 0.0}, "bench_b": base_b},
            {"bench_a": {**base_a, "a6": 1.0, "a7": 1.0, "a8": 1.0, "a9": 1.0}, "bench_b": {**base_b, "b0": 0.0}},
            {"bench_a": {**base_a, "a6": 1.0, "a7": 1.0, "a8": 1.0}, "bench_b": base_b},
            {
                "bench_a": {**base_a, "a6": 1.0, "a7": 1.0, "a8": 1.0, "a0": 0.0, "a1": 0.0},
                "bench_b": {**base_b, "b6": 1.0, "b7": 1.0, "b8": 1.0, "b9": 1.0},
            },
            {"bench_a": {**base_a, "a6": 1.0, "a7": 1.0, "a8": 1.0}, "bench_b": {**base_b, "b6": 1.0}},
        ]
        call_idx = {"n": 0}

        def _fake_run_benchmark(name: str, gen: Any, **kw: Any) -> dict[str, Any]:
            idx = call_idx["n"]
            sc = scenarios[idx][name]
            include_items = idx != 6
            if not include_items:
                bdir = Path(kw["out_dir"]) / name
                bdir.mkdir(parents=True, exist_ok=True)
                with open(bdir / "depth_2.jsonl", "w", encoding="utf-8") as f:
                    for item_id, val in sc.items():
                        f.write(
                            json.dumps(
                                {
                                    "id": item_id,
                                    "mean_score": val,
                                    "prompt": secret_prompt,
                                    "samples": [{"text": secret_completion, "score": val}],
                                }
                            )
                            + "\n"
                        )
            return _fake_bench_summary(name, sc, mean_len=100.0, include_items=include_items)

        monkeypatch.setattr("qloop.eval.runners.run_benchmark", _fake_run_benchmark)

        restore_calls: list[dict[str, Any]] = []
        orig_restore = tr.stack.restore_checkpoint

        def _spy_restore(params: Any, **kw: Any) -> Any:
            restore_calls.append(dict(kw))
            return orig_restore(params, **kw)

        monkeypatch.setattr(tr.stack, "restore_checkpoint", _spy_restore)

        # Baseline (step 0)
        g0 = tr.gate()
        assert g0["gate/decision"] == "promoted"
        assert g0["gate/n_pinned_passed"] == 12.0
        assert len(tr.best_gate_items) == 20
        champ0_params = {n: p.detach().clone() for n, p in tr.stack.named_trainable.items()}

        # Mutate params before rejected step 1 to prove rollback restores weights via restore_checkpoint
        with torch.no_grad():
            for p in tr.stack.named_trainable.values():
                p.add_(1.234)
        tr.step = 1
        call_idx["n"] = 1
        g1 = tr.gate()
        assert g1["gate/decision"].startswith("rolled back to")
        assert g1["gate/soft_hurt"] == pytest.approx(0.0)
        assert g1["gate/n_newly_helped"] == 2.0 and g1["gate/n_newly_hurt"] == 1.0
        assert len(restore_calls) == 1
        assert restore_calls[-1]["strict"] is False
        assert "expert_delta_scale" in restore_calls[-1] and "prior_stack_meta" in restore_calls[-1]
        for n, p0 in champ0_params.items():
            assert torch.allclose(tr.stack.named_trainable[n].detach(), p0, atol=1e-6)

        # Step 2: speedup bonus > 1.0 clamped so cannot mask correctness drop
        tr.step = 2
        call_idx["n"] = 2
        g2 = tr.gate()
        assert g2["gate/decision"].startswith("rolled back to")
        assert g2["gate/gain"] == pytest.approx(0.0)
        assert g2["gate/hurt"] > 0.0
        assert g2["gate/soft_hurt"] < 0.0

        # Step 3: per-benchmark drop cap blocks promotion even when overall gain > 2*hurt
        tr.step = 3
        call_idx["n"] = 3
        g3 = tr.gate()
        assert g3["gate/decision"].startswith("rolled back to")
        assert g3["gate/soft_hurt"] > 0.0

        # Step 4: clean improvement -> promoted
        tr.step = 4
        call_idx["n"] = 4
        g4 = tr.gate()
        assert g4["gate/decision"] == "promoted"
        assert g4["gate/gain"] == pytest.approx(0.15)
        assert g4["gate/hurt"] == pytest.approx(0.0)
        assert g4["gate/n_pinned_passed"] == 15.0

        # Step 5: hurt_frac=0.10 > max_hurt_frac=0.05 -> rolled back despite positive gain
        tr.step = 5
        call_idx["n"] = 5
        g5 = tr.gate()
        assert g5["gate/decision"].startswith("rolled back to")
        assert g5["gate/hurt_frac"] == pytest.approx(0.10)

        # Step 6: fallback reading depth_2.jsonl + strict contamination firewall check
        tr.step = 6
        call_idx["n"] = 6
        g6 = tr.gate()
        assert g6["gate/decision"] == "promoted"
        assert g6["gate/n_newly_helped"] == 1.0
        assert g6["gate/n_pinned_passed"] == 16.0
        diff_path = Path(run_dir) / "gate" / "step_0000006" / "soft_hurt_diff.json"
        raw_diff = diff_path.read_text(encoding="utf-8")
        assert secret_prompt not in raw_diff
        assert secret_completion not in raw_diff
        diff_obj = json.loads(raw_diff)
        assert diff_obj["newly_helped"] == ["bench_b::b6"]
        assert diff_obj["newly_hurt"] == []
        # Gate never writes held-out items to FailureLedger
        assert not (Path(run_dir) / "sirsi_failures.jsonl").exists()
    finally:
        tr.pool.shutdown(wait=False)


def test_g1_eval_at_step_0_and_resume_roundtrip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run_dir = str(tmp_path / "g1_step0_resume")
    gate_cfg = {
        "mode": "soft_hurt",
        "eval_at_step_0": True,
        "every": 0,
        "depth": 2,
        "benchmarks": [{"name": "bench_a"}],
    }
    cfg = _guard_toy_cfg(steps=1, gate=gate_cfg)
    tr = GRPOTrainer(cfg, run_dir)
    gate_steps_seen: list[int] = []

    def _fake_run_benchmark(name: str, gen: Any, **kw: Any) -> dict[str, Any]:
        gate_steps_seen.append(tr.step)
        return _fake_bench_summary(name, {"i0": 1.0, "i1": 0.0})

    def _fake_gen(prompts: list[Any], **kw: Any) -> list[list[GenOut]]:
        return [
            [GenOut("The answer is \\boxed{42}.", [20, 21, 22]), GenOut("wrong", [20, 21, 23])]
            for _ in prompts
        ]

    monkeypatch.setattr("qloop.eval.runners.run_benchmark", _fake_run_benchmark)
    monkeypatch.setattr(tr.gen, "generate", _fake_gen)
    monkeypatch.setattr(tr.gen, "encode_prompt", lambda prompt, **kw: [10, 11, 12])

    res = tr.train()
    assert res["step"] == 1
    assert gate_steps_seen == [0], "eval_at_step_0 must fire once at step 0 before step 1"
    assert tr.best_gate_items == {"bench_a::i0": 1.0, "bench_a::i1": 0.0}
    assert tr.pinned_passed == {"bench_a::i0"}

    # Resume into step 2: restore best_gate_items, best_bench_scores, pinned_passed, regret_tracker, sampler
    cfg2 = _guard_toy_cfg(steps=2, gate=gate_cfg)
    tr2 = GRPOTrainer(cfg2, run_dir)
    gate_calls_resumed: list[int] = []

    def _fake_run_bench_2(name: str, gen: Any, **kw: Any) -> dict[str, Any]:
        gate_calls_resumed.append(tr2.step)
        return _fake_bench_summary(name, {"i0": 1.0, "i1": 0.0})

    monkeypatch.setattr("qloop.eval.runners.run_benchmark", _fake_run_bench_2)
    monkeypatch.setattr(tr2.gen, "generate", _fake_gen)
    monkeypatch.setattr(tr2.gen, "encode_prompt", lambda prompt, **kw: [10, 11, 12])
    res2 = tr2.train()
    assert res2["step"] == 2
    assert gate_calls_resumed == [], "resumed run must not re-run eval_at_step_0"
    assert tr2.best_gate_items == {"bench_a::i0": 1.0, "bench_a::i1": 0.0}
    assert tr2.best_bench_scores == {"bench_a": 0.5}
    assert tr2.pinned_passed == {"bench_a::i0"}
    assert "t0" in tr2.regret_tracker.states and tr2.regret_tracker.states["t0"].ever_passed is True
    assert [t.task_id for t in tr2.sampler.pinned_anchor_pool] == ["t0"]


def test_g1_gate_weighted_score_normalization_and_step0_legacy_resume(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify weighted `gate()` score normalization, legacy `best_gate` reconciliation, and step-0 resume."""
    run_dir = str(tmp_path / "g1_weighted_resume")
    gate_cfg = {
        "mode": "soft_hurt",
        "eval_at_step_0": True,
        "every": 15,
        "depth": 2,
        "benchmarks": [
            {"name": "aime_2026", "weight": 0.5},
            {"name": "gpqa_diamond", "weight": 0.5},
        ],
    }
    tr = GRPOTrainer(_guard_toy_cfg(steps=1, gate=gate_cfg), run_dir)
    try:
        bench_scores = {"aime_2026": 0.2333, "gpqa_diamond": 0.42}
        bench_items = {
            "aime_2026": {"a0": 1.0, "a1": 0.0},
            "gpqa_diamond": {"g0": 1.0, "g1": 0.0},
        }

        def _fake_run_benchmark(name: str, gen: Any, **kw: Any) -> dict[str, Any]:
            return {
                "benchmark": name,
                "depths": {
                    "2": {
                        "n_items": len(bench_items[name]),
                        "mean_score": bench_scores[name],
                        "mean_completion_tokens": 120.0,
                        "item_scores": dict(bench_items[name]),
                    }
                },
            }

        monkeypatch.setattr("qloop.eval.runners.run_benchmark", _fake_run_benchmark)
        g0 = tr.gate()
        expected = 0.5 * 0.2333 + 0.5 * 0.42  # 0.32665, NOT 0.163325!
        assert g0["gate/decision"] == "promoted"
        assert g0["gate/score"] == pytest.approx(expected, abs=1e-9)
        assert tr.best_gate == pytest.approx(expected, abs=1e-9)
        assert tr.best_bench_scores == bench_scores
        assert len(tr.best_gate_items) == 4
    finally:
        tr.pool.shutdown(wait=False)

    # Corrupt saved `step_0000000/state.json` to mimic legacy `/ len(scores)` best_gate (0.163325)
    state_path = Path(run_dir) / "checkpoints" / "step_0000000" / "state.json"
    state_obj = json.loads(state_path.read_text(encoding="utf-8"))
    assert state_obj["step"] == 0
    assert state_obj["best_gate_items"]
    state_obj["best_gate"] = 0.163325
    state_path.write_text(json.dumps(state_obj, indent=2), encoding="utf-8")

    # Resume from step_0000000: must reconcile best_gate -> 0.32665, skip eval_at_step_0, and start at Step 1
    tr2 = GRPOTrainer(_guard_toy_cfg(steps=1, gate=gate_cfg), run_dir)
    gate_calls: list[int] = []
    steps_seen: list[int] = []

    def _fail_if_gated(name: str, gen: Any, **kw: Any) -> dict[str, Any]:
        gate_calls.append(tr2.step)
        raise AssertionError("eval_at_step_0 must be skipped when resuming step 0 with non-empty best_gate_items")

    def _fake_gen(prompts: list[Any], **kw: Any) -> list[list[GenOut]]:
        return [
            [GenOut("The answer is \\boxed{42}.", [20, 21, 22]), GenOut("wrong", [20, 21, 23])]
            for _ in prompts
        ]

    monkeypatch.setattr("qloop.eval.runners.run_benchmark", _fail_if_gated)
    monkeypatch.setattr(tr2.gen, "generate", _fake_gen)
    monkeypatch.setattr(tr2.gen, "encode_prompt", lambda prompt, **kw: [10, 11, 12])

    res = tr2.train(on_step=lambda step, _m: steps_seen.append(step))
    assert gate_calls == []
    assert steps_seen == [1]
    assert res["step"] == 1
    assert tr2.best_gate == pytest.approx(expected, abs=1e-9)
    assert tr2.best_bench_scores == bench_scores
    assert len(tr2.best_gate_items) == 4


def test_g1_aggregate_rollback_uses_restore_checkpoint(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run_dir = str(tmp_path / "g1_agg_rollback")
    cfg = _guard_toy_cfg(
        steps=1,
        gate={"mode": "aggregate", "depth": 2, "rollback": True, "benchmarks": [{"name": "bench_a"}]},
    )
    tr = GRPOTrainer(cfg, run_dir)
    try:
        vals = iter([0.8, 0.2])
        monkeypatch.setattr(
            "qloop.eval.runners.run_benchmark",
            lambda name, gen, **kw: _fake_bench_summary(name, {"i0": next(vals)}),
        )
        restore_calls: list[dict[str, Any]] = []
        orig = tr.stack.restore_checkpoint

        def _spy(params: Any, **kw: Any) -> Any:
            restore_calls.append(dict(kw))
            return orig(params, **kw)

        monkeypatch.setattr(tr.stack, "restore_checkpoint", _spy)
        assert tr.gate()["gate/decision"] == "promoted"
        tr.step = 1
        assert tr.gate()["gate/decision"].startswith("rolled back to")
        assert len(restore_calls) == 1
        assert restore_calls[0]["strict"] is False
        assert "expert_delta_scale" in restore_calls[0]
        assert "prior_stack_meta" in restore_calls[0]
    finally:
        tr.pool.shutdown(wait=False)


# ============================================================================ G2: Directional Regret Weighting


def test_g2_task_regret_tracker_transitions_and_serialization() -> None:
    cfg = SIRSIConfig(
        regret_drop_eps=0.05,
        regret_hurt_gamma=2.0,
        regret_max_weight=3.0,
        regret_chronic_weight=0.5,
    )
    tracker = TaskRegretTracker(cfg)

    # Novel -> Chronic on repeated failure without ever passing
    d1 = tracker.classify_and_update("t_hard", 0.0)
    assert d1.kind == REGRET_UNSOLVED_NOVEL and d1.weight == pytest.approx(1.0)
    d2 = tracker.classify_and_update("t_hard", 0.0)
    assert d2.kind == REGRET_UNSOLVED_CHRONIC and d2.weight == pytest.approx(0.5)
    d3 = tracker.classify_and_update("t_hard", 0.0)
    assert d3.kind == REGRET_UNSOLVED_CHRONIC and d3.weight == pytest.approx(0.5)

    # Pass -> Pinned -> Drop => NEWLY_HURT
    p1 = tracker.classify_and_update("t_med", 0.75)
    assert p1.kind == REGRET_PINNED_PASS and p1.weight == pytest.approx(1.0)
    p2 = tracker.classify_and_update("t_med", 0.72)  # within drop_eps=0.05
    assert p2.kind == REGRET_PINNED_PASS and p2.weight == pytest.approx(1.0)
    h1 = tracker.classify_and_update("t_med", 0.25)
    assert h1.kind == REGRET_NEWLY_HURT and h1.prev_pass_rate == pytest.approx(0.72)
    assert h1.weight == pytest.approx(1.0 + 2.0 * (0.75 - 0.25))  # 2.0
    # Complete collapse (best >= 0.5 and pr == 0.0) stays NEWLY_HURT with higher weight
    h2 = tracker.classify_and_update("t_med", 0.0)
    assert h2.kind == REGRET_NEWLY_HURT and h2.weight == pytest.approx(min(3.0, 1.0 + 2.0 * 0.75))
    h3 = tracker.classify_and_update("t_med", 0.0)
    assert h3.kind == REGRET_NEWLY_HURT and h3.weight == pytest.approx(2.5)

    # Check-bit regression (1 -> 0) triggers NEWLY_HURT even when ever_passed is False
    kb_names = ("extract", "ast", "compiles", "rt", "ok", "fast")
    c1 = tracker.classify_and_update("t_kb", 0.0, check_bits=(1, 1, 1, 0, 0, 0), check_names=kb_names)
    assert c1.kind == REGRET_UNSOLVED_NOVEL
    c2 = tracker.classify_and_update("t_kb", 0.0, check_bits=(1, 1, 0, 0, 0, 0), check_names=kb_names)
    assert c2.kind == REGRET_NEWLY_HURT
    assert c2.regressed_checks == ("compiles",)
    assert 1.0 < c2.weight <= 3.0

    # Round-trip serialization
    sd = tracker.state_dict()
    tracker2 = TaskRegretTracker(tracker.cfg)
    tracker2.load_state_dict(sd)
    assert tracker2.state_dict() == sd


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_g2_unlikelihood_span_loss_seq_weights(dtype: torch.dtype) -> None:
    lp = torch.tensor([[-0.5, -0.5, -0.5], [-0.5, -0.5, -0.5]], dtype=dtype, requires_grad=True)
    mask = torch.tensor([[False, True, True], [False, True, True]])

    # Exact legacy parity when seq_weights=None vs [1.0, 1.0]
    l_legacy = unlikelihood_span_loss(lp, mask, eps=1e-4)
    l_ones = unlikelihood_span_loss(lp, mask, eps=1e-4, seq_weights=[1.0, 1.0])
    assert float(l_legacy.detach()) == pytest.approx(float(l_ones.detach()), rel=1e-6)

    # [3.0, 0.5] produces exact 6:1 gradient ratio between seq 0 and seq 1, zero outside mask
    l_w = unlikelihood_span_loss(lp, mask, eps=1e-4, seq_weights=torch.tensor([3.0, 0.5]))
    assert torch.isfinite(l_w)
    l_w.backward()
    assert lp.grad is not None
    assert float(lp.grad[0, 0]) == 0.0 and float(lp.grad[1, 0]) == 0.0
    ratio = float(lp.grad[0, 1].float()) / float(lp.grad[1, 1].float())
    assert ratio == pytest.approx(6.0, rel=1e-2)

    # Single-seq scaling: weight=3.0 is 3x weight=1.0, weight=0.5 is 0.5x weight=1.0
    lp1 = torch.tensor([[0.0, -0.4]], dtype=dtype)
    m1 = torch.tensor([[True, True]])
    l1 = float(unlikelihood_span_loss(lp1, m1, seq_weights=[1.0]))
    l3 = float(unlikelihood_span_loss(lp1, m1, seq_weights=[3.0]))
    lh = float(unlikelihood_span_loss(lp1, m1, seq_weights=[0.5]))
    assert l3 == pytest.approx(3.0 * l1, rel=1e-5)
    assert lh == pytest.approx(0.5 * l1, rel=1e-5)


def test_g2_hint_and_curriculum_sculpt_stratification() -> None:
    # Legacy hint has no regression alert; NEWLY_HURT highlights regression clearly
    m_leg = build_hint_messages("Solve x+1=3", "x=5", [TAG_LOOP_DEGENERATION])
    assert "NEWLY_HURT" not in m_leg[1]["content"]
    m_hurt = build_hint_messages(
        "Solve x+1=3",
        "x=5",
        [TAG_LOOP_DEGENERATION],
        regret_kind=REGRET_NEWLY_HURT,
        prev_pass_rate=0.75,
    )
    assert "NEWLY_HURT" in m_hurt[1]["content"] and "0.75" in m_hurt[1]["content"]

    # Curriculum stratification: 5 recent NEWLY_HURT + 3 recent UNSOLVED_NOVEL (excluding chronic)
    recs: list[dict[str, Any]] = []
    for i in range(6):
        recs.append({
            "task_id": f"hurt_{i}",
            "tags": [TAG_LOOP_DEGENERATION],
            "reward": 0.0,
            "prompt": f"H{i}",
            "regret_kind": REGRET_NEWLY_HURT,
        })
    for i in range(4):
        recs.append({
            "task_id": f"novel_{i}",
            "tags": ["ALL_FAIL"],
            "reward": 0.0,
            "prompt": f"N{i}",
            "regret_kind": REGRET_UNSOLVED_NOVEL,
        })
    for i in range(4):
        recs.append({
            "task_id": f"chronic_{i}",
            "tags": ["ALL_FAIL"],
            "reward": 0.0,
            "prompt": f"C{i}",
            "regret_kind": REGRET_UNSOLVED_CHRONIC,
        })

    msgs = build_curriculum_sculpt_messages(recs, n_tasks=4)
    body = msgs[1]["content"]
    for i in range(1, 6):
        assert f"task_id=hurt_{i}" in body
    assert "task_id=hurt_0" not in body
    for i in range(1, 4):
        assert f"task_id=novel_{i}" in body
    assert "task_id=novel_0" not in body
    assert "task_id=chronic_" not in body

    # Fallback to [-8:] when records lack regret_kind
    legacy_recs = [
        {"task_id": f"leg_{i}", "tags": ["ALL_FAIL"], "reward": 0.0, "prompt": f"P{i}"}
        for i in range(12)
    ]
    leg_body = build_curriculum_sculpt_messages(legacy_recs, n_tasks=2)[1]["content"]
    assert "task_id=leg_3" not in leg_body and "task_id=leg_4" in leg_body and "task_id=leg_11" in leg_body


def test_g2_task_sampler_replay_queues_and_resume() -> None:
    tasks = [RLTask(f"t{i}", f"p{i}", "regex", {"pattern": "x"}, "g") for i in range(10)]
    s = TaskSampler([("g", 1.0, tasks)], seed=7)
    # Default probabilities (0.0) ignore queues and match pure legacy draws
    s.record_hurt(tasks[9])
    s.record_pinned(tasks[8])
    s_ref = TaskSampler([("g", 1.0, tasks)], seed=7)
    assert [t.task_id for t in s.next(4)] == [t.task_id for t in s_ref.next(4)]

    # Non-zero probabilities draw from hurt_queue and pinned_anchor_pool and survive state_dict round-trip
    s.record_hurt(tasks[5])
    sd = s.state_dict()
    assert sd["hurt_queue"] == ["t9", "t5"]
    assert sd["pinned_anchor_pool"] == ["t8"]
    s2 = TaskSampler([("g", 1.0, tasks)], seed=7)
    s2.load_state_dict(sd)
    got_h = s2.next(2, hurt_prob=1.0, pinned_prob=0.0)
    assert [t.task_id for t in got_h] == ["t9", "t5"]
    got_p = s2.next(2, hurt_prob=0.0, pinned_prob=1.0)
    assert [t.task_id for t in got_p] == ["t8", "t8"]


def test_g2_rollout_priority_sorting_and_weighted_surgical_update(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = str(tmp_path / "g2_prio")
    items = [
        {"id": "t_chronic", "prompt": "Compute 1+1", "answer": "2"},
        {"id": "t_hurt", "prompt": "Compute 2+2", "answer": "4"},
    ]
    cfg = _guard_toy_cfg(
        steps=1,
        tasks=[{"source": "inline", "name": "m", "verifier": "scorer:math", "items": items}],
        group_size=1,
        prompts_per_step=2,
        teacher={"base_url": "http://mock.local/v1", "model": "m", "max_calls_per_step": 0},
        sirsi={
            "enabled": True,
            "slap_weight": 0.2,
            "slap_eps": 1e-4,
            "slap_max_seqs_per_step": 1,
            "slap_target_groups": ["loop_adapters", "router"],
            "guided_hint_enabled": True,
            "max_hints_per_step": 1,
            "failure_ledger": True,
            "regret_enabled": True,
            "hurt_replay_prob": 0.0,
            "pinned_replay_prob": 0.0,
        },
    )
    tr = GRPOTrainer(cfg, run_dir)
    try:
        # Pre-seed mastery: t_chronic failed once (next fail -> CHRONIC 0.5),
        # t_hurt passed with 1.0 (next fail -> NEWLY_HURT 3.0).
        tr.regret_tracker.classify_and_update("t_chronic", 0.0)
        tr.regret_tracker.classify_and_update("t_hurt", 1.0)

        # Force sampler order: t_chronic FIRST, t_hurt SECOND!
        t_ch = RLTask("t_chronic", "Compute 1+1", "scorer:math", items[0], "m")
        t_hu = RLTask("t_hurt", "Compute 2+2", "scorer:math", items[1], "m")
        monkeypatch.setattr(tr.sampler, "next", lambda k, **kw: [t_ch, t_hu])

        loop_text = "SETUP STEP. " + "CHECK BOUNDARY VALUE AGAIN NOW PLEASE DO IT. " * 3
        loop_ids = [min(b, 95) for b in tr.stack.tokenizer.encode(loop_text)][:40]
        hinted_prompts: list[str] = []

        def _fake_gen(prompts: list[Any], **kw: Any) -> list[list[GenOut]]:
            if len(prompts) == 1 and "[Supervisor Hint]:" in str(prompts[0]):
                return [[GenOut("The answer is \\boxed{4}", [30, 31, 32])]]
            return [[GenOut(loop_text, loop_ids)] for _ in prompts]

        def _fake_enc(prompt: Any, **kw: Any) -> list[int]:
            return [10, 11] if "1+1" in str(prompt) else [20, 21]

        def _fake_post(
            url: str,
            payload: dict[str, Any],
            timeout: float,
            headers: dict[str, str] | None = None,
            retries: int = 3,
        ) -> dict[str, Any]:
            hinted_prompts.append(payload["messages"][1]["content"])
            return {
                "choices": [{"message": {"content": "Re-evaluate 2+2 directly."}}],
                "usage": {"total_tokens": 12},
            }

        monkeypatch.setattr(tr.gen, "generate", _fake_gen)
        monkeypatch.setattr(tr.gen, "encode_prompt", _fake_enc)
        monkeypatch.setattr("qloop.train.rl._post_json", _fake_post)

        seqs, rstats = tr.rollout()
        # t_hurt (NEWLY_HURT) MUST win BOTH the 1-slot hint budget AND 1-slot slap budget
        assert rstats["regret/newly_hurt"] == 1.0
        assert rstats["regret/unsolved_chronic"] == 1.0
        assert len(hinted_prompts) == 1
        assert "Compute 2+2" in hinted_prompts[0] and "NEWLY_HURT" in hinted_prompts[0]
        slap_seqs = [s for s in seqs if s.kind == "slap_neg"]
        assert len(slap_seqs) == 1
        assert slap_seqs[0].regret_kind == REGRET_NEWLY_HURT
        assert slap_seqs[0].regret_weight == pytest.approx(3.0)
        assert slap_seqs[0].ids[:2] == [20, 21]
        assert [t.task_id for t in tr.sampler.hurt_queue] == ["t_hurt"]

        # FailureLedger recorded regret metadata
        rows = FailureLedger(os.path.join(run_dir, "sirsi_failures.jsonl")).read_recent(limit=10)
        by_tid = {r["task_id"]: r for r in rows}
        assert by_tid["t_hurt"]["regret_kind"] == REGRET_NEWLY_HURT
        assert by_tid["t_hurt"]["prev_pass_rate"] == pytest.approx(1.0)
        assert by_tid["t_hurt"]["regret_weight"] == pytest.approx(3.0)
        assert by_tid["t_chronic"]["regret_kind"] == REGRET_UNSOLVED_CHRONIC
        assert by_tid["t_chronic"]["regret_weight"] == pytest.approx(0.5)

        # Weighted update preserves surgical slap isolation and micro-batch invariance
        assert tr.stack.loop is not None
        tr.stack.loop.set_active_passes(2)
        nontargeted = [
            n for n in tr.stack.named_trainable if param_group_of(n) not in ("loop_adapters", "router")
        ]
        before = {n: tr.stack.named_trainable[n].detach().clone() for n in nontargeted}

        s_hi = _Seq(
            [10, 11, 12, 13, 14, 15],
            2,
            -1.0,
            "slap_neg",
            span_mask=[False, True, True, True],
            regret_weight=3.0,
            regret_kind=REGRET_NEWLY_HURT,
        )
        s_lo = _Seq(
            [10, 11, 12, 13, 14, 15],
            2,
            -1.0,
            "slap_neg",
            span_mask=[False, True, True, True],
            regret_weight=0.5,
            regret_kind=REGRET_UNSOLVED_CHRONIC,
        )
        tr.rl["micro_batch_seqs"] = 1
        init_all = {n: p.detach().clone() for n, p in tr.stack.named_trainable.items()}
        st_mb1 = tr.update([s_hi, s_lo])
        for n in nontargeted:
            assert torch.equal(tr.stack.named_trainable[n].detach(), before[n])
        with torch.no_grad():
            for n, p0 in init_all.items():
                tr.stack.named_trainable[n].copy_(p0)
        tr.rl["micro_batch_seqs"] = 2
        st_mb2 = tr.update([s_hi, s_lo])
        assert st_mb1["loss/slap_neg"] == pytest.approx(st_mb2["loss/slap_neg"], rel=1e-5)
    finally:
        tr.pool.shutdown(wait=False)


# ============================================================================ G3: Tier-A Atomic Checks + std-Trap Fix


def test_g3_tier_a_checks_kernelbench_math_http_and_anti_hacking() -> None:
    # 1. KernelBench checks + AST anti-cheat + runtime-error distinction
    good_kb = (
        "```python\n"
        "import torch\n"
        "class ModelNew(torch.nn.Module):\n"
        "    def forward(self, x):\n"
        "        return x * 2\n"
        "```"
    )
    cheat_kb = "```python\nfrom kb_reference import Model\nclass ModelNew(Model):\n    pass\n```"

    r_compile = evaluate_tier_a_checks(
        "scorer:kernelbench",
        "Optimize Model",
        good_kb,
        0.0,
        {"compiled": True, "correct": False, "error": "trial 0: max abs diff 0.5 (atol=0.01, rtol=0.01)"},
        SIRSIPathology(),
    )
    assert r_compile.names == (
        "c_extractable_code",
        "c_no_cheat_ast",
        "c_compiles",
        "c_no_runtime_err",
        "c_correct",
        "c_fast_1",
    )
    assert r_compile.bits == (1, 1, 1, 1, 0, 0)
    assert r_compile.has_exec_pass is True
    assert r_compile.credit == pytest.approx(0.25 * (0.08 + 0.12 + 0.25 + 0.20))

    r_oom = evaluate_tier_a_checks(
        "scorer:kernelbench",
        "Optimize Model",
        good_kb,
        0.0,
        {"compiled": True, "correct": False, "error": "correctness: RuntimeError: CUDA out of memory"},
        SIRSIPathology(),
    )
    assert r_oom.bits == (1, 1, 1, 0, 0, 0)
    assert r_oom.credit < r_compile.credit

    r_cheat = evaluate_tier_a_checks(
        "scorer:kernelbench",
        "Optimize Model",
        cheat_kb,
        0.0,
        {"compiled": False, "correct": False},
        SIRSIPathology(),
    )
    assert r_cheat.bits[1] == 0  # c_no_cheat_ast blocked
    assert r_cheat.has_exec_pass is False
    assert r_cheat.credit <= 0.0625

    r_kb_pass = evaluate_tier_a_checks(
        "scorer:kernelbench",
        "Optimize Model",
        good_kb,
        1.2,
        {"compiled": True, "correct": True, "speedup": 1.5},
        SIRSIPathology(),
    )
    assert r_kb_pass.bits == (1, 1, 1, 1, 1, 1)
    assert r_kb_pass.credit == 0.0

    # 2. Math checks + domain derivation + anti-hacking invariants
    r_aime_ok = evaluate_tier_a_checks(
        "scorer:math",
        "AIME problem: find the integer between 0 and 999.",
        "<think>work</think>\n\\boxed{42}",
        0.0,
        {"pred": "42"},
        SIRSIPathology(),
    )
    assert r_aime_ok.names == ("c_think_closed", "c_no_loop", "c_valid_boxed_format", "c_answer_type_domain")
    assert r_aime_ok.bits == (1, 1, 1, 1)
    assert r_aime_ok.has_exec_pass is True
    assert r_aime_ok.credit == pytest.approx(0.25)

    # Out-of-domain AIME integer (1500 not in [0, 999]) fails domain check -> syntax-only credit <= 0.0625
    r_aime_oob = evaluate_tier_a_checks(
        "scorer:math",
        "AIME problem: find the integer between 0 and 999.",
        "<think>work</think>\n\\boxed{1500}",
        0.0,
        {"pred": "1500"},
        SIRSIPathology(),
    )
    assert r_aime_oob.bits == (1, 1, 1, 0)
    assert r_aime_oob.has_exec_pass is False
    assert r_aime_oob.credit == pytest.approx(0.0625)

    # Probability domain check
    prob_prompt = "What is the probability that X > 0?"
    r_prob_bad = evaluate_tier_a_checks("scorer:math", prob_prompt, "\\boxed{3/2}", 0.0, {}, SIRSIPathology())
    r_prob_ok = evaluate_tier_a_checks("scorer:math", prob_prompt, "\\boxed{3/4}", 0.0, {}, SIRSIPathology())
    assert r_prob_bad.bits[-1] == 0 and r_prob_ok.bits[-1] == 1

    # Non-math prose in \boxed{} rejected
    r_prose = evaluate_tier_a_checks("scorer:math", "Compute x.", "\\boxed{unknown}", 0.0, {}, SIRSIPathology())
    assert r_prose.bits[-1] == 0

    # Strict pathology veto: LOOP_DEGENERATION or UNCLOSED_THINK forces credit = 0.0 and has_exec_pass = False
    r_loop = evaluate_tier_a_checks(
        "scorer:math", "Compute x.", "\\boxed{42}", 0.0, {}, SIRSIPathology(tags=[TAG_LOOP_DEGENERATION])
    )
    r_uncl = evaluate_tier_a_checks(
        "scorer:math", "Compute x.", "<think>\\boxed{42}", 0.0, {}, SIRSIPathology(tags=[TAG_UNCLOSED_THINK])
    )
    assert r_loop.credit == 0.0 and r_loop.has_exec_pass is False
    assert r_uncl.credit == 0.0 and r_uncl.has_exec_pass is False

    # Strict ordering invariant: oracle_pass (1.00) > oracle_pass+slop (0.95) >> failed+checks (<= 0.25)
    slop_ok = (
        "<think>work</think>\nCertainly! Let's delve into this. In today's fast-paced world, "
        "it is important to note that the answer is \\boxed{42}."
    )
    p_slop = detect_pathologies(slop_ok, reward=1.0)
    r_oracle_clean = 1.0
    r_oracle_slop = 1.0 - p_slop.slop_penalty
    r_fail_checks = 0.0 + r_aime_ok.credit
    assert r_oracle_clean > r_oracle_slop > r_fail_checks
    adv_mix = group_advantages([r_oracle_clean, r_oracle_slop, r_fail_checks, 0.0], norm="std")
    assert adv_mix is not None and adv_mix[0] > adv_mix[1] > 0.0 > adv_mix[2] > adv_mix[3]

    # 3. HTTP / coding fallback checks + test-tamper guard
    diff_clean = "--- a/src/app.py\n+++ b/src/app.py\n@@ -1 +1 @@\n-x=1\n+x=2\n"
    diff_tamper = "--- a/tests/test_app.py\n+++ b/tests/test_app.py\n@@ -1 +1 @@\n+pytest.skip('x')\n"
    r_http_ok = evaluate_tier_a_checks(
        "http",
        "Fix bug",
        diff_clean,
        0.0,
        {"applies_cleanly": True, "existing_tests_pass": True},
        SIRSIPathology(),
    )
    r_http_tamp = evaluate_tier_a_checks(
        "http",
        "Fix bug",
        diff_tamper,
        0.0,
        {"applies_cleanly": True},
        SIRSIPathology(),
    )
    assert r_http_ok.bits == (1, 1, 1, 1, 0) and r_http_ok.has_exec_pass is True
    assert r_http_tamp.bits[2] == 0  # c_no_test_tamper caught test modification


def test_g3_rollout_std_trap_fix_and_subskill_regret(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run_dir = str(tmp_path / "g3_std_trap")
    cfg = _guard_toy_cfg(
        steps=1,
        tasks=[
            {
                "source": "inline",
                "name": "aime",
                "verifier": "scorer:math",
                "items": [{"id": "a0", "prompt": "AIME: find integer between 0 and 999.", "answer": "777"}],
            }
        ],
        group_size=2,
        prompts_per_step=1,
        sirsi={
            "enabled": True,
            "slap_weight": 0.1,
            "anti_slop_weight": 0.05,
            "failure_ledger": False,
            "regret_enabled": True,
            "atomic_checks_enabled": True,
            "atomic_check_alpha": 0.25,
            "atomic_check_min_spread": 0.08,
            "hurt_replay_prob": 0.0,
            "pinned_replay_prob": 0.0,
        },
    )
    tr = GRPOTrainer(cfg, run_dir)
    try:
        monkeypatch.setattr(tr.gen, "encode_prompt", lambda prompt, **kw: [10, 11, 12])

        # Case 1: All-fail with ONLY syntactic difference -> has_exec_pass=False -> 0 PG seqs
        monkeypatch.setattr(
            tr.gen,
            "generate",
            lambda prompts, **kw: [
                [GenOut("<think>ok</think>\\boxed{1500}", [20, 21]), GenOut("<think>unclosed", [20, 22])]
            ],
        )
        seqs1, st1 = tr.rollout()
        assert st1["groups/all_fail"] == 1.0
        assert st1["groups/used"] == 0.0
        assert [s for s in seqs1 if s.kind == "pg"] == []

        # Case 2: All-fail with exec/domain pass (42 in [0,999], credit=0.25) vs OOB (1500, credit=0.0625)
        # Spread = 0.1875 >= 0.08 -> MUST use Dr.GRPO norm="mean", NOT norm="std" (±1.0)!
        monkeypatch.setattr(
            tr.gen,
            "generate",
            lambda prompts, **kw: [
                [GenOut("<think>ok</think>\\boxed{42}", [20, 21]), GenOut("<think>ok</think>\\boxed{1500}", [20, 22])]
            ],
        )
        seqs2, st2 = tr.rollout()
        pg2 = [s for s in seqs2 if s.kind == "pg"]
        assert st2["groups/used"] == 1.0 and len(pg2) == 2
        assert pg2[0].adv == pytest.approx(0.09375) and pg2[1].adv == pytest.approx(-0.09375)
        assert all(abs(s.adv) <= 0.25 for s in pg2), "Dr.GRPO centering must bound partial-credit adv <= alpha_check"

        # Case 3: Sub-skill regression! Next rollout drops c_answer_type_domain (1 -> 0) -> NEWLY_HURT!
        monkeypatch.setattr(
            tr.gen,
            "generate",
            lambda prompts, **kw: [
                [GenOut("<think>ok</think>\\boxed{1500}", [20, 21]), GenOut("<think>ok</think>\\boxed{2000}", [20, 22])]
            ],
        )
        _, st3 = tr.rollout()
        assert st3["regret/newly_hurt"] == 1.0
        assert [t.task_id for t in tr.sampler.hurt_queue] == ["a0"]

        # Case 4: Mixed oracle group (1 oracle pass+slop vs 1 failed+checks) -> standard adv_norm="std"
        sloppy_pass = (
            "<think>ok</think>Certainly! Let's delve into this. In today's fast-paced world, "
            "it is important to note that \\boxed{777}."
        )
        monkeypatch.setattr(
            tr.gen,
            "generate",
            lambda prompts, **kw: [
                [GenOut(sloppy_pass, [20, 21]), GenOut("<think>ok</think>\\boxed{42}", [20, 22])]
            ],
        )
        seqs4, _ = tr.rollout()
        pg4 = [s for s in seqs4 if s.kind == "pg"]
        assert len(pg4) == 2
        assert pg4[0].adv > 0.5 and pg4[1].adv < -0.5
    finally:
        tr.pool.shutdown(wait=False)


# ============================================================================ Configs & Hygiene


def test_guard_stage_configs_and_hygiene() -> None:
    for rel in ("configs/stages/s4_opt_a_d1_rl.yaml", "configs/stages/s4_v2_rl.yaml"):
        fp = ROOT / rel
        raw = fp.read_text(encoding="utf-8")
        assert not re.search(r"\bHBM\b", raw, flags=re.IGNORECASE), f"forbidden HBM reference in {rel}"
        assert "opus46" not in raw.lower(), f"forbidden opus46 reference in {rel}"

        cfg = load_config(str(fp))
        assert cfg.rl.gate.mode == "soft_hurt"
        assert cfg.rl.gate.eval_at_step_0 is True
        assert float(cfg.rl.gate.lambda_hurt) == pytest.approx(2.0)
        assert float(cfg.rl.gate.max_hurt_frac) == pytest.approx(0.05)
        assert float(cfg.rl.gate.max_bench_drop) == pytest.approx(0.025)

        scfg = SIRSIConfig.from_dict(dict(cfg.rl.sirsi))
        assert scfg.regret_enabled is True
        assert scfg.atomic_checks_enabled is True

    # Legacy s4_rl.yaml remains untouched with backward-compatible defaults
    s4_legacy = load_config(str(ROOT / "configs" / "stages" / "s4_rl.yaml"))
    assert s4_legacy.rl.gate.get("mode", "aggregate") == "aggregate"
    scfg_leg = SIRSIConfig.from_dict(dict(s4_legacy.rl.sirsi))
    assert scfg_leg.atomic_checks_enabled is False

    # Hygiene across all touched source files
    for rel in ("src/qloop/eval/runners.py", "src/qloop/train/sirsi.py", "src/qloop/train/rl.py"):
        src_text = (ROOT / rel).read_text(encoding="utf-8")
        assert not re.search(r"\bHBM\b", src_text, flags=re.IGNORECASE), f"forbidden HBM reference in {rel}"
        assert "opus46" not in src_text.lower(), f"forbidden opus46 reference in {rel}"
