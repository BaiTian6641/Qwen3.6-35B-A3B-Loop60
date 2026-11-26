"""Tests for Sprint 2 Gate B Generative Reasoning Harness (``qloop.eval.sprint2_reasoning``)."""
from __future__ import annotations

import json
import sys
import types

import pytest

from qloop.eval import sprint2_reasoning as SR
from qloop.eval.testing import ByteTokenizer, ScriptedLM

# ----------------------------------------------------------------------------- CLI & budget parsing


def test_cli_arg_parsing_and_per_benchmark_overrides(tmp_path):
    ap = SR.build_arg_parser()
    args = ap.parse_args(
        [
            "--benchmarks",
            "aime_2026,gpqa_diamond,arc_agi_2",
            "--limits",
            "aime_2026=30,gpqa_diamond=50,arc_agi_2=20",
            "--n-samples",
            "aime_2026=1,gpqa_diamond=1,arc_agi_2=2",
            "--max-new-tokens",
            "aime_2026=4096,gpqa_diamond=2048,arc_agi_2=4096",
            "--temperature",
            "aime_2026=0.0,arc_agi_2=0.6",
            "--enable-thinking",
            "false",
            "--batch-size",
            "2",
            "--padded-batching",
            "--trained-groups",
            "kernelbench,arc_agi_2",
            "--fallback",
            "gpqa_diamond=supergpqa",
            "--seed",
            "123",
            "--out-dir",
            str(tmp_path),
        ]
    )

    benches = SR.parse_benchmark_list(args.benchmarks)
    assert benches == ["aime_2026", "gpqa_diamond", "arc_agi_2"]
    budgets = {
        b.name: b
        for b in SR.resolve_benchmark_budgets(
            benches,
            limits=args.limits,
            n_samples=args.n_samples,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )
    }
    assert budgets["aime_2026"].limit == 30 and budgets["aime_2026"].n_samples == 1
    assert budgets["aime_2026"].max_new_tokens == 4096 and budgets["aime_2026"].effective_temperature() == 0.0
    assert budgets["gpqa_diamond"].limit == 50 and budgets["gpqa_diamond"].n_samples == 1
    assert budgets["gpqa_diamond"].max_new_tokens == 2048 and budgets["gpqa_diamond"].effective_temperature() == 0.0
    assert budgets["arc_agi_2"].limit == 20 and budgets["arc_agi_2"].n_samples == 2
    assert budgets["arc_agi_2"].max_new_tokens == 4096 and budgets["arc_agi_2"].effective_temperature() == 0.6

    # Bare-integer defaults also work cleanly
    bare = SR.resolve_benchmark_budgets(benches, limits="10", n_samples="4", max_new_tokens="1024", temperature=None)
    assert all(b.limit == 10 and b.n_samples == 4 and b.max_new_tokens == 1024 for b in bare)
    assert all(b.effective_temperature() == 0.6 for b in bare)  # n=4 -> 0.6 default

    assert SR.resolve_chat_template_kwargs("auto") is None
    assert SR.resolve_chat_template_kwargs("true") == {"enable_thinking": True}
    assert SR.resolve_chat_template_kwargs("false") == {"enable_thinking": False}
    assert SR.parse_fallback_map(args.fallback) == {"gpqa_diamond": "supergpqa"}
    assert SR.parse_trained_groups(args.trained_groups) == ["kernelbench", "arc_agi_2"]
    assert SR.parse_trained_groups("") == []

    # --arms-json file and inline JSON parsing (with ckpt, depths, *_enabled, lsa_mode)
    arms_spec = [
        {
            "name": "stock_qwen36_yarn4x",
            "config": "configs/eval/suite.yaml",
            "ckpt": None,
            "depths": [1],
            "loop_enabled": False,
            "gdn2_enabled": False,
            "lsa_enabled": False,
            "hssd_enabled": False,
            "d1_lora_enabled": False,
        },
        {
            "name": "s2_v2_joint_bptt",
            "config": "configs/eval/suite.yaml",
            "ckpt": "runs/s2_v2_joint_bptt",
            "depths": [1, 3],
            "loop_enabled": True,
            "gdn2_enabled": True,
            "lsa_enabled": True,
            "hssd_enabled": False,
            "d1_lora_enabled": False,
            "lsa_mode": "sparse",
        },
    ]
    arms_file = tmp_path / "arms.json"
    arms_file.write_text(json.dumps(arms_spec), encoding="utf-8")
    parsed_file = SR.parse_arms_json(arms_file)
    parsed_inline = SR.parse_arms_json(json.dumps(arms_spec))
    assert [a.name for a in parsed_file] == ["stock_qwen36_yarn4x", "s2_v2_joint_bptt"]
    assert parsed_file == parsed_inline
    assert parsed_file[1].to_loader_dict()["checkpoint"] == "runs/s2_v2_joint_bptt"
    assert "lsa.mode=sparse" in parsed_file[1].to_loader_dict()["overrides"]

    # Single-arm CLI resolution
    single = SR.resolve_arm_configs(config="configs/eval/suite.yaml", ckpt="runs/s2_v2", depths=[1, 3])
    assert len(single) == 1 and single[0].name == "s2_v2" and single[0].depths == [1, 3]


# ----------------------------------------------------------------------------- smoke-cpu end-to-end


def test_smoke_cpu_end_to_end_schema_isolation_d1_lora_and_probes(tmp_path):
    rc = SR.main(["--smoke-cpu", "--out-dir", str(tmp_path), "--probe-max-new-tokens", "16"])
    assert rc == 0

    json_path = tmp_path / "sprint2_reasoning.json"
    md_path = tmp_path / "sprint2_reasoning.md"
    assert json_path.is_file() and md_path.is_file()

    rep = json.loads(json_path.read_text(encoding="utf-8"))
    md = md_path.read_text(encoding="utf-8")

    # Schema & top-level fields
    assert rep["schema"] == "sprint2_reasoning_v1"
    assert rep["mode"] == "smoke_cpu"
    assert rep["resolved_baseline_arm"] == "stock_toy_d1"
    assert rep["allow_mixed_cache"] is False
    assert rep["fallback_disclosures"] == []

    # Three arms: stock d=1, looped d=1,3, d1_lora d=1
    assert set(rep["arms"].keys()) == {"stock_toy_d1", "looped_gdn2_lsa_d3", "d1_lora_control_d1"}
    assert rep["arms"]["stock_toy_d1"]["stack"]["has_loop"] is False
    assert rep["arms"]["looped_gdn2_lsa_d3"]["stack"]["has_loop"] is True
    assert rep["arms"]["looped_gdn2_lsa_d3"]["stack"]["has_gdn2"] is True
    assert rep["arms"]["looped_gdn2_lsa_d3"]["stack"]["has_lsa"] is True
    assert rep["arms"]["d1_lora_control_d1"]["stack"]["has_d1_lora"] is True
    assert rep["arms"]["d1_lora_control_d1"]["stack"]["has_loop"] is False
    assert rep["arms"]["looped_gdn2_lsa_d3"]["effective_depths"] == ["1", "3"]

    # Throughput probes present for every (arm, depth) with required numeric fields
    for arm_name, expected_depths in (
        ("stock_toy_d1", {"native"}),
        ("looped_gdn2_lsa_d3", {"1", "3"}),
        ("d1_lora_control_d1", {"native"}),
    ):
        probes = rep["arms"][arm_name]["throughput_probes"]
        assert set(probes.keys()) == expected_depths
        for dlabel, pr in probes.items():
            assert pr["depth"] == dlabel
            assert pr["completion_tokens"] > 0
            assert pr["decode_tok_s"] > 0.0
            assert pr["wall_s"] > 0.0
            assert pr["peak_vram_gib"] == 0.0

    # Contamination flagging: arc_agi_2 flagged, aime_2026 & gpqa_diamond held-out
    assert rep["benchmarks"]["arc_agi_2"]["contamination_group"] == "arc_agi_2"
    assert rep["benchmarks"]["arc_agi_2"]["trained_group_warning"] is True
    assert rep["benchmarks"]["aime_2026"]["trained_group_warning"] is False
    assert rep["benchmarks"]["gpqa_diamond"]["trained_group_warning"] is False

    # QL60-010 run_identity isolation across arms & per-arm cache directories
    arm_model_ids = set()
    arm_ckpt_shas = set()
    for arm_name in ("stock_toy_d1", "looped_gdn2_lsa_d3", "d1_lora_control_d1"):
        arm_dir = tmp_path / "arms" / arm_name
        assert (arm_dir / "aime_2026" / "run_config.json").is_file()
        assert (arm_dir / "gpqa_diamond" / "summary.json").is_file()
        assert (arm_dir / "arc_agi_2" / "summary.json").is_file()
        rid = rep["arms"][arm_name]["run_identities"]["aime_2026"]
        arm_model_ids.add(rid["model_id"])
        arm_ckpt_shas.add(rid["checkpoint_sha"])
    assert len(arm_model_ids) == 3
    assert len(arm_ckpt_shas) == 3

    # Side-by-side comparison tables include all required columns & rows
    for bname in ("aime_2026", "gpqa_diamond", "arc_agi_2"):
        rows = rep["tables"][bname]
        assert len(rows) == 4  # stock(native), looped(1), looped(3), d1_lora(native)
        for row in rows:
            for key in (
                "mean_score",
                "pass@1",
                "delta_vs_baseline",
                "mean_completion_tokens",
                "frac_truncated",
                "decode_tok_s",
                "wall_s",
                "peak_vram_gib",
                "contamination_group",
                "trained_group_warning",
            ):
                assert key in row
        if bname == "arc_agi_2":
            assert all(r["pass@2"] is not None for r in rows)
            assert all(r["trained_group_warning"] is True for r in rows)

    # Markdown report checks
    assert "Sprint 2 Gate B" in md
    assert "TRAINED_GROUP_WARNING" in md
    assert "stock_toy_d1" in md and "looped_gdn2_lsa_d3" in md and "d1_lora_control_d1" in md
    assert "Δ vs Baseline" in md and "Decode tok/s" in md and "Peak VRAM (GiB)" in md

    # Re-running against the same out_dir resumes cleanly with allow_mixed_cache=False
    rc2 = SR.main(["--smoke-cpu", "--out-dir", str(tmp_path), "--probe-max-new-tokens", "16"])
    assert rc2 == 0

    # QL60-010: mixing a different model_id into an existing arm cache fails loudly with allow_mixed_cache=False
    _, factories = SR.build_smoke_cpu_arms(seed=42)
    mismatched_arm = [SR.ArmConfig(name="stock_toy_d1", depths=[1], model_id="toy::different_ckpt")]
    with pytest.raises(ValueError, match="eval cache identity mismatch"):
        SR.run_sprint2_reasoning(
            mismatched_arm,
            benchmarks=SR.resolve_benchmark_budgets(["aime_2026"], max_new_tokens="4"),
            out_dir=tmp_path,
            baseline_arm="stock_toy_d1",
            allow_mixed_cache=False,
            fail_fast=True,
            offline_items=SR.build_smoke_offline_items(),
            arm_factories={"stock_toy_d1": factories["stock_toy_d1"]},
            probe_max_new_tokens=4,
        )


# ----------------------------------------------------------------------------- resilient errors & explicit fallback


def test_gated_benchmark_error_capture_without_silent_substitution_and_explicit_fallback(tmp_path, monkeypatch):
    def _fake_load_dataset(hf_id, config=None, split=None, token=None, **extra):
        if hf_id == "Idavidrein/gpqa":
            raise PermissionError("401 Unauthorized: gated dataset Idavidrein/gpqa requires HF_TOKEN")
        if hf_id == "MathArena/aime_2026":
            return [{"problem_idx": "a1", "problem": "What is 6*7?", "answer": "42"}]
        if hf_id == "m-a-p/SuperGPQA":
            return [
                {
                    "uuid": "sg1",
                    "question": "Pick B",
                    "options": ["opt0", "opt1", "opt2", "opt3"],
                    "answer": "opt1",
                    "answer_letter": "B",
                }
            ]
        raise AssertionError(f"unexpected load_dataset call: {hf_id}")

    monkeypatch.setitem(sys.modules, "datasets", types.SimpleNamespace(load_dataset=_fake_load_dataset))

    arms = [SR.ArmConfig(name="scripted_arm", depths=[1], model_id="scripted::arm")]
    factories = {
        "scripted_arm": lambda: (
            {
                "model": ScriptedLM.from_text(r"Answer: B and \boxed{42}"),
                "tokenizer": ByteTokenizer(),
                "loop": None,
                "gdn2": None,
                "lsa": None,
                "hssd": None,
                "d1_lora": None,
            },
            None,
        )
    }
    budgets = SR.resolve_benchmark_budgets(["aime_2026", "gpqa_diamond"], max_new_tokens="32")

    # 1. Without --fallback: gpqa_diamond records status="error", aime_2026 succeeds (NO silent substitution!)
    out_no_fb = tmp_path / "no_fb"
    rep = SR.run_sprint2_reasoning(
        arms,
        benchmarks=budgets,
        out_dir=out_no_fb,
        baseline_arm="scripted_arm",
        arm_factories=factories,
        probe_max_new_tokens=8,
    )
    assert rep["benchmarks"]["aime_2026"]["status"] == "ok"
    assert rep["benchmarks"]["gpqa_diamond"]["status"] == "error"
    assert "401" in rep["benchmarks"]["gpqa_diamond"]["error"]
    assert rep["benchmarks"]["gpqa_diamond"]["executed_benchmark"] == "gpqa_diamond"
    assert rep["benchmarks"]["gpqa_diamond"]["fallback_used"] is False
    assert rep["fallback_disclosures"] == []
    assert rep["tables"]["aime_2026"][0]["mean_score"] == pytest.approx(1.0)
    assert rep["tables"]["gpqa_diamond"][0]["status"] == "error"
    md_no_fb = (out_no_fb / "sprint2_reasoning.md").read_text(encoding="utf-8")
    assert "no silent substitution performed" in md_no_fb

    # 2. With --fail-fast: raises immediately on gated failure
    with pytest.raises(RuntimeError, match="Idavidrein/gpqa"):
        SR.run_sprint2_reasoning(
            arms,
            benchmarks=budgets,
            out_dir=tmp_path / "fail_fast",
            baseline_arm="scripted_arm",
            fail_fast=True,
            arm_factories=factories,
            probe_max_new_tokens=8,
        )

    # 3. With explicit --fallback gpqa_diamond=supergpqa: runs fallback AND explicitly discloses in JSON + MD
    out_fb = tmp_path / "with_fb"
    rep_fb = SR.run_sprint2_reasoning(
        arms,
        benchmarks=budgets,
        out_dir=out_fb,
        baseline_arm="scripted_arm",
        fallback_map={"gpqa_diamond": "supergpqa"},
        arm_factories=factories,
        probe_max_new_tokens=8,
    )
    assert rep_fb["benchmarks"]["gpqa_diamond"]["status"] == "fallback"
    assert rep_fb["benchmarks"]["gpqa_diamond"]["executed_benchmark"] == "supergpqa"
    assert rep_fb["benchmarks"]["gpqa_diamond"]["fallback_used"] is True
    assert len(rep_fb["fallback_disclosures"]) == 1
    assert rep_fb["fallback_disclosures"][0]["executed_benchmark"] == "supergpqa"
    assert rep_fb["tables"]["gpqa_diamond"][0]["mean_score"] == pytest.approx(1.0)
    md_fb = (out_fb / "sprint2_reasoning.md").read_text(encoding="utf-8")
    assert "FALLBACK -> `supergpqa`" in md_fb
    assert "Explicit `--fallback` Benchmark Substitutions Used" in md_fb


def test_enable_thinking_kwarg_passed_only_when_explicit(tmp_path):
    seen_kwargs: list[dict[str, object]] = []

    class _SpyTokenizer(ByteTokenizer):
        def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=True, **kwargs):
            seen_kwargs.append(dict(kwargs))
            return super().apply_chat_template(
                messages,
                tokenize=tokenize,
                add_generation_prompt=add_generation_prompt,
                **kwargs,
            )

    def _make_arm():
        return (
            {
                "model": ScriptedLM.from_text(r"\boxed{42}"),
                "tokenizer": _SpyTokenizer(),
                "loop": None,
                "gdn2": None,
                "lsa": None,
                "hssd": None,
                "d1_lora": None,
            },
            None,
        )

    arms = [SR.ArmConfig(name="spy_arm", depths=[1], model_id="spy::arm")]
    budgets = SR.resolve_benchmark_budgets(["aime_2026"], max_new_tokens="16")
    offline = {"aime_2026": [{"problem_idx": "1", "problem": "6*7?", "answer": "42"}]}

    seen_kwargs.clear()
    SR.run_sprint2_reasoning(
        arms,
        benchmarks=budgets,
        out_dir=tmp_path / "auto",
        baseline_arm="spy_arm",
        enable_thinking="auto",
        offline_items=offline,
        arm_factories={"spy_arm": _make_arm},
        probe_max_new_tokens=4,
    )
    assert seen_kwargs and all("enable_thinking" not in kw for kw in seen_kwargs)

    seen_kwargs.clear()
    SR.run_sprint2_reasoning(
        arms,
        benchmarks=budgets,
        out_dir=tmp_path / "false",
        baseline_arm="spy_arm",
        enable_thinking="false",
        offline_items=offline,
        arm_factories={"spy_arm": _make_arm},
        probe_max_new_tokens=4,
    )
    assert seen_kwargs and all(kw.get("enable_thinking") is False for kw in seen_kwargs)

    seen_kwargs.clear()
    SR.run_sprint2_reasoning(
        arms,
        benchmarks=budgets,
        out_dir=tmp_path / "true",
        baseline_arm="spy_arm",
        enable_thinking="true",
        offline_items=offline,
        arm_factories={"spy_arm": _make_arm},
        probe_max_new_tokens=4,
    )
    assert seen_kwargs and all(kw.get("enable_thinking") is True for kw in seen_kwargs)
