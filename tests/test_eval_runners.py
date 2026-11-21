"""Registry adapters / prompt builders, run_benchmark (multi-depth, resumable) and run_suite."""
from __future__ import annotations

import json
import os

import pytest

from qloop.eval import registry as R
from qloop.eval.generate import Generator
from qloop.eval.runners import run_benchmark, run_suite, summarize_records
from qloop.eval.testing import ByteTokenizer, ScriptedLM, build_toy_generator

TOY_SPEC = {"prelude_end": 4, "core_end": 12, "num_passes": 2, "target_virtual_layers": 24}
AIME_ROWS = [{"problem_idx": 1, "problem": "What is 6*7?", "answer": "42"},
             {"problem_idx": 2, "problem": "What is 40+2?", "answer": "42"},
             {"problem_idx": 3, "problem": "What is 3+4?", "answer": "7"}]


class CountingGenerator:
    def __init__(self, inner):
        self.inner = inner
        self.calls = []
        self.loop_controller = inner.loop_controller
        self.batch_size = inner.batch_size

    def generate(self, prompts, **kw):
        self.calls.append((len(prompts), kw.get("depth")))
        return self.inner.generate(prompts, **kw)


def _scripted(text=r"so \boxed{42}"):
    return Generator(ScriptedLM.from_text(text), ByteTokenizer())


# ----------------------------------------------------------------------------- registry

def test_registry_covers_required_benchmarks():
    ids = {s.hf_id for s in R.list_specs()}
    for hf in ("cais/hle", "ScaleAI/SWE-bench_Pro", "MathArena/aime_2026", "livecodebench/code_generation_lite",
               "Idavidrein/gpqa", "open-thoughts/OpenThoughts-TB-dev-v2", "arcprize/ARC-AGI-2",
               "arcprize/arc_agi_2_human_testing", "harborframework/terminal-bench-2.0", "Qwen/AgentWorldBench",
               "lilt-ai/tau-bench-lilt", "gaia-benchmark/GAIA", "ScalingIntelligence/KernelBench",
               "zai-org/LongBench-v2", "MMMU/MMMU_Pro"):
        assert hf in ids, hf
    assert R.get_spec("agentworldbench").contamination_group == R.get_spec("terminal_bench_2").contamination_group
    assert R.contamination_conflicts(["terminal_bench"]) == {
        "terminal_bench_2": "terminal_bench",
        "terminal_bench_4_1": "terminal_bench",
        "agentworldbench": "terminal_bench",
    }
    assert R.get_spec("gpqa_diamond").config == "gpqa_diamond"
    assert "remote_sandbox" in R.get_spec("terminal_bench_2").requires
    with pytest.raises(KeyError, match="registered"):
        R.get_spec("nope")
    local = {s.name for s in R.list_specs(local_only=True, capabilities={"code_exec"})}
    assert {"aime_2026", "gpqa_diamond", "hle", "arc_agi_2", "livecodebench", "longbench_v2"} <= local
    assert not local & {"terminal_bench_2", "kernelbench_l3", "mmmu_pro", "gaia", "swe_bench_pro"}
    for s in R.list_specs():  # every prompt builder is callable on its adapter output form
        assert s.prompt_builder is not None


def test_load_items_without_datasets_or_offline():
    try:
        import datasets  # noqa: F401
        pytest.skip("datasets installed")
    except ImportError:
        pass
    with pytest.raises(ImportError, match="offline_items"):
        R.load_items(R.get_spec("aime_2026"), 2)


def test_adapters_and_prompts():
    gpqa_row = {"Record ID": "rec1", "Question": "Q?", "Correct Answer": "right", "Incorrect Answer 1": "w1",
                "Incorrect Answer 2": "w2", "Incorrect Answer 3": "w3"}
    a = R.load_items(R.get_spec("gpqa_diamond"), offline_items=[gpqa_row])[0]
    b = R.load_items(R.get_spec("gpqa_diamond"), offline_items=[gpqa_row])[0]
    assert a == b and a["choices"][R.LETTERS.index(a["answer"])] == "right" and len(a["choices"]) == 4
    msgs = R.get_spec("gpqa_diamond").build_prompt(a)
    assert "Answer: X" in msgs[-1]["content"] and "D) " in msgs[-1]["content"]

    task = {"id": "t1", "train": [{"input": [[1]], "output": [[2]]}],
            "test": [{"input": [[3]], "output": [[4]]}, {"input": [[5]], "output": [[6]]}]}
    items = R.load_items(R.get_spec("arc_agi_2"), offline_items=[{"id": "t1", "train": json.dumps(task["train"]),
                                                                  "test": json.dumps(task["test"])}])
    assert [it["id"] for it in items] == ["t1#0", "t1#1"] and items[1]["answer"] == [[6]]
    assert "Test input:\n5" in R.arc_prompt(items[1])[0]["content"]

    hle = R.load_items(R.get_spec("hle"), offline_items=[
        {"id": "h1", "question": "q", "answer": "a", "answer_type": "exactMatch", "image": ""},
        {"id": "h2", "question": "q", "answer": "a", "answer_type": "exactMatch", "image": "data:..."}])
    assert [h["id"] for h in hle] == ["h1"]  # image questions skipped for text-only models
    assert R.hle_prompt(hle[0])[0]["role"] == "system"

    lcb = R.load_items(R.get_spec("livecodebench"), offline_items=[
        {"question_id": "old", "question_content": "x", "contest_date": "2023-01-01T00:00:00",
         "public_test_cases": "[]", "private_test_cases": "", "metadata": "{}"},
        {"question_id": "new", "question_content": "Print 1", "contest_date": "2025-06-01T00:00:00",
         "public_test_cases": json.dumps([{"input": "", "output": "1\n", "testtype": "stdin"}]),
         "private_test_cases": "", "metadata": "{}", "starter_code": ""}])
    assert [x["id"] for x in lcb] == ["new"] and lcb[0]["public_tests"][0]["output"] == "1\n"
    assert "stdin" in R.code_prompt(lcb[0])[0]["content"]

    lb = R.load_items(R.get_spec("longbench_v2"), offline_items=[
        {"_id": "l1", "question": "q", "choice_A": "a", "choice_B": "b", "choice_C": "c", "choice_D": "d",
         "answer": "C", "context": "x" * 1000}])
    assert lb[0]["answer"] == "C"
    assert len(R.longbench_prompt(lb[0], max_context_chars=100)[0]["content"]) < 600
    with pytest.raises(ValueError, match="adapter failed"):
        R.load_items(R.get_spec("aime_2026"), offline_items=[{"foo": 1}])


# ----------------------------------------------------------------------------- run_benchmark

def test_run_benchmark_scores_and_writes(tmp_path):
    gen = _scripted()
    s = run_benchmark("aime_2026", gen, depths=[1, 2], max_new_tokens=64, out_dir=str(tmp_path),
                      offline_items=AIME_ROWS)
    assert list(s["depths"]) == ["native"]  # no loop controller -> native depth only
    d = s["depths"]["native"]
    assert d["n_items"] == 3 and d["mean_score"] == pytest.approx(2 / 3) and d["pass@1"] == pytest.approx(2 / 3)
    bdir = tmp_path / "aime_2026"
    lines = (bdir / "depth_native.jsonl").read_text().strip().split("\n")
    assert len(lines) == 3
    rec = json.loads(lines[0])
    assert rec["samples"][0]["pred"] == 42 and rec["samples"][0]["finish_reason"] == "stop"
    assert json.loads((bdir / "summary.json").read_text())["depth_curve"] == {"native": pytest.approx(2 / 3)}


def test_run_benchmark_resume(tmp_path):
    gen = CountingGenerator(_scripted())
    run_benchmark("aime_2026", gen, limit=2, max_new_tokens=64, out_dir=str(tmp_path), offline_items=AIME_ROWS)
    assert sum(n for n, _ in gen.calls) == 2
    gen.calls.clear()
    s = run_benchmark("aime_2026", gen, limit=3, max_new_tokens=64, out_dir=str(tmp_path), offline_items=AIME_ROWS)
    assert sum(n for n, _ in gen.calls) == 1  # only the new item
    assert s["depths"]["native"]["n_items"] == 3
    gen.calls.clear()
    path = tmp_path / "aime_2026" / "depth_native.jsonl"
    lines = path.read_text().strip().split("\n")
    path.write_text("\n".join(lines[:2]) + "\n" + lines[2][:20])  # simulate a crash mid-write
    s = run_benchmark("aime_2026", gen, limit=3, max_new_tokens=64, out_dir=str(tmp_path), offline_items=AIME_ROWS)
    assert sum(n for n, _ in gen.calls) == 1 and s["depths"]["native"]["n_items"] == 3


def test_run_benchmark_depth_curve_on_looped_toy(tmp_path):
    gen = CountingGenerator(build_toy_generator(loop_spec=TOY_SPEC))
    rows = [{"Record ID": f"r{i}", "Question": f"Q{i}?", "Correct Answer": "yes", "Incorrect Answer 1": "no",
             "Incorrect Answer 2": "maybe", "Incorrect Answer 3": "never"} for i in range(2)]
    s = run_benchmark("gpqa_diamond", gen, depths=[1, 2], n_samples=2, max_new_tokens=4, out_dir=str(tmp_path),
                      offline_items=rows, seed=3)
    assert set(s["depths"]) == {"1", "2"} and set(s["depth_curve"]) == {"1", "2"}
    for d in ("1", "2"):
        m = s["depths"][d]
        assert m["n_items"] == 2 and m["n_samples"] == 2 and "pass@2" in m
        assert m["frac_truncated"] == 1.0 and m["mean_completion_tokens"] == 4
    assert {dep for _, dep in gen.calls} == {1, 2}
    assert gen.inner.loop_controller.active_passes == 2
    assert os.path.exists(tmp_path / "gpqa_diamond" / "depth_1.jsonl")
    assert os.path.exists(tmp_path / "gpqa_diamond" / "depth_2.jsonl")
    # re-run is a no-op (fully resumed) and reproduces the summary
    gen.calls.clear()
    s2 = run_benchmark("gpqa_diamond", gen, depths=[1, 2], n_samples=2, max_new_tokens=4, out_dir=str(tmp_path),
                       offline_items=rows, seed=3)
    assert gen.calls == [] and s2["depth_curve"] == s["depth_curve"]


def test_run_benchmark_refuses_unscorable(tmp_path):
    with pytest.raises(AssertionError, match="external"):
        run_benchmark("terminal_bench_2", _scripted(), out_dir=str(tmp_path), offline_items=[{"id": 1}])
    with pytest.raises(RuntimeError, match="gpu_exec"):
        run_benchmark("kernelbench_l3", _scripted(), out_dir=str(tmp_path), capabilities=[],
                      offline_items=[{"code": "x", "level": 3, "problem_id": 1}])


def test_summarize_maj():
    recs = [{"samples": [{"score": 1.0, "pred": 5, "completion_tokens": 3, "finish_reason": "stop"},
                         {"score": 0.0, "pred": 6, "completion_tokens": 3, "finish_reason": "stop"},
                         {"score": 1.0, "pred": 5, "completion_tokens": 3, "finish_reason": "length"}]}]
    s = summarize_records(recs, "aime")
    assert s["maj@3"] == 1.0 and s["pass@1"] == pytest.approx(2 / 3) and s["pass@2"] == 1.0
    assert s["frac_truncated"] == pytest.approx(1 / 3)


def test_run_suite(tmp_path):
    cfg = {"benchmarks": ["aime_2026", {"name": "terminal_bench_2"}, {"name": "kernelbench_l3"}, "missing_bench",
                          {"name": "gpqa_diamond", "limit": 1}],
           "defaults": {"max_new_tokens": 32, "depths": [1]},
           "capabilities": ["code_exec"], "trained_contamination_groups": ["aime_2026"]}
    gpqa = [{"Record ID": "x", "Question": "Q", "Correct Answer": "a", "Incorrect Answer 1": "b",
             "Incorrect Answer 2": "c", "Incorrect Answer 3": "d"}]
    out = run_suite(cfg, _scripted("Answer: A"), str(tmp_path),
                    offline_items={"aime_2026": AIME_ROWS, "gpqa_diamond": gpqa})
    b = out["benchmarks"]
    assert b["aime_2026"]["status"] == "ok" and b["aime_2026"]["contaminated"] is True
    assert b["terminal_bench_2"]["status"] == "skipped" and "harbor" in b["terminal_bench_2"]["reason"]
    assert b["kernelbench_l3"]["status"] == "skipped" and "gpu_exec" in b["kernelbench_l3"]["reason"]
    assert b["missing_bench"]["status"] == "error"
    assert b["gpqa_diamond"]["depths"]["native"]["n_items"] == 1
    assert (tmp_path / "suite_summary.json").exists()
