"""Late-2026 frontier benchmark registry, adapters, scorers (MRCR, 10-way MCQ), revision passthrough,
Harbor Terminal-Bench 4.1 defaults, suite.yaml validation, and decontamination group registration."""
from __future__ import annotations

import dataclasses
import difflib
import json
import os
import sys
import types

import pytest

from qloop.config import load_config
from qloop.data import Decontaminator
from qloop.data import mixture as M
from qloop.data import registry as DR
from qloop.eval import external as X
from qloop.eval import registry as R
from qloop.eval import scoring as S
from qloop.eval.generate import Generator
from qloop.eval.runners import run_benchmark
from qloop.eval.testing import ByteTokenizer, ScriptedLM
from qloop.model.hssd import HSSDConfig
from qloop.model.loop import LoopSpec
from qloop.model.lsa import LSAConfig

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ----------------------------------------------------------------------------- registry specs

def test_2026_benchmarks_registered_with_verified_schemas():
    expected = {
        "terminal_bench_4_1": ("harborframework/terminal-bench", None, "test", "v4.1.0", False, "terminal_bench"),
        "terminal_bench_2": ("harborframework/terminal-bench-2.0", None, "train", None, False, "terminal_bench"),
        "hle": ("cais/hle", None, "test", None, True, "hle"),
        "hle_rolling": ("cais/hle-rolling", None, "test", None, True, "hle"),
        "hle_diamond": ("cais/hle-diamond", None, "test", None, True, "hle"),
        "hle_verified": ("lmms-lab/HLE-Verified", None, "test", None, False, "hle"),
        "gpqa_diamond": ("Idavidrein/gpqa", "gpqa_diamond", "train", None, True, "gpqa"),
        "gpqa_extended": ("Idavidrein/gpqa", "gpqa_extended", "train", None, True, "gpqa"),
        "supergpqa": ("m-a-p/SuperGPQA", None, "train", None, False, "supergpqa"),
        "mrcr": ("openai/mrcr", None, "train", None, False, "mrcr"),
        "browsecomp_long": ("openai/BrowseCompLongContext", None, "train", None, False, "browsecomp"),
        "hmmt_feb_2026": ("MathArena/hmmt_feb_2026", None, "train", None, False, "hmmt_2026"),
        "apex_shortlist": ("MathArena/apex-shortlist", None, "train", None, False, "apex_math"),
        "swe_bench_pro_hard": ("ScaleAI/SWE-bench_Pro", "hard", "test", None, False, "swe_bench"),
        "deep_swe": ("datacurve/deep-swe", None, "test", None, True, "swe_bench"),
        "mmlu_pro": ("TIGER-Lab/MMLU-Pro", None, "test", None, False, "mmlu_pro"),
        "frames": ("google/frames-benchmark", None, "test", None, False, "frames"),
    }
    for name, (hf_id, config, split, revision, gated, group) in expected.items():
        spec = R.get_spec(name)
        assert spec.hf_id == hf_id
        assert spec.config == config
        assert spec.split == split
        assert spec.revision == revision
        assert spec.gated is gated
        assert spec.contamination_group == group
        assert spec.prompt_builder is not None

    tb41 = R.get_spec("terminal_bench_4_1")
    assert tb41.harness == "harbor" and tb41.scorer is None
    assert tb41.extra["harbor_dataset"] == "terminal-bench/terminal-bench@4.1.0"
    assert "remote_sandbox" in tb41.requires

    swe_hard = R.get_spec("swe_bench_pro_hard")
    assert swe_hard.harness == "harbor" and swe_hard.extra["harbor_dataset"] == "swebenchpro-hard"
    assert R.get_spec("deep_swe").harness == "harbor"

    local = {s.name for s in R.list_specs(local_only=True, capabilities={"code_exec"})}
    assert {"hle_rolling", "hle_diamond", "hle_verified", "gpqa_extended", "supergpqa", "mrcr",
            "browsecomp_long", "hmmt_feb_2026", "apex_shortlist", "mmlu_pro", "frames"} <= local
    assert not local & {"terminal_bench_4_1", "swe_bench_pro_hard", "deep_swe"}


# ----------------------------------------------------------------------------- adapters & prompt builders

def test_adapt_hle_verified_subsets_and_multimodal():
    rows = [
        {"id": "v1", "question": "Q gold", "answer": "42", "subset": "gold", "subset_raw": "gold",
         "answer_type": "exactMatch", "raw_subject": "Math", "category": "Math",
         "problem_is_valid": True, "answer_is_valid": True, "rationale_is_valid": True, "has_image": False},
        {"id": "v2", "question": "Q rev", "answer": "7", "subset": "Revision", "subset_raw": "revision",
         "answer_type": "exactMatch", "raw_subject": "Physics", "category": "Physics",
         "problem_is_valid": True, "answer_is_valid": False, "rationale_is_valid": True, "has_image": "false"},
        {"id": "v3", "question": "Q uncertain", "answer": "x", "subset": "uncertain", "subset_raw": "uncertain",
         "answer_type": "exactMatch", "raw_subject": "Bio", "category": "Bio",
         "problem_is_valid": False, "answer_is_valid": False, "rationale_is_valid": False, "has_image": False},
        {"id": "v4", "question": "Q img", "answer": "cat", "subset": "gold", "subset_raw": "gold",
         "answer_type": "exactMatch", "raw_subject": "Art", "category": "Art",
         "problem_is_valid": True, "answer_is_valid": True, "rationale_is_valid": True, "has_image": True},
    ]
    spec = R.get_spec("hle_verified")
    items = R.load_items(spec, offline_items=rows)
    assert [it["id"] for it in items] == ["v1", "v2"]
    assert R.hle_prompt(items[0])[0]["role"] == "system"

    mm_items = R.load_items(spec, offline_items=rows, include_multimodal=True)
    assert [it["id"] for it in mm_items] == ["v1", "v2", "v4"]

    gold_only = dataclasses.replace(spec, extra={"subsets": ["gold"]})
    assert [it["id"] for it in R.load_items(gold_only, offline_items=rows)] == ["v1"]

    widened = dataclasses.replace(spec, extra={"subsets": ["gold", "revision", "uncertain"]})
    assert [it["id"] for it in R.load_items(widened, offline_items=rows)] == ["v1", "v2", "v3"]

    # direct call defaults to {"gold", "revision"}
    assert len(R.adapt_hle_verified(rows[0], 0)) == 1
    assert R.adapt_hle_verified(rows[2], 2) == []

    # minimal offline row without "subset" key is kept consistently by both adapt_hle_verified and load_items,
    # while explicit empty/None "subset" rows are filtered out
    minimal = {"id": "v_min", "question": "Minimal Q", "answer": "9"}
    empty_sub = {"id": "v_empty", "question": "Empty Q", "answer": "9", "subset": ""}
    none_sub = {"id": "v_none", "question": "None Q", "answer": "9", "subset": None}
    assert R.adapt_hle_verified(minimal, 0)[0]["subset"] is None
    assert R.adapt_hle_verified(empty_sub, 1) == []
    assert R.adapt_hle_verified(none_sub, 2) == []
    assert [it["id"] for it in R.load_items(spec, offline_items=[minimal, empty_sub, none_sub])] == ["v_min"]


def test_adapt_supergpqa_and_mmlu_pro_10_way_mcq():
    opts = [f"opt_{i}" for i in range(9)] + ["A protein complex"]
    sg_row = {
        "uuid": "sg-1", "question": "Which complex?", "options": opts,
        "answer": "A protein complex", "answer_letter": "J",
        "discipline": "Biology", "field": "Molecular", "subfield": "Proteomics",
        "difficulty": "hard", "is_calculation": False,
    }
    sg = R.load_items(R.get_spec("supergpqa"), offline_items=[sg_row])[0]
    assert sg["id"] == "sg-1" and sg["answer"] == "J" and len(sg["choices"]) == 10
    msg = R.get_spec("supergpqa").build_prompt(sg)[0]["content"]
    assert "A, B, C, D, E, F, G, H, I, J" in msg and "J) A protein complex" in msg

    # fallback without answer_letter: exact option text match wins over leading 'A '
    sg_no_letter = dict(sg_row)
    sg_no_letter.pop("answer_letter")
    assert R.load_items(R.get_spec("supergpqa"), offline_items=[sg_no_letter])[0]["answer"] == "J"

    mp_rows = [
        {"question_id": 101, "question": "Pick I", "options": [f"c{i}" for i in range(10)],
         "answer": "I", "answer_index": 8, "cot_content": "", "category": "math", "src": "ori"},
        {"question_id": 102, "question": "Pick A via index 0", "options": [f"c{i}" for i in range(10)],
         "answer": "", "answer_index": 0, "cot_content": "", "category": "law", "src": "ori"},
    ]
    mp = R.load_items(R.get_spec("mmlu_pro"), offline_items=mp_rows)
    assert [(x["id"], x["answer"]) for x in mp] == [("101", "I"), ("102", "A")]


def test_adapt_mrcr_and_prompt_fallbacks():
    conv = [
        {"role": "user", "content": "Write a poem about clouds."},
        {"role": "assistant", "content": "Soft white drifting skies."},
        {"role": "user", "content": "Prepend XYZ987 to the 1st poem about clouds."},
    ]
    rows = [
        {"prompt": json.dumps(conv), "answer": "XYZ987Soft white drifting skies.",
         "random_string_to_prepend": "XYZ987", "n_needles": 2, "desired_msg_index": 1,
         "total_messages": 3, "n_chars": 120, "date_added": "2025-04-01"},
        {"prompt": conv, "answer": "XYZ987Soft white drifting skies.",
         "random_string_to_prepend": "XYZ987"},
        {"prompt": "Plain fallback prompt", "answer": "ABCok", "random_string_to_prepend": "ABC"},
    ]
    spec = R.get_spec("mrcr")
    items = R.load_items(spec, offline_items=rows)
    assert items[0]["messages"] == conv
    assert spec.build_prompt(items[0]) == conv
    assert spec.build_prompt(items[1]) == conv
    assert spec.build_prompt(items[2]) == [{"role": "user", "content": "Plain fallback prompt"}]
    # direct mrcr_prompt call on raw JSON prompt dict
    assert R.mrcr_prompt({"prompt": json.dumps(conv)}) == conv


def test_adapt_math_arena_browsecomp_frames_terminal_bench_gpqa_extended():
    hmmt_row = {"problem_idx": 5, "problem": "Find x.", "answer": r"-\frac{1}{21}", "problem_type": "Algebra"}
    hmmt = R.load_items(R.get_spec("hmmt_feb_2026"), offline_items=[hmmt_row])[0]
    assert hmmt["id"] == "5" and hmmt["answer"] == r"-\frac{1}{21}" and hmmt["integer_answer"] is False
    prompt = R.get_spec("hmmt_feb_2026").build_prompt(hmmt)[0]["content"]
    assert "integer between 0 and 999" not in prompt and r"\boxed{}" in prompt
    assert S.MATH.score(r"\boxed{-1/21}", hmmt) == 1.0
    assert S.MATH.score(r"\boxed{-\dfrac{1}{21}}", hmmt) == 1.0
    assert S.MATH.score(r"\boxed{\frac{-1}{21}}", hmmt) == 1.0

    apex_row = {"problem_idx": 12, "problem": "Prove/find.", "answer": r"2\sqrt{3}", "source": "IMO-SL"}
    apex = R.load_items(R.get_spec("apex_shortlist"), offline_items=[apex_row])[0]
    assert apex["source"] == "IMO-SL" and S.MATH.score(r"\boxed{\sqrt{12}}", apex) == 1.0

    bc_row = {"problem": "Which treaty?", "answer": "Treaty of Westphalia", "urls": ["https://ex.org"], "canary": "c"}
    bc = R.load_items(R.get_spec("browsecomp_long"), offline_items=[bc_row])[0]
    assert bc["question"] == "Which treaty?" and S.HLE.score("Exact Answer: Treaty of Westphalia", bc) == 1.0

    fr_row = {"Unnamed: 0": 7, "Prompt": "Who discovered X?", "Answer": "Marie Curie"}
    fr = R.load_items(R.get_spec("frames"), offline_items=[fr_row])[0]
    assert fr["id"] == "7" and S.HLE.score("Exact Answer: Marie Curie", fr) == 1.0

    tb_row = {
        "name": "fix-nginx-tls", "instruction": "Repair TLS config.", "description": "TLS",
        "keywords": ["nginx"], "schema_version": "4.1", "config": {}, "metadata": {}, "files": [], "location": "x",
    }
    tb = R.load_items(R.get_spec("terminal_bench_4_1"), offline_items=[tb_row])[0]
    assert tb["id"] == "fix-nginx-tls" and tb["instruction"] == "Repair TLS config."
    assert R.get_spec("terminal_bench_4_1").build_prompt(tb) == [{"role": "user", "content": "Repair TLS config."}]

    gpqa_row = {"Record ID": "ext-1", "Question": "Q?", "Correct Answer": "right",
                "Incorrect Answer 1": "w1", "Incorrect Answer 2": "w2", "Incorrect Answer 3": "w3"}
    gext = R.load_items(R.get_spec("gpqa_extended"), offline_items=[gpqa_row])[0]
    assert gext["choices"][R.LETTERS.index(gext["answer"])] == "right"


# ----------------------------------------------------------------------------- scorers: MRCR & 10-way MCQ

def test_mrcr_scorer_prefix_think_and_partial_ratio():
    item = {"answer": "AbC123The quick brown fox jumps over the lazy dog.",
            "random_string_to_prepend": "AbC123"}
    assert S.get_scorer("mrcr") is S.MRCR

    ev_exact = S.MRCR.evaluate("AbC123The quick brown fox jumps over the lazy dog.", item)
    assert ev_exact["score"] == 1.0 and ev_exact["prefix_ok"] is True

    think_resp = "<think>\nSearching context...\n</think>\n\nAbC123The quick brown fox jumps over the lazy dog."
    ev_think = S.MRCR.evaluate(think_resp, item)
    assert ev_think["score"] == 1.0 and ev_think["prefix_ok"] is True

    # Missing prefix must score 0.0 even if body matches verbatim
    ev_missing = S.MRCR.evaluate("The quick brown fox jumps over the lazy dog.", item)
    assert ev_missing["score"] == 0.0 and ev_missing["prefix_ok"] is False

    # Chatty preamble before prefix fails startswith(prefix) -> 0.0
    ev_chatty = S.MRCR.evaluate("Here is the answer: AbC123The quick brown fox jumps over the lazy dog.", item)
    assert ev_chatty["score"] == 0.0 and ev_chatty["prefix_ok"] is False

    # Prefix only inside <think> block -> 0.0
    assert S.MRCR.score("<think>AbC123The quick brown fox jumps over the lazy dog.</think>", item) == 0.0

    # Partial body match matches difflib.SequenceMatcher ratio exactly
    partial_body = "The quick red fox jumps over the sleeping dog."
    expected_ratio = difflib.SequenceMatcher(None, partial_body,
                                             "The quick brown fox jumps over the lazy dog.").ratio()
    ev_partial = S.MRCR.evaluate("<think>ok</think>\nAbC123" + partial_body, item)
    assert ev_partial["prefix_ok"] is True and ev_partial["score"] == pytest.approx(expected_ratio)


def test_mrcr_end_to_end_run_benchmark(tmp_path):
    rows = [
        {"prompt": json.dumps([{"role": "user", "content": "Say PREhello"}]),
         "answer": "PREhello", "random_string_to_prepend": "PRE"},
        {"prompt": json.dumps([{"role": "user", "content": "Say ZZZhello"}]),
         "answer": "ZZZhello", "random_string_to_prepend": "ZZZ"},
    ]
    gen = Generator(ScriptedLM.from_text("<think>check</think>\n\nPREhello"), ByteTokenizer())
    s = run_benchmark("mrcr", gen, max_new_tokens=64, out_dir=str(tmp_path), offline_items=rows)
    assert s["depths"]["native"]["mean_score"] == pytest.approx(0.5)


@pytest.mark.parametrize("letter", list("ABCDEFGHIJ"))
def test_mcq_10_way_all_letters_a_through_j(letter):
    choices = [f"choice_{i}" for i in range(10)]
    item = {"choices": choices, "answer": letter}
    assert S.MCQ.score(f"Step by step...\nAnswer: {letter}", item) == 1.0
    assert S.MCQ.score(f"**Answer**: ({letter})", item) == 1.0
    assert S.MCQ.score(rf"Therefore \boxed{{\mathrm{{{letter}}}}}", item) == 1.0


def test_mcq_10_way_pronoun_i_not_false_positive():
    choices = [f"choice_{i}" for i in range(10)]
    # Prose "answer I will..." must NOT hijack extraction before "**G**"
    txt = "To find the answer I will compute each term carefully.\n\n**G**"
    assert S.extract_mcq(txt, n_choices=10) == "G"
    assert S.MCQ.score(txt, {"choices": choices, "answer": "G"}) == 1.0
    # Explicit "Answer: I" or "The answer is (I)." DOES extract I
    assert S.extract_mcq("Therefore the answer is (I).", n_choices=10) == "I"
    assert S.extract_mcq("Answer: I", n_choices=10) == "I"
    # 4-way MCQ still rejects E..J
    assert S.extract_mcq("Answer: I", n_choices=4) is None


# ----------------------------------------------------------------------------- revision passthrough

def test_eval_load_items_revision_passthrough_only_when_set(monkeypatch):
    calls = []

    class FakeDS(list):
        column_names = ("name", "instruction")

    def fake_load_dataset(hf_id, config=None, split=None, token=None, **extra):
        calls.append({"hf_id": hf_id, "config": config, "split": split, "extra": extra})
        if hf_id == "harborframework/terminal-bench":
            return FakeDS([{"name": "t1", "instruction": "do x"}])
        return FakeDS([{"problem_idx": 1, "problem": "1+1", "answer": "2"}])

    monkeypatch.setitem(sys.modules, "datasets", types.SimpleNamespace(load_dataset=fake_load_dataset))

    # Spec WITHOUT revision must NOT pass revision kwarg
    items_no_rev = R.load_items(R.get_spec("hmmt_feb_2026"))
    assert len(items_no_rev) == 1
    assert "revision" not in calls[-1]["extra"]

    # Spec WITH revision MUST pass revision="v4.1.0"
    items_rev = R.load_items(R.get_spec("terminal_bench_4_1"))
    assert len(items_rev) == 1 and items_rev[0]["id"] == "t1"
    assert calls[-1]["extra"] == {"revision": "v4.1.0"}


def test_data_raw_rows_revision_passthrough_only_when_set(monkeypatch):
    calls = []

    def fake_load_dataset(hf_id, config=None, split=None, streaming=None, **extra):
        calls.append({"hf_id": hf_id, "config": config, "split": split, "extra": extra})
        if hf_id == "harborframework/terminal-bench":
            return [{"instruction": "Fix the broken systemd unit file in /etc/systemd."}]
        return [{"problem": "Compute the sum of digits."}]

    monkeypatch.setitem(sys.modules, "datasets", types.SimpleNamespace(load_dataset=fake_load_dataset))

    rows_no_rev = list(M.raw_rows(DR.get_spec("hmmt_feb_2026"), 0, 0, src_cfg={"shuffle_buffer": 0}, streaming=False))
    assert len(rows_no_rev) == 1
    assert "revision" not in calls[-1]["extra"]

    rows_rev = list(M.raw_rows(DR.get_spec("terminal_bench_4_1"), 0, 0, src_cfg={"shuffle_buffer": 0}, streaming=False))
    assert len(rows_rev) == 1
    assert calls[-1]["extra"] == {"revision": "v4.1.0"}


# ----------------------------------------------------------------------------- Harbor & suite.yaml

def test_harbor_defaults_to_terminal_bench_4_1():
    assert X.HarborConfig is X.HarborRun
    run = X.HarborRun()
    assert run.dataset == "terminal-bench/terminal-bench@4.1.0"
    cmd = X.build_harbor_command(run)
    assert cmd[cmd.index("--dataset") + 1] == "terminal-bench/terminal-bench@4.1.0"


def test_eval_suite_yaml_loads_and_matches_2026_slate():
    cfg = load_config(os.path.join(ROOT, "configs", "eval", "suite.yaml"))
    loop = LoopSpec.from_dict(dict(cfg.get("loop") or {}))
    hssd = HSSDConfig.from_dict(dict(cfg.get("hssd") or {}))
    lsa = LSAConfig.from_dict(dict(cfg.get("lsa") or {}))
    hssd.validate()
    lsa.validate()
    assert loop.num_passes == 3 and loop.virtual_layers(40) == 88
    assert cfg.hssd.enabled is True and cfg.lsa.enabled is True and cfg.lsa.mode == "sparse"
    assert cfg.eval.defaults.depths == [1, 2, 3]
    assert cfg.eval.external.harbor.dataset == "terminal-bench/terminal-bench@4.1.0"

    names = [b["name"] if isinstance(b, dict) else b for b in cfg.eval.benchmarks]
    expected_slate = [
        "aime_2026", "hmmt_feb_2026", "gpqa_diamond", "supergpqa", "hle_diamond", "hle_rolling",
        "arc_agi_2", "livecodebench", "kernelbench_l3", "mrcr", "longbench_v2", "mmlu_pro",
    ]
    assert names == expected_slate
    for n in names:
        spec = R.get_spec(n)
        assert spec.scorer is not None


# ----------------------------------------------------------------------------- decontam groups

def test_decontam_eval_groups_and_mrcr_unwrapping(monkeypatch):
    group_expect = {
        "hle": {"hle", "hle_rolling", "hle_diamond", "hle_verified"},
        "gpqa": {"gpqa_diamond", "gpqa_extended"},
        "supergpqa": {"supergpqa"},
        "terminal_bench": {"terminal_bench_4_1"},
        "mrcr": {"mrcr"},
        "hmmt_2026": {"hmmt_feb_2026"},
        "apex_math": {"apex_shortlist"},
        "browsecomp": {"browsecomp_long"},
        "mmlu_pro": {"mmlu_pro"},
        "frames": {"frames"},
    }
    for group, expected_names in group_expect.items():
        got = {s.name for s in DR.eval_specs_for_group(group)}
        assert expected_names <= got, (group, got)

    tb_spec = DR.get_spec("terminal_bench_4_1")
    assert tb_spec.hf_id == "harborframework/terminal-bench"
    assert tb_spec.revision == "v4.1.0"
    assert tb_spec.text_fields == ["instruction"]
    assert tb_spec.contamination_group == "terminal_bench"

    # Mock datasets for MRCR JSON chat unwrapping + mixed gated/open HLE warn fallback
    mrcr_msg = (
        "Line one of a unique benchmark needle\n"
        "line two continues across a newline cleanly."
    )
    mrcr_conv = [
        {"role": "user", "content": mrcr_msg},
        {"role": "assistant", "content": "Acknowledged the multi-line needle carefully."},
    ]
    hle_open_q = "Open audited HLE verified question spanning more than thirteen distinct unique words here."
    tb_instr = "Configure nginx reverse proxy with custom mTLS certificates inside the container workspace."

    def fake_load_dataset(hf_id, config=None, split=None, streaming=None, **extra):
        if hf_id == "openai/mrcr":
            return [{"prompt": json.dumps(mrcr_conv), "answer": "PREneedle answer words"}]
        if hf_id.startswith("cais/hle"):
            raise PermissionError("401 gated")
        if hf_id == "lmms-lab/HLE-Verified":
            return [{"question": hle_open_q}]
        if hf_id == "harborframework/terminal-bench":
            assert extra.get("revision") == "v4.1.0"
            return [{"instruction": tb_instr}]
        raise AssertionError(f"unexpected {hf_id}")

    monkeypatch.setitem(sys.modules, "datasets", types.SimpleNamespace(load_dataset=fake_load_dataset))

    # MRCR JSON unwrapping catches 13-grams even across an embedded newline!
    provs = M.default_eval_providers(["mrcr", "terminal_bench"])
    dec = Decontaminator.from_providers(provs, n=13)
    assert dec.is_contaminated(
        "Leaked: Line one of a unique benchmark needle line two continues across a newline cleanly."
    ) == (True, "mrcr")
    assert dec.is_contaminated(
        "Task: Configure nginx reverse proxy with custom mTLS certificates inside the container workspace."
    ) == (True, "terminal_bench")

    # Mixed gated/open HLE group: on_missing="error" fails; on_missing="warn" still indexes open HLE-Verified!
    with pytest.raises(RuntimeError, match="hle"):
        Decontaminator.from_providers(M.default_eval_providers(["hle"], on_missing="error"), n=13, on_missing="error")
    dec_hle = Decontaminator.from_providers(M.default_eval_providers(["hle"], on_missing="warn"), n=13,
                                            on_missing="warn")
    assert dec_hle.is_contaminated(
        "Open audited HLE verified question spanning more than thirteen distinct unique words here."
    ) == (True, "hle")
