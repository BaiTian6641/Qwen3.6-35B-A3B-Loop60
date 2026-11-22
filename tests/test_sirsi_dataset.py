"""Offline unit tests for Track B SIRSI dataset audit, triage, salvage, synthesis, and export."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import pytest
import yaml

from qloop.data.formats import Example
from qloop.data.sirsi_dataset import (
    ALL_AUDITED_SOURCES,
    ALLOWED_LICENSES,
    EXCLUDED_SOURCES,
    UPSTREAM_SOURCES,
    AsyncTeacherClient,
    CrossSourceDeduplicator,
    SourceCitation,
    TrackBSynthesizer,
    _scrub_secret,
    build_or_load_decontaminator,
    check_domain_sanity,
    detect_pathology,
    deterministic_salvage,
    load_golden_seeds,
    publish_to_hub,
    render_dataset_card,
    run_streaming_pipeline,
    score_anti_slop,
    score_three_beat_cot,
    stream_source_rows,
    strip_eval_boilerplate,
    triage_example,
    verify_synthesized_row,
    verify_upstream_licenses,
    write_jsonl_records,
    write_parquet_records,
)

GOLDEN_SEEDS_PATH = Path(__file__).resolve().parent.parent / "configs" / "data" / "sirsi_golden_seeds.json"


# ============================================================================ Helpers / Mocks

class _FakeCardData:
    def __init__(self, license_val: Any):
        self._lic = license_val

    def to_dict(self) -> dict[str, Any]:
        return {"license": self._lic} if self._lic is not None else {}


class _FakeDatasetInfo:
    def __init__(self, license_val: Any, tags: list[str] | None = None, sha: str = "abcdef123456"):
        self.card_data = _FakeCardData(license_val)
        self.tags = tags or []
        self.sha = sha


class _FakeHfApi:
    def __init__(self, mapping: dict[str, _FakeDatasetInfo]):
        self._mapping = mapping

    def dataset_info(self, repo_id: str) -> _FakeDatasetInfo:
        if repo_id not in self._mapping:
            raise KeyError(repo_id)
        return self._mapping[repo_id]


class _MockTeacherClient(AsyncTeacherClient):
    """Offline mock teacher client recording calls and returning scripted replies."""

    def __init__(self, responses: dict[str, dict[str, Any]] | list[dict[str, Any]]):
        super().__init__(provider="hf_router", model="mock/teacher-1", api_key="hf_fake_secret_token_for_tests_012345")
        self._responses = responses
        self.calls: list[list[dict[str, str]]] = []

    async def chat_completion(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int = 2048,
        temperature: float = 0.4,
    ) -> dict[str, Any]:
        self.calls.append(messages)
        if isinstance(self._responses, list):
            idx = min(len(self.calls) - 1, len(self._responses) - 1)
            return self._responses[idx]
        user_txt = messages[-1]["content"]
        for needle, reply in self._responses.items():
            if needle in user_txt:
                return reply
        return {"ok": False, "status_code": 500, "content": "", "error": "unmatched_mock"}


def _clean_chat_example(
    source: str = "openthoughts3",
    prompt: str = "Compute the sum of the first 10 positive integers. Put your answer in \\boxed{}.",
    reasoning: str = (
        "Beat 1 (Framing): Given n = 10, the goal is S = 1 + 2 + ... + 10 using Gauss's invariant n(n+1)/2. "
        "Substituting n = 10 yields 10 * 11 / 2 = 55.\n"
        "Beat 2 (Adversarial Self-Audit): Double-check by direct pairwise summation: (1+10)+(2+9)+(3+8)+(4+7)+(5+6) "
        "= 5 * 11 = 55. Verify boundary n = 1 gives 1*2/2 = 1. No off-by-one error.\n"
        "Beat 3 (Synthesis): Present the formula and final boxed integer cleanly."
    ),
    response: str = (
        "Using the arithmetic series formula $n(n+1)/2$ for $n = 10$: "
        "$\\frac{10 \\cdot 11}{2} = \\boxed{55}$."
    ),
    row_index: int = 0,
    extra_messages: list[dict[str, Any]] | None = None,
) -> Example:
    msgs: list[dict[str, Any]] = list(extra_messages or [])
    msgs.append({"role": "user", "content": prompt, "reasoning": None})
    msgs.append({"role": "assistant", "content": response, "reasoning": reasoning})
    return Example(source=source, kind="chat", messages=msgs, row_index=row_index)


# ============================================================================ 1. License & Citation Registry

def test_license_registry_exclusions_offline():
    assert set(UPSTREAM_SOURCES) == {
        "fable51", "ot_agent_sft", "openthoughts3", "nemotron_post",
        "opus46_3000x", "opus46_3300x", "opus46_traceinv", "arc_agi2_train",
    }
    assert set(EXCLUDED_SOURCES) == {"agentic_fable5", "fable5_distill"}
    for src in UPSTREAM_SOURCES.values():
        assert src.license in ALLOWED_LICENSES
        assert src.bibtex.strip() and src.citation_key

    rep = verify_upstream_licenses(live=False)
    assert rep["agentic_fable5"]["included"] is False
    assert "agpl-3.0" in rep["agentic_fable5"]["reason"]
    assert rep["fable5_distill"]["included"] is False
    assert "unknown" in rep["fable5_distill"]["reason"]
    assert rep["fable51"]["included"] is True
    assert rep["nemotron_post"]["included"] is True

    with pytest.raises(ValueError, match="Refusing to stream excluded source"):
        list(stream_source_rows(EXCLUDED_SOURCES["agentic_fable5"], max_rows=1))


def test_license_live_api_override_blocks_incompatible():
    fake_api = _FakeHfApi({
        UPSTREAM_SOURCES["fable51"].hf_id: _FakeDatasetInfo("agpl-3.0"),
        UPSTREAM_SOURCES["openthoughts3"].hf_id: _FakeDatasetInfo(None, tags=["license:apache-2.0"]),
    })
    rep = verify_upstream_licenses(
        [UPSTREAM_SOURCES["fable51"], UPSTREAM_SOURCES["openthoughts3"]],
        live=True,
        api=fake_api,
    )
    assert rep["fable51"]["included"] is False
    assert rep["fable51"]["live_license"] == "agpl-3.0"
    assert rep["openthoughts3"]["included"] is True
    assert rep["openthoughts3"]["live_license"] == "apache-2.0"


def test_triage_drops_excluded_source_immediately():
    ex = _clean_chat_example(source="agentic_fable5")
    dec = triage_example(ex, EXCLUDED_SOURCES["agentic_fable5"])
    assert dec.verdict == "DROP"
    assert any("license_incompatible" in r for r in dec.reasons)


# ============================================================================ 2. Anti-Slop & 3-Beat Scoring

def test_anti_slop_scoring_clean_vs_slop():
    clean = "Cache eviction cascades occur when maxmemory drops 10x; enforce a 25% diff guard in CI."
    s_clean = score_anti_slop(clean)
    assert s_clean["slop_score"] == 0.0
    assert not s_clean["slop_flags"]

    slop_wall = (
        "Certainly! Great question! In today's fast-paced landscape, let's delve into this tapestry:\n"
        "- **Point One:** It is important to note this game-changer.\n"
        "- **Point Two:** Seamless integration across teams.\n"
        "- **Point Three:** A comprehensive overview of value.\n"
        "- **Point Four:** Embark on a vibrant ecosystem.\n"
        "- **Point Five:** A testament to reliability.\n"
        "- **Point Six:** Let's break this down further.\n"
        "In conclusion, hope this helps! Feel free to ask if you have any other questions!"
    )
    s_slop = score_anti_slop(slop_wall)
    assert s_slop["slop_score"] >= 0.70
    assert any("delve" in f for f in s_slop["slop_flags"])
    assert any("bold_bullet_wall" in f for f in s_slop["slop_flags"])


def test_three_beat_cot_scoring():
    strong = score_three_beat_cot(
        reasoning=(
            "Given constraint x > 0, first define f(x) = x + 1/x. By AM-GM f(x) >= 2. "
            "Wait: let's verify the boundary case x = 1 by substituting back: 1 + 1 = 2, and test x = 0.5 -> 2.5 > 2."
        ),
        response="By AM-GM, $x + 1/x \\ge 2$ with equality at $x = 1$, so the minimum is $\\boxed{2}$.",
    )
    assert strong["has_beat2_audit"] is True
    assert strong["three_beat_score"] >= 0.70

    weak = score_three_beat_cot(
        reasoning="Compute x plus one over x. That equals two.",
        response="The answer is 2.",
    )
    assert weak["has_beat2_audit"] is False
    assert weak["three_beat_score"] < 0.55


# ============================================================================ 3. Pathology & Domain Gates

def test_pathology_detection_all_failure_modes():
    # Empty reasoning
    assert "pathology:reasoning_empty" in detect_pathology("q", "", "ans")
    # Redacted reasoning
    assert "pathology:reasoning_redacted" in detect_pathology("q", "[redacted]", "ans")
    # Unclosed code fence
    assert "pathology:unclosed_code_fence" in detect_pathology("q", "valid reasoning here", "```python\nx = 1\n")
    # Refusal / incomplete prompt (including disguised "no actual problem statement" templates)
    ref = "I notice you haven't actually included a problem statement for me to solve."
    assert "pathology:refusal_or_incomplete_prompt" in detect_pathology("q", "Checking input...", ref)
    ref2 = "However, **you haven't provided the actual problem statement or question**."
    assert "pathology:refusal_or_incomplete_prompt" in detect_pathology("q", "Checking input...", ref2)
    # Repetition loop
    loop_txt = "\n".join(["Let me re-evaluate the exact same equation carefully now."] * 6)
    assert "pathology:repetition_loop" in detect_pathology("q", loop_txt, "42")
    # Corrupted bytes
    assert "pathology:corrupted_text" in detect_pathology("q", "ok", "bad\x00bytes")
    # Mid-sentence truncation
    trunc = (
        "This proof proceeds by induction on the positive integer k and carefully examines every case. " * 5
        + "and therefore the"
    )
    assert "pathology:mid_sentence_truncation" in detect_pathology("q", "valid", trunc)


def test_domain_sanity_python_arc_and_agentic():
    # Broken Python AST -> hard drop
    bad_py = "Here is the code:\n```python\ndef broken(x:\n    return x + 1\n```"
    hard, _, _ = check_domain_sanity("code", "Write a fn", "think", bad_py)
    assert "domain:python_syntax_error" in hard

    # Valid Python AST -> clean
    good_py = "```python\ndef add(a: int, b: int) -> int:\n    return a + b\n```"
    hard2, _, _ = check_domain_sanity("code", "Write add", "think", good_py)
    assert not hard2

    # ARC valid (including 1x1 and blank-row pixel grids) & invalid grids
    hard_arc_ok, _, ans_arc = check_domain_sanity("arc", "ARC", "", "01\n10")
    assert not hard_arc_ok and json.loads(ans_arc) == [[0, 1], [1, 0]]
    hard_arc_1x1, _, ans_1x1 = check_domain_sanity("arc", "ARC", "", "7")
    assert not hard_arc_1x1 and json.loads(ans_1x1) == [[7]]
    blank_grid = "\n".join(["00000000000000000000"] * 8)
    assert "pathology:repetition_loop" not in detect_pathology("ARC", "", blank_grid, allow_empty_reasoning=True)
    hard_arc_bad, _, _ = check_domain_sanity("arc", "ARC", "", "not a grid at all")
    assert "domain:arc_grid_invalid" in hard_arc_bad

    # Agentic redacted placeholders -> SALVAGE (not DROP), invalid JSON -> DROP
    msgs_redacted = [
        {"role": "user", "content": "Inspect repo"},
        {
            "role": "assistant",
            "content": "",
            "reasoning": "List dir first.",
            "tool_calls": [
                {"type": "function", "function": {"name": "bash", "arguments": '{"command": "<bash_command>"}'}}
            ],
        },
    ]
    h_r, s_r, _ = check_domain_sanity("agentic", "Inspect repo", "List dir", "", messages=msgs_redacted)
    assert not h_r and "salvage:redacted_tool_args" in s_r

    msgs_bad_json = [
        {"role": "user", "content": "Inspect repo"},
        {
            "role": "assistant",
            "content": "",
            "reasoning": "Call tool.",
            "tool_calls": [{"type": "function", "function": {"name": "bash", "arguments": "{not valid json"}}],
        },
    ]
    h_b, _, _ = check_domain_sanity("agentic", "Inspect repo", "Call tool", "", messages=msgs_bad_json)
    assert "domain:invalid_tool_json" in h_b


# ============================================================================ 4. Deterministic Salvage & Triage

def test_deterministic_salvage_and_answer_invariance():
    msgs = [
        {"role": "system", "content": "detailed thinking on", "reasoning": None},
        {"role": "user", "content": "What is 6 * 7? Put your answer in \\boxed{}.", "reasoning": None},
        {
            "role": "assistant",
            "content": (
                "<thought>Given 6 and 7, 6*7 = 42. Let's verify: 7+7+7+7+7+7 = 42. "
                "Edge check: 6*(5+2)=30+12=42.</thought>"
                "<final_answer>Certainly! Great question! The product is \\boxed{42}.\n\n"
                "Hope this helps! Let me know if you have any other questions!</final_answer>"
            ),
            "reasoning": None,
        },
    ]
    repaired, actions = deterministic_salvage(msgs, "math")
    assert all(m["role"] != "system" for m in repaired)
    assert "strip_nemotron_thinking_sysprompt" in actions
    assert "normalize_thought_tags" in actions
    assert "unwrap_final_answer_tag" in actions
    assert "strip_sycophantic_opener" in actions
    assert "strip_boilerplate_closer" in actions
    asst = repaired[-1]
    assert "\\boxed{42}" in asst["content"]
    assert not asst["content"].startswith("Certainly!")
    assert "Hope this helps" not in asst["content"]
    assert "42" in (asst["reasoning"] or "")


def test_triage_verdicts_keep_salvage_drop():
    # 1. Clean + Nemotron sysprompt -> deterministically salvaged to KEEP
    ex_keep = _clean_chat_example(
        source="nemotron_post",
        extra_messages=[{"role": "system", "content": "detailed thinking on", "reasoning": None}],
    )
    dec_k = triage_example(ex_keep, UPSTREAM_SOURCES["nemotron_post"], split_used="math")
    assert dec_k.verdict == "KEEP"
    assert dec_k.record is not None
    assert dec_k.record["provenance"]["transform_stage"] == "deterministic_salvage"
    assert "strip_nemotron_thinking_sysprompt" in dec_k.record["provenance"]["salvage_actions"]

    # 2. Raw ARC pair without CoT -> SALVAGE (queued for Track B CoT synthesis)
    ex_arc = Example(
        source="arc_agi2_train",
        kind="chat",
        messages=[
            {"role": "user", "content": "Transform grid:\n01\n10", "reasoning": None},
            {"role": "assistant", "content": "10\n01", "reasoning": None},
        ],
        row_index=0,
    )
    dec_a = triage_example(ex_arc, UPSTREAM_SOURCES["arc_agi2_train"])
    assert dec_a.verdict == "SALVAGE"
    assert "salvage:missing_cot_verified_answer" in dec_a.reasons
    assert dec_a.record is not None and dec_a.record["extracted_answer"] == "[[1, 0], [0, 1]]"

    # 3. Missing Beat-2 self-audit -> SALVAGE
    ex_no_b2 = _clean_chat_example(
        reasoning=(
            "We compute the sum of integers from one to ten by multiplying ten by eleven and dividing by two. "
            "Ten times eleven equals one hundred ten. Dividing by two yields fifty five directly."
        ),
    )
    dec_s = triage_example(ex_no_b2, UPSTREAM_SOURCES["openthoughts3"])
    assert dec_s.verdict == "SALVAGE"
    assert "salvage:missing_beat2_audit" in dec_s.reasons


# ============================================================================ 5. Dedup & 13-Gram Decontam

def test_cross_source_deduplicator_exact_and_near():
    dd = CrossSourceDeduplicator()
    p1 = (
        "Let ABC be an acute triangle with circumcircle Omega and orthocenter H. "
        "Prove that the reflection of H across midpoint M lies on Omega."
    )
    r1 = "By standard homothety centered at centroid G with ratio -1/2, the reflection lands directly on Omega."

    assert dd.check_and_add("opus46_3000x", p1, r1) == (False, "", None)
    # Exact match in opus46_3300x
    is_dup, reason, orig = dd.check_and_add("opus46_3300x", p1, r1)
    assert is_dup and reason == "dup:exact_sha256" and orig == "opus46_3000x"

    # Same prompt, slightly modified response
    is_dup2, reason2, orig2 = dd.check_and_add("opus46_3300x", p1, r1 + " QED.")
    assert is_dup2 and reason2 == "dup:exact_prompt_sha256" and orig2 == "opus46_3000x"

    # Near-duplicate (one word tweaked in a long prompt)
    p1_near = p1.replace("orthocenter H", "orthocenter point H")
    is_dup3, reason3, orig3 = dd.check_and_add("opus46_3300x", p1_near, r1)
    assert is_dup3 and reason3 == "dup:near_jaccard" and orig3 == "opus46_3000x"

    # Distinct problem
    p2 = "Find the number of spanning trees in the complete bipartite graph K_{3,4} using Kirchhoff's theorem."
    assert dd.check_and_add("opus46_3300x", p2, "432")[0] is False
    assert dd.summary()["overlap_matrix"]["opus46_3300x<-opus46_3000x"] == 3


def test_decontamination_13gram_drops_contaminated_rows():
    eval_text = (
        "A strange polynomial of degree seventeen has integer coefficients and attains the value "
        "forty two at five distinct prime numbers."
    )
    decontam, rep = build_or_load_decontaminator(eval_texts_override={"aime_2026": [eval_text]}, n=13)
    assert rep["index_size"] > 0

    ex_leak = _clean_chat_example(prompt=f"Solve: {eval_text} Find its constant term.")
    dec = triage_example(ex_leak, UPSTREAM_SOURCES["openthoughts3"], decontaminator=decontam)
    assert dec.verdict == "DROP"
    assert dec.reasons == ["contaminated:aime_2026"]

    ex_clean = _clean_chat_example()
    dec_ok = triage_example(ex_clean, UPSTREAM_SOURCES["openthoughts3"], decontaminator=decontam)
    assert dec_ok.verdict == "KEEP"


def test_false_positive_guards_arc_kernelbench_and_agent_preambles():
    # 1. ARC boilerplate header stripped -> distinct training task NOT flagged as arc_agi_2 leak
    arc_eval_prompt = (
        "Find the common transformation rule in the examples and apply it to the test input. "
        "Grids are rows of digits 0-9 (colours).\n\nExample 1 input:\n1234567890\n9876543210\n"
    )
    cleaned_arc = strip_eval_boilerplate("arc_agi_2", arc_eval_prompt)
    assert cleaned_arc.startswith("Example 1 input:")
    dc_arc, _ = build_or_load_decontaminator(eval_texts_override={"arc_agi_2": [cleaned_arc]}, n=13)
    arc_train_prompt = (
        "Find the common transformation rule in the examples and apply it to the test input. "
        "Grids are rows of digits 0-9 (colours).\n\nExample 1 input:\n0011\n1100\n"
    )
    assert dc_arc.is_contaminated(arc_train_prompt) == (False, None)

    # 2. KernelBench Apache-2.0 header stripped -> repo LICENSE header NOT flagged as kernelbench leak
    kb_code = (
        "# Copyright 2024 Example Corp\n"
        "# Licensed under the Apache License, Version 2.0 (the 'License'); you may not use this file "
        "except in compliance with the License.\n"
        "import torch\nclass Model(torch.nn.Module):\n    def forward(self, x):\n        return x * 2\n"
    )
    cleaned_kb = strip_eval_boilerplate("kernelbench", kb_code)
    assert cleaned_kb.startswith("import torch")
    dc_kb, _ = build_or_load_decontaminator(eval_texts_override={"kernelbench": [cleaned_kb]}, n=13)
    apache_file = (
        "Licensed under the Apache License, Version 2.0 (the 'License'); you may not use this file "
        "except in compliance with the License. Regular python web service without torch."
    )
    assert dc_kb.is_contaminated(apache_file) == (False, None)

    # 3. Two distinct agentic tasks sharing a long 400-word harness preamble must NOT collide in dedup
    preamble = " ".join(f"harness_rule_{i}" for i in range(420))
    task_a = f"{preamble}\nTask: fix OAuth2 DeviceAuthorizationEndpoint bug in oauthlib."
    task_b = f"{preamble}\nTask: implement sparse CSR matrix multiplication kernel in C++."
    dd = CrossSourceDeduplicator()
    assert dd.check_and_add("ot_agent_sft", task_a, '{"analysis": "fixed oauthlib"}')[0] is False
    assert dd.check_and_add("ot_agent_sft", task_b, '{"analysis": "implemented sparse csr"}')[0] is False


# ============================================================================ 6. Golden Seeds Validation

def test_golden_seeds_all_18_pass_strict_gates():
    assert GOLDEN_SEEDS_PATH.exists()
    sft, pref = load_golden_seeds(GOLDEN_SEEDS_PATH)
    assert len(sft) == 18
    assert len(pref) == 12
    for row in sft:
        assert row["provenance"]["citation_key"] == "author_original"
        assert row["provenance"]["source_license"] == "apache-2.0"
        assert row["quality_scores"]["slop_score"] < 0.15
        assert row["quality_scores"]["three_beat_score"] >= 0.65
    for p in pref:
        assert p["chosen_response"] != p["rejected_response"]
        assert p["critique"].strip()


# ============================================================================ 7. Track B Async Synthesis + Resume

def test_track_b_synthesis_accept_reject_and_resume(tmp_path: Path):
    eval_leak = "Find all pairs of twin primes p and q such that p squared plus q cubed equals a fifth power."
    decontam, _ = build_or_load_decontaminator(eval_texts_override={"hmmt_2026": [eval_leak]}, n=13)

    # 3 salvage rows: row 0 -> valid rewrite, row 1 -> wrong answer (REJECT), row 2 -> slop (REJECT)
    def _mk_salvage(sid: str, prompt: str, ans: str) -> dict[str, Any]:
        return {
            "id": sid,
            "domain": "math",
            "prompt": prompt,
            "thinking": "Draft thought.",
            "response": f"The answer is \\boxed{{{ans}}}.",
            "extracted_answer": ans,
            "triage_reasons": ["salvage:missing_beat2_audit"],
            "provenance": {
                "source_dataset": "open-thoughts/OpenThoughts3-1.2M",
                "source_name": "openthoughts3",
                "source_config": "default",
                "source_split": "train",
                "source_license": "apache-2.0",
                "source_row_index": 0,
                "citation_key": "guha2025openthoughts",
                "transform_stage": "queued_for_track_b",
                "salvage_actions": [],
            },
        }

    rows = [
        _mk_salvage("s_ok", "Compute 6 * 7 in \\boxed{}.", "42"),
        _mk_salvage("s_bad_ans", "Compute 9 * 9 in \\boxed{}.", "81"),
        _mk_salvage("s_slop", "Compute 5 * 5 in \\boxed{}.", "25"),
    ]

    mock_client = _MockTeacherClient({
        "6 * 7": {
            "ok": True,
            "status_code": 200,
            "error": "",
            "content": (
                "<think>\nBeat 1 (Framing): Given 6 and 7, compute the product 6 * 7 = 42. "
                "Beat 2 (Self-Audit): Verify by splitting 6 * (5 + 2) = 30 + 12 = 42; check boundary 6 * 1 = 6.\n"
                "</think>\n\nMultiply $6 \\times 7 = \\boxed{42}$."
            ),
        },
        "9 * 9": {
            "ok": True,
            "status_code": 200,
            "error": "",
            "content": (
                "<think>\nBeat 1: Given 9 and 9, compute 9*9. Beat 2: Verify edge case carefully -> 99.\n</think>\n\n"
                "The product is \\boxed{99}."
            ),
        },
        "5 * 5": {
            "ok": True,
            "status_code": 200,
            "error": "",
            "content": (
                "<think>\nBeat 1: Given 5 and 5. Beat 2: Verify 5*5 = 25.\n</think>\n\n"
                "Certainly! Great question! Let's delve into the tapestry of 5 * 5. "
                "It is important to note this game-changer gives \\boxed{25}. In conclusion, hope this helps!"
            ),
        },
    })

    synth = TrackBSynthesizer(mock_client, out_dir=tmp_path, decontaminator=decontam)
    sft1, pref1 = asyncio.run(synth.run_salvage_rewrite(rows, max_items=3))
    assert len(sft1) == 1 and sft1[0]["id"] == "s_ok_rw"
    assert len(pref1) == 1 and pref1[0]["category"] == "salvage_rewrite"
    assert synth.stats["salvage_rewrite:rejected:answer_mismatch"] == 1
    assert len(mock_client.calls) == 3

    # Resume test: running again MUST skip all 3 checkpointed IDs and issue 0 new API calls!
    sft2, pref2 = asyncio.run(synth.run_salvage_rewrite(rows, max_items=3))
    assert len(mock_client.calls) == 3
    assert len(sft2) == 1 and len(pref2) == 1

    # Decontam rejection inside verify_synthesized_row
    ok_d, reasons_d, _ = verify_synthesized_row(
        domain="math",
        prompt=eval_leak,
        thinking="Beat 1: Given prime pair. Beat 2: Check edge cases and verify solution carefully.",
        response="Answer: \\boxed{0}.",
        decontaminator=decontam,
    )
    assert not ok_d and any("contaminated" in r for r in reasons_d)

    # Structured modes (aesthetic_craft & sirsi_recovery)
    struct_payload = json.dumps({
        "prompt": "Design a clean status badge in CSS.",
        "chosen_thinking": (
            "Beat 1 (Framing): Given a telemetry badge, define clear contrast and tabular numerals. "
            "Beat 2 (Self-Audit): Check WCAG AA contrast and verify no neon glow competes with alerts."
        ),
        "chosen_response": "Use `#166534` text on `#dcfce7` surface with 11px uppercase 600-weight type.",
        "rejected_thinking": "Add glowing purple gradients.",
        "rejected_response": "Certainly! Let's delve into a glowing neon game-changer!",
        "critique": "Chosen eliminates neon slop and enforces WCAG contrast.",
    })
    mock_struct = _MockTeacherClient([{"ok": True, "status_code": 200, "content": struct_payload, "error": ""}])
    synth_s = TrackBSynthesizer(mock_struct, out_dir=tmp_path, decontaminator=decontam)
    aes_sft, aes_pref = asyncio.run(synth_s.run_structured_mode("aesthetic_craft", max_items=1))
    assert len(aes_sft) == 1 and len(aes_pref) == 1
    rec_sft, rec_pref = asyncio.run(synth_s.run_structured_mode("sirsi_recovery", max_items=1))
    assert len(rec_sft) == 1 and len(rec_pref) == 1


def test_secret_scrubbing_and_probe():
    secret = "hf_superSecretTokenValue1234567890"
    assert secret not in _scrub_secret(f"Authorization failed for {secret} at router", secret)
    mock_402 = _MockTeacherClient([{"ok": False, "status_code": 402, "content": "", "error": "Purchase credits"}])
    probe = asyncio.run(mock_402.probe())
    assert probe["ok"] is False and probe["status_code"] == 402


# ============================================================================ 8. End-to-End Export, Card & Dry-Run

def test_end_to_end_offline_pipeline_parquet_readme_and_dry_run(tmp_path: Path):
    good_raw = {
        "conversations": [
            {"from": "human", "value": "Find 3^4. Put your answer in \\boxed{}."},
            {
                "from": "gpt",
                "value": (
                    "<think>Beat 1: Given 3^4, compute (3^2)^2 = 9^2 = 81. "
                    "Beat 2: Verify by repeated multiplication: 3*3=9, 9*3=27, 27*3=81. Matches.</think>\n\n"
                    "$3^4 = \\boxed{81}$."
                ),
            },
        ]
    }
    bad_raw = {
        "conversations": [
            {"from": "human", "value": "Broken prompt"},
            {"from": "gpt", "value": "<think></think>empty reasoning"},
        ]
    }
    summary = run_streaming_pipeline(
        out_dir=tmp_path / "ds",
        source_names=["openthoughts3", "agentic_fable5", "fable5_distill"],
        max_per_source=5,
        golden_seeds_path=GOLDEN_SEEDS_PATH,
        eval_texts_override={"aime_2026": ["unrelated benchmark text with more than thirteen unique tokens here"]},
        live_license_check=False,
        row_stream_override={"openthoughts3": [("train", 0, good_raw), ("train", 1, bad_raw)]},
    )

    assert summary["output_counts"]["sft_core"] == 19  # 18 golden + 1 clean openthoughts3
    assert summary["output_counts"]["sirsi_preferences"] == 12
    assert summary["per_source"]["agentic_fable5"]["excluded"] is True
    assert summary["per_source"]["fable5_distill"]["excluded"] is True

    # Parquet readback verification
    sft_tbl = pq.read_table(tmp_path / "ds" / "data" / "sft_core.parquet")
    assert sft_tbl.num_rows == 19
    assert "provenance" in sft_tbl.column_names and "quality_scores" in sft_tbl.column_names

    # README YAML & BibTeX verification
    readme = (tmp_path / "ds" / "README.md").read_text(encoding="utf-8")
    assert "@misc{guha2025openthoughts" in readme
    assert "agpl-3.0" in readme and "EXCLUDED" in readme
    yaml_block = readme.split("---")[1]
    meta = yaml.safe_load(yaml_block)
    assert meta["license"] in ("apache-2.0", "cc-by-4.0")
    assert [c["config_name"] for c in meta["configs"]] == ["sft_core", "salvage_queue", "sirsi_preferences"]

    # Dry-run Hub manifest validation
    manifest = publish_to_hub(tmp_path / "ds", "TassP0lar/Loop64-Fable-SIRSI-v1", public=True, dry_run=True)
    assert manifest["validated"] is True and manifest["uploaded"] is False
    assert manifest["visibility"] == "public"
    assert len(manifest["files"]) == 5

    # Secret leak guard must refuse publishing if an HF token pattern appears
    (tmp_path / "ds" / "README.md").write_text(readme + "\nhf_ABCDEFGHIJKLMNOPQRSTUVWXYZ1234\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="potential secret pattern"):
        publish_to_hub(tmp_path / "ds", "TassP0lar/Loop64-Fable-SIRSI-v1", dry_run=True)

    # JSONL & empty Parquet export sanity
    jl_path = tmp_path / "out.jsonl"
    assert write_jsonl_records(sft_tbl.to_pylist()[:2], jl_path) == 2
    assert len(jl_path.read_text(encoding="utf-8").splitlines()) == 2
    empty_pq = tmp_path / "empty.parquet"
    assert write_parquet_records([], empty_pq, kind="pref") == 0
    assert pq.read_table(empty_pq).num_rows == 0

    # Direct card renderer smoke
    card2 = render_dataset_card(summary)
    assert "Loop64-Fable-SIRSI-v1" in card2
    assert all(k in ALL_AUDITED_SOURCES for k in ("fable51", "agentic_fable5", "fable5_distill"))
    assert isinstance(SourceCitation.__dataclass_fields__, dict)


def test_cli_dry_run_manifest_validation(tmp_path: Path):
    import importlib.util

    script_path = Path(__file__).resolve().parent.parent / "scripts" / "build_sirsi_dataset.py"
    spec = importlib.util.spec_from_file_location("build_sirsi_dataset", script_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    out_dir = tmp_path / "cli_ds"
    run_streaming_pipeline(
        out_dir=out_dir,
        source_names=["agentic_fable5"],
        max_per_source=2,
        golden_seeds_path=GOLDEN_SEEDS_PATH,
        eval_texts_override={"aime_2026": ["unrelated benchmark text with more than thirteen unique tokens here"]},
        live_license_check=False,
    )
    rc = mod.main([
        "--push-to-hub",
        "--repo-id", "TassP0lar/Loop64-Fable-SIRSI-v1",
        "--public",
        "--dry-run",
        "--out-dir", str(out_dir),
    ])
    assert rc == 0

