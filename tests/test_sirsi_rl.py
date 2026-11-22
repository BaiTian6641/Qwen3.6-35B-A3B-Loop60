"""Unit and end-to-end tests for Stage 4 SI-RSI (Supervised Instructed RSI)."""
from __future__ import annotations

import os
from typing import Any

import pytest
import torch

from qloop.config import load_config
from qloop.data.decontam import Decontaminator
from qloop.data.tokenizer import ToyTokenizer
from qloop.eval.generate import GenOut
from qloop.model.experts import param_group_of
from qloop.train.rl import GRPOTrainer, TeacherClient, _Seq, group_advantages
from qloop.train.sirsi import (
    TAG_AI_SLOP,
    TAG_LOOP_DEGENERATION,
    TAG_PREMATURE_EXIT,
    TAG_UNCLOSED_THINK,
    FailureLedger,
    SIRSIConfig,
    build_curriculum_sculpt_messages,
    build_hint_messages,
    detect_pathologies,
    map_char_span_to_token_mask,
    parse_and_decontaminate_sculpted_tasks,
    unlikelihood_span_loss,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ============================================================================ pathology detector


def test_detect_pathologies_loop_degeneration_and_exact_onset():
    prefix = "First we compute the discriminant carefully and obtain delta equals nine. "
    phrase = "Wait let me double check the sign of the quadratic coefficient again. "
    text = prefix + phrase * 3
    res = detect_pathologies(text, reward=0.0)
    assert TAG_LOOP_DEGENERATION in res.tags
    expected_onset = len(prefix) + len(phrase)
    assert res.loop_onset_char == expected_onset
    assert text[res.loop_onset_char :].startswith("Wait let me double check")

    # 12-gram cycle repeating 3x
    phrase12 = (
        "Let us recheck every single boundary condition on the interval from zero to one carefully. "
    )
    text12 = "Step one: " + phrase12 * 3
    res12 = detect_pathologies(text12, reward=0.0)
    assert TAG_LOOP_DEGENERATION in res12.tags
    assert res12.loop_onset_char == len("Step one: ") + len(phrase12)

    # High duplicate tail fraction with shorter 4-word repeating tail
    tail_text = "We solve the linear system step by step using row reduction. " + "check x and y " * 8
    res_tail = detect_pathologies(tail_text, reward=0.0)
    assert TAG_LOOP_DEGENERATION in res_tail.tags
    assert res_tail.loop_onset_char is not None and res_tail.loop_onset_char > 0

    # Unspaced character-level loop
    char_loop = "Result: " + "abcd1234" * 8
    res_char = detect_pathologies(char_loop, reward=0.0)
    assert TAG_LOOP_DEGENERATION in res_char.tags
    assert res_char.loop_onset_char is not None

    # Clean solution has no loop degeneration
    clean = (
        "<think>To compute 17 * 13, split 17 * 10 = 170 and 17 * 3 = 51. Summing gives 221.</think>\n"
        "The product is \\boxed{221}."
    )
    res_clean = detect_pathologies(clean, reward=1.0)
    assert res_clean.tags == []
    assert res_clean.loop_onset_char is None
    assert res_clean.slop_hits == []
    assert res_clean.slop_penalty == 0.0


def test_detect_pathologies_unclosed_think_premature_exit_and_antislop_invariants():
    # UNCLOSED_THINK: never inspects inside unclosed <think> for AI_SLOP
    unclosed = "<think>Certainly! Let's delve into this fast-paced problem and compute x = 5"
    p_unclosed = detect_pathologies(unclosed, reward=0.0)
    assert TAG_UNCLOSED_THINK in p_unclosed.tags
    assert TAG_AI_SLOP not in p_unclosed.tags
    assert TAG_PREMATURE_EXIT not in p_unclosed.tags
    assert p_unclosed.slop_penalty == 0.0

    # Re-opened unclosed <think> after a closed block
    reopened = "<think>part 1</think> ok <think>still open"
    assert TAG_UNCLOSED_THINK in detect_pathologies(reopened, reward=0.0).tags

    # PREMATURE_EXIT: empty / short final answer when reward <= 0, but NOT when reward > 0
    assert TAG_PREMATURE_EXIT in detect_pathologies("<think>reasoning...</think>   ", reward=0.0).tags
    assert TAG_PREMATURE_EXIT in detect_pathologies("no", reward=0.0).tags
    assert TAG_PREMATURE_EXIT not in detect_pathologies("<think>reasoning...</think> 42", reward=1.0).tags

    # AI_SLOP: strictly final-answer prose (clichés inside <think> MUST NOT be penalized)
    think_only_slop = (
        "<think>Certainly! Let's delve into this. In today's fast-paced world, "
        "it is important to note that as an AI language model we compute 6 * 7 = 42.</think>\n"
        "\\boxed{42}"
    )
    p_think = detect_pathologies(think_only_slop, reward=1.0)
    assert TAG_AI_SLOP not in p_think.tags
    assert p_think.slop_hits == []
    assert p_think.slop_penalty == 0.0

    # AI_SLOP in final answer: bounded tie-breaker never flips incorrect > correct
    sloppy_answer = (
        "<think>6 * 7 = 42.</think>\n"
        "Certainly! Let's delve into this calculation. In today's fast-paced world, "
        "it is important to note that as an AI language model the answer is \\boxed{42}."
    )
    p_slop = detect_pathologies(sloppy_answer, reward=1.0, cfg={"max_slop_penalty": 0.05})
    assert TAG_AI_SLOP in p_slop.tags
    assert len(p_slop.slop_hits) >= 3
    assert 0.0 < p_slop.slop_penalty <= 0.05

    # Reward ordering invariant: correct+crisp > correct+sloppy >> wrong+crisp > wrong+sloppy
    r_correct_crisp = 1.0 - p_think.slop_penalty
    r_correct_sloppy = 1.0 - p_slop.slop_penalty
    r_wrong_crisp = 0.0
    r_wrong_sloppy = 0.0 - p_slop.slop_penalty
    assert r_correct_crisp > r_correct_sloppy > r_wrong_crisp > r_wrong_sloppy

    # GRPO group advantage check: correct+sloppy stays positive, wrong+crisp stays negative
    adv_mixed = group_advantages([r_correct_crisp, r_correct_sloppy, r_wrong_crisp, r_wrong_sloppy])
    assert adv_mixed is not None
    assert adv_mixed[0] > adv_mixed[1] > 0.0 > adv_mixed[2] > adv_mixed[3]

    # All-pass tie-breaker: vibrant direct response beats boilerplate-heavy response
    adv_tie = group_advantages([r_correct_crisp, r_correct_sloppy])
    assert adv_tie is not None and adv_tie[0] > 0.0 > adv_tie[1]


# ============================================================================ unlikelihood loss & span mask


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_unlikelihood_span_loss_finiteness_mask_zero_grad_and_direction(dtype):
    # 1. Strictly finite even when token_logprobs == 0.0 (including bf16!)
    lp0 = torch.tensor([0.0, -0.5, 0.0], dtype=dtype, requires_grad=True)
    mask = torch.tensor([True, False, True])
    loss0 = unlikelihood_span_loss(lp0, mask, eps=1e-4)
    assert torch.isfinite(loss0) and float(loss0.detach()) > 0.0
    loss0.backward()
    assert lp0.grad is not None and torch.isfinite(lp0.grad).all()
    assert float(lp0.grad[1]) == 0.0  # exact zero grad outside span_mask

    # 2. Empty span_mask returns exact 0.0 with clean autograd graph
    lp_empty = torch.tensor([-0.2, -0.8], dtype=dtype, requires_grad=True)
    loss_empty = unlikelihood_span_loss(lp_empty, torch.tensor([False, False]), eps=1e-4)
    assert float(loss_empty.detach()) == 0.0 and loss_empty.requires_grad
    loss_empty.backward()
    assert lp_empty.grad is not None and torch.all(lp_empty.grad == 0.0)

    # 3. Exact zero gradient outside span_mask + directional update lowers looped-token probs
    logits = torch.tensor([[3.0, 0.0, 0.0], [3.0, 0.0, 0.0]], dtype=torch.float32, requires_grad=True)
    targets = torch.tensor([0, 0])
    span = torch.tensor([False, True])
    log_probs = torch.log_softmax(logits, dim=-1).gather(1, targets[:, None]).squeeze(1)
    p_before = torch.exp(log_probs.detach())
    loss = unlikelihood_span_loss(log_probs, span, eps=1e-4)
    loss.backward()
    assert logits.grad is not None
    assert torch.all(logits.grad[0] == 0.0), "outside span_mask must have strictly zero gradient"
    assert float(logits.grad[1, 0]) > 0.0, "positive grad on target logit lowers probability under GD"

    with torch.no_grad():
        logits_after = logits - 0.1 * logits.grad
    p_after = torch.softmax(logits_after, dim=-1).gather(1, targets[:, None]).squeeze(1)
    assert float(p_after[0]) == pytest.approx(float(p_before[0]), abs=1e-7)
    assert float(p_after[1]) < float(p_before[1]), "unlikelihood step must reduce looped-token probability"


def test_map_char_span_to_token_mask_toy_fast_and_fallback():
    tok = ToyTokenizer()
    text = "Valid prefix. Repeat me now. Repeat me now."
    onset = len("Valid prefix. Repeat me now. ")
    ids = tok.encode(text)
    mask = map_char_span_to_token_mask(tok, ids, text, onset)
    assert len(mask) == len(ids)
    first_true = mask.index(True)
    assert tok.decode(ids[:first_true]) == text[:onset]
    assert tok.decode(ids[first_true:]) == text[onset:]

    # Trailing EOS is preserved as unpenalized (False)
    ids_eos = [*ids, tok.eos_token_id]
    mask_eos = map_char_span_to_token_mask(tok, ids_eos, text, onset)
    assert len(mask_eos) == len(ids_eos)
    assert mask_eos[-1] is False and mask_eos[:-1] == mask

    # Mock HF fast tokenizer with offset_mapping
    class _FastMockTok:
        eos_token_id = 99

        def __call__(self, s: str, add_special_tokens: bool = False, return_offsets_mapping: bool = False):
            assert not add_special_tokens and return_offsets_mapping
            return {"input_ids": [10, 11, 12], "offset_mapping": [(0, 4), (4, 9), (9, 14)]}

    fmask = map_char_span_to_token_mask(_FastMockTok(), [10, 11, 12, 99], "abcd efghi jklm", onset_char=9)
    assert fmask == [False, False, True, False]

    # Fallback when tokenizer is None or onset_char is None
    fb = map_char_span_to_token_mask(None, [1, 2, 3, 4, 5, 6], "", None, fallback_tail_frac=0.5)
    assert fb == [False, False, False, True, True, True]


# ============================================================================ surgical slap_target_groups


def _sirsi_toy_cfg(*, fused: bool = False, slap_target_groups: list[str] | None = None,
                   teacher: dict[str, Any] | None = None, guided_hint_enabled: bool = False) -> dict[str, Any]:
    return {
        "seed": 0,
        "stage": "sirsi-test",
        "model": {"name": "toy", "dtype": "float32", "device_map": "cpu", "toy": {"seed": 0}},
        "loop": {"enabled": True, "prelude_end": 4, "core_end": 12, "num_passes": 2, "target_virtual_layers": 24},
        "train": {
            "steps": 1,
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
            "tasks": [{"source": "inline", "name": "m", "verifier": "scorer:math",
                       "items": [{"id": "t0", "prompt": "Compute 6 * 7.", "answer": "42"}]}],
            "group_size": 2,
            "prompts_per_step": 1,
            "max_new_tokens": 16,
            "temperature": 1.0,
            "micro_batch_seqs": 2,
            "ppo_epochs": 1,
            "teacher": teacher or {},
            "sirsi": {
                "enabled": True,
                "slap_weight": 0.2,
                "slap_eps": 1e-4,
                "slap_max_seqs_per_step": 4,
                "slap_target_groups": list(slap_target_groups or []),
                "anti_slop_weight": 0.05,
                "guided_hint_enabled": guided_hint_enabled,
                "max_hints_per_step": 2,
                "failure_ledger": True,
            },
        },
    }


@pytest.mark.parametrize("fused", [False, True])
def test_surgical_slap_target_groups_gradient_routing(tmp_path, fused):
    run_dir = str(tmp_path / f"surgical_{fused}")
    cfg = _sirsi_toy_cfg(fused=fused, slap_target_groups=["loop_adapters", "router"])
    tr = GRPOTrainer(cfg, run_dir)
    # Ensure pass=2 is active so loop_adapters participate in the forward graph
    assert tr.stack.loop is not None
    tr.stack.loop.set_active_passes(2)

    targeted_names = [
        n for n in tr.stack.named_trainable if param_group_of(n) in ("loop_adapters", "router")
    ]
    nontargeted_names = [
        n for n in tr.stack.named_trainable if param_group_of(n) not in ("loop_adapters", "router")
    ]
    assert targeted_names, "expected loop_adapters/router trainables"
    assert nontargeted_names, "expected non-targeted trainables (e.g. norms)"

    before = {n: p.detach().clone() for n, p in tr.stack.named_trainable.items()}

    # Pure slap_neg step: verify targeted params get nonzero slap grad & update, non-targeted get ZERO
    slap_seq = _Seq([10, 11, 12, 13, 14, 15, 16, 17], 2, -1.0, "slap_neg",
                    span_mask=[False, False, True, True, True, True], tags=(TAG_LOOP_DEGENERATION,))

    captured_grads: dict[str, torch.Tensor | None] = {}
    if not fused:
        orig_step = tr.opt.step

        def _spy_step():
            for n, p in tr.stack.named_trainable.items():
                captured_grads[n] = None if p.grad is None else p.grad.detach().clone()
            return orig_step()

        tr.opt.step = _spy_step  # type: ignore[method-assign]

    st = tr.update([slap_seq])
    assert st["loss/slap_neg"] > 0.0
    assert st["optim/optim/skipped_params"] == 0.0
    assert st["optim/optim/skipped_total"] == 0.0
    assert all(p.grad is None for p in tr.stack.named_trainable.values())

    if not fused:
        for n in nontargeted_names:
            assert captured_grads[n] is None, f"non-targeted param {n} received slap gradient!"
        assert any(
            captured_grads[n] is not None and float(captured_grads[n].abs().max()) > 0.0
            for n in targeted_names
        )

    # Weight deltas: targeted params moved, non-targeted params bit-identical!
    for n in nontargeted_names:
        assert torch.equal(tr.stack.named_trainable[n].detach(), before[n]), (
            f"non-targeted param {n} mutated during surgical slap_neg (fused={fused})"
        )
    assert any(
        not torch.equal(tr.stack.named_trainable[n].detach(), before[n])
        for n in targeted_names
    ), f"no targeted param updated during surgical slap_neg (fused={fused})"

    # Mixed [slap_neg + pg] step also succeeds cleanly with zero skipped params
    pg_pos = _Seq([10, 11, 20, 21, 22, 23], 2, 1.0, "pg")
    pg_neg = _Seq([10, 11, 24, 25, 26, 27], 2, -1.0, "pg")
    st_mix = tr.update([slap_seq, pg_pos, pg_neg])
    assert st_mix["loss/slap_neg"] > 0.0
    assert st_mix["optim/optim/skipped_params"] == 0.0
    assert st_mix["optim/optim/skipped_total"] == 0.0
    assert all(p.grad is None for p in tr.stack.named_trainable.values())


def test_sirsi_aux_loss_hygiene_with_hssd_and_multi_epoch(tmp_path):
    """Ensure stale no-grad HSSD/loop aux never leaks across surgical slap_neg, pg, and entropy."""
    run_dir = str(tmp_path / "sirsi_aux")
    cfg = _sirsi_toy_cfg(fused=False, slap_target_groups=["loop_adapters", "router"])
    cfg["hssd"] = {
        "enabled": True,
        "chunk_size": 8,
        "window_chunks": 2,
        "num_pages": 4,
        "k_pages": 2,
        "theta_low_rank": True,
        "segment_chunks": 2,
        "checkpoint_segments": False,
    }
    cfg["train"]["train_hssd"] = True
    cfg["train"]["aux_weights"] = {"loop_aux": 1.0, "hssd_budget": 0.5}
    cfg["loop"]["per_pass_router"] = True
    cfg["loop"]["router_diversity_weight"] = 0.01
    cfg["rl"]["ppo_epochs"] = 2
    tr = GRPOTrainer(cfg, run_dir)
    try:
        assert tr.stack.loop is not None and tr.stack.hssd is not None
        tr.stack.loop.set_active_passes(2)
        hssd_before = {
            n: p.detach().clone()
            for n, p in tr.stack.named_trainable.items()
            if param_group_of(n) == "hssd_adapters"
        }
        assert hssd_before

        # Inject stale no-grad aux simulating rollout leftovers
        for ad in tr.stack.hssd.adapters.values():
            ad._pending_aux.append(torch.tensor(9999.0))

        slap_seq = _Seq(
            [10, 11, 12, 13, 14, 15, 16, 17],
            2,
            -1.0,
            "slap_neg",
            span_mask=[False, False, True, True, True, True],
        )
        st_slap = tr.update([slap_seq])
        assert st_slap["loss/slap_neg"] > 0.0
        assert st_slap["optim/optim/skipped_total"] == 0.0
        # Surgical slap_neg drained HSSD pending aux AND did not mutate hssd_adapters!
        assert all(len(ad._pending_aux) == 0 for ad in tr.stack.hssd.adapters.values())
        for n, p0 in hssd_before.items():
            assert torch.equal(tr.stack.named_trainable[n].detach(), p0)

        # Mixed [slap_neg + pg] with stale no-grad aux injected again
        for ad in tr.stack.hssd.adapters.values():
            ad._pending_aux.append(torch.tensor(9999.0))
        pg_pos = _Seq([10, 11, 20, 21, 22, 23, 24, 25], 2, 1.0, "pg")
        pg_neg = _Seq([10, 11, 26, 27, 28, 29, 30, 31], 2, -1.0, "pg")
        st = tr.update([slap_seq, pg_pos, pg_neg])
        assert st["loss/slap_neg"] > 0.0
        assert st["loss/hssd_budget"] < 10.0
        assert st["optim/optim/skipped_total"] == 0.0
        assert all(len(ad._pending_aux) == 0 for ad in tr.stack.hssd.adapters.values())
    finally:
        tr.pool.shutdown(wait=False)


# ============================================================================ end-to-end GRPO + mocked Supervisor


def test_grpo_sirsi_end_to_end_loop_degen_and_mocked_teacher(tmp_path, monkeypatch):
    secret = "sk-super-secret-never-leak"
    monkeypatch.setenv("MOCK_SUPERVISOR_KEY", secret)
    run_dir = str(tmp_path / "sirsi_e2e")
    cfg = _sirsi_toy_cfg(
        fused=False,
        slap_target_groups=["loop_adapters", "router"],
        teacher={
            "base_url": "http://mock-supervisor.local/v1",
            "model": "frontier-mock",
            "api_key_env": "MOCK_SUPERVISOR_KEY",
            "max_calls_per_step": 2,
            "sft_weight": 0.1,
        },
        guided_hint_enabled=False,
    )
    tr = GRPOTrainer(cfg, run_dir)
    assert tr.teacher is not None
    assert secret not in repr(tr.teacher)

    # Toy model vocab_size=97 -> keep token IDs strictly in [1, 90] (ASCII printable subsets < 97)
    # Use lowercase-free or safe char codes? ToyConfig vocab_size=97 covers bytes 0..96 (0x20..0x60 = uppercase/digits!)
    loop_text = "SETUP STEP. " + "CHECK BOUNDARY VALUE AGAIN NOW PLEASE DO IT. " * 3
    loop_ids = [min(b, 95) for b in tr.stack.tokenizer.encode(loop_text)][:48]
    unclosed_text = "<think>STILL COMPUTING WITHOUT CLOSING"
    unclosed_ids = [min(b, 95) for b in tr.stack.tokenizer.encode(unclosed_text)][:24]

    monkeypatch.setattr(
        tr.gen,
        "generate",
        lambda prompts, **kw: [[GenOut(loop_text, loop_ids), GenOut(unclosed_text, unclosed_ids)] for _ in prompts],
    )
    monkeypatch.setattr(tr.gen, "encode_prompt", lambda prompt, **kw: [10, 11, 12, 13])
    monkeypatch.setattr(tr.gen, "encode_text", lambda text: [min(b, 95) for b in tr.stack.tokenizer.encode(text)][:20])
    monkeypatch.setattr(tr, "_eos", lambda: 96)

    def _fake_post(url: str, payload: dict[str, Any], timeout: float, headers: dict[str, str] | None = None,
                   retries: int = 3) -> dict[str, Any]:
        assert headers and headers.get("Authorization") == f"Bearer {secret}"
        return {
            "choices": [{"message": {"content": "The answer is \\boxed{42}."}}],
            "usage": {"total_tokens": 37},
        }

    monkeypatch.setattr("qloop.train.rl._post_json", _fake_post)

    res = tr.train()
    last = res["last"]
    assert res["step"] == 1
    assert last["sirsi/loop_degen_rate"] == pytest.approx(0.5)
    assert last["sirsi/unclosed_think_rate"] == pytest.approx(0.5)
    assert last["sirsi/n_slap_seqs"] == 2.0
    assert last["teacher/calls"] == 1.0 and last["teacher/verified"] == 1.0
    assert last["loss/slap_neg"] > 0.0
    assert last["loss/teacher"] > 0.0 and last["loss/teacher_nll"] == last["loss/teacher"]
    assert last["optim/optim/skipped_total"] == 0.0

    ledger_path = os.path.join(run_dir, "sirsi_failures.jsonl")
    assert os.path.exists(ledger_path)
    with open(ledger_path, encoding="utf-8") as f:
        raw_ledger = f.read()
    assert secret not in raw_ledger
    rows = FailureLedger(ledger_path).read_recent(limit=10)
    assert len(rows) == 2
    assert TAG_LOOP_DEGENERATION in rows[0]["tags"]
    assert rows[0]["loop_onset_char"] is not None
    assert rows[0]["task_id"] == "t0"
    assert TAG_UNCLOSED_THINK in rows[1]["tags"]


def test_grpo_sirsi_guided_hint_recovery_and_teacher_budget_caps(tmp_path, monkeypatch):
    run_dir = str(tmp_path / "sirsi_hint")
    cfg = _sirsi_toy_cfg(
        fused=False,
        teacher={
            "base_url": "http://mock-supervisor.local/v1",
            "model": "frontier-mock",
            "max_calls_per_step": 2,
            "max_total_calls": 2,
        },
        guided_hint_enabled=True,
    )
    tr = GRPOTrainer(cfg, run_dir)
    assert tr.teacher is not None

    gen_calls: list[Any] = []

    def _fake_generate(prompts, **kw):
        gen_calls.append(prompts)
        if len(gen_calls) == 1:
            return [[GenOut("wrong answer 41", [20, 21, 22]), GenOut("wrong answer 40", [20, 21, 23])]]
        # Guided retry receives the Supervisor Hint and succeeds!
        assert "[Supervisor Hint]:" in str(prompts[0])
        return [[GenOut("Now I see it: \\boxed{42}", [30, 31, 32, 33])]]

    monkeypatch.setattr(tr.gen, "generate", _fake_generate)
    monkeypatch.setattr(tr.gen, "encode_prompt", lambda prompt, **kw: [10, 11, 12])

    api_calls: list[dict[str, Any]] = []

    def _fake_post(url: str, payload: dict[str, Any], timeout: float, headers: dict[str, str] | None = None,
                   retries: int = 3) -> dict[str, Any]:
        api_calls.append(payload)
        return {
            "choices": [{"message": {"content": "Check the product of 6 and 7 directly before rounding."}}],
            "usage": {"total_tokens": 18},
        }

    monkeypatch.setattr("qloop.train.rl._post_json", _fake_post)

    seqs, rstats = tr.rollout()
    assert rstats["sirsi/hint_calls"] == 1.0
    assert rstats["sirsi/hint_recovered"] == 1.0
    assert rstats["teacher/calls"] == 0.0  # skipped expensive solve() after guided recovery!
    # Recovery seq trains on ORIGINAL un-hinted prompt p_ids=[10,11,12] + recovery tokens [30,31,32,33]
    rec_seqs = [s for s in seqs if s.kind == "teacher"]
    assert len(rec_seqs) == 1
    assert rec_seqs[0].ids == [10, 11, 12, 30, 31, 32, 33]
    assert rec_seqs[0].n_prompt == 3

    # Verify TeacherClient budget cap enforcement
    tc = TeacherClient({"base_url": "http://mock.local/v1", "model": "m", "max_total_calls": 1})
    hint_msgs = build_hint_messages("Compute 6*7", "bad", [TAG_LOOP_DEGENERATION])
    assert len(hint_msgs) == 2 and TAG_LOOP_DEGENERATION in hint_msgs[1]["content"]
    assert tc.hint("Compute 6*7", "bad", [TAG_LOOP_DEGENERATION])
    with pytest.raises(RuntimeError, match="budget reached"):
        tc.hint("Compute 6*7", "bad")
    tr.pool.shutdown(wait=False)


def test_antislop_does_not_positively_reinforce_all_fail_groups(tmp_path, monkeypatch):
    """Regression: 3 crisp-wrong + 1 sloppy-wrong (all_fail) must emit 0 PG seqs and groups/used == 0.0."""
    run_dir = str(tmp_path / "sirsi_all_fail_slop")
    cfg = _sirsi_toy_cfg(fused=False)
    cfg["rl"]["group_size"] = 4
    tr = GRPOTrainer(cfg, run_dir)
    try:
        crisp_wrong = "After checking carefully, the product is \\boxed{41}."
        sloppy_wrong = (
            "Certainly! Let's delve into this calculation. In today's fast-paced world, "
            "it is important to note that the answer is \\boxed{41}."
        )
        samples = [
            GenOut(crisp_wrong, [20, 21, 22, 23]),
            GenOut(crisp_wrong, [20, 21, 22, 24]),
            GenOut(crisp_wrong, [20, 21, 22, 25]),
            GenOut(sloppy_wrong, [20, 21, 22, 26]),
        ]
        monkeypatch.setattr(tr.gen, "generate", lambda prompts, **kw: [samples for _ in prompts])
        monkeypatch.setattr(tr.gen, "encode_prompt", lambda prompt, **kw: [10, 11, 12])

        seqs, rstats = tr.rollout()
        assert rstats["groups/all_fail"] == 1.0
        assert rstats["groups/used"] == 0.0
        assert [s for s in seqs if s.kind == "pg"] == []
        assert rstats["sirsi/ai_slop_rate"] == pytest.approx(0.25)
        assert rstats["sirsi/mean_slop_penalty"] > 0.0
        rows = FailureLedger(os.path.join(run_dir, "sirsi_failures.jsonl")).read_recent(limit=10)
        assert any(TAG_AI_SLOP in r["tags"] for r in rows)
    finally:
        tr.pool.shutdown(wait=False)


# ============================================================================ curriculum sculpting & decontamination


def test_parse_and_decontaminate_sculpted_tasks_blocks_contaminated(monkeypatch):
    decontam = Decontaminator(n=13)
    held_out_long = (
        "Find the number of ordered pairs of positive integers whose product equals "
        "two thousand twenty six factorial divided by twelve."
    )
    held_out_short = "Compute the sum of all prime divisors of nine hundred."
    decontam.add_texts("aime_2026", [held_out_long, held_out_short])

    msgs = build_curriculum_sculpt_messages(
        [{"task_id": "t1", "tags": [TAG_LOOP_DEGENERATION], "reward": 0.0, "prompt": "Solve x^2=9"}],
        n_tasks=3,
    )
    assert len(msgs) == 2 and TAG_LOOP_DEGENERATION in msgs[1]["content"]

    raw_supervisor_reply = f"""Here are the sculpted tasks:
```json
[
  {{
    "id": "sculpt-clean",
    "prompt": "A rectangular garden has perimeter 34 and area 60. Find its longer side in \\\\boxed{{}}.",
    "answer": "12",
    "verifier": "scorer:math",
    "target_tags": ["LOOP_DEGENERATION"]
  }},
  {{
    "id": "sculpt-leaked-long",
    "prompt": "Bonus challenge: {held_out_long} Put answer in \\\\boxed{{}}.",
    "answer": "17",
    "verifier": "scorer:math"
  }},
  {{
    "id": "sculpt-leaked-short",
    "prompt": "{held_out_short}",
    "answer": "10",
    "verifier": "scorer:math"
  }},
  {{
    "id": "sculpt-bad-verifier",
    "prompt": "Arbitrary remote call attempt.",
    "answer": "1",
    "verifier": "http"
  }}
]
```"""
    tasks = parse_and_decontaminate_sculpted_tasks(raw_supervisor_reply, decontaminator=decontam)
    assert [t["id"] for t in tasks] == ["sculpt-clean"]
    assert tasks[0]["answer"] == "12"
    assert tasks[0]["verifier"] == "scorer:math"
    assert decontam.hits[("aime_2026", "sirsi_sculpt")] == 2

    # End-to-end TeacherClient.sculpt_curriculum() wiring
    monkeypatch.setattr(
        "qloop.train.rl._post_json",
        lambda url, payload, timeout, headers=None, retries=3: {
            "choices": [{"message": {"content": raw_supervisor_reply}}],
            "usage": {"total_tokens": 120},
        },
    )
    tc = TeacherClient({"base_url": "http://mock.local/v1", "model": "frontier-mock"})
    sculpted = tc.sculpt_curriculum(
        [{"task_id": "t1", "tags": [TAG_LOOP_DEGENERATION], "reward": 0.0, "prompt": "p"}],
        n_tasks=4,
        decontaminator=decontam,
    )
    assert len(sculpted) == 1 and sculpted[0]["id"] == "sculpt-clean"


# ============================================================================ s4_rl.yaml validation


def test_s4_rl_yaml_sirsi_config_and_invariants():
    cfg = load_config(os.path.join(ROOT, "configs", "stages", "s4_rl.yaml"))
    assert cfg.gdn2.enabled is False
    assert cfg.hssd.enabled is True
    assert cfg.lsa.enabled is True and cfg.lsa.mode == "sparse"
    assert cfg.model.yarn_target_len == 1048576
    assert cfg.train.optim.fused is False
    assert cfg.train.sha256_frozen_experts is True
    assert dict(cfg.rl.teacher) == {}

    scfg = SIRSIConfig.from_dict(dict(cfg.rl.sirsi))
    assert scfg.enabled is True
    assert scfg.slap_weight == pytest.approx(0.1)
    assert scfg.slap_eps == pytest.approx(1e-4)
    assert scfg.slap_max_seqs_per_step == 4
    assert scfg.slap_target_groups == ["loop_adapters", "router"]
    assert scfg.anti_slop_weight == pytest.approx(0.05)
    assert scfg.effective_max_slop_penalty == pytest.approx(0.05)
    assert scfg.guided_hint_enabled is True
    assert scfg.max_hints_per_step == 2
    assert scfg.failure_ledger is True
