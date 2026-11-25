"""Sprint 1 Track A regression suite:
  - QL60-003 / A1: Deterministic SHA-1 hash-based document holdout (train/eval 0 overlap,
    cross-source duplicate consistency, and `eval_quick.ood_sources`).
  - QL60-009 / C1: Valid-token-weighted BPTT CE normalization (`1 vs 3` LM-head gradient parity,
    zero-token segment safety) and pure CE `loss/lm` logged separately from `loss/aux`.
  - QL60-004: Exact 5-step checkpoint resume including private `trainer.rng` depth-sequence parity
    and explicit `weights_only` resume mode with automatic `opt.step_count` / `lr_factor` sync.
  - QL60-010: Model-bound evaluation cache identity (`model_id`, `checkpoint_sha`, `active_passes`,
    `config_hash`, `dataset_rev`) rejecting silent cross-model/checkpoint reuse.
"""
from __future__ import annotations

import copy
import hashlib
import json
import logging
import os
import shutil
import time
from typing import Any

import pytest
import torch
import torch.nn.functional as F

from qloop.data import ToyTokenizer, build_train_stream
from qloop.data.formats import Example
from qloop.data.mixture import doc_holdout_bucket, doc_holdout_key, is_eval_holdout_doc
from qloop.eval.generate import Generator
from qloop.eval.runners import run_benchmark, run_suite
from qloop.eval.testing import ByteTokenizer, ScriptedLM, build_toy_generator
from qloop.model.toy import ToyConfig, ToyForCausalLM
from qloop.train.bptt import BPTTConfig, SegmentedKVBPTTRunner
from qloop.train.optim import WarmupCosine
from qloop.train.trainer import LMTrainer

# ==============================================================================
# 1. QL60-003 / A1 — Deterministic Hash-Based Train/Eval Document Holdout
# ==============================================================================

_EIGHT_DOC_MARKERS = [
    "DOC_MARKER_ALPHA_000",
    "DOC_MARKER_BRAVO_001",
    "DOC_MARKER_CHARLIE_002",
    "DOC_MARKER_DELTA_003",
    "DOC_MARKER_ECHO_004",
    "DOC_MARKER_FOXTROT_005",
    "DOC_MARKER_GOLF_006",
    "DOC_MARKER_HOTEL_007",
]


def _make_eight_docs() -> list[dict[str, str]]:
    return [
        {
            "id": f"doc-{i}",
            "text": f"{tag} " + f"payload token sequence number {i} for deterministic partition test. " * 2,
        }
        for i, tag in enumerate(_EIGHT_DOC_MARKERS)
    ]


def _pick_nonempty_holdout_params(docs: list[dict[str, str]]) -> tuple[int, int]:
    """Find a small (holdout_mod, holdout_eval_buckets) where both splits of `docs` are non-empty."""
    for h_mod, h_eval in ((4, 1), (3, 1), (2, 1)):
        eval_flags = [
            is_eval_holdout_doc(
                Example("finite8", "text", text=d["text"]),
                holdout_mod=h_mod,
                holdout_eval_buckets=h_eval,
            )
            for d in docs
        ]
        if any(eval_flags) and not all(eval_flags):
            return h_mod, h_eval
    raise AssertionError("failed to find non-trivial holdout split for 8-doc fixture")


def test_ql60_003_eight_doc_finite_dataset_zero_overlap(monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> None:
    """Review 1 QL60-003 repro: an 8-doc finite dataset has EXACT 0 document overlap between train and eval."""
    import qloop.data.mixture as mix_mod

    docs = _make_eight_docs()
    h_mod, h_eval = _pick_nonempty_holdout_params(docs)

    orig_raw_rows = mix_mod.raw_rows

    def _patched_raw_rows(spec, seed, epoch, skip=0, src_cfg=None, streaming=True):
        if spec.name == "finite8":
            # Deterministic per-(seed, epoch) shuffle of the SAME 8 finite docs (mirrors HF IterableDataset.shuffle)
            import random as _py_random

            rows = list(docs)
            _py_random.Random(int(seed) * 13 + int(epoch)).shuffle(rows)
            yield from rows[skip:]
            return
        yield from orig_raw_rows(spec, seed, epoch, skip=skip, src_cfg=src_cfg, streaming=streaming)

    monkeypatch.setattr(mix_mod, "raw_rows", _patched_raw_rows)

    cfg = {
        "seed": 13,
        "stage": "ql60_003_test",
        "model": {"name": "toy", "dtype": "float32", "device_map": "cpu", "toy": {"seed": 0}},
        "loop": {"enabled": False},
        "train": {"steps": 1, "trainable": [r"input_layernorm"], "optim": {"lr": 1e-3}},
        "data": {
            "seq_len": 128,
            "holdout_mod": h_mod,
            "holdout_eval_buckets": h_eval,
            "sources": [
                {
                    "name": "finite8",
                    "hf_id": "local/finite8",
                    "role": "pretrain",
                    "format": "text",
                    "weight": 1.0,
                }
            ],
        },
        "eval_quick": {"n_batches": 4, "depths": [1]},
    }
    tr = LMTrainer(cfg, str(tmp_path / "run_finite8"))
    train_stream = tr._build_stream()
    eval_batches = tr._eval_set()

    assert train_stream.split_role == "train"
    assert len(eval_batches) >= 1
    tok = tr.stack.tokenizer

    # Collect all unique train batches under default `dedupe=True` (must terminate finitely, never hang!)
    train_batches = []
    for _ in range(32):
        try:
            train_batches.append(next(train_stream))
        except StopIteration:
            break
    assert len(train_batches) >= 1
    train_text = "\n".join(tok.decode(b["input_ids"], skip_special_tokens=False) for b in train_batches)
    eval_text = "\n".join(tok.decode(b["input_ids"], skip_special_tokens=False) for b in eval_batches)

    train_seen = {m for m in _EIGHT_DOC_MARKERS if m in train_text}
    eval_seen = {m for m in _EIGHT_DOC_MARKERS if m in eval_text}

    # Exact 0 document overlap and exact full partition of all 8 docs!
    assert len(train_seen) > 0, "train split must be non-empty"
    assert len(eval_seen) > 0, "eval split must be non-empty"
    assert train_seen & eval_seen == set(), f"train/eval overlap detected: {train_seen & eval_seen}"
    assert train_seen | eval_seen == set(_EIGHT_DOC_MARKERS)

    # Verify exact SHA-1 formula matches `doc_holdout_bucket`
    for doc, marker in zip(docs, _EIGHT_DOC_MARKERS):
        ex = Example("finite8", "text", text=doc["text"])
        key = doc_holdout_key(ex)
        expected_bucket = int(hashlib.sha1(key.encode("utf-8")).hexdigest()[:8], 16) % h_mod
        assert doc_holdout_bucket(key, h_mod) == expected_bucket
        if expected_bucket < h_eval:
            assert marker in eval_seen and marker not in train_seen
        else:
            assert marker in train_seen and marker not in eval_seen

    # Verify explicit `holdout_mod: 0` disables partitioning (backward-compat escape hatch)
    cfg_off = copy.deepcopy(cfg)
    cfg_off["data"]["holdout_mod"] = 0
    tr_off = LMTrainer(cfg_off, str(tmp_path / "run_off"))
    assert tr_off._build_stream().split_role == "all"

    # Verify single-source `toy_text` without explicit `holdout_mod` stays `split_role="all"`
    cfg_toy = copy.deepcopy(cfg)
    cfg_toy["data"] = {"seq_len": 32, "sources": [{"name": "toy_text", "weight": 1.0}]}
    tr_toy = LMTrainer(cfg_toy, str(tmp_path / "run_toy"))
    assert tr_toy._build_stream().split_role == "all"


def test_ql60_003_cross_source_duplicates_and_exact_stream_resume(monkeypatch: pytest.MonkeyPatch) -> None:
    """Duplicate rows across distinct sources (different IDs / whitespace) hash to the exact same side."""
    import qloop.data.mixture as mix_mod

    docs = _make_eight_docs()
    h_mod, h_eval = _pick_nonempty_holdout_params(docs)

    for i, doc in enumerate(docs):
        ex_a = Example("source_a", "text", text=doc["text"], row_index=i)
        ex_b = Example("source_b", "text", text="  " + doc["text"].replace(" ", "   ") + "\n", row_index=999 - i)
        assert doc_holdout_key(ex_a) == doc_holdout_key(ex_b)
        assert is_eval_holdout_doc(ex_a, holdout_mod=h_mod, holdout_eval_buckets=h_eval) == is_eval_holdout_doc(
            ex_b, holdout_mod=h_mod, holdout_eval_buckets=h_eval
        )

    # Chat duplicate across two sources with different source names & row indices
    c_a = Example(
        "opus46_3000x",
        "chat",
        messages=[
            {"role": "user", "content": "Solve x + 2 = 5."},
            {"role": "assistant", "content": "x = 3", "reasoning": "Subtract 2 from 5."},
        ],
        row_index=0,
    )
    c_b = Example(
        "opus46_3300x",
        "chat",
        messages=[
            {"role": "user", "content": "Solve   x + 2 = 5."},
            {"role": "assistant", "content": "x = 3", "reasoning": "Different reasoning trace."},
        ],
        row_index=42,
    )
    assert doc_holdout_key(c_a) == doc_holdout_key(c_b)
    assert is_eval_holdout_doc(c_a, holdout_mod=100, holdout_eval_buckets=2) == is_eval_holdout_doc(
        c_b, holdout_mod=100, holdout_eval_buckets=2
    )

    # Exact `TrainStream` resume under `split_role="train"`
    orig_raw_rows = mix_mod.raw_rows

    def _patched(spec, seed, epoch, skip=0, src_cfg=None, streaming=True):
        if spec.name in ("src_a", "src_b"):
            yield from docs[skip:]
            return
        yield from orig_raw_rows(spec, seed, epoch, skip=skip, src_cfg=src_cfg, streaming=streaming)

    monkeypatch.setattr(mix_mod, "raw_rows", _patched)
    tok = ToyTokenizer()
    dcfg = {
        "seq_len": 64,
        "dedupe": False,
        "sources": [
            {"name": "src_a", "hf_id": "local/a", "role": "pretrain", "format": "text", "weight": 0.5},
            {"name": "src_b", "hf_id": "local/b", "role": "pretrain", "format": "text", "weight": 0.5},
        ],
    }
    s1 = build_train_stream(dcfg, tok, seed=5, split_role="train", holdout_mod=h_mod, holdout_eval_buckets=h_eval)
    for _ in range(4):
        next(s1)
    sd = json.loads(json.dumps(s1.state_dict()))
    expected = [next(s1) for _ in range(6)]
    s2 = build_train_stream(dcfg, tok, seed=5, split_role="train", holdout_mod=h_mod, holdout_eval_buckets=h_eval)
    s2.load_state_dict(sd)
    got = [next(s2) for _ in range(6)]
    for x, y in zip(expected, got):
        assert torch.equal(x["input_ids"], y["input_ids"])
        assert torch.equal(x["labels"], y["labels"])


def test_ql60_003_eval_quick_ood_sources(monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> None:
    """`eval_quick.ood_sources` evaluates OOD validation separately (`eval/ood_loss_d*`) with 0 train overlap."""
    import qloop.data.mixture as mix_mod

    id_docs = _make_eight_docs()
    ood_docs = [
        {"id": f"ood-{i}", "text": f"OOD_UNIQUE_SENTINEL_{i} " + "disjoint validation prose. " * 4}
        for i in range(4)
    ]
    h_mod, h_eval = _pick_nonempty_holdout_params(id_docs)
    orig_raw_rows = mix_mod.raw_rows

    def _patched(spec, seed, epoch, skip=0, src_cfg=None, streaming=True):
        if spec.name == "id_ds":
            yield from id_docs[skip:]
            return
        if spec.name == "ood_ds":
            yield from ood_docs[skip:]
            return
        yield from orig_raw_rows(spec, seed, epoch, skip=skip, src_cfg=src_cfg, streaming=streaming)

    monkeypatch.setattr(mix_mod, "raw_rows", _patched)

    cfg = {
        "seed": 3,
        "stage": "ood_eval_test",
        "model": {"name": "toy", "dtype": "float32", "device_map": "cpu", "toy": {"seed": 0}},
        "loop": {"enabled": True, "prelude_end": 4, "core_end": 12, "num_passes": 2, "target_virtual_layers": 24},
        "train": {
            "steps": 1,
            "log_every": 1,
            "ckpt_every": 10,
            "trainable": [r"input_layernorm"],
            "optim": {"lr": 1e-3},
        },
        "data": {
            "seq_len": 64,
            "holdout_mod": h_mod,
            "holdout_eval_buckets": h_eval,
            "sources": [{"name": "id_ds", "hf_id": "local/id", "role": "pretrain", "format": "text", "weight": 1.0}],
        },
        "eval_quick": {
            "n_batches": 2,
            "depths": [1, 2],
            "ood_sources": [
                {"name": "ood_ds", "hf_id": "local/ood", "role": "pretrain", "format": "text", "weight": 1.0}
            ],
        },
    }
    tr = LMTrainer(cfg, str(tmp_path / "ood_run"))
    res = tr.train()
    assert len(tr.eval_batches) >= 1
    assert len(tr.ood_eval_batches) == 2

    tok = tr.stack.tokenizer
    ood_text = "\n".join(tok.decode(b["input_ids"], skip_special_tokens=False) for b in tr.ood_eval_batches)
    assert "OOD_UNIQUE_SENTINEL_" in ood_text
    assert not any(m in ood_text for m in _EIGHT_DOC_MARKERS)

    ev = res["eval"]
    expected_eval_keys = (
        "eval/loss_d1",
        "eval/loss_d2",
        "eval/depth_gain",
        "eval/ood_loss_d1",
        "eval/ood_loss_d2",
        "eval/ood_depth_gain",
    )
    for key in expected_eval_keys:
        assert key in ev and torch.isfinite(torch.tensor(ev[key]))


def test_ql60_003_default_dedupe_tiny_finite_eval_terminates_without_hang(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    """Blocker 2a regression: default `dedupe=True` on a tiny finite dataset with `< seq_len` unique eval tokens
    flushes buffered eval tokens and terminates finitely (`< 1s`), NEVER spinning infinitely.
    """
    import qloop.data.mixture as mix_mod

    docs = _make_eight_docs()
    h_mod, h_eval = _pick_nonempty_holdout_params(docs)
    orig_raw_rows = mix_mod.raw_rows

    def _patched(spec, seed, epoch, skip=0, src_cfg=None, streaming=True):
        if spec.name == "finite8_tiny":
            yield from docs[skip:]
            return
        yield from orig_raw_rows(spec, seed, epoch, skip=skip, src_cfg=src_cfg, streaming=streaming)

    monkeypatch.setattr(mix_mod, "raw_rows", _patched)

    # `seq_len=2048` is much larger than the total tokens of the 1-2 held-out docs -> packer has `< seq_len` tokens!
    dcfg = {
        "seq_len": 2048,
        "holdout_mod": h_mod,
        "holdout_eval_buckets": h_eval,
        "sources": [
            {"name": "finite8_tiny", "hf_id": "local/finite8_tiny", "role": "pretrain", "format": "text", "weight": 1.0}
        ],
    }

    # 1. Direct `build_train_stream` with default `dedupe=True` and `single_pass=False`
    tok = ToyTokenizer()
    t0 = time.monotonic()
    direct_eval = build_train_stream(dcfg, tok, seed=0, split_role="eval")
    flushed_rows = list(direct_eval)
    elapsed_direct = time.monotonic() - t0
    assert elapsed_direct < 1.0, f"direct eval stream took too long: {elapsed_direct:.2f}s"
    assert len(flushed_rows) == 1
    assert flushed_rows[0]["input_ids"].shape == (2048,)
    assert flushed_rows[0]["n_loss_tokens"] > 0

    # 2. `LMTrainer._eval_set()` with default `dedupe=True` (`single_pass=True`) + train/eval 0 overlap
    cfg = {
        "seed": 0,
        "stage": "tiny_dedupe_test",
        "model": {"name": "toy", "dtype": "float32", "device_map": "cpu", "toy": {"seed": 0}},
        "loop": {"enabled": False},
        "train": {"steps": 1, "trainable": [r"input_layernorm"], "optim": {"lr": 1e-3}},
        "data": dcfg,
        "eval_quick": {"n_batches": 4, "depths": [1]},
    }
    tr = LMTrainer(cfg, str(tmp_path / "tiny_dedupe"))
    t1 = time.monotonic()
    ev_batches = tr._eval_set()
    tr_batches = list(tr._build_stream())
    elapsed_tr = time.monotonic() - t1
    assert elapsed_tr < 1.0, f"LMTrainer stream collection took too long: {elapsed_tr:.2f}s"
    assert len(ev_batches) == 1
    assert len(tr_batches) == 1
    ev_txt = tok.decode(ev_batches[0]["input_ids"], skip_special_tokens=False)
    tr_txt = tok.decode(tr_batches[0]["input_ids"], skip_special_tokens=False)
    ev_markers = {m for m in _EIGHT_DOC_MARKERS if m in ev_txt}
    tr_markers = {m for m in _EIGHT_DOC_MARKERS if m in tr_txt}
    assert ev_markers and tr_markers and (ev_markers & tr_markers == set())


def test_ql60_003_zero_eval_bucket_tiny_dataset_multisource_and_toy_cot_no_crash(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any, caplog: pytest.LogCaptureFixture
) -> None:
    """Blocker 2b regression:
    1) 8-doc finite dataset under default 100/2 (0 docs in eval bucket) warns and returns `[]` without crashing train().
    2) Multi-source mixture where one tiny source has 0 eval docs skips it cleanly and evaluates the non-empty source.
    3) Synthetic `toy_cot` with `toy_rows=8` (default config) is exempted alongside `toy_text` and runs cleanly.
    """
    import qloop.data.mixture as mix_mod

    docs_8 = _make_eight_docs()
    # Confirm 0 of `_make_eight_docs()` land in default 100/2 eval buckets
    assert not any(
        is_eval_holdout_doc(Example("finite8_zero", "text", text=d["text"]), holdout_mod=100, holdout_eval_buckets=2)
        for d in docs_8
    )
    docs_big = [
        {"id": f"big-{i}", "text": f"BIG_CORPUS_DOC_{i:04d} " + "sufficient text payload for eval holdout. " * 3}
        for i in range(200)
    ]
    orig_raw_rows = mix_mod.raw_rows

    def _patched(spec, seed, epoch, skip=0, src_cfg=None, streaming=True):
        if spec.name == "finite8_zero":
            yield from docs_8[skip:]
            return
        if spec.name == "big_corpus":
            yield from docs_big[skip:]
            return
        yield from orig_raw_rows(spec, seed, epoch, skip=skip, src_cfg=src_cfg, streaming=streaming)

    monkeypatch.setattr(mix_mod, "raw_rows", _patched)

    # 1. Single 8-doc source with default 100/2 (0 eval docs) + default dedupe=True -> warns, returns [], train() OK!
    cfg_zero = {
        "seed": 1,
        "stage": "zero_eval_8doc",
        "model": {"name": "toy", "dtype": "float32", "device_map": "cpu", "toy": {"seed": 1}},
        "loop": {"enabled": False},
        "train": {
            "steps": 1,
            "log_every": 1,
            "ckpt_every": 10,
            "trainable": [r"input_layernorm"],
            "optim": {"lr": 1e-3},
        },
        "data": {
            "seq_len": 64,
            "sources": [
                {"name": "finite8_zero", "hf_id": "local/f8z", "role": "pretrain", "format": "text", "weight": 1.0}
            ],
        },
        "eval_quick": {"n_batches": 2, "depths": [1]},
    }
    tr_zero = LMTrainer(cfg_zero, str(tmp_path / "zero_eval"))
    with caplog.at_level(logging.WARNING, logger="qloop"):
        res_zero = tr_zero.train()
    assert res_zero["step"] == 1
    assert tr_zero.eval_batches == []
    assert any("0/2 batches" in rec.message for rec in caplog.records)

    # 2. Multi-source `[big_corpus, finite8_zero]` under default 100/2 -> skips empty source, returns eval batches!
    cfg_mix = copy.deepcopy(cfg_zero)
    cfg_mix["data"]["sources"] = [
        {"name": "big_corpus", "hf_id": "local/big", "role": "pretrain", "format": "text", "weight": 0.5},
        {"name": "finite8_zero", "hf_id": "local/f8z", "role": "pretrain", "format": "text", "weight": 0.5},
    ]
    tr_mix = LMTrainer(cfg_mix, str(tmp_path / "mix_eval"))
    mix_eval = tr_mix._eval_set()
    assert len(mix_eval) == 2
    tok = tr_mix.stack.tokenizer
    mix_eval_txt = "\n".join(tok.decode(b["input_ids"], skip_special_tokens=False) for b in mix_eval)
    assert "BIG_CORPUS_DOC_" in mix_eval_txt

    # 3. Synthetic `toy_cot` with `toy_rows=8` under default config (no explicit `holdout_mod`) -> exempted & runs!
    cfg_cot = {
        "seed": 2,
        "stage": "toy_cot_small",
        "model": {"name": "toy", "dtype": "float32", "device_map": "cpu", "toy": {"seed": 2}},
        "loop": {"enabled": False},
        "train": {
            "steps": 1,
            "log_every": 1,
            "ckpt_every": 10,
            "trainable": [r"input_layernorm"],
            "optim": {"lr": 1e-3},
        },
        "data": {"seq_len": 32, "sources": [{"name": "toy_cot", "weight": 1.0, "toy_rows": 8}]},
        "eval_quick": {"n_batches": 2, "depths": [1]},
    }
    tr_cot = LMTrainer(cfg_cot, str(tmp_path / "toy_cot8"))
    assert tr_cot._build_stream().split_role == "all"
    res_cot = tr_cot.train()
    assert res_cot["step"] == 1
    assert len(tr_cot.eval_batches) == 2


# ==============================================================================
# 2. QL60-009 / C1 — Valid-Token BPTT Normalization & Pure CE `loss/lm` Logging
# ==============================================================================

@pytest.mark.parametrize("use_fp64", [False, True])
def test_ql60_009_two_segment_1_vs_3_valid_token_lm_head_grad_parity(use_fp64: bool) -> None:
    """Review 1 QL60-009 repro: 2 segments (1 vs 3 valid tokens) match full-sequence CE & lm_head grads."""
    torch.manual_seed(42)
    cfg = ToyConfig(
        vocab_size=48,
        hidden_size=32,
        num_hidden_layers=4,
        full_attention_interval=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        partial_rotary_factor=0.5,
    )
    model = ToyForCausalLM(cfg)
    if use_fp64:
        model = model.double()
    model.train()

    ids = torch.randint(0, cfg.vocab_size, (1, 16), generator=torch.Generator().manual_seed(7))
    labels = torch.full_like(ids, -100)
    # Segment 0 targets are `labels[:, 1:9]` -> exactly 1 valid token (index 4)
    labels[:, 4] = ids[:, 4]
    # Segment 1 targets are `labels[:, 9:16]` -> exactly 3 valid tokens (indices 10, 12, 15)
    labels[:, 10] = ids[:, 10]
    labels[:, 12] = ids[:, 12]
    labels[:, 15] = ids[:, 15]

    # Reference: exact un-segmented full-sequence valid-token-mean cross entropy
    model.zero_grad(set_to_none=True)
    full_out = model(ids)
    ref_ce = F.cross_entropy(
        full_out.logits[:, :-1].reshape(-1, cfg.vocab_size).float(),
        labels[:, 1:].reshape(-1),
        ignore_index=-100,
    )
    ref_ce.backward()
    ref_lm_head_grad = model.lm_head.weight.grad.detach().clone()

    # Segmented BPTT (`2 x 8`, full replay `replay_k=8` so KV states are exact)
    model.zero_grad(set_to_none=True)
    runner = SegmentedKVBPTTRunner(
        model,
        cfg=BPTTConfig(
            seg_len=8,
            n_segments=2,
            replay_k=8,
            replay_enabled=True,
            offload_prior_hidden=False,
            replay_indexer_weight=0.0,
        ),
    )
    res = runner.run_step(
        ids,
        labels=labels,
        backward=True,
        hssd_aux_weight=0.0,
        lsa_aux_weight=0.0,
        loop_aux_weight=0.0,
    )

    assert res.segment_valid_tokens == [1, 3]
    assert res.aux_loss == 0.0
    assert res.total_loss == pytest.approx(res.ce_loss, rel=1e-9, abs=1e-12)
    assert res.stats["bptt/valid_tokens"] == 4.0

    # Verify exact parity with true valid-token-weighted CE (and NOT naive 0.5*(seg0+seg1) average!)
    tol = 1e-6 if use_fp64 else 1e-5
    assert abs(res.ce_loss - float(ref_ce.detach())) < tol
    assert model.lm_head.weight.grad is not None
    torch.testing.assert_close(
        model.lm_head.weight.grad,
        ref_lm_head_grad,
        rtol=1e-6 if use_fp64 else 1e-4,
        atol=1e-6 if use_fp64 else 1e-5,
    )
    naive_seg_avg = 0.5 * (res.segment_ce_losses[0] + res.segment_ce_losses[1])
    assert abs(res.ce_loss - naive_seg_avg) > 1e-3, "fixture must distinguish valid-token weighting from naive 1/n_segs"

    # Edge cases: all-masked segment `[0, 3]`, `[1, 0]`, and completely masked `[0, 0]` produce zero NaN / div-by-zero
    for zero_case in ("seg0_only", "seg1_only", "all_masked"):
        lbl_z = torch.full_like(ids, -100)
        if zero_case == "seg0_only":
            lbl_z[:, 4] = ids[:, 4]
        elif zero_case == "seg1_only":
            lbl_z[:, 10] = ids[:, 10]
            lbl_z[:, 12] = ids[:, 12]
        model.zero_grad(set_to_none=True)
        rz = runner.run_step(ids, labels=lbl_z, backward=True, lsa_aux_weight=0.0, loop_aux_weight=0.0)
        assert torch.isfinite(torch.tensor(rz.ce_loss))
        assert torch.isfinite(torch.tensor(rz.aux_loss))
        assert torch.isfinite(torch.tensor(rz.total_loss))
        if zero_case == "all_masked":
            assert rz.ce_loss == 0.0 and rz.aux_loss == 0.0 and rz.total_loss == 0.0


def test_c1_bptt_trainer_logs_pure_ce_loss_lm_separate_from_loss_aux(tmp_path: Any) -> None:
    """C1 regression: BPTT `loss/lm` logs pure valid-token CE only (aux-invariant), separate from `loss/aux`."""
    def _make_cfg(lsa_w: float, replay_w: float, run_name: str) -> dict[str, Any]:
        return {
            "seed": 5,
            "stage": run_name,
            "model": {"name": "toy", "dtype": "float32", "device_map": "cpu", "toy": {"seed": 5}},
            "loop": {
                "enabled": True,
                "prelude_end": 4,
                "core_end": 12,
                "num_passes": 2,
                "target_virtual_layers": 24,
                "train_pass_probs": [0.0, 1.0],
                "checkpoint": False,
            },
            "lsa": {
                "enabled": True,
                "mode": "sparse",
                "distill_weight": 1.0,
                "index_dim": 8,
                "block_size": 16,
                "pool": 2,
                "avg_bucket_keys": 4,
                "max_bucket_keys": 8,
                "num_select": 2,
                "window": 16,
                "sinks": 4,
                "query_chunk": 16,
            },
            "bptt": {
                "enabled": True,
                "n_segments": 2,
                "seg_len": 16,
                "replay_k": 6,
                "offload_prior_hidden": False,
                "replay_indexer_weight": replay_w,
            },
            "data": {"seq_len": 32, "sources": [{"name": "toy_text", "weight": 1.0}]},
            "eval_quick": {"n_batches": 0},
            "train": {
                "steps": 1,
                "micro_batch": 1,
                "grad_accum": 1,
                "log_every": 1,
                "ckpt_every": 10,
                "train_loop_adapters": True,
                "train_lsa": True,
                "trainable": [r"self_attn\.(k_proj|v_proj|o_proj)"],
                "aux_weights": {"lsa_kl": lsa_w, "loop_aux": 1.0},
                "optim": {"lr": 1e-3, "fused": False, "impl": "builtin"},
            },
        }

    tr_zero = LMTrainer(_make_cfg(lsa_w=0.0, replay_w=0.0, run_name="bptt_c1_zero"), str(tmp_path / "zero"))
    res_zero = tr_zero.train()

    tr_heavy = LMTrainer(_make_cfg(lsa_w=5.0, replay_w=2.0, run_name="bptt_c1_heavy"), str(tmp_path / "heavy"))
    res_heavy = tr_heavy.train()

    m0 = res_zero["last"]
    m1 = res_heavy["last"]

    # Pure CE `loss/lm` on step 1 MUST be identical regardless of massive aux weights!
    assert m0["loss/lm"] == pytest.approx(m1["loss/lm"], rel=1e-5, abs=1e-5)
    assert m0["loss/aux"] == pytest.approx(0.0, abs=1e-7)
    assert m1["loss/aux"] > 0.0
    assert m1["loss/total"] == pytest.approx(m1["loss/lm"] + m1["loss/aux"], rel=1e-5, abs=1e-5)
    assert m1["loss/total"] > m1["loss/lm"]
    # Backward-compatible BPTT telemetry keys preserved
    assert m1["loss/bptt_segments"] == 2.0
    assert "loss/bptt_replayed_tokens" in m1
    assert "loss/bptt_offload_to_cpu" in m1


# ==============================================================================
# 3. QL60-004 — Exact 5-Step Resume, Private RNG & Explicit Weights-Only Mode
# ==============================================================================

def _resume_cfg(steps: int, ckpt_every: int, *, reset_lr: bool = False) -> dict[str, Any]:
    return {
        "seed": 17,
        "stage": "ql60_004_resume",
        "model": {"name": "toy", "dtype": "float32", "device_map": "cpu", "toy": {"seed": 17}},
        "loop": {
            "enabled": True,
            "prelude_end": 4,
            "core_end": 12,
            "num_passes": 3,
            "target_virtual_layers": 32,
            "train_pass_probs": [0.34, 0.33, 0.33],
            "checkpoint": False,
        },
        "data": {"seq_len": 32, "sources": [{"name": "toy_text", "weight": 1.0}]},
        "eval_quick": {"n_batches": 0},
        "train": {
            "steps": steps,
            "micro_batch": 1,
            "grad_accum": 1,
            "log_every": 1,
            "ckpt_every": ckpt_every,
            "keep_last": 5,
            "prefetch_batches": 2,
            "reset_lr_on_weights_only": reset_lr,
            "train_loop_adapters": True,
            "trainable": [r"input_layernorm"],
            "optim": {"lr": 2e-3, "warmup_steps": 2, "fused": False, "impl": "builtin"},
        },
    }


def _read_metrics(run_dir: str) -> list[dict[str, Any]]:
    with open(os.path.join(run_dir, "metrics.jsonl"), encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def test_ql60_004_exact_5_step_resume_and_private_rng_depth_parity(tmp_path: Any) -> None:
    """Review 1 QL60-004 repro: interrupted+resumed 5-step run matches uninterrupted depths, RNG, losses & weights."""
    dir_orig = str(tmp_path / "orig_5step")
    tr_orig = LMTrainer(_resume_cfg(steps=5, ckpt_every=2), dir_orig)
    res_orig = tr_orig.train()
    assert res_orig["step"] == 5
    orig_recs = _read_metrics(dir_orig)
    orig_depths = [r["depth"] for r in orig_recs]
    orig_losses = [r["loss/lm"] for r in orig_recs]
    assert len(orig_depths) == 5

    # Verify `state.json` serialized `trainer_rng`
    ckpt2 = os.path.join(dir_orig, "checkpoints", "step_0000002")
    with open(os.path.join(ckpt2, "state.json"), encoding="utf-8") as f:
        st2 = json.load(f)
    assert "trainer_rng" in st2 and isinstance(st2["trainer_rng"], list) and len(st2["trainer_rng"]) == 3

    # Copy step-2 checkpoint into a brand-new run_dir and resume to step 5
    dir_res = str(tmp_path / "resumed_5step")
    os.makedirs(os.path.join(dir_res, "checkpoints"), exist_ok=True)
    shutil.copytree(ckpt2, os.path.join(dir_res, "checkpoints", "step_0000002"))
    with open(os.path.join(dir_res, "checkpoints", "LATEST"), "w", encoding="utf-8") as f:
        f.write("step_0000002")

    tr_res = LMTrainer(_resume_cfg(steps=5, ckpt_every=10), dir_res)
    res_res = tr_res.train()
    assert res_res["step"] == 5
    assert tr_res.resume_mode == "exact"
    assert res_res["resume_mode"] == "exact"

    res_recs = _read_metrics(dir_res)
    res_depths = [r["depth"] for r in res_recs]
    res_losses = [r["loss/lm"] for r in res_recs]

    # Exact depth sequence on resumed steps 3, 4, 5 AND exact final `self.rng.getstate()`!
    assert res_depths == orig_depths[2:], (res_depths, orig_depths)
    assert tr_orig.rng.getstate() == tr_res.rng.getstate()
    for a, b in zip(orig_losses[2:], res_losses):
        assert b == pytest.approx(a, rel=1e-5, abs=1e-5)
    for name, p_orig in tr_orig.stack.named_trainable.items():
        torch.testing.assert_close(tr_res.stack.named_trainable[name], p_orig, rtol=1e-5, atol=1e-6)


def test_ql60_004_weights_only_resume_syncs_lr_and_honors_explicit_reset(
    tmp_path: Any, caplog: pytest.LogCaptureFixture
) -> None:
    """Missing `optim.pt` (e.g. weights-only Hub resume @ step 450/600) sets `resume_mode='weights_only'`,
    warns clearly, and syncs `opt.step_count=450` (`lr_factor` at step 450, NOT warmup reset!) unless
    `train.reset_lr_on_weights_only=True`.
    """
    run_dir = str(tmp_path / "s2_v2_450")
    cfg = _resume_cfg(steps=600, ckpt_every=100)
    cfg["train"]["optim"]["warmup_steps"] = 50
    cfg["train"]["optim"]["min_lr_ratio"] = 0.1

    tr_save = LMTrainer(cfg, run_dir)
    # Advance private rng a few times & save at step 450
    for _ in range(7):
        tr_save.rng.random()
    expected_rng_state = tr_save.rng.getstate()
    tr_save.step = 450
    tr_save.tokens_seen = 450 * 32
    ckpt_path = tr_save.save(completed=False)

    # Simulate fresh-session weights-only Hub pull: remove `optim.pt` (and `rng.pt`!)
    os.remove(os.path.join(ckpt_path, "optim.pt"))
    os.remove(os.path.join(ckpt_path, "rng.pt"))

    expected_sched = WarmupCosine(total_steps=600, warmup_steps=50, min_ratio=0.1)
    expected_lrf_450 = expected_sched(450)
    assert 0.20 < expected_lrf_450 < 0.30  # ~0.2553, far below peak warmup 0.974!

    # 1. Default safe weights-only resume: syncs `opt.step_count = 450` & group LRs, restores `self.rng`
    tr_wo = LMTrainer(copy.deepcopy(cfg), run_dir)
    with caplog.at_level(logging.WARNING, logger="qloop"):
        tr_wo.maybe_resume()
    assert tr_wo.resume_mode == "weights_only"
    assert tr_wo.step == 450
    assert tr_wo.opt.step_count == 450
    assert tr_wo.opt.lr_factor == pytest.approx(expected_lrf_450, rel=1e-9)
    for name, ent in tr_wo.opt.entries.items():
        assert tr_wo.opt.current_lr(name) == pytest.approx(ent.base_lr * expected_lrf_450, rel=1e-9)
    assert tr_wo.rng.getstate() == expected_rng_state
    assert any("weights-only" in rec.message and "optim.pt" in rec.message for rec in caplog.records)

    # 2. Explicit opt-in `reset_lr_on_weights_only: True`: leaves `opt.step_count == 0`
    cfg_reset = copy.deepcopy(cfg)
    cfg_reset["train"]["reset_lr_on_weights_only"] = True
    tr_reset = LMTrainer(cfg_reset, run_dir)
    tr_reset.maybe_resume()
    assert tr_reset.resume_mode == "weights_only"
    assert tr_reset.step == 450
    assert tr_reset.opt.step_count == 0
    assert tr_reset.opt.lr_factor == pytest.approx(expected_sched(0), rel=1e-9)


# ==============================================================================
# 4. QL60-010 — Model-Bound Evaluation Cache Identity
# ==============================================================================

_AIME_ITEMS = [
    {"problem_idx": 1, "problem": "What is 6*7?", "answer": "42"},
    {"problem_idx": 2, "problem": "What is 40+2?", "answer": "42"},
    {"problem_idx": 3, "problem": "What is 3+4?", "answer": "7"},
]
_LOOP_SPEC = {"prelude_end": 4, "core_end": 12, "num_passes": 2, "target_virtual_layers": 24}


def test_ql60_010_persists_identity_and_allows_matching_resume(tmp_path: Any) -> None:
    """`run_benchmark` persists canonical `run_identity` and allows clean resume when identity matches."""
    ckpt_dir = tmp_path / "ckpt_step_100"
    ckpt_dir.mkdir()
    (ckpt_dir / "trainable.pt").write_bytes(b"weights-v1")
    (ckpt_dir / "state.json").write_text('{"step": 100}', encoding="utf-8")

    out_dir = str(tmp_path / "eval_out")
    gen = Generator(ScriptedLM.from_text(r"so \boxed{42}"), ByteTokenizer())

    s1 = run_benchmark(
        "aime_2026",
        gen,
        limit=2,
        max_new_tokens=32,
        out_dir=out_dir,
        offline_items=_AIME_ITEMS,
        model_id="qwen36-loop64-s2",
        checkpoint_path=str(ckpt_dir),
    )
    rid = s1["run_identity"]
    assert set(rid.keys()) == {"model_id", "checkpoint_sha", "active_passes", "config_hash", "dataset_rev"}
    assert rid["model_id"] == "qwen36-loop64-s2"
    assert rid["checkpoint_sha"]

    cfg_disk = json.loads((tmp_path / "eval_out" / "aime_2026" / "run_config.json").read_text(encoding="utf-8"))
    assert cfg_disk["run_identity"] == rid

    # Matching model + checkpoint resumes cleanly on limit=3
    calls_before = gen.model.forward_calls
    s2 = run_benchmark(
        "aime_2026",
        gen,
        limit=3,
        max_new_tokens=32,
        out_dir=out_dir,
        offline_items=_AIME_ITEMS,
        model_id="qwen36-loop64-s2",
        checkpoint_path=str(ckpt_dir),
    )
    assert s2["depths"]["native"]["n_items"] == 3
    assert gen.model.forward_calls > calls_before


def test_ql60_010_refuses_cross_model_checkpoint_and_legacy_cache_reuse(tmp_path: Any) -> None:
    """Silent cache reuse across different models, checkpoints, weights, active_passes, or legacy dirs raises."""
    out_dir = str(tmp_path / "shared_eval")
    gen_a = build_toy_generator(seed=0, loop_spec=_LOOP_SPEC)
    run_benchmark(
        "aime_2026",
        gen_a,
        limit=1,
        depths=[1, 2],
        max_new_tokens=4,
        out_dir=out_dir,
        offline_items=_AIME_ITEMS,
    )

    # 1. Different model weights (different seed) -> MUST raise ValueError!
    gen_b = build_toy_generator(seed=1, loop_spec=_LOOP_SPEC)
    with pytest.raises(ValueError, match="identity mismatch"):
        run_benchmark(
            "aime_2026",
            gen_b,
            limit=2,
            depths=[1, 2],
            max_new_tokens=4,
            out_dir=out_dir,
            offline_items=_AIME_ITEMS,
        )

    # 1b. Different `active_passes` on same generator -> MUST raise ValueError!
    gen_a.loop_controller.set_active_passes(1)
    with pytest.raises(ValueError, match="active_passes"):
        run_benchmark(
            "aime_2026",
            gen_a,
            limit=2,
            depths=[1, 2],
            max_new_tokens=4,
            out_dir=out_dir,
            offline_items=_AIME_ITEMS,
        )
    gen_a.loop_controller.set_active_passes(2)

    # 2. Same model after in-place weight mutation (simulating updated checkpoint) -> MUST raise ValueError!
    with torch.no_grad():
        next(gen_a.model.parameters()).add_(1.0)
    with pytest.raises(ValueError, match="identity mismatch"):
        run_benchmark(
            "aime_2026",
            gen_a,
            limit=2,
            depths=[1, 2],
            max_new_tokens=4,
            out_dir=out_dir,
            offline_items=_AIME_ITEMS,
        )

    # 3. Different ScriptedLM outputs / model_id / active_passes / dataset_rev -> ALL raise ValueError!
    out_script = str(tmp_path / "script_eval")
    run_benchmark(
        "aime_2026",
        Generator(ScriptedLM.from_text(r"\boxed{42}"), ByteTokenizer()),
        limit=1,
        max_new_tokens=16,
        out_dir=out_script,
        offline_items=_AIME_ITEMS,
    )
    with pytest.raises(ValueError, match="identity mismatch"):
        run_benchmark(
            "aime_2026",
            Generator(ScriptedLM.from_text(r"\boxed{99}"), ByteTokenizer()),
            limit=2,
            max_new_tokens=16,
            out_dir=out_script,
            offline_items=_AIME_ITEMS,
        )
    with pytest.raises(ValueError, match="identity mismatch"):
        run_benchmark(
            "aime_2026",
            Generator(ScriptedLM.from_text(r"\boxed{42}"), ByteTokenizer()),
            limit=2,
            max_new_tokens=16,
            out_dir=out_script,
            offline_items=_AIME_ITEMS,
            dataset_rev="v2-patched",
        )

    # 4. Explicit `allow_mixed_cache=True` overrides the guard cleanly
    mixed = run_benchmark(
        "aime_2026",
        Generator(ScriptedLM.from_text(r"\boxed{99}"), ByteTokenizer()),
        limit=2,
        max_new_tokens=16,
        out_dir=out_script,
        offline_items=_AIME_ITEMS,
        allow_mixed_cache=True,
    )
    assert mixed["depths"]["native"]["n_items"] == 2

    # 5. Legacy cache missing `run_identity` in `run_config.json` -> refuses by default, allows with flag
    legacy_dir = tmp_path / "legacy_eval"
    bdir = legacy_dir / "aime_2026"
    bdir.mkdir(parents=True)
    (bdir / "run_config.json").write_text(json.dumps({"benchmark": "aime_2026"}), encoding="utf-8")
    with pytest.raises(ValueError, match="lacks `run_identity`"):
        run_benchmark(
            "aime_2026",
            Generator(ScriptedLM.from_text(r"\boxed{42}"), ByteTokenizer()),
            limit=1,
            max_new_tokens=16,
            out_dir=str(legacy_dir),
            offline_items=_AIME_ITEMS,
        )

    # 6. `run_suite` surfaces cross-model cache mismatch as error (and raises when fail_fast=True)
    suite_dir = str(tmp_path / "suite_eval")
    suite_cfg = {
        "benchmarks": [{"name": "aime_2026", "limit": 1}],
        "defaults": {"max_new_tokens": 16, "depths": [1]},
        "fail_fast": True,
    }
    run_suite(
        suite_cfg,
        Generator(ScriptedLM.from_text(r"\boxed{42}"), ByteTokenizer()),
        suite_dir,
        offline_items={"aime_2026": _AIME_ITEMS},
    )
    with pytest.raises(ValueError, match="identity mismatch"):
        run_suite(
            suite_cfg,
            Generator(ScriptedLM.from_text(r"\boxed{99}"), ByteTokenizer()),
            suite_dir,
            offline_items={"aime_2026": _AIME_ITEMS},
        )
