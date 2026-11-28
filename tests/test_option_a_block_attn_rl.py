"""Comprehensive CPU verification suite for Option A (`40L` `d=1`) Block-Attention Calibrator & GRPO-lambda RL.

Covers:
  1. Exact chunked FP32 KL + CE (`chunked_kl_ce_loss`) forward & backward parity vs naive FP32.
  2. `BlockAttentionCalibrator` end-to-end on `ToyForCausalLM` (`loop.enabled=False` Option A AND
     `loop.enabled=True` Option B depth-agnostic compatibility):
       * Non-zero finite gradients into `self_attn.{q,k,v,o}_proj`, `q_norm`/`k_norm`, `gdn2_adapter`,
         `shared_expert`, and `norms` (and `loop_adapters` when `loop.enabled=True`).
       * Frozen MoE experts SHA-256 bit-identity (`verify_frozen_experts()`).
       * Measurable decrease in `K0` Arm (c) NLL gap / KL vs Contiguous Oracle over short calibration.
       * Zero-leakage episode disjointness (`synthetic_seed_offset >= 10000` vs `K0` `seed=42`).
       * `drop_inactive_loop_adapters=True` cleanly strips `loop_adapters.*` when migrating
         `s2_v2` `88L` -> `40L` `d=1`, while default `False` preserves 100% legacy parity.
       * CLI `stage_type: block_attn_calib` dispatch (`qloop.cli.main`) and exact checkpoint save/resume.
  3. `GRPO-lambda` accuracy-gated overlong + `concise_bonus` in `rl.py`:
       * Legacy `accuracy_gated=False` exact parity.
       * `accuracy_gated=True` gives `0.0` overlong penalty on `0%`-pass groups and full penalty on `100%`-pass groups.
       * `concise_bonus > 0` boosts shorter verified-correct samples only when `pass_rate >= min_pass_rate`,
         and NEVER boosts wrong completions (`r <= 0`).
       * `GRPOTrainer` end-to-end with `loop.enabled=False`, `slap_target_groups: [router, shared_expert]`,
         and regression `gate()` (`depth: 1` on non-looped stack).
  4. Shipped Option A stage configs (`s3_opt_a_block_attn_calib.yaml`, `s4_opt_a_d1_rl.yaml`):
       * Load via `load_config()`, zero `opus46` sources, zero `HBM` mentions, pruned `GDN-2-1M`,
         `D-LSA 1.0x` (`num_query_passes=1`), and clean loading via both `sprint2_audit._load_arm_stack`
         and `hub.load_qloop_from_hub`.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
import torch
import torch.nn.functional as F
import yaml

from qloop.cli import main as cli_main
from qloop.config import load_config
from qloop.eval.generate import GenOut
from qloop.eval.kill_gate_k0 import K0Example, run_k0_suite
from qloop.eval.sprint2_audit import _load_arm_stack, _sync_cfg_to_checkpoint_payload
from qloop.hub import HubSyncManager, load_qloop_from_hub
from qloop.model.experts import param_group_of
from qloop.model.gdn2 import GDN2Config
from qloop.model.lsa import LSAConfig
from qloop.train.block_attn_calib import (
    BlockAttentionCalibrator,
    BlockAttnCalibConfig,
    BlockEpisode,
    ModelK0Evaluator,
    build_disjoint_k0_eval_examples,
    build_synthetic_calib_episode,
    chunked_kl_ce_loss,
)
from qloop.train.checkpoint import CheckpointManager
from qloop.train.rl import GRPOTrainer, _Seq
from qloop.train.stack import build_stack
from qloop.train.trainer import LMTrainer


@dataclass
class _CommitEntry:
    commit_id: str
    title: str
    created_at: str


class _MockHfApi:
    """Minimal in-memory mock of `huggingface_hub.HfApi` for offline Hub sync tests."""

    def __init__(self) -> None:
        self.repos: dict[str, dict[str, Any]] = {}
        self.files: dict[tuple[str, str], bytes] = {}
        self.commits: list[_CommitEntry] = []

    def create_repo(self, repo_id: str, repo_type: str = "model", private: bool = False, exist_ok: bool = True) -> None:
        self.repos.setdefault(repo_id, {"repo_type": repo_type, "private": private})

    def list_repo_files(self, repo_id: str, repo_type: str = "model") -> list[str]:
        return sorted(path for (rid, path) in self.files if rid == repo_id)

    def list_repo_commits(self, repo_id: str, repo_type: str = "model") -> list[_CommitEntry]:
        return list(reversed(self.commits))

    def commit_bundle(self, repo_id: str, additions: list[tuple[str, str | bytes]], commit_message: str) -> str:
        for repo_path, src in additions:
            if isinstance(src, (bytes, bytearray)):
                payload = bytes(src)
            else:
                with open(src, "rb") as f:
                    payload = f.read()
            self.files[(repo_id, repo_path)] = payload
        oid = f"optablock{len(self.commits) + 1:04d}"
        self.commits.append(_CommitEntry(oid, commit_message, f"2026-11-28T00:00:{len(self.commits):02d}Z"))
        return oid

    def download_file(self, repo_id: str, path_in_repo: str, dst_path: str) -> bool:
        key = (repo_id, path_in_repo)
        if key not in self.files:
            return False
        os.makedirs(os.path.dirname(os.path.abspath(dst_path)), exist_ok=True)
        with open(dst_path, "wb") as f:
            f.write(self.files[key])
        return True

ROOT = Path(__file__).resolve().parent.parent


def _toy_opt_a_block_attn_cfg(
    *,
    steps: int = 3,
    loop_enabled: bool = False,
    loop_checkpoint: bool = False,
    synthetic_mix_prob: float = 0.5,
    contiguous_lm_every: int = 2,
    eval_k0_examples: int = 2,
    eval_k0_mid_examples: int = 2,
    eval_every: int = 0,
    ckpt_every: int = 10,
    lr: float = 2e-3,
    init_from: str | None = None,
    drop_inactive_loop_adapters: bool = False,
    include_data: bool = True,
) -> dict[str, Any]:
    trainable = [
        r"self_attn\.(q_proj|k_proj|v_proj|o_proj|q_norm|k_norm)",
        r"shared_expert\.",
        r"input_layernorm|post_attention_layernorm",
    ]
    allowed = ["gdn2_adapters", "lsa_indexers", "attention", "shared_expert", "norms"]
    if loop_enabled:
        allowed.append("loop_adapters")
    cfg: dict[str, Any] = {
        "seed": 7,
        "stage": "s3_opt_a_toy_calib",
        "stage_type": "block_attn_calib",
        "run": {"name": "s3_opt_a_toy_calib"},
        "model": {
            "name": "toy",
            "dtype": "float32",
            "device_map": "cpu",
            "toy": {
                "seed": 7,
                "num_hidden_layers": 8,
                "hidden_size": 32,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "head_dim": 64,
                "partial_rotary_factor": 0.25,
                "rope_theta": 10000.0,
                "linear_num_heads": 2,
                "linear_head_dim": 8,
                "num_experts": 4,
                "num_experts_per_tok": 2,
                "moe_intermediate_size": 16,
                "shared_expert_intermediate_size": 16,
            },
        },
        "loop": (
            {
                "enabled": True,
                "prelude_end": 2,
                "core_end": 6,
                "num_passes": 2,
                "target_virtual_layers": 12,
                "kv_share": True,
                "checkpoint": loop_checkpoint,
                "train_pass_probs": [0.0, 1.0],
            }
            if loop_enabled
            else {"enabled": False}
        ),
        "gdn2": {
            "enabled": True,
            "rank": 8,
            "chunk_size": 8,
            "decay_log_min": -3.0,
            "decay_log_max": 0.5,
            "qed_enabled": False,
            "neg_eigval_max_boost": 0.0,
            "backend": "pytorch",
        },
        "hssd": {"enabled": False},
        "lsa": {
            "enabled": True,
            "mode": "sparse",
            "compact_active_slots": True,
            "max_selected_keys": 16,
            "num_query_passes": 2 if loop_enabled else 1,
            "index_dim": 8,
            "index_on": "nope",
            "pool": 2,
            "block_size": 8,
            "avg_bucket_keys": 2,
            "max_bucket_keys": 4,
            "num_select": 2,
            "window": 8,
            "sinks": 2,
            "search": "flat",
            "temperature": 0.1,
            "query_chunk": 16,
        },
        "experts": {"enabled": False},
        "train": {
            "init_from": init_from,
            "drop_inactive_loop_adapters": drop_inactive_loop_adapters,
            "steps": steps,
            "micro_batch": 1,
            "grad_accum": 1,
            "log_every": 1,
            "ckpt_every": ckpt_every,
            "eval_every": eval_every,
            "sha256_frozen_experts": True,
            "train_loop_adapters": loop_enabled,
            "train_hssd": False,
            "train_gdn2": True,
            "train_lsa": True,
            "trainable": trainable,
            "allowed_groups": allowed,
            "aux_weights": {},
            "optim": {
                "lr": lr,
                "warmup_steps": 1,
                "fused": False,
                "max_grad_norm": 1.0,
                "apollo": {"min_dim": 128},
                "lr_mult": {"gdn2_adapter": 5.0, "lsa_indexer": 2.0},
            },
        },
        "block_attn": {
            "kl_weight": 0.7,
            "ce_weight": 0.3,
            "kl_temperature": 1.0,
            "ce_chunk_size": 4,
            "contiguous_lm_every": contiguous_lm_every,
            "n_sessions": 2,
            "session_len": 16,
            "suffix_len": 8,
            "live_sink_len": 2,
            "sink_tokens": 2,
            "rotary_dim": 16,
            "rope_theta": 10000.0,
            "synthetic_mix_prob": synthetic_mix_prob,
            "synthetic_seed_offset": 10000,
            "eval_k0_examples": eval_k0_examples,
            "eval_k0_mid_examples": eval_k0_mid_examples,
        },
    }
    if include_data:
        cfg["data"] = {
            "seq_len": 48,
            "respect_doc_boundaries": True,
            "sources": [{"name": "toy_text", "weight": 1.0}],
        }
    return cfg


# ==============================================================================
# 1. Chunked KL + CE Exact Forward & Backward Parity
# ==============================================================================


@pytest.mark.parametrize("temperature", [1.0, 2.0])
@pytest.mark.parametrize("has_ignore", [False, True])
def test_chunked_kl_ce_loss_matches_naive_fp32_forward_and_backward(temperature: float, has_ignore: bool) -> None:
    torch.manual_seed(11)
    t_len, vocab = 11, 29
    z_s_base = torch.randn(1, t_len, vocab, dtype=torch.float32)
    z_t = torch.randn(1, t_len, vocab, dtype=torch.float32)
    targets = torch.randint(0, vocab, (1, t_len), dtype=torch.long)
    if has_ignore:
        targets[0, 2] = -100
        targets[0, 7] = -100

    # Chunked autograd path (chunk_size=3 -> non-divisible chunks!)
    z_s_chunk = z_s_base.clone().requires_grad_(True)
    loss_c, kl_c, ce_c = chunked_kl_ce_loss(
        z_s_chunk,
        z_t,
        targets,
        kl_weight=0.7,
        ce_weight=0.3,
        temperature=temperature,
        chunk_size=3,
    )
    loss_c.backward()

    # Naive full-tensor reference
    z_s_ref = z_s_base.clone().requires_grad_(True)
    log_pt = F.log_softmax(z_t.reshape(-1, vocab) / temperature, dim=-1)
    log_ps = F.log_softmax(z_s_ref.reshape(-1, vocab) / temperature, dim=-1)
    kl_ref = (log_pt.exp() * (log_pt - log_ps)).sum(dim=-1).mean()
    ce_ref = F.cross_entropy(z_s_ref.reshape(-1, vocab), targets.reshape(-1), ignore_index=-100)
    loss_ref = 0.7 * (temperature**2) * kl_ref + 0.3 * ce_ref
    loss_ref.backward()

    assert float(loss_c.detach()) == pytest.approx(float(loss_ref.detach()), rel=1e-5, abs=1e-6)
    assert kl_c == pytest.approx(float(kl_ref.detach()), rel=1e-5, abs=1e-6)
    assert ce_c == pytest.approx(float(ce_ref.detach()), rel=1e-5, abs=1e-6)
    assert z_s_chunk.grad is not None and z_s_ref.grad is not None
    torch.testing.assert_close(z_s_chunk.grad, z_s_ref.grad, rtol=1e-5, atol=1e-6)


# ==============================================================================
# 2. BlockAttentionCalibrator: Gradients, Frozen Experts, Option A & Option B
# ==============================================================================


@pytest.mark.parametrize("loop_enabled", [False, True])
def test_block_attention_calibrator_gradients_and_frozen_experts(tmp_path: Path, loop_enabled: bool) -> None:
    run_dir = str(tmp_path / f"calib_loop_{loop_enabled}")
    cfg = _toy_opt_a_block_attn_cfg(
        steps=2,
        loop_enabled=loop_enabled,
        loop_checkpoint=loop_enabled,  # verify loop checkpointing is safely disabled on cached suffix!
        synthetic_mix_prob=0.5,
        contiguous_lm_every=0,
        eval_k0_examples=2,
        eval_k0_mid_examples=2,
        lr=1e-2,
    )
    cal = BlockAttentionCalibrator(cfg, run_dir)
    assert cal.stack.frozen_experts_sha256 is not None
    sha_before = cal.stack.frozen_experts_sha256

    # Capture gradients right before optimizer steps on step 1 and step 2
    seen_nonzero_by_group: dict[str, set[str]] = {}
    orig_step = cal.opt.step

    def _spy_step() -> dict[str, float]:
        for name, p in cal.stack.named_trainable.items():
            if p.grad is not None:
                assert torch.isfinite(p.grad).all(), f"non-finite grad in {name}"
                if float(p.grad.abs().max()) > 0.0:
                    seen_nonzero_by_group.setdefault(param_group_of(name), set()).add(name)
        return orig_step()

    cal.opt.step = _spy_step  # type: ignore[method-assign]
    res = cal.train()

    assert res["step"] == 2
    assert cal.stack.verify_frozen_experts() == sha_before
    # Verify non-zero finite gradients flowed into every expected group!
    attn_grads = seen_nonzero_by_group.get("attention", set())
    for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
        assert any(proj in n for n in attn_grads), f"missing non-zero grad for {proj}: {attn_grads}"
    assert seen_nonzero_by_group.get("gdn2_adapters"), "missing non-zero grad for gdn2_adapter"
    assert seen_nonzero_by_group.get("shared_expert"), "missing non-zero grad for shared_expert"
    assert seen_nonzero_by_group.get("norms"), "missing non-zero grad for norms"
    if loop_enabled:
        assert seen_nonzero_by_group.get("loop_adapters"), "missing non-zero grad for loop_adapters in Option B"

    # Verify live K0 metrics logged in metrics.jsonl and state.json
    k0_res = res["k0"]
    for key in ("k0/arm_c_nll_gap", "k0/arm_c_em_retention", "k0/arm_c_tok_acc", "k0/verdict"):
        assert key in k0_res
    metrics_lines = [
        json.loads(line) for line in (Path(run_dir) / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert any("k0/arm_c_nll_gap" in rec and rec["step"] == 0 for rec in metrics_lines)
    assert any("k0/verdict" in rec and rec["step"] == 2 for rec in metrics_lines)
    state_json = json.loads((Path(res["checkpoint"]) / "state.json").read_text(encoding="utf-8"))
    assert "k0" in state_json and "k0/arm_c_nll_gap" in state_json["k0"]


def test_block_attention_calibrator_reduces_arm_c_nll_gap_vs_oracle(tmp_path: Path) -> None:
    """Verify short Block-Attention calibration measurably closes Arm (c) NLL gap vs Contiguous Oracle."""
    run_dir = str(tmp_path / "calib_reduce_gap")
    cfg = _toy_opt_a_block_attn_cfg(
        steps=10,
        loop_enabled=False,
        synthetic_mix_prob=1.0,
        contiguous_lm_every=0,
        eval_k0_examples=0,
        lr=8e-3,
        include_data=False,
    )
    cfg["block_attn"]["kl_weight"] = 0.7
    cfg["block_attn"]["ce_weight"] = 0.3
    cal = BlockAttentionCalibrator(cfg, run_dir)

    # Construct a fixed held-out multi-session evaluation set whose Arm (c) discrepancy we track
    eval_examples = [
        K0Example(
            example_id=f"toy_k0_gap_{i}",
            sessions=[
                f"Block header padding text. Record {i}A sets EV_VAL_{i} = {100 + i} with checksum ok.",
                f"Block header padding text. Record {i}B logs auxiliary counter {200 + i} done.",
            ],
            live_prompt=f"Live query header: report EV_VAL_{i} value now: ",
            answer=f"{100 + i}",
            category="multi_hop",
        )
        for i in range(4)
    ]

    # Pre-train a brief contiguous step on the full oracle sequences so Oracle Arm (a) beats Arm (c) cleanly,
    # creating a realistic positive `YELLOW_B`-like Arm (c) NLL gap (`nll_c - nll_a > 0`) before calibration.
    cal.stack.model.train()
    pre_opt = torch.optim.AdamW(list(cal.stack.named_trainable.values()), lr=1.5e-2)
    for _ in range(12):
        pre_opt.zero_grad(set_to_none=True)
        tot = torch.zeros((), dtype=torch.float32)
        for ex in eval_examples:
            full_text = ex.sessions[0] + ex.sessions[1] + ex.live_prompt + ex.answer
            ids = torch.tensor([cal.stack.tokenizer.encode(full_text)], dtype=torch.long)
            out = cal.stack.model(input_ids=ids, labels=ids, use_cache=False)
            tot = tot + out.loss
        (tot / len(eval_examples)).backward()
        pre_opt.step()

    ev = ModelK0Evaluator(
        cal.stack.model,
        cal.stack.tokenizer,
        device=cal.stack.device,
        live_sink_len=int(cal.bcfg.live_sink_len),
        lsa_manager=cal.stack.lsa,
    )
    rep_before = run_k0_suite(eval_examples, ev, cal.k0_cfg)
    gap_before = float(rep_before.arms["arm_c"].nll_gap)
    nll_c_before = float(rep_before.arms["arm_c"].answer_nll)
    assert gap_before > 0.05, f"expected positive initial Arm (c) NLL gap before calibration, got {gap_before}"

    # Calibrate directly with BlockAttentionCalibrator on exact multi-session block episodes
    cal_episodes: list[BlockEpisode] = []
    for ex in eval_examples:
        live_ids = cal.stack.tokenizer.encode(ex.live_prompt)
        sink_n = min(int(cal.bcfg.live_sink_len), max(0, len(live_ids) - 1))
        cal_episodes.append(
            BlockEpisode(
                live_sink_ids=live_ids[:sink_n],
                sessions_ids=[cal.stack.tokenizer.encode(s) for s in ex.sessions],
                live_rest_ids=live_ids[sink_n:],
                suffix_ids=cal.stack.tokenizer.encode(ex.answer),
                source="synthetic",
                category="multi_hop",
            )
        )
    ep_cursor = {"i": 0}

    def _next_ep():
        ep = cal_episodes[ep_cursor["i"] % len(cal_episodes)]
        ep_cursor["i"] += 1
        return ep

    cal._sample_episode = _next_ep  # type: ignore[method-assign]
    cal.train()

    rep_after = run_k0_suite(eval_examples, ev, cal.k0_cfg)
    gap_after = float(rep_after.arms["arm_c"].nll_gap)
    nll_c_after = float(rep_after.arms["arm_c"].answer_nll)

    assert nll_c_after < nll_c_before, f"Arm (c) NLL did not decrease: {nll_c_before:.4f} -> {nll_c_after:.4f}"
    assert abs(gap_after) < abs(gap_before) and gap_after < gap_before, (
        f"Arm (c) NLL gap vs Contiguous Oracle did not decrease: {gap_before:.4f} -> {gap_after:.4f}"
    )


def test_zero_leakage_episode_disjointness_and_seed_guard() -> None:
    with pytest.raises(ValueError, match="synthetic_seed_offset"):
        BlockAttnCalibConfig(synthetic_seed_offset=42)

    from qloop.data.tokenizer import ToyTokenizer

    tok = ToyTokenizer()
    bcfg = BlockAttnCalibConfig(
        n_sessions=3,
        session_len=32,
        suffix_len=8,
        live_sink_len=4,
        sink_tokens=4,
        synthetic_seed_offset=10_000,
    )
    eval_exs = build_disjoint_k0_eval_examples(12, seed=42)
    assert len(eval_exs) == 12
    eval_blob = "\n".join(" ".join(ex.sessions) + " " + ex.live_prompt for ex in eval_exs)

    # All 3 synthetic training templates (QA, temporal_update, code) must be strictly disjoint from K0 eval identifiers
    categories = set()
    saw_superseded = False
    for idx in range(9):
        ep = build_synthetic_calib_episode(tok, bcfg, episode_idx=idx, base_seed=0)
        categories.add(ep.category)
        if ep.superseded_indices:
            saw_superseded = True
        assert len(ep.sessions_ids) == 3
        assert len(ep.live_rest_ids) >= 1 and len(ep.suffix_ids) >= 1
        ep_text = tok.decode(
            ep.live_sink_ids + [t for s in ep.sessions_ids for t in s] + ep.live_rest_ids + ep.suffix_ids
        )
        for forbidden in ("MAX_RETRIES", "CACHE_PORT", "DEFAULT_TIMEOUT_S", "EV_PORT", "EV_MAX_HOPS", "eval_gateway"):
            assert forbidden not in ep_text
        for train_prefix in ("calib_", "TR_"):
            assert train_prefix not in eval_blob
    assert categories == {"synthetic_multi_session_qa", "synthetic_temporal_update", "synthetic_code_retrieval"}
    assert saw_superseded is True


# ==============================================================================
# 3. drop_inactive_loop_adapters=True Migration (s2_v2 88L -> Option A 40L d=1)
# ==============================================================================


def test_drop_inactive_loop_adapters_migration_and_legacy_parity(tmp_path: Path) -> None:
    # Build synthetic s2_v2-like 3-pass looped stack (GDN-2 + QED + neg_eigval + 3-pass LSA + 3-pass loop)
    s2_v2_cfg: dict[str, Any] = {
        "seed": 3,
        "stage": "s2_v2_joint_bptt",
        "model": {
            "name": "toy",
            "dtype": "float32",
            "device_map": "cpu",
            "toy": {
                "seed": 3,
                "num_hidden_layers": 8,
                "hidden_size": 32,
                "head_dim": 64,
                "partial_rotary_factor": 0.25,
            },
        },
        "loop": {
            "enabled": True,
            "prelude_end": 2,
            "core_end": 6,
            "num_passes": 3,
            "target_virtual_layers": 16,
            "kv_share": True,
            "checkpoint": False,
        },
        "gdn2": {
            "enabled": True,
            "rank": 8,
            "chunk_size": 8,
            "qed_enabled": True,
            "neg_eigval_max_boost": 0.5,
            "backend": "pytorch",
        },
        "lsa": {
            "enabled": True,
            "mode": "sparse",
            "num_query_passes": 3,
            "index_dim": 8,
            "index_on": "nope",
            "pool": 2,
            "block_size": 8,
            "avg_bucket_keys": 2,
            "max_bucket_keys": 4,
            "num_select": 2,
            "window": 8,
            "sinks": 2,
        },
        "train": {
            "train_loop_adapters": True,
            "train_gdn2": True,
            "train_lsa": True,
            "trainable": [r"input_layernorm|post_attention_layernorm"],
        },
    }
    st_s2 = build_stack(s2_v2_cfg)
    with torch.no_grad():
        for name, p in st_s2.named_trainable.items():
            if "gdn2_adapter.down.weight" in name:
                p.fill_(0.123)
            elif "lsa_indexer.wk" in name:
                p.fill_(0.456)
    s2_dir = str(tmp_path / "s2_v2_joint_bptt")
    s2_params = st_s2.checkpoint_params()
    assert any(k.startswith("loop_adapters.") for k in s2_params)
    assert any(k.endswith("gdn2_adapter.qed_lambda_raw") for k in s2_params)
    assert any("q_proj_extra." in k for k in s2_params)
    CheckpointManager(s2_dir, keep_last=2).save(
        10, s2_params, None, {"stage": "s2_v2_joint_bptt", "stack": st_s2.describe()}
    )

    # Target Option A stack: loop=False, pruned GDN-2 (qed=False, neg_eigval=0.0), D-LSA 1.0x (num_query_passes=1)
    opt_a_cfg = _toy_opt_a_block_attn_cfg(steps=1, loop_enabled=False, eval_k0_examples=0)

    # 1. Legacy default (`drop_inactive_loop_adapters=False`): keeps `loop_adapters.*` in passthrough_tensors
    st_legacy = build_stack(opt_a_cfg)
    st_legacy.restore_checkpoint(s2_params, strict=False, prior_stack_meta=st_s2.describe())
    assert any(k.startswith("loop_adapters.") for k in st_legacy.passthrough_tensors)

    # 2. `drop_inactive_loop_adapters=True`: cleanly drops all `loop_adapters.*` AND prunes QED/neg_eigval/extra LSA
    st_clean = build_stack(opt_a_cfg)
    st_clean.restore_checkpoint(
        s2_params,
        strict=False,
        prior_stack_meta=st_s2.describe(),
        drop_inactive_loop_adapters=True,
    )
    assert not any("loop_adapters." in k for k in st_clean.passthrough_tensors)
    ck_out = st_clean.checkpoint_params()
    assert not any("loop_adapters." in k for k in ck_out)
    assert not any(k.endswith("gdn2_adapter.qed_lambda_raw") for k in ck_out)
    assert not any(k.endswith("gdn2_adapter.neg_gamma_raw") for k in ck_out)
    assert not any("q_proj_extra." in k for k in ck_out)
    # Shared GDN-2 & LSA weights transferred faithfully!
    for name, p in st_clean.named_trainable.items():
        if "gdn2_adapter.down.weight" in name:
            assert torch.allclose(p, torch.full_like(p, 0.123))
        elif "lsa_indexer.wk" in name:
            assert torch.allclose(p, torch.full_like(p, 0.456))

    # 3. Verify `BlockAttentionCalibrator` and `LMTrainer` both forward `train.drop_inactive_loop_adapters`
    cal_cfg = _toy_opt_a_block_attn_cfg(
        steps=1,
        init_from=s2_dir,
        drop_inactive_loop_adapters=True,
        eval_k0_examples=0,
        contiguous_lm_every=0,
    )
    cal = BlockAttentionCalibrator(cal_cfg, str(tmp_path / "cal_migrated"))
    cal.maybe_resume()
    assert cal.resume_mode == "init_from"
    assert not any("loop_adapters." in k for k in cal.stack.passthrough_tensors)

    lm_cfg = dict(cal_cfg)
    lm_cfg["stage_type"] = "lm"
    lm_tr = LMTrainer(lm_cfg, str(tmp_path / "lm_migrated"))
    lm_tr.maybe_resume()
    assert lm_tr.resume_mode == "init_from"
    assert not any("loop_adapters." in k for k in lm_tr.stack.passthrough_tensors)


# ==============================================================================
# 4. CLI `stage_type: block_attn_calib` Dispatch & Exact Checkpoint Save/Resume
# ==============================================================================


def test_cli_block_attn_calib_dispatch_and_exact_resume(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run_root = tmp_path / "runs"
    monkeypatch.setenv("QLOOP_RUN_ROOT", str(run_root))
    cfg = _toy_opt_a_block_attn_cfg(
        steps=2,
        loop_enabled=False,
        synthetic_mix_prob=0.5,
        contiguous_lm_every=2,
        eval_k0_examples=2,
        eval_k0_mid_examples=2,
        eval_every=2,
        ckpt_every=2,
    )
    cfg_path = tmp_path / "opt_a_calib.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    # Run 2 steps via CLI (`stage_type: block_attn_calib`)
    rc1 = cli_main(["train", "-c", str(cfg_path)])
    assert rc1 == 0
    run_dir = run_root / "s3_opt_a_toy_calib"
    ck_mgr = CheckpointManager(str(run_dir), keep_last=3)
    ck2 = ck_mgr.load()
    assert ck2 is not None and int(ck2["state"]["step"]) == 2
    assert "k0" in ck2["state"] and "k0/verdict" in ck2["state"]["k0"]

    # Resume to 4 steps via CLI override and verify exact resume + contiguous anchor logging
    rc2 = cli_main(["train", "-c", str(cfg_path), "-s", "train.steps=4"])
    assert rc2 == 0
    ck4 = ck_mgr.load()
    assert ck4 is not None and int(ck4["state"]["step"]) == 4
    assert ck4["state"]["resume_mode"] == "exact"

    recs = [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()]
    assert any(rec.get("loss/contiguous_anchor") == 1.0 for rec in recs)
    assert any(rec.get("step") == 4 and "k0/arm_c_nll_gap" in rec for rec in recs)


# ==============================================================================
# 5. GRPO-lambda Accuracy-Gated Overlong + concise_bonus + Option A d=1 RL
# ==============================================================================


def _rl_opt_a_toy_cfg(
    *,
    steps: int = 1,
    overlong: dict[str, Any] | None = None,
    gate_every: int = 0,
    fused: bool = False,
) -> dict[str, Any]:
    return {
        "seed": 0,
        "stage": "s4_opt_a_d1_rl_toy",
        "stage_type": "rl",
        "model": {
            "name": "toy",
            "dtype": "float32",
            "device_map": "cpu",
            "toy": {
                "seed": 0,
                "num_hidden_layers": 8,
                "hidden_size": 32,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "head_dim": 16,
                "partial_rotary_factor": 0.25,
                "num_experts": 4,
                "num_experts_per_tok": 2,
            },
        },
        "loop": {"enabled": False},
        "gdn2": {
            "enabled": True,
            "rank": 8,
            "chunk_size": 8,
            "qed_enabled": False,
            "neg_eigval_max_boost": 0.0,
            "backend": "pytorch",
        },
        "hssd": {"enabled": False},
        "lsa": {
            "enabled": True,
            "mode": "sparse",
            "num_query_passes": 1,
            "index_dim": 8,
            "index_on": "nope",
            "pool": 2,
            "block_size": 8,
            "avg_bucket_keys": 2,
            "max_bucket_keys": 4,
            "num_select": 2,
            "window": 8,
            "sinks": 2,
        },
        "experts": {"enabled": False},
        "train": {
            "steps": steps,
            "ckpt_every": 10,
            "sha256_frozen_experts": True,
            "train_loop_adapters": False,
            "train_gdn2": True,
            "train_lsa": True,
            "trainable": [
                r"mlp\.gate\.weight$",
                r"shared_expert\.",
                r"self_attn\.(o_proj|q_norm|k_norm)",
                r"input_layernorm|post_attention_layernorm",
            ],
            "allowed_groups": ["gdn2_adapters", "lsa_indexers", "router", "shared_expert", "attention", "norms"],
            "optim": {
                "lr": 1e-3,
                "fused": fused,
                "max_grad_norm": 0.5,
                "apollo": {"min_dim": 128},
            },
        },
        "rl": {
            "tasks": [
                {
                    "source": "inline",
                    "name": "m",
                    "verifier": "scorer:math",
                    "items": [{"id": "t0", "prompt": "Compute 6 * 7.", "answer": "42"}],
                }
            ],
            "group_size": 4,
            "prompts_per_step": 1,
            "max_new_tokens": 100,
            "temperature": 1.0,
            "micro_batch_seqs": 2,
            "ppo_epochs": 1,
            "overlong": overlong
            if overlong is not None
            else {
                "enabled": True,
                "buffer": 20,
                "penalty": 1.0,
                "accuracy_gated": True,
                "concise_bonus": 0.15,
                "target_tokens": 60,
                "min_pass_rate": 0.25,
            },
            "sirsi": {
                "enabled": True,
                "slap_weight": 0.1,
                "slap_eps": 1e-4,
                "slap_max_seqs_per_step": 4,
                "slap_target_groups": ["router", "shared_expert"],
                "anti_slop_weight": 0.05,
                "failure_ledger": False,
            },
            "gate": {
                "every": gate_every,
                "depth": 1,
                "benchmarks": [
                    {
                        "name": "aime_2026",
                        "limit": 2,
                        "max_new_tokens": 4,
                        "offline_items": [
                            {"id": "a", "problem": "1+1?", "answer": "2"},
                            {"id": "b", "problem": "2+2?", "answer": "4"},
                        ],
                    }
                ],
            },
        },
    }


def test_grpo_lambda_overlong_and_concise_bonus_unit_semantics(tmp_path: Path) -> None:
    # 1. Legacy mode (`accuracy_gated=False`, no `concise_bonus`) exact parity
    cfg_leg = _rl_opt_a_toy_cfg(overlong={"enabled": True, "buffer": 20, "penalty": 1.0})
    tr_leg = GRPOTrainer(cfg_leg, str(tmp_path / "rl_legacy"))
    try:
        # start = 100 - 20 = 80; at 90 tokens -> -1.0 * (10 / 20) = -0.5 regardless of pass_rate
        assert tr_leg._overlong(75, pass_rate=0.0) == 0.0
        assert tr_leg._overlong(90) == pytest.approx(-0.5)
        assert tr_leg._overlong(90, pass_rate=0.0) == pytest.approx(-0.5)
        assert tr_leg._overlong(90, pass_rate=1.0) == pytest.approx(-0.5)
        assert tr_leg._concise_bonus(10, reward=1.0, pass_rate=1.0) == 0.0
    finally:
        tr_leg.pool.shutdown(wait=False)

    # 2. GRPO-lambda accuracy_gated=True + concise_bonus
    cfg_lam = _rl_opt_a_toy_cfg(
        overlong={
            "enabled": True,
            "buffer": 20,
            "penalty": 1.0,
            "accuracy_gated": True,
            "concise_bonus": 0.15,
            "target_tokens": 60,
            "min_pass_rate": 0.25,
        }
    )
    tr_lam = GRPOTrainer(cfg_lam, str(tmp_path / "rl_lambda"))
    try:
        # Accuracy-gated overlong: 0% pass -> 0.0, 50% pass -> 0.5x, 100% pass -> 1.0x
        assert tr_lam._overlong(100, pass_rate=0.0) == 0.0
        assert tr_lam._overlong(90, pass_rate=0.0) == 0.0
        assert tr_lam._overlong(90, pass_rate=0.5) == pytest.approx(-0.25)
        assert tr_lam._overlong(90, pass_rate=1.0) == pytest.approx(-0.5)
        assert tr_lam._overlong(100, pass_rate=1.0) == pytest.approx(-1.0)

        # concise_bonus: strictly verified-correct samples (`reward > 0`) when `pass_rate >= min_pass_rate`
        b_short_ok = tr_lam._concise_bonus(15, reward=1.0, pass_rate=0.5)  # 0.15 * (1 - 15/60) = 0.1125
        b_long_ok = tr_lam._concise_bonus(45, reward=1.0, pass_rate=0.5)  # 0.15 * (1 - 45/60) = 0.0375
        b_over_ok = tr_lam._concise_bonus(80, reward=1.0, pass_rate=0.5)  # clamped to 0.0
        assert b_short_ok == pytest.approx(0.1125)
        assert b_long_ok == pytest.approx(0.0375)
        assert b_short_ok > b_long_ok > b_over_ok == 0.0

        # NEVER boosts short wrong answers (`reward <= 0`)!
        assert tr_lam._concise_bonus(1, reward=0.0, pass_rate=0.75) == 0.0
        assert tr_lam._concise_bonus(1, reward=-0.5, pass_rate=1.0) == 0.0

        # Gated off when `pass_rate < min_pass_rate` (0.25)
        assert tr_lam._concise_bonus(10, reward=1.0, pass_rate=0.20) == 0.0
        assert tr_lam._concise_bonus(10, reward=1.0, pass_rate=0.25) == pytest.approx(0.15 * (1.0 - 10.0 / 60.0))
    finally:
        tr_lam.pool.shutdown(wait=False)


def test_grpo_lambda_rollout_stats_and_advantage_shaping(tmp_path: Path) -> None:
    cfg = _rl_opt_a_toy_cfg(
        overlong={
            "enabled": True,
            "buffer": 20,
            "penalty": 1.0,
            "accuracy_gated": True,
            "concise_bonus": 0.15,
            "target_tokens": 60,
            "min_pass_rate": 0.25,
        }
    )
    cfg["rl"]["prompts_per_step"] = 2
    cfg["rl"]["group_size"] = 4
    cfg["rl"]["sirsi"]["enabled"] = False
    cfg["rl"]["tasks"] = [
        {
            "source": "inline",
            "name": "m",
            "verifier": "scorer:math",
            "items": [
                {"id": "hard", "prompt": "Hard problem.", "answer": "42"},
                {"id": "easy", "prompt": "Easy problem.", "answer": "7"},
            ],
        }
    ]
    tr = GRPOTrainer(cfg, str(tmp_path / "rl_rollout_lambda"))
    try:
        # Mock generator:
        #   Prompt 0 (`hard`): 0/4 pass with very different lengths (10 vs 100 tok) -> ALL receive 0.0 shaping ->
        #     zero variance -> group dropped cleanly (no spurious punishment for long reasoning on hard prompts!)
        #   Prompt 1 (`easy`): 4/4 pass with 2 short (15 tok) + 2 long (90 tok) ->
        #     short correct gets concise_bonus (+0.1125) and 0 overlong; long correct gets 0 bonus and -0.5 overlong ->
        #     positive adv on short correct, negative adv on long overlong!
        def _fake_generate(prompts, **_kw):
            out = []
            for p in prompts:
                if "Hard" in str(p):
                    out.append(
                        [
                            GenOut("wrong \\boxed{0}", [1] * 10),
                            GenOut("wrong \\boxed{1}", [1] * 100),
                            GenOut("wrong \\boxed{2}", [1] * 95),
                            GenOut("wrong \\boxed{3}", [1] * 20),
                        ]
                    )
                else:
                    out.append(
                        [
                            GenOut("short \\boxed{7}", [2] * 15),
                            GenOut("short \\boxed{7}", [2] * 15),
                            GenOut("long \\boxed{7}", [2] * 90),
                            GenOut("long \\boxed{7}", [2] * 90),
                        ]
                    )
            return out

        tr.gen.generate = _fake_generate  # type: ignore[method-assign]
        seqs, stats = tr.rollout()
        assert stats["groups/mean_pass_rate"] == pytest.approx(0.5)
        # Hard prompt: 4 samples * 0.0 = 0.0; Easy prompt: 2 * 0.0 + 2 * (-0.5) = -1.0 -> mean over 8 = -0.125
        assert stats["rewards/overlong_mean"] == pytest.approx(-0.125)
        # Only Easy prompt produced PG seqs (Hard had 0 variance because accuracy_gated zeroed length penalty!)
        assert len(seqs) == 4
        assert seqs[0].adv > 0.0 and seqs[1].adv > 0.0
        assert seqs[2].adv < 0.0 and seqs[3].adv < 0.0
    finally:
        tr.pool.shutdown(wait=False)


def test_grpo_option_a_d1_non_looped_slap_and_gate_e2e(tmp_path: Path) -> None:
    """Verify `GRPOTrainer` runs cleanly with `loop.enabled=False`, `slap_target_groups`, and `gate()`."""
    cfg = _rl_opt_a_toy_cfg(steps=2, gate_every=2, fused=False)
    cfg["rl"]["max_new_tokens"] = 6
    cfg["rl"]["overlong"]["buffer"] = 2
    cfg["rl"]["overlong"]["target_tokens"] = 4
    run_dir = str(tmp_path / "rl_opt_a_e2e")
    tr = GRPOTrainer(cfg, run_dir)
    assert tr.stack.loop is None
    assert tr.stack.gdn2 is not None and tr.stack.gdn2.cfg.qed_enabled is False
    assert tr.stack.lsa is not None and tr.stack.lsa.cfg.num_query_passes == 1

    # Verify `_slap_target_params` is non-empty and restricted strictly to `router` + `shared_expert`
    assert tr._slap_target_params is not None and len(tr._slap_target_params) > 0
    target_ids = {id(p) for p in tr._slap_target_params}
    targeted_groups = {
        param_group_of(n) for n, p in tr.stack.named_trainable.items() if id(p) in target_ids
    }
    assert targeted_groups == {"router", "shared_expert"}

    # Verify surgical `slap_neg` updates ONLY `router`/`shared_expert` (norms/gdn2/attention untouched!)
    before = {n: p.detach().clone() for n, p in tr.stack.named_trainable.items()}
    slap_seq = _Seq([10, 11, 12, 13, 14, 15], 2, -1.0, "slap_neg", span_mask=[False, True, True, True])
    st_slap = tr.update([slap_seq])
    assert st_slap["loss/slap_neg"] > 0.0
    for n, p in tr.stack.named_trainable.items():
        if id(p) not in target_ids:
            assert torch.equal(p.detach(), before[n]), f"non-targeted param {n} mutated by slap_neg"
    assert any(
        not torch.equal(p.detach(), before[n])
        for n, p in tr.stack.named_trainable.items()
        if id(p) in target_ids
    )

    # End-to-end 2-step RL train with regression gate (`depth: 1` on `loop=None`) + frozen experts verification
    sha_before = tr.stack.frozen_experts_sha256
    assert sha_before is not None
    res = tr.train()
    assert res["step"] == 2
    assert tr.stack.verify_frozen_experts() == sha_before
    assert "groups/mean_pass_rate" in res["last"]
    assert "rewards/overlong_mean" in res["last"]
    assert os.path.exists(os.path.join(run_dir, "checkpoints", "PROMOTED"))


# ==============================================================================
# 6. Shipped Option A Configs + sprint2_audit & hub.load_qloop_from_hub Sync
# ==============================================================================


def test_option_a_stage_configs_hygiene_and_schema() -> None:
    s3_path = ROOT / "configs" / "stages" / "s3_opt_a_block_attn_calib.yaml"
    s4_path = ROOT / "configs" / "stages" / "s4_opt_a_d1_rl.yaml"
    calib_src = (ROOT / "src" / "qloop" / "train" / "block_attn_calib.py").read_text(encoding="utf-8")

    for fp in (s3_path, s4_path):
        raw = fp.read_text(encoding="utf-8")
        assert not re.search(r"\bHBM\b", raw, flags=re.IGNORECASE), f"forbidden HBM reference in {fp}"
        assert "opus46" not in raw.lower(), f"forbidden opus46 reference in {fp}"
    assert not re.search(r"\bHBM\b", calib_src, flags=re.IGNORECASE), "forbidden HBM reference in block_attn_calib.py"

    s3 = load_config(str(s3_path))
    s4 = load_config(str(s4_path))

    # Shared Option A 40L d=1 pruned GDN-2-1M + D-LSA 1.0x invariants
    for name, cfg in (("s3_opt_a_block_attn_calib", s3), ("s4_opt_a_d1_rl", s4)):
        assert cfg.stage == name
        assert cfg.model.name == "unsloth/Qwen3.6-35B-A3B-NVFP4"
        assert int(cfg.model.yarn_target_len) == 1_048_576
        assert cfg.loop.enabled is False
        assert cfg.gdn2.enabled is True
        assert cfg.gdn2.qed_enabled is False
        assert float(cfg.gdn2.neg_eigval_max_boost) == 0.0
        assert cfg.hssd.enabled is False
        assert cfg.lsa.enabled is True
        assert cfg.lsa.mode == "sparse"
        assert int(cfg.lsa.num_query_passes) == 1
        assert cfg.experts.enabled is False
        assert cfg.train.sha256_frozen_experts is True
        GDN2Config.from_dict(dict(cfg.gdn2)).validate()
        LSAConfig.from_dict(dict(cfg.lsa)).validate()

    # S3 Option A specifics
    assert s3.stage_type == "block_attn_calib"
    assert s3.train.init_from == "runs/s2_v2_joint_bptt"
    assert s3.train.drop_inactive_loop_adapters is True
    assert int(s3.train.steps) == 300
    assert int(s3.data.holdout_mod) == 100
    assert int(s3.data.holdout_eval_buckets) == 2
    src_names = [s["name"] for s in s3.data.sources]
    assert not any("opus46" in n.lower() for n in src_names)
    assert set(src_names) == {"openthoughts3", "nemotron_post", "finepdfs_mix", "fineweb_edu"}
    bcfg = BlockAttnCalibConfig.from_dict(dict(s3.block_attn))
    assert bcfg.kl_weight == pytest.approx(0.7)
    assert bcfg.ce_weight == pytest.approx(0.3)
    assert bcfg.synthetic_seed_offset >= 10_000
    assert bcfg.eval_k0_examples == 20

    # S4 Option A specifics
    assert s4.stage_type == "rl"
    assert s4.train.init_from == "runs/s3_opt_a_block_attn_calib"
    assert int(s4.train.steps) == 300
    assert int(s4.rl.max_new_tokens) == 4096
    assert s4.rl.overlong.enabled is True
    assert s4.rl.overlong.accuracy_gated is True
    assert float(s4.rl.overlong.concise_bonus) == pytest.approx(0.15)
    assert int(s4.rl.overlong.target_tokens) == 1536
    assert float(s4.rl.overlong.min_pass_rate) == pytest.approx(0.25)
    assert list(s4.rl.sirsi.slap_target_groups) == ["router", "shared_expert"]
    assert float(s4.rl.sirsi.anti_slop_weight) == pytest.approx(0.05)
    assert int(s4.rl.gate.depth) == 1
    assert [b["name"] for b in s4.rl.gate.benchmarks] == ["aime_2026", "gpqa_diamond"]


def test_sprint2_audit_and_hub_loader_sync_option_a_checkpoint(tmp_path: Path) -> None:
    """Verify `sprint2_audit._load_arm_stack` and `hub.load_qloop_from_hub` load Option A checkpoints cleanly."""
    run_dir = str(tmp_path / "s3_opt_a_block_attn_calib")
    cfg = _toy_opt_a_block_attn_cfg(
        steps=2,
        loop_enabled=False,
        synthetic_mix_prob=0.5,
        contiguous_lm_every=0,
        eval_k0_examples=0,
    )
    cfg["stage"] = "s3_opt_a_block_attn_calib"
    cfg["run"] = {"name": "s3_opt_a_block_attn_calib"}
    cal = BlockAttentionCalibrator(cfg, run_dir)
    res = cal.train()

    probe_ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]], dtype=torch.long)
    cal.stack.model.eval()
    with torch.no_grad():
        ref_logits = cal.stack.model(input_ids=probe_ids, use_cache=False).logits.clone()

    # 1. `sprint2_audit._load_arm_stack` with matching YAML + mismatched 3-pass `s2_v2` YAML (empty smeta)
    cfg_path = tmp_path / "opt_a.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    st_audit, _ = _load_arm_stack({"name": "opt_a", "config": str(cfg_path), "checkpoint": run_dir})
    assert st_audit.loop is None
    assert st_audit.gdn2 is not None and st_audit.gdn2.cfg.qed_enabled is False
    assert st_audit.gdn2.cfg.neg_eigval_max_boost == 0.0
    assert st_audit.lsa is not None and st_audit.lsa.cfg.num_query_passes == 1
    with torch.no_grad():
        audit_logits = st_audit.model(input_ids=probe_ids, use_cache=False).logits
    torch.testing.assert_close(audit_logits, ref_logits, rtol=1e-5, atol=1e-5)

    # Also test `_sync_cfg_to_checkpoint_payload` when `smeta` is empty and base config came from 3-pass s2_v2!
    ckpt_pt = Path(res["checkpoint"]) / "trainable.pt"
    saved_params = torch.load(str(ckpt_pt), map_location="cpu")
    stale_3pass_cfg = {
        "loop": {"enabled": True, "num_passes": 3},
        "gdn2": {"enabled": True, "rank": 64, "qed_enabled": True, "neg_eigval_max_boost": 0.5},
        "lsa": {"enabled": True, "num_query_passes": 3, "index_dim": 64},
        "experts": {"enabled": True},
    }
    _sync_cfg_to_checkpoint_payload(stale_3pass_cfg, saved_params, {}, {})
    assert stale_3pass_cfg["loop"]["enabled"] is False
    assert stale_3pass_cfg["gdn2"]["enabled"] is True
    assert stale_3pass_cfg["gdn2"]["qed_enabled"] is False
    assert stale_3pass_cfg["gdn2"]["neg_eigval_max_boost"] == 0.0
    assert stale_3pass_cfg["gdn2"]["rank"] == 8
    assert stale_3pass_cfg["lsa"]["enabled"] is True
    assert stale_3pass_cfg["lsa"]["num_query_passes"] == 1
    assert stale_3pass_cfg["lsa"]["index_dim"] == 8

    # 2. `hub.load_qloop_from_hub` round-trip via _MockHfApi
    api = _MockHfApi()
    repo_id = "TassP0lar/test-opt-a"
    hub = HubSyncManager(repo_id=repo_id, api=api)
    state_json = Path(res["checkpoint"]) / "state.json"
    hub.push_checkpoint(
        "s3_opt_a_block_attn_calib",
        res["checkpoint"],
        config_dict=cfg,
        completed=True,
        target_steps=2,
        include_optim=False,
    )
    st_hub = load_qloop_from_hub(repo_id, stage_name="s3_opt_a_block_attn_calib", api=api)
    assert st_hub.loop is None
    assert st_hub.gdn2 is not None and st_hub.gdn2.cfg.qed_enabled is False
    assert st_hub.gdn2.cfg.neg_eigval_max_boost == 0.0
    assert st_hub.lsa is not None and st_hub.lsa.cfg.num_query_passes == 1
    with torch.no_grad():
        hub_logits = st_hub.model(input_ids=probe_ids, use_cache=False).logits
    torch.testing.assert_close(hub_logits, ref_logits, rtol=1e-5, atol=1e-5)
    assert state_json.is_file()
