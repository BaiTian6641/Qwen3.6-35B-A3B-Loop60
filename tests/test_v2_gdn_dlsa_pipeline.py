"""Comprehensive tests for the clean v2 Native-GDN + 88L Loop + 1M D-LSA pipeline.

Covers:
  1. Stage configs (`s3_v2_long_1m_lsa.yaml`, `s2_v2_joint_bptt.yaml`, `s4_v2_rl.yaml`):
     zero raw `fable51`, `hssd.enabled=False`, `gdn2.enabled=False`, `lsa.enabled=True`,
     `optim.fused=False`, `sha256_frozen_experts=True`, mHC protection & emphasis knobs (#392),
     and untouched legacy HSSD v3.2 ablation configs.
  2. `loop64_fable_sirsi` registry spec + `formats.normalize_row` parsing (including `tools_json`,
     `tool_calls_json`, reasoning, content, multi-turn agentic traces, and live `sft_core.parquet`
     verification when available).
  3. mHC `LoopController.stats()` telemetry (`bridge_subspace_up_abs_mean_p{2,3}`,
     `bridge_step_emb_norm_p{2,3}`) and full gradient flow through `bridge`, `gates`,
     `pass_router`, and `pass_attn` across `s3_v2`, `s2_v2`, and `s4_v2_rl`.
  4. Native-GDN (`hssd.enabled=False`) + 88L Loop + D-LSA + `transformers 5.24.0` cache audit:
     `_maybe_create_initial_cache`, `_ReplayKVCacheProxy` (`cache.layers[i]`, `has_previous_state`,
     missing `get_conv_state`, single-arg `get_recurrent_state`, in-place `.copy_()` checkpoint
     safety), `long_calib` with `fp8_kv_cache=True`, `bptt.SegmentedKVBPTTRunner`, and `LMTrainer`/
     `GRPOTrainer` smoke runs with `hssd=None`.
  5. `hub.py` + `cli.py` v2 GDN stage wiring (`V3_GDN_PIPELINE_ORDER`, `resolve_stage_config_path`,
     README/manifest updates, `s4_v2_rl` never gets `hssd.enabled=true`, `eval` resolves `s4_v2_rl`).
  6. `colab_supervisor.py` v2 `--stages` and exponential backoff (`60s -> 120s -> 240s -> 300s`)
     on transient Colab `503 Service Unavailable` / GPU capacity failures.
"""
from __future__ import annotations

import copy
import json
import os
import pathlib
import subprocess
import sys
import types
from dataclasses import dataclass
from typing import Any

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from qloop.cli import _pipeline_stage_cmds
from qloop.colab_supervisor import (
    BOOTSTRAP_SENTINEL,
    DEFAULT_MAX_RESPAWN_BACKOFF_SEC,
    PROBE_SENTINEL,
    TAIL_SENTINEL,
    ColabCommandError,
    ColabSupervisor,
    SupervisorConfig,
    build_arg_parser,
    build_remote_run_pipeline_script,
    compute_respawn_backoff_sec,
    config_from_args,
    is_transient_capacity_error,
)
from qloop.config import load_config
from qloop.data.formats import normalize_row, validate_example
from qloop.data.registry import eval_groups, get_spec
from qloop.data.tokenizer import render_conversation
from qloop.hub import (
    V3_GDN_PIPELINE_ORDER,
    V3_PIPELINE_ORDER,
    HubManifest,
    HubSyncManager,
    render_model_card,
    resolve_pipeline_order,
    resolve_stage_config_path,
)
from qloop.model.loop import apply_loop
from qloop.model.lsa import install_lsa
from qloop.model.toy import ToyConfig, ToyForCausalLM, ToyHybridCache
from qloop.train.bptt import (
    BPTTConfig,
    SegmentedKVBPTTRunner,
    _maybe_create_initial_cache,
    _ReplayKVCacheProxy,
)
from qloop.train.long_calib import prefix_suffix_calibration_step
from qloop.train.rl import GRPOTrainer
from qloop.train.stack import build_stack
from qloop.train.trainer import LMTrainer

ROOT = pathlib.Path(__file__).resolve().parents[1]


# ============================================================================ 1. Stage Configs


def test_v2_gdn_stage_configs_and_zero_raw_fable51() -> None:
    cfg_dir = ROOT / "configs" / "stages"
    v2_paths = {
        "s3_v2_long_1m_lsa": cfg_dir / "s3_v2_long_1m_lsa.yaml",
        "s2_v2_joint_bptt": cfg_dir / "s2_v2_joint_bptt.yaml",
        "s4_v2_rl": cfg_dir / "s4_v2_rl.yaml",
    }
    cfgs = {}
    for name, path in v2_paths.items():
        assert path.is_file(), f"missing config {path}"
        raw_text = path.read_text(encoding="utf-8")
        assert "fable51" not in raw_text, f"{name} must not reference raw fable51 anywhere"
        cfg = load_config(str(path))
        cfgs[name] = cfg
        assert cfg["stage"] == name
        assert cfg["run"]["name"] == name
        assert cfg["model"]["yarn_target_len"] == 1_048_576
        assert cfg["hssd"]["enabled"] is False
        assert cfg["lsa"]["enabled"] is True
        assert cfg["train"]["optim"]["fused"] is False
        assert cfg["train"]["sha256_frozen_experts"] is True
        assert cfg["train"]["train_hssd"] is False
        assert cfg["train"]["train_loop_adapters"] is True
        assert cfg["train"]["train_lsa"] is True
        # 88L Loop invariants
        lp = cfg["loop"]
        assert lp["enabled"] is True
        assert (lp["prelude_end"], lp["core_end"], lp["num_passes"], lp["target_virtual_layers"]) == (12, 36, 3, 88)
        assert lp["kv_share"] is True
        assert lp["bridge_mode"] == "subspace"
        assert lp["bridge_rank"] == 256
        assert lp["bridge_init"] == [0.05, 0.85, 0.10]
        assert lp["attn_adapter_rank"] == 64
        assert lp["per_pass_router"] is True
        assert lp["router_adapter_rank"] == 32
        assert "hssd_adapter" not in cfg["train"]["optim"]["lr_mult"]
        assert "hssd_adapters" not in cfg["train"]["allowed_groups"]

    # s3_v2_long_1m_lsa specifics (native GDN untouched: gdn2.enabled=False)
    s3 = cfgs["s3_v2_long_1m_lsa"]
    assert s3["gdn2"]["enabled"] is False
    assert s3["train"]["train_gdn2"] is False
    assert "gdn2_adapter" not in s3["train"]["optim"]["lr_mult"]
    assert "gdn2_adapters" not in s3["train"]["allowed_groups"]
    assert s3["train"]["init_from"] == "runs/s1_loop_a6b"
    assert s3["train"]["steps"] == 600
    assert s3["train"]["micro_batch"] == 1
    assert s3["train"]["grad_accum"] == 2
    assert s3["train"]["ckpt_every"] == 25
    assert s3["loop"]["train_pass_probs"] == [0.15, 0.50, 0.35]
    assert s3["lsa"]["mode"] == "hybrid"
    assert s3["lsa"]["compact_active_slots"] is True
    assert s3["lsa"]["max_selected_keys"] == 768
    assert s3["lsa"]["recall_threshold"] == 0.70
    assert s3["lsa"]["hybrid_sparse_every"] == 2
    assert s3["lsa"]["num_query_passes"] == 3
    assert s3["long_calib"]["enabled"] is True
    assert s3["long_calib"]["prefix_len"] == 524_288
    assert s3["long_calib"]["max_prefix_len"] == 1_015_808
    assert s3["long_calib"]["suffix_len"] == 32_768
    assert s3["long_calib"]["chunk_size"] == 16_384
    assert s3["long_calib"]["fp8_kv_cache"] is True
    assert s3["long_calib"]["prefix_lsa_mode"] == "sparse"
    assert s3["long_calib"]["calib_every_steps"] == 10
    assert s3["long_calib"]["sparse_prefix_after_step"] == 80
    assert dict(s3["train"]["aux_weights"]) == {"loop_aux": 1.0, "lsa_kl": 1.0}
    assert s3["train"]["optim"]["lr"] == pytest.approx(1.0e-4)
    assert s3["train"]["optim"]["warmup_steps"] == 40
    assert s3["train"]["optim"]["max_grad_norm"] == 1.0
    # #392 mHC protection: loop_adapters=2.0, lsa_indexer=10.0
    assert dict(s3["train"]["optim"]["lr_mult"]) == {"lsa_indexer": 10.0, "loop_adapters": 2.0}
    assert s3["data"]["seq_len"] == 65_536
    assert s3["data"]["respect_doc_boundaries"] is False
    s3_srcs = {row["name"]: row["weight"] for row in s3["data"]["sources"]}
    assert s3_srcs == {
        "finepdfs_mix": 0.45,
        "fineweb_edu": 0.25,
        "openthoughts3": 0.15,
        "loop64_fable_sirsi": 0.15,
    }
    assert sum(s3_srcs.values()) == pytest.approx(1.0)
    for src_name in s3_srcs:
        get_spec(src_name)

    # s2_v2_joint_bptt specifics (GDN-2-1M + Bounded QED + Soft Neg-Eigval enabled)
    s2 = cfgs["s2_v2_joint_bptt"]
    assert s2["gdn2"]["enabled"] is True
    assert s2["gdn2"]["rank"] == 64
    assert s2["gdn2"]["chunk_size"] == 64
    assert s2["gdn2"]["decay_log_min"] == pytest.approx(-3.0)
    assert s2["gdn2"]["decay_log_max"] == pytest.approx(0.5)
    assert s2["gdn2"]["qed_enabled"] is True
    assert s2["gdn2"]["qed_max_lambda"] == pytest.approx(0.35)
    assert s2["gdn2"]["neg_eigval_max_boost"] == pytest.approx(0.5)
    assert s2["gdn2"]["backend"] == "auto"
    assert s2["train"]["train_gdn2"] is True
    assert "gdn2_adapters" in s2["train"]["allowed_groups"]
    assert s2["train"]["init_from"] == "runs/s3_v2_long_1m_lsa"
    assert s2["train"]["steps"] == 600
    assert s2["train"]["micro_batch"] == 1
    assert s2["train"]["grad_accum"] == 2
    assert s2["train"]["ckpt_every"] == 25
    # #392 R=3 emphasis
    assert s2["loop"]["train_pass_probs"] == [0.10, 0.35, 0.55]
    assert s2["lsa"]["mode"] == "sparse"
    assert s2["bptt"]["enabled"] is True
    assert s2["bptt"]["n_segments"] == 4
    assert s2["bptt"]["seg_len"] == 32_768
    assert s2["bptt"]["replay_k"] == 2048
    assert s2["bptt"]["offload_prior_hidden"] == "auto"
    assert s2["bptt"]["query_sim_chunk"] == 1024
    assert dict(s2["train"]["aux_weights"]) == {"loop_aux": 1.0, "lsa_kl": 0.2}
    assert s2["train"]["optim"]["lr"] == pytest.approx(2.0e-5)
    assert s2["train"]["optim"]["warmup_steps"] == 50
    assert dict(s2["train"]["optim"]["lr_mult"]) == {"loop_adapters": 10.0, "gdn2_adapter": 5.0, "lsa_indexer": 5.0}
    assert s2["data"]["seq_len"] == 131_072
    assert s2["data"]["respect_doc_boundaries"] is True
    s2_srcs = {row["name"]: row["weight"] for row in s2["data"]["sources"]}
    assert s2_srcs == {
        "loop64_fable_sirsi": 0.25,
        "openthoughts3": 0.20,
        "nemotron_post": 0.20,
        "ot_agent_sft": 0.10,
        "opus46_3000x": 0.08,
        "opus46_traceinv": 0.07,
        "arc_agi2_train": 0.05,
        "finepdfs_mix": 0.05,
    }
    assert sum(s2_srcs.values()) == pytest.approx(1.0)
    for src_name in s2_srcs:
        get_spec(src_name)
    valid_groups = set(eval_groups())
    assert s2["data"]["decontaminate"]["enabled"] is True
    assert set(s2["data"]["decontaminate"]["groups"]).issubset(valid_groups)

    # s4_v2_rl specifics (GDN-2-1M + Bounded QED + Soft Neg-Eigval enabled; slap_target_groups untouched)
    s4 = cfgs["s4_v2_rl"]
    assert s4["stage_type"] == "rl"
    assert s4["gdn2"]["enabled"] is True
    assert s4["gdn2"]["rank"] == 64
    assert s4["gdn2"]["chunk_size"] == 64
    assert s4["gdn2"]["decay_log_min"] == pytest.approx(-3.0)
    assert s4["gdn2"]["decay_log_max"] == pytest.approx(0.5)
    assert s4["gdn2"]["qed_enabled"] is True
    assert s4["gdn2"]["qed_max_lambda"] == pytest.approx(0.35)
    assert s4["gdn2"]["neg_eigval_max_boost"] == pytest.approx(0.5)
    assert s4["gdn2"]["backend"] == "auto"
    assert s4["train"]["train_gdn2"] is True
    assert s4["train"]["init_from"] == "runs/s2_v2_joint_bptt"
    assert s4["loop"]["train_pass_probs"] == [0.10, 0.35, 0.55]
    assert s4["lsa"]["mode"] == "sparse"
    assert dict(s4["train"]["aux_weights"]) == {"loop_aux": 1.0}
    assert s4["train"]["optim"]["lr_mult"]["loop_adapters"] == 5.0
    assert s4["train"]["optim"]["lr_mult"]["gdn2_adapter"] == 2.0
    assert s4["train"]["optim"]["lr_mult"]["lsa_indexer"] == 2.0
    assert s4["train"]["allowed_groups"] == [
        "loop_adapters",
        "gdn2_adapters",
        "lsa_indexers",
        "router",
        "shared_expert",
        "attention",
        "norms",
        "expert_deltas",
    ]
    assert s4["rl"]["sirsi"]["enabled"] is True
    assert s4["rl"]["sirsi"]["slap_target_groups"] == ["loop_adapters", "router"]
    assert s4["rl"]["sirsi"]["anti_slop_weight"] == pytest.approx(0.05)
    assert s4["rl"]["gate"]["depth"] == 3

    # Old HSSD v3.2 negative-ablation archive configs remain intact
    for old_st in ("s1b_hssd_distill", "s3_long_1m_lsa", "s2_joint_bptt", "s4_rl"):
        old_cfg = load_config(str(cfg_dir / f"{old_st}.yaml"))
        assert old_cfg["hssd"]["enabled"] is True


# ============================================================================ 2. Dataset Registry & formats.py


def test_loop64_fable_sirsi_registry_and_formats_parsing() -> None:
    spec = get_spec("loop64_fable_sirsi")
    assert spec.hf_id == "TassP0lar/Loop64-Fable-SIRSI-v1"
    assert spec.config == "sft_core"
    assert spec.split == "train"
    assert spec.role == "sft"
    assert spec.format == "messages"
    assert spec.license == "cc-by-4.0"
    assert spec.stages == ["sft", "reasoning", "agentic"]
    assert spec.require_reasoning is True
    assert spec.verified is True

    # Exact sft_core Arrow row layout including tool_calls_json + tools_json + empty reasoning on user/tool
    tools_payload = [
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read a workspace file",
                "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
            },
        }
    ]
    tc_payload = [
        {
            "id": "call_1",
            "type": "function",
            "function": {"name": "read_file", "arguments": json.dumps({"path": "src/limiter.py"})},
        }
    ]
    agentic_row = {
        "id": "seed_cot_agentic_test",
        "domain": "agentic",
        "messages": [
            {"role": "user", "content": "Fix utc cutoff bug.", "reasoning": "", "tool_calls_json": "", "name": ""},
            {
                "role": "assistant",
                "content": "",
                "reasoning": "Inspect src/limiter.py first to verify timezone handling.",
                "tool_calls_json": json.dumps(tc_payload),
                "name": "",
            },
            {
                "role": "tool",
                "content": "def allow(now=None): ...",
                "reasoning": "",
                "tool_calls_json": "",
                "name": "read_file",
            },
            {
                "role": "assistant",
                "content": "Fixed by using timezone-aware UTC datetimes.",
                "reasoning": "Replace naive datetime.utcnow() with datetime.now(timezone.utc) and verify.",
                "tool_calls_json": "",
                "name": "",
            },
        ],
        "prompt": "Fix utc cutoff bug.",
        "thinking": "Replace naive datetime.utcnow() with datetime.now(timezone.utc) and verify.",
        "response": "Fixed by using timezone-aware UTC datetimes.",
        "tools_json": json.dumps(tools_payload),
        "verdict": "KEEP",
        "triage_reasons": ["golden_seed_verified"],
        "extracted_answer": "",
    }
    exs = normalize_row(agentic_row, spec, 0)
    assert len(exs) == 1
    ex = exs[0]
    assert validate_example(ex, spec) is None
    assert ex.tools is not None and len(ex.tools) == 1
    assert ex.tools[0]["function"]["name"] == "read_file"
    assert ex.messages is not None and len(ex.messages) == 4
    assert ex.messages[1]["tool_calls"] == [
        {"type": "function", "function": {"name": "read_file", "arguments": '{"path": "src/limiter.py"}'}}
    ]
    assert ex.messages[2]["name"] == "read_file"
    rendered = "".join(t for t, _ in render_conversation(ex.messages, tools=ex.tools))
    assert "<tools>" in rendered and "<tool_call>" in rendered and "<tool_response>" in rendered
    assert "<think>" in rendered

    # Also test default empty tools_json="[]" produces ex.tools is None (no bogus # Tools header)
    math_row = {
        "id": "seed_cot_math_test",
        "domain": "math",
        "messages": [
            {"role": "user", "content": "Compute 2+2.", "reasoning": "", "tool_calls_json": "", "name": ""},
            {
                "role": "assistant",
                "content": "\\boxed{4}",
                "reasoning": "Add 2 and 2 to obtain 4; verify 4 - 2 = 2.",
                "tool_calls_json": "",
                "name": "",
            },
        ],
        "tools_json": "[]",
    }
    ex_m = normalize_row(math_row, spec, 1)[0]
    assert validate_example(ex_m, spec) is None
    assert ex_m.tools is None

    # If the real sft_core.parquet exists in this checkout or main repo, validate every single row!
    parquet_candidates = [
        ROOT / "runs" / "datasets" / "Loop64-Fable-SIRSI-v1" / "data" / "sft_core.parquet",
        pathlib.Path("/Users/tianrui6641/Qwen3.6-35B-A3B-Loop60/runs/datasets/Loop64-Fable-SIRSI-v1/data/sft_core.parquet"),
    ]
    pq_path = next((p for p in parquet_candidates if p.is_file()), None)
    if pq_path is not None:
        import pyarrow.parquet as pq

        tbl = pq.read_table(pq_path)
        n_tools = 0
        n_tc_turns = 0
        for i in range(len(tbl)):
            row = {k: tbl[k][i].as_py() for k in tbl.column_names}
            parsed = normalize_row(row, spec, i)
            assert len(parsed) == 1
            assert validate_example(parsed[0], spec) is None, f"unexpected drop on sft_core row {i}"
            if parsed[0].tools:
                n_tools += 1
            for m in parsed[0].messages or []:
                if m.get("tool_calls"):
                    n_tc_turns += 1
        assert len(tbl) > 0
        assert n_tools >= 1 and n_tc_turns >= 1


# ============================================================================ 3. mHC Telemetry & Gradients (#392)


def _toy_v2_stack_cfg(stage_name: str, *, checkpoint: bool = False) -> dict[str, Any]:
    real_cfg = load_config(str(ROOT / "configs" / "stages" / f"{stage_name}.yaml")).to_dict()
    real_cfg["model"] = {
        "name": "toy",
        "toy": {
            "num_hidden_layers": 8,
            "hidden_size": 32,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 8,
            "num_experts": 4,
            "num_experts_per_tok": 2,
        },
    }
    real_cfg["loop"].update(
        {
            "prelude_end": 0,
            "core_end": 8,
            "num_passes": 3,
            "target_virtual_layers": 24,
            "bridge_rank": 8,
            "attn_adapter_rank": 4,
            "router_adapter_rank": 4,
            "checkpoint": checkpoint,
        }
    )
    real_cfg["lsa"].update(
        {
            "index_dim": 8,
            "index_on": "all",
            "pool": 2,
            "block_size": 8,
            "avg_bucket_keys": 4,
            "max_bucket_keys": 8,
            "num_select": 2,
            "window": 8,
            "sinks": 2,
            "distill_query_frac": 1.0,
            "query_chunk": 16,
        }
    )
    return real_cfg


@pytest.mark.parametrize("stage_name", ["s3_v2_long_1m_lsa", "s2_v2_joint_bptt", "s4_v2_rl"])
def test_mhc_bridge_gates_router_attn_grads_and_telemetry(stage_name: str) -> None:
    torch.manual_seed(42)
    cfg = _toy_v2_stack_cfg(stage_name, checkpoint=False)
    st = build_stack(cfg)
    assert st.hssd is None
    if stage_name == "s3_v2_long_1m_lsa":
        assert st.gdn2 is None
    else:
        assert st.gdn2 is not None
    assert st.loop is not None and st.lsa is not None

    # Simulate healed s1_loop_a6b checkpoint init (non-zero gates, router/attn LoRA up weights, step_emb)
    # so every mHC component (down, up, norm, mix_logits, step_emb, gates, pass_router, pass_attn) is live.
    with torch.no_grad():
        st.loop.adapters.gates.fill_(0.15)
        st.loop.adapters.bridge.step_emb.uniform_(-0.05, 0.05)  # type: ignore[union-attr]
        assert st.loop.adapters.pass_router is not None
        st.loop.adapters.pass_router.up.uniform_(-0.05, 0.05)
        st.loop.adapters.pass_router.repel.fill_(0.05)
        assert st.loop.adapters.pass_attn is not None
        for pdict in st.loop.adapters.pass_attn.layers.values():
            for proj in pdict.values():
                proj.up.uniform_(-0.05, 0.05)

    # All mHC bridge / gates / pass_router / pass_attn params must require grad
    mhc_named = [
        (n, p)
        for n, p in st.named_trainable.items()
        if n.startswith(
            ("loop_adapters.bridge.", "loop_adapters.gates", "loop_adapters.pass_router.", "loop_adapters.pass_attn.")
        )
    ]
    assert mhc_named, f"no mHC loop_adapters params found in {stage_name}"
    for n, p in mhc_named:
        assert p.requires_grad, f"{stage_name}: {n} does not require_grad"

    st.loop.set_active_passes(3)
    ids = torch.randint(0, 256, (1, 24))
    h = st.hidden(ids)
    logits = st.model.lm_head(h)
    loss = F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]), ids[:, 1:].reshape(-1))
    loss.backward()

    for n, p in mhc_named:
        assert p.grad is not None, f"{stage_name}: {n} received None grad"
        assert torch.isfinite(p.grad).all(), f"{stage_name}: {n} received non-finite grad"
        assert float(p.grad.abs().max()) > 0.0, f"{stage_name}: {n} received all-zero grad"

    # Verify LoopController.stats() telemetry (#392)
    stats = st.loop.stats()
    for p_num in (2, 3):
        assert f"loop/gate_abs_mean_p{p_num}" in stats
        for stream_nm in ("h_prev", "anchor", "h_prev_rms_matched"):
            assert f"loop/bridge_w_{stream_nm}_p{p_num}" in stats
        assert f"loop/bridge_subspace_up_abs_mean_p{p_num}" in stats
        assert stats[f"loop/bridge_subspace_up_abs_mean_p{p_num}"] > 0.0
        assert f"loop/bridge_step_emb_norm_p{p_num}" in stats
        assert stats[f"loop/bridge_step_emb_norm_p{p_num}"] > 0.0


# ============================================================================ 4. Native-GDN + TF 5.24.0 Cache Audit


@dataclass
class _MockTF524TextOut:
    last_hidden_state: torch.Tensor
    cache: Any = None


@dataclass
class _MockTF524LMOut:
    logits: torch.Tensor
    loss: torch.Tensor | None = None
    cache: Any = None


class _MockHF524DynamicLayer:
    supports_early_init = True

    def __init__(self) -> None:
        self.keys: torch.Tensor | None = None
        self.values: torch.Tensor | None = None

    @property
    def is_initialized(self) -> bool:
        return self.keys is not None and self.keys.numel() > 0

    def update(self, k: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.keys is None or self.keys.numel() == 0:
            self.keys = k
            self.values = v
        else:
            self.keys = torch.cat([self.keys, k], dim=-2)
            self.values = torch.cat([self.values, v], dim=-2)
        return self.keys, self.values

    def get_seq_length(self) -> int:
        if self.keys is None or self.keys.ndim < 3 or self.keys.numel() == 0:
            return 0
        return int(self.keys.shape[-2])


class _MockHF524LinearAttentionLayer:
    """Faithful mirror of `transformers 5.24.0` `LinearAttentionLayer` (in-place `.copy_`, no `get_conv_state`)."""

    supports_early_init = False

    def __init__(self) -> None:
        self.conv_states: list[torch.Tensor | None] = [None]
        self.recurrent_states: list[torch.Tensor | None] = [None]
        self.conv_kernel_size: dict[int, int] = {}
        self.is_conv_states_initialized: dict[int, bool] = {0: False}
        self.is_recurrent_states_initialized: dict[int, bool] = {0: False}
        self.has_previous_state: dict[int, bool] = {0: False}

    def update_conv_state(
        self, conv_states: torch.Tensor, state_idx: int = 0, conv_kernel_size: int | None = None
    ) -> torch.Tensor:
        if not self.is_conv_states_initialized[state_idx]:
            self.conv_kernel_size[state_idx] = (
                conv_states.shape[-1] if conv_kernel_size is None else int(conv_kernel_size)
            )
            self.conv_states[state_idx] = torch.zeros(
                (*conv_states.shape[:-1], self.conv_kernel_size[state_idx]),
                dtype=conv_states.dtype,
                device=conv_states.device,
            )
            self.is_conv_states_initialized[state_idx] = True
        if not self.has_previous_state[state_idx]:
            full_cs = conv_states
            if full_cs.shape[-1] < self.conv_kernel_size[state_idx]:
                full_cs = F.pad(full_cs, (self.conv_kernel_size[state_idx] - full_cs.shape[-1], 0), value=0)
            self.has_previous_state[state_idx] = True
        else:
            prev = self.conv_states[state_idx]
            assert prev is not None
            full_cs = torch.cat([prev, conv_states], dim=-1)
        dst = self.conv_states[state_idx]
        assert dst is not None
        dst.copy_(full_cs[..., -self.conv_kernel_size[state_idx] :])
        return full_cs

    def update_recurrent_state(self, recurrent_states: torch.Tensor, state_idx: int = 0) -> torch.Tensor:
        if not self.is_recurrent_states_initialized[state_idx]:
            self.recurrent_states[state_idx] = torch.zeros_like(recurrent_states)
            self.is_recurrent_states_initialized[state_idx] = True
        dst = self.recurrent_states[state_idx]
        assert dst is not None
        dst.copy_(recurrent_states)
        return dst


class _MockHF524DynamicCache:
    """Faithful `transformers 5.24.0` `DynamicCache` (`layers[i]`, no `get_conv_state`, 1-arg `get_recurrent_state`)."""

    def __init__(self, config: Any) -> None:
        self.layers: list[Any] = [
            _MockHF524DynamicLayer() if lt == "full_attention" else _MockHF524LinearAttentionLayer()
            for lt in config.layer_types
        ]

    def __len__(self) -> int:
        return len(self.layers)

    @property
    def is_initialized(self) -> bool:
        init_layers = [lyr for lyr in self.layers if lyr.supports_early_init]
        return len(init_layers) > 0 and all(lyr.is_initialized for lyr in init_layers)

    def update(self, k: torch.Tensor, v: torch.Tensor, layer_idx: int, *_a: Any, **_kw: Any):
        return self.layers[layer_idx].update(k, v)

    def has_previous_state(self, layer_idx: int) -> bool:
        return bool(self.layers[layer_idx].has_previous_state[0])

    def update_conv_state(self, conv_states: torch.Tensor, layer_idx: int, **kw: Any) -> torch.Tensor:
        return self.layers[layer_idx].update_conv_state(conv_states, **kw)

    def get_recurrent_state(self, layer_idx: int) -> torch.Tensor | None:
        lyr = self.layers[layer_idx]
        return lyr.recurrent_states[0] if lyr.has_previous_state[0] else None

    def update_recurrent_state(self, recurrent_states: torch.Tensor, layer_idx: int) -> torch.Tensor:
        return self.layers[layer_idx].update_recurrent_state(recurrent_states)

    def get_seq_length(self, layer_idx: int = 0) -> int:
        if isinstance(self.layers[layer_idx], _MockHF524DynamicLayer):
            return self.layers[layer_idx].get_seq_length()
        first_full = next(lyr for lyr in self.layers if isinstance(lyr, _MockHF524DynamicLayer))
        return first_full.get_seq_length()

    def get_mask_sizes(self, cache_position: torch.Tensor, layer_idx: int) -> tuple[int, int]:
        return self.get_seq_length(layer_idx) + int(cache_position.shape[0]), 0


def _patch_toy_with_tf524_dynamic_cache(model: ToyForCausalLM) -> None:
    """Patch `ToyForCausalLM` to use strict `transformers 5.24.0` `cache=` + `_MockHF524DynamicCache`."""
    for layer in model.model.layers:
        if layer.layer_type == "linear_attention":
            lin = layer.linear_attn

            def _lin_fwd(
                _self: nn.Module,
                hidden_states: torch.Tensor,
                *,
                cache: Any = None,
                attention_mask: torch.Tensor | None = None,
            ) -> torch.Tensor:
                del attention_mask
                b, t, _ = hidden_states.shape
                qkv = _self.in_proj_qkv(hidden_states).transpose(1, 2)
                if cache is not None:
                    had_prev = bool(cache.has_previous_state(_self.layer_idx))
                    full_cs = cache.update_conv_state(qkv, _self.layer_idx, conv_kernel_size=_self.ck)
                    if had_prev:
                        seq = full_cs[..., -(t + _self.ck - 1) :]
                    else:
                        seq = torch.cat([qkv.new_zeros(b, qkv.shape[1], _self.ck - 1), qkv], dim=-1)
                else:
                    seq = torch.cat([qkv.new_zeros(b, qkv.shape[1], _self.ck - 1), qkv], dim=-1)
                qkv_act = F.silu(_self.conv1d(seq))[..., -t:].transpose(1, 2)
                q, k, v = qkv_act.view(b, t, 3, _self.nh, _self.d).unbind(2)
                q = F.normalize(q.float(), dim=-1)
                k = F.normalize(k.float(), dim=-1)
                v = v.float()
                bb, aa = _self.in_proj_ba(hidden_states).float().chunk(2, dim=-1)
                beta = torch.sigmoid(bb)
                g = -_self.A_log.float().exp() * F.softplus(aa + _self.dt_bias.float())
                init_state = cache.get_recurrent_state(_self.layer_idx) if cache is not None else None
                core_out, final_state = _self.chunk_gated_delta_rule(
                    q, k, v, g=g, beta=beta, initial_state=init_state, output_final_state=cache is not None
                )
                if cache is not None and final_state is not None:
                    cache.update_recurrent_state(final_state, _self.layer_idx)
                o = _self.norm(core_out.to(hidden_states.dtype))
                return _self.out_proj(o.reshape(b, t, _self.nh * _self.d))

            lin.forward = types.MethodType(_lin_fwd, lin)  # type: ignore[method-assign]
        else:
            attn = layer.self_attn
            orig_attn_fwd = attn.forward

            def _attn_fwd(
                _self: nn.Module,
                hidden_states: torch.Tensor,
                position_embeddings: tuple[torch.Tensor, torch.Tensor],
                attention_mask: torch.Tensor | None = None,
                cache: Any = None,
                cache_position: torch.Tensor | None = None,
                _orig=orig_attn_fwd,
            ) -> torch.Tensor:
                return _orig(hidden_states, position_embeddings, attention_mask, cache, cache_position)

            attn.forward = types.MethodType(_attn_fwd, attn)  # type: ignore[method-assign]

        def _dec_fwd(
            _self: nn.Module,
            hidden_states: torch.Tensor,
            *,
            position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
            attention_mask: torch.Tensor | None = None,
            position_ids: torch.Tensor | None = None,
            cache: Any = None,
            cache_position: torch.Tensor | None = None,
        ) -> torch.Tensor:
            residual = hidden_states
            h = _self.input_layernorm(hidden_states)
            if _self.layer_type == "linear_attention":
                h = _self.linear_attn(h, cache=cache, attention_mask=attention_mask)
            else:
                h = _self.self_attn(
                    hidden_states=h,
                    position_embeddings=position_embeddings,
                    attention_mask=attention_mask,
                    cache=cache,
                    cache_position=cache_position,
                )
            h = residual + h
            residual = h
            h = _self.mlp(_self.post_attention_layernorm(h))
            return residual + h

        layer.forward = types.MethodType(_dec_fwd, layer)  # type: ignore[method-assign]

    tm = model.model

    def _tm_fwd(
        _self: nn.Module,
        input_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        cache: Any = None,
        use_cache: bool = False,
        attention_mask: Any = None,
    ) -> _MockTF524TextOut:
        del attention_mask
        h = _self.embed_tokens(input_ids) if inputs_embeds is None else inputs_embeds
        if use_cache and cache is None:
            cache = _MockHF524DynamicCache(_self.config)
        start = cache.get_seq_length() if cache is not None else 0
        cache_position = torch.arange(start, start + h.shape[1], device=h.device)
        position_ids = cache_position[None].expand(h.shape[0], -1)
        pos = _self.rotary_emb(h, position_ids)
        for lyr in _self.layers[: _self.config.num_hidden_layers]:
            h = lyr(
                h,
                position_embeddings=pos,
                attention_mask=None,
                position_ids=position_ids,
                cache=cache,
                cache_position=cache_position,
            )
        return _MockTF524TextOut(_self.norm(h), cache=cache)

    tm.forward = types.MethodType(_tm_fwd, tm)  # type: ignore[method-assign]

    def _lm_fwd(
        _self: nn.Module,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
        cache: Any = None,
        use_cache: bool = False,
        attention_mask: Any = None,
    ) -> _MockTF524LMOut:
        out = _self.model(input_ids=input_ids, cache=cache, use_cache=use_cache, attention_mask=attention_mask)
        logits = _self.lm_head(out.last_hidden_state)
        return _MockTF524LMOut(logits=logits, cache=out.cache)

    model.forward = types.MethodType(_lm_fwd, model)  # type: ignore[method-assign]


def test_maybe_create_initial_cache_prefers_moe_then_dense_then_dynamic_cache(monkeypatch) -> None:
    class _FakeMoeCache:
        def __init__(self, cfg: Any) -> None:
            self.cfg = cfg

    class _FakeDenseCache:
        def __init__(self, cfg: Any) -> None:
            self.cfg = cfg

    class _FakeDynamicCache:
        def __init__(self, config: Any = None) -> None:
            self.config = config

    # 1) MoE config prefers Qwen3_5MoeDynamicCache when available
    fake_tf = types.ModuleType("transformers")
    fake_models = types.ModuleType("transformers.models")
    fake_q35_moe = types.ModuleType("transformers.models.qwen3_5_moe")
    fake_q35_moe_mod = types.ModuleType("transformers.models.qwen3_5_moe.modeling_qwen3_5_moe")
    fake_q35_moe_mod.Qwen3_5MoeDynamicCache = _FakeMoeCache  # type: ignore[attr-defined]
    fake_q35 = types.ModuleType("transformers.models.qwen3_5")
    fake_q35_mod = types.ModuleType("transformers.models.qwen3_5.modeling_qwen3_5")
    fake_q35_mod.Qwen3_5DynamicCache = _FakeDenseCache  # type: ignore[attr-defined]
    fake_cu = types.ModuleType("transformers.cache_utils")
    fake_cu.DynamicCache = _FakeDynamicCache  # type: ignore[attr-defined]

    for k, m in {
        "transformers": fake_tf,
        "transformers.models": fake_models,
        "transformers.models.qwen3_5_moe": fake_q35_moe,
        "transformers.models.qwen3_5_moe.modeling_qwen3_5_moe": fake_q35_moe_mod,
        "transformers.models.qwen3_5": fake_q35,
        "transformers.models.qwen3_5.modeling_qwen3_5": fake_q35_mod,
        "transformers.cache_utils": fake_cu,
    }.items():
        monkeypatch.setitem(sys.modules, k, m)

    moe_tm = types.SimpleNamespace(config=types.SimpleNamespace(model_type="qwen3_5_moe_text", num_experts=256))
    c_moe = _maybe_create_initial_cache(moe_tm)  # type: ignore[arg-type]
    assert isinstance(c_moe, _FakeMoeCache)

    dense_tm = types.SimpleNamespace(config=types.SimpleNamespace(model_type="qwen3_5_text"))
    c_dense = _maybe_create_initial_cache(dense_tm)  # type: ignore[arg-type]
    assert isinstance(c_dense, _FakeDenseCache)

    # 2) transformers 5.24.0 unified DynamicCache fallback (no specialized Qwen3_5*DynamicCache classes)
    delattr(fake_q35_moe_mod, "Qwen3_5MoeDynamicCache")
    delattr(fake_q35_mod, "Qwen3_5DynamicCache")
    c_dyn = _maybe_create_initial_cache(moe_tm)  # type: ignore[arg-type]
    assert isinstance(c_dyn, _FakeDynamicCache)
    assert c_dyn.config is moe_tm.config


def test_replay_kv_cache_proxy_checkpoint_safety_hf524_layers_and_list_caches(monkeypatch) -> None:
    import qloop.train.bptt as bptt_mod

    cfg = ToyConfig(num_hidden_layers=4, hidden_size=16, num_attention_heads=2, num_key_value_heads=1, head_dim=8)
    base = _MockHF524DynamicCache(cfg)
    proxy = _ReplayKVCacheProxy(base)
    assert bool(proxy) is True
    assert len(proxy) == 4
    assert isinstance(proxy.layers, list)

    # Segment 0 forward: has_previous_state(0) is False, update_conv_state mutates base in-place
    assert proxy.has_previous_state(0) is False
    q0 = torch.randn(1, 8, 6, requires_grad=True)
    cs0 = proxy.update_conv_state(q0, 0, conv_kernel_size=4)
    assert cs0.requires_grad and cs0.shape == (1, 8, 6)
    assert base.has_previous_state(0) is True
    assert proxy.get_recurrent_state(0) is None
    r0 = torch.randn(1, 2, 4, 4)
    proxy.update_recurrent_state(r0, 0)
    assert base.get_recurrent_state(0) is not None

    # Simulate checkpoint backward recomputation on Segment 0: must STILL see pre-segment False / None!
    monkeypatch.setattr(bptt_mod, "_in_backward", lambda: True)
    assert proxy.has_previous_state(0) is False
    assert proxy.layers[0].has_previous_state[0] is False
    cs0_bwd = proxy.update_conv_state(q0, 0, conv_kernel_size=4)
    assert torch.equal(cs0_bwd, cs0)
    assert proxy.get_recurrent_state(0) is None
    # Writing in backward must NOT overwrite Segment 0's saved recurrent state in base!
    proxy.update_recurrent_state(torch.zeros_like(r0), 0)
    assert torch.allclose(base.get_recurrent_state(0), r0)  # type: ignore[arg-type]
    monkeypatch.setattr(bptt_mod, "_in_backward", lambda: False)

    # Segment 1: fresh proxy must see Segment 0's conv & recurrent state (even with NO get_conv_state on base!)
    seg0_conv_tail = base.layers[0].conv_states[0].clone()
    seg0_rec = base.layers[0].recurrent_states[0].clone()
    proxy1 = _ReplayKVCacheProxy(base)
    assert proxy1.has_previous_state(0) is True
    q1 = torch.randn(1, 8, 6, requires_grad=True)
    cs1 = proxy1.update_conv_state(q1, 0, conv_kernel_size=4)
    # Must concatenate Segment 0's 4-token conv tail + Segment 1's 6 tokens = 10 tokens!
    assert cs1.shape == (1, 8, 10)
    assert torch.allclose(cs1[..., :4], seg0_conv_tail)
    assert torch.allclose(cs1[..., 4:], q1)
    assert cs1.requires_grad
    rec1_in = proxy1.get_recurrent_state(0)
    assert rec1_in is not None and torch.allclose(rec1_in, seg0_rec)
    r1 = torch.randn(1, 2, 4, 4)
    proxy1.update_recurrent_state(r1, 0)

    # Checkpoint backward recomputation on Segment 1 must see Segment 0's tail (NOT Segment 1's mutated tail)!
    monkeypatch.setattr(bptt_mod, "_in_backward", lambda: True)
    assert proxy1.has_previous_state(0) is True
    cs1_bwd = proxy1.update_conv_state(q1, 0, conv_kernel_size=4)
    assert torch.allclose(cs1_bwd, cs1)
    rec1_bwd = proxy1.get_recurrent_state(0)
    assert rec1_bwd is not None and torch.allclose(rec1_bwd, seg0_rec)
    monkeypatch.setattr(bptt_mod, "_in_backward", lambda: False)

    # List-attribute cache (ToyHybridCache): in-place `.copy_()` on backing must not corrupt backward snapshot
    toy_c = ToyHybridCache(cfg)
    orig_t = torch.ones(1, 8, 3)
    toy_c.conv_states[0] = orig_t.clone()
    p_toy = _ReplayKVCacheProxy(toy_c)
    p_toy.conv_states[0].copy_(torch.full_like(orig_t, 9.0))
    monkeypatch.setattr(bptt_mod, "_in_backward", lambda: True)
    assert torch.allclose(p_toy.conv_states[0], orig_t)
    monkeypatch.setattr(bptt_mod, "_in_backward", lambda: False)


@pytest.mark.parametrize("readonly_base", [False, True])
def test_replay_proxy_inplace_conv_and_rec_update_checkpoint_grad_parity(readonly_base: bool) -> None:
    """Regression test (#397 Finding 1): 1-token in-place `causal_conv1d_update` / recurrent update
    via `get_conv_state`/`get_recurrent_state` must match uncheckpointed gradients, preserve live
    forward updates when `readonly_base=False`, and never mutate `base` when `readonly_base=True`."""
    from torch.utils.checkpoint import checkpoint as torch_ckpt

    class _CacheWithGetters:
        def __init__(self, conv: torch.Tensor, rec: torch.Tensor) -> None:
            self._conv = conv
            self._rec = rec

        def has_previous_state(self, layer_idx: int) -> bool:
            del layer_idx
            return True

        def get_conv_state(self, layer_idx: int) -> torch.Tensor:
            del layer_idx
            return self._conv

        def get_recurrent_state(self, layer_idx: int) -> torch.Tensor:
            del layer_idx
            return self._rec

        def update_conv_state(
            self, conv_states: torch.Tensor, layer_idx: int, conv_kernel_size: int | None = None
        ) -> torch.Tensor:
            del layer_idx
            k = int(conv_kernel_size or self._conv.shape[-1])
            full = torch.cat([self._conv, conv_states], dim=-1)
            self._conv.copy_(full[..., -k:])
            return full

        def update_recurrent_state(self, recurrent_states: torch.Tensor, layer_idx: int) -> torch.Tensor:
            del layer_idx
            self._rec.copy_(recurrent_states)
            return self._rec

    torch.manual_seed(123)
    w_conv = nn.Parameter(torch.randn(4, 3))
    w_proj = nn.Parameter(torch.randn(4, 4))
    x_init = torch.randn(1, 4, 1)
    init_conv = torch.randn(1, 4, 3)
    init_rec = torch.randn(1, 4, 4)

    def _step_fn(
        x_in: torch.Tensor,
        w_c: torch.Tensor,
        w_p: torch.Tensor,
        cache_obj: Any,
    ) -> torch.Tensor:
        # Mirror 1-token `causal_conv1d_update`: mutate conv_state in-place and compute output from it
        cs = cache_obj.get_conv_state(0)
        assert cs is not None
        prev_snap = cs.clone()
        new_cs = torch.cat([prev_snap[..., 1:], x_in.detach()], dim=-1)
        cs.copy_(new_cs)
        # Output depends on pre-update conv tail + live x_in (matching causal_conv1d_update semantics)
        win = torch.cat([prev_snap[..., 1:], x_in], dim=-1)
        y_conv = (win * w_c.unsqueeze(0)).sum(dim=-1)  # (1, 4)
        rs = cache_obj.get_recurrent_state(0)
        assert rs is not None
        r_prev = rs.clone()
        rs.add_(0.25 * y_conv.detach().unsqueeze(-1))
        y_rec = torch.einsum("bi,bij->bj", y_conv, r_prev) @ w_p
        return y_rec

    # Reference: uncheckpointed direct execution on a clean base cache
    base_ref = _CacheWithGetters(init_conv.clone(), init_rec.clone())
    x_ref = x_init.clone().requires_grad_(True)
    wc_ref = w_conv.detach().clone().requires_grad_(True)
    wp_ref = w_proj.detach().clone().requires_grad_(True)
    out_ref = _step_fn(x_ref, wc_ref, wp_ref, base_ref)
    out_ref.pow(2).sum().backward()

    # Checkpointed execution through `_ReplayKVCacheProxy`
    base_chk = _CacheWithGetters(init_conv.clone(), init_rec.clone())
    proxy_chk = _ReplayKVCacheProxy(base_chk, readonly_base=readonly_base)
    x_chk = x_init.clone().requires_grad_(True)
    wc_chk = w_conv.detach().clone().requires_grad_(True)
    wp_chk = w_proj.detach().clone().requires_grad_(True)

    def _wrapped(xi: torch.Tensor, wc: torch.Tensor, wp: torch.Tensor) -> torch.Tensor:
        return _step_fn(xi, wc, wp, proxy_chk)

    out_chk = torch_ckpt(_wrapped, x_chk, wc_chk, wp_chk, use_reentrant=False)
    assert isinstance(out_chk, torch.Tensor)
    out_chk.pow(2).sum().backward()

    assert torch.allclose(out_chk, out_ref, atol=1e-6, rtol=1e-6)
    assert x_chk.grad is not None and x_ref.grad is not None
    assert wc_chk.grad is not None and wc_ref.grad is not None
    assert wp_chk.grad is not None and wp_ref.grad is not None
    assert torch.allclose(x_chk.grad, x_ref.grad, atol=1e-6, rtol=1e-6)
    assert torch.allclose(wc_chk.grad, wc_ref.grad, atol=1e-6, rtol=1e-6)
    assert torch.allclose(wp_chk.grad, wp_ref.grad, atol=1e-6, rtol=1e-6)

    if readonly_base:
        assert torch.equal(base_chk._conv, init_conv)
        assert torch.equal(base_chk._rec, init_rec)
    else:
        assert torch.allclose(base_chk._conv, base_ref._conv, atol=1e-6)
        assert torch.allclose(base_chk._rec, base_ref._rec, atol=1e-6)

    # Also verify `get_conv_state`/`get_recurrent_state` reflect `update_*` within the same forward
    base_seq = _CacheWithGetters(init_conv.clone(), init_rec.clone())
    proxy_seq = _ReplayKVCacheProxy(base_seq, readonly_base=readonly_base)
    upd_c = torch.full((1, 4, 1), 7.5)
    upd_r = torch.full((1, 4, 4), -3.25)
    proxy_seq.update_conv_state(upd_c, 0, conv_kernel_size=3)
    proxy_seq.update_recurrent_state(upd_r, 0)
    got_c = proxy_seq.get_conv_state(0)
    got_r = proxy_seq.get_recurrent_state(0)
    assert got_c is not None and torch.allclose(got_c[..., -1:], upd_c)
    assert got_r is not None and torch.allclose(got_r, upd_r)
    if readonly_base:
        assert torch.equal(base_seq._conv, init_conv)
        assert torch.equal(base_seq._rec, init_rec)
    else:
        assert torch.allclose(base_seq._conv[..., -1:], upd_c)
        assert torch.allclose(base_seq._rec, upd_r)


def test_replay_proxy_layer_fp8_mask_sizes_bool_hps_zero_len_bwd_and_list_sync(monkeypatch) -> None:
    """Regression tests for #397 Findings 2, 3, and Nit 1:
    - `get_mask_sizes(int)` and `proxy.layers[i].{get_mask_sizes, is_initialized}` under `fp8_prefix_kv`
    - `_ReplayCacheLayerProxy` plain-`bool` `has_previous_state` + guarded `__setattr__` + `reset_segment()`
    - 0-length `update()` during `_in_backward()` when `readonly_base=False` and `slot not in live_map`
    - `_SlotStateListProxy` negative index & slicing after external `base` list extension."""
    import qloop.train.bptt as bptt_mod
    from qloop.model.offload import FP8KVCacheQuantizer
    from qloop.train.long_calib import _extract_fp8_prefix_kv_

    # --- Finding 2a & 2b: `get_mask_sizes(int)` + `proxy.layers[full_idx].{get_mask_sizes, is_initialized}` ---
    cfg = ToyConfig(num_hidden_layers=4, hidden_size=16, num_attention_heads=2, num_key_value_heads=1, head_dim=8)
    base524 = _MockHF524DynamicCache(cfg)
    full_idx = next(i for i, lt in enumerate(cfg.layer_types) if lt == "full_attention")
    k_pref = torch.randn(1, 1, 16, 8)
    v_pref = torch.randn(1, 1, 16, 8)
    base524.update(k_pref, v_pref, full_idx)
    fp8_q = FP8KVCacheQuantizer()
    fp8_kv = _extract_fp8_prefix_kv_(base524, fp8_q)
    if fp8_q.supports_fp8(k_pref.device):
        assert base524.layers[full_idx].is_initialized is False  # emptied by _extract_fp8_prefix_kv_
        proxy_fp8 = _ReplayKVCacheProxy(base524, readonly_base=True, fp8_prefix_kv=fp8_kv)
        assert proxy_fp8.get_mask_sizes(8, full_idx) == (24, 0)
        assert proxy_fp8.get_mask_sizes(torch.arange(16, 24), full_idx) == (24, 0)
        assert proxy_fp8.layers[full_idx].is_initialized is True
        assert proxy_fp8.layers[full_idx].get_mask_sizes(8) == (24, 0)
        assert proxy_fp8.layers[full_idx].get_mask_sizes(torch.arange(16, 24)) == (24, 0)

    # --- Finding 2c: plain-`bool` `has_previous_state` + guarded `__setattr__` + `reset_segment()` ---
    class _BoolHpsLayer:
        def __init__(self) -> None:
            self.has_previous_state: bool = False
            self.conv_states: torch.Tensor | None = torch.zeros(1, 4, 3)

    class _BoolHpsCache:
        def __init__(self) -> None:
            self.layers = [_BoolHpsLayer()]

        def __len__(self) -> int:
            return len(self.layers)

    c_bool = _BoolHpsCache()
    p_bool = _ReplayKVCacheProxy(c_bool, readonly_base=False)
    assert p_bool.layers[0].has_previous_state is False
    new_cs = torch.ones(1, 4, 3)
    p_bool.layers[0].has_previous_state = True
    p_bool.layers[0].conv_states = new_cs
    assert c_bool.layers[0].has_previous_state is True
    assert torch.equal(c_bool.layers[0].conv_states, new_cs)  # type: ignore[arg-type]

    monkeypatch.setattr(bptt_mod, "_in_backward", lambda: True)
    assert p_bool.layers[0].has_previous_state is False
    assert torch.equal(p_bool.layers[0].conv_states, torch.zeros(1, 4, 3))
    # Guarded setattr in backward must be a no-op and leave base intact
    p_bool.layers[0].has_previous_state = False
    p_bool.layers[0].conv_states = torch.full((1, 4, 3), 99.0)
    assert c_bool.layers[0].has_previous_state is True
    assert torch.equal(c_bool.layers[0].conv_states, new_cs)  # type: ignore[arg-type]
    monkeypatch.setattr(bptt_mod, "_in_backward", lambda: False)

    # After `reset_segment()`, Segment 1 snapshots must refresh to `True` / `new_cs`
    p_bool.reset_segment()
    monkeypatch.setattr(bptt_mod, "_in_backward", lambda: True)
    assert p_bool.layers[0].has_previous_state is True
    assert torch.equal(p_bool.layers[0].conv_states, new_cs)
    monkeypatch.setattr(bptt_mod, "_in_backward", lambda: False)

    # Under `readonly_base=True`, mutations stay in overlay and never touch `_base_layer`
    c_ro = _BoolHpsCache()
    p_ro = _ReplayKVCacheProxy(c_ro, readonly_base=True)
    p_ro.layers[0].has_previous_state = True
    p_ro.layers[0].conv_states = torch.full((1, 4, 3), 5.0)
    assert p_ro.layers[0].has_previous_state is True
    assert torch.equal(p_ro.layers[0].conv_states, torch.full((1, 4, 3), 5.0))
    assert c_ro.layers[0].has_previous_state is False
    assert torch.equal(c_ro.layers[0].conv_states, torch.zeros(1, 4, 3))  # type: ignore[arg-type]

    # --- Finding 3: 0-length `update()` during `_in_backward()` when `readonly_base=False` & `slot not in live_map` ---
    base_z = _MockHF524DynamicCache(cfg)
    k0 = torch.randn(1, 1, 12, 8)
    v0 = torch.randn(1, 1, 12, 8)
    base_z.update(k0, v0, full_idx)
    p_z = _ReplayKVCacheProxy(base_z, readonly_base=False)
    empty_k = torch.empty(1, 1, 0, 8)
    empty_v = torch.empty(1, 1, 0, 8)

    # Case 3a: neither `live_map` nor `raw_pref` populated -> falls back to `_extract_cached_kv(base, slot)`
    monkeypatch.setattr(bptt_mod, "_in_backward", lambda: True)
    k_ret_a, v_ret_a = p_z.update(empty_k, empty_v, full_idx)
    assert k_ret_a.shape == (1, 1, 12, 8) and v_ret_a.shape == (1, 1, 12, 8)
    assert torch.equal(k_ret_a, k0) and torch.equal(v_ret_a, v0)
    # Case 3b: `raw_pref` populated, `live_map` absent -> returns `raw_pref[slot]`
    k_rp = torch.randn(1, 1, 10, 8)
    v_rp = torch.randn(1, 1, 10, 8)
    object.__getattribute__(p_z, "_raw_prefix_kv")[full_idx] = (k_rp, v_rp)
    k_ret_b, v_ret_b = p_z.update(empty_k, empty_v, full_idx)
    assert torch.equal(k_ret_b, k_rp) and torch.equal(v_ret_b, v_rp)
    monkeypatch.setattr(bptt_mod, "_in_backward", lambda: False)

    # --- Nit 1: `_SlotStateListProxy` after external `base` list extension (negative index & slicing) ---
    toy_ext = ToyHybridCache(cfg)
    toy_ext.conv_states[0] = torch.ones(1, 4, 3)
    p_ext = _ReplayKVCacheProxy(toy_ext, readonly_base=False)
    ext_orig = torch.full((1, 4, 3), 2.5)
    toy_ext.conv_states.append(ext_orig.clone())
    assert len(p_ext.conv_states) == len(toy_ext.conv_states)
    # Negative index & slice work after external append, and in-place mutation does not corrupt backward snapshot
    p_ext.conv_states[-1].copy_(torch.full((1, 4, 3), 8.0))
    assert torch.equal(toy_ext.conv_states[-1], torch.full((1, 4, 3), 8.0))
    monkeypatch.setattr(bptt_mod, "_in_backward", lambda: True)
    assert torch.equal(p_ext.conv_states[-1], ext_orig)
    sl_bwd = p_ext.conv_states[-2:]
    assert len(sl_bwd) == 2 and torch.equal(sl_bwd[-1], ext_orig)
    monkeypatch.setattr(bptt_mod, "_in_backward", lambda: False)

    p_ext_ro = _ReplayKVCacheProxy(toy_ext, readonly_base=True)
    ro_orig = torch.full((1, 4, 3), 3.5)
    toy_ext.conv_states.append(ro_orig.clone())
    p_ext_ro.conv_states[-1].copy_(torch.full((1, 4, 3), 9.0))
    assert torch.equal(p_ext_ro.conv_states[-1], torch.full((1, 4, 3), 9.0))
    assert torch.equal(toy_ext.conv_states[-1], ro_orig)
    assert torch.equal(p_ext_ro.conv_states[-1:][0], torch.full((1, 4, 3), 9.0))


def test_long_calib_and_bptt_native_gdn_loop_lsa_tf524(monkeypatch) -> None:
    torch.manual_seed(7)
    # 1) Long-prefix + short-suffix calibration with `fp8_kv_cache=True` on TF 5.24.0 DynamicCache + Native GDN
    cfg = ToyConfig(
        num_hidden_layers=4,
        hidden_size=32,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        num_experts=4,
        num_experts_per_tok=2,
        vocab_size=128,
    )
    model = ToyForCausalLM(cfg)
    _patch_toy_with_tf524_dynamic_cache(model)
    lsa_mgr = install_lsa(
        model,
        {
            "mode": "hybrid",
            "num_query_passes": 3,
            "index_dim": 8,
            "index_on": "all",
            "pool": 2,
            "block_size": 8,
            "avg_bucket_keys": 4,
            "max_bucket_keys": 8,
            "num_select": 2,
            "window": 8,
            "sinks": 2,
            "distill_query_frac": 1.0,
            "query_chunk": 16,
        },
    )
    ctl = apply_loop(
        model,
        {
            "prelude_end": 0,
            "core_end": 4,
            "num_passes": 3,
            "target_virtual_layers": 12,
            "kv_share": True,
            "bridge_mode": "subspace",
            "bridge_rank": 8,
            "bridge_init": [0.05, 0.85, 0.10],
            "attn_adapter_rank": 4,
            "per_pass_router": True,
            "router_adapter_rank": 4,
            "checkpoint": True,
        },
    )
    with torch.no_grad():
        ctl.adapters.gates.fill_(0.15)
    assert getattr(model, "_hssd_handle", None) is None

    # Force `_maybe_create_initial_cache` to return `_MockHF524DynamicCache` for BPTT test below too
    ids_lc = torch.randint(0, cfg.vocab_size, (1, 32))
    model.train()
    res_lc = prefix_suffix_calibration_step(
        model,
        ids_lc,
        prefix_len=16,
        chunk_size=8,
        labels=ids_lc.clone(),
        backward=True,
        fp8_kv_cache=True,
        lsa_manager=lsa_mgr,
        global_step=100,
        prefix_lsa_mode="sparse",
        sparse_prefix_after_step=80,
    )
    assert res_lc.num_prefix_chunks == 2
    assert res_lc.prefix_len == 16 and res_lc.suffix_len == 16
    assert res_lc.prefix_lsa_mode_used == "sparse"
    assert torch.isfinite(res_lc.loss)
    assert any("lsa_indexer" in k and v > 0 for k, v in res_lc.grad_norms.items())
    assert any("loop_adapters" in k and v > 0 for k, v in res_lc.grad_norms.items())

    # Explicitly verify `_extract_fp8_prefix_kv_` leaves native GDN `conv_states` & `recurrent_states` untouched!
    from qloop.model.offload import FP8KVCacheQuantizer
    from qloop.train.long_calib import _extract_fp8_prefix_kv_

    with torch.no_grad():
        out_p = model.model(input_ids=ids_lc[:, :16], cache=None, use_cache=True)
    raw_c = out_p.cache
    assert isinstance(raw_c, _MockHF524DynamicCache)
    lin_idx = next(i for i, lt in enumerate(model.model.config.layer_types) if lt == "linear_attention")
    full_idx = next(i for i, lt in enumerate(model.model.config.layer_types) if lt == "full_attention")
    conv_before = raw_c.layers[lin_idx].conv_states[0].clone()
    rec_before = raw_c.layers[lin_idx].recurrent_states[0].clone()
    fp8_q = FP8KVCacheQuantizer()
    fp8_kv = _extract_fp8_prefix_kv_(raw_c, fp8_q)
    # Linear layers' conv/recurrent states must be bit-identical before & after FP8 extraction
    assert lin_idx not in fp8_kv
    assert torch.equal(raw_c.layers[lin_idx].conv_states[0], conv_before)
    assert torch.equal(raw_c.layers[lin_idx].recurrent_states[0], rec_before)
    if fp8_q.supports_fp8(ids_lc.device):
        assert full_idx in fp8_kv
        proxy_fp8 = _ReplayKVCacheProxy(raw_c, readonly_base=True, fp8_prefix_kv=fp8_kv)
        assert proxy_fp8.is_initialized is True
        assert proxy_fp8.get_seq_length(full_idx) == 16
        assert proxy_fp8.get_mask_sizes(torch.arange(16, 24), full_idx) == (24, 0)

    # 2) Segmented 4-segment BPTT with `hssd.enabled=False` + `lsa.mode="sparse"` + `checkpoint=True` on TF 5.24.0 cache
    import qloop.train.bptt as bptt_mod

    model.zero_grad(set_to_none=True)
    lsa_mgr.set_mode("sparse")
    monkeypatch.setattr(bptt_mod, "_maybe_create_initial_cache", lambda tm: _MockHF524DynamicCache(tm.config))
    runner = SegmentedKVBPTTRunner(
        model,
        cfg=BPTTConfig(
            enabled=True,
            n_segments=4,
            seg_len=8,
            replay_k=4,
            offload_prior_hidden=False,
            query_sim_chunk=8,
        ),
    )
    ids_bptt = torch.randint(0, cfg.vocab_size, (1, 32))
    res_bptt = runner.run_step(
        ids_bptt,
        labels=ids_bptt.clone(),
        backward=True,
        hssd_aux_weight=0.0,
        lsa_aux_weight=0.2,
    )
    assert res_bptt.num_segments == 4
    assert res_bptt.replayed_tokens_per_seg[1:] == [4, 4, 4]
    assert all(torch.isfinite(torch.tensor(v)) for v in res_bptt.segment_losses)
    # Verify cross-segment replay gradients reach k_proj, v_proj, lsa_indexer, and loop_adapters
    named_p = dict(model.named_parameters())

    def _has_nz_grad(substr: str) -> bool:
        return any(
            substr in n and p.grad is not None and float(p.grad.abs().max()) > 0
            for n, p in named_p.items()
        )

    assert _has_nz_grad("k_proj")
    assert _has_nz_grad("v_proj")
    assert _has_nz_grad("lsa_indexer")
    assert _has_nz_grad("loop_adapters")


def test_trainer_and_grpo_smoke_without_hssd(tmp_path) -> None:
    torch.manual_seed(0)
    # LMTrainer smoke with s3_v2 config (long_calib + hybrid LSA, hssd=False)
    s3_cfg = _toy_v2_stack_cfg("s3_v2_long_1m_lsa", checkpoint=True)
    s3_cfg["train"].update(
        {
            "init_from": None,
            "steps": 2,
            "micro_batch": 1,
            "grad_accum": 1,
            "log_every": 1,
            "ckpt_every": 2,
            "eval_every": 0,
            "prefetch_batches": 0,
        }
    )
    s3_cfg["long_calib"].update(
        {
            "enabled": True,
            "prefix_len": 16,
            "max_prefix_len": 16,
            "suffix_len": 16,
            "chunk_size": 8,
            "calib_every_steps": 2,
        }
    )
    s3_cfg["data"] = {
        "seq_len": 16,
        "respect_doc_boundaries": False,
        "sources": [{"name": "toy_text", "weight": 1.0}],
        "decontaminate": {"enabled": False},
    }
    s3_cfg["eval_quick"] = {"n_batches": 1, "depths": [1, 3]}
    s3_cfg["checkpoint"] = {"sync": {"enabled": False}}
    s3_dir = str(tmp_path / "s3_v2_long_1m_lsa")
    tr = LMTrainer(s3_cfg, s3_dir)
    assert tr.stack.hssd is None and tr.stack.gdn2 is None and tr.stack.lsa is not None
    out_s3 = tr.train()
    assert out_s3["step"] == 2

    # LMTrainer smoke with s2_v2_joint_bptt config (init_from s3_v2, 4x8 BPTT + sparse LSA, hssd=False)
    s2_cfg = _toy_v2_stack_cfg("s2_v2_joint_bptt", checkpoint=True)
    s2_cfg["train"].update(
        {
            "init_from": s3_dir,
            "steps": 1,
            "micro_batch": 1,
            "grad_accum": 1,
            "log_every": 1,
            "ckpt_every": 1,
            "eval_every": 0,
            "prefetch_batches": 0,
        }
    )
    s2_cfg["bptt"].update(
        {
            "enabled": True,
            "n_segments": 4,
            "seg_len": 8,
            "replay_k": 4,
            "offload_prior_hidden": False,
            "query_sim_chunk": 8,
        }
    )
    s2_cfg["data"] = {
        "seq_len": 32,
        "respect_doc_boundaries": True,
        "sources": [{"name": "toy_cot", "weight": 1.0}],
        "decontaminate": {"enabled": False},
    }
    s2_cfg["eval_quick"] = {"n_batches": 1, "depths": [1, 3]}
    s2_cfg["checkpoint"] = {"sync": {"enabled": False}}
    s2_dir = str(tmp_path / "s2_v2_joint_bptt")
    tr_s2 = LMTrainer(s2_cfg, s2_dir)
    assert tr_s2.stack.hssd is None and tr_s2.stack.gdn2 is not None and tr_s2.stack.lsa is not None
    out_s2 = tr_s2.train()
    assert out_s2["step"] == 1

    # GRPOTrainer smoke with s4_v2_rl config (init_from s2_v2, hssd=False, gdn2=True, lsa=sparse)
    s4_cfg = _toy_v2_stack_cfg("s4_v2_rl", checkpoint=False)
    s4_cfg["train"].update({"init_from": s2_dir, "steps": 1, "ckpt_every": 1})
    s4_cfg["rl"] = {
        "tasks": [{"source": "toy_arith", "n": 4, "verifier": "scorer:math", "weight": 1.0}],
        "group_size": 2,
        "prompts_per_step": 1,
        "max_new_tokens": 6,
        "micro_batch_seqs": 2,
        "verifier_workers": 1,
        "sirsi": copy.deepcopy(s4_cfg["rl"]["sirsi"]),
        "gate": {"every": 0},
    }
    s4_cfg["checkpoint"] = {"sync": {"enabled": False}}
    s4_dir = str(tmp_path / "s4_v2_rl")
    rl_tr = GRPOTrainer(s4_cfg, s4_dir)
    assert rl_tr.stack.hssd is None and rl_tr.stack.gdn2 is not None and rl_tr.stack.lsa is not None
    out_s4 = rl_tr.train()
    assert out_s4["step"] == 1


# ============================================================================ 5. Hub & CLI v2 GDN Stage Wiring


class _FakeHubAPI:
    def __init__(self) -> None:
        self.files: dict[str, bytes] = {}
        self.commits: list[str] = []

    def create_repo(self, **_: Any) -> None:
        return None

    def list_repo_files(self, **_: Any) -> list[str]:
        return sorted(self.files)

    def download_file(self, _repo: str, path_in_repo: str, dst: str) -> bool:
        if path_in_repo not in self.files:
            return False
        os.makedirs(os.path.dirname(os.path.abspath(dst)), exist_ok=True)
        with open(dst, "wb") as f:
            f.write(self.files[path_in_repo])
        return True

    def commit_bundle(self, _repo: str, additions: list[tuple[str, str | bytes]], msg: str) -> str:
        for rp, src in additions:
            if isinstance(src, bytes):
                self.files[rp] = src
            else:
                with open(src, "rb") as f:
                    self.files[rp] = f.read()
        self.commits.append(msg)
        return f"oid_{len(self.commits):06d}"


def test_hub_and_cli_v2_gdn_pipeline_wiring(tmp_path) -> None:
    assert V3_GDN_PIPELINE_ORDER == (
        "s0_probe",
        "s1_loop_a6b",
        "s3_v2_long_1m_lsa",
        "s2_v2_joint_bptt",
        "s4_v2_rl",
    )
    for st in V3_GDN_PIPELINE_ORDER:
        p = resolve_stage_config_path(st, str(ROOT / "configs"))
        assert os.path.isfile(p)

    # Even when archived HSSD stages exist in manifest, presence of any v2 GDN stage selects V3_GDN_PIPELINE_ORDER
    mixed_stages = {
        "s0_probe": {"status": "completed"},
        "s1_loop_a6b": {"status": "completed"},
        "s1b_hssd_distill": {"status": "completed"},
        "s3_v2_long_1m_lsa": {"status": "in_progress", "step": 100},
    }
    assert resolve_pipeline_order(mixed_stages) == V3_GDN_PIPELINE_ORDER
    assert resolve_pipeline_order({"s1b_hssd_distill": {"status": "completed"}}) == V3_PIPELINE_ORDER

    # Hub push_checkpoint clears stale hssd metadata and renders Native-GDN + D-LSA README
    api = _FakeHubAPI()
    hub = HubSyncManager("TassP0lar/test-v2-gdn", token="hf_test", api=api)
    prior_manifest = HubManifest(
        repo_id="TassP0lar/test-v2-gdn",
        architecture={
            "physical_layers": 40,
            "virtual_layers": 88,
            "linear_attention": "hssd",
            "hssd_rank": 64,
            "hssd_pages": 8,
        },
        stages={"s1b_hssd_distill": {"status": "completed", "step": 200}},
    )
    api.files["manifest.json"] = prior_manifest.to_json().encode("utf-8")

    ckpt_dir = tmp_path / "step_0000025"
    ckpt_dir.mkdir()
    torch.save({"loop_adapters.gates": torch.zeros(2, 8)}, ckpt_dir / "trainable.pt")
    (ckpt_dir / "state.json").write_text(
        json.dumps(
            {
                "step": 25,
                "stage": "s3_v2_long_1m_lsa",
                "stack": {
                    "loop": {"prelude_end": 12, "core_end": 36, "num_passes": 3},
                    "hssd": None,
                    "gdn2": None,
                    "lsa": {"mode": "hybrid", "index_dim": 64},
                },
            }
        ),
        encoding="utf-8",
    )
    oid = hub.push_checkpoint("s3_v2_long_1m_lsa", str(ckpt_dir), completed=False, target_steps=600)
    assert oid is not None
    updated = hub.fetch_manifest()
    assert updated.architecture["linear_attention"] == "stock"
    assert updated.architecture["sparse_attention"] == "d-lsa"
    assert "hssd_rank" not in updated.architecture
    assert "hssd_pages" not in updated.architecture
    readme = render_model_card(updated)
    assert "s3_v2_long_1m_lsa" in readme and "s2_v2_joint_bptt" in readme and "s4_v2_rl" in readme
    assert "Native Qwen3.6 Gated DeltaNet" in readme

    # CLI `_pipeline_stage_cmds`: `s4_v2_rl` NEVER gets `hssd.enabled=true` and `eval` resolves to `s4_v2_rl`
    all_v2_gdn = (*V3_GDN_PIPELINE_ORDER, "eval")
    cmds_s0 = _pipeline_stage_cmds(
        "s0_probe",
        config_root=str(ROOT / "configs"),
        run_root="/content/runs",
        repo_id="TassP0lar/Qwen3.6-35B-A3B-Loop64",
        debug_level=1,
        overrides=[],
        time_budget_hours=21.5,
        scan_batches=64,
        scan_seq_len=4096,
        all_stages=all_v2_gdn,
    )
    assert "--structured" in cmds_s0[0]

    cmds_s4_v2 = _pipeline_stage_cmds(
        "s4_v2_rl",
        config_root=str(ROOT / "configs"),
        run_root="/content/runs",
        repo_id="TassP0lar/Qwen3.6-35B-A3B-Loop64",
        debug_level=1,
        overrides=[],
        time_budget_hours=21.5,
        scan_batches=64,
        scan_seq_len=4096,
        all_stages=all_v2_gdn,
    )
    assert len(cmds_s4_v2) == 1
    flat_s4 = " ".join(cmds_s4_v2[0])
    assert "s4_v2_rl.yaml" in flat_s4
    assert "hssd.enabled=true" not in flat_s4

    cmds_eval = _pipeline_stage_cmds(
        "eval",
        config_root=str(ROOT / "configs"),
        run_root="/content/runs",
        repo_id="TassP0lar/Qwen3.6-35B-A3B-Loop64",
        debug_level=1,
        overrides=[],
        time_budget_hours=21.5,
        scan_batches=64,
        scan_seq_len=4096,
        all_stages=all_v2_gdn,
    )
    assert len(cmds_eval) == 1
    eval_cmd = cmds_eval[0]
    ckpt_idx = eval_cmd.index("--ckpt")
    assert eval_cmd[ckpt_idx + 1] == "/content/runs/s4_v2_rl"
    flat_eval = " ".join(eval_cmd)
    assert "hssd.enabled=false" in flat_eval
    assert "hssd.enabled=true" not in flat_eval
    assert "lsa.enabled=true" in flat_eval

    # Old HSSD v3.2 pipeline still injects hssd.enabled=true for `s4_rl` and resolves `eval` to `s4_rl`
    cmds_old_s4 = _pipeline_stage_cmds(
        "s4_rl",
        config_root=str(ROOT / "configs"),
        run_root="/content/runs",
        repo_id="TassP0lar/Qwen3.6-35B-A3B-Loop64",
        debug_level=1,
        overrides=[],
        time_budget_hours=21.5,
        scan_batches=64,
        scan_seq_len=4096,
        all_stages=(*V3_PIPELINE_ORDER, "eval"),
    )
    assert "hssd.enabled=true" in " ".join(cmds_old_s4[0])


# ============================================================================ 6. Colab Supervisor 503 Exp Backoff


def _cp(cmd: list[str], rc: int, stdout: str = "", stderr: str = "") -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(args=cmd, returncode=rc, stdout=stdout, stderr=stderr)


def test_colab_supervisor_v2_stages_and_503_exponential_backoff(tmp_path) -> None:
    # 1) CLI arg parsing for `--stages` with v2 GDN stages + `--max-respawn-backoff-sec`
    parser = build_arg_parser()
    v2_csv = ",".join(V3_GDN_PIPELINE_ORDER)
    args_default = parser.parse_args(["--stages", v2_csv])
    cfg_default = config_from_args(args_default)
    assert cfg_default.stages == V3_GDN_PIPELINE_ORDER
    assert cfg_default.max_respawn_backoff_sec == DEFAULT_MAX_RESPAWN_BACKOFF_SEC == 300.0
    script = build_remote_run_pipeline_script(cfg_default)
    assert f"--stages {v2_csv}" in script

    args_custom = parser.parse_args(["--stages", v2_csv, "--max-respawn-backoff-sec", "180"])
    cfg_custom = config_from_args(args_custom)
    assert cfg_custom.max_respawn_backoff_sec == 180.0

    # 2) Unit test `is_transient_capacity_error` and `compute_respawn_backoff_sec`
    assert is_transient_capacity_error("HTTP 503 Service Unavailable: backend overloaded")
    assert is_transient_capacity_error("Error: Cannot connect to GPU backend due to GPU capacity")
    assert not is_transient_capacity_error("SyntaxError: invalid syntax in script")
    # #397 Nit 2: Exception whose 503 marker appears before the truncated 600-char tail of `str(exc)`
    long_stderr_err = ColabCommandError(
        "colab new failed (rc=1): " + ("x" * 600),
        returncode=1,
        cmd=("colab", "new"),
        stdout="",
        stderr="HTTP 503 Service Unavailable: transient capacity\n" + ("x" * 1000),
    )
    assert "503" not in str(long_stderr_err)
    assert is_transient_capacity_error(long_stderr_err) is True
    long_stdout_err = ColabCommandError(
        "colab new failed (rc=1): " + ("y" * 600),
        returncode=1,
        cmd=("colab", "new"),
        stdout="No GPU capacity available right now\n" + ("y" * 1000),
        stderr="",
    )
    assert "capacity" not in str(long_stdout_err)
    assert is_transient_capacity_error(long_stdout_err) is True

    seq_300 = [
        compute_respawn_backoff_sec(k, base_backoff_sec=15.0, max_backoff_sec=300.0, transient_capacity=True)
        for k in (1, 2, 3, 4, 5)
    ]
    assert seq_300 == [60.0, 120.0, 240.0, 300.0, 300.0]

    seq_180 = [
        compute_respawn_backoff_sec(k, base_backoff_sec=15.0, max_backoff_sec=180.0, transient_capacity=True)
        for k in (1, 2, 3, 4)
    ]
    assert seq_180 == [60.0, 120.0, 180.0, 180.0]

    # 3) End-to-end preemption respawn encountering 4 consecutive `503 Service Unavailable` errors
    #    before succeeding on retry 5: verifies delays escalate `15 -> 60 -> 120 -> 240 -> 300`!
    cfg = SupervisorConfig(
        colab_bin="/fake/colab",
        session="loop64",
        gpu="G4",
        repo="TassP0lar/Qwen3.6-35B-A3B-Loop64",
        stages=("s3_v2_long_1m_lsa", "s2_v2_joint_bptt", "s4_v2_rl"),
        max_respawns=5,
        respawn_backoff_sec=15.0,
        max_respawn_backoff_sec=300.0,
        interval_sec=60.0,
        log_file=str(tmp_path / "sup_503.log"),
        tail_file=str(tmp_path / "sup_503_tail.log"),
    )
    sleeps: list[float] = []
    session_alive = True
    new_attempts = 0
    probe_calls = 0

    def manifest_fn(_repo: str) -> dict[str, Any]:
        if probe_calls >= 2:
            return {
                "stages": {
                    "s3_v2_long_1m_lsa": {"status": "completed", "step": 600},
                    "s2_v2_joint_bptt": {"status": "completed", "step": 600},
                    "s4_v2_rl": {"status": "completed", "step": 400},
                }
            }
        return {"stages": {"s3_v2_long_1m_lsa": {"status": "in_progress", "step": 50}}}

    def runner_503(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        nonlocal session_alive, new_attempts, probe_calls
        sub = cmd[1]
        if sub == "sessions":
            if session_alive:
                return _cp(cmd, 0, stdout="[loop64] gpu-g4 | Hardware: G4 | Shape: Standard | Variant: GPU\n")
            return _cp(cmd, 0, stdout="[colab] No active sessions found on server.\n")
        if sub == "new":
            new_attempts += 1
            if new_attempts <= 4:
                return _cp(cmd, 1, stderr="HTTP 503 Service Unavailable: No G4 GPU capacity right now")
            session_alive = True
            return _cp(cmd, 0, stdout="[colab] Session READY.\n")
        if sub == "url":
            return _cp(cmd, 0, stdout="https://colab.research.google.com/notebooks/empty.ipynb\n")
        if sub == "stop":
            session_alive = False
            return _cp(cmd, 0, stdout="[colab] Session terminated.\n")
        if sub == "exec":
            code = kwargs["input"]
            if BOOTSTRAP_SENTINEL in code:
                return _cp(cmd, 0, stdout=f'{BOOTSTRAP_SENTINEL}{{"ok": true, "launched": true, "pid": 4242}}\n')
            if TAIL_SENTINEL in code:
                return _cp(cmd, 0, stdout=f'{TAIL_SENTINEL}{{"tail": "done\\n"}}\n')
            probe_calls += 1
            if probe_calls == 1:
                session_alive = False
                return _cp(cmd, 1, stdout="[colab] Session 'loop64' not found.\n")
            return _cp(
                cmd,
                0,
                stdout=f'{PROBE_SENTINEL}{{"pid_alive": false, "exit_code": 0, "gpu_summary": "G4", '
                f'"last_log_lines": ["done"], "disk_free_gib": 150.0}}\n',
            )
        raise AssertionError(f"unexpected cmd: {cmd}")

    sup = ColabSupervisor(
        cfg,
        hf_token="hf_BackoffTestToken1234",
        runner=runner_503,
        manifest_fetcher=manifest_fn,
        sleep_fn=sleeps.append,
    )
    assert sup.run() == 0
    assert new_attempts == 5
    # Initial preemption wait (15s), followed by 4 consecutive 503 exponential backoffs (60 -> 120 -> 240 -> 300s)
    assert sleeps == [15.0, 60.0, 120.0, 240.0, 300.0]

    # 4) Initial `colab new` hitting `503 Service Unavailable` twice also backs off `60 -> 120` and recovers!
    cfg_init = SupervisorConfig(
        colab_bin="/fake/colab",
        session="loop64",
        gpu="G4",
        repo="TassP0lar/Qwen3.6-35B-A3B-Loop64",
        stages=("s3_v2_long_1m_lsa",),
        max_respawns=3,
        respawn_backoff_sec=15.0,
        max_respawn_backoff_sec=300.0,
        log_file=str(tmp_path / "sup_init_503.log"),
        tail_file=str(tmp_path / "sup_init_503_tail.log"),
    )
    init_sleeps: list[float] = []
    init_new_calls = 0
    init_alive = False

    def init_503_runner(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        nonlocal init_new_calls, init_alive
        sub = cmd[1]
        if sub == "sessions":
            if init_alive:
                return _cp(cmd, 0, stdout="[loop64] gpu-g4 | Hardware: G4 | Shape: Standard | Variant: GPU\n")
            return _cp(cmd, 0, stdout="[colab] No active sessions found on server.\n")
        if sub == "new":
            init_new_calls += 1
            if init_new_calls <= 2:
                return _cp(cmd, 1, stderr="503 Service Unavailable")
            init_alive = True
            return _cp(cmd, 0, stdout="[colab] Session READY.\n")
        if sub == "url":
            return _cp(cmd, 0, stdout="https://colab.research.google.com/notebooks/empty.ipynb\n")
        if sub == "stop":
            init_alive = False
            return _cp(cmd, 0, stdout="[colab] Session terminated.\n")
        if sub == "exec":
            code = kwargs["input"]
            if BOOTSTRAP_SENTINEL in code:
                return _cp(cmd, 0, stdout=f'{BOOTSTRAP_SENTINEL}{{"ok": true, "launched": true, "pid": 5151}}\n')
            if TAIL_SENTINEL in code:
                return _cp(cmd, 0, stdout=f'{TAIL_SENTINEL}{{"tail": "ok\\n"}}\n')
            return _cp(
                cmd,
                0,
                stdout=f'{PROBE_SENTINEL}{{"pid_alive": false, "exit_code": 0, "gpu_summary": "G4", '
                f'"last_log_lines": ["done"], "disk_free_gib": 150.0}}\n',
            )
        raise AssertionError(f"unexpected cmd: {cmd}")

    sup_init = ColabSupervisor(
        cfg_init,
        hf_token="hf_Init503Token1234",
        runner=init_503_runner,
        manifest_fetcher=lambda _: {"stages": {"s3_v2_long_1m_lsa": {"status": "completed", "step": 600}}},
        sleep_fn=init_sleeps.append,
    )
    assert sup_init.run() == 0
    assert init_new_calls == 3
    assert init_sleeps == [60.0, 120.0]
