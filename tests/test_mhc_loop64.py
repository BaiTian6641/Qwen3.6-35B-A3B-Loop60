"""Comprehensive unit and integration tests for Option B 64-layer `mHC` loop fixes (`s2_v3_loop64_mhc`)."""
from __future__ import annotations

import copy
import json
import math
import os
from pathlib import Path
from typing import Any

import pytest
import torch

from qloop.config import Cfg, load_config
from qloop.eval.sprint2_audit import (
    _load_arm_stack,
    _sync_cfg_to_checkpoint_payload,
    ablate_low_anchor_probe,
    compute_ablation_state_sha256,
)
from qloop.hub import HubSyncManager, load_qloop_from_hub
from qloop.model.gdn2 import GDN2Config
from qloop.model.hssd import HSSDConfig
from qloop.model.loop import LoopSpec, apply_loop, build_schedule
from qloop.model.lsa import LSAConfig
from qloop.model.mhc import V3_BRIDGE_WEIGHTS, LoopBridge, ReEntryBridge, _rms
from qloop.model.toy import build_toy
from qloop.train.checkpoint import CheckpointManager
from qloop.train.optim import FusedStepOptimizer
from qloop.train.stack import build_stack
from qloop.train.trainer import LMTrainer

ROOT = Path(__file__).resolve().parent.parent
QWEN_TYPES = ["full_attention" if (i + 1) % 4 == 0 else "linear_attention" for i in range(40)]
TOY_TYPES = ["full_attention" if (i + 1) % 4 == 0 else "linear_attention" for i in range(16)]
TOY_SPEC_2 = {"prelude_end": 4, "core_end": 12, "num_passes": 2, "target_virtual_layers": 24}


def _ids(b: int = 2, t: int = 12, seed: int = 7, vocab: int = 97) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, vocab, (b, t), generator=g)


# ==============================================================================
# 1. Backward Compatibility & Legacy Bit-Identity
# ==============================================================================


@pytest.mark.parametrize("adapter_mode", ["concat", "subspace"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_loop_bridge_defaults_are_bit_identical_to_legacy(adapter_mode: str, dtype: torch.dtype) -> None:
    torch.manual_seed(42)
    br = LoopBridge(
        hidden=16,
        n_extra_passes=2,
        init_weights=V3_BRIDGE_WEIGHTS,
        rank=4,
        step_embedding=True,
        adapter_mode=adapter_mode,
        up_init_std=0.05,
    ).to(dtype=dtype)
    assert br.drop_raw_h_prev is False
    assert br.rms_preserve is False
    assert br._anchor_cap is None
    assert br.anchor_cap_excess_loss(0) is None

    with torch.no_grad():
        br.step_emb.normal_(0.0, 0.5)
        if adapter_mode == "concat":
            br.up.weight.normal_(0.0, 0.1)

    h = torch.randn(2, 5, 16, dtype=dtype) * 5.0
    anchor = torch.randn(2, 5, 16, dtype=dtype)
    out = br(h, anchor, 1)

    # Exact legacy formula
    w = torch.softmax(br.mix_logits[1].float() if dtype == torch.float32 else br.mix_logits[1], dim=-1)
    assert torch.equal(br.raw_weights(1), w)
    assert torch.equal(br.weights(1), w)
    rms_anc = _rms(anchor)
    matched = h * (rms_anc / _rms(h))
    expected = w[0] * h + w[1] * anchor + w[2] * matched + br.step_emb[1]
    if adapter_mode == "concat":
        delta = br.up(torch.nn.functional.silu(br.down(torch.cat([h, anchor], dim=-1))))
        expected = expected + delta
    else:
        u_hat = br.norms[1](h)
        delta = br.ups[1](torch.nn.functional.silu(br.downs[1](u_hat)))
        expected = expected + (rms_anc / math.sqrt(16)) * delta
    assert torch.equal(out, expected.to(dtype))


def test_legacy_loop_controller_defaults_unchanged() -> None:
    model = build_toy()
    ids = _ids()
    ref = model(ids).logits
    ctl = apply_loop(model, TOY_SPEC_2)
    assert ctl.adapters.bridge._anchor_cap is None
    assert not hasattr(ctl.adapters.gates, "_qloop_project")
    assert ctl.set_step(10, 100) is None
    assert ctl.aux_loss() is None
    assert torch.equal(model(ids).logits, ref)
    st = ctl.stats()
    assert "loop/bridge_anchor_cap" not in st
    assert "loop/gate_mean_p2" in st
    assert "loop/gate_min_p2" in st
    assert "loop/gate_max_p2" in st

    # Without gate_clamp, negative/above-1 gates remain unconstrained (legacy behavior)
    with torch.no_grad():
        ctl.adapters.gates.fill_(-0.5)
    out_neg = model(ids).logits
    assert not torch.equal(out_neg, ref)


# ==============================================================================
# 2. drop_raw_h_prev, Anchor Cap, Excess Routing, and anchor_cap_excess_loss
# ==============================================================================


def test_drop_raw_h_prev_zeros_stream0_and_eliminates_norm_shock() -> None:
    br = ReEntryBridge(
        hidden=16,
        n_extra_passes=1,
        init_weights=(0.30, 0.50, 0.20),
        rank=0,
        step_embedding=False,
        drop_raw_h_prev=True,
        max_anchor_weight=0.25,
    )
    raw = br.raw_weights(0).detach()
    eff = br.weights(0).detach()
    assert float(raw[0]) == 0.0
    assert float(eff[0]) == 0.0
    assert float(raw[1] + raw[2]) == pytest.approx(1.0, abs=1e-6)
    assert float(eff[1]) == pytest.approx(0.25, abs=1e-6)
    assert float(eff[2]) == pytest.approx(0.75, abs=1e-6)
    assert float(eff.sum()) == pytest.approx(1.0, abs=1e-6)

    # Output is invariant to positive scalar rescaling of h (no raw h_prev norm shock!)
    h = torch.randn(2, 4, 16)
    anchor = torch.randn(2, 4, 16)
    out_1 = br(h, anchor, 0)
    out_100 = br(h * 100.0, anchor, 0)
    torch.testing.assert_close(out_1, out_100, rtol=1e-4, atol=1e-5)

    # Gradients through raw_weights / excess loss are finite and zero on dropped logit 0
    loss = br.anchor_cap_excess_loss(0)
    assert loss is not None and float(loss.detach()) > 0.0
    loss.backward()
    g = br.mix_logits.grad
    assert g is not None and torch.isfinite(g).all()
    assert float(g[0, 0]) == 0.0
    assert float(g[0, 1]) > 0.0
    assert float(g[0, 2]) < 0.0


def test_anchor_cap_routes_excess_strictly_to_stream2_and_trains_mix_logits() -> None:
    br = LoopBridge(
        hidden=12,
        n_extra_passes=1,
        init_weights=(0.10, 0.80, 0.10),
        rank=0,
        step_embedding=False,
        drop_raw_h_prev=False,
        max_anchor_weight=0.30,
    )
    raw = br.raw_weights(0).detach()
    eff = br.weights(0).detach()
    # Excess (0.80 - 0.30 = 0.50) goes strictly onto Stream 2, NEVER Stream 0
    assert float(eff[0]) == pytest.approx(float(raw[0]), abs=1e-6)
    assert float(eff[1]) == pytest.approx(0.30, abs=1e-6)
    assert float(eff[2]) == pytest.approx(float(raw[2]) + (float(raw[1]) - 0.30), abs=1e-6)
    assert float(eff.sum()) == pytest.approx(1.0, abs=1e-6)

    # Validation on set_anchor_cap
    for bad in (0.0, -0.1, 1.01, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            br.set_anchor_cap(bad)
    br.set_anchor_cap(None)
    assert br.anchor_cap_excess_loss(0) is None
    assert torch.equal(br.weights(0), br.raw_weights(0))

    # Gradient descent on anchor_cap_excess_loss pulls raw_w1 smoothly below cap
    br.set_anchor_cap(0.20)
    opt = torch.optim.SGD([br.mix_logits], lr=2.0)
    w1_start = float(br.raw_weights(0)[1].detach())
    for _ in range(40):
        opt.zero_grad()
        loss = br.anchor_cap_excess_loss(0)
        assert loss is not None
        loss.backward()
        opt.step()
    w1_end = float(br.raw_weights(0)[1].detach())
    assert w1_end < w1_start
    assert w1_end < 0.25


# ==============================================================================
# 3. rms_preserve Non-Expansive Ball Projection
# ==============================================================================


def test_rms_preserve_enforces_strict_non_expansive_ball_and_scales_step_emb() -> None:
    torch.manual_seed(11)
    br = ReEntryBridge(
        hidden=32,
        n_extra_passes=1,
        init_weights=V3_BRIDGE_WEIGHTS,
        rank=8,
        step_embedding=True,
        up_init_std=1.0,
        drop_raw_h_prev=True,
        max_anchor_weight=0.20,
        rms_preserve=True,
    )
    with torch.no_grad():
        br.step_emb.fill_(10.0)
        br.ups[0].weight.normal_(0.0, 2.0)

    h = torch.randn(3, 7, 32) * 50.0
    anchor = torch.randn(3, 7, 32) * 0.5
    out = br(h, anchor, 0)
    true_rms_out = out.float().pow(2).mean(-1, keepdim=True).sqrt()
    rms_anc = _rms(anchor)
    assert bool((true_rms_out <= rms_anc + 1e-6).all()), (
        f"rms_preserve violated: max ratio={float((true_rms_out / rms_anc).max()):.6f}"
    )

    # Gradients flow through rms_preserve cleanly
    out.pow(2).mean().backward()
    assert br.step_emb.grad is not None and torch.isfinite(br.step_emb.grad).all()
    assert br.ups[0].weight.grad is not None and torch.isfinite(br.ups[0].weight.grad).all()
    assert br.downs[0].weight.grad is not None and torch.isfinite(br.downs[0].weight.grad).all()

    # Non-expansive (ball, NOT sphere): inputs whose combined norm is BELOW rms(anchor) are NOT amplified
    br_small = LoopBridge(
        hidden=16,
        n_extra_passes=1,
        init_weights=(0.05, 0.50, 0.45),
        rank=0,
        step_embedding=False,
        drop_raw_h_prev=True,
        rms_preserve=True,
    )
    anc = torch.ones(1, 2, 16) * 2.0
    h_opp = -anc  # opposing direction -> convex combo has norm << rms(anc)
    out_opp = br_small(h_opp, anc, 0)
    rms_out_opp = _rms(out_opp)
    assert bool((rms_out_opp < _rms(anc) * 0.5).all())


# ==============================================================================
# 4. LoopSpec Validation, Cosine Schedule, aux_loss, and gate_clamp
# ==============================================================================


def test_loop_spec_validation_and_cosine_anchor_schedule() -> None:
    spec = LoopSpec.from_dict({
        **TOY_SPEC_2,
        "bridge_drop_raw_h_prev": True,
        "bridge_rms_preserve": True,
        "bridge_anchor_max_start": 0.85,
        "bridge_anchor_max_end": 0.20,
        "bridge_anchor_anneal_frac": 0.60,
        "bridge_anchor_aux_weight": 0.10,
        "gate_clamp": [0.0, 1.0],
    })
    assert spec.gate_clamp == (0.0, 1.0)
    assert spec.validate(TOY_TYPES) == []

    # Invalid configs rejected
    with pytest.raises(ValueError, match="gate_clamp"):
        LoopSpec.from_dict({**TOY_SPEC_2, "gate_clamp": [0.0]})
    with pytest.raises(ValueError, match="gate_clamp"):
        LoopSpec.from_dict({**TOY_SPEC_2, "gate_clamp": [0.8, 0.2]}).validate(TOY_TYPES)
    with pytest.raises(ValueError, match="bridge_anchor_anneal_frac"):
        LoopSpec.from_dict({**TOY_SPEC_2, "bridge_anchor_anneal_frac": 0.0}).validate(TOY_TYPES)
    with pytest.raises(ValueError, match="bridge_anchor_max_end"):
        LoopSpec.from_dict({**TOY_SPEC_2, "bridge_anchor_max_end": 1.2}).validate(TOY_TYPES)
    with pytest.raises(ValueError, match="bridge_anchor_aux_weight"):
        LoopSpec.from_dict({**TOY_SPEC_2, "bridge_anchor_aux_weight": -0.1}).validate(TOY_TYPES)

    model = build_toy()
    ctl = apply_loop(model, spec)
    # Inference/eval default initializes anchor cap to `bridge_anchor_max_end` (0.20)
    assert ctl.adapters.bridge._anchor_cap == pytest.approx(0.20)
    assert getattr(ctl.adapters.gates, "_qloop_project", None) == (0.0, 1.0)

    # Cosine schedule verification across steps 0..500 with anneal_frac=0.60 (horizon=300)
    assert ctl.set_step(0, 500) == pytest.approx(0.85)
    assert ctl.adapters.bridge._anchor_cap == pytest.approx(0.85)
    assert ctl.set_step(150, 500) == pytest.approx(0.525)
    assert ctl.set_step(300, 500) == pytest.approx(0.20)
    assert ctl.set_step(500, 500) == pytest.approx(0.20)

    # aux_loss includes bridge_anchor_aux_weight when active_passes >= 2, skipped when active_passes == 1
    ctl.set_active_passes(2)
    aux_d2 = ctl.aux_loss()
    assert aux_d2 is not None and float(aux_d2.detach()) > 0.0
    aux_d2.backward()
    assert ctl.adapters.bridge.mix_logits.grad is not None
    assert float(ctl.adapters.bridge.mix_logits.grad[0, 1]) > 0.0

    ctl.set_active_passes(1)
    assert ctl.aux_loss() is None

    # Stats expose anchor cap and gate distribution
    st = ctl.stats()
    assert st["loop/bridge_anchor_cap"] == pytest.approx(0.20)
    assert st["loop/gate_min_p2"] == pytest.approx(0.0)
    assert st["loop/gate_max_p2"] == pytest.approx(0.0)
    assert st["loop/gate_mean_p2"] == pytest.approx(0.0)


@pytest.mark.parametrize("fused", [False, True])
def test_gate_clamp_forward_guard_step0_grad_and_optimizer_projection(fused: bool) -> None:
    model = build_toy()
    ids = _ids()
    ref = model(ids).logits.detach()
    spec = LoopSpec.from_dict({
        **TOY_SPEC_2,
        "gate_init": -0.3,  # out of [0, 1], must be clamped to 0.0 at init
        "gate_clamp": [0.0, 1.0],
        "bridge_drop_raw_h_prev": True,
        "bridge_rms_preserve": True,
    })
    ctl = apply_loop(model, spec)
    assert float(ctl.adapters.gates.detach().min()) == 0.0 and float(ctl.adapters.gates.detach().max()) == 0.0
    assert getattr(ctl.adapters.gates, "_qloop_project", None) == (0.0, 1.0)
    assert torch.equal(model(ids).logits, ref)

    # Step-0 gradient at gate == 0.0 is non-zero despite clamp(0.0, 1.0)
    model.train()
    loss = model(ids, labels=ids).loss
    loss.backward()
    assert ctl.adapters.gates.grad is not None
    assert float(ctl.adapters.gates.grad.abs().sum()) > 0.0

    # Forward _combine() clamps even BEFORE optimizer projection if gates is perturbed out-of-range
    with torch.no_grad():
        ctl.adapters.gates.fill_(-0.75)
    assert torch.equal(model(ids).logits, ref), "negative gate must be clamped to 0.0 in _combine()"

    with torch.no_grad():
        ctl.adapters.gates.fill_(2.5)
    out_gt1 = model(ids).logits.detach()
    with torch.no_grad():
        ctl.adapters.gates.fill_(1.0)
    out_eq1 = model(ids).logits.detach()
    assert torch.equal(out_gt1, out_eq1), "gate > 1.0 must be clamped to 1.0 in _combine()"

    # Optimizer projects gates back into [0.0, 1.0] after step()
    opt = FusedStepOptimizer.from_config(
        [("loop_adapters.gates", ctl.adapters.gates)],
        {"lr": 5.0, "warmup_steps": 0, "fused": fused, "impl": "builtin", "total_steps": 5},
    )
    opt.zero_grad(set_to_none=True)
    opt.set_boundary(True)
    with torch.no_grad():
        ctl.adapters.gates.fill_(0.5)
    loss2 = model(ids, labels=ids).loss
    loss2.backward()
    opt.step()
    assert float(ctl.adapters.gates.detach().min()) >= 0.0
    assert float(ctl.adapters.gates.detach().max()) <= 1.0


# ==============================================================================
# 5. End-to-End num_passes=3 -> 2 Warm-Start, Passthrough Hygiene & Trainer Sync
# ==============================================================================


def _toy_s2v2_cfg(steps: int = 2) -> dict[str, Any]:
    return {
        "seed": 3,
        "stage": "s2_v2_joint_bptt",
        "model": {"name": "toy", "dtype": "float32", "device_map": "cpu"},
        "loop": {
            "enabled": True,
            "prelude_end": 4,
            "core_end": 12,
            "num_passes": 3,
            "target_virtual_layers": 32,
            "kv_share": True,
            "bridge_init": [0.05, 0.85, 0.10],
            "bridge_rank": 8,
            "bridge_mode": "subspace",
            "bridge_up_std": 0.01,
            "attn_adapter_rank": 4,
            "step_embedding": True,
            "gate_init": 0.0,
            "per_pass_router": True,
            "router_adapter_rank": 4,
            "mor": False,
            "checkpoint": False,
            "train_pass_probs": [0.1, 0.4, 0.5],
        },
        "gdn2": {
            "enabled": True,
            "rank": 4,
            "chunk_size": 16,
            "segment_chunks": 2,
            "qed_enabled": True,
            "qed_max_lambda": 0.35,
            "neg_eigval_max_boost": 0.5,
            "backend": "pytorch",
        },
        "hssd": {"enabled": False},
        "lsa": {
            "enabled": True,
            "mode": "sparse",
            "num_query_passes": 3,
            "index_dim": 8,
            "index_on": "nope",
            "rotary_dim": 4,
            "pool": 2,
            "block_size": 16,
            "avg_bucket_keys": 4,
            "max_bucket_keys": 8,
            "num_select": 2,
            "window": 16,
            "sinks": 2,
            "query_chunk": 16,
        },
        "experts": {"enabled": False},
        "train": {
            "steps": steps,
            "micro_batch": 1,
            "grad_accum": 1,
            "log_every": 1,
            "ckpt_every": steps,
            "eval_every": 0,
            "keep_last": 1,
            "sha256_frozen_experts": True,
            "train_loop_adapters": True,
            "train_gdn2": True,
            "train_lsa": True,
            "trainable": [
                r"self_attn\.(k_proj|v_proj|o_proj|q_norm|k_norm)",
                r"input_layernorm|post_attention_layernorm",
            ],
            "allowed_groups": ["loop_adapters", "gdn2_adapters", "lsa_indexers", "attention", "norms"],
            "aux_weights": {"loop_aux": 1.0, "lsa_kl": 0.2},
            "optim": {
                "lr": 1e-3,
                "warmup_steps": 1,
                "fused": False,
                "max_grad_norm": 1.0,
                "impl": "builtin",
            },
        },
        "data": {"seq_len": 32, "sources": [{"name": "toy_cot", "weight": 1.0}]},
        "eval_quick": {"n_batches": 1, "depths": [1, 2, 3]},
    }


def _toy_s2v3_cfg(init_from: str | None = None, steps: int = 4) -> dict[str, Any]:
    cfg = copy.deepcopy(_toy_s2v2_cfg(steps=steps))
    cfg["stage"] = "s2_v3_loop64_mhc"
    cfg["loop"].update({
        "num_passes": 2,
        "target_virtual_layers": 24,
        "bridge_drop_raw_h_prev": True,
        "bridge_rms_preserve": True,
        "bridge_anchor_max_start": 0.85,
        "bridge_anchor_max_end": 0.20,
        "bridge_anchor_anneal_frac": 0.60,
        "bridge_anchor_aux_weight": 0.10,
        "gate_clamp": [0.0, 1.0],
        "train_pass_probs": [0.0, 1.0],
    })
    cfg["gdn2"].update({"qed_enabled": False, "neg_eigval_max_boost": 0.0})
    cfg["lsa"].update({"num_query_passes": 2})
    cfg["train"]["init_from"] = init_from
    cfg["train"]["eval_every"] = 2
    cfg["eval_quick"] = {"n_batches": 1, "depths": [1, 2]}
    return cfg


def test_warm_start_3_to_2_passes_slicing_pruning_clamping_and_trainer_e2e(tmp_path: Path) -> None:
    # 1. Build synthetic s2_v2 (3-pass, QED+neg_eigval, 3 query passes) and populate distinct pass-2 vs pass-3 values
    st_v2 = build_stack(_toy_s2v2_cfg())
    assert st_v2.loop is not None and st_v2.gdn2 is not None and st_v2.lsa is not None
    with torch.no_grad():
        # Include negative and >1.0 channels on pass-2 gates to test warm-start clamping into [0, 1]
        st_v2.loop.adapters.gates[0].fill_(0.25)
        st_v2.loop.adapters.gates[0, 0] = -0.15
        st_v2.loop.adapters.gates[0, 1] = 1.40
        st_v2.loop.adapters.gates[1].fill_(0.99)
        st_v2.loop.adapters.bridge.step_emb[0].fill_(0.123)
        st_v2.loop.adapters.bridge.step_emb[1].fill_(9.999)
        st_v2.loop.adapters.bridge.downs[0].weight.fill_(0.07)
        st_v2.loop.adapters.bridge.downs[1].weight.fill_(0.88)
        st_v2.loop.adapters.pass_router.bias[0].fill_(0.05)
        st_v2.loop.adapters.pass_router.bias[1].fill_(0.77)
        for pdict in st_v2.loop.adapters.pass_attn.layers.values():
            for proj in pdict.values():
                proj.up[0].fill_(0.03)
                proj.up[1].fill_(0.66)

    saved_params = st_v2.checkpoint_params()
    # Add an inactive-subsystem passthrough tensor (hssd is disabled in both stages) to verify it IS preserved
    inactive_key = "model.layers.0.linear_attn.hssd_adapter.carried_dummy"
    saved_params[inactive_key] = torch.ones(4)
    assert any(k.endswith("gdn2_adapter.qed_lambda_raw") for k in saved_params)
    assert any(k.endswith("gdn2_adapter.neg_gamma_raw") for k in saved_params)
    assert any(k.endswith("lsa_indexer.q_proj_extra.1.weight") for k in saved_params)
    assert "loop_adapters.bridge.downs.1.weight" in saved_params

    v2_dir = tmp_path / "s2_v2_joint_bptt"
    CheckpointManager(str(v2_dir), keep_last=1).save(
        step=600,
        params=saved_params,
        optim_state=None,
        state={"stage": "s2_v2_joint_bptt", "stack": st_v2.describe()},
    )

    # 2. Build s2_v3 (2-pass, QED/neg_eigval pruned, 2 query passes) and warm-start
    st_v3 = build_stack(_toy_s2v3_cfg())
    assert st_v3.loop is not None
    sha_before = st_v3.frozen_experts_sha256
    assert sha_before is not None

    # strict=True must still reject shape mismatch / missing keys
    with pytest.raises(RuntimeError):
        st_v3.restore_checkpoint(saved_params, strict=True, prior_stack_meta=st_v2.describe())

    # Genuine inner-dimension shape mismatch must still raise even with strict=False
    bad_params = dict(saved_params)
    bad_params["loop_adapters.gates"] = torch.zeros(1, st_v3.loop.adapters.gates.shape[1] + 1)
    with pytest.raises(RuntimeError, match="shape mismatch"):
        st_v3.restore_checkpoint(bad_params, strict=False, prior_stack_meta=st_v2.describe())

    # Clean strict=False warm-start
    st_v3 = build_stack(_toy_s2v3_cfg())
    counts = st_v3.restore_checkpoint(saved_params, strict=False, prior_stack_meta=st_v2.describe())
    assert counts["trainable"] > 0
    assert st_v3.verify_frozen_experts() == sha_before

    # Pass-2 weights sliced and restored accurately; gates clamped into [0, 1]
    assert float(st_v3.loop.adapters.gates.detach()[0, 0]) == 0.0
    assert float(st_v3.loop.adapters.gates.detach()[0, 1]) == 1.0
    assert float(st_v3.loop.adapters.gates.detach()[0, 2]) == pytest.approx(0.25)
    assert float(st_v3.loop.adapters.bridge.step_emb.detach()[0, 0]) == pytest.approx(0.123)
    assert float(st_v3.loop.adapters.bridge.downs[0].weight.detach()[0, 0]) == pytest.approx(0.07)
    assert float(st_v3.loop.adapters.pass_router.bias.detach()[0, 0, 0]) == pytest.approx(0.05)

    # Passthrough hygiene: pruned active-subsystem keys are dropped; inactive hssd key is preserved
    pt = st_v3.passthrough_tensors
    assert inactive_key in pt
    assert not any(k.startswith("loop_adapters.") for k in pt)
    assert not any("gdn2_adapter." in k for k in pt)
    assert not any("lsa_indexer." in k for k in pt)

    # 3. Full LMTrainer E2E warm-start + step-0 anchor cap sync + cosine annealing + checkpoint hygiene
    v3_dir = tmp_path / "s2_v3_loop64_mhc"
    trainer = LMTrainer(Cfg(_toy_s2v3_cfg(init_from=str(v2_dir), steps=4)), str(v3_dir))
    observed_step0_cap: list[float | None] = []
    orig_quick_eval = trainer.quick_eval

    def _spy_quick_eval() -> dict[str, float]:
        if trainer.step == 0 and trainer.stack.loop is not None:
            observed_step0_cap.append(trainer.stack.loop.adapters.bridge._anchor_cap)
        return orig_quick_eval()

    trainer.quick_eval = _spy_quick_eval  # type: ignore[method-assign]
    res = trainer.train()
    assert res["step"] == 4
    assert res["resume_mode"] == "init_from"
    # Before step-0 eval, trainer.train() synced anchor cap to start=0.85 (not eval default 0.20!)
    assert observed_step0_cap == [pytest.approx(0.85)]
    # By step=4 (past 0.6 * 4 = 2.4), anchor cap annealed cleanly to 0.20
    assert trainer.stack.loop is not None
    assert trainer.stack.loop.adapters.bridge._anchor_cap == pytest.approx(0.20)
    assert float(trainer.stack.loop.adapters.gates.detach().min()) >= 0.0
    assert float(trainer.stack.loop.adapters.gates.detach().max()) <= 1.0
    assert res["last"]["loop/bridge_anchor_cap"] == pytest.approx(0.20)
    assert "loss/loop_aux" in res["last"]

    # Saved s2_v3 checkpoint contains zero pruned QED/neg_eigval/pass-3 tensors
    saved_v3 = CheckpointManager(str(v3_dir), keep_last=1).load()
    assert saved_v3 is not None
    v3_keys = set(saved_v3["params"].keys())
    assert not any(k.endswith("qed_lambda_raw") for k in v3_keys)
    assert not any(k.endswith("neg_gamma_raw") for k in v3_keys)
    assert not any("q_proj_extra.1" in k for k in v3_keys)
    assert "loop_adapters.bridge.downs.1.weight" not in v3_keys


# ==============================================================================
# 6. Production Stage Config Verification (configs/stages/s2_v3_loop64_mhc.yaml)
# ==============================================================================


def test_s2_v3_loop64_mhc_config_loads_validates_and_enforces_policies() -> None:
    cfg_path = ROOT / "configs" / "stages" / "s2_v3_loop64_mhc.yaml"
    assert cfg_path.is_file()
    raw_text = cfg_path.read_text(encoding="utf-8")
    assert "hbm" not in raw_text.lower(), "s2_v3_loop64_mhc.yaml must not mention HBM (use 96 GB GDDR7 ECC VRAM)"

    cfg = load_config(str(cfg_path))
    assert cfg.stage == "s2_v3_loop64_mhc"
    assert cfg.stage_type == "lm"
    assert cfg.model.name == "unsloth/Qwen3.6-35B-A3B-NVFP4"
    assert cfg.model.yarn_target_len == 1048576

    loop_spec = LoopSpec.from_dict(dict(cfg.loop))
    assert loop_spec.prelude_end == 12
    assert loop_spec.core_end == 36
    assert loop_spec.num_passes == 2
    assert loop_spec.target_virtual_layers == 64
    assert loop_spec.virtual_layers(40) == 64
    assert loop_spec.validate(QWEN_TYPES) == []
    assert len(build_schedule(loop_spec, QWEN_TYPES)) == 64
    assert loop_spec.bridge_drop_raw_h_prev is True
    assert loop_spec.bridge_rms_preserve is True
    assert loop_spec.bridge_anchor_max_start == pytest.approx(0.85)
    assert loop_spec.bridge_anchor_max_end == pytest.approx(0.20)
    assert loop_spec.bridge_anchor_anneal_frac == pytest.approx(0.60)
    assert loop_spec.bridge_anchor_aux_weight == pytest.approx(0.10)
    assert loop_spec.gate_clamp == (0.0, 1.0)
    assert loop_spec.train_pass_probs == [0.15, 0.85]

    gdn2_cfg = GDN2Config.from_dict(dict(cfg.gdn2))
    gdn2_cfg.validate()
    assert cfg.gdn2.enabled is True
    assert gdn2_cfg.qed_enabled is False
    assert gdn2_cfg.neg_eigval_max_boost == 0.0

    hssd_cfg = HSSDConfig.from_dict(dict(cfg.hssd))
    hssd_cfg.validate()
    assert cfg.hssd.enabled is False

    lsa_cfg = LSAConfig.from_dict(dict(cfg.lsa))
    lsa_cfg.validate()
    assert cfg.lsa.enabled is True
    assert lsa_cfg.mode == "sparse"
    assert lsa_cfg.num_query_passes == 2

    assert cfg.bptt.enabled is False
    assert cfg.experts.enabled is False
    assert cfg.train.init_from == "runs/s2_v2_joint_bptt"
    assert cfg.train.steps == 500
    assert cfg.train.grad_accum == 8
    assert cfg.train.log_every == 5
    assert cfg.train.ckpt_every == 25
    assert cfg.train.sha256_frozen_experts is True
    assert dict(cfg.train.optim.lr_mult) == {
        "loop_adapters": 20.0,
        "gdn2_adapter": 3.0,
        "lsa_indexer": 3.0,
    }
    assert dict(cfg.train.aux_weights) == {"loop_aux": 1.0, "lsa_kl": 0.2}

    # Strict permissive clean-room dataset policy: ZERO opus46 subsets
    sources = list(cfg.data.sources)
    names = [s["name"] for s in sources]
    assert not any("opus46" in nm.lower() for nm in names)
    assert names == [
        "openthoughts3",
        "nemotron_post",
        "finepdfs_mix",
        "fineweb_edu",
        "loop64_fable_sirsi",
        "arc_agi2_train",
    ]
    assert sum(float(s["weight"]) for s in sources) == pytest.approx(1.0)
    assert cfg.data.seq_len == 16384
    assert cfg.data.decontaminate.enabled is True
    assert list(cfg.eval_quick.depths) == [1, 2]


# ==============================================================================
# 7. sprint2_audit Config Sync, Ablation SHA-256, and Hub Loader Round-Trip
# ==============================================================================


class _MockHfApi:
    """Minimal in-memory HF Hub API stub matching `HubSyncManager`'s injected `api` contract."""

    def __init__(self) -> None:
        self.files: dict[tuple[str, str], bytes] = {}
        self._commits = 0

    def create_repo(self, repo_id: str, **kwargs: Any) -> None:
        return None

    def list_repo_files(self, repo_id: str, repo_type: str = "model") -> list[str]:
        return sorted(path for (rid, path) in self.files if rid == repo_id)

    def commit_bundle(self, repo_id: str, additions: list[tuple[str, str | bytes]], commit_message: str) -> str:
        for repo_path, src in additions:
            if isinstance(src, (bytes, bytearray)):
                self.files[(repo_id, repo_path)] = bytes(src)
            else:
                self.files[(repo_id, repo_path)] = Path(src).read_bytes()
        self._commits += 1
        return f"mockoid{self._commits:04d}"

    def download_file(self, repo_id: str, path_in_repo: str, dst_path: str) -> bool:
        key = (repo_id, path_in_repo)
        if key not in self.files:
            return False
        os.makedirs(os.path.dirname(os.path.abspath(dst_path)), exist_ok=True)
        Path(dst_path).write_bytes(self.files[key])
        return True


def test_sprint2_audit_sync_ablation_and_hub_round_trip(tmp_path: Path) -> None:
    # Build & train a short s2_v3 stack and save checkpoint
    v3_cfg = _toy_s2v3_cfg(steps=2)
    st_v3 = build_stack(v3_cfg)
    assert st_v3.loop is not None
    with torch.no_grad():
        st_v3.loop.adapters.gates.fill_(0.35)
        st_v3.loop.adapters.bridge.step_emb.fill_(0.2)
    st_v3.model.eval()
    ids = _ids()
    with torch.no_grad():
        expected_v3 = st_v3.model(input_ids=ids).logits.clone()
    assert st_v3.loop.adapters.bridge._anchor_cap == pytest.approx(0.20)

    v3_run = tmp_path / "runs" / "s2_v3_loop64_mhc"
    ckpt_dir = CheckpointManager(str(v3_run), keep_last=1).save(
        step=500,
        params=st_v3.checkpoint_params(),
        optim_state=None,
        state={"stage": "s2_v3_loop64_mhc", "stack": st_v3.describe()},
        completed=True,
        config_dict=v3_cfg,
    )
    loaded_state = json.loads((Path(ckpt_dir) / "state.json").read_text(encoding="utf-8"))
    # JSON round-trip converts gate_clamp tuple -> list; verify sync + LoopSpec.from_dict handles it
    assert loaded_state["stack"]["loop"]["gate_clamp"] == [0.0, 1.0]

    # (a) Syncing an s2_v2-shaped cfg_dict to the s2_v3 checkpoint payload
    s2v2_cfg_dict = copy.deepcopy(_toy_s2v2_cfg())
    _sync_cfg_to_checkpoint_payload(s2v2_cfg_dict, st_v3.checkpoint_params(), loaded_state["stack"], {})
    assert s2v2_cfg_dict["loop"]["num_passes"] == 2
    assert s2v2_cfg_dict["loop"]["bridge_drop_raw_h_prev"] is True
    assert s2v2_cfg_dict["loop"]["bridge_rms_preserve"] is True
    assert s2v2_cfg_dict["loop"]["bridge_anchor_max_end"] == pytest.approx(0.20)
    assert s2v2_cfg_dict["loop"]["gate_clamp"] == [0.0, 1.0]
    assert s2v2_cfg_dict["gdn2"]["qed_enabled"] is False
    assert s2v2_cfg_dict["gdn2"]["neg_eigval_max_boost"] == 0.0
    assert s2v2_cfg_dict["lsa"]["num_query_passes"] == 2

    # End-to-end _load_arm_stack using a legacy-shaped YAML config file pointing at the s2_v3 checkpoint
    legacy_yaml = tmp_path / "legacy_s2v2.yaml"
    import yaml

    legacy_yaml.write_text(yaml.safe_dump(_toy_s2v2_cfg()), encoding="utf-8")
    reloaded_st, _lbl = _load_arm_stack({"name": "v3", "config": str(legacy_yaml), "checkpoint": str(v3_run)})
    assert reloaded_st.loop is not None
    assert reloaded_st.loop.spec.num_passes == 2
    assert reloaded_st.loop.spec.gate_clamp == (0.0, 1.0)
    assert reloaded_st.loop.adapters.bridge._anchor_cap == pytest.approx(0.20)
    assert reloaded_st.gdn2 is not None and reloaded_st.gdn2.cfg.qed_enabled is False
    with torch.no_grad():
        got_v3 = reloaded_st.model(input_ids=ids).logits
    torch.testing.assert_close(got_v3, expected_v3, atol=1e-6, rtol=1e-6)

    # (b) Reverse guard: syncing an s2_v3-shaped cfg_dict to a LEGACY s2_v2 checkpoint (missing the 7 new keys)
    st_v2 = build_stack(_toy_s2v2_cfg())
    legacy_smeta = copy.deepcopy(st_v2.describe())
    for new_k in (
        "bridge_drop_raw_h_prev",
        "bridge_rms_preserve",
        "bridge_anchor_max_start",
        "bridge_anchor_max_end",
        "bridge_anchor_anneal_frac",
        "bridge_anchor_aux_weight",
        "gate_clamp",
    ):
        legacy_smeta["loop"].pop(new_k, None)
    v3_cfg_dict = copy.deepcopy(v3_cfg)
    _sync_cfg_to_checkpoint_payload(v3_cfg_dict, st_v2.checkpoint_params(), legacy_smeta, {})
    rebuilt_spec = LoopSpec.from_dict(v3_cfg_dict["loop"])
    assert rebuilt_spec.num_passes == 3
    assert rebuilt_spec.bridge_drop_raw_h_prev is False
    assert rebuilt_spec.bridge_rms_preserve is False
    assert rebuilt_spec.bridge_anchor_max_end is None
    assert rebuilt_spec.gate_clamp is None

    # (c) ablate_low_anchor_probe preserves exact SHA-256 on s2_v3 stack
    sha_pre = compute_ablation_state_sha256(
        reloaded_st.model, loop=reloaded_st.loop, gdn2=reloaded_st.gdn2, lsa=reloaded_st.lsa
    )
    with ablate_low_anchor_probe(reloaded_st.model, loop=reloaded_st.loop, max_anchor_weight=0.10) as info:
        assert info["applicable"] is True
        assert info["weights_after"][0][1] <= 0.10 + 1e-6
    sha_post = compute_ablation_state_sha256(
        reloaded_st.model, loop=reloaded_st.loop, gdn2=reloaded_st.gdn2, lsa=reloaded_st.lsa
    )
    assert sha_pre == sha_post

    # (d) Hub round-trip: load s2_v3_loop64_mhc from a Hub repo whose root qloop_config.json came from s2_v2_joint_bptt
    api = _MockHfApi()
    repo_id = "TassP0lar/test-mhc-loop64"
    hub = HubSyncManager(repo_id=repo_id, api=api)
    hub.push_checkpoint("s2_v3_loop64_mhc", ckpt_dir, completed=True, config_dict=v3_cfg)
    v2_run = tmp_path / "runs" / "s2_v2_joint_bptt"
    v2_ckpt_dir = CheckpointManager(str(v2_run), keep_last=1).save(
        step=600,
        params=st_v2.checkpoint_params(),
        optim_state=None,
        state={"stage": "s2_v2_joint_bptt", "stack": legacy_smeta},
        completed=True,
        config_dict=_toy_s2v2_cfg(),
    )
    hub.push_checkpoint("s2_v2_joint_bptt", v2_ckpt_dir, completed=True, config_dict=_toy_s2v2_cfg())

    st_hub_v3 = load_qloop_from_hub(repo_id, stage_name="s2_v3_loop64_mhc", api=api)
    assert st_hub_v3.loop is not None
    assert st_hub_v3.loop.spec.num_passes == 2
    assert st_hub_v3.loop.spec.gate_clamp == (0.0, 1.0)
    assert st_hub_v3.loop.adapters.bridge._anchor_cap == pytest.approx(0.20)
    assert st_hub_v3.gdn2 is not None and st_hub_v3.gdn2.cfg.qed_enabled is False
    with torch.no_grad():
        torch.testing.assert_close(st_hub_v3.model(input_ids=ids).logits, expected_v3, atol=1e-6, rtol=1e-6)

    # Reverse Hub order: push s2_v3 last (root qloop_config.json has the 7 new mHC fields),
    # then load legacy s2_v2_joint_bptt and confirm none of the 7 new mHC fields leak into it
    hub.push_checkpoint("s2_v3_loop64_mhc", ckpt_dir, completed=True, config_dict=v3_cfg)
    st_hub_v2 = load_qloop_from_hub(repo_id, stage_name="s2_v2_joint_bptt", api=api)
    assert st_hub_v2.loop is not None
    assert st_hub_v2.loop.spec.num_passes == 3
    assert st_hub_v2.loop.spec.bridge_drop_raw_h_prev is False
    assert st_hub_v2.loop.spec.bridge_rms_preserve is False
    assert st_hub_v2.loop.spec.bridge_anchor_max_end is None
    assert st_hub_v2.loop.spec.gate_clamp is None
    assert st_hub_v2.loop.adapters.bridge._anchor_cap is None
