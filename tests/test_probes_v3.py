"""Gate 0p zero-training diagnostic probes & structured report tests (`probes.py`)."""
from __future__ import annotations

import json

import pytest
import torch

from qloop.config import apply_overrides
from qloop.model import hssd as hssd_mod
from qloop.model.hssd import HSSDConfig
from qloop.model.loop import LoopSpec, sha256_frozen_experts
from qloop.model.probes import (
    candidate_specs,
    format_probe_report_markdown,
    hssd_ablation_matrix,
    probe_expert_headroom,
    probe_positional_extrapolation,
    probe_reentry_anchors,
    profile_hssd_layer,
    run_structured_probe_suite,
)
from qloop.model.toy import build_toy


def _batches(n: int = 2, b: int = 2, t: int = 24, seed: int = 7) -> list[torch.Tensor]:
    gen = torch.Generator().manual_seed(seed)
    return [torch.randint(0, 97, (b, t), generator=gen) for _ in range(n)]


def test_probe_expert_headroom_schema_determinism_and_tuple_router_fp64():
    model = build_toy().double()
    for p in model.parameters():
        p.requires_grad_(False)
    batches = _batches(n=2, b=2, t=20, seed=11)
    ref_out = model(batches[0]).logits.detach().clone()

    rep1 = probe_expert_headroom(model, batches, n_perturb=4)
    rep2 = probe_expert_headroom(model, batches, n_perturb=4)
    assert rep1["gate"] == "0p-1" and rep1["ok"] is True
    assert rep1["n_perturb"] == 4
    assert rep1["hard_token_headroom_nats"] >= 0.0
    assert rep1["all_token_headroom_nats"] >= 0.0
    assert 0.0 <= rep1["mean_router_jaccard"] <= 1.0
    assert len(rep1["trial_mean_nlls"]) == 4
    assert rep1["hard_token_headroom_nats"] == pytest.approx(rep2["hard_token_headroom_nats"], abs=1e-12)
    assert torch.equal(model(batches[0]).logits, ref_out)

    # Also verify compatibility with HF>=5 tuple TopKRouter returning (probs, top_scores, top_indices)
    class TupleRouter(torch.nn.Module):
        def __init__(self, linear, top_k: int, norm_topk: bool):
            super().__init__()
            self.weight = linear.weight
            self.top_k = top_k
            self.norm_topk = norm_topk

        def forward(self, x):
            flat = x.reshape(-1, x.shape[-1])
            probs = torch.softmax(flat @ self.weight.t(), dim=-1)
            top_w, top_idx = probs.topk(self.top_k, dim=-1)
            if self.norm_topk:
                top_w = top_w / top_w.sum(-1, keepdim=True)
            return probs, top_w.to(x.dtype), top_idx

    class TupleMoE(torch.nn.Module):
        def __init__(self, orig):
            super().__init__()
            self.gate = TupleRouter(orig.gate, orig.top_k, orig.norm_topk)
            self.experts = orig.experts
            self.shared_expert = orig.shared_expert
            self.shared_expert_gate = orig.shared_expert_gate

        def forward(self, x):
            bsz, seq, hid = x.shape
            flat = x.reshape(-1, hid)
            _p, tw, ti = self.gate(flat)
            out = self.experts(flat, ti, tw)
            out = out + torch.sigmoid(self.shared_expert_gate(flat)) * self.shared_expert(flat)
            return out.view(bsz, seq, hid)

    for lyr in model.model.layers:
        lyr.mlp = TupleMoE(lyr.mlp)
    rep_tuple = probe_expert_headroom(model, batches[0], n_perturb=3)
    assert rep_tuple["ok"] is True and len(rep_tuple["trial_router_jaccards"]) == 3


def test_probe_reentry_anchors_subspace_alignment_and_monotonic_sweep():
    model = build_toy().double()
    batches = _batches(n=2, b=2, t=18, seed=19)
    ref_out = model(batches[0]).logits.detach().clone()

    weights = (0.99, 0.90, 0.85, 0.70, 0.50)
    rep = probe_reentry_anchors(model, batches, anchor_weights=weights, subspace_rank=16)
    assert rep["gate"] == "0p-2" and rep["ok"] is True
    assert rep["procrustes_cosine"] >= rep["raw_alignment_cosine"] - 1e-9
    assert rep["subspace_cosine"] >= rep["raw_alignment_cosine"] - 1e-9
    assert 0.0 <= rep["procrustes_explained_variance"] <= 1.0 + 1e-9
    assert 0.0 <= rep["explained_variance"] <= 1.0 + 1e-9
    assert len(rep["sweep"]) == len(weights)
    assert len(rep["recommended_bridge_init"]) == 3
    assert sum(rep["recommended_bridge_init"]) == pytest.approx(1.0, abs=1e-5)

    # Higher w_anchor must strictly increase manifold cosine toward anchor
    cosines = [row["manifold_cosine"] for row in rep["sweep"]]
    assert cosines == sorted(cosines, reverse=True)
    # Near-pure anchor (0.99) preserves higher Pass-2 routing Jaccard than weak anchor (0.50)
    assert rep["sweep"][0]["pass2_routing_jaccard"] >= rep["sweep"][-1]["pass2_routing_jaccard"]
    assert torch.equal(model(batches[0]).logits, ref_out)


def test_probe_positional_extrapolation_modes_and_validation():
    model = build_toy().double()
    ids = _batches(n=1, b=1, t=24, seed=23)[0]
    ref_out = model(ids).logits.detach().clone()

    rep = probe_positional_extrapolation(model, ids, modes=("native", "yarn", "rerope"), factor=4.0)
    assert rep["gate"] == "0p-3" and rep["ok"] is True
    assert set(rep["modes"].keys()) == {"native", "yarn", "rerope"}
    assert rep["recommended_mode"] in ("yarn", "rerope")
    assert rep["target_max_position_embeddings"] == 4 * rep["orig_max_position_embeddings"]

    nat = rep["modes"]["native"]
    yarn = rep["modes"]["yarn"]
    rerope = rep["modes"]["rerope"]
    assert yarn["needle_retrieval_acc"] > nat["needle_retrieval_acc"]
    assert rerope["needle_retrieval_acc"] > nat["needle_retrieval_acc"]
    assert max(yarn["passkey_logit_margin"], rerope["passkey_logit_margin"]) > nat["passkey_logit_margin"]
    assert torch.equal(model(ids).logits, ref_out)

    with pytest.raises(ValueError, match="unknown positional modes"):
        probe_positional_extrapolation(model, ids, modes=("bogus",), factor=4.0)
    with pytest.raises(ValueError, match="factor"):
        probe_positional_extrapolation(model, ids, modes=("native",), factor=0.5)

    # Verify Qwen3.6 262K -> 1M (4x) rotary geometry (`head_dim=64, partial_rotary_factor=0.25, theta=10M`)
    qwen_like = build_toy(
        num_hidden_layers=4,
        hidden_size=64,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=64,
        partial_rotary_factor=0.25,
        rope_theta=10_000_000.0,
    ).double()
    qwen_like.config.max_position_embeddings = 262_144
    rep_q = probe_positional_extrapolation(qwen_like, ids, modes=("native", "yarn", "rerope"), factor=4.0)
    assert rep_q["target_max_position_embeddings"] == 1_048_576
    assert rep_q["rotary_dim"] == 16
    assert rep_q["recommended_mode"] in ("yarn", "rerope")
    assert rep_q["best_passkey_logit_margin"] > rep_q["modes"]["native"]["passkey_logit_margin"]


def test_qwen36_88_layer_candidate_specs():
    types = ["full_attention" if (i + 1) % 4 == 0 else "linear_attention" for i in range(40)]
    specs_88 = candidate_specs(types, target_virtual=88, passes_options=(2, 3))
    assert specs_88
    assert any(s.prelude_end == 8 and s.core_end == 32 and s.num_passes == 3 for s in specs_88)


def test_structured_probe_suite_json_markdown_and_invariance_fp64():
    model = build_toy().double()
    for p in model.parameters():
        p.requires_grad_(False)
    ids = _batches(n=2, b=1, t=24, seed=31)
    sha_ref = sha256_frozen_experts(model)
    logits_ref = model(ids[0]).logits.detach().clone()

    # Pass target_virtual=64 on 16L toy to verify automatic fallback to valid 24L toy target
    report = run_structured_probe_suite(
        model,
        ids,
        target_virtual=64,
        n_perturb=3,
        subspace_rank=16,
        decode_steps=3,
    )

    # 1. Clean JSON round-trip with zero Tensor/NaN serialization errors
    dumped = json.dumps(report, indent=2)
    parsed = json.loads(dumped)
    assert parsed["schema_version"] == "qloop.probe_report.v3"
    assert set(parsed.keys()) >= {
        "schema_version",
        "summary",
        "gates",
        "core_ranges",
        "probes",
        "recommended_overrides",
        "markdown",
    }
    assert parsed["summary"]["requested_target_virtual"] == 64
    assert parsed["summary"]["effective_target_virtual"] == 24
    assert parsed["summary"]["all_critical_ok"] is True
    assert parsed["summary"]["frozen_experts_sha256"] == sha_ref
    assert set(parsed["gates"].keys()) == {
        "0a_backward",
        "0a_surgery",
        "0p_1_expert_headroom",
        "0p_2_reentry_anchors",
        "0p_3_positional_extrapolation",
    }
    assert parsed["gates"]["0a_backward"]["passed"] is True
    assert parsed["gates"]["0a_surgery"]["passed"] is True
    assert parsed["probes"]["surgery"]["decode_vs_full_maxdiff"] < 1e-10

    # 2. Markdown report formatting & shell-safe CLI overrides
    md = format_probe_report_markdown(parsed)
    assert md == parsed["markdown"]
    assert "# Loop-HSSD v3 Zero-Training Diagnostic Probe Report" in md
    assert "Gate `0p-1`" in md and "Gate `0p-2`" in md and "Gate `0p-3`" in md
    assert "-s loop.bridge_mode=subspace" in md

    cli_lines = [ln.strip()[3:] for ln in md.splitlines() if ln.strip().startswith("-s ")]
    assert cli_lines
    assert all(" " not in item for item in cli_lines)  # safe for unquoted shlex.split
    cfg_over = apply_overrides({}, cli_lines)
    spec_from_over = LoopSpec.from_dict(cfg_over["loop"])
    assert spec_from_over.bridge_mode == "subspace"

    # 3. Model weights, frozen-expert SHA-256, and forward logits remain 100% untouched
    assert sha256_frozen_experts(model) == sha_ref
    assert torch.equal(model(ids[0]).logits, logits_ref)


def test_profile_hssd_layer_and_ablation_matrix_schema_and_reexports():
    rep = profile_hssd_layer(
        hidden_size=128,
        num_v_heads=4,
        head_k_dim=32,
        head_v_dim=32,
        seq_len=192,
        stream_chunk_len=64,
        state_dtype="bfloat16",
        theta_low_rank=True,
        rank=16,
        device="cpu",
        dtype=torch.float32,
    )
    required_keys = {
        "dimensions",
        "adapter_params_per_layer",
        "adapter_params_30_layers",
        "theta_param_fraction",
        "full_prefill",
        "streamed_prefill",
        "retained_cache",
        "single_token_decode",
        "training_fwd_bwd",
        "gdn_baseline",
    }
    assert required_keys <= set(rep.keys())
    parsed = json.loads(json.dumps(rep))
    assert parsed["adapter_params_30_layers"] == 30 * parsed["adapter_params_per_layer"]
    assert 0.0 < parsed["theta_param_fraction"] < 1.0
    assert parsed["retained_cache"]["state_dtype"] == "bfloat16"
    assert parsed["retained_cache"]["theta_cum_dtype"] == "float64"
    assert parsed["retained_cache"]["has_compact_storage"] is True
    assert parsed["single_token_decode"]["has_compact_storage"] is True
    assert parsed["streamed_prefill"]["has_compact_storage"] is True
    assert parsed["training_fwd_bwd"]["grad_finite"] is True
    assert parsed["full_prefill"]["gamma_reduction_factor"] > 10.0
    assert parsed["gdn_baseline"]["state_bytes"] > 0

    rep_re = hssd_mod.profile_hssd_layer(
        hidden_size=64,
        num_v_heads=2,
        head_k_dim=16,
        head_v_dim=16,
        seq_len=128,
        stream_chunk_len=64,
        state_dtype="auto",
        theta_low_rank=True,
        rank=8,
        device="cpu",
        dtype=torch.float64,
    )
    assert rep_re["streamed_prefill"]["max_diff_vs_full_prefill"] < 1e-12

    ablations = hssd_ablation_matrix()
    assert ablations == hssd_mod.hssd_ablation_matrix()
    expected_ablation_keys = [
        "1_original_gdn",
        "2_gdn_plus_local_swa",
        "3_multipage_no_phase",
        "4_surprise_controlled",
        "5_phase_enhanced",
        "6_complete_hssd",
    ]
    assert list(ablations.keys()) == expected_ablation_keys
    json.dumps(ablations)
    assert ablations["1_original_gdn"]["enabled"] is False
    for key in expected_ablation_keys[1:]:
        entry = ablations[key]
        assert entry["enabled"] is True
        cfg = HSSDConfig.from_dict(entry["hssd"])
        cfg.validate()


def test_surgery_check_max_probe_tokens_hssd_and_hf5_recurrent_rule():
    from qloop.model.probes import surgery_check

    model = build_toy(num_hidden_layers=8, hidden_size=32).double()
    # Mimic HF>=5 Qwen3_5GatedDeltaNet (`recurrent_gated_delta_rule` instead of `fused_recurrent_gated_delta_rule`)
    for lyr in model.model.layers:
        la = getattr(lyr, "linear_attn", None)
        if la is not None:
            orig_rec = la.fused_recurrent_gated_delta_rule
            delattr(la, "fused_recurrent_gated_delta_rule")
            la.recurrent_gated_delta_rule = orig_rec
            orig_fwd = la.forward

            def _hf5_fwd(
                hidden_states: torch.Tensor,
                *fa: object,
                _la: torch.nn.Module = la,
                _of: object = orig_fwd,
                **fkw: object,
            ) -> torch.Tensor:
                _la.fused_recurrent_gated_delta_rule = _la.recurrent_gated_delta_rule  # type: ignore[attr-defined]
                try:
                    return _of(hidden_states, *fa, **fkw)  # type: ignore[operator]
                finally:
                    delattr(_la, "fused_recurrent_gated_delta_rule")

            la.forward = _hf5_fwd  # type: ignore[method-assign]

    long_ids = torch.randint(0, model.config.vocab_size, (1, 320), generator=torch.Generator().manual_seed(51))
    ref_slice = model(long_ids[:, :64]).logits.detach().clone()
    loop_spec = {"prelude_end": 2, "core_end": 6, "num_passes": 2, "target_virtual_layers": 12}

    rep_g = surgery_check(
        model,
        long_ids,
        loop_spec,
        None,
        decode_steps=4,
        atol=1e-9,
        gdn2_cfg={"rank": 8, "chunk_size": 8},
        max_probe_tokens=64,
    )
    assert rep_g["probe_seq_len"] == 64
    assert rep_g["init_identity_maxdiff"] == 0.0
    assert rep_g["decode_vs_full_maxdiff"] < 1e-10
    assert rep_g["decode_vs_full_cosine"] > 0.999999
    assert rep_g["decode_vs_full_top1_match"] == 1.0
    assert rep_g["base_decode_vs_full_maxdiff"] < 1e-12
    assert rep_g["low_precision_calibrated"] is False
    assert torch.equal(model(long_ids[:, :64]).logits, ref_slice)

    rep_h = surgery_check(
        model,
        long_ids,
        loop_spec,
        None,
        decode_steps=4,
        atol=1e-9,
        hssd_cfg={"rank": 8, "chunk_size": 8, "window_chunks": 2, "checkpoint_segments": False},
        max_probe_tokens=64,
    )
    assert rep_h["probe_seq_len"] == 64
    assert rep_h["decode_vs_full_maxdiff"] < 1e-10
    assert rep_h["decode_vs_full_cosine"] > 0.999999
    assert torch.equal(model(long_ids[:, :64]).logits, ref_slice)


def test_surgery_check_bf16_moe_calibration_vs_broken_cache_detection():
    from qloop.model.probes import surgery_check

    model = build_toy(num_hidden_layers=12, hidden_size=64).to(torch.bfloat16)
    ids = torch.randint(0, model.config.vocab_size, (1, 32), generator=torch.Generator().manual_seed(93))
    loop_spec = {"prelude_end": 4, "core_end": 8, "num_passes": 2, "target_virtual_layers": 16}

    # 1A. Healthy bf16 model with baseline chunk-vs-recurrent ULP drift (`base_diff ~ 0.25`, `diff ~ 0.25 < eff_atol`,
    #     `cosine >= 0.9985`): precision-calibrated check MUST pass!
    orig_head_fwd = model.lm_head.forward

    def _bf16_ulp_baseline_head(x: torch.Tensor) -> torch.Tensor:
        logits = orig_head_fwd(x)
        if x.shape[1] == 1:
            logits = logits.clone()
            logits[..., 0] = logits[..., 0] + torch.tensor(0.25, dtype=logits.dtype, device=logits.device)
        return logits

    model.lm_head.forward = _bf16_ulp_baseline_head  # type: ignore[method-assign]
    try:
        rep_ok = surgery_check(
            model,
            ids,
            loop_spec,
            None,
            decode_steps=4,
            atol=2e-2,
            gdn2_cfg={"rank": 8, "chunk_size": 8},
        )
        assert rep_ok["low_precision_calibrated"] is True
        assert rep_ok["decode_vs_full_maxdiff"] >= 0.20
        assert rep_ok["decode_vs_full_cosine"] >= 0.9985
    finally:
        model.lm_head.forward = orig_head_fwd  # type: ignore[method-assign]

    # 1B. Healthy bf16 model where ONLY 1 decode step of perturbed wrapped pass hits a rare router-tie ULP flip
    #     (`base_diff ~ 0`, `diff ~ 0.22 > eff_atol=0.15`, `mae < 0.01`, `cosine >= 0.9995`): MUST pass!
    tie_calls = [0]

    def _bf16_rare_tie_head(x: torch.Tensor) -> torch.Tensor:
        logits = orig_head_fwd(x)
        if x.shape[1] == 1 and len(model.model.layers) > 12:
            tie_calls[0] += 1
            if tie_calls[0] == 2:
                logits = logits.clone()
                logits[..., 0] = logits[..., 0] + torch.tensor(0.22, dtype=logits.dtype, device=logits.device)
        return logits

    model.lm_head.forward = _bf16_rare_tie_head  # type: ignore[method-assign]
    try:
        rep_tie = surgery_check(
            model,
            ids,
            loop_spec,
            None,
            decode_steps=4,
            atol=2e-2,
            gdn2_cfg={"rank": 8, "chunk_size": 8},
        )
        assert rep_tie["low_precision_calibrated"] is True
        assert rep_tie["decode_vs_full_cosine"] >= 0.9995
        assert rep_tie["decode_vs_full_mae"] < 0.01
    finally:
        model.lm_head.forward = orig_head_fwd  # type: ignore[method-assign]

    # 2. Genuine broken cache / unwrapped recurrent decode (`diff > 1.0`, `base_diff ~ 0`, degraded cosine/MAE)
    #    MUST fail loudly AND still leave `model` 100% unwrapped and restored!
    model.train()
    ref_clean = model.eval()(ids).logits.detach().clone()
    model.train()

    def _broken_wrapped_decode_head(x: torch.Tensor) -> torch.Tensor:
        logits = orig_head_fwd(x)
        if x.shape[1] == 1 and len(model.model.layers) > 12:
            gen_c = torch.Generator().manual_seed(13)
            corrupt = torch.randn(logits.shape, generator=gen_c, dtype=torch.float32).to(logits.dtype) * 1.1
            logits = logits + corrupt
        return logits

    model.lm_head.forward = _broken_wrapped_decode_head  # type: ignore[method-assign]
    try:
        with pytest.raises(AssertionError, match="cached decode diverges from full forward"):
            surgery_check(
                model,
                ids,
                loop_spec,
                None,
                decode_steps=4,
                atol=2e-2,
                gdn2_cfg={"rank": 8, "chunk_size": 8},
            )
    finally:
        model.lm_head.forward = orig_head_fwd  # type: ignore[method-assign]

    assert model.training is True
    assert len(model.model.layers) == 12
    model.eval()
    assert torch.equal(model(ids).logits, ref_clean)

    # 3. Pure unquantized fp32 keeps strict `atol` enforcement (`0.12 > 2e-2` fails!)
    m_fp32 = build_toy(num_hidden_layers=12, hidden_size=64).float().eval()
    orig_fp32_head = m_fp32.lm_head.forward

    def _fp32_drift_head(x: torch.Tensor) -> torch.Tensor:
        logits = orig_fp32_head(x)
        if x.shape[1] == 1 and len(m_fp32.model.layers) > 12:
            logits = logits.clone()
            logits[..., 0] = logits[..., 0] + 0.12
        return logits

    m_fp32.lm_head.forward = _fp32_drift_head  # type: ignore[method-assign]
    try:
        with pytest.raises(AssertionError, match="cached decode diverges from full forward"):
            surgery_check(m_fp32, ids, loop_spec, None, decode_steps=3, atol=2e-2)
    finally:
        m_fp32.lm_head.forward = orig_fp32_head  # type: ignore[method-assign]


