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


def test_surgery_check_blackwell_dynamic_fp8_profile_and_broken_cache():
    from qloop.model.probes import surgery_check

    model = build_toy(num_hidden_layers=20, hidden_size=64, vocab_size=128).to(torch.bfloat16)
    model.config.quantization_config = {"quant_method": "fp8", "activation_scheme": "dynamic"}
    ids = torch.randint(0, model.config.vocab_size, (1, 24), generator=torch.Generator().manual_seed(42))
    # 20 physical -> 44 virtual layers gives exact `depth_mult = 2.2` (matching 40L -> 88L Qwen3.6-FP8)
    loop_spec = {"prelude_end": 4, "core_end": 16, "num_passes": 3, "target_virtual_layers": 44}
    orig_head = model.lm_head.forward
    ref_clean = model.eval()(ids).logits.detach().clone()

    def _make_dynamic_fp8_head(mode: str = "healthy"):
        def _head(x: torch.Tensor) -> torch.Tensor:
            raw = orig_head(x).float()
            v = raw / raw.norm(dim=-1, keepdim=True).clamp_min(1e-6) * 32.0
            top_idx = v.argmax(dim=-1, keepdim=True)
            v.scatter_add_(-1, top_idx, torch.full_like(top_idx, 18.0, dtype=torch.float32))
            if x.shape[1] == 1:
                is_looped = len(model.model.layers) > 20
                vocab = v.shape[-1]
                t_idx = int(top_idx[0, 0].item())
                spike_idx = (t_idx + 1) % vocab
                if is_looped and mode == "broken_cos":
                    # Genuine broken cache: severe directional rotation (`cos < 0.80`)
                    gen_b = torch.Generator().manual_seed(77)
                    rot = torch.randn(vocab, generator=gen_b, dtype=torch.float32).to(v.device) * 4.5
                    v = -0.35 * v + rot.view(1, 1, vocab)
                    return v.to(x.dtype)
                if is_looped and mode == "broken_mae":
                    # Genuine broken cache: global logit drift (`mae > 1.8`) even with positive correlation
                    signs = torch.tensor(
                        [2.2 if (i % 2 == 0) else -2.2 for i in range(vocab)],
                        dtype=torch.float32,
                        device=v.device,
                    )
                    v = v + signs.view(1, 1, vocab)
                    return v.to(x.dtype)

                t_diff = 11.421875 if is_looped else 6.515625
                t_mae = 0.5933 if is_looped else 0.4555
                n_active = 126 if is_looped else 52
                rem_l1 = max(0.0, vocab * t_mae - t_diff)
                fill_mag = rem_l1 / n_active
                delta = torch.zeros(vocab, dtype=torch.float32, device=v.device)
                placed = 0
                for i in range(vocab):
                    if i in (t_idx, spike_idx):
                        continue
                    if placed < n_active:
                        delta[i] = fill_mag if (placed % 2 == 0) else -fill_mag
                        placed += 1
                delta[spike_idx] = -t_diff
                v = v + delta.view(1, 1, vocab)
            return v.to(x.dtype)

        return _head

    # 1. Healthy Blackwell dynamic-FP8 profile (`base_diff ~ 6.5`, `base_mae ~ 0.405`, `base_cos ~ 0.974`,
    #    `depth_mult = 2.2`, `diff ~ 11.4`, `mae ~ 0.53`, `cos ~ 0.958`, `top1_match = 1.0`) MUST pass!
    model.lm_head.forward = _make_dynamic_fp8_head("healthy")  # type: ignore[method-assign]
    try:
        rep = surgery_check(
            model,
            ids,
            loop_spec,
            None,
            decode_steps=8,
            atol=2e-2,
            gdn2_cfg={"rank": 8, "chunk_size": 8},
        )
    finally:
        model.lm_head.forward = orig_head  # type: ignore[method-assign]

    assert rep["low_precision_calibrated"] is True
    assert rep["depth_mult"] == pytest.approx(2.2, abs=1e-6)
    assert rep["base_decode_vs_full_maxdiff"] == pytest.approx(6.515625, abs=0.25)
    assert rep["base_decode_vs_full_mae"] == pytest.approx(0.405, abs=0.03)
    assert rep["base_decode_vs_full_cosine"] == pytest.approx(0.974, abs=0.006)
    assert rep["base_decode_vs_full_top1_match"] == 1.0
    assert rep["decode_vs_full_maxdiff"] == pytest.approx(11.421875, abs=0.25)
    assert rep["decode_vs_full_mae"] == pytest.approx(0.527, abs=0.03)
    assert rep["decode_vs_full_cosine"] == pytest.approx(0.958, abs=0.006)
    assert rep["decode_vs_full_top1_match"] == 1.0
    assert 0.85 <= rep["effective_min_cosine"] < rep["decode_vs_full_cosine"]
    assert rep["effective_mae_atol"] > rep["decode_vs_full_mae"]
    assert torch.equal(model(ids).logits, ref_clean)

    # 2A. Negative test on same dynamic-FP8 mock: genuine broken cache with `cos < 0.80` MUST fail!
    model.lm_head.forward = _make_dynamic_fp8_head("broken_cos")  # type: ignore[method-assign]
    try:
        with pytest.raises(AssertionError, match="cached decode diverges from full forward"):
            surgery_check(
                model,
                ids,
                loop_spec,
                None,
                decode_steps=8,
                atol=2e-2,
                gdn2_cfg={"rank": 8, "chunk_size": 8},
            )
    finally:
        model.lm_head.forward = orig_head  # type: ignore[method-assign]
    assert len(model.model.layers) == 20
    assert torch.equal(model(ids).logits, ref_clean)

    # 2B. Negative test on same dynamic-FP8 mock: genuine broken cache with `mae > 1.8` MUST fail!
    model.lm_head.forward = _make_dynamic_fp8_head("broken_mae")  # type: ignore[method-assign]
    try:
        with pytest.raises(AssertionError, match="cached decode diverges from full forward"):
            surgery_check(
                model,
                ids,
                loop_spec,
                None,
                decode_steps=8,
                atol=2e-2,
                gdn2_cfg={"rank": 8, "chunk_size": 8},
            )
    finally:
        model.lm_head.forward = orig_head  # type: ignore[method-assign]
    assert len(model.model.layers) == 20
    assert torch.equal(model(ids).logits, ref_clean)

    # 3. Clean bf16 (`base_infid <= 2e-3`) broken-cache sensitivity preserved (`cos ~ 0.997 < 0.9985` fails!)
    clean_bf16 = build_toy(num_hidden_layers=12, hidden_size=64, vocab_size=128).to(torch.bfloat16)
    orig_clean_head = clean_bf16.lm_head.forward

    def _subtle_bf16_cos_drop(x: torch.Tensor) -> torch.Tensor:
        raw = orig_clean_head(x).float()
        v = raw / raw.norm(dim=-1, keepdim=True).clamp_min(1e-6) * 10.0
        if x.shape[1] == 1 and len(clean_bf16.model.layers) > 12:
            # Orthogonal jitter yielding step cosine ~ 0.9965 (`mean cos ~ 0.997 < 0.9985`) while diff < 0.15
            vocab = v.shape[-1]
            ortho = torch.tensor(
                [0.075 if (i % 2 == 0) else -0.075 for i in range(vocab)],
                dtype=torch.float32,
                device=v.device,
            )
            ortho = ortho - (ortho * v[0, 0]).sum() / (v[0, 0].pow(2).sum().clamp_min(1e-6)) * v[0, 0]
            ortho = ortho / ortho.norm().clamp_min(1e-6) * 0.85
            v = v + ortho.view(1, 1, vocab)
        return v.to(x.dtype)

    clean_bf16.lm_head.forward = _subtle_bf16_cos_drop  # type: ignore[method-assign]
    try:
        with pytest.raises(AssertionError, match="cached decode diverges from full forward"):
            surgery_check(
                clean_bf16,
                ids,
                {"prelude_end": 4, "core_end": 8, "num_passes": 2, "target_virtual_layers": 16},
                None,
                decode_steps=8,
                atol=2e-2,
            )
    finally:
        clean_bf16.lm_head.forward = orig_clean_head  # type: ignore[method-assign]


def test_assert_logits_unchanged_fast_path_deterministic_leak_and_kernel_jitter():
    from types import SimpleNamespace

    from qloop.model.probes import _assert_logits_unchanged

    ids = torch.tensor([[1, 2, 3, 4]])
    gen = torch.Generator().manual_seed(101)
    ref = torch.randn(1, 4, 64, generator=gen, dtype=torch.float32) * 5.0

    # 1. Exact `torch.equal` fast path: zero extra model forward calls
    calls = [0]

    class CountingModel(torch.nn.Module):
        def forward(self, input_ids: torch.Tensor):
            calls[0] += 1
            return SimpleNamespace(logits=ref.clone())

    det_model = CountingModel()
    _assert_logits_unchanged(det_model, ids, ref.clone(), ref, "fast path must not fail")
    assert calls[0] == 0

    # 2. Deterministic leak rejection (`noise == 0` MUST raise AssertionError even for tiny 4e-4 leak!)
    leaked = ref.clone()
    leaked[0, 1, 3] += 4e-4

    class DeterministicLeakModel(torch.nn.Module):
        def forward(self, input_ids: torch.Tensor):
            calls[0] += 1
            return SimpleNamespace(logits=leaked.clone())

    with pytest.raises(AssertionError, match="deterministic leak"):
        _assert_logits_unchanged(DeterministicLeakModel(), ids, leaked, ref, "deterministic leak")
    assert calls[0] == 1

    # 3. Tiny stochastic 1-ULP kernel jitter (`noise > 0`, `d <= 2.5 * noise`, `cos >= 0.99999`) passes
    jitter_step = [0]

    class StochasticKernelModel(torch.nn.Module):
        def forward(self, input_ids: torch.Tensor):
            jitter_step[0] += 1
            out = ref.clone()
            out[0, 0, jitter_step[0] % 8] += 2e-4
            return SimpleNamespace(logits=out)

    stoch_model = StochasticKernelModel()
    got_jitter = stoch_model(ids).logits
    assert not torch.equal(got_jitter, ref)
    _assert_logits_unchanged(stoch_model, ids, got_jitter, ref, "stochastic jitter should pass")

    # 4. Large drift on a noisy kernel (`d >> 2.5 * noise`) still fails hard (both one-shot and persistent)
    corrupted = ref.clone()
    corrupted[0, 0, 0] += 0.05
    with pytest.raises(AssertionError, match="real drift"):
        _assert_logits_unchanged(stoch_model, ids, corrupted, ref, "real drift")

    class PersistentLeakStochasticModel(torch.nn.Module):
        def forward(self, input_ids: torch.Tensor):
            jitter_step[0] += 1
            out = corrupted.clone()
            out[0, 0, 1 + (jitter_step[0] % 7)] += 2e-4
            return SimpleNamespace(logits=out)

    pers_model = PersistentLeakStochasticModel()
    got_pers = pers_model(ids).logits
    with pytest.raises(AssertionError, match="persistent noisy drift"):
        _assert_logits_unchanged(pers_model, ids, got_pers, ref, "persistent noisy drift")


@pytest.mark.skipif(not hasattr(torch, "float8_e4m3fn"), reason="torch lacks float8_e4m3fn")
def test_fp8_router_headroom_mrope_extrapolation_and_bf16_hssd_suite(monkeypatch):
    from qloop.model import probes as probes_mod

    # 1. Quantized FP8 TupleRouter in `probe_expert_headroom`: must use scaled `_dequant_linear_weight`
    #    rather than raw unscaled `upcast(w_mod)` (which would be 50x off under `weight_scale_inv=0.02`)!
    model_ref = build_toy(num_hidden_layers=8, hidden_size=32).float().eval()
    model_fp8 = build_toy(num_hidden_layers=8, hidden_size=32).float().eval()
    model_fp8.load_state_dict(model_ref.state_dict())
    scale_val = 0.02

    class ScaledTupleRouter(torch.nn.Module):
        def __init__(self, w_fp32: torch.Tensor, top_k: int, norm_topk: bool, quantize: bool):
            super().__init__()
            self.top_k = top_k
            self.norm_topk = norm_topk
            w_q = (w_fp32 / scale_val).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
            if quantize:
                self.weight = torch.nn.Parameter(w_q, requires_grad=False)
                self.register_buffer("weight_scale_inv", torch.tensor(scale_val, dtype=torch.float32))
            else:
                self.weight = torch.nn.Parameter(w_q.float() * scale_val, requires_grad=False)

        def forward(self, x: torch.Tensor):
            flat = x.reshape(-1, x.shape[-1]).float()
            w_eff = (
                self.weight.float() * self.weight_scale_inv.float()
                if hasattr(self, "weight_scale_inv")
                else self.weight.float()
            )
            probs = torch.softmax(flat @ w_eff.t(), dim=-1)
            top_w, top_idx = probs.topk(self.top_k, dim=-1)
            if self.norm_topk:
                top_w = top_w / top_w.sum(-1, keepdim=True)
            return probs.to(x.dtype), top_w.to(x.dtype), top_idx

    class WrappedMoE(torch.nn.Module):
        def __init__(self, orig: torch.nn.Module, quantize: bool):
            super().__init__()
            self.gate = ScaledTupleRouter(orig.gate.weight.detach(), orig.top_k, orig.norm_topk, quantize)
            self.experts = orig.experts
            self.shared_expert = orig.shared_expert
            self.shared_expert_gate = orig.shared_expert_gate

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            bsz, seq, hid = x.shape
            flat = x.reshape(-1, hid)
            _p, tw, ti = self.gate(flat)
            out = self.experts(flat, ti, tw)
            out = out + torch.sigmoid(self.shared_expert_gate(flat)) * self.shared_expert(flat)
            return out.view(bsz, seq, hid)

    for lyr_r, lyr_q in zip(model_ref.model.layers, model_fp8.model.layers):
        lyr_r.mlp = WrappedMoE(lyr_r.mlp, quantize=False)
        lyr_q.mlp = WrappedMoE(lyr_q.mlp, quantize=True)

    batches = _batches(n=1, b=2, t=20, seed=61)
    rep_ref = probe_expert_headroom(model_ref, batches, n_perturb=3)
    rep_fp8 = probe_expert_headroom(model_fp8, batches, n_perturb=3)
    assert rep_fp8["hard_token_headroom_nats"] == pytest.approx(rep_ref["hard_token_headroom_nats"], abs=1e-6)
    assert rep_fp8["mean_router_jaccard"] == pytest.approx(rep_ref["mean_router_jaccard"], abs=1e-6)

    # Also verify graceful fallback to `is_probs` / `upcast(full)` when `_dequant_linear_weight` returns None
    monkeypatch.setattr(probes_mod, "_dequant_linear_weight", lambda *_a, **_kw: None)
    rep_fallback = probe_expert_headroom(model_fp8, batches, n_perturb=2)
    assert rep_fallback["ok"] is True
    monkeypatch.undo()

    # 2. 3D MRoPE `[3, B, T]` `position_ids` + nested `rope_parameters["full_attention"]` in Gate 0p-3
    mrope_model = build_toy(
        num_hidden_layers=4,
        hidden_size=64,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=64,
        partial_rotary_factor=0.25,
        rope_theta=10_000_000.0,
    ).to(torch.bfloat16)
    mrope_model.config.max_position_embeddings = 262_144
    mrope_model.config.rope_parameters = {
        "full_attention": {"rope_theta": 10_000_000.0, "partial_rotary_factor": 0.25}
    }
    orig_rot_fwd = mrope_model.model.rotary_emb.forward
    inner_rot = mrope_model.model.rotary_emb

    class DirectMRoPE(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.inv_freq = inner_rot.inv_freq

        def forward(self, x: torch.Tensor, position_ids: torch.Tensor):
            pos2 = position_ids[0] if position_ids.ndim == 3 else position_ids
            return orig_rot_fwd(x, pos2)

    direct_rot = DirectMRoPE()
    mrope_model.model.rotary_emb = direct_rot  # type: ignore[assignment]
    orig_tm_fwd = mrope_model.model.forward
    mrope_cos_norms: list[float] = []

    def _mrope_tm_fwd(input_ids=None, *a, **kw):
        if input_ids is not None and kw.get("position_ids") is None:
            bsz, seq = input_ids.shape
            pos3 = torch.arange(seq, device=input_ids.device).view(1, 1, seq).expand(3, bsz, seq)
            c3, _s3 = mrope_model.model.rotary_emb(
                torch.empty(bsz, seq, 64, device=input_ids.device),
                pos3,
            )
            mrope_cos_norms.append(float(c3.float().norm()))
        return orig_tm_fwd(input_ids, *a, **kw)

    mrope_model.model.forward = _mrope_tm_fwd  # type: ignore[method-assign]
    rep_mrope = probe_positional_extrapolation(mrope_model, batches[0], modes=("native", "yarn", "rerope"), factor=4.0)
    assert rep_mrope["ok"] is True
    assert rep_mrope["recommended_mode"] in ("yarn", "rerope")
    assert len(mrope_cos_norms) >= 4
    # YaRN mscale > 1.0 changes 3D MRoPE cos norm relative to native
    assert len({round(v, 4) for v in mrope_cos_norms}) >= 2

    # 3. `run_structured_probe_suite` passes `state_dtype="bfloat16"` to `apply_hssd` on bf16/quantized models
    seen_state_dtypes: list[str] = []
    orig_apply_hssd = hssd_mod.apply_hssd

    def _spy_apply_hssd(m: torch.nn.Module, cfg):
        if isinstance(cfg, dict):
            seen_state_dtypes.append(str(cfg.get("state_dtype")))
        return orig_apply_hssd(m, cfg)

    monkeypatch.setattr(hssd_mod, "apply_hssd", _spy_apply_hssd)
    bf16_toy = build_toy(num_hidden_layers=8, hidden_size=32).to(torch.bfloat16)
    for p in bf16_toy.parameters():
        p.requires_grad_(False)
    suite_rep = run_structured_probe_suite(
        bf16_toy,
        batches[0],
        target_virtual=12,
        n_perturb=2,
        subspace_rank=8,
        decode_steps=2,
    )
    assert suite_rep["summary"]["all_critical_ok"] is True
    assert seen_state_dtypes == ["bfloat16"]



