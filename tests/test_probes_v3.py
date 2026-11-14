"""Gate 0p zero-training diagnostic probes & structured report tests (`probes.py`)."""
from __future__ import annotations

import json

import pytest
import torch

from qloop.config import apply_overrides
from qloop.model.loop import LoopSpec, sha256_frozen_experts
from qloop.model.probes import (
    candidate_specs,
    format_probe_report_markdown,
    probe_expert_headroom,
    probe_positional_extrapolation,
    probe_reentry_anchors,
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
