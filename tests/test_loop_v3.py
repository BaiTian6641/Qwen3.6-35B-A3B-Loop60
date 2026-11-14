"""Loop-HSSD v3 (`Qwen3.6-35B-A6B-LoopHSSD`) loop-core & ReEntryBridge verification suite."""
from __future__ import annotations

import copy
import math

import pytest
import torch
import torch.nn.functional as F

from qloop.model.introspect import inspect_model
from qloop.model.loop import (
    LoopSpec,
    PassAttnAdapters,
    apply_loop,
    build_schedule,
    sha256_frozen_experts,
)
from qloop.model.mhc import V3_BRIDGE_WEIGHTS, LoopBridge, ReEntryBridge, _rms
from qloop.model.toy import build_toy

TOY_V3_SPEC = {
    "prelude_end": 4,
    "core_end": 12,
    "num_passes": 3,
    "target_virtual_layers": 32,
    "kv_share": True,
    "bridge_init": list(V3_BRIDGE_WEIGHTS),
    "bridge_rank": 16,
    "bridge_mode": "subspace",
    "bridge_up_std": 0.01,
    "attn_adapter_rank": 8,
    "per_pass_router": True,
    "router_adapter_rank": 8,
}


def _ids(b: int = 2, t: int = 10, seed: int = 11, vocab: int = 97) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    return torch.randint(0, vocab, (b, t), generator=gen)


def _randomize_v3_adapters(ctl, scale: float = 0.35, seed: int = 19) -> None:
    gen = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in ctl.trainable_parameters():
            p.add_(torch.randn(p.shape, generator=gen, dtype=p.dtype) * scale)


def test_reentry_bridge_v3_weights_orthogonality_and_ledger():
    torch.manual_seed(42)
    br = ReEntryBridge(hidden=64, n_extra_passes=2, rank=16, up_init_std=0.01).double()
    assert br.adapter_mode == "subspace"
    assert LoopBridge.STREAMS == ("h_prev", "anchor", "h_prev_rms_matched")

    for p in range(2):
        w = br.weights(p)
        torch.testing.assert_close(w, torch.tensor([0.05, 0.85, 0.10], dtype=torch.float64), rtol=0, atol=1e-7)
        gram = br.downs[p].weight @ br.downs[p].weight.T
        torch.testing.assert_close(gram, torch.eye(16, dtype=torch.float64), rtol=0, atol=1e-6)
        up_std = float(br.ups[p].weight.detach().std())
        assert 0.005 < up_std < 0.02

    # When up_init_std=0.0 and step_emb=0, output equals pure convex simplex mix in fp64
    br_zero = LoopBridge(
        hidden=32,
        n_extra_passes=2,
        init_weights=V3_BRIDGE_WEIGHTS,
        rank=8,
        adapter_mode="subspace",
        up_init_std=0.0,
    ).double()
    h = torch.randn(2, 5, 32, dtype=torch.float64) * 7.5
    anchor = torch.randn(2, 5, 32, dtype=torch.float64) * 1.2
    out = br_zero(h, anchor, extra_pass=1)
    matched = h * (_rms(anchor) / _rms(h))
    w_ref = br_zero.weights(1)
    expected = w_ref[0] * h + w_ref[1] * anchor + w_ref[2] * matched
    torch.testing.assert_close(out, expected, rtol=1e-12, atol=1e-12)

    # Verify _rms(anchor) / sqrt(hidden) scaling of subspace branch
    br_scale = ReEntryBridge(hidden=32, n_extra_passes=1, rank=8, step_embedding=False, up_init_std=0.02).double()
    delta_raw = br_scale.ups[0](F.silu(br_scale.downs[0](br_scale.norms[0](h))))
    expected_scaled = expected + (_rms(anchor) / math.sqrt(32)) * delta_raw
    torch.testing.assert_close(br_scale(h, anchor, 0), expected_scaled, rtol=1e-12, atol=1e-12)

    # Verify Qwen3.6-35B-A3B R=3 ReEntryBridge + gates parameter ledger (~2.11M params, Spec §2.3)
    br_qwen = ReEntryBridge(hidden=2048, n_extra_passes=2, rank=256)
    n_bridge = sum(p.numel() for p in br_qwen.parameters())
    n_gates = 2 * 2048
    assert n_bridge + n_gates == 2_109_446  # 2.11M


def test_loop_bridge_legacy_backward_compatibility_and_validation():
    h = torch.randn(2, 4, 16, dtype=torch.float64)
    a = torch.randn(2, 4, 16, dtype=torch.float64)
    br_r0 = LoopBridge(16, 1, init_weights=(0.6, 0.2, 0.2), rank=0).double()
    br_cat = LoopBridge(16, 1, init_weights=(0.6, 0.2, 0.2), rank=4, adapter_mode="concat").double()
    # Zero-init `up` in legacy concat mode matches rank=0 bit-for-bit at init
    assert torch.equal(br_r0(h, a, 0), br_cat(h, a, 0))

    with pytest.raises(ValueError, match="adapter_mode"):
        LoopBridge(16, 1, rank=4, adapter_mode="invalid")
    with pytest.raises(ValueError, match="init_weights"):
        LoopBridge(16, 1, init_weights=(0.5, 0.5))
    with pytest.raises(ValueError, match="attn_proj_names"):
        LoopSpec(attn_adapter_rank=8, attn_proj_names=("q_proj", "k_proj"), kv_share=True).validate(
            ["linear_attention", "linear_attention", "linear_attention", "full_attention"] * 10
        )


def test_qwen36_r1_r2_r3_schedule_is_40_64_88_virtual_layers():
    types = ["full_attention" if (i + 1) % 4 == 0 else "linear_attention" for i in range(40)]
    for tv in (88, None):
        spec = LoopSpec(
            prelude_end=8,
            core_end=32,
            num_passes=3,
            target_virtual_layers=tv,
            bridge_init=V3_BRIDGE_WEIGHTS,
            bridge_rank=256,
            bridge_mode="subspace",
            attn_adapter_rank=64,
        )
        assert spec.validate(types) == []
        sched = build_schedule(spec, types)
        assert len(sched) == 88
        assert [s.phys for s in sched[0:8]] == list(range(0, 8))
        assert [s.phys for s in sched[8:32]] == list(range(8, 32))
        assert [s.phys for s in sched[32:56]] == list(range(8, 32))
        assert [s.phys for s in sched[56:80]] == list(range(8, 32))
        assert [s.phys for s in sched[80:88]] == list(range(32, 40))
        # Pass-2 and Pass-3 core full-attn slots reuse Pass-1 cache_idx (1.0x KV cache growth)
        for v_offset in (24, 48):
            s_full = sched[11 + v_offset]
            assert s_full.phys == 11 and s_full.layer_type == "full_attention" and s_full.cache_idx == 11
            s_lin = sched[8 + v_offset]
            assert s_lin.phys == 8 and s_lin.layer_type == "linear_attention" and s_lin.cache_idx == 8 + v_offset


def test_v3_step0_bit_identity_and_step1_nonzero_gradients_fp64():
    torch.manual_seed(7)
    model = build_toy().double()
    for p in model.parameters():
        p.requires_grad_(False)
    ids = _ids(b=2, t=10, seed=5)

    model.eval()
    ref = model(ids).logits.detach().clone()

    ctl = apply_loop(model, TOY_V3_SPEC)
    assert isinstance(ctl.adapters.pass_attn, PassAttnAdapters)
    assert ctl.adapters.pass_gates is ctl.adapters.gates

    # 1. Exact IEEE-754 step-0 identity (`torch.equal`) despite non-zero ReEntryBridge `up` weights
    out0 = model(ids).logits
    assert torch.equal(ref, out0)

    # 2. Step-0 backward: pass_gates receives non-zero gradient immediately at alpha=0
    model.train()
    loss0 = model(ids, labels=ids).loss
    loss0.backward()
    g0 = ctl.adapters.pass_gates.grad
    assert g0 is not None
    assert float(g0[0].abs().sum()) > 1e-8 and float(g0[1].abs().sum()) > 1e-8

    # 3. Take 1 optimizer step -> Step-1 backward delivers non-zero gradients to gates, bridge, Q/O LoRA & router
    opt = torch.optim.AdamW(ctl.trainable_parameters(), lr=1e-2)
    opt.step()
    opt.zero_grad(set_to_none=True)

    loss1 = model(ids, labels=ids).loss
    loss1.backward()

    assert ctl.adapters.pass_gates.grad is not None and float(ctl.adapters.pass_gates.grad.abs().sum()) > 1e-8
    br = ctl.adapters.bridge
    assert br.mix_logits.grad is not None and float(br.mix_logits.grad.abs().sum()) > 1e-9
    assert br.step_emb is not None and br.step_emb.grad is not None and float(br.step_emb.grad.abs().sum()) > 1e-8
    for p_idx in range(2):
        assert br.norms[p_idx].weight.grad is not None and float(br.norms[p_idx].weight.grad.abs().sum()) > 1e-9
        assert br.downs[p_idx].weight.grad is not None and float(br.downs[p_idx].weight.grad.abs().sum()) > 1e-8
        assert br.ups[p_idx].weight.grad is not None and float(br.ups[p_idx].weight.grad.abs().sum()) > 1e-8

    pa = ctl.adapters.pass_attn
    assert pa is not None
    for pdict in pa.layers.values():
        for proj in pdict.values():
            assert proj.up.grad is not None
            assert float(proj.up.grad[0].abs().sum()) > 1e-9
            assert float(proj.up.grad[1].abs().sum()) > 1e-9

    pr = ctl.adapters.pass_router
    assert pr is not None and pr.up.grad is not None
    assert float(pr.up.grad[0].abs().sum()) > 1e-9 and float(pr.up.grad[1].abs().sum()) > 1e-9


def test_pass_attn_adapters_specialize_qo_while_sharing_kv_fp64():
    model = build_toy().double()
    ctl = apply_loop(model, TOY_V3_SPEC)
    _randomize_v3_adapters(ctl, scale=0.4, seed=23)
    model.eval()

    attn = inspect_model(model).text_model.layers[7].inner.self_attn  # physical layer 7 (core full_attention)
    seen: dict[str, list[torch.Tensor]] = {"q": [], "k": [], "v": [], "o": []}
    handles = [
        attn.q_proj.register_forward_hook(lambda _m, _i, out: seen["q"].append(out.detach().clone())),
        attn.k_proj.register_forward_hook(lambda _m, _i, out: seen["k"].append(out.detach().clone())),
        attn.v_proj.register_forward_hook(lambda _m, _i, out: seen["v"].append(out.detach().clone())),
        attn.o_proj.register_forward_hook(lambda _m, _i, out: seen["o"].append(out.detach().clone())),
    ]
    try:
        model(_ids(b=2, t=8, seed=31))
    finally:
        for h in handles:
            h.remove()

    assert len(seen["k"]) == 3 and len(seen["q"]) == 3
    # K and V remain bit-identical across Passes 1, 2, 3 (`kv_share=True`)
    assert torch.equal(seen["k"][0], seen["k"][1]) and torch.equal(seen["k"][0], seen["k"][2])
    assert torch.equal(seen["v"][0], seen["v"][1]) and torch.equal(seen["v"][0], seen["v"][2])
    # Q and O specialize across Passes 1, 2, 3 (`PassAttnAdapters`)
    assert not torch.allclose(seen["q"][0], seen["q"][1])
    assert not torch.allclose(seen["q"][1], seen["q"][2])
    assert not torch.allclose(seen["o"][0], seen["o"][1])
    assert not torch.allclose(seen["o"][1], seen["o"][2])


def test_v3_checkpoint_gradient_parity_with_subspace_and_pass_attn():
    base = build_toy(seed=0).double()
    grads = []
    for ckpt in (False, True):
        torch.manual_seed(123)
        model = copy.deepcopy(base)
        ctl = apply_loop(model, dict(TOY_V3_SPEC, checkpoint=ckpt, router_diversity_weight=0.1))
        _randomize_v3_adapters(ctl, scale=0.3, seed=41)
        model.train()
        ids = _ids(b=2, t=8, seed=13)
        loss = model(ids, labels=ids).loss + ctl.aux_loss()
        loss.backward()
        grads.append({n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None})
    assert grads[0].keys() == grads[1].keys()
    for name in grads[0]:
        torch.testing.assert_close(grads[0][name], grads[1][name], rtol=1e-10, atol=1e-11, msg=name)


@pytest.mark.parametrize("mor", [False, True])
def test_dynamic_r1_r2_r3_cached_decode_matches_full_forward_fp64(mor: bool):
    model = build_toy().double()
    ids = _ids(b=2, t=11, seed=29)
    model.eval()
    stock_ref = model(ids).logits.detach().clone()

    spec = dict(TOY_V3_SPEC, target_virtual_layers=None, mor=mor)
    ctl = apply_loop(model, spec)
    _randomize_v3_adapters(ctl, scale=0.45, seed=37)
    if mor:
        assert ctl.adapters.mor_router is not None
        with torch.no_grad():
            ctl.adapters.mor_router.weight.normal_(0, 2.0)
            ctl.adapters.mor_router.bias.zero_()

    expected_vlayers = {1: 16, 2: 24, 3: 32}
    outputs_by_r: dict[int, torch.Tensor] = {}

    for r in (1, 2, 3):
        ctl.set_active_passes(r)
        assert ctl.active_passes == r
        assert ctl.active_virtual_layers == expected_vlayers[r]

        full = model(ids).logits
        outputs_by_r[r] = full.detach().clone()
        if r == 1:
            assert torch.equal(full, stock_ref)

        # Prefill 6 tokens + decode 5 tokens one-by-one using HF cache
        prefill = model(ids[:, :6], use_cache=True)
        cache = prefill.past_key_values
        chunks = [prefill.logits]
        for t in range(6, ids.shape[1]):
            step_out = model(ids[:, t : t + 1], past_key_values=cache, use_cache=True)
            chunks.append(step_out.logits)
        inc = torch.cat(chunks, dim=1)

        max_diff = float((full - inc).detach().abs().max())
        assert max_diff < 1e-10, f"R={r} (mor={mor}) decode vs full max_diff={max_diff}"
        assert cache.get_seq_length() == ids.shape[1]
        # Pass-2 (vidx=15,19) and Pass-3 (vidx=23,27) full-attention cache slots remain unallocated
        for v_full in (15, 19, 23, 27):
            assert cache.key_cache[v_full] is None

    if not mor:
        assert not torch.allclose(outputs_by_r[1], outputs_by_r[2])
        assert not torch.allclose(outputs_by_r[2], outputs_by_r[3])

    with pytest.raises(ValueError, match="active passes"):
        ctl.set_active_passes(0)
    with pytest.raises(ValueError, match="active passes"):
        ctl.set_active_passes(4)


def test_virtual_layer_forwards_current_pass_to_lsa_indexer():
    model = build_toy()
    ctl = apply_loop(model, TOY_V3_SPEC)
    recorded_passes: list[int] = []

    class DummyIndexer:
        def set_current_pass(self, r: int) -> None:
            recorded_passes.append(r)

    attn7 = inspect_model(model).text_model.layers[7].inner.self_attn
    attn7.lsa_indexer = DummyIndexer()
    try:
        model(_ids(b=1, t=6))
    finally:
        del attn7.lsa_indexer
        ctl.unwrap()
    # Each core pass invokes set_current_pass(r) on entry and resets to 1 in finally: [1, 1, 2, 1, 3, 1]
    assert recorded_passes == [1, 1, 2, 1, 3, 1]


def test_sha256_frozen_experts_tripwire():
    model = build_toy().double()
    for p in model.parameters():
        p.requires_grad_(False)

    h0 = sha256_frozen_experts(model)
    assert len(h0) == 64 and h0 == sha256_frozen_experts(model)

    # Wrapping with R=3 loop + mutating trainable loop_adapters must NOT alter frozen-expert digest
    ctl = apply_loop(model, TOY_V3_SPEC)
    assert sha256_frozen_experts(model) == h0
    _randomize_v3_adapters(ctl, scale=1.0, seed=99)
    assert sha256_frozen_experts(model) == h0

    # Microscopic 1e-12 mutation to a frozen routed expert immediately trips the SHA-256 checksum
    exp_w = model.model.layers[1].inner.mlp.experts.gate_up_proj[0]
    with torch.no_grad():
        orig_val = float(exp_w[0, 0])
        exp_w[0, 0] = orig_val + 1e-12
    h_mut = sha256_frozen_experts(model)
    assert h_mut != h0

    # Restoring exact bit value restores exact digest
    with torch.no_grad():
        exp_w[0, 0] = orig_val
    assert sha256_frozen_experts(model) == h0

    # Mutating frozen shared_expert also trips digest; unfreezing removes parameter from frozen set
    sh_w = model.model.layers[0].inner.mlp.shared_expert.gate_proj.weight
    sh_w.requires_grad_(True)
    assert sha256_frozen_experts(model) != h0
    sh_w.requires_grad_(False)
    assert sha256_frozen_experts(model) == h0

    ctl.unwrap()
    assert sha256_frozen_experts(model) == h0
