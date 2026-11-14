"""Comprehensive unit tests for NVIDIA Gated DeltaNet-2 (`src/qloop/model/gdn2.py`)."""
from __future__ import annotations

import sys
import types

import pytest
import torch
from torch import nn

from qloop.model.gdn2 import (
    GDN2Config,
    chunk_gdn2,
    install_gdn2,
    logistic_shift,
    recurrent_gdn2,
)
from qloop.model.loop import apply_loop
from qloop.model.probes import surgery_check
from qloop.model.toy import build_toy

TOY_LOOP = {
    "prelude_end": 4,
    "core_end": 12,
    "num_passes": 2,
    "target_virtual_layers": 24,
    "per_pass_router": True,
    "router_adapter_rank": 8,
}


def test_gdn2_config_validation():
    cfg = GDN2Config.from_dict({"enabled": True, "rank": 32, "chunk_size": 32, "segment_chunks": 4})
    cfg.validate()
    with pytest.raises(ValueError, match="unknown gdn2"):
        GDN2Config.from_dict({"bogus_key": 1})
    with pytest.raises(ValueError, match="rank"):
        GDN2Config(rank=0).validate()
    with pytest.raises(ValueError, match="chunk_size"):
        GDN2Config(chunk_size=0).validate()
    with pytest.raises(ValueError, match="gate_logit_cap"):
        GDN2Config(gate_logit_cap=0.0).validate()


def test_logistic_shift_bit_identity_and_singularity_free_boundaries():
    torch.manual_seed(0)
    for dt in (torch.float32, torch.float64):
        beta = torch.rand(4, 7, 8, dtype=dt)
        zeros = torch.zeros_like(beta)
        assert torch.equal(logistic_shift(beta, zeros), beta)

    # Interior exactness vs sigmoid(logit(beta) + delta) in fp64
    beta64 = torch.linspace(1e-4, 1.0 - 1e-4, 101, dtype=torch.float64).view(1, 101, 1)
    delta64 = torch.linspace(-6.0, 6.0, 101, dtype=torch.float64).view(1, 101, 1)
    got = logistic_shift(beta64, delta64)
    ref = torch.sigmoid(torch.logit(beta64) + delta64)
    torch.testing.assert_close(got, ref, rtol=1e-12, atol=1e-12)

    # Exact boundary safety at beta = 0 and beta = 1 (no log(0) / NaN gradients)
    b_edge = torch.tensor([0.0, 1.0], dtype=torch.float64)
    d_edge = torch.tensor([5.0, -5.0], dtype=torch.float64, requires_grad=True)
    y_edge = logistic_shift(b_edge, d_edge)
    assert torch.equal(y_edge.detach(), b_edge)
    y_edge.sum().backward()
    assert d_edge.grad is not None and torch.isfinite(d_edge.grad).all()
    assert torch.equal(d_edge.grad, torch.zeros_like(d_edge))


@pytest.mark.parametrize("extreme_decay", [False, True])
def test_chunk_gdn2_matches_recurrent_gdn2_forward_and_backward_fp64(extreme_decay):
    """Verify chunkwise WY GDN-2 == sequential recurrent GDN-2 in fp64 (incl. padding, GQA, & checkpointing)."""
    torch.manual_seed(42)
    bsz, seq_len = 2, 37  # non-divisible by chunk_size=16; T_pad=48 > seg_len=32 triggers 2 checkpointed segments
    hk, hv, dk, dv = 2, 4, 8, 12

    def _leaf(shape, fn):
        t = fn(shape).to(torch.float64).detach().requires_grad_(True)
        return t

    q = _leaf((bsz, seq_len, hk, dk), torch.randn)
    k = _leaf((bsz, seq_len, hk, dk), torch.randn)
    v = _leaf((bsz, seq_len, hv, dv), torch.randn)
    if extreme_decay:
        # 16 * ~11.5 = ~184 > 150 -> exercises `_decayed_gram_fallback`
        g_raw = -(10.0 + 3.0 * torch.rand(bsz, seq_len, hv, dk, dtype=torch.float64))
    else:
        g_raw = -(0.02 + 0.35 * torch.rand(bsz, seq_len, hv, dk, dtype=torch.float64))
    g_vec = g_raw.detach().requires_grad_(True)
    b_gate = torch.rand(bsz, seq_len, hv, dk, dtype=torch.float64).clamp(0.05, 0.95).detach().requires_grad_(True)
    w_gate = torch.rand(bsz, seq_len, hv, dv, dtype=torch.float64).clamp(0.05, 0.95).detach().requires_grad_(True)
    s0 = (0.25 * torch.randn(bsz, hv, dk, dv, dtype=torch.float64)).detach().requires_grad_(True)

    inputs_rec = [t.detach().clone().requires_grad_(True) for t in (q, k, v, g_vec, b_gate, w_gate, s0)]
    out_rec, st_rec = recurrent_gdn2(
        inputs_rec[0],
        inputs_rec[1],
        inputs_rec[2],
        inputs_rec[3],
        inputs_rec[4],
        inputs_rec[5],
        initial_state=inputs_rec[6],
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
    )
    out_chk, st_chk = chunk_gdn2(
        q,
        k,
        v,
        g_vec,
        b_gate,
        w_gate,
        chunk_size=16,
        segment_chunks=2,
        checkpoint_segments=True,
        initial_state=s0,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
    )
    assert st_rec is not None and st_chk is not None
    torch.testing.assert_close(out_chk, out_rec, rtol=1e-10, atol=1e-10)
    torch.testing.assert_close(st_chk, st_rec, rtol=1e-10, atol=1e-10)

    grad_out = torch.randn_like(out_rec)
    grad_st = torch.randn_like(st_rec)
    ((out_rec * grad_out).sum() + (st_rec * grad_st).sum()).backward()
    ((out_chk * grad_out).sum() + (st_chk * grad_st).sum()).backward()

    for name, t_chk, t_rec in zip(
        ("q", "k", "v", "g_vec", "b_gate", "w_gate", "s0"),
        (q, k, v, g_vec, b_gate, w_gate, s0),
        inputs_rec,
    ):
        assert t_chk.grad is not None and t_rec.grad is not None, name
        torch.testing.assert_close(t_chk.grad, t_rec.grad, rtol=1e-9, atol=1e-9, msg=f"grad mismatch on {name}")


def test_install_gdn2_step0_bit_identity_and_clean_uninstall():
    model = build_toy().double()
    ids = torch.randint(4, model.config.vocab_size, (2, 19))
    model.eval()
    ref_eval = model(ids).logits

    with pytest.raises(ValueError, match="not linear_attention"):
        install_gdn2(model, {"layers": [3]})  # layer 3 is full_attention

    gdn2 = install_gdn2(model, {"rank": 8, "chunk_size": 8, "segment_chunks": 2})
    assert len(gdn2.adapters) == 12
    ctl = apply_loop(model, TOY_LOOP)

    # Eval step-0: fast bit-exact passthrough
    out_eval = model(ids).logits
    assert torch.equal(out_eval, ref_eval)
    st0 = gdn2.stats()
    assert st0["gdn2/adapter_norm"] == 0.0
    assert st0["gdn2/erase_write_l1"] == 0.0

    # Train step-0: executes full `GDN2Adapter.compute_gates` + `chunk_gdn2` WY path and matches stock to ~1e-13
    model.train()
    out_train = model(ids).logits
    torch.testing.assert_close(out_train, ref_eval, rtol=1e-11, atol=1e-11)
    assert gdn2.stats()["gdn2/erase_write_l1"] == 0.0
    assert gdn2.stats()["gdn2/decay_spread"] == 0.0

    # Perturb weights, then unwrap + uninstall -> model must be 100% bit-identical to original
    with torch.no_grad():
        for ad in gdn2.adapters.values():
            ad.erase_up.weight.normal_(0, 0.2)
            ad.write_up.weight.normal_(0, 0.2)
            ad.decay_up.weight.normal_(0, 0.2)
    ctl.unwrap()
    gdn2.uninstall()
    model.eval()
    assert not any(
        hasattr(layer.linear_attn, "gdn2_adapter")
        for i, layer in enumerate(model.model.layers)
        if (i + 1) % 4 != 0
    )
    assert torch.equal(model(ids).logits, ref_eval)


def test_gdn2_decoupled_training_and_cached_decode_with_mor_write_mask():
    torch.manual_seed(7)
    model = build_toy().double()
    for p in model.parameters():
        p.requires_grad_(False)

    gdn2 = install_gdn2(model, {"rank": 8, "chunk_size": 4, "segment_chunks": 2})
    ctl = apply_loop(model, dict(TOY_LOOP, mor=True, write_mask=True, kv_share=True, gate_init=0.2))
    train_params = gdn2.trainable_parameters() + ctl.trainable_parameters()
    for p in train_params:
        p.requires_grad_(True)
    opt = torch.optim.AdamW(train_params, lr=2e-2)

    ids = torch.randint(4, model.config.vocab_size, (2, 11))
    model.train()
    losses = []
    for _ in range(4):
        opt.zero_grad()
        loss = model(ids, labels=ids).loss
        loss.backward()
        opt.step()
        losses.append(float(loss.detach()))

    assert losses[-1] < losses[0]
    st = gdn2.stats()
    assert st["gdn2/adapter_norm"] > 0.0
    assert st["gdn2/erase_write_l1"] > 0.0, "erase and write gates must decouple once trained"
    assert st["gdn2/decay_spread"] > 0.0, "channel-wise decay must diversify across head dims"

    # Force mixed token exits in MoR and verify prefill (`chunk_gdn2`) + decode (`recurrent_gdn2`) == full forward
    with torch.no_grad():
        ctl.adapters.mor_router.weight.normal_(0, 2.0)
        ctl.adapters.mor_router.bias.zero_()
    model.eval()
    full = model(ids).logits
    keep = ctl.ctx.keep[2]
    assert 0 < keep.sum() < keep.numel(), "expected mixed exited and continuing MoR tokens"

    prefill = model(ids[:, :5], use_cache=True)
    cache, steps = prefill.past_key_values, [prefill.logits]
    for t in range(5, ids.shape[1]):
        step_out = model(ids[:, t:t + 1], past_key_values=cache, use_cache=True)
        steps.append(step_out.logits)
    inc = torch.cat(steps, dim=1)
    torch.testing.assert_close(inc, full, rtol=1e-9, atol=1e-10)


def test_gdn2_surgery_check_and_module_level_fallback():
    model = build_toy().double()
    ids = torch.randint(4, model.config.vocab_size, (1, 16))
    rep = surgery_check(
        model,
        ids,
        TOY_LOOP,
        lsa_cfg={"index_dim": 4, "block_size": 8, "pool": 2, "avg_bucket_keys": 2, "max_bucket_keys": 4,
                 "window": 8, "sinks": 2, "num_select": 2},
        decode_steps=4,
        atol=1e-8,
        gdn2_cfg={"rank": 6, "chunk_size": 8, "segment_chunks": 2},
    )
    assert rep["init_identity_maxdiff"] == 0.0
    assert rep["decode_vs_full_maxdiff"] < 1e-9

    # Also test module-level fallback + `head_first=True` tensor layout for custom HF builds
    mod_name = "_fake_hf_gdn_mod"
    fake_mod = types.ModuleType(mod_name)

    def _head_first_rule(q, k, v, g, beta, initial_state=None, output_final_state=False, head_first=True, **kw):
        # q, k, v: [B, H, T, D]; g, beta: [B, H, T]
        qt, kt, vt = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        gt = g.transpose(1, 2).unsqueeze(-1).expand_as(kt)
        bt = beta.transpose(1, 2).unsqueeze(-1).expand_as(kt)
        wt = beta.transpose(1, 2).unsqueeze(-1).expand_as(vt)
        o, st = recurrent_gdn2(
            qt, kt, vt, gt, bt, wt, initial_state=initial_state, output_final_state=output_final_state
        )
        return o.transpose(1, 2), st

    fake_mod.chunk_gated_delta_rule = _head_first_rule
    sys.modules[mod_name] = fake_mod
    try:
        class ModLevelLinearAttn(nn.Module):
            def __init__(self):
                super().__init__()
                self.layer_idx = 0
                self.num_k_heads = self.num_v_heads = 2
                self.head_k_dim = self.head_v_dim = 4
                self.in_proj_qkv = nn.Linear(8, 24, bias=False)
                self.A_log = nn.Parameter(torch.zeros(2))
                self.out_proj = nn.Linear(8, 8, bias=False)

            def forward(self, hidden_states, attention_mask=None):
                b, t, _ = hidden_states.shape
                q, k, v = self.in_proj_qkv(hidden_states).view(b, t, 3, 2, 4).unbind(2)
                g = -torch.ones(b, 2, t, dtype=hidden_states.dtype) * 0.1
                beta = torch.full((b, 2, t), 0.5, dtype=hidden_states.dtype)
                o, _ = fake_mod.chunk_gated_delta_rule(
                    q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), g, beta, head_first=True
                )
                return self.out_proj(o.transpose(1, 2).reshape(b, t, 8))

        ModLevelLinearAttn.__module__ = mod_name

        class MiniLayer(nn.Module):
            def __init__(self):
                super().__init__()
                self.layer_type = "linear_attention"
                self.linear_attn = ModLevelLinearAttn()

            def forward(self, x):
                return self.linear_attn(x)

        class MiniText(nn.Module):
            def __init__(self):
                super().__init__()
                self.config = types.SimpleNamespace(
                    num_hidden_layers=1, layer_types=["linear_attention"], hidden_size=8, vocab_size=16
                )
                self.embed_tokens = nn.Embedding(16, 8)
                self.layers = nn.ModuleList([MiniLayer()])
                self.norm = nn.LayerNorm(8)

        class MiniRoot(nn.Module):
            def __init__(self):
                super().__init__()
                self.model = MiniText()
                self.lm_head = nn.Linear(8, 16, bias=False)

        mini = MiniRoot().double()
        x = torch.randn(2, 6, 8, dtype=torch.float64)
        mini.eval()
        y_ref = mini.model.layers[0](x)
        mgr = install_gdn2(mini, {"rank": 4, "chunk_size": 4})
        assert torch.equal(mini.model.layers[0](x), y_ref)
        with torch.no_grad():
            mgr.adapters[0].erase_up.weight.fill_(0.3)
        y_mod = mini.model.layers[0](x)
        assert not torch.equal(y_mod, y_ref)
        mgr.uninstall()
        assert fake_mod.chunk_gated_delta_rule is _head_first_rule
        assert torch.equal(mini.model.layers[0](x), y_ref)
    finally:
        sys.modules.pop(mod_name, None)
