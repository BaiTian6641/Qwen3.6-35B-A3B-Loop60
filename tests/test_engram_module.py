"""Fast, deterministic pure-CPU unit tests for Phase-3 `ZipfianTieredMoEEngram` & `AnchoredMoEEngramBank`.

Verifies all contracts in `src/qloop/memory/engram.py`:
  1. Strict Coda `>= 36` guard (`Spark-X2.6` `S2`) raising `ValueError` unless `allow_coda=True`.
  2. Step-0 exact `torch.equal` (`max_abs_diff == 0.0`) with `gamma=0`, even under non-zero weights
     and active inputs, plus finite non-zero Step-0 `gamma.grad` and full-graph gradient flow.
  3. All 5 stability shields individually:
       (a) Shield 2: `min_active_pos` prefix suppression (`exact 0.0` on `t < min_active_pos`).
       (b) Shield 3: DeepSeek Signed-Sqrt RMSNorm QK gate finite, bounded in `(0, 1)`, monotonic,
           and differentiable with finite gradients.
       (c) Shield 4: `WISE` activation scope gate (`exact 0.0` on orthogonal general tokens +
           `state_dict` reload parity) and relative norm cap (`||delta|| / ||h|| <= 0.03 + 1e-6`
           even under huge weights).
       (d) Shield 5: Independent Sigmoid + positive `s_null` strictly satisfies `0 <= sum g_hat < 1`
           across `N_user in {0, 4, 8, 16}` x `top_k in {1, 2, 4, 8}`, produces exact `0.0` when
           all scores `<= tau` (no NaN/div-by-zero), isolates experts `>= N_user`, enforces module
           dropout (`p=0.15`), and samples Matryoshka nested slices during `elastic_training=True`.
  4. Causal N-gram hashing (`2`-gram, `3`-gram, `skip-2`-gram) strict causality (`0.0` diff at `<= t`
     when mutating `input_ids[:, t+1:]`), `export_ngram_cues()` executing zero SwiGLU experts, and
     `rollback_conv_state(keep_tokens)` matching exact recomputed prefix (`max_diff < 1e-6`) across
     all rejection points `keep_tokens in 0..K`.
  5. Exact `2048`-D parameter formulas for Compact (`1.25B / +0.529B`), Native-Grain
     (`1.25B / +0.379B`), Full Spark-X (`1.89B / +0.945B`), `12 KiB/anchor` `conv1d(k=4)` state,
     and exact `sum(p.numel())` parity against live instantiated modules.
"""
from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from qloop.memory.engram import (
    DEFAULT_CORE_LOOP_SPEC,
    AnchoredMoEEngramBank,
    EngramConfig,
    ZipfianTieredMoEEngram,
    compute_causal_ngram_hashes,
    signed_sqrt,
)
from qloop.model.loop import LoopSpec


def _toy_config(**overrides: object) -> EngramConfig:
    base: dict[str, object] = {
        "d_model": 32,
        "anchor_layers": (15, 19),
        "num_passes": 2,
        "num_experts": 16,
        "top_k": 4,
        "moe_intermediate_size": 24,
        "shared_intermediate_size": 32,
        "tier1a_buckets": 128,
        "tier1b_buckets": 256,
        "tier1_dim": 16,
        "conv_kernel_size": 4,
        "gamma_init": 0.0,
        "scope_threshold": 0.65,
        "max_rel_delta_norm": 0.03,
        "router_threshold": 0.1,
        "module_drop_prob": 0.15,
        "elastic_training": False,
        "seed": 2026,
    }
    base.update(overrides)
    return EngramConfig(**base)  # type: ignore[arg-type]


def _activate_nonzero_weights(bank: AnchoredMoEEngramBank, *, scale: float = 1.0, gamma_val: float = 1.0) -> None:
    """Populate Tier-1 embeddings, conv1d weights, and `gamma` with deterministic non-zero values."""
    gen = torch.Generator(device="cpu").manual_seed(777)
    with torch.no_grad():
        bank.tier1.tier1a_embed.weight.normal_(0.0, scale, generator=gen)
        bank.tier1.tier1b_embed.weight.normal_(0.0, scale, generator=gen)
        for lyr in bank.anchor_layers:
            anc = bank.anchor(lyr)
            anc.conv1d.weight.normal_(0.0, scale, generator=gen)
            anc.gamma.fill_(gamma_val)


# ==============================================================================
# 1. Coda >= 36 Guard & Config Validation
# ==============================================================================
def test_coda_guard_and_config_validation():
    default_cfg = EngramConfig()
    assert default_cfg.anchor_layers == (15, 19, 23, 27, 31, 35)
    assert EngramConfig.from_loop_spec(DEFAULT_CORE_LOOP_SPEC).anchor_layers == (15, 19, 23, 27, 31, 35)

    # Coda layers >= 36 MUST raise ValueError unless `allow_coda=True`
    for bad_layers in ((36,), (39,), (15, 19, 39)):
        with pytest.raises(ValueError, match="Coda anchor layers"):
            _toy_config(anchor_layers=bad_layers)

    # LoopSpec override also enforces custom coda boundary
    custom_spec = LoopSpec(prelude_end=4, core_end=20, num_passes=2, target_virtual_layers=32)
    with pytest.raises(ValueError, match="Coda anchor layers"):
        AnchoredMoEEngramBank(_toy_config(anchor_layers=(15, 23)), loop_spec=custom_spec)

    # Explicit opt-in via `allow_coda=True` succeeds
    coda_ok = _toy_config(anchor_layers=(15, 39), allow_coda=True)
    bank_coda = AnchoredMoEEngramBank(coda_ok)
    assert bank_coda.has_layer(39)
    assert not bank_coda.has_layer(35)

    # `from_dict` and invalid hyperparameter guards
    parsed = EngramConfig.from_dict({"d_model": 32, "anchor_layers": [15, 19], "top_k": 4})
    assert parsed.anchor_layers == (15, 19)
    with pytest.raises(ValueError, match="unknown engram"):
        EngramConfig.from_dict({"not_a_real_engram_field": 123})
    with pytest.raises(ValueError, match="top_k"):
        _toy_config(num_experts=8, top_k=9)
    with pytest.raises(ValueError, match="min_active_pos"):
        _toy_config(min_active_pos=-1)
    with pytest.raises(ValueError, match="module_drop_prob"):
        _toy_config(module_drop_prob=1.5)
    with pytest.raises(ValueError, match="max_rel_delta_norm"):
        _toy_config(max_rel_delta_norm=0.0)


# ==============================================================================
# 2. Shield 1: Step-0 Exact Identity (`gamma=0`) & Clean Autograd Flow
# ==============================================================================
def test_step0_exact_identity_and_grad_flow():
    torch.manual_seed(101)
    cfg = _toy_config(d_model=32, num_experts=16, top_k=4, gamma_init=0.0)
    bank = AnchoredMoEEngramBank(cfg)

    # Verify strict Step-0 zero initializations
    assert torch.equal(bank.tier1.tier1a_embed.weight, torch.zeros_like(bank.tier1.tier1a_embed.weight))
    assert torch.equal(bank.tier1.tier1b_embed.weight, torch.zeros_like(bank.tier1.tier1b_embed.weight))
    assert torch.equal(bank.anchor(15).conv1d.weight, torch.zeros_like(bank.anchor(15).conv1d.weight))
    assert torch.equal(bank.anchor(15).gamma, torch.zeros_like(bank.anchor(15).gamma))

    # Even with LARGE non-zero Tier-1, conv1d, router, and expert weights, `gamma=0` MUST be exact `0.0`!
    _activate_nonzero_weights(bank, scale=3.0, gamma_val=0.0)
    bank.eval()

    ids = torch.randint(0, 50_000, (2, 8), dtype=torch.long)
    for dt in (torch.float32, torch.bfloat16, torch.float64):
        h = torch.randn(2, 8, cfg.d_model, dtype=dt) * 2.5
        for p_idx in range(cfg.num_passes):
            delta = bank(15, h, input_ids=ids, pass_idx=p_idx)
            assert delta.dtype == dt
            assert torch.equal(delta, torch.zeros_like(h))
            assert float(delta.abs().max().item()) == 0.0
            assert torch.equal(h + delta, h)

    # Step-0 backward pass (`gamma == 0.0`): `gamma.grad` must be finite & non-zero, zero NaNs anywhere!
    bank.train()
    h_grad = torch.randn(2, 6, cfg.d_model, dtype=torch.float32, requires_grad=True)
    ids_grad = torch.randint(0, 50_000, (2, 6), dtype=torch.long)
    delta0 = bank(15, h_grad, input_ids=ids_grad, pass_idx=0, force_module_drop=False)
    assert torch.equal(delta0, torch.zeros_like(h_grad))
    loss0 = (h_grad + delta0).square().sum()
    loss0.backward()
    g_grad = bank.anchor(15).gamma.grad
    assert g_grad is not None
    assert torch.isfinite(g_grad).all()
    assert abs(float(g_grad[0].item())) > 1e-6
    for name, p in bank.named_parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all(), f"non-finite Step-0 grad on {name}"

    # Step-1 full-graph gradient flow once `gamma != 0`
    bank.zero_grad(set_to_none=True)
    with torch.no_grad():
        bank.anchor(15).gamma.fill_(0.25)
    h1 = torch.randn(2, 6, cfg.d_model, dtype=torch.float32, requires_grad=True)
    delta1 = bank(15, h1, input_ids=ids_grad, pass_idx=0, force_module_drop=False)
    assert float(delta1.abs().max().item()) > 0.0
    delta1.square().sum().backward()
    anc = bank.anchor(15)
    for required_param in (
        bank.tier1.tier1a_embed.weight,
        bank.tier1.tier1b_embed.weight,
        anc.key_proj.weight,
        anc.val_proj.weight,
        anc.rmsnorm_q.weight,
        anc.rmsnorm_k.weight,
        anc.conv1d.weight,
        anc.router.weight,
        anc.router.bias,
        anc.s_null_raw,
        anc.routed_gate_proj,
        anc.shared_down_proj.weight,
        anc.gamma,
    ):
        assert required_param.grad is not None
        assert torch.isfinite(required_param.grad).all()
        assert float(required_param.grad.abs().max().item()) > 0.0


# ==============================================================================
# 3a. Shield 2: Prompt-Prefix Suppression (`min_active_pos`)
# ==============================================================================
def test_shield2_min_active_pos_prefix_suppression():
    torch.manual_seed(202)
    cfg = _toy_config(d_model=32, min_active_pos=4)
    bank = AnchoredMoEEngramBank(cfg)
    _activate_nonzero_weights(bank, scale=1.0, gamma_val=0.5)
    bank.eval()

    h = torch.randn(2, 10, cfg.d_model)
    ids = torch.randint(0, 1000, (2, 10))
    delta, aux = bank(15, h, input_ids=ids, return_aux=True)

    # Prefix positions `[0..3]` receive EXACT `0.0`; active positions `[4..9]` are non-zero
    assert torch.equal(delta[:, :4], torch.zeros_like(delta[:, :4]))
    assert bool((~aux.active_pos_mask[:, :4]).all())
    assert bool(aux.active_pos_mask[:, 4:].all())
    assert float(delta[:, 4:].norm(dim=-1).min().item()) > 0.0

    # Explicit `positions` tensor override (e.g., incremental decode token positions)
    pos_custom = torch.tensor([[1, 3, 4, 9], [0, 5, 2, 12]], dtype=torch.long)
    delta_pos = bank(15, h[:, :4], input_ids=ids[:, :4], positions=pos_custom)
    suppressed = pos_custom < 4
    active = pos_custom >= 4
    assert torch.equal(delta_pos[suppressed], torch.zeros_like(delta_pos[suppressed]))
    assert float(delta_pos[active].norm(dim=-1).min().item()) > 0.0


# ==============================================================================
# 3b. Shield 3: DeepSeek Signed-Sqrt RMSNorm Query-Key Gate (`arXiv:2601.07372`)
# ==============================================================================
def test_shield3_signed_sqrt_rmsnorm_qk_gate():
    torch.manual_seed(303)
    # Signed-sqrt properties: odd symmetry, exact zero, matches sign(x)*sqrt(|x|), smooth finite grad at 0
    x_probe = torch.tensor([-9.0, -0.25, 0.0, 0.25, 9.0], dtype=torch.float64, requires_grad=True)
    y_probe = signed_sqrt(x_probe, eps=1e-6)
    assert float(y_probe[2].item()) == 0.0
    assert torch.allclose(y_probe, -y_probe.flip(0), atol=1e-12)
    expected = torch.sign(x_probe.detach()) * torch.sqrt(x_probe.detach().abs())
    assert float((y_probe.detach() - expected).abs().max().item()) < 1e-6
    y_probe.sum().backward()
    assert x_probe.grad is not None and torch.isfinite(x_probe.grad).all()
    assert float(x_probe.grad[2].item()) > 0.0

    # QK gate finiteness, strict `(0, 1)` bounds, alignment monotonicity, and gradients
    cfg = _toy_config(d_model=48, tier1_dim=24)
    anc = ZipfianTieredMoEEngram(cfg, anchor_layer=15, own_tier1=True)
    anc.eval()

    e_cue = torch.randn(2, 5, cfg.tier1_dim, requires_grad=True)
    with torch.no_grad():
        k_aligned = F.linear(e_cue, anc.key_proj.weight)
    h_aligned = k_aligned.detach().clone().requires_grad_(True)
    h_anti = (-k_aligned.detach()).clone().requires_grad_(True)
    h_huge = (k_aligned.detach() * 1e5).clone().requires_grad_(True)

    alpha_pos, gv_pos = anc.compute_qk_gate(h_aligned, e_cue)
    alpha_neg, gv_neg = anc.compute_qk_gate(h_anti, e_cue)
    alpha_huge, gv_huge = anc.compute_qk_gate(h_huge, e_cue)

    for alpha_t, gv_t in ((alpha_pos, gv_pos), (alpha_neg, gv_neg), (alpha_huge, gv_huge)):
        assert torch.isfinite(alpha_t).all() and torch.isfinite(gv_t).all()
        assert bool(((alpha_t > 0.0) & (alpha_t < 1.0)).all())

    assert float(alpha_pos.mean().item()) > 0.85
    assert float(alpha_neg.mean().item()) < 0.15
    assert bool((alpha_pos > alpha_neg).all())

    # Gradients flow cleanly through both `h` and `e_cue` and QK parameters
    (gv_pos.square().sum() + alpha_pos.sum()).backward()
    assert h_aligned.grad is not None and torch.isfinite(h_aligned.grad).all()
    assert float(h_aligned.grad.abs().max().item()) > 0.0
    assert e_cue.grad is not None and torch.isfinite(e_cue.grad).all()
    assert float(e_cue.grad.abs().max().item()) > 0.0
    assert anc.key_proj.weight.grad is not None and float(anc.key_proj.weight.grad.abs().max().item()) > 0.0
    assert anc.val_proj.weight.grad is not None and float(anc.val_proj.weight.grad.abs().max().item()) > 0.0


# ==============================================================================
# 3c. Shield 4: WISE Activation Scope Gate + Relative Norm Cap (`<= 0.03`)
# ==============================================================================
def test_shield4_wise_scope_gate_and_relative_norm_cap():
    torch.manual_seed(404)
    d_model = 64
    cfg = _toy_config(d_model=d_model, scope_threshold=0.65, max_rel_delta_norm=0.03)
    bank = AnchoredMoEEngramBank(cfg)
    _activate_nonzero_weights(bank, scale=1.0, gamma_val=1.0)
    bank.eval()

    # Construct cleanly separated orthogonal subspaces: `[:32]` = general tokens, `[32:]` = domain tokens
    h_general = torch.zeros(2, 6, d_model)
    h_general[..., :32] = torch.randn(2, 6, 32)
    h_domain = torch.zeros(2, 6, d_model)
    h_domain[..., 32:] = F.normalize(torch.randn(2, 6, 32), dim=-1) * 4.0

    # Register domain prototypes -> general tokens are strictly orthogonal (`cos == 0 < 0.65`)
    bank.register_scope_prototypes(15, h_domain.reshape(-1, d_model))
    ids = torch.randint(0, 1000, (2, 6))

    delta_gen, aux_gen = bank(15, h_general, input_ids=ids, return_aux=True)
    delta_dom, aux_dom = bank(15, h_domain, input_ids=ids, return_aux=True)
    assert torch.equal(delta_gen, torch.zeros_like(h_general))
    assert bool((~aux_gen.in_scope_mask).all())
    assert bool(aux_dom.in_scope_mask.all())
    assert float(delta_dom.norm(dim=-1).min().item()) > 0.0

    # Verify `state_dict` -> `load_state_dict` round-trip preserves dynamic `scope_prototypes` buffer
    sd = {k: v.clone() for k, v in bank.state_dict().items()}
    reloaded = AnchoredMoEEngramBank(cfg)
    reloaded.load_state_dict(sd)
    reloaded.eval()
    assert torch.equal(reloaded(15, h_general, input_ids=ids), torch.zeros_like(h_general))
    assert torch.equal(reloaded(15, h_domain, input_ids=ids), delta_dom)

    # Clear prototypes -> reverts cleanly to unrestricted pass-through
    bank.clear_scope_prototypes(15)
    assert float(bank(15, h_general, input_ids=ids).norm(dim=-1).min().item()) > 0.0

    # Relative norm cap `||delta|| / ||h|| <= 0.03 + 1e-6` even under HUGE weights (`1e4`) and `gamma=100`!
    with torch.no_grad():
        anc = bank.anchor(15)
        anc.routed_down_proj.mul_(1e4)
        anc.shared_down_proj.weight.mul_(1e4)
        anc.gamma.fill_(100.0)
    h_stress = torch.randn(3, 8, d_model) * 3.0
    ids_stress = torch.randint(0, 1000, (3, 8))
    delta_capped, aux_capped = bank(15, h_stress, input_ids=ids_stress, return_aux=True)
    ratios = delta_capped.norm(dim=-1) / h_stress.norm(dim=-1)
    assert torch.isfinite(delta_capped).all()
    assert bool((ratios <= 0.03 + 1e-6).all()), f"max ratio {float(ratios.max().item())} exceeded 0.03 + 1e-6"
    assert bool((ratios > 0.029).all()), "expected huge weights to saturate the 0.03 norm cap"
    assert float(aux_capped.pre_cap_rel_norm.min().item()) > 1.0
    assert float(aux_capped.post_cap_rel_norm.max().item()) <= 0.03 + 1e-6


# ==============================================================================
# 3d. Shield 5: Independent Sigmoid + `s_null` + Matryoshka + Module Dropout
# ==============================================================================
def test_shield5_independent_sigmoid_null_sink_matryoshka_and_dropout():
    torch.manual_seed(505)
    cfg = _toy_config(
        d_model=32,
        num_experts=16,
        top_k=8,
        router_threshold=0.1,
        module_drop_prob=0.15,
        elastic_training=True,
    )
    anc = ZipfianTieredMoEEngram(cfg, anchor_layer=15, own_tier1=True)
    with torch.no_grad():
        anc.gamma.fill_(0.5)
    anc.eval()

    h = torch.randn(2, 10, cfg.d_model)
    ids = torch.randint(0, 1000, (2, 10))

    # Sweep `N_user in {0, 4, 8, 16}` x `top_k in {1, 2, 4, 8}` including adversarial huge logits & tiny `s_null_raw`
    for s_null_raw_val, router_bias_val in ((0.0, 0.0), (-100.0, 50.0)):
        with torch.no_grad():
            anc.s_null_raw.fill_(s_null_raw_val)
            anc.router.bias.fill_(router_bias_val)
        for n_user in (0, 4, 8, 16):
            for k_val in (1, 2, 4, 8):
                delta, aux = anc(h, input_ids=ids, active_experts=n_user, active_top_k=k_val, return_aux=True)
                g_sum = aux.route.full_weights.sum(dim=-1)
                assert aux.route.full_weights.shape == (2, 10, n_user)
                assert float(aux.route.s_null.item()) > 0.0
                assert torch.isfinite(delta).all()
                assert torch.isfinite(g_sum).all()
                assert bool((aux.route.full_weights >= 0.0).all())
                assert bool(((g_sum >= 0.0) & (g_sum < 1.0)).all()), (
                    f"sum g_hat violated [0, 1) for N={n_user}, K={k_val}: max={float(g_sum.max().item())}"
                )
                assert torch.allclose(g_sum, aux.route.total_gate.squeeze(-1), atol=1e-6)
                if n_user == 0:
                    assert torch.equal(delta, torch.zeros_like(h))
                    assert float(g_sum.abs().max().item()) == 0.0

    # Strict Matryoshka prefix isolation: corrupting experts `>= 8` with NaN MUST NOT affect `active_experts=8`!
    with torch.no_grad():
        anc.s_null_raw.fill_(0.0)
        anc.router.bias.fill_(1.0)
    ref_n8 = anc(h, input_ids=ids, active_experts=8, active_top_k=4).clone()
    saved_tail = anc.routed_down_proj[8:].clone()
    with torch.no_grad():
        anc.routed_down_proj[8:].fill_(float("nan"))
    out_n8 = anc(h, input_ids=ids, active_experts=8, active_top_k=4)
    assert torch.equal(out_n8, ref_n8)
    with torch.no_grad():
        anc.routed_down_proj[8:].copy_(saved_tail)

    # All scores `<= tau`: MUST yield exact `0.0` (zero NaN / zero div-by-zero!)
    with torch.no_grad():
        anc.router.bias.fill_(-30.0)
    delta_below, aux_below = anc(h, input_ids=ids, active_experts=16, active_top_k=4, return_aux=True)
    assert torch.isfinite(delta_below).all()
    assert torch.equal(delta_below, torch.zeros_like(h))
    assert torch.equal(aux_below.route.full_weights, torch.zeros_like(aux_below.route.full_weights))
    assert float(aux_below.route.total_gate.abs().max().item()) == 0.0

    # Training module dropout (`p=0.15`) + Elastic Matryoshka sampling (`elastic_training=True`)
    with torch.no_grad():
        anc.router.bias.fill_(1.0)
    anc.train()
    h_big = torch.randn(400, 4, cfg.d_model)
    ids_big = torch.randint(0, 1000, (400, 4))
    delta_tr, aux_tr = anc(h_big, input_ids=ids_big, active_experts=16, active_top_k=4, return_aux=True)
    drop_frac = float((~aux_tr.module_keep_mask).float().mean().item())
    assert 0.08 < drop_frac < 0.25, f"expected ~0.15 module drop rate, got {drop_frac}"
    dropped_rows = (~aux_tr.module_keep_mask.squeeze(-1).squeeze(-1))
    assert torch.equal(delta_tr[dropped_rows], torch.zeros_like(delta_tr[dropped_rows]))

    # Elastic Matryoshka prefix sampling draws diverse nested `(N_batch, K_batch)` during `train()`
    seen_n: set[int] = set()
    seen_k: set[int] = set()
    for _ in range(40):
        _, aux_el = anc(h[:1, :2], input_ids=ids[:1, :2], force_module_drop=False, return_aux=True)
        seen_n.add(aux_el.route.active_experts)
        seen_k.add(aux_el.route.active_top_k)
        assert aux_el.route.active_top_k <= min(cfg.top_k, aux_el.route.active_experts)
    assert len(seen_n) >= 2 and len(seen_k) >= 2

    # In `eval()` mode, never drops and deterministically uses full `(num_experts, top_k)`
    anc.eval()
    _, aux_ev = anc(h_big[:8], input_ids=ids_big[:8], return_aux=True)
    assert bool(aux_ev.module_keep_mask.all())
    assert aux_ev.route.active_experts == cfg.num_experts
    assert aux_ev.route.active_top_k == cfg.top_k


# ==============================================================================
# 4. Causal N-Gram Hashing & `O(1)` `rollback_conv_state()` Speculative Rejection
# ==============================================================================
def test_causal_ngram_hashing_and_conv1d_speculative_rollback():
    torch.manual_seed(606)
    cfg = _toy_config(d_model=32, anchor_layers=(15, 19), min_active_pos=3, conv_kernel_size=4)
    bank = AnchoredMoEEngramBank(cfg)
    _activate_nonzero_weights(bank, scale=0.8, gamma_val=0.4)
    bank.eval()

    bsz, seq_len = 2, 12
    h = torch.randn(bsz, seq_len, cfg.d_model)
    ids = torch.randint(0, 248_070, (bsz, seq_len), dtype=torch.long)

    # 4a. Strict causality across every prefix cut `t`: mutating `ids[:, t+1:]` produces exact `0.0` diff
    # on N-gram hashes, `export_ngram_cues()`, and causal `conv1d` features, and `< 1e-7` (<= 1 ULP BLAS
    # batch-tiling) on full routed SwiGLU output!
    anc15 = bank.anchor(15)
    base_hashes = compute_causal_ngram_hashes(ids, cfg.tier1a_buckets, cfg.tier1b_buckets)
    base_cues = bank.export_ngram_cues(ids)
    _, base_gv = anc15.compute_qk_gate(h, base_cues)
    base_conv = anc15._apply_causal_conv1d(
        base_gv,
        pass_idx=0,
        use_conv_cache=False,
        conv_state=None,
        token_window=None,
        pos_start=0,
        cache_dtype=h.dtype,
    )
    base_out = bank(15, h, input_ids=ids)

    for t in range(seq_len - 1):
        ids_mut = ids.clone()
        ids_mut[:, t + 1 :] = (ids_mut[:, t + 1 :] + 13_337) % 248_070
        mut_hashes = compute_causal_ngram_hashes(ids_mut, cfg.tier1a_buckets, cfg.tier1b_buckets)
        for bh, mh in zip(base_hashes, mut_hashes):
            assert torch.equal(bh[:, : t + 1], mh[:, : t + 1])
        mut_cues = bank.export_ngram_cues(ids_mut)
        assert torch.equal(base_cues[:, : t + 1], mut_cues[:, : t + 1])
        _, mut_gv = anc15.compute_qk_gate(h, mut_cues)
        mut_conv = anc15._apply_causal_conv1d(
            mut_gv,
            pass_idx=0,
            use_conv_cache=False,
            conv_state=None,
            token_window=None,
            pos_start=0,
            cache_dtype=h.dtype,
        )
        assert torch.equal(base_conv[:, : t + 1], mut_conv[:, : t + 1])
        mut_out = bank(15, h, input_ids=ids_mut)
        assert float((base_out[:, : t + 1] - mut_out[:, : t + 1]).abs().max().item()) < 1e-7

    # 4b. Speculative draft verification & `O(1)` `rollback_conv_state(keep_tokens)` across ALL `keep in 0..K`
    t_pre, k_draft, t_cont = 6, 5, 4
    h_pre = torch.randn(bsz, t_pre, cfg.d_model)
    ids_pre = torch.randint(0, 248_070, (bsz, t_pre), dtype=torch.long)

    h_draft = torch.randn(bsz, k_draft, cfg.d_model)
    ids_draft = torch.randint(0, 248_070, (bsz, k_draft), dtype=torch.long)

    h_cont = torch.randn(bsz, t_cont, cfg.d_model)
    ids_cont = torch.randint(0, 248_070, (bsz, t_cont), dtype=torch.long)

    for keep in range(k_draft + 1):
        bank.reset_conv_state()
        # Step 1: Prefill prefix
        out_pre_p0 = bank(15, h_pre, input_ids=ids_pre, pass_idx=0, use_conv_cache=True)
        out_pre_p1 = bank(15, h_pre * 0.9, input_ids=ids_pre, pass_idx=1, use_conv_cache=True)
        # Step 2: Parallel verification of `K` draft tokens (trailing `K - keep` tokens will be rejected!)
        out_draft_p0 = bank(15, h_draft, input_ids=ids_draft, pass_idx=0, use_conv_cache=True)
        out_draft_p1 = bank(15, h_draft * 0.9, input_ids=ids_draft, pass_idx=1, use_conv_cache=True)
        # Step 3: O(1) speculative rejection rollback across both passes!
        bank.rollback_conv_state(keep, layer_idx=15)
        st0 = bank.anchor(15).get_conv_state(0)
        assert st0 is not None
        assert st0.conv_buffer.shape == (bsz, cfg.d_model, cfg.conv_kernel_size - 1)
        assert st0.pos_cursor == t_pre + keep
        # Step 4: Real continuation after accepted prefix
        out_cont_p0 = bank(15, h_cont, input_ids=ids_cont, pass_idx=0, use_conv_cache=True)
        out_cont_p1 = bank(15, h_cont * 0.9, input_ids=ids_cont, pass_idx=1, use_conv_cache=True)

        # Compare against exact monolithic recomputation on `[prefix, draft[:keep], continuation]`
        h_exact = torch.cat([h_pre, h_draft[:, :keep], h_cont], dim=1)
        ids_exact = torch.cat([ids_pre, ids_draft[:, :keep], ids_cont], dim=1)
        ref_p0 = bank(15, h_exact, input_ids=ids_exact, pass_idx=0, use_conv_cache=False)
        ref_p1 = bank(15, h_exact * 0.9, input_ids=ids_exact, pass_idx=1, use_conv_cache=False)

        streamed_p0 = torch.cat([out_pre_p0, out_draft_p0[:, :keep], out_cont_p0], dim=1)
        streamed_p1 = torch.cat([out_pre_p1, out_draft_p1[:, :keep], out_cont_p1], dim=1)
        diff_p0 = float((streamed_p0 - ref_p0).abs().max().item())
        diff_p1 = float((streamed_p1 - ref_p1).abs().max().item())
        assert diff_p0 < 1e-6, f"keep={keep} pass-0 rollback max_diff={diff_p0} >= 1e-6"
        assert diff_p1 < 1e-6, f"keep={keep} pass-1 rollback max_diff={diff_p1} >= 1e-6"

    # Out-of-range rollback guard
    with pytest.raises(ValueError, match="keep_tokens"):
        bank.rollback_conv_state(999, layer_idx=15)

    # Multi-anchor bank-wide rollback (`layer_idx=None`), live `bf16` state bytes, empty/zero-norm safety
    bank.reset_conv_state()
    h_bf16 = torch.randn(1, 5, cfg.d_model, dtype=torch.bfloat16)
    ids_bf16 = torch.randint(0, 1000, (1, 5), dtype=torch.long)
    _ = bank(15, h_bf16, input_ids=ids_bf16, use_conv_cache=True)
    _ = bank(19, h_bf16, input_ids=ids_bf16, use_conv_cache=True)
    rolled_all = bank.rollback_conv_state(2)
    assert isinstance(rolled_all, dict) and set(rolled_all.keys()) == {15, 19}
    st_bf16 = bank.anchor(15).conv_state
    assert st_bf16 is not None and st_bf16.dtype == torch.bfloat16
    assert st_bf16.numel() * st_bf16.element_size() == bank.conv_state_bytes_per_anchor(batch_size=1, dtype_bytes=2)

    # Zero-norm tokens & empty `T=0` sequence produce exact `0.0` with finite zero-NaN gradients
    h_zero = torch.zeros(2, 4, cfg.d_model, dtype=torch.float32, requires_grad=True)
    out_zero = bank(15, h_zero, input_ids=ids_pre[:, :4])
    assert torch.equal(out_zero, torch.zeros_like(h_zero))
    out_zero.sum().backward()
    assert h_zero.grad is not None and torch.isfinite(h_zero.grad).all()
    assert torch.equal(bank(15, torch.empty(2, 0, cfg.d_model)), torch.empty(2, 0, cfg.d_model))


# ==============================================================================
# 5. `export_ngram_cues` (Zero-SwiGLU) & Exact `2048`-D Parameter/State Formulas
# ==============================================================================
def test_export_ngram_cues_zero_swiglu_and_exact_2048d_formulas():
    torch.manual_seed(707)
    cfg_toy = _toy_config(d_model=32, anchor_layers=(15, 19, 23))
    bank_toy = AnchoredMoEEngramBank(cfg_toy)
    _activate_nonzero_weights(bank_toy, scale=0.5, gamma_val=0.2)

    # Confirm `export_ngram_cues()` NEVER triggers any anchor/router/conv/shared-SwiGLU module!
    fired: list[str] = []
    handles = []
    for lyr in cfg_toy.anchor_layers:
        anc = bank_toy.anchor(lyr)
        for sub_name, sub_mod in (
            (f"L{lyr}.anchor", anc),
            (f"L{lyr}.router", anc.router),
            (f"L{lyr}.conv1d", anc.conv1d),
            (f"L{lyr}.shared_gate", anc.shared_gate_proj),
        ):
            handles.append(
                sub_mod.register_forward_hook(lambda _m, _i, _o, tag=sub_name: fired.append(tag))
            )
    try:
        ids = torch.randint(0, 1000, (2, 7))
        cues = bank_toy.export_ngram_cues(ids)
        assert cues.shape == (2, 7, cfg_toy.tier1_dim)
        assert float(cues.abs().max().item()) > 0.0
        assert fired == [], f"export_ngram_cues unexpectedly executed SwiGLU/anchor hooks: {fired}"
    finally:
        for handle in handles:
            handle.remove()

    # Exact parameter-count parity between closed-form `param_summary()` and live `sum(p.numel())`!
    actual_bank_params = sum(p.numel() for p in bank_toy.parameters())
    assert bank_toy.param_summary()["total_params"] == actual_bank_params

    standalone = ZipfianTieredMoEEngram(cfg_toy, anchor_layer=15, own_tier1=True)
    assert standalone.param_summary()["total_params"] == sum(p.numel() for p in standalone.parameters())
    assert bank_toy.anchor(15).param_summary()["total_params"] == sum(
        p.numel() for p in bank_toy.anchor(15).parameters()
    )

    # Exact `2048`-D closed-form formula verification across all 3 verified presets (zero 7 GB alloc!)
    compact = EngramConfig.compact()
    native = EngramConfig.native_grain()
    spark_x = EngramConfig.full_spark_x()

    s_compact = compact.param_summary()
    s_native = native.param_summary()
    s_spark = spark_x.param_summary()

    # Compact: 32/6/1024/1024 -> `1.25B` MoE total / `+0.529B` active (12 virtual visits)
    assert s_compact["num_anchors"] == 6 and s_compact["num_virtual_visits"] == 12
    assert math.isclose(s_compact["moe_total_params_b"], 1.246101702, rel_tol=0.0, abs_tol=1e-9)
    assert round(float(s_compact["moe_total_params_b"]), 2) == 1.25
    assert math.isclose(s_compact["moe_active_params_b"], 0.529269132, rel_tol=0.0, abs_tol=1e-9)
    assert round(float(s_compact["moe_active_params_b"]), 3) == 0.529
    assert math.isclose(float(s_compact["nvfp4_moe_gib"]), 0.6528, abs_tol=1e-3)

    # Native-Grain: 64/8/512/1024 -> `1.25B` MoE total / `+0.379B` active
    assert math.isclose(s_native["moe_total_params_b"], 1.246495110, rel_tol=0.0, abs_tol=1e-9)
    assert round(float(s_native["moe_total_params_b"]), 2) == 1.25
    assert math.isclose(s_native["moe_active_params_b"], 0.379061004, rel_tol=0.0, abs_tol=1e-9)
    assert round(float(s_native["moe_active_params_b"]), 3) == 0.379
    assert math.isclose(float(s_native["nvfp4_moe_gib"]), 0.6530, abs_tol=1e-3)

    # Full Spark-X: 32/7/1536/2048 -> `1.89B` MoE total / `+0.945B` active (`0.085 GB` shared + `0.446 GB` routed)
    assert math.isclose(s_spark["moe_total_params_b"], 1.887830214, rel_tol=0.0, abs_tol=1e-9)
    assert round(float(s_spark["moe_total_params_b"]), 2) == 1.89
    assert math.isclose(s_spark["moe_active_params_b"], 0.944505228, rel_tol=0.0, abs_tol=1e-9)
    assert round(float(s_spark["moe_active_params_b"]), 3) == 0.945
    assert s_spark["shared_active_params"] == 150_994_944
    assert s_spark["routed_active_params"] == 792_723_456
    assert math.isclose(float(s_spark["nvfp4_shared_active_gb"]), 0.08493, abs_tol=1e-4)
    assert math.isclose(float(s_spark["nvfp4_routed_active_gb"]), 0.44591, abs_tol=1e-4)
    assert math.isclose(float(s_spark["nvfp4_moe_gib"]), 0.9890, abs_tol=1e-3)

    # Exact `12 KiB/anchor` `conv1d(k=4)` rolling state (`3 * 2048 * 2 B = 12,288 B`, `72 KiB` total across 6 anchors)
    for preset_cfg, summary in ((compact, s_compact), (native, s_native), (spark_x, s_spark)):
        assert preset_cfg.conv_state_bytes_per_anchor() == 12_288
        assert preset_cfg.conv_state_bytes() == 73_728
        assert summary["conv_state_bytes_per_anchor"] == 12_288
        assert summary["conv_state_kib_per_anchor"] == 12.0
        assert summary["conv_state_bytes_total"] == 73_728
