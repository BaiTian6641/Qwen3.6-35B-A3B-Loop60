"""Comprehensive `float64` mathematical & integration suite for `HSSD v3` and `HSSDDistiller` (M3).

Verifies:
  1. Fix B1 disjoint horizon partition (`0` overlap, `0` gap) + `chunk_hssd == recurrent_hssd`
     (`max_err < 1e-11` in `float64`) across `>= 12` chunks for BOTH `(C=64, window_chunks=8, T=768)`
     AND `(C=8, window_chunks=2, T=96)`, exercising live chunk-start surprise `S_{nC}^{(0)}`,
     `k_far = max(0, n - window_chunks)` transitions, 2D phase rotations, over-relaxed `b_t in (0, 2)`,
     and checkpointed backward gradients.
  2. Fixes H2 & B3 + B5: Finite smooth gradient right at `g = 0.0` (`g -> 0`), bit-exact state
     preservation `S_t == S_{t-1}` across all `M=8` pages, and `PeakBoltzmannGate` size invariance.
  3. Fix B2: Two-sided tiered write-mass budget `handle.aux_loss()` gradients & optimization driving
     empirical page duties into `[0.2 * mu_max, mu_max]`.
  4. Full `ToyHybridMoEForCausalLM` wrapped with `apply_hssd` + `apply_loop(passes=2)` cached
     incremental decode == full forward (`fp64`, `< 1e-10`), MoR early-exit write-mask state
     non-mutation, and bit-exact `remove_hssd(model)` restoration.
  5. `HSSDDistiller` chunked-vocab KL matches exact full-vocab KL (`< 1e-6` fp32, `< 1e-12` fp64)
     forward & backward without materializing `[B, T, V]` logits, and reduces student KL over 5 steps.
"""
from __future__ import annotations

import copy
import hashlib
import math

import pytest
import torch
import torch.nn.functional as F

from qloop.model.gdn2 import GDN2Config, install_gdn2
from qloop.model.hssd import (
    DEFAULT_MU_MAX,
    HSSDCacheState,
    HSSDConfig,
    PeakBoltzmannGate,
    apply_hssd,
    chunk_hssd,
    compute_write_budget_loss,
    disjoint_horizon_bounds,
    recurrent_hssd,
    recurrent_hssd_step,
    remove_hssd,
)
from qloop.model.loop import apply_loop
from qloop.model.toy import ToyForCausalLM as ToyHybridMoEForCausalLM
from qloop.model.toy import build_toy
from qloop.train.distill import (
    HSSDDistiller,
    chunked_vocab_kl_from_logits,
    chunked_vocab_kl_loss,
    exact_full_vocab_kl_loss,
)

TOY_LOOP_SPEC = {"prelude_end": 4, "core_end": 12, "num_passes": 2, "target_virtual_layers": 24}


def _sha256_tensors(tensors: list[torch.Tensor]) -> str:
    h = hashlib.sha256()
    for t in tensors:
        h.update(t.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def _make_synthetic_hssd_inputs(
    *,
    bsz: int,
    seq_len: int,
    hv: int,
    dk: int,
    dv: int,
    num_pages: int = 8,
    seed: int = 2026,
    requires_grad: bool = False,
) -> dict[str, torch.Tensor]:
    gen = torch.Generator().manual_seed(seed)
    dt = torch.float64
    half_dk = dk // 2

    q = F.normalize(torch.randn(bsz, seq_len, hv, dk, generator=gen, dtype=dt), dim=-1)
    k = F.normalize(torch.randn(bsz, seq_len, hv, dk, generator=gen, dtype=dt), dim=-1)
    v = torch.randn(bsz, seq_len, hv, dv, generator=gen, dtype=dt) * 0.4

    gamma_pair = torch.empty(bsz, seq_len, num_pages, hv, half_dk, dtype=dt).uniform_(1e-4, 4e-2, generator=gen)
    gamma = gamma_pair.repeat_interleave(2, dim=-1)
    # Include over-relaxed values in (1, 2) to exercise negative Householder eigenvalues!
    b_gate = torch.empty(bsz, seq_len, hv, dk, dtype=dt).uniform_(0.35, 1.85, generator=gen)
    w_gate = torch.empty(bsz, seq_len, hv, dv, dtype=dt).uniform_(0.20, 0.95, generator=gen)
    theta = torch.empty(bsz, seq_len, hv, half_dk, dtype=dt).uniform_(-0.75 * math.pi, 0.75 * math.pi, generator=gen)

    surprise_bias = torch.randn(bsz, seq_len, hv, generator=gen, dtype=dt) * 0.3
    surprise_slope = torch.full((hv,), 3.0, dtype=dt)
    surprise_thresh = torch.full((hv,), 0.75, dtype=dt)

    write_logits = torch.randn(bsz, seq_len, num_pages, generator=gen, dtype=dt)
    read_logits = torch.randn(bsz, seq_len, num_pages, generator=gen, dtype=dt)

    q_swa = torch.randn(bsz, seq_len, hv, dk, generator=gen, dtype=dt)
    k_swa = torch.randn(bsz, seq_len, hv, dk, generator=gen, dtype=dt)
    v_swa = torch.randn(bsz, seq_len, hv, dv, generator=gen, dtype=dt) * 0.5

    tensors = {
        "query": q,
        "key": k,
        "value": v,
        "gamma": gamma,
        "b_gate": b_gate,
        "w_gate": w_gate,
        "theta": theta,
        "surprise_bias": surprise_bias,
        "surprise_slope": surprise_slope,
        "surprise_thresh": surprise_thresh,
        "write_logits": write_logits,
        "read_logits": read_logits,
        "q_swa": q_swa,
        "k_swa": k_swa,
        "v_swa": v_swa,
    }
    if requires_grad:
        for t in tensors.values():
            t.requires_grad_(True)
    return tensors


# ==============================================================================
# TEST 1: Disjoint Horizon Zero-Overlap + Chunk vs Recurrent Equivalence (fp64)
# ==============================================================================
@pytest.mark.parametrize(
    ("chunk_size", "window_chunks", "seq_len"),
    [
        (64, 8, 768),  # 12 chunks of 64; exercises t <= 512 and t in [513..768] (k_far = 0..3)
        (8, 2, 96),    # 12 chunks of 8; exercises k_far = 0..9 rapidly
    ],
)
def test_01_disjoint_horizon_and_chunk_vs_recurrent_fp64(chunk_size: int, window_chunks: int, seq_len: int):
    # 1a: Exact disjointness & completeness invariant across all t = 1..seq_len
    seen_k_far: set[int] = set()
    for t_1based in range(1, seq_len + 1):
        n, k_far, far_cutoff = disjoint_horizon_bounds(t_1based, chunk_size, window_chunks)
        assert n == (t_1based - 1) // chunk_size
        assert k_far == max(0, n - window_chunks)
        seen_k_far.add(k_far)

        far_set = set(range(1, far_cutoff + 1))
        loc_set = set(range(far_cutoff + 1, t_1based + 1))
        assert far_set.isdisjoint(loc_set), f"Overlap at t={t_1based}"
        assert (far_set | loc_set) == set(range(1, t_1based + 1)), f"Gap at t={t_1based}"
        if t_1based > window_chunks * chunk_size:
            assert window_chunks * chunk_size + 1 <= len(loc_set) <= (window_chunks + 1) * chunk_size
        else:
            assert far_cutoff == 0 and len(loc_set) == t_1based

    assert max(seen_k_far) >= 3, "Must exercise non-zero k_far boundary transitions"

    # 1b: chunk_hssd == step-by-step recurrent_hssd (< 1e-11 in float64) across >= 12 chunks
    hv = 2
    dk = 16 if chunk_size == 64 else 8
    dv = dk
    inp = _make_synthetic_hssd_inputs(
        bsz=2, seq_len=seq_len, hv=hv, dk=dk, dv=dv, num_pages=8, seed=100 + chunk_size
    )
    gate = PeakBoltzmannGate(hv, kappa_init=2.1, tau_init=1.9).double()

    out_chunk, st_chunk = chunk_hssd(
        **inp,
        boltzmann_gate=gate,
        chunk_size=chunk_size,
        window_chunks=window_chunks,
        num_pages=8,
        top_k_pages=2,
        segment_chunks=3,
        checkpoint_segments=True,
        output_final_state=True,
        scale=1.0,
    )
    out_rec, st_rec = recurrent_hssd(
        **inp,
        boltzmann_gate=gate,
        chunk_size=chunk_size,
        window_chunks=window_chunks,
        num_pages=8,
        top_k_pages=2,
        output_final_state=True,
        scale=1.0,
    )

    assert st_chunk is not None and st_rec is not None
    max_out_err = float((out_chunk - out_rec).detach().abs().max())
    max_state_err = float((st_chunk.page_states - st_rec.page_states).detach().abs().max())
    max_cstart_err = float((st_chunk.s_chunk_start - st_rec.s_chunk_start).detach().abs().max())
    max_far_err = float((st_chunk.boundary_ring[0] - st_rec.boundary_ring[0]).detach().abs().max())

    assert max_out_err < 1e-11, f"max_out_err={max_out_err}"
    assert max_state_err < 1e-11, f"max_state_err={max_state_err}"
    assert max_cstart_err < 1e-11, f"max_cstart_err={max_cstart_err}"
    assert max_far_err < 1e-11, f"max_far_err={max_far_err}"

    # Prove both chunk-start primary state S_{nC}^{(0)} and archived far boundary S_boundary[k_far] evolved!
    assert float(st_chunk.s_chunk_start.abs().max()) > 1e-3
    assert float(st_chunk.boundary_ring[0].abs().max()) > 1e-3


def test_01b_chunk_vs_recurrent_checkpointed_backward_and_partial_tail_fp64():
    """Verify backward gradients & non-chunk-multiple `T = 101` (`12 * 8 + 5`) match in fp64."""
    hv, dk, dv = 2, 8, 8
    inp_a = _make_synthetic_hssd_inputs(bsz=2, seq_len=101, hv=hv, dk=dk, dv=dv, seed=77, requires_grad=True)
    inp_b = {k: v.detach().clone().requires_grad_(True) for k, v in inp_a.items()}
    gate_a = PeakBoltzmannGate(hv, kappa_init=2.0, tau_init=2.0).double()
    gate_b = copy.deepcopy(gate_a)

    out_c, st_c, duty_c = chunk_hssd(
        **inp_a,
        boltzmann_gate=gate_a,
        chunk_size=8,
        window_chunks=2,
        segment_chunks=3,
        checkpoint_segments=True,
        output_final_state=True,
        return_aux=True,
    )
    out_r, st_r, duty_r = recurrent_hssd(
        **inp_b,
        boltzmann_gate=gate_b,
        chunk_size=8,
        window_chunks=2,
        output_final_state=True,
        return_aux=True,
    )
    assert st_c is not None and st_r is not None
    assert float((out_c - out_r).detach().abs().max()) < 1e-11
    assert float((duty_c - duty_r).detach().abs().max()) < 1e-12

    weights = torch.linspace(0.5, 1.5, out_c.numel(), dtype=torch.float64).reshape_as(out_c)
    loss_c = (out_c * weights).sum() + 0.3 * st_c.page_states.pow(2).sum() + duty_c.sum()
    loss_r = (out_r * weights).sum() + 0.3 * st_r.page_states.pow(2).sum() + duty_r.sum()
    loss_c.backward()
    loss_r.backward()

    for key in inp_a:
        ga, gb = inp_a[key].grad, inp_b[key].grad
        assert ga is not None and gb is not None
        assert torch.isfinite(ga).all()
        assert float((ga - gb).abs().max()) < 1e-10, f"grad mismatch on {key}"
    assert float((gate_a.log_kappa.grad - gate_b.log_kappa.grad).abs().max()) < 1e-10
    assert float((gate_a.tau.grad - gate_b.tau.grad).abs().max()) < 1e-10


# ==============================================================================
# TEST 2: Finite Gradient at g -> 0 + Exact State Preservation + Boltzmann Gate
# ==============================================================================
def test_02_root_surprise_zero_grad_finiteness_and_exact_non_erasure():
    torch.manual_seed(2026)
    dt = torch.float64
    bsz, seq_len, hv, dk, dv, num_pages = 2, 16, 2, 8, 8, 8
    inp = _make_synthetic_hssd_inputs(bsz=bsz, seq_len=seq_len, hv=hv, dk=dk, dv=dv, num_pages=num_pages, seed=314)
    s_init = torch.randn(bsz, num_pages, hv, dk, dv, dtype=dt)
    gate = PeakBoltzmannGate(hv).double()

    # 2a: Exact g = 0.0 on chunk_hssd -> bit-exact S_t == S_0 across all M=8 pages + finite gradient!
    g_zero_chunk = torch.zeros(bsz, seq_len, hv, dtype=dt, requires_grad=True)
    out_c, st_c = chunk_hssd(
        **inp,
        boltzmann_gate=gate,
        chunk_size=8,
        window_chunks=2,
        num_pages=num_pages,
        initial_state=s_init,
        output_final_state=True,
        g_override=g_zero_chunk,
    )
    assert st_c is not None
    assert torch.equal(st_c.page_states.detach(), s_init), "chunk_hssd mutated state at g = 0!"
    loss_c = out_c.pow(2).sum() + (st_c.page_states * torch.randn_like(s_init)).sum()
    loss_c.backward()
    assert g_zero_chunk.grad is not None and torch.isfinite(g_zero_chunk.grad).all()
    assert torch.equal(g_zero_chunk.grad, torch.zeros_like(g_zero_chunk))  # d(g^2)/dg|_{g=0} == 0 (never inf/NaN)

    # 2b: Exact g = 0.0 on single-step recurrent_hssd_step -> bit-exact S_t == S_{t-1} + finite grad!
    st0 = HSSDCacheState.from_tensor(s_init, num_pages=num_pages, chunk_size=8, window_chunks=2)
    g_zero_step = torch.zeros(bsz, 1, hv, dtype=dt, requires_grad=True)
    step_inp = {k: (v[:, :1] if v.ndim >= 3 else v) for k, v in inp.items()}
    out_s, st_s = recurrent_hssd_step(
        **step_inp,
        state=st0,
        boltzmann_gate=gate,
        g_override=g_zero_step,
    )
    assert torch.equal(st_s.page_states.detach(), s_init), "recurrent_hssd_step mutated state at g = 0!"
    (out_s.sum() + st_s.page_states.sum()).backward()
    assert g_zero_step.grad is not None and torch.isfinite(g_zero_step.grad).all()

    # 2c: Asymptotic g -> 0+ (g = 1e-6 vs g = 1e-10) is finite, non-zero, and scales smoothly as O(g)
    grads_near_zero: list[float] = []
    probe_w = torch.randn_like(s_init)
    for eps in (1e-6, 1e-10):
        g_eps = torch.full((bsz, 1, hv), eps, dtype=dt, requires_grad=True)
        _, st_eps = recurrent_hssd_step(**step_inp, state=st0, boltzmann_gate=gate, g_override=g_eps)
        (st_eps.page_states * probe_w).sum().backward()
        assert g_eps.grad is not None and torch.isfinite(g_eps.grad).all()
        gnorm = float(g_eps.grad.norm())
        assert gnorm > 0.0
        grads_near_zero.append(gnorm)
    assert abs((grads_near_zero[0] / grads_near_zero[1]) - 1e4) < 1.0

    # 2d: PeakBoltzmannGate size invariance across N_t in {64, 256, 576} & all-masked safety
    pgate = PeakBoltzmannGate(1, kappa_init=2.0, tau_init=2.2).double()
    d_probe = 128
    torch.manual_seed(100)
    for n_win in (64, 256, 576):
        q_p = F.rms_norm(torch.randn(d_probe, dtype=dt), (d_probe,))
        k_miss = F.rms_norm(torch.randn(n_win, d_probe, dtype=dt), (d_probe,))
        k_hit = k_miss.clone()
        k_hit[n_win // 2] = F.rms_norm(q_p + 0.05 * torch.randn(d_probe, dtype=dt), (d_probe,))
        p_miss = float(pgate((k_miss @ q_p) / math.sqrt(d_probe)).detach())
        p_hit = float(pgate((k_hit @ q_p) / math.sqrt(d_probe)).detach())
        assert p_hit > 0.95 and p_miss < 0.08 and (p_hit - p_miss) > 0.88

    sc_masked = torch.randn(2, 1, 8, dtype=dt, requires_grad=True)
    vm_empty = torch.zeros(2, 1, 8, dtype=torch.bool)
    p_empty = pgate(sc_masked, valid_mask=vm_empty)
    assert torch.equal(p_empty.detach(), torch.zeros(2, 1, dtype=dt))
    p_empty.sum().backward()
    assert sc_masked.grad is not None and torch.isfinite(sc_masked.grad).all()


# ==============================================================================
# TEST 3: Tiered M=8 Write-Mass Budget aux_loss() Gradient & Page Duty Bounds
# ==============================================================================
def test_03_write_budget_aux_loss_gradients_and_tiered_page_duty_bounds():
    torch.manual_seed(42)
    model = build_toy(num_hidden_layers=8, hidden_size=32).double()
    assert isinstance(model, ToyHybridMoEForCausalLM)
    cfg = HSSDConfig(rank=16, chunk_size=8, window_chunks=2, budget_weight=10.0)
    handle = apply_hssd(model, cfg)
    model.train()

    ids = torch.randint(0, model.config.vocab_size, (2, 24))
    _ = model(ids)
    aux0 = handle.aux_loss()
    assert aux0.ndim == 0 and float(aux0.detach()) > 0.0

    aux0.backward()
    ad0 = handle.adapters[0]
    assert ad0.write_router.weight.grad is not None and float(ad0.write_router.weight.grad.abs().sum()) > 0.0
    assert ad0.write_router.bias.grad is not None and float(ad0.write_router.bias.grad.abs().sum()) > 0.0
    assert ad0.surprise_proj.bias.grad is not None and float(ad0.surprise_proj.bias.grad.abs().sum()) > 0.0
    assert ad0.log_surprise_slope.grad is not None and torch.isfinite(ad0.log_surprise_slope.grad).all()

    # Optimize HSSD parameters on aux_loss() and verify empirical page duties land inside [0.2 * mu_max, mu_max]
    opt = torch.optim.Adam(handle.trainable_parameters(), lr=0.08)
    final_aux = float(aux0.detach())
    for _ in range(180):
        opt.zero_grad()
        _ = model(ids)
        loss_bud = handle.aux_loss()
        loss_bud.backward()
        opt.step()
        final_aux = float(loss_bud.detach())

    assert final_aux < float(aux0.detach()) * 0.01
    mu_max = torch.tensor(DEFAULT_MU_MAX, dtype=torch.float64)
    mu_min = 0.2 * mu_max
    duties = handle.page_duties()
    assert len(duties) == 6  # 6 linear layers out of 8 in 3:1 Toy model
    for lid, u_vec in duties.items():
        assert bool((u_vec <= mu_max + 5e-4).all()), f"layer {lid} exceeded mu_max: {u_vec.tolist()}"
        assert bool((u_vec >= mu_min - 5e-4).all()), f"layer {lid} fell below 0.2*mu_max: {u_vec.tolist()}"
        # Vault pages (0, 1) have ~100x stricter duty than Scratchpad pages (4..7)
        assert float(u_vec[:2].mean()) < 0.0035
        assert float(u_vec[4:].mean()) >= 0.048

    # Verify compute_write_budget_loss is identically zero inside the feasible box [0.2 * mu_max, mu_max]
    feasible_u = 0.6 * mu_max
    assert float(compute_write_budget_loss(feasible_u, mu_max, mu_min_ratio=0.2)) == 0.0


# ==============================================================================
# TEST 4: ToyHybridMoEForCausalLM + apply_hssd + apply_loop(passes=2) + MoR
# ==============================================================================
def _perturb_hssd_and_loop(handle, ctl, *, mor: bool, seed: int = 19) -> None:
    gen = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in handle.trainable_parameters():
            p.add_(torch.randn(p.shape, generator=gen, dtype=p.dtype) * 0.35)
        for ad in handle.adapters.values():
            # Push some erase channels positive so b_gate enters (1, 2) over-relaxed regime
            ad.erase_up.weight.add_(0.8)
            ad.theta_proj.weight.normal_(0.0, 0.25, generator=gen)
        for p in ctl.trainable_parameters():
            p.add_(torch.randn(p.shape, generator=gen, dtype=p.dtype) * 0.4)
        if mor and ctl.adapters.mor_router is not None:
            ctl.adapters.mor_router.weight.normal_(0.0, 2.0, generator=gen)
            ctl.adapters.mor_router.bias.zero_()


@pytest.mark.parametrize("mor", [False, True])
def test_04_toy_hssd_loop_incremental_decode_and_mor_write_mask_fp64(mor: bool):
    model = build_toy().double()
    ids = torch.randint(0, model.config.vocab_size, (2, 29), generator=torch.Generator().manual_seed(123))
    orig_logits = model(ids).logits.detach().clone()

    hssd_cfg = HSSDConfig(rank=16, chunk_size=8, window_chunks=2, segment_chunks=2, checkpoint_segments=False)
    handle = apply_hssd(model, hssd_cfg)
    ctl = apply_loop(model, dict(TOY_LOOP_SPEC, mor=mor, kv_share=True, per_pass_router=True))
    _perturb_hssd_and_loop(handle, ctl, mor=mor, seed=91 if mor else 37)
    model.eval()

    full_logits = model(ids).logits
    if mor:
        keep_full = ctl.ctx.keep[2]
        assert 0 < keep_full.sum() < keep_full.numel(), "MoR test requires both exited and active tokens"

    # Non-chunk-aligned prefill (5 tokens < C=8), multi-token chunk (6 tokens crossing boundary 8),
    # then token-by-token decode across t = 11..28 (crossing C=16, 24 so k_far = max(0, n - 2) > 0!)
    out_p = model(ids[:, :5], use_cache=True)
    cache = out_p.past_key_values
    pieces = [out_p.logits]

    out_mid = model(ids[:, 5:11], past_key_values=cache, use_cache=True)
    pieces.append(out_mid.logits)

    saw_exited_freeze_check = False
    for t in range(11, ids.shape[1]):
        prev_pass2_state = cache.recurrent_states[12].clone() if cache.recurrent_states[12] is not None else None
        out_t = model(ids[:, t : t + 1], past_key_values=cache, use_cache=True)
        pieces.append(out_t.logits)

        if mor and prev_pass2_state is not None:
            keep_t = ctl.ctx.keep[2][:, 0]
            cur_pass2_state = cache.recurrent_states[12]
            assert isinstance(cur_pass2_state, HSSDCacheState)
            for b in range(ids.shape[0]):
                if float(keep_t[b]) == 0.0:
                    saw_exited_freeze_check = True
                    assert torch.equal(cur_pass2_state.page_states[b], prev_pass2_state.page_states[b])
                    assert torch.equal(cur_pass2_state.theta_cum[b], prev_pass2_state.theta_cum[b])

    if mor:
        assert saw_exited_freeze_check, "Expected at least one early-exited token during decode steps"

    inc_logits = torch.cat(pieces, dim=1)
    max_err = float((inc_logits - full_logits).detach().abs().max())
    assert max_err < 1e-10, f"Incremental decode mismatch (mor={mor}): max_err={max_err}"

    # Verify per-pass virtual cache slots are isolated HSSDCacheState instances
    st_p1 = cache.recurrent_states[4]
    st_p2 = cache.recurrent_states[12]
    assert isinstance(st_p1, HSSDCacheState) and isinstance(st_p2, HSSDCacheState)
    assert st_p1.step_count == ids.shape[1] and st_p2.step_count == ids.shape[1]
    assert not torch.equal(st_p1.page_states, st_p2.page_states)

    # Clean unwrap + remove_hssd restores exact step-0 model output
    ctl.unwrap()
    remove_hssd(model)
    assert not hasattr(model, "_hssd_handle")
    assert not hasattr(model.model.layers[0].linear_attn, "hssd_adapter")
    assert torch.equal(model(ids).logits, orig_logits)


# ==============================================================================
# TEST 5: HSSDDistiller Chunked-Vocab KL Parity & 5-Step Student Convergence
# ==============================================================================
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("chunk_size", [16, 4096])
@pytest.mark.parametrize("temperature", [1.0, 2.0])
def test_05a_chunked_vocab_kl_matches_exact_full_vocab_forward_and_backward(
    dtype: torch.dtype, chunk_size: int, temperature: float
):
    gen = torch.Generator().manual_seed(505)
    bsz, seq_len, hidden, vocab = 3, 11, 24, 113  # prime vocab > 16 => 8 uneven vocab tiles at chunk_size=16!

    h_t = torch.randn(bsz, seq_len, hidden, generator=gen, dtype=dtype)
    w_t = torch.randn(vocab, hidden, generator=gen, dtype=dtype) * 0.3
    b_t = torch.randn(vocab, generator=gen, dtype=dtype) * 0.1

    h_s1 = torch.randn(bsz, seq_len, hidden, generator=gen, dtype=dtype, requires_grad=True)
    w_s1 = (torch.randn(vocab, hidden, generator=gen, dtype=dtype) * 0.3).detach().requires_grad_(True)
    b_s1 = (torch.randn(vocab, generator=gen, dtype=dtype) * 0.1).detach().requires_grad_(True)

    h_s2 = h_s1.detach().clone().requires_grad_(True)
    w_s2 = w_s1.detach().clone().requires_grad_(True)
    b_s2 = b_s1.detach().clone().requires_grad_(True)

    mask = torch.ones(bsz, seq_len, dtype=torch.bool)
    mask[0, -2:] = False
    mask[2, 0] = False

    kl_chunk, n_tok = chunked_vocab_kl_loss(
        h_t,
        w_t,
        h_s1,
        w_s1,
        b_teacher=b_t,
        b_student=b_s1,
        loss_mask=mask,
        chunk_size=chunk_size,
        token_chunk_size=7,
        temperature=temperature,
    )
    assert n_tok == int(mask.sum().item())
    kl_chunk.backward()

    logits_t = F.linear(h_t, w_t, b_t)
    logits_s = F.linear(h_s2, w_s2, b_s2)
    kl_exact = exact_full_vocab_kl_loss(logits_t, logits_s, loss_mask=mask, temperature=temperature)
    kl_exact.backward()

    tol = 1e-12 if dtype == torch.float64 else 1e-6
    assert abs(float(kl_chunk.detach() - kl_exact.detach())) < tol
    assert float((h_s1.grad - h_s2.grad).abs().max()) < tol
    assert float((w_s1.grad - w_s2.grad).abs().max()) < tol
    assert float((b_s1.grad - b_s2.grad).abs().max()) < tol

    # Also check chunked_vocab_kl_from_logits
    ls_direct = logits_s.detach().clone().requires_grad_(True)
    kl_from_logits = chunked_vocab_kl_from_logits(
        logits_t.detach(), ls_direct, loss_mask=mask, chunk_size=chunk_size, temperature=temperature
    )
    assert abs(float(kl_from_logits.detach() - kl_exact.detach())) < tol


def test_05b_hssd_distiller_reduces_student_kl_over_five_steps():
    # Track A Teacher: Toy model with randomized GDN-2 adapters (frozen)
    teacher = build_toy(seed=10, num_hidden_layers=8, hidden_size=32, vocab_size=97)
    gdn2_mgr = install_gdn2(teacher, GDN2Config(rank=16, chunk_size=8))
    gen = torch.Generator().manual_seed(2026)
    with torch.no_grad():
        for p in gdn2_mgr.trainable_parameters():
            p.add_(torch.randn(p.shape, generator=gen, dtype=p.dtype) * 0.6)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    # Track B Student: Same base weights retrofitted with HSSD v3 + 2-pass Loop
    student = build_toy(seed=10, num_hidden_layers=8, hidden_size=32, vocab_size=97)
    for p in student.parameters():
        p.requires_grad_(False)
    frozen_student_params = [p for p in student.parameters()]
    hash_frozen_before = _sha256_tensors(frozen_student_params)

    handle = apply_hssd(student, HSSDConfig(rank=16, chunk_size=8, window_chunks=2, budget_weight=0.5))
    loop_spec = {"prelude_end": 2, "core_end": 6, "num_passes": 2, "target_virtual_layers": 12, "per_pass_router": True}
    ctl = apply_loop(student, loop_spec)
    student.train()

    distiller = HSSDDistiller(
        teacher,
        student,
        handle=handle,
        chunk_size=16,  # splits vocab=97 across 7 chunks without materializing full logits
        temperature=1.5,
        hidden_mse_weight=0.1,
        aux_loss_weight=0.2,
        ce_loss_weight=0.05,
    )

    ids = torch.randint(0, 97, (4, 24), generator=gen)
    trainable = handle.trainable_parameters() + ctl.trainable_parameters()
    opt = torch.optim.AdamW(trainable, lr=2e-2, weight_decay=0.0)

    kl_history: list[float] = []
    for _step in range(6):
        opt.zero_grad()
        out = distiller(ids, labels=ids)

        # Verify chunked-vocab KL inside distiller matches exact full-vocab KL (< 1e-6)
        with torch.no_grad():
            student.eval()
            exact_kl = float(
                exact_full_vocab_kl_loss(teacher(ids).logits, student(ids).logits, temperature=1.5).item()
            )
            student.train()
        assert abs(float(out.kl_loss.detach()) - exact_kl) < 1e-6
        kl_history.append(exact_kl)

        if len(kl_history) <= 5:
            out.loss.backward()
            opt.step()

    # Student KL after 5 optimizer steps is strictly and substantially lower than step 0!
    assert kl_history[-1] < kl_history[0] * 0.75, f"Expected KL reduction over 5 steps, got {kl_history}"
    assert _sha256_tensors(frozen_student_params) == hash_frozen_before, "Frozen base/MoE weights mutated!"


def test_06_stage_yaml_config_schema_compatibility():
    from qloop.train.distill import HSSDDistillConfig

    hcfg = HSSDConfig.from_dict(
        {
            "enabled": True,
            "chunk_size": 64,
            "window_chunks": 8,
            "num_pages": 8,
            "k_pages": 2,
            "budget_weight": 1.0,
            "segment_chunks": 16,
            "checkpoint_segments": True,
        }
    )
    hcfg.validate()
    assert hcfg.k_pages == 2 and hcfg.top_k_pages == 2

    dcfg = HSSDDistillConfig.from_dict(
        {
            "enabled": True,
            "teacher_ckpt": "runs/s1_loop_a6b",
            "teacher_track": "gdn2",
            "temperature": 1.0,
            "kl_weight": 1.0,
            "hidden_mse_weight": 0.1,
            "vocab_chunk_size": 4096,
        }
    )
    dcfg.validate()
    assert dcfg.chunk_size == 4096 and dcfg.vocab_chunk_size == 4096


def test_07_loop_checkpoint_and_distiller_backward_grad_parity_fp64():
    """Verify nested Loop checkpointing + HSSD segment checkpointing + MoR + HSSDDistiller backward."""
    teacher = build_toy(seed=17, num_hidden_layers=12, hidden_size=32, vocab_size=97).double()
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    base_student = build_toy(seed=23, num_hidden_layers=12, hidden_size=32, vocab_size=97).double()
    ids = torch.randint(0, 97, (2, 24), generator=torch.Generator().manual_seed(77))
    loop_base = {
        "prelude_end": 4,
        "core_end": 8,
        "num_passes": 2,
        "target_virtual_layers": 16,
        "mor": True,
        "write_mask": True,
        "mor_target_depth": 1.5,
        "per_pass_router": True,
        "router_diversity_weight": 0.1,
    }

    grads_by_ckpt: list[dict[str, torch.Tensor]] = []
    losses_by_ckpt: list[float] = []
    for ckpt_loop, ckpt_hssd in ((False, False), (True, False), (True, True)):
        torch.manual_seed(4242)
        student = copy.deepcopy(base_student)
        handle = apply_hssd(
            student,
            HSSDConfig(rank=16, chunk_size=8, window_chunks=2, segment_chunks=1, checkpoint_segments=ckpt_hssd),
        )
        ctl = apply_loop(student, dict(loop_base, checkpoint=ckpt_loop))
        _perturb_hssd_and_loop(handle, ctl, mor=True, seed=99)
        student.train()

        distiller = HSSDDistiller(
            teacher,
            student,
            handle=handle,
            chunk_size=16,
            temperature=1.5,
            hidden_mse_weight=0.1,
            aux_loss_weight=0.5,
            ce_loss_weight=0.1,
        )
        out = distiller(ids, labels=ids)
        # Ensure MoR actually mixed exited and continuing tokens during checkpointed forward/backward
        keep = ctl.ctx.keep[2]
        assert 0 < keep.sum() < keep.numel()
        out.loss.backward()

        losses_by_ckpt.append(float(out.loss.detach()))
        grads_by_ckpt.append({n: p.grad.detach().clone() for n, p in student.named_parameters() if p.grad is not None})

    for idx in (1, 2):
        assert abs(losses_by_ckpt[0] - losses_by_ckpt[idx]) < 1e-12
        assert grads_by_ckpt[0].keys() == grads_by_ckpt[idx].keys() and len(grads_by_ckpt[0]) > 20
        for name in grads_by_ckpt[0]:
            g0, g_ck = grads_by_ckpt[0][name], grads_by_ckpt[idx][name]
            assert torch.isfinite(g_ck).all(), f"Non-finite gradient in {name} (mode {idx})"
            assert float((g0 - g_ck).abs().max()) < 1e-11, f"Checkpoint grad mismatch on {name} (mode {idx})"

    # Also verify bfloat16 nested checkpoint + MoR + distiller backward runs cleanly
    torch.manual_seed(4242)
    bf16_teacher = build_toy(seed=17, num_hidden_layers=12, hidden_size=32, vocab_size=97).to(torch.bfloat16).eval()
    bf16_student = build_toy(seed=23, num_hidden_layers=12, hidden_size=32, vocab_size=97).to(torch.bfloat16)
    bf16_handle = apply_hssd(
        bf16_student, HSSDConfig(rank=16, chunk_size=8, window_chunks=2, segment_chunks=1, checkpoint_segments=True)
    )
    bf16_ctl = apply_loop(bf16_student, dict(loop_base, checkpoint=True))
    _perturb_hssd_and_loop(bf16_handle, bf16_ctl, mor=True, seed=99)
    bf16_student.train()
    bf16_out = HSSDDistiller(bf16_teacher, bf16_student, bf16_handle, chunk_size=16, hidden_mse_weight=0.1)(
        ids, labels=ids
    )
    bf16_out.loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in bf16_handle.trainable_parameters())


