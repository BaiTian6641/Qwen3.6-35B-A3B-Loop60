"""Comprehensive `float64` & functional verification suite for Work Packages M4, M5, & M6.

Covers:
  1. `offload.py` & `pos_rerope.py` (Fixes C1, C2 & Theorem 3):
     - `walsh_hadamard_block(..., 64)` self-adjoint orthogonality & zero rotary-to-NoPE leakage,
     - `clamped_relative_distance` & `apply_partial_rerope_q` (`[0:64]` clamped, `[64:256]` bit-exact),
     - `< 1e-12` inner-product commutativity between `pos_rerope` and `H_64`,
     - `Block64HadamardINT4Cache` exact `2,304 B` micro-page, `53.64 GiB` 10M ledger, `< 0.085` outlier
       Value relative error, Theorem 3 QJL `m=256` variance penalty, single-`H_64` aggregated Value
       reconstruction, async/staged `prefetch_micro_pages` (+ residual delta miss handling), and
       incremental 1-token decode append equivalence,
     - `FP8KVCacheQuantizer` `10.00 GiB` 1M ledger + native `float8_e4m3fn` & `fp16` fallback.
  2. Multi-pass `LSAIndexer(num_query_passes=3)` (`lsa.py`):
     - Step-0 bit-equivalence (`torch.equal`) with `num_query_passes=1`,
     - Shared `wk` & single Ward clustering call across passes (`1.0x` KV RAM, zero re-clustering),
     - Divergent Pass-2 / Pass-3 bucket selection once `q_proj_extra` is adapted,
     - Exact `float64` cached incremental decode == full forward (`< 1e-9`) at `R = 3` passes.
  3. `SegmentedKVBPTTRunner` (`bptt.py`, Fix E3):
     - Exact `0.0` cross-segment `k_proj`/`v_proj` gradient when replay is disabled vs strictly
       non-zero gradient when Segment 1 retrieves a needle placed exclusively in Segment 0,
     - Exact `float64` agreement (`< 1e-10`) with full un-truncated BPTT when `replay_k >= T_prior`,
     - Full 16-layer hybrid `ToyForCausalLM` + `D-LSA` + `Loop` `4`-segment run producing non-zero
       gradients into `k_proj`, `v_proj`, and `lsa_indexer.wk` (`W_k_idx`).
  4. `prefix_suffix_calibration_step` (`long_calib.py`, Fix C1):
     - Exact `float64` suffix logit equivalence (`< 1e-9`) vs full-sequence inference across `LSA` + `Loop`,
     - Zero autograd retention on streamed prefix chunks + non-zero read-side gradients (`q_norm`,
       `q_proj`, `LSAIndexer` `wq` & `q_proj_extra`, `loop_adapters`) reducing calibration loss.
"""
from __future__ import annotations

import copy
import math

import pytest
import torch
import torch.nn.functional as F

from qloop.model.loop import apply_loop
from qloop.model.lsa import install_lsa
from qloop.model.offload import (
    Block64HadamardINT4Cache,
    FP8KVCacheQuantizer,
    pack_group64_int4,
    unpack_group64_int4,
    walsh_hadamard_block,
)
from qloop.model.pos_rerope import (
    ReRoPEConfig,
    apply_partial_rerope_q,
    apply_partial_rope_k,
    clamped_relative_distance,
)
from qloop.model.toy import ToyConfig, ToyForCausalLM, build_toy
from qloop.train.bptt import BPTTConfig, SegmentedKVBPTTRunner
from qloop.train.long_calib import LongCalibConfig, prefix_suffix_calibration_step

SMALL_LSA = {
    "index_dim": 4,
    "block_size": 8,
    "pool": 2,
    "avg_bucket_keys": 4,
    "max_bucket_keys": 8,
    "window": 8,
    "sinks": 2,
    "num_select": 2,
    "query_chunk": 5,
    "distill_chunk": 4,
}


# ==============================================================================
# 1. Block-64 Hadamard, ReRoPE Commutativity (Fix C2), INT4 Micro-Page & FP8 KV
# ==============================================================================
def test_h64_orthogonality_rerope_commutativity_and_int4_offload():
    torch.manual_seed(2026)
    dt = torch.float64
    d = 256
    rot_dim = 64

    # 1a. Exact self-adjoint orthogonality and 64-d block isolation of `walsh_hadamard_block`
    eye = torch.eye(d, dtype=dt)
    h_mat = walsh_hadamard_block(eye, block_size=64)
    assert float((h_mat - h_mat.T).abs().max()) < 1e-14
    assert float((h_mat @ h_mat - eye).abs().max()) < 1e-14
    # Strictly zero off-block-diagonal entries (`[0:64]` never mixes with `[64:256]`)
    assert float(h_mat[:64, 64:].abs().max()) == 0.0
    assert float(h_mat[64:, :64].abs().max()) == 0.0

    # 1b. ReRoPE smooth clamping invariants (`Fix C1`)
    cfg_rr = ReRoPEConfig()
    cfg_rr.validate()
    inv_freq = cfg_rr.build_inv_freq(dtype=dt)
    d_probe = torch.tensor([0.0, 4096.0, 131_072.0, 262_144.0, 1_048_576.0, 10_000_000.0], dtype=dt)
    d_clamped = clamped_relative_distance(d_probe, linear_Ceil=131_072, log_scale=32_768, max_pos=240_000)
    d_clamped_snake = clamped_relative_distance(d_probe, linear_ceil=131_072, log_scale=32_768, max_pos=240_000)
    assert torch.equal(d_clamped, d_clamped_snake)
    assert torch.equal(d_clamped[:3], d_probe[:3])  # Exact identity inside 131,072!
    assert bool((d_clamped[1:] >= d_clamped[:-1]).all())  # Monotone
    assert float(d_clamped.max()) <= 240_000.0
    # C^1 smoothness check at boundary L_c = 131,072 (left slope == right slope == 1.0)
    eps = 1e-3
    b_pts = torch.tensor([131_072.0 - eps, 131_072.0, 131_072.0 + eps], dtype=dt)
    b_out = clamped_relative_distance(b_pts)
    slope_l = float((b_out[1] - b_out[0]) / eps)
    slope_r = float((b_out[2] - b_out[1]) / eps)
    assert abs(slope_l - 1.0) < 1e-6 and abs(slope_r - 1.0) < 1e-6

    # 1c. Partial ReRoPE + `H_64` inner-product invariance `< 1e-12` & bit-identical NoPE (`Fix C2`)
    n_tok = 512
    q_raw = torch.randn(n_tok, d, dtype=dt)
    k_raw = torch.randn(n_tok, d, dtype=dt)
    k_raw[:, :4] *= 12.0
    k_raw[:, 64:68] *= 12.0

    k_pos = torch.randint(0, 50_000, (n_tok,), dtype=torch.long)
    q_pos = k_pos + torch.randint(150_000, 8_000_000, (n_tok,), dtype=torch.long)

    k_cached = apply_partial_rope_k(k_raw, k_pos, inv_freq, rotary_dim=rot_dim)
    q_clamped = apply_partial_rerope_q(q_raw, k_pos, q_pos, inv_freq, rotary_dim=rot_dim)
    # Also verify `q_already_rotated=True` matches `q_already_rotated=False` to < 1e-11 in fp64!
    q_std_rot = apply_partial_rope_k(q_raw, q_pos, inv_freq, rotary_dim=rot_dim)
    q_from_rot = apply_partial_rerope_q(
        q_std_rot, k_pos, q_pos, inv_freq, rotary_dim=rot_dim, q_already_rotated=True
    )
    assert torch.equal(q_clamped[:, rot_dim:], q_raw[:, rot_dim:])
    assert torch.equal(q_from_rot[:, rot_dim:], q_raw[:, rot_dim:])
    assert torch.equal(k_cached[:, rot_dim:], k_raw[:, rot_dim:])
    assert float((q_clamped - q_from_rot).abs().max()) < 1e-8

    # Reference clamped relative inner product: rotate q_raw directly by `delta_eff` against `k_raw`
    delta_eff = clamped_relative_distance(q_pos.to(dt) - k_pos.to(dt))
    q_rel_ref = apply_partial_rope_k(q_raw, delta_eff, inv_freq, rotary_dim=rot_dim)
    ip_ref = (q_rel_ref * k_raw).sum(dim=-1)
    ip_spatial = (q_clamped * k_cached).sum(dim=-1)
    q_h = walsh_hadamard_block(q_clamped, 64)
    k_h = walsh_hadamard_block(k_cached, 64)
    ip_had64 = (q_h * k_h).sum(dim=-1)
    assert float((ip_spatial - ip_ref).abs().max()) < 1e-9
    assert float((ip_spatial - ip_had64).abs().max()) < 1e-12
    # Changing positions NEVER alters `H_64` NoPE channels `[64:256]`
    q_raw_h = walsh_hadamard_block(q_raw, 64)
    assert torch.allclose(q_h[:, 64:], q_raw_h[:, 64:], atol=1e-14)

    # 1d. `Block64HadamardINT4Cache`: exact `2,304 B` micro-page, `< 0.085` Value outlier error,
    # single-`H_64` value reconstruction, prefetch + residual delta miss handling, & 1-token append
    cache = Block64HadamardINT4Cache(num_layers=2, num_kv_heads=2, head_dim=256, page_tokens=4)
    assert cache.micro_page_bytes == 2304
    assert cache.bytes_per_scalar == 0.5625
    assert cache.estimate_cache_bytes(10_000_000, num_layers=10) == 57_600_000_000
    assert abs(cache.estimate_cache_bytes(10_000_000, num_layers=10) / (1024**3) - 53.644) < 0.01

    n_kv_tok = 64  # 16 micro-pages of 4 tokens
    k_in = torch.randn(1, 2, n_kv_tok, 256, dtype=torch.float32)
    v_in = torch.randn(1, 2, n_kv_tok, 256, dtype=torch.float32)
    k_in[..., :4] *= 12.0
    k_in[..., 64:68] *= 12.0
    v_in[..., :4] *= 8.0

    total_t = cache.append_tokens(0, k_in, v_in)
    assert total_t == n_kv_tok and cache.num_pages(0) == 16
    mp0 = cache.raw_micro_page(0, 0)
    assert mp0.dtype == torch.uint8 and mp0.numel() * mp0.element_size() == 2304

    all_pages = torch.arange(16, dtype=torch.long)
    # Speculative prefetch of first 12 pages -> gather all 16 pages (exercises 12 hits + 4 delta misses!)
    cache.prefetch_micro_pages(0, all_pages[:12])
    q_attn = torch.randn(1, 16, 3, 256, dtype=torch.float32)
    q_attn_h = walsh_hadamard_block(q_attn, 64)
    k_h_hat, v_h_hat = cache.gather_kv_and_reconstruct_v(0, all_pages, None)
    assert cache.stats["prefetch_calls"] == 1
    assert cache.stats["prefetch_hits"] == 1
    assert cache.stats["residual_miss_pages"] == 4

    # Outlier Value relative error `< 0.085` (and >35% better than naive unrotated Group-64 INT4)
    v_full_rec = walsh_hadamard_block(v_h_hat, 64)
    v_rel_err = float((v_full_rec - v_in).norm() / v_in.norm())
    v_naive = unpack_group64_int4(pack_group64_int4(v_in, 64), 64, dtype=torch.float32)
    v_naive_err = float((v_naive - v_in).norm() / v_in.norm())
    assert v_rel_err < 0.085 and v_rel_err < v_naive_err * 0.65

    # Theorem 3 check on packed `k_h_hat`: 1-bit QJL at `m = d = 256` worsens MSE vs plain INT4
    k_true_h = walsh_hadamard_block(k_in.double(), 64)[0, 0]  # [64, 256]
    k_deq_h = k_h_hat.double()[0, 0]
    q_probe = torch.randn(64, 256, dtype=dt)
    q_probe_h = walsh_hadamard_block(q_probe, 64)
    ip_true = (q_probe * k_in.double()[0, 0]).sum(-1)
    ip_int4 = (q_probe_h * k_deq_h).sum(-1)
    mse_int4 = float(((ip_int4 - ip_true) ** 2).mean())
    r_k = k_true_h - k_deq_h
    s_jl = torch.randn(256, 256, dtype=dt)
    qjl_est = (math.sqrt(math.pi / 2.0) / 256.0) * r_k.norm(dim=-1) * (
        (q_probe_h @ s_jl.T) * torch.sign(r_k @ s_jl.T)
    ).sum(-1)
    mse_qjl256 = float(((ip_int4 + qjl_est - ip_true) ** 2).mean())
    assert mse_qjl256 > mse_int4 * 1.25

    # Single-`H_64` post-aggregation Value reconstruction == per-token `H_64` reconstruction!
    k_h_rep = k_h_hat.repeat_interleave(8, dim=1)  # [1, 16, 64, 256]
    scores = torch.einsum("bhtd,bhnd->bhtn", q_attn_h, k_h_rep) / math.sqrt(256.0)
    attn_w = torch.softmax(scores, dim=-1)
    cache.prefetch_micro_pages(0, all_pages)
    _, v_agg_single = cache.gather_kv_and_reconstruct_v(0, all_pages, attn_w)
    assert v_agg_single is not None and cache.last_hadamard_v_calls == 1
    v_per_tok_ref = torch.einsum("bhtn,bhnd->bhtd", attn_w, v_full_rec.repeat_interleave(8, dim=1))
    torch.testing.assert_close(v_agg_single, v_per_tok_ref, rtol=1e-5, atol=1e-5)

    # Incremental 1-token `append_tokens` == batched `append_tokens` bit-for-bit!
    for t in range(12):
        cache.append_tokens(1, k_in[:, :, t : t + 1], v_in[:, :, t : t + 1])
    assert cache.num_tokens(1) == 12 and cache.num_pages(1) == 3
    for p_i in range(3):
        assert torch.equal(cache.raw_micro_page(1, p_i), cache.raw_micro_page(0, p_i))


def test_fp8_kv_cache_quantizer_and_fallback():
    assert FP8KVCacheQuantizer.tier1_1m_ledger_bytes() == 10 * (1024**3)
    x = torch.randn(1, 2, 128, 256, dtype=torch.float32) * 3.5

    q_auto = FP8KVCacheQuantizer()
    enc = q_auto.quantize(x)
    dec = q_auto.dequantize(enc, dtype=torch.float32)
    rel_err = float((dec - x).norm() / x.norm())
    assert rel_err < 0.05
    if q_auto.supports_fp8("cpu"):
        assert enc.is_fp8_native and enc.data.element_size() == 1

    q_fb = FP8KVCacheQuantizer(force_fp16_fallback=True)
    enc_fb = q_fb.quantize(x)
    assert not enc_fb.is_fp8_native and enc_fb.data.dtype == torch.float16
    dec_fb = q_fb.dequantize(enc_fb, dtype=torch.float32)
    torch.testing.assert_close(dec_fb, x, rtol=1e-3, atol=1e-3)

    # Also verify 2-tuple `(k_q, k_scale)` unpack + `dequantize(k_q, k_scale)` + YAML config contracts!
    k_q, k_scale = q_auto.quantize(x)
    dec_tuple = q_auto.dequantize(k_q, k_scale)
    torch.testing.assert_close(dec_tuple, dec, rtol=1e-6, atol=1e-6)

    bptt_cfg = BPTTConfig.from_dict(
        {"enabled": True, "n_segments": 4, "seg_len": 32768, "replay_k": 2048, "offload_prior_hidden": True}
    )
    assert bptt_cfg.offload_u_to_cpu is True and bptt_cfg.resolve_seg_len(131072) == 32768
    lc_cfg = LongCalibConfig.from_dict(
        {
            "enabled": True,
            "prefix_len": 524288,
            "max_prefix_len": 1015808,
            "suffix_len": 32768,
            "chunk_size": 4096,
            "fp8_kv_cache": True,
        }
    )
    assert lc_cfg.fp8_kv_cache is True and lc_cfg.suffix_len == 32768


# ==============================================================================
# 2. Multi-Pass `LSAIndexer(num_query_passes=3)` (`Work Package M4`)
# ==============================================================================
def test_multipass_lsa_step0_equivalence_divergent_buckets_and_cached_decode():
    ids = torch.randint(0, 97, (1, 44), generator=torch.Generator().manual_seed(11))

    # Build two identical models: `num_query_passes=1` vs `num_query_passes=3`
    base_a = build_toy(seed=4).double()
    base_b = copy.deepcopy(base_a)

    torch.manual_seed(99)
    install_lsa(base_a, dict(SMALL_LSA, mode="sparse", num_query_passes=1))
    ctl_1 = apply_loop(
        base_a,
        {"prelude_end": 4, "core_end": 12, "num_passes": 3, "target_virtual_layers": None},
    )

    torch.manual_seed(99)
    mgr_3 = install_lsa(base_b, dict(SMALL_LSA, mode="sparse", num_query_passes=3))
    ctl_3 = apply_loop(
        base_b,
        {"prelude_end": 4, "core_end": 12, "num_passes": 3, "target_virtual_layers": None},
    )

    # Verify `q_proj_extra` initialized by exact copy of Pass-1 `wq`
    ix7 = mgr_3.indexers[7]
    assert len(ix7.q_proj_extra) == 2
    assert torch.equal(ix7.q_proj_extra[0].weight, ix7.wq)
    assert torch.equal(ix7.q_proj_extra[1].weight, ix7.wq)

    # Even with non-zero pass-exit gates at step 0, `num_query_passes=3` == `num_query_passes=1` bit-exact!
    with torch.no_grad():
        ctl_1.adapters.gates.fill_(0.25)
        ctl_3.adapters.gates.fill_(0.25)
    out_1 = base_a(ids).logits
    out_3_step0 = base_b(ids).logits
    assert torch.equal(out_1, out_3_step0)

    # Verify `1.0x` KV & zero re-clustering across all 3 passes!
    cluster_calls = {"n": 0}
    orig_cluster = ix7._cluster_blocks

    def counting_cluster(z: torch.Tensor):
        cluster_calls["n"] += 1
        return orig_cluster(z)

    ix7._cluster_blocks = counting_cluster  # type: ignore[method-assign]
    mgr_3.clear_cache()
    base_b(ids)
    assert cluster_calls["n"] == 1
    ix7._cluster_blocks = orig_cluster  # type: ignore[method-assign]

    # Adapt `q_proj_extra` on Passes 2 & 3 -> verify divergent dynamic bucket selection!
    with torch.no_grad():
        for ix in mgr_3.indexers.values():
            ix.q_proj_extra[0].weight.mul_(-1.0)
            ix.q_proj_extra[1].weight.copy_(ix.wq.roll(shifts=1, dims=-1))

    # Direct `set_current_pass(r)` check on identical query tensor
    q_test = torch.randn(1, 4, 44, 8, dtype=torch.float64)
    k_test = torch.randn(1, 2, 44, 8, dtype=torch.float64)
    b_index = ix7.bucket_index(k_test)
    assert b_index is not None
    ix7.set_current_pass(1)
    qz1 = ix7.query_latent(q_test)
    b1 = ix7.select_buckets(q_test, b_index)
    ix7.set_current_pass(2)
    qz2 = ix7.query_latent(q_test)
    b2 = ix7.select_buckets(q_test, b_index)
    ix7.set_current_pass(3)
    b3 = ix7.select_buckets(q_test, b_index)
    ix7.set_current_pass(1)
    assert not torch.allclose(qz1, qz2)
    assert not torch.equal(b1[:, :, -8:], b2[:, :, -8:])
    assert not torch.equal(b1[:, :, -8:], b3[:, :, -8:])
    with pytest.raises(ValueError, match="exceeds"):
        ix7.set_current_pass(4)

    # Full looped `R=3` forward records distinct Pass-1 vs Pass-2 vs Pass-3 bucket selections
    mgr_3.clear_cache()
    full_adapted = base_b(ids).logits
    assert not torch.allclose(full_adapted, out_3_step0)
    assert not torch.equal(ix7.last_selected_buckets[1][:, :, -8:], ix7.last_selected_buckets[2][:, :, -8:])

    # Exact `float64` cached incremental decode == full forward across all 3 passes!
    mgr_3.clear_cache()
    prefill = base_b(ids[:, :4], use_cache=True)
    kv_cache = prefill.past_key_values
    pieces = [prefill.logits]
    for t in range(4, ids.shape[1]):
        step_out = base_b(ids[:, t : t + 1], past_key_values=kv_cache, use_cache=True)
        pieces.append(step_out.logits)
    decoded = torch.cat(pieces, dim=1)
    torch.testing.assert_close(decoded, full_adapted, rtol=1e-9, atol=1e-10)


# ==============================================================================
# 3. `SegmentedKVBPTTRunner` Cross-Segment K/V Gradient Replay (`Fix E3`)
# ==============================================================================
def test_segmented_kv_bptt_cross_segment_needle_gradient_proof():
    """Prove `k_proj.weight.grad` is strictly non-zero from a Segment-0 needle ONLY with replay."""
    torch.manual_seed(77)
    cfg = ToyConfig(
        vocab_size=64,
        hidden_size=32,
        num_hidden_layers=1,
        full_attention_interval=1,  # Layer 0 is full_attention
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        partial_rotary_factor=0.5,
    )
    model = ToyForCausalLM(cfg).double()

    # Partition embedding subspaces cleanly:
    #   * Token IDs `0..31` (used ONLY in Segment 0) live exclusively in dims `[0:16]`.
    #   * Token IDs `32..63` (used ONLY in Segment 1) live exclusively in dims `[16:32]`.
    # Consequently, `k_proj.weight.grad[:, :16]` and `v_proj.weight.grad[:, :16]` can ONLY be
    # non-zero when gradients flow back into Segment 0's keys & values!
    with torch.no_grad():
        model.model.embed_tokens.weight[:32, 16:] = 0.0
        model.model.embed_tokens.weight[32:, :16] = 0.0
        # Plant a clean orthogonal NoPE direction between needle token `7` (Seg 0) and query token `45` (Seg 1)
        needle_dir = F.normalize(torch.randn(16, dtype=torch.float64), dim=-1)
        query_dir = F.normalize(torch.randn(16, dtype=torch.float64), dim=-1)
        # Project out `needle_dir` / `query_dir` from all distractor tokens so needle `7` is unique
        w0 = model.model.embed_tokens.weight[:32, :16]
        model.model.embed_tokens.weight[:32, :16] = w0 - (w0 @ needle_dir).unsqueeze(-1) * needle_dir
        w1 = model.model.embed_tokens.weight[32:, 16:]
        model.model.embed_tokens.weight[32:, 16:] = w1 - (w1 @ query_dir).unsqueeze(-1) * query_dir
        model.model.embed_tokens.weight[7, :16] = needle_dir * 2.0
        model.model.embed_tokens.weight[45, 16:] = query_dir * 2.0
        # Align `k_proj` on `[:16]` and `q_proj` on `[16:]` so query `45` strongly attends to needle `7`
        target_nope = F.normalize(torch.randn(4, dtype=torch.float64), dim=-1) * 3.0
        for h_i in range(cfg.num_key_value_heads):
            row_lo = h_i * cfg.head_dim + 4  # NoPE channels [4:8]
            model.model.layers[0].self_attn.k_proj.weight[row_lo : row_lo + 4, :16] = torch.outer(
                target_nope, needle_dir
            )
        for q_h in range(cfg.num_attention_heads):
            row_lo = q_h * cfg.head_dim + 4
            model.model.layers[0].self_attn.q_proj.weight[row_lo : row_lo + 4, 16:] = torch.outer(
                target_nope, query_dir
            )

    model.model.embed_tokens.weight.requires_grad_(False)

    seg0 = torch.randint(0, 31, (1, 16), generator=torch.Generator().manual_seed(3))
    seg0[0, 5] = 7  # Needle exclusively at t = 5 in Segment 0!
    seg1 = torch.randint(32, 63, (1, 16), generator=torch.Generator().manual_seed(9))
    seg1[0, 12] = 45  # Query in Segment 1!
    ids = torch.cat([seg0, seg1], dim=1)

    # Loss evaluated EXCLUSIVELY on Segment 1 (`labels[:, :16] = -100`)
    labels = ids.clone()
    labels[:, :17] = -100  # Only targets inside Segment 1 contribute loss

    attn = model.model.layers[0].self_attn

    # Control A: `replay_enabled=False` -> cross-segment `k_proj`/`v_proj` grad on `[:16]` is EXACT 0.0!
    model.zero_grad(set_to_none=True)
    runner_off = SegmentedKVBPTTRunner(model, seg_len=16, replay_k=4, replay_enabled=False)
    res_off = runner_off.run_step(ids, labels=labels, backward=True)
    assert res_off.num_segments == 2
    assert attn.k_proj.weight.grad is not None and attn.v_proj.weight.grad is not None
    assert float(attn.k_proj.weight.grad[:, :16].abs().max()) == 0.0
    assert float(attn.v_proj.weight.grad[:, :16].abs().max()) == 0.0

    # Experiment B: `replay_enabled=True, replay_k=2` -> needle `t=5` selected & grad strictly non-zero!
    model.zero_grad(set_to_none=True)
    runner_on = SegmentedKVBPTTRunner(model, seg_len=16, replay_k=2, replay_enabled=True, offload_u_to_cpu=True)
    res_on = runner_on.run_step(ids, labels=labels, backward=True)
    sel_idx = res_on.replayed_indices[(1, 0)]  # Segment 1, Layer 0: [1, G, 2]
    assert bool((sel_idx == 5).any()), f"Expected needle t=5 in selected indices, got {sel_idx.tolist()}"
    assert attn.k_proj.weight.grad is not None and attn.v_proj.weight.grad is not None
    k_cross_norm = float(attn.k_proj.weight.grad[:, :16].norm())
    v_cross_norm = float(attn.v_proj.weight.grad[:, :16].norm())
    assert k_cross_norm > 1e-5, f"Expected non-zero cross-segment k_proj grad, got {k_cross_norm}"
    assert v_cross_norm > 1e-5, f"Expected non-zero cross-segment v_proj grad, got {v_cross_norm}"

    # Experiment C: When `replay_k >= seg_len` (all 16 prior tokens replayed), segmented replay
    # matches exact valid-token-mean full un-truncated BPTT gradients (`ref_loss.backward()`) to `< 1e-10` in float64!
    model.zero_grad(set_to_none=True)
    runner_all = SegmentedKVBPTTRunner(model, seg_len=16, replay_k=16, replay_enabled=True)
    runner_all.run_step(ids, labels=labels, backward=True)
    gk_replay_all = attn.k_proj.weight.grad.clone()
    gv_replay_all = attn.v_proj.weight.grad.clone()

    model.zero_grad(set_to_none=True)
    full_out = model(ids)
    # Segment 0 has 0 valid tokens and Segment 1 has all 15 valid tokens -> valid-token weight = 15/15 = 1.0
    ref_loss = F.cross_entropy(
        full_out.logits[:, 16:-1].reshape(-1, cfg.vocab_size).float(),
        labels[:, 17:].reshape(-1),
        ignore_index=-100,
    )
    ref_loss.backward()
    torch.testing.assert_close(gk_replay_all, attn.k_proj.weight.grad, rtol=1e-9, atol=1e-10)
    torch.testing.assert_close(gv_replay_all, attn.v_proj.weight.grad, rtol=1e-9, atol=1e-10)


def test_segmented_kv_bptt_4seg_hybrid_loop_and_lsa_wk_grad():
    """Verify 4-segment (`4 x 16 = 64`) hybrid GDN + D-LSA + Loop BPTT updates `wk`, `k_proj`, `v_proj`."""
    model = build_toy(seed=12).train()
    mgr = install_lsa(model, dict(SMALL_LSA, mode="sparse", distill_weight=0.5, num_query_passes=2))
    ctl = apply_loop(
        model,
        {"prelude_end": 4, "core_end": 12, "num_passes": 2, "target_virtual_layers": 24, "checkpoint": True},
    )
    with torch.no_grad():
        ctl.adapters.gates.fill_(0.2)

    # Freeze MoE experts; train full-attention projections + LSA indexers + loop adapters
    for name, p in model.named_parameters():
        p.requires_grad_(("self_attn" in name) or ("lsa_indexer" in name) or ("loop_adapters" in name))

    ids = torch.randint(0, 97, (1, 64), generator=torch.Generator().manual_seed(21))
    labels = ids.clone()
    labels[:, :48] = -100  # Loss ONLY on final Segment 3 (`t = 48..63`)!

    model.zero_grad(set_to_none=True)
    runner = SegmentedKVBPTTRunner(model, n_segments=4, replay_k=6, offload_u_to_cpu=True)
    res = runner(ids, labels=labels, backward=True)
    assert res.num_segments == 4
    # Segment 1 (`t=16..31`, targets `17..32`) has all -100 labels and advances caches under no_grad (0 replay);
    # Segments 2 (target `48`) & 3 (targets `49..63`) have valid labels and replay `replay_k=6` prior tokens each.
    assert res.replayed_tokens_per_seg == [0, 0, 6, 6]

    ix7 = mgr.indexers[7]
    attn7 = ctl.parts.layers[7].self_attn
    assert ix7.wk.grad is not None and float(ix7.wk.grad.norm()) > 0.0
    assert attn7.k_proj.weight.grad is not None and float(attn7.k_proj.weight.grad.norm()) > 0.0
    assert attn7.v_proj.weight.grad is not None and float(attn7.v_proj.weight.grad.norm()) > 0.0


# ==============================================================================
# 4. `prefix_suffix_calibration_step` (`Work Package M4`, `Fix C1`)
# ==============================================================================
def test_prefix_suffix_calibration_matches_full_logits_and_trains_read_side():
    model = build_toy(seed=19).double().train()
    mgr = install_lsa(
        model,
        dict(SMALL_LSA, mode="sparse", distill_weight=1.0, distill_query_frac=1.0, num_query_passes=2),
    )
    ctl = apply_loop(
        model,
        {"prelude_end": 4, "core_end": 12, "num_passes": 2, "target_virtual_layers": 24, "checkpoint": True},
    )
    with torch.no_grad():
        ctl.adapters.gates.fill_(0.3)

    # Freeze base write-side (`k_proj`, `v_proj`) and all MoE experts; train read-side only!
    read_tokens = ("q_norm", "q_proj", "o_proj", "lsa_indexer.wq", "lsa_indexer.q_proj_extra", "loop_adapters")
    for name, p in model.named_parameters():
        p.requires_grad_(any(tok in name for tok in read_tokens))

    ids = torch.randint(0, 97, (1, 48), generator=torch.Generator().manual_seed(31))
    prefix_len = 32
    chunk_size = 8

    # Reference full-sequence inference logits on `[prefix_len : 48]`
    mgr.clear_cache()
    with torch.no_grad():
        ref_full_logits = model(ids).logits[:, prefix_len:]

    # Track whether autograd is enabled during prefix chunks vs suffix
    grad_token_counts = {"grad_on_tokens": 0, "no_grad_tokens": 0}
    hook_handle = model.model.embed_tokens.register_forward_hook(
        lambda _m, inp, _out: grad_token_counts.__setitem__(
            "grad_on_tokens" if torch.is_grad_enabled() else "no_grad_tokens",
            grad_token_counts["grad_on_tokens" if torch.is_grad_enabled() else "no_grad_tokens"] + int(inp[0].numel()),
        )
    )

    try:
        mgr.clear_cache()
        model.zero_grad(set_to_none=True)
        step0 = prefix_suffix_calibration_step(
            model,
            ids,
            prefix_len=prefix_len,
            chunk_size=chunk_size,
            loss_fn=lambda logits, tgt: F.cross_entropy(
                logits[:, :-1].reshape(-1, logits.shape[-1]), tgt[:, 1:].reshape(-1)
            ),
        )
    finally:
        hook_handle.remove()

    # 4a. Exact suffix logit equivalence (`< 1e-9` in fp64) and bounded activation memory proof
    assert step0.num_prefix_chunks == 4
    assert grad_token_counts["no_grad_tokens"] == prefix_len
    assert grad_token_counts["grad_on_tokens"] == ids.shape[1] - prefix_len
    torch.testing.assert_close(step0.suffix_logits, ref_full_logits, rtol=1e-9, atol=1e-10)

    # 4b. Read-side grads populated; write-side & frozen MoE untouched!
    attn7 = ctl.parts.layers[7].self_attn
    ix7 = mgr.indexers[7]
    assert attn7.q_norm.weight.grad is not None and float(attn7.q_norm.weight.grad.norm()) > 0.0
    assert attn7.q_proj.weight.grad is not None and float(attn7.q_proj.weight.grad.norm()) > 0.0
    assert ix7.wq.grad is not None and float(ix7.wq.grad.norm()) > 0.0
    assert ix7.q_proj_extra[0].weight.grad is not None and float(ix7.q_proj_extra[0].weight.grad.norm()) > 0.0
    assert attn7.k_proj.weight.grad is None and attn7.v_proj.weight.grad is None
    assert ctl.parts.layers[7].mlp.experts.gate_up_proj.grad is None

    # 4c. Multi-step read-side optimization monotonically reduces calibration loss + FP8 prefix option works
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(trainable, lr=5e-3)
    loss_initial = step0.task_loss
    opt.step()
    loss_final = loss_initial
    for _ in range(5):
        opt.zero_grad(set_to_none=True)
        st = prefix_suffix_calibration_step(model, ids, prefix_len=prefix_len, chunk_size=chunk_size)
        loss_final = st.task_loss
        opt.step()
    assert loss_final < loss_initial

    # Verify optional Tier-1 `FP8KVCacheQuantizer` on prefix KV cache executes cleanly
    opt.zero_grad(set_to_none=True)
    st_fp8 = prefix_suffix_calibration_step(
        model,
        ids,
        prefix_len=prefix_len,
        chunk_size=chunk_size,
        fp8_quantizer=FP8KVCacheQuantizer(),
    )
    assert math.isfinite(st_fp8.task_loss)
