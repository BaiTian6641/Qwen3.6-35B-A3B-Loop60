"""Sprint 1 Track B regression suite: LSA caller masks, unshared KV index cache,
mean-reduced KL + sparse recall telemetry, and deterministic router detection.

Covers:
  * QL60-007: Caller attention masks (left/right padding, 2D/3D/4D bool/int/additive float,
              2-doc block-diagonal isolation, strict sparse zero-leakage, NaN-free backward)
  * QL60-008: Pass-keyed BucketIndex cache when `kv_share=False` (full vs decode < 1e-12 in fp64)
              while preserving physical single-index reuse when `kv_share=True`
  * C2 & B2:  Mean-reduced `LSAManager.aux_loss()` (default `"mean"` vs `"sum"`) and live
              `lsa/last_recall` telemetry in `mode="sparse"` on the `recall_check_every` cadence
  * C3:       Deterministic install-time router output detection in `LoopController`
"""
from __future__ import annotations

import copy
from typing import Any

import pytest
import torch
import torch.nn.functional as F
from torch import nn

import qloop.model.loop as loop_mod
from qloop.model.loop import LoopSpec, _detect_router_returns_probs, apply_loop
from qloop.model.lsa import (
    _DENSE_FN,
    LSAConfig,
    LSAIndexer,
    install_lsa,
    lsa_attention,
)
from qloop.model.toy import build_toy, dense_causal_attention

SMALL_LSA: dict[str, Any] = {
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


def _masked_dense_ref(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    allow_4d: torch.Tensor,
    *,
    bias_4d: torch.Tensor | None = None,
    scaling: float | None = None,
) -> torch.Tensor:
    """Exact upcast masked dense attention reference returning `[B, Tq, H, D]`."""
    h, dh = q.shape[1], q.shape[3]
    g = k.shape[1]
    sc = (dh ** -0.5) if scaling is None else float(scaling)
    k_rep = k.repeat_interleave(h // g, dim=1)
    v_rep = v.repeat_interleave(h // g, dim=1)
    dt = torch.float64 if q.dtype == torch.float64 else torch.float32
    scores = torch.matmul(q.to(dt), k_rep.to(dt).transpose(-1, -2)) * sc
    if bias_4d is not None:
        scores = scores + torch.where(allow_4d, bias_4d.to(dt), torch.zeros_like(bias_4d, dtype=dt))
    has_any = allow_4d.any(dim=-1, keepdim=True)
    scores_safe = torch.where(
        allow_4d,
        scores,
        torch.where(has_any, scores.new_full((), float("-inf")), scores.new_zeros(())),
    )
    attn = torch.where(has_any, torch.softmax(scores_safe, dim=-1), torch.zeros_like(scores_safe))
    out = torch.matmul(attn, v_rep.to(dt)).to(v.dtype)
    return out.transpose(1, 2).contiguous()


def _make_lsa_mod(
    h: int,
    g: int,
    dh: int,
    *,
    num_select: int = 10_000,
    per_head_select: bool = False,
    strict_mask: bool = False,
    mode: str = "sparse",
    distill_weight: float = 0.0,
    detach_inputs: bool = True,
    seed: int = 17,
) -> tuple[nn.Module, LSAIndexer]:
    torch.manual_seed(seed)
    cfg = LSAConfig(
        mode=mode,
        index_dim=4,
        rotary_dim=4,
        block_size=8,
        pool=2,
        avg_bucket_keys=4,
        max_bucket_keys=8,
        window=8,
        sinks=2,
        num_select=num_select,
        query_chunk=7,
        per_head_select=per_head_select,
        strict_mask=strict_mask,
        distill_weight=distill_weight,
        distill_query_frac=1.0,
        detach_inputs=detach_inputs,
    )
    ix = LSAIndexer(h, g, dh, cfg, layer_id=0)
    mod = nn.Module()
    mod.lsa_indexer = ix
    return mod, ix


# ============================================================================ QL60-007: Caller Masks


def repro_padding_mask(
    *,
    pad_side: str = "left",
    mask_kind: str = "bool_4d",
    per_head_select: bool = False,
    dtype: torch.dtype = torch.float32,
) -> float:
    """Review 1 Appendix D padding-mask reproduction returning max valid-token error."""
    gen = torch.Generator().manual_seed(42)
    b, h, g, t, d = 2, 4, 2, 40, 16
    q = torch.randn(b, h, t, d, generator=gen, dtype=dtype)
    k = torch.randn(b, g, t, d, generator=gen, dtype=dtype)
    v = torch.randn(b, g, t, d, generator=gen, dtype=dtype)

    # Asymmetric per-batch padding lengths (batch 0 has 12 padded tokens, batch 1 has 6)
    pad_lens = [12, 6]
    key_valid = torch.ones((b, t), dtype=torch.bool)
    for bi, p in enumerate(pad_lens):
        if pad_side == "left":
            key_valid[bi, :p] = False
            k[bi, :, :p] = 10.0
            v[bi, :, :p] = 100.0
        else:
            key_valid[bi, t - p :] = False
            k[bi, :, t - p :] = 10.0
            v[bi, :, t - p :] = 100.0

    causal = torch.tril(torch.ones((t, t), dtype=torch.bool))
    allow_4d = (causal[None, None, :, :] & key_valid[:, None, None, :])  # [B, 1, T, T]

    if mask_kind == "bool_4d":
        caller_mask: torch.Tensor = allow_4d
    elif mask_kind == "float_inf_4d":
        caller_mask = torch.where(
            allow_4d,
            torch.zeros(b, 1, t, t, dtype=dtype),
            torch.full((b, 1, t, t), float("-inf"), dtype=dtype),
        )
    elif mask_kind == "float_finfo_4d":
        fmin = torch.finfo(dtype).min
        caller_mask = torch.where(
            allow_4d,
            torch.zeros(b, 1, t, t, dtype=dtype),
            torch.full((b, 1, t, t), fmin, dtype=dtype),
        )
    elif mask_kind == "bool_2d":
        caller_mask = key_valid
    elif mask_kind == "int_2d":
        caller_mask = key_valid.to(torch.long)
    elif mask_kind == "bool_3d":
        caller_mask = allow_4d[:, 0]
    else:
        raise ValueError(mask_kind)

    mod, _ix = _make_lsa_mod(h, g, d, num_select=10_000, per_head_select=per_head_select)
    if dtype == torch.float64:
        mod = mod.double()

    saved = _DENSE_FN[0]
    _DENSE_FN[0] = dense_causal_attention
    try:
        with torch.no_grad():
            out_lsa, _ = lsa_attention(mod, q, k, v, attention_mask=caller_mask)
            out_ref = _masked_dense_ref(q, k, v, allow_4d)
    finally:
        _DENSE_FN[0] = saved

    assert torch.isfinite(out_lsa).all()
    max_err = 0.0
    for bi in range(b):
        v_idx = key_valid[bi]
        err = float((out_lsa[bi, v_idx] - out_ref[bi, v_idx]).abs().max())
        max_err = max(max_err, err)
    return max_err


@pytest.mark.parametrize("pad_side", ["left", "right"])
@pytest.mark.parametrize(
    "mask_kind",
    ["bool_4d", "float_inf_4d", "float_finfo_4d", "bool_2d", "int_2d", "bool_3d"],
)
@pytest.mark.parametrize("per_head_select", [False, True])
def test_ql60_007_repro_padding_mask_matches_dense_under_1e6(
    pad_side: str, mask_kind: str, per_head_select: bool
) -> None:
    err = repro_padding_mask(pad_side=pad_side, mask_kind=mask_kind, per_head_select=per_head_select)
    assert err < 1e-6, f"pad_side={pad_side} mask_kind={mask_kind} per_head={per_head_select}: err={err}"


@pytest.mark.parametrize("mask_dtype", ["bool", "float_inf"])
@pytest.mark.parametrize("per_head_select", [False, True])
def test_ql60_007_two_doc_block_diagonal_isolation_and_strict_sparse_zero_leakage(
    mask_dtype: str, per_head_select: bool
) -> None:
    """2-doc packed block-diagonal mask isolates Doc 1 & Doc 2 in both all-bucket and strict-sparse modes."""
    gen = torch.Generator().manual_seed(123)
    b, h, g, t, d = 1, 4, 2, 48, 16
    split = 21  # intentionally mid-block (block_size=8 => block 2 spans [16..23] across Doc 1 & Doc 2!)
    q = torch.randn(b, h, t, d, generator=gen, dtype=torch.float64)
    k = torch.randn(b, g, t, d, generator=gen, dtype=torch.float64)
    v = torch.randn(b, g, t, d, generator=gen, dtype=torch.float64)

    doc_id = (torch.arange(t) >= split).long()
    causal = torch.tril(torch.ones((t, t), dtype=torch.bool))
    allow_4d = (causal & (doc_id[:, None] == doc_id[None, :]))[None, None]  # [1, 1, T, T]
    if mask_dtype == "bool":
        caller_mask: torch.Tensor = allow_4d
    else:
        caller_mask = torch.where(
            allow_4d,
            torch.zeros(1, 1, t, t, dtype=torch.float64),
            torch.full((1, 1, t, t), float("-inf"), dtype=torch.float64),
        )

    saved = _DENSE_FN[0]
    _DENSE_FN[0] = dense_causal_attention
    try:
        # 1) All-buckets exact dense parity (< 1e-12 in fp64, well below 1e-6)
        mod_all, _ = _make_lsa_mod(h, g, d, num_select=10_000, per_head_select=per_head_select)
        mod_all = mod_all.double()
        with torch.no_grad():
            out_all, _ = lsa_attention(mod_all, q, k, v, attention_mask=caller_mask)
            ref = _masked_dense_ref(q, k, v, allow_4d)
        assert float((out_all - ref).abs().max()) < 1e-12

        # 2) Strict sparse (num_select=2): Doc 1 perturbation must have EXACT 0.0 effect on Doc 2
        #    (and vice versa), even when split falls on a clean block boundary (`split_blk = 24`)!
        mod_sp, _ = _make_lsa_mod(h, g, d, num_select=2, per_head_select=per_head_select)
        mod_sp = mod_sp.double()
        split_blk = 24
        doc_blk = (torch.arange(t) >= split_blk).long()
        allow_blk = (causal & (doc_blk[:, None] == doc_blk[None, :]))[None, None]
        mask_blk = (
            allow_blk
            if mask_dtype == "bool"
            else torch.where(
                allow_blk,
                torch.zeros(1, 1, t, t, dtype=torch.float64),
                torch.full((1, 1, t, t), float("-inf"), dtype=torch.float64),
            )
        )
        k_pert1, v_pert1 = k.clone(), v.clone()
        k_pert1[:, :, :split_blk] = k_pert1[:, :, :split_blk] * 50.0 + 123.0
        v_pert1[:, :, :split_blk] = v_pert1[:, :, :split_blk] * 50.0 - 456.0
        k_pert2, v_pert2 = k.clone(), v.clone()
        k_pert2[:, :, split_blk:] = k_pert2[:, :, split_blk:] * 50.0 - 789.0
        v_pert2[:, :, split_blk:] = v_pert2[:, :, split_blk:] * 50.0 + 321.0
        with torch.no_grad():
            out_base, _ = lsa_attention(mod_sp, q, k, v, attention_mask=mask_blk)
            out_p1, _ = lsa_attention(mod_sp, q, k_pert1, v_pert1, attention_mask=mask_blk)
            out_p2, _ = lsa_attention(mod_sp, q, k_pert2, v_pert2, attention_mask=mask_blk)
        assert float((out_base[:, split_blk:] - out_p1[:, split_blk:]).abs().max()) == 0.0
        assert float((out_base[:, :split_blk] - out_p2[:, :split_blk]).abs().max()) == 0.0
    finally:
        _DENSE_FN[0] = saved


def test_ql60_007_strict_sparse_left_pad_invariance_finite_bias_decode_and_nan_free_backward() -> None:
    """Verify strict-sparse left-pad invariance, additive finite bias, decode mask, strict_mask, & grads."""
    gen = torch.Generator().manual_seed(77)
    b, h, g, t, d = 1, 4, 2, 40, 16
    pad = 10  # crosses into block 1 ([8..15]) so boundary block contains BOTH pad & valid keys!
    q = torch.randn(b, h, t, d, generator=gen, dtype=torch.float64, requires_grad=True)
    k = torch.randn(b, g, t, d, generator=gen, dtype=torch.float64, requires_grad=True)
    v = torch.randn(b, g, t, d, generator=gen, dtype=torch.float64, requires_grad=True)

    valid_k = torch.arange(t) >= pad
    causal = torch.tril(torch.ones((t, t), dtype=torch.bool))
    allow_4d = (causal & valid_k[None, :])[None, None]

    saved = _DENSE_FN[0]
    _DENSE_FN[0] = dense_causal_attention
    try:
        # 1) Strict sparse (`num_select=2`): perturbing padded tokens (including mixed boundary block 1)
        #    causes ZERO change on valid tokens because disallowed keys are zeroed in `build()` AND masked out!
        mod_sp, _ = _make_lsa_mod(h, g, d, num_select=2)
        mod_sp = mod_sp.double()
        k_alt = k.detach().clone()
        v_alt = v.detach().clone()
        k_alt[:, :, :pad] = 999.0
        v_alt[:, :, :pad] = -999.0
        with torch.no_grad():
            o1, _ = lsa_attention(mod_sp, q.detach(), k.detach(), v.detach(), attention_mask=allow_4d)
            o2, _ = lsa_attention(mod_sp, q.detach(), k_alt, v_alt, attention_mask=allow_4d)
        assert float((o1[:, pad:] - o2[:, pad:]).abs().max()) == 0.0

        # 2) Backward through all-masked padded query rows produces finite grads & exact 0 grad on padded KV
        o_grad, _ = lsa_attention(mod_sp, q, k, v, attention_mask=allow_4d)
        assert torch.isfinite(o_grad).all()
        assert torch.equal(o_grad[:, :pad], torch.zeros_like(o_grad[:, :pad]))
        o_grad.pow(2).sum().backward()
        assert q.grad is not None and torch.isfinite(q.grad).all()
        assert k.grad is not None and torch.isfinite(k.grad).all()
        assert v.grad is not None and torch.isfinite(v.grad).all()
        assert float(k.grad[:, :, :pad].abs().max()) == 0.0
        assert float(v.grad[:, :, :pad].abs().max()) == 0.0

        # 3) Additive float mask with finite bias + -inf disallowed + decode step (Tq=1)
        bias_vals = torch.randn(1, 1, t, t, generator=gen, dtype=torch.float64) * 0.25
        add_mask = torch.where(allow_4d, bias_vals, torch.full_like(bias_vals, float("-inf")))
        mod_all, _ = _make_lsa_mod(h, g, d, num_select=10_000)
        mod_all = mod_all.double()
        with torch.no_grad():
            o_bias, _ = lsa_attention(mod_all, q.detach(), k.detach(), v.detach(), attention_mask=add_mask)
            ref_bias = _masked_dense_ref(q.detach(), k.detach(), v.detach(), allow_4d, bias_4d=bias_vals)
            assert float((o_bias[:, pad:] - ref_bias[:, pad:]).abs().max()) < 1e-12

            # Single-token decode step with 2D & 4D mask
            q_dec = q.detach()[:, :, -1:]
            o_dec_2d, _ = lsa_attention(mod_all, q_dec, k.detach(), v.detach(), attention_mask=valid_k[None])
            o_dec_4d, _ = lsa_attention(mod_all, q_dec, k.detach(), v.detach(), attention_mask=allow_4d[:, :, -1:, :])
            ref_dec = _masked_dense_ref(q_dec, k.detach(), v.detach(), allow_4d[:, :, -1:, :])
            assert float((o_dec_2d - ref_dec).abs().max()) < 1e-12
            assert float((o_dec_4d - ref_dec).abs().max()) < 1e-12

        # 4) Masked KL distillation (`detach_inputs=True` vs `detach_inputs=False`) stays finite & matches
        mod_d1, ix_d1 = _make_lsa_mod(h, g, d, mode="distill", distill_weight=1.0, detach_inputs=True)
        mod_d2, ix_d2 = _make_lsa_mod(h, g, d, mode="distill", distill_weight=1.0, detach_inputs=False)
        mod_d1, mod_d2 = mod_d1.double(), mod_d2.double()
        ix_d2.load_state_dict(ix_d1.state_dict())
        torch.manual_seed(555)
        lsa_attention(mod_d1, q.detach(), k.detach(), v.detach(), attention_mask=allow_4d)
        kl1 = sum(ix_d1.pop_losses())
        assert isinstance(kl1, torch.Tensor) and torch.isfinite(kl1)
        kl1.backward()
        torch.manual_seed(555)
        lsa_attention(mod_d2, q.detach(), k.detach(), v.detach(), attention_mask=allow_4d)
        kl2 = sum(ix_d2.pop_losses())
        assert isinstance(kl2, torch.Tensor) and torch.isfinite(kl2)
        kl2.backward()
        torch.testing.assert_close(kl1.detach(), kl2.detach(), rtol=1e-10, atol=1e-10)
        assert ix_d1.wq.grad is not None and ix_d2.wq.grad is not None
        assert ix_d1.wk.grad is not None and ix_d2.wk.grad is not None
        torch.testing.assert_close(ix_d1.wq.grad, ix_d2.wq.grad, rtol=1e-9, atol=1e-9)
        torch.testing.assert_close(ix_d1.wk.grad, ix_d2.wk.grad, rtol=1e-9, atol=1e-9)

        # 5) `strict_mask=True` still raises RuntimeError
        mod_strict, _ = _make_lsa_mod(h, g, d, strict_mask=True)
        with pytest.raises(RuntimeError, match="strict_mask=true"):
            lsa_attention(
                mod_strict,
                q.detach().float(),
                k.detach().float(),
                v.detach().float(),
                attention_mask=allow_4d,
            )
    finally:
        _DENSE_FN[0] = saved


# ============================================================================ QL60-008: Unshared KV Cache


def repro_unshared_kv(*, kv_share: bool, num_passes: int = 2, search: str = "flat") -> tuple[float, int]:
    """Review 1 Appendix D `kv_share=True/False` full-vs-decode probe in float64."""
    model = build_toy(seed=11).double()
    ids = torch.randint(0, 97, (2, 42), generator=torch.Generator().manual_seed(5))
    mgr = install_lsa(
        model,
        dict(SMALL_LSA, mode="sparse", search=search, tree_group=2, tree_beam=1, num_select=2),
    )
    target_v = 16 + 8 * (num_passes - 1)
    ctl = apply_loop(
        model,
        {
            "prelude_end": 4,
            "core_end": 12,
            "num_passes": num_passes,
            "target_virtual_layers": target_v,
            "kv_share": kv_share,
        },
    )
    gen = torch.Generator().manual_seed(19)
    with torch.no_grad():
        ctl.adapters.gates.fill_(0.45)
        for p in ctl.trainable_parameters():
            p.add_(torch.randn(p.shape, generator=gen, dtype=p.dtype) * 0.3)

    # Count `_cluster_blocks` invocations on physical core full-attention layer 7 during one full forward
    ix7 = mgr.indexers[7]
    cluster_calls = [0]
    orig_cluster = ix7._cluster_blocks

    def _counting_cluster(z: torch.Tensor) -> Any:
        cluster_calls[0] += 1
        return orig_cluster(z)

    ix7._cluster_blocks = _counting_cluster  # type: ignore[method-assign]
    mgr.clear_cache()
    full = model(ids).logits
    n_clusters_full = cluster_calls[0]
    ix7._cluster_blocks = orig_cluster  # type: ignore[method-assign]

    mgr.clear_cache()
    out = model(ids[:, :10], use_cache=True)
    cache, steps = out.past_key_values, [out.logits]
    for step_t in range(10, ids.shape[1]):
        out = model(ids[:, step_t : step_t + 1], past_key_values=cache, use_cache=True)
        steps.append(out.logits)
    dec = torch.cat(steps, dim=1)
    max_err = float((full - dec).detach().abs().max())
    return max_err, n_clusters_full


@pytest.mark.parametrize("kv_share", [True, False])
@pytest.mark.parametrize("num_passes", [2, 3])
def test_ql60_008_repro_unshared_kv_fp64_under_1e12(kv_share: bool, num_passes: int) -> None:
    max_err, n_clusters_full = repro_unshared_kv(kv_share=kv_share, num_passes=num_passes, search="flat")
    assert max_err < 1e-12, f"kv_share={kv_share} num_passes={num_passes}: max_err={max_err}"
    if kv_share:
        # Physical single-index reuse across all passes (zero extra memory / clustering)
        assert n_clusters_full == 1
    else:
        # Independent BucketIndex per pass when K/V differs across passes
        assert n_clusters_full == num_passes


def test_ql60_008_unshared_kv_tree_search_and_checkpoint_parity() -> None:
    max_err, n_clusters_full = repro_unshared_kv(kv_share=False, num_passes=2, search="tree")
    assert max_err < 1e-12
    assert n_clusters_full == 2

    # Checkpointed vs uncheckpointed training parity with `kv_share=False`
    base_a = build_toy(seed=7, num_hidden_layers=8).double()
    base_b = copy.deepcopy(base_a)
    lsa_kw = dict(SMALL_LSA, mode="sparse", num_select=2, distill_weight=0.0)
    spec_a = LoopSpec(
        prelude_end=2, core_end=6, num_passes=2, target_virtual_layers=12, kv_share=False, checkpoint=True
    )
    spec_b = LoopSpec(
        prelude_end=2, core_end=6, num_passes=2, target_virtual_layers=12, kv_share=False, checkpoint=False
    )
    torch.manual_seed(303)
    install_lsa(base_a, lsa_kw)
    ctl_a = apply_loop(base_a, spec_a)
    torch.manual_seed(303)
    install_lsa(base_b, lsa_kw)
    ctl_b = apply_loop(base_b, spec_b)
    with torch.no_grad():
        ctl_a.adapters.gates.fill_(0.25)
        ctl_b.adapters.gates.fill_(0.25)
    base_a.train()
    base_b.train()
    ids = torch.randint(0, 64, (1, 32), generator=torch.Generator().manual_seed(9))
    la = base_a(ids, labels=ids).loss
    lb = base_b(ids, labels=ids).loss
    torch.testing.assert_close(la.detach(), lb.detach(), rtol=1e-10, atol=1e-10)
    la.backward()
    lb.backward()
    for (na, pa), (_nb, pb) in zip(ctl_a.adapters.named_parameters(), ctl_b.adapters.named_parameters()):
        if pa.grad is not None or pb.grad is not None:
            assert pa.grad is not None and pb.grad is not None, na
            torch.testing.assert_close(pa.grad, pb.grad, rtol=1e-9, atol=1e-9)


# ============================================================================ C2 & B2: aux_loss + sparse recall


def test_c2_aux_loss_mean_default_vs_sum_scaling_and_validation() -> None:
    assert LSAConfig().aux_reduction == "mean"
    LSAConfig(aux_reduction="mean").validate()
    LSAConfig(aux_reduction="sum").validate()
    with pytest.raises(ValueError, match="aux_reduction"):
        LSAConfig(aux_reduction="median").validate()

    # 16 physical layers -> 4 full-attention layers; 2-pass core loops 2 of them => 6 KL terms total
    base_mean = build_toy(seed=3).double()
    base_sum = copy.deepcopy(base_mean)
    ids = torch.randint(0, 97, (1, 40), generator=torch.Generator().manual_seed(11))

    torch.manual_seed(404)
    mgr_mean = install_lsa(base_mean, dict(SMALL_LSA, mode="distill", distill_query_frac=1.0, aux_reduction="mean"))
    ctl_mean = apply_loop(base_mean, {"prelude_end": 4, "core_end": 12, "num_passes": 2, "target_virtual_layers": 24})
    torch.manual_seed(404)
    mgr_sum = install_lsa(base_sum, dict(SMALL_LSA, mode="distill", distill_query_frac=1.0, aux_reduction="sum"))
    ctl_sum = apply_loop(base_sum, {"prelude_end": 4, "core_end": 12, "num_passes": 2, "target_virtual_layers": 24})
    with torch.no_grad():
        ctl_mean.adapters.gates.fill_(0.2)
        ctl_sum.adapters.gates.fill_(0.2)

    torch.manual_seed(808)
    base_mean(ids)
    n_terms = sum(len(ix.pending) for ix in mgr_mean.indexers.values())
    assert n_terms == 6
    loss_mean = mgr_mean.aux_loss()
    assert loss_mean is not None

    torch.manual_seed(808)
    base_sum(ids)
    loss_sum = mgr_sum.aux_loss()
    assert loss_sum is not None

    # Mean stays O(1) while sum scales by N = 6 (exact ratio N)
    torch.testing.assert_close(loss_sum.detach(), loss_mean.detach() * n_terms, rtol=1e-12, atol=1e-12)

    # Gradients scale by exact 1 / N
    loss_mean.backward()
    loss_sum.backward()
    for lid in mgr_mean.indexers:
        g_m = mgr_mean.indexers[lid].wq.grad
        g_s = mgr_sum.indexers[lid].wq.grad
        assert g_m is not None and g_s is not None
        torch.testing.assert_close(g_s, g_m * n_terms, rtol=1e-11, atol=1e-11)

    # Per-call override `aux_loss(reduction=...)` works cleanly
    torch.manual_seed(808)
    base_mean(ids)
    override_sum = mgr_mean.aux_loss(reduction="sum")
    assert override_sum is not None
    torch.testing.assert_close(override_sum.detach(), loss_sum.detach(), rtol=1e-12, atol=1e-12)


def test_b2_live_last_recall_telemetry_in_sparse_mode() -> None:
    """`lsa/last_recall` updates on `recall_check_every` cadence in `mode='sparse'` (train+distill and eval)."""
    model = build_toy(seed=13).double()
    ids = torch.randint(0, 97, (1, 40), generator=torch.Generator().manual_seed(21))
    mgr = install_lsa(
        model,
        dict(SMALL_LSA, mode="sparse", num_select=2, distill_weight=1.0, recall_check_every=5),
    )

    assert mgr.last_recall == 0.0
    assert mgr.stats()["lsa/last_recall"] == 0.0

    # Step 0 (train + distill_weight > 0): immediately measures genuine non-zero recall
    model.train()
    mgr.set_step(0)
    model(ids)
    mgr.aux_loss()  # drain KL
    r0 = mgr.stats()["lsa/last_recall"]
    assert 0.0 < r0 <= 1.0
    assert all(ix._recall_checks == 1 for ix in mgr.indexers.values())

    # Steps 1..4: skipped due to `recall_check_every=5` cadence
    for s in (1, 2, 3, 4):
        mgr.set_step(s)
        model(ids)
        mgr.aux_loss()
    assert all(ix._recall_checks == 1 for ix in mgr.indexers.values())

    # Step 5: cadence triggers fresh recall update
    mgr.set_step(5)
    model(ids)
    mgr.aux_loss()
    assert all(ix._recall_checks == 2 for ix in mgr.indexers.values())
    assert 0.0 < mgr.stats()["lsa/last_recall"] <= 1.0

    # Eval mode (`model.eval()` + `no_grad`) STILL tracks live recall even with `distill_weight=0.0`!
    model_eval = build_toy(seed=13).double().eval()
    mgr_eval = install_lsa(
        model_eval,
        dict(SMALL_LSA, mode="sparse", num_select=10_000, distill_weight=0.0, recall_check_every=5),
    )
    with torch.no_grad():
        model_eval(ids)
    assert mgr_eval.stats()["lsa/last_recall"] == pytest.approx(1.0)
    assert all(ix._recall_checks == 1 for ix in mgr_eval.indexers.values())

    # Training with `distill_weight=0.0` and grad enabled skips recall sampling
    model_nodist = build_toy(seed=13).double().train()
    mgr_nodist = install_lsa(
        model_nodist,
        dict(SMALL_LSA, mode="sparse", num_select=2, distill_weight=0.0, recall_check_every=5),
    )
    model_nodist(ids)
    assert all(ix._recall_checks == 0 for ix in mgr_nodist.indexers.values())
    assert mgr_nodist.stats()["lsa/last_recall"] == 0.0


# ============================================================================ C3: Deterministic Router Detection


def test_c3_deterministic_install_time_router_detection(monkeypatch: pytest.MonkeyPatch) -> None:
    """Install-time class/attribute detection eliminates fragile first-call `_looks_like_probs` sniffing."""

    class Qwen3_5MoeTopKRouter(nn.Module):
        def __init__(self, linear: nn.Linear, top_k: int) -> None:
            super().__init__()
            self.weight = linear.weight
            self.top_k = top_k
            self.num_experts = linear.out_features
            self.adversarial_first_call = True

        def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            flat = x.reshape(-1, x.shape[-1])
            if self.adversarial_first_call:
                # Adversarial logits that are non-negative and sum to 1.0 — WOULD fool `_looks_like_probs`!
                logits = torch.full(
                    (flat.shape[0], self.num_experts),
                    1.0 / self.num_experts,
                    dtype=x.dtype,
                    device=x.device,
                )
            else:
                logits = F.linear(flat, self.weight)
            probs = F.softmax(logits.float(), dim=-1)
            top_w, top_idx = probs.topk(self.top_k, dim=-1)
            top_w = top_w / top_w.sum(-1, keepdim=True)
            return logits, top_w.to(x.dtype), top_idx

    class Qwen3MoeTopKRouter(nn.Module):
        pass

    class ExplicitProbWrapper(nn.Module):
        returns_probs = True

        def __init__(self, linear: nn.Linear, top_k: int) -> None:
            super().__init__()
            self.weight = linear.weight
            self.top_k = top_k

        def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            flat = x.reshape(-1, x.shape[-1])
            probs = F.softmax(F.linear(flat.float(), self.weight.float()), dim=-1)
            # Simulate bf16 sum drift (> 1e-3) that WOULD cause a false negative in `_looks_like_probs`!
            drifted_probs = (probs * 1.015).to(x.dtype)
            top_w, top_idx = probs.topk(self.top_k, dim=-1)
            top_w = top_w / top_w.sum(-1, keepdim=True)
            return drifted_probs, top_w.to(x.dtype), top_idx

    class CustomProbRouter(nn.Module):
        pass

    class UnknownMockTupleRouter(nn.Module):
        def __init__(self, linear: nn.Linear, top_k: int) -> None:
            super().__init__()
            self.weight = linear.weight
            self.top_k = top_k

        def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            flat = x.reshape(-1, x.shape[-1])
            probs = F.softmax(F.linear(flat, self.weight), dim=-1)
            top_w, top_idx = probs.topk(self.top_k, dim=-1)
            top_w = top_w / top_w.sum(-1, keepdim=True)
            return probs, top_w.to(x.dtype), top_idx

    # 1) Direct unit assertions on `_detect_router_returns_probs`
    lin = nn.Linear(32, 8, bias=False)
    assert _detect_router_returns_probs(lin) is False
    assert _detect_router_returns_probs(Qwen3_5MoeTopKRouter(lin, 2)) is False
    assert _detect_router_returns_probs(Qwen3MoeTopKRouter()) is False
    assert _detect_router_returns_probs(ExplicitProbWrapper(lin, 2)) is True
    assert _detect_router_returns_probs(CustomProbRouter()) is True

    flag_false = nn.Module()
    flag_false.return_probs = False  # type: ignore[attr-defined]
    assert _detect_router_returns_probs(flag_false) is False

    flag_kind_p = nn.Module()
    flag_kind_p.router_output_kind = "probs"  # type: ignore[attr-defined]
    assert _detect_router_returns_probs(flag_kind_p) is True

    flag_kind_l = nn.Module()
    flag_kind_l.router_output_kind = "logits"  # type: ignore[attr-defined]
    assert _detect_router_returns_probs(flag_kind_l) is False

    assert _detect_router_returns_probs(UnknownMockTupleRouter(lin, 2)) is None

    def _wrap_moe_with_router(model: nn.Module, make_router: Any) -> list[nn.Module]:
        routers: list[nn.Module] = []
        for lyr in model.model.layers:  # type: ignore[union-attr]
            orig_mlp = lyr.mlp
            rt = make_router(orig_mlp.gate, orig_mlp.top_k)
            routers.append(rt)

            class _TupleMoe(nn.Module):
                def __init__(self, mlp: nn.Module, gate_mod: nn.Module) -> None:
                    super().__init__()
                    self.gate = gate_mod
                    self.experts = mlp.experts
                    self.shared_expert = mlp.shared_expert
                    self.shared_expert_gate = mlp.shared_expert_gate

                def forward(self, x: torch.Tensor) -> torch.Tensor:
                    b_, t_, h_ = x.shape
                    flat = x.reshape(-1, h_)
                    _full, top_w, top_idx = self.gate(flat)
                    out = self.experts(flat, top_idx, top_w)
                    out = out + torch.sigmoid(self.shared_expert_gate(flat)) * self.shared_expert(flat)
                    return out.view(b_, t_, h_)

            lyr.mlp = _TupleMoe(orig_mlp, rt)
        return routers

    ids = torch.randint(0, 97, (2, 6), generator=torch.Generator().manual_seed(3))
    spec = {
        "prelude_end": 4,
        "core_end": 12,
        "num_passes": 2,
        "target_virtual_layers": 24,
        "per_pass_router": True,
        "router_diversity_weight": 0.1,
    }

    # 2) Poison `_looks_like_probs` so any runtime value-sniffing on known router types fails immediately!
    orig_sniff = loop_mod._looks_like_probs

    def _forbid_sniff(_t: torch.Tensor) -> bool:
        raise AssertionError("_looks_like_probs must NOT be called for deterministically detected routers")

    monkeypatch.setattr(loop_mod, "_looks_like_probs", _forbid_sniff)

    # 2a) Plain `nn.Linear` routers: pre-cached as `False` at install time
    m_lin = build_toy(seed=1)
    ctl_lin = apply_loop(m_lin, spec)
    assert all(ctl_lin._is_probs_cache[p] is False for p in range(4, 12))
    m_lin(ids)

    # 2b) HF `Qwen3_5MoeTopKRouter` with adversarial `[1/E, ..., 1/E]` first-call logits: STILL `False`!
    m_q35 = build_toy(seed=2).double()
    q35_routers = _wrap_moe_with_router(m_q35, Qwen3_5MoeTopKRouter)
    ctl_q35 = apply_loop(m_q35, spec)
    assert all(ctl_q35._is_probs_cache[p] is False for p in range(4, 12))
    m_q35(ids)  # adversarial first call succeeds without sniffing!
    for rt in q35_routers:
        rt.adversarial_first_call = False  # type: ignore[attr-defined]
    ctl_q35.set_router_repel(25.0)
    with torch.no_grad():
        ctl_q35.adapters.gates.fill_(0.3)
    out_q35 = m_q35(ids).logits
    assert torch.isfinite(out_q35).all()
    assert ctl_q35.stats()["loop/reroute_rate"] == pytest.approx(1.0)

    # 2c) Probability-returning wrapper (`returns_probs=True`) with drifted sum: STILL `True` without sniffing!
    m_prob = build_toy(seed=3)
    _wrap_moe_with_router(m_prob, ExplicitProbWrapper)
    ctl_prob = apply_loop(m_prob, spec)
    assert all(ctl_prob._is_probs_cache[p] is True for p in range(4, 12))
    out_prob = m_prob(ids).logits
    assert torch.isfinite(out_prob).all()

    # 3) Unknown custom test mock falls back cleanly to `_looks_like_probs` on first call and caches it
    sniff_calls = [0]

    def _counting_sniff(t: torch.Tensor) -> bool:
        sniff_calls[0] += 1
        return orig_sniff(t)

    monkeypatch.setattr(loop_mod, "_looks_like_probs", _counting_sniff)
    m_unk = build_toy(seed=4).double()
    _wrap_moe_with_router(m_unk, UnknownMockTupleRouter)
    ctl_unk = apply_loop(m_unk, spec)
    assert ctl_unk._is_probs_cache == {}  # unknown mock not pre-classified
    m_unk(ids)
    assert all(ctl_unk._is_probs_cache[p] is True for p in range(4, 12))
    calls_after_first = sniff_calls[0]
    assert calls_after_first == 8  # once per core layer on first pass
    m_unk(ids)
    assert sniff_calls[0] == calls_after_first  # cached on subsequent calls
