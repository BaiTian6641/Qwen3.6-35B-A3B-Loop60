"""Verification suite for D2 (pre-head chunked CE) and D3 (sparse prefix streaming) in `long_calib.py`."""
from __future__ import annotations

import copy
from typing import Any

import pytest
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils._python_dispatch import TorchDispatchMode

from qloop.model.loop import apply_loop
from qloop.model.lsa import LSAConfig, install_lsa
from qloop.model.toy import ToyConfig, ToyForCausalLM
from qloop.train.long_calib import (
    CalibrationStepResult,
    LongCalibConfig,
    LongCalibStepResult,
    prefix_suffix_calibration_step,
)

SMALL_LSA = {
    "index_dim": 8,
    "block_size": 8,
    "pool": 2,
    "avg_bucket_keys": 4,
    "max_bucket_keys": 8,
    "window": 8,
    "sinks": 2,
    "num_select": 2,
    "query_chunk": 8,
    "distill_chunk": 4,
}


class _MaxVocabMatmulTracker(TorchDispatchMode):
    """Record maximum `[N, V]` activation rows materialized on vocabulary dimension `vocab_size`."""

    def __init__(self, vocab_size: int, excluded_storages: set[int]) -> None:
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.excluded_storages = excluded_storages
        self.max_vocab_rows: int = 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        out = func(*args, **(kwargs or {}))
        tensors = out if isinstance(out, (tuple, list)) else (out,)
        for t in tensors:
            if isinstance(t, torch.Tensor) and t.ndim >= 2 and t.shape[-1] == self.vocab_size:
                # Exclude `lm_head.weight.t()` / `embed_tokens.weight.t()` parameter storage views `[D, V]`
                if t.untyped_storage().data_ptr() in self.excluded_storages:
                    continue
                rows = int(t.numel() // self.vocab_size)
                self.max_vocab_rows = max(self.max_vocab_rows, rows)
        return out


# ============================================================================ D2: Pre-head hidden + chunked CE


@pytest.mark.parametrize("dtype,atol", [(torch.float32, 1e-5), (torch.float64, 1e-10)])
def test_d2_never_materializes_full_suffix_logits_and_matches_dense_ce(
    dtype: torch.dtype, atol: float
):
    """D2: `lm_head` full `[B, suffix_len, V]` is NEVER materialized; `suffix_logits is None`; CE & grads exact."""
    assert LongCalibStepResult is CalibrationStepResult

    torch.manual_seed(2026)
    vocab_size = 257  # distinct prime vocab size so `[rows, V]` tensors are uniquely identifiable
    t_cfg = ToyConfig(
        vocab_size=vocab_size,
        hidden_size=32,
        num_hidden_layers=4,
        full_attention_interval=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        partial_rotary_factor=0.5,
    )
    t_cfg_ref = copy.deepcopy(t_cfg)
    model_chunked = ToyForCausalLM(t_cfg).to(dtype).train()
    install_lsa(
        model_chunked,
        dict(SMALL_LSA, mode="distill", distill_weight=0.5, distill_query_frac=1.0, num_query_passes=2),
    )
    ctl_c = apply_loop(
        model_chunked,
        {"prelude_end": 0, "core_end": 4, "num_passes": 2, "target_virtual_layers": 8, "checkpoint": True},
    )
    with torch.no_grad():
        ctl_c.adapters.gates.fill_(0.25)

    # Clone identical reference model for dense CE comparison
    model_ref = ToyForCausalLM(t_cfg_ref).to(dtype).train()
    install_lsa(
        model_ref,
        dict(SMALL_LSA, mode="distill", distill_weight=0.5, distill_query_frac=1.0, num_query_passes=2),
    )
    apply_loop(
        model_ref,
        {"prelude_end": 0, "core_end": 4, "num_passes": 2, "target_virtual_layers": 8, "checkpoint": True},
    )
    model_ref.load_state_dict(model_chunked.state_dict())

    read_names = ("q_norm", "q_proj", "o_proj", "lsa_indexer.wq", "lsa_indexer.q_proj_extra", "loop_adapters")
    for m in (model_chunked, model_ref):
        for name, p in m.named_parameters():
            p.requires_grad_(any(tok in name for tok in read_names))

    ids = torch.randint(0, vocab_size, (1, 96), generator=torch.Generator().manual_seed(55))
    labels = ids.clone()
    # Mask several suffix positions with `ignore_index = -100` to verify exact alignment & masking
    labels[:, 32:40] = -100
    labels[:, -5:] = -100
    prefix_len = 32
    suffix_len = 64
    ce_chunk_size = 12  # Strictly smaller than `suffix_len = 64`!

    lm_head_forward_calls = {"n": 0}
    hook = model_chunked.lm_head.register_forward_hook(
        lambda _m, _i, _o: lm_head_forward_calls.__setitem__("n", lm_head_forward_calls["n"] + 1)
    )

    param_storages = {p.untyped_storage().data_ptr() for p in model_chunked.parameters()}
    try:
        torch.manual_seed(777)
        with _MaxVocabMatmulTracker(vocab_size, param_storages) as tracker:
            res_chunked = prefix_suffix_calibration_step(
                model_chunked,
                ids,
                prefix_len=prefix_len,
                chunk_size=16,
                labels=labels,
                ce_chunk_size=ce_chunk_size,
            )
    finally:
        hook.remove()

    # 1) `lm_head.forward` was NEVER called and `suffix_logits` is None!
    assert lm_head_forward_calls["n"] == 0
    assert res_chunked.suffix_logits is None
    assert res_chunked.prefix_len == prefix_len and res_chunked.suffix_len == suffix_len
    # 2) Peak `[rows, V]` tensor materialized across BOTH forward & backward is `<= ce_chunk_size` (`12 << 64`)!
    assert 0 < tracker.max_vocab_rows <= ce_chunk_size, (
        f"Expected max vocab rows <= {ce_chunk_size}, got {tracker.max_vocab_rows}"
    )

    # 3) Compare against explicit dense CE reference with identical RNG seed for sampled LSA KL
    torch.manual_seed(777)
    res_dense = prefix_suffix_calibration_step(
        model_ref,
        ids,
        prefix_len=prefix_len,
        chunk_size=16,
        labels=labels,
        loss_fn=lambda logits, tgt: F.cross_entropy(
            logits[:, :-1].reshape(-1, logits.shape[-1]),
            tgt[:, 1:].reshape(-1),
            ignore_index=-100,
        ),
    )
    assert abs(res_chunked.task_loss - res_dense.task_loss) < atol
    assert abs(float(res_chunked.loss) - float(res_dense.loss)) < atol

    # 4) Verify read-side adapter gradients match dense CE reference!
    ref_params = dict(model_ref.named_parameters())
    checked = 0
    for name, p in model_chunked.named_parameters():
        if p.requires_grad:
            assert p.grad is not None and ref_params[name].grad is not None
            torch.testing.assert_close(p.grad, ref_params[name].grad, rtol=atol * 10, atol=atol)
            checked += 1
    assert checked >= 6

    # 5) Verify `return_logits=True` on `LongCalibConfig` populates `suffix_logits` ONLY when requested
    cfg_with_logits = LongCalibConfig(return_logits=True)
    res_with_logits = prefix_suffix_calibration_step(
        model_chunked,
        ids,
        prefix_len=prefix_len,
        chunk_size=16,
        labels=labels,
        backward=False,
        cfg=cfg_with_logits,
    )
    assert res_with_logits.suffix_logits is not None
    assert res_with_logits.suffix_logits.shape == (1, suffix_len, vocab_size)


def test_d2_toy_bare_module_without_separate_lm_head_remains_compatible():
    """D2 compatibility: custom toy models without separate `.model` / `.lm_head` still work cleanly."""

    class _MinimalCausalToy(nn.Module):
        def __init__(self, vocab: int = 32, d: int = 16) -> None:
            super().__init__()
            self.emb = nn.Embedding(vocab, d)
            self.proj = nn.Linear(d, vocab, bias=False)

        def forward(self, input_ids: torch.Tensor, past_key_values=None, use_cache: bool = False):
            prev = 0 if past_key_values is None else int(past_key_values)
            h = self.emb(input_ids)
            logits = self.proj(h)
            nxt_cache = prev + int(input_ids.shape[1]) if use_cache else None
            return type("Out", (), {"logits": logits, "past_key_values": nxt_cache})()

    toy = _MinimalCausalToy()
    ids = torch.randint(0, 32, (1, 20), generator=torch.Generator().manual_seed(3))
    res = prefix_suffix_calibration_step(toy, ids, prefix_len=12, chunk_size=6)
    assert res.suffix_logits is None
    assert toy.proj.weight.grad is not None and float(toy.proj.weight.grad.norm()) > 0.0


# ============================================================================ D3: Sparse-mode prefix streaming


def test_d3_prefix_switches_lsa_to_sparse_and_restores_training_mode_on_suffix():
    """D3: `lsa_manager` switches to `prefix_lsa_mode` ('sparse') on prefix chunks & restores mode on suffix."""
    torch.manual_seed(91)
    t_cfg = ToyConfig(
        vocab_size=64,
        hidden_size=32,
        num_hidden_layers=2,
        full_attention_interval=2,  # Layer 1 is full_attention
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        partial_rotary_factor=0.5,
    )
    model = ToyForCausalLM(t_cfg).train()
    mgr = install_lsa(
        model,
        LSAConfig(
            **SMALL_LSA,
            mode="hybrid",
            recall_threshold=0.80,
            hybrid_sparse_every=2,
            distill_weight=1.0,
            distill_query_frac=1.0,
        ),
    )
    lc_cfg = LongCalibConfig.from_dict(
        {
            "enabled": True,
            "prefix_len": 32,
            "max_prefix_len": 64,
            "suffix_len": 16,
            "chunk_size": 8,
            "prefix_lsa_mode": "sparse",
            "calib_every_steps": 10,
            "sparse_prefix_after_step": 100,
        }
    )
    lc_cfg.validate()
    assert lc_cfg.prefix_lsa_mode == "sparse"
    assert lc_cfg.calib_every_steps == 10
    assert lc_cfg.sparse_prefix_after_step == 100

    ids = torch.randint(0, 64, (1, 48), generator=torch.Generator().manual_seed(12))
    attn = model.model.layers[1].self_attn
    seen_modes: list[tuple[bool, str]] = []
    hook = attn.register_forward_pre_hook(
        lambda _m, _i: seen_modes.append((torch.is_grad_enabled(), mgr.cfg.mode))
    )

    try:
        # 1) Early step (`global_step=10 < 100`, `recall_ready=False`): prefix keeps training mode (`"hybrid"`)
        seen_modes.clear()
        mgr.recall_ready = False
        mgr.indexers[1].recall_ready = False
        res_early = prefix_suffix_calibration_step(
            model,
            ids,
            prefix_len=32,
            chunk_size=8,
            lsa_manager=mgr,
            global_step=10,
            cfg=lc_cfg,
        )
        assert res_early.prefix_lsa_mode_used == "hybrid"
        assert [m for grad_on, m in seen_modes if not grad_on] == ["hybrid"] * 4
        assert [m for grad_on, m in seen_modes if grad_on] == ["hybrid"]
        assert mgr.cfg.mode == "hybrid"

        # 2) Step `>= sparse_prefix_after_step` (`global_step=100`): prefix runs `"sparse"`, suffix restores `"hybrid"`!
        seen_modes.clear()
        mgr.clear_cache()
        cluster_calls = {"n": 0}
        orig_cluster = mgr.indexers[1]._cluster_blocks

        def counting_cluster(z):
            cluster_calls["n"] += 1
            return orig_cluster(z)

        mgr.indexers[1]._cluster_blocks = counting_cluster  # type: ignore[method-assign]
        try:
            res_after = prefix_suffix_calibration_step(
                model,
                ids,
                prefix_len=32,
                chunk_size=8,
                lsa_manager=mgr,
                global_step=100,
                cfg=lc_cfg,
            )
        finally:
            mgr.indexers[1]._cluster_blocks = orig_cluster  # type: ignore[method-assign]

        assert res_after.prefix_lsa_mode_used == "sparse"
        assert [m for grad_on, m in seen_modes if not grad_on] == ["sparse"] * 4
        assert [m for grad_on, m in seen_modes if grad_on] == ["hybrid"]
        assert mgr.cfg.mode == "hybrid"
        # Incremental cache across chunks: each new 8-token chunk adds 1 block, never re-clustering old blocks!
        assert cluster_calls["n"] <= 5

        # 3) `recall_ready=True` at `global_step=5 < 100`: ALSO switches prefix to `"sparse"` & restores `"hybrid"`!
        seen_modes.clear()
        mgr.recall_ready = True
        mgr.indexers[1].recall_ready = True
        mgr.set_step(1)  # odd hybrid step on suffix
        res_ready = prefix_suffix_calibration_step(
            model,
            ids,
            prefix_len=32,
            chunk_size=8,
            lsa_manager=mgr,
            global_step=5,
            cfg=lc_cfg,
        )
        assert res_ready.prefix_lsa_mode_used == "sparse"
        assert [m for grad_on, m in seen_modes if not grad_on] == ["sparse"] * 4
        assert [m for grad_on, m in seen_modes if grad_on] == ["hybrid"]
        assert mgr.cfg.mode == "hybrid"
    finally:
        hook.remove()


def test_transformers_v5_dynamic_cache_fp8_quantize_and_checkpoint_safe_proxy():
    """Verify `_quantize_prefix_cache_fp8_` and `_ReplayKVCacheProxy` on `transformers 5.x` `DynamicCache`
    (`cache.layers[i].keys`/`values` + in-place `.copy_()` `LinearAttentionLayer` conv/recurrent states)."""
    from torch.utils.checkpoint import checkpoint

    from qloop.model.offload import FP8KVCacheQuantizer
    from qloop.train.bptt import _ReplayKVCacheProxy
    from qloop.train.long_calib import _quantize_prefix_cache_fp8_

    class _MockV5LinearLayer:
        def __init__(self) -> None:
            self.conv_states: list[torch.Tensor | None] = [None]
            self.recurrent_states: list[torch.Tensor | None] = [None]
            self.has_previous_state = [False]

        def get_conv_state(self, state_idx: int = 0) -> torch.Tensor | None:
            return self.conv_states[state_idx] if self.has_previous_state[state_idx] else None

        def update_conv_state(
            self, conv_states: torch.Tensor, state_idx: int = 0, conv_kernel_size: int = 4
        ) -> torch.Tensor:
            if not self.has_previous_state[state_idx]:
                self.conv_states[state_idx] = torch.zeros(
                    (*conv_states.shape[:-1], conv_kernel_size), dtype=conv_states.dtype
                )
                self.has_previous_state[state_idx] = True
                full = conv_states
            else:
                prev = self.conv_states[state_idx]
                assert prev is not None
                full = torch.cat([prev, conv_states], dim=-1)
            assert self.conv_states[state_idx] is not None
            # In-place `.copy_()` exactly matching `transformers 5.23.0` `LinearAttentionLayer`!
            self.conv_states[state_idx].copy_(full[..., -conv_kernel_size:])
            return full

        def get_recurrent_state(self, state_idx: int = 0) -> torch.Tensor | None:
            return self.recurrent_states[state_idx] if self.has_previous_state[state_idx] else None

        def update_recurrent_state(self, rec_state: torch.Tensor, state_idx: int = 0) -> torch.Tensor:
            if self.recurrent_states[state_idx] is None:
                self.recurrent_states[state_idx] = torch.zeros_like(rec_state)
            self.recurrent_states[state_idx].copy_(rec_state)
            return self.recurrent_states[state_idx]

    class _MockV5FullLayer:
        def __init__(self, k: torch.Tensor, v: torch.Tensor) -> None:
            self.keys = k
            self.values = v

        def update(self, k: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            self.keys = torch.cat([self.keys, k], dim=-2)
            self.values = torch.cat([self.values, v], dim=-2)
            return self.keys, self.values

    class _MockV5DynamicCache:
        def __init__(self, k: torch.Tensor, v: torch.Tensor) -> None:
            self.layers: list[Any] = [_MockV5LinearLayer(), _MockV5FullLayer(k, v)]

        def get_conv_state(self, layer_idx: int, state_idx: int = 0) -> torch.Tensor | None:
            return self.layers[layer_idx].get_conv_state(state_idx)

        def update_conv_state(
            self, conv_states: torch.Tensor, layer_idx: int, state_idx: int = 0, conv_kernel_size: int = 4
        ) -> torch.Tensor:
            return self.layers[layer_idx].update_conv_state(conv_states, state_idx, conv_kernel_size)

        def get_recurrent_state(self, layer_idx: int, state_idx: int = 0) -> torch.Tensor | None:
            return self.layers[layer_idx].get_recurrent_state(state_idx)

        def update_recurrent_state(
            self, rec_state: torch.Tensor, layer_idx: int, state_idx: int = 0
        ) -> torch.Tensor:
            return self.layers[layer_idx].update_recurrent_state(rec_state, state_idx)

        def update(
            self, key_states: torch.Tensor, value_states: torch.Tensor, layer_idx: int
        ) -> tuple[torch.Tensor, torch.Tensor]:
            return self.layers[layer_idx].update(key_states, value_states)

    torch.manual_seed(11)
    k0 = torch.randn(1, 2, 16, 8)
    v0 = torch.randn(1, 2, 16, 8)
    cache = _MockV5DynamicCache(k0.clone(), v0.clone())
    cache.update_conv_state(torch.randn(1, 4, 6), layer_idx=0, conv_kernel_size=4)
    rec0 = torch.randn(1, 4, 4)
    cache.update_recurrent_state(rec0.clone(), layer_idx=0)
    conv_before = cache.get_conv_state(0).clone()  # type: ignore[union-attr]

    # 1. `_quantize_prefix_cache_fp8_` operates on `cache.layers[i].keys/values` (not a no-op!)
    quant = FP8KVCacheQuantizer()
    _quantize_prefix_cache_fp8_(cache, quant)
    expected_k = quant.dequantize(quant.quantize(k0), dtype=k0.dtype)
    torch.testing.assert_close(cache.layers[1].keys, expected_k)
    assert not torch.equal(cache.layers[1].keys, k0)

    # 2. `_ReplayKVCacheProxy(cache, readonly_base=True)` stays checkpoint-safe across `use_reentrant=False`
    # even when `LinearAttentionLayer.update_*` mutates in-place via `.copy_()`, AND never mutates KV `base_cache`!
    proxy = _ReplayKVCacheProxy(cache, readonly_base=True)
    x = torch.randn(1, 4, 6, requires_grad=True)
    k_new = torch.randn(1, 2, 4, 8, requires_grad=True)
    v_new = torch.randn(1, 2, 4, 8, requires_grad=True)
    seen_rec_during_recomp: list[torch.Tensor] = []

    def _ckpt_block(x_in: torch.Tensor, kn: torch.Tensor, vn: torch.Tensor) -> torch.Tensor:
        r_prev = proxy.get_recurrent_state(0)
        assert r_prev is not None
        seen_rec_during_recomp.append(r_prev.detach().clone())
        full_conv = proxy.update_conv_state(x_in, 0, conv_kernel_size=4)
        proxy.update_recurrent_state(r_prev + 5.0, 0)
        kf, vf = proxy.update(kn, vn, 1)
        return full_conv.sum() + (kf * vf).sum()

    out = checkpoint(_ckpt_block, x, k_new, v_new, use_reentrant=False)
    out.backward()
    assert len(seen_rec_during_recomp) == 2
    # Both forward AND backward recomputation saw the exact pre-segment snapshot `rec0` (NOT mutated `rec0 + 5.0`)!
    torch.testing.assert_close(seen_rec_during_recomp[0], rec0)
    torch.testing.assert_close(seen_rec_during_recomp[1], rec0)
    # Gradients flow through live `x`, `k_new`, `v_new`, and `readonly_base=True` did NOT grow `cache.layers[1].keys`!
    assert x.grad is not None and torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0
    assert k_new.grad is not None and v_new.grad is not None
    assert cache.layers[1].keys.shape[-2] == 16
    assert conv_before.shape == (1, 4, 4)


def test_long_calib_clears_hssd_pending_aux_before_backward():
    """Unconsumed `hssd_adapter._pending_aux` must be cleared BEFORE `total_loss.backward()`
    so live `bud_loss` tensors cannot pin linear-layer checkpoint frames across backward."""
    from qloop.model.hssd import HSSDConfig, apply_hssd

    torch.manual_seed(29)
    t_cfg = ToyConfig(
        vocab_size=64,
        hidden_size=32,
        num_hidden_layers=2,
        full_attention_interval=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        partial_rotary_factor=0.5,
    )
    model = ToyForCausalLM(t_cfg).train()
    hssd_handle = apply_hssd(
        model,
        HSSDConfig(
            chunk_size=4,
            window_chunks=2,
            num_pages=4,
            k_pages=2,
            segment_chunks=2,
            checkpoint_segments=True,
            checkpoint_segments_min_len=4,
        ),
    )
    # Freeze HSSD adapters (mirrors `s3_long_1m_lsa` with `train_hssd: false`) while leaving other params live
    for ad in hssd_handle.adapters.values():
        for p in ad.parameters():
            p.requires_grad_(False)

    observed_pending_during_bwd: list[int] = []

    def _bwd_hook(_grad: torch.Tensor) -> None:
        observed_pending_during_bwd.append(sum(len(ad._pending_aux) for ad in hssd_handle.adapters.values()))

    hook_handle = model.lm_head.weight.register_hook(_bwd_hook)
    try:
        ids = torch.randint(0, 64, (1, 24), generator=torch.Generator().manual_seed(7))
        res = prefix_suffix_calibration_step(
            model,
            ids,
            prefix_len=8,
            chunk_size=4,
            backward=True,
        )
    finally:
        hook_handle.remove()

    assert observed_pending_during_bwd, "expected backward hook to fire on lm_head.weight"
    assert all(cnt == 0 for cnt in observed_pending_during_bwd), (
        f"expected 0 pending HSSD aux tensors during backward, saw {observed_pending_during_bwd}"
    )
    assert all(len(ad._pending_aux) == 0 for ad in hssd_handle.adapters.values())
    assert res.task_loss > 0.0


def test_bptt_chunked_replay_indexer_kl_exact_grad_parity():
    """Verify `_chunked_replay_indexer_kl` matches dense KL and `(dKL/dqz, dKL/dkz)` to 1e-10."""
    from qloop.train.bptt import _chunked_replay_indexer_kl

    torch.manual_seed(41)
    bsz, g, hg, t_seg, k_sel, d_nope, d_idx = 2, 2, 4, 17, 9, 12, 8
    tau = 0.25
    scale = 1.0 / (d_nope ** 0.5)
    qn_t = torch.randn(bsz, g, hg, t_seg, d_nope, dtype=torch.float64)
    kn_t = torch.randn(bsz, g, k_sel, d_nope, dtype=torch.float64)

    for per_head in (False, True):
        qz_shape = (bsz, g * hg, t_seg, d_idx) if per_head else (bsz, g, t_seg, d_idx)
        qz0 = torch.randn(qz_shape, dtype=torch.float64)
        kz0 = torch.randn(bsz, g, k_sel, d_idx, dtype=torch.float64)

        qz_ref = qz0.clone().requires_grad_(True)
        kz_ref = kz0.clone().requires_grad_(True)
        raw_sc = torch.einsum("bghcd,bgkd->bghck", qn_t, kn_t) * scale
        if per_head:
            tgt_p = torch.softmax(raw_sc, dim=-1)
            pred = torch.einsum("bghcd,bgkd->bghck", qz_ref.view(bsz, g, hg, t_seg, d_idx), kz_ref) / tau
        else:
            tgt_p = torch.softmax(raw_sc, dim=-1).mean(dim=2)
            pred = torch.einsum("bgcd,bgkd->bgck", qz_ref, kz_ref) / tau
        log_pred = torch.log_softmax(pred, dim=-1)
        kl_ref = (tgt_p * (torch.log(tgt_p.clamp_min(1e-30)) - log_pred)).sum(-1).mean()
        kl_ref.backward()

        qz_c = qz0.clone().requires_grad_(True)
        kz_c = kz0.clone().requires_grad_(True)
        kl_c = _chunked_replay_indexer_kl(
            qn_t,
            kn_t,
            qz_c,
            kz_c,
            scale_nope=scale,
            temperature=tau,
            per_head_q=per_head,
            chunk_size=5,
        )
        kl_c.backward()

        torch.testing.assert_close(kl_c, kl_ref, rtol=1e-10, atol=1e-10)
        assert qz_c.grad is not None and qz_ref.grad is not None
        assert kz_c.grad is not None and kz_ref.grad is not None
        torch.testing.assert_close(qz_c.grad, qz_ref.grad, rtol=1e-10, atol=1e-10)
        torch.testing.assert_close(kz_c.grad, kz_ref.grad, rtol=1e-10, atol=1e-10)


def test_long_calib_fp8_prefix_extraction_lean_parity_and_ctx_cleanup():
    """Verify chunked FP8 prefix KV extraction, `_ReplayKVCacheProxy.get_seq_length`,
    compact `SharedKVView` suffix-only stash, and post-calibration cleanup of loop/LSA caches."""
    from qloop.model.loop import LoopSpec, apply_loop
    from qloop.model.offload import FP8KVCacheQuantizer
    from qloop.train.bptt import _dequantize_fp8_kv_lean, _ReplayKVCacheProxy
    from qloop.train.long_calib import _extract_fp8_prefix_kv_, _quantize_fp8_tensor_lean

    # 1. Chunked `_quantize_fp8_tensor_lean` & `_dequantize_fp8_kv_lean` parity vs FP8KVCacheQuantizer
    torch.manual_seed(53)
    quant = FP8KVCacheQuantizer()
    x = torch.randn(1, 2, 100, 16, dtype=torch.float32) * 3.0
    q_ref = quant.quantize(x)
    q_lean = _quantize_fp8_tensor_lean(x, quant, chunk_tokens=32)
    assert q_lean.is_fp8_native == q_ref.is_fp8_native
    torch.testing.assert_close(q_lean.scale, q_ref.scale)
    torch.testing.assert_close(
        _dequantize_fp8_kv_lean(q_lean, torch.float32, chunk_tokens=32),
        quant.dequantize(q_ref, dtype=torch.float32),
    )

    # 2. `_extract_fp8_prefix_kv_` + `_ReplayKVCacheProxy.get_seq_length` & on-the-fly dequant
    class _V5Layer:
        def __init__(self, k: torch.Tensor, v: torch.Tensor) -> None:
            self.keys = k
            self.values = v

    class _V5Cache:
        def __init__(self, k: torch.Tensor, v: torch.Tensor) -> None:
            self.layers = [_V5Layer(k, v)]

        def update(self, k: torch.Tensor, v: torch.Tensor, idx: int):
            self.layers[idx].keys = torch.cat([self.layers[idx].keys, k], dim=-2)
            self.layers[idx].values = torch.cat([self.layers[idx].values, v], dim=-2)
            return self.layers[idx].keys, self.layers[idx].values

        def get_seq_length(self, layer_idx: int = 0) -> int:
            return int(self.layers[layer_idx].keys.shape[-2])

    k0 = torch.randn(1, 2, 40, 16, dtype=torch.float32)
    v0 = torch.randn(1, 2, 40, 16, dtype=torch.float32)
    c = _V5Cache(k0.clone(), v0.clone())
    extracted = _extract_fp8_prefix_kv_(c, quant)
    if quant.supports_fp8(k0.device):
        assert 0 in extracted and c.layers[0].keys.numel() == 0
        proxy = _ReplayKVCacheProxy(c, readonly_base=True, fp8_prefix_kv=extracted)
        assert proxy.get_seq_length() == 40
        assert proxy.get_seq_length(0) == 40
        ks, vs = torch.randn(1, 2, 8, 16), torch.randn(1, 2, 8, 16)
        k_out, v_out = proxy.update(ks, vs, 0)
        assert k_out.shape == (1, 2, 48, 16) and v_out.shape == (1, 2, 48, 16)
        torch.testing.assert_close(k_out[:, :, :40], quant.dequantize(quant.quantize(k0), dtype=torch.float32))
        torch.testing.assert_close(k_out[:, :, 40:], ks)

    # 3. End-to-end looped + LSA `prefix_suffix_calibration_step` clears `ctl.ctx.pkv_stash` and `lsa_ix._cache`!
    t_cfg = ToyConfig(
        vocab_size=64,
        hidden_size=32,
        num_hidden_layers=4,
        full_attention_interval=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        partial_rotary_factor=0.5,
    )
    model = ToyForCausalLM(t_cfg).train()
    mgr = install_lsa(model, LSAConfig(**SMALL_LSA, mode="hybrid", num_query_passes=2))
    ctl = apply_loop(
        model,
        LoopSpec(prelude_end=1, core_end=3, num_passes=2, target_virtual_layers=6, kv_share=True, checkpoint=True),
    )
    ids = torch.randint(0, 64, (1, 24), generator=torch.Generator().manual_seed(9))
    res = prefix_suffix_calibration_step(
        model, ids, prefix_len=12, chunk_size=6, fp8_kv_cache=True, lsa_manager=mgr, backward=True
    )
    assert res.task_loss > 0.0
    assert len(ctl.ctx.pkv_stash) == 0
    assert ctl.ctx.anchor is None
    assert all(len(ix._cache) == 0 for ix in mgr.indexers.values())



