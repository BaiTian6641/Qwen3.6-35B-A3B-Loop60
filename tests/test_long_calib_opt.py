"""Verification suite for D2 (pre-head chunked CE) and D3 (sparse prefix streaming) in `long_calib.py`."""
from __future__ import annotations

import copy

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
