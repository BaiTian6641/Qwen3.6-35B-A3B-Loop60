"""Tests for Hybrid GDN-2 State-Snapshot Merge (K0 Arm e) and RL enable_thinking / DAPO-Math-17k prep."""
from __future__ import annotations

import math
import re
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from qloop.config import load_config
from qloop.debug import setup_run
from qloop.eval.kill_gate_k0 import (
    CANONICAL_ARMS,
    HFCheckpointK0Evaluator,
    K0Config,
    K0Example,
    _extract_linear_states,
    _merge_linear_states_into,
    build_arg_parser,
    canonicalize_arm_name,
    format_k0_markdown,
    run_k0_suite,
)
from qloop.eval.testing import ByteTokenizer
from qloop.infer.engine import QuantizedHybridCache
from qloop.memory.kv_blob import (
    KVBlobConfig,
    KVMemBlob,
    gdn2_state_bytes,
    pack_kv_blob,
    quantize_dequantize_gdn2_state,
)
from qloop.model.toy import ToyConfig, ToyForCausalLM, ToyHybridCache
from qloop.train.rl import GRPOTrainer, RLTask, _hf_tasks, load_rl_tasks, verify
from qloop.train.sirsi import SIRSIConfig
from qloop.train.stack import build_stack

ROOT = Path(__file__).resolve().parents[1]


# ==============================================================================
# 1. KVMemBlob GDN-2 State Quantization, Byte Accounting & Optional Fields
# ==============================================================================


def test_gdn2_state_quantize_dequantize_and_bytes() -> None:
    torch.manual_seed(7)
    # Two heads with drastically different scales to verify per-head symmetric int8 scaling
    state = torch.randn(1, 2, 16, 16, dtype=torch.float32)
    state[:, 0] *= 50.0
    state[:, 1] *= 0.01

    q_fp32 = quantize_dequantize_gdn2_state(state, mode="fp32")
    assert q_fp32.dtype == torch.float32
    assert torch.equal(q_fp32, state)

    q_bf16 = quantize_dequantize_gdn2_state(state, mode="bf16")
    assert q_bf16.dtype == torch.float32
    assert torch.equal(q_bf16, state.to(torch.bfloat16).to(torch.float32))
    assert (q_bf16 - state).abs().max().item() < 0.5

    q_int8 = quantize_dequantize_gdn2_state(state, mode="int8")
    assert q_int8.dtype == torch.float32
    assert torch.isfinite(q_int8).all()
    # Head 1 (scale ~0.01) must retain tight relative accuracy thanks to per-head scaling
    head1_rel_err = (q_int8[:, 1] - state[:, 1]).norm().item() / state[:, 1].norm().item()
    assert head1_rel_err < 0.02

    # All-zero state must clamp scale >= 1e-8 and stay exact zero without NaN/Inf
    zeros = torch.zeros(1, 2, 8, 8, dtype=torch.float32)
    q_zero = quantize_dequantize_gdn2_state(zeros, mode="int8")
    assert torch.isfinite(q_zero).all() and torch.equal(q_zero, zeros)

    # Byte accounting across modes
    states = {0: state, 1: torch.randn(1, 2, 16, 16)}
    b_fp32 = gdn2_state_bytes(states, mode="fp32")
    b_bf16 = gdn2_state_bytes(states, mode="bf16")
    b_int8 = gdn2_state_bytes(states, mode="int8")
    n_elems = 2 * state.numel()
    assert b_fp32 == n_elems * 4
    assert b_bf16 == n_elems * 2
    assert b_int8 == n_elems * 1 + (2 * 2) * 4  # 2 layers * 2 heads * float32 scale
    assert b_int8 < b_bf16 < b_fp32
    assert gdn2_state_bytes(None, mode="bf16") == 0
    assert gdn2_state_bytes({}, mode="int8") == 0

    with pytest.raises(ValueError, match="unknown gdn2 state quant mode"):
        quantize_dequantize_gdn2_state(state, mode="fp16")
    with pytest.raises(ValueError, match="unknown gdn2 state quant mode"):
        gdn2_state_bytes(states, mode="int4")

    # KVMemBlob optional GDN-2 fields default to None and accept explicit dicts
    bcfg = KVBlobConfig(num_global_layers=2, num_kv_heads=2, head_dim=64, rotary_dim=32, blob_tokens=8, sink_tokens=0)
    k = torch.randn(2, 2, 8, 64)
    v = torch.randn(2, 2, 8, 64)
    blob_default = pack_kv_blob(k, v, blob_id="b0", config=bcfg)
    assert blob_default.gdn2_recurrent_states is None
    assert blob_default.gdn2_conv_states is None

    blob_with_gdn2 = KVMemBlob(
        header=blob_default.header,
        packed_k=blob_default.packed_k,
        packed_v=blob_default.packed_v,
        rotary_dim=blob_default.rotary_dim,
        gdn2_recurrent_states={0: state},
        gdn2_conv_states={0: torch.randn(1, 4, 3)},
    )
    assert blob_with_gdn2.gdn2_recurrent_states is not None and 0 in blob_with_gdn2.gdn2_recurrent_states
    assert blob_with_gdn2.gdn2_conv_states is not None and 0 in blob_with_gdn2.gdn2_conv_states


# ==============================================================================
# 2. _extract_linear_states & _merge_linear_states_into (Toy, Quantized & HF 5.24)
# ==============================================================================


def test_merge_linear_states_toy_and_hf524_caches() -> None:
    torch.manual_seed(11)
    tcfg = ToyConfig(num_hidden_layers=4, full_attention_interval=4, linear_num_heads=2, linear_head_dim=8)
    sink = ToyHybridCache(tcfg)
    s0 = ToyHybridCache(tcfg)
    s1 = ToyHybridCache(tcfg)
    dst = ToyHybridCache(tcfg)

    for idx in (0, 1, 2):
        sink.recurrent_states[idx] = torch.randn(1, 2, 8, 8)
        s0.recurrent_states[idx] = torch.randn(1, 2, 8, 8)
        s1.recurrent_states[idx] = torch.randn(1, 2, 8, 8)
        sink.conv_states[idx] = torch.randn(1, 6, 2)
        s0.conv_states[idx] = torch.randn(1, 6, 2)
        s1.conv_states[idx] = torch.randn(1, 6, 2)

    decay = 0.7
    gdn2_bytes = _merge_linear_states_into(sink, [s0, s1], dst, decay=decay, quant="fp32")
    assert gdn2_bytes == 3 * (1 * 2 * 8 * 8) * 4

    for idx in (0, 1, 2):
        expected = (decay**2) * sink.recurrent_states[idx] + decay * s0.recurrent_states[idx] + s1.recurrent_states[idx]
        assert dst.recurrent_states[idx] is not None
        torch.testing.assert_close(dst.recurrent_states[idx], expected, rtol=1e-5, atol=1e-5)
        # Conv state comes from last active session (s1)
        assert dst.conv_states[idx] is not None
        assert torch.equal(dst.conv_states[idx], s1.conv_states[idx])
    assert dst.recurrent_states[3] is None and dst.conv_states[3] is None

    # Path 2a: HF 5.24 LinearAttentionLayer (list slots + metadata dicts + subsequent update_conv_state)
    class _TF524LinLayer:
        def __init__(self) -> None:
            self.conv_states: list[torch.Tensor | None] = [None]
            self.recurrent_states: list[torch.Tensor | None] = [None]
            self.conv_kernel_size: dict[int, int] = {}
            self.is_conv_states_initialized: dict[int, bool] = {0: False}
            self.is_recurrent_states_initialized: dict[int, bool] = {0: False}
            self.has_previous_state: dict[int, bool] = {0: False}
            self.batch_size: int | None = None

        def update_conv_state(self, conv_states: torch.Tensor, state_idx: int = 0) -> torch.Tensor:
            assert self.has_previous_state[state_idx] is True
            prev = self.conv_states[state_idx]
            assert prev is not None
            full_cs = torch.cat([prev, conv_states], dim=-1)
            self.conv_states[state_idx] = full_cs[..., -self.conv_kernel_size[state_idx] :]
            return full_cs

    class _TF524FullLayer:
        def __init__(self) -> None:
            self.keys: torch.Tensor | None = None
            self.values: torch.Tensor | None = None

    class _TF524PropViewCache:
        """Mirrors HF 5.24 caches exposing read-only `@property` list views over `.layers[i]`."""

        def __init__(self, layers: list[Any]) -> None:
            self.layers = layers

        @property
        def conv_states(self) -> list[torch.Tensor | None]:
            return [
                getattr(lyr, "conv_states", [None])[0] if hasattr(lyr, "conv_states") else None
                for lyr in self.layers
            ]

        @property
        def recurrent_states(self) -> list[torch.Tensor | None]:
            return [
                getattr(lyr, "recurrent_states", [None])[0] if hasattr(lyr, "recurrent_states") else None
                for lyr in self.layers
            ]

    def _make_524_cache(r_val: torch.Tensor | None, c_val: torch.Tensor | None) -> Any:
        lin = _TF524LinLayer()
        if r_val is not None:
            lin.recurrent_states[0] = r_val.clone()
            lin.is_recurrent_states_initialized[0] = True
            lin.has_previous_state[0] = True
        if c_val is not None:
            lin.conv_states[0] = c_val.clone()
            lin.is_conv_states_initialized[0] = True
            lin.has_previous_state[0] = True
            lin.conv_kernel_size[0] = int(c_val.shape[-1])
            lin.batch_size = int(c_val.shape[0])
        return _TF524PropViewCache([lin, _TF524FullLayer()])

    r_sink = torch.randn(1, 2, 8, 8, dtype=torch.bfloat16)
    r_sess = torch.randn(1, 2, 8, 8, dtype=torch.bfloat16)
    c_sink = torch.randn(1, 4, 3, dtype=torch.bfloat16)
    c_sess = torch.randn(1, 4, 3, dtype=torch.bfloat16)
    c524_sink = _make_524_cache(r_sink, c_sink)
    c524_sess = _make_524_cache(r_sess, c_sess)
    c524_dst = _make_524_cache(None, None)

    b_524 = _merge_linear_states_into(c524_sink, [c524_sess], c524_dst, decay=0.6, quant="int8")
    assert b_524 == gdn2_state_bytes({0: r_sess}, mode="int8")
    merged_r = c524_dst.layers[0].recurrent_states[0]
    assert merged_r is not None and merged_r.dtype == torch.bfloat16
    exp_524 = (0.6 * r_sink.float() + quantize_dequantize_gdn2_state(r_sess, mode="int8")).to(torch.bfloat16)
    torch.testing.assert_close(merged_r, exp_524, rtol=1e-2, atol=1e-2)
    assert torch.equal(c524_dst.layers[0].conv_states[0], c_sess)
    # Subsequent live update concatenates cleanly using preserved metadata
    full_cs = c524_dst.layers[0].update_conv_state(torch.randn(1, 4, 2, dtype=torch.bfloat16))
    assert full_cs.shape == (1, 4, 5)

    # Path 2b: Singular `.layers[i].{conv_state,recurrent_state}` tensor attrs
    def _make_singular_cache(r_val: torch.Tensor | None, c_val: torch.Tensor | None) -> Any:
        return SimpleNamespace(
            layers=[
                SimpleNamespace(conv_state=c_val, recurrent_state=r_val),
                SimpleNamespace(keys=None, values=None),
            ]
        )

    sc_sink = _make_singular_cache(r_sink, c_sink)
    sc_sess = _make_singular_cache(r_sess, c_sess)
    sc_dst = _make_singular_cache(torch.zeros_like(r_sink), torch.zeros_like(c_sink))
    b_sing = _merge_linear_states_into(sc_sink, [sc_sess], sc_dst, decay=0.5, quant="bf16")
    assert b_sing == gdn2_state_bytes({0: r_sess}, mode="bf16")
    conv_out, rec_out = _extract_linear_states(sc_dst)
    assert set(rec_out) == {0} and set(conv_out) == {0}
    assert rec_out[0].dtype == torch.bfloat16
    exp_sing = (0.5 * r_sink.float() + quantize_dequantize_gdn2_state(r_sess, mode="bf16")).to(torch.bfloat16)
    torch.testing.assert_close(rec_out[0], exp_sing, rtol=1e-2, atol=1e-2)
    assert torch.equal(conv_out[0], c_sess)

    # QuantizedHybridCache with cyclic `.layers` proxies must not recurse
    qh_sink = QuantizedHybridCache(num_layers=2, layer_types=["linear_attention", "full_attention"], mode="fp8")
    qh_sess = QuantizedHybridCache(num_layers=2, layer_types=["linear_attention", "full_attention"], mode="fp8")
    qh_dst = QuantizedHybridCache(num_layers=2, layer_types=["linear_attention", "full_attention"], mode="fp8")
    qh_sink.recurrent_states[0] = torch.randn(1, 2, 4, 4)
    qh_sess.recurrent_states[0] = torch.randn(1, 2, 4, 4)
    qh_sink.conv_states[0] = torch.randn(1, 4, 3)
    qh_sess.conv_states[0] = torch.randn(1, 4, 3)
    b_qh = _merge_linear_states_into(qh_sink, [qh_sess], qh_dst, decay=0.7, quant="bf16")
    assert b_qh == gdn2_state_bytes({0: qh_sess.recurrent_states[0]}, mode="bf16")
    assert qh_dst.recurrent_states[0] is not None


def test_merge_linear_states_tf524_dict_slots_inplace_copy_and_isolation() -> None:
    """Regression test for real `transformers >= 5.20 / 5.24.1` `LinearAttentionCacheLayerMixin`.

    Real TF 5.24.1 initializes `conv_states`, `recurrent_states`, and `conv_kernel_size` via
    `dict.fromkeys(range(1))` (`{0: None}` -> `{0: tensor}`) and updates buffers in-place with
    `self.recurrent_states[0].copy_(...)` / `self.conv_states[0].copy_(...)`.
    """
    torch.manual_seed(29)

    class _TF524DictLinearLayer:
        def __init__(self) -> None:
            self.conv_states: dict[int, torch.Tensor | None] = dict.fromkeys(range(1))
            self.recurrent_states: dict[int, torch.Tensor | None] = dict.fromkeys(range(1))
            self.conv_kernel_size: dict[int, int | None] = dict.fromkeys(range(1))
            self.is_conv_states_initialized: dict[int, bool] = {0: False}
            self.is_recurrent_states_initialized: dict[int, bool] = {0: False}
            self.has_previous_state: dict[int, bool] = {0: False}
            self.batch_size: int | None = None
            self.device: torch.device | None = None
            self.dtype: torch.dtype | None = None

        def update_conv_state(
            self, conv_states: torch.Tensor, state_idx: int = 0, conv_kernel_size: int | None = None
        ) -> torch.Tensor:
            if not self.is_conv_states_initialized[state_idx]:
                k = conv_states.shape[-1] if conv_kernel_size is None else int(conv_kernel_size)
                self.conv_kernel_size[state_idx] = k
                self.conv_states[state_idx] = torch.zeros(
                    (*conv_states.shape[:-1], k), dtype=conv_states.dtype, device=conv_states.device
                )
                self.is_conv_states_initialized[state_idx] = True
                self.batch_size = int(conv_states.shape[0])
                self.device = conv_states.device
                self.dtype = conv_states.dtype
            if not self.has_previous_state[state_idx]:
                full_cs = conv_states
                self.has_previous_state[state_idx] = True
            else:
                prev = self.conv_states[state_idx]
                assert prev is not None
                full_cs = torch.cat([prev, conv_states], dim=-1)
            k_size = self.conv_kernel_size[state_idx]
            assert k_size is not None
            dst_buf = self.conv_states[state_idx]
            assert isinstance(dst_buf, torch.Tensor)
            dst_buf.copy_(full_cs[..., -k_size:])
            return full_cs

        def update_recurrent_state(self, recurrent_states: torch.Tensor, state_idx: int = 0) -> torch.Tensor:
            if not self.is_recurrent_states_initialized[state_idx]:
                self.recurrent_states[state_idx] = torch.zeros_like(recurrent_states)
                self.is_recurrent_states_initialized[state_idx] = True
                self.device = recurrent_states.device
                self.dtype = recurrent_states.dtype
            dst_buf = self.recurrent_states[state_idx]
            assert isinstance(dst_buf, torch.Tensor)
            dst_buf.copy_(recurrent_states)
            return dst_buf

    class _TF524DictFullLayer:
        def __init__(self) -> None:
            self.keys: torch.Tensor | None = None
            self.values: torch.Tensor | None = None

    class _TF524DictDynamicCache:
        def __init__(self) -> None:
            self.layers: list[Any] = [_TF524DictLinearLayer(), _TF524DictFullLayer()]

    def _populate(r_val: torch.Tensor, c_val: torch.Tensor, *, keep_meta_none: bool = False) -> _TF524DictDynamicCache:
        cache = _TF524DictDynamicCache()
        lyr = cache.layers[0]
        if keep_meta_none:
            lyr.conv_states[0] = c_val.clone()
            lyr.recurrent_states[0] = r_val.clone()
        else:
            lyr.update_conv_state(c_val, conv_kernel_size=int(c_val.shape[-1]))
            lyr.update_recurrent_state(r_val)
        return cache

    r_sink = torch.full((1, 2, 4, 4), 10.0, dtype=torch.float32)
    r_s0 = torch.full((1, 2, 4, 4), 20.0, dtype=torch.float32)
    r_s1 = torch.full((1, 2, 4, 4), 30.0, dtype=torch.float32)
    c_sink = torch.full((1, 4, 3), 1.0, dtype=torch.float32)
    c_s0 = torch.full((1, 4, 3), 2.0, dtype=torch.float32)
    c_s1 = torch.full((1, 4, 3), 3.0, dtype=torch.float32)

    sink_cache = _populate(r_sink, c_sink)
    s0_cache = _populate(r_s0, c_s0)
    # Deliberately keep conv_kernel_size={0: None}, device=None, dtype=None on s1 to verify metadata backfill
    s1_cache = _populate(r_s1, c_s1, keep_meta_none=True)
    dst_cache = _TF524DictDynamicCache()

    decay = 0.7
    gdn2_bytes = _merge_linear_states_into(sink_cache, [s0_cache, s1_cache], dst_cache, decay=decay, quant="fp32")
    assert gdn2_bytes == r_s1.numel() * 4

    dst_lin = dst_cache.layers[0]
    assert isinstance(dst_lin.recurrent_states, dict)
    assert isinstance(dst_lin.conv_states, dict)
    merged_r = dst_lin.recurrent_states[0]
    merged_c = dst_lin.conv_states[0]
    assert isinstance(merged_r, torch.Tensor) and isinstance(merged_c, torch.Tensor)

    # Exact decay-weighted sum: 0.7^2 * 10 + 0.7^1 * 20 + 0.7^0 * 30 = 4.9 + 14.0 + 30.0 = 48.9
    expected_r = (decay**2) * r_sink + decay * r_s0 + r_s1
    torch.testing.assert_close(merged_r, expected_r, rtol=1e-5, atol=1e-5)
    assert torch.equal(merged_c, c_s1)
    assert dst_lin.conv_kernel_size == {0: 3}
    assert dst_lin.is_conv_states_initialized == {0: True}
    assert dst_lin.is_recurrent_states_initialized == {0: True}
    assert dst_lin.has_previous_state == {0: True}
    assert dst_lin.batch_size == 1
    assert dst_lin.device == r_s1.device
    assert dst_lin.dtype == r_s1.dtype

    # Subsequent in-place `.copy_()` updates on dst_cache MUST succeed and NEVER mutate source caches
    next_conv = torch.full((1, 4, 2), 9.0, dtype=torch.float32)
    full_cs = dst_lin.update_conv_state(next_conv)
    assert full_cs.shape == (1, 4, 5)
    assert torch.equal(full_cs[..., :3], c_s1) and torch.equal(full_cs[..., 3:], next_conv)

    next_rec = torch.full((1, 2, 4, 4), 99.0, dtype=torch.float32)
    updated_r = dst_lin.update_recurrent_state(next_rec)
    assert torch.equal(updated_r, next_rec)

    # Source caches remain 100% unmutated
    assert torch.equal(sink_cache.layers[0].recurrent_states[0], r_sink)
    assert torch.equal(s0_cache.layers[0].recurrent_states[0], r_s0)
    assert torch.equal(s1_cache.layers[0].recurrent_states[0], r_s1)
    assert torch.equal(sink_cache.layers[0].conv_states[0], c_sink)
    assert torch.equal(s0_cache.layers[0].conv_states[0], c_s0)
    assert torch.equal(s1_cache.layers[0].conv_states[0], c_s1)

    # Verify fresh dst_cache without prior sink metadata also backfills conv_kernel_size={0: None}, device, dtype
    dst_no_sink = _TF524DictDynamicCache()
    b_no_sink = _merge_linear_states_into(None, [s1_cache], dst_no_sink, decay=0.7, quant="bf16")
    assert b_no_sink == gdn2_state_bytes({0: r_s1}, mode="bf16")
    d_ns = dst_no_sink.layers[0]
    assert d_ns.conv_kernel_size == {0: 3}
    assert d_ns.device == r_s1.device
    assert d_ns.dtype == r_s1.dtype
    assert d_ns.batch_size == 1
    assert d_ns.has_previous_state == {0: True}
    d_ns.update_recurrent_state(torch.full_like(r_s1, 77.0))
    assert torch.equal(s1_cache.layers[0].recurrent_states[0], r_s1)


# ==============================================================================
# 3. End-to-End HFCheckpointK0Evaluator + run_k0_suite + Markdown + CLI Flags
# ==============================================================================


def test_k0_arm_e_end_to_end_and_backward_compat() -> None:
    torch.manual_seed(17)
    tok = ByteTokenizer()
    tcfg = ToyConfig(
        vocab_size=260,
        hidden_size=32,
        num_hidden_layers=8,
        full_attention_interval=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=128,
        partial_rotary_factor=0.5,
        rope_theta=10_000.0,
    )
    model = ToyForCausalLM(tcfg).eval()
    evaluator = HFCheckpointK0Evaluator(model, tok, device="cpu", live_sink_len=2)
    examples = [
        K0Example(
            example_id="gdn2_merge_ex1",
            sessions=[
                "Superseded session: PORT = 1111.",
                "Active session 1: PORT = 8443 in gateway.py.",
                "Active session 2: WORKER_THREADS = 16 in gateway.py.",
            ],
            live_prompt="Live prefix. What is PORT in gateway.py? ",
            answer="8443",
            category="multi_hop",
            superseded_session_indices=(0,),
        )
    ]

    # Backward-compatible default (eval_gdn2_state_merge=False): arm_e completely omitted
    cfg_off = K0Config(rotary_dim=64, rope_theta=10_000.0, sink_tokens=2, refresh_frac=0.10)
    assert cfg_off.eval_gdn2_state_merge is False
    assert cfg_off.gdn2_state_decay == pytest.approx(0.7)
    assert cfg_off.gdn2_state_quant == "bf16"
    rep_off = run_k0_suite(examples, evaluator, cfg_off)
    assert set(rep_off.arms) == set(CANONICAL_ARMS)
    assert "arm_e" not in rep_off.per_category["multi_hop"]
    md_off = format_k0_markdown(rep_off)
    assert "Arm (e)" not in md_off

    # Arm (e) enabled with bf16 and int8
    for quant_mode in ("bf16", "int8"):
        cfg_on = K0Config(
            rotary_dim=64,
            rope_theta=10_000.0,
            sink_tokens=2,
            refresh_frac=0.10,
            eval_gdn2_state_merge=True,
            gdn2_state_decay=0.7,
            gdn2_state_quant=quant_mode,
        )
        ep_map = evaluator(examples[0], cfg_on)
        assert "arm_e" in ep_map
        assert ep_map["arm_e"].extra["n_merged_sessions"] == 2  # superseded index 0 excluded
        gdn2_b = ep_map["arm_e"].extra["gdn2_state_bytes"]
        assert gdn2_b > 0
        assert ep_map["arm_e"].kv_bytes_mounted == ep_map["arm_c"].kv_bytes_mounted + gdn2_b

        rep_on = run_k0_suite(examples, evaluator, cfg_on)
        assert set(rep_on.arms) == {*CANONICAL_ARMS, "arm_e"}
        assert rep_on.get_arm("e") is rep_on.arms["arm_e"]
        assert canonicalize_arm_name("arm_e") == "arm_e"
        assert math.isfinite(rep_on.arms["arm_e"].answer_nll)
        assert math.isfinite(rep_on.arms["arm_e"].nll_gap)
        assert rep_on.arms["arm_e"].kv_bytes_mounted > rep_on.arms["arm_c"].kv_bytes_mounted
        assert "arm_e" in rep_on.per_category["multi_hop"]
        md_on = format_k0_markdown(rep_on)
        assert "Arm (e) Arm (c) + GDN-2 State Merge" in md_on
        assert "Arm (e) +GDN-2" in md_on
        assert f"`quant = {quant_mode}`" in md_on

    # CLI flags parse cleanly
    ap = build_arg_parser()
    args = ap.parse_args(
        ["--smoke-cpu", "--eval-gdn2-state-merge", "--gdn2-state-decay", "0.6", "--gdn2-state-quant", "int8"]
    )
    assert args.eval_gdn2_state_merge is True
    assert args.gdn2_state_decay == pytest.approx(0.6)
    assert args.gdn2_state_quant == "int8"

    # Config validation rejects bad decay/quant
    with pytest.raises(ValueError, match="gdn2_state_decay"):
        K0Config(gdn2_state_decay=1.5)
    with pytest.raises(ValueError, match="gdn2_state_quant"):
        K0Config(gdn2_state_quant="fp8")

    # End-to-end check on a live `build_stack(...)` Option-A-style stack (gdn2 + lsa, loop disabled)
    stack_cfg = {
        "seed": 0,
        "model": {
            "name": "toy",
            "dtype": "float32",
            "device_map": "cpu",
            "toy": {
                "seed": 0,
                "vocab_size": 260,
                "hidden_size": 32,
                "num_hidden_layers": 8,
                "full_attention_interval": 4,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "head_dim": 128,
                "partial_rotary_factor": 0.5,
            },
        },
        "loop": {"enabled": False},
        "gdn2": {"enabled": True, "rank": 4, "chunk_size": 8, "qed_enabled": False, "neg_eigval_max_boost": 0.0},
        "lsa": {
            "enabled": True,
            "mode": "sparse",
            "num_query_passes": 1,
            "index_dim": 8,
            "pool": 2,
            "avg_bucket_keys": 4,
            "max_bucket_keys": 8,
            "block_size": 16,
            "window": 16,
            "sinks": 2,
        },
        "train": {"trainable": [r"mlp\.gate\.weight$"]},
    }
    st = build_stack(stack_cfg)
    st.model.eval()
    st_eval = HFCheckpointK0Evaluator(st.model, st.tokenizer, device=st.device, live_sink_len=2)
    st_rep = run_k0_suite(
        examples,
        st_eval,
        K0Config(
            rotary_dim=64,
            rope_theta=10_000.0,
            sink_tokens=2,
            refresh_frac=0.10,
            eval_gdn2_state_merge=True,
            gdn2_state_decay=0.7,
            gdn2_state_quant="bf16",
        ),
    )
    assert set(st_rep.arms) == {*CANONICAL_ARMS, "arm_e"}
    assert math.isfinite(st_rep.arms["arm_e"].answer_nll)
    assert st_rep.arms["arm_e"].kv_bytes_mounted > st_rep.arms["arm_c"].kv_bytes_mounted


# ==============================================================================
# 4. GRPOTrainer rl.enable_thinking Forwarding & DAPO-Math-17k _hf_tasks Parsing
# ==============================================================================


def _toy_rl_cfg(*, enable_thinking: Any = "OMIT") -> dict[str, Any]:
    rl_cfg: dict[str, Any] = {
        "tasks": [{"source": "toy_arith", "n": 8, "verifier": "scorer:math"}],
        "group_size": 2,
        "prompts_per_step": 1,
        "max_new_tokens": 4,
        "temperature": 1.0,
        "micro_batch_seqs": 2,
        "gate": {
            "every": 0,
            "benchmarks": [
                {
                    "name": "aime_2026",
                    "limit": 1,
                    "max_new_tokens": 4,
                    "offline_items": [{"id": "q1", "problem": "Compute 2+2.", "answer": "4"}],
                }
            ],
        },
    }
    if enable_thinking != "OMIT":
        rl_cfg["enable_thinking"] = enable_thinking
    return {
        "seed": 0,
        "stage": "rl-think-test",
        "model": {"name": "toy", "dtype": "float32", "device_map": "cpu", "toy": {"seed": 0}},
        "loop": {"enabled": False},
        "train": {
            "steps": 1,
            "ckpt_every": 1,
            "trainable": [r"mlp\.gate\.weight$"],
            "optim": {"lr": 1e-3, "fused": False},
        },
        "rl": rl_cfg,
    }


def test_grpo_trainer_enable_thinking_forwarding_and_gate(tmp_path: Path) -> None:
    # 1. Default omitted -> chat_template_kwargs is empty (None forwarded)
    rd_default = str(tmp_path / "rl_default")
    setup_run(rd_default, level=1)
    tr_default = GRPOTrainer(_toy_rl_cfg(enable_thinking="OMIT"), rd_default)
    try:
        assert tr_default.gen.chat_template_kwargs == {}
    finally:
        tr_default.pool.shutdown(wait=False)

    # 2. enable_thinking: False -> chat_template_kwargs={"enable_thinking": False}
    rd_off = str(tmp_path / "rl_no_think")
    setup_run(rd_off, level=1)
    tr_off = GRPOTrainer(_toy_rl_cfg(enable_thinking=False), rd_off)
    try:
        assert tr_off.gen.chat_template_kwargs == {"enable_thinking": False}

        # Verify render_prompt & gate() forward enable_thinking=False into apply_chat_template
        seen_kwargs: list[dict[str, Any]] = []
        orig_apply = tr_off.stack.tokenizer.apply_chat_template

        def _spy_apply(messages: Any, **kwargs: Any) -> Any:
            seen_kwargs.append(dict(kwargs))
            return orig_apply(messages, **kwargs)

        tr_off.stack.tokenizer.apply_chat_template = _spy_apply
        rendered = tr_off.gen.render_prompt([{"role": "user", "content": "What is 2+2?"}])
        assert isinstance(rendered, str) and len(rendered) > 0
        assert seen_kwargs[-1].get("enable_thinking") is False

        seen_kwargs.clear()
        tr_off.gate()
        assert seen_kwargs and all(kw.get("enable_thinking") is False for kw in seen_kwargs)
    finally:
        tr_off.pool.shutdown(wait=False)

    # 3. enable_thinking: True -> chat_template_kwargs={"enable_thinking": True}
    rd_on = str(tmp_path / "rl_think")
    setup_run(rd_on, level=1)
    tr_on = GRPOTrainer(_toy_rl_cfg(enable_thinking=True), rd_on)
    try:
        assert tr_on.gen.chat_template_kwargs == {"enable_thinking": True}
    finally:
        tr_on.pool.shutdown(wait=False)


def test_hf_tasks_dapo_math_nested_prompt_and_ground_truth(monkeypatch: pytest.MonkeyPatch) -> None:
    class _FakeDataset:
        def __init__(self, rows: list[dict[str, Any]]) -> None:
            self._rows = list(rows)

        def shuffle(self, seed: int = 0) -> _FakeDataset:
            return self

        def select(self, indices: Any) -> _FakeDataset:
            return _FakeDataset([self._rows[i] for i in indices])

        def __len__(self) -> int:
            return len(self._rows)

        def __iter__(self) -> Any:
            return iter(self._rows)

    live_dapo_rows = [
        {
            "data_source": "math_dapo",
            "prompt": [{"role": "user", "content": "Solve 6 * 7. Put your answer in \\boxed{}."}],
            "ability": "MATH",
            "reward_model": {"style": "rule-lighteval/MATH_v2", "ground_truth": "42"},
            "extra_info": {"index": "dapo-0"},
        },
        {
            "id": "flat-1",
            "prompt": "Solve 10 + 5.",
            "solution": "15",
        },
    ]

    fake_datasets_mod = types.ModuleType("datasets")
    fake_datasets_mod.load_dataset = lambda *a, **kw: _FakeDataset(live_dapo_rows)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "datasets", fake_datasets_mod)

    # 1. Explicit dotted answer_field (`reward_model.ground_truth`) + fallback to `solution`
    src = {
        "source": "hf",
        "name": "dapo_math",
        "hf_id": "BytedTsinghua-SIA/DAPO-Math-17k",
        "split": "train",
        "prompt_field": "prompt",
        "answer_field": "reward_model.ground_truth",
        "verifier": "scorer:math",
        "weight": 1.0,
    }
    rows = _hf_tasks(src, seed=0)
    assert len(rows) == 2
    assert isinstance(rows[0]["prompt"], list)
    assert rows[0]["prompt"][0]["role"] == "user"
    assert "6 * 7" in rows[0]["prompt"][0]["content"]
    assert rows[0]["answer"] == "42"
    assert rows[1]["prompt"] == "Solve 10 + 5."
    assert rows[1]["answer"] == "15"

    # 2. Legacy answer_field="solution" still falls back cleanly to reward_model.ground_truth!
    src_legacy = {**src, "answer_field": "solution"}
    rows_legacy = _hf_tasks(src_legacy, seed=0)
    assert rows_legacy[0]["answer"] == "42"
    assert rows_legacy[1]["answer"] == "15"

    # 3. End-to-end load_rl_tasks + scorer:math verification
    groups = load_rl_tasks({"tasks": [src]}, seed=0)
    assert len(groups) == 1 and groups[0][0] == "dapo_math"
    tasks: list[RLTask] = groups[0][2]
    assert verify(tasks[0], "Therefore the answer is \\boxed{42}.")[0] == 1.0
    assert verify(tasks[0], "Therefore the answer is \\boxed{41}.")[0] == 0.0
    assert verify(tasks[1], "\\boxed{15}")[0] == 1.0


# ==============================================================================
# 5. Shipped s4_opt_a_d1_rl.yaml Tuning & Hygiene Verification
# ==============================================================================


def test_s4_opt_a_d1_rl_config_tuning_and_hygiene() -> None:
    cfg_path = ROOT / "configs" / "stages" / "s4_opt_a_d1_rl.yaml"
    raw = cfg_path.read_text(encoding="utf-8")
    assert not re.search(r"\bHBM\b", raw, flags=re.IGNORECASE), "forbidden HBM reference in s4_opt_a_d1_rl.yaml"
    assert "opus46" not in raw.lower(), "forbidden opus46 reference in s4_opt_a_d1_rl.yaml"

    cfg = load_config(str(cfg_path))
    assert cfg.stage == "s4_opt_a_d1_rl"
    assert cfg.stage_type == "rl"
    assert cfg.loop.enabled is False
    assert cfg.train.sha256_frozen_experts is True
    assert int(cfg.train.steps) == 40
    assert int(cfg.train.ckpt_every) == 10

    assert cfg.rl.enable_thinking is False
    assert int(cfg.rl.group_size) == 4
    assert int(cfg.rl.prompts_per_step) == 4
    assert int(cfg.rl.max_new_tokens) == 2048
    assert len(cfg.rl.tasks) == 1
    task0 = cfg.rl.tasks[0]
    assert task0["source"] == "hf"
    assert task0["name"] == "dapo_math"
    assert task0["hf_id"] == "BytedTsinghua-SIA/DAPO-Math-17k"
    assert task0["split"] == "train"
    assert task0["prompt_field"] == "prompt"
    assert task0["answer_field"] == "reward_model.ground_truth"
    assert task0["verifier"] == "scorer:math"
    assert float(task0["weight"]) == pytest.approx(1.0)

    # GRPO-lambda overlong shaping
    assert cfg.rl.overlong.enabled is True
    assert cfg.rl.overlong.accuracy_gated is True
    assert int(cfg.rl.overlong.buffer) == 384
    assert int(cfg.rl.overlong.target_tokens) == 1024
    assert float(cfg.rl.overlong.concise_bonus) == pytest.approx(0.15)
    assert float(cfg.rl.overlong.min_pass_rate) == pytest.approx(0.25)

    # GUARD G1-G3 + SI-RSI intact
    assert cfg.rl.gate.mode == "soft_hurt"
    assert cfg.rl.gate.eval_at_step_0 is True
    assert int(cfg.rl.gate.every) == 20
    assert int(cfg.rl.gate.depth) == 1
    benches = list(cfg.rl.gate.benchmarks)
    assert len(benches) == 2
    assert benches[0] == {"name": "aime_2026", "limit": 30, "max_new_tokens": 2048, "weight": 0.5}
    assert benches[1] == {"name": "gpqa_diamond", "limit": 50, "max_new_tokens": 1024, "weight": 0.5}

    scfg = SIRSIConfig.from_dict(dict(cfg.rl.sirsi))
    assert scfg.enabled is True
    assert scfg.regret_enabled is True
    assert scfg.atomic_checks_enabled is True
