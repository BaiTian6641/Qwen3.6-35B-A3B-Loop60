"""Comprehensive offline unit & integration tests for Blackwell (`sm_120`) FP8 & NVFP4 support.

Covers:
  1. `dequantize_fp8_tensor` (`[128, 128]` block-scale, non-divisible edge dims, 2D/3D) and
     `dequantize_nvfp4_tensor` (exact IEEE E2M1 nibble unpacking + `compressed-tensors` scale modes).
  2. `_load_hf` `quantization_config` propagation onto `text_config` + `model.language_model.*`
     ignore-list normalization + `load_model` automatic autograd bridge installation.
  3. Inference-only mock `FineGrainedFP8` and `CompressedLinear` / NVFP4 `ToyForCausalLM` (2D linears
     + fused-3D `[E, 2I, H]` & transposed `[E, H, 2I]` MoE experts + `ModuleList` experts):
     - Without bridge: backward fails or starves upstream adapters of gradients.
     - With `install_quantized_autograd_bridge` (idempotent): `no_grad()`/`eval()` uses fast native
       forwards (`native_call_count` increments), while grad-enabled forward+backward yields exact
       `dL/dx` & `dL/d(top_k_weights)`, active-only expert dequantization, nonzero gradients on
       `LoopAdapters` + `PassRouterAdapters` + `HSSDAdapter` + `GDN2Adapter`, and zero persistent
       parameter/buffer bloat.
  4. `sha256_frozen_experts` raw-byte hashing across `float8_e4m3fn` and `uint8` NVFP4 weights +
     scale/global_scale buffers and 1-byte tamper detection.
  5. `HotExpertDeltas` / `install_hot_experts` auto-wiring `auto_dequant_expert_fn` on FP8 & NVFP4.
  6. `export_whole_model_pack` preserving compact quantized weights + scale/packed buffers,
     preserving `quantization_config` in `config.json`, and keeping `hot_experts` unmerged in the
     sidecar state dict when `merge_hot_experts=True` is requested on a quantized backbone.
"""
from __future__ import annotations

import ast
import copy
import importlib.util
import json
import logging
import math
import pathlib
import sys
import types
from typing import Any

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from qloop.data.tokenizer import ToyTokenizer
from qloop.export_pack import export_whole_model_pack
from qloop.infer.hf_custom_modeling import (
    CONFIG_FILENAME,
    MODELING_FILENAME,
    Qwen36LoopHSSDForCausalLM,
    load_safetensors_file,
    write_standalone_hf_modeling_files,
)
from qloop.model import quant_autograd as qa
from qloop.model.experts import (
    _EXPERT_DELTA_KEY_RE,
    HotExpertDeltas,
    attach_hot_expert_deltas,
    freeze_all_but,
    install_hot_experts,
    sha256_frozen_experts,
)
from qloop.model.gdn2 import GDN2Config, install_gdn2
from qloop.model.hssd import HSSDConfig, apply_hssd
from qloop.model.introspect import inspect_model
from qloop.model.loading import (
    _propagate_quantization_config_to_text_config,
    load_model,
)
from qloop.model.loop import LoopSpec, apply_loop
from qloop.model.probes import probe_backward, run_structured_probe_suite
from qloop.model.quant_autograd import (
    auto_dequant_expert_fn,
    dequantize_fp8_tensor,
    dequantize_nvfp4_tensor,
    detect_model_quantization,
    install_quantized_autograd_bridge,
    is_quantized_module,
    quantize_fp8_blockwise,
    quantize_nvfp4_tensor,
)
from qloop.model.toy import ToyConfig, ToyForCausalLM, build_toy
from qloop.train.stack import Stack

# ============================================================================ mock Blackwell modules


class MockFineGrainedFP8Linear(nn.Linear):
    """Simulates HF `FP8Linear(nn.Linear)` (`FineGrainedFP8`) whose native kernel is inference-only."""

    def __init__(
        self,
        linear: nn.Linear,
        block_size: tuple[int, int] = (128, 128),
    ) -> None:
        super().__init__(int(linear.in_features), int(linear.out_features), bias=False, device="meta")
        self.block_size = tuple(block_size)
        q_w, s_inv = quantize_fp8_blockwise(linear.weight.detach(), block_size=block_size)
        self.weight = nn.Parameter(q_w, requires_grad=False)
        self.register_buffer("weight_scale_inv", s_inv)
        if linear.bias is not None:
            self.bias = nn.Parameter(linear.bias.detach().clone(), requires_grad=False)
        else:
            self.register_parameter("bias", None)
        self.native_call_count = 0

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.native_call_count += 1
        w = dequantize_fp8_tensor(self.weight, self.weight_scale_inv, dtype=x.dtype, block_size=self.block_size)
        b = self.bias.to(dtype=x.dtype) if self.bias is not None else None
        return F.linear(x, w, b)


class _BrokenBackwardFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, x: torch.Tensor, w_fp8: torch.Tensor, s_inv: torch.Tensor) -> torch.Tensor:
        w = dequantize_fp8_tensor(w_fp8, s_inv, dtype=x.dtype)
        return F.linear(x, w)

    @staticmethod
    def backward(ctx: Any, grad_out: torch.Tensor) -> tuple[torch.Tensor, None, None]:
        raise RuntimeError("cutlass sm_120 fp8 inference kernel does not implement backward")


class MockRaisingFP8Linear(nn.Linear):
    """Simulates a quantized kernel that runs forward in autograd but raises in `backward()`."""

    def __init__(self, linear: nn.Linear, block_size: tuple[int, int] = (128, 128)) -> None:
        super().__init__(int(linear.in_features), int(linear.out_features), bias=False, device="meta")
        q_w, s_inv = quantize_fp8_blockwise(linear.weight.detach(), block_size=block_size)
        self.weight = nn.Parameter(q_w, requires_grad=False)
        self.register_buffer("weight_scale_inv", s_inv)
        self.register_parameter("bias", None)
        self.native_call_count = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.native_call_count += 1
        return _BrokenBackwardFn.apply(x, self.weight, self.weight_scale_inv)


class MockCompressedNVFP4Linear(nn.Linear):
    """Simulates `compressed-tensors` `CompressedLinear(nn.Linear)` (`nvfp4-pack-quantized`)."""

    def __init__(
        self,
        linear: nn.Linear,
        group_size: int = 16,
        global_scale_mode: str = "divisor",
    ) -> None:
        super().__init__(int(linear.in_features), int(linear.out_features), bias=False, device="meta")
        self._parameters.pop("weight", None)
        self.group_size = int(group_size)
        self.global_scale_mode = global_scale_mode
        packed, w_scale, w_gs = quantize_nvfp4_tensor(
            linear.weight.detach(),
            group_size=group_size,
            global_scale_mode=global_scale_mode,
        )
        self.register_buffer("weight_packed", packed)
        self.register_buffer("weight_scale", w_scale)
        self.register_buffer("weight_global_scale", w_gs)
        if linear.bias is not None:
            self.bias = nn.Parameter(linear.bias.detach().clone(), requires_grad=False)
        else:
            self.register_parameter("bias", None)
        self.native_call_count = 0

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.native_call_count += 1
        w = dequantize_nvfp4_tensor(
            self.weight_packed,
            self.weight_scale,
            self.weight_global_scale,
            dtype=x.dtype,
            group_size=self.group_size,
            global_scale_mode=self.global_scale_mode,
        )
        b = self.bias.to(dtype=x.dtype) if self.bias is not None else None
        return F.linear(x, w, b)


class MockFP8FusedExperts(nn.Module):
    """Simulates fused-3D FP8 `Qwen3_5MoeExperts` with inference-only `@torch.no_grad()` forward."""

    def __init__(
        self,
        gate_up: torch.Tensor,
        down: torch.Tensor,
        *,
        block_size: tuple[int, int] = (128, 128),
        transposed: bool = False,
    ) -> None:
        super().__init__()
        self.num_experts = int(gate_up.shape[0])
        self.intermediate_size = int(down.shape[2])
        self.hidden_size = int(down.shape[1])
        self.block_size = tuple(block_size)
        self.transposed = bool(transposed)

        gu_src = gate_up.detach().transpose(1, 2).contiguous() if transposed else gate_up.detach()
        dn_src = down.detach().transpose(1, 2).contiguous() if transposed else down.detach()
        gu_q, gu_s = quantize_fp8_blockwise(gu_src, block_size=block_size)
        dn_q, dn_s = quantize_fp8_blockwise(dn_src, block_size=block_size)

        self.gate_up_proj = nn.Parameter(gu_q, requires_grad=False)
        self.down_proj = nn.Parameter(dn_q, requires_grad=False)
        self.register_buffer("gate_up_proj_scale_inv", gu_s)
        self.register_buffer("down_proj_scale_inv", dn_s)
        self.native_call_count = 0

    @torch.no_grad()
    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        self.native_call_count += 1
        out = torch.zeros_like(hidden_states)
        for e in torch.unique(top_k_index).tolist():
            eid = int(e)
            tok, slot = (top_k_index == eid).nonzero(as_tuple=True)
            if tok.numel() == 0:
                continue
            gu = dequantize_fp8_tensor(
                self.gate_up_proj[eid],
                self.gate_up_proj_scale_inv[eid],
                dtype=hidden_states.dtype,
                block_size=self.block_size,
            )
            dn = dequantize_fp8_tensor(
                self.down_proj[eid],
                self.down_proj_scale_inv[eid],
                dtype=hidden_states.dtype,
                block_size=self.block_size,
            )
            if self.transposed:
                gu = gu.t().contiguous()
                dn = dn.t().contiguous()
            gate, up = F.linear(hidden_states[tok], gu).chunk(2, dim=-1)
            y = F.linear(F.silu(gate) * up, dn)
            out.index_add_(0, tok, y * top_k_weights[tok, slot, None].to(hidden_states.dtype))
        return out


class MockNVFP4FusedExperts(nn.Module):
    """Simulates fused-3D NVFP4 `experts` (`uint8` packed + `float8_e4m3fn` group-16 scales)."""

    def __init__(
        self,
        gate_up: torch.Tensor,
        down: torch.Tensor,
        *,
        group_size: int = 16,
        global_scale_mode: str = "divisor",
    ) -> None:
        super().__init__()
        self.num_experts = int(gate_up.shape[0])
        self.intermediate_size = int(down.shape[2])
        self.hidden_size = int(down.shape[1])
        self.group_size = int(group_size)
        self.global_scale_mode = global_scale_mode

        gu_p, gu_s, gu_gs = quantize_nvfp4_tensor(
            gate_up.detach(),
            group_size=group_size,
            global_scale_mode=global_scale_mode,
        )
        dn_p, dn_s, dn_gs = quantize_nvfp4_tensor(
            down.detach(),
            group_size=group_size,
            global_scale_mode=global_scale_mode,
        )
        self.register_buffer("gate_up_proj_packed", gu_p)
        self.register_buffer("gate_up_proj_scale", gu_s)
        self.register_buffer("gate_up_proj_global_scale", gu_gs)
        self.register_buffer("down_proj_packed", dn_p)
        self.register_buffer("down_proj_scale", dn_s)
        self.register_buffer("down_proj_global_scale", dn_gs)
        self.native_call_count = 0

    @torch.no_grad()
    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        self.native_call_count += 1
        out = torch.zeros_like(hidden_states)
        for e in torch.unique(top_k_index).tolist():
            eid = int(e)
            tok, slot = (top_k_index == eid).nonzero(as_tuple=True)
            if tok.numel() == 0:
                continue
            gu = dequantize_nvfp4_tensor(
                self.gate_up_proj_packed[eid],
                self.gate_up_proj_scale[eid],
                self.gate_up_proj_global_scale,
                dtype=hidden_states.dtype,
                group_size=self.group_size,
                global_scale_mode=self.global_scale_mode,
            )
            dn = dequantize_nvfp4_tensor(
                self.down_proj_packed[eid],
                self.down_proj_scale[eid],
                self.down_proj_global_scale,
                dtype=hidden_states.dtype,
                group_size=self.group_size,
                global_scale_mode=self.global_scale_mode,
            )
            gate, up = F.linear(hidden_states[tok], gu).chunk(2, dim=-1)
            y = F.linear(F.silu(gate) * up, dn)
            out.index_add_(0, tok, y * top_k_weights[tok, slot, None].to(hidden_states.dtype))
        return out


def _quantize_toy_model_fp8(
    model: ToyForCausalLM,
    *,
    block_size: tuple[int, int] = (16, 16),
    transposed_experts: bool = False,
) -> ToyForCausalLM:
    """Convert a `ToyForCausalLM` in-place to mock `Qwen3.6-35B-A3B-FP8` (leaving `lm_head`/routers float)."""
    qcfg = {
        "quant_method": "fp8",
        "fmt": "e4m3",
        "weight_block_size": list(block_size),
        "modules_to_not_convert": ["lm_head", "*.mlp.gate", "*.mlp.shared_expert_gate"],
    }
    model.config.quantization_config = copy.deepcopy(qcfg)  # type: ignore[attr-defined]
    model.model.config.quantization_config = copy.deepcopy(qcfg)  # type: ignore[attr-defined]

    for layer in model.model.layers:
        if layer.layer_type == "linear_attention":
            la = layer.linear_attn
            la.in_proj_qkv = MockFineGrainedFP8Linear(la.in_proj_qkv, block_size=block_size)  # type: ignore[assignment]
            la.out_proj = MockFineGrainedFP8Linear(la.out_proj, block_size=block_size)  # type: ignore[assignment]
        else:
            sa = layer.self_attn
            sa.q_proj = MockFineGrainedFP8Linear(sa.q_proj, block_size=block_size)  # type: ignore[assignment]
            sa.k_proj = MockFineGrainedFP8Linear(sa.k_proj, block_size=block_size)  # type: ignore[assignment]
            sa.v_proj = MockFineGrainedFP8Linear(sa.v_proj, block_size=block_size)  # type: ignore[assignment]
            sa.o_proj = MockFineGrainedFP8Linear(sa.o_proj, block_size=block_size)  # type: ignore[assignment]
        se = layer.mlp.shared_expert
        se.gate_proj = MockFineGrainedFP8Linear(se.gate_proj, block_size=block_size)  # type: ignore[assignment]
        se.up_proj = MockFineGrainedFP8Linear(se.up_proj, block_size=block_size)  # type: ignore[assignment]
        se.down_proj = MockFineGrainedFP8Linear(se.down_proj, block_size=block_size)  # type: ignore[assignment]
        exp = layer.mlp.experts
        layer.mlp.experts = MockFP8FusedExperts(  # type: ignore[assignment]
            exp.gate_up_proj,
            exp.down_proj,
            block_size=block_size,
            transposed=transposed_experts,
        )
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def _quantize_toy_model_nvfp4(
    model: ToyForCausalLM,
    *,
    group_size: int = 16,
    ignore_full_attn: bool = True,
) -> ToyForCausalLM:
    """Convert a `ToyForCausalLM` in-place to mock `unsloth/Qwen3.6-35B-A3B-NVFP4-Fast`."""
    qcfg = {
        "quant_method": "compressed-tensors",
        "format": "nvfp4-pack-quantized",
        "ignore": ["lm_head", "mtp.*", "*.attn.*"],
        "config_groups": {
            "group_0": {
                "targets": ["Linear"],
                "weights": {"num_bits": 4, "type": "float", "group_size": group_size},
            }
        },
    }
    model.config.quantization_config = copy.deepcopy(qcfg)  # type: ignore[attr-defined]
    model.model.config.quantization_config = copy.deepcopy(qcfg)  # type: ignore[attr-defined]

    for layer in model.model.layers:
        if layer.layer_type == "linear_attention":
            la = layer.linear_attn
            la.in_proj_qkv = MockCompressedNVFP4Linear(  # type: ignore[assignment]
                la.in_proj_qkv,
                group_size=group_size,
            )
            la.out_proj = MockCompressedNVFP4Linear(la.out_proj, group_size=group_size)  # type: ignore[assignment]
        elif not ignore_full_attn:
            sa = layer.self_attn
            sa.q_proj = MockCompressedNVFP4Linear(sa.q_proj, group_size=group_size)  # type: ignore[assignment]
            sa.k_proj = MockCompressedNVFP4Linear(sa.k_proj, group_size=group_size)  # type: ignore[assignment]
            sa.v_proj = MockCompressedNVFP4Linear(sa.v_proj, group_size=group_size)  # type: ignore[assignment]
            sa.o_proj = MockCompressedNVFP4Linear(sa.o_proj, group_size=group_size)  # type: ignore[assignment]
        se = layer.mlp.shared_expert
        se.gate_proj = MockCompressedNVFP4Linear(se.gate_proj, group_size=group_size)  # type: ignore[assignment]
        se.up_proj = MockCompressedNVFP4Linear(se.up_proj, group_size=group_size)  # type: ignore[assignment]
        se.down_proj = MockCompressedNVFP4Linear(se.down_proj, group_size=group_size)  # type: ignore[assignment]
        exp = layer.mlp.experts
        layer.mlp.experts = MockNVFP4FusedExperts(  # type: ignore[assignment]
            exp.gate_up_proj,
            exp.down_proj,
            group_size=group_size,
        )
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def _build_float_reference_from_quantized(qmodel: ToyForCausalLM) -> ToyForCausalLM:
    """Materialize a pure-float32 `ToyForCausalLM` with the exact dequantized weights of `qmodel`."""
    ref = ToyForCausalLM(copy.deepcopy(qmodel.config)).eval()
    with torch.no_grad():
        ref.model.embed_tokens.weight.copy_(qmodel.model.embed_tokens.weight.float())
        ref.model.norm.weight.copy_(qmodel.model.norm.weight.float())
        ref.lm_head.weight.copy_(qmodel.lm_head.weight.float())
        for r_layer, q_layer in zip(ref.model.layers, qmodel.model.layers):
            r_layer.input_layernorm.weight.copy_(q_layer.input_layernorm.weight.float())
            r_layer.post_attention_layernorm.weight.copy_(q_layer.post_attention_layernorm.weight.float())
            if q_layer.layer_type == "linear_attention":
                r_la, q_la = r_layer.linear_attn, q_layer.linear_attn
                for attr in ("in_proj_qkv", "in_proj_ba", "out_proj"):
                    q_sub = getattr(q_la, attr)
                    r_sub = getattr(r_la, attr)
                    if is_quantized_module(q_sub):
                        w_deq = qa._dequant_linear_weight(q_sub, dtype=torch.float32)
                        r_sub.weight.copy_(w_deq)
                    else:
                        r_sub.weight.copy_(q_sub.weight.float())
                r_la.conv1d.weight.copy_(q_la.conv1d.weight.float())
                r_la.A_log.copy_(q_la.A_log.float())
                r_la.dt_bias.copy_(q_la.dt_bias.float())
                r_la.norm.weight.copy_(q_la.norm.weight.float())
            else:
                r_sa, q_sa = r_layer.self_attn, q_layer.self_attn
                for attr in ("q_proj", "k_proj", "v_proj", "o_proj"):
                    q_sub = getattr(q_sa, attr)
                    r_sub = getattr(r_sa, attr)
                    if is_quantized_module(q_sub):
                        w_deq = qa._dequant_linear_weight(q_sub, dtype=torch.float32)
                        r_sub.weight.copy_(w_deq)
                    else:
                        r_sub.weight.copy_(q_sub.weight.float())
                r_sa.q_norm.weight.copy_(q_sa.q_norm.weight.float())
                r_sa.k_norm.weight.copy_(q_sa.k_norm.weight.float())

            r_layer.mlp.gate.weight.copy_(q_layer.mlp.gate.weight.float())
            r_layer.mlp.shared_expert_gate.weight.copy_(q_layer.mlp.shared_expert_gate.weight.float())
            for attr in ("gate_proj", "up_proj", "down_proj"):
                q_sub = getattr(q_layer.mlp.shared_expert, attr)
                r_sub = getattr(r_layer.mlp.shared_expert, attr)
                w_deq = qa._dequant_linear_weight(q_sub, dtype=torch.float32)
                r_sub.weight.copy_(w_deq)
            for eid in range(ref.config.num_experts):
                gu_e, dn_e = auto_dequant_expert_fn(q_layer.mlp.experts, eid)
                r_layer.mlp.experts.gate_up_proj[eid].copy_(gu_e)
                r_layer.mlp.experts.down_proj[eid].copy_(dn_e)
    for p in ref.parameters():
        p.requires_grad_(False)
    return ref


def _persistent_state_bytes(model: nn.Module) -> int:
    total = sum(p.numel() * p.element_size() for p in model.parameters())
    total += sum(b.numel() * b.element_size() for b in model.buffers())
    return int(total)


# ============================================================================ 1. dequantize helpers


def test_dequantize_fp8_tensor_2d_3d_and_non_divisible_blocks() -> None:
    g = torch.Generator().manual_seed(2026)
    # 2D with non-divisible dimensions against [128, 128] FineGrainedFP8 blocks
    w2 = torch.randn(300, 150, generator=g, dtype=torch.float32) * 0.75
    q2, s2 = quantize_fp8_blockwise(w2, block_size=(128, 128))
    assert q2.dtype == torch.float8_e4m3fn
    assert tuple(s2.shape) == (3, 2)

    # Note: without explicit block_size, _infer_block_dim resolves 300 / 3 -> 128 (standard HF block)
    deq2 = dequantize_fp8_tensor(q2, s2, dtype=torch.bfloat16)
    assert deq2.dtype == torch.bfloat16
    assert tuple(deq2.shape) == (300, 150)
    rel_err_2d = float((deq2.float() - w2).norm() / w2.norm())
    assert rel_err_2d < 0.08

    # 3D fused expert shape [E, 2I, H] with non-divisible edge dimensions
    w3 = torch.randn(5, 260, 140, generator=g, dtype=torch.float32) * 0.4
    q3, s3 = quantize_fp8_blockwise(w3, block_size=(128, 128))
    assert tuple(s3.shape) == (5, 3, 2)
    deq3 = dequantize_fp8_tensor(q3, s3, dtype=torch.float32)
    assert tuple(deq3.shape) == (5, 260, 140)
    rel_err_3d = float((deq3 - w3).norm() / w3.norm())
    assert rel_err_3d < 0.08

    # Scalar & per-channel scale_inv + None fallback
    s_scalar = torch.tensor(0.25, dtype=torch.float32)
    deq_sc = dequantize_fp8_tensor(q2, s_scalar, dtype=torch.float32)
    assert torch.allclose(deq_sc, q2.float() * 0.25)
    s_chan = torch.linspace(0.1, 0.5, steps=300, dtype=torch.float32)
    deq_ch = dequantize_fp8_tensor(q2, s_chan, dtype=torch.float32)
    assert torch.allclose(deq_ch, q2.float() * s_chan.unsqueeze(1))
    deq_none = dequantize_fp8_tensor(q2, None, dtype=torch.float32)
    assert torch.allclose(deq_none, q2.float())


def test_dequantize_nvfp4_e2m1_exact_table_and_scale_conventions() -> None:
    # Pack all 16 E2M1 nibble codes [0x0 .. 0xF] into 8 uint8 bytes (low nibble, high nibble)
    # Codes 0..7 -> [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
    # Codes 8..15 -> [-0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]
    bytes_list = [((2 * i + 1) << 4) | (2 * i) for i in range(8)]
    packed = torch.tensor([bytes_list], dtype=torch.uint8)  # [1, 8] -> unpacks to [1, 16]
    unit_scale = torch.ones((1, 1), dtype=torch.float8_e4m3fn)
    global_scale = torch.tensor(2.0, dtype=torch.float32)

    expected_vals = torch.tensor(
        [[0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]],
        dtype=torch.float32,
    )
    # compressed-tensors divisor convention: fp4 * scale / global_scale
    deq_div = dequantize_nvfp4_tensor(
        packed,
        unit_scale,
        global_scale,
        dtype=torch.float32,
        group_size=16,
        global_scale_mode="divisor",
    )
    assert float((deq_div - expected_vals / 2.0).abs().max()) == 0.0

    # Multiplier convention: fp4 * scale * global_scale
    deq_mul = dequantize_nvfp4_tensor(
        packed,
        unit_scale,
        global_scale,
        dtype=torch.float32,
        group_size=16,
        global_scale_mode="multiplier",
    )
    assert float((deq_mul - expected_vals * 2.0).abs().max()) == 0.0

    # 2D & 3D round-trip quantization accuracy with group-16 float8_e4m3fn scales
    g = torch.Generator().manual_seed(99)
    w3 = torch.randn(4, 32, 64, generator=g, dtype=torch.float32) * 1.2
    p3, s3, gs3 = quantize_nvfp4_tensor(w3, group_size=16, global_scale_mode="divisor")
    assert p3.dtype == torch.uint8 and tuple(p3.shape) == (4, 32, 32)
    assert s3.dtype == torch.float8_e4m3fn and tuple(s3.shape) == (4, 32, 4)
    rec3 = dequantize_nvfp4_tensor(p3, s3, gs3, dtype=torch.bfloat16, group_size=16)
    assert rec3.dtype == torch.bfloat16 and tuple(rec3.shape) == (4, 32, 64)
    rel_err = float((rec3.float() - w3).norm() / w3.norm())
    assert rel_err < 0.22


# ============================================================================ 2. loading.py propagation


def test_load_hf_propagates_and_normalizes_quantization_config_for_text_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FakeTextCfg:
        def __init__(self) -> None:
            self.model_type = "qwen3_5_moe_text"
            self.hidden_size = 32
            self.num_hidden_layers = 4
            self.quantization_config = None

    class _FakeTopCfg:
        def __init__(self, qcfg: Any) -> None:
            self.model_type = "qwen3_5"
            self.text_config = _FakeTextCfg()
            self.quantization_config = qcfg

    nvfp4_qcfg = {
        "quant_method": "compressed-tensors",
        "format": "nvfp4-pack-quantized",
        "ignore": [
            "lm_head",
            "mtp.*",
            "*.attn.*",
            "model.language_model.layers.0.mlp.gate",
            "language_model.layers.1.mlp.shared_expert_gate",
            "model.visual.blocks.0.attn.qkv",
            "visual.merger.mlp.0",
        ],
        "config_groups": {
            "group_0": {
                "targets": ["Linear"],
                "ignore": [
                    "model.language_model.embed_tokens",
                    "model.visual.patch_embed",
                ],
            }
        },
    }
    top_cfg = _FakeTopCfg(nvfp4_qcfg)
    propagated = _propagate_quantization_config_to_text_config(top_cfg, top_cfg.text_config)
    tc_qcfg = propagated.quantization_config
    assert isinstance(tc_qcfg, dict)
    assert tc_qcfg["quant_method"] == "compressed-tensors"
    assert tc_qcfg["ignore"] == [
        "lm_head",
        "mtp.*",
        "*.attn.*",
        "model.layers.0.mlp.gate",
        "model.layers.1.mlp.shared_expert_gate",
    ]
    assert tc_qcfg["config_groups"]["group_0"]["ignore"] == ["model.embed_tokens"]
    # Original top-level quantization_config must not be mutated in place
    assert "model.visual.blocks.0.attn.qkv" in nvfp4_qcfg["ignore"]

    # End-to-end `load_model` test with mocked `transformers` returning an FP8-quantized model
    captured_call: dict[str, Any] = {}
    fp8_qcfg_obj = types.SimpleNamespace(
        to_dict=lambda: {
            "quant_method": "fp8",
            "fmt": "e4m3",
            "weight_block_size": [128, 128],
            "modules_to_not_convert": [
                "lm_head",
                "model.language_model.layers.0.mlp.gate",
                "model.visual.blocks",
            ],
        }
    )

    def _fake_from_pretrained_cfg(name: str, **kw: Any) -> Any:
        return _FakeTopCfg(fp8_qcfg_obj)

    def _fake_from_pretrained_model(name: str, **kw: Any) -> nn.Module:
        captured_call["name"] = name
        captured_call["kwargs"] = kw
        toy = build_toy(
            seed=7,
            num_hidden_layers=4,
            hidden_size=32,
            moe_intermediate_size=16,
            shared_expert_intermediate_size=16,
            num_experts=4,
        )
        return _quantize_toy_model_fp8(toy, block_size=(16, 16))

    fake_tf = types.ModuleType("transformers")
    fake_tf.__version__ = "5.23.0"  # type: ignore[attr-defined]
    fake_tf.AutoConfig = types.SimpleNamespace(from_pretrained=_fake_from_pretrained_cfg)  # type: ignore[attr-defined]
    fake_tf.AutoModelForCausalLM = types.SimpleNamespace(  # type: ignore[attr-defined]
        from_pretrained=_fake_from_pretrained_model
    )
    fake_tf.AutoTokenizer = types.SimpleNamespace(  # type: ignore[attr-defined]
        from_pretrained=lambda *_a, **_kw: ToyTokenizer()
    )
    monkeypatch.setitem(__import__("sys").modules, "transformers", fake_tf)

    loaded_model, _tok, info = load_model(
        {
            "name": "Qwen/Qwen3.6-35B-A3B-FP8",
            "dtype": "float32",
            "device_map": "cpu",
            "text_only": True,
            "attn_implementation": "sdpa",
            "trust_remote_code": False,
        }
    )
    passed_cfg = captured_call["kwargs"]["config"]
    assert passed_cfg.quantization_config["quant_method"] == "fp8"
    assert passed_cfg.quantization_config["modules_to_not_convert"] == [
        "lm_head",
        "model.layers.0.mlp.gate",
    ]
    assert info["quantization"]["is_quantized"] is True
    assert info["quantization"]["quant_method"] == "fp8"
    assert info["quantization"]["bridged_total"] > 0
    assert getattr(loaded_model.model.layers[0].mlp.experts, "_qloop_quant_bridge_installed", False) is True


# ============================================================================ 3. autograd bridge


@pytest.mark.parametrize("scheme", ["fp8", "nvfp4"])
def test_quantized_autograd_bridge_native_fast_path_and_active_only_moe(
    scheme: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch.manual_seed(11)
    h_dim, inter_dim, num_experts = 32, 16, 12
    lin_ref = nn.Linear(h_dim, 48, bias=False)
    gu_ref = torch.randn(num_experts, 2 * inter_dim, h_dim) / math.sqrt(h_dim)
    dn_ref = torch.randn(num_experts, h_dim, inter_dim) / math.sqrt(inter_dim)

    if scheme == "fp8":
        qlin: nn.Module = MockFineGrainedFP8Linear(lin_ref, block_size=(16, 16))
        qmoe: nn.Module = MockFP8FusedExperts(gu_ref, dn_ref, block_size=(16, 16))
    else:
        qlin = MockCompressedNVFP4Linear(lin_ref, group_size=16)
        qmoe = MockNVFP4FusedExperts(gu_ref, dn_ref, group_size=16)

    wrapper = nn.ModuleDict({"lin": qlin, "experts": qmoe})

    # BEFORE bridge: inference-only `@torch.no_grad()` forward severs autograd completely
    x0 = torch.randn(3, h_dim, requires_grad=True)
    y0 = qlin(x0)
    assert not y0.requires_grad

    top_idx = torch.tensor([[1, 5], [5, 9], [1, 2]], dtype=torch.long)  # 4 unique out of 12 experts
    top_w0 = torch.softmax(torch.randn(3, 2), dim=-1).detach().requires_grad_(True)
    m0 = qmoe(x0, top_idx, top_w0)
    assert not m0.requires_grad

    # Install bridge & verify idempotency on second call
    bytes_before = _persistent_state_bytes(wrapper)
    rep1 = install_quantized_autograd_bridge(wrapper)
    assert rep1["bridged_linear_count"] == 1
    assert rep1["bridged_moe_count"] == 1
    assert rep1["native_fwd_ok_count"] == 2
    rep2 = install_quantized_autograd_bridge(wrapper)
    assert rep2["bridged_linear_count"] == 0
    assert rep2["bridged_moe_count"] == 0
    assert rep2["already_bridged_count"] == 2
    assert rep2["native_fwd_ok_count"] == 2

    # 1) `no_grad()` / `eval()` MUST hit the fast native quantized forward (`native_call_count`++)
    qlin.native_call_count = 0  # type: ignore[attr-defined]
    qmoe.native_call_count = 0  # type: ignore[attr-defined]
    with torch.no_grad():
        y_eval = qlin(x0.detach())
        m_eval = qmoe(x0.detach(), top_idx, top_w0.detach())
    assert qlin.native_call_count == 1  # type: ignore[attr-defined]
    assert qmoe.native_call_count == 1  # type: ignore[attr-defined]

    # Even with grad enabled, if NO input tensor requires_grad, native fast path is still used!
    y_nograd_inp = qlin(x0.detach())
    m_nograd_inp = qmoe(x0.detach(), top_idx, top_w0.detach())
    assert qlin.native_call_count == 2  # type: ignore[attr-defined]
    assert qmoe.native_call_count == 2  # type: ignore[attr-defined]
    assert torch.allclose(y_eval, y_nograd_inp)
    assert torch.allclose(m_eval, m_nograd_inp)

    # 2) Spy on `_dequant_experts_chunk` & `_dequant_single_expert` to prove ONLY active routed
    #    experts are dequantized in a single batched 3D chunk call (`0` unused experts touched!)
    seen_chunks: list[list[int]] = []
    single_calls: list[int] = []
    orig_dequant_chunk = qa._dequant_experts_chunk
    orig_dequant_single = qa._dequant_single_expert

    def _spy_chunk(
        mod: nn.Module,
        expert_ids: list[int],
        dtype: torch.dtype = torch.bfloat16,
        *,
        nvfp4_mode_override: str | None = None,
    ) -> tuple[torch.Tensor | list[torch.Tensor], torch.Tensor | list[torch.Tensor]]:
        seen_chunks.append([int(e) for e in expert_ids])
        return orig_dequant_chunk(mod, expert_ids, dtype=dtype, nvfp4_mode_override=nvfp4_mode_override)

    def _spy_single(
        mod: nn.Module,
        e: int,
        dtype: torch.dtype = torch.float32,
        *,
        nvfp4_mode_override: str | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        single_calls.append(int(e))
        return orig_dequant_single(mod, e, dtype=dtype, nvfp4_mode_override=nvfp4_mode_override)

    monkeypatch.setattr(qa, "_dequant_experts_chunk", _spy_chunk)
    monkeypatch.setattr(qa, "_dequant_single_expert", _spy_single)

    x1 = x0.detach().clone().requires_grad_(True)
    w1 = top_w0.detach().clone().requires_grad_(True)
    out_lin = qlin(x1)
    out_moe = qmoe(x1, top_idx, w1)
    # Forward stayed on bridged path (`native_call_count` stayed at 2)
    assert qlin.native_call_count == 2  # type: ignore[attr-defined]
    assert qmoe.native_call_count == 2  # type: ignore[attr-defined]
    assert seen_chunks == [[1, 2, 5, 9]]  # 1 batched 3D chunk call; 8 unused experts NEVER dequantized!
    assert single_calls == []

    loss = (out_lin.pow(2).mean() + out_moe.pow(2).mean())
    loss.backward()
    # Backward recomputes ONLY the 4 active experts in 1 batched 3D chunk call (zero unused!)
    assert seen_chunks == [[1, 2, 5, 9], [1, 2, 5, 9]]
    assert single_calls == []
    assert x1.grad is not None and torch.isfinite(x1.grad).all() and float(x1.grad.norm()) > 0.0
    assert w1.grad is not None and torch.isfinite(w1.grad).all() and float(w1.grad.norm()) > 0.0

    # Compare against exact float-dequantized reference for both `dL/dx` and `dL/d(top_w)`
    x_ref = x0.detach().clone().requires_grad_(True)
    w_ref = top_w0.detach().clone().requires_grad_(True)
    w_lin_deq = qa._dequant_linear_weight(qlin, dtype=x_ref.dtype)
    out_lin_ref = F.linear(x_ref, w_lin_deq)
    out_moe_ref = torch.zeros_like(x_ref)
    for eid in (1, 2, 5, 9):
        tok, slot = (top_idx == eid).nonzero(as_tuple=True)
        gu_e, dn_e = orig_dequant_single(qmoe, eid, dtype=torch.float32)
        gate, up = F.linear(x_ref[tok], gu_e.to(x_ref.dtype)).chunk(2, dim=-1)
        ye = F.linear(F.silu(gate) * up, dn_e.to(x_ref.dtype))
        out_moe_ref.index_add_(0, tok, ye * w_ref[tok, slot, None])
    loss_ref = out_lin_ref.pow(2).mean() + out_moe_ref.pow(2).mean()
    loss_ref.backward()

    assert torch.allclose(out_lin, out_lin_ref, atol=1e-5, rtol=1e-5)
    assert torch.allclose(out_moe, out_moe_ref, atol=1e-5, rtol=1e-5)
    assert torch.allclose(x1.grad, x_ref.grad, atol=1e-5, rtol=1e-5)
    assert torch.allclose(w1.grad, w_ref.grad, atol=1e-5, rtol=1e-5)

    # Zero persistent VRAM / parameter / buffer bloat
    assert _persistent_state_bytes(wrapper) == bytes_before
    if scheme == "fp8":
        assert qlin.weight.dtype == torch.float8_e4m3fn  # type: ignore[union-attr]
        assert qmoe.gate_up_proj.dtype == torch.float8_e4m3fn  # type: ignore[union-attr]
    else:
        assert qlin.weight_packed.dtype == torch.uint8  # type: ignore[union-attr]
        assert qmoe.gate_up_proj_packed.dtype == torch.uint8  # type: ignore[union-attr]


def test_transposed_fp8_experts_and_modulelist_experts_dequant_and_bridge() -> None:
    torch.manual_seed(23)
    e_cnt, inter_dim, h_dim = 4, 16, 32
    gu = torch.randn(e_cnt, 2 * inter_dim, h_dim) * 0.2
    dn = torch.randn(e_cnt, h_dim, inter_dim) * 0.2

    # 1) Transposed fused-3D `[E, H, 2I]` + `[E, I, H]` FP8 experts
    exp_std = MockFP8FusedExperts(gu, dn, block_size=(16, 16), transposed=False)
    exp_tr = MockFP8FusedExperts(gu, dn, block_size=(16, 16), transposed=True)
    assert tuple(exp_tr.gate_up_proj.shape) == (e_cnt, h_dim, 2 * inter_dim)
    assert tuple(exp_tr.down_proj.shape) == (e_cnt, inter_dim, h_dim)

    gu0_std, dn0_std = auto_dequant_expert_fn(exp_std, 0)
    gu0_tr, dn0_tr = auto_dequant_expert_fn(exp_tr, 0)
    assert tuple(gu0_tr.shape) == (2 * inter_dim, h_dim)
    assert tuple(dn0_tr.shape) == (h_dim, inter_dim)
    assert torch.allclose(gu0_std, gu0_tr, atol=1e-6)
    assert torch.allclose(dn0_std, dn0_tr, atol=1e-6)

    install_quantized_autograd_bridge(exp_tr)
    x = torch.randn(4, h_dim, requires_grad=True)
    idx = torch.tensor([[0, 2], [1, 3], [0, 1], [2, 3]], dtype=torch.long)
    w = torch.softmax(torch.randn(4, 2), dim=-1).detach().requires_grad_(True)
    out = exp_tr(x, idx, w)
    out.pow(2).sum().backward()
    assert x.grad is not None and float(x.grad.norm()) > 0.0
    assert w.grad is not None and float(w.grad.norm()) > 0.0

    # 2) `ModuleList` per-expert (`gate_proj` + `up_proj` + `down_proj`) quantized with NVFP4
    class _SingleNVFP4Expert(nn.Module):
        def __init__(self, eid: int) -> None:
            super().__init__()
            gp = nn.Linear(h_dim, inter_dim, bias=False)
            up = nn.Linear(h_dim, inter_dim, bias=False)
            dp = nn.Linear(inter_dim, h_dim, bias=False)
            with torch.no_grad():
                gp.weight.copy_(gu[eid, :inter_dim])
                up.weight.copy_(gu[eid, inter_dim:])
                dp.weight.copy_(dn[eid])
            self.gate_proj = MockCompressedNVFP4Linear(gp, group_size=16)
            self.up_proj = MockCompressedNVFP4Linear(up, group_size=16)
            self.down_proj = MockCompressedNVFP4Linear(dp, group_size=16)

    ml_experts = nn.ModuleList([_SingleNVFP4Expert(i) for i in range(e_cnt)])
    gu_ml, dn_ml = auto_dequant_expert_fn(ml_experts, 1)
    assert tuple(gu_ml.shape) == (2 * inter_dim, h_dim)
    assert tuple(dn_ml.shape) == (h_dim, inter_dim)
    assert gu_ml.dtype == torch.float32 and dn_ml.dtype == torch.float32


def test_probe_backward_recovers_when_native_fp8_kernel_raises() -> None:
    model = build_toy(
        seed=5,
        num_hidden_layers=4,
        hidden_size=32,
        moe_intermediate_size=16,
        shared_expert_intermediate_size=16,
        num_experts=4,
    )
    model.lm_head = MockRaisingFP8Linear(model.lm_head, block_size=(16, 16))  # type: ignore[assignment]
    for p in model.parameters():
        p.requires_grad_(False)

    ids = torch.randint(0, model.config.vocab_size, (2, 8))
    with pytest.raises(RuntimeError, match="does not implement backward"):
        probe_backward(model, ids)

    rep_bridge = install_quantized_autograd_bridge(model)
    assert rep_bridge["bridged_linear_count"] == 1
    pb = probe_backward(model, ids)
    assert pb["ok"] is True
    assert pb["dead_layers"] == []
    assert pb["input_grad_norm"] > 0.0


@pytest.mark.parametrize("scheme", ["fp8", "nvfp4"])
def test_full_adapter_gradients_loop_router_hssd_gdn2_on_quantized_backbone(scheme: str) -> None:
    def _make_quant_toy() -> ToyForCausalLM:
        base = build_toy(
            seed=31,
            num_hidden_layers=8,
            hidden_size=32,
            moe_intermediate_size=16,
            shared_expert_intermediate_size=16,
            num_experts=6,
            num_experts_per_tok=2,
        )
        if scheme == "fp8":
            return _quantize_toy_model_fp8(base, block_size=(16, 16))
        return _quantize_toy_model_nvfp4(base, group_size=16)

    ids = torch.randint(0, 97, (2, 10), generator=torch.Generator().manual_seed(77))
    spec = LoopSpec(
        prelude_end=2,
        core_end=6,
        num_passes=2,
        target_virtual_layers=None,
        gate_init=0.25,  # nonzero gate so pass-2 core + PassRouterAdapters actively contribute at step 0
        per_pass_router=True,
        router_adapter_rank=4,
        attn_adapter_rank=4,
        checkpoint=False,
    )

    # A) WITHOUT bridge: `PassRouterAdapters` and `HSSDAdapter` get ZERO gradients!
    unbridged = _make_quant_toy()
    hssd_u_handle = apply_hssd(unbridged, HSSDConfig(num_pages=4, k_pages=2, chunk_size=4, rank=4))
    ctl_unbridged = apply_loop(unbridged, spec)
    unbridged.train()
    out_u = unbridged(ids, labels=ids)
    assert out_u.loss is not None
    out_u.loss.backward()
    assert ctl_unbridged.adapters.pass_router is not None
    pr_unbridged_norm = sum(
        float(p.grad.abs().sum()) for p in ctl_unbridged.adapters.pass_router.parameters() if p.grad is not None
    )
    hssd_unbridged_norm = sum(
        float(p.grad.abs().sum())
        for ad in hssd_u_handle.adapters.values()
        for p in ad.parameters()
        if p.grad is not None
    )
    assert pr_unbridged_norm == 0.0
    assert hssd_unbridged_norm == 0.0

    # B) WITH `install_quantized_autograd_bridge`:
    bridged = _make_quant_toy()
    bytes_base = _persistent_state_bytes(bridged)
    b_rep = install_quantized_autograd_bridge(bridged)
    assert b_rep["is_quantized"] is True
    assert b_rep["bridged_linear_count"] > 0
    assert b_rep["bridged_moe_count"] == 8
    assert _persistent_state_bytes(bridged) == bytes_base

    # Verify `emb.grad` matches float-dequantized reference BEFORE adding adapters
    float_ref = _build_float_reference_from_quantized(bridged)
    emb_q = bridged.model.embed_tokens(ids).detach().clone().requires_grad_(True)
    emb_r = emb_q.detach().clone().requires_grad_(True)
    logits_q = bridged.lm_head(bridged.model(inputs_embeds=emb_q).last_hidden_state)
    logits_r = float_ref.lm_head(float_ref.model(inputs_embeds=emb_r).last_hidden_state)
    loss_q = F.cross_entropy(logits_q[:, :-1].reshape(-1, logits_q.shape[-1]), ids[:, 1:].reshape(-1))
    loss_r = F.cross_entropy(logits_r[:, :-1].reshape(-1, logits_r.shape[-1]), ids[:, 1:].reshape(-1))
    loss_q.backward()
    loss_r.backward()
    assert emb_q.grad is not None and emb_r.grad is not None
    assert torch.allclose(emb_q.grad, emb_r.grad, atol=1e-4, rtol=1e-4)

    # Probe backward passes cleanly
    pb_rep = probe_backward(bridged, ids)
    assert pb_rep["ok"] is True
    assert pb_rep["dead_layers"] == []

    # Attach `HSSDAdapter` + `LoopAdapters` (`gates`, `bridge`, `PassRouterAdapters`, `pass_attn`)
    hssd_handle = apply_hssd(bridged, HSSDConfig(num_pages=4, k_pages=2, chunk_size=4, rank=4))
    ctl = apply_loop(bridged, spec)
    bridged.train()
    bridged.zero_grad(set_to_none=True)
    out_b = bridged(ids, labels=ids)
    assert out_b.loss is not None
    out_b.loss.backward()

    assert ctl.adapters.gates.grad is not None and float(ctl.adapters.gates.grad.norm()) > 0.0
    assert ctl.adapters.bridge is not None
    bridge_grad = sum(float(p.grad.norm()) for p in ctl.adapters.bridge.parameters() if p.grad is not None)
    assert bridge_grad > 0.0

    assert ctl.adapters.pass_router is not None
    router_grad = sum(
        float(p.grad.norm()) for p in ctl.adapters.pass_router.parameters() if p.grad is not None
    )
    assert router_grad > 0.0

    hssd_grad = sum(
        float(p.grad.norm())
        for ad in hssd_handle.adapters.values()
        for p in ad.parameters()
        if p.grad is not None
    )
    assert hssd_grad > 0.0

    # Also verify `GDN2Adapter` receives nonzero gradients on a bridged quantized model
    bridged_gdn2 = _make_quant_toy()
    install_quantized_autograd_bridge(bridged_gdn2)
    gdn2_mgr = install_gdn2(bridged_gdn2, GDN2Config(rank=4, chunk_size=4))
    bridged_gdn2.train()
    out_g = bridged_gdn2(ids, labels=ids)
    assert out_g.loss is not None
    out_g.loss.backward()
    gdn2_grad = sum(
        float(p.grad.norm())
        for ad in gdn2_mgr.adapters.values()
        for p in ad.parameters()
        if p.grad is not None
    )
    assert gdn2_grad > 0.0


# ============================================================================ 4. sha256_frozen_experts


@pytest.mark.parametrize("scheme", ["fp8", "nvfp4"])
def test_sha256_frozen_experts_hashes_quantized_weights_and_scale_buffers(scheme: str) -> None:
    base = build_toy(
        seed=42,
        num_hidden_layers=4,
        hidden_size=32,
        moe_intermediate_size=16,
        shared_expert_intermediate_size=16,
        num_experts=4,
    )
    model = (
        _quantize_toy_model_fp8(base, block_size=(16, 16))
        if scheme == "fp8"
        else _quantize_toy_model_nvfp4(base, group_size=16)
    )
    install_quantized_autograd_bridge(model)

    digest_0 = sha256_frozen_experts(model)
    assert isinstance(digest_0, str) and len(digest_0) == 64

    # Invariant across `apply_loop` and across `HotExpertDeltas` parameter mutations
    deltas = install_hot_experts(model, {0: [0, 2], 2: [1]}, rank=4)
    apply_loop(model, LoopSpec(prelude_end=1, core_end=3, num_passes=2, target_virtual_layers=None))
    with torch.no_grad():
        for p in deltas.parameters():
            p.add_(0.125)
    assert sha256_frozen_experts(model) == digest_0

    # 1-byte tamper in quantized weight tensor is caught
    phys0_exp = model.model.layers[0].inner.mlp.experts
    if scheme == "fp8":
        raw_u8 = phys0_exp.gate_up_proj.data.reshape(-1).view(torch.uint8)
        raw_u8[0] ^= 1
        assert sha256_frozen_experts(model) != digest_0
        raw_u8[0] ^= 1
        assert sha256_frozen_experts(model) == digest_0

        # Scale buffer tamper is ALSO caught
        orig_s = float(phys0_exp.gate_up_proj_scale_inv.reshape(-1)[0].item())
        phys0_exp.gate_up_proj_scale_inv.reshape(-1)[0] = orig_s + 0.5
        assert sha256_frozen_experts(model) != digest_0
        phys0_exp.gate_up_proj_scale_inv.reshape(-1)[0] = orig_s
        assert sha256_frozen_experts(model) == digest_0
    else:
        raw_u8 = phys0_exp.gate_up_proj_packed.reshape(-1).view(torch.uint8)
        raw_u8[0] ^= 1
        assert sha256_frozen_experts(model) != digest_0
        raw_u8[0] ^= 1
        assert sha256_frozen_experts(model) == digest_0

        # 0-D scalar `gate_up_proj_global_scale` buffer tamper is ALSO caught
        orig_gs = float(phys0_exp.gate_up_proj_global_scale.item())
        phys0_exp.gate_up_proj_global_scale.fill_(orig_gs * 2.0)
        assert sha256_frozen_experts(model) != digest_0
        phys0_exp.gate_up_proj_global_scale.fill_(orig_gs)
        assert sha256_frozen_experts(model) == digest_0


# ============================================================================ 5. HotExpertDeltas


@pytest.mark.parametrize("scheme", ["fp8", "nvfp4"])
def test_hot_expert_deltas_autowires_dequant_fn_and_trains_on_quantized_experts(scheme: str) -> None:
    base = build_toy(
        seed=55,
        num_hidden_layers=4,
        hidden_size=32,
        moe_intermediate_size=16,
        shared_expert_intermediate_size=16,
        num_experts=4,
        num_experts_per_tok=2,
    )
    model = (
        _quantize_toy_model_fp8(base, block_size=(16, 16))
        if scheme == "fp8"
        else _quantize_toy_model_nvfp4(base, group_size=16)
    )
    install_quantized_autograd_bridge(model)
    sha_before = sha256_frozen_experts(model)

    ids = torch.randint(0, model.config.vocab_size, (2, 8), generator=torch.Generator().manual_seed(13))
    with torch.no_grad():
        logits_before = model(ids).logits.clone()

    # Pass `dequant_fn=None` -> must auto-default to `auto_dequant_expert_fn`!
    hot = {0: [0, 1, 2, 3], 1: [0, 1, 2, 3]}
    deltas = install_hot_experts(model, hot, rank=4)
    assert isinstance(deltas, HotExpertDeltas)
    assert deltas.dequant_fn is auto_dequant_expert_fn

    # Bit-identical at zero-init
    with torch.no_grad():
        logits_zero = model(ids).logits
    assert torch.equal(logits_before, logits_zero)

    # Train 2 steps on `HotExpertDeltas` and verify nonzero grads + loss decrease + frozen base intact
    opt = torch.optim.AdamW(deltas.parameters(), lr=5e-2)
    model.train()
    out0 = model(ids, labels=ids)
    assert out0.loss is not None
    loss0 = float(out0.loss.detach())
    out0.loss.backward()
    delta_grad_norm = sum(float(p.grad.norm()) for p in deltas.parameters() if p.grad is not None)
    assert delta_grad_norm > 0.0
    opt.step()
    opt.zero_grad(set_to_none=True)

    out1 = model(ids, labels=ids)
    assert out1.loss is not None
    loss1 = float(out1.loss.detach())
    assert loss1 < loss0
    assert sha256_frozen_experts(model) == sha_before


# ============================================================================ 6. export_whole_model_pack


@pytest.mark.parametrize("scheme", ["fp8", "nvfp4"])
def test_export_whole_model_pack_preserves_quantized_buffers_config_and_sidecar_hot_experts(
    scheme: str,
    tmp_path: Any,
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = build_toy(
        seed=88,
        num_hidden_layers=4,
        hidden_size=32,
        moe_intermediate_size=16,
        shared_expert_intermediate_size=16,
        num_experts=4,
    )
    model = (
        _quantize_toy_model_fp8(base, block_size=(16, 16))
        if scheme == "fp8"
        else _quantize_toy_model_nvfp4(base, group_size=16)
    )
    install_quantized_autograd_bridge(model)
    deltas = attach_hot_expert_deltas(model, {0: [0, 2], 2: [1]}, rank=4)
    with torch.no_grad():
        for name, p in deltas.named_parameters():
            if name.endswith("_B"):
                p.normal_(0.0, 0.05)
    apply_loop(
        model,
        LoopSpec(prelude_end=1, core_end=3, num_passes=2, target_virtual_layers=None, gate_init=0.1),
    )

    out_dir = tmp_path / f"pack_{scheme}"
    monkeypatch.setattr(logging.getLogger("qloop"), "propagate", True)
    with caplog.at_level(logging.INFO, logger="qloop"):
        res = export_whole_model_pack(
            model,
            out_dir,
            tokenizer=ToyTokenizer(),
            merge_hot_experts=True,  # requested True on quantized base -> must keep sidecar safely!
        )

    assert res["merged_hot_experts"] == 0
    assert any("Keeping hot_experts unmerged" in rec.message for rec in caplog.records)

    # Verify config.json preserves `quantization_config` and marks `hot_experts.merged_into_base = False`
    cfg_json = json.loads((out_dir / "config.json").read_text(encoding="utf-8"))
    expected_qmethod = "fp8" if scheme == "fp8" else "compressed-tensors"
    assert cfg_json["quantization_config"]["quant_method"] == expected_qmethod
    assert cfg_json["base_config"]["quantization_config"]["quant_method"] == expected_qmethod
    he_cfg = cfg_json["qloop_v3_arch"]["hot_experts"]
    assert he_cfg["enabled"] is True
    assert he_cfg["merged_into_base"] is False
    assert he_cfg["hot_map"] == {"0": [0, 2], "2": [1]}
    assert cfg_json["qloop_v3_arch"]["quantization"]["is_quantized"] is True

    # Inspect exported safetensors shard(s): compact quantized dtypes + all scale buffers + sidecar deltas
    idx_json = json.loads((out_dir / "model.safetensors.index.json").read_text(encoding="utf-8"))
    weight_map: dict[str, str] = idx_json["weight_map"]
    loaded_tensors: dict[str, torch.Tensor] = {}
    for shard_file in sorted(set(weight_map.values())):
        loaded_tensors.update(load_safetensors_file(str(out_dir / shard_file)))

    # Sidecar hot-expert delta keys present in exported pack & matching `_EXPERT_DELTA_KEY_RE`
    ed_keys = sorted(k for k in loaded_tensors if "expert_deltas.deltas." in k)
    assert len(ed_keys) == 3 * 4  # 3 hot experts x 4 LoRA matrices (gate_up_A/B, down_A/B)
    assert all(_EXPERT_DELTA_KEY_RE.search(k) is not None for k in ed_keys)

    if scheme == "fp8":
        gu_t = loaded_tensors["model.layers.0.mlp.experts.gate_up_proj"]
        gu_s = loaded_tensors["model.layers.0.mlp.experts.gate_up_proj_scale_inv"]
        dn_s = loaded_tensors["model.layers.0.mlp.experts.down_proj_scale_inv"]
        lin_w = loaded_tensors["model.layers.0.linear_attn.in_proj_qkv.weight"]
        lin_s = loaded_tensors["model.layers.0.linear_attn.in_proj_qkv.weight_scale_inv"]
        assert gu_t.dtype == torch.float8_e4m3fn
        assert gu_s.dtype == torch.float32
        assert dn_s.dtype == torch.float32
        assert lin_w.dtype == torch.float8_e4m3fn
        assert lin_s.dtype == torch.float32
    else:
        gu_p = loaded_tensors["model.layers.0.mlp.experts.gate_up_proj_packed"]
        gu_s = loaded_tensors["model.layers.0.mlp.experts.gate_up_proj_scale"]
        gu_gs = loaded_tensors["model.layers.0.mlp.experts.gate_up_proj_global_scale"]
        lin_p = loaded_tensors["model.layers.0.linear_attn.in_proj_qkv.weight_packed"]
        lin_s = loaded_tensors["model.layers.0.linear_attn.in_proj_qkv.weight_scale"]
        lin_gs = loaded_tensors["model.layers.0.linear_attn.in_proj_qkv.weight_global_scale"]
        assert gu_p.dtype == torch.uint8
        assert gu_s.dtype == torch.float8_e4m3fn
        assert gu_gs.dtype == torch.float32
        assert lin_p.dtype == torch.uint8
        assert lin_s.dtype == torch.float8_e4m3fn
        assert lin_gs.dtype == torch.float32

    assert detect_model_quantization(model)["is_quantized"] is True


@pytest.mark.parametrize("scheme", ["fp8", "nvfp4"])
def test_export_pack_passthrough_tensors_sidecar_keys_on_quantized_stack(
    scheme: str,
    tmp_path: Any,
) -> None:
    base = build_toy(
        seed=91,
        num_hidden_layers=4,
        hidden_size=32,
        moe_intermediate_size=16,
        shared_expert_intermediate_size=16,
        num_experts=4,
    )
    model = (
        _quantize_toy_model_fp8(base, block_size=(16, 16))
        if scheme == "fp8"
        else _quantize_toy_model_nvfp4(base, group_size=16)
    )
    install_quantized_autograd_bridge(model)

    # Generate canonical `HotExpertDeltas` state_dict keys on a throwaway handle, then export via
    # `Stack.passthrough_tensors` with `deltas=None` (prior-stage checkpoint carry-over).
    ref_deltas = HotExpertDeltas(model, {0: [2], 2: [1]}, rank=4)
    passthrough = {
        f"model.expert_deltas.{k}": v.detach().clone()
        for k, v in ref_deltas.state_dict().items()
    }
    expected_ed_keys = sorted(passthrough.keys())
    assert all(_EXPERT_DELTA_KEY_RE.search(k) is not None for k in expected_ed_keys)

    stack = Stack(
        model=model,
        tokenizer=ToyTokenizer(),
        parts=inspect_model(model),
        deltas=None,
        passthrough_tensors=passthrough,
    )
    out_dir = tmp_path / f"pt_pack_{scheme}"
    res = export_whole_model_pack(stack, out_dir, merge_hot_experts=True)
    assert res["merged_hot_experts"] == 0

    cfg_json = json.loads((out_dir / "config.json").read_text(encoding="utf-8"))
    he_cfg = cfg_json["qloop_v3_arch"]["hot_experts"]
    assert he_cfg["enabled"] is True
    assert he_cfg["merged_into_base"] is False
    assert he_cfg["rank"] == 4
    assert he_cfg["hot_map"] == {"0": [2], "2": [1]}

    idx_json = json.loads((out_dir / "model.safetensors.index.json").read_text(encoding="utf-8"))
    loaded_tensors: dict[str, torch.Tensor] = {}
    for shard_file in sorted(set(idx_json["weight_map"].values())):
        loaded_tensors.update(load_safetensors_file(str(out_dir / shard_file)))

    exported_ed_keys = sorted(k for k in loaded_tensors if "expert_deltas.deltas." in k)
    assert exported_ed_keys == expected_ed_keys
    assert all(_EXPERT_DELTA_KEY_RE.search(k) is not None for k in exported_ed_keys)
    for k in expected_ed_keys:
        assert torch.equal(loaded_tensors[k], passthrough[k])


def test_freeze_all_but_keeps_fp8_and_nvfp4_weights_frozen_and_catches_typos(
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(logging.getLogger("qloop"), "propagate", True)

    # 1) FP8 backbone: `s2_joint_bptt` (`self_attn.(k_proj|v_proj|o_proj|q_norm|k_norm)`),
    #    `s1_loop_heal` / `s2_sft` / `s4_rl` (`shared_expert\.`, `linear_attn\.(in_proj|out_proj)`).
    fp8_model = _quantize_toy_model_fp8(
        build_toy(
            seed=101,
            num_hidden_layers=4,
            hidden_size=32,
            moe_intermediate_size=16,
            shared_expert_intermediate_size=16,
            num_experts=4,
        ),
        block_size=(16, 16),
    )
    sha_fp8_before = sha256_frozen_experts(fp8_model)

    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="qloop"):
        rep_fp8_se = freeze_all_but(fp8_model, [r"shared_expert\."])
    assert rep_fp8_se.n_trainable == 0
    assert any("keeping quantized non-float weights frozen" in r.message for r in caplog.records)
    assert all(
        not p.requires_grad
        for p in fp8_model.parameters()
        if p.dtype == torch.float8_e4m3fn
    )
    assert sha256_frozen_experts(fp8_model) == sha_fp8_before

    rep_fp8_s2 = freeze_all_but(
        fp8_model,
        [
            r"self_attn\.(k_proj|v_proj|o_proj|q_norm|k_norm)",
            r"linear_attn\.(in_proj|out_proj)",
            r"shared_expert\.",
        ],
    )
    # Float `q_norm`, `k_norm`, and `in_proj_ba` are trainable; all FP8 weights remain frozen!
    assert rep_fp8_s2.n_trainable > 0
    for pname, p in fp8_model.named_parameters():
        if p.requires_grad:
            assert p.dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64), pname
        if p.dtype == torch.float8_e4m3fn:
            assert not p.requires_grad, pname
    assert sha256_frozen_experts(fp8_model) == sha_fp8_before

    with pytest.raises(AssertionError, match="matched no parameter"):
        freeze_all_but(fp8_model, [r"nonexistent_typo_pattern_xyz\."])
    with pytest.raises(AssertionError, match="forbidden groups"):
        freeze_all_but(fp8_model, [r"mlp\.experts\."])

    # 2) NVFP4 backbone: `CompressedLinear` has `uint8` `weight_packed` buffers and NO `weight` param.
    nvfp4_model = _quantize_toy_model_nvfp4(
        build_toy(
            seed=102,
            num_hidden_layers=4,
            hidden_size=32,
            moe_intermediate_size=16,
            shared_expert_intermediate_size=16,
            num_experts=4,
        ),
        group_size=16,
    )
    sha_nvfp4_before = sha256_frozen_experts(nvfp4_model)

    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="qloop"):
        rep_nvfp4_se = freeze_all_but(nvfp4_model, [r"shared_expert\."])
    assert rep_nvfp4_se.n_trainable == 0
    assert any("keeping quantized non-float weights frozen" in r.message for r in caplog.records)
    assert sha256_frozen_experts(nvfp4_model) == sha_nvfp4_before

    # Genuine unmatched typo MUST still raise under `strict_patterns=True`!
    with pytest.raises(AssertionError, match="matched no parameter"):
        freeze_all_but(nvfp4_model, [r"nonexistent_typo_pattern_xyz\."])
    with pytest.raises(AssertionError, match="forbidden groups"):
        freeze_all_but(nvfp4_model, [r"mlp\.experts\."])


# ========================================== 7. Missing-`kernels` FP8 Fallback & Chunked-3D MoE

_MISSING_KERNELS_MSG = (
    "finegrained-fp8 kernel unavailable: `kernels` is either not installed or uses an incompatible version. "
    "Please install a compatible version (0.18.2 <= version < 0.19.0), e.g. `pip install kernels==0.18.2`"
)


class MockNoKernelsFP8Linear(nn.Linear):
    """Simulates `transformers` `FP8Linear` on Colab when `kernels` is missing (raises even under `no_grad()`)."""

    def __init__(
        self,
        linear: nn.Linear,
        block_size: tuple[int, int] = (16, 16),
        *,
        scale_attr: str = "weight_scale_inv",
    ) -> None:
        super().__init__(int(linear.in_features), int(linear.out_features), bias=False, device="meta")
        self.weight_block_size = tuple(block_size)
        q_w, s_inv = quantize_fp8_blockwise(linear.weight.detach(), block_size=block_size)
        self.weight = nn.Parameter(q_w, requires_grad=False)
        self.register_buffer(scale_attr, s_inv)
        if linear.bias is not None:
            self.bias = nn.Parameter(linear.bias.detach().clone(), requires_grad=False)
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        del x
        raise ImportError(_MISSING_KERNELS_MSG)


class MockNoKernelsFP8Experts(nn.Module):
    """Simulates `transformers` `FP8Experts` raising `ImportError` unconditionally across 3D/2D scale layouts."""

    def __init__(
        self,
        gate_up: torch.Tensor,
        down: torch.Tensor,
        block_size: tuple[int, int] = (16, 16),
        *,
        flatten_2d_scale: bool = False,
        scale_names: tuple[str, str] = ("gate_up_proj_scale_inv", "down_proj_scale_inv"),
    ) -> None:
        super().__init__()
        self.num_experts = int(gate_up.shape[0])
        self.weight_block_size = tuple(block_size)
        gu_q, gu_s = quantize_fp8_blockwise(gate_up, block_size=block_size)
        dn_q, dn_s = quantize_fp8_blockwise(down, block_size=block_size)
        if flatten_2d_scale:
            gu_s = gu_s.reshape(self.num_experts * gu_s.shape[1], gu_s.shape[2]).contiguous()
            dn_s = dn_s.reshape(self.num_experts * dn_s.shape[1], dn_s.shape[2]).contiguous()
        self.gate_up_proj = nn.Parameter(gu_q, requires_grad=False)
        self.down_proj = nn.Parameter(dn_q, requires_grad=False)
        self.register_buffer(scale_names[0], gu_s)
        self.register_buffer(scale_names[1], dn_s)
        self.act_fn = F.silu

    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        del hidden_states, top_k_index, top_k_weights
        raise ImportError(_MISSING_KERNELS_MSG)


def _quantize_toy_model_fp8_missing_kernels(
    base_model: ToyForCausalLM,
    block_size: tuple[int, int] = (16, 16),
    *,
    flatten_2d_scale: bool = False,
    moe_scale_names: tuple[str, str] = ("gate_up_proj_scale_inv", "down_proj_scale_inv"),
    lin_scale_attr: str = "weight_scale_inv",
) -> ToyForCausalLM:
    m = copy.deepcopy(base_model)
    m.config.quantization_config = {  # type: ignore[attr-defined]
        "quant_method": "fp8",
        "activation_scheme": "dynamic",
        "weight_block_size": list(block_size),
    }
    for layer in m.model.layers:
        if hasattr(layer, "self_attn"):
            sa = layer.self_attn
            for attr in ("q_proj", "k_proj", "v_proj", "o_proj"):
                setattr(sa, attr, MockNoKernelsFP8Linear(getattr(sa, attr), block_size, scale_attr=lin_scale_attr))
        else:
            la = layer.linear_attn
            for attr in ("in_proj_qkv", "in_proj_qkvz", "out_proj"):
                if hasattr(la, attr):
                    setattr(la, attr, MockNoKernelsFP8Linear(getattr(la, attr), block_size, scale_attr=lin_scale_attr))
        se = layer.mlp.shared_expert
        for attr in ("gate_proj", "up_proj", "down_proj"):
            setattr(se, attr, MockNoKernelsFP8Linear(getattr(se, attr), block_size, scale_attr=lin_scale_attr))
        exp = layer.mlp.experts
        layer.mlp.experts = MockNoKernelsFP8Experts(
            exp.gate_up_proj.detach(),
            exp.down_proj.detach(),
            block_size=block_size,
            flatten_2d_scale=flatten_2d_scale,
            scale_names=moe_scale_names,
        )
    for p in m.parameters():
        p.requires_grad_(False)
    return m


@pytest.mark.parametrize(
    ("flatten_2d_scale", "moe_scale_names", "lin_scale_attr"),
    [
        (False, ("gate_up_proj_scale_inv", "down_proj_scale_inv"), "weight_scale_inv"),
        (True, ("gate_up_proj_scale_inv", "down_proj_scale_inv"), "weight_scale_inv"),
        (True, ("weight_scale_up", "weight_scale_down"), "weight_scale_inv"),
        (False, ("weight_scale_inv_up", "weight_scale_inv_down"), "scale_inv"),
        (True, ("custom_gu_blk_buf", "custom_dn_blk_buf"), "custom_lin_blk_buf"),
    ],
)
def test_fp8_missing_kernels_nograd_fallback_probe_suite_and_backward(
    flatten_2d_scale: bool,
    moe_scale_names: tuple[str, str],
    lin_scale_attr: str,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Exact Colab Blackwell `Qwen3.6-35B-A3B-FP8` crash repro (`ImportError` under `torch.no_grad()`)."""
    monkeypatch.setattr(logging.getLogger("qloop"), "propagate", True)
    base_float = build_toy(
        seed=303,
        num_hidden_layers=6,
        hidden_size=32,
        moe_intermediate_size=16,
        shared_expert_intermediate_size=16,
        num_experts=6,
        num_experts_per_tok=2,
        layer_types=(
            "linear_attention",
            "full_attention",
            "linear_attention",
            "full_attention",
            "linear_attention",
            "full_attention",
        ),
    ).eval()
    tok = ToyTokenizer()
    inp = torch.randint(4, 96, (2, 12))
    with torch.no_grad():
        float_ref_logits = base_float(inp).logits.detach().clone()

    raw_fp8 = _quantize_toy_model_fp8_missing_kernels(
        base_float,
        block_size=(16, 16),
        flatten_2d_scale=flatten_2d_scale,
        moe_scale_names=moe_scale_names,
        lin_scale_attr=lin_scale_attr,
    )
    # Without bridge, `torch.no_grad()` immediately crashes with `ImportError: finegrained-fp8 kernel unavailable`
    with pytest.raises(ImportError, match="finegrained-fp8 kernel unavailable"), torch.no_grad():
        raw_fp8(inp)

    # Load through `load_model` -> installs bridge, records `native_fwd_ok_count == 0`, and emits clear hint!
    monkeypatch.setattr("qloop.model.loading._load_hf", lambda _cfg: (raw_fp8, tok))
    caplog.clear()
    with caplog.at_level(logging.INFO, logger="qloop"):
        model, _loaded_tok, info = load_model({"name": "Qwen/Qwen3.6-35B-A3B-FP8", "device_map": "cpu"})

    qinfo = info["quantization"]
    assert qinfo["is_quantized"] is True
    assert qinfo["native_fwd_ok_count"] == 0
    assert qinfo["native_grad_ok_count"] == 0
    assert qinfo["bridged_total"] == qinfo["total_quantized_modules"]
    assert any("native_fwd_ok=0" in r.message for r in caplog.records)
    assert any(
        r.levelno == logging.WARNING and 'pip install "kernels>=0.18.2,<0.19.0"' in r.message
        for r in caplog.records
    )

    # 1) `torch.no_grad()` forward succeeds cleanly and matches float reference (`< 1e-2` FP8 quant error)
    with torch.no_grad():
        nograd_logits = model(inp).logits
        nograd_logits_2 = model(inp).logits
    assert torch.isfinite(nograd_logits).all()
    assert torch.equal(nograd_logits, nograd_logits_2)  # exact determinism required by probe suite!
    # Also compute exact dequantized-FP8 float reference to verify `< 1e-5` bridge fidelity
    deq_ref_model = copy.deepcopy(base_float).eval()
    with torch.no_grad():
        for ( _n, dm), (_qn, qm) in zip(
            [(n, m) for n, m in deq_ref_model.named_modules()],
            [(n, m) for n, m in model.named_modules()],
        ):
            if isinstance(qm, MockNoKernelsFP8Linear) and isinstance(dm, nn.Linear):
                dm.weight.copy_(qa._dequant_linear_weight(qm, dtype=dm.weight.dtype))
            elif isinstance(qm, MockNoKernelsFP8Experts) and hasattr(dm, "gate_up_proj"):
                for ei in range(qm.num_experts):
                    gue, dne = auto_dequant_expert_fn(qm, ei)
                    dm.gate_up_proj[ei].copy_(gue)
                    dm.down_proj[ei].copy_(dne)
        exact_fp8_ref_logits = deq_ref_model(inp).logits
    assert (nograd_logits - exact_fp8_ref_logits).abs().max().item() < 1e-5
    # Also check bfloat16 Tensor Core path (`< 1e-2` vs bfloat16 dequantized reference)
    model_bf16 = copy.deepcopy(model)
    deq_ref_bf16 = copy.deepcopy(deq_ref_model)
    for m_cast in (model_bf16, deq_ref_bf16):
        for p_b in m_cast.parameters():
            if p_b.is_floating_point() and p_b.dtype != torch.float8_e4m3fn:
                p_b.data = p_b.data.to(torch.bfloat16)
    with torch.no_grad():
        bf16_logits = model_bf16(inp).logits.float()
        bf16_deq_ref = deq_ref_bf16(inp).logits.float()
    assert model_bf16(inp).logits.dtype == torch.bfloat16
    assert (bf16_logits - bf16_deq_ref).abs().max().item() < 1e-2
    cos = F.cosine_similarity(nograd_logits.reshape(-1), float_ref_logits.reshape(-1), dim=0).item()
    assert cos > 0.95

    # 2) Full `run_structured_probe_suite` (where Colab crashed at `ref_logits = model(input_ids=b0)`)
    if flatten_2d_scale and moe_scale_names == ("gate_up_proj_scale_inv", "down_proj_scale_inv"):
        suite_out = run_structured_probe_suite(
            model,
            [inp],
            target_virtual=64,
            n_perturb=2,
            subspace_rank=16,
            decode_steps=3,
        )
        assert suite_out["summary"]["all_critical_ok"] is True
        assert suite_out["gates"]["0a_backward"]["passed"] is True
        assert suite_out["gates"]["0a_surgery"]["passed"] is True

    # 3) Autograd backward propagates exact `dL/dh` and `dL/d(top_k_weights)` through quantized layers!
    x_grad = torch.randn(3, 32, requires_grad=True)
    top_idx = torch.tensor([[0, 3], [2, 5], [1, 4]], dtype=torch.long)
    top_w = torch.softmax(torch.randn(3, 2), dim=-1).detach().requires_grad_(True)
    qlin0 = model.model.layers[0].mlp.shared_expert.gate_proj
    qmoe0 = model.model.layers[0].mlp.experts
    y_lin = qlin0(x_grad)
    y_moe = qmoe0(x_grad, top_idx, top_w)
    (y_lin.pow(2).mean() + y_moe.pow(2).mean()).backward()

    x_ref = x_grad.detach().clone().requires_grad_(True)
    w_ref = top_w.detach().clone().requires_grad_(True)
    dlin0 = deq_ref_model.model.layers[0].mlp.shared_expert.gate_proj
    dmoe0 = deq_ref_model.model.layers[0].mlp.experts
    y_lin_r = dlin0(x_ref)
    y_moe_r = torch.zeros_like(x_ref)
    for eid in range(6):
        t_i, s_i = (top_idx == eid).nonzero(as_tuple=True)
        if t_i.numel() > 0:
            gt, up = F.linear(x_ref[t_i], dmoe0.gate_up_proj[eid]).chunk(2, dim=-1)
            ye = F.linear(F.silu(gt) * up, dmoe0.down_proj[eid])
            y_moe_r.index_add_(0, t_i, ye * w_ref[t_i, s_i, None])
    (y_lin_r.pow(2).mean() + y_moe_r.pow(2).mean()).backward()

    assert x_grad.grad is not None and x_ref.grad is not None
    assert top_w.grad is not None and w_ref.grad is not None
    assert (y_lin - y_lin_r).abs().max().item() < 1e-5
    assert (y_moe - y_moe_r).abs().max().item() < 1e-5
    assert (x_grad.grad - x_ref.grad).abs().max().item() < 1e-5
    assert (top_w.grad - w_ref.grad).abs().max().item() < 1e-5


def test_dynamic_runtime_kernel_failure_fallback_nvfp4_warning_and_unscaled_fp8_guard(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Verify runtime kernel failure recovery, NVFP4 warning hint, and missing-scale FP8 RuntimeError."""
    monkeypatch.setattr(logging.getLogger("qloop"), "propagate", True)
    torch.manual_seed(404)

    # 1) Dynamic runtime `ImportError` after probe initially succeeded (`native_fwd_ok=True`)
    fail_flag = {"fail": False}

    class FlakyRuntimeFP8Linear(MockFineGrainedFP8Linear):
        @torch.no_grad()
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            if fail_flag["fail"]:
                raise ImportError("lazy runtime kernel load failed mid-run")
            return super().forward(x)

    class FlakyRuntimeFP8Experts(MockFP8FusedExperts):
        @torch.no_grad()
        def forward(
            self,
            hidden_states: torch.Tensor,
            top_k_index: torch.Tensor,
            top_k_weights: torch.Tensor,
        ) -> torch.Tensor:
            if fail_flag["fail"]:
                raise NotImplementedError("sm_120 runtime shape unsupported")
            return super().forward(hidden_states, top_k_index, top_k_weights)

    lin_a = FlakyRuntimeFP8Linear(nn.Linear(32, 24, bias=False), block_size=(16, 16))
    lin_b = FlakyRuntimeFP8Linear(nn.Linear(32, 24, bias=False), block_size=(16, 16))
    gu = torch.randn(4, 32, 32) * 0.2
    dn = torch.randn(4, 32, 16) * 0.2
    moe_a = FlakyRuntimeFP8Experts(gu, dn, block_size=(16, 16))
    box = nn.ModuleDict({"la": lin_a, "lb": lin_b, "ma": moe_a})

    rep = install_quantized_autograd_bridge(box)
    assert rep["native_fwd_ok_count"] == 3
    assert lin_a._qloop_native_fwd_ok is True  # type: ignore[attr-defined]
    assert moe_a._qloop_native_fwd_ok is True  # type: ignore[attr-defined]

    x = torch.randn(4, 32)
    idx = torch.tensor([[0, 2], [1, 3], [2, 0], [3, 1]], dtype=torch.long)
    tw = torch.softmax(torch.randn(4, 2), dim=-1)
    with torch.no_grad():
        ref_la = lin_a(x)
        ref_lb = lin_b(x)
        ref_ma = moe_a(x, idx, tw)

    # Now trigger runtime kernel failure -> must catch, flip `_qloop_native_fwd_ok=False`, warn ONCE per class!
    fail_flag["fail"] = True
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="qloop"), torch.no_grad():
        fb_la = lin_a(x)
        fb_lb = lin_b(x)
        fb_ma = moe_a(x, idx, tw)
    assert lin_a._qloop_native_fwd_ok is False  # type: ignore[attr-defined]
    assert lin_b._qloop_native_fwd_ok is False  # type: ignore[attr-defined]
    assert moe_a._qloop_native_fwd_ok is False  # type: ignore[attr-defined]
    assert torch.allclose(fb_la, ref_la, atol=1e-5, rtol=1e-5)
    assert torch.allclose(fb_lb, ref_lb, atol=1e-5, rtol=1e-5)
    assert torch.allclose(fb_ma, ref_ma, atol=1e-5, rtol=1e-5)
    lin_warns = [r for r in caplog.records if "FlakyRuntimeFP8Linear" in r.message]
    moe_warns = [r for r in caplog.records if "FlakyRuntimeFP8Experts" in r.message]
    assert len(lin_warns) == 1  # deduplicated across `lin_a` & `lin_b`!
    assert len(moe_warns) == 1

    # 2) NVFP4 `load_model` missing-kernel warning advises `pip install compressed-tensors`
    nvfp4_broken = _quantize_toy_model_nvfp4(build_toy(seed=405, num_hidden_layers=2, hidden_size=32), group_size=16)
    for mod in nvfp4_broken.modules():
        if isinstance(mod, (MockCompressedNVFP4Linear, MockNVFP4FusedExperts)):
            def _raise_nvfp4(*_a: Any, **_kw: Any) -> Any:
                raise ImportError("compressed_tensors kernel missing")
            mod.forward = _raise_nvfp4  # type: ignore[method-assign]
    monkeypatch.setattr("qloop.model.loading._load_hf", lambda _cfg: (nvfp4_broken, ToyTokenizer()))
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="qloop"):
        _, _, nv_info = load_model({"name": "unsloth/Qwen3.6-35B-A3B-NVFP4", "device_map": "cpu"})
    assert nv_info["quantization"]["native_fwd_ok_count"] == 0
    assert any("pip install compressed-tensors" in r.message for r in caplog.records)

    # 3) Missing FP8 scale guard: `dequantize_fp8_tensor(w, None)` stays identity, while module-level raises!
    w_raw = torch.randn(16, 16)
    w_fp8 = w_raw.to(torch.float8_e4m3fn)
    assert torch.equal(dequantize_fp8_tensor(w_fp8, None, dtype=torch.float32), w_fp8.float())

    unscaled_lin = MockNoKernelsFP8Linear(nn.Linear(16, 16, bias=False), block_size=(16, 16))
    unscaled_lin._buffers.pop("weight_scale_inv", None)
    install_quantized_autograd_bridge(unscaled_lin)
    with pytest.raises(RuntimeError, match=r"(no|missing) scale tensor.*parameters="):
        unscaled_lin(torch.randn(2, 16))
    unscaled_lin._qloop_allow_unscaled_fp8 = True  # type: ignore[attr-defined]
    out_allowed = unscaled_lin(torch.randn(2, 16))
    assert out_allowed.shape == (2, 16)

    unscaled_moe = MockNoKernelsFP8Experts(torch.randn(3, 16, 16), torch.randn(3, 16, 8), block_size=(16, 16))
    unscaled_moe._buffers.pop("down_proj_scale_inv", None)
    install_quantized_autograd_bridge(unscaled_moe)
    with pytest.raises(RuntimeError, match=r"missing scale tensor.*parameters="):
        unscaled_moe(torch.randn(2, 16), torch.tensor([[0, 1], [1, 2]]), torch.ones(2, 2) * 0.5)


@pytest.mark.parametrize(
    "variant",
    ["fp8_3d", "fp8_flat2d", "nvfp4"],
)
def test_chunked_3d_moe_dispatch_exact_parity_across_40_experts_with_padding_and_dups(
    variant: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify batched chunked-3D (`E=40 > 32`) MoE forward + backward parity (`< 1e-5`) with `-1` & dup slots."""
    torch.manual_seed(505)
    num_experts, h_dim, inter_dim = 40, 32, 16
    n_tokens, top_k = 29, 4
    gu_ref = torch.randn(num_experts, 2 * inter_dim, h_dim) / math.sqrt(h_dim)
    dn_ref = torch.randn(num_experts, h_dim, inter_dim) / math.sqrt(inter_dim)

    if variant == "fp8_3d":
        qmoe: nn.Module = MockNoKernelsFP8Experts(gu_ref, dn_ref, block_size=(16, 16), flatten_2d_scale=False)
    elif variant == "fp8_flat2d":
        qmoe = MockNoKernelsFP8Experts(gu_ref, dn_ref, block_size=(16, 16), flatten_2d_scale=True)
    else:
        qmoe = MockNVFP4FusedExperts(gu_ref, dn_ref, group_size=16)
        def _fail_nvfp4(*_a: Any, **_kw: Any) -> Any:
            raise ImportError("force fallback")
        qmoe.forward = _fail_nvfp4  # type: ignore[method-assign]

    rep = install_quantized_autograd_bridge(qmoe)
    assert rep["native_fwd_ok_count"] == 0
    assert rep["bridged_moe_count"] == 1

    # Construct routing covering 37 unique experts (> 32 chunk boundary -> 2 chunks: 32 + 5!),
    # PLUS `-1` masked padding slots AND duplicate expert IDs on the same token!
    top_idx = torch.arange(n_tokens * top_k, dtype=torch.long).reshape(n_tokens, top_k) % 37
    top_idx[0] = torch.tensor([7, 7, 35, -1], dtype=torch.long)  # duplicate expert 7 + `-1` masked slot!
    top_idx[5] = torch.tensor([-1, -1, 19, 36], dtype=torch.long)
    top_idx[11] = torch.tensor([-1, -1, -1, -1], dtype=torch.long)  # fully masked token!
    top_idx[17] = torch.tensor([33, 33, 33, 2], dtype=torch.long)  # triple duplicate expert 33!
    raw_w = torch.rand(n_tokens, top_k, dtype=torch.float32) * 0.6 + 0.1

    # Spy on `_dequant_experts_chunk` to verify chunking (`[32, 5]`) and 0 unused experts
    chunk_lens: list[int] = []
    seen_eids: set[int] = set()
    orig_chunk_fn = qa._dequant_experts_chunk

    def _spy_chunk_40(
        mod: nn.Module,
        expert_ids: list[int],
        dtype: torch.dtype = torch.bfloat16,
        *,
        nvfp4_mode_override: str | None = None,
    ) -> tuple[torch.Tensor | list[torch.Tensor], torch.Tensor | list[torch.Tensor]]:
        chunk_lens.append(len(expert_ids))
        seen_eids.update(expert_ids)
        return orig_chunk_fn(mod, expert_ids, dtype=dtype, nvfp4_mode_override=nvfp4_mode_override)

    monkeypatch.setattr(qa, "_dequant_experts_chunk", _spy_chunk_40)

    x_base = torch.randn(n_tokens, h_dim, dtype=torch.float32)
    with torch.no_grad():
        y_nograd = qmoe(x_base, top_idx, raw_w)
    assert chunk_lens == [32, 5]
    assert 37 not in seen_eids and 38 not in seen_eids and 39 not in seen_eids

    chunk_lens.clear()
    x_grad = x_base.detach().clone().requires_grad_(True)
    w_grad = raw_w.detach().clone().requires_grad_(True)
    y_grad = qmoe(x_grad, top_idx, w_grad)
    loss_target = torch.linspace(-0.7, 1.1, steps=y_grad.numel(), dtype=torch.float32).reshape_as(y_grad)
    loss = (y_grad * loss_target).sum()
    loss.backward()
    assert chunk_lens == [32, 5, 32, 5]  # 2 chunks in forward + 2 chunks in backward

    # Naive per-(token, slot) SwiGLU reference in fp32
    x_ref = x_base.detach().clone().requires_grad_(True)
    w_ref = raw_w.detach().clone().requires_grad_(True)
    deq_cache = {e: auto_dequant_expert_fn(qmoe, e) for e in range(37)}
    y_ref = torch.zeros_like(x_ref)
    for t in range(n_tokens):
        acc_t = torch.zeros(h_dim, dtype=torch.float32)
        for s in range(top_k):
            eid = int(top_idx[t, s].item())
            if 0 <= eid < num_experts:
                gu_e, dn_e = deq_cache[eid]
                gt, up = F.linear(x_ref[t : t + 1], gu_e).chunk(2, dim=-1)
                ye = F.linear(F.silu(gt) * up, dn_e).squeeze(0)
                acc_t = acc_t + ye * w_ref[t, s]
        y_ref[t] = acc_t
    (y_ref * loss_target).sum().backward()

    assert x_grad.grad is not None and x_ref.grad is not None
    assert w_grad.grad is not None and w_ref.grad is not None
    assert (y_nograd - y_ref.detach()).abs().max().item() < 1e-5
    assert (y_grad.detach() - y_ref.detach()).abs().max().item() < 1e-5
    assert (x_grad.grad - x_ref.grad).abs().max().item() < 1e-5
    assert (w_grad.grad - w_ref.grad).abs().max().item() < 1e-5
    # Masked `-1` slots MUST receive exact 0.0 gradient!
    assert float(w_grad.grad[0, 3].abs().item()) == 0.0
    assert torch.all(w_grad.grad[11] == 0.0)


def test_standalone_whole_model_pack_fp8_missing_kernels_safety_both_paths(
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify both `_HAS_QLOOP=True` and emitted `_HAS_QLOOP=False` packs survive missing `kernels`."""
    tok = ToyTokenizer()
    tcfg = ToyConfig(
        vocab_size=tok.vocab_size,
        hidden_size=32,
        moe_intermediate_size=16,
        shared_expert_intermediate_size=16,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        partial_rotary_factor=0.5,
        linear_num_heads=4,
        linear_head_dim=8,
        linear_conv_kernel_dim=4,
        num_experts=4,
        num_experts_per_tok=2,
        layer_types=("linear_attention", "full_attention", "linear_attention", "full_attention"),
    )
    torch.manual_seed(606)
    base_float = ToyForCausalLM(tcfg).eval()
    fp8_bb = _quantize_toy_model_fp8_missing_kernels(
        base_float,
        block_size=(16, 16),
        flatten_2d_scale=True,
        moe_scale_names=("weight_scale_up", "weight_scale_down"),
    )
    install_quantized_autograd_bridge(fp8_bb)
    loop_spec = LoopSpec(
        prelude_end=1,
        core_end=3,
        num_passes=3,
        target_virtual_layers=None,
        per_pass_router=True,
        router_adapter_rank=4,
    )
    loop_ctrl = apply_loop(fp8_bb, loop_spec)
    with torch.no_grad():
        loop_ctrl.adapters.gates.copy_(torch.tensor([[0.35], [-0.25]]).expand_as(loop_ctrl.adapters.gates))
        assert loop_ctrl.adapters.pass_router is not None
        loop_ctrl.adapters.pass_router.up.normal_(0.0, 0.2)

    pack_dir = tmp_path / "fp8_nokernels_pack"
    stack = Stack(
        model=fp8_bb,
        tokenizer=tok,
        parts=inspect_model(fp8_bb),
        loop=loop_ctrl,
        load_info={"name": "Qwen/Qwen3.6-35B-A3B-FP8"},
    )
    export_whole_model_pack(stack, pack_dir, max_shard_size_bytes=65536)

    # Mark exported config as non-toy HF backbone so both `_HAS_QLOOP=True` and `_HAS_QLOOP=False`
    # construct via `AutoModelForCausalLM.from_config(...)` returning a fresh `MockNoKernelsFP8` backbone!
    cfg_path = pack_dir / "config.json"
    cfg_data = json.loads(cfg_path.read_text(encoding="utf-8"))
    cfg_data["base_config"]["model_type"] = "qwen3_5_moe"
    cfg_data["qloop_v3_arch"]["backbone_kind"] = "hf"
    cfg_data["qloop_v3_arch"]["base_model_type"] = "qwen3_5_moe"
    cfg_path.write_text(json.dumps(cfg_data, indent=2), encoding="utf-8")
    write_standalone_hf_modeling_files(pack_dir)
    for py_fn in (CONFIG_FILENAME, MODELING_FILENAME):
        ast.parse((pack_dir / py_fn).read_text(encoding="utf-8"), filename=py_fn, feature_version=(3, 9))

    def _fresh_fp8_nokernels_backbone() -> ToyForCausalLM:
        return _quantize_toy_model_fp8_missing_kernels(
            ToyForCausalLM(tcfg).eval(),
            block_size=(16, 16),
            flatten_2d_scale=True,
            moe_scale_names=("weight_scale_up", "weight_scale_down"),
        )

    stub_tf = types.ModuleType("transformers")
    stub_tf.__version__ = "5.23.0"  # type: ignore[attr-defined]

    class _PreTrainedConfig:
        model_type = "qwen36_loophssd"

        def __init__(self, **kw: Any) -> None:
            for k, v in kw.items():
                setattr(self, k, v)

        @classmethod
        def from_pretrained(cls, local_dir: Any, **_kw: Any) -> Any:
            raw = json.loads((pathlib.Path(local_dir) / "config.json").read_text(encoding="utf-8"))
            return cls(**raw)

    class _PreTrainedModel(nn.Module):
        config_class: Any = None

        def __init__(self, config: Any) -> None:
            super().__init__()
            self.config = config

        def post_init(self) -> None:
            pass

    class _GenerationMixin:
        pass

    class _AutoConfig:
        @staticmethod
        def for_model(_model_type: str, **kw: Any) -> ToyConfig:
            valid = {f for f in ToyConfig.__dataclass_fields__ if f in kw}
            clean = {k: kw[k] for k in valid}
            if isinstance(clean.get("layer_types"), list):
                clean["layer_types"] = tuple(clean["layer_types"])
            return ToyConfig(**clean)

    class _AutoModelForCausalLM:
        @staticmethod
        def from_config(_cfg: Any, **_kw: Any) -> ToyForCausalLM:
            return _fresh_fp8_nokernels_backbone()

    stub_tf.PretrainedConfig = _PreTrainedConfig  # type: ignore[attr-defined]
    stub_tf.PreTrainedConfig = _PreTrainedConfig  # type: ignore[attr-defined]
    stub_tf.PreTrainedModel = _PreTrainedModel  # type: ignore[attr-defined]
    stub_tf.GenerationMixin = _GenerationMixin  # type: ignore[attr-defined]
    stub_tf.AutoConfig = _AutoConfig  # type: ignore[attr-defined]
    stub_tf.AutoModelForCausalLM = _AutoModelForCausalLM  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "transformers", stub_tf)

    prompt = torch.tensor([[7, 19, 31, 43, 55, 67, 79, 91, 103, 115]], dtype=torch.long)
    orig_refs: dict[int, torch.Tensor] = {}
    with torch.no_grad():
        for r in (1, 2, 3):
            loop_ctrl.set_active_passes(r)
            orig_refs[r] = fp8_bb(prompt, use_cache=False).logits.detach().clone()

    # Path A: `_HAS_QLOOP=True` `Qwen36LoopHSSDForCausalLM.from_pretrained`
    qloop_loaded = Qwen36LoopHSSDForCausalLM.from_pretrained(pack_dir, strict=True)
    with torch.no_grad():
        for r in (1, 2, 3):
            with qloop_loaded.depth(r):
                out_q = qloop_loaded(prompt, use_cache=False).logits
                assert (out_q - orig_refs[r]).abs().max().item() < 1e-5

    # Path B: Emitted standalone `modeling_qwen36_loophssd.py` with `qloop` hidden (`_HAS_QLOOP=False`)
    real_find_spec = importlib.util.find_spec
    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda name, *a, **kw: (
            None if (name == "qloop" or name.startswith("qloop.")) else real_find_spec(name, *a, **kw)
        ),
    )
    for k_mod in [k for k in list(sys.modules) if k == "qloop" or k.startswith("qloop.")]:
        monkeypatch.delitem(sys.modules, k_mod, raising=False)

    pkg_name = "_standalone_fp8_nokernels_pkg"
    pkg = types.ModuleType(pkg_name)
    pkg.__path__ = [str(pack_dir)]  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, pkg_name, pkg)

    spec_c = importlib.util.spec_from_file_location(
        f"{pkg_name}.configuration_qwen36_loophssd",
        pack_dir / CONFIG_FILENAME,
    )
    assert spec_c is not None and spec_c.loader is not None
    mod_c = importlib.util.module_from_spec(spec_c)
    monkeypatch.setitem(sys.modules, spec_c.name, mod_c)
    spec_c.loader.exec_module(mod_c)

    spec_m = importlib.util.spec_from_file_location(
        f"{pkg_name}.modeling_qwen36_loophssd",
        pack_dir / MODELING_FILENAME,
    )
    assert spec_m is not None and spec_m.loader is not None
    mod_m = importlib.util.module_from_spec(spec_m)
    monkeypatch.setitem(sys.modules, spec_m.name, mod_m)
    spec_m.loader.exec_module(mod_m)
    assert mod_m._HAS_QLOOP is False

    standalone_loaded = mod_m.Qwen36LoopHSSDForCausalLM.from_pretrained(str(pack_dir), strict=True)
    with torch.no_grad():
        for r in (1, 2, 3):
            with standalone_loaded.depth(r):
                out_st = standalone_loaded(prompt, use_cache=False).logits
                assert (out_st - orig_refs[r]).abs().max().item() < 1e-5
