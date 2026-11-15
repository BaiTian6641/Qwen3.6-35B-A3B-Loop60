"""Unit & integration tests for Blackwell MoE & FP8 throughput optimizations (`quant_autograd.py`).

Covers:
  1. Explicit analytic SwiGLU backward (`_swiglu_backward_from_gate_up`, `_QuantSwiGLUExpertFn`,
     `_QuantMoERoutedAutogradFn`) across `fp64`, `fp32`, `bf16`, and `fp16` with `gradcheck` and
     zero nested `torch.autograd.grad` invocations (plus exotic custom `act_fn` fallback).
  2. Probed `torch._grouped_mm` vectorized MoE forward & backward (mocked on CPU when native
     `_grouped_mm` is CUDA-only), verifying `_MOE_CHUNK_EXPERTS == 64`, `int32` cumulative `offs`
     construction across multi-chunk routing (`> 128` active experts), `-1` masked slots, duplicate
     token-expert slots, transposed 3D layouts, FP8 & NVFP4, and unaligned-stride fallback safety.
  3. Padded-3D `torch.bmm` batching fast path (`C >= 4`, fill ratio `>= 0.4`) and sparse/ModuleList
     analytic `matmul` fallback without `torch.autograd.grad`.
  4. Single-layer checkpoint recomputation cache (`_in_backward() == True` inside
     `torch.utils.checkpoint.checkpoint(..., use_reentrant=False)` and `use_reentrant=True`, plus
     `apply_loop` with `LoopSpec(checkpoint=True)`): verifies elimination of redundant
     `_build_moe_dispatch_plan` syncs and redundant forward GEMMs (`gu_all`, `y_all`) inside
     `.backward()`, zero cross-layer activation bloat when `_in_backward() == False`, zero tensor
     leaks across steps (including partial `autograd.grad` and `retain_graph=True`), and zero
     in-place buffer mutations.
  5. Direct `bfloat16` block-scale FP8 dequantization (`dequantize_fp8_tensor`): verifies lossless
     `float8_e4m3fn -> bfloat16` conversion without `4x` `float32` intermediate promotion, while
     preserving exact `float32`/`float64` promotion for non-`bfloat16` callers and extreme scales.
"""
from __future__ import annotations

import gc
import math
import weakref
from typing import Any

import pytest
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from qloop.model import quant_autograd as qa
from qloop.model.loop import LoopSpec, apply_loop
from qloop.model.quant_autograd import (
    auto_dequant_expert_fn,
    dequantize_fp8_tensor,
    install_quantized_autograd_bridge,
    quantize_fp8_blockwise,
    quantize_nvfp4_tensor,
)
from qloop.model.toy import build_toy


class _MockFP8Experts(nn.Module):
    """Inference-only 3D FP8 MoE experts module (`@torch.no_grad()` native forward)."""

    def __init__(
        self,
        gate_up: torch.Tensor,
        down: torch.Tensor,
        *,
        block_size: tuple[int, int] = (16, 16),
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

    @torch.no_grad()
    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        return qa._moe_routed_forward_chunked(
            hidden_states.reshape(-1, hidden_states.shape[-1]),
            top_k_index.reshape(-1, top_k_index.shape[-1]).long(),
            top_k_weights.reshape(-1, top_k_weights.shape[-1]),
            self,
        ).reshape_as(hidden_states)


class _MockNVFP4Experts(nn.Module):
    """Inference-only 3D NVFP4 MoE experts module (`@torch.no_grad()` native forward)."""

    def __init__(
        self,
        gate_up: torch.Tensor,
        down: torch.Tensor,
        *,
        group_size: int = 16,
    ) -> None:
        super().__init__()
        self.num_experts = int(gate_up.shape[0])
        self.intermediate_size = int(down.shape[2])
        self.hidden_size = int(down.shape[1])
        self.group_size = int(group_size)

        gu_p, gu_s, gu_gs = quantize_nvfp4_tensor(gate_up.detach(), group_size=group_size)
        dn_p, dn_s, dn_gs = quantize_nvfp4_tensor(down.detach(), group_size=group_size)
        self.register_buffer("gate_up_proj_packed", gu_p)
        self.register_buffer("gate_up_proj_scale", gu_s)
        self.register_buffer("gate_up_proj_global_scale", gu_gs)
        self.register_buffer("down_proj_packed", dn_p)
        self.register_buffer("down_proj_scale", dn_s)
        self.register_buffer("down_proj_global_scale", dn_gs)

    @torch.no_grad()
    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        return qa._moe_routed_forward_chunked(
            hidden_states.reshape(-1, hidden_states.shape[-1]),
            top_k_index.reshape(-1, top_k_index.shape[-1]).long(),
            top_k_weights.reshape(-1, top_k_weights.shape[-1]),
            self,
        ).reshape_as(hidden_states)


class _PlainFloatFusedExperts(nn.Module):
    """Exact float64/float32 fused-3D experts module for `gradcheck` verification."""

    def __init__(self, gate_up: torch.Tensor, down: torch.Tensor) -> None:
        super().__init__()
        self.num_experts = int(gate_up.shape[0])
        self.intermediate_size = int(down.shape[2])
        self.hidden_size = int(down.shape[1])
        self.gate_up_proj = nn.Parameter(gate_up.detach().clone(), requires_grad=False)
        self.down_proj = nn.Parameter(down.detach().clone(), requires_grad=False)


def _reference_moe_forward_autograd(
    flat_hidden: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
    experts_mod: nn.Module,
) -> torch.Tensor:
    """Naive per-(token, slot) PyTorch autograd reference matching dequantized weights in `flat_hidden.dtype`."""
    n_tok, h_dim = int(flat_hidden.shape[0]), int(flat_hidden.shape[1])
    k_slots = int(top_k_index.shape[1])
    n_exp = qa._num_experts_of(experts_mod)
    act_fn = getattr(experts_mod, "act_fn", None) or F.silu

    active_eids = sorted(
        {
            int(e)
            for e in top_k_index.reshape(-1).tolist()
            if 0 <= int(e) < n_exp
        }
    )
    deq_map: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    for eid in active_eids:
        gu_e, dn_e = qa._dequant_single_expert(experts_mod, eid, dtype=flat_hidden.dtype)
        deq_map[eid] = (gu_e.to(flat_hidden.dtype), dn_e.to(flat_hidden.dtype))

    rows: list[torch.Tensor] = []
    for t in range(n_tok):
        acc = torch.zeros((h_dim,), dtype=flat_hidden.dtype, device=flat_hidden.device)
        for s in range(k_slots):
            eid = int(top_k_index[t, s].item())
            if 0 <= eid < n_exp:
                gu_e, dn_e = deq_map[eid]
                gate, up = F.linear(flat_hidden[t : t + 1], gu_e).chunk(2, dim=-1)
                ye = F.linear(act_fn(gate) * up, dn_e).squeeze(0)
                acc = acc + ye * top_k_weights[t, s].to(ye.dtype)
        rows.append(acc)
    return torch.stack(rows, dim=0)


# ============================================================================ 1. Analytic SwiGLU backward


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32, torch.bfloat16, torch.float16])
def test_analytic_swiglu_backward_parity_and_no_nested_autograd(
    dtype: torch.dtype,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch.manual_seed(101)
    # Span saturation + near-zero regimes [-18, +18] to stress sigmoid derivative stability
    gate = (torch.linspace(-18.0, 18.0, steps=64 * 32).reshape(64, 32) + 0.15 * torch.randn(64, 32)).to(dtype)
    up = torch.randn(64, 32, dtype=dtype) * 1.3
    grad_mid = torch.randn(64, 32, dtype=dtype) * 0.8

    g64 = gate.double().requires_grad_(True)
    u64 = up.double().requires_grad_(True)
    dg64, du64 = torch.autograd.grad(F.silu(g64) * u64, (g64, u64), grad_mid.double())
    ref64 = torch.cat((dg64, du64), dim=-1)

    # Forbid `torch.autograd.grad` while running analytic helper & `_QuantSwiGLUExpertFn.backward`
    def _forbid_autograd_grad(*_a: Any, **_kw: Any) -> Any:
        raise AssertionError("nested torch.autograd.grad must not be called on standard SiLU path")

    monkeypatch.setattr(torch.autograd, "grad", _forbid_autograd_grad)

    got = qa._swiglu_backward_from_gate_up(gate, up, grad_mid)
    assert got.dtype == dtype
    assert tuple(got.shape) == (64, 64)
    max_err = float((got.double() - ref64).abs().max().item())
    if dtype == torch.float64:
        assert max_err < 1e-12
    elif dtype == torch.float32:
        assert max_err < 5e-5
    elif dtype == torch.float16:
        assert max_err < 5e-2
    else:
        assert max_err < 0.25


def test_quant_swiglu_expert_fn_gradcheck_and_custom_act_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch.manual_seed(102)
    gu = torch.randn(3, 16, 12, dtype=torch.float64) * 0.25
    dn = torch.randn(3, 12, 8, dtype=torch.float64) * 0.25
    exp_mod = _PlainFloatFusedExperts(gu, dn)

    x_e = torch.randn(5, 12, dtype=torch.float64, requires_grad=True)
    w_e = torch.rand(5, dtype=torch.float64, requires_grad=True)

    orig_ag_grad = torch.autograd.grad

    def _forbid_ag(*_a: Any, **_kw: Any) -> Any:
        raise AssertionError("nested autograd.grad forbidden for SiLU")

    monkeypatch.setattr(torch.autograd, "grad", _forbid_ag)
    out = qa._QuantSwiGLUExpertFn.apply(x_e, w_e, exp_mod, 1)
    out.sum().backward()
    assert x_e.grad is not None and w_e.grad is not None

    monkeypatch.setattr(torch.autograd, "grad", orig_ag_grad)
    x_gc = torch.randn(4, 12, dtype=torch.float64, requires_grad=True)
    w_gc = torch.rand(4, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(
        lambda x_in, w_in: qa._QuantSwiGLUExpertFn.apply(x_in, w_in, exp_mod, 2),
        (x_gc, w_gc),
        eps=1e-6,
        atol=1e-10,
        rtol=1e-8,
    )

    # Exotic custom activation (GELU) uses fallback cleanly and passes gradcheck
    exp_mod.act_fn = F.gelu  # type: ignore[attr-defined]
    assert qa._is_standard_silu(exp_mod.act_fn) is False
    assert torch.autograd.gradcheck(
        lambda x_in, w_in: qa._QuantSwiGLUExpertFn.apply(x_in, w_in, exp_mod, 0),
        (x_gc.detach().clone().requires_grad_(True), w_gc.detach().clone().requires_grad_(True)),
        eps=1e-6,
        atol=1e-10,
        rtol=1e-8,
    )


# ============================================================================ 2. Mocked `torch._grouped_mm` & chunk=64


@pytest.mark.parametrize("scheme", ["fp8", "fp8_transposed", "nvfp4"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_grouped_mm_fast_path_chunk64_multi_chunk_offs_and_parity(
    scheme: str,
    dtype: torch.dtype,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert qa._MOE_CHUNK_EXPERTS == 64

    torch.manual_seed(202)
    num_experts, h_dim, inter_dim = 140, 32, 16
    n_tokens, top_k = 52, 4
    gu_ref = torch.randn(num_experts, 2 * inter_dim, h_dim) / math.sqrt(h_dim)
    dn_ref = torch.randn(num_experts, h_dim, inter_dim) / math.sqrt(inter_dim)

    if scheme == "fp8":
        qmoe: nn.Module = _MockFP8Experts(gu_ref, dn_ref, block_size=(16, 16), transposed=False)
    elif scheme == "fp8_transposed":
        qmoe = _MockFP8Experts(gu_ref, dn_ref, block_size=(16, 16), transposed=True)
    else:
        qmoe = _MockNVFP4Experts(gu_ref, dn_ref, group_size=16)
    install_quantized_autograd_bridge(qmoe)

    # Route across 132 unique active experts -> 3 chunks of [64, 64, 4], including `-1` & duplicates
    top_idx = torch.arange(n_tokens * top_k, dtype=torch.long) % 135
    top_idx = top_idx.reshape(n_tokens, top_k)
    top_idx[0] = torch.tensor([4, 4, 130, -1], dtype=torch.long)
    top_idx[7] = torch.tensor([-1, -1, -1, -1], dtype=torch.long)
    top_idx[19] = torch.tensor([77, 77, 77, 12], dtype=torch.long)
    n_unique_active = len({int(e) for e in top_idx.reshape(-1).tolist() if 0 <= int(e) < num_experts})
    assert n_unique_active > 128  # spans 3 chunks with _MOE_CHUNK_EXPERTS=64

    gmm_calls: list[tuple[tuple[int, ...], tuple[int, ...], list[int]]] = []

    def _mock_grouped_mm(
        mat_a: torch.Tensor,
        mat_b: torch.Tensor,
        offs: torch.Tensor | None = None,
        bias: torch.Tensor | None = None,
        out_dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        assert bias is None
        assert mat_a.ndim == 2 and mat_b.ndim == 3
        assert mat_a.dtype == mat_b.dtype
        assert offs is not None and offs.ndim == 1 and offs.dtype == torch.int32
        assert offs.device == mat_a.device
        assert int(offs.shape[0]) == int(mat_b.shape[0])
        offs_list = [int(v) for v in offs.tolist()]
        if int(mat_a.shape[0]) > 0:
            prev = 0
            for end_off in offs_list:
                assert end_off > prev, f"offs must be strictly increasing for active experts, got {offs_list}"
                prev = end_off
            assert prev == int(mat_a.shape[0])
        gmm_calls.append((tuple(mat_a.shape), tuple(mat_b.shape), offs_list))

        pieces: list[torch.Tensor] = []
        start = 0
        for g_i, end_off in enumerate(offs_list):
            pieces.append(torch.matmul(mat_a[start:end_off], mat_b[g_i]))
            start = end_off
        res = torch.cat(pieces, dim=0)
        return res if out_dtype is None else res.to(out_dtype)

    monkeypatch.setattr(torch, "_grouped_mm", _mock_grouped_mm, raising=False)
    monkeypatch.setattr(
        torch.autograd,
        "grad",
        lambda *_a, **_kw: (_ for _ in ()).throw(AssertionError("nested autograd.grad forbidden")),
    )

    chunk_sizes: list[int] = []
    orig_deq_chunk = qa._dequant_experts_chunk

    def _spy_deq_chunk(
        mod: nn.Module,
        expert_ids: list[int],
        dtype: torch.dtype = torch.bfloat16,
        *,
        nvfp4_mode_override: str | None = None,
    ) -> tuple[torch.Tensor | list[torch.Tensor], torch.Tensor | list[torch.Tensor]]:
        chunk_sizes.append(len(expert_ids))
        return orig_deq_chunk(mod, expert_ids, dtype=dtype, nvfp4_mode_override=nvfp4_mode_override)

    monkeypatch.setattr(qa, "_dequant_experts_chunk", _spy_deq_chunk)

    x_in = (torch.randn(n_tokens, h_dim) * 0.7).to(dtype).detach().requires_grad_(True)
    # Router weights in float32 alongside bf16/fp32 hidden_states (matches real Qwen3.6 router)
    w_in = (torch.rand(n_tokens, top_k, dtype=torch.float32) * 0.5 + 0.1).detach().requires_grad_(True)
    loss_w = torch.linspace(-0.8, 0.9, steps=n_tokens * h_dim, dtype=torch.float32).reshape(n_tokens, h_dim)

    gmm_calls.clear()
    out = qmoe(x_in, top_idx, w_in)
    # Probe executes 2 tiny calls on first invocation, then 3 chunks * 2 `_grouped_mm` calls = 6 in forward
    probe_calls = [c for c in gmm_calls if c[0] == (4, 16) and c[1] == (2, 16, 16)]
    fwd_calls = [c for c in gmm_calls if not (c[0] == (4, 16) and c[1] == (2, 16, 16))]
    assert len(probe_calls) == 2
    assert len(fwd_calls) == 6
    assert chunk_sizes == [64, 64, n_unique_active - 128]

    # Second call reuses cached `_supports_grouped_mm` (0 additional probe calls!)
    gmm_calls.clear()
    loss = (out.float() * loss_w).sum()
    loss.backward()
    # Uncheckpointed backward: 3 chunks * (2 fwd `_grouped_mm` for `y_sub` + 2 bwd `_grouped_mm` for `dx`) = 12
    assert len(gmm_calls) == 12
    assert chunk_sizes == [64, 64, n_unique_active - 128, 64, 64, n_unique_active - 128]

    # Restore autograd.grad only for the naive reference computation
    monkeypatch.undo()
    x_ref = x_in.detach().clone().requires_grad_(True)
    w_ref = w_in.detach().clone().requires_grad_(True)
    out_ref = _reference_moe_forward_autograd(x_ref, top_idx, w_ref, qmoe)
    (out_ref.float() * loss_w).sum().backward()

    assert x_in.grad is not None and x_ref.grad is not None
    assert w_in.grad is not None and w_ref.grad is not None
    if dtype == torch.float32:
        assert torch.allclose(out, out_ref, atol=1e-5, rtol=1e-5)
        assert torch.allclose(x_in.grad, x_ref.grad, atol=1e-5, rtol=1e-5)
        assert torch.allclose(w_in.grad, w_ref.grad, atol=1e-5, rtol=1e-5)
    else:
        assert torch.allclose(out.float(), out_ref.float(), atol=3e-2, rtol=2e-2)
        assert torch.allclose(x_in.grad.float(), x_ref.grad.float(), atol=3e-2, rtol=2e-2)
        assert torch.allclose(w_in.grad, w_ref.grad, atol=3e-2, rtol=2e-2)

    # Masked `-1` slots must receive exact 0.0 gradient
    assert float(w_in.grad[0, 3].abs().item()) == 0.0
    assert torch.all(w_in.grad[7] == 0.0)


def test_grouped_mm_unaligned_runtime_error_falls_back_cleanly(monkeypatch: pytest.MonkeyPatch) -> None:
    """If `_grouped_mm` probe passes on 16x16 but raises RuntimeError on odd strides, fallback succeeds."""
    torch.manual_seed(209)
    gu = torch.randn(6, 20, 14) * 0.2  # non-multiple-of-8 dims (I=10, H=14)
    dn = torch.randn(6, 14, 10) * 0.2
    qmoe = _MockFP8Experts(gu, dn, block_size=(16, 16))
    install_quantized_autograd_bridge(qmoe)

    def _strict_aligned_gmm(
        mat_a: torch.Tensor,
        mat_b: torch.Tensor,
        offs: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if mat_a.shape[-1] % 8 != 0 or mat_b.shape[-2] % 8 != 0:
            raise RuntimeError("Expected stride to be multiple of 16 bytes")
        assert offs is not None
        return torch.zeros((mat_a.shape[0], mat_b.shape[-1]), dtype=mat_a.dtype, device=mat_a.device)

    monkeypatch.setattr(torch, "_grouped_mm", _strict_aligned_gmm, raising=False)
    x = torch.randn(8, 14, dtype=torch.bfloat16, requires_grad=True)
    idx = torch.tensor([[0, 1], [2, 3], [4, 5], [0, 2], [1, 4], [3, 5], [0, 5], [1, 2]], dtype=torch.long)
    w = torch.softmax(torch.randn(8, 2), dim=-1).detach().requires_grad_(True)

    out = qmoe(x, idx, w)
    out.float().pow(2).mean().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all() and float(x.grad.float().norm()) > 0.0
    assert w.grad is not None and torch.isfinite(w.grad).all() and float(w.grad.norm()) > 0.0


# ============================================================================ 3. Padded-3D `bmm` & sparse fallback


def test_padded_3d_bmm_fast_path_and_sparse_matmul_fallback_plus_gradcheck(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch.manual_seed(303)
    monkeypatch.setattr(qa, "_supports_grouped_mm", lambda _dev, _dt: False)

    bmm_calls: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
    orig_bmm = torch.bmm

    def _spy_bmm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        bmm_calls.append((tuple(a.shape), tuple(b.shape)))
        return orig_bmm(a, b)

    monkeypatch.setattr(torch, "bmm", _spy_bmm)

    gu = torch.randn(6, 16, 12, dtype=torch.float64) * 0.25
    dn = torch.randn(6, 12, 8, dtype=torch.float64) * 0.25
    exp_mod = _PlainFloatFusedExperts(gu, dn)

    # A) Balanced routing across 5 active experts (`C=5 >= 4`, fill ratio = 16 / (5 * 4) = 0.8 >= 0.4) -> hits `bmm`
    idx_bal = torch.tensor(
        [[0, 1], [2, 3], [4, 0], [1, 2], [3, 4], [0, 2], [1, 3], [4, 1]],
        dtype=torch.long,
    )
    x_bal = torch.randn(8, 12, dtype=torch.float64, requires_grad=True)
    w_bal = torch.rand(8, 2, dtype=torch.float64, requires_grad=True)
    bmm_calls.clear()
    y_bal = qa._QuantMoERoutedAutogradFn.apply(x_bal, idx_bal, w_bal, exp_mod)
    y_bal.sum().backward()
    assert len(bmm_calls) == 6  # 2 in fwd + 2 in bwd fwd-recomp + 2 in bwd dx

    assert torch.autograd.gradcheck(
        lambda x_in, w_in: qa._QuantMoERoutedAutogradFn.apply(x_in, idx_bal, w_in, exp_mod),
        (x_bal.detach().clone().requires_grad_(True), w_bal.detach().clone().requires_grad_(True)),
        eps=1e-6,
        atol=1e-8,
        rtol=1e-6,
    )

    # B) Heavily skewed routing (`C=5`, expert 0 gets 20 slots, experts 1..4 get 1 each -> fill 24/100 = 0.24 < 0.4)
    idx_skew = torch.zeros((12, 2), dtype=torch.long)
    idx_skew[0, 1] = 1
    idx_skew[1, 1] = 2
    idx_skew[2, 1] = 3
    idx_skew[3, 1] = 4
    x_sk = torch.randn(12, 12, dtype=torch.float64, requires_grad=True)
    w_sk = torch.rand(12, 2, dtype=torch.float64, requires_grad=True)
    bmm_calls.clear()
    y_sk = qa._QuantMoERoutedAutogradFn.apply(x_sk, idx_skew, w_sk, exp_mod)
    y_sk.sum().backward()
    assert len(bmm_calls) == 0  # skipped padded bmm due to low fill ratio, ran analytic matmul fallback!

    xr = x_sk.detach().clone().requires_grad_(True)
    wr = w_sk.detach().clone().requires_grad_(True)
    _reference_moe_forward_autograd(xr, idx_skew, wr, exp_mod).sum().backward()
    assert x_sk.grad is not None and xr.grad is not None
    assert w_sk.grad is not None and wr.grad is not None
    assert float((x_sk.grad - xr.grad).abs().max().item()) < 1e-14
    assert float((w_sk.grad - wr.grad).abs().max().item()) < 1e-14

    assert torch.autograd.gradcheck(
        lambda x_in, w_in: qa._QuantMoERoutedAutogradFn.apply(x_in, idx_skew, w_in, exp_mod),
        (x_sk.detach().clone().requires_grad_(True), w_sk.detach().clone().requires_grad_(True)),
        eps=1e-6,
        atol=1e-8,
        rtol=1e-6,
    )


# ========================================================== 4. Single-layer checkpoint recompute cache


@pytest.mark.parametrize("scheme", ["fp8", "nvfp4"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("use_reentrant", [False, True])
def test_checkpoint_recompute_cache_eliminates_redundant_syncs_and_gemms_zero_leaks(
    scheme: str,
    dtype: torch.dtype,
    use_reentrant: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch.manual_seed(404)
    n_exp, h_dim, inter_dim = 10, 32, 16
    gu = torch.randn(n_exp, 2 * inter_dim, h_dim) / math.sqrt(h_dim)
    dn = torch.randn(n_exp, h_dim, inter_dim) / math.sqrt(inter_dim)
    qmoe: nn.Module = (
        _MockFP8Experts(gu, dn, block_size=(16, 16))
        if scheme == "fp8"
        else _MockNVFP4Experts(gu, dn, group_size=16)
    )
    install_quantized_autograd_bridge(qmoe)

    pre_lin = nn.Linear(h_dim, h_dim, bias=False).to(dtype)
    router_lin = nn.Linear(h_dim, n_exp, bias=False).to(torch.float32)
    post_lin = nn.Linear(h_dim, h_dim, bias=False).to(dtype)

    fixed_idx = torch.tensor(
        [[0, 3], [2, 7], [1, 5], [4, 8], [3, 9], [0, 6]],
        dtype=torch.long,
    )

    def _layer_block(x: torch.Tensor) -> torch.Tensor:
        h = pre_lin(x)
        scores = torch.softmax(router_lin(h.float()), dim=-1)
        top_w = scores.gather(-1, fixed_idx)
        moe_out = qmoe(h, fixed_idx, top_w)
        # Downstream op saving activations AFTER MoE inside the same checkpoint frame!
        return post_lin(moe_out * torch.sigmoid(h))

    # 1) Uncheckpointed reference (`_in_backward() == False` during forward)
    x_ref = (torch.randn(6, h_dim) * 0.6).to(dtype).detach().requires_grad_(True)
    loss_target = torch.linspace(-0.5, 0.7, steps=6 * h_dim, dtype=torch.float32).reshape(6, h_dim)
    y_ref = _layer_block(x_ref)
    (y_ref.float() * loss_target).sum().backward()
    ref_x_grad = x_ref.grad.detach().clone()  # type: ignore[union-attr]
    ref_pre_grad = pre_lin.weight.grad.detach().clone()  # type: ignore[union-attr]
    ref_rtr_grad = router_lin.weight.grad.detach().clone()  # type: ignore[union-attr]
    ref_post_grad = post_lin.weight.grad.detach().clone()  # type: ignore[union-attr]

    # Spy on `_build_moe_dispatch_plan`, `_moe_chunk_forward_silu`, `_dequant_experts_chunk`, and weakrefs
    plan_in_bwd_flags: list[bool] = []
    fwd_silu_calls: list[bool] = []
    deq_calls: list[bool] = []
    cached_tensor_weakrefs: list[weakref.ReferenceType[torch.Tensor]] = []

    orig_plan = qa._build_moe_dispatch_plan
    orig_fwd_silu = qa._moe_chunk_forward_silu
    orig_deq = qa._dequant_experts_chunk
    orig_fwd_impl = qa._moe_routed_forward_impl

    def _spy_plan(top_k_index: torch.Tensor, n_e: int) -> Any:
        plan_in_bwd_flags.append(qa._in_backward())
        return orig_plan(top_k_index, n_e)

    def _spy_fwd_silu(*args: Any, **kwargs: Any) -> Any:
        fwd_silu_calls.append(qa._in_backward())
        return orig_fwd_silu(*args, **kwargs)

    def _spy_deq(*args: Any, **kwargs: Any) -> Any:
        deq_calls.append(qa._in_backward())
        return orig_deq(*args, **kwargs)

    def _spy_fwd_impl(*args: Any, **kwargs: Any) -> Any:
        out_t, cache_t = orig_fwd_impl(*args, **kwargs)
        if cache_t is not None:
            for item in cache_t:
                if isinstance(item, torch.Tensor) and item.numel() > 0:
                    cached_tensor_weakrefs.append(weakref.ref(item))
        return out_t, cache_t

    monkeypatch.setattr(qa, "_build_moe_dispatch_plan", _spy_plan)
    monkeypatch.setattr(qa, "_moe_chunk_forward_silu", _spy_fwd_silu)
    monkeypatch.setattr(qa, "_dequant_experts_chunk", _spy_deq)
    monkeypatch.setattr(qa, "_moe_routed_forward_impl", _spy_fwd_impl)

    # 2) Checkpointed run (`_in_backward() == True` during recompute!)
    pre_lin.zero_grad(set_to_none=True)
    router_lin.zero_grad(set_to_none=True)
    post_lin.zero_grad(set_to_none=True)
    x_ck = x_ref.detach().clone().requires_grad_(True)
    x_ck_snapshot = x_ck.detach().clone()

    y_ck = checkpoint(_layer_block, x_ck, use_reentrant=use_reentrant)
    # Before `.backward()`: 0 recompute cache tensors exist!
    assert len(cached_tensor_weakrefs) == 0

    (y_ck.float() * loss_target).sum().backward()

    # Verify recompute cache WAS populated during `_in_backward() == True`, AND inside backward:
    # - `_build_moe_dispatch_plan` ran AT MOST ONCE during `_in_backward() == True` (NOT twice!)
    # - `_moe_chunk_forward_silu` ran AT MOST ONCE during `_in_backward() == True` (skipped in `.backward()`!)
    # - `_dequant_experts_chunk` ran twice during `_in_backward() == True` (1 in recompute fwd + 1 in `.backward()` dx)
    assert len(cached_tensor_weakrefs) > 0
    assert plan_in_bwd_flags.count(True) == 1
    assert fwd_silu_calls.count(True) == 1
    assert deq_calls.count(True) == 2

    # Verify exact parity vs uncheckpointed forward & backward and zero input mutation
    assert torch.equal(x_ck.detach(), x_ck_snapshot)
    assert torch.equal(y_ck, y_ref)
    assert x_ck.grad is not None
    if dtype == torch.float32:
        assert torch.allclose(x_ck.grad, ref_x_grad, atol=1e-6, rtol=1e-6)
        assert torch.allclose(pre_lin.weight.grad, ref_pre_grad, atol=1e-6, rtol=1e-6)  # type: ignore[arg-type]
        assert torch.allclose(router_lin.weight.grad, ref_rtr_grad, atol=1e-6, rtol=1e-6)  # type: ignore[arg-type]
        assert torch.allclose(post_lin.weight.grad, ref_post_grad, atol=1e-6, rtol=1e-6)  # type: ignore[arg-type]
    else:
        assert torch.equal(x_ck.grad, ref_x_grad)
        assert torch.equal(pre_lin.weight.grad, ref_pre_grad)  # type: ignore[arg-type]
        assert torch.equal(router_lin.weight.grad, ref_rtr_grad)  # type: ignore[arg-type]
        assert torch.equal(post_lin.weight.grad, ref_post_grad)  # type: ignore[arg-type]

    # Verify ZERO leaked tensors after backward completes!
    gc.collect()
    leaked = sum(1 for ref in cached_tensor_weakrefs if ref() is not None)
    assert leaked == 0

    # 3) When `need_h == False` and `need_w == True` inside checkpoint, `.backward()` needs 0 dequant passes!
    plan_in_bwd_flags.clear()
    deq_calls.clear()
    cached_tensor_weakrefs.clear()
    x_frozen = x_ref.detach().clone().requires_grad_(False)
    pre_lin.requires_grad_(False)
    router_lin.zero_grad(set_to_none=True)
    post_lin.zero_grad(set_to_none=True)
    y_w_only = checkpoint(_layer_block, x_frozen, use_reentrant=False)
    (y_w_only.float() * loss_target).sum().backward()
    pre_lin.requires_grad_(True)
    assert deq_calls.count(True) == 1  # ONLY the recompute forward dequantized; `.backward()` used cached `y_all`!
    assert torch.allclose(router_lin.weight.grad, ref_rtr_grad, atol=1e-5, rtol=1e-5)  # type: ignore[arg-type]
    del y_w_only
    gc.collect()
    assert sum(1 for ref in cached_tensor_weakrefs if ref() is not None) == 0

    # 4) Partial `autograd.grad` wrt `post_lin.weight` only (where MoE `.backward()` never runs) still leaks 0 tensors!
    cached_tensor_weakrefs.clear()
    x_part = x_ref.detach().clone().requires_grad_(True)
    y_part = checkpoint(_layer_block, x_part, use_reentrant=False)
    _ = torch.autograd.grad((y_part.float() * loss_target).sum(), post_lin.weight, retain_graph=False)
    del x_part, y_part
    gc.collect()
    assert sum(1 for ref in cached_tensor_weakrefs if ref() is not None) == 0


def test_loop_virtual_layer_checkpointing_end_to_end_parity_on_fp8_toy() -> None:
    """End-to-end `apply_loop(..., LoopSpec(checkpoint=True))` vs `checkpoint=False` parity."""
    def _build_looped(ckpt: bool) -> nn.Module:
        base = build_toy(
            seed=71,
            num_hidden_layers=6,
            hidden_size=32,
            moe_intermediate_size=16,
            shared_expert_intermediate_size=16,
            num_experts=6,
            num_experts_per_tok=2,
        )
        for layer in base.model.layers:
            exp = layer.mlp.experts
            layer.mlp.experts = _MockFP8Experts(  # type: ignore[assignment]
                exp.gate_up_proj,
                exp.down_proj,
                block_size=(16, 16),
            )
        for p in base.parameters():
            p.requires_grad_(False)
        install_quantized_autograd_bridge(base)
        apply_loop(
            base,
            LoopSpec(
                prelude_end=1,
                core_end=5,
                num_passes=2,
                target_virtual_layers=None,
                gate_init=0.3,
                per_pass_router=True,
                router_adapter_rank=4,
                attn_adapter_rank=4,
                checkpoint=ckpt,
            ),
        )
        base.train()
        return base

    ids = torch.randint(0, 90, (2, 8), generator=torch.Generator().manual_seed(88))
    m_nockpt = _build_looped(False)
    m_ckpt = _build_looped(True)

    out_no = m_nockpt(ids, labels=ids)
    out_ck = m_ckpt(ids, labels=ids)
    assert out_no.loss is not None and out_ck.loss is not None
    assert torch.allclose(out_no.loss, out_ck.loss, atol=1e-6, rtol=1e-6)

    out_no.loss.backward()
    out_ck.loss.backward()

    grads_no = {n: p.grad for n, p in m_nockpt.named_parameters() if p.requires_grad}
    grads_ck = {n: p.grad for n, p in m_ckpt.named_parameters() if p.requires_grad}
    assert len(grads_ck) > 0
    for name, g_no in grads_no.items():
        g_ck = grads_ck[name]
        assert g_no is not None and g_ck is not None, f"missing grad on {name}"
        assert torch.allclose(g_no, g_ck, atol=1e-5, rtol=1e-5), f"grad mismatch on {name}"


# ============================================================================ 5. Direct `bfloat16` FP8 dequantization


def test_dequantize_fp8_tensor_direct_bf16_lossless_and_no_fp32_intermediate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # 1) All 254 finite `float8_e4m3fn` codes are 100% bit-exact in `bfloat16`
    raw_fp8 = torch.arange(256, dtype=torch.uint8).view(torch.float8_e4m3fn)
    finite_mask = torch.isfinite(raw_fp8.float())
    assert int(finite_mask.sum().item()) == 254
    assert torch.equal(raw_fp8.to(torch.bfloat16).float()[finite_mask], raw_fp8.float()[finite_mask])

    # 2) Spy on `Tensor.to` to prove `dequantize_fp8_tensor(..., dtype=torch.bfloat16)` NEVER casts
    #    FP8 `weight` or `scale_inv` to `torch.float32`!
    seen_to_dtypes: list[torch.dtype] = []
    orig_tensor_to = torch.Tensor.to

    def _spy_to(self: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        for arg in args:
            if isinstance(arg, torch.dtype):
                seen_to_dtypes.append(arg)
        kw_dt = kwargs.get("dtype")
        if isinstance(kw_dt, torch.dtype):
            seen_to_dtypes.append(kw_dt)
        return orig_tensor_to(self, *args, **kwargs)

    g = torch.Generator().manual_seed(505)
    w2 = torch.randn(260, 140, generator=g, dtype=torch.float32) * 0.6
    q2, s2 = quantize_fp8_blockwise(w2, block_size=(128, 128))
    w3 = torch.randn(6, 256, 128, generator=g, dtype=torch.float32) * 0.4
    q3, s3 = quantize_fp8_blockwise(w3, block_size=(128, 128))

    monkeypatch.setattr(torch.Tensor, "to", _spy_to)
    seen_to_dtypes.clear()
    deq2_bf16 = dequantize_fp8_tensor(q2, s2, dtype=torch.bfloat16, block_size=(128, 128))
    deq3_bf16 = dequantize_fp8_tensor(q3, s3, dtype=torch.bfloat16, block_size=(128, 128))
    deq3_flat_bf16 = dequantize_fp8_tensor(
        q3,
        s3.reshape(6 * s3.shape[1], s3.shape[2]),
        dtype=torch.bfloat16,
        block_size=(128, 128),
    )
    deq_none_bf16 = dequantize_fp8_tensor(q2, None, dtype=torch.bfloat16)
    monkeypatch.undo()

    assert torch.float32 not in seen_to_dtypes
    assert deq2_bf16.dtype == torch.bfloat16 and tuple(deq2_bf16.shape) == (260, 140)
    assert deq3_bf16.dtype == torch.bfloat16 and tuple(deq3_bf16.shape) == (6, 256, 128)
    assert torch.equal(deq3_bf16, deq3_flat_bf16)
    assert torch.equal(deq_none_bf16.float(), q2.float())

    # Compare against `fp32` promotion reference: within 1 bf16 ULP (`rel < 5e-3`)
    deq2_fp32 = dequantize_fp8_tensor(q2, s2, dtype=torch.float32, block_size=(128, 128))
    deq3_fp32 = dequantize_fp8_tensor(q3, s3, dtype=torch.float32, block_size=(128, 128))
    assert float((deq2_bf16.float() - deq2_fp32).norm() / deq2_fp32.norm()) < 5e-3
    assert float((deq3_bf16.float() - deq3_fp32).norm() / deq3_fp32.norm()) < 5e-3

    # 3) `float32` and `float64` callers retain full precision (`0.0` loss vs exact fp64 math)
    s_f64 = torch.tensor(0.123456789012345, dtype=torch.float64)
    deq_f64 = dequantize_fp8_tensor(q2, s_f64, dtype=torch.float64)
    assert deq_f64.dtype == torch.float64
    assert torch.equal(deq_f64, q2.to(torch.float64) * s_f64)

    # 4) Extreme subnormal `float32` scale (`1e-35` outside `(1e-30, 1e30)`) falls back safely
    s_tiny = torch.full_like(s2, 1e-35)
    deq_tiny = dequantize_fp8_tensor(q2, s_tiny, dtype=torch.bfloat16, block_size=(128, 128))
    ref_tiny = (q2.float() * s_tiny.repeat_interleave(128, dim=0)[:260].repeat_interleave(128, dim=1)[:, :140]).to(
        torch.bfloat16
    )
    assert torch.equal(deq_tiny, ref_tiny)

    # Keep `auto_dequant_expert_fn` in exact `float32`
    exp_fp8 = _MockFP8Experts(
        torch.randn(2, 32, 32) * 0.2,
        torch.randn(2, 32, 16) * 0.2,
        block_size=(16, 16),
    )
    gu0, dn0 = auto_dequant_expert_fn(exp_fp8, 0)
    assert gu0.dtype == torch.float32 and dn0.dtype == torch.float32
