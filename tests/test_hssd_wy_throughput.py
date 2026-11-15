"""Tests for HSSD v3.2 WY Tensor-Core block-doubling solver, zero-sync compaction & outer-checkpoint gating."""
from __future__ import annotations

import ast
import importlib.util
import sys
import time
import types
from pathlib import Path
from typing import Any

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils.checkpoint import checkpoint

import qloop.model.hssd as hssd_mod
from qloop.data.tokenizer import ToyTokenizer
from qloop.export_pack import export_whole_model_pack
from qloop.infer import CONFIG_FILENAME, MODELING_FILENAME
from qloop.model.hssd import (
    HSSDConfig,
    PeakBoltzmannGate,
    _build_wy_unit_lower,
    _in_backward,
    _in_outer_checkpoint,
    _routed_bank_readout,
    _solve_unit_lower_wy,
    _wy_kda_state_update,
    apply_hssd,
    chunk_hssd,
    recurrent_hssd,
)
from qloop.model.introspect import inspect_model
from qloop.model.loop import LoopSpec, apply_loop
from qloop.model.toy import ToyConfig, ToyForCausalLM
from qloop.train.stack import Stack


class _SyncOpCounter(TorchDispatchMode):
    """Count `aten.nonzero` and `aten._local_scalar_dense` (`.item()`) dispatches."""

    def __init__(self) -> None:
        super().__init__()
        self.nonzero_calls = 0
        self.scalar_item_calls = 0
        self.dtoh_copy_calls = 0

    def __torch_dispatch__(
        self,
        func: Any,
        types_arg: tuple[type, ...],
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> Any:
        del types_arg
        kw = {} if kwargs is None else kwargs
        name = str(func)
        if "aten.nonzero" in name:
            self.nonzero_calls += 1
        elif "aten._local_scalar_dense" in name:
            self.scalar_item_calls += 1
        elif "aten._to_copy" in name and kw.get("device") == torch.device("cpu"):
            self.dtoh_copy_calls += 1
        return func(*args, **kw)


def _make_hssd_inputs(
    *,
    bsz: int = 2,
    seq_len: int = 128,
    hv: int = 4,
    dk: int = 16,
    dv: int = 16,
    num_pages: int = 8,
    seed: int = 1234,
    dtype: torch.dtype = torch.float64,
    over_relaxed: bool = True,
) -> dict[str, Any]:
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(bsz, seq_len, hv, dk, generator=g, dtype=dtype, requires_grad=True)
    k_raw = torch.randn(bsz, seq_len, hv, dk, generator=g, dtype=dtype)
    k = (k_raw / k_raw.norm(dim=-1, keepdim=True).clamp_min(1e-6)).detach().requires_grad_(True)
    v = torch.randn(bsz, seq_len, hv, dv, generator=g, dtype=dtype, requires_grad=True)

    ps_gamma = (
        torch.linspace(0.002, 0.05, num_pages, dtype=dtype).clone().detach().requires_grad_(True)
    )
    ch_gamma = (
        (0.01 + 0.04 * torch.rand(bsz, seq_len, hv, dk // 2, generator=g, dtype=dtype))
        .detach()
        .requires_grad_(True)
    )
    b_hi = 1.95 if over_relaxed else 0.95
    b_gate = (
        (0.15 + (b_hi - 0.15) * torch.rand(bsz, seq_len, hv, dk, generator=g, dtype=dtype))
        .detach()
        .requires_grad_(True)
    )
    w_gate = (
        (0.10 + 0.85 * torch.rand(bsz, seq_len, hv, dv, generator=g, dtype=dtype))
        .detach()
        .requires_grad_(True)
    )
    theta = (0.3 * torch.randn(bsz, seq_len, hv, dk // 2, generator=g, dtype=dtype)).detach().requires_grad_(True)
    sbias = (0.5 * torch.randn(bsz, seq_len, hv, generator=g, dtype=dtype)).detach().requires_grad_(True)
    sslope = torch.full((hv,), 3.0, dtype=dtype, requires_grad=True)
    sthresh = torch.full((hv,), 0.75, dtype=dtype, requires_grad=True)
    w_logits = torch.randn(bsz, seq_len, num_pages, generator=g, dtype=dtype, requires_grad=True)
    r_logits = torch.randn(bsz, seq_len, num_pages, generator=g, dtype=dtype, requires_grad=True)
    q_swa = torch.randn(bsz, seq_len, hv, dk, generator=g, dtype=dtype, requires_grad=True)
    k_swa = torch.randn(bsz, seq_len, hv, dk, generator=g, dtype=dtype, requires_grad=True)
    v_swa = torch.randn(bsz, seq_len, hv, dv, generator=g, dtype=dtype, requires_grad=True)
    bgate = PeakBoltzmannGate(hv, kappa_init=2.0, tau_init=0.8).to(dtype=dtype)
    swa_slope = torch.full((hv,), 0.05, dtype=dtype, requires_grad=True)

    return {
        "query": q,
        "key": k,
        "value": v,
        "ps_gamma": ps_gamma,
        "ch_gamma": ch_gamma,
        "b_gate": b_gate,
        "w_gate": w_gate,
        "theta": theta,
        "surprise_bias": sbias,
        "surprise_slope": sslope,
        "surprise_thresh": sthresh,
        "write_logits": w_logits,
        "read_logits": r_logits,
        "q_swa": q_swa,
        "k_swa": k_swa,
        "v_swa": v_swa,
        "boltzmann_gate": bgate,
        "swa_recency_slope": swa_slope,
    }


def _clone_inputs(inp: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, val in inp.items():
        if isinstance(val, torch.Tensor):
            out[k] = val.detach().clone().requires_grad_(val.requires_grad)
        elif isinstance(val, PeakBoltzmannGate):
            bg = PeakBoltzmannGate(val.num_heads).to(dtype=val.log_kappa.dtype, device=val.log_kappa.device)
            with torch.no_grad():
                bg.log_kappa.copy_(val.log_kappa)
                bg.tau.copy_(val.tau)
            out[k] = bg
        else:
            out[k] = val
    return out


def _load_standalone_without_qloop(pack_dir: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    cfg_py = pack_dir / CONFIG_FILENAME
    mod_py = pack_dir / MODELING_FILENAME
    ast.parse(cfg_py.read_text(encoding="utf-8"), feature_version=(3, 9))
    ast.parse(mod_py.read_text(encoding="utf-8"), feature_version=(3, 9))

    real_find_spec = importlib.util.find_spec

    def _hide_qloop(name: str, package: Any = None) -> Any:
        if name == "qloop" or name.startswith("qloop."):
            return None
        return real_find_spec(name, package)

    monkeypatch.setattr(importlib.util, "find_spec", _hide_qloop)

    stub_tf = types.ModuleType("transformers")

    class _PreTrainedConfig:
        model_type = "qwen36_loophssd"

        def __init__(self, **kwargs: Any) -> None:
            for k, v in kwargs.items():
                setattr(self, k, v)

    class _PreTrainedModel(torch.nn.Module):
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
        def for_model(model_type: str, **kw: Any) -> ToyConfig:
            del model_type
            valid = {f for f in ToyConfig.__dataclass_fields__ if f in kw}
            clean = {k: kw[k] for k in valid}
            if "layer_types" in clean and isinstance(clean["layer_types"], list):
                clean["layer_types"] = tuple(clean["layer_types"])
            return ToyConfig(**clean)

    class _AutoModelForCausalLM:
        @staticmethod
        def from_config(cfg: ToyConfig, **kw: Any) -> ToyForCausalLM:
            del kw
            return ToyForCausalLM(cfg)

    stub_tf.PreTrainedConfig = _PreTrainedConfig  # type: ignore[attr-defined]
    stub_tf.PreTrainedModel = _PreTrainedModel  # type: ignore[attr-defined]
    stub_tf.GenerationMixin = _GenerationMixin  # type: ignore[attr-defined]
    stub_tf.AutoConfig = _AutoConfig  # type: ignore[attr-defined]
    stub_tf.AutoModelForCausalLM = _AutoModelForCausalLM  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "transformers", stub_tf)

    pkg_name = f"_standalone_wy_tp_{abs(hash(str(pack_dir)))}"
    pkg = types.ModuleType(pkg_name)
    pkg.__path__ = [str(pack_dir)]  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, pkg_name, pkg)

    spec_cfg = importlib.util.spec_from_file_location(f"{pkg_name}.configuration_qwen36_loophssd", cfg_py)
    assert spec_cfg is not None and spec_cfg.loader is not None
    mod_cfg = importlib.util.module_from_spec(spec_cfg)
    monkeypatch.setitem(sys.modules, spec_cfg.name, mod_cfg)
    spec_cfg.loader.exec_module(mod_cfg)

    spec_mod = importlib.util.spec_from_file_location(f"{pkg_name}.modeling_qwen36_loophssd", mod_py)
    assert spec_mod is not None and spec_mod.loader is not None
    mod_model = importlib.util.module_from_spec(spec_mod)
    monkeypatch.setitem(sys.modules, spec_mod.name, mod_model)
    spec_mod.loader.exec_module(mod_model)
    assert mod_model._HAS_QLOOP is False
    return mod_model


@pytest.mark.parametrize("c_len", [1, 2, 7, 16, 37, 64, 80])
@pytest.mark.parametrize("over_relaxed", [False, True])
def test_solve_unit_lower_wy_fp64_parity_and_unitriangular_grads(
    c_len: int,
    over_relaxed: bool,
) -> None:
    """Verify `< 1e-12` fp64 forward & backward parity vs `solve_triangular` across sizes and `||L|| > 1`."""
    g = torch.Generator().manual_seed(100 + c_len * 3 + int(over_relaxed))
    bsz, hv, dk, dv = 3, 4, 16, 20

    # Construct realistic WY L matrix (including over-relaxed b_t -> 1.98 so ||L|| > 1)
    k_raw = torch.randn(bsz, hv, c_len, dk, generator=g, dtype=torch.float64)
    k_norm = k_raw / k_raw.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    b_scale = 1.95 if over_relaxed else 0.85
    k_eff = (b_scale ** 0.5) * k_norm
    log_a = -0.02 * torch.rand(bsz, hv, c_len, dk, generator=g, dtype=torch.float64)
    l_strict, _ = _build_wy_unit_lower(k_eff, log_a)
    if over_relaxed and c_len >= 16:
        assert float(torch.linalg.matrix_norm(l_strict, ord=2).max().detach()) > 1.0

    # Inject non-zero noise on diagonal/upper-triangle to verify strict unitriangular semantics
    upper_noise = torch.triu(torch.randn(bsz, hv, c_len, c_len, generator=g, dtype=torch.float64))
    l_in_a = (l_strict + upper_noise).detach().clone().requires_grad_(True)
    l_in_b = (l_strict + upper_noise).detach().clone().requires_grad_(True)

    rhs_a = torch.randn(bsz, hv, c_len, dv, generator=g, dtype=torch.float64, requires_grad=True)
    rhs_b = rhs_a.detach().clone().requires_grad_(True)
    grad_out = torch.randn(bsz, hv, c_len, dv, generator=g, dtype=torch.float64)

    out_fast = _solve_unit_lower_wy(l_in_a, rhs_a)
    eye = torch.eye(c_len, dtype=torch.float64)
    out_ref = torch.linalg.solve_triangular(eye + l_in_b, rhs_b, upper=False, unitriangular=True)

    fwd_diff = float((out_fast - out_ref).detach().abs().max())
    assert fwd_diff < 1e-12, f"c_len={c_len} over_relaxed={over_relaxed}: fwd max diff {fwd_diff}"

    (out_fast * grad_out).sum().backward()
    (out_ref * grad_out).sum().backward()
    assert l_in_a.grad is not None and l_in_b.grad is not None
    assert rhs_a.grad is not None and rhs_b.grad is not None

    # Diagonal and upper-triangle gradients must be identically 0.0
    assert float(torch.triu(l_in_a.grad).abs().max()) == 0.0
    grad_l_diff = float((l_in_a.grad - l_in_b.grad).abs().max())
    grad_rhs_diff = float((rhs_a.grad - rhs_b.grad).abs().max())
    assert grad_l_diff < 1e-12, f"c_len={c_len}: grad_L diff {grad_l_diff}"
    assert grad_rhs_diff < 1e-12, f"c_len={c_len}: grad_rhs diff {grad_rhs_diff}"


@pytest.mark.parametrize("c_len", [7, 16, 37])
def test_solve_unit_lower_wy_gradcheck(c_len: int) -> None:
    """Verify exact analytical Jacobians via `torch.autograd.gradcheck` on non-power-of-2 & over-relaxed L."""
    g = torch.Generator().manual_seed(900 + c_len)
    l_raw = 0.45 * torch.randn(2, c_len, c_len, generator=g, dtype=torch.float64)
    l_mat = l_raw.tril(-1).detach().clone().requires_grad_(True)
    rhs = torch.randn(2, c_len, 5, generator=g, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(_solve_unit_lower_wy, (l_mat, rhs), eps=1e-6, atol=1e-5, rtol=1e-4)


@pytest.mark.parametrize("surprise_mode", ["routed", "segment_start"])
def test_chunk_hssd_zero_solve_triangular_and_zero_per_chunk_sync_ops(
    monkeypatch: pytest.MonkeyPatch,
    surprise_mode: str,
) -> None:
    """Verify `chunk_hssd` (`C <= 64`) never calls `solve_triangular`, `.nonzero()`, or `.item()`."""

    def _forbid_solve_triangular(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("torch.linalg.solve_triangular must not be called for C <= 64!")

    monkeypatch.setattr(torch.linalg, "solve_triangular", _forbid_solve_triangular)

    inp_compact = _make_hssd_inputs(bsz=2, seq_len=160, hv=4, dk=16, dv=16, num_pages=8, seed=42)
    inp_dense = _clone_inputs(inp_compact)

    with _SyncOpCounter() as sync_counter:
        out_c, st_c, duty_c = chunk_hssd(
            inp_compact["query"],
            inp_compact["key"],
            inp_compact["value"],
            (inp_compact["ps_gamma"], inp_compact["ch_gamma"]),
            inp_compact["b_gate"],
            inp_compact["w_gate"],
            inp_compact["theta"],
            inp_compact["surprise_bias"],
            inp_compact["surprise_slope"],
            inp_compact["surprise_thresh"],
            inp_compact["write_logits"],
            inp_compact["read_logits"],
            inp_compact["q_swa"],
            inp_compact["k_swa"],
            inp_compact["v_swa"],
            boltzmann_gate=inp_compact["boltzmann_gate"],
            swa_recency_slope=inp_compact["swa_recency_slope"],
            chunk_size=32,
            window_chunks=2,
            num_pages=8,
            top_k_pages=2,
            segment_chunks=2,
            checkpoint_segments=True,
            output_final_state=True,
            surprise_mode=surprise_mode,
            write_mode="full",
            compact_active_pages=True,
            return_aux=True,
        )
        loss_c = out_c.square().mean() + duty_c.sum()
        loss_c.backward()

    assert sync_counter.nonzero_calls == 0, f"Expected 0 nonzero ops, got {sync_counter.nonzero_calls}"
    assert sync_counter.scalar_item_calls == 0, (
        f"Expected 0 .item() scalar syncs, got {sync_counter.scalar_item_calls}"
    )

    out_d, st_d, duty_d = chunk_hssd(
        inp_dense["query"],
        inp_dense["key"],
        inp_dense["value"],
        (inp_dense["ps_gamma"], inp_dense["ch_gamma"]),
        inp_dense["b_gate"],
        inp_dense["w_gate"],
        inp_dense["theta"],
        inp_dense["surprise_bias"],
        inp_dense["surprise_slope"],
        inp_dense["surprise_thresh"],
        inp_dense["write_logits"],
        inp_dense["read_logits"],
        inp_dense["q_swa"],
        inp_dense["k_swa"],
        inp_dense["v_swa"],
        boltzmann_gate=inp_dense["boltzmann_gate"],
        swa_recency_slope=inp_dense["swa_recency_slope"],
        chunk_size=32,
        window_chunks=2,
        num_pages=8,
        top_k_pages=2,
        segment_chunks=2,
        checkpoint_segments=False,
        output_final_state=True,
        surprise_mode=surprise_mode,
        write_mode="full",
        compact_active_pages=False,
        return_aux=True,
    )
    loss_d = out_d.square().mean() + duty_d.sum()
    loss_d.backward()

    assert st_c is not None and st_d is not None
    assert float((out_c - out_d).detach().abs().max()) < 1e-12
    assert float((st_c.page_states - st_d.page_states).detach().abs().max()) < 1e-12
    for key in ("query", "key", "value", "b_gate", "w_gate", "theta", "write_logits", "read_logits"):
        gc = inp_compact[key].grad
        gd = inp_dense[key].grad
        assert gc is not None and gd is not None
        assert float((gc - gd).abs().max()) < 1e-12, f"{key} grad diff: {float((gc - gd).abs().max())}"


def test_standalone_helpers_single_plan_sync_parity() -> None:
    """Verify standalone `_wy_kda_state_update` and `_routed_bank_readout` execute with 0 nonzero/.item()."""
    g = torch.Generator().manual_seed(77)
    bsz, num_pages, hv, dk, dv, c_len = 2, 8, 4, 16, 16, 24
    p0 = torch.randn(bsz, num_pages, hv, dk, dv, generator=g, dtype=torch.float64, requires_grad=True)
    ke = (0.3 * torch.randn(bsz, num_pages, hv, c_len, dk, generator=g, dtype=torch.float64))
    ve = torch.randn(bsz, num_pages, hv, c_len, dv, generator=g, dtype=torch.float64)
    la = -0.03 * torch.rand(bsz, num_pages, hv, c_len, dk, generator=g, dtype=torch.float64)

    top_idx = torch.randint(0, 4, (bsz, c_len, 2), generator=g)  # pages 4..7 untouched -> triggers compaction
    act_mask = (
        torch.zeros((bsz, c_len, num_pages), dtype=torch.bool).scatter_(2, top_idx, True).transpose(1, 2)
    )
    mask5 = act_mask[:, :, None, :, None]
    ke_m = torch.where(mask5, ke, torch.zeros_like(ke)).detach().requires_grad_(True)
    ve_m = torch.where(mask5, ve, torch.zeros_like(ve)).detach().requires_grad_(True)
    la_m = torch.where(mask5, la, torch.zeros_like(la)).detach().requires_grad_(True)

    with _SyncOpCounter() as cnt:
        p_comp = _wy_kda_state_update(p0, ke_m, ve_m, la_m, active_mask=act_mask, compact_active_pages=True)
        vec_c = torch.randn(bsz, c_len, hv, dk, generator=g, dtype=torch.float64, requires_grad=True)
        top_w = torch.full((bsz, c_len, 2), 0.5, dtype=torch.float64)
        r_comp = _routed_bank_readout(vec_c, p_comp, top_idx, top_w, compact_active_pages=True)
        (p_comp.sum() + r_comp.sum()).backward()

    assert cnt.nonzero_calls == 0
    assert cnt.scalar_item_calls == 0

    p_ref = _wy_kda_state_update(
        p0.detach(), ke_m.detach(), ve_m.detach(), la_m.detach(), active_mask=None, compact_active_pages=False
    )
    r_ref = _routed_bank_readout(vec_c.detach(), p_ref, top_idx, top_w, compact_active_pages=False)
    assert float((p_comp.detach() - p_ref).abs().max()) < 1e-12
    assert float((r_comp.detach() - r_ref).abs().max()) < 1e-12


def test_outer_checkpoint_skips_redundant_nested_segment_recompute_and_matches_recurrent_fp64(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify `chunk_hssd` inside outer `checkpoint(..., use_reentrant=False/True)` skips nested checkpointing."""
    inp_outer_nonreentrant = _make_hssd_inputs(bsz=2, seq_len=96, hv=3, dk=12, dv=12, num_pages=6, seed=2026)
    inp_outer_reentrant = _clone_inputs(inp_outer_nonreentrant)
    inp_inner_only = _clone_inputs(inp_outer_nonreentrant)
    inp_rec = _clone_inputs(inp_outer_nonreentrant)

    orig_run_group = hssd_mod._run_chunk_group
    call_log: list[tuple[bool, bool]] = []

    def _counting_run_group(*args: Any, **kwargs: Any) -> Any:
        call_log.append((_in_backward(), _in_outer_checkpoint()))
        return orig_run_group(*args, **kwargs)

    monkeypatch.setattr(hssd_mod, "_run_chunk_group", _counting_run_group)

    def _run_chunk(d: dict[str, Any], *, ckpt_segs: bool = True, min_len: int = 0) -> torch.Tensor:
        out, _ = chunk_hssd(
            d["query"],
            d["key"],
            d["value"],
            (d["ps_gamma"], d["ch_gamma"]),
            d["b_gate"],
            d["w_gate"],
            d["theta"],
            d["surprise_bias"],
            d["surprise_slope"],
            d["surprise_thresh"],
            d["write_logits"],
            d["read_logits"],
            d["q_swa"],
            d["k_swa"],
            d["v_swa"],
            boltzmann_gate=d["boltzmann_gate"],
            swa_recency_slope=d["swa_recency_slope"],
            chunk_size=16,
            window_chunks=2,
            num_pages=6,
            top_k_pages=2,
            segment_chunks=2,  # 96 tokens / (2 * 16) = 3 macro-segments
            checkpoint_segments=ckpt_segs,
            checkpoint_segments_min_len=min_len,
            output_final_state=False,
            write_mode="full",
        )
        return out

    # 1. Standalone chunk_hssd(checkpoint_segments=True): 3 segments * (1 fwd + 1 bwd) = 6 calls
    call_log.clear()
    out_inner = _run_chunk(inp_inner_only, ckpt_segs=True)
    grad_out = torch.randn_like(out_inner)
    (out_inner * grad_out).sum().backward()
    assert len(call_log) == 6, f"Expected 6 segment calls standalone, got {len(call_log)}: {call_log}"

    # 2. Wrapped in outer checkpoint(..., use_reentrant=False) (VirtualLayer pattern):
    #    Must run ONLY 3 (outer fwd) + 3 (outer bwd with _in_backward()==True) = 6 calls, NOT 9!
    call_log.clear()

    def _outer_fn_nonreentrant(q_t: torch.Tensor) -> torch.Tensor:
        d_local = dict(inp_outer_nonreentrant)
        d_local["query"] = q_t
        return _run_chunk(d_local, ckpt_segs=True)

    out_outer_nr = checkpoint(_outer_fn_nonreentrant, inp_outer_nonreentrant["query"], use_reentrant=False)
    (out_outer_nr * grad_out).sum().backward()
    assert len(call_log) == 6, (
        f"Nested checkpoint was NOT skipped inside outer use_reentrant=False checkpoint: "
        f"expected 6 calls, got {len(call_log)} ({call_log})"
    )
    assert sum(1 for in_bwd, _ in call_log if in_bwd) == 3
    assert all(in_outer for _, in_outer in call_log)

    # 3. Wrapped in outer checkpoint(..., use_reentrant=True): also 3 + 3 = 6 calls, with _in_backward()==True!
    call_log.clear()

    def _outer_fn_reentrant(q_t: torch.Tensor) -> torch.Tensor:
        d_local = dict(inp_outer_reentrant)
        d_local["query"] = q_t
        return _run_chunk(d_local, ckpt_segs=True)

    out_outer_r = checkpoint(_outer_fn_reentrant, inp_outer_reentrant["query"], use_reentrant=True)
    (out_outer_r * grad_out).sum().backward()
    assert len(call_log) == 6, f"Expected 6 calls under use_reentrant=True, got {len(call_log)}"
    assert sum(1 for in_bwd, _ in call_log if in_bwd) == 3

    # 4. Exact fp64 parity vs step-by-step recurrent_hssd!
    out_rec, _ = recurrent_hssd(
        inp_rec["query"],
        inp_rec["key"],
        inp_rec["value"],
        (inp_rec["ps_gamma"], inp_rec["ch_gamma"]),
        inp_rec["b_gate"],
        inp_rec["w_gate"],
        inp_rec["theta"],
        inp_rec["surprise_bias"],
        inp_rec["surprise_slope"],
        inp_rec["surprise_thresh"],
        inp_rec["write_logits"],
        inp_rec["read_logits"],
        inp_rec["q_swa"],
        inp_rec["k_swa"],
        inp_rec["v_swa"],
        boltzmann_gate=inp_rec["boltzmann_gate"],
        swa_recency_slope=inp_rec["swa_recency_slope"],
        chunk_size=16,
        window_chunks=2,
        num_pages=6,
        top_k_pages=2,
        segment_chunks=2,
        output_final_state=False,
        write_mode="full",
    )
    (out_rec * grad_out).sum().backward()

    assert float((out_outer_nr - out_rec).detach().abs().max()) < 1e-12
    assert float((out_outer_r - out_rec).detach().abs().max()) < 1e-12
    assert float((out_inner - out_rec).detach().abs().max()) < 1e-12
    for key in ("query", "key", "value", "b_gate", "w_gate", "theta", "write_logits", "read_logits"):
        g_nr = inp_outer_nonreentrant[key].grad
        g_in = inp_inner_only[key].grad
        g_rc = inp_rec[key].grad
        assert g_nr is not None and g_in is not None and g_rc is not None
        assert float((g_nr - g_rc).abs().max()) < 1e-12, f"{key} outer_nr vs rec grad diff"
        assert float((g_in - g_rc).abs().max()) < 1e-12, f"{key} inner vs rec grad diff"

    # 5. Verify checkpoint_segments_min_len gating & validation
    with pytest.raises(ValueError, match="checkpoint_segments_min_len"):
        HSSDConfig(checkpoint_segments_min_len=-1).validate()

    call_log.clear()
    inp_minlen = _clone_inputs(inp_outer_nonreentrant)
    out_minlen = _run_chunk(inp_minlen, ckpt_segs=True, min_len=128)  # seq_len=96 < 128 -> uncheckpointed
    (out_minlen * grad_out).sum().backward()
    assert len(call_log) == 3, f"Expected 3 uncheckpointed segment calls when T < min_len, got {len(call_log)}"


def test_looped_virtual_layer_hssd_zero_scalar_sync_and_lazy_stats(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify full `ToyForCausalLM` + `apply_hssd` + `apply_loop(checkpoint=True)` runs zero sync stalls."""
    torch.manual_seed(314)
    tok = ToyTokenizer()
    cfg = ToyConfig(
        vocab_size=tok.vocab_size,
        hidden_size=32,
        moe_intermediate_size=16,
        shared_expert_intermediate_size=16,
        num_hidden_layers=4,
        layer_types=["linear_attention", "linear_attention", "full_attention", "linear_attention"],
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        linear_num_heads=4,
        linear_head_dim=8,
        num_experts=4,
        num_experts_per_tok=2,
    )
    model = ToyForCausalLM(cfg)
    hssd_handle = apply_hssd(
        model,
        HSSDConfig(
            rank=16,
            chunk_size=16,
            window_chunks=2,
            num_pages=8,
            k_pages=2,
            segment_chunks=2,
            checkpoint_segments=True,
            checkpoint_segments_min_len=0,
            write_mode="full",
        ),
    )
    apply_loop(
        model,
        LoopSpec(
            prelude_end=1,
            core_end=3,
            num_passes=2,
            target_virtual_layers=6,
            checkpoint=True,
        ),
    )

    orig_run_group = hssd_mod._run_chunk_group
    orig_chunk_hssd = hssd_mod.chunk_hssd
    group_calls = [0]
    hssd_nonzero_calls = [0]

    def _track_group(*args: Any, **kwargs: Any) -> Any:
        group_calls[0] += 1
        return orig_run_group(*args, **kwargs)

    def _track_chunk_hssd(*args: Any, **kwargs: Any) -> Any:
        with _SyncOpCounter() as hssd_sc:
            res = orig_chunk_hssd(*args, **kwargs)
        hssd_nonzero_calls[0] += hssd_sc.nonzero_calls
        return res

    monkeypatch.setattr(hssd_mod, "_run_chunk_group", _track_group)
    monkeypatch.setattr(hssd_mod, "chunk_hssd", _track_chunk_hssd)

    ids = torch.randint(0, 64, (2, 64))  # 64 tokens = 4 chunks = 2 segments per linear layer
    with _SyncOpCounter() as sync_cnt:
        out = model(input_ids=ids, labels=ids)
        loss = out.loss + hssd_handle.aux_loss()
        loss.backward()

    assert hssd_nonzero_calls[0] == 0
    assert sync_cnt.scalar_item_calls == 0
    # 4 virtual linear passes (layer 0 prelude + layer 1 pass1 + layer 1 pass2 + layer 3 coda):
    # All 4 are VirtualLayer-wrapped -> 2 segments * 2 (forward + outer recompute, NOT 3!) = 4 calls each (16 total)
    assert group_calls[0] == 16, f"Expected 16 segment calls (no 3x nested recompute), got {group_calls[0]}"

    stats = hssd_handle.stats()
    assert set(stats) == {
        "hssd/budget_aux_loss",
        "hssd/vault_duty",
        "hssd/scratch_duty",
        "hssd/dense_router_prob",
        "hssd/dead_page_count",
    }


def test_standalone_hf_export_uses_fast_wy_solver_with_exact_parity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify standalone HF export uses `_solve_unit_lower_wy` (`0` `solve_triangular` calls) with `< 1e-5` diff."""
    torch.manual_seed(515)
    tok = ToyTokenizer()
    tcfg = ToyConfig(
        vocab_size=tok.vocab_size,
        hidden_size=32,
        moe_intermediate_size=16,
        shared_expert_intermediate_size=16,
        num_hidden_layers=3,
        layer_types=("linear_attention", "full_attention", "linear_attention"),
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        linear_num_heads=4,
        linear_head_dim=8,
        num_experts=4,
        num_experts_per_tok=2,
    )
    model = ToyForCausalLM(tcfg).eval()
    hssd_h = apply_hssd(
        model,
        HSSDConfig(
            rank=16,
            chunk_size=16,
            window_chunks=2,
            num_pages=8,
            k_pages=2,
            segment_chunks=2,
            checkpoint_segments=True,
            checkpoint_segments_min_len=64,
            write_mode="full",
        ),
    )
    loop_h = apply_loop(
        model,
        LoopSpec(
            prelude_end=1,
            core_end=2,
            num_passes=2,
            target_virtual_layers=4,
        ),
    )
    gen = torch.Generator().manual_seed(808)
    with torch.no_grad():
        for ad in hssd_h.adapters.values():
            for p in ad.parameters():
                p.copy_(0.15 * torch.randn(p.shape, generator=gen, dtype=p.dtype))
    model.eval()

    ids = torch.randint(0, tok.vocab_size, (2, 48), generator=gen)
    with torch.no_grad():
        ref_logits = model(input_ids=ids, use_cache=False).logits.detach().clone()

    stack = Stack(
        model=model,
        tokenizer=tok,
        parts=inspect_model(model),
        loop=loop_h,
        hssd=hssd_h,
        lsa=None,
        deltas=None,
        load_info={"name": "toy"},
        hot_map={},
    )
    pack_dir = tmp_path / "pack"
    export_whole_model_pack(stack, pack_dir, merge_in_place=False)

    standalone_mod = _load_standalone_without_qloop(pack_dir, monkeypatch)
    standalone_model = standalone_mod.Qwen36LoopHSSDForCausalLM.from_pretrained(str(pack_dir), strict=True).eval()

    def _forbid_trsm(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("Standalone HF prefill must use _solve_unit_lower_wy for C <= 64!")

    orig_sa_group = standalone_mod._run_chunk_group
    sa_hssd_nonzero = [0]
    sa_hssd_scalar_items = [0]
    sa_wy_calls = [0]
    orig_sa_wy = standalone_mod._solve_unit_lower_wy

    def _track_sa_wy(*args: Any, **kwargs: Any) -> Any:
        sa_wy_calls[0] += 1
        return orig_sa_wy(*args, **kwargs)

    def _track_sa_group(*args: Any, **kwargs: Any) -> Any:
        with _SyncOpCounter() as sa_sc:
            res = orig_sa_group(*args, **kwargs)
        sa_hssd_nonzero[0] += sa_sc.nonzero_calls
        sa_hssd_scalar_items[0] += sa_sc.scalar_item_calls
        return res

    monkeypatch.setattr(torch.linalg, "solve_triangular", _forbid_trsm)
    monkeypatch.setattr(standalone_mod, "_solve_unit_lower_wy", _track_sa_wy)
    monkeypatch.setattr(standalone_mod, "_run_chunk_group", _track_sa_group)
    with torch.no_grad():
        st_logits = standalone_model(input_ids=ids, use_cache=False).logits
    assert sa_wy_calls[0] > 0
    assert sa_hssd_nonzero[0] == 0
    assert sa_hssd_scalar_items[0] == 0
    max_diff = float((st_logits - ref_logits).abs().max().detach())
    assert max_diff < 1e-5, f"Standalone HF export max diff {max_diff}"


def test_solve_unit_lower_wy_throughput_benchmark() -> None:
    """Benchmark `_solve_unit_lower_wy` vs `torch.linalg.solve_triangular` at `[16, 32, 64, 64]`."""
    def _noop_sync() -> None:
        return None

    if torch.cuda.is_available():
        device = torch.device("cuda")
        sync_fn = torch.cuda.synchronize
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        sync_fn = torch.mps.synchronize
    else:
        device = torch.device("cpu")
        sync_fn = _noop_sync

    torch.manual_seed(99)
    l_mat = (0.15 * torch.randn(16, 32, 64, 64, device=device, dtype=torch.float32)).tril(-1)
    rhs = torch.randn(16, 32, 64, 64, device=device, dtype=torch.float32)
    eye = torch.eye(64, device=device, dtype=torch.float32)

    with torch.no_grad():
        out_fast = _solve_unit_lower_wy(l_mat, rhs)
        out_ref = torch.linalg.solve_triangular(eye + l_mat, rhs, upper=False, unitriangular=True)
    sync_fn()
    assert float((out_fast.cpu() - out_ref.cpu()).abs().max()) < 1e-5

    n_warm, n_iters = (3, 10) if device.type != "cpu" else (2, 5)
    for _ in range(n_warm):
        l1 = l_mat.detach().requires_grad_(True)
        r1 = rhs.detach().requires_grad_(True)
        _solve_unit_lower_wy(l1, r1).sum().backward()
        l2 = l_mat.detach().requires_grad_(True)
        r2 = rhs.detach().requires_grad_(True)
        torch.linalg.solve_triangular(eye + l2, r2, upper=False, unitriangular=True).sum().backward()
    sync_fn()

    t0 = time.perf_counter()
    for _ in range(n_iters):
        l1 = l_mat.detach().requires_grad_(True)
        r1 = rhs.detach().requires_grad_(True)
        _solve_unit_lower_wy(l1, r1).sum().backward()
    sync_fn()
    dt_fast = (time.perf_counter() - t0) / n_iters

    t0 = time.perf_counter()
    for _ in range(n_iters):
        l2 = l_mat.detach().requires_grad_(True)
        r2 = rhs.detach().requires_grad_(True)
        torch.linalg.solve_triangular(eye + l2, r2, upper=False, unitriangular=True).sum().backward()
    sync_fn()
    dt_trsm = (time.perf_counter() - t0) / n_iters

    assert dt_fast > 0.0 and dt_trsm > 0.0
    if device.type in ("cuda", "mps"):
        assert dt_fast < dt_trsm, (
            f"Expected GPU speedup for _solve_unit_lower_wy ({dt_fast * 1e3:.2f} ms) "
            f"vs solve_triangular ({dt_trsm * 1e3:.2f} ms) on {device.type}"
        )
