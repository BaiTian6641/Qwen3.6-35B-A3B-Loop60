from __future__ import annotations

import ast
import importlib.util
import sys
import types
from pathlib import Path
from typing import Any

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from qloop.data.tokenizer import ToyTokenizer
from qloop.export_pack import export_whole_model_pack
from qloop.infer.hf_custom_modeling import _STANDALONE_CONFIG_SRC, _STANDALONE_MODELING_SRC
from qloop.model.hssd import (
    HSSDCacheState,
    HSSDConfig,
    HSSDHybridCache,
    PeakBoltzmannGate,
    _wy_kda_state_update,
    apply_hssd,
    chunk_hssd,
    recurrent_hssd,
    recurrent_hssd_step,
)
from qloop.model.introspect import inspect_model
from qloop.model.probes import profile_hssd_layer
from qloop.model.toy import ToyConfig, ToyForCausalLM
from qloop.train.stack import Stack


def _make_random_hssd_inputs(
    *,
    bsz: int,
    seq_len: int,
    hv: int,
    dk: int,
    dv: int,
    num_pages: int,
    dtype: torch.dtype,
    seed: int = 1234,
) -> dict[str, Any]:
    gen = torch.Generator().manual_seed(seed)
    q = torch.randn(bsz, seq_len, hv, dk, generator=gen, dtype=torch.float64).to(dtype)
    k = torch.randn(bsz, seq_len, hv, dk, generator=gen, dtype=torch.float64).to(dtype)
    v = torch.randn(bsz, seq_len, hv, dv, generator=gen, dtype=torch.float64).to(dtype)
    q = torch.nn.functional.normalize(q, dim=-1)
    k = torch.nn.functional.normalize(k, dim=-1)

    ps = (torch.rand(num_pages, generator=gen, dtype=torch.float64) * 0.04 + 1e-3).to(dtype)
    cp = (torch.rand(bsz, seq_len, hv, dk // 2, generator=gen, dtype=torch.float64) * 0.5 + 0.05).to(dtype)
    b_gate = (torch.rand(bsz, seq_len, hv, dk, generator=gen, dtype=torch.float64) * 1.4 + 0.1).to(dtype)
    w_gate = (torch.rand(bsz, seq_len, hv, dv, generator=gen, dtype=torch.float64) * 0.8 + 0.1).to(dtype)
    theta = (torch.randn(bsz, seq_len, hv, dk // 2, generator=gen, dtype=torch.float64) * 0.3).to(dtype)
    sb = (torch.randn(bsz, seq_len, hv, generator=gen, dtype=torch.float64) * 0.25).to(dtype)
    slp = (torch.rand(hv, generator=gen, dtype=torch.float64) * 2.0 + 1.5).to(dtype)
    thr = (torch.rand(hv, generator=gen, dtype=torch.float64) * 0.4 + 0.5).to(dtype)
    wl = torch.randn(bsz, seq_len, num_pages, generator=gen, dtype=torch.float64).to(dtype)
    rl = torch.randn(bsz, seq_len, num_pages, generator=gen, dtype=torch.float64).to(dtype)
    q_swa = torch.randn(bsz, seq_len, hv, dk, generator=gen, dtype=torch.float64).to(dtype)
    k_swa = torch.randn(bsz, seq_len, hv, dk, generator=gen, dtype=torch.float64).to(dtype)
    v_swa = torch.randn(bsz, seq_len, hv, dv, generator=gen, dtype=torch.float64).to(dtype)
    gate = PeakBoltzmannGate(hv, kappa_init=2.1, tau_init=1.8).to(dtype=dtype)
    return {
        "q": q,
        "k": k,
        "v": v,
        "gamma": (ps, cp),
        "b_gate": b_gate,
        "w_gate": w_gate,
        "theta": theta,
        "sb": sb,
        "slp": slp,
        "thr": thr,
        "wl": wl,
        "rl": rl,
        "q_swa": q_swa,
        "k_swa": k_swa,
        "v_swa": v_swa,
        "gate": gate,
    }


def _load_standalone_module_without_qloop(pack_dir: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    cfg_path = pack_dir / "configuration_qwen36_loophssd.py"
    mod_path = pack_dir / "modeling_qwen36_loophssd.py"

    saved_qloop = {k: v for k, v in sys.modules.items() if k == "qloop" or k.startswith("qloop.")}
    for k in list(saved_qloop):
        monkeypatch.delitem(sys.modules, k, raising=False)

    orig_find_spec = importlib.util.find_spec

    def _hide_qloop(name: str, package: str | None = None):
        if name == "qloop" or name.startswith("qloop."):
            return None
        return orig_find_spec(name, package)

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

    pkg_name = f"_sa_fast_pkg_{abs(hash(str(pack_dir))) % (10**8)}"
    pkg = types.ModuleType(pkg_name)
    pkg.__path__ = [str(pack_dir)]  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, pkg_name, pkg)
    cfg_mod_name = f"{pkg_name}.configuration_qwen36_loophssd"
    mdl_mod_name = f"{pkg_name}.modeling_qwen36_loophssd"

    cfg_spec = importlib.util.spec_from_file_location(cfg_mod_name, cfg_path)
    assert cfg_spec is not None and cfg_spec.loader is not None
    cfg_mod = importlib.util.module_from_spec(cfg_spec)
    monkeypatch.setitem(sys.modules, cfg_mod_name, cfg_mod)
    monkeypatch.setitem(sys.modules, "configuration_qwen36_loophssd", cfg_mod)
    cfg_spec.loader.exec_module(cfg_mod)

    mdl_spec = importlib.util.spec_from_file_location(mdl_mod_name, mod_path)
    assert mdl_spec is not None and mdl_spec.loader is not None
    mdl_mod = importlib.util.module_from_spec(mdl_spec)
    monkeypatch.setitem(sys.modules, mdl_mod_name, mdl_mod)
    mdl_spec.loader.exec_module(mdl_mod)
    assert mdl_mod._HAS_QLOOP is False
    return mdl_mod


class _TensorAllocRecorder(TorchDispatchMode):
    def __init__(self) -> None:
        super().__init__()
        self.allocated_shapes_dtypes: list[tuple[tuple[int, ...], torch.dtype]] = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = {} if kwargs is None else kwargs
        out = func(*args, **kwargs)
        if isinstance(out, torch.Tensor):
            self.allocated_shapes_dtypes.append((tuple(out.shape), out.dtype))
        elif isinstance(out, (tuple, list)):
            for item in out:
                if isinstance(item, torch.Tensor):
                    self.allocated_shapes_dtypes.append((tuple(item.shape), item.dtype))
        return out


def test_hssd_config_segment_chunks_default_is_four() -> None:
    cfg = HSSDConfig()
    assert cfg.segment_chunks == 4
    cfg_dict = HSSDConfig.from_dict({"rank": 32})
    assert cfg_dict.segment_chunks == 4


@pytest.mark.parametrize("surprise_mode", ["routed", "page0"])
def test_fp64_exact_parity_aligned_and_unaligned_batched_swa_and_vectorized_wy(surprise_mode: str) -> None:
    bsz, hv, dk, dv = 2, 3, 16, 16
    num_pages, top_k_pages = 8, 2
    chunk_size, window_chunks = 64, 8

    # 1. Aligned T=256 (4 full C=64 chunks -> batched .unfold SWA + 28-pair Block-8 WY + active-page slice)
    inp_256 = _make_random_hssd_inputs(
        bsz=bsz, seq_len=256, hv=hv, dk=dk, dv=dv, num_pages=num_pages, dtype=torch.float64, seed=2026
    )
    with torch.no_grad():
        out_chunk_ng, st_chunk_ng, duty_chunk_ng = chunk_hssd(
            inp_256["q"],
            inp_256["k"],
            inp_256["v"],
            inp_256["gamma"],
            inp_256["b_gate"],
            inp_256["w_gate"],
            inp_256["theta"],
            inp_256["sb"],
            inp_256["slp"],
            inp_256["thr"],
            inp_256["wl"],
            inp_256["rl"],
            inp_256["q_swa"],
            inp_256["k_swa"],
            inp_256["v_swa"],
            boltzmann_gate=inp_256["gate"],
            chunk_size=chunk_size,
            window_chunks=window_chunks,
            num_pages=num_pages,
            top_k_pages=top_k_pages,
            segment_chunks=4,
            checkpoint_segments=False,
            output_final_state=True,
            surprise_mode=surprise_mode,
            return_aux=True,
        )
        out_rec, st_rec, duty_rec = recurrent_hssd(
            inp_256["q"],
            inp_256["k"],
            inp_256["v"],
            inp_256["gamma"],
            inp_256["b_gate"],
            inp_256["w_gate"],
            inp_256["theta"],
            inp_256["sb"],
            inp_256["slp"],
            inp_256["thr"],
            inp_256["wl"],
            inp_256["rl"],
            inp_256["q_swa"],
            inp_256["k_swa"],
            inp_256["v_swa"],
            boltzmann_gate=inp_256["gate"],
            chunk_size=chunk_size,
            window_chunks=window_chunks,
            num_pages=num_pages,
            top_k_pages=top_k_pages,
            output_final_state=True,
            surprise_mode=surprise_mode,
            return_aux=True,
        )

        st_step = HSSDCacheState.zeros(
            bsz, num_pages, hv, dk, dv, chunk_size=chunk_size, window_chunks=window_chunks, dtype=torch.float64
        )
        step_outs = []
        ps_g, cp_g = inp_256["gamma"]
        for t in range(256):
            sl = slice(t, t + 1)
            o_t, st_step = recurrent_hssd_step(
                inp_256["q"][:, sl],
                inp_256["k"][:, sl],
                inp_256["v"][:, sl],
                (ps_g, cp_g[:, sl]),
                inp_256["b_gate"][:, sl],
                inp_256["w_gate"][:, sl],
                inp_256["theta"][:, sl],
                inp_256["sb"][:, sl],
                inp_256["slp"],
                inp_256["thr"],
                inp_256["wl"][:, sl],
                inp_256["rl"][:, sl],
                inp_256["q_swa"][:, sl],
                inp_256["k_swa"][:, sl],
                inp_256["v_swa"][:, sl],
                st_step,
                boltzmann_gate=inp_256["gate"],
                top_k_pages=top_k_pages,
                surprise_mode=surprise_mode,
            )
            step_outs.append(o_t)
        out_steps = torch.cat(step_outs, dim=1)

    # Also check grad-enabled path (full-bank WY + segment checkpointing across segment_chunks=2)
    q_grad = inp_256["q"].clone().requires_grad_(True)
    out_chunk_grad, st_chunk_grad = chunk_hssd(
        q_grad,
        inp_256["k"],
        inp_256["v"],
        inp_256["gamma"],
        inp_256["b_gate"],
        inp_256["w_gate"],
        inp_256["theta"],
        inp_256["sb"],
        inp_256["slp"],
        inp_256["thr"],
        inp_256["wl"],
        inp_256["rl"],
        inp_256["q_swa"],
        inp_256["k_swa"],
        inp_256["v_swa"],
        boltzmann_gate=inp_256["gate"],
        chunk_size=chunk_size,
        window_chunks=window_chunks,
        num_pages=num_pages,
        top_k_pages=top_k_pages,
        segment_chunks=2,
        checkpoint_segments=True,
        output_final_state=True,
        surprise_mode=surprise_mode,
    )

    assert st_chunk_ng is not None and st_rec is not None and st_chunk_grad is not None
    assert float((out_chunk_ng - out_rec).abs().max()) < 1e-10
    assert float((out_steps - out_rec).abs().max()) < 1e-10
    assert float((out_chunk_grad.detach() - out_rec).abs().max()) < 1e-10
    assert float((duty_chunk_ng - duty_rec).abs().max()) < 1e-12
    assert st_chunk_ng.state_equal(st_rec, atol=1e-10, rtol=1e-10)
    assert st_step.state_equal(st_rec, atol=1e-10, rtol=1e-10)
    assert st_chunk_grad.detach().state_equal(st_rec, atol=1e-10, rtol=1e-10)

    # 2. Unaligned streaming: step_offset=17, followed by T=150 (total 167 tokens)
    inp_167 = _make_random_hssd_inputs(
        bsz=bsz, seq_len=167, hv=hv, dk=dk, dv=dv, num_pages=num_pages, dtype=torch.float64, seed=3030
    )
    ps167, cp167 = inp_167["gamma"]
    with torch.no_grad():
        o_pre17, st_17 = chunk_hssd(
            inp_167["q"][:, :17],
            inp_167["k"][:, :17],
            inp_167["v"][:, :17],
            (ps167, cp167[:, :17]),
            inp_167["b_gate"][:, :17],
            inp_167["w_gate"][:, :17],
            inp_167["theta"][:, :17],
            inp_167["sb"][:, :17],
            inp_167["slp"],
            inp_167["thr"],
            inp_167["wl"][:, :17],
            inp_167["rl"][:, :17],
            inp_167["q_swa"][:, :17],
            inp_167["k_swa"][:, :17],
            inp_167["v_swa"][:, :17],
            boltzmann_gate=inp_167["gate"],
            chunk_size=chunk_size,
            window_chunks=window_chunks,
            num_pages=num_pages,
            top_k_pages=top_k_pages,
            output_final_state=True,
            surprise_mode=surprise_mode,
        )
        assert st_17 is not None and st_17.step_count_int == 17

        o_rem150_chunk, st_167_chunk = chunk_hssd(
            inp_167["q"][:, 17:],
            inp_167["k"][:, 17:],
            inp_167["v"][:, 17:],
            (ps167, cp167[:, 17:]),
            inp_167["b_gate"][:, 17:],
            inp_167["w_gate"][:, 17:],
            inp_167["theta"][:, 17:],
            inp_167["sb"][:, 17:],
            inp_167["slp"],
            inp_167["thr"],
            inp_167["wl"][:, 17:],
            inp_167["rl"][:, 17:],
            inp_167["q_swa"][:, 17:],
            inp_167["k_swa"][:, 17:],
            inp_167["v_swa"][:, 17:],
            boltzmann_gate=inp_167["gate"],
            chunk_size=chunk_size,
            window_chunks=window_chunks,
            num_pages=num_pages,
            top_k_pages=top_k_pages,
            segment_chunks=2,
            initial_state=st_17,
            output_final_state=True,
            surprise_mode=surprise_mode,
        )
        o_full167_rec, st_167_rec = recurrent_hssd(
            inp_167["q"],
            inp_167["k"],
            inp_167["v"],
            inp_167["gamma"],
            inp_167["b_gate"],
            inp_167["w_gate"],
            inp_167["theta"],
            inp_167["sb"],
            inp_167["slp"],
            inp_167["thr"],
            inp_167["wl"],
            inp_167["rl"],
            inp_167["q_swa"],
            inp_167["k_swa"],
            inp_167["v_swa"],
            boltzmann_gate=inp_167["gate"],
            chunk_size=chunk_size,
            window_chunks=window_chunks,
            num_pages=num_pages,
            top_k_pages=top_k_pages,
            output_final_state=True,
            surprise_mode=surprise_mode,
        )
    assert st_167_chunk is not None and st_167_rec is not None
    o_stream_167 = torch.cat([o_pre17, o_rem150_chunk], dim=1)
    assert float((o_stream_167 - o_full167_rec).abs().max()) < 1e-10
    assert st_167_chunk.state_equal(st_167_rec, atol=1e-10, rtol=1e-10)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("c_len", [64, 53, 6])
def test_wy_kda_span_10000_extreme_decay_stability_zero_bool_sync_and_inference_mode(
    dtype: torch.dtype, c_len: int
) -> None:
    gen = torch.Generator().manual_seed(777 + c_len)
    bsz, mp, hv, dk, dv = 2, 4, 2, 16, 16
    per_step_decay = -10500.0 / float(c_len)  # total cumulative span = 10,500 > 10,000!

    p0 = torch.randn(bsz, mp, hv, dk, dv, generator=gen, dtype=dtype, requires_grad=True)
    ke = (torch.randn(bsz, mp, hv, c_len, dk, generator=gen, dtype=dtype) * 0.2).detach().requires_grad_(True)
    ve = torch.randn(bsz, mp, hv, c_len, dv, generator=gen, dtype=dtype, requires_grad=True)
    la = (
        torch.full((bsz, mp, hv, c_len, dk), per_step_decay, dtype=dtype)
        - torch.rand(bsz, mp, hv, c_len, dk, generator=gen, dtype=dtype) * 5.0
    ).detach().requires_grad_(True)

    # First populate cache under torch.inference_mode() to prove zero inference-tensor taint
    with torch.inference_mode():
        out_inf = _wy_kda_state_update(p0.detach(), ke.detach(), ve.detach(), la.detach())
        assert torch.isfinite(out_inf).all()

    # Forbid Tensor.__bool__ during forward to prove zero host syncs inside _wy_kda_state_update
    orig_bool = torch.Tensor.__bool__

    def _forbid_bool(self: torch.Tensor) -> bool:
        raise AssertionError("Unexpected Tensor.__bool__ host sync inside _wy_kda_state_update")

    torch.Tensor.__bool__ = _forbid_bool  # type: ignore[method-assign]
    try:
        p_next = _wy_kda_state_update(p0, ke, ve, la)
    finally:
        torch.Tensor.__bool__ = orig_bool  # type: ignore[method-assign]

    assert torch.isfinite(p_next).all()
    loss = (p_next * torch.randn_like(p_next)).sum()
    loss.backward()

    for name, grad in (("p0", p0.grad), ("ke", ke.grad), ("ve", ve.grad), ("la", la.grad)):
        assert grad is not None, f"missing grad for {name}"
        assert torch.isfinite(grad).all(), f"non-finite grad for {name} at span=10,500"


def test_bfloat16_decode_step_compact_no_aliasing_parity_and_zero_full_state_upcast() -> None:
    bsz, hv, dk, dv = 1, 4, 32, 32
    num_pages, top_k_pages = 8, 2
    chunk_size, window_chunks = 64, 8
    n_steps = 70  # crosses chunk boundary 64

    inp = _make_random_hssd_inputs(
        bsz=bsz, seq_len=n_steps, hv=hv, dk=dk, dv=dv, num_pages=num_pages, dtype=torch.bfloat16, seed=9001
    )
    ps_g, cp_g = inp["gamma"]

    st = HSSDCacheState.zeros(
        bsz, num_pages, hv, dk, dv, chunk_size=chunk_size, window_chunks=window_chunks, dtype=torch.bfloat16
    )
    assert st.dtype == torch.bfloat16
    assert st.theta_cum.dtype == torch.float64
    assert st.has_compact_storage()

    step_outs: list[torch.Tensor] = []
    chunk1_outs: list[torch.Tensor] = []
    st_c1 = st.clone()

    full_bank_shape = (bsz, num_pages, hv, dk, dv)
    full_k_ring_shape = (bsz, (window_chunks + 1) * chunk_size, hv, dk)
    full_v_ring_shape = (bsz, (window_chunks + 1) * chunk_size, hv, dv)

    with torch.no_grad():
        for t in range(n_steps):
            sl = slice(t, t + 1)
            mid_ring_ptrs_before = [int(b.untyped_storage().data_ptr()) for b in st.boundary_ring[1:-1]]

            if t == 65:
                with _TensorAllocRecorder() as rec_step:
                    o_t, st = recurrent_hssd_step(
                        inp["q"][:, sl],
                        inp["k"][:, sl],
                        inp["v"][:, sl],
                        (ps_g, cp_g[:, sl]),
                        inp["b_gate"][:, sl],
                        inp["w_gate"][:, sl],
                        inp["theta"][:, sl],
                        inp["sb"][:, sl],
                        inp["slp"],
                        inp["thr"],
                        inp["wl"][:, sl],
                        inp["rl"][:, sl],
                        inp["q_swa"][:, sl],
                        inp["k_swa"][:, sl],
                        inp["v_swa"][:, sl],
                        st,
                        boltzmann_gate=inp["gate"],
                        top_k_pages=top_k_pages,
                        state_dtype="bfloat16",
                    )
                fp32_allocs = [s for s, dt in rec_step.allocated_shapes_dtypes if dt == torch.float32]
                assert full_bank_shape not in fp32_allocs, f"Unexpected full 8-page fp32 alloc: {fp32_allocs}"
                assert full_k_ring_shape not in fp32_allocs, f"Unexpected full 576-slot fp32 k_ring: {fp32_allocs}"
                assert full_v_ring_shape not in fp32_allocs, f"Unexpected full 576-slot fp32 v_ring: {fp32_allocs}"
            else:
                o_t, st = recurrent_hssd_step(
                    inp["q"][:, sl],
                    inp["k"][:, sl],
                    inp["v"][:, sl],
                    (ps_g, cp_g[:, sl]),
                    inp["b_gate"][:, sl],
                    inp["w_gate"][:, sl],
                    inp["theta"][:, sl],
                    inp["sb"][:, sl],
                    inp["slp"],
                    inp["thr"],
                    inp["wl"][:, sl],
                    inp["rl"][:, sl],
                    inp["q_swa"][:, sl],
                    inp["k_swa"][:, sl],
                    inp["v_swa"][:, sl],
                    st,
                    boltzmann_gate=inp["gate"],
                    top_k_pages=top_k_pages,
                    state_dtype="bfloat16",
                )

            with _TensorAllocRecorder() as rec_c1:
                o_c1, st_c1_next = chunk_hssd(
                    inp["q"][:, sl],
                    inp["k"][:, sl],
                    inp["v"][:, sl],
                    (ps_g, cp_g[:, sl]),
                    inp["b_gate"][:, sl],
                    inp["w_gate"][:, sl],
                    inp["theta"][:, sl],
                    inp["sb"][:, sl],
                    inp["slp"],
                    inp["thr"],
                    inp["wl"][:, sl],
                    inp["rl"][:, sl],
                    inp["q_swa"][:, sl],
                    inp["k_swa"][:, sl],
                    inp["v_swa"][:, sl],
                    boltzmann_gate=inp["gate"],
                    chunk_size=chunk_size,
                    window_chunks=window_chunks,
                    num_pages=num_pages,
                    top_k_pages=top_k_pages,
                    initial_state=st_c1,
                    output_final_state=True,
                    state_dtype="bfloat16",
                )
            assert st_c1_next is not None
            st_c1 = st_c1_next
            fp32_c1 = [s for s, dt in rec_c1.allocated_shapes_dtypes if dt == torch.float32]
            assert full_bank_shape not in fp32_c1

            step_outs.append(o_t)
            chunk1_outs.append(o_c1)

            assert st.dtype == torch.bfloat16
            assert st.theta_cum.dtype == torch.float64
            assert st.has_compact_storage(), f"step {t + 1}: state is not compact"
            page_ptr = int(st.page_states.untyped_storage().data_ptr())
            ring_ptrs = {int(b.untyped_storage().data_ptr()) for b in st.boundary_ring}
            assert page_ptr not in ring_ptrs, f"step {t + 1}: page_states aliases boundary_ring!"

            if (t + 1) % chunk_size != 0:
                mid_ring_ptrs_after = [int(b.untyped_storage().data_ptr()) for b in st.boundary_ring[1:-1]]
                assert mid_ring_ptrs_after == mid_ring_ptrs_before

        out_rec_bf16, st_rec_bf16 = recurrent_hssd(
            inp["q"],
            inp["k"],
            inp["v"],
            inp["gamma"],
            inp["b_gate"],
            inp["w_gate"],
            inp["theta"],
            inp["sb"],
            inp["slp"],
            inp["thr"],
            inp["wl"],
            inp["rl"],
            inp["q_swa"],
            inp["k_swa"],
            inp["v_swa"],
            boltzmann_gate=inp["gate"],
            chunk_size=chunk_size,
            window_chunks=window_chunks,
            num_pages=num_pages,
            top_k_pages=top_k_pages,
            output_final_state=True,
            state_dtype="bfloat16",
        )

        # Also check 0.0-ULP single-step decode parity vs upcast state on current-step SWA k/v
        o_ref_up, _ = recurrent_hssd_step(
            inp["q"][:, 64:65],
            inp["k"][:, 64:65],
            inp["v"][:, 64:65],
            (ps_g, cp_g[:, 64:65]),
            inp["b_gate"][:, 64:65],
            inp["w_gate"][:, 64:65],
            inp["theta"][:, 64:65],
            inp["sb"][:, 64:65],
            inp["slp"],
            inp["thr"],
            inp["wl"][:, 64:65],
            inp["rl"][:, 64:65],
            inp["q_swa"][:, 64:65],
            inp["k_swa"][:, 64:65],
            inp["v_swa"][:, 64:65],
            st_rec_bf16.to(torch.float32),  # type: ignore[union-attr]
            boltzmann_gate=inp["gate"],
            top_k_pages=top_k_pages,
            state_dtype="auto",
        )
        o_fast_bf16, _ = recurrent_hssd_step(
            inp["q"][:, 64:65],
            inp["k"][:, 64:65],
            inp["v"][:, 64:65],
            (ps_g, cp_g[:, 64:65]),
            inp["b_gate"][:, 64:65],
            inp["w_gate"][:, 64:65],
            inp["theta"][:, 64:65],
            inp["sb"][:, 64:65],
            inp["slp"],
            inp["thr"],
            inp["wl"][:, 64:65],
            inp["rl"][:, 64:65],
            inp["q_swa"][:, 64:65],
            inp["k_swa"][:, 64:65],
            inp["v_swa"][:, 64:65],
            st_rec_bf16,  # type: ignore[arg-type]
            boltzmann_gate=inp["gate"],
            top_k_pages=top_k_pages,
            state_dtype="bfloat16",
        )
        assert torch.equal(o_fast_bf16, o_ref_up)

    assert st_rec_bf16 is not None
    out_steps_cat = torch.cat(step_outs, dim=1)
    out_c1_cat = torch.cat(chunk1_outs, dim=1)
    assert torch.equal(out_steps_cat, out_rec_bf16)
    assert torch.equal(out_c1_cat, out_rec_bf16)
    assert st.state_equal(st_rec_bf16, atol=0.0, rtol=0.0)
    assert st_c1.state_equal(st_rec_bf16, atol=0.0, rtol=0.0)

    # Also verify `_make_hssd_rule_wrapper` via `apply_hssd` does zero full-state fp32 upcast on seq_len=1 decode
    tok = ToyTokenizer()
    model = ToyForCausalLM(ToyConfig(vocab_size=tok.vocab_size, hidden_size=32, num_hidden_layers=2)).to(
        torch.bfloat16
    )
    model.eval()
    apply_hssd(
        model,
        HSSDConfig(
            rank=8,
            num_pages=8,
            k_pages=2,
            chunk_size=16,
            window_chunks=2,
            state_dtype="bfloat16",
        ),
    )
    cache = HSSDHybridCache(num_layers=2)
    ids = torch.randint(0, tok.vocab_size, (1, 18))
    with torch.no_grad():
        model(ids[:, :17], past_key_values=cache, use_cache=True)
        lin_st = next(s for s in cache.recurrent_states if isinstance(s, HSSDCacheState))
        model_bank_shape = tuple(lin_st.page_states.shape)
        with _TensorAllocRecorder() as rec_wrap:
            model(ids[:, 17:18], past_key_values=cache, use_cache=True)
    wrap_fp32 = [s for s, dt in rec_wrap.allocated_shapes_dtypes if dt == torch.float32]
    assert model_bank_shape not in wrap_fp32


def test_gradcheck_vectorized_wy_batched_swa_chunk_and_step() -> None:
    gen = torch.Generator().manual_seed(5150)

    # 1. gradcheck on vectorized Block-8 `_wy_kda_state_update` with c_len=20 (2 full 8-blocks + partial 4-block)
    p0 = (torch.randn(1, 2, 2, 4, 4, generator=gen, dtype=torch.float64) * 0.3).requires_grad_(True)
    ke = (torch.randn(1, 2, 2, 20, 4, generator=gen, dtype=torch.float64) * 0.25).requires_grad_(True)
    ve = (torch.randn(1, 2, 2, 20, 4, generator=gen, dtype=torch.float64) * 0.3).requires_grad_(True)
    la = (-torch.rand(1, 2, 2, 20, 4, generator=gen, dtype=torch.float64) * 0.5 - 0.05).requires_grad_(True)
    assert torch.autograd.gradcheck(_wy_kda_state_update, (p0, ke, ve, la), eps=1e-6, atol=1e-5, rtol=1e-4)

    # 2. gradcheck on `chunk_hssd` exercising batched `.unfold` SWA + checkpointed & uncheckpointed segments
    inp = _make_random_hssd_inputs(
        bsz=1, seq_len=16, hv=2, dk=4, dv=4, num_pages=3, dtype=torch.float64, seed=6160
    )
    q = inp["q"].clone().requires_grad_(True)
    k = inp["k"].clone().requires_grad_(True)
    v = inp["v"].clone().requires_grad_(True)
    qs = inp["q_swa"].clone().requires_grad_(True)
    ks = inp["k_swa"].clone().requires_grad_(True)
    vs = inp["v_swa"].clone().requires_grad_(True)
    ps_g, cp_g = inp["gamma"]
    cp_req = cp_g.clone().requires_grad_(True)

    for ckpt in (False, True):

        def _chunk_fn(
            q_: torch.Tensor,
            k_: torch.Tensor,
            v_: torch.Tensor,
            cp_: torch.Tensor,
            qs_: torch.Tensor,
            ks_: torch.Tensor,
            vs_: torch.Tensor,
            _ckpt: bool = ckpt,
        ) -> torch.Tensor:
            o, st_f = chunk_hssd(
                q_,
                k_,
                v_,
                (ps_g, cp_),
                inp["b_gate"],
                inp["w_gate"],
                inp["theta"],
                inp["sb"],
                inp["slp"],
                inp["thr"],
                inp["wl"],
                inp["rl"],
                qs_,
                ks_,
                vs_,
                boltzmann_gate=inp["gate"],
                chunk_size=4,
                window_chunks=2,
                num_pages=3,
                top_k_pages=2,
                segment_chunks=2,
                checkpoint_segments=_ckpt,
                output_final_state=True,
            )
            assert st_f is not None
            return o.sum() + 0.25 * st_f.page_states.sum()

        assert torch.autograd.gradcheck(
            _chunk_fn, (q, k, v, cp_req, qs, ks, vs), eps=1e-6, atol=1e-5, rtol=1e-4
        )

    # 3. gradcheck on `recurrent_hssd_step`
    st_base = HSSDCacheState.zeros(1, 3, 2, 4, 4, chunk_size=4, window_chunks=2, dtype=torch.float64)
    q1 = inp["q"][:, :1].clone().requires_grad_(True)
    k1 = inp["k"][:, :1].clone().requires_grad_(True)
    v1 = inp["v"][:, :1].clone().requires_grad_(True)
    qs1 = inp["q_swa"][:, :1].clone().requires_grad_(True)
    ks1 = inp["k_swa"][:, :1].clone().requires_grad_(True)
    vs1 = inp["v_swa"][:, :1].clone().requires_grad_(True)

    def _step_fn(
        q_: torch.Tensor,
        k_: torch.Tensor,
        v_: torch.Tensor,
        qs_: torch.Tensor,
        ks_: torch.Tensor,
        vs_: torch.Tensor,
    ) -> torch.Tensor:
        o_s, st_s = recurrent_hssd_step(
            q_,
            k_,
            v_,
            (ps_g, cp_g[:, :1]),
            inp["b_gate"][:, :1],
            inp["w_gate"][:, :1],
            inp["theta"][:, :1],
            inp["sb"][:, :1],
            inp["slp"],
            inp["thr"],
            inp["wl"][:, :1],
            inp["rl"][:, :1],
            qs_,
            ks_,
            vs_,
            st_base,
            boltzmann_gate=inp["gate"],
            top_k_pages=2,
        )
        return o_s.sum() + 0.25 * st_s.page_states.sum()

    assert torch.autograd.gradcheck(_step_fn, (q1, k1, v1, qs1, ks1, vs1), eps=1e-6, atol=1e-5, rtol=1e-4)


def test_standalone_hf_custom_modeling_fast_kernels_and_pack_parity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Verify Python 3.9 AST validity of standalone sources
    ast.parse(_STANDALONE_CONFIG_SRC, filename="configuration_qwen36_loophssd.py", feature_version=(3, 9))
    ast.parse(_STANDALONE_MODELING_SRC, filename="modeling_qwen36_loophssd.py", feature_version=(3, 9))

    tok = ToyTokenizer()
    base_model = ToyForCausalLM(ToyConfig(vocab_size=tok.vocab_size, hidden_size=32, num_hidden_layers=4)).eval()
    hssd_cfg = HSSDConfig(
        rank=8,
        num_pages=8,
        k_pages=2,
        chunk_size=16,
        window_chunks=2,
        segment_chunks=2,
        state_dtype="bfloat16",
    )
    hssd_handle = apply_hssd(base_model, hssd_cfg)
    with torch.no_grad():
        gen_w = torch.Generator().manual_seed(777)
        for ad in hssd_handle.adapters.values():
            for p in ad.parameters():
                if p.ndim >= 2 and float(p.abs().max()) == 0.0:
                    p.normal_(0.0, 0.08, generator=gen_w)

    stack = Stack(
        model=base_model,
        tokenizer=tok,
        parts=inspect_model(base_model),
        hssd=hssd_handle,
        load_info={"name": "toy"},
    )
    pack_dir = tmp_path / "fast_hssd_pack"
    export_whole_model_pack(stack, pack_dir, max_shard_size_bytes=65536)

    ids = torch.randint(0, tok.vocab_size, (1, 38), generator=torch.Generator().manual_seed(42))
    with torch.no_grad():
        ref_full = base_model(ids).logits
        ref_cache = HSSDHybridCache(num_layers=4)
        ref_pre = base_model(ids[:, :34], past_key_values=ref_cache, use_cache=True).logits
        ref_dec = [
            base_model(ids[:, pos : pos + 1], past_key_values=ref_cache, use_cache=True).logits
            for pos in range(34, 38)
        ]
        ref_inc = torch.cat([ref_pre, *ref_dec], dim=1)

    sa_mod = _load_standalone_module_without_qloop(pack_dir, monkeypatch)
    assert sa_mod._StandaloneHSSDAdapter(32, 2, 8, 8, {}, 0).segment_chunks == 4

    # 1. Verify standalone fp64 chunkwise WY + batched SWA vs _rec_only vs step-by-step parity
    ad64_cfg = {
        "num_pages": 8,
        "k_pages": 2,
        "chunk_size": 16,
        "window_chunks": 2,
        "segment_chunks": 2,
        "state_dtype": "auto",
    }
    ad64 = sa_mod._StandaloneHSSDAdapter(32, 2, 8, 8, ad64_cfg, 0).to(torch.float64)
    gen64 = torch.Generator().manual_seed(8080)
    with torch.no_grad():
        for p in ad64.parameters():
            if p.ndim >= 2 and float(p.abs().max()) == 0.0:
                p.normal_(0.0, 0.12, generator=gen64)
    x64 = torch.randn(2, 48, 32, generator=gen64, dtype=torch.float64)
    q64 = torch.randn(2, 48, 2, 8, generator=gen64, dtype=torch.float64)
    k64 = torch.randn(2, 48, 2, 8, generator=gen64, dtype=torch.float64)
    v64 = torch.randn(2, 48, 2, 8, generator=gen64, dtype=torch.float64)
    g64 = -torch.rand(2, 48, 2, generator=gen64, dtype=torch.float64) * 0.3
    b64 = torch.rand(2, 48, 2, generator=gen64, dtype=torch.float64)

    with torch.no_grad():
        o_sa_chunk, st_sa_chunk = sa_mod._recurrent_hssd(
            ad64, x64, None, q64, k64, v64, g64, b64, output_final_state=True, use_qk_l2norm_in_kernel=True
        )
        o_sa_rec, st_sa_rec = sa_mod._recurrent_hssd(
            ad64,
            x64,
            None,
            q64,
            k64,
            v64,
            g64,
            b64,
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
            _rec_only=True,
        )
    assert float((o_sa_chunk - o_sa_rec).abs().max()) < 1e-10
    assert st_sa_chunk.state_equal(st_sa_rec, atol=1e-10, rtol=1e-10)

    # 2. Verify standalone bfloat16 decode does zero full-bank fp32 upcast & zero aliasing across boundary
    ad_bf16 = sa_mod._StandaloneHSSDAdapter(
        32,
        2,
        8,
        8,
        {"num_pages": 8, "k_pages": 2, "chunk_size": 16, "window_chunks": 2, "state_dtype": "bfloat16"},
        0,
    ).to(torch.bfloat16)
    x_bf = x64[:1, :20].to(torch.bfloat16)
    q_bf = q64[:1, :20].to(torch.bfloat16)
    k_bf = k64[:1, :20].to(torch.bfloat16)
    v_bf = v64[:1, :20].to(torch.bfloat16)
    g_bf = g64[:1, :20].to(torch.bfloat16)
    b_bf = b64[:1, :20].to(torch.bfloat16)
    st_bf = None
    with torch.no_grad():
        for pos in range(20):
            sl = slice(pos, pos + 1)
            with _TensorAllocRecorder() as rec_sa:
                _, st_bf = sa_mod._recurrent_hssd(
                    ad_bf16,
                    x_bf[:, sl],
                    None,
                    q_bf[:, sl],
                    k_bf[:, sl],
                    v_bf[:, sl],
                    g_bf[:, sl],
                    b_bf[:, sl],
                    initial_state=st_bf,
                    output_final_state=True,
                )
            sa_fp32 = [s for s, dt in rec_sa.allocated_shapes_dtypes if dt == torch.float32]
            assert (1, 8, 2, 8, 8) not in sa_fp32
            assert st_bf.has_compact_storage()
            assert int(st_bf.page_states.untyped_storage().data_ptr()) not in {
                int(r.untyped_storage().data_ptr()) for r in st_bf.boundary_ring
            }

    # 3. Verify exported Whole Model Pack standalone reload forward + cached decode parity
    sa_model = sa_mod.Qwen36LoopHSSDForCausalLM.from_pretrained(str(pack_dir)).eval()
    with torch.no_grad():
        sa_full = sa_model(ids).logits
        out_p = sa_model(ids[:, :34], use_cache=True)
        pkv = out_p.past_key_values
        sa_dec = []
        for pos in range(34, 38):
            out_step = sa_model(ids[:, pos : pos + 1], past_key_values=pkv, use_cache=True)
            pkv = out_step.past_key_values
            sa_dec.append(out_step.logits)
        sa_inc = torch.cat([out_p.logits, *sa_dec], dim=1)

    assert float((sa_full - ref_full).abs().max()) < 1e-4
    assert float((sa_inc - ref_inc).abs().max()) < 1e-4


def test_profile_hssd_layer_runs_cleanly_with_segment_chunks_four() -> None:
    assert HSSDConfig().segment_chunks == 4
    res = profile_hssd_layer(
        hidden_size=64,
        num_v_heads=4,
        head_k_dim=16,
        head_v_dim=16,
        seq_len=128,
        stream_chunk_len=64,
        rank=16,
        device="cpu",
        dtype=torch.float32,
        state_dtype="bfloat16",
    )
    assert res["retained_cache"]["state_dtype"] == "bfloat16"
    assert res["retained_cache"]["has_compact_storage"] is True
    assert res["single_token_decode"]["has_compact_storage"] is True
    assert res["training_fwd_bwd"]["grad_finite"] is True
    assert res["streamed_prefill"]["max_diff_vs_full_prefill"] < 2e-3

    res_fp32 = profile_hssd_layer(
        hidden_size=64,
        num_v_heads=4,
        head_k_dim=16,
        head_v_dim=16,
        seq_len=128,
        stream_chunk_len=64,
        rank=16,
        device="cpu",
        dtype=torch.float32,
        state_dtype="float32",
    )
    assert res_fp32["streamed_prefill"]["max_diff_vs_full_prefill"] < 1e-6
