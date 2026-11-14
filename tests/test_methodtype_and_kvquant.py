"""Regression tests for universal MethodType forward wrappers, QuantizedKVCache safety,
sha256_frozen_experts FP8/uint8 parameter+buffer hashing, and 3D moe_nvfp4 pack export/reload."""
from __future__ import annotations

import ast
import copy
import functools
import importlib.util
import sys
import types
from pathlib import Path
from typing import Any

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from qloop.data.tokenizer import ToyTokenizer
from qloop.export_pack import (
    export_model_pack,
    load_exported_pack,
)
from qloop.infer import CONFIG_FILENAME, MODELING_FILENAME
from qloop.model.gdn2 import GDN2Config, install_gdn2
from qloop.model.hssd import HSSDConfig, apply_hssd
from qloop.model.introspect import inspect_model
from qloop.model.loop import LoopSpec, SharedKVView, apply_loop, sha256_frozen_experts
from qloop.model.lsa import LSAConfig, install_lsa, lsa_attention
from qloop.model.quant_autograd import install_quantized_autograd_bridge
from qloop.model.toy import ToyConfig, ToyForCausalLM
from qloop.train.stack import Stack

_FP8_E4M3 = getattr(torch, "float8_e4m3fn", torch.float16)
_E2M1_POS = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def _make_hybrid_toy_cfg(vocab_size: int = 264) -> ToyConfig:
    return ToyConfig(
        vocab_size=vocab_size,
        hidden_size=32,
        moe_intermediate_size=32,
        shared_expert_intermediate_size=16,
        num_hidden_layers=6,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        partial_rotary_factor=0.5,
        linear_num_heads=4,
        linear_head_dim=8,
        linear_conv_kernel_dim=4,
        num_experts=6,
        num_experts_per_tok=2,
        layer_types=(
            "linear_attention",  # L0 prelude
            "full_attention",    # L1 prelude
            "linear_attention",  # L2 core
            "linear_attention",  # L3 core
            "full_attention",    # L4 core
            "full_attention",    # L5 coda
        ),
    )


def _assert_all_forwards_are_methodtype(root: nn.Module, *, allow_legacy_quant_bridge: bool = False) -> int:
    """Mimic compressed_tensors.offload.module.offload_module inspecting `mod.forward.__func__`."""
    count = 0
    for name, mod in root.named_modules():
        if (
            allow_legacy_quant_bridge
            and hasattr(mod, "_qloop_orig_forward")
            and not hasattr(mod, "_qloop_had_inst_forward")
        ):
            continue
        fwd = mod.forward
        assert isinstance(fwd, types.MethodType), (
            f"submodule {name!r} ({type(mod).__name__}) has non-MethodType forward: {type(fwd)}"
        )
        assert hasattr(fwd, "__func__") and callable(fwd.__func__), (
            f"submodule {name!r} ({type(mod).__name__}) forward lacks callable __func__"
        )
        assert fwd.__self__ is mod, f"submodule {name!r} forward.__self__ is not bound to module"
        count += 1
    assert count > 0
    return count


class _FrozenQuantSchemeConfig:
    """Mimics compressed-tensors Pydantic QuantizationArgs rejecting unknown attributes."""

    __slots__ = ("num_bits", "type")

    def __init__(self) -> None:
        self.num_bits = 8
        self.type = "float"


class _MockSubmoduleQuantizedKVCache(nn.Module):
    """Mimics compressed-tensors `QuantizedKVCache` attached as `self_attn.kv_cache`."""

    def __init__(self, num_kv_heads: int, k_scale: float = 0.25, v_scale: float = 0.5) -> None:
        super().__init__()
        self.config = _FrozenQuantSchemeConfig()
        self.register_buffer("k_scale", torch.full((num_kv_heads,), k_scale, dtype=torch.float32))
        self.register_buffer("v_scale", torch.full((num_kv_heads,), v_scale, dtype=torch.float32))
        self.call_count = 0

    def forward(self, k: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        self.call_count += 1
        ks = self.k_scale.view(1, -1, 1, 1).to(device=k.device)
        vs = self.v_scale.view(1, -1, 1, 1).to(device=v.device)
        kq = ((k.float() / ks).clamp(-448.0, 448.0).to(_FP8_E4M3).float() * ks).to(k.dtype)
        vq = ((v.float() / vs).clamp(-448.0, 448.0).to(_FP8_E4M3).float() * vs).to(v.dtype)
        return kq, vq


def _attach_ct_offload_wrappers(
    root: nn.Module,
) -> tuple[dict[str, types.MethodType], set[str], dict[str, int]]:
    """Attach compressed_tensors-style MethodType wrappers on ~half of submodules, leaving rest class-bound."""
    pre_inst_fwds: dict[str, types.MethodType] = {}
    originally_class_bound: set[str] = set()
    call_counts: dict[str, int] = {}

    for idx, (name, mod) in enumerate(root.named_modules()):
        if not name:
            originally_class_bound.add(name)
            continue
        # Wrap every full/linear attention, kv_cache, experts, router, and alternating leaf modules
        should_wrap = (
            name.endswith(("self_attn", "linear_attn", "kv_cache", "experts", "gate", "k_proj", "q_proj"))
            or idx % 2 == 0
        )
        if not should_wrap:
            if "forward" not in mod.__dict__:
                originally_class_bound.add(name)
            else:
                pre_inst_fwds[name] = mod.__dict__["forward"]
            continue

        orig_func = mod.forward.__func__
        call_counts[name] = 0

        @functools.wraps(orig_func)
        def _offload_fwd(
            _self: nn.Module,
            *args: Any,
            _fn: Any = orig_func,
            _nm: str = name,
            **kwargs: Any,
        ) -> Any:
            call_counts[_nm] += 1
            return _fn(_self, *args, **kwargs)

        bound = types.MethodType(_offload_fwd, mod)
        mod.forward = bound  # type: ignore[method-assign]
        pre_inst_fwds[name] = bound

    return pre_inst_fwds, originally_class_bound, call_counts


def _pack_tensor_to_nvfp4(
    w_float: torch.Tensor,
    global_scale: torch.Tensor,
    *,
    group_size: int = 16,
    mul_gs: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize a 2D `[M, K]` or 3D `[E, M, K]` tensor into NVFP4 `(packed_u8, scale_fp8, exact_deq)`."""
    assert w_float.shape[-1] % group_size == 0 and group_size % 2 == 0
    wf = w_float.detach().float()
    gf = global_scale.detach().float().to(wf.device)
    if gf.numel() == 1:
        gf_b = gf.reshape(())
    elif gf.ndim == 1 and gf.shape[0] == wf.shape[0]:
        gf_b = gf.reshape(wf.shape[0], *([1] * (wf.ndim - 1)))
    elif wf.ndim == 3 and gf.ndim == 2 and gf.shape == (wf.shape[0], wf.shape[1]):
        gf_b = gf.unsqueeze(-1)
    else:
        raise ValueError(f"unsupported global_scale shape {tuple(gf.shape)} for {tuple(wf.shape)}")

    # Since dequant computes `(e2m1 * block_scale) / gs` (when mul_gs=False), pre-scale by `* gs`
    w_eff = (wf / gf_b.clamp_min(1e-12)) if mul_gs else (wf * gf_b)
    grouped = w_eff.reshape(*wf.shape[:-1], wf.shape[-1] // group_size, group_size)
    amax = grouped.abs().amax(dim=-1).clamp_min(1e-6)
    raw_scale = (amax / 6.0).clamp(1e-4, 448.0)
    scale_fp8 = raw_scale.to(_FP8_E4M3)
    scale_f32 = scale_fp8.float().clamp_min(1e-6)

    normed = grouped / scale_f32.unsqueeze(-1)
    signs = (normed < 0).to(torch.uint8)
    mags = normed.abs()
    pos_grid = torch.tensor(_E2M1_POS, device=wf.device, dtype=torch.float32)
    idx = (mags.unsqueeze(-1) - pos_grid).abs().argmin(dim=-1).to(torch.uint8)
    nibbles = ((signs << 3) | idx).reshape(*wf.shape[:-1], wf.shape[-1])
    lo = nibbles[..., 0::2]
    hi = nibbles[..., 1::2]
    packed_u8 = ((hi << 4) | (lo & 0x0F)).contiguous()

    # Compute exact dequantized reference from the packed representation
    e2m1_lut = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
        device=wf.device,
        dtype=torch.float32,
    )
    unp = torch.stack((packed_u8 & 0x0F, (packed_u8 >> 4) & 0x0F), dim=-1).reshape(wf.shape)
    deq = (e2m1_lut[unp.long()].reshape(grouped.shape) * scale_f32.unsqueeze(-1)).reshape(wf.shape)
    deq = (deq * gf_b) if mul_gs else (deq / gf_b)
    return packed_u8, scale_fp8, deq.to(w_float.dtype)


def _convert_toy_experts_to_nvfp4(
    exp_mod: nn.Module,
    *,
    gu_gs_shape: str = "2d",
    use_buffers: bool = False,
    seed: int = 101,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Repack a `ToyExperts` module in-place into 3D `moe_nvfp4`, returning exact dequant `(gu, dn)`."""
    gen = torch.Generator().manual_seed(seed)
    gu_orig = exp_mod.gate_up_proj.detach().clone()
    dn_orig = exp_mod.down_proj.detach().clone()
    n_exp, two_i, _hidden = gu_orig.shape
    inter = two_i // 2

    if gu_gs_shape == "2d":
        # Deliberately distinct scales for gate rows vs up rows (12x ratio) to catch any slice/broadcast bug!
        gate_gs = torch.empty((n_exp, inter), dtype=torch.float32).uniform_(0.25, 0.45, generator=gen)
        up_gs = torch.empty((n_exp, inter), dtype=torch.float32).uniform_(3.20, 5.10, generator=gen)
        gu_gs = torch.cat([gate_gs, up_gs], dim=1).contiguous()
    elif gu_gs_shape == "1d":
        gu_gs = torch.empty((n_exp,), dtype=torch.float32).uniform_(0.60, 2.40, generator=gen)
    elif gu_gs_shape == "scalar":
        gu_gs = torch.tensor(1.75, dtype=torch.float32)
    else:
        raise ValueError(gu_gs_shape)

    dn_gs = (
        torch.tensor(1.35, dtype=torch.float32)
        if gu_gs_shape == "scalar"
        else torch.empty((n_exp,), dtype=torch.float32).uniform_(0.50, 2.20, generator=gen)
    )

    gu_pk, gu_sc, gu_deq = _pack_tensor_to_nvfp4(gu_orig, gu_gs, group_size=16)
    dn_pk, dn_sc, dn_deq = _pack_tensor_to_nvfp4(dn_orig, dn_gs, group_size=16)

    exp_mod._parameters.pop("gate_up_proj", None)
    exp_mod._parameters.pop("down_proj", None)
    tensors = {
        "gate_up_proj_packed": gu_pk,
        "gate_up_proj_scale": gu_sc,
        "gate_up_proj_global_scale": gu_gs,
        "down_proj_packed": dn_pk,
        "down_proj_scale": dn_sc,
        "down_proj_global_scale": dn_gs,
    }
    for attr_nm, val in tensors.items():
        if use_buffers:
            exp_mod.register_buffer(attr_nm, val)
        else:
            exp_mod.register_parameter(attr_nm, nn.Parameter(val, requires_grad=False))
    exp_mod.group_size = 16  # type: ignore[attr-defined]
    exp_mod._qloop_nvfp4_scale_mode = "divisor"  # type: ignore[attr-defined]
    return gu_deq, dn_deq


def _load_standalone_modeling_without_qloop(pack_dir: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    """Import emitted `modeling_qwen36_loophssd.py` with `qloop` hidden (`_HAS_QLOOP is False`)."""
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

    class _PretrainedConfig:
        model_type = "qwen36_loophssd"

        def __init__(self, **kwargs: Any) -> None:
            for k, v in kwargs.items():
                setattr(self, k, v)

        def to_dict(self) -> dict[str, Any]:
            return {k: copy.deepcopy(v) for k, v in self.__dict__.items() if not k.startswith("_")}

        @classmethod
        def from_pretrained(cls, path: str, **kw: Any) -> Any:
            del kw
            import json as _json

            with open(Path(path) / "config.json", encoding="utf-8") as fh:
                return cls(**_json.load(fh))

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

    stub_tf.PretrainedConfig = _PretrainedConfig  # type: ignore[attr-defined]
    stub_tf.PreTrainedConfig = _PretrainedConfig  # type: ignore[attr-defined]
    stub_tf.PreTrainedModel = _PreTrainedModel  # type: ignore[attr-defined]
    stub_tf.GenerationMixin = _GenerationMixin  # type: ignore[attr-defined]
    stub_tf.AutoConfig = _AutoConfig  # type: ignore[attr-defined]
    stub_tf.AutoModelForCausalLM = _AutoModelForCausalLM  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "transformers", stub_tf)

    pkg_name = f"_st_mt_kv_{abs(hash(str(pack_dir)))}"
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


# =============================================== 1. Universal MethodType & Exact Unwrap Restoration


@pytest.mark.parametrize("linear_stack", ["gdn2", "hssd"])
def test_methodtype_forward_wrappers_and_exact_offload_unwrap_restoration(linear_stack: str) -> None:
    """Verify MethodType.__func__ across apply_loop, install_lsa, install_gdn2, apply_hssd & exact unwrap."""
    torch.manual_seed(42)
    cfg = _make_hybrid_toy_cfg()
    model = ToyForCausalLM(cfg).eval()

    # Attach QuantizedKVCache submodules to all full-attention layers (L1, L4, L5)
    for li in (1, 4, 5):
        model.model.layers[li].self_attn.kv_cache = _MockSubmoduleQuantizedKVCache(cfg.num_key_value_heads)

    # Also bind a MethodType `chunk_gated_delta_rule` on linear_attn to test bound helper wrapping/restoration
    lin0 = model.model.layers[0].linear_attn

    def _custom_chunk_rule(_self: nn.Module, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, *a: Any, **kw: Any):
        from qloop.model.toy import _toy_gated_delta_rule

        return _toy_gated_delta_rule(q, k, v, *a, **kw)

    orig_lin0_rule = types.MethodType(_custom_chunk_rule, lin0)
    lin0.chunk_gated_delta_rule = orig_lin0_rule  # type: ignore[attr-defined]

    # Simulate HF `kernels` `_kernel_repo` + instance `forward` on L3 linear_attn
    lin3 = model.model.layers[3].linear_attn
    lin3_orig_cls_fwd = type(lin3).forward

    def _mock_kernel_fwd(_self: nn.Module, *a: Any, **kw: Any) -> Any:
        return lin3_orig_cls_fwd(_self, *a, **kw)

    saved_kernel_fwd = types.MethodType(_mock_kernel_fwd, lin3)

    pre_inst_fwds, orig_cls_bound, call_counts = _attach_ct_offload_wrappers(model)
    lin3.__dict__["forward"] = saved_kernel_fwd
    lin3.__dict__["_kernel_repo"] = "kernels-community/gated-deltanet"
    pre_inst_fwds["model.layers.3.linear_attn"] = saved_kernel_fwd
    orig_cls_bound.discard("model.layers.3.linear_attn")

    _assert_all_forwards_are_methodtype(model)

    inp = torch.tensor([[5, 19, 44, 81, 102, 13, 67, 92, 150, 211, 38, 76]], dtype=torch.long)
    with torch.no_grad():
        pre_wrap_logits = model(inp, use_cache=False).logits.detach().clone()
    assert sum(call_counts.values()) > 0

    lsa = install_lsa(
        model,
        LSAConfig(
            mode="sparse",
            index_dim=8,
            block_size=4,
            pool=2,
            avg_bucket_keys=2,
            max_bucket_keys=4,
            num_select=1,
            window=4,
            sinks=1,
            query_chunk=4,
            num_query_passes=3,
        ),
    )
    _assert_all_forwards_are_methodtype(model)

    if linear_stack == "gdn2":
        lin_mgr: Any = install_gdn2(model, GDN2Config(rank=8, chunk_size=4))
    else:
        lin_mgr = apply_hssd(model, HSSDConfig(num_pages=4, k_pages=2, rank=8, chunk_size=4, window_chunks=1))
    _assert_all_forwards_are_methodtype(model)
    assert isinstance(lin0.chunk_gated_delta_rule, types.MethodType)
    assert hasattr(lin0.chunk_gated_delta_rule, "__func__")

    loop = apply_loop(
        model,
        LoopSpec(
            prelude_end=2,
            core_end=5,
            num_passes=3,
            target_virtual_layers=12,
            kv_share=True,
            bridge_mode="subspace",
            bridge_rank=8,
            attn_adapter_rank=4,
            per_pass_router=True,
            router_adapter_rank=4,
            mor=True,
        ),
    )
    # Every single module in the fully-wrapped 12-virtual-layer stack MUST have MethodType forward + __func__!
    n_checked = _assert_all_forwards_are_methodtype(model)
    assert n_checked >= 40

    # Run forward + backward through the wrapped stack
    model.train()
    out_wrapped = model(inp, use_cache=False).logits
    loss = out_wrapped.float().square().mean()
    loss.backward()
    model.zero_grad(set_to_none=True)
    model.eval()

    # Unwrap in reverse installation order and verify exact descriptor identity restoration
    loop.unwrap()
    lin_mgr.uninstall()
    lsa.uninstall()

    _assert_all_forwards_are_methodtype(model)
    assert lin0.__dict__.get("chunk_gated_delta_rule") is orig_lin0_rule
    assert lin3.__dict__.get("_kernel_repo") == "kernels-community/gated-deltanet"

    for name, mod in model.named_modules():
        if name in pre_inst_fwds:
            assert "forward" in mod.__dict__, f"expected instance forward restored on {name!r}"
            assert mod.__dict__["forward"] is pre_inst_fwds[name], f"instance forward identity lost on {name!r}"
        elif name in orig_cls_bound:
            assert "forward" not in mod.__dict__, (
                f"module {name!r} had no instance forward before wrap, but 'forward' lingered in __dict__"
            )

    with torch.no_grad():
        post_unwrap_logits = model(inp, use_cache=False).logits
    assert torch.equal(post_unwrap_logits, pre_wrap_logits)


def test_standalone_quant_fallback_methodtype_and_uninstall_restoration(tmp_path: Path, monkeypatch) -> None:
    """Verify standalone `_HAS_QLOOP=False` `_st_install_quant_fallback` uses MethodType and restores cleanly."""
    torch.manual_seed(7)
    tok = ToyTokenizer()
    cfg = _make_hybrid_toy_cfg(tok.vocab_size)
    base = ToyForCausalLM(cfg).eval()
    stack = Stack(model=base, tokenizer=tok, parts=inspect_model(base), load_info={"name": "toy"})
    pack_dir = tmp_path / "st_methodtype_pack"
    export_model_pack(stack, pack_dir)

    st_mod = _load_standalone_modeling_without_qloop(pack_dir, monkeypatch)

    # Build a synthetic module tree with linear_nvfp4 (pre-wrapped with offload MethodType) and moe_nvfp4 (class-bound)
    root = ToyForCausalLM(cfg).eval()
    q_lin = root.model.layers[1].self_attn.q_proj
    w_orig = q_lin.weight.detach().clone()
    gs_lin = torch.tensor(1.5, dtype=torch.float32)
    wp, ws, _ = _pack_tensor_to_nvfp4(w_orig, gs_lin, group_size=16)
    q_lin._parameters.pop("weight", None)
    q_lin.register_buffer("weight_packed", wp)
    q_lin.register_buffer("weight_scale", ws)
    q_lin.register_buffer("weight_global_scale", gs_lin)

    orig_q_func = q_lin.forward.__func__
    pre_q_fwd = types.MethodType(lambda _s, *a, _f=orig_q_func, **k: _f(_s, *a, **k), q_lin)
    q_lin.forward = pre_q_fwd  # type: ignore[method-assign]

    exp2 = root.model.layers[2].mlp.experts
    assert "forward" not in exp2.__dict__
    _convert_toy_experts_to_nvfp4(exp2, gu_gs_shape="2d", use_buffers=False, seed=19)

    st_mod._st_install_quant_fallback(root)
    _assert_all_forwards_are_methodtype(root)
    assert q_lin.__dict__["forward"] is not pre_q_fwd
    assert "forward" in exp2.__dict__

    with torch.no_grad():
        logits = root(torch.tensor([[4, 9, 15, 22]]), use_cache=False).logits
    assert torch.isfinite(logits).all()

    st_mod._st_uninstall_quant_fallback(root)
    assert q_lin.__dict__.get("forward") is pre_q_fwd
    assert "forward" not in exp2.__dict__
    _assert_all_forwards_are_methodtype(root)


# ========================================== 2. sha256_frozen_experts FP8/uint8 + OffloadCache Safety


class _CloningOffloadDict(dict):  # type: ignore[type-arg]
    """Mimics compressed_tensors OffloadCache returning newly materialized Tensor instances on iteration."""

    def items(self):  # type: ignore[override]
        return [(k, v.clone() if isinstance(v, torch.Tensor) else v) for k, v in super().items()]

    def values(self):  # type: ignore[override]
        return [v.clone() if isinstance(v, torch.Tensor) else v for v in super().values()]


def test_sha256_frozen_experts_fp8_uint8_params_buffers_and_offload_loop_invariance() -> None:
    """Verify sha256_frozen_experts hashes frozen FP8/uint8 expert params AND buffers deterministically."""
    torch.manual_seed(99)
    cfg = _make_hybrid_toy_cfg()
    model = ToyForCausalLM(cfg).eval()
    for p in model.parameters():
        p.requires_grad_(False)

    # Layer 2 experts: NVFP4 stored as frozen nn.Parameters (uint8 + float8_e4m3fn + float32)
    _convert_toy_experts_to_nvfp4(model.model.layers[2].mlp.experts, gu_gs_shape="2d", use_buffers=False, seed=21)
    # Layer 3 experts: NVFP4 stored in _buffers (uint8 + float8_e4m3fn + float32)
    _convert_toy_experts_to_nvfp4(model.model.layers[3].mlp.experts, gu_gs_shape="2d", use_buffers=True, seed=22)
    # Layer 4 experts: OffloadCache-style dict subclass on _buffers yielding fresh Tensor clones per iteration
    _convert_toy_experts_to_nvfp4(model.model.layers[4].mlp.experts, gu_gs_shape="2d", use_buffers=True, seed=23)
    exp4 = model.model.layers[4].mlp.experts
    exp4._buffers = _CloningOffloadDict(exp4._buffers)  # type: ignore[assignment]

    digest_before = sha256_frozen_experts(model)
    assert len(digest_before) == 64

    loop = apply_loop(
        model,
        LoopSpec(prelude_end=2, core_end=5, num_passes=3, target_virtual_layers=12, kv_share=True),
    )
    digest_during = sha256_frozen_experts(model)
    assert digest_during == digest_before, "sha256_frozen_experts must be invariant under VirtualLayer wrapping"

    # Mutate 1 byte of a uint8 packed expert Parameter -> digest must change
    p_u8 = model.model.layers[0].inner.mlp.experts if hasattr(model.model.layers[0], "inner") else None
    assert p_u8 is not None
    gu_param = loop._orig["layers"][2].mlp.experts.gate_up_proj_packed
    orig_byte = int(gu_param.view(-1)[0].item())
    with torch.no_grad():
        gu_param.view(-1)[0] = (orig_byte ^ 0x01) & 0xFF
    assert sha256_frozen_experts(model) != digest_before
    with torch.no_grad():
        gu_param.view(-1)[0] = orig_byte
    assert sha256_frozen_experts(model) == digest_before

    # Mutate 1 element of a float8_e4m3fn expert scale BUFFER -> digest must change
    sc_buf = loop._orig["layers"][3].mlp.experts._buffers["gate_up_proj_scale"]
    assert sc_buf is not None
    orig_sc = sc_buf.clone()
    with torch.no_grad():
        sc_buf.copy_((sc_buf.float() * 2.0).to(_FP8_E4M3))
    assert sha256_frozen_experts(model) != digest_before
    with torch.no_grad():
        sc_buf.copy_(orig_sc)
    assert sha256_frozen_experts(model) == digest_before

    loop.unwrap()
    assert sha256_frozen_experts(model) == digest_before


# ======================================================= 3. QuantizedKVCache + SharedKVView + LSA Safety


class _StrictQuantizedCache:
    """Mimics a compressed-tensors QuantizedKVCache wrapping inner FP8 storage + crashing on 0-len amax."""

    def __init__(self, num_layers: int, num_kv_heads: int) -> None:
        self.k_scales = [torch.full((num_kv_heads,), 0.125, dtype=torch.float32) for _ in range(num_layers)]
        self.v_scales = [torch.full((num_kv_heads,), 0.25, dtype=torch.float32) for _ in range(num_layers)]
        self.cache = types.SimpleNamespace(
            key_cache=[None] * num_layers,
            value_cache=[None] * num_layers,
            conv_states=[None] * num_layers,
            recurrent_states=[None] * num_layers,
        )
        self.update_calls: dict[int, int] = {}

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del cache_kwargs
        # Dynamic quantizers computing `amax` crash if fed an empty 0-token slice `[..., :0, :]`!
        if key_states.shape[-2] == 0:
            raise AssertionError("QuantizedKVCache.update must NOT be invoked with 0-length slice on pass>=2!")
        self.update_calls[layer_idx] = self.update_calls.get(layer_idx, 0) + 1
        ks = self.k_scales[layer_idx].view(1, -1, 1, 1).to(key_states.device)
        vs = self.v_scales[layer_idx].view(1, -1, 1, 1).to(value_states.device)
        kq = (key_states.float() / ks).clamp(-448.0, 448.0).to(_FP8_E4M3)
        vq = (value_states.float() / vs).clamp(-448.0, 448.0).to(_FP8_E4M3)
        if self.cache.key_cache[layer_idx] is None:
            self.cache.key_cache[layer_idx] = kq
            self.cache.value_cache[layer_idx] = vq
        else:
            self.cache.key_cache[layer_idx] = torch.cat([self.cache.key_cache[layer_idx], kq], dim=-2)
            self.cache.value_cache[layer_idx] = torch.cat([self.cache.value_cache[layer_idx], vq], dim=-2)
        k_deq = (self.cache.key_cache[layer_idx].float() * ks).to(key_states.dtype)
        v_deq = (self.cache.value_cache[layer_idx].float() * vs).to(value_states.dtype)
        return k_deq, v_deq


def test_quantized_kv_cache_shared_kv_view_and_lsa_no_double_quant() -> None:
    """Verify SharedKVView & install_lsa do not double-quantize, corrupt, or crash on QuantizedKVCache."""
    torch.manual_seed(123)
    qcache = _StrictQuantizedCache(num_layers=12, num_kv_heads=2)
    k_in = torch.randn(1, 2, 6, 8)
    v_in = torch.randn(1, 2, 6, 8)

    # 1) Pass-1 recording + Pass-2/3 replay via SharedKVView stash: exact bit-identity & 1 update call
    stash: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    v_pass1 = SharedKVView(qcache, stash=stash, record=True)
    k1, v1 = v_pass1.update(k_in, v_in, 4)
    assert qcache.update_calls[4] == 1

    v_pass2 = SharedKVView(qcache, stash=stash, record=False)
    k2, v2 = v_pass2.update(torch.randn_like(k_in), torch.randn_like(v_in), 4)
    assert qcache.update_calls[4] == 1, "Pass >= 2 must not call QuantizedKVCache.update again"
    assert torch.equal(k2, k1) and torch.equal(v2, v1)

    # 2) Direct un-stashed SharedKVView(qcache) fallback: dequantizes inner FP8 cache without calling update(:0)
    v_direct = SharedKVView(qcache)
    kd, vd = v_direct.update(torch.randn_like(k_in), torch.randn_like(v_in), 4)
    assert qcache.update_calls[4] == 1
    assert torch.equal(kd, k1) and torch.equal(vd, v1)

    # 3) Core self_attn.kv_cache submodule called inside attention: Pass 1 runs once, Pass 2/3 reuse Pass-1 KV
    cfg = _make_hybrid_toy_cfg()
    model = ToyForCausalLM(cfg).eval()
    kvc_core = _MockSubmoduleQuantizedKVCache(cfg.num_key_value_heads)
    attn_core = model.model.layers[4].self_attn
    attn_core.kv_cache = kvc_core
    orig_k_proj_fwd = attn_core.k_proj.forward
    orig_v_proj_fwd = attn_core.v_proj.forward

    # Wire self_attn to invoke `self.kv_cache(k, v)` so we verify kv_cache submodule runs ONCE across 3 passes!
    orig_attn_func = attn_core.forward.__func__

    @functools.wraps(orig_attn_func)
    def _attn_calling_kvc(_self: nn.Module, hidden_states: torch.Tensor, *a: Any, **kw: Any) -> Any:
        # Exercise self.kv_cache on projected K/V shape [B, G, T, D]
        bsz, seqlen, _ = hidden_states.shape
        dummy_k = torch.ones(bsz, cfg.num_key_value_heads, seqlen, cfg.head_dim, dtype=hidden_states.dtype)
        dummy_v = torch.ones_like(dummy_k) * 2.0
        kq, vq = _self.kv_cache(dummy_k, dummy_v)
        assert kq.dtype == hidden_states.dtype and vq.dtype == hidden_states.dtype
        return orig_attn_func(_self, hidden_states, *a, **kw)

    attn_core.forward = types.MethodType(_attn_calling_kvc, attn_core)  # type: ignore[method-assign]
    del orig_k_proj_fwd, orig_v_proj_fwd

    lsa = install_lsa(
        model,
        LSAConfig(
            mode="sparse",
            index_dim=8,
            block_size=4,
            pool=2,
            avg_bucket_keys=2,
            max_bucket_keys=4,
            num_select=1,
            window=4,
            sinks=1,
            query_chunk=4,
        ),
    )
    loop = apply_loop(model, LoopSpec(prelude_end=2, core_end=5, num_passes=3, target_virtual_layers=12, kv_share=True))
    _assert_all_forwards_are_methodtype(model)

    with torch.no_grad():
        out = model(torch.tensor([[3, 7, 11, 15, 19, 23, 27, 31, 35, 39]]), use_cache=False).logits
    assert torch.isfinite(out).all()
    assert kvc_core.call_count == 1, f"core kv_cache submodule must run once on pass 1, ran {kvc_core.call_count}x"

    # 4) Verify lsa_attention handles raw FP8 key/value tensors cleanly using self_attn.kv_cache scales
    q_f32 = torch.randn(1, 4, 8, 8)
    k_fp8 = torch.randn(1, 2, 8, 8).clamp(-10.0, 10.0).to(_FP8_E4M3)
    v_fp8 = torch.randn(1, 2, 8, 8).clamp(-10.0, 10.0).to(_FP8_E4M3)
    with torch.no_grad():
        attn_out, _ = lsa_attention(attn_core, q_f32, k_fp8, v_fp8)
    assert attn_out.dtype == q_f32.dtype and torch.isfinite(attn_out).all()

    loop.unwrap()
    lsa.uninstall()


# ============================ 4. Standalone Export & Reload Parity for 3D moe_nvfp4 ([E, 2I], [E], [])


@pytest.mark.parametrize("use_buffers", [False, True])
def test_export_and_standalone_reload_parity_3d_moe_nvfp4_2d_global_scale(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    use_buffers: bool,
) -> None:
    """Verify export_model_pack + load_exported_pack + standalone _HAS_QLOOP=False on 3D moe_nvfp4 [E, 2I]."""
    torch.manual_seed(314)
    gen = torch.Generator().manual_seed(3140)
    tok = ToyTokenizer()
    cfg = _make_hybrid_toy_cfg(tok.vocab_size)

    # Build both a dense reference model (holding exact dequantized weights) and the repacked 3D moe_nvfp4 model
    model_nvfp4 = ToyForCausalLM(cfg).eval()
    model_dense_ref = copy.deepcopy(model_nvfp4).eval()

    for li in range(cfg.num_hidden_layers):
        gu_deq, dn_deq = _convert_toy_experts_to_nvfp4(
            model_nvfp4.model.layers[li].mlp.experts,
            gu_gs_shape="2d",
            use_buffers=use_buffers,
            seed=500 + li,
        )
        with torch.no_grad():
            model_dense_ref.model.layers[li].mlp.experts.gate_up_proj.copy_(gu_deq)
            model_dense_ref.model.layers[li].mlp.experts.down_proj.copy_(dn_deq)

    # Confirm [E, 2I] and [E] shapes on repacked experts
    exp0 = model_nvfp4.model.layers[0].mlp.experts
    assert exp0.gate_up_proj_packed.shape == (6, 64, 16) and exp0.gate_up_proj_packed.dtype == torch.uint8
    assert exp0.gate_up_proj_scale.shape == (6, 64, 2) and exp0.gate_up_proj_scale.dtype == _FP8_E4M3
    assert exp0.gate_up_proj_global_scale.shape == (6, 64) and exp0.gate_up_proj_global_scale.dtype == torch.float32
    assert exp0.down_proj_packed.shape == (6, 32, 16) and exp0.down_proj_packed.dtype == torch.uint8
    assert exp0.down_proj_scale.shape == (6, 32, 2) and exp0.down_proj_scale.dtype == _FP8_E4M3
    assert exp0.down_proj_global_scale.shape == (6,) and exp0.down_proj_global_scale.dtype == torch.float32

    install_quantized_autograd_bridge(model_nvfp4)
    _assert_all_forwards_are_methodtype(model_nvfp4, allow_legacy_quant_bridge=True)

    # Attach non-zero GDN-2 + sparse LSA + 3-pass Loop onto BOTH models identically
    for m in (model_nvfp4, model_dense_ref):
        g_mgr = install_gdn2(m, GDN2Config(rank=8, chunk_size=4))
        l_mgr = install_lsa(
            m,
            LSAConfig(
                mode="sparse",
                index_dim=8,
                block_size=4,
                pool=2,
                avg_bucket_keys=2,
                max_bucket_keys=4,
                num_select=1,
                window=4,
                sinks=1,
                query_chunk=4,
                num_query_passes=3,
            ),
        )
        lp_ctl = apply_loop(
            m,
            LoopSpec(
                prelude_end=2,
                core_end=5,
                num_passes=3,
                target_virtual_layers=12,
                kv_share=True,
                gate_init=0.40,
                bridge_mode="subspace",
                bridge_rank=8,
                per_pass_router=True,
                router_adapter_rank=4,
                router_repel_init=1.2,
            ),
        )
        m._test_handles = (g_mgr, l_mgr, lp_ctl)  # type: ignore[attr-defined]

    # Copy identical non-zero adapter weights onto both models
    g_q, l_q, lp_q = model_nvfp4._test_handles  # type: ignore[attr-defined]
    g_d, l_d, lp_d = model_dense_ref._test_handles  # type: ignore[attr-defined]
    with torch.no_grad():
        for li, ad in g_q.adapters.items():
            ad.erase_up.weight.normal_(0.0, 0.08, generator=gen)
            ad.write_up.weight.normal_(0.0, 0.08, generator=gen)
            ad.decay_up.weight.normal_(0.0, 0.05, generator=gen)
            g_d.adapters[li].load_state_dict(ad.state_dict())
        for li, ix in l_q.indexers.items():
            ix.wk.normal_(0.0, 0.12, generator=gen)
            ix.wq.normal_(0.0, 0.12, generator=gen)
            l_d.indexers[li].load_state_dict(ix.state_dict())
        lp_q.adapters.gates[0].fill_(0.35)
        lp_q.adapters.gates[1].fill_(0.45)
        for u in lp_q.adapters.bridge.ups:
            u.weight.normal_(0.0, 0.08, generator=gen)
        assert lp_q.adapters.pass_router is not None
        lp_q.adapters.pass_router.up.normal_(0.0, 0.12, generator=gen)
        lp_d.adapters.load_state_dict(lp_q.adapters.state_dict())

    del model_nvfp4._test_handles, model_dense_ref._test_handles  # type: ignore[attr-defined]

    prompt = torch.tensor([[6, 18, 33, 49, 72, 95, 118, 144, 171, 203, 229, 41, 87, 129]], dtype=torch.long)
    ref_logits: dict[int, torch.Tensor] = {}
    with torch.no_grad():
        for r in (1, 2, 3):
            lp_q.set_active_passes(r)
            lp_d.set_active_passes(r)
            q_out = model_nvfp4(prompt, use_cache=False).logits.detach().clone()
            d_out = model_dense_ref(prompt, use_cache=False).logits.detach().clone()
            assert (q_out - d_out).abs().max().item() < 1e-5
            ref_logits[r] = q_out

    stack_q = Stack(
        model=model_nvfp4,
        tokenizer=tok,
        parts=inspect_model(model_nvfp4),
        loop=lp_q,
        gdn2=g_q,
        lsa=l_q,
        load_info={"name": "toy"},
    )
    pack_dir = tmp_path / f"moe_nvfp4_pack_{'buf' if use_buffers else 'param'}"
    export_model_pack(stack_q, pack_dir, max_shard_size_bytes=65536)

    # 1) Reload via `load_exported_pack` (qloop-installed strict path)
    loaded = load_exported_pack(pack_dir, strict=True)
    _assert_all_forwards_are_methodtype(loaded.model, allow_legacy_quant_bridge=True)
    reloaded_exp0 = loaded.loop._orig["layers"][0].mlp.experts if loaded.loop is not None else None
    assert reloaded_exp0 is not None
    if use_buffers:
        assert "gate_up_proj_packed" in reloaded_exp0._buffers
    else:
        assert "gate_up_proj_packed" in reloaded_exp0._parameters

    with torch.no_grad():
        for r in (1, 2, 3):
            with loaded.depth(r):
                l_out = loaded(prompt, use_cache=False).logits
                maxdiff = (l_out - ref_logits[r]).abs().max().item()
                assert maxdiff < 1e-4, f"load_exported_pack maxdiff={maxdiff} at depth={r}"

    # 2) Reload via standalone `modeling_qwen36_loophssd.py` (`_HAS_QLOOP = False`, `strict=True`)
    st_mod = _load_standalone_modeling_without_qloop(pack_dir, monkeypatch)
    sm = st_mod.Qwen36LoopHSSDForCausalLM.from_pretrained(str(pack_dir), strict=True)
    _assert_all_forwards_are_methodtype(sm)

    with torch.no_grad():
        for r in (1, 2, 3):
            with sm.depth(r):
                s_out = sm(prompt, use_cache=False).logits
                maxdiff = (s_out - ref_logits[r]).abs().max().item()
                assert maxdiff < 1e-4, f"standalone _HAS_QLOOP=False maxdiff={maxdiff} at depth={r}"

    # 3) Verify standalone single-expert vs batched chunked-3D global-scale parity for [E, 2I], [E], and scalar []
    for gs_kind in ("2d", "1d", "scalar"):
        toy_e = ToyForCausalLM(cfg).model.layers[0].mlp.experts.eval()
        gu_ref, dn_ref = _convert_toy_experts_to_nvfp4(toy_e, gu_gs_shape=gs_kind, use_buffers=False, seed=77)
        for eid in range(cfg.num_experts):
            gu_1, dn_1 = st_mod._st_deq_moe_expert(toy_e, eid, torch.float32)
            assert (gu_1 - gu_ref[eid]).abs().max().item() < 1e-6
            assert (dn_1 - dn_ref[eid]).abs().max().item() < 1e-6

        x_tok = torch.randn(5, cfg.hidden_size)
        top_i = torch.tensor([[0, 3], [2, 5], [1, 4], [0, 2], [3, 5]], dtype=torch.long)
        top_w = F.softmax(torch.randn(5, 2), dim=-1)
        y_st = st_mod._st_moe_fallback(toy_e, (x_tok, top_i, top_w), {})

        # Exact manual per-expert reference
        y_ref = torch.zeros_like(x_tok)
        for ti in range(5):
            for si in range(2):
                e = int(top_i[ti, si].item())
                g_part, u_part = F.linear(x_tok[ti : ti + 1], gu_ref[e]).chunk(2, dim=-1)
                y_ref[ti : ti + 1] += top_w[ti, si] * F.linear(F.silu(g_part) * u_part, dn_ref[e])
        assert (y_st - y_ref).abs().max().item() < 1e-5

    # 4) Verify standalone `_st_install_quant_fallback` / `_st_uninstall_quant_fallback` exact descriptor restoration
    probe_root = ToyForCausalLM(cfg).eval()
    _convert_toy_experts_to_nvfp4(probe_root.model.layers[0].mlp.experts, gu_gs_shape="2d", use_buffers=use_buffers)
    _convert_toy_experts_to_nvfp4(probe_root.model.layers[1].mlp.experts, gu_gs_shape="2d", use_buffers=use_buffers)
    exp_a = probe_root.model.layers[0].mlp.experts  # pre-existing instance MethodType wrapper
    exp_b = probe_root.model.layers[1].mlp.experts  # class-bound forward only

    def _offload_exp_fwd(_self: nn.Module, *a: Any, **kw: Any) -> Any:
        return type(_self).forward(_self, *a, **kw)

    pre_exp_a_fwd = types.MethodType(_offload_exp_fwd, exp_a)
    exp_a.forward = pre_exp_a_fwd  # type: ignore[method-assign]
    assert "forward" in exp_a.__dict__ and "forward" not in exp_b.__dict__

    st_mod._st_install_quant_fallback(probe_root)
    _assert_all_forwards_are_methodtype(probe_root)
    st_mod._st_uninstall_quant_fallback(probe_root)
    assert exp_a.__dict__.get("forward") is pre_exp_a_fwd
    assert "forward" not in exp_b.__dict__
    _assert_all_forwards_are_methodtype(probe_root)
