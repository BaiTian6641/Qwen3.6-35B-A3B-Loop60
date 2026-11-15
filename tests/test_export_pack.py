"""End-to-end tests for HSSD v3.2 Whole Model Pack serialization, reload & standalone HF parity."""
from __future__ import annotations

import ast
import importlib.util
import json
import shutil
import sys
import types
from pathlib import Path
from typing import Any

import pytest
import torch

from qloop.data.tokenizer import ToyTokenizer
from qloop.export_pack import (
    SAFETENSORS_INDEX_FILE,
    export_whole_model_pack,
    load_whole_model_pack,
)
from qloop.infer import CONFIG_FILENAME, MODELING_FILENAME
from qloop.infer.hf_custom_modeling import load_safetensors_file, save_safetensors_file
from qloop.model.experts import attach_hot_expert_deltas
from qloop.model.hssd import HSSDConfig, apply_hssd
from qloop.model.introspect import inspect_model
from qloop.model.loop import LoopSpec, apply_loop
from qloop.model.lsa import LSAConfig, install_lsa
from qloop.model.toy import ToyConfig, ToyForCausalLM
from qloop.train.stack import Stack


def _load_standalone_modeling_without_qloop(pack_dir: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    """Load emitted `modeling_qwen36_loophssd.py` with `qloop` hidden so `_HAS_QLOOP is False`."""
    cfg_py = pack_dir / CONFIG_FILENAME
    mod_py = pack_dir / MODELING_FILENAME
    ast.parse(cfg_py.read_text(encoding="utf-8"), feature_version=(3, 9))
    ast.parse(mod_py.read_text(encoding="utf-8"), feature_version=(3, 9))

    real_find_spec = importlib.util.find_spec

    def _no_qloop_find_spec(name: str, package: Any = None) -> Any:
        if name == "qloop" or name.startswith("qloop."):
            return None
        return real_find_spec(name, package)

    monkeypatch.setattr(importlib.util, "find_spec", _no_qloop_find_spec)

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

    pkg_name = f"_standalone_v32_pack_{abs(hash(str(pack_dir)))}"
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


def _populate_hssd_v32_weights(ad: Any, gen: torch.Generator) -> None:
    with torch.no_grad():
        ad.down.weight.normal_(0.0, 0.08, generator=gen)
        ad.erase_up.weight.normal_(0.0, 0.08, generator=gen)
        ad.write_up.weight.normal_(0.0, 0.08, generator=gen)
        ad.decay_up.weight.normal_(0.0, 0.06, generator=gen)
        if getattr(ad, "theta_down", None) is not None and getattr(ad, "theta_up", None) is not None:
            ad.theta_down.weight.normal_(0.0, 0.08, generator=gen)
            ad.theta_up.weight.normal_(0.0, 0.08, generator=gen)
            if ad.theta_up.bias is not None:
                ad.theta_up.bias.normal_(0.0, 0.05, generator=gen)
        elif getattr(ad, "theta_proj", None) is not None:
            ad.theta_proj.weight.normal_(0.0, 0.08, generator=gen)
        ad.surprise_proj.weight.normal_(0.0, 0.08, generator=gen)
        ad.write_router.weight.normal_(0.0, 0.12, generator=gen)
        ad.read_router.weight.normal_(0.0, 0.12, generator=gen)
        ad.swa_q_up.weight.normal_(0.0, 0.08, generator=gen)
        ad.swa_k_up.weight.normal_(0.0, 0.08, generator=gen)
        ad.swa_v_up.weight.normal_(0.0, 0.08, generator=gen)
        ad.swa_recency_slope.copy_(torch.linspace(0.15, 0.75, ad.swa_recency_slope.numel()))


def _build_v32_hssd_toy_stack(
    *,
    seed: int = 320,
    blend_init: float = 0.0,
    blend_warmup_steps: int = 5,
    surprise_mode: str = "segment_start",
) -> Stack:
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed + 77)
    tok = ToyTokenizer()

    tcfg = ToyConfig(
        vocab_size=tok.vocab_size,
        hidden_size=48,
        moe_intermediate_size=24,
        shared_expert_intermediate_size=24,
        num_hidden_layers=5,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        partial_rotary_factor=0.5,
        linear_num_heads=4,
        linear_head_dim=12,
        linear_conv_kernel_dim=4,
        num_experts=6,
        num_experts_per_tok=2,
        layer_types=(
            "linear_attention",
            "full_attention",
            "linear_attention",
            "full_attention",
            "linear_attention",
        ),
    )
    model = ToyForCausalLM(tcfg).eval()

    hcfg = HSSDConfig(
        num_pages=4,
        top_k_pages=2,
        rank=8,
        chunk_size=8,
        window_chunks=2,
        segment_chunks=2,
        write_mode="full",
        use_stock_decay_init=True,
        swa_recency_init=0.35,
        horizon_tokens=131_072,
        surprise_mode=surprise_mode,
        blend_init=blend_init,
        blend_warmup_steps=blend_warmup_steps,
        compact_active_pages=True,
        compact_decode_cache=True,
    )
    hssd_h = apply_hssd(model, hcfg)
    for ad in hssd_h.adapters.values():
        _populate_hssd_v32_weights(ad, gen)

    lsa_h = install_lsa(
        model,
        LSAConfig(
            mode="sparse",
            index_dim=16,
            block_size=8,
            pool=2,
            avg_bucket_keys=4,
            max_bucket_keys=8,
            num_select=2,
            window=8,
            sinks=2,
            query_chunk=8,
            num_query_passes=2,
        ),
    )
    with torch.no_grad():
        for ix in lsa_h.indexers.values():
            ix.wk.normal_(0.0, 0.12, generator=gen)
            ix.wq.normal_(0.0, 0.12, generator=gen)
            for qp in getattr(ix, "q_proj_extra", ()):
                qp.weight.normal_(0.0, 0.10, generator=gen)

    hot_map = {2: [0, 3], 3: [1, 4]}
    deltas_h = attach_hot_expert_deltas(model, hot_map, rank=6, alpha=6.0)
    with torch.no_grad():
        for ld in deltas_h.deltas.values():
            for p in ld.gate_up_B.values():
                p.normal_(0.0, 0.04, generator=gen)
            for p in ld.down_B.values():
                p.normal_(0.0, 0.04, generator=gen)

    loop_h = apply_loop(
        model,
        LoopSpec(
            prelude_end=2,
            core_end=4,
            num_passes=2,
            target_virtual_layers=7,
            kv_share=True,
            gate_init=0.35,
            bridge_rank=8,
            bridge_mode="subspace",
            attn_adapter_rank=8,
            step_embedding=True,
            per_pass_router=True,
            router_adapter_rank=8,
            router_repel_init=1.2,
            mor=False,
        ),
    )
    with torch.no_grad():
        loop_h.adapters.gates[0].fill_(0.42)
        for u in loop_h.adapters.bridge.ups:
            u.weight.normal_(0.0, 0.08, generator=gen)
        if loop_h.adapters.pass_attn is not None:
            for pdict in loop_h.adapters.pass_attn.layers.values():
                for proj in pdict.values():
                    proj.up.normal_(0.0, 0.08, generator=gen)
        if loop_h.adapters.pass_router is not None:
            loop_h.adapters.pass_router.up.normal_(0.0, 0.12, generator=gen)
            loop_h.adapters.pass_router.bias.normal_(0.0, 0.06, generator=gen)

    model.eval()
    return Stack(
        model=model,
        tokenizer=tok,
        parts=inspect_model(model),
        loop=loop_h,
        hssd=hssd_h,
        lsa=lsa_h,
        deltas=deltas_h,
        load_info={"name": "toy"},
        hot_map=hot_map,
    )


def test_v32_export_whole_model_pack_metadata_and_live_plus_standalone_prefill_decode_parity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stack = _build_v32_hssd_toy_stack(seed=321, blend_init=0.0, blend_warmup_steps=5)
    assert stack.hssd is not None and stack.loop is not None
    # Advance scheduled blend warmup to completion (step 5 -> blend == 1.0)
    assert stack.hssd.step_blend(5) == 1.0

    # 52 tokens > (W + 4) * C = 48 so far-horizon boundary replay + segment_start decode are fully exercised!
    seq_36 = torch.tensor([[((i * 19 + 11) % 235) + 5 for i in range(52)]], dtype=torch.long)

    ref_logits: dict[int, torch.Tensor] = {}
    with torch.no_grad():
        for r in (1, 2):
            stack.loop.set_active_passes(r)
            ref_logits[r] = stack.model(seq_36, use_cache=False).logits.detach().clone()

    pack_dir = tmp_path / "hssd_v32_pack"
    export_whole_model_pack(stack, pack_dir, max_shard_size_bytes=65536, merge_in_place=False)

    cfg_json = json.loads((pack_dir / "config.json").read_text(encoding="utf-8"))
    hssd_arch = cfg_json["qloop_v3_arch"]["hssd"]
    assert hssd_arch["enabled"] is True
    assert hssd_arch["write_mode"] == "full"
    assert hssd_arch["use_stock_decay_init"] is True
    assert abs(float(hssd_arch["swa_recency_init"]) - 0.35) < 1e-9
    assert hssd_arch["horizon_tokens"] == 131_072
    assert hssd_arch["surprise_mode"] == "segment_start"
    assert abs(float(hssd_arch["blend_init"]) - 1.0) < 1e-9
    assert hssd_arch["compact_active_pages"] is True
    assert hssd_arch["compact_decode_cache"] is True

    idx_data = json.loads((pack_dir / SAFETENSORS_INDEX_FILE).read_text(encoding="utf-8"))
    assert any(k.endswith(".hssd_adapter.swa_recency_slope") for k in idx_data["weight_map"])

    loaded_live = load_whole_model_pack(pack_dir)
    mod_sa = _load_standalone_modeling_without_qloop(pack_dir, monkeypatch)
    loaded_sa = mod_sa.Qwen36LoopHSSDForCausalLM.from_pretrained(str(pack_dir), strict=True)

    with torch.no_grad():
        for r in (1, 2):
            with loaded_live.depth(r), loaded_sa.depth(r):
                live_full = loaded_live(seq_36, use_cache=False).logits
                sa_full = loaded_sa(seq_36, use_cache=False).logits
                assert (live_full - ref_logits[r]).abs().max().item() < 1e-5
                assert (sa_full - ref_logits[r]).abs().max().item() < 1e-5
                assert (sa_full - live_full).abs().max().item() < 1e-5

                # Multi-chunk prefill (11 + 9 = 20 tokens) + 32 single-token decode steps across chunk & segment bounds
                lp1 = loaded_live(seq_36[:, :11], use_cache=True)
                lp2 = loaded_live(seq_36[:, 11:20], past_key_values=lp1.past_key_values, use_cache=True)
                sp1 = loaded_sa(seq_36[:, :11], use_cache=True)
                sp2 = loaded_sa(seq_36[:, 11:20], past_key_values=sp1.past_key_values, use_cache=True)

                l_pkv, s_pkv = lp2.past_key_values, sp2.past_key_values
                l_parts = [lp1.logits, lp2.logits]
                s_parts = [sp1.logits, sp2.logits]
                for step in range(20, 52):
                    ld = loaded_live(seq_36[:, step : step + 1], past_key_values=l_pkv, use_cache=True)
                    sd = loaded_sa(seq_36[:, step : step + 1], past_key_values=s_pkv, use_cache=True)
                    l_parts.append(ld.logits)
                    s_parts.append(sd.logits)
                    l_pkv, s_pkv = ld.past_key_values, sd.past_key_values

                l_cat = torch.cat(l_parts, dim=1)
                s_cat = torch.cat(s_parts, dim=1)
                assert (l_cat - ref_logits[r]).abs().max().item() < 1e-5
                assert (s_cat - ref_logits[r]).abs().max().item() < 1e-5
                assert (s_cat - l_cat).abs().max().item() < 1e-5

                checked_states = 0
                for n_st, s_st in zip(l_pkv.recurrent_states, s_pkv.recurrent_states):
                    if s_st is not None:
                        assert n_st.is_compact_ring is True
                        assert s_st.is_compact_ring is True
                        assert s_st.state_equal(n_st, atol=1e-5, rtol=1e-4)
                        checked_states += 1
                assert checked_states > 0

    # Strict loader must reject missing `.hssd_adapter.swa_recency_slope` on a v3.2 pack
    corrupt_dir = tmp_path / "corrupt_missing_swa_slope"
    shutil.copytree(pack_dir, corrupt_dir)
    c_idx_path = corrupt_dir / SAFETENSORS_INDEX_FILE
    c_idx = json.loads(c_idx_path.read_text(encoding="utf-8"))
    slope_key = next(k for k in c_idx["weight_map"] if k.endswith(".hssd_adapter.swa_recency_slope"))
    slope_shard = c_idx["weight_map"].pop(slope_key)
    shard_tensors = load_safetensors_file(str(corrupt_dir / slope_shard))
    del shard_tensors[slope_key]
    save_safetensors_file(shard_tensors, str(corrupt_dir / slope_shard))
    c_idx_path.write_text(json.dumps(c_idx), encoding="utf-8")

    with pytest.raises(RuntimeError, match=r"missing_backbone=.*swa_recency_slope"):
        mod_sa.Qwen36LoopHSSDForCausalLM.from_pretrained(str(corrupt_dir), strict=True)


def test_v32_export_pack_mid_warmup_blend_preserved_in_live_and_standalone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stack = _build_v32_hssd_toy_stack(seed=329, blend_init=0.0, blend_warmup_steps=10, surprise_mode="chunk_start")
    assert stack.hssd is not None and stack.loop is not None
    prompt = torch.tensor([[7, 29, 54, 112, 83, 19, 177, 42, 95, 61, 140, 23, 88, 131, 16, 204]], dtype=torch.long)

    with torch.no_grad():
        stack.loop.set_active_passes(2)
        stack.hssd.set_blend(0.0)
        logits_b0 = stack.model(prompt, use_cache=False).logits.detach().clone()
        stack.hssd.set_blend(1.0)
        logits_b1 = stack.model(prompt, use_cache=False).logits.detach().clone()
        stack.hssd.set_blend(0.35)
        logits_b035 = stack.model(prompt, use_cache=False).logits.detach().clone()

    assert (logits_b035 - logits_b0).abs().max().item() > 1e-3
    assert (logits_b035 - logits_b1).abs().max().item() > 1e-3

    pack_dir = tmp_path / "hssd_v32_mid_blend_pack"
    export_whole_model_pack(stack, pack_dir, max_shard_size_bytes=65536, merge_in_place=False)

    cfg_json = json.loads((pack_dir / "config.json").read_text(encoding="utf-8"))
    assert abs(float(cfg_json["qloop_v3_arch"]["hssd"]["blend_init"]) - 0.35) < 1e-9

    loaded_live = load_whole_model_pack(pack_dir)
    mod_sa = _load_standalone_modeling_without_qloop(pack_dir, monkeypatch)
    loaded_sa = mod_sa.Qwen36LoopHSSDForCausalLM.from_pretrained(str(pack_dir), strict=True)

    with torch.no_grad(), loaded_live.depth(2), loaded_sa.depth(2):
        live_out = loaded_live(prompt, use_cache=False).logits
        sa_out = loaded_sa(prompt, use_cache=False).logits
        assert (live_out - logits_b035).abs().max().item() < 1e-5
        assert (sa_out - logits_b035).abs().max().item() < 1e-5
