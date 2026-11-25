"""Sprint 1 Track C regression suite: QL60-001, QL60-002, QL60-005, QL60-006, and D5.

Covers:
  1. `QL60-001`: `QuantizedHybridCache` modern `transformers 5.24+` `Cache(layers=...)` init,
     legacy fallback, full hybrid cache contract (`conv_states`, `recurrent_states`,
     `has_previous_state`, `get_seq_length`, `get_max_cache_shape`, `get_mask_sizes`,
     `reorder_cache`, `crop`, `reset`, `layers[idx]`), sparse `gather_kv`, and prefill /
     chunked-prefill / decode / depth `1/2/3` switching across `"native"`, `"fp8"`, and
     `"hadamard_int4"`.
  2. `QL60-002`: Exported Whole Model Pack HF `.generate()` contract across BOTH in-package
     and standalone emitted modeling files: `CausalLMPackOutput` (`ModelOutput` subclass +
     `.get()` / `__getitem__` / `keys()` / `to_tuple()`), outer virtual hybrid `layer_types` &
     `num_hidden_layers` (synced on `set_active_passes`), `logits_to_keep=0` returning full
     `(B, T, V)` logits, `prepare_inputs_for_generation` with implicit & explicit hybrid
     `past_key_values`, and `qloop_format_version` / `qloop_commit` metadata mismatch warnings.
  3. `QL60-005`: Strict release completion gate in `qloop.colab._local_stage_done` and Review 1
     Appendix F paused `s4_v2_rl` fixture (`LATEST = step000001`, Hub `is_completed=False` ->
     `all_stages_completed=False`, `pack_shipped=False`, zero ship commands).
  4. `QL60-006`: Clean v2 default pipeline stages across `qloop.colab` and
     `qloop.colab_supervisor` while keeping explicit legacy HSSD `--stages` usable.
  5. `D5`: Zero hardcoded `/Users/tianrui6641` paths in `src/qloop/report.py` and
     `scripts/build_sirsi_dataset.py`, plus `"pyarrow>=16.0"` in `pyproject.toml` `[dev]`.
"""
from __future__ import annotations

import ast
import dataclasses
import importlib
import importlib.util
import inspect
import json
import os
import sys
import types
import warnings
from collections import OrderedDict
from pathlib import Path
from typing import Any, ClassVar

import pytest
import torch

from qloop.colab import (
    DEFAULT_V2_STAGES,
    LEGACY_HSSD_STAGES,
    _local_stage_done,
    run_v3_e2e_and_ship_pack,
)
from qloop.colab_supervisor import (
    DEFAULT_STAGES as SUPERVISOR_DEFAULT_STAGES,
)
from qloop.colab_supervisor import (
    SupervisorConfig,
    build_arg_parser,
    config_from_args,
)
from qloop.data.tokenizer import ToyTokenizer
from qloop.export_pack import (
    PACK_MANIFEST_FILE,
    SAFETENSORS_INDEX_FILE,
    export_whole_model_pack,
    load_whole_model_pack,
)
from qloop.infer.engine import (
    InferenceConfig,
    OptimizedLoopInferenceEngine,
    QuantizedHybridCache,
)
from qloop.infer.hf_custom_modeling import (
    CONFIG_FILENAME,
    MODELING_FILENAME,
    QLOOP_FORMAT_VERSION,
    CausalLMPackOutput,
    Qwen36LoopHSSDConfig,
    Qwen36LoopHSSDForCausalLM,
    resolve_qloop_commit,
)
from qloop.model.gdn2 import GDN2Config, install_gdn2
from qloop.model.introspect import inspect_model
from qloop.model.loop import LoopSpec, apply_loop
from qloop.model.lsa import LSAConfig, install_lsa
from qloop.model.toy import ToyConfig, ToyForCausalLM, ToyHybridCache
from qloop.train.stack import Stack

ROOT = Path(__file__).resolve().parent.parent


def _build_loop_toy_stack(seed: int = 101) -> Stack:
    """Build a 6L -> 12L (3-pass) hybrid ToyForCausalLM with GDN-2 + D-LSA + Loop."""
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed + 17)
    tok = ToyTokenizer()
    tcfg = ToyConfig(
        vocab_size=tok.vocab_size,
        hidden_size=48,
        moe_intermediate_size=24,
        shared_expert_intermediate_size=24,
        num_hidden_layers=6,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        partial_rotary_factor=0.5,
        linear_num_heads=4,
        linear_head_dim=12,
        linear_conv_kernel_dim=4,
        num_experts=8,
        num_experts_per_tok=2,
        layer_types=[
            "linear_attention",  # 0 prelude
            "full_attention",    # 1 prelude
            "linear_attention",  # 2 core
            "linear_attention",  # 3 core
            "full_attention",    # 4 core
            "linear_attention",  # 5 coda (deliberately linear so coda cache_idx=11 checks hybrid types!)
        ],
    )
    model = ToyForCausalLM(tcfg).eval()

    gdn2 = install_gdn2(model, GDN2Config(rank=8, chunk_size=8))
    with torch.no_grad():
        for ad in gdn2.adapters.values():
            ad.erase_up.weight.normal_(0.0, 0.08, generator=gen)
            ad.write_up.weight.normal_(0.0, 0.08, generator=gen)
            ad.decay_up.weight.normal_(0.0, 0.05, generator=gen)

    lsa = install_lsa(
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
        ),
    )
    with torch.no_grad():
        for ix in lsa.indexers.values():
            ix.wk.normal_(0.0, 0.15, generator=gen)
            ix.wq.normal_(0.0, 0.15, generator=gen)

    loop = apply_loop(
        model,
        LoopSpec(
            prelude_end=2,
            core_end=5,
            num_passes=3,
            target_virtual_layers=12,
            kv_share=True,
            gate_init=0.30,
            bridge_rank=8,
            step_embedding=True,
            per_pass_router=True,
            router_adapter_rank=8,
            router_repel_init=1.0,
        ),
    )
    with torch.no_grad():
        loop.adapters.gates[0].fill_(0.35)
        loop.adapters.gates[1].fill_(0.45)
        if loop.adapters.bridge.rank > 0:
            loop.adapters.bridge.up.weight.normal_(0.0, 0.08, generator=gen)
        if loop.adapters.pass_router is not None:
            loop.adapters.pass_router.up.normal_(0.0, 0.12, generator=gen)

    model.eval()
    return Stack(
        model=model,
        tokenizer=tok,
        parts=inspect_model(model),
        loop=loop,
        gdn2=gdn2,
        lsa=lsa,
        load_info={"name": "toy"},
    )


def _make_mock_tf524_modules(*, strict_cache_init: bool = True) -> dict[str, types.ModuleType]:
    """Build faithful `transformers 5.24+` mock modules (Cache, ModelOutput, GenerationMixin, DynamicCache)."""
    tf_mod = types.ModuleType("transformers")
    tf_mod.__version__ = "5.24.1"  # type: ignore[attr-defined]
    cu_mod = types.ModuleType("transformers.cache_utils")
    mo_mod = types.ModuleType("transformers.modeling_outputs")

    class _MockCacheLayerMixin:
        pass

    class _MockTF524Cache:
        """Simulates `transformers 5.24.1` `Cache.__init__` raising ValueError without layers/layer_class."""

        init_calls: ClassVar[list[dict[str, Any]]] = []

        def __init__(self, layers: Any = None, layer_class_to_replicate: Any = None) -> None:
            type(self).init_calls.append({"layers": layers, "layer_class_to_replicate": layer_class_to_replicate})
            if strict_cache_init and ((layers is None) == (layer_class_to_replicate is None)):
                raise ValueError("You should provide exactly one of layers or layer_class_to_replicate")
            self.layers = list(layers) if layers is not None else []

    class _MockDynamicLayer(_MockCacheLayerMixin):
        supports_early_init = True

        def __init__(self) -> None:
            self.keys: torch.Tensor | None = None
            self.values: torch.Tensor | None = None
            self.is_initialized = False

        def update(self, k: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            if self.keys is None:
                self.keys, self.values = k, v
            else:
                self.keys = torch.cat([self.keys, k], dim=-2)
                self.values = torch.cat([self.values, v], dim=-2)  # type: ignore[list-item]
            self.is_initialized = True
            return self.keys, self.values  # type: ignore[return-value]

        def get_seq_length(self) -> int:
            return 0 if self.keys is None else int(self.keys.shape[-2])

    class _MockLinearAttentionLayer(_MockCacheLayerMixin):
        supports_early_init = False

        def __init__(self) -> None:
            self.conv_states: list[torch.Tensor | None] = [None]
            self.recurrent_states: list[Any] = [None]
            self.has_previous_state = {0: False}

    class _MockHybridDynamicCache(_MockTF524Cache):
        """Simulates HF 5.24 `DynamicCache(config)` checking `config.layer_types` strictly."""

        def __init__(self, config: Any) -> None:
            ltypes = getattr(config, "layer_types", None)
            n_layers = int(getattr(config, "num_hidden_layers", 0))
            if ltypes is None:
                ltypes = ["full_attention"] * n_layers
            built: list[Any] = [
                _MockLinearAttentionLayer() if lt == "linear_attention" else _MockDynamicLayer()
                for lt in ltypes
            ]
            super().__init__(layers=built)
            self.layer_types = list(ltypes)
            self.key_cache: list[torch.Tensor | None] = [None] * len(built)
            self.value_cache: list[torch.Tensor | None] = [None] * len(built)
            self.conv_states: list[torch.Tensor | None] = [None] * len(built)
            self.recurrent_states: list[Any] = [None] * len(built)
            self.first_full = (
                self.layer_types.index("full_attention") if "full_attention" in self.layer_types else 0
            )
            lin_idx = [i for i, lt in enumerate(self.layer_types) if lt == "linear_attention"]
            self.last_linear_layer = lin_idx[-1] if lin_idx else None

        def get_conv_state(self, layer_idx: int) -> torch.Tensor | None:
            lyr = self.layers[layer_idx]
            if not isinstance(lyr, _MockLinearAttentionLayer):
                raise ValueError(
                    f"Cannot get conv state for non linear attention layer {layer_idx} "
                    f"of type {type(lyr).__name__}"
                )
            return self.conv_states[layer_idx]

        def update(
            self,
            k: torch.Tensor,
            v: torch.Tensor,
            layer_idx: int,
            cache_kwargs: Any = None,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            del cache_kwargs
            lyr = self.layers[layer_idx]
            if not isinstance(lyr, _MockDynamicLayer):
                raise ValueError(f"Expected DynamicLayer at {layer_idx}, got {type(lyr).__name__}")
            if k.shape[-2] == 0 and lyr.keys is not None and lyr.values is not None:
                return lyr.keys, lyr.values
            rk, rv = lyr.update(k, v)
            self.key_cache[layer_idx] = rk
            self.value_cache[layer_idx] = rv
            return rk, rv

        def get_seq_length(self, layer_idx: int | None = None) -> int:
            idx = self.first_full if layer_idx is None else int(layer_idx)
            lyr = self.layers[idx]
            if isinstance(lyr, _MockDynamicLayer):
                return lyr.get_seq_length()
            return self.layers[self.first_full].get_seq_length()  # type: ignore[no-any-return]

        @property
        def is_initialized(self) -> bool:
            init_lyrs = [lyr for lyr in self.layers if getattr(lyr, "supports_early_init", False)]
            return bool(init_lyrs) and all(lyr.is_initialized for lyr in init_lyrs)

    class _MockModelOutput(OrderedDict):  # type: ignore[type-arg]
        """Mirrors HF `transformers.utils.generic.ModelOutput` requiring `@dataclass` subclasses."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            if self.__class__ != _MockModelOutput and not dataclasses.is_dataclass(self):
                raise TypeError(f"{self.__module__}.{self.__class__.__name__} has to be a dataclass.")

        def to_tuple(self) -> tuple[Any, ...]:
            return tuple(self[k] for k in self.keys())

    class _MockPretrainedConfig:
        model_type: str = "qwen36_loophssd"

        def __init__(self, **kwargs: Any) -> None:
            self.tie_word_embeddings = bool(kwargs.pop("tie_word_embeddings", False))
            self.architectures = list(kwargs.pop("architectures", ["Qwen36LoopHSSDForCausalLM"]))
            for k, v in kwargs.items():
                setattr(self, k, v)

        def to_dict(self) -> dict[str, Any]:
            out = {k: v for k, v in self.__dict__.items() if not k.startswith("_")}
            out["model_type"] = self.model_type
            return out

        def to_json_string(self, use_diff: bool = False) -> str:
            del use_diff
            return json.dumps(self.to_dict(), indent=2, sort_keys=True, default=str) + "\n"

        def save_pretrained(self, save_directory: str | os.PathLike[str], **_: Any) -> None:
            os.makedirs(save_directory, exist_ok=True)
            with open(os.path.join(str(save_directory), "config.json"), "w", encoding="utf-8") as f:
                f.write(self.to_json_string())

        @classmethod
        def from_pretrained(cls, pretrained_model_name_or_path: str | os.PathLike[str], **kw: Any) -> Any:
            kw.pop("trust_remote_code", None)
            cfg_p = os.path.join(str(pretrained_model_name_or_path), "config.json")
            with open(cfg_p, encoding="utf-8") as f:
                data = json.load(f)
            data.update(kw)
            return cls(**data)

    class _MockGenerationMixin:
        """Faithful mini-replica of HF 5.24 `GenerationMixin.generate()` greedy loop."""

        @torch.no_grad()
        def generate(
            self,
            input_ids: torch.Tensor,
            *,
            max_new_tokens: int = 3,
            past_key_values: Any = None,
            logits_to_keep: int = 0,
            use_cache: bool = True,
            **kwargs: Any,
        ) -> torch.Tensor:
            cur_ids = input_ids
            pkv = past_key_values
            if use_cache and pkv is None:
                pkv = _MockHybridDynamicCache(self.config)  # type: ignore[attr-defined]
            # Validate every linear layer slot on hybrid DynamicCache just like real HF Qwen3_5Moe
            if isinstance(pkv, _MockHybridDynamicCache):
                for idx, lt in enumerate(getattr(self.config, "layer_types", [])):  # type: ignore[attr-defined]
                    if lt == "linear_attention":
                        pkv.get_conv_state(idx)

            for step in range(int(max_new_tokens)):
                past_len = pkv.get_seq_length() if (pkv is not None and hasattr(pkv, "get_seq_length")) else 0
                if step == 0 and past_len == 0:
                    cache_pos = torch.arange(0, cur_ids.shape[1], device=cur_ids.device)
                elif step == 0 and past_len > 0:
                    n_new = max(1, cur_ids.shape[1] - past_len) if cur_ids.shape[1] > past_len else cur_ids.shape[1]
                    cache_pos = torch.arange(past_len, past_len + n_new, device=cur_ids.device)
                else:
                    cache_pos = torch.tensor([past_len], device=cur_ids.device)

                model_inputs = self.prepare_inputs_for_generation(  # type: ignore[attr-defined]
                    cur_ids,
                    past_key_values=pkv,
                    cache_position=cache_pos,
                    use_cache=use_cache,
                    logits_to_keep=logits_to_keep,
                    **kwargs,
                )
                outputs = self(**model_inputs)  # type: ignore[operator]
                # HF GenerationMixin uses `.get("past_key_values")` / `.get("logits")`!
                assert hasattr(outputs, "get"), "HF generate() requires ModelOutput.get()"
                logits = outputs.get("logits")
                assert logits is not None and logits.shape[1] >= 1, (
                    f"empty logits shape: {getattr(logits, 'shape', None)}"
                )
                pkv = outputs.get("past_key_values")
                nxt = logits[:, -1:, :].argmax(dim=-1)
                cur_ids = torch.cat([cur_ids, nxt], dim=1)
            return cur_ids

    class _MockPreTrainedModel(torch.nn.Module):
        config_class: Any = None

        def __init__(self, config: Any) -> None:
            super().__init__()
            self.config = config

        def post_init(self) -> None:
            return None

    class _MockAutoConfig:
        @staticmethod
        def for_model(model_type: str, **kw: Any) -> ToyConfig:
            del model_type
            valid = {f for f in ToyConfig.__dataclass_fields__ if f in kw}
            clean = {k: kw[k] for k in valid}
            if "layer_types" in clean and isinstance(clean["layer_types"], list):
                clean["layer_types"] = list(clean["layer_types"])
            return ToyConfig(**clean)

    class _MockAutoModelForCausalLM:
        @staticmethod
        def from_config(cfg: ToyConfig, **kw: Any) -> ToyForCausalLM:
            del kw
            return ToyForCausalLM(cfg)

    cu_mod.Cache = _MockTF524Cache  # type: ignore[attr-defined]
    cu_mod.CacheLayerMixin = _MockCacheLayerMixin  # type: ignore[attr-defined]
    cu_mod.DynamicCache = _MockHybridDynamicCache  # type: ignore[attr-defined]
    mo_mod.ModelOutput = _MockModelOutput  # type: ignore[attr-defined]

    tf_mod.PretrainedConfig = _MockPretrainedConfig  # type: ignore[attr-defined]
    tf_mod.PreTrainedConfig = _MockPretrainedConfig  # type: ignore[attr-defined]
    tf_mod.PreTrainedModel = _MockPreTrainedModel  # type: ignore[attr-defined]
    tf_mod.GenerationMixin = _MockGenerationMixin  # type: ignore[attr-defined]
    tf_mod.AutoConfig = _MockAutoConfig  # type: ignore[attr-defined]
    tf_mod.AutoModelForCausalLM = _MockAutoModelForCausalLM  # type: ignore[attr-defined]
    tf_mod.cache_utils = cu_mod  # type: ignore[attr-defined]
    tf_mod.modeling_outputs = mo_mod  # type: ignore[attr-defined]

    return {
        "transformers": tf_mod,
        "transformers.cache_utils": cu_mod,
        "transformers.modeling_outputs": mo_mod,
    }


def _load_standalone_pack_modules(
    pack_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    tf_mods: dict[str, types.ModuleType] | None = None,
) -> tuple[Any, Any]:
    """Import emitted standalone `configuration_*` and `modeling_*` with `qloop` hidden."""
    cfg_py = pack_dir / CONFIG_FILENAME
    mod_py = pack_dir / MODELING_FILENAME
    ast.parse(cfg_py.read_text(encoding="utf-8"), filename=CONFIG_FILENAME, feature_version=(3, 9))
    ast.parse(mod_py.read_text(encoding="utf-8"), filename=MODELING_FILENAME, feature_version=(3, 9))

    real_find_spec = importlib.util.find_spec
    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda name, pkg=None: None if (name == "qloop" or name.startswith("qloop.")) else real_find_spec(name, pkg),
    )
    mods = tf_mods if tf_mods is not None else _make_mock_tf524_modules()
    for mod_name, mod_obj in mods.items():
        monkeypatch.setitem(sys.modules, mod_name, mod_obj)

    pkg_name = f"_sa_sprint1_{abs(hash((str(pack_dir), id(mods))))}"
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
    return mod_cfg, mod_model


# ============================================================================ 1. QL60-001


def test_ql60_001_quantized_hybrid_cache_init_tf524_and_legacy(monkeypatch: pytest.MonkeyPatch) -> None:
    """`QuantizedHybridCache.__init__` must succeed cleanly under TF 5.24 `ValueError` and legacy `TypeError`."""
    import qloop.infer.engine as engine_mod

    # 1) Simulated transformers 5.24.1 `Cache.__init__` requiring `layers` OR `layer_class_to_replicate`
    tf_mods = _make_mock_tf524_modules(strict_cache_init=True)
    tf_cache_cls = tf_mods["transformers.cache_utils"].Cache
    tf_layer_cls = tf_mods["transformers.cache_utils"].CacheLayerMixin
    tf_cache_cls.init_calls.clear()

    for mname, mobj in tf_mods.items():
        monkeypatch.setitem(sys.modules, mname, mobj)

    reloaded = importlib.reload(engine_mod)
    try:
        ltypes = ["linear_attention", "linear_attention", "full_attention", "linear_attention"]
        qc = reloaded.QuantizedHybridCache(num_layers=4, layer_types=ltypes, mode="fp8")
        assert isinstance(qc, tf_cache_cls)
        assert len(tf_cache_cls.init_calls) == 1
        assert isinstance(tf_cache_cls.init_calls[0]["layers"], list)
        assert len(tf_cache_cls.init_calls[0]["layers"]) == 4
        assert len(qc.layers) == 4
        assert all(isinstance(lyr, tf_layer_cls) for lyr in qc.layers)
        assert qc.layers[0].supports_early_init is False
        assert qc.layers[2].supports_early_init is True
        assert qc.last_linear_layer == 3

        # 2) Simulated legacy transformers 4.x/5.0 `Cache.__init__(self)` raising `TypeError` on `layers=...`
        legacy_calls: list[int] = []

        class _LegacyCache:
            def __init__(self) -> None:
                legacy_calls.append(1)

        tf_mods["transformers.cache_utils"].Cache = _LegacyCache  # type: ignore[attr-defined]
        reloaded_legacy = importlib.reload(engine_mod)
        qc_leg = reloaded_legacy.QuantizedHybridCache(num_layers=2, layer_types=["linear_attention", "full_attention"])
        assert isinstance(qc_leg, _LegacyCache)
        assert legacy_calls == [1]
        assert len(qc_leg.layers) == 2

        # 3) Simulated strict `Cache.__init__` rejecting custom layer wrappers but accepting `layers=[]`
        empty_only_calls: list[int] = []

        class _StrictEmptyLayersCache:
            def __init__(self, layers: list[Any] | None = None, layer_class_to_replicate: Any = None) -> None:
                if (layers is None) == (layer_class_to_replicate is None):
                    raise ValueError("You should provide exactly one of layers or layer_class_to_replicate")
                if layers is not None and len(layers) > 0:
                    raise ValueError("unexpected custom layer wrapper")
                empty_only_calls.append(1)
                self.layers = list(layers or [])

        tf_mods["transformers.cache_utils"].Cache = _StrictEmptyLayersCache  # type: ignore[attr-defined]
        reloaded_strict = importlib.reload(engine_mod)
        qc_strict = reloaded_strict.QuantizedHybridCache(
            num_layers=2, layer_types=["linear_attention", "full_attention"]
        )
        assert isinstance(qc_strict, _StrictEmptyLayersCache)
        assert empty_only_calls == [1]
        assert len(qc_strict.layers) == 2
    finally:
        # Restore module cleanly without mocked transformers
        for mname in tuple(tf_mods):
            monkeypatch.delitem(sys.modules, mname, raising=False)
        importlib.reload(engine_mod)


@pytest.mark.parametrize("mode", ["native", "fp8", "hadamard_int4"])
def test_ql60_001_hybrid_cache_contract_methods_and_sparse_gather(mode: str) -> None:
    """Verify full HF 5.24+ hybrid cache contract, per-layer `DynamicLayer` views, and sparse `gather_kv`."""
    ltypes = ["linear_attention", "full_attention", "linear_attention", "full_attention"]
    cache = QuantizedHybridCache(num_layers=4, layer_types=ltypes, mode=mode)
    assert len(cache) == 4
    assert cache.first_full == 1
    assert cache.last_linear_layer == 2
    assert cache.get_max_cache_shape() is None
    assert cache.is_initialized is False
    assert bool(cache.has_previous_state) is False
    assert cache.has_previous_state(0) is False
    assert cache.layers[0].has_previous_state[0] is False
    assert cache.get_seq_length() == 0
    assert cache.get_seq_length(0) == 0
    assert cache.get_mask_sizes(torch.arange(0, 6), layer_idx=1) == (6, 0)

    # Update conv & recurrent state on linear layers via HF 5.24+ methods
    conv0 = torch.randn(2, 16, 4)
    rec0 = torch.randn(2, 4, 8, 8)
    full_conv = cache.update_conv_state(conv0, layer_idx=0, conv_kernel_size=4)
    assert full_conv.shape == (2, 16, 4)
    assert torch.equal(cache.get_conv_state(0), conv0)  # type: ignore[arg-type]
    cache.update_recurrent_state(rec0, layer_idx=0)
    assert torch.equal(cache.get_recurrent_state(0), rec0)
    assert cache.has_previous_state(0) is True
    assert cache.layers[0].has_previous_state[0] is True
    assert cache.layers[0].is_initialized is True
    assert cache.has_previous_state(2) is False
    assert bool(cache.has_previous_state) is False  # last_linear_layer == 2 not yet populated

    cache.layers[2].update_conv_state(conv0 * 0.5, conv_kernel_size=4)
    cache.layers[2].update_recurrent_state(rec0 * 0.5)
    assert bool(cache.has_previous_state) is True

    # Prefill + append KV on full-attention layers (via both cache.update and cache.layers[idx].update)
    k1 = torch.randn(2, 2, 6, 16)
    v1 = torch.randn(2, 2, 6, 16)
    k1_out, v1_out = cache.update(k1, v1, layer_idx=1)
    assert k1_out.shape == (2, 2, 6, 16) and v1_out.shape == (2, 2, 6, 16)
    k3_out, v3_out = cache.layers[3].update(k1, v1)
    assert k3_out.shape == (2, 2, 6, 16) and v3_out.shape == (2, 2, 6, 16)

    # Decode append (T=1)
    k_dec = torch.randn(2, 2, 1, 16)
    v_dec = torch.randn(2, 2, 1, 16)
    k_all, v_all = cache.update(k_dec, v_dec, layer_idx=1)
    assert k_all.shape == (2, 2, 7, 16)
    assert cache.get_seq_length() == 7
    assert cache.get_seq_length(0) == 7  # linear layer queries first_full just like HF DynamicCache!
    assert cache.get_seq_length(1) == 7
    assert cache.layers[1].get_seq_length() == 7
    assert cache.layers[1].is_initialized is True
    assert cache.get_mask_sizes(torch.tensor([7]), layer_idx=1) == (8, 0)
    assert cache.layers[1].get_mask_sizes(torch.tensor([7])) == (8, 0)
    assert cache.is_initialized is True

    # Sparse gather_kv must exactly match full dequantized slice without materializing unused tokens
    sel_idx = torch.tensor([0, 3, 6], dtype=torch.long)
    gk, gv = cache.gather_kv(1, sel_idx, dtype=torch.float32)
    assert gk.shape == (2, 2, 3, 16) and gv.shape == (2, 2, 3, 16)
    assert torch.equal(gk, k_all.index_select(-2, sel_idx))
    assert torch.equal(gv, v_all.index_select(-2, sel_idx))

    # Reorder, batch_repeat_interleave, crop, and reset
    beam = torch.tensor([1, 0], dtype=torch.long)
    cache.reorder_cache(beam)
    k_reord, v_reord = cache.update(k_dec[..., :0, :], v_dec[..., :0, :], layer_idx=1)
    assert torch.allclose(k_reord, k_all[beam], atol=1e-5)
    assert torch.allclose(v_reord, v_all[beam], atol=1e-5)
    assert torch.equal(cache.conv_states[0], conv0[beam])  # type: ignore[arg-type]

    cache.batch_repeat_interleave(2)
    assert cache.layers[1].keys is not None and cache.layers[1].keys.shape[0] == 4
    assert cache.conv_states[0] is not None and cache.conv_states[0].shape[0] == 4

    cache.crop(4)
    assert cache.get_seq_length(1) == 4
    cache.crop(-1)  # relative crop: 4 - 1 = 3
    assert cache.get_seq_length(1) == 3

    cache.reset()
    assert cache.get_seq_length() == 0
    assert cache.is_initialized is False
    assert bool(cache.has_previous_state) is False
    assert cache.kv_bytes() == 0 and cache.recurrent_bytes() == 0

    # Verify update docstring honestly documents full K/V materialization vs sparse gather_kv
    doc = (QuantizedHybridCache.update.__doc__ or "").lower()
    assert "materializes" in doc and "gather_kv" in doc


@pytest.mark.parametrize("mode", ["native", "fp8", "hadamard_int4"])
def test_ql60_001_engine_prefill_chunked_decode_depths_across_quant_modes(mode: str) -> None:
    """Verify prefill, chunked prefill, decode, reorder/reset, and depth `1/2/3` across all KV modes."""
    stack = _build_loop_toy_stack(seed=109)
    engine = OptimizedLoopInferenceEngine.from_stack(
        stack,
        cfg=InferenceConfig(prefill_chunk_size=5, kv_cache_quant=mode, temperature=0.0),
    )
    prompt = torch.tensor([[7, 15, 23, 31, 42, 55, 63, 71, 82, 91, 103]], dtype=torch.long)

    kv_mbs: list[float] = []
    for depth in (1, 2, 3):
        with engine.depth(depth):
            assert engine.active_passes == depth
            # Explicit QuantizedHybridCache chunked prefill + single-token decode + reorder + reset
            n_virt = int(engine.text_model.config.num_hidden_layers)
            ltypes = list(engine.text_model.config.layer_types)
            qcache = QuantizedHybridCache(num_layers=n_virt, layer_types=ltypes, mode=mode)
            logits_chunked, qcache, n_chunks = engine.prefill(prompt, cache=qcache, chunk_size=4)
            assert n_chunks == 3
            assert logits_chunked.shape == (1, 1, stack.tokenizer.vocab_size)
            assert qcache.get_seq_length() == prompt.shape[1]
            # 1-token decode step
            nxt_tok = logits_chunked[:, -1:, :].argmax(dim=-1)
            tm_out, qcache, _ = engine._run_text_model_step(nxt_tok, qcache)
            assert tm_out.last_hidden_state.shape == (1, 1, 48)
            assert qcache.get_seq_length() == prompt.shape[1] + 1
            # Reorder (identity) + crop + reset
            qcache.reorder_cache(torch.tensor([0], dtype=torch.long))
            qcache.crop(prompt.shape[1])
            assert qcache.get_seq_length() == prompt.shape[1]
            qcache.reset()
            assert qcache.get_seq_length() == 0

            # End-to-end engine.generate with 1.0x SharedKV verification
            res = engine.generate(prompt, max_new_tokens=3, ignore_eos=True)
            assert len(res.token_ids) == 3
            assert res.metrics.active_passes == depth
            kv_mbs.append(res.metrics.kv_cache_mb)

    assert kv_mbs[0] > 0.0
    assert abs(kv_mbs[1] - kv_mbs[0]) < 1e-6
    assert abs(kv_mbs[2] - kv_mbs[0]) < 1e-6


# ============================================================================ 2. QL60-002


def test_ql60_002_causal_lm_pack_output_model_output_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`CausalLMPackOutput` must inherit HF `ModelOutput` when available and support full dict/tuple API always."""
    logits = torch.randn(1, 4, 16)
    loss = torch.tensor(1.25)
    dummy_cache = {"seq": 4}

    # 1) Offline / without transformers (current module state)
    out = CausalLMPackOutput(logits=logits, past_key_values=dummy_cache)
    assert isinstance(out, OrderedDict)
    assert out.logits is logits
    assert out.past_key_values is dummy_cache
    assert out.cache is dummy_cache
    assert out.get("logits") is logits
    assert out.get("past_key_values") is dummy_cache
    assert out.get("cache") is dummy_cache
    assert out.get("loss") is None
    assert out.get("missing", "fallback") == "fallback"
    assert out["logits"] is logits
    assert out["cache"] is dummy_cache
    assert out[0] is logits
    assert out[1] is dummy_cache
    assert list(out.keys()) == ["logits", "past_key_values"]
    assert out.to_tuple() == (logits, dummy_cache)

    out_with_loss = CausalLMPackOutput(logits=logits, loss=loss, past_key_values=dummy_cache)
    assert list(out_with_loss.keys()) == ["loss", "logits", "past_key_values"]
    assert out_with_loss.to_tuple() == (loss, logits, dummy_cache)
    assert out_with_loss[0] is loss and out_with_loss[1] is logits

    # Dynamic attribute update stays synced in OrderedDict
    new_cache = {"seq": 5}
    out.past_key_values = new_cache
    assert out.get("past_key_values") is new_cache
    assert out["past_key_values"] is new_cache

    # 2) With mock HF `transformers.modeling_outputs.ModelOutput` (both in-package and standalone!)
    tf_mods = _make_mock_tf524_modules()
    mock_model_output_cls = tf_mods["transformers.modeling_outputs"].ModelOutput

    stack = _build_loop_toy_stack(seed=113)
    pack_dir = tmp_path / "mo_pack"
    export_whole_model_pack(stack, pack_dir, max_shard_size_bytes=65536, merge_in_place=False)

    _, mod_sa = _load_standalone_pack_modules(pack_dir, monkeypatch, tf_mods=tf_mods)
    sa_out = mod_sa.CausalLMPackOutput(logits=logits, past_key_values=dummy_cache)
    assert isinstance(sa_out, mock_model_output_cls)
    assert dataclasses.is_dataclass(sa_out)
    assert sa_out.get("logits") is logits
    assert sa_out.get("past_key_values") is dummy_cache
    assert sa_out.get("cache") is dummy_cache
    assert sa_out["logits"] is logits
    assert sa_out[0] is logits
    assert list(sa_out.keys()) == ["logits", "past_key_values"]
    assert sa_out.to_tuple() == (logits, dummy_cache)

    import qloop.infer.hf_custom_modeling as hcm_mod

    for mname, mobj in tf_mods.items():
        monkeypatch.setitem(sys.modules, mname, mobj)
    reloaded_hcm = importlib.reload(hcm_mod)
    try:
        inpkg_out = reloaded_hcm.CausalLMPackOutput(logits=logits, past_key_values=dummy_cache)
        assert isinstance(inpkg_out, mock_model_output_cls)
        assert dataclasses.is_dataclass(inpkg_out)
        assert inpkg_out.get("logits") is logits
        assert inpkg_out.get("past_key_values") is dummy_cache
        assert inpkg_out.to_tuple() == (logits, dummy_cache)
    finally:
        for mname in tuple(tf_mods):
            monkeypatch.delitem(sys.modules, mname, raising=False)
        importlib.reload(hcm_mod)


def test_ql60_002_outer_config_virtual_layers_logits_to_keep_zero_and_hf_generate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify virtual hybrid `layer_types`, `logits_to_keep=0`, HF `.generate()` (implicit & explicit cache),
    standalone codegen parity, and non-breaking version/commit mismatch warnings."""
    stack = _build_loop_toy_stack(seed=127)
    pack_dir = tmp_path / "hf_gen_pack"
    export_whole_model_pack(stack, pack_dir, max_shard_size_bytes=65536, merge_in_place=False)

    # 1) Exported config.json + manifest + index embed virtual hybrid layer_types & version metadata
    cfg_json = json.loads((pack_dir / "config.json").read_text(encoding="utf-8"))
    idx_json = json.loads((pack_dir / SAFETENSORS_INDEX_FILE).read_text(encoding="utf-8"))
    man_json = json.loads((pack_dir / PACK_MANIFEST_FILE).read_text(encoding="utf-8"))

    expected_virt_lt = [
        "linear_attention", "full_attention",                                # prelude (2)
        "linear_attention", "linear_attention", "full_attention",            # core pass 1 (3)
        "linear_attention", "linear_attention", "full_attention",            # core pass 2 (3)
        "linear_attention", "linear_attention", "full_attention",            # core pass 3 (3)
        "linear_attention",                                                  # coda (1)
    ]
    assert cfg_json["base_config"]["num_hidden_layers"] == 6
    assert cfg_json["num_hidden_layers"] == 12
    assert cfg_json["layer_types"] == expected_virt_lt
    assert cfg_json["qloop_format_version"] == QLOOP_FORMAT_VERSION
    assert cfg_json["qloop_commit"] == resolve_qloop_commit()
    assert idx_json["metadata"]["qloop_format_version"] == QLOOP_FORMAT_VERSION
    assert idx_json["metadata"]["qloop_commit"] == resolve_qloop_commit()
    assert man_json["qloop_format_version"] == QLOOP_FORMAT_VERSION
    assert man_json["qloop_commit"] == resolve_qloop_commit()

    # 1b) Legacy config.json without outer layer_types / with physical num_hidden_layers upgrades cleanly
    legacy_cfg = Qwen36LoopHSSDConfig(
        base_config=cfg_json["base_config"],
        qloop_v3_arch=cfg_json["qloop_v3_arch"],
        num_hidden_layers=6,
    )
    assert legacy_cfg.num_hidden_layers == 12
    assert legacy_cfg.layer_types == expected_virt_lt

    # 2) In-package model: config depth sync + logits_to_keep=0 / 1 / 3 / tensor + prepare_inputs_for_generation
    loaded = load_whole_model_pack(pack_dir)
    hf_model = loaded.hf_model
    assert hf_model is not None

    prompt = torch.tensor([[11, 22, 33, 44, 55, 66]], dtype=torch.long)
    with torch.no_grad():
        for r, eff_n in ((1, 6), (2, 9), (3, 12)):
            with hf_model.depth(r):
                assert hf_model.config.active_passes == r
                assert hf_model.config.effective_num_hidden_layers == eff_n
                assert len(hf_model.config.active_layer_types) == eff_n
                assert hf_model.config.num_hidden_layers == 12
                assert hf_model.config.layer_types == expected_virt_lt
                assert hf_model.config.qloop_v3_arch["loop"]["active_passes"] == r

                out_full = hf_model(prompt, use_cache=False)
                out_k0 = hf_model(prompt, use_cache=False, logits_to_keep=0)
                out_k1 = hf_model(prompt, use_cache=False, logits_to_keep=1)
                out_k3 = hf_model(prompt, use_cache=False, logits_to_keep=3)
                out_kt = hf_model(prompt, use_cache=False, logits_to_keep=torch.tensor([0, 4]))

                # Critical QL60-002 fix: logits_to_keep=0 returns ALL (B, T, V) logits, NOT (B, 0, V)!
                assert out_k0.logits.shape == (1, 6, stack.tokenizer.vocab_size)
                assert torch.allclose(out_k0.logits, out_full.logits, atol=1e-6)
                assert out_k1.logits.shape == (1, 1, stack.tokenizer.vocab_size)
                assert torch.allclose(out_k1.logits, out_full.logits[:, -1:, :], atol=1e-6)
                assert out_k3.logits.shape == (1, 3, stack.tokenizer.vocab_size)
                assert torch.allclose(out_k3.logits, out_full.logits[:, -3:, :], atol=1e-6)
                assert out_kt.logits.shape == (1, 2, stack.tokenizer.vocab_size)
                assert torch.allclose(out_kt.logits, out_full.logits[:, [0, 4], :], atol=1e-6)
                assert out_k0.get("logits") is out_k0.logits
                assert out_full.get("logits") is out_full.logits

    # 3) Standalone emitted model with mock HF `GenerationMixin` + hybrid `DynamicCache`:
    #    test BOTH implicit cache `.generate(ids, max_new_tokens=3)` AND explicit `past_key_values=cache`!
    tf_mods = _make_mock_tf524_modules(strict_cache_init=True)
    _, mod_sa = _load_standalone_pack_modules(pack_dir, monkeypatch, tf_mods=tf_mods)
    sa_model = mod_sa.Qwen36LoopHSSDForCausalLM.from_pretrained(str(pack_dir), strict=True)
    assert sa_model.config.layer_types == expected_virt_lt
    assert sa_model.config.num_hidden_layers == 12

    with torch.no_grad():
        for r, eff_n in ((1, 6), (2, 9), (3, 12)):
            with sa_model.depth(r):
                assert sa_model.config.active_passes == r
                assert sa_model.config.effective_num_hidden_layers == eff_n
                assert sa_model.config.layer_types == expected_virt_lt

                # Standalone logits_to_keep=0 parity vs in-package
                hf_model.set_active_passes(r)
                ref_logits = hf_model(prompt, use_cache=False, logits_to_keep=0).logits
                sa_out_k0 = sa_model(prompt, use_cache=False, logits_to_keep=0)
                assert sa_out_k0.logits.shape == (1, 6, stack.tokenizer.vocab_size)
                assert (sa_out_k0.logits - ref_logits).abs().max().item() < 1e-5
                assert sa_out_k0.get("logits") is sa_out_k0.logits

                # Reference greedy continuation via ToyForCausalLM.greedy_generate
                ref_seq = hf_model.backbone.greedy_generate(prompt, max_new_tokens=3)

                # (a) HF `.generate(ids, max_new_tokens=3)` with implicit hybrid DynamicCache & logits_to_keep=0
                gen_implicit_k0 = sa_model.generate(prompt, max_new_tokens=3, logits_to_keep=0)
                assert torch.equal(gen_implicit_k0, ref_seq)

                # (b) HF `.generate(ids, max_new_tokens=3)` with logits_to_keep=1
                gen_implicit_k1 = sa_model.generate(prompt, max_new_tokens=3, logits_to_keep=1)
                assert torch.equal(gen_implicit_k1, ref_seq)

                # (c) HF `.generate(ids, max_new_tokens=3, past_key_values=explicit_cache)` (QuantizedHybridCache)
                q_nat = QuantizedHybridCache(num_layers=12, layer_types=expected_virt_lt, mode="native")
                gen_explicit_nat = sa_model.generate(prompt, max_new_tokens=3, past_key_values=q_nat, logits_to_keep=1)
                assert torch.equal(gen_explicit_nat, ref_seq)
                assert q_nat.get_seq_length() == prompt.shape[1] + 2

                q_fp8 = QuantizedHybridCache(num_layers=12, layer_types=expected_virt_lt, mode="fp8")
                gen_explicit_fp8 = sa_model.generate(prompt, max_new_tokens=3, past_key_values=q_fp8, logits_to_keep=0)
                assert gen_explicit_fp8.shape == (1, prompt.shape[1] + 3)
                assert q_fp8.get_seq_length() == prompt.shape[1] + 2

                # (d) Explicit `ToyHybridCache` passed to `.generate()`
                toy_c = ToyHybridCache(sa_model._find_tm().config)
                gen_explicit_toy = sa_model.generate(prompt, max_new_tokens=3, past_key_values=toy_c, logits_to_keep=1)
                assert torch.equal(gen_explicit_toy, ref_seq)

    # 3b) In-package `Qwen36LoopHSSDForCausalLM.generate()` under mock HF `GenerationMixin` + hybrid `DynamicCache`
    import qloop.infer.hf_custom_modeling as hcm_mod

    for mname, mobj in tf_mods.items():
        monkeypatch.setitem(sys.modules, mname, mobj)
    reloaded_hcm = importlib.reload(hcm_mod)
    try:
        inpkg_gen_model = reloaded_hcm.Qwen36LoopHSSDForCausalLM.from_pretrained(pack_dir)
        with torch.no_grad():
            for r in (1, 2, 3):
                with inpkg_gen_model.depth(r):
                    ref_seq = inpkg_gen_model.backbone.greedy_generate(prompt, max_new_tokens=3)
                    out_imp = inpkg_gen_model.generate(prompt, max_new_tokens=3, logits_to_keep=0)
                    assert torch.equal(out_imp, ref_seq)
                    q_exp = QuantizedHybridCache(num_layers=12, layer_types=expected_virt_lt, mode="native")
                    out_exp = inpkg_gen_model.generate(
                        prompt, max_new_tokens=3, past_key_values=q_exp, logits_to_keep=1
                    )
                    assert torch.equal(out_exp, ref_seq)
    finally:
        for mname in tuple(tf_mods):
            monkeypatch.delitem(sys.modules, mname, raising=False)
        importlib.reload(hcm_mod)

    # 4) Non-breaking warning when qloop_format_version / qloop_commit mismatches
    mut_cfg = json.loads((pack_dir / "config.json").read_text(encoding="utf-8"))
    mut_cfg["qloop_format_version"] = "2.0"
    mut_cfg["qloop_commit"] = "0000deadbeef"
    (pack_dir / "config.json").write_text(json.dumps(mut_cfg, indent=2), encoding="utf-8")

    with warnings.catch_warnings(record=True) as rec_inpkg:
        warnings.simplefilter("always")
        warn_model = Qwen36LoopHSSDForCausalLM.from_pretrained(pack_dir)
    msgs_inpkg = [str(w.message) for w in rec_inpkg]
    assert any("qloop_format_version" in m for m in msgs_inpkg)
    assert any("qloop_commit" in m for m in msgs_inpkg)
    assert warn_model(prompt, use_cache=False, logits_to_keep=0).logits.shape == (1, 6, stack.tokenizer.vocab_size)

    with warnings.catch_warnings(record=True) as rec_sa:
        warnings.simplefilter("always")
        warn_sa = mod_sa.Qwen36LoopHSSDForCausalLM.from_pretrained(str(pack_dir), strict=True)
    assert any("qloop_format_version" in str(w.message) for w in rec_sa)
    assert warn_sa(prompt, use_cache=False, logits_to_keep=0).logits.shape == (1, 6, stack.tokenizer.vocab_size)


# ============================================================================ 3. QL60-005


def test_ql60_005_appendix_f_paused_s4_v2_rl_blocks_pack_shipping(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Review 1 Appendix F fixture: paused `s4_v2_rl` with ONLY `LATEST=step000001` MUST NOT ship pack."""
    monkeypatch.setenv("QLOOP_RUN_ROOT", str(tmp_path))

    # Prior stages marked completed on Hub; final `s4_v2_rl` paused at step 1 with ONLY `checkpoints/LATEST`
    s4_ck_dir = tmp_path / "s4_v2_rl" / "checkpoints"
    (s4_ck_dir / "step000001").mkdir(parents=True, exist_ok=True)
    (s4_ck_dir / "LATEST").write_text("step000001\n", encoding="utf-8")

    class _PausedS4Manifest:
        def is_completed(self, stage: str) -> bool:
            return stage != "s4_v2_rl"

    monkeypatch.setattr("qloop.hub.HubSyncManager.fetch_manifest", lambda self: _PausedS4Manifest())
    monkeypatch.setattr(
        "qloop.colab.ensure_quant_runtime_packages",
        lambda base_precision="nvfp4", *, auto_install=True: {
            "precision": str(base_precision),
            "package": "stub",
            "pip_spec": "",
            "status": "ready",
        },
    )
    issued_cmds: list[str] = []
    monkeypatch.setattr("qloop.colab.q", lambda cmd, check=True: issued_cmds.append(cmd) or 0)

    res = run_v3_e2e_and_ship_pack(
        hf_repo="test/repo",
        pack_repo="test/pack",
        stages=DEFAULT_V2_STAGES,
    )
    assert res["pipeline_rc"] == 0
    assert res["all_stages_completed"] is False
    assert res["pack_shipped"] is False
    assert not any(c.startswith("export-pack") for c in issued_cmds)
    assert not any(c.startswith("sync disk") for c in issued_cmds)
    assert len(issued_cmds) == 1 and issued_cmds[0].startswith("sync pipeline")


def test_ql60_005_local_stage_done_strict_markers_and_metadata(tmp_path: Path) -> None:
    """Verify `_local_stage_done` rejects partial `LATEST` and accepts explicit completion proof."""
    run_root = str(tmp_path)

    # 1) Bare LATEST without state.json -> False
    ck = tmp_path / "s4_v2_rl" / "checkpoints"
    (ck / "step000001").mkdir(parents=True)
    (ck / "LATEST").write_text("step000001\n", encoding="utf-8")
    assert _local_stage_done(run_root, "s4_v2_rl") is False

    # 2) LATEST with incomplete step < target_steps -> False
    (ck / "step000001" / "state.json").write_text(
        json.dumps({"step": 1, "target_steps": 400, "status": "in_progress"}), encoding="utf-8"
    )
    assert _local_stage_done(run_root, "s4_v2_rl") is False

    # 3) LATEST with only {"step": 1} (no target_steps / status / completed) -> False
    (ck / "step000001" / "state.json").write_text(json.dumps({"step": 1}), encoding="utf-8")
    assert _local_stage_done(run_root, "s4_v2_rl") is False

    # 4) LATEST with verified step >= target_steps -> True
    (ck / "step000001" / "state.json").write_text(
        json.dumps({"step": 400, "target_steps": 400}), encoding="utf-8"
    )
    assert _local_stage_done(run_root, "s4_v2_rl") is True

    # 5) LATEST with explicit status == "completed" -> True
    (ck / "step000001" / "state.json").write_text(json.dumps({"status": "completed"}), encoding="utf-8")
    assert _local_stage_done(run_root, "s4_v2_rl") is True

    # 6) Explicit COMPLETED marker -> True even with partial LATEST
    ck2 = tmp_path / "s2_v2_joint_bptt" / "checkpoints"
    (ck2 / "step000050").mkdir(parents=True)
    (ck2 / "LATEST").write_text("step000050\n", encoding="utf-8")
    assert _local_stage_done(run_root, "s2_v2_joint_bptt") is False
    (ck2 / "COMPLETED").write_text("step000050\n", encoding="utf-8")
    assert _local_stage_done(run_root, "s2_v2_joint_bptt") is True

    # 7) PROMOTED marker: accepted when finalized, rejected when explicitly contradicted by incomplete state.json
    ck3 = tmp_path / "s4_rl" / "checkpoints"
    (ck3 / "step000025").mkdir(parents=True)
    (ck3 / "PROMOTED").write_text("step000025\n", encoding="utf-8")
    (ck3 / "step000025" / "state.json").write_text(
        json.dumps({"step": 25, "target_steps": 400, "status": "in_progress"}), encoding="utf-8"
    )
    assert _local_stage_done(run_root, "s4_rl") is False
    (ck3 / "step000025" / "state.json").write_text(
        json.dumps({"step": 400, "target_steps": 400, "status": "completed"}), encoding="utf-8"
    )
    assert _local_stage_done(run_root, "s4_rl") is True


# ============================================================================ 4. QL60-006


def test_ql60_006_clean_v2_default_stages_and_explicit_hssd_opt_in(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Default orchestration must use clean v2 stages while keeping legacy HSSD usable via explicit `--stages`."""
    expected_v2 = (
        "s0_probe",
        "s1_loop_a6b",
        "s3_v2_long_1m_lsa",
        "s2_v2_joint_bptt",
        "s4_v2_rl",
    )
    assert expected_v2 == DEFAULT_V2_STAGES
    assert expected_v2 == SUPERVISOR_DEFAULT_STAGES
    assert SupervisorConfig().stages == expected_v2

    parsed_default = config_from_args(build_arg_parser().parse_args([]))
    assert parsed_default.stages == expected_v2
    assert "s1b_hssd_distill" not in parsed_default.stages

    sig_default = inspect.signature(run_v3_e2e_and_ship_pack).parameters["stages"].default
    assert tuple(sig_default) == expected_v2

    # Legacy HSSD remains explicitly usable via `--stages` on supervisor CLI and `run_v3_e2e_and_ship_pack`
    legacy_csv = ",".join(LEGACY_HSSD_STAGES)
    parsed_legacy = config_from_args(build_arg_parser().parse_args(["--stages", legacy_csv]))
    assert parsed_legacy.stages == LEGACY_HSSD_STAGES

    monkeypatch.setenv("QLOOP_RUN_ROOT", str(tmp_path))

    class _NoneDoneManifest:
        def is_completed(self, _stage: str) -> bool:
            return False

    monkeypatch.setattr("qloop.hub.HubSyncManager.fetch_manifest", lambda self: _NoneDoneManifest())
    monkeypatch.setattr(
        "qloop.colab.ensure_quant_runtime_packages",
        lambda base_precision="nvfp4", *, auto_install=True: {
            "precision": str(base_precision),
            "package": "stub",
            "pip_spec": "",
            "status": "ready",
        },
    )
    cmds: list[str] = []
    monkeypatch.setattr("qloop.colab.q", lambda cmd, check=True: cmds.append(cmd) or 0)

    res_v2 = run_v3_e2e_and_ship_pack(hf_repo="test/repo", pack_repo="test/pack")
    assert tuple(res_v2["stages"]) == expected_v2
    assert f"--stages {','.join(expected_v2)}" in cmds[-1]

    res_hssd = run_v3_e2e_and_ship_pack(hf_repo="test/repo", pack_repo="test/pack", stages=LEGACY_HSSD_STAGES)
    assert tuple(res_hssd["stages"]) == LEGACY_HSSD_STAGES
    assert f"--stages {legacy_csv}" in cmds[-1]


# ============================================================================ 5. D5


def test_d5_portable_paths_and_pyproject_dev_pyarrow() -> None:
    """Zero hardcoded `/Users/tianrui6641` in `report.py` / `build_sirsi_dataset.py` and `pyarrow>=16.0` in dev."""
    report_path = ROOT / "src" / "qloop" / "report.py"
    builder_path = ROOT / "scripts" / "build_sirsi_dataset.py"
    pyproject_path = ROOT / "pyproject.toml"

    report_src = report_path.read_text(encoding="utf-8")
    builder_src = builder_path.read_text(encoding="utf-8")
    pyproject_src = pyproject_path.read_text(encoding="utf-8")

    assert "/Users/tianrui6641" not in report_src
    assert "/Users/tianrui6641" not in builder_src
    assert "shutil.which(" in report_src and "Path.home()" in report_src
    assert "Path(__file__).resolve().parents[1]" in builder_src

    # Import & validate resolved runtime defaults
    import qloop.report as report_mod

    assert isinstance(report_mod.DEFAULT_COLAB_BIN, str) and len(report_mod.DEFAULT_COLAB_BIN) > 0

    spec = importlib.util.spec_from_file_location("_test_d5_sirsi_builder", builder_path)
    assert spec is not None and spec.loader is not None
    builder_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder_mod)
    assert Path(builder_mod.DEFAULT_OUT_DIR) == ROOT / "runs" / "datasets" / "Loop64-Fable-SIRSI-v1"
    assert Path(builder_mod.DEFAULT_DECONTAM_CACHE) == ROOT / "runs" / "datasets" / ".decontam_cache"

    # Verify pyarrow>=16.0 is declared in `[project.optional-dependencies].dev`
    in_opt_deps = False
    dev_line = ""
    for line in pyproject_src.splitlines():
        s = line.strip()
        if s.startswith("["):
            in_opt_deps = s == "[project.optional-dependencies]"
        elif in_opt_deps and s.startswith("dev"):
            dev_line = s
            break
    assert "pyarrow>=16.0" in dev_line and "pytest" in dev_line
