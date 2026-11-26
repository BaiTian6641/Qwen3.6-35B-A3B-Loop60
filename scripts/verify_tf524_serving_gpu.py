#!/usr/bin/env python3
"""Live `transformers 5.24.x` serving & Whole Model Pack audit CLI for Sprint 2.

Verifies three Sprint 1/2 serving invariants in one command — either completely
offline on CPU (`--smoke-cpu`, `< 10s`, zero network/GPU) or against a real
checkpoint / exported Whole Model Pack on GPU (`--checkpoint ... --strict-tf5`):

  1. `QL60-001` (`check_quantized_hybrid_cache`):
     `QuantizedHybridCache` under real `transformers.cache_utils.Cache` /
     `CacheLayerMixin` (or strict `abc.ABC` TF 5.24 contract offline), across
     `modes=("bf16", "fp8", "hadamard_int4")`, both `kv_share=True` (`1.0x` KV
     RAM via `SharedKVView`) and `kv_share=False`, plus full cache contract
     methods and write-through per-layer state proxies.
  2. `QL60-002` (`check_pack_model_and_generate`):
     Both in-package `Qwen36LoopHSSDForCausalLM` and freshly emitted standalone
     `configuration_qwen36_loophssd.py` / `modeling_qwen36_loophssd.py`
     (Python 3.9 AST parse + `py_compile` + isolated `_HAS_QLOOP=False` import),
     `CausalLMPackOutput` (`ModelOutput` contract + `.cache` alias + item
     mutation), `logits_to_keep=0` returning full `(B, T, V)`, outer config
     `layer_types`/`num_hidden_layers` sync on `set_active_passes(d)`, 2D/3D
     MRoPE `position_ids` slicing in `prepare_inputs_for_generation`, token-for-
     token greedy decode parity vs uncached full recompute across `depths`, and
     `qloop_format_version` / `qloop_commit` metadata.
  3. `QL60-005` (`check_stage_completion_gate`):
     `_local_stage_done()` in `qloop.colab` rejecting mid-run `LATEST` and
     incomplete `PROMOTED` checkpoints (`step < target_steps`) and accepting
     verified completed stages (`step >= target_steps` or `COMPLETED` marker).

Python 3.9+ compatible. Emits structured JSON and Markdown audit reports.
"""
from __future__ import annotations

import abc
import argparse
import ast
import copy
import dataclasses
import datetime as _dt
import gc
import importlib
import importlib.util
import json
import os
import py_compile
import shutil
import sys
import tempfile
import time
import types
import warnings
from collections import OrderedDict
from collections.abc import Sequence
from pathlib import Path
from typing import Any, ClassVar

import torch
from torch import nn

# Allow direct invocation (`python scripts/verify_tf524_serving_gpu.py`) without
# requiring an editable install when `PYTHONPATH=src` is omitted.
_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC_DIR = _REPO_ROOT / "src"
if _SRC_DIR.is_dir() and str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from qloop.colab import _local_stage_done  # noqa: E402
from qloop.data.tokenizer import ToyTokenizer  # noqa: E402
from qloop.export_pack import (  # noqa: E402
    PACK_MANIFEST_FILE,
    SAFETENSORS_INDEX_FILE,
    LoadedModelPack,
    export_whole_model_pack,
    extract_arch_spec,
)
from qloop.infer.engine import QuantizedHybridCache  # noqa: E402
from qloop.infer.hf_custom_modeling import (  # noqa: E402
    CONFIG_FILENAME,
    MODELING_FILENAME,
    QLOOP_FORMAT_VERSION,
    CausalLMPackOutput,
    Qwen36LoopHSSDConfig,
    Qwen36LoopHSSDForCausalLM,
    resolve_pack_directory,
    resolve_qloop_commit,
    write_standalone_hf_modeling_files,
)
from qloop.model._cache_compat import cache_kwarg_name, extract_output_cache  # noqa: E402
from qloop.model.gdn2 import GDN2Config, install_gdn2  # noqa: E402
from qloop.model.introspect import find_text_model, inspect_model  # noqa: E402
from qloop.model.loop import LoopSpec, SharedKVView, apply_loop  # noqa: E402
from qloop.model.lsa import LSAConfig, install_lsa  # noqa: E402
from qloop.model.toy import ToyConfig, ToyForCausalLM, ToyHybridCache  # noqa: E402
from qloop.train.stack import Stack  # noqa: E402

DEFAULT_MODES: tuple[str, ...] = ("bf16", "fp8", "hadamard_int4")
DEFAULT_DEPTHS: tuple[int, ...] = (1, 2, 3)
DEFAULT_PROMPT = "1, 2, 3, 4, 5, 6, 7, 8, 9, 10,"
_SMALL_MODEL_PARAM_LIMIT = 50_000_000


# ============================================================================ environment & TF 5.24 helpers


def _utc_now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def detect_transformers_env() -> dict[str, Any]:
    """Detect installed `transformers` version and whether real TF >= 5 is active."""
    info: dict[str, Any] = {
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "torch_version": str(torch.__version__),
        "cuda_available": bool(torch.cuda.is_available()),
        "gpu_name": str(torch.cuda.get_device_name(0)) if torch.cuda.is_available() else None,
        "transformers_installed": False,
        "transformers_version": None,
        "transformers_major": None,
        "transformers_is_v5_plus": False,
        "has_cache_layer_mixin": False,
        "qloop_format_version": QLOOP_FORMAT_VERSION,
        "qloop_commit": resolve_qloop_commit(),
    }
    try:
        import transformers  # type: ignore[import-not-found]
        from transformers import cache_utils  # type: ignore[import-not-found]

        ver = str(getattr(transformers, "__version__", "0.0.0"))
        head = ver.split(".", 1)[0]
        major = int(head) if head.isdigit() else None
        info["transformers_installed"] = True
        info["transformers_version"] = ver
        info["transformers_major"] = major
        info["transformers_is_v5_plus"] = bool(major is not None and major >= 5)
        info["has_cache_layer_mixin"] = hasattr(cache_utils, "CacheLayerMixin")
    except Exception:
        pass
    return info


def _make_strict_tf524_mock_modules() -> dict[str, types.ModuleType]:
    """Build strict `abc.ABC` `transformers 5.24.x` contract modules for offline CPU verification."""
    tf_mod = types.ModuleType("transformers")
    tf_mod.__version__ = "5.24.1"  # type: ignore[attr-defined]
    cu_mod = types.ModuleType("transformers.cache_utils")
    mo_mod = types.ModuleType("transformers.modeling_outputs")

    class _StrictCacheLayerMixin(abc.ABC):
        """Strict TF 5.24+ `CacheLayerMixin(ABC)` requiring all abstract methods and read-only `is_croppable`."""

        is_compileable = False
        is_sliding = False

        def __init__(self) -> None:
            self.keys: torch.Tensor | None = None
            self.values: torch.Tensor | None = None
            self.is_initialized = False

        @property
        def is_croppable(self) -> bool:
            return False

        @abc.abstractmethod
        def lazy_initialization(self, key_states: torch.Tensor, value_states: torch.Tensor) -> None: ...

        @abc.abstractmethod
        def update(
            self,
            key_states: torch.Tensor,
            value_states: torch.Tensor,
            cache_kwargs: dict[str, Any] | None = None,
        ) -> tuple[torch.Tensor, torch.Tensor]: ...

        @abc.abstractmethod
        def get_mask_sizes(self, cache_position: torch.Tensor) -> tuple[int, int]: ...

        @abc.abstractmethod
        def get_seq_length(self) -> int: ...

        @abc.abstractmethod
        def get_max_cache_shape(self) -> int | None: ...

        @abc.abstractmethod
        def get_max_length(self) -> int: ...

    class _StrictTF524Cache:
        """Strict TF 5.24+ `Cache` requiring `layers` XOR `layer_class_to_replicate`."""

        init_calls: ClassVar[list[dict[str, Any]]] = []

        def __init__(self, layers: Any = None, layer_class_to_replicate: Any = None) -> None:
            type(self).init_calls.append({"layers": layers, "layer_class_to_replicate": layer_class_to_replicate})
            if (layers is None) == (layer_class_to_replicate is None):
                raise ValueError("You should provide exactly one of layers or layer_class_to_replicate")
            self.layers = list(layers) if layers is not None else []
            _ = self.is_sliding
            _ = self.is_compileable

        @property
        def is_compileable(self) -> bool:
            return bool(self.layers) and all(bool(getattr(lyr, "is_compileable", False)) for lyr in self.layers)

        @property
        def is_sliding(self) -> list[bool]:
            return [bool(getattr(lyr, "is_sliding", False)) for lyr in self.layers]

        def is_recording_past(self, layer_idx: int) -> bool:
            return bool(getattr(self.layers[layer_idx], "record_past", False))

    class _StrictDynamicLayer(_StrictCacheLayerMixin):
        supports_early_init = True

        @property
        def is_croppable(self) -> bool:
            return True

        def lazy_initialization(self, key_states: torch.Tensor, value_states: torch.Tensor) -> None:
            del key_states, value_states

        def update(
            self,
            key_states: torch.Tensor,
            value_states: torch.Tensor,
            cache_kwargs: dict[str, Any] | None = None,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            del cache_kwargs
            if self.keys is None:
                self.keys, self.values = key_states, value_states
            else:
                self.keys = torch.cat([self.keys, key_states], dim=-2)
                self.values = torch.cat([self.values, value_states], dim=-2)  # type: ignore[list-item]
            self.is_initialized = True
            return self.keys, self.values  # type: ignore[return-value]

        def get_mask_sizes(self, cache_position: torch.Tensor) -> tuple[int, int]:
            return self.get_seq_length() + int(cache_position.shape[0]), 0

        def get_seq_length(self) -> int:
            return 0 if self.keys is None else int(self.keys.shape[-2])

        def get_max_cache_shape(self) -> int | None:
            return None

        def get_max_length(self) -> int:
            return -1

    class _StrictLinearAttentionLayer(_StrictCacheLayerMixin):
        supports_early_init = False

        def __init__(self) -> None:
            super().__init__()
            self.conv_states: list[torch.Tensor | None] = [None]
            self.recurrent_states: list[Any] = [None]
            self.has_previous_state = {0: False}

        def lazy_initialization(self, key_states: torch.Tensor, value_states: torch.Tensor) -> None:
            del key_states, value_states

        def update(
            self,
            key_states: torch.Tensor,
            value_states: torch.Tensor,
            cache_kwargs: dict[str, Any] | None = None,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            del cache_kwargs
            return key_states, value_states

        def get_mask_sizes(self, cache_position: torch.Tensor) -> tuple[int, int]:
            return int(cache_position.shape[0]), 0

        def get_seq_length(self) -> int:
            return 0

        def get_max_cache_shape(self) -> int | None:
            return None

        def get_max_length(self) -> int:
            return -1

    class _StrictHybridDynamicCache(_StrictTF524Cache):
        """Strict TF 5.24 hybrid `DynamicCache(config)` driven by `config.layer_types`."""

        def __init__(self, config: Any) -> None:
            ltypes = getattr(config, "layer_types", None)
            n_layers = int(getattr(config, "num_hidden_layers", 0))
            if ltypes is None:
                ltypes = ["full_attention"] * n_layers
            built: list[Any] = [
                _StrictLinearAttentionLayer() if lt == "linear_attention" else _StrictDynamicLayer()
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
            if not isinstance(lyr, _StrictLinearAttentionLayer):
                raise ValueError(f"Cannot get conv state for non-linear layer {layer_idx}: {type(lyr).__name__}")
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
            if not isinstance(lyr, _StrictDynamicLayer):
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
            if isinstance(lyr, _StrictDynamicLayer):
                return lyr.get_seq_length()
            return int(self.layers[self.first_full].get_seq_length())

    class _StrictModelOutput(OrderedDict):  # type: ignore[type-arg]
        """Mirrors HF `ModelOutput` requiring `@dataclass` subclasses."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            if self.__class__ != _StrictModelOutput and not dataclasses.is_dataclass(self):
                raise TypeError(f"{self.__module__}.{self.__class__.__name__} has to be a dataclass.")

        def to_tuple(self) -> tuple[Any, ...]:
            return tuple(self[k] for k in self.keys())

    class _StrictPretrainedConfig:
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

    class _StrictGenerationMixin:
        """Faithful TF 5.24 greedy `GenerationMixin.generate()` replica for offline smoke tests."""

        @torch.no_grad()
        def generate(
            self,
            input_ids: torch.Tensor,
            *,
            max_new_tokens: int = 6,
            do_sample: bool = False,
            use_cache: bool = True,
            past_key_values: Any = None,
            logits_to_keep: int = 0,
            attention_mask: torch.Tensor | None = None,
            **kwargs: Any,
        ) -> torch.Tensor:
            if do_sample:
                raise ValueError("Offline mock generate() supports greedy decoding (do_sample=False) only")
            cur_ids = input_ids
            pkv = past_key_values
            if use_cache and pkv is None:
                pkv = _StrictHybridDynamicCache(self.config)  # type: ignore[attr-defined]
            if isinstance(pkv, _StrictHybridDynamicCache):
                for idx, lt in enumerate(getattr(self.config, "layer_types", [])):  # type: ignore[attr-defined]
                    if lt == "linear_attention":
                        pkv.get_conv_state(idx)

            cur_mask = attention_mask
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
                    attention_mask=cur_mask,
                    cache_position=cache_pos,
                    use_cache=use_cache,
                    logits_to_keep=logits_to_keep,
                    **kwargs,
                )
                outputs = self(**model_inputs)  # type: ignore[operator]
                if not hasattr(outputs, "get"):
                    raise AssertionError("HF generate() requires ModelOutput.get()")
                logits = outputs.get("logits")
                if logits is None or logits.shape[1] < 1:
                    raise AssertionError(f"empty logits shape: {getattr(logits, 'shape', None)}")
                pkv = outputs.get("past_key_values")
                nxt = logits[:, -1:, :].argmax(dim=-1)
                cur_ids = torch.cat([cur_ids, nxt], dim=1)
                if cur_mask is not None:
                    cur_mask = torch.cat([cur_mask, torch.ones_like(nxt)], dim=1)
            return cur_ids

    class _StrictPreTrainedModel(nn.Module):
        config_class: Any = None

        def __init__(self, config: Any) -> None:
            super().__init__()
            self.config = config

        def post_init(self) -> None:
            return None

    class _StrictAutoConfig:
        @staticmethod
        def for_model(model_type: str, **kw: Any) -> ToyConfig:
            del model_type
            valid = {f for f in ToyConfig.__dataclass_fields__ if f in kw}
            clean = {k: kw[k] for k in valid}
            if "layer_types" in clean and isinstance(clean["layer_types"], list):
                clean["layer_types"] = list(clean["layer_types"])
            return ToyConfig(**clean)

    class _StrictAutoModelForCausalLM:
        @staticmethod
        def from_config(cfg: ToyConfig, **kw: Any) -> ToyForCausalLM:
            del kw
            return ToyForCausalLM(cfg)

    cu_mod.Cache = _StrictTF524Cache  # type: ignore[attr-defined]
    cu_mod.CacheLayerMixin = _StrictCacheLayerMixin  # type: ignore[attr-defined]
    cu_mod.DynamicCache = _StrictHybridDynamicCache  # type: ignore[attr-defined]
    mo_mod.ModelOutput = _StrictModelOutput  # type: ignore[attr-defined]

    tf_mod.PretrainedConfig = _StrictPretrainedConfig  # type: ignore[attr-defined]
    tf_mod.PreTrainedConfig = _StrictPretrainedConfig  # type: ignore[attr-defined]
    tf_mod.PreTrainedModel = _StrictPreTrainedModel  # type: ignore[attr-defined]
    tf_mod.GenerationMixin = _StrictGenerationMixin  # type: ignore[attr-defined]
    tf_mod.AutoConfig = _StrictAutoConfig  # type: ignore[attr-defined]
    tf_mod.AutoModelForCausalLM = _StrictAutoModelForCausalLM  # type: ignore[attr-defined]
    tf_mod.cache_utils = cu_mod  # type: ignore[attr-defined]
    tf_mod.modeling_outputs = mo_mod  # type: ignore[attr-defined]

    return {
        "transformers": tf_mod,
        "transformers.cache_utils": cu_mod,
        "transformers.modeling_outputs": mo_mod,
    }


def _build_smoke_toy_stack(*, seed: int = 2026, kv_share: bool = True) -> Stack:
    """Build a compact 6L -> 12L (3-pass) hybrid ToyForCausalLM stack with GDN-2 + D-LSA + Loop."""
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed + 19)
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
            "linear_attention",  # 5 coda
        ],
    )
    model = ToyForCausalLM(tcfg).eval()

    gdn2 = install_gdn2(model, GDN2Config(rank=8, chunk_size=8))
    with torch.no_grad():
        for ad in gdn2.adapters.values():
            ad.erase_up.weight.normal_(0.0, 0.08, generator=gen)
            ad.write_up.weight.normal_(0.0, 0.08, generator=gen)
            ad.decay_up.weight.normal_(0.0, 0.05, generator=gen)

    # Keep window >= prompt_len + max_new_tokens so sparse D-LSA stays exact on short smoke decode
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
            window=32,
            sinks=4,
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
            kv_share=kv_share,
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


def _is_toy_model(obj: Any) -> bool:
    if isinstance(obj, LoadedModelPack) and obj.hf_model is not None:
        obj = obj.hf_model
    elif isinstance(obj, Stack):
        obj = obj.model

    cfg = getattr(obj, "config", None)
    if cfg is not None:
        arch = getattr(cfg, "qloop_v3_arch", None)
        if isinstance(arch, dict):
            if str(arch.get("backbone_kind") or "").lower() == "toy":
                return True
            if str(arch.get("base_model_type") or "").lower().startswith("toy"):
                return True
        bcfg = getattr(cfg, "base_config", None)
        if isinstance(bcfg, dict) and str(bcfg.get("model_type") or "").lower().startswith("toy"):
            return True
        mtype = str(getattr(cfg, "model_type", "") or "").lower()
        if mtype.startswith("toy"):
            return True

    backbone = getattr(obj, "backbone", None)
    if isinstance(backbone, nn.Module) and backbone is not obj:
        return _is_toy_model(backbone)

    if isinstance(obj, ToyForCausalLM):
        return True
    return type(obj).__name__.startswith("Toy")


def _model_param_count(mod: nn.Module) -> int:
    seen: set[int] = set()
    total = 0
    for p in mod.parameters():
        if id(p) in seen:
            continue
        seen.add(id(p))
        total += int(p.numel())
    return total


def _model_device(mod: nn.Module) -> torch.device:
    for p in mod.parameters():
        if not p.is_meta:
            return p.device
    return torch.device("cpu")


def _wrap_stack_as_hf_pack_model(stack: Stack) -> Qwen36LoopHSSDForCausalLM:
    """Zero-copy wrap a live `Stack` inside `Qwen36LoopHSSDForCausalLM` without duplicating weights."""
    if isinstance(stack, LoadedModelPack) and stack.hf_model is not None:
        return stack.hf_model
    if isinstance(stack.model, Qwen36LoopHSSDForCausalLM):
        return stack.model

    base_cfg, arch = extract_arch_spec(stack, merged_hot_experts=False)
    tok = stack.tokenizer
    for key in ("eos_token_id", "pad_token_id", "bos_token_id"):
        if base_cfg.get(key) is None and tok is not None and getattr(tok, key, None) is not None:
            val = getattr(tok, key)
            if isinstance(val, (int, list)):
                base_cfg[key] = val

    hf_cfg = Qwen36LoopHSSDConfig(
        base_config=base_cfg,
        qloop_v3_arch=arch,
        qloop_format_version=QLOOP_FORMAT_VERSION,
        qloop_commit=resolve_qloop_commit(),
    )
    inst = object.__new__(Qwen36LoopHSSDForCausalLM)
    # Initialize real HF PreTrainedModel / GenerationMixin (or offline nn.Module fallback) cleanly.
    super(Qwen36LoopHSSDForCausalLM, inst).__init__(hf_cfg)
    inst.config = hf_cfg
    inst.backbone = stack.model
    inst.gdn2_manager = stack.gdn2
    inst.hssd_handle = stack.hssd
    inst.lsa_manager = stack.lsa
    inst.hot_deltas = stack.deltas
    inst.loop_controller = stack.loop
    inst._sync_outer_config()
    inst.eval()
    return inst


def _extract_cache_geometry(model_or_cfg: Any) -> dict[str, Any]:
    """Extract virtual layer schedule and tensor shapes from a model, Stack, config, or dict."""
    model_obj: nn.Module | None = None
    cfg_obj: Any = None
    if isinstance(model_or_cfg, LoadedModelPack):
        model_obj = model_or_cfg.hf_model or model_or_cfg.model
        cfg_obj = model_or_cfg.pack_config or getattr(model_obj, "config", None)
    elif isinstance(model_or_cfg, Stack):
        model_obj = model_or_cfg.model
        cfg_obj = getattr(find_text_model(model_or_cfg.model), "config", None)
    elif isinstance(model_or_cfg, Qwen36LoopHSSDForCausalLM):
        model_obj = model_or_cfg
        cfg_obj = model_or_cfg.config
    elif isinstance(model_or_cfg, nn.Module):
        model_obj = model_or_cfg
        try:
            cfg_obj = getattr(find_text_model(model_or_cfg), "config", None)
        except Exception:
            cfg_obj = getattr(model_or_cfg, "config", None)
    else:
        cfg_obj = model_or_cfg

    def _get(key: str, default: Any = None) -> Any:
        if isinstance(cfg_obj, dict):
            val = cfg_obj.get(key)
            if val is None and isinstance(cfg_obj.get("base_config"), dict):
                val = cfg_obj["base_config"].get(key)
            return default if val is None else val
        val = getattr(cfg_obj, key, None)
        if val is None and isinstance(getattr(cfg_obj, "base_config", None), dict):
            val = cfg_obj.base_config.get(key)
        return default if val is None else val

    raw_lt = _get("layer_types")
    if isinstance(raw_lt, (list, tuple)) and len(raw_lt) > 0:
        ltypes = [str(x) for x in raw_lt]
    else:
        n_h = int(_get("num_hidden_layers", 4) or 4)
        interval = int(_get("full_attention_interval", 4) or 4)
        if interval > 0 and n_h >= 2:
            ltypes = ["full_attention" if (i + 1) % interval == 0 else "linear_attention" for i in range(n_h)]
        else:
            ltypes = ["linear_attention", "full_attention", "linear_attention", "full_attention"]

    # Ensure the contract probe has >= 2 linear and >= 2 full layers (keeps real schedules verbatim).
    probe_ltypes = list(ltypes)
    if probe_ltypes.count("linear_attention") < 2 or probe_ltypes.count("full_attention") < 2:
        probe_ltypes = ["linear_attention", "full_attention", "linear_attention", "full_attention"]

    num_kv_heads = int(_get("num_key_value_heads", 2) or 2)
    head_dim = int(_get("head_dim", 16) or 16)
    lin_heads = int(_get("linear_num_heads", _get("linear_num_value_heads", 4)) or 4)
    lin_dim = int(_get("linear_head_dim", _get("linear_value_head_dim", 8)) or 8)
    conv_k = int(_get("linear_conv_kernel_dim", 4) or 4)

    # Cap synthetic tensor test dims so contract unit tests stay fast even for 35B configs (head_dim=256).
    test_kv_heads = min(max(1, num_kv_heads), 4)
    test_head_dim = min(max(8, head_dim), 64)
    test_lin_heads = min(max(1, lin_heads), 4)
    test_lin_dim = min(max(4, lin_dim), 16)
    test_conv_c = max(8, test_lin_heads * test_lin_dim)
    test_conv_k = max(2, min(conv_k, 8))

    return {
        "model_obj": model_obj,
        "layer_types": ltypes,
        "probe_layer_types": probe_ltypes,
        "num_layers": len(ltypes),
        "probe_num_layers": len(probe_ltypes),
        "test_kv_heads": test_kv_heads,
        "test_head_dim": test_head_dim,
        "test_lin_heads": test_lin_heads,
        "test_lin_dim": test_lin_dim,
        "test_conv_c": test_conv_c,
        "test_conv_k": test_conv_k,
    }


def _probe_quantized_cache_strict_abc(num_layers: int, layer_types: list[str]) -> dict[str, Any]:
    """Verify `QuantizedHybridCache` / `_QuantizedCacheLayer` against strict `abc.ABC` TF 5.24 bases.

    Loads `qloop/infer/engine.py` under an isolated private module name so the live
    `qloop.infer.engine` module and its class identities are never mutated.
    """
    import qloop.infer.engine as live_engine

    mock_mods = _make_strict_tf524_mock_modules()
    saved: dict[str, Any] = {name: sys.modules.get(name) for name in mock_mods}
    probe_name = "qloop.infer._tf524_audit_engine_probe"
    try:
        for name, mod in mock_mods.items():
            sys.modules[name] = mod
        spec = importlib.util.spec_from_file_location(probe_name, live_engine.__file__)
        if spec is None or spec.loader is None:
            raise RuntimeError("failed to create module spec for qloop.infer.engine")
        probe_mod = importlib.util.module_from_spec(spec)
        sys.modules[probe_name] = probe_mod
        spec.loader.exec_module(probe_mod)

        cache_cls = mock_mods["transformers.cache_utils"].Cache
        layer_cls = mock_mods["transformers.cache_utils"].CacheLayerMixin
        cache_cls.init_calls.clear()

        qc = probe_mod.QuantizedHybridCache(num_layers=num_layers, layer_types=layer_types, mode="fp8")
        if not isinstance(qc, cache_cls):
            raise AssertionError("QuantizedHybridCache does not inherit strict TF 5.24 Cache")
        if len(cache_cls.init_calls) != 1 or len(cache_cls.init_calls[0]["layers"]) != num_layers:
            raise AssertionError("QuantizedHybridCache did not pass layers=... to Cache.__init__")
        if len(qc.layers) != num_layers or not all(isinstance(lyr, layer_cls) for lyr in qc.layers):
            raise AssertionError("QuantizedHybridCache.layers do not inherit strict abc.ABC CacheLayerMixin")

        lin_idx = layer_types.index("linear_attention")
        full_idx = layer_types.index("full_attention")
        if qc.layers[lin_idx].is_croppable is not False or qc.layers[full_idx].is_croppable is not True:
            raise AssertionError("unexpected is_croppable on CacheLayerMixin views")
        if qc.layers[lin_idx].supports_early_init is not False or qc.layers[full_idx].supports_early_init is not True:
            raise AssertionError("unexpected supports_early_init on CacheLayerMixin views")
        if qc.layers[full_idx].get_max_length() != -1:
            raise AssertionError("unexpected get_max_length() value")
        qc.layers[full_idx].lazy_initialization(torch.randn(1, 2, 1, 16), torch.randn(1, 2, 1, 16))

        if qc.is_compileable is not False:
            raise AssertionError(f"expected is_compileable=False, got {qc.is_compileable!r}")
        if qc.is_sliding != [False] * num_layers:
            raise AssertionError(f"unexpected is_sliding: {qc.is_sliding!r}")
        if qc.is_recording_past(lin_idx) is not False or qc.is_recording_past(full_idx) is not False:
            raise AssertionError("expected is_recording_past() == False")

        return {
            "contract_mode": "strict_abc_mock",
            "cache_base": cache_cls.__name__,
            "layer_base": layer_cls.__name__,
            "layers_verified": num_layers,
        }
    finally:
        sys.modules.pop(probe_name, None)
        for name, prev in saved.items():
            if prev is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = prev


def _verify_live_model_kv_modes(
    model_obj: Any,
    modes: tuple[str, ...],
) -> dict[str, Any]:
    """Run a short prefill + decode forward pass on a live model using `QuantizedHybridCache`."""
    if isinstance(model_obj, Qwen36LoopHSSDForCausalLM):
        backbone = model_obj.backbone
        loop_ctl = model_obj.loop_controller
    else:
        backbone = model_obj
        tm_tmp = find_text_model(backbone)
        layers_tmp = getattr(tm_tmp, "layers", None)
        loop_ctl = (
            getattr(layers_tmp[0], "_ctl", None)
            if isinstance(layers_tmp, nn.ModuleList) and layers_tmp
            else None
        )

    tm = find_text_model(backbone)
    dev = _model_device(backbone)
    n_virt = int(getattr(tm.config, "num_hidden_layers", len(getattr(tm, "layers", []))))
    ltypes = list(getattr(tm.config, "layer_types", ["full_attention"] * n_virt))
    vocab = int(getattr(tm.config, "vocab_size", 64) or 64)
    prompt = torch.tensor([[min(7 + i * 5, vocab - 1) for i in range(6)]], dtype=torch.long, device=dev)
    next_tok = torch.tensor([[min(13, vocab - 1)]], dtype=torch.long, device=dev)

    kv_share_flag = bool(loop_ctl.spec.kv_share) if loop_ctl is not None else False
    max_r = int(loop_ctl.spec.num_passes) if loop_ctl is not None else 1
    orig_r = int(loop_ctl.active_passes) if loop_ctl is not None else 1
    test_depths = (1, max_r) if max_r > 1 else (1,)
    kw_name = cache_kwarg_name(tm)

    per_mode: dict[str, Any] = {}
    try:
        with torch.no_grad():
            for mode in modes:
                depth_kv_bytes: dict[str, int] = {}
                for d in test_depths:
                    if loop_ctl is not None:
                        loop_ctl.set_active_passes(d)
                    qcache = QuantizedHybridCache(num_layers=n_virt, layer_types=ltypes, mode=mode)
                    out_p = tm(input_ids=prompt, **{kw_name: qcache, "use_cache": True})
                    c_out = extract_output_cache(out_p) or qcache
                    if c_out.get_seq_length() != int(prompt.shape[1]):
                        raise AssertionError(
                            f"live model prefill seq_len mismatch for mode={mode}, depth={d}: "
                            f"got {c_out.get_seq_length()}, expected {prompt.shape[1]}"
                        )
                    out_d = tm(input_ids=next_tok, **{kw_name: c_out, "use_cache": True})
                    h_last = out_d.last_hidden_state
                    if not bool(torch.isfinite(h_last.float()).all().item()):
                        raise AssertionError(f"non-finite hidden state for mode={mode}, depth={d}")
                    if c_out.get_seq_length() != int(prompt.shape[1]) + 1:
                        raise AssertionError(
                            f"live model decode seq_len mismatch for mode={mode}, depth={d}: "
                            f"got {c_out.get_seq_length()}, expected {prompt.shape[1] + 1}"
                        )
                    depth_kv_bytes[f"d{d}"] = int(qcache.kv_bytes())
                    del out_p, out_d, c_out, qcache
                    if dev.type == "cuda":
                        torch.cuda.empty_cache()

                if kv_share_flag and len(test_depths) == 2:
                    b1 = depth_kv_bytes[f"d{test_depths[0]}"]
                    b2 = depth_kv_bytes[f"d{test_depths[1]}"]
                    if b1 <= 0 or b1 != b2:
                        raise AssertionError(
                            f"live model kv_share=True failed 1.0x KV RAM invariant for mode={mode}: "
                            f"d{test_depths[0]}={b1} vs d{test_depths[1]}={b2}"
                        )
                per_mode[mode] = {
                    "kv_bytes_by_depth": depth_kv_bytes,
                    "kv_share_1x_verified": bool(kv_share_flag and len(test_depths) == 2),
                }
    finally:
        if loop_ctl is not None:
            loop_ctl.set_active_passes(orig_r)

    return {
        "device": str(dev),
        "kv_share": kv_share_flag,
        "depths_tested": list(test_depths),
        "per_mode": per_mode,
    }


# ============================================================================ 1. QL60-001


def check_quantized_hybrid_cache(
    model_or_cfg: Any,
    modes: Sequence[str] = DEFAULT_MODES,
) -> dict[str, Any]:
    """`QL60-001`: Verify `QuantizedHybridCache` TF 5.24 contract, `kv_share=True/False`, and all methods."""
    name = "QL60-001:quantized_hybrid_cache"
    mode_list = tuple(str(m) for m in modes)
    t0 = time.perf_counter()
    metrics: dict[str, Any] = {
        "modes": list(mode_list),
        "subchecks": {},
        "per_mode": {},
    }

    try:
        if not mode_list:
            raise ValueError("modes must contain at least one KV cache mode")

        env = detect_transformers_env()
        geom = _extract_cache_geometry(model_or_cfg)
        ltypes = geom["probe_layer_types"]
        n_layers = geom["probe_num_layers"]
        lin_indices = [i for i, lt in enumerate(ltypes) if lt == "linear_attention"]
        full_indices = [i for i, lt in enumerate(ltypes) if lt == "full_attention"]
        lin0, lin_last = lin_indices[0], lin_indices[-1]
        full0, full1 = full_indices[0], full_indices[1]

        # 1) TF 5.24 `Cache` / `CacheLayerMixin` inheritance verification
        if env["transformers_installed"]:
            from transformers.cache_utils import Cache as RealHFCache  # type: ignore[import-not-found]
            from transformers.cache_utils import (
                CacheLayerMixin as RealHFCacheLayerMixin,  # type: ignore[import-not-found]
            )

            probe_cache = QuantizedHybridCache(num_layers=n_layers, layer_types=ltypes, mode=mode_list[0])
            if not isinstance(probe_cache, RealHFCache):
                raise AssertionError(
                    f"QuantizedHybridCache is not a subclass of real transformers.cache_utils.Cache "
                    f"(version {env['transformers_version']})"
                )
            if len(probe_cache.layers) != n_layers or not all(
                isinstance(lyr, RealHFCacheLayerMixin) for lyr in probe_cache.layers
            ):
                raise AssertionError("QuantizedHybridCache.layers do not inherit real CacheLayerMixin")
            if probe_cache.layers[lin0].is_croppable is not False or probe_cache.layers[full0].is_croppable is not True:
                raise AssertionError("unexpected is_croppable on real-transformers CacheLayerMixin views")
            if (
                probe_cache.layers[lin0].supports_early_init is not False
                or probe_cache.layers[full0].supports_early_init is not True
            ):
                raise AssertionError("unexpected supports_early_init on real-transformers CacheLayerMixin views")
            if probe_cache.layers[full0].get_max_length() != -1:
                raise AssertionError("unexpected get_max_length() on real-transformers CacheLayerMixin view")
            probe_cache.layers[full0].lazy_initialization(torch.randn(1, 2, 1, 16), torch.randn(1, 2, 1, 16))
            if probe_cache.is_compileable is not False or probe_cache.is_sliding != [False] * n_layers:
                raise AssertionError("unexpected is_compileable / is_sliding on real-transformers QuantizedHybridCache")
            if probe_cache.is_recording_past(lin0) is not False or probe_cache.is_recording_past(full0) is not False:
                raise AssertionError("unexpected is_recording_past() on real-transformers QuantizedHybridCache")
            metrics["tf524_contract"] = {
                "contract_mode": "real_transformers",
                "transformers_version": env["transformers_version"],
                "layers_verified": n_layers,
            }
        else:
            metrics["tf524_contract"] = _probe_quantized_cache_strict_abc(n_layers, ltypes)
        metrics["subchecks"]["tf524_cache_and_layer_mixin"] = True

        # Check @property descriptors on class
        if not isinstance(getattr(QuantizedHybridCache, "is_compileable", None), property):
            raise AssertionError("QuantizedHybridCache.is_compileable must be a @property")
        if not isinstance(getattr(QuantizedHybridCache, "is_sliding", None), property):
            raise AssertionError("QuantizedHybridCache.is_sliding must be a @property")
        metrics["subchecks"]["property_descriptors"] = True

        # 2) Full contract + kv_share=True (SharedKVView) + kv_share=False across all modes
        b_sz = 2
        g_kv = geom["test_kv_heads"]
        h_dim = geom["test_head_dim"]
        conv_c = geom["test_conv_c"]
        conv_k = geom["test_conv_k"]
        lin_h = geom["test_lin_heads"]
        lin_d = geom["test_lin_dim"]

        torch.manual_seed(17)
        k_prefill = torch.randn(b_sz, g_kv, 6, h_dim, dtype=torch.float32)
        v_prefill = torch.randn(b_sz, g_kv, 6, h_dim, dtype=torch.float32)
        k_dec = torch.randn(b_sz, g_kv, 1, h_dim, dtype=torch.float32)
        v_dec = torch.randn(b_sz, g_kv, 1, h_dim, dtype=torch.float32)
        k_p2_alt = torch.randn(b_sz, g_kv, 6, h_dim, dtype=torch.float32)
        v_p2_alt = torch.randn(b_sz, g_kv, 6, h_dim, dtype=torch.float32)
        conv_init = torch.randn(b_sz, conv_c, conv_k, dtype=torch.float32)
        conv_step = torch.randn(b_sz, conv_c, 1, dtype=torch.float32)
        rec_init = torch.randn(b_sz, lin_h, lin_d, lin_d, dtype=torch.float32)

        for mode in mode_list:
            # --- 2a) Core contract methods & properties ---
            cache = QuantizedHybridCache(num_layers=n_layers, layer_types=ltypes, mode=mode)
            if len(cache) != n_layers:
                raise AssertionError(f"mode={mode}: len(cache)={len(cache)} != {n_layers}")
            if cache.is_compileable is not False:
                raise AssertionError(f"mode={mode}: is_compileable={cache.is_compileable!r}")
            if cache.is_sliding != [False] * n_layers:
                raise AssertionError(f"mode={mode}: is_sliding mismatch")
            if cache.is_recording_past(lin0) is not False or cache.is_recording_past(full0) is not False:
                raise AssertionError(f"mode={mode}: is_recording_past returned non-False")
            if cache.get_seq_length() != 0 or cache.get_seq_length(lin0) != 0:
                raise AssertionError(f"mode={mode}: initial seq_length != 0")
            if cache.get_mask_sizes(torch.arange(6), layer_idx=full0) != (6, 0):
                raise AssertionError(f"mode={mode}: initial get_mask_sizes mismatch")
            if cache.has_previous_state(lin0) is not False or bool(cache.has_previous_state) is not False:
                raise AssertionError(f"mode={mode}: initial has_previous_state must be False")

            # Write-through on layers[i].conv_states[0] and layers[i].recurrent_states[0]
            cache.layers[lin0].conv_states[0] = conv_init
            cache.layers[lin0].recurrent_states[0] = rec_init
            if cache.conv_states[lin0] is not conv_init or cache.layers[lin0].conv_states != [conv_init]:
                raise AssertionError(f"mode={mode}: layers[{lin0}].conv_states[0] write-through failed")
            if cache.recurrent_states[lin0] is not rec_init or cache.layers[lin0].recurrent_states != [rec_init]:
                raise AssertionError(f"mode={mode}: layers[{lin0}].recurrent_states[0] write-through failed")
            if not cache.has_previous_state(lin0) or not cache.layers[lin0].has_previous_state[0]:
                raise AssertionError(f"mode={mode}: has_previous_state({lin0}) False after write-through")
            if cache.has_previous_state(lin_last) is not False or bool(cache.has_previous_state) is not False:
                raise AssertionError(f"mode={mode}: last linear layer should still be False before write")

            full_conv = cache.update_conv_state(conv_step, lin0, 0, conv_k)
            expected_conv = torch.cat([conv_init, conv_step], dim=-1)[..., -conv_k:]
            if full_conv.shape != (b_sz, conv_c, conv_k + 1) or not torch.equal(cache.conv_states[lin0], expected_conv):  # type: ignore[arg-type]
                raise AssertionError(f"mode={mode}: update_conv_state rolling window mismatch")

            cache.layers[lin_last].conv_states[0] = conv_init * 0.5
            cache.layers[lin_last].recurrent_states[0] = rec_init * 0.5
            if not bool(cache.has_previous_state):
                raise AssertionError(f"mode={mode}: bool(has_previous_state) False after populating last linear layer")

            # update (prefill + decode) on full layers
            k_pf, v_pf = cache.update(k_prefill, v_prefill, layer_idx=full0)
            if k_pf.shape != k_prefill.shape or v_pf.shape != v_prefill.shape:
                raise AssertionError(f"mode={mode}: prefill KV shape mismatch")
            k_f1, v_f1 = cache.layers[full1].update(k_prefill, v_prefill)
            if k_f1.shape != k_prefill.shape or v_f1.shape != v_prefill.shape:
                raise AssertionError(f"mode={mode}: layers[{full1}].update shape mismatch")

            k_all, v_all = cache.update(k_dec, v_dec, layer_idx=full0)
            if k_all.shape != (b_sz, g_kv, 7, h_dim) or v_all.shape != (b_sz, g_kv, 7, h_dim):
                raise AssertionError(f"mode={mode}: decode append KV shape mismatch")
            if (
                cache.get_seq_length() != 7
                or cache.get_seq_length(lin0) != 7
                or cache.layers[full0].get_seq_length() != 7
            ):
                raise AssertionError(f"mode={mode}: get_seq_length after decode != 7")
            if cache.get_mask_sizes(torch.tensor([7]), layer_idx=full0) != (8, 0):
                raise AssertionError(f"mode={mode}: get_mask_sizes after decode != (8, 0)")
            if cache.layers[full0].get_mask_sizes(torch.tensor([7])) != (8, 0):
                raise AssertionError(f"mode={mode}: layer.get_mask_sizes after decode != (8, 0)")

            # gather_kv exact parity with full dequantized slice
            sel_idx = torch.tensor([0, 3, 6], dtype=torch.long)
            gk, gv = cache.gather_kv(full0, sel_idx, dtype=torch.float32)
            if gk.shape != (b_sz, g_kv, 3, h_dim) or gv.shape != (b_sz, g_kv, 3, h_dim):
                raise AssertionError(f"mode={mode}: gather_kv shape mismatch")
            if not torch.equal(gk, k_all.index_select(-2, sel_idx)) or not torch.equal(
                gv, v_all.index_select(-2, sel_idx)
            ):
                raise AssertionError(f"mode={mode}: gather_kv values diverged from update() slice")

            # reorder_cache
            beam = torch.tensor([1, 0], dtype=torch.long)
            cache.reorder_cache(beam)
            k_reord, v_reord = cache.update(k_dec[..., :0, :], v_dec[..., :0, :], layer_idx=full0)
            if not torch.allclose(k_reord, k_all[beam], atol=1e-5) or not torch.allclose(
                v_reord, v_all[beam], atol=1e-5
            ):
                raise AssertionError(f"mode={mode}: reorder_cache KV mismatch")
            if not torch.equal(cache.conv_states[lin0], expected_conv[beam]):  # type: ignore[arg-type]
                raise AssertionError(f"mode={mode}: reorder_cache conv_states mismatch")
            if not torch.equal(cache.recurrent_states[lin0], rec_init[beam]):
                raise AssertionError(f"mode={mode}: reorder_cache recurrent_states mismatch")

            # batch_repeat_interleave
            cache.batch_repeat_interleave(2)
            if cache.layers[full0].keys is None or cache.layers[full0].keys.shape[0] != b_sz * 2:
                raise AssertionError(f"mode={mode}: batch_repeat_interleave KV batch dim mismatch")
            if cache.conv_states[lin0] is None or cache.conv_states[lin0].shape[0] != b_sz * 2:
                raise AssertionError(f"mode={mode}: batch_repeat_interleave conv batch dim mismatch")
            if cache.recurrent_states[lin0] is None or cache.recurrent_states[lin0].shape[0] != b_sz * 2:
                raise AssertionError(f"mode={mode}: batch_repeat_interleave recurrent batch dim mismatch")

            # crop (absolute + negative relative)
            cache.crop(4)
            if cache.get_seq_length(full0) != 4:
                raise AssertionError(f"mode={mode}: crop(4) failed, got {cache.get_seq_length(full0)}")
            cache.crop(-1)
            if cache.get_seq_length(full0) != 3:
                raise AssertionError(f"mode={mode}: crop(-1) failed, got {cache.get_seq_length(full0)}")

            # reset
            cache.reset()
            if (
                cache.get_seq_length() != 0
                or cache.is_initialized is not False
                or bool(cache.has_previous_state) is not False
                or cache.kv_bytes() != 0
                or cache.recurrent_bytes() != 0
            ):
                raise AssertionError(f"mode={mode}: reset() did not clear all cache state")

            # --- 2b) kv_share=True (1.0x KV RAM via SharedKVView) ---
            c_shared = QuantizedHybridCache(num_layers=n_layers, layer_types=ltypes, mode=mode)
            stash: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
            view_rec = SharedKVView(c_shared, stash, record=True)
            view_read_stashed = SharedKVView(c_shared, stash, record=False)
            view_read_zero_len = SharedKVView(c_shared, None, record=False)

            k_s1, v_s1 = view_rec.update(k_prefill, v_prefill, full0)
            bytes_p1 = int(c_shared.kv_bytes())
            if bytes_p1 <= 0:
                raise AssertionError(f"mode={mode}: kv_bytes after pass-1 write must be > 0")

            k_s2_st, v_s2_st = view_read_stashed.update(k_p2_alt, v_p2_alt, full0)
            k_s2_zl, v_s2_zl = view_read_zero_len.update(k_p2_alt, v_p2_alt, full0)
            bytes_p2_shared = int(c_shared.kv_bytes())
            if bytes_p2_shared != bytes_p1:
                raise AssertionError(
                    f"mode={mode}: kv_share=True grew KV bytes from {bytes_p1} to {bytes_p2_shared}"
                )
            if c_shared.get_seq_length(full0) != int(k_prefill.shape[-2]):
                raise AssertionError(f"mode={mode}: kv_share=True altered seq_length on pass-2 read")
            if not torch.equal(k_s2_st, k_s1) or not torch.equal(v_s2_st, v_s1):
                raise AssertionError(f"mode={mode}: SharedKVView(stash) returned mismatched pass-1 K/V")
            if not torch.equal(k_s2_zl, k_s1) or not torch.equal(v_s2_zl, v_s1):
                raise AssertionError(f"mode={mode}: SharedKVView(zero-len) returned mismatched pass-1 K/V")

            # Decode append under SharedKVView
            k_s1_dec, v_s1_dec = view_rec.update(k_dec, v_dec, full0)
            bytes_dec1 = int(c_shared.kv_bytes())
            k_s2_dec, v_s2_dec = view_read_zero_len.update(k_dec, v_dec, full0)
            if int(c_shared.kv_bytes()) != bytes_dec1 or c_shared.get_seq_length(full0) != 7:
                raise AssertionError(f"mode={mode}: SharedKVView decode step violated 1.0x KV invariant")
            if not torch.equal(k_s2_dec, k_s1_dec) or not torch.equal(v_s2_dec, v_s1_dec):
                raise AssertionError(f"mode={mode}: SharedKVView decode read mismatch")

            # --- 2c) kv_share=False (independent per-pass KV slots) ---
            c_unshared = QuantizedHybridCache(num_layers=n_layers, layer_types=ltypes, mode=mode)
            k_u1, _ = c_unshared.update(k_prefill, v_prefill, layer_idx=full0)
            bytes_u1 = int(c_unshared.kv_bytes())
            k_u2, _ = c_unshared.update(k_p2_alt, v_p2_alt, layer_idx=full1)
            bytes_u2 = int(c_unshared.kv_bytes())
            if bytes_u2 <= bytes_u1:
                raise AssertionError(
                    f"mode={mode}: kv_share=False did not allocate independent pass-2 KV slot "
                    f"({bytes_u1} -> {bytes_u2})"
                )
            if c_unshared.get_seq_length(full0) != 6 or c_unshared.get_seq_length(full1) != 6:
                raise AssertionError(f"mode={mode}: kv_share=False slot lengths mismatch")
            if torch.allclose(k_u1, k_u2):
                raise AssertionError(f"mode={mode}: kv_share=False aliased pass-1 and pass-2 slots")

            metrics["per_mode"][mode] = {
                "kv_bytes_shared_p1": bytes_p1,
                "kv_bytes_shared_p2": bytes_p2_shared,
                "kv_share_true_ratio": float(bytes_p2_shared) / float(bytes_p1),
                "kv_bytes_unshared_p1": bytes_u1,
                "kv_bytes_unshared_p2": bytes_u2,
                "kv_share_false_ratio": float(bytes_u2) / float(bytes_u1),
            }

        # Verify relative compression ordering if bf16 + quantized modes are present
        if "bf16" in metrics["per_mode"]:
            bf16_b = metrics["per_mode"]["bf16"]["kv_bytes_shared_p1"]
            if "fp8" in metrics["per_mode"]:
                fp8_b = metrics["per_mode"]["fp8"]["kv_bytes_shared_p1"]
                metrics["fp8_vs_bf16_ratio"] = round(fp8_b / float(bf16_b), 4)
                if fp8_b >= bf16_b:
                    raise AssertionError(f"fp8 KV bytes ({fp8_b}) not smaller than bf16 ({bf16_b})")
            if "hadamard_int4" in metrics["per_mode"]:
                i4_b = metrics["per_mode"]["hadamard_int4"]["kv_bytes_shared_p1"]
                metrics["hadamard_int4_vs_bf16_ratio"] = round(i4_b / float(bf16_b), 4)
                if i4_b >= bf16_b:
                    raise AssertionError(f"hadamard_int4 KV bytes ({i4_b}) not smaller than bf16 ({bf16_b})")

        metrics["subchecks"]["kv_share_true_and_false"] = True
        metrics["subchecks"]["contract_methods_and_proxies"] = True

        # 3) Live model forward check (if a model / Stack was supplied)
        if geom["model_obj"] is not None:
            metrics["live_model"] = _verify_live_model_kv_modes(geom["model_obj"], mode_list)
            metrics["subchecks"]["live_model_forward"] = True

            # Also verify model-level kv_share=False scaling when running on toy models
            if _is_toy_model(geom["model_obj"]):
                unshared_stack = _build_smoke_toy_stack(seed=2031, kv_share=False)
                metrics["toy_kv_share_false_model"] = _verify_live_model_kv_modes(unshared_stack.model, mode_list)
                for mode in mode_list:
                    d_map = metrics["toy_kv_share_false_model"]["per_mode"][mode]["kv_bytes_by_depth"]
                    if d_map["d3"] <= d_map["d1"]:
                        raise AssertionError(
                            f"toy kv_share=False model did not grow KV bytes across depths for mode={mode}: {d_map}"
                        )
                metrics["subchecks"]["toy_kv_share_false_model_forward"] = True

        metrics["elapsed_s"] = round(time.perf_counter() - t0, 4)
        return {
            "name": name,
            "passed": True,
            "detail": (
                f"QuantizedHybridCache passed all TF 5.24 contract checks across modes={list(mode_list)} "
                f"({metrics['tf524_contract']['contract_mode']}), kv_share=True (1.0x) and kv_share=False."
            ),
            "metrics": metrics,
        }
    except Exception as exc:
        metrics["elapsed_s"] = round(time.perf_counter() - t0, 4)
        return {
            "name": name,
            "passed": False,
            "detail": f"{type(exc).__name__}: {exc}",
            "metrics": metrics,
        }


# ============================================================================ 2. QL60-002


def _verify_model_output_contract(
    output_cls: type[Any],
    expected_base_cls: type[Any] | None,
    *,
    label: str,
) -> dict[str, Any]:
    """Verify `CausalLMPackOutput` satisfies HF `ModelOutput`, dict, tuple, `.cache`, and mutation APIs."""
    logits = torch.randn(1, 4, 16)
    loss = torch.tensor(1.25)
    dummy_cache = {"seq": 4}

    out = output_cls(logits=logits, past_key_values=dummy_cache)
    if not isinstance(out, OrderedDict):
        raise AssertionError(f"{label}: CausalLMPackOutput is not an OrderedDict")
    if expected_base_cls is not None and not isinstance(out, expected_base_cls):
        raise AssertionError(f"{label}: CausalLMPackOutput does not inherit {expected_base_cls.__name__}")
    if not dataclasses.is_dataclass(out):
        raise AssertionError(f"{label}: CausalLMPackOutput must be a dataclass")

    if out.logits is not logits or out.past_key_values is not dummy_cache or out.cache is not dummy_cache:
        raise AssertionError(f"{label}: attribute / .cache alias mismatch")
    if (
        out.get("logits") is not logits
        or out.get("past_key_values") is not dummy_cache
        or out.get("cache") is not dummy_cache
        or out.get("loss") is not None
        or out.get("missing", "fallback") != "fallback"
    ):
        raise AssertionError(f"{label}: .get() contract mismatch")
    if out["logits"] is not logits or out["cache"] is not dummy_cache:
        raise AssertionError(f"{label}: __getitem__ string access mismatch")
    if out[0] is not logits or out[1] is not dummy_cache:
        raise AssertionError(f"{label}: __getitem__ integer/tuple access mismatch")
    if list(out.keys()) != ["logits", "past_key_values"] or out.to_tuple() != (logits, dummy_cache):
        raise AssertionError(f"{label}: keys() / to_tuple() mismatch")

    out_loss = output_cls(logits=logits, loss=loss, past_key_values=dummy_cache)
    if list(out_loss.keys()) != ["loss", "logits", "past_key_values"] or out_loss.to_tuple() != (
        loss,
        logits,
        dummy_cache,
    ):
        raise AssertionError(f"{label}: loss ordering in keys()/to_tuple() mismatch")

    # __setitem__ / __delitem__ sync with attributes & .cache alias
    logits2 = torch.randn(1, 4, 16)
    cache2 = {"seq": 5}
    out["logits"] = logits2
    if out.logits is not logits2 or out.get("logits") is not logits2:
        raise AssertionError(f"{label}: __setitem__('logits') did not sync attribute")
    out["cache"] = cache2
    if out.past_key_values is not cache2 or out.cache is not cache2 or out["past_key_values"] is not cache2:
        raise AssertionError(f"{label}: __setitem__('cache') did not sync past_key_values")
    out["custom_meta"] = 42
    if out["custom_meta"] != 42 or getattr(out, "custom_meta", None) != 42:
        raise AssertionError(f"{label}: __setitem__ custom key sync failed")
    del out["custom_meta"]
    if "custom_meta" in out or hasattr(out, "custom_meta"):
        raise AssertionError(f"{label}: __delitem__ custom key failed")
    del out["cache"]
    if out.past_key_values is not None or "past_key_values" in out:
        raise AssertionError(f"{label}: __delitem__('cache') did not clear past_key_values")
    out["past_key_values"] = dummy_cache
    del out["past_key_values"]
    if out.past_key_values is not None or "past_key_values" in out:
        raise AssertionError(f"{label}: __delitem__('past_key_values') did not clear field")

    return {
        "inherits_expected_base": bool(expected_base_cls is None or isinstance(out, expected_base_cls)),
        "is_dataclass": True,
        "cache_alias_and_mutation_ok": True,
    }


def _verify_position_ids_slicing(model: Any, *, label: str) -> dict[str, Any]:
    """Verify `prepare_inputs_for_generation` slices 2D `[B, T]` and 3D `[3, B, T]` MRoPE `position_ids`."""

    class _PastLenCache:
        def __init__(self, seq_len: int) -> None:
            self._seq_len = seq_len

        def get_seq_length(self) -> int:
            return self._seq_len

    dev = _model_device(model)
    full_ids = torch.tensor([[11, 22, 33, 44, 55, 66]], dtype=torch.long, device=dev)
    pos_2d = torch.arange(6, dtype=torch.long, device=dev).unsqueeze(0)
    pos_3d = torch.stack(
        [
            torch.arange(6, dtype=torch.long, device=dev),
            torch.arange(6, dtype=torch.long, device=dev) + 100,
            torch.arange(6, dtype=torch.long, device=dev) + 200,
        ],
        dim=0,
    ).unsqueeze(1)  # [3, 1, 6]
    cpos = torch.tensor([5], dtype=torch.long, device=dev)
    past5 = _PastLenCache(5)
    past0 = _PastLenCache(0)

    # past_length == 0 keeps full sequence
    mi_p0 = model.prepare_inputs_for_generation(
        full_ids,
        past_key_values=past0,
        cache_position=torch.arange(6, dtype=torch.long, device=dev),
        position_ids=pos_3d,
    )
    if tuple(mi_p0["input_ids"].shape) != (1, 6) or tuple(mi_p0["position_ids"].shape) != (3, 1, 6):
        raise AssertionError(f"{label}: past_length=0 sliced inputs unexpectedly")

    # past_length > 0 with cache_position (2D & 3D MRoPE)
    mi_2d = model.prepare_inputs_for_generation(
        full_ids, past_key_values=past5, cache_position=cpos, position_ids=pos_2d
    )
    if tuple(mi_2d["input_ids"].shape) != (1, 1) or not torch.equal(mi_2d["position_ids"], pos_2d[:, -1:]):
        raise AssertionError(f"{label}: 2D position_ids slicing with cache_position failed")

    mi_3d = model.prepare_inputs_for_generation(
        full_ids, past_key_values=past5, cache_position=cpos, position_ids=pos_3d
    )
    if tuple(mi_3d["input_ids"].shape) != (1, 1) or tuple(mi_3d["position_ids"].shape) != (3, 1, 1):
        raise AssertionError(f"{label}: 3D MRoPE position_ids shape mismatch with cache_position")
    if not torch.equal(mi_3d["position_ids"], pos_3d[..., -1:]):
        raise AssertionError(f"{label}: 3D MRoPE position_ids values mismatch with cache_position")

    # past_length > 0 without cache_position (2D & 3D MRoPE)
    mi_2d_nocp = model.prepare_inputs_for_generation(full_ids, past_key_values=past5, position_ids=pos_2d)
    if tuple(mi_2d_nocp["input_ids"].shape) != (1, 1) or not torch.equal(mi_2d_nocp["position_ids"], pos_2d[:, -1:]):
        raise AssertionError(f"{label}: 2D position_ids slicing without cache_position failed")

    mi_3d_nocp = model.prepare_inputs_for_generation(full_ids, past_key_values=past5, position_ids=pos_3d)
    if tuple(mi_3d_nocp["position_ids"].shape) != (3, 1, 1) or not torch.equal(
        mi_3d_nocp["position_ids"], pos_3d[..., -1:]
    ):
        raise AssertionError(f"{label}: 3D MRoPE position_ids slicing without cache_position failed")

    return {"2d_with_cache_pos": True, "3d_mrope_with_cache_pos": True, "2d_and_3d_without_cache_pos": True}


def _expected_virtual_geometry(config: Any, depth: int) -> tuple[list[str], int]:
    """Compute expected `(full_virtual_layer_types, effective_num_hidden_layers)` for a depth."""
    bcfg = getattr(config, "base_config", None) or {}
    arch = getattr(config, "qloop_v3_arch", None) or {}
    phys_lt = list(bcfg.get("layer_types") or [])
    if not phys_lt:
        n_phys = int(bcfg.get("num_hidden_layers") or getattr(config, "physical_num_hidden_layers", 0) or 0)
        interval = bcfg.get("full_attention_interval")
        mtype = str(bcfg.get("model_type") or "").lower()
        if interval is None and ("qwen3_5" in mtype or "qwen36" in mtype or mtype.startswith("toy")):
            interval = 4
        if interval is not None and int(interval) > 0:
            iv = int(interval)
            phys_lt = ["full_attention" if (i + 1) % iv == 0 else "linear_attention" for i in range(n_phys)]
        else:
            phys_lt = ["full_attention"] * n_phys

    loop_cfg = arch.get("loop") if isinstance(arch, dict) else None
    if isinstance(loop_cfg, dict) and loop_cfg.get("enabled") and phys_lt:
        p_end = int(loop_cfg.get("prelude_end", 0))
        c_end = int(loop_cfg.get("core_end", len(phys_lt)))
        max_r = max(1, int(loop_cfg.get("num_passes", 1)))
        eff_r = max(1, min(int(depth), max_r))
        full_virt = phys_lt[:p_end] + phys_lt[p_end:c_end] * max_r + phys_lt[c_end:]
        eff_n = p_end + (c_end - p_end) * eff_r + (len(phys_lt) - c_end)
        return full_virt, eff_n
    return phys_lt, len(phys_lt)


def _extract_eos_ids(model: Any) -> set[int]:
    ids: set[int] = set()
    for holder in (getattr(model, "generation_config", None), getattr(model, "config", None)):
        if holder is None:
            continue
        raw = getattr(holder, "eos_token_id", None)
        if isinstance(raw, int):
            ids.add(raw)
        elif isinstance(raw, (list, tuple, set)):
            ids.update(int(x) for x in raw if x is not None)
    return ids


@torch.no_grad()
def _uncached_greedy_decode(
    model: Any,
    prompt_ids: torch.Tensor,
    max_new_tokens: int,
    *,
    stop_eos_ids: set[int] | None = None,
) -> torch.Tensor:
    """Reference uncached full-recompute greedy decode (zero KV/recurrent cache)."""
    cur = prompt_ids.clone()
    eos_set = stop_eos_ids or set()
    for _ in range(int(max_new_tokens)):
        out = model(input_ids=cur, use_cache=False, logits_to_keep=1)
        logits = out.logits if hasattr(out, "logits") else out[0]
        nxt = logits[:, -1:, :].argmax(dim=-1)
        cur = torch.cat([cur, nxt], dim=1)
        if eos_set and int(nxt.item()) in eos_set:
            break
    return cur


@torch.no_grad()
def _iterative_prepared_greedy_decode(
    model: Any,
    prompt_ids: torch.Tensor,
    max_new_tokens: int,
    *,
    past_key_values: Any = None,
    logits_to_keep: int = 1,
) -> torch.Tensor:
    """Exact HF-style iterative `prepare_inputs_for_generation` + `forward` greedy loop for offline CPU."""
    cur_ids = prompt_ids.clone()
    pkv = past_key_values
    attn_mask = torch.ones_like(cur_ids)
    for step in range(int(max_new_tokens)):
        past_len = pkv.get_seq_length() if (pkv is not None and hasattr(pkv, "get_seq_length")) else 0
        if step == 0 and past_len == 0:
            cache_pos = torch.arange(0, cur_ids.shape[1], device=cur_ids.device)
        else:
            cache_pos = torch.tensor([past_len], device=cur_ids.device)
        model_inputs = model.prepare_inputs_for_generation(
            cur_ids,
            past_key_values=pkv,
            attention_mask=attn_mask,
            cache_position=cache_pos,
            use_cache=True,
            logits_to_keep=logits_to_keep,
        )
        out = model(**model_inputs)
        logits = out.get("logits") if hasattr(out, "get") else out.logits
        pkv = out.get("past_key_values") if hasattr(out, "get") else extract_output_cache(out)
        nxt = logits[:, -1:, :].argmax(dim=-1)
        cur_ids = torch.cat([cur_ids, nxt], dim=1)
        attn_mask = torch.cat([attn_mask, torch.ones_like(nxt)], dim=1)
    return cur_ids


def _build_clean_hf_gen_config(model: Any, max_new_tokens: int) -> Any:
    """Build a deterministic greedy `GenerationConfig` stripped of sampling kwargs when real TF is active."""
    try:
        from transformers import GenerationConfig  # type: ignore[import-not-found]
    except ImportError:
        return None
    eos = getattr(getattr(model, "generation_config", None), "eos_token_id", None)
    if eos is None:
        eos = getattr(model.config, "eos_token_id", None)
    pad = getattr(getattr(model, "generation_config", None), "pad_token_id", None)
    if pad is None:
        pad = getattr(model.config, "pad_token_id", None)
    if pad is None:
        pad = eos[0] if isinstance(eos, list) and eos else eos
    kw: dict[str, Any] = {
        "max_new_tokens": int(max_new_tokens),
        "do_sample": False,
        "use_cache": True,
    }
    if eos is not None:
        kw["eos_token_id"] = eos
    if pad is not None:
        kw["pad_token_id"] = pad
    return GenerationConfig(**kw)


def _run_model_pack_checks(
    model: Any,
    *,
    depths: tuple[int, ...],
    max_new_tokens: int,
    prompt_ids: torch.Tensor,
    label: str,
    use_real_tf_generate: bool,
) -> dict[str, Any]:
    """Run config sync, `logits_to_keep=0`, `position_ids` 2D/3D slicing, and greedy parity on `model`."""
    dev = _model_device(model)
    prompt = prompt_ids.to(dev)
    vocab_size = int(getattr(model.config, "vocab_size", 0) or 0)
    is_toy = _is_toy_model(model)
    has_generate = hasattr(model, "generate") and callable(model.generate)

    pos_metrics = _verify_position_ids_slicing(model, label=label)
    orig_passes = int(getattr(model, "active_passes", 1))
    max_passes = int(getattr(model, "max_passes", 1))
    valid_depths = tuple(d for d in depths if 1 <= int(d) <= max_passes)
    if not valid_depths:
        raise ValueError(f"{label}: none of requested depths {depths} are within [1, {max_passes}]")

    depth_results: dict[str, Any] = {}
    try:
        with torch.no_grad():
            for d in valid_depths:
                model.set_active_passes(d)
                expected_full_lt, expected_eff_n = _expected_virtual_geometry(model.config, d)
                if list(model.config.layer_types) != expected_full_lt:
                    raise AssertionError(
                        f"{label} d={d}: config.layer_types len={len(model.config.layer_types)} "
                        f"!= expected len={len(expected_full_lt)}"
                    )
                if int(model.config.num_hidden_layers) != len(expected_full_lt):
                    raise AssertionError(
                        f"{label} d={d}: config.num_hidden_layers={model.config.num_hidden_layers} "
                        f"!= {len(expected_full_lt)}"
                    )
                if int(getattr(model.config, "active_passes", d)) != d:
                    raise AssertionError(f"{label} d={d}: config.active_passes not synced")
                if int(getattr(model.config, "effective_num_hidden_layers", expected_eff_n)) != expected_eff_n:
                    raise AssertionError(
                        f"{label} d={d}: effective_num_hidden_layers="
                        f"{getattr(model.config, 'effective_num_hidden_layers', None)} != {expected_eff_n}"
                    )
                act_lt = getattr(model.config, "active_layer_types", None)
                if act_lt is not None and len(act_lt) != expected_eff_n:
                    raise AssertionError(f"{label} d={d}: len(active_layer_types)={len(act_lt)} != {expected_eff_n}")

                # logits_to_keep=0 must return full (B, T, V), NEVER (B, 0, V)!
                out_full = model(prompt, use_cache=False)
                out_k0 = model(prompt, use_cache=False, logits_to_keep=0)
                out_k1 = model(prompt, use_cache=False, logits_to_keep=1)
                exp_b, exp_t = int(prompt.shape[0]), int(prompt.shape[1])
                eff_vocab = vocab_size if vocab_size > 0 else int(out_full.logits.shape[-1])
                if tuple(out_k0.logits.shape) != (exp_b, exp_t, eff_vocab):
                    raise AssertionError(
                        f"{label} d={d}: logits_to_keep=0 returned shape {tuple(out_k0.logits.shape)}, "
                        f"expected {(exp_b, exp_t, eff_vocab)}"
                    )
                if tuple(out_k1.logits.shape) != (exp_b, 1, eff_vocab):
                    raise AssertionError(
                        f"{label} d={d}: logits_to_keep=1 returned shape {tuple(out_k1.logits.shape)}, "
                        f"expected {(exp_b, 1, eff_vocab)}"
                    )
                k0_max_diff = float((out_k0.logits.float() - out_full.logits.float()).abs().max().item())
                k1_max_diff = float((out_k1.logits.float() - out_full.logits[:, -1:, :].float()).abs().max().item())
                tol = 1e-3 if out_full.logits.dtype in (torch.bfloat16, torch.float16) else 1e-5
                if k0_max_diff > tol or k1_max_diff > tol:
                    raise AssertionError(
                        f"{label} d={d}: logits_to_keep diff exceeded tol={tol} "
                        f"(k0={k0_max_diff:.6g}, k1={k1_max_diff:.6g})"
                    )
                if not hasattr(out_k0, "get") or out_k0.get("logits") is not out_k0.logits:
                    raise AssertionError(f"{label} d={d}: logits_to_keep=0 output lacks ModelOutput.get('logits')")

                # Reference uncached greedy decode
                eos_ids = _extract_eos_ids(model) if (use_real_tf_generate and has_generate) else set()
                ref_seq = _uncached_greedy_decode(
                    model,
                    prompt,
                    max_new_tokens,
                    stop_eos_ids=eos_ids if eos_ids else None,
                )

                gen_modes: dict[str, Any] = {}
                attn_mask = torch.ones_like(prompt)

                if has_generate:
                    gen_kw: dict[str, Any] = {
                        "max_new_tokens": int(max_new_tokens),
                        "do_sample": False,
                        "use_cache": True,
                        "attention_mask": attn_mask,
                    }
                    if use_real_tf_generate:
                        clean_gcfg = _build_clean_hf_gen_config(model, max_new_tokens)
                        if clean_gcfg is not None:
                            gen_kw["generation_config"] = clean_gcfg

                    # Implicit HF hybrid DynamicCache (for real HF backbones OR offline mock GenerationMixin)
                    if not (use_real_tf_generate and is_toy):
                        gen_imp = model.generate(prompt, **gen_kw)
                        if not torch.equal(gen_imp, ref_seq):
                            raise AssertionError(
                                f"{label} d={d}: implicit-cache generate() diverged from uncached greedy: "
                                f"got={gen_imp[0, exp_t:].tolist()} vs ref={ref_seq[0, exp_t:].tolist()}"
                            )
                        gen_modes["generate_implicit_cache_parity"] = True

                    # Explicit QuantizedHybridCache inside generate()
                    q_mode = "native" if is_toy else "bf16"
                    qcache = QuantizedHybridCache(
                        num_layers=len(expected_full_lt),
                        layer_types=expected_full_lt,
                        mode=q_mode,
                    )
                    gen_exp = model.generate(prompt, past_key_values=qcache, **gen_kw)
                    if not torch.equal(gen_exp, ref_seq):
                        raise AssertionError(
                            f"{label} d={d}: QuantizedHybridCache({q_mode}) generate() diverged from uncached greedy: "
                            f"got={gen_exp[0, exp_t:].tolist()} vs ref={ref_seq[0, exp_t:].tolist()}"
                        )
                    gen_modes[f"generate_quantized_cache_{q_mode}_parity"] = True
                else:
                    # Offline in-package without transformers: iterative prepare_inputs_for_generation + forward
                    qcache = QuantizedHybridCache(
                        num_layers=len(expected_full_lt),
                        layer_types=expected_full_lt,
                        mode="native",
                    )
                    iter_q = _iterative_prepared_greedy_decode(
                        model, prompt, max_new_tokens, past_key_values=qcache, logits_to_keep=1
                    )
                    if not torch.equal(iter_q, ref_seq):
                        raise AssertionError(
                            f"{label} d={d}: iterative prepared greedy (QuantizedHybridCache) diverged from uncached: "
                            f"got={iter_q[0, exp_t:].tolist()} vs ref={ref_seq[0, exp_t:].tolist()}"
                        )
                    gen_modes["iterative_prepared_quantized_cache_parity"] = True

                    if is_toy:
                        tm_cfg = model._text_model.config if hasattr(model, "_text_model") else model._find_tm().config
                        toy_c = ToyHybridCache(tm_cfg)
                        iter_toy = _iterative_prepared_greedy_decode(
                            model, prompt, max_new_tokens, past_key_values=toy_c, logits_to_keep=0
                        )
                        if not torch.equal(iter_toy, ref_seq):
                            raise AssertionError(
                                f"{label} d={d}: iterative prepared greedy (ToyHybridCache) diverged from uncached"
                            )
                        gen_modes["iterative_prepared_toy_cache_parity"] = True

                depth_results[f"d{d}"] = {
                    "virtual_num_hidden_layers": int(model.config.num_hidden_layers),
                    "effective_num_hidden_layers": int(expected_eff_n),
                    "logits_to_keep_0_shape": list(out_k0.logits.shape),
                    "logits_to_keep_0_max_diff": k0_max_diff,
                    "logits_to_keep_1_max_diff": k1_max_diff,
                    "generated_tokens": ref_seq[0, exp_t:].tolist(),
                    **gen_modes,
                }
                if dev.type == "cuda":
                    torch.cuda.empty_cache()
    finally:
        model.set_active_passes(orig_passes)

    return {
        "has_hf_generate": has_generate,
        "use_real_tf_generate": use_real_tf_generate,
        "position_ids": pos_metrics,
        "depths": depth_results,
    }


def _import_standalone_pack_modules(
    sa_dir: Path,
    *,
    use_real_tf: bool,
    enable_toy_auto_fallback: bool,
) -> tuple[Any, Any, Any, dict[str, Any]]:
    """Compile-check (Py3.9 AST + py_compile) and import standalone pack files with `qloop` hidden.

    Returns `(mod_cfg, mod_sa, expected_model_output_base, cleanup_state)`. Caller MUST pass
    `cleanup_state` to `_cleanup_standalone_import(cleanup_state)` in a `finally` block.
    """
    cfg_py = sa_dir / CONFIG_FILENAME
    mod_py = sa_dir / MODELING_FILENAME
    if not cfg_py.is_file() or not mod_py.is_file():
        raise FileNotFoundError(f"standalone files missing in {sa_dir}")

    # 1) Python 3.9 AST parse + py_compile
    ast.parse(cfg_py.read_text(encoding="utf-8"), filename=CONFIG_FILENAME, feature_version=(3, 9))
    ast.parse(mod_py.read_text(encoding="utf-8"), filename=MODELING_FILENAME, feature_version=(3, 9))
    py_compile.compile(str(cfg_py), doraise=True)
    py_compile.compile(str(mod_py), doraise=True)

    real_find_spec = importlib.util.find_spec
    mock_mods: dict[str, types.ModuleType] | None = None
    saved_tf: dict[str, Any] = {}
    pkg_name = f"_qloop_s2_sa_{abs(hash((str(sa_dir), time.time_ns())))}"
    added_modules: list[str] = []

    def _hide_qloop_find_spec(name: str, package: str | None = None) -> Any:
        if name == "qloop" or name.startswith("qloop."):
            return None
        return real_find_spec(name, package)

    cleanup_state: dict[str, Any] = {
        "real_find_spec": real_find_spec,
        "saved_tf": saved_tf,
        "added_modules": added_modules,
    }

    try:
        if not use_real_tf:
            mock_mods = _make_strict_tf524_mock_modules()
            for mname, mobj in mock_mods.items():
                saved_tf[mname] = sys.modules.get(mname)
                sys.modules[mname] = mobj
            expected_mo_base: Any = mock_mods["transformers.modeling_outputs"].ModelOutput
        else:
            from transformers.modeling_outputs import ModelOutput as RealModelOutput  # type: ignore[import-not-found]

            expected_mo_base = RealModelOutput

        importlib.util.find_spec = _hide_qloop_find_spec  # type: ignore[assignment]

        pkg = types.ModuleType(pkg_name)
        pkg.__path__ = [str(sa_dir)]  # type: ignore[attr-defined]
        sys.modules[pkg_name] = pkg
        added_modules.append(pkg_name)

        spec_cfg = importlib.util.spec_from_file_location(f"{pkg_name}.configuration_qwen36_loophssd", cfg_py)
        if spec_cfg is None or spec_cfg.loader is None:
            raise RuntimeError("failed to build spec for standalone configuration module")
        mod_cfg = importlib.util.module_from_spec(spec_cfg)
        sys.modules[spec_cfg.name] = mod_cfg
        added_modules.append(spec_cfg.name)
        spec_cfg.loader.exec_module(mod_cfg)

        spec_mod = importlib.util.spec_from_file_location(f"{pkg_name}.modeling_qwen36_loophssd", mod_py)
        if spec_mod is None or spec_mod.loader is None:
            raise RuntimeError("failed to build spec for standalone modeling module")
        mod_sa = importlib.util.module_from_spec(spec_mod)
        sys.modules[spec_mod.name] = mod_sa
        added_modules.append(spec_mod.name)
        spec_mod.loader.exec_module(mod_sa)

        if getattr(mod_sa, "_HAS_QLOOP", True) is not False:
            raise AssertionError("standalone modeling module failed to enter _HAS_QLOOP=False branch")

        # When running a toy pack inside an environment with real `transformers` installed,
        # real `AutoConfig.for_model("toy_qwen3_5_moe")` is unknown; wrap ONLY on `mod_sa`
        # so real `PreTrainedModel`, `GenerationMixin`, and `ModelOutput` stay 100% genuine.
        if use_real_tf and enable_toy_auto_fallback:
            orig_auto_cfg = mod_sa.AutoConfig
            orig_auto_lm = mod_sa.AutoModelForCausalLM

            class _ToyAwareAutoConfig:
                @staticmethod
                def for_model(model_type: str, **kw: Any) -> Any:
                    if str(model_type).startswith("toy"):
                        valid = {f for f in ToyConfig.__dataclass_fields__ if f in kw}
                        clean = {k: kw[k] for k in valid}
                        if "layer_types" in clean and isinstance(clean["layer_types"], list):
                            clean["layer_types"] = list(clean["layer_types"])
                        return ToyConfig(**clean)
                    return orig_auto_cfg.for_model(model_type, **kw)

            class _ToyAwareAutoModel:
                @staticmethod
                def from_config(cfg: Any, **kw: Any) -> Any:
                    if isinstance(cfg, ToyConfig):
                        return ToyForCausalLM(cfg)
                    return orig_auto_lm.from_config(cfg, **kw)

            mod_sa.AutoConfig = _ToyAwareAutoConfig
            mod_sa.AutoModelForCausalLM = _ToyAwareAutoModel

        return mod_cfg, mod_sa, expected_mo_base, cleanup_state
    except Exception:
        _cleanup_standalone_import(cleanup_state)
        raise


def _cleanup_standalone_import(state: dict[str, Any]) -> None:
    importlib.util.find_spec = state["real_find_spec"]
    for mname in reversed(state["added_modules"]):
        sys.modules.pop(mname, None)
    for mname, prev in state["saved_tf"].items():
        if prev is None:
            sys.modules.pop(mname, None)
        else:
            sys.modules[mname] = prev


def _verify_pack_metadata_files(pack_dir: Path, *, expect_current_commit: bool) -> dict[str, Any]:
    """Verify `qloop_format_version` and `qloop_commit` in `config.json` and `qloop_pack_manifest.json`."""
    cfg_path = pack_dir / "config.json"
    if not cfg_path.is_file():
        raise FileNotFoundError(f"config.json not found in {pack_dir}")
    cfg_json = json.loads(cfg_path.read_text(encoding="utf-8"))

    cfg_fmt = cfg_json.get("qloop_format_version")
    cfg_commit = cfg_json.get("qloop_commit")
    if str(cfg_fmt) != str(QLOOP_FORMAT_VERSION):
        raise AssertionError(
            f"config.json qloop_format_version={cfg_fmt!r} != expected {QLOOP_FORMAT_VERSION!r}"
        )
    if not cfg_commit or not isinstance(cfg_commit, str):
        raise AssertionError(f"config.json missing valid qloop_commit: {cfg_commit!r}")
    cur_commit = resolve_qloop_commit()
    if expect_current_commit and cur_commit not in ("unknown", cfg_commit):
        raise AssertionError(f"config.json qloop_commit={cfg_commit!r} != current {cur_commit!r}")

    man_path = pack_dir / PACK_MANIFEST_FILE
    man_fmt = None
    man_commit = None
    if man_path.is_file():
        man_json = json.loads(man_path.read_text(encoding="utf-8"))
        man_fmt = man_json.get("qloop_format_version")
        man_commit = man_json.get("qloop_commit")
        if str(man_fmt) != str(QLOOP_FORMAT_VERSION):
            raise AssertionError(
                f"{PACK_MANIFEST_FILE} qloop_format_version={man_fmt!r} != expected {QLOOP_FORMAT_VERSION!r}"
            )
        if not man_commit or not isinstance(man_commit, str):
            raise AssertionError(f"{PACK_MANIFEST_FILE} missing valid qloop_commit: {man_commit!r}")
        if expect_current_commit and cur_commit not in ("unknown", man_commit):
            raise AssertionError(f"{PACK_MANIFEST_FILE} qloop_commit={man_commit!r} != current {cur_commit!r}")
    else:
        raise FileNotFoundError(f"{PACK_MANIFEST_FILE} not found in {pack_dir}")

    idx_path = pack_dir / SAFETENSORS_INDEX_FILE
    idx_checked = False
    if idx_path.is_file():
        idx_json = json.loads(idx_path.read_text(encoding="utf-8"))
        idx_meta = idx_json.get("metadata") or {}
        if str(idx_meta.get("qloop_format_version")) != str(QLOOP_FORMAT_VERSION):
            raise AssertionError(
                f"{SAFETENSORS_INDEX_FILE} metadata qloop_format_version="
                f"{idx_meta.get('qloop_format_version')!r} != {QLOOP_FORMAT_VERSION!r}"
            )
        if not idx_meta.get("qloop_commit"):
            raise AssertionError(f"{SAFETENSORS_INDEX_FILE} metadata missing qloop_commit")
        idx_checked = True

    return {
        "qloop_format_version": cfg_fmt,
        "qloop_commit": cfg_commit,
        "manifest_qloop_format_version": man_fmt,
        "manifest_qloop_commit": man_commit,
        "safetensors_index_verified": idx_checked,
    }


def _stage_standalone_pack_dir(src_pack_dir: Path, dst_dir: Path) -> None:
    """Symlink safetensors shards + copy lightweight metadata and emit fresh standalone `.py` files."""
    dst_dir.mkdir(parents=True, exist_ok=True)
    for item in src_pack_dir.iterdir():
        if item.name in (CONFIG_FILENAME, MODELING_FILENAME):
            continue
        target = dst_dir / item.name
        if item.suffix == ".safetensors":
            os.symlink(item.resolve(), target)
        elif item.is_file():
            shutil.copy2(item, target)
    write_standalone_hf_modeling_files(dst_dir)


def _default_prompt_ids(model: Any, tokenizer: Any = None, prompt_text: str | None = None) -> torch.Tensor:
    if tokenizer is not None and prompt_text:
        enc = tokenizer.encode(prompt_text, add_special_tokens=False)
        if hasattr(enc, "tolist"):
            enc = enc.tolist()
        if isinstance(enc, list) and len(enc) >= 2:
            return torch.tensor([[int(t) for t in enc[:16]]], dtype=torch.long)
    vocab = int(getattr(model.config, "vocab_size", 64) or 64)
    return torch.tensor([[min(11 + i * 9, vocab - 1) for i in range(6)]], dtype=torch.long)


def check_pack_model_and_generate(
    pack_dir_or_model: Any,
    depths: Sequence[int] = DEFAULT_DEPTHS,
    max_new_tokens: int = 6,
    *,
    tokenizer: Any = None,
    prompt_text: str | None = None,
    device: str | None = None,
    dtype: str | torch.dtype | None = "auto",
) -> dict[str, Any]:
    """`QL60-002`: Verify in-package + standalone pack modeling, `CausalLMPackOutput`, and greedy parity."""
    name = "QL60-002:pack_model_and_generate"
    depth_tuple = tuple(int(d) for d in depths)
    t0 = time.perf_counter()
    metrics: dict[str, Any] = {
        "requested_depths": list(depth_tuple),
        "max_new_tokens": int(max_new_tokens),
        "subchecks": {},
    }

    tmp_dirs: list[tempfile.TemporaryDirectory[str]] = []
    pack_pull_tmp: tempfile.TemporaryDirectory[str] | None = None
    sa_cleanup: dict[str, Any] | None = None
    try:
        env = detect_transformers_env()
        use_real_tf = bool(env["transformers_installed"])
        if use_real_tf:
            from transformers.modeling_outputs import ModelOutput as RealModelOutput  # type: ignore[import-not-found]

            inpkg_mo_base: Any = RealModelOutput
        else:
            inpkg_mo_base = None

        # 1) In-package `CausalLMPackOutput` contract
        metrics["in_package_model_output"] = _verify_model_output_contract(
            CausalLMPackOutput, inpkg_mo_base, label="in-package"
        )
        metrics["subchecks"]["in_package_model_output"] = True

        # 2) Resolve input -> (inpkg_model, pack_dir_for_standalone, expect_current_commit, is_large_live_stack)
        eff_dev = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        pack_dir: Path | None = None
        expect_current_commit = True
        inpkg_model: Qwen36LoopHSSDForCausalLM
        owned_inpkg = False
        is_large_live_stack = False
        eff_tok = tokenizer

        if isinstance(pack_dir_or_model, (str, os.PathLike)):
            local_dir, pack_pull_tmp = resolve_pack_directory(pack_dir_or_model)
            pack_dir = Path(local_dir)
            expect_current_commit = False
            inpkg_model = Qwen36LoopHSSDForCausalLM.from_pretrained(
                pack_dir, device_map=eff_dev, dtype=dtype, strict=True
            )
            owned_inpkg = True
            if eff_tok is None:
                try:
                    from qloop.export_pack import _load_pack_tokenizer

                    eff_tok = _load_pack_tokenizer(str(pack_dir))
                except Exception:
                    eff_tok = None
        elif isinstance(pack_dir_or_model, LoadedModelPack):
            if eff_tok is None:
                eff_tok = pack_dir_or_model.tokenizer
            inpkg_model = _wrap_stack_as_hf_pack_model(pack_dir_or_model)
            if _model_param_count(inpkg_model) <= _SMALL_MODEL_PARAM_LIMIT:
                td = tempfile.TemporaryDirectory(prefix="qloop_s2_pack_")
                tmp_dirs.append(td)
                pack_dir = Path(td.name)
                export_whole_model_pack(
                    pack_dir_or_model,
                    pack_dir,
                    tokenizer=eff_tok,
                    max_shard_size_bytes=256 * 1024,
                    merge_in_place=False,
                )
            else:
                is_large_live_stack = True
        elif isinstance(pack_dir_or_model, Stack):
            if eff_tok is None:
                eff_tok = pack_dir_or_model.tokenizer
            if _model_param_count(pack_dir_or_model.model) <= _SMALL_MODEL_PARAM_LIMIT:
                td = tempfile.TemporaryDirectory(prefix="qloop_s2_pack_")
                tmp_dirs.append(td)
                pack_dir = Path(td.name)
                export_whole_model_pack(
                    pack_dir_or_model,
                    pack_dir,
                    tokenizer=eff_tok,
                    max_shard_size_bytes=256 * 1024,
                    merge_in_place=False,
                )
                inpkg_model = Qwen36LoopHSSDForCausalLM.from_pretrained(
                    pack_dir, device_map=_model_device(pack_dir_or_model.model), strict=True
                )
                owned_inpkg = True
            else:
                inpkg_model = _wrap_stack_as_hf_pack_model(pack_dir_or_model)
                is_large_live_stack = True
        elif isinstance(pack_dir_or_model, Qwen36LoopHSSDForCausalLM):
            inpkg_model = pack_dir_or_model
            if _model_param_count(inpkg_model) <= _SMALL_MODEL_PARAM_LIMIT:
                td = tempfile.TemporaryDirectory(prefix="qloop_s2_pack_")
                tmp_dirs.append(td)
                pack_dir = Path(td.name)
                export_whole_model_pack(
                    inpkg_model,
                    pack_dir,
                    tokenizer=eff_tok or ToyTokenizer(),
                    max_shard_size_bytes=256 * 1024,
                    merge_in_place=False,
                )
            else:
                is_large_live_stack = True
        elif isinstance(pack_dir_or_model, nn.Module):
            if _model_param_count(pack_dir_or_model) <= _SMALL_MODEL_PARAM_LIMIT:
                td = tempfile.TemporaryDirectory(prefix="qloop_s2_pack_")
                tmp_dirs.append(td)
                pack_dir = Path(td.name)
                export_whole_model_pack(
                    pack_dir_or_model,
                    pack_dir,
                    tokenizer=eff_tok or ToyTokenizer(),
                    max_shard_size_bytes=256 * 1024,
                    merge_in_place=False,
                )
                inpkg_model = Qwen36LoopHSSDForCausalLM.from_pretrained(
                    pack_dir, device_map=_model_device(pack_dir_or_model), strict=True
                )
                owned_inpkg = True
            else:
                tmp_stack = Stack(model=pack_dir_or_model, tokenizer=eff_tok, parts=inspect_model(pack_dir_or_model))
                inpkg_model = _wrap_stack_as_hf_pack_model(tmp_stack)
                is_large_live_stack = True
        else:
            raise TypeError(
                "pack_dir_or_model must be a pack directory/repo, Qwen36LoopHSSDForCausalLM, Stack, or nn.Module"
            )

        inpkg_model.eval()
        prompt_ids = _default_prompt_ids(inpkg_model, tokenizer=eff_tok, prompt_text=prompt_text)

        # 3) Run in-package model checks
        metrics["in_package"] = _run_model_pack_checks(
            inpkg_model,
            depths=depth_tuple,
            max_new_tokens=int(max_new_tokens),
            prompt_ids=prompt_ids,
            label="in-package",
            use_real_tf_generate=use_real_tf,
        )
        metrics["subchecks"]["in_package_model_and_generate"] = True

        # Snapshot metadata & release GPU memory before loading a large standalone pack so peak VRAM stays 1x
        inpkg_param_count = _model_param_count(inpkg_model)
        inpkg_is_toy = _is_toy_model(inpkg_model)
        inpkg_dev = str(_model_device(inpkg_model))
        inpkg_cfg_snapshot = copy.deepcopy(inpkg_model.config.to_dict())
        if owned_inpkg and inpkg_param_count > _SMALL_MODEL_PARAM_LIMIT:
            del inpkg_model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # 4) Standalone verification (freshly emitted files -> Py3.9 AST + py_compile + isolated import)
        if is_large_live_stack:
            # Verify live 35B config serialization + standalone config reload without dumping 70 GB shards,
            # AND run full standalone export/import/generate parity on a freshly exported Toy pack under
            # the exact same runtime `transformers` environment!
            td_cfg = tempfile.TemporaryDirectory(prefix="qloop_s2_live_sa_cfg_")
            tmp_dirs.append(td_cfg)
            live_cfg_dir = Path(td_cfg.name)
            write_standalone_hf_modeling_files(live_cfg_dir)
            inpkg_model.config.save_pretrained(live_cfg_dir)

            td_toy = tempfile.TemporaryDirectory(prefix="qloop_s2_sa_toy_pack_")
            tmp_dirs.append(td_toy)
            pack_dir = Path(td_toy.name)
            toy_stack = _build_smoke_toy_stack(seed=2029)
            export_whole_model_pack(
                toy_stack,
                pack_dir,
                max_shard_size_bytes=256 * 1024,
                merge_in_place=False,
            )
            expect_current_commit = True
            sa_prompt_ids = _default_prompt_ids(toy_stack.model, tokenizer=toy_stack.tokenizer)
            sa_dev = "cpu"
            sa_is_toy = True
        else:
            assert pack_dir is not None
            live_cfg_dir = None
            sa_prompt_ids = prompt_ids
            sa_dev = eff_dev if owned_inpkg else inpkg_dev
            sa_is_toy = inpkg_is_toy

        assert pack_dir is not None
        metrics["pack_metadata"] = _verify_pack_metadata_files(pack_dir, expect_current_commit=expect_current_commit)
        metrics["subchecks"]["pack_metadata"] = True

        # Always emit fresh standalone files into a staged temp dir (symlinking shards -> 0 extra disk!)
        td_sa = tempfile.TemporaryDirectory(prefix="qloop_s2_standalone_")
        tmp_dirs.append(td_sa)
        sa_dir = Path(td_sa.name)
        _stage_standalone_pack_dir(pack_dir, sa_dir)

        mod_cfg, mod_sa, sa_mo_base, sa_cleanup = _import_standalone_pack_modules(
            sa_dir,
            use_real_tf=use_real_tf,
            enable_toy_auto_fallback=sa_is_toy,
        )
        metrics["subchecks"]["standalone_ast_py39_and_py_compile"] = True

        # Standalone `CausalLMPackOutput` contract
        metrics["standalone_model_output"] = _verify_model_output_contract(
            mod_sa.CausalLMPackOutput, sa_mo_base, label="standalone"
        )
        metrics["subchecks"]["standalone_model_output"] = True

        # If we had a live 35B stack, also verify standalone Qwen36LoopHSSDConfig round-trips its exact 88L config
        if live_cfg_dir is not None:
            reloaded_live_cfg = mod_cfg.Qwen36LoopHSSDConfig.from_pretrained(str(live_cfg_dir))
            if list(reloaded_live_cfg.layer_types) != list(inpkg_cfg_snapshot.get("layer_types", [])):
                raise AssertionError("standalone Qwen36LoopHSSDConfig failed to preserve live virtual layer_types")
            if int(reloaded_live_cfg.num_hidden_layers) != int(inpkg_cfg_snapshot.get("num_hidden_layers", 0)):
                raise AssertionError("standalone Qwen36LoopHSSDConfig failed to preserve live num_hidden_layers")
            metrics["subchecks"]["standalone_live_config_roundtrip"] = True

        # Load standalone model via `from_pretrained` and run full QL60-002 checks
        sa_model = mod_sa.Qwen36LoopHSSDForCausalLM.from_pretrained(
            str(sa_dir), device_map=sa_dev, dtype=dtype if not sa_is_toy else None, strict=True
        )
        metrics["standalone"] = _run_model_pack_checks(
            sa_model,
            depths=depth_tuple,
            max_new_tokens=int(max_new_tokens),
            prompt_ids=sa_prompt_ids,
            label="standalone",
            use_real_tf_generate=use_real_tf,
        )
        metrics["standalone"]["verified_on_toy_companion"] = bool(is_large_live_stack)
        metrics["subchecks"]["standalone_model_and_generate"] = True

        # Verify non-breaking format/commit mismatch warning on standalone loader (without mutating source pack)
        mut_cfg = json.loads((sa_dir / "config.json").read_text(encoding="utf-8"))
        mut_cfg["qloop_format_version"] = "0.0-mismatch-probe"
        (sa_dir / "config.json").write_text(json.dumps(mut_cfg, indent=2), encoding="utf-8")
        if sa_is_toy:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                _ = mod_sa.Qwen36LoopHSSDForCausalLM.from_pretrained(str(sa_dir), strict=True)
            if not any("qloop_format_version" in str(w.message) for w in caught):
                raise AssertionError("standalone loader did not emit warning on qloop_format_version mismatch")
            metrics["subchecks"]["format_version_mismatch_warning"] = True

        del sa_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        metrics["elapsed_s"] = round(time.perf_counter() - t0, 4)
        return {
            "name": name,
            "passed": True,
            "detail": (
                f"Both in-package and standalone Qwen36LoopHSSDForCausalLM passed QL60-002 across "
                f"depths={list(depth_tuple)} (max_new_tokens={max_new_tokens}, "
                f"transformers={'real ' + str(env['transformers_version']) if use_real_tf else 'offline_mock'})."
            ),
            "metrics": metrics,
        }
    except Exception as exc:
        metrics["elapsed_s"] = round(time.perf_counter() - t0, 4)
        return {
            "name": name,
            "passed": False,
            "detail": f"{type(exc).__name__}: {exc}",
            "metrics": metrics,
        }
    finally:
        if sa_cleanup is not None:
            _cleanup_standalone_import(sa_cleanup)
        if pack_pull_tmp is not None:
            pack_pull_tmp.cleanup()
        for td in reversed(tmp_dirs):
            td.cleanup()


# ============================================================================ 3. QL60-005


def check_stage_completion_gate() -> dict[str, Any]:
    """`QL60-005`: Verify `qloop.colab._local_stage_done()` strict completion gate."""
    name = "QL60-005:stage_completion_gate"
    t0 = time.perf_counter()
    cases: dict[str, dict[str, Any]] = {}

    def _run_case(case_name: str, got: bool, expected: bool) -> None:
        ok = bool(got) is bool(expected)
        cases[case_name] = {"got": bool(got), "expected": bool(expected), "passed": ok}
        if not ok:
            raise AssertionError(f"case {case_name!r}: _local_stage_done returned {got!r}, expected {expected!r}")

    try:
        with tempfile.TemporaryDirectory(prefix="qloop_s2_stage_gate_") as td:
            root = Path(td)

            # 1) Bare LATEST with no state.json -> False
            ck1 = root / "s2_v2_bare_latest" / "checkpoints"
            (ck1 / "step_0000100").mkdir(parents=True)
            (ck1 / "LATEST").write_text("step_0000100\n", encoding="utf-8")
            _run_case("bare_latest_rejected", _local_stage_done(str(root), "s2_v2_bare_latest"), False)

            # 2) Mid-run LATEST with step < target_steps -> False
            (ck1 / "step_0000100" / "state.json").write_text(
                json.dumps({"step": 100, "target_steps": 600, "status": "in_progress"}), encoding="utf-8"
            )
            _run_case("midrun_latest_step_lt_target_rejected", _local_stage_done(str(root), "s2_v2_bare_latest"), False)

            # 3) Mid-run LATEST with CheckpointManager.save() metadata (no status/target_steps) -> False
            (ck1 / "step_0000100" / "state.json").write_text(
                json.dumps({"stage": "s2_v2_joint_bptt", "step": 250, "saved_at": 1700000000.0, "n_params": 1234}),
                encoding="utf-8",
            )
            _run_case("midrun_latest_save_meta_rejected", _local_stage_done(str(root), "s2_v2_bare_latest"), False)

            # 4) Bare PROMOTED with no state.json -> False
            ck2 = root / "s4_v2_rl_promoted" / "checkpoints"
            (ck2 / "step_0000025").mkdir(parents=True)
            (ck2 / "PROMOTED").write_text("step_0000025\n", encoding="utf-8")
            _run_case("bare_promoted_rejected", _local_stage_done(str(root), "s4_v2_rl_promoted"), False)

            # 5) Mid-run RL PROMOTED with CheckpointManager.save() metadata -> False
            (ck2 / "step_0000025" / "state.json").write_text(
                json.dumps({"stage": "s4_v2_rl", "step": 25, "saved_at": 1700000000.0, "n_params": 1234}),
                encoding="utf-8",
            )
            _run_case("midrun_promoted_save_meta_rejected", _local_stage_done(str(root), "s4_v2_rl_promoted"), False)

            # 6) Incomplete PROMOTED with step < target_steps -> False
            (ck2 / "step_0000025" / "state.json").write_text(
                json.dumps({"step": 25, "target_steps": 400, "status": "in_progress"}), encoding="utf-8"
            )
            _run_case(
                "incomplete_promoted_step_lt_target_rejected",
                _local_stage_done(str(root), "s4_v2_rl_promoted"),
                False,
            )

            # 7) Verified completed LATEST with step >= target_steps -> True
            (ck1 / "step_0000100" / "state.json").write_text(
                json.dumps({"step": 600, "target_steps": 600}), encoding="utf-8"
            )
            _run_case(
                "verified_latest_step_ge_target_accepted",
                _local_stage_done(str(root), "s2_v2_bare_latest"),
                True,
            )

            # 8) Verified completed PROMOTED with step >= target_steps -> True
            (ck2 / "step_0000025" / "state.json").write_text(
                json.dumps({"step": 400, "target_steps": 400, "status": "completed"}), encoding="utf-8"
            )
            _run_case(
                "verified_promoted_step_ge_target_accepted",
                _local_stage_done(str(root), "s4_v2_rl_promoted"),
                True,
            )

            # 9) Explicit checkpoints/COMPLETED marker -> True (even alongside incomplete LATEST)
            ck3 = root / "s2_v2_completed_marker" / "checkpoints"
            (ck3 / "step_0000050").mkdir(parents=True)
            (ck3 / "LATEST").write_text("step_0000050\n", encoding="utf-8")
            (ck3 / "step_0000050" / "state.json").write_text(
                json.dumps({"step": 50, "target_steps": 600, "status": "in_progress"}), encoding="utf-8"
            )
            _run_case(
                "before_completed_marker_rejected",
                _local_stage_done(str(root), "s2_v2_completed_marker"),
                False,
            )
            (ck3 / "COMPLETED").write_text("step_0000050\n", encoding="utf-8")
            _run_case(
                "checkpoints_completed_marker_accepted",
                _local_stage_done(str(root), "s2_v2_completed_marker"),
                True,
            )

            # 10) Stage-root COMPLETED marker -> True
            st4 = root / "s3_v2_root_completed"
            st4.mkdir(parents=True)
            (st4 / "COMPLETED").write_text("done\n", encoding="utf-8")
            _run_case(
                "stage_root_completed_marker_accepted",
                _local_stage_done(str(root), "s3_v2_root_completed"),
                True,
            )

            # 11) Stage-level summary.json paused vs completed
            st5 = root / "s4_v2_summary"
            st5.mkdir(parents=True)
            (st5 / "summary.json").write_text(json.dumps({"status": "paused"}), encoding="utf-8")
            _run_case("summary_paused_rejected", _local_stage_done(str(root), "s4_v2_summary"), False)
            (st5 / "summary.json").write_text(json.dumps({"status": "completed"}), encoding="utf-8")
            _run_case("summary_completed_accepted", _local_stage_done(str(root), "s4_v2_summary"), True)

        elapsed = round(time.perf_counter() - t0, 4)
        return {
            "name": name,
            "passed": True,
            "detail": f"All {len(cases)} _local_stage_done() rejection/acceptance checks passed.",
            "metrics": {"cases_tested": len(cases), "cases": cases, "elapsed_s": elapsed},
        }
    except Exception as exc:
        elapsed = round(time.perf_counter() - t0, 4)
        return {
            "name": name,
            "passed": False,
            "detail": f"{type(exc).__name__}: {exc}",
            "metrics": {"cases_tested": len(cases), "cases": cases, "elapsed_s": elapsed},
        }


# ============================================================================ checkpoint loader & reporting


def _is_whole_model_pack_dir(path: str) -> bool:
    p = Path(path)
    if not p.is_dir():
        return False
    if (p / PACK_MANIFEST_FILE).is_file() or (p / MODELING_FILENAME).is_file():
        return True
    cfg_p = p / "config.json"
    if cfg_p.is_file():
        try:
            data = json.loads(cfg_p.read_text(encoding="utf-8"))
            if str(data.get("model_type") or "") == "qwen36_loophssd":
                return True
        except Exception:
            pass
    return False


def _looks_like_hf_repo_id(raw: str) -> bool:
    """True when `raw` looks like a non-existent Hugging Face `owner/repo` ID (not a local/runs path)."""
    s = raw.strip()
    if not s or os.path.exists(s) or s.startswith(("/", ".", "~")):
        return False
    parts = s.split("/")
    if len(parts) != 2 or not parts[0] or not parts[1]:
        return False
    return parts[0] not in ("runs", "checkpoints", "configs", "tmp", "var", "content")


def _load_target_for_audit(
    checkpoint: str,
    config_path: str | None,
    *,
    overrides: list[str] | None = None,
) -> tuple[Any, Any, str]:
    """Load either an exported Whole Model Pack (local/Hub) or a training checkpoint `Stack`."""
    if _is_whole_model_pack_dir(checkpoint):
        return checkpoint, None, "whole_model_pack_dir"

    if not os.path.exists(checkpoint) and config_path is None:
        # Treat as Hugging Face Whole Model Pack repo ID when no training config is supplied
        return checkpoint, None, "whole_model_pack_hf_repo"

    if not config_path:
        raise ValueError(
            f"checkpoint {checkpoint!r} is not an exported Whole Model Pack directory; "
            "pass `--config <stage.yaml>` to load a training checkpoint Stack"
        )

    from qloop.cli import _stack_with_ckpt
    from qloop.config import Cfg, load_config

    cfg = load_config(config_path, overrides=overrides)
    eff_ckpt = checkpoint
    target_kind = "training_checkpoint_stack"
    if _looks_like_hf_repo_id(checkpoint):
        ckpt_section = cfg.get("checkpoint")
        if not isinstance(ckpt_section, dict):
            ckpt_section = Cfg()
            cfg["checkpoint"] = ckpt_section
        sync_section = ckpt_section.get("sync")
        if not isinstance(sync_section, dict):
            sync_section = Cfg()
            ckpt_section["sync"] = sync_section
        sync_section["enabled"] = True
        sync_section["hf_repo"] = checkpoint.strip()

        stage_name = str(cfg.get("stage") or (cfg.get("run") or {}).get("name") or "").strip()
        if not stage_name:
            raise ValueError(
                f"cannot resolve stage name from {config_path!r} for HF Hub checkpoint repo {checkpoint!r}"
            )
        eff_ckpt = f"runs/{stage_name}"
        target_kind = "training_checkpoint_hf_repo"

    stack = _stack_with_ckpt(cfg, eff_ckpt)
    stack.model.eval()
    return stack, stack.tokenizer, target_kind


def build_audit_report(
    *,
    mode: str,
    env: dict[str, Any],
    checks: list[dict[str, Any]],
    elapsed_s: float,
    target_info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble the deterministic JSON audit report payload."""
    return {
        "schema_version": 1,
        "timestamp_utc": _utc_now(),
        "mode": mode,
        "overall_passed": all(bool(c.get("passed")) for c in checks),
        "passed_count": sum(1 for c in checks if c.get("passed")),
        "total_checks": len(checks),
        "elapsed_s": round(float(elapsed_s), 4),
        "environment": env,
        "target": target_info or {},
        "checks": checks,
    }


def render_markdown_report(report: dict[str, Any]) -> str:
    """Render a concise, human-readable Markdown summary of the serving audit."""
    env = report.get("environment") or {}
    target = report.get("target") or {}
    verdict = "PASS" if report.get("overall_passed") else "FAIL"

    lines: list[str] = [
        f"# Sprint 2 `transformers 5.24.x` Serving & Export Audit — **{verdict}**",
        "",
        f"- **Timestamp (UTC):** `{report.get('timestamp_utc')}`",
        f"- **Mode:** `{report.get('mode')}` · **Elapsed:** `{report.get('elapsed_s')}s`",
        f"- **Checks Passed:** `{report.get('passed_count')}/{report.get('total_checks')}`",
        f"- **Runtime:** Python `{env.get('python_version')}` · PyTorch `{env.get('torch_version')}` · "
        f"`transformers` `{env.get('transformers_version') or 'not-installed'}` "
        f"(v5+: `{env.get('transformers_is_v5_plus')}`)",
        f"- **Device:** `{env.get('gpu_name') or 'CPU'}` · "
        f"**Pack Format:** `{env.get('qloop_format_version')}` · **Commit:** `{env.get('qloop_commit')}`",
    ]
    if target:
        target_parts = [f"`{k}={v}`" for k, v in sorted(target.items()) if v is not None]
        if target_parts:
            lines.append(f"- **Target:** {' · '.join(target_parts)}")

    lines.extend(
        [
            "",
            "## Check Summary",
            "",
            "| Check | Status | Elapsed (s) | Detail |",
            "| :--- | :---: | ---: | :--- |",
        ]
    )
    for chk in report.get("checks", []):
        st = "PASS" if chk.get("passed") else "FAIL"
        el = (chk.get("metrics") or {}).get("elapsed_s", "")
        detail = str(chk.get("detail") or "").replace("\n", " ").replace("|", "\\|")
        lines.append(f"| `{chk.get('name')}` | **{st}** | `{el}` | {detail} |")

    # Highlight KV cache mode compression metrics when present
    for chk in report.get("checks", []):
        if chk.get("name") == "QL60-001:quantized_hybrid_cache":
            per_mode = (chk.get("metrics") or {}).get("per_mode") or {}
            if per_mode:
                lines.extend(
                    [
                        "",
                        "## `QL60-001` KV Cache Modes (`kv_share=True` vs `kv_share=False`)",
                        "",
                        "| Mode | Shared P1 Bytes | Shared P2 Bytes | `kv_share=True` Ratio | `kv_share=False` Ratio |",
                        "| :--- | ---: | ---: | ---: | ---: |",
                    ]
                )
                for mname, mdata in per_mode.items():
                    lines.append(
                        f"| `{mname}` | `{mdata.get('kv_bytes_shared_p1')}` | "
                        f"`{mdata.get('kv_bytes_shared_p2')}` | "
                        f"`{mdata.get('kv_share_true_ratio'):.2f}x` | "
                        f"`{mdata.get('kv_share_false_ratio'):.2f}x` |"
                    )
        elif chk.get("name") == "QL60-002:pack_model_and_generate":
            inpkg = (chk.get("metrics") or {}).get("in_package") or {}
            depths_map = inpkg.get("depths") or {}
            if depths_map:
                lines.extend(
                    [
                        "",
                        "## `QL60-002` Depth Parity & `logits_to_keep=0`",
                        "",
                        "| Depth | Virtual L | Effective L | `logits_to_keep=0` Shape | Max Diff | Generated Tokens |",
                        "| :--- | ---: | ---: | :--- | ---: | :--- |",
                    ]
                )
                for dkey, dval in depths_map.items():
                    lines.append(
                        f"| `{dkey}` | `{dval.get('virtual_num_hidden_layers')}` | "
                        f"`{dval.get('effective_num_hidden_layers')}` | "
                        f"`{dval.get('logits_to_keep_0_shape')}` | "
                        f"`{dval.get('logits_to_keep_0_max_diff'):.2e}` | "
                        f"`{dval.get('generated_tokens')}` |"
                    )

    return "\n".join(lines) + "\n"


def _parse_csv_ints(raw: str) -> tuple[int, ...]:
    out = tuple(int(x.strip()) for x in raw.split(",") if x.strip())
    if not out:
        raise argparse.ArgumentTypeError("expected at least one integer depth")
    return out


def _parse_csv_strs(raw: str) -> tuple[str, ...]:
    out = tuple(x.strip() for x in raw.split(",") if x.strip())
    if not out:
        raise argparse.ArgumentTypeError("expected at least one mode")
    return out


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Verify QL60-001, QL60-002, and QL60-005 against live transformers 5.24.x or offline CPU."
    )
    ap.add_argument(
        "--smoke-cpu",
        action="store_true",
        help="Run all 3 check groups completely offline on CPU using ToyForCausalLM + apply_loop + export_pack.",
    )
    ap.add_argument(
        "--checkpoint",
        default=None,
        help="Path to a training stage/checkpoint dir, exported Whole Model Pack dir, or HF Hub repo ID.",
    )
    ap.add_argument(
        "--config",
        default=None,
        help="Stage YAML config path (required when --checkpoint points to a training checkpoint).",
    )
    ap.add_argument(
        "-s",
        "--set",
        action="append",
        default=[],
        help="Optional dotted config override (repeatable, e.g. -s model.device=cuda:0).",
    )
    ap.add_argument(
        "--depths",
        type=_parse_csv_ints,
        default=DEFAULT_DEPTHS,
        help="Comma-separated active pass depths to verify (default: 1,2,3).",
    )
    ap.add_argument(
        "--modes",
        type=_parse_csv_strs,
        default=DEFAULT_MODES,
        help="Comma-separated QuantizedHybridCache modes to verify (default: bf16,fp8,hadamard_int4).",
    )
    ap.add_argument(
        "--max-new-tokens",
        type=int,
        default=8,
        help="Number of greedy decode tokens for cached vs uncached parity verification (default: 8).",
    )
    ap.add_argument(
        "--prompt",
        default=DEFAULT_PROMPT,
        help="Prompt string for real-model greedy decode parity checks.",
    )
    ap.add_argument(
        "--device",
        default=None,
        help="Optional device override for pack loading (e.g. cpu, cuda:0, auto).",
    )
    ap.add_argument(
        "--dtype",
        default="auto",
        help="Optional dtype override for pack loading (default: auto).",
    )
    ap.add_argument(
        "--strict-tf5",
        action="store_true",
        help="Fail immediately if real `transformers >= 5` is not installed.",
    )
    ap.add_argument(
        "--output-json",
        "--out-json",
        dest="output_json",
        type=Path,
        default=None,
        help="Optional path to write the structured JSON audit report.",
    )
    ap.add_argument(
        "--output-md",
        "--out-md",
        dest="output_md",
        type=Path,
        default=None,
        help="Optional path to write the Markdown audit report.",
    )
    ap.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress stdout summary table.",
    )
    args = ap.parse_args(argv)
    if not args.smoke_cpu and not args.checkpoint:
        ap.error("specify either --smoke-cpu or --checkpoint <dir_or_hf_repo>")
    return args


def run_audit(args: argparse.Namespace) -> dict[str, Any]:
    """Execute all configured audit checks and return the full structured report dict."""
    t_all = time.perf_counter()
    env = detect_transformers_env()
    env["strict_tf5"] = bool(args.strict_tf5)
    checks: list[dict[str, Any]] = []
    target_info: dict[str, Any] = {
        "depths": list(args.depths),
        "modes": list(args.modes),
        "max_new_tokens": int(args.max_new_tokens),
    }

    if args.strict_tf5:
        tf5_ok = bool(env["transformers_installed"] and env["transformers_is_v5_plus"])
        checks.append(
            {
                "name": "ENV:strict_tf5",
                "passed": tf5_ok,
                "detail": (
                    f"Real transformers {env['transformers_version']} (>= 5) verified."
                    if tf5_ok
                    else (
                        f"--strict-tf5 requires real transformers >= 5, "
                        f"found {env['transformers_version'] or 'none'}."
                    )
                ),
                "metrics": {
                    "transformers_installed": env["transformers_installed"],
                    "transformers_version": env["transformers_version"],
                    "transformers_major": env["transformers_major"],
                    "elapsed_s": 0.0,
                },
            }
        )
        if not tf5_ok:
            return build_audit_report(
                mode="smoke_cpu" if args.smoke_cpu else "checkpoint",
                env=env,
                checks=checks,
                elapsed_s=time.perf_counter() - t_all,
                target_info=target_info,
            )

    if args.smoke_cpu:
        target_info["kind"] = "toy_smoke_cpu"
        with tempfile.TemporaryDirectory(prefix="qloop_smoke_cpu_pack_") as td:
            smoke_stack = _build_smoke_toy_stack(seed=2026, kv_share=True)
            pack_dir = Path(td) / "toy_pack"
            export_whole_model_pack(
                smoke_stack,
                pack_dir,
                max_shard_size_bytes=256 * 1024,
                merge_in_place=False,
            )
            checks.append(check_quantized_hybrid_cache(smoke_stack, modes=args.modes))
            checks.append(
                check_pack_model_and_generate(
                    pack_dir,
                    depths=args.depths,
                    max_new_tokens=int(args.max_new_tokens),
                    tokenizer=smoke_stack.tokenizer,
                    device="cpu",
                )
            )
            checks.append(check_stage_completion_gate())
        return build_audit_report(
            mode="smoke_cpu",
            env=env,
            checks=checks,
            elapsed_s=time.perf_counter() - t_all,
            target_info=target_info,
        )

    # Real checkpoint / Whole Model Pack mode
    target_info["checkpoint"] = str(args.checkpoint)
    if args.config:
        target_info["config"] = str(args.config)
    target_obj, tok, target_kind = _load_target_for_audit(
        str(args.checkpoint),
        args.config,
        overrides=list(args.set or []),
    )
    target_info["kind"] = target_kind

    if isinstance(target_obj, (str, os.PathLike)):
        # Load config cleanly for QL60-001 without holding duplicate GPU weights during QL60-002
        local_dir, pull_tmp = resolve_pack_directory(target_obj)
        try:
            pack_cfg = Qwen36LoopHSSDConfig.from_pretrained(local_dir)
            checks.append(check_quantized_hybrid_cache(pack_cfg, modes=args.modes))
            checks.append(
                check_pack_model_and_generate(
                    local_dir,
                    depths=args.depths,
                    max_new_tokens=int(args.max_new_tokens),
                    prompt_text=args.prompt,
                    device=args.device,
                    dtype=args.dtype,
                )
            )
        finally:
            if pull_tmp is not None:
                pull_tmp.cleanup()
    else:
        checks.append(check_quantized_hybrid_cache(target_obj, modes=args.modes))
        checks.append(
            check_pack_model_and_generate(
                target_obj,
                depths=args.depths,
                max_new_tokens=int(args.max_new_tokens),
                tokenizer=tok,
                prompt_text=args.prompt,
                device=args.device,
                dtype=args.dtype,
            )
        )
    checks.append(check_stage_completion_gate())

    return build_audit_report(
        mode="checkpoint",
        env=env,
        checks=checks,
        elapsed_s=time.perf_counter() - t_all,
        target_info=target_info,
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = run_audit(args)

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    md_text = render_markdown_report(report)
    if args.output_md is not None:
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        args.output_md.write_text(md_text, encoding="utf-8")

    if not args.quiet:
        for chk in report["checks"]:
            status = "PASS" if chk["passed"] else "FAIL"
            print(f"[{status}] {chk['name']}: {chk['detail']}")
        overall = "PASS" if report["overall_passed"] else "FAIL"
        print(
            f"OVERALL: {overall} ({report['passed_count']}/{report['total_checks']} checks, "
            f"{report['elapsed_s']:.2f}s)"
        )

    return 0 if report["overall_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
