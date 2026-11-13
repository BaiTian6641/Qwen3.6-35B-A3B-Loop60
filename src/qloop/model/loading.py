"""Load the base model (real HF Qwen3.6-35B-A3B or the toy) with verbose diagnostics.

`model:` config block:
    name: Qwen/Qwen3.6-35B-A3B        # or unsloth/Qwen3.6-35B-A3B-NVFP4-Fast, or "toy"
    revision: null
    dtype: bfloat16
    attn_implementation: sdpa         # dense fallback used under D-LSA
    device_map: cuda:0
    text_only: true                   # load *ForCausalLM (no vision tower) when possible
    yarn_target_len: null             # e.g. 1048576 -> YaRN factor 4 on the 262K native RoPE
    trust_remote_code: false
    low_cpu_mem_usage: true
    toy: {}                           # ToyConfig overrides when name == toy
"""
from __future__ import annotations

import logging
import time
from collections import Counter
from typing import Any

import torch
from torch import nn

from ..debug import cuda_mem_brief, stage
from .introspect import inspect_model, summarize

log = logging.getLogger("qloop")

_DTYPES = {"bfloat16": torch.bfloat16, "bf16": torch.bfloat16, "float16": torch.float16, "fp16": torch.float16,
           "float32": torch.float32, "fp32": torch.float32, "float64": torch.float64}


def resolve_dtype(name: str | None) -> torch.dtype:
    if name is None:
        return torch.bfloat16
    if name not in _DTYPES:
        raise ValueError(f"unknown dtype {name!r}; choose from {sorted(_DTYPES)}")
    return _DTYPES[name]


def resolve_device(device_map: Any) -> str:
    if isinstance(device_map, str) and device_map.startswith("cuda") and not torch.cuda.is_available():
        log.warning("device_map=%s requested but CUDA is unavailable -> using cpu (dev laptop mode)", device_map)
        return "cpu"
    return device_map if isinstance(device_map, str) else "cpu"


def param_report(model: nn.Module) -> dict[str, Any]:
    by_dtype: Counter = Counter()
    by_kind: Counter = Counter()
    trainable = 0
    for name, p in model.named_parameters():
        by_dtype[str(p.dtype).replace("torch.", "")] += p.numel()
        kind = ("experts" if ".experts." in name else "router" if name.endswith("mlp.gate.weight") else
                "shared_expert" if "shared_expert" in name else "linear_attn" if "linear_attn" in name else
                "self_attn" if "self_attn" in name else "embed" if "embed" in name else
                "lm_head" if "lm_head" in name else "visual" if "visual" in name else "other")
        by_kind[kind] += p.numel()
        trainable += p.numel() if p.requires_grad else 0
    quant_modules = Counter(type(m).__name__ for m in model.modules()
                            if any(t in type(m).__name__.lower() for t in ("fp8", "fp4", "quant", "4bit", "8bit")))
    return {"params_by_dtype": dict(by_dtype), "params_by_kind": dict(by_kind),
            "trainable": trainable, "quantized_module_types": dict(quant_modules)}


def _load_toy(mcfg: dict[str, Any]):
    from ..data.tokenizer import ToyTokenizer
    from .toy import ToyConfig, ToyForCausalLM

    tok = ToyTokenizer()
    over = dict(mcfg.get("toy") or {})
    over.setdefault("vocab_size", tok.vocab_size)
    torch.manual_seed(int(over.pop("seed", 0)))
    cfg = ToyConfig(**over)
    if mcfg.get("yarn_target_len"):
        from .lsa import apply_yarn

        cfg.max_position_embeddings = 4096
        apply_yarn(cfg, int(mcfg["yarn_target_len"]))
    model = ToyForCausalLM(cfg).to(resolve_dtype(mcfg.get("dtype", "float32")))
    return model, tok


def _load_hf(mcfg: dict[str, Any]):
    try:
        import transformers
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    except Exception as e:
        raise RuntimeError(f"transformers is required for real-model loading: {e!r}. "
                           "Run scripts/colab_setup.sh or `pip install -e .[colab]`.") from e
    name = mcfg["name"]
    common = {"revision": mcfg.get("revision"), "trust_remote_code": bool(mcfg.get("trust_remote_code", False))}
    with stage("load config"):
        config = AutoConfig.from_pretrained(name, **common)
        mt = getattr(config, "model_type", "?")
        log.info("model_type=%s transformers=%s", mt, transformers.__version__)
        text_cfg = getattr(config, "text_config", config)
        for k in ("num_hidden_layers", "hidden_size", "num_experts", "num_experts_per_tok", "num_attention_heads",
                  "num_key_value_heads", "head_dim", "full_attention_interval", "max_position_embeddings"):
            log.info("  text_config.%s = %s", k, getattr(text_cfg, k, "?"))
        if mcfg.get("yarn_target_len"):
            from .lsa import apply_yarn

            apply_yarn(text_cfg, int(mcfg["yarn_target_len"]))
    kwargs = dict(common, torch_dtype=resolve_dtype(mcfg.get("dtype")), device_map=mcfg.get("device_map", "cuda:0"),
                  attn_implementation=mcfg.get("attn_implementation", "sdpa"),
                  low_cpu_mem_usage=bool(mcfg.get("low_cpu_mem_usage", True)))
    with stage("load weights"):
        t0 = time.time()
        model = None
        errors = []
        if mcfg.get("text_only", True):
            try:
                model = AutoModelForCausalLM.from_pretrained(name, config=text_cfg, **kwargs)
            except Exception as e:
                errors.append(f"AutoModelForCausalLM(text_config): {e!r}")
                log.warning("text-only load failed (%r); falling back to the full multimodal class", e)
        if model is None:
            try:
                from transformers import AutoModelForImageTextToText

                model = AutoModelForImageTextToText.from_pretrained(name, config=config, **kwargs)
            except Exception as e:
                errors.append(f"AutoModelForImageTextToText: {e!r}")
                raise RuntimeError("could not load model:\n  " + "\n  ".join(errors)) from e
        log.info("weights loaded in %.0fs %s", time.time() - t0, cuda_mem_brief("| "))
    tok = AutoTokenizer.from_pretrained(mcfg.get("tokenizer") or name, **common)
    return model, tok


def load_model(mcfg: dict[str, Any]):
    """Returns (model, tokenizer, info). Base parameters are frozen on return."""
    name = mcfg.get("name", "toy")
    model, tok = _load_toy(mcfg) if name == "toy" else _load_hf(mcfg)
    if name == "toy":
        model = model.to(resolve_device(mcfg.get("device_map", "cpu")))
    for p in model.parameters():
        p.requires_grad_(False)
    parts = inspect_model(model)
    info = param_report(model)
    log.info("%s", summarize(parts))
    for k, v in info.items():
        log.info("  %s: %s", k, v)
    if info["quantized_module_types"]:
        log.warning("quantized modules present %s: run `qloop probe --backward` once to confirm gradients flow "
                    "through frozen quantized experts on this transformers/kernel version",
                    info["quantized_module_types"])
    return model, tok, info
