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

import copy
import gc
import logging
import time
from collections import Counter
from collections.abc import Iterable
from typing import Any

import torch
from torch import nn

from ..debug import cuda_mem_brief, stage
from .introspect import inspect_model, summarize
from .quant_autograd import install_quantized_autograd_bridge, is_quantized_module

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
    quant_modules = Counter(
        type(m).__name__
        for m in model.modules()
        if is_quantized_module(m)
        or any(t in type(m).__name__.lower() for t in ("fp8", "fp4", "quant", "4bit", "8bit", "compressed"))
    )
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


def _missing_decoder_keys(loading_info: Any) -> list[str]:
    if not isinstance(loading_info, dict):
        return []
    return [k for k in loading_info.get("missing_keys", ()) if "layers." in k or "embed_tokens" in k]


def _prune_visual_tower(model: nn.Module) -> int:
    """Drop unused vision tower modules when `text_only=True` loaded a multimodal wrapper."""
    freed = 0
    for owner in (model, getattr(model, "model", None)):
        vis = getattr(owner, "visual", None)
        if isinstance(vis, nn.Module):
            freed += sum(p.numel() for p in vis.parameters())
            owner.visual = None
    if freed > 0:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        log.info("pruned unused visual tower (%.2fB params) for text_only mode %s", freed / 1e9, cuda_mem_brief("| "))
    return freed


_IGNORE_LIST_KEYS = ("ignore", "modules_to_not_convert", "skip_modules", "llm_int8_skip_modules")


def _is_visual_ignore_pattern(entry: str) -> bool:
    pat = entry[3:] if entry.startswith("re:") else entry
    pat_l = pat.lstrip("^")
    if pat_l in ("visual", "model.visual", r"model\.visual"):
        return True
    if pat_l.startswith(("visual.", "visual*", r"visual\.", "model.visual.", "model.visual*", r"model\.visual\.")):
        return True
    return entry.startswith("re:") and pat in (".*visual.*", ".*visual", r".*\.visual\..*")


def _normalize_text_only_ignore_entry(entry: str) -> str | None:
    if _is_visual_ignore_pattern(entry):
        return None
    if entry.startswith("re:"):
        body = entry[3:]
        body = body.replace(r"model\.language_model\.", r"model\.")
        body = body.replace("model.language_model.", "model.")
        if body.startswith(r"^language_model\."):
            body = r"^model\." + body[len(r"^language_model\."):]
        elif body.startswith("language_model."):
            body = "model." + body[len("language_model."):]
        return "re:" + body
    if entry in ("model.language_model", "language_model"):
        return "model"
    if entry.startswith("model.language_model."):
        return "model." + entry[len("model.language_model."):]
    if entry.startswith("language_model.model."):
        return "model." + entry[len("language_model.model."):]
    if entry.startswith("language_model."):
        return "model." + entry[len("language_model."):]
    return entry


def _normalize_text_only_ignore_list(items: Iterable[Any]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for raw in items:
        if not isinstance(raw, str):
            continue
        norm = _normalize_text_only_ignore_entry(raw)
        if norm is not None and norm not in seen:
            seen.add(norm)
            out.append(norm)
    return out


def _normalize_quant_config_for_text_only(qcfg: Any) -> Any:
    if isinstance(qcfg, dict):
        qcopy: Any = copy.deepcopy(qcfg)
    elif hasattr(qcfg, "to_dict") and callable(qcfg.to_dict):
        qcopy = copy.deepcopy(dict(qcfg.to_dict()))
    else:
        qcopy = copy.deepcopy(qcfg)

    if isinstance(qcopy, dict):
        for k in _IGNORE_LIST_KEYS:
            if isinstance(qcopy.get(k), (list, tuple)):
                qcopy[k] = _normalize_text_only_ignore_list(qcopy[k])
        cgroups = qcopy.get("config_groups")
        if isinstance(cgroups, dict):
            for grp in cgroups.values():
                if isinstance(grp, dict) and isinstance(grp.get("ignore"), (list, tuple)):
                    grp["ignore"] = _normalize_text_only_ignore_list(grp["ignore"])
        return qcopy

    for k in _IGNORE_LIST_KEYS:
        val = getattr(qcopy, k, None)
        if isinstance(val, (list, tuple)):
            setattr(qcopy, k, _normalize_text_only_ignore_list(val))
    cgroups = getattr(qcopy, "config_groups", None)
    if isinstance(cgroups, dict):
        for grp in cgroups.values():
            if isinstance(grp, dict) and isinstance(grp.get("ignore"), (list, tuple)):
                grp["ignore"] = _normalize_text_only_ignore_list(grp["ignore"])
            elif hasattr(grp, "ignore") and isinstance(getattr(grp, "ignore", None), (list, tuple)):
                grp.ignore = _normalize_text_only_ignore_list(grp.ignore)
    return qcopy


def _propagate_quantization_config_to_text_config(config: Any, text_cfg: Any) -> Any:
    """Copy & path-normalize top-level `quantization_config` onto `text_cfg` when missing."""
    if text_cfg is config:
        return text_cfg
    top_qcfg = getattr(config, "quantization_config", None)
    cur_qcfg = getattr(text_cfg, "quantization_config", None)
    has_top = top_qcfg is not None and (not isinstance(top_qcfg, dict) or len(top_qcfg) > 0)
    has_cur = cur_qcfg is not None and (not isinstance(cur_qcfg, dict) or len(cur_qcfg) > 0)
    if not has_top or has_cur:
        return text_cfg
    text_cfg_copy = copy.deepcopy(text_cfg)
    text_cfg_copy.quantization_config = _normalize_quant_config_for_text_only(top_qcfg)
    return text_cfg_copy


def _load_hf(mcfg: dict[str, Any]):
    try:
        import transformers
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    except Exception as e:
        raise RuntimeError(f"transformers is required for real-model loading: {e!r}. "
                           "Run scripts/colab_setup.sh or `pip install -e .[colab]`.") from e
    name = mcfg["name"]
    common = {"revision": mcfg.get("revision"), "trust_remote_code": bool(mcfg.get("trust_remote_code", False))}
    text_only = bool(mcfg.get("text_only", True))
    with stage("load config"):
        config = AutoConfig.from_pretrained(name, **common)
        mt = getattr(config, "model_type", "?")
        log.info("model_type=%s transformers=%s", mt, transformers.__version__)
        raw_text_cfg = getattr(config, "text_config", config)
        text_cfg = _propagate_quantization_config_to_text_config(config, raw_text_cfg) if text_only else raw_text_cfg
        for k in ("num_hidden_layers", "hidden_size", "num_experts", "num_experts_per_tok", "num_attention_heads",
                  "num_key_value_heads", "head_dim", "full_attention_interval", "max_position_embeddings"):
            log.info("  text_config.%s = %s", k, getattr(text_cfg, k, "?"))
        if getattr(text_cfg, "quantization_config", None) is not None:
            qraw = text_cfg.quantization_config
            qm = qraw.get("quant_method") if isinstance(qraw, dict) else getattr(qraw, "quant_method", "?")
            log.info("  text_config.quantization_config.quant_method = %s", qm)
        if mcfg.get("yarn_target_len"):
            from .lsa import apply_yarn

            apply_yarn(text_cfg, int(mcfg["yarn_target_len"]))
    ver_head = str(getattr(transformers, "__version__", "5")).split(".", 1)[0]
    dtype_key = "dtype" if (ver_head.isdigit() and int(ver_head) >= 5) else "torch_dtype"
    kwargs: dict[str, Any] = dict(
        common,
        device_map=mcfg.get("device_map", "cuda:0"),
        attn_implementation=mcfg.get("attn_implementation", "sdpa"),
        low_cpu_mem_usage=bool(mcfg.get("low_cpu_mem_usage", True)),
        **{dtype_key: resolve_dtype(mcfg.get("dtype"))},
    )
    with stage("load weights"):
        t0 = time.time()
        model = None
        errors = []
        if text_only:
            try:
                res = AutoModelForCausalLM.from_pretrained(name, config=text_cfg, output_loading_info=True, **kwargs)
                cand, linfo = res if isinstance(res, tuple) else (res, {})
                missing = _missing_decoder_keys(linfo)
                if missing:
                    errors.append(f"AutoModelForCausalLM(text_config): {len(missing)} missing keys ({missing[:3]})")
                    log.warning(
                        "AutoModelForCausalLM left %d decoder keys uninitialized (%s); "
                        "falling back to AutoModelForImageTextToText + visual prune",
                        len(missing),
                        missing[:3],
                    )
                    del cand
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                else:
                    model = cand
            except Exception as e:
                errors.append(f"AutoModelForCausalLM(text_config): {e!r}")
                log.warning("text-only load failed (%r); falling back to the full multimodal class", e)
        if model is None:
            try:
                from transformers import AutoModelForImageTextToText

                res = AutoModelForImageTextToText.from_pretrained(
                    name, config=config, output_loading_info=True, **kwargs
                )
                model, linfo = res if isinstance(res, tuple) else (res, {})
                missing = _missing_decoder_keys(linfo)
                if missing:
                    raise RuntimeError(
                        f"AutoModelForImageTextToText left {len(missing)} language keys missing: {missing[:5]}"
                    )
                if text_only:
                    _prune_visual_tower(model)
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
    quant_info = install_quantized_autograd_bridge(model)
    parts = inspect_model(model)
    info = param_report(model)
    info["quantization"] = quant_info
    log.info("%s", summarize(parts))
    for k, v in info.items():
        log.info("  %s: %s", k, v)
    if quant_info.get("is_quantized"):
        log.info(
            "quantized base model ready: method=%s format=%s bridged=%d/%d modules "
            "(linear=%d, moe=%d, native_grad_ok=%d)",
            quant_info.get("quant_method"),
            quant_info.get("format"),
            quant_info.get("bridged_total", 0),
            quant_info.get("total_quantized_modules", 0),
            quant_info.get("bridged_linear_count", 0),
            quant_info.get("bridged_moe_count", 0),
            quant_info.get("native_grad_ok_count", 0),
        )
    elif info["quantized_module_types"]:
        log.warning("quantized modules present %s: run `qloop probe --backward` once to confirm gradients flow "
                    "through frozen quantized experts on this transformers/kernel version",
                    info["quantized_module_types"])
    return model, tok, info
