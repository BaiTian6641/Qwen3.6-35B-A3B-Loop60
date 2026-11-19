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
import importlib.util
import itertools
import json
import logging
import math
import os
import re
import struct
import sys
import time
from collections import Counter
from collections.abc import Iterable
from typing import Any

import torch
from torch import nn

from ..debug import cuda_mem_brief, require, stage
from .introspect import find_text_model, inspect_model, summarize
from .quant_autograd import (
    FP8_DTYPES,
    _register_repacked_tensor,
    _remove_module_tensor_attr,
    dequantize_fp8_tensor,
    dequantize_nvfp4_tensor,
    install_quantized_autograd_bridge,
    is_quantized_module,
    repack_decompressed_nvfp4_moe_experts,
)

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


def _param_kind_for_name(name: str) -> str:
    return (
        "experts" if ".experts." in name else "router" if name.endswith("mlp.gate.weight") else
        "shared_expert" if "shared_expert" in name else "linear_attn" if "linear_attn" in name else
        "self_attn" if "self_attn" in name else "embed" if "embed" in name else
        "lm_head" if "lm_head" in name else "visual" if "visual" in name else "other"
    )


def param_report(model: nn.Module) -> dict[str, Any]:
    by_dtype: Counter = Counter()
    by_kind: Counter = Counter()
    trainable = 0
    seen_ids: set[int] = set()
    for name, p in model.named_parameters():
        seen_ids.add(id(p))
        by_dtype[str(p.dtype).replace("torch.", "")] += p.numel()
        by_kind[_param_kind_for_name(name)] += p.numel()
        trainable += p.numel() if p.requires_grad else 0

    quant_mod_names = {mname for mname, m in model.named_modules() if is_quantized_module(m)}
    non_float_quant_dtypes = (*FP8_DTYPES, torch.uint8, torch.int8, torch.int16, torch.int32)
    for bname, buf in model.named_buffers():
        if id(buf) in seen_ids:
            continue
        parent = bname.rsplit(".", 1)[0] if "." in bname else ""
        if parent in quant_mod_names and buf.dtype in non_float_quant_dtypes:
            seen_ids.add(id(buf))
            by_dtype[str(buf.dtype).replace("torch.", "")] += buf.numel()
            by_kind[_param_kind_for_name(bname)] += buf.numel()

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


def _is_package_installed(pkg_name: str) -> bool:
    mod = sys.modules.get(pkg_name)
    if mod is not None and getattr(mod, "__spec__", None) is None:
        return True
    if pkg_name in sys.modules and mod is None:
        return False
    try:
        return importlib.util.find_spec(pkg_name) is not None
    except (ImportError, ValueError):
        return mod is not None


def _is_deferred_ct_decompress_hook(hook: Any) -> bool:
    cur = hook
    for _ in range(4):
        if hasattr(cur, "func"):
            cur = cur.func
        elif hasattr(cur, "__wrapped__"):
            cur = cur.__wrapped__
        elif hasattr(cur, "__func__"):
            cur = cur.__func__
        else:
            break
    for obj in (hook, cur, type(hook), type(cur)):
        name = str(getattr(obj, "__name__", "") or "").lower()
        qual = str(getattr(obj, "__qualname__", "") or "").lower()
        mod_name = str(getattr(obj, "__module__", "") or "").lower()
        if "ct_decompress" in name or "ct_decompress" in qual:
            return True
        if "decompress_hook" in name or "decompress_hook" in qual:
            return True
        if mod_name.startswith("compressed_tensors") and "compressor" in mod_name:
            return True
    return False


def _has_deferred_ct_decompress_hook(model: nn.Module) -> bool:
    for sub in model.modules():
        pre_hooks = getattr(sub, "_forward_pre_hooks", None)
        if pre_hooks:
            for hook in pre_hooks.values():
                if _is_deferred_ct_decompress_hook(hook):
                    return True
    return False


def _flush_deferred_quant_hooks(model: nn.Module) -> bool:
    """Trigger one-shot `compressed-tensors` `ct_decompress_hook` on the pristine un-patched model."""
    if not _has_deferred_ct_decompress_hook(model):
        return False

    first_dev = torch.device("cpu")
    for t in itertools.chain(model.parameters(), model.buffers()):
        if isinstance(t, torch.Tensor) and t.device.type != "meta":
            first_dev = t.device
            break

    dummy_ids = torch.zeros((1, 2), dtype=torch.long, device=first_dev)
    with torch.no_grad():
        try:
            try:
                model(input_ids=dummy_ids, use_cache=False)
            except TypeError:
                model(dummy_ids)
        except Exception as exc:
            if _has_deferred_ct_decompress_hook(model):
                raise
            log.debug("dummy forward after ct_decompress_hook raised %r (hook already cleared)", exc)

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    log.info("flushed deferred compressed-tensors decompression hook(s) on %s %s", first_dev, cuda_mem_brief("| "))
    return True


# ============================================================================ direct zero-spike safetensors loader

_ST_DTYPE_FROM_CODE: dict[str, torch.dtype] = {
    "F64": torch.float64,
    "F32": torch.float32,
    "F16": torch.float16,
    "BF16": torch.bfloat16,
    "I64": torch.int64,
    "I32": torch.int32,
    "I16": torch.int16,
    "I8": torch.int8,
    "U8": torch.uint8,
    "BOOL": torch.bool,
}
for _st_code, _st_attr in (
    ("F8_E4M3", "float8_e4m3fn"),
    ("F8_E5M2", "float8_e5m2"),
    ("F8_E4M3FNUZ", "float8_e4m3fnuz"),
    ("F8_E5M2FNUZ", "float8_e5m2fnuz"),
):
    if hasattr(torch, _st_attr):
        _ST_DTYPE_FROM_CODE[_st_code] = getattr(torch, _st_attr)

_IGNORED_DIRECT_SCALAR_ATTRS: frozenset[str] = frozenset(
    {
        "input_global_scale",
        "input_scale",
        "activation_scale",
        "k_scale",
        "v_scale",
        "q_scale",
        "o_scale",
        "weight_shape",
        "weight_zero_point",
    }
)

_LINEAR_PART_ATTRS: frozenset[str] = frozenset(
    {
        "weight",
        "weight_packed",
        "weight_scale",
        "weight_scale_inv",
        "weight_global_scale",
        "weight_scale_2",
        "bias",
    }
)

_EXPERT_2D_KEY_RE = re.compile(
    r"^(?P<experts_path>.+\.mlp\.experts)\.(?P<eid>\d+)\."
    r"(?P<proj>gate_proj|up_proj|down_proj)\.(?P<attr>[A-Za-z0-9_]+)$"
)
_LAYER_IDX_RE = re.compile(r"(?:^|\.)layers\.(\d+)(?:\.|$)")


class _NoSafetensorsFoundError(FileNotFoundError):
    """Raised when no `.safetensors` files can be resolved for a repository/directory."""


class _FallbackSafeOpenReader:
    """Seek-based `.safetensors` reader used when `safetensors` C/Rust package is not installed."""

    def __init__(self, path: str) -> None:
        self.path = path
        self._fp: Any = None
        self._data_base: int = 0
        self._entries: dict[str, dict[str, Any]] = {}

    def __enter__(self) -> _FallbackSafeOpenReader:
        self._fp = open(self.path, "rb")
        hdr_len_bytes = self._fp.read(8)
        require(len(hdr_len_bytes) == 8, f"corrupt safetensors file (short header): {self.path!r}")
        (hdr_len,) = struct.unpack("<Q", hdr_len_bytes)
        hdr_raw = self._fp.read(hdr_len)
        require(len(hdr_raw) == hdr_len, f"corrupt safetensors file (truncated header): {self.path!r}")
        header = json.loads(hdr_raw.decode("utf-8"))
        self._data_base = 8 + int(hdr_len)
        self._entries = {k: v for k, v in header.items() if k != "__metadata__"}
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if self._fp is not None:
            self._fp.close()
            self._fp = None

    def keys(self) -> list[str]:
        return list(self._entries.keys())

    def get_tensor(self, key: str) -> torch.Tensor:
        meta = self._entries[key]
        dt = _ST_DTYPE_FROM_CODE.get(str(meta["dtype"]))
        if dt is None:
            raise ValueError(f"unsupported safetensors dtype {meta['dtype']!r} for key {key!r}")
        shape = [int(d) for d in meta["shape"]]
        beg, end = int(meta["data_offsets"][0]), int(meta["data_offsets"][1])
        nbytes = end - beg
        if nbytes == 0:
            return torch.empty(shape, dtype=dt)
        assert self._fp is not None
        self._fp.seek(self._data_base + beg)
        raw = bytearray(self._fp.read(nbytes))
        require(len(raw) == nbytes, f"corrupt safetensors payload for {key!r} in {self.path!r}")
        return torch.frombuffer(raw, dtype=torch.uint8).view(dt).reshape(shape)


def _open_safetensors_shard(path: str) -> Any:
    try:
        from safetensors import safe_open  # type: ignore[import-not-found]

        return safe_open(path, framework="pt", device="cpu")
    except ImportError:
        return _FallbackSafeOpenReader(path)


def _is_visual_key(key: str) -> bool:
    if key in ("visual", "model.visual"):
        return True
    return key.startswith(("visual.", "model.visual."))


def _is_mtp_key(key: str) -> bool:
    if key in ("mtp", "model.mtp", "language_model.mtp", "model.language_model.mtp"):
        return True
    return key.startswith(
        (
            "mtp.",
            "model.mtp.",
            "language_model.mtp.",
            "model.language_model.mtp.",
        )
    )


def _is_visual_or_mtp_key(key: str) -> bool:
    return _is_visual_key(key) or _is_mtp_key(key)


def _should_skip_index_key(key: str, *, text_only: bool, skip_mtp: bool = True) -> bool:
    if skip_mtp and _is_mtp_key(key):
        return True
    return bool(text_only and _is_visual_key(key))


def _normalize_direct_tensor_key(
    raw_key: str,
    *,
    text_only: bool,
    skip_mtp: bool = True,
) -> str | None:
    if skip_mtp and _is_mtp_key(raw_key):
        return None
    if text_only and _is_visual_key(raw_key):
        return None
    key = raw_key
    if text_only:
        if key.startswith("model.language_model.lm_head."):
            key = "lm_head." + key[len("model.language_model.lm_head."):]
        elif key.startswith("language_model.lm_head."):
            key = "lm_head." + key[len("language_model.lm_head."):]
        elif key.startswith("model.language_model."):
            key = "model." + key[len("model.language_model."):]
        elif key.startswith("language_model.model."):
            key = "model." + key[len("language_model.model."):]
        elif key.startswith("language_model."):
            key = "model." + key[len("language_model."):]
    leaf = key.rsplit(".", 1)[-1]
    if leaf in _IGNORED_DIRECT_SCALAR_ATTRS:
        return None
    return key


def _extract_quant_method(cfg: Any) -> str | None:
    if cfg is None:
        return None
    qcfg = getattr(cfg, "quantization_config", None)
    if qcfg is None:
        return None
    qm = qcfg.get("quant_method") if isinstance(qcfg, dict) else getattr(qcfg, "quant_method", None)
    return str(qm).strip().lower() if qm else None


def _resolve_direct_safetensors_shards(
    name: str | os.PathLike[str],
    *,
    text_only: bool,
    skip_mtp: bool = True,
    revision: str | None = None,
    token: str | None = None,
) -> list[str]:
    path_str = os.fspath(name)
    if os.path.isdir(path_str):
        index_path = os.path.join(path_str, "model.safetensors.index.json")
        single_path = os.path.join(path_str, "model.safetensors")
        if os.path.isfile(index_path):
            with open(index_path, encoding="utf-8") as f:
                idx_data = json.load(f)
            weight_map = dict(idx_data.get("weight_map") or {})
            needed_files = list(
                dict.fromkeys(
                    sfile
                    for k, sfile in weight_map.items()
                    if not _should_skip_index_key(k, text_only=text_only, skip_mtp=skip_mtp)
                )
            )
            require(bool(needed_files), f"no language shards remained after filtering {index_path!r}")
            local_shards: list[str] = []
            for sf in needed_files:
                sp = os.path.join(path_str, sf)
                require(os.path.isfile(sp), f"missing required safetensors shard {sp!r}")
                local_shards.append(sp)
            return local_shards
        if os.path.isfile(single_path):
            return [single_path]
        found = sorted(fn for fn in os.listdir(path_str) if fn.endswith(".safetensors"))
        if found:
            return [os.path.join(path_str, fn) for fn in found]
        raise _NoSafetensorsFoundError(f"no `.safetensors` files found in local directory {path_str!r}")

    try:
        import huggingface_hub  # type: ignore[import-not-found]
        from huggingface_hub import hf_hub_download  # type: ignore[import-not-found]
    except ImportError as exc:
        raise _NoSafetensorsFoundError(
            f"huggingface_hub is required to download safetensors shards for {path_str!r}"
        ) from exc
    snapshot_download = getattr(huggingface_hub, "snapshot_download", None)

    dl_kw: dict[str, Any] = {"repo_id": path_str, "revision": revision}
    if token is not None:
        dl_kw["token"] = token

    index_local: str | None = None
    try:
        index_local = hf_hub_download(filename="model.safetensors.index.json", **dl_kw)
    except Exception:
        index_local = None

    if index_local is not None and os.path.isfile(index_local):
        with open(index_local, encoding="utf-8") as f:
            idx_data = json.load(f)
        weight_map = dict(idx_data.get("weight_map") or {})
        all_shards = list(dict.fromkeys(weight_map.values()))
        needed_files = list(
            dict.fromkeys(
                sfile
                for k, sfile in weight_map.items()
                if not _should_skip_index_key(k, text_only=text_only, skip_mtp=skip_mtp)
            )
        )
        require(bool(needed_files), f"no language shards remained after filtering {path_str!r}")
        skipped = len(all_shards) - len(needed_files)
        if skipped > 0:
            log.info(
                "direct safetensors loader: downloading %d/%d shards (skipped %d visual/MTP-only shard(s))",
                len(needed_files),
                len(all_shards),
                skipped,
            )
        if callable(snapshot_download):
            try:
                snap_dir = snapshot_download(allow_patterns=needed_files, **dl_kw)
                resolved = [os.path.join(snap_dir, sf) for sf in needed_files]
                if all(os.path.isfile(p) for p in resolved):
                    return resolved
            except Exception:
                pass
        return [hf_hub_download(filename=sf, **dl_kw) for sf in needed_files]

    try:
        single_local = hf_hub_download(filename="model.safetensors", **dl_kw)
        return [single_local]
    except Exception as exc:
        raise _NoSafetensorsFoundError(
            f"could not locate `model.safetensors.index.json` or `model.safetensors` for {path_str!r}: {exc!r}"
        ) from exc


def _strip_quantization_config_copy(cfg: Any) -> tuple[Any, Any]:
    orig_qcfg = copy.deepcopy(getattr(cfg, "quantization_config", None))
    clean_cfg = copy.deepcopy(cfg)
    if hasattr(clean_cfg, "quantization_config"):
        try:
            clean_cfg.quantization_config = None
        except Exception:
            pass
    if hasattr(clean_cfg, "__dict__") and "quantization_config" in clean_cfg.__dict__:
        clean_cfg.__dict__["quantization_config"] = None
    sub_tc = getattr(clean_cfg, "text_config", None)
    if sub_tc is not None and sub_tc is not clean_cfg:
        if hasattr(sub_tc, "quantization_config"):
            try:
                sub_tc.quantization_config = None
            except Exception:
                pass
        if hasattr(sub_tc, "__dict__") and "quantization_config" in sub_tc.__dict__:
            sub_tc.__dict__["quantization_config"] = None
    return clean_cfg, orig_qcfg


def _install_cross_device_align_hook(mod: nn.Module, target_dev: torch.device) -> None:
    def _align_pre_hook(_m: nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any]):
        changed = False
        new_args = []
        for a in args:
            if isinstance(a, torch.Tensor) and a.device != target_dev:
                new_args.append(a.to(target_dev))
                changed = True
            else:
                new_args.append(a)
        new_kw = kwargs
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor) and v.device != target_dev:
                if new_kw is kwargs:
                    new_kw = dict(kwargs)
                new_kw[k] = v.to(target_dev)
                changed = True
        return (tuple(new_args), new_kw) if changed else None

    mod.register_forward_pre_hook(_align_pre_hook, with_kwargs=True)


def _resolve_direct_device_plan(
    model: nn.Module,
    device_map: Any,
    *,
    num_layers: int,
    tie_word_embeddings: bool,
) -> tuple[torch.device, dict[int, torch.device], bool]:
    if isinstance(device_map, torch.device):
        if device_map.type == "cuda" and not torch.cuda.is_available():
            return torch.device("cpu"), {}, False
        return device_map, {}, False

    if isinstance(device_map, str) and device_map == "auto":
        if not torch.cuda.is_available() or torch.cuda.device_count() <= 0:
            return torch.device("cpu"), {}, False
        n_gpus = int(torch.cuda.device_count())
        if n_gpus == 1 or num_layers <= 1:
            return torch.device("cuda:0"), {}, False
        layer_devs: dict[int, torch.device] = {}
        for lid in range(num_layers):
            gpu_idx = min(n_gpus - 1, (lid * n_gpus) // max(num_layers, 1))
            layer_devs[lid] = torch.device(f"cuda:{gpu_idx}")
        first_dev = torch.device("cuda:0")
        last_dev = first_dev if tie_word_embeddings else torch.device(f"cuda:{n_gpus - 1}")
        tm = find_text_model(model)
        layers_list = getattr(tm, "layers", None)
        if isinstance(layers_list, nn.ModuleList):
            for lid, layer_mod in enumerate(layers_list):
                ldev = layer_devs.get(lid, first_dev)
                if lid == 0 or ldev != layer_devs.get(lid - 1, first_dev):
                    _install_cross_device_align_hook(layer_mod, ldev)
        norm_mod = getattr(tm, "norm", None)
        if isinstance(norm_mod, nn.Module):
            _install_cross_device_align_hook(norm_mod, last_dev)
        head_mod = getattr(model, "lm_head", None)
        if isinstance(head_mod, nn.Module):
            _install_cross_device_align_hook(head_mod, last_dev)
        return first_dev, layer_devs, True

    resolved = resolve_device(device_map)
    return torch.device(resolved), {}, False


def _device_for_module_path(
    mod_path: str,
    default_dev: torch.device,
    layer_devs: dict[int, torch.device],
    *,
    is_multi_gpu: bool,
    tie_word_embeddings: bool,
) -> torch.device:
    if not layer_devs:
        return default_dev
    m = _LAYER_IDX_RE.search(mod_path)
    if m is not None:
        return layer_devs.get(int(m.group(1)), default_dev)
    if (
        is_multi_gpu
        and not tie_word_embeddings
        and (mod_path in ("lm_head", "model.norm", "norm") or mod_path.endswith((".lm_head", ".norm")))
    ):
        return max(layer_devs.values(), key=lambda d: d.index or 0)
    return default_dev


def _reinit_rotary_and_meta_buffers(
    model: nn.Module,
    fallback_cfg: Any,
    *,
    default_dev: torch.device,
    layer_devs: dict[int, torch.device],
    is_multi_gpu: bool,
    tie_word_embeddings: bool,
) -> None:
    """Materialize non-persistent rotary embeddings and any `meta` buffers onto target devices."""
    for mod_path, sub in model.named_modules():
        is_rotary = (
            "inv_freq" in sub._buffers
            or hasattr(sub, "inv_freq")
            or "rotary" in type(sub).__name__.lower()
        )
        if not is_rotary:
            continue
        mod_dev = _device_for_module_path(
            mod_path,
            default_dev,
            layer_devs,
            is_multi_gpu=is_multi_gpu,
            tie_word_embeddings=tie_word_embeddings,
        )
        rot_cfg = getattr(sub, "config", None) or getattr(fallback_cfg, "text_config", fallback_cfg)
        rot_cls = type(sub)
        fresh: nn.Module | None = None
        inv_buf = sub._buffers.get("inv_freq")
        inferred_dim = int(inv_buf.numel() * 2) if isinstance(inv_buf, torch.Tensor) and inv_buf.ndim == 1 else None
        cand_calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = [
            ((rot_cfg,), {}),
            ((), {"config": rot_cfg}),
            ((rot_cfg,), {"device": "cpu"}),
        ]
        if inferred_dim is not None and inferred_dim > 0:
            cand_calls.append(((inferred_dim,), {}))
        with torch.device("cpu"):
            for r_args, r_kw in cand_calls:
                try:
                    fresh = rot_cls(*r_args, **r_kw)
                    break
                except TypeError:
                    continue
        if fresh is not None:
            for bname, bval in fresh._buffers.items():
                sub._buffers[bname] = bval.to(device=mod_dev) if isinstance(bval, torch.Tensor) else bval
            if hasattr(fresh, "_non_persistent_buffers_set"):
                sub._non_persistent_buffers_set = set(fresh._non_persistent_buffers_set)
            for k, v in fresh.__dict__.items():
                if k.startswith("_") or k in ("training",):
                    continue
                if isinstance(v, torch.Tensor):
                    setattr(sub, k, v.to(device=mod_dev))
                else:
                    setattr(sub, k, copy.deepcopy(v))
        elif callable(getattr(sub, "rope_init_fn", None)):
            with torch.device("cpu"):
                inv_freq, attn_scaling = sub.rope_init_fn(rot_cfg, device=None)
            inv_dev = inv_freq.to(device=mod_dev)
            sub.register_buffer("inv_freq", inv_dev, persistent=False)
            if "original_inv_freq" in sub._buffers or hasattr(sub, "original_inv_freq"):
                sub.register_buffer("original_inv_freq", inv_dev.clone(), persistent=False)
            sub.attention_scaling = attn_scaling  # type: ignore[attr-defined]
        elif inferred_dim is not None and inferred_dim > 0:
            raw_rp = getattr(rot_cfg, "rope_parameters", None)
            rp_flat = (
                raw_rp.get("full_attention")
                if isinstance(raw_rp, dict) and isinstance(raw_rp.get("full_attention"), dict)
                else (raw_rp if isinstance(raw_rp, dict) else {})
            )
            theta_val = (
                getattr(sub, "theta", None)
                or getattr(rot_cfg, "rope_theta", None)
                or rp_flat.get("rope_theta")
                or 10000.0
            )
            theta = float(theta_val)
            inv_dev = 1.0 / (
                theta ** (torch.arange(0, inferred_dim, 2, dtype=torch.float32, device=mod_dev) / float(inferred_dim))
            )
            sub.register_buffer("inv_freq", inv_dev, persistent=False)


class _LayerMoEBuilder:
    """Streaming 2D-per-expert -> 3D-fused MoE assembler supporting NVFP4, FP8, and dense experts."""

    def __init__(
        self,
        experts_path: str,
        experts_mod: nn.Module,
        layer_dev: torch.device,
        target_dtype: torch.dtype,
    ) -> None:
        self.experts_path = experts_path
        self.experts_mod = experts_mod
        self.layer_dev = layer_dev
        self.target_dtype = target_dtype
        gu_meta = getattr(experts_mod, "gate_up_proj", None)
        dn_meta = getattr(experts_mod, "down_proj", None)
        require(
            isinstance(gu_meta, torch.Tensor)
            and isinstance(dn_meta, torch.Tensor)
            and gu_meta.ndim == 3
            and dn_meta.ndim == 3,
            f"expected 3D meta gate_up_proj/down_proj on {experts_path!r}",
        )
        assert isinstance(gu_meta, torch.Tensor) and isinstance(dn_meta, torch.Tensor)
        self.num_experts = int(gu_meta.shape[0])
        self.two_i = int(gu_meta.shape[1])
        self.i_dim = self.two_i // 2
        self.h_dim = int(gu_meta.shape[2])
        require(
            tuple(dn_meta.shape) == (self.num_experts, self.h_dim, self.i_dim),
            f"unexpected down_proj shape {tuple(dn_meta.shape)} on {experts_path!r}",
        )
        self.mode: str | None = None
        self.saw_weight_scale_2 = False
        self.seen_weights: set[tuple[int, str]] = set()
        self.seen_scales: set[tuple[int, str]] = set()
        self.seen_global_scales: set[tuple[int, str]] = set()

        # NVFP4 buffers
        self.gu_packed: torch.Tensor | None = None
        self.dn_packed: torch.Tensor | None = None
        self.gu_scale: torch.Tensor | None = None
        self.dn_scale: torch.Tensor | None = None
        self.gu_gs: torch.Tensor | None = None
        self.dn_gs: torch.Tensor | None = None
        self.n_grp_gu: int | None = None
        self.n_grp_dn: int | None = None

        # FP8 / dense buffers
        self.gu_weight: torch.Tensor | None = None
        self.dn_weight: torch.Tensor | None = None
        self.gu_scale_inv: torch.Tensor | None = None
        self.dn_scale_inv: torch.Tensor | None = None
        self.s0_gu: int | None = None
        self.s1_gu: int | None = None

    def _set_mode(self, new_mode: str) -> None:
        if self.mode is None:
            self.mode = new_mode
            _remove_module_tensor_attr(self.experts_mod, "gate_up_proj")
            _remove_module_tensor_attr(self.experts_mod, "down_proj")
            return
        require(
            self.mode == new_mode,
            f"conflicting MoE quantization formats inside {self.experts_path!r}: {self.mode!r} vs {new_mode!r}",
        )

    def _ensure_nvfp4_core(self) -> None:
        self._set_mode("nvfp4")
        if self.gu_packed is None:
            require(self.h_dim % 2 == 0 and self.i_dim % 2 == 0, f"odd NVFP4 dims on {self.experts_path!r}")
            self.gu_packed = torch.empty(
                (self.num_experts, self.two_i, self.h_dim // 2), dtype=torch.uint8, device=self.layer_dev
            )
            self.dn_packed = torch.empty(
                (self.num_experts, self.h_dim, self.i_dim // 2), dtype=torch.uint8, device=self.layer_dev
            )
            self.gu_gs = torch.ones((self.num_experts, self.two_i, 1), dtype=torch.float32, device=self.layer_dev)
            self.dn_gs = torch.ones((self.num_experts, 1, 1), dtype=torch.float32, device=self.layer_dev)

    def _ensure_fp8_weights(self, w_dtype: torch.dtype) -> None:
        self._set_mode("fp8")
        if self.gu_weight is None:
            self.gu_weight = torch.empty(
                (self.num_experts, self.two_i, self.h_dim), dtype=w_dtype, device=self.layer_dev
            )
            self.dn_weight = torch.empty(
                (self.num_experts, self.h_dim, self.i_dim), dtype=w_dtype, device=self.layer_dev
            )

    def _ensure_dense_weights(self) -> None:
        self._set_mode("dense")
        if self.gu_weight is None:
            self.gu_weight = torch.empty(
                (self.num_experts, self.two_i, self.h_dim), dtype=self.target_dtype, device=self.layer_dev
            )
            self.dn_weight = torch.empty(
                (self.num_experts, self.h_dim, self.i_dim), dtype=self.target_dtype, device=self.layer_dev
            )

    @torch.no_grad()
    def ingest(self, eid: int, proj: str, attr: str, tensor: torch.Tensor) -> None:
        require(0 <= eid < self.num_experts, f"expert index {eid} out of range [0, {self.num_experts})")

        if attr in ("weight_packed", "weight_global_scale", "weight_scale_2") or (
            attr == "weight_scale" and tensor.dtype in FP8_DTYPES
        ) or (attr == "weight" and tensor.dtype in (torch.uint8, torch.int8)):
            self._ensure_nvfp4_core()
            assert self.gu_packed is not None and self.dn_packed is not None
            assert self.gu_gs is not None and self.dn_gs is not None

            if attr in ("weight_packed", "weight"):
                u8 = tensor if tensor.dtype == torch.uint8 else tensor.contiguous().view(torch.uint8)
                if proj == "gate_proj":
                    require(tuple(u8.shape) == (self.i_dim, self.h_dim // 2), f"bad shape {tuple(u8.shape)}")
                    self.gu_packed[eid, : self.i_dim, :].copy_(u8.to(device=self.layer_dev))
                elif proj == "up_proj":
                    require(tuple(u8.shape) == (self.i_dim, self.h_dim // 2), f"bad shape {tuple(u8.shape)}")
                    self.gu_packed[eid, self.i_dim :, :].copy_(u8.to(device=self.layer_dev))
                else:
                    require(tuple(u8.shape) == (self.h_dim, self.i_dim // 2), f"bad shape {tuple(u8.shape)}")
                    self.dn_packed[eid, :, :].copy_(u8.to(device=self.layer_dev))
                self.seen_weights.add((eid, proj))
                return

            if attr == "weight_scale":
                require(tensor.ndim == 2, f"expected 2D NVFP4 weight_scale, got {tuple(tensor.shape)}")
                if proj in ("gate_proj", "up_proj"):
                    n_grp = int(tensor.shape[1])
                    require(int(tensor.shape[0]) == self.i_dim, f"bad gate/up scale shape {tuple(tensor.shape)}")
                    if self.gu_scale is None:
                        self.n_grp_gu = n_grp
                        self.gu_scale = torch.empty(
                            (self.num_experts, self.two_i, n_grp), dtype=tensor.dtype, device=self.layer_dev
                        )
                    require(n_grp == self.n_grp_gu, f"inconsistent n_grp {n_grp} vs {self.n_grp_gu}")
                    if proj == "gate_proj":
                        self.gu_scale[eid, : self.i_dim, :].copy_(tensor.to(device=self.layer_dev))
                    else:
                        self.gu_scale[eid, self.i_dim :, :].copy_(tensor.to(device=self.layer_dev))
                else:
                    n_grp = int(tensor.shape[1])
                    require(int(tensor.shape[0]) == self.h_dim, f"bad down scale shape {tuple(tensor.shape)}")
                    if self.dn_scale is None:
                        self.n_grp_dn = n_grp
                        self.dn_scale = torch.empty(
                            (self.num_experts, self.h_dim, n_grp), dtype=tensor.dtype, device=self.layer_dev
                        )
                    require(n_grp == self.n_grp_dn, f"inconsistent down n_grp {n_grp} vs {self.n_grp_dn}")
                    self.dn_scale[eid, :, :].copy_(tensor.to(device=self.layer_dev))
                self.seen_scales.add((eid, proj))
                return

            if attr in ("weight_global_scale", "weight_scale_2"):
                if attr == "weight_scale_2":
                    self.saw_weight_scale_2 = True
                gs_f32 = tensor.to(device=self.layer_dev, dtype=torch.float32)
                if proj == "gate_proj":
                    if gs_f32.numel() == 1:
                        self.gu_gs[eid, : self.i_dim, 0].fill_(gs_f32.reshape(()))
                    else:
                        self.gu_gs[eid, : self.i_dim, :].copy_(gs_f32.reshape(self.i_dim, 1))
                elif proj == "up_proj":
                    if gs_f32.numel() == 1:
                        self.gu_gs[eid, self.i_dim :, 0].fill_(gs_f32.reshape(()))
                    else:
                        self.gu_gs[eid, self.i_dim :, :].copy_(gs_f32.reshape(self.i_dim, 1))
                elif gs_f32.numel() == 1:
                    if self.dn_gs.shape[1] == 1:
                        self.dn_gs[eid, 0, 0].fill_(gs_f32.reshape(()))
                    else:
                        self.dn_gs[eid, :, 0].fill_(gs_f32.reshape(()))
                else:
                    if self.dn_gs.shape[1] == 1:
                        expanded = self.dn_gs.repeat(1, self.h_dim, 1)
                        self.dn_gs = expanded
                    self.dn_gs[eid, :, :].copy_(gs_f32.reshape(self.h_dim, 1))
                self.seen_global_scales.add((eid, proj))
                return

        if attr in ("weight_scale_inv", "weight_scale") or (attr == "weight" and tensor.dtype in FP8_DTYPES):
            fp8_dt = tensor.dtype if (attr == "weight" and tensor.dtype in FP8_DTYPES) else FP8_DTYPES[0]
            self._ensure_fp8_weights(fp8_dt)
            assert self.gu_weight is not None and self.dn_weight is not None

            if attr == "weight":
                if proj == "gate_proj":
                    require(
                        tuple(tensor.shape) == (self.i_dim, self.h_dim),
                        f"bad FP8 gate shape {tuple(tensor.shape)}",
                    )
                    self.gu_weight[eid, : self.i_dim, :].copy_(tensor.to(device=self.layer_dev))
                elif proj == "up_proj":
                    require(
                        tuple(tensor.shape) == (self.i_dim, self.h_dim),
                        f"bad FP8 up shape {tuple(tensor.shape)}",
                    )
                    self.gu_weight[eid, self.i_dim :, :].copy_(tensor.to(device=self.layer_dev))
                else:
                    require(
                        tuple(tensor.shape) == (self.h_dim, self.i_dim),
                        f"bad FP8 down shape {tuple(tensor.shape)}",
                    )
                    self.dn_weight[eid, :, :].copy_(tensor.to(device=self.layer_dev))
                self.seen_weights.add((eid, proj))
                return

            s_t = tensor
            if proj in ("gate_proj", "up_proj"):
                if s_t.numel() == 1:
                    s_t = s_t.reshape(1, 1).expand(self.i_dim, 1).contiguous()
                elif s_t.ndim == 1:
                    s_t = s_t.reshape(-1, 1) if s_t.shape[0] == self.i_dim else s_t.reshape(1, -1)
                s0, s1 = int(s_t.shape[0]), int(s_t.shape[1])
                if self.gu_scale_inv is None:
                    self.s0_gu, self.s1_gu = s0, s1
                    self.gu_scale_inv = torch.empty(
                        (self.num_experts, 2 * s0, s1), dtype=s_t.dtype, device=self.layer_dev
                    )
                require((s0, s1) == (self.s0_gu, self.s1_gu), f"inconsistent FP8 scale shape {(s0, s1)}")
                if proj == "gate_proj":
                    self.gu_scale_inv[eid, :s0, :].copy_(s_t.to(device=self.layer_dev))
                else:
                    self.gu_scale_inv[eid, s0:, :].copy_(s_t.to(device=self.layer_dev))
            else:
                if s_t.numel() == 1:
                    s_t = s_t.reshape(1, 1)
                elif s_t.ndim == 1:
                    s_t = s_t.reshape(-1, 1) if s_t.shape[0] == self.h_dim else s_t.reshape(1, -1)
                s0_dn, s1_dn = int(s_t.shape[0]), int(s_t.shape[1])
                if self.dn_scale_inv is None:
                    self.dn_scale_inv = torch.empty(
                        (self.num_experts, s0_dn, s1_dn), dtype=s_t.dtype, device=self.layer_dev
                    )
                require(
                    tuple(self.dn_scale_inv.shape[1:]) == (s0_dn, s1_dn),
                    f"inconsistent FP8 down scale shape {(s0_dn, s1_dn)}",
                )
                self.dn_scale_inv[eid, :, :].copy_(s_t.to(device=self.layer_dev))
            self.seen_scales.add((eid, proj))
            return

        if attr == "weight" and tensor.is_floating_point():
            self._ensure_dense_weights()
            assert self.gu_weight is not None and self.dn_weight is not None
            t_cast = tensor.to(device=self.layer_dev, dtype=self.target_dtype)
            if proj == "gate_proj":
                self.gu_weight[eid, : self.i_dim, :].copy_(t_cast)
            elif proj == "up_proj":
                self.gu_weight[eid, self.i_dim :, :].copy_(t_cast)
            else:
                self.dn_weight[eid, :, :].copy_(t_cast)
            self.seen_weights.add((eid, proj))
            return

        raise RuntimeError(f"unsupported MoE expert tensor attribute {self.experts_path}.{eid}.{proj}.{attr}")

    def finalize(self) -> None:
        expected_pairs = {
            (eid, proj)
            for eid in range(self.num_experts)
            for proj in ("gate_proj", "up_proj", "down_proj")
        }
        missing_w = sorted(expected_pairs - self.seen_weights)
        require(
            not missing_w,
            f"incomplete MoE weights on {self.experts_path!r}: missing {len(missing_w)} slices ({missing_w[:5]})",
        )

        if self.mode == "nvfp4":
            missing_s = sorted(expected_pairs - self.seen_scales)
            missing_gs = sorted(expected_pairs - self.seen_global_scales)
            require(
                not missing_s and not missing_gs,
                f"incomplete NVFP4 scales on {self.experts_path!r}: "
                f"missing_scale={missing_s[:5]}, missing_global_scale={missing_gs[:5]}",
            )
            assert self.gu_packed is not None and self.dn_packed is not None
            assert self.gu_scale is not None and self.dn_scale is not None
            assert self.gu_gs is not None and self.dn_gs is not None
            assert self.n_grp_gu is not None and self.n_grp_gu > 0
            _register_repacked_tensor(self.experts_mod, "gate_up_proj_packed", self.gu_packed)
            _register_repacked_tensor(self.experts_mod, "gate_up_proj_scale", self.gu_scale)
            _register_repacked_tensor(self.experts_mod, "gate_up_proj_global_scale", self.gu_gs)
            _register_repacked_tensor(self.experts_mod, "down_proj_packed", self.dn_packed)
            _register_repacked_tensor(self.experts_mod, "down_proj_scale", self.dn_scale)
            _register_repacked_tensor(self.experts_mod, "down_proj_global_scale", self.dn_gs)
            gsize = self.h_dim // self.n_grp_gu if self.h_dim % self.n_grp_gu == 0 else 16
            self.experts_mod.group_size = int(gsize)  # type: ignore[attr-defined]
            self.experts_mod._qloop_nvfp4_scale_mode = (  # type: ignore[attr-defined]
                "multiplier" if self.saw_weight_scale_2 else "divisor"
            )
            return

        if self.mode == "fp8":
            missing_s = sorted(expected_pairs - self.seen_scales)
            require(
                not missing_s,
                f"incomplete FP8 scales on {self.experts_path!r}: missing {len(missing_s)} slices ({missing_s[:5]})",
            )
            assert self.gu_weight is not None and self.dn_weight is not None
            assert self.gu_scale_inv is not None and self.dn_scale_inv is not None
            self.experts_mod.register_parameter("gate_up_proj", nn.Parameter(self.gu_weight, requires_grad=False))
            self.experts_mod.register_parameter("down_proj", nn.Parameter(self.dn_weight, requires_grad=False))
            self.experts_mod.register_buffer("gate_up_proj_scale_inv", self.gu_scale_inv)
            self.experts_mod.register_buffer("down_proj_scale_inv", self.dn_scale_inv)
            if (
                self.s0_gu is not None
                and self.s1_gu is not None
                and self.s0_gu not in (1, self.i_dim)
                and self.s1_gu not in (1, self.h_dim)
            ):
                self.experts_mod.weight_block_size = (  # type: ignore[attr-defined]
                    math.ceil(self.i_dim / self.s0_gu),
                    math.ceil(self.h_dim / self.s1_gu),
                )
            return

        assert self.gu_weight is not None and self.dn_weight is not None
        self.experts_mod.register_parameter("gate_up_proj", nn.Parameter(self.gu_weight, requires_grad=False))
        self.experts_mod.register_parameter("down_proj", nn.Parameter(self.dn_weight, requires_grad=False))


def _linear_parts_complete(mod: nn.Module, parts: dict[str, torch.Tensor]) -> bool:
    need_bias = getattr(mod, "bias", None) is not None
    if need_bias and "bias" not in parts:
        return False
    if "weight_packed" in parts or (
        "weight" in parts and parts["weight"].dtype in (torch.uint8, torch.int8)
    ):
        return "weight_scale" in parts and ("weight_global_scale" in parts or "weight_scale_2" in parts)
    if "weight" in parts:
        w = parts["weight"]
        if w.dtype in FP8_DTYPES:
            return "weight_scale" in parts or "weight_scale_inv" in parts
        if w.is_floating_point():
            scale_keys = ("weight_scale", "weight_scale_inv", "weight_global_scale", "weight_scale_2")
            return not any(k in parts for k in scale_keys)
    return False


def _materialize_buffered_linear(
    mod_path: str,
    mod: nn.Module,
    parts: dict[str, torch.Tensor],
    *,
    mod_dev: torch.device,
    target_dtype: torch.dtype,
    meta_param_dtypes: dict[str, torch.dtype],
    keep_dense_quantized: bool,
    loaded_param_keys: set[str],
    replaced_param_keys: set[str],
) -> None:
    is_lm_head = mod_path == "lm_head" or mod_path.endswith(".lm_head")
    force_dequant = (not keep_dense_quantized) or is_lm_head
    w_key = f"{mod_path}.weight" if mod_path else "weight"
    b_key = f"{mod_path}.bias" if mod_path else "bias"
    meta_dt = meta_param_dtypes.get(w_key)
    out_dt = (
        torch.float32
        if (meta_dt == torch.float32 and target_dtype in (torch.bfloat16, torch.float16))
        else target_dtype
    )
    out_f = int(getattr(mod, "out_features", 0))
    in_f = int(getattr(mod, "in_features", 0))

    if "weight_packed" in parts or ("weight" in parts and parts["weight"].dtype in (torch.uint8, torch.int8)):
        wp = parts["weight_packed"] if "weight_packed" in parts else parts["weight"]
        ws = parts.get("weight_scale")
        has_s2 = "weight_scale_2" in parts
        wgs = parts.get("weight_scale_2") if has_s2 else parts.get("weight_global_scale")
        require(ws is not None, f"NVFP4 linear {mod_path!r} missing weight_scale")
        require(wgs is not None, f"NVFP4 linear {mod_path!r} missing weight_global_scale / weight_scale_2")
        assert ws is not None and wgs is not None
        u8 = wp if wp.dtype == torch.uint8 else wp.contiguous().view(torch.uint8)
        k_unpacked = int(u8.shape[-1]) * 2
        n_grp = int(ws.shape[-1]) if ws.ndim > 0 else 1
        gsize = k_unpacked // n_grp if (n_grp > 0 and k_unpacked % n_grp == 0) else 16
        if force_dequant:
            w_deq = dequantize_nvfp4_tensor(
                u8,
                ws,
                wgs,
                dtype=out_dt,
                group_size=gsize,
                global_scale_mode="multiplier" if has_s2 else "divisor",
            )
            if out_f > 0 and in_f > 0:
                require(tuple(w_deq.shape) == (out_f, in_f), f"shape mismatch on {w_key}: {tuple(w_deq.shape)}")
            mod.register_parameter("weight", nn.Parameter(w_deq.to(device=mod_dev, dtype=out_dt), requires_grad=False))
            loaded_param_keys.add(w_key)
        else:
            _remove_module_tensor_attr(mod, "weight")
            replaced_param_keys.add(w_key)
            _register_repacked_tensor(mod, "weight_packed", u8.to(device=mod_dev))
            _register_repacked_tensor(mod, "weight_scale", ws.to(device=mod_dev))
            gs_name = "weight_scale_2" if has_s2 else "weight_global_scale"
            _register_repacked_tensor(mod, gs_name, wgs.to(device=mod_dev, dtype=torch.float32))
            mod.group_size = int(gsize)  # type: ignore[attr-defined]
            mod._qloop_nvfp4_scale_mode = "multiplier" if has_s2 else "divisor"  # type: ignore[attr-defined]
    elif "weight" in parts and parts["weight"].dtype in FP8_DTYPES:
        wf = parts["weight"]
        ws = parts.get("weight_scale_inv") if "weight_scale_inv" in parts else parts.get("weight_scale")
        require(ws is not None, f"FP8 linear {mod_path!r} missing weight_scale / weight_scale_inv")
        assert ws is not None
        bsize: tuple[int, int] | None = None
        if (
            ws.ndim == 2
            and int(ws.shape[0]) not in (1, int(wf.shape[0]))
            and int(ws.shape[1]) not in (1, int(wf.shape[1]))
        ):
            bsize = (math.ceil(int(wf.shape[0]) / int(ws.shape[0])), math.ceil(int(wf.shape[1]) / int(ws.shape[1])))
        if force_dequant:
            w_deq = dequantize_fp8_tensor(wf, ws, dtype=out_dt, block_size=bsize)
            if out_f > 0 and in_f > 0:
                require(tuple(w_deq.shape) == (out_f, in_f), f"shape mismatch on {w_key}: {tuple(w_deq.shape)}")
            mod.register_parameter("weight", nn.Parameter(w_deq.to(device=mod_dev, dtype=out_dt), requires_grad=False))
            loaded_param_keys.add(w_key)
        else:
            mod.register_parameter("weight", nn.Parameter(wf.to(device=mod_dev), requires_grad=False))
            mod.register_buffer("weight_scale_inv", ws.to(device=mod_dev))
            if bsize is not None:
                mod.weight_block_size = bsize  # type: ignore[attr-defined]
            loaded_param_keys.add(w_key)
    elif "weight" in parts:
        w_plain = parts["weight"]
        if out_f > 0 and in_f > 0:
            require(tuple(w_plain.shape) == (out_f, in_f), f"shape mismatch on {w_key}: {tuple(w_plain.shape)}")
        mod.register_parameter("weight", nn.Parameter(w_plain.to(device=mod_dev, dtype=out_dt), requires_grad=False))
        loaded_param_keys.add(w_key)
    else:
        raise RuntimeError(f"linear module {mod_path!r} received incomplete tensors: {sorted(parts.keys())}")

    if "bias" in parts:
        b_dt = (
            torch.float32
            if (meta_param_dtypes.get(b_key) == torch.float32 and target_dtype in (torch.bfloat16, torch.float16))
            else target_dtype
        )
        mod.register_parameter("bias", nn.Parameter(parts["bias"].to(device=mod_dev, dtype=b_dt), requires_grad=False))
        loaded_param_keys.add(b_key)


def load_compressed_safetensors_direct(
    name: str | os.PathLike[str],
    *,
    config: Any = None,
    text_cfg: Any = None,
    text_only: bool = True,
    dtype: str | torch.dtype | None = torch.bfloat16,
    device_map: Any = "cuda:0",
    attn_implementation: str = "sdpa",
    keep_dense_quantized: bool = False,
    install_bridge: bool = True,
    revision: str | None = None,
    trust_remote_code: bool = False,
    token: str | None = None,
    _resolved_shards: list[str] | None = None,
) -> nn.Module:
    """Zero-spike direct `.safetensors` loader for `compressed-tensors` / `modelopt` Qwen3.6 models.

    Bypasses `transformers` `CompressedTensorsDequantize` (which silently drops `weight_global_scale`
    on 3D MoE experts, spikes VRAM to ~65 GiB, and fails on mixed NVFP4 + channel-FP8 expert layers).
    """
    import transformers
    from transformers import AutoConfig, AutoModelForCausalLM

    target_dtype = dtype if isinstance(dtype, torch.dtype) else resolve_dtype(dtype)
    common_cfg_kw: dict[str, Any] = {"revision": revision, "trust_remote_code": bool(trust_remote_code)}
    if token is not None:
        common_cfg_kw["token"] = token

    if config is None:
        config = AutoConfig.from_pretrained(name, **common_cfg_kw)
    has_vision_cfg = getattr(config, "vision_config", None) is not None
    effective_text_only = bool(text_only) or not has_vision_cfg
    if text_cfg is None:
        raw_tc = getattr(config, "text_config", config)
        text_cfg = _propagate_quantization_config_for_direct(config, raw_tc)

    shard_paths = (
        list(_resolved_shards)
        if _resolved_shards is not None
        else _resolve_direct_safetensors_shards(
            name,
            text_only=effective_text_only,
            skip_mtp=True,
            revision=revision,
            token=token,
        )
    )

    target_cfg = text_cfg if effective_text_only else config
    clean_cfg, orig_qcfg = _strip_quantization_config_copy(target_cfg)
    ver_head = str(getattr(transformers, "__version__", "5")).split(".", 1)[0]
    dtype_key = "dtype" if (ver_head.isdigit() and int(ver_head) >= 5) else "torch_dtype"

    with torch.device("meta"):
        if effective_text_only:
            try:
                model = AutoModelForCausalLM.from_config(
                    clean_cfg, attn_implementation=attn_implementation, **{dtype_key: target_dtype}
                )
            except TypeError:
                model = AutoModelForCausalLM.from_config(clean_cfg, attn_implementation=attn_implementation)
        else:
            from transformers import AutoModelForImageTextToText

            try:
                model = AutoModelForImageTextToText.from_config(
                    clean_cfg, attn_implementation=attn_implementation, **{dtype_key: target_dtype}
                )
            except TypeError:
                model = AutoModelForImageTextToText.from_config(clean_cfg, attn_implementation=attn_implementation)

    if orig_qcfg is not None:
        model.config.quantization_config = copy.deepcopy(orig_qcfg)
        tm_for_qcfg = find_text_model(model)
        if getattr(tm_for_qcfg, "config", None) is not None and tm_for_qcfg.config is not model.config:
            tm_for_qcfg.config.quantization_config = copy.deepcopy(orig_qcfg)

    meta_param_dtypes: dict[str, torch.dtype] = {k: p.dtype for k, p in model.named_parameters()}
    expected_param_keys: set[str] = set(meta_param_dtypes.keys())
    mod_map: dict[str, nn.Module] = dict(model.named_modules())
    has_mtp_mod = any(mp == "mtp" or mp.endswith(".mtp") for mp in mod_map)

    tm = find_text_model(model)
    num_layers = int(getattr(tm.config, "num_hidden_layers", len(getattr(tm, "layers", ()))))
    tie_we = bool(getattr(target_cfg, "tie_word_embeddings", False) or getattr(tm.config, "tie_word_embeddings", False))
    default_dev, layer_devs, is_multi_gpu = _resolve_direct_device_plan(
        model,
        device_map,
        num_layers=num_layers,
        tie_word_embeddings=tie_we,
    )

    _reinit_rotary_and_meta_buffers(
        model,
        clean_cfg,
        default_dev=default_dev,
        layer_devs=layer_devs,
        is_multi_gpu=is_multi_gpu,
        tie_word_embeddings=tie_we,
    )

    moe_builders: dict[str, _LayerMoEBuilder] = {}
    for mpath, sub in mod_map.items():
        if mpath.endswith(".mlp.experts"):
            ldev = _device_for_module_path(
                mpath,
                default_dev,
                layer_devs,
                is_multi_gpu=is_multi_gpu,
                tie_word_embeddings=tie_we,
            )
            moe_builders[mpath] = _LayerMoEBuilder(mpath, sub, ldev, target_dtype)

    pending_linears: dict[str, dict[str, torch.Tensor]] = {}
    loaded_param_keys: set[str] = set()
    replaced_param_keys: set[str] = set()
    direct_3d_moe_paths: set[str] = set()

    for spath in shard_paths:
        with _open_safetensors_shard(spath) as reader:
            shard_keys = list(reader.keys())
            for raw_key in shard_keys:
                key = _normalize_direct_tensor_key(
                    raw_key,
                    text_only=effective_text_only,
                    skip_mtp=not has_mtp_mod,
                )
                if key is None:
                    continue
                if not effective_text_only:
                    if key.startswith("visual.") and "visual" not in mod_map and "model.visual" in mod_map:
                        key = "model." + key
                    elif (
                        key.startswith("model.language_model.lm_head.")
                        and "model.language_model.lm_head" not in mod_map
                        and "lm_head" in mod_map
                    ):
                        key = "lm_head." + key[len("model.language_model.lm_head."):]
                    elif (
                        key.startswith("language_model.lm_head.")
                        and "language_model.lm_head" not in mod_map
                        and "lm_head" in mod_map
                    ):
                        key = "lm_head." + key[len("language_model.lm_head."):]

                m_exp = _EXPERT_2D_KEY_RE.match(key)
                if m_exp is not None:
                    exp_path = m_exp.group("experts_path")
                    eid = int(m_exp.group("eid"))
                    proj = m_exp.group("proj")
                    attr = m_exp.group("attr")
                    if attr in _IGNORED_DIRECT_SCALAR_ATTRS:
                        continue
                    builder = moe_builders.get(exp_path)
                    require(builder is not None, f"MoE module {exp_path!r} not found in skeleton")
                    assert builder is not None
                    tensor = reader.get_tensor(raw_key)
                    builder.ingest(eid, proj, attr, tensor)
                    del tensor
                    continue

                parent_path, sep, attr_name = key.rpartition(".")
                parent_mod = mod_map.get(parent_path) if sep else model
                require(parent_mod is not None, f"unexpected module path {parent_path!r} for tensor {key!r}")
                assert parent_mod is not None

                # Support pre-fused 3D MoE tensors if present
                if parent_path.endswith(".mlp.experts"):
                    tensor = reader.get_tensor(raw_key)
                    ldev = _device_for_module_path(
                        parent_path,
                        default_dev,
                        layer_devs,
                        is_multi_gpu=is_multi_gpu,
                        tie_word_embeddings=tie_we,
                    )
                    direct_3d_moe_paths.add(parent_path)
                    if attr_name in ("gate_up_proj_packed", "down_proj_packed"):
                        dense_nm = "gate_up_proj" if attr_name == "gate_up_proj_packed" else "down_proj"
                        _remove_module_tensor_attr(parent_mod, dense_nm)
                        replaced_param_keys.add(f"{parent_path}.{dense_nm}")
                        _register_repacked_tensor(parent_mod, attr_name, tensor.to(device=ldev))
                    elif attr_name in (
                        "gate_up_proj_scale",
                        "down_proj_scale",
                        "gate_up_proj_global_scale",
                        "down_proj_global_scale",
                        "gate_up_proj_scale_2",
                        "down_proj_scale_2",
                    ):
                        dt_s = (
                            torch.float32
                            if ("global_scale" in attr_name or "scale_2" in attr_name)
                            else tensor.dtype
                        )
                        _register_repacked_tensor(parent_mod, attr_name, tensor.to(device=ldev, dtype=dt_s))
                        if attr_name.endswith("_scale_2"):
                            parent_mod._qloop_nvfp4_scale_mode = "multiplier"  # type: ignore[attr-defined]
                        elif attr_name.endswith("_global_scale") and not hasattr(parent_mod, "_qloop_nvfp4_scale_mode"):
                            parent_mod._qloop_nvfp4_scale_mode = "divisor"  # type: ignore[attr-defined]
                    elif attr_name in ("gate_up_proj_scale_inv", "down_proj_scale_inv"):
                        parent_mod.register_buffer(attr_name, tensor.to(device=ldev))
                    elif attr_name in ("gate_up_proj", "down_proj"):
                        t_out = (
                            tensor.to(device=ldev)
                            if tensor.dtype in FP8_DTYPES
                            else tensor.to(device=ldev, dtype=target_dtype)
                        )
                        parent_mod.register_parameter(attr_name, nn.Parameter(t_out, requires_grad=False))
                        loaded_param_keys.add(key)
                    del tensor
                    continue

                is_linear_mod = (
                    isinstance(parent_mod, nn.Linear)
                    or (
                        not isinstance(parent_mod, nn.Embedding)
                        and isinstance(getattr(parent_mod, "in_features", None), int)
                        and isinstance(getattr(parent_mod, "out_features", None), int)
                    )
                )
                if is_linear_mod and attr_name in _LINEAR_PART_ATTRS:
                    tensor = reader.get_tensor(raw_key)
                    bucket = pending_linears.setdefault(parent_path, {})
                    bucket[attr_name] = tensor
                    if _linear_parts_complete(parent_mod, bucket):
                        mdev = _device_for_module_path(
                            parent_path,
                            default_dev,
                            layer_devs,
                            is_multi_gpu=is_multi_gpu,
                            tie_word_embeddings=tie_we,
                        )
                        _materialize_buffered_linear(
                            parent_path,
                            parent_mod,
                            bucket,
                            mod_dev=mdev,
                            target_dtype=target_dtype,
                            meta_param_dtypes=meta_param_dtypes,
                            keep_dense_quantized=keep_dense_quantized,
                            loaded_param_keys=loaded_param_keys,
                            replaced_param_keys=replaced_param_keys,
                        )
                        del pending_linears[parent_path]
                    del tensor
                    continue

                tensor = reader.get_tensor(raw_key)
                mdev = _device_for_module_path(
                    parent_path,
                    default_dev,
                    layer_devs,
                    is_multi_gpu=is_multi_gpu,
                    tie_word_embeddings=tie_we,
                )
                if attr_name in parent_mod._parameters:
                    orig_p = parent_mod._parameters[attr_name]
                    if orig_p is not None:
                        require(
                            tuple(tensor.shape) == tuple(orig_p.shape),
                            f"shape mismatch on {key}: expected {tuple(orig_p.shape)}, got {tuple(tensor.shape)}",
                        )
                    if tensor.is_floating_point() and tensor.dtype not in FP8_DTYPES:
                        meta_dt = meta_param_dtypes.get(key)
                        out_dt = (
                            torch.float32
                            if (meta_dt == torch.float32 and target_dtype in (torch.bfloat16, torch.float16))
                            else target_dtype
                        )
                        t_dev = tensor.to(device=mdev, dtype=out_dt)
                    else:
                        t_dev = tensor.to(device=mdev)
                    parent_mod.register_parameter(attr_name, nn.Parameter(t_dev, requires_grad=False))
                    loaded_param_keys.add(key)
                elif attr_name in parent_mod._buffers:
                    orig_b = parent_mod._buffers[attr_name]
                    if orig_b is not None and orig_b.is_floating_point() and tensor.is_floating_point():
                        t_dev = tensor.to(device=mdev, dtype=orig_b.dtype)
                    else:
                        t_dev = tensor.to(device=mdev)
                    parent_mod._buffers[attr_name] = t_dev
                else:
                    raise RuntimeError(f"unexpected parameter/buffer {key!r} on {type(parent_mod).__name__}")
                del tensor

    for mod_path, bucket in list(pending_linears.items()):
        parent_mod = mod_map[mod_path]
        mdev = _device_for_module_path(
            mod_path,
            default_dev,
            layer_devs,
            is_multi_gpu=is_multi_gpu,
            tie_word_embeddings=tie_we,
        )
        _materialize_buffered_linear(
            mod_path,
            parent_mod,
            bucket,
            mod_dev=mdev,
            target_dtype=target_dtype,
            meta_param_dtypes=meta_param_dtypes,
            keep_dense_quantized=keep_dense_quantized,
            loaded_param_keys=loaded_param_keys,
            replaced_param_keys=replaced_param_keys,
        )
        del pending_linears[mod_path]

    for exp_path, builder in moe_builders.items():
        if exp_path in direct_3d_moe_paths and builder.mode is None:
            exp_mod = builder.experts_mod
            if getattr(exp_mod, "group_size", None) is None and isinstance(
                getattr(exp_mod, "gate_up_proj_scale", None), torch.Tensor
            ):
                n_grp = int(exp_mod.gate_up_proj_scale.shape[-1])
                exp_mod.group_size = builder.h_dim // n_grp if n_grp > 0 else 16  # type: ignore[attr-defined]
            continue
        builder.finalize()
        replaced_param_keys.add(f"{exp_path}.gate_up_proj")
        replaced_param_keys.add(f"{exp_path}.down_proj")

    if tie_we:
        head_mod = getattr(model, "lm_head", None)
        emb_mod = getattr(tm, "embed_tokens", None)
        if (
            head_mod is not None
            and emb_mod is not None
            and isinstance(getattr(emb_mod, "weight", None), nn.Parameter)
            and not any(k == "lm_head.weight" or k.endswith(".lm_head.weight") for k in loaded_param_keys)
        ):
            head_dev = _device_for_module_path(
                "lm_head",
                default_dev,
                layer_devs,
                is_multi_gpu=is_multi_gpu,
                tie_word_embeddings=tie_we,
            )
            if emb_mod.weight.device == head_dev:
                head_mod.weight = emb_mod.weight
            else:
                head_mod.register_parameter(
                    "weight", nn.Parameter(emb_mod.weight.detach().to(device=head_dev), requires_grad=False)
                )
            for ek in expected_param_keys:
                if ek == "lm_head.weight" or ek.endswith(".lm_head.weight"):
                    loaded_param_keys.add(ek)

    # Materialize any remaining non-persistent meta buffers (e.g. causal mask / token_type_ids)
    for mpath, sub in model.named_modules():
        mdev = _device_for_module_path(
            mpath,
            default_dev,
            layer_devs,
            is_multi_gpu=is_multi_gpu,
            tie_word_embeddings=tie_we,
        )
        for bname, buf in list(sub._buffers.items()):
            if isinstance(buf, torch.Tensor) and buf.is_meta:
                sub._buffers[bname] = torch.zeros(buf.shape, dtype=buf.dtype, device=mdev)

    missing_params = sorted(expected_param_keys - loaded_param_keys - replaced_param_keys)
    if effective_text_only:
        missing_params = [k for k in missing_params if not _is_visual_or_mtp_key(k)]
    elif not has_mtp_mod:
        missing_params = [k for k in missing_params if not _is_mtp_key(k)]
    require(
        not missing_params,
        f"direct safetensors loader left {len(missing_params)} parameter(s) uninitialized: {missing_params[:10]}",
    )

    meta_params_left = [
        f"{mpath}.{pname}" if mpath else pname
        for mpath, sub in model.named_modules()
        for pname, p in sub._parameters.items()
        if isinstance(p, torch.Tensor) and p.is_meta
    ]
    meta_bufs_left = [
        f"{mpath}.{bname}" if mpath else bname
        for mpath, sub in model.named_modules()
        for bname, b in sub._buffers.items()
        if isinstance(b, torch.Tensor) and b.is_meta
    ]
    require(not meta_params_left, f"meta parameters remained after direct load: {meta_params_left[:10]}")
    require(not meta_bufs_left, f"meta buffers remained after direct load: {meta_bufs_left[:10]}")

    if effective_text_only:
        _prune_visual_tower(model)

    for p in model.parameters():
        p.requires_grad_(False)
    if install_bridge:
        install_quantized_autograd_bridge(model)

    model._qloop_direct_safetensors_loaded = True  # type: ignore[attr-defined]
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return model


def _propagate_quantization_config_for_direct(config: Any, text_cfg: Any) -> Any:
    return _propagate_quantization_config_to_text_config(config, text_cfg)


def _should_attempt_direct_safetensors_load(
    mcfg: dict[str, Any],
    *,
    name: Any,
    config: Any,
    text_cfg: Any,
    transformers_mod: Any,
    causal_lm_cls: Any,
) -> tuple[bool, bool]:
    """Return `(attempt_direct, strict_direct)` for `_load_hf`."""
    direct_opt = mcfg.get("direct_safetensors_load")
    if direct_opt is False:
        return False, False
    if direct_opt is True:
        return True, True

    qm = _extract_quant_method(text_cfg) or _extract_quant_method(config)
    if qm not in ("compressed-tensors", "modelopt"):
        return False, False

    if not callable(getattr(causal_lm_cls, "from_config", None)):
        return False, False

    path_str = os.fspath(name) if isinstance(name, (str, os.PathLike)) else str(name)
    if os.path.isdir(path_str):
        return True, False

    # Respect unit tests that monkeypatch `AutoModelForCausalLM.from_pretrained` on fake repo names
    # without mocking `huggingface_hub` (prevents unintended Hub network calls in offline tests).
    hf_hub_loaded = sys.modules.get("huggingface_hub")
    hf_hub_is_mock = hf_hub_loaded is not None and getattr(hf_hub_loaded, "__spec__", None) is None
    if not hf_hub_is_mock:
        if getattr(transformers_mod, "__spec__", None) is None:
            return False, False
        fp_fn = getattr(causal_lm_cls, "from_pretrained", None)
        fp_mod = str(getattr(fp_fn, "__module__", "") or "")
        if fp_mod and not fp_mod.startswith("transformers"):
            return False, False
    return True, False


def _load_hf(mcfg: dict[str, Any]):
    try:
        import transformers
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    except Exception as e:
        raise RuntimeError(f"transformers is required for real-model loading: {e!r}. "
                           "Run scripts/colab_setup.sh or `pip install -e .[colab]`.") from e
    name = mcfg["name"]
    common = {"revision": mcfg.get("revision"), "trust_remote_code": bool(mcfg.get("trust_remote_code", False))}
    if mcfg.get("token") is not None:
        common["token"] = mcfg["token"]
    text_only = bool(mcfg.get("text_only", True))
    with stage("load config"):
        config = AutoConfig.from_pretrained(name, **common)
        mt = getattr(config, "model_type", "?")
        log.info("model_type=%s transformers=%s", mt, transformers.__version__)
        has_vision_cfg = getattr(config, "vision_config", None) is not None
        effective_text_only = text_only or not has_vision_cfg
        raw_text_cfg = getattr(config, "text_config", config)
        text_cfg = _propagate_quantization_config_to_text_config(config, raw_text_cfg)
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
    target_dtype = resolve_dtype(mcfg.get("dtype"))
    kwargs: dict[str, Any] = dict(
        common,
        device_map=mcfg.get("device_map", "cuda:0"),
        attn_implementation=mcfg.get("attn_implementation", "sdpa"),
        low_cpu_mem_usage=bool(mcfg.get("low_cpu_mem_usage", True)),
        **{dtype_key: target_dtype},
    )
    attempt_direct, strict_direct = _should_attempt_direct_safetensors_load(
        mcfg,
        name=name,
        config=config,
        text_cfg=text_cfg,
        transformers_mod=transformers,
        causal_lm_cls=AutoModelForCausalLM,
    )
    with stage("load weights"):
        t0 = time.time()
        model = None
        errors = []
        if attempt_direct:
            resolved_shards: list[str] | None = None
            try:
                resolved_shards = _resolve_direct_safetensors_shards(
                    name,
                    text_only=effective_text_only,
                    skip_mtp=True,
                    revision=mcfg.get("revision"),
                    token=mcfg.get("token"),
                )
            except _NoSafetensorsFoundError as exc:
                if strict_direct:
                    raise
                log.debug("direct safetensors shard resolution skipped (%r); falling back to from_pretrained", exc)
            if resolved_shards is not None:
                model = load_compressed_safetensors_direct(
                    name,
                    config=config,
                    text_cfg=text_cfg,
                    text_only=effective_text_only,
                    dtype=target_dtype,
                    device_map=mcfg.get("device_map", "cuda:0"),
                    attn_implementation=str(mcfg.get("attn_implementation", "sdpa") or "sdpa"),
                    keep_dense_quantized=bool(mcfg.get("keep_dense_quantized", False)),
                    install_bridge=False,
                    revision=mcfg.get("revision"),
                    trust_remote_code=bool(mcfg.get("trust_remote_code", False)),
                    token=mcfg.get("token"),
                    _resolved_shards=resolved_shards,
                )
        if model is None and text_only:
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
        _flush_deferred_quant_hooks(model)
        log.info("weights loaded in %.0fs %s", time.time() - t0, cuda_mem_brief("| "))
    tok = AutoTokenizer.from_pretrained(mcfg.get("tokenizer") or name, **common)
    return model, tok


def load_model(mcfg: dict[str, Any]):
    """Returns (model, tokenizer, info). Base parameters are frozen on return."""
    from ..utils.tf32 import configure_tf32, is_tf32_configured

    if mcfg.get("allow_tf32") is not None:
        tf32_info = configure_tf32(bool(mcfg["allow_tf32"]))
    elif not is_tf32_configured():
        tf32_info = configure_tf32(True)
    else:
        tf32_info = None
    name = mcfg.get("name", "toy")
    model, tok = _load_toy(mcfg) if name == "toy" else _load_hf(mcfg)
    repack_info: dict[str, Any] | None = None
    if name == "toy":
        model = model.to(resolve_device(mcfg.get("device_map", "cpu")))
    else:
        _flush_deferred_quant_hooks(model)
        if bool(mcfg.get("repack_nvfp4_moe", True)):
            repack_info = repack_decompressed_nvfp4_moe_experts(model)
    for p in model.parameters():
        p.requires_grad_(False)
    quant_info = install_quantized_autograd_bridge(model)
    if repack_info is not None and int(repack_info.get("repacked_moe_modules", 0)) > 0:
        quant_info["repacked_moe_modules"] = int(repack_info["repacked_moe_modules"])
    parts = inspect_model(model)
    info = param_report(model)
    info["quantization"] = quant_info
    if getattr(model, "_qloop_direct_safetensors_loaded", False):
        info["direct_safetensors_load"] = True
    if tf32_info is not None:
        info["tf32"] = tf32_info
    if repack_info is not None and int(repack_info.get("repacked_moe_modules", 0)) > 0:
        info["nvfp4_moe_repack"] = repack_info
    log.info("%s", summarize(parts))
    for k, v in info.items():
        log.info("  %s: %s", k, v)
    if quant_info.get("is_quantized"):
        log.info(
            "quantized base model ready: method=%s format=%s bridged=%d/%d modules "
            "(linear=%d, moe=%d, native_fwd_ok=%d, native_grad_ok=%d)",
            quant_info.get("quant_method"),
            quant_info.get("format"),
            quant_info.get("bridged_total", 0),
            quant_info.get("total_quantized_modules", 0),
            quant_info.get("bridged_linear_count", 0),
            quant_info.get("bridged_moe_count", 0),
            quant_info.get("native_fwd_ok_count", 0),
            quant_info.get("native_grad_ok_count", 0),
        )
        lin_nvfp4_cnt = int(quant_info.get("linear_nvfp4_count", 0))
        moe_nvfp4_cnt = int(quant_info.get("moe_nvfp4_count", 0))
        fp8_cnt = int(quant_info.get("linear_fp8_count", 0)) + int(quant_info.get("moe_fp8_count", 0))
        native_fwd_ok_cnt = int(quant_info.get("native_fwd_ok_count", 0))
        qmethod = str(quant_info.get("quant_method") or "").lower()
        qfmt = str(quant_info.get("format") or "").lower()
        has_nvfp4 = (lin_nvfp4_cnt + moe_nvfp4_cnt) > 0 or "nvfp4" in qfmt or qmethod == "compressed-tensors"
        ct_installed = _is_package_installed("compressed_tensors")
        kernels_installed = _is_package_installed("kernels")

        if moe_nvfp4_cnt > 0 or (has_nvfp4 and ct_installed and native_fwd_ok_cnt == 0):
            log.info(
                "NVFP4 / compressed-tensors modules (moe_nvfp4=%d, moe_fp8=%d, linear_nvfp4=%d) running via "
                "qloop's vectorized single-sync chunked-3D Tensor-Core engine (bf16 compute, 4-bit/8-bit HBM storage).",
                moe_nvfp4_cnt,
                int(quant_info.get("moe_fp8_count", 0)),
                lin_nvfp4_cnt,
            )

        if native_fwd_ok_cnt == 0:
            if has_nvfp4:
                if not ct_installed and (lin_nvfp4_cnt > 0 or (moe_nvfp4_cnt == 0 and fp8_cnt == 0)):
                    log.warning(
                        "native quantized inference kernels unavailable (native_fwd_ok=0/%d for method=%s); "
                        "continuing via transient bf16 fallback. For native Blackwell kernel acceleration, "
                        "run `pip install compressed-tensors`.",
                        quant_info.get("total_quantized_modules", 0),
                        quant_info.get("quant_method"),
                    )
                elif fp8_cnt > 0 and lin_nvfp4_cnt == 0 and moe_nvfp4_cnt == 0 and not kernels_installed:
                    log.warning(
                        "native FP8 inference kernels unavailable (native_fwd_ok=0/%d for method=%s); "
                        "continuing via transient bf16 fallback. For native Blackwell kernel acceleration, "
                        'run `pip install "kernels>=0.18.2,<0.19.0"`.',
                        quant_info.get("total_quantized_modules", 0),
                        quant_info.get("quant_method"),
                    )
            elif not kernels_installed:
                log.warning(
                    "native quantized inference kernels unavailable (native_fwd_ok=0/%d for method=%s); "
                    "continuing via transient bf16 fallback. For native Blackwell kernel acceleration, "
                    'run `pip install "kernels>=0.18.2,<0.19.0"`.',
                    quant_info.get("total_quantized_modules", 0),
                    quant_info.get("quant_method"),
                )
            else:
                log.info(
                    "FP8 modules (count=%d) running via qloop's transient Tensor-Core engine.",
                    quant_info.get("total_quantized_modules", 0),
                )
    elif info["quantized_module_types"]:
        log.warning("quantized modules present %s: run `qloop probe --backward` once to confirm gradients flow "
                    "through frozen quantized experts on this transformers/kernel version",
                    info["quantized_module_types"])
    return model, tok, info
