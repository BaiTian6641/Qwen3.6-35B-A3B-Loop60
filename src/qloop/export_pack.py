"""Whole-Model Hugging Face Pack Exporter, Low-Disk Hub Streamer & Pack Loader.

Implements end-to-end packaging of `Qwen3.6-35B-A6B-LoopHSSD` (and `ToyForCausalLM` test stacks)
into self-contained Hugging Face repositories / directories containing:
  * Merged base + Loop-HSSD v3 / GDN-2 / D-LSA / Sandwich-Loop sharded `.safetensors` weights
    (`model-00001-of-NNNNN.safetensors` + `model.safetensors.index.json`),
  * Self-contained `config.json` with `auto_map` + `qloop_v3_arch` surgical recipe,
  * Standalone `configuration_qwen36_loophssd.py` and `modeling_qwen36_loophssd.py`,
  * `generation_config.json`, tokenizer artifacts, `qloop_pack_manifest.json` (with per-shard
    SHA-256 digests and subsystem parameter ledger), and a comprehensive `README.md` Model Card.

Supports **Colab 230 GB SSD Low-Disk Streaming** (`stream_low_disk=True` / `stream_whole_model_pack`):
each `~4.5 GB` shard is serialized, hashed, uploaded to Hugging Face Hub, and immediately unlinked
from local storage before the next shard is materialized (`< 6 GB` peak temporary disk even for a
70 GB `35B` model).
"""
from __future__ import annotations

import copy
import datetime as _dt
import hashlib
import json
import logging
import os
import shutil
import tempfile
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import nn

from .data.tokenizer import ToyTokenizer, load_tokenizer
from .debug import require
from .hub import DEFAULT_BASE_MODEL, DEFAULT_HF_REPO, resolve_hf_token
from .infer.hf_custom_modeling import (
    AUTO_MAP,
    CONFIG_FILENAME,
    MODELING_FILENAME,
    Qwen36LoopHSSDConfig,
    Qwen36LoopHSSDForCausalLM,
    _DepthContext,
    resolve_pack_directory,
    save_safetensors_file,
    write_standalone_hf_modeling_files,
)
from .model.experts import (
    _EXPERT_DELTA_KEY_RE,
    HotExpertDeltas,
    _experts_are_plain_float,
    fold_expert_deltas_state_into_base_,
    physical_layers,
)
from .model.gdn2 import GDN2Adapter, GDN2Manager
from .model.introspect import (
    find_text_model,
    inspect_model,
)
from .model.loop import LoopController
from .model.lsa import LSAIndexer, LSAManager
from .model.quant_autograd import detect_model_quantization, is_quantized_module
from .train.stack import Stack

log = logging.getLogger("qloop")

PACK_MANIFEST_FILE = "qloop_pack_manifest.json"
SAFETENSORS_INDEX_FILE = "model.safetensors.index.json"
GENERATION_CONFIG_FILE = "generation_config.json"
TOY_TOKENIZER_MARKER = "qloop_toy_tokenizer"


def _utc_now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _sha256_file(path: str, chunk_bytes: int = 4 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            blk = f.read(chunk_bytes)
            if not blk:
                break
            h.update(blk)
    return h.hexdigest()


def _dtype_name(dt: torch.dtype) -> str:
    return str(dt).replace("torch.", "")


# ============================================================================ config & loaded pack

@dataclass
class WholeModelPackConfig:
    """Configuration for exporting or streaming a self-contained Whole Model Pack."""

    repo_id: str = DEFAULT_HF_REPO
    out_dir: str | None = None
    max_shard_size_bytes: int = 4_500_000_000
    merge_hot_experts: bool = True
    merge_in_place: bool = True
    stream_low_disk: bool = True
    private: bool = False
    base_model: str = DEFAULT_BASE_MODEL
    commit_message: str = "Upload Qwen3.6-35B-A6B-LoopHSSD Whole Model Pack"


class LoadedModelPack(Stack):
    """Hydrated Whole Model Pack combining `Stack` ergonomics with direct `nn.Module` / depth calls."""

    hssd: Any = None
    hf_model: Qwen36LoopHSSDForCausalLM | None = None
    pack_config: Qwen36LoopHSSDConfig | None = None
    manifest: dict[str, Any] | None = None

    def set_active_passes(self, r: int) -> None:
        if self.hf_model is not None:
            self.hf_model.set_active_passes(int(r))
        elif self.loop is not None:
            self.loop.set_active_passes(int(r))
        else:
            require(int(r) == 1, f"loaded pack has no LoopController; cannot set depth={r}")

    @property
    def active_passes(self) -> int:
        if self.hf_model is not None:
            return self.hf_model.active_passes
        return 1 if self.loop is None else int(self.loop.active_passes)

    def depth(self, r: int) -> _DepthContext:
        prev = self.active_passes
        self.set_active_passes(int(r))
        return _DepthContext(self, prev, int(r))

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        target = self.hf_model if self.hf_model is not None else self.model
        return target(*args, **kwargs)

    def __getitem__(self, key: str) -> Any:
        if key == "model":
            return self.hf_model if self.hf_model is not None else self.model
        if hasattr(self, key):
            return getattr(self, key)
        raise KeyError(key)


# ============================================================================ introspection & recipe extraction

@dataclass
class _UnwrappedSource:
    backbone: nn.Module
    tokenizer: Any
    loop: LoopController | None
    gdn2: GDN2Manager | None
    lsa: LSAManager | None
    deltas: HotExpertDeltas | None
    hssd_cfg: dict[str, Any] | None
    stack: Stack | None
    base_model_name: str


def _detect_loop_controller(backbone: nn.Module) -> LoopController | None:
    tm = find_text_model(backbone)
    layers = getattr(tm, "layers", None)
    if isinstance(layers, nn.ModuleList) and len(layers) > 0:
        ctl = getattr(layers[0], "_ctl", None)
        if isinstance(ctl, LoopController):
            return ctl
    return None


def _detect_hot_deltas(backbone: nn.Module) -> HotExpertDeltas | None:
    tm = find_text_model(backbone)
    ed = getattr(tm, "expert_deltas", None)
    return ed if isinstance(ed, HotExpertDeltas) else None


def _unwrap_source(stack_or_model: Any, tokenizer: Any = None) -> _UnwrappedSource:
    if isinstance(stack_or_model, Stack):
        st = stack_or_model
        raw_model = st.model
        hcfg_st: dict[str, Any] | None = None
        if isinstance(raw_model, Qwen36LoopHSSDForCausalLM):
            bb = raw_model.backbone
            arch_st = raw_model.config.qloop_v3_arch or {}
            if (arch_st.get("hssd") or {}).get("enabled"):
                hcfg_st = dict(arch_st["hssd"])
            loop = st.loop or raw_model.loop_controller or _detect_loop_controller(bb)
            gdn2 = st.gdn2 or raw_model.gdn2_manager
            lsa = st.lsa or raw_model.lsa_manager
            deltas = st.deltas or raw_model.hot_deltas or _detect_hot_deltas(bb)
        else:
            bb = raw_model
            loop = st.loop or _detect_loop_controller(bb)
            gdn2 = st.gdn2
            lsa = st.lsa
            deltas = st.deltas or _detect_hot_deltas(bb)
        tok = tokenizer if tokenizer is not None else st.tokenizer
        bname = str((st.load_info or {}).get("name") or DEFAULT_BASE_MODEL)
        return _UnwrappedSource(bb, tok, loop, gdn2, lsa, deltas, hcfg_st, st, bname)

    if isinstance(stack_or_model, Qwen36LoopHSSDForCausalLM):
        bb = stack_or_model.backbone
        arch = stack_or_model.config.qloop_v3_arch or {}
        hcfg = dict(arch.get("hssd") or {}) if (arch.get("hssd") or {}).get("enabled") else None
        bname = str(arch.get("base_model") or DEFAULT_BASE_MODEL)
        return _UnwrappedSource(
            backbone=bb,
            tokenizer=tokenizer,
            loop=stack_or_model.loop_controller or _detect_loop_controller(bb),
            gdn2=stack_or_model.gdn2_manager,
            lsa=stack_or_model.lsa_manager,
            deltas=stack_or_model.hot_deltas or _detect_hot_deltas(bb),
            hssd_cfg=hcfg,
            stack=None,
            base_model_name=bname,
        )

    require(isinstance(stack_or_model, nn.Module), "stack_or_model must be a Stack or nn.Module")
    bb = stack_or_model
    return _UnwrappedSource(
        backbone=bb,
        tokenizer=tokenizer,
        loop=_detect_loop_controller(bb),
        gdn2=None,
        lsa=None,
        deltas=_detect_hot_deltas(bb),
        hssd_cfg=None,
        stack=None,
        base_model_name=DEFAULT_BASE_MODEL,
    )


def _serialize_quantization_config(qcfg: Any) -> dict[str, Any] | None:
    if qcfg is None:
        return None
    if isinstance(qcfg, dict):
        raw_d = copy.deepcopy(qcfg)
    elif hasattr(qcfg, "to_dict"):
        raw_d = copy.deepcopy(dict(qcfg.to_dict()))
    elif hasattr(qcfg, "__dict__"):
        raw_d = {
            k: copy.deepcopy(v)
            for k, v in vars(qcfg).items()
            if not k.startswith("__") and not callable(v)
        }
    else:
        return None
    cleaned = _json_clean(raw_d)
    return cleaned if isinstance(cleaned, dict) and bool(cleaned) else None


def _extract_physical_base_config(
    backbone: nn.Module,
    loop: LoopController | None,
    lsa: LSAManager | None,
) -> tuple[dict[str, Any], str]:
    tc = find_text_model(backbone).config
    if hasattr(tc, "to_dict"):
        bcfg = copy.deepcopy(dict(tc.to_dict()))
    else:
        bcfg = {
            k: copy.deepcopy(v)
            for k, v in vars(tc).items()
            if not k.startswith("__") and not callable(v)
        }

    raw_qcfg = bcfg.get("quantization_config") or getattr(tc, "quantization_config", None)
    if not raw_qcfg:
        bb_cfg = getattr(backbone, "config", None)
        if bb_cfg is not None:
            raw_qcfg = getattr(bb_cfg, "quantization_config", None)
    ser_qcfg = _serialize_quantization_config(raw_qcfg)
    if ser_qcfg is not None:
        bcfg["quantization_config"] = ser_qcfg

    phys_pairs, _, _ = physical_layers(backbone)
    n_phys = len(phys_pairs)
    if loop is not None:
        orig_n = loop._orig.get("num_hidden_layers", n_phys)
        orig_lt = loop._orig.get("layer_types")
        bcfg["num_hidden_layers"] = int(orig_n)
        if orig_lt is not None:
            bcfg["layer_types"] = list(orig_lt)
    else:
        bcfg["num_hidden_layers"] = int(bcfg.get("num_hidden_layers", n_phys))

    mtype = str(bcfg.get("model_type") or "")
    kind = "toy" if (mtype.startswith("toy") or type(backbone).__name__.startswith("Toy")) else "hf"

    attn_impl = bcfg.get("_attn_implementation")
    if attn_impl == "qloop_lsa":
        if lsa is not None and getattr(lsa, "_prev_impl", None):
            bcfg["_attn_implementation"] = str(lsa._prev_impl)
        else:
            bcfg["_attn_implementation"] = "eager" if kind == "toy" else "sdpa"

    json_safe: dict[str, Any] = {}
    for k, v in bcfg.items():
        json_safe[k] = _json_clean(v)
    return json_safe, kind


def _infer_gdn2_arch(src: _UnwrappedSource) -> dict[str, Any]:
    if src.gdn2 is not None:
        cfg_d = asdict(src.gdn2.cfg)
        cfg_d["layers"] = sorted(int(i) for i in src.gdn2.adapters)
        return {"enabled": True, **cfg_d}
    if src.stack is not None and isinstance(src.stack.carried_gdn2_cfg, dict):
        return {"enabled": True, **copy.deepcopy(src.stack.carried_gdn2_cfg)}

    phys_pairs, _, _ = physical_layers(src.backbone)
    found_layers: list[int] = []
    first_ad: GDN2Adapter | None = None
    for idx, layer in phys_pairs:
        lin = getattr(layer, "linear_attn", None)
        ad = getattr(lin, "gdn2_adapter", None) if lin is not None else None
        if isinstance(ad, GDN2Adapter):
            found_layers.append(int(idx))
            if first_ad is None:
                first_ad = ad
    if first_ad is not None:
        return {
            "enabled": True,
            "rank": int(first_ad.rank),
            "layers": found_layers,
            "chunk_size": 64,
            "gate_logit_cap": float(first_ad.gate_logit_cap),
            "decay_log_cap": float(first_ad.decay_log_cap),
            "force_recurrent": False,
        }
    return {"enabled": False}


def _infer_lsa_arch(src: _UnwrappedSource) -> dict[str, Any]:
    if src.lsa is not None:
        cfg_d = asdict(src.lsa.cfg)
        cfg_d["layers"] = sorted(int(i) for i in src.lsa.indexers)
        return {"enabled": True, **cfg_d}
    if src.stack is not None and isinstance(src.stack.carried_lsa_cfg, dict):
        return {"enabled": True, **copy.deepcopy(src.stack.carried_lsa_cfg)}

    phys_pairs, _, _ = physical_layers(src.backbone)
    found_layers: list[int] = []
    first_ix: LSAIndexer | None = None
    for idx, layer in phys_pairs:
        s_attn = getattr(layer, "self_attn", None)
        ix = getattr(s_attn, "lsa_indexer", None) if s_attn is not None else None
        if isinstance(ix, LSAIndexer):
            found_layers.append(int(idx))
            if first_ix is None:
                first_ix = ix
    if first_ix is not None:
        cfg_d = asdict(first_ix.cfg)
        cfg_d["layers"] = found_layers
        return {"enabled": True, **cfg_d}
    return {"enabled": False}


def _json_clean(obj: Any) -> Any:
    if isinstance(obj, torch.dtype):
        return _dtype_name(obj)
    if isinstance(obj, (list, tuple)):
        return [_json_clean(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): _json_clean(v) for k, v in obj.items()}
    return obj


def _extract_hssd_mod_meta(mod: nn.Module) -> dict[str, Any]:
    meta: dict[str, Any] = {}
    cfg_obj = getattr(mod, "cfg", None)
    if cfg_obj is not None:
        if hasattr(cfg_obj, "__dataclass_fields__"):
            meta.update(asdict(cfg_obj))
        elif hasattr(cfg_obj, "__dict__"):
            meta.update({k: v for k, v in vars(cfg_obj).items() if not k.startswith("_")})
    else:
        attr_map = (
            ("num_pages", "num_pages"),
            ("k_pages", "k_pages"),
            ("chunk_size", "chunk_size"),
            ("window_chunks", "window_chunks"),
            ("rank", "rank"),
            ("gate_logit_cap", "cap_g"),
            ("decay_log_cap", "cap_d"),
            ("theta_low_rank", "theta_low_rank"),
            ("state_dtype", "state_dtype"),
            ("paired_b_gate", "paired_b_gate"),
            ("use_swa", "use_swa"),
            ("use_phase", "use_phase"),
            ("use_surprise", "use_surprise"),
            ("surprise_mode", "surprise_mode"),
        )
        for out_k, attr_k in attr_map:
            if hasattr(mod, attr_k):
                meta[out_k] = getattr(mod, attr_k)
    if "theta_low_rank" not in meta:
        meta["theta_low_rank"] = bool(hasattr(mod, "theta_down"))
    return meta


def _with_hssd_v31_defaults(cfg_d: dict[str, Any], first_mod: nn.Module | None) -> dict[str, Any]:
    defaults: dict[str, Any] = {
        "theta_low_rank": bool(hasattr(first_mod, "theta_down")) if first_mod is not None else True,
        "state_dtype": "auto",
        "paired_b_gate": False,
        "use_swa": True,
        "use_phase": True,
        "use_surprise": True,
        "surprise_mode": "routed",
    }
    merged = {**defaults, **cfg_d}
    merged["enabled"] = True
    return _json_clean(merged)


def _infer_hssd_arch(src: _UnwrappedSource) -> dict[str, Any]:
    phys_pairs, _, _ = physical_layers(src.backbone)
    hssd_layers: list[int] = []
    first_mod: nn.Module | None = None
    for idx, layer in phys_pairs:
        lin = getattr(layer, "linear_attn", None)
        if lin is None:
            continue
        for attr in ("hssd", "hssd_adapter", "hssd_module"):
            mod = getattr(lin, attr, None)
            if isinstance(mod, nn.Module):
                hssd_layers.append(int(idx))
                if first_mod is None:
                    first_mod = mod
                break

    mod_meta = _extract_hssd_mod_meta(first_mod) if first_mod is not None else {}

    h_handle = getattr(src.stack, "hssd", None) if src.stack is not None else None
    if h_handle is None:
        h_handle = getattr(src.backbone, "_hssd_handle", None)
    handle_meta: dict[str, Any] = {}
    if h_handle is not None and hasattr(h_handle, "cfg"):
        handle_meta = (
            asdict(h_handle.cfg) if hasattr(h_handle.cfg, "__dataclass_fields__") else dict(vars(h_handle.cfg))
        )
        if hasattr(h_handle, "adapters") and isinstance(h_handle.adapters, dict):
            handle_meta["layers"] = sorted(int(i) for i in h_handle.adapters)

    if src.hssd_cfg is not None:
        combined = {**mod_meta, **handle_meta, **copy.deepcopy(src.hssd_cfg)}
        if "layers" not in combined and hssd_layers:
            combined["layers"] = hssd_layers
        return _with_hssd_v31_defaults(combined, first_mod)

    if handle_meta:
        return _with_hssd_v31_defaults({**mod_meta, **handle_meta}, first_mod)

    carried_hssd = getattr(src.stack, "carried_hssd_cfg", None) if src.stack is not None else None
    if isinstance(carried_hssd, dict):
        combined = {**mod_meta, **copy.deepcopy(carried_hssd)}
        if "layers" not in combined and hssd_layers:
            combined["layers"] = hssd_layers
        return _with_hssd_v31_defaults(combined, first_mod)

    if hssd_layers:
        return _with_hssd_v31_defaults({"layers": hssd_layers, **mod_meta}, first_mod)
    return {"enabled": False}


_QUANT_BUFFER_EXACT_NAMES = frozenset(
    {
        "weight_scale_inv",
        "scale_inv",
        "weight_scale",
        "weight_global_scale",
        "weight_scale_2",
        "weight_packed",
        "weight_zero_point",
        "weight_shape",
        "input_scale",
        "input_global_scale",
        "gate_up_proj_scale_inv",
        "down_proj_scale_inv",
        "gate_up_proj_weight_scale_inv",
        "down_proj_weight_scale_inv",
        "gate_up_proj_packed",
        "down_proj_packed",
        "gate_up_proj_scale",
        "down_proj_scale",
        "gate_up_proj_global_scale",
        "down_proj_global_scale",
        "gate_up_proj_scale_2",
        "down_proj_scale_2",
    }
)
_QUANT_BUFFER_SUFFIXES = (
    "_scale",
    "_scale_inv",
    "_global_scale",
    "_scale_2",
    "_packed",
    "_zero_point",
    "_qweight",
    "_scales",
    "_qzeros",
)
_QUANT_WEIGHT_ATTR_NAMES = frozenset(
    {"weight", "bias", "gate_up_proj", "down_proj", "gate_proj", "up_proj"}
)


def _is_quant_buffer_name(name: str) -> bool:
    return name in _QUANT_BUFFER_EXACT_NAMES or name.endswith(_QUANT_BUFFER_SUFFIXES)


def _experts_disallow_inplace_merge(exp: nn.Module) -> bool:
    if not _experts_are_plain_float(exp):
        return True
    return any(is_quantized_module(sub) for _n, sub in exp.named_modules())


def _backbone_has_unmergeable_hot_experts(src: _UnwrappedSource) -> bool:
    phys_map = dict(physical_layers(src.backbone)[0])
    target_layers: set[int] = set()
    if src.deltas is not None:
        target_layers.update(int(lid) for lid in src.deltas.hot)
    if src.stack is not None and src.stack.passthrough_tensors:
        for k in src.stack.passthrough_tensors:
            m = _EXPERT_DELTA_KEY_RE.search(k)
            if m is not None and (int(m.group(1)), int(m.group(3))) not in src.stack._folded_experts:
                target_layers.add(int(m.group(1)))
    check_layers = sorted(target_layers) if target_layers else sorted(phys_map.keys())
    for lid in check_layers:
        layer = phys_map.get(lid)
        mlp = getattr(layer, "mlp", None) if layer is not None else None
        exp = getattr(mlp, "experts", None) if mlp is not None else None
        if isinstance(exp, nn.Module) and _experts_disallow_inplace_merge(exp):
            return True
    return False


def _resolve_effective_merge_hot_experts(
    src: _UnwrappedSource,
    merge_hot_experts: bool,
    *,
    log_downgrade: bool = True,
) -> bool:
    if not merge_hot_experts:
        return False
    has_deltas = src.deltas is not None or bool(_collect_unfolded_passthrough_deltas(src.stack))
    if has_deltas and _backbone_has_unmergeable_hot_experts(src):
        if log_downgrade:
            log.info(
                "Keeping hot_experts unmerged in Whole Model Pack sidecar/state_dict: "
                "base MoE experts are quantized (FP8/NVFP4) and cannot absorb BF16 deltas in-place."
            )
        return False
    return True


def extract_arch_spec(
    stack_or_model: Any,
    arch_spec: dict[str, Any] | None = None,
    *,
    merged_hot_experts: bool = True,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Extract `(physical_base_config, qloop_v3_arch)` from a `Stack` or wrapped CausalLM."""
    src = _unwrap_source(stack_or_model)
    eff_merged_he = _resolve_effective_merge_hot_experts(src, merged_hot_experts, log_downgrade=False)
    bcfg, kind = _extract_physical_base_config(src.backbone, src.loop, src.lsa)
    tm = find_text_model(src.backbone)
    emb_dtype = _dtype_name(tm.embed_tokens.weight.dtype)
    phys_pairs, _, _ = physical_layers(src.backbone)
    n_phys = len(phys_pairs)

    if src.loop is not None:
        spec_d = asdict(src.loop.spec)
        spec_d["bridge_init"] = list(spec_d["bridge_init"])
        if spec_d.get("train_pass_probs") is not None:
            spec_d["train_pass_probs"] = list(spec_d["train_pass_probs"])
        core_len = int(src.loop.spec.core_len)
        max_r = int(src.loop.spec.num_passes)
        virt_layers = n_phys + (max_r - 1) * core_len
        depths = {str(r): n_phys + (r - 1) * core_len for r in range(1, max_r + 1)}
        loop_arch: dict[str, Any] = _json_clean({
            "enabled": True,
            **spec_d,
            "active_passes": int(src.loop.active_passes),
        })
        kv_share = bool(src.loop.spec.kv_share)
    else:
        virt_layers = n_phys
        depths = {"1": n_phys}
        loop_arch = {"enabled": False}
        kv_share = True

    unfolded_pt = _collect_unfolded_passthrough_deltas(src.stack)
    if src.deltas is not None:
        he_arch: dict[str, Any] = {
            "enabled": True,
            "merged_into_base": bool(eff_merged_he),
            "rank": int(src.deltas.rank),
            "alpha": float(src.deltas.scale * src.deltas.rank) if src.deltas.rank > 0 else None,
            "scale": float(src.deltas.scale),
            "hot_map": {str(k): [int(e) for e in v] for k, v in sorted(src.deltas.hot.items())},
        }
    elif unfolded_pt and not eff_merged_he:
        hm_pt: dict[int, list[int]] = {}
        inferred_rank = 0
        for (lid, eid), parts in sorted(unfolded_pt.items()):
            hm_pt.setdefault(int(lid), []).append(int(eid))
            if "gate_up_A" in parts:
                inferred_rank = int(parts["gate_up_A"].shape[0])
        he_arch = {
            "enabled": True,
            "merged_into_base": False,
            "rank": inferred_rank,
            "alpha": float(inferred_rank) if inferred_rank > 0 else None,
            "scale": 1.0,
            "hot_map": {str(k): sorted(set(v)) for k, v in sorted(hm_pt.items())},
        }
    else:
        he_arch = {"enabled": False, "merged_into_base": True}

    quant_info = detect_model_quantization(src.backbone)
    arch: dict[str, Any] = {
        "schema_version": 3,
        "backbone_kind": kind,
        "base_model": "toy" if kind == "toy" else src.base_model_name,
        "base_model_type": str(bcfg.get("model_type") or ("toy_qwen3_5_moe" if kind == "toy" else "qwen3_5_moe")),
        "base_dtype": emb_dtype,
        "physical_layers": n_phys,
        "virtual_layers": virt_layers,
        "active_depths": depths,
        "loop": loop_arch,
        "gdn2": _infer_gdn2_arch(src),
        "hssd": _infer_hssd_arch(src),
        "lsa": _infer_lsa_arch(src),
        "hot_experts": he_arch,
        "kv_cache": {
            "shared_kv_across_passes": kv_share,
            "kv_memory_multiplier": "1.0x" if kv_share else f"{loop_arch.get('num_passes', 1)}.0x",
            "supported_quant_modes": ["native", "fp8", "hadamard_int4"],
        },
    }
    if quant_info.get("is_quantized") or bcfg.get("quantization_config"):
        arch["quantization"] = _json_clean(quant_info)
    if arch_spec:
        for k, v in arch_spec.items():
            if isinstance(v, dict) and isinstance(arch.get(k), dict):
                arch[k].update(copy.deepcopy(v))
            else:
                arch[k] = copy.deepcopy(v)
    return bcfg, arch


# ============================================================================ canonical tensor planner

@dataclass
class _TensorEntry:
    key: str
    shape: tuple[int, ...]
    dtype: torch.dtype
    nbytes: int
    subsystem: str
    materialize: Callable[[], torch.Tensor]


def _classify_subsystem(key: str) -> str:
    if ".loop_adapters." in key or key.startswith("loop_adapters."):
        return "loop_adapters"
    if ".gdn2_adapter." in key:
        return "gdn2"
    if ".hssd" in key:
        return "hssd"
    if ".lsa_indexer." in key:
        return "lsa"
    if ".expert_deltas." in key or key.startswith("expert_deltas."):
        return "hot_expert_deltas"
    if ".mlp.experts." in key or ".mlp.shared_expert" in key or ".mlp.gate." in key:
        return "moe_experts_and_routers"
    if "embed_tokens" in key or "lm_head" in key:
        return "embeddings_and_head"
    return "backbone_attention_and_norms"


def _module_own_state_tensors(mod: nn.Module) -> list[tuple[str, torch.Tensor]]:
    """Yield `(subkey, tensor)` for parameters, persistent buffers, and quantization buffers/tensors."""
    out: list[tuple[str, torch.Tensor]] = []
    seen_keys: set[str] = set()
    for pname, p in mod.named_parameters():
        seen_keys.add(pname)
        out.append((pname, p))

    non_persist_by_mod: dict[int, set[str]] = {
        id(sub): set(getattr(sub, "_non_persistent_buffers_set", set()))
        for _n, sub in mod.named_modules()
    }
    for mname, sub in mod.named_modules():
        np_set = non_persist_by_mod.get(id(sub), set())
        sub_is_quant = is_quantized_module(sub)
        for bname, buf in sub._buffers.items():
            if buf is None:
                continue
            if bname in np_set and not _is_quant_buffer_name(bname):
                continue
            full = f"{mname}.{bname}" if mname else bname
            if full not in seen_keys:
                seen_keys.add(full)
                out.append((full, buf))
        for aname, val in sub.__dict__.items():
            if aname.startswith("_") or aname in sub._parameters or aname in sub._buffers:
                continue
            if not isinstance(val, torch.Tensor):
                continue
            if _is_quant_buffer_name(aname) or (sub_is_quant and aname in _QUANT_WEIGHT_ATTR_NAMES):
                full = f"{mname}.{aname}" if mname else aname
                if full not in seen_keys:
                    seen_keys.add(full)
                    out.append((full, val))
    return out


def _collect_unfolded_passthrough_deltas(
    stack: Stack | None,
) -> dict[tuple[int, int], dict[str, torch.Tensor]]:
    if stack is None or not stack.passthrough_tensors:
        return {}
    grouped: dict[tuple[int, int], dict[str, torch.Tensor]] = {}
    for k, t in stack.passthrough_tensors.items():
        m = _EXPERT_DELTA_KEY_RE.search(k)
        if m is None:
            continue
        pair = (int(m.group(1)), int(m.group(3)))
        if pair in stack._folded_experts:
            continue
        grouped.setdefault(pair, {})[m.group(2)] = t
    return grouped


def _prepare_canonical_entries(
    src: _UnwrappedSource,
    *,
    merge_hot_experts: bool = True,
    merge_in_place: bool = True,
) -> tuple[list[_TensorEntry], int]:
    """Return `(canonical_entries, num_merged_experts)` with zero virtual-layer duplication."""
    merge_hot_experts = _resolve_effective_merge_hot_experts(
        src,
        merge_hot_experts,
        log_downgrade=False,
    )
    bb = src.backbone
    tm = find_text_model(bb)
    tm_prefix = ""
    for mod_name, mod in bb.named_modules():
        if mod is tm:
            tm_prefix = mod_name
            break

    def _tm_key(sub: str) -> str:
        return f"{tm_prefix}.{sub}" if tm_prefix else sub

    merged_count = 0
    lazy_expert_patches: dict[tuple[int, str], Callable[[torch.Tensor], torch.Tensor]] = {}

    if merge_hot_experts:
        if merge_in_place:
            if src.stack is not None and src.stack.passthrough_tensors:
                folded = fold_expert_deltas_state_into_base_(
                    bb,
                    src.stack.passthrough_tensors,
                    skip=src.stack._folded_experts,
                )
                src.stack._folded_experts |= folded
                merged_count += len(folded)
            if src.deltas is not None:
                merged_count += int(src.deltas.merge_into_base_())
                src.deltas.detach()
                if getattr(tm, "expert_deltas", None) is src.deltas:
                    delattr(tm, "expert_deltas")
                if src.stack is not None and src.stack.deltas is src.deltas:
                    src.stack.deltas = None
                src.deltas = None
        else:
            extra_pt = _collect_unfolded_passthrough_deltas(src.stack)
            phys_map = dict(physical_layers(bb)[0])
            tc = tm.config
            hidden_sz = getattr(tc, "hidden_size", None)

            layers_with_deltas: set[int] = set()
            if src.deltas is not None:
                layers_with_deltas.update(int(lid) for lid in src.deltas.hot)
            layers_with_deltas.update(int(lid) for (lid, _e) in extra_pt)

            for l_idx in sorted(layers_with_deltas):
                if l_idx not in phys_map:
                    continue
                deltas_ref = src.deltas
                pt_for_layer = {e: parts for (ll, e), parts in extra_pt.items() if ll == l_idx}
                merged_Here = set()
                if deltas_ref is not None and l_idx in deltas_ref.hot:
                    merged_Here.update(int(e) for e in deltas_ref.hot[l_idx])
                merged_Here.update(pt_for_layer.keys())
                merged_count += len(merged_Here)

                def _make_patcher(
                    layer_i: int,
                    proj_kind: str,
                    d_ref: HotExpertDeltas | None = deltas_ref,
                    pt_map: dict[int, dict[str, torch.Tensor]] = pt_for_layer,
                    h_dim: int | None = hidden_sz,
                ) -> Callable[[torch.Tensor], torch.Tensor]:
                    @torch.no_grad()
                    def _patch(base_t: torch.Tensor) -> torch.Tensor:
                        cloned = base_t.detach().to("cpu").contiguous().clone()
                        if proj_kind == "gate_up_proj":
                            transposed = bool(h_dim is not None and cloned.shape[-1] != h_dim)
                        else:
                            transposed = bool(h_dim is not None and cloned.shape[-2] != h_dim)

                        for exp_id, parts in sorted(pt_map.items()):
                            if "gate_up" in parts and "down" in parts:
                                dw = parts["gate_up" if proj_kind == "gate_up_proj" else "down"].float()
                            else:
                                a_k = "gate_up_A" if proj_kind == "gate_up_proj" else "down_A"
                                b_k = "gate_up_B" if proj_kind == "gate_up_proj" else "down_B"
                                dw = parts[b_k].float() @ parts[a_k].float()
                            dw_cpu = (dw.t() if transposed else dw).to(dtype=cloned.dtype, device="cpu")
                            cloned[exp_id] += dw_cpu

                        if d_ref is not None and layer_i in d_ref.hot:
                            info = d_ref._infos[layer_i]
                            for exp_id in d_ref.hot[layer_i]:
                                dgu, ddn = d_ref.delta_weights(layer_i, exp_id)
                                dw = dgu if proj_kind == "gate_up_proj" else ddn
                                dw_cpu = (dw.t() if info.transposed else dw).detach().to(
                                    device="cpu", dtype=cloned.dtype
                                )
                                cloned[exp_id] += dw_cpu
                        return cloned

                    return _patch

                lazy_expert_patches[(l_idx, "gate_up_proj")] = _make_patcher(l_idx, "gate_up_proj")
                lazy_expert_patches[(l_idx, "down_proj")] = _make_patcher(l_idx, "down_proj")

    entries: list[_TensorEntry] = []
    emitted_keys: set[str] = set()
    covered_param_ids: set[int] = set()
    tie_word_emb = bool(getattr(tm.config, "tie_word_embeddings", False))
    emb_param = getattr(getattr(tm, "embed_tokens", None), "weight", None)

    layers_prefix = _tm_key("layers.")
    loop_prefix = _tm_key("loop_adapters.")
    ed_prefix = _tm_key("expert_deltas.")

    # 1) Backbone non-layer, non-loop, non-expert-delta tensors (embeddings, final norm, lm_head)
    for k, t in _module_own_state_tensors(bb):
        if k.startswith(layers_prefix) or k.startswith(loop_prefix):
            continue
        if merge_hot_experts and (k.startswith(ed_prefix) or ".expert_deltas." in k):
            continue
        if tie_word_emb and k.endswith("lm_head.weight") and emb_param is not None and t is emb_param:
            covered_param_ids.add(id(t))
            continue
        emitted_keys.add(k)
        if isinstance(t, nn.Parameter):
            covered_param_ids.add(id(t))
        entries.append(
            _TensorEntry(
                key=k,
                shape=tuple(int(d) for d in t.shape),
                dtype=t.dtype,
                nbytes=int(t.numel() * t.element_size()),
                subsystem=_classify_subsystem(k),
                materialize=lambda _t=t: _t.detach().to("cpu").contiguous(),
            )
        )

    # 2) Physical layers 0 .. N_phys - 1 (deduplicates VirtualLayer wrappers automatically)
    phys_pairs, _, _ = physical_layers(bb)
    for phys_idx, phys_mod in phys_pairs:
        for subk, t in _module_own_state_tensors(phys_mod):
            canon_k = _tm_key(f"layers.{phys_idx}.{subk}")
            if canon_k in emitted_keys:
                continue
            emitted_keys.add(canon_k)
            if isinstance(t, nn.Parameter):
                covered_param_ids.add(id(t))
            patch_fn = None
            if subk == "mlp.experts.gate_up_proj":
                patch_fn = lazy_expert_patches.get((phys_idx, "gate_up_proj"))
            elif subk == "mlp.experts.down_proj":
                patch_fn = lazy_expert_patches.get((phys_idx, "down_proj"))
            if patch_fn is not None:
                def _mat_patched(
                    _t: torch.Tensor = t,
                    _pf: Callable[[torch.Tensor], torch.Tensor] = patch_fn,
                ) -> torch.Tensor:
                    return _pf(_t)

                mat_fn: Callable[[], torch.Tensor] = _mat_patched
            else:
                def _mat_plain(_t: torch.Tensor = t) -> torch.Tensor:
                    return _t.detach().to("cpu").contiguous()

                mat_fn = _mat_plain
            entries.append(
                _TensorEntry(
                    key=canon_k,
                    shape=tuple(int(d) for d in t.shape),
                    dtype=t.dtype,
                    nbytes=int(t.numel() * t.element_size()),
                    subsystem=_classify_subsystem(canon_k),
                    materialize=mat_fn,
                )
            )

    # 3) Loop adapters (`gates`, `bridge`, `pass_router`, `mor_router`)
    loop_mod = src.loop.adapters if src.loop is not None else getattr(tm, "loop_adapters", None)
    if isinstance(loop_mod, nn.Module):
        for subk, t in _module_own_state_tensors(loop_mod):
            canon_k = _tm_key(f"loop_adapters.{subk}")
            if canon_k in emitted_keys:
                continue
            emitted_keys.add(canon_k)
            if isinstance(t, nn.Parameter):
                covered_param_ids.add(id(t))
            entries.append(
                _TensorEntry(
                    key=canon_k,
                    shape=tuple(int(d) for d in t.shape),
                    dtype=t.dtype,
                    nbytes=int(t.numel() * t.element_size()),
                    subsystem="loop_adapters",
                    materialize=lambda _t=t: _t.detach().to("cpu").contiguous(),
                )
            )

    # 3b) Sidecar hot-expert deltas when `merge_hot_experts` is False (e.g. quantized FP8/NVFP4 base)
    if not merge_hot_experts and src.deltas is not None:
        for subk, t in _module_own_state_tensors(src.deltas):
            canon_k = _tm_key(f"expert_deltas.{subk}")
            if canon_k in emitted_keys:
                continue
            emitted_keys.add(canon_k)
            if isinstance(t, nn.Parameter):
                covered_param_ids.add(id(t))
            entries.append(
                _TensorEntry(
                    key=canon_k,
                    shape=tuple(int(d) for d in t.shape),
                    dtype=t.dtype,
                    nbytes=int(t.numel() * t.element_size()),
                    subsystem="hot_expert_deltas",
                    materialize=lambda _t=t: _t.detach().to("cpu").contiguous(),
                )
            )
    if not merge_hot_experts and src.stack is not None and src.stack.passthrough_tensors:
        for raw_k, t in sorted(src.stack.passthrough_tensors.items()):
            m = _EXPERT_DELTA_KEY_RE.search(raw_k)
            if m is None:
                continue
            pair = (int(m.group(1)), int(m.group(3)))
            if pair in src.stack._folded_experts:
                continue
            canon_k = _tm_key(f"expert_deltas.deltas.L{pair[0]}.{m.group(2)}.{pair[1]}")
            if canon_k in emitted_keys:
                continue
            emitted_keys.add(canon_k)
            entries.append(
                _TensorEntry(
                    key=canon_k,
                    shape=tuple(int(d) for d in t.shape),
                    dtype=t.dtype,
                    nbytes=int(t.numel() * t.element_size()),
                    subsystem="hot_expert_deltas",
                    materialize=lambda _t=t: _t.detach().to("cpu").contiguous(),
                )
            )

    # 4) Verify 100% parameter coverage on backbone (excluding merged expert_deltas)
    skipped_ed_ids: set[int] = set()
    if merge_hot_experts and src.deltas is not None:
        skipped_ed_ids = {id(p) for p in src.deltas.parameters()}
    missing_params = [
        pname
        for pname, p in bb.named_parameters()
        if id(p) not in covered_param_ids and id(p) not in skipped_ed_ids
    ]
    require(
        not missing_params,
        "canonical Whole Model Pack export missed parameters on backbone",
        missing=missing_params[:10],
    )
    return entries, merged_count


def _plan_shards(
    entries: list[_TensorEntry],
    max_shard_size_bytes: int,
) -> list[tuple[str, list[_TensorEntry]]]:
    limit = max(1, int(max_shard_size_bytes))
    buckets: list[list[_TensorEntry]] = []
    cur: list[_TensorEntry] = []
    cur_bytes = 0
    for ent in entries:
        if cur and (cur_bytes + ent.nbytes > limit):
            buckets.append(cur)
            cur = []
            cur_bytes = 0
        cur.append(ent)
        cur_bytes += ent.nbytes
    if cur or not buckets:
        buckets.append(cur)
    total = len(buckets)
    return [(f"model-{idx:05d}-of-{total:05d}.safetensors", bucket) for idx, bucket in enumerate(buckets, start=1)]


# ============================================================================ metadata & model card builders

def _save_tokenizer_artifacts(tokenizer: Any, out_dir: str) -> list[str]:
    if tokenizer is None:
        return []
    written: list[str] = []
    if hasattr(tokenizer, "save_pretrained") and not isinstance(tokenizer, ToyTokenizer):
        before = set(os.listdir(out_dir)) if os.path.isdir(out_dir) else set()
        tokenizer.save_pretrained(out_dir)
        after = set(os.listdir(out_dir))
        written.extend(sorted(after - before))
        return written

    if isinstance(tokenizer, ToyTokenizer):
        tok_cfg = {
            "tokenizer_class": "ToyTokenizer",
            TOY_TOKENIZER_MARKER: True,
            "vocab_size": int(tokenizer.vocab_size),
            "bos_token": tokenizer.bos_token,
            "eos_token": tokenizer.eos_token,
            "pad_token": tokenizer.pad_token,
            "eos_token_id": int(tokenizer.eos_token_id),
            "pad_token_id": int(tokenizer.pad_token_id),
        }
        spec_map = {
            "bos_token": tokenizer.bos_token or "",
            "eos_token": tokenizer.eos_token,
            "pad_token": tokenizer.pad_token,
            "additional_special_tokens": list(getattr(tokenizer, "added", ())),
        }
        tc_path = os.path.join(out_dir, "tokenizer_config.json")
        sm_path = os.path.join(out_dir, "special_tokens_map.json")
        with open(tc_path, "w", encoding="utf-8") as f:
            json.dump(tok_cfg, f, indent=2, sort_keys=True)
            f.write("\n")
        with open(sm_path, "w", encoding="utf-8") as f:
            json.dump(spec_map, f, indent=2, sort_keys=True)
            f.write("\n")
        written.extend(["tokenizer_config.json", "special_tokens_map.json"])
    return written


def _load_pack_tokenizer(local_dir: str, token: str | None = None) -> Any:
    tc_path = os.path.join(local_dir, "tokenizer_config.json")
    if os.path.isfile(tc_path):
        try:
            with open(tc_path, encoding="utf-8") as f:
                tcfg = json.load(f)
            if tcfg.get(TOY_TOKENIZER_MARKER) or tcfg.get("tokenizer_class") == "ToyTokenizer":
                return ToyTokenizer()
        except Exception:
            pass
    for tok_file in ("tokenizer.json", "tokenizer.model", "vocab.json", "tokenizer_config.json"):
        if os.path.isfile(os.path.join(local_dir, tok_file)):
            try:
                return load_tokenizer(local_dir, token=token)
            except Exception as e:
                log.warning("failed to load HF tokenizer from %s (%r); falling back to ToyTokenizer", local_dir, e)
                return ToyTokenizer()
    return ToyTokenizer()


def _build_generation_config(base_cfg: dict[str, Any], tokenizer: Any) -> dict[str, Any]:
    eos_id = getattr(tokenizer, "eos_token_id", None)
    if eos_id is None:
        eos_id = base_cfg.get("eos_token_id", 258)
    pad_id = getattr(tokenizer, "pad_token_id", None)
    if pad_id is None:
        pad_id = base_cfg.get("pad_token_id", eos_id)
    bos_id = getattr(tokenizer, "bos_token_id", None)
    if bos_id is None:
        bos_id = base_cfg.get("bos_token_id")
    gen_cfg: dict[str, Any] = {
        "_from_model_config": False,
        "do_sample": True,
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 20,
        "eos_token_id": eos_id,
        "pad_token_id": pad_id,
    }
    if bos_id is not None:
        gen_cfg["bos_token_id"] = bos_id
    return gen_cfg


def _build_parameter_ledger(entries: list[_TensorEntry]) -> dict[str, dict[str, int]]:
    ledger: dict[str, dict[str, int]] = {}
    for ent in entries:
        n_el = 1
        for d in ent.shape:
            n_el *= int(d)
        slot = ledger.setdefault(ent.subsystem, {"tensors": 0, "parameters": 0, "bytes": 0})
        slot["tensors"] += 1
        slot["parameters"] += int(n_el)
        slot["bytes"] += int(ent.nbytes)
    return ledger


def _render_pack_model_card(
    *,
    repo_id: str,
    arch: dict[str, Any],
    manifest: dict[str, Any],
    eval_summary: dict[str, Any] | None = None,
) -> str:
    base_model = str(arch.get("base_model") or DEFAULT_BASE_MODEL)
    phys_l = int(arch.get("physical_layers", 40))
    virt_l = int(arch.get("virtual_layers", phys_l))
    depths = arch.get("active_depths") or {"1": phys_l}
    loop_info = arch.get("loop") or {}
    p_end = loop_info.get("prelude_end", 0)
    c_end = loop_info.get("core_end", phys_l)
    n_shards = int(manifest.get("num_shards", 1))
    tot_params = int(manifest.get("total_parameters", 0))
    tot_gib = float(manifest.get("total_bytes", 0)) / float(2**30)
    lin_kind = (
        "HSSD v3"
        if (arch.get("hssd") or {}).get("enabled")
        else ("GDN-2" if (arch.get("gdn2") or {}).get("enabled") else "Gated DeltaNet")
    )
    lsa_on = bool((arch.get("lsa") or {}).get("enabled"))
    yaml_base_line = f"base_model: {base_model}\n" if base_model != "toy" else ""

    depth_rows = []
    shared_kv = bool((arch.get("kv_cache") or {}).get("shared_kv_across_passes", True))
    for r_str, v_cnt in sorted(depths.items(), key=lambda kv: int(kv[0])):
        r_i = int(r_str)
        kv_mult = "1.0x (SharedKV)" if shared_kv else f"{r_i}.0x"
        depth_rows.append(f"| `R = {r_i}` | `{v_cnt}L` | `{r_i}` core pass(es) | `{kv_mult}` |")
    depth_table = "\n".join(depth_rows)

    ledger_rows = []
    for sub_name, stats in sorted((manifest.get("parameter_ledger") or {}).items()):
        mb = stats["bytes"] / (1024.0 * 1024.0)
        ledger_rows.append(
            f"| `{sub_name}` | `{stats['tensors']}` | `{stats['parameters']:,}` | `{mb:.2f} MiB` |"
        )
    ledger_table = "\n".join(ledger_rows)

    eval_section = ""
    if eval_summary:
        eval_section = (
            "\n## Evaluation Summary\n\n```json\n"
            + json.dumps(eval_summary, indent=2, sort_keys=True, default=str)
            + "\n```\n"
        )

    sandwich_sched = f"[0..{p_end}) -> R x [{p_end}..{c_end}) -> [{c_end}..{phys_l})"
    full_attn_desc = (
        "Dynamic Landmark Sparse Attention (D-LSA) + 1.0x SharedKV" if lsa_on else "1.0x SharedKV"
    )

    return f"""---
library_name: transformers
license: apache-2.0
{yaml_base_line}pipeline_tag: text-generation
tags:
  - qwen3.6
  - moe
  - looped-transformer
  - test-time-compute
  - gdn2
  - loophssd
  - dynamic-sparse-attention
  - whole-model-pack
---

# {repo_id} — Self-Contained Whole Model Pack

Complete, standalone Hugging Face **Whole Model Pack** for **Qwen3.6-LoopHSSD v3** (`{phys_l}L` physical ->
up to `{virt_l}L` virtual Sandwich-Looped Hybrid MoE with `{lin_kind}`, `1.0x SharedKV`, and
`{'D-LSA' if lsa_on else 'Full Attention'}`). All hot-expert LoRA deltas are folded directly into the
base MoE expert weights so inference runs with **zero LoRA hook overhead**.

## Pack Summary

- **Physical / Max Virtual Layers**: `{phys_l}L` (`{sandwich_sched}`) -> `{virt_l}L` virtual
- **Linear Attention Engine**: `{lin_kind}` (`O(1)` recurrent state per pass slot)
- **Full Attention Engine**: `{full_attn_desc}`
- **Total Parameters**: `{tot_params:,}` (`{tot_gib:.3f} GiB` across `{n_shards}` `.safetensors` shard(s))

### Dynamic Test-Time Depth (`R in {{1, 2, 3}}` Without Reloading Weights)

| Depth Mode | Effective Depth | Core Passes | Full-Attn KV Footprint |
| :--- | :---: | :---: | :---: |
{depth_table}

## Quickstart 1: Shipped `OptimizedLoopInferenceEngine` (Chunked Prefill + FP8 KV + Dynamic Depth)

```python
from qloop.infer import InferenceConfig, OptimizedLoopInferenceEngine

cfg = InferenceConfig(
    active_passes=3,
    prefill_chunk_size=4096,   # last-token-only logits -> 128K-1M context safe
    kv_cache_quant="fp8",      # 50% full-attn KV HBM savings with 1.0x SharedKV
    temperature=0.0,
)
engine = OptimizedLoopInferenceEngine.from_pretrained("{repo_id}", cfg=cfg)

# Switch depth dynamically on the exact same loaded weights (40L / 64L / 88L)
with engine.depth(2):
    res = engine.generate("Solve x^2 - 5x + 6 = 0 step by step.", max_new_tokens=128)
    print(res.text)
    print(res.metrics)
```

## Quickstart 2: Standard Hugging Face `AutoModelForCausalLM` (`trust_remote_code=True`)

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("{repo_id}", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    "{repo_id}",
    trust_remote_code=True,
    device_map="auto",
)
model.set_active_passes(2)  # 1, 2, or 3 passes dynamically
inputs = tokenizer("Hello from Loop-HSSD!", return_tensors="pt").to(model.backbone.device)
out = model(**inputs, logits_to_keep=1)
```

## Parameter Ledger

| Subsystem | Tensors | Parameters | Size |
| :--- | ---: | ---: | ---: |
{ledger_table}
{eval_section}"""


def _write_non_shard_pack_files(
    out_dir: str,
    *,
    repo_id: str,
    base_cfg: dict[str, Any],
    arch: dict[str, Any],
    tokenizer: Any,
    weight_map: dict[str, str],
    shard_digests: list[dict[str, Any]],
    entries: list[_TensorEntry],
    merged_experts: int,
    eval_summary: dict[str, Any] | None = None,
) -> tuple[list[str], dict[str, Any]]:
    os.makedirs(out_dir, exist_ok=True)
    written: list[str] = []

    total_bytes = int(sum(s["size_bytes"] for s in shard_digests))
    total_tensor_bytes = int(sum(e.nbytes for e in entries))
    ledger = _build_parameter_ledger(entries)
    total_params = int(sum(v["parameters"] for v in ledger.values()))

    # 1) model.safetensors.index.json
    index_payload = {
        "metadata": {
            "total_size": total_tensor_bytes,
            "format": "pt",
            "qloop_pack_version": "3.0",
        },
        "weight_map": weight_map,
    }
    index_path = os.path.join(out_dir, SAFETENSORS_INDEX_FILE)
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(index_payload, f, indent=2, sort_keys=True)
        f.write("\n")
    written.append(SAFETENSORS_INDEX_FILE)

    # 2) config.json (with auto_map + qloop_v3_arch + preserved quantization_config)
    extra_cfg_kw: dict[str, Any] = {}
    if base_cfg.get("quantization_config"):
        extra_cfg_kw["quantization_config"] = copy.deepcopy(base_cfg["quantization_config"])
    hf_cfg = Qwen36LoopHSSDConfig(
        base_config=base_cfg,
        qloop_v3_arch=arch,
        auto_map=dict(AUTO_MAP),
        architectures=["Qwen36LoopHSSDForCausalLM"],
        **extra_cfg_kw,
    )
    hf_cfg.save_pretrained(out_dir)
    written.append("config.json")

    # 3) generation_config.json
    gen_cfg = _build_generation_config(base_cfg, tokenizer)
    gen_path = os.path.join(out_dir, GENERATION_CONFIG_FILE)
    with open(gen_path, "w", encoding="utf-8") as f:
        json.dump(gen_cfg, f, indent=2, sort_keys=True)
        f.write("\n")
    written.append(GENERATION_CONFIG_FILE)

    # 4) Standalone HF custom modeling files
    write_standalone_hf_modeling_files(out_dir)
    written.extend([CONFIG_FILENAME, MODELING_FILENAME])

    # 5) Tokenizer files
    tok_files = _save_tokenizer_artifacts(tokenizer, out_dir)
    for fn in tok_files:
        if fn not in written:
            written.append(fn)

    # 6) qloop_pack_manifest.json
    manifest: dict[str, Any] = {
        "schema_version": 3,
        "pack_type": "qloop_whole_model_pack",
        "repo_id": repo_id,
        "created_at": _utc_now(),
        "num_shards": len(shard_digests),
        "total_bytes": total_bytes,
        "total_tensor_bytes": total_tensor_bytes,
        "total_parameters": total_params,
        "merged_hot_experts_count": int(merged_experts),
        "shards": shard_digests,
        "parameter_ledger": ledger,
        "qloop_v3_arch": arch,
    }
    if eval_summary:
        manifest["eval_summary"] = eval_summary
    manifest_path = os.path.join(out_dir, PACK_MANIFEST_FILE)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True, default=str)
        f.write("\n")
    written.append(PACK_MANIFEST_FILE)

    # 7) Comprehensive README.md Model Card
    readme_text = _render_pack_model_card(
        repo_id=repo_id,
        arch=arch,
        manifest=manifest,
        eval_summary=eval_summary,
    )
    readme_path = os.path.join(out_dir, "README.md")
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(readme_text)
    written.append("README.md")

    return written, manifest


# ============================================================================ local export

def export_whole_model_pack(
    stack_or_model: Any,
    out_dir: str | os.PathLike[str],
    tokenizer: Any = None,
    arch_spec: dict[str, Any] | None = None,
    max_shard_size_bytes: int = 4_500_000_000,
    merge_hot_experts: bool = True,
    *,
    merge_in_place: bool = True,
    repo_id: str | None = None,
    eval_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Export a self-contained Hugging Face Whole Model Pack to `out_dir`."""
    target_dir = os.path.abspath(os.fspath(out_dir))
    os.makedirs(target_dir, exist_ok=True)

    src = _unwrap_source(stack_or_model, tokenizer=tokenizer)
    eff_merge_he = _resolve_effective_merge_hot_experts(src, merge_hot_experts, log_downgrade=True)
    base_cfg, arch = extract_arch_spec(
        stack_or_model,
        arch_spec=arch_spec,
        merged_hot_experts=eff_merge_he,
    )
    entries, merged_cnt = _prepare_canonical_entries(
        src,
        merge_hot_experts=eff_merge_he,
        merge_in_place=merge_in_place,
    )
    planned = _plan_shards(entries, max_shard_size_bytes)

    weight_map: dict[str, str] = {}
    shard_digests: list[dict[str, Any]] = []
    shard_filenames: list[str] = []

    for shard_idx, (shard_name, shard_entries) in enumerate(planned, start=1):
        tensors: dict[str, torch.Tensor] = {}
        for ent in shard_entries:
            tensors[ent.key] = ent.materialize()
            weight_map[ent.key] = shard_name
        shard_path = os.path.join(target_dir, shard_name)
        size_bytes = save_safetensors_file(
            tensors,
            shard_path,
            metadata={"qloop_pack": "3.0", "shard": f"{shard_idx}/{len(planned)}"},
        )
        del tensors
        sha256_hex = _sha256_file(shard_path)
        shard_digests.append(
            {
                "filename": shard_name,
                "num_tensors": len(shard_entries),
                "size_bytes": int(size_bytes),
                "sha256": sha256_hex,
            }
        )
        shard_filenames.append(shard_name)

    meta_files, manifest = _write_non_shard_pack_files(
        target_dir,
        repo_id=repo_id or os.path.basename(target_dir) or DEFAULT_HF_REPO,
        base_cfg=base_cfg,
        arch=arch,
        tokenizer=src.tokenizer,
        weight_map=weight_map,
        shard_digests=shard_digests,
        entries=entries,
        merged_experts=merged_cnt,
        eval_summary=eval_summary,
    )

    log.info(
        "exported Whole Model Pack -> %s (%d shard(s), %.2f MiB, %d params, %d merged experts)",
        target_dir,
        len(shard_filenames),
        manifest["total_bytes"] / (1024.0 * 1024.0),
        manifest["total_parameters"],
        merged_cnt,
    )
    return {
        "out_dir": target_dir,
        "num_shards": len(shard_filenames),
        "shard_files": shard_filenames,
        "metadata_files": meta_files,
        "files": shard_filenames + meta_files,
        "total_bytes": manifest["total_bytes"],
        "total_parameters": manifest["total_parameters"],
        "merged_hot_experts": merged_cnt,
        "parameter_ledger": manifest["parameter_ledger"],
        "index_path": os.path.join(target_dir, SAFETENSORS_INDEX_FILE),
        "config_path": os.path.join(target_dir, "config.json"),
        "manifest_path": os.path.join(target_dir, PACK_MANIFEST_FILE),
        "manifest": manifest,
    }


# ============================================================================ low-disk HF Hub streaming

def _purge_hf_transient_upload_cache() -> None:
    """Purge transient HF xet/upload staging caches so multi-GB shard blobs never accumulate on Colab."""
    hf_home = os.environ.get("HF_HOME") or os.path.expanduser("~/.cache/huggingface")
    for sub in ("xet", "upload"):
        p = os.path.join(hf_home, sub)
        if os.path.isdir(p):
            shutil.rmtree(p, ignore_errors=True)


def _resolve_hub_api(api: Any, token: str | None) -> Any:
    if api is not None:
        return api
    from huggingface_hub import HfApi  # type: ignore[import-not-found]

    return HfApi(token=resolve_hf_token(token))


def _ensure_hub_repo(api: Any, repo_id: str, private: bool = False) -> None:
    if hasattr(api, "create_repo"):
        api.create_repo(repo_id=repo_id, repo_type="model", private=bool(private), exist_ok=True)


def _upload_one_file(
    api: Any,
    repo_id: str,
    local_path: str,
    path_in_repo: str,
    commit_message: str,
) -> str:
    if hasattr(api, "upload_file"):
        res = api.upload_file(
            path_or_fileobj=local_path,
            path_in_repo=path_in_repo,
            repo_id=repo_id,
            repo_type="model",
            commit_message=commit_message,
        )
        return str(getattr(res, "oid", None) or res or "uploaded")
    if hasattr(api, "commit_bundle"):
        return str(api.commit_bundle(repo_id, [(path_in_repo, local_path)], commit_message))
    if hasattr(api, "create_commit"):
        from huggingface_hub import CommitOperationAdd  # type: ignore[import-not-found]

        op = CommitOperationAdd(path_in_repo=path_in_repo, path_or_fileobj=local_path)
        info = api.create_commit(
            repo_id=repo_id,
            repo_type="model",
            operations=[op],
            commit_message=commit_message,
        )
        return str(getattr(info, "oid", None) or "committed")
    raise RuntimeError(f"Unsupported HfApi object {type(api).__name__}: lacks upload_file / create_commit")


def _upload_metadata_batch(
    api: Any,
    repo_id: str,
    base_dir: str,
    rel_files: list[str],
    commit_message: str,
) -> str:
    if hasattr(api, "commit_bundle"):
        pairs = [(fn, os.path.join(base_dir, fn)) for fn in rel_files]
        return str(api.commit_bundle(repo_id, pairs, commit_message))
    if hasattr(api, "create_commit"):
        try:
            from huggingface_hub import CommitOperationAdd  # type: ignore[import-not-found]

            ops = [
                CommitOperationAdd(path_in_repo=fn, path_or_fileobj=os.path.join(base_dir, fn))
                for fn in rel_files
            ]
            info = api.create_commit(
                repo_id=repo_id,
                repo_type="model",
                operations=ops,
                commit_message=commit_message,
            )
            return str(getattr(info, "oid", None) or "committed")
        except ImportError:
            pass
    last_ref = "uploaded"
    for fn in rel_files:
        last_ref = _upload_one_file(api, repo_id, os.path.join(base_dir, fn), fn, commit_message)
    return last_ref


def stream_whole_model_pack(
    stack_or_model: Any,
    repo_id: str,
    *,
    token: str | None = None,
    tokenizer: Any = None,
    arch_spec: dict[str, Any] | None = None,
    max_shard_size_bytes: int = 4_500_000_000,
    merge_hot_experts: bool = True,
    merge_in_place: bool = True,
    private: bool = False,
    commit_message: str = "Upload Qwen3.6-35B-A6B-LoopHSSD Whole Model Pack",
    staging_dir: str | os.PathLike[str] | None = None,
    eval_summary: dict[str, Any] | None = None,
    clean_hf_cache_each_shard: bool = True,
    api: Any = None,
) -> dict[str, Any]:
    """Stream a Whole Model Pack to HF Hub one shard at a time (`< 6 GB` peak temporary disk)."""
    require(bool(repo_id and str(repo_id).strip()), "repo_id must be a non-empty Hugging Face repo id")
    hub_api = _resolve_hub_api(api, token)
    _ensure_hub_repo(hub_api, repo_id, private=private)

    src = _unwrap_source(stack_or_model, tokenizer=tokenizer)
    eff_merge_he = _resolve_effective_merge_hot_experts(src, merge_hot_experts, log_downgrade=True)
    base_cfg, arch = extract_arch_spec(
        stack_or_model,
        arch_spec=arch_spec,
        merged_hot_experts=eff_merge_he,
    )
    entries, merged_cnt = _prepare_canonical_entries(
        src,
        merge_hot_experts=eff_merge_he,
        merge_in_place=merge_in_place,
    )
    planned = _plan_shards(entries, max_shard_size_bytes)

    own_tmp: tempfile.TemporaryDirectory[str] | None = None
    if staging_dir is None:
        own_tmp = tempfile.TemporaryDirectory(prefix="qloop_stream_pack_")
        work_dir = own_tmp.name
    else:
        work_dir = os.path.abspath(os.fspath(staging_dir))
        os.makedirs(work_dir, exist_ok=True)

    weight_map: dict[str, str] = {}
    shard_digests: list[dict[str, Any]] = []
    uploaded_files: list[str] = []
    peak_temp_disk_bytes = 0
    commits: list[str] = []

    try:
        for shard_idx, (shard_name, shard_entries) in enumerate(planned, start=1):
            tensors: dict[str, torch.Tensor] = {}
            for ent in shard_entries:
                tensors[ent.key] = ent.materialize()
                weight_map[ent.key] = shard_name
            shard_path = os.path.join(work_dir, shard_name)
            size_bytes = save_safetensors_file(
                tensors,
                shard_path,
                metadata={"qloop_pack": "3.0", "shard": f"{shard_idx}/{len(planned)}"},
            )
            del tensors
            peak_temp_disk_bytes = max(peak_temp_disk_bytes, int(size_bytes))
            sha256_hex = _sha256_file(shard_path)

            try:
                ref = _upload_one_file(
                    hub_api,
                    repo_id,
                    shard_path,
                    shard_name,
                    f"{commit_message} [{shard_idx}/{len(planned)}: {shard_name}]",
                )
                commits.append(ref)
                uploaded_files.append(shard_name)
            finally:
                # Immediately unlink shard before materializing the next shard!
                if os.path.exists(shard_path):
                    os.unlink(shard_path)
                if clean_hf_cache_each_shard:
                    _purge_hf_transient_upload_cache()

            shard_digests.append(
                {
                    "filename": shard_name,
                    "num_tensors": len(shard_entries),
                    "size_bytes": int(size_bytes),
                    "sha256": sha256_hex,
                }
            )

        meta_files, manifest = _write_non_shard_pack_files(
            work_dir,
            repo_id=repo_id,
            base_cfg=base_cfg,
            arch=arch,
            tokenizer=src.tokenizer,
            weight_map=weight_map,
            shard_digests=shard_digests,
            entries=entries,
            merged_experts=merged_cnt,
            eval_summary=eval_summary,
        )
        meta_bytes = sum(os.path.getsize(os.path.join(work_dir, fn)) for fn in meta_files)
        peak_temp_disk_bytes = max(peak_temp_disk_bytes, int(meta_bytes))

        try:
            meta_ref = _upload_metadata_batch(
                hub_api,
                repo_id,
                work_dir,
                meta_files,
                f"{commit_message} [index, custom modeling & model card]",
            )
            commits.append(meta_ref)
            uploaded_files.extend(meta_files)
        finally:
            for fn in meta_files:
                fp = os.path.join(work_dir, fn)
                if os.path.exists(fp):
                    os.unlink(fp)
            if clean_hf_cache_each_shard:
                _purge_hf_transient_upload_cache()
    finally:
        if own_tmp is not None:
            own_tmp.cleanup()

    log.info(
        "streamed Whole Model Pack -> hf://%s (%d shard(s), peak_temp=%.2f MiB, total=%.2f MiB)",
        repo_id,
        len(planned),
        peak_temp_disk_bytes / (1024.0 * 1024.0),
        manifest["total_bytes"] / (1024.0 * 1024.0),
    )
    return {
        "repo_id": repo_id,
        "stream_low_disk": True,
        "num_shards": len(planned),
        "uploaded_files": uploaded_files,
        "peak_temp_disk_bytes": peak_temp_disk_bytes,
        "total_bytes": manifest["total_bytes"],
        "total_parameters": manifest["total_parameters"],
        "merged_hot_experts": merged_cnt,
        "commits": commits,
        "manifest": manifest,
    }


def push_whole_model_pack(
    stack_or_model: Any,
    repo_id: str,
    token: str | None = None,
    stream_low_disk: bool = True,
    *,
    tokenizer: Any = None,
    arch_spec: dict[str, Any] | None = None,
    max_shard_size_bytes: int = 4_500_000_000,
    merge_hot_experts: bool = True,
    merge_in_place: bool = True,
    private: bool = False,
    out_dir: str | os.PathLike[str] | None = None,
    commit_message: str = "Upload Qwen3.6-35B-A6B-LoopHSSD Whole Model Pack",
    eval_summary: dict[str, Any] | None = None,
    api: Any = None,
) -> dict[str, Any]:
    """Push a complete Whole Model Pack to Hugging Face Hub (defaults to Colab low-disk streaming)."""
    if stream_low_disk:
        return stream_whole_model_pack(
            stack_or_model,
            repo_id,
            token=token,
            tokenizer=tokenizer,
            arch_spec=arch_spec,
            max_shard_size_bytes=max_shard_size_bytes,
            merge_hot_experts=merge_hot_experts,
            merge_in_place=merge_in_place,
            private=private,
            commit_message=commit_message,
            staging_dir=out_dir,
            eval_summary=eval_summary,
            api=api,
        )

    hub_api = _resolve_hub_api(api, token)
    _ensure_hub_repo(hub_api, repo_id, private=private)

    own_tmp: tempfile.TemporaryDirectory[str] | None = None
    if out_dir is None:
        own_tmp = tempfile.TemporaryDirectory(prefix="qloop_full_pack_")
        export_dir = own_tmp.name
    else:
        export_dir = os.path.abspath(os.fspath(out_dir))

    try:
        exp = export_whole_model_pack(
            stack_or_model,
            export_dir,
            tokenizer=tokenizer,
            arch_spec=arch_spec,
            max_shard_size_bytes=max_shard_size_bytes,
            merge_hot_experts=merge_hot_experts,
            merge_in_place=merge_in_place,
            repo_id=repo_id,
            eval_summary=eval_summary,
        )
        commits: list[str] = []
        for sfile in exp["shard_files"]:
            ref = _upload_one_file(
                hub_api,
                repo_id,
                os.path.join(export_dir, sfile),
                sfile,
                f"{commit_message} [{sfile}]",
            )
            commits.append(ref)
        meta_ref = _upload_metadata_batch(
            hub_api,
            repo_id,
            export_dir,
            exp["metadata_files"],
            f"{commit_message} [metadata]",
        )
        commits.append(meta_ref)
        return {
            "repo_id": repo_id,
            "stream_low_disk": False,
            "out_dir": None if own_tmp is not None else export_dir,
            "num_shards": exp["num_shards"],
            "uploaded_files": list(exp["files"]),
            "peak_temp_disk_bytes": exp["total_bytes"],
            "total_bytes": exp["total_bytes"],
            "total_parameters": exp["total_parameters"],
            "merged_hot_experts": exp["merged_hot_experts"],
            "commits": commits,
            "manifest": exp["manifest"],
        }
    finally:
        if own_tmp is not None:
            own_tmp.cleanup()


# ============================================================================ pack loader

def load_whole_model_pack(
    pack_dir_or_repo: str | os.PathLike[str],
    *,
    device: str | torch.device | None = None,
    device_map: str | torch.device | None = None,
    dtype: str | torch.dtype | None = "auto",
    active_passes: int | None = None,
    token: str | None = None,
    revision: str | None = None,
    local_files_only: bool = False,
    strict: bool = True,
    api: Any = None,
) -> LoadedModelPack:
    """Hydrate a Whole Model Pack from a local directory or HF Hub repo into a `LoadedModelPack`."""
    local_dir, tmp_handle = resolve_pack_directory(
        pack_dir_or_repo,
        token=token,
        revision=revision,
        local_files_only=local_files_only,
        api=api,
    )
    try:
        hf_model = Qwen36LoopHSSDForCausalLM.from_pretrained(
            local_dir,
            device_map=device if device is not None else device_map,
            dtype=dtype,
            strict=strict,
        )
        if active_passes is not None:
            hf_model.set_active_passes(int(active_passes))
        tok = _load_pack_tokenizer(local_dir, token=resolve_hf_token(token))
        manifest: dict[str, Any] = {}
        manifest_path = os.path.join(local_dir, PACK_MANIFEST_FILE)
        if os.path.isfile(manifest_path):
            with open(manifest_path, encoding="utf-8") as f:
                manifest = json.load(f)
    finally:
        if tmp_handle is not None:
            tmp_handle.cleanup()

    parts = inspect_model(hf_model.backbone)
    pack = LoadedModelPack(
        model=hf_model.backbone,
        tokenizer=tok,
        parts=parts,
        loop=hf_model.loop_controller,
        gdn2=hf_model.gdn2_manager,
        lsa=hf_model.lsa_manager,
        deltas=hf_model.hot_deltas,
        load_info={
            "source": str(pack_dir_or_repo),
            "pack_type": "qloop_whole_model_pack",
            "active_passes": hf_model.active_passes,
        },
    )
    pack.hssd = hf_model.hssd_handle
    pack.hf_model = hf_model
    pack.pack_config = hf_model.config
    pack.manifest = manifest
    return pack


__all__ = [
    "GENERATION_CONFIG_FILE",
    "PACK_MANIFEST_FILE",
    "SAFETENSORS_INDEX_FILE",
    "LoadedModelPack",
    "WholeModelPackConfig",
    "export_whole_model_pack",
    "extract_arch_spec",
    "load_whole_model_pack",
    "push_whole_model_pack",
    "stream_whole_model_pack",
]
