"""Self-contained Hugging Face custom Config & CausalLM for Whole Model Packs.

Provides:
  * `Qwen36LoopHSSDConfig` & `Qwen36LoopHSSDForCausalLM`: subclasses `transformers.PretrainedConfig`
    and `transformers.PreTrainedModel` when `transformers` is installed, with a lightweight pure-PyTorch
    fallback when running offline unit tests without `transformers`.
  * `save_safetensors_file` & `load_safetensors_file`: uses `safetensors.torch` when installed, with a
    100% wire-compatible binary `.safetensors` serializer/deserializer fallback.
  * `write_standalone_hf_modeling_files(out_dir)`: emits clean, Python-3.9-compatible standalone
    `configuration_qwen36_loophssd.py` and `modeling_qwen36_loophssd.py` into exported Whole Model Packs
    so `AutoModelForCausalLM.from_pretrained(pack_dir_or_repo, trust_remote_code=True)` reconstructs the
    exact `Loop-HSSD v3` / `Loop-GDN2` architecture and loads sharded `.safetensors` weights cleanly.
"""
from __future__ import annotations

import copy
import json
import logging
import os
import struct
import tempfile
from dataclasses import fields
from typing import Any

import torch
from torch import nn

from ..debug import require
from ..hub import resolve_hf_token
from ..model.experts import HotExpertDeltas, attach_hot_expert_deltas
from ..model.gdn2 import GDN2Config, GDN2Manager, install_gdn2
from ..model.introspect import find_lm_head, find_text_model
from ..model.loading import resolve_device, resolve_dtype
from ..model.loop import LoopController, LoopSpec, apply_loop
from ..model.lsa import LSAConfig, LSAManager, apply_yarn, install_lsa
from ..model.quant_autograd import FP8_DTYPES, install_quantized_autograd_bridge

log = logging.getLogger("qloop")

CONFIG_FILENAME = "configuration_qwen36_loophssd.py"
MODELING_FILENAME = "modeling_qwen36_loophssd.py"
AUTO_MAP: dict[str, str] = {
    "AutoConfig": "configuration_qwen36_loophssd.Qwen36LoopHSSDConfig",
    "AutoModelForCausalLM": "modeling_qwen36_loophssd.Qwen36LoopHSSDForCausalLM",
}

# ============================================================================ binary safetensors I/O

_DTYPE_TO_ST: dict[torch.dtype, str] = {
    torch.float64: "F64",
    torch.float32: "F32",
    torch.float16: "F16",
    torch.bfloat16: "BF16",
    torch.int64: "I64",
    torch.int32: "I32",
    torch.int16: "I16",
    torch.int8: "I8",
    torch.uint8: "U8",
    torch.bool: "BOOL",
}
if hasattr(torch, "float8_e4m3fn"):
    _DTYPE_TO_ST[torch.float8_e4m3fn] = "F8_E4M3"
if hasattr(torch, "float8_e5m2"):
    _DTYPE_TO_ST[torch.float8_e5m2] = "F8_E5M2"
_ST_TO_DTYPE: dict[str, torch.dtype] = {v: k for k, v in _DTYPE_TO_ST.items()}


def _dedup_shared_storage(tensors: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Ensure every tensor is detached CPU contiguous and no two keys alias the same storage."""
    seen_ptrs: set[int] = set()
    clean: dict[str, torch.Tensor] = {}
    for k, t in tensors.items():
        tc = t.detach().to("cpu").contiguous()
        if tc.numel() > 0:
            ptr = int(tc.untyped_storage().data_ptr())
            if ptr in seen_ptrs:
                tc = tc.clone()
                ptr = int(tc.untyped_storage().data_ptr())
            seen_ptrs.add(ptr)
        clean[k] = tc
    return clean


def save_safetensors_file(
    tensors: dict[str, torch.Tensor],
    path: str,
    metadata: dict[str, str] | None = None,
) -> int:
    """Write a standard Hugging Face `.safetensors` shard file and return its byte size."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    meta = {"format": "pt"}
    if metadata:
        meta.update({str(k): str(v) for k, v in metadata.items()})
    clean = _dedup_shared_storage(tensors)
    try:
        from safetensors.torch import save_file  # type: ignore[import-not-found]

        save_file(clean, path, metadata=meta)
        return os.path.getsize(path)
    except ImportError:
        pass

    header: dict[str, Any] = {"__metadata__": meta}
    offset = 0
    prepared: list[torch.Tensor] = []
    for k, tc in clean.items():
        st_code = _DTYPE_TO_ST.get(tc.dtype)
        require(st_code is not None, f"unsupported dtype {tc.dtype} for safetensors key {k!r}")
        nbytes = int(tc.numel() * tc.element_size())
        header[k] = {
            "dtype": st_code,
            "shape": [int(d) for d in tc.shape],
            "data_offsets": [offset, offset + nbytes],
        }
        offset += nbytes
        prepared.append(tc)

    hdr_raw = json.dumps(header, separators=(",", ":")).encode("utf-8")
    pad = (8 - (len(hdr_raw) % 8)) % 8
    if pad:
        hdr_raw = hdr_raw + (b" " * pad)

    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(hdr_raw)))
        f.write(hdr_raw)
        for tc in prepared:
            if tc.numel() > 0:
                f.write(memoryview(tc.reshape(-1).view(torch.uint8).numpy()))
    return os.path.getsize(path)


def load_safetensors_file(path: str, device: str | torch.device = "cpu") -> dict[str, torch.Tensor]:
    """Load a `.safetensors` shard file into a `{name: Tensor}` dict."""
    dev_str = str(device)
    try:
        from safetensors.torch import load_file  # type: ignore[import-not-found]

        return dict(load_file(path, device=dev_str))
    except ImportError:
        pass

    out: dict[str, torch.Tensor] = {}
    with open(path, "rb") as f:
        raw_n = f.read(8)
        require(len(raw_n) == 8, f"truncated safetensors file {path!r}")
        (hdr_len,) = struct.unpack("<Q", raw_n)
        hdr = json.loads(f.read(hdr_len).decode("utf-8"))
        data_start = 8 + hdr_len
        for k, info in hdr.items():
            if k == "__metadata__":
                continue
            dt = _ST_TO_DTYPE.get(info["dtype"])
            require(dt is not None, f"unknown safetensors dtype {info['dtype']!r} for {k!r}")
            assert dt is not None
            shape = [int(d) for d in info["shape"]]
            beg, end = int(info["data_offsets"][0]), int(info["data_offsets"][1])
            nbytes = end - beg
            if nbytes == 0:
                t = torch.empty(shape, dtype=dt)
            else:
                f.seek(data_start + beg)
                buf = bytearray(nbytes)
                read_n = f.readinto(buf)
                require(read_n == nbytes, f"truncated tensor payload for {k!r} in {path!r}")
                t = torch.frombuffer(buf, dtype=torch.uint8).view(dt).reshape(shape).clone()
            if dev_str != "cpu":
                t = t.to(device)
            out[k] = t
    return out


# ============================================================================ HF base classes / fallbacks

try:
    from transformers import PretrainedConfig as _HFPretrainedConfig  # type: ignore[import-not-found]
    from transformers import PreTrainedModel as _HFPreTrainedModel  # type: ignore[import-not-found]

    try:
        from transformers import GenerationMixin as _HFGenerationMixin  # type: ignore[import-not-found]
    except ImportError:
        _HFGenerationMixin = object  # type: ignore[assignment,misc]

    _HAS_TRANSFORMERS = True
except ImportError:
    _HAS_TRANSFORMERS = False

    class _HFPretrainedConfig:  # type: ignore[no-redef]
        model_type: str = "qwen36_loophssd"

        def __init__(self, **kwargs: Any) -> None:
            self.tie_word_embeddings = bool(kwargs.pop("tie_word_embeddings", False))
            self.architectures = list(kwargs.pop("architectures", ["Qwen36LoopHSSDForCausalLM"]))
            for k, v in kwargs.items():
                setattr(self, k, v)

        def to_dict(self) -> dict[str, Any]:
            out = {k: copy.deepcopy(v) for k, v in self.__dict__.items() if not k.startswith("_")}
            out["model_type"] = self.model_type
            return out

        def to_json_string(self, use_diff: bool = False) -> str:
            return json.dumps(self.to_dict(), indent=2, sort_keys=True, default=str) + "\n"

        def save_pretrained(self, save_directory: str | os.PathLike[str], **_: Any) -> None:
            os.makedirs(save_directory, exist_ok=True)
            cfg_path = os.path.join(str(save_directory), "config.json")
            with open(cfg_path, "w", encoding="utf-8") as f:
                f.write(self.to_json_string())

    class _HFPreTrainedModel(nn.Module):  # type: ignore[no-redef]
        config_class: Any = None

        def __init__(self, config: Any) -> None:
            super().__init__()
            self.config = config

        def post_init(self) -> None:
            return None

    _HFGenerationMixin = object  # type: ignore[assignment,misc]


if (
    _HAS_TRANSFORMERS
    and isinstance(_HFGenerationMixin, type)
    and not issubclass(_HFPreTrainedModel, _HFGenerationMixin)
):

    class _ModelBase(_HFPreTrainedModel, _HFGenerationMixin):  # type: ignore[misc]
        pass

else:
    _ModelBase = _HFPreTrainedModel


# ============================================================================ pack resolution helper

def resolve_pack_directory(
    pack_dir_or_repo: str | os.PathLike[str],
    *,
    token: str | None = None,
    revision: str | None = None,
    local_files_only: bool = False,
    api: Any = None,
) -> tuple[str, tempfile.TemporaryDirectory[str] | None]:
    """Return `(local_dir, optional_temp_dir_handle)` for a local directory or HF Hub repo ID."""
    raw = os.fspath(pack_dir_or_repo)
    if os.path.isdir(raw):
        return os.path.abspath(raw), None

    tok = resolve_hf_token(token)
    if api is not None:
        tmp = tempfile.TemporaryDirectory(prefix="qloop_pack_pull_")
        files = list(api.list_repo_files(repo_id=raw, repo_type="model"))
        require(bool(files), f"repository {raw!r} contains no files on provided api")
        for rel in files:
            dst = os.path.join(tmp.name, rel)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            ok = api.download_file(raw, rel, dst)
            require(bool(ok), f"failed to download {rel!r} from {raw!r}")
        return tmp.name, tmp

    try:
        from huggingface_hub import snapshot_download  # type: ignore[import-not-found]
    except ImportError as e:
        raise RuntimeError(
            f"{raw!r} is not a local directory and `huggingface_hub` is not installed to download from Hub"
        ) from e
    cached_dir = snapshot_download(
        repo_id=raw,
        repo_type="model",
        token=tok,
        revision=revision,
        local_files_only=local_files_only,
    )
    return os.path.abspath(cached_dir), None


# ============================================================================ Config & CausalLM

class Qwen36LoopHSSDConfig(_HFPretrainedConfig):
    """Self-contained HF configuration storing the physical backbone config + `qloop_v3_arch` recipe."""

    model_type = "qwen36_loophssd"

    def __init__(
        self,
        base_config: dict[str, Any] | None = None,
        qloop_v3_arch: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        bcfg = copy.deepcopy(dict(base_config or {}))
        arch = copy.deepcopy(dict(qloop_v3_arch or {}))
        kwargs.setdefault("tie_word_embeddings", bool(bcfg.get("tie_word_embeddings", False)))
        kwargs.setdefault("architectures", ["Qwen36LoopHSSDForCausalLM"])
        kwargs.setdefault("auto_map", dict(AUTO_MAP))
        for mirror_key in (
            "vocab_size",
            "hidden_size",
            "num_hidden_layers",
            "num_attention_heads",
            "num_key_value_heads",
            "head_dim",
            "rms_norm_eps",
            "eos_token_id",
            "pad_token_id",
            "bos_token_id",
        ):
            if mirror_key in bcfg and mirror_key not in kwargs:
                kwargs[mirror_key] = bcfg[mirror_key]
        super().__init__(**kwargs)
        self.base_config = bcfg
        self.qloop_v3_arch = arch
        self.auto_map = dict(kwargs.get("auto_map") or AUTO_MAP)
        self.architectures = list(kwargs.get("architectures") or ["Qwen36LoopHSSDForCausalLM"])
        self.virtual_num_hidden_layers = int(
            arch.get("virtual_layers") or bcfg.get("num_hidden_layers") or getattr(self, "num_hidden_layers", 0)
        )

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | os.PathLike[str],
        *args: Any,
        token: str | None = None,
        revision: str | None = None,
        local_files_only: bool = False,
        api: Any = None,
        **kwargs: Any,
    ) -> Qwen36LoopHSSDConfig:
        return_unused = bool(kwargs.pop("return_unused_kwargs", False))
        kwargs.pop("trust_remote_code", None)
        local_dir, tmp_handle = resolve_pack_directory(
            pretrained_model_name_or_path,
            token=token,
            revision=revision,
            local_files_only=local_files_only,
            api=api,
        )
        try:
            cfg_path = os.path.join(local_dir, "config.json")
            require(os.path.isfile(cfg_path), f"config.json not found in {local_dir!r}")
            with open(cfg_path, encoding="utf-8") as f:
                raw = json.load(f)
        finally:
            if tmp_handle is not None:
                tmp_handle.cleanup()
        raw.update(kwargs)
        cfg = cls(**raw)
        if return_unused:
            return cfg, {}  # type: ignore[return-value]
        return cfg


class _DepthContext:
    """Dual-purpose depth handle: works BOTH as `engine.depth(2)` setter AND `with engine.depth(2):`."""

    def __init__(self, owner: Any, prev_passes: int, target_passes: int) -> None:
        self._owner = owner
        self._prev = int(prev_passes)
        self.target = int(target_passes)

    def __enter__(self) -> Any:
        self._owner.set_active_passes(self.target)
        return self._owner

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self._owner.set_active_passes(self._prev)


class CausalLMPackOutput:
    """HF `CausalLMOutputWithPast`-compatible container supporting attribute, tuple, and dict access."""

    def __init__(
        self,
        logits: torch.Tensor,
        loss: torch.Tensor | None = None,
        past_key_values: Any = None,
        hidden_states: Any = None,
        attentions: Any = None,
    ) -> None:
        self.logits = logits
        self.loss = loss
        self.past_key_values = past_key_values
        self.hidden_states = hidden_states
        self.attentions = attentions

    def __getitem__(self, key: int | str) -> Any:
        if isinstance(key, str):
            return getattr(self, key)
        items = [x for x in (self.loss, self.logits, self.past_key_values, self.hidden_states) if x is not None]
        return items[key]


def _build_physical_backbone(config: Qwen36LoopHSSDConfig, dtype_override: torch.dtype | None = None) -> nn.Module:
    arch = config.qloop_v3_arch or {}
    bcfg = copy.deepcopy(dict(config.base_config or {}))
    base_dtype = dtype_override or resolve_dtype(arch.get("base_dtype", "float32"))
    kind = arch.get("backbone_kind")
    if kind is None:
        kind = "toy" if str(bcfg.get("model_type", "")).startswith("toy") else "hf"

    if kind == "toy":
        from ..model.toy import ToyConfig, ToyForCausalLM

        known_toy = {f.name for f in fields(ToyConfig)}
        toy_kw = {k: v for k, v in bcfg.items() if k in known_toy}
        if toy_kw.get("_attn_implementation") == "qloop_lsa":
            toy_kw["_attn_implementation"] = "eager"
        tcfg = ToyConfig(**toy_kw)
        for extra_k in ("max_position_embeddings", "rope_scaling", "rope_parameters"):
            if extra_k in bcfg:
                setattr(tcfg, extra_k, copy.deepcopy(bcfg[extra_k]))
        return ToyForCausalLM(tcfg).to(dtype=base_dtype)

    try:
        import transformers  # type: ignore[import-not-found]
        from transformers import AutoConfig, AutoModelForCausalLM  # type: ignore[import-not-found]
    except ImportError as e:
        raise RuntimeError(
            "Loading a real Hugging Face Qwen3.6 backbone requires `transformers` installed"
        ) from e

    base_model_type = str(arch.get("base_model_type") or bcfg.get("model_type") or "qwen3_5_moe")
    clean_bcfg = {k: v for k, v in bcfg.items() if k != "model_type"}
    attn_impl = str(clean_bcfg.pop("_attn_implementation", "sdpa") or "sdpa")
    if attn_impl == "qloop_lsa":
        attn_impl = "sdpa"
    hf_cfg = AutoConfig.for_model(base_model_type, **clean_bcfg)
    hf_cfg._attn_implementation = attn_impl
    if arch.get("yarn_target_len"):
        apply_yarn(hf_cfg, int(arch["yarn_target_len"]))
    ver_head = str(getattr(transformers, "__version__", "5")).split(".", 1)[0]
    dtype_kw = "dtype" if (ver_head.isdigit() and int(ver_head) >= 5) else "torch_dtype"
    return AutoModelForCausalLM.from_config(hf_cfg, attn_implementation=attn_impl, **{dtype_kw: base_dtype})


class Qwen36LoopHSSDForCausalLM(_ModelBase):
    """Whole-Model Hugging Face wrapper for `Qwen3.6-35B-A6B-LoopHSSD` (and `ToyForCausalLM` packs)."""

    config_class = Qwen36LoopHSSDConfig
    supports_gradient_checkpointing = True

    def __init__(
        self,
        config: Qwen36LoopHSSDConfig,
        *,
        _defer_loop: bool = False,
        _dtype_override: torch.dtype | None = None,
    ) -> None:
        super().__init__(config)
        self.config = config
        self.backbone: nn.Module = _build_physical_backbone(config, dtype_override=_dtype_override)
        install_quantized_autograd_bridge(self.backbone)
        self.gdn2_manager: GDN2Manager | None = None
        self.hssd_handle: Any = None
        self.lsa_manager: LSAManager | None = None
        self.hot_deltas: HotExpertDeltas | None = None
        self.loop_controller: LoopController | None = None

        arch = config.qloop_v3_arch or {}
        gcfg = dict(arch.get("gdn2") or {})
        if gcfg.pop("enabled", False):
            self.gdn2_manager = install_gdn2(self.backbone, GDN2Config.from_dict(gcfg))

        hcfg = dict(arch.get("hssd") or {})
        if hcfg.pop("enabled", False):
            from ..model import hssd as hssd_mod  # type: ignore[attr-defined]

            apply_fn = getattr(hssd_mod, "apply_hssd", None) or getattr(hssd_mod, "install_hssd", None)
            require(callable(apply_fn), "qloop.model.hssd has no apply_hssd() function")
            hcfg_cls = getattr(hssd_mod, "HSSDConfig", None)
            if hcfg_cls is not None and hasattr(hcfg_cls, "__dataclass_fields__"):
                known_h = set(hcfg_cls.__dataclass_fields__.keys()) | {"top_k_pages"}
                hcfg = {k: v for k, v in hcfg.items() if k in known_h}
            self.hssd_handle = apply_fn(self.backbone, hcfg)

        lcfg = dict(arch.get("lsa") or {})
        if lcfg.pop("enabled", False):
            self.lsa_manager = install_lsa(self.backbone, LSAConfig.from_dict(lcfg))

        he_cfg = dict(arch.get("hot_experts") or {})
        if he_cfg.get("enabled", False) and not he_cfg.get("merged_into_base", True):
            raw_hm = he_cfg.get("hot_map") or {}
            hot_map = {int(k): [int(e) for e in v] for k, v in raw_hm.items()}
            if hot_map:
                self.hot_deltas = attach_hot_expert_deltas(
                    self.backbone,
                    hot_map,
                    rank=int(he_cfg.get("rank", 0)),
                    alpha=he_cfg.get("alpha"),
                )

        if not _defer_loop:
            self._finalize_loop(None)

    def _finalize_loop(self, loop_tensors: dict[str, torch.Tensor] | None) -> None:
        arch = self.config.qloop_v3_arch or {}
        loop_cfg = dict(arch.get("loop") or {})
        active_passes = loop_cfg.pop("active_passes", None)
        if loop_cfg.pop("enabled", False):
            spec = LoopSpec.from_dict(loop_cfg)
            self.loop_controller = apply_loop(self.backbone, spec)
            if loop_tensors is not None:
                ad_params = dict(self.loop_controller.adapters.named_parameters())
                ad_bufs = dict(self.loop_controller.adapters.named_buffers())
                missing = [k for k in ad_params if k not in loop_tensors]
                unexpected = [k for k in loop_tensors if k not in ad_params and k not in ad_bufs]
                require(
                    not missing and not unexpected,
                    "loop_adapters state mismatch during Whole Model Pack load",
                    missing=missing,
                    unexpected=unexpected,
                )
                with torch.no_grad():
                    for k, t in loop_tensors.items():
                        target = ad_params.get(k) if k in ad_params else ad_bufs[k]
                        require(
                            tuple(target.shape) == tuple(t.shape),
                            f"shape mismatch for loop_adapters.{k}",
                            expected=target,
                            loaded=t,
                        )
                        target.copy_(t.to(device=target.device, dtype=target.dtype))
            if active_passes is not None:
                self.loop_controller.set_active_passes(int(active_passes))

    @property
    def _text_model(self) -> nn.Module:
        return find_text_model(self.backbone)

    @property
    def _lm_head(self) -> nn.Module:
        head = find_lm_head(self.backbone, getattr(self.config, "vocab_size", None))
        require(head is not None, "backbone has no lm_head module")
        assert head is not None
        return head

    def get_input_embeddings(self) -> nn.Module:
        return self._text_model.embed_tokens

    def set_input_embeddings(self, value: nn.Module) -> None:
        self._text_model.embed_tokens = value

    def get_output_embeddings(self) -> nn.Module:
        return self._lm_head

    def set_output_embeddings(self, new_embeddings: nn.Module) -> None:
        if hasattr(self.backbone, "lm_head"):
            self.backbone.lm_head = new_embeddings

    @property
    def active_passes(self) -> int:
        return 1 if self.loop_controller is None else int(self.loop_controller.active_passes)

    @property
    def max_passes(self) -> int:
        return 1 if self.loop_controller is None else int(self.loop_controller.spec.num_passes)

    def set_active_passes(self, r: int) -> None:
        if self.loop_controller is None:
            require(int(r) == 1, f"model has no LoopController; cannot set active_passes={r}")
            return
        self.loop_controller.set_active_passes(int(r))

    def depth(self, r: int) -> _DepthContext:
        prev = self.active_passes
        self.set_active_passes(int(r))
        return _DepthContext(self, prev, int(r))

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        past_key_values: Any = None,
        inputs_embeds: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        logits_to_keep: int | None = None,
        **kwargs: Any,
    ) -> Any:
        if self.loop_controller is not None and past_key_values is not None:
            self.loop_controller._sync_cache_last_linear(past_key_values)
        if logits_to_keep is not None and labels is None:
            tm_kw: dict[str, Any] = {
                "input_ids": input_ids,
                "past_key_values": past_key_values,
                "use_cache": bool(use_cache),
            }
            if inputs_embeds is not None:
                tm_kw["inputs_embeds"] = inputs_embeds
            if attention_mask is not None:
                tm_kw["attention_mask"] = attention_mask
            if position_ids is not None:
                tm_kw["position_ids"] = position_ids
            if output_hidden_states is not None:
                tm_kw["output_hidden_states"] = output_hidden_states
            tm_out = self._text_model(**tm_kw, **kwargs)
            h_last = tm_out.last_hidden_state
            keep = int(logits_to_keep)
            h_slice = h_last[:, -keep:, :] if keep > 0 else h_last[:, :0, :]
            logits = self._lm_head(h_slice)
            out = CausalLMPackOutput(
                logits=logits,
                loss=None,
                past_key_values=getattr(tm_out, "past_key_values", None),
                hidden_states=getattr(tm_out, "hidden_states", None),
            )
            return out if return_dict is not False else (logits, out.past_key_values)

        call_kw: dict[str, Any] = dict(kwargs)
        if attention_mask is not None:
            call_kw["attention_mask"] = attention_mask
        if position_ids is not None:
            call_kw["position_ids"] = position_ids
        if past_key_values is not None:
            call_kw["past_key_values"] = past_key_values
        if inputs_embeds is not None:
            call_kw["inputs_embeds"] = inputs_embeds
        if labels is not None:
            call_kw["labels"] = labels
        if use_cache is not None:
            call_kw["use_cache"] = use_cache
        if output_attentions is not None:
            call_kw["output_attentions"] = output_attentions
        if output_hidden_states is not None:
            call_kw["output_hidden_states"] = output_hidden_states
        res = self.backbone(input_ids=input_ids, **call_kw)
        if return_dict is False:
            return (res.logits, getattr(res, "past_key_values", None))
        return res

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.Tensor,
        past_key_values: Any = None,
        attention_mask: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        if past_key_values is not None:
            seq_len = getattr(past_key_values, "get_seq_length", lambda: 0)()
            if seq_len > 0 and input_ids.shape[1] > 1:
                input_ids = input_ids[:, -1:]
        return {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "use_cache": True,
            "attention_mask": attention_mask,
            **kwargs,
        }

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | os.PathLike[str],
        *model_args: Any,
        config: Qwen36LoopHSSDConfig | None = None,
        token: str | None = None,
        revision: str | None = None,
        local_files_only: bool = False,
        device_map: str | torch.device | None = None,
        torch_dtype: str | torch.dtype | None = None,
        dtype: str | torch.dtype | None = None,
        strict: bool = True,
        api: Any = None,
        **kwargs: Any,
    ) -> Qwen36LoopHSSDForCausalLM:
        del model_args
        kwargs.pop("trust_remote_code", None)
        explicit_dt = dtype if dtype is not None else torch_dtype
        dt_override: torch.dtype | None = None
        if isinstance(explicit_dt, torch.dtype):
            dt_override = explicit_dt
        elif isinstance(explicit_dt, str) and explicit_dt != "auto":
            dt_override = resolve_dtype(explicit_dt)

        local_dir, tmp_handle = resolve_pack_directory(
            pretrained_model_name_or_path,
            token=token,
            revision=revision,
            local_files_only=local_files_only,
            api=api,
        )
        try:
            if config is None:
                config = Qwen36LoopHSSDConfig.from_pretrained(local_dir)
            elif not isinstance(config, Qwen36LoopHSSDConfig):
                cfg_dict = config.to_dict() if hasattr(config, "to_dict") else dict(vars(config))
                config = Qwen36LoopHSSDConfig(**cfg_dict)

            instance = cls(config, _defer_loop=True, _dtype_override=dt_override)
            param_map = dict(instance.backbone.named_parameters())
            buf_map = dict(instance.backbone.named_buffers())
            mod_map = dict(instance.backbone.named_modules())
            expected_params = set(param_map.keys())
            q_arch = dict((config.qloop_v3_arch or {}).get("quantization") or {})
            nvfp4_mod_meta = dict(q_arch.get("nvfp4_modules") or {})
            dyn_quant_attrs = {
                "weight_packed",
                "weight_scale",
                "weight_scale_inv",
                "weight_global_scale",
                "weight_scale_2",
                "gate_up_proj_packed",
                "gate_up_proj_scale",
                "gate_up_proj_scale_inv",
                "gate_up_proj_global_scale",
                "gate_up_global_scale",
                "gate_up_proj_scale_2",
                "down_proj_packed",
                "down_proj_scale",
                "down_proj_scale_inv",
                "down_proj_global_scale",
                "down_global_scale",
                "down_proj_scale_2",
                "scale_inv",
                "global_scale",
                "input_scale",
                "input_global_scale",
                "activation_scale",
                "k_scale",
                "v_scale",
                "q_scale",
                "o_scale",
                "weight_shape",
            }
            replaced_dense_for = {
                "gate_up_proj_packed": "gate_up_proj",
                "down_proj_packed": "down_proj",
                "weight_packed": "weight",
            }

            index_path = os.path.join(local_dir, "model.safetensors.index.json")
            single_path = os.path.join(local_dir, "model.safetensors")
            if os.path.isfile(index_path):
                with open(index_path, encoding="utf-8") as f:
                    idx_data = json.load(f)
                weight_map = dict(idx_data.get("weight_map") or {})
                shard_files = list(dict.fromkeys(weight_map.values()))
            elif os.path.isfile(single_path):
                shard_files = ["model.safetensors"]
            else:
                found = sorted(fn for fn in os.listdir(local_dir) if fn.endswith(".safetensors"))
                require(bool(found), f"no .safetensors files found in {local_dir!r}")
                shard_files = found

            loaded_backbone: set[str] = set()
            unexpected_keys: list[str] = []
            loop_tensors: dict[str, torch.Tensor] = {}

            for sfile in shard_files:
                spath = os.path.join(local_dir, sfile)
                require(os.path.isfile(spath), f"missing shard file {spath!r}")
                shard_sd = load_safetensors_file(spath, device="cpu")
                for raw_key, tensor in shard_sd.items():
                    key = raw_key[len("backbone."):] if raw_key.startswith("backbone.") else raw_key
                    if key.startswith("loop_adapters."):
                        loop_tensors[key[len("loop_adapters."):]] = tensor
                        continue
                    marker = ".loop_adapters."
                    if marker in key:
                        loop_tensors[key.split(marker, 1)[1]] = tensor
                        continue
                    target = param_map.get(key)
                    is_param = target is not None
                    if target is None:
                        target = buf_map.get(key)
                    if (
                        target is None
                        and bool(getattr(config, "tie_word_embeddings", False))
                        and key.endswith("lm_head.weight")
                    ):
                        head_mod = getattr(instance.backbone, "lm_head", None)
                        if head_mod is not None and isinstance(getattr(head_mod, "weight", None), torch.Tensor):
                            target = head_mod.weight
                            is_param = True
                    if target is None:
                        p_path, sep, a_name = key.rpartition(".")
                        p_mod = mod_map.get(p_path) if sep else None
                        if (
                            p_mod is None
                            and sep
                            and p_path.endswith(".kv_cache")
                            and a_name in ("k_scale", "v_scale", "q_scale", "quant_scale")
                        ):
                            gp_path, gp_sep, kvc_nm = p_path.rpartition(".")
                            gp_mod = mod_map.get(gp_path) if gp_sep else None
                            if gp_mod is not None and kvc_nm == "kv_cache":
                                kvc_stub = nn.Module()
                                gp_mod.kv_cache = kvc_stub  # type: ignore[attr-defined]
                                mod_map[p_path] = kvc_stub
                                p_mod = kvc_stub
                        if p_mod is not None and a_name in dyn_quant_attrs:
                            dense_attr = replaced_dense_for.get(a_name)
                            if dense_attr is not None:
                                dense_key = f"{p_path}.{dense_attr}" if p_path else dense_attr
                                p_mod._parameters.pop(dense_attr, None)
                                p_mod._buffers.pop(dense_attr, None)
                                param_map.pop(dense_key, None)
                                buf_map.pop(dense_key, None)
                                expected_params.discard(dense_key)
                            m_meta = nvfp4_mod_meta.get(p_path) or {}
                            if m_meta.get("storage") == "buffers":
                                p_mod.register_buffer(a_name, tensor.clone())
                                buf_map[key] = p_mod._buffers[a_name]  # type: ignore[assignment]
                            else:
                                new_p = nn.Parameter(tensor.clone(), requires_grad=False)
                                p_mod.register_parameter(a_name, new_p)
                                param_map[key] = new_p
                                expected_params.add(key)
                                loaded_backbone.add(key)
                            if a_name.endswith("_packed") or m_meta:
                                if getattr(p_mod, "group_size", None) is None:
                                    p_mod.group_size = int(m_meta.get("group_size", 16))  # type: ignore[attr-defined]
                                sm = m_meta.get("scale_mode")
                                if sm in ("divisor", "multiplier"):
                                    p_mod._qloop_nvfp4_scale_mode = str(sm)  # type: ignore[attr-defined]
                                elif a_name.endswith("_global_scale") and not hasattr(
                                    p_mod, "_qloop_nvfp4_scale_mode"
                                ):
                                    p_mod._qloop_nvfp4_scale_mode = "divisor"  # type: ignore[attr-defined]
                            continue
                        unexpected_keys.append(key)
                        continue
                    require(
                        tuple(target.shape) == tuple(tensor.shape),
                        f"shape mismatch for {key}: model {tuple(target.shape)} vs shard {tuple(tensor.shape)}",
                    )
                    with torch.no_grad():
                        if tensor.dtype in FP8_DTYPES or not tensor.is_floating_point():
                            if isinstance(target, nn.Parameter):
                                target.data = tensor.to(device=target.device)
                            else:
                                p_path, _, a_nm = key.rpartition(".")
                                p_mod = mod_map.get(p_path)
                                if p_mod is not None and a_nm in p_mod._buffers:
                                    p_mod._buffers[a_nm] = tensor.to(device=target.device)
                                else:
                                    target.copy_(tensor.to(device=target.device, dtype=target.dtype))
                        elif dt_override is None and target.dtype != tensor.dtype:
                            target.data = tensor.to(device=target.device, dtype=tensor.dtype)
                        else:
                            target.copy_(tensor.to(device=target.device, dtype=target.dtype))
                    if is_param:
                        loaded_backbone.add(key)
                del shard_sd

            if bool(getattr(config, "tie_word_embeddings", False)):
                head_mod = getattr(instance.backbone, "lm_head", None)
                emb_mod = getattr(instance._text_model, "embed_tokens", None)
                if (
                    head_mod is not None
                    and emb_mod is not None
                    and isinstance(getattr(head_mod, "weight", None), nn.Parameter)
                    and isinstance(getattr(emb_mod, "weight", None), nn.Parameter)
                ):
                    head_mod.weight = emb_mod.weight
                    if any(lk.endswith("embed_tokens.weight") for lk in loaded_backbone):
                        for ek in list(expected_params):
                            if ek.endswith("lm_head.weight"):
                                loaded_backbone.add(ek)

            if strict:
                missing_backbone = sorted(expected_params - loaded_backbone)
                require(
                    not missing_backbone and not unexpected_keys,
                    "Whole Model Pack state_dict mismatch",
                    missing=missing_backbone[:10],
                    unexpected=unexpected_keys[:10],
                )

            install_quantized_autograd_bridge(instance.backbone)
            instance._finalize_loop(loop_tensors)
        finally:
            if tmp_handle is not None:
                tmp_handle.cleanup()

        target_dev: str | torch.device = "cpu"
        if isinstance(device_map, torch.device):
            target_dev = device_map
        elif isinstance(device_map, str):
            if device_map == "auto":
                target_dev = "cuda:0" if torch.cuda.is_available() else "cpu"
            else:
                target_dev = resolve_device(device_map)
        elif kwargs.get("device") is not None:
            target_dev = kwargs["device"]

        instance.to(target_dev)
        install_quantized_autograd_bridge(instance.backbone)
        for p in instance.parameters():
            p.requires_grad_(False)
        instance.eval()
        return instance


# ============================================================================ standalone HF files emitter

_STANDALONE_CONFIG_SRC = '''"""Standalone Hugging Face Config for Qwen3.6-35B-A6B-LoopHSSD Whole Model Packs."""
from __future__ import annotations

import copy
import json
import os
from typing import Any

try:
    from transformers import PretrainedConfig
except ImportError:

    class PretrainedConfig:  # type: ignore[no-redef]
        model_type: str = "qwen36_loophssd"

        def __init__(self, **kwargs: Any) -> None:
            self.tie_word_embeddings = bool(kwargs.pop("tie_word_embeddings", False))
            self.architectures = list(kwargs.pop("architectures", ["Qwen36LoopHSSDForCausalLM"]))
            for k, v in kwargs.items():
                setattr(self, k, v)

        def to_dict(self) -> dict[str, Any]:
            out = {k: copy.deepcopy(v) for k, v in self.__dict__.items() if not k.startswith("_")}
            out["model_type"] = self.model_type
            return out

        def to_json_string(self, use_diff: bool = False) -> str:
            del use_diff
            return json.dumps(self.to_dict(), indent=2, sort_keys=True, default=str) + "\\n"

        def save_pretrained(self, save_directory: str | os.PathLike[str], **_: Any) -> None:
            os.makedirs(save_directory, exist_ok=True)
            with open(os.path.join(str(save_directory), "config.json"), "w", encoding="utf-8") as f:
                f.write(self.to_json_string())

        @classmethod
        def from_pretrained(cls, pretrained_model_name_or_path: str | os.PathLike[str], **kw: Any):
            kw.pop("trust_remote_code", None)
            cfg_p = os.path.join(str(pretrained_model_name_or_path), "config.json")
            with open(cfg_p, encoding="utf-8") as f:
                data = json.load(f)
            data.update(kw)
            return cls(**data)


AUTO_MAP: dict[str, str] = {
    "AutoConfig": "configuration_qwen36_loophssd.Qwen36LoopHSSDConfig",
    "AutoModelForCausalLM": "modeling_qwen36_loophssd.Qwen36LoopHSSDForCausalLM",
}


class Qwen36LoopHSSDConfig(PretrainedConfig):
    """Configuration for `Qwen3.6-35B-A6B-LoopHSSD` (`R in {1, 2, 3}` -> `40L / 64L / 88L`)."""

    model_type = "qwen36_loophssd"

    def __init__(
        self,
        base_config: dict[str, Any] | None = None,
        qloop_v3_arch: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        bcfg = copy.deepcopy(dict(base_config or {}))
        arch = copy.deepcopy(dict(qloop_v3_arch or {}))
        kwargs.setdefault("tie_word_embeddings", bool(bcfg.get("tie_word_embeddings", False)))
        kwargs.setdefault("architectures", ["Qwen36LoopHSSDForCausalLM"])
        kwargs.setdefault("auto_map", dict(AUTO_MAP))
        for key in (
            "vocab_size",
            "hidden_size",
            "num_hidden_layers",
            "num_attention_heads",
            "num_key_value_heads",
            "head_dim",
            "rms_norm_eps",
            "eos_token_id",
            "pad_token_id",
            "bos_token_id",
        ):
            if key in bcfg and key not in kwargs:
                kwargs[key] = bcfg[key]
        super().__init__(**kwargs)
        self.base_config = bcfg
        self.qloop_v3_arch = arch
        self.auto_map = dict(kwargs.get("auto_map") or AUTO_MAP)
        self.architectures = list(kwargs.get("architectures") or ["Qwen36LoopHSSDForCausalLM"])
        self.virtual_num_hidden_layers = int(
            arch.get("virtual_layers") or bcfg.get("num_hidden_layers") or getattr(self, "num_hidden_layers", 0)
        )
'''

_STANDALONE_MODELING_SRC = '''"""Standalone Hugging Face Modeling for Qwen3.6-35B-A6B-LoopHSSD Whole Model Packs.

Supports `AutoModelForCausalLM.from_pretrained(pack_dir_or_repo, trust_remote_code=True)`:
  1. When `qloop` is installed, delegates directly to `qloop.infer.hf_custom_modeling` so every
     surgical subsystem (`Loop`, `GDN-2`, `HSSD v3`, `D-LSA`, `ToyForCausalLM`) runs natively.
  2. When `qloop` is NOT installed (standalone `transformers` + `torch` environment), reconstructs
     the HF base backbone, installs standalone `GDN-2` / `HSSD v3` + `D-LSA` (`num_query_passes`)
     + `Sandwich Loop` (`ReEntryBridge`, `PassAttnAdapters`, `PassRouterAdapters`, `MoR`, `R=1/2/3`),
     and streams sharded `.safetensors` weights with strict zero-dropped-key verification.
"""
from __future__ import annotations

import copy
import functools
import importlib.util
import json
import math
import os
import struct
import sys
import types
from collections import OrderedDict
from typing import Any, Callable

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

try:
    from .configuration_qwen36_loophssd import Qwen36LoopHSSDConfig
except ImportError:
    from configuration_qwen36_loophssd import Qwen36LoopHSSDConfig  # type: ignore[no-redef]


_HAS_QLOOP = importlib.util.find_spec("qloop") is not None

if _HAS_QLOOP:
    from qloop.infer.hf_custom_modeling import Qwen36LoopHSSDForCausalLM
else:
    from transformers import AutoConfig, AutoModelForCausalLM, PreTrainedModel

    try:
        from transformers import GenerationMixin
    except ImportError:
        GenerationMixin = object  # type: ignore[assignment,misc]

    _BaseCls = (
        type("_StandaloneBase", (PreTrainedModel, GenerationMixin), {})
        if isinstance(GenerationMixin, type) and not issubclass(PreTrainedModel, GenerationMixin)
        else PreTrainedModel
    )

    _ST_MAP: dict[str, torch.dtype] = {
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
    if hasattr(torch, "float8_e4m3fn"):
        _ST_MAP["F8_E4M3"] = torch.float8_e4m3fn
    if hasattr(torch, "float8_e5m2"):
        _ST_MAP["F8_E5M2"] = torch.float8_e5m2

    def _upcast(x: torch.Tensor) -> torch.Tensor:
        return x if x.dtype == torch.float64 else x.float()

    def _resolve_dt(spec: Any) -> torch.dtype:
        if isinstance(spec, torch.dtype):
            return spec
        s = str(spec or "float32").replace("torch.", "").lower()
        return {
            "float32": torch.float32,
            "fp32": torch.float32,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float16": torch.float16,
            "fp16": torch.float16,
            "float64": torch.float64,
            "fp64": torch.float64,
        }.get(s, torch.float32)

    def _load_st(path: str) -> dict[str, torch.Tensor]:
        try:
            from safetensors.torch import load_file

            return dict(load_file(path, device="cpu"))
        except ImportError:
            pass
        out: dict[str, torch.Tensor] = {}
        with open(path, "rb") as f:
            (hlen,) = struct.unpack("<Q", f.read(8))
            hdr = json.loads(f.read(hlen).decode("utf-8"))
            base = 8 + hlen
            for k, info in hdr.items():
                if k == "__metadata__":
                    continue
                dt = _ST_MAP[info["dtype"]]
                shp = [int(d) for d in info["shape"]]
                beg, end = int(info["data_offsets"][0]), int(info["data_offsets"][1])
                nb = end - beg
                if nb == 0:
                    out[k] = torch.empty(shp, dtype=dt)
                else:
                    f.seek(base + beg)
                    buf = bytearray(nb)
                    f.readinto(buf)
                    out[k] = torch.frombuffer(buf, dtype=torch.uint8).view(dt).reshape(shp).clone()
        return out

    def _logistic_shift(beta: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        s = torch.tanh(0.5 * delta)
        denom = (1.0 + (2.0 * beta - 1.0) * s).clamp_min(1e-12)
        return (beta + (2.0 * beta * (1.0 - beta) * s) / denom).clamp(0.0, 1.0)

    def _safe_sqrt(x: torch.Tensor) -> torch.Tensor:
        pos = x > 0.0
        safe = torch.where(pos, x, torch.ones_like(x))
        return torch.where(pos, torch.sqrt(safe), torch.zeros_like(x))

    def _safe_norm(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
        return _safe_sqrt((x * x).sum(dim=dim))

    def _rms_norm_scaled(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        xf = _upcast(x)
        return (xf * torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + eps)) * _upcast(weight)

    def _apply_2d_rotation(x: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        pairs = x.reshape(*x.shape[:-1], x.shape[-1] // 2, 2)
        cos_t = torch.cos(theta).to(dtype=x.dtype)
        sin_t = torch.sin(theta).to(dtype=x.dtype)
        x0, x1 = pairs[..., 0], pairs[..., 1]
        return torch.stack([x0 * cos_t - x1 * sin_t, x0 * sin_t + x1 * cos_t], dim=-1).reshape_as(x)

    def _compact(t: torch.Tensor) -> torch.Tensor:
        c = t.contiguous()
        need = int(c.numel() * c.element_size())
        if int(c.untyped_storage().nbytes()) != need or int(c.storage_offset()) != 0:
            return c.clone()
        return c

    def _is_hssd_state_duck(obj: Any) -> bool:
        return (
            not isinstance(obj, torch.Tensor)
            and hasattr(obj, "page_states")
            and hasattr(obj, "boundary_ring")
            and callable(getattr(obj, "_iter_tensors", None))
        )

    def _ensure_dynamo_mark_static_address_compat() -> None:
        dynamo = getattr(torch, "_dynamo", None)
        orig_fn = getattr(dynamo, "mark_static_address", None) if dynamo is not None else None
        if not callable(orig_fn) or getattr(orig_fn, "_qloop_hssd_aware", False):
            return

        @functools.wraps(orig_fn)
        def _hssd_mark_static_address(t: Any, *args: Any, **kwargs: Any) -> Any:
            if _is_hssd_state_duck(t):
                seen: set[int] = set()
                for sub_t in t._iter_tensors():
                    if isinstance(sub_t, torch.Tensor) and id(sub_t) not in seen:
                        seen.add(id(sub_t))
                        orig_fn(sub_t, *args, **kwargs)
                return None
            return orig_fn(t, *args, **kwargs)

        forbid = getattr(dynamo, "forbid_in_graph", None)
        wrapped = forbid(_hssd_mark_static_address) if callable(forbid) else _hssd_mark_static_address
        wrapped._qloop_hssd_aware = True  # type: ignore[attr-defined]
        wrapped._qloop_orig_mark_static_address = orig_fn  # type: ignore[attr-defined]
        dynamo.mark_static_address = wrapped  # type: ignore[attr-defined]

    _ensure_dynamo_mark_static_address_compat()

    def _resolve_state_dt(spec: Any, acc_dt: torch.dtype) -> torch.dtype:
        s = str(spec or "auto").strip().lower()
        if s in ("auto", ""):
            return acc_dt
        return _resolve_dt(s)

    def _prepare_qkv(
        q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, use_l2: bool
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        qf, kf, vf = _upcast(q), _upcast(k), _upcast(v)
        hv = vf.shape[2]
        if qf.shape[2] != hv:
            qf = qf.repeat_interleave(hv // qf.shape[2], dim=2)
        if kf.shape[2] != hv:
            kf = kf.repeat_interleave(hv // kf.shape[2], dim=2)
        if use_l2:
            qf = qf * torch.rsqrt((qf * qf).sum(-1, keepdim=True) + 1e-6)
            kf = kf * torch.rsqrt((kf * kf).sum(-1, keepdim=True) + 1e-6)
        return qf, kf, vf

    # ---------------------------------------------------------------- GDN-2
    class _StandaloneGDN2Adapter(nn.Module):
        def __init__(self, hs: int, hv: int, dk: int, dv: int, rank: int, cap_g: float, cap_d: float) -> None:
            super().__init__()
            r = max(1, min(int(rank), int(hs)))
            self.hv, self.dk, self.dv = int(hv), int(dk), int(dv)
            self.cap_g, self.cap_d = float(cap_g), float(cap_d)
            self.down = nn.Linear(hs, r, bias=False)
            self.erase_up = nn.Linear(r, hv * dk, bias=False)
            self.write_up = nn.Linear(r, hv * dv, bias=False)
            self.decay_up = nn.Linear(r, hv * dk, bias=False)

        def gates(self, x: torch.Tensor, g: torch.Tensor, beta: torch.Tensor):
            b, t, _ = x.shape
            xf = _upcast(x)
            h = F.silu(F.linear(xf, _upcast(self.down.weight)))
            de = (self.cap_g * torch.tanh(F.linear(h, _upcast(self.erase_up.weight)))).view(b, t, self.hv, self.dk)
            dw = (self.cap_g * torch.tanh(F.linear(h, _upcast(self.write_up.weight)))).view(b, t, self.hv, self.dv)
            dd = (self.cap_d * torch.tanh(F.linear(h, _upcast(self.decay_up.weight)))).view(b, t, self.hv, self.dk)
            bf = _upcast(beta).clamp(0.0, 1.0).unsqueeze(-1)
            return _upcast(g).clamp(max=0.0).unsqueeze(-1) * dd.exp(), _logistic_shift(bf, de), _logistic_shift(bf, dw)

    def _recurrent_gdn2(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        gv: torch.Tensor,
        bg: torch.Tensor,
        wg: torch.Tensor,
        init_st: torch.Tensor | None,
        out_final: bool,
        use_l2: bool,
        scale: float | None,
    ):
        dt = q.dtype
        qf, kf, vf = _prepare_qkv(q, k, v, use_l2)
        qf = qf * ((qf.shape[-1] ** -0.5) if scale is None else float(scale))
        b, t, hv, dk = kf.shape
        dv = vf.shape[-1]
        st = qf.new_zeros(b, hv, dk, dv) if init_st is None else _upcast(init_st).clone()
        outs = []
        for i in range(t):
            st = st * gv[:, i].exp().unsqueeze(-1)
            kt, vt = kf[:, i], vf[:, i]
            mem = (st * (bg[:, i] * kt).unsqueeze(-1)).sum(-2)
            st = st + kt.unsqueeze(-1) * ((wg[:, i] * vt) - mem).unsqueeze(-2)
            outs.append((st * qf[:, i].unsqueeze(-1)).sum(-2))
        return torch.stack(outs, dim=1).to(dt), (st if out_final else None)

    # ---------------------------------------------------------------- HSSD v3
    class _StandalonePeakBoltzmannGate(nn.Module):
        def __init__(self, num_heads: int = 1, kappa_init: float = 2.0, tau_init: float = 2.2) -> None:
            super().__init__()
            self.num_heads = int(num_heads)
            self.log_kappa = nn.Parameter(torch.full((self.num_heads,), math.log(float(kappa_init))))
            self.tau = nn.Parameter(torch.full((self.num_heads,), float(tau_init)))

        def forward(self, scores: torch.Tensor, valid_mask: torch.Tensor | None = None) -> torch.Tensor:
            sf = _upcast(scores)
            if valid_mask is None:
                n_val = float(sf.shape[-1])
                lse = torch.logsumexp(sf, dim=-1)
                max_sc = sf.max(dim=-1).values
                log_n = sf.new_tensor(math.log(max(n_val, 1.0)))
                gumbel = sf.new_tensor(math.sqrt(2.0 * math.log(max(n_val, 2.0))))
                has_valid: torch.Tensor | None = None
            else:
                vm = valid_mask.to(dtype=torch.bool).expand_as(sf)
                counts = vm.sum(dim=-1)
                has_valid = counts > 0
                safe_vm = torch.where(has_valid.unsqueeze(-1), vm, torch.ones_like(vm))
                masked = sf.masked_fill(~safe_vm, float("-inf"))
                lse = torch.logsumexp(masked, dim=-1)
                max_sc = masked.max(dim=-1).values
                n_eff = counts.clamp_min(1).to(sf.dtype)
                log_n = torch.log(n_eff)
                gumbel = torch.sqrt(2.0 * torch.log(n_eff.clamp_min(2.0)))
            stat = 0.5 * (lse - log_n) + 0.5 * (max_sc - gumbel)
            kappa = _upcast(self.log_kappa).exp().to(stat.dtype)
            tau = _upcast(self.tau).to(stat.dtype)
            if self.num_heads == 1 or stat.ndim == 0:
                kb, tb = kappa[0], tau[0]
            elif stat.ndim >= 2 and stat.shape[1] == self.num_heads:
                shp = [1, self.num_heads] + [1] * (stat.ndim - 2)
                kb, tb = kappa.view(*shp), tau.view(*shp)
            else:
                kb, tb = kappa, tau
            p_hit = torch.sigmoid(kb * (stat - tb))
            if has_valid is not None:
                p_hit = torch.where(has_valid, p_hit, torch.zeros_like(p_hit))
            return p_hit.to(scores.dtype)

    class _HSSDCacheState:
        def __init__(
            self,
            step_count: int | torch.Tensor,
            chunk_size: int,
            window_chunks: int,
            page_states: torch.Tensor,
            theta_cum: torch.Tensor,
            boundary_ring: tuple[torch.Tensor, ...],
            swa_k_ring: torch.Tensor,
            swa_v_ring: torch.Tensor,
            swa_valid_ring: torch.Tensor,
            *,
            s_chunk_start: Any = None,
        ) -> None:
            del s_chunk_start
            bsz = int(page_states.shape[0])
            dev = page_states.device
            if isinstance(step_count, torch.Tensor):
                if step_count.ndim == 0:
                    sc_t = torch.full((bsz,), int(step_count.item()), dtype=torch.long, device=dev)
                elif step_count.device == dev and step_count.dtype == torch.long and step_count.shape == (bsz,):
                    sc_t = step_count
                else:
                    sc_t = _compact(step_count.to(device=dev, dtype=torch.long).reshape(bsz))
            else:
                sc_t = torch.full((bsz,), int(step_count), dtype=torch.long, device=dev)
            self.step_count = sc_t
            self.chunk_size = int(chunk_size)
            self.window_chunks = int(window_chunks)
            self.page_states = page_states
            self.theta_cum = (
                theta_cum
                if (theta_cum.device == dev and theta_cum.dtype == torch.float64)
                else _compact(theta_cum.to(device=dev, dtype=torch.float64))
            )
            self.boundary_ring = tuple(boundary_ring)
            self.swa_k_ring = swa_k_ring
            self.swa_v_ring = swa_v_ring
            self.swa_valid_ring = (
                swa_valid_ring
                if (swa_valid_ring.device == dev and swa_valid_ring.dtype == torch.bool)
                else swa_valid_ring.to(device=dev, dtype=torch.bool)
            )

        def __repr__(self) -> str:
            return (
                f"_HSSDCacheState(shape={tuple(self.shape)}, dtype={self.dtype}, "
                f"theta_dtype={self.theta_cum.dtype}, step_count={self.step_count.tolist()}, "
                f"chunk_size={self.chunk_size}, window_chunks={self.window_chunks})"
            )

        @property
        def s_chunk_start(self) -> torch.Tensor:
            return self.boundary_ring[-1][:, 0]

        @property
        def s_chunk_start_bank(self) -> torch.Tensor:
            return self.boundary_ring[-1]

        @property
        def step_count_int(self) -> int:
            return int(self.step_count[0].item()) if self.step_count.numel() > 0 else 0

        @property
        def max_swa_len(self) -> int:
            return (self.window_chunks + 1) * self.chunk_size

        @property
        def shape(self) -> torch.Size:
            return self.page_states.shape

        @property
        def ndim(self) -> int:
            return self.page_states.ndim

        def dim(self) -> int:
            return self.page_states.dim()

        def __len__(self) -> int:
            return int(self.page_states.shape[0])

        @property
        def dtype(self) -> torch.dtype:
            return self.page_states.dtype

        @property
        def device(self) -> torch.device:
            return self.page_states.device

        def size(self, dim: int | None = None) -> torch.Size | int:
            return self.page_states.size() if dim is None else self.page_states.size(dim)

        def numel(self) -> int:
            return int(self.page_states.numel())

        def _iter_tensors(self) -> tuple[torch.Tensor, ...]:
            return (
                self.step_count,
                self.page_states,
                self.theta_cum,
                *self.boundary_ring,
                self.swa_k_ring,
                self.swa_v_ring,
                self.swa_valid_ring,
            )

        def unique_storage_bytes(self) -> int:
            seen: set[int] = set()
            tot = 0
            for t in self._iter_tensors():
                if t.numel() == 0:
                    continue
                st = t.untyped_storage()
                ptr = int(st.data_ptr())
                if ptr not in seen:
                    seen.add(ptr)
                    tot += int(st.nbytes())
            return tot

        def has_compact_storage(self) -> bool:
            for t in self._iter_tensors():
                if not t.is_contiguous():
                    return False
                if t.numel() == 0:
                    if int(t.untyped_storage().nbytes()) != 0 or int(t.storage_offset()) != 0:
                        return False
                    continue
                need = int(t.numel() * t.element_size())
                if int(t.untyped_storage().nbytes()) != need or int(t.storage_offset()) != 0:
                    return False
            return True

        def is_contiguous(self) -> bool:
            return all(t.is_contiguous() for t in self._iter_tensors())

        def contiguous(self) -> _HSSDCacheState:
            if self.has_compact_storage():
                return self
            memo: dict[tuple[int, int, tuple[int, ...], tuple[int, ...]], torch.Tensor] = {}

            def _cp(t: torch.Tensor) -> torch.Tensor:
                if t.numel() > 0:
                    key = (
                        int(t.untyped_storage().data_ptr()),
                        int(t.storage_offset()),
                        tuple(t.shape),
                        tuple(t.stride()),
                    )
                    if key in memo:
                        return memo[key]
                    out_t = _compact(t)
                    memo[key] = out_t
                    return out_t
                return _compact(t)

            return _HSSDCacheState(
                step_count=_compact(self.step_count),
                chunk_size=self.chunk_size,
                window_chunks=self.window_chunks,
                page_states=_cp(self.page_states),
                theta_cum=_compact(self.theta_cum),
                boundary_ring=tuple(_cp(t) for t in self.boundary_ring),
                swa_k_ring=_compact(self.swa_k_ring),
                swa_v_ring=_compact(self.swa_v_ring),
                swa_valid_ring=_compact(self.swa_valid_ring),
            )

        def compact(self) -> _HSSDCacheState:
            return self.contiguous()

        def zero_(self) -> _HSSDCacheState:
            seen_ptrs: set[int] = (
                {int(self.page_states.untyped_storage().data_ptr())} if self.page_states.numel() > 0 else set()
            )
            fixed_ring: list[torch.Tensor] = []
            unaliased = False
            for ring_t in self.boundary_ring:
                cur_t = ring_t
                if cur_t.numel() > 0:
                    ptr = int(cur_t.untyped_storage().data_ptr())
                    if ptr in seen_ptrs:
                        cur_t = torch.zeros_like(cur_t).contiguous()
                        ptr = int(cur_t.untyped_storage().data_ptr())
                        unaliased = True
                    else:
                        cur_t.zero_()
                    seen_ptrs.add(ptr)
                else:
                    cur_t.zero_()
                fixed_ring.append(cur_t)
            if unaliased:
                self.boundary_ring = tuple(fixed_ring)
            self.step_count.zero_()
            self.page_states.zero_()
            self.theta_cum.zero_()
            self.swa_k_ring.zero_()
            self.swa_v_ring.zero_()
            self.swa_valid_ring.zero_()
            return self

        @classmethod
        def zeros(
            cls,
            bsz: int,
            num_pages: int,
            hv: int,
            dk: int,
            dv: int,
            *,
            chunk_size: int = 64,
            window_chunks: int = 8,
            device: torch.device | str | None = None,
            dtype: torch.dtype = torch.float32,
            distinct_ring_buffers: bool = False,
        ) -> _HSSDCacheState:
            ps = torch.zeros(bsz, num_pages, hv, dk, dv, device=device, dtype=dtype)
            th = torch.zeros(bsz, hv, dk // 2, device=device, dtype=torch.float64)
            if distinct_ring_buffers:
                br = tuple(torch.zeros_like(ps) for _ in range(window_chunks + 1))
            else:
                br = tuple(ps for _ in range(window_chunks + 1))
            mw = (window_chunks + 1) * chunk_size
            kr = torch.zeros(bsz, mw, hv, dk, device=device, dtype=dtype)
            vr = torch.zeros(bsz, mw, hv, dv, device=device, dtype=dtype)
            vlr = torch.zeros(bsz, mw, device=device, dtype=torch.bool)
            sc = torch.zeros(bsz, device=device, dtype=torch.long)
            return cls(sc, int(chunk_size), int(window_chunks), ps, th, br, kr, vr, vlr)

        @classmethod
        def empty_like_state(
            cls,
            other: Any,
            *,
            device: torch.device | str | None = None,
            dtype: torch.dtype | None = None,
        ) -> _HSSDCacheState:
            dev = other.device if device is None else torch.device(device)
            dt = other.dtype if dtype is None else dtype
            bsz, mp, hv, dk, dv = other.page_states.shape
            ps = torch.empty(bsz, mp, hv, dk, dv, device=dev, dtype=dt)
            th = torch.empty(bsz, hv, dk // 2, device=dev, dtype=torch.float64)
            br = tuple(torch.empty_like(ps) for _ in range(int(other.window_chunks) + 1))
            mw = (int(other.window_chunks) + 1) * int(other.chunk_size)
            kr = torch.empty(bsz, mw, hv, dk, device=dev, dtype=dt)
            vr = torch.empty(bsz, mw, hv, dv, device=dev, dtype=dt)
            vlr = torch.empty(bsz, mw, device=dev, dtype=torch.bool)
            sc = torch.empty(bsz, device=dev, dtype=torch.long)
            return cls(sc, int(other.chunk_size), int(other.window_chunks), ps, th, br, kr, vr, vlr)

        @classmethod
        def from_tensor(
            cls,
            state_tensor: torch.Tensor,
            *,
            num_pages: int = 8,
            chunk_size: int = 64,
            window_chunks: int = 8,
        ) -> _HSSDCacheState:
            st = _upcast(state_tensor)
            if st.ndim == 4:
                bsz, hv, dk, dv = st.shape
                ps = st.unsqueeze(1).expand(bsz, num_pages, hv, dk, dv).clone().contiguous()
            elif st.ndim == 5:
                bsz, mp, hv, dk, dv = st.shape
                if mp != num_pages:
                    raise ValueError(f"expected {num_pages} state pages, got {mp}")
                ps = st.clone().contiguous()
            else:
                raise ValueError(f"initial_state tensor must be 4D or 5D, got shape {tuple(st.shape)}")
            base = cls.zeros(
                bsz,
                num_pages,
                hv,
                dk,
                dv,
                chunk_size=chunk_size,
                window_chunks=window_chunks,
                device=ps.device,
                dtype=ps.dtype,
            )
            base.page_states = ps
            base.boundary_ring = tuple(ps for _ in range(window_chunks + 1))
            return base

        def to(self, *args: Any, **kwargs: Any) -> _HSSDCacheState:
            memo: dict[int, torch.Tensor] = {}
            non_blocking = bool(kwargs.get("non_blocking", False))
            ps = _compact(self.page_states.to(*args, **kwargs))
            if self.page_states.numel() > 0:
                memo[int(self.page_states.untyped_storage().data_ptr())] = ps
            dev, dt = ps.device, ps.dtype

            def _cv(t: torch.Tensor) -> torch.Tensor:
                if t.numel() > 0:
                    ptr = int(t.untyped_storage().data_ptr())
                    if ptr in memo:
                        return memo[ptr]
                    out_t = _compact(t.to(device=dev, dtype=dt, non_blocking=non_blocking))
                    memo[ptr] = out_t
                    return out_t
                return _compact(t.to(device=dev, dtype=dt, non_blocking=non_blocking))

            return _HSSDCacheState(
                step_count=_compact(self.step_count.to(device=dev, dtype=torch.long, non_blocking=non_blocking)),
                chunk_size=self.chunk_size,
                window_chunks=self.window_chunks,
                page_states=ps,
                theta_cum=_compact(self.theta_cum.to(device=dev, dtype=torch.float64, non_blocking=non_blocking)),
                boundary_ring=tuple(_cv(t) for t in self.boundary_ring),
                swa_k_ring=_compact(self.swa_k_ring.to(device=dev, dtype=dt, non_blocking=non_blocking)),
                swa_v_ring=_compact(self.swa_v_ring.to(device=dev, dtype=dt, non_blocking=non_blocking)),
                swa_valid_ring=_compact(
                    self.swa_valid_ring.to(device=dev, dtype=torch.bool, non_blocking=non_blocking)
                ),
            )

        def clone(self) -> _HSSDCacheState:
            memo: dict[int, torch.Tensor] = {}

            def _cl(t: torch.Tensor) -> torch.Tensor:
                ptr = int(t.untyped_storage().data_ptr()) if t.numel() > 0 else id(t)
                if ptr not in memo:
                    memo[ptr] = t.clone().contiguous()
                return memo[ptr]

            return _HSSDCacheState(
                step_count=self.step_count.clone().contiguous(),
                chunk_size=self.chunk_size,
                window_chunks=self.window_chunks,
                page_states=_cl(self.page_states),
                theta_cum=self.theta_cum.clone().contiguous(),
                boundary_ring=tuple(_cl(t) for t in self.boundary_ring),
                swa_k_ring=self.swa_k_ring.clone().contiguous(),
                swa_v_ring=self.swa_v_ring.clone().contiguous(),
                swa_valid_ring=self.swa_valid_ring.clone().contiguous(),
            )

        def detach(self) -> _HSSDCacheState:
            memo: dict[int, torch.Tensor] = {}

            def _dt(t: torch.Tensor) -> torch.Tensor:
                ptr = int(t.untyped_storage().data_ptr()) if t.numel() > 0 else id(t)
                if ptr not in memo:
                    memo[ptr] = _compact(t.detach())
                return memo[ptr]

            return _HSSDCacheState(
                step_count=_compact(self.step_count.detach()),
                chunk_size=self.chunk_size,
                window_chunks=self.window_chunks,
                page_states=_dt(self.page_states),
                theta_cum=_compact(self.theta_cum.detach()),
                boundary_ring=tuple(_dt(t) for t in self.boundary_ring),
                swa_k_ring=_compact(self.swa_k_ring.detach()),
                swa_v_ring=_compact(self.swa_v_ring.detach()),
                swa_valid_ring=_compact(self.swa_valid_ring.detach()),
            )

        def copy_(self, other: Any) -> _HSSDCacheState:
            if not (hasattr(other, "page_states") and hasattr(other, "boundary_ring")):
                raise TypeError(f"cannot copy_ from {type(other).__name__} into _HSSDCacheState")
            if len(self.boundary_ring) != len(other.boundary_ring):
                raise ValueError("boundary_ring length mismatch in _HSSDCacheState.copy_")
            seen_ptrs = (
                {int(self.page_states.untyped_storage().data_ptr())} if self.page_states.numel() > 0 else set()
            )
            unaliased_ring: list[torch.Tensor] = []
            unaliased = False
            for t in self.boundary_ring:
                cur_t = t
                if cur_t.numel() > 0:
                    ptr = int(cur_t.untyped_storage().data_ptr())
                    if ptr in seen_ptrs:
                        cur_t = cur_t.clone().contiguous()
                        ptr = int(cur_t.untyped_storage().data_ptr())
                        unaliased = True
                    seen_ptrs.add(ptr)
                unaliased_ring.append(cur_t)
            if unaliased:
                self.boundary_ring = tuple(unaliased_ring)
            self.chunk_size = int(other.chunk_size)
            self.window_chunks = int(other.window_chunks)
            if isinstance(other.step_count, torch.Tensor):
                osc = other.step_count.to(device=self.device, dtype=torch.long).reshape_as(self.step_count)
                self.step_count.copy_(osc)
            else:
                self.step_count.fill_(int(other.step_count))
            self.page_states.copy_(other.page_states)
            self.theta_cum.copy_(other.theta_cum)
            for dst, src in zip(self.boundary_ring, other.boundary_ring):
                dst.copy_(src)
            self.swa_k_ring.copy_(other.swa_k_ring)
            self.swa_v_ring.copy_(other.swa_v_ring)
            self.swa_valid_ring.copy_(other.swa_valid_ring)
            return self

        def index_select(self, dim: int, index: torch.Tensor) -> _HSSDCacheState:
            d = dim if dim >= 0 else dim + self.ndim
            if d != 0:
                raise ValueError(f"_HSSDCacheState.index_select only supports batch dim 0, got dim={dim}")
            idx = index.to(device=self.device, dtype=torch.long)
            memo: dict[int, torch.Tensor] = {}

            def _sel(t: torch.Tensor) -> torch.Tensor:
                if t.numel() > 0:
                    ptr = int(t.untyped_storage().data_ptr())
                    if ptr in memo:
                        return memo[ptr]
                    out_t = _compact(torch.index_select(t, 0, idx))
                    memo[ptr] = out_t
                    return out_t
                return _compact(torch.index_select(t, 0, idx))

            return _HSSDCacheState(
                step_count=_compact(torch.index_select(self.step_count, 0, idx)),
                chunk_size=self.chunk_size,
                window_chunks=self.window_chunks,
                page_states=_sel(self.page_states),
                theta_cum=_compact(torch.index_select(self.theta_cum, 0, idx)),
                boundary_ring=tuple(_sel(t) for t in self.boundary_ring),
                swa_k_ring=_compact(torch.index_select(self.swa_k_ring, 0, idx)),
                swa_v_ring=_compact(torch.index_select(self.swa_v_ring, 0, idx)),
                swa_valid_ring=_compact(torch.index_select(self.swa_valid_ring, 0, idx)),
            )

        def __getitem__(self, idx: Any) -> _HSSDCacheState:
            if isinstance(idx, tuple):
                rest = idx[1:]
                if any(x is not Ellipsis and x != slice(None) for x in rest):
                    raise IndexError("_HSSDCacheState indexing only supports batch dimension 0")
                idx = idx[0]
            if isinstance(idx, int):
                bsz = len(self)
                norm_i = idx if idx >= 0 else bsz + idx
                if not 0 <= norm_i < bsz:
                    raise IndexError(f"batch index {idx} out of range for batch size {bsz}")
                return self.index_select(0, torch.tensor([norm_i], device=self.device, dtype=torch.long))
            if isinstance(idx, slice):
                indices = torch.arange(len(self), device=self.device, dtype=torch.long)[idx]
                return self.index_select(0, indices)
            if isinstance(idx, (list, tuple)):
                return self.index_select(0, torch.tensor(idx, device=self.device, dtype=torch.long))
            if isinstance(idx, torch.Tensor):
                if idx.dtype == torch.bool:
                    if idx.ndim != 1 or idx.shape[0] != len(self):
                        raise IndexError("boolean batch mask must have shape [B]")
                    indices = idx.to(device=self.device).nonzero(as_tuple=False).squeeze(-1)
                    return self.index_select(0, indices)
                return self.index_select(0, idx)
            raise TypeError(f"Unsupported _HSSDCacheState index type: {type(idx).__name__}")

        def reorder_cache(self, beam_idx: torch.Tensor) -> _HSSDCacheState:
            return self.index_select(0, beam_idx)

        def repeat_interleave(
            self,
            repeats: int | torch.Tensor,
            dim: int | None = 0,
            *,
            output_size: int | None = None,
        ) -> _HSSDCacheState:
            if dim is None:
                raise ValueError("_HSSDCacheState.repeat_interleave requires explicit batch dim=0, got dim=None")
            d = dim if dim >= 0 else dim + self.ndim
            if d != 0:
                raise ValueError(f"_HSSDCacheState.repeat_interleave only supports batch dim=0, got dim={dim}")
            rep = repeats.to(device=self.device) if isinstance(repeats, torch.Tensor) else repeats
            memo: dict[int, torch.Tensor] = {}

            def _ri(t: torch.Tensor) -> torch.Tensor:
                if t.numel() > 0:
                    ptr = int(t.untyped_storage().data_ptr())
                    if ptr in memo:
                        return memo[ptr]
                    out_t = _compact(t.repeat_interleave(rep, dim=0, output_size=output_size))
                    memo[ptr] = out_t
                    return out_t
                return _compact(t.repeat_interleave(rep, dim=0, output_size=output_size))

            return _HSSDCacheState(
                step_count=_compact(self.step_count.repeat_interleave(rep, dim=0, output_size=output_size)),
                chunk_size=self.chunk_size,
                window_chunks=self.window_chunks,
                page_states=_ri(self.page_states),
                theta_cum=_compact(self.theta_cum.repeat_interleave(rep, dim=0, output_size=output_size)),
                boundary_ring=tuple(_ri(t) for t in self.boundary_ring),
                swa_k_ring=_compact(self.swa_k_ring.repeat_interleave(rep, dim=0, output_size=output_size)),
                swa_v_ring=_compact(self.swa_v_ring.repeat_interleave(rep, dim=0, output_size=output_size)),
                swa_valid_ring=_compact(self.swa_valid_ring.repeat_interleave(rep, dim=0, output_size=output_size)),
            )

        def state_equal(self, other: Any, *, atol: float = 1e-5, rtol: float = 1e-4) -> bool:
            if not (hasattr(other, "page_states") and hasattr(other, "boundary_ring")):
                return False
            if self.chunk_size != int(other.chunk_size) or self.window_chunks != int(other.window_chunks):
                return False
            if self.page_states.shape != other.page_states.shape:
                return False
            if len(self.boundary_ring) != len(other.boundary_ring):
                return False
            bsz = int(self.page_states.shape[0])
            osc_raw = other.step_count
            if isinstance(osc_raw, torch.Tensor):
                osc = osc_raw.to(device=self.device, dtype=torch.long).reshape(-1)
            else:
                osc = torch.full((bsz,), int(osc_raw), dtype=torch.long, device=self.device)
            if osc.shape != self.step_count.shape or not bool(torch.equal(self.step_count, osc)):
                return False
            cmp_dt = (
                torch.float64
                if torch.float64 in (self.page_states.dtype, other.page_states.dtype)
                else torch.float32
            )
            if not bool(
                torch.allclose(
                    self.page_states.to(cmp_dt),
                    other.page_states.to(device=self.device, dtype=cmp_dt),
                    atol=atol,
                    rtol=rtol,
                )
            ):
                return False
            oth_th = other.theta_cum.to(device=self.device, dtype=torch.float64)
            if self.theta_cum.shape != oth_th.shape or not bool(
                torch.allclose(self.theta_cum, oth_th, atol=atol, rtol=rtol)
            ):
                return False
            for a, b in zip(self.boundary_ring, other.boundary_ring):
                if a.shape != b.shape or not bool(
                    torch.allclose(a.to(cmp_dt), b.to(device=self.device, dtype=cmp_dt), atol=atol, rtol=rtol)
                ):
                    return False
            oth_val = other.swa_valid_ring.to(device=self.device, dtype=torch.bool)
            if self.swa_valid_ring.shape != oth_val.shape or not bool(torch.equal(self.swa_valid_ring, oth_val)):
                return False
            if self.swa_k_ring.shape != other.swa_k_ring.shape or self.swa_v_ring.shape != other.swa_v_ring.shape:
                return False
            vm = self.swa_valid_ring
            if bool(vm.any()):
                okr = other.swa_k_ring.to(device=self.device, dtype=cmp_dt)
                ovr = other.swa_v_ring.to(device=self.device, dtype=cmp_dt)
                if not bool(torch.allclose(self.swa_k_ring.to(cmp_dt)[vm], okr[vm], atol=atol, rtol=rtol)):
                    return False
                if not bool(torch.allclose(self.swa_v_ring.to(cmp_dt)[vm], ovr[vm], atol=atol, rtol=rtol)):
                    return False
            return True

        @classmethod
        def __torch_function__(
            cls,
            func: Any,
            types: tuple[type, ...],
            args: tuple[Any, ...] = (),
            kwargs: dict[str, Any] | None = None,
        ) -> Any:
            del types
            kw = {} if kwargs is None else dict(kwargs)
            if func is torch.zeros_like:
                src: _HSSDCacheState = args[0]
                dt = src.dtype if kw.get("dtype") is None else kw["dtype"]
                dev = src.device if kw.get("device") is None else torch.device(kw["device"])
                bsz, mp, hv, dk, dv = src.page_states.shape
                return cls.zeros(
                    bsz,
                    mp,
                    hv,
                    dk,
                    dv,
                    chunk_size=src.chunk_size,
                    window_chunks=src.window_chunks,
                    device=dev,
                    dtype=dt,
                    distinct_ring_buffers=True,
                )
            if func is torch.empty_like:
                src = args[0]
                return cls.empty_like_state(src, device=kw.get("device"), dtype=kw.get("dtype"))
            if func is torch.clone:
                return args[0].clone()
            if func is torch.detach:
                return args[0].detach()
            if func is torch.index_select:
                src = args[0]
                dim = int(args[1]) if len(args) >= 2 else int(kw["dim"])
                index = args[2] if len(args) >= 3 else kw["index"]
                return src.index_select(dim, index)
            if func is torch.repeat_interleave:
                src = args[0]
                repeats = kw["repeats"] if "repeats" in kw else args[1]
                dim = kw.get("dim", args[2] if len(args) >= 3 else None)
                return src.repeat_interleave(repeats, dim=dim, output_size=kw.get("output_size"))
            if func is torch.cat:
                states = list(args[0])
                dim = int(args[1]) if len(args) >= 2 else int(kw.get("dim", 0))
                if not states:
                    raise ValueError("torch.cat expected a non-empty sequence of _HSSDCacheState")
                c0 = states[0]
                if dim not in (0, -c0.ndim):
                    raise ValueError(f"_HSSDCacheState only supports torch.cat along batch dim 0, got dim={dim}")
                for s in states[1:]:
                    if s.chunk_size != c0.chunk_size or s.window_chunks != c0.window_chunks:
                        raise ValueError("cannot cat _HSSDCacheState instances with mismatched chunk/window config")
                dev, dt = c0.device, c0.dtype
                return cls(
                    step_count=torch.cat([s.step_count.to(device=dev, dtype=torch.long) for s in states], dim=0),
                    chunk_size=c0.chunk_size,
                    window_chunks=c0.window_chunks,
                    page_states=_compact(torch.cat([s.page_states.to(device=dev, dtype=dt) for s in states], dim=0)),
                    theta_cum=_compact(
                        torch.cat([s.theta_cum.to(device=dev, dtype=torch.float64) for s in states], dim=0)
                    ),
                    boundary_ring=tuple(
                        _compact(torch.cat([s.boundary_ring[j].to(device=dev, dtype=dt) for s in states], dim=0))
                        for j in range(len(c0.boundary_ring))
                    ),
                    swa_k_ring=_compact(torch.cat([s.swa_k_ring.to(device=dev, dtype=dt) for s in states], dim=0)),
                    swa_v_ring=_compact(torch.cat([s.swa_v_ring.to(device=dev, dtype=dt) for s in states], dim=0)),
                    swa_valid_ring=_compact(
                        torch.cat([s.swa_valid_ring.to(device=dev, dtype=torch.bool) for s in states], dim=0)
                    ),
                )
            return NotImplemented

    HSSDCacheState = _HSSDCacheState

    class _StandaloneHSSDAdapter(nn.Module):
        def __init__(self, hs: int, hv: int, dk: int, dv: int, hcfg: dict[str, Any], layer_id: int) -> None:
            super().__init__()
            self.layer_id = int(layer_id)
            self.hidden_size = int(hs)
            self.num_v_heads = int(hv)
            self.head_k_dim = int(dk)
            self.head_v_dim = int(dv)
            self.half_k_dim = self.head_k_dim // 2
            self.num_pages = int(hcfg.get("num_pages", 8))
            self.k_pages = int(hcfg.get("top_k_pages") or hcfg.get("k_pages", 2))
            self.chunk_size = int(hcfg.get("chunk_size", 64))
            self.window_chunks = int(hcfg.get("window_chunks", 8))
            self.segment_chunks = int(hcfg.get("segment_chunks", 4))
            self.checkpoint_segments = bool(hcfg.get("checkpoint_segments", True))
            self.cap_g = float(hcfg.get("gate_logit_cap", 8.0))
            self.cap_d = float(hcfg.get("decay_log_cap", 2.0))
            self.rank = int(hcfg.get("rank", 64))
            eff_rank = max(1, min(self.rank, self.hidden_size))
            self.theta_low_rank = bool(hcfg.get("theta_low_rank", True))
            self.state_dtype = str(hcfg.get("state_dtype", "auto"))
            self.paired_b_gate = bool(hcfg.get("paired_b_gate", False))
            self.use_swa = bool(hcfg.get("use_swa", True))
            self.use_phase = bool(hcfg.get("use_phase", True))
            self.use_surprise = bool(hcfg.get("use_surprise", True))
            self.surprise_mode = str(hcfg.get("surprise_mode", "routed"))
            if self.surprise_mode not in ("routed", "page0"):
                raise ValueError(f"Unsupported HSSD surprise_mode={self.surprise_mode!r}")

            self.down = nn.Linear(self.hidden_size, eff_rank, bias=False)
            self.erase_up = nn.Linear(eff_rank, self.num_v_heads * self.head_k_dim, bias=False)
            self.write_up = nn.Linear(eff_rank, self.num_v_heads * self.head_v_dim, bias=False)
            self.decay_up = nn.Linear(eff_rank, self.num_v_heads * self.half_k_dim, bias=False)
            if self.theta_low_rank:
                self.theta_down = nn.Linear(self.hidden_size, eff_rank, bias=False)
                self.theta_up = nn.Linear(eff_rank, self.num_v_heads * self.half_k_dim, bias=False)
            else:
                self.theta_proj = nn.Linear(self.hidden_size, self.num_v_heads * self.half_k_dim, bias=False)
            self.surprise_proj = nn.Linear(self.hidden_size, self.num_v_heads, bias=True)
            slp_init = float(hcfg.get("surprise_slope_init", 3.0))
            thr_init = float(hcfg.get("surprise_thresh_init", 0.85))
            self.log_surprise_slope = nn.Parameter(torch.full((self.num_v_heads,), math.log(slp_init)))
            self.surprise_thresh = nn.Parameter(torch.full((self.num_v_heads,), thr_init))
            self.write_router = nn.Linear(self.hidden_size, self.num_pages, bias=True)
            self.read_router = nn.Linear(self.num_v_heads * self.head_k_dim, self.num_pages, bias=True)
            self.log_page_gamma = nn.Parameter(torch.zeros(self.num_pages))
            self.swa_q_up = nn.Linear(eff_rank, self.num_v_heads * self.head_k_dim, bias=False)
            self.swa_k_up = nn.Linear(eff_rank, self.num_v_heads * self.head_k_dim, bias=False)
            self.swa_v_up = nn.Linear(eff_rank, self.num_v_heads * self.head_v_dim, bias=False)
            self.swa_q_norm = nn.Parameter(torch.ones(self.head_k_dim))
            self.swa_k_norm = nn.Parameter(torch.ones(self.head_k_dim))
            self.boltzmann_gate = _StandalonePeakBoltzmannGate(
                self.num_v_heads,
                kappa_init=float(hcfg.get("kappa_init", 2.0)),
                tau_init=float(hcfg.get("tau_init", 2.2)),
            )
            gmin = float(hcfg.get("decay_gamma_min", 1e-4))
            gmax = float(hcfg.get("decay_gamma_max", 5e-2))
            if self.num_pages == 1:
                gprior = torch.tensor([gmin], dtype=torch.float32)
            else:
                gprior = torch.exp(torch.linspace(math.log(gmin), math.log(gmax), self.num_pages))
            self.register_buffer("page_gamma_prior", gprior, persistent=False)

        def project_inputs(
            self,
            x_in: torch.Tensor,
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            g_base: torch.Tensor,
            beta_base: torch.Tensor,
            *,
            use_qk_l2norm: bool = False,
        ) -> dict[str, torch.Tensor]:
            bsz, seq_len, _ = x_in.shape
            hv, dk, dv, hdk = self.num_v_heads, self.head_k_dim, self.head_v_dim, self.half_k_dim
            qp, kp, vp = _prepare_qkv(query, key, value, use_qk_l2norm)
            xf = _upcast(x_in)
            h = F.silu(F.linear(xf, _upcast(self.down.weight)))
            de = (self.cap_g * torch.tanh(F.linear(h, _upcast(self.erase_up.weight)))).view(bsz, seq_len, hv, dk)
            dw = (self.cap_g * torch.tanh(F.linear(h, _upcast(self.write_up.weight)))).view(bsz, seq_len, hv, dv)
            dd = (self.cap_d * torch.tanh(F.linear(h, _upcast(self.decay_up.weight)))).view(bsz, seq_len, hv, hdk)
            bf = _upcast(beta_base).clamp(0.0, 1.0).unsqueeze(-1)
            bg = 2.0 * _logistic_shift(0.5 * bf, de)
            if self.paired_b_gate:
                bg = (
                    bg.view(bsz, seq_len, hv, hdk, 2)
                    .mean(dim=-1, keepdim=True)
                    .expand(bsz, seq_len, hv, hdk, 2)
                    .reshape(bsz, seq_len, hv, dk)
                )
            wg = _logistic_shift(bf, dw)
            brate = 1.0 + 0.1 * (-_upcast(g_base).clamp(max=0.0)).tanh().unsqueeze(-1)
            gamma_chan_pair = brate * dd.exp()
            pscale = _upcast(self.page_gamma_prior) * _upcast(self.log_page_gamma).clamp(-3.0, 3.0).exp()
            if not self.use_phase:
                theta = xf.new_zeros(bsz, seq_len, hv, hdk)
            elif self.theta_low_rank:
                th_raw = F.linear(F.linear(xf, _upcast(self.theta_down.weight)), _upcast(self.theta_up.weight))
                theta = math.pi * torch.tanh(th_raw).view(bsz, seq_len, hv, hdk)
            else:
                theta = math.pi * torch.tanh(F.linear(xf, _upcast(self.theta_proj.weight))).view(bsz, seq_len, hv, hdk)
            sb = F.linear(xf, _upcast(self.surprise_proj.weight), _upcast(self.surprise_proj.bias))
            wl = F.linear(xf, _upcast(self.write_router.weight), _upcast(self.write_router.bias))
            rl = F.linear(
                qp.reshape(bsz, seq_len, hv * dk), _upcast(self.read_router.weight), _upcast(self.read_router.bias)
            )
            rdk = math.sqrt(dk)
            dqs = F.linear(h, _upcast(self.swa_q_up.weight)).view(bsz, seq_len, hv, dk)
            dks = F.linear(h, _upcast(self.swa_k_up.weight)).view(bsz, seq_len, hv, dk)
            dvs = F.linear(h, _upcast(self.swa_v_up.weight)).view(bsz, seq_len, hv, dv)
            qs = _rms_norm_scaled(rdk * qp + dqs, self.swa_q_norm)
            ks = _rms_norm_scaled(rdk * kp + dks, self.swa_k_norm)
            vs = vp + dvs
            return {
                "q": qp,
                "k": kp,
                "v": vp,
                "gamma_page_scale": pscale,
                "gamma_chan_pair": gamma_chan_pair,
                "b_gate": bg,
                "w_gate": wg,
                "theta": theta,
                "sb": sb,
                "wl": wl,
                "rl": rl,
                "qs": qs,
                "ks": ks,
                "vs": vs,
            }

    def _is_compiling() -> bool:
        compiler = getattr(torch, "compiler", None)
        fn = getattr(compiler, "is_compiling", None) if compiler is not None else None
        return bool(fn()) if callable(fn) else False

    def _disjoint_horizon_bounds(t_1based: int, chunk_size: int = 64, window_chunks: int = 8) -> tuple[int, int, int]:
        n = (int(t_1based) - 1) // int(chunk_size)
        k_far = max(0, n - int(window_chunks))
        return n, k_far, k_far * int(chunk_size)

    def _chunk_slices_for_span(step_offset: int, seq_len: int, chunk_size: int) -> list[tuple[int, int]]:
        slices: list[tuple[int, int]] = []
        cur = 0
        while cur < seq_len:
            g_cur = step_offset + cur
            rem = chunk_size - (g_cur % chunk_size)
            nxt = min(seq_len, cur + rem)
            slices.append((cur, nxt))
            cur = nxt
        return slices

    _WY_BLOCK_META_CACHE: dict[
        tuple[torch.device, int], tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    ] = {}

    def _get_wy_block_meta(
        n_blk: int, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if _is_compiling():
            idx8 = torch.arange(8, device=device)
            strict_lo_8 = idx8[:, None] > idx8[None, :]
            diag_lin = torch.arange(n_blk, device=device, dtype=torch.long) * (n_blk + 1)
            if n_blk >= 2:
                tril = torch.tril_indices(n_blk, n_blk, offset=-1, device=device)
                bi_idx = tril[0]
                bj_idx = tril[1]
                all_lin_idx = torch.cat([diag_lin, bi_idx * n_blk + bj_idx], dim=0)
            else:
                empty_l = torch.empty((0,), dtype=torch.long, device=device)
                bi_idx, bj_idx, all_lin_idx = empty_l, empty_l, diag_lin
            return strict_lo_8, bi_idx, bj_idx, all_lin_idx

        key = (device, n_blk)
        cached = _WY_BLOCK_META_CACHE.get(key)
        if cached is not None:
            return cached
        with torch.inference_mode(False), torch.no_grad():
            idx8 = torch.arange(8, device=device)
            strict_lo_8 = idx8[:, None] > idx8[None, :]
            diag_lin = torch.arange(n_blk, device=device, dtype=torch.long) * (n_blk + 1)
            if n_blk >= 2:
                tril = torch.tril_indices(n_blk, n_blk, offset=-1, device=device)
                bi_idx = tril[0]
                bj_idx = tril[1]
                all_lin_idx = torch.cat([diag_lin, bi_idx * n_blk + bj_idx], dim=0)
            else:
                empty_l = torch.empty((0,), dtype=torch.long, device=device)
                bi_idx, bj_idx, all_lin_idx = empty_l, empty_l, diag_lin
        res = (strict_lo_8, bi_idx, bj_idx, all_lin_idx)
        _WY_BLOCK_META_CACHE[key] = res
        return res

    def _wy_kda_state_update(
        page_states: torch.Tensor,
        k_eff: torch.Tensor,
        v_eff: torch.Tensor,
        log_a: torch.Tensor,
    ) -> torch.Tensor:
        log_a_neg = log_a.clamp(max=0.0)
        c_len = k_eff.shape[-2]
        if c_len == 1:
            dec = log_a_neg.squeeze(-2).exp().unsqueeze(-1) * page_states
            ke = k_eff.squeeze(-2)
            ve = v_eff.squeeze(-2)
            mem = (ke.unsqueeze(-2) @ dec).squeeze(-2)
            return dec + ke.unsqueeze(-1) * (ve - mem).unsqueeze(-2)

        b_sz = 8
        n_blk = (c_len + b_sz - 1) // b_sz
        c_pad = n_blk * b_sz
        pad_t = c_pad - c_len
        dk = k_eff.shape[-1]
        lead_shape = k_eff.shape[:-2]

        if pad_t > 0:
            k_pad = F.pad(k_eff, (0, 0, 0, pad_t), value=0.0)
            log_a_pad = F.pad(log_a_neg, (0, 0, 0, pad_t), value=0.0)
        else:
            k_pad = k_eff
            log_a_pad = log_a_neg

        cum_log_pad = torch.cumsum(log_a_pad, dim=-2)
        cum_log_a = cum_log_pad[..., :c_len, :] if pad_t > 0 else cum_log_pad

        k_blk = k_pad.reshape((*lead_shape, n_blk, b_sz, dk))
        g_blk = cum_log_pad.reshape((*lead_shape, n_blk, b_sz, dk))
        strict_lo_8, bi_idx, bj_idx, all_lin_idx = _get_wy_block_meta(n_blk, k_eff.device)

        diff_d = g_blk[..., :, None, :] - g_blk[..., None, :, :]
        safe_d = torch.where(strict_lo_8.unsqueeze(-1), diff_d, torch.zeros_like(diff_d))
        diag_blks = (k_blk[..., :, None, :] * k_blk[..., None, :, :] * safe_d.exp()).sum(dim=-1)
        diag_blks = torch.where(strict_lo_8, diag_blks, torch.zeros_like(diag_blks))

        if n_blk == 1:
            l_mat = diag_blks.squeeze(-3)[..., :c_len, :c_len]
        else:
            rhs_all = k_blk[..., :-1, :, :] * (g_blk[..., :-1, 7:8, :] - g_blk[..., :-1, :, :]).exp()
            rhs_k = rhs_all.index_select(-3, bj_idx)
            k_i_pairs = k_blk.index_select(-3, bi_idx)
            g_i_pairs = g_blk.index_select(-3, bi_idx)
            g_piv_pairs = g_blk[..., 7:8, :].index_select(-3, bj_idx)
            lhs = k_i_pairs * (g_i_pairs - g_piv_pairs).exp()
            off_blks = lhs @ rhs_k.transpose(-1, -2)

            active_blks = torch.cat([diag_blks, off_blks], dim=-3)
            n_act = active_blks.shape[-3]
            idx_exp = all_lin_idx.view((*(1,) * len(lead_shape), n_act, 1, 1)).expand(
                (*lead_shape, n_act, b_sz, b_sz)
            )
            grid = k_eff.new_zeros((*lead_shape, n_blk * n_blk, b_sz, b_sz)).scatter(-3, idx_exp, active_blks)
            l_pad = grid.view((*lead_shape, n_blk, n_blk, b_sz, b_sz)).transpose(-3, -2).reshape(
                (*lead_shape, c_pad, c_pad)
            )
            l_mat = l_pad[..., :c_len, :c_len] if pad_t > 0 else l_pad

        gamma_from_0 = cum_log_a.exp()
        rhs = v_eff - (gamma_from_0 * k_eff) @ page_states
        eye = torch.eye(c_len, dtype=k_eff.dtype, device=k_eff.device)
        u_wy = torch.linalg.solve_triangular(eye + l_mat, rhs, upper=False, unitriangular=True)

        gamma_to_end = (cum_log_a[..., -1:, :] - cum_log_a).exp()
        d_chunk = cum_log_a[..., -1, :].exp().unsqueeze(-1)
        return d_chunk * page_states + (gamma_to_end * k_eff).transpose(-1, -2) @ u_wy

    def _routed_bank_readout(
        vec_c: torch.Tensor,
        state_bank: torch.Tensor,
        top_idx: torch.Tensor,
        top_w: torch.Tensor,
    ) -> torch.Tensor:
        bsz, c_len, hv, dk = vec_c.shape
        num_pages = state_bank.shape[1]
        dv = state_bank.shape[-1]
        k_sel = top_idx.shape[-1]

        if k_sel < num_pages and c_len == 1:
            g_idx = top_idx[:, 0, :, None, None, None].expand(bsz, k_sel, hv, dk, dv)
            s_sel = torch.gather(state_bank, dim=1, index=g_idx)
            if s_sel.dtype != vec_c.dtype:
                s_sel = s_sel.to(dtype=vec_c.dtype)
            v_q = vec_c.permute(0, 2, 1, 3).unsqueeze(1)
            picked_1 = (v_q @ s_sel).squeeze(-2)
            return (picked_1 * top_w[:, 0, :, None, None]).sum(dim=1, keepdim=True)

        if k_sel < num_pages and c_len <= 4:
            flat_idx = top_idx.reshape(bsz, c_len * k_sel, 1, 1, 1).expand(bsz, c_len * k_sel, hv, dk, dv)
            s_sel = torch.gather(state_bank, dim=1, index=flat_idx).view(bsz, c_len, k_sel, hv, dk, dv)
            if s_sel.dtype != vec_c.dtype:
                s_sel = s_sel.to(dtype=vec_c.dtype)
            v_q = vec_c.unsqueeze(2).unsqueeze(-2)
            picked_s = (v_q @ s_sel).squeeze(-2)
            return (picked_s * top_w.unsqueeze(-1).unsqueeze(-1)).sum(dim=2)

        bank_f = state_bank if state_bank.dtype == vec_c.dtype else state_bank.to(dtype=vec_c.dtype)
        v_all = vec_c.permute(0, 2, 1, 3).unsqueeze(1)
        all_out = (v_all @ bank_f).permute(0, 3, 1, 2, 4)
        gather_idx = top_idx.unsqueeze(-1).unsqueeze(-1).expand(bsz, c_len, k_sel, hv, dv)
        picked = torch.gather(all_out, dim=2, index=gather_idx)
        return (picked * top_w.unsqueeze(-1).unsqueeze(-1)).sum(dim=2)

    def _far_readout(
        q_hat_c: torch.Tensor,
        s_far: torch.Tensor,
        read_logits_c: torch.Tensor,
        top_k_pages: int,
    ) -> torch.Tensor:
        pi_read = F.softmax(read_logits_c, dim=-1)
        top_vals, top_idx = torch.topk(pi_read, k=top_k_pages, dim=-1)
        top_w = top_vals / top_vals.sum(dim=-1, keepdim=True).clamp_min(1e-30)
        return _routed_bank_readout(q_hat_c, s_far, top_idx, top_w)

    def _compute_surprise_and_routing(
        k_hat_c: torch.Tensor,
        v_c: torch.Tensor,
        s_chunk_start_bank: torch.Tensor,
        sbias_c: torch.Tensor,
        surprise_slope: torch.Tensor,
        surprise_thresh: torch.Tensor,
        write_logits_c: torch.Tensor,
        top_k_pages: int,
        wmask_c: torch.Tensor | None,
        *,
        use_surprise: bool = True,
        surprise_mode: str = "routed",
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bsz, c_len, hv, _dk = k_hat_c.shape
        pi_write = F.softmax(write_logits_c, dim=-1)
        top_vals, top_idx = torch.topk(pi_write, k=top_k_pages, dim=-1)
        top_probs = top_vals / top_vals.sum(dim=-1, keepdim=True).clamp_min(1e-30)

        if not use_surprise:
            g_root = torch.ones(bsz, c_len, hv, device=k_hat_c.device, dtype=k_hat_c.dtype)
        else:
            if surprise_mode == "routed" and s_chunk_start_bank.ndim == 5:
                pred_v = _routed_bank_readout(k_hat_c, s_chunk_start_bank, top_idx, top_probs)
            else:
                s0 = s_chunk_start_bank[:, 0] if s_chunk_start_bank.ndim == 5 else s_chunk_start_bank
                if s0.dtype != k_hat_c.dtype:
                    s0 = s0.to(dtype=k_hat_c.dtype)
                pred_v = (k_hat_c.transpose(1, 2) @ s0).transpose(1, 2)
            eps_err = _safe_norm(v_c - pred_v, dim=-1) / (_safe_norm(v_c, dim=-1) + 1e-6)
            slope = surprise_slope.view(1, 1, hv)
            thresh = surprise_thresh.view(1, 1, hv)
            sb_eff = sbias_c.squeeze(-1) if (sbias_c.ndim == 4 and sbias_c.shape[-1] == 1) else sbias_c
            g_root = torch.sigmoid(slope * (eps_err - thresh) + sb_eff)

        if wmask_c is not None:
            g_root = g_root * wmask_c.unsqueeze(-1)

        top_roots = torch.sqrt(top_probs.clamp_min(1e-30))
        g_active = top_roots.unsqueeze(-1) * g_root.unsqueeze(2)
        write_root = torch.zeros_like(pi_write).scatter(-1, top_idx, top_roots)
        g_page = write_root.unsqueeze(-1) * g_root.unsqueeze(2)
        return g_page, top_idx, g_active

    def _coerce_hssd_state(
        init_st: Any,
        *,
        bsz: int,
        num_pages: int,
        hv: int,
        dk: int,
        dv: int,
        c_size: int,
        w_chunks: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> _HSSDCacheState:
        if init_st is None:
            return _HSSDCacheState.zeros(
                bsz, num_pages, hv, dk, dv, chunk_size=c_size, window_chunks=w_chunks, device=device, dtype=dtype
            )
        if isinstance(init_st, _HSSDCacheState):
            if (
                init_st.device == device
                and init_st.dtype == dtype
                and init_st.theta_cum.dtype == torch.float64
                and init_st.swa_k_ring.dtype == dtype
                and init_st.swa_v_ring.dtype == dtype
                and all(t.dtype == dtype for t in init_st.boundary_ring)
                and init_st.has_compact_storage()
            ):
                return init_st
            return init_st.to(device=device, dtype=dtype)
        if hasattr(init_st, "page_states") and hasattr(init_st, "boundary_ring"):
            ps = _compact(init_st.page_states.to(device=device, dtype=dtype))
            sc_raw = init_st.step_count
            if isinstance(sc_raw, torch.Tensor):
                sc_coerced: int | torch.Tensor = sc_raw.to(device=device, dtype=torch.long)
            else:
                sc_coerced = int(sc_raw)
            return _HSSDCacheState(
                step_count=sc_coerced,
                chunk_size=int(init_st.chunk_size),
                window_chunks=int(init_st.window_chunks),
                page_states=ps,
                theta_cum=_compact(init_st.theta_cum.to(device=device, dtype=torch.float64)),
                boundary_ring=tuple(_compact(t.to(device=device, dtype=dtype)) for t in init_st.boundary_ring),
                swa_k_ring=_compact(init_st.swa_k_ring.to(device=device, dtype=dtype)),
                swa_v_ring=_compact(init_st.swa_v_ring.to(device=device, dtype=dtype)),
                swa_valid_ring=_compact(init_st.swa_valid_ring.to(device=device, dtype=torch.bool)),
            )
        if isinstance(init_st, torch.Tensor):
            st_t = _upcast(init_st).to(device=device, dtype=dtype)
            ps = st_t.unsqueeze(1).expand(bsz, num_pages, hv, dk, dv).clone() if st_t.ndim == 4 else st_t.clone()
            base = _HSSDCacheState.zeros(
                bsz, num_pages, hv, dk, dv, chunk_size=c_size, window_chunks=w_chunks, device=device, dtype=dtype
            )
            base.page_states = _compact(ps)
            base.boundary_ring = tuple(ps.clone() for _ in range(w_chunks + 1))
            return base
        raise TypeError(f"Unsupported HSSD initial_state: {type(init_st).__name__}")

    def _recurrent_hssd_step(
        q_in: torch.Tensor,
        k_in: torch.Tensor,
        v_in: torch.Tensor,
        gamma_page_scale: torch.Tensor,
        gamma_chan_pair: torch.Tensor,
        b_gate: torch.Tensor,
        w_gate: torch.Tensor,
        theta: torch.Tensor,
        surprise_bias: torch.Tensor,
        surprise_slope: torch.Tensor,
        surprise_thresh: torch.Tensor,
        write_logits: torch.Tensor,
        read_logits: torch.Tensor,
        q_swa: torch.Tensor,
        k_swa: torch.Tensor,
        v_swa: torch.Tensor,
        state: _HSSDCacheState,
        *,
        boltzmann_gate: _StandalonePeakBoltzmannGate,
        top_k_pages: int,
        write_mask: torch.Tensor | None,
        scale: float | None,
        acc_dt: torch.dtype,
        target_dt: torch.dtype,
        use_swa: bool,
        use_phase: bool,
        use_surprise: bool,
        surprise_mode: str,
    ) -> tuple[torch.Tensor, _HSSDCacheState]:
        bsz, _t1, hv, dv = v_in.shape
        dk = k_in.shape[-1]
        m_pages = state.page_states.shape[1]
        st = _coerce_hssd_state(
            state,
            bsz=bsz,
            num_pages=m_pages,
            hv=hv,
            dk=dk,
            dv=dv,
            c_size=int(state.chunk_size),
            w_chunks=int(state.window_chunks),
            device=q_in.device,
            dtype=target_dt,
        )

        q_i = q_in.to(acc_dt)
        k_i = k_in.to(acc_dt)
        v_i = v_in.to(acc_dt)
        ps_gamma = gamma_page_scale.to(device=q_in.device, dtype=acc_dt).clamp_min(0.0)
        ch_gamma = gamma_chan_pair.to(device=q_in.device, dtype=acc_dt).clamp_min(0.0)
        bf = b_gate.to(acc_dt)
        wf = w_gate.to(acc_dt)
        sb = surprise_bias.to(acc_dt)
        slp = surprise_slope.to(acc_dt)
        thr = surprise_thresh.to(acc_dt)
        wl = write_logits.to(acc_dt)
        rl = read_logits.to(acc_dt)
        qs = q_swa.to(acc_dt)
        ks = k_swa.to(acc_dt)
        vs = v_swa.to(acc_dt)

        th64 = theta.to(dtype=torch.float64)
        if write_mask is not None:
            wm = write_mask.reshape(bsz, 1).to(device=q_in.device, dtype=acc_dt).clamp(0.0, 1.0)
            valid_tok = wm.squeeze(1) > 0.5
            th64 = th64 * wm.to(dtype=torch.float64).unsqueeze(-1).unsqueeze(-1)
        else:
            wm = None
            valid_tok = torch.ones(bsz, device=q_in.device, dtype=torch.bool)
        if not use_phase:
            th64 = torch.zeros_like(th64)

        idx0 = int(st.step_count[0].item())
        c_size = st.chunk_size
        w_chunks = st.window_chunks
        s_cstart_bank = st.boundary_ring[-1]

        theta_cum = _compact(torch.remainder(st.theta_cum + th64.squeeze(1), 2.0 * math.pi))
        sc = (dk ** -0.5) if scale is None else float(scale)
        q_hat = _apply_2d_rotation(q_i, -theta_cum.unsqueeze(1)) if use_phase else q_i
        if sc != 1.0:
            q_hat = q_hat * sc
        k_hat = _apply_2d_rotation(k_i, -theta_cum.unsqueeze(1)) if use_phase else k_i

        o_far = _far_readout(q_hat, st.boundary_ring[0], rl, top_k_pages)

        max_win = st.max_swa_len
        slot = idx0 % max_win
        slot_idx = torch.tensor([slot], device=q_in.device, dtype=torch.long)
        ks_ring = ks if ks.dtype == target_dt else ks.to(dtype=target_dt)
        vs_ring = vs if vs.dtype == target_dt else vs.to(dtype=target_dt)
        swa_kr = _compact(st.swa_k_ring.index_copy(1, slot_idx, ks_ring))
        swa_vr = _compact(st.swa_v_ring.index_copy(1, slot_idx, vs_ring))
        swa_vl = _compact(st.swa_valid_ring.index_copy(1, slot_idx, valid_tok.unsqueeze(1)))

        if use_swa:
            _n_c, _k_f, far_cut = _disjoint_horizon_bounds(idx0 + 1, c_size, w_chunks)
            win_idx = torch.arange(far_cut, idx0 + 1, device=q_in.device) % max_win
            if win_idx.numel() > 1:
                prior_win = win_idx[:-1]
                k_prior = st.swa_k_ring.index_select(1, prior_win)
                v_prior = st.swa_v_ring.index_select(1, prior_win)
                if k_prior.dtype != acc_dt:
                    k_prior = k_prior.to(dtype=acc_dt)
                if v_prior.dtype != acc_dt:
                    v_prior = v_prior.to(dtype=acc_dt)
                k_win = torch.cat([k_prior, ks], dim=1).transpose(1, 2)
                v_win = torch.cat([v_prior, vs], dim=1).transpose(1, 2)
            else:
                k_win = ks.transpose(1, 2)
                v_win = vs.transpose(1, 2)
            val_win = swa_vl.index_select(1, win_idx)[:, None, None, :]

            scores = (qs.transpose(1, 2) @ k_win.transpose(-1, -2)) * (dk ** -0.5)
            has_v = val_win.any(dim=-1)
            safe_v = torch.where(has_v.unsqueeze(-1), val_win, torch.ones_like(val_win))
            probs = torch.softmax(scores.masked_fill(~safe_v, float("-inf")), dim=-1)
            probs = torch.where(has_v.unsqueeze(-1), probs, torch.zeros_like(probs))
            o_loc = (probs @ v_win).transpose(1, 2)
            p_hit = boltzmann_gate(scores, valid_mask=val_win).transpose(1, 2).unsqueeze(-1)
            out = p_hit * o_loc + (1.0 - p_hit) * o_far
        else:
            out = o_far

        _g_page, top_idx, g_active = _compute_surprise_and_routing(
            k_hat,
            v_i,
            s_cstart_bank,
            sb,
            slp,
            thr,
            wl,
            top_k_pages,
            wm,
            use_surprise=use_surprise,
            surprise_mode=surprise_mode,
        )
        sqrt_b = _safe_sqrt(bf[:, 0])
        sqrt_w = _safe_sqrt(wf[:, 0])
        g_sel = g_active[:, 0]
        t_idx = top_idx[:, 0]
        base_pages = st.page_states
        p_idx = t_idx[:, :, None, None, None].expand(bsz, top_k_pages, hv, dk, dv)
        s_sel = torch.gather(base_pages, dim=1, index=p_idx)
        if s_sel.dtype != acc_dt:
            s_sel = s_sel.to(dtype=acc_dt)

        ch0 = ch_gamma[:, 0]
        if ch0.shape[-1] == dk // 2:
            ch0 = ch0.repeat_interleave(2, dim=-1)
        ps_sel = ps_gamma.index_select(0, t_idx.reshape(-1)).view(bsz, top_k_pages, 1, 1)
        gam_sel = ps_sel * ch0.unsqueeze(1)

        dec_sel = (-(g_sel * g_sel).unsqueeze(-1) * gam_sel).clamp(max=0.0).exp().unsqueeze(-1) * s_sel
        ke_sel = g_sel.unsqueeze(-1) * (sqrt_b * k_hat[:, 0]).unsqueeze(1)
        ve_sel = g_sel.unsqueeze(-1) * (sqrt_w * v_i[:, 0]).unsqueeze(1)
        mem_sel = (ke_sel.unsqueeze(-2) @ dec_sel).squeeze(-2)
        updated_sel = dec_sel + ke_sel.unsqueeze(-1) * (ve_sel - mem_sel).unsqueeze(-2)
        updated_sel_cast = updated_sel if updated_sel.dtype == target_dt else updated_sel.to(dtype=target_dt)
        new_pages = _compact(base_pages.scatter(1, p_idx, updated_sel_cast))

        new_step_int = idx0 + 1
        new_step_vec = _compact(st.step_count + 1)
        new_bring = (*st.boundary_ring[1:], new_pages.clone()) if new_step_int % c_size == 0 else st.boundary_ring

        next_st = _HSSDCacheState(
            step_count=new_step_vec,
            chunk_size=c_size,
            window_chunks=w_chunks,
            page_states=new_pages,
            theta_cum=theta_cum,
            boundary_ring=new_bring,
            swa_k_ring=swa_kr,
            swa_v_ring=swa_vr,
            swa_valid_ring=swa_vl,
        )
        return out, next_st

    def _run_chunk_group(
        chunk_bounds: list[tuple[int, int]],
        *,
        step_offset: int,
        swa_base_global: int,
        chunk_size: int,
        window_chunks: int,
        num_pages: int,
        top_k_pages: int,
        use_swa: bool,
        use_surprise: bool,
        surprise_mode: str,
        q_hat: torch.Tensor,
        k_hat: torch.Tensor,
        v: torch.Tensor,
        gamma_page_scale: torch.Tensor,
        gamma_chan_seg: torch.Tensor,
        bf: torch.Tensor,
        wf: torch.Tensor,
        sb: torch.Tensor,
        slp: torch.Tensor,
        thr: torch.Tensor,
        wl: torch.Tensor,
        rl: torch.Tensor,
        qs: torch.Tensor,
        all_ks: torch.Tensor,
        all_vs: torch.Tensor,
        all_valid_swa: torch.Tensor,
        wm: torch.Tensor | None,
        gate_log_kappa: torch.Tensor,
        gate_tau: torch.Tensor,
        page_states: torch.Tensor,
        boundary_ring: tuple[torch.Tensor, ...],
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, ...]]:
        bsz = q_hat.shape[0]
        dk = k_hat.shape[-1]
        hv = k_hat.shape[2]
        outs: list[torch.Tensor] = []

        kappa = gate_log_kappa.exp()
        kappa_b = kappa[0] if kappa.numel() == 1 else kappa.view(1, hv, 1)
        tau_b = gate_tau[0] if gate_tau.numel() == 1 else gate_tau.view(1, hv, 1)

        n_grp = len(chunk_bounds)
        base_l0 = chunk_bounds[0][0] if n_grp > 0 else 0
        g_start = step_offset + base_l0
        win_prior = window_chunks * chunk_size
        can_batch_swa = (
            use_swa
            and n_grp > 1
            and g_start % chunk_size == 0
            and swa_base_global <= max(0, g_start - win_prior)
            and all(
                chunk_bounds[c][0] == base_l0 + c * chunk_size
                and chunk_bounds[c][1] == base_l0 + (c + 1) * chunk_size
                for c in range(n_grp)
            )
        )

        o_loc_batched: torch.Tensor | None = None
        p_hit_batched: torch.Tensor | None = None
        if can_batch_swa:
            win_full = (window_chunks + 1) * chunk_size
            buf_len = (window_chunks + n_grp) * chunk_size
            want_start = g_start - win_prior
            if want_start >= swa_base_global:
                s_off = want_start - swa_base_global
                ks_buf = all_ks[:, s_off : s_off + buf_len]
                vs_buf = all_vs[:, s_off : s_off + buf_len]
                val_buf = all_valid_swa[:, s_off : s_off + buf_len]
            else:
                pad_left = swa_base_global - want_start
                take_len = buf_len - pad_left
                ks_buf = F.pad(all_ks[:, :take_len], (0, 0, 0, 0, pad_left, 0), value=0.0)
                vs_buf = F.pad(all_vs[:, :take_len], (0, 0, 0, 0, pad_left, 0), value=0.0)
                val_buf = F.pad(all_valid_swa[:, :take_len], (pad_left, 0), value=False)

            k_win_t = ks_buf.unfold(1, win_full, chunk_size)
            v_win_b = vs_buf.unfold(1, win_full, chunk_size).transpose(-1, -2)
            val_win_b = val_buf.unfold(1, win_full, chunk_size)

            q_loc_b = (
                qs[:, base_l0 : base_l0 + n_grp * chunk_size]
                .reshape(bsz, n_grp, chunk_size, hv, dk)
                .transpose(2, 3)
            )
            scores_b = (q_loc_b @ k_win_t) * (dk ** -0.5)

            r_idx = torch.arange(chunk_size, device=qs.device)[:, None]
            w_idx = torch.arange(win_full, device=qs.device)[None, :]
            causal_b = w_idx <= (win_prior + r_idx)
            valid_mask_b = causal_b[None, None, None, :, :] & val_win_b[:, :, None, None, :]

            has_valid_b = valid_mask_b.any(dim=-1)
            safe_vm_b = torch.where(has_valid_b.unsqueeze(-1), valid_mask_b, torch.ones_like(valid_mask_b))
            masked_sc_b = scores_b.masked_fill(~safe_vm_b, float("-inf"))

            probs_b = torch.softmax(masked_sc_b, dim=-1)
            probs_b = torch.where(has_valid_b.unsqueeze(-1), probs_b, torch.zeros_like(probs_b))
            o_loc_batched = (probs_b @ v_win_b).transpose(2, 3)

            counts_b = valid_mask_b.expand_as(scores_b).sum(dim=-1)
            has_v_sc_b = counts_b > 0
            n_eff_b = counts_b.clamp_min(1).to(scores_b.dtype)
            lse_b = torch.logsumexp(masked_sc_b, dim=-1)
            max_sc_b = masked_sc_b.max(dim=-1).values
            stat_b = 0.5 * (lse_b - torch.log(n_eff_b)) + 0.5 * (
                max_sc_b - torch.sqrt(2.0 * torch.log(n_eff_b.clamp_min(2.0)))
            )
            p_hit_b = torch.sigmoid(kappa_b * (stat_b - tau_b))
            p_hit_batched = torch.where(has_v_sc_b, p_hit_b, torch.zeros_like(p_hit_b)).transpose(2, 3).unsqueeze(-1)

        p_cur = page_states
        ring_cur = boundary_ring

        for c_i, (l0, l1) in enumerate(chunk_bounds):
            g0 = step_offset + l0
            g1 = step_offset + l1

            q_c = q_hat[:, l0:l1]
            k_c = k_hat[:, l0:l1]
            v_c = v[:, l0:l1]
            rl_c = rl[:, l0:l1]
            wl_c = wl[:, l0:l1]
            sb_c = sb[:, l0:l1]
            wm_c = None if wm is None else wm[:, l0:l1]

            o_far = _far_readout(q_c, ring_cur[0], rl_c, top_k_pages)

            if o_loc_batched is not None and p_hit_batched is not None:
                o_loc = o_loc_batched[:, c_i]
                p_hit = p_hit_batched[:, c_i]
                outs.append(p_hit * o_loc + (1.0 - p_hit) * o_far)
            elif use_swa:
                _n_idx, _k_far, far_cutoff = _disjoint_horizon_bounds(g0 + 1, chunk_size, window_chunks)
                s0 = far_cutoff - swa_base_global
                s1 = g1 - swa_base_global
                k_win = all_ks[:, s0:s1].transpose(1, 2)
                v_win = all_vs[:, s0:s1].transpose(1, 2)
                val_keys = all_valid_swa[:, s0:s1]

                q_pos = torch.arange(g0, g1, device=q_c.device)[:, None]
                k_pos = torch.arange(far_cutoff, g1, device=q_c.device)[None, :]
                valid_mask = (k_pos <= q_pos)[None, None, :, :] & val_keys[:, None, None, :]

                q_loc = qs[:, l0:l1].transpose(1, 2)
                scores = (q_loc @ k_win.transpose(-1, -2)) * (dk ** -0.5)
                has_valid = valid_mask.any(dim=-1)
                safe_vm = torch.where(has_valid.unsqueeze(-1), valid_mask, torch.ones_like(valid_mask))
                masked_sc = scores.masked_fill(~safe_vm, float("-inf"))

                probs = torch.softmax(masked_sc, dim=-1)
                probs = torch.where(has_valid.unsqueeze(-1), probs, torch.zeros_like(probs))
                o_loc = (probs @ v_win).transpose(1, 2)

                counts = valid_mask.expand_as(scores).sum(dim=-1)
                has_v_sc = counts > 0
                n_eff = counts.clamp_min(1).to(scores.dtype)
                lse = torch.logsumexp(masked_sc, dim=-1)
                max_sc = masked_sc.max(dim=-1).values
                stat = 0.5 * (lse - torch.log(n_eff)) + 0.5 * (
                    max_sc - torch.sqrt(2.0 * torch.log(n_eff.clamp_min(2.0)))
                )
                p_hit = torch.sigmoid(kappa_b * (stat - tau_b))
                p_hit = torch.where(has_v_sc, p_hit, torch.zeros_like(p_hit)).transpose(1, 2).unsqueeze(-1)
                outs.append(p_hit * o_loc + (1.0 - p_hit) * o_far)
            else:
                outs.append(o_far)

            g_page, top_idx, _g_act = _compute_surprise_and_routing(
                k_c,
                v_c,
                ring_cur[-1],
                sb_c,
                slp,
                thr,
                wl_c,
                top_k_pages,
                wm_c,
                use_surprise=use_surprise,
                surprise_mode=surprise_mode,
            )

            chan_c = gamma_chan_seg[:, l0:l1]
            if chan_c.shape[-1] == dk // 2:
                chan_c = chan_c.repeat_interleave(2, dim=-1)
            gamma_c = gamma_page_scale.view(1, 1, num_pages, 1, 1) * chan_c.unsqueeze(2)

            sqrt_b = _safe_sqrt(bf[:, l0:l1])
            sqrt_w = _safe_sqrt(wf[:, l0:l1])
            k_eff = (g_page.unsqueeze(-1) * (sqrt_b * k_c).unsqueeze(2)).permute(0, 2, 3, 1, 4)
            v_eff = (g_page.unsqueeze(-1) * (sqrt_w * v_c).unsqueeze(2)).permute(0, 2, 3, 1, 4)
            log_a = (-(g_page * g_page).unsqueeze(-1) * gamma_c).permute(0, 2, 3, 1, 4)
            if not torch.is_grad_enabled() and not _is_compiling() and top_k_pages < num_pages:
                uniq_idx = torch.unique(top_idx)
                n_uniq = int(uniq_idx.numel())
                if 0 < n_uniq < num_pages:
                    p_act = _wy_kda_state_update(
                        p_cur.index_select(1, uniq_idx),
                        k_eff.index_select(1, uniq_idx),
                        v_eff.index_select(1, uniq_idx),
                        log_a.index_select(1, uniq_idx),
                    )
                    p_cur = p_cur.index_copy(1, uniq_idx, p_act)
                else:
                    p_cur = _wy_kda_state_update(p_cur, k_eff, v_eff, log_a)
            else:
                p_cur = _wy_kda_state_update(p_cur, k_eff, v_eff, log_a)

            if g1 % chunk_size == 0:
                ring_cur = (*ring_cur[1:], p_cur)

        seg_out = outs[0] if len(outs) == 1 else torch.cat(outs, dim=1)
        return seg_out, p_cur, ring_cur

    def _run_hssd_homog(
        adapter: _StandaloneHSSDAdapter,
        x_in: torch.Tensor,
        wmask: torch.Tensor | None,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        st: _HSSDCacheState,
        *,
        out_final: bool,
        use_l2: bool,
        scale: float | None,
        init_dtype: torch.dtype,
        acc_dt: torch.dtype,
        ret_dt: torch.dtype,
        rec_only: bool = False,
    ) -> tuple[torch.Tensor, _HSSDCacheState | None]:
        bsz, seq_len, _ = x_in.shape
        hv, dk, dv = adapter.num_v_heads, adapter.head_k_dim, adapter.head_v_dim
        if seq_len == 0:
            empty_o = query.new_zeros(bsz, 0, hv, dv, dtype=init_dtype)
            if not out_final:
                return empty_o, None
            if st.dtype == ret_dt and st.theta_cum.dtype == torch.float64 and st.has_compact_storage():
                return empty_o, st
            return empty_o, st.to(ret_dt).compact()

        proj = adapter.project_inputs(x_in, query, key, value, g, beta, use_qk_l2norm=use_l2)
        c_size, w_chunks, m_pages, tk_pages = (
            adapter.chunk_size,
            adapter.window_chunks,
            adapter.num_pages,
            adapter.k_pages,
        )
        slp = _upcast(adapter.log_surprise_slope).exp().to(acc_dt)
        thr = _upcast(adapter.surprise_thresh).to(acc_dt)

        if seq_len == 1:
            o_1, st_1 = _recurrent_hssd_step(
                proj["q"],
                proj["k"],
                proj["v"],
                proj["gamma_page_scale"],
                proj["gamma_chan_pair"],
                proj["b_gate"],
                proj["w_gate"],
                proj["theta"],
                proj["sb"],
                slp,
                thr,
                proj["wl"],
                proj["rl"],
                proj["qs"],
                proj["ks"],
                proj["vs"],
                st,
                boltzmann_gate=adapter.boltzmann_gate,
                top_k_pages=tk_pages,
                write_mask=wmask,
                scale=scale,
                acc_dt=acc_dt,
                target_dt=ret_dt,
                use_swa=adapter.use_swa,
                use_phase=adapter.use_phase,
                use_surprise=adapter.use_surprise,
                surprise_mode=adapter.surprise_mode,
            )
            return o_1.to(init_dtype), (st_1 if out_final else None)

        if rec_only:
            outs_r: list[torch.Tensor] = []
            st_r = st
            for i in range(seq_len):
                sl_i = slice(i, i + 1)
                wm_i = None if wmask is None else wmask[:, sl_i]
                o_i, st_r = _recurrent_hssd_step(
                    proj["q"][:, sl_i],
                    proj["k"][:, sl_i],
                    proj["v"][:, sl_i],
                    proj["gamma_page_scale"],
                    proj["gamma_chan_pair"][:, sl_i],
                    proj["b_gate"][:, sl_i],
                    proj["w_gate"][:, sl_i],
                    proj["theta"][:, sl_i],
                    proj["sb"][:, sl_i],
                    slp,
                    thr,
                    proj["wl"][:, sl_i],
                    proj["rl"][:, sl_i],
                    proj["qs"][:, sl_i],
                    proj["ks"][:, sl_i],
                    proj["vs"][:, sl_i],
                    st_r,
                    boltzmann_gate=adapter.boltzmann_gate,
                    top_k_pages=tk_pages,
                    write_mask=wm_i,
                    scale=scale,
                    acc_dt=acc_dt,
                    target_dt=ret_dt,
                    use_swa=adapter.use_swa,
                    use_phase=adapter.use_phase,
                    use_surprise=adapter.use_surprise,
                    surprise_mode=adapter.surprise_mode,
                )
                outs_r.append(o_i)
            out_r = torch.cat(outs_r, dim=1).to(init_dtype)
            return out_r, (st_r if out_final else None)

        st_acc = st if st.dtype == acc_dt else st.to(device=query.device, dtype=acc_dt)
        q_f = proj["q"].to(acc_dt)
        k_f = proj["k"].to(acc_dt)
        v_f = proj["v"].to(acc_dt)
        ps_gamma = proj["gamma_page_scale"].to(device=query.device, dtype=acc_dt).clamp_min(0.0)
        ch_gamma = proj["gamma_chan_pair"].to(device=query.device, dtype=acc_dt).clamp_min(0.0)
        bf = proj["b_gate"].to(acc_dt)
        wf = proj["w_gate"].to(acc_dt)
        sb = proj["sb"].to(acc_dt)
        wl = proj["wl"].to(acc_dt)
        rl = proj["rl"].to(acc_dt)
        qs = proj["qs"].to(acc_dt)
        ks = proj["ks"].to(acc_dt)
        vs = proj["vs"].to(acc_dt)

        th64 = proj["theta"].to(dtype=torch.float64)
        if wmask is not None:
            wm = wmask.reshape(bsz, seq_len).to(device=query.device, dtype=acc_dt).clamp(0.0, 1.0)
            valid_cur = wm > 0.5
            th64 = th64 * wm.to(dtype=torch.float64).unsqueeze(-1).unsqueeze(-1)
        else:
            wm = None
            valid_cur = torch.ones(bsz, seq_len, device=query.device, dtype=torch.bool)
        if not adapter.use_phase:
            th64 = torch.zeros_like(th64)

        theta_cum_all = torch.remainder(
            st_acc.theta_cum.to(dtype=torch.float64).unsqueeze(1) + torch.cumsum(th64, dim=1),
            2.0 * math.pi,
        )
        sc = (dk ** -0.5) if scale is None else float(scale)
        q_hat = _apply_2d_rotation(q_f, -theta_cum_all) if adapter.use_phase else q_f
        if sc != 1.0:
            q_hat = q_hat * sc
        k_hat = _apply_2d_rotation(k_f, -theta_cum_all) if adapter.use_phase else k_f

        step_offset = int(st_acc.step_count[0].item())
        max_win = st_acc.max_swa_len
        _, _, swa_base_global = _disjoint_horizon_bounds(step_offset + 1, c_size, w_chunks)
        if step_offset > swa_base_global:
            prior_idx = torch.arange(swa_base_global, step_offset, device=query.device) % max_win
            all_ks = torch.cat([st_acc.swa_k_ring.index_select(1, prior_idx), ks], dim=1)
            all_vs = torch.cat([st_acc.swa_v_ring.index_select(1, prior_idx), vs], dim=1)
            all_valid = torch.cat([st_acc.swa_valid_ring.index_select(1, prior_idx), valid_cur], dim=1)
        else:
            all_ks = ks
            all_vs = vs
            all_valid = valid_cur

        gate_log_kappa = _upcast(adapter.boltzmann_gate.log_kappa).to(acc_dt)
        gate_tau = _upcast(adapter.boltzmann_gate.tau).to(acc_dt)

        all_chunks = _chunk_slices_for_span(step_offset, seq_len, c_size)
        seg_step = max(1, int(adapter.segment_chunks))
        need_grad = any(
            isinstance(t, torch.Tensor) and t.requires_grad
            for t in (
                q_hat,
                k_hat,
                v_f,
                ps_gamma,
                ch_gamma,
                bf,
                wf,
                sb,
                slp,
                thr,
                wl,
                rl,
                qs,
                all_ks,
                all_vs,
                gate_log_kappa,
                gate_tau,
                st_acc.page_states,
            )
        )
        use_ckpt = bool(
            adapter.checkpoint_segments and torch.is_grad_enabled() and need_grad and len(all_chunks) > seg_step
        )

        p_cur = st_acc.page_states
        ring_cur = st_acc.boundary_ring
        seg_outs: list[torch.Tensor] = []

        for c_idx in range(0, len(all_chunks), seg_step):
            c_group = all_chunks[c_idx : c_idx + seg_step]
            l_start, l_end = c_group[0][0], c_group[-1][1]
            g_first = step_offset + l_start
            g_last = step_offset + l_end
            _, _, grp_far_cut = _disjoint_horizon_bounds(g_first + 1, c_size, w_chunks)
            s_lo = grp_far_cut - swa_base_global
            s_hi = g_last - swa_base_global
            rel_bounds = [(a - l_start, b - l_start) for a, b in c_group]

            q_seg = q_hat[:, l_start:l_end]
            k_seg = k_hat[:, l_start:l_end]
            v_seg = v_f[:, l_start:l_end]
            ch_seg = ch_gamma[:, l_start:l_end]
            bf_seg = bf[:, l_start:l_end]
            wf_seg = wf[:, l_start:l_end]
            sb_seg = sb[:, l_start:l_end]
            wl_seg = wl[:, l_start:l_end]
            rl_seg = rl[:, l_start:l_end]
            qs_seg = qs[:, l_start:l_end]
            ks_seg = all_ks[:, s_lo:s_hi]
            vs_seg = all_vs[:, s_lo:s_hi]
            val_seg = all_valid[:, s_lo:s_hi]
            wm_seg = None if wm is None else wm[:, l_start:l_end]

            if use_ckpt:

                def _ckpt_fn(
                    qc: torch.Tensor,
                    kc: torch.Tensor,
                    vc: torch.Tensor,
                    psc: torch.Tensor,
                    chc: torch.Tensor,
                    bfc: torch.Tensor,
                    wfc: torch.Tensor,
                    sbc: torch.Tensor,
                    slpc: torch.Tensor,
                    thrc: torch.Tensor,
                    wlc: torch.Tensor,
                    rlc: torch.Tensor,
                    qsc: torch.Tensor,
                    ksc: torch.Tensor,
                    vsc: torch.Tensor,
                    glk: torch.Tensor,
                    gtau: torch.Tensor,
                    pst: torch.Tensor,
                    *ring_tensors: torch.Tensor,
                    _rb: list[tuple[int, int]] = rel_bounds,
                    _goff: int = g_first,
                    _sbase: int = grp_far_cut,
                    _val: torch.Tensor = val_seg,
                    _wm: torch.Tensor | None = wm_seg,
                ):
                    o_g, p_n, r_n = _run_chunk_group(
                        _rb,
                        step_offset=_goff,
                        swa_base_global=_sbase,
                        chunk_size=c_size,
                        window_chunks=w_chunks,
                        num_pages=m_pages,
                        top_k_pages=tk_pages,
                        use_swa=adapter.use_swa,
                        use_surprise=adapter.use_surprise,
                        surprise_mode=adapter.surprise_mode,
                        q_hat=qc,
                        k_hat=kc,
                        v=vc,
                        gamma_page_scale=psc,
                        gamma_chan_seg=chc,
                        bf=bfc,
                        wf=wfc,
                        sb=sbc,
                        slp=slpc,
                        thr=thrc,
                        wl=wlc,
                        rl=rlc,
                        qs=qsc,
                        all_ks=ksc,
                        all_vs=vsc,
                        all_valid_swa=_val,
                        wm=_wm,
                        gate_log_kappa=glk,
                        gate_tau=gtau,
                        page_states=pst,
                        boundary_ring=ring_tensors,
                    )
                    return (o_g, p_n, *r_n)

                ckpt_out = checkpoint(
                    _ckpt_fn,
                    q_seg,
                    k_seg,
                    v_seg,
                    ps_gamma,
                    ch_seg,
                    bf_seg,
                    wf_seg,
                    sb_seg,
                    slp,
                    thr,
                    wl_seg,
                    rl_seg,
                    qs_seg,
                    ks_seg,
                    vs_seg,
                    gate_log_kappa,
                    gate_tau,
                    p_cur,
                    *ring_cur,
                    use_reentrant=False,
                )
                o_grp, p_cur = ckpt_out[0], ckpt_out[1]
                ring_cur = tuple(ckpt_out[2:])
            else:
                o_grp, p_cur, ring_cur = _run_chunk_group(
                    rel_bounds,
                    step_offset=g_first,
                    swa_base_global=grp_far_cut,
                    chunk_size=c_size,
                    window_chunks=w_chunks,
                    num_pages=m_pages,
                    top_k_pages=tk_pages,
                    use_swa=adapter.use_swa,
                    use_surprise=adapter.use_surprise,
                    surprise_mode=adapter.surprise_mode,
                    q_hat=q_seg,
                    k_hat=k_seg,
                    v=v_seg,
                    gamma_page_scale=ps_gamma,
                    gamma_chan_seg=ch_seg,
                    bf=bf_seg,
                    wf=wf_seg,
                    sb=sb_seg,
                    slp=slp,
                    thr=thr,
                    wl=wl_seg,
                    rl=rl_seg,
                    qs=qs_seg,
                    all_ks=ks_seg,
                    all_vs=vs_seg,
                    all_valid_swa=val_seg,
                    wm=wm_seg,
                    gate_log_kappa=gate_log_kappa,
                    gate_tau=gate_tau,
                    page_states=p_cur,
                    boundary_ring=ring_cur,
                )
            seg_outs.append(o_grp)

        out = (seg_outs[0] if len(seg_outs) == 1 else torch.cat(seg_outs, dim=1)).to(init_dtype)
        final_st: _HSSDCacheState | None = None
        if out_final:
            end_step = step_offset + seq_len
            keep_from = max(swa_base_global, end_step - max_win)
            rel_from = keep_from - swa_base_global
            ring_slots = torch.arange(keep_from, end_step, device=query.device) % max_win
            new_kr = _compact(st_acc.swa_k_ring.index_copy(1, ring_slots, all_ks[:, rel_from:]))
            new_vr = _compact(st_acc.swa_v_ring.index_copy(1, ring_slots, all_vs[:, rel_from:]))
            new_vl = _compact(st_acc.swa_valid_ring.index_copy(1, ring_slots, all_valid[:, rel_from:]))
            term_theta = theta_cum_all[:, -1].clone().contiguous()
            raw_final = _HSSDCacheState(
                step_count=st_acc.step_count + seq_len,
                chunk_size=c_size,
                window_chunks=w_chunks,
                page_states=_compact(p_cur),
                theta_cum=term_theta,
                boundary_ring=ring_cur,
                swa_k_ring=new_kr,
                swa_v_ring=new_vr,
                swa_valid_ring=new_vl,
            )
            if raw_final.dtype == ret_dt and raw_final.has_compact_storage():
                final_st = raw_final
            else:
                final_st = raw_final.to(ret_dt).compact()
        return out, final_st

    def _recurrent_hssd(
        adapter: _StandaloneHSSDAdapter,
        x_in: torch.Tensor,
        wmask: torch.Tensor | None,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *args: Any,
        **kwargs: Any,
    ):
        rec_only = bool(kwargs.pop("_rec_only", False))
        g = args[0] if len(args) >= 1 else kwargs.get("g")
        beta = args[1] if len(args) >= 2 else kwargs.get("beta")
        if not (isinstance(g, torch.Tensor) and isinstance(beta, torch.Tensor)):
            raise RuntimeError("HSSD wrapper expected Tensor `g` and `beta`")
        init_st = kwargs.get("initial_state")
        for ex in args[2:]:
            if init_st is None and (isinstance(ex, torch.Tensor) or hasattr(ex, "page_states")):
                init_st = ex
        out_final = bool(kwargs.get("output_final_state", False))
        use_l2 = bool(kwargs.get("use_qk_l2norm_in_kernel", False))
        scale = kwargs.get("scale")
        pad_mask: torch.Tensor | None = kwargs.get("pad_mask")
        cu_seqlens = kwargs.get("cu_seqlens")
        seq_idx = kwargs.get("seq_idx")

        bsz, seq_len, _ = x_in.shape
        hv, dk, dv = adapter.num_v_heads, adapter.head_k_dim, adapter.head_v_dim
        head_first = bool(kwargs.get("head_first", False)) or (
            seq_len != hv and value.shape[1] == hv and value.shape[2] == seq_len
        )
        if head_first:
            query, key, value = query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2)
            g, beta = g.transpose(1, 2), beta.transpose(1, 2)

        init_dtype = query.dtype
        acc_dt = torch.float64 if init_dtype == torch.float64 else torch.float32
        ret_dt = _resolve_state_dt(adapter.state_dtype, acc_dt)
        c_size, w_chunks, m_pages = adapter.chunk_size, adapter.window_chunks, adapter.num_pages

        if cu_seqlens is not None and seq_idx is not None:
            raise ValueError("Cannot pass both `cu_seqlens` and `seq_idx` to HSSD")

        if cu_seqlens is not None:
            if bsz != 1:
                raise ValueError(f"cu_seqlens requires batch size B == 1, got B={bsz}")
            if init_st is not None:
                raise ValueError("cu_seqlens packed sequences do not support non-None initial_state")
            cu_t = torch.as_tensor(cu_seqlens, dtype=torch.long, device="cpu")
            if (
                cu_t.ndim != 1
                or cu_t.numel() < 2
                or int(cu_t[0].item()) != 0
                or int(cu_t[-1].item()) != seq_len
                or not bool((cu_t[1:] >= cu_t[:-1]).all())
            ):
                raise ValueError(f"Invalid cu_seqlens={cu_t.tolist()} for seq_len={seq_len}")
            seg_outs: list[torch.Tensor] = []
            last_st: _HSSDCacheState | None = None
            n_segs = int(cu_t.numel() - 1)
            for d in range(n_segs):
                s0, s1 = int(cu_t[d].item()), int(cu_t[d + 1].item())
                need_st = out_final and (d == n_segs - 1)
                seg_dt = ret_dt if ((s1 - s0) <= 1 or rec_only) else acc_dt
                z_st = _HSSDCacheState.zeros(
                    1, m_pages, hv, dk, dv, chunk_size=c_size, window_chunks=w_chunks, device=query.device, dtype=seg_dt
                )
                wm_d = None if wmask is None else wmask[:, s0:s1]
                pm_d = None if pad_mask is None else pad_mask[:, s0:s1]
                seg_o, seg_st = _recurrent_hssd(
                    adapter,
                    x_in[:, s0:s1],
                    wm_d,
                    query[:, s0:s1],
                    key[:, s0:s1],
                    value[:, s0:s1],
                    g[:, s0:s1],
                    beta[:, s0:s1],
                    initial_state=z_st,
                    output_final_state=need_st,
                    use_qk_l2norm_in_kernel=use_l2,
                    scale=scale,
                    head_first=False,
                    pad_mask=pm_d,
                    _rec_only=rec_only,
                )
                seg_outs.append(seg_o)
                if need_st:
                    last_st = seg_st
            out_cu = torch.cat(seg_outs, dim=1) if seg_outs else query.new_zeros(1, 0, hv, dv, dtype=init_dtype)
            if head_first:
                out_cu = out_cu.transpose(1, 2)
            return out_cu, (last_st if out_final else None)

        if seq_idx is not None:
            if init_st is not None:
                raise ValueError("seq_idx packed sequences do not support non-None initial_state")
            if not isinstance(seq_idx, torch.Tensor) or seq_idx.shape != (bsz, seq_len):
                raise ValueError(f"seq_idx must be Tensor of shape {(bsz, seq_len)}")
            if seq_len > 1 and not bool((seq_idx[:, 1:] >= seq_idx[:, :-1]).all()):
                raise ValueError("seq_idx must be monotonic non-decreasing per batch row")
            row_outs_s: list[torch.Tensor] = []
            row_sts_s: list[_HSSDCacheState] = []
            for b in range(bsz):
                srow = seq_idx[b]
                b_outs: list[torch.Tensor] = []
                b_last: _HSSDCacheState | None = None
                pos = 0
                while pos < seq_len:
                    end = pos + 1
                    while end < seq_len and int(srow[end].item()) == int(srow[pos].item()):
                        end += 1
                    seg_dt_b = ret_dt if ((end - pos) <= 1 or rec_only) else acc_dt
                    z_b = _HSSDCacheState.zeros(
                        1,
                        m_pages,
                        hv,
                        dk,
                        dv,
                        chunk_size=c_size,
                        window_chunks=w_chunks,
                        device=query.device,
                        dtype=seg_dt_b,
                    )
                    if int(srow[pos].item()) < 0:
                        b_outs.append(query.new_zeros(1, end - pos, hv, dv, dtype=init_dtype))
                        b_last = z_b if z_b.dtype == ret_dt else z_b.to(ret_dt).compact()
                    else:
                        wm_s = None if wmask is None else wmask[b : b + 1, pos:end]
                        pm_s = None if pad_mask is None else pad_mask[b : b + 1, pos:end]
                        o_seg, st_seg = _recurrent_hssd(
                            adapter,
                            x_in[b : b + 1, pos:end],
                            wm_s,
                            query[b : b + 1, pos:end],
                            key[b : b + 1, pos:end],
                            value[b : b + 1, pos:end],
                            g[b : b + 1, pos:end],
                            beta[b : b + 1, pos:end],
                            initial_state=z_b,
                            output_final_state=out_final,
                            use_qk_l2norm_in_kernel=use_l2,
                            scale=scale,
                            head_first=False,
                            pad_mask=pm_s,
                            _rec_only=rec_only,
                        )
                        b_outs.append(o_seg)
                        b_last = st_seg
                    pos = end
                row_outs_s.append(
                    torch.cat(b_outs, dim=1) if b_outs else query.new_zeros(1, 0, hv, dv, dtype=init_dtype)
                )
                if out_final:
                    if b_last is None:
                        b_last = _HSSDCacheState.zeros(
                            1,
                            m_pages,
                            hv,
                            dk,
                            dv,
                            chunk_size=c_size,
                            window_chunks=w_chunks,
                            device=query.device,
                            dtype=ret_dt,
                        )
                    row_sts_s.append(b_last)
            out_s = torch.cat(row_outs_s, dim=0)
            if head_first:
                out_s = out_s.transpose(1, 2)
            return out_s, (torch.cat(row_sts_s, dim=0) if out_final else None)

        work_dt = ret_dt if (seq_len <= 1 or rec_only) else acc_dt
        st = _coerce_hssd_state(
            init_st,
            bsz=bsz,
            num_pages=m_pages,
            hv=hv,
            dk=dk,
            dv=dv,
            c_size=c_size,
            w_chunks=w_chunks,
            device=query.device,
            dtype=work_dt,
        )

        pm_norm: torch.Tensor | None = None
        if pad_mask is not None:
            if pad_mask.ndim != 2 or pad_mask.shape[0] != bsz or pad_mask.shape[1] < seq_len:
                raise ValueError(
                    f"pad_mask must be 2D [B={bsz}, >={seq_len}], got shape {tuple(pad_mask.shape)}"
                )
            pm_norm = pad_mask[:, -seq_len:].to(device=query.device)

        pm_has_zeros = pm_norm is not None and not bool((pm_norm > 0.5).all())
        inhomog_steps = bsz > 1 and not bool((st.step_count == st.step_count[0]).all())

        if pm_has_zeros or inhomog_steps:
            row_outs: list[torch.Tensor] = []
            row_states: list[_HSSDCacheState] = []
            for b in range(bsz):
                st_b = st[b : b + 1]
                if pm_norm is not None:
                    valid_idx = (pm_norm[b] > 0.5).nonzero(as_tuple=False).squeeze(-1)
                else:
                    valid_idx = torch.arange(seq_len, device=query.device, dtype=torch.long)
                if valid_idx.numel() == 0:
                    row_outs.append(query.new_zeros(1, seq_len, hv, dv, dtype=init_dtype))
                    if out_final:
                        if st_b.dtype == ret_dt and st_b.has_compact_storage():
                            row_states.append(st_b)
                        else:
                            row_states.append(st_b.to(ret_dt).compact())
                else:
                    wm_b = None if wmask is None else wmask[b : b + 1].index_select(1, valid_idx)
                    o_val, st_b_out = _run_hssd_homog(
                        adapter,
                        x_in[b : b + 1].index_select(1, valid_idx),
                        wm_b,
                        query[b : b + 1].index_select(1, valid_idx),
                        key[b : b + 1].index_select(1, valid_idx),
                        value[b : b + 1].index_select(1, valid_idx),
                        g[b : b + 1].index_select(1, valid_idx),
                        beta[b : b + 1].index_select(1, valid_idx),
                        st_b,
                        out_final=out_final,
                        use_l2=use_l2,
                        scale=scale,
                        init_dtype=init_dtype,
                        acc_dt=acc_dt,
                        ret_dt=ret_dt,
                        rec_only=rec_only,
                    )
                    if valid_idx.numel() == seq_len:
                        row_outs.append(o_val)
                    else:
                        full_b = query.new_zeros(1, seq_len, hv, dv, dtype=init_dtype)
                        full_b.index_copy_(1, valid_idx, o_val)
                        row_outs.append(full_b)
                    if out_final and st_b_out is not None:
                        row_states.append(st_b_out)
            out_all = torch.cat(row_outs, dim=0)
            if head_first:
                out_all = out_all.transpose(1, 2)
            return out_all, (torch.cat(row_states, dim=0) if out_final else None)

        out, final_st = _run_hssd_homog(
            adapter,
            x_in,
            wmask,
            query,
            key,
            value,
            g,
            beta,
            st,
            out_final=out_final,
            use_l2=use_l2,
            scale=scale,
            init_dtype=init_dtype,
            acc_dt=acc_dt,
            ret_dt=ret_dt,
            rec_only=rec_only,
        )
        if head_first:
            out = out.transpose(1, 2)
        return out, final_st

    # ---------------------------------------------------------------- D-LSA
    def _ward_cluster(x: torch.Tensor, n_clusters: int, max_size: int) -> torch.Tensor:
        nrow, n, _ = x.shape
        dev = x.device
        if n_clusters >= n or max_size < 2:
            return torch.arange(n, device=dev).expand(nrow, n).clone()
        xf = _upcast(x).clone()
        cent = xf.clone()
        size = torch.ones(nrow, n, device=dev, dtype=xf.dtype)
        active = torch.ones(nrow, n, device=dev, dtype=torch.bool)
        labels = torch.arange(n, device=dev).expand(nrow, n).clone()
        inf = torch.finfo(xf.dtype).max
        dist = (0.5 * torch.cdist(xf, xf).pow(2)).masked_fill(torch.eye(n, device=dev, dtype=torch.bool), inf)
        rows = torch.arange(nrow, device=dev)
        for _ in range(n - n_clusters):
            best, idx = dist.view(nrow, -1).min(dim=-1)
            ok = best < inf
            if not bool(ok.any()):
                break
            i, j = idx // n, idx % n
            a, b = torch.minimum(i, j)[ok], torch.maximum(i, j)[ok]
            r = rows[ok]
            na, nb = size[r, a], size[r, b]
            cent[r, a] = (na[:, None] * cent[r, a] + nb[:, None] * cent[r, b]) / (na + nb)[:, None]
            size[r, a], size[r, b], active[r, b] = na + nb, 0, False
            labels[r] = torch.where(labels[r] == b[:, None], a[:, None], labels[r])
            sa, sk = size[r, a][:, None], size[r]
            d_new = (sa * sk / (sa + sk).clamp_min(1e-12)) * (cent[r] - cent[r, a][:, None]).pow(2).sum(-1)
            bad = (~active[r]) | (sa + sk > max_size)
            bad[torch.arange(len(r), device=dev), a] = True
            d_new = d_new.masked_fill(bad, inf)
            dist[r, a, :], dist[r, :, a], dist[r, b, :], dist[r, :, b] = d_new, d_new, inf, inf
        return labels

    def _members_table(labels: torch.Tensor, max_members: int) -> torch.Tensor:
        nrow, n = labels.shape
        dev = labels.device
        ar = torch.arange(n, device=dev)
        order = torch.argsort(labels * n + ar, dim=-1)
        slab = labels.gather(-1, order)
        cnts = torch.zeros(nrow, n, device=dev, dtype=torch.long).scatter_add_(1, labels, torch.ones_like(labels))
        offs = cnts.cumsum(-1) - cnts
        rank = ar.expand(nrow, n) - offs.gather(1, slab)
        out = torch.full((nrow, n, max_members), -1, device=dev, dtype=torch.long)
        rr = torch.arange(nrow, device=dev)[:, None].expand(nrow, n)
        out[rr, slab, rank] = order
        return out

    class _StandalonePassQueryProj(nn.Module):
        def __init__(self, nh: int, din: int, idx_dim: int) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.zeros(nh, din, idx_dim))

        def forward(self, q_sl: torch.Tensor) -> torch.Tensor:
            return torch.einsum("bhtd,hde->bhte", q_sl.to(self.weight.dtype), self.weight)

    class _StandaloneLSAIndexer(nn.Module):
        def __init__(self, nh: int, nkv: int, hd: int, rot: int, lcfg: dict[str, Any]) -> None:
            super().__init__()
            self.h, self.g, self.dh = int(nh), int(nkv), int(hd)
            self.hg = self.h // self.g
            idx_on = str(lcfg.get("index_on", "nope"))
            self.lo = int(rot) if idx_on == "nope" else 0
            din = self.dh - self.lo
            idx_dim = int(lcfg.get("index_dim", 64))
            self.n_q = max(1, int(lcfg.get("num_query_passes", 1)))
            self.mode = str(lcfg.get("mode", "dense"))
            self.search = str(lcfg.get("search", "tree"))
            self.block_size = int(lcfg.get("block_size", 512))
            self.pool = int(lcfg.get("pool", 4))
            self.avg_bucket_keys = int(lcfg.get("avg_bucket_keys", 32))
            self.max_bucket_keys = int(lcfg.get("max_bucket_keys", 64))
            self.num_select = int(lcfg.get("num_select", 32))
            self.sinks = int(lcfg.get("sinks", 16))
            win = lcfg.get("window")
            self.window = self.block_size if win is None else int(win)
            self.query_chunk = int(lcfg.get("query_chunk", 128))
            self.temperature = float(lcfg.get("temperature", 0.25))
            self.tree_group = int(lcfg.get("tree_group", 16))
            self.tree_beam = int(lcfg.get("tree_beam", 4))
            self.wk = nn.Parameter(torch.zeros(self.g, din, idx_dim))
            self.wq = nn.Parameter(torch.zeros(self.h, din, idx_dim))
            self.q_proj_extra = nn.ModuleList(
                [_StandalonePassQueryProj(self.h, din, idx_dim) for _ in range(max(0, self.n_q - 1))]
            )
            self._current_pass = 1
            self._cache: OrderedDict[tuple[Any, ...], Any] = OrderedDict()

        def set_current_pass(self, r: int) -> None:
            self._current_pass = 1 if int(r) <= 1 else min(int(r), self.n_q)

        def query_latent(self, q: torch.Tensor, pass_idx: int | None = None) -> torch.Tensor:
            r = self._current_pass if pass_idx is None else max(1, min(int(pass_idx), self.n_q))
            q_sl = q[..., self.lo :]
            if r <= 1 or self.n_q <= 1:
                z = torch.einsum("bhtd,hde->bhte", q_sl.to(self.wq.dtype), self.wq)
            else:
                z = self.q_proj_extra[r - 2](q_sl)
            z = F.normalize(_upcast(z), dim=-1)
            b, _, t, d = z.shape
            return z.view(b, self.g, self.hg, t, d).sum(2)

        def key_latent(self, k: torch.Tensor) -> torch.Tensor:
            z = torch.einsum("bgtd,gde->bgte", k[..., self.lo :].to(self.wk.dtype), self.wk)
            return F.normalize(_upcast(z), dim=-1)

        def sparse_attend(
            self,
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            scaling: float,
        ) -> torch.Tensor:
            b, h, tq, dh = query.shape
            g, tk = key.shape[1], key.shape[2]
            bs, pool = self.block_size, self.pool
            n = bs // pool
            nb = tk // bs
            dev = query.device
            qpos_all = torch.arange(tk - tq, tk, device=dev)
            qz_all = self.query_latent(query)
            qg = query.view(b, g, h // g, tq, dh)

            cent = active = members = slot_end = None
            if nb > 0:
                zk = self.key_latent(key[:, :, : nb * bs]).view(b, g, nb, bs, -1)
                units = F.normalize(zk.reshape(b, g, nb, n, pool, -1).mean(4), dim=-1)
                labels = _ward_cluster(
                    units.reshape(-1, n, units.shape[-1]),
                    max(1, bs // self.avg_bucket_keys),
                    self.max_bucket_keys // pool,
                ).view(b, g, nb, n)
                umem = _members_table(labels.view(-1, n), self.max_bucket_keys // pool).view(
                    b, g, nb, n, self.max_bucket_keys // pool
                )
                z_u = zk.view(b, g, nb, n, pool, -1).sum(4)
                cent = F.normalize(
                    torch.zeros_like(z_u).scatter_add(3, labels.unsqueeze(-1).expand_as(z_u), z_u), dim=-1
                ).reshape(b, g, nb * n, -1)
                cnts = torch.zeros_like(labels).scatter_add_(3, labels, torch.ones_like(labels))
                active = (cnts > 0).view(b, g, nb * n)
                upos = torch.arange(nb, device=dev).view(1, 1, nb, 1, 1) * bs + umem * pool
                ks = upos.unsqueeze(-1) + torch.arange(pool, device=dev)
                members = torch.where(umem.unsqueeze(-1) >= 0, ks, torch.full_like(ks, -1)).view(b, g, nb * n, -1)
                slot_end = ((torch.arange(nb * n, device=dev) // n) + 1) * bs

            outs = []
            w, ns = self.window, self.sinks
            for c0 in range(0, tq, self.query_chunk):
                sl = slice(c0, min(tq, c0 + self.query_chunk))
                qc, qzc, qpos = qg[:, :, :, sl], qz_all[:, :, sl], qpos_all[sl]
                lo, hi = max(0, int(qpos[0]) - w + 1), int(qpos[-1]) + 1
                kw, vw = key[:, :, lo:hi], value[:, :, lo:hi]
                kpos = torch.arange(lo, hi, device=dev)
                win_lo = qpos - w + 1
                m_win = (kpos[None] <= qpos[:, None]) & (kpos[None] >= win_lo[:, None])
                ns_c = min(ns, int(qpos[-1]) + 1)
                spos = torch.arange(ns_c, device=dev)
                m_sink = (spos[None] < win_lo[:, None]) & (spos[None] <= qpos[:, None])
                s_parts = [torch.einsum("bghcd,bgld->bghcl", qc, kw) * scaling]
                masks = [m_win[None, None, None]]
                if ns_c:
                    s_parts.append(torch.einsum("bghcd,bgld->bghcl", qc, key[:, :, :ns_c]) * scaling)
                    masks.append(m_sink[None, None, None])
                kb = vb = None
                if cent is not None and active is not None and members is not None and slot_end is not None:
                    frontier = (qpos // bs) * bs
                    elig = active[:, :, None, :] & (slot_end[None, None, None, :] <= frontier[None, None, :, None])
                    sc = torch.einsum("bgcd,bgsd->bgcs", qzc, cent.to(qzc.dtype)) / self.temperature
                    if self.search == "tree":
                        tg = self.tree_group
                        ng = -(-nb // tg)
                        if ng > self.tree_beam:
                            bsum = (cent.view(b, g, nb, n, -1) * active.view(b, g, nb, n, 1)).sum(3)
                            if ng * tg > nb:
                                bsum = F.pad(bsum, (0, 0, 0, ng * tg - nb))
                            cs = bsum.view(b, g, ng, tg, -1).cumsum(3)
                            full = F.normalize(cs[:, :, :, -1], dim=-1)
                            fb = qpos // bs
                            jq, rem = fb // tg, fb % tg
                            ss = torch.einsum("bgcd,bgnd->bgcn", qzc, full.to(qzc.dtype))
                            part = F.normalize(cs[:, :, jq.clamp(max=ng - 1), (rem - 1).clamp(min=0)], dim=-1)
                            sp = (qzc * part.to(qzc.dtype)).sum(-1)
                            jn = torch.arange(ng, device=dev)
                            is_p = jn[None, :] == jq[:, None]
                            ss = torch.where(is_p[None, None], sp.unsqueeze(-1), ss)
                            v_sup = (jn[None, :] < jq[:, None]) | (is_p & (rem[:, None] > 0))
                            top_s = ss.masked_fill(~v_sup[None, None], float("-inf")).topk(
                                min(self.tree_beam, ng), dim=-1
                            )
                            ksup = torch.zeros(b, g, qzc.shape[2], ng, dtype=torch.bool, device=dev)
                            ksup.scatter_(-1, top_s.indices, torch.isfinite(top_s.values))
                            keep_s = ksup.repeat_interleave(tg * n, dim=-1)[..., : cent.shape[2]]
                            sc = sc.masked_fill(~keep_s, float("-inf"))
                    sc = sc.masked_fill(~elig, float("-inf"))
                    top = sc.topk(min(self.num_select, sc.shape[-1]), dim=-1)
                    mem = members.unsqueeze(2).expand(-1, -1, qzc.shape[2], -1, -1)
                    mem = mem.gather(3, top.indices.unsqueeze(-1).expand(-1, -1, -1, -1, mem.shape[-1]))
                    mem = torch.where(
                        torch.isfinite(top.values).unsqueeze(-1), mem, torch.full_like(mem, -1)
                    ).flatten(-2)
                    m_b = (mem >= ns_c) & (mem < win_lo[None, None, :, None])
                    kb_n = mem.shape[-1]
                    gidx = mem.clamp_min(0).view(b, g, qzc.shape[2] * kb_n, 1).expand(-1, -1, -1, key.shape[-1])
                    kb = key.gather(2, gidx).view(b, g, qzc.shape[2], kb_n, -1)
                    vb = value.gather(2, gidx).view(b, g, qzc.shape[2], kb_n, -1)
                    s_parts.append(torch.einsum("bghcd,bgckd->bghck", qc, kb) * scaling)
                    masks.append(m_b[:, :, None])
                s_all = torch.cat([_upcast(sp).masked_fill(~mk, float("-inf")) for sp, mk in zip(s_parts, masks)], -1)
                attn = torch.softmax(s_all, dim=-1).to(value.dtype)
                parts = torch.split(attn, [sp.shape[-1] for sp in s_parts], dim=-1)
                o_c = torch.einsum("bghcl,bgld->bghcd", parts[0], vw)
                pi = 1
                if ns_c:
                    o_c = o_c + torch.einsum("bghcl,bgld->bghcd", parts[1], value[:, :, :ns_c])
                    pi = 2
                if kb is not None and vb is not None:
                    o_c = o_c + torch.einsum("bghck,bgckd->bghcd", parts[pi], vb)
                outs.append(o_c)
            return torch.cat(outs, dim=3).reshape(b, h, tq, dh).transpose(1, 2).contiguous()

    # ---------------------------------------------------------------- Loop (ReEntryBridge, PassAttn, PassRouter, MoR)
    class _StandaloneBridge(nn.Module):
        def __init__(
            self,
            hidden: int,
            extra: int,
            init_w: tuple[float, ...],
            rank: int,
            step_emb: bool,
            mode: str = "concat",
        ) -> None:
            super().__init__()
            self.hidden = int(hidden)
            s = sum(init_w)
            logits = torch.tensor([math.log(w / s) for w in init_w])
            self.mix_logits = nn.Parameter(logits.repeat(extra, 1))
            self.step_emb = nn.Parameter(torch.zeros(extra, self.hidden)) if step_emb else None
            self.rank = int(rank)
            self.adapter_mode = mode
            if self.rank > 0:
                if mode == "subspace":
                    self.norms = nn.ModuleList([nn.RMSNorm(self.hidden, eps=1e-6) for _ in range(extra)])
                    self.downs = nn.ModuleList(
                        [nn.Linear(self.hidden, self.rank, bias=False) for _ in range(extra)]
                    )
                    self.ups = nn.ModuleList(
                        [nn.Linear(self.rank, self.hidden, bias=False) for _ in range(extra)]
                    )
                else:
                    self.down = nn.Linear(2 * self.hidden, self.rank, bias=False)
                    self.up = nn.Linear(self.rank, self.hidden, bias=False)

        def forward(self, h: torch.Tensor, anchor: torch.Tensor, p: int) -> torch.Tensor:
            w = F.softmax(_upcast(self.mix_logits[p]), dim=-1)
            hf, af = _upcast(h), _upcast(anchor)
            rh = hf.pow(2).mean(-1, keepdim=True).add(1e-6).sqrt()
            ra = af.pow(2).mean(-1, keepdim=True).add(1e-6).sqrt()
            x = w[0] * hf + w[1] * af + w[2] * (hf * (ra / rh))
            if self.step_emb is not None:
                x = x + _upcast(self.step_emb[p])
            if self.rank > 0:
                if self.adapter_mode == "subspace":
                    nm, dm, um = self.norms[p], self.downs[p], self.ups[p]
                    u = nm(hf.to(nm.weight.dtype))
                    du = um(F.silu(dm(u.to(dm.weight.dtype))).to(um.weight.dtype))
                    x = x + (ra / math.sqrt(self.hidden)) * _upcast(du)
                else:
                    cat = torch.cat([hf, af], dim=-1).to(self.down.weight.dtype)
                    du = self.up(F.silu(self.down(cat)).to(self.up.weight.dtype))
                    x = x + _upcast(du)
            return x.to(h.dtype)

    class _StandalonePassRouter(nn.Module):
        def __init__(self, hidden: int, n_exp: int, core_len: int, extra: int, rank: int) -> None:
            super().__init__()
            self.num_experts = int(n_exp)
            self.rank = min(int(rank), int(hidden), int(n_exp)) if int(rank) > 0 else 0
            if self.rank > 0:
                self.down = nn.Parameter(torch.zeros(extra, core_len, self.rank, hidden))
                self.up = nn.Parameter(torch.zeros(extra, core_len, n_exp, self.rank))
            else:
                self.weight = nn.Parameter(torch.zeros(extra, core_len, n_exp, hidden))
            self.bias = nn.Parameter(torch.zeros(extra, core_len, n_exp))
            self.repel = nn.Parameter(torch.zeros(extra, core_len))

        def is_zero(self, p_idx: int, c_idx: int) -> bool:
            w = self.up[p_idx, c_idx] if self.rank > 0 else self.weight[p_idx, c_idx]
            b = self.bias[p_idx, c_idx]
            r = self.repel[p_idx, c_idx]
            return bool((w == 0).all() and (b == 0).all() and float(r) == 0.0)

        def delta_logits(
            self, flat_x: torch.Tensor, p_idx: int, c_idx: int, prev_mask: torch.Tensor | None
        ) -> torch.Tensor:
            x = _upcast(flat_x).to(self.bias.dtype)
            if self.rank > 0:
                d = F.linear(F.linear(x, self.down[p_idx, c_idx]), self.up[p_idx, c_idx], self.bias[p_idx, c_idx])
            else:
                d = F.linear(x, self.weight[p_idx, c_idx], self.bias[p_idx, c_idx])
            if prev_mask is not None:
                d = d - self.repel[p_idx, c_idx] * prev_mask.to(d.dtype)
            return d

    class _StandalonePassLoRAProj(nn.Module):
        def __init__(self, in_f: int, out_f: int, extra: int, rank: int, scale: float) -> None:
            super().__init__()
            self.scale = float(scale)
            self.down = nn.Parameter(torch.zeros(extra, rank, in_f))
            self.up = nn.Parameter(torch.zeros(extra, out_f, rank))

        def is_zero(self, p_idx: int) -> bool:
            return bool((self.up[p_idx] == 0).all())

        def delta(self, x_in: torch.Tensor, p_idx: int) -> torch.Tensor:
            x = _upcast(x_in).to(self.down.dtype)
            return F.linear(F.linear(x, self.down[p_idx]), self.up[p_idx]) * self.scale

    class _StandalonePassAttn(nn.Module):
        def __init__(
            self,
            proj_specs: dict[int, dict[str, tuple[int, int]]],
            extra: int,
            rank: int,
            alpha: float | None = None,
        ) -> None:
            super().__init__()
            r = int(rank)
            scale = float(alpha if alpha is not None else r) / max(r, 1)
            self.layers = nn.ModuleDict()
            for phys in sorted(proj_specs):
                pdict = nn.ModuleDict()
                for pname, (in_f, out_f) in sorted(proj_specs[phys].items()):
                    pdict[pname] = _StandalonePassLoRAProj(in_f, out_f, int(extra), r, scale)
                self.layers[f"L{phys}"] = pdict

        def is_zero(self, phys: int, pname: str, p_idx: int) -> bool:
            return bool(self.layers[f"L{phys}"][pname].is_zero(p_idx))  # type: ignore[operator]

        def delta(self, x_in: torch.Tensor, phys: int, pname: str, p_idx: int) -> torch.Tensor:
            return self.layers[f"L{phys}"][pname].delta(x_in, p_idx)  # type: ignore[operator]

    class _StandaloneLoopAdapters(nn.Module):
        def __init__(
            self,
            hidden: int,
            spec: dict[str, Any],
            n_exp: int | None,
            attn_proj_specs: dict[int, dict[str, tuple[int, int]]] | None = None,
        ) -> None:
            super().__init__()
            n_passes = int(spec.get("num_passes", 2))
            extra = max(n_passes - 1, 1)
            core_len = int(spec.get("core_end", 32)) - int(spec.get("prelude_end", 8))
            self.gates = nn.Parameter(torch.full((extra, hidden), float(spec.get("gate_init", 0.0))))
            self.bridge = _StandaloneBridge(
                hidden,
                extra,
                tuple(spec.get("bridge_init", (0.6, 0.2, 0.2))),
                int(spec.get("bridge_rank", 0)),
                bool(spec.get("step_embedding", True)),
                str(spec.get("bridge_mode", "concat")),
            )
            self.pass_router = (
                _StandalonePassRouter(hidden, n_exp, core_len, extra, int(spec.get("router_adapter_rank", 32)))
                if (spec.get("per_pass_router", True) and n_passes > 1 and n_exp is not None and core_len > 0)
                else None
            )
            attn_rank = int(spec.get("attn_adapter_rank", 0))
            self.pass_attn = (
                _StandalonePassAttn(attn_proj_specs, extra, attn_rank, spec.get("attn_adapter_alpha"))
                if (attn_rank > 0 and n_passes > 1 and attn_proj_specs)
                else None
            )
            self.mor_router = nn.Linear(hidden, n_passes) if spec.get("mor", False) else None

    def _st_resolve_kv_scale(
        lyr: Any,
        cache: Any,
        layer_idx: int,
        *,
        is_key: bool,
    ) -> torch.Tensor | None:
        attr_single = "k_scale" if is_key else "v_scale"
        attr_plural = "k_scales" if is_key else "v_scales"
        for holder in (lyr, cache):
            if holder is None:
                continue
            sc = getattr(holder, attr_single, None)
            if isinstance(sc, torch.Tensor):
                return sc
            scs = getattr(holder, attr_plural, None)
            if isinstance(scs, (list, tuple, nn.ParameterList)) and 0 <= layer_idx < len(scs):
                cand = scs[layer_idx]
                if isinstance(cand, torch.Tensor):
                    return cand
            elif isinstance(scs, dict) and layer_idx in scs and isinstance(scs[layer_idx], torch.Tensor):
                return scs[layer_idx]
        return None

    def _st_dequant_kv_slot(
        t: torch.Tensor,
        out_dt: torch.dtype,
        lyr: Any,
        cache: Any,
        layer_idx: int,
        *,
        is_key: bool,
    ) -> torch.Tensor:
        if t.dtype in _FP8_DTYPES or not t.is_floating_point():
            sc = _st_resolve_kv_scale(lyr, cache, layer_idx, is_key=is_key)
            if isinstance(sc, torch.Tensor):
                sf = sc.to(device=t.device, dtype=torch.float32)
                if sf.numel() == 1:
                    sf = sf.reshape(())
                elif sf.ndim == 1 and t.ndim == 4 and sf.shape[0] == t.shape[1]:
                    sf = sf.view(1, -1, 1, 1)
                elif sf.ndim < t.ndim:
                    while sf.ndim < t.ndim:
                        sf = sf.unsqueeze(-1)
                return (t.to(torch.float32) * sf).to(out_dt)
            return t.to(out_dt)
        return t if t.dtype == out_dt else t.to(out_dt)

    class _SharedKVProxy:
        def __init__(
            self,
            cache: Any,
            stash: dict[int, tuple[torch.Tensor, torch.Tensor]] | None = None,
            *,
            record: bool = False,
        ) -> None:
            object.__setattr__(self, "_cache", cache)
            object.__setattr__(self, "_stash", stash)
            object.__setattr__(self, "_record", bool(record))

        def __getattr__(self, name: str) -> Any:
            return getattr(object.__getattribute__(self, "_cache"), name)

        def __setattr__(self, name: str, value: Any) -> None:
            setattr(object.__getattribute__(self, "_cache"), name, value)

        def __getitem__(self, idx: Any) -> Any:
            return object.__getattribute__(self, "_cache")[idx]

        def __len__(self) -> int:
            return len(object.__getattribute__(self, "_cache"))

        def __bool__(self) -> bool:
            return True

        def update(self, k: torch.Tensor, v: torch.Tensor, layer_idx: int, *a: Any, **kw: Any) -> Any:
            cache = object.__getattribute__(self, "_cache")
            stash = object.__getattribute__(self, "_stash")
            record = object.__getattribute__(self, "_record")
            lidx = int(layer_idx)
            if record:
                res = cache.update(k, v, layer_idx, *a, **kw)
                if stash is not None and isinstance(res, tuple) and len(res) >= 2:
                    stash[lidx] = (res[0], res[1])
                return res
            if stash is not None and lidx in stash:
                return stash[lidx]
            if type(cache).__name__ in ("QuantizedHybridCache", "_QuantizedKV"):
                return cache.update(k[..., :0, :], v[..., :0, :], layer_idx, *a, **kw)
            inner_cache = getattr(cache, "cache", None)
            if inner_cache is None:
                inner_cache = getattr(cache, "_cache", None)
            for holder in (cache, inner_cache):
                if holder is None:
                    continue
                layers = getattr(holder, "layers", None)
                if isinstance(layers, (list, tuple, nn.ModuleList)) and 0 <= lidx < len(layers):
                    lyr = layers[lidx]
                    deq_fn = getattr(lyr, "_dequantize", None)
                    qk, qv = getattr(lyr, "_quantized_keys", None), getattr(lyr, "_quantized_values", None)
                    if (
                        callable(deq_fn)
                        and isinstance(qk, torch.Tensor)
                        and isinstance(qv, torch.Tensor)
                        and qk.numel() > 0
                    ):
                        return deq_fn(qk).to(k.dtype), deq_fn(qv).to(v.dtype)
                    k_t, v_t = getattr(lyr, "keys", None), getattr(lyr, "values", None)
                    if isinstance(k_t, torch.Tensor) and isinstance(v_t, torch.Tensor) and k_t.numel() > 0:
                        return (
                            _st_dequant_kv_slot(k_t, k.dtype, lyr, cache, lidx, is_key=True),
                            _st_dequant_kv_slot(v_t, v.dtype, lyr, cache, lidx, is_key=False),
                        )
                kc, vc = getattr(holder, "key_cache", None), getattr(holder, "value_cache", None)
                if (
                    isinstance(kc, list)
                    and isinstance(vc, list)
                    and 0 <= lidx < len(kc)
                    and 0 <= lidx < len(vc)
                    and isinstance(kc[lidx], torch.Tensor)
                    and isinstance(vc[lidx], torch.Tensor)
                    and kc[lidx].numel() > 0
                ):
                    return (
                        _st_dequant_kv_slot(kc[lidx], k.dtype, None, cache, lidx, is_key=True),
                        _st_dequant_kv_slot(vc[lidx], v.dtype, None, cache, lidx, is_key=False),
                    )
            return cache.update(k[..., :0, :], v[..., :0, :], layer_idx, *a, **kw)

    class _StandaloneDepthContext:
        def __init__(self, owner: Any, prev: int, target: int) -> None:
            self._owner = owner
            self._prev = int(prev)
            self.target = int(target)

        def __enter__(self) -> Any:
            self._owner.set_active_passes(self.target)
            return self._owner

        def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
            self._owner.set_active_passes(self._prev)

    class _StandaloneCausalLMOutput:
        def __init__(
            self,
            logits: torch.Tensor,
            loss: torch.Tensor | None = None,
            past_key_values: Any = None,
            hidden_states: Any = None,
            attentions: Any = None,
        ) -> None:
            self.logits = logits
            self.loss = loss
            self.past_key_values = past_key_values
            self.hidden_states = hidden_states
            self.attentions = attentions

        def __getitem__(self, key: int | str) -> Any:
            if isinstance(key, str):
                return getattr(self, key)
            items = [x for x in (self.loss, self.logits, self.past_key_values, self.hidden_states) if x is not None]
            return items[key]

    def _find_children(parent: nn.Module, names: list[str] | tuple[str, ...]) -> dict[str, nn.Module]:
        out: dict[str, nn.Module] = {}
        for nm in names:
            cur: Any = parent
            for part in nm.split("."):
                cur = getattr(cur, part, None)
                if cur is None:
                    break
            if isinstance(cur, nn.Module):
                out[nm] = cur
        return out

    def _linear_dims(mod: nn.Module) -> tuple[int, int]:
        inf, outf = getattr(mod, "in_features", None), getattr(mod, "out_features", None)
        if isinstance(inf, int) and isinstance(outf, int) and inf > 0 and outf > 0:
            return inf, outf
        wp = getattr(mod, "weight_packed", None)
        if isinstance(wp, torch.Tensor) and wp.ndim == 2:
            return int(wp.shape[1]) * int(wp.element_size()) * 2, int(wp.shape[0])
        w = getattr(mod, "weight", None)
        if isinstance(w, torch.Tensor) and w.ndim == 2:
            if w.dtype in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64) and isinstance(
                getattr(mod, "weight_scale", None), torch.Tensor
            ):
                return int(w.shape[1]) * int(w.element_size()) * 2, int(w.shape[0])
            return int(w.shape[1]), int(w.shape[0])
        raise RuntimeError(f"cannot determine linear dims on {type(mod).__name__}")

    _FP8_DTYPES: tuple[torch.dtype, ...] = tuple(
        getattr(torch, nm)
        for nm in ("float8_e4m3fn", "float8_e4m3fnuz", "float8_e5m2", "float8_e5m2fnuz")
        if hasattr(torch, nm)
    )
    _PACKED_INT_DTYPES: tuple[torch.dtype, ...] = (
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    )
    _E2M1_TABLE: tuple[float, ...] = (
        0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
        -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
    )
    _ST_LUT_CACHE: dict[torch.device, torch.Tensor] = {}
    _ST_LIN_S_ATTRS: tuple[str, ...] = ("weight_scale_inv", "scale_inv", "weight_scale", "scale", "weight_scales")
    _ST_GU_S_ATTRS: tuple[str, ...] = (
        "gate_up_proj_scale_inv",
        "gate_up_proj_weight_scale_inv",
        "gate_up_proj_scale",
        "gate_up_proj_scales",
        "gate_up_scale_inv",
        "gate_up_scale",
        "weight_scale_inv_up",
        "weight_scale_up",
        "scale_inv_up",
        "scale_up",
        "w13_weight_scale_inv",
        "w13_weight_scale",
    )
    _ST_DN_S_ATTRS: tuple[str, ...] = (
        "down_proj_scale_inv",
        "down_proj_weight_scale_inv",
        "down_proj_scale",
        "down_proj_scales",
        "down_scale_inv",
        "down_scale",
        "weight_scale_inv_down",
        "weight_scale_down",
        "scale_inv_down",
        "scale_down",
        "w2_weight_scale_inv",
        "w2_weight_scale",
    )

    def _st_first_tensor(mod: nn.Module, names: tuple[str, ...]) -> tuple[str | None, torch.Tensor | None]:
        for nm in names:
            v = getattr(mod, nm, None)
            if isinstance(v, torch.Tensor):
                return nm, v
        return None, None

    def _st_infer_bdim(dim_sz: int, s_sz: int, hint: int | None) -> int:
        if s_sz <= 0:
            raise ValueError(f"invalid scale dim {s_sz}")
        if s_sz == dim_sz:
            return 1
        if s_sz == 1:
            return dim_sz
        if hint is not None and hint > 0 and math.ceil(dim_sz / hint) == s_sz:
            return hint
        for c in (128, 64, 32, 16, 256, 512):
            if math.ceil(dim_sz / c) == s_sz:
                return c
        b = math.ceil(dim_sz / s_sz)
        if math.ceil(dim_sz / b) == s_sz:
            return b
        raise ValueError(f"cannot infer FP8 block dim for dim={dim_sz}, scale_dim={s_sz}")

    def _st_bsize_of(mod: nn.Module) -> tuple[int, int] | None:
        for nm in ("weight_block_size", "block_size"):
            v = getattr(mod, nm, None)
            if isinstance(v, (tuple, list)) and len(v) == 2:
                return int(v[0]), int(v[1])
        return None

    def _st_resolve_fp8_scale(
        mod: nn.Module,
        w: torch.Tensor,
        names: tuple[str, ...],
        role: str,
    ) -> torch.Tensor:
        _, s = _st_first_tensor(mod, names)
        if s is None:
            seen: set[str] = set()
            cands: list[tuple[str, torch.Tensor]] = []
            for d in (mod._parameters, mod._buffers, vars(mod)):
                for k, v in d.items():
                    if not k.startswith("_") and k not in seen and isinstance(v, torch.Tensor):
                        seen.add(k)
                        cands.append((k, v))
            bs = _st_bsize_of(mod)
            b0, b1 = (bs[0], bs[1]) if bs is not None else (128, 128)
            neg = ("down", "w2") if role == "gu" else (("gate_up", "up", "w13") if role == "dn" else ())
            pos = ("gate_up", "up", "w13", "gu") if role == "gu" else (("down", "dn", "w2") if role == "dn" else ())
            best_sc = 0
            for k, t in cands:
                kl = k.lower()
                if t is w or "bias" in kl or not t.is_floating_point() or t.dtype in _FP8_DTYPES:
                    continue
                if any(nt in kl for nt in neg):
                    continue
                sc = 0
                if w.ndim == 2:
                    es0, es1 = math.ceil(int(w.shape[0]) / b0), math.ceil(int(w.shape[1]) / b1)
                    if t.ndim == 2 and tuple(t.shape) == (es0, es1):
                        sc = 4 if "scale" in kl else 2
                    elif "scale" in kl:
                        sc = 1
                elif w.ndim == 3:
                    ec, m, n = int(w.shape[0]), int(w.shape[1]), int(w.shape[2])
                    es0, es1 = math.ceil(m / b0), math.ceil(n / b1)
                    rm = any(pt in kl for pt in pos)
                    if (t.ndim == 3 and tuple(t.shape) == (ec, es0, es1)) or (
                        t.ndim == 2 and tuple(t.shape) == (ec * es0, es1)
                    ):
                        sc = 5 if rm else (4 if "scale" in kl else 2)
                    elif rm and "scale" in kl:
                        sc = 2
                if sc > best_sc:
                    best_sc = sc
                    s = t
        if s is None and not getattr(mod, "_qloop_allow_unscaled_fp8", False):
            avail = sorted(k for k, v in mod._parameters.items() if v is not None) + sorted(
                k for k, v in mod._buffers.items() if v is not None
            )
            raise RuntimeError(f"FP8 module {type(mod).__name__} missing scale tensor (available={avail})")
        return s  # type: ignore[return-value]

    def _st_deq_fp8(
        weight: torch.Tensor,
        scale_inv: torch.Tensor | None,
        dtype: torch.dtype,
        block_size: tuple[int, int] | None = None,
    ) -> torch.Tensor:
        w = weight.to(torch.float32)
        if scale_inv is None:
            return w.to(dtype=dtype)
        s = scale_inv.to(device=weight.device, dtype=torch.float32)
        h0 = block_size[0] if block_size is not None else None
        h1 = block_size[1] if block_size is not None else None
        if w.ndim == 2:
            m, k = int(w.shape[0]), int(w.shape[1])
            if s.ndim == 0 or s.numel() == 1:
                res = w * s.reshape(())
            elif s.ndim == 1:
                sn = int(s.shape[0])
                if sn == m:
                    res = w * s[:, None]
                elif sn == k:
                    res = w * s[None, :]
                elif m == 1:
                    bk = _st_infer_bdim(k, sn, h1)
                    if k == sn * bk:
                        res = (w.reshape(1, sn, bk) * s.reshape(1, sn, 1)).reshape(1, k)
                    else:
                        res = w * s.repeat_interleave(bk, dim=0)[:k].unsqueeze(0)
                elif k == 1:
                    bm = _st_infer_bdim(m, sn, h0)
                    if m == sn * bm:
                        res = (w.reshape(sn, bm, 1) * s.reshape(sn, 1, 1)).reshape(m, 1)
                    else:
                        res = w * s.repeat_interleave(bm, dim=0)[:m].unsqueeze(1)
                else:
                    raise ValueError(f"1D FP8 scale {tuple(s.shape)} incompatible with 2D weight {tuple(w.shape)}")
            else:
                s0, s1 = int(s.shape[0]), int(s.shape[1])
                if s0 in (1, m) and s1 in (1, k):
                    res = w * s
                else:
                    b0, b1 = _st_infer_bdim(m, s0, h0), _st_infer_bdim(k, s1, h1)
                    if math.ceil(m / b0) != s0 or math.ceil(k / b1) != s1:
                        raise ValueError(
                            f"2D FP8 scale {tuple(s.shape)} incompatible with weight {tuple(w.shape)}"
                        )
                    if m == s0 * b0 and k == s1 * b1:
                        res = (w.reshape(s0, b0, s1, b1) * s.reshape(s0, 1, s1, 1)).reshape(m, k)
                    else:
                        se = s if b0 == 1 else s.repeat_interleave(b0, dim=0)[:m, :]
                        se = se if b1 == 1 else se.repeat_interleave(b1, dim=1)[:, :k]
                        res = w * se
            return res.to(dtype=dtype)
        ec, m, k = int(w.shape[0]), int(w.shape[1]), int(w.shape[2])
        if s.ndim == 2 and ec > 1 and s.shape not in ((ec, m), (ec, k)) and int(s.shape[0]) % ec == 0:
            s = s.reshape(ec, int(s.shape[0]) // ec, int(s.shape[1]))
        if s.ndim == 0 or s.numel() == 1:
            res = w * s.reshape(())
        elif s.ndim == 1 and s.shape[0] == ec:
            res = w * s.reshape(ec, 1, 1)
        elif s.ndim == 2:
            res = w * (s[:, :, None] if s.shape == (ec, m) else s[:, None, :])
        else:
            es, s0, s1 = int(s.shape[0]), int(s.shape[1]), int(s.shape[2])
            if s0 in (1, m) and s1 in (1, k):
                res = w * s
            else:
                b0, b1 = _st_infer_bdim(m, s0, h0), _st_infer_bdim(k, s1, h1)
                if m == s0 * b0 and k == s1 * b1:
                    res = (w.reshape(ec, s0, b0, s1, b1) * s.reshape(es, s0, 1, s1, 1)).reshape(ec, m, k)
                else:
                    se = s if b0 == 1 else s.repeat_interleave(b0, dim=1)[:, :m, :]
                    se = se if b1 == 1 else se.repeat_interleave(b1, dim=2)[:, :, :k]
                    res = w * se
        return res.to(dtype=dtype)

    def _st_deq_nvfp4(
        wp: torch.Tensor,
        ws: torch.Tensor,
        gs: torch.Tensor | None,
        dtype: torch.dtype,
        gsize: int = 16,
        mul_gs: bool = False,
    ) -> torch.Tensor:
        u8 = wp if wp.dtype == torch.uint8 else wp.contiguous().view(torch.uint8)
        lut = _ST_LUT_CACHE.get(u8.device)
        if lut is None:
            with torch.inference_mode(False), torch.no_grad():
                lut = torch.tensor(_E2M1_TABLE, dtype=torch.float32, device=u8.device)
            _ST_LUT_CACHE[u8.device] = lut
        ku = int(u8.shape[-1]) * 2
        nib = torch.stack((u8 & 0x0F, (u8 >> 4) & 0x0F), dim=-1).reshape(*u8.shape[:-1], ku)
        fv = lut[nib.long()]
        s = ws.to(device=u8.device, dtype=torch.float32)
        ng = int(s.shape[-1]) if s.ndim > 0 else 1
        if s.ndim == 0 or s.numel() == 1:
            deq = fv * s.reshape(())
        elif ng == 1:
            deq = fv * (s.unsqueeze(-1) if s.ndim < fv.ndim else s)
        elif ng * gsize == ku:
            deq = (fv.reshape(*s.shape, gsize) * s.unsqueeze(-1)).reshape(fv.shape)
        else:
            deq = fv * s.repeat_interleave(gsize, dim=-1)[..., :ku]
        if isinstance(gs, torch.Tensor):
            gf = gs.to(device=u8.device, dtype=torch.float32)
            if gf.numel() == 1:
                gf = gf.reshape(())
            elif gf.ndim == 1 and gf.shape[0] == deq.shape[0]:
                gf = gf.reshape(deq.shape[0], *([1] * (deq.ndim - 1)))
            elif deq.ndim == 3 and gf.ndim == 2 and gf.shape[0] == deq.shape[0] and gf.shape[1] in (1, deq.shape[1]):
                gf = gf.unsqueeze(-1)
            deq = deq * gf if mul_gs else deq / gf
        return deq.to(dtype=dtype)

    def _st_quant_kind(mod: nn.Module) -> str | None:
        if isinstance(mod, (nn.Embedding, nn.ModuleList, nn.ModuleDict, nn.Sequential)):
            return None
        gu_p = getattr(mod, "gate_up_proj_packed", None)
        dn_p = getattr(mod, "down_proj_packed", None)
        gu_s = getattr(mod, "gate_up_proj_scale", None)
        if (
            isinstance(gu_p, torch.Tensor)
            and isinstance(dn_p, torch.Tensor)
            and gu_p.ndim == 3
            and gu_p.dtype in _PACKED_INT_DTYPES
            and isinstance(gu_s, torch.Tensor)
        ):
            return "moe_nvfp4"
        gu = getattr(mod, "gate_up_proj", None)
        dn = getattr(mod, "down_proj", None)
        if isinstance(gu, torch.Tensor) and isinstance(dn, torch.Tensor) and gu.ndim == 3 and dn.ndim == 3:
            if gu.dtype in _FP8_DTYPES and dn.dtype in _FP8_DTYPES:
                return "moe_fp8"
            if gu.dtype in _PACKED_INT_DTYPES and isinstance(gu_s, torch.Tensor):
                return "moe_nvfp4"
        wp = getattr(mod, "weight_packed", None)
        ws = getattr(mod, "weight_scale", None)
        if isinstance(wp, torch.Tensor) and wp.ndim == 2 and wp.dtype in _PACKED_INT_DTYPES and isinstance(
            ws, torch.Tensor
        ):
            return "linear_nvfp4"
        w = getattr(mod, "weight", None)
        if isinstance(w, torch.Tensor) and w.ndim == 2:
            if w.dtype in _FP8_DTYPES:
                return "linear_fp8"
            if w.dtype in _PACKED_INT_DTYPES and isinstance(ws, torch.Tensor):
                return "linear_nvfp4"
        return None

    def _st_slice_scale_single(
        s: torch.Tensor | None,
        eid: int,
        n_exp: int,
        m_dim: int | None = None,
    ) -> torch.Tensor | None:
        if s is None or s.ndim == 0 or s.numel() == 1:
            return s
        if s.ndim == 2 and n_exp > 0:
            r, c = int(s.shape[0]), int(s.shape[1])
            if r != n_exp and r % n_exp == 0:
                return s.reshape(n_exp, r // n_exp, c)[eid]
            if r == n_exp and m_dim is not None and c != m_dim:
                return s[eid].unsqueeze(0)
            if r == n_exp:
                return s[eid]
        if s.shape[0] == n_exp and s.ndim in (1, 3):
            return s[eid]
        return s

    def _st_slice_scale_batch(
        s: torch.Tensor | None,
        eids: torch.Tensor,
        n_exp: int,
        m_dim: int | None = None,
    ) -> torch.Tensor | None:
        if s is None or s.ndim == 0 or s.numel() == 1:
            return s
        ei = eids.to(device=s.device, dtype=torch.long)
        if s.ndim == 2 and n_exp > 0:
            r, c = int(s.shape[0]), int(s.shape[1])
            if r != n_exp and r % n_exp == 0:
                return s.reshape(n_exp, r // n_exp, c).index_select(0, ei)
            if r == n_exp and m_dim is not None and c != m_dim:
                return s.reshape(n_exp, 1, c).index_select(0, ei)
            if r == n_exp:
                return s.index_select(0, ei)
        if s.shape[0] == n_exp and s.ndim in (1, 3):
            return s.index_select(0, ei)
        return s

    def _st_nvfp4_mul_gs(mod: nn.Module, gs_nm: str | None, scale2_name: str) -> bool:
        smode = getattr(mod, "_qloop_nvfp4_scale_mode", None)
        if smode in ("divisor", "multiplier"):
            return bool(smode == "multiplier")
        return bool(gs_nm == scale2_name)

    def _st_deq_lin(mod: nn.Module, dt: torch.dtype) -> torch.Tensor:
        kind = _st_quant_kind(mod)
        if kind == "linear_fp8":
            s = _st_resolve_fp8_scale(mod, mod.weight, _ST_LIN_S_ATTRS, "lin")
            return _st_deq_fp8(mod.weight, s, dt, _st_bsize_of(mod))
        wp = getattr(mod, "weight_packed", None)
        if not isinstance(wp, torch.Tensor):
            wp = mod.weight
        gs_nm, gs = _st_first_tensor(mod, ("weight_global_scale", "global_scale", "weight_scale_2"))
        gsize = int(getattr(mod, "group_size", None) or 16)
        return _st_deq_nvfp4(
            wp, mod.weight_scale, gs, dt, gsize, mul_gs=_st_nvfp4_mul_gs(mod, gs_nm, "weight_scale_2")
        )

    def _st_deq_moe_expert(mod: nn.Module, eid: int, dt: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        kind = _st_quant_kind(mod)
        if kind == "moe_fp8":
            gu_a, dn_a = mod.gate_up_proj, mod.down_proj
            n_exp = int(gu_a.shape[0])
            bs = _st_bsize_of(mod)
            gu_s = _st_resolve_fp8_scale(mod, gu_a, _ST_GU_S_ATTRS, "gu")
            dn_s = _st_resolve_fp8_scale(mod, dn_a, _ST_DN_S_ATTRS, "dn")
            guc = _st_deq_fp8(gu_a[eid], _st_slice_scale_single(gu_s, eid, n_exp, int(gu_a.shape[1])), dt, bs)
            dnc = _st_deq_fp8(dn_a[eid], _st_slice_scale_single(dn_s, eid, n_exp, int(dn_a.shape[1])), dt, bs)
        else:
            gu_raw = getattr(mod, "gate_up_proj_packed", None)
            if not isinstance(gu_raw, torch.Tensor):
                gu_raw = mod.gate_up_proj
            dn_raw = getattr(mod, "down_proj_packed", None)
            if not isinstance(dn_raw, torch.Tensor):
                dn_raw = mod.down_proj
            n_exp = int(gu_raw.shape[0])
            gsz = int(getattr(mod, "group_size", None) or 16)
            gs_nm, gu_gs = _st_first_tensor(
                mod, ("gate_up_proj_global_scale", "gate_up_global_scale", "gate_up_proj_scale_2")
            )
            dn_nm, dn_gs = _st_first_tensor(
                mod, ("down_proj_global_scale", "down_global_scale", "down_proj_scale_2")
            )
            gu_s_e = _st_slice_scale_single(mod.gate_up_proj_scale, eid, n_exp, int(gu_raw.shape[1]))
            dn_s_e = _st_slice_scale_single(mod.down_proj_scale, eid, n_exp, int(dn_raw.shape[1]))
            assert gu_s_e is not None and dn_s_e is not None
            guc = _st_deq_nvfp4(
                gu_raw[eid],
                gu_s_e,
                _st_slice_scale_single(gu_gs, eid, n_exp),
                dt,
                gsz,
                mul_gs=_st_nvfp4_mul_gs(mod, gs_nm, "gate_up_proj_scale_2"),
            )
            dnc = _st_deq_nvfp4(
                dn_raw[eid],
                dn_s_e,
                _st_slice_scale_single(dn_gs, eid, n_exp),
                dt,
                gsz,
                mul_gs=_st_nvfp4_mul_gs(mod, dn_nm, "down_proj_scale_2"),
            )
        if guc.shape[0] == dnc.shape[1] and guc.shape[1] == 2 * dnc.shape[0]:
            guc = guc.t().contiguous()
            dnc = dnc.t().contiguous()
        return guc, dnc

    def _st_moe_fallback(mod: nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any]) -> torch.Tensor:
        pos_t = [a for a in args if isinstance(a, torch.Tensor)]
        hidden = pos_t[0] if pos_t else next(
            kwargs[k] for k in ("hidden_states", "x", "input") if isinstance(kwargs.get(k), torch.Tensor)
        )
        rem = pos_t[1:] + [
            v for k, v in kwargs.items() if k not in ("hidden_states", "x", "input") and isinstance(v, torch.Tensor)
        ]
        top_idx = next(
            (kwargs[k] for k in ("top_k_index", "selected_experts", "router_indices") if k in kwargs),
            None,
        )
        top_w = next(
            (kwargs[k] for k in ("top_k_weights", "routing_weights", "router_scores") if k in kwargs),
            None,
        )
        for t in rem:
            if top_idx is None and not t.is_floating_point() and not t.is_complex():
                top_idx = t
            elif top_w is None and t.is_floating_point():
                top_w = t
        assert isinstance(top_idx, torch.Tensor) and isinstance(top_w, torch.Tensor)
        orig_sh = hidden.shape
        fh = hidden.reshape(-1, orig_sh[-1])
        if fh.numel() == 0 or top_idx.numel() == 0:
            return torch.zeros_like(hidden)
        idx2 = top_idx.reshape(fh.shape[0], -1).long()
        w2 = top_w.reshape(fh.shape[0], -1)
        ks = int(idx2.shape[1])
        kind = _st_quant_kind(mod)
        gu_raw = getattr(mod, "gate_up_proj_packed", None)
        if not isinstance(gu_raw, torch.Tensor):
            gu_raw = mod.gate_up_proj
        n_exp = int(gu_raw.shape[0])
        fidx = idx2.reshape(-1)
        safe = torch.where((fidx >= 0) & (fidx < n_exp), fidx, fidx.new_full((), n_exp))
        se, perm = safe.sort(dim=0, stable=True)
        cnts = torch.bincount(se, minlength=n_exp + 1).tolist()
        spans: list[tuple[int, int, int]] = []
        off = 0
        for eid in range(n_exp):
            c = int(cnts[eid])
            if c > 0:
                spans.append((eid, off, c))
                off += c
        out = torch.zeros_like(fh)
        if not spans:
            return out.reshape(orig_sh)
        pv = perm[:off]
        tv = torch.div(pv, ks, rounding_mode="floor")
        sv = pv % ks
        xs = fh.index_select(0, tv)
        ws_tok = w2[tv, sv]
        act = getattr(mod, "act_fn", None) or F.silu
        bs = _st_bsize_of(mod)
        for c0 in range(0, len(spans), 32):
            csp = spans[c0 : c0 + 32]
            eids_t = torch.tensor([e for e, _, _ in csp], device=gu_raw.device, dtype=torch.long)
            if kind == "moe_fp8":
                gu_a, dn_a = mod.gate_up_proj, mod.down_proj
                gu_s = _st_resolve_fp8_scale(mod, gu_a, _ST_GU_S_ATTRS, "gu")
                dn_s = _st_resolve_fp8_scale(mod, dn_a, _ST_DN_S_ATTRS, "dn")
                guc = _st_deq_fp8(
                    gu_a.index_select(0, eids_t),
                    _st_slice_scale_batch(gu_s, eids_t, n_exp, int(gu_a.shape[1])),
                    fh.dtype,
                    bs,
                )
                dnc = _st_deq_fp8(
                    dn_a.index_select(0, eids_t),
                    _st_slice_scale_batch(dn_s, eids_t, n_exp, int(dn_a.shape[1])),
                    fh.dtype,
                    bs,
                )
            else:
                dn_raw = getattr(mod, "down_proj_packed", None)
                if not isinstance(dn_raw, torch.Tensor):
                    dn_raw = mod.down_proj
                gs_nm, gu_gs = _st_first_tensor(
                    mod, ("gate_up_proj_global_scale", "gate_up_global_scale", "gate_up_proj_scale_2")
                )
                dn_nm, dn_gs = _st_first_tensor(
                    mod, ("down_proj_global_scale", "down_global_scale", "down_proj_scale_2")
                )
                gsz = int(getattr(mod, "group_size", None) or 16)
                gu_s_b = _st_slice_scale_batch(mod.gate_up_proj_scale, eids_t, n_exp, int(gu_raw.shape[1]))
                dn_s_b = _st_slice_scale_batch(mod.down_proj_scale, eids_t, n_exp, int(dn_raw.shape[1]))
                assert gu_s_b is not None and dn_s_b is not None
                guc = _st_deq_nvfp4(
                    gu_raw.index_select(0, eids_t),
                    gu_s_b,
                    _st_slice_scale_batch(gu_gs, eids_t, n_exp),
                    fh.dtype,
                    gsz,
                    mul_gs=_st_nvfp4_mul_gs(mod, gs_nm, "gate_up_proj_scale_2"),
                )
                dnc = _st_deq_nvfp4(
                    dn_raw.index_select(0, eids_t),
                    dn_s_b,
                    _st_slice_scale_batch(dn_gs, eids_t, n_exp),
                    fh.dtype,
                    gsz,
                    mul_gs=_st_nvfp4_mul_gs(mod, dn_nm, "down_proj_scale_2"),
                )
            if guc.shape[1] == dnc.shape[2] and guc.shape[2] == 2 * dnc.shape[1]:
                guc = guc.transpose(1, 2).contiguous()
                dnc = dnc.transpose(1, 2).contiguous()
            for li, (_eid, o0, cn) in enumerate(csp):
                xe = xs[o0 : o0 + cn]
                we = ws_tok[o0 : o0 + cn]
                gue = guc[li].to(device=xe.device, dtype=xe.dtype)
                dne = dnc[li].to(device=xe.device, dtype=xe.dtype)
                gt, up = F.linear(xe, gue).chunk(2, dim=-1)
                ye = F.linear(act(gt) * up, dne)
                out.index_add_(0, tv[o0 : o0 + cn], ye * we.unsqueeze(-1).to(ye.dtype))
            del guc, dnc
        return out.reshape(orig_sh)

    def _st_install_quant_fallback(root: nn.Module) -> None:
        seen: set[int] = set()
        for mod in root.modules():
            if id(mod) in seen:
                continue
            seen.add(id(mod))
            kind = _st_quant_kind(mod)
            if kind is None:
                continue
            if kind in ("linear_fp8", "linear_nvfp4"):
                inf, outf = _linear_dims(mod)
                if not isinstance(getattr(mod, "in_features", None), int):
                    mod.in_features = inf  # type: ignore[attr-defined]
                if not isinstance(getattr(mod, "out_features", None), int):
                    mod.out_features = outf  # type: ignore[attr-defined]
            if getattr(mod, "_qloop_standalone_qfb", False):
                continue
            had_inst_fwd = "forward" in mod.__dict__
            orig_inst_fwd = mod.__dict__.get("forward")
            orig_fwd = mod.forward
            raw_fn = getattr(orig_fwd, "__func__", orig_fwd)
            mod._qloop_st_had_inst_fwd = had_inst_fwd  # type: ignore[attr-defined]
            mod._qloop_st_orig_inst_fwd = orig_inst_fwd  # type: ignore[attr-defined]
            if kind in ("linear_fp8", "linear_nvfp4"):
                if kind == "linear_nvfp4" and not isinstance(getattr(mod, "weight", None), torch.Tensor):
                    mod._qloop_fb_only = True  # type: ignore[attr-defined]

                def _make_qlin(ofn: Callable[..., Any] = orig_fwd, rfn: Callable[..., Any] = raw_fn):
                    @functools.wraps(rfn)
                    def _qlin(_self: nn.Module, *a: Any, **kw: Any) -> Any:
                        cur_kind = _st_quant_kind(_self)
                        if cur_kind is None:
                            x0 = a[0] if a else next(v for v in kw.values() if isinstance(v, torch.Tensor))
                            w0 = _self.weight.to(device=x0.device, dtype=x0.dtype)
                            b0 = getattr(_self, "bias", None)
                            bc0 = b0.to(device=x0.device, dtype=x0.dtype) if isinstance(b0, torch.Tensor) else None
                            return F.linear(x0, w0, bc0)
                        if not getattr(_self, "_qloop_fb_only", False):
                            try:
                                return ofn(*a, **kw)
                            except (ImportError, ModuleNotFoundError, NotImplementedError, AttributeError):
                                _self._qloop_fb_only = True  # type: ignore[attr-defined]
                        x = a[0] if a else next(v for v in kw.values() if isinstance(v, torch.Tensor))
                        wd = _st_deq_lin(_self, x.dtype)
                        b = getattr(_self, "bias", None)
                        bc = b.to(device=x.device, dtype=x.dtype) if isinstance(b, torch.Tensor) else None
                        return F.linear(x, wd, bc)

                    return _qlin

                mod.forward = types.MethodType(_make_qlin(), mod)  # type: ignore[method-assign]
            else:
                if kind == "moe_nvfp4" and not isinstance(getattr(mod, "gate_up_proj", None), torch.Tensor):
                    mod._qloop_fb_only = True  # type: ignore[attr-defined]

                def _make_qmoe(ofn: Callable[..., Any] = orig_fwd, rfn: Callable[..., Any] = raw_fn):
                    @functools.wraps(rfn)
                    def _qmoe(_self: nn.Module, *a: Any, **kw: Any) -> Any:
                        if _st_quant_kind(_self) is None:
                            return ofn(*a, **kw)
                        if not getattr(_self, "_qloop_fb_only", False):
                            try:
                                return ofn(*a, **kw)
                            except (ImportError, ModuleNotFoundError, NotImplementedError, AttributeError):
                                _self._qloop_fb_only = True  # type: ignore[attr-defined]
                        return _st_moe_fallback(_self, a, kw)

                    return _qmoe

                mod.forward = types.MethodType(_make_qmoe(), mod)  # type: ignore[method-assign]
            mod._qloop_standalone_qfb = True  # type: ignore[attr-defined]

    def _st_uninstall_quant_fallback(root: nn.Module) -> None:
        seen: set[int] = set()
        for mod in root.modules():
            if id(mod) in seen:
                continue
            seen.add(id(mod))
            if not getattr(mod, "_qloop_standalone_qfb", False):
                continue
            had_inst = bool(getattr(mod, "_qloop_st_had_inst_fwd", False))
            orig_inst = getattr(mod, "_qloop_st_orig_inst_fwd", None)
            if had_inst:
                mod.forward = orig_inst  # type: ignore[method-assign]
            else:
                mod.__dict__.pop("forward", None)
            for flag in (
                "_qloop_standalone_qfb",
                "_qloop_st_had_inst_fwd",
                "_qloop_st_orig_inst_fwd",
                "_qloop_fb_only",
            ):
                mod.__dict__.pop(flag, None)

    def _router_of(layer: nn.Module) -> nn.Module | None:
        mlp = getattr(layer, "mlp", None)
        if mlp is None:
            return None
        for attr in ("gate", "router"):
            r = getattr(mlp, attr, None)
            if isinstance(r, nn.Module) and not hasattr(r, "gate_proj"):
                return r
        return None

    def _looks_like_probs(t: torch.Tensor) -> bool:
        with torch.no_grad():
            s = t.detach().float().reshape(-1, t.shape[-1])[:16]
            return bool((s >= -1e-5).all() and (s <= 1.0 + 1e-5).all() and (s.sum(-1) - 1.0).abs().max() < 1e-2)

    class _StandaloneVirtualLayer(nn.Module):
        def __init__(
            self,
            owner: Qwen36LoopHSSDForCausalLM,
            vidx: int,
            phys: int,
            r: int,
            entry: bool,
            exit_flag: bool,
            cache_idx: int,
            ltype: str,
            inner: nn.Module,
        ) -> None:
            super().__init__()
            self.inner = inner
            self.layer_type = ltype
            self.attention_type = ltype
            object.__setattr__(self, "_owner", owner)
            object.__setattr__(self, "_meta", (vidx, phys, r, entry, exit_flag, cache_idx))
            idx_mods = [m for m in inner.modules() if isinstance(getattr(m, "layer_idx", None), int)]
            object.__setattr__(self, "_idx_mods", idx_mods)

        def forward(self, hidden_states: torch.Tensor, *args: Any, **kwargs: Any) -> Any:
            ow = self._owner
            vidx, phys, r, entry, exit_flag, cidx = self._meta
            if vidx == 0:
                ow._loop_ctx = {"anchor": None, "start": {}, "kv": {}, "pkv": {}, "routing": {}, "keep": {}}
            if r > ow.active_passes:
                return (hidden_states, *([None] * ow._tuple_extra)) if ow._tuple_out else hidden_states
            ctx = ow._loop_ctx
            x = hidden_states
            if entry:
                if r == 1:
                    ctx["anchor"] = x
                    ctx["start"][1] = x
                    if ow._loop_spec.get("mor", False) and ow._loop_adapters is not None:
                        mr = ow._loop_adapters.mor_router
                        if mr is not None:
                            probs = F.softmax(mr(_upcast(x).to(mr.weight.dtype)), dim=-1)
                            depth = probs.argmax(-1) + 1
                            n_p = int(ow._loop_spec.get("num_passes", 2))
                            for pr in range(2, n_p + 1):
                                ctx["keep"][pr] = (depth >= pr).float()
                else:
                    ctx["start"][r] = x
                    assert ow._loop_adapters is not None
                    x = ow._loop_adapters.bridge(x, ctx["anchor"], r - 2)
            pkv_k = next((k for k in ("past_key_values", "past_key_value", "cache") if k in kwargs), None)
            pkv = kwargs.get(pkv_k) if pkv_k is not None else None
            if vidx == 0 and pkv is not None:
                ow._sync_cache_last_linear(pkv)
            if pkv is not None and pkv_k is not None:
                if cidx != vidx:
                    pkv_layers = getattr(pkv, "layers", None)
                    if isinstance(pkv_layers, (list, tuple, nn.ModuleList)) and 0 <= vidx < len(pkv_layers):
                        v_lyr = pkv_layers[vidx]
                        if hasattr(v_lyr, "supports_early_init") and getattr(v_lyr, "keys", None) is None:
                            v_lyr.supports_early_init = False
                    kwargs = dict(kwargs)
                    kwargs[pkv_k] = _SharedKVProxy(pkv, stash=ctx["pkv"], record=False)
                elif (
                    bool(ow._loop_spec.get("kv_share", True))
                    and int(ow._loop_spec.get("num_passes", 2)) > 1
                    and r == 1
                    and self.layer_type == "full_attention"
                    and int(ow._loop_spec.get("prelude_end", 8)) <= phys < int(ow._loop_spec.get("core_end", 32))
                ):
                    kwargs = dict(kwargs)
                    kwargs[pkv_k] = _SharedKVProxy(pkv, stash=ctx["pkv"], record=True)
            prev = ow._cur_pass
            ow._cur_pass = r
            for m in self._idx_mods:
                m.layer_idx = cidx
            attn_mod = getattr(self.inner, "self_attn", None)
            lsa_ix = getattr(attn_mod, "lsa_indexer", None) if attn_mod is not None else None
            set_pass = getattr(lsa_ix, "set_current_pass", None) if lsa_ix is not None else None
            if callable(set_pass):
                set_pass(max(r, 1))
            try:
                out = self.inner(x, *args, **kwargs)
            finally:
                if callable(set_pass):
                    set_pass(1)
                ow._cur_pass = prev
            is_tup = isinstance(out, tuple)
            ow._tuple_out = is_tup
            ow._tuple_extra = (len(out) - 1) if is_tup else 0
            h = out[0] if is_tup else out
            if exit_flag and r >= 2 and ow._loop_adapters is not None:
                st = ctx["start"][r]
                coef = ow._loop_adapters.gates[r - 2].to(h.dtype)
                if r in ctx["keep"]:
                    coef = coef * ctx["keep"][r].to(h.dtype).unsqueeze(-1)
                h = st + coef * (h - st)
            return (h, *out[1:]) if is_tup else h

    class Qwen36LoopHSSDForCausalLM(_BaseCls):  # type: ignore[no-redef,misc]
        config_class = Qwen36LoopHSSDConfig

        def __init__(
            self,
            config: Qwen36LoopHSSDConfig,
            *,
            _defer_loop: bool = False,
            _dtype_override: torch.dtype | None = None,
        ) -> None:
            super().__init__(config)
            self.config = config
            arch = config.qloop_v3_arch or {}
            bcfg = copy.deepcopy(dict(config.base_config or {}))
            base_dt = _dtype_override or _resolve_dt(arch.get("base_dtype", "float32"))
            mtype = str(arch.get("base_model_type") or bcfg.pop("model_type", "qwen3_5_moe"))
            bcfg.pop("model_type", None)
            attn_impl = str(bcfg.pop("_attn_implementation", "sdpa") or "sdpa")
            if attn_impl == "qloop_lsa":
                attn_impl = "sdpa"
            hf_cfg = AutoConfig.for_model(mtype, **bcfg)
            hf_cfg._attn_implementation = attn_impl
            if arch.get("yarn_target_len"):
                orig_l = getattr(hf_cfg, "max_position_embeddings", None) or 262144
                fac = max(1.0, float(arch["yarn_target_len"]) / float(orig_l))
                rope = {"rope_type": "yarn", "factor": fac, "original_max_position_embeddings": int(orig_l)}
                rp = getattr(hf_cfg, "rope_parameters", None)
                if isinstance(rp, dict):
                    rp = dict(rp)
                    rp.update(rope)
                    hf_cfg.rope_parameters = rp
                hf_cfg.rope_scaling = dict(rope)
                hf_cfg.max_position_embeddings = int(max(int(arch["yarn_target_len"]), int(orig_l)))
            try:
                self.backbone = AutoModelForCausalLM.from_config(
                    hf_cfg, attn_implementation=attn_impl, torch_dtype=base_dt
                )
            except TypeError:
                self.backbone = AutoModelForCausalLM.from_config(hf_cfg, attn_implementation=attn_impl)
                self.backbone.to(dtype=base_dt)
            _st_install_quant_fallback(self.backbone)
            self._cur_pass = 0
            self._tuple_out = False
            self._tuple_extra = 0
            self._loop_ctx: dict[str, Any] = {}
            self._loop_spec: dict[str, Any] = {}
            self._loop_adapters: _StandaloneLoopAdapters | None = None
            self._is_probs_cache: dict[int, bool] = {}
            self._amask_stack: list[torch.Tensor] = []
            self._slot_meta: list[tuple[int, int, str]] = []
            self.active_passes = 1
            self._install_submodules(arch, hf_cfg)
            if not _defer_loop:
                self._apply_standalone_loop(arch)

        def _sync_cache_last_linear(self, pkv: Any) -> None:
            if pkv is None or not isinstance(getattr(pkv, "last_linear_layer", None), int):
                return
            n_virt = len(self._slot_meta)
            if n_virt > 0:
                for attr in ("conv_states", "recurrent_states", "key_cache", "value_cache"):
                    slots = getattr(pkv, attr, None)
                    if isinstance(slots, list) and len(slots) < n_virt:
                        slots.extend([None] * (n_virt - len(slots)))
            for vidx, r_slot, lt in reversed(self._slot_meta):
                if lt == "linear_attention" and r_slot <= self.active_passes:
                    if pkv.last_linear_layer != vidx:
                        pkv.last_linear_layer = vidx
                    return

        def _find_tm(self) -> nn.Module:
            for _n, mod in self.backbone.named_modules():
                if isinstance(getattr(mod, "layers", None), nn.ModuleList) and hasattr(mod, "embed_tokens"):
                    return mod
            raise RuntimeError("could not locate text_model on backbone")

        def _active_mor_wmask(self, bsz: int, seq_len: int) -> torch.Tensor | None:
            r = self._cur_pass
            if r < 2 or not (self._loop_spec.get("mor", False) and self._loop_spec.get("write_mask", True)):
                return None
            keep = self._loop_ctx.get("keep", {}).get(r)
            if isinstance(keep, torch.Tensor) and keep.shape == (bsz, seq_len):
                return keep
            return None

        def _install_submodules(self, arch: dict[str, Any], hf_cfg: Any) -> None:
            tm = self._find_tm()
            ref_dt = torch.float64 if tm.embed_tokens.weight.dtype == torch.float64 else torch.float32
            chunk_attrs = ("chunk_gated_delta_rule", "torch_chunk_gated_delta_rule")
            rec_attrs = (
                "recurrent_gated_delta_rule",
                "fused_recurrent_gated_delta_rule",
                "torch_recurrent_gated_delta_rule",
            )
            rule_specs = [(a, False) for a in chunk_attrs] + [(a, True) for a in rec_attrs]

            gcfg = dict(arch.get("gdn2") or {})
            if gcfg.get("enabled"):
                rank = int(gcfg.get("rank", 64))
                cap_g = float(gcfg.get("gate_logit_cap", 8.0))
                cap_d = float(gcfg.get("decay_log_cap", 1.0))
                patched_gmods: set[tuple[str, str]] = set()
                active_gdn2: list[
                    tuple[_StandaloneGDN2Adapter, list[torch.Tensor], list[int], bool, bool]
                ] = []

                def _run_gdn2_rule(
                    ad_r: _StandaloneGDN2Adapter,
                    x_in: torch.Tensor,
                    q: torch.Tensor,
                    k: torch.Tensor,
                    v: torch.Tensor,
                    *r_args: Any,
                    **r_kw: Any,
                ):
                    g = r_args[0] if len(r_args) >= 1 else r_kw["g"]
                    beta = r_args[1] if len(r_args) >= 2 else r_kw["beta"]
                    init_st = r_kw.get("initial_state")
                    for ex in r_args[2:]:
                        if isinstance(ex, torch.Tensor) and ex.ndim == 4 and init_st is None:
                            init_st = ex
                    out_final = bool(r_kw.get("output_final_state", False))
                    use_l2 = bool(r_kw.get("use_qk_l2norm_in_kernel", False))
                    scale = r_kw.get("scale")
                    seq_len = x_in.shape[1]
                    hfirst = bool(r_kw.get("head_first", False)) or (
                        seq_len != ad_r.hv and v.shape[1] == ad_r.hv and v.shape[2] == seq_len
                    )
                    if hfirst:
                        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
                        g, beta = g.transpose(1, 2), beta.transpose(1, 2)
                    gv, bg, wg = ad_r.gates(x_in, g, beta)
                    out, fst = _recurrent_gdn2(q, k, v, gv, bg, wg, init_st, out_final, use_l2, scale)
                    if hfirst:
                        out = out.transpose(1, 2)
                    return out, fst

                for i, layer in enumerate(tm.layers):
                    attn = getattr(layer, "linear_attn", None)
                    if attn is None or (gcfg.get("layers") is not None and i not in gcfg["layers"]):
                        continue
                    hs = int(attn.out_proj.out_features)
                    hv = int(getattr(attn, "num_v_heads", None) or attn.A_log.numel())
                    dv = int(getattr(attn, "head_v_dim", None) or (attn.out_proj.in_features // hv))
                    dk = int(getattr(attn, "head_k_dim", None) or dv)
                    ad = _StandaloneGDN2Adapter(hs, hv, dk, dv, rank, cap_g, cap_d).to(dtype=ref_dt)
                    attn.gdn2_adapter = ad
                    x_box: list[torch.Tensor] = []
                    in_g_box: list[int] = [0]
                    if "_kernel_repo" in attn.__dict__ and "forward" in attn.__dict__:
                        attn.__dict__.pop("forward", None)
                        attn.__dict__.pop("_kernel_repo", None)
                    g_wc = False
                    g_wr = False
                    for attr, rec_only in rule_specs:
                        orig_attr_fn = getattr(attn, attr, None)
                        if callable(orig_attr_fn):
                            if rec_only:
                                g_wr = True
                            else:
                                g_wc = True

                            def _make_grule(
                                ad_r: _StandaloneGDN2Adapter = ad,
                                xb: list[torch.Tensor] = x_box,
                                ib: list[int] = in_g_box,
                                ofn: Callable[..., Any] = orig_attr_fn,
                            ):
                                def _grule(q, k, v, *ra, **rkw):
                                    if ib[0] > 0:
                                        return ofn(q, k, v, *ra, **rkw)
                                    ib[0] += 1
                                    try:
                                        return _run_gdn2_rule(ad_r, xb[-1], q, k, v, *ra, **rkw)
                                    finally:
                                        ib[0] -= 1

                                return _grule

                            gr_fn = _make_grule()
                            if (
                                isinstance(orig_attr_fn, types.MethodType)
                                and getattr(orig_attr_fn, "__self__", None) is attn
                            ):
                                raw_gfn = getattr(orig_attr_fn, "__func__", orig_attr_fn)

                                @functools.wraps(raw_gfn)
                                def _bound_grule(_s: nn.Module, *ga: Any, _gw: Callable[..., Any] = gr_fn, **gkw: Any):
                                    return _gw(*ga, **gkw)

                                setattr(attn, attr, types.MethodType(_bound_grule, attn))
                            else:
                                setattr(attn, attr, gr_fn)

                    def _make_gpre(
                        ad_r: _StandaloneGDN2Adapter = ad,
                        xb: list[torch.Tensor] = x_box,
                        ib: list[int] = in_g_box,
                        wc_f: bool = g_wc,
                        wr_f: bool = g_wr,
                    ):
                        def _gpre(_m: nn.Module, a: tuple[Any, ...], kw: dict[str, Any]) -> None:
                            x_in = a[0] if a else (kw["hidden_states"] if "hidden_states" in kw else kw["x"])
                            amask = kw.get("attention_mask", a[2] if len(a) >= 3 else None)
                            if (
                                isinstance(amask, torch.Tensor)
                                and amask.ndim == 2
                                and amask.shape[0] == x_in.shape[0]
                                and amask.shape[1] >= x_in.shape[1]
                                and x_in.shape[1] > 1
                            ):
                                x_in = x_in * amask[:, -x_in.shape[1] :, None].to(x_in.dtype)
                            xb.append(x_in)
                            active_gdn2.append((ad_r, xb, ib, wc_f, wr_f))

                        return _gpre

                    def _make_gpost(xb: list[torch.Tensor] = x_box):
                        def _gpost(_m: nn.Module, _i: Any, _o: Any) -> None:
                            if xb:
                                xb.pop()
                            if active_gdn2:
                                active_gdn2.pop()

                        return _gpost

                    attn.register_forward_pre_hook(_make_gpre(), with_kwargs=True)
                    attn.register_forward_hook(_make_gpost())
                    if not (g_wc and g_wr):
                        mname = type(attn).__module__
                        hmod = sys.modules.get(mname)
                        if hmod is not None:
                            miss_g: list[tuple[str, bool]] = []
                            if not g_wc:
                                miss_g.extend((nm, False) for nm in chunk_attrs)
                            if not g_wr:
                                miss_g.extend((nm, True) for nm in rec_attrs)
                            for fn_nm, rec_only in miss_g:
                                if (mname, fn_nm) in patched_gmods:
                                    continue
                                orig_fn = getattr(hmod, fn_nm, None)
                                if callable(orig_fn):
                                    patched_gmods.add((mname, fn_nm))

                                    def _gmdisp(
                                        q,
                                        k,
                                        v,
                                        *ra,
                                        _orig: Callable[..., Any] = orig_fn,
                                        _rec: bool = rec_only,
                                        **rkw,
                                    ):
                                        if not active_gdn2:
                                            return _orig(q, k, v, *ra, **rkw)
                                        cad, cxb, cir, cwc, cwr = active_gdn2[-1]
                                        if cir[0] > 0 or (_rec and cwr) or ((not _rec) and cwc):
                                            return _orig(q, k, v, *ra, **rkw)
                                        cir[0] += 1
                                        try:
                                            return _run_gdn2_rule(cad, cxb[-1], q, k, v, *ra, **rkw)
                                        finally:
                                            cir[0] -= 1

                                    setattr(hmod, fn_nm, _gmdisp)

            hcfg = dict(arch.get("hssd") or {})
            if hcfg.get("enabled"):
                _ensure_dynamo_mark_static_address_compat()
                for host_m in (self.backbone, tm):
                    pushed_box: list[bool] = []

                    def _make_ampre(pb: list[bool] = pushed_box):
                        def _ampre(_m: nn.Module, a: tuple[Any, ...], kw: dict[str, Any]) -> None:
                            cand = kw.get("attention_mask", a[1] if len(a) >= 2 else None)
                            if isinstance(cand, torch.Tensor) and cand.ndim == 2:
                                self._amask_stack.append(cand)
                                pb.append(True)
                            else:
                                pb.append(False)

                        return _ampre

                    def _make_ampost(pb: list[bool] = pushed_box):
                        def _ampost(_m: nn.Module, _i: Any, _o: Any) -> None:
                            if pb and pb.pop() and self._amask_stack:
                                self._amask_stack.pop()

                        return _ampost

                    host_m.register_forward_pre_hook(_make_ampre(), with_kwargs=True)
                    host_m.register_forward_hook(_make_ampost())

                patched_mods: set[tuple[str, str]] = set()
                active_hssd: list[
                    tuple[
                        _StandaloneHSSDAdapter,
                        list[tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]],
                        list[int],
                        bool,
                        bool,
                    ]
                ] = []
                for i, layer in enumerate(tm.layers):
                    attn = getattr(layer, "linear_attn", None)
                    if attn is None or (hcfg.get("layers") is not None and i not in hcfg["layers"]):
                        continue
                    hs = int(attn.out_proj.out_features)
                    hv = int(getattr(attn, "num_v_heads", None) or attn.A_log.numel())
                    dv = int(getattr(attn, "head_v_dim", None) or (attn.out_proj.in_features // hv))
                    dk = int(getattr(attn, "head_k_dim", None) or dv)
                    had = _StandaloneHSSDAdapter(hs, hv, dk, dv, hcfg, i).to(dtype=ref_dt)
                    attn.hssd_adapter = had
                    st_box: list[tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]] = []
                    in_h_box: list[int] = [0]
                    if "_kernel_repo" in attn.__dict__ and "forward" in attn.__dict__:
                        attn.__dict__.pop("forward", None)
                        attn.__dict__.pop("_kernel_repo", None)
                    h_wc = False
                    h_wr = False
                    for attr, rec_only in rule_specs:
                        orig_h_fn = getattr(attn, attr, None)
                        if callable(orig_h_fn):
                            if rec_only:
                                h_wr = True
                            else:
                                h_wc = True

                            def _make_hrule(
                                ad_r: _StandaloneHSSDAdapter = had,
                                sb_r: list[tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]] = st_box,
                                ib: list[int] = in_h_box,
                                ofn: Callable[..., Any] = orig_h_fn,
                                ro_f: bool = rec_only,
                            ):
                                def _hrule(q, k, v, *a, **kw):
                                    if ib[0] > 0:
                                        return ofn(q, k, v, *a, **kw)
                                    ib[0] += 1
                                    try:
                                        return _recurrent_hssd(
                                            ad_r,
                                            sb_r[-1][0],
                                            sb_r[-1][1],
                                            q,
                                            k,
                                            v,
                                            *a,
                                            pad_mask=sb_r[-1][2],
                                            _rec_only=ro_f,
                                            **kw,
                                        )
                                    finally:
                                        ib[0] -= 1

                                return _hrule

                            hr_fn = _make_hrule()
                            if (
                                isinstance(orig_h_fn, types.MethodType)
                                and getattr(orig_h_fn, "__self__", None) is attn
                            ):
                                raw_hfn = getattr(orig_h_fn, "__func__", orig_h_fn)

                                @functools.wraps(raw_hfn)
                                def _bound_hrule(_s: nn.Module, *ha: Any, _hw: Callable[..., Any] = hr_fn, **hkw: Any):
                                    return _hw(*ha, **hkw)

                                setattr(attn, attr, types.MethodType(_bound_hrule, attn))
                            else:
                                setattr(attn, attr, hr_fn)

                    def _make_hpre(
                        ad_r: _StandaloneHSSDAdapter = had,
                        sb_r: list[tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]] = st_box,
                        ib: list[int] = in_h_box,
                        wc_f: bool = h_wc,
                        wr_f: bool = h_wr,
                    ):
                        def _hpre(
                            _m: nn.Module, a: tuple[Any, ...], kw: dict[str, Any]
                        ) -> tuple[tuple[Any, ...], dict[str, Any]] | None:
                            x_in = a[0] if a else (kw["hidden_states"] if "hidden_states" in kw else kw["x"])
                            amask = kw.get("attention_mask", a[2] if len(a) >= 3 else None)
                            if amask is None and self._amask_stack:
                                amask = self._amask_stack[-1]
                            wm = self._active_mor_wmask(x_in.shape[0], x_in.shape[1])
                            pm: torch.Tensor | None = None
                            ret_pair: tuple[tuple[Any, ...], dict[str, Any]] | None = None
                            if (
                                isinstance(amask, torch.Tensor)
                                and amask.ndim == 2
                                and amask.shape[0] == x_in.shape[0]
                                and amask.shape[1] >= x_in.shape[1]
                            ):
                                pm = amask[:, -x_in.shape[1] :]
                                x_in = x_in * pm.to(dtype=x_in.dtype).unsqueeze(-1)
                                if a:
                                    ret_pair = ((x_in, *a[1:]), kw)
                                elif "hidden_states" in kw:
                                    nkw = dict(kw)
                                    nkw["hidden_states"] = x_in
                                    ret_pair = (a, nkw)
                                elif "x" in kw:
                                    nkw = dict(kw)
                                    nkw["x"] = x_in
                                    ret_pair = (a, nkw)
                            x_ad = x_in * wm.to(dtype=x_in.dtype).unsqueeze(-1) if wm is not None else x_in
                            sb_r.append((x_ad, wm, pm))
                            active_hssd.append((ad_r, sb_r, ib, wc_f, wr_f))
                            return ret_pair

                        return _hpre

                    def _make_hpost(
                        sb_r: list[tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]] = st_box,
                    ):
                        def _hpost(_m: nn.Module, _i: Any, _o: Any) -> None:
                            if sb_r:
                                sb_r.pop()
                            if active_hssd:
                                active_hssd.pop()

                        return _hpost

                    attn.register_forward_pre_hook(_make_hpre(), with_kwargs=True)
                    attn.register_forward_hook(_make_hpost())

                    if not (h_wc and h_wr):
                        mname = type(attn).__module__
                        hmod = sys.modules.get(mname)
                        if hmod is not None:
                            miss_h: list[tuple[str, bool]] = []
                            if not h_wc:
                                miss_h.extend((nm, False) for nm in chunk_attrs)
                            if not h_wr:
                                miss_h.extend((nm, True) for nm in rec_attrs)
                            for fn_nm, rec_only in miss_h:
                                if (mname, fn_nm) in patched_mods:
                                    continue
                                orig_fn = getattr(hmod, fn_nm, None)
                                if callable(orig_fn):
                                    patched_mods.add((mname, fn_nm))

                                    def _mdisp(
                                        q,
                                        k,
                                        v,
                                        *a,
                                        _orig: Callable[..., Any] = orig_fn,
                                        _rec: bool = rec_only,
                                        **kw,
                                    ):
                                        if not active_hssd:
                                            return _orig(q, k, v, *a, **kw)
                                        cad, csb, cir, cwc, cwr = active_hssd[-1]
                                        if cir[0] > 0 or (_rec and cwr) or ((not _rec) and cwc):
                                            return _orig(q, k, v, *a, **kw)
                                        cir[0] += 1
                                        try:
                                            return _recurrent_hssd(
                                                cad,
                                                csb[-1][0],
                                                csb[-1][1],
                                                q,
                                                k,
                                                v,
                                                *a,
                                                pad_mask=csb[-1][2],
                                                _rec_only=_rec,
                                                **kw,
                                            )
                                        finally:
                                            cir[0] -= 1

                                    setattr(hmod, fn_nm, _mdisp)

            lcfg = dict(arch.get("lsa") or {})
            if lcfg.get("enabled"):
                hd = getattr(hf_cfg, "head_dim", None) or (hf_cfg.hidden_size // hf_cfg.num_attention_heads)
                rope_p = getattr(hf_cfg, "rope_parameters", None) or {}
                prf = getattr(hf_cfg, "partial_rotary_factor", None) or rope_p.get("partial_rotary_factor", 1.0)
                rot = int(lcfg.get("rotary_dim") if lcfg.get("rotary_dim") is not None else int(hd * prf))
                for i, layer in enumerate(tm.layers):
                    attn = getattr(layer, "self_attn", None)
                    if attn is None or (lcfg.get("layers") is not None and i not in lcfg["layers"]):
                        continue
                    attn.lsa_indexer = _StandaloneLSAIndexer(
                        hf_cfg.num_attention_heads, hf_cfg.num_key_value_heads, hd, rot, lcfg
                    ).to(dtype=ref_dt)

                if str(lcfg.get("mode", "dense")) == "sparse":
                    self._install_standalone_lsa_dispatch(tm)

        def _install_standalone_lsa_dispatch(self, tm: nn.Module) -> None:
            tc = tm.config
            prev_impl = getattr(tc, "_attn_implementation", None) or "sdpa"
            lsa_name = "qloop_lsa"
            layer0_attn = next((getattr(ly, "self_attn", None) for ly in tm.layers if hasattr(ly, "self_attn")), None)
            host_mod = sys.modules.get(type(layer0_attn).__module__) if layer0_attn is not None else None
            toy_table = getattr(host_mod, "TOY_ATTENTION_FUNCTIONS", None) if host_mod is not None else None
            if isinstance(toy_table, dict):
                prior_toy = toy_table.get(lsa_name)
                dense_key = prev_impl if (prev_impl != lsa_name and prev_impl in toy_table) else "eager"
                dense_fn = toy_table.get(dense_key) or next(iter(toy_table.values()), None)
                if callable(dense_fn):

                    def _lsa_fn(
                        module: nn.Module,
                        q: torch.Tensor,
                        k: torch.Tensor,
                        v: torch.Tensor,
                        attention_mask: torch.Tensor | None = None,
                        scaling: float | None = None,
                        **kw: Any,
                    ):
                        ix = getattr(module, "lsa_indexer", None)
                        if ix is not None and not isinstance(ix, _StandaloneLSAIndexer) and callable(prior_toy):
                            return prior_toy(module, q, k, v, attention_mask, scaling=scaling, **kw)
                        sc = (q.shape[-1] ** -0.5) if scaling is None else float(scaling)
                        if not isinstance(ix, _StandaloneLSAIndexer) or ix.mode != "sparse" or k.shape[2] <= ix.window:
                            return dense_fn(module, q, k, v, attention_mask, scaling=sc, **kw)
                        return ix.sparse_attend(q, k, v, sc), None

                    toy_table[lsa_name] = _lsa_fn
                    tc._attn_implementation = lsa_name
                    return
            try:
                from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

                prior_hf = ALL_ATTENTION_FUNCTIONS.get(lsa_name, None)
                fb_key = (
                    prev_impl
                    if (prev_impl != lsa_name and prev_impl in ALL_ATTENTION_FUNCTIONS)
                    else ("sdpa" if "sdpa" in ALL_ATTENTION_FUNCTIONS else "eager")
                )
                if fb_key in ALL_ATTENTION_FUNCTIONS:
                    dense_hf = ALL_ATTENTION_FUNCTIONS[fb_key]

                    def _hf_lsa(
                        module: nn.Module,
                        q: torch.Tensor,
                        k: torch.Tensor,
                        v: torch.Tensor,
                        attention_mask: torch.Tensor | None = None,
                        scaling: float | None = None,
                        **kw: Any,
                    ):
                        ix = getattr(module, "lsa_indexer", None)
                        if ix is not None and not isinstance(ix, _StandaloneLSAIndexer) and callable(prior_hf):
                            return prior_hf(module, q, k, v, attention_mask, scaling=scaling, **kw)
                        sc = (q.shape[-1] ** -0.5) if scaling is None else float(scaling)
                        if not isinstance(ix, _StandaloneLSAIndexer) or ix.mode != "sparse" or k.shape[2] <= ix.window:
                            return dense_hf(module, q, k, v, attention_mask, scaling=sc, **kw)
                        return ix.sparse_attend(q, k, v, sc), None

                    ALL_ATTENTION_FUNCTIONS[lsa_name] = _hf_lsa
                    tc._attn_implementation = lsa_name
                    for ly in tm.layers:
                        sa = getattr(getattr(ly, "inner", ly), "self_attn", None)
                        scfg = getattr(sa, "config", None) if sa is not None else None
                        if scfg is not None:
                            try:
                                scfg._attn_implementation = lsa_name
                            except Exception:
                                pass
            except Exception:
                pass

        def _apply_standalone_loop(self, arch: dict[str, Any]) -> None:
            lcfg = dict(arch.get("loop") or {})
            if not lcfg.get("enabled"):
                return
            self._loop_spec = lcfg
            tm = self._find_tm()
            ref_dt = torch.float64 if tm.embed_tokens.weight.dtype == torch.float64 else torch.float32
            orig_layers = tm.layers
            n_phys = len(orig_layers)
            p_end, c_end = int(lcfg.get("prelude_end", 8)), int(lcfg.get("core_end", 32))
            n_passes = int(lcfg.get("num_passes", 2))
            has_routers = any(_router_of(orig_layers[i]) is not None for i in range(p_end, c_end))
            n_exp = int(tm.config.num_experts) if (has_routers and getattr(tm.config, "num_experts", None)) else None
            top_k = int(getattr(tm.config, "num_experts_per_tok", 8))
            norm_topk = bool(getattr(tm.config, "norm_topk_prob", True))

            attn_rank = int(lcfg.get("attn_adapter_rank", 0))
            attn_proj_names = tuple(lcfg.get("attn_proj_names", ("q_proj", "o_proj")))
            attn_specs: dict[int, dict[str, tuple[int, int]]] = {}
            attn_mods: list[tuple[int, str, nn.Module]] = []
            if attn_rank > 0 and n_passes > 1:
                for phys in range(p_end, c_end):
                    s_attn = getattr(orig_layers[phys], "self_attn", None)
                    if s_attn is None:
                        continue
                    found = _find_children(s_attn, attn_proj_names)
                    if len(found) != len(attn_proj_names):
                        raise RuntimeError(f"layer {phys} full-attn lacks {attn_proj_names}")
                    attn_specs[phys] = {}
                    for nm, mod in found.items():
                        pkey = nm.replace(".", "__")
                        attn_specs[phys][pkey] = _linear_dims(mod)
                        attn_mods.append((phys, pkey, mod))

            self._loop_adapters = _StandaloneLoopAdapters(
                tm.config.hidden_size, lcfg, n_exp, attn_proj_specs=attn_specs
            ).to(dtype=ref_dt)
            tm.loop_adapters = self._loop_adapters
            self.active_passes = int(lcfg.get("active_passes", n_passes))

            kv_names = tuple(lcfg.get("kv_proj_names", ("k_proj", "v_proj")))
            if lcfg.get("kv_share", True) and n_passes > 1:
                for phys in range(p_end, c_end):
                    s_attn = getattr(orig_layers[phys], "self_attn", None)
                    if s_attn is None:
                        continue
                    for nm, proj in _find_children(s_attn, kv_names).items():
                        proj.register_forward_hook(self._make_kv_hook(phys, nm))
                    kvc_mod = getattr(s_attn, "kv_cache", None)
                    if isinstance(kvc_mod, nn.Module):
                        self._wrap_standalone_kv_cache(phys, kvc_mod)

            if self._loop_adapters.pass_attn is not None:
                for phys, pkey, mod in attn_mods:
                    mod.register_forward_hook(self._make_attn_lora_hook(phys, pkey))

            if lcfg.get("mor", False) and lcfg.get("write_mask", True) and n_passes > 1:
                wz = tuple(lcfg.get("write_zero_names", ("in_proj_qkvz", "in_proj_qkv")))
                wn = tuple(lcfg.get("write_neg_names", ("in_proj_ba", "in_proj_b", "in_proj_a")))
                neg_v = float(lcfg.get("write_neg_value", -30.0))
                for phys in range(p_end, c_end):
                    l_attn = getattr(orig_layers[phys], "linear_attn", None)
                    if l_attn is None:
                        continue
                    for mod in _find_children(l_attn, wz).values():
                        mod.register_forward_hook(self._make_write_hook(None))
                    for mod in _find_children(l_attn, wn).values():
                        mod.register_forward_hook(self._make_write_hook(neg_v))
                    gad = getattr(l_attn, "gdn2_adapter", None)
                    if isinstance(gad, nn.Module) and isinstance(getattr(gad, "down", None), nn.Module):
                        gad.down.register_forward_hook(self._make_write_hook(None))

            if self._loop_adapters.pass_router is not None:
                for phys in range(p_end, c_end):
                    rt = _router_of(orig_layers[phys])
                    if rt is not None:
                        rt.register_forward_hook(
                            self._make_pass_router_hook(phys, phys - p_end, top_k, norm_topk), prepend=True
                        )

            order = [(i, 0) for i in range(p_end)]
            for r in range(1, n_passes + 1):
                order += [(i, r) for i in range(p_end, c_end)]
            order += [(i, 0) for i in range(c_end, n_phys)]
            v_layers = []
            ltypes = []
            for v, (i, r) in enumerate(order):
                lt = "full_attention" if hasattr(orig_layers[i], "self_attn") else "linear_attention"
                share = bool(lcfg.get("kv_share", True) and r >= 2 and lt == "full_attention")
                v_layers.append(
                    _StandaloneVirtualLayer(
                        self,
                        v,
                        i,
                        r,
                        r >= 1 and i == p_end,
                        r >= 1 and i == c_end - 1,
                        i if share else v,
                        lt,
                        orig_layers[i],
                    )
                )
                ltypes.append(lt)
            tm.layers = nn.ModuleList(v_layers)
            tm.config.num_hidden_layers = len(v_layers)
            tm.config.layer_types = ltypes
            self._slot_meta = [(v, r, lt) for v, ((_i, r), lt) in enumerate(zip(order, ltypes))]

            def _tm_cache_pre(_m: nn.Module, _a: tuple[Any, ...], kw: dict[str, Any]) -> None:
                pkv = kw.get("past_key_values")
                if pkv is None:
                    pkv = kw.get("past_key_value")
                if pkv is None:
                    pkv = kw.get("cache")
                if pkv is not None:
                    self._sync_cache_last_linear(pkv)

            tm.register_forward_pre_hook(_tm_cache_pre, with_kwargs=True)

        def _wrap_standalone_kv_cache(self, phys: int, kvc_mod: nn.Module) -> None:
            orig_fwd = kvc_mod.forward
            raw_fn = getattr(orig_fwd, "__func__", orig_fwd)

            @functools.wraps(raw_fn)
            def _st_kvc(_self: nn.Module, *a: Any, **kw: Any) -> Any:
                r = self._cur_pass
                if r < 1:
                    return orig_fwd(*a, **kw)
                key = (phys, "__kv_cache_mod__")
                if r == 1:
                    res = orig_fwd(*a, **kw)
                    self._loop_ctx.setdefault("kv", {})[key] = res
                    return res
                kv_d = self._loop_ctx.get("kv", {})
                if key in kv_d:
                    return kv_d[key]
                return orig_fwd(*a, **kw)

            kvc_mod.forward = types.MethodType(_st_kvc, kvc_mod)  # type: ignore[method-assign]

        def _make_kv_hook(self, phys: int, nm: str):
            def _hook(_m: nn.Module, _inp: Any, out: torch.Tensor):
                r = self._cur_pass
                if r < 1:
                    return None
                if r == 1:
                    self._loop_ctx["kv"][(phys, nm)] = out
                    return None
                return self._loop_ctx["kv"][(phys, nm)]

            return _hook

        def _make_attn_lora_hook(self, phys: int, pkey: str):
            def _hook(_m: nn.Module, inp: Any, out: torch.Tensor):
                r = self._cur_pass
                pa = self._loop_adapters.pass_attn if self._loop_adapters is not None else None
                if r < 2 or pa is None:
                    return None
                p_idx = r - 2
                if not torch.is_grad_enabled() and pa.is_zero(phys, pkey, p_idx):
                    return None
                return out + pa.delta(inp[0], phys, pkey, p_idx).to(out.dtype).view_as(out)

            return _hook

        def _make_write_hook(self, neg_val: float | None):
            def _hook(_m: nn.Module, _inp: Any, out: torch.Tensor):
                r = self._cur_pass
                if r < 2:
                    return None
                keep = self._loop_ctx.get("keep", {}).get(r)
                if keep is None:
                    return None
                k = keep.to(out.dtype).unsqueeze(-1)
                return out * k if neg_val is None else out * k + neg_val * (1.0 - k)

            return _hook

        def _make_pass_router_hook(self, phys: int, core_idx: int, top_k: int, norm_topk: bool):
            def _hook(mod: nn.Module, inp: Any, out: Any):
                r = self._cur_pass
                pr_mod = self._loop_adapters.pass_router if self._loop_adapters is not None else None
                if r < 1 or pr_mod is None:
                    return None
                e = pr_mod.num_experts
                x_in = inp[0]
                flat_x = x_in.reshape(-1, x_in.shape[-1])
                outs = out if isinstance(out, (tuple, list)) else (out,)
                ts = [o for o in outs if isinstance(o, torch.Tensor)]
                full = next((o for o in ts if o.is_floating_point() and o.shape[-1] == e), None)
                if full is None:
                    raise RuntimeError(f"router on layer {phys} lacks [..., {e}] float tensor")
                if phys not in self._is_probs_cache:
                    self._is_probs_cache[phys] = isinstance(out, (tuple, list)) and _looks_like_probs(full)
                is_probs = self._is_probs_cache[phys]
                idx = next((o for o in ts if not o.is_floating_point() and o.shape[-1] == top_k), None)
                sc = next((o for o in ts if o.is_floating_point() and o.shape[-1] == top_k and o is not full), None)

                if r == 1:
                    p1 = _upcast(full.reshape(-1, e)) if is_probs else F.softmax(_upcast(full.reshape(-1, e)), dim=-1)
                    t1 = (
                        idx.detach().reshape(-1, top_k).long()
                        if (idx is not None and top_k != e)
                        else p1.detach().topk(top_k, dim=-1).indices
                    )
                    self._loop_ctx["routing"][(phys, 1)] = t1
                    return None

                p_idx = r - 2
                if not torch.is_grad_enabled() and pr_mod.is_zero(p_idx, core_idx):
                    pr = _upcast(full.reshape(-1, e)) if is_probs else F.softmax(_upcast(full.reshape(-1, e)), dim=-1)
                    tr = (
                        idx.detach().reshape(-1, top_k).long()
                        if (idx is not None and top_k != e)
                        else pr.detach().topk(top_k, dim=-1).indices
                    )
                    self._loop_ctx["routing"][(phys, r)] = tr
                    return None

                prev_mask: torch.Tensor | None = None
                for p_prev in range(1, r):
                    pidx = self._loop_ctx["routing"].get((phys, p_prev))
                    if pidx is not None and pidx.shape[0] == flat_x.shape[0]:
                        if prev_mask is None:
                            prev_mask = flat_x.new_zeros(flat_x.shape[0], e, dtype=torch.float32)
                        prev_mask.scatter_(1, pidx, 1.0)

                delta = pr_mod.delta_logits(flat_x, p_idx, core_idx, prev_mask)
                if not isinstance(out, (tuple, list)):
                    new_out = out + delta.to(out.dtype).view_as(out)
                    pr = F.softmax(_upcast(new_out.reshape(-1, e)), dim=-1)
                    self._loop_ctx["routing"][(phys, r)] = pr.detach().topk(top_k, dim=-1).indices
                    return new_out

                w_mod = getattr(mod, "weight", None)
                b_mod = getattr(mod, "bias", None)
                b_up = _upcast(b_mod) if isinstance(b_mod, torch.Tensor) else None
                acc_dt = torch.float64 if flat_x.dtype == torch.float64 else torch.float32
                base_l: torch.Tensor | None = None
                q_kind = _st_quant_kind(mod)
                is_fp8_w = isinstance(w_mod, torch.Tensor) and (
                    w_mod.dtype in _FP8_DTYPES or "float8" in str(w_mod.dtype)
                )
                _, s_attr = _st_first_tensor(mod, _ST_LIN_S_ATTRS)
                if q_kind in ("linear_fp8", "linear_nvfp4") or is_fp8_w or (
                    isinstance(w_mod, torch.Tensor) and w_mod.ndim == 2 and s_attr is not None
                ):
                    w_deq: torch.Tensor | None = None
                    try:
                        if q_kind in ("linear_fp8", "linear_nvfp4"):
                            w_deq = _st_deq_lin(mod, acc_dt)
                        elif isinstance(w_mod, torch.Tensor):
                            s_res = (
                                s_attr
                                if s_attr is not None
                                else _st_resolve_fp8_scale(mod, w_mod, _ST_LIN_S_ATTRS, "lin")
                            )
                            w_deq = _st_deq_fp8(w_mod, s_res, acc_dt, _st_bsize_of(mod))
                    except (RuntimeError, TypeError, ValueError):
                        w_deq = None
                    if isinstance(w_deq, torch.Tensor) and w_deq.shape == (e, flat_x.shape[-1]):
                        x_acc = _upcast(flat_x).to(dtype=acc_dt)
                        b_acc = b_up.to(dtype=acc_dt) if b_up is not None else None
                        base_l = F.linear(x_acc, w_deq, b_acc)
                elif (
                    isinstance(w_mod, torch.Tensor)
                    and w_mod.is_floating_point()
                    and w_mod.shape == (e, flat_x.shape[-1])
                ):
                    base_l = F.linear(_upcast(flat_x), _upcast(w_mod), b_up)

                if base_l is None:
                    if is_probs:
                        base_l = _upcast(full.reshape(-1, e)).clamp_min(1e-30).log()
                    else:
                        base_l = _upcast(full.reshape(-1, e))
                new_l = base_l + delta.to(base_l.dtype)
                pr = F.softmax(new_l, dim=-1)
                top_w, top_idx = pr.topk(top_k, dim=-1)
                if norm_topk:
                    top_w = top_w / top_w.sum(-1, keepdim=True)
                self._loop_ctx["routing"][(phys, r)] = top_idx.detach()
                new_full = (pr if is_probs else new_l).to(full.dtype).view_as(full)
                rebuilt = []
                for item in out:
                    if item is full:
                        rebuilt.append(new_full)
                    elif sc is not None and item is sc:
                        rebuilt.append(top_w.to(sc.dtype).view_as(sc))
                    elif idx is not None and item is idx:
                        rebuilt.append(top_idx.to(idx.dtype).view_as(idx))
                    else:
                        rebuilt.append(item)
                return tuple(rebuilt) if isinstance(out, tuple) else rebuilt

            return _hook

        @property
        def max_passes(self) -> int:
            return 1 if self._loop_adapters is None else int(self._loop_spec.get("num_passes", 1))

        def set_active_passes(self, r: int) -> None:
            ri = int(r)
            if self._loop_adapters is None:
                if ri != 1:
                    raise ValueError(f"model has no loop adapters; cannot set active_passes={ri}")
                return
            if not 1 <= ri <= self.max_passes:
                raise ValueError(f"active_passes must be in [1, {self.max_passes}], got {ri}")
            self.active_passes = ri

        def depth(self, r: int) -> _StandaloneDepthContext:
            prev = self.active_passes
            self.set_active_passes(int(r))
            return _StandaloneDepthContext(self, prev, int(r))

        def get_input_embeddings(self) -> nn.Module:
            return self._find_tm().embed_tokens

        def get_output_embeddings(self) -> nn.Module:
            if hasattr(self.backbone, "lm_head"):
                return self.backbone.lm_head
            if hasattr(self.backbone, "get_output_embeddings"):
                return self.backbone.get_output_embeddings()
            raise RuntimeError("backbone has no lm_head module")

        def forward(
            self,
            input_ids: torch.Tensor | None = None,
            attention_mask: torch.Tensor | None = None,
            position_ids: torch.Tensor | None = None,
            past_key_values: Any = None,
            inputs_embeds: torch.Tensor | None = None,
            labels: torch.Tensor | None = None,
            use_cache: bool | None = None,
            output_attentions: bool | None = None,
            output_hidden_states: bool | None = None,
            return_dict: bool | None = None,
            logits_to_keep: int | None = None,
            **kwargs: Any,
        ) -> Any:
            try:
                if past_key_values is not None:
                    self._sync_cache_last_linear(past_key_values)
                if logits_to_keep is not None and labels is None:
                    tm_kw: dict[str, Any] = {
                        "input_ids": input_ids,
                        "past_key_values": past_key_values,
                        "use_cache": bool(use_cache),
                    }
                    if inputs_embeds is not None:
                        tm_kw["inputs_embeds"] = inputs_embeds
                    if attention_mask is not None:
                        tm_kw["attention_mask"] = attention_mask
                    if position_ids is not None:
                        tm_kw["position_ids"] = position_ids
                    if output_hidden_states is not None:
                        tm_kw["output_hidden_states"] = output_hidden_states
                    tm_out = self._find_tm()(**tm_kw, **kwargs)
                    h_last = tm_out.last_hidden_state
                    keep = int(logits_to_keep)
                    h_slice = h_last[:, -keep:, :] if keep > 0 else h_last[:, :0, :]
                    logits = self.get_output_embeddings()(h_slice)
                    out = _StandaloneCausalLMOutput(
                        logits=logits,
                        loss=None,
                        past_key_values=getattr(tm_out, "past_key_values", None),
                        hidden_states=getattr(tm_out, "hidden_states", None),
                    )
                    return out if return_dict is not False else (logits, out.past_key_values)

                call_kw: dict[str, Any] = dict(kwargs)
                if attention_mask is not None:
                    call_kw["attention_mask"] = attention_mask
                if position_ids is not None:
                    call_kw["position_ids"] = position_ids
                if past_key_values is not None:
                    call_kw["past_key_values"] = past_key_values
                if inputs_embeds is not None:
                    call_kw["inputs_embeds"] = inputs_embeds
                if labels is not None:
                    call_kw["labels"] = labels
                if use_cache is not None:
                    call_kw["use_cache"] = use_cache
                if output_attentions is not None:
                    call_kw["output_attentions"] = output_attentions
                if output_hidden_states is not None:
                    call_kw["output_hidden_states"] = output_hidden_states
                res = self.backbone(input_ids=input_ids, **call_kw)
                return (res.logits, getattr(res, "past_key_values", None)) if return_dict is False else res
            finally:
                self._amask_stack.clear()

        @classmethod
        def from_pretrained(
            cls,
            pretrained_model_name_or_path: str | os.PathLike[str],
            *args: Any,
            config: Qwen36LoopHSSDConfig | None = None,
            token: str | None = None,
            revision: str | None = None,
            local_files_only: bool = False,
            device_map: str | torch.device | None = None,
            torch_dtype: str | torch.dtype | None = None,
            dtype: str | torch.dtype | None = None,
            strict: bool = True,
            **kwargs: Any,
        ):
            del args
            kwargs.pop("trust_remote_code", None)
            explicit_dt = dtype if dtype is not None else torch_dtype
            dt_override: torch.dtype | None = None
            if isinstance(explicit_dt, torch.dtype):
                dt_override = explicit_dt
            elif isinstance(explicit_dt, str) and explicit_dt != "auto":
                dt_override = _resolve_dt(explicit_dt)

            local_dir = os.fspath(pretrained_model_name_or_path)
            if not os.path.isdir(local_dir):
                from huggingface_hub import snapshot_download

                local_dir = snapshot_download(
                    repo_id=str(local_dir),
                    repo_type="model",
                    token=token,
                    revision=revision,
                    local_files_only=local_files_only,
                )
            if config is None:
                config = Qwen36LoopHSSDConfig.from_pretrained(local_dir)
            elif not isinstance(config, Qwen36LoopHSSDConfig):
                cdict = config.to_dict() if hasattr(config, "to_dict") else dict(vars(config))
                config = Qwen36LoopHSSDConfig(**cdict)

            inst = cls(config, _defer_loop=True, _dtype_override=dt_override)
            pmap = dict(inst.backbone.named_parameters())
            bmap = dict(inst.backbone.named_buffers())
            mmap = dict(inst.backbone.named_modules())
            expected_bb = set(pmap.keys())
            tie_we = bool(getattr(config, "tie_word_embeddings", False))
            q_arch = dict((config.qloop_v3_arch or {}).get("quantization") or {})
            nvfp4_meta_map = dict(q_arch.get("nvfp4_modules") or {})
            dyn_qattrs = {
                "weight_packed",
                "weight_scale",
                "weight_scale_inv",
                "weight_global_scale",
                "weight_scale_2",
                "gate_up_proj_packed",
                "gate_up_proj_scale",
                "gate_up_proj_scale_inv",
                "gate_up_proj_global_scale",
                "gate_up_global_scale",
                "gate_up_proj_scale_2",
                "down_proj_packed",
                "down_proj_scale",
                "down_proj_scale_inv",
                "down_proj_global_scale",
                "down_global_scale",
                "down_proj_scale_2",
                "scale_inv",
                "global_scale",
                "input_scale",
                "input_global_scale",
                "activation_scale",
                "k_scale",
                "v_scale",
                "q_scale",
                "o_scale",
                "weight_shape",
            }
            repl_dense = {
                "gate_up_proj_packed": "gate_up_proj",
                "down_proj_packed": "down_proj",
                "weight_packed": "weight",
            }

            idx_p = os.path.join(local_dir, "model.safetensors.index.json")
            single_p = os.path.join(local_dir, "model.safetensors")
            if os.path.isfile(idx_p):
                with open(idx_p, encoding="utf-8") as f:
                    sfiles = list(dict.fromkeys(json.load(f)["weight_map"].values()))
            elif os.path.isfile(single_p):
                sfiles = ["model.safetensors"]
            else:
                sfiles = sorted(fn for fn in os.listdir(local_dir) if fn.endswith(".safetensors"))
                if not sfiles:
                    raise RuntimeError(f"no .safetensors files found in {local_dir!r}")

            loaded_bb: set[str] = set()
            unexpected_bb: list[str] = []
            loop_sd: dict[str, torch.Tensor] = {}

            for sf in sfiles:
                sp = os.path.join(local_dir, sf)
                if not os.path.isfile(sp):
                    raise RuntimeError(f"missing shard file {sp!r}")
                for raw_k, t in _load_st(sp).items():
                    k = raw_k[len("backbone.") :] if raw_k.startswith("backbone.") else raw_k
                    if k.startswith("loop_adapters."):
                        loop_sd[k[len("loop_adapters.") :]] = t
                        continue
                    if ".loop_adapters." in k:
                        loop_sd[k.split(".loop_adapters.", 1)[1]] = t
                        continue
                    target = pmap.get(k)
                    is_param = target is not None
                    if target is None:
                        target = bmap.get(k)
                    if target is None and tie_we and k.endswith("lm_head.weight"):
                        hm = getattr(inst.backbone, "lm_head", None)
                        if hm is not None and isinstance(getattr(hm, "weight", None), torch.Tensor):
                            target = hm.weight
                            is_param = True
                    if target is None:
                        p_path, sep, a_nm = k.rpartition(".")
                        p_mod = mmap.get(p_path) if sep else None
                        if (
                            p_mod is None
                            and sep
                            and p_path.endswith(".kv_cache")
                            and a_nm in ("k_scale", "v_scale", "q_scale", "quant_scale")
                        ):
                            gp_path, gp_sep, kvc_nm = p_path.rpartition(".")
                            gp_mod = mmap.get(gp_path) if gp_sep else None
                            if gp_mod is not None and kvc_nm == "kv_cache":
                                kvc_stub = nn.Module()
                                gp_mod.kv_cache = kvc_stub  # type: ignore[attr-defined]
                                mmap[p_path] = kvc_stub
                                p_mod = kvc_stub
                        if p_mod is not None and a_nm in dyn_qattrs:
                            d_attr = repl_dense.get(a_nm)
                            if d_attr is not None:
                                d_key = f"{p_path}.{d_attr}" if p_path else d_attr
                                p_mod._parameters.pop(d_attr, None)
                                p_mod._buffers.pop(d_attr, None)
                                pmap.pop(d_key, None)
                                bmap.pop(d_key, None)
                                expected_bb.discard(d_key)
                            m_meta = nvfp4_meta_map.get(p_path) or {}
                            if m_meta.get("storage") == "buffers":
                                p_mod.register_buffer(a_nm, t.clone())
                                bmap[k] = p_mod._buffers[a_nm]  # type: ignore[assignment]
                            else:
                                np_val = nn.Parameter(t.clone(), requires_grad=False)
                                p_mod.register_parameter(a_nm, np_val)
                                pmap[k] = np_val
                                expected_bb.add(k)
                                loaded_bb.add(k)
                            if a_nm.endswith("_packed") or m_meta:
                                if getattr(p_mod, "group_size", None) is None:
                                    p_mod.group_size = int(m_meta.get("group_size", 16))  # type: ignore[attr-defined]
                                sm = m_meta.get("scale_mode")
                                if sm in ("divisor", "multiplier"):
                                    p_mod._qloop_nvfp4_scale_mode = str(sm)  # type: ignore[attr-defined]
                                elif a_nm.endswith("_global_scale") and not hasattr(
                                    p_mod, "_qloop_nvfp4_scale_mode"
                                ):
                                    p_mod._qloop_nvfp4_scale_mode = "divisor"  # type: ignore[attr-defined]
                            continue
                        unexpected_bb.append(k)
                        continue
                    if tuple(target.shape) != tuple(t.shape):
                        raise RuntimeError(
                            f"shape mismatch for {k}: model {tuple(target.shape)} vs shard {tuple(t.shape)}"
                        )
                    with torch.no_grad():
                        if t.dtype in _FP8_DTYPES or not t.is_floating_point():
                            if isinstance(target, nn.Parameter):
                                target.data = t.to(device=target.device)
                            else:
                                p_path, _, a_nm = k.rpartition(".")
                                p_mod = mmap.get(p_path)
                                if p_mod is not None and a_nm in p_mod._buffers:
                                    p_mod._buffers[a_nm] = t.to(device=target.device)
                                else:
                                    target.copy_(t.to(device=target.device, dtype=target.dtype))
                        elif dt_override is None and target.dtype != t.dtype:
                            target.data = t.to(device=target.device, dtype=t.dtype)
                        else:
                            target.copy_(t.to(device=target.device, dtype=target.dtype))
                    if is_param:
                        loaded_bb.add(k)

            if tie_we:
                hm = getattr(inst.backbone, "lm_head", None)
                em = getattr(inst._find_tm(), "embed_tokens", None)
                if (
                    hm is not None
                    and em is not None
                    and isinstance(getattr(hm, "weight", None), nn.Parameter)
                    and isinstance(getattr(em, "weight", None), nn.Parameter)
                ):
                    hm.weight = em.weight
                    if any(lk.endswith("embed_tokens.weight") for lk in loaded_bb):
                        for ek in list(expected_bb):
                            if ek.endswith("lm_head.weight"):
                                loaded_bb.add(ek)

            _st_install_quant_fallback(inst.backbone)
            inst._apply_standalone_loop(config.qloop_v3_arch or {})
            missing_loop: list[str] = []
            unexpected_loop: list[str] = []
            if inst._loop_adapters is not None:
                lp = dict(inst._loop_adapters.named_parameters())
                lb = dict(inst._loop_adapters.named_buffers())
                missing_loop = sorted(k for k in lp if k not in loop_sd)
                unexpected_loop = sorted(k for k in loop_sd if k not in lp and k not in lb)
                with torch.no_grad():
                    for k, t in loop_sd.items():
                        ltgt = lp.get(k) if k in lp else lb.get(k)
                        if ltgt is None:
                            continue
                        if tuple(ltgt.shape) != tuple(t.shape):
                            raise RuntimeError(
                                f"shape mismatch for loop_adapters.{k}: {tuple(ltgt.shape)} vs {tuple(t.shape)}"
                            )
                        ltgt.copy_(t.to(device=ltgt.device, dtype=ltgt.dtype))
            elif loop_sd:
                unexpected_loop = sorted(f"loop_adapters.{k}" for k in loop_sd)

            if strict:
                missing_bb = sorted(expected_bb - loaded_bb)
                if missing_bb or unexpected_bb or missing_loop or unexpected_loop:
                    raise RuntimeError(
                        "Whole Model Pack standalone state_dict mismatch: "
                        f"missing_backbone={missing_bb[:10]}, unexpected_backbone={unexpected_bb[:10]}, "
                        f"missing_loop={missing_loop[:10]}, unexpected_loop={unexpected_loop[:10]}"
                    )

            if isinstance(device_map, torch.device):
                inst.to(device_map)
            elif isinstance(device_map, str):
                if device_map == "auto":
                    if torch.cuda.is_available():
                        inst.to("cuda:0")
                else:
                    inst.to(device_map)
            inst.eval()
            return inst
'''


def write_standalone_hf_modeling_files(out_dir: str | os.PathLike[str]) -> dict[str, str]:
    """Write standalone `configuration_qwen36_loophssd.py` and `modeling_qwen36_loophssd.py` to `out_dir`."""
    target_dir = os.fspath(out_dir)
    os.makedirs(target_dir, exist_ok=True)
    cfg_file = os.path.join(target_dir, CONFIG_FILENAME)
    mod_file = os.path.join(target_dir, MODELING_FILENAME)
    with open(cfg_file, "w", encoding="utf-8") as f:
        f.write(_STANDALONE_CONFIG_SRC)
    with open(mod_file, "w", encoding="utf-8") as f:
        f.write(_STANDALONE_MODELING_SRC)
    return {CONFIG_FILENAME: cfg_file, MODELING_FILENAME: mod_file}


__all__ = [
    "AUTO_MAP",
    "CONFIG_FILENAME",
    "MODELING_FILENAME",
    "CausalLMPackOutput",
    "Qwen36LoopHSSDConfig",
    "Qwen36LoopHSSDForCausalLM",
    "load_safetensors_file",
    "resolve_pack_directory",
    "save_safetensors_file",
    "write_standalone_hf_modeling_files",
]
