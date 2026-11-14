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
            expected_params = set(param_map.keys())

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
                        unexpected_keys.append(key)
                        continue
                    require(
                        tuple(target.shape) == tuple(tensor.shape),
                        f"shape mismatch for {key}: model {tuple(target.shape)} vs shard {tuple(tensor.shape)}",
                    )
                    with torch.no_grad():
                        if dt_override is None and target.dtype != tensor.dtype:
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
import importlib.util
import json
import math
import os
import struct
import sys
from collections import OrderedDict
from typing import Any, Callable

import torch
import torch.nn.functional as F
from torch import nn

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
        return x.float() if x.dtype in (torch.float16, torch.bfloat16) else x

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
        cos_t, sin_t = torch.cos(theta), torch.sin(theta)
        x0, x1 = pairs[..., 0], pairs[..., 1]
        return torch.stack([x0 * cos_t - x1 * sin_t, x0 * sin_t + x1 * cos_t], dim=-1).reshape_as(x)

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
            step_count: int,
            chunk_size: int,
            window_chunks: int,
            page_states: torch.Tensor,
            s_chunk_start: torch.Tensor,
            theta_cum: torch.Tensor,
            boundary_ring: tuple[torch.Tensor, ...],
            swa_k_ring: torch.Tensor,
            swa_v_ring: torch.Tensor,
            swa_valid_ring: torch.Tensor,
        ) -> None:
            self.step_count = int(step_count)
            self.chunk_size = int(chunk_size)
            self.window_chunks = int(window_chunks)
            self.page_states = page_states
            self.s_chunk_start = s_chunk_start
            self.theta_cum = theta_cum
            self.boundary_ring = boundary_ring
            self.swa_k_ring = swa_k_ring
            self.swa_v_ring = swa_v_ring
            self.swa_valid_ring = swa_valid_ring

        @property
        def max_swa_len(self) -> int:
            return (self.window_chunks + 1) * self.chunk_size

        @classmethod
        def zeros(
            cls,
            bsz: int,
            num_pages: int,
            hv: int,
            dk: int,
            dv: int,
            *,
            chunk_size: int,
            window_chunks: int,
            device: torch.device,
            dtype: torch.dtype,
        ) -> _HSSDCacheState:
            ps = torch.zeros(bsz, num_pages, hv, dk, dv, device=device, dtype=dtype)
            sc = ps[:, 0].clone()
            th = torch.zeros(bsz, hv, dk // 2, device=device, dtype=dtype)
            br = tuple(ps for _ in range(window_chunks + 1))
            mw = (window_chunks + 1) * chunk_size
            kr = torch.zeros(bsz, mw, hv, dk, device=device, dtype=dtype)
            vr = torch.zeros(bsz, mw, hv, dv, device=device, dtype=dtype)
            vlr = torch.zeros(bsz, mw, device=device, dtype=torch.bool)
            return cls(0, int(chunk_size), int(window_chunks), ps, sc, th, br, kr, vr, vlr)

        def to(self, device: torch.device, dtype: torch.dtype) -> _HSSDCacheState:
            ps = self.page_states.to(device=device, dtype=dtype)
            return _HSSDCacheState(
                step_count=self.step_count,
                chunk_size=self.chunk_size,
                window_chunks=self.window_chunks,
                page_states=ps,
                s_chunk_start=self.s_chunk_start.to(device=device, dtype=dtype),
                theta_cum=self.theta_cum.to(device=device, dtype=dtype),
                boundary_ring=tuple(t.to(device=device, dtype=dtype) for t in self.boundary_ring),
                swa_k_ring=self.swa_k_ring.to(device=device, dtype=dtype),
                swa_v_ring=self.swa_v_ring.to(device=device, dtype=dtype),
                swa_valid_ring=self.swa_valid_ring.to(device=device),
            )

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
            self.cap_g = float(hcfg.get("gate_logit_cap", 8.0))
            self.cap_d = float(hcfg.get("decay_log_cap", 2.0))
            eff_rank = max(1, min(int(hcfg.get("rank", 64)), self.hidden_size))

            self.down = nn.Linear(self.hidden_size, eff_rank, bias=False)
            self.erase_up = nn.Linear(eff_rank, self.num_v_heads * self.head_k_dim, bias=False)
            self.write_up = nn.Linear(eff_rank, self.num_v_heads * self.head_v_dim, bias=False)
            self.decay_up = nn.Linear(eff_rank, self.num_v_heads * self.half_k_dim, bias=False)
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
            wg = _logistic_shift(bf, dw)
            brate = 1.0 + 0.1 * (-_upcast(g_base).clamp(max=0.0)).tanh().unsqueeze(-1)
            cmod = (brate * dd.exp()).repeat_interleave(2, dim=-1)
            pscale = _upcast(self.page_gamma_prior) * _upcast(self.log_page_gamma).clamp(-3.0, 3.0).exp()
            gamma = pscale.view(1, 1, self.num_pages, 1, 1) * cmod.unsqueeze(2)
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
                "gamma": gamma,
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
        if hasattr(init_st, "page_states") and hasattr(init_st, "boundary_ring"):
            ps = init_st.page_states.to(device=device, dtype=dtype)
            return _HSSDCacheState(
                step_count=int(init_st.step_count),
                chunk_size=int(init_st.chunk_size),
                window_chunks=int(init_st.window_chunks),
                page_states=ps,
                s_chunk_start=init_st.s_chunk_start.to(device=device, dtype=dtype),
                theta_cum=init_st.theta_cum.to(device=device, dtype=dtype),
                boundary_ring=tuple(t.to(device=device, dtype=dtype) for t in init_st.boundary_ring),
                swa_k_ring=init_st.swa_k_ring.to(device=device, dtype=dtype),
                swa_v_ring=init_st.swa_v_ring.to(device=device, dtype=dtype),
                swa_valid_ring=init_st.swa_valid_ring.to(device=device),
            )
        if isinstance(init_st, torch.Tensor):
            st_t = _upcast(init_st).to(device=device, dtype=dtype)
            ps = st_t.unsqueeze(1).expand(bsz, num_pages, hv, dk, dv).clone() if st_t.ndim == 4 else st_t.clone()
            base = _HSSDCacheState.zeros(
                bsz, num_pages, hv, dk, dv, chunk_size=c_size, window_chunks=w_chunks, device=device, dtype=dtype
            )
            base.page_states = ps
            base.s_chunk_start = ps[:, 0].clone()
            base.boundary_ring = tuple(ps for _ in range(w_chunks + 1))
            return base
        raise TypeError(f"Unsupported HSSD initial_state: {type(init_st).__name__}")

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

        bsz, seq_len, _ = x_in.shape
        hv, dk, dv = adapter.num_v_heads, adapter.head_k_dim, adapter.head_v_dim
        head_first = bool(kwargs.get("head_first", False)) or (
            seq_len != hv and value.shape[1] == hv and value.shape[2] == seq_len
        )
        if head_first:
            query, key, value = query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2)
            g, beta = g.transpose(1, 2), beta.transpose(1, 2)

        init_dtype = query.dtype
        proj = adapter.project_inputs(x_in, query, key, value, g, beta, use_qk_l2norm=use_l2)
        acc_dt = torch.float64 if init_dtype == torch.float64 else torch.float32
        c_size, w_chunks, m_pages, tk_pages = (
            adapter.chunk_size,
            adapter.window_chunks,
            adapter.num_pages,
            adapter.k_pages,
        )
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
            dtype=acc_dt,
        )
        slp = _upcast(adapter.log_surprise_slope).exp().to(acc_dt).view(1, 1, hv)
        thr = _upcast(adapter.surprise_thresh).to(acc_dt).view(1, 1, hv)
        sc = (dk ** -0.5) if scale is None else float(scale)
        max_win = st.max_swa_len
        outs = []

        for i in range(seq_len):
            q_i = proj["q"][:, i : i + 1].to(acc_dt)
            k_i = proj["k"][:, i : i + 1].to(acc_dt)
            v_i = proj["v"][:, i : i + 1].to(acc_dt)
            gam_i = proj["gamma"][:, i : i + 1].to(acc_dt).clamp_min(0.0)
            bf_i = proj["b_gate"][:, i : i + 1].to(acc_dt)
            wf_i = proj["w_gate"][:, i : i + 1].to(acc_dt)
            th_i = proj["theta"][:, i : i + 1].to(acc_dt)
            sb_i = proj["sb"][:, i : i + 1].to(acc_dt)
            wl_i = proj["wl"][:, i : i + 1].to(acc_dt)
            rl_i = proj["rl"][:, i : i + 1].to(acc_dt)
            qs_i = proj["qs"][:, i : i + 1].to(acc_dt)
            ks_i = proj["ks"][:, i : i + 1].to(acc_dt)
            vs_i = proj["vs"][:, i : i + 1].to(acc_dt)

            if wmask is not None:
                wm = wmask[:, i : i + 1].reshape(bsz, 1).to(device=query.device, dtype=acc_dt).clamp(0.0, 1.0)
                valid_tok = wm.squeeze(1) > 0.5
                th_i = th_i * wm.unsqueeze(-1).unsqueeze(-1)
            else:
                wm = None
                valid_tok = torch.ones(bsz, device=query.device, dtype=torch.bool)

            idx0 = st.step_count
            s_cstart = st.page_states[:, 0] if (idx0 % c_size == 0) else st.s_chunk_start
            theta_cum = torch.remainder(st.theta_cum + th_i.squeeze(1), 2.0 * math.pi)
            q_hat = _apply_2d_rotation(q_i, -theta_cum.unsqueeze(1))
            if sc != 1.0:
                q_hat = q_hat * sc
            k_hat = _apply_2d_rotation(k_i, -theta_cum.unsqueeze(1))

            # 1. Far-horizon top-k page readout from S_boundary[max(0, n - W)] == st.boundary_ring[0]
            pi_r = F.softmax(rl_i, dim=-1)
            tv_r, ti_r = torch.topk(pi_r, k=tk_pages, dim=-1)
            tw_r = tv_r / tv_r.sum(dim=-1, keepdim=True).clamp_min(1e-30)
            all_far = (q_hat.permute(0, 2, 1, 3).unsqueeze(1) @ st.boundary_ring[0]).permute(0, 3, 1, 2, 4)
            gidx_r = ti_r.unsqueeze(-1).unsqueeze(-1).expand(bsz, 1, tk_pages, hv, dv)
            o_far = (torch.gather(all_far, dim=2, index=gidx_r) * tw_r.unsqueeze(-1).unsqueeze(-1)).sum(dim=2)

            # 2. Disjoint circular SWA ring update + PeakBoltzmannGate blend
            slot_idx = torch.tensor([idx0 % max_win], device=query.device, dtype=torch.long)
            swa_kr = st.swa_k_ring.index_copy(1, slot_idx, ks_i)
            swa_vr = st.swa_v_ring.index_copy(1, slot_idx, vs_i)
            swa_vl = st.swa_valid_ring.index_copy(1, slot_idx, valid_tok.unsqueeze(1))

            far_cut = max(0, (idx0 // c_size) - w_chunks) * c_size
            win_idx = torch.arange(far_cut, idx0 + 1, device=query.device) % max_win
            k_win = swa_kr.index_select(1, win_idx).transpose(1, 2)
            v_win = swa_vr.index_select(1, win_idx).transpose(1, 2)
            val_win = swa_vl.index_select(1, win_idx)[:, None, None, :]

            scores = (qs_i.transpose(1, 2) @ k_win.transpose(-1, -2)) * (dk ** -0.5)
            has_v = val_win.any(dim=-1)
            safe_v = torch.where(has_v.unsqueeze(-1), val_win, torch.ones_like(val_win))
            probs = torch.softmax(scores.masked_fill(~safe_v, float("-inf")), dim=-1)
            probs = torch.where(has_v.unsqueeze(-1), probs, torch.zeros_like(probs))
            o_loc = (probs @ v_win).transpose(1, 2)
            p_hit = adapter.boltzmann_gate(scores, valid_mask=val_win).transpose(1, 2).unsqueeze(-1)
            outs.append(p_hit * o_loc + (1.0 - p_hit) * o_far)

            # 3. Smooth chunk-start surprise s = g^2 + top-k write routing + over-relaxed DPLR update
            pred_v = (k_hat.transpose(1, 2) @ s_cstart).transpose(1, 2)
            eps_err = _safe_norm(v_i - pred_v, dim=-1) / (_safe_norm(v_i, dim=-1) + 1e-6)
            g_root = torch.sigmoid(slp * (eps_err - thr) + sb_i)
            if wm is not None:
                g_root = g_root * wm.unsqueeze(-1)
            pi_w = F.softmax(wl_i, dim=-1)
            tv_w, ti_w = torch.topk(pi_w, k=tk_pages, dim=-1)
            tr_w = torch.sqrt((tv_w / tv_w.sum(dim=-1, keepdim=True).clamp_min(1e-30)).clamp_min(1e-30))
            w_root = torch.zeros_like(pi_w).scatter(-1, ti_w, tr_w)
            g_page = w_root.unsqueeze(-1) * g_root.unsqueeze(2)

            ke = (g_page.unsqueeze(-1) * (_safe_sqrt(bf_i) * k_hat).unsqueeze(2)).squeeze(1)
            ve = (g_page.unsqueeze(-1) * (wf_i * v_i).unsqueeze(2)).squeeze(1)
            dec = (-(g_page * g_page).unsqueeze(-1) * gam_i).squeeze(1).exp().unsqueeze(-1) * st.page_states
            mem = (ke.unsqueeze(-2) @ dec).squeeze(-2)
            new_pages = dec + ke.unsqueeze(-1) * (ve - mem).unsqueeze(-2)

            new_step = idx0 + 1
            if new_step % c_size == 0:
                new_bring = (*st.boundary_ring[1:], new_pages)
                new_cstart = new_pages[:, 0]
            else:
                new_bring = st.boundary_ring
                new_cstart = s_cstart

            st = _HSSDCacheState(
                step_count=new_step,
                chunk_size=c_size,
                window_chunks=w_chunks,
                page_states=new_pages,
                s_chunk_start=new_cstart,
                theta_cum=theta_cum,
                boundary_ring=new_bring,
                swa_k_ring=swa_kr,
                swa_v_ring=swa_vr,
                swa_valid_ring=swa_vl,
            )

        out = torch.cat(outs, dim=1).to(init_dtype)
        if head_first:
            out = out.transpose(1, 2)
        return out, (st if out_final else None)

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

    class _SharedKVProxy:
        def __init__(self, cache: Any) -> None:
            object.__setattr__(self, "_cache", cache)

        def __getattr__(self, name: str) -> Any:
            return getattr(object.__getattribute__(self, "_cache"), name)

        def __setattr__(self, name: str, value: Any) -> None:
            setattr(object.__getattribute__(self, "_cache"), name, value)

        def __getitem__(self, idx: Any) -> Any:
            return object.__getattribute__(self, "_cache")[idx]

        def update(self, k: torch.Tensor, v: torch.Tensor, layer_idx: int, *a: Any, **kw: Any) -> Any:
            return object.__getattribute__(self, "_cache").update(k[..., :0, :], v[..., :0, :], layer_idx, *a, **kw)

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
        if isinstance(inf, int) and isinstance(outf, int):
            return inf, outf
        w = getattr(mod, "weight", None)
        if isinstance(w, torch.Tensor) and w.ndim == 2:
            return int(w.shape[1]), int(w.shape[0])
        raise RuntimeError(f"cannot determine linear dims on {type(mod).__name__}")

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
            vidx, _phys, r, entry, exit_flag, cidx = self._meta
            if vidx == 0:
                ow._loop_ctx = {"anchor": None, "start": {}, "kv": {}, "routing": {}, "keep": {}}
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
            pkv = kwargs.get("past_key_values")
            if pkv is not None and cidx != vidx:
                kwargs = dict(kwargs)
                kwargs["past_key_values"] = _SharedKVProxy(pkv)
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
            self._cur_pass = 0
            self._tuple_out = False
            self._tuple_extra = 0
            self._loop_ctx: dict[str, Any] = {}
            self._loop_spec: dict[str, Any] = {}
            self._loop_adapters: _StandaloneLoopAdapters | None = None
            self._is_probs_cache: dict[int, bool] = {}
            self.active_passes = 1
            self._install_submodules(arch, hf_cfg)
            if not _defer_loop:
                self._apply_standalone_loop(arch)

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

            gcfg = dict(arch.get("gdn2") or {})
            if gcfg.get("enabled"):
                rank = int(gcfg.get("rank", 64))
                cap_g = float(gcfg.get("gate_logit_cap", 8.0))
                cap_d = float(gcfg.get("decay_log_cap", 1.0))
                patched_gmods: set[str] = set()
                active_gdn2: list[tuple[_StandaloneGDN2Adapter, list[torch.Tensor]]] = []

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

                    def _make_gpre(ad_r: _StandaloneGDN2Adapter = ad, xb: list[torch.Tensor] = x_box):
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
                            active_gdn2.append((ad_r, xb))

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
                    g_wrapped = False
                    for attr in ("chunk_gated_delta_rule", "fused_recurrent_gated_delta_rule"):
                        if callable(getattr(attn, attr, None)):
                            g_wrapped = True

                            def _make_grule(ad_r: _StandaloneGDN2Adapter = ad, xb: list[torch.Tensor] = x_box):
                                def _grule(q, k, v, *ra, **rkw):
                                    return _run_gdn2_rule(ad_r, xb[-1], q, k, v, *ra, **rkw)

                                return _grule

                            setattr(attn, attr, _make_grule())
                    if not g_wrapped:
                        mname = type(attn).__module__
                        hmod = sys.modules.get(mname)
                        if hmod is not None and mname not in patched_gmods:
                            for fn_nm in (
                                "chunk_gated_delta_rule",
                                "torch_chunk_gated_delta_rule",
                                "fused_recurrent_gated_delta_rule",
                                "torch_recurrent_gated_delta_rule",
                            ):
                                orig_fn = getattr(hmod, fn_nm, None)
                                if callable(orig_fn):
                                    patched_gmods.add(mname)

                                    def _gmdisp(q, k, v, *ra, _orig: Callable[..., Any] = orig_fn, **rkw):
                                        if not active_gdn2:
                                            return _orig(q, k, v, *ra, **rkw)
                                        cad, cxb = active_gdn2[-1]
                                        return _run_gdn2_rule(cad, cxb[-1], q, k, v, *ra, **rkw)

                                    setattr(hmod, fn_nm, _gmdisp)

            hcfg = dict(arch.get("hssd") or {})
            if hcfg.get("enabled"):
                patched_mods: set[str] = set()
                active_hssd: list[tuple[_StandaloneHSSDAdapter, list[tuple[torch.Tensor, torch.Tensor | None]]]] = []
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
                    st_box: list[tuple[torch.Tensor, torch.Tensor | None]] = []

                    def _make_hpre(
                        ad_r: _StandaloneHSSDAdapter = had,
                        sb_r: list[tuple[torch.Tensor, torch.Tensor | None]] = st_box,
                    ):
                        def _hpre(_m: nn.Module, a: tuple[Any, ...], kw: dict[str, Any]) -> None:
                            x_in = a[0] if a else (kw["hidden_states"] if "hidden_states" in kw else kw["x"])
                            amask = kw.get("attention_mask", a[2] if len(a) >= 3 else None)
                            wm = self._active_mor_wmask(x_in.shape[0], x_in.shape[1])
                            if (
                                isinstance(amask, torch.Tensor)
                                and amask.ndim == 2
                                and amask.shape[0] == x_in.shape[0]
                                and amask.shape[1] >= x_in.shape[1]
                            ):
                                am_sl = amask[:, -x_in.shape[1] :].to(x_in.dtype)
                                if x_in.shape[1] > 1:
                                    x_in = x_in * am_sl.unsqueeze(-1)
                                wm = am_sl if wm is None else wm.to(am_sl.dtype) * am_sl
                            if wm is not None:
                                x_in = x_in * wm.to(x_in.dtype).unsqueeze(-1)
                            sb_r.append((x_in, wm))
                            active_hssd.append((ad_r, sb_r))

                        return _hpre

                    def _make_hpost(sb_r: list[tuple[torch.Tensor, torch.Tensor | None]] = st_box):
                        def _hpost(_m: nn.Module, _i: Any, _o: Any) -> None:
                            if sb_r:
                                sb_r.pop()
                            if active_hssd:
                                active_hssd.pop()

                        return _hpost

                    attn.register_forward_pre_hook(_make_hpre(), with_kwargs=True)
                    attn.register_forward_hook(_make_hpost())

                    wrapped_any = False
                    for attr in ("chunk_gated_delta_rule", "fused_recurrent_gated_delta_rule"):
                        if callable(getattr(attn, attr, None)):
                            wrapped_any = True
                            setattr(
                                attn,
                                attr,
                                lambda q, k, v, *a, _ad=had, _sb=st_box, **kw: _recurrent_hssd(
                                    _ad, _sb[-1][0], _sb[-1][1], q, k, v, *a, **kw
                                ),
                            )
                    if not wrapped_any:
                        mname = type(attn).__module__
                        hmod = sys.modules.get(mname)
                        if hmod is not None and mname not in patched_mods:
                            for fn_nm in (
                                "chunk_gated_delta_rule",
                                "torch_chunk_gated_delta_rule",
                                "fused_recurrent_gated_delta_rule",
                                "torch_recurrent_gated_delta_rule",
                            ):
                                orig_fn = getattr(hmod, fn_nm, None)
                                if callable(orig_fn):
                                    patched_mods.add(mname)

                                    def _mdisp(q, k, v, *a, _orig: Callable[..., Any] = orig_fn, **kw):
                                        if not active_hssd:
                                            return _orig(q, k, v, *a, **kw)
                                        cad, csb = active_hssd[-1]
                                        return _recurrent_hssd(cad, csb[-1][0], csb[-1][1], q, k, v, *a, **kw)

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
                if isinstance(w_mod, torch.Tensor) and w_mod.shape == (e, flat_x.shape[-1]):
                    b_mod = getattr(mod, "bias", None)
                    b_up = _upcast(b_mod) if isinstance(b_mod, torch.Tensor) else None
                    base_l = F.linear(_upcast(flat_x), _upcast(w_mod), b_up)
                elif is_probs:
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
            expected_bb = set(pmap.keys())
            tie_we = bool(getattr(config, "tie_word_embeddings", False))

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
                        unexpected_bb.append(k)
                        continue
                    if tuple(target.shape) != tuple(t.shape):
                        raise RuntimeError(
                            f"shape mismatch for {k}: model {tuple(target.shape)} vs shard {tuple(t.shape)}"
                        )
                    with torch.no_grad():
                        if dt_override is None and target.dtype != t.dtype:
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
