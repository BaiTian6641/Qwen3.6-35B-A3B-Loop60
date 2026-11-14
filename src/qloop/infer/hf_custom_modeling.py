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
     the HF base backbone, installs standalone `GDN-2` + `LSA` + `Sandwich Loop` (`R=1/2/3`), and
     streams sharded `.safetensors` weights cleanly.
"""
from __future__ import annotations

import copy
import importlib.util
import json
import math
import os
import struct
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

try:
    from .configuration_qwen36_loophssd import Qwen36LoopHSSDConfig
except ImportError:
    from configuration_qwen36_loophssd import Qwen36LoopHSSDConfig  # type: ignore[no-redef]


_HAS_QLOOP = importlib.util.find_spec("qloop") is not None

if _HAS_QLOOP:
    from qloop.infer.hf_custom_modeling import Qwen36LoopHSSDForCausalLM  # noqa: F401
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
            xf = x.float()
            h = F.silu(F.linear(xf, self.down.weight.float()))
            de = (self.cap_g * torch.tanh(F.linear(h, self.erase_up.weight.float()))).view(b, t, self.hv, self.dk)
            dw = (self.cap_g * torch.tanh(F.linear(h, self.write_up.weight.float()))).view(b, t, self.hv, self.dv)
            dd = (self.cap_d * torch.tanh(F.linear(h, self.decay_up.weight.float()))).view(b, t, self.hv, self.dk)
            bf = beta.float().clamp(0.0, 1.0).unsqueeze(-1)
            return g.float().clamp(max=0.0).unsqueeze(-1) * dd.exp(), _logistic_shift(bf, de), _logistic_shift(bf, dw)

    def _recurrent_gdn2(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, gv: torch.Tensor,
                        bg: torch.Tensor, wg: torch.Tensor, init_st: torch.Tensor | None,
                        out_final: bool, use_l2: bool, scale: float | None):
        dt = q.dtype
        qf, kf, vf = q.float(), k.float(), v.float()
        hv = vf.shape[2]
        if qf.shape[2] != hv:
            qf = qf.repeat_interleave(hv // qf.shape[2], dim=2)
        if kf.shape[2] != hv:
            kf = kf.repeat_interleave(hv // kf.shape[2], dim=2)
        if use_l2:
            qf = qf * torch.rsqrt((qf * qf).sum(-1, keepdim=True) + 1e-6)
            kf = kf * torch.rsqrt((kf * kf).sum(-1, keepdim=True) + 1e-6)
        qf = qf * ((qf.shape[-1] ** -0.5) if scale is None else float(scale))
        b, t, _, dk = kf.shape
        dv = vf.shape[-1]
        st = qf.new_zeros(b, hv, dk, dv) if init_st is None else init_st.float().clone()
        outs = []
        for i in range(t):
            st = st * gv[:, i].exp().unsqueeze(-1)
            kt, vt = kf[:, i], vf[:, i]
            mem = (st * (bg[:, i] * kt).unsqueeze(-1)).sum(-2)
            st = st + kt.unsqueeze(-1) * ((wg[:, i] * vt) - mem).unsqueeze(-2)
            outs.append((st * qf[:, i].unsqueeze(-1)).sum(-2))
        return torch.stack(outs, dim=1).to(dt), (st if out_final else None)

    class _StandaloneLSAIndexer(nn.Module):
        def __init__(self, nh: int, nkv: int, hd: int, rot: int, idx_dim: int, index_on: str, n_q: int = 1) -> None:
            super().__init__()
            lo = rot if index_on == "nope" else 0
            din = hd - lo
            self.wk = nn.Parameter(torch.zeros(nkv, din, idx_dim))
            self.wq = nn.Parameter(torch.zeros(nh, din, idx_dim))
            if int(n_q) > 1:
                self.q_proj_extra = nn.ParameterList(
                    [nn.Parameter(torch.zeros(nh, din, idx_dim)) for _ in range(int(n_q) - 1)]
                )

    class _StandaloneBridge(nn.Module):
        def __init__(self, hidden: int, extra: int, init_w: tuple[float, ...], rank: int,
                     step_emb: bool, mode: str = "concat") -> None:
            super().__init__()
            s = sum(init_w)
            logits = torch.tensor([math.log(w / s) for w in init_w])
            self.mix_logits = nn.Parameter(logits.repeat(extra, 1))
            self.step_emb = nn.Parameter(torch.zeros(extra, hidden)) if step_emb else None
            self.rank = int(rank)
            self.adapter_mode = mode
            if self.rank > 0:
                if mode == "subspace":
                    self.norms = nn.ModuleList([nn.RMSNorm(hidden) for _ in range(extra)])
                    self.down = nn.ModuleList([nn.Linear(hidden, self.rank, bias=False) for _ in range(extra)])
                    self.up = nn.ModuleList([nn.Linear(self.rank, hidden, bias=False) for _ in range(extra)])
                else:
                    self.down = nn.Linear(2 * hidden, self.rank, bias=False)
                    self.up = nn.Linear(self.rank, hidden, bias=False)

        def forward(self, h: torch.Tensor, anchor: torch.Tensor, p: int) -> torch.Tensor:
            w = F.softmax(self.mix_logits[p].float(), dim=-1)
            rh = h.float().pow(2).mean(-1, keepdim=True).add(1e-6).sqrt()
            ra = anchor.float().pow(2).mean(-1, keepdim=True).add(1e-6).sqrt()
            x = w[0] * h.float() + w[1] * anchor.float() + w[2] * (h.float() * (ra / rh))
            if self.step_emb is not None:
                x = x + self.step_emb[p].float()
            if self.rank > 0:
                if self.adapter_mode == "subspace":
                    u = self.norms[p](h.to(self.down[p].weight.dtype))
                    du = self.up[p](F.silu(self.down[p](u)))
                    x = x + (ra / math.sqrt(h.shape[-1])) * du.float()
                else:
                    cat = torch.cat([h, anchor], dim=-1).to(self.down.weight.dtype)
                    x = x + self.up(F.silu(self.down(cat))).float()
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

    class _StandaloneLoopAdapters(nn.Module):
        def __init__(self, hidden: int, spec: dict[str, Any], n_exp: int | None) -> None:
            super().__init__()
            n_passes = int(spec.get("num_passes", 2))
            extra = max(n_passes - 1, 1)
            core_len = int(spec.get("core_end", 32)) - int(spec.get("prelude_end", 8))
            self.gates = nn.Parameter(torch.zeros(extra, hidden))
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
            self.mor_router = nn.Linear(hidden, n_passes) if spec.get("mor", False) else None

    class _SharedKVProxy:
        def __init__(self, cache: Any) -> None:
            object.__setattr__(self, "_cache", cache)

        def __getattr__(self, name: str) -> Any:
            return getattr(object.__getattribute__(self, "_cache"), name)

        def update(self, k: torch.Tensor, v: torch.Tensor, layer_idx: int, *a: Any, **kw: Any) -> Any:
            return object.__getattribute__(self, "_cache").update(k[..., :0, :], v[..., :0, :], layer_idx, *a, **kw)

    class _StandaloneVirtualLayer(nn.Module):
        def __init__(self, owner: Qwen36LoopHSSDForCausalLM, vidx: int, phys: int, r: int,
                     entry: bool, exit_flag: bool, cache_idx: int, ltype: str, inner: nn.Module) -> None:
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
                ow._loop_ctx = {"anchor": None, "start": {}, "kv": {}, "routing": {}}
            if r > ow.active_passes:
                return (hidden_states,) if ow._tuple_out else hidden_states
            ctx = ow._loop_ctx
            x = hidden_states
            if entry:
                if r == 1:
                    ctx["anchor"] = x
                    ctx["start"][1] = x
                else:
                    ctx["start"][r] = x
                    x = ow._loop_adapters.bridge(x, ctx["anchor"], r - 2)
            pkv = kwargs.get("past_key_values")
            if pkv is not None and cidx != vidx:
                kwargs = dict(kwargs)
                kwargs["past_key_values"] = _SharedKVProxy(pkv)
            prev = ow._cur_pass
            ow._cur_pass = r
            for m in self._idx_mods:
                m.layer_idx = cidx
            try:
                out = self.inner(x, *args, **kwargs)
            finally:
                ow._cur_pass = prev
            is_tup = isinstance(out, tuple)
            ow._tuple_out = is_tup
            h = out[0] if is_tup else out
            if exit_flag and r >= 2:
                st = ctx["start"][r]
                g = ow._loop_adapters.gates[r - 2].to(h.dtype)
                h = st + g * (h - st)
            return (h, *out[1:]) if is_tup else h

    class Qwen36LoopHSSDForCausalLM(_BaseCls):  # type: ignore[no-redef,misc]
        config_class = Qwen36LoopHSSDConfig

        def __init__(self, config: Qwen36LoopHSSDConfig, *, _defer_loop: bool = False) -> None:
            super().__init__(config)
            self.config = config
            arch = config.qloop_v3_arch or {}
            bcfg = copy.deepcopy(dict(config.base_config or {}))
            mtype = str(arch.get("base_model_type") or bcfg.pop("model_type", "qwen3_5_moe"))
            bcfg.pop("model_type", None)
            attn_impl = str(bcfg.pop("_attn_implementation", "sdpa") or "sdpa")
            if attn_impl == "qloop_lsa":
                attn_impl = "sdpa"
            hf_cfg = AutoConfig.for_model(mtype, **bcfg)
            self.backbone = AutoModelForCausalLM.from_config(hf_cfg, attn_implementation=attn_impl)
            self._cur_pass = 0
            self._tuple_out = False
            self._loop_ctx: dict[str, Any] = {}
            self._loop_adapters: _StandaloneLoopAdapters | None = None
            self.active_passes = 1
            self._install_submodules(arch, hf_cfg)
            if not _defer_loop:
                self._apply_standalone_loop(arch)

        def _find_tm(self) -> nn.Module:
            for _n, mod in self.backbone.named_modules():
                if isinstance(getattr(mod, "layers", None), nn.ModuleList) and hasattr(mod, "embed_tokens"):
                    return mod
            raise RuntimeError("could not locate text_model on backbone")

        def _install_submodules(self, arch: dict[str, Any], hf_cfg: Any) -> None:
            tm = self._find_tm()
            gcfg = dict(arch.get("gdn2") or {})
            if gcfg.get("enabled"):
                rank = int(gcfg.get("rank", 64))
                cap_g = float(gcfg.get("gate_logit_cap", 8.0))
                cap_d = float(gcfg.get("decay_log_cap", 1.0))
                for i, layer in enumerate(tm.layers):
                    attn = getattr(layer, "linear_attn", None)
                    if attn is None or (gcfg.get("layers") is not None and i not in gcfg["layers"]):
                        continue
                    hs = int(attn.out_proj.out_features)
                    hv = int(getattr(attn, "num_v_heads", None) or attn.A_log.numel())
                    dv = int(getattr(attn, "head_v_dim", None) or (attn.out_proj.in_features // hv))
                    dk = int(getattr(attn, "head_k_dim", None) or dv)
                    ad = _StandaloneGDN2Adapter(hs, hv, dk, dv, rank, cap_g, cap_d)
                    attn.gdn2_adapter = ad
                    x_box: list[torch.Tensor] = []
                    attn.register_forward_pre_hook(
                        lambda _m, a, kw, xb=x_box: xb.append(a[0] if a else kw["hidden_states"]),
                        with_kwargs=True,
                    )
                    attn.register_forward_hook(lambda _m, _i, _o, xb=x_box: xb.pop() if xb else None)
                    for attr in ("chunk_gated_delta_rule", "fused_recurrent_gated_delta_rule"):
                        if callable(getattr(attn, attr, None)):
                            setattr(
                                attn,
                                attr,
                                lambda q, k, v, g, beta, initial_state=None, output_final_state=False,
                                use_qk_l2norm_in_kernel=False, scale=None, _ad=ad, _xb=x_box, **_kw: _recurrent_gdn2(
                                    q, k, v, *_ad.gates(_xb[-1], g, beta), initial_state,
                                    output_final_state, use_qk_l2norm_in_kernel, scale,
                                ),
                            )
            lcfg = dict(arch.get("lsa") or {})
            if lcfg.get("enabled"):
                hd = getattr(hf_cfg, "head_dim", None) or (hf_cfg.hidden_size // hf_cfg.num_attention_heads)
                prf = getattr(hf_cfg, "partial_rotary_factor", 0.25)
                rot = int(lcfg.get("rotary_dim") or int(hd * prf))
                for i, layer in enumerate(tm.layers):
                    attn = getattr(layer, "self_attn", None)
                    if attn is None or (lcfg.get("layers") is not None and i not in lcfg["layers"]):
                        continue
                    attn.lsa_indexer = _StandaloneLSAIndexer(
                        hf_cfg.num_attention_heads,
                        hf_cfg.num_key_value_heads,
                        hd,
                        rot,
                        int(lcfg.get("index_dim", 64)),
                        str(lcfg.get("index_on", "nope")),
                        int(lcfg.get("num_query_passes", 1)),
                    )

        def _apply_standalone_loop(self, arch: dict[str, Any]) -> None:
            lcfg = dict(arch.get("loop") or {})
            if not lcfg.get("enabled"):
                return
            tm = self._find_tm()
            orig_layers = tm.layers
            n_phys = len(orig_layers)
            p_end, c_end = int(lcfg.get("prelude_end", 8)), int(lcfg.get("core_end", 32))
            n_passes = int(lcfg.get("num_passes", 2))
            n_exp = getattr(tm.config, "num_experts", None)
            self._loop_adapters = _StandaloneLoopAdapters(tm.config.hidden_size, lcfg, n_exp)
            tm.loop_adapters = self._loop_adapters
            self.active_passes = int(lcfg.get("active_passes", n_passes))
            if lcfg.get("kv_share", True) and n_passes > 1:
                for phys in range(p_end, c_end):
                    s_attn = getattr(orig_layers[phys], "self_attn", None)
                    if s_attn is None:
                        continue
                    for nm in ("k_proj", "v_proj"):
                        proj = getattr(s_attn, nm, None)
                        if proj is not None:
                            proj.register_forward_hook(self._make_kv_hook(phys, nm))
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
                        self, v, i, r, r >= 1 and i == p_end, r >= 1 and i == c_end - 1, i if share else v,
                        lt, orig_layers[i],
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

        def set_active_passes(self, r: int) -> None:
            self.active_passes = int(r)

        def get_input_embeddings(self) -> nn.Module:
            return self._find_tm().embed_tokens

        def get_output_embeddings(self) -> nn.Module:
            return self.backbone.lm_head

        def forward(self, input_ids: torch.Tensor | None = None, **kwargs: Any) -> Any:
            return self.backbone(input_ids=input_ids, **kwargs)

        @classmethod
        def from_pretrained(cls, pretrained_model_name_or_path: str | os.PathLike[str], *args: Any,
                            config: Qwen36LoopHSSDConfig | None = None, token: str | None = None,
                            device_map: str | None = None, **kwargs: Any):
            del args
            kwargs.pop("trust_remote_code", None)
            local_dir = str(pretrained_model_name_or_path)
            if not os.path.isdir(local_dir):
                from huggingface_hub import snapshot_download

                local_dir = snapshot_download(repo_id=local_dir, repo_type="model", token=token)
            if config is None:
                config = Qwen36LoopHSSDConfig.from_pretrained(local_dir)
            inst = cls(config, _defer_loop=True)
            pmap = dict(inst.backbone.named_parameters())
            bmap = dict(inst.backbone.named_buffers())
            idx_p = os.path.join(local_dir, "model.safetensors.index.json")
            if os.path.isfile(idx_p):
                with open(idx_p, encoding="utf-8") as f:
                    sfiles = list(dict.fromkeys(json.load(f)["weight_map"].values()))
            else:
                sfiles = sorted(fn for fn in os.listdir(local_dir) if fn.endswith(".safetensors"))
            loop_sd: dict[str, torch.Tensor] = {}
            for sf in sfiles:
                for k, t in _load_st(os.path.join(local_dir, sf)).items():
                    if ".loop_adapters." in k:
                        loop_sd[k.split(".loop_adapters.", 1)[1]] = t
                    elif k.startswith("loop_adapters."):
                        loop_sd[k[len("loop_adapters."):]] = t
                    elif k in pmap:
                        with torch.no_grad():
                            pmap[k].copy_(t.to(pmap[k].dtype))
                    elif k in bmap:
                        with torch.no_grad():
                            bmap[k].copy_(t.to(bmap[k].dtype))
            inst._apply_standalone_loop(config.qloop_v3_arch or {})
            if inst._loop_adapters is not None and loop_sd:
                lmap = dict(inst._loop_adapters.named_parameters())
                with torch.no_grad():
                    for k, t in loop_sd.items():
                        if k in lmap:
                            lmap[k].copy_(t.to(lmap[k].dtype))
            if device_map and device_map != "auto":
                inst.to(device_map)
            elif device_map == "auto" and torch.cuda.is_available():
                inst.to("cuda:0")
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
