"""Comprehensive tests for direct zero-spike NVFP4/FP8 safetensors loading & Whole Model Pack parity.

Covers:
  1. Unsloth `NVFP4` Standard mixed multi-shard layout (`L0..2` `moe_nvfp4` with distinct `gate_proj`
     `weight_global_scale=31104.0` vs `up_proj` `weight_global_scale=19840.0`, `L3` channel-FP8 `moe_fp8`,
     `shared_expert` NVFP4/FP8, channel-FP8 `self_attn`/`linear_attn`/`lm_head`, `{k,v}_scale`,
     `input_global_scale`, `model.language_model.*` prefix, and unread/deleted `shard-visual-mtp.safetensors`
     under `text_only=True`).
  2. Unsloth `NVFP4-Fast` all-NVFP4 layout with cross-shard tensor splits (including scale/global_scale
     arriving before `weight_packed`) and `keep_dense_quantized=True` (`lm_head` always dequantized).
  3. NVIDIA ModelOpt `weight_scale_2` multiplier layout (`quant_method="modelopt"`).
  4. Remote Hugging Face Hub shard filtering (`hf_hub_download` never downloads visual/MTP-only shards).
  5. `qloop.train.loss` (`lm_loss`, `token_logprobs`, `token_entropy`) safety on quantized `lm_head`.
  6. Full E2E `Loop + HSSD v3.2 + D-LSA + HotExpertDeltas + lm_loss` step + `export_whole_model_pack` +
     `Qwen36LoopHSSDForCausalLM.from_pretrained` (native & standalone `_HAS_QLOOP=False`) reload parity
     with zero dense 3D MoE allocation during `from_pretrained`.
"""
from __future__ import annotations

import copy
import importlib.util
import json
import math
import shutil
import sys
import types
from pathlib import Path
from typing import Any

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from qloop.data.tokenizer import ToyTokenizer
from qloop.export_pack import SAFETENSORS_INDEX_FILE, export_whole_model_pack
from qloop.infer import CONFIG_FILENAME, MODELING_FILENAME, Qwen36LoopHSSDForCausalLM
from qloop.infer.hf_custom_modeling import CausalLMPackOutput, save_safetensors_file
from qloop.model.experts import attach_hot_expert_deltas
from qloop.model.hssd import HSSDConfig, apply_hssd
from qloop.model.introspect import find_text_model, inspect_model
from qloop.model.loading import load_model
from qloop.model.loop import LoopSpec, apply_loop
from qloop.model.lsa import LSAConfig, install_lsa
from qloop.model.quant_autograd import (
    FP8_DTYPES,
    dequantize_fp8_tensor,
    dequantize_nvfp4_tensor,
    install_quantized_autograd_bridge,
    is_quantized_module,
)
from qloop.train.loss import lm_loss, token_entropy, token_logprobs
from qloop.train.stack import Stack

_HAS_FP8 = bool(FP8_DTYPES)
_E2M1_POS = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32)


def _cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    af = a.detach().float().reshape(-1)
    bf = b.detach().float().reshape(-1)
    return float(F.cosine_similarity(af.unsqueeze(0), bf.unsqueeze(0)).item())


# ============================================================================ exact NVFP4 / FP8 test encoders


def _encode_nvfp4_2d(
    weight: torch.Tensor,
    *,
    global_scale: float,
    mode: str = "divisor",
    group_size: int = 16,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Encode 2D `[M, K]` float tensor into `(packed_u8, scale_fp8, global_scale_f32, exact_ref_f32)`."""
    assert _HAS_FP8, "PyTorch float8_e4m3fn required"
    fp8_dt = torch.float8_e4m3fn
    wf = weight.detach().float()
    m, k = wf.shape
    assert k % group_size == 0 and group_size % 2 == 0
    n_grp = k // group_size
    wg = wf.reshape(m, n_grp, group_size)
    amax = wg.abs().amax(dim=-1)  # [M, n_grp]
    # Under divisor: real = fp4 * scale / gs  =>  scale = (amax / 6.0) * gs
    # Under multiplier: real = fp4 * scale * gs  =>  scale = (amax / 6.0) / gs
    raw_scale = (amax / 6.0) * global_scale if mode == "divisor" else (amax / 6.0) / global_scale
    scale_fp8 = raw_scale.clamp(1e-6, 448.0).to(fp8_dt)
    eff_group_scale = (
        scale_fp8.float() / float(global_scale)
        if mode == "divisor"
        else scale_fp8.float() * float(global_scale)
    )
    scaled = wg / eff_group_scale.unsqueeze(-1).clamp_min(1e-12)
    signs = (scaled < 0).to(torch.uint8) << 3
    mags = scaled.abs()
    pos_grid = _E2M1_POS.to(device=wf.device)
    idx = (mags.unsqueeze(-1) - pos_grid).abs().argmin(dim=-1).to(torch.uint8)
    codes = (signs | idx).reshape(m, k)
    lo = codes[:, 0::2] & 0x0F
    hi = (codes[:, 1::2] & 0x0F) << 4
    packed_u8 = (lo | hi).contiguous()
    gs_tensor = torch.tensor(float(global_scale), dtype=torch.float32)
    ref_f32 = dequantize_nvfp4_tensor(
        packed_u8,
        scale_fp8,
        gs_tensor,
        dtype=torch.float32,
        group_size=group_size,
        global_scale_mode=mode,
    )
    return packed_u8, scale_fp8, gs_tensor, ref_f32


def _encode_fp8_channel_2d(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Encode 2D `[M, K]` float tensor into channel-wise `(w_fp8 [M, K], scale_bf16 [M, 1], exact_ref_f32)`."""
    assert _HAS_FP8, "PyTorch float8_e4m3fn required"
    fp8_dt = torch.float8_e4m3fn
    wf = weight.detach().float()
    amax = wf.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8)
    scale_bf16 = (amax / 448.0).to(torch.bfloat16)
    scale_f32 = scale_bf16.float().clamp_min(1e-12)
    w_fp8 = (wf / scale_f32).clamp(-448.0, 448.0).to(fp8_dt)
    ref_f32 = dequantize_fp8_tensor(w_fp8, scale_bf16, dtype=torch.float32)
    return w_fp8, scale_bf16, ref_f32


# ============================================================================ faithful Qwen3_5Moe HF fixture


def _install_qwen35_moe_transformers_shim(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    """Install a faithful `transformers==5.23` `Qwen3_5MoeForCausalLM` shim when `transformers` is absent."""
    if importlib.util.find_spec("transformers") is not None:
        import transformers  # type: ignore[import-not-found]

        return transformers

    tf_mod = types.ModuleType("transformers")
    tf_mod.__version__ = "5.23.0"
    # Give __spec__ a valid ModuleSpec so `_should_attempt_direct_safetensors_load` treats it as real HF
    tf_mod.__spec__ = importlib.machinery.ModuleSpec("transformers", loader=None)

    modeling_utils_mod = types.ModuleType("transformers.modeling_utils")
    masking_utils_mod = types.ModuleType("transformers.masking_utils")

    def _sdpa_attention_forward(
        module: nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        scaling: float | None = None,
        **_kwargs: Any,
    ) -> tuple[torch.Tensor, None]:
        _b, hq, tq, dh = query.shape
        tk = key.shape[2]
        hk = key.shape[1]
        if hq != hk:
            key = key.repeat_interleave(hq // hk, dim=1)
            value = value.repeat_interleave(hq // hk, dim=1)
        sc = (dh ** -0.5) if scaling is None else float(scaling)
        scores = torch.matmul(query.float(), key.float().transpose(-1, -2)) * sc
        qpos = torch.arange(tk - tq, tk, device=query.device)[:, None]
        kpos = torch.arange(tk, device=query.device)[None, :]
        causal = kpos <= qpos
        scores = scores.masked_fill(~causal[None, None, :, :], float("-inf"))
        if isinstance(attention_mask, torch.Tensor) and attention_mask.ndim == 4:
            scores = scores + attention_mask.float()
        probs = torch.softmax(scores, dim=-1).to(value.dtype)
        out = torch.matmul(probs, value).transpose(1, 2).contiguous()
        return out, None

    ALL_ATTENTION_FUNCTIONS: dict[str, Any] = {
        "sdpa": _sdpa_attention_forward,
        "eager": _sdpa_attention_forward,
    }
    ALL_MASK_ATTENTION_FUNCTIONS: dict[str, Any] = {
        "sdpa": lambda *_a, **_k: None,
        "eager": lambda *_a, **_k: None,
    }

    class AttentionInterface:
        @staticmethod
        def register(name: str, fn: Any) -> None:
            ALL_ATTENTION_FUNCTIONS[name] = fn

    class AttentionMaskInterface:
        @staticmethod
        def register(name: str, fn: Any) -> None:
            ALL_MASK_ATTENTION_FUNCTIONS[name] = fn

    modeling_utils_mod.ALL_ATTENTION_FUNCTIONS = ALL_ATTENTION_FUNCTIONS  # type: ignore[attr-defined]
    masking_utils_mod.ALL_MASK_ATTENTION_FUNCTIONS = ALL_MASK_ATTENTION_FUNCTIONS  # type: ignore[attr-defined]

    class PretrainedConfig:
        model_type: str = ""

        def __init__(self, **kwargs: Any) -> None:
            self.tie_word_embeddings = bool(kwargs.pop("tie_word_embeddings", False))
            self._attn_implementation = str(kwargs.pop("_attn_implementation", "sdpa") or "sdpa")
            for k, v in kwargs.items():
                setattr(self, k, v)

        def to_dict(self) -> dict[str, Any]:
            out: dict[str, Any] = {"model_type": self.model_type}
            for k, v in vars(self).items():
                if k.startswith("__") or callable(v):
                    continue
                if isinstance(v, PretrainedConfig):
                    out[k] = v.to_dict()
                else:
                    out[k] = copy.deepcopy(v)
            return out

        @classmethod
        def from_pretrained(cls, pretrained_model_name_or_path: str | Path, **_kwargs: Any) -> Any:
            cfg_p = Path(pretrained_model_name_or_path) / "config.json"
            raw = json.loads(cfg_p.read_text(encoding="utf-8"))
            if cls is PretrainedConfig:
                return AutoConfig.for_model(raw.get("model_type", "qwen3_5_moe"), **raw)
            return cls(**raw)

    class Qwen3_5MoeTextConfig(PretrainedConfig):
        model_type = "qwen3_5_moe_text"

        def __init__(
            self,
            vocab_size: int = 128,
            hidden_size: int = 64,
            num_hidden_layers: int = 4,
            layer_types: list[str] | None = None,
            num_attention_heads: int = 4,
            num_key_value_heads: int = 2,
            head_dim: int = 16,
            partial_rotary_factor: float = 0.5,
            rope_theta: float = 10000.0,
            max_position_embeddings: int = 4096,
            linear_num_key_heads: int = 2,
            linear_num_value_heads: int = 4,
            linear_key_head_dim: int = 16,
            linear_value_head_dim: int = 16,
            linear_conv_kernel_dim: int = 4,
            moe_intermediate_size: int = 32,
            shared_expert_intermediate_size: int = 32,
            num_experts: int = 4,
            num_experts_per_tok: int = 2,
            norm_topk_prob: bool = True,
            rms_norm_eps: float = 1e-6,
            **kwargs: Any,
        ) -> None:
            kwargs.pop("model_type", None)
            super().__init__(**kwargs)
            self.vocab_size = int(vocab_size)
            self.hidden_size = int(hidden_size)
            self.num_hidden_layers = int(num_hidden_layers)
            self.layer_types = list(
                layer_types
                if layer_types is not None
                else [
                    "full_attention" if (i + 1) % 4 == 0 else "linear_attention"
                    for i in range(self.num_hidden_layers)
                ]
            )
            self.num_attention_heads = int(num_attention_heads)
            self.num_key_value_heads = int(num_key_value_heads)
            self.head_dim = int(head_dim)
            self.partial_rotary_factor = float(partial_rotary_factor)
            self.rope_theta = float(rope_theta)
            self.max_position_embeddings = int(max_position_embeddings)
            self.linear_num_key_heads = int(linear_num_key_heads)
            self.linear_num_value_heads = int(linear_num_value_heads)
            self.linear_key_head_dim = int(linear_key_head_dim)
            self.linear_value_head_dim = int(linear_value_head_dim)
            self.linear_conv_kernel_dim = int(linear_conv_kernel_dim)
            self.moe_intermediate_size = int(moe_intermediate_size)
            self.shared_expert_intermediate_size = int(shared_expert_intermediate_size)
            self.num_experts = int(num_experts)
            self.num_experts_per_tok = int(num_experts_per_tok)
            self.norm_topk_prob = bool(norm_topk_prob)
            self.rms_norm_eps = float(rms_norm_eps)
            if not hasattr(self, "rope_parameters") or self.rope_parameters is None:
                self.rope_parameters = {
                    "rope_type": "default",
                    "rope_theta": self.rope_theta,
                    "partial_rotary_factor": self.partial_rotary_factor,
                }

    class Qwen3_5MoeConfig(PretrainedConfig):
        model_type = "qwen3_5_moe"

        def __init__(
            self,
            text_config: dict[str, Any] | Qwen3_5MoeTextConfig | None = None,
            vision_config: dict[str, Any] | None = None,
            **kwargs: Any,
        ) -> None:
            kwargs.pop("model_type", None)
            super().__init__(**kwargs)
            if isinstance(text_config, Qwen3_5MoeTextConfig):
                self.text_config = text_config
            elif isinstance(text_config, dict):
                self.text_config = Qwen3_5MoeTextConfig(**text_config)
            else:
                self.text_config = Qwen3_5MoeTextConfig()
            self.vision_config = dict(vision_config or {"hidden_size": 32, "depth": 1})

    class PreTrainedModel(nn.Module):
        config_class: Any = PretrainedConfig

        def __init__(self, config: Any) -> None:
            super().__init__()
            self.config = config

    class Qwen3_5MoeRMSNorm(nn.Module):
        """Zero-centered `(1 + weight)` RMSNorm matching HF `Qwen3_5MoeRMSNorm`."""

        def __init__(self, dim: int, eps: float = 1e-6) -> None:
            super().__init__()
            self.eps = float(eps)
            self.weight = nn.Parameter(torch.zeros(dim))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            xf = x.float()
            normed = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
            return (normed * (1.0 + self.weight.float())).to(x.dtype)

    class Qwen3_5MoeRMSNormGated(nn.Module):
        def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.ones(hidden_size))
            self.variance_epsilon = float(eps)

        def forward(self, hidden_states: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
            dt = hidden_states.dtype
            hf = hidden_states.float()
            variance = hf.pow(2).mean(-1, keepdim=True)
            hf = hf * torch.rsqrt(variance + self.variance_epsilon)
            hf = self.weight.float() * hf * F.silu(gate.float())
            return hf.to(dt)

    class Qwen3_5MoeTextRotaryEmbedding(nn.Module):
        def __init__(self, config: Qwen3_5MoeTextConfig, device: Any = None) -> None:
            super().__init__()
            self.config = config
            rp = getattr(config, "rope_parameters", None) or getattr(config, "rope_scaling", None) or {}
            prf = float(rp.get("partial_rotary_factor", getattr(config, "partial_rotary_factor", 0.5)))
            theta = float(rp.get("rope_theta", getattr(config, "rope_theta", 10000.0)))
            rot_dim = int(config.head_dim * prf)
            inv_freq = 1.0 / (
                theta ** (torch.arange(0, rot_dim, 2, dtype=torch.float32, device=device) / float(rot_dim))
            )
            rope_type = str(rp.get("rope_type", "default"))
            factor = float(rp.get("factor", 1.0))
            if rope_type == "yarn" and factor > 1.0:
                inv_freq = inv_freq / factor
                self.attention_scaling = 0.1 * math.log(factor) + 1.0
            else:
                self.attention_scaling = 1.0
            self.register_buffer("inv_freq", inv_freq, persistent=False)
            self.register_buffer("original_inv_freq", inv_freq.clone(), persistent=False)

        def forward(self, x: torch.Tensor, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            freqs = position_ids[..., None].float() * self.inv_freq.float()
            emb = torch.cat([freqs, freqs], dim=-1)
            cos = (emb.cos() * self.attention_scaling).to(dtype=x.dtype)
            sin = (emb.sin() * self.attention_scaling).to(dtype=x.dtype)
            return cos, sin

    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([-x2, x1], dim=-1)

    def _apply_partial_rope(
        q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        rot_dim = cos.shape[-1]
        c = cos.unsqueeze(1)
        s = sin.unsqueeze(1)
        qr, qp = q[..., :rot_dim], q[..., rot_dim:]
        kr, kp = k[..., :rot_dim], k[..., rot_dim:]
        qo = torch.cat([(qr * c) + (_rotate_half(qr) * s), qp], dim=-1)
        ko = torch.cat([(kr * c) + (_rotate_half(kr) * s), kp], dim=-1)
        return qo, ko

    def _torch_recurrent_gated_delta_rule(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        initial_state: torch.Tensor | None = None,
        output_final_state: bool = False,
        use_qk_l2norm_in_kernel: bool = False,
        scale: float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        dt = query.dtype
        qf, kf, vf = query.float(), key.float(), value.float()
        hv = vf.shape[2]
        if qf.shape[2] != hv:
            qf = qf.repeat_interleave(hv // qf.shape[2], dim=2)
        if kf.shape[2] != hv:
            kf = kf.repeat_interleave(hv // kf.shape[2], dim=2)
        if use_qk_l2norm_in_kernel:
            qf = qf * torch.rsqrt((qf * qf).sum(-1, keepdim=True) + 1e-6)
            kf = kf * torch.rsqrt((kf * kf).sum(-1, keepdim=True) + 1e-6)
        qf = qf * ((qf.shape[-1] ** -0.5) if scale is None else float(scale))
        b, t, _, dk = kf.shape
        dv = vf.shape[-1]
        st = qf.new_zeros((b, hv, dk, dv)) if initial_state is None else initial_state.float().clone()
        gf = g.float()
        bf = beta.float()
        outs = []
        for ti in range(t):
            st = st * gf[:, ti].exp().unsqueeze(-1).unsqueeze(-1)
            kt = kf[:, ti]
            vt = vf[:, ti]
            bt = bf[:, ti].unsqueeze(-1)
            kv_mem = (st * kt.unsqueeze(-1)).sum(dim=-2)
            delta = (vt - kv_mem) * bt
            st = st + kt.unsqueeze(-1) * delta.unsqueeze(-2)
            outs.append((st * qf[:, ti].unsqueeze(-1)).sum(dim=-2))
        out = torch.stack(outs, dim=1).to(dt)
        return out, (st if output_final_state else None)

    class Qwen3_5MoeGatedDeltaNet(nn.Module):
        def __init__(self, config: Qwen3_5MoeTextConfig, layer_idx: int) -> None:
            super().__init__()
            self.hidden_size = config.hidden_size
            self.num_v_heads = config.linear_num_value_heads
            self.num_k_heads = config.linear_num_key_heads
            self.head_k_dim = config.linear_key_head_dim
            self.head_v_dim = config.linear_value_head_dim
            self.key_dim = self.head_k_dim * self.num_k_heads
            self.value_dim = self.head_v_dim * self.num_v_heads
            self.conv_kernel_size = config.linear_conv_kernel_dim
            self.layer_idx = layer_idx
            self.conv_dim = self.key_dim * 2 + self.value_dim
            self.conv1d = nn.Conv1d(
                in_channels=self.conv_dim,
                out_channels=self.conv_dim,
                bias=False,
                kernel_size=self.conv_kernel_size,
                groups=self.conv_dim,
                padding=self.conv_kernel_size - 1,
            )
            self.in_proj_qkv = nn.Linear(self.hidden_size, self.key_dim * 2 + self.value_dim, bias=False)
            self.in_proj_z = nn.Linear(self.hidden_size, self.value_dim, bias=False)
            self.in_proj_b = nn.Linear(self.hidden_size, self.num_v_heads, bias=False)
            self.in_proj_a = nn.Linear(self.hidden_size, self.num_v_heads, bias=False)
            self.dt_bias = nn.Parameter(torch.ones(self.num_v_heads))
            a_init = torch.arange(1, self.num_v_heads + 1, dtype=torch.float32)
            self.A_log = nn.Parameter(torch.log(a_init))
            self.norm = Qwen3_5MoeRMSNormGated(self.head_v_dim, eps=config.rms_norm_eps)
            self.out_proj = nn.Linear(self.value_dim, self.hidden_size, bias=False)
            self.chunk_gated_delta_rule = _torch_recurrent_gated_delta_rule
            self.recurrent_gated_delta_rule = _torch_recurrent_gated_delta_rule

        def forward(
            self,
            hidden_states: torch.Tensor,
            cache_params: Any = None,
            attention_mask: torch.Tensor | None = None,
        ) -> torch.Tensor:
            del attention_mask
            bsz, seq_len, _ = hidden_states.shape
            mixed_qkv = self.in_proj_qkv(hidden_states).transpose(1, 2)
            z = self.in_proj_z(hidden_states).reshape(bsz, seq_len, -1, self.head_v_dim)
            b = self.in_proj_b(hidden_states)
            a = self.in_proj_a(hidden_states)
            mixed_qkv = F.silu(self.conv1d(mixed_qkv)[:, :, :seq_len]).transpose(1, 2)
            q, k, v = torch.split(mixed_qkv, [self.key_dim, self.key_dim, self.value_dim], dim=-1)
            q = q.reshape(bsz, seq_len, -1, self.head_k_dim)
            k = k.reshape(bsz, seq_len, -1, self.head_k_dim)
            v = v.reshape(bsz, seq_len, -1, self.head_v_dim)
            beta = b.sigmoid()
            g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias.float())
            init_state = None
            if cache_params is not None and hasattr(cache_params, "recurrent_states"):
                init_state = cache_params.recurrent_states[self.layer_idx]
            rule_fn = self.chunk_gated_delta_rule if seq_len > 1 else self.recurrent_gated_delta_rule
            core_out, last_state = rule_fn(
                q,
                k,
                v,
                g=g,
                beta=beta,
                initial_state=init_state,
                output_final_state=cache_params is not None,
                use_qk_l2norm_in_kernel=True,
            )
            if cache_params is not None and hasattr(cache_params, "recurrent_states"):
                cache_params.recurrent_states[self.layer_idx] = last_state
            core_flat = self.norm(core_out.reshape(-1, self.head_v_dim), z.reshape(-1, self.head_v_dim))
            return self.out_proj(core_flat.reshape(bsz, seq_len, -1))

    class Qwen3_5MoeAttention(nn.Module):
        def __init__(self, config: Qwen3_5MoeTextConfig, layer_idx: int) -> None:
            super().__init__()
            self.config = config
            self.layer_idx = layer_idx
            self.head_dim = config.head_dim
            self.num_heads = config.num_attention_heads
            self.num_kv_heads = config.num_key_value_heads
            self.scaling = self.head_dim ** -0.5
            self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim * 2, bias=False)
            self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
            self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
            self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=False)
            self.q_norm = Qwen3_5MoeRMSNorm(self.head_dim, eps=config.rms_norm_eps)
            self.k_norm = Qwen3_5MoeRMSNorm(self.head_dim, eps=config.rms_norm_eps)

        def forward(
            self,
            hidden_states: torch.Tensor,
            position_embeddings: tuple[torch.Tensor, torch.Tensor],
            attention_mask: torch.Tensor | None = None,
            past_key_values: Any = None,
            **_kwargs: Any,
        ) -> tuple[torch.Tensor, Any]:
            bsz, seq_len, _ = hidden_states.shape
            qg = self.q_proj(hidden_states).view(bsz, seq_len, self.num_heads, self.head_dim * 2)
            q, gate = torch.chunk(qg, 2, dim=-1)
            gate = gate.reshape(bsz, seq_len, -1)
            q = self.q_norm(q).transpose(1, 2)
            k = self.k_norm(self.k_proj(hidden_states).view(bsz, seq_len, self.num_kv_heads, self.head_dim)).transpose(
                1, 2
            )
            v = self.v_proj(hidden_states).view(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
            cos, sin = position_embeddings
            q, k = _apply_partial_rope(q, k, cos, sin)
            if past_key_values is not None and hasattr(past_key_values, "update"):
                k, v = past_key_values.update(k, v, self.layer_idx)
            impl_name = getattr(self.config, "_attn_implementation", "sdpa") or "sdpa"
            attn_fn = ALL_ATTENTION_FUNCTIONS.get(impl_name, _sdpa_attention_forward)
            attn_out, attn_w = attn_fn(self, q, k, v, attention_mask, scaling=self.scaling)
            attn_out = attn_out.reshape(bsz, seq_len, -1) * torch.sigmoid(gate)
            return self.o_proj(attn_out), attn_w

    class Qwen3_5MoeMLP(nn.Module):
        def __init__(self, hidden_size: int, intermediate_size: int) -> None:
            super().__init__()
            self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
            self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
            self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

    class Qwen3_5MoeExperts(nn.Module):
        def __init__(self, config: Qwen3_5MoeTextConfig) -> None:
            super().__init__()
            self.num_experts = config.num_experts
            self.hidden_dim = config.hidden_size
            self.intermediate_dim = config.moe_intermediate_size
            self.gate_up_proj = nn.Parameter(
                torch.empty(self.num_experts, 2 * self.intermediate_dim, self.hidden_dim)
            )
            self.down_proj = nn.Parameter(
                torch.empty(self.num_experts, self.hidden_dim, self.intermediate_dim)
            )
            self.act_fn = F.silu

        def forward(
            self,
            hidden_states: torch.Tensor,
            top_k_index: torch.Tensor,
            top_k_weights: torch.Tensor,
        ) -> torch.Tensor:
            final_h = torch.zeros_like(hidden_states)
            for eid in range(self.num_experts):
                tok_idx, slot_idx = torch.where(top_k_index == eid)
                if tok_idx.numel() == 0:
                    continue
                cur = hidden_states[tok_idx]
                gate, up = F.linear(cur, self.gate_up_proj[eid]).chunk(2, dim=-1)
                cur_out = F.linear(self.act_fn(gate) * up, self.down_proj[eid])
                cur_out = cur_out * top_k_weights[tok_idx, slot_idx, None].to(cur_out.dtype)
                final_h.index_add_(0, tok_idx, cur_out.to(final_h.dtype))
            return final_h

    class Qwen3_5MoeTopKRouter(nn.Module):
        def __init__(self, config: Qwen3_5MoeTextConfig) -> None:
            super().__init__()
            self.top_k = config.num_experts_per_tok
            self.num_experts = config.num_experts
            self.norm_topk_prob = config.norm_topk_prob
            self.hidden_dim = config.hidden_size
            self.weight = nn.Parameter(torch.zeros(self.num_experts, self.hidden_dim))

        def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            flat = hidden_states.reshape(-1, self.hidden_dim)
            router_logits = F.linear(flat.float(), self.weight.float())
            router_probs = F.softmax(router_logits, dim=-1, dtype=torch.float32)
            top_sc, top_idx = torch.topk(router_probs, self.top_k, dim=-1)
            if self.norm_topk_prob:
                top_sc = top_sc / top_sc.sum(dim=-1, keepdim=True)
            return router_probs.to(hidden_states.dtype), top_sc.to(hidden_states.dtype), top_idx

    class Qwen3_5MoeSparseMoeBlock(nn.Module):
        def __init__(self, config: Qwen3_5MoeTextConfig) -> None:
            super().__init__()
            self.gate = Qwen3_5MoeTopKRouter(config)
            self.experts = Qwen3_5MoeExperts(config)
            self.shared_expert = Qwen3_5MoeMLP(config.hidden_size, config.shared_expert_intermediate_size)
            self.shared_expert_gate = nn.Linear(config.hidden_size, 1, bias=False)

        def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
            bsz, seq_len, hdim = hidden_states.shape
            flat = hidden_states.view(-1, hdim)
            _, routing_weights, selected_experts = self.gate(flat)
            routed = self.experts(flat, selected_experts, routing_weights)
            shared = torch.sigmoid(self.shared_expert_gate(flat)) * self.shared_expert(flat)
            return (routed + shared).reshape(bsz, seq_len, hdim)

    class Qwen3_5MoeDecoderLayer(nn.Module):
        def __init__(self, config: Qwen3_5MoeTextConfig, layer_idx: int) -> None:
            super().__init__()
            self.layer_type = config.layer_types[layer_idx]
            if self.layer_type == "linear_attention":
                self.linear_attn = Qwen3_5MoeGatedDeltaNet(config, layer_idx)
            else:
                self.self_attn = Qwen3_5MoeAttention(config, layer_idx)
            self.mlp = Qwen3_5MoeSparseMoeBlock(config)
            self.input_layernorm = Qwen3_5MoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            self.post_attention_layernorm = Qwen3_5MoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        def forward(
            self,
            hidden_states: torch.Tensor,
            position_embeddings: tuple[torch.Tensor, torch.Tensor],
            attention_mask: torch.Tensor | None = None,
            past_key_values: Any = None,
            **_kwargs: Any,
        ) -> torch.Tensor:
            residual = hidden_states
            normed = self.input_layernorm(hidden_states)
            if self.layer_type == "linear_attention":
                h = self.linear_attn(normed, cache_params=past_key_values, attention_mask=attention_mask)
            else:
                h, _ = self.self_attn(
                    normed,
                    position_embeddings=position_embeddings,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                )
            hidden_states = residual + h
            residual = hidden_states
            hidden_states = residual + self.mlp(self.post_attention_layernorm(hidden_states))
            return hidden_states

    class _TextOutput:
        def __init__(self, last_hidden_state: torch.Tensor, past_key_values: Any = None) -> None:
            self.last_hidden_state = last_hidden_state
            self.past_key_values = past_key_values
            self.hidden_states = None

    class Qwen3_5MoeTextModel(PreTrainedModel):
        config_class = Qwen3_5MoeTextConfig

        def __init__(self, config: Qwen3_5MoeTextConfig) -> None:
            super().__init__(config)
            self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
            self.layers = nn.ModuleList(
                [Qwen3_5MoeDecoderLayer(config, i) for i in range(config.num_hidden_layers)]
            )
            self.norm = Qwen3_5MoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            self.rotary_emb = Qwen3_5MoeTextRotaryEmbedding(config)

        def forward(
            self,
            input_ids: torch.Tensor | None = None,
            attention_mask: torch.Tensor | None = None,
            position_ids: torch.Tensor | None = None,
            past_key_values: Any = None,
            inputs_embeds: torch.Tensor | None = None,
            use_cache: bool | None = None,
            **_kwargs: Any,
        ) -> _TextOutput:
            del use_cache
            h = self.embed_tokens(input_ids) if inputs_embeds is None else inputs_embeds
            bsz, seq_len, _ = h.shape
            if position_ids is None:
                past_len = past_key_values.get_seq_length() if hasattr(past_key_values, "get_seq_length") else 0
                position_ids = torch.arange(past_len, past_len + seq_len, device=h.device).unsqueeze(0).expand(bsz, -1)
            pos_emb = self.rotary_emb(h, position_ids)
            for layer in self.layers[: self.config.num_hidden_layers]:
                h = layer(
                    h,
                    position_embeddings=pos_emb,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                )
            return _TextOutput(last_hidden_state=self.norm(h), past_key_values=past_key_values)

    class Qwen3_5MoeForCausalLM(PreTrainedModel):
        config_class = Qwen3_5MoeTextConfig

        def __init__(self, config: Qwen3_5MoeTextConfig | Qwen3_5MoeConfig) -> None:
            tcfg = config.text_config if isinstance(config, Qwen3_5MoeConfig) else config
            super().__init__(tcfg)
            self.model = Qwen3_5MoeTextModel(tcfg)
            self.vocab_size = tcfg.vocab_size
            self.lm_head = nn.Linear(tcfg.hidden_size, tcfg.vocab_size, bias=False)

        def get_input_embeddings(self) -> nn.Module:
            return self.model.embed_tokens

        def get_output_embeddings(self) -> nn.Module:
            return self.lm_head

        def forward(
            self,
            input_ids: torch.Tensor | None = None,
            attention_mask: torch.Tensor | None = None,
            position_ids: torch.Tensor | None = None,
            past_key_values: Any = None,
            inputs_embeds: torch.Tensor | None = None,
            use_cache: bool | None = None,
            logits_to_keep: int | torch.Tensor = 0,
            **_kwargs: Any,
        ) -> CausalLMPackOutput:
            out = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
            )
            h = out.last_hidden_state
            slice_idx = (
                slice(-int(logits_to_keep), None)
                if isinstance(logits_to_keep, int) and logits_to_keep > 0
                else slice(None)
            )
            logits = self.lm_head(h[:, slice_idx, :])
            return CausalLMPackOutput(logits=logits, past_key_values=out.past_key_values)

    class AutoConfig:
        @staticmethod
        def for_model(model_type: str, **kwargs: Any) -> Any:
            kwargs.pop("model_type", None)
            if model_type == "qwen3_5_moe":
                return Qwen3_5MoeConfig(**kwargs)
            if model_type == "qwen3_5_moe_text":
                return Qwen3_5MoeTextConfig(**kwargs)
            raise ValueError(f"unknown model_type {model_type!r}")

        @staticmethod
        def from_pretrained(pretrained_model_name_or_path: str | Path, **_kwargs: Any) -> Any:
            cfg_p = Path(pretrained_model_name_or_path) / "config.json"
            raw = json.loads(cfg_p.read_text(encoding="utf-8"))
            mtype = str(raw.pop("model_type", "qwen3_5_moe"))
            return AutoConfig.for_model(mtype, **raw)

    class AutoModelForCausalLM:
        @staticmethod
        def from_config(config: Any, **kwargs: Any) -> nn.Module:
            attn_impl = kwargs.get("attn_implementation")
            dt = kwargs.get("dtype") or kwargs.get("torch_dtype")
            tcfg = config.text_config if isinstance(config, Qwen3_5MoeConfig) else config
            if attn_impl:
                tcfg._attn_implementation = str(attn_impl)
            prev_dt = torch.get_default_dtype()
            if isinstance(dt, torch.dtype):
                torch.set_default_dtype(dt)
            try:
                m = Qwen3_5MoeForCausalLM(tcfg)
            finally:
                torch.set_default_dtype(prev_dt)
            return m

        @staticmethod
        def from_pretrained(*_args: Any, **_kwargs: Any) -> nn.Module:
            raise AssertionError("AutoModelForCausalLM.from_pretrained must be bypassed by direct loader!")

    class AutoTokenizer:
        @staticmethod
        def from_pretrained(*_args: Any, **_kwargs: Any) -> ToyTokenizer:
            return ToyTokenizer()

    # Bind real module names onto methods so `_should_attempt_direct_safetensors_load` recognizes them
    for cls_obj in (
        PretrainedConfig,
        Qwen3_5MoeTextConfig,
        Qwen3_5MoeConfig,
        PreTrainedModel,
        Qwen3_5MoeTextRotaryEmbedding,
        Qwen3_5MoeGatedDeltaNet,
        Qwen3_5MoeAttention,
        Qwen3_5MoeTextModel,
        Qwen3_5MoeForCausalLM,
        AutoConfig,
        AutoModelForCausalLM,
        AutoTokenizer,
    ):
        cls_obj.__module__ = "transformers.models.qwen3_5_moe.modeling_qwen3_5_moe"
    AutoModelForCausalLM.from_pretrained.__module__ = "transformers.models.auto.auto_factory"  # type: ignore[attr-defined]

    tf_mod.PretrainedConfig = PretrainedConfig  # type: ignore[attr-defined]
    tf_mod.PreTrainedModel = PreTrainedModel  # type: ignore[attr-defined]
    tf_mod.GenerationMixin = object  # type: ignore[attr-defined]
    tf_mod.AttentionInterface = AttentionInterface  # type: ignore[attr-defined]
    tf_mod.AttentionMaskInterface = AttentionMaskInterface  # type: ignore[attr-defined]
    tf_mod.AutoConfig = AutoConfig  # type: ignore[attr-defined]
    tf_mod.AutoModelForCausalLM = AutoModelForCausalLM  # type: ignore[attr-defined]
    tf_mod.AutoTokenizer = AutoTokenizer  # type: ignore[attr-defined]
    tf_mod.Qwen3_5MoeTextConfig = Qwen3_5MoeTextConfig  # type: ignore[attr-defined]
    tf_mod.Qwen3_5MoeConfig = Qwen3_5MoeConfig  # type: ignore[attr-defined]
    tf_mod.Qwen3_5MoeForCausalLM = Qwen3_5MoeForCausalLM  # type: ignore[attr-defined]

    sub_qwen_mod = types.ModuleType("transformers.models.qwen3_5_moe.modeling_qwen3_5_moe")
    sub_qwen_mod._torch_recurrent_gated_delta_rule = _torch_recurrent_gated_delta_rule  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "transformers", tf_mod)
    monkeypatch.setitem(sys.modules, "transformers.modeling_utils", modeling_utils_mod)
    monkeypatch.setitem(sys.modules, "transformers.masking_utils", masking_utils_mod)
    monkeypatch.setitem(sys.modules, "transformers.models.qwen3_5_moe.modeling_qwen3_5_moe", sub_qwen_mod)
    return tf_mod


# ============================================================================ synthetic HF checkpoint builders


def _base_tiny_text_config_dict(vocab_size: int = 128) -> dict[str, Any]:
    return {
        "model_type": "qwen3_5_moe_text",
        "vocab_size": vocab_size,
        "hidden_size": 64,
        "num_hidden_layers": 4,
        "layer_types": ["linear_attention", "linear_attention", "linear_attention", "full_attention"],
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 16,
        "partial_rotary_factor": 0.5,
        "rope_theta": 10000.0,
        "max_position_embeddings": 4096,
        "linear_num_key_heads": 2,
        "linear_num_value_heads": 4,
        "linear_key_head_dim": 16,
        "linear_value_head_dim": 16,
        "linear_conv_kernel_dim": 4,
        "moe_intermediate_size": 32,
        "shared_expert_intermediate_size": 32,
        "num_experts": 4,
        "num_experts_per_tok": 2,
        "norm_topk_prob": True,
        "rms_norm_eps": 1e-6,
    }


def _seed_reference_dense_model(
    tf_mod: Any, seed: int = 1234, vocab_size: int = 128
) -> tuple[nn.Module, dict[str, Any]]:
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed + 77)
    tcfg_d = _base_tiny_text_config_dict(vocab_size=vocab_size)
    tcfg_kw = {k: v for k, v in tcfg_d.items() if k != "model_type"}
    tcfg = tf_mod.AutoConfig.for_model("qwen3_5_moe_text", **tcfg_kw)
    ref_model = tf_mod.AutoModelForCausalLM.from_config(tcfg, attn_implementation="sdpa", dtype=torch.float32)
    with torch.no_grad():
        for name, p in ref_model.named_parameters():
            if name.endswith("A_log"):
                p.copy_(torch.log(torch.arange(1, p.numel() + 1, dtype=p.dtype)))
            elif name.endswith("dt_bias"):
                p.uniform_(0.5, 1.5, generator=gen)
            elif "norm.weight" in name and "linear_attn.norm" not in name:
                p.normal_(0.0, 0.08, generator=gen)
            elif "linear_attn.norm.weight" in name:
                p.uniform_(0.8, 1.2, generator=gen)
            else:
                p.normal_(0.0, 0.12, generator=gen)
    ref_model.eval()
    return ref_model, tcfg_d


def _write_checkpoint_from_reference(
    ckpt_dir: Path,
    ref_model: nn.Module,
    tcfg_d: dict[str, Any],
    *,
    layout: str,
) -> nn.Module:
    """Write a multi-shard `.safetensors` checkpoint and return an exact-dequantized dense reference model.

    `layout`:
      - `"unsloth_standard"`: L0..2 `moe_nvfp4` (distinct `gate_proj` `wgs=31104.0` vs `up_proj` `wgs=19840.0`),
        L3 channel-FP8 `moe_fp8`, `shared_expert` NVFP4 (L0..2) & channel-FP8 (L3), channel-FP8 dense linears +
        `lm_head`, `{k,v}_scale` + `input_global_scale`, `model.language_model.*` VLM prefix, plus isolated
        `shard-visual-mtp.safetensors`.
      - `"unsloth_fast"`: L0..3 all `moe_nvfp4`, NVFP4 `shared_expert` + dense linears, cross-shard split
        including reverse scale-before-weight arrival.
      - `"modelopt"`: `quant_method="modelopt"` with `weight_scale_2` multiplier (`0.0025` / `0.004`).
    """
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    deq_ref = copy.deepcopy(ref_model).float().eval()
    ref_tm = find_text_model(deq_ref)

    shard1: dict[str, torch.Tensor] = {}
    shard2: dict[str, torch.Tensor] = {}
    shard3: dict[str, torch.Tensor] = {}
    shard_vis_mtp: dict[str, torch.Tensor] = {
        "model.visual.patch_embed.proj.weight": torch.randn(32, 3, 2, 2, dtype=torch.bfloat16),
        "mtp.layers.0.eh_proj.weight": torch.randn(64, 128, dtype=torch.bfloat16),
        "model.mtp.norm.weight": torch.ones(64, dtype=torch.bfloat16),
    }

    pfx = "model.language_model" if layout in ("unsloth_standard", "unsloth_fast") else "model"
    lm_head_pfx = "model.language_model.lm_head" if layout == "unsloth_standard" else "lm_head"

    # Embeddings & final norm
    emb_bf16 = ref_tm.embed_tokens.weight.detach().to(torch.bfloat16)
    norm_bf16 = ref_tm.norm.weight.detach().to(torch.bfloat16)
    with torch.no_grad():
        ref_tm.embed_tokens.weight.copy_(emb_bf16.float())
        ref_tm.norm.weight.copy_(norm_bf16.float())
    shard1[f"{pfx}.embed_tokens.weight"] = emb_bf16
    shard2[f"{pfx}.norm.weight"] = norm_bf16

    # lm_head (channel-FP8 in unsloth_standard/fast, NVFP4 scale_2 in modelopt)
    if layout == "modelopt":
        wp, ws, wgs, w_ref = _encode_nvfp4_2d(
            deq_ref.lm_head.weight.data, global_scale=0.0035, mode="multiplier"
        )
        shard1[f"{lm_head_pfx}.weight_packed"] = wp
        shard2[f"{lm_head_pfx}.weight_scale"] = ws
        shard3[f"{lm_head_pfx}.weight_scale_2"] = wgs
    else:
        wf, ws_ch, w_ref = _encode_fp8_channel_2d(deq_ref.lm_head.weight.data)
        shard1[f"{lm_head_pfx}.weight"] = wf
        shard2[f"{lm_head_pfx}.weight_scale"] = ws_ch
    with torch.no_grad():
        deq_ref.lm_head.weight.copy_(w_ref)

    def _quant_2d_linear(lin_mod: nn.Linear, key_base: str, use_nvfp4: bool, split_cross_shard: bool = False) -> None:
        if use_nvfp4:
            is_mopt = layout == "modelopt"
            gs_val = 0.0025 if is_mopt else 28672.0
            wp, ws, wgs, l_ref = _encode_nvfp4_2d(
                lin_mod.weight.data,
                global_scale=gs_val,
                mode="multiplier" if is_mopt else "divisor",
            )
            gs_key = "weight_scale_2" if is_mopt else "weight_global_scale"
            if split_cross_shard:
                shard1[f"{key_base}.weight_packed"] = wp
                shard2[f"{key_base}.weight_scale"] = ws
                shard3[f"{key_base}.{gs_key}"] = wgs
            else:
                shard1[f"{key_base}.weight_packed"] = wp
                shard1[f"{key_base}.weight_scale"] = ws
                shard1[f"{key_base}.{gs_key}"] = wgs
                shard1[f"{key_base}.input_global_scale"] = torch.tensor(2688.0, dtype=torch.float32)
        else:
            wf, ws_ch, l_ref = _encode_fp8_channel_2d(lin_mod.weight.data)
            shard1[f"{key_base}.weight"] = wf
            shard2[f"{key_base}.weight_scale"] = ws_ch
            shard2[f"{key_base}.weight_shape"] = torch.tensor(list(wf.shape), dtype=torch.int64)
        with torch.no_grad():
            lin_mod.weight.copy_(l_ref)

    for li, layer in enumerate(ref_tm.layers):
        lpfx = f"{pfx}.layers.{li}"
        for norm_nm in ("input_layernorm", "post_attention_layernorm"):
            nm_mod = getattr(layer, norm_nm)
            nb = nm_mod.weight.detach().to(torch.bfloat16)
            with torch.no_grad():
                nm_mod.weight.copy_(nb.float())
            shard1[f"{lpfx}.{norm_nm}.weight"] = nb

        if hasattr(layer, "linear_attn"):
            la = layer.linear_attn
            for p_nm in ("in_proj_qkv", "in_proj_z", "out_proj"):
                _quant_2d_linear(
                    getattr(la, p_nm),
                    f"{lpfx}.linear_attn.{p_nm}",
                    use_nvfp4=(layout in ("unsloth_fast", "modelopt")),
                    split_cross_shard=(layout == "unsloth_fast" and li == 0),
                )
            for dense_nm in ("in_proj_a", "in_proj_b", "conv1d", "norm"):
                dm = getattr(la, dense_nm)
                wb = dm.weight.detach().to(torch.bfloat16)
                with torch.no_grad():
                    dm.weight.copy_(wb.float())
                shard1[f"{lpfx}.linear_attn.{dense_nm}.weight"] = wb
            for param_nm in ("A_log", "dt_bias"):
                pt = getattr(la, param_nm)
                pb = pt.detach().to(torch.bfloat16)
                with torch.no_grad():
                    pt.copy_(pb.float())
                shard1[f"{lpfx}.linear_attn.{param_nm}"] = pb
        else:
            sa = layer.self_attn
            for p_nm in ("q_proj", "k_proj", "v_proj", "o_proj"):
                _quant_2d_linear(
                    getattr(sa, p_nm),
                    f"{lpfx}.self_attn.{p_nm}",
                    use_nvfp4=(layout in ("unsloth_fast", "modelopt")),
                )
            shard1[f"{lpfx}.self_attn.k_scale"] = torch.tensor(0.125, dtype=torch.float32)
            shard1[f"{lpfx}.self_attn.v_scale"] = torch.tensor(0.125, dtype=torch.float32)
            for qk_nm in ("q_norm", "k_norm"):
                qm = getattr(sa, qk_nm)
                qb = qm.weight.detach().to(torch.bfloat16)
                with torch.no_grad():
                    qm.weight.copy_(qb.float())
                shard1[f"{lpfx}.self_attn.{qk_nm}.weight"] = qb

        # Router & shared_expert_gate (dense BF16)
        mlp = layer.mlp
        for rg_nm in ("gate", "shared_expert_gate"):
            rgm = getattr(mlp, rg_nm)
            rgb = rgm.weight.detach().to(torch.bfloat16)
            with torch.no_grad():
                rgm.weight.copy_(rgb.float())
            shard1[f"{lpfx}.mlp.{rg_nm}.weight"] = rgb

        # Shared expert (NVFP4 on L0..2 of unsloth_standard and all layers of unsloth_fast/modelopt; FP8 on L3 standard)
        se_nvfp4 = not (layout == "unsloth_standard" and li == 3)
        for sp_nm in ("gate_proj", "up_proj", "down_proj"):
            _quant_2d_linear(getattr(mlp.shared_expert, sp_nm), f"{lpfx}.mlp.shared_expert.{sp_nm}", use_nvfp4=se_nvfp4)

        # Routed 3D MoE experts -> serialized as per-expert 2D tensors in safetensors!
        exp = mlp.experts
        n_exp = exp.gate_up_proj.shape[0]
        i_dim = exp.down_proj.shape[2]
        layer_is_fp8_moe = layout == "unsloth_standard" and li == 3
        for eid in range(n_exp):
            g_w = exp.gate_up_proj.data[eid, :i_dim, :]
            u_w = exp.gate_up_proj.data[eid, i_dim:, :]
            d_w = exp.down_proj.data[eid]
            epfx = f"{lpfx}.mlp.experts.{eid}"
            if layer_is_fp8_moe:
                gf, gs_ch, g_ref = _encode_fp8_channel_2d(g_w)
                uf, us_ch, u_ref = _encode_fp8_channel_2d(u_w)
                df, ds_ch, d_ref = _encode_fp8_channel_2d(d_w)
                # Put scale BEFORE weight on eid==1 to test scale-first arrival on moe_fp8 too
                if eid == 1:
                    shard1[f"{epfx}.gate_proj.weight_scale"] = gs_ch
                    shard1[f"{epfx}.up_proj.weight_scale"] = us_ch
                    shard1[f"{epfx}.down_proj.weight_scale"] = ds_ch
                    shard2[f"{epfx}.gate_proj.weight"] = gf
                    shard2[f"{epfx}.up_proj.weight"] = uf
                    shard2[f"{epfx}.down_proj.weight"] = df
                else:
                    shard2[f"{epfx}.gate_proj.weight"] = gf
                    shard2[f"{epfx}.gate_proj.weight_scale"] = gs_ch
                    shard2[f"{epfx}.up_proj.weight"] = uf
                    shard2[f"{epfx}.up_proj.weight_scale"] = us_ch
                    shard2[f"{epfx}.down_proj.weight"] = df
                    shard2[f"{epfx}.down_proj.weight_scale"] = ds_ch
            else:
                is_mopt = layout == "modelopt"
                mode = "multiplier" if is_mopt else "divisor"
                # Distinct gate vs up global_scales (31104.0 vs 19840.0) on Unsloth NVFP4!
                g_gs_val = (0.0025 + 0.0003 * eid) if is_mopt else (31104.0 + 256.0 * eid)
                u_gs_val = (0.0040 + 0.0002 * eid) if is_mopt else (19840.0 + 128.0 * eid)
                d_gs_val = (0.0030 + 0.0001 * eid) if is_mopt else (24576.0 + 192.0 * eid)
                gp, gs, g_gs, g_ref = _encode_nvfp4_2d(g_w, global_scale=g_gs_val, mode=mode)
                up, us, u_gs, u_ref = _encode_nvfp4_2d(u_w, global_scale=u_gs_val, mode=mode)
                dp, ds, d_gs, d_ref = _encode_nvfp4_2d(d_w, global_scale=d_gs_val, mode=mode)
                gs_attr = "weight_scale_2" if is_mopt else "weight_global_scale"

                for proj_nm, wp_t, ws_t, wgs_t in (
                    ("gate_proj", gp, gs, g_gs),
                    ("up_proj", up, us, u_gs),
                    ("down_proj", dp, ds, d_gs),
                ):
                    if eid == 0:
                        # Cross-shard forward order: packed in shard1, scale in shard2, global_scale in shard3
                        shard1[f"{epfx}.{proj_nm}.weight_packed"] = wp_t
                        shard2[f"{epfx}.{proj_nm}.weight_scale"] = ws_t
                        shard3[f"{epfx}.{proj_nm}.{gs_attr}"] = wgs_t
                        shard3[f"{epfx}.{proj_nm}.input_global_scale"] = torch.tensor(2688.0, dtype=torch.float32)
                    elif eid == 1:
                        # Cross-shard REVERSE order: global_scale in shard1, scale in shard2, packed in shard3!
                        shard1[f"{epfx}.{proj_nm}.{gs_attr}"] = wgs_t
                        shard2[f"{epfx}.{proj_nm}.weight_scale"] = ws_t
                        shard3[f"{epfx}.{proj_nm}.weight_packed"] = wp_t
                    else:
                        shard2[f"{epfx}.{proj_nm}.weight_packed"] = wp_t
                        shard2[f"{epfx}.{proj_nm}.weight_scale"] = ws_t
                        shard2[f"{epfx}.{proj_nm}.{gs_attr}"] = wgs_t

            with torch.no_grad():
                exp.gate_up_proj.data[eid, :i_dim, :].copy_(g_ref)
                exp.gate_up_proj.data[eid, i_dim:, :].copy_(u_ref)
                exp.down_proj.data[eid].copy_(d_ref)

    shard_Specs = [
        ("model-00001-of-00004.safetensors", shard1),
        ("model-00002-of-00004.safetensors", shard2),
        ("model-00003-of-00004.safetensors", shard3),
        ("shard-visual-mtp.safetensors", shard_vis_mtp),
    ]
    weight_map: dict[str, str] = {}
    total_bytes = 0
    for sname, stensors in shard_Specs:
        spath = ckpt_dir / sname
        total_bytes += save_safetensors_file(stensors, str(spath))
        for k in stensors:
            weight_map[k] = sname

    (ckpt_dir / SAFETENSORS_INDEX_FILE).write_text(
        json.dumps({"metadata": {"total_size": total_bytes}, "weight_map": weight_map}, indent=2),
        encoding="utf-8",
    )

    if layout == "modelopt":
        qcfg = {"quant_method": "modelopt", "quant_algo": "NVFP4"}
        root_cfg = {**tcfg_d, "quantization_config": qcfg}
    else:
        qcfg = {
            "quant_method": "compressed-tensors",
            "format": "nvfp4-pack-quantized",
            "config_groups": {
                "group_0": {
                    "targets": ["Linear"],
                    "weights": {"num_bits": 4, "type": "float", "strategy": "tensor_group", "group_size": 16},
                }
            },
        }
        root_cfg = {
            "model_type": "qwen3_5_moe",
            "text_config": copy.deepcopy(tcfg_d),
            "vision_config": {"hidden_size": 32, "depth": 1},
            "quantization_config": qcfg,
            "tie_word_embeddings": False,
        }
    (ckpt_dir / "config.json").write_text(json.dumps(root_cfg, indent=2), encoding="utf-8")
    return deq_ref


# ============================================================================ 1. Unsloth Standard Mixed NVFP4 + FP8


@pytest.mark.skipif(not _HAS_FP8, reason="PyTorch float8_e4m3fn required")
def test_unsloth_nvfp4_standard_mixed_multishard_direct_load_and_grad_parity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tf_mod = _install_qwen35_moe_transformers_shim(monkeypatch)
    ref_dense_init, tcfg_d = _seed_reference_dense_model(tf_mod, seed=101)
    ckpt_dir = tmp_path / "unsloth_nvfp4_standard"
    ref_deq_fp32 = _write_checkpoint_from_reference(ckpt_dir, ref_dense_init, tcfg_d, layout="unsloth_standard")

    # Delete the isolated visual+MTP shard on disk! `text_only=True` MUST filter `weight_map` before opening shards!
    (ckpt_dir / "shard-visual-mtp.safetensors").unlink()

    # Verify exact float32 forward + backward parity AND bfloat16 forward + backward parity
    input_ids = torch.tensor([[3, 17, 42, 88, 19, 65, 91, 12], [7, 29, 54, 11, 73, 104, 38, 61]], dtype=torch.long)

    model_fp32, _tok32, info_fp32 = load_model(
        {
            "name": str(ckpt_dir),
            "dtype": "float32",
            "device_map": "cpu",
            "attn_implementation": "sdpa",
            "text_only": True,
            "yarn_target_len": 16384,
        }
    )
    assert info_fp32["direct_safetensors_load"] is True
    assert info_fp32.get("nvfp4_moe_repack", {}).get("repacked_moe_modules", 0) == 0
    assert info_fp32["quantization"].get("repacked_moe_modules", 0) == 0
    assert info_fp32["quantization"]["moe_nvfp4_count"] == 3
    assert info_fp32["quantization"]["moe_fp8_count"] == 1

    # 0 meta parameters and 0 meta buffers anywhere in loaded model
    assert [n for n, p in model_fp32.named_parameters() if p.is_meta] == []
    assert [n for n, b in model_fp32.named_buffers() if b.is_meta] == []

    tm32 = find_text_model(model_fp32)
    assert not tm32.rotary_emb.inv_freq.is_meta
    assert not tm32.rotary_emb.original_inv_freq.is_meta
    assert int(tm32.config.max_position_embeddings) == 16384
    assert tm32.config.rope_scaling["rope_type"] == "yarn"

    # Verify distinct gate vs up global_scale preserved losslessly in `[E, 2*I, 1]`
    l0_gs = tm32.layers[0].mlp.experts.gate_up_proj_global_scale
    assert tuple(l0_gs.shape) == (4, 64, 1)
    assert torch.allclose(l0_gs[0, :32, 0], torch.full((32,), 31104.0))
    assert torch.allclose(l0_gs[0, 32:, 0], torch.full((32,), 19840.0))
    assert tm32.layers[3].mlp.experts.gate_up_proj.dtype == torch.float8_e4m3fn
    assert tuple(tm32.layers[3].mlp.experts.gate_up_proj_scale_inv.shape) == (4, 64, 1)

    # Verify `A_log` dtype matches `from_config`
    fresh_meta_cfg = copy.deepcopy(tm32.config)
    fresh_meta_cfg.quantization_config = None
    with torch.device("meta"):
        skel32 = tf_mod.AutoModelForCausalLM.from_config(
            fresh_meta_cfg, attn_implementation="sdpa", dtype=torch.float32
        )
    assert tm32.layers[0].linear_attn.A_log.dtype == find_text_model(skel32).layers[0].linear_attn.A_log.dtype

    # Apply identical YaRN config onto `ref_deq_fp32` so rotary matches
    from qloop.model.lsa import apply_yarn

    ref_tm = find_text_model(ref_deq_fp32)
    apply_yarn(ref_tm.config, 16384)
    ref_tm.rotary_emb = type(ref_tm.rotary_emb)(ref_tm.config)

    # Forward + backward input-embedding gradient parity in fp32
    emb_in_q = tm32.embed_tokens(input_ids).detach().clone().requires_grad_(True)
    emb_in_r = ref_tm.embed_tokens(input_ids).detach().clone().requires_grad_(True)
    logits_q = model_fp32(inputs_embeds=emb_in_q, use_cache=False).logits
    logits_r = ref_deq_fp32(inputs_embeds=emb_in_r, use_cache=False).logits
    assert _cosine_sim(logits_q, logits_r) > 0.99999
    assert (logits_q - logits_r).abs().max().item() < 1e-5

    grad_seed = torch.randn_like(logits_r)
    (logits_q * grad_seed).mean().backward()
    (logits_r * grad_seed).mean().backward()
    assert emb_in_q.grad is not None and emb_in_r.grad is not None
    assert _cosine_sim(emb_in_q.grad, emb_in_r.grad) > 0.99999
    assert (emb_in_q.grad - emb_in_r.grad).abs().max().item() < 1e-5
    assert torch.allclose(emb_in_q.grad, emb_in_r.grad, rtol=1e-5, atol=1e-5)

    # Also verify bfloat16 load (`target_dtype = bfloat16`), `A_log` fp32 dtype match, and `bf16` parity
    model_bf16, *_ = load_model(
        {
            "name": str(ckpt_dir),
            "dtype": "bfloat16",
            "device_map": "cpu",
            "attn_implementation": "sdpa",
            "text_only": True,
            "yarn_target_len": 16384,
        }
    )
    tm_bf16 = find_text_model(model_bf16)
    with torch.device("meta"):
        skel_bf16 = tf_mod.AutoModelForCausalLM.from_config(
            fresh_meta_cfg, attn_implementation="sdpa", dtype=torch.bfloat16
        )
    assert tm_bf16.layers[0].linear_attn.A_log.dtype == torch.float32
    assert tm_bf16.layers[0].linear_attn.A_log.dtype == find_text_model(skel_bf16).layers[0].linear_attn.A_log.dtype
    ref_deq_bf16 = copy.deepcopy(ref_deq_fp32).to(dtype=torch.bfloat16).eval()
    for lyr in find_text_model(ref_deq_bf16).layers:
        if hasattr(lyr, "linear_attn"):
            lyr.linear_attn.A_log.data = lyr.linear_attn.A_log.data.float()
    emb_q16 = find_text_model(model_bf16).embed_tokens(input_ids).detach().clone().requires_grad_(True)
    emb_r16 = find_text_model(ref_deq_bf16).embed_tokens(input_ids).detach().clone().requires_grad_(True)
    lq16 = model_bf16(inputs_embeds=emb_q16, use_cache=False).logits
    lr16 = ref_deq_bf16(inputs_embeds=emb_r16, use_cache=False).logits
    assert _cosine_sim(lq16, lr16) > 0.99999
    assert (lq16.float() - lr16.float()).abs().mean().item() < 1e-3
    assert (lq16.float() - lr16.float()).abs().max().item() <= 0.016  # <= 1 bf16 ULP (2^-6) at |logit| ~ 2
    g16 = grad_seed.to(dtype=torch.bfloat16)
    (lq16 * g16).mean().backward()
    (lr16 * g16).mean().backward()
    assert emb_q16.grad is not None and emb_r16.grad is not None
    assert _cosine_sim(emb_q16.grad, emb_r16.grad) > 0.9999
    assert (emb_q16.grad.float() - emb_r16.grad.float()).abs().mean().item() < 1e-3
    assert (emb_q16.grad.float() - emb_r16.grad.float()).abs().max().item() <= 0.004  # <= 1 bf16 ULP (2^-8)


# ============================================================================ 2. Unsloth NVFP4-Fast + Cross-Shard


@pytest.mark.skipif(not _HAS_FP8, reason="PyTorch float8_e4m3fn required")
def test_unsloth_nvfp4_fast_cross_shard_and_keep_dense_quantized(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tf_mod = _install_qwen35_moe_transformers_shim(monkeypatch)
    ref_dense_init, tcfg_d = _seed_reference_dense_model(tf_mod, seed=202)
    ckpt_dir = tmp_path / "unsloth_nvfp4_fast"
    ref_deq_fp32 = _write_checkpoint_from_reference(ckpt_dir, ref_dense_init, tcfg_d, layout="unsloth_fast")
    # Corrupt visual+MTP shard with invalid bytes to prove `text_only=True` never opens it
    (ckpt_dir / "shard-visual-mtp.safetensors").write_bytes(b"NOT_A_VALID_SAFETENSORS_HEADER")

    model_q, _tok_q, info_q = load_model(
        {
            "name": str(ckpt_dir),
            "dtype": "float32",
            "device_map": "cpu",
            "attn_implementation": "sdpa",
            "text_only": True,
            "keep_dense_quantized": True,
        }
    )
    assert info_q["direct_safetensors_load"] is True
    assert info_q["quantization"]["moe_nvfp4_count"] == 4
    assert info_q["quantization"]["moe_fp8_count"] == 0
    assert info_q["quantization"]["linear_nvfp4_count"] > 0

    # Even with `keep_dense_quantized=True`, `lm_head` MUST always be dequantized to `target_dtype`!
    assert not is_quantized_module(model_q.lm_head)
    assert model_q.lm_head.weight.dtype == torch.float32
    assert is_quantized_module(find_text_model(model_q).layers[0].linear_attn.in_proj_qkv)

    ids = torch.tensor([[5, 22, 67, 89, 14, 33]], dtype=torch.long)
    emb_q = find_text_model(model_q).embed_tokens(ids).detach().clone().requires_grad_(True)
    emb_r = find_text_model(ref_deq_fp32).embed_tokens(ids).detach().clone().requires_grad_(True)
    lq = model_q(inputs_embeds=emb_q, use_cache=False).logits
    lr = ref_deq_fp32(inputs_embeds=emb_r, use_cache=False).logits
    assert _cosine_sim(lq, lr) > 0.99999
    assert (lq - lr).abs().max().item() < 1e-5
    lq.mean().backward()
    lr.mean().backward()
    assert emb_q.grad is not None and emb_r.grad is not None
    assert _cosine_sim(emb_q.grad, emb_r.grad) > 0.99999
    assert (emb_q.grad - emb_r.grad).abs().max().item() < 1e-5
    assert torch.allclose(emb_q.grad, emb_r.grad, rtol=1e-5, atol=1e-5)


# ============================================================================ 3. ModelOpt `weight_scale_2` Multiplier


@pytest.mark.skipif(not _HAS_FP8, reason="PyTorch float8_e4m3fn required")
def test_modelopt_weight_scale_2_multiplier_direct_load_parity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tf_mod = _install_qwen35_moe_transformers_shim(monkeypatch)
    ref_dense_init, tcfg_d = _seed_reference_dense_model(tf_mod, seed=303)
    ckpt_dir = tmp_path / "modelopt_nvfp4"
    ref_deq_fp32 = _write_checkpoint_from_reference(ckpt_dir, ref_dense_init, tcfg_d, layout="modelopt")
    (ckpt_dir / "shard-visual-mtp.safetensors").unlink()

    model_mopt, _tok_m, info_mopt = load_model(
        {
            "name": str(ckpt_dir),
            "dtype": "float32",
            "device_map": "cpu",
            "attn_implementation": "sdpa",
            "text_only": True,
        }
    )
    assert info_mopt["direct_safetensors_load"] is True
    tm = find_text_model(model_mopt)
    for layer in tm.layers:
        assert layer.mlp.experts._qloop_nvfp4_scale_mode == "multiplier"

    ids = torch.tensor([[9, 41, 18, 76, 52, 103]], dtype=torch.long)
    emb_q = tm.embed_tokens(ids).detach().clone().requires_grad_(True)
    emb_r = find_text_model(ref_deq_fp32).embed_tokens(ids).detach().clone().requires_grad_(True)
    lq = model_mopt(inputs_embeds=emb_q, use_cache=False).logits
    lr = ref_deq_fp32(inputs_embeds=emb_r, use_cache=False).logits
    assert _cosine_sim(lq, lr) > 0.99999
    assert (lq - lr).abs().max().item() < 1e-5
    lq.sum().backward()
    lr.sum().backward()
    assert emb_q.grad is not None and emb_r.grad is not None
    assert _cosine_sim(emb_q.grad, emb_r.grad) > 0.99999


# ============================================================================ 4. Remote Hub Pre-Download Shard Filter


@pytest.mark.skipif(not _HAS_FP8, reason="PyTorch float8_e4m3fn required")
def test_remote_hf_hub_shard_filtering_never_downloads_visual_or_mtp_shards(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tf_mod = _install_qwen35_moe_transformers_shim(monkeypatch)
    ref_dense_init, tcfg_d = _seed_reference_dense_model(tf_mod, seed=404)
    remote_store = tmp_path / "mock_hf_hub_repo"
    _write_checkpoint_from_reference(remote_store, ref_dense_init, tcfg_d, layout="unsloth_standard")

    requested_files: list[str] = []
    hf_hub_mod = types.ModuleType("huggingface_hub")

    def _mock_hf_hub_download(*, repo_id: str, filename: str, **_kwargs: Any) -> str:
        assert repo_id == "unsloth/Qwen3.6-35B-A3B-NVFP4"
        if filename == "shard-visual-mtp.safetensors":
            raise AssertionError("visual/MTP-only shard must NEVER be downloaded when text_only=True!")
        requested_files.append(filename)
        target = remote_store / filename
        if not target.is_file():
            raise FileNotFoundError(filename)
        return str(target)

    hf_hub_mod.hf_hub_download = _mock_hf_hub_download  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "huggingface_hub", hf_hub_mod)

    orig_cfg_from_pretrained = tf_mod.AutoConfig.from_pretrained

    def _remote_cfg_from_pretrained(name: str | Path, **kw: Any) -> Any:
        if str(name) == "unsloth/Qwen3.6-35B-A3B-NVFP4":
            return orig_cfg_from_pretrained(remote_store, **kw)
        return orig_cfg_from_pretrained(name, **kw)

    monkeypatch.setattr(tf_mod.AutoConfig, "from_pretrained", staticmethod(_remote_cfg_from_pretrained))

    model, _tok_r, info = load_model(
        {
            "name": "unsloth/Qwen3.6-35B-A3B-NVFP4",
            "dtype": "bfloat16",
            "device_map": "cpu",
            "attn_implementation": "sdpa",
            "text_only": True,
        }
    )
    assert info["direct_safetensors_load"] is True
    assert "shard-visual-mtp.safetensors" not in requested_files
    assert SAFETENSORS_INDEX_FILE in requested_files
    assert len([f for f in requested_files if f.endswith(".safetensors")]) == 3
    assert [n for n, p in model.named_parameters() if p.is_meta] == []


# ============================================================================ 5. `loss.py` Quantized `lm_head` Safety


@pytest.mark.skipif(not _HAS_FP8, reason="PyTorch float8_e4m3fn required")
def test_loss_functions_handle_quantized_lm_head_module_and_bridged_tensor() -> None:
    torch.manual_seed(505)
    vocab, hidden = 96, 64
    raw_w = torch.randn(vocab, hidden, dtype=torch.float32) * 0.15

    # Case A: `linear_nvfp4` `lm_head` module
    wp, ws, wgs, w_ref_nv = _encode_nvfp4_2d(raw_w, global_scale=24576.0, mode="divisor")
    nv_head = nn.Linear(hidden, vocab, bias=False)
    nv_head._parameters.pop("weight", None)
    nv_head.register_parameter("weight_packed", nn.Parameter(wp, requires_grad=False))
    nv_head.register_parameter("weight_scale", nn.Parameter(ws, requires_grad=False))
    nv_head.register_parameter("weight_global_scale", nn.Parameter(wgs, requires_grad=False))
    nv_head.group_size = 16  # type: ignore[attr-defined]
    nv_head._qloop_nvfp4_scale_mode = "divisor"  # type: ignore[attr-defined]
    install_quantized_autograd_bridge(nv_head)

    # Case B: `linear_fp8` `lm_head` module (AND passing `fp8_head.weight` Tensor directly!)
    wf, ws_ch, w_ref_fp8 = _encode_fp8_channel_2d(raw_w)
    fp8_head = nn.Linear(hidden, vocab, bias=False)
    fp8_head.weight = nn.Parameter(wf, requires_grad=False)
    fp8_head.register_buffer("weight_scale_inv", ws_ch)
    install_quantized_autograd_bridge(fp8_head)

    labels = torch.tensor([[4, 19, 33, -100, 71, 12], [8, 52, 61, 14, 27, 90]], dtype=torch.long)

    for q_arg, w_ref in ((nv_head, w_ref_nv), (fp8_head, w_ref_fp8), (fp8_head.weight, w_ref_fp8)):
        for impl_name in ("chunked", "naive"):
            h_q = torch.randn(2, 6, hidden, requires_grad=True)
            h_r = h_q.detach().clone().requires_grad_(True)
            loss_q, n_q = lm_loss(h_q, q_arg, labels, impl=impl_name, chunk_size=3)
            loss_r, n_r = lm_loss(h_r, w_ref, labels, impl=impl_name, chunk_size=3)
            assert int(n_q.item()) == int(n_r.item())
            assert torch.allclose(loss_q, loss_r, atol=1e-5, rtol=1e-5)
            loss_q.backward()
            loss_r.backward()
            assert h_q.grad is not None and h_r.grad is not None
            assert (h_q.grad - h_r.grad).abs().max().item() < 1e-5

        h_eval = torch.randn(2, 6, hidden)
        tgt_ids = torch.clamp(labels, min=0)
        lp_q = token_logprobs(h_eval, q_arg, tgt_ids, chunk_size=3)
        lp_r = token_logprobs(h_eval, w_ref, tgt_ids, chunk_size=3)
        assert torch.allclose(lp_q, lp_r, atol=1e-5, rtol=1e-5)

        ent_q = token_entropy(h_eval, q_arg, chunk_size=3)
        ent_r = token_entropy(h_eval, w_ref, chunk_size=3)
        assert torch.allclose(ent_q, ent_r, atol=1e-5, rtol=1e-5)


# ============================================================================ 6. Full E2E Train Step + Pack Reload


def _load_emitted_standalone_without_qloop(pack_dir: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    real_find_spec = importlib.util.find_spec

    def _hide_qloop(name: str, package: str | None = None) -> Any:
        if name == "qloop" or name.startswith("qloop."):
            return None
        return real_find_spec(name, package)

    monkeypatch.setattr(importlib.util, "find_spec", _hide_qloop)
    pkg_name = f"_nvfp4_standalone_pack_{abs(hash(str(pack_dir)))}"
    pkg = types.ModuleType(pkg_name)
    pkg.__path__ = [str(pack_dir)]  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, pkg_name, pkg)

    cfg_spec = importlib.util.spec_from_file_location(
        f"{pkg_name}.configuration_qwen36_loophssd", pack_dir / CONFIG_FILENAME
    )
    assert cfg_spec is not None and cfg_spec.loader is not None
    cfg_mod = importlib.util.module_from_spec(cfg_spec)
    monkeypatch.setitem(sys.modules, cfg_spec.name, cfg_mod)
    cfg_spec.loader.exec_module(cfg_mod)

    mod_spec = importlib.util.spec_from_file_location(
        f"{pkg_name}.modeling_qwen36_loophssd", pack_dir / MODELING_FILENAME
    )
    assert mod_spec is not None and mod_spec.loader is not None
    mod_mod = importlib.util.module_from_spec(mod_spec)
    monkeypatch.setitem(sys.modules, mod_spec.name, mod_mod)
    mod_spec.loader.exec_module(mod_mod)
    assert mod_mod._HAS_QLOOP is False
    return mod_mod


@pytest.mark.skipif(not _HAS_FP8, reason="PyTorch float8_e4m3fn required")
def test_mixed_nvfp4_e2e_loop_hssd_dlsa_hot_experts_step_and_pack_reload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tok = ToyTokenizer()
    tf_mod = _install_qwen35_moe_transformers_shim(monkeypatch)
    ref_dense_init, tcfg_d = _seed_reference_dense_model(tf_mod, seed=606, vocab_size=tok.vocab_size)
    ckpt_dir = tmp_path / "mixed_nvfp4_e2e_ckpt"
    _write_checkpoint_from_reference(ckpt_dir, ref_dense_init, tcfg_d, layout="unsloth_standard")
    (ckpt_dir / "shard-visual-mtp.safetensors").unlink()

    model, _tok_e, load_info = load_model(
        {
            "name": str(ckpt_dir),
            "dtype": "bfloat16",
            "device_map": "cpu",
            "attn_implementation": "sdpa",
            "text_only": True,
            "yarn_target_len": 16384,
        }
    )
    assert load_info["direct_safetensors_load"] is True

    # 1) Install HSSD v3.2 on linear_attention layers (0, 1, 2)
    hssd_h = apply_hssd(model, HSSDConfig(num_pages=4, k_pages=2, rank=8, chunk_size=4, window_chunks=2))
    # 2) Install sparse D-LSA on full_attention layer (3) with multi-pass query projections
    lsa_h = install_lsa(
        model,
        LSAConfig(
            mode="sparse",
            index_dim=16,
            block_size=4,
            pool=2,
            avg_bucket_keys=2,
            max_bucket_keys=4,
            num_select=2,
            window=4,
            sinks=2,
            query_chunk=4,
            num_query_passes=2,
        ),
    )
    # 3) Apply 2-pass Sandwich Loop (prelude L0 | core L1,L2 x 2 | coda L3 -> 6 virtual layers)
    loop_h = apply_loop(
        model,
        LoopSpec(
            prelude_end=1,
            core_end=3,
            num_passes=2,
            target_virtual_layers=6,
            kv_share=True,
            gate_init=0.25,
            bridge_rank=8,
            step_embedding=True,
            per_pass_router=True,
            router_adapter_rank=8,
            router_repel_init=1.0,
        ),
    )
    # 4) Attach HotExpertDeltas on BOTH `moe_nvfp4` layer 1 AND `moe_fp8` layer 3!
    hot_map = {1: [0, 2], 3: [1, 3]}
    deltas_h = attach_hot_expert_deltas(model, hot_map, rank=4, alpha=4.0, dtype=torch.float32)

    # Freeze base weights; train only surgical adapters (`hssd`, `lsa`, `loop_adapters`, `expert_deltas`)
    trainable_params: list[nn.Parameter] = []
    for pname, p in model.named_parameters():
        is_surg = any(
            tag in pname for tag in ("hssd_adapter.", "lsa_indexer.", "loop_adapters.", "expert_deltas.")
        )
        p.requires_grad_(is_surg)
        if is_surg:
            trainable_params.append(p)
    assert len(trainable_params) > 0

    # Nudge zero-init projections slightly so 1 real gradient step moves every subsystem nontrivially
    gen = torch.Generator().manual_seed(777)
    with torch.no_grad():
        for ad in hssd_h.adapters.values():
            ad.erase_up.weight.normal_(0.0, 0.05, generator=gen)
            ad.write_up.weight.normal_(0.0, 0.05, generator=gen)
            ad.swa_v_up.weight.normal_(0.0, 0.05, generator=gen)
        for ix in lsa_h.indexers.values():
            ix.wk.normal_(0.0, 0.12, generator=gen)
            ix.wq.normal_(0.0, 0.12, generator=gen)
        for ld in deltas_h.deltas.values():
            for p in ld.gate_up_B.values():
                p.normal_(0.0, 0.06, generator=gen)
            for p in ld.down_B.values():
                p.normal_(0.0, 0.06, generator=gen)
        if loop_h.adapters.pass_router is not None:
            loop_h.adapters.pass_router.up.normal_(0.0, 0.15, generator=gen)

    # Execute a real forward + `lm_loss` + `backward()` + `AdamW.step()` on the mixed NVFP4/FP8 looped model!
    opt = torch.optim.AdamW(trainable_params, lr=5e-3)
    train_ids = torch.tensor([[4, 19, 37, 82, 51, 63, 28, 94, 15, 72, 44, 108]], dtype=torch.long)
    tm_live = find_text_model(model)
    opt.zero_grad()
    h_out = tm_live(input_ids=train_ids, use_cache=False).last_hidden_state
    loss, _ntok = lm_loss(h_out, model.lm_head, train_ids, chunk_size=4)
    assert torch.isfinite(loss) and float(loss.item()) < 10.0
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().max().item() > 0 for p in deltas_h.parameters())
    assert any(p.grad is not None and p.grad.abs().max().item() > 0 for p in loop_h.adapters.parameters())
    opt.step()
    opt.zero_grad(set_to_none=True)
    model.eval()

    # Capture live reference logits at R=1 and R=2
    eval_ids = torch.tensor([[6, 23, 58, 91, 14, 77, 39, 102, 48, 65, 19, 84]], dtype=torch.long)
    live_logits: dict[int, torch.Tensor] = {}
    with torch.no_grad():
        for r in (1, 2):
            loop_h.set_active_passes(r)
            live_logits[r] = model(input_ids=eval_ids, use_cache=False).logits.detach().float().clone()
        loop_h.set_active_passes(2)

    # Export Whole Model Pack
    stack = Stack(
        model=model,
        tokenizer=tok,
        parts=inspect_model(model),
        loop=loop_h,
        lsa=lsa_h,
        deltas=deltas_h,
        hssd=hssd_h,
        load_info=load_info,
        hot_map=hot_map,
    )
    pack_dir = tmp_path / "exported_mixed_nvfp4_pack"
    export_whole_model_pack(stack, pack_dir, max_shard_size_bytes=32768)

    pack_cfg = json.loads((pack_dir / "config.json").read_text(encoding="utf-8"))
    q_arch = pack_cfg["qloop_v3_arch"]["quantization"]
    assert q_arch["moe_nvfp4_count"] == 3
    assert q_arch["moe_fp8_count"] == 1
    assert pack_cfg["qloop_v3_arch"]["hot_experts"]["enabled"] is True
    assert pack_cfg["qloop_v3_arch"]["hot_experts"]["merged_into_base"] is False

    # Spy on `AutoModelForCausalLM.from_config` to prove 3D MoE experts are built on `"meta"` (0 60-GiB bloat!)
    moe_init_devices: list[str] = []
    orig_from_config = tf_mod.AutoModelForCausalLM.from_config

    def _spy_from_config(*a: Any, **kw: Any) -> nn.Module:
        m = orig_from_config(*a, **kw)
        for sub_n, sub_m in m.named_modules():
            if sub_n.endswith(".mlp.experts") and hasattr(sub_m, "gate_up_proj"):
                moe_init_devices.append(sub_m.gate_up_proj.device.type)
        return m

    monkeypatch.setattr(tf_mod.AutoModelForCausalLM, "from_config", staticmethod(_spy_from_config))

    # Reload via native `Qwen36LoopHSSDForCausalLM.from_pretrained`
    reloaded_native = Qwen36LoopHSSDForCausalLM.from_pretrained(str(pack_dir), device_map="cpu", strict=True)
    assert moe_init_devices == ["meta", "meta", "meta", "meta"]
    assert [n for n, p in reloaded_native.named_parameters() if p.is_meta] == []
    assert [n for n, b in reloaded_native.named_buffers() if b.is_meta] == []

    with torch.no_grad():
        for r in (1, 2):
            with reloaded_native.depth(r):
                nat_r = reloaded_native(eval_ids, use_cache=False).logits.float()
                assert _cosine_sim(nat_r, live_logits[r]) > 0.99999
                assert (nat_r - live_logits[r]).abs().max().item() < 1e-4

    # Reload via standalone emitted `modeling_qwen36_loophssd.py` (`_HAS_QLOOP == False`)!
    moe_init_devices.clear()
    st_mod = _load_emitted_standalone_without_qloop(pack_dir, monkeypatch)
    reloaded_standalone = st_mod.Qwen36LoopHSSDForCausalLM.from_pretrained(
        str(pack_dir), device_map="cpu", strict=True
    )
    assert moe_init_devices == ["meta", "meta", "meta", "meta"]
    assert [n for n, p in reloaded_standalone.named_parameters() if p.is_meta] == []
    assert [n for n, b in reloaded_standalone.named_buffers() if b.is_meta] == []

    with torch.no_grad():
        for r in (1, 2):
            with reloaded_standalone.depth(r):
                st_r = reloaded_standalone(eval_ids, use_cache=False).logits.float()
                assert _cosine_sim(st_r, live_logits[r]) > 0.99999
                assert (st_r - live_logits[r]).abs().max().item() < 1e-3


# ============================================================================ 7. Adapter Grad + Tied + Strict Check


@pytest.mark.skipif(not _HAS_FP8, reason="PyTorch float8_e4m3fn required")
def test_adapter_grad_parity_tie_embeddings_and_incomplete_expert_rejection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tf_mod = _install_qwen35_moe_transformers_shim(monkeypatch)
    ref_dense_init, tcfg_d = _seed_reference_dense_model(tf_mod, seed=808)
    ckpt_dir = tmp_path / "adapter_grad_and_tie_ckpt"
    ref_deq_fp32 = _write_checkpoint_from_reference(ckpt_dir, ref_dense_init, tcfg_d, layout="unsloth_standard")
    (ckpt_dir / "shard-visual-mtp.safetensors").unlink()

    # 1) Surgical adapter (`HotExpertDeltas` + `HSSD`) exact gradient parity vs dense reference in float32
    model_q, _, _ = load_model(
        {
            "name": str(ckpt_dir),
            "dtype": "float32",
            "device_map": "cpu",
            "attn_implementation": "sdpa",
            "text_only": True,
        }
    )
    for p in ref_deq_fp32.parameters():
        p.requires_grad_(False)

    hot_map = {0: [1, 3], 3: [0, 2]}
    dq = attach_hot_expert_deltas(model_q, hot_map, rank=4, alpha=4.0, dtype=torch.float32)
    dr = attach_hot_expert_deltas(ref_deq_fp32, hot_map, rank=4, alpha=4.0, dtype=torch.float32)
    gen = torch.Generator().manual_seed(909)
    with torch.no_grad():
        for ld_q, ld_r in zip(dq.deltas.values(), dr.deltas.values()):
            for k in ld_q.gate_up_A:
                ld_q.gate_up_A[k].normal_(0.0, 0.05, generator=gen)
                ld_r.gate_up_A[k].copy_(ld_q.gate_up_A[k])
                ld_q.gate_up_B[k].normal_(0.0, 0.05, generator=gen)
                ld_r.gate_up_B[k].copy_(ld_q.gate_up_B[k])
                ld_q.down_A[k].normal_(0.0, 0.05, generator=gen)
                ld_r.down_A[k].copy_(ld_q.down_A[k])
                ld_q.down_B[k].normal_(0.0, 0.05, generator=gen)
                ld_r.down_B[k].copy_(ld_q.down_B[k])

    ids = torch.tensor([[8, 31, 64, 19, 102, 47]], dtype=torch.long)
    loss_q, _ = lm_loss(find_text_model(model_q)(input_ids=ids).last_hidden_state, model_q.lm_head, ids, chunk_size=3)
    loss_r, _ = lm_loss(
        find_text_model(ref_deq_fp32)(input_ids=ids).last_hidden_state, ref_deq_fp32.lm_head, ids, chunk_size=3
    )
    assert torch.allclose(loss_q, loss_r, rtol=1e-5, atol=1e-5)
    loss_q.backward()
    loss_r.backward()
    for (nq, pq), (nr, pr) in zip(dq.named_parameters(), dr.named_parameters()):
        assert nq == nr
        assert pq.grad is not None and pr.grad is not None
        assert _cosine_sim(pq.grad, pr.grad) > 0.99999
        assert (pq.grad - pr.grad).abs().max().item() < 1e-5

    # 2) `tie_word_embeddings=True` when `lm_head.*` is absent from safetensors shards
    tie_dir = tmp_path / "tied_ckpt"
    shutil.copytree(ckpt_dir, tie_dir)
    idx_data = json.loads((tie_dir / SAFETENSORS_INDEX_FILE).read_text(encoding="utf-8"))
    idx_data["weight_map"] = {k: v for k, v in idx_data["weight_map"].items() if "lm_head" not in k}
    (tie_dir / SAFETENSORS_INDEX_FILE).write_text(json.dumps(idx_data), encoding="utf-8")
    # Also strip `lm_head.*` from all safetensors shards on disk
    from qloop.infer.hf_custom_modeling import load_safetensors_file

    for sp in sorted(tie_dir.glob("*.safetensors")):
        s_filtered = {k: v for k, v in load_safetensors_file(str(sp)).items() if "lm_head" not in k}
        save_safetensors_file(s_filtered, str(sp))
    cfg_raw = json.loads((tie_dir / "config.json").read_text(encoding="utf-8"))
    cfg_raw["tie_word_embeddings"] = True
    cfg_raw["text_config"]["tie_word_embeddings"] = True
    (tie_dir / "config.json").write_text(json.dumps(cfg_raw), encoding="utf-8")

    model_tied, _, _ = load_model(
        {
            "name": str(tie_dir),
            "dtype": "bfloat16",
            "device_map": "cpu",
            "attn_implementation": "sdpa",
            "text_only": True,
        }
    )
    assert model_tied.lm_head.weight.data_ptr() == find_text_model(model_tied).embed_tokens.weight.data_ptr()

    # 3) Missing expert slice in shard strictly raises `QloopError` / `RuntimeError`
    bad_dir = tmp_path / "incomplete_expert_ckpt"
    shutil.copytree(tie_dir, bad_dir)
    s2_path = bad_dir / "model-00002-of-00004.safetensors"
    s2_bad = {
        k: v
        for k, v in load_safetensors_file(str(s2_path)).items()
        if k != "model.language_model.layers.0.mlp.experts.2.down_proj.weight_packed"
    }
    save_safetensors_file(s2_bad, str(s2_path))
    with pytest.raises(Exception, match="incomplete MoE weights"):
        load_model({"name": str(bad_dir), "dtype": "bfloat16", "device_map": "cpu", "text_only": True})

