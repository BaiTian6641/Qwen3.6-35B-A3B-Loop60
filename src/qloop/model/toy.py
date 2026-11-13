"""Structurally faithful *toy* of HF `qwen3_5_moe` for CPU development and tests.

Mirrors the parts of the real model that the loop retrofit touches:
  * `model.model.layers` ModuleList iterated as `self.layers[: config.num_hidden_layers]`
  * per-layer `layer_type` ("linear_attention" | "full_attention"), 3:1 pattern
  * Gated DeltaNet with short causal conv + recurrent state, cached per `layer_idx`
  * gated full attention (q_proj emits query+gate), q/k RMSNorm, partial RoPE, GQA
  * sparse MoE with router `gate`, fused 3D `experts.gate_up_proj/down_proj`,
    `shared_expert` + `shared_expert_gate`
  * a hybrid cache whose slots are sized from `config.num_hidden_layers` / `layer_types`

It is intentionally slow and simple (python loop over time for DeltaNet). Real-model
equivalence is checked on Colab by `qloop surgery-check`.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import nn

from ._dtype import upcast


@dataclass
class ToyConfig:
    vocab_size: int = 97
    hidden_size: int = 32
    num_hidden_layers: int = 16
    full_attention_interval: int = 4
    layer_types: list[str] = field(default_factory=list)
    num_attention_heads: int = 4
    num_key_value_heads: int = 2
    head_dim: int = 8
    partial_rotary_factor: float = 0.5
    rope_theta: float = 10000.0
    linear_num_heads: int = 2
    linear_head_dim: int = 8
    linear_conv_kernel_dim: int = 3
    num_experts: int = 8
    num_experts_per_tok: int = 2
    moe_intermediate_size: int = 16
    shared_expert_intermediate_size: int = 16
    norm_topk_prob: bool = True
    rms_norm_eps: float = 1e-6
    model_type: str = "toy_qwen3_5_moe"
    _attn_implementation: str = "eager"

    def __post_init__(self) -> None:
        if not self.layer_types:
            self.layer_types = [
                "full_attention" if (i + 1) % self.full_attention_interval == 0 else "linear_attention"
                for i in range(self.num_hidden_layers)
            ]


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xf = upcast(x)
        xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return (xf * upcast(self.weight)).to(x.dtype)


class ToyHybridCache:
    """Per-layer slots, sized from the (possibly loop-patched) config, like HF's hybrid cache."""

    def __init__(self, config: ToyConfig):
        n = config.num_hidden_layers
        self.layer_types = list(config.layer_types)
        self.key_cache: list[torch.Tensor | None] = [None] * n
        self.value_cache: list[torch.Tensor | None] = [None] * n
        self.conv_states: list[torch.Tensor | None] = [None] * n
        self.recurrent_states: list[torch.Tensor | None] = [None] * n
        self.first_full = self.layer_types.index("full_attention")

    def update(self, k: torch.Tensor, v: torch.Tensor, layer_idx: int, cache_kwargs=None):
        if self.key_cache[layer_idx] is None:
            self.key_cache[layer_idx], self.value_cache[layer_idx] = k, v
        else:
            self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], k], dim=-2)
            self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], v], dim=-2)
        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def get_seq_length(self, layer_idx: int | None = None) -> int:
        kc = self.key_cache[self.first_full if layer_idx is None else layer_idx]
        return 0 if kc is None else kc.shape[-2]


class ToyRotary(nn.Module):
    def __init__(self, config: ToyConfig):
        super().__init__()
        self.rot_dim = int(config.head_dim * config.partial_rotary_factor)
        inv = 1.0 / (config.rope_theta ** (torch.arange(0, self.rot_dim, 2).float() / self.rot_dim))
        self.register_buffer("inv_freq", inv, persistent=False)

    def forward(self, x: torch.Tensor, position_ids: torch.Tensor):
        freqs = position_ids[..., None].to(self.inv_freq.dtype) * self.inv_freq  # [B,T,rot/2]
        emb = torch.cat([freqs, freqs], dim=-1)
        return emb.cos().to(x.dtype), emb.sin().to(x.dtype)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    a, b = x.chunk(2, dim=-1)
    return torch.cat([-b, a], dim=-1)


def apply_partial_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    r = cos.shape[-1]
    xr, xp = x[..., :r], x[..., r:]
    cos, sin = cos[:, None], sin[:, None]  # [B,1,T,r]
    return torch.cat([xr * cos + _rotate_half(xr) * sin, xp], dim=-1)


def dense_causal_attention(module, query, key, value, attention_mask=None, scaling=None, dropout=0.0, **kwargs):
    """HF attention-interface signature: q [B,H,Tq,D], k/v [B,G,Tk,D] -> ([B,Tq,H,D], None).

    Queries are the last Tq positions of the Tk keys (cache semantics, no padding).
    """
    h, g = query.shape[1], key.shape[1]
    tq, tk = query.shape[-2], key.shape[-2]
    k, v = key.repeat_interleave(h // g, dim=1), value.repeat_interleave(h // g, dim=1)
    qpos = torch.arange(tk - tq, tk, device=query.device)[:, None]
    mask = torch.arange(tk, device=query.device)[None, :] <= qpos
    o = F.scaled_dot_product_attention(query, k, v, attn_mask=mask, scale=scaling)
    return o.transpose(1, 2), None


# like transformers.AttentionInterface: config._attn_implementation -> function
TOY_ATTENTION_FUNCTIONS = {"eager": dense_causal_attention, "sdpa": dense_causal_attention}


class ToyGatedAttention(nn.Module):
    def __init__(self, config: ToyConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.h, self.kv, self.d = config.num_attention_heads, config.num_key_value_heads, config.head_dim
        self.scaling = self.d ** -0.5
        hs = config.hidden_size
        self.q_proj = nn.Linear(hs, self.h * self.d * 2, bias=False)  # query + output gate
        self.k_proj = nn.Linear(hs, self.kv * self.d, bias=False)
        self.v_proj = nn.Linear(hs, self.kv * self.d, bias=False)
        self.o_proj = nn.Linear(self.h * self.d, hs, bias=False)
        self.q_norm = RMSNorm(self.d, config.rms_norm_eps)
        self.k_norm = RMSNorm(self.d, config.rms_norm_eps)

    def forward(self, x, position_embeddings, attention_mask=None, past_key_values=None, cache_position=None):
        b, t, _ = x.shape
        qg = self.q_proj(x).view(b, t, self.h, 2 * self.d)
        q, gate = qg.chunk(2, dim=-1)
        q = self.q_norm(q).transpose(1, 2)
        k = self.k_norm(self.k_proj(x).view(b, t, self.kv, self.d)).transpose(1, 2)
        v = self.v_proj(x).view(b, t, self.kv, self.d).transpose(1, 2)
        cos, sin = position_embeddings
        q, k = apply_partial_rope(q, cos, sin), apply_partial_rope(k, cos, sin)
        if past_key_values is not None:
            k, v = past_key_values.update(k, v, self.layer_idx, {"cache_position": cache_position})
        attn_fn = TOY_ATTENTION_FUNCTIONS[self.config._attn_implementation]
        o, _ = attn_fn(self, q, k, v, attention_mask, scaling=self.scaling)
        o = o.reshape(b, t, self.h, self.d) * torch.sigmoid(gate)
        return self.o_proj(o.reshape(b, t, self.h * self.d))


class ToyGatedDeltaNet(nn.Module):
    def __init__(self, config: ToyConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.nh, self.d = config.linear_num_heads, config.linear_head_dim
        self.ck = config.linear_conv_kernel_dim
        hs, c = config.hidden_size, 3 * config.linear_num_heads * config.linear_head_dim
        self.in_proj_qkv = nn.Linear(hs, c, bias=False)
        self.in_proj_ba = nn.Linear(hs, 2 * self.nh, bias=False)
        self.conv1d = nn.Conv1d(c, c, self.ck, groups=c, bias=False)
        self.A_log = nn.Parameter(torch.zeros(self.nh).uniform_(-1.0, 0.5))
        self.dt_bias = nn.Parameter(torch.zeros(self.nh))
        self.norm = RMSNorm(self.d, config.rms_norm_eps)
        self.out_proj = nn.Linear(self.nh * self.d, hs, bias=False)

    def forward(self, x, cache_params=None, attention_mask=None):
        b, t, _ = x.shape
        qkv = self.in_proj_qkv(x).transpose(1, 2)  # [B,C,T]
        prev = cache_params.conv_states[self.layer_idx] if cache_params is not None else None
        if prev is None:
            prev = qkv.new_zeros(b, qkv.shape[1], self.ck - 1)
        seq = torch.cat([prev, qkv], dim=-1)
        if cache_params is not None:
            cache_params.conv_states[self.layer_idx] = seq[..., -(self.ck - 1):]
        qkv = F.silu(self.conv1d(seq)).transpose(1, 2)  # [B,T,C]
        q, k, v = qkv.view(b, t, 3, self.nh, self.d).unbind(2)
        q, k = F.normalize(upcast(q), dim=-1), F.normalize(upcast(k), dim=-1)
        v = upcast(v)
        bb, aa = upcast(self.in_proj_ba(x)).chunk(2, dim=-1)
        beta = torch.sigmoid(bb)
        g = -upcast(self.A_log).exp() * F.softplus(aa + upcast(self.dt_bias))  # log-decay <= 0
        state = cache_params.recurrent_states[self.layer_idx] if cache_params is not None else None
        if state is None:
            state = x.new_zeros(b, self.nh, self.d, self.d, dtype=q.dtype)
        outs = []
        for i in range(t):
            state = state * g[:, i].exp()[..., None, None]
            kt, vt = k[:, i], v[:, i]
            mem = (state * kt[..., :, None]).sum(-2)
            delta = (vt - mem) * beta[:, i][..., None]
            state = state + kt[..., :, None] * delta[..., None, :]
            outs.append((state * q[:, i][..., :, None]).sum(-2))
        if cache_params is not None:
            cache_params.recurrent_states[self.layer_idx] = state
        o = self.norm(torch.stack(outs, dim=1).to(x.dtype))
        return self.out_proj(o.reshape(b, t, self.nh * self.d))


class ToyExperts(nn.Module):
    """Fused 3D expert weights like HF `Qwen3_5MoeExperts` (no per-expert nn.Linear!)."""

    def __init__(self, config: ToyConfig):
        super().__init__()
        e, i, h = config.num_experts, config.moe_intermediate_size, config.hidden_size
        self.num_experts = e
        self.gate_up_proj = nn.Parameter(torch.randn(e, 2 * i, h) / math.sqrt(h))
        self.down_proj = nn.Parameter(torch.randn(e, h, i) / math.sqrt(i))

    def expert_forward(self, x: torch.Tensor, e: int) -> torch.Tensor:
        gate, up = F.linear(x, self.gate_up_proj[e]).chunk(2, dim=-1)
        return F.linear(F.silu(gate) * up, self.down_proj[e])

    def forward(self, x, top_idx, top_w):
        out = torch.zeros_like(x)
        for e in range(self.num_experts):
            tok, slot = (top_idx == e).nonzero(as_tuple=True)
            if tok.numel():
                out.index_add_(0, tok, self.expert_forward(x[tok], e) * top_w[tok, slot, None].to(x.dtype))
        return out


class ToyMLP(nn.Module):
    def __init__(self, h: int, i: int):
        super().__init__()
        self.gate_proj = nn.Linear(h, i, bias=False)
        self.up_proj = nn.Linear(h, i, bias=False)
        self.down_proj = nn.Linear(i, h, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class ToySparseMoeBlock(nn.Module):
    def __init__(self, config: ToyConfig):
        super().__init__()
        self.top_k, self.norm_topk = config.num_experts_per_tok, config.norm_topk_prob
        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        self.experts = ToyExperts(config)
        self.shared_expert = ToyMLP(config.hidden_size, config.shared_expert_intermediate_size)
        self.shared_expert_gate = nn.Linear(config.hidden_size, 1, bias=False)

    def forward(self, x):
        b, t, h = x.shape
        flat = x.reshape(-1, h)
        logits = self.gate(flat)
        probs = F.softmax(upcast(logits), dim=-1)
        top_w, top_idx = probs.topk(self.top_k, dim=-1)
        if self.norm_topk:
            top_w = top_w / top_w.sum(-1, keepdim=True)
        out = self.experts(flat, top_idx, top_w)
        out = out + torch.sigmoid(self.shared_expert_gate(flat)) * self.shared_expert(flat)
        return out.view(b, t, h)


class ToyDecoderLayer(nn.Module):
    def __init__(self, config: ToyConfig, layer_idx: int):
        super().__init__()
        self.layer_type = config.layer_types[layer_idx]
        if self.layer_type == "linear_attention":
            self.linear_attn = ToyGatedDeltaNet(config, layer_idx)
        else:
            self.self_attn = ToyGatedAttention(config, layer_idx)
        self.mlp = ToySparseMoeBlock(config)
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(self, hidden_states, position_embeddings=None, attention_mask=None, *, position_ids=None,
                past_key_values=None, cache_position=None, **kwargs):
        residual = hidden_states
        h = self.input_layernorm(hidden_states)
        if self.layer_type == "linear_attention":
            h = self.linear_attn(h, cache_params=past_key_values, attention_mask=attention_mask)
        else:
            h = self.self_attn(h, position_embeddings, attention_mask, past_key_values, cache_position)
        h = residual + h
        residual = h
        h = self.mlp(self.post_attention_layernorm(h))
        return residual + h


@dataclass
class ToyOutput:
    last_hidden_state: torch.Tensor
    past_key_values: ToyHybridCache | None = None
    hidden_states: tuple | None = None


class ToyTextModel(nn.Module):
    def __init__(self, config: ToyConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(ToyDecoderLayer(config, i) for i in range(config.num_hidden_layers))
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.rotary_emb = ToyRotary(config)

    def forward(self, input_ids=None, inputs_embeds=None, past_key_values=None, use_cache=False,
                output_hidden_states=False, **kwargs):
        h = self.embed_tokens(input_ids) if inputs_embeds is None else inputs_embeds
        if use_cache and past_key_values is None:
            past_key_values = ToyHybridCache(self.config)
        start = past_key_values.get_seq_length() if past_key_values is not None else 0
        cache_position = torch.arange(start, start + h.shape[1], device=h.device)
        position_ids = cache_position[None].expand(h.shape[0], -1)
        pos = self.rotary_emb(h, position_ids)
        all_h = [h] if output_hidden_states else None
        for layer in self.layers[: self.config.num_hidden_layers]:
            h = layer(h, position_embeddings=pos, attention_mask=None, position_ids=position_ids,
                      past_key_values=past_key_values, cache_position=cache_position)
            if output_hidden_states:
                all_h.append(h)
        h = self.norm(h)
        return ToyOutput(h, past_key_values, tuple(all_h) if all_h else None)


@dataclass
class ToyCausalLMOutput:
    logits: torch.Tensor
    loss: torch.Tensor | None = None
    past_key_values: ToyHybridCache | None = None


class ToyForCausalLM(nn.Module):
    def __init__(self, config: ToyConfig):
        super().__init__()
        self.config = config
        self.model = ToyTextModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, input_ids, labels=None, past_key_values=None, use_cache=False, **kwargs):
        out = self.model(input_ids=input_ids, past_key_values=past_key_values, use_cache=use_cache)
        logits = self.lm_head(out.last_hidden_state)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]).float(),
                                   labels[:, 1:].reshape(-1), ignore_index=-100)
        return ToyCausalLMOutput(logits, loss, out.past_key_values)

    @torch.no_grad()
    def greedy_generate(self, input_ids: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        out = self(input_ids, use_cache=True)
        cache, seq = out.past_key_values, input_ids
        for _ in range(max_new_tokens):
            nxt = out.logits[:, -1:].argmax(-1)
            seq = torch.cat([seq, nxt], dim=1)
            out = self(nxt, past_key_values=cache, use_cache=True)
        return seq


def build_toy(seed: int = 0, **overrides) -> ToyForCausalLM:
    torch.manual_seed(seed)
    return ToyForCausalLM(ToyConfig(**overrides)).eval()
