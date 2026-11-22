"""Regression tests for `transformers 5.24.0` `cache_params` / `past_key_values` -> `cache` rename.

Simulates strict `transformers 5.24.0` `Qwen3_5Moe*` forward signatures where:
  - `TextModel.forward(..., cache=...)`, `DecoderLayer.forward(..., cache=...)`,
    `GatedDeltaNet.forward(..., cache=...)`, and `Attention.forward(..., cache=...)`
    accept ONLY `cache=` (NOT `past_key_values=` or `cache_params=`);
  - model outputs expose ONLY `.cache` (NOT `.past_key_values`).

Also verifies backward compatibility with `transformers <= 5.23` (`cache_params`, `past_key_values`,
`past_key_value`), `ToyModel` positional `args[3]`, and `None`-value kwarg shadowing.
"""
from __future__ import annotations

import types
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

import qloop.model.hssd as hssd_mod
from qloop.eval.generate import Generator
from qloop.infer.engine import OptimizedLoopInferenceEngine
from qloop.infer.hf_custom_modeling import CausalLMPackOutput, Qwen36LoopHSSDConfig, Qwen36LoopHSSDForCausalLM
from qloop.model._cache_compat import (
    cache_kwarg_name,
    extract_cache_from_kwargs,
    extract_cache_kwarg,
    extract_output_cache,
)
from qloop.model.hssd import install_hssd
from qloop.model.loop import SharedKVView, apply_loop
from qloop.model.lsa import install_lsa, lsa_attention
from qloop.model.probes import _cached_forward_seq, _greedy
from qloop.model.toy import ToyConfig, ToyForCausalLM, ToyHybridCache
from qloop.train.bptt import BPTTConfig, SegmentedKVBPTTRunner, _ReplayKVCacheProxy
from qloop.train.long_calib import prefix_suffix_calibration_step


@dataclass
class _TF524ModelOutput:
    """Strict `transformers 5.24.0` text-model output: exposes `.cache` ONLY (no `.past_key_values`)."""

    last_hidden_state: torch.Tensor
    cache: Any = None
    hidden_states: tuple[torch.Tensor, ...] | None = None


@dataclass
class _TF524CausalLMOutput:
    """Strict `transformers 5.24.0` CausalLM output: exposes `.cache` ONLY (no `.past_key_values`)."""

    logits: torch.Tensor
    loss: torch.Tensor | None = None
    cache: Any = None


def _convert_toy_to_strict_tf524(model: ToyForCausalLM) -> dict[str, list[dict[str, Any]]]:
    """Patch a `ToyForCausalLM` instance in-place so every layer uses strict `transformers 5.24.0` `cache=` kwargs.

    Every patched `forward` rejects `past_key_values` / `cache_params` (no `**kwargs` swallowing them)
    and returns `_TF524ModelOutput` / `_TF524CausalLMOutput` with `.cache` ONLY.
    """
    calls: dict[str, list[dict[str, Any]]] = {
        "text_model": [],
        "decoder_layer": [],
        "linear_attn": [],
        "self_attn": [],
    }

    for layer in model.model.layers:
        if layer.layer_type == "linear_attention":
            lin = layer.linear_attn
            orig_lin_fwd = lin.forward

            def _lin_fwd(
                _self: nn.Module,
                hidden_states: torch.Tensor,
                *,
                cache: Any = None,
                attention_mask: torch.Tensor | None = None,
                _orig=orig_lin_fwd,
            ) -> torch.Tensor:
                calls["linear_attn"].append({"layer_idx": _self.layer_idx, "cache": cache})
                return _orig(hidden_states, cache_params=cache, attention_mask=attention_mask)

            lin.forward = types.MethodType(_lin_fwd, lin)  # type: ignore[method-assign]
        else:
            attn = layer.self_attn
            orig_attn_fwd = attn.forward

            def _attn_fwd(
                _self: nn.Module,
                hidden_states: torch.Tensor,
                position_embeddings: tuple[torch.Tensor, torch.Tensor],
                attention_mask: torch.Tensor | None = None,
                cache: Any = None,
                cache_position: torch.Tensor | None = None,
                _orig=orig_attn_fwd,
            ) -> torch.Tensor:
                calls["self_attn"].append({"layer_idx": _self.layer_idx, "cache": cache})
                return _orig(hidden_states, position_embeddings, attention_mask, cache, cache_position)

            attn.forward = types.MethodType(_attn_fwd, attn)  # type: ignore[method-assign]

        def _dec_fwd(
            _self: nn.Module,
            hidden_states: torch.Tensor,
            *,
            position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
            attention_mask: torch.Tensor | None = None,
            position_ids: torch.Tensor | None = None,
            cache: Any = None,
            cache_position: torch.Tensor | None = None,
        ) -> torch.Tensor:
            calls["decoder_layer"].append({"layer_type": _self.layer_type, "cache": cache})
            residual = hidden_states
            h = _self.input_layernorm(hidden_states)
            if _self.layer_type == "linear_attention":
                h = _self.linear_attn(h, cache=cache, attention_mask=attention_mask)
            else:
                h = _self.self_attn(
                    hidden_states=h,
                    position_embeddings=position_embeddings,
                    attention_mask=attention_mask,
                    cache=cache,
                    cache_position=cache_position,
                )
            h = residual + h
            residual = h
            h = _self.mlp(_self.post_attention_layernorm(h))
            return residual + h

        layer.forward = types.MethodType(_dec_fwd, layer)  # type: ignore[method-assign]

    tm = model.model

    def _tm_fwd(
        _self: nn.Module,
        input_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        cache: Any = None,
        use_cache: bool = False,
        output_hidden_states: bool = False,
        attention_mask: Any = None,
    ) -> _TF524ModelOutput:
        calls["text_model"].append({"cache": cache, "use_cache": use_cache})
        h = _self.embed_tokens(input_ids) if inputs_embeds is None else inputs_embeds
        if use_cache and cache is None:
            cache = ToyHybridCache(_self.config)
        start = cache.get_seq_length() if cache is not None else 0
        cache_position = torch.arange(start, start + h.shape[1], device=h.device)
        position_ids = cache_position[None].expand(h.shape[0], -1)
        pos = _self.rotary_emb(h, position_ids)
        all_h = [h] if output_hidden_states else None
        for lyr in _self.layers[: _self.config.num_hidden_layers]:
            h = lyr(
                h,
                position_embeddings=pos,
                attention_mask=None,
                position_ids=position_ids,
                cache=cache,
                cache_position=cache_position,
            )
            if output_hidden_states and all_h is not None:
                all_h.append(h)
        h = _self.norm(h)
        return _TF524ModelOutput(h, cache=cache, hidden_states=tuple(all_h) if all_h else None)

    tm.forward = types.MethodType(_tm_fwd, tm)  # type: ignore[method-assign]

    def _lm_fwd(
        _self: nn.Module,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
        cache: Any = None,
        use_cache: bool = False,
        attention_mask: Any = None,
    ) -> _TF524CausalLMOutput:
        out = _self.model(input_ids=input_ids, cache=cache, use_cache=use_cache, attention_mask=attention_mask)
        logits = _self.lm_head(out.last_hidden_state)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, logits.shape[-1]).float(),
                labels[:, 1:].reshape(-1),
                ignore_index=-100,
            )
        return _TF524CausalLMOutput(logits=logits, loss=loss, cache=out.cache)

    model.forward = types.MethodType(_lm_fwd, model)  # type: ignore[method-assign]
    return calls


# ============================================================================ 1. Shared helpers


def test_cache_compat_helpers_priority_and_falsy_cache() -> None:
    c_new, c_old = object(), object()

    assert extract_cache_kwarg({"cache": c_new}) == ("cache", c_new)
    assert extract_cache_kwarg({"cache_params": c_old}) == ("cache_params", c_old)
    assert extract_cache_kwarg({"past_key_values": c_old}) == ("past_key_values", c_old)
    assert extract_cache_kwarg({"past_key_value": c_old}) == ("past_key_value", c_old)
    # Skips None entries instead of getting fooled by an explicit `past_key_values=None` or `cache=None`
    assert extract_cache_kwarg({"past_key_values": None, "cache": c_new}) == ("cache", c_new)
    assert extract_cache_kwarg({"cache": None, "cache_params": c_old}) == ("cache_params", c_old)
    assert extract_cache_from_kwargs({"cache": None, "past_key_values": None}) is None

    class _EmptyCache:
        def __len__(self) -> int:
            return 0

    empty_c = _EmptyCache()
    assert not bool(empty_c)
    assert extract_output_cache(_TF524ModelOutput(torch.zeros(1), cache=empty_c)) is empty_c

    class _DeprecatedPastOut:
        def __init__(self, c: Any) -> None:
            self.cache = c

        @property
        def past_key_values(self) -> Any:
            raise AssertionError("should read .cache first without touching deprecated .past_key_values")

    assert extract_output_cache(_DeprecatedPastOut(c_new)) is c_new
    assert extract_output_cache(type("LegacyOut", (), {"past_key_values": c_old})()) is c_old

    class _M524(nn.Module):
        def forward(self, input_ids: torch.Tensor, cache: Any = None) -> torch.Tensor:
            return input_ids

    class _MLegacy(nn.Module):
        def forward(self, input_ids: torch.Tensor, past_key_values: Any = None) -> torch.Tensor:
            return input_ids

    class _MVarKw(nn.Module):
        def forward(self, *args: Any, **kwargs: Any) -> Any:
            return args

    assert cache_kwarg_name(_M524()) == "cache"
    assert cache_kwarg_name(_MLegacy()) == "past_key_values"
    assert cache_kwarg_name(_MVarKw()) == "past_key_values"


# ============================================================================ 2. HSSD pre_hook (`detach_fs`)


def test_hssd_pre_hook_extracts_detach_fs_from_cache_kwarg(monkeypatch) -> None:
    """`hssd.py` pre_hook extracts `_detach_recurrent_state=True` from `cache=` so `_chunk_hssd_triton_fast` fires."""
    torch.manual_seed(0)
    cfg = ToyConfig(num_hidden_layers=4, hidden_size=32, num_attention_heads=4, num_key_value_heads=2, head_dim=8)
    model = ToyForCausalLM(cfg)
    _convert_toy_to_strict_tf524(model)
    install_hssd(
        model,
        {
            "chunk_size": 4,
            "window_chunks": 2,
            "num_pages": 4,
            "k_pages": 2,
            "segment_chunks": 2,
        },
    )

    seen_calls: list[tuple[bool, bool]] = []
    real_chunk_hssd = hssd_mod.chunk_hssd

    def _spy_chunk_hssd(*args: Any, **kwargs: Any) -> Any:
        seen_calls.append(
            (bool(kwargs.get("output_final_state", False)), bool(kwargs.get("detach_final_state", False)))
        )
        return real_chunk_hssd(*args, **kwargs)

    monkeypatch.setattr(hssd_mod, "chunk_hssd", _spy_chunk_hssd)

    ids = torch.randint(0, cfg.vocab_size, (1, 12))
    base_cache = ToyHybridCache(cfg)
    proxy = _ReplayKVCacheProxy(base_cache, zero_copy_hssd=True)
    assert getattr(proxy, "_detach_recurrent_state", False) is True

    # 1) Strict TF 5.24 `cache=proxy` -> EVERY linear layer must get `output_final_state=True, detach_final_state=True`
    seen_calls.clear()
    out = model(ids, cache=proxy, use_cache=True)
    assert out.cache is proxy
    n_linear = sum(1 for lt in cfg.layer_types if lt == "linear_attention")
    assert len(seen_calls) == n_linear
    for out_final, detach_fs in seen_calls:
        assert out_final is True
        assert detach_fs is True
        # Fast-kernel eligibility gate in `hssd.py`: `st_detached and (not output_final_state or detach_final_state)`
        assert (not out_final) or detach_fs

    # 2) Plain cache without `_detach_recurrent_state` passed via `cache=` -> `detach_final_state=False`
    seen_calls.clear()
    plain_cache = ToyHybridCache(cfg)
    model(ids, cache=plain_cache, use_cache=True)
    assert len(seen_calls) == n_linear
    assert all(out_final is True and detach_fs is False for out_final, detach_fs in seen_calls)

    # 3) Legacy `cache_params=proxy` and `past_key_values=proxy` still work on standard ToyModel
    legacy_model = ToyForCausalLM(cfg)
    install_hssd(legacy_model, {"chunk_size": 4, "window_chunks": 2, "num_pages": 4, "k_pages": 2, "segment_chunks": 2})
    seen_calls.clear()
    legacy_model(ids, past_key_values=_ReplayKVCacheProxy(ToyHybridCache(cfg)), use_cache=True)
    assert len(seen_calls) == n_linear
    assert all(out_final is True and detach_fs is True for out_final, detach_fs in seen_calls)


# ============================================================================ 3. LSA pre_hook & lsa_attention


def test_lsa_pre_hook_and_lsa_attention_extract_cache_from_tf524_kwarg() -> None:
    """Both `_make_attn_pkv_pre_hook` and `lsa_attention` recognize `cache=` (NO `past_key_values`)."""
    torch.manual_seed(1)
    cfg = ToyConfig(num_hidden_layers=4, hidden_size=32, num_attention_heads=4, num_key_value_heads=2, head_dim=8)
    model = ToyForCausalLM(cfg)
    _convert_toy_to_strict_tf524(model)
    lsa_mgr = install_lsa(
        model,
        {
            "mode": "sparse",
            "block_size": 8,
            "pool": 2,
            "avg_bucket_keys": 4,
            "max_bucket_keys": 4,
            "num_select": 2,
            "window": 8,
            "sinks": 2,
        },
    )
    assert len(lsa_mgr.indexers) == 1
    full_idx, ix = next(iter(lsa_mgr.indexers.items()))
    attn_mod = model.model.layers[full_idx].self_attn

    # 1) End-to-end strict TF 5.24 forward passes `cache=pkv` -> pre_hook records `id(pkv)`
    pkv = ToyHybridCache(cfg)
    ids = torch.randint(0, cfg.vocab_size, (1, 16))
    with torch.no_grad():
        model(ids, cache=pkv, use_cache=True)
    pkv_ref, pkv_id, pkv_ok = ix._active_pkv
    assert pkv_ok is True
    assert pkv_id == id(pkv)
    assert pkv_ref is not None and pkv_ref() is pkv

    # 2) Direct `lsa_attention(..., cache=pkv)` (no `past_key_values`/`cache_key`) sets `_active_pkv` + hits cache
    ix._active_pkv = (None, None, False)
    ix._cache.clear()
    q = torch.randn(1, cfg.num_attention_heads, 2, cfg.head_dim)
    k = torch.randn(1, cfg.num_key_value_heads, 16, cfg.head_dim)
    v = torch.randn(1, cfg.num_key_value_heads, 16, cfg.head_dim)
    with torch.no_grad():
        out1, _ = lsa_attention(attn_mod, q, k, v, attention_mask=None, cache=pkv)
    assert out1.shape == (1, 2, cfg.num_attention_heads, cfg.head_dim)
    assert ix._active_pkv[1] == id(pkv)
    assert ix._active_pkv[2] is True
    assert any(key[0] == "pkv" and key[1] == id(pkv) for key in ix._cache)

    # 3) Also works with `cache=SharedKVView(pkv)` and `cache_params=pkv`
    ix._active_pkv = (None, None, False)
    with torch.no_grad():
        lsa_attention(attn_mod, q, k, v, attention_mask=None, cache=SharedKVView(pkv))
    assert ix._active_pkv[1] == id(pkv) and ix._active_pkv[2] is True

    ix._active_pkv = (None, None, False)
    with torch.no_grad():
        lsa_attention(attn_mod, q, k, v, attention_mask=None, cache_params=pkv)
    assert ix._active_pkv[1] == id(pkv) and ix._active_pkv[2] is True


# ============================================================================ 4. Loop pre_hook & VirtualLayer


def test_loop_pre_hook_and_virtual_layer_extract_and_rewrite_tf524_cache() -> None:
    """`loop.py` `_tm_cache_pre` and `VirtualLayer.forward` handle `cache=` (and `past_key_values=None, cache=...`)."""
    torch.manual_seed(2)
    cfg = ToyConfig(num_hidden_layers=8, hidden_size=32, num_attention_heads=4, num_key_value_heads=2, head_dim=8)
    model = ToyForCausalLM(cfg)
    calls = _convert_toy_to_strict_tf524(model)
    ctl = apply_loop(
        model,
        {
            "prelude_end": 2,
            "core_end": 6,
            "num_passes": 2,
            "target_virtual_layers": None,
            "kv_share": True,
            "checkpoint": False,
        },
    )

    # Cache with intentionally stale `last_linear_layer` and short slot lists
    pkv = ToyHybridCache(cfg)
    pkv.last_linear_layer = 0
    pkv.conv_states = [None] * 2
    pkv.recurrent_states = [None] * 2
    pkv.key_cache = [None] * 2
    pkv.value_cache = [None] * 2

    ids = torch.randint(0, cfg.vocab_size, (1, 6))
    with torch.no_grad():
        model(ids, cache=pkv, use_cache=True)

    # `_tm_cache_pre` must have synced `last_linear_layer` and extended slots from `cache=`
    assert pkv.last_linear_layer == ctl._last_active_linear_vidx()
    assert len(pkv.conv_states) == len(ctl.schedule)
    assert len(pkv.recurrent_states) == len(ctl.schedule)

    # Core full layer (phys=3) must get `SharedKVView(record=True)` on Pass 1 & `record=False` on Pass 2 via `cache=`
    core_full_calls = [c for c in calls["self_attn"] if c["layer_idx"] == 3]
    assert len(core_full_calls) == 2
    assert isinstance(core_full_calls[0]["cache"], SharedKVView)
    assert object.__getattribute__(core_full_calls[0]["cache"], "_record") is True
    assert isinstance(core_full_calls[1]["cache"], SharedKVView)
    assert object.__getattribute__(core_full_calls[1]["cache"], "_record") is False

    # Verify VirtualLayer skips `past_key_values=None` and still wraps `cache=pkv`
    v_pass2 = next(
        vl for vl in model.model.layers if vl.slot.layer_type == "full_attention" and vl.slot.r == 2
    )
    seen_inner: list[dict[str, Any]] = []
    orig_inner = v_pass2.inner

    class _SpyInner(nn.Module):
        def forward(self, x: torch.Tensor, **kw: Any) -> torch.Tensor:
            seen_inner.append(dict(kw))
            return x

    v_pass2.inner = _SpyInner()
    try:
        v_pass2(torch.zeros(1, 2, cfg.hidden_size), past_key_values=None, cache=pkv)
    finally:
        v_pass2.inner = orig_inner
    assert len(seen_inner) == 1
    assert isinstance(seen_inner[0]["cache"], SharedKVView)
    assert seen_inner[0]["past_key_values"] is None


# ============================================================================ 5. BPTT attention pre_hook & full run


def test_bptt_tf524_end_to_end_and_cache_kwarg_rewrite() -> None:
    """`SegmentedKVBPTTRunner` works end-to-end on strict TF 5.24 (`cache=`/`.cache`) and rewrites `cache=`."""
    torch.manual_seed(3)
    cfg = ToyConfig(num_hidden_layers=8, hidden_size=32, num_attention_heads=4, num_key_value_heads=2, head_dim=8)
    model = ToyForCausalLM(cfg)
    _convert_toy_to_strict_tf524(model)
    install_hssd(
        model,
        {"chunk_size": 4, "window_chunks": 2, "num_pages": 4, "k_pages": 2, "segment_chunks": 2},
    )
    lsa_mgr = install_lsa(
        model,
        {
            "mode": "sparse",
            "block_size": 8,
            "pool": 2,
            "avg_bucket_keys": 4,
            "max_bucket_keys": 4,
            "num_select": 2,
            "window": 8,
            "sinks": 2,
            "num_query_passes": 2,
        },
    )
    apply_loop(
        model,
        {
            "prelude_end": 2,
            "core_end": 6,
            "num_passes": 2,
            "target_virtual_layers": None,
            "kv_share": True,
            "checkpoint": True,
        },
    )
    model.train()

    runner = SegmentedKVBPTTRunner(
        model,
        cfg=BPTTConfig(seg_len=16, n_segments=2, replay_k=4, offload_prior_hidden=False, replay_indexer_weight=0.2),
    )
    ids = torch.randint(0, cfg.vocab_size, (1, 32))
    res = runner.run_step(ids, labels=ids.clone(), backward=True)
    assert res.num_segments == 2
    assert res.total_loss > 0.0
    assert res.replayed_tokens_per_seg == [0, 4]
    # Core full-attention layer (phys=3) AND coda full-attention layer (phys=7) both replayed on Segment 1
    assert (1, 3) in res.replayed_indices
    assert (1, 7) in res.replayed_indices
    assert any(ix.wk.grad is not None and ix.wk.grad.abs().sum().item() > 0 for ix in lsa_mgr.indexers.values())

    # Verify direct `self_attn` kwarg rewriting: `cache=base_cache` -> `cache=proxy` (not `past_key_values`),
    # `past_key_values=base_cache` -> `past_key_values=proxy`, positional `args[3]`, and `SharedKVView` skip.
    torch.manual_seed(4)
    cfg2 = ToyConfig(num_hidden_layers=4, hidden_size=32, num_attention_heads=4, num_key_value_heads=2, head_dim=8)
    m2 = ToyForCausalLM(cfg2)
    attn_calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
    full_layer = m2.model.layers[3]
    orig_sa = full_layer.self_attn.forward

    def _recording_sa(_self: nn.Module, *a: Any, **kw: Any) -> torch.Tensor:
        attn_calls.append((a, dict(kw)))
        pos_emb = a[1] if len(a) > 1 else kw["position_embeddings"]
        mask = a[2] if len(a) > 2 else kw.get("attention_mask")
        c_obj = a[3] if len(a) > 3 else extract_cache_from_kwargs(kw)
        cpos = a[4] if len(a) > 4 else kw.get("cache_position")
        x_in = a[0] if len(a) > 0 else kw["hidden_states"]
        return orig_sa(x_in, pos_emb, mask, c_obj, cpos)

    full_layer.self_attn.forward = types.MethodType(_recording_sa, full_layer.self_attn)  # type: ignore[method-assign]

    # Force decoder layer to unwrap `_ReplayKVCacheProxy` back to `base_cache` across calling conventions
    modes = ["cache_kw", "pkv_kw", "positional", "shared_view_cache_kw"]
    mode_box = [modes[0]]

    def _unwrapping_dec(
        _self: nn.Module,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        attention_mask: torch.Tensor | None = None,
        *,
        position_ids: torch.Tensor | None = None,
        past_key_values: Any = None,
        cache_position: torch.Tensor | None = None,
        **kw: Any,
    ) -> torch.Tensor:
        residual = hidden_states
        h = _self.input_layernorm(hidden_states)
        raw_c = past_key_values.base_cache if isinstance(past_key_values, _ReplayKVCacheProxy) else past_key_values
        m = mode_box[0]
        if m == "cache_kw":
            h = _self.self_attn(
                hidden_states=h,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                cache=raw_c,
                cache_position=cache_position,
            )
        elif m == "pkv_kw":
            h = _self.self_attn(
                hidden_states=h,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                past_key_values=raw_c,
                cache_position=cache_position,
            )
        elif m == "positional":
            h = _self.self_attn(h, position_embeddings, attention_mask, raw_c, cache_position)
        else:
            # SharedKVView(record=False) passed via `cache=` must be skipped cleanly (no replay)
            h = _self.self_attn(
                hidden_states=h,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                cache=SharedKVView(past_key_values, record=False),
                cache_position=cache_position,
            )
        h = residual + h
        return h + _self.mlp(_self.post_attention_layernorm(h))

    full_layer.forward = types.MethodType(_unwrapping_dec, full_layer)  # type: ignore[method-assign]
    r2 = SegmentedKVBPTTRunner(
        m2, cfg=BPTTConfig(seg_len=8, n_segments=2, replay_k=3, offload_prior_hidden=False, replay_indexer_weight=0.0)
    )
    seq2 = torch.randint(0, cfg2.vocab_size, (1, 16))

    for m in ("cache_kw", "pkv_kw", "positional"):
        mode_box[0] = m
        attn_calls.clear()
        res_m = r2.run_step(seq2, labels=seq2.clone(), backward=False)
        assert res_m.replayed_tokens_per_seg == [0, 3], f"replay failed for mode={m}"
        assert len(attn_calls) == 2
        a_last, kw_last = attn_calls[-1]
        if m == "cache_kw":
            assert isinstance(kw_last.get("cache"), _ReplayKVCacheProxy)
            assert "past_key_values" not in kw_last
        elif m == "pkv_kw":
            assert isinstance(kw_last.get("past_key_values"), _ReplayKVCacheProxy)
            assert "cache" not in kw_last
        else:
            assert len(a_last) >= 4 and isinstance(a_last[3], _ReplayKVCacheProxy)

    # `SharedKVView(record=False)` passed via `cache=` must be detected and skipped (0 replayed tokens)
    mode_box[0] = "shared_view_cache_kw"
    attn_calls.clear()
    res_skip = r2.run_step(seq2, labels=seq2.clone(), backward=False)
    assert res_skip.replayed_tokens_per_seg == [0, 0]


# ============================================================================ 6. Downstream callers (`outputs.cache`)


def test_downstream_callers_support_tf524_cache_kwarg_and_output_cache() -> None:
    """`long_calib`, `probes`, `eval.generate`, `infer.engine`, and `hf_custom_modeling` all work with TF 5.24."""
    torch.manual_seed(5)
    cfg = ToyConfig(num_hidden_layers=4, hidden_size=32, num_attention_heads=4, num_key_value_heads=2, head_dim=8)
    model = ToyForCausalLM(cfg)
    _convert_toy_to_strict_tf524(model)

    # 1) `prefix_suffix_calibration_step`
    ids = torch.randint(0, cfg.vocab_size, (1, 20))
    lc_res = prefix_suffix_calibration_step(model, ids, prefix_len=12, chunk_size=6, backward=True)
    assert lc_res.num_prefix_chunks == 2
    assert lc_res.task_loss > 0.0

    # 2) `probes._greedy` & `probes._cached_forward_seq`
    model.eval()
    seq_g, logits_g = _greedy(model, ids[:, :6], steps=3)
    assert seq_g.shape == (1, 9) and logits_g.shape == (1, 4, cfg.vocab_size)
    logits_tf = _cached_forward_seq(model, ids[:, :10], prefix_len=6)
    assert logits_tf.shape == (1, 5, cfg.vocab_size)

    # 3) `eval.generate.Generator`
    class _MiniTok:
        eos_token_id = 0
        pad_token_id = 0

        def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
            return [1, 2, 3, 4]

        def decode(self, tids: list[int], skip_special_tokens: bool = True) -> str:
            return ",".join(str(x) for x in tids)

    gen = Generator(model, _MiniTok(), device="cpu")
    assert gen._cache_kw == "cache"
    outs = gen.generate(["hello"], max_new_tokens=3, temperature=0.0)
    assert len(outs) == 1 and len(outs[0]) == 1 and outs[0][0].completion_tokens == 3

    # 4) `OptimizedLoopInferenceEngine`
    eng = OptimizedLoopInferenceEngine(model)
    logits_last, eng_cache, n_chunks = eng.prefill(ids[:, :8], chunk_size=4)
    assert logits_last.shape == (1, 1, cfg.vocab_size)
    assert n_chunks == 2 and eng_cache is not None
    tm_step_out, next_cache, _ = eng._run_text_model_step(ids[:, 8:9], eng_cache, allow_mor_early_exit=False)
    assert tm_step_out.last_hidden_state.shape == (1, 1, cfg.hidden_size)
    assert next_cache is not None

    # 5) `CausalLMPackOutput.cache` & `Qwen36LoopHSSDForCausalLM.forward(..., cache=...)`
    pack_out = CausalLMPackOutput(logits=logits_last, past_key_values=eng_cache)
    assert pack_out.cache is eng_cache

    pack_cfg = Qwen36LoopHSSDConfig(
        base_config={
            "model_type": "toy",
            "num_hidden_layers": 4,
            "hidden_size": 32,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 8,
        },
        qloop_v3_arch={},
    )
    wrapper = Qwen36LoopHSSDForCausalLM(pack_cfg)
    wrapper.backbone = model
    w_cache = ToyHybridCache(cfg)
    w_out = wrapper(input_ids=ids[:, :4], cache=w_cache, use_cache=True, logits_to_keep=1)
    assert w_out.logits.shape == (1, 1, cfg.vocab_size)
    assert w_out.cache is w_cache
    assert w_out.past_key_values is w_cache
    w_out2 = wrapper(input_ids=ids[:, 4:5], cache=w_cache, use_cache=True)
    assert w_out2.cache is w_cache
