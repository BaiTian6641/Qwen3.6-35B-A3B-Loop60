"""Cheap empirical probes that de-risk the expensive stages (run them first on Colab).

* `probe_backward`              - do gradients flow through the frozen (possibly NVFP4/FP8) base to
                                  the input? Names the first layer where the gradient vanishes.
* `probe_core_ranges`           - ETD-style choice of the looped core for a target virtual depth:
                                  per-layer angular distance of the residual stream + zero-shot
                                  "loopability" (loss increase when the core is simply run twice).
* `surgery_check`               - on the REAL model: loop/LSA wrapping is output-identical at init
                                  and cached decode == full forward (catches HF cache/API drift).
* `probe_expert_headroom`       - Gate 0p-1: hard-token NLL headroom under Top-k router logit
                                  perturbations across core MoE layers.
* `probe_reentry_anchors`       - Gate 0p-2: Layer `core_end-1` -> Layer `prelude_end` cosine
                                  alignment, Procrustes / low-rank subspace explained variance, and
                                  Pass-2 routing Jaccard across anchor weights.
* `probe_positional_extrapolation` - Gate 0p-3: NLL and synthetic passkey/needle retrieval logit
                                  margin across `("native", "yarn", "rerope")` at `factor`x context.
* `profile_hssd_layer`          - Gate 0p-4: Single-layer HSSD v3.1 vs single-state GDN profiler
                                  (prefill, streamed prefill, decode, fwd+bwd, compact cache bytes).
* `hssd_ablation_matrix`        - Gate 1-Ablate: 6-way matched-budget HSSD mechanism isolation matrix.
* `run_structured_probe_suite`  - executes `Gate 0a` + `Gate 0p-1..3` + core-range sweep, returning
                                  a JSON-serializable report dict + Markdown summary.
"""
from __future__ import annotations

import copy
import json
import logging
import math
import time
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from ..debug import cuda_mem_brief, describe, require
from ._dtype import upcast
from .introspect import FULL, LINEAR, inspect_model, router_of
from .loop import LoopSpec, _looks_like_probs, apply_loop, sha256_frozen_experts
from .mhc import V3_BRIDGE_WEIGHTS
from .quant_autograd import FP8_DTYPES, _dequant_linear_weight, is_quantized_module

log = logging.getLogger("qloop")


def _device_of(model: nn.Module) -> torch.device:
    return next(model.parameters()).device


def _as_batches(ids: torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor, ...]) -> list[torch.Tensor]:
    if isinstance(ids, torch.Tensor):
        t = ids.unsqueeze(0) if ids.ndim == 1 else ids
        require(t.ndim == 2 and t.shape[1] >= 2, "probe ids tensor must have shape [B, T>=2]", ids=t)
        return [t]
    batches = [b.unsqueeze(0) if b.ndim == 1 else b for b in ids]
    require(bool(batches), "probe requires at least one token batch")
    for b in batches:
        require(b.ndim == 2 and b.shape[1] >= 2, "each probe batch must have shape [B, T>=2]", batch=b)
    return batches


def _lm_loss(model: nn.Module, ids: torch.Tensor) -> torch.Tensor:
    logits = model(input_ids=ids).logits
    return F.cross_entropy(logits[:, :-1].float().reshape(-1, logits.shape[-1]), ids[:, 1:].reshape(-1))


def _token_nll(model: nn.Module, ids: torch.Tensor) -> torch.Tensor:
    logits = model(input_ids=ids).logits
    vocab = logits.shape[-1]
    return F.cross_entropy(
        logits[:, :-1].float().reshape(-1, vocab),
        ids[:, 1:].reshape(-1),
        reduction="none",
    )


def _macro_align(layer_types: list[str]) -> int:
    fa = [i for i, t in enumerate(layer_types) if t == FULL]
    return (fa[1] - fa[0]) if len(fa) > 1 else 1


def _default_core_bounds(layer_types: list[str]) -> tuple[int, int]:
    n = len(layer_types)
    align = _macro_align(layer_types)
    if n == 40 and align == 4:
        return 8, 32
    n_blocks = max(n // max(align, 1), 1)
    if n_blocks >= 4:
        p_blk = max(1, n_blocks // 5 if n_blocks >= 10 else n_blocks // 4)
        c_blk = min(n_blocks - 1, n_blocks - p_blk)
        if p_blk < c_blk:
            return p_blk * align, c_blk * align
    p = max(1, n // 4)
    c = max(p + 1, min(n - 1, (3 * n) // 4))
    return p, c


def _to_jsonable(obj: Any) -> Any:
    if obj is None or isinstance(obj, (bool, int, str)):
        return obj
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, torch.Tensor):
        if obj.numel() == 1:
            return _to_jsonable(float(obj.detach().cpu().item()))
        return _to_jsonable(obj.detach().cpu().tolist())
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if hasattr(obj, "item") and callable(obj.item):
        return _to_jsonable(obj.item())
    return str(obj)


# ---------------------------------------------------------------------------------------- backward

def probe_backward(model: nn.Module, ids: torch.Tensor) -> dict[str, Any]:
    parts = inspect_model(model)
    b_ids = _as_batches(ids)[0]
    had_no_grads = all(p.grad is None for p in model.parameters())
    grads: dict[int, float] = {}
    handles = []

    def fwd_hook(i: int):
        def hook(_m, _inp, out):
            h = out[0] if isinstance(out, tuple) else out
            if isinstance(h, torch.Tensor) and h.requires_grad:
                h.register_hook(lambda g, i=i: grads.__setitem__(i, float(g.float().norm())))
        return hook

    for i, layer in enumerate(parts.layers):
        handles.append(layer.register_forward_hook(fwd_hook(i)))
    emb = parts.text_model.embed_tokens(b_ids).detach().requires_grad_(True)
    t0 = time.time()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    try:
        out = model(inputs_embeds=emb) if "inputs_embeds" in model.forward.__code__.co_varnames else None
        if out is None:
            hidden = parts.text_model(inputs_embeds=emb).last_hidden_state
            require(parts.lm_head is not None, "model has no lm_head for backward probe")
            assert parts.lm_head is not None
            logits = parts.lm_head(hidden)
        else:
            logits = out.logits
        loss = F.cross_entropy(logits[:, :-1].float().reshape(-1, logits.shape[-1]), b_ids[:, 1:].reshape(-1))
        loss.backward()
    finally:
        for h in handles:
            h.remove()
        if had_no_grads:
            model.zero_grad(set_to_none=True)
    g_in = emb.grad
    ok = g_in is not None and bool(torch.isfinite(g_in).all()) and float(g_in.norm()) > 0
    dead = [i for i in range(parts.num_layers) if grads.get(i, 0.0) == 0.0]
    rep = {
        "ok": ok,
        "loss": float(loss.detach()),
        "input_grad_norm": float(g_in.norm()) if g_in is not None else None,
        "layer_grad_norms": grads,
        "dead_layers": dead,
        "seconds": time.time() - t0,
        "mem": cuda_mem_brief(),
    }
    if not ok:
        log.error("backward probe FAILED: input grad %s; layers with zero/None grad: %s", describe(g_in), dead)
    else:
        log.info("backward probe ok: loss %.3f, |dL/dx| %.3g, %.1fs %s", rep["loss"], rep["input_grad_norm"],
                 rep["seconds"], rep["mem"])
    return rep


# ---------------------------------------------------------------------------------------- core ranges

def candidate_specs(
    layer_types: list[str],
    target_virtual: int,
    passes_options: tuple[int, ...] = (2, 3),
    align: int | None = None,
) -> list[LoopSpec]:
    n = len(layer_types)
    if align is None:
        align = _macro_align(layer_types)
    out = []
    for r in passes_options:
        if (target_virtual - n) % (r - 1):
            continue
        core = (target_virtual - n) // (r - 1)
        if core <= 0 or core > n:
            continue
        for p in range(align, n - core, align):
            c = p + core
            if c > n - align or core % align:
                continue
            out.append(LoopSpec(prelude_end=p, core_end=c, num_passes=r, target_virtual_layers=target_virtual))
    return out


@torch.no_grad()
def probe_core_ranges(
    model: nn.Module,
    batches: list[torch.Tensor] | torch.Tensor,
    target_virtual: int = 64,
    passes_options: tuple[int, ...] = (2, 3),
    zero_shot: bool = True,
) -> list[dict[str, Any]]:
    parts = inspect_model(model)
    batch_list = _as_batches(batches)
    n = parts.num_layers
    ang = torch.zeros(n, dtype=torch.float64)
    base_losses = []
    for ids in batch_list:
        out = parts.text_model(input_ids=ids, output_hidden_states=True)
        hs = out.hidden_states
        require(hs is not None and len(hs) >= n + 1, "need output_hidden_states", n=n)
        assert hs is not None
        for i in range(n):
            a, b = hs[i].float(), hs[i + 1].float()
            cos = F.cosine_similarity(a, b, dim=-1).clamp(-1, 1)
            ang[i] += (torch.arccos(cos) / math.pi).mean().double().cpu()
        base_losses.append(float(_lm_loss(model, ids)))
    ang /= len(batch_list)
    base = sum(base_losses) / len(base_losses)
    log.info("per-layer angular distance (ETD): %s", [round(float(x), 3) for x in ang])
    rows = []
    for spec in candidate_specs(parts.layer_types, target_virtual, passes_options):
        core_ang = float(ang[spec.prelude_end:spec.core_end].mean())
        row: dict[str, Any] = {
            "prelude_end": spec.prelude_end,
            "core_end": spec.core_end,
            "num_passes": spec.num_passes,
            "core_len": spec.core_len,
            "core_angular": core_ang,
            "base_loss": base,
        }
        if zero_shot:
            s = copy.copy(spec)
            s.gate_init, s.bridge_init = 1.0, (0.98, 0.01, 0.01)
            ctl = apply_loop(model, s)
            try:
                ls = [float(_lm_loss(model, ids)) for ids in batch_list]
            finally:
                ctl.unwrap()
            row["looped_loss_zero_shot"] = sum(ls) / len(ls)
            row["loss_increase"] = row["looped_loss_zero_shot"] - base
        rows.append(row)
    key = (lambda r: (r["loss_increase"], r["core_angular"])) if zero_shot else (lambda r: r["core_angular"])
    rows.sort(key=key)
    for r in rows[:10]:
        log.info("candidate %s", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})
    return rows


# ---------------------------------------------------------------------------------------- surgery check

@torch.no_grad()
def _greedy(model: nn.Module, ids: torch.Tensor, steps: int) -> tuple[torch.Tensor, torch.Tensor]:
    out = model(input_ids=ids, use_cache=True)
    cache, seq, logits = out.past_key_values, ids, [out.logits[:, -1]]
    for _ in range(steps):
        nxt = logits[-1].argmax(-1, keepdim=True)
        seq = torch.cat([seq, nxt], 1)
        out = model(input_ids=nxt, past_key_values=cache, use_cache=True)
        if getattr(out, "past_key_values", None) is not None:
            cache = out.past_key_values
        logits.append(out.logits[:, -1])
    return seq, torch.stack(logits, 1)


@torch.no_grad()
def _cached_forward_seq(model: nn.Module, seq: torch.Tensor, prefix_len: int) -> torch.Tensor:
    """Teacher-forced cached decode over `seq[:, prefix_len:]` after prefilling `seq[:, :prefix_len]`."""
    out = model(input_ids=seq[:, :prefix_len], use_cache=True)
    cache = out.past_key_values
    logits = [out.logits[:, -1]]
    for pos in range(prefix_len, int(seq.shape[1])):
        nxt = seq[:, pos : pos + 1]
        out = model(input_ids=nxt, past_key_values=cache, use_cache=True)
        if getattr(out, "past_key_values", None) is not None:
            cache = out.past_key_values
        logits.append(out.logits[:, -1])
    return torch.stack(logits, 1)


def _is_low_prec_or_quant_model(model: nn.Module, sample_dtype: torch.dtype) -> bool:
    if sample_dtype in (torch.bfloat16, torch.float16):
        return True
    for owner in (model, getattr(model, "model", None)):
        cfg = getattr(owner, "config", None)
        if cfg is None:
            continue
        if bool(getattr(cfg, "quantization_config", None)):
            return True
        txt_cfg = getattr(cfg, "text_config", None)
        if bool(getattr(txt_cfg, "quantization_config", None)):
            return True
    low_dtypes = {torch.bfloat16, torch.float16}
    for t in (*model.parameters(), *model.buffers()):
        if t.dtype in low_dtypes or t.dtype in FP8_DTYPES or "float8" in str(t.dtype):
            return True
    quant_attrs = ("weight_packed", "gate_up_proj_packed", "down_proj_packed", "weight_scale_inv", "qweight")
    return any(
        is_quantized_module(mod) or any(hasattr(mod, attr) for attr in quant_attrs)
        for mod in model.modules()
    )


def _mean_logit_cosine(a_f32: torch.Tensor, b_f32: torch.Tensor) -> float:
    flat_a = a_f32.reshape(-1, a_f32.shape[-1])
    flat_b = b_f32.reshape(-1, b_f32.shape[-1])
    both_zero = (flat_a.norm(dim=-1) == 0) & (flat_b.norm(dim=-1) == 0)
    step_cos = F.cosine_similarity(flat_a, flat_b, dim=-1, eps=1e-8)
    step_cos = torch.where(both_zero, torch.ones_like(step_cos), step_cos)
    return float(step_cos.mean())


def _assert_logits_unchanged(
    model: nn.Module,
    ids: torch.Tensor,
    got_logits: torch.Tensor,
    ref_logits: torch.Tensor,
    msg: str,
) -> None:
    if torch.equal(got_logits, ref_logits):
        return
    require(
        got_logits.shape == ref_logits.shape and got_logits.dtype == ref_logits.dtype,
        msg,
        got_shape=tuple(got_logits.shape),
        ref_shape=tuple(ref_logits.shape),
        got_dtype=got_logits.dtype,
        ref_dtype=ref_logits.dtype,
    )
    with torch.no_grad():
        ctrl = model(input_ids=ids).logits
    noise_got = float((ctrl.float() - got_logits.float()).abs().max())
    noise_ref = float((ctrl.float() - ref_logits.float()).abs().max())
    noise = min(noise_got, noise_ref)
    d = float((got_logits.float() - ref_logits.float()).abs().max())
    c = _mean_logit_cosine(got_logits.float(), ref_logits.float())
    if noise > 0.0 and d <= max(2.5 * noise, 1e-3) and c >= 0.99999:
        return
    require(False, msg, maxdiff=d, kernel_noise=noise, cosine=c)


@torch.no_grad()
def surgery_check(
    model: nn.Module,
    ids: torch.Tensor,
    loop_spec: dict[str, Any] | None,
    lsa_cfg: dict[str, Any] | None,
    decode_steps: int = 8,
    atol: float = 2e-2,
    gdn2_cfg: dict[str, Any] | None = None,
    hssd_cfg: dict[str, Any] | None = None,
    max_probe_tokens: int = 256,
) -> dict[str, Any]:
    """Run on the real model in eval mode. Raises with a precise message on any violation."""
    from .gdn2 import install_gdn2
    from .hssd import apply_hssd
    from .lsa import install_lsa

    require(
        not (gdn2_cfg and hssd_cfg),
        "surgery_check accepts either gdn2_cfg or hssd_cfg, not both",
    )
    raw_ids = _as_batches(ids)[0]
    max_tok = int(max_probe_tokens)
    probe_ids = raw_ids[:, : min(int(raw_ids.shape[1]), max_tok)] if max_tok > 0 else raw_ids
    prefix_len = int(probe_ids.shape[1])
    require(prefix_len >= 1, "surgery_check requires at least 1 input token", prefix_len=prefix_len)

    was_training = model.training
    model.eval()
    ref = model(input_ids=probe_ids).logits
    rep: dict[str, Any] = {"probe_seq_len": prefix_len}
    gdn2 = install_gdn2(model, gdn2_cfg) if gdn2_cfg else None
    hssd = apply_hssd(model, hssd_cfg) if hssd_cfg else None
    mgr = install_lsa(model, dict(lsa_cfg, mode="distill")) if lsa_cfg else None
    ctl = apply_loop(model, loop_spec) if loop_spec else None
    try:
        out = model(input_ids=probe_ids).logits
        require(torch.isfinite(out).all(), "wrapped model produced non-finite logits at init")
        if hssd is None:
            rep["init_identity_maxdiff"] = float((out.float() - ref.float()).abs().max())
            _assert_logits_unchanged(
                model,
                probe_ids,
                out,
                ref,
                "wrapped model is not bit-identical at init",
            )
        else:
            rep["hssd_vs_gdn1_init_maxdiff"] = float((out.float() - ref.float()).abs().max())
        if ctl is not None:
            with torch.no_grad():
                ctl.adapters.gates.fill_(0.05)
                if ctl.adapters.pass_attn is not None:
                    gen_pa = torch.Generator(device="cpu").manual_seed(29)
                    for pdict in ctl.adapters.pass_attn.layers.values():
                        for proj in pdict.values():
                            noise_pa = torch.randn(proj.up.shape, generator=gen_pa, dtype=torch.float32) * 0.02
                            proj.up.copy_(noise_pa.to(device=proj.up.device, dtype=proj.up.dtype))
        if gdn2 is not None:
            gen = torch.Generator(device="cpu").manual_seed(17)
            for ad in gdn2.adapters.values():
                for w in (ad.erase_up.weight, ad.write_up.weight, ad.decay_up.weight):
                    noise = torch.randn(w.shape, generator=gen, dtype=torch.float32) * 0.02
                    w.copy_(noise.to(device=w.device, dtype=w.dtype))
        if hssd is not None:
            gen_h = torch.Generator(device="cpu").manual_seed(37)
            for had in hssd.adapters.values():
                for h_attr in ("b_up", "w_up", "gamma_up"):
                    h_mod = getattr(had, h_attr, None)
                    hw = getattr(h_mod, "weight", None)
                    if isinstance(hw, torch.Tensor):
                        noise_h = torch.randn(hw.shape, generator=gen_h, dtype=torch.float32) * 0.02
                        hw.copy_(noise_h.to(device=hw.device, dtype=hw.dtype))
        seq, inc_logits = _greedy(model, probe_ids, decode_steps)
        full = model(input_ids=seq).logits[:, prefix_len - 1 :]
        if mgr is not None:
            mgr.set_mode("sparse")
            mgr.clear_cache()
            sp = model(input_ids=probe_ids).logits
            rep["lsa_sparse_vs_dense_maxdiff"] = float((sp.float() - out.float()).abs().max())
    finally:
        if ctl is not None:
            ctl.unwrap()
        if mgr is not None:
            mgr.uninstall()
        if gdn2 is not None:
            gdn2.uninstall()
        if hssd is not None:
            hssd.uninstall()

    try:
        back = model(input_ids=probe_ids).logits
        _assert_logits_unchanged(
            model,
            probe_ids,
            back,
            ref,
            "model not restored after unwrap/uninstall",
        )
        base_inc = _cached_forward_seq(model, seq, prefix_len)
        base_full = model(input_ids=seq).logits[:, prefix_len - 1 :]
    finally:
        if was_training:
            model.train()

    f32_full = full.float()
    f32_inc = inc_logits.float()
    abs_err = (f32_full - f32_inc).abs()
    diff = float(abs_err.max())
    mae = float(abs_err.mean())
    cos = _mean_logit_cosine(f32_full, f32_inc)
    top1_match = float((f32_full.argmax(dim=-1) == f32_inc.argmax(dim=-1)).float().mean())

    base_f32_full = base_full.float()
    base_f32_inc = base_inc.float()
    base_err = (base_f32_full - base_f32_inc).abs()
    base_diff = float(base_err.max())
    base_mae = float(base_err.mean())
    base_cos = _mean_logit_cosine(base_f32_full, base_f32_inc)
    base_top1 = float((base_f32_full.argmax(dim=-1) == base_f32_inc.argmax(dim=-1)).float().mean())

    phys_layers = int(ctl.parts.num_layers) if ctl is not None else 1
    virt_layers = int(ctl.active_virtual_layers) if ctl is not None else phys_layers
    depth_mult = max(1.0, float(virt_layers) / max(float(phys_layers), 1.0))

    is_low_prec = (out.dtype != torch.float64) and _is_low_prec_or_quant_model(model, out.dtype)
    rep["decode_vs_full_maxdiff"] = diff
    rep["decode_vs_full_mae"] = mae
    rep["decode_vs_full_cosine"] = cos
    rep["decode_vs_full_top1_match"] = top1_match
    rep["base_decode_vs_full_maxdiff"] = base_diff
    rep["base_decode_vs_full_mae"] = base_mae
    rep["base_decode_vs_full_cosine"] = base_cos
    rep["base_decode_vs_full_top1_match"] = base_top1
    rep["depth_mult"] = depth_mult
    rep["low_precision_calibrated"] = bool(is_low_prec)

    if not is_low_prec:
        eff_atol = max(float(atol), 2.0 * base_diff)
        rep["effective_decode_atol"] = eff_atol
        require(
            diff < eff_atol,
            "cached decode diverges from full forward (cache semantics broken)",
            diff=diff,
            mae=mae,
            cosine=cos,
            base_diff=base_diff,
            base_cos=base_cos,
            depth_mult=depth_mult,
            eff_atol=eff_atol,
        )
    else:
        base_infid = max(0.0, 1.0 - base_cos)
        if base_infid <= 2e-3 and base_mae <= 0.05:
            min_cos = 0.9985
            strict_cos = 0.9995
            eff_atol = max(float(atol), 2.5 * base_diff + 0.15)
            eff_mae_atol = max(0.04, 2.5 * base_mae + 0.015)
        else:
            min_cos = max(0.85, 1.0 - (base_infid * (1.6 * depth_mult) + 0.015))
            strict_cos = max(min_cos, 1.0 - (base_infid * (1.25 * depth_mult) + 0.008))
            eff_atol = max(float(atol), (1.5 + 0.6 * depth_mult) * base_diff + 0.25)
            eff_mae_atol = max(0.05, (1.4 + 0.5 * depth_mult) * base_mae + 0.025)
        rep["effective_decode_atol"] = eff_atol
        rep["effective_min_cosine"] = min_cos
        rep["effective_mae_atol"] = eff_mae_atol
        ok = (cos >= min_cos) and (
            (diff < eff_atol and mae < eff_mae_atol * 1.25)
            or (cos >= strict_cos and mae < eff_mae_atol)
        )
        require(
            ok,
            "cached decode diverges from full forward (cache semantics broken)",
            diff=diff,
            mae=mae,
            cosine=cos,
            top1_match=top1_match,
            base_diff=base_diff,
            base_mae=base_mae,
            base_cos=base_cos,
            min_cos=min_cos,
            eff_atol=eff_atol,
            eff_mae_atol=eff_mae_atol,
            depth_mult=depth_mult,
        )
    log.info("surgery check passed: %s", rep)
    return rep


# ---------------------------------------------------------------------------------------- Gate 0p-1: expert headroom

def _topk_jaccard(idx_a: torch.Tensor, idx_b: torch.Tensor) -> float:
    inter = (idx_a.unsqueeze(-1) == idx_b.unsqueeze(-2)).any(-1).sum(-1).float()
    k = max(int(idx_a.shape[-1]), 1)
    return float((inter / (2 * k - inter).clamp_min(1.0)).mean())


@torch.no_grad()
def probe_expert_headroom(
    model: nn.Module,
    ids: torch.Tensor | list[torch.Tensor],
    n_perturb: int = 4,
    *,
    noise_std: float = 0.5,
    hard_quantile: float = 0.75,
    core_range: tuple[int, int] | None = None,
    pass_threshold_nats: float = 0.08,
    seed: int = 2026,
) -> dict[str, Any]:
    """Gate 0p-1: Measure hard-token NLL headroom under Top-k router logit perturbations."""
    require(n_perturb >= 1, "n_perturb must be >= 1", n_perturb=n_perturb)
    require(0.0 <= hard_quantile < 1.0, "hard_quantile must be in [0, 1)", hard_quantile=hard_quantile)
    t0 = time.time()
    parts = inspect_model(model)
    batches = _as_batches(ids)
    lo, hi = core_range if core_range is not None else _default_core_bounds(parts.layer_types)
    num_experts = int(getattr(parts.text_config, "num_experts", 0) or 0)
    top_k = int(getattr(parts.text_config, "num_experts_per_tok", 8) or 8)
    norm_topk = bool(getattr(parts.text_config, "norm_topk_prob", True))

    was_training = model.training
    model.eval()
    base_routes: dict[tuple[int, int], torch.Tensor] = {}
    is_probs_cache: dict[int, bool] = {}

    def _extract_router(phys: int, out: Any) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None, bool]:
        outs = out if isinstance(out, (tuple, list)) else (out,)
        tensors = [o for o in outs if isinstance(o, torch.Tensor)]
        full = next((o for o in tensors if o.is_floating_point() and o.shape[-1] == num_experts), None)
        require(full is not None, f"layer {phys} router output lacks [..., {num_experts}] float tensor")
        assert full is not None
        if phys not in is_probs_cache:
            is_probs_cache[phys] = isinstance(out, (tuple, list)) and _looks_like_probs(full)
        is_p = is_probs_cache[phys]
        idx = next((o for o in tensors if not o.is_floating_point() and o.shape[-1] == top_k), None)
        sc = next((o for o in tensors if o.is_floating_point() and o.shape[-1] == top_k and o is not full), None)
        return full, sc, idx, is_p

    core_routers = [(phys, router_of(parts.layers[phys])) for phys in range(lo, hi)]
    core_routers = [(phys, rmod) for phys, rmod in core_routers if rmod is not None]

    try:
        # 1. Baseline NLL + record Pass-1 Top-k expert assignments
        call_box = [0]
        rec_handles = []
        for phys, rmod in core_routers:
            def make_rec(p_layer: int):
                def _rec(_m: nn.Module, _inp: Any, out: Any) -> None:
                    full, _sc, idx, is_p = _extract_router(p_layer, out)
                    probs = upcast(full.reshape(-1, num_experts)) if is_p else F.softmax(
                        upcast(full.reshape(-1, num_experts)), dim=-1
                    )
                    top_1 = idx.detach().reshape(-1, top_k).long() if (idx is not None and top_k != num_experts) else (
                        probs.detach().topk(top_k, dim=-1).indices
                    )
                    base_routes[(call_box[0], p_layer)] = top_1
                return _rec
            rec_handles.append(rmod.register_forward_hook(make_rec(phys)))

        base_nlls = []
        try:
            for b_idx, b_ids in enumerate(batches):
                call_box[0] = b_idx
                base_nlls.append(_token_nll(model, b_ids).detach().cpu().double())
        finally:
            for h in rec_handles:
                h.remove()
        base_nll_vec = torch.cat(base_nlls, dim=0)

        n_pred = int(base_nll_vec.numel())
        n_hard = max(1, math.ceil((1.0 - hard_quantile) * n_pred))
        hard_idx = base_nll_vec.topk(n_hard, largest=True).indices

        # 2. Perturbed Top-k router trials across core layers
        trial_vecs: list[torch.Tensor] = []
        trial_jaccards: list[float] = []
        for trial in range(n_perturb):
            jacc_acc: list[float] = []
            pert_handles = []
            repel_frac = (trial + 1) / max(n_perturb, 1)

            for phys, rmod in core_routers:
                def make_pert(
                    p_layer: int,
                    trial_id: int = trial,
                    r_frac: float = repel_frac,
                    j_sink: list[float] = jacc_acc,
                ):
                    def _pert(mod: nn.Module, inp: Any, out: Any) -> Any:
                        b_idx = call_box[0]
                        full, sc, idx, is_p = _extract_router(p_layer, out)
                        flat_x = inp[0].reshape(-1, inp[0].shape[-1])
                        x_up = upcast(flat_x)
                        w_mod = getattr(mod, "weight", None)
                        w_up: torch.Tensor | None = None
                        if is_quantized_module(mod) or (
                            isinstance(w_mod, torch.Tensor) and w_mod.dtype in FP8_DTYPES
                        ):
                            try:
                                w_deq = _dequant_linear_weight(mod, torch.float32)
                            except (RuntimeError, TypeError, ValueError):
                                w_deq = None
                            if isinstance(w_deq, torch.Tensor) and w_deq.shape == (num_experts, flat_x.shape[-1]):
                                w_up = w_deq.to(dtype=x_up.dtype)
                        elif (
                            isinstance(w_mod, torch.Tensor)
                            and w_mod.shape == (num_experts, flat_x.shape[-1])
                            and w_mod.is_floating_point()
                        ):
                            w_up = upcast(w_mod).to(dtype=x_up.dtype)

                        if w_up is not None:
                            b_mod = getattr(mod, "bias", None)
                            bias_up = (
                                upcast(b_mod).to(dtype=x_up.dtype) if isinstance(b_mod, torch.Tensor) else None
                            )
                            base_logits = F.linear(x_up, w_up, bias_up)
                        elif is_p:
                            base_logits = upcast(full.reshape(-1, num_experts)).clamp_min(1e-30).log()
                        else:
                            base_logits = upcast(full.reshape(-1, num_experts))

                        logit_std = float(base_logits.detach().float().std().clamp_min(1e-3))
                        gen = torch.Generator(device="cpu").manual_seed(
                            seed + trial_id * 7919 + p_layer * 131 + b_idx * 17
                        )
                        noise = torch.randn(base_logits.shape, generator=gen, dtype=torch.float32) * (
                            logit_std * noise_std
                        )
                        delta = noise.to(device=base_logits.device, dtype=base_logits.dtype)
                        ref_top = base_routes.get((b_idx, p_layer))
                        if ref_top is not None and ref_top.shape[0] == base_logits.shape[0]:
                            mask = torch.zeros_like(base_logits).scatter_(1, ref_top, 1.0)
                            delta = delta - (0.45 * r_frac * logit_std) * mask

                        new_logits = base_logits + delta
                        probs_r = F.softmax(new_logits, dim=-1)
                        top_w, top_idx = probs_r.topk(top_k, dim=-1)
                        if norm_topk:
                            top_w = top_w / top_w.sum(-1, keepdim=True)
                        if ref_top is not None and ref_top.shape == top_idx.shape:
                            j_sink.append(_topk_jaccard(ref_top, top_idx))

                        if not isinstance(out, (tuple, list)):
                            return new_logits.to(out.dtype).view_as(out)
                        new_full = (probs_r if is_p else new_logits).to(full.dtype).view_as(full)
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
                    return _pert
                pert_handles.append(rmod.register_forward_hook(make_pert(phys)))

            try:
                t_nlls = []
                for b_idx, b_ids in enumerate(batches):
                    call_box[0] = b_idx
                    t_nlls.append(_token_nll(model, b_ids).detach().cpu().double())
                trial_vecs.append(torch.cat(t_nlls, dim=0))
                trial_jaccards.append(sum(jacc_acc) / max(len(jacc_acc), 1))
            finally:
                for h in pert_handles:
                    h.remove()
    finally:
        if was_training:
            model.train()

    stacked_trials = torch.stack(trial_vecs, dim=0)  # [n_perturb, N_pred]
    raw_min_nll = stacked_trials.min(dim=0).values
    oracle_nll = torch.minimum(base_nll_vec, raw_min_nll)
    hard_headroom = float((base_nll_vec[hard_idx] - oracle_nll[hard_idx]).mean())
    all_headroom = float((base_nll_vec - oracle_nll).mean())
    unclamped_hard = float((base_nll_vec[hard_idx] - raw_min_nll[hard_idx]).mean())
    improved_hard_frac = float((raw_min_nll[hard_idx] < base_nll_vec[hard_idx]).double().mean())
    mean_jacc = sum(trial_jaccards) / max(len(trial_jaccards), 1)

    rep = {
        "gate": "0p-1",
        "ok": True,
        "passed": bool(hard_headroom >= pass_threshold_nats),
        "pass_threshold_nats": float(pass_threshold_nats),
        "n_perturb": int(n_perturb),
        "noise_std": float(noise_std),
        "hard_quantile": float(hard_quantile),
        "core_range": [int(lo), int(hi)],
        "num_experts": num_experts,
        "top_k": top_k,
        "num_tokens_evaluated": n_pred,
        "num_hard_tokens": n_hard,
        "base_nll": float(base_nll_vec.mean()),
        "hard_token_base_nll": float(base_nll_vec[hard_idx].mean()),
        "oracle_perturbed_nll": float(oracle_nll.mean()),
        "hard_token_oracle_nll": float(oracle_nll[hard_idx].mean()),
        "all_token_headroom_nats": all_headroom,
        "hard_token_headroom_nats": hard_headroom,
        "unclamped_hard_delta_nats": unclamped_hard,
        "improved_hard_token_frac": improved_hard_frac,
        "mean_router_jaccard": float(mean_jacc),
        "trial_mean_nlls": [float(v.mean()) for v in trial_vecs],
        "trial_router_jaccards": [float(j) for j in trial_jaccards],
        "seconds": time.time() - t0,
    }
    log.info(
        "Gate 0p-1 expert headroom: hard_headroom=%.4f nats (base=%.3f -> oracle=%.3f, jaccard=%.3f, pass=%s)",
        hard_headroom, rep["hard_token_base_nll"], rep["hard_token_oracle_nll"], mean_jacc, rep["passed"],
    )
    return _to_jsonable(rep)


# ---------------------------------------------------------------------------------------- Gate 0p-2: re-entry anchors

def _rms64(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return x.pow(2).mean(dim=-1, keepdim=True).add(eps).sqrt()


@torch.no_grad()
def probe_reentry_anchors(
    model: nn.Module,
    ids: torch.Tensor | list[torch.Tensor],
    anchor_weights: tuple[float, ...] | list[float] = (0.99, 0.90, 0.85, 0.70, 0.50),
    subspace_rank: int = 64,
    *,
    prelude_end: int | None = None,
    core_end: int | None = None,
    pass_threshold_cosine: float = 0.90,
) -> dict[str, Any]:
    """Gate 0p-2: Subspace alignment (`core_end-1 -> prelude_end`), Procrustes variance & anchor sweep."""
    require(len(anchor_weights) >= 1, "anchor_weights must be non-empty")
    require(subspace_rank >= 1, "subspace_rank must be >= 1", subspace_rank=subspace_rank)
    for w in anchor_weights:
        require(0.0 < float(w) < 1.0, "each anchor weight must be strictly in (0, 1)", weight=w)
    t0 = time.time()
    parts = inspect_model(model)
    batches = _as_batches(ids)
    def_p, def_c = _default_core_bounds(parts.layer_types)
    p_end = int(prelude_end if prelude_end is not None else def_p)
    c_end = int(core_end if core_end is not None else def_c)
    require(0 < p_end < c_end <= parts.num_layers, "invalid prelude_end/core_end", p_end=p_end, c_end=c_end)

    was_training = model.training
    model.eval()
    try:
        # 1. Extract h_anchor (Layer prelude_end input) and h_core_exit (Layer core_end-1 output)
        anc_chunks: list[torch.Tensor] = []
        exit_chunks: list[torch.Tensor] = []
        base_losses: list[float] = []
        for b_ids in batches:
            out = parts.text_model(input_ids=b_ids, output_hidden_states=True)
            hs = out.hidden_states
            require(hs is not None and len(hs) > c_end, "need output_hidden_states through core_end", c_end=c_end)
            assert hs is not None
            d = hs[p_end].shape[-1]
            anc_chunks.append(hs[p_end].detach().reshape(-1, d).cpu().double())
            exit_chunks.append(hs[c_end].detach().reshape(-1, d).cpu().double())
            base_losses.append(float(_lm_loss(model, b_ids)))

        h_anc = torch.cat(anc_chunks, dim=0)
        h_exit = torch.cat(exit_chunks, dim=0)
        base_nll = sum(base_losses) / len(base_losses)
        n_tok, hidden = h_anc.shape
        eff_rank = min(int(subspace_rank), hidden, n_tok)

        raw_cos = float(F.cosine_similarity(h_exit, h_anc, dim=-1).mean())
        rms_ratio = float((_rms64(h_exit) / _rms64(h_anc)).mean())

        # 2. Orthogonal Procrustes + Low-Rank Subspace Alignment in fp64
        u_norm = h_exit / _rms64(h_exit)
        y_norm = h_anc / _rms64(h_anc)
        cov = u_norm.T @ y_norm
        p_mat, s_vals, vh_mat = torch.linalg.svd(cov, full_matrices=False)
        q_orth = p_mat @ vh_mat
        y_procrustes = (u_norm @ q_orth) * _rms64(h_anc)
        procrustes_cos = float(F.cosine_similarity(y_procrustes, h_anc, dim=-1).mean())
        procrustes_var = float(s_vals[:eff_rank].pow(2).sum() / s_vals.pow(2).sum().clamp_min(1e-12))

        gram = u_norm.T @ u_norm + 1e-6 * torch.eye(hidden, dtype=torch.float64)
        w_ls = torch.linalg.solve(gram, u_norm.T @ h_anc)
        y_ls = u_norm @ w_ls
        u_s, s_s, vh_s = torch.linalg.svd(y_ls, full_matrices=False)
        y_subspace = (u_s[:, :eff_rank] * s_s[:eff_rank]) @ vh_s[:eff_rank, :]
        subspace_cos = float(F.cosine_similarity(y_subspace, h_anc, dim=-1).mean())
        centered_anc = h_anc - h_anc.mean(dim=0, keepdim=True)
        denom_var = centered_anc.pow(2).sum().clamp_min(1e-12)
        subspace_r2 = float(max(0.0, 1.0 - float((h_anc - y_subspace).pow(2).sum() / denom_var)))

        # 3. Sweep across anchor weights to measure manifold cosine & Pass-2 routing Jaccard
        sweep_rows: list[dict[str, Any]] = []
        for w_anc_raw in anchor_weights:
            w_anc = float(w_anc_raw)
            rem = max(1.0 - w_anc, 1e-6)
            w_prev = rem / 3.0
            w_rms = (2.0 * rem) / 3.0
            bridge_init = (w_prev, w_anc, w_rms)

            matched = h_exit * (_rms64(h_anc) / _rms64(h_exit))
            bridged = w_prev * h_exit + w_anc * h_anc + w_rms * matched
            manifold_cos = float(F.cosine_similarity(bridged, h_anc, dim=-1).mean())

            spec = LoopSpec(
                prelude_end=p_end,
                core_end=c_end,
                num_passes=2,
                target_virtual_layers=None,
                bridge_init=bridge_init,
                bridge_rank=0,
                gate_init=1.0,
                per_pass_router=False,
                collect_routing=True,
            )
            ctl = apply_loop(model, spec)
            try:
                jacc_list: list[float] = []
                reroute_list: list[float] = []
                loop_losses: list[float] = []
                for b_ids in batches:
                    loop_losses.append(float(_lm_loss(model, b_ids)))
                    st = ctl.stats()
                    if "loop/expert_jaccard_p12" in st:
                        jacc_list.append(st["loop/expert_jaccard_p12"])
                    if "loop/reroute_rate" in st:
                        reroute_list.append(st["loop/reroute_rate"])
            finally:
                ctl.unwrap()

            zs_nll = sum(loop_losses) / len(loop_losses)
            jacc = sum(jacc_list) / len(jacc_list) if jacc_list else 1.0
            reroute = sum(reroute_list) / len(reroute_list) if reroute_list else 0.0
            sweep_rows.append({
                "anchor_weight": w_anc,
                "bridge_init": [round(w_prev, 6), round(w_anc, 6), round(w_rms, 6)],
                "manifold_cosine": manifold_cos,
                "pass2_routing_jaccard": jacc,
                "pass2_reroute_rate": reroute,
                "zero_shot_loop_nll": zs_nll,
                "nll_increase": zs_nll - base_nll,
            })
    finally:
        if was_training:
            model.train()

    # Choose recommended anchor: prefer ~0.85 sweet spot (manifold_cosine >= 0.85 without 0.99 lock-in)
    viable = [r for r in sweep_rows if r["manifold_cosine"] >= 0.85 and r["anchor_weight"] <= 0.92]
    if viable:
        best_row = min(viable, key=lambda r: (abs(r["anchor_weight"] - 0.85), r["nll_increase"]))
    else:
        best_row = max(sweep_rows, key=lambda r: r["manifold_cosine"] - 0.25 * max(r["nll_increase"], 0.0))

    best_align_cos = max(procrustes_cos, subspace_cos)
    rep = {
        "gate": "0p-2",
        "ok": True,
        "passed": bool(best_align_cos >= pass_threshold_cosine or procrustes_var >= 0.70),
        "pass_threshold_cosine": float(pass_threshold_cosine),
        "prelude_end": p_end,
        "core_end": c_end,
        "subspace_rank": int(subspace_rank),
        "effective_rank": eff_rank,
        "base_nll": base_nll,
        "rms_ratio_exit_to_anchor": rms_ratio,
        "raw_alignment_cosine": raw_cos,
        "procrustes_cosine": procrustes_cos,
        "subspace_cosine": subspace_cos,
        "procrustes_explained_variance": procrustes_var,
        "explained_variance": subspace_r2,
        "sweep": sweep_rows,
        "recommended_anchor_weight": float(best_row["anchor_weight"]),
        "recommended_bridge_init": list(
            V3_BRIDGE_WEIGHTS if abs(best_row["anchor_weight"] - 0.85) < 1e-4 else best_row["bridge_init"]
        ),
        "seconds": time.time() - t0,
    }
    log.info(
        "Gate 0p-2 reentry anchors: raw_cos=%.3f -> procrustes=%.3f / subspace=%.3f (var=%.3f), rec_anchor=%.2f",
        raw_cos, procrustes_cos, subspace_cos, procrustes_var, rep["recommended_anchor_weight"],
    )
    return _to_jsonable(rep)


# ---------------------------------------------------------------------------------------- Gate 0p-3: 1M RoPE / ReRoPE

def _yarn_inv_freq(
    inv_freq: torch.Tensor,
    orig_max_pos: int,
    factor: float,
    alpha: float = 1.0,
    beta: float = 32.0,
) -> tuple[torch.Tensor, float]:
    if factor <= 1.0:
        return inv_freq.clone(), 1.0
    freq = inv_freq.double()
    wavelength = (2.0 * math.pi) / freq.clamp_min(1e-30)
    rotations = float(orig_max_pos) / wavelength
    ramp = ((rotations - alpha) / max(beta - alpha, 1e-6)).clamp(0.0, 1.0)
    # ramp=1 -> high-freq (keep untouched); ramp=0 -> low-freq (divide by factor)
    scaled = (1.0 - ramp) * (freq / float(factor)) + ramp * freq
    mscale = 1.0 + 0.1 * math.log(float(factor))
    return scaled, mscale


def _rerope_clamp_positions(pos: torch.Tensor, orig_max_pos: int) -> torch.Tensor:
    lc = 0.5 * float(orig_max_pos)
    sc = 0.125 * float(orig_max_pos)
    pmax = (240000.0 / 262144.0) * float(orig_max_pos)
    over = (pos - lc).clamp_min(0.0)
    clamped = lc + sc * torch.log1p(over / max(sc, 1.0))
    return torch.where(pos <= lc, pos, clamped.clamp_max(pmax))


def _resolve_rotary_params(parts: Any, seq_len: int) -> tuple[nn.Module | None, torch.Tensor, int, int]:
    tc = parts.text_config
    rotary = getattr(parts.text_model, "rotary_emb", None)
    head_dim = int(getattr(tc, "head_dim", None) or (tc.hidden_size // tc.num_attention_heads))
    raw_rope_params = getattr(tc, "rope_parameters", None) or {}
    rope_params = (
        raw_rope_params.get("full_attention")
        if isinstance(raw_rope_params, dict) and isinstance(raw_rope_params.get("full_attention"), dict)
        else raw_rope_params
    )
    if not isinstance(rope_params, dict):
        rope_params = {}
    prf = float(getattr(tc, "partial_rotary_factor", None) or rope_params.get("partial_rotary_factor", 0.25))
    rot_dim = max(2, 2 * (round(head_dim * prf) // 2))
    theta = float(getattr(tc, "rope_theta", None) or rope_params.get("rope_theta", 10000.0))
    inv = getattr(rotary, "inv_freq", None) if rotary is not None else None
    if isinstance(inv, torch.Tensor) and inv.numel() >= 1:
        inv_freq = inv.detach().double().cpu().clone()
        rot_dim = 2 * int(inv_freq.numel())
    else:
        inv_freq = 1.0 / (theta ** (torch.arange(0, rot_dim, 2, dtype=torch.float64) / rot_dim))
    cfg_max = int(getattr(tc, "max_position_embeddings", 0) or 0)
    if cfg_max > 0:
        orig_max = cfg_max
    else:
        min_freq = float(inv_freq.min().clamp_min(1e-12))
        quarter_wave = round(math.pi / (4.0 * min_freq))
        orig_max = max(seq_len, min(262144, max(64, quarter_wave)))
    return rotary, inv_freq, rot_dim, orig_max


def _apply_half_rotation_64(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    r = cos.shape[-1]
    xr, xp = x[..., :r], x[..., r:]
    a, b = xr.chunk(2, dim=-1)
    rot = xr * cos + torch.cat([-b, a], dim=-1) * sin
    return torch.cat([rot, xp], dim=-1)


def _synthetic_needle_margin(
    mode: str,
    inv_freq: torch.Tensor,
    head_dim: int,
    num_heads: int,
    orig_max_pos: int,
    factor: float,
    seed: int = 2026,
) -> tuple[float, float]:
    """Measure rotary needle-in-a-haystack Q.K logit margin & retrieval accuracy at `factor * orig_max_pos`.

    Simulates a query retrieving a semantic key stored at early position `0.1 * L_target` from query
    position `0.9 * L_target` (`Delta = 0.8 * factor * L_0` > `L_0`) amidst `31` distractor keys,
    calibrated to a head whose rotary subspace expects in-domain phases from `[0, L_0]`.
    """
    gen = torch.Generator(device="cpu").manual_seed(seed)
    n_trials = 16
    n_hay = 32
    rot_dim = 2 * int(inv_freq.numel())
    nope_dim = max(head_dim - rot_dim, 0)
    target_len = float(orig_max_pos) * float(factor)

    # Horizon-crossing wavelength band [~4*L_0, ~8*L_0] where in-domain phase stays in [0, pi/2]
    wavelen = (2.0 * math.pi) / inv_freq.clamp_min(1e-30)
    center_wl = max(1.5 * float(factor) * float(orig_max_pos), 1.0)
    band_w = torch.exp(-0.5 * (torch.log(wavelen / center_wl) / 0.65) ** 2) + 0.05
    band_w = band_w / band_w.norm().clamp_min(1e-12) * math.sqrt(rot_dim / 2.0)

    pos_keys = torch.linspace(0.02 * target_len, 0.88 * target_len, steps=n_hay, dtype=torch.float64)
    pos_query = torch.tensor([0.92 * target_len], dtype=torch.float64)

    if mode == "yarn":
        eff_inv, mscale = _yarn_inv_freq(inv_freq, orig_max_pos, factor)
        pk_eff, pq_eff = pos_keys, pos_query
    elif mode == "rerope":
        eff_inv, mscale = inv_freq, 1.0
        # Query-time relative distance clamping (ARCHITECTURE_SPEC §6.1)
        rel = _rerope_clamp_positions((pos_query - pos_keys).clamp_min(0.0), orig_max_pos)
        pk_eff = torch.zeros_like(pos_keys)
        pq_eff = rel
    else:
        eff_inv, mscale = inv_freq, 1.0
        pk_eff, pq_eff = pos_keys, pos_query

    margins: list[float] = []
    hits = 0
    total = n_trials * max(num_heads, 1)
    for tidx in range(total):
        needle_slot = tidx % 6  # multi-depth early-context needles (~2%..16% of target_len)
        r_dir = F.normalize(torch.randn(rot_dim // 2, generator=gen, dtype=torch.float64), dim=-1) * band_w
        proto_rot = torch.cat([r_dir, r_dir], dim=-1) * math.sqrt(0.8 * head_dim / max(rot_dim, 1))
        if nope_dim > 0:
            proto_nope = F.normalize(torch.randn(nope_dim, generator=gen, dtype=torch.float64), dim=-1) * math.sqrt(
                0.2 * head_dim
            )
            proto = torch.cat([proto_rot, proto_nope], dim=-1)
        else:
            proto = proto_rot
        k_raw = torch.randn(n_hay, head_dim, generator=gen, dtype=torch.float64)
        k_raw = k_raw * torch.rsqrt(k_raw.pow(2).mean(dim=-1, keepdim=True) + 1e-8)
        k_raw[needle_slot] = proto * torch.rsqrt(proto.pow(2).mean(dim=-1, keepdim=True) + 1e-8)
        q_raw = proto + 0.12 * torch.randn(head_dim, generator=gen, dtype=torch.float64)
        q_raw = q_raw * torch.rsqrt(q_raw.pow(2).mean(dim=-1, keepdim=True) + 1e-8)

        if mode == "rerope":
            f_rel = pq_eff[:, None] * eff_inv[None, :]
            c_rel = torch.cat([f_rel, f_rel], dim=-1).cos() * mscale
            s_rel = torch.cat([f_rel, f_rel], dim=-1).sin() * mscale
            q_rot = _apply_half_rotation_64(q_raw.expand(n_hay, head_dim), c_rel, s_rel)
            scores = (q_rot * k_raw).sum(dim=-1) / math.sqrt(head_dim)
        else:
            fq = pq_eff[:, None] * eff_inv[None, :]
            fk = pk_eff[:, None] * eff_inv[None, :]
            cq = torch.cat([fq, fq], dim=-1).cos() * mscale
            sq = torch.cat([fq, fq], dim=-1).sin() * mscale
            ck = torch.cat([fk, fk], dim=-1).cos() * mscale
            sk = torch.cat([fk, fk], dim=-1).sin() * mscale
            q_rot = _apply_half_rotation_64(q_raw.unsqueeze(0), cq, sq)
            k_rot = _apply_half_rotation_64(k_raw, ck, sk)
            scores = (q_rot * k_rot).sum(dim=-1) / math.sqrt(head_dim)

        needle_sc = float(scores[needle_slot])
        distractor_mask = torch.ones(n_hay, dtype=torch.bool)
        distractor_mask[needle_slot] = False
        max_dist = float(scores[distractor_mask].max())
        margins.append(needle_sc - max_dist)
        if needle_sc > max_dist:
            hits += 1
    return sum(margins) / max(len(margins), 1), hits / max(total, 1)


def _build_passkey_ids(sample_ids: torch.Tensor, vocab_size: int) -> tuple[torch.Tensor, int]:
    seq = sample_ids[0].detach().clone()
    t_len = max(int(seq.numel()), 16)
    if seq.numel() < t_len:
        reps = (t_len + seq.numel() - 1) // seq.numel()
        seq = seq.repeat(reps)[:t_len]
    v = max(int(vocab_size), 8)
    c0, c1, c2 = 2 % v, 3 % v, 4 % v
    passkey_tok = (int(seq[0].item()) + 7) % v
    if passkey_tok in (c0, c1, c2):
        passkey_tok = (passkey_tok + 3) % v
    ins = min(2, max(0, t_len - 8))
    seq[ins] = c0
    seq[ins + 1] = c1
    seq[ins + 2] = c2
    seq[ins + 3] = passkey_tok
    seq[-3] = c0
    seq[-2] = c1
    seq[-1] = c2
    return seq.unsqueeze(0), passkey_tok


@torch.no_grad()
def probe_positional_extrapolation(
    model: nn.Module,
    ids: torch.Tensor | list[torch.Tensor],
    modes: tuple[str, ...] | list[str] = ("native", "yarn", "rerope"),
    factor: float = 4.0,
) -> dict[str, Any]:
    """Gate 0p-3: Compare extrapolated NLL and synthetic passkey/needle retrieval across positional modes."""
    valid_modes = {"native", "yarn", "rerope"}
    mode_list = [str(m) for m in modes]
    require(bool(mode_list), "modes must be non-empty")
    unknown = [m for m in mode_list if m not in valid_modes]
    if unknown:
        raise ValueError(f"unknown positional modes {unknown}; valid: {sorted(valid_modes)}")
    if factor < 1.0:
        raise ValueError(f"extrapolation factor must be >= 1.0, got {factor}")

    t0 = time.time()
    parts = inspect_model(model)
    batches = _as_batches(ids)
    tc = parts.text_config
    seq_len = int(batches[0].shape[1])
    rotary_mod, inv_freq, rot_dim, orig_max_pos = _resolve_rotary_params(parts, seq_len)
    target_max_pos = round(orig_max_pos * float(factor))
    head_dim = int(getattr(tc, "head_dim", None) or (tc.hidden_size // tc.num_attention_heads))
    num_heads = int(getattr(tc, "num_attention_heads", 4))
    vocab_size = int(getattr(tc, "vocab_size", 97))
    pk_ids, pk_tok = _build_passkey_ids(batches[0], vocab_size)

    was_training = model.training
    model.eval()
    try:
        in_domain_nll = sum(float(_lm_loss(model, b)) for b in batches) / len(batches)
        mode_results: dict[str, dict[str, Any]] = {}

        for mode in mode_list:
            eff_inv, mscale = (
                _yarn_inv_freq(inv_freq, orig_max_pos, factor)
                if mode == "yarn"
                else (inv_freq, 1.0)
            )

            def _rotary_hook(
                _m: nn.Module,
                inp: Any,
                kw: dict[str, Any],
                out: Any,
                _mode: str = mode,
                _inv: torch.Tensor = eff_inv,
                _ms: float = mscale,
            ) -> Any:
                if not (isinstance(out, tuple) and len(out) >= 2):
                    return out
                orig_cos, orig_sin = out[0], out[1]
                if not (isinstance(orig_cos, torch.Tensor) and isinstance(orig_sin, torch.Tensor)):
                    return out
                pos_ids = inp[1] if (isinstance(inp, tuple) and len(inp) >= 2) else kw.get("position_ids")
                if not isinstance(pos_ids, torch.Tensor):
                    return out
                # Collapse leading MRoPE stream axis (e.g. [3, B, T] -> [B, T] on Qwen3.5/3.6 VLM)
                while pos_ids.ndim > orig_cos.ndim - 1 and pos_ids.shape[1:] == orig_cos.shape[:-1]:
                    pos_ids = pos_ids[0]
                if pos_ids.shape != orig_cos.shape[:-1]:
                    return out
                t_cur = max(int(pos_ids.shape[-1]), 2)
                span = float(max(target_max_pos - 1, 1))
                stretched = pos_ids.double() * (span / float(t_cur - 1))
                if _mode == "rerope":
                    stretched = _rerope_clamp_positions(stretched, orig_max_pos)
                inv_dev = _inv.to(device=pos_ids.device, dtype=torch.float64)
                freqs = stretched[..., None] * inv_dev
                emb = torch.cat([freqs, freqs], dim=-1)
                if emb.numel() != orig_cos.numel():
                    return out
                new_cos = (emb.cos() * _ms).to(dtype=orig_cos.dtype).reshape_as(orig_cos)
                new_sin = (emb.sin() * _ms).to(dtype=orig_sin.dtype).reshape_as(orig_sin)
                return (new_cos, new_sin, *out[2:])

            handle = (
                rotary_mod.register_forward_hook(_rotary_hook, with_kwargs=True)
                if rotary_mod is not None
                else None
            )
            try:
                ext_nll = sum(float(_lm_loss(model, b)) for b in batches) / len(batches)
                pk_logits = model(input_ids=pk_ids).logits[0, -1].float()
                target_logit = float(pk_logits[pk_tok])
                mean_other = float((pk_logits.sum() - pk_logits[pk_tok]) / max(pk_logits.numel() - 1, 1))
                lm_margin = target_logit - mean_other
            finally:
                if handle is not None:
                    handle.remove()

            attn_margin, needle_acc = _synthetic_needle_margin(
                mode, inv_freq, head_dim, num_heads, orig_max_pos, factor
            )
            combined_margin = float(attn_margin + 0.05 * lm_margin)
            mode_results[mode] = {
                "mode": mode,
                "extrapolated_nll": ext_nll,
                "nll_degradation": ext_nll - in_domain_nll,
                "passkey_logit_margin": combined_margin,
                "attn_needle_margin": attn_margin,
                "lm_passkey_margin": lm_margin,
                "needle_retrieval_acc": needle_acc,
            }
    finally:
        if was_training:
            model.train()

    cand_modes = [m for m in mode_list if m != "native"] or mode_list
    best_mode = max(
        cand_modes,
        key=lambda m: (
            mode_results[m]["needle_retrieval_acc"],
            mode_results[m]["passkey_logit_margin"],
            -mode_results[m]["extrapolated_nll"],
        ),
    )
    best_entry = mode_results[best_mode]
    rep = {
        "gate": "0p-3",
        "ok": True,
        "passed": bool(best_entry["needle_retrieval_acc"] >= 0.90 or best_entry["passkey_logit_margin"] > 0.0),
        "factor": float(factor),
        "orig_max_position_embeddings": orig_max_pos,
        "target_max_position_embeddings": target_max_pos,
        "rotary_dim": rot_dim,
        "in_domain_nll": in_domain_nll,
        "modes": mode_results,
        "recommended_mode": best_mode,
        "best_passkey_logit_margin": float(best_entry["passkey_logit_margin"]),
        "best_needle_retrieval_acc": float(best_entry["needle_retrieval_acc"]),
        "seconds": time.time() - t0,
    }
    log.info(
        "Gate 0p-3 positional extrapolation (%.1fx): winner=%s (acc=%.3f, margin=%.3f, nll=%.3f)",
        factor, best_mode, rep["best_needle_retrieval_acc"], rep["best_passkey_logit_margin"],
        best_entry["extrapolated_nll"],
    )
    return _to_jsonable(rep)


# ---------------------------------------------------------------------------------------- structured suite & markdown

def format_probe_report_markdown(report: dict[str, Any]) -> str:
    """Format a `run_structured_probe_suite` output dict into a concise GitHub-Flavored Markdown report."""
    summ = report.get("summary") or {}
    gates = report.get("gates") or {}
    probes = report.get("probes") or {}
    overrides = report.get("recommended_overrides") or {}
    core_rows = report.get("core_ranges") or []

    lines = [
        "# Loop-HSSD v3 Zero-Training Diagnostic Probe Report",
        "",
        f"- **Model:** `{summ.get('model_type', 'unknown')}` "
        f"(`{summ.get('num_physical_layers', '?')}L`, `hidden={summ.get('hidden_size', '?')}`, "
        f"`experts={summ.get('num_experts', '?')} x top-{summ.get('top_k', '?')}`)",
        f"- **Target Virtual Depth:** `{summ.get('effective_target_virtual', '?')}L` "
        f"(requested `{summ.get('requested_target_virtual', '?')}L`)",
        f"- **Critical Hardware/Surgery Checks:** `{'PASS' if summ.get('all_critical_ok') else 'FAIL'}` "
        f"| **Gates Passed:** `{summ.get('num_gates_passed', 0)}/{summ.get('num_gates_total', 0)}` "
        f"| **Elapsed:** `{float(summ.get('total_seconds') or 0.0):.2f}s`",
        f"- **Frozen-MoE SHA-256:** `{str(summ.get('frozen_experts_sha256', ''))[:16]}...`",
        "",
        "## 1. Kill-Gate Summary (`Gate 0a` & `Gate 0p-1..3`)",
        "",
        "| Gate | Status | Metric | Value |",
        "| :--- | :---: | :--- | ---: |",
    ]
    for gname, ginfo in gates.items():
        status = "PASS" if ginfo.get("passed") else "WARN"
        val = ginfo.get("value")
        val_s = f"`{val:.4f}`" if isinstance(val, (int, float)) else f"`{val}`"
        lines.append(f"| `{gname}` | **{status}** | `{ginfo.get('metric', '-')}` | {val_s} |")

    eh = probes.get("expert_headroom") or {}
    if eh:
        lines.extend([
            "",
            "## 2. Gate `0p-1`: Frozen MoE Expert Headroom",
            "",
            f"- **Base Hard-Token NLL:** `{float(eh.get('hard_token_base_nll') or 0.0):.4f}` -> "
            f"**Oracle Top-k Perturbed NLL:** `{float(eh.get('hard_token_oracle_nll') or 0.0):.4f}` "
            f"(**Headroom:** `{float(eh.get('hard_token_headroom_nats') or 0.0):+.4f}` nats, "
            f"mean Jaccard `{float(eh.get('mean_router_jaccard') or 0.0):.3f}`)",
        ])

    ra = probes.get("reentry_anchors") or {}
    if ra:
        lines.extend([
            "",
            "## 3. Gate `0p-2`: Subspace Re-Entry Alignment & Anchor Sweep",
            "",
            f"- **Raw Alignment Cosine (`L{int(ra.get('core_end', 1)) - 1} -> L{ra.get('prelude_end', 0)}`):** "
            f"`{float(ra.get('raw_alignment_cosine') or 0.0):.4f}` | "
            f"**Procrustes Cosine:** `{float(ra.get('procrustes_cosine') or 0.0):.4f}` | "
            f"**Rank-`{ra.get('effective_rank', '?')}` Cosine:** `{float(ra.get('subspace_cosine') or 0.0):.4f}` "
            f"(spectral var `{float(ra.get('procrustes_explained_variance') or 0.0):.3f}`)",
            "",
            "| `w_anchor` | `bridge_init` | Manifold Cosine | Pass-2 Jaccard | Zero-Shot NLL Delta |",
            "| ---: | :--- | ---: | ---: | ---: |",
        ])
        for row in ra.get("sweep") or []:
            lines.append(
                f"| `{row['anchor_weight']:.2f}` | `{row['bridge_init']}` | "
                f"`{row['manifold_cosine']:.4f}` | `{row['pass2_routing_jaccard']:.4f}` | "
                f"`{row['nll_increase']:+.4f}` |"
            )

    pe = probes.get("positional_extrapolation") or {}
    if pe:
        lines.extend([
            "",
            f"## 4. Gate `0p-3`: Positional Extrapolation (`{pe.get('factor', 4.0)}x` Horizon)",
            "",
            "| Mode | Extrapolated NLL | Passkey Margin | Needle Retrieval Acc |",
            "| :--- | ---: | ---: | ---: |",
        ])
        for mname, minfo in (pe.get("modes") or {}).items():
            mark = " *(winner)*" if mname == pe.get("recommended_mode") else ""
            lines.append(
                f"| `{mname}`{mark} | `{minfo['extrapolated_nll']:.4f}` | "
                f"`{minfo['passkey_logit_margin']:+.4f}` | `{minfo['needle_retrieval_acc']:.3f}` |"
            )

    if core_rows:
        lines.extend([
            "",
            "## 5. Top ETD Core Schedule Candidates",
            "",
            "| `prelude_end` | `core_end` | `num_passes` | Core Angular | Zero-Shot Loss Delta |",
            "| ---: | ---: | ---: | ---: | ---: |",
        ])
        for crow in core_rows[:5]:
            delta_s = f"`{crow['loss_increase']:+.4f}`" if "loss_increase" in crow else "`-`"
            lines.append(
                f"| `{crow['prelude_end']}` | `{crow['core_end']}` | `{crow['num_passes']}` | "
                f"`{crow['core_angular']:.4f}` | {delta_s} |"
            )

    if overrides:
        lines.extend([
            "",
            "## 6. Recommended Calibrated Overrides",
            "",
            "```bash",
        ])
        for k, v in overrides.items():
            v_str = json.dumps(v, separators=(",", ":")) if isinstance(v, (list, dict, bool)) else str(v)
            lines.append(f"-s {k}={v_str}")
        lines.append("```")

    return "\n".join(lines) + "\n"


def run_structured_probe_suite(
    model: nn.Module,
    ids: torch.Tensor | list[torch.Tensor],
    target_virtual: int = 64,
    *,
    n_perturb: int = 4,
    anchor_weights: tuple[float, ...] = (0.99, 0.90, 0.85, 0.70, 0.50),
    subspace_rank: int = 64,
    pos_modes: tuple[str, ...] = ("native", "yarn", "rerope"),
    pos_factor: float = 4.0,
    decode_steps: int = 4,
) -> dict[str, Any]:
    """Run `Gate 0a` (`backward`, `surgery`) + `probe_core_ranges` + `Gate 0p-1..3` diagnostic suite."""
    t0 = time.time()
    parts = inspect_model(model)
    batches = _as_batches(ids)
    b0 = batches[0]
    was_training = model.training
    model.eval()

    with torch.no_grad():
        ref_logits = model(input_ids=b0).logits.detach().clone()
    sha_before = sha256_frozen_experts(model)

    # Resolve valid virtual-layer target for both 40L Qwen3.6 (64L) and smaller models (e.g. 16L Toy -> 24L)
    eff_target_virtual = int(target_virtual)
    if not candidate_specs(parts.layer_types, eff_target_virtual):
        p_def, c_def = _default_core_bounds(parts.layer_types)
        eff_target_virtual = parts.num_layers + (c_def - p_def)

    try:
        backward_rep = probe_backward(model, b0)
        core_rows = probe_core_ranges(model, batches, target_virtual=eff_target_virtual)
        if core_rows:
            best_core = core_rows[0]
            p_end, c_end, r_passes = (
                int(best_core["prelude_end"]),
                int(best_core["core_end"]),
                int(best_core["num_passes"]),
            )
        else:
            p_end, c_end = _default_core_bounds(parts.layer_types)
            r_passes = 2
            best_core = {"prelude_end": p_end, "core_end": c_end, "num_passes": r_passes}

        hidden = int(getattr(parts.text_config, "hidden_size", 32))
        embed_w = getattr(parts.text_model.embed_tokens, "weight", None)
        ref_dtype = embed_w.dtype if isinstance(embed_w, torch.Tensor) else ref_logits.dtype
        atol = 1e-9 if ref_dtype == torch.float64 else 2e-2
        loop_surgery_spec = {
            "prelude_end": p_end,
            "core_end": c_end,
            "num_passes": r_passes,
            "target_virtual_layers": parts.num_layers + (c_end - p_end) * (r_passes - 1),
            "bridge_mode": "subspace",
            "bridge_init": list(V3_BRIDGE_WEIGHTS),
            "bridge_rank": min(16, hidden),
            "attn_adapter_rank": min(8, hidden),
        }
        probe_seq_len = min(int(b0.shape[1]), 256)
        lsa_surgery_cfg: dict[str, Any] | None = None
        if probe_seq_len >= 16:
            blk = 8 if probe_seq_len < 128 else 64
            pool = 2 if blk == 8 else 4
            lsa_surgery_cfg = {
                "index_dim": min(64, max(4, hidden // 4)),
                "block_size": blk,
                "pool": pool,
                "avg_bucket_keys": pool * 2,
                "max_bucket_keys": blk,
                "window": blk,
                "sinks": min(64, max(2, blk // 4)),
                "num_select": min(64, max(2, blk // (pool * 2))),
            }
        has_linear = LINEAR in parts.layer_types
        gdn2_surgery_cfg = (
            {
                "rank": min(8, hidden),
                "chunk_size": 8 if probe_seq_len < 64 else 64,
                "segment_chunks": 2,
            }
            if has_linear
            else None
        )
        surgery_rep = surgery_check(
            model,
            b0,
            loop_surgery_spec,
            lsa_surgery_cfg,
            decode_steps=decode_steps,
            atol=atol,
            gdn2_cfg=gdn2_surgery_cfg,
        )
        hssd_surgery_rep: dict[str, Any] | None = None
        if has_linear:
            hssd_state_dtype = (
                "bfloat16"
                if (ref_logits.dtype != torch.float64 and _is_low_prec_or_quant_model(model, ref_logits.dtype))
                else "auto"
            )
            hssd_surgery_rep = surgery_check(
                model,
                b0,
                loop_surgery_spec,
                None,
                decode_steps=min(int(decode_steps), 4),
                atol=atol,
                hssd_cfg={
                    "rank": min(8, hidden),
                    "chunk_size": 8 if probe_seq_len < 64 else 64,
                    "window_chunks": 2,
                    "checkpoint_segments": False,
                    "state_dtype": hssd_state_dtype,
                },
            )

        headroom_rep = probe_expert_headroom(
            model, batches, n_perturb=n_perturb, core_range=(p_end, c_end)
        )
        reentry_rep = probe_reentry_anchors(
            model,
            batches,
            anchor_weights=anchor_weights,
            subspace_rank=subspace_rank,
            prelude_end=p_end,
            core_end=c_end,
        )
        pos_rep = probe_positional_extrapolation(model, batches, modes=pos_modes, factor=pos_factor)

        with torch.no_grad():
            after_logits = model(input_ids=b0).logits
        sha_after = sha256_frozen_experts(model)
        _assert_logits_unchanged(
            model,
            b0,
            after_logits,
            ref_logits,
            "probe suite left residual state on model",
        )
        require(sha_after == sha_before, "probe suite mutated frozen expert parameters")
    finally:
        if was_training:
            model.train()

    gates = {
        "0a_backward": {
            "passed": bool(backward_rep["ok"]),
            "metric": "input_grad_norm",
            "value": backward_rep["input_grad_norm"],
        },
        "0a_surgery": {
            "passed": True,
            "metric": "decode_vs_full_maxdiff",
            "value": surgery_rep["decode_vs_full_maxdiff"],
        },
        "0p_1_expert_headroom": {
            "passed": bool(headroom_rep["passed"]),
            "metric": "hard_token_headroom_nats",
            "value": headroom_rep["hard_token_headroom_nats"],
        },
        "0p_2_reentry_anchors": {
            "passed": bool(reentry_rep["passed"]),
            "metric": "procrustes_cosine",
            "value": reentry_rep["procrustes_cosine"],
        },
        "0p_3_positional_extrapolation": {
            "passed": bool(pos_rep["passed"]),
            "metric": "best_passkey_logit_margin",
            "value": pos_rep["best_passkey_logit_margin"],
        },
    }
    recommended_overrides = {
        "loop.prelude_end": p_end,
        "loop.core_end": c_end,
        "loop.num_passes": r_passes,
        "loop.target_virtual_layers": parts.num_layers + (c_end - p_end) * (r_passes - 1),
        "loop.bridge_mode": "subspace",
        "loop.bridge_init": list(reentry_rep["recommended_bridge_init"]),
        "loop.bridge_rank": int(subspace_rank),
        "loop.per_pass_router": True,
        "pos.mode": str(pos_rep["recommended_mode"]),
        "pos.factor": float(pos_factor),
    }
    summary = {
        "model_type": str(getattr(parts.text_config, "model_type", type(model).__name__)),
        "num_physical_layers": int(parts.num_layers),
        "hidden_size": int(getattr(parts.text_config, "hidden_size", 0)),
        "num_experts": int(getattr(parts.text_config, "num_experts", 0) or 0),
        "top_k": int(getattr(parts.text_config, "num_experts_per_tok", 0) or 0),
        "requested_target_virtual": int(target_virtual),
        "effective_target_virtual": int(eff_target_virtual),
        "all_critical_ok": bool(backward_rep["ok"]),
        "num_gates_passed": sum(1 for g in gates.values() if g["passed"]),
        "num_gates_total": len(gates),
        "frozen_experts_sha256": sha_after,
        "total_seconds": time.time() - t0,
    }
    report: dict[str, Any] = _to_jsonable({
        "schema_version": "qloop.probe_report.v3",
        "summary": summary,
        "gates": gates,
        "core_ranges": core_rows[:10],
        "probes": {
            "backward": backward_rep,
            "surgery": surgery_rep,
            **({"hssd_surgery": hssd_surgery_rep} if hssd_surgery_rep is not None else {}),
            "expert_headroom": headroom_rep,
            "reentry_anchors": reentry_rep,
            "positional_extrapolation": pos_rep,
        },
        "recommended_overrides": recommended_overrides,
    })
    report["markdown"] = format_probe_report_markdown(report)
    return report


def _sync_if_cuda(dev: torch.device) -> None:
    if dev.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(dev)


def _peak_cuda_bytes(dev: torch.device) -> int:
    if dev.type == "cuda" and torch.cuda.is_available():
        return int(torch.cuda.max_memory_allocated(dev))
    return 0


def _reset_cuda_peak(dev: torch.device) -> None:
    if dev.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(dev)
        torch.cuda.reset_peak_memory_stats(dev)


def _chunked_gdn_baseline_prefill(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g_base: torch.Tensor,
    beta_base: torch.Tensor,
    *,
    chunk_size: int = 64,
    state_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Single-state (`M=1`) chunked WY Gated DeltaNet baseline for apples-to-apples layer profiling."""
    from .hssd import _prepare_qkv_unscaled, _safe_sqrt

    init_dtype = query.dtype
    acc_dt = torch.float64 if init_dtype == torch.float64 else torch.float32
    bsz, seq_len, hv, dv = value.shape
    dk = key.shape[-1]
    q, k, v = _prepare_qkv_unscaled(query, key, value, True)
    q = q * (dk ** -0.5)
    gf = upcast(g_base).to(acc_dt).clamp(max=0.0)
    bf = upcast(beta_base).to(acc_dt).clamp(0.0, 2.0)

    state = q.new_zeros(bsz, hv, dk, dv)
    outs: list[torch.Tensor] = []
    for c0 in range(0, seq_len, chunk_size):
        c1 = min(seq_len, c0 + chunk_size)
        c_len = c1 - c0
        qc = q[:, c0:c1].transpose(1, 2)                         # [B, H_v, C, d_k]
        kc = k[:, c0:c1].transpose(1, 2)                         # [B, H_v, C, d_k]
        vc = v[:, c0:c1].transpose(1, 2)                         # [B, H_v, C, d_v]
        gc = gf[:, c0:c1].transpose(1, 2).unsqueeze(-1)          # [B, H_v, C, 1]
        bc = _safe_sqrt(bf[:, c0:c1].transpose(1, 2).unsqueeze(-1))

        ke = kc * bc
        ve = vc * bc
        g_cum = torch.cumsum(gc, dim=-2)                         # [B, H_v, C, 1] <= 0
        idx = torch.arange(c_len, device=q.device)
        strict_tril = idx[:, None] > idx[None, :]
        causal_tril = idx[:, None] >= idx[None, :]

        diff = (g_cum[:, :, :, None, 0] - g_cum[:, :, None, :, 0]).masked_fill(~causal_tril, 0.0)
        decay_mat = torch.exp(diff)
        gram = kc @ kc.transpose(-1, -2)
        strict_m = (gram * bc * bc.transpose(-1, -2) * decay_mat).masked_fill(~strict_tril, 0.0)
        eye = torch.eye(c_len, device=q.device, dtype=acc_dt)
        inv_l = torch.linalg.solve_triangular(eye + strict_m, eye, upper=False, unitriangular=True)

        u_v = inv_l @ ve
        w_k = inv_l @ (ke * torch.exp(g_cum))
        q_dec = qc * torch.exp(g_cum)
        attn = ((qc @ kc.transpose(-1, -2)) * bc.transpose(-1, -2) * decay_mat).masked_fill(~causal_tril, 0.0)
        o_c = (q_dec @ state) + ((attn - (q_dec @ w_k.transpose(-1, -2)).masked_fill(~causal_tril, 0.0)) @ u_v)
        outs.append(o_c.transpose(1, 2))

        g_tail = g_cum[:, :, -1:, :] - g_cum
        s_dec = state * torch.exp(g_cum[:, :, -1, :]).unsqueeze(-1)
        u_tilde = u_v - (w_k @ state)
        state = s_dec + (ke * torch.exp(g_tail)).transpose(-1, -2) @ u_tilde

    out = torch.cat(outs, dim=1).to(init_dtype) if outs else query.new_zeros(bsz, 0, hv, dv)
    return out, state.to(dtype=state_dtype).contiguous()


def _gdn_baseline_decode_step(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g_base: torch.Tensor,
    beta_base: torch.Tensor,
    state: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Single-token (`T=1`) single-state Gated DeltaNet decode step."""
    from .hssd import _prepare_qkv_unscaled

    init_dtype = query.dtype
    acc_dt = torch.float64 if init_dtype == torch.float64 else torch.float32
    dk = key.shape[-1]
    q, k, v = _prepare_qkv_unscaled(query, key, value, True)
    q = (q * (dk ** -0.5))[:, 0]
    kt = k[:, 0]
    vt = v[:, 0]
    gf = upcast(g_base[:, 0]).to(acc_dt).clamp(max=0.0)
    bf = upcast(beta_base[:, 0]).to(acc_dt).clamp(0.0, 2.0)

    s_acc = state.to(dtype=acc_dt) * torch.exp(gf)[..., None, None]
    mem = (s_acc * kt[..., :, None]).sum(dim=-2)
    delta = (vt - mem) * bf[..., None]
    s_next = s_acc + kt[..., :, None] * delta[..., None, :]
    out = (s_next * q[..., :, None]).sum(dim=-2).unsqueeze(1).to(dtype=init_dtype)
    return out, s_next.to(dtype=state.dtype).contiguous()


def profile_hssd_layer(
    *,
    hidden_size: int = 2048,
    num_v_heads: int = 32,
    head_k_dim: int = 128,
    head_v_dim: int = 128,
    seq_len: int = 1024,
    stream_chunk_len: int = 256,
    state_dtype: str = "bfloat16",
    theta_low_rank: bool = True,
    rank: int = 64,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
) -> dict[str, Any]:
    """Profile a single HSSD v3.1 linear-attention layer vs single-state GDN (`Gate 0p-4`).

    Benchmarks full-sequence prefill, streamed chunk prefill, retained decode cache unique backing
    bytes (at chunk boundaries and mid-chunk), single-token decode (`O(k)` active-page update), and
    training forward+backward with actual sparse Top-`k` STE write-budget loss.
    """
    from .hssd import (
        HSSDAdapter,
        HSSDConfig,
        _resolve_state_dtype,
        chunk_hssd,
        compute_write_budget_loss,
        has_fla,
        recurrent_hssd_step,
    )

    require(seq_len >= 2, "profile_hssd_layer requires seq_len >= 2", seq_len=seq_len)
    require(
        stream_chunk_len >= 1,
        "profile_hssd_layer requires stream_chunk_len >= 1",
        stream_chunk_len=stream_chunk_len,
    )
    require(head_k_dim >= 2 and head_k_dim % 2 == 0, "head_k_dim must be even >= 2", head_k_dim=head_k_dim)

    dev = torch.device(device)
    acc_dt = torch.float64 if dtype == torch.float64 else torch.float32
    resolved_state_dt = _resolve_state_dtype(state_dtype, acc_dt)

    cfg = HSSDConfig(
        rank=int(rank),
        theta_low_rank=bool(theta_low_rank),
        state_dtype=str(state_dtype),
        checkpoint_segments=True,
    )
    cfg.validate()

    gen = torch.Generator(device="cpu").manual_seed(2026)
    adapter = HSSDAdapter(
        int(hidden_size),
        int(num_v_heads),
        int(head_k_dim),
        int(head_v_dim),
        cfg,
        layer_id=0,
    ).to(device=dev, dtype=dtype)

    n_params = sum(p.numel() for p in adapter.parameters())
    if adapter.theta_down is not None and adapter.theta_up is not None:
        theta_params = adapter.theta_down.weight.numel() + adapter.theta_up.weight.numel()
    elif adapter.theta_proj is not None:
        theta_params = adapter.theta_proj.weight.numel()
    else:
        theta_params = 0

    bsz = 1
    tot_t = int(seq_len) + 1
    x_all = (torch.randn(bsz, tot_t, hidden_size, generator=gen, dtype=torch.float32) * 0.25).to(dev, dtype)
    q_all = torch.randn(bsz, tot_t, num_v_heads, head_k_dim, generator=gen, dtype=torch.float32).to(dev, dtype)
    k_all = torch.randn(bsz, tot_t, num_v_heads, head_k_dim, generator=gen, dtype=torch.float32).to(dev, dtype)
    v_all = torch.randn(bsz, tot_t, num_v_heads, head_v_dim, generator=gen, dtype=torch.float32).to(dev, dtype)
    g_all = (-0.02 - 0.08 * torch.rand(bsz, tot_t, num_v_heads, generator=gen, dtype=torch.float32)).to(dev, dtype)
    beta_all = (0.15 + 0.5 * torch.rand(bsz, tot_t, num_v_heads, generator=gen, dtype=torch.float32)).to(dev, dtype)

    x_pre, q_pre, k_pre, v_pre = x_all[:, :seq_len], q_all[:, :seq_len], k_all[:, :seq_len], v_all[:, :seq_len]
    g_pre, beta_pre = g_all[:, :seq_len], beta_all[:, :seq_len]

    # 1. Full prefill
    adapter.eval()
    with torch.no_grad():
        _reset_cuda_peak(dev)
        t0 = time.perf_counter()
        proj_pre = adapter.project_inputs(x_pre, q_pre, k_pre, v_pre, g_pre, beta_pre, use_qk_l2norm=True)
        gamma_factored = (proj_pre["gamma_page_scale"], proj_pre["gamma_chan_pair"])
        out_full, st_full = chunk_hssd(
            proj_pre["query"],
            proj_pre["key"],
            proj_pre["value"],
            gamma_factored,
            proj_pre["b_gate"],
            proj_pre["w_gate"],
            proj_pre["theta"],
            proj_pre["surprise_bias"],
            adapter.log_surprise_slope.exp(),
            adapter.surprise_thresh,
            proj_pre["write_logits"],
            proj_pre["read_logits"],
            proj_pre["q_swa"],
            proj_pre["k_swa"],
            proj_pre["v_swa"],
            boltzmann_gate=adapter.boltzmann_gate,
            chunk_size=cfg.chunk_size,
            window_chunks=cfg.window_chunks,
            num_pages=cfg.num_pages,
            top_k_pages=cfg.k_pages,
            segment_chunks=cfg.segment_chunks,
            checkpoint_segments=False,
            initial_state=None,
            output_final_state=True,
            use_qk_l2norm_in_kernel=False,
            state_dtype=cfg.state_dtype,
            use_swa=cfg.use_swa,
            use_phase=cfg.use_phase,
            use_surprise=cfg.use_surprise,
            surprise_mode=cfg.surprise_mode,
        )
        _sync_if_cuda(dev)
        full_sec = max(time.perf_counter() - t0, 1e-9)
        full_peak = _peak_cuda_bytes(dev)

    assert st_full is not None
    gamma_factored_bytes = int(
        proj_pre["gamma_page_scale"].numel() * proj_pre["gamma_page_scale"].element_size()
        + proj_pre["gamma_chan_pair"].numel() * proj_pre["gamma_chan_pair"].element_size()
    )
    gamma_legacy_5d_bytes = int(
        bsz * seq_len * cfg.num_pages * num_v_heads * head_k_dim * proj_pre["gamma_chan_pair"].element_size()
    )

    # 2. Streamed prefill in `stream_chunk_len` segments
    with torch.no_grad():
        _reset_cuda_peak(dev)
        t0 = time.perf_counter()
        st_stream = None
        stream_outs: list[torch.Tensor] = []
        n_segs = 0
        for s0 in range(0, seq_len, stream_chunk_len):
            s1 = min(seq_len, s0 + stream_chunk_len)
            n_segs += 1
            ps = adapter.project_inputs(
                x_pre[:, s0:s1],
                q_pre[:, s0:s1],
                k_pre[:, s0:s1],
                v_pre[:, s0:s1],
                g_pre[:, s0:s1],
                beta_pre[:, s0:s1],
                use_qk_l2norm=True,
            )
            o_seg, st_stream = chunk_hssd(
                ps["query"],
                ps["key"],
                ps["value"],
                (ps["gamma_page_scale"], ps["gamma_chan_pair"]),
                ps["b_gate"],
                ps["w_gate"],
                ps["theta"],
                ps["surprise_bias"],
                adapter.log_surprise_slope.exp(),
                adapter.surprise_thresh,
                ps["write_logits"],
                ps["read_logits"],
                ps["q_swa"],
                ps["k_swa"],
                ps["v_swa"],
                boltzmann_gate=adapter.boltzmann_gate,
                chunk_size=cfg.chunk_size,
                window_chunks=cfg.window_chunks,
                num_pages=cfg.num_pages,
                top_k_pages=cfg.k_pages,
                segment_chunks=cfg.segment_chunks,
                checkpoint_segments=False,
                initial_state=st_stream,
                output_final_state=True,
                use_qk_l2norm_in_kernel=False,
                state_dtype=cfg.state_dtype,
                use_swa=cfg.use_swa,
                use_phase=cfg.use_phase,
                use_surprise=cfg.use_surprise,
                surprise_mode=cfg.surprise_mode,
            )
            stream_outs.append(o_seg)
        out_stream = torch.cat(stream_outs, dim=1)
        _sync_if_cuda(dev)
        stream_sec = max(time.perf_counter() - t0, 1e-9)
        stream_peak = _peak_cuda_bytes(dev)

    assert st_stream is not None
    stream_diff = float((out_stream.float() - out_full.float()).abs().max().item())

    # 3. Single-token decode (T = 1, active-page O(k) state update)
    x_dec = x_all[:, seq_len : seq_len + 1]
    q_dec = q_all[:, seq_len : seq_len + 1]
    k_dec = k_all[:, seq_len : seq_len + 1]
    v_dec = v_all[:, seq_len : seq_len + 1]
    g_dec = g_all[:, seq_len : seq_len + 1]
    beta_dec = beta_all[:, seq_len : seq_len + 1]

    with torch.no_grad():
        _reset_cuda_peak(dev)
        t0 = time.perf_counter()
        pd = adapter.project_inputs(x_dec, q_dec, k_dec, v_dec, g_dec, beta_dec, use_qk_l2norm=True)
        _o_dec, st_decode = recurrent_hssd_step(
            pd["query"],
            pd["key"],
            pd["value"],
            (pd["gamma_page_scale"], pd["gamma_chan_pair"]),
            pd["b_gate"],
            pd["w_gate"],
            pd["theta"],
            pd["surprise_bias"],
            adapter.log_surprise_slope.exp(),
            adapter.surprise_thresh,
            pd["write_logits"],
            pd["read_logits"],
            pd["q_swa"],
            pd["k_swa"],
            pd["v_swa"],
            st_full,
            boltzmann_gate=adapter.boltzmann_gate,
            top_k_pages=cfg.k_pages,
            use_qk_l2norm_in_kernel=False,
            state_dtype=cfg.state_dtype,
            use_swa=cfg.use_swa,
            use_phase=cfg.use_phase,
            use_surprise=cfg.use_surprise,
            surprise_mode=cfg.surprise_mode,
        )
        _sync_if_cuda(dev)
        decode_sec = max(time.perf_counter() - t0, 1e-9)
        decode_peak = _peak_cuda_bytes(dev)

    # 4. Training forward + backward with actual sparse Top-k STE write-budget loss
    adapter.train()
    adapter.zero_grad(set_to_none=True)
    x_tr = x_pre.detach().clone().requires_grad_(True)
    q_tr = q_pre.detach().clone().requires_grad_(True)
    k_tr = k_pre.detach().clone().requires_grad_(True)
    v_tr = v_pre.detach().clone().requires_grad_(True)

    _reset_cuda_peak(dev)
    t0 = time.perf_counter()
    pt = adapter.project_inputs(x_tr, q_tr, k_tr, v_tr, g_pre, beta_pre, use_qk_l2norm=True)
    out_tr, _st_tr, duty_tr = chunk_hssd(
        pt["query"],
        pt["key"],
        pt["value"],
        (pt["gamma_page_scale"], pt["gamma_chan_pair"]),
        pt["b_gate"],
        pt["w_gate"],
        pt["theta"],
        pt["surprise_bias"],
        adapter.log_surprise_slope.exp(),
        adapter.surprise_thresh,
        pt["write_logits"],
        pt["read_logits"],
        pt["q_swa"],
        pt["k_swa"],
        pt["v_swa"],
        boltzmann_gate=adapter.boltzmann_gate,
        chunk_size=cfg.chunk_size,
        window_chunks=cfg.window_chunks,
        num_pages=cfg.num_pages,
        top_k_pages=cfg.k_pages,
        segment_chunks=cfg.segment_chunks,
        checkpoint_segments=cfg.checkpoint_segments,
        initial_state=None,
        output_final_state=False,
        use_qk_l2norm_in_kernel=False,
        state_dtype=cfg.state_dtype,
        use_swa=cfg.use_swa,
        use_phase=cfg.use_phase,
        use_surprise=cfg.use_surprise,
        surprise_mode=cfg.surprise_mode,
        return_aux=True,
    )
    mu_max_t = torch.tensor(cfg.resolved_mu_max(), device=duty_tr.device, dtype=duty_tr.dtype)
    bud_loss = compute_write_budget_loss(duty_tr, mu_max_t, mu_min_ratio=cfg.mu_min_ratio)
    total_loss = out_tr.float().pow(2).mean() + cfg.budget_weight * bud_loss
    total_loss.backward()
    _sync_if_cuda(dev)
    train_sec = max(time.perf_counter() - t0, 1e-9)
    train_peak = _peak_cuda_bytes(dev)

    grad_finite = bool(
        x_tr.grad is not None
        and torch.isfinite(x_tr.grad).all()
        and all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in adapter.parameters())
    )
    dead_pages = int((duty_tr.detach() < 1e-6).sum().item())
    adapter.zero_grad(set_to_none=True)

    # 5. Single-state Gated DeltaNet baseline (prefill + decode + state bytes)
    with torch.no_grad():
        _sync_if_cuda(dev)
        t0 = time.perf_counter()
        _gdn_out, gdn_state = _chunked_gdn_baseline_prefill(
            q_pre, k_pre, v_pre, g_pre, beta_pre, chunk_size=cfg.chunk_size, state_dtype=resolved_state_dt
        )
        _sync_if_cuda(dev)
        gdn_pre_sec = max(time.perf_counter() - t0, 1e-9)

        t0 = time.perf_counter()
        _gdn_dec_out, gdn_state_dec = _gdn_baseline_decode_step(
            q_dec, k_dec, v_dec, g_dec, beta_dec, gdn_state
        )
        _sync_if_cuda(dev)
        gdn_dec_sec = max(time.perf_counter() - t0, 1e-9)

    gdn_state_bytes = int(gdn_state_dec.untyped_storage().nbytes())
    retained_bytes = int(st_full.unique_storage_bytes())
    mid_chunk_bytes = int(st_decode.unique_storage_bytes())

    report = {
        "dimensions": {
            "batch_size": bsz,
            "hidden_size": int(hidden_size),
            "num_v_heads": int(num_v_heads),
            "head_k_dim": int(head_k_dim),
            "head_v_dim": int(head_v_dim),
            "seq_len": int(seq_len),
            "stream_chunk_len": int(stream_chunk_len),
            "chunk_size": int(cfg.chunk_size),
            "window_chunks": int(cfg.window_chunks),
            "num_pages": int(cfg.num_pages),
            "top_k_pages": int(cfg.k_pages),
            "rank": int(cfg.rank),
            "theta_low_rank": bool(cfg.theta_low_rank),
            "state_dtype": str(cfg.state_dtype),
            "input_dtype": str(dtype).replace("torch.", ""),
            "device": str(dev),
        },
        "adapter_params_per_layer": int(n_params),
        "adapter_params_30_layers": int(30 * n_params),
        "theta_param_fraction": float(theta_params / max(n_params, 1)),
        "full_prefill": {
            "seq_len": int(seq_len),
            "seconds": float(full_sec),
            "ms": float(full_sec * 1000.0),
            "tokens_per_sec": float(seq_len / full_sec),
            "peak_cuda_bytes": int(full_peak),
            "peak_cuda_mib": float(full_peak / (1024.0 * 1024.0)),
            "gamma_factored_bytes": gamma_factored_bytes,
            "gamma_legacy_5d_bytes": gamma_legacy_5d_bytes,
            "gamma_reduction_factor": float(gamma_legacy_5d_bytes / max(gamma_factored_bytes, 1)),
        },
        "streamed_prefill": {
            "seq_len": int(seq_len),
            "stream_chunk_len": int(stream_chunk_len),
            "num_segments": int(n_segs),
            "seconds": float(stream_sec),
            "ms": float(stream_sec * 1000.0),
            "tokens_per_sec": float(seq_len / stream_sec),
            "peak_cuda_bytes": int(stream_peak),
            "peak_cuda_mib": float(stream_peak / (1024.0 * 1024.0)),
            "max_diff_vs_full_prefill": stream_diff,
            "has_compact_storage": bool(st_stream.has_compact_storage()),
        },
        "retained_cache": {
            "state_dtype": str(st_full.page_states.dtype).replace("torch.", ""),
            "theta_cum_dtype": str(st_full.theta_cum.dtype).replace("torch.", ""),
            "theta_cum_storage_bytes": int(st_full.theta_cum.untyped_storage().nbytes()),
            "has_compact_storage": bool(st_full.has_compact_storage() and st_decode.has_compact_storage()),
            "unique_storage_bytes": retained_bytes,
            "unique_storage_mib": float(retained_bytes / (1024.0 * 1024.0)),
            "mid_chunk_unique_storage_bytes": mid_chunk_bytes,
            "mid_chunk_unique_storage_mib": float(mid_chunk_bytes / (1024.0 * 1024.0)),
            "30_layers_1pass_gib": float(30.0 * mid_chunk_bytes / (1024.0 ** 3)),
            "step_count": int(st_full.step_count_int),
        },
        "single_token_decode": {
            "seconds": float(decode_sec),
            "ms": float(decode_sec * 1000.0),
            "tokens_per_sec": float(1.0 / decode_sec),
            "peak_cuda_bytes": int(decode_peak),
            "active_write_pages": int(cfg.k_pages),
            "total_pages": int(cfg.num_pages),
            "has_compact_storage": bool(st_decode.has_compact_storage()),
        },
        "training_fwd_bwd": {
            "seq_len": int(seq_len),
            "seconds": float(train_sec),
            "ms": float(train_sec * 1000.0),
            "tokens_per_sec": float(seq_len / train_sec),
            "peak_cuda_bytes": int(train_peak),
            "peak_cuda_mib": float(train_peak / (1024.0 * 1024.0)),
            "grad_finite": grad_finite,
            "budget_loss": float(bud_loss.detach().cpu().item()),
            "dead_page_count": dead_pages,
        },
        "gdn_baseline": {
            "prefill_seconds": float(gdn_pre_sec),
            "prefill_ms": float(gdn_pre_sec * 1000.0),
            "prefill_tokens_per_sec": float(seq_len / gdn_pre_sec),
            "decode_seconds": float(gdn_dec_sec),
            "decode_ms": float(gdn_dec_sec * 1000.0),
            "decode_tokens_per_sec": float(1.0 / gdn_dec_sec),
            "state_bytes": gdn_state_bytes,
            "state_mib": float(gdn_state_bytes / (1024.0 * 1024.0)),
            "hssd_vs_gdn_prefill_time_ratio": float(full_sec / gdn_pre_sec),
            "hssd_vs_gdn_decode_time_ratio": float(decode_sec / gdn_dec_sec),
            "hssd_vs_gdn_cache_ratio": float(mid_chunk_bytes / max(gdn_state_bytes, 1)),
            "has_fla": bool(has_fla()),
        },
    }
    return _to_jsonable(report)


def hssd_ablation_matrix() -> dict[str, dict[str, Any]]:
    """Return the 6-way matched-budget HSSD mechanism-isolation matrix (`Gate 1-Ablate`)."""
    eval_tasks = [
        "short_context_lm",
        "repeated_key_updates",
        "distractor_resistance",
        "state_tracking",
        "mqar_recall_vs_interfering_writes",
        "code_reasoning",
    ]
    return {
        "1_original_gdn": {
            "enabled": False,
            "name": "Original GDN",
            "description": "Unmodified single-state Gated DeltaNet baseline (no HSSD adapter installed).",
            "hssd": {
                "enabled": False,
            },
            "mechanisms": {
                "local_swa": False,
                "multipage": False,
                "surprise_gate": False,
                "phase_rotation": False,
            },
            "eval_tasks": list(eval_tasks),
        },
        "2_gdn_plus_local_swa": {
            "enabled": True,
            "name": "GDN + Local SWA",
            "description": (
                "Single-page far state (M=1, k=1) with disjoint causal local sliding-window attention "
                "and Peak-Boltzmann branch arbiter; no phase rotation or surprise gating."
            ),
            "hssd": {
                "enabled": True,
                "rank": 64,
                "num_pages": 1,
                "k_pages": 1,
                "use_swa": True,
                "use_phase": False,
                "use_surprise": False,
                "surprise_mode": "page0",
                "theta_low_rank": True,
                "state_dtype": "bfloat16",
            },
            "mechanisms": {
                "local_swa": True,
                "multipage": False,
                "surprise_gate": False,
                "phase_rotation": False,
            },
            "eval_tasks": list(eval_tasks),
        },
        "3_multipage_no_phase": {
            "enabled": True,
            "name": "Multipage without Phase",
            "description": (
                "M=8 multi-timescale pages with Top-2 sparse write/read routing and disjoint local SWA, "
                "without content-adaptive SO(2) phase rotation or surprise gating."
            ),
            "hssd": {
                "enabled": True,
                "rank": 64,
                "num_pages": 8,
                "k_pages": 2,
                "use_swa": True,
                "use_phase": False,
                "use_surprise": False,
                "surprise_mode": "routed",
                "theta_low_rank": True,
                "state_dtype": "bfloat16",
            },
            "mechanisms": {
                "local_swa": True,
                "multipage": True,
                "surprise_gate": False,
                "phase_rotation": False,
            },
            "eval_tasks": list(eval_tasks),
        },
        "4_surprise_controlled": {
            "enabled": True,
            "name": "Surprise-Controlled Memory",
            "description": (
                "M=8 multi-timescale pages + local SWA + routed chunk-start root-surprise gate and "
                "actual sparse Top-2 STE write-mass budget (without phase rotation)."
            ),
            "hssd": {
                "enabled": True,
                "rank": 64,
                "num_pages": 8,
                "k_pages": 2,
                "use_swa": True,
                "use_phase": False,
                "use_surprise": True,
                "surprise_mode": "routed",
                "theta_low_rank": True,
                "state_dtype": "bfloat16",
            },
            "mechanisms": {
                "local_swa": True,
                "multipage": True,
                "surprise_gate": True,
                "phase_rotation": False,
            },
            "eval_tasks": list(eval_tasks),
        },
        "5_phase_enhanced": {
            "enabled": True,
            "name": "Phase-Enhanced Recurrence",
            "description": (
                "M=8 multi-timescale pages + local SWA + low-rank float64-accumulated SO(2)^(d_k/2) "
                "phase rotation (without surprise write gating)."
            ),
            "hssd": {
                "enabled": True,
                "rank": 64,
                "num_pages": 8,
                "k_pages": 2,
                "use_swa": True,
                "use_phase": True,
                "use_surprise": False,
                "surprise_mode": "routed",
                "theta_low_rank": True,
                "state_dtype": "bfloat16",
            },
            "mechanisms": {
                "local_swa": True,
                "multipage": True,
                "surprise_gate": False,
                "phase_rotation": True,
            },
            "eval_tasks": list(eval_tasks),
        },
        "6_complete_hssd": {
            "enabled": True,
            "name": "Complete HSSD v3.1",
            "description": (
                "Full HSSD v3.1 combining disjoint local SWA + Peak-Boltzmann arbiter + M=8 sparse Top-2 "
                "page bank + routed chunk-start surprise gate + float64-accumulated SO(2) phase rotation."
            ),
            "hssd": {
                "enabled": True,
                "rank": 64,
                "num_pages": 8,
                "k_pages": 2,
                "use_swa": True,
                "use_phase": True,
                "use_surprise": True,
                "surprise_mode": "routed",
                "theta_low_rank": True,
                "state_dtype": "bfloat16",
            },
            "mechanisms": {
                "local_swa": True,
                "multipage": True,
                "surprise_gate": True,
                "phase_rotation": True,
            },
            "eval_tasks": list(eval_tasks),
        },
    }


def device_ids(tokenizer: Any, texts: list[str], seq_len: int, device: torch.device) -> list[torch.Tensor]:
    out = []
    for t in texts:
        ids = tokenizer.encode(t, add_special_tokens=False)[:seq_len]
        if len(ids) < 8:
            continue
        out.append(torch.tensor([ids], device=device))
    require(bool(out), "no probe texts long enough", n_texts=len(texts))
    return out


__all__ = [
    "candidate_specs",
    "device_ids",
    "format_probe_report_markdown",
    "hssd_ablation_matrix",
    "probe_backward",
    "probe_core_ranges",
    "probe_expert_headroom",
    "probe_positional_extrapolation",
    "probe_reentry_anchors",
    "profile_hssd_layer",
    "run_structured_probe_suite",
    "surgery_check",
]
