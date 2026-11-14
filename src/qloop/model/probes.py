"""Cheap empirical probes that de-risk the expensive stages (run them first on Colab).

* `probe_backward`   - do gradients flow through the frozen (possibly NVFP4/FP8) base to the
                       input? Names the first layer where the gradient vanishes.
* `probe_core_ranges`- ETD-style choice of the looped core for a target virtual depth (64):
                       per-layer angular distance of the residual stream + zero-shot
                       "loopability" (loss increase when the core is simply run twice).
* `surgery_check`    - on the REAL model: loop/LSA wrapping is output-identical at init and
                       cached decode == full forward (catches HF cache/API drift).
"""
from __future__ import annotations

import copy
import logging
import math
import time
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from ..debug import cuda_mem_brief, describe, require
from .introspect import FULL, inspect_model
from .loop import LoopSpec, apply_loop

log = logging.getLogger("qloop")


def _device_of(model: nn.Module) -> torch.device:
    return next(model.parameters()).device


def _lm_loss(model: nn.Module, ids: torch.Tensor) -> torch.Tensor:
    logits = model(input_ids=ids).logits
    return F.cross_entropy(logits[:, :-1].float().reshape(-1, logits.shape[-1]), ids[:, 1:].reshape(-1))


# ---------------------------------------------------------------------------------------- backward

def probe_backward(model: nn.Module, ids: torch.Tensor) -> dict[str, Any]:
    parts = inspect_model(model)
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
    emb = parts.text_model.embed_tokens(ids).detach().requires_grad_(True)
    t0 = time.time()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    try:
        out = model(inputs_embeds=emb) if "inputs_embeds" in model.forward.__code__.co_varnames else None
        if out is None:
            hidden = parts.text_model(inputs_embeds=emb).last_hidden_state
            logits = parts.lm_head(hidden)
        else:
            logits = out.logits
        loss = F.cross_entropy(logits[:, :-1].float().reshape(-1, logits.shape[-1]), ids[:, 1:].reshape(-1))
        loss.backward()
    finally:
        for h in handles:
            h.remove()
    g_in = emb.grad
    ok = g_in is not None and bool(torch.isfinite(g_in).all()) and float(g_in.norm()) > 0
    dead = [i for i in range(parts.num_layers) if grads.get(i, 0.0) == 0.0]
    rep = {"ok": ok, "loss": float(loss.detach()), "input_grad_norm": float(g_in.norm()) if g_in is not None else None,
           "layer_grad_norms": grads, "dead_layers": dead, "seconds": time.time() - t0,
           "mem": cuda_mem_brief()}
    if not ok:
        log.error("backward probe FAILED: input grad %s; layers with zero/None grad: %s", describe(g_in), dead)
    else:
        log.info("backward probe ok: loss %.3f, |dL/dx| %.3g, %.1fs %s", rep["loss"], rep["input_grad_norm"],
                 rep["seconds"], rep["mem"])
    return rep


# ---------------------------------------------------------------------------------------- core ranges

def candidate_specs(layer_types: list[str], target_virtual: int, passes_options=(2, 3),
                    align: int | None = None) -> list[LoopSpec]:
    n = len(layer_types)
    if align is None:
        fa = [i for i, t in enumerate(layer_types) if t == FULL]
        align = (fa[1] - fa[0]) if len(fa) > 1 else 1
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
def probe_core_ranges(model: nn.Module, batches: list[torch.Tensor], target_virtual: int = 64,
                      passes_options=(2, 3), zero_shot: bool = True) -> list[dict[str, Any]]:
    parts = inspect_model(model)
    n = parts.num_layers
    ang = torch.zeros(n, dtype=torch.float64)
    base_losses = []
    for ids in batches:
        out = parts.text_model(input_ids=ids, output_hidden_states=True)
        hs = out.hidden_states
        require(hs is not None and len(hs) >= n + 1, "need output_hidden_states", n=n)
        for i in range(n):
            a, b = hs[i].float(), hs[i + 1].float()
            cos = F.cosine_similarity(a, b, dim=-1).clamp(-1, 1)
            ang[i] += (torch.arccos(cos) / math.pi).mean().double().cpu()
        base_losses.append(float(_lm_loss(model, ids)))
    ang /= len(batches)
    base = sum(base_losses) / len(base_losses)
    log.info("per-layer angular distance (ETD): %s", [round(float(x), 3) for x in ang])
    rows = []
    for spec in candidate_specs(parts.layer_types, target_virtual, passes_options):
        core_ang = float(ang[spec.prelude_end:spec.core_end].mean())
        row = {"prelude_end": spec.prelude_end, "core_end": spec.core_end, "num_passes": spec.num_passes,
               "core_len": spec.core_len, "core_angular": core_ang, "base_loss": base}
        if zero_shot:
            s = copy.copy(spec)
            s.gate_init, s.bridge_init = 1.0, (0.98, 0.01, 0.01)
            ctl = apply_loop(model, s)
            try:
                ls = [float(_lm_loss(model, ids)) for ids in batches]
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
        logits.append(out.logits[:, -1])
    return seq, torch.stack(logits, 1)


@torch.no_grad()
def surgery_check(
    model: nn.Module,
    ids: torch.Tensor,
    loop_spec: dict[str, Any] | None,
    lsa_cfg: dict[str, Any] | None,
    decode_steps: int = 8,
    atol: float = 2e-2,
    gdn2_cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run on the real model in eval mode. Raises with a precise message on any violation."""
    from .gdn2 import install_gdn2
    from .lsa import install_lsa

    model.eval()
    ref = model(input_ids=ids).logits
    rep: dict[str, Any] = {}
    gdn2 = install_gdn2(model, gdn2_cfg) if gdn2_cfg else None
    mgr = install_lsa(model, dict(lsa_cfg, mode="distill")) if lsa_cfg else None
    ctl = apply_loop(model, loop_spec) if loop_spec else None
    try:
        out = model(input_ids=ids).logits
        rep["init_identity_maxdiff"] = float((out.float() - ref.float()).abs().max())
        require(torch.equal(out, ref), "wrapped model is not bit-identical at init",
                maxdiff=rep["init_identity_maxdiff"])
        if ctl is not None:
            with torch.no_grad():
                ctl.adapters.gates.fill_(0.05)
        if gdn2 is not None:
            gen = torch.Generator(device="cpu").manual_seed(17)
            for ad in gdn2.adapters.values():
                for w in (ad.erase_up.weight, ad.write_up.weight, ad.decay_up.weight):
                    noise = torch.randn(w.shape, generator=gen, dtype=torch.float32) * 0.02
                    w.copy_(noise.to(device=w.device, dtype=w.dtype))
        seq, inc_logits = _greedy(model, ids, decode_steps)
        full = model(input_ids=seq).logits[:, ids.shape[1] - 1:]
        diff = float((full.float() - inc_logits.float()).abs().max())
        rep["decode_vs_full_maxdiff"] = diff
        require(diff < atol, "cached decode diverges from full forward (cache semantics broken)", diff=diff)
        if mgr is not None:
            mgr.set_mode("sparse")
            mgr.clear_cache()
            sp = model(input_ids=ids).logits
            rep["lsa_sparse_vs_dense_maxdiff"] = float((sp.float() - out.float()).abs().max())
    finally:
        if ctl is not None:
            ctl.unwrap()
        if mgr is not None:
            mgr.uninstall()
        if gdn2 is not None:
            gdn2.uninstall()
    back = model(input_ids=ids).logits
    require(torch.equal(back, ref), "model not restored after unwrap/uninstall")
    log.info("surgery check passed: %s", rep)
    return rep


def device_ids(tokenizer: Any, texts: list[str], seq_len: int, device: torch.device) -> list[torch.Tensor]:
    out = []
    for t in texts:
        ids = tokenizer.encode(t, add_special_tokens=False)[:seq_len]
        if len(ids) < 8:
            continue
        out.append(torch.tensor([ids], device=device))
    require(bool(out), "no probe texts long enough", n_texts=len(texts))
    return out


__all__ = ["candidate_specs", "device_ids", "probe_backward", "probe_core_ranges", "surgery_check"]
