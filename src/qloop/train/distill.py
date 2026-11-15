"""Memory-safe Teacher -> Student KL Distillation (`HSSDDistiller`) for Loop-HSSD v3.2.

Distills `Track A` (stock Qwen3.6 GDN or `GDN-2`, e.g. `s1_loop_a6b`) into `Track B`
(`HSSD v3.2`, with or without looped-depth `apply_loop`) without ever materializing full
`[B, T, 248320]` fp32 logit tensors or duplicating the 35 GB frozen MoE backbone:

  * **Zero-Duplicate-VRAM Shared-Backbone Teacher (`SharedBackboneTeacher`):** Loads only the
    small non-HSSD trainable delta tensors (`~180-250 MB`: `loop_adapters`, `norms`, and
    optional `gdn2_adapter`) onto the existing student backbone (`0.00 GiB` duplicate frozen
    expert HBM), swapping `param.data` handles and linear-attention execution mode inside
    `with shared_teacher.activate():` and restoring all handles unconditionally on exit.
  * **Phase 1 — Online Per-Layer Teacher-Forced Warm-Up (`run_layerwise_warmup_microstep`):**
    Runs ONE `torch.no_grad()` teacher-forced forward pass per micro-batch; at each of the
    `N_calls` linear-attention executions (`66` in 88L mode), runs only that layer's student
    `HSSDAdapter + chunk_hssd` under `torch.enable_grad()` on detached teacher inputs and calls
    `.backward()` immediately (`~0.5 GiB` peak extra VRAM, zero MoE backward, zero checkpoint
    recomputation).
  * **Phase 2 — 2D Token x Vocab Tiled KL (`_ChunkedVocabKL`):** Evaluates
    `KL(p_teacher || p_student)` at temperature `T` directly from pre-head hidden states
    `(h_t, h_s)` and `lm_head` weights `(W_t, W_s)` in vocab tiles of `chunk_size = 4096`
    (`<= 64 MiB` peak instead of `65+ GiB`), combined with scheduled `step_blend`, optional
    layer-wise hidden MSE, and write-mass budget regularization.

Clean-room Apache-2.0 / MIT PyTorch implementation.
"""
from __future__ import annotations

import contextlib
import copy
import functools
import json
import logging
import os
import sys
import types
from collections.abc import Iterator
from dataclasses import dataclass, fields
from typing import Any, Callable

import torch
import torch.nn.functional as F
from torch import nn

from ..debug import require
from ..model.hssd import (
    _HSSD_CHUNK_RULE_ATTRS,
    _HSSD_REC_RULE_ATTRS,
    HSSDAdapter,
    HSSDHandle,
    _resolve_physical_layers,
)
from ..model.introspect import LINEAR, attention_module, find_lm_head, find_text_model
from .checkpoint import stable_names
from .loss import lm_loss
from .optim import defer_micro_step

log = logging.getLogger("qloop")


def _acc_dtype(t: torch.Tensor) -> torch.dtype:
    return torch.float64 if t.dtype == torch.float64 else torch.float32


# ============================================================================ config & outputs

@dataclass
class HSSDDistillConfig:
    chunk_size: int = 4096
    vocab_chunk_size: int | None = None
    token_chunk_size: int = 4096
    temperature: float = 1.0
    kl_weight: float = 1.0
    scale_kl_by_temp_sq: bool = False
    hidden_mse_weight: float = 0.0
    aux_loss_weight: float = 1.0
    ce_loss_weight: float = 0.0
    ignore_index: int = -100
    mask_kl_by_labels: bool = False
    include_loop_aux: bool = True
    teacher_ckpt: str | None = None
    teacher_track: str = "stock"
    share_backbone: bool = True
    layerwise_warmup_steps: int = 400
    layerwise_mse_weight: float = 1.0

    def __post_init__(self) -> None:
        if self.vocab_chunk_size is not None:
            self.chunk_size = int(self.vocab_chunk_size)
        else:
            self.vocab_chunk_size = int(self.chunk_size)
        self.teacher_track = str(self.teacher_track or "stock").strip().lower()

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> HSSDDistillConfig:
        clean = {k: v for k, v in dict(d or {}).items() if k != "enabled"}
        known = {f.name for f in fields(cls)}
        bad = set(clean) - known
        if bad:
            raise ValueError(f"unknown distill.* keys {sorted(bad)}; valid: {sorted(known)}")
        return cls(**clean)

    def validate(self) -> None:
        if self.chunk_size < 1:
            raise ValueError(f"distill.chunk_size must be >= 1, got {self.chunk_size}")
        if self.token_chunk_size < 1:
            raise ValueError(f"distill.token_chunk_size must be >= 1, got {self.token_chunk_size}")
        if self.temperature <= 0.0:
            raise ValueError(f"distill.temperature must be > 0, got {self.temperature}")
        if self.layerwise_warmup_steps < 0:
            raise ValueError(f"distill.layerwise_warmup_steps must be >= 0, got {self.layerwise_warmup_steps}")
        if self.teacher_track not in ("stock", "auto", "gdn2"):
            raise ValueError(
                f"distill.teacher_track must be one of ('stock', 'auto', 'gdn2'), got {self.teacher_track!r}"
            )
        if (
            self.kl_weight < 0.0
            or self.hidden_mse_weight < 0.0
            or self.aux_loss_weight < 0.0
            or self.ce_loss_weight < 0.0
            or self.layerwise_mse_weight < 0.0
        ):
            raise ValueError("distill loss weights must be >= 0")


@dataclass
class DistillOutput:
    loss: torch.Tensor
    kl_loss: torch.Tensor
    hidden_mse_loss: torch.Tensor
    aux_loss: torch.Tensor
    ce_loss: torch.Tensor
    n_tokens: int

    def stats(self) -> dict[str, float]:
        return {
            "distill/loss": float(self.loss.detach()),
            "distill/kl_loss": float(self.kl_loss.detach()),
            "distill/hidden_mse": float(self.hidden_mse_loss.detach()),
            "distill/aux_loss": float(self.aux_loss.detach()),
            "distill/ce_loss": float(self.ce_loss.detach()),
            "distill/tokens": float(self.n_tokens),
        }


# ============================================================================ chunked-vocab KL

class _ChunkedVocabKL(torch.autograd.Function):
    """Exact `KL(softmax(z_t / T) || softmax(z_s / T))` tiled along vocab & token axes."""

    @staticmethod
    def forward(
        ctx: Any,
        h_teacher: torch.Tensor,
        w_teacher: torch.Tensor,
        b_teacher: torch.Tensor,
        h_student: torch.Tensor,
        w_student: torch.Tensor,
        b_student: torch.Tensor,
        has_bt: bool,
        has_bs: bool,
        vocab_chunk_size: int,
        token_chunk_size: int,
        temperature: float,
    ) -> torch.Tensor:
        acc = _acc_dtype(h_student)
        n_tok = h_student.shape[0]
        vocab_size = w_student.shape[0]
        inv_temp = 1.0 / float(temperature)

        lse_t = torch.empty(n_tok, dtype=acc, device=h_student.device)
        lse_s = torch.empty(n_tok, dtype=acc, device=h_student.device)
        kl_sum = torch.zeros((), dtype=acc, device=h_student.device)

        for t0 in range(0, n_tok, token_chunk_size):
            t1 = min(n_tok, t0 + token_chunk_size)
            ht_blk = h_teacher[t0:t1].detach().to(acc)
            hs_blk = h_student[t0:t1].detach().to(acc)

            lt_cur: torch.Tensor | None = None
            ls_cur: torch.Tensor | None = None
            for v0 in range(0, vocab_size, vocab_chunk_size):
                v1 = min(vocab_size, v0 + vocab_chunk_size)
                wt_v = w_teacher[v0:v1].detach().to(acc)
                ws_v = w_student[v0:v1].detach().to(acc)
                bt_v = b_teacher[v0:v1].detach().to(acc) if has_bt else None
                bs_v = b_student[v0:v1].detach().to(acc) if has_bs else None

                zt = F.linear(ht_blk, wt_v, bt_v) * inv_temp
                zs = F.linear(hs_blk, ws_v, bs_v) * inv_temp
                lt_v = torch.logsumexp(zt, dim=-1)
                ls_v = torch.logsumexp(zs, dim=-1)
                lt_cur = lt_v if lt_cur is None else torch.logaddexp(lt_cur, lt_v)
                ls_cur = ls_v if ls_cur is None else torch.logaddexp(ls_cur, ls_v)

            assert lt_cur is not None and ls_cur is not None
            lse_t[t0:t1] = lt_cur
            lse_s[t0:t1] = ls_cur

            for v0 in range(0, vocab_size, vocab_chunk_size):
                v1 = min(vocab_size, v0 + vocab_chunk_size)
                wt_v = w_teacher[v0:v1].detach().to(acc)
                ws_v = w_student[v0:v1].detach().to(acc)
                bt_v = b_teacher[v0:v1].detach().to(acc) if has_bt else None
                bs_v = b_student[v0:v1].detach().to(acc) if has_bs else None

                zt = F.linear(ht_blk, wt_v, bt_v) * inv_temp
                zs = F.linear(hs_blk, ws_v, bs_v) * inv_temp
                log_pt = zt - lt_cur[:, None]
                log_ps = zs - ls_cur[:, None]
                pt = torch.exp(log_pt)
                kl_sum = kl_sum + (pt * (log_pt - log_ps)).sum()

        ctx.save_for_backward(h_teacher, w_teacher, b_teacher, h_student, w_student, b_student, lse_t, lse_s)
        ctx.has_bt = has_bt
        ctx.has_bs = has_bs
        ctx.vocab_chunk_size = vocab_chunk_size
        ctx.token_chunk_size = token_chunk_size
        ctx.temperature = temperature
        return kl_sum / float(n_tok)

    @staticmethod
    def backward(ctx: Any, grad_out: torch.Tensor) -> Any:
        h_teacher, w_teacher, b_teacher, h_student, w_student, b_student, lse_t, lse_s = ctx.saved_tensors
        acc = lse_s.dtype
        n_tok = h_student.shape[0]
        vocab_size = w_student.shape[0]
        v_chunk = ctx.vocab_chunk_size
        t_chunk = ctx.token_chunk_size
        inv_temp = 1.0 / float(ctx.temperature)
        g_scale = grad_out.to(acc) * (inv_temp / float(n_tok))

        need_hs = bool(ctx.needs_input_grad[3])
        need_ws = bool(ctx.needs_input_grad[4])
        need_bs = bool(ctx.has_bs and ctx.needs_input_grad[5])

        grad_hs = torch.zeros(h_student.shape, dtype=acc, device=h_student.device) if need_hs else None
        grad_ws = torch.zeros(w_student.shape, dtype=acc, device=w_student.device) if need_ws else None
        grad_bs = torch.zeros(b_student.shape, dtype=acc, device=b_student.device) if need_bs else None

        for t0 in range(0, n_tok, t_chunk):
            t1 = min(n_tok, t0 + t_chunk)
            ht_blk = h_teacher[t0:t1].to(acc)
            hs_blk = h_student[t0:t1].to(acc)
            lt_blk = lse_t[t0:t1, None]
            ls_blk = lse_s[t0:t1, None]

            for v0 in range(0, vocab_size, v_chunk):
                v1 = min(vocab_size, v0 + v_chunk)
                wt_v = w_teacher[v0:v1].to(acc)
                ws_v = w_student[v0:v1].to(acc)
                bt_v = b_teacher[v0:v1].to(acc) if ctx.has_bt else None
                bs_v = b_student[v0:v1].to(acc) if ctx.has_bs else None

                zt = F.linear(ht_blk, wt_v, bt_v) * inv_temp
                zs = F.linear(hs_blk, ws_v, bs_v) * inv_temp
                pt = torch.exp(zt - lt_blk)
                ps = torch.exp(zs - ls_blk)
                dz = g_scale * (ps - pt)

                if grad_hs is not None:
                    grad_hs[t0:t1].add_(dz @ ws_v)
                if grad_ws is not None:
                    grad_ws[v0:v1].add_(dz.transpose(0, 1) @ hs_blk)
                if grad_bs is not None:
                    grad_bs[v0:v1].add_(dz.sum(dim=0))

        gh_out = grad_hs.to(h_student.dtype) if grad_hs is not None else None
        gw_out = grad_ws.to(w_student.dtype) if grad_ws is not None else None
        gb_out = grad_bs.to(b_student.dtype) if grad_bs is not None else None
        return None, None, None, gh_out, gw_out, gb_out, None, None, None, None, None


class _ChunkedLogitsKL(torch.autograd.Function):
    """Chunked-vocab `KL(p_teacher || p_student)` on precomputed logit tensors."""

    @staticmethod
    def forward(
        ctx: Any,
        logits_teacher: torch.Tensor,
        logits_student: torch.Tensor,
        vocab_chunk_size: int,
        temperature: float,
    ) -> torch.Tensor:
        acc = _acc_dtype(logits_student)
        n_tok, vocab_size = logits_student.shape
        inv_temp = 1.0 / float(temperature)

        lse_t: torch.Tensor | None = None
        lse_s: torch.Tensor | None = None
        for v0 in range(0, vocab_size, vocab_chunk_size):
            v1 = min(vocab_size, v0 + vocab_chunk_size)
            zt = logits_teacher[:, v0:v1].detach().to(acc) * inv_temp
            zs = logits_student[:, v0:v1].detach().to(acc) * inv_temp
            lt_v = torch.logsumexp(zt, dim=-1)
            ls_v = torch.logsumexp(zs, dim=-1)
            lse_t = lt_v if lse_t is None else torch.logaddexp(lse_t, lt_v)
            lse_s = ls_v if lse_s is None else torch.logaddexp(lse_s, ls_v)

        assert lse_t is not None and lse_s is not None
        kl_sum = torch.zeros((), dtype=acc, device=logits_student.device)
        for v0 in range(0, vocab_size, vocab_chunk_size):
            v1 = min(vocab_size, v0 + vocab_chunk_size)
            zt = logits_teacher[:, v0:v1].detach().to(acc) * inv_temp
            zs = logits_student[:, v0:v1].detach().to(acc) * inv_temp
            log_pt = zt - lse_t[:, None]
            log_ps = zs - lse_s[:, None]
            pt = torch.exp(log_pt)
            kl_sum = kl_sum + (pt * (log_pt - log_ps)).sum()

        ctx.save_for_backward(logits_teacher, logits_student, lse_t, lse_s)
        ctx.vocab_chunk_size = vocab_chunk_size
        ctx.temperature = temperature
        return kl_sum / float(n_tok)

    @staticmethod
    def backward(ctx: Any, grad_out: torch.Tensor) -> Any:
        logits_teacher, logits_student, lse_t, lse_s = ctx.saved_tensors
        if not ctx.needs_input_grad[1]:
            return None, None, None, None
        acc = lse_s.dtype
        n_tok, vocab_size = logits_student.shape
        v_chunk = ctx.vocab_chunk_size
        inv_temp = 1.0 / float(ctx.temperature)
        g_scale = grad_out.to(acc) * (inv_temp / float(n_tok))
        grad_s = torch.empty_like(logits_student)

        for v0 in range(0, vocab_size, v_chunk):
            v1 = min(vocab_size, v0 + v_chunk)
            zt = logits_teacher[:, v0:v1].to(acc) * inv_temp
            zs = logits_student[:, v0:v1].to(acc) * inv_temp
            pt = torch.exp(zt - lse_t[:, None])
            ps = torch.exp(zs - lse_s[:, None])
            grad_s[:, v0:v1] = (g_scale * (ps - pt)).to(logits_student.dtype)
        return None, grad_s, None, None


def chunked_vocab_kl_loss(
    h_teacher: torch.Tensor,
    w_teacher: torch.Tensor,
    h_student: torch.Tensor,
    w_student: torch.Tensor,
    *,
    b_teacher: torch.Tensor | None = None,
    b_student: torch.Tensor | None = None,
    loss_mask: torch.Tensor | None = None,
    chunk_size: int = 4096,
    token_chunk_size: int = 4096,
    temperature: float = 1.0,
) -> tuple[torch.Tensor, int]:
    """Compute `(kl_loss, n_valid_tokens)` without materializing `[B, T, V]` logits."""
    require(h_teacher.shape[:-1] == h_student.shape[:-1], "teacher/student token shapes must match",
            h_teacher=h_teacher, h_student=h_student)
    require(w_teacher.shape[0] == w_student.shape[0], "teacher/student vocab sizes must match",
            w_teacher=w_teacher, w_student=w_student)
    require(chunk_size > 0 and token_chunk_size > 0 and temperature > 0.0,
            "chunk_size, token_chunk_size, and temperature must be > 0")

    ht_flat = h_teacher.detach().reshape(-1, h_teacher.shape[-1])
    hs_flat = h_student.reshape(-1, h_student.shape[-1])
    if loss_mask is not None:
        valid = loss_mask.reshape(-1).to(dtype=torch.bool)
        idx = valid.nonzero(as_tuple=True)[0]
        ht_flat = ht_flat.index_select(0, idx)
        hs_flat = hs_flat.index_select(0, idx)

    n_valid = int(hs_flat.shape[0])
    if n_valid == 0:
        acc = _acc_dtype(h_student)
        return (h_student.sum() * 0.0).to(acc), 0

    empty_b = hs_flat.new_empty(0)
    bt_arg = b_teacher.detach() if b_teacher is not None else empty_b
    bs_arg = b_student if b_student is not None else empty_b
    kl = _ChunkedVocabKL.apply(
        ht_flat,
        w_teacher.detach(),
        bt_arg,
        hs_flat,
        w_student,
        bs_arg,
        b_teacher is not None,
        b_student is not None,
        int(chunk_size),
        int(token_chunk_size),
        float(temperature),
    )
    return kl, n_valid


def chunked_vocab_kl_from_logits(
    logits_teacher: torch.Tensor,
    logits_student: torch.Tensor,
    *,
    loss_mask: torch.Tensor | None = None,
    chunk_size: int = 4096,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Compute `KL(p_teacher || p_student)` across vocab chunks of `chunk_size` on logit tensors."""
    require(logits_teacher.shape == logits_student.shape, "logit shapes must match",
            teacher=logits_teacher, student=logits_student)
    require(chunk_size > 0 and temperature > 0.0, "chunk_size and temperature must be > 0")
    lt_flat = logits_teacher.detach().reshape(-1, logits_teacher.shape[-1])
    ls_flat = logits_student.reshape(-1, logits_student.shape[-1])
    if loss_mask is not None:
        idx = loss_mask.reshape(-1).to(dtype=torch.bool).nonzero(as_tuple=True)[0]
        lt_flat = lt_flat.index_select(0, idx)
        ls_flat = ls_flat.index_select(0, idx)
    if ls_flat.shape[0] == 0:
        return (logits_student.sum() * 0.0).to(_acc_dtype(logits_student))
    return _ChunkedLogitsKL.apply(lt_flat, ls_flat, int(chunk_size), float(temperature))


def exact_full_vocab_kl_loss(
    logits_teacher: torch.Tensor,
    logits_student: torch.Tensor,
    *,
    loss_mask: torch.Tensor | None = None,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Reference full-vocab `KL(p_teacher || p_student)` for unit-test parity verification."""
    acc = _acc_dtype(logits_student)
    lt = logits_teacher.detach().reshape(-1, logits_teacher.shape[-1]).to(acc) / float(temperature)
    ls = logits_student.reshape(-1, logits_student.shape[-1]).to(acc) / float(temperature)
    if loss_mask is not None:
        idx = loss_mask.reshape(-1).to(dtype=torch.bool).nonzero(as_tuple=True)[0]
        lt = lt.index_select(0, idx)
        ls = ls.index_select(0, idx)
    if ls.shape[0] == 0:
        return (logits_student.sum() * 0.0).to(acc)
    log_pt = F.log_softmax(lt, dim=-1)
    log_ps = F.log_softmax(ls, dim=-1)
    pt = torch.exp(log_pt)
    return (pt * (log_pt - log_ps)).sum(dim=-1).mean()


# ============================================================================ shared-backbone teacher & warmup

_ALL_HSSD_RULE_ATTRS: tuple[tuple[str, bool], ...] = tuple(
    [(attr, False) for attr in _HSSD_CHUNK_RULE_ATTRS]
    + [(attr, True) for attr in _HSSD_REC_RULE_ATTRS]
)


def _extract_orig_rule_fn(fn: Any) -> Callable[..., Any] | None:
    """Recover the underlying stock `orig_fn` wrapped by `apply_hssd` (`_make_hssd_rule_wrapper`)."""
    cur = fn
    if isinstance(cur, types.MethodType):
        cur = cur.__func__
    kwdefs = getattr(cur, "__kwdefaults__", None)
    if isinstance(kwdefs, dict):
        if callable(kwdefs.get("_orig")):
            return kwdefs["_orig"]
        if callable(kwdefs.get("_rw")):
            return _extract_orig_rule_fn(kwdefs["_rw"])
    code = getattr(cur, "__code__", None)
    closure = getattr(cur, "__closure__", None)
    if code is not None and closure is not None:
        free = getattr(code, "co_freevars", ())
        for var_name, cell in zip(free, closure):
            if var_name in ("orig_fn", "_orig"):
                try:
                    val = cell.cell_contents
                except ValueError:
                    continue
                if callable(val):
                    return val
    return None


def _detach_tree(obj: Any) -> Any:
    if isinstance(obj, torch.Tensor):
        return obj.detach()
    if isinstance(obj, tuple):
        return tuple(_detach_tree(x) for x in obj)
    if isinstance(obj, list):
        return [_detach_tree(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _detach_tree(v) for k, v in obj.items()}
    return obj


def _collect_named_params(stack_or_model: Any) -> tuple[nn.Module, dict[str, nn.Parameter], dict[str, nn.Parameter]]:
    if hasattr(stack_or_model, "model") and isinstance(stack_or_model.model, nn.Module):
        model = stack_or_model.model
        named_all = dict(getattr(stack_or_model, "named_all", None) or {})
        named_tr = dict(getattr(stack_or_model, "named_trainable", None) or {})
        if named_all:
            if not named_tr:
                named_tr = {k: p for k, p in named_all.items() if p.requires_grad}
            return model, named_all, named_tr
    else:
        model = stack_or_model

    names = stable_names(model)
    tm = find_text_model(model)
    layers = getattr(tm, "layers", None)
    ctl = getattr(layers[0], "_ctl", None) if isinstance(layers, nn.ModuleList) and len(layers) > 0 else None
    if ctl is not None and getattr(ctl, "adapters", None) is not None:
        for n, p in ctl.adapters.named_parameters():
            names[id(p)] = f"loop_adapters.{n}"
    params = {id(p): p for p in model.parameters()}
    if ctl is not None and getattr(ctl, "adapters", None) is not None:
        params.update({id(p): p for p in ctl.adapters.parameters()})
    named_all = {names[i]: p for i, p in params.items() if i in names}
    named_tr = {k: p for k, p in named_all.items() if p.requires_grad}
    return model, named_all, named_tr


def _load_teacher_ckpt_bundle(
    teacher_ckpt: str | None,
    *,
    run_root: str | None = None,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    if not teacher_ckpt:
        return {}, {}
    from ..hub import resolve_run_path
    from .checkpoint import CheckpointManager

    candidates: list[str] = []
    resolved = resolve_run_path(teacher_ckpt, run_root=run_root)
    if resolved:
        candidates.append(resolved)
    if teacher_ckpt not in candidates:
        candidates.append(teacher_ckpt)

    target: str | None = None
    for cand in candidates:
        if cand and os.path.exists(cand):
            target = cand
            break
    if target is None:
        log.info(
            "SharedBackboneTeacher: teacher_ckpt %r not found on disk; snapshotting initial student non-HSSD deltas",
            teacher_ckpt,
        )
        return {}, {}

    if os.path.isfile(target):
        raw = torch.load(target, map_location="cpu", weights_only=True)
        if isinstance(raw, dict) and "params" in raw and isinstance(raw["params"], dict):
            return dict(raw["params"]), dict(raw.get("state") or {})
        if isinstance(raw, dict):
            return {k: v for k, v in raw.items() if isinstance(v, torch.Tensor)}, {}
        return {}, {}

    if os.path.isdir(os.path.join(target, "checkpoints")):
        mgr = CheckpointManager(target, keep_last=0)
        step_dir = mgr.promoted() or mgr.latest()
        if step_dir is not None:
            ck = mgr.load(step_dir)
            if ck is not None:
                return dict(ck.get("params") or {}), dict(ck.get("state") or {})

    for fname in ("trainable.pt", "adapter_state.pt", "qloop_adapters.pt"):
        fpath = os.path.join(target, fname)
        if os.path.isfile(fpath):
            params = torch.load(fpath, map_location="cpu", weights_only=True)
            state: dict[str, Any] = {}
            sjson = os.path.join(target, "state.json")
            if os.path.isfile(sjson):
                with open(sjson, encoding="utf-8") as f:
                    state = json.load(f)
            if isinstance(params, dict):
                return {k: v for k, v in params.items() if isinstance(v, torch.Tensor)}, state
    return {}, {}


def _expected_linear_calls(model: nn.Module, hssd: HSSDHandle, depth: int | None = None) -> int:
    text_model = find_text_model(model)
    layers = getattr(text_model, "layers", None)
    ctl = getattr(layers[0], "_ctl", None) if isinstance(layers, nn.ModuleList) and len(layers) > 0 else None
    target_ids = {int(i) for i in hssd.adapters}
    if ctl is not None and getattr(ctl, "schedule", None) is not None:
        eff_depth = int(depth if depth is not None else getattr(ctl, "active_passes", 1))
        return max(1, sum(1 for slot in ctl.schedule if slot.r <= eff_depth and int(slot.phys) in target_ids))
    return max(1, len(target_ids))


class SharedBackboneTeacher:
    """Zero-duplicate-VRAM teacher sharing the single student backbone (`0.00 GiB` extra expert HBM).

    Loads ONLY the small non-HSSD trainable delta tensors (`loop_adapters`, `norms`, and optional
    `gdn2_adapter` weights, ~180-250 MB on Qwen3.6-35B) from `teacher_ckpt` (or snapshots the
    student's initial non-HSSD trainable parameters when `teacher_ckpt` is absent). Inside
    `with teacher.activate():`, swaps `param.data` handles to the frozen teacher tensors and
    switches linear-attention layers to teacher mode (`gdn2` if present in `teacher_ckpt`, else
    stock `orig_fn`), restoring every student handle and training flag unconditionally in `finally:`.
    """

    def __init__(
        self,
        stack_or_model: Any,
        *,
        teacher_ckpt: str | None = None,
        teacher_track: str = "stock",
        run_root: str | None = None,
    ) -> None:
        self.stack_or_model = stack_or_model
        self.model, self.named_all, self.named_trainable = _collect_named_params(stack_or_model)
        self.text_model = find_text_model(self.model)
        self.hssd: HSSDHandle | None = (
            getattr(stack_or_model, "hssd", None) or getattr(self.model, "_hssd_handle", None)
        )
        self.teacher_ckpt = teacher_ckpt
        self.teacher_track = str(teacher_track or "stock").strip().lower()
        if self.teacher_track not in ("stock", "auto", "gdn2"):
            raise ValueError(
                f"SharedBackboneTeacher: teacher_track must be one of ('stock', 'auto', 'gdn2'), got {teacher_track!r}"
            )

        ckpt_params, ckpt_state = _load_teacher_ckpt_bundle(teacher_ckpt, run_root=run_root)
        self.loaded_from_ckpt = bool(ckpt_params)
        self.teacher_deltas: dict[str, torch.Tensor] = {}
        gdn2_tensors: dict[str, torch.Tensor] = {}

        for name, tensor in ckpt_params.items():
            if "hssd_adapter." in name:
                continue
            if "gdn2_adapter." in name:
                if self.teacher_track != "stock":
                    gdn2_tensors[name] = tensor
                continue
            target_p = self.named_all.get(name)
            if target_p is not None and tuple(target_p.shape) == tuple(tensor.shape):
                self.teacher_deltas[name] = tensor.detach().to(device=target_p.device, dtype=target_p.dtype).clone()

        # Snapshot any non-HSSD student trainable parameters not covered by teacher_ckpt
        # (ensures step-0 teacher parity in standalone tests or when additional norms are unfrozen).
        for name, p in self.named_trainable.items():
            if "hssd_adapter." in name or name in self.teacher_deltas:
                continue
            self.teacher_deltas[name] = p.detach().clone()

        self.gdn2_adapters: dict[int, nn.Module] = {}
        if gdn2_tensors and self.teacher_track in ("auto", "gdn2") and self.hssd is not None:
            self._init_gdn2_teacher_adapters(gdn2_tensors, ckpt_state)

        delta_mb = self.extra_vram_bytes() / (1024.0 * 1024.0)
        log.info(
            "SharedBackboneTeacher initialized: %d delta tensors (%.2f MiB, from_ckpt=%s, gdn2_layers=%d, "
            "duplicate_backbone_vram=0.00 GiB)",
            len(self.teacher_deltas),
            delta_mb,
            self.loaded_from_ckpt,
            len(self.gdn2_adapters),
        )

    def _init_gdn2_teacher_adapters(
        self,
        gdn2_tensors: dict[str, torch.Tensor],
        ckpt_state: dict[str, Any],
    ) -> None:
        from ..model.gdn2 import GDN2Adapter, GDN2Config

        assert self.hssd is not None
        gcfg_raw = ((ckpt_state.get("stack") or {}).get("gdn2")) or {}
        gcfg = GDN2Config.from_dict(gcfg_raw) if isinstance(gcfg_raw, dict) else GDN2Config()
        ref = self.text_model.embed_tokens.weight
        p_dtype = torch.float64 if ref.dtype == torch.float64 else torch.float32

        for layer_id, h_ad in self.hssd.adapters.items():
            prefix_candidates = (f"layers.{layer_id}.", f".layers.{layer_id}.")
            sub_sd: dict[str, torch.Tensor] = {}
            for full_k, val in gdn2_tensors.items():
                if "gdn2_adapter." not in full_k:
                    continue
                if any(pc in f".{full_k}" for pc in prefix_candidates):
                    short_k = full_k.split("gdn2_adapter.", 1)[1]
                    sub_sd[short_k] = val
            if not sub_sd:
                continue
            gad = GDN2Adapter(
                h_ad.hidden_size,
                h_ad.num_v_heads,
                h_ad.head_k_dim,
                h_ad.head_v_dim,
                gcfg,
                layer_id,
            ).to(device=ref.device, dtype=p_dtype)
            gad.load_state_dict(sub_sd, strict=False)
            gad.eval()
            for p in gad.parameters():
                p.requires_grad_(False)
            self.gdn2_adapters[int(layer_id)] = gad

    def extra_vram_bytes(self) -> int:
        total = sum(t.numel() * t.element_size() for t in self.teacher_deltas.values())
        for gad in self.gdn2_adapters.values():
            total += sum(p.numel() * p.element_size() for p in gad.parameters())
        return int(total)

    @contextlib.contextmanager
    def _patch_linear_rules(
        self,
        *,
        student_step_hook: Callable[[HSSDAdapter, Callable[..., Any], tuple[Any, ...], dict[str, Any], Any], None]
        | None = None,
    ) -> Iterator[None]:
        if self.hssd is None or not self.hssd.adapters:
            yield
            return

        # Fast path when no per-layer student callback and no GDN-2 teacher adapters are needed,
        # and HSSDHandle natively provides `teacher_mode()`.
        teacher_mode_cm = getattr(self.hssd, "teacher_mode", None)
        if student_step_hook is None and not self.gdn2_adapters and callable(teacher_mode_cm):
            with teacher_mode_cm():
                yield
            return

        _make_gdn2_wrapper: Any = None
        if self.gdn2_adapters:
            from ..model.gdn2 import _make_rule_wrapper as _make_gdn2_wrapper

        _, phys_layers, _ = _resolve_physical_layers(self.model)
        temp_hooks: list[Any] = []
        restorers: list[Callable[[], None]] = []
        patched_mods: set[tuple[int, str]] = set()
        active_stack: list[tuple[nn.Module, HSSDAdapter, list[torch.Tensor], list[int]]] = []

        try:
            for layer_id, adapter in self.hssd.adapters.items():
                attn = attention_module(phys_layers[layer_id], LINEAR)
                x_stack: list[torch.Tensor] = []
                gdn2_box: list[int] = [0]

                def _make_pre(
                    a_mod: nn.Module = attn,
                    ad_ref: HSSDAdapter = adapter,
                    xs_ref: list[torch.Tensor] = x_stack,
                    gb_ref: list[int] = gdn2_box,
                ):
                    def _pre(_mod: nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
                        x_in = (
                            args[0]
                            if args
                            else (kwargs.get("hidden_states") if "hidden_states" in kwargs else kwargs.get("x"))
                        )
                        if isinstance(x_in, torch.Tensor):
                            xs_ref.append(x_in.detach())
                        active_stack.append((a_mod, ad_ref, xs_ref, gb_ref))

                    return _pre

                def _make_post(xs_ref: list[torch.Tensor] = x_stack):
                    def _post(_mod: nn.Module, _inp: Any, _out: Any) -> None:
                        if xs_ref:
                            xs_ref.pop()
                        if active_stack:
                            active_stack.pop()

                    return _post

                temp_hooks.append(attn.register_forward_pre_hook(_make_pre(), with_kwargs=True))
                temp_hooks.append(attn.register_forward_hook(_make_post()))

                for attr, rec_only in _ALL_HSSD_RULE_ATTRS:
                    if attr in attn.__dict__:
                        student_fn = attn.__dict__[attr]
                        orig_fn = _extract_orig_rule_fn(student_fn)
                        require(
                            callable(orig_fn),
                            f"SharedBackboneTeacher: could not extract orig_fn on layer {layer_id}.{attr}",
                        )
                        assert orig_fn is not None
                        gad = self.gdn2_adapters.get(int(layer_id))
                        teacher_fn: Callable[..., Any]
                        if gad is not None:
                            teacher_fn = _make_gdn2_wrapper(
                                attn, gad, x_stack, orig_fn, recurrent_only=rec_only, in_rule_box=gdn2_box
                            )
                        else:
                            teacher_fn = orig_fn

                        def _make_inst_interceptor(
                            ad_ref: HSSDAdapter = adapter,
                            t_fn: Callable[..., Any] = teacher_fn,
                            s_fn: Callable[..., Any] = student_fn,
                        ) -> Callable[..., Any]:
                            def _intercept(*r_args: Any, **r_kwargs: Any) -> Any:
                                with torch.no_grad():
                                    t_res = t_fn(*r_args, **r_kwargs)
                                if student_step_hook is not None:
                                    student_step_hook(ad_ref, s_fn, r_args, r_kwargs, t_res)
                                return t_res

                            return _intercept

                        intercept_fn = _make_inst_interceptor()
                        if isinstance(student_fn, types.MethodType) and getattr(student_fn, "__self__", None) is attn:
                            raw_fn = getattr(student_fn, "__func__", student_fn)

                            @functools.wraps(raw_fn)
                            def _bound_intercept(
                                _self: nn.Module,
                                *b_args: Any,
                                _ifn: Callable[..., Any] = intercept_fn,
                                **b_kwargs: Any,
                            ) -> Any:
                                return _ifn(*b_args, **b_kwargs)

                            setattr(attn, attr, types.MethodType(_bound_intercept, attn))
                        else:
                            setattr(attn, attr, intercept_fn)
                        restorers.append(lambda a=attn, k=attr, old=student_fn: setattr(a, k, old))
                    else:
                        host_mod = sys.modules.get(type(attn).__module__)
                        if host_mod is None:
                            continue
                        mod_key = (id(host_mod), attr)
                        if mod_key in patched_mods:
                            continue
                        mod_fn = getattr(host_mod, attr, None)
                        orig_mfn = _extract_orig_rule_fn(mod_fn) if callable(mod_fn) else None
                        if orig_mfn is None:
                            continue
                        patched_mods.add(mod_key)
                        assert callable(mod_fn)

                        def _make_mod_interceptor(
                            s_mfn: Callable[..., Any] = mod_fn,
                            o_mfn: Callable[..., Any] = orig_mfn,
                            is_rec: bool = rec_only,
                        ) -> Callable[..., Any]:
                            def _mod_intercept(*r_args: Any, **r_kwargs: Any) -> Any:
                                if not active_stack:
                                    return o_mfn(*r_args, **r_kwargs)
                                cur_attn, cur_ad, cur_xs, cur_gb = active_stack[-1]
                                gad = self.gdn2_adapters.get(int(cur_ad.layer_id))
                                if gad is not None:
                                    t_fn = _make_gdn2_wrapper(
                                        cur_attn, gad, cur_xs, o_mfn, recurrent_only=is_rec, in_rule_box=cur_gb
                                    )
                                else:
                                    t_fn = o_mfn
                                with torch.no_grad():
                                    t_res = t_fn(*r_args, **r_kwargs)
                                if student_step_hook is not None:
                                    student_step_hook(cur_ad, s_mfn, r_args, r_kwargs, t_res)
                                return t_res

                            return _mod_intercept

                        setattr(host_mod, attr, _make_mod_interceptor())
                        restorers.append(lambda hm=host_mod, k=attr, old=mod_fn: setattr(hm, k, old))
            yield
        finally:
            for restore in reversed(restorers):
                restore()
            for h in reversed(temp_hooks):
                h.remove()

    @contextlib.contextmanager
    def activate(
        self,
        *,
        student_step_hook: Callable[[HSSDAdapter, Callable[..., Any], tuple[Any, ...], dict[str, Any], Any], None]
        | None = None,
    ) -> Iterator[SharedBackboneTeacher]:
        """Temporarily swap non-HSSD delta `.data` tensors and linear-attention rule to teacher mode."""
        was_model_training = bool(self.model.training)
        layers = getattr(self.text_model, "layers", None)
        ctl = getattr(layers[0], "_ctl", None) if isinstance(layers, nn.ModuleList) and len(layers) > 0 else None
        loop_adapters = getattr(ctl, "adapters", None)
        was_loop_training = bool(loop_adapters.training) if isinstance(loop_adapters, nn.Module) else False

        saved_data: dict[str, torch.Tensor] = {}
        self.model.eval()
        if isinstance(loop_adapters, nn.Module):
            loop_adapters.eval()
        try:
            with torch.no_grad():
                for name, t_tensor in self.teacher_deltas.items():
                    p = self.named_all.get(name)
                    if p is None:
                        continue
                    saved_data[name] = p.data
                    p.data = t_tensor
            with self._patch_linear_rules(student_step_hook=student_step_hook):
                yield self
        finally:
            with torch.no_grad():
                for name, orig_d in saved_data.items():
                    self.named_all[name].data = orig_d
            if was_model_training:
                self.model.train()
            if isinstance(loop_adapters, nn.Module) and was_loop_training:
                loop_adapters.train()


def run_layerwise_warmup_microstep(
    stack_or_model: Any,
    shared_teacher: SharedBackboneTeacher,
    opt: Any,
    input_ids: torch.Tensor,
    *,
    position_ids: torch.Tensor | None = None,
    attention_mask: torch.Tensor | None = None,
    depth: int | None = None,
    grad_accum: int = 1,
    mse_weight: float = 1.0,
    budget_weight: float = 1.0,
) -> dict[str, float]:
    """Phase 1 (`step < layerwise_warmup_steps`): Online Per-Layer Teacher-Forced HSSD Warm-Up.

    Runs ONE `torch.no_grad()` teacher forward pass through the shared backbone. At each of the
    `N_calls` linear-attention executions (`66` in 88L mode):
      1. Teacher computes `o_teacher` (`no_grad`) and feeds `o_teacher` forward so downstream
         layers receive exact teacher-forced inputs.
      2. Under `torch.enable_grad()`, runs ONLY that layer's student `HSSDAdapter + chunk_hssd`
         on detached teacher inputs, computes `(mse_weight * MSE + budget_weight * budget) / (N_calls * grad_accum)`,
         and calls `.backward()` immediately so that single layer's ~0.5 GB activation graph is
         freed before the next layer executes (zero MoE backward, zero checkpoint recomputation).
    """
    hssd = shared_teacher.hssd
    require(hssd is not None and bool(hssd.adapters), "run_layerwise_warmup_microstep requires active HSSD adapters")
    assert hssd is not None

    n_expected = _expected_linear_calls(shared_teacher.model, hssd, depth=depth)
    inv_denom = 1.0 / float(max(1, n_expected) * max(1, int(grad_accum)))

    mse_sum_dev: list[torch.Tensor] = []
    bud_sum_dev: list[torch.Tensor] = []
    loss_sum_dev: list[torch.Tensor] = []

    set_blend = getattr(hssd, "set_blend", None)
    prev_blend: float | None = None
    if callable(set_blend):
        raw_b = getattr(hssd, "blend", getattr(hssd, "_blend", None))
        if raw_b is not None:
            prev_blend = float(raw_b)

    orig_ckpt_seg = bool(getattr(hssd.cfg, "checkpoint_segments", True))

    def _per_call_student_step(
        adapter: HSSDAdapter,
        student_rule_fn: Callable[..., Any],
        r_args: tuple[Any, ...],
        r_kwargs: dict[str, Any],
        teacher_res: Any,
    ) -> None:
        o_teacher = teacher_res[0] if isinstance(teacher_res, tuple) else teacher_res
        det_args = _detach_tree(r_args)
        det_kwargs = _detach_tree(r_kwargs)
        if "output_final_state" in det_kwargs:
            det_kwargs["output_final_state"] = False
        if "initial_state" in det_kwargs:
            det_kwargs["initial_state"] = None

        adapter.clear_aux()
        was_ad_training = bool(adapter.training)
        if not was_ad_training:
            adapter.train()
        try:
            with torch.enable_grad():
                s_res = student_rule_fn(*det_args, **det_kwargs)
                o_student = s_res[0] if isinstance(s_res, tuple) else s_res
                pending_aux = getattr(adapter, "_pending_aux", None)
                if pending_aux:
                    call_budget = torch.stack(list(pending_aux)).mean().float()
                    adapter.clear_aux()
                elif callable(getattr(adapter, "pop_aux_loss", None)):
                    call_budget = adapter.pop_aux_loss().float()
                else:
                    call_budget = o_student.new_zeros((), dtype=torch.float32)
                call_mse = F.mse_loss(o_student.float(), o_teacher.detach().float())
                raw_loss = float(mse_weight) * call_mse + float(budget_weight) * call_budget
                (raw_loss * inv_denom).backward()
        finally:
            adapter.clear_aux()
            if not was_ad_training:
                adapter.eval()

        mse_sum_dev.append(call_mse.detach())
        bud_sum_dev.append(call_budget.detach())
        loss_sum_dev.append(raw_loss.detach())

    loop_ctl = getattr(stack_or_model, "loop", None)
    depth_cm = (
        loop_ctl.depth(int(depth))
        if (depth is not None and loop_ctl is not None and callable(getattr(loop_ctl, "depth", None)))
        else contextlib.nullcontext()
    )

    with defer_micro_step(opt):
        if callable(set_blend):
            set_blend(1.0)
        hssd.cfg.checkpoint_segments = False
        try:
            with depth_cm, shared_teacher.activate(student_step_hook=_per_call_student_step), torch.no_grad():
                _run_text_backbone(
                    shared_teacher.text_model,
                    input_ids,
                    attention_mask,
                    output_hidden_states=False,
                    position_ids=position_ids,
                )
        finally:
            hssd.cfg.checkpoint_segments = orig_ckpt_seg
            if callable(set_blend) and prev_blend is not None:
                set_blend(prev_blend)
            for ad in hssd.adapters.values():
                ad.clear_aux()

    n_calls = len(loss_sum_dev)
    if n_calls == 0:
        return {
            "distill/phase": 1.0,
            "distill/layerwise_mse": 0.0,
            "distill/hssd_budget": 0.0,
            "distill/loss": 0.0,
            "distill/calls": 0.0,
        }
    packed = torch.stack(
        [
            torch.stack(mse_sum_dev).mean(),
            torch.stack(bud_sum_dev).mean(),
            torch.stack(loss_sum_dev).mean(),
        ]
    ).to(device="cpu", dtype=torch.float64)
    mean_mse, mean_bud, mean_loss = (float(x) for x in packed.numpy())
    return {
        "distill/phase": 1.0,
        "distill/layerwise_mse": mean_mse,
        "distill/hssd_budget": mean_bud,
        "distill/loss": mean_loss,
        "distill/calls": float(n_calls),
    }


def compute_shared_distill_loss(
    stack_or_model: Any,
    shared_teacher: SharedBackboneTeacher,
    cfg: HSSDDistillConfig,
    input_ids: torch.Tensor,
    *,
    position_ids: torch.Tensor | None = None,
    attention_mask: torch.Tensor | None = None,
    labels: torch.Tensor | None = None,
    loss_mask: torch.Tensor | None = None,
    depth: int | None = None,
    step: int | None = None,
) -> DistillOutput:
    """Phase 2 (`step >= layerwise_warmup_steps`): End-to-End Shared-Backbone KL Distillation."""
    hssd = shared_teacher.hssd
    if step is not None and hssd is not None and callable(getattr(hssd, "step_blend", None)):
        blend_step = max(0, int(step) - int(cfg.layerwise_warmup_steps))
        hssd.step_blend(blend_step)

    want_hidden = cfg.hidden_mse_weight > 0.0
    loop_ctl = getattr(stack_or_model, "loop", None)
    depth_cm = (
        loop_ctl.depth(int(depth))
        if (depth is not None and loop_ctl is not None and callable(getattr(loop_ctl, "depth", None)))
        else contextlib.nullcontext()
    )

    with depth_cm:
        with shared_teacher.activate(), torch.no_grad():
            h_teacher, t_all = _run_text_backbone(
                shared_teacher.text_model,
                input_ids,
                attention_mask,
                want_hidden,
                position_ids=position_ids,
            )
            h_teacher = h_teacher.detach()
            if t_all is not None:
                t_all = tuple(t.detach() for t in t_all)
            w_t_raw, b_t_raw = _extract_lm_head_params(shared_teacher.model)
            w_t = w_t_raw.detach()
            b_t = b_t_raw.detach() if b_t_raw is not None else None

        h_student, s_all = _run_text_backbone(
            shared_teacher.text_model,
            input_ids,
            attention_mask,
            want_hidden,
            position_ids=position_ids,
        )
        w_s, b_s = _extract_lm_head_params(shared_teacher.model)

    eff_mask = loss_mask
    if eff_mask is None and attention_mask is not None and attention_mask.shape == input_ids.shape:
        eff_mask = attention_mask != 0
    if eff_mask is None and labels is not None and cfg.mask_kl_by_labels:
        eff_mask = labels != cfg.ignore_index

    kl_loss, n_valid = chunked_vocab_kl_loss(
        h_teacher,
        w_t,
        h_student,
        w_s,
        b_teacher=b_t,
        b_student=b_s,
        loss_mask=eff_mask,
        chunk_size=cfg.chunk_size,
        token_chunk_size=cfg.token_chunk_size,
        temperature=cfg.temperature,
    )
    kl_scale = float(cfg.kl_weight) * ((cfg.temperature ** 2) if cfg.scale_kl_by_temp_sq else 1.0)
    kl_term = kl_scale * kl_loss

    acc = kl_loss.dtype
    if want_hidden:
        pairs = _align_layer_hidden_pairs(
            shared_teacher.text_model, shared_teacher.text_model, t_all, s_all, h_teacher, h_student
        )
        mse_terms = []
        for th, sh in pairs:
            if th.shape != sh.shape:
                continue
            diff_sq = (sh.to(acc) - th.to(acc)).pow(2).mean(dim=-1)
            if eff_mask is not None:
                m_float = eff_mask.to(dtype=acc)
                mse_terms.append((diff_sq * m_float).sum() / m_float.sum().clamp_min(1.0))
            else:
                mse_terms.append(diff_sq.mean())
        hidden_mse_loss = torch.stack(mse_terms).mean() if mse_terms else kl_loss.new_zeros(())
    else:
        hidden_mse_loss = kl_loss.new_zeros(())

    aux_loss = kl_loss.new_zeros(())
    if hasattr(stack_or_model, "aux_losses") and callable(stack_or_model.aux_losses):
        for k, v in stack_or_model.aux_losses().items():
            if k == "loop_aux" and not cfg.include_loop_aux:
                continue
            aux_loss = aux_loss + v.to(acc)
    else:
        if hssd is not None:
            aux_loss = aux_loss + hssd.aux_loss().to(acc)
        if cfg.include_loop_aux:
            layers = getattr(shared_teacher.text_model, "layers", None)
            ctl = getattr(layers[0], "_ctl", None) if isinstance(layers, nn.ModuleList) and len(layers) > 0 else None
            if ctl is not None and callable(getattr(ctl, "aux_loss", None)):
                la = ctl.aux_loss()
                if la is not None:
                    aux_loss = aux_loss + la.to(acc)

    if labels is not None and cfg.ce_loss_weight > 0.0:
        ce_raw, _ = lm_loss(
            h_student,
            w_s,
            labels,
            impl="chunked",
            chunk_size=cfg.token_chunk_size,
            shift=True,
            ignore_index=cfg.ignore_index,
        )
        ce_loss = ce_raw.to(acc)
    else:
        ce_loss = kl_loss.new_zeros(())

    total_loss = (
        kl_term
        + float(cfg.hidden_mse_weight) * hidden_mse_loss
        + float(cfg.aux_loss_weight) * aux_loss
        + float(cfg.ce_loss_weight) * ce_loss
    )
    return DistillOutput(
        loss=total_loss,
        kl_loss=kl_loss,
        hidden_mse_loss=hidden_mse_loss,
        aux_loss=aux_loss,
        ce_loss=ce_loss,
        n_tokens=n_valid,
    )


# ============================================================================ distiller

def _extract_lm_head_params(model: nn.Module) -> tuple[torch.Tensor, torch.Tensor | None]:
    tm = find_text_model(model)
    cfg = getattr(tm, "config", getattr(model, "config", None))
    vocab_size = getattr(cfg, "vocab_size", None)
    head = find_lm_head(model, vocab_size)
    require(head is not None and hasattr(head, "weight"), "Could not locate `lm_head` with `.weight`", model=model)
    assert head is not None
    bias = getattr(head, "bias", None)
    return head.weight, (bias if isinstance(bias, torch.Tensor) else None)


def _align_layer_hidden_pairs(
    teacher_tm: nn.Module,
    student_tm: nn.Module,
    t_states: tuple[torch.Tensor, ...] | None,
    s_states: tuple[torch.Tensor, ...] | None,
    t_last: torch.Tensor,
    s_last: torch.Tensor,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Pair teacher and student hidden states (handling both 1-to-1 and looped student schedules)."""
    if not t_states or not s_states:
        return [(t_last, s_last)]

    t_layers = t_states[1:]  # drop initial embedding output at index 0
    s_layers = s_states[1:]
    if len(t_layers) == len(s_layers) and len(t_layers) > 0:
        return [*zip(t_layers, s_layers), (t_last, s_last)]

    s_mod_layers = getattr(student_tm, "layers", None)
    ctl = getattr(s_mod_layers[0], "_ctl", None) if (isinstance(s_mod_layers, nn.ModuleList) and s_mod_layers) else None
    schedule = getattr(ctl, "schedule", None)
    active_passes = int(getattr(ctl, "active_passes", 1)) if ctl is not None else 1
    if schedule is not None and len(schedule) == len(s_layers):
        last_vidx_for_phys: dict[int, int] = {}
        for slot in schedule:
            if slot.r <= active_passes:
                last_vidx_for_phys[int(slot.phys)] = int(slot.vidx)
        pairs: list[tuple[torch.Tensor, torch.Tensor]] = []
        for phys_idx, t_h in enumerate(t_layers):
            vidx = last_vidx_for_phys.get(phys_idx)
            if vidx is not None and vidx < len(s_layers) and t_h.shape == s_layers[vidx].shape:
                pairs.append((t_h, s_layers[vidx]))
        pairs.append((t_last, s_last))
        return pairs

    return [(t_last, s_last)]


def _run_text_backbone(
    text_model: nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None,
    output_hidden_states: bool,
    *,
    position_ids: torch.Tensor | None = None,
) -> tuple[torch.Tensor, tuple[torch.Tensor, ...] | None]:
    kwargs: dict[str, Any] = {"input_ids": input_ids, "use_cache": False, "output_hidden_states": output_hidden_states}
    if attention_mask is not None:
        kwargs["attention_mask"] = attention_mask
    if position_ids is not None:
        kwargs["position_ids"] = position_ids
    out = text_model(**kwargs)
    last_h = getattr(out, "last_hidden_state", None)
    if last_h is None and isinstance(out, (tuple, list)) and len(out) > 0:
        last_h = out[0]
    require(isinstance(last_h, torch.Tensor) and last_h.ndim == 3, "text_model returned no 3D `last_hidden_state`")
    assert isinstance(last_h, torch.Tensor)
    all_h = getattr(out, "hidden_states", None)
    return last_h, (tuple(all_h) if isinstance(all_h, (tuple, list)) else None)


class HSSDDistiller:
    """Teacher-student KL + layer-MSE + write-budget distiller (`Track A -> Track B HSSD v3`)."""

    def __init__(
        self,
        teacher: nn.Module | SharedBackboneTeacher,
        student: nn.Module,
        handle: HSSDHandle | None = None,
        cfg: HSSDDistillConfig | dict[str, Any] | None = None,
        *,
        chunk_size: int | None = None,
        temperature: float | None = None,
        hidden_mse_weight: float | None = None,
        aux_loss_weight: float | None = None,
        ce_loss_weight: float | None = None,
    ):
        if not isinstance(cfg, HSSDDistillConfig):
            cfg = copy.deepcopy(HSSDDistillConfig.from_dict(cfg))
        else:
            cfg = copy.deepcopy(cfg)
        if chunk_size is not None:
            cfg.chunk_size = int(chunk_size)
        if temperature is not None:
            cfg.temperature = float(temperature)
        if hidden_mse_weight is not None:
            cfg.hidden_mse_weight = float(hidden_mse_weight)
        if aux_loss_weight is not None:
            cfg.aux_loss_weight = float(aux_loss_weight)
        if ce_loss_weight is not None:
            cfg.ce_loss_weight = float(ce_loss_weight)
        cfg.validate()

        self.shared_teacher: SharedBackboneTeacher | None = (
            teacher if isinstance(teacher, SharedBackboneTeacher) else None
        )
        self.teacher: nn.Module = teacher.model if isinstance(teacher, SharedBackboneTeacher) else teacher
        self.student = student
        self.cfg = cfg
        self.handle: HSSDHandle | None = handle or getattr(student, "_hssd_handle", None)
        self.teacher_tm = find_text_model(self.teacher)
        self.student_tm = find_text_model(student)

    def _loop_aux_loss(self) -> torch.Tensor | None:
        if not self.cfg.include_loop_aux:
            return None
        layers = getattr(self.student_tm, "layers", None)
        if not isinstance(layers, nn.ModuleList) or len(layers) == 0:
            return None
        ctl = getattr(layers[0], "_ctl", None)
        if ctl is None or not callable(getattr(ctl, "aux_loss", None)):
            return None
        return ctl.aux_loss()

    def compute_loss(
        self,
        input_ids: torch.Tensor,
        *,
        position_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        loss_mask: torch.Tensor | None = None,
        depth: int | None = None,
        step: int | None = None,
    ) -> DistillOutput:
        if self.shared_teacher is not None:
            return compute_shared_distill_loss(
                self.shared_teacher.stack_or_model,
                self.shared_teacher,
                self.cfg,
                input_ids,
                position_ids=position_ids,
                attention_mask=attention_mask,
                labels=labels,
                loss_mask=loss_mask,
                depth=depth,
                step=step,
            )

        want_hidden = self.cfg.hidden_mse_weight > 0.0
        was_teacher_training = bool(self.teacher.training)
        if was_teacher_training:
            self.teacher.eval()
        try:
            with torch.no_grad():
                h_teacher, t_all = _run_text_backbone(
                    self.teacher_tm, input_ids, attention_mask, want_hidden, position_ids=position_ids
                )
                h_teacher = h_teacher.detach()
                if t_all is not None:
                    t_all = tuple(t.detach() for t in t_all)
        finally:
            if was_teacher_training:
                self.teacher.train()

        h_student, s_all = _run_text_backbone(
            self.student_tm, input_ids, attention_mask, want_hidden, position_ids=position_ids
        )
        w_t, b_t = _extract_lm_head_params(self.teacher)
        w_s, b_s = _extract_lm_head_params(self.student)

        eff_mask = loss_mask
        if eff_mask is None and attention_mask is not None and attention_mask.shape == input_ids.shape:
            eff_mask = attention_mask != 0
        if eff_mask is None and labels is not None and self.cfg.mask_kl_by_labels:
            eff_mask = labels != self.cfg.ignore_index

        kl_loss, n_valid = chunked_vocab_kl_loss(
            h_teacher,
            w_t,
            h_student,
            w_s,
            b_teacher=b_t,
            b_student=b_s,
            loss_mask=eff_mask,
            chunk_size=self.cfg.chunk_size,
            token_chunk_size=self.cfg.token_chunk_size,
            temperature=self.cfg.temperature,
        )
        kl_scale = float(self.cfg.kl_weight) * (
            (self.cfg.temperature ** 2) if self.cfg.scale_kl_by_temp_sq else 1.0
        )
        kl_term = kl_scale * kl_loss

        acc = kl_loss.dtype
        if want_hidden:
            pairs = _align_layer_hidden_pairs(self.teacher_tm, self.student_tm, t_all, s_all, h_teacher, h_student)
            mse_terms = []
            for th, sh in pairs:
                if th.shape != sh.shape:
                    continue
                diff_sq = (sh.to(acc) - th.to(acc)).pow(2).mean(dim=-1)
                if eff_mask is not None:
                    m_float = eff_mask.to(dtype=acc)
                    mse_terms.append((diff_sq * m_float).sum() / m_float.sum().clamp_min(1.0))
                else:
                    mse_terms.append(diff_sq.mean())
            hidden_mse_loss = torch.stack(mse_terms).mean() if mse_terms else kl_loss.new_zeros(())
        else:
            hidden_mse_loss = kl_loss.new_zeros(())

        aux_loss = kl_loss.new_zeros(())
        if self.handle is not None:
            aux_loss = aux_loss + self.handle.aux_loss().to(acc)
        loop_aux = self._loop_aux_loss()
        if loop_aux is not None:
            aux_loss = aux_loss + loop_aux.to(acc)

        if labels is not None and self.cfg.ce_loss_weight > 0.0:
            ce_raw, _ = lm_loss(
                h_student,
                w_s,
                labels,
                impl="chunked",
                chunk_size=self.cfg.token_chunk_size,
                shift=True,
                ignore_index=self.cfg.ignore_index,
            )
            ce_loss = ce_raw.to(acc)
        else:
            ce_loss = kl_loss.new_zeros(())

        total_loss = (
            kl_term
            + float(self.cfg.hidden_mse_weight) * hidden_mse_loss
            + float(self.cfg.aux_loss_weight) * aux_loss
            + float(self.cfg.ce_loss_weight) * ce_loss
        )
        return DistillOutput(
            loss=total_loss,
            kl_loss=kl_loss,
            hidden_mse_loss=hidden_mse_loss,
            aux_loss=aux_loss,
            ce_loss=ce_loss,
            n_tokens=n_valid,
        )

    def __call__(
        self,
        input_ids: torch.Tensor,
        *,
        position_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        loss_mask: torch.Tensor | None = None,
        depth: int | None = None,
        step: int | None = None,
    ) -> DistillOutput:
        return self.compute_loss(
            input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            labels=labels,
            loss_mask=loss_mask,
            depth=depth,
            step=step,
        )


__all__ = [
    "DistillOutput",
    "HSSDDistillConfig",
    "HSSDDistiller",
    "SharedBackboneTeacher",
    "chunked_vocab_kl_from_logits",
    "chunked_vocab_kl_loss",
    "compute_shared_distill_loss",
    "exact_full_vocab_kl_loss",
    "run_layerwise_warmup_microstep",
]
