"""Memory-safe Teacher -> Student KL Distillation (`HSSDDistiller`) for Loop-HSSD v3.

Distills `Track A` (stock Qwen3.6 GDN or `GDN-2`) into `Track B` (`HSSD v3`, with or without
looped-depth `apply_loop`) without ever materializing full `[B, T, 248320]` fp32 logit tensors:

  * **2D Token x Vocab Tiled KL (`_ChunkedVocabKL`):** Evaluates `KL(p_teacher || p_student)` at
    temperature `T` directly from pre-head hidden states `(h_t, h_s)` and `lm_head` weights
    `(W_t, W_s)` in vocab tiles of `chunk_size = 4096`. Forward uses a 2-pass exact online
    `logaddexp` reduction saving only 1D `logsumexp` vectors (`[N]`); backward recomputes one
    `[tile_tok, 4096]` block at a time (`<= 64 MiB` peak instead of `65+ GiB`).
  * **Layer-Wise Hidden Alignment MSE:** Optional cosine/MSE hidden-state alignment supporting
    both 1-to-1 physical layers and looped virtual-layer schedules (mapping each physical layer
    to its final pass exit in `student`).
  * **Write-Mass Budget Integration:** Automatically folds `handle.aux_loss()` (and optional
    ground-truth `lm_loss` CE) into the composite student objective.

Clean-room Apache-2.0 / MIT PyTorch implementation.
"""
from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from ..debug import require
from ..model.hssd import HSSDHandle
from ..model.introspect import find_lm_head, find_text_model
from .loss import lm_loss


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
    teacher_track: str = "gdn2"

    def __post_init__(self) -> None:
        if self.vocab_chunk_size is not None:
            self.chunk_size = int(self.vocab_chunk_size)
        else:
            self.vocab_chunk_size = int(self.chunk_size)

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
        if (
            self.kl_weight < 0.0
            or self.hidden_mse_weight < 0.0
            or self.aux_loss_weight < 0.0
            or self.ce_loss_weight < 0.0
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
) -> tuple[torch.Tensor, tuple[torch.Tensor, ...] | None]:
    kwargs: dict[str, Any] = {"input_ids": input_ids, "use_cache": False, "output_hidden_states": output_hidden_states}
    if attention_mask is not None:
        kwargs["attention_mask"] = attention_mask
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
        teacher: nn.Module,
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
            cfg = HSSDDistillConfig.from_dict(cfg)
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

        self.teacher = teacher
        self.student = student
        self.cfg = cfg
        self.handle: HSSDHandle | None = handle or getattr(student, "_hssd_handle", None)
        self.teacher_tm = find_text_model(teacher)
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
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        loss_mask: torch.Tensor | None = None,
    ) -> DistillOutput:
        want_hidden = self.cfg.hidden_mse_weight > 0.0
        was_teacher_training = bool(self.teacher.training)
        if was_teacher_training:
            self.teacher.eval()
        try:
            with torch.no_grad():
                h_teacher, t_all = _run_text_backbone(self.teacher_tm, input_ids, attention_mask, want_hidden)
                h_teacher = h_teacher.detach()
                if t_all is not None:
                    t_all = tuple(t.detach() for t in t_all)
        finally:
            if was_teacher_training:
                self.teacher.train()

        h_student, s_all = _run_text_backbone(self.student_tm, input_ids, attention_mask, want_hidden)
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
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        loss_mask: torch.Tensor | None = None,
    ) -> DistillOutput:
        return self.compute_loss(input_ids, attention_mask=attention_mask, labels=labels, loss_mask=loss_mask)


__all__ = [
    "DistillOutput",
    "HSSDDistillConfig",
    "HSSDDistiller",
    "chunked_vocab_kl_from_logits",
    "chunked_vocab_kl_loss",
    "exact_full_vocab_kl_loss",
]
