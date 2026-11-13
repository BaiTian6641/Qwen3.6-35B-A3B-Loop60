"""Memory-lean LM losses: never materialise the full [N, V] logits in the autograd graph.

Qwen3.6 has V = 248k: fp32 logits for 16k tokens are ~16 GB, plus the same again for the
softmax gradient. Paths:

* "cce"     - Apple's Cut Cross-Entropy (`cut_cross_entropy.linear_cross_entropy`), CUDA, bf16/fp16.
* "chunked" - custom `torch.autograd.Function`: forward computes per-chunk logits and keeps only
              the per-token logsumexp; backward recomputes each chunk's logits and produces the
              hidden / weight gradients chunk by chunk (weight grad only if it requires grad).
              Peak extra memory = one [chunk, V] fp32 block.
* "naive"   - full logits (reference / tests).

`token_logprobs` (differentiable, same chunking) and `token_entropy` (no grad) serve GRPO.
"""
from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn.functional as F

from qloop.debug import require

log = logging.getLogger("qloop")


def cce_available() -> bool:
    try:
        import cut_cross_entropy  # noqa: F401
    except Exception:
        return False
    return True


def _acc_dtype(t: torch.Tensor) -> torch.dtype:
    return torch.float64 if t.dtype == torch.float64 else torch.float32


def _raw_logits(h: torch.Tensor, w: torch.Tensor, acc: torch.dtype) -> torch.Tensor:
    """[n, V] logits; matmul in the input dtype (fp32 accumulate on GPU), result upcast."""
    return F.linear(h, w.to(h.dtype)).to(acc)


def _softcap(x: torch.Tensor, cap: float | None) -> torch.Tensor:
    return x if cap is None else cap * torch.tanh(x / cap)


def _softcap_grad(z: torch.Tensor, cap: float | None) -> torch.Tensor | None:
    """d softcap(x) / dx expressed through the capped value z: 1 - (z / cap)^2."""
    return None if cap is None else 1 - (z / cap) ** 2


def _chunk_backward(h: torch.Tensor, w: torch.Tensor, dz_fn: Any, chunk_size: int, need_h: bool,
                    need_w: bool) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Shared chunked backward: dz_fn(s, e, raw_logits) -> dL/d(raw logits) for rows [s, e)."""
    acc = _acc_dtype(h)
    gh = torch.empty_like(h) if need_h else None
    gw = torch.zeros(w.shape, dtype=acc, device=w.device) if need_w else None
    wc = w.to(h.dtype)
    for s in range(0, h.shape[0], chunk_size):
        e = min(s + chunk_size, h.shape[0])
        hc = h[s:e]
        draw = dz_fn(s, e, _raw_logits(hc, w, acc)).to(h.dtype)
        if gh is not None:
            gh[s:e] = draw @ wc
        if gw is not None:
            gw.add_((draw.t() @ hc).to(acc))
    return gh, (gw.to(w.dtype) if gw is not None else None)


class _ChunkedLinearCE(torch.autograd.Function):
    """sum_i [ lse_i - z_i[t_i] + z_coef * lse_i^2 ],  z = softcap(h W^T). All rows are valid."""

    @staticmethod
    def forward(ctx: Any, h: torch.Tensor, w: torch.Tensor, targets: torch.Tensor, chunk_size: int,
                softcap: float | None, z_coef: float) -> torch.Tensor:
        acc = _acc_dtype(h)
        n = h.shape[0]
        lse = torch.empty(n, dtype=acc, device=h.device)
        total = torch.zeros((), dtype=acc, device=h.device)
        for s in range(0, n, chunk_size):
            e = min(s + chunk_size, n)
            z = _softcap(_raw_logits(h[s:e], w, acc), softcap)
            lc = torch.logsumexp(z, dim=-1)
            tgt = z.gather(1, targets[s:e, None]).squeeze(1)
            total = total + (lc - tgt).sum()
            if z_coef:
                total = total + z_coef * (lc * lc).sum()
            lse[s:e] = lc
        ctx.save_for_backward(h, w, targets, lse)
        ctx.chunk_size, ctx.softcap, ctx.z_coef = chunk_size, softcap, z_coef
        return total

    @staticmethod
    def backward(ctx: Any, grad_out: torch.Tensor) -> Any:
        h, w, targets, lse = ctx.saved_tensors
        cap, z_coef = ctx.softcap, ctx.z_coef
        g = grad_out.to(lse.dtype)

        def dz_fn(s: int, e: int, raw: torch.Tensor) -> torch.Tensor:
            z = _softcap(raw, cap)
            lc = lse[s:e, None]
            dz = torch.exp(z - lc)
            if z_coef:
                dz = dz * (1 + 2 * z_coef * lc)
            dz[torch.arange(e - s, device=dz.device), targets[s:e]] -= 1
            dz = dz * g
            sg = _softcap_grad(z, cap)
            return dz if sg is None else dz * sg

        gh, gw = _chunk_backward(h, w, dz_fn, ctx.chunk_size, ctx.needs_input_grad[0], ctx.needs_input_grad[1])
        return gh, gw, None, None, None, None


class _ChunkedTokenLogprob(torch.autograd.Function):
    """lp_i = z_i[t_i] - logsumexp(z_i),  z = softcap(h W^T) / temperature."""

    @staticmethod
    def forward(ctx: Any, h: torch.Tensor, w: torch.Tensor, targets: torch.Tensor, chunk_size: int,
                temperature: float, softcap: float | None) -> torch.Tensor:
        acc = _acc_dtype(h)
        n = h.shape[0]
        lse = torch.empty(n, dtype=acc, device=h.device)
        out = torch.empty(n, dtype=acc, device=h.device)
        for s in range(0, n, chunk_size):
            e = min(s + chunk_size, n)
            z = _softcap(_raw_logits(h[s:e], w, acc), softcap) / temperature
            lc = torch.logsumexp(z, dim=-1)
            out[s:e] = z.gather(1, targets[s:e, None]).squeeze(1) - lc
            lse[s:e] = lc
        ctx.save_for_backward(h, w, targets, lse)
        ctx.chunk_size, ctx.temperature, ctx.softcap = chunk_size, temperature, softcap
        return out

    @staticmethod
    def backward(ctx: Any, grad_out: torch.Tensor) -> Any:
        h, w, targets, lse = ctx.saved_tensors
        cap, temp = ctx.softcap, ctx.temperature
        g = grad_out.to(lse.dtype)

        def dz_fn(s: int, e: int, raw: torch.Tensor) -> torch.Tensor:
            zc = _softcap(raw, cap)
            p = torch.exp(zc / temp - lse[s:e, None])
            gc = g[s:e, None]
            dz = -p * gc
            rows = torch.arange(e - s, device=dz.device)
            dz[rows, targets[s:e]] += g[s:e]
            dz = dz / temp
            sg = _softcap_grad(zc, cap)
            return dz if sg is None else dz * sg

        gh, gw = _chunk_backward(h, w, dz_fn, ctx.chunk_size, ctx.needs_input_grad[0], ctx.needs_input_grad[1])
        return gh, gw, None, None, None, None


# ============================================================================ public API

def _flatten_shift(hidden: torch.Tensor, labels: torch.Tensor, shift: bool) -> tuple[torch.Tensor, torch.Tensor]:
    require(hidden.dim() in (2, 3), "hidden must be [B,T,H] or [N,H]", hidden=hidden)
    require(tuple(labels.shape) == tuple(hidden.shape[:-1]), "labels shape must match hidden[..., :-1]",
            hidden=hidden, labels=labels)
    if shift:
        if hidden.dim() == 3:
            hidden, labels = hidden[:, :-1], labels[:, 1:]
        else:
            hidden, labels = hidden[:-1], labels[1:]
    return hidden.reshape(-1, hidden.shape[-1]), labels.reshape(-1)


def _check_vocab(weight: torch.Tensor, hidden: torch.Tensor) -> None:
    require(weight.dim() == 2 and weight.shape[1] == hidden.shape[-1], "weight must be [V, H] matching hidden",
            weight=weight, hidden=hidden)


def lm_loss(hidden: torch.Tensor, weight: torch.Tensor, labels: torch.Tensor, impl: str = "auto",
            chunk_size: int = 4096, shift: bool = True, softcap: float | None = None, z_loss: float = 0.0,
            ignore_index: int = -100) -> tuple[torch.Tensor, torch.Tensor]:
    """Mean next-token cross-entropy of `hidden @ weight.T` against `labels` (+ z_loss * lse^2).

    hidden [B,T,H] with labels [B,T] (shift along T) or [N,H] with labels [N] (shift along N,
    i.e. one packed sequence; pass shift=False if already aligned). Returns
    `(loss_mean, n_tokens)`; `n_tokens` is a 0-d int64 tensor (no host sync) - for
    token-weighted gradient accumulation use `loss_mean * n_tokens / total_tokens`.
    impl: "auto" (cce on CUDA bf16/fp16 when installed and z_loss == 0, else chunked) | "cce" |
    "chunked" | "naive".
    """
    _check_vocab(weight, hidden)
    require(impl in ("auto", "cce", "chunked", "naive"), f"unknown lm_loss impl {impl!r}")
    require(chunk_size > 0, "chunk_size must be positive", chunk_size=chunk_size)
    if impl == "auto":
        ok = cce_available() and hidden.is_cuda and hidden.dtype in (torch.bfloat16, torch.float16) and not z_loss
        impl = "cce" if ok else "chunked"
    if impl == "cce":
        return _lm_loss_cce(hidden, weight, labels, shift, softcap, z_loss, ignore_index)
    h, y = _flatten_shift(hidden, labels, shift)
    valid = y != ignore_index
    n_tok = valid.sum()
    if impl == "naive":
        logits = _softcap(_raw_logits(h, weight, _acc_dtype(h)), softcap)
        total = F.cross_entropy(logits, y, ignore_index=ignore_index, reduction="sum")
        if z_loss:
            lse = torch.logsumexp(logits, dim=-1)
            total = total + z_loss * (lse[valid] ** 2).sum()
    else:
        idx = valid.nonzero(as_tuple=True)[0]
        hv, yv = h.index_select(0, idx), y.index_select(0, idx)
        require(bool(((yv >= 0) & (yv < weight.shape[0])).all()) if yv.numel() else True,
                "label id out of range [0, V)", labels=yv, vocab=weight.shape[0])
        total = _ChunkedLinearCE.apply(hv, weight, yv, int(chunk_size), softcap, float(z_loss))
    return total / n_tok.clamp_min(1).to(total.dtype), n_tok


def _lm_loss_cce(hidden: torch.Tensor, weight: torch.Tensor, labels: torch.Tensor, shift: bool,
                 softcap: float | None, z_loss: float, ignore_index: int) -> tuple[torch.Tensor, torch.Tensor]:
    if not cce_available():
        raise ImportError("impl='cce' but `cut_cross_entropy` is not installed: "
                          "pip install 'cut-cross-entropy @ git+https://github.com/apple/ml-cross-entropy.git'")
    require(not z_loss, "z_loss is not supported by the CCE path; use impl='chunked'")
    from cut_cross_entropy import linear_cross_entropy

    w = weight if weight.dtype == hidden.dtype else weight.to(hidden.dtype)
    kw: dict[str, Any] = {"ignore_index": ignore_index, "reduction": "sum"}
    if softcap is not None:
        kw["softcap"] = softcap
    y_shift = labels[:, 1:] if (shift and labels.dim() == 2) else (labels[1:] if shift else labels)
    n_tok = (y_shift != ignore_index).sum()
    total = None
    if shift and hidden.dim() == 3:
        try:
            total = linear_cross_entropy(hidden, w, labels, shift=True, **kw)
        except TypeError:  # old CCE without `shift=`
            total = None
    if total is None:
        h, y = _flatten_shift(hidden, labels, shift)
        total = linear_cross_entropy(h, w, y, **kw)
    total = total.float()
    return total / n_tok.clamp_min(1).to(total.dtype), n_tok


def token_logprobs(hidden: torch.Tensor, weight: torch.Tensor, target_ids: torch.Tensor, chunk_size: int = 4096,
                   temperature: float = 1.0, softcap: float | None = None) -> torch.Tensor:
    """log softmax(softcap(h W^T) / temperature)[target], differentiable w.r.t. hidden (and weight
    if it requires grad) without materialising [N, V] in the graph. hidden [..., H], target_ids
    [...]; returns fp32 (fp64 for fp64 input) with target_ids' shape ([N] for 2D input)."""
    _check_vocab(weight, hidden)
    require(tuple(target_ids.shape) == tuple(hidden.shape[:-1]), "target_ids must match hidden[..., :-1]",
            hidden=hidden, target_ids=target_ids)
    require(temperature > 0, "temperature must be > 0", temperature=temperature)
    h = hidden.reshape(-1, hidden.shape[-1])
    y = target_ids.reshape(-1).long()
    out = _ChunkedTokenLogprob.apply(h, weight, y, int(chunk_size), float(temperature), softcap)
    return out.view(target_ids.shape)


@torch.no_grad()
def token_entropy(hidden: torch.Tensor, weight: torch.Tensor, chunk_size: int = 4096, temperature: float = 1.0,
                  softcap: float | None = None) -> torch.Tensor:
    """Per-token entropy (nats) of softmax(softcap(h W^T) / temperature); no grad. Shape hidden[..., :-1]."""
    _check_vocab(weight, hidden)
    h = hidden.reshape(-1, hidden.shape[-1])
    acc = _acc_dtype(h)
    out = torch.empty(h.shape[0], dtype=acc, device=h.device)
    for s in range(0, h.shape[0], chunk_size):
        e = min(s + chunk_size, h.shape[0])
        z = _softcap(_raw_logits(h[s:e], weight, acc), softcap) / temperature
        lse = torch.logsumexp(z, dim=-1, keepdim=True)
        p = torch.exp(z - lse)
        out[s:e] = lse.squeeze(-1) - (p * z).sum(-1)
    return out.view(hidden.shape[:-1])
