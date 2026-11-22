"""Cache-based sampling loop shared by eval, GRPO rollouts and the OpenAI-compatible server.

Works with anything that follows the HF causal-LM calling convention

    out = model(input_ids=ids, past_key_values=cache, use_cache=True)
    out.logits [B, T, V], out.past_key_values

i.e. the toy model, the loop-retrofitted model and stock HF models. Design choices:

* **Per-sequence by default.** The toy / looped model have no attention mask support (Gated
  DeltaNet recurrent states would be polluted by pad tokens), so prompts are run one at a time.
  The `n` samples of ONE prompt are still batched (same length => no padding needed).
  `padded_batching=True` + `batch_size>1` enables LEFT-padded multi-prompt batches with an
  attention mask, for models that honour it (HF Qwen3.5/3.6 masks padded DeltaNet inputs).
* **Reproducible sampling.** Every (prompt, sample) row owns its own `torch.Generator`, seeded
  from the per-prompt seed => results do not depend on batch composition or resume order.
* **Loop depth.** `depth=r` calls `loop_controller.set_active_passes(r)` for the duration of the
  call and restores the previous value afterwards (also on error).
* **Logprobs.** `return_logprobs=True` returns the log-probability of each sampled token under
  the *unmodified* model distribution (log_softmax of raw logits, temperature 1, no top-p),
  which is the quantity a policy-gradient trainer recomputes.
"""
from __future__ import annotations

import contextlib
import inspect
import logging
import time
from collections.abc import Iterator, Sequence
from dataclasses import asdict, dataclass
from typing import Any, Union

import torch

from qloop.debug import describe, require
from qloop.model._cache_compat import extract_output_cache

log = logging.getLogger("qloop")

Prompt = Union[str, list]  # plain text (no template) or a list of chat messages


@dataclass
class GenOut:
    text: str
    token_ids: list[int]
    logprobs: list[float] | None = None
    finish_reason: str = "length"      # "stop" (eos token / stop string) | "length"
    prompt_tokens: int = 0
    stop_reason: str | int | None = None  # eos token id or stop string that ended the sample

    @property
    def completion_tokens(self) -> int:
        return len(self.token_ids)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _model_device(model: Any) -> torch.device:
    try:
        return next(model.parameters()).device
    except (StopIteration, AttributeError, TypeError):
        return torch.device("cpu")


def _forward_params(model: Any) -> set[str]:
    try:
        sig = inspect.signature(model.forward)
    except (TypeError, ValueError, AttributeError):
        return set()
    return {n for n, p in sig.parameters.items() if p.kind is not inspect.Parameter.VAR_KEYWORD}


def _as_id_list(x: Any) -> list[int]:
    """Normalise the many return types of HF tokenizers to a flat list[int]."""
    if isinstance(x, dict) or hasattr(x, "input_ids") or (hasattr(x, "keys") and "input_ids" in x):
        x = x["input_ids"]
    if isinstance(x, torch.Tensor):
        x = x.tolist()
    if hasattr(x, "tolist") and not isinstance(x, list):
        x = x.tolist()
    if isinstance(x, (list, tuple)) and x and isinstance(x[0], (list, tuple)):
        require(len(x) == 1, "tokenizer returned a batch where one sequence was expected", n=len(x))
        x = x[0]
    return [int(t) for t in x]


def _norm_message(m: Any) -> dict[str, Any]:
    require(isinstance(m, dict) and "role" in m, "chat message must be a dict with 'role'", message=m)
    m = dict(m)
    content = m.get("content")
    if isinstance(content, list):  # OpenAI content parts -> plain text
        texts = []
        for part in content:
            if isinstance(part, str):
                texts.append(part)
            elif isinstance(part, dict) and part.get("type") in ("text", "input_text"):
                texts.append(str(part.get("text", "")))
            elif isinstance(part, dict):
                raise ValueError(f"unsupported content part type {part.get('type')!r} (text-only Generator)")
        m["content"] = "".join(texts)
    elif content is None:
        m["content"] = ""
    return m


class Generator:
    """`generate(prompts, ...) -> list (per prompt) of list (per sample) of GenOut`."""

    def __init__(self, model: Any, tokenizer: Any, device: Any = None, loop_controller: Any = None, *,
                 batch_size: int = 1, padded_batching: bool = False, eos_token_ids: Sequence[int] | None = None,
                 max_context: int | None = None, chat_template_kwargs: dict[str, Any] | None = None):
        self.model = model
        self.tokenizer = tokenizer
        self.device = torch.device(device) if device is not None else _model_device(model)
        self.loop_controller = loop_controller
        self.batch_size = max(1, int(batch_size))
        self.padded_batching = bool(padded_batching)
        self.max_context = max_context
        self.chat_template_kwargs = dict(chat_template_kwargs or {})
        self.eos_token_ids = self._resolve_eos(eos_token_ids)
        params = _forward_params(model)
        self._accepts_mask = "attention_mask" in params
        self._logits_kw = next((k for k in ("logits_to_keep", "num_logits_to_keep") if k in params), None)
        self._cache_kw = "cache" if ("cache" in params and "past_key_values" not in params) else "past_key_values"
        if self.padded_batching and not self._accepts_mask:
            raise ValueError(f"padded_batching=True but {type(model).__name__}.forward has no attention_mask "
                             "argument; left padding would silently corrupt outputs")
        log.debug("Generator: device=%s eos=%s batch_size=%d padded=%s logits_kw=%s loop=%s", self.device,
                  self.eos_token_ids, self.batch_size, self.padded_batching, self._logits_kw,
                  loop_controller is not None)

    # ------------------------------------------------------------------ tokenizer helpers
    def _resolve_eos(self, explicit: Sequence[int] | None) -> set[int]:
        if explicit is not None:
            return {int(t) for t in explicit}
        ids: set[int] = set()
        for src in (getattr(self.tokenizer, "eos_token_id", None),
                    getattr(getattr(self.model, "generation_config", None), "eos_token_id", None)):
            if src is None:
                continue
            ids.update(int(t) for t in (src if isinstance(src, (list, tuple, set)) else [src]))
        if not ids:
            log.warning("Generator: no eos token id found on tokenizer/model; generation stops only at "
                        "max_new_tokens or stop strings")
        return ids

    @property
    def max_depth(self) -> int | None:
        return None if self.loop_controller is None else int(self.loop_controller.spec.num_passes)

    def _pad_id(self) -> int:
        pid = getattr(self.tokenizer, "pad_token_id", None)
        if pid is None:
            pid = min(self.eos_token_ids) if self.eos_token_ids else 0
        return int(pid)

    def encode_text(self, text: str) -> list[int]:
        try:
            return _as_id_list(self.tokenizer.encode(text, add_special_tokens=False))
        except TypeError:
            return _as_id_list(self.tokenizer.encode(text))

    def decode(self, ids: Sequence[int]) -> str:
        try:
            return self.tokenizer.decode(list(ids), skip_special_tokens=True)
        except TypeError:
            return self.tokenizer.decode(list(ids))

    def render_prompt(self, prompt: Prompt, chat_template_kwargs: dict[str, Any] | None = None) -> str:
        if isinstance(prompt, str):
            return prompt
        require(isinstance(prompt, (list, tuple)) and len(prompt) > 0,
                "prompt must be a str or a non-empty list of chat messages", prompt=prompt)
        kw = dict(self.chat_template_kwargs)
        kw.update(chat_template_kwargs or {})
        messages = [_norm_message(m) for m in prompt]
        out = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, **kw)
        if isinstance(out, str):
            return out
        return self.decode(_as_id_list(out))  # tokenizer ignored tokenize=False

    def encode_prompt(self, prompt: Prompt, chat_template_kwargs: dict[str, Any] | None = None) -> list[int]:
        ids = self.encode_text(self.render_prompt(prompt, chat_template_kwargs))
        require(len(ids) > 0, "prompt encoded to zero tokens", prompt=prompt)
        if self.max_context is not None and len(ids) >= self.max_context:
            raise ValueError(f"prompt has {len(ids)} tokens >= max_context={self.max_context}")
        return ids

    # ------------------------------------------------------------------ context managers
    @contextlib.contextmanager
    def depth(self, depth: int | None) -> Iterator[None]:
        """Temporarily set the loop depth (no-op for depth=None)."""
        if depth is None:
            yield
            return
        lc = self.loop_controller
        require(lc is not None, "loop depth requested but this Generator has no loop_controller "
                "(pass the LoopController returned by apply_loop)", depth=depth)
        prev = lc.active_passes
        lc.set_active_passes(int(depth))
        try:
            yield
        finally:
            lc.set_active_passes(prev)

    @contextlib.contextmanager
    def _eval_mode(self) -> Iterator[None]:
        was_training = bool(getattr(self.model, "training", False))
        if was_training:
            self.model.eval()
        try:
            yield
        finally:
            if was_training:
                self.model.train()

    # ------------------------------------------------------------------ public API
    def generate(self, prompts: Sequence[Prompt], max_new_tokens: int = 256, temperature: float = 0.0,
                 top_p: float = 1.0, n: int = 1, stop: Sequence[str] | None = None,
                 seed: int | Sequence[int] = 0, *, top_k: int = 0, depth: int | None = None,
                 return_logprobs: bool = False, chat_template_kwargs: dict[str, Any] | None = None,
                 batch_size: int | None = None) -> list[list[GenOut]]:
        if isinstance(prompts, (str, dict)):
            raise TypeError("generate() takes a LIST of prompts; wrap a single prompt as [prompt]")
        prompts = list(prompts)
        require(n >= 1 and max_new_tokens >= 0, "need n >= 1 and max_new_tokens >= 0", n=n,
                max_new_tokens=max_new_tokens)
        require(0.0 < top_p <= 1.0, "top_p must be in (0, 1]", top_p=top_p)
        if isinstance(seed, int):
            seeds = [seed * 1_000_003 + i for i in range(len(prompts))]
        else:
            seeds = [int(s) for s in seed]
            require(len(seeds) == len(prompts), "need one seed per prompt", seeds=len(seeds), prompts=len(prompts))
        stops = [s for s in (stop or []) if s]
        encoded = [self.encode_prompt(p, chat_template_kwargs) for p in prompts]
        bs = max(1, int(batch_size or self.batch_size))
        if not self.padded_batching:
            bs = 1
        results: list[list[GenOut]] = [[] for _ in prompts]
        t0 = time.time()
        with self.depth(depth), self._eval_mode(), torch.no_grad():
            order = sorted(range(len(prompts)), key=lambda i: -len(encoded[i])) if bs > 1 else range(len(prompts))
            order = list(order)
            for c in range(0, len(order), bs):
                chunk = order[c:c + bs]
                rows, keys, row_seeds = [], [], []
                for pi in chunk:
                    for j in range(n):
                        rows.append(encoded[pi])
                        keys.append(pi)
                        row_seeds.append((seeds[pi] * 7919 + j) % (2**62))
                outs = self._generate_rows(rows, row_seeds, max_new_tokens, temperature, top_p, top_k, stops,
                                           return_logprobs)
                for pi, o in zip(keys, outs):
                    results[pi].append(o)
        ntok = sum(o.completion_tokens for r in results for o in r)
        dt = max(time.time() - t0, 1e-9)
        log.debug("generate: %d prompts x n=%d depth=%s -> %d tokens in %.1fs (%.1f tok/s)", len(prompts), n,
                  depth, ntok, dt, ntok / dt)
        return results

    # ------------------------------------------------------------------ internals
    def _forward(self, ids: torch.Tensor, cache: Any, mask: torch.Tensor | None, prefill: bool) -> Any:
        kwargs: dict[str, Any] = {"use_cache": True}
        if cache is not None:
            kwargs[self._cache_kw] = cache
        if mask is not None:
            kwargs["attention_mask"] = mask
        if prefill and self._logits_kw:
            kwargs[self._logits_kw] = 1
        out = self.model(input_ids=ids, **kwargs)
        logits = getattr(out, "logits", None)
        if logits is None and isinstance(out, (tuple, list)):
            logits = out[0]
        require(logits is not None and logits.dim() == 3, "model output has no [B,T,V] .logits", out=out)
        new_cache = extract_output_cache(out)
        require(new_cache is not None, "model returned no past_key_values/cache with use_cache=True; Generator needs a "
                "cache-capable model", model=type(self.model).__name__)
        return logits[:, -1, :].float(), new_cache

    def _sample(self, logits: torch.Tensor, temperature: float, top_p: float, top_k: int,
                gens: list[torch.Generator] | None) -> torch.Tensor:
        if temperature <= 0 or gens is None:
            return logits.argmax(dim=-1)
        x = logits / float(temperature)
        if top_k and top_k > 0 and top_k < x.shape[-1]:
            kth = torch.topk(x, top_k, dim=-1).values[..., -1:]
            x = x.masked_fill(x < kth, float("-inf"))
        probs = torch.softmax(x, dim=-1)
        if top_p < 1.0:
            sp, si = probs.sort(dim=-1, descending=True)
            cum = sp.cumsum(dim=-1)
            sp = sp.masked_fill(cum - sp > top_p, 0.0)  # always keeps the top-1 token
            probs = torch.zeros_like(probs).scatter_(-1, si, sp)
        probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-30)
        picks = [torch.multinomial(probs[i], 1, generator=gens[i]) for i in range(probs.shape[0])]
        return torch.cat(picks).to(logits.device)

    def _hit_stop(self, toks: list[int], stops: list[str], window: int) -> str | None:
        tail = self.decode(toks[-window:])
        for s in stops:
            if s in tail:
                return s
        return None

    def _generate_rows(self, rows: list[list[int]], seeds: list[int], max_new_tokens: int, temperature: float,
                       top_p: float, top_k: int, stops: list[str], want_lp: bool) -> list[GenOut]:
        b = len(rows)
        lens = [len(r) for r in rows]
        t = max(lens)
        padded = any(x != t for x in lens)
        require(not padded or self.padded_batching, "internal: rows of different length without padded batching",
                lens=lens)
        pad = self._pad_id()
        ids = torch.full((b, t), pad, dtype=torch.long)
        mask = torch.zeros((b, t), dtype=torch.long)
        for i, r in enumerate(rows):
            ids[i, t - len(r):] = torch.tensor(r, dtype=torch.long)
            mask[i, t - len(r):] = 1
        ids, mask = ids.to(self.device), mask.to(self.device)
        attn = mask if padded else None
        sampling = temperature > 0
        gens = None
        if sampling:
            gdev = self.device if self.device.type in ("cuda", "cpu") else torch.device("cpu")
            gens = [torch.Generator(device=gdev).manual_seed(int(s)) for s in seeds]
        window = (max(len(s) for s in stops) + 16) if stops else 0

        toks: list[list[int]] = [[] for _ in range(b)]
        lps: list[list[float]] = [[] for _ in range(b)]
        finish = ["length"] * b
        reason: list[str | int | None] = [None] * b
        done = [False] * b
        if max_new_tokens == 0:
            return [GenOut("", [], [] if want_lp else None, "length", lens[i]) for i in range(b)]
        try:
            logits, cache = self._forward(ids, None, attn, prefill=True)
        except Exception as e:
            raise RuntimeError(f"prefill failed for batch of {b} rows (len {t}): {type(e).__name__}: {e} | "
                               f"input_ids={describe(ids)}") from e
        for step in range(max_new_tokens):
            on_gen_dev = gens is None or gens[0].device == logits.device
            logits_s = logits if on_gen_dev else logits.to(gens[0].device)
            nxt = self._sample(logits_s, temperature, top_p, top_k, gens).to(logits.device)
            lp_vals = None
            if want_lp:
                lp_vals = torch.log_softmax(logits, dim=-1).gather(1, nxt[:, None]).squeeze(1).tolist()
            nxt_l = nxt.tolist()
            for i in range(b):
                if done[i]:
                    continue
                tok = int(nxt_l[i])
                toks[i].append(tok)
                if lp_vals is not None:
                    lps[i].append(float(lp_vals[i]))
                if tok in self.eos_token_ids:
                    done[i], finish[i], reason[i] = True, "stop", tok
                elif stops:
                    hit = self._hit_stop(toks[i], stops, window)
                    if hit is not None:
                        done[i], finish[i], reason[i] = True, "stop", hit
                if not done[i] and self.max_context is not None and lens[i] + len(toks[i]) >= self.max_context:
                    done[i] = True  # finish_reason stays "length"
            if all(done) or step == max_new_tokens - 1:
                break
            feed = torch.tensor([[tk[-1] if not d else pad] for tk, d in zip(toks, done)], dtype=torch.long,
                                device=self.device)
            if attn is not None:
                attn = torch.cat([attn, torch.ones((b, 1), dtype=attn.dtype, device=attn.device)], dim=1)
            logits, cache = self._forward(feed, cache, attn, prefill=False)

        outs = []
        for i in range(b):
            body = toks[i][:-1] if (finish[i] == "stop" and isinstance(reason[i], int)) else toks[i]
            text = self.decode(body)
            if finish[i] == "stop" and isinstance(reason[i], str):
                cut = min((text.find(s) for s in stops if s in text), default=-1)
                if cut >= 0:
                    text = text[:cut]
            outs.append(GenOut(text, toks[i], lps[i] if want_lp else None, finish[i], lens[i], reason[i]))
        return outs
