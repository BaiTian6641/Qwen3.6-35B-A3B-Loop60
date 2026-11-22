"""Stage S4/S5: GRPO recursive self-improvement with verifiable rewards + teacher guidance.

One step:
  1. sample loop depth r (rollout and update use the SAME depth => on-policy)
  2. sample `prompts_per_step` tasks, generate `group_size` completions each (cache-based sampler)
  3. verifiable rewards in a thread pool (math/AIME/MCQ/ARC/HLE/GAIA scorers, LiveCodeBench-style
     unit tests, real KernelBench compile+correctness(+speedup bonus), regex format rewards, or an
     HTTP reward service for container tasks), optional DAPO overlong shaping
  4. group-relative advantages (std-normalised GRPO or Dr.GRPO mean-only); zero-variance groups
     carry no signal and are dropped
  5. groups where *every* sample failed -> ask an external teacher (OpenAI-compatible endpoint),
     verify its answer, and add a weighted NLL term on verified teacher solutions (TGPO-style)
  6. token-level clipped surrogate with asymmetric "clip-higher" bounds, optional k3 KL to a
     frozen snapshot of the initial trainables, entropy monitoring with adaptive temperature
  7. regression gate every N steps on held-out benchmarks: promote the checkpoint only if the
     score improves and completions do not balloon; otherwise roll back to the last promoted one.
"""
from __future__ import annotations

import json
import logging
import math
import os
import random
import re
import time
import urllib.error
import urllib.request
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable

import torch

from ..debug import require, stage
from .checkpoint import CheckpointManager, restore_trainable
from .loss import token_entropy, token_logprobs
from .optim import FusedStepOptimizer
from .sirsi import (
    TAG_AI_SLOP,
    TAG_LOOP_DEGENERATION,
    TAG_PREMATURE_EXIT,
    TAG_UNCLOSED_THINK,
    FailureLedger,
    SIRSIConfig,
    build_curriculum_sculpt_messages,
    build_hint_messages,
    detect_pathologies,
    map_char_span_to_token_mask,
    parse_and_decontaminate_sculpted_tasks,
    resolve_slap_target_params,
    unlikelihood_span_loss,
)
from .stack import Stack, build_stack
from .trainer import MetricsLogger

log = logging.getLogger("qloop")


# ============================================================================ tasks

@dataclass
class RLTask:
    task_id: str
    prompt: Any                 # str or chat messages
    verifier: str               # scorer:<name> | regex | http
    item: dict[str, Any] = field(default_factory=dict)
    source: str = ""


def _toy_arith_tasks(n: int, seed: int) -> list[dict[str, Any]]:
    r = random.Random(seed)
    out = []
    for i in range(n):
        a, b = r.randint(2, 99), r.randint(2, 99)
        op = r.choice(["+", "-", "*"])
        ans = a + b if op == "+" else a - b if op == "-" else a * b
        out.append({"id": f"toy_arith-{i}", "prompt": f"Compute {a} {op} {b}. Put the final answer in \\boxed{{}}.",
                    "answer": str(ans)})
    return out


def _hf_tasks(src: dict[str, Any], seed: int) -> list[dict[str, Any]]:
    from datasets import load_dataset

    ds = load_dataset(src["hf_id"], src.get("config"), split=src.get("split", "train"),
                      token=os.environ.get("HF_TOKEN"))
    if src.get("shuffle", True):
        ds = ds.shuffle(seed=seed)
    if src.get("max_rows"):
        ds = ds.select(range(min(len(ds), int(src["max_rows"]))))
    pf, af = src.get("prompt_field", "problem"), src.get("answer_field", "answer")
    rows = []
    for i, row in enumerate(ds):
        require(pf in row, f"rl source {src['hf_id']}: prompt_field '{pf}' missing", keys=sorted(row))
        rows.append({"id": str(row.get("id", f"{src['hf_id']}-{i}")), "prompt": row[pf],
                     "answer": row.get(af), **{k: row[k] for k in src.get("keep_fields", []) if k in row}})
    return rows


def _registry_tasks(src: dict[str, Any], seed: int) -> list[dict[str, Any]]:
    from ..data.formats import normalize_rl_task
    from ..data.mixture import raw_rows
    from ..data.registry import get_spec

    spec = get_spec(src["name"])
    out = []
    for i, row in enumerate(raw_rows(spec, seed, 0, 0, src)):
        if src.get("max_rows") and i >= int(src["max_rows"]):
            break
        t = normalize_rl_task(row, spec, i)
        item = {"id": t["task_id"], "prompt": t["prompt"], **t["meta"]}
        if spec.format == "rl_kernelbench":
            item["ref_src"] = t["meta"]["code"]
        out.append(item)
    return out


def load_rl_tasks(rl_cfg: dict[str, Any], seed: int) -> list[tuple[str, float, list[RLTask]]]:
    groups = []
    for k, src in enumerate(rl_cfg.get("tasks") or []):
        kind = src.get("source", "inline")
        if kind == "toy_arith":
            rows = _toy_arith_tasks(int(src.get("n", 256)), seed + k)
        elif kind == "inline":
            rows = list(src.get("items") or [])
        elif kind == "hf":
            rows = _hf_tasks(src, seed + k)
        elif kind == "registry":
            rows = _registry_tasks(src, seed + k)
        else:
            raise ValueError(f"unknown rl task source {kind!r} (toy_arith|inline|hf|registry)")
        verifier = src.get("verifier", "scorer:math")
        name = src.get("name", f"{kind}{k}")
        extra = {kk: src[kk] for kk in ("pattern", "url", "timeout", "speedup_bonus", "device") if kk in src}
        tasks = [RLTask(str(r.get("id", f"{name}-{i}")), r["prompt"], verifier, {**r, **extra}, name)
                 for i, r in enumerate(rows)]
        require(bool(tasks), f"rl task source '{name}' produced no tasks", src=src)
        groups.append((name, float(src.get("weight", 1.0)), tasks))
        log.info("[rl] task source %-14s %6d tasks verifier=%s weight=%.2f", name, len(tasks), verifier,
                 float(src.get("weight", 1.0)))
    require(bool(groups), "rl.tasks is empty")
    return groups


class TaskSampler:
    """Weighted over sources, without replacement inside a source per epoch; resumable."""

    def __init__(self, groups: list[tuple[str, float, list[RLTask]]], seed: int):
        self.groups, self.seed = groups, seed
        self.rng = random.Random(seed)
        self.cursor = {name: 0 for name, _, _ in groups}
        self.epoch = {name: 0 for name, _, _ in groups}
        self.order: dict[str, list[int]] = {}
        for name, _, tasks in groups:
            self._shuffle(name, len(tasks))

    def _shuffle(self, name: str, n: int) -> None:
        idx = list(range(n))
        random.Random(f"{self.seed}-{name}-{self.epoch[name]}").shuffle(idx)
        self.order[name] = idx

    def next(self, k: int) -> list[RLTask]:
        names = [g[0] for g in self.groups]
        weights = [g[1] for g in self.groups]
        out = []
        for _ in range(k):
            gi = self.rng.choices(range(len(names)), weights=weights)[0]
            name, _, tasks = self.groups[gi]
            if self.cursor[name] >= len(tasks):
                self.epoch[name] += 1
                self.cursor[name] = 0
                self._shuffle(name, len(tasks))
                log.info("[rl] task source %s: epoch %d", name, self.epoch[name])
            out.append(tasks[self.order[name][self.cursor[name]]])
            self.cursor[name] += 1
        return out

    def state_dict(self) -> dict[str, Any]:
        return {"rng": self.rng.getstate(), "cursor": dict(self.cursor), "epoch": dict(self.epoch)}

    def load_state_dict(self, sd: dict[str, Any]) -> None:
        st = sd["rng"]
        self.rng.setstate((st[0], tuple(st[1]), st[2]) if isinstance(st, list) else st)
        self.cursor.update(sd["cursor"])
        for name, ep in sd["epoch"].items():
            self.epoch[name] = int(ep)
            n = next(len(t) for nm, _, t in self.groups if nm == name)
            self._shuffle(name, n)


# ============================================================================ verifiers

def _post_json(url: str, payload: dict[str, Any], timeout: float, headers: dict[str, str] | None = None,
               retries: int = 3) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    err: Exception | None = None
    for attempt in range(retries):
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json", **(headers or {})})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", "replace")[:500]
            err = RuntimeError(f"HTTP {e.code} from {url}: {body}")
            if e.code < 500 and e.code != 429:
                break
        except Exception as e:
            err = e
        time.sleep(min(2 ** attempt, 8))
    raise RuntimeError(f"POST {url} failed after {retries} attempts: {err!r}")


def verify(task: RLTask, text: str) -> tuple[float, dict[str, Any]]:
    v = task.verifier
    if v.startswith("scorer:"):
        from ..eval.scoring import get_scorer

        ev = get_scorer(v.split(":", 1)[1]).evaluate(text, task.item)
        reward = float(ev.get("score", 0.0))
        bonus = float(task.item.get("speedup_bonus", 0.0))
        if bonus and ev.get("correct") and ev.get("speedup"):
            reward += bonus * max(0.0, min(1.0, math.log2(max(float(ev["speedup"]), 1e-9))))
        return reward, {k: ev.get(k) for k in ("pred", "compiled", "correct", "speedup", "error") if k in ev}
    if v == "regex":
        pat = task.item.get("pattern")
        require(bool(pat), "regex verifier needs `pattern`", task=task.task_id)
        return float(re.search(pat, text) is not None), {}
    if v == "http":
        url = task.item.get("url")
        require(bool(url), "http verifier needs `url`", task=task.task_id)
        res = _post_json(url, {"task_id": task.task_id, "completion": text, "item": task.item},
                         float(task.item.get("timeout", 600)))
        return float(res.get("reward", 0.0)), {k: v2 for k, v2 in res.items() if k != "reward"}
    raise ValueError(f"unknown verifier {v!r} (scorer:<name> | regex | http)")


# ============================================================================ teacher

class TeacherClient:
    """OpenAI-compatible chat endpoint (frontier API or a local vLLM teacher)."""

    def __init__(self, cfg: dict[str, Any]):
        self.url = cfg["base_url"].rstrip("/") + "/chat/completions"
        self.model = cfg["model"]
        key = os.environ.get(cfg.get("api_key_env", "TEACHER_API_KEY"), "")
        self.headers = {"Authorization": f"Bearer {key}"} if key else {}
        self.max_tokens = int(cfg.get("max_tokens", 8192))
        self.temperature = float(cfg.get("temperature", 0.6))
        self.timeout = float(cfg.get("timeout", 600))
        self.system = cfg.get("system")
        self.max_total_calls: int | None = (
            int(cfg["max_total_calls"]) if cfg.get("max_total_calls") is not None else None
        )
        self.max_total_tokens: int | None = (
            int(cfg["max_total_tokens"]) if cfg.get("max_total_tokens") is not None else None
        )
        self.calls = self.tokens = 0

    def __repr__(self) -> str:
        return f"TeacherClient(url={self.url!r}, model={self.model!r}, calls={self.calls}, tokens={self.tokens})"

    def _chat(
        self,
        msgs: list[dict[str, Any]],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
        include_reasoning: bool = True,
    ) -> str:
        if self.max_total_calls is not None and self.calls >= self.max_total_calls:
            raise RuntimeError(f"teacher call budget reached ({self.calls}/{self.max_total_calls})")
        if self.max_total_tokens is not None and self.tokens >= self.max_total_tokens:
            raise RuntimeError(f"teacher token budget reached ({self.tokens}/{self.max_total_tokens})")
        payload = {
            "model": self.model,
            "messages": msgs,
            "max_tokens": self.max_tokens if max_tokens is None else int(max_tokens),
            "temperature": self.temperature if temperature is None else float(temperature),
        }
        res = _post_json(self.url, payload, self.timeout, self.headers)
        self.calls += 1
        self.tokens += int((res.get("usage") or {}).get("total_tokens", 0))
        msg = res["choices"][0]["message"]
        reasoning = msg.get("reasoning_content") or msg.get("reasoning") or ""
        content = msg.get("content") or ""
        if include_reasoning and reasoning:
            return f"<think>\n{reasoning}\n</think>\n\n{content}"
        return content

    def solve(self, prompt: Any) -> str:
        msgs = list(prompt) if isinstance(prompt, list) else [{"role": "user", "content": str(prompt)}]
        if self.system:
            msgs = [{"role": "system", "content": self.system}, *msgs]
        return self._chat(msgs, include_reasoning=True)

    def hint(self, prompt: Any, bad_completion: str, tags: Sequence[str] = ()) -> str:
        """Request a concise 1-sentence supervisory nudge before divergence (SI-RSI Tier 2)."""
        msgs = build_hint_messages(prompt, bad_completion, tags)
        raw = self._chat(
            msgs,
            max_tokens=min(self.max_tokens, 256),
            temperature=min(self.temperature, 0.4),
            include_reasoning=False,
        )
        return raw.strip()

    def sculpt_curriculum(
        self,
        failure_records: Sequence[dict[str, Any]],
        n_tasks: int = 4,
        *,
        decontaminator: Any = None,
    ) -> list[dict[str, Any]]:
        """Synthesize targeted verifiable micro-tasks from failure records with decontamination."""
        msgs = build_curriculum_sculpt_messages(failure_records, n_tasks=n_tasks)
        raw = self._chat(
            msgs,
            max_tokens=min(self.max_tokens, 2048),
            temperature=self.temperature,
            include_reasoning=False,
        )
        return parse_and_decontaminate_sculpted_tasks(raw, decontaminator=decontaminator)


def _append_hint_to_prompt(prompt: Any, hint_text: str) -> Any:
    nudge = f"\n\n[Supervisor Hint]: {hint_text.strip()}\n"
    if isinstance(prompt, str):
        return prompt + nudge
    if isinstance(prompt, list):
        msgs = [dict(m) if isinstance(m, dict) else m for m in prompt]
        if msgs and isinstance(msgs[-1], dict) and msgs[-1].get("role") == "user" and isinstance(
            msgs[-1].get("content"), str
        ):
            msgs[-1]["content"] = msgs[-1]["content"] + nudge
        else:
            msgs.append({"role": "user", "content": f"[Supervisor Hint]: {hint_text.strip()}"})
        return msgs
    return str(prompt) + nudge


# ============================================================================ trainer

def group_advantages(rewards: list[float], norm: str = "std", eps: float = 1e-4) -> list[float] | None:
    n = len(rewards)
    mean = sum(rewards) / n
    var = sum((r - mean) ** 2 for r in rewards) / max(n - 1, 1)
    if var <= 1e-12:
        return None
    std = math.sqrt(var)
    return [(r - mean) / (std + eps) if norm == "std" else (r - mean) for r in rewards]


@dataclass
class _Seq:
    ids: list[int]
    n_prompt: int
    adv: float
    kind: str  # "pg" | "teacher" | "slap_neg"
    span_mask: list[bool] | None = None
    tags: tuple[str, ...] = ()


class GRPOTrainer:
    def __init__(self, cfg: dict[str, Any], run_dir: str, stack: Stack | None = None, hub_api: Any = None):
        from ..eval.generate import Generator

        self.cfg, self.run_dir = cfg, run_dir
        self.rl = dict(cfg.get("rl") or {})
        self.tcfg = dict(cfg.get("train") or {})
        sync_cfg = (cfg.get("checkpoint") or {}).get("sync")
        if stack is None and sync_cfg:
            from ..hub import HubSyncManager

            hub = HubSyncManager.from_sync_cfg(sync_cfg, api=hub_api)
            if hub is not None:
                stage_id = str(cfg.get("stage") or os.path.basename(os.path.abspath(run_dir)))
                run_root = os.path.dirname(os.path.abspath(run_dir))
                hub.prepare_stage_workspace(stage_id, dict(cfg), run_root)
        self.stack = stack or build_stack(cfg)
        require(bool(self.stack.named_trainable), "nothing is trainable")
        self.seed = int(cfg.get("seed", 0))
        self.rng = random.Random(self.seed)
        ocfg = dict(self.tcfg.get("optim") or {})
        ocfg.setdefault("total_steps", int(self.tcfg.get("steps", 200)))
        self.opt = FusedStepOptimizer.from_config(list(self.stack.named_trainable.items()), ocfg)
        self.ckpt = CheckpointManager(run_dir, keep_last=int(self.tcfg.get("keep_last", 3)),
                                      sync=sync_cfg, hub_api=hub_api)
        self.metrics = MetricsLogger(run_dir, cfg)
        self.gen = Generator(self.stack.model, self.stack.tokenizer, loop_controller=self.stack.loop,
                             max_context=self.rl.get("max_context"))
        self.sampler = TaskSampler(load_rl_tasks(self.rl, self.seed), self.seed)
        self.teacher = TeacherClient(self.rl["teacher"]) if (self.rl.get("teacher") or {}).get("base_url") else None
        self.sirsi = SIRSIConfig.from_dict(self.rl.get("sirsi"))
        self.failure_ledger: FailureLedger | None = (
            FailureLedger(os.path.join(run_dir, self.sirsi.failure_ledger_filename))
            if (self.sirsi.enabled and self.sirsi.failure_ledger)
            else None
        )
        self._slap_target_params = (
            resolve_slap_target_params(self.stack.named_trainable, self.sirsi.slap_target_groups)
            if (self.sirsi.enabled and self.sirsi.slap_target_groups)
            else None
        )
        if self._slap_target_params is not None and not self._slap_target_params:
            log.warning(
                "[sirsi] slap_target_groups=%r matched 0 trainable params; slap_neg grads will be skipped",
                self.sirsi.slap_target_groups,
            )
        self.temperature = float(self.rl.get("temperature", 1.0))
        self.init_entropy: float | None = None
        self.ref: dict[str, torch.Tensor] | None = None
        if float(self.rl.get("kl_coef", 0.0)) > 0:
            self.ref = {n: p.detach().clone() for n, p in self.stack.named_trainable.items()}
        self.best_gate: float | None = None
        self.best_len: float | None = None
        self.step = 0
        self.pool = ThreadPoolExecutor(max_workers=int(self.rl.get("verifier_workers", 8)))

    # ------------------------------------------------------------------ helpers
    def _eos(self) -> int | None:
        e = sorted(self.gen.eos_token_ids)
        return e[0] if e else None

    def _batch(self, seqs: list[_Seq]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Right-padded (safe for causal attention + recurrent DeltaNet) ids, loss mask, advantages."""
        t = max(len(s.ids) for s in seqs)
        pad = self.gen._pad_id()
        ids = torch.full((len(seqs), t), pad, dtype=torch.long)
        mask = torch.zeros(len(seqs), t, dtype=torch.float32)
        adv = torch.zeros(len(seqs), dtype=torch.float32)
        for i, s in enumerate(seqs):
            ids[i, :len(s.ids)] = torch.tensor(s.ids, dtype=torch.long)
            n_comp = len(s.ids) - s.n_prompt
            if s.span_mask is not None:
                require(
                    len(s.span_mask) == n_comp,
                    "span_mask length must match completion length",
                    span_mask=len(s.span_mask),
                    n_comp=n_comp,
                )
                if n_comp > 0:
                    mask[i, s.n_prompt:len(s.ids)] = torch.tensor(
                        [1.0 if b else 0.0 for b in s.span_mask], dtype=torch.float32
                    )
            else:
                mask[i, s.n_prompt:len(s.ids)] = 1.0
            adv[i] = s.adv
        dev = self.stack.device
        return ids.to(dev), mask.to(dev), adv.to(dev)

    def _logprobs(self, ids: torch.Tensor) -> torch.Tensor:
        h = self.stack.hidden(ids)
        return token_logprobs(h[:, :-1], self.stack.lm_head_weight, ids[:, 1:], temperature=self.temperature)

    @torch.no_grad()
    def _ref_logprobs(self, ids: torch.Tensor) -> torch.Tensor:
        assert self.ref is not None
        saved = {}
        for n, p in self.stack.named_trainable.items():
            saved[n] = p.data
            p.data = self.ref[n]
        try:
            return self._logprobs(ids)
        finally:
            for n, p in self.stack.named_trainable.items():
                p.data = saved[n]

    def _overlong(self, n_tokens: int) -> float:
        oc = self.rl.get("overlong") or {}
        if not oc.get("enabled"):
            return 0.0
        max_new = int(self.rl.get("max_new_tokens", 1024))
        buf = int(oc.get("buffer", max_new // 8))
        start = max_new - buf
        return -float(oc.get("penalty", 1.0)) * max(0.0, (n_tokens - start) / max(buf, 1))

    # ------------------------------------------------------------------ one RL step
    def rollout(self) -> tuple[list[_Seq], dict[str, float]]:
        rl = self.rl
        sirsi = self.sirsi
        g = int(rl.get("group_size", 8))
        tasks = self.sampler.next(int(rl.get("prompts_per_step", 4)))
        with stage("rl:rollout"):
            outs = self.gen.generate([t.prompt for t in tasks], max_new_tokens=int(rl.get("max_new_tokens", 1024)),
                                     temperature=self.temperature, top_p=float(rl.get("top_p", 1.0)), n=g,
                                     seed=self.seed * 100_003 + self.step, stop=rl.get("stop"))
        with stage("rl:verify"):
            futs = [[self.pool.submit(verify, t, o.text) for o in samples] for t, samples in zip(tasks, outs)]
            rewards = [[f.result()[0] for f in row] for row in futs]
        seqs: list[_Seq] = []
        slap_seqs: list[_Seq] = []
        stats = {"reward/mean": 0.0, "groups/all_fail": 0.0, "groups/all_pass": 0.0, "groups/used": 0.0}
        lens, flat, teacher_ok, teacher_calls = [], [], 0, 0
        hint_calls = hint_ok = failures_logged = 0
        n_total = n_loop = n_unclosed = n_premature = n_slop = 0
        slop_pen_sum = 0.0
        max_teacher = int((rl.get("teacher") or {}).get("max_calls_per_step", 2))
        for t, samples, rw in zip(tasks, outs, rewards):
            p_ids = self.gen.encode_prompt(t.prompt)
            all_fail = all(r <= 0 for r in rw)
            paths = (
                [detect_pathologies(o.text, reward=r, cfg=sirsi) for o, r in zip(samples, rw)]
                if sirsi.enabled
                else []
            )
            shaped: list[float] = []
            for idx_s, (r, o) in enumerate(zip(rw, samples)):
                s_pen = 0.0
                if sirsi.enabled:
                    p_info = paths[idx_s]
                    n_total += 1
                    if TAG_LOOP_DEGENERATION in p_info.tags:
                        n_loop += 1
                    if TAG_UNCLOSED_THINK in p_info.tags:
                        n_unclosed += 1
                    if TAG_PREMATURE_EXIT in p_info.tags:
                        n_premature += 1
                    if TAG_AI_SLOP in p_info.tags:
                        n_slop += 1
                    if sirsi.anti_slop_weight > 0.0:
                        s_pen = p_info.slop_penalty
                        slop_pen_sum += s_pen
                    if (
                        sirsi.slap_weight > 0.0
                        and len(slap_seqs) < sirsi.slap_max_seqs_per_step
                        and o.token_ids
                        and r <= 0.0
                        and (TAG_LOOP_DEGENERATION in p_info.tags or TAG_UNCLOSED_THINK in p_info.tags)
                    ):
                        smask = map_char_span_to_token_mask(
                            self.stack.tokenizer,
                            list(o.token_ids),
                            o.text,
                            p_info.loop_onset_char,
                            fallback_tail_frac=sirsi.slap_fallback_tail_frac,
                        )
                        if any(smask):
                            slap_seqs.append(
                                _Seq(
                                    p_ids + list(o.token_ids),
                                    len(p_ids),
                                    -1.0,
                                    "slap_neg",
                                    span_mask=smask,
                                    tags=tuple(p_info.tags),
                                )
                            )
                # Apply anti-slop tie-breaker only when at least one sample succeeded (`not all_fail`),
                # preventing std-normalized advantage amplification from positively reinforcing wrong answers.
                shaped.append(r + self._overlong(o.completion_tokens) - (0.0 if all_fail else s_pen))

            flat += rw
            lens += [o.completion_tokens for o in samples]

            if self.failure_ledger is not None:
                logged_any = False
                for o, r, p_info in zip(samples, rw, paths):
                    if p_info.tags:
                        self.failure_ledger.record(
                            step=self.step,
                            task_id=t.task_id,
                            tags=p_info.tags,
                            reward=r,
                            loop_onset_char=p_info.loop_onset_char,
                            prompt=t.prompt,
                            bad_completion=o.text,
                        )
                        failures_logged += 1
                        logged_any = True
                if all_fail and not logged_any and samples:
                    self.failure_ledger.record(
                        step=self.step,
                        task_id=t.task_id,
                        tags=["ALL_FAIL"],
                        reward=rw[0],
                        loop_onset_char=None,
                        prompt=t.prompt,
                        bad_completion=samples[0].text,
                    )
                    failures_logged += 1

            if all_fail:
                stats["groups/all_fail"] += 1
                recovered_by_hint = False
                if (
                    sirsi.enabled
                    and sirsi.guided_hint_enabled
                    and self.teacher is not None
                    and hint_calls < sirsi.max_hints_per_step
                    and samples
                ):
                    bad_idx = next((i for i, pi in enumerate(paths) if pi.tags), 0)
                    bad_o = samples[bad_idx]
                    bad_tags = paths[bad_idx].tags if paths else ()
                    hint_calls += 1
                    try:
                        hint_text = self.teacher.hint(t.prompt, bad_o.text, bad_tags)
                    except Exception as e:
                        log.warning("[sirsi] supervisor hint failed on %s: %r", t.task_id, e)
                        hint_text = ""
                    if hint_text:
                        guided_prompt = _append_hint_to_prompt(t.prompt, hint_text)
                        try:
                            retry_outs = self.gen.generate(
                                [guided_prompt],
                                max_new_tokens=int(rl.get("max_new_tokens", 1024)),
                                temperature=self.temperature,
                                top_p=float(rl.get("top_p", 1.0)),
                                n=sirsi.guided_retry_n,
                                seed=self.seed * 100_003 + self.step * 31 + hint_calls,
                                stop=rl.get("stop"),
                            )[0]
                        except Exception as e:
                            log.warning("[sirsi] guided retry failed on %s: %r", t.task_id, e)
                            retry_outs = []
                        for ro in retry_outs:
                            r_retry, _ = verify(t, ro.text)
                            if r_retry > 0 and ro.token_ids:
                                hint_ok += 1
                                recovered_by_hint = True
                                seqs.append(
                                    _Seq(p_ids + list(ro.token_ids), len(p_ids), 1.0, "teacher", tags=("GUIDED_HINT",))
                                )
                                break
                if not recovered_by_hint and self.teacher is not None and teacher_calls < max_teacher:
                    teacher_calls += 1
                    try:
                        text = self.teacher.solve(t.prompt)
                        r_t, _ = verify(t, text)
                    except Exception as e:
                        log.warning("[rl] teacher failed on %s: %r", t.task_id, e)
                        r_t, text = 0.0, ""
                    if r_t > 0:
                        teacher_ok += 1
                        c_ids = self.gen.encode_text(text) + ([self._eos()] if self._eos() is not None else [])
                        seqs.append(_Seq(p_ids + c_ids, len(p_ids), 1.0, "teacher"))
            if all(r >= 1 for r in rw):
                stats["groups/all_pass"] += 1
            adv = group_advantages(shaped, rl.get("adv_norm", "std"))
            if adv is None:
                continue
            stats["groups/used"] += 1
            for o, a in zip(samples, adv):
                if o.token_ids:
                    seqs.append(_Seq(p_ids + list(o.token_ids), len(p_ids), float(a), "pg"))
        seqs.extend(slap_seqs)
        n = max(len(tasks), 1)
        stats["reward/mean"] = sum(flat) / max(len(flat), 1)
        stats["reward/pass_rate"] = sum(r >= 1 for r in flat) / max(len(flat), 1)
        for k in ("groups/all_fail", "groups/all_pass", "groups/used"):
            stats[k] /= n
        stats["completion/mean_tokens"] = sum(lens) / max(len(lens), 1)
        stats["teacher/calls"], stats["teacher/verified"] = float(teacher_calls), float(teacher_ok)
        if sirsi.enabled:
            denom = float(max(n_total, 1))
            stats["sirsi/loop_degen_rate"] = n_loop / denom
            stats["sirsi/unclosed_think_rate"] = n_unclosed / denom
            stats["sirsi/premature_exit_rate"] = n_premature / denom
            stats["sirsi/ai_slop_rate"] = n_slop / denom
            stats["sirsi/mean_slop_penalty"] = slop_pen_sum / denom
            stats["sirsi/n_slap_seqs"] = float(len(slap_seqs))
            stats["sirsi/hint_calls"] = float(hint_calls)
            stats["sirsi/hint_recovered"] = float(hint_ok)
            stats["sirsi/failures_logged"] = float(failures_logged)
        return seqs, stats

    def update(self, seqs: list[_Seq]) -> dict[str, float]:
        rl = self.rl
        mb = int(rl.get("micro_batch_seqs", 4))
        eps_lo, eps_hi = float(rl.get("clip_low", 0.2)), float(rl.get("clip_high", 0.28))
        kl_coef, sft_w = float(rl.get("kl_coef", 0.0)), float((rl.get("teacher") or {}).get("sft_weight", 0.1))
        slap_w = float(self.sirsi.slap_weight) if self.sirsi.enabled else 0.0
        slap_eps = float(self.sirsi.slap_eps)
        aux_w = dict(self.tcfg.get("aux_weights") or {})
        epochs = int(rl.get("ppo_epochs", 1))
        slap = [s for s in seqs if s.kind == "slap_neg"]
        pg = [s for s in seqs if s.kind == "pg"]
        te = [s for s in seqs if s.kind == "teacher"]
        # Place slap_neg chunks before pg/teacher so surgical autograd.grad accumulates cleanly before
        # any boundary fused-hook backward; works identically in both fused=True and fused=False modes.
        chunks = (
            [slap[i:i + mb] for i in range(0, len(slap), mb)]
            + [pg[i:i + mb] for i in range(0, len(pg), mb)]
            + [te[i:i + mb] for i in range(0, len(te), mb)]
        )
        n_slap_tok = max(
            1,
            sum(
                sum(1 for b in (s.span_mask[1:] if s.n_prompt == 0 else s.span_mask) if b)
                if s.span_mask is not None
                else max(0, len(s.ids) - max(1, s.n_prompt))
                for s in slap
            ),
        )
        n_pg_tok = max(1, sum(len(s.ids) - s.n_prompt for s in pg))
        n_te_tok = max(1, sum(len(s.ids) - s.n_prompt for s in te))
        old: list[torch.Tensor | None] = [None] * len(chunks)
        if epochs > 1:
            with torch.no_grad():
                old = [
                    self._logprobs(self._batch(c)[0]).detach() if c[0].kind == "pg" else None
                    for c in chunks
                ]
        # Drop any stale no-grad aux terms left behind by `rollout()` or `old` logprob evaluation
        self.stack.aux_losses()
        stats = {
            "loss/pg": 0.0,
            "loss/kl": 0.0,
            "loss/teacher_nll": 0.0,
            "loss/teacher": 0.0,
            "loss/slap_neg": 0.0,
            "clip_frac": 0.0,
        }
        if not chunks:
            stats["temperature"] = self.temperature
            return stats
        clip_n = tok_n = 0.0
        for _ep in range(epochs):
            self.opt.zero_grad(set_to_none=True)
            for ci, chunk in enumerate(chunks):
                self.opt.set_boundary(ci == len(chunks) - 1)
                ids, mask, adv = self._batch(chunk)
                m = mask[:, 1:]
                lp = self._logprobs(ids)
                aux_now = self.stack.aux_losses()
                kind = chunk[0].kind
                if kind == "slap_neg":
                    span_bool = m > 0.5
                    chunk_toks = span_bool.sum()
                    ul_mean = unlikelihood_span_loss(lp, span_bool, eps=slap_eps)
                    loss = ul_mean * (chunk_toks.to(ul_mean.dtype) / float(n_slap_tok)) * slap_w
                    stats["loss/slap_neg"] += float(loss.detach())
                    if self._slap_target_params is not None:
                        if self._slap_target_params and loss.requires_grad:
                            grads = torch.autograd.grad(loss, self._slap_target_params, allow_unused=True)
                            with torch.no_grad():
                                for p, g_t in zip(self._slap_target_params, grads):
                                    if g_t is None:
                                        continue
                                    g_det = g_t.detach()
                                    if p.grad is None:
                                        p.grad = g_det.clone()
                                    else:
                                        p.grad.add_(g_det)
                            del grads
                        del aux_now, lp, loss, ul_mean, span_bool, chunk_toks, ids, mask, m, adv
                        continue
                elif kind == "teacher":
                    loss = -(lp * m).sum() / n_te_tok * sft_w
                    t_val = float(loss.detach())
                    stats["loss/teacher_nll"] += t_val
                    stats["loss/teacher"] += t_val
                else:
                    ref_lp = old[ci] if old[ci] is not None else lp.detach()
                    ratio = torch.exp(lp - ref_lp)
                    a = adv[:, None]
                    surr = torch.minimum(ratio * a, torch.clamp(ratio, 1 - eps_lo, 1 + eps_hi) * a)
                    loss = -(surr * m).sum() / n_pg_tok
                    clip_n += float((((ratio < 1 - eps_lo) | (ratio > 1 + eps_hi)).float() * m).sum())
                    tok_n += float(m.sum())
                    stats["loss/pg"] += float(loss.detach())
                    if kl_coef > 0 and self.ref is not None:
                        rlp = self._ref_logprobs(ids)
                        self.stack.aux_losses()
                        d = rlp - lp
                        kl = ((torch.exp(d) - d - 1) * m).sum() / n_pg_tok
                        loss = loss + kl_coef * kl
                        stats["loss/kl"] += float(kl.detach())
                for a_name, a_loss in aux_now.items():
                    loss = loss + float(aux_w.get(a_name, 1.0)) * a_loss
                    stats[f"loss/{a_name}"] = stats.get(f"loss/{a_name}", 0.0) + float(a_loss.detach())
                loss.backward()
            stats.update({f"optim/{k}": v for k, v in self.opt.step().items()})
            self.opt.zero_grad(set_to_none=True)
        stats["clip_frac"] = clip_n / max(tok_n, 1.0)
        if pg:
            with torch.no_grad():
                ids, mask, _ = self._batch(pg[:mb])
                h = self.stack.hidden(ids)
                ent = token_entropy(h[:, :-1], self.stack.lm_head_weight, temperature=self.temperature)
                stats["entropy"] = float((ent * mask[:, 1:]).sum() / mask[:, 1:].sum().clamp_min(1))
                self.stack.aux_losses()
            self._adapt_temperature(stats["entropy"])
        stats["temperature"] = self.temperature
        return stats

    def _adapt_temperature(self, ent: float) -> None:
        if self.init_entropy is None:
            self.init_entropy = ent
            return
        floor = float(self.rl.get("entropy_floor_frac", 0.6)) * self.init_entropy
        if ent < floor:
            new = min(float(self.rl.get("temperature_max", 1.3)), self.temperature * 1.05)
            if new != self.temperature:
                log.warning("[rl] entropy %.3f < %.0f%% of initial (%.3f): temperature %.3f -> %.3f", ent,
                            100 * float(self.rl.get("entropy_floor_frac", 0.6)), self.init_entropy,
                            self.temperature, new)
            self.temperature = new

    # ------------------------------------------------------------------ regression gate
    def gate(self) -> dict[str, Any]:
        from ..eval.runners import run_benchmark

        gcfg = dict(self.rl.get("gate") or {})
        depth = gcfg.get("depth") or (self.stack.loop.spec.num_passes if self.stack.loop else None)
        scores, lens = [], []
        res = {}
        for entry in gcfg.get("benchmarks") or []:
            b = {"name": entry} if isinstance(entry, str) else dict(entry)
            out_dir = os.path.join(self.run_dir, "gate", f"step_{self.step:07d}")
            summ = run_benchmark(b["name"], self.gen, limit=b.get("limit"), depths=[depth],
                                 n_samples=int(b.get("n_samples", 1)), max_new_tokens=b.get("max_new_tokens"),
                                 out_dir=out_dir, offline_items=b.get("offline_items"), seed=int(b.get("seed", 0)))
            d = next(iter(summ["depths"].values()))
            scores.append(float(d.get("mean_score", 0.0)) * float(b.get("weight", 1.0)))
            lens.append(float(d.get("mean_completion_tokens", 0.0)))
            res[b["name"]] = d.get("mean_score")
        if not scores:
            return {}
        score, mean_len = sum(scores) / len(scores), sum(lens) / len(lens)
        gate_metrics = {"gate/score": score, "gate/mean_len": mean_len, **{f"gate/{k}": v for k, v in res.items()}}
        path = self._save_ckpt(metrics=gate_metrics)
        ok_len = self.best_len is None or mean_len <= float(gcfg.get("max_len_growth", 1.5)) * self.best_len
        improved = self.best_gate is None or score >= self.best_gate + float(gcfg.get("min_delta", 0.0))
        if improved and ok_len:
            self.best_gate, self.best_len = score, mean_len
            self.ckpt.promote(path, metrics=gate_metrics)
            decision = "promoted"
        else:
            prom = self.ckpt.promoted()
            if prom and gcfg.get("rollback", True):
                ck = self.ckpt.load(prom)
                assert ck is not None
                restore_trainable(self.stack.named_trainable, ck["params"], strict=False)
                if ck["optim"] is not None:
                    self.opt.load_state_dict(ck["optim"])
                decision = f"rolled back to {os.path.basename(prom)}"
            else:
                decision = "rejected (no rollback)"
        log.info("[rl] gate step %d: score %.4f (best %s) len %.0f -> %s", self.step, score, self.best_gate,
                 mean_len, decision)
        return {**gate_metrics, "gate/decision": decision}

    # ------------------------------------------------------------------ main loop
    def _state(self) -> dict[str, Any]:
        return {"stage": self.cfg.get("stage"), "sampler": self.sampler.state_dict(), "temperature": self.temperature,
                "init_entropy": self.init_entropy, "best_gate": self.best_gate, "best_len": self.best_len,
                "stack": self.stack.describe()}

    def _save_ckpt(self, *, completed: bool = False, metrics: dict[str, Any] | None = None) -> str:
        raw_cfg = self.cfg.to_dict() if hasattr(self.cfg, "to_dict") else dict(self.cfg)
        return self.ckpt.save(
            self.step,
            self.stack.checkpoint_params(),
            self.opt.state_dict(),
            self._state(),
            completed=completed,
            target_steps=int(self.tcfg.get("steps", 200)),
            metrics=metrics,
            config_dict=raw_cfg,
        )

    def _resume(self) -> None:
        from ..hub import resolve_run_path

        run_root = os.path.dirname(os.path.abspath(self.run_dir))
        if not (self.tcfg.get("resume", True) and self.ckpt.latest()):
            init = resolve_run_path(self.tcfg.get("init_from"), run_root=run_root)
            if init:
                ck = CheckpointManager(init, keep_last=0).load() if os.path.isdir(init) else None
                require(ck is not None, "train.init_from: no checkpoint", init_from=init)
                assert ck is not None
                smeta = (ck.get("state") or {}).get("stack") or {}
                sc = float((smeta.get("expert_delta_scale")) or 1.0)
                self.stack.restore_checkpoint(
                    ck["params"], strict=False, expert_delta_scale=sc, prior_stack_meta=smeta
                )
                if self.ref is not None:
                    self.ref = {n: p.detach().clone() for n, p in self.stack.named_trainable.items()}
            return
        ck = self.ckpt.load()
        assert ck is not None
        st = ck["state"]
        smeta = st.get("stack") or {}
        sc = float((smeta.get("expert_delta_scale")) or 1.0)
        self.stack.restore_checkpoint(ck["params"], strict=True, expert_delta_scale=sc, prior_stack_meta=smeta)
        if ck["optim"] is not None:
            self.opt.load_state_dict(ck["optim"])
        self.step = int(st["step"])
        self.sampler.load_state_dict(st["sampler"])
        self.temperature = float(st.get("temperature", self.temperature))
        self.init_entropy = st.get("init_entropy")
        self.best_gate, self.best_len = st.get("best_gate"), st.get("best_len")

    def train(self, on_step: Callable[[int, dict[str, Any]], None] | None = None) -> dict[str, Any]:
        steps = int(self.tcfg.get("steps", 200))
        ckpt_every = int(self.tcfg.get("ckpt_every", 50))
        gate_every = int((self.rl.get("gate") or {}).get("every", 0))
        budget = float(self.tcfg.get("time_budget_hours", 0)) * 3600
        self._resume()
        t0 = time.time()
        last: dict[str, Any] = {}
        try:
            while self.step < steps:
                depth = self.stack.loop.sample_active_passes(self.rng) if self.stack.loop else 1
                ts = time.time()
                seqs, rstats = self.rollout()
                t_roll = time.time() - ts
                ustats = self.update(seqs) if seqs else {"skipped_update": 1.0}
                self.step += 1
                last = {**rstats, **ustats, "depth": depth, "time/rollout_s": t_roll,
                        "time/step_s": time.time() - ts, "n_seqs": len(seqs), **self.stack.stats()}
                log.info("[rl] step %d/%d reward %.3f pass %.3f used %.2f len %.0f ent %s T %.2f (%.0fs)", self.step,
                         steps, last["reward/mean"], last["reward/pass_rate"], last["groups/used"],
                         last["completion/mean_tokens"], round(last.get("entropy", float("nan")), 3),
                         self.temperature, last["time/step_s"])
                self.metrics.log(self.step, last)
                if on_step:
                    on_step(self.step, last)
                if gate_every and self.step % gate_every == 0:
                    self.metrics.log(self.step, self.gate())
                elif self.step % ckpt_every == 0:
                    self._save_ckpt(metrics=last)
                if budget and time.time() - t0 > budget:
                    log.warning("[rl] time budget reached")
                    break
        except KeyboardInterrupt:
            self._save_ckpt(metrics=last)
            raise
        finally:
            self.pool.shutdown(wait=False)
        if self.stack.frozen_experts_sha256 is not None:
            self.stack.verify_frozen_experts()
        path = self._save_ckpt(completed=(self.step >= steps), metrics=last)
        return {"step": self.step, "checkpoint": path, "last": last}
