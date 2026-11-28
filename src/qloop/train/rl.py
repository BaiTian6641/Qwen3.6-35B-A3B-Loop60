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
from collections import deque
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable

import torch

from ..debug import require, stage
from .checkpoint import CheckpointManager
from .loss import token_entropy, token_logprobs
from .optim import FusedStepOptimizer
from .sirsi import (
    REGRET_NEWLY_HURT,
    REGRET_PINNED_PASS,
    REGRET_UNSOLVED_CHRONIC,
    REGRET_UNSOLVED_NOVEL,
    TAG_AI_SLOP,
    TAG_LOOP_DEGENERATION,
    TAG_PREMATURE_EXIT,
    TAG_UNCLOSED_THINK,
    FailureLedger,
    RegretDecision,
    SIRSIConfig,
    TaskRegretTracker,
    build_curriculum_sculpt_messages,
    build_hint_messages,
    detect_pathologies,
    evaluate_tier_a_checks,
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


def _extract_nested_field(row: Mapping[str, Any], field_path: str) -> Any:
    """Resolve a flat or dotted key path (e.g. `'reward_model.ground_truth'`) on a row mapping."""
    if field_path in row:
        return row[field_path]
    if "." in field_path:
        cur: Any = row
        for part in field_path.split("."):
            if isinstance(cur, Mapping) and part in cur:
                cur = cur[part]
            else:
                return None
        return cur
    return None


def _normalize_hf_prompt(raw: Any) -> Any:
    """Normalize HF prompt values while preserving chat message lists for `apply_chat_template`."""
    if hasattr(raw, "tolist") and not isinstance(raw, (str, bytes, list, tuple, Mapping)):
        raw = raw.tolist()
    if isinstance(raw, str):
        return raw
    if isinstance(raw, Mapping):
        msg = dict(raw)
        msg.setdefault("role", "user")
        if "content" not in msg and "text" in msg:
            msg["content"] = msg["text"]
        if msg.get("content") is None:
            msg["content"] = ""
        return [msg]
    if isinstance(raw, (list, tuple)):
        if not raw:
            return ""
        if all(isinstance(m, Mapping) for m in raw):
            msgs: list[dict[str, Any]] = []
            for m in raw:
                msg = dict(m)
                msg.setdefault("role", "user")
                if "content" not in msg and "text" in msg:
                    msg["content"] = msg["text"]
                if msg.get("content") is None:
                    msg["content"] = ""
                msgs.append(msg)
            return msgs
        return "\n".join(str(x) for x in raw)
    return str(raw)


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
        raw_prompt = _extract_nested_field(row, pf)
        require(raw_prompt is not None, f"rl source {src['hf_id']}: prompt_field '{pf}' missing", keys=sorted(row))
        prompt = _normalize_hf_prompt(raw_prompt)
        ans = _extract_nested_field(row, af)
        if ans is None:
            for fallback_key in ("reward_model.ground_truth", "solution", "answer", "ground_truth"):
                if fallback_key == af:
                    continue
                ans = _extract_nested_field(row, fallback_key)
                if ans is not None:
                    break
        if isinstance(ans, Mapping) and "ground_truth" in ans:
            ans = ans["ground_truth"]
        rows.append({"id": str(row.get("id", f"{src['hf_id']}-{i}")), "prompt": prompt,
                     "answer": ans, **{k: row[k] for k in src.get("keep_fields", []) if k in row}})
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
        self._by_id: dict[str, RLTask] = {}
        for name, _, tasks in groups:
            self._shuffle(name, len(tasks))
            for t in tasks:
                self._by_id[t.task_id] = t
        self.hurt_queue: deque[RLTask] = deque(maxlen=64)
        self.pinned_anchor_pool: list[RLTask] = []
        self._pinned_ids: set[str] = set()

    def _shuffle(self, name: str, n: int) -> None:
        idx = list(range(n))
        random.Random(f"{self.seed}-{name}-{self.epoch[name]}").shuffle(idx)
        self.order[name] = idx

    def record_hurt(self, task: RLTask) -> None:
        self._by_id.setdefault(task.task_id, task)
        self.hurt_queue = deque((t for t in self.hurt_queue if t.task_id != task.task_id), maxlen=64)
        self.hurt_queue.append(task)

    def record_pinned(self, task: RLTask) -> None:
        self._by_id.setdefault(task.task_id, task)
        if task.task_id in self._pinned_ids:
            return
        if len(self.pinned_anchor_pool) >= 256:
            evicted = self.pinned_anchor_pool.pop(0)
            self._pinned_ids.discard(evicted.task_id)
        self.pinned_anchor_pool.append(task)
        self._pinned_ids.add(task.task_id)

    def next(self, k: int, *, hurt_prob: float = 0.0, pinned_prob: float = 0.0) -> list[RLTask]:
        names = [g[0] for g in self.groups]
        weights = [g[1] for g in self.groups]
        hp = max(0.0, float(hurt_prob))
        pp = max(0.0, float(pinned_prob))
        use_replay = hp > 0.0 or pp > 0.0
        out = []
        for _ in range(k):
            if use_replay and (self.hurt_queue or self.pinned_anchor_pool):
                u = self.rng.random()
                if u < hp and self.hurt_queue:
                    out.append(self.hurt_queue.popleft())
                    continue
                if hp <= u < hp + pp and self.pinned_anchor_pool:
                    out.append(self.rng.choice(self.pinned_anchor_pool))
                    continue
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
        return {
            "rng": self.rng.getstate(),
            "cursor": dict(self.cursor),
            "epoch": dict(self.epoch),
            "hurt_queue": [t.task_id for t in self.hurt_queue],
            "pinned_anchor_pool": [t.task_id for t in self.pinned_anchor_pool],
        }

    def load_state_dict(self, sd: dict[str, Any]) -> None:
        st = sd["rng"]
        self.rng.setstate((st[0], tuple(st[1]), st[2]) if isinstance(st, list) else st)
        self.cursor.update(sd["cursor"])
        for name, ep in sd["epoch"].items():
            self.epoch[name] = int(ep)
            n = next(len(t) for nm, _, t in self.groups if nm == name)
            self._shuffle(name, n)
        self.hurt_queue.clear()
        for tid in sd.get("hurt_queue") or ():
            t = self._by_id.get(str(tid))
            if t is not None:
                self.hurt_queue.append(t)
        self.pinned_anchor_pool.clear()
        self._pinned_ids.clear()
        for tid in sd.get("pinned_anchor_pool") or ():
            t = self._by_id.get(str(tid))
            if t is not None and t.task_id not in self._pinned_ids:
                self.pinned_anchor_pool.append(t)
                self._pinned_ids.add(t.task_id)
                if len(self.pinned_anchor_pool) >= 256:
                    break


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

    def hint(
        self,
        prompt: Any,
        bad_completion: str,
        tags: Sequence[str] = (),
        *,
        regret_kind: str | None = None,
        prev_pass_rate: float | None = None,
    ) -> str:
        """Request a concise 1-sentence supervisory nudge before divergence (SI-RSI Tier 2)."""
        msgs = build_hint_messages(
            prompt,
            bad_completion,
            tags,
            regret_kind=regret_kind,
            prev_pass_rate=prev_pass_rate,
        )
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
    regret_weight: float = 1.0
    regret_kind: str = REGRET_UNSOLVED_NOVEL


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
        eth = self.rl.get("enable_thinking", None)
        chat_template_kwargs: dict[str, Any] | None = (
            {"enable_thinking": bool(eth)} if eth is not None else None
        )
        self.gen = Generator(self.stack.model, self.stack.tokenizer, loop_controller=self.stack.loop,
                             max_context=self.rl.get("max_context"),
                             chat_template_kwargs=chat_template_kwargs)
        self.sampler = TaskSampler(load_rl_tasks(self.rl, self.seed), self.seed)
        self.teacher = TeacherClient(self.rl["teacher"]) if (self.rl.get("teacher") or {}).get("base_url") else None
        self.sirsi = SIRSIConfig.from_dict(self.rl.get("sirsi"))
        self.regret_tracker = TaskRegretTracker(self.sirsi)
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
        self._logprob_chunk_size = max(1, int(self.rl.get("logprob_chunk_size", 512)))
        self.init_entropy: float | None = None
        self.ref: dict[str, torch.Tensor] | None = None
        if float(self.rl.get("kl_coef", 0.0)) > 0:
            self.ref = {n: p.detach().clone() for n, p in self.stack.named_trainable.items()}
        self.best_gate: float | None = None
        self.best_len: float | None = None
        self.best_gate_items: dict[str, float] = {}
        self.best_bench_scores: dict[str, float] = {}
        self.pinned_passed: set[str] = set()
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
        return token_logprobs(
            h[:, :-1],
            self.stack.lm_head_weight,
            ids[:, 1:],
            chunk_size=self._logprob_chunk_size,
            temperature=self.temperature,
        )

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

    def _overlong(self, n_tokens: int, *, pass_rate: float | None = None) -> float:
        oc = self.rl.get("overlong") or {}
        if not oc.get("enabled"):
            return 0.0
        max_new = int(self.rl.get("max_new_tokens", 1024))
        buf = int(oc.get("buffer", max_new // 8))
        start = max_new - buf
        raw_pen = -float(oc.get("penalty", 1.0)) * max(0.0, (n_tokens - start) / max(buf, 1))
        if raw_pen == 0.0:
            return 0.0
        if oc.get("accuracy_gated", False) and pass_rate is not None:
            gate = max(0.0, min(1.0, float(pass_rate)))
            return 0.0 if gate == 0.0 else raw_pen * gate
        return raw_pen

    def _concise_bonus(self, n_tokens: int, *, reward: float, pass_rate: float) -> float:
        oc = self.rl.get("overlong") or {}
        if oc.get("enabled") is False:
            return 0.0
        bonus = float(oc.get("concise_bonus", 0.0))
        if bonus <= 0.0 or reward <= 0.0:
            return 0.0
        min_pr = float(oc.get("min_pass_rate", 0.25))
        if pass_rate < min_pr:
            return 0.0
        target = oc.get("target_tokens")
        if target is None:
            target = int(self.rl.get("max_new_tokens", 1024))
        target_f = float(target)
        if target_f <= 0.0:
            return 0.0
        return bonus * max(0.0, min(1.0, 1.0 - float(n_tokens) / target_f))

    # ------------------------------------------------------------------ one RL step
    def rollout(self) -> tuple[list[_Seq], dict[str, float]]:
        rl = self.rl
        sirsi = self.sirsi
        g = int(rl.get("group_size", 8))
        hurt_p = sirsi.hurt_replay_prob if (sirsi.enabled and sirsi.regret_enabled) else 0.0
        pinned_p = sirsi.pinned_replay_prob if (sirsi.enabled and sirsi.regret_enabled) else 0.0
        tasks = self.sampler.next(
            int(rl.get("prompts_per_step", 4)),
            hurt_prob=hurt_p,
            pinned_prob=pinned_p,
        )
        with stage("rl:rollout"):
            outs = self.gen.generate([t.prompt for t in tasks], max_new_tokens=int(rl.get("max_new_tokens", 1024)),
                                     temperature=self.temperature, top_p=float(rl.get("top_p", 1.0)), n=g,
                                     seed=self.seed * 100_003 + self.step, stop=rl.get("stop"))
        with stage("rl:verify"):
            futs = [[self.pool.submit(verify, t, o.text) for o in samples] for t, samples in zip(tasks, outs)]
            raw_results = [[f.result() for f in row] for row in futs]
            rewards = [[float(r) for r, _ in row] for row in raw_results]
            metas = [[m if isinstance(m, dict) else {} for _, m in row] for row in raw_results]
        seqs: list[_Seq] = []
        slap_candidates: list[_Seq] = []
        all_fail_entries: list[tuple[RLTask, list[int], list[Any], list[Any], RegretDecision]] = []
        stats = {"reward/mean": 0.0, "groups/all_fail": 0.0, "groups/all_pass": 0.0, "groups/used": 0.0}
        lens, flat, teacher_ok, teacher_calls = [], [], 0, 0
        group_pass_rates: list[float] = []
        overlong_vals: list[float] = []
        hint_calls = hint_ok = failures_logged = 0
        n_total = n_loop = n_unclosed = n_premature = n_slop = 0
        n_regret_hurt = n_regret_chronic = n_regret_pinned = 0
        slop_pen_sum = 0.0
        max_teacher = int((rl.get("teacher") or {}).get("max_calls_per_step", 2))
        for t, samples, rw, ms in zip(tasks, outs, rewards, metas):
            p_ids = self.gen.encode_prompt(t.prompt)
            all_fail = all(r <= 0 for r in rw)
            all_fail_oracle = all(r < 1.0 for r in rw)
            pass_rate = sum(1 for r in rw if r > 0.0) / max(len(rw), 1)
            group_pass_rates.append(pass_rate)
            paths = (
                [detect_pathologies(o.text, reward=r, cfg=sirsi) for o, r in zip(samples, rw)]
                if sirsi.enabled
                else []
            )
            checks = (
                [
                    evaluate_tier_a_checks(
                        t.verifier,
                        t.prompt,
                        o.text,
                        r,
                        m,
                        p_info,
                        alpha_check=sirsi.atomic_check_alpha,
                    )
                    for o, r, m, p_info in zip(samples, rw, ms, paths)
                ]
                if (sirsi.enabled and sirsi.atomic_checks_enabled)
                else []
            )
            best_check_bits: tuple[int, ...] = ()
            check_names: tuple[str, ...] = ()
            if checks and checks[0].bits:
                best_check_bits = tuple(
                    max(ch.bits[j] for ch in checks if len(ch.bits) > j)
                    for j in range(len(checks[0].bits))
                )
                check_names = checks[0].names

            if sirsi.enabled and sirsi.regret_enabled:
                dec = self.regret_tracker.classify_and_update(
                    t.task_id,
                    pass_rate,
                    check_bits=best_check_bits,
                    check_names=check_names,
                )
                if dec.kind == REGRET_NEWLY_HURT:
                    n_regret_hurt += 1
                    self.sampler.record_hurt(t)
                elif dec.kind == REGRET_UNSOLVED_CHRONIC:
                    n_regret_chronic += 1
                elif dec.kind == REGRET_PINNED_PASS:
                    n_regret_pinned += 1
                    self.sampler.record_pinned(t)
            else:
                dec = RegretDecision(REGRET_UNSOLVED_NOVEL, 1.0, 0.0, ())

            shaped_base: list[float] = []
            shaped_checks: list[float] = []
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
                            slap_candidates.append(
                                _Seq(
                                    p_ids + list(o.token_ids),
                                    len(p_ids),
                                    -1.0,
                                    "slap_neg",
                                    span_mask=smask,
                                    tags=tuple(p_info.tags),
                                    regret_weight=dec.weight,
                                    regret_kind=dec.kind,
                                )
                            )
                # Apply anti-slop tie-breaker only when at least one sample succeeded (`not all_fail`),
                # preventing std-normalized advantage amplification from positively reinforcing wrong answers.
                ol_pen = self._overlong(o.completion_tokens, pass_rate=pass_rate)
                c_bon = self._concise_bonus(o.completion_tokens, reward=r, pass_rate=pass_rate)
                overlong_vals.append(ol_pen)
                base_val = r + ol_pen + c_bon - (0.0 if all_fail else s_pen)
                shaped_base.append(base_val)
                check_credit = checks[idx_s].credit if (checks and r < 1.0) else 0.0
                shaped_checks.append(base_val + check_credit)

            flat += rw
            lens += [o.completion_tokens for o in samples]

            if self.failure_ledger is not None:
                logged_any = False
                regret_extra = {
                    "regret_kind": dec.kind,
                    "prev_pass_rate": dec.prev_pass_rate,
                    "regret_weight": dec.weight,
                    "regressed_checks": list(dec.regressed_checks),
                }
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
                            **regret_extra,
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
                        **regret_extra,
                    )
                    failures_logged += 1

            if all_fail:
                stats["groups/all_fail"] += 1
                all_fail_entries.append((t, p_ids, samples, paths, dec))
            if all(r >= 1 for r in rw):
                stats["groups/all_pass"] += 1

            if checks:
                if not all_fail_oracle:
                    adv = group_advantages(shaped_checks, rl.get("adv_norm", "std"))
                else:
                    has_exec = any(ch.has_exec_pass for ch in checks)
                    spread = (max(shaped_checks) - min(shaped_checks)) if shaped_checks else 0.0
                    if has_exec and spread >= sirsi.atomic_check_min_spread:
                        # Dr.GRPO unnormalized centering keeps partial-credit advs strictly <= alpha_check.
                        adv = group_advantages(shaped_checks, norm="mean")
                    else:
                        adv = group_advantages(shaped_base, rl.get("adv_norm", "std"))
            else:
                adv = group_advantages(shaped_base, rl.get("adv_norm", "std"))
            if adv is None:
                continue
            stats["groups/used"] += 1
            for o, a in zip(samples, adv):
                if o.token_ids:
                    seqs.append(
                        _Seq(
                            p_ids + list(o.token_ids),
                            len(p_ids),
                            float(a),
                            "pg",
                            regret_weight=dec.weight,
                            regret_kind=dec.kind,
                        )
                    )

        # G2: Prioritize NEWLY_HURT over novel/chronic before slicing hint & slap budgets.
        if all_fail_entries:
            all_fail_entries.sort(
                key=lambda item: (item[4].kind == REGRET_NEWLY_HURT, item[4].weight),
                reverse=True,
            )
            for t, p_ids, samples, paths, dec in all_fail_entries:
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
                        hint_text = self.teacher.hint(
                            t.prompt,
                            bad_o.text,
                            bad_tags,
                            regret_kind=dec.kind if sirsi.regret_enabled else None,
                            prev_pass_rate=dec.prev_pass_rate if sirsi.regret_enabled else None,
                        )
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
                                    _Seq(
                                        p_ids + list(ro.token_ids),
                                        len(p_ids),
                                        1.0,
                                        "teacher",
                                        tags=("GUIDED_HINT",),
                                        regret_weight=dec.weight,
                                        regret_kind=dec.kind,
                                    )
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
                        seqs.append(
                            _Seq(
                                p_ids + c_ids,
                                len(p_ids),
                                1.0,
                                "teacher",
                                regret_weight=dec.weight,
                                regret_kind=dec.kind,
                            )
                        )

        if sirsi.enabled and sirsi.slap_max_seqs_per_step > 0:
            slap_candidates.sort(
                key=lambda s: (s.regret_kind == REGRET_NEWLY_HURT, s.regret_weight),
                reverse=True,
            )
            slap_seqs = slap_candidates[: sirsi.slap_max_seqs_per_step]
        else:
            slap_seqs = []
        seqs.extend(slap_seqs)
        n = max(len(tasks), 1)
        stats["reward/mean"] = sum(flat) / max(len(flat), 1)
        stats["reward/pass_rate"] = sum(r >= 1 for r in flat) / max(len(flat), 1)
        stats["groups/mean_pass_rate"] = sum(group_pass_rates) / max(len(group_pass_rates), 1)
        stats["rewards/overlong_mean"] = sum(overlong_vals) / max(len(overlong_vals), 1)
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
            if sirsi.regret_enabled:
                stats["regret/newly_hurt"] = float(n_regret_hurt)
                stats["regret/unsolved_chronic"] = float(n_regret_chronic)
                stats["regret/pinned_pass"] = float(n_regret_pinned)
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
        self.stack.model.train()
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
                    sw = torch.tensor(
                        [float(s.regret_weight) for s in chunk],
                        dtype=torch.float32,
                        device=lp.device,
                    )
                    ul_mean = unlikelihood_span_loss(lp, span_bool, eps=slap_eps, seq_weights=sw)
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
                        del aux_now, lp, loss, ul_mean, span_bool, chunk_toks, sw, ids, mask, m, adv
                        continue
                    del ul_mean, span_bool, chunk_toks, sw
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
                    del ref_lp, ratio, a, surr
                    if kl_coef > 0 and self.ref is not None:
                        rlp = self._ref_logprobs(ids)
                        self.stack.aux_losses()
                        d = rlp - lp
                        kl = ((torch.exp(d) - d - 1) * m).sum() / n_pg_tok
                        loss = loss + kl_coef * kl
                        stats["loss/kl"] += float(kl.detach())
                        del rlp, d, kl
                for a_name, a_loss in aux_now.items():
                    loss = loss + float(aux_w.get(a_name, 1.0)) * a_loss
                    stats[f"loss/{a_name}"] = stats.get(f"loss/{a_name}", 0.0) + float(a_loss.detach())
                loss.backward()
                del ids, mask, adv, m, lp, loss, aux_now
            stats.update({f"optim/{k}": v for k, v in self.opt.step().items()})
            self.opt.zero_grad(set_to_none=True)
        stats["clip_frac"] = clip_n / max(tok_n, 1.0)
        if pg:
            with torch.no_grad():
                ids, mask, _ = self._batch(pg[:mb])
                h = self.stack.hidden(ids)
                ent = token_entropy(
                    h[:, :-1],
                    self.stack.lm_head_weight,
                    chunk_size=self._logprob_chunk_size,
                    temperature=self.temperature,
                )
                stats["entropy"] = float((ent * mask[:, 1:]).sum() / mask[:, 1:].sum().clamp_min(1))
                self.stack.aux_losses()
                del ids, mask, h, ent
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
        mode = str(gcfg.get("mode", "aggregate")).strip().lower()
        require(mode in ("aggregate", "soft_hurt"), "rl.gate.mode must be 'aggregate' or 'soft_hurt'", mode=mode)
        depth = (gcfg.get("depth") or self.stack.loop.spec.num_passes) if self.stack.loop is not None else None
        gate_dir = os.path.join(self.run_dir, "gate", f"step_{self.step:07d}")
        scores: list[float] = []
        weights: list[float] = []
        lens: list[float] = []
        res: dict[str, float] = {}
        cur_items: dict[str, float] = {}
        item_weights: dict[str, float] = {}
        for entry in gcfg.get("benchmarks") or []:
            b = {"name": entry} if isinstance(entry, str) else dict(entry)
            b_name = str(b["name"])
            b_weight = float(b.get("weight", 1.0))
            summ = run_benchmark(b_name, self.gen, limit=b.get("limit"), depths=[depth],
                                 n_samples=int(b.get("n_samples", 1)), max_new_tokens=b.get("max_new_tokens"),
                                 out_dir=gate_dir, offline_items=b.get("offline_items"), seed=int(b.get("seed", 0)))
            label, d = next(iter(summ["depths"].items()))
            b_score = float(d.get("mean_score", 0.0))
            scores.append(b_score * b_weight)
            weights.append(b_weight)
            lens.append(float(d.get("mean_completion_tokens", 0.0)))
            res[b_name] = b_score
            if mode == "soft_hurt":
                raw_items = d.get("item_scores")
                if not isinstance(raw_items, dict):
                    raw_items = {}
                    jpath = os.path.join(gate_dir, b_name, f"depth_{label}.jsonl")
                    if os.path.exists(jpath):
                        with open(jpath, encoding="utf-8") as jf:
                            for line in jf:
                                s_ln = line.strip()
                                if not s_ln:
                                    continue
                                try:
                                    rec = json.loads(s_ln)
                                except json.JSONDecodeError:
                                    continue
                                if isinstance(rec, dict) and "id" in rec and "mean_score" in rec:
                                    raw_items[str(rec["id"])] = float(rec["mean_score"])
                w_i = b_weight / float(max(len(raw_items), 1))
                for item_id, item_sc in raw_items.items():
                    key = f"{b_name}::{item_id}"
                    cur_items[key] = max(0.0, min(1.0, float(item_sc)))
                    item_weights[key] = w_i
        if not scores:
            return {}
        score, mean_len = sum(scores) / max(sum(weights), 1e-12), sum(lens) / len(lens)
        gate_metrics: dict[str, Any] = {
            "gate/score": score,
            "gate/mean_len": mean_len,
            **{f"gate/{k}": v for k, v in res.items()},
        }
        ok_len = self.best_len is None or mean_len <= float(gcfg.get("max_len_growth", 1.5)) * self.best_len
        min_delta = float(gcfg.get("min_delta", 0.0))

        diff_payload: dict[str, Any] | None = None
        if mode == "soft_hurt":
            tau = float(gcfg.get("pass_threshold", 0.5))
            lambda_hurt = float(gcfg.get("lambda_hurt", 2.0))
            min_soft_hurt = float(gcfg.get("min_soft_hurt", 0.0))
            max_hurt_frac = float(gcfg.get("max_hurt_frac", 0.05))
            max_bench_drop = float(gcfg.get("max_bench_drop", 0.025))
            is_baseline = not self.best_gate_items
            if is_baseline:
                common = sorted(cur_items)
                newly_helped: list[str] = []
                newly_hurt: list[str] = []
                gain = hurt = soft_hurt = hurt_frac = 0.0
                ok_bench = True
                should_promote = ok_len
            else:
                old = self.best_gate_items
                common = [k for k in cur_items if k in old]
                w_sum = sum(item_weights[k] for k in common)
                if w_sum <= 0.0:
                    w_sum = 1.0
                newly_helped = [k for k in common if old[k] < tau and cur_items[k] >= tau]
                newly_hurt = [k for k in common if old[k] >= tau and cur_items[k] < tau]
                gain = sum(item_weights[k] * max(0.0, cur_items[k] - old[k]) for k in common) / w_sum
                hurt = sum(item_weights[k] * max(0.0, old[k] - cur_items[k]) for k in common) / w_sum
                soft_hurt = gain - lambda_hurt * hurt
                hurt_frac = len(newly_hurt) / float(max(len(common), 1))
                ok_bench = all(
                    res[b_k] + 1e-12 >= self.best_bench_scores.get(b_k, 0.0) - max_bench_drop
                    for b_k in res
                )
                soft_ok = (
                    (soft_hurt > 0.0)
                    if hurt > 0.0
                    else (
                        soft_hurt >= min_soft_hurt
                        and score + 1e-12 >= (self.best_gate if self.best_gate is not None else 0.0) + min_delta
                    )
                )
                should_promote = ok_len and ok_bench and (hurt_frac <= max_hurt_frac + 1e-12) and soft_ok

            if should_promote:
                self.best_gate, self.best_len = score, mean_len
                self.best_gate_items = dict(cur_items)
                self.best_bench_scores = dict(res)
                if is_baseline:
                    self.pinned_passed = {k for k, v in cur_items.items() if v >= tau}
                else:
                    self.pinned_passed |= {k for k, v in cur_items.items() if v >= tau}

            gate_metrics.update({
                "gate/soft_hurt": float(soft_hurt),
                "gate/gain": float(gain),
                "gate/hurt": float(hurt),
                "gate/hurt_frac": float(hurt_frac),
                "gate/n_newly_helped": float(len(newly_helped)),
                "gate/n_newly_hurt": float(len(newly_hurt)),
                "gate/n_pinned_passed": float(len(self.pinned_passed)),
            })
            diff_payload = {
                "step": int(self.step),
                "is_baseline": bool(is_baseline),
                "score": float(score),
                "best_gate": None if self.best_gate is None else float(self.best_gate),
                "mean_len": float(mean_len),
                "best_len": None if self.best_len is None else float(self.best_len),
                "tau": float(tau),
                "lambda_hurt": float(lambda_hurt),
                "min_soft_hurt": float(min_soft_hurt),
                "max_hurt_frac": float(max_hurt_frac),
                "max_bench_drop": float(max_bench_drop),
                "gain": float(gain),
                "hurt": float(hurt),
                "soft_hurt": float(soft_hurt),
                "hurt_frac": float(hurt_frac),
                "ok_len": bool(ok_len),
                "ok_bench": bool(ok_bench),
                "n_common": len(common),
                "n_newly_helped": len(newly_helped),
                "n_newly_hurt": len(newly_hurt),
                "n_pinned_passed": len(self.pinned_passed),
                "newly_helped": sorted(newly_helped),
                "newly_hurt": sorted(newly_hurt),
                "bench_scores": {k: float(v) for k, v in res.items()},
                "best_bench_scores": {k: float(v) for k, v in self.best_bench_scores.items()},
                "item_scores": {k: float(v) for k, v in sorted(cur_items.items())},
            }
        else:
            improved = self.best_gate is None or score >= self.best_gate + min_delta
            should_promote = improved and ok_len
            if should_promote:
                self.best_gate, self.best_len = score, mean_len

        path = self._save_ckpt(metrics=gate_metrics)
        if should_promote:
            self.ckpt.promote(path, metrics=gate_metrics)
            decision = "promoted"
        else:
            prom = self.ckpt.promoted()
            if prom and gcfg.get("rollback", True):
                ck = self.ckpt.load(prom)
                assert ck is not None
                st_ck = ck.get("state") or {}
                smeta = st_ck.get("stack") or {}
                sc = float((smeta.get("expert_delta_scale")) or 1.0)
                self.stack.restore_checkpoint(
                    ck["params"],
                    strict=False,
                    expert_delta_scale=sc,
                    prior_stack_meta=smeta,
                )
                if ck["optim"] is not None:
                    self.opt.load_state_dict(ck["optim"])
                decision = f"rolled back to {os.path.basename(prom)}"
            else:
                decision = "rejected (no rollback)"

        if diff_payload is not None:
            diff_payload["decision"] = decision
            os.makedirs(gate_dir, exist_ok=True)
            with open(os.path.join(gate_dir, "soft_hurt_diff.json"), "w", encoding="utf-8") as df:
                json.dump(diff_payload, df, indent=2)

        log.info("[rl] gate step %d: score %.4f (best %s) len %.0f -> %s", self.step, score, self.best_gate,
                 mean_len, decision)
        return {**gate_metrics, "gate/decision": decision}

    # ------------------------------------------------------------------ main loop
    def _state(self) -> dict[str, Any]:
        return {
            "stage": self.cfg.get("stage"),
            "sampler": self.sampler.state_dict(),
            "temperature": self.temperature,
            "init_entropy": self.init_entropy,
            "best_gate": self.best_gate,
            "best_len": self.best_len,
            "best_gate_items": dict(self.best_gate_items),
            "best_bench_scores": dict(self.best_bench_scores),
            "pinned_passed": sorted(self.pinned_passed),
            "regret_tracker": self.regret_tracker.state_dict(),
            "stack": self.stack.describe(),
        }

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
                    ck["params"],
                    strict=False,
                    expert_delta_scale=sc,
                    prior_stack_meta=smeta,
                    drop_inactive_loop_adapters=bool(self.tcfg.get("drop_inactive_loop_adapters", False)),
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
        self.best_gate_items = {str(k): float(v) for k, v in (st.get("best_gate_items") or {}).items()}
        self.best_bench_scores = {str(k): float(v) for k, v in (st.get("best_bench_scores") or {}).items()}
        if self.best_bench_scores:
            gcfg = dict(self.rl.get("gate") or {})
            w_num = 0.0
            w_den = 0.0
            for entry in gcfg.get("benchmarks") or []:
                b = {"name": entry} if isinstance(entry, str) else dict(entry)
                b_name = str(b["name"])
                if b_name in self.best_bench_scores:
                    b_weight = float(b.get("weight", 1.0))
                    w_num += self.best_bench_scores[b_name] * b_weight
                    w_den += b_weight
            if w_den > 0.0:
                self.best_gate = w_num / max(w_den, 1e-12)
        self.pinned_passed = {str(k) for k in (st.get("pinned_passed") or ())}
        if isinstance(st.get("regret_tracker"), dict):
            self.regret_tracker.load_state_dict(st["regret_tracker"])

    def train(self, on_step: Callable[[int, dict[str, Any]], None] | None = None) -> dict[str, Any]:
        steps = int(self.tcfg.get("steps", 200))
        ckpt_every = int(self.tcfg.get("ckpt_every", 50))
        gcfg = dict(self.rl.get("gate") or {})
        gate_every = int(gcfg.get("every", 0))
        budget = float(self.tcfg.get("time_budget_hours", 0)) * 3600
        self._resume()
        t0 = time.time()
        last: dict[str, Any] = {}
        try:
            if gcfg.get("eval_at_step_0", False) and self.step == 0 and not self.best_gate_items:
                g0 = self.gate()
                if g0:
                    self.metrics.log(self.step, g0)
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
