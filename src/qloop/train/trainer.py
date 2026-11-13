"""Language-model trainer for stages S1 (loop heal), S2 (SFT), S3 (long context / D-LSA distill).

Per optimizer step:
    sample loop depth (keeps depth-1 == stock graph valid)  ->  grad_accum micro-batches:
    hidden = text_model(ids) ; loss = chunked/CCE LM loss + loop aux + LSA KL ; backward
    (FusedStepOptimizer steps every parameter inside backward on the last micro-batch)
Logs JSONL metrics (+ optional wandb), runs a depth-sweep held-out eval (the go/no-go signal for
the loop: loss at depth 2 must beat depth 1), checkpoints atomically, honours a wall-clock budget
(Colab sessions end) and resumes exactly (trainables + optimizer + data stream + RNG).
"""
from __future__ import annotations

import json
import logging
import math
import os
import random
import signal
import time
from typing import Any

import torch

from ..debug import cuda_mem_brief, require, stage
from .checkpoint import CheckpointManager
from .loss import lm_loss
from .optim import FusedStepOptimizer
from .stack import Stack, build_stack

log = logging.getLogger("qloop")


class MetricsLogger:
    def __init__(self, run_dir: str, cfg: dict[str, Any]):
        self.path = os.path.join(run_dir, "metrics.jsonl")
        self.wandb = None
        wcfg = (cfg.get("log") or {}).get("wandb")
        if wcfg:
            try:
                import wandb

                self.wandb = wandb.init(project=wcfg.get("project", "qloop"), name=wcfg.get("name"),
                                        config=cfg, resume="allow", id=wcfg.get("id"))
            except Exception as e:
                log.warning("wandb disabled: %r", e)

    def log(self, step: int, metrics: dict[str, Any]) -> None:
        rec = {"step": step, "time": time.time(), **metrics}
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, default=float) + "\n")
        if self.wandb is not None:
            self.wandb.log(metrics, step=step)


def _to_device(batch: list[dict[str, Any]], device: torch.device, pass_pos: bool) -> dict[str, torch.Tensor]:
    ids = torch.stack([b["input_ids"] for b in batch]).to(device)
    labels = torch.stack([b["labels"] for b in batch]).to(device)
    out = {"input_ids": ids, "labels": labels}
    if pass_pos and "position_ids" in batch[0]:
        out["position_ids"] = torch.stack([b["position_ids"] for b in batch]).to(device)
    return out


class LMTrainer:
    def __init__(self, cfg: dict[str, Any], run_dir: str, stack: Stack | None = None, hub_api: Any = None):
        self.cfg, self.run_dir = cfg, run_dir
        self.tcfg = dict(cfg.get("train") or {})
        self._hub_api = hub_api
        sync_cfg = (cfg.get("checkpoint") or {}).get("sync")
        if stack is None and sync_cfg:
            from ..hub import HubSyncManager

            hub = HubSyncManager.from_sync_cfg(sync_cfg, api=hub_api)
            if hub is not None:
                stage_id = str(cfg.get("stage") or os.path.basename(os.path.abspath(run_dir)))
                run_root = os.path.dirname(os.path.abspath(run_dir))
                hub.prepare_stage_workspace(stage_id, dict(cfg), run_root)
        self.stack = stack or build_stack(cfg)
        require(bool(self.stack.named_trainable), "nothing is trainable - check train.trainable / loop / lsa / experts")
        self.seed = int(cfg.get("seed", 0))
        self.rng = random.Random(self.seed)
        torch.manual_seed(self.seed)
        ocfg = dict(self.tcfg.get("optim") or {})
        ocfg.setdefault("total_steps", int(self.tcfg.get("steps", 1000)))
        ocfg.setdefault("seed", self.seed)
        self.opt = FusedStepOptimizer.from_config(list(self.stack.named_trainable.items()), ocfg)
        self.ckpt = CheckpointManager(run_dir, keep_last=int(self.tcfg.get("keep_last", 3)),
                                      sync=sync_cfg, hub_api=hub_api)
        self.metrics = MetricsLogger(run_dir, cfg)
        self.step = 0
        self.tokens_seen = 0
        self._stop = False
        self.stream = None
        self.eval_batches: list[dict[str, Any]] = []

    # ------------------------------------------------------------------ data
    def _build_stream(self):
        from ..data.mixture import build_train_stream

        dcfg = dict(self.cfg.get("data") or {})
        require(bool(dcfg.get("sources")), "data.sources is empty")
        return build_train_stream(dcfg, self.stack.tokenizer, seed=self.seed)

    def _eval_set(self) -> list[dict[str, Any]]:
        from ..data.mixture import build_train_stream

        ecfg = dict(self.cfg.get("eval_quick") or {})
        n = int(ecfg.get("n_batches", 0))
        if n <= 0:
            return []
        dcfg = dict(self.cfg.get("data") or {})
        dcfg.update(ecfg.get("data_overrides") or {})
        stream = build_train_stream(dcfg, self.stack.tokenizer, seed=self.seed + 7_777_777)
        return [next(stream) for _ in range(n)]

    # ------------------------------------------------------------------ resume / save
    def _state(self) -> dict[str, Any]:
        return {"stage": self.cfg.get("stage"), "tokens_seen": self.tokens_seen, "stack": self.stack.describe(),
                "data": self.stream.state_dict() if self.stream is not None else None,
                "config": self.cfg.get("_meta_", {})}

    def save(self, *, completed: bool = False, metrics: dict[str, Any] | None = None) -> str:
        raw_cfg = self.cfg.to_dict() if hasattr(self.cfg, "to_dict") else dict(self.cfg)
        return self.ckpt.save(
            self.step,
            self.stack.checkpoint_params(),
            self.opt.state_dict(),
            self._state(),
            completed=completed,
            target_steps=int(self.tcfg.get("steps", 1000)),
            metrics=metrics,
            config_dict=raw_cfg,
        )

    def maybe_resume(self) -> None:
        from ..hub import resolve_run_path

        run_root = os.path.dirname(os.path.abspath(self.run_dir))
        init = resolve_run_path(self.tcfg.get("init_from"), run_root=run_root)
        if self.tcfg.get("resume", True) and self.ckpt.latest():
            ck = self.ckpt.load(map_location="cpu")
            assert ck is not None
            st = ck["state"]
            smeta = st.get("stack") or {}
            sc = float((smeta.get("expert_delta_scale")) or 1.0)
            self.stack.restore_checkpoint(ck["params"], strict=True, expert_delta_scale=sc, prior_stack_meta=smeta)
            if ck["optim"] is not None:
                self.opt.load_state_dict(ck["optim"])
            self.step, self.tokens_seen = int(st["step"]), int(st.get("tokens_seen", 0))
            if st.get("data") is not None and self.stream is not None:
                self.stream.load_state_dict(st["data"])
            log.info("resumed at step %d (%.2fM tokens seen)", self.step, self.tokens_seen / 1e6)
        elif init:
            ck = CheckpointManager(init, keep_last=0).load(map_location="cpu") if os.path.isdir(init) else None
            require(ck is not None, "train.init_from: no checkpoint found", init_from=init)
            assert ck is not None
            smeta = (ck.get("state") or {}).get("stack") or {}
            sc = float((smeta.get("expert_delta_scale")) or 1.0)
            counts = self.stack.restore_checkpoint(
                ck["params"], strict=False, expert_delta_scale=sc, prior_stack_meta=smeta
            )
            log.info("initialised from %s (%s; optimizer fresh)", ck["path"], counts)

    # ------------------------------------------------------------------ eval
    @torch.no_grad()
    def quick_eval(self) -> dict[str, float]:
        if not self.eval_batches:
            return {}
        st, out = self.stack, {}
        default_depths = [1, st.loop.spec.num_passes] if st.loop else [1]
        depths = list((self.cfg.get("eval_quick") or {}).get("depths") or default_depths)
        was_training = st.model.training
        st.model.eval()
        prev = st.loop.active_passes if st.loop else None
        try:
            for d in depths:
                if st.loop is not None:
                    st.loop.set_active_passes(int(d))
                elif d != 1:
                    continue
                tot, ntok = 0.0, 0
                for b in self.eval_batches:
                    bt = _to_device([b], st.device, bool(self.tcfg.get("pass_position_ids", False)))
                    h = st.hidden(bt["input_ids"], bt.get("position_ids"))
                    loss, n = lm_loss(h, st.lm_head_weight, bt["labels"], impl="chunked")
                    tot += float(loss) * int(n)
                    ntok += int(n)
                out[f"eval/loss_d{d}"] = tot / max(ntok, 1)
        finally:
            if st.loop is not None and prev is not None:
                st.loop.set_active_passes(prev)
            if st.lsa is not None:
                st.lsa.aux_loss()  # drop KL terms produced in eval
            st.model.train(was_training)
        if "eval/loss_d1" in out and len(depths) > 1:
            out["eval/depth_gain"] = out["eval/loss_d1"] - out[f"eval/loss_d{max(depths)}"]
        log.info("eval: %s", {k: round(v, 4) for k, v in out.items()})
        return out

    # ------------------------------------------------------------------ loop
    def _install_signal_handlers(self) -> None:
        def handler(signum, frame):
            log.warning("signal %s received: will checkpoint and stop after this step", signum)
            self._stop = True
        try:
            signal.signal(signal.SIGTERM, handler)
        except Exception:
            pass

    def train(self) -> dict[str, Any]:
        st, t = self.stack, self.tcfg
        steps, accum = int(t.get("steps", 1000)), int(t.get("grad_accum", 1))
        mb = int(t.get("micro_batch", 1))
        log_every, ckpt_every = int(t.get("log_every", 10)), int(t.get("ckpt_every", 200))
        eval_every = int(t.get("eval_every", 0))
        budget = float(t.get("time_budget_hours", 0)) * 3600
        aux_w = dict(t.get("aux_weights") or {})
        impl = t.get("loss_impl", "auto")
        pass_pos = bool(t.get("pass_position_ids", False))
        with stage("data:build"):
            self.stream = self._build_stream()
            self.eval_batches = self._eval_set()
        self.maybe_resume()
        self._install_signal_handlers()
        st.model.train()
        t_start = time.time()
        t_log, tok_log = time.time(), 0
        if self.step == 0 and eval_every:
            self.metrics.log(0, self.quick_eval())
        last: dict[str, Any] = {}
        try:
            while self.step < steps and not self._stop:
                depth = st.loop.sample_active_passes(self.rng) if st.loop is not None else 1
                losses: dict[str, float] = {}
                for k in range(accum):
                    batch = _to_device([next(self.stream) for _ in range(mb)], st.device, pass_pos)
                    self.opt.set_boundary(k == accum - 1)
                    h = st.hidden(batch["input_ids"], batch.get("position_ids"))
                    loss, ntok = lm_loss(h, st.lm_head_weight, batch["labels"], impl=impl)
                    total = loss
                    for name, a in st.aux_losses().items():
                        total = total + float(aux_w.get(name, 1.0)) * a
                        losses[name] = losses.get(name, 0.0) + float(a.detach()) / accum
                    (total / accum).backward()
                    losses["lm"] = losses.get("lm", 0.0) + float(loss.detach()) / accum
                    n = int(ntok)
                    self.tokens_seen += int(batch["input_ids"].numel())
                    tok_log += int(batch["input_ids"].numel())
                    if not math.isfinite(losses["lm"]):
                        raise FloatingPointError(f"non-finite LM loss at step {self.step} (micro {k}, {n} tokens)")
                ostats = self.opt.step()
                self.step += 1
                if self.step % log_every == 0 or self.step == 1:
                    dt = time.time() - t_log
                    last = {**{f"loss/{k}": v for k, v in losses.items()}, "depth": depth,
                            "tok_per_s": tok_log / max(dt, 1e-6), "tokens_seen": self.tokens_seen,
                            "lr_factor": self.opt.lr_factor, **{f"optim/{k}": v for k, v in ostats.items()},
                            **st.stats()}
                    if torch.cuda.is_available():
                        last["mem/peak_gib"] = torch.cuda.max_memory_allocated() / 2**30
                    log.info("step %d/%d lm %.4f d=%d %.0f tok/s %s", self.step, steps, losses["lm"], depth,
                             last["tok_per_s"], cuda_mem_brief())
                    self.metrics.log(self.step, last)
                    t_log, tok_log = time.time(), 0
                if eval_every and self.step % eval_every == 0:
                    ev = self.quick_eval()
                    last.update(ev)
                    self.metrics.log(self.step, ev)
                if self.step % ckpt_every == 0:
                    self.save(metrics=last)
                if budget and time.time() - t_start > budget:
                    log.warning("time budget %.1fh reached: checkpointing and stopping", budget / 3600)
                    break
        except KeyboardInterrupt:
            log.warning("interrupted: saving checkpoint before exit")
            self.save(metrics=last)
            raise
        if hasattr(self.stream, "log_stats"):
            self.stream.log_stats()
        final = self.quick_eval() if self.eval_batches else {}
        if final:
            self.metrics.log(self.step, final)
        path = self.save(completed=(self.step >= steps), metrics={**last, **final})
        return {"step": self.step, "checkpoint": path, "last": last, "eval": final}
