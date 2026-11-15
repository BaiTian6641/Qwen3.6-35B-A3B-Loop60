"""Language-model trainer for all Loop-HSSD v3.2 stages:
  * S1 (`s1_loop_a6b`): 88L looped MoE healing with chunked LM loss + loop aux
  * S1b (`s1b_hssd_distill`): Zero-duplicate-VRAM `SharedBackboneTeacher` + 2-phase HSSD
    distillation (Phase 1 online per-layer teacher-forced warm-up -> Phase 2 end-to-end
    chunked-vocab KL + scheduled `step_blend` + write-budget regularization)
  * S3 (`s3_long_1m_lsa`): Alternating 64K `hybrid` D-LSA steps and scheduled 512K-1M
    sparse-prefix + 32K chunked-CE suffix calibration (`prefix_suffix_calibration_step`)
  * S2 (`s2_joint_bptt`): Segmented KV-BPTT (`SegmentedKVBPTTRunner`, `4 x 32K = 128K`) with
    sparse top-K key/value & indexer replay when `seq_len >= 2 * seg_len`

Logs JSONL metrics (+ optional wandb), runs a depth-sweep held-out eval, checkpoints atomically,
honours a wall-clock budget (Colab sessions end) and resumes exactly (trainables + optimizer +
data stream + RNG).
"""
from __future__ import annotations

import contextlib
import inspect
import json
import logging
import math
import os
import random
import signal
import time
from typing import Any

import torch
import torch.nn.functional as F

from ..debug import cuda_mem_brief, require, stage
from ..utils.tf32 import configure_tf32
from .bptt import BPTTConfig, SegmentedKVBPTTRunner
from .checkpoint import CheckpointManager
from .distill import (
    HSSDDistillConfig,
    SharedBackboneTeacher,
    compute_shared_distill_loss,
    run_layerwise_warmup_microstep,
)
from .long_calib import prefix_suffix_calibration_step
from .loss import lm_loss
from .optim import FusedStepOptimizer, defer_micro_step
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

                self.wandb = wandb.init(
                    project=wcfg.get("project", "qloop"),
                    name=wcfg.get("name"),
                    config=cfg,
                    resume="allow",
                    id=wcfg.get("id"),
                )
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
        configure_tf32(bool(self.tcfg.get("allow_tf32", True)))
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
        self.opt = FusedStepOptimizer.from_config(
            list(self.stack.named_trainable.items()),
            ocfg,
            grad_accum=int(self.tcfg.get("grad_accum", 1)),
        )
        self.ckpt = CheckpointManager(
            run_dir,
            keep_last=int(self.tcfg.get("keep_last", 3)),
            sync=sync_cfg,
            hub_api=hub_api,
        )
        self.metrics = MetricsLogger(run_dir, cfg)
        self.step = 0
        self.tokens_seen = 0
        self._stop = False
        self.stream: Any = None
        self.eval_batches: list[dict[str, Any]] = []
        self.shared_teacher: SharedBackboneTeacher | None = None

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

    def _next_long_calib_batch(self, total_len: int, pass_pos: bool) -> dict[str, torch.Tensor]:
        id_parts: list[torch.Tensor] = []
        lbl_parts: list[torch.Tensor] = []
        cur_len = 0
        while cur_len < total_len:
            item = next(self.stream)
            ids_1d = item["input_ids"]
            lbl_1d = item["labels"]
            id_parts.append(ids_1d)
            lbl_parts.append(lbl_1d)
            cur_len += int(ids_1d.numel())
        full_ids = torch.cat(id_parts, dim=-1)[:total_len].unsqueeze(0).to(self.stack.device)
        full_lbl = torch.cat(lbl_parts, dim=-1)[:total_len].unsqueeze(0).to(self.stack.device)
        out = {"input_ids": full_ids, "labels": full_lbl}
        if pass_pos:
            out["position_ids"] = torch.arange(total_len, device=self.stack.device, dtype=torch.long).unsqueeze(0)
        return out

    # ------------------------------------------------------------------ resume / save
    def _state(self) -> dict[str, Any]:
        return {
            "stage": self.cfg.get("stage"),
            "tokens_seen": self.tokens_seen,
            "stack": self.stack.describe(),
            "data": self.stream.state_dict() if self.stream is not None else None,
            "config": self.cfg.get("_meta_", {}),
        }

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
            if st.hssd is not None:
                for ad in st.hssd.adapters.values():
                    ad.clear_aux()
            st.model.train(was_training)
        if "eval/loss_d1" in out and len(depths) > 1:
            out["eval/depth_gain"] = out["eval/loss_d1"] - out[f"eval/loss_d{max(depths)}"]
        log.info("eval: %s", {k: round(v, 4) for k, v in out.items()})
        return out

    # ------------------------------------------------------------------ specialized step helpers
    def _run_distill_step(
        self,
        distill_cfg: HSSDDistillConfig,
        shared_teacher: SharedBackboneTeacher,
        *,
        accum: int,
        mb: int,
        pass_pos: bool,
        depth: int,
    ) -> tuple[dict[str, float], int]:
        st = self.stack
        losses: dict[str, float] = {}
        step_toks = 0

        if self.step < distill_cfg.layerwise_warmup_steps:
            for k in range(accum):
                batch = _to_device([next(self.stream) for _ in range(mb)], st.device, pass_pos)
                self.opt.set_boundary(k == accum - 1)
                w_stats = run_layerwise_warmup_microstep(
                    st,
                    shared_teacher,
                    self.opt,
                    batch["input_ids"],
                    position_ids=batch.get("position_ids"),
                    depth=None,
                    grad_accum=accum,
                    mse_weight=distill_cfg.layerwise_mse_weight,
                    budget_weight=distill_cfg.aux_loss_weight,
                )
                losses["lm"] = losses.get("lm", 0.0) + w_stats["distill/loss"] / accum
                losses["layerwise_mse"] = losses.get("layerwise_mse", 0.0) + w_stats["distill/layerwise_mse"] / accum
                losses["hssd_budget"] = losses.get("hssd_budget", 0.0) + w_stats["distill/hssd_budget"] / accum
                losses["distill_phase"] = 1.0
                nb = int(batch["input_ids"].numel())
                step_toks += nb
                if not math.isfinite(losses["lm"]):
                    raise FloatingPointError(f"non-finite Phase-1 distill loss at step {self.step} (micro {k})")
            return losses, step_toks

        if st.hssd is not None and callable(getattr(st.hssd, "step_blend", None)):
            blend_step = max(0, self.step - int(distill_cfg.layerwise_warmup_steps))
            losses["hssd_blend"] = float(st.hssd.step_blend(blend_step))

        for k in range(accum):
            batch = _to_device([next(self.stream) for _ in range(mb)], st.device, pass_pos)
            self.opt.set_boundary(k == accum - 1)
            dout = compute_shared_distill_loss(
                st,
                shared_teacher,
                distill_cfg,
                batch["input_ids"],
                position_ids=batch.get("position_ids"),
                labels=batch.get("labels"),
                depth=None,
                step=None,
            )
            (dout.loss / accum).backward()
            losses["lm"] = losses.get("lm", 0.0) + float(dout.loss.detach()) / accum
            losses["distill_kl"] = losses.get("distill_kl", 0.0) + float(dout.kl_loss.detach()) / accum
            if distill_cfg.hidden_mse_weight > 0.0:
                losses["hidden_mse"] = losses.get("hidden_mse", 0.0) + float(dout.hidden_mse_loss.detach()) / accum
            losses["distill_aux"] = losses.get("distill_aux", 0.0) + float(dout.aux_loss.detach()) / accum
            losses["distill_phase"] = 2.0
            nb = int(batch["input_ids"].numel())
            step_toks += nb
            if not math.isfinite(losses["lm"]):
                raise FloatingPointError(f"non-finite Phase-2 distill loss at step {self.step} (micro {k})")
        return losses, step_toks

    def _run_long_calib_step(
        self,
        lcalib_raw: dict[str, Any],
        *,
        aux_w: dict[str, Any],
        pass_pos: bool,
    ) -> tuple[dict[str, float], int]:
        st = self.stack
        min_prefix = int(lcalib_raw.get("prefix_len", 524_288))
        max_prefix = int(lcalib_raw.get("max_prefix_len", min_prefix))
        suffix_len = int(lcalib_raw.get("suffix_len", 32_768))
        chunk_size = max(1, int(lcalib_raw.get("chunk_size", 4096)))
        fp8_kv = bool(lcalib_raw.get("fp8_kv_cache", True))
        prefix_lsa_mode = str(lcalib_raw.get("prefix_lsa_mode", "sparse"))
        sparse_after = int(lcalib_raw.get("sparse_prefix_after_step", 80))

        if max_prefix > min_prefix:
            lo_c = max(1, min_prefix // chunk_size)
            hi_c = max(lo_c, max_prefix // chunk_size)
            eff_prefix = self.rng.randint(lo_c, hi_c) * chunk_size
        else:
            eff_prefix = min_prefix

        stream_seq_len = max(1, int((self.cfg.get("data") or {}).get("seq_len", 65_536)))
        if eff_prefix + suffix_len > 32 * stream_seq_len:
            eff_suffix = max(4, stream_seq_len)
            eff_prefix = max(4, stream_seq_len)
            eff_chunk = max(2, min(chunk_size, eff_prefix))
        else:
            eff_suffix = suffix_len
            eff_chunk = min(chunk_size, max(1, eff_prefix))

        total_len = eff_prefix + eff_suffix
        cbatch = self._next_long_calib_batch(total_len, pass_pos)

        sig = inspect.signature(prefix_suffix_calibration_step)
        extra_kw: dict[str, Any] = {}
        if "lsa_manager" in sig.parameters:
            extra_kw["lsa_manager"] = st.lsa
        if "global_step" in sig.parameters:
            extra_kw["global_step"] = self.step
        if "prefix_lsa_mode" in sig.parameters:
            extra_kw["prefix_lsa_mode"] = prefix_lsa_mode
        if "sparse_prefix_after_step" in sig.parameters:
            extra_kw["sparse_prefix_after_step"] = sparse_after

        # Fallback temporary sparse-prefix switch if standalone `long_calib.py` does not yet accept `lsa_manager`.
        fallback_temp_cm = contextlib.nullcontext()
        if (
            "lsa_manager" not in sig.parameters
            and st.lsa is not None
            and callable(getattr(st.lsa, "temporary_mode", None))
            and (self.step >= sparse_after or bool(getattr(st.lsa, "recall_ready", False)))
        ):
            fallback_temp_cm = st.lsa.temporary_mode(prefix_lsa_mode)

        self.opt.set_boundary(True)
        with defer_micro_step(self.opt), fallback_temp_cm:
            res = prefix_suffix_calibration_step(
                st.model,
                cbatch["input_ids"],
                prefix_len=eff_prefix,
                chunk_size=eff_chunk,
                labels=cbatch["labels"],
                backward=True,
                lsa_aux_weight=float(aux_w.get("lsa_kl", 1.0)),
                loop_aux_weight=float(aux_w.get("loop_aux", 1.0)),
                fp8_kv_cache=fp8_kv,
                **extra_kw,
            )
            if st.hssd is not None:
                for ad in st.hssd.adapters.values():
                    ad.clear_aux()

        lm_val = float(res.task_loss)
        if not math.isfinite(lm_val):
            raise FloatingPointError(f"non-finite long_calib loss at step {self.step}")
        losses = {
            "lm": lm_val,
            "lsa_kl": float(res.lsa_aux_loss),
            "calib_prefix_len": float(res.prefix_len),
            "calib_suffix_len": float(res.suffix_len),
            "long_calib_active": 1.0,
        }
        return losses, int(cbatch["input_ids"].numel())

    def _run_bptt_microstep(
        self,
        bptt_runner: SegmentedKVBPTTRunner,
        batch: dict[str, torch.Tensor],
        *,
        accum: int,
        aux_w: dict[str, Any],
    ) -> dict[str, float]:
        st = self.stack

        def _seg_loss_fn(_s_idx: int, seg_logits: torch.Tensor, seg_targets: torch.Tensor | None) -> torch.Tensor:
            require(seg_targets is not None, "BPTT segment targets must not be None")
            assert seg_targets is not None
            lg = seg_logits[:, : seg_targets.shape[1]]
            ce = F.cross_entropy(lg.reshape(-1, lg.shape[-1]).float(), seg_targets.reshape(-1), ignore_index=-100)
            if st.hssd is not None:
                ce = ce + float(aux_w.get("hssd_budget", 1.0)) * st.hssd.aux_loss().to(ce.dtype)
            return ce

        scale_hooks: list[Any] = []
        if accum > 1:
            inv_acc = 1.0 / float(accum)
            for p in st.named_trainable.values():
                if p.requires_grad:
                    scale_hooks.append(p.register_hook(lambda g, s=inv_acc: g * s))

        try:
            with defer_micro_step(self.opt):
                res = bptt_runner.run_step(
                    batch["input_ids"],
                    labels=batch["labels"],
                    loss_fn=_seg_loss_fn,
                    backward=True,
                )
        finally:
            for h in scale_hooks:
                h.remove()
            if st.hssd is not None:
                for ad in st.hssd.adapters.values():
                    ad.clear_aux()

        return {
            "lm": float(res.total_loss),
            "bptt_segments": float(res.num_segments),
            "bptt_replayed_tokens": float(res.stats.get("bptt/mean_replayed_tokens", 0.0)),
        }

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
        configure_tf32(bool(t.get("allow_tf32", True)))
        steps, accum = int(t.get("steps", 1000)), int(t.get("grad_accum", 1))
        mb = int(t.get("micro_batch", 1))
        log_every, ckpt_every = int(t.get("log_every", 10)), int(t.get("ckpt_every", 200))
        eval_every = int(t.get("eval_every", 0))
        budget = float(t.get("time_budget_hours", 0)) * 3600
        aux_w = dict(t.get("aux_weights") or {})
        impl = t.get("loss_impl", "auto")
        pass_pos = bool(t.get("pass_position_ids", False))

        distill_raw = dict(self.cfg.get("distill") or {})
        distill_on = bool(distill_raw.get("enabled", False))
        distill_cfg: HSSDDistillConfig | None = None
        if distill_on:
            distill_cfg = HSSDDistillConfig.from_dict(distill_raw)
            distill_cfg.validate()

        lcalib_raw = dict(self.cfg.get("long_calib") or {})
        lcalib_on = bool(lcalib_raw.get("enabled", False))
        calib_every = int(lcalib_raw.get("calib_every_steps", 10)) if lcalib_on else 0

        bptt_raw = dict(self.cfg.get("bptt") or {})
        bptt_on = bool(bptt_raw.get("enabled", False))
        bptt_cfg = BPTTConfig.from_dict(bptt_raw) if bptt_on else None
        bptt_runner = SegmentedKVBPTTRunner(st.model, cfg=bptt_cfg) if (bptt_on and bptt_cfg is not None) else None
        bptt_min_seq_len = 2 * int(bptt_cfg.seg_len) if bptt_cfg is not None else 0

        with stage("data:build"):
            self.stream = self._build_stream()
            self.eval_batches = self._eval_set()
        self.maybe_resume()
        if distill_on and distill_cfg is not None:
            self.shared_teacher = SharedBackboneTeacher(
                st,
                teacher_ckpt=distill_cfg.teacher_ckpt,
                teacher_track=distill_cfg.teacher_track,
                run_root=os.path.dirname(os.path.abspath(self.run_dir)),
            )

        self._install_signal_handlers()
        st.model.train()
        t_start = time.time()
        max_step0_loss = float(t.get("max_step0_loss", 10.0))
        model_name = (self.cfg.get("model") or {}).get("name")
        check_step0 = max_step0_loss > 0.0 and model_name not in (None, "", "toy")
        if self.step == 0 and (eval_every or (check_step0 and self.eval_batches)):
            ev = self.quick_eval()
            if eval_every or ev:
                self.metrics.log(0, ev)
            if check_step0 and ev:
                loss_d1 = ev.get("eval/loss_d1")
                if loss_d1 is None:
                    d_vals = [
                        float(v)
                        for k, v in ev.items()
                        if k.startswith("eval/loss_d") and isinstance(v, (int, float))
                    ]
                    if d_vals:
                        loss_d1 = next((x for x in d_vals if not math.isfinite(x)), min(d_vals))
                if loss_d1 is not None and (not math.isfinite(float(loss_d1)) or float(loss_d1) > max_step0_loss):
                    loss_f = float(loss_d1)
                    raise RuntimeError(
                        f"Step-0 eval loss ({loss_f:.4f}) exceeds train.max_step0_loss={max_step0_loss:.2f} "
                        "(uniform random vocab loss is ~12.42); base model weights appear corrupted or "
                        "mis-dequantized (e.g. incompatible quantized checkpoint layout). "
                        "Use model.name=Qwen/Qwen3.6-35B-A3B-FP8 or set train.max_step0_loss=0 to disable this check."
                    )
        # Reset throughput timer AFTER step-0 quick_eval() so initial eval duration never pollutes step-1 tok/s.
        t_log, tok_log = time.time(), 0
        last: dict[str, Any] = {}
        try:
            while self.step < steps and not self._stop:
                if st.lsa is not None and callable(getattr(st.lsa, "set_step", None)):
                    st.lsa.set_step(self.step)
                depth = st.loop.sample_active_passes(self.rng) if st.loop is not None else 1
                losses: dict[str, float] = {}

                if distill_on and distill_cfg is not None and self.shared_teacher is not None:
                    losses, step_toks = self._run_distill_step(
                        distill_cfg,
                        self.shared_teacher,
                        accum=accum,
                        mb=mb,
                        pass_pos=pass_pos,
                        depth=depth,
                    )
                    self.tokens_seen += step_toks
                    tok_log += step_toks
                elif lcalib_on and calib_every > 0 and ((self.step + 1) % calib_every == 0):
                    losses, step_toks = self._run_long_calib_step(lcalib_raw, aux_w=aux_w, pass_pos=pass_pos)
                    self.tokens_seen += step_toks
                    tok_log += step_toks
                else:
                    for k in range(accum):
                        batch = _to_device([next(self.stream) for _ in range(mb)], st.device, pass_pos)
                        self.opt.set_boundary(k == accum - 1)
                        seq_len_cur = int(batch["input_ids"].shape[1])
                        if bptt_runner is not None and seq_len_cur >= bptt_min_seq_len:
                            b_stats = self._run_bptt_microstep(bptt_runner, batch, accum=accum, aux_w=aux_w)
                            for bk, bv in b_stats.items():
                                losses[bk] = losses.get(bk, 0.0) + bv / accum
                            n = int(batch["input_ids"].numel())
                        else:
                            h = st.hidden(batch["input_ids"], batch.get("position_ids"))
                            loss, ntok = lm_loss(h, st.lm_head_weight, batch["labels"], impl=impl)
                            total = loss
                            for name, a in st.aux_losses().items():
                                total = total + float(aux_w.get(name, 1.0)) * a
                                losses[name] = losses.get(name, 0.0) + float(a.detach()) / accum
                            (total / accum).backward()
                            losses["lm"] = losses.get("lm", 0.0) + float(loss.detach()) / accum
                            n = int(ntok)
                        if lcalib_on:
                            losses["long_calib_active"] = 0.0
                        self.tokens_seen += int(batch["input_ids"].numel())
                        tok_log += int(batch["input_ids"].numel())
                        if not math.isfinite(losses["lm"]):
                            raise FloatingPointError(f"non-finite LM loss at step {self.step} (micro {k}, {n} tokens)")

                ostats = self.opt.step()
                self.step += 1
                if self.step % log_every == 0 or self.step == 1:
                    dt = time.time() - t_log
                    last = {
                        **{f"loss/{k}": v for k, v in losses.items()},
                        "depth": depth,
                        "tok_per_s": tok_log / max(dt, 1e-6),
                        "tokens_seen": self.tokens_seen,
                        "lr_factor": self.opt.lr_factor,
                        **{f"optim/{k}": v for k, v in ostats.items()},
                        **st.stats(),
                    }
                    if torch.cuda.is_available():
                        last["mem/peak_gib"] = torch.cuda.max_memory_allocated() / 2**30
                    log.info(
                        "step %d/%d lm %.4f d=%d %.0f tok/s %s",
                        self.step,
                        steps,
                        losses["lm"],
                        depth,
                        last["tok_per_s"],
                        cuda_mem_brief(),
                    )
                    self.metrics.log(self.step, last)
                    t_log, tok_log = time.time(), 0
                if eval_every and self.step % eval_every == 0:
                    ev = self.quick_eval()
                    last.update(ev)
                    self.metrics.log(self.step, ev)
                    if tok_log == 0:
                        t_log = time.time()
                if self.step % ckpt_every == 0:
                    self.save(metrics=last)
                    if tok_log == 0:
                        t_log = time.time()
                if budget and time.time() - t_start > budget:
                    log.warning("time budget %.1fh reached: checkpointing and stopping", budget / 3600)
                    break
        except KeyboardInterrupt:
            log.warning("interrupted: saving checkpoint before exit")
            self.save(metrics=last)
            raise
        if hasattr(self.stream, "log_stats"):
            self.stream.log_stats()
        if st.frozen_experts_sha256 is not None:
            st.verify_frozen_experts()
        final = self.quick_eval() if self.eval_batches else {}
        if final:
            self.metrics.log(self.step, final)
        path = self.save(completed=(self.step >= steps), metrics={**last, **final})
        return {"step": self.step, "checkpoint": path, "last": last, "eval": final}
