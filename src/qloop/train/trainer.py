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
import copy
import inspect
import json
import logging
import math
import os
import queue
import random
import signal
import threading
import time
from typing import Any

import torch

from ..debug import cuda_mem_brief, require, stage
from ..utils.tf32 import configure_tf32
from .bptt import BPTTConfig, SegmentedKVBPTTRunner
from .checkpoint import CheckpointManager, deserialize_py_rng_state, serialize_py_rng_state
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


_PREFETCH_STOP = object()
_PREFETCH_ERR = object()


class _AsyncBatchPrefetcher:
    """Single-thread deterministic background prefetcher with pinned-memory H2D prep and exact state tracking.

    Overlaps CPU data streaming, 13-gram decontamination, tokenization, and sequence packing with GPU
    compute while preserving 100% bit-exact `stream.state_dict()` checkpoint/resume determinism:
    `state_dict()` always returns the snapshot captured immediately after the last *consumed* item,
    never the prefetched-ahead cursor.
    """

    def __init__(self, stream: Any, *, maxsize: int = 2, pin_memory: bool = False) -> None:
        self._stream = stream
        self._pin_memory = bool(pin_memory and torch.cuda.is_available())
        self._has_state = callable(getattr(stream, "state_dict", None))
        self._last_state: Any = copy.deepcopy(stream.state_dict()) if self._has_state else None
        self._q: queue.Queue[Any] = queue.Queue(maxsize=max(1, int(maxsize)))
        self._stop_ev = threading.Event()
        self._closed = False
        self._thread = threading.Thread(
            target=self._worker,
            name="qloop-data-prefetch",
            daemon=True,
        )
        self._thread.start()

    def _put_or_stop(self, payload: Any) -> bool:
        while not self._stop_ev.is_set():
            try:
                self._q.put(payload, timeout=0.05)
                return True
            except queue.Full:
                continue
        return False

    def _worker(self) -> None:
        try:
            while not self._stop_ev.is_set():
                try:
                    item = next(self._stream)
                except StopIteration:
                    self._put_or_stop((_PREFETCH_STOP, None))
                    return
                sd = copy.deepcopy(self._stream.state_dict()) if self._has_state else None
                if self._pin_memory and isinstance(item, dict):
                    pinned: dict[str, Any] = {}
                    for k, v in item.items():
                        if isinstance(v, torch.Tensor) and v.device.type == "cpu" and not v.is_pinned():
                            pinned[k] = v.pin_memory()
                        else:
                            pinned[k] = v
                    item = pinned
                if not self._put_or_stop((item, sd)):
                    return
        except BaseException as exc:
            self._put_or_stop((_PREFETCH_ERR, exc))

    def next_item(self) -> Any:
        item, sd = self._q.get()
        if item is _PREFETCH_STOP:
            raise StopIteration
        if item is _PREFETCH_ERR:
            raise sd
        self._last_state = sd
        return item

    def state_dict(self) -> Any:
        return copy.deepcopy(self._last_state) if self._last_state is not None else None

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._stop_ev.set()
        while True:
            try:
                self._q.get_nowait()
            except queue.Empty:
                break
        if self._thread.is_alive():
            self._thread.join(timeout=5.0)


def _to_device(batch: list[dict[str, Any]], device: torch.device, pass_pos: bool) -> dict[str, torch.Tensor]:
    is_cuda = device.type == "cuda"
    if len(batch) == 1:
        ids_cpu = batch[0]["input_ids"].unsqueeze(0)
        lbl_cpu = batch[0]["labels"].unsqueeze(0)
    else:
        ids_cpu = torch.stack([b["input_ids"] for b in batch])
        lbl_cpu = torch.stack([b["labels"] for b in batch])
    if is_cuda:
        if ids_cpu.device.type == "cpu" and not ids_cpu.is_pinned():
            ids_cpu = ids_cpu.pin_memory()
        if lbl_cpu.device.type == "cpu" and not lbl_cpu.is_pinned():
            lbl_cpu = lbl_cpu.pin_memory()
    ids = ids_cpu.to(device, non_blocking=is_cuda)
    labels = lbl_cpu.to(device, non_blocking=is_cuda)
    out = {"input_ids": ids, "labels": labels}
    if pass_pos and "position_ids" in batch[0]:
        pos_cpu = (
            batch[0]["position_ids"].unsqueeze(0)
            if len(batch) == 1
            else torch.stack([b["position_ids"] for b in batch])
        )
        if is_cuda and pos_cpu.device.type == "cpu" and not pos_cpu.is_pinned():
            pos_cpu = pos_cpu.pin_memory()
        out["position_ids"] = pos_cpu.to(device, non_blocking=is_cuda)
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
        self.resume_mode = "fresh"
        self._stop = False
        self.stream: Any = None
        self._prefetcher: _AsyncBatchPrefetcher | None = None
        self.eval_batches: list[dict[str, Any]] = []
        self.ood_eval_batches: list[dict[str, Any]] = []
        self.shared_teacher: SharedBackboneTeacher | None = None

    def _next_stream_item(self) -> Any:
        if self._prefetcher is not None:
            return self._prefetcher.next_item()
        return next(self.stream)

    def _start_prefetcher(self) -> None:
        prefetch_n = int(self.tcfg.get("prefetch_batches", 2))
        if prefetch_n > 0 and self.stream is not None and self._prefetcher is None:
            pin_mem = bool(self.stack.device.type == "cuda")
            self._prefetcher = _AsyncBatchPrefetcher(self.stream, maxsize=prefetch_n, pin_memory=pin_mem)

    def _stop_prefetcher(self) -> None:
        if self._prefetcher is not None:
            self._prefetcher.close()

    # ------------------------------------------------------------------ data
    def _holdout_cfg(self) -> tuple[int, int, bool, bool]:
        dcfg = dict(self.cfg.get("data") or {})
        ecfg = dict(self.cfg.get("eval_quick") or {})
        n_eval = int(ecfg.get("n_batches", 0))
        explicit_holdout = (
            "holdout_mod" in dcfg
            or "holdout_eval_buckets" in dcfg
            or "holdout_mod" in ecfg
            or "holdout_eval_buckets" in ecfg
        )
        holdout_mod = int(dcfg.get("holdout_mod", ecfg.get("holdout_mod", 100)))
        holdout_eval_buckets = int(dcfg.get("holdout_eval_buckets", ecfg.get("holdout_eval_buckets", 2)))
        active = n_eval > 0 and holdout_mod > 0 and holdout_eval_buckets > 0
        return holdout_mod, holdout_eval_buckets, active, explicit_holdout

    @staticmethod
    def _apply_holdout_to_cfg(
        dcfg: dict[str, Any],
        *,
        role: str,
        holdout_mod: int,
        holdout_eval_buckets: int,
        active: bool,
        explicit_holdout: bool,
    ) -> tuple[dict[str, Any], str]:
        out = dict(dcfg)
        out["holdout_mod"] = holdout_mod
        out["holdout_eval_buckets"] = holdout_eval_buckets
        if not active:
            return out, str(out.get("split_role", "all"))
        patched_sources: list[Any] = []
        any_partitioned = False
        for src in list(out.get("sources") or []):
            if not isinstance(src, dict):
                patched_sources.append(src)
                continue
            s = dict(src)
            if "split_role" not in s:
                src_explicit = "holdout_mod" in s or "holdout_eval_buckets" in s or explicit_holdout
                src_mod = int(s.get("holdout_mod", holdout_mod))
                is_synthetic_toy = s.get("name") in ("toy_text", "toy_cot")
                if src_mod <= 0 or (is_synthetic_toy and not src_explicit):
                    s["split_role"] = "all"
                else:
                    s["split_role"] = role
            if s.get("split_role") == role:
                any_partitioned = True
            patched_sources.append(s)
        out["sources"] = patched_sources
        eff_role = role if any_partitioned else str(out.get("split_role", "all"))
        return out, eff_role

    def _build_stream(self):
        from ..data.mixture import build_train_stream

        dcfg = dict(self.cfg.get("data") or {})
        require(bool(dcfg.get("sources")), "data.sources is empty")
        h_mod, h_eval, active, explicit_holdout = self._holdout_cfg()
        stream_cfg, eff_role = self._apply_holdout_to_cfg(
            dcfg,
            role="train",
            holdout_mod=h_mod,
            holdout_eval_buckets=h_eval,
            active=active,
            explicit_holdout=explicit_holdout,
        )
        return build_train_stream(
            stream_cfg,
            self.stack.tokenizer,
            seed=self.seed,
            split_role=eff_role,
            holdout_mod=h_mod,
            holdout_eval_buckets=h_eval,
        )

    @staticmethod
    def _collect_eval_batches(
        stream: Any,
        n: int,
        *,
        label: str,
        split_role: str,
        holdout_mod: int,
        holdout_eval_buckets: int,
    ) -> list[dict[str, Any]]:
        batches: list[dict[str, Any]] = []
        for _ in range(n):
            try:
                batches.append(next(stream))
            except StopIteration:
                break
            except RuntimeError as exc:
                if "produced no items" not in str(exc):
                    raise
                log.warning(
                    "[eval] %s exhausted with empty split (%s; split=%s, holdout=%d/%d)",
                    label,
                    exc,
                    split_role,
                    holdout_eval_buckets,
                    holdout_mod,
                )
                break
        if len(batches) < n:
            log.warning(
                "[eval] %s produced %d/%d batches (split=%s, holdout=%d/%d); "
                "using available held-out batches without wrapping",
                label,
                len(batches),
                n,
                split_role,
                holdout_eval_buckets,
                holdout_mod,
            )
        return batches

    def _eval_set(self) -> list[dict[str, Any]]:
        from ..data.mixture import build_train_stream

        ecfg = dict(self.cfg.get("eval_quick") or {})
        n = int(ecfg.get("n_batches", 0))
        if n <= 0:
            return []
        dcfg = dict(self.cfg.get("data") or {})
        dcfg.update(ecfg.get("data_overrides") or {})
        h_mod, h_eval, active, explicit_holdout = self._holdout_cfg()
        eval_cfg, eff_role = self._apply_holdout_to_cfg(
            dcfg,
            role="eval",
            holdout_mod=h_mod,
            holdout_eval_buckets=h_eval,
            active=active,
            explicit_holdout=explicit_holdout,
        )
        stream = build_train_stream(
            eval_cfg,
            self.stack.tokenizer,
            seed=self.seed + 7_777_777,
            split_role=eff_role,
            holdout_mod=h_mod,
            holdout_eval_buckets=h_eval,
            single_pass=True,
            skip_empty_sources=True,
        )
        return self._collect_eval_batches(
            stream,
            n,
            label="eval_quick",
            split_role=eff_role,
            holdout_mod=h_mod,
            holdout_eval_buckets=h_eval,
        )

    def _ood_eval_set(self) -> list[dict[str, Any]]:
        from ..data.mixture import build_train_stream

        ecfg = dict(self.cfg.get("eval_quick") or {})
        ood_sources = ecfg.get("ood_sources")
        if not ood_sources:
            return []
        n = int(ecfg.get("ood_n_batches", ecfg.get("n_batches", 0)))
        if n <= 0:
            return []
        dcfg = dict(self.cfg.get("data") or {})
        dcfg.update(ecfg.get("data_overrides") or {})
        dcfg["sources"] = [dict(s) if isinstance(s, dict) else s for s in ood_sources]
        train_names = {
            s.get("name") for s in (self.cfg.get("data") or {}).get("sources") or [] if isinstance(s, dict)
        }
        h_mod, h_eval, active, explicit_holdout = self._holdout_cfg()
        patched: list[Any] = []
        any_eval = False
        for src in dcfg["sources"]:
            if not isinstance(src, dict):
                patched.append(src)
                continue
            s = dict(src)
            if "split_role" not in s:
                if s.get("name") in train_names and h_mod > 0 and h_eval > 0 and (
                    active or explicit_holdout
                ):
                    s["split_role"] = "eval"
                    any_eval = True
                else:
                    s["split_role"] = str(ecfg.get("ood_split_role", "all"))
            elif s["split_role"] == "eval":
                any_eval = True
            patched.append(s)
        dcfg["sources"] = patched
        eff_role = "eval" if any_eval else str(ecfg.get("ood_split_role", "all"))
        stream = build_train_stream(
            dcfg,
            self.stack.tokenizer,
            seed=self.seed + 8_888_888,
            split_role=eff_role,
            holdout_mod=h_mod,
            holdout_eval_buckets=h_eval,
            single_pass=True,
            skip_empty_sources=True,
        )
        return self._collect_eval_batches(
            stream,
            n,
            label="eval_quick.ood",
            split_role=eff_role,
            holdout_mod=h_mod,
            holdout_eval_buckets=h_eval,
        )

    def _next_long_calib_batch(self, total_len: int, pass_pos: bool) -> dict[str, torch.Tensor]:
        id_parts: list[torch.Tensor] = []
        lbl_parts: list[torch.Tensor] = []
        cur_len = 0
        while cur_len < total_len:
            item = self._next_stream_item()
            ids_1d = item["input_ids"]
            lbl_1d = item["labels"]
            id_parts.append(ids_1d)
            lbl_parts.append(lbl_1d)
            cur_len += int(ids_1d.numel())
        is_cuda = self.stack.device.type == "cuda"
        full_ids_cpu = torch.cat(id_parts, dim=-1)[:total_len].unsqueeze(0)
        full_lbl_cpu = torch.cat(lbl_parts, dim=-1)[:total_len].unsqueeze(0)
        if is_cuda:
            if full_ids_cpu.device.type == "cpu" and not full_ids_cpu.is_pinned():
                full_ids_cpu = full_ids_cpu.pin_memory()
            if full_lbl_cpu.device.type == "cpu" and not full_lbl_cpu.is_pinned():
                full_lbl_cpu = full_lbl_cpu.pin_memory()
        full_ids = full_ids_cpu.to(self.stack.device, non_blocking=is_cuda)
        full_lbl = full_lbl_cpu.to(self.stack.device, non_blocking=is_cuda)
        out = {"input_ids": full_ids, "labels": full_lbl}
        if pass_pos:
            out["position_ids"] = torch.arange(total_len, device=self.stack.device, dtype=torch.long).unsqueeze(0)
        return out

    # ------------------------------------------------------------------ resume / save
    def _state(self) -> dict[str, Any]:
        if self._prefetcher is not None:
            data_state = self._prefetcher.state_dict()
        elif self.stream is not None and callable(getattr(self.stream, "state_dict", None)):
            data_state = self.stream.state_dict()
        else:
            data_state = None
        return {
            "stage": self.cfg.get("stage"),
            "tokens_seen": self.tokens_seen,
            "stack": self.stack.describe(),
            "data": data_state,
            "trainer_rng": serialize_py_rng_state(self.rng),
            "resume_mode": self.resume_mode,
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
            extra_rng={"trainer_rng": self.rng.getstate()},
        )

    def maybe_resume(self) -> None:
        from ..hub import resolve_run_path

        run_root = os.path.dirname(os.path.abspath(self.run_dir))
        init = resolve_run_path(self.tcfg.get("init_from"), run_root=run_root)
        if self.tcfg.get("resume", True) and self.ckpt.latest():
            ck = self.ckpt.load(map_location="cpu", restore_rng=True)
            assert ck is not None
            st = ck["state"]
            smeta = st.get("stack") or {}
            sc = float((smeta.get("expert_delta_scale")) or 1.0)
            self.stack.restore_checkpoint(ck["params"], strict=True, expert_delta_scale=sc, prior_stack_meta=smeta)
            self.step, self.tokens_seen = int(st["step"]), int(st.get("tokens_seen", 0))
            if ck["optim"] is not None:
                self.opt.load_state_dict(ck["optim"])
                self.resume_mode = "exact"
            else:
                self.resume_mode = "weights_only"
                reset_lr = bool(self.tcfg.get("reset_lr_on_weights_only", False))
                if not reset_lr:
                    self.opt.sync_step(self.step)
                log.warning(
                    "weights-only resume at step %d (%s): optim.pt is absent - optimizer moments start fresh "
                    "(resume_mode=%r, reset_lr_on_weights_only=%s, opt.step_count=%d, lr_factor=%.4f)",
                    self.step,
                    ck["path"],
                    self.resume_mode,
                    reset_lr,
                    self.opt.step_count,
                    self.opt.lr_factor,
                )
            trainer_rng = st.get("trainer_rng")
            if trainer_rng is None and isinstance(ck.get("rng"), dict):
                trainer_rng = ck["rng"].get("trainer_rng")
            if trainer_rng is not None:
                self.rng.setstate(deserialize_py_rng_state(trainer_rng))
            if st.get("data") is not None and self.stream is not None:
                self.stream.load_state_dict(st["data"])
            log.info(
                "resumed at step %d (%.2fM tokens seen, mode=%s, lr_factor=%.4f)",
                self.step,
                self.tokens_seen / 1e6,
                self.resume_mode,
                self.opt.lr_factor,
            )
        elif init:
            ck = (
                CheckpointManager(init, keep_last=0).load(map_location="cpu", restore_rng=False)
                if os.path.isdir(init)
                else None
            )
            require(ck is not None, "train.init_from: no checkpoint found", init_from=init)
            assert ck is not None
            smeta = (ck.get("state") or {}).get("stack") or {}
            sc = float((smeta.get("expert_delta_scale")) or 1.0)
            counts = self.stack.restore_checkpoint(
                ck["params"],
                strict=False,
                expert_delta_scale=sc,
                prior_stack_meta=smeta,
                drop_inactive_loop_adapters=bool(self.tcfg.get("drop_inactive_loop_adapters", False)),
            )
            self.resume_mode = "init_from"
            log.info("initialised from %s (%s; optimizer fresh)", ck["path"], counts)

    # ------------------------------------------------------------------ eval
    @torch.no_grad()
    def _eval_batch_list(self, batches: list[dict[str, Any]], depths: list[int], prefix: str) -> dict[str, float]:
        if not batches:
            return {}
        st, out = self.stack, {}
        pass_pos = bool(self.tcfg.get("pass_position_ids", False))
        for d in depths:
            if st.loop is not None:
                st.loop.set_active_passes(int(d))
            elif d != 1:
                continue
            tot, ntok = 0.0, 0
            for b in batches:
                bt = _to_device([b], st.device, pass_pos)
                h = st.hidden(bt["input_ids"], bt.get("position_ids"))
                loss, n = lm_loss(h, st.lm_head_weight, bt["labels"], impl="chunked")
                tot += float(loss) * int(n)
                ntok += int(n)
            out[f"{prefix}loss_d{d}"] = tot / max(ntok, 1)
        if f"{prefix}loss_d1" in out and len(depths) > 1:
            out[f"{prefix}depth_gain"] = out[f"{prefix}loss_d1"] - out[f"{prefix}loss_d{max(depths)}"]
        return out

    @torch.no_grad()
    def quick_eval(self) -> dict[str, float]:
        if not self.eval_batches and not self.ood_eval_batches:
            return {}
        st, out = self.stack, {}
        default_depths = [1, st.loop.spec.num_passes] if st.loop else [1]
        depths = [int(d) for d in ((self.cfg.get("eval_quick") or {}).get("depths") or default_depths)]
        was_training = st.model.training
        st.model.eval()
        prev = st.loop.active_passes if st.loop else None
        try:
            if self.eval_batches:
                out.update(self._eval_batch_list(self.eval_batches, depths, "eval/"))
            if self.ood_eval_batches:
                out.update(self._eval_batch_list(self.ood_eval_batches, depths, "eval/ood_"))
        finally:
            if st.loop is not None:
                if prev is not None:
                    st.loop.set_active_passes(prev)
                if callable(getattr(st.loop, "release_graph", None)):
                    st.loop.release_graph(clear_routing=True)
            if st.lsa is not None:
                st.lsa.aux_loss()  # drop KL terms produced in eval
                st.lsa.clear_cache()
            if st.hssd is not None:
                for ad in st.hssd.adapters.values():
                    ad.clear_aux()
            st.model.train(was_training)
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
                batch = _to_device([self._next_stream_item() for _ in range(mb)], st.device, pass_pos)
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
            batch = _to_device([self._next_stream_item() for _ in range(mb)], st.device, pass_pos)
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
        sig = inspect.signature(bptt_runner.run_step)
        supports_loss_scale = "loss_scale" in sig.parameters

        scale_hooks: list[Any] = []
        extra_kw: dict[str, Any] = {}
        if supports_loss_scale:
            extra_kw["loss_scale"] = 1.0 / float(max(1, accum))
        elif accum > 1:
            inv_acc = 1.0 / float(accum)
            for p in st.named_trainable.values():
                if p.requires_grad:
                    scale_hooks.append(p.register_hook(lambda g, s=inv_acc: g * s))

        try:
            with defer_micro_step(self.opt):
                res = bptt_runner.run_step(
                    batch["input_ids"],
                    labels=batch["labels"],
                    loss_fn=None,
                    backward=True,
                    hssd_aux_weight=float(aux_w.get("hssd_budget", 1.0)) if st.hssd is not None else 0.0,
                    lsa_aux_weight=float(aux_w.get("lsa_kl", 1.0)),
                    loop_aux_weight=float(aux_w.get("loop_aux", 1.0)),
                    **extra_kw,
                )
        finally:
            for h in scale_hooks:
                h.remove()
            if st.hssd is not None:
                for ad in st.hssd.adapters.values():
                    ad.clear_aux()

        out: dict[str, float] = {
            "lm": float(res.ce_loss),
            "aux": float(res.aux_loss),
            "total": float(res.total_loss),
            "bptt_segments": float(res.num_segments),
            "bptt_replayed_tokens": float(res.stats.get("bptt/mean_replayed_tokens", 0.0)),
            "bptt_offload_to_cpu": float(res.stats.get("bptt/offload_to_cpu", 0.0)),
        }
        for comp_k, stat_k in (
            ("lsa_kl", "bptt/lsa_kl"),
            ("loop_aux", "bptt/loop_aux"),
            ("hssd_aux", "bptt/hssd_aux"),
            ("replay_aux", "bptt/replay_aux"),
        ):
            val = float(res.stats.get(stat_k, 0.0))
            if val != 0.0:
                out[comp_k] = val
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
            self.ood_eval_batches = self._ood_eval_set()
        self.maybe_resume()
        if st.loop is not None and callable(getattr(st.loop, "set_step", None)):
            st.loop.set_step(self.step, steps)
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
        has_any_eval = bool(self.eval_batches or self.ood_eval_batches)
        if self.step == 0 and (eval_every or (check_step0 and has_any_eval)):
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
                        "(uniform random vocab loss is ~12.42); base model weights appear corrupted, "
                        "tokenizer vocabulary is mismatched, or an unsupported third-party quantizer altered scales. "
                        "Verify model.name (default unsloth/Qwen3.6-35B-A3B-NVFP4) and cached safetensors shards, "
                        "or set train.max_step0_loss=0 to disable this check."
                    )
        # Reset throughput timer AFTER step-0 quick_eval() so initial eval duration never pollutes step-1 tok/s.
        t_log, tok_log = time.time(), 0
        last: dict[str, Any] = {}
        self.opt.zero_grad(set_to_none=True)
        self._start_prefetcher()
        try:
            while self.step < steps and not self._stop:
                if st.lsa is not None and callable(getattr(st.lsa, "set_step", None)):
                    st.lsa.set_step(self.step)
                if st.loop is not None and callable(getattr(st.loop, "set_step", None)):
                    st.loop.set_step(self.step, steps)
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
                        batch = _to_device([self._next_stream_item() for _ in range(mb)], st.device, pass_pos)
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
                            aux_total = 0.0
                            for name, a in st.aux_losses().items():
                                w_a = float(aux_w.get(name, 1.0))
                                total = total + w_a * a
                                a_val = float(a.detach())
                                losses[name] = losses.get(name, 0.0) + a_val / accum
                                aux_total += w_a * a_val
                            (total / accum).backward()
                            if st.loop is not None and callable(getattr(st.loop, "release_graph", None)):
                                st.loop.release_graph()
                            losses["lm"] = losses.get("lm", 0.0) + float(loss.detach()) / accum
                            losses["aux"] = losses.get("aux", 0.0) + aux_total / accum
                            n = int(ntok)
                        if lcalib_on:
                            losses["long_calib_active"] = 0.0
                        self.tokens_seen += int(batch["input_ids"].numel())
                        tok_log += int(batch["input_ids"].numel())
                        if not math.isfinite(losses["lm"]) or not math.isfinite(losses.get("aux", 0.0)):
                            raise FloatingPointError(f"non-finite LM loss at step {self.step} (micro {k}, {n} tokens)")

                ostats = self.opt.step()
                self.opt.zero_grad(set_to_none=True)
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
                if ckpt_every and self.step % ckpt_every == 0:
                    self.save(metrics=last)
                    if tok_log == 0:
                        t_log = time.time()
                if budget and time.time() - t_start > budget:
                    log.warning("time budget %.1fh reached: checkpointing and stopping", budget / 3600)
                    break
        except KeyboardInterrupt:
            log.warning("interrupted: saving checkpoint before exit")
            self._stop_prefetcher()
            self.save(metrics=last)
            raise
        finally:
            self._stop_prefetcher()
        if hasattr(self.stream, "log_stats"):
            self.stream.log_stats()
        if st.frozen_experts_sha256 is not None:
            st.verify_frozen_experts()
        if st.loop is not None and callable(getattr(st.loop, "set_step", None)):
            st.loop.set_step(self.step, steps)
        final = self.quick_eval() if (self.eval_batches or self.ood_eval_batches) else {}
        if final:
            self.metrics.log(self.step, final)
        path = self.save(completed=(self.step >= steps), metrics={**last, **final})
        self._prefetcher = None
        return {"step": self.step, "checkpoint": path, "last": last, "eval": final, "resume_mode": self.resume_mode}
