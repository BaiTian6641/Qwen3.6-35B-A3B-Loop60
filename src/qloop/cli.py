"""`python -m qloop <command>` - the only entry point the Colab notebook calls.

Every command runs in a fresh process (so CUDA_LAUNCH_BLOCKING & friends apply and a crash
never poisons the notebook kernel), gets a run directory, a log file and a crash report.

  doctor        environment table (+ --freeze lockfile, --datasets access check)
  probe         --backward | --core-ranges | --surgery | --structured
  scan-experts  ESFT router-affinity scan -> experts json (feeds experts.stats_path)
  train         LM stages (loop heal / HSSD / SFT / long-context + D-LSA) or RL stage (GRPO/RSI)
  eval          benchmark suite with the loop-depth sweep
  infer         OptimizedLoopInferenceEngine on a shipped Whole Model Pack
  sync          HF Hub commit-history sync, 230 GB disk guard & self-resuming stage pipeline
  serve         OpenAI-compatible server for external agent harnesses (Harbor, tau-bench)
  external      run Harbor / tau-bench against a served endpoint
  export        trainable adapters bundle, optional merged depth-1 HF checkpoint
  export-pack   self-contained sharded safetensors Whole Model Pack (+ low-disk HF streaming)
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from typing import Any, NoReturn

from .config import Cfg, load_config
from .debug import crash_guard, require, setup_run, stage

log = logging.getLogger("qloop")


def _run_root(cfg: Cfg | None = None) -> str:
    return os.environ.get("QLOOP_RUN_ROOT") or (cfg.get_path("run.root") if cfg else None) or "runs"


def _run_dir(cfg: Cfg | None, name: str) -> str:
    rname = (cfg.get_path("run.name") if cfg else None) or name
    return os.path.join(_run_root(cfg), rname)


def _load(args: argparse.Namespace) -> Cfg:
    return load_config(args.config, args.set)


def _common(p: argparse.ArgumentParser, config: bool = True) -> None:
    if config:
        p.add_argument("--config", "-c", required=True, help="YAML config (supports _base_ inheritance)")
    p.add_argument("--set", "-s", action="append", default=[], metavar="KEY=VALUE", help="dotted override")
    p.add_argument("--debug", type=int, default=None, help="debug level 0-3 (overrides debug.level)")


def _resolve_allow_tf32(cfg: Cfg | None) -> bool:
    if cfg is None:
        return True
    val = cfg.get_path("train.allow_tf32")
    if val is None:
        val = cfg.get_path("model.allow_tf32")
    return True if val is None else bool(val)


def _start(cfg: Cfg | None, name: str, args: argparse.Namespace) -> str:
    run_dir = _run_dir(cfg, name)
    level = args.debug if args.debug is not None else int((cfg.get_path("debug.level") if cfg else None) or 1)
    setup_run(run_dir, level)
    if name != "external":
        from .utils.tf32 import configure_tf32

        configure_tf32(_resolve_allow_tf32(cfg))
    if cfg is not None:
        with open(os.path.join(run_dir, f"config_{int(time.time())}.json"), "w", encoding="utf-8") as f:
            json.dump(cfg.to_dict(), f, indent=2)
    return run_dir


def _hub_for_cfg(cfg: Cfg | None):
    from .hub import HubSyncManager

    sync_cfg = (cfg.get("checkpoint") or {}).get("sync") if cfg is not None else None
    return HubSyncManager.from_sync_cfg(sync_cfg)


def _stack_with_ckpt(cfg: Cfg, ckpt: str | None):
    from .hub import resolve_run_path
    from .train.checkpoint import CheckpointManager
    from .train.stack import build_stack

    run_root = _run_root(cfg)
    resolved_ckpt = resolve_run_path(ckpt, run_root=run_root)
    hub = _hub_for_cfg(cfg)
    if hub is not None:
        hub.pull_artifacts(run_root)
        if resolved_ckpt and not os.path.exists(resolved_ckpt):
            ckpt_stage = os.path.basename(resolved_ckpt.rstrip("/"))
            hub.pull_stage_checkpoint(ckpt_stage, run_root, prefer_promoted=True, include_optim=False)
            resolved_ckpt = resolve_run_path(ckpt, run_root=run_root)

    st = build_stack(cfg)
    if resolved_ckpt:
        path = resolved_ckpt
        if os.path.isdir(os.path.join(resolved_ckpt, "checkpoints")):
            mgr = CheckpointManager(resolved_ckpt, keep_last=0)
            path = mgr.promoted() or mgr.latest()
            require(path is not None, "no LATEST or PROMOTED checkpoint found", ckpt=resolved_ckpt)
        assert path is not None
        ck = CheckpointManager(os.path.dirname(os.path.dirname(path)), keep_last=0).load(path)
        require(ck is not None, "failed to load checkpoint", path=path)
        assert ck is not None
        smeta = (ck.get("state") or {}).get("stack") or {}
        sc = float((smeta.get("expert_delta_scale")) or 1.0)
        st.restore_checkpoint(ck["params"], strict=False, expert_delta_scale=sc, prior_stack_meta=smeta)
    for p in st.named_trainable.values():
        p.requires_grad_(False)
    return st


def _probe_ids(st: Any, cfg: Cfg, n: int, seq_len: int) -> list[Any]:
    from .data.mixture import build_train_stream

    dcfg = dict(cfg.get("data") or {})
    dcfg["seq_len"] = seq_len
    stream = build_train_stream(dcfg, st.tokenizer, seed=int(cfg.get("seed", 0)) + 99)
    return [next(stream)["input_ids"][None].to(st.device) for _ in range(n)]


# ---------------------------------------------------------------------------------------- commands

def cmd_doctor(args: argparse.Namespace) -> int:
    from .env.doctor import run_doctor

    setup_run(_run_dir(None, "doctor"), args.debug or 1)
    return run_doctor(args.freeze, args.datasets, args.model)


def cmd_probe(args: argparse.Namespace) -> int:
    from .model.loading import load_model
    from .model.probes import (
        probe_backward,
        probe_core_ranges,
        run_structured_probe_suite,
        surgery_check,
    )

    cfg = _load(args)
    run_dir = _start(cfg, "probe", args)
    out: dict[str, Any] = {}
    with crash_guard(run_dir):
        model, tok, _ = load_model(dict(cfg.get("model") or {}))
        from .model.introspect import inspect_model
        from .train.stack import Stack

        st = Stack(model, tok, inspect_model(model))
        seq = int(args.seq_len)
        if getattr(args, "structured", False):
            with stage("probe:structured"):
                tgt = int(args.target_virtual or cfg.get_path("loop.target_virtual_layers") or 88)
                batches = _probe_ids(st, cfg, int(args.n), seq)
                rep = run_structured_probe_suite(model, batches, target_virtual=tgt)
            json_path = args.out_json or os.path.join(run_dir, "probe_report.json")
            md_path = args.out_md or os.path.join(run_dir, "probe_report.md")
            os.makedirs(os.path.dirname(os.path.abspath(json_path)), exist_ok=True)
            os.makedirs(os.path.dirname(os.path.abspath(md_path)), exist_ok=True)
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(rep, f, indent=2, default=str)
            with open(md_path, "w", encoding="utf-8") as f:
                f.write(str(rep.get("markdown") or ""))
            log.info("structured probe report -> %s | %s", json_path, md_path)
            hub = _hub_for_cfg(cfg)
            if hub is not None:
                hub.push_artifacts(
                    "s0_probe",
                    {
                        "artifacts/probes/probe_report.json": json_path,
                        "artifacts/probes/probe_report.md": md_path,
                    },
                    completed=False,
                )
            if not (args.backward or args.core_ranges or args.surgery):
                return 0
            out["structured"] = rep
        if args.backward:
            with stage("probe:backward"):
                out["backward"] = probe_backward(model, _probe_ids(st, cfg, 1, seq)[0])
        if args.core_ranges:
            with stage("probe:core-ranges"):
                tgt_cr = int(args.target_virtual or cfg.get_path("loop.target_virtual_layers") or 64)
                rows = probe_core_ranges(model, _probe_ids(st, cfg, args.n, seq), target_virtual=tgt_cr)
                out["core_ranges"] = rows[:20]
        if args.surgery:
            with stage("probe:surgery"):
                loop = {k: v for k, v in dict(cfg.get("loop") or {}).items() if k != "enabled"}
                lsa = {k: v for k, v in dict(cfg.get("lsa") or {}).items() if k != "enabled"}
                gdn2_raw = dict(cfg.get("gdn2") or {})
                gdn2 = {k: v for k, v in gdn2_raw.items() if k != "enabled"} if gdn2_raw.get("enabled") else None
                ids = _probe_ids(st, cfg, 1, seq)[0]
                out["surgery"] = surgery_check(model, ids, loop or None, lsa or None, gdn2_cfg=gdn2)
        probe_path = os.path.join(run_dir, "probe.json")
        with open(probe_path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, default=str)
        log.info("probe results -> %s", probe_path)
        hub = _hub_for_cfg(cfg)
        if hub is not None:
            hub.push_artifacts("s0_probe", {"artifacts/s0_probe/probe.json": probe_path}, completed=False)
    return 0


def cmd_scan_experts(args: argparse.Namespace) -> int:
    from .hub import resolve_run_path
    from .model.experts import scan_router_affinity

    cfg = _load(args)
    run_dir = _start(cfg, "scan_experts", args)
    with crash_guard(run_dir):
        st = _stack_with_ckpt(cfg, args.ckpt)
        ids = _probe_ids(st, cfg, int(args.batches), int(args.seq_len))
        stats = scan_router_affinity(st.model, ids, controller=st.loop)
        stats.log_summary()
        out = (
            resolve_run_path(args.out, run_root=_run_root(cfg))
            if args.out
            else os.path.join(run_dir, "esft_stats.json")
        )
        assert out is not None
        os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
        stats.to_json(out)
        log.info("expert stats -> %s (use experts.stats_path=%s)", out, out)
        hub = _hub_for_cfg(cfg)
        if hub is not None:
            if args.ckpt:
                hub.push_artifacts(
                    "s1_rescan",
                    {"artifacts/s1_loop_heal/esft_stats_looped.json": out},
                    completed=True,
                )
            else:
                hub.push_artifacts(
                    "s0_probe",
                    {"artifacts/s0_probe/esft_stats.json": out},
                    completed=True,
                )
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    cfg = _load(args)
    run_dir = _start(cfg, cfg.get("stage", "train"), args)
    with crash_guard(run_dir, extra={"config": cfg.get_path("_meta_.config_path")}):
        kind = cfg.get("stage_type", "lm")
        if kind == "rl":
            from .train.rl import GRPOTrainer

            res = GRPOTrainer(cfg, run_dir).train()
        elif kind == "lm":
            from .train.trainer import LMTrainer

            res = LMTrainer(cfg, run_dir).train()
        else:
            raise ValueError(f"stage_type must be lm|rl, got {kind!r}")
        log.info("done: %s", json.dumps(res, default=str)[:2000])
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    from .eval.generate import Generator
    from .eval.runners import run_suite

    cfg = _load(args)
    run_dir = _start(cfg, "eval", args)
    with crash_guard(run_dir):
        st = _stack_with_ckpt(cfg, args.ckpt)
        gen = Generator(st.model, st.tokenizer, loop_controller=st.loop,
                        batch_size=int(cfg.get_path("eval.batch_size") or 1),
                        padded_batching=bool(cfg.get_path("eval.padded_batching") or False))
        if st.lsa is not None and cfg.get_path("eval.lsa_mode"):
            st.lsa.set_mode(cfg.get_path("eval.lsa_mode"))
        eval_dir = os.path.join(run_dir, "eval")
        res = run_suite(dict(cfg.get("eval") or {}), gen, eval_dir)
        log.info("suite summary: %s", json.dumps(res, default=str)[:4000])
        hub = _hub_for_cfg(cfg)
        if hub is not None:
            summ_path = os.path.join(eval_dir, "suite_summary.json")
            hub.push_artifacts(
                "eval",
                {"eval/suite_summary.json": summ_path},
                completed=True,
                eval_summary=res.get("benchmarks", res),
            )
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    from .eval.generate import Generator
    from .eval.server import serve

    cfg = _load(args)
    run_dir = _start(cfg, "serve", args)
    with crash_guard(run_dir):
        st = _stack_with_ckpt(cfg, args.ckpt)
        gen = Generator(st.model, st.tokenizer, loop_controller=st.loop)
        serve(gen, host=args.host, port=int(args.port), model_name=args.model_name, debug=(args.debug or 1) >= 2)
    return 0


def cmd_external(args: argparse.Namespace) -> int:
    from .eval import external as ext

    cfg = _load(args)
    run_dir = _start(cfg, "external", args)
    with crash_guard(run_dir):
        ecfg = dict(cfg.get_path("eval.external") or {})
        if args.harness == "harbor":
            run = ext.HarborRun(**dict(ecfg.get("harbor") or {}))
            chk = ext.check_harbor_env(env=run.env, model=run.model, api_base=run.api_base)
            chk.raise_if_failed()
            rc = ext.run_harbor(run, os.path.join(run_dir, "harbor"), dry_run=args.dry_run)
        else:
            run = ext.TauBenchRun(**dict(ecfg.get("tau") or {}))
            rc = ext.run_tau_bench(run, os.path.join(run_dir, "tau"), dry_run=args.dry_run)
        log.info("external harness finished: %s", rc)
    return 0


def cmd_export(args: argparse.Namespace) -> int:
    import torch

    cfg = _load(args)
    run_dir = _start(cfg, "export", args)
    with crash_guard(run_dir):
        st = _stack_with_ckpt(cfg, args.ckpt)
        os.makedirs(args.out, exist_ok=True)
        params = {n: t.detach().cpu() for n, t in st.checkpoint_params().items()}
        torch.save(params, os.path.join(args.out, "qloop_adapters.pt"))
        meta = {"base_model": cfg.get_path("model.name"), "stack": st.describe(), "config": cfg.to_dict()}
        with open(os.path.join(args.out, "qloop_config.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, default=str)
        log.info("adapter bundle -> %s (%d tensors)", args.out, len(params))
        if args.merged_depth1:
            # depth-1 == the stock Qwen3.6 graph: merge expert deltas, drop loop/LSA, save a vLLM-servable HF ckpt
            if st.deltas is not None:
                n = st.deltas.merge_into_base_()
                log.info("merged %d expert deltas into the base weights", n)
                st.deltas.detach()
            if st.loop is not None:
                st.loop.unwrap()
            if st.lsa is not None:
                st.lsa.uninstall()
            target = os.path.join(args.out, "merged_depth1")
            st.model.save_pretrained(target, safe_serialization=True)
            st.tokenizer.save_pretrained(target)
            log.info("merged depth-1 checkpoint -> %s (serve with vLLM; loop adapters NOT included)", target)
    return 0


def cmd_export_pack(args: argparse.Namespace) -> int:
    from .export_pack import export_whole_model_pack, push_whole_model_pack

    cfg = _load(args)
    run_dir = _start(cfg, "export_pack", args)
    with crash_guard(run_dir):
        require(
            bool(args.out or args.push_repo),
            "`qloop export-pack` requires at least one of `--out <dir>` or `--push-repo <repo_id>`",
        )
        st = _stack_with_ckpt(cfg, args.ckpt)
        merge_hot = not bool(args.no_merge_hot_experts)
        max_shard = int(args.max_shard_bytes)
        if args.push_repo:
            if args.out and args.stream_low_disk:
                export_whole_model_pack(
                    st,
                    args.out,
                    tokenizer=st.tokenizer,
                    max_shard_size_bytes=max_shard,
                    merge_hot_experts=merge_hot,
                    merge_in_place=False,
                    repo_id=args.push_repo,
                )
            res = push_whole_model_pack(
                st,
                args.push_repo,
                stream_low_disk=bool(args.stream_low_disk),
                tokenizer=st.tokenizer,
                max_shard_size_bytes=max_shard,
                merge_hot_experts=merge_hot,
                private=bool(args.private),
                out_dir=None if args.stream_low_disk else args.out,
            )
            log.info("pushed Whole Model Pack -> hf://%s (%s shards)", args.push_repo, res.get("num_shards"))
        else:
            assert args.out is not None
            res = export_whole_model_pack(
                st,
                args.out,
                tokenizer=st.tokenizer,
                max_shard_size_bytes=max_shard,
                merge_hot_experts=merge_hot,
            )
            log.info("exported Whole Model Pack -> %s (%s shards)", args.out, res.get("num_shards"))
    return 0


def cmd_infer(args: argparse.Namespace) -> int:
    from .infer import InferenceConfig, OptimizedLoopInferenceEngine

    run_dir = _start(None, "infer", args)
    with crash_guard(run_dir):
        icfg = InferenceConfig(
            active_passes=args.depth,
            prefill_chunk_size=int(args.prefill_chunk_size),
            kv_cache_quant=str(args.kv_quant),
            temperature=float(args.temperature),
            max_new_tokens=int(args.max_new_tokens),
        )
        engine = OptimizedLoopInferenceEngine.from_pretrained(args.pack, cfg=icfg)
        if args.depth is not None:
            with engine.depth(int(args.depth)):
                gen = engine.generate(
                    args.prompt,
                    max_new_tokens=int(args.max_new_tokens),
                    temperature=float(args.temperature),
                )
        else:
            gen = engine.generate(
                args.prompt,
                max_new_tokens=int(args.max_new_tokens),
                temperature=float(args.temperature),
            )
        payload: dict[str, Any] = gen.to_dict() if hasattr(gen, "to_dict") else {"text": str(gen)}
        if args.benchmark:
            payload["benchmark"] = engine.benchmark()
        print(json.dumps(payload, indent=2, default=str))
    return 0


def _pipeline_stage_cmds(
    stage_name: str,
    *,
    config_root: str,
    run_root: str,
    repo_id: str,
    debug_level: int,
    overrides: list[str],
    time_budget_hours: float | None,
    scan_batches: int,
    scan_seq_len: int,
    all_stages: tuple[str, ...] = (),
) -> list[list[str]]:
    from .hub import _V3_STAGE_SET, resolve_stage_config_path

    py = [sys.executable, "-u", "-m", "qloop"]
    common = ["--debug", str(debug_level), "-s", f"checkpoint.sync.hf_repo={repo_id}"]
    for ov in overrides:
        common.extend(["-s", ov])
    train_extra = list(common)
    if time_budget_hours is not None:
        train_extra.extend(["-s", f"train.time_budget_hours={time_budget_hours}"])

    _v2_only_stages = {"s1_loop_heal", "s1_rescan", "s2_sft", "s3_long_lsa"}
    is_v2_pipeline = any(s in _v2_only_stages for s in all_stages)
    is_v3_pipeline = any(s in _V3_STAGE_SET for s in all_stages) or (
        not is_v2_pipeline and any(s in ("s4_rl", "eval") for s in all_stages)
    )

    if stage_name == "s0_probe":
        cpath = resolve_stage_config_path("s0_probe", config_root)
        out_stats = os.path.join(run_root, "s0_probe", "esft_stats.json")
        probe_flags = ["--structured"] if is_v3_pipeline else ["--backward", "--surgery", "--core-ranges"]
        return [
            [*py, "probe", "-c", cpath, *probe_flags, *common],
            [*py, "scan-experts", "-c", cpath, "--batches", str(scan_batches), "--seq-len", str(scan_seq_len),
             "--out", out_stats, *common],
        ]
    if stage_name == "s1_rescan":
        cpath = resolve_stage_config_path("s1_rescan", config_root)
        ckpt = os.path.join(run_root, "s1_loop_heal")
        out_stats = os.path.join(run_root, "s1_loop_heal", "esft_stats_looped.json")
        return [[*py, "scan-experts", "-c", cpath, "--ckpt", ckpt, "--batches", str(scan_batches),
                 "--seq-len", str(scan_seq_len), "--out", out_stats, *common]]
    if stage_name == "eval":
        cpath = resolve_stage_config_path("eval", config_root)
        eval_extra = list(common)
        if is_v3_pipeline:
            ov_keys = {ov.split("=", 1)[0].strip() for ov in overrides if "=" in ov}
            v3_eval_defaults = (
                ("model.yarn_target_len", "1048576"),
                ("loop.prelude_end", "12"),
                ("loop.core_end", "36"),
                ("loop.num_passes", "3"),
                ("loop.target_virtual_layers", "88"),
                ("loop.train_pass_probs", "[0.15,0.50,0.35]"),
                ("loop.bridge_mode", "subspace"),
                ("loop.bridge_rank", "256"),
                ("loop.bridge_init", "[0.05,0.85,0.10]"),
                ("loop.attn_adapter_rank", "64"),
                ("gdn2.enabled", "false"),
                ("hssd.enabled", "true"),
                ("hssd.chunk_size", "64"),
                ("hssd.window_chunks", "8"),
                ("hssd.num_pages", "8"),
                ("hssd.k_pages", "2"),
                ("hssd.theta_low_rank", "true"),
                ("hssd.surprise_mode", "routed"),
                ("hssd.state_dtype", "bfloat16"),
                ("hssd.segment_chunks", "4"),
                ("lsa.num_query_passes", "3"),
                ("eval.defaults.depths", "[1,2,3]"),
            )
            for k, v in v3_eval_defaults:
                if k not in ov_keys:
                    eval_extra.extend(["-s", f"{k}={v}"])
        return [[*py, "eval", "-c", cpath, "--ckpt", os.path.join(run_root, "s4_rl"), *eval_extra]]

    cpath = resolve_stage_config_path(stage_name, config_root)
    stage_extra = list(train_extra)
    if stage_name == "s4_rl":
        ov_keys = {ov.split("=", 1)[0].strip() for ov in overrides if "=" in ov}
        if is_v3_pipeline:
            v3_rl_defaults = [
                ("model.yarn_target_len", "1048576"),
                ("loop.prelude_end", "12"),
                ("loop.core_end", "36"),
                ("loop.num_passes", "3"),
                ("loop.target_virtual_layers", "88"),
                ("loop.train_pass_probs", "[0.10,0.35,0.55]"),
                ("loop.bridge_mode", "subspace"),
                ("loop.bridge_rank", "256"),
                ("loop.bridge_init", "[0.05,0.85,0.10]"),
                ("loop.attn_adapter_rank", "64"),
                ("gdn2.enabled", "false"),
                ("hssd.enabled", "true"),
                ("hssd.theta_low_rank", "true"),
                ("hssd.surprise_mode", "routed"),
                ("hssd.state_dtype", "bfloat16"),
                ("hssd.segment_chunks", "4"),
                ("lsa.enabled", "true"),
                ("lsa.mode", "sparse"),
                ("lsa.num_query_passes", "3"),
                ("rl.gate.depth", "3"),
                (
                    "train.allowed_groups",
                    '["loop_adapters","hssd_adapters","gdn2_adapters","lsa_indexers","router","shared_expert","deltanet","attention","norms","expert_deltas"]',
                ),
            ]
            prior_v3 = [s for s in all_stages if s in _V3_STAGE_SET]
            if "s2_joint_bptt" in all_stages:
                v3_rl_defaults.insert(0, ("train.init_from", "runs/s2_joint_bptt"))
            elif prior_v3:
                v3_rl_defaults.insert(0, ("train.init_from", f"runs/{prior_v3[-1]}"))
            for k, v in v3_rl_defaults:
                if k not in ov_keys:
                    stage_extra.extend(["-s", f"{k}={v}"])
            looped_a6b_stats = os.path.join(run_root, "s1_loop_a6b", "esft_stats_looped.json")
            if "experts.enabled" not in ov_keys:
                if os.path.isfile(looped_a6b_stats) and "experts.stats_path" not in ov_keys:
                    stage_extra.extend(["-s", "experts.stats_path=runs/s1_loop_a6b/esft_stats_looped.json"])
                else:
                    stage_extra.extend(["-s", "experts.enabled=false"])
        elif is_v2_pipeline:
            v2_rl_defaults = [
                ("train.init_from", "runs/s3_long_lsa"),
                ("loop.prelude_end", "8"),
                ("loop.core_end", "32"),
                ("loop.num_passes", "2"),
                ("loop.target_virtual_layers", "64"),
                ("loop.bridge_mode", "concat"),
                ("loop.bridge_rank", "0"),
                ("loop.bridge_init", "[0.6,0.2,0.2]"),
                ("loop.attn_adapter_rank", "0"),
                ("loop.train_pass_probs", "[0.1,0.9]"),
                ("gdn2.enabled", "true"),
                ("hssd.enabled", "false"),
                ("lsa.enabled", "false"),
                ("rl.gate.depth", "2"),
                ("eval_quick.depths", "[1,2]"),
            ]
            for k, v in v2_rl_defaults:
                if k not in ov_keys:
                    stage_extra.extend(["-s", f"{k}={v}"])
            looped_v2_stats = os.path.join(run_root, "s1_loop_heal", "esft_stats_looped.json")
            if "experts.enabled" not in ov_keys and os.path.isfile(looped_v2_stats):
                stage_extra.extend(["-s", "experts.enabled=true"])
                if "experts.stats_path" not in ov_keys:
                    stage_extra.extend(["-s", "experts.stats_path=runs/s1_loop_heal/esft_stats_looped.json"])
    return [[*py, "train", "-c", cpath, *stage_extra]]


def cmd_sync(args: argparse.Namespace) -> int:
    from .hub import DEFAULT_HF_REPO, HubSyncManager
    from .train.checkpoint import CheckpointManager

    run_root = _run_root(None)
    sync_dir = os.path.join(run_root, "sync")
    setup_run(sync_dir, args.debug or 1)
    with crash_guard(sync_dir):
        repo_id = args.repo or os.environ.get("QLOOP_HF_REPO") or DEFAULT_HF_REPO
        hub = HubSyncManager(repo_id=repo_id, private=False)
        return _run_sync_action(args, hub=hub, repo_id=repo_id, run_root=run_root, ckpt_cls=CheckpointManager)


def _run_sync_action(
    args: argparse.Namespace,
    *,
    hub: Any,
    repo_id: str,
    run_root: str,
    ckpt_cls: Any,
) -> int:
    from .hub import _V3_STAGE_SET, PIPELINE_ORDER, V3_PIPELINE_ORDER, disk_report, reclaim_disk_space

    if args.action == "disk":
        if args.clean:
            keep = tuple(s.strip() for s in (args.keep or "").split(",") if s.strip())
            res = reclaim_disk_space(run_root, keep_stages=keep, clean_hf_staging=True, clean_pip_cache=args.pip)
            print(json.dumps(res, indent=2))
        print(json.dumps(disk_report(run_root), indent=2))
        return 0

    if args.action == "status":
        manifest = hub.fetch_manifest()
        commits = hub.recent_commits(limit=8)
        disk = disk_report(run_root)
        order = V3_PIPELINE_ORDER if any(s in manifest.stages for s in _V3_STAGE_SET) else PIPELINE_ORDER
        print(f"HF Repo       : https://huggingface.co/{repo_id} (public)")
        print(f"Next Stage    : {manifest.next_incomplete_stage(order) or 'ALL COMPLETED'}")
        print(f"Latest Commit : stage={manifest.latest_stage or '-'} step={manifest.latest_step} "
              f"at {manifest.updated_at}")
        print(f"Local Disk    : {disk['free_gib']:.1f} GiB free / {disk['total_gib']:.1f} GiB total "
              f"(HF base cache={disk['hf_hub_weights_gib']:.1f} GiB, runs={disk['runs_gib']:.2f} GiB)")
        print("\nStage Progress:")
        print(f"  {'stage':<18} {'status':<12} {'step':<14} {'lm_loss':<10} {'depth_gain':<12} {'reroute'}")
        print("  " + "-" * 78)
        for st in order:
            info = manifest.stages.get(st) or {}
            status = info.get("status", "pending")
            step = info.get("step", 0)
            tgt = info.get("target_steps")
            step_s = f"{step}/{tgt}" if tgt else str(step)
            met = info.get("metrics") or {}
            lm = met.get("loss/lm", met.get("eval/loss_d2"))
            dg = met.get("eval/depth_gain")
            rr = met.get("loop/reroute_rate")
            lm_s = f"{lm:.4f}" if isinstance(lm, (int, float)) else "-"
            dg_s = f"{dg:+.4f}" if isinstance(dg, (int, float)) else "-"
            rr_s = f"{rr:.3f}" if isinstance(rr, (int, float)) else "-"
            print(f"  {st:<18} {status:<12} {step_s:<14} {lm_s:<10} {dg_s:<12} {rr_s}")
        if commits:
            print("\nRecent Hub Commits:")
            for c in commits:
                cid = str(c.get("commit_id") or "")[:8]
                print(f"  {cid}  {c.get('created_at', '')[:19]}  {c.get('title', '')}")
        return 0

    if args.action == "pull":
        manifest = hub.fetch_manifest()
        pulled = hub.pull_artifacts(run_root)
        target_stage = args.stage or manifest.latest_checkpoint_stage
        ckpt_dir = None
        if target_stage:
            ckpt_dir = hub.pull_stage_checkpoint(
                target_stage, run_root, prefer_promoted=args.promoted, include_optim=not args.no_optim,
                manifest=manifest,
            )
        print(json.dumps({"repo_id": repo_id, "pulled_artifacts": pulled, "checkpoint": ckpt_dir}, indent=2))
        return 0

    if args.action == "push":
        require(bool(args.stage), "`qloop sync push` requires --stage <stage_name>")
        sdir = os.path.join(run_root, args.stage)
        mgr = ckpt_cls(sdir, keep_last=0)
        cpath = mgr.promoted() if args.promoted else (mgr.latest() or mgr.promoted())
        require(cpath is not None, f"no local checkpoint found under {sdir}")
        assert cpath is not None
        oid = hub.push_checkpoint(args.stage, cpath, completed=args.completed, promoted=args.promoted,
                                  include_optim=args.include_optim)
        require(oid is not None, f"push to hf://{repo_id} failed; verify HF_TOKEN write access")
        print(json.dumps({"repo_id": repo_id, "stage": args.stage, "commit": oid}, indent=2))
        return 0

    if args.action == "pipeline":
        stages = (
            tuple(s.strip() for s in args.stages.split(",") if s.strip())
            if args.stages
            else PIPELINE_ORDER
        )
        lvl = args.debug if args.debug is not None else 1
        env = dict(os.environ, QLOOP_HF_REPO=repo_id, QLOOP_RUN_ROOT=run_root)
        pipeline_t0 = time.monotonic()
        stages_executed = 0
        for st_name in stages:
            manifest = hub.fetch_manifest()
            if manifest.is_completed(st_name) and not args.force:
                log.info("[pipeline] %s already completed on hf://%s (step=%d) - skipping",
                         st_name, repo_id, manifest.stage_step(st_name))
                if args.until == st_name:
                    break
                continue
            stage_budget_hours = args.time_budget_hours
            if stage_budget_hours is not None and float(stage_budget_hours) > 0 and stages_executed > 0:
                elapsed_hours = max(0.0, (time.monotonic() - pipeline_t0) / 3600.0)
                remaining_hours = float(stage_budget_hours) - elapsed_hours
                if remaining_hours <= 0.0:
                    log.warning(
                        "[pipeline] %s paused cleanly before completion "
                        "(step=%d; session time budget %.2fh exhausted). "
                        "Synced to hf://%s - re-run `qloop sync pipeline` in a fresh session to continue.",
                        st_name,
                        manifest.stage_step(st_name),
                        float(stage_budget_hours),
                        repo_id,
                    )
                    return 0
                stage_budget_hours = round(remaining_hours, 4) if remaining_hours >= 1e-3 else remaining_hours
            reclaim_disk_space(run_root, keep_stages=(st_name,), clean_hf_staging=True)
            cmds = _pipeline_stage_cmds(
                st_name,
                config_root=args.config_root,
                run_root=run_root,
                repo_id=repo_id,
                debug_level=lvl,
                overrides=list(args.set or []),
                time_budget_hours=stage_budget_hours,
                scan_batches=int(args.scan_batches),
                scan_seq_len=int(args.scan_seq_len),
                all_stages=stages,
            )
            log.info("[pipeline] === starting stage %s (%d command(s)) ===", st_name, len(cmds))
            for cmd in cmds:
                log.info("[pipeline] $ %s", " ".join(cmd))
                rc = subprocess.call(cmd, env=env)
                if rc != 0:
                    log.error("[pipeline] stage %s failed with exit code %d", st_name, rc)
                    return rc
            stages_executed += 1
            manifest = hub.fetch_manifest()
            if not manifest.is_completed(st_name):
                log.warning(
                    "[pipeline] %s paused cleanly before completion (step=%d; likely session time budget). "
                    "Synced to hf://%s - re-run `qloop sync pipeline` in a fresh session to continue.",
                    st_name, manifest.stage_step(st_name), repo_id,
                )
                return 0
            if st_name in ("s1_loop_heal", "s1_loop_a6b"):
                s1_m = (manifest.stages.get(st_name) or {}).get("metrics") or {}
                log.info("[pipeline] %s complete: depth_gain=%s reroute_rate=%s",
                         st_name, s1_m.get("eval/depth_gain"), s1_m.get("loop/reroute_rate"))
            if args.until == st_name:
                break
        log.info("[pipeline] requested pipeline stages finished and synced to https://huggingface.co/%s", repo_id)
        return 0

    raise ValueError(f"unknown sync action {args.action!r}")


# ---------------------------------------------------------------------------------------- parser

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="qloop", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("doctor", help="environment checks")
    _common(p, config=False)
    p.add_argument("--freeze", default=None, help="write pip freeze lockfile here")
    p.add_argument("--datasets", action="store_true", help="verify dataset access on the Hub")
    p.add_argument("--model", default=None, help="verify access to this model repo")
    p.set_defaults(fn=cmd_doctor)

    p = sub.add_parser("probe", help="backward / core-range / surgery / structured Gate 0p probes")
    _common(p)
    p.add_argument("--backward", action="store_true")
    p.add_argument("--core-ranges", action="store_true")
    p.add_argument("--surgery", action="store_true")
    p.add_argument("--structured", action="store_true", help="run Gate 0a + Gate 0p-1..3 structured probe suite")
    p.add_argument("--out-json", default=None, help="path to write structured probe_report.json")
    p.add_argument("--out-md", default=None, help="path to write structured probe_report.md")
    p.add_argument("--target-virtual", type=int, default=None, help="target virtual depth (e.g. 64 or 88)")
    p.add_argument("--n", type=int, default=4, help="number of probe sequences")
    p.add_argument("--seq-len", type=int, default=2048)
    p.set_defaults(fn=cmd_probe)

    p = sub.add_parser("scan-experts", help="ESFT router-affinity scan")
    _common(p)
    p.add_argument("--batches", type=int, default=64)
    p.add_argument("--seq-len", type=int, default=4096)
    p.add_argument("--out", default=None)
    p.add_argument("--ckpt", default=None, help="run dir or checkpoint dir to load trainables from")
    p.set_defaults(fn=cmd_scan_experts)

    p = sub.add_parser("train", help="run a training stage")
    _common(p)
    p.set_defaults(fn=cmd_train)

    p = sub.add_parser("eval", help="benchmark suite with loop-depth sweep")
    _common(p)
    p.add_argument("--ckpt", default=None)
    p.set_defaults(fn=cmd_eval)

    p = sub.add_parser("infer", help="run OptimizedLoopInferenceEngine on a Whole Model Pack")
    _common(p, config=False)
    p.add_argument("--pack", required=True, help="local Whole Model Pack directory or HF repo_id")
    p.add_argument("--prompt", default="Prove step-by-step that sqrt(2) is irrational.")
    p.add_argument("--depth", type=int, default=None, help="active loop pass depth (1, 2, or 3)")
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--prefill-chunk-size", type=int, default=4096)
    p.add_argument("--kv-quant", default="native", choices=["native", "fp8", "hadamard_int4"])
    p.add_argument("--benchmark", action="store_true")
    p.set_defaults(fn=cmd_infer)

    p = sub.add_parser("sync", help="HF Hub commit-history sync, 230 GB disk guard & pipeline runner")
    _common(p, config=False)
    p.add_argument("action", choices=["status", "pull", "push", "disk", "pipeline"])
    p.add_argument("--repo", default=None, help="HF public repo id (default: TassP0lar/Qwen3.6-35B-A3B-Loop64)")
    p.add_argument("--stage", default=None, help="stage name for pull/push")
    p.add_argument("--stages", default=None, help="comma-separated stages for pipeline")
    p.add_argument("--until", default=None, help="stop pipeline after completing this stage")
    p.add_argument("--force", action="store_true", help="re-run stages even if completed on HF Hub")
    p.add_argument("--completed", action="store_true", help="mark stage completed when pushing")
    p.add_argument("--promoted", action="store_true", help="push/pull promoted slot")
    p.add_argument("--include-optim", action="store_true", help="include optim.pt when pushing manually")
    p.add_argument("--no-optim", action="store_true", help="skip optim.pt when pulling")
    p.add_argument("--clean", action="store_true", help="reclaim superseded local checkpoints & HF staging caches")
    p.add_argument("--keep", default=None, help="comma-separated stages to keep during `sync disk --clean`")
    p.add_argument("--pip", action="store_true", help="also purge ~/.cache/pip during `sync disk --clean`")
    p.add_argument(
        "--time-budget-hours",
        type=float,
        default=None,
        help="Colab session wall-clock budget in hours across pipeline stages",
    )
    p.add_argument("--scan-batches", type=int, default=256, help="batches for ESFT router scans in pipeline")
    p.add_argument("--scan-seq-len", type=int, default=4096, help="seq_len for ESFT router scans in pipeline")
    p.add_argument("--config-root", default="configs", help="config directory root for pipeline")
    p.set_defaults(fn=cmd_sync)

    p = sub.add_parser("serve", help="OpenAI-compatible server")
    _common(p)
    p.add_argument("--ckpt", default=None)
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--model-name", default="qloop")
    p.set_defaults(fn=cmd_serve)

    p = sub.add_parser("external", help="Harbor / tau-bench against a served endpoint")
    _common(p)
    p.add_argument("harness", choices=["harbor", "tau"])
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(fn=cmd_external)

    p = sub.add_parser("export", help="export adapters (+ merged depth-1 HF checkpoint)")
    _common(p)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--merged-depth1", action="store_true")
    p.set_defaults(fn=cmd_export)

    p = sub.add_parser("export-pack", help="export or stream self-contained Whole Model Pack")
    _common(p)
    p.add_argument("--ckpt", default=None, help="run dir or checkpoint dir to load trainables from")
    p.add_argument("--out", default=None, help="local directory to write Whole Model Pack")
    p.add_argument("--push-repo", default=None, help="HF Hub repo_id to push Whole Model Pack to")
    p.add_argument("--stream-low-disk", action="store_true", help="stream one shard at a time (< 6 GB temp disk)")
    p.add_argument("--max-shard-bytes", type=int, default=4_500_000_000, help="maximum safetensors shard size in bytes")
    p.add_argument("--no-merge-hot-experts", action="store_true", help="keep HotExpertDeltas unmerged")
    p.add_argument("--private", action="store_true", help="create private HF repo when pushing")
    p.set_defaults(fn=cmd_export_pack)
    return ap


def _normalize_exit_code(code: object) -> int:
    if code is None:
        return 0
    if isinstance(code, int):
        return int(code)
    try:
        print(code, file=sys.stderr)
    except Exception:
        pass
    return 1


def flush_and_hard_exit(rc: int) -> NoReturn:
    """Synchronize CUDA, flush stdio/logs, and bypass threading._shutdown() via os._exit().

    Third-party streaming stacks (`datasets` IterableDataset + `hf_xet` background pools)
    can leave non-daemon worker threads alive after a stage finishes, causing standard
    `sys.exit()` (`threading._shutdown()`) to block indefinitely while holding GPU VRAM.
    Set `QLOOP_NO_FAST_EXIT=1` (`true`/`yes`) to fall back to `SystemExit`.
    """
    try:
        if "torch" in sys.modules:
            torch_mod = sys.modules["torch"]
            if (
                hasattr(torch_mod, "cuda")
                and torch_mod.cuda.is_available()
                and torch_mod.cuda.is_initialized()
            ):
                torch_mod.cuda.synchronize()
    except Exception:
        pass
    try:
        if sys.stdout is not None:
            sys.stdout.flush()
    except Exception:
        pass
    try:
        if sys.stderr is not None:
            sys.stderr.flush()
    except Exception:
        pass
    try:
        logging.shutdown()
    except Exception:
        pass
    try:
        if sys.stdout is not None:
            sys.stdout.flush()
    except Exception:
        pass
    try:
        if sys.stderr is not None:
            sys.stderr.flush()
    except Exception:
        pass
    code = int(rc)
    if os.environ.get("QLOOP_NO_FAST_EXIT", "").strip().lower() in ("1", "true", "yes"):
        raise SystemExit(code)
    os._exit(code)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.fn(args) or 0)
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130
    except Exception:
        return 1  # crash_guard already printed the summary + report path


def entrypoint(argv: list[str] | None = None) -> None:
    try:
        rc = main(argv)
    except SystemExit as exc:
        rc = _normalize_exit_code(exc.code)
    except KeyboardInterrupt:
        try:
            print("interrupted", file=sys.stderr)
        except Exception:
            pass
        rc = 130
    except BaseException:
        rc = 1
    flush_and_hard_exit(rc)


if __name__ == "__main__":
    entrypoint()
