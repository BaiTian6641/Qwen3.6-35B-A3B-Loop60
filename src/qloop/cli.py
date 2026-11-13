"""`python -m qloop <command>` - the only entry point the Colab notebook calls.

Every command runs in a fresh process (so CUDA_LAUNCH_BLOCKING & friends apply and a crash
never poisons the notebook kernel), gets a run directory, a log file and a crash report.

  doctor        environment table (+ --freeze lockfile, --datasets access check)
  probe         --backward | --core-ranges | --surgery  (cheap de-risking on the real model)
  scan-experts  ESFT router-affinity scan -> experts json (feeds experts.stats_path)
  train         LM stages (loop heal / SFT / long-context + D-LSA) or RL stage (GRPO/RSI)
  eval          benchmark suite with the loop-depth sweep
  sync          HF Hub commit-history sync, 230 GB disk guard & self-resuming stage pipeline
  serve         OpenAI-compatible server for external agent harnesses (Harbor, tau-bench)
  external      run Harbor / tau-bench against a served endpoint
  export        trainable adapters bundle, optional merged depth-1 HF checkpoint
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from typing import Any

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


def _start(cfg: Cfg | None, name: str, args: argparse.Namespace) -> str:
    run_dir = _run_dir(cfg, name)
    level = args.debug if args.debug is not None else int((cfg.get_path("debug.level") if cfg else None) or 1)
    setup_run(run_dir, level)
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
    from .model.probes import probe_backward, probe_core_ranges, surgery_check

    cfg = _load(args)
    run_dir = _start(cfg, "probe", args)
    out: dict[str, Any] = {}
    with crash_guard(run_dir):
        model, tok, _ = load_model(dict(cfg.get("model") or {}))
        from .model.introspect import inspect_model
        from .train.stack import Stack

        st = Stack(model, tok, inspect_model(model))
        seq = int(args.seq_len)
        if args.backward:
            with stage("probe:backward"):
                out["backward"] = probe_backward(model, _probe_ids(st, cfg, 1, seq)[0])
        if args.core_ranges:
            with stage("probe:core-ranges"):
                rows = probe_core_ranges(model, _probe_ids(st, cfg, args.n, seq),
                                         target_virtual=int(cfg.get_path("loop.target_virtual_layers") or 64))
                out["core_ranges"] = rows[:20]
        if args.surgery:
            with stage("probe:surgery"):
                loop = {k: v for k, v in dict(cfg.get("loop") or {}).items() if k != "enabled"}
                lsa = {k: v for k, v in dict(cfg.get("lsa") or {}).items() if k != "enabled"}
                ids = _probe_ids(st, cfg, 1, seq)[0]
                out["surgery"] = surgery_check(model, ids, loop or None, lsa or None)
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
) -> list[list[str]]:
    py = [sys.executable, "-u", "-m", "qloop"]
    common = ["--debug", str(debug_level), "-s", f"checkpoint.sync.hf_repo={repo_id}"]
    for ov in overrides:
        common.extend(["-s", ov])
    train_extra = list(common)
    if time_budget_hours is not None:
        train_extra.extend(["-s", f"train.time_budget_hours={time_budget_hours}"])

    if stage_name == "s0_probe":
        cpath = os.path.join(config_root, "stages", "s0_probe.yaml")
        out_stats = os.path.join(run_root, "s0_probe", "esft_stats.json")
        return [
            [*py, "probe", "-c", cpath, "--backward", "--surgery", "--core-ranges", *common],
            [*py, "scan-experts", "-c", cpath, "--batches", str(scan_batches), "--seq-len", str(scan_seq_len),
             "--out", out_stats, *common],
        ]
    if stage_name == "s1_loop_heal":
        return [[*py, "train", "-c", os.path.join(config_root, "stages", "s1_loop_heal.yaml"), *train_extra]]
    if stage_name == "s1_rescan":
        cpath = os.path.join(config_root, "stages", "s1_loop_heal.yaml")
        ckpt = os.path.join(run_root, "s1_loop_heal")
        out_stats = os.path.join(run_root, "s1_loop_heal", "esft_stats_looped.json")
        return [[*py, "scan-experts", "-c", cpath, "--ckpt", ckpt, "--batches", str(scan_batches),
                 "--seq-len", str(scan_seq_len), "--out", out_stats, *common]]
    if stage_name in ("s2_sft", "s3_long_lsa", "s4_rl"):
        return [[*py, "train", "-c", os.path.join(config_root, "stages", f"{stage_name}.yaml"), *train_extra]]
    if stage_name == "eval":
        return [[*py, "eval", "-c", os.path.join(config_root, "eval", "suite.yaml"),
                 "--ckpt", os.path.join(run_root, "s4_rl"), *common]]
    raise ValueError(f"unknown pipeline stage {stage_name!r}")


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
    from .hub import PIPELINE_ORDER, disk_report, reclaim_disk_space

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
        print(f"HF Repo       : https://huggingface.co/{repo_id} (public)")
        print(f"Next Stage    : {manifest.next_incomplete_stage() or 'ALL COMPLETED'}")
        print(f"Latest Commit : stage={manifest.latest_stage or '-'} step={manifest.latest_step} "
              f"at {manifest.updated_at}")
        print(f"Local Disk    : {disk['free_gib']:.1f} GiB free / {disk['total_gib']:.1f} GiB total "
              f"(HF base cache={disk['hf_hub_weights_gib']:.1f} GiB, runs={disk['runs_gib']:.2f} GiB)")
        print("\nStage Progress:")
        print(f"  {'stage':<14} {'status':<12} {'step':<14} {'lm_loss':<10} {'depth_gain':<12} {'reroute'}")
        print("  " + "-" * 74)
        for st in PIPELINE_ORDER:
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
            print(f"  {st:<14} {status:<12} {step_s:<14} {lm_s:<10} {dg_s:<12} {rr_s}")
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
        for st_name in stages:
            manifest = hub.fetch_manifest()
            if manifest.is_completed(st_name) and not args.force:
                log.info("[pipeline] %s already completed on hf://%s (step=%d) - skipping",
                         st_name, repo_id, manifest.stage_step(st_name))
                if args.until == st_name:
                    break
                continue
            reclaim_disk_space(run_root, keep_stages=(st_name,), clean_hf_staging=True)
            cmds = _pipeline_stage_cmds(
                st_name,
                config_root=args.config_root,
                run_root=run_root,
                repo_id=repo_id,
                debug_level=lvl,
                overrides=list(args.set or []),
                time_budget_hours=args.time_budget_hours,
                scan_batches=int(args.scan_batches),
                scan_seq_len=int(args.scan_seq_len),
            )
            log.info("[pipeline] === starting stage %s (%d command(s)) ===", st_name, len(cmds))
            for cmd in cmds:
                log.info("[pipeline] $ %s", " ".join(cmd))
                rc = subprocess.call(cmd, env=env)
                if rc != 0:
                    log.error("[pipeline] stage %s failed with exit code %d", st_name, rc)
                    return rc
            manifest = hub.fetch_manifest()
            if not manifest.is_completed(st_name):
                log.warning(
                    "[pipeline] %s paused cleanly before completion (step=%d; likely session time budget). "
                    "Synced to hf://%s - re-run `qloop sync pipeline` in a fresh session to continue.",
                    st_name, manifest.stage_step(st_name), repo_id,
                )
                return 0
            if st_name == "s1_loop_heal":
                s1_m = (manifest.stages.get("s1_loop_heal") or {}).get("metrics") or {}
                log.info("[pipeline] S1 loop heal complete: depth_gain=%s reroute_rate=%s",
                         s1_m.get("eval/depth_gain"), s1_m.get("loop/reroute_rate"))
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

    p = sub.add_parser("probe", help="backward / core-range / surgery probes")
    _common(p)
    p.add_argument("--backward", action="store_true")
    p.add_argument("--core-ranges", action="store_true")
    p.add_argument("--surgery", action="store_true")
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
    p.add_argument("--time-budget-hours", type=float, default=None, help="per-stage Colab session wall-clock budget")
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
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.fn(args) or 0)
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130
    except Exception:
        return 1  # crash_guard already printed the summary + report path


if __name__ == "__main__":
    sys.exit(main())
