"""`python -m qloop <command>` - the only entry point the Colab notebook calls.

Every command runs in a fresh process (so CUDA_LAUNCH_BLOCKING & friends apply and a crash
never poisons the notebook kernel), gets a run directory, a log file and a crash report.

  doctor        environment table (+ --freeze lockfile, --datasets access check)
  probe         --backward | --core-ranges | --surgery  (cheap de-risking on the real model)
  scan-experts  ESFT router-affinity scan -> experts json (feeds experts.stats_path)
  train         LM stages (loop heal / SFT / long-context + D-LSA) or RL stage (GRPO/RSI)
  eval          benchmark suite with the loop-depth sweep
  serve         OpenAI-compatible server for external agent harnesses (Harbor, tau-bench)
  external      run Harbor / tau-bench against a served endpoint
  export        trainable adapters bundle, optional merged depth-1 HF checkpoint
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from typing import Any

from .config import Cfg, load_config
from .debug import crash_guard, setup_run, stage

log = logging.getLogger("qloop")


def _run_dir(cfg: Cfg | None, name: str) -> str:
    root = os.environ.get("QLOOP_RUN_ROOT") or (cfg.get_path("run.root") if cfg else None) or "runs"
    rname = (cfg.get_path("run.name") if cfg else None) or name
    return os.path.join(root, rname)


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


def _stack_with_ckpt(cfg: Cfg, ckpt: str | None):
    from .train.checkpoint import CheckpointManager, restore_trainable
    from .train.stack import build_stack

    st = build_stack(cfg)
    if ckpt:
        path = ckpt
        if os.path.isdir(os.path.join(ckpt, "checkpoints")):
            mgr = CheckpointManager(ckpt, keep_last=0)
            path = mgr.promoted() or mgr.latest()
        ck = CheckpointManager(os.path.dirname(os.path.dirname(path)), keep_last=0).load(path)
        restore_trainable(st.named_trainable, ck["params"], strict=False)
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
        with open(os.path.join(run_dir, "probe.json"), "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, default=str)
        log.info("probe results -> %s", os.path.join(run_dir, "probe.json"))
    return 0


def cmd_scan_experts(args: argparse.Namespace) -> int:
    from .model.experts import scan_router_affinity

    cfg = _load(args)
    run_dir = _start(cfg, "scan_experts", args)
    with crash_guard(run_dir):
        st = _stack_with_ckpt(cfg, args.ckpt)
        ids = _probe_ids(st, cfg, int(args.batches), int(args.seq_len))
        stats = scan_router_affinity(st.model, ids, controller=st.loop)
        stats.log_summary()
        out = args.out or os.path.join(run_dir, "esft_stats.json")
        stats.to_json(out)
        log.info("expert stats -> %s (use experts.stats_path=%s)", out, out)
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
        res = run_suite(dict(cfg.get("eval") or {}), gen, os.path.join(run_dir, "eval"))
        log.info("suite summary: %s", json.dumps(res, default=str)[:4000])
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

    from .train.checkpoint import collect_trainable

    cfg = _load(args)
    run_dir = _start(cfg, "export", args)
    with crash_guard(run_dir):
        st = _stack_with_ckpt(cfg, args.ckpt)
        os.makedirs(args.out, exist_ok=True)
        params = {n: p.detach().cpu() for n, p in st.named_trainable.items()}
        if not params:
            params = {n: t.cpu() for n, t in collect_trainable(st.named_trainable).items()}
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
