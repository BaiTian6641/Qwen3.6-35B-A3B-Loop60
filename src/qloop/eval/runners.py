"""Run locally-scorable benchmarks at several loop depths (the test-time depth-scaling curve).

Layout written by `run_benchmark(spec, ...)` under `out_dir/<spec.name>/`:
    run_config.json        parameters of the (first) run
    depth_<d>.jsonl        one record per item: samples (text, tokens, finish, score, pred, ...)
    summary.json           per-depth metrics + "depth_curve" {depth: mean score}

Resumable: items whose id is already in depth_<d>.jsonl are skipped (a truncated last line from
a crash is ignored and regenerated). Seeds are derived from (seed, item id) so a resumed run
produces the same samples as an uninterrupted one.
"""
from __future__ import annotations

import json
import logging
import os
import time
import traceback
import zlib
from collections.abc import Iterable, Sequence
from typing import Any

from qloop.debug import require, stage

from .registry import BenchmarkSpec, get_spec, load_items, missing_requirements
from .scoring import get_scorer, majority_vote, pass_at_k

log = logging.getLogger("qloop")


def _depth_label(d: int | None) -> str:
    return "native" if d is None else str(int(d))


def _read_jsonl(path: str) -> dict[str, dict[str, Any]]:
    done: dict[str, dict[str, Any]] = {}
    if not os.path.exists(path):
        return done
    bad = 0
    with open(path, encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except ValueError:
                bad += 1
                continue
            done[str(rec["id"])] = rec
    if bad:
        log.warning("%s: ignored %d unreadable line(s) (interrupted write?) - those items will be redone", path, bad)
    return done


def _item_seed(seed: int, item_id: str) -> int:
    return (int(seed) * 1_000_003 + zlib.crc32(str(item_id).encode("utf-8"))) % (2**31)


def _ks(n: int) -> list[int]:
    ks, k = [], 1
    while k <= n:
        ks.append(k)
        k *= 2
    if n not in ks:
        ks.append(n)
    return ks


def summarize_records(records: Sequence[dict[str, Any]], scorer_name: str | None = None) -> dict[str, Any]:
    if not records:
        return {"n_items": 0}
    n_samples = min(len(r["samples"]) for r in records)
    scores = [[float(s["score"]) for s in r["samples"]] for r in records]
    flat = [x for row in scores for x in row]
    out: dict[str, Any] = {
        "n_items": len(records),
        "n_samples": n_samples,
        "mean_score": sum(flat) / len(flat),
        "mean_completion_tokens": sum(s["completion_tokens"] for r in records for s in r["samples"]) / len(flat),
        "frac_truncated": sum(s["finish_reason"] == "length" for r in records for s in r["samples"]) / len(flat),
    }
    for k in _ks(n_samples):
        out[f"pass@{k}"] = sum(pass_at_k(len(row), sum(x >= 1.0 for x in row), k) for row in scores) / len(scores)
    if n_samples > 1 and scorer_name:
        scorer = get_scorer(scorer_name)
        maj = []
        for r in records:
            preds = [s.get("pred") for s in r["samples"]]
            if all(p is None for p in preds):
                continue
            win = majority_vote(preds, scorer.equal)
            maj.append(float(any(s.get("pred") == win and s["score"] >= 1.0 for s in r["samples"])) if win is not None
                       else 0.0)
        if maj:
            out[f"maj@{n_samples}"] = sum(maj) / len(maj)
    if any("speedup" in s for r in records for s in r["samples"]):
        samples = [s for r in records for s in r["samples"]]
        sp = [s["speedup"] for s in samples if s.get("correct") and s.get("speedup")]
        out["compiled_rate"] = sum(bool(s.get("compiled")) for s in samples) / len(samples)
        out["fast_1"] = sum(bool(s.get("fast_1")) for s in samples) / len(samples)
        out["mean_speedup_correct"] = (sum(sp) / len(sp)) if sp else None
    return out


def run_benchmark(spec: BenchmarkSpec | str, generator: Any, limit: int | None = None,
                  depths: Sequence[int | None] | None = (1, 2), n_samples: int = 1, max_new_tokens: int | None = None,
                  out_dir: str = "runs/eval", *, temperature: float | None = None, top_p: float | None = None,
                  top_k: int = 0, seed: int = 0, offline_items: list[dict[str, Any]] | None = None,
                  items: list[dict[str, Any]] | None = None, capabilities: Iterable[str] | None = None,
                  chunk_size: int | None = None) -> dict[str, Any]:
    spec = get_spec(spec) if isinstance(spec, str) else spec
    require(spec.scorer is not None, f"benchmark {spec.name} has no local scorer; use qloop.eval.external "
            f"(harness={spec.harness})")
    missing = missing_requirements(spec, capabilities)
    if missing:
        raise RuntimeError(f"benchmark {spec.name} needs capabilities {sorted(missing)} that are not available "
                           f"(have {sorted(capabilities) if capabilities is not None else 'auto-detected'}). "
                           f"Notes: {spec.notes}")
    scorer = get_scorer(spec.scorer)
    lc = getattr(generator, "loop_controller", None)
    if lc is None:
        if depths and any(d is not None for d in depths):
            log.warning("%s: generator has no loop_controller -> evaluating native depth only (requested %s)",
                        spec.name, list(depths))
        depths = [None]
    depths = list(depths or [None])
    max_new = int(max_new_tokens or spec.max_new_tokens)
    if temperature is None:
        temperature = 0.0 if n_samples == 1 else 0.6
    if top_p is None:
        top_p = 1.0 if temperature <= 0 else 0.95
    bdir = os.path.join(out_dir, spec.name)
    os.makedirs(bdir, exist_ok=True)

    with stage(f"eval:{spec.name}"):
        if items is None:
            items = load_items(spec, limit, offline_items)
        elif limit is not None:
            items = items[:limit]
        require(len(items) > 0, f"{spec.name}: no items to evaluate", limit=limit)
        cfg = {"benchmark": spec.name, "hf_id": spec.hf_id, "config": spec.config, "split": spec.split,
               "n_samples": n_samples, "max_new_tokens": max_new, "temperature": temperature, "top_p": top_p,
               "top_k": top_k, "seed": seed, "depths": [_depth_label(d) for d in depths]}
        cfg_path = os.path.join(bdir, "run_config.json")
        if os.path.exists(cfg_path):
            with open(cfg_path, encoding="utf-8") as f:
                old = json.load(f)
            diff = {k: (old.get(k), v) for k, v in cfg.items() if k != "depths" and old.get(k) != v}
            if diff:
                log.warning("%s: resuming with different parameters than the existing run: %s (old, new). Records "
                            "from both runs will be mixed - use a fresh out_dir for a clean comparison.",
                            spec.name, diff)
        else:
            with open(cfg_path, "w", encoding="utf-8") as f:
                json.dump(cfg, f, indent=2)

        bs = int(chunk_size or max(1, getattr(generator, "batch_size", 1)))
        per_depth: dict[str, Any] = {}
        for d in depths:
            label = _depth_label(d)
            path = os.path.join(bdir, f"depth_{label}.jsonl")
            done = _read_jsonl(path)
            todo = [it for it in items if str(it["id"]) not in done]
            log.info("%s depth=%s: %d items (%d done, %d to run), n=%d, max_new=%d", spec.name, label, len(items),
                     len(items) - len(todo), len(todo), n_samples, max_new)
            t0 = time.time()
            with open(path, "a", encoding="utf-8") as fout:
                for c in range(0, len(todo), bs):
                    chunk = todo[c:c + bs]
                    prompts = [spec.build_prompt(it) for it in chunk]
                    seeds = [_item_seed(seed, it["id"]) for it in chunk]
                    outs = generator.generate(prompts, max_new_tokens=max_new, temperature=temperature, top_p=top_p,
                                              n=n_samples, seed=seeds, top_k=top_k, depth=d)
                    for it, samples in zip(chunk, outs):
                        recs = []
                        for o in samples:
                            try:
                                ev = scorer.evaluate(o.text, it)
                            except Exception as e:  # a scorer bug must not kill a multi-hour run
                                log.error("%s: scorer failed on item %s: %r", spec.name, it["id"], e)
                                ev = {"score": 0.0, "pred": None, "error": f"scorer: {type(e).__name__}: {e}",
                                      "traceback": traceback.format_exc()[-2000:]}
                            pred = ev.get("pred")
                            if pred is not None and not isinstance(pred, (str, int, float, bool)):
                                ev["pred"] = json.dumps(pred, default=str)
                            recs.append({"text": o.text, "completion_tokens": o.completion_tokens,
                                         "prompt_tokens": o.prompt_tokens, "finish_reason": o.finish_reason, **ev})
                        rec = {"id": str(it["id"]), "depth": label, "answer": it.get("answer"), "samples": recs,
                               "mean_score": sum(r["score"] for r in recs) / len(recs)}
                        fout.write(json.dumps(rec, default=str) + "\n")
                        fout.flush()
                        done[rec["id"]] = rec
            dt = time.time() - t0
            records = [done[str(it["id"])] for it in items if str(it["id"]) in done]
            summ = summarize_records(records, spec.scorer)
            summ["wall_s_this_run"] = round(dt, 2)
            per_depth[label] = summ
            log.info("%s depth=%s: mean=%.4f %s", spec.name, label, summ.get("mean_score", float("nan")),
                     {k: round(v, 4) for k, v in summ.items() if k.startswith(("pass@", "maj@"))})
        summary = {"benchmark": spec.name, "kind": spec.kind, "scorer": spec.scorer,
                   "contamination_group": spec.contamination_group, "config": cfg, "depths": per_depth,
                   "depth_curve": {k: v.get("mean_score") for k, v in per_depth.items()}}
        with open(os.path.join(bdir, "summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, default=str)
    return summary


def _as_dict(x: Any) -> dict[str, Any]:
    return dict(x) if isinstance(x, dict) else {}


def run_suite(suite_cfg: dict[str, Any], generator: Any, out_dir: str,
              offline_items: dict[str, list[dict[str, Any]]] | None = None) -> dict[str, Any]:
    """suite_cfg:
        benchmarks: [aime_2026, {name: gpqa_diamond, limit: 50, n_samples: 4}, ...]
        defaults: {depths: [1, 2], n_samples: 1, max_new_tokens: null, limit: null, temperature: null}
        capabilities: [code_exec, gpu_exec]          # optional override of auto-detection
        trained_contamination_groups: [terminal_bench]
        fail_fast: false
    """
    cfg = _as_dict(suite_cfg.get("eval", suite_cfg)) if "benchmarks" not in suite_cfg else _as_dict(suite_cfg)
    require("benchmarks" in cfg, "suite config needs a 'benchmarks' list", keys=sorted(cfg))
    defaults = _as_dict(cfg.get("defaults"))
    caps = cfg.get("capabilities")
    trained = set(cfg.get("trained_contamination_groups") or [])
    fail_fast = bool(cfg.get("fail_fast", False))
    os.makedirs(out_dir, exist_ok=True)
    results: dict[str, Any] = {}
    allowed = {"limit", "depths", "n_samples", "max_new_tokens", "temperature", "top_p", "top_k", "seed",
               "chunk_size"}
    for entry in cfg["benchmarks"]:
        e = {"name": entry} if isinstance(entry, str) else _as_dict(entry)
        name = e.get("name")
        try:
            spec = get_spec(name)
        except KeyError as ex:
            results[str(name)] = {"status": "error", "error": str(ex)}
            if fail_fast:
                raise
            continue
        if spec.contamination_group in trained:
            log.warning("suite: %s is in contamination group %r which was used for training - its score is NOT a "
                        "held-out result", name, spec.contamination_group)
        if spec.scorer is None or missing_requirements(spec, caps):
            why = (f"needs {sorted(missing_requirements(spec, caps))}" if spec.scorer is not None
                   else f"external harness {spec.harness}")
            results[name] = {"status": "skipped", "reason": f"not locally scorable: {why}"}
            log.info("suite: skip %s (%s)", name, why)
            continue
        params = {k: v for k, v in {**defaults, **e}.items() if k in allowed and v is not None}
        try:
            summ = run_benchmark(spec, generator, out_dir=out_dir, capabilities=caps,
                                 offline_items=(offline_items or {}).get(name), **params)
            results[name] = {"status": "ok", "contaminated": spec.contamination_group in trained, **summ}
        except Exception as ex:
            log.error("suite: %s failed: %s: %s", name, type(ex).__name__, ex)
            results[name] = {"status": "error", "error": f"{type(ex).__name__}: {ex}",
                             "traceback": traceback.format_exc()[-4000:]}
            if fail_fast:
                raise
    table = {n: r.get("depth_curve") for n, r in results.items() if r.get("status") == "ok"}
    out = {"benchmarks": results, "depth_curves": table}
    with open(os.path.join(out_dir, "suite_summary.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=str)
    return out
