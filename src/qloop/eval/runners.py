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

import hashlib
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


_RUN_IDENTITY_KEYS = ("model_id", "checkpoint_sha", "active_passes", "config_hash", "dataset_rev")


def _unwrap_generator(generator: Any) -> tuple[Any, Any, Any]:
    """Return `(core_generator, model, loop_controller)` unwrapping thin wrappers like `CountingGenerator`."""
    core = generator
    for _ in range(4):
        if getattr(core, "model", None) is not None:
            break
        nxt = getattr(core, "inner", None)
        if nxt is None or nxt is core:
            break
        core = nxt
    model = getattr(generator, "model", None)
    if model is None:
        model = getattr(core, "model", None)
    lc = getattr(generator, "loop_controller", None)
    if lc is None:
        lc = getattr(core, "loop_controller", None)
    return core, model, lc


def _checkpoint_path_digest(path: str) -> str:
    """Deterministic fingerprint of a checkpoint file or directory (realpath + file stats + small-file hashes)."""
    real = os.path.realpath(path)
    h = hashlib.sha256()
    h.update(real.encode("utf-8", "replace"))
    max_hash_bytes = 16 * 1024 * 1024
    if os.path.isdir(real):
        target = real
        latest_ptr = os.path.join(real, "checkpoints", "LATEST")
        if os.path.isfile(latest_ptr):
            try:
                with open(latest_ptr, encoding="utf-8") as f:
                    step_name = f.read().strip()
                cand = os.path.join(real, "checkpoints", step_name)
                if os.path.isdir(cand):
                    target = os.path.realpath(cand)
                    h.update(b"\x00latest:")
                    h.update(target.encode("utf-8", "replace"))
            except OSError:
                pass
        for fn in sorted(os.listdir(target)):
            fp = os.path.join(target, fn)
            if not os.path.isfile(fp):
                continue
            try:
                st = os.stat(fp)
            except OSError:
                continue
            h.update(f"\x00{fn}:{st.st_size}:{st.st_mtime_ns}".encode("utf-8", "replace"))
            if st.st_size <= max_hash_bytes:
                try:
                    with open(fp, "rb") as f:
                        h.update(hashlib.sha256(f.read()).digest())
                except OSError:
                    pass
    elif os.path.isfile(real):
        st = os.stat(real)
        h.update(f"\x00file:{st.st_size}:{st.st_mtime_ns}".encode("utf-8", "replace"))
        if st.st_size <= max_hash_bytes:
            with open(real, "rb") as f:
                h.update(hashlib.sha256(f.read()).digest())
    return f"path:{h.hexdigest()[:24]}"


def _model_weights_digest(model: Any, generator: Any, core: Any) -> str:
    """Fast O(1)-memory deterministic digest of model weights/script so distinct checkpoints cannot collide."""
    h = hashlib.sha256()
    if model is not None:
        h.update(f"{type(model).__module__}.{type(model).__qualname__}".encode())
        if hasattr(model, "script"):
            script_payload = {
                "script": list(getattr(model, "script", [])),
                "vocab_size": getattr(model, "vocab_size", None),
                "eos_id": getattr(model, "eos_id", None),
                "margin": getattr(model, "margin", None),
            }
            h.update(json.dumps(script_payload, sort_keys=True, default=str).encode())
        named_params = getattr(model, "named_parameters", None)
        if callable(named_params):
            import torch

            with torch.no_grad():
                for name, p in named_params():
                    h.update(f"\x00{name}:{tuple(p.shape)}:{p.dtype}".encode())
                    t = p.detach().reshape(-1)
                    n = int(t.numel())
                    if n == 0:
                        continue
                    if n <= 4096:
                        sl = t
                    else:
                        step = max(1, n // 512)
                        sl = torch.cat([t[::step][:512], t[-16:]], dim=0)
                    sample = (
                        sl.to(device="cpu", dtype=torch.float32)
                        if sl.is_floating_point()
                        else sl.to(device="cpu", dtype=torch.int64)
                    )
                    h.update(sample.contiguous().numpy().tobytes())
        return f"weights:{h.hexdigest()[:24]}"

    h.update(f"{type(core).__module__}.{type(core).__qualname__}".encode())
    for attr in ("name", "tag", "version", "script", "response", "answer"):
        for obj in (generator, core):
            val = getattr(obj, attr, None)
            if val is not None and isinstance(val, (str, int, float, bool, list, tuple)):
                h.update(f"\x00{attr}:{json.dumps(val, sort_keys=True, default=str)}".encode())
                break
    return f"stub:{h.hexdigest()[:24]}"


def _extract_model_id(generator: Any, core: Any, model: Any, explicit: str | None) -> str:
    if explicit is not None and str(explicit).strip():
        return str(explicit).strip()
    for obj in (generator, core, model):
        if obj is None:
            continue
        for attr in ("model_id", "model_name", "name_or_path"):
            val = getattr(obj, attr, None)
            if isinstance(val, str) and val.strip():
                return val.strip()
    cfg = getattr(model, "config", None) if model is not None else None
    if cfg is not None:
        for attr in ("_name_or_path", "name_or_path"):
            val = getattr(cfg, attr, None)
            if isinstance(val, str) and val.strip():
                return val.strip()
        mtype = getattr(cfg, "model_type", None)
        if isinstance(mtype, str) and mtype.strip():
            return f"{type(model).__module__}.{type(model).__qualname__}:{mtype.strip()}"
    if model is not None:
        return f"{type(model).__module__}.{type(model).__qualname__}"
    return f"{type(core).__module__}.{type(core).__qualname__}"


def _extract_active_passes(generator: Any, core: Any, model: Any, lc: Any) -> int | None:
    if lc is not None and getattr(lc, "active_passes", None) is not None:
        return int(lc.active_passes)
    for obj in (generator, core, model):
        if obj is None:
            continue
        ap = getattr(obj, "active_passes", None)
        if ap is not None:
            return int(ap)
    return None


def _extract_checkpoint_sha(
    generator: Any,
    core: Any,
    model: Any,
    explicit_sha: str | None,
    explicit_path: str | None,
) -> str:
    if explicit_sha is not None and str(explicit_sha).strip():
        return str(explicit_sha).strip()
    for obj in (generator, core, model):
        if obj is None:
            continue
        for attr in ("checkpoint_sha", "_qloop_checkpoint_sha"):
            val = getattr(obj, attr, None)
            if val is not None and str(val).strip():
                return str(val).strip()
    ckpt_path = explicit_path
    if not ckpt_path:
        for obj in (generator, core, model):
            if obj is None:
                continue
            for attr in ("checkpoint_path", "_qloop_checkpoint_path", "ckpt_path"):
                val = getattr(obj, attr, None)
                if isinstance(val, str) and val.strip():
                    ckpt_path = val.strip()
                    break
            if ckpt_path:
                break
    if ckpt_path:
        p_dig = (
            _checkpoint_path_digest(ckpt_path)
            if os.path.exists(ckpt_path)
            else f"ref:{hashlib.sha256(ckpt_path.encode('utf-8')).hexdigest()[:24]}"
        )
        if model is not None:
            return f"{p_dig}|{_model_weights_digest(model, generator, core)}"
        return p_dig
    return _model_weights_digest(model, generator, core)


def _compute_config_hash(cfg: dict[str, Any], core: Any, model: Any, lc: Any) -> str:
    payload: dict[str, Any] = {
        k: v for k, v in cfg.items() if k not in ("depths", "run_identity")
    }
    ctk = getattr(core, "chat_template_kwargs", None)
    if isinstance(ctk, dict) and ctk:
        payload["chat_template_kwargs"] = ctk
    if lc is not None and getattr(lc, "spec", None) is not None:
        sp = lc.spec
        payload["loop_spec"] = {
            k: getattr(sp, k, None)
            for k in ("prelude_end", "core_end", "num_passes", "kv_share", "bridge_mode", "per_pass_router", "mor")
        }
    mcfg = getattr(model, "config", None) if model is not None else None
    if mcfg is not None:
        if callable(getattr(mcfg, "to_dict", None)):
            try:
                payload["model_config"] = mcfg.to_dict()
            except Exception:
                payload["model_config"] = repr(mcfg)
        else:
            payload["model_config"] = {
                k: getattr(mcfg, k, None)
                for k in ("model_type", "vocab_size", "hidden_size", "num_hidden_layers", "num_attention_heads")
                if hasattr(mcfg, k)
            }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(raw.encode("utf-8", "replace")).hexdigest()[:24]


def build_run_identity(
    spec: BenchmarkSpec,
    generator: Any,
    cfg: dict[str, Any],
    *,
    model_id: str | None = None,
    checkpoint_sha: str | None = None,
    checkpoint_path: str | None = None,
    dataset_rev: str | None = None,
) -> dict[str, Any]:
    """Build a canonical model-bound eval cache identity fingerprint."""
    core, model, lc = _unwrap_generator(generator)
    resolved_model_id = _extract_model_id(generator, core, model, model_id)
    resolved_ckpt_sha = _extract_checkpoint_sha(generator, core, model, checkpoint_sha, checkpoint_path)
    resolved_passes = _extract_active_passes(generator, core, model, lc)
    resolved_cfg_hash = _compute_config_hash(cfg, core, model, lc)
    if dataset_rev is not None:
        resolved_rev = str(dataset_rev)
    elif spec.revision is not None:
        resolved_rev = str(spec.revision)
    else:
        resolved_rev = f"{spec.hf_id}:{spec.config or ''}:{spec.split}"
    return {
        "model_id": resolved_model_id,
        "checkpoint_sha": resolved_ckpt_sha,
        "active_passes": resolved_passes,
        "config_hash": resolved_cfg_hash,
        "dataset_rev": resolved_rev,
    }


def _existing_depth_caches(bdir: str) -> list[str]:
    if not os.path.isdir(bdir):
        return []
    out: list[str] = []
    for fn in sorted(os.listdir(bdir)):
        if fn.startswith("depth_") and fn.endswith(".jsonl"):
            fp = os.path.join(bdir, fn)
            try:
                if os.path.getsize(fp) > 0:
                    out.append(fn)
            except OSError:
                continue
    return out


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
                  chunk_size: int | None = None, allow_mixed_cache: bool = False,
                  model_id: str | None = None, checkpoint_sha: str | None = None,
                  checkpoint_path: str | None = None, dataset_rev: str | None = None) -> dict[str, Any]:
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
        cfg: dict[str, Any] = {
            "benchmark": spec.name, "hf_id": spec.hf_id, "config": spec.config, "split": spec.split,
            "n_samples": n_samples, "max_new_tokens": max_new, "temperature": temperature, "top_p": top_p,
            "top_k": top_k, "seed": seed, "depths": [_depth_label(d) for d in depths],
        }
        if spec.revision is not None:
            cfg["revision"] = spec.revision
        run_identity = build_run_identity(
            spec,
            generator,
            cfg,
            model_id=model_id,
            checkpoint_sha=checkpoint_sha,
            checkpoint_path=checkpoint_path,
            dataset_rev=dataset_rev,
        )
        cfg["run_identity"] = run_identity
        cfg_path = os.path.join(bdir, "run_config.json")
        if os.path.exists(cfg_path):
            with open(cfg_path, encoding="utf-8") as f:
                old = json.load(f)
            old_id = old.get("run_identity")
            if not isinstance(old_id, dict):
                if not allow_mixed_cache:
                    raise ValueError(
                        f"{spec.name}: existing eval cache in {bdir!r} lacks `run_identity` metadata "
                        "(cannot verify checkpoint/model identity). Use a fresh out_dir or pass "
                        "allow_mixed_cache=True to override."
                    )
                log.warning(
                    "%s: allow_mixed_cache=True: resuming legacy cache in %s without run_identity metadata",
                    spec.name,
                    bdir,
                )
            else:
                id_diff = {
                    k: (old_id.get(k), run_identity.get(k))
                    for k in _RUN_IDENTITY_KEYS
                    if old_id.get(k) != run_identity.get(k)
                }
                param_diff = {
                    k: (old.get(k), v)
                    for k, v in cfg.items()
                    if k not in ("depths", "run_identity") and old.get(k) != v
                }
                if id_diff or param_diff:
                    if not allow_mixed_cache:
                        raise ValueError(
                            f"{spec.name}: eval cache identity mismatch in {bdir!r} "
                            f"(identity_diff={id_diff}, config_diff={param_diff}). "
                            "Refusing silent reuse across different checkpoint/model/config identities - "
                            "use a fresh out_dir or pass allow_mixed_cache=True."
                        )
                    log.warning(
                        "%s: allow_mixed_cache=True: resuming with different identity/parameters: "
                        "identity=%s config=%s (old, new). Records from both runs will be mixed.",
                        spec.name,
                        id_diff,
                        param_diff,
                    )
        else:
            existing_caches = _existing_depth_caches(bdir)
            if existing_caches and not allow_mixed_cache:
                raise ValueError(
                    f"{spec.name}: existing cached records {existing_caches} in {bdir!r} without "
                    "run_config.json run_identity metadata. Use a fresh out_dir or pass allow_mixed_cache=True."
                )
            if existing_caches:
                log.warning(
                    "%s: allow_mixed_cache=True: reusing unattributed depth caches %s in %s",
                    spec.name,
                    existing_caches,
                    bdir,
                )
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
                   "contamination_group": spec.contamination_group, "run_identity": run_identity,
                   "config": cfg, "depths": per_depth,
                   "depth_curve": {k: v.get("mean_score") for k, v in per_depth.items()}}
        with open(os.path.join(bdir, "summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, default=str)
    return summary


def _as_dict(x: Any) -> dict[str, Any]:
    return dict(x) if isinstance(x, dict) else {}


def run_suite(suite_cfg: dict[str, Any], generator: Any, out_dir: str,
              offline_items: dict[str, list[dict[str, Any]]] | None = None, *,
              allow_mixed_cache: bool | None = None, model_id: str | None = None,
              checkpoint_sha: str | None = None, checkpoint_path: str | None = None) -> dict[str, Any]:
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
    suite_allow_mixed = (
        bool(allow_mixed_cache) if allow_mixed_cache is not None else bool(cfg.get("allow_mixed_cache", False))
    )
    suite_model_id = model_id if model_id is not None else cfg.get("model_id")
    suite_ckpt_sha = checkpoint_sha if checkpoint_sha is not None else cfg.get("checkpoint_sha")
    suite_ckpt_path = checkpoint_path if checkpoint_path is not None else cfg.get("checkpoint_path")
    os.makedirs(out_dir, exist_ok=True)
    results: dict[str, Any] = {}
    allowed = {"limit", "depths", "n_samples", "max_new_tokens", "temperature", "top_p", "top_k", "seed",
               "chunk_size", "allow_mixed_cache", "model_id", "checkpoint_sha", "checkpoint_path", "dataset_rev"}
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
        merged = {
            "allow_mixed_cache": suite_allow_mixed,
            "model_id": suite_model_id,
            "checkpoint_sha": suite_ckpt_sha,
            "checkpoint_path": suite_ckpt_path,
            **defaults,
            **e,
        }
        params = {k: v for k, v in merged.items() if k in allowed and v is not None}
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
