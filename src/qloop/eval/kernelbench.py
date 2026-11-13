"""KernelBench evaluation: correctness + speedup of a generated `ModelNew` vs reference `Model`.

Reference source (KernelBench format) defines `Model(nn.Module)`, `get_inputs()`,
`get_init_inputs()`. The candidate source defines `ModelNew` with the same constructor
signature (typically using `torch.utils.cpp_extension.load_inline` or Triton).

Everything runs in a fresh subprocess (isolates CUDA context corruption, segfaults and
infinite loops; killed as a process group on timeout) with its own TORCH_EXTENSIONS_DIR:

    compile:      import candidate + construct ModelNew (load_inline builds happen here)
    correctness:  `n_correct_trials` random inputs (seeded), outputs allclose(atol, rtol)
    performance:  CUDA events (GPU) or perf_counter (CPU), median over `n_perf_trials`

Like KernelBench, both models are constructed right after `torch.manual_seed(seed)` so that
modules with the same parameter layout get identical weights.
"""
from __future__ import annotations

import json
import logging
import os
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, field
from typing import Any

from .code_exec import extract_code
from .scoring import Scorer

log = logging.getLogger("qloop")

_WORKER = r'''
import importlib.util, json, statistics, sys, time, traceback


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _to(x, dev):
    import torch
    if isinstance(x, torch.Tensor):
        return x.to(dev)
    if isinstance(x, (list, tuple)):
        return type(x)(_to(v, dev) for v in x)
    return x


def _cmp(a, b, atol, rtol):
    import torch
    if isinstance(a, torch.Tensor) or isinstance(b, torch.Tensor):
        if not (isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor)):
            return False, None, f"type mismatch {type(a).__name__} vs {type(b).__name__}"
        if a.shape != b.shape:
            return False, None, f"shape mismatch: ref {tuple(a.shape)} vs new {tuple(b.shape)}"
        af, bf = a.detach().float().cpu(), b.detach().float().cpu()
        if not torch.isfinite(bf).all() and torch.isfinite(af).all():
            return False, None, "candidate produced NaN/Inf"
        diff = (af - bf).abs()
        mx = float(diff.max()) if diff.numel() else 0.0
        ok = bool(torch.allclose(bf, af, atol=atol, rtol=rtol, equal_nan=True))
        return ok, mx, "" if ok else f"max abs diff {mx:.3g} (atol={atol}, rtol={rtol})"
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            return False, None, f"output length mismatch {len(a)} vs {len(b)}"
        worst = 0.0
        for x, y in zip(a, b):
            ok, mx, msg = _cmp(x, y, atol, rtol)
            if not ok:
                return ok, mx, msg
            worst = max(worst, mx or 0.0)
        return True, worst, ""
    try:
        ok = abs(float(a) - float(b)) <= atol + rtol * abs(float(a))
        return ok, abs(float(a) - float(b)), "" if ok else f"scalar mismatch {a} vs {b}"
    except Exception:
        return a == b, None, "" if a == b else f"value mismatch {a!r} vs {b!r}"


def _time(model, inputs, dev, warmup, trials):
    import torch
    with torch.no_grad():
        for _ in range(warmup):
            model(*inputs)
        times = []
        if dev.type == "cuda":
            torch.cuda.synchronize(dev)
            for _ in range(trials):
                s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                s.record()
                model(*inputs)
                e.record()
                torch.cuda.synchronize(dev)
                times.append(s.elapsed_time(e))
        else:
            for _ in range(trials):
                t0 = time.perf_counter()
                model(*inputs)
                times.append((time.perf_counter() - t0) * 1000.0)
    return times


def main():
    cfg = json.load(open(sys.argv[1]))
    res = {"compiled": False, "correct": False, "trials_passed": 0, "error": None, "stage": "start"}
    try:
        import torch
        dev = torch.device(cfg["device"])
        seed = int(cfg["seed"])
        res["stage"] = "load_reference"
        ref = _load(cfg["ref_path"], "kb_reference")
        init = _to(ref.get_init_inputs(), dev)
        torch.manual_seed(seed)
        ref_model = ref.Model(*init).to(dev).eval()
        res["stage"] = "compile"
        cand = _load(cfg["cand_path"], "kb_candidate")
        if not hasattr(cand, "ModelNew"):
            raise AttributeError("candidate source defines no `ModelNew`")
        torch.manual_seed(seed)
        new_model = cand.ModelNew(*init).to(dev).eval()
        if dev.type == "cuda":
            torch.cuda.synchronize(dev)
        res["compiled"] = True
        res["stage"] = "correctness"
        worst = 0.0
        for t in range(int(cfg["n_correct_trials"])):
            torch.manual_seed(seed + 1 + t)
            inputs = _to(ref.get_inputs(), dev)
            with torch.no_grad():
                o_ref = ref_model(*inputs)
                o_new = new_model(*inputs)
            if dev.type == "cuda":
                torch.cuda.synchronize(dev)
            ok, mx, msg = _cmp(o_ref, o_new, cfg["atol"], cfg["rtol"])
            worst = max(worst, mx or 0.0)
            if not ok:
                res["error"] = f"trial {t}: {msg}"
                break
            res["trials_passed"] += 1
        res["max_abs_diff"] = worst
        res["correct"] = res["trials_passed"] == int(cfg["n_correct_trials"])
        if res["correct"] and cfg["measure_perf"]:
            res["stage"] = "performance"
            torch.manual_seed(seed)
            inputs = _to(ref.get_inputs(), dev)
            tr = _time(ref_model, inputs, dev, cfg["warmup"], cfg["n_perf_trials"])
            tn = _time(new_model, inputs, dev, cfg["warmup"], cfg["n_perf_trials"])
            res["ref_ms"], res["new_ms"] = statistics.median(tr), statistics.median(tn)
            res["speedup"] = res["ref_ms"] / max(res["new_ms"], 1e-9)
            res["timer"] = "cuda_events" if dev.type == "cuda" else "perf_counter"
        res["stage"] = "done"
    except BaseException as e:
        res["error"] = f"{res['stage']}: {type(e).__name__}: {e}"
        res["traceback"] = traceback.format_exc()[-6000:]
    with open(cfg["out_path"], "w") as f:
        json.dump(res, f)


if __name__ == "__main__":
    main()
'''


@dataclass
class KernelResult:
    compiled: bool = False
    correct: bool = False
    speedup: float | None = None
    ref_ms: float | None = None
    new_ms: float | None = None
    max_abs_diff: float | None = None
    trials_passed: int = 0
    stage: str = "start"
    error: str | None = None
    timed_out: bool = False
    device: str = "cpu"
    wall_s: float = 0.0
    log: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _auto_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def evaluate_kernel(ref_src: str, cand_src: str, *, device: str = "auto", n_correct_trials: int = 5,
                    n_perf_trials: int = 20, warmup: int = 3, atol: float = 1e-2, rtol: float = 1e-2,
                    timeout: float = 600.0, seed: int = 42, measure_perf: bool = True,
                    python: str = sys.executable, keep_dir: str | None = None) -> KernelResult:
    dev = _auto_device(device)
    res = KernelResult(device=dev)
    if not cand_src or not cand_src.strip():
        res.stage, res.error = "compile", "empty candidate source"
        return res
    tmp_ctx = tempfile.TemporaryDirectory(prefix="qloop_kb_") if keep_dir is None else None
    root = tmp_ctx.name if tmp_ctx is not None else keep_dir
    os.makedirs(root, exist_ok=True)
    try:
        paths = {k: os.path.join(root, f"{k}.py") for k in ("worker", "reference", "candidate")}
        for k, src in (("worker", _WORKER), ("reference", ref_src), ("candidate", cand_src)):
            with open(paths[k], "w", encoding="utf-8") as f:
                f.write(src)
        cfg = {"ref_path": paths["reference"], "cand_path": paths["candidate"], "device": dev, "seed": seed,
               "n_correct_trials": n_correct_trials, "n_perf_trials": n_perf_trials, "warmup": warmup,
               "atol": atol, "rtol": rtol, "measure_perf": measure_perf,
               "out_path": os.path.join(root, "result.json")}
        cfg_path = os.path.join(root, "config.json")
        with open(cfg_path, "w", encoding="utf-8") as f:
            json.dump(cfg, f)
        env = dict(os.environ)
        env["TORCH_EXTENSIONS_DIR"] = os.path.join(root, "torch_ext")
        env.setdefault("PYTHONDONTWRITEBYTECODE", "1")
        t0 = time.time()
        proc = subprocess.Popen([python, paths["worker"], cfg_path], stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, cwd=root, env=env, start_new_session=True)
        try:
            out, _ = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                proc.kill()
            out, _ = proc.communicate()
            res.timed_out = True
        res.wall_s = time.time() - t0
        res.log = (out or "")[-8000:]
        if os.path.exists(cfg["out_path"]):
            with open(cfg["out_path"], encoding="utf-8") as f:
                data = json.load(f)
            for k in ("compiled", "correct", "speedup", "ref_ms", "new_ms", "max_abs_diff", "trials_passed",
                      "stage", "error"):
                if k in data:
                    setattr(res, k, data[k])
            if data.get("traceback"):
                res.log = (res.log + "\n" + data["traceback"]).strip()[-12000:]
            if data.get("timer"):
                res.extra["timer"] = data["timer"]
        elif res.timed_out:
            res.error = f"timeout after {timeout:.0f}s (stage unknown; likely compile or hang)"
        else:
            res.error = f"worker crashed with exit code {proc.returncode} (segfault / OOM?): {res.log[-500:]}"
        if res.timed_out and res.error is None:
            res.error = f"timeout after {timeout:.0f}s"
        log.debug("kernelbench: compiled=%s correct=%s speedup=%s err=%s (%.1fs)", res.compiled, res.correct,
                  res.speedup, res.error, res.wall_s)
        return res
    finally:
        if tmp_ctx is not None:
            tmp_ctx.cleanup()


def extract_kernel_source(text: str) -> str | None:
    return extract_code(text, "python", must_contain="ModelNew") or extract_code(text, "python")


def _kb_eval(text: str, item: dict[str, Any]) -> dict[str, Any]:
    src = extract_kernel_source(text)
    if src is None:
        return {"score": 0.0, "pred": None, "compiled": False, "correct": False, "speedup": None,
                "error": "no code block found"}
    kw = {k: item[k] for k in ("device", "atol", "rtol", "timeout", "n_correct_trials", "n_perf_trials") if k in item}
    r = evaluate_kernel(item["ref_src"], src, **kw)
    return {"score": float(r.correct), "pred": None, "compiled": r.compiled, "correct": r.correct,
            "speedup": r.speedup, "fast_1": bool(r.correct and (r.speedup or 0.0) > 1.0), "error": r.error}


KERNELBENCH_SCORER = Scorer("kernelbench", lambda t, it: None, lambda p, it: False, evaluate_fn=_kb_eval)
