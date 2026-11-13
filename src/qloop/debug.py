"""Verbose-but-readable debugging infrastructure.

Design goal: the Colab notebook stays tiny, every failure is still fully diagnosable.

* `setup_run(run_dir, level)`   - console + file logging, env flags per debug level.
* `crash_guard(run_dir, ...)`   - context manager: on ANY exception writes
     <run_dir>/crash/crash_<ts>.md with: traceback + local tensor summaries per frame,
     environment, CUDA memory summary, recent log tail, and (for OOM) a memory snapshot.
     The console only shows a short, actionable summary + the report path.
* `stage(name)`                 - named, timed sections; the active stage name is put
     into crash reports so you know *where* in the pipeline it died.
* `register_nonfinite_hooks()`  - forward hooks that pinpoint the first module that
     produced NaN/Inf (debug level >= 2).

Debug levels (cfg.debug.level):
  0 quiet   - INFO console, DEBUG file
  1 normal  - + per-step memory/timing, stage banners          (default)
  2 verbose - + NaN/Inf module hooks, TORCH_SHOW_CPP_STACKTRACES, sync timing
  3 forensic- + CUDA_LAUNCH_BLOCKING=1, autograd anomaly mode, memory history recording
"""
from __future__ import annotations

import collections
import contextlib
import datetime as _dt
import logging
import os
import platform
import sys
import time
import traceback
from collections.abc import Iterator
from typing import Any

log = logging.getLogger("qloop")

_STAGE_STACK: list[str] = []
_LOG_TAIL: collections.deque[str] = collections.deque(maxlen=400)
_LEVEL = 1


class _TailHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:  # pragma: no cover - trivial
        try:
            _LOG_TAIL.append(self.format(record))
        except Exception:
            pass


def debug_level() -> int:
    return _LEVEL


def setup_run(run_dir: str, level: int = 1) -> str:
    """Configure logging + debug env flags. Returns path of the log file.

    NOTE: CUDA_LAUNCH_BLOCKING must be set before CUDA initialises, which is why the
    notebook launches every stage as a fresh subprocess (`python -m qloop ...`).
    """
    global _LEVEL
    _LEVEL = int(level)
    os.makedirs(run_dir, exist_ok=True)
    if _LEVEL >= 2:
        os.environ.setdefault("TORCH_SHOW_CPP_STACKTRACES", "1")
        os.environ.setdefault("PYTHONFAULTHANDLER", "1")
    if _LEVEL >= 3:
        os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "1")
        os.environ.setdefault("TORCH_USE_CUDA_DSA", "1")
    import faulthandler

    faulthandler.enable(all_threads=True)

    log.setLevel(logging.DEBUG)
    log.handlers.clear()
    log.propagate = False
    fmt = logging.Formatter("%(asctime)s %(levelname).1s [%(stage)s] %(message)s", "%H:%M:%S")

    class _StageFilter(logging.Filter):
        def filter(self, record: logging.LogRecord) -> bool:
            record.stage = _STAGE_STACK[-1] if _STAGE_STACK else "-"
            return True

    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO if _LEVEL <= 1 else logging.DEBUG)
    log_path = os.path.join(run_dir, "run.log")
    fileh = logging.FileHandler(log_path, encoding="utf-8")
    fileh.setLevel(logging.DEBUG)
    tail = _TailHandler()
    tail.setLevel(logging.DEBUG)
    for h in (console, fileh, tail):
        h.setFormatter(fmt)
        h.addFilter(_StageFilter())
        log.addHandler(h)

    if _LEVEL >= 3:
        try:
            import torch

            torch.autograd.set_detect_anomaly(True)
            if torch.cuda.is_available():
                torch.cuda.memory._record_memory_history(max_entries=200_000)
            log.warning("forensic debug: anomaly mode + memory history ON (slow!)")
        except Exception as e:  # pragma: no cover
            log.warning("could not enable forensic torch debugging: %r", e)
    log.info("run dir: %s | debug level %d | log file %s", run_dir, _LEVEL, log_path)
    return log_path


@contextlib.contextmanager
def stage(name: str) -> Iterator[None]:
    _STAGE_STACK.append(name)
    t0 = time.time()
    if _LEVEL >= 1:
        log.info("▶ %s", name)
    try:
        yield
    finally:
        dt = time.time() - t0
        if _LEVEL >= 1:
            log.info("◀ %s (%.1fs)%s", name, dt, cuda_mem_brief(prefix=" | "))
        _STAGE_STACK.pop()


# ----------------------------------------------------------------------------- summaries

def describe(obj: Any, max_len: int = 160) -> str:
    """One-line summary of any object; tensors get shape/dtype/device/finite stats."""
    try:
        import torch

        if isinstance(obj, torch.Tensor):
            s = f"Tensor{tuple(obj.shape)} {str(obj.dtype).replace('torch.', '')} {obj.device}"
            if obj.requires_grad:
                s += " grad"
            if obj.numel() and obj.is_floating_point() and obj.numel() <= 50_000_000:
                with torch.no_grad():
                    f = obj.detach().float()
                    nonfin = (~torch.isfinite(f)).sum().item()
                    s += f" absmax={f.abs().max().item():.3g} mean={f.mean().item():.3g}"
                    if nonfin:
                        s += f" NONFINITE={nonfin}"
            return s
        if isinstance(obj, torch.nn.Module):
            n = sum(p.numel() for p in obj.parameters(recurse=True))
            return f"{type(obj).__name__}(params={n/1e6:.1f}M)"
    except Exception:
        pass
    try:
        r = repr(obj)
    except Exception:
        r = f"<unrepr-able {type(obj).__name__}>"
    return r if len(r) <= max_len else r[: max_len - 3] + "..."


def cuda_mem_brief(prefix: str = "") -> str:
    try:
        import torch

        if not torch.cuda.is_available() or not torch.cuda.is_initialized():
            return ""
        a = torch.cuda.memory_allocated() / 2**30
        r = torch.cuda.memory_reserved() / 2**30
        p = torch.cuda.max_memory_allocated() / 2**30
        return f"{prefix}mem alloc {a:.1f}G / reserved {r:.1f}G / peak {p:.1f}G"
    except Exception:
        return ""


def env_report() -> dict[str, str]:
    info: dict[str, str] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "argv": " ".join(sys.argv),
    }
    for mod in ("torch", "transformers", "accelerate", "datasets", "apollo_torch",
                "cut_cross_entropy", "fla", "causal_conv1d", "flash_attn", "vllm", "triton"):
        try:
            m = __import__(mod)
            info[mod] = getattr(m, "__version__", "installed")
        except Exception as e:
            info[mod] = f"missing ({type(e).__name__})"
    try:
        import torch

        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            info["gpu"] = f"{p.name} sm_{p.major}{p.minor} {p.total_memory/2**30:.1f}GiB"
            info["torch.cuda"] = str(torch.version.cuda)
            info["arch_list"] = ",".join(torch.cuda.get_arch_list())
    except Exception:
        pass
    for k in ("CUDA_LAUNCH_BLOCKING", "PYTORCH_CUDA_ALLOC_CONF", "HF_HOME", "QLOOP_RUN_ROOT"):
        if k in os.environ:
            info[f"env.{k}"] = os.environ[k]
    return info


# ----------------------------------------------------------------------------- crash reports

_HINTS = [
    ("out of memory", "CUDA OOM. Check the memory snapshot / peak above. Levers: lower data.seq_len, "
     "raise train.checkpoint_every_layer, loop.max_passes, disable loop.mhc_streams, or set "
     "train.offload_checkpoints=true. Re-run with debug.level=3 for an allocation timeline."),
    ("device-side assert", "A CUDA kernel assert fired; the reported line is often wrong. Re-run with "
     "debug.level=3 (CUDA_LAUNCH_BLOCKING=1). Most common cause: label/token id >= vocab size or "
     "an expert index out of range."),
    ("no kernel image", "The installed torch/kernel wheel was not built for this GPU arch (RTX PRO 6000 "
     "Blackwell = sm_120). Run `python -m qloop doctor` and reinstall the flagged package."),
    ("does not recognize this architecture", "transformers is too old for model_type=qwen3_5_moe. "
     "Upgrade: pip install -U transformers."),
    ("nonfinite", "NaN/Inf detected. Re-run with debug.level=2 to locate the first module producing it."),
    ("401", "HF auth failed - set HF_TOKEN in Colab secrets (gated datasets need accepted terms)."),
    ("gated", "Dataset/model is gated - accept the terms on the HF page with the account of HF_TOKEN."),
]


def _frame_locals(tb: Any, max_frames: int = 12) -> list[str]:
    out: list[str] = []
    frames = []
    while tb is not None:
        frames.append(tb)
        tb = tb.tb_next
    for t in frames[-max_frames:]:
        f = t.tb_frame
        fname = f.f_code.co_filename
        if "site-packages/torch/nn/modules/module.py" in fname:
            continue
        out.append(f"#### {os.path.basename(fname)}:{t.tb_lineno} in `{f.f_code.co_name}`")
        for k, v in list(f.f_locals.items())[:40]:
            if k.startswith("__"):
                continue
            out.append(f"- `{k}` = {describe(v)}")
    return out


def write_crash_report(run_dir: str, exc: BaseException, extra: dict[str, Any] | None = None) -> str:
    ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    cdir = os.path.join(run_dir, "crash")
    os.makedirs(cdir, exist_ok=True)
    path = os.path.join(cdir, f"crash_{ts}.md")
    msg = f"{type(exc).__name__}: {exc}"
    hints = [h for k, h in _HINTS if k.lower() in msg.lower()]
    lines = [f"# qloop crash report {ts}", "", f"**stage**: `{' > '.join(_STAGE_STACK) or '-'}`",
             f"**error**: `{msg[:2000]}`", ""]
    if hints:
        lines += ["## Hints", *[f"- {h}" for h in hints], ""]
    lines += ["## Traceback", "```", "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)), "```", ""]
    lines += ["## Locals (innermost frames)", *_frame_locals(exc.__traceback__), ""]
    lines += ["## Environment", *[f"- {k}: {v}" for k, v in env_report().items()], ""]
    if extra:
        lines += ["## Extra", *[f"- {k}: {describe(v, 400)}" for k, v in extra.items()], ""]
    try:
        import torch

        if torch.cuda.is_available() and torch.cuda.is_initialized():
            lines += ["## CUDA memory", "```", torch.cuda.memory_summary(abbreviated=True), "```", ""]
            if "out of memory" in msg.lower() or _LEVEL >= 3:
                snap = os.path.join(cdir, f"mem_snapshot_{ts}.pickle")
                try:
                    torch.cuda.memory._dump_snapshot(snap)
                    lines.append(f"memory snapshot: `{snap}` (open at https://pytorch.org/memory_viz)")
                except Exception as e:
                    lines.append(f"(memory snapshot unavailable: {e!r}; enable with debug.level=3)")
    except Exception:
        pass
    lines += ["", "## Log tail", "```", *list(_LOG_TAIL)[-150:], "```"]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return path


@contextlib.contextmanager
def crash_guard(run_dir: str, extra: dict[str, Any] | None = None, reraise: bool = True) -> Iterator[None]:
    try:
        yield
    except KeyboardInterrupt:
        log.warning("interrupted by user")
        raise
    except BaseException as e:
        path = write_crash_report(run_dir, e, extra)
        msg = f"{type(e).__name__}: {str(e)[:500]}"
        hints = [h for k, h in _HINTS if k.lower() in msg.lower()]
        bar = "=" * 88
        print(f"\n{bar}\n❌ FAILED in stage [{' > '.join(_STAGE_STACK) or '-'}]\n   {msg}", file=sys.stderr)
        for h in hints:
            print(f"   hint: {h}", file=sys.stderr)
        print(f"   full report: {path}\n{bar}", file=sys.stderr)
        if reraise:
            raise


# ----------------------------------------------------------------------------- NaN hooks

def register_nonfinite_hooks(model: Any, raise_on_first: bool = True) -> list[Any]:
    """Forward hooks that name the first module emitting NaN/Inf. Debug level >= 2."""
    import torch

    handles = []
    state = {"tripped": False}

    def _check(name: str):
        def hook(mod: Any, inp: Any, out: Any) -> None:
            if state["tripped"]:
                return
            outs = out if isinstance(out, (tuple, list)) else (out,)
            for o in outs:
                if isinstance(o, torch.Tensor) and o.is_floating_point() and not torch.isfinite(o).all():
                    state["tripped"] = True
                    ins = [describe(i) for i in (inp if isinstance(inp, tuple) else (inp,))]
                    msg = f"nonfinite output first produced by module '{name}' ({type(mod).__name__}); inputs: {ins}"
                    log.error(msg)
                    if raise_on_first:
                        raise FloatingPointError(msg)
        return hook

    for name, mod in model.named_modules():
        if len(list(mod.children())) == 0:
            handles.append(mod.register_forward_hook(_check(name)))
    log.info("registered NaN/Inf hooks on %d leaf modules", len(handles))
    return handles


def require(cond: bool, msg: str, **ctx: Any) -> None:
    """assert with context: `require(x.shape[-1]==2048, 'hidden mismatch', x=x)`."""
    if not cond:
        detail = "; ".join(f"{k}={describe(v)}" for k, v in ctx.items())
        raise AssertionError(f"{msg}" + (f" | {detail}" if detail else ""))
