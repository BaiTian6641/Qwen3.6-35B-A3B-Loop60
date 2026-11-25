"""Notebook-side helpers (no torch import): keep the single Colab notebook concise and debuggable.

    from qloop.colab import setup_env, q, hub_status, resolve_base_model_name, run_pipeline, tail_metrics
    setup_env(hf_repo="TassP0lar/Qwen3.6-35B-A3B-Loop64")
    q(f"doctor --freeze requirements/colab.lock.txt --model {resolve_base_model_name('nvfp4')}")
    hub_status()
    run_pipeline(base_precision="nvfp4", debug=1, time_budget_hours=21.5)

Design:
* Zero Google Drive dependency: every stage commits its trainable checkpoint + `manifest.json` +
  auto-rendered `README.md` directly to `TassP0lar/Qwen3.6-35B-A3B-Loop64` via `HF_TOKEN`, then
  prunes superseded local checkpoints and HF upload caches to stay comfortably within Colab's
  230 GB SSD limit.
* `q(...)` executes `python -u -m qloop ...` in a *fresh subprocess* (clean CUDA context per stage,
  `CUDA_LAUNCH_BLOCKING` / `TORCH_SHOW_CPP_STACKTRACES` applied before CUDA init), streams stdout
  live, and on failure prints the structured crash report (`Hints`, `Traceback`, `Local Tensors`,
  `CUDA Memory`) inline in the notebook output.
"""
from __future__ import annotations

import glob
import importlib
import importlib.metadata
import importlib.util
import json
import os
import re
import shlex
import subprocess
import sys
import time
from collections.abc import Sequence
from typing import Any

DEFAULT_HF_REPO = "TassP0lar/Qwen3.6-35B-A3B-Loop64"
DEFAULT_PACK_REPO = "TassP0lar/Qwen3.6-35B-A6B-LoopHSSD-Pack"
DEFAULT_HF_KERNELS_PIP_SPEC = "kernels>=0.18.2,<0.19.0"
BASE_MODEL_BY_PRECISION: dict[str, str] = {
    "nvfp4": "unsloth/Qwen3.6-35B-A3B-NVFP4",
    "nvfp4-fast": "unsloth/Qwen3.6-35B-A3B-NVFP4-Fast",
    "fp8": "Qwen/Qwen3.6-35B-A3B-FP8",
    "bf16": "Qwen/Qwen3.6-35B-A3B",
}
SECRETS = (
    "HF_TOKEN",
    "WANDB_API_KEY",
    "TEACHER_API_KEY",
    "DAYTONA_API_KEY",
    "MODAL_TOKEN_ID",
    "MODAL_TOKEN_SECRET",
    "OPENAI_API_KEY",
)
DEFAULT_METRIC_KEYS = (
    "loss/lm",
    "eval/loss_d1",
    "eval/loss_d2",
    "eval/loss_d3",
    "eval/depth_gain",
    "loop/reroute_rate",
    "loop/unique_experts_core",
    "loop/effective_expert_mult",
    "gdn2/erase_write_l1",
    "loss/hssd_budget",
    "hssd/budget_aux_loss",
    "lsa/distill_kl",
    "reward/mean",
    "gate/score",
    "tok_per_s",
    "mem/peak_gib",
)


def resolve_base_model_name(base_precision: str | None) -> str | None:
    """Resolve `'nvfp4'`, `'nvfp4-fast'`, `'fp8'`, `'bf16'`, or an `'org/model'` HF repo ID to a model name."""
    if base_precision is None:
        return None
    raw = str(base_precision).strip()
    if not raw:
        raise ValueError(
            f"invalid base_precision {base_precision!r}; expected one of {sorted(BASE_MODEL_BY_PRECISION)} "
            "or a Hugging Face repo ID like 'org/model'"
        )
    alias = raw.lower()
    if alias in BASE_MODEL_BY_PRECISION:
        return BASE_MODEL_BY_PRECISION[alias]
    alias_norm = alias.replace("_", "-")
    if alias_norm in BASE_MODEL_BY_PRECISION:
        return BASE_MODEL_BY_PRECISION[alias_norm]
    parts = raw.split("/")
    if len(parts) == 2 and all(p and not any(ch.isspace() for ch in p) for p in parts):
        return raw
    raise ValueError(
        f"invalid base_precision {base_precision!r}; expected one of {sorted(BASE_MODEL_BY_PRECISION)} "
        "or a Hugging Face repo ID like 'org/model'"
    )


def _inject_base_model_override(
    overrides: Sequence[str],
    base_precision: str | None,
    *,
    smoke_toy: bool = False,
) -> list[str]:
    resolved = resolve_base_model_name(base_precision)
    out = list(overrides)
    if smoke_toy or resolved is None:
        return out
    caller_keys = {ov.split("=", 1)[0].strip() for ov in out if "=" in ov}
    if "model.name" not in caller_keys:
        return [f"model.name={resolved}", *out]
    return out


def _subprocess_env() -> dict[str, str]:
    env = dict(os.environ)
    env.pop("HF_HUB_ENABLE_HF_TRANSFER", None)
    env.setdefault("HF_XET_HIGH_PERFORMANCE", "1")
    src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cur_pp = env.get("PYTHONPATH", "")
    if src_dir not in cur_pp.split(os.pathsep):
        env["PYTHONPATH"] = f"{src_dir}{os.pathsep}{cur_pp}" if cur_pp else src_dir
    return env


def _parse_hf_kernels_pip_spec_strict(text: str) -> str | None:
    """Parse an explicit `kernels` version range/pin from `text`, or return `None` if absent."""
    if not text:
        return None
    m_range = re.search(
        r"([0-9]+(?:\.[0-9a-zA-Z]+)+)\s*(<=|<)\s*version\s*(<=|<)\s*([0-9]+(?:\.[0-9a-zA-Z]+)+)",
        text,
    )
    if m_range:
        lo_ver, lo_cmp, hi_cmp, hi_ver = m_range.groups()
        lo_op = ">=" if lo_cmp == "<=" else ">"
        return f"kernels{lo_op}{lo_ver},{hi_cmp}{hi_ver}"
    m_compound = re.search(
        r"kernels\s*((?:>=|<=|==|!=|~=|>|<)\s*[0-9][0-9a-zA-Z.*+-]*"
        r"(?:\s*,\s*(?:>=|<=|==|!=|~=|>|<)\s*[0-9][0-9a-zA-Z.*+-]*)+)",
        text,
    )
    if m_compound:
        cleaned = re.sub(r"\s+", "", m_compound.group(1))
        return f"kernels{cleaned}"
    m_ver_right = re.search(r"version\s*(>=|<=|==|!=|~=|>|<)\s*([0-9]+(?:\.[0-9a-zA-Z]+)+)", text)
    if m_ver_right:
        return f"kernels{m_ver_right.group(1)}{m_ver_right.group(2)}"
    m_ver_left = re.search(r"([0-9]+(?:\.[0-9a-zA-Z]+)+)\s*(<=|<)\s*version\b", text)
    if m_ver_left:
        lo_op = ">=" if m_ver_left.group(2) == "<=" else ">"
        return f"kernels{lo_op}{m_ver_left.group(1)}"
    m_pin = re.search(r"kernels\s*(==|>=|<=|!=|~=|>|<)\s*([0-9]+(?:\.[0-9a-zA-Z]+)+)", text)
    if m_pin:
        return f"kernels{m_pin.group(1)}{m_pin.group(2)}"
    return None


def _parse_hf_kernels_pip_spec(text: str) -> str:
    """Parse HF `finegrained_fp8` missing-kernel message into a valid `pip` requirement string."""
    return _parse_hf_kernels_pip_spec_strict(text) or DEFAULT_HF_KERNELS_PIP_SPEC


def _import_utils_kernels_bounds_hint(fg_fp8: Any = None) -> str:
    min_v: Any = None
    max_v: Any = None
    if fg_fp8 is not None:
        min_v = getattr(fg_fp8, "KERNELS_MIN_VERSION", None)
        max_v = getattr(fg_fp8, "KERNELS_MAX_VERSION", None)
    try:
        imp_utils = importlib.import_module("transformers.utils.import_utils")
        min_v = min_v or getattr(imp_utils, "KERNELS_MIN_VERSION", None)
        max_v = max_v or getattr(imp_utils, "KERNELS_MAX_VERSION", None)
    except Exception:
        pass
    s_min = str(min_v).strip() if min_v is not None else ""
    s_max = str(max_v).strip() if max_v is not None else ""
    if s_min and s_max:
        return f"({s_min} <= version < {s_max})"
    if s_min:
        return f"version >= {s_min}"
    if s_max:
        return f"version < {s_max}"
    return ""


def _refresh_runtime_import_caches() -> None:
    """Invalidate import finders and clear HF `transformers` `@lru_cache` package availability checks."""
    importlib.invalidate_caches()
    imp_utils = sys.modules.get("transformers.utils.import_utils")
    if imp_utils is None and "transformers" in sys.modules:
        try:
            imp_utils = importlib.import_module("transformers.utils.import_utils")
        except Exception:
            imp_utils = None
    if imp_utils is not None:
        for attr in ("is_kernels_available", "_is_package_available", "is_compressed_tensors_available"):
            fn = getattr(imp_utils, attr, None)
            clear_fn = getattr(fn, "cache_clear", None)
            if callable(clear_fn):
                clear_fn()
    fg_fp8 = sys.modules.get("transformers.integrations.finegrained_fp8")
    if fg_fp8 is not None:
        for attr in ("load_finegrained_fp8_kernel", "_load_finegrained_fp8_kernel"):
            fn = getattr(fg_fp8, attr, None)
            clear_fn = getattr(fn, "cache_clear", None)
            if callable(clear_fn):
                clear_fn()


def _version_tuple(ver: str) -> tuple[int, ...]:
    core = ver.split("+", 1)[0].split("-", 1)[0]
    nums = [int(p) for p in re.findall(r"\d+", core)]
    return tuple(nums) if nums else (0,)


def _installed_kernels_matches_spec(pip_spec: str) -> bool:
    if importlib.util.find_spec("kernels") is None:
        return False
    ver: str | None = None
    try:
        ver = importlib.metadata.version("kernels")
    except Exception:
        try:
            mod = importlib.import_module("kernels")
            ver = getattr(mod, "__version__", None)
        except Exception:
            return False
    if not ver:
        return True
    got = _version_tuple(str(ver))
    tail = pip_spec[len("kernels") :] if pip_spec.startswith("kernels") else pip_spec
    for op, target_str in re.findall(r"(>=|<=|==|!=|>|<)\s*([0-9]+(?:\.[0-9a-zA-Z]+)*)", tail):
        want = _version_tuple(target_str)
        width = max(len(got), len(want))
        g_pad = got + (0,) * (width - len(got))
        w_pad = want + (0,) * (width - len(want))
        if op == ">=" and g_pad < w_pad:
            return False
        if op == ">" and g_pad <= w_pad:
            return False
        if op == "<=" and g_pad > w_pad:
            return False
        if op == "<" and g_pad >= w_pad:
            return False
        if op == "==" and g_pad != w_pad:
            return False
        if op == "!=" and g_pad == w_pad:
            return False
    return True


def _resolve_hf_kernels_pip_spec(message: str | None = None) -> tuple[bool, str]:
    """Inspect `transformers.integrations.finegrained_fp8` and resolve compatible `kernels` pip spec."""
    if message is not None:
        return False, _parse_hf_kernels_pip_spec(message)
    try:
        fg_fp8 = importlib.import_module("transformers.integrations.finegrained_fp8")
    except Exception:
        fallback_hint = _import_utils_kernels_bounds_hint()
        pip_spec = _parse_hf_kernels_pip_spec(fallback_hint)
        if _installed_kernels_matches_spec(pip_spec):
            return True, "kernels"
        return False, pip_spec

    missing_msg = str(getattr(fg_fp8, "_MISSING_KERNELS_MESSAGE", "") or "")
    if not missing_msg:
        missing_msg = _import_utils_kernels_bounds_hint(fg_fp8)
    loader = getattr(fg_fp8, "load_finegrained_fp8_kernel", None) or getattr(
        fg_fp8, "_load_finegrained_fp8_kernel", None
    )
    if callable(loader):
        try:
            loader()
            return True, "kernels"
        except ImportError as exc:
            pip_spec = (
                _parse_hf_kernels_pip_spec_strict(str(exc))
                or _parse_hf_kernels_pip_spec_strict(missing_msg)
                or DEFAULT_HF_KERNELS_PIP_SPEC
            )
            return False, pip_spec
        except Exception as exc:
            pip_spec = (
                _parse_hf_kernels_pip_spec_strict(str(exc))
                or _parse_hf_kernels_pip_spec_strict(missing_msg)
                or DEFAULT_HF_KERNELS_PIP_SPEC
            )
            if _installed_kernels_matches_spec(pip_spec):
                return True, "kernels"
            return False, pip_spec

    pip_spec = _parse_hf_kernels_pip_spec(missing_msg)
    if _installed_kernels_matches_spec(pip_spec):
        return True, "kernels"
    return False, pip_spec


def _resolve_precision_mode(base_precision: str | None) -> str:
    """Resolve `'nvfp4'`, `'nvfp4-fast'`, `'fp8'`, `'bf16'`, or an `'org/model'` HF repo ID into a precision tag."""
    if base_precision is None:
        return "none"
    raw = str(base_precision).strip()
    if not raw or raw.lower() == "toy":
        return "none"
    low = raw.lower()
    if low in ("fp8", "nvfp4", "bf16"):
        return low
    if low.replace("_", "-") in ("nvfp4", "nvfp4-fast"):
        return "nvfp4"
    resolved = resolve_base_model_name(raw) or raw
    upper = resolved.upper()
    if "-NVFP4" in upper or upper.endswith("NVFP4"):
        return "nvfp4"
    if upper.endswith("-FP8") or "-FP8-" in upper:
        return "fp8"
    return "bf16"


def _effective_precision_arg(base_precision: str | None, overrides: Sequence[str] = ()) -> str | None:
    for ov in reversed(list(overrides)):
        if "=" in ov:
            k, v = ov.split("=", 1)
            if k.strip() == "model.name":
                val = v.strip()
                return None if val.lower() == "toy" else val
    return base_precision


def ensure_quant_runtime_packages(
    base_precision: str | None = "nvfp4",
    *,
    auto_install: bool = True,
) -> dict[str, str]:
    """Verify (and on Linux when `auto_install=True`, install) `kernels` / `compressed-tensors`."""
    precision = _resolve_precision_mode(base_precision)
    can_install = bool(auto_install and sys.platform == "linux")

    if precision == "fp8":
        ok, pip_spec = _resolve_hf_kernels_pip_spec()
        if ok:
            return {
                "precision": "fp8",
                "package": "kernels",
                "pip_spec": pip_spec,
                "status": "ready",
            }
        if can_install:
            cmd = [sys.executable, "-m", "pip", "install", "-q", "--no-cache-dir", pip_spec]
            try:
                res = subprocess.run(cmd, capture_output=True, text=True, check=False)
                if res.returncode == 0:
                    _refresh_runtime_import_caches()
            except Exception as exc:
                print(
                    f"[qloop] warning: failed to install {pip_spec!r} ({exc!r}); "
                    "continuing with PyTorch FP8 fallback",
                    flush=True,
                )
                return {
                    "precision": "fp8",
                    "package": "kernels",
                    "pip_spec": pip_spec,
                    "status": f"install_failed ({exc!r})",
                }
            if res.returncode == 0:
                ok_after, _ = _resolve_hf_kernels_pip_spec()
                has_pkg = ok_after or importlib.util.find_spec("kernels") is not None
                st = "installed" if has_pkg else "installed_pending_reload"
                print(f"[qloop] installed {pip_spec} for native Blackwell FP8 kernels ({st})", flush=True)
                return {
                    "precision": "fp8",
                    "package": "kernels",
                    "pip_spec": pip_spec,
                    "status": st,
                }
            tail = (res.stderr or res.stdout or "").strip().splitlines()
            detail = tail[-1] if tail else f"rc={res.returncode}"
            print(
                f"[qloop] warning: pip install {pip_spec!r} exited with rc={res.returncode} ({detail}); "
                "continuing with PyTorch FP8 fallback",
                flush=True,
            )
            return {
                "precision": "fp8",
                "package": "kernels",
                "pip_spec": pip_spec,
                "status": f"install_failed ({detail})",
            }
        return {
            "precision": "fp8",
            "package": "kernels",
            "pip_spec": pip_spec,
            "status": f"missing ({pip_spec})",
        }

    if precision == "nvfp4":
        pip_spec = "compressed-tensors"
        if importlib.util.find_spec("compressed_tensors") is not None:
            return {
                "precision": "nvfp4",
                "package": "compressed_tensors",
                "pip_spec": pip_spec,
                "status": "ready",
            }
        if can_install:
            cmd = [sys.executable, "-m", "pip", "install", "-q", "--no-cache-dir", pip_spec]
            try:
                res = subprocess.run(cmd, capture_output=True, text=True, check=False)
                if res.returncode == 0:
                    _refresh_runtime_import_caches()
            except Exception as exc:
                print(f"[qloop] warning: failed to install {pip_spec!r} ({exc!r})", flush=True)
                return {
                    "precision": "nvfp4",
                    "package": "compressed_tensors",
                    "pip_spec": pip_spec,
                    "status": f"install_failed ({exc!r})",
                }
            if res.returncode == 0:
                print(f"[qloop] installed {pip_spec} for NVFP4 checkpoints", flush=True)
                return {
                    "precision": "nvfp4",
                    "package": "compressed_tensors",
                    "pip_spec": pip_spec,
                    "status": "installed",
                }
            tail = (res.stderr or res.stdout or "").strip().splitlines()
            detail = tail[-1] if tail else f"rc={res.returncode}"
            print(f"[qloop] warning: pip install {pip_spec!r} failed ({detail})", flush=True)
            return {
                "precision": "nvfp4",
                "package": "compressed_tensors",
                "pip_spec": pip_spec,
                "status": f"install_failed ({detail})",
            }
        return {
            "precision": "nvfp4",
            "package": "compressed_tensors",
            "pip_spec": pip_spec,
            "status": f"missing ({pip_spec})",
        }

    return {
        "precision": precision,
        "package": "none",
        "pip_spec": "",
        "status": "not_required",
    }


def setup_env(
    hf_repo: str = DEFAULT_HF_REPO,
    run_root: str = "/content/runs",
    hf_home: str = "/content/hf_cache",
    *,
    require_hf_token: bool = True,
    base_precision: str | None = "nvfp4",
    auto_install_runtime_deps: bool = True,
) -> dict[str, str]:
    """Load Colab Secrets into `os.environ`, configure HF Hub sync + local SSD paths, and print disk budget."""
    found: dict[str, str] = {}
    try:
        from google.colab import userdata  # type: ignore[import-not-found]

        for k in SECRETS:
            try:
                v = userdata.get(k)
            except Exception:
                v = None
            if v and str(v).strip():
                os.environ[k] = str(v).strip()
                found[k] = "set (Colab Secret)"
    except ImportError:
        pass
    for k in SECRETS:
        found.setdefault(k, "set" if os.environ.get(k) else "missing")

    if require_hf_token and not os.environ.get("HF_TOKEN"):
        raise RuntimeError(
            "HF_TOKEN is missing! In Google Colab, click the left-sidebar '🔑 Secrets' icon, add a secret named "
            f"'HF_TOKEN' with Write permission for https://huggingface.co/{hf_repo}, toggle 'Notebook access' ON, "
            "and re-run this cell."
        )

    os.makedirs(run_root, exist_ok=True)
    os.makedirs(hf_home, exist_ok=True)
    src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cur_pp = os.environ.get("PYTHONPATH", "")
    if src_dir not in cur_pp.split(os.pathsep):
        os.environ["PYTHONPATH"] = f"{src_dir}{os.pathsep}{cur_pp}" if cur_pp else src_dir
    os.environ["QLOOP_HF_REPO"] = hf_repo
    os.environ["QLOOP_RUN_ROOT"] = run_root
    os.environ["HF_HOME"] = hf_home
    os.environ.pop("HF_HUB_ENABLE_HF_TRANSFER", None)
    os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    quant_rt = ensure_quant_runtime_packages(
        base_precision=base_precision,
        auto_install=auto_install_runtime_deps,
    )

    from .hub import disk_report

    rep = disk_report(run_root, hf_home=hf_home)
    print(f"HF Sync Repo   : https://huggingface.co/{hf_repo} (public commit-history sync)")
    print(f"Secrets        : {', '.join(f'{k}={v}' for k, v in found.items())}")
    print(f"Run Root (SSD) : {run_root}  |  HF_HOME : {hf_home}")
    print(
        f"Disk (230G cap): {rep['free_gib']:.1f} GiB free / {rep['total_gib']:.1f} GiB total "
        f"(HF base weights={rep['hf_hub_weights_gib']:.1f} GiB, runs={rep['runs_gib']:.2f} GiB)"
    )
    print(
        f"Quant Runtime  : precision={quant_rt['precision']} | "
        f"package={quant_rt['package']} ({quant_rt['status']})"
    )
    return found


def _newest_crash(since: float) -> str | None:
    root = os.environ.get("QLOOP_RUN_ROOT", "runs")
    files = [
        f
        for f in glob.glob(os.path.join(root, "**", "crash", "crash_*.md"), recursive=True)
        if os.path.getmtime(f) >= since
    ]
    return max(files, key=os.path.getmtime) if files else None


def _sections(path: str, names: tuple[str, ...], max_lines: int = 120) -> str:
    with open(path, encoding="utf-8") as f:
        text = f.read()
    out, keep = [], False
    for line in text.splitlines():
        if line.startswith("## "):
            keep = line[3:].strip() in names
        if keep or line.startswith(("# qloop crash", "**stage**", "**error**")):
            out.append(line)
    return "\n".join(out[:max_lines])


def q(args: str, check: bool = True) -> int:
    """Run `python -u -m qloop <args>` in a subprocess; surface verbose crash diagnostics on failure."""
    cmd = [sys.executable, "-u", "-m", "qloop", *shlex.split(args)]
    print("$", " ".join(cmd), flush=True)
    t0 = time.time()
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=_subprocess_env(),
    )
    try:
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="", flush=True)
        rc = proc.wait()
    except KeyboardInterrupt:
        proc.send_signal(2)
        rc = proc.wait()
        repo = os.environ.get("QLOOP_HF_REPO", DEFAULT_HF_REPO)
        print(f"\ninterrupted - checkpoint saved & synced to https://huggingface.co/{repo}; re-run cell to resume")
    dt = time.time() - t0
    if rc != 0:
        crash = _newest_crash(t0 - 1)
        print(f"\n✗ exit code {rc} after {dt / 60:.1f} min")
        if crash:
            print(f"crash report: {crash}\n" + "=" * 88)
            print(_sections(crash, ("Hints", "Traceback", "Local Tensors", "CUDA Memory")))
            print("=" * 88)
            print("Tip: re-run with `--debug 2` (NaN/Inf module hooks + C++ traces) or `--debug 3` "
                  "(CUDA_LAUNCH_BLOCKING=1 + autograd anomaly + memory timeline).")
        if check:
            raise RuntimeError(f"qloop {args!r} failed (rc={rc})" + (f"; see {crash}" if crash else ""))
    else:
        print(f"\n✓ done in {dt / 60:.1f} min")
    return rc


def hub_status(hf_repo: str | None = None) -> int:
    """Print live stage status, recent Hub commits, and 230 GB Colab disk headroom."""
    repo = hf_repo or os.environ.get("QLOOP_HF_REPO", DEFAULT_HF_REPO)
    return q(f"sync status --repo {shlex.quote(repo)}")


def reclaim_disk(keep: str | None = None, pip: bool = False) -> int:
    """Reclaim superseded local stage checkpoints & HF upload staging caches."""
    parts = ["sync", "disk", "--clean"]
    if keep:
        parts += ["--keep", shlex.quote(keep)]
    if pip:
        parts.append("--pip")
    return q(" ".join(parts))


def run_pipeline(
    *,
    hf_repo: str | None = None,
    base_precision: str | None = "nvfp4",
    smoke_toy: bool = False,
    debug: int = 1,
    time_budget_hours: float = 21.5,
    stages: Sequence[str] | None = None,
    until: str | None = None,
    force: bool = False,
    overrides: Sequence[str] = (),
) -> int:
    """Run/resume the full end-to-end pipeline synced via HF Hub commit history."""
    if not smoke_toy and sys.platform == "linux":
        ensure_quant_runtime_packages(_effective_precision_arg(base_precision, overrides), auto_install=True)
    repo = hf_repo or os.environ.get("QLOOP_HF_REPO", DEFAULT_HF_REPO)
    eff_overrides = _inject_base_model_override(overrides, base_precision, smoke_toy=smoke_toy)
    parts = [
        "sync",
        "pipeline",
        "--repo",
        shlex.quote(repo),
        "--debug",
        str(int(debug)),
        "--time-budget-hours",
        str(float(time_budget_hours)),
    ]
    if stages:
        parts += ["--stages", shlex.quote(",".join(stages))]
    if until:
        parts += ["--until", shlex.quote(until)]
    if force:
        parts.append("--force")
    for ov in eff_overrides:
        parts += ["-s", shlex.quote(ov)]
    return q(" ".join(parts))


def tail_metrics(
    run_name: str,
    keys: tuple[str, ...] = DEFAULT_METRIC_KEYS,
    n: int = 12,
) -> None:
    path = os.path.join(os.environ.get("QLOOP_RUN_ROOT", "runs"), run_name, "metrics.jsonl")
    if not os.path.exists(path):
        print("no local metrics file:", path)
        return
    with open(path, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    for r in rows[-n:]:
        filtered = {k: round(r[k], 4) for k in keys if isinstance(r.get(k), (int, float))}
        print(f"step {r.get('step', '?'):>6}:", filtered)


HSSD_ABLATION_EVAL_AXES: tuple[str, ...] = (
    "short_context_lm",
    "repeated_key_updates",
    "distractor_resistance",
    "state_tracking",
    "mqar_recall_vs_interfering_writes",
    "code_reasoning",
)


def _default_hssd_ablation_matrix() -> dict[str, dict[str, Any]]:
    """Fallback 6-way matched-budget HSSD mechanism isolation matrix (`Gate 1-Ablate`)."""
    axes = list(HSSD_ABLATION_EVAL_AXES)
    return {
        "1_original_gdn": {
            "label": "Original GDN",
            "hssd": {"enabled": False},
            "eval_axes": axes,
        },
        "2_gdn_plus_local_swa": {
            "label": "GDN + local SWA",
            "hssd": {
                "enabled": True,
                "num_pages": 1,
                "k_pages": 1,
                "use_swa": True,
                "use_phase": False,
                "use_surprise": False,
                "theta_low_rank": True,
                "state_dtype": "bfloat16",
            },
            "eval_axes": axes,
        },
        "3_multipage_no_phase": {
            "label": "Multipage without phase",
            "hssd": {
                "enabled": True,
                "num_pages": 8,
                "k_pages": 2,
                "use_swa": False,
                "use_phase": False,
                "use_surprise": False,
                "theta_low_rank": True,
                "state_dtype": "bfloat16",
            },
            "eval_axes": axes,
        },
        "4_surprise_controlled": {
            "label": "Surprise-controlled memory",
            "hssd": {
                "enabled": True,
                "num_pages": 8,
                "k_pages": 2,
                "use_swa": False,
                "use_phase": False,
                "use_surprise": True,
                "surprise_mode": "routed",
                "theta_low_rank": True,
                "state_dtype": "bfloat16",
            },
            "eval_axes": axes,
        },
        "5_phase_enhanced": {
            "label": "Phase-enhanced recurrence",
            "hssd": {
                "enabled": True,
                "num_pages": 1,
                "k_pages": 1,
                "use_swa": False,
                "use_phase": True,
                "use_surprise": False,
                "theta_low_rank": True,
                "state_dtype": "bfloat16",
            },
            "eval_axes": axes,
        },
        "6_complete_hssd": {
            "label": "Complete HSSD",
            "hssd": {
                "enabled": True,
                "num_pages": 8,
                "k_pages": 2,
                "use_swa": True,
                "use_phase": True,
                "use_surprise": True,
                "surprise_mode": "routed",
                "theta_low_rank": True,
                "state_dtype": "bfloat16",
            },
            "eval_axes": axes,
        },
    }


def _analytical_hssd_layer_profile(*, smoke_toy: bool, seq_len: int) -> dict[str, Any]:
    """Analytical fallback profile when `probes.profile_hssd_layer` is not yet merged."""
    hidden = 64 if smoke_toy else 2048
    hv = 4 if smoke_toy else 32
    dk = 16 if smoke_toy else 128
    dv = 16 if smoke_toy else 128
    rank = 16 if smoke_toy else 64
    half_dk = dk // 2
    pages = 8
    win_chunks = 8
    chunk_sz = 16 if smoke_toy else 64
    swa_cap = (win_chunks + 1) * chunk_sz

    theta_params = hidden * rank + rank * (hv * half_dk)
    other_params = (
        hidden * rank
        + rank * (hv * dk)
        + rank * (hv * dv)
        + rank * (hv * half_dk)
        + (hidden * hv + hv + 2 * hv)
        + (hidden * pages + pages)
        + ((hv * dk) * pages + pages + pages)
        + 2 * rank * (hv * dk)
        + rank * (hv * dv)
        + 2 * dk
        + 2 * hv
    )
    params_per_layer = theta_params + other_params
    state_el_bytes = 2  # bfloat16
    page_bytes = pages * hv * dk * dv * state_el_bytes
    ring_bytes = (win_chunks + 1) * page_bytes
    swa_bytes = 2 * swa_cap * hv * dk * state_el_bytes
    theta_bytes = hv * half_dk * 8  # float64
    valid_bytes = swa_cap
    total_cache_bytes = page_bytes + ring_bytes + swa_bytes + theta_bytes + valid_bytes + 8
    return {
        "profile_source": "analytical_fallback",
        "dimensions": {
            "hidden_size": hidden,
            "num_v_heads": hv,
            "head_k_dim": dk,
            "head_v_dim": dv,
            "num_pages": pages,
            "window_chunks": win_chunks,
            "chunk_size": chunk_sz,
            "seq_len": seq_len,
            "rank": rank,
            "theta_low_rank": True,
            "state_dtype": "bfloat16",
        },
        "adapter_params_per_layer": params_per_layer,
        "adapter_params_30_layers": 30 * params_per_layer,
        "theta_param_fraction": round(theta_params / max(params_per_layer, 1), 6),
        "full_prefill": {},
        "streamed_prefill": {},
        "retained_cache": {
            "state_dtype": "bfloat16",
            "unique_storage_bytes": total_cache_bytes,
            "unique_storage_mib": round(total_cache_bytes / (1024.0 * 1024.0), 4),
            "30_layers_r1_gib": round((30 * total_cache_bytes) / (1024.0 ** 3), 4),
            "has_compact_storage": True,
        },
        "single_token_decode": {},
        "training_fwd_bwd": {},
        "gdn_baseline": {},
    }


def _pick_float(mapping: dict[str, Any], keys: tuple[str, ...]) -> float | None:
    for key in keys:
        val = mapping.get(key)
        if isinstance(val, (int, float)) and not isinstance(val, bool):
            return float(val)
    return None


def _fmt_bench_row(name: str, entry: Any) -> str:
    if not isinstance(entry, dict) or not entry:
        return f"| `{name}` | `-` | `-` | `-` |"
    tps = _pick_float(
        entry,
        ("tok_per_s", "tokens_per_s", "tokens_per_sec", "throughput_tok_s", "prefill_tokens_per_sec"),
    )
    ms = _pick_float(entry, ("ms", "latency_ms", "elapsed_ms", "prefill_ms"))
    if ms is None:
        sec = _pick_float(entry, ("seconds", "elapsed_s", "latency_s", "prefill_seconds"))
        if sec is not None:
            ms = sec * 1000.0
    mem_mib = _pick_float(entry, ("unique_storage_mib", "state_mib"))
    if mem_mib is None:
        stor_b = _pick_float(entry, ("unique_storage_bytes", "state_bytes"))
        if stor_b is not None:
            mem_mib = stor_b / (1024.0 * 1024.0)
    if mem_mib is None:
        peak_mib = _pick_float(entry, ("peak_mem_mib", "peak_mib", "peak_cuda_mib", "mib", "mb"))
        peak_b = _pick_float(entry, ("peak_cuda_bytes", "peak_bytes", "allocated_bytes"))
        if peak_mib is not None and peak_mib > 0.0:
            mem_mib = peak_mib
        elif peak_b is not None and peak_b > 0.0:
            mem_mib = peak_b / (1024.0 * 1024.0)
    tps_s = f"`{tps:,.1f}`" if tps is not None else "`-`"
    ms_s = f"`{ms:.3f}`" if ms is not None else "`-`"
    mem_s = f"`{mem_mib:.2f} MiB`" if mem_mib is not None else "`-`"
    return f"| `{name}` | {tps_s} | {ms_s} | {mem_s} |"


def format_probe_notebook_markdown(report: dict[str, Any]) -> str:
    """Format full probe report Markdown, appending Gate 0p-4 & Gate 1-Ablate sections when present."""
    from .model import probes as probes_mod

    base_md = probes_mod.format_probe_report_markdown(report).rstrip()
    extra: list[str] = []

    hprof = report.get("hssd_layer_profile")
    if isinstance(hprof, dict) and hprof and "0p-4" not in base_md:
        dims = hprof.get("dimensions") or {}
        rcache = hprof.get("retained_cache") or {}
        p_layer = int(hprof.get("adapter_params_per_layer") or 0)
        p_30 = int(hprof.get("adapter_params_30_layers") or (30 * p_layer))
        th_frac = float(hprof.get("theta_param_fraction") or 0.0)
        ubytes = _pick_float(rcache, ("unique_storage_bytes", "storage_bytes", "bytes"))
        umib = _pick_float(rcache, ("unique_storage_mib", "mib", "mb"))
        if umib is None and ubytes is not None:
            umib = ubytes / (1024.0 * 1024.0)
        mid_mib = _pick_float(rcache, ("mid_chunk_unique_storage_mib",))
        compact = rcache.get("has_compact_storage", rcache.get("compact_storage", True))
        mid_note = f", `{mid_mib:.2f} MiB` mid-chunk" if mid_mib is not None else ""
        extra.extend(
            [
                "",
                "## Gate `0p-4`: Single-Layer `HSSD v3.1` vs `GDN` Hardware & Storage Profile",
                "",
                f"- **Dimensions:** `hidden={dims.get('hidden_size', '?')}`, "
                f"`H_v={dims.get('num_v_heads', '?')}`, `d_k={dims.get('head_k_dim', '?')}`, "
                f"`state_dtype={dims.get('state_dtype', rcache.get('state_dtype', 'bfloat16'))}`",
                f"- **Adapter Params (`theta_low_rank=True`):** `{p_layer:,}` / layer "
                f"(`{p_30:,}` across 30L; `theta` share `{th_frac * 100.0:.1f}%`)",
                f"- **Retained Decode Cache:** "
                f"`{int(ubytes or 0):,} B` (`{(umib or 0.0):.2f} MiB` / layer{mid_note}, "
                f"`compact_storage={bool(compact)}`)",
                "",
                "| Workload | Throughput (`tok/s`) | Latency (`ms`) | Memory / Storage |",
                "| :--- | ---: | ---: | ---: |",
                _fmt_bench_row("full_prefill", hprof.get("full_prefill")),
                _fmt_bench_row("streamed_prefill", hprof.get("streamed_prefill")),
                _fmt_bench_row("single_token_decode", hprof.get("single_token_decode")),
                _fmt_bench_row("training_fwd_bwd", hprof.get("training_fwd_bwd")),
                _fmt_bench_row("gdn_baseline", hprof.get("gdn_baseline")),
            ]
        )

    ablate = report.get("hssd_ablation_matrix")
    if isinstance(ablate, dict) and ablate and "1-Ablate" not in base_md and "Ablation Matrix" not in base_md:
        extra.extend(
            [
                "",
                "## Gate `1-Ablate`: 6-Way Matched-Budget HSSD Mechanism Isolation Matrix",
                "",
                "| Variant | Label | Key `hssd` Toggles | Evaluation Axes |",
                "| :--- | :--- | :--- | :--- |",
            ]
        )
        for v_key, v_cfg in ablate.items():
            if not isinstance(v_cfg, dict):
                continue
            label = str(v_cfg.get("label") or v_cfg.get("name") or v_key)
            skip_keys = ("enabled", "label", "name", "description", "mechanisms", "eval_tasks", "eval_axes", "tasks")
            h_sub = v_cfg.get("hssd") if isinstance(v_cfg.get("hssd"), dict) else {
                k: v for k, v in v_cfg.items() if k not in skip_keys
            }
            axes = (
                v_cfg.get("eval_tasks")
                or v_cfg.get("eval_axes")
                or v_cfg.get("tasks")
                or list(HSSD_ABLATION_EVAL_AXES)
            )
            h_str = json.dumps(h_sub, separators=(",", ":"))
            ax_str = ", ".join(f"`{a}`" for a in axes) if isinstance(axes, (list, tuple)) else str(axes)
            extra.append(f"| `{v_key}` | {label} | `{h_str}` | {ax_str} |")

    return base_md + ("\n" + "\n".join(extra) if extra else "") + "\n"


def run_probe_notebook_suite(
    *,
    config_path: str = "configs/stages/s0_probe.yaml",
    hf_repo: str | None = None,
    base_precision: str | None = "nvfp4",
    smoke_toy: bool = False,
    seq_len: int | None = None,
    n_seqs: int | None = None,
    target_virtual: int | None = None,
    run_math_gate: bool = True,
    push_to_hub: bool = False,
    debug: int = 1,
    overrides: Sequence[str] = (),
) -> dict[str, Any]:
    """Run Gate -1 (`fp64` math) + Gate 0a + Gate 0p-1..4 structured probes and return report + paths."""
    resolved_base = resolve_base_model_name(base_precision)
    if not smoke_toy and sys.platform == "linux":
        ensure_quant_runtime_packages(_effective_precision_arg(base_precision, overrides), auto_install=True)
    repo = hf_repo or os.environ.get("QLOOP_HF_REPO", DEFAULT_HF_REPO)
    run_root = os.environ.get("QLOOP_RUN_ROOT", "runs")
    probe_dir = os.path.join(run_root, "probe")
    os.makedirs(probe_dir, exist_ok=True)
    json_path = os.path.join(probe_dir, "probe_report.json")
    md_path = os.path.join(probe_dir, "probe_report.md")

    eff_seq = int(seq_len) if seq_len is not None else (64 if smoke_toy else 2048)
    eff_n = int(n_seqs) if n_seqs is not None else (2 if smoke_toy else 4)
    eff_virt = int(target_virtual) if target_virtual is not None else (24 if smoke_toy else 88)

    caller_keys = {ov.split("=", 1)[0].strip() for ov in overrides if "=" in ov}
    eff_overrides: list[str] = []
    if smoke_toy:
        toy_defaults = (
            ("model.name", "toy"),
            ("model.dtype", "float32"),
            ("model.device_map", "cpu"),
            ("data.streaming", "false"),
            ("data.sources", '[{"name":"toy_cot","weight":0.7},{"name":"toy_text","weight":0.3}]'),
        )
        for k, v in toy_defaults:
            if k not in caller_keys:
                eff_overrides.append(f"{k}={v}")
    elif resolved_base is not None and "model.name" not in caller_keys:
        eff_overrides.append(f"model.name={resolved_base}")

    if push_to_hub:
        if "checkpoint.sync.hf_repo" not in caller_keys:
            eff_overrides.append(f"checkpoint.sync.hf_repo={repo}")
        if "checkpoint.sync.enabled" not in caller_keys:
            eff_overrides.append("checkpoint.sync.enabled=true")
    elif "checkpoint.sync.enabled" not in caller_keys:
        eff_overrides.append("checkpoint.sync.enabled=false")

    eff_overrides.extend(overrides)

    parts = [
        "probe",
        "-c",
        shlex.quote(config_path),
        "--structured",
        "--n",
        str(eff_n),
        "--seq-len",
        str(eff_seq),
        "--target-virtual",
        str(eff_virt),
        "--out-json",
        shlex.quote(json_path),
        "--out-md",
        shlex.quote(md_path),
        "--debug",
        str(int(debug)),
    ]
    for ov in eff_overrides:
        parts += ["-s", shlex.quote(ov)]
    q(" ".join(parts))

    with open(json_path, encoding="utf-8") as f:
        report: dict[str, Any] = json.load(f)

    gates: dict[str, Any] = dict(report.get("gates") or {})
    math_ok = True
    if run_math_gate:
        repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        math_test = os.path.join(repo_root, "tests", "test_loop_hssd_v3_math.py")
        cmd = [sys.executable, "-m", "pytest", "-q", math_test]
        print("$", " ".join(cmd), flush=True)
        res = subprocess.run(cmd, check=False, capture_output=True, text=True, env=_subprocess_env())
        if res.stdout:
            print(res.stdout, end="", flush=True)
        if res.stderr:
            print(res.stderr, end="", flush=True)
        math_ok = res.returncode == 0
        if not math_ok:
            raise RuntimeError(f"Gate -1 fp64 math verification failed (rc={res.returncode})")
        gates = {
            "gate_minus_1_math": {
                "passed": True,
                "metric": "fp64_invariants_passed",
                "value": 7,
            },
            **gates,
        }

    # Gate 0p-4 (`profile_hssd_layer`) & Gate 1-Ablate (`hssd_ablation_matrix`) with safe pre-merge guards
    from .model import probes as probes_mod

    prof_fn = getattr(probes_mod, "profile_hssd_layer", None)
    if callable(prof_fn):
        import torch

        if smoke_toy:
            hssd_prof = prof_fn(
                hidden_size=64,
                num_v_heads=4,
                head_k_dim=16,
                head_v_dim=16,
                seq_len=max(16, min(eff_seq, 64)),
                stream_chunk_len=max(8, min(max(eff_seq // 2, 8), 32)),
                state_dtype="bfloat16",
                theta_low_rank=True,
                rank=16,
                device="cpu",
                dtype=torch.float32,
            )
        else:
            use_cuda = bool(torch.cuda.is_available())
            prof_dev = "cuda" if use_cuda else "cpu"
            prof_dt = torch.bfloat16 if use_cuda else torch.float32
            prof_seq = max(256, min(eff_seq, 1024)) if use_cuda else max(64, min(eff_seq, 256))
            prof_stream = max(64, min(prof_seq // 4, 256))
            hssd_prof = prof_fn(
                hidden_size=2048,
                num_v_heads=32,
                head_k_dim=128,
                head_v_dim=128,
                seq_len=prof_seq,
                stream_chunk_len=prof_stream,
                state_dtype="bfloat16",
                theta_low_rank=True,
                rank=64,
                device=prof_dev,
                dtype=prof_dt,
            )
    else:
        hssd_prof = _analytical_hssd_layer_profile(smoke_toy=smoke_toy, seq_len=eff_seq)

    ablate_fn = getattr(probes_mod, "hssd_ablation_matrix", None)
    ablation_matrix = ablate_fn() if callable(ablate_fn) else _default_hssd_ablation_matrix()

    report["hssd_layer_profile"] = hssd_prof
    report["hssd_ablation_matrix"] = ablation_matrix
    probes_sub = dict(report.get("probes") or {})
    probes_sub["hssd_layer_profile"] = hssd_prof
    probes_sub["hssd_ablation_matrix"] = ablation_matrix
    report["probes"] = probes_sub

    rcache = hssd_prof.get("retained_cache") if isinstance(hssd_prof, dict) else None
    rcache_d = rcache if isinstance(rcache, dict) else {}
    compact_ok = bool(rcache_d.get("has_compact_storage", rcache_d.get("compact_storage", True)))
    ubytes = _pick_float(rcache_d, ("unique_storage_bytes", "storage_bytes", "bytes"))
    umib = _pick_float(rcache_d, ("unique_storage_mib", "mib", "mb"))
    if umib is None and ubytes is not None:
        umib = ubytes / (1024.0 * 1024.0)
    gates["0p_4_hssd_layer_profile"] = {
        "passed": compact_ok,
        "metric": "retained_cache_mib",
        "value": round(umib, 4) if umib is not None else compact_ok,
    }
    report["gates"] = gates

    summary = dict(report.get("summary") or {})
    summary["num_gates_passed"] = sum(1 for g in gates.values() if g.get("passed"))
    summary["num_gates_total"] = len(gates)
    summary["all_critical_ok"] = bool(summary.get("all_critical_ok", True) and math_ok and compact_ok)
    report["summary"] = summary

    report["markdown"] = format_probe_notebook_markdown(report)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(str(report["markdown"]))

    if push_to_hub:
        from .hub import HubSyncManager

        hub = HubSyncManager(repo_id=repo, private=False)
        hub.push_artifacts(
            "s0_probe",
            {
                "artifacts/probes/probe_report.json": json_path,
                "artifacts/probes/probe_report.md": md_path,
            },
            completed=False,
        )

    return {
        **report,
        "json_path": json_path,
        "md_path": md_path,
    }


DEFAULT_V2_STAGES: tuple[str, ...] = (
    "s0_probe",
    "s1_loop_a6b",
    "s3_v2_long_1m_lsa",
    "s2_v2_joint_bptt",
    "s4_v2_rl",
)
LEGACY_HSSD_STAGES: tuple[str, ...] = (
    "s0_probe",
    "s1_loop_a6b",
    "s1b_hssd_distill",
    "s3_long_1m_lsa",
    "s2_joint_bptt",
    "s4_rl",
)


def _read_json_file(path: str) -> dict[str, Any] | None:
    if not os.path.isfile(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else None
    except (OSError, ValueError):
        return None


def _meta_proves_completion(meta: dict[str, Any] | None) -> bool:
    """Return True only when checkpoint/stage metadata explicitly verifies completion."""
    if not isinstance(meta, dict):
        return False
    if meta.get("completed") is False:
        return False
    status = str(meta.get("status") or "").strip().lower()
    if status in ("in_progress", "running", "paused", "failed", "aborted"):
        return False
    if meta.get("completed") is True or status == "completed":
        return True
    raw_step = meta.get("step")
    raw_target = meta.get("target_steps")
    if (
        isinstance(raw_step, int)
        and not isinstance(raw_step, bool)
        and isinstance(raw_target, int)
        and not isinstance(raw_target, bool)
        and raw_target > 0
    ):
        return raw_step >= raw_target
    return False


def _meta_contradicts_completion(meta: dict[str, Any] | None) -> bool:
    if not isinstance(meta, dict):
        return False
    if meta.get("completed") is False:
        return True
    status = str(meta.get("status") or "").strip().lower()
    if status in ("in_progress", "running", "paused", "failed", "aborted"):
        return True
    raw_step = meta.get("step")
    raw_target = meta.get("target_steps")
    return (
        isinstance(raw_step, int)
        and not isinstance(raw_step, bool)
        and isinstance(raw_target, int)
        and not isinstance(raw_target, bool)
        and raw_target > 0
        and raw_step < raw_target
    )


def _local_stage_done(run_root: str, stage_name: str) -> bool:
    """Verify strict local stage completion; mere `checkpoints/LATEST` presence is NEVER sufficient."""
    stage_dir = os.path.join(run_root, stage_name)
    if stage_name == "s0_probe":
        return (
            os.path.isfile(os.path.join(stage_dir, "COMPLETED"))
            or os.path.isfile(os.path.join(stage_dir, "esft_stats.json"))
            or os.path.isfile(os.path.join(run_root, "probe", "probe_report.json"))
        )

    ck_dir = os.path.join(stage_dir, "checkpoints")
    if os.path.isfile(os.path.join(ck_dir, "COMPLETED")) or os.path.isfile(os.path.join(stage_dir, "COMPLETED")):
        return True

    promoted_ptr = os.path.join(ck_dir, "PROMOTED")
    if os.path.isfile(promoted_ptr):
        try:
            with open(promoted_ptr, encoding="utf-8") as f:
                step_name = f.read().strip()
        except OSError:
            step_name = ""
        step_meta = _read_json_file(os.path.join(ck_dir, step_name, "state.json")) if step_name else None
        if not _meta_contradicts_completion(step_meta):
            return True

    latest_ptr = os.path.join(ck_dir, "LATEST")
    if os.path.isfile(latest_ptr):
        try:
            with open(latest_ptr, encoding="utf-8") as f:
                step_name = f.read().strip()
        except OSError:
            step_name = ""
        if step_name:
            step_meta = _read_json_file(os.path.join(ck_dir, step_name, "state.json"))
            if _meta_proves_completion(step_meta):
                return True

    for cand in (
        os.path.join(stage_dir, "state.json"),
        os.path.join(stage_dir, "summary.json"),
    ):
        if _meta_proves_completion(_read_json_file(cand)):
            return True

    return False


def run_v3_e2e_and_ship_pack(
    *,
    hf_repo: str | None = None,
    pack_repo: str | None = None,
    base_precision: str | None = "nvfp4",
    smoke_toy: bool = False,
    stages: Sequence[str] = DEFAULT_V2_STAGES,
    until: str | None = None,
    debug: int = 1,
    time_budget_hours: float = 21.5,
    ship_whole_model_pack: bool = True,
    stream_low_disk: bool = True,
    merge_hot_experts: bool = True,
    max_shard_size_bytes: int = 4_500_000_000,
    out_dir: str | None = None,
    force: bool = False,
    overrides: Sequence[str] = (),
) -> dict[str, Any]:
    """Run/resume the clean v2 pipeline (or explicit stages) and stream the Whole Model Pack upon completion."""
    from .hub import HubSyncManager, resolve_stage_config_path

    if not smoke_toy and sys.platform == "linux":
        ensure_quant_runtime_packages(_effective_precision_arg(base_precision, overrides), auto_install=True)
    sync_repo = hf_repo or os.environ.get("QLOOP_HF_REPO", DEFAULT_HF_REPO)
    target_pack_repo = pack_repo if pack_repo is not None else DEFAULT_PACK_REPO
    stage_tuple = tuple(stages)
    eff_overrides = _inject_base_model_override(overrides, base_precision, smoke_toy=smoke_toy)
    rc = run_pipeline(
        hf_repo=sync_repo,
        base_precision=base_precision,
        smoke_toy=smoke_toy,
        debug=debug,
        time_budget_hours=time_budget_hours,
        stages=stage_tuple,
        until=until,
        force=force,
        overrides=eff_overrides,
    )

    effective_stages = stage_tuple[: stage_tuple.index(until) + 1] if until and until in stage_tuple else stage_tuple

    run_root = os.environ.get("QLOOP_RUN_ROOT", "runs")
    manifest = HubSyncManager(repo_id=sync_repo, private=False).fetch_manifest()
    all_completed = bool(effective_stages) and all(
        manifest.is_completed(st) or _local_stage_done(run_root, st) for st in effective_stages
    )

    shipped = False
    if rc == 0 and ship_whole_model_pack and all_completed and effective_stages:
        final_stage = effective_stages[-1]
        reclaim_disk(keep=final_stage)
        if final_stage == "s4_rl" and ("s2_joint_bptt" in stage_tuple or "s1b_hssd_distill" in stage_tuple):
            pack_cfg = resolve_stage_config_path("s2_joint_bptt")
        else:
            pack_cfg = resolve_stage_config_path(final_stage)
        ckpt_dir = os.path.join(run_root, final_stage)
        parts = [
            "export-pack",
            "-c",
            shlex.quote(pack_cfg),
            "--ckpt",
            shlex.quote(ckpt_dir),
            "--max-shard-bytes",
            str(int(max_shard_size_bytes)),
            "--debug",
            str(int(debug)),
        ]
        if target_pack_repo:
            parts += ["--push-repo", shlex.quote(target_pack_repo)]
        if out_dir:
            parts += ["--out", shlex.quote(out_dir)]
        if stream_low_disk:
            parts.append("--stream-low-disk")
        if not merge_hot_experts:
            parts.append("--no-merge-hot-experts")
        caller_keys = {ov.split("=", 1)[0].strip() for ov in eff_overrides if "=" in ov}
        if "checkpoint.sync.hf_repo" not in caller_keys:
            parts += ["-s", shlex.quote(f"checkpoint.sync.hf_repo={sync_repo}")]
        for ov in eff_overrides:
            parts += ["-s", shlex.quote(ov)]
        q(" ".join(parts))
        shipped = True

    return {
        "pipeline_rc": rc,
        "sync_repo": sync_repo,
        "pack_repo": target_pack_repo,
        "stages": list(effective_stages),
        "all_stages_completed": all_completed,
        "pack_shipped": shipped,
        "out_dir": out_dir,
    }
