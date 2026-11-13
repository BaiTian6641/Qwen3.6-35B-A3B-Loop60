"""Environment doctor: one table telling you exactly what is wrong before a 20 GB download.

    python -m qloop doctor [--freeze requirements/colab.lock.txt] [--datasets] [--model NAME]

Each check returns (status, detail, fix). FAIL => non-zero exit code.
"""
from __future__ import annotations

import importlib
import logging
import os
import subprocess
import sys
from dataclasses import dataclass
from typing import Callable

log = logging.getLogger("qloop")

PASS, WARN, FAIL = "PASS", "WARN", "FAIL"


@dataclass
class Check:
    name: str
    status: str
    detail: str
    fix: str = ""


def _version(mod: str) -> str | None:
    try:
        m = importlib.import_module(mod)
    except Exception:
        return None
    return getattr(m, "__version__", "installed")


def check_python() -> Check:
    v = sys.version_info
    if v < (3, 9):
        return Check("python", FAIL, sys.version.split()[0], "use python >= 3.9")
    return Check("python", PASS, sys.version.split()[0])


def check_torch_gpu() -> list[Check]:
    out = []
    try:
        import torch
    except Exception as e:
        return [Check("torch", FAIL, repr(e), "pip install torch (Colab ships it preinstalled)")]
    out.append(Check("torch", PASS, f"{torch.__version__} (cuda {torch.version.cuda})"))
    if not torch.cuda.is_available():
        out.append(Check("gpu", WARN, "CUDA not available (fine for local dev on the toy model)",
                         "Colab: Runtime > Change runtime type > G4 (RTX PRO 6000 Blackwell)"))
        return out
    p = torch.cuda.get_device_properties(0)
    gib = p.total_memory / 2**30
    sm = f"sm_{p.major}{p.minor}"
    status = PASS if gib >= 90 else WARN
    out.append(Check("gpu", status, f"{p.name} {sm} {gib:.1f} GiB",
                     "" if status == PASS else "35B-A3B training recipes assume ~96 GB (RTX PRO 6000 Blackwell)"))
    archs = torch.cuda.get_arch_list()
    want = f"sm_{p.major}{p.minor}"
    if want not in archs and f"compute_{p.major}{p.minor}" not in archs:
        out.append(Check("torch arch", FAIL, f"{want} not in torch build {archs}",
                         "install a torch wheel built for this GPU (Blackwell sm_120 needs CUDA >= 12.8 wheels)"))
    else:
        out.append(Check("torch arch", PASS, f"{want} supported"))
    if p.major == 12:
        out.append(Check("attention kernels", PASS if _version("flash_attn") is None else WARN,
                         "Blackwell workstation (sm_120): FlashAttention-3 is Hopper-only; using SDPA/cuDNN or FA2",
                         "keep model.attn_implementation=sdpa unless a sm_120 FA build is verified"))
    try:
        torch.zeros(1, device="cuda").add_(1)
        x = torch.randn(256, 256, device="cuda", dtype=torch.bfloat16)
        (x @ x).sum().item()
        out.append(Check("cuda smoke", PASS, "bf16 matmul ok"))
    except Exception as e:
        out.append(Check("cuda smoke", FAIL, repr(e), "restart the runtime; check driver / torch build"))
    return out


def check_packages() -> list[Check]:
    out = []
    tv = _version("transformers")
    if tv is None:
        out.append(Check("transformers", WARN, "not installed", "pip install -U transformers (needed for real model)"))
    else:
        try:
            importlib.import_module("transformers.models.qwen3_5_moe")
            out.append(Check("transformers", PASS, f"{tv} (qwen3_5_moe available)"))
        except Exception:
            out.append(Check("transformers", FAIL, f"{tv} lacks qwen3_5_moe", "pip install -U transformers"))
    specs: list[tuple[str, str, str, str]] = [
        ("fla", "flash-linear-attention", WARN, "Gated DeltaNet fast kernels; without them HF uses a slow torch path"),
        ("causal_conv1d", "causal-conv1d", WARN, "short-conv kernel for DeltaNet (fallback is slower)"),
        ("apollo_torch", "apollo-torch", WARN, "APOLLO optimizer (built-in faithful fallback is used otherwise)"),
        ("cut_cross_entropy", "cut-cross-entropy", WARN, "CCE kernel (built-in chunked CE fallback otherwise)"),
        ("datasets", "datasets", WARN, "needed to stream training/eval data"),
        ("huggingface_hub", "huggingface_hub", WARN, "downloads + checkpoint sync"),
        ("triton", "triton", WARN, "required by fla / cce kernels"),
    ]
    for mod, pip, sev, why in specs:
        v = _version(mod)
        out.append(Check(mod, PASS if v else sev, v or f"missing - {why}", "" if v else f"pip install {pip}"))
    return out


def check_system() -> list[Check]:
    from ..hub import disk_report, resolve_hf_token

    out = []
    rep = disk_report()
    gb = float(rep["free_gib"])
    cached_w = float(rep["hf_hub_weights_gib"])
    ok_disk = (gb + cached_w) >= 115.0 and gb >= 40.0
    out.append(Check(
        "disk",
        PASS if ok_disk else WARN,
        f"{gb:.0f} GiB free of {rep['total_gib']:.0f} GiB (HF weights cache: {cached_w:.1f} GiB, "
        f"runs: {rep['runs_gib']:.1f} GiB)",
        "" if ok_disk else "Colab 230 GB SSD: run `python -m qloop sync disk --clean` to reclaim staging/old ckpts",
    ))
    try:
        import psutil

        ram = psutil.virtual_memory().total / 2**30
        out.append(Check("ram", PASS if ram >= 80 else WARN, f"{ram:.0f} GiB"))
    except Exception:
        pass
    tok = resolve_hf_token()
    hf_repo = os.environ.get("QLOOP_HF_REPO", "TassP0lar/Qwen3.6-35B-A3B-Loop64")
    out.append(Check(
        "hf token",
        PASS if tok else WARN,
        f"set (sync target: {hf_repo})" if tok else "missing",
        "" if tok else f"Colab: add write-enabled HF_TOKEN to Secrets to sync commits to {hf_repo}",
    ))
    docker = os.path.exists("/var/run/docker.sock")
    docker_msg = "available" if docker else "no docker daemon (expected on Colab)"
    out.append(Check("docker", PASS if docker else WARN, docker_msg,
                     "" if docker else "container benchmarks (TB2, SWE-bench Pro) run via Harbor --env daytona|modal"))
    return out


def check_hf_access(model: str) -> Check:
    try:
        from huggingface_hub import HfApi

        info = HfApi().model_info(model)
        return Check("model access", PASS, f"{model} ({len(info.siblings or [])} files)")
    except Exception as e:
        return Check("model access", FAIL, f"{model}: {e!r}", "check the name / HF_TOKEN / network")


def freeze(path: str) -> Check:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    res = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True, check=False)
    if res.returncode:
        return Check("freeze", FAIL, res.stderr[-500:])
    with open(path, "w", encoding="utf-8") as f:
        f.write(res.stdout)
    return Check("freeze", PASS, f"wrote {path} ({len(res.stdout.splitlines())} pins)")


def run_doctor(freeze_path: str | None = None, datasets: bool = False, model: str | None = None) -> int:
    checks: list[Check] = [check_python()]
    groups: list[Callable[[], list[Check]]] = [check_torch_gpu, check_packages, check_system]
    for g in groups:
        try:
            checks += g()
        except Exception as e:
            checks.append(Check(g.__name__, FAIL, f"check crashed: {e!r}"))
    if model:
        checks.append(check_hf_access(model))
    if datasets:
        try:
            from ..data.registry import REGISTRY, format_verify_report, verify_specs

            rep = verify_specs(list(REGISTRY.values()))
            print(format_verify_report(rep))
            for r in rep:
                status = PASS if r.get("ok") else (WARN if r.get("exists") else FAIL)
                checks.append(Check(f"dataset {r.get('name')}", status, str(r.get("detail") or r.get("access"))[:160],
                                    "" if r.get("ok") else "accept terms on the HF page / check HF_TOKEN"))
        except Exception as e:
            checks.append(Check("datasets", WARN, f"verification failed: {e!r}"))
    if freeze_path:
        checks.append(freeze(freeze_path))
    w = max(len(c.name) for c in checks)
    lines = [f"{'check':<{w}}  status  detail", "-" * (w + 60)]
    for c in checks:
        icon = {"PASS": "✅", "WARN": "⚠️ ", "FAIL": "❌"}[c.status]
        lines.append(f"{c.name:<{w}}  {icon}{c.status}  {c.detail}")
        if c.fix and c.status != PASS:
            lines.append(f"{'':<{w}}          ↳ fix: {c.fix}")
    print("\n".join(lines))
    fails = [c for c in checks if c.status == FAIL]
    print(f"\n{len(checks)} checks: {sum(c.status == PASS for c in checks)} pass, "
          f"{sum(c.status == WARN for c in checks)} warn, {len(fails)} fail")
    return 1 if fails else 0


__all__ = ["Check", "freeze", "run_doctor"]
