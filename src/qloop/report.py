"""Reproducible experiment telemetry archiver, SVG chart builder, and report generator.

Collects verified experimental evidence across:
  1. Full commit-by-commit Hugging Face Hub `manifest.json` history + probe/quality artifacts,
  2. Local `runs/colab_supervisor.log` (v1), `runs/colab_supervisor_v2.log` (`s3_v2`), &
     `runs/colab_supervisor_s2v2.log` (`s2_v2`),
  3. Optional local stage `metrics.jsonl` files & verified local JSON artifacts
     (`truth_check_summary.json`, `bench_v2_smoke_dplr/summary.json`,
     `Loop64-Fable-SIRSI-v1/audit_summary.json`, `eval_s3v2_step600_gate/summary.json`),
  4. Optional read-only Colab session snapshot (disabled by default to avoid disturbing live runs),

sanitizes all outputs to guarantee zero credential/token leakage, exports structured
JSON/CSV archives + 7 pure-SVG vector charts, and generates IEEE-formatted academic reports
in English (`docs/EXPERIMENT_REPORT.md`) and Simplified Chinese (`docs/EXPERIMENT_REPORT_ZH.md`).
"""
from __future__ import annotations

import argparse
import csv
import datetime as _dt
import json
import math
import os
import re
import shutil
import subprocess
from typing import Any

from qloop.report_charts import ALL_CHART_RENDERERS
from qloop.report_citations import validate_ieee_citations
from qloop.report_paper_en import (
    render_experiment_report_md,  # noqa: F401
    render_experiment_report_md_with_order,
)
from qloop.report_paper_zh import render_experiment_report_zh_md

DEFAULT_REPO = "TassP0lar/Qwen3.6-35B-A3B-Loop64"
DEFAULT_DATASET_REPO = "TassP0lar/Loop64-Fable-SIRSI-v1"
DEFAULT_SESSION = "loop64"
DEFAULT_COLAB_BIN = "/Users/tianrui6641/.local/bin/colab"
STAGES: tuple[str, ...] = (
    "s1_loop_a6b",
    "s1b_hssd_distill",
    "s3_long_1m_lsa",
    "s2_joint_bptt",
    "s3_v2_long_1m_lsa",
    "s2_v2_joint_bptt",
)

# Regexes for secret / private tunnel redaction
_SECRET_SUB_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"hf_[A-Za-z0-9]{16,}"), "[REDACTED_HF_TOKEN]"),
    (re.compile(r"(?i)(authorization\s*[:=]\s*bearer\s+)[^\s\"']+"), r"\1[REDACTED]"),
    (re.compile(r"(?i)(\bbearer\s+)(?!\[REDACTED\])[A-Za-z0-9._\-]{12,}"), r"\1[REDACTED]"),
    (re.compile(r"(?i)(HF_TOKEN\s*[:=]\s*['\"]?)[^\s'\",;]+"), r"\1[REDACTED]"),
    (re.compile(r"(?i)(TEACHER_API_KEY\s*[:=]\s*['\"]?)[^\s'\",;]+"), r"\1[REDACTED]"),
    (re.compile(r"https://colab\.research\.google\.com/[^\s\"']+"), "[REDACTED_COLAB_URL]"),
    (re.compile(r"dbu=[^&\s\"'#]+"), "dbu=[REDACTED]"),
)
_SECRET_DETECT_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"hf_[A-Za-z0-9]{16,}"),
    re.compile(r"(?i)\bbearer\s+(?!\[REDACTED\])[A-Za-z0-9._\-]{12,}"),
    re.compile(r"(?i)HF_TOKEN\s*=\s*['\"]?(?!\[REDACTED\]|\$)[A-Za-z0-9._\-]{12,}"),
)

# Verified step-0 quick-eval anchors captured in live launch logs (prior to step-25 Hub push).
# Only applied for the canonical production repo when not overridden by loaded records and
# never injected in synthetic offline unit-test runs.
VERIFIED_LAUNCH_ANCHORS: dict[str, dict[str, Any]] = {
    "s3_long_1m_lsa": {
        "step": 0,
        "d1": 2.0343,
        "d2": 1.9932,
        "d3": 1.9593,
        "depth_gain": 0.0750,
        "commit_sha": "513879f",
        "ts": "2026-11-19T10:01:58Z",
        "source": "verified_launch_log",
    },
    "s2_joint_bptt": {
        "step": 0,
        "d1": 1.4642,
        "d2": 1.4203,
        "d3": 1.3508,
        "depth_gain": 0.1135,
        "commit_sha": "2e37bf6",
        "ts": "2026-11-20T22:34:02Z",
        "source": "verified_launch_log",
    },
}

# Verified Blackwell sm_120 microbenchmark & systems milestones (from docs/PERF_PROFILE_S2.md,
# runs/bench_v2_smoke_dplr/summary.json, runs/colab_supervisor*.log, and git commits
# fbd7a35, a0877a3, 513879f, 2e37bf6, b2c5513, 5a82677, 5aaf55d, 28b88e4, 919f2b8, 2c78ffc).
VERIFIED_SYSTEMS_MILESTONES: list[dict[str, str]] = [
    {
        "component": "NVFP4 MoE Dequant (Triton E2M1)",
        "commit": "b2c5513 (41082e4)",
        "measurement": "0.12 ms (1,144–1,148 GB/s) vs 2.22–2.23 ms eager (E=64, M=512, K=2048)",
        "impact": (
            "18.99× speedup, saturates 71.9% of 1,597 GB/s GDDR7 ECC peak, "
            "18.5× less GDDR7 traffic, max_diff=0.000000 (bf16)"
        ),
    },
    {
        "component": "D-LSA Sparse FlashAttention (fwd+bwd)",
        "commit": "b2c5513 (ff27029)",
        "measurement": "174.56 ms (125.30 ms kernel) vs 1,627.16 ms ref (Tq=32K, Tk=131K, D=256)",
        "impact": "9.32× E2E / 12.99× kernel speedup; ~70 TB/step GDDR7 eliminated; cos >= 0.999973",
    },
    {
        "component": "D-LSA 2-Pass Triton Distill KL",
        "commit": "b2c5513 (41082e4)",
        "measurement": "46.42 ms vs 270.49 ms legacy (Tq=32K, Tk=131K, nq=2048, nb=256)",
        "impact": "5.83× speedup; ~90,000 host syncs/step eliminated; cos_wq/wk = 1.000000",
    },
    {
        "component": "BPTT Auto GDDR7-Resident / Async DMA",
        "commit": "b2c5513 (786fdeb)",
        "measurement": "0.64 ms GDDR7-resident (2.56 ms async pinned DMA) vs 34.46 ms blocking",
        "impact": "54.03× vs blocking (13.5× DMA); 10.7 GiB/step PCIe eliminated on 96 GB GDDR7",
    },
    {
        "component": "HSSD v3.2 Triton WY & Scan Kernels",
        "commit": "fbd7a35 (f48dc72)",
        "measurement": "Fused Triton chunk_hssd fwd+bwd + 64×64 block-doubling WY solver",
        "impact": "28–33× kernel acceleration; Phase-1/2 fits < 69.0 GiB on single 96 GB GDDR7",
    },
    {
        "component": "Live 1,048,576-Token FP8 Calibration Pass",
        "commit": "513879f / 28b88e4",
        "measurement": "1,015,808 prefix + 32,768 suffix = 1,048,576 tok sparse FP8 long_calib",
        "impact": "v1 HSSD: 78.60 GiB peak (0 leak); clean v2 Native-GDN: 69.52 GiB peak (-9.08 GiB, 0 skips)",
    },
    {
        "component": "HSSD Phase-2 O(T) Jacobian Stability Fix",
        "commit": "3a17c0d (a0877a3)",
        "measurement": "Scale cumulative-phase backward by chunk_size/T + live blend + atomic guard",
        "impact": "Eliminated 2.53e15–6.12e19 grad blowup & 194,510 skips → 0 / 1,100 skips",
    },
    {
        "component": "T=131K Triton int32 Pointer Overflow Fix",
        "commit": "2e37bf6",
        "measurement": "t*20480 > 2^31-1 wrapped at t=104,858; chunked all_up + tl.int64 across 32 kernels",
        "impact": "Eliminated s2 step-0 NaN (0.00e+00 parity error); 10.74 GiB → <=671 MiB transient",
    },
    {
        "component": "Zero-Label BPTT C++-Bridged VRAM Leak Fix",
        "commit": "5a82677",
        "measurement": "Broke LoopContext + saved_tensors_hooks C++ autograd cycles on -100 segments",
        "impact": "Alloc 52.0G (OOM) / 32.6G (baseline) → 25.3–25.4 GiB flat (-7.0 GiB vs baseline, 32h+)",
    },
    {
        "component": "transformers 5.24.0 Cache Kwarg & Proxy Fix",
        "commit": "5aaf55d / 28b88e4",
        "measurement": "Unified cache kwarg + isolated conv/recurrent snapshots in _ReplayKVCacheProxy",
        "impact": "10.6× recovery (89 → 942.6 tok/s avg, pk 1,109); 0.000e+00 1-tok checkpoint grad parity",
    },
    {
        "component": "GDN-2-1M+QED Triton Prep & fla 0.6.0 DPLR",
        "commit": "919f2b8 / 2c78ffc",
        "measurement": "T=32K: 53.55 ms fused Triton vs 106.93 ms PyTorch prep; T=1: 0.0787 vs 0.1895 ms",
        "impact": "2.00× T=32K & 2.41× T=1 prep speedup (12.3×–18.5× vs ref), 0.0 Step-0 diff, zero-sync is_zero()",
    },
    {
        "component": "GDN-2 Checkpointed 4×8K Macro-Segmenting & need_grad Gate",
        "commit": "cd0580b / 574898d",
        "measurement": "T=32K wrapper VRAM delta 23.051 → 7.032 GiB; Gate 3 4×32K d=3 peak 67.194 GiB",
        "impact": "3.28× wrapper VRAM cut (-16.02 GiB/layer), 38/38 CUDA pytest green, 0 NaNs, 27.8 GiB headroom",
    },
]


def sanitize_text(text: str) -> str:
    """Redact any secret tokens or private Colab tunnel URLs from text."""
    out = text
    for pat, repl in _SECRET_SUB_PATTERNS:
        out = pat.sub(repl, out)
    return out


def sanitize_obj(obj: Any) -> Any:
    """Recursively sanitize dicts, lists, and strings."""
    if isinstance(obj, str):
        return sanitize_text(obj)
    if isinstance(obj, dict):
        cleaned: dict[str, Any] = {}
        for k, v in obj.items():
            ks = str(k)
            if any(tok in ks.lower() for tok in ("hf_token", "api_key", "secret", "password", "authorization")):
                cleaned[ks] = "[REDACTED]"
            else:
                cleaned[ks] = sanitize_obj(v)
        return cleaned
    if isinstance(obj, list):
        return [sanitize_obj(x) for x in obj]
    if isinstance(obj, tuple):
        return [sanitize_obj(x) for x in obj]
    return obj


def contains_secret(text: str) -> bool:
    """Return True if text appears to contain an unredacted token or secret."""
    return any(p.search(text) is not None for p in _SECRET_DETECT_PATTERNS)


def _safe_float(val: Any) -> float | None:
    if isinstance(val, bool) or val is None:
        return None
    if isinstance(val, (int, float)):
        f = float(val)
        return f if math.isfinite(f) else None
    try:
        f = float(val)
        return f if math.isfinite(f) else None
    except (TypeError, ValueError):
        return None


def _safe_int(val: Any) -> int | None:
    if isinstance(val, bool) or val is None:
        return None
    if isinstance(val, int):
        return val
    f = _safe_float(val)
    return round(f) if f is not None else None


def _mean(vals: list[float]) -> float | None:
    return (sum(vals) / len(vals)) if vals else None


def _parse_iso(ts: str | None) -> _dt.datetime | None:
    if not ts:
        return None
    s = ts.strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    try:
        return _dt.datetime.fromisoformat(s)
    except ValueError:
        return None


def compute_kv_memory_1m() -> list[dict[str, Any]]:
    """Exact closed-form 1M-token (1,048,576) context-state VRAM breakdown (GiB, derived formula)."""
    t_1m = 1_048_576
    gib = float(1024 ** 3)

    def kv_gib(layers: int, kv_heads: int, head_dim: int, bpe: float) -> float:
        return round((layers * 2 * kv_heads * head_dim * bpe * t_1m) / gib, 2)

    # Clean v2 Native-GDN / GDN-2-1M + 88L Loop + FP8 1.0x SharedKV + D-LSA:
    # 10 physical full-attn layers x 2(K,V) x 2 KV heads x 256 dim x 1B (FP8 E4M3) x 1M = 10.00 GiB
    # 66 virtual linear passes @ R=3 (66/30 * 0.15 GiB stock GDN state budget = 0.33 GiB; 0 extra bytes for GDN-2)
    # 10 physical full-attn D-LSA Ward centroid tables = 0.32 GiB
    v2_kv = kv_gib(10, 2, 256, 1.0)  # 10.00 GiB exact
    v2_gdn = 0.33  # 66 virtual Native-GDN / GDN-2-1M recurrent + conv state budget (O(1))
    ours_idx = 0.32  # 320 MiB D-LSA Ward centroids
    v2_tot = round(v2_kv + v2_gdn + ours_idx, 2)  # 10.65 GiB

    # Clean v2 Inference-Time Block-64 Walsh-Hadamard INT4 Serving Projection:
    # Under the table's consistent 0.33 GiB 66L GDN budget:
    #   engine.py _HadamardINT4Slot (symmetric 34/64 = 0.53125 B/scalar -> 5.31 GiB KV + 0.33 + 0.32 = 5.96 GiB)
    #   offload.py Block64HadamardINT4Cache (asymmetric 36/64 = 0.5625 B/scalar -> 5.63 GiB KV + 0.33 + 0.32 = 6.28 GiB)
    # (Using exact raw fp32+conv GDN state 0.133 GiB instead of 0.33 GiB yields 5.77–6.08 GiB INT4 / 10.45 GiB FP8.)
    int4_kv = kv_gib(10, 2, 256, 34.0 / 64.0)  # 5.31 GiB (exact 5.3125 GiB, symmetric engine.py)
    int4_asym_kv_exact = (10 * 2 * 2 * 256 * (36.0 / 64.0) * t_1m) / gib  # 5.625 GiB exact (offload.py)
    int4_tot = round(int4_kv + v2_gdn + ours_idx, 2)  # 5.96 GiB (5.3125 + 0.33 + 0.32 = 5.9625)
    int4_asym_tot = round(int4_asym_kv_exact + v2_gdn + ours_idx + 1e-9, 2)  # 6.28 GiB (6.275 half-up)

    # LoopHSSD v3.2 (v1 ablation): 10L FP8 SharedKV + 66L HSSD bf16 state (~5.50 GiB) + 0.32 GiB D-LSA = 15.82 GiB
    ours_hssd = 5.50
    hssd_tot = round(v2_kv + ours_hssd + ours_idx, 2)  # 15.82 GiB

    # Stock Qwen3.6-35B-A3B (40L @ R=1): 10 full-attn layers BF16 + 30 stock GDN states (0.15 GiB)
    stock_kv = kv_gib(10, 2, 256, 2.0)  # 20.00 GiB
    stock_gdn = 0.15
    stock_tot = round(stock_kv + stock_gdn, 2)  # 20.15 GiB

    # Inkling 2 Mini (48L @ R=3, TML Nov 2026): 8 unshared logical global layers (1+2x3+1), 8 KV heads, D=128, F16 (2B)
    # + 40 SWA layers x 1024 window x 2 x 8 x 128 x 2B = 0.16 GiB
    inkling_kv = kv_gib(8, 8, 128, 2.0)  # 32.00 GiB exact
    inkling_swa = round((40 * 2 * 8 * 128 * 2 * 1024) / gib, 2)  # 0.16 GiB
    inkling_tot = round(inkling_kv + inkling_swa, 2)  # 32.16 GiB

    # Unshared 88L Hybrid (No SharedKV, Native GDN): 22 virtual full-attn layers BF16 + 66L GDN
    unshared_hyb_kv = kv_gib(22, 2, 256, 2.0)  # 44.00 GiB
    unshared_hyb_tot = round(unshared_hyb_kv + v2_gdn, 2)  # 44.33 GiB

    # Standard Unshared 88L Full-Attention Transformer (same 2 KV heads, D=256, BF16)
    unshared_88_kv = kv_gib(88, 2, 256, 2.0)  # 176.00 GiB

    return [
        {
            "name": "Clean v2 Hadamard INT4 Serving (88L)",
            "subtitle": "10L Blk-64 H_64 INT4 SharedKV + 66L GDN-2 + D-LSA (proj.)",
            "virtual_layers": 88,
            "global_kv_layers_stored": 10,
            "kv_dtype": "Blk-64 INT4 (0.53–0.56B)",
            "kv_gib": int4_kv,
            "linear_or_swa_gib": v2_gdn,
            "indexer_gib": ours_idx,
            "total_gib": int4_tot,
            "highlight": False,
            "is_projection": True,
            "callout": (
                f"~{int4_tot:.2f}–{int4_asym_tot:.2f} GiB serving proj. "
                f"({inkling_tot / int4_asym_tot:.1f}×–{inkling_tot / int4_tot:.1f}× vs Inkling)"
            ),
        },
        {
            "name": "Clean v2 Native-GDN / GDN-2-1M (88L @ R=3)",
            "subtitle": "10L FP8 1.0× SharedKV + 66L GDN/GDN-2 + D-LSA",
            "virtual_layers": 88,
            "global_kv_layers_stored": 10,
            "kv_dtype": "FP8 E4M3 (1B)",
            "kv_gib": v2_kv,
            "linear_or_swa_gib": v2_gdn,
            "indexer_gib": ours_idx,
            "total_gib": v2_tot,
            "highlight": True,
            "is_projection": False,
            "callout": (
                f"{v2_tot:.2f} GiB (10.00 KV + 0.33 GDN + 0.32 D-LSA · "
                f"{inkling_tot / v2_tot:.2f}× vs Inkling)"
            ),
        },
        {
            "name": "LoopHSSD v3.2 (88L @ R=3)",
            "subtitle": "v1 ablation: 10L FP8 SharedKV + 66L HSSD (+5.17 GiB)",
            "virtual_layers": 88,
            "global_kv_layers_stored": 10,
            "kv_dtype": "FP8 E4M3 (1B)",
            "kv_gib": v2_kv,
            "linear_or_swa_gib": ours_hssd,
            "indexer_gib": ours_idx,
            "total_gib": hssd_tot,
            "highlight": False,
            "is_projection": False,
            "callout": f"{hssd_tot:.2f} GiB (10.00 KV + ~5.50 HSSD + 0.32 D-LSA · +5.17 GiB vs v2)",
        },
        {
            "name": "Stock Qwen3.6-35B-A3B (40L @ R=1)",
            "subtitle": "10L BF16 KV + 30L GDN recurrent state",
            "virtual_layers": 40,
            "global_kv_layers_stored": 10,
            "kv_dtype": "BF16 (2B)",
            "kv_gib": stock_kv,
            "linear_or_swa_gib": stock_gdn,
            "indexer_gib": 0.0,
            "total_gib": stock_tot,
            "highlight": False,
            "is_projection": False,
            "callout": f"{stock_tot:.2f} GiB (20.00 BF16 KV + 0.15 GDN · {stock_tot / v2_tot:.2f}× v2)",
        },
        {
            "name": "Inkling 2 Mini (48L @ R=3)",
            "subtitle": "8 logical global F16 KV (8h×128d) + 40 SWA",
            "virtual_layers": 48,
            "global_kv_layers_stored": 8,
            "kv_dtype": "F16 (2B)",
            "kv_gib": inkling_kv,
            "linear_or_swa_gib": inkling_swa,
            "indexer_gib": 0.0,
            "total_gib": inkling_tot,
            "highlight": False,
            "is_projection": False,
            "callout": f"{inkling_tot:.2f} GiB (32.00 F16 KV · {inkling_tot / v2_tot:.2f}× Clean v2)",
        },
        {
            "name": "Unshared 88L Hybrid (No SharedKV)",
            "subtitle": "22 virtual global layers BF16 + 66L Native GDN",
            "virtual_layers": 88,
            "global_kv_layers_stored": 22,
            "kv_dtype": "BF16 (2B)",
            "kv_gib": unshared_hyb_kv,
            "linear_or_swa_gib": v2_gdn,
            "indexer_gib": 0.0,
            "total_gib": unshared_hyb_tot,
            "highlight": False,
            "is_projection": False,
            "callout": f"{unshared_hyb_tot:.2f} GiB (44.00 KV · {unshared_hyb_tot / v2_tot:.2f}× Clean v2)",
        },
        {
            "name": "Standard Unshared 88L Full-Attn",
            "subtitle": "88 global layers × 2 KV heads × 256d BF16",
            "virtual_layers": 88,
            "global_kv_layers_stored": 88,
            "kv_dtype": "BF16 (2B)",
            "kv_gib": unshared_88_kv,
            "linear_or_swa_gib": 0.0,
            "indexer_gib": 0.0,
            "total_gib": unshared_88_kv,
            "highlight": False,
            "is_projection": False,
            "callout": f"{unshared_88_kv:.2f} GiB ({unshared_88_kv / v2_tot:.1f}× Clean v2 · OOM > 96 GB)",
        },
    ]


# ---------------------------------------------------------------------------
# Collectors: HF Hub, Live Colab Session (opt-in), Supervisor Logs, Local JSONs
# ---------------------------------------------------------------------------
def collect_hub_history(
    repo: str,
    offline_dir: str | None = None,
    offline: bool = False,
    token: str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None, dict[str, Any] | None]:
    """Pull full commit-by-commit manifest.json history + probe/quality JSONs."""
    if offline_dir:
        commits_data: list[dict[str, Any]] = []
        for fn in ("hub_commits.json", "hub_manifest_history.json"):
            p = os.path.join(offline_dir, fn)
            if os.path.isfile(p):
                with open(p, encoding="utf-8") as f:
                    loaded = json.load(f)
                commits_data = loaded.get("commits", loaded) if isinstance(loaded, dict) else loaded
                break
        probe_report = None
        pr_path = os.path.join(offline_dir, "probe_report.json")
        if os.path.isfile(pr_path):
            with open(pr_path, encoding="utf-8") as f:
                probe_report = json.load(f)
        quality_check = None
        qc_path = os.path.join(offline_dir, "quality_check.json")
        if os.path.isfile(qc_path):
            with open(qc_path, encoding="utf-8") as f:
                quality_check = json.load(f)
        return sanitize_obj(commits_data), sanitize_obj(probe_report), sanitize_obj(quality_check)

    if offline:
        return [], None, None

    from huggingface_hub import HfApi, hf_hub_download

    eff_token = token if token is not None else (os.environ.get("HF_TOKEN") or None)
    api = HfApi(token=eff_token)
    repo_files = set(api.list_repo_files(repo_id=repo, repo_type="model"))
    commits = api.list_repo_commits(repo_id=repo, repo_type="model")

    records: list[dict[str, Any]] = []
    for c in reversed(commits):
        rec: dict[str, Any] = {
            "commit_id": c.commit_id,
            "short_sha": c.commit_id[:8],
            "created_at": c.created_at.isoformat() if c.created_at else None,
            "title": sanitize_text(c.title or ""),
            "message": sanitize_text(c.message or ""),
        }
        try:
            local_path = hf_hub_download(
                repo_id=repo,
                filename="manifest.json",
                revision=c.commit_id,
                token=eff_token,
            )
            with open(local_path, encoding="utf-8") as f:
                manifest = json.load(f)
            rec["manifest"] = sanitize_obj(
                {k: v for k, v in manifest.items() if k != "history"}
            )
        except Exception as exc:
            rec["manifest"] = None
            rec["manifest_error"] = type(exc).__name__
        records.append(rec)

    probe_report = None
    if "artifacts/probes/probe_report.json" in repo_files:
        p = hf_hub_download(
            repo_id=repo,
            filename="artifacts/probes/probe_report.json",
            token=eff_token,
        )
        with open(p, encoding="utf-8") as f:
            probe_report = sanitize_obj(json.load(f))

    quality_check = None
    if "artifacts/s1_loop_a6b/quality_check.json" in repo_files:
        p = hf_hub_download(
            repo_id=repo,
            filename="artifacts/s1_loop_a6b/quality_check.json",
            token=eff_token,
        )
        with open(p, encoding="utf-8") as f:
            quality_check = sanitize_obj(json.load(f))

    return records, probe_report, quality_check


def fetch_live_session_metrics(
    session: str | None,
    remote_path: str = "/content/runs/s2_joint_bptt/metrics.jsonl",
    colab_bin: str = DEFAULT_COLAB_BIN,
    timeout: int = 60,
    offline_dir: str | None = None,
    offline: bool = False,
) -> list[dict[str, Any]]:
    """Read-only snapshot of live metrics.jsonl (only when explicitly requested via session)."""
    if offline_dir:
        p = os.path.join(offline_dir, "live_s2_metrics.jsonl")
        if not os.path.isfile(p):
            return []
        rows: list[dict[str, Any]] = []
        with open(p, encoding="utf-8") as f:
            for raw_ln in f:
                s = raw_ln.strip()
                if s.startswith("{"):
                    try:
                        rows.append(sanitize_obj(json.loads(s)))
                    except json.JSONDecodeError:
                        continue
        return rows

    if offline or not session:
        return []

    exe = colab_bin if os.path.isfile(colab_bin) else shutil.which("colab")
    if not exe:
        return []

    cmd = [exe, "exec", "-s", session, "--timeout", str(timeout)]
    try:
        proc = subprocess.run(
            cmd,
            input=f"cat {remote_path}\n",
            text=True,
            capture_output=True,
            timeout=timeout + 15,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return []

    if proc.returncode != 0:
        return []

    rows = []
    for ln in proc.stdout.splitlines():
        s = ln.strip()
        if not s.startswith("{"):
            continue
        try:
            rows.append(sanitize_obj(json.loads(s)))
        except json.JSONDecodeError:
            continue
    return rows


_TARGET_TO_STAGE = {
    1000: "s1_loop_a6b",
    1500: "s1b_hssd_distill",
    600: "s3_long_1m_lsa",
    1200: "s2_joint_bptt",
}
_BOOT_RE = re.compile(
    r"^(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z)\s+\[BOOT\]\s+ready\s+session=(\S+)\s+launched=(\S+)\s+pid=(\S+)\s+head=([0-9a-f]+)"
)
_KEEPALIVE_RE = re.compile(
    r"^(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z)\s+\[KEEPALIVE\].*?pid_alive=(\w+).*?"
    r"util=(\d+)%\s+mem=(\d+)/(\d+)MiB\s+pwr=([0-9.]+)W\s+temp=(\d+)C(?:\s*\|\s*(.*))?$"
)
_STEP_RE = re.compile(
    r"step\s+(\d+)/(\d+)\s+lm\s+([0-9.]+)\s+d=(\d+)\s+([0-9.]+)\s+tok/s\s+mem\s+alloc\s+([0-9.]+)G\s*/\s*reserved\s+([0-9.]+)G\s*/\s*peak\s+([0-9.]+)G"
)
_FAIL_RE = re.compile(r"stage\s+(\S+)\s+failed\s+with\s+exit\s+code\s+(\d+)")
_RUN_DIR_RE = re.compile(r"run dir:\s+/content/runs/([a-zA-Z0-9_]+)")


def parse_supervisor_log(log_path: str | None, stage_override: str | None = None) -> dict[str, Any]:
    """Parse colab_supervisor*.log for sessions, throughput, VRAM, active GPU-hours, and s2 regimes."""
    empty: dict[str, Any] = {
        "present": False,
        "boots": [],
        "failures": [],
        "step_samples": [],
        "stage_head_summary": [],
        "s2_regimes": {},
        "active_gpu_hours_by_head": {},
        "active_gpu_hours_total": 0.0,
        "tf524_slowdown_window": None,
    }
    if not log_path or not os.path.isfile(log_path):
        return empty

    boots: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    step_samples: list[dict[str, Any]] = []
    seen_step_keys: set[tuple[str, str, int]] = set()
    cur_head = "unknown"
    cur_run_dir_stage: str | None = stage_override

    # Exact active GPU-hours computation (deduping identical timestamps, capping gap <= 30 min)
    active_hours_by_head: dict[str, float] = {}
    seen_ka_ts: set[str] = set()
    prev_ts: _dt.datetime | None = None
    prev_head: str | None = None
    tf524_pre_rows: list[tuple[str, int, int, float]] = []
    tf524_post_rows: list[tuple[str, int, int, float]] = []
    tf524_pre_s = 0.0
    tf524_post_s = 0.0

    with open(log_path, encoding="utf-8", errors="replace") as f:
        for raw_ln in f:
            ln = raw_ln.rstrip("\n")
            mb = _BOOT_RE.match(ln)
            if mb:
                ts, sess, launched, pid, head = mb.groups()
                cur_head = head
                cur_run_dir_stage = stage_override
                prev_ts = None
                boots.append(
                    {
                        "ts": ts,
                        "session": sess,
                        "launched": launched == "True",
                        "pid": pid,
                        "head": head,
                    }
                )
                continue

            mk = _KEEPALIVE_RE.match(ln)
            if not mk:
                continue
            ts, pid_alive, util, mem_used, mem_tot, pwr, temp, tail = mk.groups()
            tail = tail or ""

            mrd = _RUN_DIR_RE.search(tail)
            if mrd:
                cur_run_dir_stage = mrd.group(1)

            # Track exact active GPU hours (ignoring duplicate supervisor keepalive echoes)
            if ts not in seen_ka_ts:
                seen_ka_ts.add(ts)
                t_obj = _parse_iso(ts)
                if pid_alive == "True" and t_obj is not None:
                    if prev_ts is not None and prev_head == cur_head:
                        delta_s = (t_obj - prev_ts).total_seconds()
                        if 0 < delta_s <= 1800:
                            active_hours_by_head[cur_head] = active_hours_by_head.get(cur_head, 0.0) + delta_s / 3600.0
                            if cur_head == "8409b6e":
                                if ts < "2026-11-22T18:08:00Z":
                                    tf524_pre_s += delta_s
                                else:
                                    tf524_post_s += delta_s
                    prev_ts = t_obj
                    prev_head = cur_head
                    if cur_head == "8409b6e":
                        if ts < "2026-11-22T18:08:00Z":
                            tf524_pre_rows.append((ts, int(util), int(mem_used), float(pwr)))
                        else:
                            tf524_post_rows.append((ts, int(util), int(mem_used), float(pwr)))
                else:
                    prev_ts = None

            mf = _FAIL_RE.search(tail)
            if mf:
                rec = {"ts": ts, "head": cur_head, "stage": mf.group(1), "exit_code": int(mf.group(2))}
                if not failures or (failures[-1]["stage"], failures[-1]["head"], failures[-1]["ts"][:13]) != (
                    rec["stage"],
                    rec["head"],
                    rec["ts"][:13],
                ):
                    failures.append(rec)

            ms = _STEP_RE.search(tail)
            if ms:
                step, target, lm, depth, tps, alloc_g, res_g, peak_g = ms.groups()
                tgt_i = int(target)
                if cur_run_dir_stage in ("s3_long_1m_lsa", "s3_v2_long_1m_lsa", "s2_v2_joint_bptt") and tgt_i == 600:
                    stage = cur_run_dir_stage
                else:
                    stage = _TARGET_TO_STAGE.get(tgt_i, cur_run_dir_stage or f"target_{target}")
                st_int = int(step)
                dedup_k = (stage, cur_head, st_int)
                if dedup_k in seen_step_keys:
                    continue
                seen_step_keys.add(dedup_k)
                step_samples.append(
                    {
                        "ts": ts,
                        "stage": stage,
                        "head": cur_head,
                        "step": st_int,
                        "target_steps": tgt_i,
                        "loss_lm": float(lm),
                        "depth": int(depth),
                        "tok_per_s": float(tps),
                        "mem_alloc_gib": float(alloc_g),
                        "mem_reserved_gib": float(res_g),
                        "mem_peak_gib": float(peak_g),
                        "gpu_util_pct": int(util),
                        "gpu_mem_mib": int(mem_used),
                        "gpu_mem_total_mib": int(mem_tot),
                        "gpu_power_w": float(pwr),
                        "gpu_temp_c": int(temp),
                    }
                )

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in step_samples:
        grouped.setdefault((row["stage"], row["head"]), []).append(row)

    stage_head_summary: list[dict[str, Any]] = []
    for (stage, head), rows in grouped.items():
        tps_l = [r["tok_per_s"] for r in rows]
        alloc_l = [r["mem_alloc_gib"] for r in rows]
        peak_l = [r["mem_peak_gib"] for r in rows]
        pwr_l = [r["gpu_power_w"] for r in rows]
        util_l = [float(r["gpu_util_pct"]) for r in rows]
        steps_l = [r["step"] for r in rows]
        t0 = _parse_iso(rows[0]["ts"])
        t1 = _parse_iso(rows[-1]["ts"])
        span_h = round((t1 - t0).total_seconds() / 3600.0, 2) if (t0 and t1 and t1 >= t0) else 0.0
        stage_head_summary.append(
            {
                "stage": stage,
                "head": head,
                "n_samples": len(rows),
                "min_step": min(steps_l),
                "max_step": max(steps_l),
                "tok_per_s_min": round(min(tps_l), 1),
                "tok_per_s_max": round(max(tps_l), 1),
                "tok_per_s_mean": round(sum(tps_l) / len(tps_l), 1),
                "alloc_min_gib": round(min(alloc_l), 2),
                "alloc_max_gib": round(max(alloc_l), 2),
                "peak_min_gib": round(min(peak_l), 2),
                "peak_max_gib": round(max(peak_l), 2),
                "power_mean_w": round(sum(pwr_l) / len(pwr_l), 1),
                "util_mean_pct": round(sum(util_l) / len(util_l), 1),
                "observed_span_hours": span_h,
                "active_gpu_hours": round(active_hours_by_head.get(head, 0.0), 2),
            }
        )

    s2_regimes: dict[str, dict[str, Any]] = {}
    for item in stage_head_summary:
        if item["stage"] != "s2_joint_bptt":
            continue
        hd = item["head"]
        if hd == "2e37bf6" or item["max_step"] <= 43:
            s2_regimes["pre_opt_2e37bf6"] = item
        elif hd == "b2c5513":
            s2_regimes["post_opt_leaky_b2c5513"] = item
        elif hd == "5a82677":
            s2_regimes["post_leak_fix_5a82677"] = item
        elif hd == "8409b6e":
            s2_regimes["post_tf524_fix_2f5a6ba"] = item
        else:
            s2_regimes[f"head_{hd}"] = item

    tf524_window = None
    if len(tf524_pre_rows) >= 2:
        tot_8409_h = round(active_hours_by_head.get("8409b6e", 0.0), 2)
        pre_h = round(tf524_pre_s / 3600.0, 2)
        post_h = round(max(0.0, tot_8409_h - pre_h), 2) if tf524_post_s > 0 else 0.0
        tf524_window = {
            "slowdown_keepalives": len(tf524_pre_rows),
            "slowdown_hours": pre_h,
            "slowdown_mean_power_w": round(sum(x[3] for x in tf524_pre_rows) / len(tf524_pre_rows), 1),
            "slowdown_max_driver_mem_mib": max(x[2] for x in tf524_pre_rows),
            "post_fix_keepalives": len(tf524_post_rows),
            "post_fix_hours": post_h,
            "post_fix_mean_power_w": (
                round(sum(x[3] for x in tf524_post_rows) / len(tf524_post_rows), 1) if tf524_post_rows else None
            ),
            "post_fix_max_driver_mem_mib": max((x[2] for x in tf524_post_rows), default=None),
        }

    rounded_hours = {k: round(v, 2) for k, v in active_hours_by_head.items()}
    return {
        "present": True,
        "boots": boots,
        "failures": failures,
        "step_samples": step_samples,
        "stage_head_summary": stage_head_summary,
        "s2_regimes": s2_regimes,
        "active_gpu_hours_by_head": rounded_hours,
        "active_gpu_hours_total": round(sum(active_hours_by_head.values()), 2),
        "tf524_slowdown_window": tf524_window,
    }


def _merge_supervisor_logs(
    sup_v1: dict[str, Any],
    sup_v2: dict[str, Any],
    sup_s2v2: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Combine v1, v2 (s3_v2), and s2_v2 supervisor log parses cleanly without dropping any."""
    sups = [s for s in (sup_v1, sup_v2, sup_s2v2) if s and s.get("present")]
    if not sups:
        return sup_v1
    merged_hours: dict[str, float] = {}
    boots: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    step_samples: list[dict[str, Any]] = []
    stage_head_summary: list[dict[str, Any]] = []
    for s in sups:
        for k, v in (s.get("active_gpu_hours_by_head") or {}).items():
            merged_hours[k] = round(merged_hours.get(k, 0.0) + float(v), 2)
        boots.extend(s.get("boots") or [])
        failures.extend(s.get("failures") or [])
        step_samples.extend(s.get("step_samples") or [])
        stage_head_summary.extend(s.get("stage_head_summary") or [])

    h_v1 = round(float(sup_v1.get("active_gpu_hours_total") or 0.0), 2) if sup_v1.get("present") else 0.0
    h_v2 = round(float(sup_v2.get("active_gpu_hours_total") or 0.0), 2) if sup_v2.get("present") else 0.0
    h_s2v2 = (
        round(float((sup_s2v2 or {}).get("active_gpu_hours_total") or 0.0), 2)
        if (sup_s2v2 or {}).get("present")
        else 0.0
    )
    return {
        "present": True,
        "boots": boots,
        "failures": failures,
        "step_samples": step_samples,
        "stage_head_summary": stage_head_summary,
        "s2_regimes": dict(sup_v1.get("s2_regimes") or {}),
        "active_gpu_hours_by_head": merged_hours,
        "active_gpu_hours_total": round(h_v1 + h_v2 + h_s2v2, 2),
        "active_gpu_hours_v1": h_v1,
        "active_gpu_hours_v2": h_v2,
        "active_gpu_hours_s2v2": h_s2v2,
        "tf524_slowdown_window": sup_v1.get("tf524_slowdown_window"),
    }


def load_local_stage_metrics(runs_dir: str | None) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any] | None]:
    """Load optional local runs/<stage>_metrics.jsonl & s1_quality_check.json."""
    out: dict[str, list[dict[str, Any]]] = {}
    qc: dict[str, Any] | None = None
    if not runs_dir or not os.path.isdir(runs_dir):
        return out, qc

    for st in STAGES:
        p = os.path.join(runs_dir, f"{st}_metrics.jsonl")
        if os.path.isfile(p):
            rows: list[dict[str, Any]] = []
            with open(p, encoding="utf-8") as f:
                for ln in f:
                    s = ln.strip()
                    if s.startswith("{"):
                        try:
                            rows.append(sanitize_obj(json.loads(s)))
                        except json.JSONDecodeError:
                            continue
            out[st] = rows

    qc_p = os.path.join(runs_dir, "s1_quality_check.json")
    if os.path.isfile(qc_p):
        with open(qc_p, encoding="utf-8") as f:
            qc = sanitize_obj(json.load(f))
    return out, qc


def load_truth_check_summary(path: str | None) -> dict[str, Any] | None:
    """Load and sanitize runs/eval_truth_check_s2_step500/truth_check_summary.json."""
    if not path or not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    return sanitize_obj(raw)


def load_dplr_bench_summary(path: str | None) -> dict[str, Any] | None:
    """Load and sanitize a compact structured summary from runs/bench_v2_smoke_dplr/summary.json."""
    if not path or not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    part_a = raw.get("part_a_v2_smoke") or {}
    part_b = raw.get("part_b_dplr_gdn2_bench") or {}
    parity = part_b.get("parity") or {}
    benches = part_b.get("benchmarks") or {}

    def _compact_metrics(d: dict[str, Any] | None) -> dict[str, Any] | None:
        if not isinstance(d, dict):
            return None
        out_m: dict[str, Any] = {}
        for k, v in d.items():
            if k not in ("max_abs_diff", "mean_abs_diff", "rel_l2", "cosine", "finite"):
                continue
            if isinstance(v, float):
                out_m[k] = float(f"{v:.6g}") if (v != 0.0 and abs(v) < 1e-4) else round(v, 8)
            else:
                out_m[k] = v
        return out_m

    def _min_grad_cos(grads: dict[str, Any] | None) -> float | None:
        if not isinstance(grads, dict):
            return None
        vals = [
            float(v["cosine"])
            for v in grads.values()
            if isinstance(v, dict) and isinstance(v.get("cosine"), (int, float))
        ]
        return round(min(vals), 6) if vals else None

    fp32_p = parity.get("fp32") or {}
    bf16_p = parity.get("bf16") or {}
    fp32_dplr = ((fp32_p.get("kernels") or {}).get("chunk_dplr_delta_rule") or {})
    bf16_dplr = ((bf16_p.get("kernels") or {}).get("chunk_dplr_delta_rule") or {})
    fla_gdn2_p = parity.get("fla_ops_gdn2") or {}

    compact = {
        "timestamp_utc": raw.get("timestamp_utc"),
        "environment": raw.get("environment"),
        "part_a_smoke": {
            "part_a_passed": part_a.get("part_a_passed"),
            "data_stream": {
                "hf_id": (part_a.get("data_stream") or {}).get("hf_id"),
                "examples_kept": (part_a.get("data_stream") or {}).get("examples_kept"),
                "reasoning_turns": (part_a.get("data_stream") or {}).get("reasoning_turns"),
                "tool_call_turns": (part_a.get("data_stream") or {}).get("tool_call_turns"),
            },
            "smoke_s3v2": {
                "ok": (part_a.get("smoke_s3v2") or {}).get("ok"),
                "elapsed_sec": (part_a.get("smoke_s3v2") or {}).get("elapsed_sec"),
                "all_metrics_finite": (part_a.get("smoke_s3v2") or {}).get("all_metrics_finite"),
                "lm_losses": (part_a.get("smoke_s3v2") or {}).get("lm_losses"),
                "peak_gib": ((part_a.get("smoke_s3v2") or {}).get("records") or [{}])[-1].get("mem/peak_gib"),
                "step2_tok_per_s": ((part_a.get("smoke_s3v2") or {}).get("records") or [{}])[-1].get("tok_per_s"),
            },
            "smoke_s2v2": {
                "ok": (part_a.get("smoke_s2v2") or {}).get("ok"),
                "elapsed_sec": (part_a.get("smoke_s2v2") or {}).get("elapsed_sec"),
                "all_metrics_finite": (part_a.get("smoke_s2v2") or {}).get("all_metrics_finite"),
                "summary_metrics": (part_a.get("smoke_s2v2") or {}).get("summary_metrics"),
            },
        },
        "fla_version": (part_b.get("fla_discovery") or {}).get("fla_version"),
        "identified_formula": parity.get("identified_formula"),
        "mapping_doc": fp32_p.get("mapping_doc"),
        "parity_summary": {
            "fp32_chunk_dplr_e2e_o": _compact_metrics((fp32_dplr.get("e2e_gdn2_qed") or {}).get("forward_o")),
            "fp32_chunk_dplr_e2e_s": _compact_metrics((fp32_dplr.get("e2e_gdn2_qed") or {}).get("forward_s")),
            "fp32_chunk_dplr_e2e_min_grad_cos": _min_grad_cos((fp32_dplr.get("e2e_gdn2_qed") or {}).get("grads_e2e")),
            "fp32_T1_decode_o": _compact_metrics((fp32_p.get("T1_decode_parity") or {}).get("o")),
            "fp32_T1_decode_s": _compact_metrics((fp32_p.get("T1_decode_parity") or {}).get("final_state")),
            "bf16_chunk_dplr_e2e_o": _compact_metrics((bf16_dplr.get("e2e_gdn2_qed") or {}).get("forward_o")),
            "bf16_chunk_dplr_e2e_s": _compact_metrics((bf16_dplr.get("e2e_gdn2_qed") or {}).get("forward_s")),
            "bf16_chunk_dplr_e2e_min_grad_cos": _min_grad_cos((bf16_dplr.get("e2e_gdn2_qed") or {}).get("grads_e2e")),
            "bf16_T1_decode_o": _compact_metrics((bf16_p.get("T1_decode_parity") or {}).get("o")),
            "bf16_T1_decode_s": _compact_metrics((bf16_p.get("T1_decode_parity") or {}).get("final_state")),
            "bf16_fla_chunk_gdn2_o": _compact_metrics((fla_gdn2_p.get("chunk_gdn2_bf16") or {}).get("forward_o")),
            "bf16_fla_chunk_gdn2_min_grad_cos": _min_grad_cos((fla_gdn2_p.get("chunk_gdn2_bf16") or {}).get("grads")),
            "bf16_fla_recurrent_gdn2_o": _compact_metrics(
                (fla_gdn2_p.get("fused_recurrent_gdn2_bf16") or {}).get("forward_o")
            ),
            "stability_T4096_H32_bf16": parity.get("stability_T4096_H32_bf16"),
        },
        "T32768_fwd_bwd": {
            k: v for k, v in (benches.get("T32768_fwd_bwd") or {}).items()
            if isinstance(v, dict) and "total" in v
        },
        "T1_decode": benches.get("T1_decode") or {},
    }
    return sanitize_obj(compact)


def load_dataset_audit_summary(path: str | None) -> dict[str, Any] | None:
    """Load and sanitize structured summary from Loop64-Fable-SIRSI-v1/audit_summary.json."""
    if not path or not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    compact = {
        "repo_id": raw.get("repo_id"),
        "release_version": raw.get("release_version"),
        "elapsed_sec": raw.get("elapsed_sec"),
        "output_counts": raw.get("output_counts"),
        "licenses": raw.get("licenses"),
        "per_source": raw.get("per_source"),
        "dedup": raw.get("dedup"),
        "decontam": {
            k: v for k, v in (raw.get("decontam") or {}).items()
            if k != "cache_dir"
        },
        "track_b_local_agent": raw.get("track_b_local_agent"),
        "v11_fable_upgrade": raw.get("v11_fable_upgrade"),
        "v12_fable_notebook": raw.get("v12_fable_notebook"),
    }
    return sanitize_obj(compact)


def load_s3v2_gate_summary(path: str | None) -> dict[str, Any] | None:
    """Load and sanitize compact Gate 1/2/3 summary from runs/eval_s3v2_step600_gate/summary.json."""
    if not path or not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)

    g1 = raw.get("gate1_cuda_gdn2") or {}
    g2 = raw.get("gate2_s3v2_step600_truth_check") or {}
    g3 = raw.get("gate3_s2v2_4x32k_d3_smoke") or {}

    compact_runs: list[dict[str, Any]] = []
    for r in g2.get("runs") or []:
        if not isinstance(r, dict):
            continue
        compact_runs.append(
            {
                k: v
                for k, v in r.items()
                if k not in ("text", "first_240")
            }
        )

    compact = {
        "timestamp_utc": raw.get("timestamp_utc"),
        "all_gates_passed": raw.get("all_gates_passed"),
        "verdict": raw.get("verdict"),
        "environment": raw.get("environment"),
        "gate1_cuda_gdn2": {
            "gate1_passed": g1.get("gate1_passed"),
            "elapsed_sec": g1.get("elapsed_sec"),
            "pytest": g1.get("pytest"),
            "benchmarks": g1.get("benchmarks"),
        },
        "gate2_s3v2_step600_truth_check": {
            "gate2_passed": g2.get("gate2_passed"),
            "elapsed_sec": g2.get("elapsed_sec"),
            "checkpoint": g2.get("checkpoint"),
            "checks": g2.get("checks"),
            "runs": compact_runs,
        },
        "gate3_s2v2_4x32k_d3_smoke": {
            k: v
            for k, v in g3.items()
            if k not in ("log_tail", "step_log_lines")
        },
    }
    return sanitize_obj(compact)


# ---------------------------------------------------------------------------
# Normalization, Rollback Classification & Deduplication
# ---------------------------------------------------------------------------
_TITLE_STAGE_RE = re.compile(r"\[qloop:([a-zA-Z0-9_]+)\]")
_ROLLBACK_STEP_RE = re.compile(r"step=(\d+)")


def _extract_metrics_row(
    stage: str,
    step: int,
    status: str,
    met: dict[str, Any],
    commit_sha: str | None = None,
    ts: str | None = None,
    source: str = "hub_manifest",
) -> dict[str, Any]:
    kl_vals = [float(v) for k, v in met.items() if k.startswith("lsa/distill_kl/L") and _safe_float(v) is not None]
    rec_vals = [float(v) for k, v in met.items() if k.startswith("lsa/recall/L") and _safe_float(v) is not None]
    mrec_vals = [
        float(v) for k, v in met.items() if k.startswith("lsa/min_head_recall/L") and _safe_float(v) is not None
    ]
    row: dict[str, Any] = {
        "stage": stage,
        "step": step,
        "status": status,
        "commit_sha": commit_sha or "",
        "ts": ts or "",
        "source": source,
        "depth": _safe_int(met.get("depth")),
        "loss_lm": _safe_float(met.get("loss/lm")),
        "tokens_seen": _safe_int(met.get("tokens_seen")),
        "tok_per_s": _safe_float(met.get("tok_per_s")),
        "lr_factor": _safe_float(met.get("lr_factor")),
        "grad_norm": _safe_float(met.get("optim/optim/global_grad_norm")),
        "grad_norm_ema": _safe_float(met.get("optim/optim/grad_norm_ema")),
        "skipped_params": _safe_float(met.get("optim/optim/skipped_params")),
        "skipped_total": _safe_float(met.get("optim/optim/skipped_total")),
        "mem_alloc_gib": _safe_float(met.get("mem/alloc_gib")),
        "mem_peak_gib": _safe_float(met.get("mem/peak_gib")),
        "eval_loss_d1": _safe_float(met.get("eval/loss_d1")),
        "eval_loss_d2": _safe_float(met.get("eval/loss_d2")),
        "eval_loss_d3": _safe_float(met.get("eval/loss_d3")),
        "eval_depth_gain": _safe_float(met.get("eval/depth_gain")),
        "reroute_rate": _safe_float(met.get("loop/reroute_rate")),
        "unique_experts_core": _safe_float(met.get("loop/unique_experts_core")),
        "effective_expert_mult": _safe_float(met.get("loop/effective_expert_mult")),
        "gate_abs_mean_p2": _safe_float(met.get("loop/gate_abs_mean_p2")),
        "gate_abs_mean_p3": _safe_float(met.get("loop/gate_abs_mean_p3")),
        "bridge_w_h_prev_p3": _safe_float(met.get("loop/bridge_w_h_prev_p3")),
        "bridge_w_anchor_p3": _safe_float(met.get("loop/bridge_w_anchor_p3")),
        "bridge_subspace_up_abs_mean_p3": _safe_float(met.get("loop/bridge_subspace_up_abs_mean_p3")),
        "bridge_step_emb_norm_p3": _safe_float(met.get("loop/bridge_step_emb_norm_p3")),
        "distill_phase": _safe_float(met.get("loss/distill_phase")),
        "hssd_blend": _safe_float(met.get("loss/hssd_blend")),
        "distill_kl": _safe_float(met.get("loss/distill_kl")),
        "layerwise_mse": _safe_float(met.get("loss/layerwise_mse")),
        "hssd_budget_aux": _safe_float(met.get("hssd/budget_aux_loss")),
        "hssd_vault_duty": _safe_float(met.get("hssd/vault_duty")),
        "hssd_dead_page_count": _safe_float(met.get("hssd/dead_page_count")),
        "lsa_kl_total": _safe_float(met.get("loss/lsa_kl")),
        "lsa_kl_mean": _mean(kl_vals),
        "lsa_recall": _safe_float(met.get("lsa/last_recall")) or _mean(rec_vals),
        "lsa_min_head_recall": _mean(mrec_vals),
        "calib_prefix_len": _safe_int(met.get("loss/calib_prefix_len")),
        "bptt_replayed_tokens": _safe_int(met.get("loss/bptt_replayed_tokens")),
        "bptt_offload_to_cpu": _safe_float(met.get("loss/bptt_offload_to_cpu")),
        "gdn2_adapter_norm": _safe_float(met.get("gdn2/adapter_norm")),
        "gdn2_qed_lambda_abs_mean": _safe_float(met.get("gdn2/qed_lambda_abs_mean")),
        "gdn2_neg_eigval_scale_mean": _safe_float(met.get("gdn2/neg_eigval_scale_mean")),
        "gdn2_decay_horizon_mult_max": _safe_float(met.get("gdn2/decay_horizon_mult_max")),
        "gdn2_erase_write_l1": _safe_float(met.get("gdn2/erase_write_l1")),
        "gdn2_decay_spread": _safe_float(met.get("gdn2/decay_spread")),
    }
    return row


def normalize_experiment_data(
    raw_commits: list[dict[str, Any]],
    live_s2: list[dict[str, Any]],
    supervisor_data: dict[str, Any],
    local_metrics: dict[str, list[dict[str, Any]]],
    probe_report: dict[str, Any] | None,
    quality_check: dict[str, Any] | None,
    repo: str = DEFAULT_REPO,
    use_launch_anchors: bool = True,
    truth_check: dict[str, Any] | None = None,
    dplr_bench: dict[str, Any] | None = None,
    dataset_audit: dict[str, Any] | None = None,
    s3v2_gate: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Classify rollbacks, deduplicate by (stage, step), merge live/supervisor/artifact telemetry."""
    s1_reset_idx: int | None = None
    s1b_rollback_idx: int | None = None
    s1b_rollback_step = 400
    for idx, c in enumerate(raw_commits):
        title = str(c.get("title") or "")
        if "[qloop:reset]" in title and "s1_loop_a6b" in title:
            s1_reset_idx = idx
        if "[qloop:s1b_hssd_distill]" in title and "rollback" in title.lower():
            s1b_rollback_idx = idx
            ms = _ROLLBACK_STEP_RE.search(title)
            if ms:
                s1b_rollback_step = int(ms.group(1))

    hub_history_annotated: list[dict[str, Any]] = []
    rolled_back_events: list[dict[str, Any]] = []
    dedup_rows: dict[str, dict[int, dict[str, Any]]] = {st: {} for st in STAGES}
    stage_latest_meta: dict[str, dict[str, Any]] = {}
    raw_eval_events: dict[str, list[dict[str, Any]]] = {st: [] for st in STAGES}

    for idx, c in enumerate(raw_commits):
        title = str(c.get("title") or "")
        sha = str(c.get("short_sha") or str(c.get("commit_id") or "")[:8])
        ts = str(c.get("created_at") or "")
        m = c.get("manifest")
        stage_match = _TITLE_STAGE_RE.search(title)
        title_stage = stage_match.group(1) if stage_match else ""

        rolled_back = False
        rollback_reason = ""
        step_val: int | None = None
        status_val = ""
        stage_met: dict[str, Any] = {}

        if isinstance(m, dict):
            stages_d = m.get("stages") or {}
            if title_stage in stages_d and isinstance(stages_d[title_stage], dict):
                st_info = stages_d[title_stage]
                step_val = _safe_int(st_info.get("step"))
                status_val = str(st_info.get("status") or "")
                stage_met = st_info.get("metrics") or {}
                if title_stage in STAGES and "rollback" not in title.lower() and "quality check" not in title.lower():
                    stage_latest_meta[title_stage] = {
                        "status": status_val,
                        "step": step_val,
                        "target_steps": _safe_int(st_info.get("target_steps")),
                        "n_params": _safe_int(st_info.get("n_params")),
                        "updated_at": st_info.get("updated_at"),
                        "commit_sha": sha,
                    }

        if s1_reset_idx is not None and idx < s1_reset_idx and title_stage == "s1_loop_a6b":
            rolled_back = True
            rollback_reason = "reset_poisoned_s1"
        elif (
            s1b_rollback_idx is not None
            and idx < s1b_rollback_idx
            and title_stage == "s1b_hssd_distill"
            and step_val is not None
            and step_val > s1b_rollback_step
        ):
            rolled_back = True
            rollback_reason = "s1b_phase2_jacobian_explosion"

        entry = {
            "commit_id": c.get("commit_id", ""),
            "short_sha": sha,
            "created_at": ts,
            "title": title,
            "stage": title_stage,
            "step": step_val,
            "status": status_val,
            "rolled_back": rolled_back,
            "rollback_reason": rollback_reason,
            "loss_lm": _safe_float(stage_met.get("loss/lm")),
            "grad_norm": _safe_float(stage_met.get("optim/optim/global_grad_norm")),
            "skipped_total": _safe_float(stage_met.get("optim/optim/skipped_total")),
            "tok_per_s": _safe_float(stage_met.get("tok_per_s")),
            "mem_peak_gib": _safe_float(stage_met.get("mem/peak_gib")),
            "eval_loss_d1": _safe_float(stage_met.get("eval/loss_d1")),
            "eval_loss_d2": _safe_float(stage_met.get("eval/loss_d2")),
            "eval_loss_d3": _safe_float(stage_met.get("eval/loss_d3")),
            "eval_depth_gain": _safe_float(stage_met.get("eval/depth_gain")),
        }
        hub_history_annotated.append(entry)

        if title_stage not in STAGES or step_val is None or not stage_met:
            continue
        if "rollback" in title.lower() or "quality check" in title.lower():
            continue

        row = _extract_metrics_row(
            stage=title_stage,
            step=step_val,
            status=status_val or "ckpt",
            met=stage_met,
            commit_sha=sha,
            ts=ts,
            source="hub_manifest",
        )
        if rolled_back:
            row["rolled_back"] = True
            row["rollback_reason"] = rollback_reason
            rolled_back_events.append(row)
            continue

        prev = dedup_rows[title_stage].get(step_val)
        if prev is None or status_val == "completed" or prev.get("status") != "completed":
            dedup_rows[title_stage][step_val] = row

        if row["eval_loss_d1"] is not None and row["eval_loss_d3"] is not None:
            raw_eval_events[title_stage].append(
                {
                    "step": step_val,
                    "status": status_val,
                    "d1": row["eval_loss_d1"],
                    "d2": row["eval_loss_d2"],
                    "d3": row["eval_loss_d3"],
                    "depth_gain": row["eval_depth_gain"],
                    "commit_sha": sha,
                    "ts": ts,
                    "source": "hub_manifest",
                }
            )

    # Pass 2: merge optional live s2_joint_bptt metrics.jsonl
    for rec in live_s2:
        step_v = _safe_int(rec.get("step"))
        if step_v is None:
            continue
        is_eval_only = ("eval/loss_d1" in rec) and ("loss/lm" not in rec)
        ts_v = ""
        if _safe_float(rec.get("time")) is not None:
            ts_v = _dt.datetime.fromtimestamp(float(rec["time"]), tz=_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        if is_eval_only:
            raw_eval_events["s2_joint_bptt"].append(
                {
                    "step": step_v,
                    "status": "eval",
                    "d1": _safe_float(rec.get("eval/loss_d1")),
                    "d2": _safe_float(rec.get("eval/loss_d2")),
                    "d3": _safe_float(rec.get("eval/loss_d3")),
                    "depth_gain": _safe_float(rec.get("eval/depth_gain")),
                    "commit_sha": "",
                    "ts": ts_v,
                    "source": "live_metrics_jsonl",
                }
            )
            continue
        row = _extract_metrics_row(
            stage="s2_joint_bptt",
            step=step_v,
            status="live",
            met=rec,
            commit_sha="",
            ts=ts_v,
            source="live_metrics_jsonl",
        )
        existing = dedup_rows["s2_joint_bptt"].get(step_v)
        if existing is None:
            dedup_rows["s2_joint_bptt"][step_v] = row
        else:
            for k, v in row.items():
                if existing.get(k) is None and v is not None:
                    existing[k] = v

    # Pass 3: enrich with local stage metrics.jsonl (e.g. step-0 evals for s1/s1b)
    for st, rows_l in local_metrics.items():
        if st not in STAGES:
            continue
        for rec in rows_l:
            step_v = _safe_int(rec.get("step"))
            if step_v is None:
                continue
            if "eval/loss_d1" in rec and "eval/loss_d3" in rec:
                if st == "s1b_hssd_distill" and step_v > 400 and (_safe_float(rec.get("eval/loss_d1")) or 0.0) > 5.0:
                    continue
                raw_eval_events[st].append(
                    {
                        "step": step_v,
                        "status": "eval",
                        "d1": _safe_float(rec.get("eval/loss_d1")),
                        "d2": _safe_float(rec.get("eval/loss_d2")),
                        "d3": _safe_float(rec.get("eval/loss_d3")),
                        "depth_gain": _safe_float(rec.get("eval/depth_gain")),
                        "commit_sha": "",
                        "ts": "",
                        "source": "local_metrics_jsonl",
                    }
                )

    # Pass 4: enrich stage tables with supervisor log mem_alloc_gib & intermediate s2 steps
    for srow in supervisor_data.get("step_samples") or []:
        st = srow["stage"]
        step_v = srow["step"]
        head = srow["head"]
        if st not in STAGES:
            continue
        if st == "s1b_hssd_distill" and head == "fbd7a35" and step_v > 400:
            continue
        if st == "s2_joint_bptt" and head == "b2c5513":
            continue
        existing = dedup_rows[st].get(step_v)
        if existing is not None:
            if existing.get("mem_alloc_gib") is None:
                existing["mem_alloc_gib"] = srow["mem_alloc_gib"]
            if existing.get("mem_peak_gib") is None:
                existing["mem_peak_gib"] = srow["mem_peak_gib"]
        elif st in ("s2_joint_bptt", "s2_v2_joint_bptt"):
            dedup_rows[st][step_v] = {
                "stage": st,
                "step": step_v,
                "status": f"sup_{head}",
                "commit_sha": head,
                "ts": srow["ts"],
                "source": "supervisor_log",
                "depth": srow["depth"],
                "loss_lm": srow["loss_lm"],
                "tokens_seen": step_v * 262_144 if st == "s2_joint_bptt" else None,
                "tok_per_s": srow["tok_per_s"],
                "lr_factor": None,
                "grad_norm": None,
                "grad_norm_ema": None,
                "skipped_params": None,
                "skipped_total": None,
                "mem_alloc_gib": srow["mem_alloc_gib"],
                "mem_peak_gib": srow["mem_peak_gib"],
                "eval_loss_d1": None,
                "eval_loss_d2": None,
                "eval_loss_d3": None,
                "eval_depth_gain": None,
                "reroute_rate": None,
                "unique_experts_core": None,
                "effective_expert_mult": None,
                "gate_abs_mean_p2": None,
                "gate_abs_mean_p3": None,
                "bridge_w_h_prev_p3": None,
                "bridge_w_anchor_p3": None,
                "bridge_subspace_up_abs_mean_p3": None,
                "bridge_step_emb_norm_p3": None,
                "distill_phase": None,
                "hssd_blend": None,
                "distill_kl": None,
                "layerwise_mse": None,
                "hssd_budget_aux": None,
                "hssd_vault_duty": None,
                "hssd_dead_page_count": None,
                "lsa_kl_total": None,
                "lsa_kl_mean": None,
                "lsa_recall": None,
                "lsa_min_head_recall": None,
                "calib_prefix_len": None,
                "bptt_replayed_tokens": 2048,
                "bptt_offload_to_cpu": 0.0 if head in ("5a82677", "8409b6e", "574898d") else None,
                "gdn2_adapter_norm": None,
                "gdn2_qed_lambda_abs_mean": None,
                "gdn2_neg_eigval_scale_mean": None,
                "gdn2_decay_horizon_mult_max": None,
                "gdn2_erase_write_l1": None,
                "gdn2_decay_spread": None,
            }

    # Pass 5: verified step-0 launch anchors (only for canonical repo when missing)
    if use_launch_anchors and repo == DEFAULT_REPO:
        for st, anchor in VERIFIED_LAUNCH_ANCHORS.items():
            if not any(ev["step"] == 0 for ev in raw_eval_events.get(st, [])):
                raw_eval_events[st].append(dict(anchor))

    # Deduplicate eval curves per stage: keep earliest step per unique (d1,d2,d3) signature
    eval_curves: dict[str, list[dict[str, Any]]] = {}
    for st in STAGES:
        evs = sorted(raw_eval_events[st], key=lambda x: (int(x["step"]), 0 if x.get("status") != "completed" else 1))
        uniq_by_sig: dict[tuple[float, float, float], dict[str, Any]] = {}
        ordered_evs: list[dict[str, Any]] = []
        for ev in evs:
            d1, d2, d3 = ev.get("d1"), ev.get("d2"), ev.get("d3")
            if d1 is None or d3 is None:
                continue
            sig = (round(float(d1), 4), round(float(d2 or 0.0), 4), round(float(d3), 4))
            if sig in uniq_by_sig:
                prev_ev = uniq_by_sig[sig]
                if ev["step"] == prev_ev["step"] and ev.get("status") == "completed":
                    prev_ev.update(ev)
                continue
            same_step = [x for x in ordered_evs if x["step"] == ev["step"]]
            if same_step:
                if ev.get("status") == "completed":
                    same_step[0].update(ev)
                continue
            ev_copy = dict(ev)
            if ev_copy.get("depth_gain") is None:
                ev_copy["depth_gain"] = round(float(d1) - float(d3), 6)
            uniq_by_sig[sig] = ev_copy
            ordered_evs.append(ev_copy)
        eval_curves[st] = sorted(ordered_evs, key=lambda x: int(x["step"]))

    stage_tables: dict[str, list[dict[str, Any]]] = {}
    for st in STAGES:
        stage_tables[st] = [dedup_rows[st][k] for k in sorted(dedup_rows[st])]

    kv_memory_1m = compute_kv_memory_1m()
    stage_summaries: dict[str, dict[str, Any]] = {}
    total_tokens = 0
    for st in STAGES:
        rows = stage_tables[st]
        evs = eval_curves[st]
        meta = stage_latest_meta.get(st, {})
        tps_vals = [float(r["tok_per_s"]) for r in rows if r.get("tok_per_s") is not None]
        pk_vals = [float(r["mem_peak_gib"]) for r in rows if r.get("mem_peak_gib") is not None]
        sk_vals = [float(r["skipped_total"]) for r in rows if r.get("skipped_total") is not None]
        gn_vals = [float(r["grad_norm"]) for r in rows if r.get("grad_norm") is not None]
        tok_seen = max([int(r["tokens_seen"]) for r in rows if r.get("tokens_seen") is not None] or [0])
        total_tokens += tok_seen

        ts_objs = [t for t in (_parse_iso(r.get("ts")) for r in rows if r.get("source") == "hub_manifest") if t]
        hub_span_h = round((max(ts_objs) - min(ts_objs)).total_seconds() / 3600.0, 2) if len(ts_objs) >= 2 else None

        p2_skips = [
            float(r["skipped_total"])
            for r in rows
            if st == "s1b_hssd_distill" and int(r["step"]) > 400 and r.get("skipped_total") is not None
        ]
        post_skip_max = (
            round(max(p2_skips), 1) if p2_skips else (round(max(sk_vals), 1) if sk_vals else None)
        )
        latest_hub_step = _safe_int(meta.get("step"))
        latest_row_step = _safe_int(rows[-1]["step"]) if rows else 0
        latest_step_val = max(latest_hub_step or 0, latest_row_step or 0)
        stage_summaries[st] = {
            "status": meta.get("status") or (rows[-1]["status"] if rows else "not_started"),
            "latest_step": latest_step_val,
            "latest_hub_step": latest_hub_step,
            "target_steps": meta.get("target_steps"),
            "latest_commit_sha": meta.get("commit_sha") or (rows[-1]["commit_sha"] if rows else ""),
            "n_params": meta.get("n_params"),
            "pct_of_34_661b": round(100.0 * meta["n_params"] / 34_661_000_000, 3) if meta.get("n_params") else None,
            "n_rows": len(rows),
            "tokens_seen": tok_seen,
            "tok_per_s_mean": round(sum(tps_vals) / len(tps_vals), 1) if tps_vals else None,
            "tok_per_s_min": round(min(tps_vals), 1) if tps_vals else None,
            "tok_per_s_max": round(max(tps_vals), 1) if tps_vals else None,
            "mem_peak_max_gib": round(max(pk_vals), 2) if pk_vals else None,
            "skipped_total_max": round(max(sk_vals), 1) if sk_vals else None,
            "post_fix_skipped_max": post_skip_max,
            "grad_norm_final": round(gn_vals[-1], 4) if gn_vals else None,
            "hub_span_hours": hub_span_h,
            "eval_first": evs[0] if evs else None,
            "eval_last": evs[-1] if evs else None,
            "eval_best_depth_gain": max(evs, key=lambda e: float(e.get("depth_gain") or -999.0)) if evs else None,
        }

    s1b_rb = [r for r in rolled_back_events if r.get("stage") == "s1b_hssd_distill"]
    rollback_summary = {
        "s1_reset_commits": len([r for r in rolled_back_events if r.get("stage") == "s1_loop_a6b"]),
        "s1b_phase2_explosion_commits": len(s1b_rb),
        "s1b_explosion_step_range": [min(r["step"] for r in s1b_rb), max(r["step"] for r in s1b_rb)] if s1b_rb else [],
        "s1b_explosion_peak_loss": round(max(float(r["loss_lm"]) for r in s1b_rb if r.get("loss_lm") is not None), 4)
        if any(r.get("loss_lm") is not None for r in s1b_rb)
        else None,
        "s1b_explosion_max_grad_norm": max(float(r["grad_norm"]) for r in s1b_rb if r.get("grad_norm") is not None)
        if any(r.get("grad_norm") is not None for r in s1b_rb)
        else None,
        "s1b_explosion_max_skipped": max(
            float(r["skipped_total"]) for r in s1b_rb if r.get("skipped_total") is not None
        )
        if any(r.get("skipped_total") is not None for r in s1b_rb)
        else None,
    }

    # Exact log-derived single-GPU hours ledger
    by_head = supervisor_data.get("active_gpu_hours_by_head") or {}
    probe_h = round(float(((probe_report or {}).get("summary") or {}).get("total_seconds") or 0.0) / 3600.0, 2)
    qc_h = round(float((quality_check or {}).get("total_eval_wall_sec") or 0.0) / 3600.0, 2)
    part_a_sm = (dplr_bench or {}).get("part_a_smoke") or {}
    smoke_s3_s = float((part_a_sm.get("smoke_s3v2") or {}).get("elapsed_sec") or 0.0)
    smoke_s2_s = float((part_a_sm.get("smoke_s2v2") or {}).get("elapsed_sec") or 0.0)
    smoke_h = round((smoke_s3_s + smoke_s2_s) / 3600.0, 2)
    gate_sec = (
        float(((s3v2_gate or {}).get("gate1_cuda_gdn2") or {}).get("elapsed_sec") or 0.0)
        + float(((s3v2_gate or {}).get("gate2_s3v2_step600_truth_check") or {}).get("elapsed_sec") or 0.0)
        + float(((s3v2_gate or {}).get("gate3_s2v2_4x32k_d3_smoke") or {}).get("elapsed_sec") or 0.0)
    )
    gpu_hours = {
        "s0_probe_h": probe_h,
        "s1_loop_a6b_h": round(float(by_head.get("c22bf62", 0.0)), 2),
        "s1_quality_check_h": qc_h,
        "s1b_hssd_p1_and_prefix_h": round(float(by_head.get("fbd7a35", 0.0)), 2),
        "s1b_hssd_p2_clean_h": round(float(by_head.get("3a17c0d", 0.0)), 2),
        "s3_hssd_h": round(float(by_head.get("513879f", 0.0)) + float(by_head.get("4f09463", 0.0)), 2),
        "s2_hssd_preopt_h": round(float(by_head.get("2e37bf6", 0.0)), 2),
        "s2_hssd_leaky_h": round(float(by_head.get("b2c5513", 0.0)), 2),
        "s2_hssd_5a82677_h": round(float(by_head.get("5a82677", 0.0)), 2),
        "s2_hssd_8409b6e_total_h": round(float(by_head.get("8409b6e", 0.0)), 2),
        "s2_hssd_tf524_slowdown_h": (supervisor_data.get("tf524_slowdown_window") or {}).get("slowdown_hours"),
        "s2_hssd_2f5a6ba_post_fix_h": (supervisor_data.get("tf524_slowdown_window") or {}).get("post_fix_hours"),
        "v2_smoke_part_a_h": smoke_h,
        "s3_v2_native_gdn_h": round(float(by_head.get("28b88e4", 0.0)), 2),
        "s3v2_gate123_h": round(gate_sec / 3600.0, 2),
        "s2_v2_gdn2_h": round(float(by_head.get("574898d", 0.0)), 2),
        "supervisor_v1_active_h": round(
            float(supervisor_data.get("active_gpu_hours_v1") or supervisor_data.get("active_gpu_hours_total") or 0.0), 2
        ),
        "supervisor_v2_active_h": round(float(supervisor_data.get("active_gpu_hours_v2") or 0.0), 2),
        "supervisor_s2v2_active_h": round(float(supervisor_data.get("active_gpu_hours_s2v2") or 0.0), 2),
        "supervisor_total_active_h": round(float(supervisor_data.get("active_gpu_hours_total") or 0.0), 2),
    }

    summary_metrics = {
        "generated_at_utc": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "repo": repo,
        "dataset_repo": DEFAULT_DATASET_REPO,
        "base_model": "unsloth/Qwen3.6-35B-A3B-NVFP4",
        "hardware": (
            "1x NVIDIA RTX PRO 6000 Blackwell Server Edition "
            "(GB202, sm_120, 96 GB GDDR7 ECC, 512-bit, 1,597 GB/s, 24,064 CUDA cores, 600 W TDP)"
        ),
        "total_hub_commits": len(raw_commits),
        "total_tokens_trained": total_tokens,
        "gpu_hours": gpu_hours,
        "stages": stage_summaries,
        "eval_curves": eval_curves,
        "rollback_summary": rollback_summary,
        "supervisor_s2_regimes": supervisor_data.get("s2_regimes") or {},
        "supervisor_stage_head_summary": supervisor_data.get("stage_head_summary") or [],
        "tf524_slowdown_window": supervisor_data.get("tf524_slowdown_window"),
        "kv_memory_1m": kv_memory_1m,
        "probe_summary": (probe_report or {}).get("summary") if isinstance(probe_report, dict) else None,
        "probe_gates": (probe_report or {}).get("gates") if isinstance(probe_report, dict) else None,
        "truth_check": truth_check,
        "dplr_bench": dplr_bench,
        "dataset_audit": dataset_audit,
        "s3v2_gate": s3v2_gate,
        "systems_milestones": VERIFIED_SYSTEMS_MILESTONES,
    }

    return {
        "repo": repo,
        "hub_manifest_history": {
            "repo": repo,
            "total_commits": len(hub_history_annotated),
            "commits": hub_history_annotated,
        },
        "rolled_back_events": rolled_back_events,
        "stage_tables": stage_tables,
        "eval_curves": eval_curves,
        "summary_metrics": summary_metrics,
        "supervisor": supervisor_data,
        "probe_report": probe_report,
        "quality_check": quality_check,
        "truth_check": truth_check,
        "dplr_bench": dplr_bench,
        "dataset_audit": dataset_audit,
        "s3v2_gate": s3v2_gate,
        "kv_memory_1m": kv_memory_1m,
    }


# ---------------------------------------------------------------------------
# CSV Writer
# ---------------------------------------------------------------------------
STAGE_CSV_COLUMNS: dict[str, list[str]] = {
    "s1_loop_a6b": [
        "step", "status", "commit_sha", "ts", "source", "depth", "loss_lm", "tokens_seen",
        "tok_per_s", "lr_factor", "grad_norm", "skipped_total", "mem_alloc_gib", "mem_peak_gib",
        "reroute_rate", "unique_experts_core", "effective_expert_mult",
        "eval_loss_d1", "eval_loss_d2", "eval_loss_d3", "eval_depth_gain",
    ],
    "s1b_hssd_distill": [
        "step", "status", "commit_sha", "ts", "source", "depth", "distill_phase", "hssd_blend",
        "loss_lm", "distill_kl", "layerwise_mse", "hssd_budget_aux", "tokens_seen", "tok_per_s",
        "grad_norm", "skipped_total", "mem_alloc_gib", "mem_peak_gib",
        "eval_loss_d1", "eval_loss_d2", "eval_loss_d3", "eval_depth_gain",
    ],
    "s3_long_1m_lsa": [
        "step", "status", "commit_sha", "ts", "source", "depth", "loss_lm", "lsa_kl_total",
        "lsa_kl_mean", "lsa_recall", "lsa_min_head_recall", "calib_prefix_len", "tokens_seen",
        "tok_per_s", "grad_norm", "skipped_total", "mem_alloc_gib", "mem_peak_gib",
        "eval_loss_d1", "eval_loss_d2", "eval_loss_d3", "eval_depth_gain",
    ],
    "s2_joint_bptt": [
        "step", "status", "commit_sha", "ts", "source", "depth", "loss_lm", "lsa_kl_mean",
        "hssd_vault_duty", "hssd_dead_page_count",
        "bptt_replayed_tokens", "bptt_offload_to_cpu", "tokens_seen", "tok_per_s", "lr_factor",
        "grad_norm", "skipped_total", "mem_alloc_gib", "mem_peak_gib", "reroute_rate",
        "gate_abs_mean_p2", "gate_abs_mean_p3",
        "eval_loss_d1", "eval_loss_d2", "eval_loss_d3", "eval_depth_gain",
    ],
    "s3_v2_long_1m_lsa": [
        "step", "status", "commit_sha", "ts", "source", "depth", "loss_lm", "lsa_kl_total",
        "lsa_kl_mean", "lsa_recall", "lsa_min_head_recall", "calib_prefix_len", "tokens_seen",
        "tok_per_s", "grad_norm", "skipped_total", "mem_alloc_gib", "mem_peak_gib",
        "gate_abs_mean_p2", "gate_abs_mean_p3", "bridge_w_h_prev_p3", "bridge_w_anchor_p3",
        "bridge_subspace_up_abs_mean_p3", "bridge_step_emb_norm_p3",
        "eval_loss_d1", "eval_loss_d2", "eval_loss_d3", "eval_depth_gain",
    ],
    "s2_v2_joint_bptt": [
        "step", "status", "commit_sha", "ts", "source", "depth", "loss_lm", "lsa_kl_mean",
        "bptt_replayed_tokens", "bptt_offload_to_cpu", "tokens_seen", "tok_per_s", "lr_factor",
        "grad_norm", "skipped_total", "mem_alloc_gib", "mem_peak_gib", "reroute_rate",
        "unique_experts_core", "effective_expert_mult",
        "gate_abs_mean_p2", "gate_abs_mean_p3", "bridge_w_anchor_p3",
        "gdn2_adapter_norm", "gdn2_qed_lambda_abs_mean", "gdn2_neg_eigval_scale_mean",
        "gdn2_decay_horizon_mult_max", "gdn2_erase_write_l1", "gdn2_decay_spread",
        "eval_loss_d1", "eval_loss_d2", "eval_loss_d3", "eval_depth_gain",
    ],
}


def _format_csv_cell(val: Any) -> str:
    if val is None:
        return ""
    if isinstance(val, float):
        return f"{val:.6g}" if (abs(val) < 1e-3 and val != 0.0) else f"{val:.6f}".rstrip("0").rstrip(".")
    return str(val)


def write_stage_csv(path: str, stage: str, rows: list[dict[str, Any]]) -> None:
    cols = STAGE_CSV_COLUMNS[stage]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(cols)
        for r in rows:
            writer.writerow([_format_csv_cell(r.get(c)) for c in cols])


# ---------------------------------------------------------------------------
# Publication-Ready Markdown Report Renderer (Unified v1 -> Forensics -> v2)
# ---------------------------------------------------------------------------
def _fmt_f(val: Any, digits: int = 4, signed: bool = False) -> str:
    f = _safe_float(val)
    if f is None:
        return "—"
    return f"{f:+.{digits}f}" if signed else f"{f:.{digits}f}"


def _fmt_i(val: Any) -> str:
    i = _safe_int(val)
    return f"{i:,}" if i is not None else "—"


# render_experiment_report_md, render_experiment_report_md_with_order, and
# render_experiment_report_zh_md are implemented in qloop.report_paper_en / qloop.report_paper_zh
# and re-exported at module level above.

# ---------------------------------------------------------------------------
# End-to-End Export Orchestrator & CLI
# ---------------------------------------------------------------------------
def _assert_secret_free(path: str, text: str) -> None:
    if contains_secret(text):
        raise RuntimeError(f"Refusing to write {path}: detected potential unredacted secret/token!")


def _resolve_artifact(explicit: str | None, search_roots: list[str], rel_candidates: tuple[str, ...]) -> str | None:
    if explicit and os.path.isfile(explicit):
        return explicit
    for root in search_roots:
        if not root:
            continue
        for rel in rel_candidates:
            cand = os.path.join(root, rel)
            if os.path.isfile(cand):
                return cand
    return None


def export_experiment_report(
    repo: str = DEFAULT_REPO,
    session: str | None = None,
    supervisor_log: str | None = "runs/colab_supervisor.log",
    supervisor_v2_log: str | None = "runs/colab_supervisor_v2.log",
    supervisor_s2v2_log: str | None = "runs/colab_supervisor_s2v2.log",
    runs_dir: str | None = None,
    truth_check_path: str | None = None,
    dplr_bench_path: str | None = None,
    dataset_audit_path: str | None = None,
    s3v2_gate_path: str | None = None,
    out_dir: str = "docs/experiment_logs",
    report_path: str = "docs/EXPERIMENT_REPORT.md",
    report_zh_path: str | None = None,
    offline_dir: str | None = None,
    offline: bool = False,
    colab_bin: str = DEFAULT_COLAB_BIN,
) -> dict[str, Any]:
    """Collect telemetry, archive CSV/JSON, render 7 SVGs, and write EN (+ optional ZH) academic reports."""
    raw_commits, probe_report, quality_check = collect_hub_history(
        repo=repo,
        offline_dir=offline_dir,
        offline=offline,
    )
    live_s2 = fetch_live_session_metrics(
        session=session,
        colab_bin=colab_bin,
        offline_dir=offline_dir,
        offline=offline,
    )

    eff_sup_log = supervisor_log
    eff_sup_v2_log = supervisor_v2_log
    eff_sup_s2v2_log = supervisor_s2v2_log
    if offline_dir:
        cand_sup = os.path.join(offline_dir, "colab_supervisor.log")
        eff_sup_log = cand_sup if os.path.isfile(cand_sup) else (
            eff_sup_log if (eff_sup_log and os.path.isfile(eff_sup_log)) else None
        )
        cand_sup_v2 = os.path.join(offline_dir, "colab_supervisor_v2.log")
        eff_sup_v2_log = cand_sup_v2 if os.path.isfile(cand_sup_v2) else None
        cand_sup_s2v2 = os.path.join(offline_dir, "colab_supervisor_s2v2.log")
        eff_sup_s2v2_log = cand_sup_s2v2 if os.path.isfile(cand_sup_s2v2) else None

    sup_v1 = parse_supervisor_log(eff_sup_log)
    sup_v2 = parse_supervisor_log(eff_sup_v2_log, stage_override="s3_v2_long_1m_lsa")
    sup_s2v2 = parse_supervisor_log(eff_sup_s2v2_log, stage_override="s2_v2_joint_bptt")
    supervisor_data = _merge_supervisor_logs(sup_v1, sup_v2, sup_s2v2)

    eff_runs_dir = runs_dir
    if eff_runs_dir is None and eff_sup_log and os.path.isfile(eff_sup_log):
        eff_runs_dir = os.path.dirname(os.path.abspath(eff_sup_log))
    local_metrics, local_qc = load_local_stage_metrics(offline_dir if offline_dir else eff_runs_dir)
    if quality_check is None and local_qc is not None:
        quality_check = local_qc

    search_roots = [offline_dir] if offline_dir else [r for r in (eff_runs_dir, "runs") if r]
    tc_file = _resolve_artifact(
        truth_check_path,
        search_roots,
        ("truth_check_summary.json", "eval_truth_check_s2_step500/truth_check_summary.json"),
    )
    db_file = _resolve_artifact(
        dplr_bench_path,
        search_roots,
        ("dplr_bench_summary.json", "bench_v2_smoke_dplr/summary.json"),
    )
    da_file = _resolve_artifact(
        dataset_audit_path,
        search_roots,
        ("dataset_audit_summary.json", "datasets/Loop64-Fable-SIRSI-v1/audit_summary.json"),
    )
    gate_file = _resolve_artifact(
        s3v2_gate_path,
        search_roots,
        ("s3v2_gate_summary.json", "eval_s3v2_step600_gate/summary.json"),
    )
    truth_check = load_truth_check_summary(tc_file)
    dplr_bench = load_dplr_bench_summary(db_file)
    dataset_audit = load_dataset_audit_summary(da_file)
    s3v2_gate = load_s3v2_gate_summary(gate_file)

    use_anchors = (not offline_dir) and (repo == DEFAULT_REPO)
    bundle = normalize_experiment_data(
        raw_commits=raw_commits,
        live_s2=live_s2,
        supervisor_data=supervisor_data,
        local_metrics=local_metrics,
        probe_report=probe_report,
        quality_check=quality_check,
        repo=repo,
        use_launch_anchors=use_anchors,
        truth_check=truth_check,
        dplr_bench=dplr_bench,
        dataset_audit=dataset_audit,
        s3v2_gate=s3v2_gate,
    )

    charts_dir = os.path.join(out_dir, "charts")
    os.makedirs(charts_dir, exist_ok=True)
    report_parent = os.path.dirname(os.path.abspath(report_path))
    if report_parent:
        os.makedirs(report_parent, exist_ok=True)

    written_files: list[str] = []

    def _write_json(filename: str, payload: Any) -> None:
        p = os.path.join(out_dir, filename)
        txt = json.dumps(payload, indent=2, sort_keys=False) + "\n"
        _assert_secret_free(p, txt)
        with open(p, "w", encoding="utf-8") as f:
            f.write(txt)
        written_files.append(p)

    # 1. hub_manifest_history.json
    _write_json("hub_manifest_history.json", bundle["hub_manifest_history"])

    # 2. Per-stage CSVs (v1 + clean v2)
    for st in STAGES:
        csv_path = os.path.join(out_dir, f"{st}.csv")
        write_stage_csv(csv_path, st, bundle["stage_tables"][st])
        with open(csv_path, encoding="utf-8") as f:
            _assert_secret_free(csv_path, f.read())
        written_files.append(csv_path)

    # 3. summary_metrics.json + archived sanitized local verified JSON artifacts
    _write_json("summary_metrics.json", bundle["summary_metrics"])
    if truth_check is not None:
        _write_json("truth_check_summary.json", truth_check)
    if dplr_bench is not None:
        _write_json("dplr_bench_summary.json", dplr_bench)
    if dataset_audit is not None:
        _write_json("dataset_audit_summary.json", dataset_audit)
    if s3v2_gate is not None:
        _write_json("s3v2_gate_summary.json", s3v2_gate)

    # 4. 7 Pure-SVG vector charts
    for fname, renderer in ALL_CHART_RENDERERS.items():
        svg_path = os.path.join(charts_dir, fname)
        svg_text = renderer(bundle)
        _assert_secret_free(svg_path, svg_text)
        with open(svg_path, "w", encoding="utf-8") as f:
            f.write(svg_text)
        written_files.append(svg_path)

    # 5. EXPERIMENT_REPORT.md (English IEEE academic paper)
    rel_charts = os.path.relpath(charts_dir, start=report_parent or ".").replace("\\", "/")
    md_text, cite_order = render_experiment_report_md_with_order(bundle, chart_rel_dir=rel_charts)
    _assert_secret_free(report_path, md_text)
    validate_ieee_citations(md_text)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(md_text)
    written_files.append(report_path)

    # 6. Optional EXPERIMENT_REPORT_ZH.md (Simplified Chinese IEEE academic paper)
    if report_zh_path:
        zh_parent = os.path.dirname(os.path.abspath(report_zh_path))
        if zh_parent:
            os.makedirs(zh_parent, exist_ok=True)
        rel_charts_zh = os.path.relpath(charts_dir, start=zh_parent or ".").replace("\\", "/")
        zh_text = render_experiment_report_zh_md(
            bundle,
            chart_rel_dir=rel_charts_zh,
            expected_cite_order=cite_order,
        )
        _assert_secret_free(report_zh_path, zh_text)
        validate_ieee_citations(zh_text)
        with open(report_zh_path, "w", encoding="utf-8") as f:
            f.write(zh_text)
        written_files.append(report_zh_path)

    return {
        "repo": repo,
        "written_files": written_files,
        "total_commits": len(raw_commits),
        "stage_row_counts": {st: len(bundle["stage_tables"][st]) for st in STAGES},
        "rolled_back_count": len(bundle["rolled_back_events"]),
        "live_s2_rows": len(live_s2),
        "ieee_reference_count": len(cite_order),
        "ingested_artifacts": {
            "truth_check": truth_check is not None,
            "dplr_bench": dplr_bench is not None,
            "dataset_audit": dataset_audit is not None,
            "s3v2_gate": s3v2_gate is not None,
            "supervisor_v2": bool(sup_v2.get("present")),
            "supervisor_s2v2": bool(sup_s2v2.get("present")),
        },
    }


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=(
            "Export verified v1 + v2 experiment telemetry (HF Hub + supervisor logs + local artifacts), "
            "7 pure-SVG charts, and EN/ZH IEEE academic reports."
        )
    )
    ap.add_argument("--repo", default=DEFAULT_REPO, help="Hugging Face Hub repository ID")
    ap.add_argument(
        "--session",
        default=None,
        help="Optional Colab session name for read-only live metrics snapshot (disabled by default)",
    )
    ap.add_argument("--supervisor-log", default="runs/colab_supervisor.log", help="Path to local colab_supervisor.log")
    ap.add_argument(
        "--supervisor-v2-log",
        default="runs/colab_supervisor_v2.log",
        help="Path to local colab_supervisor_v2.log",
    )
    ap.add_argument(
        "--supervisor-s2v2-log",
        default="runs/colab_supervisor_s2v2.log",
        help="Path to local colab_supervisor_s2v2.log",
    )
    ap.add_argument(
        "--runs-dir",
        default=None,
        help="Directory containing local stage metrics & verified JSON artifacts",
    )
    ap.add_argument("--truth-check", default=None, help="Explicit path to truth_check_summary.json")
    ap.add_argument("--dplr-bench", default=None, help="Explicit path to bench_v2_smoke_dplr/summary.json")
    ap.add_argument("--dataset-audit", default=None, help="Explicit path to Loop64-Fable-SIRSI-v1/audit_summary.json")
    ap.add_argument("--s3v2-gate", default=None, help="Explicit path to eval_s3v2_step600_gate/summary.json")
    ap.add_argument("--out-dir", default="docs/experiment_logs", help="Output directory for raw CSV/JSON and charts/")
    ap.add_argument("--report", default="docs/EXPERIMENT_REPORT.md", help="Output English Markdown report path")
    ap.add_argument(
        "--report-zh",
        default="docs/EXPERIMENT_REPORT_ZH.md",
        help="Output Simplified Chinese Markdown report path (pass '' to skip)",
    )
    ap.add_argument("--offline-dir", default=None, help="Offline fixture directory (zero network/Colab calls)")
    ap.add_argument("--offline", action="store_true", help="Skip network and live Colab session queries")
    ap.add_argument("--colab-bin", default=DEFAULT_COLAB_BIN, help="Path to colab CLI binary")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    res = export_experiment_report(
        repo=args.repo,
        session=args.session,
        supervisor_log=args.supervisor_log,
        supervisor_v2_log=args.supervisor_v2_log,
        supervisor_s2v2_log=args.supervisor_s2v2_log,
        runs_dir=args.runs_dir,
        truth_check_path=args.truth_check,
        dplr_bench_path=args.dplr_bench,
        dataset_audit_path=args.dataset_audit,
        s3v2_gate_path=args.s3v2_gate,
        out_dir=args.out_dir,
        report_path=args.report,
        report_zh_path=args.report_zh or None,
        offline_dir=args.offline_dir,
        offline=args.offline,
        colab_bin=args.colab_bin,
    )
    print(json.dumps(res, indent=2))
    return 0
