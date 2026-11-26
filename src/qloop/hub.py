"""Hugging Face Hub commit-history state sync + 230 GB Colab disk budget manager.

Designed for multi-session Google Colab training pushing by stage commits to a public HF repo
(`TassP0lar/Qwen3.6-35B-A3B-Loop64`) authenticated via `HF_TOKEN`:

  1. **Commit-based stage tracking**: Every periodic checkpoint, stage completion, ESFT scan,
     probe report and evaluation summary is committed atomically to Hugging Face Hub alongside an
     authoritative `manifest.json` and auto-rendered `README.md` Model Card.
  2. **Zero-Drive cross-session auto-resume**: A fresh Colab runtime queries `manifest.json` (and
     Hub commit history), skips already-completed stages, pulls *only* the latest active or
     prerequisite trainable bundle (~0.5-3.5 GB, never duplicating the ~22.3 GiB frozen NVFP4 base weights),
     and resumes at the exact optimizer step and data-stream cursor.
  3. **230 GB Colab disk guard**: Keeps `keep_last=1` locally, strips completed-stage optimizer
     states, deletes superseded stage checkpoint directories once verified on Hub, and purges
     transient HF xet/upload staging caches while preserving cached `unsloth/Qwen3.6-35B-A3B-NVFP4` weights.
"""
from __future__ import annotations

import datetime as _dt
import glob
import json
import logging
import os
import shutil
import tempfile
from dataclasses import dataclass, field, fields
from typing import Any

log = logging.getLogger("qloop")

DEFAULT_HF_REPO = "TassP0lar/Qwen3.6-35B-A3B-Loop64"
DEFAULT_PACK_REPO = "TassP0lar/Qwen3.6-35B-A6B-LoopHSSD-Pack"
DEFAULT_BASE_MODEL = "unsloth/Qwen3.6-35B-A3B-NVFP4"
MANIFEST_FILE = "manifest.json"
PIPELINE_ORDER: tuple[str, ...] = (
    "s0_probe",
    "s1_loop_heal",
    "s1_rescan",
    "s2_sft",
    "s3_long_lsa",
    "s4_rl",
    "eval",
)
V3_PIPELINE_ORDER: tuple[str, ...] = (
    "s0_probe",
    "s1_loop_a6b",
    "s1b_hssd_distill",
    "s3_long_1m_lsa",
    "s2_joint_bptt",
    "s4_rl",
)
V3_GDN_PIPELINE_ORDER: tuple[str, ...] = (
    "s0_probe",
    "s1_loop_a6b",
    "s3_v2_long_1m_lsa",
    "s2_v2_joint_bptt",
    "s4_v2_rl",
)
_V3_STAGE_SET = frozenset({"s1_loop_a6b", "s1b_hssd_distill", "s3_long_1m_lsa", "s2_joint_bptt"})
_V3_GDN_STAGE_SET = frozenset({"s3_v2_long_1m_lsa", "s2_v2_joint_bptt", "s4_v2_rl"})


def resolve_pipeline_order(stages_map: dict[str, Any] | Any) -> tuple[str, ...]:
    """Select the active pipeline stage ordering from manifest stage keys."""
    if stages_map:
        if any(s in stages_map for s in _V3_GDN_STAGE_SET):
            return V3_GDN_PIPELINE_ORDER
        if any(s in stages_map for s in _V3_STAGE_SET):
            return V3_PIPELINE_ORDER
    return PIPELINE_ORDER

# Artifact files produced outside `checkpoints/` that downstream stages depend on.
ARTIFACT_PATHS: dict[str, tuple[str, str]] = {
    "s0_probe/probe.json": ("artifacts/s0_probe/probe.json", "s0_probe/probe.json"),
    "s0_probe/probe_report.json": ("artifacts/probes/probe_report.json", "s0_probe/probe_report.json"),
    "s0_probe/probe_report.md": ("artifacts/probes/probe_report.md", "s0_probe/probe_report.md"),
    "s0_probe/esft_stats.json": ("artifacts/s0_probe/esft_stats.json", "s0_probe/esft_stats.json"),
    "s1_loop_heal/esft_stats_looped.json": (
        "artifacts/s1_loop_heal/esft_stats_looped.json",
        "s1_loop_heal/esft_stats_looped.json",
    ),
    "s1_loop_a6b/esft_stats_looped.json": (
        "artifacts/s1_loop_a6b/esft_stats_looped.json",
        "s1_loop_a6b/esft_stats_looped.json",
    ),
}


def resolve_stage_config_path(stage_name: str, config_root: str = "configs") -> str:
    """Resolve a v2 or v3 pipeline stage identifier to its YAML config path under `config_root`."""
    st = stage_name.strip()
    if st == "eval":
        return os.path.join(config_root, "eval", "suite.yaml")
    if st == "s1_rescan":
        return os.path.join(config_root, "stages", "s1_loop_heal.yaml")
    cand = os.path.join(config_root, "stages", f"{st}.yaml")
    if st in PIPELINE_ORDER or st in V3_PIPELINE_ORDER or st in V3_GDN_PIPELINE_ORDER or os.path.isfile(cand):
        return cand
    raise ValueError(f"unknown pipeline stage {stage_name!r}")


def _utc_now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def resolve_run_path(path: str | None, run_root: str | None = None) -> str | None:
    """Resolve `runs/<subpath>` against `$QLOOP_RUN_ROOT` when custom run roots are used."""
    if not path:
        return None
    if os.path.exists(path):
        return path
    root = run_root or os.environ.get("QLOOP_RUN_ROOT")
    if root:
        norm = path.replace("\\", "/")
        rel = norm[5:] if norm.startswith("runs/") else norm
        cand = os.path.join(root, rel)
        if os.path.exists(cand) or norm.startswith("runs/"):
            return cand
    return path


def resolve_hf_token(explicit: str | None = None) -> str | None:
    """Resolve HF write token from explicit arg -> env (`HF_TOKEN`) -> Colab Secrets -> HF cache."""
    if explicit and explicit.strip():
        return explicit.strip()
    for env_key in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        val = os.environ.get(env_key)
        if val and val.strip():
            return val.strip()
    try:
        from google.colab import userdata  # type: ignore[import-not-found]

        val = userdata.get("HF_TOKEN")
        if val and str(val).strip():
            tok = str(val).strip()
            os.environ.setdefault("HF_TOKEN", tok)
            return tok
    except Exception:
        pass
    try:
        from huggingface_hub import get_token

        val = get_token()
        if val and str(val).strip():
            return str(val).strip()
    except Exception:
        pass
    return None


# ============================================================================ 230 GB disk guard

def _dir_bytes(path: str) -> int:
    try:
        if not path or not os.path.exists(path):
            return 0
        if os.path.isfile(path):
            return os.path.getsize(path)
    except OSError:
        return 0
    total = 0
    try:
        for root, _dirs, files in os.walk(path):
            for fn in files:
                fp = os.path.join(root, fn)
                try:
                    if not os.path.islink(fp):
                        total += os.path.getsize(fp)
                except OSError:
                    pass
    except OSError:
        pass
    return total


def _hf_home(explicit: str | None = None) -> str:
    if explicit:
        return explicit
    return os.environ.get("HF_HOME") or os.path.expanduser("~/.cache/huggingface")


def disk_report(run_root: str | None = None, hf_home: str | None = None) -> dict[str, Any]:
    """Return disk capacity + directory breakdown in GiB (tailored for Colab's ~230 GB SSD)."""
    rroot = run_root or os.environ.get("QLOOP_RUN_ROOT") or "runs"
    probe_dir = rroot if os.path.exists(rroot) else os.getcwd()
    usage = shutil.disk_usage(probe_dir)
    hfh = _hf_home(hf_home)
    gib = float(2**30)
    runs_b = _dir_bytes(rroot)
    hub_b = _dir_bytes(os.path.join(hfh, "hub"))
    ds_b = _dir_bytes(os.path.join(hfh, "datasets"))
    xet_b = _dir_bytes(os.path.join(hfh, "xet")) + _dir_bytes(os.path.join(hfh, "upload"))
    return {
        "mount_path": os.path.abspath(probe_dir),
        "total_gib": round(usage.total / gib, 2),
        "used_gib": round(usage.used / gib, 2),
        "free_gib": round(usage.free / gib, 2),
        "runs_gib": round(runs_b / gib, 3),
        "hf_hub_weights_gib": round(hub_b / gib, 3),
        "hf_datasets_cache_gib": round(ds_b / gib, 3),
        "hf_staging_cache_gib": round(xet_b / gib, 3),
    }


def reclaim_disk_space(
    run_root: str | None = None,
    *,
    keep_stages: tuple[str, ...] = (),
    strip_optim_stages: tuple[str, ...] = (),
    clean_hf_staging: bool = True,
    clean_pip_cache: bool = False,
    hf_home: str | None = None,
) -> dict[str, Any]:
    """Reclaim Colab disk space without touching cached base model weights (`$HF_HOME/hub`).

    * Checkpoint directories (`<run_root>/<stage>/checkpoints`) for stages NOT in `keep_stages`
      are removed (small logs, `metrics.jsonl`, `probe.json`, and `esft_stats*.json` stay intact).
    * Stages in `strip_optim_stages` keep `trainable.pt` + `state.json` (for `train.init_from`
      and `distill.teacher_ckpt`), while their `optim.pt` / `rng.pt` and non-pointer step folders
      are removed.
    * Transient Hugging Face upload/xet caches are wiped so multi-GB commit staging blobs never
      accumulate across stages on Colab's 230 GB SSD.
    """
    rroot = run_root or os.environ.get("QLOOP_RUN_ROOT") or "runs"
    before = disk_report(rroot, hf_home=hf_home)
    freed_bytes = 0
    removed: list[str] = []
    keep_set = set(keep_stages) | set(strip_optim_stages)

    if os.path.isdir(rroot):
        for entry in sorted(os.listdir(rroot)):
            sdir = os.path.join(rroot, entry)
            cdir = os.path.join(sdir, "checkpoints")
            if not os.path.isdir(cdir):
                continue
            if entry not in keep_set:
                freed_bytes += _dir_bytes(cdir)
                shutil.rmtree(cdir, ignore_errors=True)
                removed.append(cdir)
                continue
            # Keep only LATEST / PROMOTED step directories inside kept stages
            ptrs: set[str] = set()
            for pname in ("LATEST", "PROMOTED"):
                pp = os.path.join(cdir, pname)
                if os.path.isfile(pp):
                    try:
                        with open(pp, encoding="utf-8") as f:
                            ptrs.add(f.read().strip())
                    except OSError:
                        pass
            for sub in sorted(os.listdir(cdir)):
                sub_p = os.path.join(cdir, sub)
                if not os.path.isdir(sub_p):
                    continue
                if sub.startswith(".tmp_") or (sub.startswith("step_") and ptrs and sub not in ptrs):
                    freed_bytes += _dir_bytes(sub_p)
                    shutil.rmtree(sub_p, ignore_errors=True)
                    removed.append(sub_p)
                elif entry in strip_optim_stages:
                    for extra_fn in ("optim.pt", "rng.pt"):
                        ef = os.path.join(sub_p, extra_fn)
                        if os.path.isfile(ef):
                            freed_bytes += _dir_bytes(ef)
                            try:
                                os.remove(ef)
                                removed.append(ef)
                            except OSError:
                                pass

    if clean_hf_staging:
        hfh = _hf_home(hf_home)
        bases = {hfh}
        if not hf_home and not os.environ.get("HF_HOME"):
            bases.add(os.path.expanduser("~/.cache/huggingface"))
        for base in bases:
            for transient in ("xet", "upload"):
                tp = os.path.join(base, transient)
                try:
                    if os.path.isdir(tp):
                        freed_bytes += _dir_bytes(tp)
                        shutil.rmtree(tp, ignore_errors=True)
                        removed.append(tp)
                except OSError:
                    pass
        try:
            for tmp_glob in glob.glob(os.path.join(tempfile.gettempdir(), "tmp*hf*")):
                if os.path.exists(tmp_glob):
                    freed_bytes += _dir_bytes(tmp_glob)
                    shutil.rmtree(tmp_glob, ignore_errors=True)
                    removed.append(tmp_glob)
        except OSError:
            pass

    if clean_pip_cache:
        pip_c = os.path.expanduser("~/.cache/pip")
        try:
            if os.path.isdir(pip_c):
                freed_bytes += _dir_bytes(pip_c)
                shutil.rmtree(pip_c, ignore_errors=True)
                removed.append(pip_c)
        except OSError:
            pass

    after = disk_report(rroot, hf_home=hf_home)
    freed_gib = round(freed_bytes / float(2**30), 3)
    if removed:
        log.info(
            "disk cleanup: reclaimed %.2f GiB (%d items) | free %.1f GiB / %.1f GiB",
            freed_gib, len(removed), after["free_gib"], after["total_gib"],
        )
    return {
        "freed_gib": freed_gib,
        "before_free_gib": before["free_gib"],
        "after_free_gib": after["free_gib"],
        "total_gib": after["total_gib"],
        "removed": removed,
    }


# ============================================================================ manifest & model card

@dataclass
class HubManifest:
    schema_version: int = 1
    repo_id: str = DEFAULT_HF_REPO
    base_model: str = DEFAULT_BASE_MODEL
    architecture: dict[str, Any] = field(default_factory=lambda: {
        "physical_layers": 40,
        "virtual_layers": 64,
        "prelude_end": 8,
        "core_end": 32,
        "num_passes": 2,
        "linear_attention": "hssd",
        "per_pass_router": True,
        "kv_share": True,
        "routed_experts_per_layer": 256,
        "top_k_per_pass": 8,
    })
    latest_stage: str | None = None
    latest_step: int = 0
    latest_checkpoint_stage: str | None = None
    updated_at: str = field(default_factory=_utc_now)
    stages: dict[str, dict[str, Any]] = field(default_factory=dict)
    eval_summary: dict[str, Any] = field(default_factory=dict)
    history: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "repo_id": self.repo_id,
            "base_model": self.base_model,
            "architecture": self.architecture,
            "latest_stage": self.latest_stage,
            "latest_step": self.latest_step,
            "latest_checkpoint_stage": self.latest_checkpoint_stage,
            "updated_at": self.updated_at,
            "stages": self.stages,
            "eval_summary": self.eval_summary,
            "history": self.history[-100:],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, default=str)

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None, repo_id: str = DEFAULT_HF_REPO) -> HubManifest:
        if not raw:
            return cls(repo_id=repo_id)
        m = cls(
            schema_version=int(raw.get("schema_version", 1)),
            repo_id=str(raw.get("repo_id") or repo_id),
            base_model=str(raw.get("base_model") or DEFAULT_BASE_MODEL),
            latest_stage=raw.get("latest_stage"),
            latest_step=int(raw.get("latest_step") or 0),
            latest_checkpoint_stage=raw.get("latest_checkpoint_stage"),
            updated_at=str(raw.get("updated_at") or _utc_now()),
            stages=dict(raw.get("stages") or {}),
            eval_summary=dict(raw.get("eval_summary") or {}),
            history=list(raw.get("history") or []),
        )
        if isinstance(raw.get("architecture"), dict):
            m.architecture.update(raw["architecture"])
        return m

    def is_completed(self, stage_name: str) -> bool:
        return (self.stages.get(stage_name) or {}).get("status") == "completed"

    def stage_step(self, stage_name: str) -> int:
        return int((self.stages.get(stage_name) or {}).get("step") or 0)

    def next_incomplete_stage(self, order: tuple[str, ...] = PIPELINE_ORDER) -> str | None:
        for st in order:
            if not self.is_completed(st):
                return st
        return None


def _fmt_num(val: Any, digits: int = 4) -> str:
    if isinstance(val, (int, float)):
        return f"{val:.{digits}f}"
    return "-"


def render_model_card(manifest: HubManifest) -> str:
    """Render a complete Hugging Face `README.md` model card from live pipeline state."""
    arch = manifest.architecture
    v_layers = arch.get("virtual_layers", 64)
    p_layers = arch.get("physical_layers", 40)
    pre = arch.get("prelude_end", 8)
    cend = arch.get("core_end", 32)
    passes = arch.get("num_passes", 2)
    raw_lin = arch.get("linear_attention")
    lin_attn = str(raw_lin).strip().lower() if raw_lin else "hssd"
    if lin_attn == "gdn2":
        lin_tags = ["  - gated-deltanet-2"]
        lin_bullets = [
            "- **NVIDIA Gated DeltaNet-2 (`GDN-2`, arXiv:2605.22791) Upgrade**: All 30 Gated DeltaNet "
            "linear-attention layers are upgraded with decoupled channel-wise erase gates (`b_t`), channel-wise "
            "write gates (`w_t`), and channel-wise decay (`g_t <= 0`) via chunkwise WY Tensor-Core "
            "parallelization while starting bit-identical at step 0.",
            "- **Hybrid State Isolation + MoR Recursive KV Sharing**: Independent `(layer, pass)` recurrent/conv "
            "slots for the core GDN-2 layers, paired with pass-1 KV + D-LSA bucket sharing on the core Gated Full "
            "Attention layers (zero KV memory multiplication at 1M context).",
            "- **1M Context YaRN + Distilled Latent Sparse Attention (`D-LSA`)**: Non-rotary key head indexing "
            "with batched Ward clustering for sub-quadratic 1M-token retrieval.",
        ]
    elif lin_attn == "stock":
        is_gdn_dlsa = any(s in manifest.stages for s in _V3_GDN_STAGE_SET) or arch.get("sparse_attention") == "d-lsa"
        lin_tags = ["  - gated-deltanet", "  - d-lsa"] if is_gdn_dlsa else ["  - gated-deltanet"]
        lead_bullet = (
            "- **Native Qwen3.6 Gated DeltaNet Linear Attention + 88L Loop + D-LSA**: Keeps all 30 native "
            "Qwen3.6 Gated DeltaNet linear-attention layers intact (zero HSSD/GDN-2 overhead) while scaling "
            "virtual depth to 88L and full-attention context to 1M with `D-LSA`."
            if is_gdn_dlsa
            else "- **Fast Stock Qwen3.6 Gated DeltaNet Linear Attention (`-> HSSD v3.2`)**: Heals the recurrent ETD "
            "loop directly over native Qwen3.6 linear-attention layers with zero adapter overhead prior to "
            "`HSSD v3.2` distillation."
        )
        lin_bullets = [
            lead_bullet,
            "- **Hybrid State Isolation + MoR Recursive KV Sharing**: Independent `(layer, pass)` recurrent/conv "
            "slots for the core linear-attention layers, paired with 1.0x pass-1 FP8 KV + per-pass `D-LSA` bucket "
            "sharing on Gated Full Attention layers (zero KV memory multiplication at 1M context).",
            "- **1M Context YaRN & Distilled Latent Sparse Attention (`D-LSA`)**: Position-free NoPE head "
            "indexing with active-slot compacted Ward clustering for sub-quadratic 1M-token retrieval.",
        ]
    else:
        lin_tags = ["  - hssd", "  - d-lsa"]
        lin_bullets = [
            "- **Horizon-Split Surprise-Gated Delta Memory (`HSSD v3.2`)**: Upgrades all 30 linear-attention "
            "layers with a strictly disjoint local SWA + multi-timescale `M=8` Top-2 routed page bank "
            "(`write_mode=\"full\"`, learnable SWA recency slope, low-rank 2D phase rotation, and chunkwise WY "
            "Tensor-Core parallelization).",
            "- **Hybrid State Isolation + MoR Recursive KV Sharing**: Independent `(layer, pass)` compact "
            "lagged-replay HSSD state caches across recurrent core passes, paired with 1.0x pass-1 FP8 KV + "
            "per-pass `D-LSA` bucket sharing on Gated Full Attention layers (zero KV memory multiplication at 1M "
            "context).",
            "- **1M Context YaRN & Distilled Latent Sparse Attention (`D-LSA`)**: Position-free NoPE head "
            "indexing with active-slot compacted Ward clustering and hybrid sparse+KL calibration for "
            "sub-quadratic 1M-token retrieval.",
        ]
    lines = [
        "---",
        "library_name: transformers",
        "license: apache-2.0",
        f"base_model: {manifest.base_model}",
        "tags:",
        "  - qwen3.6",
        "  - moe",
        "  - looped-transformer",
        *lin_tags,
        "  - dynamic-expert-routing",
        "  - mhc",
        "  - latent-sparse-attention",
        "  - grpo",
        "---",
        "",
        f"# {manifest.repo_id.split('/')[-1]}",
        "",
        f"**{v_layers}-Virtual-Layer Encode-Think-Decode Looped MoE Transformer** retrofitted onto "
        f"[`{manifest.base_model}`](https://huggingface.co/{manifest.base_model}) (`{p_layers}` physical layers, "
        "`35B` total parameters, `256` routed experts/layer).",
        "",
        "## Key Architectural Upgrades",
        f"- **Encode-Think-Decode (`{v_layers}L` Virtual Depth)**: Prelude `[0, {pre})` (`x1`) + "
        f"Recurrent Core `[{pre}, {cend})` (`x{passes}`) + Coda `[{cend}, {p_layers})` (`x1`).",
        "- **Cross-Pass Dynamic Expert Re-Routing (Beyond 3B Active Params)**: Learnable per-pass router adapters "
        "(`PassRouterAdapters`) and cross-pass expert repulsion allow Pass 2+ to wake up distinct Top-8 experts from "
        "the frozen 32.2B MoE pool at each recurrent pass (expanding unique consulted parameters from ~3.0B up to "
        "~4.5B-5.9B per token without adding base weight memory).",
        "- **Manifold-Constrained Hyper-Connection (`mHC`) Bridge**: Doubly-stochastic Birkhoff-Sinkhorn mixing with "
        "prelude anchor injection and zero-gated step-0 bit-exact identity.",
        *lin_bullets,
        "",
        "## Multi-Stage Training Progress (Synced via HF Commit History)",
        "",
        f"- **Latest Stage**: `{manifest.latest_stage or 'initializing'}` (step `{manifest.latest_step}`)",
        f"- **Last Updated**: `{manifest.updated_at}`",
        "",
        "| Stage | Status | Step | LM Loss | Depth Gain (`L_d1 - L_d2`) | Pass-2 Reroute Rate | Updated (UTC) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :--- |",
    ]
    order = resolve_pipeline_order(manifest.stages)
    for st in order:
        info = manifest.stages.get(st) or {}
        status = info.get("status", "pending")
        step = info.get("step", 0)
        tgt = info.get("target_steps")
        step_str = f"{step}/{tgt}" if tgt else str(step)
        met = info.get("metrics") or {}
        lm = _fmt_num(met.get("loss/lm", met.get("eval/loss_d2")))
        dg = _fmt_num(met.get("eval/depth_gain"))
        rr = _fmt_num(met.get("loop/reroute_rate"), 3)
        upd = info.get("updated_at", "-")
        lines.append(f"| `{st}` | **{status}** | `{step_str}` | `{lm}` | `{dg}` | `{rr}` | `{upd}` |")

    if manifest.eval_summary:
        bench_map = (
            manifest.eval_summary["benchmarks"]
            if isinstance(manifest.eval_summary.get("benchmarks"), dict)
            else manifest.eval_summary
        )
        lines += [
            "",
            "## Frontier Benchmark Depth-Sweep Scorecard",
            "",
            "| Benchmark | Depth 1 (`40L` Stock) | Depth 2 (`64L` Looped) | Delta (`d2 - d1`) |",
            "| :--- | :---: | :---: | :---: |",
        ]
        for bname, bdata in sorted(bench_map.items()):
            if not isinstance(bdata, dict):
                continue
            depths = bdata.get("depths") or {}
            d1 = (depths.get("1") or depths.get(1) or {}).get("mean_score")
            d2 = (depths.get("2") or depths.get(2) or {}).get("mean_score")
            delta = (d2 - d1) if isinstance(d1, (int, float)) and isinstance(d2, (int, float)) else None
            lines.append(f"| `{bname}` | `{_fmt_num(d1)}` | `{_fmt_num(d2)}` | `{_fmt_num(delta)}` |")

    lines += [
        "",
        "## Quick-Start Python Loading",
        "```python",
        "from qloop.hub import load_qloop_from_hub",
        "from qloop.eval.generate import Generator",
        "",
        f'stack = load_qloop_from_hub("{manifest.repo_id}")',
        "gen = Generator(stack.model, stack.tokenizer, loop_controller=stack.loop)",
        'with gen.depth(2):  # 64 virtual layers with cross-pass dynamic expert routing',
        '    print(gen.generate(["Solve over Z: x^3 + y^3 + z^3 = 42."], max_new_tokens=2048)[0].text)',
        "```",
        "",
    ]
    return "\n".join(lines)


# ============================================================================ Hub client

class HubSyncManager:
    """Crash-safe Hugging Face Hub commit synchroniser for stage checkpoints and manifests.

    Accepts an optional `api` object (implementing `create_repo`, `create_commit`, `hf_hub_download`
    or `download_file`, `list_repo_files`, `list_repo_commits`) so offline unit tests can exercise
    every code path without touching the network.
    """

    def __init__(
        self,
        repo_id: str = DEFAULT_HF_REPO,
        *,
        token: str | None = None,
        private: bool = False,
        squash_on_quota: bool = True,
        api: Any = None,
    ):
        self.repo_id = repo_id.strip()
        self.private = bool(private)
        self.squash_on_quota = bool(squash_on_quota)
        self.token = resolve_hf_token(token)
        self._api = api
        self._ensured = False

    @classmethod
    def from_sync_cfg(cls, sync_cfg: dict[str, Any] | None, api: Any = None) -> HubSyncManager | None:
        cfg = dict(sync_cfg or {})
        if cfg.get("enabled", True) is False:
            return None
        repo = cfg.get("hf_repo") if "hf_repo" in cfg else os.environ.get("QLOOP_HF_REPO")
        if not repo:
            return None
        return cls(
            repo_id=str(repo),
            token=cfg.get("token"),
            private=bool(cfg.get("private", False)),
            squash_on_quota=bool(cfg.get("squash_on_quota", True)),
            api=api,
        )

    @property
    def api(self) -> Any:
        if self._api is None:
            from huggingface_hub import HfApi

            self._api = HfApi(token=self.token)
        return self._api

    def ensure_repo(self) -> bool:
        if self._ensured:
            return True
        try:
            self.api.create_repo(
                repo_id=self.repo_id,
                repo_type="model",
                private=self.private,
                exist_ok=True,
            )
            self._ensured = True
            return True
        except Exception as e:
            log.warning("[hub] create_repo(%s, private=%s) failed: %r", self.repo_id, self.private, e)
            return False

    # ------------------------------------------------------------------ low-level I/O
    def list_files(self) -> list[str]:
        try:
            return list(self.api.list_repo_files(repo_id=self.repo_id, repo_type="model"))
        except Exception:
            return []

    def recent_commits(self, limit: int = 15) -> list[dict[str, Any]]:
        try:
            commits = self.api.list_repo_commits(repo_id=self.repo_id, repo_type="model")
            out = []
            for c in list(commits)[:limit]:
                out.append({
                    "commit_id": getattr(c, "commit_id", ""),
                    "title": getattr(c, "title", ""),
                    "created_at": str(getattr(c, "created_at", "")),
                })
            return out
        except Exception as e:
            log.debug("[hub] list_repo_commits failed: %r", e)
            return []

    def download_file(self, path_in_repo: str, dst_path: str) -> bool:
        os.makedirs(os.path.dirname(os.path.abspath(dst_path)), exist_ok=True)
        if self._api is not None and hasattr(self._api, "download_file"):
            return bool(self._api.download_file(self.repo_id, path_in_repo, dst_path))
        try:
            from huggingface_hub import hf_hub_download

            with tempfile.TemporaryDirectory(prefix="qloop_hf_dl_") as tmp_dl:
                cached = hf_hub_download(
                    repo_id=self.repo_id,
                    filename=path_in_repo,
                    repo_type="model",
                    token=self.token,
                    local_dir=tmp_dl,
                )
                shutil.copy2(cached, dst_path)
            return True
        except Exception as e:
            log.debug("[hub] download %s failed: %r", path_in_repo, e)
            return False

    def _commit_files(self, additions: list[tuple[str, str | bytes]], commit_message: str) -> str | None:
        """Atomically commit `[(path_in_repo, local_path_or_bytes), ...]` to `self.repo_id`."""
        if not self.ensure_repo():
            return None
        if hasattr(self.api, "commit_bundle"):
            return str(self.api.commit_bundle(self.repo_id, additions, commit_message))
        try:
            from huggingface_hub import CommitOperationAdd

            ops = [
                CommitOperationAdd(path_in_repo=repo_p, path_or_fileobj=src)
                for repo_p, src in additions
            ]
            info = self.api.create_commit(
                repo_id=self.repo_id,
                repo_type="model",
                operations=ops,
                commit_message=commit_message,
            )
            return str(getattr(info, "oid", None) or getattr(info, "commit_url", "committed"))
        except Exception as exc:
            msg = str(exc).lower()
            if self.squash_on_quota and ("storage" in msg or "quota" in msg or "413" in msg):
                try:
                    log.warning("[hub] storage limit hit on %s; squashing superseded history and retrying",
                                self.repo_id)
                    self.api.super_squash_history(repo_id=self.repo_id, repo_type="model")
                    from huggingface_hub import CommitOperationAdd

                    ops = [CommitOperationAdd(path_in_repo=rp, path_or_fileobj=s) for rp, s in additions]
                    info = self.api.create_commit(
                        repo_id=self.repo_id, repo_type="model", operations=ops, commit_message=commit_message
                    )
                    return str(getattr(info, "oid", None) or "committed")
                except Exception as exc2:
                    log.warning("[hub] retry after squash failed: %r", exc2)
                    return None
            log.warning("[hub] commit to %s failed: %r", self.repo_id, exc)
            return None

    # ------------------------------------------------------------------ manifest
    def fetch_manifest(self) -> HubManifest:
        with tempfile.TemporaryDirectory(prefix="qloop_manifest_") as td:
            dst = os.path.join(td, MANIFEST_FILE)
            if self.download_file(MANIFEST_FILE, dst) and os.path.isfile(dst):
                try:
                    with open(dst, encoding="utf-8") as f:
                        return HubManifest.from_dict(json.load(f), repo_id=self.repo_id)
                except Exception as e:
                    log.warning("[hub] malformed remote manifest.json (%r); starting clean manifest", e)
        return HubManifest(repo_id=self.repo_id)

    # ------------------------------------------------------------------ push operations
    def push_checkpoint(
        self,
        stage_name: str,
        ckpt_dir: str,
        *,
        completed: bool = False,
        promoted: bool = False,
        target_steps: int | None = None,
        include_optim: bool = False,
        metrics: dict[str, Any] | None = None,
        config_dict: dict[str, Any] | None = None,
        update_root_bundle: bool = True,
    ) -> str | None:
        """Commit a stage checkpoint directory + updated `manifest.json` + `README.md` to HF Hub."""
        state_path = os.path.join(ckpt_dir, "state.json")
        train_path = os.path.join(ckpt_dir, "trainable.pt")
        if not (os.path.isfile(state_path) and os.path.isfile(train_path)):
            log.warning("[hub] cannot push %s: missing state.json or trainable.pt", ckpt_dir)
            return None
        with open(state_path, encoding="utf-8") as f:
            st = json.load(f)
        step = int(st.get("step") or 0)
        stage_id = stage_name or str(st.get("stage") or "unknown")
        slot = "promoted" if promoted else "latest"
        prefix = f"stages/{stage_id}/{slot}"

        additions: list[tuple[str, str | bytes]] = [
            (f"{prefix}/trainable.pt", train_path),
            (f"{prefix}/state.json", state_path),
        ]
        rng_p = os.path.join(ckpt_dir, "rng.pt")
        if os.path.isfile(rng_p):
            additions.append((f"{prefix}/rng.pt", rng_p))
        opt_p = os.path.join(ckpt_dir, "optim.pt")
        if include_optim and os.path.isfile(opt_p):
            additions.append((f"{prefix}/optim.pt", opt_p))

        # Keep latest checkpoint also mirrored at `latest` when promoting so uniform resume works
        if promoted:
            additions.append((f"stages/{stage_id}/latest/trainable.pt", train_path))
            additions.append((f"stages/{stage_id}/latest/state.json", state_path))

        manifest = self.fetch_manifest()
        prev_entry = dict(manifest.stages.get(stage_id) or {})
        status = "completed" if (completed or prev_entry.get("status") == "completed") else "in_progress"
        merged_metrics = dict(prev_entry.get("metrics") or {})
        if metrics:
            for k, v in metrics.items():
                if isinstance(v, (int, float, str, bool)):
                    merged_metrics[k] = v

        # Update architecture summary if stack metadata is present
        stack_desc = st.get("stack") if isinstance(st.get("stack"), dict) else {}
        loop_desc = stack_desc.get("loop") if isinstance(stack_desc.get("loop"), dict) else {}
        if loop_desc:
            p_end = int(loop_desc.get("prelude_end", 8))
            c_end = int(loop_desc.get("core_end", 32))
            n_pass = int(loop_desc.get("num_passes", 2))
            phys_l = int(manifest.architecture.get("physical_layers", 40))
            virt_l = p_end + (c_end - p_end) * n_pass + (phys_l - c_end)
            manifest.architecture.update({
                "virtual_layers": virt_l,
                "prelude_end": p_end,
                "core_end": c_end,
                "num_passes": n_pass,
                "per_pass_router": bool(loop_desc.get("per_pass_router", True)),
                "kv_share": bool(loop_desc.get("kv_share", True)),
            })
        hssd_desc = stack_desc.get("hssd") if isinstance(stack_desc.get("hssd"), dict) else None
        gdn2_desc = stack_desc.get("gdn2") if isinstance(stack_desc.get("gdn2"), dict) else None
        lsa_desc = stack_desc.get("lsa") if isinstance(stack_desc.get("lsa"), dict) else None
        if hssd_desc is not None:
            manifest.architecture["linear_attention"] = "hssd"
            manifest.architecture["hssd_rank"] = int(hssd_desc.get("rank", 64))
            manifest.architecture["hssd_pages"] = int(hssd_desc.get("num_pages", 8))
            manifest.architecture.pop("gdn2_rank", None)
            manifest.architecture.pop("gdn2_qed", None)
            manifest.architecture.pop("gdn2_neg_eigval_max_boost", None)
            manifest.architecture.pop("gdn2_decay_log_min", None)
            manifest.architecture.pop("gdn2_decay_log_max", None)
        elif gdn2_desc is not None:
            manifest.architecture["linear_attention"] = "gdn2"
            manifest.architecture["gdn2_rank"] = int(gdn2_desc.get("rank", 64))
            manifest.architecture["gdn2_qed"] = bool(gdn2_desc.get("qed_enabled", False))
            manifest.architecture["gdn2_neg_eigval_max_boost"] = float(gdn2_desc.get("neg_eigval_max_boost", 0.0))
            if gdn2_desc.get("decay_log_min") is not None:
                manifest.architecture["gdn2_decay_log_min"] = float(gdn2_desc["decay_log_min"])
            else:
                manifest.architecture.pop("gdn2_decay_log_min", None)
            if gdn2_desc.get("decay_log_max") is not None:
                manifest.architecture["gdn2_decay_log_max"] = float(gdn2_desc["decay_log_max"])
            else:
                manifest.architecture.pop("gdn2_decay_log_max", None)
            manifest.architecture.pop("hssd_rank", None)
            manifest.architecture.pop("hssd_pages", None)
        elif stack_desc:
            manifest.architecture["linear_attention"] = "stock"
            manifest.architecture.pop("hssd_rank", None)
            manifest.architecture.pop("hssd_pages", None)
            manifest.architecture.pop("gdn2_rank", None)
            manifest.architecture.pop("gdn2_qed", None)
            manifest.architecture.pop("gdn2_neg_eigval_max_boost", None)
            manifest.architecture.pop("gdn2_decay_log_min", None)
            manifest.architecture.pop("gdn2_decay_log_max", None)
        if lsa_desc is not None:
            manifest.architecture["sparse_attention"] = "d-lsa"

        now = _utc_now()
        entry = {
            **prev_entry,
            "status": status,
            "step": max(step, int(prev_entry.get("step") or 0)),
            "target_steps": target_steps or prev_entry.get("target_steps"),
            "path_in_repo": f"stages/{stage_id}/latest",
            "promoted_path": f"stages/{stage_id}/promoted" if promoted else prev_entry.get("promoted_path"),
            "has_optim": bool(include_optim and os.path.isfile(opt_p)),
            "n_params": st.get("n_params"),
            "metrics": merged_metrics,
            "updated_at": now,
        }
        manifest.stages[stage_id] = entry
        manifest.latest_stage = stage_id
        manifest.latest_step = step
        manifest.latest_checkpoint_stage = stage_id
        manifest.updated_at = now

        if update_root_bundle and (completed or promoted or manifest.latest_checkpoint_stage == stage_id):
            additions.append(("qloop_adapters.pt", train_path))
            qcfg = {
                "repo_id": self.repo_id,
                "base_model": manifest.base_model,
                "stage": stage_id,
                "step": step,
                "status": status,
                "stack": stack_desc,
                "config": config_dict or st.get("config") or {},
                "updated_at": now,
            }
            additions.append(("qloop_config.json", json.dumps(qcfg, indent=2, default=str).encode("utf-8")))

        tag = "completed" if completed else ("promoted" if promoted else "ckpt")
        loss_val = merged_metrics.get("loss/lm", merged_metrics.get("eval/loss_d2"))
        dg_val = merged_metrics.get("eval/depth_gain")
        extra_bits = []
        if isinstance(loss_val, (int, float)):
            extra_bits.append(f"loss={loss_val:.4f}")
        if isinstance(dg_val, (int, float)):
            extra_bits.append(f"depth_gain={dg_val:+.4f}")
        suffix = (" " + " ".join(extra_bits)) if extra_bits else ""
        commit_msg = f"[qloop:{stage_id}] step={step} ({tag}){suffix}"

        manifest.history.append({
            "ts": now,
            "stage": stage_id,
            "step": step,
            "status": tag,
            "commit_message": commit_msg,
            "metrics": merged_metrics,
        })
        additions.append((MANIFEST_FILE, manifest.to_json().encode("utf-8")))
        additions.append(("README.md", render_model_card(manifest).encode("utf-8")))

        oid = self._commit_files(additions, commit_msg)
        if oid:
            log.info("[hub] committed %s -> https://huggingface.co/%s (%s)", commit_msg, self.repo_id, oid[:12])
        return oid

    def push_artifacts(
        self,
        stage_name: str,
        files: dict[str, str],
        *,
        completed: bool = True,
        metrics: dict[str, Any] | None = None,
        eval_summary: dict[str, Any] | None = None,
    ) -> str | None:
        """Commit non-checkpoint stage outputs (probes, ESFT router scans, benchmark summaries)."""
        additions: list[tuple[str, str | bytes]] = []
        uploaded_paths: list[str] = []
        for repo_path, local_path in sorted(files.items()):
            if os.path.isfile(local_path):
                additions.append((repo_path, local_path))
                uploaded_paths.append(repo_path)
        if not additions and not eval_summary:
            return None

        manifest = self.fetch_manifest()
        now = _utc_now()
        prev = dict(manifest.stages.get(stage_name) or {})
        merged_artifacts = sorted(set(prev.get("artifacts") or []) | set(uploaded_paths))
        merged_metrics = dict(prev.get("metrics") or {})
        if metrics:
            merged_metrics.update(metrics)
        manifest.stages[stage_name] = {
            **prev,
            "status": "completed" if completed else prev.get("status", "in_progress"),
            "artifacts": merged_artifacts,
            "metrics": merged_metrics,
            "updated_at": now,
        }
        if eval_summary:
            manifest.eval_summary.update(eval_summary)
        manifest.latest_stage = stage_name
        manifest.updated_at = now
        commit_msg = f"[qloop:{stage_name}] artifacts ({len(uploaded_paths)} files)"
        manifest.history.append({
            "ts": now,
            "stage": stage_name,
            "step": manifest.stage_step(stage_name),
            "status": "completed" if completed else "artifacts",
            "commit_message": commit_msg,
        })
        additions.append((MANIFEST_FILE, manifest.to_json().encode("utf-8")))
        additions.append(("README.md", render_model_card(manifest).encode("utf-8")))
        oid = self._commit_files(additions, commit_msg)
        if oid:
            log.info("[hub] committed %s -> https://huggingface.co/%s", commit_msg, self.repo_id)
        return oid

    # ------------------------------------------------------------------ pull & auto-resume
    def pull_artifacts(self, run_root: str) -> list[str]:
        """Download any remote ESFT / probe JSON artifacts missing from local `run_root`."""
        pulled: list[str] = []
        remote_files = set(self.list_files())
        for _key, (repo_rel, local_rel) in ARTIFACT_PATHS.items():
            dst = os.path.join(run_root, local_rel)
            if os.path.isfile(dst):
                continue
            if remote_files and repo_rel not in remote_files:
                continue
            if self.download_file(repo_rel, dst):
                pulled.append(dst)
                log.info("[hub] pulled artifact hf://%s/%s -> %s", self.repo_id, repo_rel, dst)
        return pulled

    def _local_latest_step(self, stage_dir: str) -> int:
        ptr = os.path.join(stage_dir, "checkpoints", "LATEST")
        if not os.path.isfile(ptr):
            return -1
        try:
            with open(ptr, encoding="utf-8") as f:
                step_name = f.read().strip()
            st_file = os.path.join(stage_dir, "checkpoints", step_name, "state.json")
            with open(st_file, encoding="utf-8") as f:
                return int(json.load(f).get("step") or 0)
        except Exception:
            return -1

    def pull_stage_checkpoint(
        self,
        stage_name: str,
        run_root: str,
        *,
        prefer_promoted: bool = False,
        include_optim: bool = True,
        manifest: HubManifest | None = None,
    ) -> str | None:
        """Ensure `run_root/<stage_name>/checkpoints/LATEST` has at least the Hub's latest step."""
        manifest = manifest or self.fetch_manifest()
        info = manifest.stages.get(stage_name)
        if not info or not info.get("path_in_repo"):
            return None
        remote_step = int(info.get("step") or 0)
        sdir = os.path.join(run_root, stage_name)
        local_step = self._local_latest_step(sdir)
        if local_step >= remote_step and local_step >= 0:
            step_dir = os.path.join(sdir, "checkpoints", f"step_{local_step:07d}")
            if os.path.isdir(step_dir):
                log.info("[hub] local %s step %d >= remote step %d; skipping download",
                         stage_name, local_step, remote_step)
                return step_dir

        repo_prefix = (
            info["promoted_path"]
            if (prefer_promoted and info.get("promoted_path"))
            else info["path_in_repo"]
        )
        cdir = os.path.join(sdir, "checkpoints")
        os.makedirs(cdir, exist_ok=True)

        with tempfile.TemporaryDirectory(dir=cdir, prefix=".hub_pull_") as tmp:
            if not self.download_file(f"{repo_prefix}/state.json", os.path.join(tmp, "state.json")):
                return None
            if not self.download_file(f"{repo_prefix}/trainable.pt", os.path.join(tmp, "trainable.pt")):
                return None
            with open(os.path.join(tmp, "state.json"), encoding="utf-8") as f:
                st = json.load(f)
            actual_step = int(st.get("step") or remote_step)
            if include_optim:
                self.download_file(f"{repo_prefix}/optim.pt", os.path.join(tmp, "optim.pt"))
                self.download_file(f"{repo_prefix}/rng.pt", os.path.join(tmp, "rng.pt"))
            final = os.path.join(cdir, f"step_{actual_step:07d}")
            shutil.rmtree(final, ignore_errors=True)
            shutil.move(tmp, final)
            os.makedirs(tmp, exist_ok=True)  # let TemporaryDirectory cleanup cleanly

        for ptr_name in (("LATEST", "PROMOTED") if (prefer_promoted and info.get("promoted_path")) else ("LATEST",)):
            tmp_ptr = os.path.join(cdir, f".{ptr_name}.tmp")
            with open(tmp_ptr, "w", encoding="utf-8") as f:
                f.write(os.path.basename(final))
            os.replace(tmp_ptr, os.path.join(cdir, ptr_name))

        log.info("[hub] pulled %s checkpoint step %d from hf://%s/%s -> %s",
                 stage_name, actual_step, self.repo_id, repo_prefix, final)
        return final

    def prepare_stage_workspace(
        self,
        stage_name: str,
        cfg: dict[str, Any],
        run_root: str,
    ) -> dict[str, Any]:
        """Pre-stage hook: pull missing ESFT stats + resume/prereq/teacher checkpoints, and prune old disk."""
        manifest = self.fetch_manifest()
        pulled_artifacts = self.pull_artifacts(run_root)
        resumed = self.pull_stage_checkpoint(stage_name, run_root, include_optim=True, manifest=manifest)

        prereq_stage: str | None = None
        init_from = (cfg.get("train") or {}).get("init_from")
        if init_from:
            prereq_stage = os.path.basename(str(init_from).replace("\\", "/").rstrip("/"))
            if resumed is None and prereq_stage:
                # Only need trainable.pt + state.json from prerequisite stage (`include_optim=False`)
                self.pull_stage_checkpoint(
                    prereq_stage, run_root, prefer_promoted=True, include_optim=False, manifest=manifest
                )

        teacher_stage: str | None = None
        distill_cfg = cfg.get("distill") or {}
        if bool(distill_cfg.get("enabled")) and distill_cfg.get("teacher_ckpt"):
            cand_teacher = os.path.basename(str(distill_cfg["teacher_ckpt"]).replace("\\", "/").rstrip("/"))
            if cand_teacher and cand_teacher != stage_name:
                if cand_teacher in manifest.stages:
                    teacher_stage = cand_teacher
                    # SharedBackboneTeacher needs frozen teacher weights even when resuming mid-stage
                    self.pull_stage_checkpoint(
                        teacher_stage, run_root, prefer_promoted=True, include_optim=False, manifest=manifest
                    )
                elif os.path.isdir(os.path.join(run_root, cand_teacher, "checkpoints")):
                    teacher_stage = cand_teacher

        strip_stages = tuple(
            dict.fromkeys(
                s for s in (prereq_stage, teacher_stage) if s and s != stage_name
            )
        )
        cleanup = reclaim_disk_space(
            run_root,
            keep_stages=(stage_name,),
            strip_optim_stages=strip_stages,
            clean_hf_staging=True,
        )
        return {
            "manifest": manifest,
            "pulled_artifacts": pulled_artifacts,
            "resumed_checkpoint": resumed,
            "prereq_stage": prereq_stage,
            "teacher_stage": teacher_stage,
            "disk": cleanup,
        }


# ============================================================================ 1-call Hub loader

def load_qloop_from_hub(
    repo_id: str = DEFAULT_HF_REPO,
    *,
    stage_name: str | None = None,
    token: str | None = None,
    overrides: list[str] | None = None,
    enable_lsa_if_present: bool = True,
    strict: bool = False,
    api: Any = None,
):
    """Load the 64-layer Looped Qwen3.6-35B-A3B `Stack` directly from Hugging Face Hub in one call."""
    import torch

    from .config import Cfg, apply_overrides
    from .train.stack import build_stack

    mgr = HubSyncManager(repo_id=repo_id, token=token, api=api)
    with tempfile.TemporaryDirectory(prefix="qloop_load_") as td:
        cfg_p = os.path.join(td, "qloop_config.json")
        if not mgr.download_file("qloop_config.json", cfg_p):
            raise FileNotFoundError(f"qloop_config.json not found in hf://{repo_id}")
        with open(cfg_p, encoding="utf-8") as f:
            meta = json.load(f)

        wt_repo = f"stages/{stage_name}/latest/trainable.pt" if stage_name else "qloop_adapters.pt"
        wt_p = os.path.join(td, "adapters.pt")
        if not mgr.download_file(wt_repo, wt_p):
            raise FileNotFoundError(f"{wt_repo} not found in hf://{repo_id}")
        saved = torch.load(wt_p, map_location="cpu")
        stage_meta: dict[str, Any] = {}
        if stage_name:
            st_p = os.path.join(td, "stage_state.json")
            if mgr.download_file(f"stages/{stage_name}/latest/state.json", st_p):
                try:
                    with open(st_p, encoding="utf-8") as f:
                        stage_meta = json.load(f)
                except Exception:
                    stage_meta = {}

    from .model.d1_lora import D1LoRAConfig
    from .model.loop import LoopSpec

    same_root_stage = (not stage_name) or (not meta.get("stage")) or (meta.get("stage") == stage_name)
    if stage_name and stage_meta.get("stack"):
        smeta = dict(stage_meta["stack"])
    elif same_root_stage:
        smeta = dict(meta.get("stack") or {})
    else:
        smeta = {}
    raw_cfg = dict(meta.get("config") or {})
    if not raw_cfg.get("model"):
        raw_cfg["model"] = {"name": meta.get("base_model", DEFAULT_BASE_MODEL), "dtype": "bfloat16",
                            "device_map": "auto", "attn_implementation": "sdpa"}
    tcfg_load = dict(raw_cfg.get("train") or {})
    tcfg_load.update({
        "gradient_checkpointing": False,
        "allowed_groups": None,
        "strict_patterns": False,
        "sha256_frozen_experts": False,
        "trainable": [],
        "train_gdn2": False,
        "train_hssd": False,
        "train_lsa": False,
        "train_d1_lora": False,
        "train_loop_adapters": False,
        "train_expert_deltas": False,
    })
    raw_cfg["train"] = tcfg_load

    has_loop = (
        any("loop_adapters." in k for k in saved)
        or bool(smeta.get("loop"))
        or (same_root_stage and "loop" not in smeta and bool((raw_cfg.get("loop") or {}).get("enabled", False)))
    )
    if has_loop:
        known_loop = {f.name for f in fields(LoopSpec)}
        prior_loop = {k: v for k, v in dict(smeta.get("loop") or {}).items() if k in known_loop}
        lpcfg = {k: v for k, v in dict(raw_cfg.get("loop") or {}).items() if k in known_loop}
        lpcfg.update(prior_loop)
        lpcfg["enabled"] = True
        raw_cfg["loop"] = lpcfg
    else:
        raw_cfg["loop"] = {"enabled": False}

    # At inference we fold any `expert_deltas` directly into float base weights via `restore_checkpoint`
    # (zero hook overhead during generation).
    raw_cfg["experts"] = {"enabled": False}
    if any("hssd_adapter." in k for k in saved):
        prior_hssd = dict(smeta.get("hssd") or {})
        hcfg = {**dict(raw_cfg.get("hssd") or {}), **prior_hssd}
        hcfg["enabled"] = True
        raw_cfg["hssd"] = hcfg
        raw_cfg["gdn2"] = {"enabled": False}
    elif any("gdn2_adapter." in k for k in saved):
        prior_gdn2 = dict(smeta.get("gdn2") or {})
        gcfg = {**dict(raw_cfg.get("gdn2") or {}), **prior_gdn2}
        for k, t in saved.items():
            if k.endswith("gdn2_adapter.down.weight") and t.ndim == 2:
                gcfg["rank"] = int(t.shape[0])
                break
        if "qed_enabled" not in prior_gdn2 and any(k.endswith("gdn2_adapter.qed_lambda_raw") for k in saved):
            gcfg["qed_enabled"] = True
        if "neg_eigval_max_boost" not in prior_gdn2 and any(
            k.endswith("gdn2_adapter.neg_gamma_raw") for k in saved
        ):
            gcfg["neg_eigval_max_boost"] = 0.5
        gcfg["enabled"] = True
        raw_cfg["gdn2"] = gcfg
        raw_cfg["hssd"] = {"enabled": False}
    else:
        raw_cfg["hssd"] = {"enabled": False}
        raw_cfg["gdn2"] = {"enabled": False}
    if enable_lsa_if_present and any("lsa_indexer." in k for k in saved):
        prior_lsa = dict(smeta.get("lsa") or {})
        lcfg = {**dict(raw_cfg.get("lsa") or {}), **prior_lsa}
        for k, t in saved.items():
            if k.endswith("lsa_indexer.wk") and t.ndim == 3:
                lcfg.setdefault("index_dim", int(t.shape[-1]))
                break
        lcfg["mode"] = "sparse"
        lcfg["enabled"] = True
        raw_cfg["lsa"] = lcfg
    else:
        raw_cfg["lsa"] = {"enabled": False}

    has_d1_lora = any("d1_lora." in k for k in saved) or bool(smeta.get("d1_lora"))
    if has_d1_lora:
        known_d1 = {f.name for f in fields(D1LoRAConfig)}
        prior_d1_raw = dict(smeta.get("d1_lora") or {})
        prior_d1 = {k: v for k, v in prior_d1_raw.items() if k in known_d1}
        d1cfg = {k: v for k, v in dict(raw_cfg.get("d1_lora") or {}).items() if k in known_d1}
        d1cfg.update(prior_d1)
        if prior_d1_raw.get("resolved_rank") is not None:
            d1cfg["rank"] = int(prior_d1_raw["resolved_rank"])
            d1cfg["target_params"] = None
        elif not prior_d1_raw:
            layer_idxs: list[int] = []
            inferred_rank: int | None = None
            for k, t in saved.items():
                if k.startswith("d1_lora.L"):
                    parts_k = k.split(".")
                    if len(parts_k) >= 4 and parts_k[1].startswith("L") and parts_k[1][1:].isdigit():
                        layer_idxs.append(int(parts_k[1][1:]))
                    if inferred_rank is None and k.endswith(".down") and getattr(t, "ndim", 0) == 2:
                        inferred_rank = int(t.shape[0])
            if inferred_rank is not None:
                d1cfg["rank"] = inferred_rank
                d1cfg["target_params"] = None
            if layer_idxs:
                d1cfg["layer_start"] = min(layer_idxs)
                d1cfg["layer_end"] = max(layer_idxs) + 1
        d1cfg["enabled"] = True
        raw_cfg["d1_lora"] = d1cfg
    elif "d1_lora" in raw_cfg:
        raw_cfg["d1_lora"] = {"enabled": False}

    if overrides:
        apply_overrides(raw_cfg, overrides)
    st = build_stack(Cfg(raw_cfg))
    scale = float((smeta.get("expert_delta_scale")) or 1.0)
    st.restore_checkpoint(saved, strict=strict, expert_delta_scale=scale, prior_stack_meta=smeta)
    for p in st.named_trainable.values():
        p.requires_grad_(False)
    st.model.eval()
    return st
