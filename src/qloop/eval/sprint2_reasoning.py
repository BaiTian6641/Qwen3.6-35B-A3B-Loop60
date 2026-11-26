"""Sprint 2 Gate B — Downstream Generative Reasoning Evaluation Harness.

Head-to-head downstream generative reasoning comparison across arms (e.g.
``stock_qwen36_yarn4x``, ``s2_v2_joint_bptt``, ``s2_2_d1_lora_control``,
``s1_loop_a6b``, ``s3_v2_long_1m_lsa``) on held-out reasoning benchmarks
(default: ``aime_2026``, ``gpqa_diamond``, ``arc_agi_2``).

Key guarantees:
  * **Arm isolation & VRAM hygiene:** Reuses ``qloop.eval.sprint2_audit._load_arm_stack``
    and cleanly unwraps/deletes every arm stack + runs ``gc.collect()`` +
    ``torch.cuda.empty_cache()`` between arms.
  * **Model-bound resumable caches (``QL60-010``):** Stores per-arm JSONL and
    ``summary.json`` caches under ``<out_dir>/arms/<arm_name>/`` via
    ``qloop.eval.runners.run_benchmark`` with explicit ``model_id``,
    ``checkpoint_path``, and ``allow_mixed_cache=False`` by default.
  * **Qwen3.6 thinking control & TF 5.24 safety:** Supports
    ``--enable-thinking auto|true|false`` (only forwarding
    ``chat_template_kwargs={"enable_thinking": bool}`` when explicitly set),
    ``--batch-size`` (default ``1``), and ``--padded-batching`` (default ``False``).
  * **Decode throughput probe:** Runs a tiny 1-prompt greedy probe
    (``max_new_tokens=32``) before scoring each ``(arm, depth)`` to record
    ``decode_tok_s``, ``wall_s``, and CUDA ``peak_vram_gib`` (``0.0`` on CPU).
  * **Honest failure & contamination reporting:** Captures gated-dataset auth
    or runtime errors as ``status="error"`` without silently substituting another
    benchmark unless explicit ``--fallback`` is supplied (and explicitly disclosed
    in both JSON and Markdown). Flags any benchmark whose contamination group is
    in ``--trained-groups`` (default ``kernelbench,arc_agi_2``) with
    ``trained_group_warning=True``.

Usage:
    python -m qloop.eval.sprint2_reasoning --smoke-cpu --out-dir /tmp/s2_reasoning
    python -m qloop.eval.sprint2_reasoning --arms-json arms.json --out-dir runs/s2_reasoning
    python -m qloop.eval.sprint2_reasoning --config configs/eval/suite.yaml --ckpt runs/s2_v2_joint_bptt --depths 1,3
"""
from __future__ import annotations

import argparse
import datetime as _dt
import gc
import json
import logging
import os
import re
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from qloop.data.tokenizer import ToyTokenizer
from qloop.eval.generate import Generator
from qloop.eval.registry import BenchmarkSpec, get_spec, load_items, missing_requirements
from qloop.eval.runners import _depth_label, run_benchmark
from qloop.eval.sprint2_audit import _BASE_ARM_SENTINELS, _load_arm_stack, _parse_int_list, build_smoke_cpu_stack
from qloop.model.d1_lora import D1LoRAConfig, install_d1_lora
from qloop.model.toy import ToyConfig, ToyForCausalLM

log = logging.getLogger("qloop")

SCHEMA_VERSION = "sprint2_reasoning_v1"
DEFAULT_BENCHMARKS: tuple[str, ...] = ("aime_2026", "gpqa_diamond", "arc_agi_2")
DEFAULT_TRAINED_GROUPS: tuple[str, ...] = ("kernelbench", "arc_agi_2")
DEFAULT_BASELINE_ARM = "stock_qwen36_yarn4x"
DEFAULT_MAX_NEW_TOKENS = 4096
DEFAULT_PROBE_MAX_NEW_TOKENS = 32
DEFAULT_PROBE_PROMPT: list[dict[str, str]] = [
    {"role": "user", "content": "Compute 17 * 23 and put the integer in \\boxed{}."}
]
_VALID_LSA_MODES = frozenset({"sparse", "dense", "distill"})
_SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9._-]+")


# ==============================================================================
# Config Dataclasses & Parsing Helpers
# ==============================================================================


@dataclass
class ArmConfig:
    """Declarative specification for one evaluation arm."""

    name: str
    config: str | None = None
    ckpt: str | None = None
    depths: list[int] | None = None
    loop_enabled: bool | None = None
    gdn2_enabled: bool | None = None
    lsa_enabled: bool | None = None
    hssd_enabled: bool | None = None
    d1_lora_enabled: bool | None = None
    lsa_mode: str | None = None
    base_only: bool = False
    hf_repo: str | None = None
    stage_name: str | None = None
    model_id: str | None = None
    overrides: list[str] = field(default_factory=list)

    def to_loader_dict(self) -> dict[str, Any]:
        """Convert to the arm dict schema consumed by ``sprint2_audit._load_arm_stack``."""
        raw_ckpt = self.ckpt.strip() if isinstance(self.ckpt, str) else self.ckpt
        is_sentinel = (
            isinstance(raw_ckpt, str)
            and raw_ckpt.lower() in _BASE_ARM_SENTINELS
            and not os.path.exists(raw_ckpt)
        )
        eff_base_only = bool(self.base_only or is_sentinel)
        eff_ckpt = None if (not raw_ckpt or is_sentinel) else raw_ckpt
        arm_overrides = list(self.overrides)
        if self.lsa_mode is not None and not any(o.startswith("lsa.mode=") for o in arm_overrides):
            arm_overrides.append(f"lsa.mode={self.lsa_mode}")
        out: dict[str, Any] = {
            "name": self.name,
            "config": self.config,
            "checkpoint": eff_ckpt,
            "base_only": eff_base_only,
            "overrides": arm_overrides,
        }
        if self.hf_repo:
            out["hf_repo"] = self.hf_repo
        if self.stage_name:
            out["stage_name"] = self.stage_name
        for key in ("loop_enabled", "gdn2_enabled", "lsa_enabled", "hssd_enabled", "d1_lora_enabled"):
            val = getattr(self, key)
            if val is not None:
                out[key] = bool(val)
        return out


@dataclass
class BenchmarkBudget:
    """Resolved per-benchmark evaluation budget."""

    name: str
    limit: int | None = None
    n_samples: int = 1
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS
    temperature: float | None = None
    top_p: float | None = None
    top_k: int = 0

    def effective_temperature(self) -> float:
        if self.temperature is not None:
            return float(self.temperature)
        return 0.0 if int(self.n_samples) == 1 else 0.6

    def effective_top_p(self) -> float:
        if self.top_p is not None:
            return float(self.top_p)
        return 1.0 if self.effective_temperature() <= 0.0 else 0.95


def _safe_arm_dirname(name: str) -> str:
    cleaned = _SAFE_NAME_RE.sub("_", str(name).strip()).strip("._")
    return cleaned or "arm"


def _concise_error(exc: BaseException, *, max_chars: int = 360) -> str:
    msg = " ".join(str(exc).split())
    text = f"{type(exc).__name__}: {msg}" if msg else type(exc).__name__
    if len(text) > max_chars:
        return text[: max_chars - 3] + "..."
    return text


def _utc_now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_benchmark_list(raw: str | Sequence[str] | None) -> list[str]:
    """Parse a comma-separated benchmark list while preserving order and deduplicating."""
    if raw is None:
        return list(DEFAULT_BENCHMARKS)
    tokens = [raw] if isinstance(raw, str) else list(raw)
    out: list[str] = []
    seen: set[str] = set()
    for tok in tokens:
        for part in str(tok).split(","):
            name = part.strip()
            if name and name not in seen:
                out.append(name)
                seen.add(name)
    if not out:
        raise ValueError("--benchmarks must name at least one benchmark")
    return out


def parse_trained_groups(raw: str | Sequence[str] | None) -> list[str]:
    """Parse comma-separated contamination groups trained on (empty string disables)."""
    if raw is None:
        return list(DEFAULT_TRAINED_GROUPS)
    if isinstance(raw, str):
        if not raw.strip():
            return []
        parts = [p.strip() for p in raw.split(",") if p.strip()]
    else:
        parts = [p.strip() for item in raw for p in str(item).split(",") if p.strip()]
    out: list[str] = []
    for p in parts:
        if p not in out:
            out.append(p)
    return out


def parse_per_benchmark_int(
    raw: str | int | None,
    *,
    default: int | None = None,
    min_value: int = 1,
    flag_name: str = "value",
) -> tuple[int | None, dict[str, int]]:
    """Parse ``"4096"`` or ``"aime_2026=4096,gpqa_diamond=2048"`` into ``(default_val, overrides)``."""
    if raw is None:
        return default, {}
    if isinstance(raw, int) and not isinstance(raw, bool):
        if raw < min_value:
            raise ValueError(f"{flag_name} must be >= {min_value}, got {raw}")
        return int(raw), {}
    text = str(raw).strip()
    if not text or text.lower() in ("none", "null"):
        return default, {}
    if "=" not in text and "," not in text:
        val = int(text)
        if val < min_value:
            raise ValueError(f"{flag_name} must be >= {min_value}, got {val}")
        return val, {}

    eff_default = default
    overrides: dict[str, int] = {}
    for entry in text.split(","):
        tok = entry.strip()
        if not tok:
            continue
        if "=" not in tok:
            raise ValueError(
                f"invalid {flag_name} entry {tok!r}; expected an integer or comma-separated benchmark=int pairs"
            )
        k, v = tok.split("=", 1)
        key = k.strip()
        v_str = v.strip()
        if not key:
            raise ValueError(f"empty benchmark key in {flag_name} entry {tok!r}")
        if v_str.lower() in ("none", "null"):
            if key in ("default", "*"):
                eff_default = None
            continue
        ival = int(v_str)
        if ival < min_value:
            raise ValueError(f"{flag_name}[{key}] must be >= {min_value}, got {ival}")
        if key in ("default", "*"):
            eff_default = ival
        else:
            overrides[key] = ival
    return eff_default, overrides


def parse_per_benchmark_float(
    raw: str | float | None,
    *,
    default: float | None = None,
    min_value: float = 0.0,
    flag_name: str = "temperature",
) -> tuple[float | None, dict[str, float]]:
    """Parse ``"0.6"`` or ``"aime_2026=0.6,gpqa_diamond=0.0"`` into ``(default_val, overrides)``."""
    if raw is None:
        return default, {}
    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
        fval = float(raw)
        if fval < min_value:
            raise ValueError(f"{flag_name} must be >= {min_value}, got {fval}")
        return fval, {}
    text = str(raw).strip()
    if not text or text.lower() in ("none", "null"):
        return default, {}
    if "=" not in text and "," not in text:
        fval = float(text)
        if fval < min_value:
            raise ValueError(f"{flag_name} must be >= {min_value}, got {fval}")
        return fval, {}

    eff_default = default
    overrides: dict[str, float] = {}
    for entry in text.split(","):
        tok = entry.strip()
        if not tok:
            continue
        if "=" not in tok:
            raise ValueError(
                f"invalid {flag_name} entry {tok!r}; expected a float or comma-separated benchmark=float pairs"
            )
        k, v = tok.split("=", 1)
        key = k.strip()
        v_str = v.strip()
        if not key:
            raise ValueError(f"empty benchmark key in {flag_name} entry {tok!r}")
        if v_str.lower() in ("none", "null"):
            if key in ("default", "*"):
                eff_default = None
            continue
        fval = float(v_str)
        if fval < min_value:
            raise ValueError(f"{flag_name}[{key}] must be >= {min_value}, got {fval}")
        if key in ("default", "*"):
            eff_default = fval
        else:
            overrides[key] = fval
    return eff_default, overrides


def parse_fallback_map(raw: str | Mapping[str, str] | None) -> dict[str, str]:
    """Parse explicit opt-in ``--fallback`` mappings (``requested=substitute,...``)."""
    if raw is None:
        return {}
    if isinstance(raw, Mapping):
        return {str(k).strip(): str(v).strip() for k, v in raw.items() if str(k).strip() and str(v).strip()}
    text = str(raw).strip()
    if not text:
        return {}
    out: dict[str, str] = {}
    for entry in text.split(","):
        tok = entry.strip()
        if not tok:
            continue
        if "=" not in tok:
            raise ValueError(f"invalid --fallback entry {tok!r}; expected requested_bench=fallback_bench")
        k, v = tok.split("=", 1)
        src, dst = k.strip(), v.strip()
        if not src or not dst:
            raise ValueError(f"invalid --fallback entry {tok!r}; both sides must be non-empty")
        out[src] = dst
    return out


def resolve_chat_template_kwargs(enable_thinking: str | bool | None) -> dict[str, Any] | None:
    """Resolve ``--enable-thinking auto|true|false`` into ``Generator.chat_template_kwargs``."""
    if enable_thinking is None:
        return None
    if isinstance(enable_thinking, bool):
        return {"enable_thinking": enable_thinking}
    norm = str(enable_thinking).strip().lower()
    if norm == "auto":
        return None
    if norm in ("true", "1", "yes", "on"):
        return {"enable_thinking": True}
    if norm in ("false", "0", "no", "off"):
        return {"enable_thinking": False}
    raise ValueError(f"--enable-thinking must be one of auto|true|false, got {enable_thinking!r}")


def _parse_optional_bool(val: Any, field_name: str) -> bool | None:
    if val is None:
        return None
    if isinstance(val, bool):
        return val
    if isinstance(val, str):
        low = val.strip().lower()
        if low in ("true", "1", "yes"):
            return True
        if low in ("false", "0", "no"):
            return False
    raise ValueError(f"arm field {field_name!r} must be bool or null, got {val!r}")


def _parse_arm_depths(raw: Any) -> list[int] | None:
    if raw is None:
        return None
    if isinstance(raw, int) and not isinstance(raw, bool):
        return [int(raw)]
    if isinstance(raw, str):
        vals = _parse_int_list(raw)
        return vals or None
    if isinstance(raw, Sequence):
        vals = [int(x) for x in raw]
        return vals or None
    raise ValueError(f"invalid arm depths: {raw!r}")


def parse_arms_json(
    arms_json: str | Path,
    *,
    default_config: str | None = None,
    default_depths: Sequence[int] | None = None,
) -> list[ArmConfig]:
    """Parse ``--arms-json <path_or_json>`` into a validated list of ``ArmConfig``."""
    raw_str = str(arms_json).strip()
    if not raw_str:
        raise ValueError("--arms-json cannot be empty")
    raw_json = Path(raw_str).read_text(encoding="utf-8") if os.path.isfile(raw_str) else raw_str
    payload = json.loads(raw_json)

    entries: list[dict[str, Any]] = []
    if isinstance(payload, list):
        for item in payload:
            if not isinstance(item, Mapping):
                raise ValueError(f"each arm entry must be a JSON object, got {type(item).__name__}")
            entries.append(dict(item))
    elif isinstance(payload, Mapping):
        if "name" in payload:
            entries.append(dict(payload))
        else:
            for k, v in payload.items():
                ent = dict(v) if isinstance(v, Mapping) else {"ckpt": v}
                ent.setdefault("name", str(k))
                entries.append(ent)
    else:
        raise ValueError(f"--arms-json must decode to a JSON list or object, got {type(payload).__name__}")

    arms: list[ArmConfig] = []
    seen_names: set[str] = set()
    fallback_depths = [int(d) for d in default_depths] if default_depths is not None else None
    for idx, ent in enumerate(entries):
        name = str(ent.get("name") or "").strip()
        if not name:
            raise ValueError(f"arm entry #{idx} is missing a non-empty 'name'")
        if name in seen_names:
            raise ValueError(f"duplicate arm name {name!r} in --arms-json")
        seen_names.add(name)

        ckpt = ent.get("ckpt") if "ckpt" in ent else ent.get("checkpoint")
        depths = _parse_arm_depths(ent.get("depths"))
        if depths is None:
            depths = list(fallback_depths) if fallback_depths is not None else None

        lsa_mode_raw = ent.get("lsa_mode")
        lsa_mode = str(lsa_mode_raw).strip() if lsa_mode_raw is not None and str(lsa_mode_raw).strip() else None
        if lsa_mode is not None and lsa_mode not in _VALID_LSA_MODES:
            raise ValueError(f"arm {name!r}: lsa_mode must be one of {sorted(_VALID_LSA_MODES)}, got {lsa_mode!r}")

        arms.append(
            ArmConfig(
                name=name,
                config=str(ent["config"]) if ent.get("config") else default_config,
                ckpt=str(ckpt) if ckpt is not None and str(ckpt).strip() else None,
                depths=depths,
                loop_enabled=_parse_optional_bool(ent.get("loop_enabled"), "loop_enabled"),
                gdn2_enabled=_parse_optional_bool(ent.get("gdn2_enabled"), "gdn2_enabled"),
                lsa_enabled=_parse_optional_bool(ent.get("lsa_enabled"), "lsa_enabled"),
                hssd_enabled=_parse_optional_bool(ent.get("hssd_enabled"), "hssd_enabled"),
                d1_lora_enabled=_parse_optional_bool(ent.get("d1_lora_enabled"), "d1_lora_enabled"),
                lsa_mode=lsa_mode,
                base_only=bool(ent.get("base_only", False)),
                hf_repo=str(ent["hf_repo"]) if ent.get("hf_repo") else None,
                stage_name=str(ent["stage_name"]) if ent.get("stage_name") else None,
                model_id=str(ent["model_id"]) if ent.get("model_id") else None,
                overrides=[str(x) for x in (ent.get("overrides") or [])],
            )
        )
    if not arms:
        raise ValueError("--arms-json decoded to 0 arms")
    return arms


def resolve_arm_configs(
    *,
    arms_json: str | Path | None = None,
    config: str | None = None,
    ckpt: str | None = None,
    depths: Sequence[int] | None = None,
    arm_name: str | None = None,
    base_only: bool = False,
) -> list[ArmConfig]:
    """Resolve CLI arm flags into a non-empty list of ``ArmConfig`` objects."""
    if arms_json is not None:
        return parse_arms_json(arms_json, default_config=config, default_depths=depths)
    if not config and not (ckpt and str(ckpt).strip().startswith("hf://")):
        raise ValueError("must provide --smoke-cpu, --arms-json <path_or_json>, or --config <yaml>")
    inferred_name = arm_name
    if not inferred_name:
        if ckpt and str(ckpt).strip().lower() not in _BASE_ARM_SENTINELS:
            inferred_name = Path(str(ckpt).rstrip("/")).name or "arm"
        elif config:
            inferred_name = Path(config).stem
        else:
            inferred_name = "arm"
    return [
        ArmConfig(
            name=str(inferred_name),
            config=config,
            ckpt=ckpt,
            depths=[int(d) for d in depths] if depths is not None else None,
            base_only=bool(base_only),
        )
    ]


def resolve_benchmark_budgets(
    benchmarks: Sequence[str],
    *,
    limits: str | int | None = None,
    n_samples: str | int | None = None,
    max_new_tokens: str | int | None = DEFAULT_MAX_NEW_TOKENS,
    temperature: str | float | None = None,
    top_p: float | None = None,
    top_k: int = 0,
) -> list[BenchmarkBudget]:
    """Combine global and per-benchmark CLI overrides into per-benchmark budgets."""
    lim_def, lim_map = parse_per_benchmark_int(limits, default=None, min_value=1, flag_name="--limits")
    ns_def, ns_map = parse_per_benchmark_int(n_samples, default=1, min_value=1, flag_name="--n-samples")
    mnt_def, mnt_map = parse_per_benchmark_int(
        max_new_tokens,
        default=DEFAULT_MAX_NEW_TOKENS,
        min_value=1,
        flag_name="--max-new-tokens",
    )
    temp_def, temp_map = parse_per_benchmark_float(
        temperature,
        default=None,
        min_value=0.0,
        flag_name="--temperature",
    )
    budgets: list[BenchmarkBudget] = []
    for name in benchmarks:
        budgets.append(
            BenchmarkBudget(
                name=name,
                limit=lim_map.get(name, lim_def),
                n_samples=int(ns_map.get(name, ns_def if ns_def is not None else 1)),
                max_new_tokens=int(mnt_map.get(name, mnt_def if mnt_def is not None else DEFAULT_MAX_NEW_TOKENS)),
                temperature=temp_map.get(name, temp_def),
                top_p=None if top_p is None else float(top_p),
                top_k=int(top_k),
            )
        )
    return budgets


# ==============================================================================
# Stack Teardown, Throughput Probe & Smoke Builders
# ==============================================================================


def _stack_attr(stack: Any, attr: str) -> Any:
    if isinstance(stack, Mapping):
        return stack.get(attr)
    return getattr(stack, attr, None)


def _teardown_arm_stack(stack: Any) -> None:
    """Cleanly unwrap/uninstall all retrofit hooks/wrappers before dropping an arm stack."""
    if stack is None:
        return
    for attr, method in (
        ("loop", "unwrap"),
        ("d1_lora", "remove"),
        ("deltas", "detach"),
        ("lsa", "uninstall"),
        ("gdn2", "uninstall"),
        ("hssd", "uninstall"),
    ):
        comp = _stack_attr(stack, attr)
        fn = getattr(comp, method, None) if comp is not None else None
        if callable(fn):
            try:
                fn()
            except Exception as exc:
                log.debug("[s2-reasoning] teardown %s.%s() ignored: %s", attr, method, exc)


def _describe_stack(stack: Any) -> dict[str, Any]:
    lsa = _stack_attr(stack, "lsa")
    loop = _stack_attr(stack, "loop")
    gdn2 = _stack_attr(stack, "gdn2")
    hssd = _stack_attr(stack, "hssd")
    d1_lora = _stack_attr(stack, "d1_lora")
    lsa_mode = None
    if lsa is not None:
        lsa_mode = getattr(getattr(lsa, "cfg", None), "mode", None)
    return {
        "has_loop": loop is not None,
        "max_loop_passes": int(loop.spec.num_passes) if loop is not None else 1,
        "has_gdn2": gdn2 is not None,
        "has_lsa": lsa is not None,
        "lsa_mode": lsa_mode,
        "has_hssd": hssd is not None,
        "has_d1_lora": d1_lora is not None,
    }


def _resolve_base_model_id(stack: Any, arm: ArmConfig) -> str:
    if arm.model_id:
        return arm.model_id
    model = _stack_attr(stack, "model")
    base = None
    for obj in (model, getattr(model, "config", None)):
        if obj is None:
            continue
        for attr in ("name_or_path", "_name_or_path", "model_type"):
            val = getattr(obj, attr, None)
            if isinstance(val, str) and val.strip():
                base = val.strip()
                break
        if base:
            break
    base = base or (type(model).__name__ if model is not None else "unknown")
    return f"{base}::{arm.name}"


def run_throughput_probe(
    generator: Generator,
    *,
    depth: int | None = None,
    max_new_tokens: int = DEFAULT_PROBE_MAX_NEW_TOKENS,
    prompt: list[dict[str, str]] | str | None = None,
    seed: int = 0,
) -> dict[str, Any]:
    """Run a tiny 1-prompt greedy decode probe recording ``decode_tok_s``, ``wall_s``, and ``peak_vram_gib``."""
    probe_prompt = DEFAULT_PROBE_PROMPT if prompt is None else prompt
    dev = getattr(generator, "device", torch.device("cpu"))
    is_cuda = torch.cuda.is_available() and getattr(dev, "type", "cpu") == "cuda"
    if is_cuda:
        torch.cuda.synchronize(dev)
        torch.cuda.reset_peak_memory_stats(dev)

    t0 = time.perf_counter()
    outs = generator.generate(
        [probe_prompt],
        max_new_tokens=int(max_new_tokens),
        temperature=0.0,
        top_p=1.0,
        n=1,
        seed=int(seed),
        depth=depth,
    )
    if is_cuda:
        torch.cuda.synchronize(dev)
        peak_vram_gib = float(torch.cuda.max_memory_allocated(dev)) / float(1024**3)
    else:
        peak_vram_gib = 0.0
    wall_s = max(time.perf_counter() - t0, 1e-9)

    sample = outs[0][0]
    completion_tokens = int(sample.completion_tokens)
    prompt_tokens = int(sample.prompt_tokens)
    decode_tok_s = float(completion_tokens) / wall_s if completion_tokens > 0 else 0.0
    return {
        "depth": _depth_label(depth),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "finish_reason": str(sample.finish_reason),
        "wall_s": round(wall_s, 4),
        "decode_tok_s": round(decode_tok_s, 2),
        "peak_vram_gib": round(peak_vram_gib, 4),
    }


def _build_smoke_d1_lora_stack(seed: int = 0) -> dict[str, Any]:
    """Build a deterministic pure-CPU ``ToyForCausalLM`` stack with ``D1LoRAManager`` installed."""
    tok = ToyTokenizer()
    torch.manual_seed(int(seed))
    cfg = ToyConfig(
        vocab_size=len(tok),
        hidden_size=16,
        num_hidden_layers=8,
        full_attention_interval=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        partial_rotary_factor=0.5,
        linear_num_heads=2,
        linear_head_dim=8,
        num_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=8,
        shared_expert_intermediate_size=8,
    )
    model = ToyForCausalLM(cfg).eval()
    d1_lora = install_d1_lora(
        model,
        D1LoRAConfig(enabled=True, layer_start=2, layer_end=6, rank=4, seed=int(seed)),
    )
    # Small deterministic perturbation so d1_lora control has distinct weights & run_identity vs stock
    gen = torch.Generator().manual_seed(int(seed) + 7777)
    with torch.no_grad():
        for pdict in d1_lora.adapters.values():
            for proj in pdict.values():
                if hasattr(proj, "up"):
                    proj.up.normal_(0.0, 0.05, generator=gen)
    return {
        "model": model,
        "tokenizer": tok,
        "loop": None,
        "gdn2": None,
        "lsa": None,
        "hssd": None,
        "d1_lora": d1_lora,
    }


def build_smoke_cpu_arms(seed: int = 42) -> tuple[list[ArmConfig], dict[str, Callable[[], tuple[Any, str | None]]]]:
    """Return ``(arm_configs, arm_factories)`` for fast CPU smoke (``stock d=1``, looped ``d=1,3``, ``d1_lora d=1``)."""
    arms = [
        ArmConfig(
            name="stock_toy_d1",
            depths=[1],
            loop_enabled=False,
            gdn2_enabled=False,
            lsa_enabled=False,
            hssd_enabled=False,
            d1_lora_enabled=False,
            base_only=True,
            model_id="toy::stock_toy_d1",
        ),
        ArmConfig(
            name="looped_gdn2_lsa_d3",
            depths=[1, 3],
            loop_enabled=True,
            gdn2_enabled=True,
            lsa_enabled=True,
            hssd_enabled=False,
            d1_lora_enabled=False,
            lsa_mode="sparse",
            model_id="toy::looped_gdn2_lsa_d3",
        ),
        ArmConfig(
            name="d1_lora_control_d1",
            depths=[1],
            loop_enabled=False,
            gdn2_enabled=False,
            lsa_enabled=False,
            hssd_enabled=False,
            d1_lora_enabled=True,
            model_id="toy::d1_lora_control_d1",
        ),
    ]
    factories: dict[str, Callable[[], tuple[Any, str | None]]] = {
        "stock_toy_d1": lambda: (build_smoke_cpu_stack(seed=seed, num_passes=1, install_retrofit=False), None),
        "looped_gdn2_lsa_d3": lambda: (
            build_smoke_cpu_stack(seed=seed, num_passes=3, trained_perturbation=True, install_retrofit=True),
            None,
        ),
        "d1_lora_control_d1": lambda: (_build_smoke_d1_lora_stack(seed=seed), None),
    }
    return arms, factories


def build_smoke_offline_items() -> dict[str, list[dict[str, Any]]]:
    """Synthetic offline items for ``aime_2026``, ``gpqa_diamond``, and ``arc_agi_2`` (``--smoke-cpu``)."""
    return {
        "aime_2026": [
            {"problem_idx": "smoke_aime_1", "problem": "Compute 6 * 7.", "answer": "42"},
            {"problem_idx": "smoke_aime_2", "problem": "Compute 10 + 15.", "answer": "25"},
        ],
        "gpqa_diamond": [
            {
                "Record ID": "smoke_gpqa_1",
                "Question": "Which subatomic particle carries a negative elementary electric charge?",
                "Correct Answer": "Electron",
                "Incorrect Answer 1": "Proton",
                "Incorrect Answer 2": "Neutron",
                "Incorrect Answer 3": "Photon",
            },
            {
                "Record ID": "smoke_gpqa_2",
                "Question": "What is the SI unit of electrical resistance?",
                "Correct Answer": "Ohm",
                "Incorrect Answer 1": "Tesla",
                "Incorrect Answer 2": "Henry",
                "Incorrect Answer 3": "Weber",
            },
        ],
        "arc_agi_2": [
            {
                "task_id": "smoke_arc_flip",
                "train": [{"input": [[0, 1], [1, 0]], "output": [[1, 0], [0, 1]]}],
                "test": [{"input": [[3, 0], [0, 3]], "output": [[0, 3], [3, 0]]}],
            },
            {
                "task_id": "smoke_arc_const",
                "train": [{"input": [[1, 1], [0, 0]], "output": [[0, 0], [1, 1]]}],
                "test": [{"input": [[2, 2], [0, 0]], "output": [[0, 0], [2, 2]]}],
            },
        ],
    }


# ==============================================================================
# Benchmark Item Loading (Resilient + Explicit Opt-In Fallback Only)
# ==============================================================================


@dataclass
class _LoadedBenchmark:
    requested_name: str
    executed_name: str
    spec: BenchmarkSpec | None
    budget: BenchmarkBudget
    status: str  # "ok" | "fallback" | "error"
    items: list[dict[str, Any]] | None = None
    error: str | None = None
    fallback_used: bool = False
    fallback_reason: str | None = None
    contamination_group: str = ""
    trained_group_warning: bool = False

    def summary_dict(self) -> dict[str, Any]:
        return {
            "requested_benchmark": self.requested_name,
            "executed_benchmark": self.executed_name,
            "status": self.status,
            "error": self.error,
            "fallback_used": bool(self.fallback_used),
            "fallback_reason": self.fallback_reason,
            "hf_id": None if self.spec is None else self.spec.hf_id,
            "config": None if self.spec is None else self.spec.config,
            "split": None if self.spec is None else self.spec.split,
            "contamination_group": self.contamination_group,
            "trained_group_warning": bool(self.trained_group_warning),
            "n_items": 0 if self.items is None else len(self.items),
            "limit": self.budget.limit,
            "n_samples": int(self.budget.n_samples),
            "max_new_tokens": int(self.budget.max_new_tokens),
            "temperature": self.budget.effective_temperature(),
            "top_p": self.budget.effective_top_p(),
            "top_k": int(self.budget.top_k),
        }


def _load_one_spec_items(
    name: str,
    budget: BenchmarkBudget,
    *,
    offline_items: Mapping[str, list[dict[str, Any]]] | None,
    capabilities: Sequence[str] | None,
) -> tuple[BenchmarkSpec, list[dict[str, Any]]]:
    spec = get_spec(name)
    if spec.scorer is None:
        raise RuntimeError(f"benchmark {name!r} has no local scorer (external harness={spec.harness!r})")
    missing = missing_requirements(spec, capabilities)
    if missing:
        raise RuntimeError(f"benchmark {name!r} requires unavailable capabilities {sorted(missing)}")
    off = None if offline_items is None else offline_items.get(name)
    items = load_items(spec, limit=budget.limit, offline_items=off)
    if not items:
        raise RuntimeError(f"benchmark {name!r} loaded 0 items (limit={budget.limit})")
    return spec, items


def _prepare_benchmarks(
    budgets: Sequence[BenchmarkBudget],
    *,
    trained_groups: Sequence[str],
    fallback_map: Mapping[str, str],
    fail_fast: bool,
    offline_items: Mapping[str, list[dict[str, Any]]] | None = None,
    capabilities: Sequence[str] | None = None,
) -> list[_LoadedBenchmark]:
    trained_set = set(trained_groups)
    loaded: list[_LoadedBenchmark] = []

    for budget in budgets:
        req = budget.name
        try:
            spec, items = _load_one_spec_items(req, budget, offline_items=offline_items, capabilities=capabilities)
            cgroup = spec.contamination_group
            loaded.append(
                _LoadedBenchmark(
                    requested_name=req,
                    executed_name=spec.name,
                    spec=spec,
                    budget=budget,
                    status="ok",
                    items=items,
                    contamination_group=cgroup,
                    trained_group_warning=bool(cgroup and cgroup in trained_set),
                )
            )
            continue
        except Exception as exc:
            err_msg = _concise_error(exc)
            log.error("[s2-reasoning] benchmark %s failed to load: %s", req, err_msg)
            fb_name = fallback_map.get(req)
            if fb_name:
                log.warning(
                    "[s2-reasoning] explicit --fallback active: substituting %s for %s due to: %s",
                    fb_name,
                    req,
                    err_msg,
                )
                try:
                    fb_spec, fb_items = _load_one_spec_items(
                        fb_name,
                        budget,
                        offline_items=offline_items,
                        capabilities=capabilities,
                    )
                    fb_cgroup = fb_spec.contamination_group
                    loaded.append(
                        _LoadedBenchmark(
                            requested_name=req,
                            executed_name=fb_spec.name,
                            spec=fb_spec,
                            budget=budget,
                            status="fallback",
                            items=fb_items,
                            error=None,
                            fallback_used=True,
                            fallback_reason=err_msg,
                            contamination_group=fb_cgroup,
                            trained_group_warning=bool(fb_cgroup and fb_cgroup in trained_set),
                        )
                    )
                    continue
                except Exception as fb_exc:
                    fb_err = _concise_error(fb_exc)
                    combined = f"{err_msg}; fallback {fb_name!r} also failed: {fb_err}"
                    if fail_fast:
                        raise RuntimeError(combined) from fb_exc
                    spec_meta = _safe_get_spec(req)
                    cgroup = "" if spec_meta is None else spec_meta.contamination_group
                    loaded.append(
                        _LoadedBenchmark(
                            requested_name=req,
                            executed_name=req,
                            spec=spec_meta,
                            budget=budget,
                            status="error",
                            error=combined,
                            contamination_group=cgroup,
                            trained_group_warning=bool(cgroup and cgroup in trained_set),
                        )
                    )
                    continue

            if fail_fast:
                raise
            spec_meta = _safe_get_spec(req)
            cgroup = "" if spec_meta is None else spec_meta.contamination_group
            loaded.append(
                _LoadedBenchmark(
                    requested_name=req,
                    executed_name=req,
                    spec=spec_meta,
                    budget=budget,
                    status="error",
                    error=err_msg,
                    contamination_group=cgroup,
                    trained_group_warning=bool(cgroup and cgroup in trained_set),
                )
            )
    return loaded


def _safe_get_spec(name: str) -> BenchmarkSpec | None:
    try:
        return get_spec(name)
    except KeyError:
        return None


# ==============================================================================
# Core Evaluation Orchestrator
# ==============================================================================


def _depth_sort_key(label: str) -> tuple[int, int, str]:
    if label == "native":
        return (0, 1, label)
    if label.isdigit():
        return (1, int(label), label)
    return (2, 0, label)


def _effective_depth_int(label: str) -> int:
    if label == "native":
        return 1
    if label.isdigit():
        return int(label)
    return 1


def _extract_maj_metric(metrics: Mapping[str, Any]) -> tuple[str | None, float | None]:
    maj_keys = [k for k in metrics if k.startswith("maj@")]
    if not maj_keys:
        return None, None

    def _k_num(k: str) -> int:
        tail = k.split("@", 1)[1]
        return int(tail) if tail.isdigit() else 0

    best_k = max(maj_keys, key=_k_num)
    val = metrics.get(best_k)
    return best_k, (None if val is None else float(val))


def _pick_baseline_depth(depths_map: Mapping[str, Any]) -> str | None:
    if not depths_map:
        return None
    if "native" in depths_map:
        return "native"
    if "1" in depths_map:
        return "1"
    return sorted(depths_map.keys(), key=_depth_sort_key)[0]


def _build_comparison_tables(
    prepared: Sequence[_LoadedBenchmark],
    arm_order: Sequence[str],
    arms_data: Mapping[str, Mapping[str, Any]],
    *,
    resolved_baseline: str,
) -> dict[str, list[dict[str, Any]]]:
    tables: dict[str, list[dict[str, Any]]] = {}
    for lb in prepared:
        bname = lb.requested_name
        rows: list[dict[str, Any]] = []

        # Lookup baseline mean_score for this benchmark (if available)
        base_score: float | None = None
        base_depth: str | None = None
        base_arm_rec = arms_data.get(resolved_baseline)
        if base_arm_rec is not None:
            base_b = (base_arm_rec.get("benchmarks") or {}).get(bname) or {}
            if base_b.get("status") in ("ok", "fallback"):
                b_depths = base_b.get("depths") or {}
                base_depth = _pick_baseline_depth(b_depths)
                if base_depth is not None and b_depths[base_depth].get("mean_score") is not None:
                    base_score = float(b_depths[base_depth]["mean_score"])

        for arm_name in arm_order:
            arm_rec = arms_data[arm_name]
            b_rec = (arm_rec.get("benchmarks") or {}).get(bname) or {}
            b_status = str(b_rec.get("status") or lb.status)
            probes = arm_rec.get("throughput_probes") or {}
            if b_status not in ("ok", "fallback") or not b_rec.get("depths"):
                probe0 = next(iter(probes.values()), {})
                rows.append(
                    {
                        "benchmark": bname,
                        "executed_benchmark": str(b_rec.get("executed_benchmark") or lb.executed_name),
                        "fallback_used": bool(b_rec.get("fallback_used", lb.fallback_used)),
                        "fallback_reason": b_rec.get("fallback_reason", lb.fallback_reason),
                        "arm": arm_name,
                        "depth": probe0.get("depth", "native"),
                        "effective_depth": _effective_depth_int(str(probe0.get("depth", "native"))),
                        "is_baseline": arm_name == resolved_baseline,
                        "baseline_arm": resolved_baseline,
                        "baseline_depth": base_depth,
                        "status": b_status,
                        "error": b_rec.get("error") or lb.error,
                        "n_items": 0,
                        "n_samples": int(lb.budget.n_samples),
                        "mean_score": None,
                        "pass@1": None,
                        "pass@2": None,
                        "maj@k": None,
                        "maj_k_label": None,
                        "delta_vs_baseline": None,
                        "mean_completion_tokens": None,
                        "frac_truncated": None,
                        "decode_tok_s": probe0.get("decode_tok_s"),
                        "wall_s": None,
                        "probe_wall_s": probe0.get("wall_s"),
                        "peak_vram_gib": probe0.get("peak_vram_gib"),
                        "contamination_group": lb.contamination_group,
                        "trained_group_warning": bool(lb.trained_group_warning),
                    }
                )
                continue

            depths_map: Mapping[str, Mapping[str, Any]] = b_rec["depths"]
            for dlabel in sorted(depths_map.keys(), key=_depth_sort_key):
                dm = depths_map[dlabel]
                mscore = dm.get("mean_score")
                mscore_f = None if mscore is None else float(mscore)
                is_base_row = arm_name == resolved_baseline and dlabel == base_depth
                delta = (
                    0.0
                    if is_base_row and mscore_f is not None
                    else (None if (mscore_f is None or base_score is None) else round(mscore_f - base_score, 6))
                )
                maj_label, maj_val = _extract_maj_metric(dm)
                rows.append(
                    {
                        "benchmark": bname,
                        "executed_benchmark": str(b_rec.get("executed_benchmark") or lb.executed_name),
                        "fallback_used": bool(b_rec.get("fallback_used", lb.fallback_used)),
                        "fallback_reason": b_rec.get("fallback_reason", lb.fallback_reason),
                        "arm": arm_name,
                        "depth": dlabel,
                        "effective_depth": _effective_depth_int(dlabel),
                        "is_baseline": bool(is_base_row),
                        "baseline_arm": resolved_baseline,
                        "baseline_depth": base_depth,
                        "status": b_status,
                        "error": None,
                        "n_items": int(dm.get("n_items", 0)),
                        "n_samples": int(dm.get("n_samples", lb.budget.n_samples)),
                        "mean_score": mscore_f,
                        "pass@1": None if dm.get("pass@1") is None else float(dm["pass@1"]),
                        "pass@2": None if dm.get("pass@2") is None else float(dm["pass@2"]),
                        "maj@k": maj_val,
                        "maj_k_label": maj_label,
                        "delta_vs_baseline": delta,
                        "mean_completion_tokens": (
                            None if dm.get("mean_completion_tokens") is None else float(dm["mean_completion_tokens"])
                        ),
                        "frac_truncated": None if dm.get("frac_truncated") is None else float(dm["frac_truncated"]),
                        "decode_tok_s": None if dm.get("decode_tok_s") is None else float(dm["decode_tok_s"]),
                        "wall_s": None if dm.get("wall_s") is None else float(dm["wall_s"]),
                        "probe_wall_s": None if dm.get("probe_wall_s") is None else float(dm["probe_wall_s"]),
                        "peak_vram_gib": None if dm.get("peak_vram_gib") is None else float(dm["peak_vram_gib"]),
                        "contamination_group": lb.contamination_group,
                        "trained_group_warning": bool(lb.trained_group_warning),
                    }
                )
        tables[bname] = rows
    return tables


def run_sprint2_reasoning(
    arms: Sequence[ArmConfig],
    *,
    benchmarks: Sequence[BenchmarkBudget],
    out_dir: str | Path,
    baseline_arm: str = DEFAULT_BASELINE_ARM,
    trained_groups: Sequence[str] = DEFAULT_TRAINED_GROUPS,
    enable_thinking: str = "auto",
    batch_size: int = 1,
    padded_batching: bool = False,
    seed: int = 42,
    allow_mixed_cache: bool = False,
    fail_fast: bool = False,
    fallback_map: Mapping[str, str] | None = None,
    probe_max_new_tokens: int = DEFAULT_PROBE_MAX_NEW_TOKENS,
    overrides: Sequence[str] = (),
    offline_items: Mapping[str, list[dict[str, Any]]] | None = None,
    capabilities: Sequence[str] | None = None,
    arm_factories: Mapping[str, Callable[[], tuple[Any, str | None]]] | None = None,
    mode: str = "eval",
) -> dict[str, Any]:
    """Run head-to-head generative reasoning evaluation across ``arms`` and write JSON/MD artifacts."""
    if not arms:
        raise ValueError("run_sprint2_reasoning requires at least one arm")
    t_start = time.perf_counter()
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    arms_root = out_path / "arms"
    arms_root.mkdir(parents=True, exist_ok=True)

    ctk = resolve_chat_template_kwargs(enable_thinking)
    eff_fallback = dict(fallback_map or {})
    eff_trained = list(trained_groups)
    arm_order = [a.name for a in arms]
    resolved_baseline = baseline_arm if baseline_arm in arm_order else arm_order[0]

    prepared = _prepare_benchmarks(
        benchmarks,
        trained_groups=eff_trained,
        fallback_map=eff_fallback,
        fail_fast=bool(fail_fast),
        offline_items=offline_items,
        capabilities=capabilities,
    )

    fallback_disclosures = [
        {
            "requested_benchmark": lb.requested_name,
            "executed_benchmark": lb.executed_name,
            "reason": lb.fallback_reason,
            "contamination_group": lb.contamination_group,
            "trained_group_warning": lb.trained_group_warning,
        }
        for lb in prepared
        if lb.fallback_used
    ]

    arms_data: dict[str, Any] = {}
    for arm in arms:
        arm_cache_dir = arms_root / _safe_arm_dirname(arm.name)
        arm_cache_dir.mkdir(parents=True, exist_ok=True)
        stack: Any = None
        gen: Generator | None = None
        resolved_ckpt: str | None = None
        try:
            if arm_factories is not None and arm.name in arm_factories:
                stack, resolved_ckpt = arm_factories[arm.name]()
            else:
                stack, resolved_ckpt = _load_arm_stack(arm.to_loader_dict(), overrides=overrides)

            lsa = _stack_attr(stack, "lsa")
            if lsa is not None and arm.lsa_mode is not None:
                lsa.set_mode(arm.lsa_mode)

            model = _stack_attr(stack, "model")
            tokenizer = _stack_attr(stack, "tokenizer")
            loop_ctl = _stack_attr(stack, "loop")
            if model is None or tokenizer is None:
                raise RuntimeError(f"arm {arm.name!r}: loaded stack missing model or tokenizer")
            if hasattr(model, "eval"):
                model.eval()

            gen = Generator(
                model,
                tokenizer,
                loop_controller=loop_ctl,
                batch_size=int(batch_size),
                padded_batching=bool(padded_batching),
                chat_template_kwargs=ctk,
            )
            eff_model_id = _resolve_base_model_id(stack, arm)
            stack_desc = _describe_stack(stack)

            # Determine effective depths for this arm
            if loop_ctl is None:
                eff_depths: list[int | None] = [None]
            elif arm.depths:
                eff_depths = [int(d) for d in arm.depths]
            else:
                eff_depths = [1, int(loop_ctl.spec.num_passes)] if int(loop_ctl.spec.num_passes) > 1 else [1]

            # 1. Tiny 1-prompt decode throughput probe before scoring each (arm, depth)
            throughput_probes: dict[str, dict[str, Any]] = {}
            for d in eff_depths:
                dlabel = _depth_label(d)
                throughput_probes[dlabel] = run_throughput_probe(
                    gen,
                    depth=d,
                    max_new_tokens=int(probe_max_new_tokens),
                    seed=int(seed),
                )

            # 2. Score each prepared benchmark with model-bound QL60-010 cache isolation
            arm_bench_results: dict[str, Any] = {}
            arm_run_identities: dict[str, Any] = {}
            dev = getattr(gen, "device", torch.device("cpu"))
            is_cuda = torch.cuda.is_available() and getattr(dev, "type", "cpu") == "cuda"

            for lb in prepared:
                bname = lb.requested_name
                if lb.status == "error" or lb.spec is None or lb.items is None:
                    arm_bench_results[bname] = {
                        "status": "error",
                        "requested_benchmark": bname,
                        "executed_benchmark": lb.executed_name,
                        "fallback_used": False,
                        "fallback_reason": None,
                        "error": lb.error,
                        "contamination_group": lb.contamination_group,
                        "trained_group_warning": bool(lb.trained_group_warning),
                        "run_identity": None,
                        "depths": {},
                    }
                    continue

                try:
                    if is_cuda:
                        torch.cuda.synchronize(dev)
                        torch.cuda.reset_peak_memory_stats(dev)
                    summ = run_benchmark(
                        lb.spec,
                        gen,
                        limit=lb.budget.limit,
                        depths=eff_depths,
                        n_samples=int(lb.budget.n_samples),
                        max_new_tokens=int(lb.budget.max_new_tokens),
                        out_dir=str(arm_cache_dir),
                        temperature=lb.budget.temperature,
                        top_p=lb.budget.top_p,
                        top_k=int(lb.budget.top_k),
                        seed=int(seed),
                        items=lb.items,
                        capabilities=capabilities,
                        allow_mixed_cache=bool(allow_mixed_cache),
                        model_id=eff_model_id,
                        checkpoint_path=resolved_ckpt,
                    )
                    if is_cuda:
                        torch.cuda.synchronize(dev)
                        eval_peak_vram_gib = float(torch.cuda.max_memory_allocated(dev)) / float(1024**3)
                    else:
                        eval_peak_vram_gib = 0.0

                    enriched_depths: dict[str, Any] = {}
                    for dlabel, dm in (summ.get("depths") or {}).items():
                        probe = throughput_probes.get(dlabel) or next(iter(throughput_probes.values()), {})
                        probe_vram = float(probe.get("peak_vram_gib", 0.0) or 0.0)
                        peak_vram = max(probe_vram, eval_peak_vram_gib) if is_cuda else 0.0
                        enriched_depths[dlabel] = {
                            **dict(dm),
                            "decode_tok_s": probe.get("decode_tok_s"),
                            "wall_s": dm.get("wall_s_this_run"),
                            "probe_wall_s": probe.get("wall_s"),
                            "peak_vram_gib": round(peak_vram, 4),
                        }

                    rid = summ.get("run_identity")
                    arm_run_identities[bname] = rid
                    arm_bench_results[bname] = {
                        "status": lb.status,
                        "requested_benchmark": bname,
                        "executed_benchmark": lb.executed_name,
                        "fallback_used": bool(lb.fallback_used),
                        "fallback_reason": lb.fallback_reason,
                        "error": None,
                        "contamination_group": lb.contamination_group,
                        "trained_group_warning": bool(lb.trained_group_warning),
                        "run_identity": rid,
                        "depths": enriched_depths,
                        "depth_curve": summ.get("depth_curve") or {},
                    }
                except Exception as exc:
                    err_msg = _concise_error(exc)
                    log.error("[s2-reasoning] arm=%s benchmark=%s failed: %s", arm.name, bname, err_msg)
                    if fail_fast:
                        raise
                    arm_bench_results[bname] = {
                        "status": "error",
                        "requested_benchmark": bname,
                        "executed_benchmark": lb.executed_name,
                        "fallback_used": bool(lb.fallback_used),
                        "fallback_reason": lb.fallback_reason,
                        "error": err_msg,
                        "contamination_group": lb.contamination_group,
                        "trained_group_warning": bool(lb.trained_group_warning),
                        "run_identity": None,
                        "depths": {},
                    }

            arms_data[arm.name] = {
                "name": arm.name,
                "config": arm.config,
                "ckpt": arm.ckpt,
                "resolved_ckpt": resolved_ckpt,
                "model_id": eff_model_id,
                "requested_depths": arm.depths,
                "effective_depths": [_depth_label(d) for d in eff_depths],
                "cache_dir": str(arm_cache_dir),
                "stack": stack_desc,
                "throughput_probes": throughput_probes,
                "run_identities": arm_run_identities,
                "benchmarks": arm_bench_results,
            }
        finally:
            _teardown_arm_stack(stack)
            del gen
            del stack
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    tables = _build_comparison_tables(prepared, arm_order, arms_data, resolved_baseline=resolved_baseline)
    elapsed_s = round(time.perf_counter() - t_start, 3)

    report: dict[str, Any] = {
        "schema": SCHEMA_VERSION,
        "schema_version": SCHEMA_VERSION,
        "mode": str(mode),
        "created_at_utc": _utc_now(),
        "baseline_arm": str(baseline_arm),
        "resolved_baseline_arm": resolved_baseline,
        "trained_groups": eff_trained,
        "enable_thinking": str(enable_thinking),
        "chat_template_kwargs": ctk or {},
        "batch_size": int(batch_size),
        "padded_batching": bool(padded_batching),
        "seed": int(seed),
        "allow_mixed_cache": bool(allow_mixed_cache),
        "fallback_map": eff_fallback,
        "fallback_disclosures": fallback_disclosures,
        "benchmarks": {lb.requested_name: lb.summary_dict() for lb in prepared},
        "arms": arms_data,
        "tables": tables,
        "metadata": {
            "out_dir": str(out_path),
            "elapsed_s": elapsed_s,
            "num_arms": len(arms),
            "num_benchmarks": len(prepared),
        },
    }
    write_reasoning_artifacts(report, out_path)
    return report


# ==============================================================================
# Markdown & Artifact Writers
# ==============================================================================


def _fmt_pct(val: float | None, *, signed: bool = False) -> str:
    if val is None:
        return "N/A"
    return f"{val * 100.0:+.2f}%" if signed else f"{val * 100.0:.2f}%"


def _fmt_num(val: float | None, digits: int = 2) -> str:
    if val is None:
        return "N/A"
    return f"{val:.{digits}f}"


def _fmt_pass2_maj(row: Mapping[str, Any]) -> str:
    bits: list[str] = []
    if row.get("pass@2") is not None:
        bits.append(f"p@2={_fmt_pct(row['pass@2'])}")
    if row.get("maj@k") is not None:
        lbl = row.get("maj_k_label") or "maj@k"
        bits.append(f"{lbl}={_fmt_pct(row['maj@k'])}")
    return " / ".join(bits) if bits else "N/A"


def format_reasoning_markdown(report: Mapping[str, Any]) -> str:
    """Format a ``sprint2_reasoning_v1`` report into side-by-side GitHub Markdown tables."""
    lines: list[str] = [
        "# Sprint 2 Gate B — Generative Reasoning Evaluation Report",
        "",
        f"- **Schema:** `{report.get('schema', SCHEMA_VERSION)}`",
        f"- **Mode:** `{report.get('mode', 'eval')}`",
        f"- **Baseline Arm:** `{report.get('resolved_baseline_arm')}`"
        + (
            f" (requested `{report.get('baseline_arm')}`)"
            if report.get("baseline_arm") != report.get("resolved_baseline_arm")
            else ""
        ),
        (
            f"- **Thinking Mode:** `{report.get('enable_thinking')}` "
            f"(`chat_template_kwargs={report.get('chat_template_kwargs')}`)"
        ),
        f"- **Batching:** `batch_size={report.get('batch_size')}`, `padded_batching={report.get('padded_batching')}`",
        f"- **Trained Contamination Groups:** `{', '.join(report.get('trained_groups') or []) or 'none'}`",
        f"- **Seed:** `{report.get('seed')}` | **Elapsed:** `{(report.get('metadata') or {}).get('elapsed_s', 0.0)}s`",
        "",
    ]

    disclosures = report.get("fallback_disclosures") or []
    if disclosures:
        lines.append("> [!WARNING]")
        lines.append("> **Explicit `--fallback` Benchmark Substitutions Used:**")
        for d in disclosures:
            lines.append(
                f"> - Requested `{d['requested_benchmark']}` -> Executed `{d['executed_benchmark']}` "
                f"(reason: `{d['reason']}`)"
            )
        lines.append("")

    flagged_benches = [
        (name, meta.get("contamination_group"))
        for name, meta in (report.get("benchmarks") or {}).items()
        if meta.get("trained_group_warning")
    ]
    if flagged_benches:
        desc = ", ".join(f"`{n}` (group=`{g}`)" for n, g in flagged_benches)
        lines.append(
            f"> [!NOTE]\n> **Trained-Group Contamination Disclosure:** {desc} overlap(s) "
            "`--trained-groups` (e.g., `s2_v2_joint_bptt` trains on `arc_agi2_train` split). "
            "Treat flagged rows as in-family rather than strictly unseen held-out."
        )
        lines.append("")

    # Section 1: Throughput & VRAM probes
    lines.append("## 1. Decode Throughput & Peak VRAM Probe (`max_new_tokens=32`, greedy)")
    lines.append("")
    lines.append(
        "| Arm | Depth | Loop | GDN-2 | D-LSA | d1-LoRA | Decode tok/s | Probe Wall (s) | Peak VRAM (GiB) |"
    )
    lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
    for arm_name, arm_rec in (report.get("arms") or {}).items():
        st = arm_rec.get("stack") or {}
        lsa_lbl = str(st.get("lsa_mode") or "on") if st.get("has_lsa") else "off"
        probes = arm_rec.get("throughput_probes") or {}
        for dlabel in sorted(probes.keys(), key=_depth_sort_key):
            pr = probes[dlabel]
            lines.append(
                f"| `{arm_name}` | `{dlabel}` | {'on' if st.get('has_loop') else 'off'} | "
                f"{'on' if st.get('has_gdn2') else 'off'} | `{lsa_lbl}` | "
                f"{'on' if st.get('has_d1_lora') else 'off'} | `{_fmt_num(pr.get('decode_tok_s'), 2)}` | "
                f"`{_fmt_num(pr.get('wall_s'), 3)}` | `{_fmt_num(pr.get('peak_vram_gib'), 3)}` |"
            )
    lines.append("")

    # Section 2: Per-benchmark head-to-head tables
    lines.append("## 2. Head-to-Head Reasoning Benchmarks vs Baseline")
    lines.append("")
    benchmarks_meta = report.get("benchmarks") or {}
    tables = report.get("tables") or {}
    for bname, bmeta in benchmarks_meta.items():
        flag_badge = (
            f"TRAINED_GROUP_WARNING (`{bmeta.get('contamination_group')}`)"
            if bmeta.get("trained_group_warning")
            else f"held-out (`{bmeta.get('contamination_group') or 'none'}`)"
        )
        header = f"### `{bname}`"
        if bmeta.get("fallback_used"):
            header += f" [FALLBACK -> `{bmeta.get('executed_benchmark')}`]"
        lines.append(header)
        lines.append("")
        lines.append(
            f"- **Status:** `{bmeta.get('status')}` | **Items:** `{bmeta.get('n_items')}` | "
            f"**n_samples:** `{bmeta.get('n_samples')}` | **max_new_tokens:** `{bmeta.get('max_new_tokens')}` | "
            f"**temp:** `{bmeta.get('temperature')}` | **Contamination:** {flag_badge}"
        )
        if bmeta.get("fallback_used"):
            lines.append(
                f"- **Fallback Disclosure:** Requested `{bname}` failed (`{bmeta.get('fallback_reason')}`); "
                f"executed explicit fallback `{bmeta.get('executed_benchmark')}`."
            )
        if bmeta.get("status") == "error":
            lines.append(f"- **Error:** `{bmeta.get('error')}` (no silent substitution performed)")
            lines.append("")
            continue
        lines.append("")
        lines.append(
            "| Arm | Depth | Mean Score | Δ vs Baseline | pass@1 | pass@2 / maj@k | Mean Tok | Trunc % | "
            "Decode tok/s | Wall (s) | Peak VRAM (GiB) | Contam Flag |"
        )
        lines.append(
            "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |"
        )
        for row in tables.get(bname) or []:
            cflag = (
                f"WARN:{row.get('contamination_group')}"
                if row.get("trained_group_warning")
                else "held-out"
            )
            if row.get("status") == "error":
                lines.append(
                    f"| `{row['arm']}` | `{row['depth']}` | **ERROR** | N/A | N/A | N/A | N/A | N/A | "
                    f"`{_fmt_num(row.get('decode_tok_s'), 2)}` | N/A | "
                    f"`{_fmt_num(row.get('peak_vram_gib'), 3)}` | `{cflag}` |"
                )
                continue
            delta_str = (
                "BASELINE"
                if row.get("is_baseline")
                else _fmt_pct(row.get("delta_vs_baseline"), signed=True)
            )
            lines.append(
                f"| `{row['arm']}` | `{row['depth']}` | `{_fmt_pct(row.get('mean_score'))}` | "
                f"`{delta_str}` | `{_fmt_pct(row.get('pass@1'))}` | `{_fmt_pass2_maj(row)}` | "
                f"`{_fmt_num(row.get('mean_completion_tokens'), 1)}` | `{_fmt_pct(row.get('frac_truncated'))}` | "
                f"`{_fmt_num(row.get('decode_tok_s'), 2)}` | `{_fmt_num(row.get('wall_s'), 2)}` | "
                f"`{_fmt_num(row.get('peak_vram_gib'), 3)}` | `{cflag}` |"
            )
        lines.append("")

    # Section 3: QL60-010 cache identity audit
    lines.append("## 3. Model-Bound Cache Identity Audit (`QL60-010`)")
    lines.append("")
    lines.append("| Arm | Benchmark | `model_id` | `checkpoint_sha` | `config_hash` | `dataset_rev` |")
    lines.append("| :--- | :--- | :--- | :---: | :---: | :--- |")
    for arm_name, arm_rec in (report.get("arms") or {}).items():
        rids = arm_rec.get("run_identities") or {}
        if not rids:
            lines.append(f"| `{arm_name}` | — | `{arm_rec.get('model_id')}` | — | — | — |")
            continue
        for bname, rid in rids.items():
            if not isinstance(rid, Mapping):
                continue
            csha = str(rid.get("checkpoint_sha") or "")
            if len(csha) > 28:
                csha = csha[:28] + "…"
            lines.append(
                f"| `{arm_name}` | `{bname}` | `{rid.get('model_id')}` | `{csha}` | "
                f"`{rid.get('config_hash')}` | `{rid.get('dataset_rev')}` |"
            )
    lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def write_reasoning_artifacts(report: Mapping[str, Any], out_dir: str | Path) -> tuple[Path, Path]:
    """Write ``<out_dir>/sprint2_reasoning.json`` and ``<out_dir>/sprint2_reasoning.md``."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    json_path = out_path / "sprint2_reasoning.json"
    md_path = out_path / "sprint2_reasoning.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=False, default=str) + "\n", encoding="utf-8")
    md_path.write_text(format_reasoning_markdown(report), encoding="utf-8")
    return json_path, md_path


# ==============================================================================
# CLI Entrypoint
# ==============================================================================


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="python -m qloop.eval.sprint2_reasoning",
        description="Sprint 2 Gate B — Head-to-Head Downstream Generative Reasoning Evaluation Harness.",
    )
    ap.add_argument(
        "--smoke-cpu",
        action="store_true",
        help="Run fast (<5s) CPU verification on toy stock/looped/d1_lora arms with offline synthetic benchmarks.",
    )
    ap.add_argument(
        "--arms-json",
        type=str,
        default=None,
        help=(
            "Path to JSON file or inline JSON list of arm specs [{name, config, ckpt, depths, "
            "loop_enabled, gdn2_enabled, lsa_enabled, hssd_enabled, d1_lora_enabled, lsa_mode}]."
        ),
    )
    ap.add_argument(
        "--config",
        type=str,
        default=None,
        help="YAML config path (single-arm CLI or default for --arms-json).",
    )
    ap.add_argument("--ckpt", type=str, default=None, help="Checkpoint path or hf:// repo spec (single-arm CLI).")
    ap.add_argument("--arm-name", type=str, default=None, help="Arm label for single-arm CLI.")
    ap.add_argument(
        "--base-only",
        action="store_true",
        help="Disable all retrofits on single-arm CLI (stock baseline).",
    )
    ap.add_argument(
        "--depths",
        nargs="+",
        default=None,
        help="Loop depths to evaluate (e.g. --depths 1,3 or --depths 1 2 3).",
    )
    ap.add_argument(
        "--baseline-arm",
        type=str,
        default=DEFAULT_BASELINE_ARM,
        help=f"Arm name used as Δ-vs-baseline reference (default: {DEFAULT_BASELINE_ARM}).",
    )
    ap.add_argument(
        "--benchmarks",
        type=str,
        default=",".join(DEFAULT_BENCHMARKS),
        help="Comma-separated registered benchmark names (default: aime_2026,gpqa_diamond,arc_agi_2).",
    )
    ap.add_argument(
        "--limits",
        type=str,
        default=None,
        help="Item limit as integer or per-benchmark map (e.g. aime_2026=30,gpqa_diamond=50,arc_agi_2=20).",
    )
    ap.add_argument(
        "--n-samples",
        type=str,
        default=None,
        help="Samples per item as integer or per-benchmark map (e.g. aime_2026=1,gpqa_diamond=1,arc_agi_2=2).",
    )
    ap.add_argument(
        "--max-new-tokens",
        type=str,
        default=None,
        help="Generation token cap as integer or per-benchmark map (default: 4096; 8 in --smoke-cpu).",
    )
    ap.add_argument(
        "--temperature",
        type=str,
        default=None,
        help="Sampling temperature override as float or per-benchmark map (default: 0.0 for n=1, 0.6 for n>1).",
    )
    ap.add_argument("--top-p", type=float, default=None, help="Nucleus top-p override.")
    ap.add_argument("--top-k", type=int, default=0, help="Top-k sampling cutoff (0 disables).")
    ap.add_argument("--seed", type=int, default=42, help="Deterministic evaluation seed (default: 42).")
    ap.add_argument(
        "--trained-groups",
        type=str,
        default=",".join(DEFAULT_TRAINED_GROUPS),
        help="Comma-separated contamination groups trained on (default: kernelbench,arc_agi_2).",
    )
    ap.add_argument(
        "--enable-thinking",
        type=str,
        choices=("auto", "true", "false"),
        default="auto",
        help="Qwen3.6 thinking control: auto (omit kwarg), true, or false (default: auto).",
    )
    ap.add_argument("--batch-size", type=int, default=1, help="Generator batch size (default: 1).")
    ap.add_argument(
        "--padded-batching",
        action="store_true",
        default=False,
        help="Enable left-padded multi-prompt batching (default: False for TF 5.24 DeltaNet safety).",
    )
    ap.add_argument(
        "--probe-max-new-tokens",
        type=int,
        default=DEFAULT_PROBE_MAX_NEW_TOKENS,
        help="Tokens generated per (arm, depth) decode throughput probe (default: 32).",
    )
    ap.add_argument(
        "--out-dir",
        type=str,
        default="runs/sprint2_reasoning",
        help="Output directory for arms/ caches, sprint2_reasoning.json, and sprint2_reasoning.md.",
    )
    ap.add_argument(
        "--allow-mixed-cache",
        action="store_true",
        default=False,
        help="Allow resuming an existing arm cache with mismatched QL60-010 identity (default: False).",
    )
    ap.add_argument(
        "--fail-fast",
        action="store_true",
        default=False,
        help="Abort immediately on the first benchmark load or evaluation error instead of recording status='error'.",
    )
    ap.add_argument(
        "--fallback",
        type=str,
        default=None,
        help="Explicit opt-in benchmark substitution map on load error (e.g. gpqa_diamond=supergpqa). "
        "Disclosed prominently in JSON and Markdown; NEVER applied silently.",
    )
    ap.add_argument("-s", "--set", dest="overrides", action="append", default=[], help="Config key=value override.")
    return ap


def main(argv: Sequence[str] | None = None) -> int:
    ap = build_arg_parser()
    args = ap.parse_args(argv)

    bench_names = parse_benchmark_list(args.benchmarks)
    trained_groups = parse_trained_groups(args.trained_groups)
    fallback_map = parse_fallback_map(args.fallback)
    cli_depths = _parse_int_list(args.depths) if args.depths else None

    if args.smoke_cpu:
        arms, factories = build_smoke_cpu_arms(seed=int(args.seed))
        offline_items = build_smoke_offline_items()
        eff_n_samples = args.n_samples if args.n_samples is not None else "aime_2026=1,gpqa_diamond=1,arc_agi_2=2"
        eff_max_new = args.max_new_tokens if args.max_new_tokens is not None else "4"
        budgets = resolve_benchmark_budgets(
            bench_names,
            limits=args.limits,
            n_samples=eff_n_samples,
            max_new_tokens=eff_max_new,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=int(args.top_k),
        )
        eff_baseline = args.baseline_arm if args.baseline_arm in {a.name for a in arms} else "stock_toy_d1"
        report = run_sprint2_reasoning(
            arms,
            benchmarks=budgets,
            out_dir=args.out_dir,
            baseline_arm=eff_baseline,
            trained_groups=trained_groups,
            enable_thinking=args.enable_thinking,
            batch_size=int(args.batch_size),
            padded_batching=bool(args.padded_batching),
            seed=int(args.seed),
            allow_mixed_cache=bool(args.allow_mixed_cache),
            fail_fast=bool(args.fail_fast),
            fallback_map=fallback_map,
            probe_max_new_tokens=int(args.probe_max_new_tokens),
            overrides=args.overrides,
            offline_items=offline_items,
            arm_factories=factories,
            mode="smoke_cpu",
        )
    else:
        try:
            arms = resolve_arm_configs(
                arms_json=args.arms_json,
                config=args.config,
                ckpt=args.ckpt,
                depths=cli_depths,
                arm_name=args.arm_name,
                base_only=bool(args.base_only),
            )
        except ValueError as exc:
            ap.error(str(exc))
        budgets = resolve_benchmark_budgets(
            bench_names,
            limits=args.limits,
            n_samples=args.n_samples,
            max_new_tokens=args.max_new_tokens if args.max_new_tokens is not None else DEFAULT_MAX_NEW_TOKENS,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=int(args.top_k),
        )
        report = run_sprint2_reasoning(
            arms,
            benchmarks=budgets,
            out_dir=args.out_dir,
            baseline_arm=args.baseline_arm,
            trained_groups=trained_groups,
            enable_thinking=args.enable_thinking,
            batch_size=int(args.batch_size),
            padded_batching=bool(args.padded_batching),
            seed=int(args.seed),
            allow_mixed_cache=bool(args.allow_mixed_cache),
            fail_fast=bool(args.fail_fast),
            fallback_map=fallback_map,
            probe_max_new_tokens=int(args.probe_max_new_tokens),
            overrides=args.overrides,
            mode="eval",
        )

    print(format_reasoning_markdown(report))
    return 0


__all__ = [
    "DEFAULT_BASELINE_ARM",
    "DEFAULT_BENCHMARKS",
    "DEFAULT_MAX_NEW_TOKENS",
    "DEFAULT_PROBE_MAX_NEW_TOKENS",
    "DEFAULT_TRAINED_GROUPS",
    "SCHEMA_VERSION",
    "ArmConfig",
    "BenchmarkBudget",
    "build_arg_parser",
    "build_smoke_cpu_arms",
    "build_smoke_offline_items",
    "format_reasoning_markdown",
    "main",
    "parse_arms_json",
    "parse_benchmark_list",
    "parse_fallback_map",
    "parse_per_benchmark_float",
    "parse_per_benchmark_int",
    "parse_trained_groups",
    "resolve_arm_configs",
    "resolve_benchmark_budgets",
    "resolve_chat_template_kwargs",
    "run_sprint2_reasoning",
    "run_throughput_probe",
    "write_reasoning_artifacts",
]


if __name__ == "__main__":
    sys.exit(main())
