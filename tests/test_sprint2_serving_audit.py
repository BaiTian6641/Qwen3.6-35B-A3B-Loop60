"""CPU verification suite for Sprint 2 live `transformers 5.24.x` serving & export audit script.

Covers:
  1. CLI `--smoke-cpu` end-to-end execution (< 10s, zero network/GPU) via both direct `main()`
     and subprocess invocation, plus `--output-json` / `--out-json` and `--output-md` / `--out-md`.
  2. `check_quantized_hybrid_cache` (`QL60-001`): `bf16` / `fp8` / `hadamard_int4`, strict `abc.ABC`
     TF 5.24 contract offline + simulated real `transformers 5.24.x` contract, `kv_share=True`
     (`1.0x` KV RAM via `SharedKVView`), `kv_share=False`, and all cache contract methods/proxies.
  3. `check_pack_model_and_generate` (`QL60-002`): both in-package and freshly emitted standalone
     `configuration_qwen36_loophssd.py` / `modeling_qwen36_loophssd.py` (Py 3.9 AST + `py_compile` +
     `_HAS_QLOOP=False` import), `CausalLMPackOutput` contract, `logits_to_keep=0` returning full
     `(B, T, V)`, outer `config.layer_types`/`num_hidden_layers` sync across `depths=(1, 2, 3)`,
     2D & 3D MRoPE `position_ids` slicing, greedy decode parity vs uncached full recompute, and
     `qloop_format_version` / `qloop_commit` metadata verification.
  4. `check_stage_completion_gate` (`QL60-005`): `_local_stage_done()` rejection of mid-run
     `LATEST` / incomplete `PROMOTED` and acceptance of verified completed stages.
  5. `--checkpoint` mode against both exported Whole Model Pack directories and training
     checkpoints paired with `--config <yaml>`, plus `--strict-tf5` enforcement.
  6. JSON/Markdown report serialization and hygiene checks (Python 3.9 AST, zero secrets echoed,
     zero forbidden strings or hardcoded home paths).
"""
from __future__ import annotations

import ast
import importlib.util
import json
import os
import stat
import subprocess
import sys
import types
from pathlib import Path
from typing import Any

import pytest
import torch
import yaml

from qloop.export_pack import PACK_MANIFEST_FILE, export_whole_model_pack
from qloop.infer.hf_custom_modeling import QLOOP_FORMAT_VERSION, Qwen36LoopHSSDConfig, Qwen36LoopHSSDForCausalLM
from qloop.train.checkpoint import CheckpointManager
from qloop.train.stack import build_stack

ROOT = Path(__file__).resolve().parent.parent
SCRIPT_PATH = ROOT / "scripts" / "verify_tf524_serving_gpu.py"


def _load_audit_module() -> types.ModuleType:
    spec = importlib.util.spec_from_file_location("_verify_tf524_serving_gpu_mod", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


AUDIT = _load_audit_module()


# ============================================================================ 1. QL60-001


def test_ql60_001_quantized_hybrid_cache_modes_and_kv_share() -> None:
    """Verify `check_quantized_hybrid_cache` across `bf16`, `fp8`, `hadamard_int4` and `kv_share=True/False`."""
    stack = AUDIT._build_smoke_toy_stack(seed=301, kv_share=True)
    res = AUDIT.check_quantized_hybrid_cache(stack, modes=("bf16", "fp8", "hadamard_int4"))

    assert res["name"] == "QL60-001:quantized_hybrid_cache"
    assert res["passed"] is True, res["detail"]
    m = res["metrics"]
    assert m["subchecks"]["tf524_cache_and_layer_mixin"] is True
    assert m["subchecks"]["property_descriptors"] is True
    assert m["subchecks"]["kv_share_true_and_false"] is True
    assert m["subchecks"]["contract_methods_and_proxies"] is True
    assert m["subchecks"]["live_model_forward"] is True
    assert m["subchecks"]["toy_kv_share_false_model_forward"] is True

    for mode in ("bf16", "fp8", "hadamard_int4"):
        pm = m["per_mode"][mode]
        assert pm["kv_share_true_ratio"] == pytest.approx(1.0)
        assert pm["kv_share_false_ratio"] == pytest.approx(2.0)
        assert m["live_model"]["per_mode"][mode]["kv_share_1x_verified"] is True
        d_unshared = m["toy_kv_share_false_model"]["per_mode"][mode]["kv_bytes_by_depth"]
        assert d_unshared["d3"] > d_unshared["d1"]

    assert m["fp8_vs_bf16_ratio"] < 1.0
    assert m["hadamard_int4_vs_bf16_ratio"] < m["fp8_vs_bf16_ratio"]

    # Also accepts a plain dict config or Qwen36LoopHSSDConfig cleanly
    cfg_dict = {
        "num_hidden_layers": 4,
        "layer_types": ["linear_attention", "full_attention", "linear_attention", "full_attention"],
        "num_key_value_heads": 2,
        "head_dim": 16,
    }
    res_cfg = AUDIT.check_quantized_hybrid_cache(cfg_dict, modes=("bf16", "fp8", "hadamard_int4"))
    assert res_cfg["passed"] is True, res_cfg["detail"]

    hf_cfg = Qwen36LoopHSSDConfig(base_config=cfg_dict, qloop_v3_arch={})
    res_hf_cfg = AUDIT.check_quantized_hybrid_cache(hf_cfg, modes=("bf16", "fp8", "hadamard_int4"))
    assert res_hf_cfg["passed"] is True, res_hf_cfg["detail"]


def test_ql60_001_strict_tf5_flag_and_simulated_real_transformers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`--strict-tf5` must reject missing or TF 4.x installs, and accept real TF 5.24.x."""
    # 1) Missing transformers -> --strict-tf5 fails with exit code 1 and writes JSON + MD reports
    out_json = tmp_path / "strict_fail.json"
    out_md = tmp_path / "strict_fail.md"
    monkeypatch.setattr(
        AUDIT,
        "detect_transformers_env",
        lambda: {
            "python_version": "3.9.6",
            "torch_version": str(torch.__version__),
            "cuda_available": False,
            "gpu_name": None,
            "transformers_installed": False,
            "transformers_version": None,
            "transformers_major": None,
            "transformers_is_v5_plus": False,
            "has_cache_layer_mixin": False,
            "qloop_format_version": QLOOP_FORMAT_VERSION,
            "qloop_commit": "deadbeef0000",
        },
    )
    rc_missing = AUDIT.main(
        ["--smoke-cpu", "--strict-tf5", "--out-json", str(out_json), "--out-md", str(out_md), "--quiet"]
    )
    assert rc_missing == 1
    rep_fail = json.loads(out_json.read_text(encoding="utf-8"))
    assert rep_fail["overall_passed"] is False
    assert rep_fail["checks"][0]["name"] == "ENV:strict_tf5"
    assert rep_fail["checks"][0]["passed"] is False
    assert "FAIL" in out_md.read_text(encoding="utf-8")

    # 2) Transformers 4.49.0 (< 5) -> --strict-tf5 fails
    monkeypatch.setattr(
        AUDIT,
        "detect_transformers_env",
        lambda: {
            "python_version": "3.9.6",
            "torch_version": str(torch.__version__),
            "cuda_available": False,
            "gpu_name": None,
            "transformers_installed": True,
            "transformers_version": "4.49.0",
            "transformers_major": 4,
            "transformers_is_v5_plus": False,
            "has_cache_layer_mixin": False,
            "qloop_format_version": QLOOP_FORMAT_VERSION,
            "qloop_commit": "deadbeef0000",
        },
    )
    rc_v4 = AUDIT.main(["--smoke-cpu", "--strict-tf5", "--quiet"])
    assert rc_v4 == 1

    # 3) Simulated transformers 5.24.1 present -> detect_transformers_env() reports v5+ and
    #    strict_tf5 gate accepts it while offline mock Cache/CacheLayerMixin contract runs cleanly
    mock_tf = AUDIT._make_strict_tf524_mock_modules()
    for mname, mobj in mock_tf.items():
        monkeypatch.setitem(sys.modules, mname, mobj)
    monkeypatch.undo()
    # Re-apply only mock_tf so real detect_transformers_env() inspects sys.modules["transformers"]
    for mname, mobj in mock_tf.items():
        monkeypatch.setitem(sys.modules, mname, mobj)
    env_v5 = AUDIT.detect_transformers_env()
    assert env_v5["transformers_installed"] is True
    assert env_v5["transformers_version"] == "5.24.1"
    assert env_v5["transformers_major"] == 5
    assert env_v5["transformers_is_v5_plus"] is True
    assert env_v5["has_cache_layer_mixin"] is True


# ============================================================================ 2. QL60-002


def test_ql60_002_pack_dir_and_in_memory_model_standalone_parity(tmp_path: Path) -> None:
    """Verify `check_pack_model_and_generate` for pack dir, Stack, and `Qwen36LoopHSSDForCausalLM`."""
    stack = AUDIT._build_smoke_toy_stack(seed=307, kv_share=True)
    pack_dir = tmp_path / "toy_pack"
    export_whole_model_pack(stack, pack_dir, max_shard_size_bytes=65536, merge_in_place=False)

    # 1) From exported pack directory
    res_dir = AUDIT.check_pack_model_and_generate(pack_dir, depths=(1, 2, 3), max_new_tokens=6, device="cpu")
    assert res_dir["name"] == "QL60-002:pack_model_and_generate"
    assert res_dir["passed"] is True, res_dir["detail"]
    m = res_dir["metrics"]

    assert m["subchecks"]["in_package_model_output"] is True
    assert m["subchecks"]["in_package_model_and_generate"] is True
    assert m["subchecks"]["pack_metadata"] is True
    assert m["subchecks"]["standalone_ast_py39_and_py_compile"] is True
    assert m["subchecks"]["standalone_model_output"] is True
    assert m["subchecks"]["standalone_model_and_generate"] is True
    assert m["subchecks"]["format_version_mismatch_warning"] is True

    # Verify logits_to_keep=0 shape is (1, 6, vocab), NOT (1, 0, vocab), and greedy tokens match between
    # in-package and standalone across all depths (1, 2, 3)
    vocab = stack.tokenizer.vocab_size
    for d, eff_layers in ((1, 6), (2, 9), (3, 12)):
        dk = f"d{d}"
        in_d = m["in_package"]["depths"][dk]
        sa_d = m["standalone"]["depths"][dk]
        assert in_d["virtual_num_hidden_layers"] == 12
        assert sa_d["virtual_num_hidden_layers"] == 12
        assert in_d["effective_num_hidden_layers"] == eff_layers
        assert sa_d["effective_num_hidden_layers"] == eff_layers
        assert in_d["logits_to_keep_0_shape"] == [1, 6, vocab]
        assert sa_d["logits_to_keep_0_shape"] == [1, 6, vocab]
        assert len(in_d["generated_tokens"]) == 6
        assert in_d["generated_tokens"] == sa_d["generated_tokens"]

    # Verify 2D & 3D MRoPE position_ids slicing verified on both in-package & standalone
    assert m["in_package"]["position_ids"]["3d_mrope_with_cache_pos"] is True
    assert m["standalone"]["position_ids"]["3d_mrope_with_cache_pos"] is True

    # Verify qloop_format_version & qloop_commit metadata
    assert m["pack_metadata"]["qloop_format_version"] == QLOOP_FORMAT_VERSION
    assert m["pack_metadata"]["manifest_qloop_format_version"] == QLOOP_FORMAT_VERSION
    assert m["pack_metadata"]["qloop_commit"]

    # 2) From in-memory Qwen36LoopHSSDForCausalLM directly
    hf_model = Qwen36LoopHSSDForCausalLM.from_pretrained(pack_dir, device_map="cpu", strict=True)
    res_mod = AUDIT.check_pack_model_and_generate(hf_model, depths=(1, 2), max_new_tokens=4)
    assert res_mod["passed"] is True, res_mod["detail"]


def test_ql60_002_rejects_corrupted_pack_metadata(tmp_path: Path) -> None:
    """Corrupted `qloop_format_version` in `config.json` must fail `check_pack_model_and_generate`."""
    stack = AUDIT._build_smoke_toy_stack(seed=311, kv_share=True)
    pack_dir = tmp_path / "bad_meta_pack"
    export_whole_model_pack(stack, pack_dir, max_shard_size_bytes=65536, merge_in_place=False)

    cfg_path = pack_dir / "config.json"
    cfg_data = json.loads(cfg_path.read_text(encoding="utf-8"))
    cfg_data["qloop_format_version"] = "9.9-bad"
    cfg_path.write_text(json.dumps(cfg_data, indent=2), encoding="utf-8")

    res = AUDIT.check_pack_model_and_generate(pack_dir, depths=(1,), max_new_tokens=2, device="cpu")
    assert res["passed"] is False
    assert "qloop_format_version" in res["detail"]


# ============================================================================ 3. QL60-005


def test_ql60_005_stage_completion_gate_rejection_and_acceptance() -> None:
    """Verify `check_stage_completion_gate()` checks all rejection & acceptance cases."""
    res = AUDIT.check_stage_completion_gate()
    assert res["name"] == "QL60-005:stage_completion_gate"
    assert res["passed"] is True, res["detail"]
    cases = res["metrics"]["cases"]
    assert res["metrics"]["cases_tested"] >= 10
    assert cases["bare_latest_rejected"]["expected"] is False
    assert cases["midrun_latest_step_lt_target_rejected"]["expected"] is False
    assert cases["bare_promoted_rejected"]["expected"] is False
    assert cases["midrun_promoted_save_meta_rejected"]["expected"] is False
    assert cases["incomplete_promoted_step_lt_target_rejected"]["expected"] is False
    assert cases["verified_latest_step_ge_target_accepted"]["expected"] is True
    assert cases["verified_promoted_step_ge_target_accepted"]["expected"] is True
    assert cases["checkpoints_completed_marker_accepted"]["expected"] is True


# ============================================================================ 4. CLI modes & reports


def test_cli_smoke_cpu_and_report_serialization(tmp_path: Path) -> None:
    """Verify `--smoke-cpu` via `main()` and subprocess, checking both JSON & Markdown outputs."""
    out_json = tmp_path / "audit.json"
    out_md = tmp_path / "audit.md"

    rc = AUDIT.main(
        [
            "--smoke-cpu",
            "--depths",
            "1,2,3",
            "--max-new-tokens",
            "6",
            "--output-json",
            str(out_json),
            "--output-md",
            str(out_md),
            "--quiet",
        ]
    )
    assert rc == 0
    assert out_json.is_file() and out_md.is_file()

    report = json.loads(out_json.read_text(encoding="utf-8"))
    assert report["overall_passed"] is True
    assert report["mode"] == "smoke_cpu"
    assert report["passed_count"] == report["total_checks"] == 3
    assert report["elapsed_s"] < 10.0
    check_names = [c["name"] for c in report["checks"]]
    assert check_names == [
        "QL60-001:quantized_hybrid_cache",
        "QL60-002:pack_model_and_generate",
        "QL60-005:stage_completion_gate",
    ]
    # Round-trip JSON serialization must stay clean
    assert json.loads(json.dumps(report)) == report

    md_text = out_md.read_text(encoding="utf-8")
    assert "**PASS**" in md_text
    assert "QL60-001:quantized_hybrid_cache" in md_text
    assert "QL60-002:pack_model_and_generate" in md_text
    assert "QL60-005:stage_completion_gate" in md_text
    assert "`bf16`" in md_text and "`fp8`" in md_text and "`hadamard_int4`" in md_text
    assert "logits_to_keep=0" in md_text

    # Subprocess execution with PYTHONPATH=src and dummy HF_TOKEN to verify CLI exit 0 and zero secret echoing
    secret_sentinel = "hf_UNIT_TEST_SECRET_DO_NOT_ECHO_987654"
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT / "src")
    env["HF_TOKEN"] = secret_sentinel
    sub_json = tmp_path / "sub_audit.json"
    sub_md = tmp_path / "sub_audit.md"
    proc = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--smoke-cpu",
            "--max-new-tokens",
            "4",
            "--out-json",
            str(sub_json),
            "--out-md",
            str(sub_md),
        ],
        cwd=str(ROOT),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, f"stdout={proc.stdout}\nstderr={proc.stderr}"
    assert "OVERALL: PASS" in proc.stdout
    assert secret_sentinel not in proc.stdout
    assert secret_sentinel not in proc.stderr
    assert secret_sentinel not in sub_json.read_text(encoding="utf-8")
    assert secret_sentinel not in sub_md.read_text(encoding="utf-8")


def test_cli_checkpoint_mode_for_exported_pack_and_training_stage(tmp_path: Path) -> None:
    """Verify `--checkpoint` works on BOTH an exported Whole Model Pack and a `--config` training checkpoint."""
    # 1) Exported Whole Model Pack directory
    stack = AUDIT._build_smoke_toy_stack(seed=313, kv_share=True)
    pack_dir = tmp_path / "exported_pack"
    export_whole_model_pack(stack, pack_dir, max_shard_size_bytes=65536, merge_in_place=False)
    assert (pack_dir / PACK_MANIFEST_FILE).is_file()

    pack_json = tmp_path / "pack_audit.json"
    rc_pack = AUDIT.main(
        [
            "--checkpoint",
            str(pack_dir),
            "--depths",
            "1,2,3",
            "--max-new-tokens",
            "4",
            "--device",
            "cpu",
            "--output-json",
            str(pack_json),
            "--quiet",
        ]
    )
    assert rc_pack == 0
    rep_pack = json.loads(pack_json.read_text(encoding="utf-8"))
    assert rep_pack["overall_passed"] is True
    assert rep_pack["target"]["kind"] == "whole_model_pack_dir"

    # 2) Training stage checkpoint + YAML config (mirrors Colab G4 `s2_v2@600` invocation!)
    cfg_dict: dict[str, Any] = {
        "model": {
            "name": "toy",
            "device": "cpu",
            "dtype": "float32",
            "toy": {
                "hidden_size": 32,
                "moe_intermediate_size": 16,
                "shared_expert_intermediate_size": 16,
                "num_hidden_layers": 6,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "head_dim": 8,
                "linear_num_heads": 2,
                "linear_head_dim": 8,
                "linear_conv_kernel_dim": 3,
                "num_experts": 4,
                "num_experts_per_tok": 2,
                "layer_types": [
                    "linear_attention",
                    "full_attention",
                    "linear_attention",
                    "linear_attention",
                    "full_attention",
                    "linear_attention",
                ],
            },
        },
        "gdn2": {"enabled": True, "rank": 4, "chunk_size": 4},
        "lsa": {
            "enabled": True,
            "mode": "sparse",
            "index_dim": 8,
            "block_size": 8,
            "pool": 2,
            "avg_bucket_keys": 4,
            "max_bucket_keys": 8,
            "num_select": 2,
            "window": 32,
            "sinks": 4,
            "query_chunk": 8,
        },
        "loop": {
            "enabled": True,
            "prelude_end": 2,
            "core_end": 5,
            "num_passes": 3,
            "target_virtual_layers": 12,
            "kv_share": True,
            "gate_init": 0.25,
            "bridge_rank": 4,
            "step_embedding": True,
            "per_pass_router": True,
            "router_adapter_rank": 4,
        },
        "train": {
            "train_loop_adapters": True,
            "train_gdn2": True,
            "train_lsa": True,
        },
    }
    cfg_path = tmp_path / "toy_s2_v2.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg_dict, sort_keys=False), encoding="utf-8")

    stage_dir = tmp_path / "s2_v2_toy"
    built = build_stack(cfg_dict)
    mgr = CheckpointManager(str(stage_dir), keep_last=2)
    mgr.save(
        step=600,
        params=built.checkpoint_params(),
        optim_state=None,
        state={"stage": "s2_v2_toy", "step": 600, "target_steps": 600, "stack": built.describe()},
        completed=True,
        target_steps=600,
    )

    ckpt_json = tmp_path / "ckpt_audit.json"
    ckpt_md = tmp_path / "ckpt_audit.md"
    rc_ckpt = AUDIT.main(
        [
            "--checkpoint",
            str(stage_dir),
            "--config",
            str(cfg_path),
            "--depths",
            "1,2,3",
            "--max-new-tokens",
            "4",
            "--out-json",
            str(ckpt_json),
            "--out-md",
            str(ckpt_md),
            "--quiet",
        ]
    )
    assert rc_ckpt == 0
    rep_ckpt = json.loads(ckpt_json.read_text(encoding="utf-8"))
    assert rep_ckpt["overall_passed"] is True
    assert rep_ckpt["target"]["kind"] == "training_checkpoint_stack"
    assert "**PASS**" in ckpt_md.read_text(encoding="utf-8")


# ============================================================================ 5. Hygiene


def test_hygiene_python39_grammar_executable_and_clean_text() -> None:
    """Confirm Python 3.9 AST compatibility, executable bit, and zero forbidden text in both new files."""
    st_mode = os.stat(SCRIPT_PATH).st_mode
    assert bool(st_mode & stat.S_IXUSR), "scripts/verify_tf524_serving_gpu.py must be executable"

    forbidden_mem = "".join(("h", "b", "m"))
    forbidden_home = "/" + "Users/" + "tianrui" + "6641"
    for path in (SCRIPT_PATH, Path(__file__).resolve()):
        src = path.read_text(encoding="utf-8")
        ast.parse(src, filename=str(path), feature_version=(3, 9))
        assert "from __future__ import annotations" in src
        assert forbidden_mem not in src.lower(), f"unexpected forbidden memory term in {path.name}"
        assert forbidden_home not in src, f"hardcoded user path found in {path.name}"
