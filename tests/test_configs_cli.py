"""Every shipped config must load and validate; the CLI must drive the toy pipeline end-to-end."""
from __future__ import annotations

import ast
import functools
import glob
import importlib.util
import json
import os
import sys
import types
from types import SimpleNamespace

import pytest

from qloop import doctor
from qloop.cli import main
from qloop.colab import (
    BASE_MODEL_BY_PRECISION,
    _resolve_hf_kernels_pip_spec,
    _subprocess_env,
    ensure_quant_runtime_packages,
    resolve_base_model_name,
    run_pipeline,
    run_probe_notebook_suite,
    run_v3_e2e_and_ship_pack,
    setup_env,
)
from qloop.config import load_config, yaml_load
from qloop.model.gdn2 import GDN2Config
from qloop.model.hssd import HSSDConfig
from qloop.model.loop import LoopSpec, build_schedule
from qloop.model.lsa import LSAConfig

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
QWEN_TYPES = ["full_attention" if (i + 1) % 4 == 0 else "linear_attention" for i in range(40)]
EXPECTED_QWEN36_DIMS = {
    "num_hidden_layers": 40,
    "hidden_size": 2048,
    "num_attention_heads": 16,
    "num_key_value_heads": 2,
    "head_dim": 256,
    "full_attention_interval": 4,
    "num_experts": 256,
    "num_experts_per_tok": 8,
    "linear_num_key_heads": 16,
    "linear_num_value_heads": 32,
    "linear_key_head_dim": 128,
    "linear_value_head_dim": 128,
    "linear_conv_kernel_dim": 4,
}


@pytest.mark.parametrize("path", sorted(glob.glob(os.path.join(ROOT, "configs", "**", "*.yaml"), recursive=True)))
def test_every_config_loads_and_validates(path):
    cfg = load_config(path)
    if "model" not in cfg:
        return  # partial (model-only / base) fragments
    loop = LoopSpec.from_dict(dict(cfg.get("loop") or {}))
    GDN2Config.from_dict(dict(cfg.get("gdn2") or {})).validate()
    HSSDConfig.from_dict(dict(cfg.get("hssd") or {})).validate()
    LSAConfig.from_dict(dict(cfg.get("lsa") or {})).validate()
    if cfg.model.name == "toy":
        types = ["full_attention" if (i + 1) % 4 == 0 else "linear_attention" for i in range(16)]
    else:
        types = QWEN_TYPES
        assert loop.virtual_layers(40) in (64, 88), f"{path}: Qwen configs must target 64 or 88 virtual layers"
        norm_path = path.replace("\\", "/")
        if "stages/" in norm_path or "eval/" in norm_path:
            assert cfg.model.name == "Qwen/Qwen3.6-35B-A3B-FP8", f"{path}: stage/eval default must be FP8"
            assert loop.per_pass_router is True
            assert (cfg.get("gdn2") or {}).get("enabled") is True or (cfg.get("hssd") or {}).get("enabled") is True
            sync = (cfg.get("checkpoint") or {}).get("sync") or {}
            assert sync.get("hf_repo") == "TassP0lar/Qwen3.6-35B-A3B-Loop64"
            assert sync.get("private") is False
    assert loop.validate(types) == []
    assert len(build_schedule(loop, types)) == loop.virtual_layers(len(types))
    if cfg.get("stage_type") in ("lm", "rl"):
        assert cfg.get("train"), f"{path}: training stage without train section"


def test_blackwell_fp8_nvfp4_and_bf16_model_configs():
    fp8 = load_config(os.path.join(ROOT, "configs", "model", "qwen36_fp8.yaml"))
    nvfp4 = load_config(os.path.join(ROOT, "configs", "model", "qwen36_nvfp4.yaml"))
    bf16 = load_config(os.path.join(ROOT, "configs", "model", "qwen36_bf16.yaml"))

    assert fp8.model.name == "Qwen/Qwen3.6-35B-A3B-FP8"
    assert fp8.model.dtype == "bfloat16"
    assert fp8.model.attn_implementation == "sdpa"
    assert fp8.model.device_map == "cuda:0"
    assert fp8.model.text_only is True
    assert fp8.model.trust_remote_code is False
    assert dict(fp8.model.expected) == EXPECTED_QWEN36_DIMS

    assert nvfp4.model.name == "unsloth/Qwen3.6-35B-A3B-NVFP4-Fast"
    assert nvfp4.model.dtype == "bfloat16"
    assert nvfp4.model.attn_implementation == "sdpa"
    assert nvfp4.model.device_map == "cuda:0"
    assert nvfp4.model.text_only is True
    assert nvfp4.model.trust_remote_code is False
    assert dict(nvfp4.model.expected) == EXPECTED_QWEN36_DIMS

    assert bf16.model.name == "Qwen/Qwen3.6-35B-A3B"


def test_resolve_base_model_name_and_colab_precision_plumbing(tmp_path, monkeypatch):
    assert BASE_MODEL_BY_PRECISION == {
        "fp8": "Qwen/Qwen3.6-35B-A3B-FP8",
        "nvfp4": "unsloth/Qwen3.6-35B-A3B-NVFP4-Fast",
        "bf16": "Qwen/Qwen3.6-35B-A3B",
    }
    assert resolve_base_model_name(None) is None
    assert resolve_base_model_name("fp8") == "Qwen/Qwen3.6-35B-A3B-FP8"
    assert resolve_base_model_name("FP8") == "Qwen/Qwen3.6-35B-A3B-FP8"
    assert resolve_base_model_name("nvfp4") == "unsloth/Qwen3.6-35B-A3B-NVFP4-Fast"
    assert resolve_base_model_name("NVFP4") == "unsloth/Qwen3.6-35B-A3B-NVFP4-Fast"
    assert resolve_base_model_name("bf16") == "Qwen/Qwen3.6-35B-A3B"
    assert resolve_base_model_name("org/custom-qwen36") == "org/custom-qwen36"

    for bad in ("", "  ", "int8", "fp16", "a/b/c", "/model", "org/", "org/bad space"):
        with pytest.raises(ValueError, match="invalid base_precision"):
            resolve_base_model_name(bad)

    calls: list[str] = []
    monkeypatch.setenv("QLOOP_RUN_ROOT", str(tmp_path))
    monkeypatch.setattr("qloop.colab.q", lambda cmd, check=True: calls.append(cmd) or 0)
    monkeypatch.setattr(
        "qloop.colab.ensure_quant_runtime_packages",
        lambda base_precision="fp8", *, auto_install=True: {
            "precision": str(base_precision),
            "package": "stub",
            "pip_spec": "",
            "status": "ready",
        },
    )

    run_pipeline(hf_repo="test/repo")
    assert "model.name=Qwen/Qwen3.6-35B-A3B-FP8" in calls[-1]

    run_pipeline(hf_repo="test/repo", base_precision="nvfp4")
    assert "model.name=unsloth/Qwen3.6-35B-A3B-NVFP4-Fast" in calls[-1]

    run_pipeline(hf_repo="test/repo", base_precision="nvfp4", overrides=("model.name=custom/override",))
    assert "model.name=custom/override" in calls[-1]
    assert "NVFP4-Fast" not in calls[-1]

    run_pipeline(hf_repo="test/repo", base_precision="nvfp4", smoke_toy=True)
    assert "model.name=" not in calls[-1]

    # Verify run_v3_e2e_and_ship_pack propagates base_precision into BOTH pipeline and export-pack
    class _DoneManifest:
        def is_completed(self, _stage: str) -> bool:
            return True

    monkeypatch.setattr("qloop.hub.HubSyncManager.fetch_manifest", lambda self: _DoneManifest())
    calls.clear()
    res = run_v3_e2e_and_ship_pack(
        hf_repo="test/repo",
        pack_repo="test/pack",
        base_precision="nvfp4",
        stages=("s0_probe", "s1_loop_a6b"),
    )
    assert res["pack_shipped"] is True
    pipe_cmd = next(c for c in calls if c.startswith("sync pipeline"))
    pack_cmd = next(c for c in calls if c.startswith("export-pack"))
    assert "model.name=unsloth/Qwen3.6-35B-A3B-NVFP4-Fast" in pipe_cmd
    assert "model.name=unsloth/Qwen3.6-35B-A3B-NVFP4-Fast" in pack_cmd

    # Verify run_probe_notebook_suite injects base_precision when smoke_toy=False and keeps toy when smoke_toy=True
    def _fake_probe_q(cmd: str, check: bool = True) -> int:
        calls.append(cmd)
        probe_dir = tmp_path / "probe"
        probe_dir.mkdir(parents=True, exist_ok=True)
        (probe_dir / "probe_report.json").write_text(
            json.dumps({"summary": {"all_critical_ok": True}, "gates": {}, "markdown": "# Probe\n"}),
            encoding="utf-8",
        )
        (probe_dir / "probe_report.md").write_text("# Probe\n", encoding="utf-8")
        return 0

    monkeypatch.setattr("qloop.colab.q", _fake_probe_q)
    monkeypatch.setattr("qloop.model.probes.profile_hssd_layer", None, raising=False)
    calls.clear()
    run_probe_notebook_suite(smoke_toy=False, base_precision="nvfp4", run_math_gate=False, push_to_hub=False)
    assert "model.name=unsloth/Qwen3.6-35B-A3B-NVFP4-Fast" in calls[-1]

    calls.clear()
    run_probe_notebook_suite(smoke_toy=True, base_precision="nvfp4", run_math_gate=False, push_to_hub=False)
    assert "model.name=toy" in calls[-1]
    assert "NVFP4-Fast" not in calls[-1]


def test_setup_env_scrubs_deprecated_hf_transfer_and_enables_xet(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_HUB_ENABLE_HF_TRANSFER", "1")
    monkeypatch.delenv("HF_XET_HIGH_PERFORMANCE", raising=False)

    setup_env(
        hf_repo="TassP0lar/Qwen3.6-35B-A3B-Loop64",
        run_root=str(tmp_path / "runs"),
        hf_home=str(tmp_path / "hf_cache"),
        require_hf_token=False,
    )
    assert "HF_HUB_ENABLE_HF_TRANSFER" not in os.environ
    assert os.environ.get("HF_XET_HIGH_PERFORMANCE") == "1"

    monkeypatch.setenv("HF_HUB_ENABLE_HF_TRANSFER", "1")
    monkeypatch.delenv("HF_XET_HIGH_PERFORMANCE", raising=False)
    sub_env = _subprocess_env()
    assert "HF_HUB_ENABLE_HF_TRANSFER" not in sub_env
    assert sub_env.get("HF_XET_HIGH_PERFORMANCE") == "1"


def test_resolve_hf_kernels_pip_spec_and_runtime_auto_install(tmp_path, monkeypatch):
    hf_err = (
        "finegrained-fp8 kernel unavailable: kernels is either not installed or uses an incompatible version. "
        "Please install a compatible version (0.18.2 <= version < 0.19.0), e.g. pip install kernels==0.18.2"
    )
    assert _resolve_hf_kernels_pip_spec(hf_err) == (False, "kernels>=0.18.2,<0.19.0")
    assert _resolve_hf_kernels_pip_spec("Please run pip install kernels==0.18.2") == (False, "kernels==0.18.2")

    real_find_spec = importlib.util.find_spec
    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda name, *a, **kw: None if name in ("kernels", "compressed_tensors") else real_find_spec(name, *a, **kw),
    )

    fake_tf = types.ModuleType("transformers")
    fake_tf_int = types.ModuleType("transformers.integrations")
    fake_fg = types.ModuleType("transformers.integrations.finegrained_fp8")
    fake_fg._MISSING_KERNELS_MESSAGE = hf_err

    def _raise_missing():
        raise ImportError(hf_err)

    fake_fg.load_finegrained_fp8_kernel = _raise_missing
    fake_tf.integrations = fake_tf_int
    fake_tf_int.finegrained_fp8 = fake_fg
    monkeypatch.setitem(sys.modules, "transformers", fake_tf)
    monkeypatch.setitem(sys.modules, "transformers.integrations", fake_tf_int)
    monkeypatch.setitem(sys.modules, "transformers.integrations.finegrained_fp8", fake_fg)

    assert _resolve_hf_kernels_pip_spec() == (False, "kernels>=0.18.2,<0.19.0")

    fake_fg._MISSING_KERNELS_MESSAGE = ""
    fake_fg.load_finegrained_fp8_kernel = lambda: (_ for _ in ()).throw(ImportError("pip install kernels==0.18.2"))
    assert _resolve_hf_kernels_pip_spec() == (False, "kernels==0.18.2")

    fake_fg.load_finegrained_fp8_kernel = object
    assert _resolve_hf_kernels_pip_spec() == (True, "kernels")

    delattr(fake_fg, "load_finegrained_fp8_kernel")
    fake_fg._MISSING_KERNELS_MESSAGE = "compatible version (0.18.2 <= version < 0.19.0)"
    assert _resolve_hf_kernels_pip_spec() == (False, "kernels>=0.18.2,<0.19.0")

    # Never invoke subprocess pip when auto_install=False or on non-Linux platforms
    pip_cmds: list[list[str]] = []

    def _forbid_subproc(cmd, *args, **kwargs):
        pip_cmds.append(list(cmd))
        raise AssertionError(f"unexpected subprocess call: {cmd}")

    monkeypatch.setattr("qloop.colab.subprocess.run", _forbid_subproc)
    monkeypatch.setattr(sys, "platform", "linux")
    assert ensure_quant_runtime_packages("fp8", auto_install=False) == {
        "precision": "fp8",
        "package": "kernels",
        "pip_spec": "kernels>=0.18.2,<0.19.0",
        "status": "missing (kernels>=0.18.2,<0.19.0)",
    }
    assert ensure_quant_runtime_packages("Qwen/Qwen3.6-35B-A3B-FP8", auto_install=False)["precision"] == "fp8"
    assert ensure_quant_runtime_packages("nvfp4", auto_install=False) == {
        "precision": "nvfp4",
        "package": "compressed_tensors",
        "pip_spec": "compressed-tensors",
        "status": "missing (compressed-tensors)",
    }
    assert (
        ensure_quant_runtime_packages("unsloth/Qwen3.6-35B-A3B-NVFP4-Fast", auto_install=False)["precision"] == "nvfp4"
    )
    assert ensure_quant_runtime_packages("bf16", auto_install=False)["status"] == "not_required"

    setup_env(
        hf_repo="TassP0lar/Qwen3.6-35B-A3B-Loop64",
        run_root=str(tmp_path / "runs_no_pip"),
        hf_home=str(tmp_path / "hf_no_pip"),
        require_hf_token=False,
        base_precision="fp8",
        auto_install_runtime_deps=False,
    )
    monkeypatch.setattr(sys, "platform", "darwin")
    ensure_quant_runtime_packages("fp8", auto_install=True)
    setup_env(
        hf_repo="TassP0lar/Qwen3.6-35B-A3B-Loop64",
        run_root=str(tmp_path / "runs_mac"),
        hf_home=str(tmp_path / "hf_mac"),
        require_hf_token=False,
        base_precision="fp8",
        auto_install_runtime_deps=True,
    )
    assert pip_cmds == []

    # Simulated Colab Linux auto-install (including non-fatal offline handling & smoke_toy skip)
    monkeypatch.setattr(sys, "platform", "linux")
    next_rc = 0

    def _record_pip(cmd, *args, **kwargs):
        pip_cmds.append(list(cmd))
        return SimpleNamespace(returncode=next_rc, stdout="", stderr="offline" if next_rc else "")

    monkeypatch.setattr("qloop.colab.subprocess.run", _record_pip)
    r_fp8 = ensure_quant_runtime_packages("fp8", auto_install=True)
    assert pip_cmds[-1] == [sys.executable, "-m", "pip", "install", "-q", "--no-cache-dir", "kernels>=0.18.2,<0.19.0"]
    assert r_fp8["precision"] == "fp8"

    r_nv4 = ensure_quant_runtime_packages("nvfp4", auto_install=True)
    assert pip_cmds[-1] == [sys.executable, "-m", "pip", "install", "-q", "--no-cache-dir", "compressed-tensors"]
    assert r_nv4["status"] == "installed"

    next_rc = 1
    r_offline = ensure_quant_runtime_packages("fp8", auto_install=True)
    assert r_offline["status"].startswith("install_failed")

    pip_cmds.clear()
    monkeypatch.setattr("qloop.colab.q", lambda cmd, check=True: 0)
    run_pipeline(hf_repo="test/repo", base_precision="fp8", smoke_toy=True)
    assert pip_cmds == [], "smoke_toy=True must never run pip on Linux"
    next_rc = 0
    run_pipeline(hf_repo="test/repo", base_precision="fp8", smoke_toy=False)
    assert len(pip_cmds) == 1


def test_hf5_underscore_loader_and_lru_cache_clear_on_pip_install(monkeypatch):
    """Reproduce transformers=5.23.0 `_load_finegrained_fp8_kernel` + `@lru_cache` `is_kernels_available`."""
    hf_err = (
        "finegrained-fp8 kernel unavailable: kernels is either not installed or uses an incompatible version. "
        "Please install a compatible version (0.18.2 <= version < 0.19.0), e.g. pip install kernels==0.18.2"
    )
    kernels_installed = False
    real_find_spec = importlib.util.find_spec

    def _fake_find_spec(name, *a, **kw):
        if name == "kernels":
            return SimpleNamespace(name="kernels") if kernels_installed else None
        return real_find_spec(name, *a, **kw)

    monkeypatch.setattr(importlib.util, "find_spec", _fake_find_spec)

    fake_tf = types.ModuleType("transformers")
    fake_tf_utils = types.ModuleType("transformers.utils")
    fake_imp_utils = types.ModuleType("transformers.utils.import_utils")
    fake_imp_utils.KERNELS_MIN_VERSION = "0.18.2"
    fake_imp_utils.KERNELS_MAX_VERSION = "0.19.0"

    @functools.lru_cache
    def _is_package_available(pkg_name: str):
        if pkg_name == "kernels":
            return (kernels_installed, "0.18.2" if kernels_installed else "N/A")
        return (False, "N/A")

    @functools.lru_cache
    def is_kernels_available() -> bool:
        return _is_package_available("kernels")[0]

    fake_imp_utils._is_package_available = _is_package_available
    fake_imp_utils.is_kernels_available = is_kernels_available

    fake_tf_int = types.ModuleType("transformers.integrations")
    fake_fg = types.ModuleType("transformers.integrations.finegrained_fp8")

    # Underscore-only `_load_finegrained_fp8_kernel` with NO `_MISSING_KERNELS_MESSAGE` attribute
    def _load_finegrained_fp8_kernel():
        if not is_kernels_available():
            raise ImportError(hf_err)
        return SimpleNamespace(name="finegrained_fp8_triton_kernel")

    fake_fg._load_finegrained_fp8_kernel = _load_finegrained_fp8_kernel
    fake_tf.utils = fake_tf_utils
    fake_tf_utils.import_utils = fake_imp_utils
    fake_tf.integrations = fake_tf_int
    fake_tf_int.finegrained_fp8 = fake_fg

    monkeypatch.setitem(sys.modules, "transformers", fake_tf)
    monkeypatch.setitem(sys.modules, "transformers.utils", fake_tf_utils)
    monkeypatch.setitem(sys.modules, "transformers.utils.import_utils", fake_imp_utils)
    monkeypatch.setitem(sys.modules, "transformers.integrations", fake_tf_int)
    monkeypatch.setitem(sys.modules, "transformers.integrations.finegrained_fp8", fake_fg)

    # 1. Underscore-only loader populates `@lru_cache` with False and resolves exact range spec
    assert _resolve_hf_kernels_pip_spec() == (False, "kernels>=0.18.2,<0.19.0")
    assert is_kernels_available() is False
    assert is_kernels_available.cache_info().currsize == 1

    # 2. Simulate runtime `pip install` on Colab Linux: must `.cache_clear()` so `ok_after` is True immediately
    monkeypatch.setattr(sys, "platform", "linux")
    pip_calls: list[list[str]] = []

    def _sim_pip_install(cmd, *args, **kwargs):
        nonlocal kernels_installed
        pip_calls.append(list(cmd))
        kernels_installed = True
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr("qloop.colab.subprocess.run", _sim_pip_install)
    first = ensure_quant_runtime_packages("fp8", auto_install=True)
    assert len(pip_calls) == 1
    assert pip_calls[0][-1] == "kernels>=0.18.2,<0.19.0"
    assert first["status"] == "installed"
    assert is_kernels_available() is True

    # 3. Subsequent Cell 3 calls (`ensure_quant_runtime_packages` / `run_pipeline`) see `ok == True` -> 0 extra pip runs
    second = ensure_quant_runtime_packages("fp8", auto_install=True)
    assert second["status"] == "ready"
    monkeypatch.setattr("qloop.colab.q", lambda cmd, check=True: 0)
    run_pipeline(hf_repo="test/repo", base_precision="fp8", smoke_toy=False)
    assert len(pip_calls) == 1

    # 4. Also verify KERNELS_MIN_VERSION / KERNELS_MAX_VERSION fallback when loader is absent
    kernels_installed = False
    is_kernels_available.cache_clear()
    _is_package_available.cache_clear()
    delattr(fake_fg, "_load_finegrained_fp8_kernel")
    fake_imp_utils.KERNELS_MIN_VERSION = "0.19.0"
    fake_imp_utils.KERNELS_MAX_VERSION = "0.20.0"
    assert _resolve_hf_kernels_pip_spec() == (False, "kernels>=0.19.0,<0.20.0")


def test_doctor_blackwell_fp8_nvfp4_and_compressed_tensors(monkeypatch):
    assert doctor.DEFAULT_MODEL == "Qwen/Qwen3.6-35B-A3B-FP8"
    assert doctor.build_parser().parse_args([]).model == "Qwen/Qwen3.6-35B-A3B-FP8"

    pkg_checks = {c.name: c for c in doctor.check_packages()}
    assert "kernels" in pkg_checks
    assert pkg_checks["kernels"].status in ("PASS", "WARN")
    assert "finegrained_fp8" in pkg_checks
    assert pkg_checks["finegrained_fp8"].status in ("PASS", "WARN")
    assert "compressed_tensors" in pkg_checks
    assert pkg_checks["compressed_tensors"].status in ("PASS", "WARN")

    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(sys, "platform", "linux")
    fake_tf = types.ModuleType("transformers")
    fake_tf_int = types.ModuleType("transformers.integrations")
    fake_fg = types.ModuleType("transformers.integrations.finegrained_fp8")
    fake_tf.integrations = fake_tf_int
    fake_tf_int.finegrained_fp8 = fake_fg
    monkeypatch.setitem(sys.modules, "transformers", fake_tf)
    monkeypatch.setitem(sys.modules, "transformers.integrations", fake_tf_int)
    monkeypatch.setitem(sys.modules, "transformers.integrations.finegrained_fp8", fake_fg)

    monkeypatch.setattr("qloop.colab._resolve_hf_kernels_pip_spec", lambda: (False, "kernels>=0.18.2,<0.19.0"))
    cuda_pkg_checks = {c.name: c for c in doctor.check_packages()}
    assert cuda_pkg_checks["finegrained_fp8"].status == "WARN"
    assert 'pip install "kernels>=0.18.2,<0.19.0"' in cuda_pkg_checks["finegrained_fp8"].fix

    monkeypatch.setattr("qloop.colab._resolve_hf_kernels_pip_spec", lambda: (True, "kernels"))
    ok_pkg_checks = {c.name: c for c in doctor.check_packages()}
    assert ok_pkg_checks["finegrained_fp8"].status == "PASS"

    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _idx: SimpleNamespace(name="NVIDIA RTX PRO 6000 Blackwell", major=12, minor=0, total_memory=96 * 2**30),
    )
    monkeypatch.setattr(torch.cuda, "get_arch_list", lambda: ["sm_90", "sm_120"])
    gpu_checks = {c.name: c for c in doctor.check_torch_gpu() if c.name != "cuda smoke"}
    assert gpu_checks["gpu"].status == "PASS"
    assert gpu_checks["torch arch"].status == "PASS"
    assert gpu_checks["fp8/nvfp4 hw"].status == "PASS"
    assert "Blackwell" in gpu_checks["fp8/nvfp4 hw"].detail


def test_notebooks_and_setup_script_blackwell_conventions():
    nb_paths = sorted(glob.glob(os.path.join(ROOT, "notebooks", "*.ipynb")))
    assert len(nb_paths) == 4
    for nb_path in nb_paths:
        with open(nb_path, encoding="utf-8") as f:
            raw = f.read()
        assert "HF_HUB_ENABLE_HF_TRANSFER" not in raw, f"{nb_path} still references HF_HUB_ENABLE_HF_TRANSFER"
        nb = json.loads(raw)
        assert nb.get("nbformat") == 4
        code_cells = ["".join(c.get("source", [])) for c in nb.get("cells", []) if c.get("cell_type") == "code"]
        assert any('BASE_PRECISION = "fp8"' in cell for cell in code_cells), f"{nb_path} missing BASE_PRECISION"
        assert any(
            "setup_env(" in cell and "base_precision=BASE_PRECISION" in cell for cell in code_cells
        ), f"{nb_path} setup_env missing base_precision=BASE_PRECISION"
        for idx, cell in enumerate(code_cells):
            lines = ["pass  # magic" if ln.lstrip().startswith(("!", "%")) else ln for ln in cell.splitlines()]
            ast.parse("\n".join(lines), filename=f"{nb_path}#cell{idx}", feature_version=(3, 9))

    with open(os.path.join(ROOT, "scripts", "colab_setup.sh"), encoding="utf-8") as f:
        setup_sh = f.read()
    assert "HF_XET_HIGH_PERFORMANCE=1" in setup_sh
    assert '"kernels>=0.18.2,<0.19.0"' in setup_sh
    assert "compressed-tensors" in setup_sh
    assert "accelerate hf_transfer" not in setup_sh

    with open(os.path.join(ROOT, "pyproject.toml"), encoding="utf-8") as f:
        pyproj = f.read()
    assert "hf_transfer" not in pyproj
    assert '"kernels>=0.18.2,<0.19.0"' in pyproj
    assert '"compressed-tensors>=0.9"' in pyproj
    assert "\ngpu = [\n" in pyproj

    with open(os.path.join(ROOT, "README.md"), encoding="utf-8") as f:
        readme = f.read()
    assert "kernels>=0.18.2,<0.19.0" in readme

    # Ensure all stage configs parse cleanly via yaml_load as well
    for st_yaml in glob.glob(os.path.join(ROOT, "configs", "stages", "*.yaml")):
        with open(st_yaml, encoding="utf-8") as f:
            assert isinstance(yaml_load(f.read()), dict)


def test_cli_toy_pipeline(tmp_path, monkeypatch):
    monkeypatch.setenv("QLOOP_RUN_ROOT", str(tmp_path))
    lm = os.path.join(ROOT, "configs", "local", "toy_lm.yaml")
    assert main(["train", "-c", lm, "-s", "train.steps=4", "-s", "train.ckpt_every=2"]) == 0
    run = tmp_path / "toy_lm"
    assert (run / "checkpoints" / "LATEST").exists() and (run / "metrics.jsonl").exists()
    assert main(["probe", "-c", lm, "--backward", "--surgery", "--core-ranges", "--n", "2", "--seq-len", "48",
                 "-s", "run.name=probe"]) == 0
    probe = json.loads((tmp_path / "probe" / "probe.json").read_text())
    assert probe["backward"]["ok"] and probe["surgery"]["init_identity_maxdiff"] == 0.0 and probe["core_ranges"]
    assert main(["scan-experts", "-c", lm, "--batches", "2", "--seq-len", "48", "-s", "run.name=scan"]) == 0
    assert (tmp_path / "scan" / "esft_stats.json").exists()
    out = tmp_path / "export"
    assert main(["export", "-c", lm, "--ckpt", str(run), "--out", str(out), "-s", "run.name=exp"]) == 0
    assert (out / "qloop_adapters.pt").exists() and (out / "qloop_config.json").exists()
    rl = os.path.join(ROOT, "configs", "local", "toy_rl.yaml")
    assert main(["train", "-c", rl, "-s", "train.steps=2"]) == 0
    assert (tmp_path / "toy_rl" / "checkpoints" / "PROMOTED").exists()


def test_cli_failure_writes_crash_report(tmp_path, monkeypatch):
    monkeypatch.setenv("QLOOP_RUN_ROOT", str(tmp_path))
    lm = os.path.join(ROOT, "configs", "local", "toy_lm.yaml")
    assert main(["train", "-c", lm, "-s", "loop.core_end=10", "-s", "run.name=bad"]) == 1
    reports = os.listdir(tmp_path / "bad" / "crash")
    assert reports and "virtual layers" in (tmp_path / "bad" / "crash" / reports[0]).read_text()


def test_v3_stages_calibrated_s0_probe_invariants():
    v3_stages = ("s1_loop_a6b", "s1b_hssd_distill", "s3_long_1m_lsa", "s2_joint_bptt", "s4_rl")
    cfgs = {st: load_config(os.path.join(ROOT, "configs", "stages", f"{st}.yaml")) for st in v3_stages}

    expected_init_chain = {
        "s1_loop_a6b": None,
        "s1b_hssd_distill": "runs/s1_loop_a6b",
        "s3_long_1m_lsa": "runs/s1b_hssd_distill",
        "s2_joint_bptt": "runs/s3_long_1m_lsa",
        "s4_rl": "runs/s2_joint_bptt",
    }

    for st, cfg in cfgs.items():
        loop = dict(cfg["loop"])
        assert loop["enabled"] is True, st
        assert loop["prelude_end"] == 12, st
        assert loop["core_end"] == 36, st
        assert loop["num_passes"] == 3, st
        assert loop["target_virtual_layers"] == 88, st
        assert loop["kv_share"] is True, st
        assert list(loop["bridge_init"]) == [0.05, 0.85, 0.10], st
        assert loop["bridge_mode"] == "subspace", st
        assert loop["bridge_rank"] == 256, st
        assert loop["bridge_up_std"] == 0.01, st
        assert loop["attn_adapter_rank"] == 64, st
        assert loop["step_embedding"] is True, st
        assert loop["gate_init"] == 0.0, st
        assert loop["per_pass_router"] is True, st
        assert loop["router_adapter_rank"] == 32, st
        assert loop["router_repel_init"] == 0.0, st
        assert loop["router_diversity_weight"] == 0.0, st
        assert loop["mor"] is False, st
        assert loop["write_mask"] is True, st
        assert loop["checkpoint"] is True, st
        assert cfg["train"].get("init_from") == expected_init_chain[st], st
        assert list(cfg["eval_quick"]["depths"]) == [1, 2, 3], st

    assert cfgs["s1_loop_a6b"]["gdn2"]["enabled"] is True
    assert cfgs["s1_loop_a6b"]["hssd"]["enabled"] is False

    expected_hssd = {
        "enabled": True,
        "chunk_size": 64,
        "window_chunks": 8,
        "num_pages": 8,
        "k_pages": 2,
        "theta_low_rank": True,
        "surprise_mode": "routed",
        "state_dtype": "bfloat16",
        "budget_weight": 1.0,
        "segment_chunks": 4,
        "checkpoint_segments": True,
    }
    for st in ("s1b_hssd_distill", "s3_long_1m_lsa", "s2_joint_bptt", "s4_rl"):
        cfg = cfgs[st]
        assert cfg["gdn2"]["enabled"] is False, st
        for hk, hv in expected_hssd.items():
            assert cfg["hssd"][hk] == hv, f"{st}: hssd.{hk}={cfg['hssd'].get(hk)!r} != {hv!r}"

    for st in ("s3_long_1m_lsa", "s2_joint_bptt", "s4_rl"):
        cfg = cfgs[st]
        assert cfg["model"]["yarn_target_len"] == 1048576, st
        assert not bool((cfg.get("rerope") or {}).get("enabled", False)), st
    assert cfgs["s3_long_1m_lsa"]["rerope"]["enabled"] is False

    for st in ("s2_joint_bptt", "s4_rl"):
        lsa = dict(cfgs[st]["lsa"])
        assert lsa["enabled"] is True and lsa["mode"] == "sparse", st
        assert lsa["num_query_passes"] == 3 and lsa["index_dim"] == 64 and lsa["index_on"] == "nope", st
        assert lsa["pool"] == 4 and lsa["block_size"] == 512, st
        assert lsa["avg_bucket_keys"] == 8 and lsa["max_bucket_keys"] == 32, st
        assert lsa["num_select"] == 64 and lsa["window"] == 512 and lsa["sinks"] == 64, st
        assert lsa["search"] == "flat" and lsa["temperature"] == 0.1 and lsa["query_chunk"] == 256, st

        exp = dict(cfgs[st]["experts"])
        assert exp["enabled"] is False, st
        assert exp["stats_path"] == "runs/s1_loop_a6b/esft_stats_looped.json", st
        assert exp["fraction"] == 0.025 and exp["by"] == "affinity" and exp["per_pass"] == "union" and exp["rank"] == 64

    s4 = cfgs["s4_rl"]
    assert list(s4["loop"]["train_pass_probs"]) == [0.10, 0.35, 0.55]
    assert s4["train"]["steps"] == 400 and s4["train"]["ckpt_every"] == 25
    assert s4["train"]["sha256_frozen_experts"] is True
    assert s4["train"]["train_loop_adapters"] is True
    assert s4["train"]["train_hssd"] is True
    assert s4["train"]["train_lsa"] is True
    assert s4["train"]["aux_weights"]["loop_aux"] == 1.0
    assert s4["train"]["aux_weights"]["hssd_budget"] == 0.5
    assert s4["train"]["optim"]["lr"] == 1.0e-6
    assert s4["train"]["optim"]["warmup_steps"] == 10
    assert s4["train"]["optim"]["max_grad_norm_per_param"] == 0.5
    assert s4["train"]["optim"]["lr_mult"]["loop_adapters"] == 5.0
    assert s4["train"]["optim"]["lr_mult"]["hssd_adapter"] == 2.0
    assert s4["train"]["optim"]["lr_mult"]["lsa_indexer"] == 2.0
    assert s4["rl"]["gate"]["depth"] == 3


def test_v3_cli_pipeline_cmds_autoskip_and_notebook_pack_priority(tmp_path, monkeypatch):
    from qloop.cli import _pipeline_stage_cmds, _run_sync_action
    from qloop.hub import PIPELINE_ORDER, V3_PIPELINE_ORDER, HubManifest

    def _extract_cfg(cmd: list[str]):
        c_idx = cmd.index("-c")
        cpath = cmd[c_idx + 1]
        ovs = [cmd[i + 1] for i, tok in enumerate(cmd[:-1]) if tok == "-s"]
        return load_config(cpath, ovs), ovs

    v3_all = (*V3_PIPELINE_ORDER, "eval")
    s4_v3_cmd = _pipeline_stage_cmds(
        "s4_rl",
        config_root=os.path.join(ROOT, "configs"),
        run_root=str(tmp_path),
        repo_id="TassP0lar/Qwen3.6-35B-A3B-Loop64",
        debug_level=1,
        overrides=[],
        time_budget_hours=None,
        scan_batches=4,
        scan_seq_len=512,
        all_stages=v3_all,
    )[0]
    s4_v3_cfg, _ = _extract_cfg(s4_v3_cmd)
    assert s4_v3_cfg.loop.prelude_end == 12 and s4_v3_cfg.loop.core_end == 36
    assert list(s4_v3_cfg.loop.train_pass_probs) == [0.10, 0.35, 0.55]
    assert s4_v3_cfg.hssd.enabled is True
    assert s4_v3_cfg.hssd.state_dtype == "bfloat16"
    assert s4_v3_cfg.hssd.segment_chunks == 4
    assert s4_v3_cfg.model.yarn_target_len == 1048576
    assert "norms" in s4_v3_cfg.train.allowed_groups and "deltanet" in s4_v3_cfg.train.allowed_groups

    eval_v3_cmd = _pipeline_stage_cmds(
        "eval",
        config_root=os.path.join(ROOT, "configs"),
        run_root=str(tmp_path),
        repo_id="TassP0lar/Qwen3.6-35B-A3B-Loop64",
        debug_level=1,
        overrides=[],
        time_budget_hours=None,
        scan_batches=4,
        scan_seq_len=512,
        all_stages=v3_all,
    )[0]
    eval_v3_cfg, _ = _extract_cfg(eval_v3_cmd)
    assert eval_v3_cfg.loop.prelude_end == 12 and eval_v3_cfg.loop.core_end == 36
    assert eval_v3_cfg.hssd.enabled is True and eval_v3_cfg.hssd.state_dtype == "bfloat16"
    assert eval_v3_cfg.hssd.segment_chunks == 4
    assert eval_v3_cfg.model.yarn_target_len == 1048576
    assert list(eval_v3_cfg["eval"]["defaults"]["depths"]) == [1, 2, 3]

    # Legacy v2 PIPELINE_ORDER compatibility check
    s4_v2_cmd = _pipeline_stage_cmds(
        "s4_rl",
        config_root=os.path.join(ROOT, "configs"),
        run_root=str(tmp_path),
        repo_id="TassP0lar/Qwen3.6-35B-A3B-Loop64",
        debug_level=1,
        overrides=[],
        time_budget_hours=None,
        scan_batches=4,
        scan_seq_len=512,
        all_stages=PIPELINE_ORDER,
    )[0]
    s4_v2_cfg, _ = _extract_cfg(s4_v2_cmd)
    assert s4_v2_cfg.loop.prelude_end == 8 and s4_v2_cfg.loop.core_end == 32
    assert s4_v2_cfg.loop.target_virtual_layers == 64
    assert s4_v2_cfg.gdn2.enabled is True and s4_v2_cfg.hssd.enabled is False
    assert s4_v2_cfg.rl.gate.depth == 2

    # Verify s0_probe auto-skips cleanly when remote HF manifest marks s0_probe completed
    subproc_calls: list[list[str]] = []
    monkeypatch.setattr("subprocess.call", lambda cmd, *a, **kw: subproc_calls.append(list(cmd)) or 0)

    class _ProbeDoneHub:
        def fetch_manifest(self) -> HubManifest:
            return HubManifest(stages={"s0_probe": {"status": "completed", "step": 1}})

    sync_args = SimpleNamespace(
        action="pipeline",
        stages="s0_probe,s1_loop_a6b",
        until="s0_probe",
        force=False,
        debug=1,
        set=[],
        time_budget_hours=None,
        scan_batches=2,
        scan_seq_len=64,
        reclaim_disk=False,
    )
    assert (
        _run_sync_action(
            sync_args,
            hub=_ProbeDoneHub(),
            repo_id="TassP0lar/Qwen3.6-35B-A3B-Loop64",
            run_root=str(tmp_path),
            ckpt_cls=None,
        )
        == 0
    )
    assert subproc_calls == [], "completed s0_probe in Hub manifest must auto-skip without spawning subprocess"

    # Verify colab_train_e2e.ipynb Cell 5 auto-selects highest completed stage for Whole Model Pack export
    e2e_nb_path = os.path.join(ROOT, "notebooks", "colab_train_e2e.ipynb")
    with open(e2e_nb_path, encoding="utf-8") as f:
        e2e_nb = json.load(f)
    cell5_src = next(
        (
            "".join(c.get("source", []))
            for c in e2e_nb["cells"]
            if c.get("cell_type") == "code" and "PACK_STAGE_PRIORITY" in "".join(c.get("source", []))
        ),
        "",
    )
    assert cell5_src, "missing PACK_STAGE_PRIORITY cell in colab_train_e2e.ipynb"
    cell5_tree = ast.parse(cell5_src, filename=f"{e2e_nb_path}#pack_cell", feature_version=(3, 9))
    pack_assigns = {
        node.targets[0].id: ast.literal_eval(node.value)
        for node in cell5_tree.body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id in ("PACK_SOURCE_STAGE", "PACK_SOURCE_CONFIG", "PACK_STAGE_PRIORITY")
    }
    assert pack_assigns["PACK_SOURCE_STAGE"] == "auto"
    assert pack_assigns["PACK_SOURCE_CONFIG"] == "auto"
    assert pack_assigns["PACK_STAGE_PRIORITY"] == (
        "s4_rl",
        "s2_joint_bptt",
        "s3_long_1m_lsa",
        "s1b_hssd_distill",
        "s1_loop_a6b",
    )


def test_v3_five_stage_toy_chain_init_from_and_strict_restore(tmp_path, monkeypatch):
    """End-to-end 5-stage Toy chain (`s1_loop_a6b -> s1b_hssd_distill -> s3_long_1m_lsa -> s2_joint_bptt -> s4_rl`).

    Exercises real stage YAMLs mapped onto 16L Toy (`prelude_end=4, core_end=12, num_passes=3, target_virtual=32`),
    verifying every `train.init_from` transition via `LMTrainer` / `GRPOTrainer`, `carried_params` HSSD preservation
    through `s3_long_1m_lsa` (`train_hssd: false`), `GRPOTrainer` `aux_weights` + `sha256_frozen_experts`, and exact
    `restore_checkpoint(..., strict=True)` forward parity across `R in {1, 2, 3}` at every stage.
    """
    import torch

    from qloop.train.checkpoint import CheckpointManager
    from qloop.train.rl import GRPOTrainer
    from qloop.train.stack import build_stack
    from qloop.train.trainer import LMTrainer

    monkeypatch.setenv("QLOOP_RUN_ROOT", str(tmp_path))

    common_toy_overrides = [
        "model.name=toy",
        "model.dtype=float32",
        "model.device_map=cpu",
        "model.toy={seed: 0}",
        "loop.prelude_end=4",
        "loop.core_end=12",
        "loop.num_passes=3",
        "loop.target_virtual_layers=32",
        "loop.bridge_rank=16",
        "loop.attn_adapter_rank=8",
        "loop.router_adapter_rank=8",
        "gdn2.rank=8",
        "gdn2.chunk_size=16",
        "gdn2.segment_chunks=2",
        "hssd.chunk_size=16",
        "hssd.window_chunks=2",
        "hssd.num_pages=4",
        "hssd.k_pages=2",
        "hssd.segment_chunks=2",
        "lsa.index_dim=4",
        "lsa.block_size=16",
        "lsa.pool=2",
        "lsa.avg_bucket_keys=4",
        "lsa.max_bucket_keys=8",
        "lsa.window=16",
        "lsa.sinks=2",
        "lsa.num_select=2",
        "lsa.query_chunk=16",
        "checkpoint.sync.enabled=false",
        "checkpoint.sync.hf_repo=null",
        "train.steps=2",
        "train.grad_accum=1",
        "train.log_every=1",
        "train.ckpt_every=2",
        "train.eval_every=0",
        "train.optim.lr=3.0e-3",
        "train.optim.warmup_steps=1",
        "train.optim.apollo={min_dim: 16}",
        "eval_quick.n_batches=1",
    ]
    lm_data_overrides = [
        "data.seq_len=32",
        "data.sources=[{name: toy_cot, weight: 0.7}, {name: toy_text, weight: 0.3}]",
        "data.decontaminate.enabled=false",
    ]

    probe_ids = torch.randint(4, 90, (1, 32))

    def _verify_strict_restore_parity(stage_cfg, stage_dir: str, trained_stack) -> dict[str, torch.Tensor]:
        trained_stack.model.eval()
        if trained_stack.lsa is not None:
            trained_stack.lsa.set_mode("sparse")
        expected: dict[int, torch.Tensor] = {}
        with torch.no_grad():
            for r in (1, 2, 3):
                trained_stack.loop.set_active_passes(r)
                expected[r] = trained_stack.model(input_ids=probe_ids).logits.detach().clone()

        mgr = CheckpointManager(stage_dir, keep_last=0)
        ck = mgr.load(mgr.latest(), map_location="cpu")
        assert ck is not None, f"missing checkpoint in {stage_dir}"
        smeta = (ck.get("state") or {}).get("stack") or {}

        fresh = build_stack(stage_cfg)
        fresh.restore_checkpoint(ck["params"], strict=True, prior_stack_meta=smeta)
        fresh.model.eval()
        if fresh.lsa is not None:
            fresh.lsa.set_mode("sparse")
        with torch.no_grad():
            for r in (1, 2, 3):
                fresh.loop.set_active_passes(r)
                got = fresh.model(input_ids=probe_ids).logits
                assert torch.allclose(got, expected[r], atol=1e-5), f"{stage_dir}: strict=True depth-{r} mismatch"
        return ck["params"]

    # 1. Stage s1_loop_a6b (Track-A GDN-2 + 3-pass Loop)
    s1_cfg = load_config(
        os.path.join(ROOT, "configs", "stages", "s1_loop_a6b.yaml"),
        [*common_toy_overrides, *lm_data_overrides],
    )
    s1_dir = str(tmp_path / "s1_loop_a6b")
    s1_tr = LMTrainer(s1_cfg, s1_dir)
    assert s1_tr.train()["step"] == 2
    _verify_strict_restore_parity(s1_cfg, s1_dir, s1_tr.stack)

    # 2. Stage s1b_hssd_distill (init_from: runs/s1_loop_a6b -> swaps GDN-2 for Track-B HSSD v3.1)
    s1b_cfg = load_config(
        os.path.join(ROOT, "configs", "stages", "s1b_hssd_distill.yaml"),
        [*common_toy_overrides, *lm_data_overrides],
    )
    assert s1b_cfg.train.init_from == "runs/s1_loop_a6b"
    s1b_dir = str(tmp_path / "s1b_hssd_distill")
    s1b_tr = LMTrainer(s1b_cfg, s1b_dir)
    assert s1b_tr.train()["step"] == 2
    assert s1b_tr.stack.carried_gdn2_cfg is not None
    s1b_params = _verify_strict_restore_parity(s1b_cfg, s1b_dir, s1b_tr.stack)
    s1b_hssd_tensors = {k: v.clone() for k, v in s1b_params.items() if "hssd_adapter." in k}
    assert s1b_hssd_tensors, "s1b_hssd_distill must save trained hssd_adapter tensors"

    # 3. Stage s3_long_1m_lsa (init_from: runs/s1b_hssd_distill; train_hssd: false carries HSSD; trains D-LSA + YaRN)
    s3_cfg = load_config(
        os.path.join(ROOT, "configs", "stages", "s3_long_1m_lsa.yaml"),
        [*common_toy_overrides, *lm_data_overrides],
    )
    assert s3_cfg.train.init_from == "runs/s1b_hssd_distill"
    assert s3_cfg.train.train_hssd is False
    assert s3_cfg.model.yarn_target_len == 1048576
    assert s3_cfg.rerope.enabled is False
    s3_dir = str(tmp_path / "s3_long_1m_lsa")
    s3_tr = LMTrainer(s3_cfg, s3_dir)
    assert s3_tr.train()["step"] == 2
    assert any("hssd_adapter." in k for k in s3_tr.stack.carried_params), "s3 must carry frozen hssd_adapter params"
    s3_params = _verify_strict_restore_parity(s3_cfg, s3_dir, s3_tr.stack)
    for hk, hv in s1b_hssd_tensors.items():
        assert hk in s3_params and torch.equal(s3_params[hk].cpu(), hv.cpu()), f"s3 dropped or mutated {hk}"

    # 4. Stage s2_joint_bptt (init_from: runs/s3_long_1m_lsa; joint HSSD + sparse D-LSA + Loop)
    s2_cfg = load_config(
        os.path.join(ROOT, "configs", "stages", "s2_joint_bptt.yaml"),
        [*common_toy_overrides, *lm_data_overrides],
    )
    assert s2_cfg.train.init_from == "runs/s3_long_1m_lsa"
    s2_dir = str(tmp_path / "s2_joint_bptt")
    s2_tr = LMTrainer(s2_cfg, s2_dir)
    assert s2_tr.train()["step"] == 2
    _verify_strict_restore_parity(s2_cfg, s2_dir, s2_tr.stack)

    # 5. Stage s4_rl (init_from: runs/s2_joint_bptt; GRPOTrainer with HSSD v3.1 + sparse D-LSA + depth-3 gate)
    rl_toy_overrides = [
        *common_toy_overrides,
        'rl.tasks=[{source: toy_arith, n: 16, verifier: regex, pattern: "[aeiou]"}]',
        "rl.group_size=4",
        "rl.prompts_per_step=2",
        "rl.max_new_tokens=6",
        "rl.micro_batch_seqs=2",
        "rl.verifier_workers=1",
        "rl.gate.every=2",
        (
            'rl.gate.benchmarks=[{name: aime_2026, limit: 2, max_new_tokens: 4, '
            'offline_items: [{id: a, problem: "1+1?", answer: "2"}, {id: b, problem: "2+2?", answer: "4"}]}]'
        ),
    ]
    s4_cfg = load_config(
        os.path.join(ROOT, "configs", "stages", "s4_rl.yaml"),
        rl_toy_overrides,
    )
    assert s4_cfg.train.init_from == "runs/s2_joint_bptt"
    assert s4_cfg.rl.gate.depth == 3
    assert s4_cfg.train.aux_weights.loop_aux == 1.0 and s4_cfg.train.aux_weights.hssd_budget == 0.5
    s4_dir = str(tmp_path / "s4_rl")
    s4_tr = GRPOTrainer(s4_cfg, s4_dir)
    s4_res = s4_tr.train()
    assert s4_res["step"] == 2
    assert os.path.exists(os.path.join(s4_dir, "checkpoints", "PROMOTED"))
    s4_tr.stack.verify_frozen_experts()
    _verify_strict_restore_parity(s4_cfg, s4_dir, s4_tr.stack)
