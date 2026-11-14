"""Every shipped config must load and validate; the CLI must drive the toy pipeline end-to-end."""
from __future__ import annotations

import ast
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
