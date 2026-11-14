"""Every shipped config must load and validate; the CLI must drive the toy pipeline end-to-end."""
import glob
import json
import os

import pytest

from qloop.cli import main
from qloop.config import load_config
from qloop.model.gdn2 import GDN2Config
from qloop.model.hssd import HSSDConfig
from qloop.model.loop import LoopSpec, build_schedule
from qloop.model.lsa import LSAConfig

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
QWEN_TYPES = ["full_attention" if (i + 1) % 4 == 0 else "linear_attention" for i in range(40)]


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
        if "stages/" in path.replace("\\", "/") or "eval/" in path.replace("\\", "/"):
            assert loop.per_pass_router is True
            assert (cfg.get("gdn2") or {}).get("enabled") is True or (cfg.get("hssd") or {}).get("enabled") is True
            sync = (cfg.get("checkpoint") or {}).get("sync") or {}
            assert sync.get("hf_repo") == "TassP0lar/Qwen3.6-35B-A3B-Loop64"
            assert sync.get("private") is False
    assert loop.validate(types) == []
    assert len(build_schedule(loop, types)) == loop.virtual_layers(len(types))
    if cfg.get("stage_type") in ("lm", "rl"):
        assert cfg.get("train"), f"{path}: training stage without train section"


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
