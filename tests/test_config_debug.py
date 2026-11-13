import os

import pytest

from qloop.config import load_config
from qloop.debug import crash_guard, describe, require, setup_run, stage


def _write(path, text):
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def test_config_inheritance_and_overrides(tmp_path):
    _write(tmp_path / "base.yaml", "a: 1\nb: {x: 1, y: 2}\nlst: [1, 2]\nlr: 3e-5\nname: cuda:0\n")
    _write(tmp_path / "child.yaml", "_base_: base.yaml\nb: {y: 3}\nlst: [9]\n")
    cfg = load_config(str(tmp_path / "child.yaml"), overrides=["b.z=4", "a=1e-4", "new.k=true"])
    assert cfg.lr == pytest.approx(3e-5) and cfg.name == "cuda:0"
    assert cfg.b.x == 1 and cfg.b.y == 3 and cfg.b.z == 4
    assert cfg.lst == [9]
    assert cfg.a == pytest.approx(1e-4)
    assert cfg.new.k is True
    assert cfg.get_path("b.y") == 3 and cfg.get_path("nope.x", 7) == 7
    with pytest.raises(AttributeError, match="Available"):
        _ = cfg.missing


def test_circular_base_detected(tmp_path):
    _write(tmp_path / "a.yaml", "_base_: b.yaml\n")
    _write(tmp_path / "b.yaml", "_base_: a.yaml\n")
    with pytest.raises(ValueError, match="circular"):
        load_config(str(tmp_path / "a.yaml"))


def test_crash_guard_writes_report(tmp_path):
    import torch

    setup_run(str(tmp_path), level=1)
    with pytest.raises(RuntimeError), crash_guard(str(tmp_path)), stage("unit"):
        bad = torch.tensor([1.0, float("nan")])  # noqa: F841 - should appear in report locals
        raise RuntimeError("boom nonfinite")
    reports = os.listdir(tmp_path / "crash")
    assert len(reports) == 1
    text = (tmp_path / "crash" / reports[0]).read_text()
    assert "boom" in text and "NONFINITE=1" in text and "unit" in text and "Hints" in text


def test_describe_and_require():
    import torch

    assert "Tensor(2, 3)" in describe(torch.zeros(2, 3))
    with pytest.raises(AssertionError, match="shape"):
        require(False, "shape mismatch", t=torch.zeros(1))
