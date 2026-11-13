import os

import pytest
import torch

from qloop.model.introspect import inspect_model
from qloop.model.probes import candidate_specs, probe_backward, probe_core_ranges, surgery_check
from qloop.model.toy import build_toy
from qloop.train.checkpoint import CheckpointManager, collect_trainable, restore_trainable, stable_names


def _batches(n=2, t=24):
    g = torch.Generator().manual_seed(0)
    return [torch.randint(0, 97, (1, t), generator=g) for _ in range(n)]


def test_candidate_specs_for_qwen36_64_layers():
    types = ["full_attention" if (i + 1) % 4 == 0 else "linear_attention" for i in range(40)]
    specs = candidate_specs(types, 64)
    assert specs, "there must be 64-layer candidates"
    for s in specs:
        assert s.virtual_layers(40) == 64 and s.prelude_end % 4 == 0 and s.core_end % 4 == 0
    assert any(s.prelude_end == 8 and s.core_end == 32 and s.num_passes == 2 for s in specs)
    assert any(s.num_passes == 3 and s.core_len == 12 for s in specs)


def test_probe_backward_and_core_ranges_on_toy():
    model = build_toy()
    for p in model.parameters():
        p.requires_grad_(False)
    rep = probe_backward(model, _batches(1)[0])
    assert rep["ok"] and not rep["dead_layers"]
    rows = probe_core_ranges(model, _batches(), target_virtual=24, passes_options=(2, 3))
    assert rows and all("loss_increase" in r for r in rows)
    assert inspect_model(model).num_layers == 16  # probes restore the model


def test_surgery_check_on_toy():
    model = build_toy()
    lsa = {"index_dim": 4, "block_size": 8, "pool": 2, "avg_bucket_keys": 4, "max_bucket_keys": 8, "window": 8,
           "sinks": 2, "num_select": 2}
    rep = surgery_check(model, _batches(1, t=30)[0], {"prelude_end": 4, "core_end": 12, "num_passes": 2,
                                                      "target_virtual_layers": 24}, lsa, decode_steps=4, atol=1e-4)
    assert rep["init_identity_maxdiff"] == 0.0 and rep["decode_vs_full_maxdiff"] < 1e-4


def test_checkpoint_roundtrip_rotation_and_promotion(tmp_path):
    model = build_toy()
    names = stable_names(model)
    named = {names[id(p)]: p for p in model.parameters()}
    for n, p in named.items():
        p.requires_grad_(n.startswith("model.layers.0."))
    mgr = CheckpointManager(str(tmp_path), keep_last=2)
    for step in (1, 2, 3):
        with torch.no_grad():
            named["model.layers.0.mlp.gate.weight"].fill_(float(step))
        path = mgr.save(step, collect_trainable(named), {"dummy": step}, {"stage": "test"})
        if step == 1:
            mgr.promote(path)
    dirs = sorted(d for d in os.listdir(tmp_path / "checkpoints") if d.startswith("step_"))
    assert dirs in (["step_0000001", "step_0000002", "step_0000003"], ["step_0000001", "step_0000003"])
    assert "step_0000001" in dirs  # promoted checkpoint survives rotation
    ck = mgr.load()
    assert ck["state"]["step"] == 3 and ck["optim"] == {"dummy": 3}
    with torch.no_grad():
        named["model.layers.0.mlp.gate.weight"].zero_()
    restore_trainable(named, ck["params"])
    assert float(named["model.layers.0.mlp.gate.weight"].detach().mean()) == 3.0
    with pytest.raises(RuntimeError, match="unexpected"):
        restore_trainable(named, {**ck["params"], "bogus": torch.zeros(1)})
