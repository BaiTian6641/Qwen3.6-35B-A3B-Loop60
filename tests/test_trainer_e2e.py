"""End-to-end: build_stack + LMTrainer on the toy model with every component enabled."""
import json
import os

import torch

from qloop.debug import setup_run
from qloop.model.experts import scan_router_affinity
from qloop.model.toy import build_toy
from qloop.train.trainer import LMTrainer

LSA = {"enabled": True, "mode": "distill", "index_dim": 4, "block_size": 16, "pool": 2, "avg_bucket_keys": 4,
       "max_bucket_keys": 8, "window": 16, "sinks": 2, "num_select": 2, "distill_query_frac": 0.5}


def _cfg(run_dir, stats_path, steps):
    return {
        "seed": 0,
        "stage": "test",
        "model": {"name": "toy", "dtype": "float32", "device_map": "cpu", "toy": {"seed": 0}},
        "lsa": LSA,
        "experts": {"enabled": True, "stats_path": stats_path, "per_layer_k": 2},
        "loop": {"enabled": True, "prelude_end": 4, "core_end": 12, "num_passes": 2, "target_virtual_layers": 24,
                 "train_pass_probs": [0.25, 0.75], "checkpoint": True},
        "train": {"steps": steps, "grad_accum": 2, "micro_batch": 1, "log_every": 1, "ckpt_every": 3,
                  "eval_every": 3, "trainable": [r"mlp\.gate\.weight$", r"self_attn\.o_proj"],
                  "optim": {"lr": 3e-3, "warmup_steps": 1, "apollo": {"min_dim": 16}}},
        "data": {"seq_len": 64, "sources": [{"name": "toy_cot", "weight": 0.7}, {"name": "toy_text", "weight": 0.3}]},
        "eval_quick": {"n_batches": 2, "depths": [1, 2]},
        "_meta_": {"config_path": "inline"},
    }


def _records(run_dir):
    with open(os.path.join(run_dir, "metrics.jsonl"), encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def _write_stats(tmp_path):
    from qloop.data.tokenizer import ToyTokenizer

    model = build_toy(vocab_size=ToyTokenizer().vocab_size)
    batches = [torch.randint(0, 200, (1, 32)) for _ in range(2)]
    stats = scan_router_affinity(model, batches)
    path = str(tmp_path / "esft.json")
    stats.to_json(path)
    return path


def test_trainer_end_to_end_and_resume(tmp_path):
    run_dir = str(tmp_path / "run")
    setup_run(run_dir, level=1)
    stats_path = _write_stats(tmp_path)
    tr = LMTrainer(_cfg(run_dir, stats_path, steps=4), run_dir)
    names = list(tr.stack.named_trainable)
    assert any(n.startswith("loop_adapters.") for n in names)
    assert any("lsa_indexer" in n for n in names)
    assert any("expert_deltas" in n for n in names)
    assert any(n.endswith("mlp.gate.weight") for n in names)
    before = {n: p.detach().clone() for n, p in tr.stack.named_trainable.items()}
    res = tr.train()
    assert res["step"] == 4
    changed = [n for n, p in tr.stack.named_trainable.items() if not torch.equal(before[n], p.detach())]
    assert any(n.startswith("loop_adapters.gates") for n in changed), "loop gate must learn from zero init"
    assert any("lsa_indexer" in n for n in changed) and any("expert_deltas" in n for n in changed)
    recs = _records(run_dir)
    assert any("eval/loss_d2" in r for r in recs) and any("loss/lsa_kl" in r for r in recs)
    assert all(r.get("loss/lm", 1.0) == r.get("loss/lm", 1.0) for r in recs)  # no NaN

    tr2 = LMTrainer(_cfg(run_dir, stats_path, steps=6), run_dir)
    assert tr2.step == 0
    res2 = tr2.train()
    assert res2["step"] == 6
    recs2 = _records(run_dir)
    assert max(r["step"] for r in recs2) == 6 and any(r["step"] == 5 for r in recs2)
