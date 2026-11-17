"""GRPO trainer end-to-end on the toy model (offline)."""
import os

import pytest

from qloop.debug import setup_run
from qloop.train.rl import GRPOTrainer, RLTask, TaskSampler, group_advantages, verify


def _cfg(steps, gate_every=0):
    return {
        "seed": 0,
        "stage": "rl-test",
        "model": {"name": "toy", "dtype": "float32", "device_map": "cpu", "toy": {"seed": 0}},
        "loop": {"enabled": True, "prelude_end": 4, "core_end": 12, "num_passes": 2, "target_virtual_layers": 24},
        "train": {"steps": steps, "ckpt_every": 2, "trainable": [r"mlp\.gate\.weight$"],
                  "optim": {"lr": 1e-2, "apollo": {"min_dim": 16}}},
        "rl": {
            "tasks": [{"source": "toy_arith", "n": 16, "verifier": "regex", "pattern": "[aeiou]"}],
            "group_size": 4, "prompts_per_step": 2, "max_new_tokens": 6, "temperature": 1.0,
            "micro_batch_seqs": 3, "kl_coef": 0.05, "ppo_epochs": 2,
            "overlong": {"enabled": True, "buffer": 2, "penalty": 0.5},
            "gate": {"every": gate_every, "benchmarks": [{
                "name": "aime_2026", "limit": 2, "max_new_tokens": 4,
                "offline_items": [{"id": "a", "problem": "1+1?", "answer": "2"},
                                  {"id": "b", "problem": "2+2?", "answer": "4"}]}]},
        },
    }


def test_group_advantages_and_verifiers():
    assert group_advantages([1.0, 1.0, 1.0]) is None
    adv = group_advantages([1.0, 0.0, 0.0, 1.0])
    assert adv is not None and abs(sum(adv)) < 1e-9 and adv[0] > 0 > adv[1]
    assert group_advantages([2.0, 0.0], norm="none") == [1.0, -1.0]
    t = RLTask("x", "p", "scorer:math", {"answer": "42"})
    assert verify(t, "so the answer is \\boxed{42}")[0] == 1.0
    assert verify(RLTask("y", "p", "regex", {"pattern": "^ok"}), "ok!")[0] == 1.0
    with pytest.raises(ValueError):
        verify(RLTask("z", "p", "bogus", {}), "")


def test_task_sampler_resume_is_exact():
    groups = [("a", 1.0, [RLTask(str(i), "p", "regex", {"pattern": "x"}, "a") for i in range(5)]),
              ("b", 2.0, [RLTask(f"b{i}", "p", "regex", {"pattern": "x"}, "b") for i in range(3)])]
    s1 = TaskSampler(groups, seed=1)
    s1.next(7)
    state = s1.state_dict()
    want = [t.task_id for t in s1.next(9)]
    s2 = TaskSampler(groups, seed=1)
    s2.load_state_dict(state)
    assert [t.task_id for t in s2.next(9)] == want


def test_grpo_end_to_end_with_gate_and_resume(tmp_path):
    run_dir = str(tmp_path / "rl")
    setup_run(run_dir, level=1)
    tr = GRPOTrainer(_cfg(steps=3, gate_every=2), run_dir)
    res = tr.train()
    assert res["step"] == 3
    assert os.path.exists(os.path.join(run_dir, "checkpoints", "PROMOTED"))
    assert "reward/mean" in res["last"] and "temperature" in res["last"]
    tr2 = GRPOTrainer(_cfg(steps=4), run_dir)
    res2 = tr2.train()
    assert res2["step"] == 4


def test_grpo_nonfused_zero_grad_after_update(tmp_path):
    from qloop.train.rl import _Seq

    run_dir = str(tmp_path / "rl_nonfused")
    setup_run(run_dir, level=1)
    cfg = _cfg(steps=1, gate_every=0)
    cfg["train"]["optim"]["fused"] = False
    cfg["train"]["optim"]["max_grad_norm"] = 0.5
    cfg["rl"]["ppo_epochs"] = 2
    tr = GRPOTrainer(cfg, run_dir)
    seqs = [
        _Seq([1, 2, 3, 4, 5, 6], 2, 1.0, "pg"),
        _Seq([1, 2, 7, 8, 9, 10], 2, -1.0, "pg"),
    ]
    st1 = tr.update(seqs)
    assert st1["optim/optim/stepped_params"] > 0
    assert all(p.grad is None for p in tr.stack.named_trainable.values())

    # Inject a one-shot Inf grad during epoch 1 and verify epoch 2 in the same update() recovers cleanly!
    target_p = next(iter(tr.stack.named_trainable.values()))
    fired = {"done": False}

    def _poison(grad):
        if not fired["done"] and tr.opt._boundary:
            fired["done"] = True
            return grad * float("inf")
        return grad

    h = target_p.register_hook(_poison)
    try:
        st2 = tr.update(seqs)
    finally:
        h.remove()
    assert st2["optim/optim/skipped_params"] == 0.0
    assert st2["optim/optim/stepped_params"] > 0
    assert all(p.grad is None for p in tr.stack.named_trainable.values())
