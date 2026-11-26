"""Tests for Hugging Face Hub commit-history sync, multi-stage cumulative handoff, and 230 GB disk guard."""
from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from typing import Any

import torch

from qloop.cli import main
from qloop.config import Cfg
from qloop.hub import (
    DEFAULT_HF_REPO,
    HubSyncManager,
    disk_report,
    load_qloop_from_hub,
    reclaim_disk_space,
    render_model_card,
)
from qloop.model.experts import scan_router_affinity
from qloop.train.stack import build_stack
from qloop.train.trainer import LMTrainer


@dataclass
class _CommitEntry:
    commit_id: str
    title: str
    created_at: str


class MockHfApi:
    """In-memory mock of `huggingface_hub.HfApi` supporting atomic commits & quota-squash simulation."""

    def __init__(self, *, fail_quota_once: bool = False):
        self.repos: dict[str, dict[str, Any]] = {}
        self.files: dict[tuple[str, str], bytes] = {}
        self.commits: list[_CommitEntry] = []
        self.squash_count = 0
        self.fail_quota_once = fail_quota_once

    def create_repo(self, repo_id: str, repo_type: str = "model", private: bool = False, exist_ok: bool = True) -> None:
        if repo_id not in self.repos:
            self.repos[repo_id] = {"repo_type": repo_type, "private": private}

    def list_repo_files(self, repo_id: str, repo_type: str = "model") -> list[str]:
        return sorted(path for (rid, path) in self.files if rid == repo_id)

    def list_repo_commits(self, repo_id: str, repo_type: str = "model") -> list[_CommitEntry]:
        return list(reversed(self.commits))

    def super_squash_history(self, repo_id: str, repo_type: str = "model") -> None:
        self.squash_count += 1

    def create_commit(self, repo_id: str, repo_type: str, operations: list[Any], commit_message: str) -> Any:
        if self.fail_quota_once:
            self.fail_quota_once = False
            raise RuntimeError("413 Payload Too Large: repository storage quota exceeded")
        return self._apply_ops(
            repo_id,
            [(op.path_in_repo, op.path_or_fileobj) for op in operations],
            commit_message,
        )

    def commit_bundle(self, repo_id: str, additions: list[tuple[str, str | bytes]], commit_message: str) -> str:
        if self.fail_quota_once:
            self.fail_quota_once = False
            # Exercise the `create_commit` quota-squash path when testing quota retry
            self.super_squash_history(repo_id=repo_id)
        info = self._apply_ops(repo_id, additions, commit_message)
        return info.oid

    def _apply_ops(self, repo_id: str, additions: list[tuple[str, str | bytes]], commit_message: str) -> Any:
        for repo_path, src in additions:
            if isinstance(src, (bytes, bytearray)):
                payload = bytes(src)
            else:
                with open(src, "rb") as f:
                    payload = f.read()
            self.files[(repo_id, repo_path)] = payload
        oid = f"deadbeef{len(self.commits) + 1:04d}"
        self.commits.append(_CommitEntry(oid, commit_message, f"2026-11-13T16:00:{len(self.commits):02d}Z"))

        class _Info:
            pass

        res = _Info()
        res.oid = oid
        return res

    def download_file(self, repo_id: str, path_in_repo: str, dst_path: str) -> bool:
        key = (repo_id, path_in_repo)
        if key not in self.files:
            return False
        os.makedirs(os.path.dirname(os.path.abspath(dst_path)), exist_ok=True)
        with open(dst_path, "wb") as f:
            f.write(self.files[key])
        return True


def _toy_base_cfg(sync_repo: str = DEFAULT_HF_REPO) -> dict[str, Any]:
    return {
        "seed": 7,
        "model": {"name": "toy", "dtype": "float32"},
        "loop": {
            "enabled": True,
            "prelude_end": 4,
            "core_end": 12,
            "num_passes": 2,
            "target_virtual_layers": 24,
            "per_pass_router": True,
            "router_adapter_rank": 8,
            "router_diversity_weight": 0.01,
            "kv_share": True,
            "checkpoint": False,
        },
        "data": {
            "seq_len": 32,
            "sources": [{"name": "toy_cot", "weight": 1.0}],
        },
        "eval_quick": {"n_batches": 1, "depths": [1, 2]},
        "checkpoint": {
            "sync": {
                "hf_repo": sync_repo,
                "private": False,
                "push_every_ckpt": True,
                "push_optim": True,
                "prune_local_after_push": True,
            }
        },
    }


def test_public_repo_cross_session_resume_and_model_card(tmp_path):
    api = MockHfApi()
    hub = HubSyncManager(repo_id=DEFAULT_HF_REPO, private=False, api=api)

    # Session 0: generate ESFT stats & push to Hub as s0_probe artifact
    s0_dir = tmp_path / "session1" / "s0_probe"
    s0_dir.mkdir(parents=True)
    base_st = build_stack(Cfg({"model": {"name": "toy", "dtype": "float32"}, "loop": {"enabled": False}}))
    stats = scan_router_affinity(base_st.model, [torch.randint(4, 100, (1, 24))])
    stats_path = str(s0_dir / "esft_stats.json")
    stats.to_json(stats_path)
    hub.push_artifacts("s0_probe", {"artifacts/s0_probe/esft_stats.json": stats_path}, completed=True)
    assert api.repos[DEFAULT_HF_REPO]["private"] is False
    assert hub.fetch_manifest().is_completed("s0_probe")

    # Session 1: start s1_loop_heal, pause early via ultra-tiny time budget after 1 step (target=3)
    run_root_1 = tmp_path / "session1"
    cfg_s1_part = Cfg({
        **_toy_base_cfg(),
        "stage": "s1_loop_heal",
        "experts": {"enabled": True, "stats_path": "runs/s0_probe/esft_stats.json", "fraction": 0.25, "rank": 4},
        "train": {
            "steps": 3,
            "micro_batch": 1,
            "grad_accum": 1,
            "log_every": 1,
            "ckpt_every": 10,
            "keep_last": 1,
            "time_budget_hours": 1e-9,  # stop cleanly right after step 1
            "trainable": ["mlp\\.gate\\.weight$", "shared_expert\\."],
            "optim": {"lr": 1e-3, "warmup_steps": 1},
        },
    })
    os.environ["QLOOP_RUN_ROOT"] = str(run_root_1)
    try:
        res1 = LMTrainer(cfg_s1_part, str(run_root_1 / "s1_loop_heal"), hub_api=api).train()
        assert res1["step"] == 1
        m1 = hub.fetch_manifest()
        assert not m1.is_completed("s1_loop_heal")
        assert m1.stage_step("s1_loop_heal") == 1
        assert m1.next_incomplete_stage() == "s1_loop_heal"

        # Session 2: brand-new empty Colab disk (`session2`) -> auto-pulls esft_stats + step-1 ckpt and completes step 3
        run_root_2 = tmp_path / "session2"
        os.environ["QLOOP_RUN_ROOT"] = str(run_root_2)
        cfg_s1_full = Cfg({
            **cfg_s1_part.to_dict(),
            "train": {**dict(cfg_s1_part["train"]), "time_budget_hours": 0},
        })
        res2 = LMTrainer(cfg_s1_full, str(run_root_2 / "s1_loop_heal"), hub_api=api).train()
        assert res2["step"] == 3
        m2 = hub.fetch_manifest()
        assert m2.is_completed("s1_loop_heal")
        assert m2.stage_step("s1_loop_heal") == 3
        assert m2.next_incomplete_stage() == "s1_rescan"

        # Completed stage strips optim.pt locally to save Colab disk
        step3_dir = run_root_2 / "s1_loop_heal" / "checkpoints" / "step_0000003"
        assert (step3_dir / "trainable.pt").is_file()
        assert not (step3_dir / "optim.pt").exists()

        # Model card README.md rendered and committed
        readme = api.files[(DEFAULT_HF_REPO, "README.md")].decode("utf-8")
        assert "Qwen3.6-35B-A3B-Loop64" in readme
        assert "`s1_loop_heal` | **completed** | `3/3`" in readme
        assert "Cross-Pass Dynamic Expert Re-Routing" in render_model_card(m2)
    finally:
        os.environ.pop("QLOOP_RUN_ROOT", None)


def test_cumulative_cross_stage_handoff_and_one_call_hub_loader(tmp_path, monkeypatch):
    """Adaptations from S1 (hot-expert deltas + shared_expert) + S2 (D-LSA indexers) must survive S3 -> Hub load."""
    api = MockHfApi()
    hub = HubSyncManager(repo_id=DEFAULT_HF_REPO, api=api)
    run_root = tmp_path / "runs"
    monkeypatch.setenv("QLOOP_RUN_ROOT", str(run_root))

    # Prepare S0 stats
    base_st = build_stack(Cfg({"model": {"name": "toy", "dtype": "float32"}, "loop": {"enabled": False}}))
    stats = scan_router_affinity(base_st.model, [torch.randint(4, 100, (1, 24))])
    s0_stats = run_root / "s0_probe" / "esft_stats.json"
    s0_stats.parent.mkdir(parents=True)
    stats.to_json(str(s0_stats))
    hub.push_artifacts("s0_probe", {"artifacts/s0_probe/esft_stats.json": str(s0_stats)}, completed=True)

    # Stage 1: train GDN-2 adapters + hot-expert deltas (rank=4) + shared_expert + loop_adapters
    cfg1 = Cfg({
        **_toy_base_cfg(),
        "stage": "s1_loop_heal",
        "gdn2": {"enabled": True, "rank": 6, "chunk_size": 16, "segment_chunks": 2},
        "experts": {"enabled": True, "stats_path": "runs/s0_probe/esft_stats.json", "per_layer_k": 2, "rank": 4},
        "train": {
            "steps": 2,
            "micro_batch": 1,
            "grad_accum": 1,
            "keep_last": 1,
            "trainable": ["shared_expert\\."],
            "optim": {"lr": 5e-3, "warmup_steps": 1},
        },
    })
    LMTrainer(cfg1, str(run_root / "s1_loop_heal"), hub_api=api).train()

    # Wipe local runs completely -> Stage 2 (`s3_long_lsa` analogue): GDN-2 frozen (`train_gdn2: False`), LSA on!
    shutil.rmtree(run_root)
    cfg2 = Cfg({
        **_toy_base_cfg(),
        "stage": "s3_long_lsa",
        "gdn2": {"enabled": True, "rank": 6, "chunk_size": 16, "segment_chunks": 2},
        "lsa": {"enabled": True, "mode": "distill", "index_dim": 16, "block_size": 16, "pool": 2,
                "avg_bucket_keys": 4, "max_bucket_keys": 8, "num_select": 4, "window": 16, "sinks": 4},
        "experts": {"enabled": False},
        "train": {
            "init_from": "runs/s1_loop_heal",
            "steps": 2,
            "micro_batch": 1,
            "grad_accum": 1,
            "keep_last": 1,
            "train_gdn2": False,
            "trainable": ["self_attn\\.(q_norm|k_norm)"],
            "optim": {"lr": 5e-3, "warmup_steps": 1},
        },
    })
    tr2 = LMTrainer(cfg2, str(run_root / "s3_long_lsa"), hub_api=api)
    tr2.train()

    # Wipe local runs again -> Stage 3: GDN-2 active, neither LSA nor experts attached, router + loop + GDN-2 trained!
    shutil.rmtree(run_root)
    cfg3 = Cfg({
        **_toy_base_cfg(),
        "stage": "s4_final",
        "gdn2": {"enabled": True, "rank": 6, "chunk_size": 16, "segment_chunks": 2},
        "lsa": {"enabled": False},
        "experts": {"enabled": False},
        "train": {
            "init_from": "runs/s3_long_lsa",
            "steps": 1,
            "micro_batch": 1,
            "grad_accum": 1,
            "keep_last": 1,
            "trainable": ["mlp\\.gate\\.weight$"],
            "optim": {"lr": 2e-3, "warmup_steps": 1},
        },
    })
    tr3 = LMTrainer(cfg3, str(run_root / "s4_final"), hub_api=api)
    tr3.train()

    # Load from Hub in 1 call: GDN-2 adapters, LSA indexers (from Stage 2), folded expert deltas & shared_expert
    # (from Stage 1), and router/loop_adapters (from Stage 3) must ALL be present!
    loaded = load_qloop_from_hub(
        DEFAULT_HF_REPO,
        overrides=["lsa.mode=distill"],  # dense forward so outputs match tr3 bit-close while keeping trained indexers
        api=api,
    )
    assert loaded.gdn2 is not None and loaded.gdn2.cfg.rank == 6
    assert loaded.lsa is not None
    assert loaded.loop is not None
    tr3.stack.model.eval()
    ids = torch.randint(4, 120, (1, 24))
    with torch.no_grad():
        h_ref = tr3.stack.hidden(ids)
        h_hub = loaded.hidden(ids)
    assert torch.allclose(h_ref, h_hub, atol=1e-5, rtol=1e-5)


def test_reclaim_disk_space_preserves_hf_hub_weights_and_esft_stats(tmp_path):
    run_root = tmp_path / "runs"
    hf_home = tmp_path / "hf_cache"
    # Simulate 70 GB base weights dir + transient xet/upload caches
    (hf_home / "hub" / "models--Qwen--Qwen3.6-35B-A3B").mkdir(parents=True)
    base_shard = hf_home / "hub" / "models--Qwen--Qwen3.6-35B-A3B" / "shard.safetensors"
    base_shard.write_bytes(b"W" * 4096)
    (hf_home / "xet").mkdir(parents=True)
    (hf_home / "xet" / "staging.bin").write_bytes(b"X" * 8192)
    (hf_home / "upload").mkdir(parents=True)
    (hf_home / "upload" / "chunk.bin").write_bytes(b"U" * 4096)

    # Old completed stage + active stage
    old_ckpt = run_root / "s1_loop_heal" / "checkpoints" / "step_0002000"
    old_ckpt.mkdir(parents=True)
    (old_ckpt / "trainable.pt").write_bytes(b"T" * 4096)
    (run_root / "s1_loop_heal" / "esft_stats_looped.json").write_text("{}")
    (run_root / "s1_loop_heal" / "metrics.jsonl").write_text("{}\n")

    cur_cdir = run_root / "s2_sft" / "checkpoints"
    for step_n in ("step_0000100", "step_0000200"):
        sd = cur_cdir / step_n
        sd.mkdir(parents=True)
        (sd / "trainable.pt").write_bytes(b"A" * 2048)
        (sd / "optim.pt").write_bytes(b"O" * 2048)
    (cur_cdir / "LATEST").write_text("step_0000200")

    res = reclaim_disk_space(str(run_root), keep_stages=("s2_sft",), clean_hf_staging=True, hf_home=str(hf_home))
    assert res["removed"]
    # Base model shard & ESFT stats untouched!
    assert base_shard.is_file()
    assert (run_root / "s1_loop_heal" / "esft_stats_looped.json").is_file()
    assert (run_root / "s1_loop_heal" / "metrics.jsonl").is_file()
    # Superseded checkpoints & HF xet/upload wiped!
    assert not (run_root / "s1_loop_heal" / "checkpoints").exists()
    assert not (cur_cdir / "step_0000100").exists()
    assert (cur_cdir / "step_0000200" / "trainable.pt").is_file()
    assert not (hf_home / "xet").exists()
    assert not (hf_home / "upload").exists()

    rep = disk_report(str(run_root), hf_home=str(hf_home))
    assert rep["total_gib"] > 0 and rep["hf_staging_cache_gib"] == 0.0


def test_quota_squash_and_cli_sync_commands(tmp_path, monkeypatch, capsys):
    api = MockHfApi(fail_quota_once=True)
    hub = HubSyncManager(repo_id=DEFAULT_HF_REPO, squash_on_quota=True, api=api)
    f = tmp_path / "probe.json"
    f.write_text('{"ok": true}')
    oid = hub.push_artifacts("s0_probe", {"artifacts/s0_probe/probe.json": str(f)}, completed=False)
    assert oid is not None
    assert api.squash_count == 1

    monkeypatch.setenv("QLOOP_RUN_ROOT", str(tmp_path / "runs"))
    assert main(["sync", "disk", "--clean"]) == 0
    out = capsys.readouterr().out
    assert "free_gib" in out


def test_colab_setup_env_requires_hf_token_and_configures_ssd_paths(tmp_path, monkeypatch):
    import pytest

    from qloop.colab import setup_env, tail_metrics

    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    monkeypatch.delenv("QLOOP_HF_REPO", raising=False)
    monkeypatch.delenv("QLOOP_RUN_ROOT", raising=False)
    monkeypatch.delenv("HF_HOME", raising=False)
    with pytest.raises(RuntimeError, match="HF_TOKEN is missing"):
        setup_env(
            hf_repo=DEFAULT_HF_REPO,
            run_root=str(tmp_path / "runs"),
            hf_home=str(tmp_path / "hf_cache"),
            require_hf_token=True,
        )

    monkeypatch.setenv("HF_TOKEN", "hf_test_dummy_token")
    found = setup_env(
        hf_repo=DEFAULT_HF_REPO,
        run_root=str(tmp_path / "runs"),
        hf_home=str(tmp_path / "hf_cache"),
    )
    assert found["HF_TOKEN"] == "set"
    assert os.environ["QLOOP_HF_REPO"] == DEFAULT_HF_REPO
    assert os.environ["QLOOP_RUN_ROOT"] == str(tmp_path / "runs")
    assert os.environ["HF_HOME"] == str(tmp_path / "hf_cache")

    s1_dir = tmp_path / "runs" / "s1_loop_heal"
    s1_dir.mkdir(parents=True, exist_ok=True)
    (s1_dir / "metrics.jsonl").write_text(
        '{"step": 10, "loss/lm": 1.2345, "eval/depth_gain": 0.0812, "loop/reroute_rate": 0.625}\n'
    )
    tail_metrics("s1_loop_heal")


def test_colab_pep420_namespace_shadow_recovery(tmp_path):
    """Reproduce Colab's `/content/qloop` PEP 420 namespace-shadow bug and verify Cell 2 recovery."""
    import importlib
    import sys

    import pytest

    content_dir = tmp_path / "content"
    shadow_repo = content_dir / "qloop"
    (shadow_repo / "src").mkdir(parents=True)

    real_src = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
    saved_path = list(sys.path)
    saved_mods = {k: v for k, v in sys.modules.items() if k == "qloop" or k.startswith("qloop.")}
    try:
        for k in list(saved_mods):
            del sys.modules[k]
        sys.path = [str(content_dir)] + [p for p in saved_path if os.path.abspath(p) != real_src]
        importlib.invalidate_caches()

        ns_mod = importlib.import_module("qloop")
        assert getattr(ns_mod, "__file__", None) is None
        with pytest.raises(ModuleNotFoundError, match="qloop\\.colab"):
            importlib.import_module("qloop.colab")

        # Execute the exact Cell 2 recovery bootstrap
        if real_src not in sys.path:
            sys.path.insert(0, real_src)
        if "qloop" in sys.modules and getattr(sys.modules["qloop"], "__file__", None) is None:
            for k in [m for m in sys.modules if m == "qloop" or m.startswith("qloop.")]:
                del sys.modules[k]
        importlib.invalidate_caches()

        colab_mod = importlib.import_module("qloop.colab")
        assert callable(colab_mod.setup_env)
        assert getattr(sys.modules["qloop"], "__file__", None) is not None
    finally:
        for k in [m for m in list(sys.modules) if m == "qloop" or m.startswith("qloop.")]:
            del sys.modules[k]
        sys.modules.update(saved_mods)
        sys.path = saved_path
        importlib.invalidate_caches()


def test_hf_loading_tf5_dtype_and_missing_keys_multimodal_fallback(monkeypatch):
    """Verify `_load_hf` uses `dtype=` on TF>=5, catches silent missing decoder keys, and prunes `.visual`."""
    import sys
    import types

    from torch import nn

    from qloop.model.loading import _load_hf

    seen_calls: list[tuple[str, dict]] = []

    class DummyMultimodalInner(nn.Module):
        def __init__(self):
            super().__init__()
            self.visual = nn.Linear(8, 8)
            self.language_model = nn.Linear(8, 8)

    class DummyMultimodalWrapper(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = DummyMultimodalInner()

    fake_tf = types.ModuleType("transformers")
    fake_tf.__version__ = "5.3.0"

    class FakeAutoConfig:
        @staticmethod
        def from_pretrained(_name, **_kw):
            cfg = types.SimpleNamespace(model_type="qwen3_6_moe", text_config=types.SimpleNamespace())
            return cfg

    class FakeAutoCausal:
        @staticmethod
        def from_pretrained(_name, **kw):
            seen_calls.append(("causal", kw))
            return nn.Linear(4, 4), {"missing_keys": ["model.layers.0.self_attn.q_proj.weight"]}

    class FakeAutoVL:
        @staticmethod
        def from_pretrained(_name, **kw):
            seen_calls.append(("vl", kw))
            return DummyMultimodalWrapper(), {"missing_keys": []}

    class FakeAutoTok:
        @staticmethod
        def from_pretrained(_name, **_kw):
            return "tok"

    fake_tf.AutoConfig = FakeAutoConfig
    fake_tf.AutoModelForCausalLM = FakeAutoCausal
    fake_tf.AutoModelForImageTextToText = FakeAutoVL
    fake_tf.AutoTokenizer = FakeAutoTok
    monkeypatch.setitem(sys.modules, "transformers", fake_tf)

    model, tok = _load_hf({"name": "Qwen/Qwen3.6-35B-A3B", "dtype": "bfloat16", "text_only": True})
    assert tok == "tok"
    assert [c[0] for c in seen_calls] == ["causal", "vl"]
    assert "dtype" in seen_calls[0][1] and "torch_dtype" not in seen_calls[0][1]
    assert seen_calls[0][1]["output_loading_info"] is True
    assert model.model.visual is None
    assert not any("visual" in n for n, _ in model.named_parameters())


def test_mid_stage_distill_resume_pulls_and_preserves_teacher_ckpt(tmp_path, monkeypatch):
    """Regression test: resuming `s1b_hssd_distill` mid-stage must still pull & preserve `s1_loop_a6b`
    (`trainable.pt` + `state.json`, without `optim.pt`) so `SharedBackboneTeacher` loads frozen teacher
    weights instead of snapshotting mid-stage student `loop_adapters`.
    """
    from qloop.train.distill import SharedBackboneTeacher

    api = MockHfApi()
    hub = HubSyncManager(repo_id=DEFAULT_HF_REPO, api=api)

    loop_cfg = {
        "enabled": True,
        "prelude_end": 4,
        "core_end": 12,
        "num_passes": 3,
        "target_virtual_layers": 32,
        "bridge_mode": "subspace",
        "bridge_rank": 8,
        "bridge_init": [0.05, 0.85, 0.10],
        "attn_adapter_rank": 4,
        "router_adapter_rank": 4,
        "per_pass_router": True,
        "kv_share": True,
        "checkpoint": False,
    }
    sync_cfg = {
        "hf_repo": DEFAULT_HF_REPO,
        "private": False,
        "push_every_ckpt": True,
        "push_optim": True,
        "prune_local_after_push": True,
    }

    # 1. Complete `s1_loop_a6b` and push to Hub (including optim.pt on remote so we verify include_optim=False)
    run_root_1 = tmp_path / "session1"
    monkeypatch.setenv("QLOOP_RUN_ROOT", str(run_root_1))
    s1_cfg = Cfg({
        "seed": 11,
        "stage": "s1_loop_a6b",
        "model": {"name": "toy", "dtype": "float32", "device_map": "cpu", "toy": {"seed": 11}},
        "loop": loop_cfg,
        "gdn2": {"enabled": False},
        "hssd": {"enabled": False},
        "lsa": {"enabled": False},
        "experts": {"enabled": False},
        "data": {"seq_len": 16, "sources": [{"name": "toy_text", "weight": 1.0}]},
        "checkpoint": {"sync": sync_cfg},
        "train": {
            "steps": 1,
            "micro_batch": 1,
            "grad_accum": 1,
            "ckpt_every": 1,
            "eval_every": 0,
            "keep_last": 1,
            "train_loop_adapters": True,
            "train_gdn2": False,
            "trainable": [r"input_layernorm|post_attention_layernorm"],
            "allowed_groups": ["loop_adapters", "norms"],
            "optim": {"lr": 5e-3, "warmup_steps": 1, "impl": "builtin"},
        },
    })
    s1_tr = LMTrainer(s1_cfg, str(run_root_1 / "s1_loop_a6b"), hub_api=api)
    s1_tr.train()
    assert (DEFAULT_HF_REPO, "stages/s1_loop_a6b/latest/optim.pt") in api.files

    ids = torch.randint(4, 100, (1, 16))
    s1_tr.stack.model.eval()
    assert s1_tr.stack.loop is not None
    s1_tr.stack.loop.set_active_passes(2)
    with torch.no_grad():
        expected_teacher_h = s1_tr.stack.hidden(ids).detach().clone()

    # 2. Simulate `s1b_hssd_distill` reaching step 250 with drifted student `loop_adapters` + `norms`,
    #    then saving/pushing step 250 (which prunes `run_root_1/s1_loop_a6b/checkpoints` locally!).
    s1b_dict = {
        **s1_cfg.to_dict(),
        "stage": "s1b_hssd_distill",
        "hssd": {
            "enabled": True,
            "write_mode": "full",
            "swa_recency_init": 0.05,
            "blend_init": 0.5,
            "blend_warmup_steps": 4,
            "compact_active_pages": True,
            "compact_decode_cache": True,
            "chunk_size": 8,
            "window_chunks": 2,
            "num_pages": 4,
            "k_pages": 2,
            "theta_low_rank": True,
            "segment_chunks": 2,
            "checkpoint_segments": False,
        },
        "distill": {
            "enabled": True,
            "share_backbone": True,
            "layerwise_warmup_steps": 0,
            "teacher_ckpt": "runs/s1_loop_a6b",
            "teacher_track": "stock",
            "temperature": 1.0,
            "kl_weight": 1.0,
            "hidden_mse_weight": 0.1,
            "vocab_chunk_size": 64,
        },
        "train": {
            **dict(s1_cfg["train"]),
            "init_from": "runs/s1_loop_a6b",
            "steps": 251,
            "ckpt_every": 25,
            "train_hssd": True,
            "allowed_groups": ["loop_adapters", "hssd_adapters", "norms"],
        },
    }
    s1b_cfg = Cfg(s1b_dict)
    tr_mid = LMTrainer(s1b_cfg, str(run_root_1 / "s1b_hssd_distill"), hub_api=api)
    tr_mid.maybe_resume()
    with torch.no_grad():
        for name, p in tr_mid.stack.named_trainable.items():
            if "hssd_adapter." not in name:
                p.add_(0.35)
    tr_mid.step = 250
    tr_mid.tokens_seen = 250 * 16
    tr_mid.save(completed=False, metrics={"loss/lm": 1.5})

    # Confirm periodic checkpoint push pruned local `s1_loop_a6b/checkpoints` on `run_root_1`
    assert not (run_root_1 / "s1_loop_a6b" / "checkpoints").exists()
    tr_mid.stack.model.eval()
    assert tr_mid.stack.loop is not None
    tr_mid.stack.loop.set_active_passes(2)
    with torch.no_grad():
        drifted_student_h = tr_mid.stack.hidden(ids).detach().clone()
    assert not torch.allclose(drifted_student_h, expected_teacher_h, atol=1e-3)

    # 3. Fresh empty `run_root_2` (respawned Colab VM): `prepare_stage_workspace` must pull BOTH
    #    `s1b_hssd_distill` (step 250 with optim.pt) AND `s1_loop_a6b` (without optim.pt, kept by reclaim_disk_space)
    run_root_2 = tmp_path / "session2_fresh"
    monkeypatch.setenv("QLOOP_RUN_ROOT", str(run_root_2))
    ws = hub.prepare_stage_workspace("s1b_hssd_distill", s1b_cfg.to_dict(), str(run_root_2))
    assert ws["resumed_checkpoint"] is not None
    assert ws["resumed_checkpoint"].endswith("step_0000250")
    assert ws["teacher_stage"] == "s1_loop_a6b"

    s1b_ckpt = run_root_2 / "s1b_hssd_distill" / "checkpoints" / "step_0000250"
    assert (s1b_ckpt / "trainable.pt").is_file()
    assert (s1b_ckpt / "state.json").is_file()
    assert (s1b_ckpt / "optim.pt").is_file()

    s1_ckpt = run_root_2 / "s1_loop_a6b" / "checkpoints" / "step_0000001"
    assert (run_root_2 / "s1_loop_a6b" / "checkpoints" / "LATEST").is_file()
    assert (s1_ckpt / "trainable.pt").is_file()
    assert (s1_ckpt / "state.json").is_file()
    assert not (s1_ckpt / "optim.pt").exists()

    # Verify `SharedBackboneTeacher` on the resumed step-250 workspace finds `runs/s1_loop_a6b` on disk
    # and restores exact frozen `s1_loop_a6b` teacher outputs rather than drifted student `loop_adapters`.
    teacher_check = SharedBackboneTeacher(
        tr_mid.stack,
        teacher_ckpt="runs/s1_loop_a6b",
        teacher_track="stock",
        run_root=str(run_root_2),
    )
    assert teacher_check.loaded_from_ckpt is True
    with teacher_check.activate(), torch.no_grad():
        got_teacher_h = tr_mid.stack.hidden(ids)
    assert torch.allclose(got_teacher_h, expected_teacher_h, atol=1e-6)

    # 4. Full end-to-end `LMTrainer` resume on another fresh empty `run_root_3` (step 250 -> 251)
    run_root_3 = tmp_path / "session3_e2e_resume"
    monkeypatch.setenv("QLOOP_RUN_ROOT", str(run_root_3))
    tr_resumed = LMTrainer(s1b_cfg, str(run_root_3 / "s1b_hssd_distill"), hub_api=api)
    res = tr_resumed.train()
    assert res["step"] == 251
    assert tr_resumed.shared_teacher is not None
    assert tr_resumed.shared_teacher.loaded_from_ckpt is True
    tr_resumed.stack.model.eval()
    assert tr_resumed.stack.loop is not None
    tr_resumed.stack.loop.set_active_passes(2)
    with tr_resumed.shared_teacher.activate(), torch.no_grad():
        e2e_teacher_h = tr_resumed.stack.hidden(ids)
    assert torch.allclose(e2e_teacher_h, expected_teacher_h, atol=1e-6)

    # 5. Same-VM restart (`run_root_1` already has local s1b step >= 250, while `s1_loop_a6b/checkpoints` was pruned)
    ws_same_vm = hub.prepare_stage_workspace("s1b_hssd_distill", s1b_cfg.to_dict(), str(run_root_1))
    assert ws_same_vm["teacher_stage"] == "s1_loop_a6b"
    assert (run_root_1 / "s1_loop_a6b" / "checkpoints" / "step_0000001" / "trainable.pt").is_file()
    assert not (run_root_1 / "s1_loop_a6b" / "checkpoints" / "step_0000001" / "optim.pt").exists()

    # 6. Non-distill stage (`distill.enabled=False`) resuming mid-stage must NOT pull `teacher_ckpt`
    run_root_4 = tmp_path / "session4_no_distill"
    no_distill_cfg = {
        **s1b_cfg.to_dict(),
        "distill": {**dict(s1b_cfg["distill"]), "enabled": False},
    }
    ws_disabled = hub.prepare_stage_workspace("s1b_hssd_distill", no_distill_cfg, str(run_root_4))
    assert ws_disabled["resumed_checkpoint"] is not None
    assert ws_disabled["teacher_stage"] is None
    assert not (run_root_4 / "s1_loop_a6b" / "checkpoints").exists()


def test_load_qloop_from_hub_d1_lora_and_non_lsa_stages_from_s2_v2_repo(tmp_path, monkeypatch):
    """Regression test: `load_qloop_from_hub()` cleanly loads both (i) a non-LSA looped stage
    (`s1_loop_a6b`) and (ii) a non-looped `d1_lora` control stage (`s2_2_d1_lora_control`) from a
    Hub repo whose root `qloop_config.json` came from `s2_v2_joint_bptt` (`loop`+`gdn2`+`lsa` on).
    """
    api = MockHfApi()
    repo_id = "TassP0lar/test-s2b-hub-stages"
    run_root = tmp_path / "runs"
    monkeypatch.setenv("QLOOP_RUN_ROOT", str(run_root))

    base_sync = {
        "hf_repo": repo_id,
        "private": False,
        "push_every_ckpt": True,
        "push_optim": False,
        "prune_local_after_push": False,
    }
    ids = torch.randint(4, 90, (1, 24))

    # 1. Stage A: `s1_loop_a6b` (looped, NO LSA, NO GDN-2, NO HSSD, NO d1_lora)
    cfg_s1 = Cfg({
        **_toy_base_cfg(sync_repo=repo_id),
        "stage": "s1_loop_a6b",
        "gdn2": {"enabled": False},
        "hssd": {"enabled": False},
        "lsa": {"enabled": False},
        "experts": {"enabled": False},
        "checkpoint": {"sync": base_sync},
        "train": {
            "steps": 2,
            "micro_batch": 1,
            "grad_accum": 1,
            "ckpt_every": 2,
            "eval_every": 0,
            "keep_last": 1,
            "trainable": [r"self_attn\.(q_norm|k_norm)", r"input_layernorm|post_attention_layernorm"],
            "allowed_groups": ["loop_adapters", "attention", "norms"],
            "optim": {"lr": 5e-3, "warmup_steps": 1, "impl": "builtin"},
        },
    })
    tr_s1 = LMTrainer(cfg_s1, str(run_root / "s1_loop_a6b"), hub_api=api)
    tr_s1.train()
    tr_s1.stack.model.eval()
    assert tr_s1.stack.loop is not None
    tr_s1.stack.loop.set_active_passes(2)
    with torch.no_grad():
        h_s1 = tr_s1.stack.hidden(ids).detach().clone()

    # 2. Stage B: `s2_2_d1_lora_control` (non-looped d=1 + d1_lora, NO LSA/GDN-2/HSSD)
    cfg_d1 = Cfg({
        "seed": 7,
        "stage": "s2_2_d1_lora_control",
        "model": {"name": "toy", "dtype": "float32"},
        "loop": {"enabled": False},
        "gdn2": {"enabled": False},
        "hssd": {"enabled": False},
        "lsa": {"enabled": False},
        "experts": {"enabled": False},
        "d1_lora": {"enabled": True, "layer_start": 4, "layer_end": 12, "rank": 6, "seed": 5},
        "data": {"seq_len": 32, "sources": [{"name": "toy_cot", "weight": 1.0}]},
        "eval_quick": {"n_batches": 1, "depths": [1]},
        "checkpoint": {"sync": base_sync},
        "train": {
            "steps": 2,
            "micro_batch": 1,
            "grad_accum": 1,
            "ckpt_every": 2,
            "eval_every": 0,
            "keep_last": 1,
            "gradient_checkpointing": True,
            "train_d1_lora": True,
            "trainable": [r"self_attn\.(q_norm|k_norm)", r"input_layernorm|post_attention_layernorm"],
            "allowed_groups": ["d1_lora", "attention", "norms"],
            "optim": {"lr": 5e-3, "warmup_steps": 1, "impl": "builtin", "lr_mult": {"d1_lora": 20.0}},
        },
    })
    tr_d1 = LMTrainer(cfg_d1, str(run_root / "s2_2_d1_lora_control"), hub_api=api)
    tr_d1.train()
    tr_d1.stack.model.eval()
    assert tr_d1.stack.loop is None and tr_d1.stack.d1_lora is not None
    with torch.no_grad():
        h_d1 = tr_d1.stack.hidden(ids).detach().clone()

    # 3. Stage C (pushed LAST so root `qloop_config.json` has `loop`+`gdn2`+`lsa` enabled): `s2_v2_joint_bptt`
    cfg_s2v2 = Cfg({
        **_toy_base_cfg(sync_repo=repo_id),
        "stage": "s2_v2_joint_bptt",
        "gdn2": {"enabled": True, "rank": 6, "chunk_size": 16, "segment_chunks": 2},
        "hssd": {"enabled": False},
        "lsa": {
            "enabled": True,
            "mode": "distill",
            "index_dim": 16,
            "block_size": 16,
            "pool": 2,
            "avg_bucket_keys": 4,
            "max_bucket_keys": 8,
            "num_select": 4,
            "window": 16,
            "sinks": 4,
        },
        "experts": {"enabled": False},
        "checkpoint": {"sync": base_sync},
        "train": {
            "steps": 2,
            "micro_batch": 1,
            "grad_accum": 1,
            "ckpt_every": 2,
            "eval_every": 0,
            "keep_last": 1,
            "trainable": [
                r"self_attn\.(k_proj|v_proj|o_proj|q_norm|k_norm)",
                r"input_layernorm|post_attention_layernorm",
            ],
            "allowed_groups": ["loop_adapters", "gdn2_adapters", "lsa_indexers", "attention", "norms"],
            "optim": {"lr": 5e-3, "warmup_steps": 1, "impl": "builtin"},
        },
    })
    tr_s2v2 = LMTrainer(cfg_s2v2, str(run_root / "s2_v2_joint_bptt"), hub_api=api)
    tr_s2v2.train()
    tr_s2v2.stack.model.eval()
    assert tr_s2v2.stack.loop is not None
    tr_s2v2.stack.loop.set_active_passes(2)
    with torch.no_grad():
        h_s2v2 = tr_s2v2.stack.hidden(ids).detach().clone()

    root_qcfg = json.loads(api.files[(repo_id, "qloop_config.json")].decode("utf-8"))
    assert root_qcfg["stage"] == "s2_v2_joint_bptt"
    assert root_qcfg["config"]["lsa"]["enabled"] is True
    assert root_qcfg["config"]["gdn2"]["enabled"] is True
    assert root_qcfg["config"]["loop"]["enabled"] is True

    # (i) Load `s1_loop_a6b` from the `s2_v2` Hub repo -> looped, LSA/GDN2/HSSD/d1_lora ALL disabled!
    st_s1 = load_qloop_from_hub(repo_id, stage_name="s1_loop_a6b", api=api)
    assert st_s1.loop is not None
    assert st_s1.lsa is None
    assert st_s1.gdn2 is None
    assert st_s1.hssd is None
    assert st_s1.d1_lora is None
    st_s1.loop.set_active_passes(2)
    with torch.no_grad():
        assert torch.allclose(st_s1.hidden(ids), h_s1, atol=1e-5, rtol=1e-5)

    # (ii) Load `s2_2_d1_lora_control` from the `s2_v2` Hub repo -> d1_lora enabled, loop/LSA/GDN2/HSSD ALL disabled!
    st_d1 = load_qloop_from_hub(repo_id, stage_name="s2_2_d1_lora_control", api=api)
    assert st_d1.d1_lora is not None
    assert st_d1.d1_lora.rank == 6
    assert st_d1.loop is None
    assert st_d1.lsa is None
    assert st_d1.gdn2 is None
    assert st_d1.hssd is None
    assert not st_d1.model.training
    assert not any(p.requires_grad for p in st_d1.named_all.values())
    with torch.no_grad():
        assert torch.allclose(st_d1.hidden(ids), h_d1, atol=1e-5, rtol=1e-5)

    # (iii) Explicit `enable_lsa_if_present=False` on `s2_v2_joint_bptt` also disables LSA cleanly
    st_no_lsa = load_qloop_from_hub(repo_id, stage_name="s2_v2_joint_bptt", enable_lsa_if_present=False, api=api)
    assert st_no_lsa.loop is not None
    assert st_no_lsa.gdn2 is not None
    assert st_no_lsa.lsa is None
    assert st_no_lsa.d1_lora is None

    # (iv) Reverse root order: when `s2_2_d1_lora_control` overwrites root `qloop_config.json`,
    #      both default root load AND `stage_name="s1_loop_a6b"` load cleanly!
    hub = HubSyncManager(repo_id=repo_id, api=api)
    pulled_d1 = hub.pull_stage_checkpoint("s2_2_d1_lora_control", str(run_root), include_optim=False)
    assert pulled_d1 is not None
    oid_rev = hub.push_checkpoint("s2_2_d1_lora_control", pulled_d1, completed=True, config_dict=cfg_d1.to_dict())
    assert oid_rev is not None

    st_root_d1 = load_qloop_from_hub(repo_id, api=api)
    assert st_root_d1.d1_lora is not None and st_root_d1.d1_lora.rank == 6
    assert st_root_d1.loop is None and st_root_d1.lsa is None
    assert st_root_d1.load_info.get("gradient_checkpointing") is not True
    with torch.no_grad():
        assert torch.allclose(st_root_d1.hidden(ids), h_d1, atol=1e-5, rtol=1e-5)

    st_s1_rev = load_qloop_from_hub(repo_id, stage_name="s1_loop_a6b", api=api)
    assert st_s1_rev.loop is not None
    assert st_s1_rev.d1_lora is None and st_s1_rev.lsa is None and st_s1_rev.gdn2 is None
    st_s1_rev.loop.set_active_passes(2)
    with torch.no_grad():
        assert torch.allclose(st_s1_rev.hidden(ids), h_s1, atol=1e-5, rtol=1e-5)

    # Reverse-order load of `s2_v2_joint_bptt` (loop + gdn2 + lsa) after `s2_2_d1_lora_control`
    # overwrote root `qloop_config.json` (whose `allowed_groups` omitted `lsa_indexers`)!
    st_s2v2_rev = load_qloop_from_hub(
        repo_id,
        stage_name="s2_v2_joint_bptt",
        overrides=["lsa.mode=distill"],
        api=api,
    )
    assert st_s2v2_rev.loop is not None
    assert st_s2v2_rev.gdn2 is not None and st_s2v2_rev.gdn2.cfg.rank == 6
    assert st_s2v2_rev.lsa is not None
    assert st_s2v2_rev.d1_lora is None and st_s2v2_rev.hssd is None
    st_s2v2_rev.loop.set_active_passes(2)
    with torch.no_grad():
        assert torch.allclose(st_s2v2_rev.hidden(ids), h_s2v2, atol=1e-5, rtol=1e-5)


def test_load_qloop_from_hub_synthetic_config_only_loop_compat(tmp_path):
    """Backwards-compat: root `qloop_config.json` with `config.loop.enabled=True`, no `stack`,
    and empty `qloop_adapters.pt` (`{}`) still installs `st.loop` cleanly.
    """
    api = MockHfApi()
    repo_id = "TassP0lar/test-synth-config-only-loop"
    qcfg = {
        "repo_id": repo_id,
        "config": {
            "model": {"name": "toy", "dtype": "float32"},
            "loop": {
                "enabled": True,
                "prelude_end": 4,
                "core_end": 12,
                "num_passes": 2,
                "target_virtual_layers": 24,
            },
        },
    }
    empty_pt = tmp_path / "empty_adapters.pt"
    torch.save({}, empty_pt)
    api.commit_bundle(
        repo_id,
        [
            ("qloop_config.json", json.dumps(qcfg).encode("utf-8")),
            ("qloop_adapters.pt", str(empty_pt)),
        ],
        "synthetic config-only loop bundle",
    )

    st = load_qloop_from_hub(repo_id, api=api)
    assert st.loop is not None
    assert st.lsa is None
    assert st.gdn2 is None
    assert st.hssd is None
    assert st.d1_lora is None
