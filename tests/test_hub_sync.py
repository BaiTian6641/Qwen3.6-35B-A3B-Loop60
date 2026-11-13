"""Tests for Hugging Face Hub commit-history sync, multi-stage cumulative handoff, and 230 GB disk guard."""
from __future__ import annotations

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

    # Stage 1: train hot-expert deltas (rank=4) + shared_expert + loop_adapters
    cfg1 = Cfg({
        **_toy_base_cfg(),
        "stage": "s1_loop_heal",
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

    # Wipe local runs completely -> Stage 2 (`s3_long_lsa` analogue): LSA on, experts off, shared_expert frozen!
    shutil.rmtree(run_root)
    cfg2 = Cfg({
        **_toy_base_cfg(),
        "stage": "s3_long_lsa",
        "lsa": {"enabled": True, "mode": "distill", "index_dim": 16, "block_size": 16, "pool": 2,
                "avg_bucket_keys": 4, "max_bucket_keys": 8, "num_select": 4, "window": 16, "sinks": 4},
        "experts": {"enabled": False},
        "train": {
            "init_from": "runs/s1_loop_heal",
            "steps": 2,
            "micro_batch": 1,
            "grad_accum": 1,
            "keep_last": 1,
            "trainable": ["self_attn\\.(q_norm|k_norm)"],
            "optim": {"lr": 5e-3, "warmup_steps": 1},
        },
    })
    tr2 = LMTrainer(cfg2, str(run_root / "s3_long_lsa"), hub_api=api)
    tr2.train()

    # Wipe local runs again -> Stage 3: neither LSA nor experts attached, only router + loop_adapters trained!
    shutil.rmtree(run_root)
    cfg3 = Cfg({
        **_toy_base_cfg(),
        "stage": "s4_final",
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

    # Load from Hub in 1 call: LSA indexers (from Stage 2), folded expert deltas & shared_expert (from Stage 1),
    # and router/loop_adapters (from Stage 3) must ALL be present!
    loaded = load_qloop_from_hub(
        DEFAULT_HF_REPO,
        overrides=["lsa.mode=distill"],  # dense forward so outputs match tr3 bit-close while keeping trained indexers
        api=api,
    )
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
