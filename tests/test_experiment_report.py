"""Unit and artifact-integrity tests for reproducible experiment report & SVG chart export."""
from __future__ import annotations

import csv
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import pytest

from qloop.report import (
    STAGES,
    _assert_secret_free,
    compute_kv_memory_1m,
    contains_secret,
    export_experiment_report,
    main,
    parse_supervisor_log,
    sanitize_obj,
)

SVG_NS = "{http://www.w3.org/2000/svg}"
FAKE_HF_TOKEN = "hf_FAKEUNITTESTTOKEN1234567890ABCD"
FAKE_BEARER = "super_secret_bearer_token_987654321"
FAKE_TUNNEL = "https://colab.research.google.com/notebooks/empty.ipynb?dbu=private_tunnel_xyz"


def _make_manifest(
    stage: str,
    step: int,
    status: str,
    metrics: dict[str, Any],
    n_params: int = 18_412_598,
    target_steps: int = 1000,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "repo_id": "test/fake-loop64",
        "stages": {
            stage: {
                "status": status,
                "step": step,
                "target_steps": target_steps,
                "n_params": n_params,
                "metrics": metrics,
                "updated_at": "2026-11-21T12:00:00Z",
            }
        },
    }


def _build_offline_fixture(offline_dir: Path) -> None:
    offline_dir.mkdir(parents=True, exist_ok=True)

    commits = [
        # s0_probe
        {
            "commit_id": "107dd28700000000",
            "short_sha": "107dd287",
            "created_at": "2026-11-14T20:54:33+00:00",
            "title": "[qloop:s0_probe] artifacts (2 files)",
            "message": f"probe ok Authorization: Bearer {FAKE_BEARER}",
            "manifest": {"stages": {"s0_probe": {"status": "completed", "metrics": {}}}},
        },
        # Poisoned s1 + reset
        {
            "commit_id": "d1d9206400000000",
            "short_sha": "d1d92064",
            "created_at": "2026-11-15T12:37:40+00:00",
            "title": "[qloop:s1_loop_a6b] step=100 (ckpt) loss=9.6577",
            "message": f"poisoned token={FAKE_HF_TOKEN}",
            "manifest": _make_manifest(
                "s1_loop_a6b", 100, "in_progress",
                {"loss/lm": 9.6577, "tokens_seen": 13107200, "tok_per_s": 466.6, "mem/peak_gib": 66.0},
            ),
        },
        {
            "commit_id": "ad64665100000000",
            "short_sha": "ad646651",
            "created_at": "2026-11-16T00:23:58+00:00",
            "title": "[qloop:reset] purge poisoned s1_loop_a6b state",
            "message": "",
            "manifest": {"stages": {}},
        },
        # Clean s1_loop_a6b (with duplicate step=1000 + quality check)
        {
            "commit_id": "4026a4f700000000",
            "short_sha": "4026a4f7",
            "created_at": "2026-11-16T07:05:44+00:00",
            "title": "[qloop:s1_loop_a6b] step=200 (ckpt) loss=0.9613 depth_gain=+0.0443",
            "message": "",
            "manifest": _make_manifest(
                "s1_loop_a6b", 200, "in_progress",
                {
                    "loss/lm": 0.9613, "depth": 2, "tokens_seen": 26214400, "tok_per_s": 2094.7,
                    "optim/optim/global_grad_norm": 0.083, "optim/optim/skipped_total": 0.0,
                    "mem/peak_gib": 54.24, "eval/loss_d1": 0.9901, "eval/loss_d2": 0.9470,
                    "eval/loss_d3": 0.9458, "eval/depth_gain": 0.0443,
                },
            ),
        },
        {
            "commit_id": "ae60f56f00000000",
            "short_sha": "ae60f56f",
            "created_at": "2026-11-16T21:14:00+00:00",
            "title": "[qloop:s1_loop_a6b] step=1000 (ckpt) loss=0.7498 depth_gain=+0.0614",
            "message": "",
            "manifest": _make_manifest(
                "s1_loop_a6b", 1000, "in_progress",
                {
                    "loss/lm": 0.7498, "depth": 2, "tokens_seen": 131072000, "tok_per_s": 2218.8,
                    "optim/optim/global_grad_norm": 0.035, "optim/optim/skipped_total": 0.0,
                    "mem/peak_gib": 54.24, "eval/loss_d1": 0.9806, "eval/loss_d2": 0.9337,
                    "eval/loss_d3": 0.9192, "eval/depth_gain": 0.0614,
                },
            ),
        },
        {
            "commit_id": "32eb565e00000000",
            "short_sha": "32eb565e",
            "created_at": "2026-11-16T21:16:05+00:00",
            "title": "[qloop:s1_loop_a6b] step=1000 (completed) loss=0.7498 depth_gain=+0.0609",
            "message": "",
            "manifest": _make_manifest(
                "s1_loop_a6b", 1000, "completed",
                {
                    "loss/lm": 0.7498, "depth": 2, "tokens_seen": 131072000, "tok_per_s": 2218.8,
                    "optim/optim/global_grad_norm": 0.035, "optim/optim/skipped_total": 0.0,
                    "mem/peak_gib": 54.24, "eval/loss_d1": 0.9800, "eval/loss_d2": 0.9336,
                    "eval/loss_d3": 0.9191, "eval/depth_gain": 0.0609,
                },
            ),
        },
        {
            "commit_id": "43ecae2400000000",
            "short_sha": "43ecae24",
            "created_at": "2026-11-17T00:08:41+00:00",
            "title": "[qloop:s1_loop_a6b] live GPU quality check report (step=1000, d=1,2,3)",
            "message": "",
            "manifest": _make_manifest("s1_loop_a6b", 1000, "completed", {"loss/lm": 0.7498}),
        },
        # s1b Phase 1 + pre-fix explosion + rollback + clean Phase 2
        {
            "commit_id": "0ff9b53700000000",
            "short_sha": "0ff9b537",
            "created_at": "2026-11-17T10:18:14+00:00",
            "title": "[qloop:s1b_hssd_distill] step=400 (ckpt) loss=0.0003 depth_gain=+0.0628",
            "message": "",
            "manifest": _make_manifest(
                "s1b_hssd_distill", 400, "in_progress",
                {
                    "loss/lm": 0.0003, "loss/distill_phase": 1.0, "loss/layerwise_mse": 7.39e-5,
                    "tokens_seen": 26214400, "tok_per_s": 1450.8, "optim/optim/global_grad_norm": 0.084,
                    "optim/optim/skipped_total": 13.0, "mem/peak_gib": 48.41,
                    "eval/loss_d1": 1.1285, "eval/loss_d2": 1.0783, "eval/loss_d3": 1.0657, "eval/depth_gain": 0.0628,
                },
                n_params=76_917_638, target_steps=1500,
            ),
        },
        {
            "commit_id": "7a2d590a00000000",
            "short_sha": "7a2d590a",
            "created_at": "2026-11-17T10:49:45+00:00",
            "title": "[qloop:s1b_hssd_distill] step=425 (ckpt) loss=9.6975",
            "message": "",
            "manifest": _make_manifest(
                "s1b_hssd_distill", 425, "in_progress",
                {
                    "loss/lm": 9.6975, "loss/distill_phase": 2.0, "loss/hssd_blend": 0.076,
                    "loss/distill_kl": 8.015, "tokens_seen": 27852800, "tok_per_s": 1030.8,
                    "optim/optim/global_grad_norm": 1.73e17, "optim/optim/skipped_total": 5193.0,
                    "mem/peak_gib": 68.97,
                },
                n_params=76_917_638, target_steps=1500,
            ),
        },
        {
            "commit_id": "dc7c4ad800000000",
            "short_sha": "dc7c4ad8",
            "created_at": "2026-11-17T15:24:18+00:00",
            "title": "[qloop:s1b_hssd_distill] step=650 (ckpt) loss=12.0742",
            "message": "",
            "manifest": _make_manifest(
                "s1b_hssd_distill", 650, "in_progress",
                {
                    "loss/lm": 12.0742, "loss/distill_phase": 2.0, "loss/hssd_blend": 0.996,
                    "loss/distill_kl": 8.306, "tokens_seen": 42598400, "tok_per_s": 766.2,
                    "optim/optim/global_grad_norm": 6.12e19, "optim/optim/skipped_total": 194510.0,
                    "mem/peak_gib": 68.97,
                },
                n_params=76_917_638, target_steps=1500,
            ),
        },
        {
            "commit_id": "dd5093d300000000",
            "short_sha": "dd5093d3",
            "created_at": "2026-11-17T22:59:48+00:00",
            "title": "[qloop:s1b_hssd_distill] rollback stage state to clean Phase-1 step=400 (0ff9b537f1cc)",
            "message": "",
            "manifest": _make_manifest(
                "s1b_hssd_distill", 400, "running", {"loss/lm": 0.0003},
                n_params=76_917_638, target_steps=1500,
            ),
        },
        {
            "commit_id": "23f832f400000000",
            "short_sha": "23f832f4",
            "created_at": "2026-11-18T06:32:47+00:00",
            "title": "[qloop:s1b_hssd_distill] step=675 (ckpt) loss=6.2521",
            "message": "",
            "manifest": _make_manifest(
                "s1b_hssd_distill", 675, "in_progress",
                {
                    "loss/lm": 6.2521, "loss/distill_phase": 2.0, "loss/hssd_blend": 1.0,
                    "loss/distill_kl": 6.1618, "tokens_seen": 44236800, "tok_per_s": 829.5,
                    "optim/optim/global_grad_norm": 336.88, "optim/optim/skipped_total": 0.0,
                    "mem/peak_gib": 68.97,
                },
                n_params=76_917_638, target_steps=1500,
            ),
        },
        {
            "commit_id": "2fb912db00000000",
            "short_sha": "2fb912db",
            "created_at": "2026-11-19T04:33:51+00:00",
            "title": "[qloop:s1b_hssd_distill] step=1500 (completed) loss=0.7253 depth_gain=+0.0626",
            "message": "",
            "manifest": _make_manifest(
                "s1b_hssd_distill", 1500, "completed",
                {
                    "loss/lm": 0.7253, "loss/distill_phase": 2.0, "loss/hssd_blend": 1.0,
                    "loss/distill_kl": 0.7181, "tokens_seen": 98304000, "tok_per_s": 785.6,
                    "optim/optim/global_grad_norm": 0.754, "optim/optim/skipped_total": 0.0,
                    "mem/peak_gib": 68.96, "eval/loss_d1": 1.7679, "eval/loss_d2": 1.7375,
                    "eval/loss_d3": 1.7052, "eval/depth_gain": 0.0626,
                },
                n_params=76_917_638, target_steps=1500,
            ),
        },
        # s3_long_1m_lsa
        {
            "commit_id": "ad3542a600000000",
            "short_sha": "ad3542a6",
            "created_at": "2026-11-19T17:12:36+00:00",
            "title": "[qloop:s3_long_1m_lsa] step=125 (ckpt) loss=2.4133",
            "message": "",
            "manifest": _make_manifest(
                "s3_long_1m_lsa", 125, "in_progress",
                {
                    "loss/lm": 2.4133, "loss/lsa_kl": 21.05, "loss/calib_prefix_len": 1015808.0,
                    "lsa/distill_kl/L3": 2.10, "lsa/recall/L3": 0.4463, "lsa/min_head_recall/L3": 0.3029,
                    "lsa/last_recall": 0.4463, "tokens_seen": 25000000, "tok_per_s": 1295.1,
                    "optim/optim/global_grad_norm": 5.41, "optim/optim/skipped_total": 0.0,
                    "mem/peak_gib": 80.67,
                },
                n_params=83_061_638, target_steps=600,
            ),
        },
        {
            "commit_id": "55c7a68000000000",
            "short_sha": "55c7a680",
            "created_at": "2026-11-20T20:09:14+00:00",
            "title": "[qloop:s3_long_1m_lsa] step=600 (completed) loss=2.7642 depth_gain=+0.1414",
            "message": "",
            "manifest": _make_manifest(
                "s3_long_1m_lsa", 600, "completed",
                {
                    "loss/lm": 2.7642, "loss/lsa_kl": 27.47, "loss/calib_prefix_len": 573440.0,
                    "lsa/distill_kl/L3": 1.98, "lsa/recall/L3": 0.4595, "lsa/min_head_recall/L3": 0.2970,
                    "lsa/last_recall": 0.4595, "tokens_seen": 120782848, "tok_per_s": 1157.6,
                    "optim/optim/global_grad_norm": 5.37, "optim/optim/skipped_total": 0.0,
                    "mem/peak_gib": 81.23, "eval/loss_d1": 2.0171, "eval/loss_d2": 1.9511,
                    "eval/loss_d3": 1.8757, "eval/depth_gain": 0.1414,
                },
                n_params=83_061_638, target_steps=600,
            ),
        },
        # s2_joint_bptt
        {
            "commit_id": "f1df5c9b00000000",
            "short_sha": "f1df5c9b",
            "created_at": "2026-11-21T05:33:03+00:00",
            "title": "[qloop:s2_joint_bptt] step=43 (ckpt) loss=5.4383 depth_gain=+0.0852",
            "message": "",
            "manifest": _make_manifest(
                "s2_joint_bptt", 43, "in_progress",
                {
                    "loss/lm": 5.4383, "depth": 3, "tokens_seen": 10485760, "tok_per_s": 453.6,
                    "optim/optim/global_grad_norm": 2.02, "optim/optim/skipped_total": 0.0,
                    "mem/peak_gib": 72.51, "eval/loss_d1": 1.3556, "eval/loss_d2": 1.3275,
                    "eval/loss_d3": 1.2704, "eval/depth_gain": 0.0852,
                },
                n_params=187_919_238, target_steps=1200,
            ),
        },
        {
            "commit_id": "876de13800000000",
            "short_sha": "876de138",
            "created_at": "2026-11-21T22:37:18+00:00",
            "title": "[qloop:s2_joint_bptt] step=200 (ckpt) loss=4.2615 depth_gain=+0.0726",
            "message": "",
            "manifest": _make_manifest(
                "s2_joint_bptt", 200, "in_progress",
                {
                    "loss/lm": 4.2615, "depth": 2, "tokens_seen": 52428800, "tok_per_s": 962.6,
                    "optim/optim/global_grad_norm": 1.23, "optim/optim/skipped_total": 0.0,
                    "loss/bptt_offload_to_cpu": 0.0, "mem/peak_gib": 70.14,
                    "eval/loss_d1": 1.2945, "eval/loss_d2": 1.2728, "eval/loss_d3": 1.2219, "eval/depth_gain": 0.0726,
                },
                n_params=187_919_238, target_steps=1200,
            ),
        },
    ]
    (offline_dir / "hub_commits.json").write_text(json.dumps(commits, indent=2), encoding="utf-8")

    probe = {
        "summary": {
            "num_gates_passed": 5, "num_gates_total": 5,
            "frozen_experts_sha256": "42f1be31694f1ad29abe", "total_seconds": 177.44,
        },
        "gates": {"0a_backward": {"passed": True, "value": 11.75}},
        "core_ranges": [
            {"prelude_end": 12, "core_end": 36, "loss_increase": 0.1386},
            {"prelude_end": 8, "core_end": 32, "loss_increase": 0.4516},
        ],
        "probes": {
            "expert_headroom": {
                "hard_token_headroom_nats": 1.3061, "improved_hard_token_frac": 0.8891, "mean_router_jaccard": 0.3365,
            },
            "reentry_anchors": {"procrustes_cosine": 0.8588, "procrustes_explained_variance": 0.9993},
            "positional_extrapolation": {
                "modes": {
                    "yarn": {"extrapolated_nll": 2.6459, "passkey_logit_margin": 13.7209, "lm_passkey_margin": 17.087},
                    "rerope": {"extrapolated_nll": 2.6603},
                    "native": {"extrapolated_nll": 2.8248},
                }
            },
        },
    }
    (offline_dir / "probe_report.json").write_text(json.dumps(probe), encoding="utf-8")

    live_s2 = [
        {
            "step": 60, "time": 1795260960.0, "loss/lm": 5.039, "depth": 2, "tok_per_s": 892.1,
            "tokens_seen": 15728640, "optim/optim/global_grad_norm": 1.40, "optim/optim/skipped_total": 0.0,
            "loss/bptt_offload_to_cpu": 0.0, "mem/peak_gib": 70.03,
            "hf_token_leak_test": FAKE_HF_TOKEN,
        },
        {
            "step": 200, "time": 1795300000.0,
            "eval/loss_d1": 1.2945, "eval/loss_d2": 1.2728, "eval/loss_d3": 1.2219, "eval/depth_gain": 0.0726,
        },
        {
            "step": 270, "time": 1795319993.0, "loss/lm": 6.166, "depth": 3, "tok_per_s": 855.6,
            "tokens_seen": 70778880, "optim/optim/global_grad_norm": 1.19, "optim/optim/skipped_total": 0.0,
            "loss/bptt_offload_to_cpu": 0.0, "mem/peak_gib": 70.14,
        },
    ]
    (offline_dir / "live_s2_metrics.jsonl").write_text(
        "\n".join(json.dumps(r) for r in live_s2) + "\n", encoding="utf-8"
    )

    ka = "session=loop64 pid_alive=True exit_code=None disk_free=209.2GiB gpu=RTX"
    sup_log = "\n".join(
        [
            f"2026-11-20T22:27:20Z [URL] session=loop64 {FAKE_TUNNEL}",
            "2026-11-20T22:27:21Z [BOOT] ready session=loop64 launched=True pid=31606 head=2e37bf6",
            f"2026-11-21T00:09:24Z [KEEPALIVE] {ka} util=98% mem=76000/97887MiB pwr=349.8W temp=55C"
            " | 00:08:17 I [-] step 10/1200 lm 5.0330 d=2 481 tok/s"
            " mem alloc 32.4G / reserved 73.6G / peak 72.5G",
            f"2026-11-21T05:07:11Z [KEEPALIVE] {ka} util=98% mem=76500/97887MiB pwr=339.8W temp=56C"
            " | 05:03:35 I [-] step 40/1200 lm 5.4383 d=3 454 tok/s"
            " mem alloc 32.6G / reserved 74.2G / peak 72.5G",
            "2026-11-21T06:28:48Z [BOOT] ready session=loop64 launched=True pid=154438 head=b2c5513",
            f"2026-11-21T07:57:13Z [KEEPALIVE] {ka} util=99% mem=97200/97887MiB pwr=358.5W temp=58C"
            " | 07:54:25 I [-] step 60/1200 lm 6.3484 d=3 930 tok/s"
            " mem alloc 52.0G / reserved 93.9G / peak 89.8G",
            "2026-11-21T08:22:20Z [KEEPALIVE] session=loop64 pid_alive=False exit_code=1 disk_free=207.8GiB"
            " gpu=RTX util=0% mem=0/97887MiB pwr=45.8W temp=22C"
            " | 08:18:34 E [-] [pipeline] stage s2_joint_bptt failed with exit code 1",
            "2026-11-21T10:41:12Z [BOOT] ready session=loop64 launched=True pid=53372 head=5a82677",
            f"2026-11-21T11:38:20Z [KEEPALIVE] {ka} util=99% mem=76147/97887MiB pwr=381.4W temp=58C"
            " | 11:36:00 I [-] step 60/1200 lm 5.0390 d=2 892 tok/s"
            " mem alloc 25.3G / reserved 73.3G / peak 70.0G",
            f"2026-11-22T04:05:25Z [KEEPALIVE] {ka} util=99% mem=76147/97887MiB pwr=372.2W temp=58C"
            " | 03:59:53 I [-] step 270/1200 lm 6.1663 d=3 856 tok/s"
            " mem alloc 25.4G / reserved 73.3G / peak 70.1G",
        ]
    )
    (offline_dir / "colab_supervisor.log").write_text(sup_log + "\n", encoding="utf-8")


def test_offline_export_e2e_dedup_rollback_svgs_and_secret_redaction(tmp_path: Path) -> None:
    offline_dir = tmp_path / "offline"
    out_dir = tmp_path / "experiment_logs"
    report_path = tmp_path / "EXPERIMENT_REPORT.md"
    _build_offline_fixture(offline_dir)

    res = export_experiment_report(
        repo="test/fake-loop64",
        session="loop64",
        supervisor_log=str(offline_dir / "colab_supervisor.log"),
        out_dir=str(out_dir),
        report_path=str(report_path),
        offline_dir=str(offline_dir),
    )

    assert res["total_commits"] == 17
    assert res["rolled_back_count"] == 3  # 1 poisoned s1 + 2 pre-fix s1b explosion commits
    assert res["live_s2_rows"] == 3

    # 1. Verify hub_manifest_history.json flags rollbacks and strips secrets
    hist = json.loads((out_dir / "hub_manifest_history.json").read_text(encoding="utf-8"))
    rb_shas = {c["short_sha"] for c in hist["commits"] if c["rolled_back"]}
    assert rb_shas == {"d1d92064", "7a2d590a", "dc7c4ad8"}

    # 2. Verify CSV deduplication, rollback exclusion, and sorted unique steps
    for st in STAGES:
        csv_p = out_dir / f"{st}.csv"
        assert csv_p.is_file()
        with open(csv_p, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        steps = [int(r["step"]) for r in rows]
        assert steps == sorted(steps)
        assert len(steps) == len(set(steps))
        assert len(rows) >= 2

    with open(out_dir / "s1_loop_a6b.csv", encoding="utf-8") as f:
        s1_rows = list(csv.DictReader(f))
    assert [int(r["step"]) for r in s1_rows] == [200, 1000]
    assert s1_rows[-1]["status"] == "completed"
    assert s1_rows[-1]["commit_sha"] == "32eb565e"

    with open(out_dir / "s1b_hssd_distill.csv", encoding="utf-8") as f:
        s1b_rows = list(csv.DictReader(f))
    # Pre-fix explosion steps 425 and 650 must NOT appear in clean s1b CSV
    assert [int(r["step"]) for r in s1b_rows] == [400, 675, 1500]
    assert float(s1b_rows[-1]["skipped_total"]) == 0.0

    # 3. Verify summary_metrics.json captures s2 regimes and rollback incident
    sm = json.loads((out_dir / "summary_metrics.json").read_text(encoding="utf-8"))
    assert sm["stages"]["s1_loop_a6b"]["latest_commit_sha"] == "32eb565e"
    assert sm["rollback_summary"]["s1_reset_commits"] == 1
    assert sm["rollback_summary"]["s1b_phase2_explosion_commits"] == 2
    assert sm["rollback_summary"]["s1b_explosion_peak_loss"] == pytest.approx(12.0742)
    assert set(sm["supervisor_s2_regimes"].keys()) == {
        "pre_opt_2e37bf6", "post_opt_leaky_b2c5513", "post_leak_fix_5a82677"
    }
    assert sm["supervisor_s2_regimes"]["post_leak_fix_5a82677"]["alloc_max_gib"] == pytest.approx(25.4)

    # 4. Verify all 5 SVG charts parse cleanly as valid XML with real plotted primitives
    expected_svgs = (
        "eval_depth_curves.svg",
        "s1b_hssd_convergence.svg",
        "s3_long_1m_dlsa.svg",
        "s2_bptt_throughput_vram.svg",
        "kv_memory_vs_inkling.svg",
    )
    for fname in expected_svgs:
        svg_p = out_dir / "charts" / fname
        assert svg_p.is_file()
        svg_txt = svg_p.read_text(encoding="utf-8")
        root = ET.fromstring(svg_txt)
        assert root.tag == f"{SVG_NS}svg"
        n_geom = (
            len(root.findall(f".//{SVG_NS}polyline"))
            + len(root.findall(f".//{SVG_NS}circle"))
            + len(root.findall(f".//{SVG_NS}rect"))
        )
        assert n_geom >= 10, f"{fname} missing plotted geometry"
        assert len(root.findall(f".//{SVG_NS}text")) >= 10, f"{fname} missing axis/labels"

    # 5. Verify EXPERIMENT_REPORT.md structure & relative SVG links
    md = report_path.read_text(encoding="utf-8")
    for fname in expected_svgs:
        assert f"experiment_logs/charts/{fname}" in md
    for heading in (
        "## 1. Executive Summary",
        "## 2. Architecture & 1M-Context Memory Efficiency",
        "## 3. Full Stage-by-Stage Empirical Results",
        "## 4. Systems & Kernel Engineering Highlights",
        "## 5. Honest Limitations, Next Milestones & Compute Grant Justification",
    ):
        assert heading in md

    # 6. Verify ZERO secret / tunnel leakage across ALL emitted files
    for p in res["written_files"]:
        content = Path(p).read_text(encoding="utf-8")
        assert not contains_secret(content), f"Secret pattern found in {p}"
        assert FAKE_HF_TOKEN not in content, f"Raw HF token leaked into {p}"
        assert FAKE_BEARER not in content, f"Bearer token leaked into {p}"
        assert "private_tunnel_xyz" not in content, f"Colab tunnel leaked into {p}"


def test_cli_offline_entrypoint(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    offline_dir = tmp_path / "offline"
    out_dir = tmp_path / "logs"
    report_path = tmp_path / "REPORT.md"
    _build_offline_fixture(offline_dir)

    rc = main(
        [
            "--repo", "test/fake-loop64",
            "--offline-dir", str(offline_dir),
            "--out-dir", str(out_dir),
            "--report", str(report_path),
        ]
    )
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert len(out["written_files"]) == 12
    assert report_path.is_file()


def test_secret_sanitization_and_guard() -> None:
    dirty = {"HF_TOKEN": FAKE_HF_TOKEN, "msg": f"use {FAKE_HF_TOKEN} and Bearer {FAKE_BEARER}"}
    clean = sanitize_obj(dirty)
    assert clean["HF_TOKEN"] == "[REDACTED]"
    assert FAKE_HF_TOKEN not in clean["msg"]
    assert FAKE_BEARER not in clean["msg"]
    assert not contains_secret(json.dumps(clean))

    with pytest.raises(RuntimeError, match="unredacted secret"):
        _assert_secret_free("dummy.json", f"oops {FAKE_HF_TOKEN}")


def test_kv_memory_1m_exact_math() -> None:
    rows = {r["name"]: r for r in compute_kv_memory_1m()}
    ours = rows["LoopHSSD v3.2 (88L @ R=3)"]
    inkling = rows["Inkling 2 Mini (48L @ R=3)"]
    unshared_88 = rows["Standard Unshared 88L Full-Attn"]
    assert ours["kv_gib"] == pytest.approx(10.00)
    assert ours["linear_or_swa_gib"] == pytest.approx(5.50)
    assert ours["total_gib"] == pytest.approx(15.82)
    assert inkling["kv_gib"] == pytest.approx(32.00)
    assert unshared_88["kv_gib"] == pytest.approx(176.00)


def test_parse_supervisor_log_missing_returns_empty() -> None:
    res = parse_supervisor_log("/nonexistent/path/colab_supervisor.log")
    assert res["present"] is False
    assert res["step_samples"] == []


def test_committed_docs_artifacts_xml_and_secret_free() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    report_p = repo_root / "docs" / "EXPERIMENT_REPORT.md"
    logs_dir = repo_root / "docs" / "experiment_logs"
    if not report_p.is_file() or not logs_dir.is_dir():
        pytest.skip("Generated docs/experiment_logs not present")

    assert not contains_secret(report_p.read_text(encoding="utf-8"))
    for p in sorted(logs_dir.rglob("*")):
        if not p.is_file():
            continue
        txt = p.read_text(encoding="utf-8")
        assert not contains_secret(txt), f"Potential secret in committed file {p}"
        if p.suffix == ".svg":
            root = ET.fromstring(txt)
            assert root.tag == f"{SVG_NS}svg"
        elif p.suffix == ".csv":
            with open(p, encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            steps = [int(r["step"]) for r in rows]
            assert steps == sorted(steps)
            assert len(steps) == len(set(steps))
