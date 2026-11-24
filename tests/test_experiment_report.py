"""Unit and artifact-integrity tests for reproducible v2 experiment report & SVG chart export."""
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

EXPECTED_SVGS = (
    "eval_depth_curves.svg",
    "s3_hssd_vs_v2_gdn_ab.svg",
    "dplr_gdn2_kernel_bench.svg",
    "s1b_hssd_convergence.svg",
    "s3_long_1m_dlsa.svg",
    "s2_bptt_throughput_vram.svg",
    "kv_memory_vs_inkling.svg",
)


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
                "updated_at": "2026-11-24T05:00:00Z",
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
                "s1_loop_a6b",
                100,
                "in_progress",
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
                "s1_loop_a6b",
                200,
                "in_progress",
                {
                    "loss/lm": 0.9613,
                    "depth": 2,
                    "tokens_seen": 26214400,
                    "tok_per_s": 2094.7,
                    "optim/optim/global_grad_norm": 0.083,
                    "optim/optim/skipped_total": 0.0,
                    "mem/peak_gib": 54.24,
                    "eval/loss_d1": 0.9901,
                    "eval/loss_d2": 0.9470,
                    "eval/loss_d3": 0.9458,
                    "eval/depth_gain": 0.0443,
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
                "s1_loop_a6b",
                1000,
                "in_progress",
                {
                    "loss/lm": 0.7498,
                    "depth": 2,
                    "tokens_seen": 131072000,
                    "tok_per_s": 2218.8,
                    "optim/optim/global_grad_norm": 0.035,
                    "optim/optim/skipped_total": 0.0,
                    "mem/peak_gib": 54.24,
                    "eval/loss_d1": 0.9806,
                    "eval/loss_d2": 0.9337,
                    "eval/loss_d3": 0.9192,
                    "eval/depth_gain": 0.0614,
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
                "s1_loop_a6b",
                1000,
                "completed",
                {
                    "loss/lm": 0.7498,
                    "depth": 2,
                    "tokens_seen": 131072000,
                    "tok_per_s": 2218.8,
                    "optim/optim/global_grad_norm": 0.035,
                    "optim/optim/skipped_total": 0.0,
                    "mem/peak_gib": 54.24,
                    "eval/loss_d1": 0.9800,
                    "eval/loss_d2": 0.9336,
                    "eval/loss_d3": 0.9191,
                    "eval/depth_gain": 0.0609,
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
                "s1b_hssd_distill",
                400,
                "in_progress",
                {
                    "loss/lm": 0.0003,
                    "loss/distill_phase": 1.0,
                    "loss/layerwise_mse": 7.39e-5,
                    "tokens_seen": 26214400,
                    "tok_per_s": 1450.8,
                    "optim/optim/global_grad_norm": 0.084,
                    "optim/optim/skipped_total": 13.0,
                    "mem/peak_gib": 48.41,
                    "eval/loss_d1": 1.1285,
                    "eval/loss_d2": 1.0783,
                    "eval/loss_d3": 1.0657,
                    "eval/depth_gain": 0.0628,
                },
                n_params=76_917_638,
                target_steps=1500,
            ),
        },
        {
            "commit_id": "7a2d590a00000000",
            "short_sha": "7a2d590a",
            "created_at": "2026-11-17T10:49:45+00:00",
            "title": "[qloop:s1b_hssd_distill] step=425 (ckpt) loss=9.6975",
            "message": "",
            "manifest": _make_manifest(
                "s1b_hssd_distill",
                425,
                "in_progress",
                {
                    "loss/lm": 9.6975,
                    "loss/distill_phase": 2.0,
                    "loss/hssd_blend": 0.076,
                    "loss/distill_kl": 8.015,
                    "tokens_seen": 27852800,
                    "tok_per_s": 1030.8,
                    "optim/optim/global_grad_norm": 1.73e17,
                    "optim/optim/skipped_total": 5193.0,
                    "mem/peak_gib": 68.97,
                },
                n_params=76_917_638,
                target_steps=1500,
            ),
        },
        {
            "commit_id": "dc7c4ad800000000",
            "short_sha": "dc7c4ad8",
            "created_at": "2026-11-17T15:24:18+00:00",
            "title": "[qloop:s1b_hssd_distill] step=650 (ckpt) loss=12.0742",
            "message": "",
            "manifest": _make_manifest(
                "s1b_hssd_distill",
                650,
                "in_progress",
                {
                    "loss/lm": 12.0742,
                    "loss/distill_phase": 2.0,
                    "loss/hssd_blend": 0.996,
                    "loss/distill_kl": 8.306,
                    "tokens_seen": 42598400,
                    "tok_per_s": 766.2,
                    "optim/optim/global_grad_norm": 6.12e19,
                    "optim/optim/skipped_total": 194510.0,
                    "mem/peak_gib": 68.97,
                },
                n_params=76_917_638,
                target_steps=1500,
            ),
        },
        {
            "commit_id": "dd5093d300000000",
            "short_sha": "dd5093d3",
            "created_at": "2026-11-17T22:59:48+00:00",
            "title": "[qloop:s1b_hssd_distill] rollback stage state to clean Phase-1 step=400 (0ff9b537f1cc)",
            "message": "",
            "manifest": _make_manifest(
                "s1b_hssd_distill",
                400,
                "running",
                {"loss/lm": 0.0003},
                n_params=76_917_638,
                target_steps=1500,
            ),
        },
        {
            "commit_id": "23f832f400000000",
            "short_sha": "23f832f4",
            "created_at": "2026-11-18T06:32:47+00:00",
            "title": "[qloop:s1b_hssd_distill] step=675 (ckpt) loss=6.2521",
            "message": "",
            "manifest": _make_manifest(
                "s1b_hssd_distill",
                675,
                "in_progress",
                {
                    "loss/lm": 6.2521,
                    "loss/distill_phase": 2.0,
                    "loss/hssd_blend": 1.0,
                    "loss/distill_kl": 6.1618,
                    "tokens_seen": 44236800,
                    "tok_per_s": 829.5,
                    "optim/optim/global_grad_norm": 336.88,
                    "optim/optim/skipped_total": 0.0,
                    "mem/peak_gib": 68.97,
                },
                n_params=76_917_638,
                target_steps=1500,
            ),
        },
        {
            "commit_id": "2fb912db00000000",
            "short_sha": "2fb912db",
            "created_at": "2026-11-19T04:33:51+00:00",
            "title": "[qloop:s1b_hssd_distill] step=1500 (completed) loss=0.7253 depth_gain=+0.0626",
            "message": "",
            "manifest": _make_manifest(
                "s1b_hssd_distill",
                1500,
                "completed",
                {
                    "loss/lm": 0.7253,
                    "loss/distill_phase": 2.0,
                    "loss/hssd_blend": 1.0,
                    "loss/distill_kl": 0.7181,
                    "tokens_seen": 98304000,
                    "tok_per_s": 785.6,
                    "optim/optim/global_grad_norm": 0.754,
                    "optim/optim/skipped_total": 0.0,
                    "mem/peak_gib": 68.96,
                    "eval/loss_d1": 1.7679,
                    "eval/loss_d2": 1.7375,
                    "eval/loss_d3": 1.7052,
                    "eval/depth_gain": 0.0626,
                },
                n_params=76_917_638,
                target_steps=1500,
            ),
        },
        # s3_long_1m_lsa (v1 HSSD)
        {
            "commit_id": "ad3542a600000000",
            "short_sha": "ad3542a6",
            "created_at": "2026-11-19T17:12:36+00:00",
            "title": "[qloop:s3_long_1m_lsa] step=125 (ckpt) loss=2.4133",
            "message": "",
            "manifest": _make_manifest(
                "s3_long_1m_lsa",
                125,
                "in_progress",
                {
                    "loss/lm": 2.4133,
                    "loss/lsa_kl": 21.05,
                    "loss/calib_prefix_len": 1015808.0,
                    "lsa/distill_kl/L3": 2.10,
                    "lsa/recall/L3": 0.4463,
                    "lsa/min_head_recall/L3": 0.3029,
                    "lsa/last_recall": 0.4463,
                    "tokens_seen": 25000000,
                    "tok_per_s": 1295.1,
                    "optim/optim/global_grad_norm": 5.41,
                    "optim/optim/skipped_total": 0.0,
                    "mem/peak_gib": 80.67,
                },
                n_params=83_061_638,
                target_steps=600,
            ),
        },
        {
            "commit_id": "200v1s3000000000",
            "short_sha": "200v1s30",
            "created_at": "2026-11-19T22:10:00+00:00",
            "title": "[qloop:s3_long_1m_lsa] step=200 (ckpt) loss=2.60 depth_gain=+0.1108",
            "message": "",
            "manifest": _make_manifest(
                "s3_long_1m_lsa",
                200,
                "in_progress",
                {
                    "loss/lm": 2.60,
                    "loss/lsa_kl": 22.0,
                    "lsa/distill_kl/L3": 2.20,
                    "lsa/recall/L3": 0.3712,
                    "lsa/min_head_recall/L3": 0.2330,
                    "lsa/last_recall": 0.3712,
                    "tokens_seen": 40000000,
                    "tok_per_s": 1020.0,
                    "optim/optim/global_grad_norm": 5.20,
                    "optim/optim/skipped_total": 0.0,
                    "mem/peak_gib": 79.10,
                    "eval/loss_d1": 2.0269,
                    "eval/loss_d2": 1.9698,
                    "eval/loss_d3": 1.9162,
                    "eval/depth_gain": 0.1108,
                },
                n_params=83_061_638,
                target_steps=600,
            ),
        },
        {
            "commit_id": "55c7a68000000000",
            "short_sha": "55c7a680",
            "created_at": "2026-11-20T20:09:14+00:00",
            "title": "[qloop:s3_long_1m_lsa] step=600 (completed) loss=2.7642 depth_gain=+0.1414",
            "message": "",
            "manifest": _make_manifest(
                "s3_long_1m_lsa",
                600,
                "completed",
                {
                    "loss/lm": 2.7642,
                    "loss/lsa_kl": 27.47,
                    "loss/calib_prefix_len": 573440.0,
                    "lsa/distill_kl/L3": 1.98,
                    "lsa/recall/L3": 0.4595,
                    "lsa/min_head_recall/L3": 0.2970,
                    "lsa/last_recall": 0.4595,
                    "tokens_seen": 120782848,
                    "tok_per_s": 1157.6,
                    "optim/optim/global_grad_norm": 5.37,
                    "optim/optim/skipped_total": 0.0,
                    "mem/peak_gib": 81.23,
                    "eval/loss_d1": 2.0171,
                    "eval/loss_d2": 1.9511,
                    "eval/loss_d3": 1.8757,
                    "eval/depth_gain": 0.1414,
                },
                n_params=83_061_638,
                target_steps=600,
            ),
        },
        # s2_joint_bptt (v1 HSSD through step 500)
        {
            "commit_id": "f1df5c9b00000000",
            "short_sha": "f1df5c9b",
            "created_at": "2026-11-21T05:33:03+00:00",
            "title": "[qloop:s2_joint_bptt] step=43 (ckpt) loss=5.4383 depth_gain=+0.0852",
            "message": "",
            "manifest": _make_manifest(
                "s2_joint_bptt",
                43,
                "in_progress",
                {
                    "loss/lm": 5.4383,
                    "depth": 3,
                    "tokens_seen": 10485760,
                    "tok_per_s": 453.6,
                    "optim/optim/global_grad_norm": 2.02,
                    "optim/optim/skipped_total": 0.0,
                    "mem/peak_gib": 72.51,
                    "hssd/vault_duty": 1.14e-6,
                    "eval/loss_d1": 1.3556,
                    "eval/loss_d2": 1.3275,
                    "eval/loss_d3": 1.2704,
                    "eval/depth_gain": 0.0852,
                },
                n_params=187_919_238,
                target_steps=1200,
            ),
        },
        {
            "commit_id": "91215ce100000000",
            "short_sha": "91215ce1",
            "created_at": "2026-11-23T00:13:09+00:00",
            "title": "[qloop:s2_joint_bptt] step=400 (ckpt) loss=5.01 depth_gain=+0.0728",
            "message": "",
            "manifest": _make_manifest(
                "s2_joint_bptt",
                400,
                "in_progress",
                {
                    "loss/lm": 5.01,
                    "depth": 3,
                    "tokens_seen": 104857600,
                    "tok_per_s": 962.6,
                    "optim/optim/global_grad_norm": 1.44,
                    "optim/optim/skipped_total": 0.0,
                    "loss/bptt_offload_to_cpu": 0.0,
                    "mem/peak_gib": 78.42,
                    "hssd/vault_duty": 3.26e-5,
                    "eval/loss_d1": 1.2600,
                    "eval/loss_d2": 1.2363,
                    "eval/loss_d3": 1.1872,
                    "eval/depth_gain": 0.0728,
                },
                n_params=187_919_238,
                target_steps=1200,
            ),
        },
        {
            "commit_id": "e050a4b800000000",
            "short_sha": "e050a4b8",
            "created_at": "2026-11-23T08:15:00+00:00",
            "title": "[qloop:s2_joint_bptt] step=500 (ckpt) loss=4.95",
            "message": "",
            "manifest": _make_manifest(
                "s2_joint_bptt",
                500,
                "in_progress",
                {
                    "loss/lm": 4.95,
                    "depth": 3,
                    "tokens_seen": 131072000,
                    "tok_per_s": 1011.0,
                    "optim/optim/global_grad_norm": 1.45,
                    "optim/optim/skipped_total": 0.0,
                    "loss/bptt_offload_to_cpu": 0.0,
                    "mem/peak_gib": 78.42,
                    "hssd/vault_duty": 3.26e-5,
                },
                n_params=187_919_238,
                target_steps=1200,
            ),
        },
        # Clean v2 s3_v2_long_1m_lsa (Native-GDN + mHC telemetry)
        {
            "commit_id": "ec34f82500000000",
            "short_sha": "ec34f825",
            "created_at": "2026-11-23T23:30:00+00:00",
            "title": "[qloop:s3_v2_long_1m_lsa] step=125 (ckpt) loss=1.55",
            "message": "",
            "manifest": _make_manifest(
                "s3_v2_long_1m_lsa",
                125,
                "in_progress",
                {
                    "loss/lm": 1.55,
                    "loss/lsa_kl": 21.0,
                    "loss/calib_prefix_len": 1015808.0,
                    "lsa/distill_kl/L3": 2.10,
                    "lsa/recall/L3": 0.5335,
                    "lsa/min_head_recall/L3": 0.3949,
                    "lsa/last_recall": 0.5335,
                    "loop/gate_abs_mean_p2": 0.0204,
                    "loop/gate_abs_mean_p3": 0.0325,
                    "loop/bridge_w_anchor_p3": 0.871,
                    "tokens_seen": 25000000,
                    "tok_per_s": 2410.0,
                    "optim/optim/global_grad_norm": 10.5,
                    "optim/optim/skipped_total": 0.0,
                    "mem/peak_gib": 69.52,
                },
                n_params=24_556_598,
                target_steps=600,
            ),
        },
        {
            "commit_id": "837fbc5d00000000",
            "short_sha": "837fbc5d",
            "created_at": "2026-11-24T01:28:15+00:00",
            "title": "[qloop:s3_v2_long_1m_lsa] step=200 (ckpt) loss=1.35 depth_gain=+0.0873",
            "message": "",
            "manifest": _make_manifest(
                "s3_v2_long_1m_lsa",
                200,
                "in_progress",
                {
                    "loss/lm": 1.35,
                    "loss/lsa_kl": 19.5,
                    "lsa/distill_kl/L3": 1.95,
                    "lsa/recall/L3": 0.6064,
                    "lsa/min_head_recall/L3": 0.4613,
                    "lsa/last_recall": 0.6064,
                    "loop/gate_abs_mean_p2": 0.0211,
                    "loop/gate_abs_mean_p3": 0.0427,
                    "loop/bridge_w_anchor_p3": 0.873,
                    "tokens_seen": 40000000,
                    "tok_per_s": 2280.0,
                    "optim/optim/global_grad_norm": 12.1,
                    "optim/optim/skipped_total": 0.0,
                    "mem/peak_gib": 69.96,
                    "eval/loss_d1": 1.3733,
                    "eval/loss_d2": 1.3011,
                    "eval/loss_d3": 1.2860,
                    "eval/depth_gain": 0.0873,
                },
                n_params=24_556_598,
                target_steps=600,
            ),
        },
        {
            "commit_id": "db49747400000000",
            "short_sha": "db497474",
            "created_at": "2026-11-24T05:48:06+00:00",
            "title": "[qloop:s3_v2_long_1m_lsa] step=375 (ckpt) loss=1.29",
            "message": "",
            "manifest": _make_manifest(
                "s3_v2_long_1m_lsa",
                375,
                "in_progress",
                {
                    "loss/lm": 1.29,
                    "loss/lsa_kl": 18.8,
                    "lsa/distill_kl/L3": 1.88,
                    "lsa/recall/L3": 0.5708,
                    "lsa/min_head_recall/L3": 0.4051,
                    "lsa/last_recall": 0.5708,
                    "loop/gate_abs_mean_p2": 0.0217,
                    "loop/gate_abs_mean_p3": 0.0489,
                    "loop/bridge_w_anchor_p3": 0.875,
                    "tokens_seen": 75268096,
                    "tok_per_s": 2350.0,
                    "optim/optim/global_grad_norm": 14.0,
                    "optim/optim/skipped_total": 0.0,
                    "mem/peak_gib": 69.96,
                },
                n_params=24_556_598,
                target_steps=600,
            ),
        },
    ]
    (offline_dir / "hub_commits.json").write_text(json.dumps(commits, indent=2), encoding="utf-8")

    probe = {
        "summary": {
            "num_gates_passed": 5,
            "num_gates_total": 5,
            "frozen_experts_sha256": "42f1be31694f1ad29abe",
            "total_seconds": 177.44,
        },
        "gates": {"0a_backward": {"passed": True, "value": 11.75}},
        "core_ranges": [
            {"prelude_end": 12, "core_end": 36, "loss_increase": 0.1386},
            {"prelude_end": 8, "core_end": 32, "loss_increase": 0.4516},
        ],
        "probes": {
            "expert_headroom": {
                "hard_token_headroom_nats": 1.3061,
                "improved_hard_token_frac": 0.8891,
                "mean_router_jaccard": 0.3365,
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

    truth_check = {
        "decode_bug_check": {
            "prefill_vs_decode_L70": {"d1": {"top1_match_rate": 1.0}, "d3": {"top1_match_rate": 1.0}},
            "prefill_vs_decode_L650_sparse": {
                "d1": {"top1_match_rate": 1.0, "mean_abs_diff": 0.1278},
                "d3": {"top1_match_rate": 1.0, "mean_abs_diff": 0.1073},
            },
            "uncached_oracle_reproduces_degradation": True,
        },
        "phase_a": {
            "stock": {"sanity_math": {"tok_per_s": 19.22}},
            "s2_step500_d1": {"sanity_math": {"tok_per_s": 7.72}},
            "s2_step500_d3": {"sanity_math": {"tok_per_s": 3.41}},
        },
        "forensics_4stage_summary": [
            {
                "checkpoint": "stock_plus_yarn1m",
                "depth": "stock",
                "needle_16k_pass": True,
                "math_rep_loop": False,
                "code_rep_loop": False,
                "math_d3": 0.8739,
                "code_d3": 1.0,
            },
            {
                "checkpoint": "s1b_hssd_distill_lsa_dense",
                "depth": "d=3",
                "needle_16k_pass": False,
                "math_rep_loop": False,
                "code_rep_loop": True,
                "math_d3": 0.1818,
                "code_d3": 0.1607,
            },
            {
                "checkpoint": "s3_long_1m_lsa_dense",
                "depth": "d=3",
                "needle_16k_pass": False,
                "math_rep_loop": True,
                "code_rep_loop": True,
                "math_d3": 0.1429,
                "code_d3": 0.0565,
            },
            {
                "checkpoint": "s2_with_s3_hssd",
                "depth": "d=3",
                "needle_16k_pass": False,
                "math_rep_loop": True,
                "code_rep_loop": False,
                "math_d3": 0.3904,
                "code_d3": 0.1494,
            },
            {
                "checkpoint": "s1_loop_a6b",
                "depth": "d=3",
                "needle_16k_pass": True,
                "math_rep_loop": False,
                "code_rep_loop": False,
                "math_d3": 0.8407,
                "code_d3": 0.9710,
            },
        ],
        "secret_test": FAKE_HF_TOKEN,
    }
    (offline_dir / "truth_check_summary.json").write_text(json.dumps(truth_check), encoding="utf-8")

    dplr_raw = {
        "timestamp_utc": "2026-11-23T19:58:21Z",
        "environment": {"hf_token": FAKE_HF_TOKEN},
        "part_a_v2_smoke": {"smoke_s3v2": {"elapsed_sec": 330.89}, "smoke_s2v2": {"elapsed_sec": 214.62}},
        "part_b_dplr_gdn2_bench": {
            "fla_version": "0.6.0",
            "parity": {
                "fp32": {
                    "kernels": {
                        "chunk_dplr_delta_rule": {
                            "e2e_gdn2_qed": {"forward_o": {"max_abs_diff": 0.00011, "cosine": 0.999999}},
                        }
                    },
                    "T1_decode_parity": {
                        "o": {"max_abs_diff": 3.73e-9, "cosine": 1.0, "rel_l2": 1e-7},
                        "final_state": {"max_abs_diff": 6e-8, "cosine": 1.0},
                    },
                },
                "bf16": {
                    "kernels": {
                        "chunk_dplr_delta_rule": {
                            "e2e_gdn2_qed": {
                                "forward_o": {"cosine": 0.999984},
                                "grads_e2e": {"q": {"cosine": 0.999966}},
                            }
                        }
                    },
                },
                "fla_ops_gdn2": {
                    "chunk_gdn2_bf16": {
                        "forward_o": {"cosine": 0.999986},
                        "grads": {"q": {"cosine": 0.999978}},
                    }
                },
                "stability_T4096_H32_bf16": {
                    "o_finite": True,
                    "s_finite": True,
                    "grads_finite": True,
                    "o_abs_max": 0.1167,
                    "s_abs_max": 1.7049,
                },
            },
            "benchmarks": {
                "T32768_fwd_bwd": {
                    "native_chunk_gated_delta_rule": {
                        "fwd": {"median_ms": 3.64},
                        "bwd": {"median_ms": 10.21},
                        "total": {"median_ms": 13.85},
                    },
                    "fla_ops_gdn2_chunk_gdn2": {
                        "fwd": {"median_ms": 8.00},
                        "bwd": {"median_ms": 27.55},
                        "total": {"median_ms": 35.55},
                    },
                    "fla_chunk_dplr_raw": {
                        "fwd": {"median_ms": 12.59},
                        "bwd": {"median_ms": 31.05},
                        "total": {"median_ms": 43.64},
                    },
                    "fla_chunk_dplr_gdn2_qed_e2e": {
                        "fwd": {"median_ms": 19.84},
                        "bwd": {"median_ms": 50.00},
                        "total": {"median_ms": 69.84},
                    },
                    "pytorch_chunk_gdn2": {
                        "fwd": {"median_ms": 92.56},
                        "bwd": {"median_ms": 565.20},
                        "total": {"median_ms": 657.32},
                    },
                },
                "T1_decode": {
                    "native_fused_recurrent_gated_delta_rule": {"median_ms": 0.0421},
                    "fla_fused_recurrent_dplr_raw": {"median_ms": 0.0430},
                    "fla_ops_gdn2_fused_recurrent_gdn2": {"median_ms": 0.0480},
                    "pytorch_recurrent_gdn2": {"median_ms": 0.1279},
                    "fla_fused_recurrent_dplr_gdn2_qed_e2e": {"median_ms": 0.1512},
                },
            },
        },
    }
    (offline_dir / "dplr_bench_summary.json").write_text(json.dumps(dplr_raw), encoding="utf-8")

    ds_audit = {
        "repo_id": "TassP0lar/Loop64-Fable-SIRSI-v1",
        "release_version": "v1.2",
        "output_counts": {"sft_core": 4081, "salvage_queue": 2076, "sirsi_preferences": 429},
        "licenses": {
            "repo_a": {"decision": "include"},
            "repo_b": {"decision": "exclude"},
        },
        "dedup": {"unique_indexed": 6525, "duplicates_dropped": 2107},
        "decontam": {"total_ngrams": 1889432, "n_suites_accessible": 12, "leaks_dropped": 31},
        "v11_fable_upgrade": {"rows_with_strict_pass": 9, "rows_synthesized": 220},
        "v12_fable_notebook": {
            "rows_strict_pass": 197,
            "rows_written": 197,
            "word_count_min": 361,
            "word_count_max": 478,
            "word_count_median": 437,
            "mean_slop": 0.0,
            "mean_3beat": 0.8156,
        },
    }
    (offline_dir / "dataset_audit_summary.json").write_text(json.dumps(ds_audit), encoding="utf-8")

    ka = "session=loop64 pid_alive=True exit_code=None disk_free=209.2GiB gpu=RTX"
    sup_log = "\n".join(
        [
            f"2026-11-20T22:27:20Z [URL] session=loop64 {FAKE_TUNNEL}",
            "2026-11-20T22:27:21Z [BOOT] ready session=loop64 launched=True pid=31606 head=2e37bf6",
            f"2026-11-21T00:09:24Z [KEEPALIVE] {ka} util=98% mem=76000/97887MiB pwr=349.8W temp=55C"
            " | 00:08:17 I [-] step 10/1200 lm 5.0330 d=2 481 tok/s"
            " mem alloc 32.4G / reserved 73.6G / peak 72.5G",
            f"2026-11-21T00:19:24Z [KEEPALIVE] {ka} util=98% mem=76500/97887MiB pwr=339.8W temp=56C"
            " | 00:18:35 I [-] step 40/1200 lm 5.4383 d=3 454 tok/s"
            " mem alloc 32.6G / reserved 74.2G / peak 72.5G",
            "2026-11-21T06:28:48Z [BOOT] ready session=loop64 launched=True pid=154438 head=b2c5513",
            f"2026-11-21T07:57:13Z [KEEPALIVE] {ka} util=99% mem=97200/97887MiB pwr=358.5W temp=58C"
            " | 07:54:25 I [-] step 60/1200 lm 6.3484 d=3 930 tok/s"
            " mem alloc 52.0G / reserved 93.9G / peak 89.8G",
            "2026-11-21T10:41:12Z [BOOT] ready session=loop64 launched=True pid=53372 head=5a82677",
            f"2026-11-21T11:38:20Z [KEEPALIVE] {ka} util=99% mem=76147/97887MiB pwr=381.4W temp=58C"
            " | 11:36:00 I [-] step 60/1200 lm 5.0390 d=2 892 tok/s"
            " mem alloc 25.3G / reserved 73.3G / peak 70.0G",
            f"2026-11-21T11:48:20Z [KEEPALIVE] {ka} util=99% mem=76147/97887MiB pwr=372.2W temp=58C"
            " | 11:45:53 I [-] step 270/1200 lm 6.1663 d=3 856 tok/s"
            " mem alloc 25.4G / reserved 73.3G / peak 70.1G",
            "2026-11-22T08:00:00Z [BOOT] ready session=loop64 launched=True pid=90001 head=8409b6e",
            f"2026-11-22T08:10:00Z [KEEPALIVE] {ka} util=99% mem=91409/97887MiB pwr=252.0W temp=52C | slow",
            f"2026-11-22T08:20:00Z [KEEPALIVE] {ka} util=99% mem=91409/97887MiB pwr=252.0W temp=52C | slow",
            f"2026-11-22T18:20:00Z [KEEPALIVE] {ka} util=98% mem=80100/97887MiB pwr=373.2W temp=58C"
            " | 18:19:00 I [-] step 350/1200 lm 5.1000 d=3 942 tok/s"
            " mem alloc 25.3G / reserved 79.0G / peak 78.4G",
            f"2026-11-22T18:30:00Z [KEEPALIVE] {ka} util=98% mem=80100/97887MiB pwr=373.2W temp=58C"
            " | 18:29:00 I [-] step 490/1200 lm 4.9500 d=3 1011 tok/s"
            " mem alloc 25.3G / reserved 79.0G / peak 78.4G",
        ]
    )
    (offline_dir / "colab_supervisor.log").write_text(sup_log + "\n", encoding="utf-8")

    sup_v2_log = "\n".join(
        [
            "2026-11-23T20:00:00Z [BOOT] ready session=loop64 launched=True pid=44001 head=28b88e4",
            f"2026-11-23T20:10:00Z [KEEPALIVE] {ka} util=99% mem=71000/97887MiB pwr=505.1W temp=62C"
            " | 20:09:00 I [-] step 25/600 lm 1.6500 d=2 2180 tok/s"
            " mem alloc 25.0G / reserved 68.0G / peak 69.5G",
            f"2026-11-23T20:20:00Z [KEEPALIVE] {ka} util=99% mem=71000/97887MiB pwr=505.1W temp=62C"
            " | 20:19:00 I [-] step 375/600 lm 1.2900 d=3 2210 tok/s"
            " mem alloc 25.1G / reserved 68.5G / peak 69.9G",
        ]
    )
    (offline_dir / "colab_supervisor_v2.log").write_text(sup_v2_log + "\n", encoding="utf-8")


def test_offline_export_e2e_v2_artifacts_7_svgs_and_secret_redaction(tmp_path: Path) -> None:
    offline_dir = tmp_path / "offline"
    out_dir = tmp_path / "experiment_logs"
    report_path = tmp_path / "EXPERIMENT_REPORT.md"
    _build_offline_fixture(offline_dir)

    res = export_experiment_report(
        repo="test/fake-loop64",
        supervisor_log=str(offline_dir / "colab_supervisor.log"),
        supervisor_v2_log=str(offline_dir / "colab_supervisor_v2.log"),
        out_dir=str(out_dir),
        report_path=str(report_path),
        offline_dir=str(offline_dir),
    )

    assert res["total_commits"] == 22
    assert res["rolled_back_count"] == 3  # 1 poisoned s1 + 2 pre-fix s1b explosion commits
    assert res["ingested_artifacts"] == {
        "truth_check": True,
        "dplr_bench": True,
        "dataset_audit": True,
        "supervisor_v2": True,
    }

    # 1. Verify hub_manifest_history.json flags rollbacks and strips secrets
    hist = json.loads((out_dir / "hub_manifest_history.json").read_text(encoding="utf-8"))
    rb_shas = {c["short_sha"] for c in hist["commits"] if c["rolled_back"]}
    assert rb_shas == {"d1d92064", "7a2d590a", "dc7c4ad8"}

    # 2. Verify CSV deduplication across all 5 stages (incl. clean v2 s3_v2_long_1m_lsa)
    assert "s3_v2_long_1m_lsa" in STAGES
    for st in STAGES:
        csv_p = out_dir / f"{st}.csv"
        assert csv_p.is_file()
        with open(csv_p, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        steps = [int(r["step"]) for r in rows]
        assert steps == sorted(steps)
        assert len(steps) == len(set(steps))
        assert len(rows) >= 2

    with open(out_dir / "s3_v2_long_1m_lsa.csv", encoding="utf-8") as f:
        s3v2_rows = list(csv.DictReader(f))
    assert [int(r["step"]) for r in s3v2_rows] == [125, 200, 375]
    assert float(s3v2_rows[-1]["gate_abs_mean_p3"]) == pytest.approx(0.0489)
    assert float(s3v2_rows[1]["eval_loss_d3"]) == pytest.approx(1.2860)

    # 3. Verify summary_metrics.json & archived local artifact JSONs
    sm = json.loads((out_dir / "summary_metrics.json").read_text(encoding="utf-8"))
    assert sm["stages"]["s2_joint_bptt"]["latest_step"] == 500
    assert sm["stages"]["s3_v2_long_1m_lsa"]["latest_step"] == 375
    assert sm["stages"]["s3_v2_long_1m_lsa"]["latest_commit_sha"] == "db497474"
    assert sm["gpu_hours"]["supervisor_v1_active_h"] > 0
    assert sm["gpu_hours"]["supervisor_v2_active_h"] > 0
    assert "post_tf524_fix_2f5a6ba" in sm["supervisor_s2_regimes"]
    assert sm["tf524_slowdown_window"]["slowdown_hours"] > 0

    for art in ("truth_check_summary.json", "dplr_bench_summary.json", "dataset_audit_summary.json"):
        assert (out_dir / art).is_file()
    dplr_arch = json.loads((out_dir / "dplr_bench_summary.json").read_text(encoding="utf-8"))
    assert dplr_arch["parity_summary"]["fp32_T1_decode_o"]["max_abs_diff"] == pytest.approx(3.73e-9)

    # 4. Verify all 7 SVG charts parse cleanly as valid XML with real plotted primitives
    for fname in EXPECTED_SVGS:
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
        assert n_geom >= 8, f"{fname} missing plotted geometry"
        assert len(root.findall(f".//{SVG_NS}text")) >= 10, f"{fname} missing axis/labels"
        assert "hbm" not in svg_txt.lower(), f"Erroneous HBM reference found in {fname}"

    # 5. Verify 7-section v3 EXPERIMENT_REPORT.md structure, 5 strict upgrades & relative SVG links
    md = report_path.read_text(encoding="utf-8")
    for fname in EXPECTED_SVGS:
        assert f"experiment_logs/charts/{fname}" in md
    for heading in (
        "## 1. Executive Summary",
        "## 2. Methodology, Architecture & Full VRAM Footprint Ledger",
        "### 2.1 Foundational Single-GPU Enabler: APOLLO Optimizer (`arXiv:2412.05270`) & `FusedStepOptimizer`",
        "### 2.2 Verified Blackwell Hardware Spec (`GDDR7 ECC`) & Three-Axis Looped Hybrid Architecture",
        "### 2.3 Inference `1M`-Context (`88L @ R=3`, `batch=1`) VRAM Footprint Ledger & `BucketIndex` Code Audit",
        "### 2.4 Single-Card Training VRAM Footprint Ledger (`96 GB GDDR7 ECC` / `94.97 GiB` Usable)",
        "## 3. Full Stage-by-Stage Empirical Results",
        "### 3.7 Phase-A Truth-Check & 4-Stage Causal Swap Forensics",
        "#### 3.7.1 Deep HSSD v3.2 Experimental Post-Mortem & Why `GDN-2` Is the Principled Move",
        "### 3.8 Controlled A/B Win: Clean Native-GDN `s3_v2_long_1m_lsa`",
        (
            "## 4. GDN-2-1M (`arXiv:2605.22791`) + Bounded QED (`arXiv:2608.13668`) + "
            "Soft Neg-Eigval via `fla 0.6.0` `DPLR`"
        ),
        "## 5. Open Audited Dataset",
        "## 6. Blackwell `sm_120` Speedup Chronology & Kernel Engineering Ledger",
        "## 7. Honest Limitations, Next Milestones & Compute Grant Justification",
    ):
        assert heading in md, f"Missing expected section heading: {heading}"
    assert "3.73e-09" in md
    assert "halted@500" in md
    assert "hbm" not in md.lower(), "Erroneous HBM reference found in EXPERIMENT_REPORT.md"
    assert "hbm" not in json.dumps(sm).lower(), "Erroneous HBM reference found in summary_metrics.json"
    for required_token in (
        "GDDR7",
        "1,597 GB/s",
        "71.9%",
        "arXiv:2412.05270",
        "FusedStepOptimizer",
        "_foreach_norm",
        "+1.04 GiB",
        "33.54 GiB",
        "11.63 GiB",
        "8,192",
        "~47–53",
        "5.0×",
        "2.50×",
        "425–775",
        "1.28e-4",
        "~129 TB/step",
        "93,700+",
        "2e37bf6",
        "5a82677",
        "2f5a6ba",
        "2c78ffc",
    ):
        assert required_token in md, f"Missing required v3 report content: {required_token}"

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
            "--repo",
            "test/fake-loop64",
            "--offline-dir",
            str(offline_dir),
            "--out-dir",
            str(out_dir),
            "--report",
            str(report_path),
        ]
    )
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    # 1 history + 5 CSVs + 1 summary + 3 archived JSONs + 7 SVGs + 1 report = 18 files
    assert len(out["written_files"]) == 18
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
    int4 = rows["Clean v2 Hadamard INT4 Serving (88L)"]
    v2 = rows["Clean v2 Native-GDN / GDN-2-1M (88L @ R=3)"]
    hssd = rows["LoopHSSD v3.2 (88L @ R=3)"]
    inkling = rows["Inkling 2 Mini (48L @ R=3)"]
    unshared_88 = rows["Standard Unshared 88L Full-Attn"]

    assert int4["kv_gib"] == pytest.approx(5.31)
    assert int4["total_gib"] == pytest.approx(5.96)
    assert v2["highlight"] is True
    assert v2["kv_gib"] == pytest.approx(10.00)
    assert v2["linear_or_swa_gib"] == pytest.approx(0.33)
    assert v2["indexer_gib"] == pytest.approx(0.32)
    assert v2["total_gib"] == pytest.approx(10.65)
    assert hssd["total_gib"] == pytest.approx(15.82)
    assert inkling["kv_gib"] == pytest.approx(32.00)
    assert inkling["total_gib"] == pytest.approx(32.16)
    assert inkling["total_gib"] / v2["total_gib"] == pytest.approx(3.02, abs=0.01)
    assert unshared_88["kv_gib"] == pytest.approx(176.00)


def test_parse_supervisor_log_missing_returns_empty() -> None:
    res = parse_supervisor_log("/nonexistent/path/colab_supervisor.log")
    assert res["present"] is False
    assert res["step_samples"] == []
    assert res["active_gpu_hours_total"] == 0.0


def test_parse_supervisor_log_boot_resets_stage_and_tf524_hours_reconcile(tmp_path: Path) -> None:
    ka = "session=loop64 pid_alive=True exit_code=None disk_free=209.2GiB gpu=RTX"
    log_p = tmp_path / "sup.log"
    log_p.write_text(
        "\n".join(
            [
                "2026-11-18T00:00:00Z [BOOT] ready session=loop64 launched=True pid=1 head=3a17c0d",
                f"2026-11-18T00:05:00Z [KEEPALIVE] {ka} util=99% mem=70000/97887MiB pwr=350.0W temp=55C"
                " | run dir: /content/runs/s1b_hssd_distill",
                f"2026-11-18T00:10:00Z [KEEPALIVE] {ka} util=99% mem=70000/97887MiB pwr=350.0W temp=55C"
                " | 00:09:00 I [-] step 500/1500 lm 1.2000 d=2 900 tok/s"
                " mem alloc 24.0G / reserved 68.0G / peak 68.9G",
                "2026-11-19T10:00:00Z [BOOT] ready session=loop64 launched=True pid=2 head=513879f",
                f"2026-11-19T10:10:00Z [KEEPALIVE] {ka} util=99% mem=81000/97887MiB pwr=360.0W temp=56C"
                " | 10:09:00 I [-] step 10/600 lm 2.4000 d=3 1050 tok/s"
                " mem alloc 25.0G / reserved 79.0G / peak 80.7G",
                "2026-11-22T17:50:00Z [BOOT] ready session=loop64 launched=True pid=3 head=8409b6e",
                f"2026-11-22T17:55:00Z [KEEPALIVE] {ka} util=99% mem=91409/97887MiB pwr=252.0W temp=52C | slow",
                f"2026-11-22T18:05:00Z [KEEPALIVE] {ka} util=99% mem=91409/97887MiB pwr=252.0W temp=52C | slow",
                f"2026-11-22T18:10:00Z [KEEPALIVE] {ka} util=98% mem=80100/97887MiB pwr=373.0W temp=58C | fast",
                f"2026-11-22T18:25:00Z [KEEPALIVE] {ka} util=98% mem=80100/97887MiB pwr=373.0W temp=58C | fast",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    res = parse_supervisor_log(str(log_p))
    s3_samples = [s for s in res["step_samples"] if s["head"] == "513879f"]
    assert len(s3_samples) == 1
    assert s3_samples[0]["stage"] == "s3_long_1m_lsa"

    tf = res["tf524_slowdown_window"]
    assert tf is not None
    assert tf["slowdown_hours"] + tf["post_fix_hours"] == pytest.approx(
        res["active_gpu_hours_by_head"]["8409b6e"], abs=1e-6
    )


def test_committed_docs_artifacts_xml_and_secret_free() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    report_p = repo_root / "docs" / "EXPERIMENT_REPORT.md"
    grant_p = repo_root / "docs" / "COMPUTE_GRANT_PROPOSAL.md"
    perf_p = repo_root / "docs" / "PERF_PROFILE_S2.md"
    logs_dir = repo_root / "docs" / "experiment_logs"
    if not report_p.is_file() or not logs_dir.is_dir():
        pytest.skip("Generated docs/experiment_logs not present")

    report_txt = report_p.read_text(encoding="utf-8")
    assert not contains_secret(report_txt)
    assert "hbm" not in report_txt.lower(), "Erroneous HBM reference in committed EXPERIMENT_REPORT.md"
    assert "1,597 GB/s" in report_txt and "71.9%" in report_txt and "GDDR7" in report_txt
    if grant_p.is_file():
        grant_txt = grant_p.read_text(encoding="utf-8")
        assert not contains_secret(grant_txt)
        assert "hbm" not in grant_txt.lower(), "Erroneous HBM reference in committed COMPUTE_GRANT_PROPOSAL.md"
        assert "GDDR7" in grant_txt and "1,597 GB/s" in grant_txt and "71.9%" in grant_txt
        assert "arXiv:2412.05270" in grant_txt and "FusedStepOptimizer" in grant_txt
        assert "10.65 GiB" in grant_txt
        assert "+1.04 GiB" in grant_txt
        assert "2,500" in grant_txt
    if perf_p.is_file():
        perf_txt = perf_p.read_text(encoding="utf-8")
        assert not contains_secret(perf_txt)
        assert "hbm" not in perf_txt.lower(), "Erroneous HBM reference in committed PERF_PROFILE_S2.md"
        assert "GDDR7" in perf_txt and "1,597 GB/s" in perf_txt

    for fname in EXPECTED_SVGS:
        assert (logs_dir / "charts" / fname).is_file(), f"Missing committed chart {fname}"

    for p in sorted(logs_dir.rglob("*")):
        if not p.is_file():
            continue
        txt = p.read_text(encoding="utf-8")
        assert not contains_secret(txt), f"Potential secret in committed file {p}"
        if p.suffix == ".svg":
            assert "hbm" not in txt.lower(), f"Erroneous HBM reference in committed chart {p.name}"
            root = ET.fromstring(txt)
            assert root.tag == f"{SVG_NS}svg"
        elif p.suffix == ".csv":
            with open(p, encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            steps = [int(r["step"]) for r in rows]
            assert steps == sorted(steps)
            assert len(steps) == len(set(steps))
