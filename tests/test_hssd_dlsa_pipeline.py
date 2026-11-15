"""Verification suite for the Stock -> HSSD v3.2 + D-LSA pipeline, 96 GiB Blackwell batching,
zero-hook `teacher_track="stock"` distillation, HSSD+D-LSA Hub model cards, and step-0 base-weight
corruption fail-fast guard.
"""
from __future__ import annotations

import os
from typing import Any

import pytest
import torch

from qloop.config import Cfg, load_config
from qloop.hub import (
    V3_PIPELINE_ORDER,
    HubManifest,
    HubSyncManager,
    render_model_card,
    resolve_stage_config_path,
)
from qloop.model.hssd import HSSDAdapter, HSSDConfig
from qloop.model.lsa import LSAConfig
from qloop.train.distill import (
    HSSDDistillConfig,
    SharedBackboneTeacher,
    compute_shared_distill_loss,
    run_layerwise_warmup_microstep,
)
from qloop.train.stack import build_stack
from qloop.train.trainer import LMTrainer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_ROOT = os.path.join(ROOT, "configs")


def test_base_and_v32_pipeline_retire_gdn2_and_configure_blackwell_batching():
    base = load_config(os.path.join(CONFIG_ROOT, "base.yaml"))
    assert base["gdn2"]["enabled"] is False
    assert base["train"]["optim"]["lr_mult"]["hssd_adapter"] == 10.0

    cfgs = {
        st: load_config(resolve_stage_config_path(st, CONFIG_ROOT))
        for st in V3_PIPELINE_ORDER
    }
    assert tuple(cfgs.keys()) == (
        "s0_probe",
        "s1_loop_a6b",
        "s1b_hssd_distill",
        "s3_long_1m_lsa",
        "s2_joint_bptt",
        "s4_rl",
    )

    for st, cfg in cfgs.items():
        assert cfg["gdn2"]["enabled"] is False, f"{st} must keep gdn2.enabled=False"

    # S0 probe: stock Qwen3.6-35B-A3B-FP8 (loop, gdn2, hssd, lsa all off)
    s0 = cfgs["s0_probe"]
    assert s0["loop"]["enabled"] is False
    assert s0["gdn2"]["enabled"] is False
    assert s0["hssd"]["enabled"] is False
    assert s0["lsa"]["enabled"] is False

    # S1 loop healing: heals 88L loop directly over fast stock linear attention at mb=4 x accum=4
    s1 = cfgs["s1_loop_a6b"]
    assert s1["gdn2"]["enabled"] is False
    assert s1["hssd"]["enabled"] is False
    assert s1["lsa"]["enabled"] is False
    assert s1["train"]["train_gdn2"] is False
    assert "gdn2_adapters" not in s1["train"]["allowed_groups"]
    assert "gdn2_adapter" not in s1["train"]["optim"]["lr_mult"]
    assert s1["train"]["micro_batch"] == 4
    assert s1["train"]["grad_accum"] == 4
    assert s1["train"]["micro_batch"] * s1["train"]["grad_accum"] * s1["data"]["seq_len"] == 131_072

    # S1b HSSD distill: stock teacher track activates zero-hook teacher_mode() fast path
    s1b = cfgs["s1b_hssd_distill"]
    assert s1b["hssd"]["enabled"] is True
    assert s1b["distill"]["enabled"] is True
    assert s1b["distill"]["teacher_track"] == "stock"


def test_hssd_and_dlsa_schema_parity_and_strict_state_dict_transfer_across_stages():
    hssd_stages = ("s1b_hssd_distill", "s3_long_1m_lsa", "s2_joint_bptt", "s4_rl")
    cfgs = {
        st: load_config(resolve_stage_config_path(st, CONFIG_ROOT))
        for st in hssd_stages
    }

    adapters: dict[str, HSSDAdapter] = {}
    for st, cfg in cfgs.items():
        hraw = dict(cfg["hssd"])
        assert hraw.pop("enabled") is True, st
        assert hraw["swa_recency_init"] == 0.05, st
        assert hraw["write_mode"] == "full", st
        assert hraw["num_pages"] == 8, st
        assert hraw["k_pages"] == 2, st
        assert hraw["theta_low_rank"] is True, st
        assert hraw["compact_active_pages"] is True, st
        assert hraw["compact_decode_cache"] is True, st

        hcfg = HSSDConfig.from_dict(hraw)
        hcfg.validate()
        adapters[st] = HSSDAdapter(
            hidden_size=64,
            num_v_heads=4,
            head_k_dim=16,
            head_v_dim=16,
            cfg=hcfg,
            layer_id=0,
        )

    # Perturb s1b adapter weights (including swa_recency_slope) and verify exact strict=True transfer
    # across s1b -> s3 -> s2 -> s4 with zero missing or unexpected keys.
    s1b_ad = adapters["s1b_hssd_distill"]
    assert hasattr(s1b_ad, "swa_recency_slope") and isinstance(s1b_ad.swa_recency_slope, torch.nn.Parameter)
    with torch.no_grad():
        for i, p in enumerate(s1b_ad.parameters()):
            p.add_(0.03 * float(i + 1))
    ref_sd = {k: v.detach().clone() for k, v in s1b_ad.state_dict().items()}

    prev_sd = ref_sd
    for next_st in ("s3_long_1m_lsa", "s2_joint_bptt", "s4_rl"):
        ad = adapters[next_st]
        incompat = ad.load_state_dict(prev_sd, strict=True)
        assert incompat.missing_keys == []
        assert incompat.unexpected_keys == []
        for k, v in ref_sd.items():
            assert torch.equal(ad.state_dict()[k], v), f"{next_st} parameter mismatch on {k}"
        prev_sd = ad.state_dict()

    # Verify S3, S2, S4 all enable D-LSA (HSSD + D-LSA) and S4 matches S2 HSSD + LSA schema
    for st in ("s3_long_1m_lsa", "s2_joint_bptt", "s4_rl"):
        lraw = dict(cfgs[st]["lsa"])
        assert lraw.pop("enabled") is True, st
        assert lraw["compact_active_slots"] is True, st
        assert lraw["max_selected_keys"] == 768, st
        assert lraw["num_query_passes"] == 3, st
        LSAConfig.from_dict(lraw).validate()

    assert dict(cfgs["s4_rl"]["hssd"]) == dict(cfgs["s2_joint_bptt"]["hssd"])
    assert dict(cfgs["s4_rl"]["lsa"]) == dict(cfgs["s2_joint_bptt"]["lsa"])


def test_distill_config_validation_and_stock_shared_teacher_fast_path(tmp_path):
    for valid_track in ("stock", "auto", "gdn2"):
        dc = HSSDDistillConfig(teacher_track=valid_track)
        dc.validate()
        assert dc.teacher_track == valid_track

    with pytest.raises(ValueError, match="teacher_track"):
        HSSDDistillConfig(teacher_track="invalid_track").validate()

    # 1. Mock s1_loop_a6b checkpoint trained over stock linear attention (gdn2=None)
    s1_dir = tmp_path / "s1_loop_a6b"
    s1_cfg = Cfg({
        "seed": 11,
        "stage": "s1_loop_a6b",
        "model": {"name": "toy", "dtype": "float32", "device_map": "cpu", "toy": {"seed": 11}},
        "loop": {
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
        },
        "gdn2": {"enabled": False},
        "hssd": {"enabled": False},
        "lsa": {"enabled": False},
        "data": {"seq_len": 16, "sources": [{"name": "toy_text", "weight": 1.0}]},
        "train": {
            "steps": 1,
            "micro_batch": 1,
            "grad_accum": 1,
            "ckpt_every": 1,
            "eval_every": 0,
            "train_loop_adapters": True,
            "train_gdn2": False,
            "trainable": [r"input_layernorm|post_attention_layernorm"],
            "allowed_groups": ["loop_adapters", "norms"],
            "optim": {"lr": 5e-3, "warmup_steps": 1, "impl": "builtin"},
        },
    })
    s1_tr = LMTrainer(s1_cfg, str(s1_dir))
    assert s1_tr.stack.gdn2 is None
    s1_tr.train()

    ids = torch.randint(4, 100, (1, 16))
    s1_tr.stack.model.eval()
    s1_tr.stack.loop.set_active_passes(2)
    with torch.no_grad():
        expected_teacher_h = s1_tr.stack.hidden(ids).detach().clone()

    # 2. Build S1b HSSD student initialized from s1_loop_a6b, perturb student params, attach stock teacher
    s1b_cfg = Cfg({
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
        "train": {
            **dict(s1_cfg["train"]),
            "init_from": str(s1_dir),
            "train_hssd": True,
            "allowed_groups": ["loop_adapters", "hssd_adapters", "norms"],
        },
    })
    student_st = build_stack(s1b_cfg)
    assert student_st.hssd is not None and student_st.gdn2 is None

    # Perturb student loop/norm params so we prove SharedBackboneTeacher swaps to s1_loop_a6b deltas
    with torch.no_grad():
        for name, p in student_st.named_trainable.items():
            if "hssd_adapter." not in name:
                p.add_(0.25)

    teacher = SharedBackboneTeacher(
        student_st,
        teacher_ckpt=str(s1_dir),
        teacher_track="stock",
        run_root=str(tmp_path),
    )
    assert teacher.loaded_from_ckpt is True
    assert teacher.teacher_track == "stock"
    assert len(teacher.gdn2_adapters) == 0

    # Verify Phase 2 uses HSSDHandle.teacher_mode() zero-hook fast path (bypass_to_orig == True, 0 extra hooks)
    lin_attn_0 = student_st.parts.layers[0].linear_attn
    pre_hooks_before = len(lin_attn_0._forward_pre_hooks)
    assert student_st.hssd._runtime.bypass_to_orig is False
    student_st.loop.set_active_passes(2)
    with teacher.activate(), torch.no_grad():
        assert student_st.hssd._runtime.bypass_to_orig is True
        assert len(lin_attn_0._forward_pre_hooks) == pre_hooks_before
        got_teacher_h = student_st.hidden(ids)
    assert student_st.hssd._runtime.bypass_to_orig is False
    assert torch.allclose(got_teacher_h, expected_teacher_h, atol=1e-6)

    # Verify both Phase 1 warmup and Phase 2 shared KL distillation execute cleanly
    opt = torch.optim.AdamW(list(student_st.named_trainable.values()), lr=1e-3)
    w_stats = run_layerwise_warmup_microstep(student_st, teacher, opt, ids, depth=2)
    assert w_stats["distill/phase"] == 1.0 and w_stats["distill/calls"] > 0

    dout = compute_shared_distill_loss(
        student_st,
        teacher,
        HSSDDistillConfig(teacher_track="stock", chunk_size=32, token_chunk_size=16, hidden_mse_weight=0.1),
        ids,
        labels=ids,
        depth=2,
        step=500,
    )
    dout.loss.backward()
    assert torch.isfinite(dout.loss)
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0
        for n, p in student_st.named_trainable.items()
        if "hssd_adapter." in n
    )


def test_step0_corrupted_base_weight_fail_fast_guard(tmp_path, monkeypatch):
    def _make_cfg(model_name: str, max_step0_loss: float = 10.0, eval_every: int = 10) -> Cfg:
        return Cfg({
            "seed": 0,
            "stage": "s1_loop_a6b",
            "model": {"name": model_name, "dtype": "float32", "device_map": "cpu"},
            "loop": {
                "enabled": True,
                "prelude_end": 4,
                "core_end": 12,
                "num_passes": 2,
                "target_virtual_layers": 24,
                "checkpoint": False,
            },
            "data": {"seq_len": 16, "sources": [{"name": "toy_text", "weight": 1.0}]},
            "eval_quick": {"n_batches": 1, "depths": [1, 2]},
            "train": {
                "steps": 2,
                "micro_batch": 1,
                "grad_accum": 1,
                "eval_every": eval_every,
                "max_step0_loss": max_step0_loss,
                "train_loop_adapters": True,
                "optim": {"lr": 1e-3, "warmup_steps": 1, "impl": "builtin"},
            },
        })

    toy_stack_cfg = _make_cfg("toy")

    # 1. Corrupted non-toy checkpoint (`eval/loss_d1 = 14.89 > 10.0`) aborts BEFORE step 1
    st_bad = build_stack(toy_stack_cfg)
    bad_cfg = _make_cfg("unsloth/Qwen3.6-35B-A3B-NVFP4-Fast", max_step0_loss=10.0, eval_every=200)
    tr_bad = LMTrainer(bad_cfg, str(tmp_path / "corrupted_nvfp4"), stack=st_bad)
    monkeypatch.setattr(tr_bad, "quick_eval", lambda: {"eval/loss_d1": 14.8912, "eval/loss_d2": 14.8912})
    with pytest.raises(RuntimeError, match=r"Step-0 eval loss \(14\.8912\) exceeds train\.max_step0_loss=10\.00"):
        tr_bad.train()
    assert tr_bad.step == 0, "Step-0 guard must abort before executing step 1"
    assert tr_bad.tokens_seen == 0

    # 2. Also catches non-finite NaN step-0 eval loss even when eval_every=0
    st_nan = build_stack(toy_stack_cfg)
    nan_cfg = _make_cfg("Qwen/Qwen3.6-35B-A3B-FP8", max_step0_loss=10.0, eval_every=0)
    tr_nan = LMTrainer(nan_cfg, str(tmp_path / "nan_step0"), stack=st_nan)
    monkeypatch.setattr(tr_nan, "quick_eval", lambda: {"eval/loss_d1": float("nan")})
    with pytest.raises(RuntimeError, match="Step-0 eval loss"):
        tr_nan.train()
    assert tr_nan.step == 0

    # 3. Disabling via train.max_step0_loss=0 allows training to proceed
    st_disabled = build_stack(toy_stack_cfg)
    dis_cfg = _make_cfg("unsloth/Qwen3.6-35B-A3B-NVFP4-Fast", max_step0_loss=0.0, eval_every=200)
    tr_dis = LMTrainer(dis_cfg, str(tmp_path / "guard_disabled"), stack=st_disabled)
    monkeypatch.setattr(tr_dis, "quick_eval", lambda: {"eval/loss_d1": 14.8912, "eval/loss_d2": 14.8912})
    res_dis = tr_dis.train()
    assert res_dis["step"] == 2

    # 4. Healthy FP8 base (`eval/loss_d1 = 0.9987 <= 10.0`) proceeds cleanly
    st_ok = build_stack(toy_stack_cfg)
    ok_cfg = _make_cfg("Qwen/Qwen3.6-35B-A3B-FP8", max_step0_loss=10.0, eval_every=200)
    tr_ok = LMTrainer(ok_cfg, str(tmp_path / "healthy_fp8"), stack=st_ok)
    monkeypatch.setattr(tr_ok, "quick_eval", lambda: {"eval/loss_d1": 0.9987, "eval/loss_d2": 0.9987})
    res_ok = tr_ok.train()
    assert res_ok["step"] == 2


def test_hub_model_card_renders_hssd_dlsa_vs_stock_vs_legacy_gdn2(tmp_path):
    # Default & v3 HSSD + D-LSA manifest
    m_hssd = HubManifest(
        architecture={
            "physical_layers": 40,
            "virtual_layers": 88,
            "prelude_end": 12,
            "core_end": 36,
            "num_passes": 3,
            "linear_attention": "hssd",
        },
        stages={"s1b_hssd_distill": {"status": "completed", "step": 1500, "target_steps": 1500}},
    )
    card_hssd = render_model_card(m_hssd)
    assert "  - hssd" in card_hssd
    assert "Horizon-Split Surprise-Gated Delta Memory (`HSSD v3.2`)" in card_hssd
    assert "Distilled Latent Sparse Attention (`D-LSA`)" in card_hssd
    assert "Gated DeltaNet-2" not in card_hssd
    assert "gated-deltanet-2" not in card_hssd

    # Stock linear attention stage (e.g. s1_loop_a6b before s1b_hssd_distill)
    m_stock = HubManifest(architecture={"linear_attention": "stock"}, stages={"s1_loop_a6b": {"status": "in_progress"}})
    card_stock = render_model_card(m_stock)
    assert "  - gated-deltanet" in card_stock
    assert "Gated DeltaNet-2" not in card_stock

    # Legacy GDN-2 manifest backward compatibility
    m_gdn2 = HubManifest(architecture={"linear_attention": "gdn2"}, stages={"s1_loop_heal": {"status": "completed"}})
    card_gdn2 = render_model_card(m_gdn2)
    assert "  - gated-deltanet-2" in card_gdn2
    assert "NVIDIA Gated DeltaNet-2 (`GDN-2`" in card_gdn2

    # Also verify HubSyncManager.push_checkpoint records linear_attention="stock" when gdn2=None, hssd=None
    class _MemApi:
        def __init__(self):
            self.files: dict[tuple[str, str], bytes] = {}

        def create_repo(self, **_kw: Any) -> None:
            pass

        def commit_bundle(self, repo_id: str, additions: list[tuple[str, str | bytes]], _msg: str) -> str:
            for rp, src in additions:
                if isinstance(src, (bytes, bytearray)):
                    self.files[(repo_id, rp)] = bytes(src)
                else:
                    with open(src, "rb") as f:
                        self.files[(repo_id, rp)] = f.read()
            return "abc123456789"

        def download_file(self, repo_id: str, path_in_repo: str, dst_path: str) -> bool:
            if (repo_id, path_in_repo) not in self.files:
                return False
            os.makedirs(os.path.dirname(os.path.abspath(dst_path)), exist_ok=True)
            with open(dst_path, "wb") as f:
                f.write(self.files[(repo_id, path_in_repo)])
            return True

    api = _MemApi()
    hub = HubSyncManager(repo_id="test/hssd-dlsa", api=api)
    ck_dir = tmp_path / "ckpt"
    ck_dir.mkdir()
    torch.save({"loop_adapters.dummy": torch.zeros(2)}, ck_dir / "trainable.pt")
    (ck_dir / "state.json").write_text(
        '{"step": 10, "stage": "s1_loop_a6b", "stack": {"loop": {"prelude_end": 12, "core_end": 36, "num_passes": 3}, '
        '"gdn2": null, "hssd": null, "lsa": null}}',
        encoding="utf-8",
    )
    hub.push_checkpoint("s1_loop_a6b", str(ck_dir), completed=True)
    m_pushed = hub.fetch_manifest()
    assert m_pushed.architecture["linear_attention"] == "stock"
    assert "Gated DeltaNet-2" not in api.files[("test/hssd-dlsa", "README.md")].decode("utf-8")
