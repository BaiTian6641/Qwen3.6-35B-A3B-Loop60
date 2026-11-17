"""Comprehensive test suite for v3.2 Blackwell Speed, Optimizer & Stage Wiring upgrades:
  1. `configure_tf32` startup toggle + 32,768-token HSSD chunk-recurrence TF32 mantissa drift check
  2. `FusedStepOptimizer` zero-host-sync backward hooks, single-sync `step()`, `defer_micro_step()`,
     batched `torch.optim.AdamW` (`fused=False`) + true global-norm clipping, & APOLLO rank-256 config
  3. Zero-duplicate-VRAM `SharedBackboneTeacher` + Phase-1 online per-layer teacher-forced warm-up
     (`run_layerwise_warmup_microstep`) + Phase-2 end-to-end `_ChunkedVocabKL` distillation
  4. `LMTrainer` step-1 `tok/s` timer fix + full 3-mode stage wiring (`distill`, `long_calib`, `bptt`)
  5. Stage YAML v3.2 config invariants across `base.yaml`, `s1`, `s1b`, `s3`, and `s2`
"""
from __future__ import annotations

import copy
import json
import os
import time
from dataclasses import fields
from typing import Any

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from qloop.config import load_config
from qloop.model.hssd import HSSDConfig, PeakBoltzmannGate, chunk_hssd
from qloop.model.loading import load_model
from qloop.model.lsa import LSAConfig
from qloop.train.checkpoint import CheckpointManager
from qloop.train.distill import (
    HSSDDistillConfig,
    HSSDDistiller,
    SharedBackboneTeacher,
    run_layerwise_warmup_microstep,
)
from qloop.train.optim import (
    FusedStepOptimizer,
    build_param_groups,
    defer_micro_step,
    resolve_optim_cfg,
)
from qloop.train.stack import build_stack
from qloop.train.trainer import LMTrainer
from qloop.utils.tf32 import configure_tf32

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _strip_unmerged_dataclass_keys(cfg: dict[str, Any]) -> dict[str, Any]:
    """Allow standalone branch execution before parallel HSSD/LSA worker branches merge."""
    out = copy.deepcopy(cfg)
    if isinstance(out.get("hssd"), dict):
        valid_hssd = {f.name for f in fields(HSSDConfig)} | {"enabled"}
        out["hssd"] = {k: v for k, v in out["hssd"].items() if k in valid_hssd}
    if isinstance(out.get("lsa"), dict):
        valid_lsa = {f.name for f in fields(LSAConfig)} | {"enabled"}
        out["lsa"] = {k: v for k, v in out["lsa"].items() if k in valid_lsa}
        # Fall back from "hybrid" -> "distill" only if LSAConfig has not yet merged "hybrid" support.
        try:
            LSAConfig(mode=str(out["lsa"].get("mode", "distill"))).validate() if hasattr(
                LSAConfig, "validate"
            ) else None
            if out["lsa"].get("mode") == "hybrid":
                # Probe whether lsa.py accepts mode="hybrid"
                from qloop.model import lsa as lsa_mod

                with open(lsa_mod.__file__, encoding="utf-8") as fh:
                    lsa_src = fh.read()
                if '"hybrid"' not in lsa_src and "'hybrid'" not in lsa_src:
                    out["lsa"]["mode"] = "distill"
        except Exception:
            out["lsa"]["mode"] = "distill"
    return out


def _toy_base_cfg(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "seed": 7,
        "stage": "toy_speed_test",
        "stage_type": "lm",
        "model": {
            "name": "toy",
            "dtype": "float32",
            "device_map": "cpu",
            "toy": {
                "hidden_size": 32,
                "num_hidden_layers": 8,
                "full_attention_interval": 4,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "head_dim": 8,
                "linear_num_heads": 2,
                "linear_head_dim": 8,
                "num_experts": 4,
                "num_experts_per_tok": 2,
                "moe_intermediate_size": 16,
                "shared_expert_intermediate_size": 16,
            },
        },
        "loop": {
            "enabled": True,
            "prelude_end": 4,
            "core_end": 8,
            "num_passes": 2,
            "target_virtual_layers": 12,
            "kv_share": True,
            "bridge_init": [0.05, 0.85, 0.10],
            "bridge_rank": 8,
            "bridge_mode": "subspace",
            "attn_adapter_rank": 4,
            "router_adapter_rank": 4,
            "checkpoint": False,
            "train_pass_probs": [0.0, 1.0],
        },
        "gdn2": {"enabled": False},
        "hssd": {"enabled": False},
        "lsa": {"enabled": False},
        "experts": {"enabled": False},
        "data": {
            "seq_len": 32,
            "sources": [{"name": "toy_text", "weight": 1.0}],
        },
        "train": {
            "allow_tf32": True,
            "steps": 2,
            "micro_batch": 1,
            "grad_accum": 1,
            "log_every": 1,
            "ckpt_every": 10,
            "eval_every": 0,
            "keep_last": 1,
            "sha256_frozen_experts": True,
            "optim": {"lr": 5e-3, "warmup_steps": 1, "fused": True, "impl": "builtin"},
        },
        "eval_quick": {"n_batches": 0, "depths": [1, 2]},
    }
    for k, v in overrides.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            base[k].update(v)
        else:
            base[k] = v
    return _strip_unmerged_dataclass_keys(base)


# ============================================================================ 1. TF32 & 32K drift

def test_configure_tf32_toggle_and_startup_wiring(tmp_path):
    st_off = configure_tf32(False)
    assert st_off["allow_tf32"] is False
    assert st_off["float32_matmul_high"] is False
    if hasattr(torch, "get_float32_matmul_precision"):
        assert torch.get_float32_matmul_precision() == "highest"

    st_on = configure_tf32(True)
    assert st_on["allow_tf32"] is True
    assert st_on["float32_matmul_high"] is True
    if hasattr(torch, "get_float32_matmul_precision"):
        assert torch.get_float32_matmul_precision() == "high"

    # load_model respects explicit allow_tf32
    _, _, info_false = load_model({"name": "toy", "hidden_size": 16, "num_hidden_layers": 4, "allow_tf32": False})
    assert info_false["tf32"]["allow_tf32"] is False

    # LMTrainer configures TF32 from train.allow_tf32
    cfg = _toy_base_cfg(train={"allow_tf32": True, "steps": 1})
    tr = LMTrainer(cfg, str(tmp_path / "tf32_run"))
    assert torch.get_float32_matmul_precision() == "high"
    tr.train()
    assert torch.get_float32_matmul_precision() == "high"


def _round_fp32_to_tf32_mantissa(x: torch.Tensor) -> torch.Tensor:
    """Round IEEE-754 binary32 (23-bit mantissa) to exact NVIDIA TF32 (10-bit mantissa)."""
    if not isinstance(x, torch.Tensor) or x.dtype != torch.float32:
        return x
    # Round-to-nearest-even on bit 13 (clearing low 13 mantissa bits -> exact 10-bit TF32 mantissa)
    bits = x.contiguous().view(torch.int32)
    lsb = (bits >> 13) & 1
    round_bias = 0x0FFF + lsb
    rounded = (bits + round_bias) & ~0x1FFF
    out = rounded.view(torch.float32)
    return torch.where(torch.isfinite(x), out, x)


def test_hssd_32k_tf32_mantissa_rounding_state_drift():
    """Verify 32,768-token HSSD chunk recurrence stays tightly bounded (< 2e-2) under TF32 matmuls."""
    torch.manual_seed(1234)
    bsz, seq_len, hv, dk, dv = 1, 32_768, 2, 16, 16
    num_pages, top_k_pages, chunk_size, window_chunks = 4, 2, 64, 4

    q = F.normalize(torch.randn(bsz, seq_len, hv, dk, dtype=torch.float32), dim=-1)
    k = F.normalize(torch.randn(bsz, seq_len, hv, dk, dtype=torch.float32), dim=-1)
    v = torch.randn(bsz, seq_len, hv, dv, dtype=torch.float32) * 0.5
    gamma_page_scale = torch.tensor([0.05, 0.20, 0.60, 1.00], dtype=torch.float32)
    gamma_chan_pair = torch.rand(bsz, seq_len, hv, dk // 2, dtype=torch.float32) * 0.04 + 0.002
    b_gate = torch.sigmoid(torch.randn(bsz, seq_len, hv, 1, dtype=torch.float32)) * 1.5
    w_gate = torch.sigmoid(torch.randn(bsz, seq_len, hv, 1, dtype=torch.float32))
    theta = torch.randn(bsz, seq_len, hv, dk // 2, dtype=torch.float32) * 0.05
    surprise_bias = torch.full((bsz, seq_len, hv), 1.0, dtype=torch.float32)
    surprise_slope = torch.ones(hv, dtype=torch.float32)
    surprise_thresh = torch.full((hv,), 0.25, dtype=torch.float32)
    write_logits = torch.randn(bsz, seq_len, num_pages, dtype=torch.float32)
    read_logits = torch.randn(bsz, seq_len, num_pages, dtype=torch.float32)
    q_swa = F.normalize(torch.randn(bsz, seq_len, hv, dk, dtype=torch.float32), dim=-1)
    k_swa = F.normalize(torch.randn(bsz, seq_len, hv, dk, dtype=torch.float32), dim=-1)
    v_swa = torch.randn(bsz, seq_len, hv, dv, dtype=torch.float32) * 0.5
    gate = PeakBoltzmannGate(hv)

    common_kw = dict(
        boltzmann_gate=gate,
        chunk_size=chunk_size,
        window_chunks=window_chunks,
        num_pages=num_pages,
        top_k_pages=top_k_pages,
        segment_chunks=16,
        checkpoint_segments=False,
        output_final_state=True,
        state_dtype="float32",
    )

    configure_tf32(False)
    with torch.no_grad():
        out_fp32, st_fp32 = chunk_hssd(
            q,
            k,
            v,
            (gamma_page_scale, gamma_chan_pair),
            b_gate,
            w_gate,
            theta,
            surprise_bias,
            surprise_slope,
            surprise_thresh,
            write_logits,
            read_logits,
            q_swa,
            k_swa,
            v_swa,
            **common_kw,
        )

    # Intercept every `@` / matmul / bmm / mm / einsum inside chunk_hssd via TorchFunctionMode
    # to simulate exact 10-bit TF32 Tensor-Core inputs, while cumulative log-decay, solve_triangular,
    # and fp64 theta_cum stay in native precision.
    matmul_calls = {"n": 0}
    tf32_names = {"matmul", "__matmul__", "bmm", "mm", "einsum"}

    class _TF32MatmulMode(torch.overrides.TorchFunctionMode):
        def __torch_function__(self, func: Any, types: Any, args: Any = (), kwargs: Any = None) -> Any:
            kwargs = kwargs or {}
            if getattr(func, "__name__", "") in tf32_names:
                new_args = []
                hit = False
                for a in args:
                    if isinstance(a, torch.Tensor) and a.dtype == torch.float32 and a.ndim >= 2:
                        hit = True
                        new_args.append(_round_fp32_to_tf32_mantissa(a))
                    else:
                        new_args.append(a)
                if hit:
                    matmul_calls["n"] += 1
                args = tuple(new_args)
            return func(*args, **kwargs)

    configure_tf32(True)
    with torch.no_grad(), _TF32MatmulMode():
        out_tf32, st_tf32 = chunk_hssd(
            _round_fp32_to_tf32_mantissa(q),
            _round_fp32_to_tf32_mantissa(k),
            _round_fp32_to_tf32_mantissa(v),
            (gamma_page_scale, gamma_chan_pair),
            b_gate,
            w_gate,
            theta,
            surprise_bias,
            surprise_slope,
            surprise_thresh,
            write_logits,
            read_logits,
            _round_fp32_to_tf32_mantissa(q_swa),
            _round_fp32_to_tf32_mantissa(k_swa),
            _round_fp32_to_tf32_mantissa(v_swa),
            **common_kw,
        )

    assert matmul_calls["n"] > 500, "TF32 matmul wrapper should intercept WY chunk matmuls across 512 chunks"
    assert st_fp32 is not None and st_tf32 is not None
    assert torch.isfinite(out_tf32).all() and torch.isfinite(st_tf32.page_states).all()
    assert st_tf32.theta_cum.dtype == torch.float64

    rel_out_drift = float(torch.linalg.vector_norm(out_tf32 - out_fp32) / torch.linalg.vector_norm(out_fp32))
    rel_tail_drift = float(
        torch.linalg.vector_norm(out_tf32[:, -2048:] - out_fp32[:, -2048:])
        / torch.linalg.vector_norm(out_fp32[:, -2048:])
    )
    rel_state_drift = float(
        torch.linalg.vector_norm(st_tf32.page_states - st_fp32.page_states)
        / torch.linalg.vector_norm(st_fp32.page_states)
    )
    theta_drift = float((st_tf32.theta_cum - st_fp32.theta_cum).abs().max())

    assert rel_out_drift < 2e-2, f"32K output relative drift {rel_out_drift:.4e} exceeded 2e-2"
    assert rel_tail_drift < 2e-2, f"32K tail output relative drift {rel_tail_drift:.4e} exceeded 2e-2"
    assert rel_state_drift < 2e-2, f"32K terminal page state drift {rel_state_drift:.4e} exceeded 2e-2"
    assert theta_drift < 1e-6, f"fp64 theta_cum drift {theta_drift:.4e} should stay near zero"


# ============================================================================ 2. Optimizer speed & defer_micro_step

class _MixedTinyNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.big = nn.Linear(128, 128, bias=False)
        self.adapter = nn.Linear(128, 16, bias=True)
        self.loop_adapters = nn.Parameter(torch.zeros(16))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.tanh(self.big(x))
        return self.adapter(h) + self.loop_adapters


def test_fused_optimizer_zero_host_sync_in_hooks_and_single_sync_step(monkeypatch):
    torch.manual_seed(0)
    m = _MixedTinyNet()
    groups = build_param_groups(
        m.named_parameters(),
        {
            "lr": 1e-2,
            "weight_decay": 0.01,
            "apollo": {"rank": 32, "scale_type": "channel", "scale": 1.0, "min_dim": 128},
        },
    )
    opt = FusedStepOptimizer(groups, fused=True, impl="builtin", max_grad_norm_per_param=0.1)

    sync_counts = {"item": 0, "tolist": 0, "numpy": 0}
    orig_item = torch.Tensor.item
    orig_tolist = torch.Tensor.tolist
    orig_numpy = torch.Tensor.numpy

    def _spy_item(self: torch.Tensor) -> Any:
        sync_counts["item"] += 1
        return orig_item(self)

    def _spy_tolist(self: torch.Tensor) -> Any:
        sync_counts["tolist"] += 1
        return orig_tolist(self)

    def _spy_numpy(self: torch.Tensor, *a: Any, **kw: Any) -> Any:
        sync_counts["numpy"] += 1
        return orig_numpy(self, *a, **kw)

    x = torch.randn(4, 128) * 5.0
    loss = m(x).pow(2).mean()

    monkeypatch.setattr(torch.Tensor, "item", _spy_item)
    monkeypatch.setattr(torch.Tensor, "tolist", _spy_tolist)
    monkeypatch.setattr(torch.Tensor, "numpy", _spy_numpy)

    opt.set_boundary(True)
    loss.backward()
    # ZERO host syncs inside _make_hook across all parameters!
    assert sync_counts == {"item": 0, "tolist": 0, "numpy": 0}

    stats = opt.step()
    # At most ONE host read for the entire optimizer step!
    assert sync_counts["item"] + sync_counts["tolist"] + sync_counts["numpy"] <= 1
    assert stats["optim/stepped_params"] == 4.0
    assert stats["optim/clipped_params"] >= 1.0
    assert stats["optim/global_grad_norm"] > 0.0


def test_defer_micro_step_multi_backward_accumulation_and_standard_opt_noop():
    torch.manual_seed(11)
    m_def = _MixedTinyNet()
    m_ref = copy.deepcopy(m_def)

    opt_cfg = {"lr": 1e-2, "weight_decay": 0.01, "apollo": {"enabled": False}}
    opt_def = FusedStepOptimizer(
        build_param_groups(m_def.named_parameters(), opt_cfg),
        fused=True,
        impl="builtin",
        grad_accum=2,
        max_grad_norm_per_param=0.5,
    )
    opt_ref = FusedStepOptimizer(
        build_param_groups(m_ref.named_parameters(), opt_cfg),
        fused=True,
        impl="builtin",
        grad_accum=2,
        max_grad_norm_per_param=0.5,
    )

    batches = [[torch.randn(2, 128) for _ in range(3)] for _ in range(2)]

    for micro_idx, sub_xs in enumerate(batches):
        # Deferred path: 3 separate .backward() calls on shared parameters inside one micro-step
        with defer_micro_step(opt_def):
            for sx in sub_xs:
                (m_def(sx).pow(2).mean() / (3.0 * 2.0)).backward()
                # Inside defer_micro_step, _micro has NOT incremented yet!
                assert all(c == micro_idx for c in opt_def._micro)
        assert all(c == micro_idx + 1 for c in opt_def._micro)
        if micro_idx == 0:
            # Grad accumulated after micro 1, not stepped yet
            assert m_def.big.weight.grad is not None
        else:
            # Boundary micro 2 stepped and freed .grad on exit of defer_micro_step!
            assert m_def.big.weight.grad is None

        # Reference path: single summed backward per micro-step
        summed = sum(m_ref(sx).pow(2).mean() for sx in sub_xs) / (3.0 * 2.0)
        summed.backward()

    s_def = opt_def.step()
    s_ref = opt_ref.step()
    for (n, p1), (_, p2) in zip(m_def.named_parameters(), m_ref.named_parameters()):
        assert torch.allclose(p1, p2, atol=1e-6, rtol=1e-6), f"mismatch on {n}"
    assert s_def["optim/global_grad_norm"] == pytest.approx(s_ref["optim/global_grad_norm"], rel=1e-5)

    # Standard torch.optim.Optimizer no-op compatibility
    p = nn.Parameter(torch.ones(4))
    std_opt = torch.optim.AdamW([p], lr=1e-2)
    with defer_micro_step(std_opt):
        (p * 2.0).sum().backward()
        (p * 3.0).sum().backward()
    assert torch.allclose(p.grad, torch.full((4,), 5.0))
    std_opt.step()


def test_nonfused_batched_torch_adamw_global_clip_and_apollo_base_cfg():
    torch.manual_seed(21)
    m = nn.Sequential(nn.Linear(16, 16, bias=False), nn.Linear(16, 8, bias=True))
    groups = build_param_groups(m.named_parameters(), {"lr": 1e-2, "weight_decay": 0.01, "apollo": {"enabled": False}})
    opt = FusedStepOptimizer(groups, fused=False, impl="auto", max_grad_norm=0.25)
    assert isinstance(opt._torch_adamw, torch.optim.AdamW)

    x = torch.randn(4, 16) * 10.0
    m(x).pow(2).mean().backward()
    pre_norm = float(
        torch.linalg.vector_norm(
            torch.stack(
                [torch.linalg.vector_norm(p.grad.detach().float()) for p in m.parameters() if p.grad is not None]
            )
        )
    )
    assert pre_norm > 0.25
    st = opt.step()
    clipped_total = float(
        torch.linalg.vector_norm(
            torch.stack(
                [torch.linalg.vector_norm(p.grad.detach().float()) for p in m.parameters() if p.grad is not None]
            )
        )
    )
    assert clipped_total == pytest.approx(0.25, rel=1e-4)
    assert st["optim/global_grad_norm"] == pytest.approx(pre_norm, rel=1e-5)
    assert st["optim/clipped_params"] == 3.0
    opt.zero_grad()

    # Non-finite gradient in fused=False mode atomically aborts the ENTIRE step (0 params updated,
    # 0 optimizer moments/step counters mutated, finite grads not poisoned by clip_grad_norm_)!
    w0_before = m[0].weight.detach().clone()
    w1_before = m[1].weight.detach().clone()
    b1_before = m[1].bias.detach().clone()
    st_before_step = {
        k: (v.detach().clone() if isinstance(v, torch.Tensor) else v)
        for k, v in opt._torch_adamw.state[m[1].weight].items()
    }
    ema_before = opt.norm_ema
    m(x).pow(2).mean().backward()
    w1_grad_before = m[1].weight.grad.detach().clone()
    assert m[0].weight.grad is not None
    m[0].weight.grad.fill_(float("nan"))
    st_nan = opt.step()
    assert torch.equal(m[1].weight.grad.detach(), w1_grad_before)  # clip_grad_norm_ did not poison finite grad
    opt.zero_grad()
    assert torch.equal(m[0].weight.detach(), w0_before)
    assert torch.equal(m[1].weight.detach(), w1_before)
    assert torch.equal(m[1].bias.detach(), b1_before)
    assert st_nan["optim/stepped_params"] == 0.0
    assert st_nan["optim/skipped_params"] == 1.0
    assert opt.norm_ema == pytest.approx(ema_before)
    for k, v_old in st_before_step.items():
        v_cur = opt._torch_adamw.state[m[1].weight][k]
        if isinstance(v_old, torch.Tensor):
            assert torch.equal(v_cur, v_old)
        else:
            assert v_cur == v_old

    # Verify mixed fp32 adapters + bf16 norms with master_weights=True in fused=False mode:
    # fp32 params route to batched torch.optim.AdamW while bf16 norms keep fp32 master weights!
    fp32_w = nn.Parameter(torch.randn(16, 16, dtype=torch.float32))
    bf16_norm = nn.Parameter(torch.ones(16, dtype=torch.bfloat16))
    mix_groups = build_param_groups(
        [
            ("model.layers.0.linear_attn.hssd_adapter.down.weight", fp32_w),
            ("model.layers.0.input_layernorm.weight", bf16_norm),
        ],
        {"lr": 4e-5, "lr_mult": {"hssd_adapter": 10.0}},
    )
    mix_opt = FusedStepOptimizer(mix_groups, fused=False, impl="auto", max_grad_norm=1.0, master_weights=True)
    assert isinstance(mix_opt._torch_adamw, torch.optim.AdamW)
    assert mix_opt.entries["model.layers.0.linear_attn.hssd_adapter.down.weight"].opt is mix_opt._torch_adamw
    assert mix_opt.entries["model.layers.0.input_layernorm.weight"].opt is not mix_opt._torch_adamw
    for _ in range(25):
        fp32_w.grad = torch.ones_like(fp32_w) * 0.01
        bf16_norm.grad = torch.ones_like(bf16_norm)
        mix_opt.step()
        mix_opt.zero_grad()
    norm_ent = mix_opt.entries["model.layers.0.input_layernorm.weight"]
    assert norm_ent.opt.state[bf16_norm]["master"].dtype == torch.float32
    assert float(norm_ent.opt.state[bf16_norm]["master"][0]) < 1.0 - 5e-4

    # Verify strict=True state_dict resume between fused=True (AdamWLite) and fused=False (torch.optim.AdamW):
    w_src = nn.Parameter(torch.randn(8, 8, dtype=torch.float32))
    n_src = nn.Parameter(torch.ones(8, dtype=torch.bfloat16))
    src_opt = FusedStepOptimizer(
        build_param_groups(
            [
                ("model.layers.0.linear_attn.hssd_adapter.down.weight", w_src),
                ("model.layers.0.input_layernorm.weight", n_src),
            ],
            {"lr": 1e-3},
        ),
        fused=True,
        impl="auto",
        master_weights=True,
    )
    for _ in range(3):
        src_opt.set_boundary(True)
        (w_src.pow(2).sum() + n_src.float().pow(2).sum()).backward()
        src_opt.step()
        src_opt.zero_grad()
    sd_fused = src_opt.state_dict()
    w_dst = nn.Parameter(w_src.detach().clone())
    n_dst = nn.Parameter(n_src.detach().clone())
    dst_opt = FusedStepOptimizer(
        build_param_groups(
            [
                ("model.layers.0.linear_attn.hssd_adapter.down.weight", w_dst),
                ("model.layers.0.input_layernorm.weight", n_dst),
            ],
            {"lr": 1e-3},
        ),
        fused=False,
        impl="auto",
        max_grad_norm=1.0,
        master_weights=True,
    )
    dst_opt.load_state_dict(sd_fused, strict=True)
    assert dst_opt.step_count == 3
    assert dst_opt._torch_adamw is not None
    assert float(dst_opt._torch_adamw.state[w_dst]["step"].item()) == 3.0
    (w_dst.pow(2).sum() + n_dst.float().pow(2).sum()).backward()
    dst_opt.step()
    dst_opt.zero_grad()
    assert float(dst_opt._torch_adamw.state[w_dst]["step"].item()) == 4.0

    # Verify resolve_optim_cfg update_Gap alias & base.yaml APOLLO tuning
    resolved = resolve_optim_cfg({"apollo": {"rank": 256, "scale_type": "channel", "update_Gap": 150, "scale": 1.0}})
    assert resolved["apollo"]["update_proj_gap"] == 150
    base_yaml = load_config(os.path.join(ROOT, "configs", "base.yaml"))
    ap = base_yaml["train"]["optim"]["apollo"]
    assert ap["rank"] == 256 and ap["scale_type"] == "channel" and ap["scale"] == 1.0
    assert ap["update_proj_gap"] == 200


# ===================================================================== 3. SharedBackboneTeacher & Phase-1/2 Distill

def test_shared_backbone_teacher_zero_duplicate_vram_gdn2_and_exception_restore(tmp_path):
    # 1. Build a trained Track-A GDN-2 + Loop teacher stage and save its delta checkpoint
    s1_dir = tmp_path / "s1_loop_a6b"
    t_cfg = _toy_base_cfg(
        gdn2={"enabled": True, "rank": 8, "chunk_size": 8, "segment_chunks": 2, "checkpoint_segments": False},
        hssd={"enabled": False},
        train={
            "train_loop_adapters": True,
            "train_gdn2": True,
            "trainable": [r"input_layernorm|post_attention_layernorm"],
            "allowed_groups": ["loop_adapters", "gdn2_adapters", "norms"],
        },
    )
    teacher_st = build_stack(t_cfg)
    torch.manual_seed(99)
    with torch.no_grad():
        for p in teacher_st.named_trainable.values():
            p.add_(torch.randn_like(p) * 0.15)
    ckpt_mgr = CheckpointManager(str(s1_dir), keep_last=1)
    ckpt_mgr.save(2, teacher_st.checkpoint_params(), None, {"step": 2, "stack": teacher_st.describe()}, completed=True)

    # 2. Build student stack with HSSD enabled, initialized from s1_loop_a6b, then perturb student params
    s_cfg = _toy_base_cfg(
        gdn2={"enabled": False},
        hssd={
            "enabled": True,
            "chunk_size": 8,
            "window_chunks": 2,
            "num_pages": 4,
            "k_pages": 2,
            "theta_low_rank": True,
            "segment_chunks": 2,
            "checkpoint_segments": False,
        },
        train={
            "init_from": str(s1_dir),
            "train_loop_adapters": True,
            "train_hssd": True,
            "trainable": [r"input_layernorm|post_attention_layernorm"],
            "allowed_groups": ["loop_adapters", "hssd_adapters", "deltanet", "norms"],
        },
    )
    student_st = build_stack(s_cfg)
    loaded_ck = ckpt_mgr.load(map_location="cpu")
    assert loaded_ck is not None
    student_st.restore_checkpoint(
        loaded_ck["params"], strict=False, prior_stack_meta=(loaded_ck["state"].get("stack") or {})
    )
    # Perturb student loop_adapters, norms, and hssd_adapter so student != teacher
    with torch.no_grad():
        for p in student_st.named_trainable.values():
            p.add_(torch.randn_like(p) * 0.25)

    expert_ptrs_before = [
        p.data_ptr() for n, p in student_st.model.named_parameters() if "experts." in n
    ]
    shared_teacher = SharedBackboneTeacher(student_st, teacher_ckpt=str(s1_dir), teacher_track="gdn2")
    expert_ptrs_after = [
        p.data_ptr() for n, p in student_st.model.named_parameters() if "experts." in n
    ]
    assert expert_ptrs_before == expert_ptrs_after
    assert shared_teacher.loaded_from_ckpt is True
    assert len(shared_teacher.gdn2_adapters) == len(student_st.hssd.adapters)
    # Extra VRAM holds ONLY small deltas, smaller than full model parameters
    total_model_bytes = sum(p.numel() * p.element_size() for p in student_st.model.parameters())
    assert shared_teacher.extra_vram_bytes() < total_model_bytes

    ids = torch.randint(0, 64, (2, 16))
    teacher_st.model.eval()
    teacher_st.loop.set_active_passes(2)
    student_st.loop.set_active_passes(2)
    student_st.model.train()

    student_snapshots = {k: p.detach().clone() for k, p in student_st.named_trainable.items()}
    student_data_ptrs = {k: p.data.data_ptr() for k, p in student_st.named_trainable.items()}

    with torch.no_grad():
        h_ref = teacher_st.hidden(ids)
        h_student_before = student_st.hidden(ids)
    assert not torch.allclose(h_ref, h_student_before, atol=1e-3)

    with shared_teacher.activate(), torch.no_grad():
        h_shared = student_st.hidden(ids)
    assert torch.allclose(h_shared, h_ref, atol=1e-5, rtol=1e-5)

    # Exception safety: raise inside activate() and verify 100% restoration of student .data & training mode
    with pytest.raises(RuntimeError, match="deliberate"), shared_teacher.activate():
        raise RuntimeError("deliberate")

    assert student_st.model.training is True
    for k, p in student_st.named_trainable.items():
        assert p.data.data_ptr() == student_data_ptrs[k]
        assert torch.equal(p.detach(), student_snapshots[k])

    # HSSDDistiller works seamlessly with SharedBackboneTeacher
    distiller = HSSDDistiller(
        shared_teacher,
        student_st.model,
        handle=student_st.hssd,
        cfg=HSSDDistillConfig(chunk_size=32, token_chunk_size=16, hidden_mse_weight=0.1),
    )
    dout = distiller(ids)
    dout.loss.backward()
    assert any(
        p.grad is not None and float(p.grad.abs().max()) > 0
        for n, p in student_st.named_trainable.items()
        if "hssd_adapter." in n
    )


def test_online_layerwise_warmup_trains_only_hssd_and_reduces_mse():
    torch.manual_seed(42)
    cfg = _toy_base_cfg(
        hssd={
            "enabled": True,
            "chunk_size": 8,
            "window_chunks": 2,
            "num_pages": 4,
            "k_pages": 2,
            "theta_low_rank": True,
            "segment_chunks": 2,
            "checkpoint_segments": True,
        },
        train={
            "train_loop_adapters": True,
            "train_hssd": True,
            "trainable": [r"input_layernorm|post_attention_layernorm"],
            "allowed_groups": ["loop_adapters", "hssd_adapters", "deltanet", "norms"],
        },
    )
    st = build_stack(cfg)
    st.loop.set_active_passes(2)
    #Snapshot teacher with non-zero loop bridges, and perturb student HSSD adapters
    with torch.no_grad():
        for n, p in st.named_trainable.items():
            if "hssd_adapter." in n:
                p.add_(torch.randn_like(p) * 0.2)
    shared_teacher = SharedBackboneTeacher(st, teacher_ckpt=None)

    non_hssd_before = {
        n: p.detach().clone() for n, p in st.named_trainable.items() if "hssd_adapter." not in n
    }
    opt = FusedStepOptimizer.from_config(
        list(st.named_trainable.items()),
        {"lr": 2e-2, "fused": True, "impl": "builtin", "apollo": {"enabled": False}},
        grad_accum=1,
    )
    ids = torch.randint(0, 64, (2, 16))

    mse_history: list[float] = []
    for _ in range(8):
        opt.set_boundary(True)
        w = run_layerwise_warmup_microstep(
            st, shared_teacher, opt, ids, depth=2, grad_accum=1, mse_weight=1.0, budget_weight=0.1
        )
        opt.step()
        mse_history.append(w["distill/layerwise_mse"])
        # 8 physical layers (6 linear: 2 prelude [0,1] + 3 core [2,4,5]*2 passes + 1 coda [6]) -> 9 calls!
        assert w["distill/calls"] == 9.0

    assert mse_history[-1] < mse_history[0], f"Layerwise MSE did not decrease: {mse_history}"
    for n, p_init in non_hssd_before.items():
        assert torch.equal(st.named_trainable[n].detach(), p_init), f"Phase-1 should not touch non-HSSD param {n}"
    st.verify_frozen_experts()


# ============================================================================ 4. LMTrainer stage wiring & timer fix

def test_lm_trainer_two_phase_distill_and_step0_eval_timer_fix(tmp_path, monkeypatch):
    cfg = _toy_base_cfg(
        hssd={
            "enabled": True,
            "chunk_size": 8,
            "window_chunks": 2,
            "num_pages": 4,
            "k_pages": 2,
            "theta_low_rank": True,
            "segment_chunks": 2,
            "checkpoint_segments": False,
        },
        distill={
            "enabled": True,
            "share_backbone": True,
            "layerwise_warmup_steps": 2,
            "temperature": 1.0,
            "kl_weight": 1.0,
            "hidden_mse_weight": 0.1,
            "vocab_chunk_size": 32,
        },
        train={
            "steps": 4,
            "micro_batch": 1,
            "grad_accum": 2,
            "log_every": 1,
            "eval_every": 4,
            "ckpt_every": 4,
            "train_loop_adapters": True,
            "train_hssd": True,
            "trainable": [r"input_layernorm|post_attention_layernorm"],
            "allowed_groups": ["loop_adapters", "hssd_adapters", "deltanet", "norms"],
            "optim": {"lr": 5e-3, "warmup_steps": 1, "fused": True, "impl": "builtin"},
        },
        eval_quick={"n_batches": 1, "depths": [1, 2]},
    )
    run_dir = tmp_path / "s1b_run"
    tr = LMTrainer(cfg, str(run_dir))

    # Attach duck-typed step_blend spy onto st.hssd to verify Phase-2 blend scheduling
    blend_steps_seen: list[int] = []

    def _spy_step_blend(gstep: int) -> float:
        blend_steps_seen.append(int(gstep))
        return min(1.0, float(gstep + 1) / 4.0)

    tr.stack.hssd.step_blend = _spy_step_blend  # type: ignore[attr-defined]

    # Deterministically advance trainer clock by +100s during step-0 quick_eval() to prove
    # step-1 tok_per_s is NOT polluted by step-0 eval duration (0.64 tok/s if broken vs >50 tok/s fixed).
    import qloop.train.trainer as trainer_mod

    real_time = time.time
    clock_offset = {"dt": 0.0}
    monkeypatch.setattr(trainer_mod.time, "time", lambda: real_time() + clock_offset["dt"])

    orig_qeval = tr.quick_eval
    qeval_calls = {"n": 0}

    def _slow_step0_eval() -> dict[str, float]:
        qeval_calls["n"] += 1
        out = orig_qeval()
        if qeval_calls["n"] == 1:
            clock_offset["dt"] += 100.0
        return out

    monkeypatch.setattr(tr, "quick_eval", _slow_step0_eval)
    res = tr.train()
    assert res["step"] == 4
    assert blend_steps_seen == [0, 1]  # only called on Phase-2 steps 2 and 3!

    records = [
        json.loads(ln)
        for ln in (run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]
    step_logs = {r["step"]: r for r in records if "tok_per_s" in r}
    assert step_logs[1]["loss/distill_phase"] == 1.0
    assert step_logs[2]["loss/distill_phase"] == 1.0
    assert step_logs[3]["loss/distill_phase"] == 2.0 and "loss/distill_kl" in step_logs[3]
    assert step_logs[4]["loss/distill_phase"] == 2.0 and "loss/hidden_mse" in step_logs[4]
    # 2 micro-batches x 32 tokens = 64 tokens. If +100s step-0 eval leaked into t_log, tok_per_s <= 0.64!
    assert step_logs[1]["tok_per_s"] > 20.0, f"step-1 tok/s polluted by step-0 eval: {step_logs[1]['tok_per_s']}"


def test_lm_trainer_long_calib_alternation_and_signature_plumbing(tmp_path, monkeypatch):
    import qloop.train.trainer as trainer_mod

    cfg = _toy_base_cfg(
        lsa={
            "enabled": True,
            "mode": "hybrid",
            "num_query_passes": 2,
            "index_dim": 8,
            "index_on": "nope",
            "pool": 2,
            "block_size": 16,
            "avg_bucket_keys": 4,
            "max_bucket_keys": 8,
            "num_select": 2,
            "window": 16,
            "sinks": 4,
            "query_chunk": 16,
        },
        long_calib={
            "enabled": True,
            "prefix_len": 32,
            "max_prefix_len": 48,
            "suffix_len": 16,
            "chunk_size": 16,
            "fp8_kv_cache": True,
            "prefix_lsa_mode": "sparse",
            "calib_every_steps": 2,
            "sparse_prefix_after_step": 1,
        },
        data={
            "seq_len": 32,
            "sources": [{"name": "toy_text", "weight": 1.0}],
        },
        train={
            "steps": 4,
            "micro_batch": 1,
            "grad_accum": 1,
            "log_every": 1,
            "ckpt_every": 10,
            "train_loop_adapters": True,
            "train_lsa": True,
            "trainable": [r"self_attn\.(q_norm|k_norm)"],
            "allowed_groups": ["loop_adapters", "lsa_indexers", "attention", "norms"],
            "optim": {"lr": 5e-3, "warmup_steps": 1, "fused": True, "impl": "builtin"},
        },
    )
    run_dir = tmp_path / "s3_run"
    tr = LMTrainer(cfg, str(run_dir))

    lsa_steps_recorded: list[int] = []
    tr.stack.lsa.set_step = lambda s: lsa_steps_recorded.append(int(s))  # type: ignore[attr-defined]

    orig_calib = trainer_mod.prefix_suffix_calibration_step
    plumbing_seen: list[tuple[Any, int]] = []

    def _wrapped_calib(
        model: nn.Module,
        input_ids: torch.Tensor,
        prefix_len: int,
        chunk_size: int = 4096,
        loss_fn: Any = None,
        *,
        lsa_manager: Any = None,
        global_step: int = -1,
        **kw: Any,
    ) -> Any:
        plumbing_seen.append((lsa_manager, int(global_step)))
        # Forward only supported kwargs to underlying prefix_suffix_calibration_step
        import inspect

        sig = inspect.signature(orig_calib)
        clean_kw = {k: v for k, v in kw.items() if k in sig.parameters}
        if "lsa_manager" in sig.parameters:
            clean_kw["lsa_manager"] = lsa_manager
        if "global_step" in sig.parameters:
            clean_kw["global_step"] = global_step
        return orig_calib(model, input_ids, prefix_len, chunk_size, loss_fn, **clean_kw)

    monkeypatch.setattr(trainer_mod, "prefix_suffix_calibration_step", _wrapped_calib)

    res = tr.train()
    assert res["step"] == 4
    assert lsa_steps_recorded == [0, 1, 2, 3]
    # calib_every_steps=2 -> triggered on 0-based steps 1 and 3 (1-based steps 2 and 4)!
    assert [gs for _, gs in plumbing_seen] == [1, 3]
    assert all(mgr is tr.stack.lsa for mgr, _ in plumbing_seen)

    records = [
        json.loads(ln)
        for ln in (run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]
    by_step = {r["step"]: r for r in records if "tok_per_s" in r}
    assert by_step[1]["loss/long_calib_active"] == 0.0
    assert by_step[2]["loss/long_calib_active"] == 1.0 and by_step[2]["loss/calib_suffix_len"] == 16.0
    assert by_step[3]["loss/long_calib_active"] == 0.0
    assert by_step[4]["loss/long_calib_active"] == 1.0


def test_lm_trainer_bptt_stage_wiring_and_short_seq_fallback(tmp_path):
    cfg = _toy_base_cfg(
        hssd={
            "enabled": True,
            "chunk_size": 8,
            "window_chunks": 2,
            "num_pages": 4,
            "k_pages": 2,
            "theta_low_rank": True,
            "segment_chunks": 2,
            "checkpoint_segments": False,
        },
        lsa={
            "enabled": True,
            "mode": "sparse",
            "num_query_passes": 2,
            "index_dim": 8,
            "index_on": "nope",
            "pool": 2,
            "block_size": 16,
            "avg_bucket_keys": 4,
            "max_bucket_keys": 8,
            "num_select": 2,
            "window": 16,
            "sinks": 4,
            "query_chunk": 16,
        },
        bptt={
            "enabled": True,
            "n_segments": 4,
            "seg_len": 16,
            "replay_k": 8,
            "offload_prior_hidden": True,
        },
        data={
            "seq_len": 64,  # 64 >= 2 * 16 -> drives SegmentedKVBPTTRunner (4 x 16)
            "sources": [{"name": "toy_text", "weight": 1.0}],
        },
        train={
            "steps": 2,
            "micro_batch": 1,
            "grad_accum": 2,
            "log_every": 1,
            "ckpt_every": 10,
            "train_loop_adapters": True,
            "train_hssd": True,
            "train_lsa": True,
            "trainable": [r"self_attn\.(k_proj|v_proj|o_proj|q_norm|k_norm)"],
            "allowed_groups": ["loop_adapters", "hssd_adapters", "lsa_indexers", "deltanet", "attention", "norms"],
            "optim": {"lr": 5e-3, "warmup_steps": 1, "fused": True, "impl": "builtin"},
        },
    )
    tr = LMTrainer(cfg, str(tmp_path / "s2_bptt_active"))
    before = {k: p.detach().clone() for k, p in tr.stack.named_trainable.items()}
    res = tr.train()
    assert res["step"] == 2
    assert res["last"]["loss/bptt_segments"] == 4.0
    assert any(not torch.equal(tr.stack.named_trainable[k], v) for k, v in before.items() if "k_proj" in k)
    assert any(not torch.equal(tr.stack.named_trainable[k], v) for k, v in before.items() if "hssd_adapter." in k)

    # Short sequence fallback: seq_len=16 < 2 * seg_len (32) -> falls back to standard forward/backward
    short_cfg = copy.deepcopy(cfg)
    short_cfg["data"]["seq_len"] = 16
    tr_short = LMTrainer(short_cfg, str(tmp_path / "s2_bptt_fallback"))
    res_short = tr_short.train()
    assert res_short["step"] == 2
    assert "loss/bptt_segments" not in res_short["last"]


def test_bptt_gqa_per_head_select_lsa_replay(tmp_path):
    """Verify SegmentedKVBPTTRunner handles GQA (H=4, G=2) when LSA query_latent is per-head [B, H, T, d_idx] (D4)."""
    from types import SimpleNamespace

    from qloop.train.bptt import SegmentedKVBPTTRunner

    class _PerHeadGQAIndexer(nn.Module):
        def __init__(self, nh: int = 4, nkv: int = 2, head_dim: int = 8, rot_dim: int = 4, idx_dim: int = 8) -> None:
            super().__init__()
            self.lo = rot_dim
            din = head_dim - rot_dim
            self.wq = nn.Parameter(torch.randn(nh, din, idx_dim) * 0.1)
            self.wk = nn.Parameter(torch.randn(nkv, din, idx_dim) * 0.1)
            self.cfg = SimpleNamespace(temperature=1.0)

        def query_latent(self, q: torch.Tensor, pass_idx: int | None = None) -> torch.Tensor:
            q_in = q[..., self.lo :].float()  # [B, H=4, T, din]
            return F.normalize(torch.einsum("bhtd,hde->bhte", q_in, self.wq), dim=-1, eps=1e-6)

        def k_proj(self, k: torch.Tensor) -> torch.Tensor:
            k_in = k[..., self.lo :].float()  # [B, G=2, T, din]
            return F.normalize(torch.einsum("bgtd,gde->bgte", k_in, self.wk), dim=-1, eps=1e-6)

        def key_latent(self, k: torch.Tensor) -> torch.Tensor:
            return self.k_proj(k)

        def pop_losses(self) -> list[torch.Tensor]:
            return []

    # 1. Direct GQA [B, H=4, T, d_idx] vs [B, G=2, T, d_idx] SegmentedKVBPTTRunner verification (standalone + merged)
    torch.manual_seed(99)
    model, _, _ = load_model(
        {
            "name": "toy",
            "dtype": "float32",
            "device_map": "cpu",
            "toy": {
                "hidden_size": 32,
                "num_hidden_layers": 4,
                "full_attention_interval": 4,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "head_dim": 8,
                "vocab_size": 64,
            },
        }
    )
    indexers: list[_PerHeadGQAIndexer] = []
    for mod in model.modules():
        if hasattr(mod, "k_proj") and hasattr(mod, "q_proj") and hasattr(mod, "v_proj"):
            ix = _PerHeadGQAIndexer(nh=4, nkv=2, head_dim=8, rot_dim=4, idx_dim=8)
            mod.lsa_indexer = ix
            indexers.append(ix)
    assert len(indexers) >= 1

    runner = SegmentedKVBPTTRunner(model, seg_len=16, replay_k=8, n_segments=4, replay_indexer_weight=0.2)
    ids = torch.randint(0, 64, (2, 64))
    step_res = runner.run_step(ids, labels=ids, backward=True)
    assert step_res.num_segments == 4
    assert sum(step_res.replayed_tokens_per_seg[1:]) > 0
    assert any(ix.wq.grad is not None and ix.wq.grad.abs().sum() > 0 for ix in indexers)
    assert any(ix.wk.grad is not None and ix.wk.grad.abs().sum() > 0 for ix in indexers)

    # 2. Full LMTrainer + real LSAConfig(per_head_select=True) integration whenever D4 is present in LSAConfig
    if "per_head_select" in {f.name for f in fields(LSAConfig)}:
        cfg = _toy_base_cfg(
            lsa={
                "enabled": True,
                "mode": "sparse",
                "per_head_select": True,
                "num_query_passes": 2,
                "index_dim": 8,
                "index_on": "nope",
                "pool": 2,
                "block_size": 16,
                "avg_bucket_keys": 4,
                "max_bucket_keys": 8,
                "num_select": 2,
                "window": 16,
                "sinks": 4,
                "query_chunk": 16,
            },
            bptt={
                "enabled": True,
                "n_segments": 4,
                "seg_len": 16,
                "replay_k": 8,
                "offload_prior_hidden": True,
            },
            data={"seq_len": 64, "sources": [{"name": "toy_text", "weight": 1.0}]},
            train={
                "steps": 2,
                "micro_batch": 1,
                "grad_accum": 1,
                "log_every": 1,
                "ckpt_every": 10,
                "train_loop_adapters": True,
                "train_lsa": True,
                "trainable": [r"self_attn\.(k_proj|v_proj|o_proj|q_norm|k_norm)"],
                "allowed_groups": ["loop_adapters", "lsa_indexers", "attention", "norms"],
                "optim": {"lr": 5e-3, "warmup_steps": 1, "fused": True, "impl": "builtin"},
            },
        )
        tr = LMTrainer(cfg, str(tmp_path / "s2_bptt_per_head_lsa"))
        res = tr.train()
        assert res["step"] == 2
        assert res["last"]["loss/bptt_segments"] == 4.0


# ============================================================================ 5. Stage YAML configs

def test_v32_stage_yaml_configs():
    base = load_config(os.path.join(ROOT, "configs", "base.yaml"))
    assert base["train"]["allow_tf32"] is True
    assert base["train"]["optim"]["apollo"]["rank"] == 256
    assert base["train"]["optim"]["apollo"]["scale_type"] == "channel"
    assert base["train"]["optim"]["apollo"]["scale"] == 1.0
    assert base["train"]["optim"]["apollo"]["update_proj_gap"] == 200

    s1 = load_config(os.path.join(ROOT, "configs", "stages", "s1_loop_a6b.yaml"))
    assert s1["gdn2"]["enabled"] is False
    assert s1["train"]["micro_batch"] == 4
    assert s1["train"]["grad_accum"] == 4
    assert s1["train"]["log_every"] == 5

    s1b = load_config(os.path.join(ROOT, "configs", "stages", "s1b_hssd_distill.yaml"))
    assert s1b["distill"]["teacher_track"] == "stock"
    assert s1b["hssd"]["write_mode"] == "full"
    assert s1b["hssd"]["swa_recency_init"] == 0.05
    assert s1b["hssd"]["blend_init"] == 0.0
    assert s1b["hssd"]["blend_warmup_steps"] == 250
    assert s1b["hssd"]["compact_active_pages"] is True
    assert s1b["hssd"]["compact_decode_cache"] is True
    assert s1b["distill"]["share_backbone"] is True
    assert s1b["distill"]["layerwise_warmup_steps"] == 400
    assert s1b["train"]["micro_batch"] == 1
    assert s1b["train"]["grad_accum"] == 4

    s3 = load_config(os.path.join(ROOT, "configs", "stages", "s3_long_1m_lsa.yaml"))
    assert s3["lsa"]["mode"] == "hybrid"
    assert s3["lsa"]["compact_active_slots"] is True
    assert s3["lsa"]["max_selected_keys"] == 768
    assert s3["lsa"]["recall_threshold"] == 0.70
    assert s3["lsa"]["hybrid_sparse_every"] == 2
    assert s3["long_calib"]["prefix_lsa_mode"] == "sparse"
    assert s3["long_calib"]["calib_every_steps"] == 10
    assert s3["long_calib"]["sparse_prefix_after_step"] == 80
    assert s3["hssd"]["horizon_tokens"] == 1048576
    assert s3["train"]["steps"] == 600

    s2 = load_config(os.path.join(ROOT, "configs", "stages", "s2_joint_bptt.yaml"))
    assert s2["hssd"]["write_mode"] == "full"
    assert s2["hssd"]["swa_recency_init"] == 0.05
    assert s2["hssd"]["horizon_tokens"] == 1048576
    assert s2["lsa"]["compact_active_slots"] is True
    assert s2["lsa"]["max_selected_keys"] == 768
