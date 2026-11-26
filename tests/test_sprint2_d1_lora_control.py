"""Sprint 2.2 Parameter-Matched d=1 (40L) LoRA Control (`A2` gate) test suite."""
from __future__ import annotations

import json
import os
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from qloop.config import load_config
from qloop.model.d1_lora import (
    QWEN36_NORM_PARAMS,
    S1_LOOP_ADAPTER_PARAMS,
    S1_LOOP_TOTAL_TRAINABLE_PARAMS,
    D1LoRAConfig,
    _LoRAResidualProj,
    compute_qwen36_d1_lora_params,
    install_d1_lora,
    solve_rank_for_target_params,
)
from qloop.model.loop import sha256_frozen_experts
from qloop.model.quant_autograd import dequantize_nvfp4_tensor, quantize_nvfp4_tensor
from qloop.model.toy import build_toy
from qloop.train.checkpoint import CheckpointManager
from qloop.train.optim import build_param_groups
from qloop.train.stack import build_stack
from qloop.train.trainer import LMTrainer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ==============================================================================
# Helpers: lightweight Qwen3.6 layout stubs & quantized mock linear
# ==============================================================================


class _ShapeOnlyLinear(nn.Module):
    """Zero-allocation linear stub exposing `(in_features, out_features)` for parameter matching."""

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)


class _StubFullAttn(nn.Module):
    def __init__(self, hidden: int = 2048, q_out: int = 8192, o_in: int = 4096) -> None:
        super().__init__()
        self.q_proj = _ShapeOnlyLinear(hidden, q_out)
        self.k_proj = _ShapeOnlyLinear(hidden, 512)
        self.v_proj = _ShapeOnlyLinear(hidden, 512)
        self.o_proj = _ShapeOnlyLinear(o_in, hidden)


class _StubLinearAttn(nn.Module):
    def __init__(self, hidden: int = 2048, *, use_qkvz: bool = False, both_in_projs: bool = False) -> None:
        super().__init__()
        if both_in_projs:
            self.in_proj_qkv = _ShapeOnlyLinear(hidden, 8192)
            self.in_proj_qkvz = _ShapeOnlyLinear(hidden, 12288)
        elif use_qkvz:
            self.in_proj_qkvz = _ShapeOnlyLinear(hidden, 12288)
        else:
            self.in_proj_qkv = _ShapeOnlyLinear(hidden, 8192)
        self.out_proj = _ShapeOnlyLinear(4096, hidden)


class _StubDecoderLayer(nn.Module):
    def __init__(self, layer_type: str, *, use_qkvz: bool = False, both_in_projs: bool = False) -> None:
        super().__init__()
        self.layer_type = layer_type
        if layer_type == "full_attention":
            self.self_attn = _StubFullAttn()
        else:
            self.linear_attn = _StubLinearAttn(use_qkvz=use_qkvz, both_in_projs=both_in_projs)


class _StubQwen36TextModel(nn.Module):
    def __init__(self, num_layers: int = 40, *, use_qkvz: bool = False, both_in_projs: bool = False) -> None:
        super().__init__()
        layer_types = ["full_attention" if (i + 1) % 4 == 0 else "linear_attention" for i in range(num_layers)]
        self.config = SimpleNamespace(num_hidden_layers=num_layers, layer_types=layer_types, hidden_size=2048)
        # Tiny 1-element reference embedding so device/dtype lookup works with near-zero RAM
        self.embed_tokens = nn.Embedding(2, 4)
        self.layers = nn.ModuleList(
            _StubDecoderLayer(lt, use_qkvz=use_qkvz, both_in_projs=both_in_projs) for lt in layer_types
        )


class _StubQwen36Model(nn.Module):
    def __init__(self, num_layers: int = 40, *, use_qkvz: bool = False, both_in_projs: bool = False) -> None:
        super().__init__()
        self.model = _StubQwen36TextModel(num_layers=num_layers, use_qkvz=use_qkvz, both_in_projs=both_in_projs)


class _MockNVFP4Linear(nn.Module):
    """Frozen buffer-backed NVFP4 linear module (no float `.weight` parameter, mirrors CompressedLinear)."""

    def __init__(self, w_float: torch.Tensor) -> None:
        super().__init__()
        packed, scale, gscale = quantize_nvfp4_tensor(w_float, group_size=16)
        self.register_buffer("weight_packed", packed)
        self.register_buffer("weight_scale", scale)
        self.register_buffer("weight_global_scale", gscale)
        self.group_size = 16

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = dequantize_nvfp4_tensor(
            self.weight_packed,
            self.weight_scale,
            self.weight_global_scale,
            dtype=x.dtype,
            group_size=self.group_size,
        )
        return F.linear(x, w)


# ==============================================================================
# 1. Step-0 exact IEEE-754 identity + Step-0 up.grad + 2-step SGD down.grad
# ==============================================================================


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_step0_exact_ieee754_identity_and_two_step_sgd_grads(dtype: torch.dtype) -> None:
    torch.manual_seed(0)
    proj = _LoRAResidualProj(in_features=32, out_features=64, rank=8, alpha=None, seed=7)
    assert proj.scale == 1.0
    assert proj.is_zero() is True

    x = torch.randn(2, 5, 32, dtype=dtype)
    base_y = torch.randn(2, 5, 64, dtype=dtype)
    delta = proj(x).to(dtype=dtype)
    assert torch.equal(base_y + delta, base_y), f"Step-0 residual violated IEEE-754 bit-identity in {dtype}"

    # Full ToyForCausalLM Step-0 bit-identity in both eval (no_grad) and train (grad-enabled) modes
    model = build_toy(seed=0).to(dtype=dtype)
    ids = torch.randint(4, 90, (2, 12))
    labels = ids.clone()

    model.eval()
    with torch.no_grad():
        base_logits_eval = model(input_ids=ids).logits.clone()
    model.train()
    base_logits_train = model(input_ids=ids).logits.detach().clone()

    mgr = install_d1_lora(model, D1LoRAConfig(enabled=True, layer_start=4, layer_end=12, rank=6, seed=3))
    for p in model.parameters():
        p.requires_grad_(False)
    for p in mgr.adapters.parameters():
        p.requires_grad_(True)

    model.eval()
    with torch.no_grad():
        lora_logits_eval = model(input_ids=ids).logits
    assert torch.equal(lora_logits_eval, base_logits_eval), "Step-0 eval logits must be exact torch.equal"

    model.train()
    out0 = model(input_ids=ids, labels=labels)
    assert torch.equal(out0.logits.detach(), base_logits_train), "Step-0 train logits must be exact torch.equal"

    # Step 0 backward: EVERY up.grad must be strictly non-zero and finite
    out0.loss.backward()
    up_params = [p for n, p in mgr.adapters.named_parameters() if n.endswith(".up")]
    down_params = [p for n, p in mgr.adapters.named_parameters() if n.endswith(".down")]
    assert len(up_params) == 16 and len(down_params) == 16  # 8 layers * 2 projs/layer
    for p in up_params:
        assert p.grad is not None
        assert torch.isfinite(p.grad).all()
        assert float(p.grad.abs().sum()) > 0.0, "Step-0 up.grad must be strictly non-zero"
    for p in down_params:
        assert p.grad is not None and torch.isfinite(p.grad).all()

    # Run 2 SGD steps: after step 1 updates `up`, step 2 MUST yield finite non-zero `down.grad`
    opt = torch.optim.SGD(list(mgr.adapters.parameters()), lr=0.1)
    opt.step()
    opt.zero_grad()

    assert mgr.stats()["d1_lora/up_abs_mean"] > 0.0

    out1 = model(input_ids=ids, labels=labels)
    out1.loss.backward()
    for p in down_params:
        assert p.grad is not None
        assert torch.isfinite(p.grad).all()
        assert float(p.grad.abs().sum()) > 0.0, "Step-1/2 down.grad must be non-zero and finite"
    opt.step()
    opt.zero_grad()

    out2 = model(input_ids=ids, labels=labels)
    assert torch.isfinite(out2.loss)


# ==============================================================================
# 2. Frozen expert SHA-256 bit-identity + LMTrainer E2E + Checkpoint Reload
# ==============================================================================


def test_trainer_e2e_d1_lora_sha256_and_checkpoint_reload_parity(tmp_path) -> None:
    run_dir = str(tmp_path / "d1_lora_run")
    cfg = {
        "seed": 0,
        "stage": "s2_2_d1_lora_toy",
        "model": {"name": "toy", "dtype": "float32", "device_map": "cpu", "toy": {"seed": 0}},
        "loop": {"enabled": False},
        "gdn2": {"enabled": False},
        "hssd": {"enabled": False},
        "lsa": {"enabled": False},
        "experts": {"enabled": False},
        "d1_lora": {"enabled": True, "layer_start": 4, "layer_end": 12, "rank": 6, "seed": 5},
        "train": {
            "steps": 3,
            "grad_accum": 2,
            "micro_batch": 1,
            "log_every": 1,
            "ckpt_every": 3,
            "eval_every": 3,
            "sha256_frozen_experts": True,
            "train_d1_lora": True,
            "trainable": [
                r"self_attn\.(q_norm|k_norm)",
                r"input_layernorm|post_attention_layernorm",
            ],
            "allowed_groups": ["d1_lora", "attention", "norms"],
            "optim": {
                "lr": 1.0e-3,
                "warmup_steps": 1,
                "lr_mult": {"d1_lora": 20.0},
            },
        },
        "data": {
            "seq_len": 48,
            "sources": [{"name": "toy_cot", "weight": 0.6}, {"name": "toy_text", "weight": 0.4}],
        },
        "eval_quick": {"n_batches": 2, "depths": [1]},
        "_meta_": {"config_path": "inline"},
    }

    tr = LMTrainer(cfg, run_dir)
    st = tr.stack
    assert st.loop is None
    assert st.d1_lora is not None
    assert st.frozen_experts_sha256 is not None

    d1_names = [n for n in st.named_trainable if n.startswith("d1_lora.")]
    assert len(d1_names) == 32  # 8 layers * 2 projs * (down + up)
    assert all(st.named_trainable[n].requires_grad for n in d1_names)

    # Verify optim.py regex lr_mult routes d1_lora.* to 20.0x automatically
    groups = build_param_groups(list(st.named_trainable.items()), cfg["train"]["optim"])
    d1_groups = [g for g in groups if any(n.startswith("d1_lora.") for n in g["names"])]
    norm_groups = [g for g in groups if not any(n.startswith("d1_lora.") for n in g["names"])]
    assert len(d1_groups) == 1 and d1_groups[0]["lr_mult"] == 20.0
    assert d1_groups[0]["lr"] == pytest.approx(2.0e-2)
    assert len(norm_groups) == 1 and norm_groups[0]["lr_mult"] == 1.0

    sha_before = sha256_frozen_experts(st.model)
    assert sha_before == st.frozen_experts_sha256
    before_d1 = {n: st.named_trainable[n].detach().clone() for n in d1_names}

    res = tr.train()
    assert res["step"] == 3

    # Frozen experts MUST stay 100% bit-identical before & after training
    sha_after = sha256_frozen_experts(st.model)
    assert sha_after == sha_before == st.verify_frozen_experts()

    # LoRA adapters must have moved during training
    assert any(not torch.equal(before_d1[n], st.named_trainable[n].detach()) for n in d1_names)
    assert st.d1_lora.stats()["d1_lora/up_abs_mean"] > 0.0

    # Metrics logged cleanly with d1_lora/* stats and d=1 eval only
    with open(os.path.join(run_dir, "metrics.jsonl"), encoding="utf-8") as f:
        recs = [json.loads(line) for line in f]
    step_recs = [r for r in recs if "loss/lm" in r]
    assert step_recs and all("d1_lora/up_abs_mean" in r and "d1_lora/n_hooks" in r for r in step_recs)
    assert any("eval/loss_d1" in r for r in recs)
    assert not any("eval/loss_d2" in r for r in recs)

    # Checkpoint save & exact reload parity
    probe_ids = torch.randint(4, 90, (2, 24))
    st.model.eval()
    with torch.no_grad():
        expected_logits = st.model(input_ids=probe_ids).logits.clone()

    mgr = CheckpointManager(run_dir, keep_last=0)
    ck = mgr.load(mgr.latest(), map_location="cpu", restore_rng=False)
    assert ck is not None
    smeta = (ck.get("state") or {}).get("stack") or {}
    assert isinstance(smeta.get("d1_lora"), dict)
    assert smeta["d1_lora"]["resolved_rank"] == 6
    assert smeta["d1_lora"]["n_hooks"] == 16

    fresh = build_stack(cfg)
    counts = fresh.restore_checkpoint(ck["params"], strict=True, prior_stack_meta=smeta)
    assert counts["trainable"] == len(st.named_trainable)
    assert sha256_frozen_experts(fresh.model) == sha_before

    fresh.model.eval()
    with torch.no_grad():
        reloaded_logits = fresh.model(input_ids=probe_ids).logits
    assert torch.equal(reloaded_logits, expected_logits), "Reloaded d1_lora checkpoint must match logits bit-for-bit"


def test_stack_d1_lora_disabled_and_freeze_policies() -> None:
    base_cfg = {
        "model": {"name": "toy", "dtype": "float32", "device_map": "cpu", "toy": {"seed": 0}},
        "loop": {"enabled": False},
        "train": {
            "trainable": [r"input_layernorm"],
            "allowed_groups": ["norms"],
        },
    }
    st_off = build_stack(base_cfg)
    assert st_off.d1_lora is None
    assert st_off.describe()["d1_lora"] is None
    assert not any(n.startswith("d1_lora.") for n in st_off.named_all)

    # allowed_groups excluding "d1_lora" keeps d1_lora frozen even when enabled=True
    cfg_excl = {
        **base_cfg,
        "d1_lora": {"enabled": True, "layer_start": 4, "layer_end": 8, "rank": 4},
    }
    st_excl = build_stack(cfg_excl)
    assert st_excl.d1_lora is not None
    assert any(n.startswith("d1_lora.") for n in st_excl.named_all)
    assert not any(n.startswith("d1_lora.") for n in st_excl.named_trainable)

    # train_d1_lora: false also keeps d1_lora frozen even if "d1_lora" is in allowed_groups
    cfg_frozen = {
        **base_cfg,
        "d1_lora": {"enabled": True, "layer_start": 4, "layer_end": 8, "rank": 4},
        "train": {
            "trainable": [r"input_layernorm"],
            "allowed_groups": ["d1_lora", "norms"],
            "train_d1_lora": False,
        },
    }
    st_frozen = build_stack(cfg_frozen)
    assert st_frozen.d1_lora is not None
    assert not any(n.startswith("d1_lora.") for n in st_frozen.named_trainable)

    # Prior-stage d1_lora metadata is carried when restoring into a stack without d1_lora
    st_off.restore_checkpoint({}, strict=False, prior_stack_meta={"d1_lora": st_frozen.d1_lora.describe()})
    assert st_off.carried_d1_lora_cfg is not None
    assert st_off.describe()["d1_lora"]["resolved_rank"] == 4


# ==============================================================================
# 3. Analytical parameter math & solve_rank_for_target_params (qkv & qkvz)
# ==============================================================================


def test_qwen36_analytical_param_math_and_rank_solver() -> None:
    # Closed-form verification on Qwen3.6-35B-A3B [12, 36): 6 full-attn + 18 linear-attn
    p46 = compute_qwen36_d1_lora_params(46)
    assert p46["n_full_layers"] == 6
    assert p46["n_linear_layers"] == 18
    assert p46["params_per_rank"] == 393_216
    assert p46["full_attn_lora_params"] == 98_304 * 46
    assert p46["linear_attn_lora_params"] == 294_912 * 46
    assert p46["lora_params"] == 18_087_936
    assert p46["norm_params"] == QWEN36_NORM_PARAMS == 168_960
    assert p46["total_with_norms"] == 18_256_896
    rel_err_46 = abs(p46["total_with_norms"] - S1_LOOP_TOTAL_TRAINABLE_PARAMS) / S1_LOOP_TOTAL_TRAINABLE_PARAMS
    assert rel_err_46 <= 0.01  # 0.85%

    p47 = compute_qwen36_d1_lora_params(47)
    assert p47["lora_params"] == 18_481_152
    assert p47["total_with_norms"] == 18_650_112
    rel_err_47 = abs(p47["total_with_norms"] - S1_LOOP_TOTAL_TRAINABLE_PARAMS) / S1_LOOP_TOTAL_TRAINABLE_PARAMS
    assert rel_err_47 <= 0.015  # 1.29%

    p39_qkvz = compute_qwen36_d1_lora_params(39, use_qkvz=True)
    assert p39_qkvz["params_per_rank"] == 466_944
    assert p39_qkvz["linear_attn_lora_params"] == 368_640 * 39
    assert p39_qkvz["lora_params"] == 18_210_816
    assert p39_qkvz["total_with_norms"] == 18_379_776
    rel_err_39 = abs(p39_qkvz["total_with_norms"] - S1_LOOP_TOTAL_TRAINABLE_PARAMS) / S1_LOOP_TOTAL_TRAINABLE_PARAMS
    assert rel_err_39 <= 0.005  # 0.18%

    # Dynamic rank solver on 40L Qwen3.6 stubs for BOTH in_proj_qkv (8192) and in_proj_qkvz (12288)
    stub_qkv = _StubQwen36Model(num_layers=40, use_qkvz=False)
    r_qkv = solve_rank_for_target_params(stub_qkv, S1_LOOP_ADAPTER_PARAMS, layer_start=12, layer_end=36)
    assert r_qkv == 46
    mgr_qkv = install_d1_lora(
        stub_qkv,
        D1LoRAConfig(enabled=True, layer_start=12, layer_end=36, target_params=S1_LOOP_ADAPTER_PARAMS),
    )
    assert mgr_qkv.rank == 46
    assert mgr_qkv.param_count() == 18_087_936
    assert abs((mgr_qkv.param_count() + QWEN36_NORM_PARAMS) - 18_410_000) / 18_410_000 <= 0.02

    stub_qkvz = _StubQwen36Model(num_layers=40, use_qkvz=True)
    r_qkvz = solve_rank_for_target_params(stub_qkvz, S1_LOOP_ADAPTER_PARAMS, layer_start=12, layer_end=36)
    assert r_qkvz == 39
    mgr_qkvz = install_d1_lora(
        stub_qkvz,
        D1LoRAConfig(enabled=True, layer_start=12, layer_end=36, target_params=S1_LOOP_ADAPTER_PARAMS),
    )
    assert mgr_qkvz.rank == 39
    assert mgr_qkvz.param_count() == 18_210_816
    assert abs((mgr_qkvz.param_count() + QWEN36_NORM_PARAMS) - 18_410_000) / 18_410_000 <= 0.02

    # Layer exposing BOTH in_proj_qkv and in_proj_qkvz must NOT double-hook (picks first matching input proj!)
    stub_both = _StubQwen36Model(num_layers=40, both_in_projs=True)
    mgr_both = install_d1_lora(stub_both, D1LoRAConfig(enabled=True, layer_start=12, layer_end=36, rank=46))
    assert len(mgr_both._hooks) == 48  # 24 layers * 2 projs/layer, NOT 24*2 + 18 = 66!
    assert mgr_both.param_count() == 18_087_936


# ==============================================================================
# 4. Quantized NVFP4 compatibility & validation errors
# ==============================================================================


def test_nvfp4_quantized_linear_hooking_and_bounds_validation() -> None:
    torch.manual_seed(0)
    model = build_toy(seed=0)
    # Replace core layer 4's linear_attn.in_proj_qkv and layer 7's self_attn.q_proj with frozen NVFP4 mocks
    l4_w = model.model.layers[4].linear_attn.in_proj_qkv.weight.detach().clone()
    l7_w = model.model.layers[7].self_attn.q_proj.weight.detach().clone()
    model.model.layers[4].linear_attn.in_proj_qkv = _MockNVFP4Linear(l4_w)
    model.model.layers[7].self_attn.q_proj = _MockNVFP4Linear(l7_w)

    l4_packed_before = model.model.layers[4].linear_attn.in_proj_qkv.weight_packed.clone()
    l7_packed_before = model.model.layers[7].self_attn.q_proj.weight_packed.clone()

    ids = torch.randint(4, 90, (1, 10))
    model.eval()
    with torch.no_grad():
        base_out = model(input_ids=ids).logits.clone()

    mgr = install_d1_lora(model, D1LoRAConfig(enabled=True, layer_start=4, layer_end=8, rank=4))
    assert mgr.stats()["d1_lora/n_hooks"] == 8.0
    assert mgr.param_count() > 0
    with torch.no_grad():
        step0_out = model(input_ids=ids).logits
    assert torch.equal(step0_out, base_out)
    assert torch.equal(model.model.layers[4].linear_attn.in_proj_qkv.weight_packed, l4_packed_before)
    assert torch.equal(model.model.layers[7].self_attn.q_proj.weight_packed, l7_packed_before)

    # Verify gradients flow to LoRA adapters attached to frozen NVFP4 linear modules without touching packed buffers
    model.train()
    for p in model.parameters():
        p.requires_grad_(False)
    for p in mgr.adapters.parameters():
        p.requires_grad_(True)
    loss = model(input_ids=ids, labels=ids).loss
    loss.backward()
    assert mgr.adapters["L4"]["in_proj_qkv"].up.grad is not None
    assert float(mgr.adapters["L4"]["in_proj_qkv"].up.grad.abs().sum()) > 0.0
    assert mgr.adapters["L7"]["q_proj"].up.grad is not None
    assert float(mgr.adapters["L7"]["q_proj"].up.grad.abs().sum()) > 0.0
    assert torch.equal(model.model.layers[4].linear_attn.in_proj_qkv.weight_packed, l4_packed_before)
    assert torch.equal(model.model.layers[7].self_attn.q_proj.weight_packed, l7_packed_before)

    # Out-of-bounds layer ranges raise clear ValueError
    model_oob = build_toy(seed=0)
    with pytest.raises(ValueError, match="out of bounds"):
        install_d1_lora(model_oob, D1LoRAConfig(enabled=True, layer_start=12, layer_end=36, rank=4))
    with pytest.raises(ValueError, match="0 <= layer_start < layer_end"):
        install_d1_lora(model_oob, D1LoRAConfig(enabled=True, layer_start=8, layer_end=8, rank=4))


# ==============================================================================
# 5. set_enabled(False) and remove() lifecycle
# ==============================================================================


def test_set_enabled_false_and_remove_restore_exact_base_output() -> None:
    torch.manual_seed(0)
    model = build_toy(seed=0).eval()
    ids = torch.randint(4, 90, (2, 12))
    with torch.no_grad():
        base_out = model(input_ids=ids).logits.clone()

    mgr = install_d1_lora(model, D1LoRAConfig(enabled=True, layer_start=4, layer_end=12, rank=4, seed=11))
    with torch.no_grad():
        for pdict in mgr.adapters.values():
            for proj in pdict.values():
                proj.up.fill_(0.25)
        active_out = model(input_ids=ids).logits.clone()
    assert not torch.equal(active_out, base_out), "non-zero LoRA weights must change model output"

    mgr.set_enabled(False)
    with torch.no_grad():
        disabled_out = model(input_ids=ids).logits
    assert torch.equal(disabled_out, base_out), "set_enabled(False) must restore exact base output"

    mgr.set_enabled(True)
    with torch.no_grad():
        reenabled_out = model(input_ids=ids).logits
    assert torch.equal(reenabled_out, active_out), "set_enabled(True) must restore exact LoRA output"

    mgr.remove()
    mgr.remove()  # idempotent
    assert len(mgr._hooks) == 0
    assert not hasattr(model.model, "d1_lora")
    with torch.no_grad():
        removed_out = model(input_ids=ids).logits
    assert torch.equal(removed_out, base_out), "remove() must cleanly detach all hooks and restore base output"


# ==============================================================================
# 6. configs/stages/s2_2_d1_lora_control.yaml validation
# ==============================================================================


def test_s2_2_d1_lora_control_stage_config_invariants() -> None:
    cfg_path = os.path.join(ROOT, "configs", "stages", "s2_2_d1_lora_control.yaml")
    with open(cfg_path, encoding="utf-8") as f:
        raw_text = f.read()
    assert "HBM" not in raw_text, "stage config must use GDDR7 ECC / VRAM wording, never HBM"

    cfg = load_config(cfg_path)
    assert cfg.stage == "s2_2_d1_lora_control"
    assert cfg.stage_type == "lm"
    assert cfg.run.name == "s2_2_d1_lora_control"
    assert cfg.model.name == "unsloth/Qwen3.6-35B-A3B-NVFP4"

    assert cfg.loop.enabled is False
    assert cfg.gdn2.enabled is False
    assert cfg.hssd.enabled is False
    assert cfg.lsa.enabled is False
    assert cfg.experts.enabled is False

    assert cfg.d1_lora.enabled is True
    assert cfg.d1_lora.layer_start == 12
    assert cfg.d1_lora.layer_end == 36
    assert cfg.d1_lora.rank == 46
    assert cfg.d1_lora.target_params == S1_LOOP_ADAPTER_PARAMS
    D1LoRAConfig.from_dict(dict(cfg.d1_lora)).validate(num_layers=40)

    assert cfg.train.steps == 1000
    assert cfg.train.micro_batch == 4
    assert cfg.train.grad_accum == 4
    assert cfg.train.gradient_checkpointing is True
    assert cfg.train.sha256_frozen_experts is True
    assert cfg.train.optim.lr == pytest.approx(3.0e-5)
    assert cfg.train.optim.warmup_steps == 50
    assert dict(cfg.train.optim.lr_mult) == {"d1_lora": 20.0}
    assert "d1_lora" in cfg.train.allowed_groups

    assert cfg.data.seq_len == 8192
    assert cfg.data.holdout_mod == 100
    assert cfg.data.holdout_eval_buckets == 2
    src_names = [s["name"] for s in cfg.data.sources]
    assert not any("opus46" in n.lower() for n in src_names), f"restricted opus46 source leaked: {src_names}"
    assert src_names == ["openthoughts3", "nemotron_post", "finepdfs_mix", "fineweb_edu", "fable51"]
    assert sum(float(s["weight"]) for s in cfg.data.sources) == pytest.approx(1.0)
    assert cfg.data.decontaminate.enabled is True
    assert cfg.data.decontaminate.ngram == 13
    assert list(cfg.data.decontaminate.groups) == ["aime_2026", "gpqa", "hle", "arc_agi_2", "kernelbench"]

    assert cfg.eval_quick.n_batches == 16
    assert list(cfg.eval_quick.depths) == [1]


# ==============================================================================
# 7. Non-reentrant gradient checkpointing for non-looped stages
# ==============================================================================


class _MockHFCheckpointableTextModel(nn.Module):
    """HF-style text decoder wrapping `ToyTextModel` with real non-reentrant activation checkpointing."""

    def __init__(self, inner: nn.Module) -> None:
        super().__init__()
        self.config = inner.config
        self.embed_tokens = inner.embed_tokens
        self.layers = inner.layers
        self.norm = inner.norm
        self.rotary_emb = inner.rotary_emb
        self.gradient_checkpointing = False
        self.gc_kwargs: dict[str, object] = {}
        self.ckpt_layer_calls = 0

    def forward(
        self,
        input_ids=None,
        inputs_embeds=None,
        past_key_values=None,
        use_cache: bool = False,
        output_hidden_states: bool = False,
        **kwargs,
    ):
        from torch.utils.checkpoint import checkpoint as torch_ckpt

        from qloop.model.toy import ToyHybridCache, ToyOutput

        h = self.embed_tokens(input_ids) if inputs_embeds is None else inputs_embeds
        if use_cache and past_key_values is None:
            past_key_values = ToyHybridCache(self.config)
        start = past_key_values.get_seq_length() if past_key_values is not None else 0
        cache_position = torch.arange(start, start + h.shape[1], device=h.device)
        position_ids = cache_position[None].expand(h.shape[0], -1)
        pos = self.rotary_emb(h, position_ids)
        all_h = [h] if output_hidden_states else None
        use_ckpt = self.gradient_checkpointing and self.training and torch.is_grad_enabled() and not use_cache
        use_reentrant = bool(self.gc_kwargs.get("use_reentrant", True))
        for layer in self.layers[: self.config.num_hidden_layers]:
            if use_ckpt:
                self.ckpt_layer_calls += 1

                def _run_layer(x_in: torch.Tensor, _layer=layer):
                    return _layer(
                        x_in,
                        position_embeddings=pos,
                        attention_mask=None,
                        position_ids=position_ids,
                        past_key_values=None,
                        cache_position=cache_position,
                    )

                h = torch_ckpt(_run_layer, h, use_reentrant=use_reentrant)
            else:
                h = layer(
                    h,
                    position_embeddings=pos,
                    attention_mask=None,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    cache_position=cache_position,
                )
            if output_hidden_states:
                all_h.append(h)
        h = self.norm(h)
        return ToyOutput(h, past_key_values, tuple(all_h) if all_h else None)


class _MockHFCheckpointableCausalLM(nn.Module):
    """Mock HF CausalLM supporting `gradient_checkpointing_enable(gradient_checkpointing_kwargs=...)`."""

    def __init__(self, seed: int = 0) -> None:
        super().__init__()
        toy = build_toy(seed=seed)
        self.config = toy.config
        self.model = _MockHFCheckpointableTextModel(toy.model)
        self.lm_head = toy.lm_head
        self.gc_calls: list[dict[str, object] | None] = []
        self.input_require_grads_called = False

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs: dict[str, object] | None = None) -> None:
        self.gc_calls.append(None if gradient_checkpointing_kwargs is None else dict(gradient_checkpointing_kwargs))
        self.model.gradient_checkpointing = True
        self.model.gc_kwargs = dict(gradient_checkpointing_kwargs or {})

    def enable_input_require_grads(self) -> None:
        self.input_require_grads_called = True

        def _make_inputs_require_grad(_mod: nn.Module, _inp, out: torch.Tensor) -> None:
            out.requires_grad_(True)

        self.model.embed_tokens.register_forward_hook(_make_inputs_require_grad)

    def forward(self, input_ids, labels=None, past_key_values=None, use_cache: bool = False, **kwargs):
        from qloop.model.toy import ToyCausalLMOutput

        out = self.model(input_ids=input_ids, past_key_values=past_key_values, use_cache=use_cache)
        logits = self.lm_head(out.last_hidden_state)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, logits.shape[-1]).float(),
                labels[:, 1:].reshape(-1),
                ignore_index=-100,
            )
        return ToyCausalLMOutput(logits, loss, out.past_key_values)


class _LegacyTextModelWrapper(nn.Module):
    """Outer wrapper lacking `gradient_checkpointing_enable`; inner `model` uses legacy zero-arg API."""

    def __init__(self, seed: int = 0) -> None:
        super().__init__()
        toy = build_toy(seed=seed)
        self.config = toy.config
        self.model = _MockHFCheckpointableTextModel(toy.model)
        self.lm_head = toy.lm_head
        self.legacy_gc_calls = 0
        self.legacy_input_grads_calls = 0

        outer = self

        def _legacy_gc_enable() -> None:
            outer.legacy_gc_calls += 1
            outer.model.gradient_checkpointing = True
            outer.model.gc_kwargs = {"use_reentrant": False}

        def _legacy_enable_input_grads() -> None:
            outer.legacy_input_grads_calls += 1

            def _hook(_mod: nn.Module, _inp, out: torch.Tensor) -> None:
                out.requires_grad_(True)

            outer.model.embed_tokens.register_forward_hook(_hook)

        self.model.gradient_checkpointing_enable = _legacy_gc_enable  # type: ignore[attr-defined]
        self.model.enable_input_require_grads = _legacy_enable_input_grads  # type: ignore[attr-defined]


def test_build_stack_gradient_checkpointing_mock_hf_models(monkeypatch: pytest.MonkeyPatch) -> None:
    import qloop.train.stack as stack_mod

    orig_load_model = stack_mod.load_model
    _, toy_tok, _ = orig_load_model({"name": "toy", "dtype": "float32", "device_map": "cpu", "toy": {"seed": 0}})

    # 1. Standard HF-style model: non-reentrant checkpointing + frozen base + d1_lora grads + eval/use_cache=True
    mock_hf = _MockHFCheckpointableCausalLM(seed=0)
    monkeypatch.setattr(stack_mod, "load_model", lambda _mcfg: (mock_hf, toy_tok, {"backend": "mock_hf"}))
    cfg_d1 = {
        "model": {"name": "mock_hf"},
        "loop": {"enabled": False},
        "d1_lora": {"enabled": True, "layer_start": 4, "layer_end": 12, "rank": 4, "seed": 2},
        "train": {
            "gradient_checkpointing": True,
            "train_d1_lora": True,
            "trainable": [],
            "allowed_groups": ["d1_lora"],
        },
    }
    st = build_stack(cfg_d1)
    assert st.loop is None
    assert st.d1_lora is not None
    assert st.load_info["gradient_checkpointing"] is True
    assert mock_hf.gc_calls == [{"use_reentrant": False}]
    assert mock_hf.input_require_grads_called is True
    assert not st.parts.text_model.embed_tokens.weight.requires_grad

    ids = torch.randint(4, 90, (2, 12))
    st.model.train()
    h = st.hidden(ids)
    logits = F.linear(h, st.lm_head_weight)
    loss = F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]).float(), ids[:, 1:].reshape(-1))
    loss.backward()
    assert mock_hf.model.ckpt_layer_calls == mock_hf.config.num_hidden_layers
    assert st.parts.text_model.embed_tokens.weight.grad is None
    up_params = [p for n, p in st.d1_lora.adapters.named_parameters() if n.endswith(".up")]
    assert len(up_params) == 16
    for p in up_params:
        assert p.grad is not None and torch.isfinite(p.grad).all()
        assert float(p.grad.abs().sum()) > 0.0

    # Verify model.eval() + use_cache=True prefill and incremental decode still work cleanly
    ckpt_calls_before_eval = mock_hf.model.ckpt_layer_calls
    st.model.eval()
    with torch.no_grad():
        out_nocache = st.model(input_ids=ids, use_cache=False)
        out_prefill = st.model(input_ids=ids[:, :-1], use_cache=True)
        assert out_prefill.past_key_values is not None
        out_step = st.model(input_ids=ids[:, -1:], past_key_values=out_prefill.past_key_values, use_cache=True)
    assert mock_hf.model.ckpt_layer_calls == ckpt_calls_before_eval
    assert torch.isfinite(out_step.logits).all()
    assert torch.allclose(out_step.logits, out_nocache.logits[:, -1:], atol=1e-5, rtol=1e-5)

    # 2. Fallback to `text_model` + `TypeError` fallback to zero-arg `gradient_checkpointing_enable()`
    legacy_mod = _LegacyTextModelWrapper(seed=1)
    monkeypatch.setattr(stack_mod, "load_model", lambda _mcfg: (legacy_mod, toy_tok, {"backend": "mock_legacy"}))
    st_legacy = build_stack(cfg_d1)
    assert st_legacy.load_info["gradient_checkpointing"] is True
    assert legacy_mod.legacy_gc_calls == 1
    assert legacy_mod.legacy_input_grads_calls == 1

    # 3. Fallback `embed_tokens` hook (when `enable_input_require_grads` is absent) is idempotent
    no_irg_mod = _MockHFCheckpointableCausalLM(seed=3)
    no_irg_mod.enable_input_require_grads = None  # type: ignore[assignment]
    assert stack_mod._enable_gradient_checkpointing(no_irg_mod, no_irg_mod.model) is True
    assert stack_mod._enable_gradient_checkpointing(no_irg_mod, no_irg_mod.model) is True
    assert getattr(no_irg_mod.model.embed_tokens, "_qloop_input_grad_hook", False) is True
    assert len(no_irg_mod.model.embed_tokens._forward_hooks) == 1

    # 4. Looped stage (`st.loop is not None`) must NOT double-enable HF gradient checkpointing
    mock_looped = _MockHFCheckpointableCausalLM(seed=2)
    monkeypatch.setattr(stack_mod, "load_model", lambda _mcfg: (mock_looped, toy_tok, {"backend": "mock_hf"}))
    cfg_loop = {
        "model": {"name": "mock_hf"},
        "loop": {"enabled": True, "prelude_end": 4, "core_end": 12, "num_passes": 2, "target_virtual_layers": 24},
        "train": {
            "gradient_checkpointing": True,
            "trainable": [r"input_layernorm"],
        },
    }
    st_looped = build_stack(cfg_loop)
    assert st_looped.loop is not None
    assert mock_looped.gc_calls == []
    assert "gradient_checkpointing" not in st_looped.load_info


def test_build_stack_gradient_checkpointing_on_toy_model() -> None:
    cfg = {
        "model": {"name": "toy", "dtype": "float32", "device_map": "cpu", "toy": {"seed": 0}},
        "loop": {"enabled": False},
        "d1_lora": {"enabled": True, "layer_start": 4, "layer_end": 12, "rank": 4, "seed": 3},
        "train": {
            "gradient_checkpointing": True,
            "train_d1_lora": True,
            "trainable": [r"input_layernorm|post_attention_layernorm"],
            "allowed_groups": ["d1_lora", "norms"],
        },
    }
    st = build_stack(cfg)
    assert st.loop is None
    assert st.d1_lora is not None
    # ToyForCausalLM lacks `gradient_checkpointing_enable()`, so build_stack skips cleanly without error
    assert st.load_info["gradient_checkpointing"] is False

    ids = torch.randint(4, 90, (2, 12))
    st.model.train()
    out = st.model(input_ids=ids, labels=ids)
    out.loss.backward()
    up_params = [p for n, p in st.d1_lora.adapters.named_parameters() if n.endswith(".up")]
    assert len(up_params) == 16
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all() and float(p.grad.abs().sum()) > 0.0 for p in up_params
    )

    st.model.eval()
    gen = st.model.greedy_generate(ids[:1, :6], max_new_tokens=3)
    assert gen.shape == (1, 9)
