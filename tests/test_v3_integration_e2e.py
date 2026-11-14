"""End-to-end post-merge integration tests for Loop-HSSD v3 (`40L / 64L / 88L A6B`).

Verifies:
1. `build_stack` with `hssd.enabled=True` + `loop(num_passes=3, bridge_mode="subspace", attn_adapter_rank=8)`
   + `lsa(num_query_passes=3)` + `sha256_frozen_experts=True`: step-0 `torch.equal` identity,
   `LMTrainer` training step, checkpoint save/restore parity across `R in {1, 2, 3}`, and stage-to-stage
   `carried_hssd_cfg` handoff.
2. `run_probe_notebook_suite(smoke_toy=True, push_to_hub=False)` structured JSON + Markdown + Gate -1 suite.
3. CLI `qloop export-pack` -> `OptimizedLoopInferenceEngine.from_pretrained` (`R=1/2/3`, `1.0x` FP8 SharedKV,
   chunked prefill, `benchmark`, and `qloop infer` CLI) + `V3_PIPELINE_ORDER` config resolution.
"""
from __future__ import annotations

import ast
import json
import os

import torch

from qloop.cli import main
from qloop.colab import run_probe_notebook_suite
from qloop.config import load_config
from qloop.hub import V3_PIPELINE_ORDER, resolve_stage_config_path
from qloop.infer import InferenceConfig, OptimizedLoopInferenceEngine
from qloop.model.experts import param_group_of
from qloop.train.checkpoint import CheckpointManager
from qloop.train.stack import build_stack
from qloop.train.trainer import LMTrainer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOY_LM = os.path.join(ROOT, "configs", "local", "toy_lm.yaml")

V3_TOY_OVERRIDES = [
    "loop.prelude_end=4",
    "loop.core_end=12",
    "loop.num_passes=3",
    "loop.target_virtual_layers=32",
    "loop.train_pass_probs=[0.15,0.50,0.35]",
    "loop.bridge_mode=subspace",
    "loop.bridge_rank=16",
    "loop.bridge_init=[0.05,0.85,0.10]",
    "loop.attn_adapter_rank=8",
    "hssd.enabled=true",
    "hssd.chunk_size=16",
    "hssd.window_chunks=2",
    "hssd.num_pages=4",
    "hssd.k_pages=2",
    "hssd.segment_chunks=2",
    "lsa.enabled=true",
    "lsa.num_query_passes=3",
    "train.sha256_frozen_experts=true",
    'train.allowed_groups=["loop_adapters","hssd_adapters","gdn2_adapters","lsa_indexers","router","attention","norms","deltanet","expert_deltas"]',
    "eval_quick.depths=[1,2,3]",
]


def test_v3_stack_trainer_hssd_loop_lsa_roundtrip(tmp_path):
    assert param_group_of("model.layers.0.linear_attn.hssd_adapter.page_keys") == "hssd_adapters"

    cfg = load_config(
        TOY_LM,
        [
            *V3_TOY_OVERRIDES,
            "train.steps=2",
            "train.grad_accum=1",
            "train.log_every=1",
            "train.ckpt_every=2",
            "train.eval_every=2",
        ],
    )

    # 1. Step-0 exact bit-identity across R=1/2/3 via zero-init PassExitGate + ReEntryBridge
    st0 = build_stack(cfg)
    st0.model.eval()
    assert st0.hssd is not None and st0.gdn2 is None
    assert st0.loop is not None and st0.loop.spec.num_passes == 3
    assert st0.lsa is not None and st0.lsa.cfg.num_query_passes == 3
    assert st0.frozen_experts_sha256 is not None

    ids = torch.randint(4, 200, (1, 32))
    with torch.no_grad():
        st0.lsa.set_mode("dense")
        st0.loop.set_active_passes(1)
        d1_logits = st0.model(input_ids=ids).logits
        st0.loop.set_active_passes(2)
        d2_logits = st0.model(input_ids=ids).logits
        st0.loop.set_active_passes(3)
        d3_logits = st0.model(input_ids=ids).logits
    assert torch.equal(d2_logits, d1_logits), "step-0 R=2 Loop-HSSD stack drifted from R=1"
    assert torch.equal(d3_logits, d1_logits), "step-0 R=3 Loop-HSSD stack drifted from R=1"

    # 2. Train 2 steps with LMTrainer + verify metrics, HSSD budget aux loss, and frozen MoE SHA-256 guard
    run_dir = str(tmp_path / "v3_train")
    trainer = LMTrainer(cfg, run_dir)
    summary = trainer.train()
    assert summary["step"] == 2
    trainer.stack.verify_frozen_experts()

    metrics_path = os.path.join(run_dir, "metrics.jsonl")
    assert os.path.isfile(metrics_path)
    with open(metrics_path, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    assert any("loss/hssd_budget" in r for r in rows)
    assert any("eval/loss_d3" in r for r in rows)

    # 3. Checkpoint save/restore parity across R in {1, 2, 3} + carried_hssd_cfg stage handoff
    trainer.stack.model.eval()
    if trainer.stack.lsa is not None:
        trainer.stack.lsa.set_mode("sparse")
    expected_by_depth: dict[int, torch.Tensor] = {}
    with torch.no_grad():
        for r in (1, 2, 3):
            trainer.stack.loop.set_active_passes(r)
            expected_by_depth[r] = trainer.stack.model(input_ids=ids).logits.detach().clone()

    mgr = CheckpointManager(run_dir, keep_last=0)
    ckpt = mgr.load(mgr.latest())
    assert ckpt is not None
    prior_meta = (ckpt.get("state") or {}).get("stack") or {}
    assert isinstance(prior_meta.get("hssd"), dict) and prior_meta["hssd"]["num_pages"] == 4

    # 3a. Exact checkpoint save/restore parity across R in {1, 2, 3}
    st_restored = build_stack(cfg)
    st_restored.restore_checkpoint(ckpt["params"], strict=True, prior_stack_meta=prior_meta)
    st_restored.model.eval()
    if st_restored.lsa is not None:
        st_restored.lsa.set_mode("sparse")

    with torch.no_grad():
        for r in (1, 2, 3):
            st_restored.loop.set_active_passes(r)
            got = st_restored.model(input_ids=ids).logits
            assert torch.allclose(got, expected_by_depth[r], atol=1e-6), f"depth {r} restore mismatch"

    # 3b. Cumulative stage handoff: carried_hssd_cfg + passthrough_tensors preserved when hssd is inactive
    passthrough_cfg = load_config(
        TOY_LM,
        [*[ov for ov in V3_TOY_OVERRIDES if not ov.startswith("hssd.")], "gdn2.enabled=false"],
    )
    st_carried = build_stack(passthrough_cfg)
    assert st_carried.hssd is None
    st_carried.restore_checkpoint(ckpt["params"], strict=False, prior_stack_meta=prior_meta)
    assert st_carried.carried_hssd_cfg is not None
    assert st_carried.describe()["hssd"]["num_pages"] == 4
    assert any("hssd_adapter." in k for k in st_carried.checkpoint_params())


def test_v3_probe_notebook_suite_smoke(tmp_path, monkeypatch):
    # Verify qloop.colab keeps zero top-level torch imports
    colab_py = os.path.join(ROOT, "src", "qloop", "colab.py")
    with open(colab_py, encoding="utf-8") as f:
        tree = ast.parse(f.read(), filename=colab_py)
    top_imports = [
        alias.name
        for node in tree.body
        if isinstance(node, ast.Import)
        for alias in node.names
    ] + [
        node.module or ""
        for node in tree.body
        if isinstance(node, ast.ImportFrom)
    ]
    assert all(not name.startswith("torch") for name in top_imports)

    monkeypatch.setenv("QLOOP_RUN_ROOT", str(tmp_path))
    rep = run_probe_notebook_suite(
        config_path=os.path.join(ROOT, "configs", "stages", "s0_probe.yaml"),
        smoke_toy=True,
        seq_len=48,
        n_seqs=2,
        target_virtual=24,
        run_math_gate=True,
        push_to_hub=False,
        debug=1,
    )
    for key in (
        "summary",
        "gates",
        "recommended_overrides",
        "hssd_layer_profile",
        "hssd_ablation_matrix",
        "markdown",
        "json_path",
        "md_path",
    ):
        assert key in rep, f"missing key {key!r} in probe report"
    assert rep["gates"]["gate_minus_1_math"]["passed"] is True
    assert rep["gates"]["0a_backward"]["passed"] is True
    assert rep["gates"]["0a_surgery"]["passed"] is True
    assert "0p_1_expert_headroom" in rep["gates"]
    assert "0p_2_reentry_anchors" in rep["gates"]
    assert "0p_3_positional_extrapolation" in rep["gates"]
    assert rep["gates"]["0p_4_hssd_layer_profile"]["passed"] is True
    assert rep["hssd_layer_profile"].get("profile_source") is None
    assert rep["hssd_layer_profile"]["retained_cache"]["has_compact_storage"] is True
    assert len(rep["hssd_ablation_matrix"]) == 6
    assert "eval_tasks" in rep["hssd_ablation_matrix"]["6_complete_hssd"]
    assert os.path.isfile(rep["json_path"]) and os.path.getsize(rep["json_path"]) > 0
    assert os.path.isfile(rep["md_path"]) and os.path.getsize(rep["md_path"]) > 0
    assert "Gate `0p-4`" in rep["markdown"]
    assert "Gate `1-Ablate`" in rep["markdown"]


def test_v3_cli_export_pack_and_inference_engine_roundtrip(tmp_path, monkeypatch):
    for st_name in V3_PIPELINE_ORDER:
        cpath = resolve_stage_config_path(st_name, os.path.join(ROOT, "configs"))
        assert os.path.isfile(cpath), f"missing config for v3 stage {st_name}: {cpath}"

    monkeypatch.setenv("QLOOP_RUN_ROOT", str(tmp_path))
    train_args = [
        "train",
        "-c",
        TOY_LM,
        "-s",
        "run.name=v3_pack_src",
        "-s",
        "train.steps=2",
        "-s",
        "train.ckpt_every=2",
    ]
    for ov in V3_TOY_OVERRIDES:
        train_args.extend(["-s", ov])
    assert main(train_args) == 0

    src_run = tmp_path / "v3_pack_src"
    pack_dir = tmp_path / "shipped_pack"
    export_args = [
        "export-pack",
        "-c",
        TOY_LM,
        "--ckpt",
        str(src_run),
        "--out",
        str(pack_dir),
        "--max-shard-bytes",
        "262144",
        "-s",
        "run.name=v3_export_pack",
    ]
    for ov in V3_TOY_OVERRIDES:
        export_args.extend(["-s", ov])
    assert main(export_args) == 0

    for fname in (
        "config.json",
        "model.safetensors.index.json",
        "qloop_pack_manifest.json",
        "configuration_qwen36_loophssd.py",
        "modeling_qwen36_loophssd.py",
    ):
        assert (pack_dir / fname).is_file(), f"missing Whole Model Pack file: {fname}"

    engine = OptimizedLoopInferenceEngine.from_pretrained(
        str(pack_dir),
        cfg=InferenceConfig(prefill_chunk_size=16, fp8_kv_cache=True, compile_decode=False),
    )
    kv_footprints: list[float] = []
    for r in (1, 2, 3):
        with engine.depth(r):
            res = engine.generate("Step-by-step proof:", max_new_tokens=6, temperature=0.0)
        assert res.metrics.generated_tokens == 6
        assert res.metrics.active_passes == r
        kv_footprints.append(res.metrics.kv_cache_mb)
    assert kv_footprints[0] > 0.0
    assert abs(kv_footprints[1] - kv_footprints[0]) < 1e-6
    assert abs(kv_footprints[2] - kv_footprints[0]) < 1e-6

    bench = engine.benchmark(prompt_len=24, gen_len=4, depths=(1, 2, 3))
    depths_map = bench.get("depths") or bench
    assert len(depths_map) == 3

    assert main(
        [
            "infer",
            "--pack",
            str(pack_dir),
            "--prompt",
            "Test CLI inference",
            "--depth",
            "3",
            "--max-new-tokens",
            "4",
            "--prefill-chunk-size",
            "16",
            "--kv-quant",
            "fp8",
        ]
    ) == 0
