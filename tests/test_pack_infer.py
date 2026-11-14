"""Comprehensive tests for Whole Model Pack export, low-disk streaming & OptimizedLoopInferenceEngine."""
from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import os
import shutil
import sys
import types
from pathlib import Path
from typing import Any

import pytest
import torch

from qloop.data.tokenizer import ToyTokenizer
from qloop.export_pack import (
    PACK_MANIFEST_FILE,
    SAFETENSORS_INDEX_FILE,
    WholeModelPackConfig,
    export_whole_model_pack,
    load_whole_model_pack,
    push_whole_model_pack,
    stream_whole_model_pack,
)
from qloop.infer import (
    CONFIG_FILENAME,
    MODELING_FILENAME,
    InferenceConfig,
    OptimizedLoopInferenceEngine,
    Qwen36LoopHSSDForCausalLM,
)
from qloop.infer.hf_custom_modeling import load_safetensors_file, save_safetensors_file
from qloop.model.experts import attach_hot_expert_deltas
from qloop.model.gdn2 import GDN2Config, install_gdn2
from qloop.model.introspect import inspect_model
from qloop.model.loop import LoopSpec, apply_loop
from qloop.model.lsa import LSAConfig, install_lsa
from qloop.model.toy import ToyConfig, ToyForCausalLM
from qloop.train.stack import Stack


def _build_full_v3_toy_stack(seed: int = 17, *, mor: bool = False) -> Stack:
    """Build a ToyForCausalLM wrapped with non-trivial GDN-2 + D-LSA + HotExpertDeltas + 3-pass Loop."""
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed + 100)
    tok = ToyTokenizer()

    tcfg = ToyConfig(
        vocab_size=tok.vocab_size,
        hidden_size=48,
        moe_intermediate_size=24,
        shared_expert_intermediate_size=24,
        num_hidden_layers=6,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        partial_rotary_factor=0.5,
        linear_num_heads=4,
        linear_head_dim=12,
        linear_conv_kernel_dim=4,
        num_experts=8,
        num_experts_per_tok=2,
        layer_types=(
            "linear_attention",  # L0 prelude
            "full_attention",    # L1 prelude (ends macro-block cleanly)
            "linear_attention",  # L2 core
            "linear_attention",  # L3 core
            "full_attention",    # L4 core (SharedKV + LSA, ends macro-block cleanly)
            "full_attention",    # L5 coda (LSA)
        ),
    )
    model = ToyForCausalLM(tcfg).eval()

    # 1) Install GDN-2 and populate non-zero learned adapter weights
    gdn2 = install_gdn2(model, GDN2Config(rank=12, chunk_size=8))
    with torch.no_grad():
        for ad in gdn2.adapters.values():
            ad.erase_up.weight.normal_(0.0, 0.10, generator=gen)
            ad.write_up.weight.normal_(0.0, 0.10, generator=gen)
            ad.decay_up.weight.normal_(0.0, 0.06, generator=gen)

    # 2) Install sparse D-LSA (`window=8` so 25-token prompt triggers sparse bucket attention!)
    lsa = install_lsa(
        model,
        LSAConfig(
            mode="sparse",
            index_dim=16,
            block_size=8,
            pool=2,
            avg_bucket_keys=4,
            max_bucket_keys=8,
            num_select=2,
            window=8,
            sinks=2,
            query_chunk=8,
        ),
    )
    with torch.no_grad():
        for ix in lsa.indexers.values():
            ix.wk.normal_(0.0, 0.20, generator=gen)
            ix.wq.normal_(0.0, 0.20, generator=gen)

    # 3) Attach HotExpertDeltas with non-zero learned B matrices
    hot_map = {2: [0, 2, 5], 3: [1, 3, 6], 4: [0, 4, 7]}
    deltas = attach_hot_expert_deltas(model, hot_map, rank=8, alpha=8.0)
    with torch.no_grad():
        for ld in deltas.deltas.values():
            for p in ld.gate_up_B.values():
                p.normal_(0.0, 0.05, generator=gen)
            for p in ld.down_B.values():
                p.normal_(0.0, 0.05, generator=gen)

    # 4) Wrap with 3-pass Sandwich Loop (`6L` -> `9L` -> `12L`) + PassRouterAdapters
    loop_spec = LoopSpec(
        prelude_end=2,
        core_end=5,
        num_passes=3,
        target_virtual_layers=12,
        kv_share=True,
        gate_init=0.35,
        bridge_rank=8,
        step_embedding=True,
        per_pass_router=True,
        router_adapter_rank=8,
        router_repel_init=1.75,
        mor=mor,
    )
    loop = apply_loop(model, loop_spec)
    with torch.no_grad():
        loop.adapters.gates[0].uniform_(0.25, 0.55, generator=gen)
        loop.adapters.gates[1].uniform_(0.30, 0.65, generator=gen)
        if loop.adapters.bridge.step_emb is not None:
            loop.adapters.bridge.step_emb.normal_(0.0, 0.06, generator=gen)
        if loop.adapters.bridge.rank > 0:
            loop.adapters.bridge.up.weight.normal_(0.0, 0.08, generator=gen)
        if loop.adapters.pass_router is not None:
            loop.adapters.pass_router.up.normal_(0.0, 0.20, generator=gen)
            loop.adapters.pass_router.bias.normal_(0.0, 0.10, generator=gen)

    model.eval()
    parts = inspect_model(model)
    return Stack(
        model=model,
        tokenizer=tok,
        parts=parts,
        loop=loop,
        gdn2=gdn2,
        lsa=lsa,
        deltas=deltas,
        load_info={"name": "toy"},
        hot_map=hot_map,
    )


class MockStreamingHfApi:
    """Mock `HfApi` verifying per-shard low-disk cleanup during `upload_file`."""

    def __init__(self, store_dir: str) -> None:
        self.store_dir = store_dir
        self.repos: dict[str, dict[str, str]] = {}
        self.upload_calls: list[dict[str, Any]] = []
        self.max_concurrent_safetensors_on_staging = 0

    def create_repo(
        self,
        repo_id: str,
        repo_type: str = "model",
        private: bool = False,
        exist_ok: bool = True,
    ) -> None:
        del repo_type, private, exist_ok
        self.repos.setdefault(repo_id, {})

    def upload_file(
        self,
        *,
        path_or_fileobj: str,
        path_in_repo: str,
        repo_id: str,
        repo_type: str = "model",
        commit_message: str = "",
    ) -> str:
        del repo_type
        assert os.path.isfile(path_or_fileobj), f"missing local upload source {path_or_fileobj}"
        parent_dir = os.path.dirname(os.path.abspath(path_or_fileobj))
        live_st = [fn for fn in os.listdir(parent_dir) if fn.endswith(".safetensors")]
        self.max_concurrent_safetensors_on_staging = max(
            self.max_concurrent_safetensors_on_staging,
            len(live_st),
        )
        if path_in_repo.endswith(".safetensors"):
            # Strictly <= 1 shard file on local staging disk at any instant!
            assert live_st == [os.path.basename(path_or_fileobj)], (
                f"expected only current shard on disk during streaming, saw {live_st}"
            )

        dst = os.path.join(self.store_dir, repo_id.replace("/", "__"), path_in_repo)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(path_or_fileobj, dst)
        self.repos.setdefault(repo_id, {})[path_in_repo] = dst
        self.upload_calls.append(
            {
                "path_in_repo": path_in_repo,
                "size_bytes": os.path.getsize(dst),
                "live_safetensors": list(live_st),
                "commit_message": commit_message,
            }
        )
        return f"oid-{len(self.upload_calls):04d}"

    def list_repo_files(self, repo_id: str, repo_type: str = "model") -> list[str]:
        del repo_type
        return sorted(self.repos.get(repo_id, {}).keys())

    def download_file(self, repo_id: str, path_in_repo: str, dst_path: str) -> bool:
        src = (self.repos.get(repo_id) or {}).get(path_in_repo)
        if not src or not os.path.isfile(src):
            return False
        os.makedirs(os.path.dirname(os.path.abspath(dst_path)), exist_ok=True)
        shutil.copy2(src, dst_path)
        return True


# ============================================================================ 1 & 2. Export + Multi-Shard Reload Parity

def test_multishard_whole_model_pack_export_and_exact_logit_equivalence(tmp_path) -> None:
    stack = _build_full_v3_toy_stack(seed=23)
    prompt = torch.tensor([[12, 45, 78, 102, 33, 91, 204, 19, 64, 88, 141, 212, 77, 53, 99, 180, 44, 61, 29]])

    # Capture pre-export reference logits across R=1, R=2, R=3
    ref_logits: dict[int, torch.Tensor] = {}
    with torch.no_grad():
        for r in (1, 2, 3):
            stack.loop.set_active_passes(r)
            ref_logits[r] = stack.model(prompt, use_cache=False).logits.detach().clone()

    # Verify R=1, R=2, R=3 are genuinely distinct before export
    assert (ref_logits[1] - ref_logits[2]).abs().max().item() > 1e-2
    assert (ref_logits[2] - ref_logits[3]).abs().max().item() > 1e-2

    # 1) Non-in-place multi-shard export (`max_shard_size_bytes=65536` forces multiple shards)
    pack_dir_copy = tmp_path / "pack_non_mutating"
    exp_copy = export_whole_model_pack(
        stack,
        pack_dir_copy,
        max_shard_size_bytes=65536,
        merge_hot_experts=True,
        merge_in_place=False,
        repo_id="test/qwen36-loophssd-toy",
        eval_summary={"gsm8k_pass1": 0.875},
    )
    assert exp_copy["num_shards"] >= 3
    assert stack.deltas is not None and stack.deltas.attached, "merge_in_place=False must not mutate live stack"

    # 2) In-place multi-shard export (`merge_in_place=True` folds deltas & detaches hooks)
    pack_dir = tmp_path / "pack_inplace"
    exp = export_whole_model_pack(
        stack,
        pack_dir,
        max_shard_size_bytes=65536,
        merge_hot_experts=True,
        merge_in_place=True,
        repo_id="test/qwen36-loophssd-toy",
    )
    assert exp["num_shards"] >= 3
    assert stack.deltas is None, "merge_in_place=True must detach HotExpertDeltas from live stack"

    # Check shard naming, SHA-256 digests, index.json, config.json, standalone .py, and README.md
    for idx, sname in enumerate(exp["shard_files"], start=1):
        assert sname == f"model-{idx:05d}-of-{exp['num_shards']:05d}.safetensors"
        spath = pack_dir / sname
        assert spath.is_file()
        actual_sha = hashlib.sha256(spath.read_bytes()).hexdigest()
        assert actual_sha == exp["manifest"]["shards"][idx - 1]["sha256"]

    index_data = json.loads((pack_dir / SAFETENSORS_INDEX_FILE).read_text(encoding="utf-8"))
    wmap = index_data["weight_map"]
    assert not any("expert_deltas" in k for k in wmap), "merged expert_deltas must not remain in shards"
    assert any("loop_adapters.gates" in k for k in wmap)
    assert any("loop_adapters.pass_router" in k for k in wmap)
    assert any("gdn2_adapter.erase_up.weight" in k for k in wmap)
    assert any("lsa_indexer.wq" in k for k in wmap)
    # Physical layers 0..5 only (never virtual layers 6..11!)
    assert not any(".layers.6." in k for k in wmap)

    cfg_json = json.loads((pack_dir / "config.json").read_text(encoding="utf-8"))
    assert cfg_json["auto_map"]["AutoModelForCausalLM"] == "modeling_qwen36_loophssd.Qwen36LoopHSSDForCausalLM"
    assert cfg_json["base_config"]["num_hidden_layers"] == 6
    assert cfg_json["qloop_v3_arch"]["virtual_layers"] == 12
    assert (pack_dir / PACK_MANIFEST_FILE).is_file()
    assert "OptimizedLoopInferenceEngine" in (pack_dir / "README.md").read_text(encoding="utf-8")

    # Capture post-merge live stack logits (bit-identical `0.0` to reload AND `< 1e-6` to pre-merge!)
    merged_logits: dict[int, torch.Tensor] = {}
    with torch.no_grad():
        for r in (1, 2, 3):
            stack.loop.set_active_passes(r)
            merged_logits[r] = stack.model(prompt, use_cache=False).logits.detach().clone()

    # Python 3.9 AST-check the emitted standalone custom modeling files!
    for py_fn in (CONFIG_FILENAME, MODELING_FILENAME):
        src_txt = (pack_dir / py_fn).read_text(encoding="utf-8")
        ast.parse(src_txt, filename=py_fn, feature_version=(3, 9))

    # Reload Path A: `load_whole_model_pack` (both non-mutating and in-place packs)
    for target_dir in (pack_dir_copy, pack_dir):
        loaded_pack = load_whole_model_pack(target_dir)
        for r in (1, 2, 3):
            with loaded_pack.depth(r):
                out_logits = loaded_pack(prompt, use_cache=False).logits
                reload_diff = (out_logits - merged_logits[r]).abs().max().item()
                assert reload_diff < 1e-6, f"load_whole_model_pack reload diff={reload_diff} at depth={r}"
                fold_ulp_diff = (out_logits - ref_logits[r]).abs().max().item()
                assert fold_ulp_diff < 5e-6, f"fp32 LoRA fold ULP diff={fold_ulp_diff} at depth={r}"

    # Reload Path B: `OptimizedLoopInferenceEngine.from_pretrained`
    engine = OptimizedLoopInferenceEngine.from_pretrained(pack_dir)
    engine.depth(1)
    assert engine.active_passes == 1 and engine.effective_layers == 6
    for r, expected_layers in ((1, 6), (2, 9), (3, 12)):
        with engine.depth(r):
            assert engine.active_passes == r and engine.effective_layers == expected_layers
            eng_logits = engine.forward_logits(prompt)
            reload_diff = (eng_logits - merged_logits[r]).abs().max().item()
            assert reload_diff < 1e-6, f"OptimizedLoopInferenceEngine reload diff={reload_diff} at depth={r}"
            assert (eng_logits - ref_logits[r]).abs().max().item() < 5e-6
    # Context manager restored depth(1)
    assert engine.active_passes == 1

    # Reload Path C: Emitted standalone `modeling_qwen36_loophssd.py` dynamic import (`trust_remote_code` path)
    sys.path.insert(0, str(pack_dir))
    try:
        mod_spec = importlib.util.spec_from_file_location(
            "_test_emitted_modeling",
            str(pack_dir / MODELING_FILENAME),
        )
        assert mod_spec is not None and mod_spec.loader is not None
        emitted_mod = importlib.util.module_from_spec(mod_spec)
        mod_spec.loader.exec_module(emitted_mod)
        standalone_model: Qwen36LoopHSSDForCausalLM = emitted_mod.Qwen36LoopHSSDForCausalLM.from_pretrained(
            str(pack_dir),
            trust_remote_code=True,
        )
        for r in (1, 2, 3):
            standalone_model.set_active_passes(r)
            st_logits = standalone_model(prompt, use_cache=False).logits
            assert (st_logits - merged_logits[r]).abs().max().item() < 1e-6
            assert (st_logits - ref_logits[r]).abs().max().item() < 5e-6
    finally:
        sys.path.remove(str(pack_dir))


# ============================================================== 3. Chunked Prefill Equivalence & Last-Token Head

def test_chunked_prefill_exact_equivalence_and_last_token_only_logits(tmp_path) -> None:
    stack = _build_full_v3_toy_stack(seed=41)
    pack_dir = tmp_path / "pack_chunked"
    export_whole_model_pack(stack, pack_dir, max_shard_size_bytes=65536)

    # 25-token prompt -> chunks of [8, 8, 8, 1], testing multi-chunk + single-token tail + sparse LSA
    rng = torch.Generator().manual_seed(99)
    prompt_25 = torch.randint(10, 240, (1, 25), generator=rng)

    # Exact fp64 check (`< 1e-12 << 1e-6` eliminating hardware AMX GEMM-vs-GEMV fp32 ULP drift)
    eng_full_64 = OptimizedLoopInferenceEngine.from_pretrained(
        pack_dir,
        cfg=InferenceConfig(prefill_chunk_size=4096, temperature=0.0),
        dtype="float64",
    )
    eng_chunked_64 = OptimizedLoopInferenceEngine.from_pretrained(
        pack_dir,
        cfg=InferenceConfig(prefill_chunk_size=8, temperature=0.0),
        dtype="float64",
    )
    for r in (1, 2, 3):
        with eng_full_64.depth(r), eng_chunked_64.depth(r):
            lc64, _, nc64 = eng_chunked_64.prefill(prompt_25, chunk_size=8)
            lf64, _, nf64 = eng_full_64.prefill(prompt_25, chunk_size=4096)
            assert nc64 == 4 and nf64 == 1
            assert (lc64 - lf64).abs().max().item() < 1e-12

    # Production fp32 check: single (1, 1, H) lm_head call, tight fp32 ULP match & bit-identical generated tokens
    eng_full = OptimizedLoopInferenceEngine.from_pretrained(
        pack_dir,
        cfg=InferenceConfig(prefill_chunk_size=4096, temperature=0.0),
    )
    eng_chunked = OptimizedLoopInferenceEngine.from_pretrained(
        pack_dir,
        cfg=InferenceConfig(prefill_chunk_size=8, temperature=0.0),
    )

    for r in (1, 2, 3):
        with eng_full.depth(r), eng_chunked.depth(r):
            # Hook lm_head on eng_chunked to verify it ONLY materializes logits on the final token [1, 1, H]
            lm_head_shapes: list[tuple[int, ...]] = []
            hook = eng_chunked.lm_head.register_forward_hook(
                lambda _m, inp, _out, rec=lm_head_shapes: rec.append(tuple(inp[0].shape))
            )
            try:
                logits_chunked, _, n_chunks = eng_chunked.prefill(prompt_25, chunk_size=8)
            finally:
                hook.remove()

            assert n_chunks == 4, f"25 tokens at chunk_size=8 must stream 4 chunks, got {n_chunks}"
            assert lm_head_shapes == [(1, 1, 48)], (
                f"Chunked prefill must call lm_head ONCE on final token (1, 1, H), saw {lm_head_shapes}"
            )

            logits_full, _, n_full = eng_full.prefill(prompt_25, chunk_size=4096)
            assert n_full == 1
            diff = (logits_chunked - logits_full).abs().max().item()
            assert diff < 5e-6, f"chunked vs full prefill fp32 logit diff={diff} at depth={r}"

            # Also verify un-cached full-sequence final logit matches
            uncached_last = eng_full.forward_logits(prompt_25)[:, -1:, :]
            assert (logits_chunked - uncached_last).abs().max().item() < 5e-6

            # Verify identical autoregressive generated tokens across 10 steps!
            gen_full = eng_full.generate(prompt_25, max_new_tokens=10, ignore_eos=True)
            gen_chunk = eng_chunked.generate(prompt_25, max_new_tokens=10, ignore_eos=True)
            assert gen_chunk.token_ids == gen_full.token_ids, (
                f"generated tokens diverged at depth={r}: {gen_chunk.token_ids} vs {gen_full.token_ids}"
            )
            assert gen_chunk.metrics.prefill_chunks == 4
            assert gen_full.metrics.prefill_chunks == 1


# ==================================================== 4. Quantized 1.0x SharedKV, Sampling, MoR, Compile & Benchmark

@pytest.mark.parametrize("quant_mode", ["fp8", "hadamard_int4"])
def test_quantized_shared_kv_cache_savings_and_depth_invariance(tmp_path, quant_mode: str) -> None:
    stack = _build_full_v3_toy_stack(seed=55)
    pack_dir = tmp_path / f"pack_{quant_mode}"
    export_whole_model_pack(stack, pack_dir, max_shard_size_bytes=131072)

    prompt = torch.randint(8, 220, (1, 32), generator=torch.Generator().manual_seed(7))
    eng_nat = OptimizedLoopInferenceEngine.from_pretrained(
        pack_dir,
        cfg=InferenceConfig(kv_cache_quant="native", active_passes=3),
    )
    eng_q = OptimizedLoopInferenceEngine.from_pretrained(
        pack_dir,
        cfg=InferenceConfig(kv_cache_quant=quant_mode, active_passes=3),
    )

    res_nat_d3 = eng_nat.generate(prompt, max_new_tokens=6, depth=3, ignore_eos=True)
    res_q_d1 = eng_q.generate(prompt, max_new_tokens=6, depth=1, ignore_eos=True)
    res_q_d3 = eng_q.generate(prompt, max_new_tokens=6, depth=3, ignore_eos=True)

    # 1.0x SharedKV across passes: full-attn KV cache footprint at R=3 MUST equal R=1!
    assert res_q_d3.metrics.kv_cache_mb == res_q_d1.metrics.kv_cache_mb
    assert res_q_d3.metrics.kv_cache_mb > 0.0
    # Quantized KV cache must use dramatically less memory than fp32 native cache
    assert res_q_d3.metrics.kv_cache_mb < 0.40 * res_nat_d3.metrics.kv_cache_mb
    # Recurrent state grows with active passes (separate O(1) linear-attn slots per pass)
    assert res_q_d3.metrics.recurrent_state_mb > res_q_d1.metrics.recurrent_state_mb
    # Cross-pass expert diversity (`PassRouterAdapters` repulsion) activates more unique experts at R=3 than R=1
    assert res_q_d3.metrics.unique_experts_per_tok > res_q_d1.metrics.unique_experts_per_tok


def test_sampling_mor_early_exit_compile_and_benchmark(tmp_path) -> None:
    stack = _build_full_v3_toy_stack(seed=71, mor=True)
    # Bias MoR router strongly toward pass 1 so causal MoR early exit triggers on decode
    assert stack.loop is not None and stack.loop.adapters.mor_router is not None
    with torch.no_grad():
        stack.loop.adapters.mor_router.bias.copy_(torch.tensor([5.0, -3.0, -3.0]))

    pack_dir = tmp_path / "pack_mor_compile"
    export_whole_model_pack(stack, pack_dir, max_shard_size_bytes=131072)

    eng = OptimizedLoopInferenceEngine.from_pretrained(
        pack_dir,
        cfg=InferenceConfig(
            active_passes=3,
            mor_early_exit=True,
            compile_decode=True,
            temperature=0.75,
            top_k=6,
            top_p=0.9,
            seed=1234,
        ),
    )
    res_a = eng.generate("Hello Loop-HSSD!", max_new_tokens=8, ignore_eos=True)
    res_b = eng.generate("Hello Loop-HSSD!", max_new_tokens=8, ignore_eos=True)
    assert res_a.token_ids == res_b.token_ids, "seeded top-k/top-p sampling must be reproducible"
    assert res_a.metrics.compiled_decode is True
    assert res_a.metrics.mean_active_passes < 3.0, "causal MoR early exit should skip deeper passes on decode"
    assert res_a["prefill_tok_s"] > 0.0 and res_a["decode_tok_s"] > 0.0

    # Stop token handling
    first_tok = res_a.token_ids[2]
    res_stop = eng.generate("Hello Loop-HSSD!", max_new_tokens=8, stop_token_ids=[first_tok])
    assert res_stop.finish_reason == "stop"
    assert res_stop.stop_token_id == first_tok
    assert len(res_stop.token_ids) <= 3

    # Verify notebook convenience alias `InferenceConfig(fp8_kv_cache=True)`
    alias_cfg = InferenceConfig(prefill_chunk_size=4096, fp8_kv_cache=True, compile_decode=False)
    assert alias_cfg.kv_cache_quant == "fp8"
    assert alias_cfg.fp8_kv_cache is True

    # Structured benchmark across depths (1, 2, 3)
    bench = eng.benchmark(prompt_len=24, gen_len=6, depths=(1, 2, 3))
    assert set(bench["depths"].keys()) == {1, 2, 3}
    assert bench["depths"][1]["effective_layers"] == 6
    assert bench["depths"][2]["effective_layers"] == 9
    assert bench["depths"][3]["effective_layers"] == 12
    assert bench[1]["effective_layers"] == 6 and bench[3]["effective_layers"] == 12


# ============================================================================ 5. Colab Low-Disk Streaming Mode

def test_stream_whole_model_pack_low_disk_cleanup_and_hub_reload(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("HF_HOME", str(tmp_path / "sandbox_hf_home"))
    stack = _build_full_v3_toy_stack(seed=89)
    prompt = torch.tensor([[9, 31, 67, 120, 44, 18, 200, 73, 52, 108]])
    ref_d3 = stack.model(prompt, use_cache=False).logits.detach().clone()

    staging_dir = tmp_path / "colab_staging"
    staging_dir.mkdir()
    hub_store = tmp_path / "mock_hub_store"
    mock_api = MockStreamingHfApi(str(hub_store))
    repo_id = "qloop-test/Qwen3.6-35B-A6B-LoopHSSD-Pack"

    cfg = WholeModelPackConfig(
        repo_id=repo_id,
        max_shard_size_bytes=65536,
        stream_low_disk=True,
    )
    stream_res = stream_whole_model_pack(
        stack,
        cfg.repo_id,
        max_shard_size_bytes=cfg.max_shard_size_bytes,
        staging_dir=staging_dir,
        clean_hf_cache_each_shard=False,
        api=mock_api,
    )

    assert stream_res["stream_low_disk"] is True
    assert stream_res["num_shards"] >= 3
    # Peak staging disk strictly bounded by a single shard (never the sum of all shards!)
    assert stream_res["peak_temp_disk_bytes"] < stream_res["total_bytes"]
    assert mock_api.max_concurrent_safetensors_on_staging == 1
    # Staging directory completely clean after stream finishes
    assert os.listdir(staging_dir) == []

    repo_files = set(mock_api.list_repo_files(repo_id))
    expected_required = {
        SAFETENSORS_INDEX_FILE,
        "config.json",
        "generation_config.json",
        CONFIG_FILENAME,
        MODELING_FILENAME,
        PACK_MANIFEST_FILE,
        "README.md",
        "tokenizer_config.json",
    }
    assert expected_required <= repo_files
    for i in range(1, stream_res["num_shards"] + 1):
        assert f"model-{i:05d}-of-{stream_res['num_shards']:05d}.safetensors" in repo_files

    # Also verify `push_whole_model_pack(..., stream_low_disk=True)` entrypoint
    repo_id_2 = "qloop-test/Qwen3.6-Push-Entry"
    push_res = push_whole_model_pack(
        stack,
        repo_id_2,
        stream_low_disk=True,
        max_shard_size_bytes=65536,
        api=mock_api,
    )
    assert push_res["num_shards"] == stream_res["num_shards"]

    # Reload directly from Mock Hub repo via `OptimizedLoopInferenceEngine.from_pretrained`
    hub_engine = OptimizedLoopInferenceEngine.from_pretrained(repo_id, api=mock_api)
    with hub_engine.depth(3):
        hub_logits = hub_engine.forward_logits(prompt)
        assert (hub_logits - ref_d3).abs().max().item() < 1e-6


# ============================================================================ 6. Standalone HF Modeling (No qloop)

def _load_standalone_modeling_without_qloop(pack_dir: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    """Load emitted `modeling_qwen36_loophssd.py` with `qloop` hidden so `_HAS_QLOOP is False`."""
    cfg_py = pack_dir / CONFIG_FILENAME
    mod_py = pack_dir / MODELING_FILENAME
    ast.parse(cfg_py.read_text(encoding="utf-8"), feature_version=(3, 9))
    ast.parse(mod_py.read_text(encoding="utf-8"), feature_version=(3, 9))

    real_find_spec = importlib.util.find_spec

    def _no_qloop_find_spec(name: str, package: Any = None) -> Any:
        if name == "qloop" or name.startswith("qloop."):
            return None
        return real_find_spec(name, package)

    monkeypatch.setattr(importlib.util, "find_spec", _no_qloop_find_spec)

    stub_tf = types.ModuleType("transformers")

    class _PreTrainedConfig:
        model_type = "qwen36_loophssd"

        def __init__(self, **kwargs: Any) -> None:
            for k, v in kwargs.items():
                setattr(self, k, v)

    class _PreTrainedModel(torch.nn.Module):
        config_class: Any = None

        def __init__(self, config: Any) -> None:
            super().__init__()
            self.config = config

        def post_init(self) -> None:
            pass

    class _GenerationMixin:
        pass

    class _AutoConfig:
        @staticmethod
        def for_model(model_type: str, **kw: Any) -> ToyConfig:
            valid = {f for f in ToyConfig.__dataclass_fields__ if f in kw}
            clean = {k: kw[k] for k in valid}
            if "layer_types" in clean and isinstance(clean["layer_types"], list):
                clean["layer_types"] = tuple(clean["layer_types"])
            return ToyConfig(**clean)

    class _AutoModelForCausalLM:
        @staticmethod
        def from_config(cfg: ToyConfig, **kw: Any) -> ToyForCausalLM:
            return ToyForCausalLM(cfg)

    stub_tf.PreTrainedConfig = _PreTrainedConfig  # type: ignore[attr-defined]
    stub_tf.PreTrainedModel = _PreTrainedModel  # type: ignore[attr-defined]
    stub_tf.GenerationMixin = _GenerationMixin  # type: ignore[attr-defined]
    stub_tf.AutoConfig = _AutoConfig  # type: ignore[attr-defined]
    stub_tf.AutoModelForCausalLM = _AutoModelForCausalLM  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "transformers", stub_tf)

    pkg_name = f"_standalone_pack_{abs(hash(str(pack_dir)))}"
    pkg = types.ModuleType(pkg_name)
    pkg.__path__ = [str(pack_dir)]  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, pkg_name, pkg)

    spec_cfg = importlib.util.spec_from_file_location(f"{pkg_name}.configuration_qwen36_loophssd", cfg_py)
    assert spec_cfg is not None and spec_cfg.loader is not None
    mod_cfg = importlib.util.module_from_spec(spec_cfg)
    monkeypatch.setitem(sys.modules, spec_cfg.name, mod_cfg)
    spec_cfg.loader.exec_module(mod_cfg)

    spec_mod = importlib.util.spec_from_file_location(f"{pkg_name}.modeling_qwen36_loophssd", mod_py)
    assert spec_mod is not None and spec_mod.loader is not None
    mod_model = importlib.util.module_from_spec(spec_mod)
    monkeypatch.setitem(sys.modules, spec_mod.name, mod_model)
    spec_mod.loader.exec_module(mod_model)

    assert mod_model._HAS_QLOOP is False
    return mod_model


def test_standalone_hf_modeling_no_qloop_parity_and_strict_safetensors(tmp_path, monkeypatch) -> None:
    """Verify `_HAS_QLOOP == False` matches native `qloop` (< 1e-5) with PassRouter + MoR + sparse LSA."""
    stack = _build_full_v3_toy_stack(seed=37, mor=True)
    assert stack.loop is not None and stack.loop.adapters.mor_router is not None
    gen = torch.Generator().manual_seed(370)
    with torch.no_grad():
        # Non-zero MoR router so tokens split across depths {1, 2, 3}, exercising write_mask!
        stack.loop.adapters.mor_router.weight.normal_(0.0, 0.45, generator=gen)
        stack.loop.adapters.mor_router.bias.copy_(torch.tensor([0.05, 0.15, 0.0]))

    # 25-token prompt (> LSA window=8) so sparse Ward bucket attention is exercised
    prompt = torch.tensor([[((i * 19 + 7) % 240) + 4 for i in range(25)]], dtype=torch.long)

    pack_dir = tmp_path / "standalone_gdn2_mor_pack"
    export_whole_model_pack(stack, pack_dir, max_shard_size_bytes=65536, merge_in_place=False)

    # Reference logits from native `qloop` loaded pack across R=1, 2, 3
    native_pack = load_whole_model_pack(pack_dir)
    assert native_pack.loop is not None
    refs: dict[int, torch.Tensor] = {}
    with torch.no_grad():
        for r in (1, 2, 3):
            with native_pack.depth(r):
                refs[r] = native_pack(prompt, use_cache=False).logits.detach().clone()
    assert (refs[1] - refs[2]).abs().max().item() > 1e-3
    assert (refs[2] - refs[3]).abs().max().item() > 1e-3

    # Load via standalone HF modeling with `qloop` completely hidden (`_HAS_QLOOP == False`)
    mod_model = _load_standalone_modeling_without_qloop(pack_dir, monkeypatch)
    sm = mod_model.Qwen36LoopHSSDForCausalLM.from_pretrained(str(pack_dir), strict=True)
    assert sm.max_passes == 3

    with torch.no_grad():
        for r in (1, 2, 3):
            with sm.depth(r):
                out_r = sm(prompt, use_cache=False).logits
                diff = (out_r - refs[r]).abs().max().item()
                assert diff < 1e-5, f"Standalone _HAS_QLOOP=False mismatch at depth={r}: {diff:.3e}"

                last_r = sm(prompt, use_cache=False, logits_to_keep=1).logits
                assert last_r.shape == (1, 1, stack.tokenizer.vocab_size)
                assert (last_r - refs[r][:, -1:, :]).abs().max().item() < 1e-5

    # Verify `strict=True` rejects unexpected `.safetensors` keys
    bad_extra_dir = tmp_path / "bad_extra_key_pack"
    shutil.copytree(pack_dir, bad_extra_dir)
    idx_path = bad_extra_dir / SAFETENSORS_INDEX_FILE
    idx_data = json.loads(idx_path.read_text(encoding="utf-8"))
    first_shard = sorted(set(idx_data["weight_map"].values()))[0]
    shard_tensors = load_safetensors_file(str(bad_extra_dir / first_shard))
    shard_tensors["model.bogus.unexpected_weight"] = torch.ones(4)
    save_safetensors_file(shard_tensors, str(bad_extra_dir / first_shard))
    idx_data["weight_map"]["model.bogus.unexpected_weight"] = first_shard
    idx_path.write_text(json.dumps(idx_data), encoding="utf-8")
    with pytest.raises(RuntimeError, match=r"unexpected_backbone=.*bogus"):
        mod_model.Qwen36LoopHSSDForCausalLM.from_pretrained(str(bad_extra_dir), strict=True)

    # Verify `strict=True` rejects missing required `.safetensors` keys
    bad_missing_dir = tmp_path / "bad_missing_key_pack"
    shutil.copytree(pack_dir, bad_missing_dir)
    idx_m_path = bad_missing_dir / SAFETENSORS_INDEX_FILE
    idx_m_data = json.loads(idx_m_path.read_text(encoding="utf-8"))
    target_key = next(k for k in idx_m_data["weight_map"] if k.endswith("loop_adapters.pass_router.up"))
    target_shard = idx_m_data["weight_map"].pop(target_key)
    m_tensors = load_safetensors_file(str(bad_missing_dir / target_shard))
    del m_tensors[target_key]
    save_safetensors_file(m_tensors, str(bad_missing_dir / target_shard))
    idx_m_path.write_text(json.dumps(idx_m_data), encoding="utf-8")
    with pytest.raises(RuntimeError, match=r"missing_loop=.*pass_router\.up"):
        mod_model.Qwen36LoopHSSDForCausalLM.from_pretrained(str(bad_missing_dir), strict=True)


def test_standalone_hf_modeling_no_qloop_v3_hssd_subspace_pass_attn_and_dlsa(tmp_path, monkeypatch) -> None:
    """Verify standalone `_HAS_QLOOP == False` supports HSSD v3 + subspace bridge + PassAttn + multi-pass LSA."""
    gen = torch.Generator().manual_seed(911)
    tok = ToyTokenizer()
    tcfg = ToyConfig(
        vocab_size=tok.vocab_size,
        hidden_size=48,
        moe_intermediate_size=24,
        shared_expert_intermediate_size=24,
        num_hidden_layers=6,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        partial_rotary_factor=0.5,
        linear_num_heads=4,
        linear_head_dim=12,
        linear_conv_kernel_dim=4,
        num_experts=8,
        num_experts_per_tok=2,
        layer_types=(
            "linear_attention",
            "full_attention",
            "linear_attention",
            "linear_attention",
            "full_attention",
            "full_attention",
        ),
    )
    prompt = torch.tensor([[((i * 13 + 11) % 240) + 4 for i in range(19)]], dtype=torch.long)

    # Branch A: Full native vs standalone `< 1e-5` parity when `qloop.model.hssd` is installed
    has_native_hssd = importlib.util.find_spec("qloop.model.hssd") is not None
    if has_native_hssd:
        from qloop.model.hssd import HSSDConfig, apply_hssd

        torch.manual_seed(91)
        native_model = ToyForCausalLM(tcfg).eval()
        hcfg = HSSDConfig(num_pages=4, k_pages=2, rank=8, chunk_size=8, window_chunks=2)
        hssd_h = apply_hssd(native_model, hcfg)
        with torch.no_grad():
            for ad in hssd_h.adapters.values():
                ad.erase_up.weight.normal_(0.0, 0.08, generator=gen)
                ad.write_up.weight.normal_(0.0, 0.08, generator=gen)
                ad.decay_up.weight.normal_(0.0, 0.06, generator=gen)
                ad.theta_proj.weight.normal_(0.0, 0.08, generator=gen)
                ad.swa_q_up.weight.normal_(0.0, 0.08, generator=gen)
                ad.swa_k_up.weight.normal_(0.0, 0.08, generator=gen)
                ad.swa_v_up.weight.normal_(0.0, 0.08, generator=gen)

        lsa_kw: dict[str, Any] = dict(
            mode="sparse",
            index_dim=16,
            block_size=8,
            pool=2,
            avg_bucket_keys=4,
            max_bucket_keys=8,
            num_select=2,
            window=8,
            sinks=2,
            query_chunk=8,
        )
        if "num_query_passes" in LSAConfig.__dataclass_fields__:
            lsa_kw["num_query_passes"] = 3
        lsa_h = install_lsa(native_model, LSAConfig(**lsa_kw))
        with torch.no_grad():
            for ix in lsa_h.indexers.values():
                ix.wk.normal_(0.0, 0.12, generator=gen)
                ix.wq.normal_(0.0, 0.12, generator=gen)
                for qp in getattr(ix, "q_proj_extra", ()):
                    qp.weight.normal_(0.0, 0.10, generator=gen)

        loop_kw: dict[str, Any] = dict(
            prelude_end=2,
            core_end=5,
            num_passes=3,
            target_virtual_layers=12,
            kv_share=True,
            gate_init=0.35,
            bridge_rank=8,
            step_embedding=True,
            per_pass_router=True,
            router_adapter_rank=8,
            router_repel_init=1.25,
            mor=True,
        )
        if "bridge_mode" in LoopSpec.__dataclass_fields__:
            loop_kw["bridge_mode"] = "subspace"
        if "attn_adapter_rank" in LoopSpec.__dataclass_fields__:
            loop_kw["attn_adapter_rank"] = 8
        loop_h = apply_loop(native_model, LoopSpec(**loop_kw))
        with torch.no_grad():
            loop_h.adapters.gates[0].uniform_(0.25, 0.45, generator=gen)
            loop_h.adapters.gates[1].uniform_(0.30, 0.50, generator=gen)
            if hasattr(loop_h.adapters.bridge, "ups"):
                for u in loop_h.adapters.bridge.ups:
                    u.weight.normal_(0.0, 0.08, generator=gen)
            if getattr(loop_h.adapters, "pass_attn", None) is not None:
                for pdict in loop_h.adapters.pass_attn.layers.values():
                    for proj in pdict.values():
                        proj.up.normal_(0.0, 0.08, generator=gen)
            if loop_h.adapters.pass_router is not None:
                loop_h.adapters.pass_router.up.normal_(0.0, 0.12, generator=gen)
                loop_h.adapters.pass_router.bias.normal_(0.0, 0.08, generator=gen)
            if loop_h.adapters.mor_router is not None:
                loop_h.adapters.mor_router.weight.normal_(0.0, 0.35, generator=gen)
                loop_h.adapters.mor_router.bias.copy_(torch.tensor([0.0, 0.15, 0.05]))

        native_model.eval()
        stack_kw: dict[str, Any] = dict(
            model=native_model,
            tokenizer=tok,
            parts=inspect_model(native_model),
            loop=loop_h,
            lsa=lsa_h,
            load_info={"name": "toy"},
        )
        if "hssd" in Stack.__dataclass_fields__:
            stack_kw["hssd"] = hssd_h
        hssd_stack = Stack(**stack_kw)
        hssd_stack.hssd = hssd_h  # type: ignore[attr-defined]
        native_pack_dir = tmp_path / "native_hssd_v3_pack"
        export_whole_model_pack(hssd_stack, native_pack_dir, max_shard_size_bytes=65536)

        # Verify `LoadedModelPack.hssd` is populated and preserved across re-export
        loaded_hssd = load_whole_model_pack(native_pack_dir)
        assert loaded_hssd.hssd is not None
        reexport_dir = tmp_path / "reexport_hssd_v3_pack"
        export_whole_model_pack(loaded_hssd, reexport_dir, max_shard_size_bytes=65536)
        re_cfg = json.loads((reexport_dir / "config.json").read_text(encoding="utf-8"))
        assert re_cfg["qloop_v3_arch"]["hssd"]["enabled"] is True

        native_refs: dict[int, torch.Tensor] = {}
        with torch.no_grad():
            for r in (1, 2, 3):
                with loaded_hssd.depth(r):
                    native_refs[r] = loaded_hssd(prompt, use_cache=False).logits.detach().clone()

        mod_hssd = _load_standalone_modeling_without_qloop(native_pack_dir, monkeypatch)
        sm_hssd = mod_hssd.Qwen36LoopHSSDForCausalLM.from_pretrained(str(native_pack_dir), strict=True)
        with torch.no_grad():
            for r in (1, 2, 3):
                with sm_hssd.depth(r):
                    out_r = sm_hssd(prompt, use_cache=False).logits
                    assert (out_r - native_refs[r]).abs().max().item() < 1e-5
                    last_r = sm_hssd(prompt, use_cache=False, logits_to_keep=1).logits
                    assert (last_r - native_refs[r][:, -1:, :]).abs().max().item() < 1e-5

    # Branch B: Always verify standalone HSSD + subspace bridge + PassAttn + multi-pass LSA strict loading
    base_stack = _build_full_v3_toy_stack(seed=53, mor=True)
    synth_dir = tmp_path / "synth_v3_loophssd_pack"
    export_whole_model_pack(base_stack, synth_dir, max_shard_size_bytes=65536)

    cfg_json = json.loads((synth_dir / "config.json").read_text(encoding="utf-8"))
    arch = cfg_json["qloop_v3_arch"]
    arch["gdn2"] = {"enabled": False}
    arch["hssd"] = {
        "enabled": True,
        "num_pages": 4,
        "k_pages": 2,
        "rank": 8,
        "chunk_size": 8,
        "window_chunks": 2,
        "gate_logit_cap": 8.0,
        "decay_log_cap": 2.0,
        "kappa_init": 2.0,
        "tau_init": 2.2,
    }
    arch["lsa"]["num_query_passes"] = 3
    arch["loop"]["bridge_mode"] = "subspace"
    arch["loop"]["attn_adapter_rank"] = 8
    arch["loop"]["attn_proj_names"] = ["q_proj", "o_proj"]
    (synth_dir / "config.json").write_text(json.dumps(cfg_json, indent=2), encoding="utf-8")

    # Replace GDN-2 keys with full HSSD v3 + subspace bridge (`downs`/`ups`/`norms`) + `pass_attn` + `q_proj_extra`
    idx_data = json.loads((synth_dir / SAFETENSORS_INDEX_FILE).read_text(encoding="utf-8"))
    all_tensors: dict[str, torch.Tensor] = {}
    for sf in sorted(set(idx_data["weight_map"].values())):
        all_tensors.update(load_safetensors_file(str(synth_dir / sf)))
        (synth_dir / sf).unlink()

    all_tensors = {
        k: v
        for k, v in all_tensors.items()
        if ".gdn2_adapter." not in k and "loop_adapters.bridge." not in k
    }
    hidden, rank, n_v, d_k, n_pages = 48, 8, 4, 12, 4
    half_k = d_k // 2  # 6
    for li in (0, 2, 3):
        pfx = f"model.layers.{li}.linear_attn.hssd_adapter"
        all_tensors[f"{pfx}.down.weight"] = torch.randn(rank, hidden, generator=gen) * 0.08
        all_tensors[f"{pfx}.erase_up.weight"] = torch.randn(n_v * d_k, rank, generator=gen) * 0.08
        all_tensors[f"{pfx}.write_up.weight"] = torch.randn(n_v * d_k, rank, generator=gen) * 0.08
        all_tensors[f"{pfx}.decay_up.weight"] = torch.randn(n_v * half_k, rank, generator=gen) * 0.06
        all_tensors[f"{pfx}.theta_proj.weight"] = torch.randn(n_v * half_k, hidden, generator=gen) * 0.08
        all_tensors[f"{pfx}.surprise_proj.weight"] = torch.randn(n_v, hidden, generator=gen) * 0.08
        all_tensors[f"{pfx}.surprise_proj.bias"] = torch.zeros(n_v)
        all_tensors[f"{pfx}.log_surprise_slope"] = torch.zeros(n_v)
        all_tensors[f"{pfx}.surprise_thresh"] = torch.full((n_v,), 0.85)
        all_tensors[f"{pfx}.write_router.weight"] = torch.randn(n_pages, hidden, generator=gen) * 0.10
        all_tensors[f"{pfx}.write_router.bias"] = torch.zeros(n_pages)
        all_tensors[f"{pfx}.read_router.weight"] = torch.randn(n_pages, n_v * d_k, generator=gen) * 0.10
        all_tensors[f"{pfx}.read_router.bias"] = torch.zeros(n_pages)
        all_tensors[f"{pfx}.log_page_gamma"] = torch.zeros(n_pages)
        all_tensors[f"{pfx}.swa_q_up.weight"] = torch.randn(n_v * d_k, rank, generator=gen) * 0.08
        all_tensors[f"{pfx}.swa_k_up.weight"] = torch.randn(n_v * d_k, rank, generator=gen) * 0.08
        all_tensors[f"{pfx}.swa_v_up.weight"] = torch.randn(n_v * d_k, rank, generator=gen) * 0.08
        all_tensors[f"{pfx}.swa_q_norm"] = torch.ones(d_k)
        all_tensors[f"{pfx}.swa_k_norm"] = torch.ones(d_k)
        all_tensors[f"{pfx}.boltzmann_gate.log_kappa"] = torch.zeros(n_v)
        all_tensors[f"{pfx}.boltzmann_gate.tau"] = torch.full((n_v,), 2.2)

    # ToyAttention: nh=4, hd=16, rot=8 -> din=8, index_dim=16
    for li in (1, 4, 5):
        for p in (0, 1):
            all_tensors[f"model.layers.{li}.self_attn.lsa_indexer.q_proj_extra.{p}.weight"] = (
                torch.randn(4, 8, 16, generator=gen) * 0.10
            )

    all_tensors["loop_adapters.bridge.step_emb"] = torch.randn(2, hidden, generator=gen) * 0.05
    all_tensors["loop_adapters.bridge.mix_logits"] = torch.tensor([[0.2, 0.0, -0.1], [-0.1, 0.1, 0.0]])
    for p in (0, 1):
        all_tensors[f"loop_adapters.bridge.norms.{p}.weight"] = torch.ones(hidden)
        all_tensors[f"loop_adapters.bridge.downs.{p}.weight"] = torch.randn(rank, hidden, generator=gen) * 0.08
        all_tensors[f"loop_adapters.bridge.ups.{p}.weight"] = torch.randn(hidden, rank, generator=gen) * 0.08

    # ToyAttention: q_proj=(2*4*16, 48)=(128, 48) [q + q_gate], o_proj=(48, 64) on core full-attn L4
    all_tensors["loop_adapters.pass_attn.layers.L4.q_proj.down"] = torch.randn(2, rank, 48, generator=gen) * 0.08
    all_tensors["loop_adapters.pass_attn.layers.L4.q_proj.up"] = torch.randn(2, 128, rank, generator=gen) * 0.08
    all_tensors["loop_adapters.pass_attn.layers.L4.o_proj.down"] = torch.randn(2, rank, 64, generator=gen) * 0.08
    all_tensors["loop_adapters.pass_attn.layers.L4.o_proj.up"] = torch.randn(2, 48, rank, generator=gen) * 0.08

    single_shard = "model-00001-of-00001.safetensors"
    save_safetensors_file(all_tensors, str(synth_dir / single_shard))
    new_idx = {
        "metadata": {"total_size": sum(t.numel() * t.element_size() for t in all_tensors.values())},
        "weight_map": {k: single_shard for k in sorted(all_tensors)},
    }
    (synth_dir / SAFETENSORS_INDEX_FILE).write_text(json.dumps(new_idx, indent=2), encoding="utf-8")

    mod_synth = _load_standalone_modeling_without_qloop(synth_dir, monkeypatch)
    sm_synth = mod_synth.Qwen36LoopHSSDForCausalLM.from_pretrained(str(synth_dir), strict=True)
    assert sm_synth.max_passes == 3

    with torch.no_grad():
        with sm_synth.depth(1):
            l1 = sm_synth(prompt, use_cache=False).logits
        with sm_synth.depth(2):
            l2 = sm_synth(prompt, use_cache=False).logits
        with sm_synth.depth(3):
            l3 = sm_synth(prompt, use_cache=False).logits
            l3_last = sm_synth(prompt, use_cache=False, logits_to_keep=1).logits

    assert torch.isfinite(l1).all() and torch.isfinite(l2).all() and torch.isfinite(l3).all()
    assert (l1 - l2).abs().max().item() > 1e-3
    assert (l2 - l3).abs().max().item() > 1e-3
    assert (l3_last - l3[:, -1:, :]).abs().max().item() < 1e-6

    # Removing `loop_adapters.bridge.ups.0.weight` must trigger `strict=True` RuntimeError
    del all_tensors["loop_adapters.bridge.ups.0.weight"]
    save_safetensors_file(all_tensors, str(synth_dir / single_shard))
    new_idx["weight_map"].pop("loop_adapters.bridge.ups.0.weight")
    (synth_dir / SAFETENSORS_INDEX_FILE).write_text(json.dumps(new_idx, indent=2), encoding="utf-8")
    with pytest.raises(RuntimeError, match=r"missing_loop=.*bridge\.ups\.0\.weight"):
        mod_synth.Qwen36LoopHSSDForCausalLM.from_pretrained(str(synth_dir), strict=True)

