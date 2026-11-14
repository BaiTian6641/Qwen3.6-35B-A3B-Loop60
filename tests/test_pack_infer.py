"""Comprehensive tests for Whole Model Pack export, low-disk streaming & OptimizedLoopInferenceEngine."""
from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import os
import shutil
import sys
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
