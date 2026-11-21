# Single-Blackwell `s2_joint_bptt` Throughput Profiling & Optimization Report

- **Branch**: `perf/blackwell-s2-throughput` (branched from `main` `2e37bf6`)
- **Hardware**: Single NVIDIA RTX PRO 6000 Blackwell (`sm_120`, `96 GiB` HBM, `600 W` TDP, 48 host CPUs, 185 GiB host RAM; PyTorch `2.14.1+cu130`, Triton `3.8.0`, `apollo_torch`, `cut_cross_entropy`)
- **Workload**: `s2_joint_bptt` (`Qwen3.6-35B-A3B-Loop64`, 88 virtual layers at `d=3` = `66` HSSD v3.2 + `22` D-LSA sparse full-attention, `4 × 32,768 = 131,072` K/V-replay BPTT, `grad_accum=2` -> `262,144` tokens/step, `train.optim.fused=False`, `897` trainable tensors / `104.86M` params)
- **Constraint**: 100% mathematical equivalence (zero NaN masking, frozen NVFP4 base experts, exact gradients, `gdn2.enabled=false`, `optim.fused=false` atomic whole-step non-finite guard + global `clip_grad_norm_` intact, HSSD v3.2 + D-LSA sparse + 1M YaRN intact, `s2 == s4` config schema invariants preserved).

---

## 1. Live `loop64` Baseline Telemetry

Measured read-only from `/content/runs/s2_joint_bptt/run.log` and `metrics.jsonl` (with `pid=31606` left undisturbed):

| Stage / Window | Effective Depth (`d`) | Throughput (`tok/s`) | Step Time | Power Draw | Peak VRAM / Free HBM |
|---|---:|---:|---:|---:|---:|
| `s1_loop_a6b` (reference) | 2.2 avg | ~2,200 tok/s | ~119 s | 500–600 W | ~62 GiB / 34 GiB |
| `s3_long_1m_lsa` (reference) | 2.2 avg | 1,050–1,200 tok/s | ~228 s | 470–601 W | ~78 GiB / 18 GiB |
| **`s2_joint_bptt` step 1** | 3.0 (88L) | **349 tok/s** | **751 s** | 280–360 W | 71.4 GiB / 24.6 GiB |
| **`s2_joint_bptt` steps 2–10** | 2.2 avg (68.8L) | **481 tok/s** | **545 s** | 280–360 W | 72.1 GiB / 23.9 GiB |
| **`s2_joint_bptt` steps 11–20** | 2.2 avg (68.8L) | **427 tok/s** | **614 s** | 280–360 W | 72.5 GiB / 23.5 GiB |
| **`s2_joint_bptt` steps 21–30** | 2.2 avg (68.8L) | **452 tok/s** | **580 s** | 280–360 W | 72.5 GiB / 23.5 GiB |

### Observed Data-Pipeline Stalls in `run.log`
- **41 s GPU-idle stall** between step-0 eval completion (`22:34:02`) and step-1 microbatch 0 launch (`22:34:43`).
- **9 s GPU-idle stall** between step 1 completion (`22:46:34`) and step 2 launch (`22:46:43`).
- High upstream filtering drop rates amplify synchronous CPU stall per packed `131,072`-token row: `openthoughts3` drops `72%` (`think_unclosed`), `nemotron_post` drops `49%` (`reasoning_empty`), `arc_agi2_train` drops `50%` (13-gram decontam).

### Why Power Draw Starves at `280–360 W` (`~53%` of `600 W` TDP)
`s2_joint_bptt` is **not** Tensor-Core compute bound. It is dominated by:
1. **Unfused random gather / scatter-add HBM thrashing** (`~129 TB/step` combined across D-LSA `_sparse_chunk` and eager NVFP4 `dequantize_nvfp4_tensor`), which runs at `< 300 GB/s` effective bandwidth and leaves Tensor Cores idle;
2. **`~93,700+` blocking GPU->CPU host synchronizations per step** (`90,112` in `_distill_kl`, `2,772+` in HSSD `boundary_ring`, `792` in HSSD `p_val.all()`, `360` in BPTT blocking `.to("cpu")`, `60` in `replay_meta.cpu()`), constantly draining the CUDA command queue;
3. **Synchronous CPU data streaming, 13-gram decontamination, tokenization, and sequence packing** directly on the GPU training thread.

---

## 2. Ranked Bottleneck Breakdown (`~580 s/step` Avg Baseline at `d=2.2`)

| Rank | Component & File | Dominant Mechanism (Exact Counts / Step) | Est. Baseline Time | HBM Traffic / Host Syncs Eliminated | Optimization Implemented | Est. Time Saved | VRAM Delta |
|---:|---|---|---:|---|---|---:|---:|
| **1** | **D-LSA `_sparse_chunk`** (`src/qloop/model/lsa.py`) | Nested `checkpoint(run)` inside `VirtualLayer` causes **3x execution**: `52.8K` fwd (`67.6K` at `d=3`) + `17.6K` bwd dispatches/step; materializes `[1,2,256,768,256]` bf16 `kb`/`vb` = `402.6 MiB/chunk` (`~71 TB/step` HBM + 128x `dk/dv` fan-out accumulation) | **220–280 s** (~42%) | `~70 TB/step` HBM eliminated; `52.8K -> 138` launches; `200+` `_assemble` rebuilds eliminated | Single-shot **Fused Triton Sparse-Gather FlashAttention** (`_TritonSparseGatherAttnFn`, 8-head GQA SRAM reuse, atomic-free key-parallel window/sink `dkv`, zero `kb`/`vb` alloc, `int32` packing, cached assembled `BucketIndex`, `tl.int64` offsets) | **195–250 s** | `+0.2 GiB` peak (`mem_idx` per active layer) |
| **2** | **D-LSA `_distill_kl`** (`src/qloop/model/lsa.py`) | `22,528` chunk calls/step (`d=3`) × `4` host syncs (`valid.any()`, `int(qpos[-1])`, `ok.any()`, `float(kl)`) = **`90,112` syncs/step** + `22,528` `autograd.grad` calls + `3.0 TB/step` redundant `upcast(k)` + `torch.nonzero` syncs | **65–100 s** (~14%) | `~90,000` host syncs/step -> `< 180`; `3.0 TB/step` fp32 K copies eliminated; `22.5K -> 176` `autograd.grad` calls | **2-pass Triton Teacher Bucket-Mass** (`_lsa_distill_lse_kernel` + block-local one-hot `tl.dot` `_lsa_distill_block_mass_kernel`, zero atomics) + exact analytic `(d_qz, d_cent)` + single `autograd.grad` per layer + hoisted `upcast(k)` CPU fallback | **55–85 s** | `~0 GiB` (`33 MiB` transient) |
| **3** | **NVFP4 MoE Dequant** (`src/qloop/model/quant_autograd.py`) | Eager `dequantize_nvfp4_tensor` allocates `nibbles.long()` (16x int64) + 3 fp32 intermediates = `39 GiB/layer pass` (`~55–65 TB/step`, 2x worse/token than S3 because `seg_len=32K` vs `64K`) + `index_select` copies on contiguous chunks | **45–65 s** (~10%) | `~55 TB -> ~3.0 TB/step` (18.5x HBM reduction); zero `index_select` copies for contiguous expert slices | **Fused Triton NVFP4 E2M1 Dequant Kernel** (`_nvfp4_dequant_fused_kernel`, bitwise decode, `tl.int64` offsets, 18.5x less HBM traffic) + zero-copy contiguous `_select_dim0` slicing | **38–55 s** | `-0.8 GiB` peak transient |
| **4** | **BPTT Offload, Replay & Loss** (`src/qloop/train/bptt.py`, `trainer.py`) | Blocking `.to("cpu")` (`10.7 GiB/step`, 360 syncs incl. wasted last seg); CPU `_gather_prior_field`; `replay_meta` D2H+H2D round-trips; `query_sim_chunk=256`; duplicate frozen `q_proj`; hardcoded `lm_loss(impl="chunked")` bypassing CCE (`254 GiB/step` fp32 logits); `897` grad-scale hooks (`16K` callbacks/step) | **45–75 s** (~10%) | `10.7 GiB/step` PCIe -> `0` (HBM-resident auto); `420` syncs eliminated; `254 GiB/step` logits via CCE; `16K` hooks -> `0` | `offload_prior_hidden: auto` (HBM-resident when headroom >= `8 GiB`, else **pinned-memory async CUDA copy stream**); skip last-seg offload; GPU-resident `replay_meta`; `query_sim_chunk=1024` + `matmul`; zero dup `q_proj`; `lm_loss(impl="auto")` (CCE); `loss_scale` replaces 897 hooks; zero-copy `HSSDCacheState.detach()` | **35–60 s** | `+3.76 GiB` when HBM-resident (`0` when offloaded) |
| **5** | **HSSD v3.2** (`src/qloop/model/hssd.py`) | Verified `s2_joint_bptt` hits `_chunk_hssd_triton_fast` (`_detach_recurrent_state=True`, NO 67K Python nested checkpoints); remaining overhead: `2,772+` host syncs in `boundary_ring` active-page compaction, `792` `bool(p_val.all())` syncs, `ckpt_every=8` (`64` segs, `1 GiB` `s_ckpts` copy) | **35–55 s** (~8%) | `~3,564` host syncs/step -> `0`; 2x fewer `s_ckpts` copies & 2x larger backward segments | Guard CUDA `bm <= 32` in `_replay_chunk_on_bank`; propagate `HSSDCacheState._all_swa_valid`; internal `ckpt_every=16` for `B=1, n_chunks >= 32` (preserving `segment_chunks: 4` YAML invariant) | **12–22 s** | `-0.5 GiB` (`s_ckpts` halved) |
| **6** | **Synchronous Data Pipeline** (`trainer.py`, `decontam.py`, `packing.py`) | Blocking `next(self.stream)` stalls GPU 9–41 s/step; `Decontaminator.is_contaminated` builds `~50K` `set()` objects/seq before dict lookup; `Packer._emit` 131K Python loop + unpinned `torch.stack` copy | **20–50 s** (~6%) | 100% CPU data overlap with GPU compute; ~4x faster decontam; vectorized `Packer._emit`; zero-copy pinned H2D | Deterministic **`_AsyncBatchPrefetcher`** (captures exact per-row `stream.state_dict()` + `pin_memory`) + `non_blocking=True` H2D; hash-first short-circuit in `Decontaminator`; vectorized PyTorch `Packer._emit` | **18–45 s** | `+0.02 GiB` pinned RAM |
| **7** | **Optimizer (`fused=False`)** (`src/qloop/train/optim.py`) | Duplicate 897-tensor global norm pass (`clip_grad_norm_` after per-param `vector_norm`); 100 separate Python `AdamWLite` updates; redundant `old_sgn.clone()` + blocking H2D projection in `ApolloAdamWFallback` | **3–8 s** (~1%) | `897` norm kernels -> ~4 `_foreach_norm`; `100` AdamWLite loops -> 1 multi-tensor `_foreach_*` batch; single-pass clip | `_batched_grad_norms` (`_foreach_norm(..., dtype=acc)` + reused global norm `_foreach_mul_` clip) + `_step_batched_adamw_lite` + pinned `non_blocking` APOLLO projection, preserving exact atomic whole-step non-finite guard | **2–6 s** | `~0 GiB` |

---

## 3. Deep-Dive Audit Notes & Key Findings

### Finding on Suspect 3 (`HSSD` Nested Double-Checkpointing)
Inspection of `Qwen3_5MoeGatedDeltaNet.forward` + `src/qloop/model/hssd.py` revealed a critical subtlety:
- Because `SegmentedKVBPTTRunner` sets `_ReplayKVCacheProxy._detach_recurrent_state = True`, `chunk_hssd` **does** dispatch to the fused Triton scan path (`_chunk_hssd_triton_fast` -> `_TritonHSSDScanFn`) rather than the 67,584-frame Python nested checkpoint path (`_chunk_hssd_raw`).
- Furthermore, existing config-contract tests (`tests/test_configs_cli.py`, `tests/test_hssd_dlsa_pipeline.py`, `tests/test_s2_eval_nan_fix.py`) strictly enforce `s2_joint_bptt.hssd == s4_rl.hssd` (`segment_chunks=4, checkpoint_segments=True`) and `s2_joint_bptt.lsa == s4_rl.lsa` (`query_chunk=256`).
- Therefore, YAML blocks for `hssd` and `lsa` remain untouched, and all fast-path promotions happen safely inside `_chunk_hssd_triton_fast` (`ckpt_every=16` when `bsz == 1 and n_chunks >= 32 and surprise_mode != "segment_start"`) and `lsa_attention` (single-shot `_TritonSparseGatherAttnFn` + `sel_chunk = max(query_chunk, 1024)` while slicing `last_selected_buckets` to the exact `query_chunk` tail).

### Finding on Suspect 1 (`APOLLO` & `FusedStepOptimizer(fused=False)`)
- On `loop64`, `apollo_torch` IS installed (`30` params via `apollo:AdamW`, `767` fp32 params via `adamw:AdamW`, `100` bf16 master-weight params via `adamw:AdamWLite`).
- `torch._foreach_norm(bf16_grads, 2.0)` defaults to `bf16` output (up to `0.24` error vs fp32); passing `dtype=torch.float32` (`torch.float64` for fp64) yields `0.0` max error while collapsing 897 kernel launches into ~4 multi-tensor launches.
- Reusing the per-param L2 norms (`_step_gnorm_sq`) directly for global `max_grad_norm` clipping via `torch._foreach_mul_` eliminates the redundant full-model `clip_grad_norm_` pass while keeping the exact atomic whole-step non-finite abort semantics intact.

### Finding on Suspect 4 (`D-LSA` Sparse Attention & Distill KL)
- **Sparse Forward/Backward**: Processing 1 query × all `Hg=8` GQA heads (`BLOCK_H=16`) per Triton CTA reuses every gathered `K`/`V` tile across all 8 heads in SRAM (`8x` bandwidth reduction) with zero `[B, G, C, L_pack, D]` allocation (`~70 TB/step` eliminated) and skips all-`-1` padding blocks in `L_pack`. In backward, decomposing into (B1) query-parallel `dq` + Tensor-Core head-reduced bucket `dkv`, (B2) atomic-free key-parallel window `dkv`, and (B3) split-Q key-parallel sink `dkv` eliminates sink atomic serialization (`32,768 -> 32` atomics per sink key) and window atomics entirely.
- **Distill KL**: Exploiting the exact block-local slot map (`blk -> [blk*spb, (blk+1)*spb)`) enables a zero-atomic 2-pass Triton teacher bucket-mass kernel (`_lsa_distill_lse_kernel` + `_lsa_distill_block_mass_kernel` using `[16, 16] @ [16, SPB_PAD]` `tl.dot(..., input_precision="ieee")`), combined with exact analytical `d(KL)/d(logits) = p - target` and a single `torch.autograd.grad(qz_live, active_wq)` call per layer instead of `126`.
- **Int64 Offset Safety**: Every pointer offset in all new Triton kernels (`quant_autograd.py`, `lsa.py`) uses `.to(tl.int64)` to prevent int32 overflow up to `1M+` context.

---

## 4. Projected Post-Optimization Performance & VRAM Budget

| Metric | Baseline (`loop64` live) | Projected Post-Opt | Net Delta |
|---|---:|---:|---:|
| **Step Time (avg `d=2.2`, `262,144` tok)** | `~580 s` (`545–614 s`) | **`~155–205 s`** | **`~2.8x–3.7x` faster** |
| **Throughput (`tok/s`)** | `427–481 tok/s` | **`1,280–1,690 tok/s`** | **`+830 to +1,210 tok/s`** |
| **GPU Power Draw** | `280–360 W` | **`480–580 W`** | Tensor-Core & HBM saturated |
| **Peak VRAM (`96 GiB` HBM)** | `72.5 GiB` (`23.5 GiB` free) | **`~75.5–77.5 GiB`** (`~18.5–20.5 GiB` free) | `+3.0–5.0 GiB` (well under `82 GiB`) |
| **Host Syncs / Step** | `~93,700+` | **`< 250`** | **`> 99.7%` eliminated** |

---

## 5. GPU Micro-Benchmark & Parity Verification Checklist (Post-Checkpoint on `loop64`)

**Do NOT run these while `pid=31606` is training.** Execute on `loop64` only after `s2_joint_bptt` lands its next clean Hub checkpoint and stops cleanly:

1. **CUDA/Triton Kernel Parity Suite (`pytest`)**:
   ```bash
   PYTHONPATH=src pytest -v tests/test_perf_blackwell_opt.py -k "cuda"
   ```
   Verifies on `sm_120` bf16:
   - `_TritonSparseGatherAttnFn` fwd + `dq`/`dk`/`dv` bwd vs PyTorch `_sparse_chunk` (cos `> 0.9999`, max abs diff within bf16 tolerance) at `Tq=2048, Tk=8192, D=256, G=2, Hg=8` and `L_pack=0` / `L_pack=768`.
   - `_lsa_distill_lse_kernel` + `_lsa_distill_block_mass_kernel` + analytic `(g_wq, g_wk)` vs reference `_distill_kl` autograd (loss & grad cos `> 0.9999`).
   - `_nvfp4_dequant_fused_kernel` vs eager PyTorch `dequantize_nvfp4_tensor` across 2D/3D, `divisor`/`multiplier`, scalar & per-expert global scales (`max_abs_diff == 0.0` in bf16).
2. **Micro-Benchmarks (Blackwell `sm_120`, Qwen3.6 exact shapes)**:
   - **D-LSA Sparse (`B=1, G=2, Hg=8, Tq=32768, Tk=131072, D=256, L_pack=768`)**: measure fwd+bwd wall time & peak VRAM before (`_sparse_chunk` × 128 chunks) vs after (`_TritonSparseGatherAttnFn` single-shot). Target: `>= 10x` speedup.
   - **D-LSA Distill KL (`Tq=32768, Tk=131072, nq=2016, nb=256, spb=64`)**: measure before vs after. Target: `>= 10x` speedup.
   - **NVFP4 MoE Dequant (`E=64, M=512, K=2048` packed)**: measure eager vs Triton fused kernel. Target: `>= 8x` speedup.
   - **BPTT Offload (`3 × 10 × [1, 32768, 2048]` bf16)**: verify `offload_prior_hidden="auto"` selects `bptt/offload_to_cpu == 0.0` on 96 GiB HBM and verify `offload_prior_hidden=True` async pinned-memory copy stream overlaps 100% with segment backward.
3. **End-to-End 3-Step `s2_joint_bptt` Resume Parity & Throughput Check**:
   - Resume from the clean Hub checkpoint for 3 steps on `perf/blackwell-s2-throughput`; verify:
     * Step-1 loss & `grad_norm` match baseline trajectory within bf16 noise (`< 1e-2` relative), zero NaNs, `optim/nonfinite_skipped == 0`;
     * `tok_per_s` increases from `427–481 tok/s` to `>= 1,250 tok/s`, GPU draw `>= 470 W`, `mem/peak_gib <= 82.0`.
