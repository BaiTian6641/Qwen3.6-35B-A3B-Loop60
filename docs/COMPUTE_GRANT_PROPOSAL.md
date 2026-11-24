# Compute Allocation Proposal: `Qwen3.6-35B-A6B-Loop64`
### Parameter-Efficient 88-Layer Looped Hybrid Reasoning & 1M-Context Retrofitting of Frozen NVFP4 MoE Models

> **Public Model Hub:** [`TassP0lar/Qwen3.6-35B-A3B-Loop64`](https://huggingface.co/TassP0lar/Qwen3.6-35B-A3B-Loop64) (`202` live commits) · **Public Dataset:** [`TassP0lar/Loop64-Fable-SIRSI-v1`](https://huggingface.co/datasets/TassP0lar/Loop64-Fable-SIRSI-v1) (`v1.2`, `cc-by-4.0`)  
> **Single-Card Development Hardware:** `1× NVIDIA RTX PRO 6000 Blackwell Server Edition` (`GB202`, `sm_120`, **`96 GB GDDR7 with ECC`** [`94.97 GiB` CUDA-usable], `512-bit` bus, **`1,597 GB/s`** Server Edition peak bandwidth, `24,064` CUDA cores, `600 W` max TDP, passive server cooling)  
> **Full Empirical Technical Reports:** English [`docs/EXPERIMENT_REPORT.md`](EXPERIMENT_REPORT.md) (`docs/papers/Loop64_Technical_Report_EN.pdf`) · Simplified Chinese [`docs/EXPERIMENT_REPORT_ZH.md`](EXPERIMENT_REPORT_ZH.md) (`docs/papers/Loop64_Technical_Report_ZH.pdf`) · Raw CSV/JSON & 7 pure-SVG figures in [`docs/experiment_logs/`](experiment_logs/)  
> **Codebase Readiness:** Open repository (`origin/main` `574898d`), **`760` collected unit/parity/CUDA tests** (incl. **`38/38` Blackwell CUDA GDN-2 pytest** green), `0` lint / Python 3.9 AST errors  
> **Primary Compute Request:** **`~2,500` GPU-hours** (`1× 8×H200/B200` node for ~2 weeks; scalable from **`~400` GPU-h** Tier 1 to **`~6,000` GPU-h** Tier 3)

---

## 1. Research Pitch & Motivation

Scaling test-time reasoning via long chains of thought alone encounters severe KV-cache and serial-token bottlenecks, while training looped or ultra-long-context transformers from scratch requires industrial-scale pretraining budgets. We investigate a sharp, parameter-efficient alternative: **can a frozen `35B` (`A3B`) NVFP4 hybrid MoE backbone (`unsloth/Qwen3.6-35B-A3B-NVFP4` [1], `30` Gated DeltaNet [2] + `10` full-attention layers, `32.212B` routed experts 100% frozen and SHA-256 verified) be retrofitted with `< 0.6%` trainable parameters into a runtime-scalable `88`-virtual-layer looped reasoner supporting `1,048,576`-token (`1M`) context inside `10.65 GiB` of context state?** Our clean v2 architecture combines three orthogonal mechanisms enabled by low-rank APOLLO optimization [3]: (1) a calibrated `[12,36)` ETD sandwich loop with manifold-constrained (`mHC`) `LoopBridge` [4] and per-pass router adapters, inspired by adaptive recursive depth [5], that scale depth dynamically across `40L` (`R=1`), `64L` (`R=2`), and `88L` (`R=3`) by unlocking diverse frozen experts across passes; (2) **Step-0 IEEE-754 bit-identical Gated DeltaNet-2 (`GDN-2-1M`) [6] + Bounded Query-Dependent Erasure (`QED`) [7] + Soft Negative Eigenvalues [8]** mapped 1-to-1 onto `fla 0.6.0` `DPLR` kernels [9] (`0` extra recurrent state bytes) with checkpointed `4×8K` macro-segmenting (`574898d`); and (3) pure `4.0×` YaRN [10] `FP8 1.0× SharedKV` with untied per-pass Latent Sparse Attention (`D-LSA`) [11]. We have already completed **`167.60` active single-GPU hours** (`627.56M` tokens, `202` public Hub commits [12], [13]) of end-to-end engineering, negative-ablation forensics, completed `s3_v2@600` training + `12/12` Gate 2 needle verification, Blackwell `sm_120` Triton kernel optimization, and live `s2_v2` (`GDN-2-1M + QED`) training on one rented GPU; we now seek a focused multi-GPU allocation to complete controlled `s2_v2` ablations, verifiable GRPO/DAPO [14], [15] + SI-RSI (`s4_v2_rl`) training, and a rigorous FLOP-matched `d ∈ {1, 2, 3}` evaluation across 16 frontier 2026 benchmarks.

---

## 2. Systems Foundation & Single-GPU Evidence (Why This Work Is Already De-Risked)

### 2.1 Foundational Enabler: APOLLO Optimizer [3] + `FusedStepOptimizer` on One `96 GB` Card

Training a `34.661B` looped MoE up to `131K–1M` tokens on a single **`96 GB GDDR7 ECC`** card without ZeRO/FSDP is enabled by **APOLLO [3]** integrated into **`FusedStepOptimizer`** (`configs/base.yaml:89-97`, `src/qloop/train/optim.py:382-421`):
- **Exact Routing Rule:** `apollo: {enabled: true, rank: 256, scale_type: channel, scale: 1.0, update_proj_gap: 200, min_dim: 128}` via `apollo_torch.APOLLOAdamW` (with built-in faithful `ApolloAdamWFallback`) routes **every large 2D trainable matrix (`min(shape) >= 128`, not in `adamw_patterns`)** to low-rank `rank=256` SGD-like memory APOLLO, while 1D norms & narrow `<128`-dim adapters use `AdamW`/`AdamWLite` (with fp32 master copies for `bf16` weights).
- **Stage-by-Stage Honest Runtime Record:** `s1_loop_a6b` (`137` adapter/norm tensors, `18.41M` params) ran `fused=True` in-backward freeing `.grad` immediately (`adamw:AdamWLite x137`, `runs/s1_loop_a6b_run.log:57-60`). Starting at commit `3a17c0d`, `s1b_hssd_distill` / `s3*` / `s2*` / `s4*` switched to `fused=False` for an atomic whole-step non-finite guard + true global `clip_grad_norm_` while keeping `apollo.enabled: true`. In both **`s2_joint_bptt`** (`187.92M` / `897` tensors) and live **`s2_v2_joint_bptt`** (`156.94M` / `387` registered tensors, `379` stepped in Gate 3), `apollo_torch` actively runs **APOLLO (`rank=256`, channel-wise) on all `30` large 2D full-attention matrices (`k_proj`, `v_proj`, `o_proj` across `10L` = `104.86M` params; vs `0.781 GiB` AdamW, `0.156 GiB` moments-only [`5.0×`, `-0.625 GiB`; `o_proj` `8.0×`] or `0.186 GiB` incl. `+0.029 GiB` fp32 proj [`4.2×` net, `-0.60 GiB` net; `o_proj` `6.4×` net])** alongside `257` fp32 `AdamW` + `100` bf16 `AdamWLite` tensors — and in `s4_v2_rl`, APOLLO covers both `10` `o_proj` + `120` `shared_expert` 2D matrices (`209.72M` params; vs `1.563 GiB` AdamW, `0.547 GiB` moments-only [`2.86×`, `-1.02 GiB`] or `0.625 GiB` incl. `+0.078 GiB` fp32 proj [`2.50×` net, `-0.94 GiB` net]).
- **Blackwell Vectorization:** `_foreach_norm(..., dtype=fp32)` collapses `897 → ~4` norm launches with `0.0` error (avoiding PyTorch's `0.24` bf16 error), reuses `_step_gnorm_sq` via `_foreach_mul_` for single-pass global clipping, batches `100` `AdamWLite` updates, and pins `non_blocking` APOLLO projection transfers.

### 2.2 De-Risked Empirical Evidence (`167.60` Active Single-GPU Hours, `627.56M` Tokens)

Every architectural, kernel, VRAM-headroom, numerical-stability, and data-hygiene risk has been resolved on **`1× NVIDIA RTX PRO 6000 Blackwell Server Edition (96 GB GDDR7 ECC, 1,597 GB/s)`**. All entries distinguish `Measured`, `Derived Formula`, and `Estimated (derived from measured rates)`; zero unmeasured downstream 2026 benchmark scores are claimed.

| Research Pillar | Class | Verified Evidence (Single Rented Blackwell GPU) | Reproducible Artifact |
| :--- | :---: | :--- | :--- |
| **Zero-Training Calibration (`s0_probe`)** | `Measured` | **`5/5`** gates passed in `177.44 s`; `1.3061` nats expert headroom (`88.9%` hard tokens); `[12,36)` core cuts zero-shot shock `3.26×`; pure `YaRN 4.0×` [10] wins `1M` extrap (`100%` needle) | `artifacts/probes/probe_report.json` (`058b4ad3`) |
| **88L Loop Healing (`s1_loop_a6b@1000`)** | `Measured` | **`18.41M` params (`0.053%`)**, `0` skips, `54.24 GiB` peak; `8K` NLL `0.9800 → 0.9191` (`depth_gain = +0.0609`); **`82.27%`** Pass-2 re-routing (`17.36` unique core experts/tok); `9/9` cache parity & `16K` needle **PASS** | Hub `32eb565e`, `s1_quality_check.json` |
| **Completed Clean v2 A/B Win (`s3_v2@600`) & Gate 2** | `Measured` | **`600/600` completed (`15.02 h`, `24.56M` params, `0` skips)**: held-out `64K` NLL **`1.3689 → 1.2996 → 1.2816`** (`depth_gain = +0.0873`, **`-0.5941` nats / `44.8%` lower PPL** vs HSSD `s3@600` `1.8757`); `512K–1M` recall **`0.6514` peak / `0.5715` final** (vs `0.4848` HSSD); **`2,365 tok/s`** (`2.25×`); peak **`69.96 GiB`** (`69.52 GiB` live **`1,048,576`-tok** pass @ `s125`); organic Pass-3 mHC gate **`+59.4%`** (`0.0325 → 0.0517`). **Gate 2:** **`12/12` SHARP**, **`7/7` (`100%`) `18K`/`64K` needle hits**, `0` loops, `0` tool poisoning, `0.0` slop score; sparse `64K` prefill `4,279 tok/s` (`26.74 GiB`) | Hub `6b521f47`, `ec34f825`, `s3v2_gate_summary.json` |
| **Gate 1 + Gate 3 + Live `s2_v2` (`GDN-2-1M + QED`)** | `Measured` | **`27.53M` GDN-2 params (`0.079%`, `47.1%` of HSSD)**, `0.0` Step-0 parity, `0` extra state bytes. **Gate 1 (`574898d`):** **`38/38` CUDA pytest green**; fused `tl.int64` Triton prep **`53.55 ms`** (`2.00×` faster than `106.93 ms` PyTorch prep) & `T=1` **`0.0787 ms`** (`2.41×`); checkpointed `4×8K` macro-segmenting cuts full-wrapper peak VRAM delta **`3.28×`** (**`23.051 → 7.032 GiB`**, `-16.02 GiB`). **Gate 3 (`4×32K`, `d=3`):** **`67.194 GiB` peak** (`27.6 GiB` alloc, `0` NaNs/skips, frozen experts SHA-256 verified). **Live `s2_v2` (`s100/600`, `7.56 h`):** **`1,040–1,173 tok/s`**, **`67.77 GiB` peak**, `0` skips; `adapter_norm` `0.0369 → 0.3884`, `horizon_mult` `1.014× → 1.544×`, QED `abs(λ)` `4.66e-5 → 3.55e-4` | `s3v2_gate_summary.json`, `s2_v2_joint_bptt.csv`, Hub `d0a1d1d0`, `574898d` |
| **1M Inference & Training VRAM Ledger** | `Derived Formula` + `Measured` | **1M Inference:** **`10.65 GiB`** FP8 SharedKV (`10.45 GiB` w/ raw `0.133 GiB` GDN; **`3.02×` smaller** than Inkling 2 Mini [16] `32.16 GiB`; `5.96–6.28 GiB` Hadamard INT4 [17]); `lsa.py:743-768` `BucketIndex` adds `+1.04 GiB` eager `int64` (`~33.54 GiB` total w/ `21.85 GiB` NVFP4+adapters, compactable to `~0.04 GiB` → `~32.5 GiB`). **Measured Training Peaks (`96 GB` GDDR7 ECC):** `54.24` (`s1`), `68.97` (`s1b`), `81.23` (`s3`), `78.42/70.1` (`s2`), `69.96` (`s3_v2`), **`67.19–67.77 GiB`** (live `s2_v2` `4×32K d=3` on `574898d`, leaving **`27.2 GiB`** headroom) | `src/qloop/model/lsa.py:743-768`, `EXPERIMENT_REPORT.md` §II-C–D |
| **Audited Dataset (`v1.2`) & SI-RSI** | `Measured` | `4,081` `sft_core` + `429` pref pairs [13], [18], [19], [20]; **`1,889,432`** live 2026 13-grams (`31` leaks blocked across 12 suites); decoupled `working_notebook` CoT raised pass rate **`4.1% → 100.0%`** (`197/197`, `0` defects); surgical `[loop_adapters, router]` SI-RSI unit-tested | `dataset_audit_summary.json`, `def3624`, `dc7a14e` |
| **Blackwell `sm_120` Speedup Chronology** | `Measured` | Diagnosed `280–360 W` (`~53%` TDP) starvation (`~129 TB/step` unfused GDDR7 thrashing + `93,700+` host syncs + `9–41 s/step` CPU stalls); 7 profiled opts (**`18.99×`** NVFP4 E2M1 reaching **`1,148 GB/s = 71.9%`** of `1,597 GB/s` GDDR7 peak, **`9.32×/12.99×`** D-LSA FlashAttn `-70 TB/step`, **`5.83×`** 2-pass KL `-90K` syncs, **`54.03×`** BPTT offload/DMA+CCE [21], HSSD sync removal, `_AsyncBatchPrefetcher`, `_foreach_norm` APOLLO/AdamW) + 5 hardening commits (`2e37bf6` `tl.int64`, `5a82677` `-7.0 GiB` leak fix, `2f5a6ba` `10.6×` TF 5.24 recovery, `2c78ffc` fused Triton QED+DPLR prep, `574898d` `4×8K` `3.28×` VRAM cut); **`760` collected tests** | `docs/PERF_PROFILE_S2.md`, `EXPERIMENT_REPORT.md` §VI |

---

## 3. HSSD v3.2 Post-Mortem & Lessons From Honest Negative Ablations

A central differentiator of our engineering methodology is catching silent failure modes early and converting negative ablations into principled architectural upgrades:

1. **HSSD v3.2 Post-Mortem → Why `GDN-2-1M` [6] + Bounded QED [7] via `DPLR` [9] Is the Principled Move:** Our v1 pipeline replaced stock GDN with custom HSSD v3.2 (`[513,576]` SWA + `M=8` log-spaced `Top-2` STE surprise-gated pages + low-rank complex phase). Across `s1b → s3 → s2@500`, logs exposed every failure mode: (i) an $O(T)$ cumulative-phase Jacobian blowup (`6.12e+19` @ `s475`, peak loss `12.0742` @ `s650`, `194,510` total skips across `s1b` steps `425–775`) fixed numerically via `chunk_size/T` scaling (`a0877a3`/`3a17c0d` → `0 / 1,100` skips); (ii) an irreversible **`+0.79` nat non-identity representation shock** (`0.919 → 1.705`); (iii) collapsed long-horizon vault pages (`vault_duty` fell from `1.81e-4` at `s1b` `s400` to `1.28e-4` peak @ `s43`, `1.14e-6` min @ `s425`, and `3.26e-5` @ `s500` in `s2` despite `distill_kl = 0.72`); (iv) failed `18K` needle retrieval **even under `LSA=dense`** (`truth_check_summary.json`, while `s1_loop_a6b` passed `16K` needle at `d=1/3` and `top1 = 1.0000` ruled out decode bugs); and (v) **`+5.17 GiB` 1M state overhead & `3.41 tok/s` `d=3` decode (`5.6×` slower than stock `19.22 tok/s`)**. Scientific lessons: enforce mandatory `0.0` Step-0 bit-identity, never rely on NLL without generative/needle truth-checks (Gate 2 on `s3_v2@600` immediately proved `12/12` sharp & `7/7` `18K/64K` needle recovery!), audit datasets before mixing, and generalize mature pretrained DeltaNet manifolds onto production `fla 0.6.0` `DPLR` kernels [9] — yielding **`GDN-2-1M + Bounded QED`** (`0.0` Step-0 diff, `0` extra state bytes, `0.0787 ms` fused Triton `T=1` decode, `47.1%` of HSSD params, `3.28×` VRAM-cut `4×8K` training).
2. **Quarantining Upstream Tool-Call Poisoning (`fable51`) & Eliminating Eval Contamination:** Forensics caught `s1b` `d=3` emitting literal `{"query": "<search_query>"}` loops, traced to `100%` placeholder-redacted `tool_calls` in raw upstream `fable51` (`<bash_command>` `6,098×`, `<file_path>` `5,972×`, `<search_query>` `2,441×`). Auditing all sources removed `2,107` duplicates, `2` non-permissive repos, `247` poisoned rows, and `31` live 2026 benchmark leaks, while decoupled `working_notebook` synthesis lifted reasoning-trace quality from `4.1%` to `100.0%` (`197/197`).
3. **Transparent Unmeasured Scope:** With `s3_v2@600` completed and Gate 1/2/3 passed, `s2_v2_joint_bptt` is currently live at step `100/600` (`7.56 h` active GPU-h, `0` skips) and `s4_v2_rl` is queued behind it; **we make zero claims on downstream 2026 frontier benchmarks yet**. Measuring those gains — and running the decisive ablations — is the exact purpose of the allocation below.

---

## 4. Concrete Three-Tier Compute Allocation Request *(`Estimated (derived from measured rates)`)*

All budgets below are **forward compute estimates derived explicitly from our measured single-GPU rates** on `1×` Blackwell (`96 GB GDDR7 ECC`):
- **Long-context `131K` K/V-replay BPTT (`s2_v2`):** Live measured `s2_v2` throughput on commit `574898d` (`4×32K`, `4×8K` macro-segmented `GDN-2+QED+DPLR`) is **`1,040–1,173 tok/s`** (`~3.7–4.2 min/step` @ `262,144` tok/step, **`67.77 GiB` peak**); a full `600`-step run (`157.3M` tok) requires **`~40–45` GPU-hours** (`~5.5` wall-hours on `8×` GPUs).
- **Verifiable GRPO/DAPO [14], [15] + SI-RSI (`s4_v2_rl`):** `400` steps × `8` prompts × `group=8` × `~4K–8K` generated tokens (`~105M–210M` generated tokens at `88L` `d=3`). At batched H200/B200 decode (`~250–500` aggregate tok/s/GPU), one `400`-step RL run requires **`~150–250` GPU-hours** (`~20–30` wall-hours on `8×` GPUs).
- **Full 16-Benchmark 2026 Depth Sweep (`d ∈ {1, 2, 3}`):** Evaluating `HLE-Diamond/Verified`, `GPQA/SuperGPQA`, `AIME 2026`, `HMMT Feb 2026`, `ARC-AGI-2` [20], `MRCR 1M`, `LongBench v2`, `KernelBench L3/L4`, `Terminal-Bench 4.1`, and `SWE-bench Pro Hard` across `3` depths + FLOP-matched `d=1` CoT requires **`~150–250` GPU-hours** per model configuration.

| Allocation Tier | Total GPU-Hours | Target Hardware / Wall Time | Funded Experiments & Scientific Scope |
| :--- | :---: | :---: | :--- |
| **Tier 1: Core Completion** | **`~400` GPU-h** | `1×–4×` H200/B200 (`4–7` days) | Complete `s2_v2` `GDN-2-1M+QED` + eval gate (`50 h`) · Focused `s4_v2_rl` (`DAPO-Math` + `KernelBench`, `200 h`) · Core 2026 depth sweep (`AIME 2026`, `GPQA`, `MRCR 1M`, `KernelBench L3`, `150 h`). |
| **Tier 2: Full Conference Paper *(Primary Recommendation)*** | **`~2,500` GPU-h** | **`1× 8×H200/B200` node (`~2` weeks)** | Full `s2_v2` (`600–1,200` steps) + multi-round `s4_v2_rl` incl. agentic rollouts (`700 h`) · **3 controlled ablations (`850 h`)**: (i) `GDN-2-1M+QED` vs. Native GDN (`200 h`), (ii) FLOP-matched `88L` loop vs. `40L` extended CoT (`350 h`), (iii) surgical `[loop_adapters, router]` SI-RSI vs. standard GRPO + D-LSA `index_dim` `64 vs 96` (`300 h`) · **Complete 16-benchmark 2026 `d ∈ {1, 2, 3}` sweep** (`750 h`) · Serving/forensics buffer (`200 h`). |
| **Tier 3: Scaling & Systems Extension** | **`~6,000` GPU-h** | `1×–2× 8×H200/B200` nodes (`~4` weeks) | All of Tier 2 (`2,500 h`) · Multi-seed variance bars + recurrence sweep `R ∈ {1, 2, 3, 4}` (`40L/64L/88L/112L`, `1,800 h`) · `1M–10M` Block-64 Hadamard INT4 [17] serving + Pass-1→Pass-3 self-speculative decoding (`900 h`) · Extended multi-turn agentic RL & Whole Model Pack release QA (`800 h`). |

---

## 5. Expected Research Outputs & Execution Readiness

- **Publication Target:** One full-length conference paper (**NeurIPS / ICML / ICLR / MLSys**) answering three open questions with complete positive and negative ablations: (1) whether `< 0.6%`-parameter looped retrofits over frozen NVFP4 MoE experts beat FLOP-matched extended CoT on hard reasoning and `1M`-context retrieval; (2) whether Step-0 bit-identical `GDN-2-1M + Bounded QED` via `DPLR` closes linear-attention long-range recall gaps without catastrophic distribution shift; and (3) whether surgical token-level unlikelihood (`SI-RSI`) stabilizes looped RL rollouts without degrading attention or base capabilities.
- **100% Open Community Artifacts:** (1) Full open weights & adapters on Hugging Face (`TassP0lar/Qwen3.6-35B-A3B-Loop64` [12]); (2) live decontaminated dataset [`TassP0lar/Loop64-Fable-SIRSI-v1`](https://huggingface.co/datasets/TassP0lar/Loop64-Fable-SIRSI-v1) [13] (`cc-by-4.0`); (3) standalone Blackwell `sm_120` Triton kernels (`NVFP4 E2M1` MoE at `71.9%` GDDR7 peak, D-LSA sparse FlashAttention + 2-pass distill KL, fused `tl.int64` QED+DPLR prep with `4×8K` `3.28×` VRAM-cut checkpointing); and (4) reproducible telemetry export, EN/ZH US-Letter technical reports, & 13-gram-decontaminated evaluation harness.
- **Day-1 Execution Readiness:** The codebase (`574898d`) is self-contained, verified by `760` collected tests (`38/38` Blackwell CUDA GDN-2 tests green), logs directly to public HF Hub manifests with atomic non-finite rollback guards, and requires zero infrastructure build-out before launching Tier-1 or Tier-2 runs immediately upon allocation.

---

## References

[1] Qwen Team, "Qwen3.6-35B-A3B," Alibaba Cloud, Model Release, Apr. 16, 2026. [Online]. Available: https://huggingface.co/Qwen/Qwen3.6-35B-A3B

[2] S. Yang, J. Kautz, and A. Hatamizadeh, "Gated Delta Networks: Improving Mamba2 with Delta Rule," in *Proc. Int. Conf. Learn. Represent. (ICLR)*, 2025, arXiv:2412.06464.

[3] H. Zhu, Z. Zhang, W. Cong, X. Liu, S. Park, V. Chandra, B. Long, D. Z. Pan, Z. Wang, and J. Lee, "APOLLO: SGD-like Memory, AdamW-level Performance," *arXiv preprint arXiv:2412.05270*, Dec. 2024.

[4] Z. Xie *et al.* (DeepSeek-AI), "mHC: Manifold-Constrained Hyper-Connections," *arXiv preprint arXiv:2512.24880*, Dec. 2025.

[5] S. Bae, Y. Kim, R. Bayat, S. Kim, J. Ha, T. Schuster, A. Fisch, H. Harutyunyan, Z. Ji, A. Courville, and S.-Y. Yun, "Mixture-of-Recursions: Learning Dynamic Recursive Depths for Adaptive Token-Level Computation," *arXiv preprint arXiv:2507.10524*, Jul. 2025.

[6] A. Hatamizadeh, Y. Choi, and J. Kautz, "Gated DeltaNet-2: Decoupling Erase and Write in Linear Attention," *arXiv preprint arXiv:2605.22791*, May 2026.

[7] D. Gupta, A. Das, and D. Gupta, "The Query Knows What to Forget: A Second Erase Direction for Linear Attention," *arXiv preprint arXiv:2608.13668*, Aug. 2026.

[8] R. Grazzi, J. Siems, A. Zela, J. K. H. Franke, F. Hutter, and M. Pontil, "Unlocking State-Tracking in Linear RNNs Through Negative Eigenvalues," in *Proc. Int. Conf. Learn. Represent. (ICLR)*, 2025, arXiv:2411.12537.

[9] S. Yang and Y. Zhang, "FLA: A Triton-Based Library for Hardware-Efficient Implementations of Linear Attention Mechanisms," GitHub Repository (fla-org/flash-linear-attention, v0.6.0), 2024–2026.

[10] B. Peng, J. Quesnelle, H. Fan, and E. Shippole, "YaRN: Efficient Context Window Extension of Large Language Models," in *Proc. Int. Conf. Learn. Represent. (ICLR)*, 2024, arXiv:2309.00071.

[11] A. Kubicek, A. Caciolai, L. Peng, S. Sumant, B. Hoover, F. Greenlee, R. M. Battleday, R. Urban, T. Hoefler, and J. C. R. Whittington, "Latent Sparse Attention: Towards Subquadratic Ultra-Long Context Attention," *arXiv preprint arXiv:2610.42147*, Oct. 2026.

[12] TassP0lar, "Qwen3.6-35B-A3B-Loop64: Open Looped Hybrid Checkpoints and Empirical Telemetry," Hugging Face Model Hub, Nov. 2026. [Online]. Available: https://huggingface.co/TassP0lar/Qwen3.6-35B-A3B-Loop64

[13] TassP0lar, "Loop64-Fable-SIRSI-v1: Curated Anti-Slop 3-Beat Looped-CoT, Aesthetic Craft, and SI-RSI Recovery Dataset," Hugging Face Datasets Hub (v1.2), Nov. 2026. [Online]. Available: https://huggingface.co/datasets/TassP0lar/Loop64-Fable-SIRSI-v1

[14] Z. Shao, P. Wang, Q. Zhu, R. Xu, J. Song, X. Bi, H. Zhang, M. Zhang, Y. K. Li, Y. Wu, and D. Guo, "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models," *arXiv preprint arXiv:2402.03300*, Feb. 2024.

[15] Q. Yu, Z. Zhang, R. Zhu, Y. Yuan, *et al.*, "DAPO: An Open-Source LLM Reinforcement Learning System at Scale," *arXiv preprint arXiv:2503.14476*, Mar. 2025.

[16] Thinking Machines Lab, "Inkling 2 Mini," Technical Release, Nov. 20, 2026. [Online]. Available: https://thinkingmachines.ai

[17] A. Zandieh, M. Daliri, M. Hadian, and V. Mirrokni, "TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate," in *Proc. Int. Conf. Learn. Represent. (ICLR)*, 2026, arXiv:2504.19874.

[18] E. Guha *et al.*, "OpenThoughts: Data Recipes for Reasoning Models," *arXiv preprint arXiv:2506.04178*, Jun. 2025.

[19] A. Bercovich *et al.*, "Llama-Nemotron: Efficient Reasoning Models," *arXiv preprint arXiv:2505.00949*, May 2025.

[20] F. Chollet, M. Knoop, G. Kamradt, B. Landers, and H. Pinkard, "ARC-AGI-2: A New Challenge for Frontier AI Reasoning Systems," *arXiv preprint arXiv:2505.11831*, May 2025.

[21] E. Wijmans, B. Huval, A. Hertzberg, V. Koltun, and P. Krähenbühl, "Cut Your Losses in Large-Vocabulary Language Models," in *Proc. Int. Conf. Learn. Represent. (ICLR)*, 2025, arXiv:2411.09009.
