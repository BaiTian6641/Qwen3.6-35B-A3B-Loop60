# `Qwen3.6-35B-A3B-Loop64` — Clean v2 Looped Hybrid MoE & `1M`-Context Backbone (`40L / 64L / 88L`)

Retrofits **`Qwen/Qwen3.6-35B-A3B`** (`40` physical layers = `10 × [3 Gated DeltaNet + 1 Gated Full Attention]`, `256` routed experts/layer, `34.661B` stored decoder parameters with `32.212B` routed MoE weights 100% frozen and SHA-256 verified) into a **Clean v2 Looped Hybrid MoE Backbone** with a calibrated **`[12, 36)` core** (`Prelude [0,12)×1 + Core [12,36)×R + Coda [36,40)×1` $\to$ `40L / 64L / 88L` virtual layers), **`GDN-2-1M + Bounded QED`** over `fla 0.6.0` DPLR kernels, **`1M` `FP8` `1.0×` `SharedKV` + per-pass `D-LSA`**, and low-rank **`APOLLO`** optimization on a single **`NVIDIA RTX PRO 6000 Blackwell Server Edition`** (`96 GB GDDR7 ECC` [`94.97 GiB` usable], `1,597 GB/s`, `sm_120`). Stage checkpoints sync to **[`TassP0lar/Qwen3.6-35B-A3B-Loop64`](https://huggingface.co/TassP0lar/Qwen3.6-35B-A3B-Loop64)** (with Whole Model Pack export targeting **[`TassP0lar/Qwen3.6-35B-A6B-LoopHSSD-Pack`](https://huggingface.co/TassP0lar/Qwen3.6-35B-A6B-LoopHSSD-Pack)** after Sprint 2B validation gates and live `QL60-002` `4/4 PASS` re-verification complete). **`HSSD v3.1/v3.2`** is explicitly retained in-repo solely as a **Retired Negative Ablation** (`docs/EXPERIMENT_REPORT.md §III-E`).

### Current Status & Tracked Documentation

> **Live Clean v2 Status (`1× NVIDIA RTX PRO 6000 Blackwell Server Edition`, `96 GB GDDR7 ECC` [`94.97 GiB` usable], `1,597 GB/s`, `sm_120`):** `s1_loop_a6b@1000`, `s3_v2_long_1m_lsa@600`, and **`s2_v2_joint_bptt@600`** (`GDN-2-1M`, `4×8K` `fla 0.6.0` DPLR, Hub commit `17382c2e166d`, `600/600` steps, `157.29M` tokens, `156.94M` trainable params, `69.97 GiB` peak VRAM, `U = 18.19` [`1.764×`, `3.716B` unique consulted params/tok]) are **completed**. **Sprint 1 (`Tracks A–D`) is 100% merged on `main`**, and **Sprint 2A Blackwell GPU validation** verified `s2_v2@600` against Stock `Qwen3.6 + YaRN 4×` on leak-free `0/4`-overlap `4K` holdout (`-10.71%` PPL, `+469σ`), unseen `wikitext-103-raw-v1` OOD (`-1.11%` PPL, `+33.6σ`, while `d=1` slightly edged `d=3`), `128K` Dense/Sparse NLL (`0.00746 / 0.00796`, `13.4×` lower than Stock, `+0.00050` sparse–dense gap), `8K–128K` 4-needle RULER (**`12/12 = 100%`** vs. Stock `2/12`), causal ablations (`GDN-2` `+156.6σ` & `PassRouter` `+6.8σ` `KEEP_CRITICAL`; `QED` `-0.31σ` & `neg_eigval` `-0.97σ` `PRUNE_CANDIDATE`), `K0` (`YELLOW_B_BLOCK_ATTN_CALIB`, `100.0%` EM retention at `6.05×` smaller KV), and live `transformers 5.24.1` `QL60-001` (`exact 1.00×` KV) + `QL60-005` (`13/13`). **`s4_v2_rl`, `LSA-NL`/`CoMem` training expansion, and public pack release remain frozen** pending **Sprint 2B (`S2.2` `~18.41M` `d=1` LoRA control)**, `QL60-002` `cache` param `4/4 PASS` re-verification, and `512K / 1M` ladder closeout (see [`docs/MIDWAY_REVIEW_AND_ACTION_PLAN.md`](docs/MIDWAY_REVIEW_AND_ACTION_PLAN.md)).

| Document | Role & Scope | Status |
| :--- | :--- | :---: |
| **[`docs/MIDWAY_REVIEW_AND_ACTION_PLAN.md`](docs/MIDWAY_REVIEW_AND_ACTION_PLAN.md)** | **Mid-Way External Audit Cross-Verification Ledger & 2-Sprint Action Plan (`v1.1`):** Sprint 1 `Tracks A–D` 100% merged ledger (`QL60-001..010`, `A1–D5`), completed `s2_v2@600` (`17382c2e166d`), full **Sprint 2A Blackwell GPU empirical results (`§7`: `S2.1` `4K–128K`, `S2.3`, `K0`, live TF `5.24.1`)**, Canonical Parameter Ledger, data provenance policy, & Sprint 2B freeze gates | **Active v1.1** |
| **[`docs/MIDWAY_REVIEW_AND_ACTION_PLAN_ZH.md`](docs/MIDWAY_REVIEW_AND_ACTION_PLAN_ZH.md)** | Mid-Way External Audit Cross-Verification Ledger & 2-Sprint Action Plan (`v1.1`, exact 1-to-1 Simplified Chinese companion with identical IEEE citation parity) | **Active v1.1** |
| **[`docs/EXPERIMENT_REPORT.md`](docs/EXPERIMENT_REPORT.md)** | Full Empirical Technical Report (English): `167.60` GPU-h telemetry, Retired HSSD v3.2 negative ablation, completed `s3_v2@600` + Gate 1/2/3, live `s2_v2` (`GDN-2-1M + QED`), Blackwell `sm_120` kernels, & IEEE refs | **Live v1.4** |
| **[`docs/EXPERIMENT_REPORT_ZH.md`](docs/EXPERIMENT_REPORT_ZH.md)** | Full Empirical Technical Report (Simplified Chinese mirror of `EXPERIMENT_REPORT.md`) | **Live v1.4** |
| **[`docs/PROPOSAL_LSA_NESTED_LEARNING.md`](docs/PROPOSAL_LSA_NESTED_LEARNING.md)** | Next-Gen Research Proposal (`v2.0`, *frozen pending Sprint 2B + `K0` `GREEN` calibration*): **Latent-Sparse Nested Learning (`LSA-NL`)** 5-Frequency CMS (`L0 → L4`), Split-Horizon `3M–4M` (`100M+` cold SSD) + `10B`-token funnel, 6 Original-LSA upgrades, 3-Axis Dynamic Compute, `<= 24 GB` profiles, `K0–K4` | **Frozen v2.0** |
| **[`docs/PROPOSAL_LSA_NESTED_LEARNING_ZH.md`](docs/PROPOSAL_LSA_NESTED_LEARNING_ZH.md)** | Next-Gen Research Proposal (`v2.0`, Simplified Chinese companion with 1-to-1 math, tables, Mermaid & `[1]..[50]` IEEE citation parity) | **Frozen v2.0** |
| **[`docs/ARCHITECTURE_NEXTGEN_LOOP64_LSANL.md`](docs/ARCHITECTURE_NEXTGEN_LOOP64_LSANL.md)** | Component-by-Component Architecture & Implementation Spec (`v1.0`, *frozen pending Sprint 2B + `K0` `GREEN`*): `C0–C6` diagrams, `src/qloop/` anchors, `<= 24 GB` serving ledger, & `K0–K4` build order | **Frozen v1.0** |
| **[`docs/ARCHITECTURE_NEXTGEN_LOOP64_LSANL_ZH.md`](docs/ARCHITECTURE_NEXTGEN_LOOP64_LSANL_ZH.md)** | Component-by-Component Architecture & Implementation Spec (`v1.0`, Simplified Chinese companion with exact 1-to-1 diagrams, tables & `[1]..[52]` IEEE citation parity) | **Frozen v1.0** |
| **[`docs/ROADMAP_NEXTGEN_LOOP64_COMEM.md`](docs/ROADMAP_NEXTGEN_LOOP64_COMEM.md)** | Next-Gen **`Qwen3.6-64L-CoMem` (`LSA-NL` v2.0)** Phased Execution Plan (*gated by Sprint 2B*), `Inkling 2 Mini` comparison, `Qwen3.6`/`Qwen3.8` tokenizer parity audit ([`scripts/verify_q36_q38_tokenizer_parity.py`](scripts/verify_q36_q38_tokenizer_parity.py)), & 3-Paper Ladder | **Gated v2.0** |
| **[`docs/ROADMAP_NEXTGEN_LOOP64_COMEM_ZH.md`](docs/ROADMAP_NEXTGEN_LOOP64_COMEM_ZH.md)** | Next-Gen **`Qwen3.6-64L-CoMem` (`LSA-NL` v2.0)** Phased Execution Plan (Simplified Chinese companion with exact 1-to-1 tables, Mermaid & `[1]..[43]` IEEE citation parity) | **Gated v2.0** |
| **[`docs/COMPUTE_GRANT_PROPOSAL.md`](docs/COMPUTE_GRANT_PROPOSAL.md)** | Professor / Lab Cooperation & Multi-GPU Compute Allocation Proposal (`v2.0`, `400 / 2,500 / 6,000` GPU-h tiers with `K0–K4` kill-gates & 3-Paper Ladder; update post-Sprint 2) | **Draft v2.0** |
| **[`docs/COMPUTE_GRANT_PROPOSAL_ZH.md`](docs/COMPUTE_GRANT_PROPOSAL_ZH.md)** | Professor / Lab Cooperation & Multi-GPU Compute Allocation Proposal (`v2.0`, Simplified Chinese companion with 1-to-1 numerical & `[1]..[54]` IEEE citation parity) | **Draft v2.0** |
| **[`docs/papers/`](docs/papers/)** | US-Letter IEEE PDFs: [`Loop64_Technical_Report_EN.pdf`](docs/papers/Loop64_Technical_Report_EN.pdf) · [`Loop64_Technical_Report_ZH.pdf`](docs/papers/Loop64_Technical_Report_ZH.pdf) | **Snapshot v1.4** |

### Canonical Parameter Accounting (`Qwen3.6-35B-A3B-Loop64`, `[12, 36)` Core)

To prevent conflating stored capacity, unique parameters consulted per token, and compute-equivalent parameter applications across looped passes, every report and card in this repository uses the exact closed-form ledger below (`expert = 3 × 2048 × 512 = 3,145,728`; `40 × 256 = 32.212B` routed; `40 × 1 = 0.126B` shared; `routed + shared = 32.338B`; `dense_unique ex-input-embed/norms = 1.9396B`):

| Accounting Metric | Definition / Regime | Parameter Count | Notes & Caveats |
| :--- | :--- | :---: | :--- |
| **Stored Base Parameters** | Total `Qwen3.6-35B-A3B` decoder weights (`40L`) | **`34.661B`** | `32.212B` frozen routed MoE (SHA-256 verified) + `0.126B` shared + `2.323B` mixers/routers/untied embed+head |
| **Unique Consulted / Token** *(ex-input-embed/norms)* | `d=1` (`U=8`) / `s1` (`U=17.36`) / `s2_v2@600` (`U=18.19`, `1.764×`) / max (`U=24`) | **`2.947B` / `3.653B` / `3.716B` / `4.154B`** | Unique weights touched across `16` prelude+coda (`K=8`) and `24` core layers (`U` distinct experts/tok; `s2_v2@450` mid-way was `U=14.96` / `3.472B`) |
| **Compute-Equivalent Applications** *(incl. untied embed+head)* | `40L (d=1)` / `64L (d=2)` / `88L (d=3)` | **`3.454B` / `4.916B` / `6.378B`** | Re-executed weight applications (`2.946B / 4.409B / 5.871B` ex-input-embed); mid-way exit-gate blend magnitude was $\lvert g \rvert = 0.022\text{–}0.058$, and `s2_v2@600` unseen OOD `d=1` edged `d=3` (`S2.2` control pending) |
| **Trainable Delta Parameters** | `s1_loop_a6b` / `s3_v2_long_1m_lsa` / `s2_v2_joint_bptt` | **`18.41M` / `24.56M` / `156.94M`** | `0.053%` / `0.071%` / `0.453%` of stored decoder parameters; `0` routed MoE params trained |

### Verified Sprint 2A Empirical Highlights (`1× RTX PRO 6000 Blackwell`, `s2_v2@600` `17382c2e166d`)

| Benchmark / Audit (`fp8` KV, `3` repeats where applicable) | Stock `Qwen3.6 + YaRN 4×` | `s2_v2_joint_bptt@600` (`d=3`) | Verified Signal & Caveat |
| :--- | :---: | :---: | :--- |
| **Leak-Free `4K` Holdout (`0/4` doc overlap, `19,093` tok)** | `0.95000` (PPL `2.586`) | **`0.83671`** (PPL `2.309`) | **`-10.71%` PPL (`+469σ`)** vs. Stock; `d1→d3` `+0.03609` (`+309σ`). |
| **Pure Unseen OOD (`wikitext-103-raw-v1` test, `30,030` tok)** | `1.71242` (PPL `5.542`) | **`1.70122`** (PPL `5.481`) | **`-1.11%` PPL (`+33.6σ`)** vs. Stock; `d=1` (`1.69971`) slightly edged `d=3` (`-4.2σ`) → Sprint 2B `d=1` LoRA control (`S2.2`) is decisive. |
| **`128K` Dense / Sparse NLL & `8K–128K` 4-Needle RULER** | `0.10026` / `N/A` · `2/12` | **`0.00746` / `0.00796`** · **`12/12 (100%)`** | **`13.4×` lower `128K` NLL**; **`+0.00050` sparse–dense gap (`129×` tighter than `s3_v2`)**; flat `GDN-2` `max_l2` (`186.2 → 191.4 → 191.0`, `0` NaNs); `512K/1M` ladder still open. |
| **`S2.3` Causal Ablations (`4K`, SHA-256 restored)** | — | `no_gdn2 +156.6σ`, `no_router +6.8σ` | `GDN-2`, `PassRouter`, `lsa_sparse` (`0.8360` recall) = `KEEP_CRITICAL`; `QED` (`-0.31σ`) & `neg_eigval` (`-0.97σ`) = `PRUNE_CANDIDATE`. |
| **Kill-Gate `K0` (`N=20`) & Live `transformers 5.24.1` Audit** | — | **`YELLOW_B`** · **`3/4 PASS`** | `K0` Arm (c): `100.0%` EM retention @ `6.05×` smaller KV (`+0.6228`-nat NLL gap); `QL60-001` (`1.00×` KV) & `QL60-005` (`13/13`) LIVE PASS; `QL60-002` `cache` kwarg re-verify pending. |

### Architectural Highlights (Clean v2 Active Stack)
- **Calibrated `[12, 36)` Sandwich Recurrence (`40L / 64L / 88L`)**: Loops `Prelude [0,12)×1 + Core [12,36)×R + Coda [36,40)×1` (`R ∈ {1, 2, 3}`) via rank-256 subspace `ReEntryBridge`, zero-initialized `PassExitGate` (`torch.equal` Step-0 identity on unquantized paths), rank-64 per-pass Q/O LoRA adapters, and rank-32 `PassRouterAdapters` — keeping all `32.212B` routed MoE weights 100% frozen (SHA-256 verified).
- **Clean v2 Linear Backbone (`Native-GDN → GDN-2-1M`)**:
  - **Active Clean v2 (`s3_v2_long_1m_lsa` → `s2_v2_joint_bptt` → `s4_v2_rl`)**: Uses stock Qwen3.6 Gated DeltaNet during `1M` `D-LSA` calibration (`s3_v2`), then enables Step-0 bit-identical **`GDN-2-1M`** (`27.53M` params, `0` extra recurrent state bytes) mapped onto production `fla 0.6.0` DPLR Triton kernels with checkpointed `4×8K` macro-segmenting (`574898d`, cutting `T=32K` peak transient VRAM `3.28×` from `23.051 → 7.032 GiB`, keeping `s2_v2@600` peak at `69.97 GiB`). Sprint 2A `S2.3` causal ablations confirmed base `GDN-2` (`+156.6σ`) is the #1 driver while Bounded `QED` (`-0.31σ`) and Soft Neg-Eigval (`-0.97σ`) are zero-lift `PRUNE_CANDIDATE`s to remove in future stages.
  - **Retired Negative Ablation (`HSSD v3.1/v3.2`)**: Retained in `src/qloop/model/hssd.py` and `configs/stages/{s1b_hssd_distill,s3_long_1m_lsa,s2_joint_bptt,s4_rl}.yaml` strictly for audit/reproducibility; retired after 4-stage causal swap proved HSSD caused a `+0.79`-nat shock, dead vault pages (`3.26e-05`), `18K` needle failure, `+5.17 GiB` extra 1M state, and `5.6×` slower decode.
- **Cross-Pass Dynamic Expert Re-Routing**: Per-pass router residual adapters recruit distinct Top-8 experts on Passes 2 and 3 (`U=18.19` core unique experts/tok in completed `s2_v2@600`, `U=17.36` in `s1`; `S2.3` `no_pass_router` `+6.8σ`), expanding unique parameters consulted per token from `2.947B` (`d=1`) to `3.653B–3.716B` (`4.154B` theoretical max at `U=24`) with zero base-weight duplication.
- **Memory-Feasible `1M` `FP8` `1.0×` `SharedKV` + Per-Pass `D-LSA` & `4×32K` K/V-Replay BPTT**:
  - **Tier 1 (`1M` GPU-Resident Analytical State; `8K–128K` Empirically Verified)**: Pure `4.0×` YaRN + `10.00 GiB` `FP8` `1.0×` `SharedKV` across `R=1..3` + `0.33 GiB` GDN state + `0.32 GiB` Ward centroids = **`10.65 GiB`** (`+1.04 GiB` current eager `int64` `BucketIndex` compactable to `~0.04 GiB`), with untied per-pass `D-LSA` query indexers (`num_query_passes=3`). Sprint 2A verified `12/12 (100%)` 4-needle RULER and `+0.00050` sparse–dense gap through `128K`; `512K / 1M` length-ladder completion remains open (`B3`).
  - **Tier 2 (`10M` Host-DDR5 Stretch, Analytical)**: `Block64HadamardINT4Cache` (`2,304 B` 4-token micro-page, `53.64 GiB` pinned DDR5 @ `10M`) keeping `[0:64)` mRoPE orthogonal to `[64:256)` NoPE for exact `ReRoPE` commutativity.
  - **`SegmentedKVBPTTRunner` (`4×32K = 128K`)**: Streams `4 × 32K` segments while differentiably re-projecting Top-`2,048` selected prior-segment K/V tokens.
- **Single-GPU `APOLLO` Optimization & Blackwell Quantized Base (`sm_120` · `96 GB GDDR7 ECC` / `230 GB` SSD)**:
  - **`APOLLO` (`rank=256, channel`) inside `FusedStepOptimizer`**: Compresses optimizer moments on large 2D `k/v/o` (`104.86M` params in `s2_v2`, `-0.60 GiB` net) and `shared_expert` (`209.72M` in `s4_v2_rl`, `-0.94 GiB` net) while preserving exact `AdamW` on narrow adapters and 1D norms.
  - **`NVFP4` (`unsloth/Qwen3.6-35B-A3B-NVFP4`, `configs/model/qwen36_nvfp4.yaml`) & `FP8` (`Qwen/Qwen3.6-35B-A3B-FP8`, `configs/model/qwen36_fp8.yaml`)**: `~21–24 GiB` (`NVFP4`) or `~35 GiB` (`FP8`) VRAM & SSD footprint with Blackwell `sm_120` kernels (`kernels>=0.18.2,<0.19.0`, auto-installed and verified by `scripts/colab_setup.sh` and `setup_env`, with a zero-crash PyTorch fallback if unavailable) plus transient on-the-fly dequantizing `quant_autograd` for full backward support without persistent `BF16` weight copies.
- **Self-Contained Whole Model Pack & `OptimizedLoopInferenceEngine`**: Streams sharded `model-00001-of-NNNNN.safetensors` one `~4.5 GB` shard at a time (`< 6 GB` peak temporary SSD on Colab's `230 GB` disk) alongside standalone `configuration_qwen36_loophssd.py` & `modeling_qwen36_loophssd.py` (`QL60-001` `QuantizedHybridCache` and `QL60-005` `13/13` completion gates live-verified on `transformers 5.24.1`; `QL60-002` `cache` kwarg `4/4 PASS` re-verification pending).

📄 **Historical Design & Audit Documentation (`docs/`):**
- 📘 **[Backbone Architecture Specification (`Loop-HSSD v3`, Historical)](docs/ARCHITECTURE_SPEC.md)** — Original `v3` blueprint (HSSD portion now retired as negative ablation; `[12, 36)` loop & `D-LSA` carried into Clean v2).
- 🛠️ **[Implementation & Kill-Gate Plan (Historical)](docs/IMPLEMENTATION_PLAN.md)** — Original `M1–M6` work packages and `Gate -1 -> Gate 4` schedule.
- 📜 **[Formal Research Essay (Historical)](docs/RESEARCH_ESSAY.md)** — Proofs of Complex-to-Real `chunk_kda`, Write-Mass Retention, and Block-64 Hadamard `INT4`.
- 🧪 **[Dataset Audit Report (`Loop64-Fable-SIRSI-v1`)](docs/DATASET_AUDIT_REPORT.md)** — Live streaming triage, dedup, and 13-gram decontamination audit (see `MIDWAY_REVIEW_AND_ACTION_PLAN.md §5` for public-release data provenance policy).
- 🧭 **[Baseline Engineering Review (Chinese)](docs/ANALYSIS.md)** — Initial engineering audit of the foundation stack.

## Specialized Google Colab Notebooks (`notebooks/`)

| Notebook | Role & Workflow |
| :--- | :--- |
| **[`notebooks/colab_probe.ipynb`](notebooks/colab_probe.ipynb)** | **Zero-Training Diagnostic & Calibration Suite**: Executes `Gate -1` (`fp64` math proofs), `Gate 0a` (hardware backward & step-0 surgery), `Gate 0p-1..4` (expert headroom, Procrustes subspace re-entry sweep, `native` vs `yarn` vs `rerope` extrapolation, single-layer profiler), writes `probe_report.{json,md}`, and prints calibrated `-s` overrides. |
| **[`notebooks/colab_train_e2e.ipynb`](notebooks/colab_train_e2e.ipynb)** | **Self-Resuming E2E Training & Whole-Model Pack Release**: Runs/resumes Clean v2 stages (`s0_probe → s1_loop_a6b → s3_v2_long_1m_lsa → s2_v2_joint_bptt → s4_v2_rl`, with `s4_v2_rl` and pack export gated by Sprint 2B completion checks) across 24h Colab sessions and streams sharded `safetensors` (`< 6 GB` temp SSD). |
| **[`notebooks/colab_infer_eval.ipynb`](notebooks/colab_infer_eval.ipynb)** | **Shipped On-Weight Inference & Frontier `R=1/2/3` Evaluation**: Loads the exported pack into `OptimizedLoopInferenceEngine` (`40L / 64L / 88L`), demos `1M FP8` + `10M H_64 INT4` offload, and runs held-out depth-curve evaluation suites (`ARC-AGI-2`, `AIME 2026`, `LongBench v2`, `KernelBench`). |
| **[`notebooks/colab_train.ipynb`](notebooks/colab_train.ipynb)** | **Single-Notebook Pipeline**: Resumable `GDN-2` + `mHC` + `D-LSA` + `GRPO` single-notebook workflow. |

## Guarantees Enforced by Unit Tests (`make check`)

| Invariant | Test Suite |
|---|---|
| `GDN-2-1M + Bounded QED + Soft Neg-Eigval` Step-0 identity, `fla 0.6.0` DPLR equivalence, `4×8K` macro-segmenting, & Clean v2 stage wiring | `tests/test_gdn2.py`, `tests/test_gdn2_1m_qed_dplr.py`, `tests/test_v2_gdn_dlsa_pipeline.py` |
| Loop-wrapped model is **bit-identical** to stock Qwen3.6 at step 0 in unquantized precision (`PassRouterAdapters`, `ReEntryBridge`, `PassExitGate`, per-pass Q/O LoRA) | `tests/test_loop.py`, `tests/test_loop_v3.py`, `tests/test_gdn2.py` |
| Per-pass `D-LSA` (`num_query_passes=3`), `ReRoPE` distance clamping, `FP8KVCacheQuantizer`, `Block64HadamardINT4Cache` (`2,304 B` micro-page + `H_64` ReRoPE commutativity), and `SegmentedKVBPTTRunner` cross-segment K/V gradients | `tests/test_lsa.py`, `tests/test_longctx_v3.py` |
| Sharded `safetensors` Whole Model Pack export, low-disk `< 6 GB` shard streaming, standalone HF `AutoModelForCausalLM` loading, and `OptimizedLoopInferenceEngine` (`R=1/2/3`, chunked prefill, `1.0×` FP8/INT4 SharedKV) | `tests/test_pack_infer.py`, `tests/test_v3_integration_e2e.py` |
| Zero-training `Gate 0p-1..4` structured probe suite & frozen MoE SHA-256 invariance | `tests/test_probes_v3.py`, `tests/test_loop_v3.py`, `tests/test_v3_integration_e2e.py` |
| Retired `HSSD v3.1/v3.2` math & audit regression suite (preserved for negative-ablation reproducibility) | `tests/test_loop_hssd_v3_math.py`, `tests/test_hssd.py`, `tests/test_hssd_audit_regressions.py` |
| Isolated `LSA-NL` memory primitives (`kv_blob.py`, `esm.py`, `engram.py` — frozen pending Sprint 2B + `K0` `GREEN` calibration) | `tests/test_lsanl_memory.py`, `tests/test_engram_module.py` |
| Sprint 1 correctness/serving fixes & Sprint 2 harnesses (`d1_lora.py`, `kill_gate_k0.py`, `sprint2_audit.py`, `verify_tf524_serving_gpu.py`) | `tests/test_sprint1_*.py`, `tests/test_sprint2_*.py`, `tests/test_kill_gate_k0.py`, `tests/test_tf524_cache_compat.py` |
| Fused in-backward `APOLLO`/`AdamW` optimizer & chunked CE / log-probs == reference values & grads | `tests/test_optim_loss.py` |
| Cross-session HF Hub commit resume, cumulative stage delta handoff, quota squash, & `230 GB` disk cleanup | `tests/test_hub_sync.py` |
| Exact trainer resume & GRPO verifiable-reward promote/rollback gate | `tests/test_trainer_e2e.py`, `tests/test_rl.py` |
| All shipped Clean v2 & historical v3 YAML configs + all 4 Colab notebooks validate on Python 3.9 AST grammar | `tests/test_configs_cli.py`, `scripts/ast_check.py` |

## Repository Layout

```text
src/qloop/
  model/  loop.py (40L/64L/88L ETD + ReEntryBridge + PassExitGate + PassRouterAdapters + MoR)
          gdn2.py (Clean v2: GDN-2-1M + Bounded QED + Soft Neg-Eigval + fla DPLR 4x8K)
          mhc.py   lsa.py (per-pass D-LSA + YaRN)
          pos_rerope.py (ReRoPE + LongPrefixCalibrator)   offload.py (FP8 + Block-64 H_64 INT4)
          experts.py (ESFT scan + HotExpertDeltas + SHA-256 guard)   probes.py (Gate 0a / 0p-1..4)
          hssd.py (Retired HSSD v3.1/v3.2 negative ablation — kept for audit reproducibility)
          d1_lora.py (Sprint 2B parameter-matched d=1 LoRA control for A2 gate)
          loading.py   introspect.py   toy.py
  memory/ kv_blob.py   esm.py   engram.py (Isolated LSA-NL primitives — frozen pending Sprint 2B + K0 GREEN)
  train/  stack.py (cumulative v2/v3 delta stack)   trainer.py (LM + long-prefix + BPTT)
          bptt.py (SegmentedKVBPTTRunner 4x32K)     rl.py (GRPO/RSI)
          optim.py (fused APOLLO)   loss.py (CCE/chunked)   checkpoint.py
  infer/  engine.py (OptimizedLoopInferenceEngine: R=1/2/3, chunked prefill, FP8/INT4 SharedKV)
          hf_custom_modeling.py (standalone Qwen36LoopHSSDConfig & Qwen36LoopHSSDForCausalLM)
  export_pack.py (Whole Model Pack sharded safetensors exporter & low-disk HF Hub streamer)
  data/   registry.py  adapters.py  packing.py  mixture.py  decontam.py (13-gram)  sirsi_dataset.py
  eval/   generate.py  scorers.py  registry.py (16 frontier benchmarks)  runners.py
          sprint2_audit.py  kill_gate_k0.py  kernelbench.py  external.py (Harbor / tau-bench)  server.py
  hub.py (HF commit sync, 230 GB disk guard, Hub loader)
  env/doctor.py   debug.py (structured crash reports)   colab.py   cli.py
configs/  base.yaml, model/, stages/ (Clean v2 & historical v3), eval/suite.yaml, local/
notebooks/colab_probe.ipynb  colab_train_e2e.ipynb  colab_infer_eval.ipynb  colab_train.ipynb
```

## Local Development (macOS Apple Silicon / CPU)

```bash
make venv     # create .venv with torch + pytest + ruff
make check    # Python 3.9 AST check (.py + .ipynb) + ruff lint + full pytest suite
.venv/bin/python -m qloop probe -c configs/local/toy_lm.yaml --structured --seq-len 64
.venv/bin/python -m qloop train -c configs/local/toy_lm.yaml   # smoke-test looped LM + D-LSA
.venv/bin/python -m qloop train -c configs/local/toy_rl.yaml   # smoke-test looped GRPO + gate
```

## End-to-End Clean v2 Colab Workflow (`Probe → Train & Validate → Ship Pack`)

1. **Calibrate (`notebooks/colab_probe.ipynb`)**: Run zero-training `Gate -1` + `Gate 0a` + `Gate 0p-1..4` probes (`BASE_PRECISION = "nvfp4"` `~21–24 GiB`, or `"fp8"` `~35 GiB`):
   ```python
   from qloop.colab import setup_env, run_probe_notebook_suite
   setup_env(hf_repo="TassP0lar/Qwen3.6-35B-A3B-Loop64", base_precision="nvfp4")  # sets HF_XET_HIGH_PERFORMANCE=1 & verifies kernels>=0.18.2,<0.19.0
   report = run_probe_notebook_suite(
       config_path="configs/stages/s0_probe.yaml",
       base_precision="nvfp4",  # "nvfp4" | "fp8" | "bf16"
       push_to_hub=True,
   )
   ```
2. **Train Clean v2 Stages (`notebooks/colab_train_e2e.ipynb`)**: Self-resumes across Colab 24h sessions (`s0_probe → s1_loop_a6b → s3_v2_long_1m_lsa → s2_v2_joint_bptt → s4_v2_rl`) and streams sharded `safetensors` (`< 6 GB` peak temporary SSD) only after verified completion gates pass:
   ```python
   from qloop.colab import setup_env, run_v3_e2e_and_ship_pack
   setup_env(hf_repo="TassP0lar/Qwen3.6-35B-A3B-Loop64", base_precision="nvfp4")
   run_v3_e2e_and_ship_pack(
       hf_repo="TassP0lar/Qwen3.6-35B-A3B-Loop64",
       pack_repo="TassP0lar/Qwen3.6-35B-A6B-LoopHSSD-Pack",
       base_precision="nvfp4",  # "nvfp4" | "fp8" | "bf16"
       time_budget_hours=21.5,
       stream_low_disk=True,
   )
   ```
   Or export / stream a validated stage checkpoint directly via CLI:
   ```bash
   python -m qloop export-pack -c configs/stages/s2_v2_joint_bptt.yaml --ckpt runs/s2_v2_joint_bptt \
       --push-repo TassP0lar/Qwen3.6-35B-A6B-LoopHSSD-Pack --stream-low-disk
   ```

## Shipped On-Weight Inference (`OptimizedLoopInferenceEngine` & Hugging Face `AutoModel`)

```python
from qloop.infer import InferenceConfig, OptimizedLoopInferenceEngine

engine = OptimizedLoopInferenceEngine.from_pretrained(
    "TassP0lar/Qwen3.6-35B-A6B-LoopHSSD-Pack",
    cfg=InferenceConfig(prefill_chunk_size=4096, fp8_kv_cache=True),
)
# Switch dynamically per request: R=1 (40L, 3.454B eq.) | R=2 (64L, 4.916B eq.) | R=3 (88L, 6.378B eq.)
with engine.depth(3):
    out = engine.generate(
        "Prove that there are infinitely many primes p = 1 (mod 4).",
        max_new_tokens=2048,
        temperature=0.0,
    )
    print(out.text, out.metrics)
```

Standard Hugging Face `AutoModelForCausalLM` (`trust_remote_code=True`) and stage-sync `load_qloop_from_hub` are also supported out of the box:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

tok = AutoTokenizer.from_pretrained("TassP0lar/Qwen3.6-35B-A6B-LoopHSSD-Pack", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    "TassP0lar/Qwen3.6-35B-A6B-LoopHSSD-Pack", trust_remote_code=True, torch_dtype="auto"
)
```
