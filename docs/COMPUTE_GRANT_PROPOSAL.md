# Compute Allocation & Lab Cooperation Proposal (`v2.0`): `Qwen3.6-64L-CoMem` (`LSA-NL`)
### Beyond Static Looped MoEs: 5-Timescale Lifelong Memory, 3-Axis Dynamic Compute, and Lossless Mid-Flight Drafting on Frozen NVFP4 Backbones

> **Public Model Hub:** [`TassP0lar/Qwen3.6-35B-A3B-Loop64`](https://huggingface.co/TassP0lar/Qwen3.6-35B-A3B-Loop64) (`202+` live commits) · **Public Dataset:** [`TassP0lar/Loop64-Fable-SIRSI-v1`](https://huggingface.co/datasets/TassP0lar/Loop64-Fable-SIRSI-v1) (`v1.2`, `cc-by-4.0`)  
> **Single-Card Development Hardware:** `1× NVIDIA RTX PRO 6000 Blackwell Server Edition` (`GB202`, `sm_120`, **`96 GB GDDR7 with ECC`** [`94.97 GiB` CUDA-usable], `512-bit` bus, **`1,597 GB/s`** Server Edition peak bandwidth, `24,064` CUDA cores, `600 W` max TDP, passive server cooling)  
> **Research & Technical Documentation:** Flagship Proposal [`docs/PROPOSAL_LSA_NESTED_LEARNING.md`](PROPOSAL_LSA_NESTED_LEARNING.md) (`v2.0`) · Roadmap [`docs/ROADMAP_NEXTGEN_LOOP64_COMEM.md`](ROADMAP_NEXTGEN_LOOP64_COMEM.md) · Empirical Reports [`docs/EXPERIMENT_REPORT.md`](EXPERIMENT_REPORT.md) (`Loop64_Technical_Report_EN.pdf`) & [`docs/EXPERIMENT_REPORT_ZH.md`](EXPERIMENT_REPORT_ZH.md) (`Loop64_Technical_Report_ZH.pdf`) · Raw CSV/JSON & 7 pure-SVG figures in [`docs/experiment_logs/`](experiment_logs/)  
> **Codebase Readiness:** Open repository (`origin/main` `574898d+`), **`760` collected unit/parity/CUDA tests** (incl. **`38/38` Blackwell CUDA GDN-2 pytest** green), `0` lint / Python 3.9 AST errors  
> **Primary Compute Request:** **`~2,500` GPU-hours** (`1× 8×` H200 / B200 / RTX PRO 6000 node for `~2` weeks; scalable from **`~400` GPU-h** Tier 1 on `2–4×` GPUs to **`~6,000` GPU-h** Tier 3)

---

## 1. Executive Research Pitch: Moving Beyond Static Looped MoEs (`Inkling 2 Mini`)

Scaling test-time reasoning via long chains of thought alone hits severe KV-cache and serial-token bottlenecks, while training looped or lifelong-memory transformers from scratch requires industrial pretraining budgets. Over **`167.60+` active single-GPU hours** on one rented **`NVIDIA RTX PRO 6000 Blackwell Server Edition (96 GB GDDR7 ECC, 1,597 GB/s)`**, we have already proved that a frozen `35B` (`A3B`) NVFP4 hybrid MoE (`unsloth/Qwen3.6-35B-A3B-NVFP4` [1], `30` Gated DeltaNet [2] + `10` full-attention layers, `32.212B` routed experts 100% frozen and SHA-256 verified) can be retrofitted with `< 0.6%` trainable parameters via low-rank **APOLLO** [3] (`arXiv:2412.05270`) into a runtime-scalable `40L / 64L / 88L` looped reasoner with **`10.65 GiB`** of `1M`-token FP8 state (**`3.02×` smaller** than `Inkling 2 Mini`'s [4] `32.16 GiB` F16 KV).

Our goal for lab collaboration and multi-GPU scaling is **explicitly NOT to clone `Inkling 2 Mini`** (a static `48L` looped model with fixed per-token compute and flat `1M` KV). Instead, we propose **`Qwen3.6-64L-CoMem` (`LSA-NL` v2.0)**—combining our de-risked single-GPU looped `GDN-2-1M + QED` backbone [5], [6], [7], [8] with five orthogonal, publication-grade research pillars (full proofs in [`PROPOSAL_LSA_NESTED_LEARNING.md`](PROPOSAL_LSA_NESTED_LEARNING.md)):

1. **`LSA-NL` 5-Timescale Lifelong Memory (`L0 → L4`) & `10B`-Token Forgetting/Graduation Funnel:** Unites *Nested Learning*'s Continuum Memory System [9] with non-parametric precision without scratch pretraining or proactive interference [10]. Replaces flat `10M` context with a **Split-Horizon `3M–4M` Logical Context** (`1M–2M` live Partition A where `98.25 MiB` `GDN-2` runs cleanly isolated + `1M–2M` GPU-mounted Partition-B Pre-RoPE `H64-INT4` KV-Mem Blobs [11], [12], [13], [14] backed by `100M+` cold SSD tokens behind `262.5 MiB` GPU headers). A **4-stage lifelong funnel** (`10B → 800M → 133M → 80M → 60M` steady-state = **`345.6 GB` SSD + `150.2 MiB` GPU headers**, vs. infeasible `57.60 TB` raw) combines salience cherry-picking, `EpiCache` [15] `192`-D NoPE compaction, `SleepGate` [10] / `Zep` [16] bi-temporal tombstones + ACT-R hardening ($R_b(t)=e^{-\Delta t/S_b}$, $S_b \leftarrow 2S_b$), and **3-stage vertical KV-to-Engram graduation** (`0` backbone retraining): *Stage A* `Spark-X2.6` [17] `AlphaEdit` [18] + `WISE` [19] `144 MiB` closed-form `ESM` (`0.0` drift); *Stage B* modular `2K–4K` `Cartridges` / `Cartridges++` / `Cartridges at Scale` [20], [21], [22] (`~11–22 MiB`, `~38×` compression); and *Stage C* single-forward-pass `<50 ms` **`Doc-to-LoRA`** [23] hypernetwork + weekly **`SMF`** [24] `AnchoredMoEEngram` [25] + **`SEAL`** [26] self-edits.
2. **Six Surgical Upgrades on Original `Latent Sparse Attention` [27] (`arXiv:2610.42147`):** Upgrades our `src/qloop/model/lsa.py` bucket-mass KL to Original LSA's **per-key forward KL with truncated-depth "rest bucket" + `calibrate_tau`**; replaces the append-only tree with mutable bi-temporal `DLMG` (`HippoRAG 2` PPR [28] + `A-MEM` [29]); adds Pre-RoPE `H64-INT4` + clamped ReRoPE cross-session mounting; compacts `BucketIndex` to `int8` (`-1.0 GiB` saved); preserves untied per-pass `W_q^{(p)}` (`num_query_passes = 2–3`); and bridges horizontal KV into vertical Engrams.
3. **3-Axis Dynamic Elastic Compute (`MoR`) & Lossless `L35` Mid-Flight Self-Speculative Drafting:** Combines **Axis 1** token-level depth routing (**`MoR`** [30] + *Recurrent Depth* [31], already scaffolded in `src/qloop/model/loop.py:548-554,771-786` with `GDN-2` write-masking: `40L` easy / `64L` standard / `88L` top `5–10%` hard), **Axis 2** elastic active parameters (`3.45B ↔ 6.38B+` via native `top_k` `8↔16` + `s_null` Matryoshka `MoE-Engram`), and **Axis 3** **true Mid-Flight `L35` `EAGLE-3` [32] (`~35M` 1-layer head + `lm_head` = `0.23×` step) + `LoopSpec` [33] + `TreeWY` [34]** (verifying `48` `GDN-2` states in 1 triangular solve with `82–99×` less state memory) + `kv_share` `O(1)` rollback. Honest MoE Zipfian coupon-collector math (`u_MoE = 2.83×–3.96×`) proves full-`40L` drafting stalls at `1.03×–1.30×` vs `64L` (`< 40L` speed), whereas our `L35` `EAGLE-3` + `TreeWY` drafter achieves **`1.57×–1.58×` vs `64L` (`1.06×–1.07×` faster than raw `40L`!)** at `α = 0.85` and **`1.74×–1.76×` vs `64L` (`1.18×–1.19×` faster than `40L`!)** at `α = 0.90`—**100% lossless `64L` output!**
4. **Consumer `<= 24 GB VRAM` Deployment Profiles:** Because `Qwen3.6`'s entire dense core is only **`2.448B` (`2.42 GiB` FP8/Q8)** while `32.212B` (`92.93%`) is routed MoE, Zipfian NVFP4 hot/cold splitting (`KTransformers` [35], `MoE-Lightning` [36], `Fiddler` [37]) + `H64-INT4` KV deploys the *exact same* `64L` + `100M`-SSD agent inside **`17.66–18.93 GiB` on `RTX 3090/4090` (`24 GB`)**, **`10.89 GiB` on `16 GB` `4080/5070 Ti`**, and **`26.98 GiB` all-GPU NVFP4 + `1M` KV on `32 GB` `RTX 5090`**.
5. **Long-Term Phase-5 Native Multimodal KV-Mem Blobs:** `Qwen3.6-35B-A3B` is natively `image-text-to-text` (`3D mRoPE [11, 11, 10]`, `partial_rotary_factor = 0.25`), so visual/UI/document/video tokens naturally yield the exact same `192`-D NoPE keys (`ReKV` [38], `M3-Agent` [39], `M+` [40]) once text/code Phases `0–4` are validated.

---

## 2. Verified Single-GPU Execution Proof (Why This Work Is Already De-Risked)

### 2.1 Foundational Enabler: APOLLO (`arXiv:2412.05270`) [3] + `FusedStepOptimizer` on One `96 GB` Card

Training a `34.661B` looped MoE up to `131K–1M` tokens on a single **`96 GB GDDR7 ECC`** card without ZeRO/FSDP is enabled by **APOLLO** [3] (`arXiv:2412.05270`) integrated into **`FusedStepOptimizer`** (`configs/base.yaml:89-97`, `src/qloop/train/optim.py:382-421`):
- **Exact Routing Rule:** `apollo: {enabled: true, rank: 256, scale_type: channel, scale: 1.0, update_proj_gap: 200, min_dim: 128}` via `apollo_torch.APOLLOAdamW` routes every large 2D trainable matrix (`min(shape) >= 128`) to low-rank `rank=256` SGD-like memory APOLLO, while 1D norms & narrow `<128`-dim adapters use `AdamW`/`AdamWLite`.
- **Stage-by-Stage Runtime Record:** `s1_loop_a6b` (`137` tensors, `18.41M` params) ran `fused=True` in-backward; from commit `3a17c0d`, `s1b` / `s3*` / `s2*` / `s4*` switched to `fused=False` for atomic whole-step non-finite guards + global `clip_grad_norm_` while keeping `apollo.enabled: true`. In **`s2_v2_joint_bptt`** (`156.94M` / `387` registered tensors, `379` stepped in Gate 3), APOLLO covers all `30` large 2D full-attention matrices (`104.86M` params; `0.156 GiB` moments-only [`5.0×`, `-0.625 GiB`] or `0.186 GiB` incl. fp32 proj [`4.2×` net] vs. `0.781 GiB` AdamW), and in `s4_v2_rl`, APOLLO covers `209.72M` params (`2.50×` net, `-0.94 GiB` net). Blackwell vectorization (`_foreach_norm(..., dtype=fp32)`, single-pass `_foreach_mul_` clipping, batched `AdamWLite`) eliminates optimizer launch bottlenecks.

### 2.2 De-Risked Empirical Evidence (`167.60+` Active Single-GPU Hours, `627.56M+` Tokens)

Every architectural, kernel, VRAM-headroom, numerical-stability, and data-hygiene risk has been resolved on **`1× NVIDIA RTX PRO 6000 Blackwell Server Edition (96 GB GDDR7 ECC, 1,597 GB/s)`**. All entries distinguish `Measured`, `Derived Formula`, and `Estimated`; zero unmeasured downstream 2026 benchmark scores are claimed.

| Research Pillar | Class | Verified Evidence (Single Rented Blackwell GPU) | Reproducible Artifact |
| :--- | :---: | :--- | :--- |
| **Zero-Training Calibration (`s0_probe`)** | `Measured` | **`5/5`** gates passed in `177.44 s`; `1.3061` nats expert headroom (`88.9%` hard tokens); `[12,36)` core cuts zero-shot shock `3.26×`; pure `YaRN 4.0×` [41] wins `1M` extrap (`100%` needle) | `probe_report.json` (`058b4ad3`) |
| **88L Loop Healing (`s1_loop_a6b@1000`)** | `Measured` | **`18.41M` params (`0.053%`)**, `0` skips, `54.24 GiB` peak; `8K` NLL `0.9800 → 0.9191` (`depth_gain = +0.0609`); **`82.27%`** Pass-2 re-routing (`17.36` unique experts/tok); `mHC` bridge [42]; `16K` needle **PASS** | Hub `32eb565e` [43] |
| **Completed Clean v2 A/B Win (`s3_v2@600`) & Gate 2** | `Measured` | **`600/600` completed (`15.02 h`, `24.56M` params, `0` skips)**: held-out `64K` NLL **`1.3689 → 1.2996 → 1.2816`** (`depth_gain = +0.0873`, **`-0.5941` nats / `44.8%` lower PPL** vs HSSD `s3@600` `1.8757`); `512K–1M` recall `0.6514` peak / `0.5715` final; **`2,365 tok/s`** (`2.25×`); peak **`69.96 GiB`** (`69.52 GiB` live **`1,048,576`-tok** pass @ `s125`). **Gate 2:** **`12/12` SHARP**, **`7/7` (`100%`) `18K`/`64K` needle hits**, `0` loops, `0.0` slop | Hub `6b521f47`, `ec34f825`, `s3v2_gate_summary.json` |
| **Gate 1 + Gate 3 + Live `s2_v2` (`GDN-2-1M + QED`)** | `Measured` | **`27.53M` GDN-2 params (`0.079%`)**, `0.0` Step-0 parity, `0` extra state bytes. **Gate 1 (`574898d`):** **`38/38` CUDA pytest green**; fused `tl.int64` Triton prep **`53.55 ms`** (`2.00×`) & `T=1` **`0.0787 ms`** (`2.41×`); checkpointed `4×8K` macro-segmenting cuts wrapper VRAM **`3.28×`** (**`23.051 → 7.032 GiB`**). **Gate 3 (`4×32K`, `d=3`):** **`67.194 GiB` peak**. **Live `s2_v2`:** archived `s100/600` (`d0a1d1d0`, `1,040–1,173 tok/s`, `67.77 GiB`, `0` skips; `horizon_mult` `1.014× → 1.544×`) and **currently running cleanly past step `340/600`** on supervised Blackwell daemon with `0` skips | `s3v2_gate_summary.json`, `s2_v2_joint_bptt.csv`, Hub `d0a1d1d0`, `574898d` |
| **1M Inference & Training VRAM Ledger** | `Derived` + `Measured` | **1M Inference:** **`10.65 GiB`** FP8 SharedKV (**`3.02×` smaller** than Inkling 2 Mini [4] `32.16 GiB`; `5.96–6.28 GiB` `H64-INT4` [11]); `lsa.py:743-768` `BucketIndex` adds **`+1.04 GiB`** eager `int64` (`~33.54 GiB` total, compactable to `~0.04 GiB` → `-1.0 GiB`). **Training Peaks (`96 GB` GDDR7 ECC):** `54.24` (`s1`), `69.96` (`s3_v2`), **`67.19–67.77 GiB`** (live `s2_v2`, **`27.2 GiB`** free headroom) | `src/qloop/model/lsa.py:743-768`, `EXPERIMENT_REPORT.md` §II |
| **Audited Dataset (`v1.2`) & Blackwell `sm_120` Kernels** | `Measured` | `4,081` `sft_core` + `429` pref pairs [44], [45], [46], [47]; `1,889,432` live 2026 13-grams (`31` leaks blocked across 12 suites); decoupled `working_notebook` CoT lifted pass rate `4.1% → 100.0%` (`197/197`). **Blackwell `sm_120`:** **`18.99×`** NVFP4 E2M1 hitting **`1,148 GB/s = 71.9%`** of `1,597 GB/s` GDDR7 peak, **`9.32×/12.99×`** D-LSA FlashAttn, **`5.83×`** 2-pass KL, **`54.03×`** BPTT DMA+CCE [48], **`760` collected tests** | `dataset_audit_summary.json`, `docs/PERF_PROFILE_S2.md` |

### 2.3 Honest Negative Ablation (HSSD v3.2 Post-Mortem) & Data Forensics

A core pillar of our lab-ready methodology is converting negative ablations into principled upgrades rather than hiding them:
- **HSSD v3.2 Post-Mortem → Clean `GDN-2-1M + QED` via `DPLR` [8]:** Our v1 custom HSSD module suffered an $O(T)$ cumulative-phase Jacobian blowup (`6.12e+19` @ `s475`, fixed via `chunk_size/T` scaling), an irreversible **`+0.79`-nat** non-identity shock (`0.919 → 1.705`), collapsed long-horizon vault duty (`1.14e-6`), failed `18K` needle recovery even under `LSA=dense`, and `3.41 tok/s` `d=3` decode. Enforcing **mandatory `0.0` Step-0 bit-identity**, pairing NLL with generative Gate 2 needle truth-checks (`12/12` sharp, `7/7` needles on `s3_v2@600`), and generalizing pretrained DeltaNet onto production `fla 0.6.0` `DPLR` kernels yielded **`GDN-2-1M + QED`** (`0.0` Step-0 diff, `0` extra state bytes, `0.0787 ms` `T=1`, `3.28×` VRAM cut).
- **Dataset Poisoning Quarantine:** Forensics caught upstream `fable51` emitting literal redacted `<search_query>` placeholders (`14,511` occurrences), motivating full 13-gram benchmark decontamination (`31` leaks removed) and `100%` pass-rate `working_notebook` synthesis.

---

## 3. Proposed Next-Phase Research Agenda & Multi-GPU Allocation Request *(`Estimated`)*

All GPU-hour estimates derive directly from measured single-card Blackwell rates (`s2_v2` `4×32K` BPTT: `1,040–1,173 tok/s` ≈ `40–45` GPU-h per `600`-step run; `64L` `s4_v2_rl` GRPO/DAPO [49], [50] rollouts: `~150–250` GPU-h per `400`-step run, accelerated `~1.6×` once `L35`+`TreeWY` drafting lands; full 16-benchmark 2026 `d ∈ {1, 2, 3}` sweep: `~150–250` GPU-h/config). Every allocation tier is protected by **hard kill-gates (`K0–K4`)** so multi-GPU hours are never spent on unvalidated assumptions.

| Allocation Tier | Total GPU-Hours | Target Hardware / Wall Time | Funded Experiments, Scientific Scope & Hard Kill-Gates |
| :--- | :---: | :---: | :--- |
| **Tier 1: Core Completion & `LSA-NL` Kill-Gates** | **`~400` GPU-h** | `2–4×` H200 / B200 / RTX PRO 6000 (`4–7` days) | Finish `s2_v2@600` + `GDN-2 ON vs OFF` ablation (`30 h`) · Execute **`K0` Pre-RoPE KV-Blob Kill-Gate** (`0.5–4 h`) & upgrade `lsa.py` to per-key rest-bucket KL + `calibrate_tau` (`40 h`) · Focused `64L` `s4_v2_rl` (`DAPO-Math` + `KernelBench`, `180 h`) · Core 2026 depth sweep (`AIME 2026`, `GPQA`, `MRCR 1M`, `KernelBench L3`, `145 h`). **Gate:** `K0` `>= 92%` oracle (or `5%` `CacheBlend`), `lsa.py` recall `> 0.75`. |
| **Tier 2: Flagship Paper 2 — `LSA-NL` + Dynamic Compute & Mid-Flight Drafting *(Primary Recommendation)*** | **`~2,500` GPU-h** | **`1× 8×` H200 / B200 / RTX PRO 6000 node (`~2` weeks)** | Full `s2_v2` + multi-round `64L` `s4_v2_rl` (`SI-RSI` coding/SWE/math/clean MCP, `650 h`) · **Full `LSA-NL` `NL-0..NL-6` Ablations (`650 h`):** `3M–4M` Split-Horizon + `100M` SSD `DLMG` + `10B`-token ACT-R/`SleepGate` funnel (`K1/K2`), `AlphaEdit`/`WISE` `ESM` + `Cartridges++` (`K3`), `AnchoredMoEEngram` + `SMF` 5-shield ablation · **3-Axis Dynamic Compute & Lossless Mid-Flight Drafting (`450 h`):** `MoR` `40/64/88L` write-masked router + `L35` `EAGLE-3` (`~35M`) + `LoopSpec` + `TreeWY` GDN-2 drafter (`K4`: `α >= 0.82`, `100%` greedy parity, `>= 1.45×` measured speedup) + consumer `<= 24 GB` (`17.66–18.93 GiB`) verification (`NL-7`) · **Complete 16-Benchmark 2026 Sweep** (`d ∈ {1,2,3}` + FLOP-matched `40L` CoT, `600 h`) · Serving/forensics buffer (`150 h`). |
| **Tier 3: Paper 3 Frontier — One-Shot `Doc-to-LoRA` & Phase-5 Multimodal Memory** | **`~6,000` GPU-h** | `1–2× 8×` H200 / B200 nodes (`~4` weeks) | All of Tier 2 (`2,500 h`) · **Stage-C One-Shot Hypernetwork (`1,500 h`):** meta-train `<50 ms` `Doc-to-LoRA` [23] KV-to-Engram hypernetwork + `SEAL` [26] self-edit RL over `100M`-token repository archives · **Phase-5 Native Multimodal KV-Mem Blobs (`1,200 h`):** re-enable `Qwen3.6` built-in ViT (`3D mRoPE [11,11,10]`, `192`-D NoPE) for streaming UI/PDF/video episodic blobs (`ReKV` [38] / `M3-Agent` [39] on `ScreenSpot Pro`, `OmniDocBench`, streaming VQA) · Multi-seed variance bars & `R ∈ {1..4}` (`40L–112L`) scaling (`800 h`). |

---

## 4. 3-Paper Publication Ladder, Lab Cooperation & Day-1 Readiness

This compute partnership maps cleanly onto a **three-paper staged publication ladder** with incremental milestones and zero wasted compute:

1. **Paper 1 (v1.5 Technical Report — Immediate, Post-Phase 0):** Completed `s2_v2@600` (`40L/64L/88L`) telemetry, `GDN-2-1M + QED` causal ablation, HSSD v3.2 negative ablation, and Blackwell `sm_120` Triton kernels (`71.9%` GDDR7 peak, `3.28×` `4×8K` VRAM cut). Already drafted as reproducible bilingual EN/ZH US-Letter IEEE PDFs ([`Loop64_Technical_Report_EN.pdf`](papers/Loop64_Technical_Report_EN.pdf), [`Loop64_Technical_Report_ZH.pdf`](papers/Loop64_Technical_Report_ZH.pdf)).
2. **Paper 2 (Flagship Multi-GPU Grant Target — NeurIPS / ICML / ICLR / MLSys 2027):** *"Latent-Sparse Nested Learning (`LSA-NL`): Split-Horizon Continuum Memory, Dynamic Recursive Compute, and Lossless Mid-Flight Drafting for Frozen MoEs."* Delivers the complete `NL-0..NL-7` empirical study across `L0 → L4`, the `10B`-token forgetting/graduation funnel (`AlphaEdit` + `Cartridges++` + `SMF`), 6 surgical LSA refinements, `MoR` + `L35` `EAGLE-3` + `TreeWY` lossless speedup (`> 40L` speed at `64L` accuracy), and `<= 24 GB` consumer deployment profiles.
3. **Paper 3 (Tier-3 / Follow-On Lab Collaboration — Frontier Lifelong Multimodal Agents):** *"Instant Context-to-Engram Compilation (`Doc-to-LoRA`) and Native Multimodal KV-Mem Blobs for Hybrid Looped MoEs."* Extends `LSA-NL` to `<50 ms` single-forward-pass hypernetwork knowledge compilation (`Doc-to-LoRA` + `SEAL`) and unified visual/text/code `192`-D NoPE episodic blobs over `Qwen3.6`'s built-in VLM tower (`ReKV` + `M3-Agent`).

- **100% Open Community Deliverables & Day-1 Readiness:** All code, weights (`TassP0lar/Qwen3.6-35B-A3B-Loop64` [43]), decontaminated datasets (`Loop64-Fable-SIRSI-v1` [44]), standalone Blackwell `sm_120` Triton kernels, and evaluation harnesses are open and verified by `760` collected tests (`38/38` Blackwell CUDA GDN-2 tests green) with atomic non-finite rollback guards—ready to launch Tier-1 or Tier-2 runs on Day 1 upon allocation.

---

## References

[1] Qwen Team, "Qwen3.6-35B-A3B," Alibaba Cloud, Model Release, Apr. 16, 2026. [Online]. Available: https://huggingface.co/Qwen/Qwen3.6-35B-A3B

[2] S. Yang, J. Kautz, and A. Hatamizadeh, "Gated Delta Networks: Improving Mamba2 with Delta Rule," in *Proc. Int. Conf. Learn. Represent. (ICLR)*, 2025, arXiv:2412.06464.

[3] H. Zhu, Z. Zhang, W. Cong, X. Liu, S. Park, V. Chandra, B. Long, D. Z. Pan, Z. Wang, and J. Lee, "APOLLO: SGD-like Memory, AdamW-level Performance," *arXiv preprint arXiv:2412.05270*, Dec. 2024.

[4] Thinking Machines Lab, "Inkling 2 Mini," Technical Release, Nov. 20, 2026. [Online]. Available: https://thinkingmachines.ai

[5] A. Hatamizadeh, Y. Choi, and J. Kautz, "Gated DeltaNet-2: Decoupling Erase and Write in Linear Attention," *arXiv preprint arXiv:2605.22791*, May 2026.

[6] D. Gupta, A. Das, and D. Gupta, "The Query Knows What to Forget: A Second Erase Direction for Linear Attention," *arXiv preprint arXiv:2608.13668*, Aug. 2026.

[7] R. Grazzi, J. Siems, A. Zela, J. K. H. Franke, F. Hutter, and M. Pontil, "Unlocking State-Tracking in Linear RNNs Through Negative Eigenvalues," in *Proc. Int. Conf. Learn. Represent. (ICLR)*, 2025, arXiv:2411.12537.

[8] S. Yang and Y. Zhang, "FLA: A Triton-Based Library for Hardware-Efficient Implementations of Linear Attention Mechanisms," GitHub Repository (`fla-org/flash-linear-attention`, v0.6.0), 2024–2026.

[9] A. Behrouz, M. Razaviyayn, P. Zhong, and V. Mirrokni, "Nested Learning: The Illusion of Deep Learning Architectures," in *Adv. Neural Inf. Process. Syst. (NeurIPS)*, 2025, arXiv:2512.24695.

[10] Y. Xie, "Learning to Forget: Sleep-Inspired Memory Consolidation for Resolving Proactive Interference in Large Language Models (SleepGate)," *arXiv preprint arXiv:2603.14517*, Mar. 2026.

[11] A. Zandieh, M. Daliri, M. Hadian, and V. Mirrokni, "TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate," in *Proc. Int. Conf. Learn. Represent. (ICLR)*, 2026, arXiv:2504.19874.

[12] J. Hu, W. Huang, W. Wang, H. Wang, T. Hu, Q. Zhang, H. Feng, X. Chen, Y. Shan, and T. Xie, "EPIC: Efficient Position-Independent Caching for Serving Large Language Models," in *Proc. Int. Conf. Mach. Learn. (ICML)*, 2025, arXiv:2410.15332.

[13] J. Yao, H. Li, Y. Liu, S. Ray, Y. Cheng, Q. Zhang, K. Du, S. Lu, and J. Jiang, "CacheBlend: Fast Large Language Model Serving for RAG with Cached Knowledge Fusion," in *Proc. ACM Eur. Conf. Comput. Syst. (EuroSys)*, 2025, arXiv:2405.16444.

[14] D. Ma, Y. Wang, and T. Lan, "Block-Attention for Efficient Prefilling," in *Proc. Int. Conf. Learn. Represent. (ICLR)*, 2025, arXiv:2409.15355.

[15] M. Kim, A. Kundu, H.-B. Kim, R. Dixit, and M. Cho, "EpiCache: Episodic KV Cache Management for Long Conversational Question Answering," *arXiv preprint arXiv:2509.17396*, Sep. 2025.

[16] P. Rasmussen, P. Paliychuk, T. Beauvais, J. Ryan, and D. Chalef, "Zep: A Temporal Knowledge Graph Architecture for Agent Memory (Graphiti)," *arXiv preprint arXiv:2501.13956*, Jan. 2025.

[17] BaiTian6641, "Spark-X2.6-Recursion: Dynamic Latent Memory Graphs, Null-Space Vertical Episodic Adapters, and Anchored MoE-Engram Stability Shields," GitHub Repository, 2026. [Online]. Available: https://github.com/BaiTian6641/Spark-X2.6-Recursion

[18] J. Fang *et al.*, "AlphaEdit: Null-Space Constrained Knowledge Editing for Language Models," in *Proc. Int. Conf. Learn. Represent. (ICLR)*, 2025, arXiv:2410.02355.

[19] P. Wang *et al.*, "WISE: Rethinking the Knowledge Memory for Lifelong Model Editing of Large Language Models," in *Adv. Neural Inf. Process. Syst. (NeurIPS)*, 2024, arXiv:2405.14768.

[20] S. Eyuboglu, R. Ehrlich, S. Arora, N. Guha, D. Zinsley, E. Liu, W. Tennien, A. Rudra, J. Zou, A. Mirhoseini, and C. Ré, "Cartridges: Lightweight and General-Purpose Long Context Representations via Self-Study," *arXiv preprint arXiv:2506.06266*, Jun. 2025.

[21] S. Laguna, J. Monteiro, M. Cuturi, P. Ablin, and E. Gualdoni, "Cartridges++: KV Cache Compression without Off-Context Derailment," *arXiv preprint arXiv:2609.35621*, Sep. 2026.

[22] M. Hardalov, G. Iglesias, and A. de Gispert, "Cartridges at Scale: Training Modular KV Caches over Large Document Collections," *arXiv preprint arXiv:2606.04557*, Jun. 2026.

[23] R. Charakorn, E. Cetin, S. Uesaka, and R. T. Lange, "Doc-to-LoRA: Learning to Instantly Internalize Contexts," in *Proc. Int. Conf. Mach. Learn. (ICML)*, 2026, arXiv:2602.15902.

[24] J. Lin, L. Zettlemoyer, G. Ghosh, W.-T. Yih, A. Markosyan, V.-P. Berges, and B. Oğuz, "Continual Learning via Sparse Memory Finetuning," *arXiv preprint arXiv:2510.15103*, Oct. 2025.

[25] DeepSeek-AI, "Conditional Memory via Scalable Lookup: A New Axis of Sparsity for Large Language Models (Engram / Signed-Sqrt RMSNorm QK Gating)," *arXiv preprint arXiv:2601.07372*, Jan. 2026.

[26] A. Zweiger, J. Pari, H. Guo, E. Akyürek, Y. Kim, and P. Agrawal, "Self-Adapting Language Models (SEAL)," *arXiv preprint arXiv:2506.10943*, Jun. 2025.

[27] A. Kubicek, A. Caciolai, L. Peng, S. Sumant, B. Hoover, F. Greenlee, R. M. Battleday, R. Urban, T. Hoefler, and J. C. R. Whittington, "Latent Sparse Attention: Towards Subquadratic Ultra-Long Context Attention," *arXiv preprint arXiv:2610.42147*, Oct. 2026.

[28] B. J. Gutiérrez, Y. Shu, W. Qi, S. Zhou, and Y. Su, "From RAG to Memory: Non-Parametric Continual Learning for Large Language Models (HippoRAG 2)," *arXiv preprint arXiv:2502.14802*, Feb. 2025.

[29] W. Xu, Z. Liang, K. Mei, H. Gao, J. Tan, and Y. Zhang, "A-MEM: Agentic Memory for LLM Agents," *arXiv preprint arXiv:2502.12110*, Feb. 2025.

[30] S. Bae, Y. Kim, R. Bayat, S. Kim, J. Ha, T. Schuster, A. Fisch, H. Harutyunyan, Z. Ji, A. Courville, and S.-Y. Yun, "Mixture-of-Recursions: Learning Dynamic Recursive Depths for Adaptive Token-Level Computation," *arXiv preprint arXiv:2507.10524*, Jul. 2025.

[31] J. Geiping, S. McLeish, N. Jain, J. Kirchenbauer, S. Singh, B. R. Bartoldson, B. Kailkhura, A. Bhatele, and T. Goldstein, "Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach," *arXiv preprint arXiv:2502.05171*, Feb. 2025.

[32] Y. Li, F. Wei, C. Zhang, and H. Zhang, "EAGLE-3: Scaling up Inference Acceleration of Large Language Models via Training-Time Test," *arXiv preprint arXiv:2503.01840*, Mar. 2025.

[33] S. Cho, L. Cui, S. Kim, D. Han, and I. Han, "LoopSpec: Pipelined Self-Speculative Decoding for Looped Transformers," *arXiv preprint arXiv:2609.17184*, Sep. 2026.

[34] S. M. Ghantasala, "TreeWY: Speculative Verification for Gated DeltaNet Hybrids," *arXiv preprint arXiv:2608.20961*, Aug. 2026.

[35] H. Chen *et al.*, "KTransformers: Unleashing the Full Potential of CPU/GPU Hybrid Inference for MoE Models," in *Proc. ACM Symp. Operating Syst. Principles (SOSP)*, 2025. [Online]. Available: https://github.com/kvcache-ai/ktransformers

[36] S. Cao, S. Liu, T. Griggs, P. Schafhalter, X. Liu, Y. Sheng, J. E. Gonzalez, M. Zaharia, and I. Stoica, "MoE-Lightning: High-Throughput MoE Inference on Memory-Constrained GPUs," in *Proc. ACM Int. Conf. Architectural Support Program. Lang. Operating Syst. (ASPLOS)*, 2025, arXiv:2411.11217.

[37] K. Kamahori, T. Tang, Y. Gu, K. Zhu, and B. Kasikci, "Fiddler: CPU-GPU Orchestration for Fast Inference of Mixture-of-Experts Models," in *Proc. Int. Conf. Learn. Represent. (ICLR)*, 2025, arXiv:2402.07033.

[38] S. Di, Z. Yu, G. Zhang, H. Li, T. Zhong, H. Cheng, B. Li, W. He, F. Shu, and H. Jiang, "Streaming Video Question-Answering with In-Context Video KV-Cache Retrieval (ReKV)," in *Proc. Int. Conf. Learn. Represent. (ICLR)*, 2025, arXiv:2503.00540.

[39] L. Long, Y. He, W. Ye, Y. Pan, Y. Lin, H. Li, J. Zhao, and W. Li, "Seeing, Listening, Remembering, and Reasoning: A Multimodal Agent with Long-Term Memory (M3-Agent)," *arXiv preprint arXiv:2508.09736*, Aug. 2025.

[40] Y. Wang, D. Krotov, Y. Hu, Y. Gao, W. Zhou, J. McAuley, D. Gutfreund, R. Feris, and Z. He, "M+: Extending MemoryLLM with Scalable Long-Term Memory," in *Proc. Int. Conf. Mach. Learn. (ICML)*, 2025, arXiv:2502.00592.

[41] B. Peng, J. Quesnelle, H. Fan, and E. Shippole, "YaRN: Efficient Context Window Extension of Large Language Models," in *Proc. Int. Conf. Learn. Represent. (ICLR)*, 2024, arXiv:2309.00071.

[42] Z. Xie *et al.* (DeepSeek-AI), "mHC: Manifold-Constrained Hyper-Connections," *arXiv preprint arXiv:2512.24880*, Dec. 2025.

[43] TassP0lar, "Qwen3.6-35B-A3B-Loop64: Open Looped Hybrid Checkpoints and Empirical Telemetry," Hugging Face Model Hub, Nov. 2026. [Online]. Available: https://huggingface.co/TassP0lar/Qwen3.6-35B-A3B-Loop64

[44] TassP0lar, "Loop64-Fable-SIRSI-v1: Curated Anti-Slop 3-Beat Looped-CoT, Aesthetic Craft, and SI-RSI Recovery Dataset," Hugging Face Datasets Hub (v1.2), Nov. 2026. [Online]. Available: https://huggingface.co/datasets/TassP0lar/Loop64-Fable-SIRSI-v1

[45] E. Guha *et al.*, "OpenThoughts: Data Recipes for Reasoning Models," *arXiv preprint arXiv:2506.04178*, Jun. 2025.

[46] A. Bercovich *et al.*, "Llama-Nemotron: Efficient Reasoning Models," *arXiv preprint arXiv:2505.00949*, May 2025.

[47] F. Chollet, M. Knoop, G. Kamradt, B. Landers, and H. Pinkard, "ARC-AGI-2: A New Challenge for Frontier AI Reasoning Systems," *arXiv preprint arXiv:2505.11831*, May 2025.

[48] E. Wijmans, B. Huval, A. Hertzberg, V. Koltun, and P. Krähenbühl, "Cut Your Losses in Large-Vocabulary Language Models," in *Proc. Int. Conf. Learn. Represent. (ICLR)*, 2025, arXiv:2411.09009.

[49] Z. Shao, P. Wang, Q. Zhu, R. Xu, J. Song, X. Bi, H. Zhang, M. Zhang, Y. K. Li, Y. Wu, and D. Guo, "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models," *arXiv preprint arXiv:2402.03300*, Feb. 2024.

[50] Q. Yu, Z. Zhang, R. Zhu, Y. Yuan, *et al.*, "DAPO: An Open-Source LLM Reinforcement Learning System at Scale," *arXiv preprint arXiv:2503.14476*, Mar. 2025.
