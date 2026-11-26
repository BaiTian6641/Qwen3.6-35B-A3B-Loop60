# Mid-Way External Review Cross-Verification Ledger & 2-Sprint Remediation Plan (`v1.2`)

> **Language Companion:** [简体中文版 (`docs/MIDWAY_REVIEW_AND_ACTION_PLAN_ZH.md`)](MIDWAY_REVIEW_AND_ACTION_PLAN_ZH.md)  
> **Audited Repo Baseline:** `0f5d7f7` (two independent external audits) · **Current `main` HEAD:** `e1cf41b` (**Sprint 1 Tracks A–D 100% Merged** + **Sprint 2A GPU Closeout `rc=0`** + Sprint 2B harness)  
> **Completed `s2_v2_joint_bptt@600` Checkpoint:** Hub commit `17382c2e166d` (`2026-11-26T03:14:49Z`, `600/600` steps, `157,286,400` [`157.29M`] tokens, `156,941,238` trainable params, `69.97 GiB` peak VRAM, `U = 18.19` [`1.764×`, `3.716B` unique consulted params/tok]; superseded mid-way audited snapshot `s2_v2@450` `a5b20b20`)  
> **Target Hardware:** `1× NVIDIA RTX PRO 6000 Blackwell Server Edition` (`96 GB GDDR7 ECC` [`94.97 GiB` usable], `1,597 GB/s`, `sm_120`)  
> **Strategic Status:** **Sprint 1 (`Tracks A–D`) COMPLETED & MERGED** · **Sprint 2A + GPU Closeout (`e1cf41b`, `rc=0`: `S2.1` `4K–1M`, `S2.3`, `K0`, Live TF `5.24.1` `4/4 PASS`) COMPLETED ON BLACKWELL GPU** · **`s4_v2_rl`, `LSA-NL`/`CoMem` training expansion, and public Whole Model Pack release REMAIN FROZEN** pending only the **2 remaining gates**: **Sprint 2B (`S2.2` `~18.41M` `d=1` LoRA control)** and **downstream generative reasoning benchmarks (`AIME 2026` / `GPQA Diamond` / `ARC-AGI-2`)**.

---

## 1. Executive Mid-Way Verdict & Sprint 1 / 2A Closeout Status

Two independent mid-way external audits evaluated this repository at commit `0f5d7f7` alongside live Hugging Face Hub checkpoint metadata:
1. **Review 1 (`GPT-6 Astra Pro` Code & Systems Audit @ `0f5d7f7`):** Identified `10` concrete code, evaluation, serving, and orchestration defects (`QL60-001` through `QL60-010`) plus parameter-accounting inconsistencies (`§6`).
2. **Review 2 (Research & Codebase Audit @ `0f5d7f7` + live Hub `s2_v2@450` `manifest.json` `a5b20b20`):** Identified `16` scientific, methodological, telemetry, and hygiene findings grouped into `A1–A3` (scientific validity), `B1–B5` (claims & telemetry), `C1–C3` (loss/indexer/router correctness), and `D1–D5` (maintainability & provenance).

We independently re-verified **every single finding** at `17216c0` against live `s2_v2@450` Hub telemetry (`a5b20b20`): **all `10/10` Review 1 findings and `16/16` Review 2 findings were confirmed.** Since then, **all four Sprint 1 CPU remediation tracks (`Tracks A–D`) have been completed and merged onto `main`**, `s2_v2_joint_bptt` finished `600/600` steps (`17382c2e166d`), isolated CPU-verified Sprint 2 harnesses (`engram.py`, `kill_gate_k0.py`, `verify_tf524_serving_gpu.py`, `sprint2_audit.py`, `d1_lora.py` + `s2_2_d1_lora_control.yaml`) merged onto `main`, and **Sprint 2A + GPU Closeout (`e1cf41b`, `rc=0`)** completed full Blackwell GPU validation (`S2.1` leak-free holdout/OOD + `8K–1M` length ladder & 4-needle RULER, `S2.3` causal ablations, `K0` Pre-RoPE KV-Blob oracle, and live `transformers 5.24.1` serving/export **`4/4 PASS`**, full numbers in `§7`).

### What Holds Up (Validated Infrastructure & Sprint 2A Closeout Proofs)
- **Single-GPU Blackwell `sm_120` Systems Stack:** Retrofitting `Qwen/Qwen3.6-35B-A3B` [1] (`30` native Gated DeltaNet [2] + `10` gated full-attention layers, `32.212B` routed MoE weights 100% frozen and SHA-256 verified) runs stably on one `NVIDIA RTX PRO 6000 Blackwell Server Edition` (`96 GB GDDR7 ECC` [`94.97 GiB` usable], `1,597 GB/s`, `sm_120`), combining `GDN-2-1M` [3], Bounded `QED` [4], soft negative eigenvalues [5], per-pass `D-LSA` [6], low-rank `APOLLO` [7], `mHC` `LoopBridge` [8], pure `4.0×` YaRN [9], and `fla 0.6.0` DPLR Triton kernels [10].
- **Verified Engineering Milestones & Sprint 2A Closeout Confirmations:** `574898d` `4×8K` macro-segmenting cuts `T=32K` `GDN2Adapter` transient VRAM `3.28×` (`23.051 → 7.032 GiB`, keeping completed `s2_v2@600` peak at `69.97 GiB`); `1M` `FP8` `1.0×` `SharedKV` analytical state is `10.65 GiB` (`3.02×` smaller than Inkling 2 Mini's `32.16 GiB` [11]); open checkpoints [12] and audited `v1.2` data artifacts [13] remain traceable. In **Sprint 2A + Closeout (`e1cf41b`, `rc=0`, `§7`)**, leak-free `0/4`-overlap holdout (`-10.71%` PPL vs. Stock, `+469σ`), unseen `wikitext-103-raw-v1` OOD (`-1.11%` PPL vs. Stock, `+33.6σ`), `8K–1M` 4-needle RULER (**`20/20 = 100.0%`** vs. Stock `9/20 = 45.0%`), `1M` (`1,048,576`-token) Sparse NLL `0.00120` (`11.9×` lower than Stock `0.01429`, `17.1×` lower than `s3_v2` `0.02048`, `1.32×` faster than Stock dense `40L` at `88L` `d=3`, `65.19 GiB` peak VRAM, `0` OOM warnings), `128K` sparse–dense `D-LSA` tightness (`+0.00050` gap, `129×` tighter than `s3_v2`), flat `GDN-2` norms through `1M` (`186.2 → 191.4 → 191.0 → 191.38 → 191.21`, `max_abs = 27.74`, `0` NaNs), and live `transformers 5.24.1` serving/export (**`4/4 PASS`**, `217.6s`: `ENV`, `QL60-001`, `QL60-002` across `d = 1, 2, 3`, `QL60-005` `13/13`) all **PASSED**.

### What Originally Failed Audit — & What Remains Open After Sprint 1 + 2A Closeout
- **Closed in Sprint 1 (`Tracks A–D` Merged on `main`) & Sprint 2A Closeout (`e1cf41b`, `rc=0`):** Pre-pack SHA-1 document holdout (`QL60-003` / `A1`), RNG + weights-only `lr_factor` resume (`QL60-004`), valid-token BPTT CE normalization and pure `loss/lm` vs. `loss/aux` (`QL60-009` / `C1`), model-bound eval cache fingerprints (`QL60-010`), sparse `lsa_attention()` caller masks (`QL60-007`), pass-keyed unshared `BucketIndex` (`QL60-008`), mean-reduced `aux_loss()` (`C2`), live sparse `lsa/last_recall` telemetry (`B2`, `0.8360` at `4K`, `0.0627` at `512K`, `0.0317` at `1M`), deterministic router detection (`C3`), Clean v2 default orchestration (`QL60-006`), strict `_local_stage_done()` pack gating (`QL60-005`, `13/13` live PASS), live `QuantizedHybridCache` (`QL60-001`, `47.60s`) & in-package/standalone `.generate(max_new_tokens=8)` across `d = 1, 2, 3` (`QL60-002`, `19.01s`, **`4/4 PASS`** on real `transformers 5.24.1`), `8K–1M` length ladder + 4-needle RULER (`20/20`) + `GDN-2` `1M` bounded state norms (`B3`, `65.19 GiB` peak, `0` OOM warnings, `0` NaNs), `Stock Qwen3.6 + YaRN 4×` head-to-head + `3`-repeat noise floor (`A3`, `B4`), `QED` / neg-eigval causal pruning (`B5`), and `D3` / `D5` hygiene & provenance policy.
- **Explicit Remaining Open Gaps (Now Only 2 Core Gates Keep Freeze Active):** (1) **Sprint 2B (`S2.2` / `A2`)** `~18.41M` parameter-matched `d=1` LoRA control (`configs/stages/s2_2_d1_lora_control.yaml`, `~18` GPU-h) has NOT yet run on GPU — especially decisive because `s2_v2@600` unseen OOD `d=1` (`1.69971`) slightly edged `d=3` (`1.70122`, `-4.2σ`); and (2) **Downstream generative reasoning benchmarks** (`AIME 2026` / `GPQA Diamond` / `ARC-AGI-2`) remain unexecuted until `S2.2` passes and `s4_v2_rl` unfreezes. *(Separately, zero-training `K0` resolved to **`YELLOW_B_BLOCK_ATTN_CALIB`** [`100.0%` EM retention at `6.05×` smaller KV, `+0.6228`-nat NLL gap], gating only future `LSA-NL`/`CoMem` training expansion via a `2–4h` Block-Attention calibration, `§7.5`.)*

> [!IMPORTANT]
> **Binding Mid-Way Freeze Rule (Updated Post-Sprint 2A Closeout):** Sprint 1 (`Tracks A–D`) and **Sprint 2A + GPU Closeout (`e1cf41b`, `rc=0`: `S2.1` `4K–1M`, `S2.3`, `K0`, live TF `5.24.1` `4/4 PASS`)** are **100% complete on Blackwell GPU** (`§7`). `s4_v2_rl`, `LSA-NL` / `CoMem` training expansion, and public Whole Model Pack export remain **strictly frozen** pending only the **2 remaining core gates**: **Sprint 2B (`S2.2` `~18.41M` `d=1` LoRA control)** and **downstream generative reasoning benchmarks (`AIME 2026` / `GPQA Diamond` / `ARC-AGI-2`)** (plus `K0` Block-Attention calibration for `LSA-NL`/`CoMem`, `§6.3`).

---

## 2. Exhaustive Cross-Verification Ledger

### 2.1 Live Hub Telemetry Ground-Truth Snapshot: Mid-Way Audited (`s2_v2@450`, `a5b20b20`) vs. Final Completed (`s2_v2@600`, `17382c2e166d`)

Every live claim in Table 2.1 traces directly to Hub commits `a5b20b20` (`2026-11-25T17:54:48Z`, mid-way audited snapshot preceded by `cec24a8d@400`) and final completed `17382c2e166d` (`2026-11-26T03:14:49Z`, `600/600` steps) [12]:

| Metric Group | Mid-Way Audited `s2_v2@450` (`a5b20b20`) | Final Completed `s2_v2@600` (`17382c2e166d`) | Diagnostic Significance |
| :--- | :--- | :--- | :--- |
| **Run & Memory** | `step=450/600`, `117,964,800` tok, `n_params=156,941,238`, `has_optim=false`, `lr_factor=0.9738` | **`step=600/600`**, **`157,286,400` (`157.29M`) tok**, `156,941,238` trainable (`32.212B` MoE frozen), **`69.97 GiB` peak VRAM** | Completed cleanly on `1× RTX PRO 6000 Blackwell` (`96 GB GDDR7 ECC`); `QL60-004` weights-only `lr_factor` bug fixed for all future runs in Track A (`70f789e`). |
| **Eval & Loss** | In-stream `eval/loss_d1=0.5026 → d3=0.4965` (`+0.0062`); KL-mixed `loss/lm=3.8421` | **Sprint 2A leak-free `4K` holdout (`0/4` overlap):** `0.87281 → 0.84861 → 0.83671` (`+0.03609`, `+309σ`; `-10.71%` PPL vs. Stock `+469σ`) | Track A (`QL60-003`, `QL60-009`) + Sprint 2A replaced leaked/KL-polluted metrics with pure LM CE; OOD `d=1` (`1.69971`) slightly edged `d=3` (`1.70122`), making `S2.2` decisive. |
| **Loop & Routing** | `bridge_w_anchor=0.831/0.866`, `gate_abs_mean=0.022/0.058`, `U=14.96` (`1.522×`, `3.472B` uniq) | **`U=18.19` core experts** (**`1.764×`** mult, **`3.716B` unique consulted params/tok**) | Expert specialization expanded further through `step 600`; `S2.3` `no_pass_router` (`+0.00251`, `+6.8σ`) confirms `PassRouterAdapters` are `KEEP_CRITICAL`. |
| **GDN-2 & D-LSA** | `adapter_norm=0.741`, `decay_horizon_mult_max=4.545×`, `qed_lambda=0.00071`, `neg_eigval=0.9998`, `lsa/last_recall=0.0` | **`gdn2/adapter_norm=0.845`**, **`decay_horizon_mult_max=7.60×`**, **`qed_lambda=0.00085`**, **`neg_eigval=0.99986`**; Sprint 2A `lsa/last_recall=0.8360` (`4K`) | `S2.3` proves base `GDN-2` (`+0.03887`, `+156.6σ`) is the #1 driver while `QED` (`-0.31σ`) & `neg_eigval` (`-0.97σ`) are empirically zero-lift **`PRUNE_CANDIDATE`**s. |

### 2.2 Review 1 Findings Ledger (`GPT-6 Astra Pro`, `QL60-001`..`QL60-010`)

| ID | Cross-Ref | Status | Verified Resolution & Evidence (`main` `e1cf41b` + Sprint 2A Closeout) | Track / Commit | Gate Outcome |
| :--- | :---: | :---: | :--- | :---: | :--- |
| **`QL60-001`** | — | **Fixed & Live PASS** | [`src/qloop/infer/engine.py`](../src/qloop/infer/engine.py): `QuantizedHybridCache` implements full `transformers 5.24+` `Cache`/`CacheLayerMixin` contract, `update()` kwargs, geometric `_FP8Slot` pre-allocation, and zero-FP32 BF16 dequant. | Track C (`2b15d4c`, `581db44`) + `7346f77` | **LIVE GPU PASS (`47.60s`, `e1cf41b`)** on real `transformers 5.24.1`: exact `1.00×` KV across `d=1` & `d=3` for `bf16`/`fp8`/`hadamard_int4` (`2.00×` for `kv_share=False`). |
| **`QL60-002`** | `D1` | **Fixed & Live PASS** | [`src/qloop/infer/hf_custom_modeling.py`](../src/qloop/infer/hf_custom_modeling.py): `ModelOutput`-compatible `CausalLMPackOutput`, synced virtual `layer_types`, `logits_to_keep=0` returning `(B, T, V)`, `qloop_commit` metadata, and explicit `cache: Any = None` in `forward()`. | Track C (`2b15d4c`, `581db44`) + `7346f77` | **LIVE GPU PASS (`19.01s`, `4/4` overall on `e1cf41b`)**: BOTH in-package and standalone `Qwen36LoopHSSDForCausalLM` passed token-for-token `.generate(max_new_tokens=8)` across `d = 1, 2, 3` (`40L / 64L / 88L`), `logits_to_keep=0` `[1, 16, 248320]`, `Argmax Match = True`. |
| **`QL60-003`** | **`A1`** | **Fixed & Verified** | [`src/qloop/train/trainer.py`](../src/qloop/train/trainer.py) & [`src/qloop/data/mixture.py`](../src/qloop/data/mixture.py): pre-pack SHA-1 document hash partitioning (`split_role="train"` / `"eval"`). | Track A (`70f789e`) + Sprint 2A | CPU 8-doc repro `0` overlap + Sprint 2A Blackwell GPU holdout **PASS (`0/4` doc overlap**, `19,093` valid tok) & unseen `wikitext-103-raw-v1` OOD (`30,030` valid tok). |
| **`QL60-004`** | — | **Fixed & Merged** | [`src/qloop/train/trainer.py`](../src/qloop/train/trainer.py) & [`src/qloop/train/checkpoint.py`](../src/qloop/train/checkpoint.py): saves/restores `self.rng` state; weights-only resume logs `resume_mode="weights_only"` and syncs `self.opt.step_count = self.step`. | Track A (`70f789e`) | CPU unit tests PASS (`test_sprint1_train_data_resume.py`): exact RNG round-trip & weights-only `lr_factor` preservation verified. |
| **`QL60-005`** | — | **Fixed & Live PASS** | [`src/qloop/colab.py`](../src/qloop/colab.py): `_local_stage_done()` requires explicit `COMPLETED`/`PROMOTED` marker or verified `step >= target_steps`, rejecting bare/midrun `LATEST`. | Track C (`2b15d4c`) | **LIVE GPU PASS (`13/13` cases, `0.002s`)** in `runs/sprint2a/serving_audit_v2.md` (`e1cf41b`, `4/4 PASS`) including paused `step000001` rejection. |
| **`QL60-006`** | `D2`, `D3` | **Fixed & Merged** | [`src/qloop/colab.py`](../src/qloop/colab.py) & [`src/qloop/colab_supervisor.py`](../src/qloop/colab_supervisor.py): default stage tuple switched to Clean v2 (`s0_probe`, `s1_loop_a6b`, `s3_v2_long_1m_lsa`, `s2_v2_joint_bptt`, `s4_v2_rl`). | Track C (`2b15d4c`) | CPU unit tests PASS (`test_sprint1_hf_serving_ops.py`); retired HSSD stages strictly opt-in via `--stages`. |
| **`QL60-007`** | — | **Fixed & Merged** | [`src/qloop/model/lsa.py`](../src/qloop/model/lsa.py): enforces boolean & additive float 2D/3D/4D `attention_mask` across sink, window, and bucket keys with zero-overhead `None` fast path. | Track B (`36d73eb`) | CPU unit tests PASS (`test_sprint1_lsa_loop.py`): left/right/block-diag mask parity `< 1e-6`. |
| **`QL60-008`** | — | **Fixed & Merged** | [`src/qloop/model/lsa.py`](../src/qloop/model/lsa.py): keys `BucketIndex` cache by `(phys_layer, cur_pass)` when `kv_share=False` while keeping physical reuse when `kv_share=True`. | Track B (`36d73eb`) | CPU unit tests PASS (`test_sprint1_lsa_loop.py`): `kv_share=False` & `True` float64 parity `< 1e-12`. |
| **`QL60-009`** | **`C1`** | **Fixed & Verified** | [`src/qloop/train/bptt.py`](../src/qloop/train/bptt.py): normalizes segment backward by `valid_tok_s / total_valid_toks` (zero-valid safe) and logs pure `loss/lm` separately from `loss/aux`. | Track A (`70f789e`) | CPU `1 vs 3` valid-token grad parity PASS; Sprint 2A pure LM CE telemetry verified across all 4 arms. |
| **`QL60-010`** | — | **Fixed & Verified** | [`src/qloop/eval/runners.py`](../src/qloop/eval/runners.py): persists canonical run identity (`model_id`, checkpoint digest, `active_passes`, `config_hash`, `dataset_rev`) and rejects cross-checkpoint cache reuse. | Track A (`70f789e`) + Sprint 2A | CPU cross-model cache rejection PASS; Sprint 2A JSON persists verified `run_identities` per arm/dataset. |

### 2.3 Review 2 Findings Ledger (Code & Research Audit, `A1–A3`, `B1–B5`, `C1–C3`, `D1–D5`)

| ID | Severity | Status | Verified Resolution & Sprint 2A Closeout Evidence (`main` `e1cf41b`) | Track / Phase | Gate Outcome |
| :--- | :---: | :---: | :--- | :---: | :--- |
| **`A1`** (`QL60-003`) | Critical | **Closed** | Pre-pack SHA-1 holdout (`70f789e`) + Sprint 2A (`§7.2`) evaluated all 4 arms on `0/4`-overlap holdout (`19,093` valid tok) & unseen `wikitext-103-raw-v1` test (`30,030` valid tok). | Track A + Sprint 2A | **PASS:** `s2_v2@600` beats Stock `-10.71%` PPL (`+469σ`) on holdout & `-1.11%` PPL (`+33.6σ`) on unseen OOD. |
| **`A2`** | Critical | **Harness Merged; GPU Pending** | `src/qloop/model/d1_lora.py` + `configs/stages/s2_2_d1_lora_control.yaml` merged (`63d8b00`, CPU-verified) matching `~18.41M` params on `[12, 36)` at `d=1`. | **Sprint 2B (`~18` GPU-h)** | **OPEN DECISIVE GATE (#1):** Must run before unfreezing `s4_v2_rl` or `LSA-NL` (especially since `s2_v2@600` OOD `d=1` edged `d=3`). |
| **`A3`** | Medium | **Closed** | Sprint 2A (`§7.2`) executed `3` repeats (`mean ± std`) + `2× d=1` nondeterminism check (`max_diff = 2.36e-4 .. 1.18e-3`) on quantized Blackwell GPU. | Sprint 2A | **PASS:** All headline deltas vs. Stock (`+20.7σ .. +469.3σ`) and key causal ablations (`+5.5σ .. +156.6σ`) exceed `3σ`. |
| **`B1`** (`R1 §6`) | High | **Closed** | Canonical 4-axis parameter ledger locked in `README.md` & `§3` (updated with `s2_v2@600` `U=18.19` / `3.716B`); `S2.3` ablated `PassRouterAdapters`. | Track D + Sprint 2A | **PASS:** `no_pass_router` degrades NLL `+0.00251` (`+6.8σ`) → `KEEP_CRITICAL`. |
| **`B2`** | High | **Closed** | Track B (`36d73eb`) enabled periodic sampled `lsa/last_recall` in `mode="sparse"`; Sprint 2A + Closeout logged live recall (`0.8360` at `4K`, `0.6171 → 0.1340` at `8K–128K`, `0.0627` at `512K`, `0.0317` at `1M`). | Track B + Sprint 2A Closeout | **PASS:** `s2_v2@600` `128K` sparse–dense gap is **`+0.00050`** (`129×` tighter than `s3_v2`'s `+0.06457`); `512K/1M` dense skipped (`--skip-dense-above 131072`). |
| **`B3`** | High | **Closed (`8K–1M` Proven)** | Sprint 2A + Closeout (`e1cf41b`, `rc=0`, `§7.3`) proved `8K / 64K / 128K / 512K / 1M` (`1,048,576`) length ladder (`fp8` KV, geometric `_FP8Slot`, zero-FP32 BF16 dequant) + 4-needle RULER **`20/20 (100.0%)`** for `s2_v2@600` & `s3_v2@600`. | Track D + Sprint 2A Closeout | **PASS (`65.19 GiB` peak [`66,759 / 97,887 MiB`], `0` OOM warnings):** `s2_v2@600` `1M` Sparse NLL `0.00120` (`11.9×` < Stock), `GDN-2` `max_l2 = 191.21` (`0` NaNs); honest caveat: synthetic repeating filler scales NLL `~1/T`. |
| **`B4`** | Medium | **Closed** | Sprint 2A + Closeout (`§7.2`, `§7.3`) benchmarked **Stock `Qwen3.6 + YaRN 4×`** head-to-head against `s1@1000`, `s3_v2@600`, and `s2_v2@600` through `1M` (`1,048,576`) tokens. | Sprint 2A Closeout | **PASS:** `s2_v2@600` achieves `13.4×` lower `128K` NLL (`0.00746` vs. `0.10026`), `11.9×` lower `1M` NLL (`0.00120` vs. `0.01429`), `20/20` vs. `9/20` RULER, and `1.32×` faster `1M` wall time (`1,688.8s` vs. `2,231.3s`). |
| **`B5`** | Low | **Closed (Prune Confirmed)** | Sprint 2A `S2.3` (`§7.4`) directly ablated `QED` & `neg_eigval` and Closeout (`§7.3`) logged `GDN-2` `max_l2` through `1M` (`186.2 → 191.4 → 191.0 → 191.38 → 191.21`, `max_abs = 27.74`, `0` NaNs). | Sprint 2A Closeout | **PASS (ACTIONABLE):** `no_qed` (`-0.00009`, `-0.31σ`) & `no_neg_eigval` (`-0.00026`, `-0.97σ`) confirmed **`PRUNE_CANDIDATE`**; prune in future stages to restore strict `[0, 1]` GDN-2 stability bounds. |
| **`C1`** (`QL60-009`) | High | **Fixed & Merged** | [`src/qloop/train/bptt.py`](../src/qloop/train/bptt.py) separates `ce_loss` (`loss/lm`) and `aux_loss` (`loss/aux`) with valid-token weighting. | Track A (`70f789e`) | CPU parity PASS & Sprint 2A pure LM CE verified. |
| **`C2`** | High | **Fixed & Merged** | [`src/qloop/model/lsa.py`](../src/qloop/model/lsa.py): `LSAManager.aux_loss()` defaults to `aux_reduction="mean"`, keeping `lsa_kl` `O(1)`. | Track B (`36d73eb`) | CPU unit tests PASS (`test_sprint1_lsa_loop.py`). |
| **`C3`** | Low | **Fixed & Merged** | [`src/qloop/model/loop.py`](../src/qloop/model/loop.py): deterministic install-time router class/attribute detection (fallback only for custom test mocks). | Track B (`36d73eb`) | CPU unit tests PASS (`test_sprint1_lsa_loop.py`). |
| **`D1`** | High | **Track C Fixed; Extract Backlog** | Track C (`2b15d4c`, `581db44`) + `7346f77` fixed `ModelOutput`, `logits_to_keep=0`, `cache: Any = None`, & `qloop_commit` (`4/4 PASS` live); extracting the `~7.9K`-line standalone string remains Post-Sprint-1 backlog. | Track C + Backlog | Live `4/4 PASS` verified; standalone extraction scheduled post-Sprint-2B. |
| **`D2`** | Medium | **Backlog** | Archive retired `hssd.py` (`13,255` lines) + v1 configs to `qloop/legacy/` while keeping negative-ablation regression tests intact. | Post-Sprint-1 Backlog | Scheduled post-Sprint-2B (default pipelines already switched to Clean v2 in `QL60-006`). |
| **`D3`** | Medium | **Closed** | Overhauled `README.md` around Clean v2 + canonical 4-axis parameter ledger; fixed stale `configs/base.yaml` comment without altering YAML keys. | Track D (`2d23615`) | Merged & verified (`test_configs_cli.py`, `test_experiment_report.py`). |
| **`D4`** | Low | **Backlog** | Split fixed prose assertions in `tests/test_experiment_report.py` into dedicated `make docs-check` target. | Post-Sprint-1 Backlog | Scheduled post-Sprint-2B. |
| **`D5`** | Medium | **Closed** | Portable paths across `report.py`, `build_sirsi_dataset.py`, & docs; `pyarrow>=16.0` added to `[dev]`; binding `§5` restricted-provenance exclusion policy locked. | Tracks C (`2b15d4c`) & D (`2d23615`) | Merged & verified (`0` hardcoded local user paths). |

---

## 3. Canonical Parameter Accounting Table

To resolve Review 1 `§6` and Review 2 `B1` permanently, this repository enforces a **strict four-axis parameter ledger** for `Qwen/Qwen3.6-35B-A3B` [1] under the calibrated `[12, 36)` core (`16` non-looped prelude+coda layers `[0,12) ∪ [36,40)` + `24` looped core layers `[12,36)`, `E = 256` routed experts/layer, `K = 8` Top-$K$, `d_model = 2048`, `d_moe = 512`, `V = 248,320` untied input embedding and LM head).

### 3.1 Exact Closed-Form Building Blocks (`tests/test_loop_hssd_v3_math.py`)

$$\begin{aligned}
P_{\text{expert}} &= 3 \times 2048 \times 512 = 3{,}145{,}728 \quad (3.1457\text{M}), \\
P_{\text{routed}} &= 40 \times 256 \times P_{\text{expert}} = 32{,}212{,}254{,}720 \quad (\mathbf{32.21225\text{B}},\; 100\%\text{ frozen}), \\
P_{\text{shared}} &= 40 \times 1 \times P_{\text{expert}} = 125{,}829{,}120 \quad (\mathbf{0.12583\text{B}}), \qquad P_{\text{routed}+\text{shared}} = \mathbf{32.33808\text{B}}, \\
P_{\text{dense,uniq}}^{\text{ex-emb}} &= P_{\text{shared}} + P_{\text{routers}} + P_{\text{lm\_head}} + P_{\text{GDN}(30)} + P_{\text{Attn}(10)} \\
&= 0.12583 + 0.02105 + 0.50856 + 1.01155 + 0.27263 = \mathbf{1.93962\text{B}}, \\
P_{\text{uniq}}(U) &= P_{\text{dense,uniq}}^{\text{ex-emb}} + (16 \times 8 + 24 U) \times P_{\text{expert}}.
\end{aligned}$$

### 3.2 Canonical Four-Axis Parameter Accounting Table

| Accounting Axis | Scope / Regime | Exact Parameter Count | Precise Breakdown & Mandatory Interpretation Note |
| :--- | :--- | :---: | :--- |
| **1. Stored Base Parameters** | Total `Qwen3.6-35B-A3B` decoder (`40L`) | **`34.661B`** | `32.212B` frozen routed MoE (SHA-256 verified) + `0.126B` shared experts + `2.323B` mixers/routers/untied embed+head (`1.012B` GDN + `0.273B` Full Attn + `0.021B` routers + `1.017B` embed+head + `~0.18M` norms). |
| **2. Unique Consulted / Token** *(ex-input-embed/norms)* | `d=1` (`U=8`) | **`2.947B`** | `1.9396B` dense unique + `1.0066B` routed (`320` expert reads across `40L`; `2.9463B` ex-norms / `2.947B` incl. norms). |
| | `s2_v2@450` (`U=14.96`, `1.522×`, mid-way) | **`3.472B`** | `1.9396B` + `(128 + 24 × 14.96) × 3.1457M = 1.5321B` unique routed (`3.4717B`). |
| | `s1@1000` (`U=17.36`) | **`3.653B`** | `1.9396B` + `(128 + 24 × 17.36) × 3.1457M = 1.7133B` unique routed (`3.6529B`). |
| | **`s2_v2@600` (`U=18.19`, `1.764×`, final)** | **`3.716B`** | `1.9396B` + `(128 + 24 × 18.19) × 3.1457M = 1.7760B` unique routed (`3.7156B`; Hub `17382c2e166d`). |
| | Theoretical max at `R=3` (`U=24`) | **`4.154B`** | `1.9396B` + `(128 + 24 × 24) × 3.1457M = 2.2146B` unique routed (`4.1542B`). |
| **3. Compute-Equivalent Applications** *(incl. untied embed+head)* | `40L` (`d=1`, `R=1`) | **`3.454B`** | Single pass (`2.946B` ex-input-embed; `2.436B` prefill GEMMs ex-embed/head). |
| | `64L` (`d=2`, `R=2`) | **`4.916B`** | `+1` core pass (`4.409B` ex-input-embed; `kv_share=True` skips `6` core `k/v` `-0.0126B` $\to$ `3.885B` prefill GEMMs). |
| | `88L` (`d=3`, `R=3`) | **`6.378B`** | `+2` core passes (`5.871B` ex-input-embed). **Caveat:** FLOP-equivalent applications do NOT equal effective reasoning lift — mid-way `s2_v2@450` exit-gate blend magnitude was **$\lvert g \rvert = 0.022\text{–}0.058$**, and `s2_v2@600` unseen OOD `d=1` (`1.69971`) still slightly edged `d=3` (`1.70122`). |
| **4. Trainable Delta Parameters** | `s1_loop_a6b` / `s3_v2` / `s2_v2` | **`18.41M` / `24.56M` / `156.94M`** | `0.053%` / `0.071%` / `0.453%` of stored decoder parameters (`0` routed MoE parameters modified). |

**Reporting Standard:** All future documentation, paper drafts, and model cards MUST distinguish **unique parameters consulted per token (`2.947B → 3.472B–3.716B`, max `4.154B`)** from **compute-equivalent parameter applications (`3.454B / 4.916B / 6.378B`)**, and always pair compute-equivalent claims with measured exit-gate magnitude (`|g|`) and parameter-matched `d=1` controls (`A2`).

---

## 4. Literature-Grounded Diagnosis of `depth_gain = 0.0062` & `D-LSA` Recall

Mid-way `s2_v2@450` telemetry (`a5b20b20`) exposed why in-stream `eval/depth_gain` compressed to `+0.0062` (`d1=0.5026 → d2=0.4999 → d3=0.4965`) and why `lsa/last_recall` read `0.0`. Four compounding mechanisms—grounded directly in recent recurrence, sparse-attention, and linear-attention literature—explained the observations and have now been directly tested in **Sprint 2A** (`§7`):

1. **Retrofitted Recurrence Shortcut Collapse (`McLeish et al.` [14], `MoR` [15], `Recurrent Depth` [16], `mHC` [8]):**
   - In `s2_v2@450`, the `mHC` `ReEntryBridge` [8] retained **`bridge_w_anchor = 0.831 / 0.866`** on the Pass-1 prelude anchor while assigning only **`bridge_w_h_prev = 0.057 / 0.050`** to the recurrent hidden state, paired with near-zero `PassExitGate` magnitudes (**`gate_abs_mean = 0.022 / 0.058`**) and Pass-3 drift back toward Pass 1 (`Jaccard p12 = 0.080` vs. `p13 = 0.438`).
   - As analyzed in *Teaching Pretrained Language Models to Think Deeper with Retrofitted Recurrence* (`McLeish et al., arXiv:2511.07384` [14]), strong input-anchor injection invites a **shortcut local minimum**. **Sprint 2A empirical verification (`§7.2`, `§7.4`):** On `s2_v2@600`, `low_anchor_probe` (`max_anchor_weight = 0.25`, `<= 0.3`) worsens holdout NLL by `+0.00370` (`+11.05σ`)—confirming the trained checkpoint depends on `w_anchor ≈ 0.85` and requires a **training curriculum**, not a zero-shot weight shift—while on unseen `wikitext-103-raw-v1` OOD, `s2_v2@600` `d=1` (`1.69971`) slightly beats `d=3` (`1.70122`, `-4.2σ`), whereas `s1@1000` (`+13.2σ`) and `s3_v2@600` (`+38.4σ`) retain positive OOD depth gain.
2. **Pretraining Scale Gap vs. Native LoopLMs (`Ouro` [17] & `Inkling 2 Mini` [11]):**
   - Native looped pretraining systems such as **Ouro** (`arXiv:2510.25741` [17]) train shared recurrent blocks from scratch across **`7.7 trillion` (`7.7T`) tokens** with an entropy-regularized adaptive-depth objective; `Recurrent Depth` [16] pretrains over `800B` tokens, and `MoR` [15] trains adaptive token-level routing end-to-end.
   - Our retrofit updates only `0.053%–0.453%` of parameters across `157.29M` tokens (`s2_v2@600`, `~49,000×` fewer tokens than Ouro [17]) atop `32.212B` frozen single-pass experts. Even with strong `4K` holdout depth gains (`+0.03609` to `+0.09327`, `+163σ .. +314σ`), the Sprint 2B `~18.41M` `d=1` LoRA control (`A2`) remains mandatory to isolate genuine recurrence lift from adapter capacity.
3. **Original `D-LSA` Distillation Contract vs. Sparse Blindness (`Original LSA` [6]):**
   - Original `D-LSA` (`arXiv:2610.42147` [6]) distills indexers using per-key rest-bucket KL with explicit temperature calibration (`calibrate_tau`), whereas [`src/qloop/model/lsa.py`](../src/qloop/model/lsa.py) optimizes bucket-mass KL and previously summed KL across all `10` layers × passes (`C2`) while leaving `lsa/last_recall = 0.0` in `mode="sparse"` (`B2`).
   - **Sprint 1 Track B + Sprint 2A outcome (`§7.3`, `§7.4`):** Mean-reduced `aux_loss()` and periodic sparse `last_recall` are merged (`36d73eb`). On `s2_v2@600`, sparse `D-LSA` achieves `lsa/last_recall = 0.8360` at `4K` (beating dense `lsa_dense` by `-0.00192` nats, `+5.5σ`) and collapses the `128K` sparse–dense NLL gap to **`+0.00050`** (`129×` tighter than `s3_v2@600`'s `+0.06457`).
4. **Live `QL60-004` `step 340` LR Reset & `B5` `GDN-2` Extension Utilization:**
   - The `step 340` fresh-session weights-only Hub resume reset `lr_factor` to `0.9738` at `step 450/600`, now permanently prevented by Track A (`70f789e`).
   - Meanwhile, `s2_v2@600` (`17382c2e166d`) reached `gdn2/adapter_norm = 0.845` and `decay_horizon_mult_max = 7.60×`, while `qed_lambda = 0.00085` and `neg_eigval = 0.99986` remained near identity. **Sprint 2A `S2.3` (`§7.4`) settled `B5` conclusively:** disabling `GDN-2` [3] causes a massive **`+0.03887` (`+156.6σ`)** regression (**#1 driver**), whereas disabling Bounded `QED` [4] (`-0.00009`, `-0.31σ`) or soft negative eigenvalues [5] (`-0.00026`, `-0.97σ`) causes **zero degradation** — formally classifying both as **`PRUNE_CANDIDATE`** so future stages can simplify the kernel and restore strict `[0, 1]` `GDN-2` stability bounds.

---

## 5. Explicit Data Provenance & Release Policy (`D5`)

Review 2 (`D5`) flagged the presence of third-party Claude-distilled datasets inside internal stage configs (`configs/stages/s0_probe.yaml`, `s1_loop_a6b.yaml`, `s2_v2_joint_bptt.yaml`) and `Loop64-Fable-SIRSI-v1` [13]:
- `opus46_3000x` (`nohurry/Opus-4.6-Reasoning-3000x-filtered`; `0.05–0.08` stage weight, `538` rows in `sft_core`)
- `opus46_traceinv` (`Jackrong/Claude-opus-4.6-TraceInversion-9000x`; `0.07` stage weight, `533` rows in `sft_core`)
- `opus46_3300x` (`Crownelius/Opus-4.6-Reasoning-3300x`; `36` deduplicated rows in `sft_core`)

Although those upstream Hub repositories carry `apache-2.0` metadata tags, synthetic traces distilled from proprietary third-party APIs carry upstream Terms-of-Service and provenance ambiguity. We therefore establish a **binding two-tier data provenance policy**:

1. **Restricted-Provenance Diagnostic Classification:** `opus46_3000x`, `opus46_traceinv`, and `opus46_3300x` are formally classified as **internal research diagnostic mixtures only**.
2. **Strict Exclusion from Public Releases & Grant Artifacts:** **Zero rows** from `opus46_*` may be used to train or publish any public Whole Model Pack checkpoint, public dataset card, or external compute grant / paper submission.
3. **Permissive Open-Weight Replacement Stack:** All public release and post-Sprint-2 clean runs MUST source 100% of reasoning and agentic data from purely permissive, open-weight / open-license corpora:
   - **`OpenThoughts3`** (`open-thoughts/OpenThoughts3-1.2M`, `apache-2.0`) [18] behind strict `think_unclosed` + repetition + AST gates;
   - **`Llama-Nemotron`** (`nvidia/Llama-Nemotron-Post-Training-Dataset`, `cc-by-4.0`) [19] with per-row NVIDIA attribution;
   - **`OpenThoughts-Agent-SFT`** (`open-thoughts/OpenThoughts-Agent-SFT-ColdStartForRL-10K`, `apache-2.0`);
   - **`ARC-AGI-2` `training` split** (`apache-2.0`) and **author-original / open-weight `SIRSI`** (`Loop64-Fable-SIRSI-v1` clean open-only build [13]), all gated by our `13`-gram `2026` benchmark decontaminator (`1,889,432` indexed 13-grams).

---

## 6. Tracked 2-Sprint Remediation & Validation Plan

### 6.1 Sprint 1 — CPU Engineering, Correctness & Hygiene Fixes (`0` GPU-h) — **100% COMPLETED & MERGED**

| Track / Branch | Status | Findings Closed | Merged Scope on `main` (`e1cf41b`) |
| :--- | :---: | :---: | :--- |
| **Track A** (`fix/sprint1-eval-bptt-resume`) | **MERGED (`70f789e`)** | `QL60-003` (`A1`), `QL60-009` (`C1`), `QL60-004`, `QL60-010` | `mixture.py`, `trainer.py`, `bptt.py`, `checkpoint.py`, `optim.py`, `runners.py`, `tests/test_sprint1_train_data_resume.py` — `0` doc overlap holdout, valid-token BPTT CE + pure `loss/lm`, exact `self.rng` + weights-only `lr_factor` sync, model-bound eval cache. |
| **Track B** (`fix/sprint1-lsa-correctness`) | **MERGED (`36d73eb`)** | `QL60-007`, `QL60-008`, `C2`, `B2`, `C3` | `lsa.py`, `loop.py`, `tests/test_sprint1_lsa_loop.py` — caller masks `< 1e-6`, pass-keyed unshared `BucketIndex` `< 1e-12`, mean-reduced `aux_loss()`, sparse `last_recall` sampling, install-time router detection. |
| **Track C** (`fix/sprint1-hf-serving-ops` + `fix/s2a-closeout-tf524-1m`) | **MERGED (`2b15d4c`, `581db44`, `7346f77`)** | `QL60-001`, `QL60-002`, `QL60-005`, `QL60-006`, `B3` VRAM, `D5` (code) | `engine.py`, `hf_custom_modeling.py`, `export_pack.py`, `sprint2_audit.py`, `colab.py`, `colab_supervisor.py`, `report.py`, `build_sirsi_dataset.py`, `pyproject.toml`, `tests/test_sprint1_hf_serving_ops.py` — TF `5.24.1` `QuantizedHybridCache` + `cache: Any = None` + geometric `_FP8Slot` BF16 dequant + `--skip-dense-above 131072` (`4/4 PASS` & `65.19 GiB` `1M` peak), `13/13` `_local_stage_done()`, Clean v2 defaults, portable paths + `pyarrow>=16.0`. |
| **Track D** (`docs/midway-review-and-cleanup`) | **MERGED (`2d23615`)** | `B1`, `D3`, `D5` (docs & policy) | `MIDWAY_REVIEW_AND_ACTION_PLAN{,_ZH}.md`, `README.md`, `DATASET_AUDIT_REPORT.md`, `IMPLEMENTATION_PLAN.md`, `configs/base.yaml` — strict IEEE `[1]..[20]` parity, `GDDR7 ECC` hygiene, `0` user paths, binding `§5` provenance policy. |
| **Isolated Sprint 2 Harnesses** | **MERGED (`591b0df..71c4a99`)** | Pre-GPU `S2.1–S2.3`, `K0`, TF `5.24` | CPU-verified [`src/qloop/memory/engram.py`](../src/qloop/memory/engram.py), [`src/qloop/eval/kill_gate_k0.py`](../src/qloop/eval/kill_gate_k0.py), [`scripts/verify_tf524_serving_gpu.py`](../scripts/verify_tf524_serving_gpu.py), [`src/qloop/eval/sprint2_audit.py`](../src/qloop/eval/sprint2_audit.py), and [`src/qloop/model/d1_lora.py`](../src/qloop/model/d1_lora.py) + [`configs/stages/s2_2_d1_lora_control.yaml`](../configs/stages/s2_2_d1_lora_control.yaml). |
| **Post-Sprint-1 Backlog** | **Deferred** | `D1`, `D2`, `D4` | Extract `~7.9K`-line standalone string into linted module; archive `hssd.py` (`13,255` lines) + v1 configs to `qloop/legacy/`; split report prose checks into `make docs-check`. |

### 6.2 Sprint 2 — Decisive Single-Card Scientific Validation (`Sprint 2A + Closeout` DONE · `Sprint 2B` Pending)

Executed on `1× NVIDIA RTX PRO 6000 Blackwell Server Edition` (`96 GB GDDR7 ECC` [`94.97 GiB` usable], `1,597 GB/s`, `sm_120`):

| Workstream | Status | Budget | Resolves | Empirical Result / Remaining Gate |
| :--- | :---: | :---: | :---: | :--- |
| **S2.1 Leak-Free Holdout + OOD & `8K–1M` Audit** | **2A + CLOSEOUT DONE** | `~8.1` GPU-h used (`~6h` 2A + `2.09h` closeout) | `A1`, `A3`, `B2`, `B3`, `B4` | **Done (`§7.2`, `§7.3`, `e1cf41b`, `rc=0`):** `s2_v2@600` beats Stock `-10.71%` PPL (`+469σ`) on `0/4`-overlap holdout, `-1.11%` PPL (`+33.6σ`) on unseen `wikitext-103` OOD, `+0.00050` `128K` sparse–dense gap, **`1M` Sparse NLL `0.00120`** (`11.9×` below Stock `0.01429`, `17.1×` below `s3_v2` `0.02048`), **`20/20 (100.0%)` `8K–1M` RULER [20]** (vs. Stock `9/20`), `GDN-2` `max_l2 = 191.21` (`0` NaNs), and **`65.19 GiB` peak VRAM** (`0` OOM warnings). |
| **S2.3 Causal Ablations + Narrow `K0` Oracle + Live TF `5.24.1`** | **2A + CLOSEOUT DONE (`4/4 PASS`)** | `~2.1` GPU-h used | `B1`, `B5`, `QL60-001/002/005`, `LSA-NL` gate | **Done (`§7.4`–`§7.6`, `e1cf41b`, `rc=0`):** `GDN-2` (`+156.6σ`), `PassRouter` (`+6.8σ`), & `lsa_sparse` (`0.8360` recall) are `KEEP_CRITICAL`; `QED` (`-0.31σ`) & `neg_eigval` (`-0.97σ`) are `PRUNE_CANDIDATE`; `K0` = **`YELLOW_B_BLOCK_ATTN_CALIB`** (`100.0%` EM retention @ `6.05×` smaller KV, `+0.6228`-nat NLL gap); TF `5.24.1` **`4/4 PASS`** (`217.6s`: `ENV`, `QL60-001`, `QL60-002` `d = 1, 2, 3`, `QL60-005` `13/13`). |
| **S2.2 Parameter-Matched `d=1` LoRA Control (Sprint 2B)** | **PENDING — GATE #1** | `~18` GPU-h | `A2` | Train `~18.41M` `d=1` (`40L`) LoRA baseline (`src/qloop/model/d1_lora.py`, `configs/stages/s2_2_d1_lora_control.yaml`) on `[12, 36)` under identical hash-partitioned `s1` data/schedule. **Decisive Gate (#1):** Looped `d >= 2` must match or beat `d=1` LoRA on leak-free holdout + unseen OOD before `s4_v2_rl`. |

### 6.3 Explicit Go / No-Go Unfreeze Criteria (Current Gate Status)

- **Unfreeze `s4_v2_rl` (SI-RSI GRPO):** Sprint 1 (`100%` **PASS**), S2.1 `4K–1M` (**PASS**), S2.3 (**PASS**), and live TF `5.24.1` `4/4` (**PASS**) are closed; requires **Sprint 2B `S2.2` `d=1` LoRA control (PENDING — Gate #1)**. Because `s2_v2@600` unseen OOD `d=1` (`1.69971`) slightly edged `d=3` (`1.70122`), `S2.2` is strictly binding: if looped `d >= 2` does not beat parameter-matched `d=1` LoRA, pause RL and iterate on a low-anchor recurrence training curriculum (`McLeish et al.` [14]) or ship a pruned `d=1` long-context `GDN-2 + D-LSA` backbone (pruning `QED` and `neg_eigval`).
- **Unfreeze `LSA-NL` / `CoMem` (`C2–C6`) Training:** Requires all `s4_v2_rl` prerequisites **plus** upgrading `K0` from `YELLOW_B_BLOCK_ATTN_CALIB` (`+0.6228`-nat NLL gap) to `GREEN` (`<= 0.050`-nat gap) via lightweight `2–4h` single-GPU Block-Attention APOLLO [7] calibration on `D-LSA` indexers + global `q/k/v/o` adapters (keeping `32.212B` MoE 100% frozen).
- **Unfreeze Public Whole Model Pack Release:** Live `transformers 5.24.1` `4/4 PASS` (`QL60-001`, `QL60-002`, `QL60-005`) is **CLOSED** (`e1cf41b`); requires `S2.2` (**Gate #1**), **downstream generative reasoning benchmarks (`AIME 2026` / `GPQA Diamond` / `ARC-AGI-2`, Gate #2)**, and **100% exclusion of `opus46_*` restricted-provenance datasets** (`§5`).

---

## 7. Sprint 2A + GPU Closeout Empirical Results (`S2.1` `4K–1M`, `S2.3`, `K0`, Live `transformers 5.24.1` `4/4 PASS`)

All results in this section were measured directly on `1× NVIDIA RTX PRO 6000 Blackwell Server Edition` (`96 GB GDDR7 ECC` [`94.97 GiB` usable], `1,597 GB/s`, `sm_120`) running Python `3.13.16`, PyTorch `2.14.1+cu130`, and `transformers 5.24.1` (Sprint 2A base pass from `a94c163` plus **Sprint 2A GPU Closeout `rc=0`** on `e1cf41b`; artifacts in `runs/sprint2a/{sprint2_audit_8k_128k.md,sprint2_audit_512k_1m.md,kill_gate_k0.md,serving_audit_v2.md,closeout_supervisor.log}`).

### 7.1 Final `s2_v2_joint_bptt@600` Completion (`17382c2e166d`)

- **Hub Checkpoint:** `TassP0lar/Qwen3.6-35B-A3B-Loop64` commit `17382c2e166d` (`2026-11-26T03:14:49Z`) [12].
- **Training & Hardware Envelope:** `600/600` steps, `157,286,400` (`157.29M`) tokens, `156,941,238` trainable parameters (`32.212B` routed MoE 100% frozen), `69.97 GiB` peak VRAM on `1× NVIDIA RTX PRO 6000 Blackwell Server Edition` (`96 GB GDDR7 ECC`, `1,597 GB/s`).
- **Final Architectural Telemetry:** `U = 18.19` unique core experts/token (`1.764×` effective expert mult, **`3.716B`** unique consulted parameters/token), `gdn2/adapter_norm = 0.845`, `decay_horizon_mult_max = 7.60×`, `qed_lambda = 0.00085`, `neg_eigval = 0.99986`.

### 7.2 Leak-Free SHA-1 Holdout (`4K`) & Pure Unseen OOD (`wikitext-103-raw-v1`) Depth Sweep (`S2.1`)

Evaluated with pure valid-token LM cross-entropy (`3` repeats, seed `42`, `eval_seq_len = 4096`) on: (i) **Leak-Free SHA-1 Holdout** (`0/4` document overlap verified against `13` training docs, `8` batches, `19,093` valid loss tokens, SHA-256 `d359f35a91560494`), and (ii) **Pure Unseen OOD** (`wikitext-103-raw-v1` test, `8` batches, `7` docs, `30,030` valid loss tokens, SHA-256 `0fb192cb5d2b1952`).

| Checkpoint Arm | Holdout `d=1 → d=2 → d=3` NLL (`PPL@d_max`) | Holdout `d1→d3` (`z`) | Holdout vs. Stock (`z`) | Unseen OOD `d=1 → d=2 → d=3` NLL (`PPL@d_max`) | OOD `d1→d3` (`z`) | OOD `d_max` vs. Stock (`z`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Stock `Qwen3.6 + YaRN 4×`** (`d=1`) [1], [9] | `0.95000` (`2.586`) | — | Baseline | `1.71242` (`5.542`) | — | Baseline |
| **`s1_loop_a6b@1000`** | `0.94292 → 0.89579 → 0.87588` (`2.401`) | `+0.06704` (`+163σ`) | `-7.14%` PPL (`+198σ`) | `1.71089 → 1.70909 → 1.70439` (`5.498`) | `+0.00649` (`+13.2σ`) | `-0.80%` PPL (`+20.7σ`) |
| **`s3_v2_long_1m_lsa@600`** | `0.94320 → 0.87827 → 0.84992` (`2.339`) | `+0.09327` (`+314σ`) | `-9.52%` PPL (`+368σ`) | `1.71055 → 1.70916 → 1.70284` (`5.490`) | `+0.00771` (`+38.4σ`) | `-0.95%` PPL (`+48.8σ`) |
| **`s2_v2_joint_bptt@600`** | `0.87281 → 0.84861 → 0.83671` (`2.309`) | `+0.03609` (`+309σ`) | **`-10.71%` PPL (`+469σ`)** | `1.69971 → 1.70360 → 1.70122` (`5.481`) | `-0.00151` (`-4.2σ`) | **`-1.11%` PPL (`+33.6σ`)** |

- **Key Scientific Reading:** All three Clean v2 stages beat **Stock `Qwen3.6 + YaRN 4×`** by `> 20σ` on both leak-free holdout and unseen OOD (`2× d=1` nondeterminism `max_diff` bounded within `2.36e-4 .. 1.18e-3`). However, while `s1@1000` (`+13.2σ`) and `s3_v2@600` (`+38.4σ`) show monotonic `d=1 → d=3` gains on unseen OOD, `s2_v2@600` achieves its best OOD NLL at `d=1` (`1.69971`, PPL `5.472`) slightly ahead of `d=3` (`1.70122`, PPL `5.481`, `-4.2σ`). This directly motivates running Sprint 2B (`S2.2`) `d=1` LoRA control before unfreezing `s4_v2_rl`.

### 7.3 `8K–1M` Ultra-Long Context Ladder, 4-Needle RULER & `GDN-2` `1M` State Norms (`fp8` KV)

Measured under `kv_cache_quant=fp8` across `8K / 64K / 128K` (Sprint 2A, `sprint2_audit_8k_128k.md`) and `512K / 1M` (`524,288 / 1,048,576`, Sprint 2A Closeout on `e1cf41b`, `rc=0`, `sprint2_audit_512k_1m.md`, geometric `_FP8Slot`, zero-FP32 BF16 dequant, `--skip-dense-above 131072`, peak VRAM **`65.19 GiB` = `66,759 / 97,887 MiB`**, **`0` OOM warnings**, `all_states_bounded=True`, `0` NaNs across all runs):

| Checkpoint Arm (Depth) | `8K` Dense / Sparse NLL | `64K` Dense / Sparse NLL | `128K` Dense / Sparse NLL | `128K` Sparse−Dense Gap | 4-Needle RULER (`8K, 64K, 128K`) | `GDN-2` `max_l2` (`8K → 64K → 128K`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Stock `Qwen3.6 + YaRN 4×`** (`d=1`) | `0.49777` / `N/A` | `0.15114` / `N/A` | `0.10026` / `N/A` | `N/A` | `0/4, 1/4, 1/4` (`2/12`) | `182.6 → 184.7 → 185.2` (`0` NaNs) |
| **`s1_loop_a6b@1000`** (`d=3`) | `0.16217` / `N/A` | `0.05358` / `N/A` | `0.03030` / `N/A` | `N/A` | `3/4, 4/4, 0/4` (`7/12`, collapses @ `128K`) | `182.6 → 184.6 → 185.2` (`0` NaNs) |
| **`s3_v2_long_1m_lsa@600`** (`d=3`) | `0.26157` / `0.28208` | `0.08195` / `0.17592` | `0.04913` / `0.11370` | `+0.06457` | **`4/4, 4/4, 4/4` (`12/12`, `100%`)** | `182.6 → 184.6 → 185.2` (`0` NaNs) |
| **`s2_v2_joint_bptt@600`** (`d=3`) | `0.10123` / `0.10728` | `0.01362` / `0.01557` | **`0.00746` / `0.00796`** | **`+0.00050`** (**`129×` tighter**) | **`4/4, 4/4, 4/4` (`12/12`, `100%`)** | **`186.2 → 191.4 → 191.0`** (`0` NaNs) |

| Checkpoint Arm (Depth, `512K / 1M` Closeout) | `512K` NLL & `lsa/last_recall` | `512K` RULER EM (`ans_nll`) & Time | `1M` (`1,048,576`) NLL & `lsa/last_recall` | `1M` RULER EM (`ans_nll`) & Time | `GDN-2` `max_l2` / `max_abs` (`512K → 1M`) | `8K–1M` Total RULER |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Stock `Qwen3.6 + YaRN 4×`** (`d=1`, dense) | `0.02454` (dense) · `N/A` | `3/4 (75.0%)` (`0.1827`) · `619.7s` | `0.01429` (dense) · `N/A` | `4/4 (100.0%)` (`0.2022`) · `2,231.3s` | `184.65 / 26.70 → 185.42 / 27.39` (`0` NaNs) | **`9/20 (45.0%)`** |
| **`s3_v2_long_1m_lsa@600`** (`d=3`, sparse `D-LSA`) | `0.03970` (sparse) · `0.0534` | **`4/4 (100.0%)`** (`0.2473`) · `558.3s` | `0.02048` (sparse) · `0.0333` | **`4/4 (100.0%)`** (`0.1925`) · `1,557.9s` | `184.59 / 26.60 → 185.35 / 27.36` (`0` NaNs) | **`20/20 (100.0%)`** |
| **`s2_v2_joint_bptt@600`** (`d=3`, `GDN-2` + sparse `D-LSA`) | **`0.00232`** (sparse) · `0.0627` | **`4/4 (100.0%)`** (`0.3088`) · `628.3s` | **`0.00120`** (sparse) · `0.0317` | **`4/4 (100.0%)`** (`0.3233`) · **`1,688.8s`** | **`191.38 / 27.23 → 191.21 / 27.74`** (`0` NaNs) | **`20/20 (100.0%)`** |

- **Ultra-Long `512K / 1M` Closeout Breakthrough (`e1cf41b`, `rc=0`, `7,513.0s`):** Replacing dynamic concatenation with geometric `_FP8Slot` pre-allocation, zero-FP32 BF16 dequantization, and `--skip-dense-above 131072` (`7346f77`) cut peak `1M` VRAM by `29.24 GiB` (`94.43 → 65.19 GiB` [`66,759 / 97,887 MiB`], **`0` OOM warnings**). Across `8K–1M`, `s2_v2_joint_bptt@600` achieves **`20/20 (100.0%)`** 4-needle RULER [20] recall (matching `s3_v2@600` `20/20`, vs. Stock `9/20 = 45.0%` and `s1` collapsing to `0/4` at `128K`), cuts `128K` Dense NLL **`13.4×`** below Stock (`0.00746` vs. `0.10026`), cuts `1M` (`1,048,576`) Sparse NLL to **`0.00120`** (**`11.9×`** lower than Stock `0.01429` and **`17.1×`** lower than `s3_v2` `0.02048`), keeps `GDN-2` recurrent state norms completely flat through `1M` (`max_l2 = 191.21`, `max_abs = 27.74`, `0` NaNs), and completes `1M` in **`1,688.8s`** — **`1.32×` faster** than Stock dense `40L` (`2,231.3s`) even while executing `88L` (`d=3`).
- **Honest Scientific Caveat:** Length-ladder NLL decreases as `~1/T` because the synthetic needle-haystack evaluation repeats low-entropy filler tokens across `T`; cross-arm comparisons on the identical `seed=42` token sequence are strictly controlled and valid, but absolute `512K / 1M` NLL is **not** equivalent to natural `1M`-token book perplexity. Furthermore, `dense` attention at `512K / 1M` was skipped on `D-LSA` [6] arms (`--skip-dense-above 131072`), so the exact `sparse−dense` NLL gap is directly measured through `128K` (`+0.00050` nats on `s2_v2@600`, `129×` tighter than `s3_v2@600`'s `+0.06457`).

### 7.4 `S2.3` Zero-Training Causal Ablation Matrix (`s2_v2@600`, `d=3`, SHA-256 Verified)

Executed on leak-free `4K` holdout (`3` repeats, bit-exact SHA-256 weight restoration `17399792700dbb1a` verified `PASS` after every arm):

| Ablation Arm | Mean ± Std NLL | `Δ NLL vs. Full` | `z`-score | `lsa/last_recall` | Verdict & Engineering Decision |
| :--- | :---: | :---: | :---: | :---: | :--- |
| `full` (`s2_v2@600`, `d=3`) | `0.83719 ± 0.00035` | `+0.00000` | `+0.00σ` | `N/A` | **Reference (`KEEP_CRITICAL`)** |
| **`no_gdn2`** | `0.87606 ± 0.00003` | **`+0.03887`** | **`+156.6σ`** | `N/A` | **`KEEP_CRITICAL` (#1 causal driver)** — decoupled `GDN-2` [3] gates carry the dominant quality lift. |
| **`no_pass_router`** | `0.83970 ± 0.00039` | **`+0.00251`** | **`+6.8σ`** | `N/A` | **`KEEP_CRITICAL`** — dynamic cross-pass expert routing contributes statistically significant gain. |
| **`lsa_sparse`** vs. `lsa_dense` | `0.83730 ± 0.00033` (dense: `0.83912`) | `+0.00010` (dense: `+0.00192`) | `+0.30σ` (dense: `+5.5σ`) | **`0.8360`** | **`KEEP_CRITICAL`** — sparse `D-LSA` [6] maintains `0.8360` recall and outperforms forcing `lsa_dense` (`+5.5σ` worse). |
| **`no_qed`** | `0.83710 ± 0.00024` | **`-0.00009`** | **`-0.31σ`** | `N/A` | **`PRUNE_CANDIDATE`** — zero empirical lift; prune Bounded `QED` [4] in future stages to simplify the kernel. |
| **`no_neg_eigval`** | `0.83694 ± 0.00014` | **`-0.00026`** | **`-0.97σ`** | `N/A` | **`PRUNE_CANDIDATE`** — zero empirical lift; prune soft negative eigenvalues [5] to restore strict `[0, 1]` `GDN-2` stability bounds. |
| **`low_anchor_probe`** (`max_anchor_weight = 0.25`, `<= 0.3`) | `0.84090 ± 0.00032` | **`+0.00370`** | **`+11.05σ`** | `N/A` | **`SHORTCUT_DIAGNOSTIC`** — trained checkpoint relies on `w_anchor ≈ 0.85`; escaping shortcut collapse (`McLeish et al.` [14]) requires a training curriculum, not a zero-shot shift. |

### 7.5 Kill-Gate `K0`: Zero-Training Pre-RoPE KV-Mem Blob Injection (`s2_v2@600`, `N=20`)

Evaluated across `20` multi-session examples (`code_recall`, `codebase_qa`, `contradictory_update`, `multi_hop`, `needle`, `temporal_update`) with `sink_tokens = 4`, `boundary_header_tokens = 4`, `rotary_dim = 64`, `refresh_frac = 5%`, `green >= 92%` + `NLL gap <= 0.050` nats:

| Arm (`s2_v2@600`, `d=3`, `N=20`) | Answer NLL | NLL Gap vs. Oracle | Token Acc | Exact Match | Oracle EM Retention | KV Mounted | `temporal_update` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Arm (a) Contiguous Oracle** | `1.4803` | `+0.0000` | `0.827` | `0.350` | `100.0%` | `43.32 MB` | `100.0%` |
| **Arm (b) Naive Spliced KV** | `2.2032` | `+0.7229` | `0.793` | `0.350` | `100.0%` | `31.21 MB` | `73.8%` |
| **Arm (c) Pre-RoPE + Sink-Strip + `H64-INT4`** | **`2.1031`** | **`+0.6228`** (`-0.1001` vs. Naive) | **`0.821`** | **`0.350`** | **`100.0%`** | **`7.16 MB` (`6.05×` smaller)** | **`95.7%`** |
| **Arm (d) Arm (c) + `5%` CacheBlend (`70` tok)** | `2.0924` | `+0.6120` | `0.821` | `0.350` | `100.0%` | `7.16 MB` | `95.7%` |

- **Verdict — `YELLOW_B_BLOCK_ATTN_CALIB`:** At `d=3`, Arm (c) preserves **`100.0%` EM retention** (`0.350` vs. `0.350`) and nearly full token accuracy (`0.821` vs. `0.827`) at **`6.05×` smaller KV** (`7.16 MB` vs. `43.32 MB`), lifts `temporal_update` from `73.8% → 95.7%`, and beats Naive splicing by `-0.1001` nats (at `d=1`, Arm (c) achieves `87.5%` EM retention, `0.827` vs. `0.839` token acc, `+0.6010` NLL gap, `temporal_update 73.8% → 100.0%`). It is held back from `GREEN` solely by the frozen-projection cross-session Dynamic Error NLL gap (`+0.6228` nats; `5%` CacheBlend `+0.6120`), motivating lightweight `2–4h` single-GPU Block-Attention APOLLO [7] calibration before any `LSA-NL` training.

### 7.6 Live `transformers 5.24.1` Serving & Export Audit (`e1cf41b`, `4/4 PASS`) & Remaining 2 Open Gates

- **Live `transformers 5.24.1` Audit (`runs/sprint2a/serving_audit_v2.md`, `2026-11-26T18:38:45Z`, `217.6s` on `e1cf41b40564`, pack format `3.1`, **`4/4 PASS`**):**
  - `ENV:strict_tf5`: **PASS** (`transformers 5.24.1`, `PyTorch 2.14.1+cu130`, `Python 3.13.16`, `NVIDIA RTX PRO 6000 Blackwell Server Edition`).
  - `QL60-001:quantized_hybrid_cache`: **LIVE PASS (`47.60s`)** — verified `88` virtual layers under real `transformers 5.24.1` `CacheLayerMixin`, exact **`1.00×` KV** across `d=1` and `d=3` for `bf16` (`143,360 B`), `fp8` (`72,240 B`, `0.5039×`; unit-cache contract `3,168 / 6,144 B = 0.5156×`), and `hadamard_int4` (`38,080 B`, `0.2656×`), and exact `2.00×` when `kv_share=False`.
  - `QL60-002:pack_model_and_generate`: **LIVE PASS (`19.01s`)** — BOTH in-package and standalone `Qwen36LoopHSSDForCausalLM` passed token-for-token `.generate(max_new_tokens=8)` parity across `d = 1, 2, 3` (`40L / 64L / 88L`), `logits_to_keep=0` shape `[1, 16, 248320]`, `Argmax Match = True`, `Repeat Diff = 1.16–1.79` (`1.79 / 1.16 / 1.53`) vs. `Max Diff = 1.25–2.28` (`2.28 / 1.36 / 1.25`).
  - `QL60-005:stage_completion_gate`: **LIVE PASS (`13/13`, `0.002s`)** — all acceptance/rejection checks on `_local_stage_done()` passed.
- **Explicit Remaining Open Gaps (Now Only 2 Core Gates; Zero Fake Claims):**
  1. **Sprint 2B (`S2.2` / `A2`) `~18.41M` Parameter-Matched `d=1` LoRA Control:** `src/qloop/model/d1_lora.py` + `configs/stages/s2_2_d1_lora_control.yaml` are merged and CPU-verified (`63d8b00`), awaiting `~18` GPU-h execution — the **decisive gate** before unfreezing `s4_v2_rl` or `LSA-NL`.
  2. **Downstream Generative Reasoning Benchmarks:** End-to-end generative evaluations on `AIME 2026`, `GPQA Diamond`, and `ARC-AGI-2` remain unexecuted until `S2.2` passes and `s4_v2_rl` unfreezes. *(Separately, future `LSA-NL`/`CoMem` training expansion also requires upgrading `K0` from `YELLOW_B_BLOCK_ATTN_CALIB` to `GREEN` via `2–4h` Block-Attention calibration, `§7.5`.)*

---

## References

[1] Qwen Team, "Qwen3.6-35B-A3B," Alibaba Cloud, Model Release, Apr. 16, 2026. [Online]. Available: https://huggingface.co/Qwen/Qwen3.6-35B-A3B

[2] S. Yang, J. Kautz, and A. Hatamizadeh, "Gated Delta Networks: Improving Mamba2 with Delta Rule," in *Proc. Int. Conf. Learn. Represent. (ICLR)*, 2025, arXiv:2412.06464.

[3] A. Hatamizadeh, Y. Choi, and J. Kautz, "Gated DeltaNet-2: Decoupling Erase and Write in Linear Attention," *arXiv preprint arXiv:2605.22791*, May 2026.

[4] D. Gupta, A. Das, and D. Gupta, "The Query Knows What to Forget: A Second Erase Direction for Linear Attention," *arXiv preprint arXiv:2608.13668*, Aug. 2026.

[5] R. Grazzi, J. Siems, A. Zela, J. K. H. Franke, F. Hutter, and M. Pontil, "Unlocking State-Tracking in Linear RNNs Through Negative Eigenvalues," in *Proc. Int. Conf. Learn. Represent. (ICLR)*, 2025, arXiv:2411.12537.

[6] A. Kubicek, A. Caciolai, L. Peng, S. Sumant, B. Hoover, F. Greenlee, R. M. Battleday, R. Urban, T. Hoefler, and J. C. R. Whittington, "Latent Sparse Attention: Towards Subquadratic Ultra-Long Context Attention," *arXiv preprint arXiv:2610.42147*, Oct. 2026.

[7] H. Zhu, Z. Zhang, W. Cong, X. Liu, S. Park, V. Chandra, B. Long, D. Z. Pan, Z. Wang, and J. Lee, "APOLLO: SGD-like Memory, AdamW-level Performance," *arXiv preprint arXiv:2412.05270*, Dec. 2024.

[8] Z. Xie *et al.* (DeepSeek-AI), "mHC: Manifold-Constrained Hyper-Connections," *arXiv preprint arXiv:2512.24880*, Dec. 2025.

[9] B. Peng, J. Quesnelle, H. Fan, and E. Shippole, "YaRN: Efficient Context Window Extension of Large Language Models," in *Proc. Int. Conf. Learn. Represent. (ICLR)*, 2024, arXiv:2309.00071.

[10] S. Yang and Y. Zhang, "FLA: A Triton-Based Library for Hardware-Efficient Implementations of Linear Attention Mechanisms," GitHub Repository (fla-org/flash-linear-attention, v0.6.0), 2024–2026.

[11] Thinking Machines Lab, "Inkling 2 Mini," Technical Release, Nov. 20, 2026. [Online]. Available: https://thinkingmachines.ai

[12] TassP0lar, "Qwen3.6-35B-A3B-Loop64: Open Looped Hybrid Checkpoints and Empirical Telemetry," Hugging Face Model Hub, Nov. 2026. [Online]. Available: https://huggingface.co/TassP0lar/Qwen3.6-35B-A3B-Loop64

[13] TassP0lar, "Loop64-Fable-SIRSI-v1: Curated Anti-Slop 3-Beat Looped-CoT, Aesthetic Craft, and SI-RSI Recovery Dataset," Hugging Face Datasets Hub (v1.2), Nov. 2026. [Online]. Available: https://huggingface.co/datasets/TassP0lar/Loop64-Fable-SIRSI-v1

[14] S. McLeish, A. Li, J. Kirchenbauer, D. S. Kalra, B. R. Bartoldson, B. Kailkhura, A. Schwarzschild, J. Geiping, T. Goldstein, and M. Goldblum, "Teaching Pretrained Language Models to Think Deeper with Retrofitted Recurrence," *arXiv preprint arXiv:2511.07384*, Nov. 2025.

[15] S. Bae, Y. Kim, R. Bayat, S. Kim, J. Ha, T. Schuster, A. Fisch, H. Harutyunyan, Z. Ji, A. Courville, and S.-Y. Yun, "Mixture-of-Recursions: Learning Dynamic Recursive Depths for Adaptive Token-Level Computation," *arXiv preprint arXiv:2507.10524*, Jul. 2025.

[16] J. Geiping, S. McLeish, N. Jain, J. Kirchenbauer, S. Singh, B. R. Bartoldson, B. Kailkhura, A. Bhatele, and T. Goldstein, "Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach," *arXiv preprint arXiv:2502.05171*, Feb. 2025.

[17] R.-J. Zhu, Z. Wang, K. Hua, T. Zhang, Z. Li, *et al.*, "Scaling Latent Reasoning via Looped Language Models," *arXiv preprint arXiv:2510.25741*, Oct. 2025.

[18] E. Guha *et al.*, "OpenThoughts: Data Recipes for Reasoning Models," *arXiv preprint arXiv:2506.04178*, Jun. 2025.

[19] A. Bercovich *et al.*, "Llama-Nemotron: Efficient Reasoning Models," *arXiv preprint arXiv:2505.00949*, May 2025.

[20] C.-P. Hsieh, S. Sun, S. Kriman, S. Acharya, D. Rekesh, F. Jia, Y. Zhang, and B. Ginsburg, "RULER: What's the Real Context Size of Your Long-Context Language Models?" *arXiv preprint arXiv:2404.06654*, Apr. 2024.
