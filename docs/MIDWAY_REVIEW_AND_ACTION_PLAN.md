# Mid-Way External Review Cross-Verification Ledger & 2-Sprint Remediation Plan (`v1.0`)

> **Language Companion:** [简体中文版 (`docs/MIDWAY_REVIEW_AND_ACTION_PLAN_ZH.md`)](MIDWAY_REVIEW_AND_ACTION_PLAN_ZH.md)  
> **Audited Repo Baseline:** `0f5d7f7` (two independent external audits) · **Current Repo HEAD:** `17216c0`  
> **Live Hub Telemetry Audited:** `s2_v2_joint_bptt@450` (commit `a5b20b20`, `2026-11-25T17:54:48Z`, `step=450/600`, `117,964,800` tokens, `n_params=156,941,238`) & `s2_v2@400` (commit `cec24a8d`)  
> **Target Hardware:** `1× NVIDIA RTX PRO 6000 Blackwell Server Edition` (`96 GB GDDR7 ECC` [`94.97 GiB` usable], `1,597 GB/s`, `sm_120`)  
> **Strategic Status:** **FREEZE** further `LSA-NL`/`CoMem` training expansion, `s4_v2_rl`, and public Whole Model Pack release until **Sprint 1** (CPU engineering fixes) and **Sprint 2** (`~25–35` GPU-h leak-free validation) pass.

---

## 1. Executive Mid-Way Verdict

Two independent mid-way external audits evaluated this repository at commit `0f5d7f7` alongside live Hugging Face Hub checkpoint metadata:
1. **Review 1 (`GPT-6 Astra Pro` Code & Systems Audit @ `0f5d7f7`):** Identified `10` concrete code, evaluation, serving, and orchestration defects (`QL60-001` through `QL60-010`) plus parameter-accounting inconsistencies (`§6`).
2. **Review 2 (Research & Codebase Audit @ `0f5d7f7` + live Hub `s2_v2@450` `manifest.json` `a5b20b20`):** Identified `16` scientific, methodological, telemetry, and hygiene findings grouped into `A1–A3` (scientific validity), `B1–B5` (claims & telemetry), `C1–C3` (loss/indexer/router correctness), and `D1–D5` (maintainability & provenance).

We independently re-verified **every single finding** against current `main` (`17216c0`, which merged isolated Phase-2A/2B `LSA-NL` primitives under `src/qloop/memory/` and bilingual specs without altering active training codepaths) and live `s2_v2@450` Hub telemetry (`a5b20b20`). **All `10/10` Review 1 findings and `16/16` Review 2 findings are confirmed.**

### What Holds Up (Validated Infrastructure)
- **Single-GPU Blackwell `sm_120` Systems Stack:** Retrofitting `Qwen/Qwen3.6-35B-A3B` [1] (`30` native Gated DeltaNet [2] + `10` gated full-attention layers, `32.212B` routed MoE weights 100% frozen and SHA-256 verified) runs stably on one `NVIDIA RTX PRO 6000 Blackwell Server Edition` (`96 GB GDDR7 ECC` [`94.97 GiB` usable], `1,597 GB/s`, `sm_120`), combining `GDN-2-1M` [3], Bounded `QED` [4], soft negative eigenvalues [5], per-pass `D-LSA` [6], low-rank `APOLLO` [7], `mHC` `LoopBridge` [8], pure `4.0×` YaRN [9], and `fla 0.6.0` DPLR Triton kernels [10].
- **Verified Engineering Milestones:** `574898d` `4×8K` macro-segmenting cuts `T=32K` `GDN2Adapter` transient VRAM `3.28×` (`23.051 → 7.032 GiB`, keeping `4×32K d=3` peak at `67.194–67.77 GiB`); `1M` `FP8` `1.0×` `SharedKV` analytical state is `10.65 GiB` (`3.02×` smaller than Inkling 2 Mini's `32.16 GiB` [11]); the negative ablation retiring `HSSD v3.2` was caught cleanly via 4-stage causal swapping; and open checkpoints [12] and audited `v1.2` data artifacts [13] remain fully traceable.

### What Failed Audit (Why We Freeze Now)
- **Scientific & Evaluation Blockers:** `eval_quick` samples from the same unpartitioned stream as training (`QL60-003` / `A1`), `depth_gain` lacks a parameter-matched `d=1` LoRA control (`A2`), and live `s2_v2@450` (`a5b20b20`) exhibits only `eval/depth_gain = +0.0062` while suffering a silent LR-schedule reset on weights-only Hub resume (`QL60-004`, `lr_factor = 0.9738`), unnormalized/KL-polluted BPTT logging (`QL60-009` / `C1` / `C2`, `loss/lm = 3.8421`), and dead sparse recall telemetry (`B2`, `lsa/last_recall = 0.0`).
- **Correctness, Serving & Provenance Blockers:** Sparse `lsa_attention()` drops caller masks (`QL60-007`), unshared `kv_share=False` reuses stale Pass-1 `BucketIndex` caches (`QL60-008`), `QuantizedHybridCache` and exported Whole Model Pack `.generate()` fail under modern `transformers 5.24+` (`QL60-001`, `QL60-002`), `_local_stage_done()` can ship incomplete packs (`QL60-005`), default orchestration still invokes retired HSSD stages (`QL60-006`), and third-party `opus46_*` subsets require explicit restricted-provenance exclusion before any public release (`D5`).

> [!IMPORTANT]
> **Binding Mid-Way Freeze Rule:** Further `LSA-NL` / `CoMem` training expansion, `s4_v2_rl`, and public Whole Model Pack export are **strictly frozen** until **Sprint 1** (CPU bug fixes across Tracks A–D) and **Sprint 2** (`~25–35` GPU-h leak-free holdout + `d=1` LoRA control + causal ablation suite) pass all acceptance gates in Section 6.

---

## 2. Exhaustive Cross-Verification Ledger

### 2.1 Live Hub Telemetry Ground-Truth Snapshot (`s2_v2_joint_bptt@450`, Commit `a5b20b20`)

Every live claim in Tables 2.1 and 2.2 traces directly to Hub commit `a5b20b20` (`2026-11-25T17:54:48Z`, preceded by `s2_v2@400` commit `cec24a8d` with identical `eval/depth_gain = 0.0062`) [12]:

| Metric Group | Live `s2_v2@450` (`a5b20b20`) Telemetry | Diagnostic Significance |
| :--- | :--- | :--- |
| **Run & Optimizer** | `step=450/600`, `tokens_seen=117,964,800`, `n_params=156,941,238`, `has_optim=false`, `lr_factor=0.9738` (`optim/lr_factor=0.9747`), `global_grad_norm=0.4753`, `clipped_params=0/370` | Proves `QL60-004` live: fresh-session weights-only resume after `step 340` reset cosine schedule back near peak LR (`0.974`). |
| **Eval & Loss** | `eval/loss_d1=0.5026` $\to$ `d2=0.4999` $\to$ `d3=0.4965`, **`eval/depth_gain=0.0062`**; train **`loss/lm=3.8421`** (incl. summed KL!) | Tiny `0.0062`-nat depth lift on unpartitioned eval (`A1`); `loss/lm=3.8421` conflates CE + summed `lsa_kl` (`QL60-009` / `C1` / `C2`). |
| **Loop & Routing** | `bridge_w_anchor=0.831/0.866`, `bridge_w_h_prev=0.057/0.050`, `gate_abs_mean=0.022/0.058`, `router_repel=-0.047`, `Jaccard p12=0.080 / p13=0.438`, `unique_experts_core=14.96`, `effective_expert_mult=1.522×` | Anchor stays pinned near `0.85` with small exit gates (`0.022–0.058`) and Pass-3 drift back toward Pass 1 (`p13=0.438` vs `p12=0.080`). |
| **GDN-2 & D-LSA** | `adapter_norm=0.741`, `erase_write_l1=0.0419`, `decay_horizon_mult_max=4.545×`, `qed_lambda_abs_mean=0.00071`, `neg_eigval_scale_mean=0.9998`, **`lsa/last_recall=0.0`** | Base GDN-2 channel gates adapt cleanly, while QED (`0.00071` vs `0.35`) & neg-eigval (`0.9998`) stay near identity (`B5`) and `last_recall` is blind in sparse mode (`B2`). |

### 2.2 Review 1 Findings Ledger (`GPT-6 Astra Pro`, `QL60-001`..`QL60-010`)

| ID | Cross-Ref | Status | Verified Code / Telemetry Evidence (`17216c0` & `a5b20b20`) | Target | Acceptance Gate |
| :--- | :---: | :---: | :--- | :---: | :--- |
| **`QL60-001`** | — | **Confirmed** | [`src/qloop/infer/engine.py:274-301`](../src/qloop/infer/engine.py#L274-L301): `QuantizedHybridCache.__init__` catches only `TypeError` on `super().__init__()`, crashing under `transformers 5.24+` `ValueError` and lacking full hybrid `Cache` methods; `update()` (`:324-337`) dequantizes full K/V on every read. | Sprint 1 Track C | Init + hybrid cache contract (`conv_states`, `recurrent_states`, `reorder_cache`, `crop`, `reset`, `get_mask_sizes`) passes under simulated TF `5.24+` across `native`/`fp8`/`hadamard_int4`. |
| **`QL60-002`** | `D1` | **Confirmed** | [`src/qloop/infer/hf_custom_modeling.py:372,685-735,8740-8765`](../src/qloop/infer/hf_custom_modeling.py#L685-L735): `CausalLMPackOutput` lacks `ModelOutput` contract; outer exported config omits effective virtual `layer_types`; `logits_to_keep=0` slices `h_last[:, -0:, :]` returning `(B, 0, V)`. | Sprint 1 Track C | `ModelOutput`-compatible output, synced virtual `layer_types`, `logits_to_keep=0` returning `(B, T, V)`, and standalone codegen `.generate()` parity pass unit tests. |
| **`QL60-003`** | **`A1`** | **Confirmed Live** | [`src/qloop/train/trainer.py:250-267`](../src/qloop/train/trainer.py#L250-L267) & [`src/qloop/data/mixture.py:62-160`](../src/qloop/data/mixture.py#L62-L160): `_eval_set()` builds from identical `data.sources` with only a seed offset (`self.seed + 7_777_777`), leaking training documents into `eval/loss_d*` (`0.5026` live). | Sprint 1 Track A + Sprint 2 | Pre-pack SHA-1 document hash partitioning (`split_role="train"` / `"eval"`) yields exact `0` doc overlap on 8-doc repro; Sprint 2 re-scores all checkpoints + OOD. |
| **`QL60-004`** | — | **Confirmed Live** | [`src/qloop/train/trainer.py:209,324-342`](../src/qloop/train/trainer.py#L324-L342) & [`src/qloop/train/checkpoint.py:121-156`](../src/qloop/train/checkpoint.py#L121-L156): `self.rng` not saved in checkpoints; when `optim.pt` is absent (`has_optim=false`), `self.opt.step_count` stays `0`, causing live `s2_v2@450` (`a5b20b20`) to jump to `lr_factor=0.9738` after `step 340` restart. | Sprint 1 Track A | `trainer.rng.getstate()` round-trips exactly on resume; weights-only resume logs explicit `resume_mode="weights_only"` and syncs `self.opt.step_count = self.step` unless opt-in reset. |
| **`QL60-005`** | — | **Confirmed** | [`src/qloop/colab.py:1133-1140,1190-1197`](../src/qloop/colab.py#L1133-L1197): `_local_stage_done()` treats mere presence of `LATEST` as completed, enabling premature Whole Model Pack export on paused/incomplete stages (`is_completed=False`). | Sprint 1 Track C | Require explicit `COMPLETED` / `PROMOTED` or verified `step >= target_steps`; Appendix F paused `step000001` fixture returns `all_stages_completed=False, pack_shipped=False`. |
| **`QL60-006`** | `D2`, `D3` | **Confirmed** | [`src/qloop/colab.py:1148-1155,1200-1201`](../src/qloop/colab.py#L1148-L1155) & [`src/qloop/colab_supervisor.py:48-55`](../src/qloop/colab_supervisor.py#L48-L55): Default stage list still runs retired HSSD stages (`s1b_hssd_distill`, `s3_long_1m_lsa`, `s2_joint_bptt`, `s4_rl`). | Sprint 1 Track C | Default stage tuple switched to Clean v2 (`s0_probe`, `s1_loop_a6b`, `s3_v2_long_1m_lsa`, `s2_v2_joint_bptt`, `s4_v2_rl`), keeping HSSD opt-in via `--stages`. |
| **`QL60-007`** | — | **Confirmed** | [`src/qloop/model/lsa.py:3128-3205`](../src/qloop/model/lsa.py#L3128-L3205): When `strict_mask=False` (default), sparse `lsa_attention()` ignores caller-supplied `attention_mask`, leaking padded/cross-doc keys (`max_err = 79.2` in Review 1 repro). | Sprint 1 Track B | Boolean & additive float 2D/3D/4D masks enforced across sink, window, and bucket keys with zero-overhead `None` fast path; left/right/block-diag repro `< 1e-6`. |
| **`QL60-008`** | — | **Confirmed** | [`src/qloop/model/lsa.py:577-615`](../src/qloop/model/lsa.py#L577-L615): `BucketIndex` cache key omits pass index (`("pkv", pkv_id, layer_id, ...)` / `("loop", id(loop_ctx), layer_id, ...)`), reusing Pass-1 index on Pass-2 when `kv_share=False` (`0.927` logit error). | Sprint 1 Track B | Key index cache by `(phys_layer, cur_pass)` when `kv_share=False` while preserving physical reuse for `kv_share=True`; float64 repro `< 1e-12` for both modes. |
| **`QL60-009`** | **`C1`** | **Confirmed Live** | [`src/qloop/train/bptt.py:2045-2093`](../src/qloop/train/bptt.py#L2045-L2093): Segment backward divides by `/ float(n_segs)` instead of valid supervised token fraction (`valid_tok_s / total_valid_toks`) and merges auxiliary terms (`lsa_kl`) into reported `loss/lm` (`3.8421` live). | Sprint 1 Track A | Normalize LM CE by `valid_tok_s / total_valid_toks` (zero-valid safe); log pure `loss/lm` separately from `loss/aux`; 2-segment `1 vs 3` valid-token grad parity passes. |
| **`QL60-010`** | — | **Confirmed** | [`src/qloop/eval/runners.py:35-53,144-165`](../src/qloop/eval/runners.py#L144-L165): Resumable `depth_<d>.jsonl` cache has no checkpoint/model identity verification and silently mixes predictions across different checkpoints in the same `out_dir`. | Sprint 1 Track A | Persist canonical run fingerprint (`model_id`, checkpoint digest/mtime, `active_passes`, `config_hash`, `dataset_rev`); raise `ValueError` on mismatch unless `allow_mixed_cache=True`. |

### 2.3 Review 2 Findings Ledger (Code & Research Audit, `A1–A3`, `B1–B5`, `C1–C3`, `D1–D5`)

| ID | Severity | Status | Verified Code / Telemetry Evidence (`17216c0` & `a5b20b20`) | Target | Acceptance Gate |
| :--- | :---: | :---: | :--- | :---: | :--- |
| **`A1`** (`QL60-003`) | Critical | **Confirmed Live** | [`src/qloop/train/trainer.py:257-267`](../src/qloop/train/trainer.py#L257-L267): `eval_quick` streams from training sources without document holdout; `s2_v2@450` `eval/loss_d1=0.5026` is in-distribution and potentially overlapping. | Sprint 1 Track A + Sprint 2 | Deterministic pre-pack hash holdout (`0` overlap) + Sprint 2 OOD re-score across all checkpoints. |
| **`A2`** | Critical | **Confirmed** | No parameter-matched `d=1` (`40L`) LoRA control exists: all loop adapters (`18.41M–156.94M`) execute only at `d >= 2`, conflating depth recurrence with added adapter capacity. | Sprint 2 (`~18` GPU-h) | Train `~18.41M` `d=1` LoRA control on `[12, 36)` with identical `s1` data/schedule; looped `d >= 2` must match or beat it on leak-free holdout + OOD before `s4_v2_rl`. |
| **`A3`** | Medium | **Confirmed Live** | Step-0 depths differ by up to `6.4e-4` nats on quantized `NVFP4` checkpoint (`s1` step-0: `d1=0.99162 / d2=0.99226 / d3=0.99183`), material relative to `s2_v2@450` `depth_gain=0.0062`. | Sprint 2 | Run 2× `d=1` nondeterminism check & report `mean ± std` over `3` repeats on all Sprint 2 evals (`> 3σ` threshold for sub-`1e-2` deltas). |
| **`B1`** (`R1 §6`) | High | **Confirmed** | Earlier README/docs mixed `"~5.97B active"` / `"A6B"` with `3.472B–3.653B` unique consulted params and `6.378B` compute-equivalent apps while live `gate_abs_mean=0.022/0.058`. | Sprint 1 Track D + Sprint 2 | Canonical 4-row Parameter Accounting Table locked in `README.md` & `§3` below; router-adapter ON/OFF ablation in Sprint 2. |
| **`B2`** | High | **Confirmed Live** | [`src/qloop/model/lsa.py:3516-3547`](../src/qloop/model/lsa.py#L3516-L3547) & `s2_v2@450` (`a5b20b20`): `lsa/last_recall = 0.0` permanently in `mode="sparse"` (`s2_v2_joint_bptt.yaml:58`), giving zero visibility into sparse retrieval quality. | Sprint 1 Track B + Sprint 2 | Periodic sampled `last_recall` enabled in `mode="sparse"`; Sprint 2 measures dense vs. sparse `D-LSA` gap across `8K–1M`. |
| **`B3`** | High | **Confirmed** | No generative or NLL capability eval exists above `64K` (`7/7` Gate 2 needles at `18K/64K` only); `1M` (`10.65 GiB`) & `10M` (`53.64 GiB` host DDR5) are memory-feasibility proofs, not yet capability proofs. | Sprint 1 Track D + Sprint 2 | Scope wording corrected in `README.md` (`memory-feasible`); Sprint 2 evaluates `128K / 512K / 1M` NLL & multi-needle RULER. |
| **`B4`** | Medium | **Confirmed** | `s3_v2@600` headline (`-0.5941` nats / `44.8%` lower PPL) compares against broken HSSD `s3@600` rather than **Stock `Qwen3.6-35B-A3B + YaRN 4×`**. | Sprint 2 | Sprint 2 benchmarks `Stock Qwen3.6 + YaRN 4×` head-to-head against `s1@1000`, `s3_v2@600`, and `s2_v2@600` on identical holdout + OOD. |
| **`B5`** | Low | **Confirmed Live** | Live `s2_v2@450` (`a5b20b20`): `qed_lambda_abs_mean=0.00071` (vs `0.35` cap) & `neg_eigval_scale_mean=0.9998` (vs `[0.5, 1.5]`) barely move while base GDN-2 gates (`adapter_norm=0.741`, `decay_horizon_mult_max=4.545×`, `erase_write_l1=0.0419`) do the work; `(0, 1.5]` extends GDN-2's `[0, 1]` stability interval. | Sprint 2 | Ablate `QED` & neg-eigval ON/OFF in Sprint 2; log recurrent state norms at `1M` tokens to confirm boundedness. |
| **`C1`** (`QL60-009`) | High | **Confirmed Live** | [`src/qloop/train/bptt.py:2045-2093`](../src/qloop/train/bptt.py#L2045-L2093): `s2_v2@450` logs `loss/lm = 3.8421` with auxiliary `lsa_kl` folded in, obscuring pure LM cross-entropy progress. | Sprint 1 Track A | Return separate `ce_loss` and `aux_loss`; log `loss/lm` strictly as valid-token-mean pure CE and `loss/aux` separately. |
| **`C2`** | High | **Confirmed** | [`src/qloop/model/lsa.py:3522-3524`](../src/qloop/model/lsa.py#L3522-L3524): `LSAManager.aux_loss()` uses `.sum()` across all `10` attention layers × passes, inflating KL to `23–84` and threatening global gradient-clip saturation. | Sprint 1 Track B | Add `aux_reduction="mean"` default (supporting `"mean"` / `"sum"`) so `lsa_kl` stays `O(1)`. |
| **`C3`** | Low | **Confirmed** | [`src/qloop/model/loop.py:884-886`](../src/qloop/model/loop.py#L884-L886): `_is_probs_cache` uses fragile first-call value-sniffing (`_looks_like_probs(full)`) instead of deterministic router class/attribute detection at install time. | Sprint 1 Track B | Deterministic install-time detection on router class/attributes (value check fallback only for unknown custom test mocks) + unit test. |
| **`D1`** | High | **Confirmed** | [`src/qloop/infer/hf_custom_modeling.py:1170-9141`](../src/qloop/infer/hf_custom_modeling.py#L1170-L1218): `~7,900`-line standalone implementation lives inside `_STANDALONE_MODELING_SRC` string literal (invisible to ruff/mypy), and line `1208` (`_HAS_QLOOP`) branches behavior on whether `qloop` is importable. | Sprint 1 Track C + Post-Sprint-1 | Track C fixes `.generate()`/`logits_to_keep=0` & embeds `qloop_commit` metadata; extract string into linted module post-Sprint-1. |
| **`D2`** | Medium | **Confirmed** | [`src/qloop/model/hssd.py`](../src/qloop/model/hssd.py) (`13,255` lines) + retired v1 stages/configs sit alongside Clean v2, inflating surface area and confusion. | Post-Sprint-1 | Archive retired HSSD module & v1 configs under `qloop/legacy/` post-Sprint-1 while keeping negative-ablation regression tests intact. |
| **`D3`** | Medium | **Confirmed** | `README.md` led with retired `Loop-HSSD v3.1` (`[8, 32)`, `~5.97B`), and [`configs/base.yaml:34-37`](../configs/base.yaml#L34-L37) still carried a stale comment calling `GDN-2` `"retired"`. | **Sprint 1 Track D (Done)** | Overhauled `README.md` around Clean v2 + canonical parameter ledger; fixed stale `configs/base.yaml` comment without changing YAML keys. |
| **`D4`** | Low | **Confirmed** | [`tests/test_experiment_report.py:1125-1190`](../tests/test_experiment_report.py#L1125-L1190): asserts fixed prose substrings (`"67.194"`, `"38/38"`) inside unit tests. | Post-Sprint-1 | Split fixed report prose checks into a dedicated `make docs-check` target post-Sprint-1 so `make check` validates pure code invariants. |
| **`D5`** | Medium | **Confirmed** | Hardcoded local macOS home paths (`<user-home>/...`) in `report.py:40`, `scripts/build_sirsi_dataset.py:7,48-49`, `docs/DATASET_AUDIT_REPORT.md:4,181,186,210`, `docs/IMPLEMENTATION_PLAN.md:3`; missing `pyarrow` in `pyproject.toml` `[dev]`; third-party `opus46_*` Claude distills in training configs (`s0`, `s1`, `s2_v2`). | **Sprint 1 Tracks C & D** | Track C cleans `report.py`/`build_sirsi_dataset.py` + adds `pyarrow>=16.0` to `[dev]`; Track D cleans doc paths & locks binding `§5` provenance policy. |

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
| | `s2_v2@450` (`U=14.96`, `1.522×` mult) | **`3.472B`** | `1.9396B` + `(128 + 24 × 14.96) × 3.1457M = 1.5321B` unique routed (`3.4717B`). |
| | `s1@1000` (`U=17.36`) | **`3.653B`** | `1.9396B` + `(128 + 24 × 17.36) × 3.1457M = 1.7133B` unique routed (`3.6529B`). |
| | Theoretical max at `R=3` (`U=24`) | **`4.154B`** | `1.9396B` + `(128 + 24 × 24) × 3.1457M = 2.2146B` unique routed (`4.1542B`). |
| **3. Compute-Equivalent Applications** *(incl. untied embed+head)* | `40L` (`d=1`, `R=1`) | **`3.454B`** | Single pass (`2.946B` ex-input-embed; `2.436B` prefill GEMMs ex-embed/head). |
| | `64L` (`d=2`, `R=2`) | **`4.916B`** | `+1` core pass (`4.409B` ex-input-embed; `kv_share=True` skips `6` core `k/v` `-0.0126B` $\to$ `3.885B` prefill GEMMs). |
| | `88L` (`d=3`, `R=3`) | **`6.378B`** | `+2` core passes (`5.871B` ex-input-embed). **Caveat:** FLOP-equivalent applications do NOT equal effective reasoning lift — live `s2_v2@450` exit-gate blend magnitude is only **$\lvert g \rvert = 0.022\text{–}0.058$** (`gate_abs_mean`). |
| **4. Trainable Delta Parameters** | `s1_loop_a6b` / `s3_v2` / `s2_v2` | **`18.41M` / `24.56M` / `156.94M`** | `0.053%` / `0.071%` / `0.453%` of stored decoder parameters (`0` routed MoE parameters modified). |

**Reporting Standard:** All future documentation, paper drafts, and model cards MUST distinguish **unique parameters consulted per token (`2.947B → 3.472B–3.653B`, max `4.154B`)** from **compute-equivalent parameter applications (`3.454B / 4.916B / 6.378B`)**, and always pair compute-equivalent claims with measured exit-gate magnitude (`|g|`) and parameter-matched `d=1` controls (`A2`).

---

## 4. Literature-Grounded Diagnosis of `depth_gain = 0.0062` & `D-LSA` Recall

Live `s2_v2@450` telemetry (`a5b20b20`) exposes why `eval/depth_gain` compressed to `+0.0062` (`d1=0.5026 → d2=0.4999 → d3=0.4965`) and why `lsa/last_recall` reads `0.0`. Four compounding mechanisms—grounded directly in recent recurrence, sparse-attention, and linear-attention literature—explain the observations:

1. **Retrofitted Recurrence Shortcut Collapse (`McLeish et al.` [14], `MoR` [15], `Recurrent Depth` [16], `mHC` [8]):**
   - In `s2_v2@450`, the `mHC` `ReEntryBridge` [8] retains **`bridge_w_anchor = 0.831 / 0.866`** on the Pass-1 prelude anchor while assigning only **`bridge_w_h_prev = 0.057 / 0.050`** to the recurrent hidden state, paired with near-zero `PassExitGate` magnitudes (**`gate_abs_mean = 0.022 / 0.058`**). Simultaneously, cross-pass routing shows **`router_repel = -0.047`** and **`Jaccard p12 = 0.080` vs. `p13 = 0.438`**—Pass 2 moves away from Pass 1, but Pass 3 drifts back toward Pass 1's anchor basin.
   - As analyzed in *Teaching Pretrained Language Models to Think Deeper with Retrofitted Recurrence* (`McLeish et al., arXiv:2511.07384` [14]), converting a pretrained feedforward model into a depth-recurrent architecture with strong input-anchor injection invites a **shortcut local minimum**: the optimizer preserves the already-strong `d=1` forward path and treats Passes `2–3` as tiny residual corrections (`|g| ~ 0.02–0.06`) unless recurrence curricula and weaker anchor pinning force genuine iterative state refinement. Sprint 2 explicitly probes a low-anchor schedule (`w_anchor <= 0.3`) alongside the `A2` `d=1` LoRA control.
2. **Pretraining Scale Gap vs. Native LoopLMs (`Ouro` [17] & `Inkling 2 Mini` [11]):**
   - Native looped pretraining systems such as **Ouro** (`arXiv:2510.25741` [17]) train shared recurrent blocks from scratch across **`7.7 trillion` tokens** with an entropy-regularized adaptive-depth objective so iterative latent manipulation is baked into every weight matrix. Similarly, `Recurrent Depth` [16] pretrains over `800B` tokens, and `MoR` [15] trains adaptive token-level routing end-to-end.
   - Our retrofit updates only `0.053%–0.453%` of parameters across `~118M–131M` tokens (`~60,000×` fewer tokens than Ouro [17]) atop `32.212B` frozen single-pass experts. Against this reality, claiming genuine iterative reasoning from a `+0.0062` unpartitioned NLL delta without a parameter-matched `d=1` LoRA baseline (`A2`) and `3`-repeat noise floor (`A3`) is scientifically untenable—hence the Sprint 2 control gate.
3. **Original `D-LSA` Distillation Contract vs. Current Bucket-Mass KL & Sparse Blindness (`Original LSA` [6]):**
   - Original `D-LSA` (`arXiv:2610.42147` [6]) distills indexers using per-key rest-bucket KL with explicit temperature calibration (`calibrate_tau`), whereas [`src/qloop/model/lsa.py:1231-1268`](../src/qloop/model/lsa.py#L1231-L1268) currently optimizes bucket-mass KL at fixed `temperature = 0.1` and sums KL across all `10` layers × passes (`lsa.py:3522-3524`, `C2`).
   - Worse, in `s2_v2_joint_bptt` (`mode="sparse"`), `lsa/last_recall` is never sampled (`0.0` at `step 450`, `B2`), leaving sparse retrieval unverified during `4×32K` BPTT. Sprint 1 Track B fixes mean-reduced `aux_loss()` and sparse-mode `last_recall` telemetry immediately; Sprint 2 benchmarks dense vs. sparse `D-LSA` across `8K–1M` and tests per-key rest-bucket KL + `calibrate_tau` before any `LSA-NL` expansion.
4. **Live `QL60-004` `step 340` LR Reset & `B5` `GDN-2` Extension Utilization:**
   - Resuming `s2_v2_joint_bptt` on a fresh Colab session after `step 340` from a weights-only Hub commit (`has_optim = false`) failed to synchronize `self.opt.step_count` to `self.step`, jumping `lr_factor` back to **`0.9738`** (`optim/lr_factor = 0.9747`) at `step 450/600`. Combined with unnormalized `/ n_segs` BPTT scaling and KL-mixed `loss/lm = 3.8421` (`QL60-009` / `C1`), late-stage optimization was knocked off its cosine decay trajectory (`depth_gain` stalled identically at `0.0062` across `cec24a8d@400` and `a5b20b20@450`).
   - Meanwhile, `GDN-2` [3] base decoupled channel gates (`adapter_norm = 0.741`, `erase_write_l1 = 0.0419`, `decay_horizon_mult_max = 4.545×`) clearly wake up, whereas Bounded `QED` [4] (`qed_lambda_abs_mean = 0.00071` vs. `0.35` cap) and soft negative eigenvalues [5] (`neg_eigval_scale_mean = 0.9998`) contribute negligibly (`B5`). Because `(0, 1.5]` extends GDN-2's `[0, 1]` stability bound, Sprint 2 ablates both extensions directly and logs `1M`-context state norms.

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

### 6.1 Sprint 1 — CPU Engineering, Correctness & Hygiene Fixes (`0` GPU-h)

| Track / Branch | Findings Closed | Strictly Owned Scope | Acceptance Gate (CPU CI) |
| :--- | :---: | :--- | :--- |
| **Track A** (`fix/sprint1-eval-bptt-resume`) | `QL60-003` (`A1`), `QL60-009` (`C1`), `QL60-004`, `QL60-010` | `mixture.py`, `trainer.py`, `bptt.py`, `checkpoint.py`, `optim.py`, `runners.py`, `tests/test_sprint1_train_data_resume.py` | `0` doc overlap on 8-doc holdout repro; `1 vs 3` valid-token BPTT grad parity & pure `loss/lm` vs `loss/aux`; exact `self.rng` round-trip & weights-only `lr_factor` sync; cross-model eval cache rejection. |
| **Track B** (`fix/sprint1-lsa-correctness`) | `QL60-007`, `QL60-008`, `C2`, `B2`, `C3` | `lsa.py`, `loop.py`, `tests/test_sprint1_lsa_loop.py` | Left/right/block-diag caller masks `< 1e-6` (zero-overhead `None` path); `kv_share=False` & `True` float64 `< 1e-12`; mean-reduced `aux_loss()`; non-zero `last_recall` in `mode="sparse"`; deterministic install-time router detection. |
| **Track C** (`fix/sprint1-hf-serving-ops`) | `QL60-001`, `QL60-002`, `QL60-005`, `QL60-006`, `D5` (code) | `engine.py`, `hf_custom_modeling.py`, `export_pack.py`, `colab.py`, `colab_supervisor.py`, `report.py`, `build_sirsi_dataset.py`, `pyproject.toml`, `tests/test_sprint1_hf_serving_ops.py` | `QuantizedHybridCache` passes simulated TF `5.24+` contract; `CausalLMPackOutput` + virtual `layer_types` + `logits_to_keep=0` `(B, T, V)` + `qloop_commit` metadata; paused `step000001` returns `pack_shipped=False`; Clean v2 default stages; portable paths + `pyarrow>=16.0` in `[dev]`. |
| **Track D** (`docs/midway-review-and-cleanup`, *this PR*) | `B1`, `D3`, `D5` (docs & policy) | `MIDWAY_REVIEW_AND_ACTION_PLAN{,_ZH}.md`, `README.md`, `DATASET_AUDIT_REPORT.md`, `IMPLEMENTATION_PLAN.md`, `configs/base.yaml` | `validate_ieee_citations()` strict `1..20` (`EN == ZH`, `0` range dashes); strict `GDDR7 ECC` hardware hygiene; `0` hardcoded user paths; `0` broken links; `test_experiment_report.py` & `test_configs_cli.py` + AST + ruff green. |
| **Post-Sprint-1 Backlog** | `D1`, `D2`, `D4` | Extract `7.9K`-line standalone string into linted module; archive `hssd.py` (`13,255` lines) + v1 configs to `qloop/legacy/`; split prose checks into `make docs-check`. | Clean `make check` + `make docs-check` separation with zero regression in v2/legacy test suites. |

### 6.2 Sprint 2 — Decisive Single-Card Scientific Validation (`~25–35` GPU-h)

Executed on `1× NVIDIA RTX PRO 6000 Blackwell Server Edition` (`96 GB GDDR7 ECC` [`94.97 GiB` usable], `1,597 GB/s`, `sm_120`) **strictly after Sprint 1 merges**:

| Workstream | Budget | Resolves | Exact Protocol & Measurable Pass Gate |
| :--- | :---: | :---: | :--- |
| **S2.1 Leak-Free Holdout + OOD & `8K–1M` Capability Audit** | `~10–14` GPU-h | `A1`, `A3`, `B2`, `B3`, `B4` | Evaluate **`Stock Qwen3.6 + YaRN 4×`** [1], [9], `s1@1000`, `s3_v2@600`, and `s2_v2@600` across `d ∈ {1, 2, 3}` (`3` repeats + 2× `d=1` nondeterminism check for `A3`) at `8K / 64K / 128K / 512K / 1M`, dense vs. sparse `D-LSA`, multi-needle **RULER** [20], and `1M` GDN-2 state-norm logging. **Gate:** `> 3σ` lift over `Stock Qwen3.6 + YaRN 4×` and bounded sparse `D-LSA` gap through `1M`. |
| **S2.2 Parameter-Matched `d=1` LoRA Control** | `~18` GPU-h | `A2` | Train a `d=1` (`40L`) LoRA baseline matching `s1_loop_a6b`'s `~18.41M` trainable parameter count on `[12, 36)` under identical hash-partitioned `s1` data, seed, and schedule. **Gate:** Looped `d >= 2` matches or beats the `d=1` control on leak-free holdout + OOD (proving recurrence contributes beyond adapter capacity). |
| **S2.3 Targeted Causal Ablations + Narrow `K0` Oracle** | `~2–4` GPU-h | `B1`, `B5`, `LSA-NL` gate | Run isolated ON/OFF ablations for `GDN-2`, `QED` / neg-eigval, and `PassRouterAdapters`, a low-anchor (`w_anchor <= 0.3`) re-entry probe testing `McLeish et al.` [14] shortcut escape, and a **`30`-min narrow `K0` Pre-RoPE KV-Blob oracle** (`src/qloop/memory/kv_blob.py`). **Gate:** Prune any zero-lift component; `K0` must show non-trivial recall recovery before any `LSA-NL` training. |

### 6.3 Explicit Go / No-Go Unfreeze Criteria

- **Unfreeze `s4_v2_rl` (SI-RSI GRPO):** Requires **Sprint 1 (`100%` pass)** + **S2.1 pass** + **S2.2 pass**. If looped `d >= 2` fails to beat the parameter-matched `d=1` LoRA control (`S2.2`), pause RL and iterate on low-anchor recurrence curricula (`S2.3`) or ship a clean `d=1` long-context model rather than spending GPU hours reinforcing a shortcut-collapsed loop.
- **Unfreeze `LSA-NL` / `CoMem` (`C2–C6`) Training:** Requires all `s4_v2_rl` prerequisites **plus** a passing `30`-min **`K0` narrow KV-Blob oracle** (`S2.3`).
- **Unfreeze Public Whole Model Pack Release:** Requires all above gates, live `transformers 5.24+` `.generate()` verification (`QL60-001`, `QL60-002`, `QL60-005`), and **100% exclusion of `opus46_*` restricted-provenance datasets** (`§5`).

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
