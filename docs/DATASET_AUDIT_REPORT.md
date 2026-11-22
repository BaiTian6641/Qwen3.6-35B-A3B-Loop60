# Loop64-Fable-SIRSI-v1 — Live Streaming Dataset Audit & Triage Report

> **Target Dataset:** `TassP0lar/Loop64-Fable-SIRSI-v1` (local build validated via `--push-to-hub --dry-run`; **not pushed publicly yet** pending review)
> **Local Persisted Path:** `/Users/tianrui6641/Qwen3.6-35B-A3B-Loop60/runs/datasets/Loop64-Fable-SIRSI-v1/`
> **Pipeline Implementation:** [`src/qloop/data/sirsi_dataset.py`](../src/qloop/data/sirsi_dataset.py) · [`scripts/build_sirsi_dataset.py`](../scripts/build_sirsi_dataset.py) · [`configs/data/sirsi_golden_seeds.json`](../configs/data/sirsi_golden_seeds.json)
> **Audit Scope:** Live streaming inspection of `9,418` raw items (`1,200` rows streamed per HF source + all `1,000` ARC-AGI-2 training tasks expanding to `1,076` test pairs + `18` author-original golden seeds; `0` full-dataset downloads), live HF Hub license verification across `10` candidate repositories, and live 13-gram decontamination against `12` accessible 2026 held-out benchmark suites (`1,889,432` unique 13-grams indexed).

---

## 1. Executive Summary & Local Parquet Build Manifest

Streaming audit + 3-way triage (`KEEP` / `SALVAGE` / `DROP`) + lossless deterministic salvage + cross-source deduplication + 13-gram decontamination produced a clean, license-compliant, citation-complete local dataset ready for Stage 2/4 SFT, Track B active rewrite, and SI-RSI preference alignment:

| Local Artifact | Rows | Size (Bytes) | SHA-256 (Prefix) | Contents & Role |
| :--- | ---: | ---: | :--- | :--- |
| `data/sft_core.parquet` | **3,625** | 50,743,251 | `41d46d261d42bd77` | Verified `KEEP` rows (`1,367` raw + `2,240` deterministically salvaged + `18` author-original golden seeds). Zero unclosed `<think>`, zero refusals, zero placeholder tool args, zero 2026 eval leaks. |
| `data/salvage_queue.parquet` | **2,599** | 8,617,925 | `1a3c2f201355427c` | Valid problems/answers queued for Track B LLM synthesis (`1,298` missing Beat-2 self-audit, `1,058` raw ARC-AGI-2 verified grids without CoT, `252` `fable51` traces with placeholder-redacted tool args, `3` thin CoT). |
| `data/sirsi_preferences.parquet` | **12** | 60,196 | `8d5b776380e6b935` | Author-original `chosen`/`rejected` preference pairs (`6` SI-RSI teacher-slap recovery + `2` UI craft + `2` technical writing + `2` software aesthetics). |
| `audit_summary.json` | — | 31,890 | `b551321fab082ca6` | Full machine-readable license, triage, dedup, decontamination, and teacher-probe ledger. |
| `README.md` | — | 11,620 | `706ef334ba192025` | Publication-grade HF Dataset Card with validated YAML metadata (`configs`, `license`, `size_categories`) and complete upstream BibTeX citations. |

**`sft_core.parquet` Composition (`3,625` rows):**
- **By Domain:** `math`: `1,505` (`41.5%`) · `agentic`: `1,157` (`31.9%`) · `code`: `601` (`16.6%`) · `science`: `352` (`9.7%`) · `recovery`: `6` (`0.2%`) · `aesthetic`: `4` (`0.1%`).
- **By License:** `apache-2.0`: `2,596` (`71.6%`) · `cc-by-4.0`: `1,029` (`28.4%`). Zero copyleft (`agpl-3.0`) or `unknown` rows.
- **By Transform Stage:** `deterministic_salvage`: `2,240` (`61.8%`) · `keep_raw`: `1,367` (`37.7%`) · `golden_seed`: `18` (`0.5%`).
- **Hub Dry-Run Validation (`--push-to-hub --repo-id TassP0lar/Loop64-Fable-SIRSI-v1 --public --dry-run`):** `validated=True`, `uploaded=False`, `yaml_license="cc-by-4.0"`, `yaml_configs=["sft_core", "salvage_queue", "sirsi_preferences"]`, `5` files (`59,464,882` bytes total), secret-leak scan clean.

---

## 2. Live Upstream License Compliance & Exclusion Ledger

Every candidate dataset's license was verified live against Hugging Face Hub metadata (`HfApi.dataset_info`) and GitHub API (`arcprize/ARC-AGI-2`). Only permissive redistribution licenses (`apache-2.0`, `cc-by-4.0`, `mit`, `odc-by`) are permitted into the streaming pipeline.

| Source Key | Upstream Repository | Live Revision SHA | Declared | Live Verified | Decision | Audit Rationale |
| :--- | :--- | :--- | :---: | :---: | :---: | :--- |
| `fable51` | `MoreThought/Fable-5.1-Max-Reasoning-Filtered-10000x` | `4904ca0c0904` | `apache-2.0` | `apache-2.0` | **INCLUDED** | Permissive Apache-2.0 (`5k/train.jsonl` split; quarantined to `salvage_queue` due to redacted tool args). |
| `ot_agent_sft` | `open-thoughts/OpenThoughts-Agent-SFT-ColdStartForRL-10K` | `261a70c35c0b` | `apache-2.0` | `apache-2.0` | **INCLUDED** | Permissive Apache-2.0; cleanest real unredacted agentic corpus. |
| `openthoughts3` | `open-thoughts/OpenThoughts3-1.2M` | `61bcf9d4eb38` | `apache-2.0` | `apache-2.0` | **INCLUDED** | Permissive Apache-2.0; requires strict `think_unclosed` filtering. |
| `nemotron_post` | `nvidia/Llama-Nemotron-Post-Training-Dataset` | `ab2a40d258a6` | `cc-by-4.0` | `cc-by-4.0` | **INCLUDED** | Permissive CC-BY-4.0 (full NVIDIA attribution + BibTeX included); `math+code+science` splits sampled evenly (`400` each). |
| `opus46_3000x` | `nohurry/Opus-4.6-Reasoning-3000x-filtered` | `1cd388e9e117` | `apache-2.0` | `apache-2.0` | **INCLUDED** | Permissive Apache-2.0; still requires refusal filtering. |
| `opus46_3300x` | `Crownelius/Opus-4.6-Reasoning-3300x` | `795d673f5e05` | `apache-2.0` | `apache-2.0` | **INCLUDED** | Permissive Apache-2.0; heavily overlaps `opus46_3000x` (85.7% cross-source duplicates removed). |
| `opus46_traceinv` | `Jackrong/Claude-opus-4.6-TraceInversion-9000x` | `dcb98612aa4e` | `apache-2.0` | `apache-2.0` | **INCLUDED** | Permissive Apache-2.0; high `missing_beat2_audit` rate -> strong Track B rewrite candidate. |
| `arc_agi2_train` | `github:arcprize/ARC-AGI-2` (`training` split only) | `main` | `apache-2.0` | `apache-2.0` | **INCLUDED** | Permissive Apache-2.0; `evaluation` split strictly held out & indexed in decontaminator. |
| `agentic_fable5` | `lordx64/agentic-distill-fable-5-sft` | `9df06dd13b69` | `agpl-3.0` | `agpl-3.0` | **EXCLUDED** | Strong copyleft (`agpl-3.0`) incompatible with permissive public Apache-2.0 / CC-BY-4.0 dataset release (`0` rows streamed). |
| `fable5_distill` | `HelioAI/Fable-5-Distill-Reasoning-462x` | `ab4e69b74e7e` | `unknown` | `unknown` | **EXCLUDED** | Unspecified/unknown license on HF Hub — unsafe for public redistribution (`0` rows streamed). |

---

## 3. Live Keep / Salvage / Drop Triage Breakdown

All included HF sources were streamed up to `1,200` rows (`nemotron_post` balanced `400 math + 400 code + 400 science`), plus all `1,000` `arc_agi2_train` tasks (`1,076` test pairs) and `18` author-original golden seeds (`elapsed=194.95s` with cached 13-gram index).

| Source | License | Streamed | KEEP | Det. Salvaged → KEEP | SALVAGE (Queue) | DROP | Keep % | Mean Slop | Mean 3-Beat | Primary Drop / Salvage Drivers |
| :--- | :---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| `sirsi_golden_seeds` | `apache-2.0` | 18 | **18** | 0 | 0 | 0 | **100.0%** | `0.000` | `0.827` | Author-original verified exemplars |
| `ot_agent_sft` | `apache-2.0` | 1,200 | **1,155** | 806 | 1 | 44 | **96.2%** | `0.000` | `0.921` | Drop: `repetition_loop` (42), `contaminated:swe_bench` (2) |
| `nemotron_post` | `cc-by-4.0` | 1,200 | **1,029** | 1,029 | 42 | 129 | **85.8%** | `0.005` | `0.874` | Drop: `think_unclosed` (71), `repetition_loop` (18), `dup` (20), `contaminated:hle` (9), `contaminated:supergpqa` (9); Salvage: `missing_beat2_audit` (42) |
| `opus46_3000x` | `apache-2.0` | 1,200 | **575** | 117 | 502 | 123 | **47.9%** | `0.005` | `0.551` | Drop: `refusal_or_incomplete_prompt` (60), `dup` (36), `unclosed_code_fence` (11), `empty_response` (10), `contaminated:hle` (3); Salvage: `missing_beat2_audit` (501) |
| `opus46_traceinv` | `apache-2.0` | 1,200 | **533** | 1 | 644 | 23 | **44.4%** | `0.000` | `0.526` | Drop: `dup(opus46_3000x)` (12), `repetition_loop` (11); Salvage: `missing_beat2_audit` (644) |
| `openthoughts3` | `apache-2.0` | 1,200 | **279** | 278 | 0 | 921 | **23.2%** | `0.004` | `0.997` | Drop: `think_unclosed` (**760**), `repetition_loop` (74), `dup` (41), `unclosed_code_fence` (37), `python_syntax_error` (6), `contaminated:hle` (3) |
| `opus46_3300x` | `apache-2.0` | 1,200 | **36** | 9 | 100 | 1,064 | **3.0%** | `0.010` | `0.438` | Drop: `dup(opus46_3000x)` (**1,028**), `refusal_or_incomplete_prompt` (25), `contaminated:hle` (3); Salvage: `missing_beat2_audit` (100) |
| `fable51` | `apache-2.0` | 1,200 | **0** | 0 | 252 | 948 | **0.0%** | `0.007` | `0.681` | Drop: `dup:exact_prompt_sha256` (**915**), `dup:exact_sha256` (31); Salvage: `redacted_tool_args` (**252**) |
| `arc_agi2_train` | `apache-2.0` | 1,000* | **0** | 0 | 1,058 | 18 | **0.0%** | `0.000` | `0.000` | Drop: `dup:near_jaccard` (16), `contaminated:arc_agi_2` (2); Salvage: `missing_cot_verified_answer` (**1,058**) |
| `agentic_fable5` | `agpl-3.0` | 0 | 0 | 0 | 0 | 0 | — | — | — | **EXCLUDED** (`license_incompatible:agpl-3.0`) |
| `fable5_distill` | `unknown` | 0 | 0 | 0 | 0 | 0 | — | — | — | **EXCLUDED** (`license_incompatible:unknown`) |

*\*`arc_agi2_train` streams `1,000` JSON task files, which expand to `1,076` individual test I/O pairs (`1,058` queued + `18` dropped).*

---

## 4. Honest Source-by-Source Audit Verdicts

### Tier 1 — Immediate High-Yield Anchors (`sft_core`)
1. **`ot_agent_sft` (`96.2% KEEP`, `mean_3beat=0.921`, `mean_slop=0.000`) — BEST AGENTIC SOURCE:**
   - **Strengths:** Genuine multi-turn `terminus-2` containerized software-engineering trajectories with real unredacted bash commands, live test outputs, explicit `<think>` verification on nearly every turn, and zero sycophancy.
   - **Weaknesses:** `42` rows hit repetition loops (terminal retry thrashing) and `2` rows matched `swe_bench_pro` (dropped by 13-gram decontam).
   - **Verdict:** **Primary agentic backbone for `sft_core`.**

2. **`nemotron_post` (`math+code+science`, `85.8% KEEP`, `mean_3beat=0.874`, `mean_slop=0.005`) — BEST STEM BACKBONE:**
   - **Strengths:** Highest clean STEM yield across balanced math, code, and science splits once the `detailed thinking on` system prompt (`1,091` rows) and whitespace quirks are deterministically stripped. Deep, naturally self-verifying reasoning.
   - **Weaknesses:** `71` rows (`5.9%`) hit unclosed `<think>` truncation; `18` contamination hits (`9` HLE + `9` SuperGPQA) prove 13-gram decontam is non-optional.
   - **Verdict:** **Primary STEM backbone for `sft_core`** (CC-BY-4.0 attribution preserved per row and in Dataset Card).

### Tier 2 — Valuable With Aggressive Filtering / Track B Upgrade
3. **`openthoughts3` (`23.2% KEEP`, `mean_3beat=0.997` on survivors) — HIGH CEILING, SEVERE TRUNCATION DEFECT:**
   - **Critical Finding:** **760 of the first 1,200 rows (`63.3%`)** fail `think_unclosed` — QwQ-32B hit its generation cap inside `<think>` (often degenerating into mixed Chinese/Korean/English garble like `stages의最后entry是... offset는2`) and **never emitted `</think>` or a final answer**. Another `74` rows had repetition loops and `37` had unclosed code fences.
   - **Strengths:** The `279` rows that DO complete cleanly have the **highest 3-beat self-audit score (`0.997`)** of any upstream corpus.
   - **Verdict:** **Keep ONLY behind strict `think_unclosed` + loop + AST gates** (never stream raw into SFT!).

4. **`opus46_3000x` (`47.9% KEEP`, `41.8% SALVAGE`) & `opus46_3300x` (`3.0% KEEP`, `88.7% DROP`):**
   - **Critical Findings:**
     1. **85.7% Cross-Source Redundancy:** `1,028 / 1,200` rows of `opus46_3300x` are exact SHA-256 duplicates of `opus46_3000x` (`1,020` exact pair + `8` exact prompt). Training on both without dedup heavily overweights duplicate problems.
     2. **Disguised Refusals Survived Upstream Filtering:** Despite `opus46_3000x`'s card claiming refusals were removed, our audit caught **`60` incomplete-problem-statement refusals** in `opus46_3000x` (plus `25` more in `opus46_3300x`) where competitive-programming formatting instructions lacked the problem statement (e.g. *"However, you haven't provided the actual problem statement or question"*).
     3. **Thin Beat-2 Self-Audit:** `501` rows in `opus46_3000x` have correct solutions but linear reasoning lacking explicit verification/backtracking -> routed to `salvage_queue` for Track B 3-beat upgrade.
   - **Verdict:** **Use `opus46_3000x` as primary, deduplicate `opus46_3300x` strictly against it, and enforce refusal + 3-beat gates.**

5. **`opus46_traceinv` (`44.4% KEEP`, `53.7% SALVAGE`) — CLEAN BUT LINEAR RECONSTRUCTED CoT:**
   - **Strengths:** Remarkably low drop rate (`1.9%`), zero slop (`0.000`), crisp answers.
   - **Weaknesses:** Because traces are reconstructed via trace inversion rather than native search, **`644 / 1,200` (`53.7%`) lack Beat-2 adversarial verification** (smooth post-hoc rationalizations without edge-case checks).
   - **Verdict:** **Keep the `533` verified rows; queue the `644` linear traces for Track B Beat-2 audit enrichment.**

### Tier 3 — Quarantine to `salvage_queue` Only (Do NOT Inject Raw Into `sft_core`)
6. **`fable51` (`MoreThought/Fable-5.1-Max-Reasoning-Filtered-10000x`, `0% KEEP`, `21.0% SALVAGE`, `79.0% DROP`) — MAJOR AUDIT FINDING:**
   - **Finding A (Schema Split):** `datasets 4.5.0` fails to parse the repo's newer `'Json'` feature type, requiring direct `HfFileSystem` streaming of `5k/train.jsonl` (note: the `full` split `10k/train.jsonl` strips `<thought>` tags and `tools` completely!).
   - **Finding B (100% Placeholder-Redacted Tool Calls):** In `5k/train.jsonl`, **100% of sampled `tool_calls` have their arguments sanitized into literal angle-bracket placeholders**: `{"path": "<file_path>"}`, `{"path": "<directory_path>"}`, `{"command": "<bash_command>"}`, `{"query": "<search_query>"}`. Training raw SFT on these rows would actively poison tool calling!
   - **Finding C (Massive Prompt Duplication):** Despite the card claiming *"heavily deduplicated"*, the first `400` rows contain only **`54` unique user prompts** (one Wave-4 prompt repeats `57` times), causing `948 / 1,200` (`79.0%`) drops under exact-prompt and exact-pair deduplication.
   - **What IS Valuable:** The `252` unique surviving rows carry crisp, zero-slop (`0.007`) Fable 5.1 `<thought>` reasoning (`mean_3beat=0.681`) and `<final_answer>` engineering reports. Quarantining them into `salvage_queue.parquet` preserves their reasoning craft for Track B reconstruction without ever corrupting `sft_core`'s tool-call targets!

7. **`arc_agi2_train` (`0% KEEP`, `98.3% SALVAGE` of `1,076` test pairs, `1.7% DROP`):**
   - **Strengths:** `1,058` unique, decontaminated, syntactically valid ARC-AGI-2 training grid transformation pairs with verified ground-truth output grids (`extracted_answer` JSON stored per row for exact `grid_equal` invariance checking).
   - **Weakness / Routing:** Raw GitHub JSON contains zero `<think>` CoT traces, so all `1,058` verified pairs go cleanly into `salvage_queue.parquet` (`salvage:missing_cot_verified_answer`) for Track B 3-beat rule synthesis rather than bare grid memorization in `sft_core`.

---

## 5. Cross-Source Deduplication & 13-Gram Decontamination

### Cross-Source Deduplication Matrix
Total unique `(prompt, response)` pairs indexed: **`6,525`**; total duplicates dropped: **`2,107`** (`1,085` `dup:exact_sha256`, `1,004` `dup:exact_prompt_sha256`, `18` `dup:near_jaccard`).

| Duplicate Source ← Original Source | Duplicates Dropped | Notes |
| :--- | ---: | :--- |
| `opus46_3300x` ← `opus46_3000x` | **1,028** | 85.7% overlap between Crownelius & nohurry releases |
| `fable51` ← `fable51` | **948** | Repeated Wave-4/Wave-8 authoring prompts within `5k/train.jsonl` |
| `openthoughts3` ← `openthoughts3` | **41** | Internal exact prompt repeats |
| `opus46_3000x` ← `opus46_3000x` | **36** | Internal exact duplicates |
| `arc_agi2_train` ← `arc_agi2_train` | **16** | Multi-test-pair / near-identical training grid tasks |
| `opus46_traceinv` ← `opus46_3000x` | **12** | Cross-source prompt overlap |
| `nemotron_post` ← `openthoughts3` | **11** | Shared competition math/code prompts |
| `nemotron_post` ← `nemotron_post` | **9** | Internal repeats |
| `opus46_3300x` ← `opus46_3300x` | **6** | Internal repeats |

### 13-Gram Decontamination Against 2026 Held-Out Benchmarks
Reuses `qloop.data.decontam.Decontaminator(n=13)` with persistent disk caching at `runs/datasets/.decontam_cache/` and columnar Parquet / targeted-file projection (avoiding multi-GB image/tarball downloads). Total index: **`1,889,432` unique 13-grams**; **31 genuine contamination hits dropped**:

| Eval Suite | Group | Status | Texts Indexed | Group 13-Grams | Hits Dropped (by Source) |
| :--- | :--- | :--- | ---: | ---: | :--- |
| `hle_diamond` | `hle` | `loaded_cache` | 992 | 300,297 *(combined `hle`)* | `nemotron_post` (9), `openthoughts3` (3), `opus46_3000x` (3), `opus46_3300x` (3) |
| `hle_rolling` | `hle` | `loaded_cache` | 991 | *(in `hle`)* | *(counted in `hle` above)* |
| `hle_verified` | `hle` | `loaded_cache` | 1,808 | *(in `hle`)* | *(counted in `hle` above)* |
| `gpqa_diamond` | `gpqa` | `loaded_cache` | 266 | 12,638 | `0` |
| `supergpqa` | `supergpqa` | `loaded_cache` | 25,618 | 724,974 | `nemotron_post` (9) |
| `aime_2026` | `aime_2026` | `loaded_cache` | 30 | 1,793 | `0` |
| `hmmt_feb_2026` | `hmmt_2026` | `loaded_cache` | 33 | 1,351 | `0` |
| `mrcr` | `mrcr` | `loaded_cache` | 2,400 | 619,730 | `0` |
| `terminal_bench_4_1` | `terminal_bench` | `loaded_cache` | 66 | 17,198 | `0` |
| `swe_bench_pro` | `swe_bench` | `loaded_cache` | 1,424 | 182,096 | `ot_agent_sft` (2) |
| `deep_swe` | `swe_bench` | `skipped_gated` | 0 | — | Gated on HF Hub (`403 Forbidden`); accept `datacurve/deep-swe` terms on HF to include. |
| `arc_agi_2` | `arc_agi_2` | `loaded_cache` | 167 | 17,167 | `arc_agi2_train` (2) |
| `kernelbench_l3_l4` | `kernelbench` | `loaded_cache` | 70 | 13,408 | `0` |

**Engineering Hygiene Note (Boilerplate False-Positive Guards):** During initial live testing, we caught and fixed two subtle decontamination false positives via `strip_eval_boilerplate()`:
1. Stripping the shared 23-word ARC wrapper instruction (`"Find the common transformation rule..."`) before indexing `arc_agi_2` reduced false hits on `arc_agi2_train` from `1,076` → **`2` genuine grid overlaps**.
2. Stripping leading `#` Apache-2.0 / MIT license header comments from `KernelBench` L3/L4 PyTorch files reduced false hits on `ot_agent_sft` repositories from `127` → **`0`** while keeping all PyTorch/CUDA kernel code 100% indexed.

---

## 6. Real Deterministic Salvage & Track B Synthesis Probe

### Real Before / After Deterministic Salvage Examples
1. **Nemotron Thinking-Mode System Prompt Stripping (`nemotron_post`, `1,091` rows repaired):**
   - *Before:* `messages[0] = {"role": "system", "content": "detailed thinking on"}`, inline `<think>...</think>` inside `output`, uneven trailing whitespace.
   - *After (`nemotron_post_9d68926d3ac11a52b616`, promoted to `KEEP`):* Control system prompt removed (`strip_nemotron_thinking_sysprompt`), `<think>` extracted cleanly into canonical `reasoning`, whitespace normalized (`repair_whitespace_and_fences`), mathematical answer preserved verbatim.
2. **Fable 5.1 `<final_answer>` Unwrapping + Redacted-Tool Quarantine (`fable51`, `182` rows unwrapped):**
   - *Before:* `<final_answer>\nWave 6, chunk 6P1 complete — all 6 catalog items authored, all gates green...\n</final_answer>` with `tool_calls: [{"function": {"name": "bash", "arguments": "{\"command\": \"<bash_command>\"}"}}]`.
   - *After (`fable51_4d558d8b9a4955152630`, routed to `salvage_queue`):* `<final_answer>` wrapper cleanly stripped (`unwrap_final_answer_tag`), quality scored (`slop=0.000`, `three_beat=0.7525`, `overall=0.901`), quarantined with `salvage:redacted_tool_args` so raw tool SFT is never poisoned.
3. **Boilerplate Closer Removal (`openthoughts3`, `10` rows; `opus46_3000x`, `11` rows):**
   - Strips pure trailing filler (`"Hope this helps! Let me know if you have any other questions!"`) ONLY when `_answers_equivalent(before, after)` confirms mathematical/grid answer invariance.

### Live `hf_router` Teacher Probe & Track B Synthesis Guidance
A single minimal `max_tokens=16` probe against `https://router.huggingface.co/v1/chat/completions` (`Qwen/Qwen2.5-72B-Instruct`, using silent `$HF_TOKEN`) returned:
- **HTTP Status:** `402 Payment Required`
- **Response Detail:** `{"error":"Purchase pre-paid credits to continue using Inference Providers. Alternatively, subscribe to PRO to get monthly included credits."}`
- **Other Provider Keys Present Locally:** `GEMINI_API_KEY=False`, `OPENAI_API_KEY=False`, `OPENROUTER_API_KEY=False`.

Because `hf_router` requires HF Inference Provider credits / PRO subscription, live Track B LLM synthesis was skipped cleanly without failing the local Parquet build. The entire async resumable Track B synthesis & rewrite engine (`salvage_rewrite`, `aesthetic_craft`, `sirsi_recovery`, with JSONL checkpointing + answer-invariance/anti-slop/decontam rejection gates) is verified offline in `tests/test_sirsi_dataset.py` and ready to run immediately once any provider key is funded or a local OpenAI-compatible server (e.g. `qloop serve` / vLLM) is started:

```bash
# Option A: Funded HF Router / HF PRO
python scripts/build_sirsi_dataset.py --synthesize --teacher-provider hf_router \
  --teacher-model Qwen/Qwen2.5-72B-Instruct --max-synth 20 \
  --out-dir /Users/tianrui6641/Qwen3.6-35B-A3B-Loop60/runs/datasets/Loop64-Fable-SIRSI-v1

# Option B: Gemini / OpenAI / OpenRouter / Local vLLM
GEMINI_API_KEY=... python scripts/build_sirsi_dataset.py --synthesize \
  --teacher-provider gemini --teacher-model gemini-2.5-pro --max-synth 20 \
  --out-dir /Users/tianrui6641/Qwen3.6-35B-A3B-Loop60/runs/datasets/Loop64-Fable-SIRSI-v1
```

---

## 7. Author-Original Golden Seeds (`configs/data/sirsi_golden_seeds.json`)

All `18` human-authored Apache-2.0 exemplars (`citation_key="author_original"`) pass every strict quality, AST, anti-slop (`0.000` slop across all `18` rows), 3-beat (`mean=0.827`), and 13-gram decontamination gate:
- **6 × 3-Beat Looped-CoT Reasoning:** `2` verified Olympiad/Markov math (`1012`, `7`), `2` Blackwell KV-cache & monotonic ring-buffer systems/code, `2` multi-turn agentic traces with real unredacted `tool_calls`.
- **6 × Aesthetic Critique-and-Refactor (`+6` `chosen`/`rejected` preference pairs):** `2` UI/Visual Swiss-grid/keyboard-first HTML+CSS, `2` anti-slop engineering postmortem & README voice, `2` flat typed Python & explicit bounded retry code.
- **6 × SI-RSI "Teacher-Slap" Recovery (`+6` `chosen`/`rejected` preference pairs):** Mid-thought `[Audit — stop: ...]` catches and verified recoveries for telescoping-sum boundary error (`325/462`), enumeration-loop onset (`55`), stdlib API hallucination (`Path.write_atomic`), pytest `src/` thrashing, half-open interval boundary assumption, and voice/slop drift.

---

## 8. Publishing to Hugging Face Hub (After Review)

The local dataset has already passed `--push-to-hub --dry-run` validation. Once approved for public release:

```bash
source ~/.zshrc >/dev/null 2>&1 || true; export HF_TOKEN
python scripts/build_sirsi_dataset.py \
  --push-to-hub \
  --repo-id TassP0lar/Loop64-Fable-SIRSI-v1 \
  --public \
  --out-dir /Users/tianrui6641/Qwen3.6-35B-A3B-Loop60/runs/datasets/Loop64-Fable-SIRSI-v1
```
