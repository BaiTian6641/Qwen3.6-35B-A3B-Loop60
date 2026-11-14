# Qwen3.6-35B-A3B-Loop60 → `Qwen3.6-35B-A6B-LoopHSSD` (`v3.1` · `40L / 64L / 88L A6B`)

Retrofits **`Qwen/Qwen3.6-35B-A3B`** (`40` physical layers = `10 × [3 Gated DeltaNet + 1 Gated Full Attention]`, `256` routed experts/layer, `~3B` single-pass active / `35B` total) into a **multi-tier Looped Hybrid MoE Reasoning & Ultra-Long-Context Backbone** (`40L ~2.95B` @ `R=1`, `64L ~4.46B` @ `R=2`, **`88L A6B ~5.97B` @ `R=3`**), synced stage-by-stage to **[`TassP0lar/Qwen3.6-35B-A3B-Loop64`](https://huggingface.co/TassP0lar/Qwen3.6-35B-A3B-Loop64)** and shipped as a self-contained Hugging Face **Whole Model Pack** at **[`TassP0lar/Qwen3.6-35B-A6B-LoopHSSD-Pack`](https://huggingface.co/TassP0lar/Qwen3.6-35B-A6B-LoopHSSD-Pack)**.

### Architectural Highlights
- **`Loop-HSSD v3.1` (`88L A6B`) Subspace-Aligned Sandwich Recursion**: Scales `Prelude [0,8) ×1 + Core [8,32) ×3 + Coda [32,40) ×1` to **88 virtual layers** (`22` virtual global-attention layers) via Procrustes-calibrated `ReEntryBridge` (`r=256`, `w_anchor=0.85`), zero-initialised `PassExitGate`, per-pass Q/O rank-64 LoRA adapters, and `PassRouterAdapters` — preserving `torch.equal` step-0 identity while keeping all `32.338B` routed MoE weights 100% frozen (SHA-256 verified).
- **Horizon-Split Surprise-Gated Delta Memory (`HSSD v3.1`) & NVIDIA `GDN-2` (`arXiv:2605.22791`)**:
  - **Track B (`HSSD v3.1`, Audit-Hardened)**: Upgrades all 30 linear-attention layers with pure-linear low-rank `theta` projections (`~1.95M/layer`, `~57.5M–58.5M` across 30L vs `176.5M` dense), disjoint `[513, 576]` local SWA (`0` overlap with `[1, 512]` `D-LSA`), `PeakBoltzmannGate`, post-rotation 2D phase WY `chunk_kda` with cloned `float64` `theta_cum` (`< 2e-6 rad` @ `1M` steps), linear-`beta` write scaling, multi-page `routed` chunk-start root-surprise `s = g^2` on `boundary_ring[-1]`, `M=8` sparse Top-`2` STE write-budgeted pages, factored `8 KiB/tok/layer` decay (`16×` smaller), and compact `81–89 MiB/layer` `bfloat16` decode caches (`HSSDHybridCache` / HF beam-search ready).
  - **Track A (`GDN-2`)**: Decoupled channel-wise erase/write/log-decay WY parallelization (`chunk_gdn2`) + `O(1)` recurrent decode (`recurrent_gdn2`).
- **Cross-Pass Dynamic Expert Re-Routing (Beyond 3B Active Params)**: Per-pass router residual adapters recruit distinct Top-8 experts on Passes 2 and 3, consulting `~4.46B–5.97B` unique parameters per token from the frozen `32.3B` MoE pool with zero base-weight duplication.
- **Two-Tier `1M–10M` Ultra-Long Context + `4×32K` K/V-Replay BPTT**:
  - **Tier 1 (`1M` GPU-Resident Primary)**: `10.00 GiB` `FP8` `SharedKV` (`1.0×` KV across `R=1..3`) + untied per-pass `D-LSA` query indexers (`num_query_passes=3`) + `ReRoPE` relative distance clamping.
  - **Tier 2 (`10M` Host-DDR5 Stretch)**: `Block64HadamardINT4Cache` (`2,304 B` 4-token micro-page, `53.64 GiB` pinned DDR5 @ `10M`) keeping `[0:64]` mRoPE strictly orthogonal to `[64:256]` NoPE for exact `ReRoPE` commutativity, prefetched 3 layers ahead.
  - **`SegmentedKVBPTTRunner` (`128K` Differentiable Retrieval)**: Streams `4 × 32K` segments while differentiably re-projecting Top-`2,048` selected prior-segment K/V tokens.
- **Self-Contained Whole Model Pack & `OptimizedLoopInferenceEngine`**: Streams sharded `model-00001-of-NNNNN.safetensors` one `~4.5 GB` shard at a time (`< 6 GB` peak temporary SSD on Colab's `230 GB` disk) alongside standalone `configuration_qwen36_loophssd.py` & `modeling_qwen36_loophssd.py`, loadable via both `OptimizedLoopInferenceEngine.from_pretrained(...)` and standard `transformers.AutoModelForCausalLM.from_pretrained(..., trust_remote_code=True)`.

📄 **Formal Architecture, Implementation Plan & Research Documentation (`docs/`):**
- 📘 **[Backbone Architecture Specification (`Loop-HSSD v3` / `35B-A6B`)](docs/ARCHITECTURE_SPEC.md)** — Definitive backbone modification blueprint closing all `v1` and `v2 Final-Pass Review` findings (`A1–A6, B1–B6, C1–C2, E1–E3`), scaling `Qwen3.6-35B-A3B` to `64L (~A4.5B)` and `88L (~A6.0B)` while reusing `>98.5%` frozen base weights.
- 🛠️ **[Comprehensive Implementation & Kill-Gate Plan](docs/IMPLEMENTATION_PLAN.md)** — Module-by-module engineering roadmap (`M1–M6`), `Gate -1 -> Gate 4` falsification criteria, and calibrated Colab G4 (`96 GB` VRAM / `230 GB` SSD) budgets.
- 📜 **[Formal Research Essay](docs/RESEARCH_ESSAY.md)** — Academic monograph with full proofs of Complex-to-Real `chunk_kda` Isomorphism, Cumulative Write-Mass Retention ($\exp(-\mathcal{B}/d_k)$), and `1-bit QJL` Variance Amplification vs. Block-64 Hadamard `INT4`.
- 🧭 **[Baseline Engineering Review (Chinese)](docs/ANALYSIS.md)** — Detailed engineering audit of the foundation stack.

## Specialized Google Colab Notebooks (`notebooks/`)

| Notebook | Role & Workflow |
| :--- | :--- |
| **[`notebooks/colab_probe.ipynb`](notebooks/colab_probe.ipynb)** | **Zero-Training Diagnostic & Calibration Suite**: Executes `Gate -1` (`fp64` math proofs), `Gate 0a` (hardware backward & step-0 surgery), `Gate 0p-1..4` (expert headroom, Procrustes subspace re-entry sweep, `native` vs `yarn` vs `rerope` extrapolation, single-layer `profile_hssd_layer` storage/throughput audit) + `Gate 1-Ablate` (`hssd_ablation_matrix`), writes `probe_report.{json,md}`, and prints calibrated `-s` overrides. |
| **[`notebooks/colab_train_e2e.ipynb`](notebooks/colab_train_e2e.ipynb)** | **Self-Resuming `Loop-HSSD v3.1` E2E Training & Whole-Model Pack Release**: Runs/resumes `s0_probe → s1_loop_a6b → s1b_hssd_distill → s3_long_1m_lsa → s2_joint_bptt → s4_rl` across 24h Colab sessions and streams the sharded `safetensors` **Whole Model Pack** (`< 6 GB` temp SSD) to `TassP0lar/Qwen3.6-35B-A6B-LoopHSSD-Pack`. |
| **[`notebooks/colab_infer_eval.ipynb`](notebooks/colab_infer_eval.ipynb)** | **Shipped On-Weight Inference & Frontier `R=1/2/3` Evaluation**: Loads the Whole Model Pack into `OptimizedLoopInferenceEngine` (`40L / 64L / 88L A6B`), demos `1M FP8` + `10M H_64 INT4` offload, and runs the held-out frontier depth-curve suite (`ARC-AGI-2`, `AIME 2026`, `LongBench v2`, `KernelBench`). |
| **[`notebooks/colab_train.ipynb`](notebooks/colab_train.ipynb)** | **Track-A `64L` Baseline Single-Notebook Pipeline**: Resumable `GDN-2` + `mHC` + `D-LSA` + `GRPO` single-notebook workflow. |

## Guarantees Enforced by Unit Tests (`make check`)

| Invariant | Test Suite |
|---|---|
| `Loop-HSSD v3.1` (`Gate -1`): Post-rotation 2D phase WY isomorphism, disjoint `[513, 576]` horizon (`0` overlap), multi-page `routed` `s = g^2` chunk-start WY equivalence, write-budget law, `ReEntryBridge` + router unlock, QJL variance & Block-64 Hadamard-ReRoPE commutativity, and `34.661B` ledger (`fp64`) | `tests/test_loop_hssd_v3_math.py` |
| `HSSD v3.1` runtime & audit regressions (`3A` compact `81–89 MiB/layer` `bf16` cache, `3B` factored decay, `3C` sparse Top-`k` STE write budget, `3D` HF cache/beam ops, `3E` `cu_seqlens`/`seq_idx` isolation, `4.1` 2D padding logical clock, `4.2` `boundary_ring[-1]` & `state_equal`, `4.3` `1M`-step `fp64` `theta_cum`, `5–6` standalone parity) | `tests/test_hssd.py`, `tests/test_hssd_audit_regressions.py` |
| Per-pass `D-LSA` (`num_query_passes=3`), `ReRoPE` distance clamping, `FP8KVCacheQuantizer`, `Block64HadamardINT4Cache` (`2,304 B` micro-page + `H_64` ReRoPE commutativity), and `SegmentedKVBPTTRunner` cross-segment K/V gradients | `tests/test_longctx_v3.py` |
| Sharded `safetensors` Whole Model Pack export, low-disk `< 6 GB` shard streaming, standalone HF `AutoModelForCausalLM` loading (`HSSD v3.1` parity), and `OptimizedLoopInferenceEngine` (`R=1/2/3`, chunked prefill, `1.0×` FP8/INT4 SharedKV) | `tests/test_pack_infer.py`, `tests/test_v3_integration_e2e.py` |
| Zero-training `Gate 0p-1..4` & `Gate 1-Ablate` structured probe suite (`probe_expert_headroom`, `probe_reentry_anchors`, `probe_positional_extrapolation`, `profile_hssd_layer`, `hssd_ablation_matrix`) & frozen MoE SHA-256 invariance | `tests/test_probes_v3.py`, `tests/test_loop_v3.py`, `tests/test_v3_integration_e2e.py` |
| Loop-wrapped + GDN-2 / Subspace-Sandwich model is **bit-identical** to stock Qwen3.6 at step 0 (including `PassRouterAdapters`, `ReEntryBridge`, `PassExitGate`, per-pass Q/O LoRA) | `tests/test_loop.py`, `tests/test_loop_v3.py`, `tests/test_gdn2.py` |
| NVIDIA GDN-2 chunkwise WY (`chunk_gdn2`) == sequential `recurrent_gdn2` in `fp64` (`1e-10` fwd / `1e-9` bwd) + MoR write-masked decode | `tests/test_gdn2.py` |
| D-LSA with all buckets == dense attention; sparse decode == sparse full forward (flat + causal tree search) | `tests/test_lsa.py` |
| Fused in-backward APOLLO/AdamW optimizer & chunked CE / log-probs == reference values & grads | `tests/test_optim_loss.py` |
| Cross-session HF Hub commit resume, cumulative stage delta handoff, quota squash, & 230 GB disk cleanup | `tests/test_hub_sync.py` |
| Exact trainer resume (trainables + optimizer + data stream + RNG) & GRPO verifiable-reward promote/rollback gate | `tests/test_trainer_e2e.py`, `tests/test_rl.py` |
| All shipped v2 (`64L`) & v3.1 (`88L`) YAML configs + all 4 Colab notebooks validate on Python 3.9 AST grammar | `tests/test_configs_cli.py`, `scripts/ast_check.py` |

## Repository Layout

```text
src/qloop/
  model/  loop.py (64L/88L ETD + ReEntryBridge + PassExitGate + PassRouterAdapters + MoR)
          hssd.py (HSSD v3.1: chunk_kda + [513,576] SWA + PeakBoltzmannGate + M=8 STE pages + HSSDHybridCache)
          gdn2.py (NVIDIA GDN-2 WY + recurrent)   mhc.py   lsa.py (per-pass D-LSA + YaRN)
          pos_rerope.py (ReRoPE + LongPrefixCalibrator)   offload.py (FP8 + Block-64 H_64 INT4)
          experts.py (ESFT scan + HotExpertDeltas + SHA-256 guard)   probes.py (Gate 0a / 0p-1..4 / 1-Ablate)
          loading.py   introspect.py   toy.py
  train/  stack.py (cumulative v2/v3 delta stack)   trainer.py (LM + long-prefix + BPTT)
          bptt.py (SegmentedKVBPTTRunner 4x32K)     rl.py (GRPO/RSI)
          optim.py (fused APOLLO)   loss.py (CCE/chunked)   checkpoint.py
  infer/  engine.py (OptimizedLoopInferenceEngine: R=1/2/3, chunked prefill, FP8/INT4 SharedKV)
          hf_custom_modeling.py (standalone Qwen36LoopHSSDConfig & Qwen36LoopHSSDForCausalLM)
  export_pack.py (Whole Model Pack sharded safetensors exporter & low-disk HF Hub streamer)
  data/   registry.py  adapters.py  packing.py  mixture.py  decontam.py (13-gram)
  eval/   generate.py  scorers.py  registry.py (16 frontier benchmarks)  runners.py
          kernelbench.py  external.py (Harbor / tau-bench)  server.py
  hub.py (HF commit sync, 230 GB disk guard, Hub loader)
  env/doctor.py   debug.py (structured crash reports)   colab.py   cli.py
configs/  base.yaml, model/, stages/ (v2 64L & v3.1 88L Loop-HSSD), eval/suite.yaml, local/
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

## End-to-End `Loop-HSSD v3.1` Colab Workflow (`Probe → Train & Ship Pack → Infer & Eval`)

1. **Calibrate (`notebooks/colab_probe.ipynb`)**: Run zero-training `Gate -1` + `Gate 0a` + `Gate 0p-1..4` + `Gate 1-Ablate` probes:
   ```python
   from qloop.colab import setup_env, run_probe_notebook_suite
   setup_env(hf_repo="TassP0lar/Qwen3.6-35B-A3B-Loop64")
   report = run_probe_notebook_suite(config_path="configs/stages/s0_probe.yaml", push_to_hub=True)
   ```
2. **Train & Stream Whole Model Pack (`notebooks/colab_train_e2e.ipynb`)**: Self-resumes across Colab 24h sessions and streams sharded `safetensors` (`< 6 GB` peak temporary SSD) to Hugging Face Hub upon final stage completion:
   ```python
   from qloop.colab import setup_env, run_v3_e2e_and_ship_pack
   setup_env(hf_repo="TassP0lar/Qwen3.6-35B-A3B-Loop64")
   run_v3_e2e_and_ship_pack(
       hf_repo="TassP0lar/Qwen3.6-35B-A3B-Loop64",
       pack_repo="TassP0lar/Qwen3.6-35B-A6B-LoopHSSD-Pack",
       time_budget_hours=21.5,
       stream_low_disk=True,
   )
   ```
   Or export / stream any promoted stage checkpoint directly via CLI:
   ```bash
   python -m qloop export-pack -c configs/stages/s2_joint_bptt.yaml --ckpt runs/s4_rl \
       --push-repo TassP0lar/Qwen3.6-35B-A6B-LoopHSSD-Pack --stream-low-disk
   ```

## Shipped On-Weight Inference (`OptimizedLoopInferenceEngine` & Hugging Face `AutoModel`)

```python
from qloop.infer import InferenceConfig, OptimizedLoopInferenceEngine

engine = OptimizedLoopInferenceEngine.from_pretrained(
    "TassP0lar/Qwen3.6-35B-A6B-LoopHSSD-Pack",
    cfg=InferenceConfig(prefill_chunk_size=4096, fp8_kv_cache=True),
)
# Switch dynamically per request: R=1 (40L ~2.95B) | R=2 (64L ~4.46B) | R=3 (88L A6B ~5.97B)
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
