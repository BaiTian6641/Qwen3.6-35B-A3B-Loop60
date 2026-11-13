# Qwen3.6-35B-A3B-Loop60 → `TassP0lar/Qwen3.6-35B-A3B-Loop64`

Retrofits **`Qwen/Qwen3.6-35B-A3B`** (`40` physical layers = `10 × [3 Gated DeltaNet + 1 Gated Full Attention]`, `256` routed experts/layer, `~3B` single-pass active / `35B` total) into a **64-virtual-layer Encode–Think–Decode Looped MoE Transformer** (`Prelude [0,8) ×1 + Core [8,32) ×2 + Coda [32,40) ×1`), synced end-to-end by stage commits to **[`TassP0lar/Qwen3.6-35B-A3B-Loop64`](https://huggingface.co/TassP0lar/Qwen3.6-35B-A3B-Loop64)**.

### Architectural Highlights
- **Cross-Pass Dynamic Expert Re-Routing (Beyond 3B Active Params)**: Per-pass router residual adapters (`PassRouterAdapters`), learnable cross-pass expert repulsion, and probability-overlap diversity loss let Pass 2+ recruit **different Top-8 experts** on each recurrent pass — consulting `~4.5B–5.9B` unique parameters from the frozen `32.2B` MoE pool per token without duplicating a single byte of base weights.
- **mHC Birkhoff Loop-Back Bridge**: Doubly-stochastic convex mixing over `[h_prev, prelude_anchor, rms_matched(h_prev)]` + step embeddings + zero-initialised pass gates (`torch.equal` bit-exact identity to stock 40L Qwen3.6 at step 0).
- **Hybrid State Isolation + MoR Recursive KV Sharing**: Independent `(physical_layer, pass)` recurrent/conv slots for the 18 core Gated DeltaNet layers, paired with pass-1 KV + D-LSA cluster reuse on the 6 core Gated Full Attention layers (`1M` FP8 KV stays fixed at `10.24 GB`, zero depth multiplication) and DeltaNet write-masking for MoR early-exit consistency.
- **1M Context YaRN + Distilled Latent Sparse Attention (`D-LSA`)**: Non-rotary (`192-d`) key head indexing with batched GPU Ward clustering, distilled at `128K` and evaluated sparse up to `1M` tokens.
- **Zero-Drive HF Commit Sync & 230 GB Colab Disk Guard (`qloop.hub`)**: Adapted from [`Spark-X2.6-Recursion`](https://github.com/BaiTian6641/Spark-X2.6-Recursion), every stage commits compact trainable bundles (`~1–3 GB`), `manifest.json`, and an auto-rendered `README.md` Model Card to `TassP0lar/Qwen3.6-35B-A3B-Loop64` via `HF_TOKEN`, prunes superseded local checkpoints & HF upload staging caches, and auto-resumes across Colab 24h session resets.

📄 **Detailed architecture & engineering review (Chinese): [docs/ANALYSIS.md](docs/ANALYSIS.md)**

## Guarantees Enforced by Unit Tests (`make check`)

| Invariant | Test Suite |
|---|---|
| Loop-wrapped model is **bit-identical** to stock Qwen3.6 at step 0 (including `PassRouterAdapters`) | `tests/test_loop.py` |
| Multi-pass dynamic expert routing recruits distinct experts per pass & composes exactly with `HotExpertDeltas` + HF≥5 `TopKRouter` tuples | `tests/test_loop.py` |
| Cached incremental decode == full forward with per-pass DeltaNet slots, KV sharing, and MoR write masks (`fp64`, `1e-10`) | `tests/test_loop.py` |
| Activation checkpointing yields identical gradients across looped core, router diversity loss, and ESFT deltas | `tests/test_loop.py`, `tests/test_experts.py` |
| D-LSA with all buckets == dense attention; sparse decode == sparse full forward (flat + causal tree search) | `tests/test_lsa.py` |
| Fused in-backward APOLLO/AdamW optimizer == reference step (grad accumulation, shared loop params, clipping) | `tests/test_optim_loss.py` |
| Chunked CE / token log-probs == naive values & grads without materialising 248K-vocab logits | `tests/test_optim_loss.py` |
| Cross-session HF Hub commit resume, cumulative `S1→S2→S3→S4→Hub` delta handoff, quota squash, & 230 GB disk cleanup | `tests/test_hub_sync.py` |
| Exact trainer resume (trainables + optimizer + data stream + RNG) & GRPO verifiable-reward promote/rollback gate | `tests/test_trainer_e2e.py`, `tests/test_rl.py` |
| All shipped YAML configs + notebook AST validate on Python 3.9 grammar | `tests/test_configs_cli.py`, `scripts/ast_check.py` |

## Repository Layout

```text
src/qloop/
  model/  loop.py (64L ETD + PassRouterAdapters + MoR)  mhc.py  lsa.py (D-LSA + YaRN)
          experts.py (ESFT scan + rank-64/full HotExpertDeltas + base folding)
          probes.py (backward / core-range / surgery)  loading.py  introspect.py  toy.py
  train/  stack.py (cumulative delta stack)  trainer.py (S1-S3 LM)  rl.py (S4 GRPO/RSI)
          optim.py (fused APOLLO)  loss.py (CCE/chunked)  checkpoint.py
  data/   registry.py  adapters.py  packing.py  mixture.py  decontam.py (13-gram)
  eval/   generate.py  scorers.py  registry.py (16 frontier benchmarks)  runners.py
          kernelbench.py  external.py (Harbor / tau-bench)  server.py
  hub.py (HF commit sync, 230 GB disk guard, 1-call Hub loader)
  env/doctor.py   debug.py (structured crash reports)   colab.py   cli.py
configs/  base.yaml, model/, stages/ (s0_probe → s4_rl), eval/suite.yaml, local/ (CPU smoke)
notebooks/colab_train.ipynb   scripts/colab_setup.sh   scripts/ast_check.py
```

## Local Development (macOS Apple Silicon / CPU)

```bash
make venv     # create .venv with torch + pytest + ruff
make check    # Python 3.9 AST check (.py + .ipynb) + ruff lint + full pytest suite
.venv/bin/python -m qloop train -c configs/local/toy_lm.yaml   # smoke-test looped LM + D-LSA
.venv/bin/python -m qloop train -c configs/local/toy_rl.yaml   # smoke-test looped GRPO + gate
```

## Single End-to-End Google Colab Workflow (`notebooks/colab_train.ipynb`)

1. Open `notebooks/colab_train.ipynb` on a Colab **G4** runtime (`RTX PRO 6000 Blackwell`, 96 GB VRAM, 230 GB SSD).
2. Add **`HF_TOKEN`** (with Write permission for `TassP0lar/Qwen3.6-35B-A3B-Loop64`) in Colab's left-sidebar `🔑 Secrets` tab.
3. Click **`Runtime → Run all`**. The notebook runs:
   ```python
   from qloop.colab import setup_env, q, hub_status, run_pipeline
   setup_env(hf_repo="TassP0lar/Qwen3.6-35B-A3B-Loop64")
   q("doctor --freeze requirements/colab.lock.txt --model Qwen/Qwen3.6-35B-A3B")
   run_pipeline(debug=1, time_budget_hours=21.5)   # s0_probe -> s1_loop_heal -> s1_rescan -> s2_sft -> s3_long_lsa -> s4_rl -> eval
   ```
   When a 24h Colab session expires, `LMTrainer` / `GRPOTrainer` checkpoints cleanly, commits the latest stage state to `TassP0lar/Qwen3.6-35B-A3B-Loop64`, and exits `0`. Pressing **`Runtime → Run all`** in a new session queries `manifest.json` on Hugging Face Hub, skips completed stages, pulls the latest `~1–3 GB` trainable bundle, and continues from the exact step.

### 1-Call Inference from Hugging Face Hub

```python
from qloop.hub import load_qloop_from_hub
from qloop.eval.generate import Generator

stack = load_qloop_from_hub("TassP0lar/Qwen3.6-35B-A3B-Loop64")
gen = Generator(stack.model, stack.tokenizer, loop_controller=stack.loop)
with gen.depth(2):  # 64 virtual layers + cross-pass dynamic expert routing
    print(gen.generate(["Prove that there are infinitely many primes p = 1 (mod 4)."], max_new_tokens=2048)[0].text)
```
