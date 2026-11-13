# Qwen3.6-35B-A3B-Loop60

Retrofits **Qwen3.6-35B-A3B** (40 layers: 10 × [3 Gated DeltaNet + 1 Gated Attention], 256-expert MoE)
into a **64-layer looped transformer** (Encode–Think–Decode: 8 + 24×2 + 8). The stack includes:
- an mHC loop-back bridge;
- MoR recursive KV sharing and optional token-level depth routing;
- **D-LSA latent sparse attention** for 1M context;
- ESFT hot-expert deltas, an in-backward APOLLO optimizer and chunked/CCE loss;
- SFT, GRPO recursive self-improvement and depth-sweep evaluation.

It targets one Colab G4 runtime (RTX PRO 6000 Blackwell, 96 GB). Development happens on a laptop against a structurally faithful toy model.

📄 **Design analysis (Chinese): [docs/ANALYSIS.md](docs/ANALYSIS.md)**: review of the original plan, corrections, and the risk-ordered roadmap.

## Guarantees enforced by tests

| Invariant | Test |
|---|---|
| Loop-wrapped model is **bit-identical** to the stock model at init (zero-init pass gates) | `test_loop.py::test_identity_at_init_is_bit_exact` |
| Cached incremental decode == full forward with per-pass DeltaNet slots, KV sharing, MoR write masks (fp64, 1e-10) | `test_loop.py::test_incremental_decode_matches_full_forward` |
| Activation checkpointing gives identical gradients | `test_loop.py`, `test_experts.py` |
| D-LSA with all buckets == dense attention; sparse decode == sparse full forward (flat + causal tree search) | `test_lsa.py` |
| Hot-expert deltas: zero-init identity, exact merged-weight equivalence, correct when a layer runs twice | `test_experts.py` |
| Fused in-backward optimizer == classic optimizer (accumulation, shared params, clipping) | `test_optim_loss.py` |
| Chunked CE / token log-probs == naive (values + grads) without materialising 248K-vocab logits | `test_optim_loss.py` |
| Exact resume (trainables + optimizer + data stream + RNG); RL promote/rollback gate | `test_trainer_e2e.py`, `test_rl.py` |
| Every shipped config validates to 64 virtual layers; CLI drives the full toy pipeline | `test_configs_cli.py` |

## Layout

```
src/qloop/
  model/  loop.py (64-layer ETD retrofit) mhc.py lsa.py (D-LSA) experts.py (ESFT + deltas)
          probes.py (backward / core-range / surgery) loading.py introspect.py toy.py
  train/  stack.py trainer.py (LM stages) rl.py (GRPO/RSI) optim.py (fused APOLLO) loss.py checkpoint.py
  data/   registry + verification, adapters, packing, mixture (resumable), decontamination
  eval/   generator (loop depth), scorers, benchmark registry, runners, KernelBench, Harbor/tau, server
  env/doctor.py   debug.py (crash reports)   colab.py   cli.py
configs/  base.yaml, model/, stages/ (s0_probe → s4_rl), eval/suite.yaml, local/ (toy smoke runs)
notebooks/colab_train.ipynb   scripts/colab_setup.sh   scripts/ast_check.py
```

## Local development (macOS / CPU)

```bash
make venv     # .venv with torch + pytest + ruff
make check    # AST check (py3.9 grammar) + ruff + unit tests (~1 min)
.venv/bin/python -m qloop train -c configs/local/toy_lm.yaml   # full LM stage on the toy model
.venv/bin/python -m qloop train -c configs/local/toy_rl.yaml   # GRPO stage on the toy model
```

## Colab

Open `notebooks/colab_train.ipynb` on a G4 runtime and add `HF_TOKEN` to Secrets. Each stage is one line:

```python
q("doctor --freeze requirements/colab.lock.txt")
q("probe -c configs/stages/s0_probe.yaml --backward --surgery --core-ranges")
q("scan-experts -c configs/stages/s0_probe.yaml")
q("train -c configs/stages/s1_loop_heal.yaml")          # re-run to resume after a session reset
q("eval -c configs/eval/suite.yaml --ckpt /content/drive/MyDrive/qloop_runs/s4_rl")
```

Every stage runs as a subprocess with its own run dir, `run.log`, `metrics.jsonl` and, on failure,
`crash/crash_<ts>.md`. The report contains the stage path, actionable hints, the traceback, local tensor
summaries per frame (shape/dtype/device/NaN count), the environment, the CUDA memory summary and a memory
snapshot on OOM. The notebook cell prints the hints and traceback inline.

Raise verbosity with `--debug 2` (NaN/Inf origin hooks, C++ stack traces) or `--debug 3`
(CUDA_LAUNCH_BLOCKING, autograd anomaly mode, allocation timeline).
