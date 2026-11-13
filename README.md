# Qwen3.6-35B-A3B-Loop60

Looped-depth (Encode–Think–Decode) retrofit and memory-lean training stack for
**Qwen3.6-35B-A3B** (hybrid Gated DeltaNet + Gated Attention, 256-expert MoE), targeting a
single Colab G4 runtime (RTX PRO 6000 Blackwell, 96 GB). Day-to-day development happens on a
laptop (CPU-only tests against a structurally faithful toy model).

> Work in progress. See `docs/` for the design analysis.

## Local development (macOS / CPU)

```bash
make venv     # .venv with torch + pytest + ruff
make check    # AST check (py3.9 grammar) + ruff lint + unit tests
```
