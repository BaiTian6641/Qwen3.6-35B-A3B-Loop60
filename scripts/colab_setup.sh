#!/usr/bin/env bash
# Colab G4 (RTX PRO 6000 Blackwell) environment setup. Idempotent; safe to re-run after a restart.
#   bash scripts/colab_setup.sh            # install
#   QLOOP_LOCK=requirements/colab.lock.txt bash scripts/colab_setup.sh   # reproduce a frozen env
set -euo pipefail
cd "$(dirname "$0")/.."
log() { printf '\n\033[1;36m[setup]\033[0m %s\n' "$*"; }

log "GPU"
nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv || echo "no nvidia-smi (CPU runtime?)"

PY=${PYTHON:-python}
log "python: $($PY --version)  torch: $($PY -c 'import torch;print(torch.__version__, torch.version.cuda)' 2>/dev/null || echo none)"

if [[ -n "${QLOOP_LOCK:-}" && -f "${QLOOP_LOCK}" ]]; then
  log "installing frozen lockfile ${QLOOP_LOCK}"
  $PY -m pip install -q --no-cache-dir -r "${QLOOP_LOCK}"
else
  # Keep Colab's preinstalled CUDA 13 torch + pyarrow/rich pins intact (--upgrade-strategy only-if-needed)
  # while ensuring transformers>=5.0 & huggingface_hub>=0.30 are up to date, with --no-cache-dir for 230 GB SSD.
  log "installing GPU stack (preserving Colab pyarrow/rich compatibility, --no-cache-dir)"
  $PY -m pip install -q --no-cache-dir -U --upgrade-strategy only-if-needed \
      "transformers>=5.0" "huggingface_hub>=0.30" "datasets>=3.0" \
      accelerate hf_transfer safetensors sentencepiece psutil "pyarrow>=19.0.0,<24" "rich>=12.4.4,<14" || \
  $PY -m pip install -q --no-cache-dir -U --upgrade-strategy only-if-needed \
      "transformers>=5.0" "huggingface_hub>=0.30" "datasets>=3.0" \
      accelerate hf_transfer safetensors sentencepiece psutil
  log "installing optional kernels (failures are reported by 'qloop doctor', not fatal)"
  $PY -m pip install -q --no-cache-dir apollo-torch || echo "apollo-torch unavailable -> built-in APOLLO fallback"
  $PY -m pip install -q --no-cache-dir "cut-cross-entropy" || echo "cut-cross-entropy unavailable -> chunked CE fallback"
  $PY -m pip install -q --no-cache-dir flash-linear-attention || echo "fla unavailable -> slow DeltaNet torch path"
  $PY -m pip install -q --no-cache-dir causal-conv1d --no-build-isolation || echo "causal-conv1d unavailable -> torch conv fallback"
fi

# Install non-editable so dist-packages/qloop/__init__.py is immediately importable in live Jupyter kernels
# (PEP 660 editable .pth hooks are only evaluated at interpreter startup).
log "installing qloop package"
$PY -m pip install -q --no-cache-dir --no-deps .

log "done. next: python -m qloop doctor --freeze requirements/colab.lock.txt"
