#!/usr/bin/env bash
# Portable environment setup for the reported T2I-RIA software stack.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${VENV_DIR:-$REPO_ROOT/.venv}"
GDINO_DIR="$REPO_ROOT/src/t2i-r1/src/utils/GroundingDINO"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu124}"

if [ "${RECREATE_VENV:-0}" = "1" ] && [ -d "$VENV_DIR" ]; then
  echo "Removing existing environment because RECREATE_VENV=1: $VENV_DIR"
  rm -rf "$VENV_DIR"
fi

if [ ! -f "$VENV_DIR/bin/activate" ]; then
  python3 -m venv "$VENV_DIR"
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

python -m pip install --upgrade pip
python -m pip install \
  torch==2.5.1 \
  torchvision==0.20.1 \
  torchaudio==2.5.1 \
  --index-url "$TORCH_INDEX_URL"
python -m pip install -r "$REPO_ROOT/requirements.txt"

if command -v nvcc >/dev/null 2>&1 || [ -n "${CUDA_HOME:-}" ]; then
  python -m pip install -e "$GDINO_DIR" --no-build-isolation
else
  echo "WARNING: nvcc/CUDA_HOME was not found; Grounding DINO was not compiled."
  echo "Install a matching CUDA toolkit, set CUDA_HOME, and run:"
  echo "  python -m pip install -e $GDINO_DIR --no-build-isolation"
fi

echo "Environment ready: $VENV_DIR"
echo "Run: source $VENV_DIR/bin/activate"
echo "Then: bash $REPO_ROOT/check_train_env.sh"
