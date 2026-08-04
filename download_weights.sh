#!/usr/bin/env bash
set -euo pipefail

# Prerequisite: install the pinned Python dependencies first. This script never
# installs packages implicitly.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REWARD_ROOT="$REPO_ROOT/src/t2i-r1/reward_weight"
mkdir -p "$REWARD_ROOT/HPSv2.1"

if command -v hf >/dev/null 2>&1; then
  HF_CLI=(hf download)
elif command -v huggingface-cli >/dev/null 2>&1; then
  HF_CLI=(huggingface-cli download)
else
  echo 'Missing Hugging Face CLI. Install requirements.txt, then rerun.' >&2
  exit 1
fi

download_repo() {
  local repo_id="$1"
  local target="$2"
  local marker="$3"
  if [ -e "$target/$marker" ]; then
    echo "present: $target"
  else
    "${HF_CLI[@]}" "$repo_id" --local-dir "$target"
  fi
}

download_repo deepseek-ai/Janus-Pro-1B "$REWARD_ROOT/Janus-Pro-1B" config.json
if [ "${DOWNLOAD_7B:-0}" = '1' ]; then
  download_repo deepseek-ai/Janus-Pro-7B "$REWARD_ROOT/Janus-Pro-7B" config.json
fi
download_repo Qwen/Qwen3-VL-2B-Instruct "$REWARD_ROOT/Qwen3-VL-2B-Instruct" config.json
download_repo google-bert/bert-base-uncased "$REWARD_ROOT/bert-base-uncased" config.json

if [ ! -s "$REWARD_ROOT/HPSv2.1/HPS_v2.1_compressed.pt" ]; then
  "${HF_CLI[@]}" xswu/HPSv2 HPS_v2.1_compressed.pt \
    --local-dir "$REWARD_ROOT/HPSv2.1"
fi

GDINO_FILE="$REWARD_ROOT/groundingdino_swint_ogc.pth"
GDINO_URL='https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth'
if [ ! -s "$GDINO_FILE" ]; then
  if command -v curl >/dev/null 2>&1; then
    curl --fail --location --continue-at - --output "$GDINO_FILE" "$GDINO_URL"
  elif command -v wget >/dev/null 2>&1; then
    wget --continue --output-document="$GDINO_FILE" "$GDINO_URL"
  else
    echo 'Neither curl nor wget is available.' >&2
    exit 1
  fi
fi

echo 'Weight download complete. Run: bash check_train_env.sh'
