#!/usr/bin/env bash
set -euo pipefail

# Generate ten images per T2I-CompBench prompt.
# Example:
#   T2I_COMPBENCH_DIR=/path/to/T2I-CompBench \
#   CUDA_VISIBLE_DEVICES=0,1 bash run_generate.sh \
#     --nproc 2 --model_path outputs/my_run/checkpoint-600 \
#     --save_root ../../../eval_results/finetuned

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
export PYTHONPATH="$SCRIPT_DIR${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

NPROC=1
MODEL_PATH=""
SAVE_ROOT="$REPO_ROOT/eval_results/finetuned"
DATASET_DIR="${T2I_COMPBENCH_DIR:+$T2I_COMPBENCH_DIR/examples/dataset}"
REASONING_PROMPT="$REPO_ROOT/data/prompt/reasoning_prompt.txt"
NUM_GENERATION=10
CFG_WEIGHT=5.0
TEMPERATURE=1.0
SEED=42
SKIP_EXISTING=(--skip_existing)
CATEGORIES=()

while [ "$#" -gt 0 ]; do
  case "$1" in
    --nproc) NPROC="$2"; shift 2 ;;
    --model_path) MODEL_PATH="$2"; shift 2 ;;
    --save_root) SAVE_ROOT="$2"; shift 2 ;;
    --dataset_dir) DATASET_DIR="$2"; shift 2 ;;
    --reasoning_prompt) REASONING_PROMPT="$2"; shift 2 ;;
    --num_generation) NUM_GENERATION="$2"; shift 2 ;;
    --cfg_weight) CFG_WEIGHT="$2"; shift 2 ;;
    --temperature) TEMPERATURE="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --no_skip) SKIP_EXISTING=(); shift ;;
    --categories)
      shift
      while [ "$#" -gt 0 ] && [[ "$1" != --* ]]; do
        CATEGORIES+=("$1")
        shift
      done
      ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

[ -n "$MODEL_PATH" ] || { echo '--model_path is required' >&2; exit 2; }
[ -n "$DATASET_DIR" ] || {
  echo 'Pass --dataset_dir or set T2I_COMPBENCH_DIR.' >&2
  exit 2
}

mkdir -p "$SAVE_ROOT/logs"
TIMESTAMP="$(date '+%Y%m%d_%H%M%S')"
LOG_FILE="$SAVE_ROOT/logs/generate_${TIMESTAMP}.log"
ln -sfn "$(basename "$LOG_FILE")" "$SAVE_ROOT/logs/generate_latest.log"
exec > >(tee "$LOG_FILE") 2>&1

echo "model=$MODEL_PATH output=$SAVE_ROOT dataset=$DATASET_DIR"
echo "workers=$NPROC images_per_prompt=$NUM_GENERATION cfg=$CFG_WEIGHT temperature=$TEMPERATURE seed=$SEED"

CATEGORY_ARGS=()
if [ "${#CATEGORIES[@]}" -gt 0 ]; then
  CATEGORY_ARGS=(--categories "${CATEGORIES[@]}")
fi

cd "$SCRIPT_DIR"
torchrun --standalone --nnodes=1 --nproc_per_node="$NPROC" \
  generate_all_eval.py \
  --model_path "$MODEL_PATH" \
  --dataset_dir "$DATASET_DIR" \
  --save_root "$SAVE_ROOT" \
  --reasoning_prompt_path "$REASONING_PROMPT" \
  --num_generation "$NUM_GENERATION" \
  --cfg_weight "$CFG_WEIGHT" \
  --temperature "$TEMPERATURE" \
  --seed "$SEED" \
  "${SKIP_EXISTING[@]}" \
  "${CATEGORY_ARGS[@]}"
