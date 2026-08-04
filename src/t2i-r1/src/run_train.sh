#!/usr/bin/env bash
set -euo pipefail

# Portable T2I-RIA launcher. The defaults reproduce the *shape* of the final
# Full-1B BF16 run; exact historical runtime records are frozen under
# reproducibility/runtime_configs/ and take precedence over launcher defaults.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
REWARD_ROOT="$REPO_ROOT/src/t2i-r1/reward_weight"

EXP_NAME="${EXP_NAME:-t2i_ria_full_1b}"
MODEL_VARIANT="${MODEL_VARIANT:-1B}"
MODEL_PATH="${MODEL_PATH:-$REWARD_ROOT/Janus-Pro-$MODEL_VARIANT}"
REWARD_FUNCS="${REWARD_FUNCS:-hps gdino vlm_attr vlm_orm}"
NPROC="${NPROC:-4}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-${PER_DEVICE_BATCH_SIZE:-2}}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-2}"
NUM_GENERATIONS="${NUM_GENERATIONS:-8}"
MAX_STEPS="${MAX_STEPS:-600}"
SAVE_STEPS="${SAVE_STEPS:-200}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-5}"
LEARNING_RATE="${LEARNING_RATE:-1e-6}"
BETA="${BETA:-0.01}"
REPORT_TO="${REPORT_TO:-none}"
DEBUG_MODE="${DEBUG_MODE:-false}"

OUTPUT_DIR="${OUTPUT_DIR:-$SCRIPT_DIR/outputs/$EXP_NAME}"
DATASET_PATH="${DATASET_PATH:-$REPO_ROOT/data/geneval_and_t2i_data_final.json}"
REASONING_PROMPT_PATH="${REASONING_PROMPT_PATH:-$REPO_ROOT/data/prompt/reasoning_prompt.txt}"
ZERO_CONFIG="${ZERO_CONFIG:-$REPO_ROOT/src/t2i-r1/configs/zero2.json}"

HPS_CKPT="${HPS_CKPT:-$REWARD_ROOT/HPSv2.1/HPS_v2.1_compressed.pt}"
GDINO_CKPT="${GDINO_CKPT:-$REWARD_ROOT/groundingdino_swint_ogc.pth}"
GDINO_CONFIG="${GDINO_CONFIG:-$SCRIPT_DIR/utils/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py}"
VLM_CKPT="${VLM_CKPT:-$REWARD_ROOT/Qwen3-VL-2B-Instruct}"
export GDINO_TEXT_ENCODER="${GDINO_TEXT_ENCODER:-$REWARD_ROOT/bert-base-uncased}"

export WANDB_PROJECT="${WANDB_PROJECT:-T2I-RIA}"
export WANDB_NAME="${WANDB_NAME:-$EXP_NAME}"
export WANDB_RUN_ID="${WANDB_RUN_ID:-$EXP_NAME}"
export WANDB_RESUME="${WANDB_RESUME:-allow}"
export DEBUG_MODE
export LOG_PATH="$OUTPUT_DIR/reward_log.txt"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

mkdir -p "$OUTPUT_DIR"
TIMESTAMP="$(date '+%Y%m%d_%H%M%S')"
LOG_FILE="$OUTPUT_DIR/train_${TIMESTAMP}.log"
ln -sfn "$(basename "$LOG_FILE")" "$OUTPUT_DIR/train_main.log"
exec > >(tee "$LOG_FILE") 2>&1

PROMPTS_PER_STEP=$((NPROC * PER_DEVICE_TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS))
IMAGES_PER_STEP=$((PROMPTS_PER_STEP * NUM_GENERATIONS))
echo "experiment=$EXP_NAME model=$MODEL_PATH rewards=$REWARD_FUNCS"
echo "workers=$NPROC prompts_per_optimizer_step=$PROMPTS_PER_STEP images_per_optimizer_step=$IMAGES_PER_STEP"
echo "output=$OUTPUT_DIR"

RESUME_ARGS=()
if [ -n "${RESUME:-}" ]; then
  RESUME_ARGS=(--resume_from_checkpoint "$RESUME")
fi

cd "$SCRIPT_DIR"
PYTHONPATH="$SCRIPT_DIR${PYTHONPATH:+:$PYTHONPATH}" \
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}" \
MASTER_PORT="${MASTER_PORT:-29500}" \
torchrun --nproc_per_node="$NPROC" --standalone --nnodes=1 \
  open_r1/grpo.py \
  --use_vllm=false \
  --deepspeed "$ZERO_CONFIG" \
  --output_dir "$OUTPUT_DIR" \
  --model_name_or_path "$MODEL_PATH" \
  --dataset_name "$DATASET_PATH" \
  --dataloader_num_workers "${DATALOADER_NUM_WORKERS:-4}" \
  --dataloader_pin_memory true \
  --max_prompt_length 512 \
  --max_completion_length 1024 \
  --temperature 1.0 \
  --num_generations "$NUM_GENERATIONS" \
  --per_device_train_batch_size "$PER_DEVICE_TRAIN_BATCH_SIZE" \
  --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS" \
  --logging_steps "${LOGGING_STEPS:-5}" \
  --bf16=true \
  --torch_dtype bfloat16 \
  --report_to "$REPORT_TO" \
  --gradient_checkpointing=false \
  --attn_implementation sdpa \
  --max_steps "$MAX_STEPS" \
  --run_name "$EXP_NAME" \
  --save_steps "$SAVE_STEPS" \
  --save_total_limit "$SAVE_TOTAL_LIMIT" \
  --save_only_model=true \
  --new_generations_image 1 \
  --image_token_num_per_image 576 \
  --cfg_weight 5 \
  --reasoning_prompt_path "$REASONING_PROMPT_PATH" \
  --reward_funcs $REWARD_FUNCS \
  --beta "$BETA" \
  --tf32=true \
  --learning_rate "$LEARNING_RATE" \
  --hps_ckpt_path "$HPS_CKPT" \
  --gdino_ckpt_path "$GDINO_CKPT" \
  --gdino_config_path "$GDINO_CONFIG" \
  --vlm_ckpt_path "$VLM_CKPT" \
  "${RESUME_ARGS[@]}"
