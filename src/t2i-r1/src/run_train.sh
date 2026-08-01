#!/bin/bash
# run_train.sh
# 正式训练脚本：BiCoT-GRPO + 4 个 compositional reward 函数
#
# ──────────────────────────────────────────────────────────────────────
# 核心约定：每次实验只改一个变量 EXP_NAME，所有产物自动同步：
#   - 输出目录    : outputs/<EXP_NAME>/
#   - 日志文件    : outputs/<EXP_NAME>/train_main_<时间戳>.log（+ train_main.log 软链）
#   - wandb run   : 名称 = run_id = $EXP_NAME（同 EXP_NAME 自动接回曲线连续）
#   - HF run_name : $EXP_NAME
# ──────────────────────────────────────────────────────────────────────
#
# 启动方式（**不要加 nohup**，会让终端 / merlin 日志窗口看不到输出）：
#
#   [A] ssh 调试期：tmux 保活，断网不死
#       tmux new -s train
#       EXP_NAME=my_run NPROC=4 CUDA_VISIBLE_DEVICES=0,1,2,3 bash src/t2i-r1/src/run_train.sh
#       # 离开但保留：Ctrl+b 然后 d        重连：tmux attach -t train
#
#   [B] merlin worker 入口（生产期）：worker 进程由平台守护
#       mlx worker launch ... -- bash /abs/path/run_train.sh
#       # merlin 平台日志窗口能直接看训练输出
#
# ──────────────────────────────────────────────────────────────────────
# 常用命令模板：
#
#   # 主任务 FULL（默认 4 个 reward）
#   EXP_NAME=train_main_4gpu_g8_bs2_a100 NPROC=4 CUDA_VISIBLE_DEVICES=0,1,2,3 \
#     bash src/t2i-r1/src/run_train.sh
#
#   # Reward ablation（串行跑，别并发；REWARD_FUNCS 用空格分隔，不要加引号传 torchrun）
#   EXP_NAME=ablation_wo_orm   REWARD_FUNCS="hps gdino vlm_attr"   NPROC=4 CUDA_VISIBLE_DEVICES=0,1,2,3 bash src/t2i-r1/src/run_train.sh
#   EXP_NAME=ablation_wo_attr  REWARD_FUNCS="hps gdino vlm_orm"    NPROC=4 CUDA_VISIBLE_DEVICES=0,1,2,3 bash src/t2i-r1/src/run_train.sh
#   EXP_NAME=ablation_wo_gdino REWARD_FUNCS="hps vlm_attr vlm_orm" NPROC=4 CUDA_VISIBLE_DEVICES=0,1,2,3 bash src/t2i-r1/src/run_train.sh
#
#   # 续训：传同一个 EXP_NAME + RESUME=<ckpt 路径>，wandb 会接回原 run
#   EXP_NAME=train_main_4gpu_g8_bs2_a100 \
#     RESUME=src/t2i-r1/src/outputs/train_main_4gpu_g8_bs2_a100/checkpoint-400 \
#     NPROC=4 CUDA_VISIBLE_DEVICES=0,1,2,3 \
#     bash src/t2i-r1/src/run_train.sh
#   # 找最新 ckpt：ls -d outputs/<EXP_NAME>/checkpoint-* | sort -V | tail -1
#
# ──────────────────────────────────────────────────────────────────────
# 产物落点（outputs/<EXP_NAME>/ 下）：
#   - train_main_<时间戳>.log  本次启动完整日志（脚本 + python + NCCL + traceback）
#   - train_main.log           软链 → 最新一次的 .log，tail -f 永远跟最新
#   - reward_log.txt           reward 详情（DEBUG_MODE=true 才有）
#   - checkpoint-<step>/       HF Trainer ckpt（带 optimizer states，可续训）
#   - runs/<timestamp>/        tensorboard tfevents（HF 自动写）
# ──────────────────────────────────────────────────────────────────────


# ── 自动定位仓库根目录（脚本位于 <REPO>/src/t2i-r1/src/run_train.sh）──
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # .../src/t2i-r1/src
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"              # 仓库根
cd "$SCRIPT_DIR"


EXP_NAME="${EXP_NAME:-train_main_4gpu_g8_bs2_a100}"
REWARD_FUNCS="${REWARD_FUNCS:-hps gdino vlm_attr vlm_orm}"

export WANDB_PROJECT="CompGen-GRPO"
export WANDB_NAME="$EXP_NAME"
export WANDB_RUN_GROUP="advanced"           # 可选，用 git 分支名分组
# 用 EXP_NAME 派生固定 wandb run id，配合 WANDB_RESUME=allow：
#   - 第一次启动 → 新建 run；后续同 EXP_NAME 启动 → 自动接回原 run（曲线连续）
export WANDB_RUN_ID="$EXP_NAME"
export WANDB_RESUME=allow
# tiger 到 api.wandb.ai 网络偶发不稳；默认 90s 常超时。放宽到 600s。
# 若仍失败，把下一行注释放开切到 offline，事后 `wandb sync outputs/<EXP>/wandb/latest-run` 上传
export WANDB_INIT_TIMEOUT=600
# export WANDB_MODE=offline                  # 出网完全不通时改成 offline，事后 wandb sync
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
export TRITON_CACHE_DIR=/tmp/triton_cache_$USER
# 单节点训练禁用 NCCL 外部 plugin（/usr/local/lib/libnccl-net.so 在该 worker 上 nccl_p2p_ib_init segfault）
# 本机只用 NVLink + shared memory，不需要 IB
export NCCL_NET_PLUGIN=none
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=0
export NCCL_SHM_DISABLE=0
export DEBUG_MODE="true"
# 压掉无害但刷屏的 FutureWarning / DeprecationWarning（timm.models.layers、image_processor_class、torch.cuda.amp.autocast 等）
export PYTHONWARNINGS="ignore::FutureWarning,ignore::DeprecationWarning,ignore::UserWarning"
# 减少 NCCL 自身刷屏（保留 WARN 级别，丢掉 INFO）
export NCCL_DEBUG=WARN

# ── 输出目录（全部基于 EXP_NAME，换实验只改 EXP_NAME 这一处）──────
OUTPUT_DIR="$SCRIPT_DIR/outputs/$EXP_NAME"
export LOG_PATH="$OUTPUT_DIR/reward_log.txt"

# ── 路径配置（统一基于 REPO_ROOT，换机器无需改）──────────────────
REWARD_WEIGHT="$REPO_ROOT/src/t2i-r1/reward_weight"
# Janus 基座：默认放在 reward_weight/Janus-Pro-1B；如用 HF 缓存可改成对应 snapshot 路径
MODEL_PATH="${MODEL_PATH:-$REWARD_WEIGHT/Janus-Pro-1B}"
HF_DATASET="$REPO_ROOT/data/geneval_and_t2i_data_final.json"

HPS_CKPT="$REWARD_WEIGHT/HPSv2.1/HPS_v2.1_compressed.pt"
GDINO_CKPT="$REWARD_WEIGHT/groundingdino_swint_ogc.pth"
GDINO_CFG="utils/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
VLM_CKPT="$REWARD_WEIGHT/Qwen3-VL-2B-Instruct"
# ──────────────────────────────────────────────────────────────

mkdir -p "$OUTPUT_DIR"

# ── 把脚本后续所有 stdout/stderr 都固化到 train_main_<时间戳>.log ──
# 同名 EXP 多次启动不会互相覆盖；最新一次同时维护一个 train_main.log 软链方便 tail
# 启动命令：EXP_NAME=xxx nohup bash run_train.sh &
TS="$(date '+%Y%m%d_%H%M%S')"
LOG_FILE="$OUTPUT_DIR/train_main_${TS}.log"
ln -sf "$(basename "$LOG_FILE")" "$OUTPUT_DIR/train_main.log"
exec > >(tee "$LOG_FILE") 2>&1

echo "[run_train.sh] EXP_NAME=$EXP_NAME"
echo "[run_train.sh] OUTPUT_DIR=$OUTPUT_DIR"
echo "[run_train.sh] LOG_FILE=$LOG_FILE"
echo "[run_train.sh] REWARD_FUNCS=$REWARD_FUNCS"
echo "[run_train.sh] start at $(date '+%F %T')"

# ── 续训：传 RESUME=<ckpt 路径> 即从该 ckpt 续训；不传则从头训 ──
# 例：RESUME=outputs/train_main_4gpu_g8_bs2_a100/checkpoint-400 EXP_NAME=... bash run_train.sh
RESUME_ARG=""
if [ -n "${RESUME:-}" ]; then
  RESUME_ARG="--resume_from_checkpoint $RESUME"
  echo "[run_train.sh] RESUME from $RESUME"
fi

PYTHONPATH="$SCRIPT_DIR":$PYTHONPATH \
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
MASTER_PORT="${MASTER_PORT:-29500}" \
torchrun --nproc_per_node="${NPROC:-1}" \
--standalone \
--nnodes="1" \
open_r1/grpo.py \
--use_vllm=false \
--deepspeed "../configs/${ZERO_CFG:-zero2.json}" \
--output_dir $OUTPUT_DIR \
--model_name_or_path $MODEL_PATH \
--dataset_name $HF_DATASET \
--dataloader_num_workers 4 \
--dataloader_pin_memory true \
--max_prompt_length 512 \
--max_completion_length 1024 \
--temperature 1.0 \
--num_generations 8 \
--per_device_train_batch_size 1 \
--gradient_accumulation_steps 2 \
--logging_steps 5 \
--bf16=true \
--torch_dtype bfloat16 \
--report_to wandb \
--gradient_checkpointing=false \
--attn_implementation sdpa \
--max_steps "${MAX_STEPS:-1600}" \
--run_name "$EXP_NAME" \
--save_steps 400 \
--save_total_limit 5 \
--save_only_model=true \
--new_generations_image 1 \
--image_token_num_per_image 576 \
--cfg_weight 5 \
--reasoning_prompt_path "$REPO_ROOT/data/prompt/reasoning_prompt.txt" \
--reward_funcs $REWARD_FUNCS \
--beta 0.01 \
--tf32=true \
--learning_rate 1e-6 \
--hps_ckpt_path $HPS_CKPT \
--gdino_ckpt_path $GDINO_CKPT \
--gdino_config_path $GDINO_CFG \
--vlm_ckpt_path $VLM_CKPT \
$RESUME_ARG

echo "[run_train.sh] finished at $(date '+%F %T') (exit=$?)"
