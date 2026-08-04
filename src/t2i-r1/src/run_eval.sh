#!/usr/bin/env bash
# =============================================================================
# T2I-CompBench Evaluation Script (v2：支持指定 GPU + 断点恢复)
# =============================================================================
#
# 作用：给 run_generate.sh 产生的图片按 T2I-CompBench 协议逐张打分。
#   - color / shape / texture       → BLIP-VQA
#   - spatial                       → UniDet (2D detection)
#   - non_spatial                   → CLIPScore
#   - complex                       → 上面 3 个 + 3-in-1 aggregation
#
# 使用方法：
#
#   T2I_COMPBENCH_DIR=/path/to/T2I-CompBench bash run_eval.sh \
#     --model <MODEL_NAME> [--task <TASK>] [--gpu <N>]
#
#   参数：
#     --model MODEL   模型子目录名（eval_results 下的目录）。
#                     支持 baseline / finetuned / both，也支持任意自定义名
#                     （比如 wo_gdino_400、full_1600 等）。默认 both。
#                     **支持多个**：用逗号分隔，脚本会串行依次跑，各自结果落到
#                     eval_results/<MODEL>/ 各自目录下。
#                     示例：--model wo_gdino_400,wo_attr_400,wo_orm_400
#     --task TASK     单个任务名或 all。可选：
#                       color, shape, texture, spatial, non_spatial, complex, all
#                     默认 all（跑 6 类）。
#                     **支持多个**：三种写法：
#                       --task spatial complex        （空格分隔）
#                       --task spatial,complex        （逗号分隔）
#                       --task all                    （6 类全跑）
#     --gpu N         指定单张 GPU（例：--gpu 3）。优先级：
#                       --gpu > 环境变量 CUDA_VISIBLE_DEVICES > 0
#                     T2I-CompBench 三个评测器都是单卡单进程，多张 GPU 用不上。
#     --bench_dir DIR T2I-CompBench clone；也可设置 T2I_COMPBENCH_DIR。
#     --eval_root DIR 生成图与评测结果根目录；默认仓库下 eval_results/。
#
# 特性（v2）：
#   1) **可中断恢复**：每个 task 开跑前会检查产物 JSON 是否已存在且记录数
#      等于 samples/*.png 数量，是的话直接 SKIP，避免重复计算。
#      complex 内部 4 个子步骤也分别做 SKIP 判断。
#   2) **失败继续**：某个 (model, task) 挂了不会中止整脚本，最后打印一份
#      OK/SKIP/FAIL 汇总；重跑同一命令会自动只补 FAIL 那几个。
#   3) **GPU 可选**：--gpu N 指定单卡；不传就沿用环境或默认 0。
#
# 示例：
#
#   # 1. 跑单个 ablation，指定 GPU 3
#   bash run_eval.sh --model wo_gdino_400 --task all --gpu 3
#
#   # 2. 并行跑多个 ablation（3 个终端各占一张卡）
#   #   终端 A:
#   bash run_eval.sh --model wo_gdino_400 --task all --gpu 0
#   #   终端 B:
#   bash run_eval.sh --model wo_attr_400  --task all --gpu 1
#   #   终端 C:
#   bash run_eval.sh --model wo_orm_400   --task all --gpu 2
#
#   # 3. 跑到一半 Ctrl+C 或崩溃，修好问题后重跑同一命令，自动跳过已完成
#   bash run_eval.sh --model wo_gdino_400 --task all --gpu 3
#
#   # 4. 只单独补一类
#   bash run_eval.sh --model wo_gdino_400 --task spatial --gpu 3
#
#   # 5. 一张卡串行跑多个 ablation（依次落各自目录）
#   bash run_eval.sh --model wo_gdino_400,wo_attr_400,wo_orm_400 --task all --gpu 3
#
# 产物：
#   eval_results/<MODEL>/<TASK>/annotation_*/vqa_result.json    # 逐图 0-1 分数
#   eval_results/<MODEL>/<TASK>/annotation_*/blip_vqa_score.txt # 类平均分
#   eval_results/logs/eval_<MODEL>_<TASK>_<TS>.log              # 本次启动日志
#   eval_results/logs/eval_latest.log                           # 软链 → 最新日志
#
# 注意：本脚本不做训练/推理，只调用 T2I-CompBench 里的评测器；请确保
#   run_generate.sh 已经产出 <MODEL>/<TASK>/samples/*.png。
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
BENCH_DIR="${T2I_COMPBENCH_DIR:-}"
EVAL_ROOT="${EVAL_ROOT:-$REPO_ROOT/eval_results}"

# 屏蔽 transformers 的 INFO/WARNING（BertLMHeadModel GenerationMixin 提示、slow image processor 等）
export TRANSFORMERS_VERBOSITY=error
export PYTHONWARNINGS="ignore::FutureWarning,ignore::DeprecationWarning,ignore::UserWarning"
export TOKENIZERS_PARALLELISM=false

# ── 默认参数 ──
MODEL="both"
TASK="all"
GPU_ARG=""

# ── 参数解析 ──
# --task 支持三种写法：
#   1) --task all                    → 6 类全跑
#   2) --task spatial,complex        → 逗号分隔
#   3) --task spatial complex        → 空格分隔（会一直吞到下一个 --flag 为止）
TASK_TOKENS=()
while [[ $# -gt 0 ]]; do
    case $1 in
        --model) MODEL="$2"; shift 2 ;;
        --task)
            shift
            # 收集所有非 --flag 的后续参数
            while [[ $# -gt 0 && "$1" != --* ]]; do
                TASK_TOKENS+=("$1")
                shift
            done
            ;;
        --gpu)   GPU_ARG="$2"; shift 2 ;;
        --bench_dir) BENCH_DIR="$2"; shift 2 ;;
        --eval_root) EVAL_ROOT="$2"; shift 2 ;;
        -h|--help)
            # 打印文件头 docstring（从 shebang 之后的 # 注释块）
            awk '/^# =====/{n++} n==1{print; next} n>=2{exit}' "$0" | sed 's/^# \?//'
            exit 0
            ;;
        *) echo "Unknown arg: $1" >&2; exit 1 ;;
    esac
done

if [ -z "$BENCH_DIR" ]; then
    echo "Pass --bench_dir or set T2I_COMPBENCH_DIR." >&2
    exit 2
fi
export EVAL_ROOT

# 合并 --task 收集到的 token（可能是 "spatial complex" 或 "spatial,complex"）→ 逗号形式
if [ ${#TASK_TOKENS[@]} -gt 0 ]; then
    TASK=$(IFS=','; echo "${TASK_TOKENS[*]}")   # 空格分隔 → "spatial,complex"
    TASK=${TASK//,,/,}                           # 万一混用了逗号+空格，压掉重复逗号
fi

# ── GPU 选卡：--gpu > 环境变量 CUDA_VISIBLE_DEVICES > 0 ──
if [ -n "$GPU_ARG" ]; then
    export CUDA_VISIBLE_DEVICES="$GPU_ARG"
elif [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
    export CUDA_VISIBLE_DEVICES=0
fi

# ── log 固化 ──
LOG_DIR="$EVAL_ROOT/logs"
mkdir -p "$LOG_DIR"
TS="$(date '+%Y%m%d_%H%M%S')"
LOG_TAG="${MODEL//,/_}"
LOG_FILE="$LOG_DIR/eval_${LOG_TAG}_${TASK}_${TS}.log"
ln -sf "$(basename "$LOG_FILE")" "$LOG_DIR/eval_latest.log"
exec > >(tee "$LOG_FILE") 2>&1

echo "[run_eval.sh] LOG_FILE=$LOG_FILE"
echo "[run_eval.sh] start at $(date '+%F %T')"
echo ""
echo "=============================================="
echo "  Model  : $MODEL"
echo "  Task   : $TASK"
echo "  GPU    : $CUDA_VISIBLE_DEVICES"
echo "=============================================="

# ── helper: 检查某个 result json 是否已完整 ─────────────────────────────
# 判据：json 存在，且 len(json) == expected（通常等于 samples/*.png 数量）
is_json_done() {
    local json_file=$1
    local expected=$2
    [ -f "$json_file" ] || return 1
    local actual
    actual=$(python3 -c "import json,sys; d=json.load(open('$json_file')); print(len(d))" 2>/dev/null)
    [ -n "$actual" ] || return 1
    [ "$actual" = "$expected" ]
}

# ── 单个 (model, task) 评测 ─────────────────────────────────────────────
# 返回码：0 = OK（跑了并成功）；1 = FAIL；2 = SKIP（已完成，无需重跑）
run_task() {
    local MODEL_NAME=$1
    local TASK_NAME=$2
    local OUT_DIR="$EVAL_ROOT/$MODEL_NAME/$TASK_NAME"

    echo ""
    echo ">>> [$MODEL_NAME] $TASK_NAME"

    # 前置检查：samples/ 得有图（gen 阶段必须先跑）
    if [ ! -d "$OUT_DIR/samples" ]; then
        echo "  [FAIL] samples dir not found: $OUT_DIR/samples (先跑 run_generate.sh?)"
        return 1
    fi
    local expected
    expected=$(find "$OUT_DIR/samples" -maxdepth 1 -name "*.png" 2>/dev/null | wc -l)
    if [ "$expected" -eq 0 ]; then
        echo "  [FAIL] 0 PNG files in $OUT_DIR/samples"
        return 1
    fi

    case $TASK_NAME in
        color|shape|texture)
            if is_json_done "$OUT_DIR/annotation_blip/vqa_result.json" "$expected"; then
                echo "  [SKIP] annotation_blip/vqa_result.json 已完整（$expected 条），跳过"
                return 2
            fi
            cd "$BENCH_DIR/BLIPvqa_eval" || return 1
            python BLIP_vqa.py --out_dir "$OUT_DIR" || return 1
            ;;
        spatial)
            if is_json_done "$OUT_DIR/labels/annotation_obj_detection_2d/vqa_result.json" "$expected"; then
                echo "  [SKIP] spatial 已完整，跳过"
                return 2
            fi
            cd "$BENCH_DIR/UniDet_eval" || return 1
            python 2D_spatial_eval.py --outpath "$OUT_DIR" || return 1
            ;;
        non_spatial)
            if is_json_done "$OUT_DIR/annotation_clip/vqa_result.json" "$expected"; then
                echo "  [SKIP] non_spatial 已完整，跳过"
                return 2
            fi
            cd "$BENCH_DIR" || return 1
            python CLIPScore_eval/CLIP_similarity.py --outpath "$OUT_DIR" || return 1
            ;;
        complex)
            # complex 有 4 个子步骤，逐个 SKIP 判断
            local step1_done=0 step2_done=0 step3_done=0 step4_done=0

            if is_json_done "$OUT_DIR/annotation_blip/vqa_result.json" "$expected"; then
                echo "  [SKIP 1/4] BLIP-VQA 已完整"
                step1_done=1
            else
                echo "  [Step 1/4] BLIP-VQA for complex..."
                cd "$BENCH_DIR/BLIPvqa_eval" || return 1
                python BLIP_vqa.py --out_dir "$OUT_DIR" || return 1
            fi

            if is_json_done "$OUT_DIR/labels/annotation_obj_detection_2d/vqa_result.json" "$expected"; then
                echo "  [SKIP 2/4] UniDet 已完整"
                step2_done=1
            else
                echo "  [Step 2/4] UniDet for complex..."
                cd "$BENCH_DIR/UniDet_eval" || return 1
                python 2D_spatial_eval.py --outpath "$OUT_DIR" || return 1
            fi

            if is_json_done "$OUT_DIR/annotation_clip/vqa_result.json" "$expected"; then
                echo "  [SKIP 3/4] CLIPScore 已完整"
                step3_done=1
            else
                echo "  [Step 3/4] CLIPScore for complex..."
                cd "$BENCH_DIR" || return 1
                python CLIPScore_eval/CLIP_similarity.py --outpath "$OUT_DIR" || return 1
            fi

            if is_json_done "$OUT_DIR/annotation_3_in_1/vqa_result.json" "$expected"; then
                echo "  [SKIP 4/4] 3-in-1 已完整"
                step4_done=1
            else
                echo "  [Step 4/4] 3-in-1 aggregation..."
                cd "$BENCH_DIR/3_in_1_eval" || return 1
                python 3_in_1.py --outpath "$OUT_DIR" || return 1
            fi

            # 4 步全 SKIP 才算整体 SKIP，否则算 OK
            if [ $((step1_done + step2_done + step3_done + step4_done)) -eq 4 ]; then
                return 2
            fi
            ;;
        *)
            echo "  [ERROR] Unknown task: $TASK_NAME"
            return 1
            ;;
    esac

    echo "  [OK] $MODEL_NAME/$TASK_NAME done"
    return 0
}

# ── model / task 展开 ─────────────────────────────────────────────────
# --model 支持：
#   1) both              → baseline + finetuned
#   2) 单个自定义名      → 只跑那一个
#   3) a,b,c 逗号分隔    → 串行依次跑
if [ "$MODEL" = "both" ]; then
    MODELS=("finetuned" "baseline")
else
    IFS=',' read -r -a MODELS <<< "$MODEL"
fi

if [ "$TASK" = "all" ]; then
    TASKS=("color" "shape" "texture" "spatial" "non_spatial" "complex")
else
    IFS=',' read -r -a TASKS <<< "$TASK"
fi

echo ""
echo "[run_eval.sh] models to run: ${MODELS[*]}"
echo "[run_eval.sh] tasks  to run: ${TASKS[*]}"

# ── 主循环：失败不中止，收集状态 ───────────────────────────────────────
STATUS_KEYS=()
STATUS_CODES=()
for m in "${MODELS[@]}"; do
    for t in "${TASKS[@]}"; do
        run_task "$m" "$t"
        rc=$?
        STATUS_KEYS+=("$m/$t")
        STATUS_CODES+=("$rc")
    done
done

# ── 运行状态汇总 ──────────────────────────────────────────────────────
echo ""
echo "=============================================="
echo "  RUN STATUS"
echo "=============================================="
n_ok=0; n_skip=0; n_fail=0
for i in "${!STATUS_KEYS[@]}"; do
    code=${STATUS_CODES[$i]}
    key=${STATUS_KEYS[$i]}
    case $code in
        0) echo "  [OK  ]  $key"; n_ok=$((n_ok+1)) ;;
        2) echo "  [SKIP]  $key"; n_skip=$((n_skip+1)) ;;
        *) echo "  [FAIL]  $key"; n_fail=$((n_fail+1)) ;;
    esac
done
echo "  ----------------------------------------"
echo "  OK=$n_ok  SKIP=$n_skip  FAIL=$n_fail"

# ── 分数汇总表（当前只支持 baseline vs finetuned 两列）─────────────────
echo ""
echo "=============================================="
echo "  CURRENT RESULTS  (baseline vs finetuned)"
echo "=============================================="

python3 - << 'PYEOF'
import json, os

EVAL_ROOT = os.environ["EVAL_ROOT"]

RESULT_PATHS = {
    "color":       "annotation_blip/vqa_result.json",
    "shape":       "annotation_blip/vqa_result.json",
    "texture":     "annotation_blip/vqa_result.json",
    "spatial":     "labels/annotation_obj_detection_2d/vqa_result.json",
    "non_spatial": "annotation_clip/vqa_result.json",
    "complex":     "annotation_3_in_1/vqa_result.json",
}

CATEGORIES = ["color", "shape", "texture", "spatial", "non_spatial", "complex"]

def load_score(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        data = json.load(f)
    if not data:
        return None
    scores = [float(d['answer']) for d in data]
    return sum(scores) / len(scores)

sep = "-" * 48
print(f"\n  {'Category':12s}  {'Baseline':>10}  {'Finetuned':>10}  {'Delta':>8}")
print(f"  {sep}")

base_vals, fine_vals = [], []
for cat in CATEGORIES:
    rel = RESULT_PATHS[cat]
    b = load_score(f"{EVAL_ROOT}/baseline/{cat}/{rel}")
    f = load_score(f"{EVAL_ROOT}/finetuned/{cat}/{rel}")
    base_vals.append(b)
    fine_vals.append(f)

    b_str = f"{b:.4f}" if b is not None else "   N/A"
    f_str = f"{f:.4f}" if f is not None else "   N/A"
    if b is not None and f is not None:
        delta_str = f"{f-b:+.4f}"
    else:
        delta_str = "   N/A"
    print(f"  {cat:12s}  {b_str:>10}  {f_str:>10}  {delta_str:>8}")

print(f"  {sep}")
valid = [(b, f) for b, f in zip(base_vals, fine_vals) if b is not None and f is not None]
if valid:
    avg_b = sum(b for b, _ in valid) / len(valid)
    avg_f = sum(f for _, f in valid) / len(valid)
    print(f"  {'Average':12s}  {avg_b:>10.4f}  {avg_f:>10.4f}  {avg_f-avg_b:>+8.4f}")
else:
    print(f"  {'Average':12s}  {'N/A':>10}  {'N/A':>10}  {'N/A':>8}")
print()
PYEOF

echo "[run_eval.sh] finished at $(date '+%F %T') (exit=$?)"
