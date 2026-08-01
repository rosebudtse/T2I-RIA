#!/bin/bash
# =============================================================================
# 上传 checkpoint 到 HuggingFace model repo（可重复运行，增量上传）
# =============================================================================
# 前置：
#   1) pip install -U huggingface_hub
#   2) hf auth login       （去 https://huggingface.co/settings/tokens 拿 write token）
#
# Usage:
#   HF_USER=<你的HF用户名> bash scripts/hf_upload/01_upload_checkpoints.sh
#
# 复用（本脚本设计目标）：
#   新增 checkpoint 想传 → 直接把它加进 CKPTS 数组，重跑即可。
#   跑之前先拉一次 HF 端的现存文件列表（list_repo_files），若
#   "$c/config.json" 已在远端 → 整个 checkpoint 前置 SKIP，不走网络上传。
#   万一漏判（remote list 拉不到 / 部分文件缺失），hf upload
#   自身也走 LFS hash 校验，会二次跳过已存在的相同 blob。
# =============================================================================

set -eu

HF_USER="${HF_USER:?环境变量 HF_USER 必须传入，例如 HF_USER=xiezifan bash 01_upload_checkpoints.sh}"
REPO_NAME="${REPO_NAME:-CompGen-GRPO-checkpoints}"
PRIVATE="${PRIVATE:-true}"   # 默认 private
REPO_ID="$HF_USER/$REPO_NAME"

OUTPUTS="/mlx_devbox/users/xiezifan/playground/CompGen-GRPO/src/t2i-r1/src/outputs"

# ── 待上传 checkpoint 列表（新增就加到这里）─────
CKPTS=(
    "ablation_wo_attr/checkpoint-400"
    "ablation_wo_attr/checkpoint-600"
    "ablation_wo_attr/checkpoint-800"
    "ablation_wo_gdino/checkpoint-600"
    "ablation_wo_orm/checkpoint-200"
    "ablation_wo_orm/checkpoint-400"
    "ablation_wo_orm/checkpoint-600"
    "ablation_wo_orm/checkpoint-800"
    "full/checkpoint-400"
    "full/checkpoint-600"
    "full_7b/checkpoint-400"
    "full_7b/checkpoint-800"
)

echo "=========================================="
echo "  Upload Checkpoints to HF"
echo "  Repo: $REPO_ID (private=$PRIVATE)"
echo "=========================================="

# ── [1] Ensure repo exists ─────────────────────
echo ""
echo "[1] Ensure repo exists"
if [ "$PRIVATE" = "true" ]; then
    hf repo create "$REPO_ID" --repo-type model --private --exist-ok
else
    hf repo create "$REPO_ID" --repo-type model --exist-ok
fi

# ── [2] 拉远端文件清单，用作前置 skip 依据 ────────
echo ""
echo "[2] Fetching remote file list from HF..."
REMOTE_FILES=$(python3 - "$REPO_ID" <<'PYEOF' 2>/dev/null || true
import sys
from huggingface_hub import HfApi
try:
    files = HfApi().list_repo_files(sys.argv[1], repo_type="model")
    print("\n".join(files))
except Exception:
    pass
PYEOF
)
n_remote=$(printf "%s\n" "$REMOTE_FILES" | grep -cve '^$' || true)
echo "  远端已有 $n_remote 个文件"

# ── [3] Upload README ────────────────────────
README="/mlx_devbox/users/xiezifan/playground/CompGen-GRPO/scripts/hf_upload/README_checkpoints.md"
if [ -f "$README" ]; then
    echo ""
    echo "[3] Uploading README..."
    hf upload "$REPO_ID" "$README" README.md --repo-type model --commit-message "Update README" || true
fi

# ── [4] 循环上传 checkpoint（前置 skip + LFS 增量兜底）─────
echo ""
echo "[4] Uploading checkpoints..."
uploaded=0
skipped_remote=0
skipped_local=0
failed=0
for c in "${CKPTS[@]}"; do
    local_dir="$OUTPUTS/$c"

    # 前置 skip: HF 已有 $c/config.json 就跳过（一次网络也不走）
    if printf "%s\n" "$REMOTE_FILES" | grep -Fxq "$c/config.json"; then
        echo "  [SKIP-REMOTE] $c  (HF 端已存在)"
        skipped_remote=$((skipped_remote+1))
        continue
    fi

    # 本地完整性检查
    if [ ! -d "$local_dir" ]; then
        echo "  [SKIP-LOCAL]  $c  (本地目录不存在)"
        skipped_local=$((skipped_local+1))
        continue
    fi
    if [ ! -f "$local_dir/config.json" ]; then
        echo "  [SKIP-LOCAL]  $c  (缺 config.json)"
        skipped_local=$((skipped_local+1))
        continue
    fi
    has_st=0
    for f in "$local_dir"/*.safetensors; do
        [ -f "$f" ] && has_st=$((has_st+1))
    done
    if [ "$has_st" = 0 ]; then
        echo "  [SKIP-LOCAL]  $c  (无 safetensors)"
        skipped_local=$((skipped_local+1))
        continue
    fi

    size=$(du -sh "$local_dir" | awk '{print $1}')
    echo ""
    echo "  → $c  ($size)"
    if hf upload "$REPO_ID" "$local_dir" "$c" \
            --repo-type model \
            --commit-message "Add/update $c"; then
        uploaded=$((uploaded+1))
    else
        echo "  [FAIL] $c 上传失败"
        failed=$((failed+1))
    fi
done

echo ""
echo "=========================================="
echo "  Done. uploaded=$uploaded  skipped_remote=$skipped_remote  skipped_local=$skipped_local  failed=$failed"
echo "  View: https://huggingface.co/$REPO_ID"
echo "=========================================="
