#!/usr/bin/env bash
set -uo pipefail

# Read-only environment check. Run from any directory:
#   bash check_train_env.sh

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REWARD_ROOT="$REPO_ROOT/src/t2i-r1/reward_weight"
SOURCE_ROOT="$REPO_ROOT/src/t2i-r1/src"

pass() { printf '[ OK ] %s\n' "$*"; }
warn() { printf '[WARN] %s\n' "$*"; }
fail() { printf '[FAIL] %s\n' "$*"; }

printf 'T2I-RIA environment check\nrepository: %s\n\n' "$REPO_ROOT"

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv,noheader
else
  fail 'nvidia-smi is unavailable (run training checks on a GPU worker)'
fi

printf '\nPython and core packages\n'
if command -v python >/dev/null 2>&1; then
  PYTHONPATH="$SOURCE_ROOT${PYTHONPATH:+:$PYTHONPATH}" python - <<'PY'
import importlib

expected = {
    "torch": "2.5.1",
    "torchvision": "0.20.1",
    "torchaudio": "2.5.1",
    "transformers": "4.57.2",
    "trl": "0.16.0",
    "deepspeed": "0.15.4",
    "wandb": "0.26.1",
}

for module, wanted in expected.items():
    try:
        loaded = importlib.import_module(module)
        actual = str(getattr(loaded, "__version__", "unknown"))
        status = " OK " if actual.split("+", 1)[0] == wanted else "WARN"
        print(f"[{status}] {module}={actual} (reference={wanted})")
    except Exception as exc:
        print(f"[FAIL] {module}: {type(exc).__name__}: {exc}")

try:
    import torch
    print(
        f"       cuda_available={torch.cuda.is_available()} "
        f"device_count={torch.cuda.device_count()}"
    )
except Exception:
    pass

for module in ("groundingdino", "janus"):
    try:
        importlib.import_module(module)
        print(f"[ OK ] {module} import")
    except Exception as exc:
        print(f"[FAIL] {module}: {type(exc).__name__}: {exc}")
PY
else
  fail 'python is unavailable'
fi

printf '\nWeights and data\n'
check_path() {
  if [ -e "$1" ]; then pass "$2: $1"; else fail "$2 missing: $1"; fi
}
check_path "$REWARD_ROOT/Janus-Pro-1B/config.json" 'Janus-Pro-1B'
check_path "$REWARD_ROOT/Qwen3-VL-2B-Instruct/config.json" 'Qwen3-VL-2B-Instruct'
check_path "$REWARD_ROOT/HPSv2.1/HPS_v2.1_compressed.pt" 'HPS v2.1'
check_path "$REWARD_ROOT/groundingdino_swint_ogc.pth" 'GroundingDINO checkpoint'
check_path "$REWARD_ROOT/bert-base-uncased/config.json" 'GroundingDINO text encoder'
check_path "$REPO_ROOT/data/geneval_and_t2i_data_final.json" 'training prompts'
check_path "$REPO_ROOT/data/prompt/reasoning_prompt.txt" 'reasoning prompt'
check_path "$REPO_ROOT/src/t2i-r1/configs/zero2.json" 'DeepSpeed ZeRO-2 config'

if compgen -G "$SOURCE_ROOT/utils/GroundingDINO/groundingdino/_C*.so" >/dev/null; then
  pass 'GroundingDINO C++ extension is present'
else
  warn 'GroundingDINO C++ extension is absent; run setup_env.sh on the target CUDA worker'
fi
