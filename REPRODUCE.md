# Reproducing T2I-RIA

This guide separates three different goals:

1. running the released implementation on a new machine;
2. recreating the reported training shape from frozen runtime records;
3. reproducing T2I-CompBench generation and evaluation.

The historical runs did not pin immutable upstream model-weight revisions.
Exact bitwise reconstruction is therefore not possible; the repository
preserves the recorded code revisions, arguments, data artifact, and evaluation
revision that remain available.

## 1. Environment

Reported core environment:

| Component | Version |
|---|---|
| Python | 3.11.2 |
| PyTorch | 2.5.1 |
| torchvision | 0.20.1+cu124 |
| torchaudio | 2.5.1+cu124 |
| Transformers | 4.57.2 |
| TRL | 0.16.0 |
| DeepSpeed | 0.15.4 |
| W&B | 0.26.1 |
| Attention backend | SDPA |

Create the environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
  --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements.txt
python -m pip install -e src/t2i-r1/src/utils/GroundingDINO --no-build-isolation
```

The final command requires a CUDA toolkit and `nvcc`. `setup_env.sh` automates
these steps without deleting an existing environment unless
`RECREATE_VENV=1` is explicitly set.

## 2. Weights

```bash
bash download_weights.sh
bash check_train_env.sh
```

The default download includes Janus-Pro-1B, Qwen3-VL-2B-Instruct, HPSv2.1,
Grounding DINO SwinT-OGC, and bert-base-uncased. Set `DOWNLOAD_7B=1` to also
download Janus-Pro-7B. See `REWARD_WEIGHTS.md` for the exact layout.

## 3. Data

Training uses the released T2I-R1 artifact:

```text
data/geneval_and_t2i_data_final.json
```

Despite the extension, it is JSONL and contains 7,223 records with 7,223 unique
prompt strings. Its SHA-256 is:

```text
96b956b5c92477ea2ff13cdd0c805f762a9d9fd5af43141ada6850ab3f6a0171
```

## 4. Runtime evidence and historical configs

The current `run_train.sh` is a portable release launcher. It is not treated as
the source of truth for every historical invocation.

The evidence hierarchy is:

1. frozen W&B `metadata.args` and run config for actual CLI values;
2. W&B-recorded Git commit for reward/trainer implementation;
3. frozen process-count and prompt-sample accounting where visible GPU inventory
   does not identify torchrun world size;
4. current launcher defaults only for new runs.

The normalized records for all six paper runs are stored in
`reproducibility/runtime_configs/paper_runs.json`.

The initial Full-1B run is the important provenance example. Its W&B record
contains group size 8, per-device batch 2, accumulation 2, beta 0.01, max steps
1,600, and save steps 400. The committed launcher at `1c5138f` instead contains
defaults 4, 1, 4, 0, 2,000, and 500. The code revision is authoritative for the
NF4/GPU-NMS reward implementation; the runtime record is authoritative for the
actual invocation. Checking out `1c5138f` alone is not a complete launcher
snapshot.

## 5. Training

### Full-1B BF16 refinement shape

```bash
NPROC=4 \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
EXP_NAME=full_reproduction \
MODEL_VARIANT=1B \
PER_DEVICE_TRAIN_BATCH_SIZE=2 \
GRADIENT_ACCUMULATION_STEPS=2 \
NUM_GENERATIONS=8 \
MAX_STEPS=600 \
SAVE_STEPS=200 \
BETA=0.01 \
LEARNING_RATE=1e-6 \
REPORT_TO=wandb \
bash src/t2i-r1/src/run_train.sh
```

This is 16 prompt instances and 128 generated candidates per optimizer step.

### Smoke test

```bash
NPROC=1 \
CUDA_VISIBLE_DEVICES=0 \
PER_DEVICE_TRAIN_BATCH_SIZE=1 \
GRADIENT_ACCUMULATION_STEPS=1 \
MAX_STEPS=5 \
REPORT_TO=none \
DEBUG_MODE=false \
bash src/t2i-r1/src/run_train.sh
```

The smoke test checks integration only; it does not reproduce a reported result.

## 6. Generation

Clone T2I-CompBench at the frozen evaluation revision:

```bash
git clone https://github.com/Karine-Huang/T2I-CompBench.git
git -C T2I-CompBench checkout 1b7094991a57f3c22abdd4f6e8ba6c1a15517073
export T2I_COMPBENCH_DIR="$PWD/T2I-CompBench"
```

Generate 10 images per prompt:

```bash
CUDA_VISIBLE_DEVICES=0,1 bash src/t2i-r1/src/run_generate.sh \
  --nproc 2 \
  --model_path src/t2i-r1/src/outputs/full/checkpoint-600 \
  --save_root eval_results/full_600 \
  --num_generation 10 \
  --cfg_weight 5.0 \
  --temperature 1.0 \
  --seed 42
```

The seed initializes one RNG stream per rank. Prompt generations consume that
stream sequentially; `--skip_existing` changes later RNG consumption if earlier
prompts are skipped. Do not describe this as independent per-prompt reseeding.

## 7. Evaluation

```bash
bash src/t2i-r1/src/run_eval.sh \
  --bench_dir "$T2I_COMPBENCH_DIR" \
  --eval_root eval_results \
  --model full_600 \
  --task all \
  --gpu 0
```

The official six categories each contain 300 prompts. With 10 images per
prompt, evaluation uses 18,000 images per checkpoint.

## 8. Known historical behaviors

The reported results retain the original implementation, including:

- raw unit-coefficient reward summation without per-reward normalization;
- default ORM routing for color, shape, and texture;
- possible credit for an empty attribute answer through substring matching;
- numeracy NMS applied to normalized `cxcywh` boxes rather than converted
  `xyxy` boxes;
- unavailable historical empty-answer, parse-failure, and NMS-failure rates.

These are disclosure boundaries, not pending statistics. Fixing them belongs to
a new controlled experiment rather than a retroactive rewrite of V1.
