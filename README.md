# T2I-RIA

**Reward Interaction in Small-Backbone BiCoT-GRPO**

T2I-RIA is an empirical study of multi-reward reinforcement learning for
compositional text-to-image generation. It adapts Janus-Pro with the BiCoT-GRPO
pipeline inherited from [T2I-R1](https://github.com/CaraJ7/T2I-R1) and studies
the interaction of four reward signals:

- HPSv2.1 preference reward;
- Grounding DINO object, spatial, and numeracy reward;
- Qwen3-VL-2B attribute reward;
- Qwen3-VL-2B ORM-style semantic reward.

The repository name `CompGen-GRPO` is retained for continuity. T2I-RIA is the
public paper and experiment name; it is an analysis setting, not a new GRPO
optimizer.

## Results

### Local evaluations

All local rows use the same generation and T2I-CompBench evaluation pipeline.
The 1B and 7B values are best-observed checkpoints selected from sparse offline
evaluations on the reported test benchmark.

| Model | Backbone | Color | Shape | Texture | Spatial | Non-sp. | Complex | Avg. |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Baseline | Janus-Pro-1B | 0.3514 | 0.2028 | 0.2740 | 0.0738 | 0.2633 | 0.2314 | 0.2328 |
| T2I-RIA | Janus-Pro-1B | 0.7985 | 0.4772 | 0.6831 | 0.2971 | 0.3046 | 0.3716 | 0.4887 |
| T2I-RIA | Janus-Pro-7B | 0.8267 | 0.5805 | 0.7317 | 0.3435 | 0.3094 | 0.4040 | 0.5326 |

The complete 1B adaptation raises the local average from 0.2328 to 0.4887
(+0.2559, +109.9% relative). This validates the complete adaptation recipe; it
does not isolate the gain of the new reward composition relative to a matched
T2I-R1 reward control.

### External context

The following values are reported by T2I-R1 and are not locally reproduced.
Its average is the arithmetic mean of the six reported category scores.

| Model | Backbone | Color | Shape | Texture | Spatial | Non-sp. | Complex | Avg. |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Baseline (reported) | Janus-Pro-7B | 0.6359 | 0.3528 | 0.4936 | 0.2061 | 0.3085 | 0.3559 | 0.3921 |
| T2I-R1 (reported) | Janus-Pro-7B | 0.8130 | 0.5852 | 0.7243 | 0.3378 | 0.3090 | 0.3993 | 0.5281 |

T2I-RIA-1B reaches 92.5% of the computed T2I-R1 average. T2I-RIA-7B is
numerically 0.0045 higher, but this external, single-run, test-selected
comparison is not evidence of superiority.

Full-precision selected results and matched-budget ablations are in
[`results/main_results.csv`](results/main_results.csv). Sparse checkpoint values
and evidence limitations are described in [`results/main_results.md`](results/main_results.md).

## Reward aggregation

The four active scores are summed with unit coefficients:

```text
R = R_hps + R_gdino + R_attr + R_orm
```

There is no per-reward normalization before aggregation. HPSv2.1 is an
unmapped cosine similarity, while the other three rewards return values in
`[0, 1]`. Unit coefficients therefore do not imply equal effective influence.
The total reward is group-normalized only after summation.

## Repository layout

```text
.
├── data/                              # 7,223 released training prompts
├── reproducibility/runtime_configs/   # frozen reconstructed run records
├── results/                           # paper-aligned result tables
├── scripts/                           # utility scripts
├── src/t2i-r1/configs/                # DeepSpeed configs
└── src/t2i-r1/src/
    ├── open_r1/                       # GRPO entrypoint and trainer
    ├── janus/                         # vendored Janus runtime
    ├── utils/                         # reward implementations + Grounding DINO
    ├── run_train.sh                   # portable training launcher
    ├── run_generate.sh                # T2I-CompBench image generation
    └── run_eval.sh                    # official evaluator wrapper
```

## Installation

The reported worker environment used Python 3.11, PyTorch 2.5.1,
torchvision 0.20.1, Transformers 4.57.2, TRL 0.16.0, and DeepSpeed 0.15.4.
The paper used SDPA; FlashAttention is not required.

For CUDA 12.4, the matching PyTorch installation is:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
  --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements.txt
python -m pip install -e src/t2i-r1/src/utils/GroundingDINO --no-build-isolation
```

`setup_env.sh` performs the same setup. A CUDA toolkit with `nvcc` is required
to compile the vendored Grounding DINO extension.

## Download weights

```bash
source .venv/bin/activate
bash download_weights.sh
bash check_train_env.sh
```

Weights are stored under `src/t2i-r1/reward_weight/` and are excluded from Git.
See [`REWARD_WEIGHTS.md`](REWARD_WEIGHTS.md) for sources, optional 7B download,
and the expected directory layout. The historical runs did not pin immutable
upstream model revisions, so exact historical snapshots cannot be recovered.

## Training

The launcher accepts environment-variable overrides. Its defaults reproduce the
reported BF16 Full-1B training shape: four processes, two prompts per device,
two accumulation steps, group size eight, 600 optimizer steps, and checkpoints
every 200 steps.

```bash
NPROC=4 CUDA_VISIBLE_DEVICES=0,1,2,3 \
  bash src/t2i-r1/src/run_train.sh
```

This corresponds to 16 prompt instances and 128 generated candidates per
optimizer step. A one-process smoke test can be run with:

```bash
NPROC=1 MAX_STEPS=5 PER_DEVICE_TRAIN_BATCH_SIZE=1 \
GRADIENT_ACCUMULATION_STEPS=1 REPORT_TO=none DEBUG_MODE=false \
  bash src/t2i-r1/src/run_train.sh
```

The exact historical runtime records are not inferred from the current
launcher. See [`reproducibility/runtime_configs/README.md`](reproducibility/runtime_configs/README.md)
and [`paper_runs.json`](reproducibility/runtime_configs/paper_runs.json).

## Generation and evaluation

Clone the official T2I-CompBench repository separately and pass its path:

```bash
export T2I_COMPBENCH_DIR=/path/to/T2I-CompBench

CUDA_VISIBLE_DEVICES=0,1 bash src/t2i-r1/src/run_generate.sh \
  --nproc 2 \
  --model_path src/t2i-r1/src/outputs/full/checkpoint-600 \
  --save_root eval_results/full_600

bash src/t2i-r1/src/run_eval.sh \
  --bench_dir "$T2I_COMPBENCH_DIR" \
  --eval_root eval_results \
  --model full_600 \
  --task all \
  --gpu 0
```

The paper used T2I-CompBench commit
`1b7094991a57f3c22abdd4f6e8ba6c1a15517073`, 300 prompts per category,
10 images per prompt, and 18,000 generated images per evaluated checkpoint.
The generation seed initializes one RNG stream per rank; it is not an
independently reset per-prompt seed, and resuming with `--skip_existing` can
change subsequent RNG consumption.

## Historical implementation boundaries

The release preserves the implementation used for the reported runs. Known
historical behaviors include:

- color, shape, and texture are routed to the default ORM template because the
  preprocessor returns the original task label;
- an empty attribute answer can pass the bidirectional substring match;
- numeracy NMS receives normalized `cxcywh` boxes although the operator expects
  `xyxy`;
- historical upstream model snapshot revisions and failure/fallback rates were
  not recorded.

These behaviors are disclosed rather than retroactively changed. Corrected
variants require new controlled experiments.

## Acknowledgements

- [T2I-R1](https://github.com/CaraJ7/T2I-R1)
- [Janus](https://github.com/deepseek-ai/Janus)
- [T2I-CompBench](https://github.com/Karine-Huang/T2I-CompBench)
- [Grounding DINO](https://github.com/IDEA-Research/GroundingDINO)
- [HPSv2](https://github.com/tgxs002/HPSv2)

## License

Project-authored code is released under Apache-2.0. Vendored components and
downloaded model weights remain subject to their upstream licenses.
