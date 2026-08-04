# AGENTS.md — T2I-RIA repository guide

This file contains stable repository facts. Current release work and transient
TODOs belong in `MEMORY.md`.

New-agent sequence: read this file, read `MEMORY.md`, inspect `git status`, then
check `nvidia-smi` and `pgrep -af 'torchrun|open_r1/grpo'` before changing a
training worker. Update `MEMORY.md` before ending a work session.

## Project identity

The public paper/setting name is **T2I-RIA (Text-to-Image Reward Interaction
Analysis)**, titled **T2I-RIA: Reward Interaction in Small-Backbone
BiCoT-GRPO**. The historical repository name `CompGen-GRPO` remains in the
GitHub URL. T2I-RIA is an analysis setting, not a new GRPO optimizer.

T2I-RIA extends the T2I-R1 BiCoT-GRPO pipeline to a small-backbone study led by
Janus-Pro-1B. Janus-Pro-7B is a scale-extension result.

## Authoritative paths

- Training entrypoint: `src/t2i-r1/src/run_train.sh`
- Trainer: `src/t2i-r1/src/open_r1/trainer/grpo_trainer.py`
- Rewards: `src/t2i-r1/src/utils/reward_*.py`
- Generation: `src/t2i-r1/src/run_generate.sh`
- Evaluation wrapper: `src/t2i-r1/src/run_eval.sh`
- Historical run records: `reproducibility/runtime_configs/paper_runs.json`
- Public reproduction guide: `REPRODUCE.md`
- Paper-aligned scores: `results/main_results.csv`

Weights live in `src/t2i-r1/reward_weight/` and must never be committed.
Training outputs, W&B directories, generated images, logs, and benchmark
artifacts are also excluded from Git.

## Reward aggregation

The four available terms are `hps`, `gdino`, `vlm_attr`, and `vlm_orm`:

```text
R = R_hps + R_gdino + R_attr + R_orm
```

This is a unit-coefficient **raw** sum. There is no per-reward normalization,
rescaling, or clipping before aggregation. HPS is an unmapped cosine similarity
while the other terms are in `[0,1]`; never describe this as equal effective
weighting. Group-wise GRPO normalization happens after summation.

HPSv2.1 is the inherited preference anchor. V1 leave-one-out diagnostics remove
Attr, GDino, or ORM, but not HPS.

## Stable experiment facts

- Training data: released T2I-R1 JSONL artifact, 7,223 unique prompts.
- Group size: 8; learning rate: `1e-6`; beta: `0.01`.
- Final paper reward model loading: Qwen3-VL-2B in BF16.
- Full-1B and w/o Attr/GDino: 4 processes, batch 2, accumulation 2, hence
  16 prompts and 128 generated candidates per optimizer step.
- w/o ORM: 4 processes, batch 1, accumulation 4, also 16/128.
- Full-7B: 4 processes, batch 1, accumulation 2, hence 8/64.
- Each setting is one training run. Use observational language; do not claim
  statistical significance or strong causal identification.
- Local evaluation: six T2I-CompBench categories, 300 prompts/category,
  10 images/prompt, fixed RNG stream, 18,000 images/checkpoint.
- Only a sparse set of saved checkpoints was evaluated. Main rows report the
  best observed test result among those checkpoints.

Historical code and runtime arguments are different evidence types. When they
conflict, follow the hierarchy documented in
`reproducibility/runtime_configs/README.md`; do not rewrite history by treating
today's launcher as the original command.

## Reproducibility boundaries

The historical implementation intentionally remains unchanged in the reward
and trainer source used for reported results. Disclosed behaviors include ORM
default routing for color/shape/texture, possible empty-answer Attr matching,
and incorrect `cxcywh` input to numeracy NMS. Upstream model revisions and
several failure rates were not recorded. Correcting these requires new
controlled experiments, not an undocumented V1 patch.

The release launcher and path handling may be made portable without claiming
that they were used historically. Any change to reward/trainer semantics must
be explicitly separated from the reported V1 implementation.

## Environment and launch

Reference core stack: Python 3.11.2, PyTorch 2.5.1 + CUDA 12.4 wheels,
torchvision 0.20.1, Transformers 4.57.2, TRL 0.16.0, DeepSpeed 0.15.4, W&B
0.26.1, and SDPA. FlashAttention is not required.

Use `setup_env.sh`, `download_weights.sh`, and `check_train_env.sh`. The default
training shape is launched with:

```bash
NPROC=4 CUDA_VISIBLE_DEVICES=0,1,2,3 bash src/t2i-r1/src/run_train.sh
```

All important parameters accept environment-variable overrides. Explain the
throughput, memory, and optimization consequences of any launcher change.

## Known operational pitfalls

- Do not pass multiple space-separated values to `--report_to`; the historical
  parser stack accepts a single value.
- Grounding DINO's extension must be compiled against the installed PyTorch and
  CUDA stack.
- Some worker builds expose only CPU torchvision NMS; the historical reward
  path moves NMS inputs to CPU.
- `GDINO_TEXT_ENCODER` selects the local bert-base-uncased snapshot; the config
  otherwise uses its public Hub identifier.
- Never terminate an active training job unless the user explicitly asks.

## Git rules

The development branch is `advanced`; GitHub's release/default branch must be
updated explicitly during publication. Never force-push, hard-reset the work
tree, commit weights or outputs, or delete another contributor's branch.
Preserve unrelated user changes.

## Maintenance

At session end, update `MEMORY.md` with the current run state, material edits,
new reproducibility risks, and prioritized next steps. Update this file only
when stable paths, environment, launch conventions, or reproducible pitfalls
change.
