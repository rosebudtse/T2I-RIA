# Reconstructed historical runtime configurations

`paper_runs.json` freezes the configurations used by the runs reported in the
T2I-RIA paper. These records are **reconstructed artifacts**, not claims that
the exact commands were committed before training.

## Evidence hierarchy

When fields disagree, use the following order:

1. Frozen W&B run metadata, config, history, and benchmark output artifacts.
2. The paper-frozen reconstruction in `paper_runs.json`, including the audited
   four-process world size used for sample accounting.
3. Source code at the recorded Git commit, which establishes implementation
   behavior but not necessarily the command-line overrides used at runtime.
4. Defaults in the current portable launcher, which are provided for future
   reproduction and are not evidence of a historical run.

The distinction matters for commit `1c5138f`: its checked-in launcher defaults
do not match the runtime arguments captured for the initial Full-1B run. The
paper and this artifact therefore use the frozen runtime record for historical
configuration claims. A W&B `visible_gpu_count` records the worker inventory;
it is not automatically the distributed world size.

## Shared evaluation protocol

Every evaluated checkpoint uses six T2I-CompBench categories, 300 prompts per
category, ten generated images per prompt, and a fixed seed stream. Benchmark
evaluation is sparse and checkpoint-level; W&B training curves are continuous
logs and must not be interpreted as benchmark curves.

No model or reward checkpoint revision was explicitly pinned at download time.
See `REPRODUCE.md` for current installation and execution instructions.
