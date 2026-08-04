# MEMORY.md — T2I-RIA public artifact status

Last updated: 2026-08-04 by Codex artifact audit

Branch: `advanced` (public artifact release is published on `origin/advanced`)

Repository: `rosebudtse/CompGen-GRPO`

## Runtime state

Paper experiments are complete. This machine is not a GPU worker:
`nvidia-smi` is unavailable and process inspection is restricted, so no
training or evaluation was started or stopped during this audit.

## Current release work

The public artifact was re-audited after the paper reached ArXiv-ready status.
The working tree now contains:

- paper-aligned T2I-RIA naming, claims, and result tables;
- a portable environment setup and a single pinned dependency surface;
- corrected weight paths and a local/Hub-selectable Grounding DINO encoder;
- portable training, generation, and evaluation launchers;
- reconstructed historical run records plus an explicit evidence hierarchy;
- release hygiene rules in `.gitignore` and `.gitattributes`;
- a `CITATION.cff` skeleton pending the ArXiv identifier.
- removal from Git tracking of the broken `image-gen` gitlink, obsolete
  archives/upload helpers, old figures/TODOs, and machine-specific logs; local
  workstation copies remain preserved and ignored.

Historical runtime claims must use
`reproducibility/runtime_configs/paper_runs.json`, not current launcher defaults.
The audited `1c5138f` launcher/runtime mismatch is disclosed there and in
`REPRODUCE.md`.

## Remaining blockers before public release

### P0

1. Validate on a clean GPU worker: environment check, Grounding DINO build,
   one-step smoke training, one-prompt generation, and evaluator startup.
2. Promote the audited artifact to the default branch. `origin/main` diverges
   by one old planning commit (`6925267`) whose StructComp document contains
   superseded V1 numbers. Integrate `main`, explicitly remove or rewrite that
   document, then open and merge an `advanced` to `main` PR. Do not allow the
   stale document to reappear through a blind merge.

### P1

1. Add the ArXiv URL/identifier to README and `CITATION.cff` after submission.
2. Publish any intended checkpoint/sample artifacts and replace or remove the
   obsolete private upload helpers.
3. Tag an immutable release and record the tag/commit in the paper artifact
   statement.

## Stable evidence limits

- Six paper settings, one training run each.
- Best-observed checkpoint selected from sparse T2I-CompBench test evaluations.
- Full-1B: 0.2328 baseline to 0.4887; Full-7B: 0.5326.
- T2I-R1 external reported category mean: 0.5281; no local checkpoint control.
- Upstream weight revisions and historical Attr/ORM/NMS failure rates are
  unavailable and must not be invented.
