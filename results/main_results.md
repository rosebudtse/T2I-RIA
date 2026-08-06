# T2I-CompBench results

## Primary local evaluations

| Setting | Backbone | Color | Shape | Texture | Spatial | Non-spatial | Complex | Average |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Baseline | Janus-Pro-1B | 0.3514 | 0.2028 | 0.2740 | 0.0738 | 0.2633 | 0.2314 | 0.2328 |
| T2I-RIA Full | Janus-Pro-1B | 0.7985 | 0.4772 | 0.6831 | 0.2971 | 0.3046 | 0.3716 | 0.4887 |
| T2I-RIA Full | Janus-Pro-7B | 0.8267 | 0.5805 | 0.7317 | 0.3435 | 0.3094 | 0.4040 | 0.5326 |

The selected 1B checkpoint improves the local average by 0.2559, or 109.9%
relative to the matched baseline. The 7B row is a scale-extension result, not
the paper's primary study.

## External reported references

| Setting | Backbone | Color | Shape | Texture | Spatial | Non-spatial | Complex | Average |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Baseline (reported) | Janus-Pro-7B | 0.6359 | 0.3528 | 0.4936 | 0.2061 | 0.3085 | 0.3559 | 0.3921 |
| T2I-R1 (reported) | Janus-Pro-7B | 0.8130 | 0.5852 | 0.7243 | 0.3378 | 0.3090 | 0.3993 | 0.5281 |

The T2I-R1 average is the arithmetic mean of its six reported category
scores. These rows were not regenerated locally, so this table deliberately
does not rank them together with the local evaluations.

## Evidence limits

- Each setting is a single training run.
- Main results are the best observed among a sparse, preselected set of saved
  checkpoints evaluated on the T2I-CompBench test prompts.
- Each local row uses 300 prompts per category, ten images per prompt, and a
  fixed seed stream (18,000 images per checkpoint).
- Selected headline results and matched reward-removal results are retained in
  `main_results.csv`. Full sparse-checkpoint scores are reported in the T2I-RIA
  technical report, while audited run identities, evaluated checkpoints, and
  checkpoint-selection mappings are recorded in
  `reproducibility/runtime_configs/paper_runs.json`.
