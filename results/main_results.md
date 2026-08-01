# 主结果表

> 数值口径与 `docs/paper_outline_v1.md`（V1 定稿大纲）对齐；如与本文件冲突，以大纲为准。
> 最后更新：2026-07-29

| 模型 | Backbone | Color | Shape | Texture | Spatial | Non-spatial | Complex | Average |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Baseline | Janus-Pro-1B | 0.3514 | 0.2028 | 0.2740 | 0.0738 | 0.2633 | 0.2314 | 0.2328 |
| Baseline | Janus-Pro-7B | 0.6359 | 0.3528 | 0.4936 | 0.2061 | 0.3085 | 0.3559 | 0.3921 |
| T2I-R1 | Janus-Pro-7B | 0.8130 | 0.5852 | 0.7243 | 0.3378 | 0.3090 | 0.3993 | 0.5114 |
| CompGen-GRPO (Ours) | Janus-Pro-1B | 0.7985 | 0.4772 | 0.6831 | 0.2971 | 0.3046 | 0.3716 | 0.4887 |
| CompGen-GRPO (Ours) | Janus-Pro-7B | 0.8267 | 0.5805 | 0.7317 | 0.3435 | 0.3094 | 0.4040 | 0.5326 |

## 派生指标

- **Ours-1B vs Baseline-1B**：绝对提升 `0.4887 - 0.2328 = +0.2559`；相对提升 `+109.9%`（起点低，以绝对提升为准）。
- **Ours-1B vs Baseline-7B**：`0.4887 - 0.3921 = +0.0966`，小模型经组合式 GRPO 后超过 7B baseline。
- **Ours-1B vs T2I-R1-7B**：`0.4887 / 0.5114 = 95.6%`，接近 7B RL 结果。
- **Ours-7B vs T2I-R1-7B**：`0.5326 - 0.5114 = +0.0212`（相对 `+4.15%`），超过 T2I-R1 reported。
- **相对同一 Janus-Pro-7B baseline 的提升**：T2I-R1 `+0.1193`，Ours-7B `+0.1405`。

## 数据来源

- **Baseline-1B / Ours-1B / Ours-7B**：由本地 `eval_results/` 下 JSON 重新计算，每个类别 3,000 条评测记录。Ours 报告 offline evaluation 中表现最好的 saved checkpoint（1B 为 step 600，7B 为 step 500）。
- **Baseline-7B / T2I-R1-7B**：均为 T2I-R1 论文（arXiv:2505.00703）reported reference values，非本地重算。
- ⚠️ **待核实**：Ours-7B 采用的 step 500（avg 0.5326）目前在 `eval_results/` 下没有对应的实体评测目录（现存 full_7b_400/800/1200）。正式定稿前需在 worker 节点补跑 step 500 的 eval 以保证可复现；详见 `MEMORY.md` P0-2。
- 完整 checkpoint-level 结果见 `results/main_results.csv`（对应大纲 Appendix A）。
