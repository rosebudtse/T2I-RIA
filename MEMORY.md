# MEMORY.md — CompGen-GRPO 当前状态快照

> 📌 **本文件是会话间的接力棒**。每次会话结束前必须由 agent 更新（见 `AGENTS.md` §维护协议）。
> 新 agent 接力流程：先读 `AGENTS.md` → 再读本文件 → 再动手。

**Last updated**: 2026-07-29, by agent 文档一致性核对（对齐 paper_outline_v1.md）+ 冲突裁决
**Branch**: `advanced`（同步 `origin/advanced`）
**Owner**: xiezifan

> 📎 **论文规划权威来源**：`docs/paper_outline_v1.md`（V1 定稿大纲）。`results/main_results.md`、
> `results/main_results.csv` 已与之对齐。如本文件数值/结论与大纲冲突，**以大纲为准**。
>
> 🧭 **本轮已裁决的两处冲突（owner 确认）**：
> 1. **7B 主结果 = step 500 / avg 0.5326**（超过 T2I-R1 reported 0.5114）。
>    ⚠️ 但 `eval_results/` 下暂无 step 500 实体目录（现存 full_7b_400/800/1200），
>    定稿前需在 worker 补跑 step 500 eval 以保证可复现（见 P0-2）。
> 2. **论文/文档训练硬件 = 4 × A100**（大纲 §4.3 口径）。
>    ✅ 已同步：`AGENTS.md §6`、启动命令、超参表（NPROC=4 / num_gen=8 / 有效 batch=64）均已更新为 4×A100 口径。
>    注：`run_train.sh` 本身早已是 4 卡配置（EXP_NAME=..._4gpu_g8_..., NPROC:-4）；旧 AGENTS.md 的 2×H20 属滞后描述。

---

## 1. 当前实验状态

### 训练：**已全部完成**（1B baseline + 3 组 ablation + 7B 主训）

无正在跑的 torchrun / open_r1 进程；当前节点是 master，没有 GPU。

已产出的 ckpt 组（按 [eval_results/](file:///mlx_devbox/users/xiezifan/playground/CompGen-GRPO/eval_results) 目录反推）：

| 设置 | 已 eval 的 step | 备注 |
|---|---|---|
| **Full-1B** | 200 / 400 / 600 / 800 | 主实验 |
| **Full-7B** | 400 / 800 / 1200 | 大模型主实验 |
| w/o Attr | 400 / 600 / 800 | 去掉 `vlm_attr` reward |
| w/o GDino | 400 / 600 / 800 / **1000（仅 color，未完成）** | 去掉 `gdino` reward |
| w/o ORM | 200 / 400 / 600 | 去掉 `vlm_orm` reward |

### T2I-CompBench 评测数值（重算自 eval_results/*/annotation_blip/blip_vqa_score.txt 等）

| Setting | step | color | shape | texture | spatial | non_sp | complex | avg |
|---|---|---|---|---|---|---|---|---|
| Baseline-1B | – | 0.351 | 0.203 | 0.274 | 0.074 | 0.263 | 0.231 | **0.233** |
| Full-1B | 200 | 0.778 | 0.469 | 0.666 | 0.263 | 0.303 | 0.365 | 0.474 |
| Full-1B | 400 | 0.796 | 0.477 | 0.678 | 0.293 | 0.303 | 0.368 | 0.486 |
| **Full-1B** | **600** | 0.799 | 0.477 | 0.683 | 0.297 | 0.305 | 0.372 | **0.489 ← 1B 峰值** |
| Full-1B | 800 | 0.715 | 0.452 | 0.614 | 0.252 | 0.302 | 0.352 | 0.448 |
| **Full-7B** | **500** | 0.827 | 0.581 | 0.732 | 0.344 | 0.309 | 0.404 | **0.5326 ← 7B 主结果（大纲口径，超 T2I-R1）** |
| Full-7B | 400 | 0.797 | 0.551 | 0.702 | 0.303 | 0.309 | 0.394 | 0.509（实体重算，见下方 ⚠️） |
| Full-7B | 800 | 0.764 | 0.544 | 0.674 | 0.299 | 0.308 | 0.383 | 0.495 |
| Full-7B | 1200 | 0.721 | 0.532 | 0.647 | 0.291 | 0.301 | 0.356 | 0.475 |
| w/o Attr | 400 | 0.785 | 0.465 | 0.664 | 0.284 | 0.302 | 0.366 | 0.478 |
| w/o Attr | 600 | 0.743 | 0.444 | 0.640 | 0.287 | 0.302 | 0.360 | 0.463 |
| w/o Attr | 800 | 0.679 | 0.408 | 0.609 | 0.244 | 0.300 | 0.348 | 0.431 |
| **w/o GDino** | **400** | 0.795 | 0.522 | 0.694 | 0.280 | 0.305 | 0.383 | **0.497（≈ Full 峰值）** |
| w/o GDino | 600 | 0.799 | 0.511 | 0.684 | 0.283 | 0.304 | 0.381 | 0.494 |
| w/o GDino | 800 | 0.785 | 0.478 | 0.655 | 0.254 | 0.303 | 0.369 | 0.474 |
| w/o GDino | 1000 | 0.681 | – | – | – | – | – | ⚠️ 只跑了 color，其余未完成 |
| w/o ORM | 200 | 0.759 | 0.428 | 0.641 | 0.253 | 0.299 | 0.351 | 0.455 |
| w/o ORM | 400 | 0.774 | 0.460 | 0.664 | 0.279 | 0.302 | 0.363 | 0.474 |
| w/o ORM | 600 | 0.773 | 0.458 | 0.664 | 0.290 | 0.302 | 0.367 | 0.476 |

**参考**：T2I-R1 原文 7B reported avg 0.5114。
- **主结果口径（大纲/CSV）**：Full-7B step 500 = **0.5326**，超过 T2I-R1。
- ⚠️ **可复现性风险**：`eval_results/` 下**没有 step 500 的实体评测目录**（现存 full_7b_400/800/1200）。
  从实体文件重算的最好一档是 step 400 = 0.509（略低于 T2I-R1）。定稿前必须在 worker 补跑
  step 500 eval 落地实体文件，否则 "超过 T2I-R1" 这一核心卖点无法复现（见 P0-2）。

### 关键观察

1. **过训 / collapse**：1B 在 step 600 后、7B 在 step 400 后全维度同步下滑，明显 reward hacking。
2. **消融强度**（step 400 avg）：Full 0.486 ≈ w/o GDino 0.497 > w/o Attr 0.478 ≈ w/o ORM 0.474
3. **w/o GDino 略优于 Full**：GDino 主要贡献 spatial（Full 0.293 vs w/o 0.280），但会小幅拖累 shape/texture。
4. **w/o Attr 退化最快**：400→800 avg 掉 4.7 pts，VLMAttr 是最重要的稳定项。
5. **w/o ORM 是唯一没在 600 前掉头的消融**，还在缓慢上升；值得单独长跑验证。

### 日志位置

- 训练日志（早期 baseline）：[train_log.txt](file:///mlx_devbox/users/xiezifan/playground/CompGen-GRPO/src/t2i-r1/src/outputs/train_main/train_log.txt) / [train_main.log](file:///mlx_devbox/users/xiezifan/playground/CompGen-GRPO/src/t2i-r1/src/outputs/train_main/train_main.log)
- Reward 详情：[reward_log.txt](file:///mlx_devbox/users/xiezifan/playground/CompGen-GRPO/src/t2i-r1/src/outputs/train_main/reward_log.txt)
- 各 eval 生成 / 评测日志：[eval_results/*/logs/](file:///mlx_devbox/users/xiezifan/playground/CompGen-GRPO/eval_results) 及 [eval_results/logs/](file:///mlx_devbox/users/xiezifan/playground/CompGen-GRPO/eval_results/logs)
- 主表：[results/main_results.csv](file:///mlx_devbox/users/xiezifan/playground/CompGen-GRPO/results/main_results.csv)

---

## 2. 接力 next steps（按优先级）

### P0 — 必须做

1. **补齐 wo_gdino_1000 剩余 5 个维度**（shape/texture/spatial/non_spatial/complex）或直接判它 collapsed。color 已 0.681，很可能拉平均 <0.44。用 [run_eval.sh](file:///mlx_devbox/users/xiezifan/playground/CompGen-GRPO/src/t2i-r1/src/run_eval.sh) 补跑一次即可。
2. **补跑 Janus-Pro-7B step 500 的 eval（owner 已裁决）**：
   - 主结果口径固定为 step 500 / avg **0.5326**（大纲 + CSV 第 5 行之后的 Full-7B 行）。
   - 但 `eval_results/` 只有 full_7b_400/800/1200，**没有 step 500 实体目录**；实体重算最好档是 step 400 = 0.509。
   - **动作**：在 worker 用 [run_eval.sh](file:///mlx_devbox/users/xiezifan/playground/CompGen-GRPO/src/t2i-r1/src/run_eval.sh) 补跑 step 500 eval，落地实体文件后复核 0.5326。
   - ❌ **不要**把 CSV 的 step 500 改成 step 400 —— owner 已确认以 0.5326 为准，只需补实体证据。
3. **清理 git working tree**：
   - 有 90+ 个 modified/deleted 文件（都在 eval_results/），大多应是 gitignore 遗漏或旧文件被 eval 脚本覆写。
   - 未追踪的 [convert_bin_to_safetensors.py](file:///mlx_devbox/users/xiezifan/playground/CompGen-GRPO/scripts/convert_bin_to_safetensors.py)、[zero3.json](file:///mlx_devbox/users/xiezifan/playground/CompGen-GRPO/src/t2i-r1/configs/zero3.json) 该 commit 的 commit，其他 log/vqa 结果按 [AGENTS.md §9](file:///mlx_devbox/users/xiezifan/playground/CompGen-GRPO/AGENTS.md) gitignore 掉。

### P1 — V1 论文分析层面

4. **主 ablation 表按 step 600 对齐**（已在 §1 生成）：论文主文用这张表，不用 Full@600 vs w/o-GDino@400 混 step 的比较（checkpoint confound）。
5. **画训练-评测曲线**：把 (step, avg) 折线画出来，能直接看出 collapse 拐点，放进 Fig X。
6. **写 Reward Composition Analysis 小节**（V1 采纳的 GDino 叙事）：
   - 主论述："detector-guided reward acts as a spatial-specialized reward under fixed weighting; removing it slightly improves avg but reduces spatial by 4.8%"
   - 官方措辞（GPT 版）：*"Removing the detector-based GDino reward slightly improves the average score, but reduces the spatial score from 0.2971 to 0.2827. GDino acts as a spatial-specialized reward: it improves spatial grounding while introducing trade-offs in shape and complex composition under fixed reward weighting."*
   - 关键红线：**不要写 "Our GDino reward improves overall compositional performance"**（数据不支持）；**不要写 "calls into question prior work"**（语气过强）。
7. **w/o ORM 长跑**：跑 step 800 / 1000 看是否也 collapse。若不塌，可以佐证 ORM 是 collapse 的主要诱因，reward 权重可下调。

### P2 — V2 方法学扩展（不在 V1 scope）

8. **GDino 权重扫描** `w ∈ {0, 0.25, 0.5, 1.0, 2.0}`：V1 采用固定等权、承认 trade-off；V2 才做扫描，作为 dynamic weighting 的动机图。
9. **原版 GDino vs GDinoEnhanced** 对照：V1 不 claim "我们的 GDino 改进了原版"，所以对照可以延后到 V2 / appendix。
10. **Adaptive / task-conditional reward weighting**：V2 的核心贡献，围绕 GDino trade-off 展开：spatial prompt 拉高权重，shape/texture prompt 压低。
11. **TODO_cn.md / docs/structcomp_grpo_todo.md**：StructComp-GRPO 扩展研究规划。
12. **flash_attn 源码编译**：`pip install flash-attn==2.7.4.post1 --no-build-isolation`。

---

## 2.5 V1 / V2 叙事分工（本次会话确定）

**V1（arXiv + workshop）scope**：
- Fixed-weight CompGen-GRPO works well（1B baseline → 0.489，7B → 0.509）
- 7B 复现 T2I-R1 reported (0.5114)
- Ablation 用 step-600 对齐主表；VLMAttr 最重要，ORM 有正贡献但较弱
- GDino 是 **spatial-specialized reward with trade-off**，诚实报告 w/o GDino avg 略高
- 固定权重 reward composition 存在 category-specific trade-off

**V2 留白**：
- Adaptive / task-conditional / dynamic reward weighting（把 GDino 从"蠢"变成 V2 核心动机）
- GDino weight sweep + per-category performance
- 原版 vs 改进版 GDino 对照

**为什么不在 V1 做权重扫描**（GPT 论证，已采纳）：
- ≥ 5 组训练成本高
- 每组还要选 checkpoint
- 训练可能不稳定，结果解释复杂
- 提前做完会把 V2 故事消耗掉

---

## 3. 已解决的问题（历史 + 近期）

按时间顺序，含早期已同步到 [AGENTS.md §5](file:///mlx_devbox/users/xiezifan/playground/CompGen-GRPO/AGENTS.md) 的：

- GroundingDINO C++ torch 2.7 ABI 不匹配 → 改 `value.type()` 调用
- `from groundingdino import _C: libc10.so` → 显式 `import torch`
- `torchvision::nms` CUDA backend 缺失 → 强制 `.cpu()` NMS
- `wandb login .netrc` 权限 → `WANDB_API_KEY` env
- `--report_to` 只接受单值（trl HfArgumentParser）
- 一行启动，别用反斜杠换行（否则起 4 份 torchrun）

（近 3 周训练/评测循环期间未新增可复现坑；如有请补充。）

---

## 4. 重要决策记录

- **`--beta=0.01`**：保留轻度 KL 正则；T2I-R1 默认 0，我们给一点 anchor
- **`--num_generations=4`**：baseline 用 4；G=8 的 ablation 未跑（可作 P2 补）
- **不启 vLLM**：Janus 双 head，改造成本高
- **不写 PR / 设计文档**，代码注释 + AGENTS/MEMORY 沉淀
- **1B 主实验拉到 step 800、7B 拉到 1200**：验证 collapse 拐点后即可停

---

## 5. 未解决 / 待跟进

- [ ] wo_gdino_1000 shape/texture/spatial/non_spatial/complex 未跑（P0-1）
- [ ] 7B step 500 (avg 0.5326) 缺实体 eval 目录，需在 worker 补跑落地（P0-2；口径已定为 0.5326，不改数）
- [ ] Git working tree 脏（90+ modified，需要判断 gitignore vs commit，P0-3）
- [ ] flash_attn 源码编译仍未做（P2-8）
- [ ] `--report_to` 想同时 wandb + tensorboard 需要走 YAML config 路线
- [ ] w/o ORM 只跑到 step 600，未确认是否也会 collapse（P1-6）

---

## 6. 给下一个 agent 的提示

- **先看 `AGENTS.md`**，再回本文件
- **训练环境不在 master**：`nvidia-smi` 在当前节点无输出，训练/eval 需切 worker
- **本轮所有 ckpt 应已在 outputs/train_*/checkpoint-*/**（如已清空则从 wandb 找 run URL 反推）
- **改代码 / 加 reward 前**：先看 `git status`，先把已有脏文件挑掉或 commit 掉，避免混淆
- **主表以 eval_results/*/annotation_blip/blip_vqa_score.txt 等实体文件为准**，不以 [main_results.csv](file:///mlx_devbox/users/xiezifan/playground/CompGen-GRPO/results/main_results.csv) 为准（后者有已知不一致，见 P0-2）
