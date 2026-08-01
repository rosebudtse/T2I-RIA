# AGENTS.md — CompGen-GRPO 仓库指南

本文件是给 agent（Claude / GPT / 等）的**稳定**仓库指南。它只描述不会随实验进度变化的事实：
仓库结构、训练入口、关键路径、已知坑+对策、环境约定、git 规则。

**当前 run 状态、wandb 链接、最新 TODO 进度**写在 `MEMORY.md`，不要写在这里。

> 📌 **新 agent 接力流程**：① 读本文件 → ② 读 `MEMORY.md`（必读） → ③ 看 `git status` →
> ④ 看 `nvidia-smi` 和 `pgrep -af torchrun` 判断有没有正在跑的训练 → ⑤ 才开始动手。
> 会话结束前**必须**更新 `MEMORY.md`（见末尾"维护协议"）。

---

## 1. 项目一句话

T2I-R1（Janus-Pro-1B + Bi-CoT-GRPO）的扩展，在多维 reward（HPSv2.1 美学 + GroundingDINO
空间/计数 + Qwen3-VL-2B 属性/整体）下做组合式 T2I 生成的强化微调。论文规划见
`TODO_cn.md`，扩展方向见 `docs/structcomp_grpo_todo.md`。

## 2. 仓库结构（关键路径）

```
CompGen-GRPO/
├── AGENTS.md                                  ← 本文件
├── MEMORY.md                                  ← 当前状态快照（agent 必读必写）
├── REPRODUCE.md                               ← 人类可读的复现步骤
├── REWARD_WEIGHTS.md                          ← reward 权重下载指引
├── README.md / TODO_cn.md / TODO_en.md
├── setup_env.sh / check_train_env.sh
├── download_weights.sh                        ← reward 权重一键下载
├── requirements.txt
├── data/
│   ├── geneval_and_t2i_data_final.json        ← 训练数据（7223 prompts）
│   └── prompt/reasoning_prompt.txt
├── src/t2i-r1/
│   ├── reward_weight/                         ← gitignore，需 download_weights.sh 拉
│   │   ├── Janus-Pro-1B/
│   │   ├── HPSv2.1/HPS_v2.1_compressed.pt
│   │   ├── groundingdino_swint_ogc.pth
│   │   ├── bert-base-uncased/                 ← GroundingDINO text encoder
│   │   └── Qwen3-VL-2B-Instruct/
│   └── src/
│       ├── run_train.sh                       ← 训练入口
│       ├── open_r1/grpo.py                    ← argparse + main
│       ├── open_r1/trainer/grpo_trainer.py    ← JanusT2IR1Trainer（核心）
│       ├── janus/models/modeling_vlm.py       ← Janus 模型（已 dataclass 化）
│       ├── utils/
│       │   ├── reward_hps.py                  ← HPS v2.1 reward
│       │   ├── reward_gdino_enhanced.py       ← GroundingDINO reward（增强版）
│       │   ├── reward_vlm.py                  ← VLMAttr + VLMOrm
│       │   └── GroundingDINO/                 ← C++ extension（已编译 _C.so）
│       ├── configs/zero2.json                 ← DeepSpeed ZeRO-2 配置
│       └── outputs/train_main/                ← 训练产物（gitignore）
│           ├── train_log.txt                  ← tee 的内层日志（有 Python traceback）
│           ├── train_main.log                 ← 外层 nohup 包装（看 torchrun warning）
│           ├── reward_log.txt                 ← DEBUG_MODE 下每步 reward 详情
│           └── runs/<timestamp>/              ← tensorboard tfevents
└── archive/                                   ← 旧/废弃代码，勿动
    └── third_party_extras/GroundingDINO/      ← 老副本，已与 utils/GroundingDINO 同步
```

## 3. 训练入口与启动

**唯一入口**：`src/t2i-r1/src/run_train.sh`

启动命令（**一行**，不要带 `\` 换行 —— 上次因换行被吃掉变成多次启动了 4 份 torchrun）：

```bash
cd /mlx_devbox/users/xiezifan/playground/CompGen-GRPO
NPROC=4 MAX_STEPS=1600 CUDA_VISIBLE_DEVICES=0,1,2,3 nohup bash src/t2i-r1/src/run_train.sh > src/t2i-r1/src/outputs/train_main/train_main.log 2>&1 &
```

启动后 sanity check：

```bash
jobs                                                # 必须只有 1 个 [1] running
pgrep -af "torchrun|open_r1/grpo"                   # 应该只有 1 个 torchrun + N 个子进程
nvidia-smi --query-gpu=index,memory.used --format=csv
tail -f src/t2i-r1/src/outputs/train_main/train_log.txt
```

看到 `Dataset length: 7223` → 加载模型 → `{'loss': ..., 'kl': ..., ...}` 就稳了。

**停止**：

```bash
pkill -f torchrun
pkill -f open_r1/grpo
sleep 2 && pgrep -af "torchrun|open_r1/grpo"        # 应该没输出
```

## 4. 关键超参与含义

`run_train.sh` 里**会经常调**的几个：

| 变量 / 参数 | 当前值 | 含义 |
|---|---|---|
| `NPROC` (env) | 4 | GPU 数 |
| `MAX_STEPS` (env) | 1600 | optimizer step 上限 |
| `--num_generations` | 8 | GRPO 每个 prompt 采几路（**影响 advantage 估计质量**） |
| `--per_device_train_batch_size` | 1 | 单卡每步 prompt 数 |
| `--gradient_accumulation_steps` | 2 | accum 后再 step |
| `--beta` | 0.01 | KL 正则系数（0 = T2I-R1 默认，ref_model 不加载） |
| `--learning_rate` | 1e-6 | Adam LR |
| `--attn_implementation` | sdpa | 当前 fallback；flash_attn2 因 ABI 不匹配未启用 |
| `--report_to` | wandb | **必须单值**，trl HfArgumentParser 不接受多值列表 |
| `--save_steps` | 400 | checkpoint 间隔 |
| `--save_total_limit` | 3 | 最多保留 3 份 ckpt |
| `--reward_funcs` | hps gdino vlm_attr vlm_orm | 4 维 reward，加 `vlm_attr`/`vlm_orm` 需要 Qwen3-VL-2B |

**有效 batch size = `NPROC × bs × num_gen × accum = 4×1×8×2 = 64` generations/step**。

**单 step 时间**：4×A100 + sdpa + num_gen=8 下待实测（旧 2×H20 + num_gen=4 配置下约 36s，仅供参考，不适用当前配置）。

## 5. 已知坑 + 对策（**不要踩第二次**）

| 症状 | 原因 | 对策 |
|---|---|---|
| `--report_to wandb tensorboard` → `arguments not used: ['tensorboard']` | trl 1.4 的 HfArgumentParser 把 `Union[None, str, list[str]]` 解析为 `str`，只接受单值 | 只写 `--report_to wandb` |
| `--report_to "wandb,tensorboard"` → `wandb,tensorboard is not supported` | transformers 4.57 不再支持逗号串 | 同上 |
| `from groundingdino import _C: libc10.so cannot open` | dlopen 顺序问题 | `utils/GroundingDINO/groundingdino/__init__.py` 和 `archive/third_party_extras/GroundingDINO/groundingdino/__init__.py` 都加 `import torch` 强制先 dlopen torch lib（已修） |
| `Could not run 'torchvision::nms' with arguments from 'CUDA' backend` | byted-torch 的 torchvision 只编了 CPU NMS kernel | `reward_gdino_enhanced.py` L345 强制 `.cpu()` 跑 NMS（已修） |
| GroundingDINO C++ 编译 `value.type()` 报 `DeprecatedTypeProperties` 不能隐式转 `ScalarType` | torch 2.7 ABI 变化 | `ms_deform_attn.h` / `ms_deform_attn_cuda.cu` 把 `value.type().is_cuda()` → `value.is_cuda()`、`AT_DISPATCH_FLOATING_TYPES(value.type(),...)` → `value.scalar_type()`（已修，两个副本都改） |
| `wandb login: PermissionError: /home/tiger/.netrc` | `.netrc` 不可写 | 用 `WANDB_API_KEY` 环境变量绕过 `.netrc` |
| flash_attn wheel `undefined symbol: ...basic_string...` | byted-torch 编译用 `_GLIBCXX_USE_CXX11_ABI=0`，官方 flash-attn wheel 全是 ABI=1 | 没救，**必须源码编译** `flash-attn==2.7.4.post1`；目前 sdpa fallback |
| 启动一次跑出 4 份 torchrun | 启动命令带反斜杠换行 + 终端 paste 时换行符被吃 | **一行启动**，不要用 `\` |
| `line 81: R/train_log.txt: No such file or directory` | 同上，`$OUTPUT_DIR` 被截断为 `R` | 同上 |

## 6. 环境约定

- **物理机**：byted 平台，master/worker 架构。**训练只能在 worker 跑**（GPU 在 worker）
- **GPU**：4 × NVIDIA A100（80GB 显存/卡）
- **torch**：byted-torch 2.7.1，`_GLIBCXX_USE_CXX11_ABI=0`（用 `python -c "import torch; print(torch._C._GLIBCXX_USE_CXX11_ABI)"` 验证）
- **transformers**：4.57（用户回退过；不要升级，会触发更多 API 变化）
- **trl**：1.4（自带 `_hf_argparser.py`，比 transformers 的 HfArgumentParser 严格）
- **deepspeed**：ZeRO-2，配置 `src/t2i-r1/configs/zero2.json`
- **bert-base-uncased**：`text_encoder_type` 在 `utils/GroundingDINO/.../GroundingDINO_SwinT_OGC.py` **硬编码绝对路径**，换机器需要改
- **PYTHONPATH**：`run_train.sh` 把 `$SCRIPT_DIR`（`src/t2i-r1/src`）加进去

## 7. Reward 函数

4 个 reward 函数在 [grpo_trainer.py L140-155](file:///mlx_devbox/users/xiezifan/playground/CompGen-GRPO/src/t2i-r1/src/open_r1/trainer/grpo_trainer.py#L140-L155) 按字符串映射实例化：

| reward_func 字符串 | 类 | 文件 |
|---|---|---|
| `hps` | `HPSv2` | `utils/reward_hps.py` |
| `gdino` | `GDinoEnhanced` | `utils/reward_gdino_enhanced.py` |
| `vlm_attr` | `VLMAttr` | `utils/reward_vlm.py` |
| `vlm_orm` | `VLMOrm` | `utils/reward_vlm.py` |

`GDinoEnhanced` 相对原版 `GDino` 有 4 处改进，详见文件头注释。**改 reward 加权**：见 `REWARD_WEIGHTS.md`（如果存在）；当前 4 个 reward 是**等权**直接相加（trl GRPO 默认行为）。

## 8. 调试 / 监控

- **wandb**：项目 `CompGen-GRPO`，run name 由 `WANDB_NAME` 控制（见 `run_train.sh` 顶部）
- **tensorboard**：`outputs/train_main/runs/<timestamp>/` 下的 tfevents（被 `--report_to` 切单值后**已禁用**；要 tb 必须切回 `"all"` 或写 yaml config）
- **reward 详情**：`DEBUG_MODE=true` 时（已开启），每步 reward 写 `outputs/train_main/reward_log.txt`
- **训练日志**：`outputs/train_main/train_log.txt`（tee 写的，有 Python traceback）
- **torchrun 日志**：`outputs/train_main/train_main.log`（nohup 包装，看 NCCL/退出信息）

## 9. Git 规则

- 主开发分支：`advanced`
- 主线/对照分支：`main`（原 T2I-R1 复现）
- **不要 commit**：`reward_weight/`、`outputs/`、`runs/`、`*.tfevents.*`、`smoke.log`、`*.pth`、`*.bin`（都在 gitignore）
- 大改 reward / trainer 前先 `git status` 看 working tree 是不是干净
- **绝不**：force push、`git reset --hard` working tree、删除别人没看过的分支
- commit message 风格参考 `git log` 最近几条（短、动词开头、中英不混用）

## 10. 给 agent 的工作守则

- **任何修改 `run_train.sh` 的训练参数都要解释 trade-off**（吞吐 / 显存 / 收敛质量）
- **修 reward 函数要在文件头注释里加一行 changelog**（已有先例）
- **不要 silently 装新 pip 包**，先问用户
- **不要写 PR 描述文 / 设计文档**除非用户明确要求
- **修代码前先看[grpo_trainer.py](file:///mlx_devbox/users/xiezifan/playground/CompGen-GRPO/src/t2i-r1/src/open_r1/trainer/grpo_trainer.py)知道 `beta=0` 走哪条路、`beta≠0` 走哪条路**（L127-134），这俩路径对 ref_model 的加载完全不一样
- **训练正在跑时**：除非用户要求停，否则**不要**杀进程；改 `run_train.sh` 不影响正在跑的 process（脚本已加载完毕），下次启动才生效

---

## 维护协议（agent 必读）

**每次会话结束前必须更新 `MEMORY.md`**，无论该会话有没有跑训练。模板见 `MEMORY.md` 顶部说明。
更新内容：
- 当前 run 状态（运行中/已完成/已停 + step / 时间 / 是否健康）
- 最近一次会话的关键决策与代码改动（一两句话足够）
- 下次接力时的 next steps（按 P0/P1/P2 排）
- 新发现的坑 → 同步到本文件第 5 节"已知坑"

**本文件（AGENTS.md）只在以下情况修改**：
- 仓库结构变了（新增目录/重要文件）
- 启动方式变了
- 发现新的、可复现的坑（加进第 5 节）
- 环境约定变了（torch / transformers 升级等）
