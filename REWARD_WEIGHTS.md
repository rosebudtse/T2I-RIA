# Model and reward weights

Model weights are not committed to this repository. `download_weights.sh`
places them under `src/t2i-r1/reward_weight/`.

## Expected layout

```text
src/t2i-r1/reward_weight/
├── Janus-Pro-1B/
├── Janus-Pro-7B/                     # optional
├── Qwen3-VL-2B-Instruct/
├── HPSv2.1/
│   └── HPS_v2.1_compressed.pt
├── bert-base-uncased/
└── groundingdino_swint_ogc.pth
```

## Automated download

Install `huggingface_hub` through `requirements.txt`, then run:

```bash
bash download_weights.sh
```

To include the 7B backbone:

```bash
DOWNLOAD_7B=1 bash download_weights.sh
```

The script uses `hf download` when available and falls back to the legacy
`huggingface-cli download` command. It does not install packages or store access
tokens.

## Sources

| Artifact | Source |
|---|---|
| Janus-Pro-1B | `deepseek-ai/Janus-Pro-1B` |
| Janus-Pro-7B | `deepseek-ai/Janus-Pro-7B` |
| Qwen3-VL-2B-Instruct | `Qwen/Qwen3-VL-2B-Instruct` |
| HPSv2.1 | `xswu/HPSv2`, file `HPS_v2.1_compressed.pt` |
| bert-base-uncased | `google-bert/bert-base-uncased` |
| Grounding DINO | official v0.1.0-alpha GitHub release |

The historical experiments downloaded upstream weights without explicit
immutable revision pins. The release therefore records repository identifiers
but does not invent unavailable snapshot hashes.

## Grounding DINO text encoder

`run_train.sh` sets `GDINO_TEXT_ENCODER` to the local
`bert-base-uncased/` directory. The vendored Grounding DINO config otherwise
falls back to the public `google-bert/bert-base-uncased` repository identifier.
This replaces the historical machine-specific absolute path without changing
the selected encoder.

## Verification

```bash
bash check_train_env.sh
```

The check verifies the expected directories and files, imports the main Python
dependencies, and reports whether the Grounding DINO extension is built.
