# Scripts

Five HICO-DET zero-shot protocols: **RF-UC**, **NF-UC**, **UV**, **UC**, **UO**.

All scripts `cd` to the repository root automatically.

> These scripts are the public training/evaluation configurations.
> Running them end-to-end also needs supporting files that are still being uploaded (see root README).

## Training (`scripts/training/`)

| Setting | Script | `zs_type` | Default output |
|---------|--------|-----------|----------------|
| RF-UC | `training/RF-UC/RF-UC.sh` | `rare_first` | `checkpoints/RF-UC/` |
| NF-UC | `training/NF-UC/NF-UC.sh` | `non_rare_first` | `checkpoints/NF-UC/` |
| UV | `training/UV/UV-B-7.sh` | `unseen_verb` | `checkpoints/UV/` |
| UC | `training/UC/UC.sh` | `uc0` | `checkpoints/UC/` |
| UO | `training/UO/UO.sh` | `unseen_object` | `checkpoints/UO/` |

```bash
bash scripts/training/RF-UC/RF-UC.sh
bash scripts/training/NF-UC/NF-UC.sh
bash scripts/training/UV/UV-B-7.sh
bash scripts/training/UC/UC.sh
bash scripts/training/UO/UO.sh
```

## Evaluation (`scripts/eval/`)

Each protocol has `eval.sh` — config matches the corresponding training script + `--eval --resume`.

```bash
CHECKPOINT_PATH=checkpoints/UV/ckpt_xxx.pt bash scripts/eval/UV/eval.sh
CHECKPOINT_PATH=checkpoints/RF-UC/ckpt_xxx.pt bash scripts/eval/RF-UC/eval.sh
```

## Other

- `download.sh` — dataset download helper
- Ablation / analysis automation — **released in stages**

Environment overrides: `PRETRAINED`, `CLIP_VIT`, `OUTPUT_DIR`, `CHECKPOINT_PATH`, `CUDA_VISIBLE_DEVICES`.
