# HSE-CDR

Code repository for **HSE-CDR**, a zero-shot Human-Object Interaction (HOI) detection method on HICO-DET.

Training/evaluation configurations, zero-shot splits, data loaders, inference/evaluation entry points, and core module code are provided here. Additional project materials will continue to be released in stages.

## Method (overview)

The framework integrates three modules (details in the manuscript):

- **TDAM** — Task Disentanglement Anchor Module
- **IMEU** — Interaction Microscopic Enhancement Unit
- **CDHRM** — Cross-Domain Hierarchical Relational Reasoning Module

## What's included here

| Content | Status |
|---------|--------|
| Environment file (`HSE-CDR_environment.yml`) | Available |
| Dataset placeholders & loaders (`hicodet/`, `datasets/`) | Available |
| Zero-shot split definitions (`datasets/zs_*.pkl`) | Available |
| Third-party dependencies (`pocket/`, `detr/`) | Available |
| Train / eval protocol scripts (`scripts/training/`, `scripts/eval/`) | Available |
| Inference / evaluation entry (`main.py`, `engine.py`) | Available |
| Core modules (`models/hse_cdr.py`, `models/transformers.py`) | Available |
| IMEU-related code | **Not uploaded yet** (will be updated later) |
| Ablation / analysis automation | **Released in stages** |
| Pretrained checkpoints | **Upon request** (see below) |

## Setup (environment)

```bash
conda env create -f HSE-CDR_environment.yml
conda activate hse-cdr
cd pocket && pip install -e . && cd ..
```

Place HICO-DET under `hicodet/hico_20160224_det/` (see `hicodet/README.md`).
Pretrained DETR/CLIP weights go under `checkpoints/` (see `checkpoints/README.md`).

## Train & evaluate (protocol scripts)

Protocol launchers are provided under `scripts/` (see [scripts/README.md](scripts/README.md)):

```bash
bash scripts/training/UV/UV-B-7.sh
CHECKPOINT_PATH=checkpoints/UV/ckpt_xxx.pt bash scripts/eval/UV/eval.sh
```

> End-to-end execution also needs supporting files that are still being uploaded in stages.
> A complete package is available to reviewers upon request through the submission system.

## Code availability

- **In stages:** remaining supporting code and ablation/analysis tools will be updated in this repository.
- **IMEU:** related code has not been uploaded yet and will be updated later.
- **Checkpoints:** available from the corresponding author upon reasonable request for academic research.
- **For peer review:** complete implementation is available upon request through the submission system (anonymous package if required).

## Citation

Citation details for **HSE-CDR** will be provided upon acceptance.

## License

This project will be released under the [MIT License](LICENSE). Third-party components (`pocket/`, `detr/`) retain their original licenses.
