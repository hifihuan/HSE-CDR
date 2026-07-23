#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"

PRETRAINED="${PRETRAINED:-checkpoints/pretrained_detr/detr-r50-hicodet.pth}"
CLIP_VIT="${CLIP_VIT:-checkpoints/pretrained_clip/ViT-B-16.pt}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-checkpoints/UV/ckpt_best.pt}"
OUTPUT_DIR="${OUTPUT_DIR:-checkpoints/UV}"
PORT="${PORT:-11547}"

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

[ -f "$CHECKPOINT_PATH" ] || { echo "Error: checkpoint not found: $CHECKPOINT_PATH"; exit 1; }

python main.py \
  --pretrained "$PRETRAINED" \
  --clip_dir_vit "$CLIP_VIT" \
  --dataset hicodet --num-workers 6 --num_classes 117 \
  --zs --zs_type unseen_verb \
  --output-dir "$OUTPUT_DIR" \
  --use_hotoken --use_prompt --N_CTX 36 --use_exp \
  --use_insadapter --adapt_dim 32 --use_prior \
  --use-three-branch-fusion --use-learnable-fusion \
  --clip_visual_width_vit 768 \
  --hyper_lambda 2.8 \
  --resume "$CHECKPOINT_PATH" \
  --eval --port "$PORT"
