#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"

port=$((RANDOM % 5000 + 5000))
id=$((RANDOM % 5000 + 5000))
gpu_num=1
export WANDB__SERVICE_WAIT=300
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

OUTPUT_DIR="${OUTPUT_DIR:-checkpoints/UC}"
PRETRAINED="${PRETRAINED:-checkpoints/pretrained_detr/detr-r50-hicodet.pth}"
CLIP_VIT="${CLIP_VIT:-checkpoints/pretrained_clip/ViT-B-16.pt}"

USE_THREE_BRANCH="${USE_THREE_BRANCH:-1}"
USE_ADAPTER="${USE_ADAPTER:-1}"
USE_LEARNABLE_FUSION="${USE_LEARNABLE_FUSION:-1}"

ARGS_FUSION=""
[ "${USE_THREE_BRANCH}" = "1" ] && ARGS_FUSION+=" --use-three-branch-fusion"
[ "${USE_LEARNABLE_FUSION}" = "1" ] && ARGS_FUSION+=" --use-learnable-fusion"

ARGS_ADAPTER=""
if [ "${USE_ADAPTER}" = "1" ]; then
  ARGS_ADAPTER="--use_insadapter --use_prior --adapt_dim 32 --adapter_alpha 1."
fi

torchrun --rdzv_id "$id" --rdzv_backend=c10d --nproc_per_node="$gpu_num" \
         --rdzv_endpoint="127.0.0.1:${port}" \
         main.py \
         --pretrained "$PRETRAINED" \
         --clip_dir_vit "$CLIP_VIT" \
         --output-dir "$OUTPUT_DIR" \
         --dataset hicodet --zs --zs_type uc0 \
         --num_classes 117 --num-workers 4 \
         --epochs 20 --use_hotoken --use_prompt --use_exp --N_CTX 36 --CSC \
         --clip_visual_width_vit 768 \
         --log-epoch-costs \
         $ARGS_ADAPTER $ARGS_FUSION \
         --print-interval 100
