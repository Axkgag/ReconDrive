#!/bin/bash

CONFIG_PATH='./configs/nuscenes/recondrive.yaml'
CHECKPOINT_PATH='./checkpoints/recondrive_stage1.ckpt'
OUTPUT_DIR='./work_dirs/stage1_vis'
DEVICE='0'

# Optional: limit scenes/samples for a quick sanity check
# MAX_SCENES=2
# MAX_SAMPLES=3

CUDA_VISIBLE_DEVICES=${DEVICE} python -m scripts.inference_stage1 \
    --cfg_path="${CONFIG_PATH}" \
    --ckpt="${CHECKPOINT_PATH}" \
    --output_dir="${OUTPUT_DIR}" \
    --device="${DEVICE}"
    # --max_scenes=${MAX_SCENES} \
    # --max_samples=${MAX_SAMPLES}
