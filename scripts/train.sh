#!/bin/bash
# Usage: bash scripts/train.sh [num_gpus] [config_path] [work_dir] [tensorboard_dir]
# Example: bash scripts/train.sh 1 configs/nuscenes/recondrive_ae.yaml ./work_dirs/ae_exp1 ./tensorboard_logs

CONFIG_PATH="${2:-./configs/nuscenes/recondrive.yaml}"
PRETRAINED_CHECKPOINT_PATH='./checkpoints/recondrive_stage1.ckpt'
WORK_DIR="${3:-}"
TENSORBOARD_DIR="${4:-}"
EXTRA_ARGS=""
WILL_RESUME=false

# Auto-detect AE mode
if [[ "${USE_AE:-0}" == "1" ]] || [[ "$CONFIG_PATH" == *"recondrive_ae.yaml" ]]; then
    EXTRA_ARGS="--use_ae"
fi

# Add work_dir if specified
if [[ -n "$WORK_DIR" ]]; then
    EXTRA_ARGS="$EXTRA_ARGS --work_dir=$WORK_DIR"
fi

# Add tensorboard_dir if specified
if [[ -n "$TENSORBOARD_DIR" ]]; then
    EXTRA_ARGS="$EXTRA_ARGS --tensorboard_dir=$TENSORBOARD_DIR"
fi

# Auto-resume if work_dir exists and has checkpoints
if [[ -n "$WORK_DIR" ]] && [[ -d "$WORK_DIR/ckpt" ]]; then
    if ls "$WORK_DIR/ckpt"/*.ckpt 1> /dev/null 2>&1; then
        echo "Found existing checkpoints in $WORK_DIR/ckpt, enabling auto-resume"
        EXTRA_ARGS="$EXTRA_ARGS --resume"
        WILL_RESUME=true
    fi
fi

# Only add pretrained_ckpt if NOT resuming
# When resuming, the checkpoint already contains all weights
if [[ "$WILL_RESUME" == false ]]; then
    EXTRA_ARGS="$EXTRA_ARGS --pretrained_ckpt=${PRETRAINED_CHECKPOINT_PATH}"
fi

python -m scripts.trainer \
    --cfg_path=${CONFIG_PATH} \
    --train_4d \
    --devices="${1:-1}" \
    ${EXTRA_ARGS}
