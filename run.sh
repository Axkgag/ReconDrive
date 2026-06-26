#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash run.sh [num_gpus] [config_path] [work_dir] [tensorboard_dir] [load_vae_from] [extra train_voxel_vae args...]
# Example:
#   bash run.sh 8 configs/nuscenes/voxel_vae.yaml work_dirs/voxel_vae /summary

REPO_DIR="/data_map/guoxiyue/ReconDrive"
cd "${REPO_DIR}"

bash scripts/install_deps.sh

NUM_GPUS="${1:-8}"
CONFIG_PATH="${2:-configs/nuscenes/voxel_vae.yaml}"
WORK_DIR="${3:-work_dirs/voxel_vae}"
TENSORBOARD_DIR="${4:-/summary}"
LOAD_VAE_FROM="${5:-}"

if [[ $# -gt 5 ]]; then
    shift 5
    EXTRA_ARGS=("$@")
else
    EXTRA_ARGS=()
fi

bash scripts/train_voxel_vae.sh \
    "${NUM_GPUS}" \
    "${CONFIG_PATH}" \
    "${WORK_DIR}" \
    "${TENSORBOARD_DIR}" \
    "${LOAD_VAE_FROM}" \
    "${EXTRA_ARGS[@]}"
