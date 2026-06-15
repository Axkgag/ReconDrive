#!/bin/bash
# Usage: bash scripts/train_voxel_vae.sh [num_gpus] [config_path] [work_dir] [tensorboard_dir] [load_vae_from] [extra args...]
# Example: bash scripts/train_voxel_vae.sh 8 configs/nuscenes/voxel_vae.yaml ./work_dirs/voxel_vae ./work_dirs/voxel_vae/tb

set -euo pipefail

NUM_GPUS="${1:-1}"
CONFIG_PATH="${2:-./configs/nuscenes/voxel_vae.yaml}"
WORK_DIR="${3:-}"
TENSORBOARD_DIR="${4:-}"
LOAD_VAE_FROM="${5:-}"

if [[ $# -gt 5 ]]; then
    shift 5
    EXTRA_ARGS=("$@")
else
    EXTRA_ARGS=()
fi

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((NUM_GPUS - 1)))
    export CUDA_VISIBLE_DEVICES
fi

SCRIPT_ARGS=(--cfg_path "${CONFIG_PATH}")

if [[ -n "${WORK_DIR}" ]]; then
    SCRIPT_ARGS+=(--work_dir "${WORK_DIR}")
fi

if [[ -n "${TENSORBOARD_DIR}" ]]; then
    SCRIPT_ARGS+=(--tb_dir "${TENSORBOARD_DIR}")
fi

if [[ -n "${LOAD_VAE_FROM}" ]]; then
    SCRIPT_ARGS+=(--load_vae_from "${LOAD_VAE_FROM}")
fi

if [[ -n "${PRETRAINED_CHECKPOINT_PATH:-}" ]]; then
    echo "Loading Stage1 checkpoint from PRETRAINED_CHECKPOINT_PATH=${PRETRAINED_CHECKPOINT_PATH}"
    SCRIPT_ARGS+=(--pretrained_ckpt "${PRETRAINED_CHECKPOINT_PATH}")
fi

SCRIPT_ARGS+=("${EXTRA_ARGS[@]}")

# Warm up gsplat CUDA JIT cache with a single process before launching DDP.
# Without this, all ranks may try to compile into the same torch_extensions build
# directory concurrently and corrupt each other's build dirs.
echo "=== Warming up gsplat CUDA JIT cache (single process) ==="
FIRST_DEVICE="$(echo "${CUDA_VISIBLE_DEVICES}" | cut -d, -f1)"
CUDA_VISIBLE_DEVICES="${FIRST_DEVICE}" python - <<'PY'
import torch

try:
    from gsplat import rasterization
except Exception:
    from gsplat.rendering import rasterization

N = 100
means = torch.randn(N, 3, device="cuda")
quats = torch.randn(N, 4, device="cuda")
scales = torch.rand(N, 3, device="cuda") * 0.1
opacities = torch.rand(N, device="cuda")
colors = torch.rand(N, 3, device="cuda")
viewmats = torch.eye(4, device="cuda")[None]
Ks = torch.tensor([[[300.0, 0.0, 128.0], [0.0, 300.0, 128.0], [0.0, 0.0, 1.0]]], device="cuda")
rasterization(means, quats, scales, opacities, colors, viewmats, Ks, 256, 256)
torch.cuda.synchronize()
print("gsplat JIT cache ready")
PY

echo "=== Training voxel VAE ==="
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "NUM_GPUS=${NUM_GPUS}"
echo "CONFIG_PATH=${CONFIG_PATH}"

if [[ "${NUM_GPUS}" -gt 1 ]]; then
    torchrun \
        --standalone \
        --nproc_per_node="${NUM_GPUS}" \
        scripts/train_voxel_vae.py \
        "${SCRIPT_ARGS[@]}"
else
    python scripts/train_voxel_vae.py "${SCRIPT_ARGS[@]}"
fi
