#!/bin/bash
# Install ReconDrive runtime dependencies that are NOT in requirements.txt.
# All packages use the Aliyun PyPI mirror for faster downloads in CN.
#
# Usage:
#   bash scripts/install_deps.sh
#
# Prerequisites:
#   - PyTorch already installed (this project uses 2.6.0 + CUDA 12.6)
#   - CUDA toolkit + nvcc available for source-built extensions

set -e

PIP_MIRROR="-i http://mirrors.aliyun.com/pypi/simple --trusted-host mirrors.aliyun.com"

echo "=== Installing yacs (config management) ==="
pip install yacs ${PIP_MIRROR}

echo "=== Installing lpips (perceptual loss) ==="
pip install lpips ${PIP_MIRROR}

echo "=== Installing kornia (CV ops, SSIMLoss) ==="
pip install kornia ${PIP_MIRROR}

echo "=== Installing e3nn (SH rotation, wigner_D) ==="
pip install e3nn ${PIP_MIRROR}

echo "=== Installing torch-scatter (built from source, ~5-10 min) ==="
pip install torch-scatter ${PIP_MIRROR}

echo "=== Installing pytorch3d (built from source, ~10-30 min) ==="
pip install "git+https://github.com/facebookresearch/pytorch3d.git" ${PIP_MIRROR}

echo "=== Running OccWM setup_env.sh ==="
bash /data_map/liangyihao/OccWM/setup_env.sh

echo ""
echo "=== All dependencies installed successfully ==="
