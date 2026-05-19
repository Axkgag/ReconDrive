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

echo "=== Installing rich ==="
pip install rich ${PIP_MIRROR}

echo "=== Installing jaxtyping ==="
pip install jaxtyping ${PIP_MIRROR}

echo "=== Installing nuscenes-devkit ==="
pip install nuscenes-devkit ${PIP_MIRROR}

echo "=== Installing gsplat ==="
pip install gsplat ${PIP_MIRROR}

echo "=== Installing kornia (CV ops, SSIMLoss) ==="
pip install kornia ${PIP_MIRROR}

echo "=== Installing e3nn (SH rotation, wigner_D) ==="
pip install e3nn ${PIP_MIRROR}

echo "=== Installing torch-scatter (built from source, ~5-10 min) ==="
pip install torch-scatter ${PIP_MIRROR}

echo "=== Installing pytorch3d (built from source, ~10-30 min) ==="
pip install "git+https://gitcode.com/gh_mirrors/py/pytorch3d.git" ${PIP_MIRROR}

echo ">>>安装 diff-gaussian-rasterization modified"
pip install "git+https://gh-proxy.com/https://github.com/dcharatan/diff-gaussian-rasterization-modified"

echo "=== Running OccWM setup_env.sh ==="
cd /data_map/liangyihao/OccWM && bash setup_env.sh

mkdir -p /root/.cache/torch/hub/checkpoints
ln -sf /data_map/liangyihao/ReconDrive/checkpoints/vgg16-397923af.pth /root/.cache/torch/hub/checkpoints/vgg16-397923af.pth

echo ""
echo "=== All dependencies installed successfully ==="
