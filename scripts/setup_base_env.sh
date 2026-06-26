#!/usr/bin/env bash
set -e

ALIYUN_PYPI="http://mirrors.aliyun.com/pypi/simple"
TRUSTED_HOST="mirrors.aliyun.com"

echo ">>> 配置 pip trusted-host"
pip config set global.trusted-host "${TRUSTED_HOST}"

echo ">>> 安装/升级 openmim"
pip install -U openmim -i "${ALIYUN_PYPI}"

echo ">>> 安装 mmengine==0.8.4"
mim install mmengine==0.8.4 -i "${ALIYUN_PYPI}"

echo ">>> 安装 mmcv==2.1.0"
mim install mmcv==2.1.0 -i "${ALIYUN_PYPI}"

echo ">>> 安装 mmdet==3.3.0"
mim install mmdet==3.3.0 -i "${ALIYUN_PYPI}"

echo ">>> 安装 mmdet3d==1.4.0"
mim install mmdet3d==1.4.0 -i "${ALIYUN_PYPI}"

echo ">>> 更新yapf版本"
pip install yapf==0.40.1 -i "${ALIYUN_PYPI}"

echo ">>> 安装 pyvirtualdisplay==3.0"
pip install pyvirtualdisplay==3.0 -i "${ALIYUN_PYPI}"

echo ">>> 安装 setuptools==59.5.0"
pip install setuptools==59.5.0 -i "${ALIYUN_PYPI}"

echo ">>> 安装 traits==6.4.2"
pip install traits==6.4.2 -i "${ALIYUN_PYPI}"

echo ">>> 安装 vtk==9.2.6"
pip install vtk==9.2.6 -i "${ALIYUN_PYPI}"

echo ">>> 安装 pyqt5==5.15.10"
pip install pyqt5==5.15.10 -i "${ALIYUN_PYPI}"

# echo ">>> 安装 mayavi==4.8.1"
# pip install mayavi==4.8.1 -i "${ALIYUN_PYPI}" --no-build-isolation

echo ">>> 安装 lpips==0.1.4"
pip install lpips==0.1.4 -i "${ALIYUN_PYPI}"

echo ">>> 安装 xvfb"
apt-get update
apt-get install -y xvfb

# echo ">>> 进入 occforecasting 并执行 setup.py develop"
# cd occforecasting
# python setup.py develop
# cd ..

echo ">>> 环境配置完成"
