# Gaussian Autoencoder 训练指南

## 功能概述

训练 Gaussian Autoencoder (AE) 时，支持以下功能：

1. **Backbone 冻结控制**：可配置是否冻结 ReconDrive stage1 模型
2. **分离权重保存**：AE 和 backbone 的权重分别保存
3. **自动恢复训练**：中断后自动从最新 checkpoint 恢复
4. **每 epoch 保存**：所有 epoch 的权重都会保存

---

## 配置选项

### `configs/nuscenes/recondrive_ae.yaml`

```yaml
model_cfg:
  ae_cfg:
    # Backbone 冻结控制
    freeze_backbone: true  # true: 只训练 AE，backbone 冻结
                          # false: 同时训练 AE 和 backbone

    # 其他 AE 配置...
    voxel_size: 0.4
    K: 4
    lambda_render: 1.0
```

---

## Backbone 冻结说明

### `freeze_backbone: true` (默认，推荐)

**行为**：
- ✅ ReconDrive stage1 模型（VGGT backbone）完全冻结
- ✅ 只训练 Gaussian Autoencoder
- ✅ 两个模型完全解耦
- ✅ 训练速度快，显存占用少

**适用场景**：
- 首次训练 AE
- 只想学习 Gaussian 的压缩表示
- 保持 stage1 模型不变

**权重保存**：
- 每个 epoch 保存 AE 权重到 `separate_ckpts/ae_epoch_XX.ckpt`
- Backbone 权重不保存（因为没有更新）

### `freeze_backbone: false`

**行为**：
- ⚠️ ReconDrive stage1 模型和 AE 同时训练
- ⚠️ 两个模型耦合在一起
- ⚠️ 训练速度慢，显存占用大

**适用场景**：
- 想同时微调 backbone 和 AE
- 端到端联合优化

**权重保存**：
- 每个 epoch 保存 AE 权重到 `separate_ckpts/ae_epoch_XX.ckpt`
- 每个 epoch 保存 backbone 权重到 `separate_ckpts/backbone_epoch_XX.ckpt`

---

## 训练命令

### 基础训练（推荐）

```bash
# 使用默认配置（freeze_backbone: true）
bash scripts/train.sh 1 configs/nuscenes/recondrive_ae.yaml ./work_dirs/ae_exp1
```

### 同时训练 backbone 和 AE

1. 修改配置文件：
```yaml
# configs/nuscenes/recondrive_ae.yaml
model_cfg:
  ae_cfg:
    freeze_backbone: false  # 改为 false
```

2. 运行训练：
```bash
bash scripts/train.sh 1 configs/nuscenes/recondrive_ae.yaml ./work_dirs/ae_exp1_joint
```

---

## 权重保存结构

训练后的目录结构：

```
work_dirs/ae_exp1/
├── ckpt/                          # PyTorch Lightning 标准 checkpoint
│   ├── last.ckpt                  # 最新的完整 checkpoint（包含 AE + backbone + optimizer）
│   ├── best_module.ckpt           # 验证集 PSNR 最高的 checkpoint
│   ├── epoch_00.ckpt              # Epoch 0 的完整 checkpoint
│   ├── epoch_01.ckpt              # Epoch 1 的完整 checkpoint
│   └── ...
├── separate_ckpts/                # 分离的权重文件（推荐用于部署）
│   ├── ae_epoch_00.ckpt           # Epoch 0 的 AE 权重（仅 AE）
│   ├── ae_epoch_01.ckpt           # Epoch 1 的 AE 权重（仅 AE）
│   ├── backbone_epoch_00.ckpt     # Epoch 0 的 backbone 权重（仅当 freeze_backbone=false）
│   └── ...
├── log/
│   ├── logs/                      # TensorBoard 日志
│   └── metrics.json               # 训练指标
└── cfg.yaml                       # 配置文件备份
```

---

## 权重加载

### 加载完整 checkpoint（恢复训练）

```python
# 使用 PyTorch Lightning 的标准方式
model = ReconDriveAE_LITModelModule.load_from_checkpoint(
    'work_dirs/ae_exp1/ckpt/last.ckpt'
)
```

### 只加载 AE 权重（推理/部署）

```python
from models.recondrive_ae_model import ReconDriveAE_LITModelModule

# 创建模型
model = ReconDriveAE_LITModelModule(cfg, save_dir='.')

# 只加载 AE 权重
model.load_ae_checkpoint('work_dirs/ae_exp1/separate_ckpts/ae_epoch_09.ckpt')
```

### 只加载 backbone 权重

```python
# 只加载 backbone 权重（如果训练时 freeze_backbone=false）
model.load_backbone_checkpoint('work_dirs/ae_exp1/separate_ckpts/backbone_epoch_09.ckpt')
```

---

## 训练监控

### 查看训练日志

```bash
# 实时查看训练输出
tail -f work_dirs/ae_exp1/log/logs/version_0/events.out.tfevents.*

# 或使用 TensorBoard
tensorboard --logdir work_dirs/ae_exp1/log/logs
```

### 关键指标

训练时会输出以下指标：

**属性重建损失**：
- `train/ae_attr`: 总属性重建损失
- `train/ae_xyz`: xyz 偏移损失
- `train/ae_rot`: 旋转损失
- `train/ae_scale`: 尺度损失
- `train/ae_opacity`: 不透明度损失
- `train/ae_sh`: 球谐系数损失

**渲染损失**：
- `train/ae_render`: 渲染图像损失
- `train/psnr`: 峰值信噪比
- `train/ssim`: 结构相似性
- `train/lpips`: 感知损失

---

## 常见问题

### Q1: 训练时 backbone 会更新吗？

**A**: 取决于 `freeze_backbone` 配置：
- `freeze_backbone: true` (默认)：backbone **不会**更新，只训练 AE
- `freeze_backbone: false`：backbone **会**更新，同时训练 AE 和 backbone

训练开始时会打印：
```
✓ Backbone (ReconDrive stage1) is FROZEN - only AE will be trained
  Frozen 983123456 backbone parameters
```

### Q2: 如何验证 backbone 是否真的被冻结？

**A**: 查看训练日志开头的输出：
```python
# 如果看到这个，说明 backbone 被冻结了
✓ Backbone (ReconDrive stage1) is FROZEN - only AE will be trained
  Frozen 983123456 backbone parameters

# 如果看到这个，说明 backbone 没有被冻结
⚠ Backbone (ReconDrive stage1) is TRAINABLE - both backbone and AE will be trained
```

### Q3: 分离的权重文件有什么用？

**A**: 
- **部署时更灵活**：可以只加载 AE 权重，不需要加载整个 LightningModule
- **模型解耦**：AE 和 backbone 完全独立，可以单独使用
- **文件更小**：AE 权重文件 ~30MB，完整 checkpoint ~4GB

### Q4: 如何只使用 AE 进行推理？

**A**: 
```python
from models.gaussian_autoencoder import GaussianAutoencoder
import torch

# 1. 创建 AE 模型
ae_cfg = {
    'voxel_size': 0.4,
    'x_range': [-40, 40],
    'y_range': [-40, 40],
    'z_range': [-1, 5.4],
    'K': 4,
}
ae = GaussianAutoencoder(ae_cfg)

# 2. 加载权重
checkpoint = torch.load('work_dirs/ae_exp1/separate_ckpts/ae_epoch_09.ckpt')
ae.load_state_dict(checkpoint['ae_state_dict'])

# 3. 推理
ae.eval()
with torch.no_grad():
    reconstructed = ae(gaussian_data)
```

---

## 推荐工作流

### 1. 首次训练 AE（backbone 冻结）

```bash
# 使用默认配置
bash scripts/train.sh 1 configs/nuscenes/recondrive_ae.yaml ./work_dirs/ae_exp1
```

### 2. 中断后恢复训练

```bash
# 运行相同命令，自动检测并恢复
bash scripts/train.sh 1 configs/nuscenes/recondrive_ae.yaml ./work_dirs/ae_exp1
```

### 3. 使用训练好的 AE

```python
# 方法 1: 加载完整 checkpoint
model = ReconDriveAE_LITModelModule.load_from_checkpoint(
    'work_dirs/ae_exp1/ckpt/best_module.ckpt'
)

# 方法 2: 只加载 AE 权重（推荐）
model = ReconDriveAE_LITModelModule(cfg, save_dir='.')
model.load_ae_checkpoint('work_dirs/ae_exp1/separate_ckpts/ae_epoch_09.ckpt')
```

---

## 性能对比

| 配置 | 训练速度 | 显存占用 | 模型解耦 | 推荐场景 |
|---|---|---|---|---|
| `freeze_backbone: true` | 快 | 低 | ✅ 完全解耦 | 首次训练 AE |
| `freeze_backbone: false` | 慢 | 高 | ❌ 耦合 | 端到端微调 |

---

## 总结

- **默认配置**（`freeze_backbone: true`）：backbone 冻结，只训练 AE，两个模型完全解耦
- **权重分离保存**：每个 epoch 自动保存 AE 权重到 `separate_ckpts/`
- **灵活部署**：可以只加载 AE 权重，不需要完整的 LightningModule
- **推荐用法**：使用默认配置训练，使用分离的 AE 权重进行部署
