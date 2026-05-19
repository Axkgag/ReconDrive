# 快速优化指南（1小时内完成，3-4× 加速）

## 优化 1：启用多进程数据加载（1.5-2× 加速）

### 修改配置文件
```bash
# 编辑 configs/nuscenes/recondrive_ae.yaml
# 将 num_workers: 0 改为 num_workers: 4
```

```yaml
# configs/nuscenes/recondrive_ae.yaml:91
num_workers: 4  # 从 0 改为 4
```

**原理**: 让 4 个进程并行加载数据，GPU 不再等待数据。

---

## 优化 2：启用混合精度训练（1.5× 加速）

### 修改训练脚本
```python
# scripts/trainer.py:190
trainer = pl.Trainer(
    max_epochs=main_cfg.get('train_epoch', 50),
    accelerator="gpu",
    devices=main_cfg['devices'],
    precision="bf16-mixed",  # ← 添加这一行
    gradient_clip_algorithm="norm",
    # ... 其他参数不变
)
```

**原理**: 使用 BF16 进行前向和部分反向传播，减少内存和计算时间。

**注意**: 如果遇到数值不稳定，可以改用 `"16-mixed"` 或保持 `"32-true"`。

---

## 优化 3：减少训练时的渲染次数（1.3× 加速）

### 方案 A：只渲染部分相机（推荐）

在 `models/recondrive_ae_model.py` 的 `training_step` 中添加：

```python
# models/recondrive_ae_model.py:336 之前添加
def training_step(self, batch_input, batch_idx):
    # ... 前面的代码不变 ...
    
    # 4. 渲染重建的 Gaussians
    batch_render_data = self.get_render_data(batch_input)
    
    # ← 添加这段代码：训练时只渲染前 3 个相机
    if self.training:
        # 只保留前 3 个相机的数据
        for key in batch_render_data:
            if isinstance(batch_render_data[key], list) and len(batch_render_data[key]) == 6:
                batch_render_data[key] = batch_render_data[key][:3]
    
    batch_splating_data = self.render_splating_imgs(batch_recontrast_recon, batch_render_data)
    # ... 后面的代码不变 ...
```

### 方案 B：减少渲染分辨率（更激进）

```python
# models/recondrive_ae_model.py
# 在 training_step 中，渲染前降低分辨率
if self.training:
    # 训练时使用 1/2 分辨率渲染
    original_height = batch_render_data['height']
    original_width = batch_render_data['width']
    batch_render_data['height'] = original_height // 2
    batch_render_data['width'] = original_width // 2
```

---

## 优化 4：减少梯度累积步数（感觉快 2×）

### 修改训练脚本
```python
# scripts/trainer.py:192
trainer = pl.Trainer(
    # ...
    accumulate_grad_batches=4,  # 从 8 改为 4
    # ...
)
```

**注意**: 这会改变有效 batch size（从 16 降到 8），可能影响收敛。如果内存允许，建议同时增加 batch_size：

```yaml
# configs/nuscenes/recondrive_ae.yaml:88
batch_size: 4  # 从 2 改为 4（如果内存允许）
```

---

## 完整修改清单

### 1. 修改 `configs/nuscenes/recondrive_ae.yaml`
```yaml
data_cfg:
  # ...
  batch_size: 4  # 可选：从 2 改为 4（如果 GPU 内存允许）
  # ...
  num_workers: 4  # 必须：从 0 改为 4
```

### 2. 修改 `scripts/trainer.py`
```python
# 第 190 行附近
trainer = pl.Trainer(
    max_epochs=main_cfg.get('train_epoch', 50),
    accelerator="gpu",
    devices=main_cfg['devices'],
    precision="bf16-mixed",  # 添加这一行
    gradient_clip_algorithm="norm",
    accumulate_grad_batches=4,  # 从 8 改为 4
    # ... 其他参数不变
)
```

### 3. 修改 `models/recondrive_ae_model.py`
```python
# 在 training_step 方法中，第 336 行附近
def training_step(self, batch_input, batch_idx):
    # ... 前面的代码 ...
    
    # 4. 渲染重建的 Gaussians
    batch_render_data = self.get_render_data(batch_input)
    
    # 添加：训练时只渲染前 3 个相机
    if self.training:
        for key in batch_render_data:
            if isinstance(batch_render_data[key], list) and len(batch_render_data[key]) == 6:
                batch_render_data[key] = batch_render_data[key][:3]
    
    batch_splating_data = self.render_splating_imgs(batch_recontrast_recon, batch_render_data)
    # ... 后面的代码不变 ...
```

---

## 测试优化效果

### 运行训练并监控速度
```bash
# 记录优化前的速度
# 观察每个 step 的时间

# 应用优化后重新训练
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/train.sh 8

# 观察速度提升
# 预期：从 ~25s/step 降到 ~6-8s/step（3-4× 加速）
```

### 使用 PyTorch Profiler 验证
```python
# 在 training_step 开头添加（仅用于测试）
if batch_idx == 10:  # 只在第 10 个 batch 分析
    from torch.profiler import profile, ProfilerActivity
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        # 原有的 training_step 代码
        pass
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
```

---

## 预期效果

| 优化项 | 加速比 | 实施难度 |
|--------|--------|---------|
| 多进程数据加载 | 1.5-2× | 极低（改 1 行） |
| 混合精度训练 | 1.5× | 极低（改 1 行） |
| 减少渲染相机 | 1.3× | 低（加几行代码） |
| 减少梯度累积 | 感觉 2× | 极低（改 1 行） |
| **总计** | **3-4×** | **低** |

**优化前**: ~25 秒/step  
**优化后**: ~6-8 秒/step

---

## 下一步：核心优化（需要 1-2 周，20-30× 加速）

完成快速优化后，如果还需要更快的速度，可以进行核心优化：

1. **替换 Sparse 3D CNN 为 torchsparse**（10-20× 加速）
2. **批量化 Chamfer Loss 计算**（50-100× 加速）

详见 `docs/TRAINING_PERFORMANCE_ANALYSIS.md` 的"优化建议"部分。
