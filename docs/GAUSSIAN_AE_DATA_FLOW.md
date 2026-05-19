# Gaussian Autoencoder 数据流详解（含 Batch 维度）

## 配置参数

```yaml
batch_size: 2           # 每个 batch 2 个场景
num_cams: 6             # 每个场景 6 个相机
height: 280, width: 518 # 每个相机图像分辨率
voxel_size: 0.4m        # 体素大小
K: 8                    # 每个体素输出的 Gaussian 数量
```

## 完整数据流（逐步详解）

### 1️⃣ **输入：ReconDrive Backbone 输出的密集 Gaussians**

```python
# 来自 recondrive_ae_model.py:320
batch_recontrast_data = self.get_recontrast_data(batch_input, batch_idx)
```

**Shape 详解：**
```python
batch_recontrast_data = {
    'xyz':          [B, N, 3]      # B=2, N=6×280×518×2 = 1,740,960
    'rot_maps':     [B, N, 4]      # 四元数旋转
    'scale_maps':   [B, N, 3]      # 3D 尺度
    'opacity_maps': [B, N, 1]      # 不透明度
    'sh_maps':      [B, N, 25, 3]  # 球谐系数 (SH degree 4)
}
```

**为什么 N = 1,740,960？**
- 6 个相机
- 每个相机 280×518 = 145,040 像素
- 2 帧（frame_ids: [0, 1]）
- 总计：6 × 145,040 × 2 = **1,740,960 个 Gaussians**

**这是密集表示**：每个像素对应一个 3D Gaussian。

---

### 2️⃣ **Voxelization：密集 → 稀疏**

```python
# voxelizer.py:146
(voxel_features, voxel_indices, voxel_centers, B,
 all_gt_features, all_gt_voxel_id) = self.voxelizer.voxelize_with_all_gt(batch_recontrast_data)
```

**处理过程：**

#### Step 2.1: 计算 Voxel Grid 尺寸
```python
x_range = [-40, 40]  → nx = 80/0.4 = 200 voxels
y_range = [-40, 40]  → ny = 80/0.4 = 200 voxels
z_range = [-1, 5.4]  → nz = 6.4/0.4 = 16 voxels

# 理论最大 voxel 数（如果全部占据）
max_voxels_per_batch = 200 × 200 × 16 = 640,000
max_voxels_total = B × 640,000 = 1,280,000
```

#### Step 2.2: 将 1,740,960 个 Gaussians 分配到 Voxels
```python
# 展平 batch 维度
xyz_flat = xyz.reshape(-1, 3)  # [B*N, 3] = [3,481,920, 3]

# 计算每个 Gaussian 属于哪个 voxel
xi = ((xyz_flat[:, 0] - x_min) / voxel_size).long()  # [3,481,920]
yi = ((xyz_flat[:, 1] - y_min) / voxel_size).long()
zi = ((xyz_flat[:, 2] - z_min) / voxel_size).long()

# 过滤出在 grid 范围内的点
valid = (xi >= 0) & (xi < 200) & (yi >= 0) & (yi < 200) & (zi >= 0) & (zi < 16)
# 假设有 1,500,000 个点在范围内（其余在 grid 外被丢弃）
```

#### Step 2.3: 每个 Voxel 只保留最高 opacity 的 Gaussian
```python
# 为每个 voxel 分配唯一 key
voxel_key = batch_id * 640,000 + z*40,000 + y*200 + x  # [1,500,000]

# 找到唯一的 voxels
unique_keys, inv_map = torch.unique(voxel_key, return_inverse=True)
M = len(unique_keys)  # 假设 M = 50,000（实际占据的 voxel 数）

# 每个 voxel 选择 opacity 最高的 Gaussian 作为代表
_, argmax = scatter_max(opacity_flat, inv_map, dim=0)
voxel_features = feat_flat[argmax]  # [M, 86] = [50,000, 86]
```

**输出 Shape：**
```python
voxel_features:  [M, 86]        # M=50,000（占据的 voxels）
voxel_indices:   [M, 4]         # (batch_id, z, y, x) 每个 voxel 的坐标
voxel_centers:   [M, 3]         # 每个 voxel 的中心 XYZ 坐标
batch_size:      2

# 用于 Chamfer Loss 的额外输出
all_gt_features: [N_valid, 86]  # N_valid=1,500,000（所有在 grid 内的 GT Gaussians）
all_gt_voxel_id: [N_valid]      # 每个 GT Gaussian 属于哪个 voxel (0..M-1)
```

**关键理解：**
- **输入**：B=2 个 batch，每个有 ~870K 个密集 Gaussians
- **输出**：M=50K 个稀疏 voxels（跨 2 个 batch）
- **压缩比**：1,740,960 → 50,000（约 **35倍压缩**）

**Batch 信息保存在哪里？**
- `voxel_indices[:, 0]` 存储 batch_id（0 或 1）
- 例如：前 25,000 个 voxels 属于 batch 0，后 25,000 个属于 batch 1

---

### 3️⃣ **Encoding Head：特征维度变换**

```python
# autoencoder.py:133
voxel_feat_enc = self.enc_head(voxel_feat)  # [M, 86] → [M, 32]
```

**Shape：**
```python
输入:  [50,000, 86]   # 86 维 Gaussian 特征
输出:  [50,000, 32]   # 压缩到 32 维（encoder 第一层通道数）
```

---

### 4️⃣ **Sparse 3D CNN Encoder：空间下采样**

```python
# autoencoder.py:134-139
coords = torch.cat([batch_id, x, y, z], dim=1)  # [M, 4]
sp_input = SparseTensor(features=voxel_feat_enc, coordinates=coords)
latent, skip1, skip2, skip3 = self.encoder(sp_input)
```

**Encoder 结构（3 次下采样）：**

```python
# Stage 0: 原始分辨率
sp_input:  SparseTensor
  .F:  [M, 32]    = [50,000, 32]
  .C:  [M, 4]     = [50,000, 4]  # (batch, x, y, z)

# ↓ SparseBlock(32→64) + Downsample(stride=2)

# Stage 1: 1/2 分辨率
skip1:  SparseTensor
  .F:  [M1, 64]   ≈ [25,000, 64]   # 空间下采样 2×，voxel 数减半
  .C:  [M1, 4]    ≈ [25,000, 4]

# ↓ SparseBlock(64→128) + Downsample(stride=2)

# Stage 2: 1/4 分辨率
skip2:  SparseTensor
  .F:  [M2, 128]  ≈ [12,500, 128]  # 再下采样 2×
  .C:  [M2, 4]    ≈ [12,500, 4]

# ↓ SparseBlock(128→256) + Downsample(stride=2)

# Stage 3: 1/8 分辨率
skip3:  SparseTensor
  .F:  [M3, 256]  ≈ [6,250, 256]   # 再下采样 2×
  .C:  [M3, 4]    ≈ [6,250, 4]

# ↓ Final Downsample(stride=2)

# Latent: 1/8 分辨率（最终 bottleneck）
latent:  SparseTensor
  .F:  [M_latent, 256]  ≈ [3,125, 256]  # 最终 latent 表示
  .C:  [M_latent, 4]    ≈ [3,125, 4]
```

**Latent 的 Batch 维度：**
```python
# latent.C 的第一列是 batch_id
batch_0_mask = (latent.C[:, 0] == 0)  # 属于第一个 batch 的 voxels
batch_1_mask = (latent.C[:, 0] == 1)  # 属于第二个 batch 的 voxels

# 假设两个 batch 的 voxel 数相近
batch_0_voxels ≈ 1,562 个
batch_1_voxels ≈ 1,563 个
总计 = 3,125 个
```

**关键理解：**
- **Latent 不是规则的 [B, C, D, H, W] 张量**
- 而是 **稀疏表示**：只存储占据的 voxels
- Batch 信息通过 `coordinates[:, 0]` 编码
- 空间分辨率：原始 voxel grid 的 1/8

---

### 5️⃣ **Sparse 3D CNN Decoder：空间上采样**

```python
# autoencoder.py:142
decoded_sp = self.decoder(latent, skip1, skip2, skip3)
```

**Decoder 结构（3 次上采样 + skip connections）：**

```python
# Latent: 1/8 分辨率
latent:  [M_latent, 256]  ≈ [3,125, 256]

# ↑ Upsample(stride=2) + skip3 + SparseBlock

# Stage 3: 1/4 分辨率
x3:  [M3', 256]  ≈ [6,250, 256]

# ↑ Upsample(stride=2) + skip2 + SparseBlock

# Stage 2: 1/2 分辨率
x2:  [M2', 128]  ≈ [12,500, 128]

# ↑ Upsample(stride=2) + skip1 + SparseBlock

# Stage 1: 原始分辨率
decoded_sp:  SparseTensor
  .F:  [M', 32]   ≈ [50,000, 32]   # 恢复到原始 voxel 分辨率
  .C:  [M', 4]    ≈ [50,000, 4]
```

**注意：**
- M' ≈ M（但不完全相等，因为 sparse conv 的边界效应）
- 通过 skip connections 保留高频细节

---

### 6️⃣ **Decoding Head：每个 Voxel 输出 K 个 Gaussians**

```python
# autoencoder.py:143
pred_raw = self.dec_head(decoded_sp.F)  # [M', 32] → [M', K, 86]
```

**Shape：**
```python
输入:  [50,000, 32]        # 每个 voxel 的 32 维特征
输出:  [50,000, 8, 86]     # 每个 voxel 输出 K=8 个 Gaussians，每个 86 维
```

**86 维 Gaussian 特征分解：**
```python
xyz_offset:  [50,000, 8, 3]   # 相对 voxel 中心的偏移
rotation:    [50,000, 8, 4]   # 四元数
scale:       [50,000, 8, 3]   # 3D 尺度
opacity:     [50,000, 8, 1]   # 不透明度
sh:          [50,000, 8, 75]  # 球谐系数 (25×3)
```

---

### 7️⃣ **Devoxelization：稀疏 → 半密集**

```python
# autoencoder.py:146
pred_recon = self.voxelizer.devoxelize(pred_raw, voxel_idx, voxel_centers, B)
```

**处理过程：**

#### Step 7.1: 应用激活函数
```python
xyz_offset = tanh(pred_raw[..., 0:3]) * (voxel_size/2)  # 限制在 voxel 内
rotation = normalize(pred_raw[..., 3:7])
scale = softplus(pred_raw[..., 7:10]) * 0.01
opacity = sigmoid(pred_raw[..., 10:11])
sh = pred_raw[..., 11:86]
```

#### Step 7.2: 计算绝对坐标
```python
# voxel_centers: [M, 3] → [M, 1, 3] (broadcast)
xyz_abs = voxel_centers.unsqueeze(1) + xyz_offset  # [M, K, 3]
```

#### Step 7.3: 按 Batch 分组
```python
batch_ids = voxel_indices[:, 0]  # [M]

# Batch 0: 假设有 25,000 个 voxels
batch_0_mask = (batch_ids == 0)
batch_0_gaussians = 25,000 × 8 = 200,000 个 Gaussians

# Batch 1: 假设有 25,000 个 voxels
batch_1_mask = (batch_ids == 1)
batch_1_gaussians = 25,000 × 8 = 200,000 个 Gaussians
```

#### Step 7.4: Padding 到相同长度
```python
# 找到最大长度
max_n = max(200,000, 200,000) = 200,000

# 输出（padding 后）
pred_recon = {
    'xyz':          [B, max_n, 3]      = [2, 200,000, 3]
    'rot_maps':     [B, max_n, 4]      = [2, 200,000, 4]
    'scale_maps':   [B, max_n, 3]      = [2, 200,000, 3]
    'opacity_maps': [B, max_n, 1]      = [2, 200,000, 1]
    'sh_maps':      [B, max_n, 25, 3]  = [2, 200,000, 25, 3]
}
```

---

## 📊 完整数据流总结表

| 阶段 | 数据结构 | Shape | Batch 信息 | 说明 |
|------|---------|-------|-----------|------|
| **输入** | Dense Tensor | `[2, 1,740,960, *]` | 第一维 | 每个像素一个 Gaussian |
| **Voxelization** | Sparse List | `[50,000, 86]` | `indices[:, 0]` | 压缩 35× |
| **Enc Head** | Sparse List | `[50,000, 32]` | `indices[:, 0]` | 特征压缩 |
| **Encoder Stage1** | SparseTensor | `[25,000, 64]` | `coords[:, 0]` | 空间 1/2 |
| **Encoder Stage2** | SparseTensor | `[12,500, 128]` | `coords[:, 0]` | 空间 1/4 |
| **Encoder Stage3** | SparseTensor | `[6,250, 256]` | `coords[:, 0]` | 空间 1/8 |
| **🔥 Latent** | SparseTensor | `[3,125, 256]` | `coords[:, 0]` | **Bottleneck** |
| **Decoder Stage3** | SparseTensor | `[6,250, 256]` | `coords[:, 0]` | 空间 1/4 |
| **Decoder Stage2** | SparseTensor | `[12,500, 128]` | `coords[:, 0]` | 空间 1/2 |
| **Decoder Stage1** | SparseTensor | `[50,000, 32]` | `coords[:, 0]` | 原始分辨率 |
| **Dec Head** | Sparse List | `[50,000, 8, 86]` | `indices[:, 0]` | K 个/voxel |
| **输出** | Dense Tensor | `[2, 200,000, *]` | 第一维 | 重建的 Gaussians |

---

## 🔑 关键要点

### 1. **Batch 维度的处理方式**
- **输入/输出**：标准的 `[B, N, ...]` 格式
- **中间层（稀疏）**：Batch ID 编码在 `coordinates[:, 0]`
- **优势**：不同 batch 的 voxel 数可以不同，无需 padding

### 2. **稀疏性的演变**
```
密集 (1.7M) → 稀疏 (50K) → 更稀疏 (3.1K latent) → 稀疏 (50K) → 半密集 (200K)
```

### 3. **压缩比**
- **输入 → Voxelization**: 1,740,960 → 50,000 (**35×**)
- **Voxelization → Latent**: 50,000 → 3,125 (**16×**)
- **总压缩**: 1,740,960 → 3,125 (**557×**)
- **输出**: 200,000 个 Gaussians（比输入少 **8.7×**）

### 4. **Latent 的实际含义**
```python
latent = SparseTensor(
    features=[3,125, 256],      # 3,125 个占据的 latent voxels
    coordinates=[3,125, 4]      # (batch_id, x/8, y/8, z/8)
)

# 对于 batch_size=2:
# - Batch 0: ~1,562 个 latent voxels
# - Batch 1: ~1,563 个 latent voxels
# - 每个 latent voxel 对应原始空间的 8×8×8 = 512 个原始 voxels
```

---

## 💡 为什么这样设计？

1. **稀疏性**：自动驾驶场景大部分是空的（天空、远处），稀疏表示节省内存
2. **Batch 灵活性**：不同场景的点云密度不同，稀疏表示无需 padding
3. **多尺度**：Encoder/Decoder 的 skip connections 保留不同尺度的细节
4. **K 个/voxel**：允许一个 voxel 内有多个 Gaussians，提高表达能力
