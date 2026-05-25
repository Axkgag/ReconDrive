import torch
import torch.nn as nn
import torch.nn.functional as F


class SparseTensor:
    """
    Lightweight sparse tensor wrapper using PyTorch native operations.

    Stores features as dense [M, C] and coordinates as [M, 4] (batch, x, y, z).
    """
    def __init__(self, features, coordinates):
        self.F = features          # [M, C]
        self.C = coordinates       # [M, 4] int32

    def __add__(self, other):
        """Element-wise addition (assumes same coordinates)."""
        return SparseTensor(self.F + other.F, self.C)


class SparseConv3d(nn.Module):
    """
    Sparse 3D convolution using dense Conv3d on occupied regions.

    For each batch, builds a dense 3D grid, applies Conv3d, then extracts
    the occupied voxels. Inefficient but avoids external dependencies.
    """
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, dimension=3):
        super().__init__()
        self.conv = nn.Conv3d(in_ch, out_ch, kernel_size, stride=stride,
                              padding=kernel_size//2, bias=False)
        self.stride = stride

    def forward(self, x):
        """x: SparseTensor → SparseTensor"""
        device = x.F.device
        batch_ids = x.C[:, 0].long()
        coords_xyz = x.C[:, 1:].long()  # [M, 3]

        # Handle empty input
        if coords_xyz.shape[0] == 0:
            return SparseTensor(
                torch.zeros(0, self.conv.out_channels, device=device),
                torch.zeros(0, 4, device=device, dtype=torch.int32)
            )

        # Determine grid size
        max_coords = coords_xyz.max(dim=0)[0] + 1
        grid_size = (max_coords[0].item(), max_coords[1].item(), max_coords[2].item())

        out_features_list = []
        out_coords_list = []

        for b in batch_ids.unique():
            mask = batch_ids == b
            feats = x.F[mask]           # [Mb, C]
            coords = coords_xyz[mask]   # [Mb, 3]

            # Build dense grid
            dense = torch.zeros(1, x.F.shape[1], *grid_size, device=device)
            dense[0, :, coords[:, 0], coords[:, 1], coords[:, 2]] = feats.T

            # Apply conv
            out_dense = self.conv(dense)  # [1, C_out, X', Y', Z']

            # Extract occupied voxels (downsample coords if stride > 1)
            if self.stride > 1:
                out_coords = coords // self.stride
            else:
                out_coords = coords

            # Clamp to output grid bounds
            out_shape = out_dense.shape[2:]
            out_coords = torch.clamp(out_coords,
                                     min=torch.zeros(3, device=device, dtype=torch.long),
                                     max=torch.tensor(out_shape, device=device, dtype=torch.long) - 1)

            out_feats = out_dense[0, :, out_coords[:, 0], out_coords[:, 1], out_coords[:, 2]].T

            # Add batch index
            out_coords_b = torch.cat([
                torch.full((out_coords.shape[0], 1), b, device=device, dtype=torch.int32),
                out_coords.int()
            ], dim=1)

            out_features_list.append(out_feats)
            out_coords_list.append(out_coords_b)

        return SparseTensor(
            torch.cat(out_features_list, dim=0),
            torch.cat(out_coords_list, dim=0)
        )


class SparseConvTranspose3d(nn.Module):
    """Sparse 3D transposed convolution (upsampling)."""
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=2, dimension=3):
        super().__init__()
        self.conv = nn.ConvTranspose3d(in_ch, out_ch, kernel_size, stride=stride,
                                       padding=kernel_size//2, output_padding=stride-1, bias=False)
        self.stride = stride

    def forward(self, x, target_coords: torch.Tensor | None = None):
        device = x.F.device
        batch_ids = x.C[:, 0].long()
        coords_xyz = x.C[:, 1:].long()

        # Handle empty input
        if coords_xyz.shape[0] == 0:
            if target_coords is None or target_coords.shape[0] == 0:
                return SparseTensor(
                    torch.zeros(0, self.conv.out_channels, device=device),
                    torch.zeros(0, 4, device=device, dtype=torch.int32)
                )
            target_coords = target_coords.to(device=device, dtype=torch.int32)
            out_features = torch.zeros(
                target_coords.shape[0],
                self.conv.out_channels,
                device=device,
            )
            return SparseTensor(out_features, target_coords)

        max_coords = coords_xyz.max(dim=0)[0] + 1
        grid_size = (max_coords[0].item(), max_coords[1].item(), max_coords[2].item())

        if target_coords is not None:
            target_coords = target_coords.to(device=device, dtype=torch.int32)
            out_features = torch.zeros(
                target_coords.shape[0],
                self.conv.out_channels,
                device=device,
            )
        else:
            out_features_list = []
            out_coords_list = []

        for b in batch_ids.unique():
            mask = batch_ids == b
            feats = x.F[mask]
            coords = coords_xyz[mask]

            dense = torch.zeros(1, x.F.shape[1], *grid_size, device=device)
            dense[0, :, coords[:, 0], coords[:, 1], coords[:, 2]] = feats.T

            out_dense = self.conv(dense)

            out_shape = out_dense.shape[2:]
            out_shape_tensor = torch.tensor(out_shape, device=device, dtype=torch.long)

            if target_coords is not None:
                target_mask = target_coords[:, 0] == b
                if not torch.any(target_mask):
                    continue
                target_xyz = target_coords[target_mask][:, 1:].long()
                valid = (target_xyz >= 0).all(dim=1) & (target_xyz < out_shape_tensor).all(dim=1)
                if torch.any(valid):
                    valid_xyz = target_xyz[valid]
                    out_feats = out_dense[0, :, valid_xyz[:, 0], valid_xyz[:, 1], valid_xyz[:, 2]].T
                    out_features[target_mask.nonzero(as_tuple=False).squeeze(1)[valid]] = out_feats
            else:
                # Upsample coordinates
                out_coords = coords * self.stride
                out_coords = torch.clamp(out_coords,
                                         min=torch.zeros(3, device=device, dtype=torch.long),
                                         max=out_shape_tensor - 1)

                out_feats = out_dense[0, :, out_coords[:, 0], out_coords[:, 1], out_coords[:, 2]].T

                out_coords_b = torch.cat([
                    torch.full((out_coords.shape[0], 1), b, device=device, dtype=torch.int32),
                    out_coords.int()
                ], dim=1)

                out_features_list.append(out_feats)
                out_coords_list.append(out_coords_b)

        if target_coords is not None:
            return SparseTensor(out_features, target_coords)
        return SparseTensor(
            torch.cat(out_features_list, dim=0),
            torch.cat(out_coords_list, dim=0)
        )


class SparseBatchNorm(nn.Module):
    """Batch normalization for sparse tensors."""
    def __init__(self, channels):
        super().__init__()
        self.bn = nn.BatchNorm1d(channels)

    def forward(self, x):
        return SparseTensor(self.bn(x.F), x.C)


class SparseReLU(nn.Module):
    """ReLU for sparse tensors."""
    def __init__(self, inplace=True):
        super().__init__()
        self.inplace = inplace

    def forward(self, x):
        return SparseTensor(F.relu(x.F, inplace=self.inplace), x.C)


class SparseLinear(nn.Module):
    """Linear layer for sparse tensors."""
    def __init__(self, in_ch, out_ch, bias=False):
        super().__init__()
        self.linear = nn.Linear(in_ch, out_ch, bias=bias)

    def forward(self, x):
        return SparseTensor(self.linear(x.F), x.C)


class SparseBlock(nn.Module):
    """Two sparse conv layers with BN+ReLU."""
    def __init__(self, in_ch, out_ch, dimension=3):
        super().__init__()
        self.conv1 = SparseConv3d(in_ch, out_ch, kernel_size=3, stride=1)
        self.bn1   = SparseBatchNorm(out_ch)
        self.conv2 = SparseConv3d(out_ch, out_ch, kernel_size=3, stride=1)
        self.bn2   = SparseBatchNorm(out_ch)
        self.act   = SparseReLU(inplace=True)

    def forward(self, x):
        x = self.act(self.bn1(self.conv1(x)))
        x = self.act(self.bn2(self.conv2(x)))
        return x


class SparseEncoder(nn.Module):
    """3-stage sparse 3D CNN encoder using PyTorch native ops."""
    def __init__(self, channels=(32, 64, 128, 256), dimension=3):
        super().__init__()
        c0, c1, c2, c3 = channels

        self.stage0 = SparseBlock(c0, c1, dimension)
        self.down1  = SparseConv3d(c1, c1, kernel_size=3, stride=2)
        self.bn_d1  = SparseBatchNorm(c1)

        self.stage1 = SparseBlock(c1, c2, dimension)
        self.down2  = SparseConv3d(c2, c2, kernel_size=3, stride=2)
        self.bn_d2  = SparseBatchNorm(c2)

        self.stage2 = SparseBlock(c2, c3, dimension)
        self.down3  = SparseConv3d(c3, c3, kernel_size=3, stride=2)
        self.bn_d3  = SparseBatchNorm(c3)

        self.act = SparseReLU(inplace=True)

    def forward(self, x):
        x     = self.stage0(x)
        skip1 = x

        x = self.act(self.bn_d1(self.down1(x)))
        x     = self.stage1(x)
        skip2 = x

        x = self.act(self.bn_d2(self.down2(x)))
        x     = self.stage2(x)
        skip3 = x

        x = self.act(self.bn_d3(self.down3(x)))
        latent = x

        return latent, skip1, skip2, skip3


class SparseDecoder(nn.Module):
    """3-stage sparse 3D CNN decoder using PyTorch native ops."""
    def __init__(self, channels=(32, 64, 128, 256), dimension=3):
        super().__init__()
        c0, c1, c2, c3 = channels

        self.up3   = SparseConvTranspose3d(c3, c2, kernel_size=3, stride=2)
        self.bn_u3 = SparseBatchNorm(c2)
        self.proj3 = SparseLinear(c3, c2, bias=False)
        self.stage3 = SparseBlock(c2, c2, dimension)

        self.up2   = SparseConvTranspose3d(c2, c1, kernel_size=3, stride=2)
        self.bn_u2 = SparseBatchNorm(c1)
        self.proj2 = SparseLinear(c2, c1, bias=False)
        self.stage2 = SparseBlock(c1, c1, dimension)

        self.up1   = SparseConvTranspose3d(c1, c0, kernel_size=3, stride=2)
        self.bn_u1 = SparseBatchNorm(c0)
        self.proj1 = SparseLinear(c1, c0, bias=False)
        self.stage1 = SparseBlock(c0, c0, dimension)

        self.act = SparseReLU(inplace=True)

    def forward(self, latent, skip1, skip2, skip3):
        x = self.act(self.bn_u3(self.up3(latent, target_coords=skip3.C)))
        x = x + self.proj3(skip3)
        x = self.stage3(x)

        x = self.act(self.bn_u2(self.up2(x, target_coords=skip2.C)))
        x = x + self.proj2(skip2)
        x = self.stage2(x)

        x = self.act(self.bn_u1(self.up1(x, target_coords=skip1.C)))
        x = x + self.proj1(skip1)
        x = self.stage1(x)

        return x
