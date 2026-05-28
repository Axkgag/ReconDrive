# Copyright (c) 2024-present.
#
# Voxel-based 3DGS head for ReconDrive (VolSplat-style lift + voxel aggregate).

from typing import List, Tuple, Union

import torch
import torch.nn as nn

from models.gaussian_util import depth2pc
from models.gaussian_autoencoder.sparse_cnn import SparseTensor, SparseEncoder, SparseDecoder, SparseLinear
from models.vggt.heads.dpt_head import DPTHead


# class VoxelFeatureRefiner(nn.Module):
#     """Sparse 3D UNet-style feature refiner (VolSplat-inspired, no MinkowskiEngine)."""

#     def __init__(self, feature_dim: int, hidden_dim: int = 64) -> None:
#         super().__init__()
#         hidden_dim = min(hidden_dim, feature_dim)
#         channels = (hidden_dim, hidden_dim * 2, hidden_dim * 4, hidden_dim * 8)

#         self.input_proj = SparseLinear(feature_dim, hidden_dim, bias=False)
#         self.encoder = SparseEncoder(channels=channels)
#         self.decoder = SparseDecoder(channels=channels)
#         self.output_proj = SparseLinear(hidden_dim, feature_dim, bias=False)

#     def forward(self, voxel_feats: torch.Tensor, voxel_coords: torch.Tensor) -> torch.Tensor:
#         if voxel_feats.numel() == 0:
#             return voxel_feats

#         coords = voxel_coords.to(dtype=torch.int32)
#         x = SparseTensor(voxel_feats, coords)
#         x = self.input_proj(x)
#         latent, skip1, skip2, skip3 = self.encoder(x)
#         x = self.decoder(latent, skip1, skip2, skip3)
#         x = self.output_proj(x)
#         return x.F


class VoxelFeatureRefiner(nn.Module):
    """Placeholder for sparse 3D CNN/UNet refinement."""

    def __init__(self):
        super().__init__()

    def forward(self, voxel_feats: torch.Tensor, voxel_coords: torch.Tensor) -> torch.Tensor:
        return voxel_feats


class VGGT_Voxel_GS_Head(nn.Module):
    def __init__(
        self,
        dim_in: int,
        patch_size: int = 14,
        sh_degree: int = 4,
        feature_dim: int = 256,
        gaussians_per_voxel: int = 1,
        voxel_size: float = 0.4,
        x_range: Tuple[float, float] = (-40.0, 40.0),
        y_range: Tuple[float, float] = (-40.0, 40.0),
        z_range: Tuple[float, float] = (-1.0, 5.4),
        pos_embed: bool = True,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.voxel_size = voxel_size
        self.x_range = x_range
        self.y_range = y_range
        self.z_range = z_range
        self.gaussians_per_voxel = gaussians_per_voxel

        self.d_sh = (sh_degree + 1) ** 2
        self.raw_gs_dim = 3 + 4 + 3 + 1 + 3 * self.d_sh  # offset + rot + scale + opacity + SH
        self.opacity_index = 3 + 4 + 3
        self.invalid_opacity = -20.0

        self.feature_head = DPTHead(
            dim_in=dim_in,
            patch_size=patch_size,
            output_dim=feature_dim,
            features=feature_dim,
            pos_embed=pos_embed,
            feature_only=True,
        )

        self.refiner = VoxelFeatureRefiner()
        self.decoder = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim, self.raw_gs_dim * self.gaussians_per_voxel),
        )

    def forward(
        self,
        aggregated_tokens_list: List[torch.Tensor],
        images: torch.Tensor,
        patch_start_idx: int,
        depth_maps: torch.Tensor,
        intrinsics: torch.Tensor,
        extrinsics: torch.Tensor,
    ) -> torch.Tensor:
        # 2D feature map from tokens: [B, S, C, H, W]
        features = self.feature_head(aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx)
        features = features.permute(0, 1, 3, 4, 2).contiguous()  # [B, S, H, W, C]

        if depth_maps.dim() == 5 and depth_maps.shape[-1] == 1:
            depth = depth_maps[..., 0]
        else:
            depth = depth_maps

        b, s, h, w = depth.shape
        device = depth.device
        c = features.shape[-1]

        features_flat = features.view(b * s, h * w, c)
        depth_flat = depth.view(b * s, h * w)

        intrinsics_4x4 = self._ensure_4x4(intrinsics, device=device, dtype=depth.dtype)
        extrinsics_4x4 = self._ensure_4x4(extrinsics, device=device, dtype=depth.dtype)
        e2c = torch.linalg.inv(extrinsics_4x4.view(b * s, 4, 4))
        k = intrinsics_4x4.view(b * s, 4, 4)

        points = depth2pc(depth.view(b * s, h, w), e2c, k)  # [B*S, H*W, 3]

        points_flat = points.reshape(-1, 3)
        feats_flat = features_flat.view(-1, c)
        batch_ids = torch.arange(b * s, device=device).unsqueeze(1).expand(b * s, h * w).reshape(-1)

        valid_mask = depth_flat.view(-1) > 0
        voxel_coords = torch.floor((points_flat - points_flat.new_tensor([self.x_range[0], self.y_range[0], self.z_range[0]])) / self.voxel_size).long()

        nx = int((self.x_range[1] - self.x_range[0]) / self.voxel_size)
        ny = int((self.y_range[1] - self.y_range[0]) / self.voxel_size)
        nz = int((self.z_range[1] - self.z_range[0]) / self.voxel_size)

        in_range = (
            (voxel_coords[:, 0] >= 0) & (voxel_coords[:, 0] < nx) &
            (voxel_coords[:, 1] >= 0) & (voxel_coords[:, 1] < ny) &
            (voxel_coords[:, 2] >= 0) & (voxel_coords[:, 2] < nz)
        )
        valid_mask = valid_mask & in_range

        valid_idx = valid_mask.nonzero(as_tuple=False).squeeze(-1)
        if valid_idx.numel() == 0:
            raw_full = torch.zeros(
                b * s * h * w, self.gaussians_per_voxel, self.raw_gs_dim, device=device, dtype=depth.dtype
            )
            raw_full[:, :, self.opacity_index] = self.invalid_opacity
            raw_full_reshaped = raw_full.view(b, s, h, w, self.gaussians_per_voxel, self.raw_gs_dim)
            return raw_full_reshaped

        coords = torch.stack(
            [batch_ids[valid_idx], voxel_coords[valid_idx, 0], voxel_coords[valid_idx, 1], voxel_coords[valid_idx, 2]],
            dim=-1,
        )
        unique_coords, inv = torch.unique(coords, dim=0, return_inverse=True)
        num_voxels = unique_coords.shape[0]

        voxel_feats = torch.zeros(num_voxels, c, device=device, dtype=depth.dtype)
        voxel_feats.scatter_add_(0, inv.unsqueeze(-1).expand(-1, c), feats_flat[valid_idx])
        counts = torch.zeros(num_voxels, 1, device=device, dtype=depth.dtype)
        counts.scatter_add_(0, inv.unsqueeze(-1), torch.ones_like(inv, dtype=depth.dtype).unsqueeze(-1))
        voxel_feats = voxel_feats / counts.clamp_min(1.0)

        voxel_feats = self.refiner(voxel_feats, unique_coords)
        voxel_params = self.decoder(voxel_feats).view(num_voxels, self.gaussians_per_voxel, self.raw_gs_dim)

        raw_full = torch.zeros(
            b * s * h * w, self.gaussians_per_voxel, self.raw_gs_dim, device=device, dtype=depth.dtype
        )
        raw_full[:, :, self.opacity_index] = self.invalid_opacity
        raw_full[valid_idx] = voxel_params[inv]
        raw_full_reshaped = raw_full.view(b, s, h, w, self.gaussians_per_voxel, self.raw_gs_dim)
        return raw_full_reshaped

    @staticmethod
    def _ensure_4x4(matrix: torch.Tensor, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if matrix.shape[-2:] == (4, 4):
            return matrix.to(device=device, dtype=dtype)
        if matrix.shape[-2:] != (3, 3):
            raise ValueError(f"Expected intrinsics/extrinsics with shape (...,3,3) or (...,4,4), got {matrix.shape}")
        eye = torch.eye(4, device=device, dtype=dtype)
        eye = eye.view((1,) * len(matrix.shape[:-2]) + (4, 4))
        expanded = eye.repeat(*matrix.shape[:-2], 1, 1)
        expanded[..., :3, :3] = matrix.to(device=device, dtype=dtype)
        return expanded
