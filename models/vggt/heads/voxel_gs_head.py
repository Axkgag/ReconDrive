# Copyright (c) 2024-present.
#
# Voxel-based 3DGS head for ReconDrive (VolSplat-style lift + voxel aggregate).

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.gaussian_util import depth2pc
from models.gaussian_autoencoder.sparse_cnn import SparseTensor, SparseEncoder, SparseDecoder, SparseLinear
from models.vggt.heads.dpt_head import DPTHead

try:
    from mmcv.ops.multi_scale_deform_attn import MultiScaleDeformableAttnFunction

    HAS_MMCV_MS_DEFORM_ATTN = True
except Exception:
    MultiScaleDeformableAttnFunction = None
    HAS_MMCV_MS_DEFORM_ATTN = False


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


class SparseLocalConv3d(nn.Module):
    """Sparse 3D local aggregation without densifying the full OCC volume."""

    def __init__(self, feature_dim: int, grid_size: Tuple[int, int, int]) -> None:
        super().__init__()
        self.grid_size = tuple(int(v) for v in grid_size)
        self.self_proj = nn.Linear(feature_dim, feature_dim)
        self.neighbor_proj = nn.Linear(feature_dim, feature_dim)
        self.norm = nn.LayerNorm(feature_dim)
        self.act = nn.GELU()

        offsets = [
            (dx, dy, dz)
            for dx in (-1, 0, 1)
            for dy in (-1, 0, 1)
            for dz in (-1, 0, 1)
            if not (dx == 0 and dy == 0 and dz == 0)
        ]
        self.register_buffer("neighbor_offsets", torch.tensor(offsets, dtype=torch.long), persistent=False)

    def _encode_coords(self, coords: torch.Tensor) -> torch.Tensor:
        nx, ny, nz = self.grid_size
        coords = coords.long()
        return (((coords[:, 0] * nx + coords[:, 1]) * ny + coords[:, 2]) * nz + coords[:, 3])

    def forward(self, feats: torch.Tensor, coords_with_batch: torch.Tensor) -> torch.Tensor:
        if feats.numel() == 0:
            return feats

        coords = coords_with_batch.to(device=feats.device, dtype=torch.long)
        keys = self._encode_coords(coords)
        sorted_keys, order = torch.sort(keys)
        sorted_feats = feats[order]

        neighbor_sum = torch.zeros_like(feats)
        neighbor_count = feats.new_zeros((feats.shape[0], 1))
        nx, ny, nz = self.grid_size

        for offset in self.neighbor_offsets.to(device=feats.device):
            neighbor_coords = coords.clone()
            neighbor_coords[:, 1:] = neighbor_coords[:, 1:] + offset.view(1, 3)
            valid = (
                (neighbor_coords[:, 1] >= 0) & (neighbor_coords[:, 1] < nx) &
                (neighbor_coords[:, 2] >= 0) & (neighbor_coords[:, 2] < ny) &
                (neighbor_coords[:, 3] >= 0) & (neighbor_coords[:, 3] < nz)
            )
            if not torch.any(valid):
                continue

            query_keys = self._encode_coords(neighbor_coords[valid])
            pos = torch.searchsorted(sorted_keys, query_keys)
            found = (pos < sorted_keys.numel()) & (sorted_keys[pos.clamp(max=sorted_keys.numel() - 1)] == query_keys)
            if not torch.any(found):
                continue

            valid_indices = valid.nonzero(as_tuple=False).squeeze(1)[found]
            source_feats = sorted_feats[pos[found]]
            neighbor_sum[valid_indices] = neighbor_sum[valid_indices] + source_feats
            neighbor_count[valid_indices] = neighbor_count[valid_indices] + 1.0

        neighbor_mean = neighbor_sum / neighbor_count.clamp_min(1.0)
        out = self.self_proj(feats) + self.neighbor_proj(neighbor_mean)
        return self.norm(feats + self.act(out))


class OccTransformerLayer(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        grid_size: Tuple[int, int, int],
        num_levels: int,
        num_heads: int,
        num_points: int,
        attn_chunk_size: int,
        ffn_ratio: float = 4.0,
    ) -> None:
        super().__init__()
        self.cross_attn = MSDeformAttn(
            embed_dim=feature_dim,
            num_heads=num_heads,
            num_levels=num_levels,
            num_points=num_points,
            chunk_size=attn_chunk_size,
        )
        self.cross_norm = nn.LayerNorm(feature_dim)
        self.sparse_conv = SparseLocalConv3d(feature_dim, grid_size)
        hidden_dim = int(feature_dim * ffn_ratio)
        self.ffn = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, feature_dim),
        )
        self.ffn_norm = nn.LayerNorm(feature_dim)

    def forward(
        self,
        query: torch.Tensor,
        coords_with_batch: torch.Tensor,
        multi_level_feats: List[torch.Tensor],
        reference_points: torch.Tensor,
        batch_idx: torch.Tensor,
        camera_mask: torch.Tensor,
    ) -> torch.Tensor:
        query = self.cross_attn(
            query=query,
            multi_level_feats=multi_level_feats,
            reference_points=reference_points,
            batch_idx=batch_idx,
            camera_mask=camera_mask,
        )
        query = self.cross_norm(query)
        query = self.sparse_conv(query, coords_with_batch)
        query = self.ffn_norm(query + self.ffn(query))
        return query


class MSDeformAttn(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        num_levels: int = 4,
        num_points: int = 4,
        chunk_size: int = 2048,
    ) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})")

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.num_points = num_points
        self.chunk_size = chunk_size
        self.head_dim = embed_dim // num_heads

        self.sampling_offsets = nn.Linear(embed_dim, num_heads * num_levels * num_points * 2)
        self.attention_weights = nn.Linear(embed_dim, num_heads * num_levels * num_points)
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        self.output_proj = nn.Linear(embed_dim, embed_dim)
        self.im2col_step = max(1, min(64, chunk_size))

    @staticmethod
    def _bilinear_sample_single_map(feat_map: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
        feat_map = feat_map.contiguous()
        c, h, w = feat_map.shape
        x = grid[..., 0] * max(w - 1, 1)
        y = grid[..., 1] * max(h - 1, 1)

        x0 = torch.floor(x).long()
        y0 = torch.floor(y).long()
        x1 = (x0 + 1).clamp(max=w - 1)
        y1 = (y0 + 1).clamp(max=h - 1)
        x0 = x0.clamp(0, w - 1)
        y0 = y0.clamp(0, h - 1)

        flat = feat_map.view(c, h * w)

        def gather(ix: torch.Tensor, iy: torch.Tensor) -> torch.Tensor:
            linear_idx = (iy * w + ix).reshape(-1)
            gathered = flat[:, linear_idx].transpose(0, 1).contiguous()
            return gathered.view(*ix.shape, c)

        feat00 = gather(x0, y0)
        feat01 = gather(x0, y1)
        feat10 = gather(x1, y0)
        feat11 = gather(x1, y1)

        wx = (x - x0.to(dtype=x.dtype)).unsqueeze(-1)
        wy = (y - y0.to(dtype=y.dtype)).unsqueeze(-1)
        w00 = (1.0 - wx) * (1.0 - wy)
        w01 = (1.0 - wx) * wy
        w10 = wx * (1.0 - wy)
        w11 = wx * wy

        sampled = feat00 * w00 + feat01 * w01 + feat10 * w10 + feat11 * w11
        valid = (
            (grid[..., 0] >= 0.0)
            & (grid[..., 0] <= 1.0)
            & (grid[..., 1] >= 0.0)
            & (grid[..., 1] <= 1.0)
        )
        return sampled * valid.unsqueeze(-1).to(dtype=sampled.dtype)

    def _sample_feature_map_per_head(self, feat_map: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
        sampled = []
        for head_id in range(self.num_heads):
            sampled.append(self._bilinear_sample_single_map(feat_map[head_id], grid[:, head_id]))
        return torch.stack(sampled, dim=1)

    @staticmethod
    def _level_start_index(spatial_shapes: torch.Tensor) -> torch.Tensor:
        level_sizes = spatial_shapes[:, 0] * spatial_shapes[:, 1]
        return torch.cat([level_sizes.new_zeros(1), level_sizes.cumsum(0)[:-1]], dim=0)

    def _forward_mmcv_cuda(
        self,
        query: torch.Tensor,
        projected_feats: List[torch.Tensor],
        reference_points: torch.Tensor,
        batch_idx: torch.Tensor,
        sampling_offsets: torch.Tensor,
        attention_weights: torch.Tensor,
        camera_weights: torch.Tensor,
    ) -> torch.Tensor:
        outputs = torch.zeros_like(query)
        num_views = reference_points.shape[1]

        for batch_id in batch_idx.unique(sorted=True).tolist():
            batch_mask = batch_idx == batch_id
            if not batch_mask.any():
                continue

            query_batch = query[batch_mask]
            ref_batch = reference_points[batch_mask]
            offsets_batch = sampling_offsets[batch_mask]
            attn_batch = attention_weights[batch_mask]
            camera_weights_batch = camera_weights[batch_mask]

            value_chunks = []
            spatial_shapes = []
            for feat in projected_feats:
                _, _, _, feat_h, feat_w = feat.shape
                for view_id in range(num_views):
                    feat_view = feat[batch_id, view_id].permute(1, 2, 0).reshape(-1, self.embed_dim)
                    value_chunks.append(feat_view)
                    spatial_shapes.append((feat_h, feat_w))

            value = torch.cat(value_chunks, dim=0).unsqueeze(0)
            value = value.view(1, value.shape[1], self.num_heads, self.head_dim)
            spatial_shapes = torch.tensor(spatial_shapes, device=query.device, dtype=torch.long)
            level_start_index = self._level_start_index(spatial_shapes)
            offset_normalizer = torch.stack([spatial_shapes[:, 1], spatial_shapes[:, 0]], dim=-1).to(
                device=query.device, dtype=query.dtype
            )

            ref_batch = ref_batch.unsqueeze(1).expand(-1, self.num_levels, -1, -1).reshape(
                ref_batch.shape[0], self.num_levels * num_views, 2
            )
            offsets_batch = offsets_batch.unsqueeze(4).expand(-1, -1, -1, -1, num_views, -1)
            offsets_batch = offsets_batch.reshape(
                offsets_batch.shape[0],
                self.num_heads,
                self.num_levels * num_views,
                self.num_points,
                2,
            )
            sampling_locations = ref_batch[:, None, :, None, :] + offsets_batch / offset_normalizer[None, None, :, None, :]

            attention_weights_batch = attn_batch.unsqueeze(-1) * camera_weights_batch[:, None, None, None, :]
            attention_weights_batch = attention_weights_batch.permute(0, 1, 2, 4, 3).reshape(
                attn_batch.shape[0],
                self.num_heads,
                self.num_levels * num_views,
                self.num_points,
            )

            output = MultiScaleDeformableAttnFunction.apply(
                value,
                spatial_shapes,
                level_start_index,
                sampling_locations.unsqueeze(0),
                attention_weights_batch.unsqueeze(0),
                torch.tensor(self.im2col_step, device=query.device, dtype=torch.int32),
            ).squeeze(0)
            outputs[batch_mask] = self.output_proj(output) + query_batch

        return outputs

    def _forward_pytorch(
        self,
        query: torch.Tensor,
        projected_feats: List[torch.Tensor],
        reference_points: torch.Tensor,
        batch_idx: torch.Tensor,
        camera_mask: torch.Tensor,
        sampling_offsets: torch.Tensor,
        attention_weights: torch.Tensor,
        camera_weights: torch.Tensor,
    ) -> torch.Tensor:
        dtype = query.dtype
        device = query.device
        dtype = query.dtype
        num_queries = query.shape[0]
        num_views = camera_mask.shape[1]

        outputs: List[torch.Tensor] = []
        view_chunk_size = 8
        for start in range(0, num_queries, self.chunk_size):
            end = min(start + self.chunk_size, num_queries)
            batch_chunk = batch_idx[start:end]
            ref_chunk = reference_points[start:end]
            offsets_chunk = sampling_offsets[start:end]
            attn_chunk = attention_weights[start:end]
            cam_mask_chunk = camera_mask[start:end]
            cam_weight_chunk = camera_weights[start:end]

            chunk_output = torch.zeros(
                end - start,
                self.num_heads,
                self.head_dim,
                device=device,
                dtype=dtype,
            )

            for level_id, feat in enumerate(projected_feats):
                _, _, _, feat_h, feat_w = feat.shape
                offset_normalizer = offsets_chunk.new_tensor([feat_w, feat_h]).view(1, 1, 1, 2)
                ref_level = ref_chunk[:, :, None, None, :] + offsets_chunk[:, None, :, level_id] / offset_normalizer
                ref_level = ref_level.clamp(0.0, 1.0)

                for view_id in range(num_views):
                    valid = cam_mask_chunk[:, view_id]
                    if not valid.any():
                        continue

                    valid_indices = valid.nonzero(as_tuple=False).squeeze(-1)
                    for sub_start in range(0, valid_indices.numel(), view_chunk_size):
                        sub_idx = valid_indices[sub_start : sub_start + view_chunk_size]
                        batch_ids = batch_chunk[sub_idx]
                        grid = ref_level[sub_idx, view_id]
                        unique_batch_ids = batch_ids.unique(sorted=True)
                        for batch_id in unique_batch_ids.tolist():
                            batch_mask = batch_ids == batch_id
                            feat_view = feat[batch_id, view_id].view(self.num_heads, self.head_dim, feat_h, feat_w)
                            sampled = self._sample_feature_map_per_head(feat_view, grid[batch_mask])

                            target_idx = sub_idx[batch_mask]
                            view_weight = cam_weight_chunk[target_idx, view_id].view(-1, 1, 1, 1)
                            sampled = sampled * attn_chunk[target_idx, :, level_id].unsqueeze(-1) * view_weight
                            chunk_output[target_idx] = chunk_output[target_idx] + sampled.sum(dim=2)

            outputs.append(chunk_output.reshape(end - start, self.embed_dim))

        output = torch.cat(outputs, dim=0)
        return self.output_proj(output) + query

    def forward(
        self,
        query: torch.Tensor,
        multi_level_feats: List[torch.Tensor],
        reference_points: torch.Tensor,
        batch_idx: torch.Tensor,
        camera_mask: torch.Tensor,
    ) -> torch.Tensor:
        if query.numel() == 0:
            return query
        if len(multi_level_feats) != self.num_levels:
            raise ValueError(f"Expected {self.num_levels} feature levels, got {len(multi_level_feats)}")

        dtype = query.dtype
        num_queries = query.shape[0]

        sampling_offsets = self.sampling_offsets(query).view(
            num_queries,
            self.num_heads,
            self.num_levels,
            self.num_points,
            2,
        )
        sampling_offsets = torch.tanh(sampling_offsets)
        attention_weights = self.attention_weights(query).view(
            num_queries,
            self.num_heads,
            self.num_levels * self.num_points,
        )
        attention_weights = F.softmax(attention_weights, dim=-1).view(
            num_queries,
            self.num_heads,
            self.num_levels,
            self.num_points,
        )

        camera_weights = camera_mask.to(dtype=dtype)
        camera_weights = camera_weights / camera_weights.sum(dim=1, keepdim=True).clamp_min(1.0)

        projected_feats = []
        for feat in multi_level_feats:
            feat_proj = self.value_proj(feat.permute(0, 1, 3, 4, 2)).permute(0, 1, 4, 2, 3).contiguous()
            projected_feats.append(feat_proj)

        if HAS_MMCV_MS_DEFORM_ATTN and query.is_cuda:
            return self._forward_mmcv_cuda(
                query=query,
                projected_feats=projected_feats,
                reference_points=reference_points,
                batch_idx=batch_idx,
                sampling_offsets=sampling_offsets,
                attention_weights=attention_weights,
                camera_weights=camera_weights,
            )
        return self._forward_pytorch(
            query=query,
            projected_feats=projected_feats,
            reference_points=reference_points,
            batch_idx=batch_idx,
            camera_mask=camera_mask,
            sampling_offsets=sampling_offsets,
            attention_weights=attention_weights,
            camera_weights=camera_weights,
        )


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
        feature_source: str = "depth_lift",
        occ_query_source: str = "occupied",
        occ_num_levels: int = 4,
        occ_num_heads: int = 8,
        occ_num_points: int = 4,
        occ_attn_chunk_size: int = 2048,
        occ_transformer_layers: int = 3,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.voxel_size = voxel_size
        self.x_range = x_range
        self.y_range = y_range
        self.z_range = z_range
        self.gaussians_per_voxel = gaussians_per_voxel
        self.feature_source = feature_source
        self.occ_query_source = occ_query_source
        self.occ_num_levels = occ_num_levels
        self.occ_transformer_layers = max(1, int(occ_transformer_layers))
        self.grid_size = (
            int((self.x_range[1] - self.x_range[0]) / self.voxel_size),
            int((self.y_range[1] - self.y_range[0]) / self.voxel_size),
            int((self.z_range[1] - self.z_range[0]) / self.voxel_size),
        )

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
        self.occ_query_embed = nn.Sequential(
            nn.Linear(3, feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim, feature_dim),
        )
        self.occ_transformer = nn.ModuleList(
            [
                OccTransformerLayer(
                    feature_dim=feature_dim,
                    grid_size=self.grid_size,
                    num_levels=occ_num_levels,
                    num_heads=occ_num_heads,
                    num_points=occ_num_points,
                    attn_chunk_size=occ_attn_chunk_size,
                )
                for _ in range(self.occ_transformer_layers)
            ]
        )
        self.decoder = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim, self.raw_gs_dim * self.gaussians_per_voxel),
        )
        self._init_gaussian_decoder()

    @staticmethod
    def _softplus_inverse(value: float) -> float:
        value_t = torch.tensor(float(value), dtype=torch.float32)
        return torch.log(torch.expm1(value_t)).item()

    def _init_gaussian_decoder(self) -> None:
        final_layer = self.decoder[-1]
        if not isinstance(final_layer, nn.Linear):
            return

        nn.init.normal_(final_layer.weight, mean=0.0, std=1e-4)
        nn.init.zeros_(final_layer.bias)

        with torch.no_grad():
            bias = final_layer.bias.view(self.gaussians_per_voxel, self.raw_gs_dim)
            max_offset = self.voxel_size * 0.5
            offset_init = torch.empty(
                self.gaussians_per_voxel,
                3,
                dtype=bias.dtype,
                device=bias.device,
            ).uniform_(-0.8 * max_offset, 0.8 * max_offset)
            offset_raw = torch.atanh((offset_init / max_offset).clamp(-0.99, 0.99))
            bias[:, 0:3] = offset_raw

            bias[:, 3:7] = bias.new_tensor([1.0, 0.0, 0.0, 0.0])

            scale_raw = self._softplus_inverse(0.1 / 0.01)
            bias[:, 7:10] = scale_raw

            bias[:, self.opacity_index] = 0.0
            bias[:, self.opacity_index + 1:] = 0.0

    def forward(
        self,
        aggregated_tokens_list: List[torch.Tensor],
        images: torch.Tensor,
        patch_start_idx: int,
        depth_maps: torch.Tensor,
        intrinsics: torch.Tensor,
        extrinsics: torch.Tensor,
        occ_inputs: Optional[Dict[str, torch.Tensor]] = None,
        feature_source: Optional[str] = None,
    ) -> torch.Tensor:
        feature_source = feature_source or self.feature_source
        if feature_source == "occ_gt":
            return self.forward_occ(
                aggregated_tokens_list=aggregated_tokens_list,
                images=images,
                patch_start_idx=patch_start_idx,
                intrinsics=intrinsics,
                extrinsics=extrinsics,
                occ_inputs=occ_inputs,
            )

        voxel_feats, unique_coords, inv, valid_idx, depth, shape, num_voxels = self.extract_voxel_features(
            aggregated_tokens_list=aggregated_tokens_list,
            images=images,
            patch_start_idx=patch_start_idx,
            depth_maps=depth_maps,
            intrinsics=intrinsics,
            extrinsics=extrinsics,
        )

        if voxel_feats is None:
            b, s, h, w = shape
            device = depth.device
            raw_full = torch.zeros(
                b * s * h * w, self.gaussians_per_voxel, self.raw_gs_dim, device=device, dtype=depth.dtype
            )
            raw_full[:, :, self.opacity_index] = self.invalid_opacity
            return raw_full.view(b, s, h, w, self.gaussians_per_voxel, self.raw_gs_dim)

        voxel_feats = self.refiner(voxel_feats, unique_coords)
        voxel_params = self.decoder(voxel_feats).view(num_voxels, self.gaussians_per_voxel, self.raw_gs_dim)

        b, s, h, w = shape
        device = depth.device
        raw_full = torch.zeros(
            b * s * h * w, self.gaussians_per_voxel, self.raw_gs_dim, device=device, dtype=depth.dtype
        )
        raw_full[:, :, self.opacity_index] = self.invalid_opacity
        raw_full[valid_idx] = voxel_params[inv]
        raw_full_reshaped = raw_full.view(b, s, h, w, self.gaussians_per_voxel, self.raw_gs_dim)
        return raw_full_reshaped

    def forward_occ(
        self,
        aggregated_tokens_list: List[torch.Tensor],
        images: torch.Tensor,
        patch_start_idx: int,
        intrinsics: torch.Tensor,
        extrinsics: torch.Tensor,
        occ_inputs: Optional[Dict[str, torch.Tensor]],
    ) -> Dict[str, torch.Tensor]:
        if occ_inputs is None:
            raise ValueError("occ_inputs must be provided when feature_source='occ_gt'")

        multi_level_feats = self.build_multilevel_features(
            aggregated_tokens_list=aggregated_tokens_list,
            images=images,
            patch_start_idx=patch_start_idx,
        )
        voxel_meta = self.build_occ_queries(
            occ_inputs=occ_inputs,
            device=multi_level_feats[0].device,
            dtype=multi_level_feats[0].dtype,
        )

        voxel_centers = voxel_meta["voxel_centers"]
        batch_idx = voxel_meta["batch_idx"]
        voxel_coords = voxel_meta["voxel_coords"]
        if voxel_centers.numel() == 0:
            raw_sparse = voxel_centers.new_zeros(0, self.gaussians_per_voxel, self.raw_gs_dim)
            return {
                "sparse_gaussians": True,
                "raw_sparse_gaussians": raw_sparse,
                "voxel_meta": voxel_meta,
            }

        query_coords = self.normalize_voxel_centers(voxel_centers)
        query = self.occ_query_embed(query_coords)
        feature_h, feature_w = multi_level_feats[0].shape[-2:]
        reference_points, camera_mask = self.project_voxels_to_cameras(
            voxel_centers=voxel_centers,
            batch_idx=batch_idx,
            intrinsics=intrinsics,
            extrinsics=extrinsics,
            feature_h=feature_h,
            feature_w=feature_w,
        )

        coords_with_batch = torch.cat([batch_idx.unsqueeze(-1), voxel_coords], dim=-1)
        voxel_feats = query
        for layer in self.occ_transformer:
            voxel_feats = layer(
                query=voxel_feats,
                coords_with_batch=coords_with_batch,
                multi_level_feats=multi_level_feats,
                reference_points=reference_points,
                batch_idx=batch_idx,
                camera_mask=camera_mask,
            )
        voxel_feats = self.refiner(voxel_feats, coords_with_batch)
        raw_sparse = self.decoder(voxel_feats).view(-1, self.gaussians_per_voxel, self.raw_gs_dim)
        return {
            "sparse_gaussians": True,
            "raw_sparse_gaussians": raw_sparse,
            "voxel_meta": voxel_meta,
        }

    def extract_voxel_features(
        self,
        aggregated_tokens_list: List[torch.Tensor],
        images: torch.Tensor,
        patch_start_idx: int,
        depth_maps: torch.Tensor,
        intrinsics: torch.Tensor,
        extrinsics: torch.Tensor,
    ) -> Tuple[
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        torch.Tensor,
        Tuple[int, int, int, int],
        Optional[int],
    ]:
        # 2D feature map from tokens: [B, S, C, H, W]
        features = self.build_multilevel_features(
            aggregated_tokens_list,
            images=images,
            patch_start_idx=patch_start_idx,
        )[0]
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
        voxel_coords = torch.floor(
            (points_flat - points_flat.new_tensor([self.x_range[0], self.y_range[0], self.z_range[0]]))
            / self.voxel_size
        ).long()

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
            return None, None, None, None, depth, (b, s, h, w), None

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
        return voxel_feats, unique_coords, inv, valid_idx, depth, (b, s, h, w), num_voxels

    def build_multilevel_features(
        self,
        aggregated_tokens_list: List[torch.Tensor],
        images: torch.Tensor,
        patch_start_idx: int,
    ) -> List[torch.Tensor]:
        features = self.feature_head(
            aggregated_tokens_list,
            images=images,
            patch_start_idx=patch_start_idx,
        )
        multi_level_feats = [features]
        pooled = features.view(-1, features.shape[2], features.shape[3], features.shape[4])
        for _ in range(1, self.occ_num_levels):
            if pooled.shape[-2] < 2 or pooled.shape[-1] < 2:
                break
            pooled = F.avg_pool2d(pooled, kernel_size=2, stride=2)
            multi_level_feats.append(
                pooled.view(features.shape[0], features.shape[1], pooled.shape[1], pooled.shape[2], pooled.shape[3])
            )
        if len(multi_level_feats) < self.occ_num_levels:
            last_feat = multi_level_feats[-1]
            while len(multi_level_feats) < self.occ_num_levels:
                multi_level_feats.append(last_feat)
        return multi_level_feats

    def build_occ_queries(
        self,
        occ_inputs: Dict[str, torch.Tensor],
        device: torch.device,
        dtype: torch.dtype,
    ) -> Dict[str, torch.Tensor]:
        occ_mask = self.get_occ_query_mask(occ_inputs, device=device)
        if occ_mask is None:
            raise ValueError("Expected OCC labels in occ_inputs for feature_source='occ_gt'")
        if occ_mask.dim() == 3:
            occ_mask = occ_mask.unsqueeze(0)

        batch_indices: List[torch.Tensor] = []
        voxel_coords_list: List[torch.Tensor] = []
        for batch_id in range(occ_mask.shape[0]):
            coords = occ_mask[batch_id].nonzero(as_tuple=False)
            if coords.numel() == 0:
                continue
            voxel_coords_list.append(coords.long())
            batch_indices.append(torch.full((coords.shape[0],), batch_id, device=device, dtype=torch.long))

        if not voxel_coords_list:
            empty_long = torch.empty(0, device=device, dtype=torch.long)
            empty_coords = torch.empty(0, 3, device=device, dtype=torch.long)
            empty_float = torch.empty(0, 3, device=device, dtype=dtype)
            return {
                "batch_idx": empty_long,
                "voxel_coords": empty_coords,
                "voxel_centers": empty_float,
            }

        voxel_coords = torch.cat(voxel_coords_list, dim=0)
        batch_idx = torch.cat(batch_indices, dim=0)
        voxel_centers = self.voxel_coords_to_world(voxel_coords, dtype=dtype)
        return {
            "batch_idx": batch_idx,
            "voxel_coords": voxel_coords,
            "voxel_centers": voxel_centers,
        }

    def get_occ_query_mask(
        self,
        occ_inputs: Dict[str, torch.Tensor],
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        occ_surface = occ_inputs.get("occ_surface")
        occ_semantics = occ_inputs.get("occ_semantics")
        occ_visible_mask = occ_inputs.get("occ_visible_mask")

        if self.occ_query_source in {"occupied", "semantics"} and occ_semantics is not None:
            occ_mask = occ_semantics != 17
        elif self.occ_query_source == "visible" and occ_visible_mask is not None:
            occ_mask = occ_visible_mask > 0
        elif occ_surface is not None:
            occ_mask = occ_surface > 0
        elif occ_semantics is not None:
            occ_mask = occ_semantics != 17
        elif occ_visible_mask is not None:
            occ_mask = occ_visible_mask > 0
        else:
            return None
        return occ_mask.to(device=device)

    def voxel_coords_to_world(self, voxel_coords: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        voxel_origin = voxel_coords.new_tensor([self.x_range[0], self.y_range[0], self.z_range[0]], dtype=dtype)
        return voxel_origin + (voxel_coords.to(dtype=dtype) + 0.5) * self.voxel_size

    def normalize_voxel_centers(self, voxel_centers: torch.Tensor) -> torch.Tensor:
        lower = voxel_centers.new_tensor([self.x_range[0], self.y_range[0], self.z_range[0]])
        upper = voxel_centers.new_tensor([self.x_range[1], self.y_range[1], self.z_range[1]])
        return ((voxel_centers - lower) / (upper - lower).clamp_min(1e-6)) * 2.0 - 1.0

    def project_voxels_to_cameras(
        self,
        voxel_centers: torch.Tensor,
        batch_idx: torch.Tensor,
        intrinsics: torch.Tensor,
        extrinsics: torch.Tensor,
        feature_h: int,
        feature_w: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        intrinsics_4x4 = self._ensure_4x4(intrinsics, device=voxel_centers.device, dtype=voxel_centers.dtype)
        extrinsics_4x4 = self._ensure_4x4(extrinsics, device=voxel_centers.device, dtype=voxel_centers.dtype)
        e2c = torch.linalg.inv(extrinsics_4x4[batch_idx])
        k = intrinsics_4x4[batch_idx]

        homogeneous = torch.cat([voxel_centers, torch.ones_like(voxel_centers[:, :1])], dim=-1)
        homogeneous = homogeneous[:, None, :, None]
        cam_points = torch.matmul(e2c, homogeneous).squeeze(-1)
        proj = torch.matmul(k[..., :3, :3], cam_points[..., :3].unsqueeze(-1)).squeeze(-1)

        depth = proj[..., 2]
        uv = proj[..., :2] / depth.clamp_min(1e-6).unsqueeze(-1)
        reference_points = torch.stack(
            [
                uv[..., 0] / max(feature_w - 1, 1),
                uv[..., 1] / max(feature_h - 1, 1),
            ],
            dim=-1,
        )
        camera_mask = (
            (depth > 1e-4)
            & (reference_points[..., 0] >= 0.0)
            & (reference_points[..., 0] <= 1.0)
            & (reference_points[..., 1] >= 0.0)
            & (reference_points[..., 1] <= 1.0)
        )
        return reference_points, camera_mask

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

    def _voxel_grid_size(self) -> Tuple[int, int, int]:
        nx = int((self.x_range[1] - self.x_range[0]) / self.voxel_size)
        ny = int((self.y_range[1] - self.y_range[0]) / self.voxel_size)
        nz = int((self.z_range[1] - self.z_range[0]) / self.voxel_size)
        return nx, ny, nz