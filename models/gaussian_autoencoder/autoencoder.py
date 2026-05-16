import torch
import torch.nn as nn

from .voxelizer import Voxelizer
from .encoding_head import EncodingHead, DecodingHead
from .sparse_cnn import SparseEncoder, SparseDecoder, SparseTensor
from .losses import GaussianAELoss


class GaussianAutoencoder(nn.Module):
    """
    3D Gaussian Autoencoder following the L3DG sparse 3D CNN design.

    Pipeline:
        recontrast_data
            → Voxelizer.voxelize()          (sub-sample: 1 Gaussian/voxel)
            → EncodingHead                  ([M, 86] → [M, 32])
            → SparseConvTensor
            → SparseEncoder                 (3× downsample, latent [M3, 256])
            → SparseDecoder                 (3× upsample + skip, [M, 32])
            → DecodingHead                  ([M, 32] → [M, K, 86])
            → Voxelizer.devoxelize()        (recontrast_data with K Gaussians/voxel)
    """

    def __init__(self, cfg: dict):
        super().__init__()

        voxel_size = cfg.get('voxel_size', 0.4)
        x_range    = cfg.get('x_range',    [-40, 40])
        y_range    = cfg.get('y_range',    [-40, 40])
        z_range    = cfg.get('z_range',    [-1, 5.4])
        K          = cfg.get('K', 4)
        channels   = tuple(cfg.get('encoder_channels', [32, 64, 128, 256]))

        self.voxelizer    = Voxelizer(voxel_size, x_range, y_range, z_range)
        self.enc_head     = EncodingHead(in_dim=86, hidden_dims=(128, 64), out_dim=channels[0])
        self.encoder      = SparseEncoder(channels=channels)
        self.decoder      = SparseDecoder(channels=channels)
        self.dec_head     = DecodingHead(in_dim=channels[0], hidden_dims=(128, 256),
                                         K=K, gauss_dim=86)

        self.loss_fn = GaussianAELoss(
            lambda_xyz     = cfg.get('lambda_xyz',     1.0),
            lambda_rot     = cfg.get('lambda_rot',     0.5),
            lambda_scale   = cfg.get('lambda_scale',   0.5),
            lambda_opacity = cfg.get('lambda_opacity', 0.5),
            lambda_sh      = cfg.get('lambda_sh',      0.1),
        )

    # ------------------------------------------------------------------

    def _build_gt_target(self, voxel_features, voxel_indices, voxel_centers, batch_size):
        """
        Build an AE training target aligned with decoder output count.

        We tile each 1-voxel GT Gaussian to K copies so the reconstructed
        Gaussian tensor and GT tensor have matching shapes for attribute loss.
        """
        K = self.dec_head.K
        gt_features_tiled = voxel_features.unsqueeze(1).expand(-1, K, -1).contiguous()
        return self.voxelizer.devoxelize(
            gt_features_tiled, voxel_indices, voxel_centers, batch_size
        )

    def encode(self, recontrast_data):
        """
        Returns:
            latent          SparseTensor
            skip1/2/3       SparseTensors (for decoder)
            voxel_indices   [M, 4]
            voxel_centers   [M, 3]
            batch_size      int
        """
        voxel_feat, voxel_idx, voxel_centers, B = \
            self.voxelizer.voxelize(recontrast_data)

        voxel_feat = self.enc_head(voxel_feat)   # [M, 32]

        # Convert voxel_idx (batch, z, y, x) → coords (batch, x, y, z)
        coords = torch.cat([
            voxel_idx[:, 0:1],  # batch
            voxel_idx[:, 3:4],  # x
            voxel_idx[:, 2:3],  # y
            voxel_idx[:, 1:2],  # z
        ], dim=1)

        sp_input = SparseTensor(features=voxel_feat, coordinates=coords)

        latent, skip1, skip2, skip3 = self.encoder(sp_input)

        return latent, skip1, skip2, skip3, voxel_idx, voxel_centers, B

    def decode(self, latent, skip1, skip2, skip3, voxel_indices, voxel_centers, batch_size):
        """
        Returns:
            recontrast_data dict (reconstructed)
        """
        decoded_sp = self.decoder(latent, skip1, skip2, skip3)  # SparseTensor [M, 32]
        decoded_feat = self.dec_head(decoded_sp.F)               # [M, K, 86]

        return self.voxelizer.devoxelize(decoded_feat, voxel_indices, voxel_centers, batch_size)

    def forward(self, recontrast_data):
        """
        Full encode → decode pass.

        Returns:
            recon_data   dict — reconstructed recontrast_data
            latent       SparseConvTensor — bottleneck representation
        """
        latent, skip1, skip2, skip3, vox_idx, vox_centers, B = \
            self.encode(recontrast_data)

        recon_data = self.decode(latent, skip1, skip2, skip3, vox_idx, vox_centers, B)

        return recon_data, latent

    def forward_with_targets(self, recontrast_data):
        """
        Forward pass that also returns an aligned GT target for attribute loss.
        """
        voxel_feat, voxel_idx, voxel_centers, B = self.voxelizer.voxelize(recontrast_data)

        voxel_feat_enc = self.enc_head(voxel_feat)
        coords = torch.cat(
            [voxel_idx[:, 0:1], voxel_idx[:, 3:4], voxel_idx[:, 2:3], voxel_idx[:, 1:2]],
            dim=1,
        )
        sp_input = SparseTensor(features=voxel_feat_enc, coordinates=coords)

        latent, skip1, skip2, skip3 = self.encoder(sp_input)
        decoded_sp = self.decoder(latent, skip1, skip2, skip3)
        decoded_feat = self.dec_head(decoded_sp.F)

        pred_recon = self.voxelizer.devoxelize(decoded_feat, voxel_idx, voxel_centers, B)
        gt_target = self._build_gt_target(voxel_feat, voxel_idx, voxel_centers, B)
        return pred_recon, latent, gt_target
