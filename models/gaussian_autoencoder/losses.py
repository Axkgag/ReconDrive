import torch
import torch.nn as nn
import torch.nn.functional as F


class GaussianAELoss(nn.Module):
    """
    Attribute reconstruction loss between decoded and original Gaussians.

    Operates on the voxelized (sub-sampled) ground-truth features so that
    the comparison is 1-to-1 with the encoder input.
    """

    def __init__(self,
                 lambda_xyz=1.0,
                 lambda_rot=0.5,
                 lambda_scale=0.5,
                 lambda_opacity=0.5,
                 lambda_sh=0.1):
        super().__init__()
        self.lambda_xyz     = lambda_xyz
        self.lambda_rot     = lambda_rot
        self.lambda_scale   = lambda_scale
        self.lambda_opacity = lambda_opacity
        self.lambda_sh      = lambda_sh

    def forward(self, pred_recon, gt_recon):
        """
        Args:
            pred_recon: recontrast_data dict from devoxelize()
                xyz          [B, N, 3]
                rot_maps     [B, N, 4]
                scale_maps   [B, N, 3]
                opacity_maps [B, N, 1]
                sh_maps      [B, N, 25, 3]
            gt_recon: same structure (original recontrast_data, padded to same N)

        Returns:
            loss_dict: {'total', 'xyz', 'rot', 'scale', 'opacity', 'sh'}
        """
        loss_xyz     = F.l1_loss(pred_recon['xyz'],          gt_recon['xyz'])
        loss_rot     = F.l1_loss(pred_recon['rot_maps'],     gt_recon['rot_maps'])
        loss_scale   = F.l1_loss(pred_recon['scale_maps'],   gt_recon['scale_maps'])
        loss_opacity = F.l1_loss(pred_recon['opacity_maps'], gt_recon['opacity_maps'])
        loss_sh      = F.l1_loss(pred_recon['sh_maps'],      gt_recon['sh_maps'])

        total = (self.lambda_xyz     * loss_xyz     +
                 self.lambda_rot     * loss_rot     +
                 self.lambda_scale   * loss_scale   +
                 self.lambda_opacity * loss_opacity +
                 self.lambda_sh      * loss_sh)

        return {
            'total':   total,
            'xyz':     loss_xyz,
            'rot':     loss_rot,
            'scale':   loss_scale,
            'opacity': loss_opacity,
            'sh':      loss_sh,
        }
