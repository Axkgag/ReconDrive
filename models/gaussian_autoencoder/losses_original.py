import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_mean


class GaussianAELoss(nn.Module):
    """
    Attribute reconstruction loss between decoded and original Gaussians.

    Simple element-wise L1 loss (used as fallback or for quick testing).
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


class GaussianChamferLoss(nn.Module):
    """
    Chamfer-based attribute loss for 3D Gaussian Autoencoder.

    For each voxel, matches predicted K Gaussians against ALL GT Gaussians
    in that voxel using nearest-neighbor on xyz, then computes attribute L1
    on the matched pairs.

    Chamfer = pred→gt (each pred finds its closest GT)
            + gt→pred (each GT finds its closest pred)

    This encourages K predictions to spread out and cover the GT distribution
    within each voxel, rather than collapsing to a single point.
    """

    def __init__(self,
                 lambda_xyz=1.0,
                 lambda_rot=0.5,
                 lambda_scale=0.5,
                 lambda_opacity=0.5,
                 lambda_sh=0.1,
                 chamfer_alpha=0.5):
        """
        Args:
            chamfer_alpha: weight balance between pred→gt and gt→pred.
                0.5 = symmetric Chamfer (default)
                1.0 = only pred→gt (each pred must be close to some GT)
                0.0 = only gt→pred (each GT must be covered by some pred)
        """
        super().__init__()
        self.lambda_xyz     = lambda_xyz
        self.lambda_rot     = lambda_rot
        self.lambda_scale   = lambda_scale
        self.lambda_opacity = lambda_opacity
        self.lambda_sh      = lambda_sh
        self.chamfer_alpha  = chamfer_alpha

    def forward(self, pred_raw, all_gt_features, all_gt_voxel_id, M, K):
        """
        Args:
            pred_raw:         [M, K, 86] — raw decoded features (pre-activation)
            all_gt_features:  [N_gt, 86] — all GT Gaussians (raw, xyz as offset)
            all_gt_voxel_id:  [N_gt]     — voxel assignment for each GT (0..M-1)
            M:                int        — number of occupied voxels
            K:                int        — Gaussians per voxel in decoder output

        Returns:
            loss_dict: {'total', 'xyz', 'rot', 'scale', 'opacity', 'sh'}
        """
        device = pred_raw.device

        # Slice indices for the 86-dim feature vector
        IDX_XYZ   = slice(0, 3)
        IDX_ROT   = slice(3, 7)
        IDX_SCALE = slice(7, 10)
        IDX_OPA   = slice(10, 11)
        IDX_SH    = slice(11, 86)

        # pred xyz offsets (apply tanh to match devoxelize behavior for matching)
        # We match on the activated xyz space so distances are meaningful.
        pred_xyz = pred_raw[..., IDX_XYZ]  # [M, K, 3], raw

        # GT xyz offsets are already in offset space (raw, no activation needed
        # since they were stored as absolute - center)
        gt_xyz = all_gt_features[:, IDX_XYZ]  # [N_gt, 3]

        # --- Per-voxel Chamfer matching ---
        # Strategy: for each voxel, compute pairwise distances between K preds
        # and all GTs in that voxel, then find nearest neighbors.
        #
        # For efficiency, we process all voxels in parallel using scatter ops.

        # Assign each pred a flat index: voxel_id * K + k
        pred_voxel_ids = torch.arange(M, device=device).unsqueeze(1).expand(M, K).reshape(-1)  # [M*K]
        pred_xyz_flat = pred_xyz.reshape(-1, 3)  # [M*K, 3]
        pred_raw_flat = pred_raw.reshape(-1, 86)  # [M*K, 86]

        # For each GT point, find the nearest pred in the same voxel
        # For each pred point, find the nearest GT in the same voxel
        # We do this voxel-by-voxel for correctness, but batch small voxels.

        # Build voxel start/end indices for GT
        gt_sorted_order = torch.argsort(all_gt_voxel_id)
        gt_sorted_voxel = all_gt_voxel_id[gt_sorted_order]
        gt_sorted_feat  = all_gt_features[gt_sorted_order]  # [N_gt, 86]
        gt_sorted_xyz   = gt_sorted_feat[:, IDX_XYZ]        # [N_gt, 3]

        # Count GT points per voxel
        gt_counts = torch.zeros(M, dtype=torch.long, device=device)
        gt_counts.scatter_add_(0, all_gt_voxel_id, torch.ones_like(all_gt_voxel_id, dtype=torch.long))
        gt_offsets = torch.zeros(M + 1, dtype=torch.long, device=device)
        gt_offsets[1:] = gt_counts.cumsum(0)

        # Accumulate losses
        loss_xyz_sum     = torch.tensor(0.0, device=device)
        loss_rot_sum     = torch.tensor(0.0, device=device)
        loss_scale_sum   = torch.tensor(0.0, device=device)
        loss_opacity_sum = torch.tensor(0.0, device=device)
        loss_sh_sum      = torch.tensor(0.0, device=device)
        total_pairs = 0

        # Process in chunks of voxels for memory efficiency
        CHUNK = 4096
        for chunk_start in range(0, M, CHUNK):
            chunk_end = min(chunk_start + CHUNK, M)

            for v in range(chunk_start, chunk_end):
                n_gt = gt_counts[v].item()
                if n_gt == 0:
                    continue

                # GT points for this voxel
                gt_start = gt_offsets[v].item()
                gt_end   = gt_offsets[v + 1].item()
                vgt_feat = gt_sorted_feat[gt_start:gt_end]  # [n_gt, 86]
                vgt_xyz  = gt_sorted_xyz[gt_start:gt_end]   # [n_gt, 3]

                # Pred points for this voxel
                vpred_feat = pred_raw_flat[v * K : (v + 1) * K]  # [K, 86]
                vpred_xyz  = pred_xyz_flat[v * K : (v + 1) * K]  # [K, 3]

                # Pairwise L2 distance on xyz: [K, n_gt]
                dist = torch.cdist(vpred_xyz.unsqueeze(0), vgt_xyz.unsqueeze(0)).squeeze(0)

                # pred→gt: for each pred, find nearest GT
                nn_gt_idx = dist.argmin(dim=1)  # [K]
                matched_gt = vgt_feat[nn_gt_idx]  # [K, 86]

                # gt→pred: for each GT, find nearest pred
                nn_pred_idx = dist.argmin(dim=0)  # [n_gt]
                matched_pred = vpred_feat[nn_pred_idx]  # [n_gt, 86]

                # pred→gt attribute loss
                p2g_xyz = (vpred_feat[:, IDX_XYZ] - matched_gt[:, IDX_XYZ]).abs().sum()
                p2g_rot = (vpred_feat[:, IDX_ROT] - matched_gt[:, IDX_ROT]).abs().sum()
                p2g_sca = (vpred_feat[:, IDX_SCALE] - matched_gt[:, IDX_SCALE]).abs().sum()
                p2g_opa = (vpred_feat[:, IDX_OPA] - matched_gt[:, IDX_OPA]).abs().sum()
                p2g_sh  = (vpred_feat[:, IDX_SH] - matched_gt[:, IDX_SH]).abs().sum()

                # gt→pred attribute loss
                g2p_xyz = (matched_pred[:, IDX_XYZ] - vgt_feat[:, IDX_XYZ]).abs().sum()
                g2p_rot = (matched_pred[:, IDX_ROT] - vgt_feat[:, IDX_ROT]).abs().sum()
                g2p_sca = (matched_pred[:, IDX_SCALE] - vgt_feat[:, IDX_SCALE]).abs().sum()
                g2p_opa = (matched_pred[:, IDX_OPA] - vgt_feat[:, IDX_OPA]).abs().sum()
                g2p_sh  = (matched_pred[:, IDX_SH] - vgt_feat[:, IDX_SH]).abs().sum()

                alpha = self.chamfer_alpha
                n_pairs = K + n_gt

                loss_xyz_sum     += alpha * p2g_xyz + (1 - alpha) * g2p_xyz
                loss_rot_sum     += alpha * p2g_rot + (1 - alpha) * g2p_rot
                loss_scale_sum   += alpha * p2g_sca + (1 - alpha) * g2p_sca
                loss_opacity_sum += alpha * p2g_opa + (1 - alpha) * g2p_opa
                loss_sh_sum      += alpha * p2g_sh  + (1 - alpha) * g2p_sh
                total_pairs += n_pairs

        # Normalize
        if total_pairs == 0:
            total_pairs = 1

        loss_xyz     = loss_xyz_sum / total_pairs
        loss_rot     = loss_rot_sum / total_pairs
        loss_scale   = loss_scale_sum / total_pairs
        loss_opacity = loss_opacity_sum / total_pairs
        loss_sh      = loss_sh_sum / total_pairs

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
