import torch
import torch.nn.functional as F
from torch_scatter import scatter_max, scatter_mean


class Voxelizer:
    """
    Converts a set of 3D Gaussians into a sparse voxel grid and back.

    Coordinate convention (nuScenes ego frame):
        X = forward, Y = left, Z = up

    MinkowskiEngine uses (batch, x, y, z) coordinate ordering.
    Internal voxel_indices are stored as (batch, z, y, x) for compatibility.
    """

    # Gaussian feature vector layout (86 dims total)
    FEAT_DIM = 86
    IDX_XYZ   = slice(0, 3)    # relative offset from voxel center
    IDX_ROT   = slice(3, 7)    # quaternion [w,x,y,z], raw (normalized at decode)
    IDX_SCALE = slice(7, 10)   # raw (softplus*0.01 at decode)
    IDX_OPA   = slice(10, 11)  # raw logit (sigmoid at decode)
    IDX_SH    = slice(11, 86)  # 75 = 3 channels × 25 SH coeffs, raw

    def __init__(self, voxel_size=0.4,
                 x_range=(-40, 40),
                 y_range=(-40, 40),
                 z_range=(-1, 5.4)):
        self.voxel_size = voxel_size
        self.x_range = x_range
        self.y_range = y_range
        self.z_range = z_range

        self.nx = round((x_range[1] - x_range[0]) / voxel_size)
        self.ny = round((y_range[1] - y_range[0]) / voxel_size)
        self.nz = round((z_range[1] - z_range[0]) / voxel_size)

        # spconv spatial_shape is (D, H, W) = (Z, Y, X)
        self.spatial_shape = [self.nz, self.ny, self.nx]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def voxelize(self, recontrast_data):
        """
        Build a sparse voxel grid from recontrast_data.

        Args:
            recontrast_data: dict with keys
                'xyz'          [B, N, 3]
                'rot_maps'     [B, N, 4]
                'scale_maps'   [B, N, 3]
                'opacity_maps' [B, N, 1]
                'sh_maps'      [B, N, 25, 3]  (or [B, N, 3, 25])

        Returns:
            voxel_features  [M, 86]   — one Gaussian per occupied voxel
            voxel_indices   [M, 4]    — (batch, z, y, x) int32
            voxel_centers   [M, 3]    — absolute XYZ of each voxel center
            batch_size      int
        """
        xyz     = recontrast_data['xyz']           # [B, N, 3]
        rot     = recontrast_data['rot_maps']      # [B, N, 4]
        scale   = recontrast_data['scale_maps']    # [B, N, 3]
        opacity = recontrast_data['opacity_maps']  # [B, N, 1]
        sh      = recontrast_data['sh_maps']       # [B, N, 25, 3] or [B, N, 3, 25]

        B, N, _ = xyz.shape
        device = xyz.device

        # Normalise sh to [B, N, 75]
        # Input can be [B, N, 25, 3] or [B, N, 3, 25]
        if sh.dim() == 4:
            if sh.shape[-1] == 3:      # [B, N, 25, 3]
                sh_flat = sh.reshape(B, N, 75)
            else:                      # [B, N, 3, 25]
                sh_flat = sh.permute(0, 1, 3, 2).reshape(B, N, 75)
        else:
            sh_flat = sh.reshape(B, N, 75)

        # Build 86-dim raw feature vector (pre-activation)
        # xyz stored as raw absolute coords first; offset computed after voxel assignment
        raw_opacity = torch.logit(opacity.clamp(1e-6, 1 - 1e-6))  # inverse sigmoid
        raw_scale   = self._inv_softplus(scale / 0.01)             # inverse of softplus*0.01
        feat = torch.cat([xyz, rot, raw_scale, raw_opacity, sh_flat], dim=-1)  # [B, N, 86]

        # Flatten batch dimension
        batch_ids = torch.arange(B, device=device).unsqueeze(1).expand(B, N).reshape(-1)  # [B*N]
        xyz_flat  = xyz.reshape(-1, 3)   # [B*N, 3]
        feat_flat = feat.reshape(-1, 86) # [B*N, 86]
        opa_flat  = opacity.reshape(-1)  # [B*N]

        # Compute voxel indices (clip points outside grid)
        xi = ((xyz_flat[:, 0] - self.x_range[0]) / self.voxel_size).long()
        yi = ((xyz_flat[:, 1] - self.y_range[0]) / self.voxel_size).long()
        zi = ((xyz_flat[:, 2] - self.z_range[0]) / self.voxel_size).long()

        valid = (xi >= 0) & (xi < self.nx) & \
                (yi >= 0) & (yi < self.ny) & \
                (zi >= 0) & (zi < self.nz)

        xi, yi, zi = xi[valid], yi[valid], zi[valid]
        batch_ids_v = batch_ids[valid]
        feat_flat_v = feat_flat[valid]
        opa_flat_v  = opa_flat[valid]

        # Unique voxel key: batch * nz*ny*nx + z*ny*nx + y*nx + x
        stride_b = self.nz * self.ny * self.nx
        voxel_key = batch_ids_v * stride_b + zi * (self.ny * self.nx) + yi * self.nx + xi

        # Sub-sample: keep the Gaussian with highest opacity per voxel
        _, argmax = scatter_max(opa_flat_v, voxel_key, dim=0)

        # scatter_max fills unused slots with 0; filter to unique keys only
        unique_keys, inv = torch.unique(voxel_key, return_inverse=True)
        # argmax indexed by unique_keys position
        _, argmax_per_unique = scatter_max(opa_flat_v, inv, dim=0)
        # map back to original point indices
        point_indices = argmax_per_unique  # [M]

        voxel_features = feat_flat_v[point_indices]   # [M, 86]
        vox_xi = xi[point_indices]
        vox_yi = yi[point_indices]
        vox_zi = zi[point_indices]
        vox_b  = batch_ids_v[point_indices]

        # Compute voxel centers
        cx = (vox_xi.float() + 0.5) * self.voxel_size + self.x_range[0]
        cy = (vox_yi.float() + 0.5) * self.voxel_size + self.y_range[0]
        cz = (vox_zi.float() + 0.5) * self.voxel_size + self.z_range[0]
        voxel_centers = torch.stack([cx, cy, cz], dim=-1)  # [M, 3]

        # Replace absolute xyz with pre-tanh offset in feature vector
        # (inverse of tanh(x) * voxel_size/2, matching decoder output space)
        xyz_abs = voxel_features[:, self.IDX_XYZ]
        voxel_features = voxel_features.clone()
        xyz_offset = xyz_abs - voxel_centers
        voxel_features[:, self.IDX_XYZ] = torch.atanh(
            (xyz_offset / (self.voxel_size / 2)).clamp(-0.999, 0.999)
        )

        # Build spconv-style indices [M, 4] = (batch, z, y, x) as int32
        voxel_indices = torch.stack(
            [vox_b, vox_zi, vox_yi, vox_xi], dim=-1
        ).to(torch.int32)

        return voxel_features, voxel_indices, voxel_centers, B

    def voxelize_with_all_gt(self, recontrast_data):
        """
        Like voxelize(), but also returns ALL GT Gaussians per voxel (not just
        the max-opacity representative). Used for Chamfer loss computation.

        Returns:
            voxel_features  [M, 86]       — 1 representative Gaussian per voxel (for encoder)
            voxel_indices   [M, 4]        — (batch, z, y, x) int32
            voxel_centers   [M, 3]        — absolute XYZ of each voxel center
            batch_size      int
            all_gt_features [N_valid, 86] — ALL valid Gaussians (with xyz as offset)
            all_gt_voxel_id [N_valid]     — which voxel (0..M-1) each GT belongs to
        """
        xyz     = recontrast_data['xyz']           # [B, N, 3]
        rot     = recontrast_data['rot_maps']      # [B, N, 4]
        scale   = recontrast_data['scale_maps']    # [B, N, 3]
        opacity = recontrast_data['opacity_maps']  # [B, N, 1]
        sh      = recontrast_data['sh_maps']       # [B, N, 25, 3] or [B, N, 3, 25]

        B, N, _ = xyz.shape
        device = xyz.device

        # Normalise sh to [B, N, 75]
        if sh.dim() == 4:
            if sh.shape[-1] == 3:
                sh_flat = sh.reshape(B, N, 75)
            else:
                sh_flat = sh.permute(0, 1, 3, 2).reshape(B, N, 75)
        else:
            sh_flat = sh.reshape(B, N, 75)

        # Build 86-dim raw feature vector
        raw_opacity = torch.logit(opacity.clamp(1e-6, 1 - 1e-6))
        raw_scale   = self._inv_softplus(scale / 0.01)
        feat = torch.cat([xyz, rot, raw_scale, raw_opacity, sh_flat], dim=-1)  # [B, N, 86]

        # Flatten batch dimension
        batch_ids = torch.arange(B, device=device).unsqueeze(1).expand(B, N).reshape(-1)
        xyz_flat  = xyz.reshape(-1, 3)
        feat_flat = feat.reshape(-1, 86)
        opa_flat  = opacity.reshape(-1)

        # Compute voxel indices
        xi = ((xyz_flat[:, 0] - self.x_range[0]) / self.voxel_size).long()
        yi = ((xyz_flat[:, 1] - self.y_range[0]) / self.voxel_size).long()
        zi = ((xyz_flat[:, 2] - self.z_range[0]) / self.voxel_size).long()

        valid = (xi >= 0) & (xi < self.nx) & \
                (yi >= 0) & (yi < self.ny) & \
                (zi >= 0) & (zi < self.nz)

        xi, yi, zi = xi[valid], yi[valid], zi[valid]
        batch_ids_v = batch_ids[valid]
        feat_flat_v = feat_flat[valid]
        opa_flat_v  = opa_flat[valid]

        # Unique voxel key
        stride_b = self.nz * self.ny * self.nx
        voxel_key = batch_ids_v * stride_b + zi * (self.ny * self.nx) + yi * self.nx + xi

        # Get unique voxels and map each point to a voxel index 0..M-1
        unique_keys, inv_map = torch.unique(voxel_key, return_inverse=True)
        M = unique_keys.shape[0]

        # Sub-sample: keep the Gaussian with highest opacity per voxel (for encoder)
        _, argmax_per_voxel = scatter_max(opa_flat_v, inv_map, dim=0)
        point_indices = argmax_per_voxel  # [M]

        voxel_features = feat_flat_v[point_indices]   # [M, 86]
        vox_xi = xi[point_indices]
        vox_yi = yi[point_indices]
        vox_zi = zi[point_indices]
        vox_b  = batch_ids_v[point_indices]

        # Compute voxel centers
        cx = (vox_xi.float() + 0.5) * self.voxel_size + self.x_range[0]
        cy = (vox_yi.float() + 0.5) * self.voxel_size + self.y_range[0]
        cz = (vox_zi.float() + 0.5) * self.voxel_size + self.z_range[0]
        voxel_centers = torch.stack([cx, cy, cz], dim=-1)  # [M, 3]

        # Replace absolute xyz with pre-tanh offset for the representative
        voxel_features = voxel_features.clone()
        xyz_offset = voxel_features[:, self.IDX_XYZ] - voxel_centers
        voxel_features[:, self.IDX_XYZ] = torch.atanh(
            (xyz_offset / (self.voxel_size / 2)).clamp(-0.999, 0.999)
        )

        # Build voxel indices
        voxel_indices = torch.stack(
            [vox_b, vox_zi, vox_yi, vox_xi], dim=-1
        ).to(torch.int32)

        # --- ALL GT features with xyz/rot converted to pre-activation space ---
        # Compute voxel center for each GT point (using its voxel assignment)
        all_gt_centers = voxel_centers[inv_map]  # [N_valid, 3]
        all_gt_features = feat_flat_v.clone()

        # xyz: actual offset → pre-tanh space (inverse of tanh(x) * voxel_size/2)
        xyz_offset = all_gt_features[:, self.IDX_XYZ] - all_gt_centers
        all_gt_features[:, self.IDX_XYZ] = torch.atanh(
            (xyz_offset / (self.voxel_size / 2)).clamp(-0.999, 0.999)
        )

        # rot: normalized quaternion → raw (no transform needed, but ensure
        # consistent scale by keeping as-is; the decoder output before
        # F.normalize will learn to produce near-unit vectors)

        all_gt_voxel_id = inv_map  # [N_valid], values in 0..M-1

        return (voxel_features, voxel_indices, voxel_centers, B,
                all_gt_features, all_gt_voxel_id)

    def devoxelize(self, decoded_features, voxel_indices, voxel_centers, batch_size):
        """
        Convert decoded voxel features back to recontrast_data format.

        Args:
            decoded_features  [M, K, 86]  — K Gaussians per voxel, raw values
            voxel_indices     [M, 4]       — (batch, z, y, x)
            voxel_centers     [M, 3]       — absolute XYZ of voxel centers
            batch_size        int

        Returns:
            recontrast_data dict with keys xyz, rot_maps, scale_maps,
            opacity_maps, sh_maps — each [B, M_b*K, *]
        """
        M, K, _ = decoded_features.shape
        device = decoded_features.device

        # Apply activations
        xyz_offset = torch.tanh(decoded_features[..., self.IDX_XYZ]) * (self.voxel_size / 2)
        rotation   = F.normalize(decoded_features[..., self.IDX_ROT], dim=-1)
        scale      = F.softplus(decoded_features[..., self.IDX_SCALE]) * 0.01
        opacity    = torch.sigmoid(decoded_features[..., self.IDX_OPA])
        sh_flat    = decoded_features[..., self.IDX_SH]  # [M, K, 75]

        # Absolute xyz: voxel_center + offset, broadcast over K
        xyz_abs = voxel_centers.unsqueeze(1) + xyz_offset  # [M, K, 3]

        # ReconDrive expects [M, K, 25, 3] for gsplat rasterization
        sh = sh_flat.reshape(M, K, 75).reshape(M, K, 3, 25).permute(0, 1, 3, 2)  # [M, K, 25, 3]

        # Group by batch
        batch_ids = voxel_indices[:, 0].long()  # [M]

        out_xyz     = []
        out_rot     = []
        out_scale   = []
        out_opacity = []
        out_sh      = []

        for b in range(batch_size):
            mask = batch_ids == b
            out_xyz.append(xyz_abs[mask].reshape(-1, 3))
            out_rot.append(rotation[mask].reshape(-1, 4))
            out_scale.append(scale[mask].reshape(-1, 3))
            out_opacity.append(opacity[mask].reshape(-1, 1))
            out_sh.append(sh[mask].reshape(-1, 25, 3))

        # Pad to same length across batch (use the max)
        max_n = max(t.shape[0] for t in out_xyz)

        def pad(tensors, fill=0.0):
            padded = []
            for t in tensors:
                n = t.shape[0]
                if n < max_n:
                    pad_shape = (max_n - n,) + t.shape[1:]
                    t = torch.cat([t, torch.full(pad_shape, fill, device=device, dtype=t.dtype)], dim=0)
                padded.append(t)
            return torch.stack(padded, dim=0)  # [B, max_n, *]

        return {
            'xyz':          pad(out_xyz),
            'rot_maps':     pad(out_rot),
            'scale_maps':   pad(out_scale),
            'opacity_maps': pad(out_opacity),
            'sh_maps':      pad(out_sh),
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _inv_softplus(x, beta=1.0, threshold=20.0):
        """Inverse of F.softplus(x, beta) for x > 0."""
        # softplus(y) = (1/beta) * log(1 + exp(beta*y))
        # inv: y = (1/beta) * log(exp(beta*x) - 1)
        bx = beta * x.clamp(min=1e-6)
        return torch.where(bx > threshold, x, (1.0 / beta) * torch.log(torch.expm1(bx)))
