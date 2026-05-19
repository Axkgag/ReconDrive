"""
ReconDriveAE: extends ReconDrive_LITModelModule with a 3D Gaussian Autoencoder.

Original recondrive_model.py is untouched. This class only overrides
training_step and validation_step to insert the AE encode→decode pass
between get_recontrast_data() and render_splating_imgs().
"""

import os
import io

import torch
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from PIL import Image

from models.recondrive_model import ReconDrive_LITModelModule
from models.gaussian_autoencoder import GaussianAutoencoder


class ReconDriveAE_LITModelModule(ReconDrive_LITModelModule):
    """
    Adds a GaussianAutoencoder on top of the frozen ReconDrive stage-2 model.

    The AE encodes the per-pixel Gaussians produced by get_recontrast_data()
    into a sparse latent grid and decodes them back to K Gaussians per voxel.
    The reconstructed Gaussians are then rendered and compared to GT images.

    Loss = lambda_attr * attr_loss + lambda_render * render_loss
    """

    def __init__(self, cfg, save_dir='.', logger=None):
        super().__init__(cfg, save_dir, logger)
        ae_cfg = cfg.get('ae_cfg', {})
        self.gaussian_ae = GaussianAutoencoder(ae_cfg)
        self.lambda_render = ae_cfg.get('lambda_render', 1.0)

        # Validation visualization settings
        self.save_val_visualizations = cfg.get('save_val_visualizations', False)
        self.val_vis_interval = int(cfg.get('val_vis_interval', 200))
        self.val_vis_max_per_epoch = int(cfg.get('val_vis_max_per_epoch', 4))
        self._val_vis_saved_this_epoch = 0

        # Training visualization settings
        self.train_vis_interval = int(cfg.get('train_vis_interval', 500))

        # Freeze backbone (ReconDrive stage1 model) by default
        self.freeze_backbone = ae_cfg.get('freeze_backbone', True)
        if self.freeze_backbone:
            self._freeze_backbone_parameters()
            print("✓ Backbone (ReconDrive stage1) is FROZEN - only AE will be trained")
        else:
            print("⚠ Backbone (ReconDrive stage1) is TRAINABLE - both backbone and AE will be trained")

    def _freeze_backbone_parameters(self):
        """Freeze all parameters in self.model (ReconDrive backbone)"""
        frozen_count = 0
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.requires_grad = False
                frozen_count += 1
        print(f"  Frozen {frozen_count} backbone parameters")

    def on_validation_epoch_start(self):
        """Reset validation visualization counter at the start of each epoch."""
        self._val_vis_saved_this_epoch = 0

    def on_train_epoch_end(self):
        """Save separate checkpoints for AE and backbone at the end of each epoch."""
        if not hasattr(self, 'save_dir'):
            return

        # Create separate checkpoint directory
        separate_ckpt_dir = os.path.join(self.save_dir, '..', 'separate_ckpts')
        os.makedirs(separate_ckpt_dir, exist_ok=True)

        epoch = self.current_epoch

        # Save AE weights only
        ae_ckpt_path = os.path.join(separate_ckpt_dir, f'ae_epoch_{epoch:02d}.ckpt')
        torch.save({
            'epoch': epoch,
            'ae_state_dict': self.gaussian_ae.state_dict(),
            'ae_config': self.gaussian_ae.voxel_size if hasattr(self.gaussian_ae, 'voxel_size') else None,
        }, ae_ckpt_path)

        # Save backbone weights only (if not frozen or if user wants to track it)
        if not self.freeze_backbone:
            backbone_ckpt_path = os.path.join(separate_ckpt_dir, f'backbone_epoch_{epoch:02d}.ckpt')
            torch.save({
                'epoch': epoch,
                'backbone_state_dict': self.model.state_dict(),
            }, backbone_ckpt_path)
            print(f"✓ Saved separate checkpoints: {ae_ckpt_path}, {backbone_ckpt_path}")
        else:
            print(f"✓ Saved AE checkpoint: {ae_ckpt_path}")

    def load_ae_checkpoint(self, ae_ckpt_path):
        """Load AE weights from a separate checkpoint file."""
        checkpoint = torch.load(ae_ckpt_path, map_location=self.device)
        self.gaussian_ae.load_state_dict(checkpoint['ae_state_dict'])
        print(f"✓ Loaded AE weights from: {ae_ckpt_path}")
        if 'epoch' in checkpoint:
            print(f"  Checkpoint epoch: {checkpoint['epoch']}")

    def load_backbone_checkpoint(self, backbone_ckpt_path):
        """Load backbone weights from a separate checkpoint file."""
        checkpoint = torch.load(backbone_ckpt_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['backbone_state_dict'])
        print(f"✓ Loaded backbone weights from: {backbone_ckpt_path}")
        if 'epoch' in checkpoint:
            print(f"  Checkpoint epoch: {checkpoint['epoch']}")

    def save_validation_step_images(self, batch_idx, splating_stage1, splating_ae,
                                    recon_stage1, recon_ae):
        """
        Save a single composite figure per step:

          Rows:    one per camera
          Columns: GT image | Stage1 render | AE render | Stage1 3D Gauss | AE 3D Gauss

        Args:
            batch_idx: batch index
            splating_stage1: render dict using original (Stage1) Gaussians
            splating_ae:     render dict using AE-reconstructed Gaussians
            recon_stage1: original recontrast_data dict (xyz, opacity_maps, sh_maps)
            recon_ae:     reconstructed recontrast_data dict
        """
        from pytorch_lightning.utilities import rank_zero_only

        @rank_zero_only
        def _save():
            vis_subdir = 'train_visualizations' if self.stage == 'train' else 'val_visualizations'
            save_dir = os.path.join(
                self.save_dir, vis_subdir, f'epoch_{self.current_epoch:04d}'
            )
            os.makedirs(save_dir, exist_ok=True)

            # Use frame 0 only — that's the scene reconstruction frame.
            frame_id = 0

            # Per-camera images (GT, stage1 render, AE render)
            gt_imgs, stage1_imgs, ae_imgs = [], [], []
            for cam_id in range(self.num_cams):
                pred_s1 = ('gaussian_color', frame_id, cam_id)
                pred_ae = ('gaussian_color', frame_id, cam_id)
                gt_key  = ('groudtruth',     frame_id, cam_id)

                if (pred_s1 not in splating_stage1 or
                    pred_ae not in splating_ae or
                    gt_key  not in splating_stage1):
                    return

                gt_imgs.append(splating_stage1[gt_key][0])      # [3, H, W]
                stage1_imgs.append(splating_stage1[pred_s1][0])
                ae_imgs.append(splating_ae[pred_ae][0])

            # 3D Gaussian scatter images (one per source) — shared across all rows
            stage1_3d_img = self._render_gaussian_scene_image(recon_stage1, batch_idx_in_batch=0)
            ae_3d_img     = self._render_gaussian_scene_image(recon_ae,     batch_idx_in_batch=0)

            # Build the composite figure
            fig, axes = plt.subplots(
                nrows=self.num_cams, ncols=5,
                figsize=(5 * 4, self.num_cams * 2.3),
                dpi=120,
            )
            col_titles = ['Image', 'Stage1 Render', 'AE Render',
                          'Stage1 3D Gauss', 'AE 3D Gauss']

            for cam_id in range(self.num_cams):
                cam_name = (self.camera_names[cam_id]
                            if cam_id < len(self.camera_names) else f'CAM_{cam_id}')

                axes[cam_id, 0].imshow(self._tensor_to_uint8(gt_imgs[cam_id]))
                axes[cam_id, 1].imshow(self._tensor_to_uint8(stage1_imgs[cam_id]))
                axes[cam_id, 2].imshow(self._tensor_to_uint8(ae_imgs[cam_id]))
                axes[cam_id, 3].imshow(stage1_3d_img)
                axes[cam_id, 4].imshow(ae_3d_img)

                axes[cam_id, 0].set_ylabel(cam_name, fontsize=9)
                for c in range(5):
                    axes[cam_id, c].set_xticks([])
                    axes[cam_id, c].set_yticks([])
                    if cam_id == 0:
                        axes[cam_id, c].set_title(col_titles[c], fontsize=10)

            plt.tight_layout()
            out_path = os.path.join(
                save_dir,
                f'val_step_{self.global_step:08d}_batch_{batch_idx:05d}.png'
            )
            fig.savefig(out_path, bbox_inches='tight')
            plt.close(fig)

        _save()

    @staticmethod
    def _tensor_to_uint8(t):
        """[C, H, W] float in [0,1] → uint8 numpy [H, W, C] for matplotlib."""
        arr = t.detach().cpu().float().clamp(0, 1).numpy()
        return (arr.transpose(1, 2, 0) * 255).astype(np.uint8)

    @staticmethod
    def _render_gaussian_scene_image(recontrast_data, batch_idx_in_batch=0,
                                     opacity_thresh=0.1, max_points=80_000,
                                     elev=25, azim=-60):
        """
        Render a 3D scatter of Gaussian positions and return as RGB numpy [H, W, 3].

        Color comes from the SH degree-0 (DC) coefficient.
        Transparent points (opacity below threshold) are filtered out.
        """
        xyz     = recontrast_data['xyz'][batch_idx_in_batch].detach().cpu().float().numpy()
        opacity = recontrast_data['opacity_maps'][batch_idx_in_batch].detach().cpu().float().numpy().squeeze(-1)
        sh      = recontrast_data['sh_maps'][batch_idx_in_batch].detach().cpu().float().numpy()

        # sh layout in this codebase: [N, 25, 3] (last dim = RGB)
        if sh.shape[-1] != 3 and sh.shape[-2] == 3:
            sh = sh.transpose(0, 2, 1)  # [N, 3, 25] → [N, 25, 3]

        mask = opacity > opacity_thresh
        if mask.sum() == 0:
            mask = np.ones_like(opacity, dtype=bool)
        xyz, opacity, sh = xyz[mask], opacity[mask], sh[mask]

        # DC SH → RGB
        C0 = 0.28209479177387814
        rgb = np.clip(sh[:, 0, :] / C0 * 0.5 + 0.5, 0.0, 1.0)

        # Subsample for speed
        if len(xyz) > max_points:
            idx = np.random.choice(len(xyz), max_points, replace=False)
            xyz, rgb, opacity = xyz[idx], rgb[idx], opacity[idx]

        # Trim 1st-99th percentile per axis to ignore stray outliers.
        if len(xyz) > 100:
            for axis in range(3):
                lo, hi = np.percentile(xyz[:, axis], [1, 99])
                m = (xyz[:, axis] >= lo) & (xyz[:, axis] <= hi)
                xyz, rgb, opacity = xyz[m], rgb[m], opacity[m]

        alpha = np.clip(opacity, 0.05, 1.0)
        rgba  = np.concatenate([rgb, alpha[:, None]], axis=1)

        fig = plt.figure(figsize=(5, 5), dpi=100)
        ax  = fig.add_subplot(111, projection='3d')
        if len(xyz) > 0:
            ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2],
                       c=rgba, s=0.3, linewidths=0, depthshade=True)
            ranges = np.array([[xyz[:, i].min(), xyz[:, i].max()] for i in range(3)])
            max_range = (ranges[:, 1] - ranges[:, 0]).max() / 2 or 1.0
            mid = ranges.mean(axis=1)
            ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
            ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
            ax.set_zlim(mid[2] - max_range, mid[2] + max_range)

        ax.set_xlabel('X', fontsize=7)
        ax.set_ylabel('Y', fontsize=7)
        ax.set_zlabel('Z', fontsize=7)
        ax.tick_params(labelsize=6)
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(f'{len(xyz):,} pts', fontsize=8)
        fig.tight_layout()

        # Render to RGB array (avoid disk I/O)
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)
        return np.array(Image.open(buf).convert('RGB'))

    @staticmethod
    def _attach_render_fields(batch_recontrast_recon):
        """
        Ensure reconstructed Gaussian dict has fields required by renderer.
        """
        xyz = batch_recontrast_recon['xyz']
        batch_recontrast_recon['forward_flow'] = torch.zeros_like(xyz)
        # AE reconstruction is scene-global (not camera-chunk ordered).
        batch_recontrast_recon['ae_global_points'] = True
        return batch_recontrast_recon

    @staticmethod
    def _extract_frame0_gaussians(batch_recontrast_data):
        """
        Stage1 backbone produces Gaussians for both frame 0 and the last
        context frame, concatenated as [frame0_points, frame_last_points].

        For visualisation we only want the frame-0 half, otherwise the two
        time-shifted point clouds rendered together produce ghosting.

        Returns a shallow copy of the dict with the front half of every
        per-point tensor and the same scalar/non-point keys preserved.
        """
        out = {}
        for key, val in batch_recontrast_data.items():
            if torch.is_tensor(val) and val.dim() >= 2:
                # All point-aligned tensors have shape [B, N_total, ...]
                # where N_total = 2 * (num_cams * H * W). Take the first half.
                n_total = val.shape[1]
                if n_total % 2 == 0:
                    out[key] = val[:, : n_total // 2]
                else:
                    out[key] = val
            else:
                out[key] = val
        return out

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def training_step(self, batch_input, batch_idx):
        self.stage = stage = 'train'
        self.prob_sample_rendered_ids()
        self._log_weights_and_grads(batch_input)

        # 1. Get original Gaussians from frozen VGGT backbone
        batch_recontrast_data = self.get_recontrast_data(batch_input, batch_idx)

        # 2. AE encode → decode (with Chamfer targets)
        batch_recontrast_recon, _, chamfer_targets = self.gaussian_ae.forward_with_targets(batch_recontrast_data)
        batch_recontrast_recon = self._attach_render_fields(batch_recontrast_recon)

        # 3. Chamfer-based attribute reconstruction loss
        loss_dict = self.gaussian_ae.loss_fn(
            chamfer_targets['pred_raw'],
            chamfer_targets['all_gt_features'],
            chamfer_targets['all_gt_voxel_id'],
            chamfer_targets['M'],
            chamfer_targets['K'],
        )
        loss_attr = loss_dict['total']

        # 4. Render reconstructed Gaussians and compute photometric loss
        batch_render_data   = self.get_render_data(batch_input)
        batch_splating_data = self.render_splating_imgs(batch_recontrast_recon, batch_render_data)
        loss_render = self.compute_gaussian_loss(batch_splating_data)

        # 5. Projection loss (unchanged from parent)
        batch_render_project_data = self.render_project_imgs(batch_input, batch_recontrast_data)
        loss_project = self.compute_project_loss(batch_render_project_data)

        loss_norm = self.compute_norm_loss(batch_recontrast_data)

        loss_all = loss_attr + self.lambda_render * loss_render + loss_project + loss_norm

        # Logging
        self.log(f'{stage}/loss_total', loss_all.item(), on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log(f'{stage}/ae_attr',    loss_attr.item(),   on_step=True, on_epoch=True, prog_bar=True,  sync_dist=True)
        self.log(f'{stage}/ae_render',  loss_render.item(), on_step=True, on_epoch=True, prog_bar=True,  sync_dist=True)
        self.log(f'{stage}/ae_xyz',     loss_dict['xyz'].item(),     on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log(f'{stage}/ae_rot',     loss_dict['rot'].item(),     on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log(f'{stage}/ae_scale',   loss_dict['scale'].item(),   on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log(f'{stage}/ae_opacity', loss_dict['opacity'].item(), on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log(f'{stage}/ae_sh',      loss_dict['sh'].item(),      on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log(f'{stage}/proj',       loss_project.item(), on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log(f'{stage}/norm',       loss_norm.item(),    on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)

        # Learning rate
        opt = self.optimizers()
        current_lr = opt.param_groups[0]['lr']
        self.log(f'{stage}/lr', current_lr, on_step=True, on_epoch=False, prog_bar=False, sync_dist=True)

        psnr, ssim, lpips = self.compute_reconstruction_metrics(batch_splating_data, stage)

        # Training visualization
        if self.global_step % self.train_vis_interval == 0:
            with torch.no_grad():
                stage1_recon_frame0 = self._extract_frame0_gaussians(batch_recontrast_data)
                stage1_recon_frame0['ae_global_points'] = True
                batch_splating_stage1 = self.render_splating_imgs(
                    stage1_recon_frame0, batch_render_data
                )
            self.save_validation_step_images(
                batch_idx,
                splating_stage1=batch_splating_stage1,
                splating_ae=batch_splating_data,
                recon_stage1=stage1_recon_frame0,
                recon_ae=batch_recontrast_recon,
            )
            del batch_splating_stage1, stage1_recon_frame0

        del batch_input, batch_recontrast_data, batch_recontrast_recon
        del batch_render_data, batch_render_project_data, batch_splating_data
        del psnr, ssim, lpips, chamfer_targets

        return loss_all

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validation_step(self, batch_input, batch_idx):
        self.stage = stage = 'val'
        self.all_render_frame_ids = range(0, 6)
        self.set_normal_params(batch_input)
        self.init_novel_view_mode()

        batch_recontrast_data = self.get_recontrast_data(batch_input)
        batch_recontrast_recon, _, chamfer_targets = self.gaussian_ae.forward_with_targets(batch_recontrast_data)
        batch_recontrast_recon = self._attach_render_fields(batch_recontrast_recon)

        loss_dict = self.gaussian_ae.loss_fn(
            chamfer_targets['pred_raw'],
            chamfer_targets['all_gt_features'],
            chamfer_targets['all_gt_voxel_id'],
            chamfer_targets['M'],
            chamfer_targets['K'],
        )
        loss_attr   = loss_dict['total']

        batch_render_data    = self.get_render_data(batch_input)
        batch_splating_ae    = self.render_splating_imgs(batch_recontrast_recon, batch_render_data)
        loss_render  = self.compute_gaussian_loss(batch_splating_ae)
        loss_depth   = self.compute_depth_loss(batch_splating_ae)

        batch_render_project_data = self.render_project_imgs(batch_input, batch_recontrast_data)
        loss_project = self.compute_project_loss(batch_render_project_data)
        loss_norm    = self.compute_norm_loss(batch_recontrast_data)

        loss_all = loss_attr + self.lambda_render * loss_render + loss_depth + loss_project + loss_norm

        self.log(f'{stage}/loss_total', loss_all.item(), on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log(f'{stage}/ae_attr',   loss_attr.item(),   on_step=True, on_epoch=True, prog_bar=True,  sync_dist=True)
        self.log(f'{stage}/ae_render', loss_render.item(), on_step=True, on_epoch=True, prog_bar=True,  sync_dist=True)
        self.log(f'{stage}/ae_xyz',    loss_dict['xyz'].item(),     on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log(f'{stage}/ae_rot',    loss_dict['rot'].item(),     on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log(f'{stage}/ae_scale',  loss_dict['scale'].item(),   on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log(f'{stage}/ae_opacity',loss_dict['opacity'].item(), on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log(f'{stage}/ae_sh',     loss_dict['sh'].item(),      on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log(f'{stage}/depth',     loss_depth.item(),  on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log(f'{stage}/proj',      loss_project.item(),on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log(f'{stage}/norm',      loss_norm.item(),   on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)

        psnr, ssim, lpips = self.compute_reconstruction_metrics(batch_splating_ae, stage)

        # Composite visualization: GT | Stage1 render | AE render | Stage1 3D | AE 3D
        if (self.save_val_visualizations and
                (batch_idx % max(self.val_vis_interval, 1) == 0) and
                self._val_vis_saved_this_epoch < self.val_vis_max_per_epoch):
            with torch.no_grad():
                # Build a "frame-0-only" view of the original Gaussians, then mark it
                # as global so render_splating_imgs skips the per-camera frame0+frame1
                # concat path that produces ghosting.
                stage1_recon_frame0 = self._extract_frame0_gaussians(batch_recontrast_data)
                stage1_recon_frame0['ae_global_points'] = True

                batch_splating_stage1 = self.render_splating_imgs(
                    stage1_recon_frame0, batch_render_data
                )
            self.save_validation_step_images(
                batch_idx,
                splating_stage1=batch_splating_stage1,
                splating_ae=batch_splating_ae,
                recon_stage1=stage1_recon_frame0,
                recon_ae=batch_recontrast_recon,
            )
            self._val_vis_saved_this_epoch += 1
            del batch_splating_stage1, stage1_recon_frame0

        del batch_input, batch_recontrast_data, batch_recontrast_recon
        del batch_render_data, batch_render_project_data, batch_splating_ae
        del psnr, ssim, lpips

        return loss_all

    # ------------------------------------------------------------------
    # configure_optimizers: train only the AE, keep VGGT frozen
    # ------------------------------------------------------------------

    def configure_optimizers(self):
        ae_params = list(self.gaussian_ae.parameters())
        if not ae_params:
            raise RuntimeError("GaussianAutoencoder has no parameters.")

        optimizer = torch.optim.AdamW(
            ae_params,
            lr=self.learning_rate,
            betas=(0.9, 0.98),
            eps=1e-7,
            weight_decay=self.weight_decay,
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=self.lr_restart_epoch,
            T_mult=self.lr_restart_mult,
            eta_min=self.learning_rate * self.lr_min_factor * 0.1,
        )

        return {
            'optimizer': optimizer,
            'lr_scheduler': {'scheduler': scheduler, 'interval': 'epoch'},
        }
