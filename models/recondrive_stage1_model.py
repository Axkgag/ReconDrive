#----------------------------------------------------------------#
# ReconDrive                                                     #
# Source code: https://github.com/TuojingAI/ReconDrive           #
# Copyright (c) TuojingAI. All rights reserved.                  #
#----------------------------------------------------------------#

import os
import io
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from models.recondrive_model import ReconDrive_LITModelModule
from einops import rearrange, reduce
from models.gaussian_util import render, focal2fov, getProjectionMatrix,  depth2pc, pc2depth, rotate_sh, quat_multiply


class ReconDriveStage1_LITModelModule(ReconDrive_LITModelModule):
    """Stage1 single-frame 3D Gaussian training module."""

    def __init__(self, cfg, save_dir='.', logger=None):
        super().__init__(cfg, save_dir, logger)
        self._configure_stage1_trainable()

        # Visualization settings
        self.vis_step = 0
        self._val_vis_saved_this_epoch = 0
        if not hasattr(self, 'occ_debug_interval'):
            self.occ_debug_interval = 200

    def _configure_stage1_trainable(self):
        # Freeze all parameters first.
        for param in self.model.parameters():
            param.requires_grad = False

        # Unfreeze depth head according to cfg.depth_head_unfreeze.
        # Levels: none | output_conv2 | output_conv1 | refinenet | full
        depth_unfreeze = getattr(self, 'depth_head_unfreeze', 'none')
        for param in self.model.depth_head.parameters():
            param.requires_grad = False
        dh = self.model.depth_head
        scratch = getattr(dh, 'scratch', None)
        if depth_unfreeze == 'full':
            for param in dh.parameters():
                param.requires_grad = True
        elif depth_unfreeze == 'refinenet' and scratch is not None:
            for attr in ('refinenet1', 'refinenet2', 'refinenet3', 'refinenet4',
                         'output_conv1', 'output_conv2'):
                mod = getattr(scratch, attr, None)
                if mod is not None:
                    for param in mod.parameters():
                        param.requires_grad = True
        elif depth_unfreeze == 'output_conv1' and scratch is not None:
            for attr in ('output_conv1', 'output_conv2'):
                mod = getattr(scratch, attr, None)
                if mod is not None:
                    for param in mod.parameters():
                        param.requires_grad = True
        elif depth_unfreeze == 'output_conv2' and scratch is not None:
            mod = getattr(scratch, 'output_conv2', None)
            if mod is not None:
                for param in mod.parameters():
                    param.requires_grad = True
        # 'none': all depth_head params remain frozen (already set above)
        for param in self.model.gs_head.parameters():
            param.requires_grad = True

        # Unfreeze LoRA parameters in aggregator.
        for name, param in self.model.aggregator.named_parameters():
            if "lora_" in name:
                param.requires_grad = True

        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        lora_params = sum(
            p.numel()
            for n, p in self.model.aggregator.named_parameters()
            if p.requires_grad and "lora_" in n
        )
        depth_params = sum(p.numel() for p in self.model.depth_head.parameters() if p.requires_grad)
        gs_params = sum(p.numel() for p in self.model.gs_head.parameters() if p.requires_grad)
        print(
            "Stage1 可训练参数:",
            f"total={total_params:,}, trainable={trainable_params:,},",
            f"lora={lora_params:,}, depth_head={depth_params:,}, gs_head={gs_params:,}",
        )

    def _set_stage1_frame_ids(self):
        self.all_render_frame_ids = [0]

    def on_validation_epoch_start(self):
        """Reset validation visualization counter at the start of each epoch."""
        self._val_vis_saved_this_epoch = 0

    def training_step(self, batch_input, batch_idx):
        self.stage = stage = 'train'

        self._set_stage1_frame_ids()
        self._log_weights_and_grads(batch_input)
        self._log_current_lrs()

        batch_recontrast_data = self.get_recontrast_data(batch_input, batch_idx)
        loss_norm = self.compute_norm_loss(batch_recontrast_data)

        batch_render_data = self.get_render_data(batch_input)
        batch_splating_data = self.render_splating_imgs(
            {**batch_recontrast_data, 'ae_global_points': True}, batch_render_data
        )
        batch_splating_data = self._apply_occ_render_mask(batch_splating_data, batch_input)

        loss_gaussian = self.compute_gaussian_loss(batch_splating_data)
        loss_depth = (
            self.compute_depth_head_loss(batch_recontrast_data, batch_input)
            if getattr(self, 'enable_depth_supervision', False)
            else None
        )

        # 新增：计算 Occ 损失（仅在启用时）
        if getattr(self, 'enable_occ_supervision', False):
            loss_occ = self.compute_occ_loss(batch_recontrast_data, batch_input)
            self.log(f'{stage}/occ', loss_occ.item(), on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        else:
            loss_occ = torch.tensor(0.0, device=self.device)

        self.log(f'{stage}/gs', loss_gaussian.item(), on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log(f'{stage}/norm', loss_norm.item(), on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)
        if loss_depth is not None:
            self.log(f'{stage}/depth', loss_depth.item(), on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)

        loss_all = loss_gaussian + loss_norm + loss_occ
        if loss_depth is not None:
            loss_all = loss_all + loss_depth
        psnr, ssim, lpips = self.compute_reconstruction_metrics(batch_splating_data, stage)

        # Training visualization
        is_global_zero = True
        if hasattr(self, "trainer") and self.trainer is not None:
            is_global_zero = self.trainer.is_global_zero
        should_visualize = (
            is_global_zero and
            self.train_vis_interval > 0 and
            self.vis_step % self.train_vis_interval == 0
        )
        if should_visualize:
            with torch.no_grad():
                self.save_validation_step_images(
                    batch_idx,
                    batch_splating_data,
                    batch_recontrast_data,
                )

        del batch_input, batch_recontrast_data, batch_render_data, batch_splating_data
        del psnr, ssim, lpips

        self.vis_step += 1

        return loss_all

    def validation_step(self, batch_input, batch_idx):
        self.stage = stage = 'val'

        self._set_stage1_frame_ids()
        batch_recontrast_data = self.get_recontrast_data(batch_input)

        batch_render_data = self.get_render_data(batch_input)
        loss_norm = self.compute_norm_loss(batch_recontrast_data)

        batch_splating_data = self.render_splating_imgs(
            {**batch_recontrast_data, 'ae_global_points': True}, batch_render_data
        )
        batch_splating_data = self._apply_occ_render_mask(batch_splating_data, batch_input)
        loss_gaussian = self.compute_gaussian_loss(batch_splating_data)
        loss_depth = (
            self.compute_depth_head_loss(batch_recontrast_data, batch_input)
            if getattr(self, 'enable_depth_supervision', False)
            else None
        )

        self.log(f'{stage}/gs', loss_gaussian.item(), on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log(f'{stage}/norm', loss_norm.item(), on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)
        if loss_depth is not None:
            self.log(f'{stage}/depth', loss_depth.item(), on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)

        loss_all = loss_gaussian + loss_norm
        if loss_depth is not None:
            loss_all = loss_all + loss_depth
        psnr, ssim, lpips = self.compute_reconstruction_metrics(batch_splating_data, stage)

        # Validation visualization (limited per epoch)
        is_global_zero = True
        if hasattr(self, "trainer") and self.trainer is not None:
            is_global_zero = self.trainer.is_global_zero
        should_visualize = (
            is_global_zero and
            self.val_vis_interval > 0 and
            self._val_vis_saved_this_epoch < self.val_vis_max_per_epoch
        )
        if should_visualize:
            with torch.no_grad():
                self.save_validation_step_images(
                    batch_idx,
                    batch_splating_data,
                    batch_recontrast_data,
                )
            self._val_vis_saved_this_epoch += 1

        del batch_input, batch_recontrast_data, batch_render_data, batch_splating_data
        del psnr, ssim, lpips

        return loss_all

    def test_step(self, batch_input, batch_idx):
        self.stage = stage = 'test'

        self._set_stage1_frame_ids()
        batch_recontrast_data = self.get_recontrast_data(batch_input)

        batch_render_data = self.get_render_data(batch_input)
        loss_norm = self.compute_norm_loss(batch_recontrast_data)

        batch_splating_data = self.render_splating_imgs(
            {**batch_recontrast_data, 'ae_global_points': True}, batch_render_data
        )
        batch_splating_data = self._apply_occ_render_mask(batch_splating_data, batch_input)
        loss_gaussian = self.compute_gaussian_loss(batch_splating_data)
        loss_depth = (
            self.compute_depth_head_loss(batch_recontrast_data, batch_input)
            if getattr(self, 'enable_depth_supervision', False)
            else None
        )

        self.log(f'{stage}/gs', loss_gaussian.item(), on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log(f'{stage}/norm', loss_norm.item(), on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)
        if loss_depth is not None:
            self.log(f'{stage}/depth', loss_depth.item(), on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)

        loss_all = loss_gaussian + loss_norm
        if loss_depth is not None:
            loss_all = loss_all + loss_depth
        psnr, ssim, lpips = self.compute_reconstruction_metrics(batch_splating_data, stage)

        del batch_input, batch_recontrast_data, batch_render_data, batch_splating_data
        del psnr, ssim, lpips

        return loss_all

    def save_validation_step_images(self, batch_idx, batch_splating_data, batch_recontrast_data):
        """
        Save composite figure for Stage1 training visualization.

        Rows: one per camera
        Columns: GT Image | Stage1 Render | Depth Map | Stage1 3D Gauss
        """
        from pytorch_lightning.utilities import rank_zero_only

        @rank_zero_only
        def _save():
            # Debug: Print Gaussian statistics per camera
            # if self.vis_step == 0:  # Only print on first visualization
            #     self._debug_print_gaussian_stats(batch_recontrast_data)

            vis_subdir = 'train_visualizations' if self.stage == 'train' else 'val_visualizations'
            save_dir = os.path.join(
                self.save_dir, vis_subdir, f'epoch_{self.current_epoch:04d}'
            )
            os.makedirs(save_dir, exist_ok=True)

            frame_id = 0
            gt_imgs, stage1_imgs, pred_depth_imgs, gt_depth_imgs = [], [], [], []

            # Pre-extract predicted depth maps: [B, num_cams*H*W] -> [num_cams, H, W]
            pred_depths_all = None
            if 'pred_depths' in batch_recontrast_data:
                from einops import rearrange as _rearrange
                pred_depths_all = _rearrange(
                    batch_recontrast_data['pred_depths'][0:1],
                    'b (c h w) -> b c h w',
                    c=self.num_cams,
                    h=self.height,
                    w=self.width,
                )[0]  # [num_cams, H, W]

            for cam_id in range(self.num_cams):
                pred_key = ('gaussian_color', frame_id, cam_id)
                gt_key   = ('groudtruth',     frame_id, cam_id)
                gt_depth_key = ('gt_depths', frame_id, cam_id)

                if pred_key not in batch_splating_data or gt_key not in batch_splating_data:
                    return

                gt_img = batch_splating_data[gt_key][0]
                pred_img = batch_splating_data[pred_key][0]

                # Use warped_mask (already gated by OCC render mask) so visualization
                # only keeps visible regions for both GT and rendered images.
                mask_key = ('warped_mask', frame_id, cam_id)
                if mask_key in batch_splating_data:
                    vis_mask = batch_splating_data[mask_key][0]
                    if vis_mask.dim() == 2:
                        vis_mask = vis_mask.unsqueeze(0)
                    if vis_mask.dim() == 3 and vis_mask.shape[0] == 1:
                        vis_mask = vis_mask.expand_as(gt_img)
                    elif vis_mask.dim() == 3 and vis_mask.shape[0] != gt_img.shape[0]:
                        vis_mask = vis_mask[:1].expand_as(gt_img)
                    vis_mask = vis_mask.to(dtype=gt_img.dtype, device=gt_img.device)
                else:
                    vis_mask = torch.ones_like(gt_img)

                gt_imgs.append((gt_img * vis_mask).clamp(0, 1))
                stage1_imgs.append((pred_img * vis_mask).clamp(0, 1))

                if pred_depths_all is not None:
                    pred_depth_imgs.append(pred_depths_all[cam_id].detach().cpu().float().numpy())
                else:
                    pred_depth_imgs.append(None)

                if gt_depth_key in batch_splating_data:
                    gt_depth_np = batch_splating_data[gt_depth_key][0].detach().cpu().float().numpy()
                    gt_depth_imgs.append(np.squeeze(gt_depth_np))
                else:
                    gt_depth_imgs.append(None)

            stage1_3d_img = self._render_gaussian_scene_image(batch_recontrast_data, batch_idx_in_batch=0)

            has_gt_depth_vis = any(depth is not None for depth in gt_depth_imgs)
            ncols = 5 if has_gt_depth_vis else 4
            fig, axes = plt.subplots(
                nrows=self.num_cams, ncols=ncols,
                figsize=(4 * ncols, self.num_cams * 2.3),
                dpi=120,
            )
            col_titles = ['GT Image', 'Stage1 Render', 'Pred Depth']
            if has_gt_depth_vis:
                col_titles.append('GT Depth')
            col_titles.append('Stage1 3D Gauss')

            for cam_id in range(self.num_cams):
                cam_name = (self.camera_names[cam_id]
                            if cam_id < len(self.camera_names) else f'CAM_{cam_id}')

                axes[cam_id, 0].imshow(self._tensor_to_uint8(gt_imgs[cam_id]))
                axes[cam_id, 1].imshow(self._tensor_to_uint8(stage1_imgs[cam_id]))

                depth_np = pred_depth_imgs[cam_id]
                if depth_np is not None:
                    axes[cam_id, 2].imshow(depth_np, cmap='magma', vmin=self.min_depth, vmax=self.max_depth)
                else:
                    axes[cam_id, 2].text(0.5, 0.5, 'N/A', ha='center', va='center',
                                         transform=axes[cam_id, 2].transAxes)

                if has_gt_depth_vis:
                    gt_depth_np = gt_depth_imgs[cam_id]
                    if gt_depth_np is not None:
                        axes[cam_id, 3].imshow(gt_depth_np, cmap='magma', vmin=self.min_depth, vmax=self.max_depth)
                    else:
                        axes[cam_id, 3].text(0.5, 0.5, 'N/A', ha='center', va='center',
                                             transform=axes[cam_id, 3].transAxes)
                    axes[cam_id, 4].imshow(stage1_3d_img)
                else:
                    axes[cam_id, 3].imshow(stage1_3d_img)

                axes[cam_id, 0].set_ylabel(cam_name, fontsize=9)
                for c in range(ncols):
                    axes[cam_id, c].set_xticks([])
                    axes[cam_id, c].set_yticks([])
                    if cam_id == 0:
                        axes[cam_id, c].set_title(col_titles[c], fontsize=10)

            plt.tight_layout()
            out_path = os.path.join(
                save_dir,
                f'step_{self.vis_step:08d}_batch_{batch_idx:05d}.png'
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

        if sh.shape[-1] != 3 and sh.shape[-2] == 3:
            sh = sh.transpose(0, 2, 1)

        mask = opacity > opacity_thresh
        if mask.sum() == 0:
            mask = np.ones_like(opacity, dtype=bool)
        xyz, opacity, sh = xyz[mask], opacity[mask], sh[mask]

        C0 = 0.28209479177387814
        rgb = np.clip(sh[:, 0, :] / C0 * 0.5 + 0.5, 0.0, 1.0)

        if len(xyz) > max_points:
            idx = np.random.choice(len(xyz), max_points, replace=False)
            xyz, rgb, opacity = xyz[idx], rgb[idx], opacity[idx]

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

        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)
        return np.array(Image.open(buf).convert('RGB'))

    def _debug_print_gaussian_stats(self, batch_recontrast_data):
        """Debug: Print Gaussian statistics per camera to diagnose rendering issues."""
        print("\n" + "="*80)
        print("DEBUG: Gaussian Statistics Per Camera")
        print("="*80)

        xyz = batch_recontrast_data['xyz'][0]  # [N, 3]
        opacity = batch_recontrast_data['opacity_maps'][0]  # [N, 1]
        scale = batch_recontrast_data['scale_maps'][0]  # [N, 3]
        rot = batch_recontrast_data['rot_maps'][0]  # [N, 4]

        # Assume Gaussians are organized by camera
        total_points = xyz.shape[0]
        points_per_cam = total_points // self.num_cams

        print(f"Total points: {total_points:,}")
        print(f"Points per camera: {points_per_cam:,}")
        print(f"Num cameras: {self.num_cams}\n")

        for cam_id in range(self.num_cams):
            start_idx = cam_id * points_per_cam
            end_idx = (cam_id + 1) * points_per_cam

            xyz_cam = xyz[start_idx:end_idx]
            opacity_cam = opacity[start_idx:end_idx]
            scale_cam = scale[start_idx:end_idx]
            rot_cam = rot[start_idx:end_idx]

            cam_name = self.camera_names[cam_id] if cam_id < len(self.camera_names) else f'CAM_{cam_id}'

            print(f"Camera {cam_id} ({cam_name}):")
            print(f"  XYZ range: X=[{xyz_cam[:, 0].min():.2f}, {xyz_cam[:, 0].max():.2f}], "
                  f"Y=[{xyz_cam[:, 1].min():.2f}, {xyz_cam[:, 1].max():.2f}], "
                  f"Z=[{xyz_cam[:, 2].min():.2f}, {xyz_cam[:, 2].max():.2f}]")
            print(f"  Opacity: mean={opacity_cam.mean():.4f}, min={opacity_cam.min():.4f}, "
                  f"max={opacity_cam.max():.4f}, >0.1: {(opacity_cam > 0.1).sum()}/{len(opacity_cam)}")
            print(f"  Scale: mean={scale_cam.mean():.4f}, min={scale_cam.min():.4f}, "
                  f"max={scale_cam.max():.4f}")
            print(f"  Rotation norm: mean={torch.norm(rot_cam, dim=-1).mean():.4f}")
            print()

        print("="*80 + "\n")

    def visualize_occ_comparison(self, occ_pred, occ_gt, save_path):
        """
        Visualize surface occupancy (GT vs predicted) without requiring occ_logits.
        occ_pred: can be one of:
          - a voxel grid [B, H, W, D] (binary or float)
          - logits/probabilities [B, H, W, D, C] (will take argmax or threshold)
          - None (function will return without saving)
        occ_gt: voxel grid [B, H, W, D]
        save_path: output image path
        """
        import os
        import numpy as np
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        if occ_pred is None or occ_gt is None:
            return

        # normalize inputs to numpy [H, W, D]
        if isinstance(occ_pred, (list, tuple)):
            occ_pred = occ_pred[0] if len(occ_pred) > 0 else None
        if occ_pred is None:
            return
        if hasattr(occ_pred, 'cpu'):
            occ_pred = occ_pred.detach().cpu().numpy()
        if hasattr(occ_gt, 'cpu'):
            occ_gt_np = occ_gt.detach().cpu().numpy()
        else:
            occ_gt_np = np.array(occ_gt)

        # If occ_pred has channels, reduce to single-class occupancy
        if occ_pred.ndim == 5:  # [B, H, W, D, C]
            try:
                pred_grid = occ_pred[0].argmax(axis=-1)
            except Exception:
                pred_grid = (occ_pred[0].max(axis=-1) > 0.5).astype(np.uint8)
        elif occ_pred.ndim == 4:  # [B, H, W, D] or [H,W,D]
            pred_grid = occ_pred[0] if occ_pred.shape[0] > 1 else occ_pred.squeeze(0)
        else:
            # fallback: try to squeeze
            pred_grid = np.squeeze(occ_pred)

        if pred_grid.ndim != 3:
            pred_grid = pred_grid.reshape(pred_grid.shape[-3:])

        gt = occ_gt_np[0] if occ_gt_np.shape[0] > 1 else occ_gt_np.squeeze(0)

        # Compute voxel centers in world coordinates
        vox_origin = np.array([-40.0, -40.0, -1.0], dtype=np.float32)
        voxel_size = float(0.4)

        H, W, D = pred_grid.shape
        gx = np.arange(0, H)
        gy = np.arange(0, W)
        gz = np.arange(0, D)
        xx, yy, zz = np.meshgrid(gx, gy, gz)
        grid_coords = np.array([xx.flatten(), yy.flatten(), zz.flatten()]).T.astype(np.float32)
        grid_coords = (grid_coords * voxel_size) + vox_origin.reshape([1, 3])

        labels_pred = pred_grid.flatten()
        labels_gt = gt.flatten()

        mask_pred = labels_pred > 0
        mask_gt = labels_gt > 0

        coords_pred = grid_coords[mask_pred]
        coords_gt = grid_coords[mask_gt]

        max_points = 100000
        if coords_pred.shape[0] > max_points:
            idx = np.random.choice(coords_pred.shape[0], max_points, replace=False)
            coords_pred = coords_pred[idx]
        if coords_gt.shape[0] > max_points:
            idx = np.random.choice(coords_gt.shape[0], max_points, replace=False)
            coords_gt = coords_gt[idx]

        fig = plt.figure(figsize=(12, 6), dpi=120)
        ax1 = fig.add_subplot(121, projection='3d')
        ax2 = fig.add_subplot(122, projection='3d')

        if coords_pred.shape[0] > 0:
            ax1.scatter(coords_pred[:, 0], coords_pred[:, 1], coords_pred[:, 2], c='red', s=0.5)
        if coords_gt.shape[0] > 0:
            ax2.scatter(coords_gt[:, 0], coords_gt[:, 1], coords_gt[:, 2], c='green', s=0.5)

        ax1.set_title('Predicted surface occ')
        ax2.set_title('Ground-truth surface occ')

        for ax in (ax1, ax2):
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_zticks([])

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, bbox_inches='tight')
        plt.close(fig)

        ax1 = fig.add_subplot(121, projection='3d', facecolor='white')
        ax2 = fig.add_subplot(122, projection='3d', facecolor='white')

        # scatter size depends on voxel_size
        s = (voxel_size * 100) ** 2 * 0.01  # heuristic to get visible points
        if coords_pred.shape[0] > 0:
            ax1.scatter(coords_pred[:, 0], coords_pred[:, 1], coords_pred[:, 2], c=cols_pred, s=1.5, depthshade=True)
        ax1.set_title(f'Predicted Occupancy ({coords_pred.shape[0]} pts)', fontsize=10)
        ax1.set_xlabel('X'); ax1.set_ylabel('Y'); ax1.set_zlabel('Z')

        if coords_gt.shape[0] > 0:
            ax2.scatter(coords_gt[:, 0], coords_gt[:, 1], coords_gt[:, 2], c=cols_gt, s=1.5, depthshade=True)
        ax2.set_title(f'Ground Truth Occupancy ({coords_gt.shape[0]} pts)', fontsize=10)
        ax2.set_xlabel('X'); ax2.set_ylabel('Y'); ax2.set_zlabel('Z')

        # set equal aspect and sensible limits based on union of points
        all_pts = None
        if coords_pred.shape[0] > 0 and coords_gt.shape[0] > 0:
            all_pts = np.vstack([coords_pred, coords_gt])
        elif coords_pred.shape[0] > 0:
            all_pts = coords_pred
        elif coords_gt.shape[0] > 0:
            all_pts = coords_gt

        if all_pts is not None and all_pts.shape[0] > 0:
            mins = all_pts.min(axis=0)
            maxs = all_pts.max(axis=0)
            mid = (mins + maxs) / 2.0
            max_range = ((maxs - mins).max() / 2.0) or 1.0
            for ax in (ax1, ax2):
                ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
                ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
                ax.set_zlim(mid[2] - max_range, mid[2] + max_range)

        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close(fig)
        return

    def get_model_outputs(self, data_dict):
        inputs = data_dict['context_frames']
        outputs = {}

        image_list = []
        c2e_extr_list = []

        for frame_cam_id in range(inputs[('color_aug', 0)].shape[1]):
            c2e_extr = inputs['c2e_extr'][:, frame_cam_id, ...]
            image_list.append(inputs[(f'color_aug', 0)][:,frame_cam_id,...])
            c2e_extr_list.append(c2e_extr)
        image_list = torch.stack(image_list,dim=1)

        # 6 -> 18
        # [4, 6, 280, 518, 1], [4, 6, 280, 518, 4], [4, 6, 280, 518, 3], [4, 6, 280, 518, 1], [4, 6, 280, 518, 3, 25], [4, 6, 280, 518, 3]
        is_voxel = (getattr(self, 'model_variant', 'standard') == 'voxel')
        if is_voxel:
            depth_override = None
            voxel_depth_source = getattr(self, 'voxel_depth_source', 'pred')
            if voxel_depth_source == 'gt':
                depth_override = self._prepare_voxel_depth_override(
                    inputs.get('gt_depth', None),
                    image_list.shape[-2:],
                    image_list.device,
                    image_list.dtype,
                )
            if depth_override is not None:
                model_out = self.model(image_list, inputs['K'], inputs['c2e_extr'], depth_maps_override=depth_override)
            else:
                model_out = self.model(image_list, inputs['K'], inputs['c2e_extr'])
        else:
            model_out = self.model(image_list)

        return model_out, image_list, c2e_extr_list

    def get_recontrast_data_from_modelout(self, data_dict, model_out, c2e_extr_list, batch_idx=0):
        inputs = data_dict['context_frames']
        outputs = {}

        depth_maps    = model_out['depth_maps']
        forward_flow  = model_out['forward_flow']

        if self.enable_nan_checks:
            self._check_finite(depth_maps, "depth_maps")
            self._check_finite(forward_flow, "forward_flow")

        batch_size   = depth_maps.shape[0]
        frame_camrea = depth_maps.shape[1]

        c2e_extr_list = torch.stack(c2e_extr_list, dim=1)  # [B, S, 4, 4]
        bfc_depth_maps = rearrange(depth_maps.squeeze(-1), 'b c h w -> (b c) h w')
        bfc_K   = rearrange(inputs['K'],       'b c i j -> (b c) i j')
        bfc_c2e = rearrange(c2e_extr_list,     'b c i j -> (b c) i j')

        outputs['pred_depths']     = rearrange(bfc_depth_maps, '(b c) h w -> b (c h w)',
                                               b=batch_size, c=frame_camrea)
        outputs['pred_depth_maps'] = depth_maps.squeeze(-1)

        is_voxel = (getattr(self, 'model_variant', 'standard') == 'voxel')
        if is_voxel:
            # ── Voxel head 专用路径 ──────────────────────────────────────────
            # xyz 已是 ego 坐标系（voxel中心 + offset），SH 已在 ego 系，无需 rotate
            xyz_vox      = model_out['xyz']          # [B, N_max, K, 3]
            rot_vox      = model_out['rot_maps']      # [B, N_max*K, 4]
            scale_vox    = model_out['scale_maps']    # [B, N_max*K, 3]
            opacity_vox  = model_out['opacity_maps']  # [B, N_max*K, 1]
            sh_vox       = model_out['sh_maps']       # [B, N_max*K, d_sh, 3]
            voxel_mask   = model_out['voxel_mask']    # [B, N_max]

            if self.enable_nan_checks:
                self._check_finite(xyz_vox,     "xyz_vox")
                self._check_finite(rot_vox,     "rot_vox")
                self._check_finite(scale_vox,   "scale_vox")
                self._check_finite(opacity_vox, "opacity_vox")
                self._check_finite(sh_vox,      "sh_vox")

            K = xyz_vox.shape[2]
            # xyz: [B, N_max, K, 3] → [B, N_max*K, 3]
            outputs['xyz']          = xyz_vox.reshape(batch_size, -1, 3)
            outputs['rot_maps']     = rot_vox
            outputs['scale_maps']   = scale_vox
            outputs['opacity_maps'] = opacity_vox
            # sh_vox: [B, N_max*K, d_sh, 3]，rasterization 期望 [N, d_sh, 3]，直接存储
            outputs['sh_maps']      = sh_vox
            outputs['voxel_mask']   = voxel_mask
            outputs['ae_global_points'] = True  # 高斯是全场景共享的，渲染时不按相机分割

            gaussians_per_voxel = K

        else:
            # ── 标准 DPT head 路径（per-pixel Gaussians）────────────────────
            rot_maps    = model_out['rot_maps']
            scale_maps  = model_out['scale_maps']
            opacity_maps = model_out['opacity_maps']
            sh_maps     = model_out['sh_maps']
            offset_maps = model_out.get('offset_maps', None)

            if self.enable_nan_checks:
                self._check_finite(rot_maps,     "rot_maps")
                self._check_finite(scale_maps,   "scale_maps")
                self._check_finite(opacity_maps, "opacity_maps")
                self._check_finite(sh_maps,      "sh_maps")
                self._check_finite(offset_maps,  "offset_maps")

            bf_e2c = torch.linalg.inv(bfc_c2e)
            bfc_xyz = depth2pc(bfc_depth_maps, bf_e2c, bfc_K)

            gaussians_per_voxel = rot_maps.shape[-2] if rot_maps.dim() == 6 else 1
            if offset_maps is not None and offset_maps.dim() == 6:
                bfc_offset = rearrange(offset_maps, 'b c h w k d -> (b c) (h w) k d')
                bfc_xyz = bfc_xyz.unsqueeze(2) + bfc_offset
                bfc_xyz = bfc_xyz.reshape(bfc_xyz.shape[0], -1, 3)
            else:
                if offset_maps is not None:
                    bfc_offset = rearrange(offset_maps, 'b c h w d -> (b c) (h w) d')
                    bfc_xyz = bfc_xyz + bfc_offset
                if gaussians_per_voxel > 1:
                    bfc_xyz = bfc_xyz.unsqueeze(2).expand(-1, -1, gaussians_per_voxel, -1)
                    bfc_xyz = bfc_xyz.reshape(bfc_xyz.shape[0], -1, 3)
            if self.enable_nan_checks:
                self._check_finite(bfc_xyz, "bfc_xyz")

            if sh_maps.dim() == 7:
                bfc_sh = rearrange(sh_maps, 'b c h w k p d -> (b c) h w k p d')
                c2w_rotations = rearrange(bfc_c2e[:, :3, :3], "b i j -> b () () () () i j")
                bfc_sh = rotate_sh(bfc_sh, c2w_rotations)
            else:
                bfc_sh = rearrange(sh_maps, 'b c h w p d -> (b c) h w p d')
                c2w_rotations = rearrange(bfc_c2e[:, :3, :3], "b i j -> b () () () i j")
                bfc_sh = rotate_sh(bfc_sh, c2w_rotations)
            if self.enable_nan_checks:
                self._check_finite(bfc_sh, "bfc_sh")

            if rot_maps.dim() == 6:
                bfc_rot_maps = rearrange(rot_maps, 'b c h w k d -> (b c) (h w k) d', d=4)
            else:
                bfc_rot_maps = rearrange(rot_maps, 'b c h w d -> (b c) (h w) d', d=4)

            outputs['xyz']      = rearrange(bfc_xyz, '(b c) p k -> b (c p) k',
                                            b=batch_size, c=frame_camrea)
            outputs['rot_maps'] = rearrange(bfc_rot_maps, '(b c) p d -> b (c p) d',
                                            b=batch_size, c=frame_camrea, d=4)

            if scale_maps.dim() == 6:
                outputs['scale_maps']   = rearrange(scale_maps,   'b c h w k d -> b (c h w k) d', d=3)
                outputs['opacity_maps'] = rearrange(opacity_maps, 'b c h w k d -> b (c h w k) d')
                outputs['sh_maps']      = rearrange(bfc_sh, '(b c) h w k p d -> b (c h w k) d p',
                                                    b=batch_size, c=frame_camrea)
            else:
                outputs['scale_maps']   = rearrange(scale_maps,   'b c h w d -> b (c h w) d', d=3)
                outputs['opacity_maps'] = rearrange(opacity_maps, 'b c h w d -> b (c h w) d')
                outputs['sh_maps']      = rearrange(bfc_sh, '(b c) h w p d -> b (c h w) d p',
                                                    b=batch_size, c=frame_camrea)

        # Generate vehicle-based 3D velocity flow
        if self.use_vehicle_flow:
            new_forward_flow = []

            # ALWAYS compute vehicle masks using SAM2 for correct flow application
            # The masks are essential for applying velocity to the correct pixels
            compute_vehicle_masks = True  # Always compute for proper flow

            all_vehicle_masks = []

            for b in range(batch_size):
                batch_flows = []
                batch_masks = []  # Always collect masks for unified format

                for c in range(frame_camrea):
                    # Determine frame index and camera index
                    frame_idx = c // self.num_cams  # 0 for frame 0, 1 for frame context_span
                    cam_idx = c % self.num_cams

                    # Get image for segmentation
                    color_tensor = inputs.get(('color_aug', 0), torch.zeros(batch_size, frame_camrea, 3, self.height, self.width))
                    seg_img = color_tensor[b, c]

                    # Initialize outputs
                    vehicle_masks = []
                    vehicle_velocities = []

                    # Determine which frame-specific annotations to use
                    if frame_idx == 0:
                        anno_key = 'vehicle_annotations_frame_0'
                    else:
                        anno_key = f'vehicle_annotations_frame_{self.context_span}'

                    # Fallback to combined annotations if frame-specific not available
                    if anno_key not in inputs and 'vehicle_annotations' in inputs:
                        anno_key = 'vehicle_annotations'
                        lookup_idx = c
                    else:
                        lookup_idx = cam_idx

                    # Process vehicle annotations if available
                    if anno_key in inputs and b < len(inputs[anno_key]):
                        try:
                            batch_data = inputs[anno_key][b]

                            if lookup_idx < len(batch_data):
                                vehicle_data = batch_data[lookup_idx]

                                # Extract bounding boxes and velocities
                                bbox_2d_list = []
                                raw_velocities = []
                                vehicle_depths = []
                                vehicle_intrinsics = []

                                if isinstance(vehicle_data, list):
                                    for vehicle in vehicle_data:
                                        if isinstance(vehicle, dict) and 'bbox_2d' in vehicle:
                                            bbox_2d_list.append(vehicle['bbox_2d'])
                                            vel = vehicle.get('velocity', [0, 0, 0])
                                            # Ensure velocity is 3D
                                            if isinstance(vel, (list, tuple)) and len(vel) > 3:
                                                vel = vel[:3]
                                            raw_velocities.append(vel)
                                            vehicle_depths.append(vehicle.get('depth', 10.0))
                                            vehicle_intrinsics.append(vehicle.get('camera_intrinsic', None))

                                # Create masks using SAM2
                                vehicle_masks = self.segment_vehicles_with_sam2(seg_img, bbox_2d_list if bbox_2d_list else None)
                                vehicle_velocities = raw_velocities
                        except Exception:
                            pass  # Skip if annotations not accessible

                    # Ensure we have masks and velocities aligned
                    if len(vehicle_velocities) < len(vehicle_masks):
                        vehicle_velocities += [[0.0, 0.0, 0.0]] * (len(vehicle_masks) - len(vehicle_velocities))

                    # Compute 3D velocity flow
                    flow = self.compute_velocity_flow(
                        vehicle_masks, 
                        vehicle_velocities,
                        (self.height, self.width)
                    )

                    batch_flows.append(torch.from_numpy(flow).to(depth_maps.device))

                    # Store combined mask for inference (always needed)
                    combined_mask = np.zeros((self.height, self.width), dtype=bool)
                    for mask in vehicle_masks:
                        if mask is not None:
                            combined_mask |= mask
                    batch_masks.append(torch.from_numpy(combined_mask).to(depth_maps.device))

                new_forward_flow.append(torch.stack(batch_flows))
                if len(batch_masks) > 0:
                    all_vehicle_masks.append(torch.stack(batch_masks))
                else:
                    # Create empty masks if no masks were collected
                    empty_masks = torch.zeros(frame_camrea, self.height, self.width, dtype=torch.bool, device=depth_maps.device)
                    all_vehicle_masks.append(empty_masks)

            flow_tensor = torch.stack(new_forward_flow)
            if not is_voxel and gaussians_per_voxel > 1:
                flow_tensor = flow_tensor.unsqueeze(4).expand(-1, -1, -1, -1, gaussians_per_voxel, -1)
                outputs['forward_flow'] = rearrange(flow_tensor, 'b c h w k d -> b (c h w k) d')
            elif not is_voxel:
                outputs['forward_flow'] = rearrange(flow_tensor, 'b c h w d -> b (c h w) d')
            else:
                outputs['forward_flow'] = flow_tensor
            # Store vehicle masks as tensor for unified format [b, c, h, w]
            if len(all_vehicle_masks) > 0:
                outputs['vehicle_masks'] = torch.stack(all_vehicle_masks)
            else:
                outputs['vehicle_masks'] = None

        else:
            # Use original flow from model
            if is_voxel:
                # voxel 路径：forward_flow 是 [B, V, H, W, 3]，保持原格式供 loss 使用
                outputs['forward_flow'] = forward_flow
            elif gaussians_per_voxel > 1:
                flow_tensor = forward_flow.unsqueeze(4).expand(-1, -1, -1, -1, gaussians_per_voxel, -1)
                outputs['forward_flow'] = rearrange(flow_tensor, 'b c h w k d -> b (c h w k) d')
            else:
                outputs['forward_flow'] = rearrange(forward_flow, 'b c h w d -> b (c h w) d')
            outputs['vehicle_masks'] = None  # No vehicle masks when not using vehicle flow

        # Perform ICP refinement early if we have multiple frames (needed for ego pose and velocity refinement)
        ego_T_ego_key = ('ego_T_ego', 0, self.context_span)
        if frame_camrea > self.num_cams and ego_T_ego_key in inputs:
            # Get ego_T_ego transformations
            ego_T_ego_0toN_initial = inputs[ego_T_ego_key]

            # Store the transformation being used (no refinement)
            outputs['ego_T_ego_original'] = ego_T_ego_0toN_initial
        
        if frame_camrea > self.num_cams:
            num_frames = frame_camrea // self.num_cams
            if num_frames != 2:
                raise NotImplementedError(f"Context frames should have exactly 2 frames (frame 0 and frame {self.context_span}), but got {num_frames} frames")

            xyz_transformed = outputs['xyz'].clone()
            if self.translate_3dgs:
                rot_maps_transformed = outputs['rot_maps'].clone()
                sh_maps_transformed = outputs['sh_maps'].clone()
            mid_point = xyz_transformed.shape[1] // 2


            # Check the original input to see if we have per-camera transformations
            ego_T_ego_key = ('ego_T_ego', 0, self.context_span)
            ego_T_ego_0toN_input = inputs.get(ego_T_ego_key, None)
            if ego_T_ego_0toN_input is not None:
                # Check if original input has per-camera transformations
                if ego_T_ego_0toN_input.dim() == 4:
                    # [batch_size, num_cameras, 4, 4] - Use camera-specific transformations
                    batch_size = xyz_transformed.shape[0]
                    points_per_camera = self.height * self.width

                    # Check if we have refined per-camera transformations
                    use_refined = 'ego_T_ego_refined' in outputs
                    if use_refined:
                        refined_transforms = outputs['ego_T_ego_refined']

                    # Transform each camera's frame N points separately
                    for cam_id in range(self.num_cams):
                        cam_start = mid_point + cam_id * points_per_camera
                        cam_end = mid_point + (cam_id + 1) * points_per_camera

                        # Get camera-specific transformation: ego0 → egoN
                        # Use refined transformation if available
                        if use_refined:
                            ego_T_ego_0toN_cam = refined_transforms[:, cam_id]
                        else:
                            ego_T_ego_0toN_cam = ego_T_ego_0toN_input[:, cam_id]  # [batch_size, 4, 4]

                        # Invert to get: egoN → ego0
                        ego_T_ego_Nto0_cam = torch.linalg.inv(ego_T_ego_0toN_cam)

                        # Transform this camera's frame N points to frame 0 ego coordinates
                        xyz_transformed[:, cam_start:cam_end, :] = self.transform_points(
                            xyz_transformed[:, cam_start:cam_end, :],
                            ego_T_ego_Nto0_cam
                        )
                    if self.translate_3dgs:
                        if ego_T_ego_Nto0_cam.dim() == 2:
                            R_Nto0 = ego_T_ego_Nto0_cam[:3, :3]  # [3, 3]
                        else:
                            R_Nto0 = ego_T_ego_Nto0_cam[:, :3, :3]  # [B, 3, 3]

                        try:
                            from pytorch3d.transforms import matrix_to_quaternion
                            q_Nto0 = matrix_to_quaternion(torch.linalg.inv(R_Nto0)) # [3,] or [B, 4]
                        except ImportError:
                            def matrix_to_quaternion_manual(R):
                                # R: [..., 3, 3]
                                tr = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
                                qw = torch.sqrt((tr + 1.0).clamp(min=0)) / 2.0
                                qx = (R[..., 2, 1] - R[..., 1, 2]) / (4 * qw + 1e-8)
                                qy = (R[..., 0, 2] - R[..., 2, 0]) / (4 * qw + 1e-8)
                                qz = (R[..., 1, 0] - R[..., 0, 1]) / (4 * qw + 1e-8)
                                return torch.stack([qw, qx, qy, qz], dim=-1)
                            q_Nto0 = matrix_to_quaternion_manual(R_Nto0) 

                        if rot_maps_transformed[:, cam_start:cam_end, :].dim() == 2:
                            # [N, 4]
                            if q_Nto0.dim() == 1:
                                q_Nto0 = q_Nto0.unsqueeze(0)  # [1, 4]
                            q_Nto0 = q_Nto0.expand(rot_maps_transformed[:, cam_start:cam_end, :].shape[0], -1)  # [N, 4]
                        else:
                            # [B, N, 4]
                            if q_Nto0.dim() == 1:
                                q_Nto0 = q_Nto0.unsqueeze(0).unsqueeze(0)  # [1, 1, 4]
                            elif q_Nto0.dim() == 2:
                                q_Nto0 = q_Nto0.unsqueeze(1)  # [B, 1, 4]
                            q_Nto0 = q_Nto0.expand(-1, rot_maps_transformed[:, cam_start:cam_end, :].shape[1], -1)  # [B, N, 4]

                        def quat_multiply(q1, q2):
                            w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
                            w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
                            w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
                            x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
                            y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
                            z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
                            return torch.stack([w, x, y, z], dim=-1)

                        rot_maps_slice = rot_maps_transformed[:, cam_start:cam_end, :].clone()
                        rot_maps_transformed_slice = quat_multiply(q_Nto0, rot_maps_slice)
                        rot_maps_transformed = torch.cat([
                            rot_maps_transformed[:, :cam_start, :],
                            rot_maps_transformed_slice,
                            rot_maps_transformed[:, cam_end:, :]
                        ], dim=1)

                        outputs['rot_maps_transformed'] = rot_maps_transformed
                        outputs['sh_maps_transformed'] = sh_maps_transformed
                    
                    outputs['xyz_transformed'] = xyz_transformed
                else:
                    # [batch_size, 4, 4] or [4, 4] - Use unified transformation for all cameras
                    # Prefer refined transformation if available
                    if 'ego_T_ego_refined' in outputs:
                        refined = outputs['ego_T_ego_refined']
                        # If refined is per-camera [batch_size, num_cameras, 4, 4], use camera 0
                        if refined.dim() == 4:
                            ego_T_ego_0toN = refined[:, 0]  # Use camera 0's refined transformation
                        else:
                            ego_T_ego_0toN = refined
                    else:
                        ego_T_ego_0toN = ego_T_ego_0toN_input

                    # Ensure batch dimension exists
                    if ego_T_ego_0toN.dim() == 2:
                        ego_T_ego_0toN = ego_T_ego_0toN.unsqueeze(0)
                    elif ego_T_ego_0toN.dim() == 3 and ego_T_ego_0toN.shape[0] == 1:
                        # Already has batch dimension of 1, expand to match batch size
                        batch_size = xyz_transformed.shape[0]
                        if batch_size > 1:
                            ego_T_ego_0toN = ego_T_ego_0toN.expand(batch_size, -1, -1)

                    ego_T_ego_Nto0 = torch.linalg.inv(ego_T_ego_0toN)
                    xyz_transformed[:, mid_point:, :] = self.transform_points(
                        xyz_transformed[:, mid_point:, :],
                        ego_T_ego_Nto0
                    )
                    outputs['xyz_transformed'] = xyz_transformed
            else:
                outputs['xyz_transformed'] = outputs['xyz']
                outputs['rot_maps_transformed'] = outputs['rot_maps']
                outputs['sh_maps_transformed'] = outputs['sh_maps']
        else:
            outputs['xyz_transformed'] = outputs['xyz']
            outputs['rot_maps_transformed'] = outputs['rot_maps']
            outputs['sh_maps_transformed'] = outputs['sh_maps']

        del bfc_K, bfc_c2e, bfc_depth_maps
        if not is_voxel:
            del bfc_xyz, rot_maps, scale_maps, opacity_maps, bfc_sh
        return outputs
