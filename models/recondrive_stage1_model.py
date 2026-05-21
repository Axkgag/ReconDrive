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


class ReconDriveStage1_LITModelModule(ReconDrive_LITModelModule):
    """Stage1 single-frame 3D Gaussian training module."""

    def __init__(self, cfg, save_dir='.', logger=None):
        super().__init__(cfg, save_dir, logger)
        self._configure_stage1_trainable()

        # Visualization settings
        self.vis_step = 0
        self._val_vis_saved_this_epoch = 0

    def _configure_stage1_trainable(self):
        # Freeze all parameters first.
        for param in self.model.parameters():
            param.requires_grad = False

        # Unfreeze depth_head and gs_head.
        for param in self.model.depth_head.parameters():
            param.requires_grad = True
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
            "Stage1 trainable parameters:",
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

        batch_recontrast_data = self.get_recontrast_data(batch_input, batch_idx)
        loss_norm = self.compute_norm_loss(batch_recontrast_data)

        batch_render_data = self.get_render_data(batch_input)
        batch_splating_data = self.render_splating_imgs(
            {**batch_recontrast_data, 'ae_global_points': True}, batch_render_data
        )

        loss_gaussian = self.compute_gaussian_loss(batch_splating_data)
        loss_depth = self.compute_depth_loss(batch_splating_data)

        self.log(f'{stage}/gs', loss_gaussian.item(), on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log(f'{stage}/norm', loss_norm.item(), on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log(f'{stage}/depth', loss_depth.item(), on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)

        loss_all = loss_gaussian + loss_norm + loss_depth
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
        loss_gaussian = self.compute_gaussian_loss(batch_splating_data)

        self.log(f'{stage}/gs', loss_gaussian.item(), on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log(f'{stage}/norm', loss_norm.item(), on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)

        loss_all = loss_gaussian + loss_norm
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
        loss_gaussian = self.compute_gaussian_loss(batch_splating_data)

        self.log(f'{stage}/gs', loss_gaussian.item(), on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log(f'{stage}/norm', loss_norm.item(), on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)

        loss_all = loss_gaussian + loss_norm
        psnr, ssim, lpips = self.compute_reconstruction_metrics(batch_splating_data, stage)

        del batch_input, batch_recontrast_data, batch_render_data, batch_splating_data
        del psnr, ssim, lpips

        return loss_all

    def save_validation_step_images(self, batch_idx, batch_splating_data, batch_recontrast_data):
        """
        Save composite figure for Stage1 training visualization.

        Rows: one per camera
        Columns: GT Image | Stage1 Render | Stage1 3D Gauss
        """
        from pytorch_lightning.utilities import rank_zero_only

        @rank_zero_only
        def _save():
            # Debug: Print Gaussian statistics per camera
            if self.vis_step == 0:  # Only print on first visualization
                self._debug_print_gaussian_stats(batch_recontrast_data)

            vis_subdir = 'train_visualizations' if self.stage == 'train' else 'val_visualizations'
            save_dir = os.path.join(
                self.save_dir, vis_subdir, f'epoch_{self.current_epoch:04d}'
            )
            os.makedirs(save_dir, exist_ok=True)

            frame_id = 0
            gt_imgs, stage1_imgs = [], []
            for cam_id in range(self.num_cams):
                pred_key = ('gaussian_color', frame_id, cam_id)
                gt_key   = ('groudtruth',     frame_id, cam_id)

                if pred_key not in batch_splating_data or gt_key not in batch_splating_data:
                    return

                gt_imgs.append(batch_splating_data[gt_key][0])
                stage1_imgs.append(batch_splating_data[pred_key][0])

            stage1_3d_img = self._render_gaussian_scene_image(batch_recontrast_data, batch_idx_in_batch=0)

            fig, axes = plt.subplots(
                nrows=self.num_cams, ncols=3,
                figsize=(3 * 4, self.num_cams * 2.3),
                dpi=120,
            )
            col_titles = ['GT Image', 'Stage1 Render', 'Stage1 3D Gauss']

            for cam_id in range(self.num_cams):
                cam_name = (self.camera_names[cam_id]
                            if cam_id < len(self.camera_names) else f'CAM_{cam_id}')

                axes[cam_id, 0].imshow(self._tensor_to_uint8(gt_imgs[cam_id]))
                axes[cam_id, 1].imshow(self._tensor_to_uint8(stage1_imgs[cam_id]))
                axes[cam_id, 2].imshow(stage1_3d_img)

                axes[cam_id, 0].set_ylabel(cam_name, fontsize=9)
                for c in range(3):
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
