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

        # 新增：如果启用了 Occ，解冻 Occ 解码器参数
        if getattr(self.model.gs_head, 'enable_occ', False):
            for param in self.model.gs_head.occ_decoder.parameters():
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
        occ_params = 0
        if getattr(self.model.gs_head, 'enable_occ', False):
            occ_params = sum(p.numel() for p in self.model.gs_head.occ_decoder.parameters() if p.requires_grad)
        print(
            "Stage1 可训练参数:",
            f"total={total_params:,}, trainable={trainable_params:,},",
            f"lora={lora_params:,}, depth_head={depth_params:,}, gs_head={gs_params:,}, occ_decoder={occ_params:,}",
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

        # 新增：计算 Occ 损失（仅在启用时）
        if getattr(self.model.gs_head, 'enable_occ', False):
            loss_occ = self.compute_occ_loss(batch_recontrast_data, batch_input)
            self.log(f'{stage}/occ', loss_occ.item(), on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        else:
            loss_occ = torch.tensor(0.0, device=self.device)

        self.log(f'{stage}/gs', loss_gaussian.item(), on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log(f'{stage}/norm', loss_norm.item(), on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log(f'{stage}/depth', loss_depth.item(), on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)

        loss_all = loss_gaussian + loss_norm + loss_depth + loss_occ
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

                # 新增：如果启用了 Occ 监督，添加 Occ 对比图
                if getattr(self.model.gs_head, 'enable_occ', False) and 'occ_logits' in batch_recontrast_data:
                    occ_gt = batch_input.get('context_frames', {}).get('occ_semantics', None)
                    if occ_gt is not None:
                        if isinstance(occ_gt, (list, tuple)):
                            occ_gt = occ_gt[0]
                        if occ_gt is not None and not isinstance(occ_gt, torch.Tensor):
                            occ_gt = torch.from_numpy(occ_gt).to(self.device)

                        if occ_gt is not None:
                            occ_vis_path = os.path.join(
                                self.save_dir, 'vis',
                                f'train_occ_step{self.vis_step:06d}.png'
                            )
                            os.makedirs(os.path.dirname(occ_vis_path), exist_ok=True)
                            self.visualize_occ_comparison(
                                batch_recontrast_data['occ_logits'],
                                occ_gt,
                                occ_vis_path
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
            gt_imgs, stage1_imgs, pred_depth_imgs = [], [], []

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

                if pred_key not in batch_splating_data or gt_key not in batch_splating_data:
                    return

                gt_imgs.append(batch_splating_data[gt_key][0])
                stage1_imgs.append(batch_splating_data[pred_key][0])

                if pred_depths_all is not None:
                    pred_depth_imgs.append(pred_depths_all[cam_id].detach().cpu().float().numpy())
                else:
                    pred_depth_imgs.append(None)

            stage1_3d_img = self._render_gaussian_scene_image(batch_recontrast_data, batch_idx_in_batch=0)

            fig, axes = plt.subplots(
                nrows=self.num_cams, ncols=4,
                figsize=(4 * 4, self.num_cams * 2.3),
                dpi=120,
            )
            col_titles = ['GT Image', 'Stage1 Render', 'Pred Depth', 'Stage1 3D Gauss']

            for cam_id in range(self.num_cams):
                cam_name = (self.camera_names[cam_id]
                            if cam_id < len(self.camera_names) else f'CAM_{cam_id}')

                axes[cam_id, 0].imshow(self._tensor_to_uint8(gt_imgs[cam_id]))
                axes[cam_id, 1].imshow(self._tensor_to_uint8(stage1_imgs[cam_id]))

                depth_np = pred_depth_imgs[cam_id]
                if depth_np is not None:
                    axes[cam_id, 2].imshow(depth_np, cmap='magma')
                else:
                    axes[cam_id, 2].text(0.5, 0.5, 'N/A', ha='center', va='center',
                                         transform=axes[cam_id, 2].transAxes)

                axes[cam_id, 3].imshow(stage1_3d_img)

                axes[cam_id, 0].set_ylabel(cam_name, fontsize=9)
                for c in range(4):
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

    def visualize_occ_comparison(self, occ_logits, occ_gt, save_path):
        """
        本地复现 COME/tools/vis_utils.draw 的可视化风格（不依赖 mayavi），生成与参考实现相似的 3D 占据可视化。

        Args:
            occ_logits: [B, S, H, W, D, C] 或 [B, H, W, D, C] 预测的 logits
            occ_gt: [B, H, W, D] 真值标签
            save_path: 最终合并图片保存路径
        """
        import os
        import time
        import numpy as np
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        # 取第一个样本并得到 [H, W, D] 类别格子
        if occ_logits.dim() == 6:  # [B, S, H, W, D, C]
            pred = occ_logits[0, 0].argmax(dim=-1).cpu().numpy()
        else:  # [B, H, W, D, C]
            pred = occ_logits[0].argmax(dim=-1).cpu().numpy()
        gt = occ_gt[0].cpu().numpy()  # [H, W, D]

        # vis_utils 中使用的颜色表（已复刻）
        colors = np.array(
            [
                [255, 120,  50, 255],       # barrier              orange
                [255, 192, 203, 255],       # bicycle              pink
                [255, 255,   0, 255],       # bus                  yellow
                [  0, 150, 245, 255],       # car                  blue
                [  0, 255, 255, 255],       # construction_vehicle cyan
                [255, 127,   0, 255],       # motorcycle           dark orange
                [255,   0,   0, 255],       # pedestrian           red
                [255, 240, 150, 255],       # traffic_cone         light yellow
                [135,  60,   0, 255],       # trailer              brown
                [160,  32, 240, 255],       # truck                purple                
                [255,   0, 255, 255],       # driveable_surface    dark pink
                [139, 137, 137, 255],
                [ 75,   0,  75, 255],       # sidewalk             dard purple
                [150, 240,  80, 255],       # terrain              light green          
                [230, 230, 250, 255],       # manmade              white
                [  0, 175,   0, 255],       # vegetation           green
            ]
        ).astype(np.uint8)
        # normalize colors to [0,1]
        cmap_rgba = (colors[:, :3].astype(np.float32) / 255.0)

        # parameters matching vis_utils
        vox_origin = np.array([-40.0, -40.0, -1.0], dtype=np.float32)
        voxel_size = float(0.4)

        def get_grid_coords(dims, resolution):
            g_xx = np.arange(0, dims[0])
            g_yy = np.arange(0, dims[1])
            g_zz = np.arange(0, dims[2])
            xx, yy, zz = np.meshgrid(g_xx, g_yy, g_zz)
            coords_grid = np.array([xx.flatten(), yy.flatten(), zz.flatten()]).T.astype(np.float32)
            resolution = np.array([resolution, resolution, resolution], dtype=np.float32).reshape([1, 3])
            coords_grid = (coords_grid * resolution) + resolution / 2.0
            return coords_grid

        # vis_utils 使用 voxels.shape -> w,h,z
        # 我们的 pred/gt 是 [H, W, D]，直接用其形状
        H, W, D = pred.shape
        # compute voxel centers in world coords
        grid_coords = get_grid_coords([H, W, D], voxel_size) + vox_origin.reshape([1, 3])  # (N,3)

        # flatten labels consistent with grid_coords ordering
        labels_pred = pred.flatten()
        labels_gt = gt.flatten()

        # Optionally insert ego car markers like vis_utils.show_ego
        try:
            show_ego = True
            if show_ego and pred.shape == (200, 200, 16):
                pred = pred.copy()
                pred[96:104, 96:104, 2:7] = 15
                pred[104:106, 96:104, 2:5] = 3
                labels_pred = pred.flatten()
            if show_ego and gt.shape == (200, 200, 16):
                gt = gt.copy()
                gt[96:104, 96:104, 2:7] = 15
                gt[104:106, 96:104, 2:5] = 3
                labels_gt = gt.flatten()
        except Exception:
            pass

        # select voxels inside [1,16] like vis_utils: (v>0) & (v<17)
        mask_pred = (labels_pred > 0) & (labels_pred < 17)
        mask_gt = (labels_gt > 0) & (labels_gt < 17)

        coords_pred = grid_coords[mask_pred]
        coords_gt = grid_coords[mask_gt]
        labels_pred_sel = labels_pred[mask_pred].astype(np.int32) - 1  # 0-based index
        labels_gt_sel = labels_gt[mask_gt].astype(np.int32) - 1

        # limit points
        max_points = 100000
        if coords_pred.shape[0] > max_points:
            idx = np.random.choice(coords_pred.shape[0], max_points, replace=False)
            coords_pred = coords_pred[idx]
            labels_pred_sel = labels_pred_sel[idx]
        if coords_gt.shape[0] > max_points:
            idx = np.random.choice(coords_gt.shape[0], max_points, replace=False)
            coords_gt = coords_gt[idx]
            labels_gt_sel = labels_gt_sel[idx]

        # map labels to colors
        def map_colors(label_inds):
            if label_inds.size == 0:
                return np.zeros((0, 4))
            cols = cmap_rgba[label_inds % cmap_rgba.shape[0]]
            return cols

        cols_pred = map_colors(labels_pred_sel)
        cols_gt = map_colors(labels_gt_sel)

        # create figure with white background and two 3D subplots
        fig = plt.figure(figsize=(14, 6), dpi=120)
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
