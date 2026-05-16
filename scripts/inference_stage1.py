#!/usr/bin/env python3
"""
Stage 1 inference script for ReconDrive.
Loads the stage1 checkpoint (DINO+VGGT), generates per-pixel Gaussians from
multi-camera images, renders them back to each camera view, and saves:
  - GT vs. rendered comparison grids (per camera)
  - 3D Gaussian scene visualisation as a PNG (matplotlib 3D scatter)
"""

import argparse
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import numpy as np
import torch
import yaml
from PIL import Image
from torch.utils.data import DataLoader

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from dataset.vggt3dgs_scene_data_module import VGGT3DGS_SceneDataModule
from dataset.vggt4dgs_scene_dataset import custom_collate_fn
from models.recondrive_model import ReconDrive_LITModelModule

CAMERA_NAMES = [
    'CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT',
    'CAM_BACK_LEFT', 'CAM_BACK_RIGHT', 'CAM_BACK',
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def to_device(data, device):
    if isinstance(data, dict):
        return {k: (v if k == 'vehicle_annotations' else to_device(v, device))
                for k, v in data.items()}
    elif isinstance(data, (list, tuple)):
        return type(data)(to_device(x, device) for x in data)
    elif torch.is_tensor(data):
        return data.to(device)
    return data


def tensor_to_uint8(t):
    """[C,H,W] float tensor in [0,1] → uint8 numpy [H,W,C]."""
    arr = t.detach().cpu().float().clamp(0, 1).numpy()
    return (arr.transpose(1, 2, 0) * 255).astype(np.uint8)


def save_comparison_grid(gt_imgs, pred_imgs, save_path, cam_names=None):
    """
    Save a 2-row grid: top row = GT, bottom row = rendered.
    gt_imgs / pred_imgs: list of [C,H,W] tensors, one per camera.
    """
    rows = []
    for label, imgs in [('GT', gt_imgs), ('Rendered', pred_imgs)]:
        row_imgs = [tensor_to_uint8(img) for img in imgs]
        row = np.concatenate(row_imgs, axis=1)  # concat along width
        rows.append(row)

    grid = np.concatenate(rows, axis=0)  # stack vertically

    # Add thin separator line between rows
    h_per_row = rows[0].shape[0]
    sep = np.full((4, grid.shape[1], 3), 200, dtype=np.uint8)
    grid = np.concatenate([rows[0], sep, rows[1]], axis=0)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    Image.fromarray(grid).save(save_path)


def save_per_camera(gt_imgs, pred_imgs, save_dir, scene_name, sample_idx, cam_names):
    """Save individual GT and rendered images per camera."""
    for cam_id, (gt, pred) in enumerate(zip(gt_imgs, pred_imgs)):
        cam = cam_names[cam_id] if cam_names else f'cam{cam_id}'
        base = os.path.join(save_dir, scene_name, f'sample_{sample_idx:04d}', cam)
        os.makedirs(base, exist_ok=True)
        Image.fromarray(tensor_to_uint8(gt)).save(os.path.join(base, 'gt.png'))
        Image.fromarray(tensor_to_uint8(pred)).save(os.path.join(base, 'rendered.png'))


# ---------------------------------------------------------------------------
# 3D Gaussian scene visualisation
# ---------------------------------------------------------------------------

def save_gaussian_scene_image(recontrast_data, save_path, batch_idx=0,
                               opacity_thresh=0.1, max_points=80_000,
                               elev=25, azim=-60):
    """
    Render a 3D scatter plot of Gaussian positions and save as PNG.

    Points are colored by their SH DC term (view-independent base color).
    Opacity is used as point alpha so low-confidence Gaussians are faint.

    Args:
        elev: elevation angle of the viewpoint (degrees)
        azim: azimuth angle of the viewpoint (degrees)
    """
    xyz     = recontrast_data['xyz'][batch_idx].cpu().float().numpy()          # [N, 3]
    opacity = recontrast_data['opacity_maps'][batch_idx].cpu().float().numpy().squeeze(-1)  # [N]
    sh      = recontrast_data['sh_maps'][batch_idx].cpu().float().numpy()      # [N, 25, 3]

    # Filter by opacity
    mask = opacity > opacity_thresh
    xyz, opacity, sh = xyz[mask], opacity[mask], sh[mask]

    # DC term → RGB  (C0 = 1 / (2*sqrt(pi)))
    C0 = 0.28209479177387814
    rgb = np.clip(sh[:, 0, :] / C0 * 0.5 + 0.5, 0.0, 1.0)  # [N, 3]

    # Subsample for rendering speed
    if len(xyz) > max_points:
        idx = np.random.choice(len(xyz), max_points, replace=False)
        xyz, rgb, opacity = xyz[idx], rgb[idx], opacity[idx]

    # Clip extreme depth outliers (keep 1st–99th percentile on each axis)
    for axis in range(3):
        lo, hi = np.percentile(xyz[:, axis], [1, 99])
        mask = (xyz[:, axis] >= lo) & (xyz[:, axis] <= hi)
        xyz, rgb, opacity = xyz[mask], rgb[mask], opacity[mask]

    # RGBA colors: use opacity as alpha
    alpha = np.clip(opacity, 0.05, 1.0)
    rgba  = np.concatenate([rgb, alpha[:, None]], axis=1)

    fig = plt.figure(figsize=(14, 10), dpi=150)
    ax  = fig.add_subplot(111, projection='3d')

    # nuScenes ego frame: X=forward, Y=left, Z=up
    ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2],
               c=rgba, s=0.3, linewidths=0, depthshade=True)

    ax.set_xlabel('X (forward, m)', fontsize=9)
    ax.set_ylabel('Y (left, m)',    fontsize=9)
    ax.set_zlabel('Z (up, m)',      fontsize=9)
    ax.set_title(f'3D Gaussian Scene  ({len(xyz):,} points)', fontsize=11)
    ax.view_init(elev=elev, azim=azim)

    # Equal aspect ratio
    ranges = np.array([[xyz[:, i].min(), xyz[:, i].max()] for i in range(3)])
    max_range = (ranges[:, 1] - ranges[:, 0]).max() / 2
    mid = ranges.mean(axis=1)
    ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
    ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
    ax.set_zlim(mid[2] - max_range, mid[2] + max_range)

    ax.tick_params(labelsize=7)
    fig.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved 3D scene: {save_path}  ({len(xyz):,} points)")

def run_stage1_inference(model, scene_dataloader, device, output_dir,
                         max_samples_per_scene=None, save_glb=True,
                         glb_opacity_thresh=0.1, glb_max_points=80_000,
                         scene3d_elev=25, scene3d_azim=-60):
    """
    For each scene, run predict_step and save:
      - GT vs. rendered comparison grids
      - 3D Gaussian scene PNG from a fixed viewpoint
    Only frame 0 (scene reconstruction) is visualised — that is the Stage 1 task.
    """
    os.makedirs(output_dir, exist_ok=True)
    num_cams = getattr(model, 'num_cams', 6)

    all_psnr = []

    with torch.no_grad():
        for scene_idx, scene_batch in enumerate(scene_dataloader):
            scene_name = scene_batch['scene_name']
            print(f"\n[Scene {scene_idx+1}/{len(scene_dataloader)}] {scene_name}")

            # Build a per-sample dataloader from the scene
            if 'samples' in scene_batch:
                from scripts.inference import SceneSampleDataset
                samples = scene_batch['samples']
                if max_samples_per_scene:
                    samples = samples[:max_samples_per_scene]
                sample_ds = SceneSampleDataset(samples)
            else:
                from scripts.inference import SceneSampleDataset
                indices = scene_batch['sample_indices']
                if max_samples_per_scene:
                    indices = indices[:max_samples_per_scene]
                sample_ds = SceneSampleDataset(
                    indices,
                    dataset=scene_batch['dataset'],
                    scene_idx=scene_batch['scene_idx'],
                )

            loader = DataLoader(
                sample_ds, batch_size=1, shuffle=False,
                num_workers=0, collate_fn=custom_collate_fn,
            )

            for sample_idx, batch in enumerate(loader):
                batch['scene_idx'] = scene_idx
                batch = to_device(batch, device)

                # predict_step returns (recontrast_data, render_data, splating_data)
                recontrast_data, render_data, splating_data = model.predict_step(batch, sample_idx)

                # Collect frame-0 GT and rendered images for all cameras
                gt_imgs, pred_imgs = [], []
                frame_psnr = []

                for cam_id in range(num_cams):
                    pred_key = ('gaussian_color', 0, cam_id)
                    gt_key   = ('groudtruth',     0, cam_id)

                    if pred_key not in splating_data or gt_key not in splating_data:
                        continue

                    pred = splating_data[pred_key][0].clamp(0, 1)   # [C,H,W]
                    gt   = splating_data[gt_key][0].clamp(0, 1)

                    gt_imgs.append(gt)
                    pred_imgs.append(pred)

                    # PSNR
                    mse = ((pred - gt) ** 2).mean().item()
                    psnr = -10 * np.log10(mse + 1e-8)
                    frame_psnr.append(psnr)

                if not gt_imgs:
                    print(f"  sample {sample_idx}: no frame-0 data, skipping")
                    continue

                mean_psnr = np.mean(frame_psnr)
                all_psnr.append(mean_psnr)
                print(f"  sample {sample_idx:03d}  PSNR={mean_psnr:.2f} dB  "
                      f"({', '.join(f'{p:.1f}' for p in frame_psnr)})")

                # Save comparison grid (all cameras side by side)
                grid_path = os.path.join(
                    output_dir, scene_name, f'sample_{sample_idx:04d}_grid.png')
                save_comparison_grid(gt_imgs, pred_imgs, grid_path, CAMERA_NAMES)

                # Save individual per-camera images
                save_per_camera(gt_imgs, pred_imgs, output_dir, scene_name,
                                sample_idx, CAMERA_NAMES)

                # 3D Gaussian scene → PNG
                if save_glb:
                    scene_path = os.path.join(
                        output_dir, scene_name, f'sample_{sample_idx:04d}_scene3d.png')
                    save_gaussian_scene_image(recontrast_data, scene_path,
                                              opacity_thresh=glb_opacity_thresh,
                                              max_points=glb_max_points,
                                              elev=scene3d_elev,
                                              azim=scene3d_azim)

    if all_psnr:
        print(f"\n{'='*50}")
        print(f"Overall mean PSNR: {np.mean(all_psnr):.4f} dB  ({len(all_psnr)} samples)")
        print(f"Results saved to: {output_dir}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Stage 1 Gaussian rendering visualisation')
    parser.add_argument('--cfg_path',    type=str, required=True)
    parser.add_argument('--ckpt',        type=str, default='./checkpoints/recondrive_stage1.ckpt')
    parser.add_argument('--output_dir',  type=str, default='./work_dirs/stage1_vis')
    parser.add_argument('--device',      type=str, default='0')
    parser.add_argument('--max_scenes',  type=int, default=None, help='Limit number of scenes')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Max samples per scene (default: all)')
    parser.add_argument('--no_scene3d',  action='store_true',
                        help='Skip saving 3D Gaussian scene PNG')
    parser.add_argument('--scene3d_opacity_thresh', type=float, default=0.1,
                        help='Opacity threshold for 3D scene visualisation (default: 0.1)')
    parser.add_argument('--scene3d_max_points', type=int, default=80_000,
                        help='Max points in 3D scene PNG (default: 80000)')
    parser.add_argument('--scene3d_elev', type=float, default=25,
                        help='Elevation angle for 3D viewpoint in degrees (default: 25)')
    parser.add_argument('--scene3d_azim', type=float, default=-60,
                        help='Azimuth angle for 3D viewpoint in degrees (default: -60)')
    args = parser.parse_args()

    # Device
    device = f'cuda:{args.device}' if not args.device.startswith('cuda') else args.device

    # Config
    with open(args.cfg_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    config['data_cfg']['batch_size'] = 1
    config['model_cfg']['batch_size'] = 1
    if 'context_span' in config['data_cfg']:
        config['model_cfg']['context_span'] = config['data_cfg']['context_span']
    if 'nuscenes_version' in config['data_cfg']:
        config['model_cfg']['nuscenes_version'] = config['data_cfg']['nuscenes_version']

    # Data
    print("Loading dataset...")
    data_module = VGGT3DGS_SceneDataModule(cfg=config['data_cfg'])
    data_module.setup(stage='test')
    scene_dataloader = data_module.test_scene_dataloader()

    if args.max_scenes:
        scene_list = []
        for i, sb in enumerate(scene_dataloader):
            if i >= args.max_scenes:
                break
            scene_list.append(sb)
        scene_dataloader = scene_list
        print(f"Limited to {len(scene_dataloader)} scenes")

    # Model
    print(f"Loading model from: {args.ckpt}")
    model = ReconDrive_LITModelModule(cfg=config['model_cfg'], save_dir='./temp_log', logger=None)
    model.load_pretrained_checkpoint(args.ckpt)
    model.to(device)
    model.eval()

    # Inference
    run_stage1_inference(model, scene_dataloader, device, args.output_dir,
                         max_samples_per_scene=args.max_samples,
                         save_glb=not args.no_scene3d,
                         glb_opacity_thresh=args.scene3d_opacity_thresh,
                         glb_max_points=args.scene3d_max_points,
                         scene3d_elev=args.scene3d_elev,
                         scene3d_azim=args.scene3d_azim)


if __name__ == '__main__':
    main()
