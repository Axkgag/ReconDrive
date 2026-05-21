#!/usr/bin/env python3
"""
Validate AE input (stage1 Gaussians) by rendering images and 3D Gaussian views.

This script:
1) Loads one scene sample.
2) Runs stage1 to get Gaussians (recontrast_data).
3) Drops Gaussians outside AE voxel range.
4) Renders per-camera images and a 3D Gaussian scatter.
"""

import argparse
import json
import sys
from pathlib import Path

import io
import numpy as np
import torch
import yaml

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.utils as vutils

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "dataset"))

from models.recondrive_model import ReconDrive_LITModelModule
try:
    from dataset.vggt3dgs_scene_data_module import VGGT3DGS_SceneDataModule
    from dataset.vggt4dgs_scene_dataset import custom_collate_fn
except ModuleNotFoundError:
    from vggt3dgs_scene_data_module import VGGT3DGS_SceneDataModule
    from vggt4dgs_scene_dataset import custom_collate_fn
from scripts.inference import SceneSampleDataset


def to_device(data, device):
    if isinstance(data, dict):
        return {k: (v if k == 'vehicle_annotations' else to_device(v, device))
                for k, v in data.items()}
    if isinstance(data, (list, tuple)):
        return type(data)(to_device(x, device) for x in data)
    if torch.is_tensor(data):
        return data.to(device)
    return data


def extract_frame0_gaussians(batch_recontrast_data):
    """Keep only frame-0 Gaussians to avoid ghosting."""
    out = {}
    for key, val in batch_recontrast_data.items():
        if torch.is_tensor(val) and val.dim() >= 2:
            n_total = val.shape[1]
            if n_total % 2 == 0:
                out[key] = val[:, : n_total // 2]
            else:
                out[key] = val
        else:
            out[key] = val
    return out


def filter_gaussians_by_range(recontrast_data, x_range, y_range, z_range):
    """Drop Gaussians outside voxel range."""
    xyz = recontrast_data['xyz']
    if xyz.shape[0] != 1:
        raise ValueError("This script expects batch_size=1 for filtering.")

    x_min, x_max = x_range
    y_min, y_max = y_range
    z_min, z_max = z_range

    mask = (
        (xyz[..., 0] >= x_min) & (xyz[..., 0] < x_max) &
        (xyz[..., 1] >= y_min) & (xyz[..., 1] < y_max) &
        (xyz[..., 2] >= z_min) & (xyz[..., 2] < z_max)
    )[0]  # [N]

    if mask.sum() == 0:
        raise RuntimeError("No Gaussians remain inside voxel range.")

    filtered = {}
    n_points = xyz.shape[1]
    for key, val in recontrast_data.items():
        if torch.is_tensor(val) and val.dim() >= 2 and val.shape[1] == n_points:
            filtered[key] = val[:, mask]
        else:
            filtered[key] = val

    return filtered, int(mask.sum().item()), int(n_points)


def apply_voxel_range_opacity(recontrast_data, x_range, y_range, z_range):
    """Zero opacity for Gaussians outside voxel range (shape-preserving)."""
    xyz = recontrast_data['xyz']
    if xyz.shape[0] != 1:
        raise ValueError("This script expects batch_size=1 for range masking.")

    x_min, x_max = x_range
    y_min, y_max = y_range
    z_min, z_max = z_range

    mask = (
        (xyz[..., 0] >= x_min) & (xyz[..., 0] < x_max) &
        (xyz[..., 1] >= y_min) & (xyz[..., 1] < y_max) &
        (xyz[..., 2] >= z_min) & (xyz[..., 2] < z_max)
    )  # [1, N]

    masked = {}
    n_points = xyz.shape[1]
    for key, val in recontrast_data.items():
        if torch.is_tensor(val) and val.dim() >= 2 and val.shape[1] == n_points:
            if key == 'opacity_maps':
                masked[key] = val * mask.unsqueeze(-1)
            else:
                masked[key] = val
        else:
            masked[key] = val

    return masked, mask[0]


def ensure_render_fields(recontrast_data):
    """Ensure render-required fields exist and mark as global points."""
    xyz = recontrast_data['xyz']
    if 'forward_flow' not in recontrast_data:
        recontrast_data['forward_flow'] = torch.zeros_like(xyz)
    recontrast_data['ae_global_points'] = True
    return recontrast_data


def compute_xyz_ranges(xyz):
    """Compute per-axis min/max for xyz tensor [B, N, 3], batch_size=1."""
    if xyz.shape[0] != 1 or xyz.shape[1] == 0:
        return {'x': [None, None], 'y': [None, None], 'z': [None, None]}
    xyz0 = xyz[0]
    mins = xyz0.min(dim=0).values
    maxs = xyz0.max(dim=0).values
    return {
        'x': [float(mins[0].item()), float(maxs[0].item())],
        'y': [float(mins[1].item()), float(maxs[1].item())],
        'z': [float(mins[2].item()), float(maxs[2].item())],
    }


def render_gaussian_scene_image(recontrast_data, batch_idx=0,
                                opacity_thresh=0.1, max_points=80_000,
                                elev=25, azim=-60):
    """
    Render a 3D scatter of Gaussian positions and return as RGB numpy [H, W, 3].
    Color uses SH degree-0 (DC) coefficients.
    """
    xyz = recontrast_data['xyz'][batch_idx].detach().cpu().float().numpy()
    opacity = recontrast_data['opacity_maps'][batch_idx].detach().cpu().float().numpy().squeeze(-1)
    sh = recontrast_data['sh_maps'][batch_idx].detach().cpu().float().numpy()

    # sh layout: [N, 25, 3] or [N, 3, 25]
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
            keep = (xyz[:, axis] >= lo) & (xyz[:, axis] <= hi)
            xyz, rgb, opacity = xyz[keep], rgb[keep], opacity[keep]

    alpha = np.clip(opacity, 0.05, 1.0)
    rgba = np.concatenate([rgb, alpha[:, None]], axis=1)

    fig = plt.figure(figsize=(5, 5), dpi=100)
    ax = fig.add_subplot(111, projection='3d')
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


def save_range_mask_overlay(gt, mask, render_range, save_path):
    """Save a 4-panel strip: GT | mask | GT*mask | range render."""
    if gt.dim() == 4:
        gt = gt.squeeze(0)
    if render_range.dim() == 4:
        render_range = render_range.squeeze(0)
    if mask.dim() == 3:
        mask = mask.squeeze(0)

    mask_rgb = mask.repeat(3, 1, 1)
    gt_masked = gt * mask_rgb
    panel = torch.cat([gt, mask_rgb, gt_masked, render_range], dim=2)
    vutils.save_image(panel, save_path)


def parse_args():
    parser = argparse.ArgumentParser(description="Validate AE input Gaussians.")
    parser.add_argument('--cfg_path', type=str, default='configs/nuscenes/recondrive_ae.yaml')
    parser.add_argument('--output_dir', type=str, default='./work_dirs/ae_input_validation')
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--scene_idx', type=int, default=0)
    parser.add_argument('--num_samples', type=int, default=1)
    parser.add_argument('--pretrained_ckpt', type=str, default='./checkpoints/recondrive_stage1.ckpt')
    parser.add_argument('--use_vehicle_flow', action='store_true', help='Enable SAM2 vehicle flow')
    parser.add_argument('--range_mask_alpha_thresh', type=float, default=None)
    return parser.parse_args()


def main():
    args = parse_args()

    with open(args.cfg_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    config['model_cfg']['batch_size'] = 1
    config['data_cfg']['batch_size'] = 1
    config['model_cfg']['use_vehicle_flow'] = bool(args.use_vehicle_flow)

    device = args.device or ('cuda:0' if torch.cuda.is_available() else 'cpu')

    data_module = VGGT3DGS_SceneDataModule(cfg=config['data_cfg'])
    data_module.setup(stage='test')
    scene_loader = data_module.test_scene_dataloader()

    scene_batch = None
    for idx, batch in enumerate(scene_loader):
        if idx == args.scene_idx:
            scene_batch = batch
            break
    if scene_batch is None:
        raise IndexError(f"scene_idx {args.scene_idx} is out of range.")

    if 'samples' in scene_batch:
        samples = scene_batch['samples'][: args.num_samples]
        sample_ds = SceneSampleDataset(samples)
    else:
        indices = scene_batch['sample_indices'][: args.num_samples]
        sample_ds = SceneSampleDataset(
            indices,
            dataset=scene_batch['dataset'],
            scene_idx=scene_batch['scene_idx'],
        )

    from torch.utils.data import DataLoader
    loader = DataLoader(sample_ds, batch_size=1, shuffle=False,
                        num_workers=0, collate_fn=custom_collate_fn)

    model = ReconDrive_LITModelModule(
        cfg=config['model_cfg'],
        save_dir=args.output_dir,
        logger=None,
    ).to(device)
    model.eval()

    if args.pretrained_ckpt and Path(args.pretrained_ckpt).exists():
        model.load_pretrained_checkpoint(args.pretrained_ckpt, strict=False, verbose=True)
    else:
        print(f"[WARN] Pretrained checkpoint not found: {args.pretrained_ckpt}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    x_range = config['model_cfg']['ae_cfg']['x_range']
    y_range = config['model_cfg']['ae_cfg']['y_range']
    z_range = config['model_cfg']['ae_cfg']['z_range']
    range_alpha_thresh = config['model_cfg']['ae_cfg'].get('range_mask_alpha_thresh', 0.01)
    if args.range_mask_alpha_thresh is not None:
        range_alpha_thresh = args.range_mask_alpha_thresh

    for sample_idx, batch_input in enumerate(loader):
        batch_input = to_device(batch_input, device)
        model.set_normal_params(batch_input)
        model.all_render_frame_ids = [0]

        with torch.no_grad():
            recontrast_data = model.get_recontrast_data(batch_input, batch_idx=0)

        recontrast_before = extract_frame0_gaussians(recontrast_data)
        recontrast_before = ensure_render_fields(recontrast_before)
        count_before = recontrast_before['xyz'].shape[1]
        ranges_before = compute_xyz_ranges(recontrast_before['xyz'])

        recontrast_range, range_mask = apply_voxel_range_opacity(
            recontrast_before, x_range, y_range, z_range
        )
        recontrast_range = ensure_render_fields(recontrast_range)

        recontrast_after, kept, _ = filter_gaussians_by_range(
            recontrast_before, x_range, y_range, z_range
        )
        recontrast_after = ensure_render_fields(recontrast_after)
        ranges_after = compute_xyz_ranges(recontrast_after['xyz'])

        render_data = model.get_render_data(batch_input)
        with torch.no_grad():
            splating_before = model.render_splating_imgs(recontrast_before, render_data)
            splating_range = model.render_splating_imgs(recontrast_range, render_data)
            splating_after = model.render_splating_imgs(recontrast_after, render_data)

        sample_dir = output_dir / f"sample_{sample_idx:03d}"
        (sample_dir / "renders_before").mkdir(parents=True, exist_ok=True)
        (sample_dir / "renders_range").mkdir(parents=True, exist_ok=True)
        (sample_dir / "renders_after").mkdir(parents=True, exist_ok=True)
        (sample_dir / "renders_masked").mkdir(parents=True, exist_ok=True)
        (sample_dir / "gaussian_3d").mkdir(parents=True, exist_ok=True)

        # Save per-camera renders (frame 0)
        range_mask_stats = {}
        for cam_id in range(model.num_cams):
            key = ('gaussian_color', 0, cam_id)
            if key in splating_before:
                img = splating_before[key][0]
                vutils.save_image(img, sample_dir / "renders_before" / f"render_cam_{cam_id}.png")
            if key in splating_range:
                img = splating_range[key][0]
                vutils.save_image(img, sample_dir / "renders_range" / f"render_cam_{cam_id}.png")
            if key in splating_after:
                img = splating_after[key][0]
                vutils.save_image(img, sample_dir / "renders_after" / f"render_cam_{cam_id}.png")
            gt_key = ('groudtruth', 0, cam_id)
            if gt_key in splating_range:
                gt = splating_range[gt_key][0]
                vutils.save_image(gt, sample_dir / "renders_after" / f"gt_cam_{cam_id}.png")

                alpha_key = ('gaussian_alpha', 0, cam_id)
                if alpha_key in splating_range:
                    alpha = splating_range[alpha_key][0]  # [1, H, W]
                    mask = (alpha > range_alpha_thresh).to(alpha.dtype)
                    range_mask_stats[str(cam_id)] = float(mask.mean().item())

                    vutils.save_image(mask, sample_dir / "renders_masked" / f"mask_cam_{cam_id}.png")
                    save_range_mask_overlay(
                        gt=gt,
                        mask=mask,
                        render_range=splating_range[key][0],
                        save_path=sample_dir / "renders_masked" / f"overlay_cam_{cam_id}.png",
                    )

        # Save 3D Gaussian scatter (before/after)
        gauss_before = render_gaussian_scene_image(recontrast_before, batch_idx=0)
        gauss_after = render_gaussian_scene_image(recontrast_after, batch_idx=0)
        Image.fromarray(gauss_before).save(sample_dir / "gaussian_3d" / "gaussian_scatter_before.png")
        Image.fromarray(gauss_after).save(sample_dir / "gaussian_3d" / "gaussian_scatter_after.png")

        stats = {
            'voxel_range': {'x': x_range, 'y': y_range, 'z': z_range},
            'before': {'count': int(count_before), 'xyz_range': ranges_before},
            'after': {'count': int(kept), 'xyz_range': ranges_after},
            'range_mask': {
                'alpha_thresh': float(range_alpha_thresh),
                'coverage_per_cam': range_mask_stats,
                'kept_by_mask': int(range_mask.sum().item()),
            },
        }
        with open(sample_dir / "stats.json", "w") as f:
            json.dump(stats, f, indent=2)

        print(
            f"[sample {sample_idx}] before={count_before} "
            f"range x={ranges_before['x']} y={ranges_before['y']} z={ranges_before['z']}"
        )
        print(
            f"[sample {sample_idx}] after={kept}/{count_before} "
            f"range x={ranges_after['x']} y={ranges_after['y']} z={ranges_after['z']}"
        )


if __name__ == '__main__':
    main()
