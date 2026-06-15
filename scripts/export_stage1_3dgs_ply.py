#!/usr/bin/env python3
"""Export Stage1-generated 3D Gaussian Splatting data to PLY files."""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Dataset
from PIL import Image, ImageDraw

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dataset.vggt4dgs_data_module import VGGT4DGS_LITDataModule
from dataset.vggt4dgs_dataset import custom_collate_fn
from models.recondrive_stage1_model import ReconDriveStage1_LITModelModule

CAMERA_NAMES = [
    'CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT',
    'CAM_BACK_LEFT', 'CAM_BACK_RIGHT', 'CAM_BACK',
]


class SceneSampleDataset(Dataset):
    """Lazy dataset wrapper over global base-dataset indices."""

    def __init__(self, dataset, sample_indices):
        self.dataset = dataset
        self.sample_indices = sample_indices

    def __len__(self):
        return len(self.sample_indices)

    def __getitem__(self, idx):
        return self.dataset[self.sample_indices[idx]]


class SceneDataLoader:
    """Scene iterator built from the same dataset path used by scripts/train.sh."""

    def __init__(self, dataset):
        self.dataset = dataset
        self.scene_batches = self._build_scene_batches()

    def _build_scene_batches(self):
        token_to_idx = {token: idx for idx, token in enumerate(self.dataset.sample_tokens)}
        scene_batches = []
        for scene_idx, scene_tokens in enumerate(self.dataset.scenes_data):
            sample_indices = [token_to_idx[token] for token in scene_tokens if token in token_to_idx]
            if not sample_indices:
                continue
            scene_batches.append({
                'scene_idx': scene_idx,
                'scene_name': self.dataset.scene_names[scene_idx],
                'scene_token': self.dataset.scene_tokens[scene_idx],
                'scene_length': len(sample_indices),
                'sample_indices': sample_indices,
                'dataset': self.dataset,
            })
        return scene_batches

    def __iter__(self):
        return iter(self.scene_batches)

    def __len__(self):
        return len(self.scene_batches)


def to_device(data, device):
    if isinstance(data, dict):
        return {k: (v if k == 'vehicle_annotations' else to_device(v, device)) for k, v in data.items()}
    if isinstance(data, (list, tuple)):
        return type(data)(to_device(x, device) for x in data)
    if torch.is_tensor(data):
        return data.to(device)
    return data


def tensor_to_numpy(value, batch_idx=0):
    """Extract one batch item from tensor/list Gaussian fields as float32 numpy."""
    if isinstance(value, (list, tuple)):
        value = value[batch_idx]
    else:
        value = value[batch_idx]
    return value.detach().cpu().float().numpy()


def normalize_sh(sh):
    """Return SH coefficients in [N, coeff, 3] layout."""
    if sh.ndim != 3:
        raise ValueError(f"Expected SH array with shape [N, C, 3] or [N, 3, C], got {sh.shape}")
    if sh.shape[-1] == 3:
        return sh
    if sh.shape[1] == 3:
        return np.transpose(sh, (0, 2, 1))
    raise ValueError(f"Cannot infer SH channel axis from shape {sh.shape}")


def gaussian_rgb_from_sh(sh_coeffs):
    """Approximate display RGB from the SH DC term."""
    c0 = 0.28209479177387814
    dc = sh_coeffs[:, 0, :]
    rgb = np.clip(dc / c0 * 0.5 + 0.5, 0.0, 1.0)
    return (rgb * 255.0).round().astype(np.uint8)


def extract_gaussians(recontrast_data, batch_idx=0, opacity_thresh=0.0, max_points=None, seed=0):
    xyz = tensor_to_numpy(recontrast_data['xyz'], batch_idx).reshape(-1, 3)
    rot = tensor_to_numpy(recontrast_data['rot_maps'], batch_idx).reshape(-1, 4)
    scale = tensor_to_numpy(recontrast_data['scale_maps'], batch_idx).reshape(-1, 3)
    opacity = tensor_to_numpy(recontrast_data['opacity_maps'], batch_idx).reshape(-1, 1)
    sh = normalize_sh(tensor_to_numpy(recontrast_data['sh_maps'], batch_idx))

    n = min(xyz.shape[0], rot.shape[0], scale.shape[0], opacity.shape[0], sh.shape[0])
    xyz, rot, scale, opacity, sh = xyz[:n], rot[:n], scale[:n], opacity[:n], sh[:n]

    finite_mask = (
        np.isfinite(xyz).all(axis=1)
        & np.isfinite(rot).all(axis=1)
        & np.isfinite(scale).all(axis=1)
        & np.isfinite(opacity).all(axis=1)
        & np.isfinite(sh.reshape(n, -1)).all(axis=1)
    )
    opacity_mask = opacity[:, 0] >= opacity_thresh
    keep = finite_mask & opacity_mask
    xyz, rot, scale, opacity, sh = xyz[keep], rot[keep], scale[keep], opacity[keep], sh[keep]

    if max_points is not None and max_points > 0 and xyz.shape[0] > max_points:
        rng = np.random.default_rng(seed)
        indices = rng.choice(xyz.shape[0], max_points, replace=False)
        indices.sort()
        xyz, rot, scale, opacity, sh = xyz[indices], rot[indices], scale[indices], opacity[indices], sh[indices]

    return {
        'xyz': xyz.astype(np.float32),
        'rot': rot.astype(np.float32),
        'scale': scale.astype(np.float32),
        'opacity': opacity.astype(np.float32),
        'sh': sh.astype(np.float32),
        'rgb': gaussian_rgb_from_sh(sh) if sh.shape[0] else np.zeros((0, 3), dtype=np.uint8),
        'num_before_filter': n,
    }


def ply_header(num_vertices, num_rest_coeffs):
    lines = [
        'ply',
        'format ascii 1.0',
        'comment Generated by scripts/export_stage1_3dgs_ply.py',
        f'element vertex {num_vertices}',
        'property float x',
        'property float y',
        'property float z',
        'property uchar red',
        'property uchar green',
        'property uchar blue',
        'property float opacity',
        'property float scale_0',
        'property float scale_1',
        'property float scale_2',
        'property float rot_0',
        'property float rot_1',
        'property float rot_2',
        'property float rot_3',
        'property float f_dc_0',
        'property float f_dc_1',
        'property float f_dc_2',
    ]
    lines.extend(f'property float f_rest_{i}' for i in range(num_rest_coeffs))
    lines.append('end_header')
    return '\n'.join(lines) + '\n'


def write_gaussians_ply(path, gaussians):
    xyz = gaussians['xyz']
    rot = gaussians['rot']
    scale = gaussians['scale']
    opacity = gaussians['opacity']
    sh = gaussians['sh']
    rgb = gaussians['rgb']

    sh_dc = sh[:, 0, :] if sh.shape[0] else np.zeros((0, 3), dtype=np.float32)
    sh_rest = sh[:, 1:, :].reshape(sh.shape[0], -1) if sh.shape[1] > 1 else np.zeros((sh.shape[0], 0), dtype=np.float32)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='ascii') as f:
        f.write(ply_header(xyz.shape[0], sh_rest.shape[1]))
        for i in range(xyz.shape[0]):
            values = [
                *xyz[i].tolist(),
                int(rgb[i, 0]), int(rgb[i, 1]), int(rgb[i, 2]),
                float(opacity[i, 0]),
                *scale[i].tolist(),
                *rot[i].tolist(),
                *sh_dc[i].tolist(),
                *sh_rest[i].tolist(),
            ]
            f.write(' '.join(str(v) for v in values) + '\n')


def tensor_to_uint8_image(tensor):
    """Convert [C,H,W] or [1,C,H,W] tensor in [0,1] to uint8 [H,W,3]."""
    if tensor.dim() == 4:
        tensor = tensor[0]
    array = tensor.detach().cpu().float().clamp(0, 1).numpy()
    if array.shape[0] == 1:
        array = np.repeat(array, 3, axis=0)
    return (array[:3].transpose(1, 2, 0) * 255.0).round().astype(np.uint8)


def apply_vis_mask(image_tensor, splating_data, frame_id, cam_id):
    mask_key = ('warped_mask', frame_id, cam_id)
    if mask_key not in splating_data:
        return image_tensor.clamp(0, 1)

    mask = splating_data[mask_key]
    if mask.dim() == 4:
        mask = mask[0]
    if mask.dim() == 2:
        mask = mask.unsqueeze(0)
    if mask.dim() == 3 and mask.shape[0] == 1:
        mask = mask.expand_as(image_tensor)
    elif mask.dim() == 3 and mask.shape[0] != image_tensor.shape[0]:
        mask = mask[:1].expand_as(image_tensor)
    mask = mask.to(dtype=image_tensor.dtype, device=image_tensor.device)
    return (image_tensor * mask).clamp(0, 1)


def save_gt_render_comparison(splating_data, save_path, num_cams, frame_id=0, camera_names=None):
    """Save a two-column GT/Render comparison with one row per camera."""
    rows = []
    label_width = 150
    header_height = 28
    separator = 4

    for cam_id in range(num_cams):
        pred_key = ('gaussian_color', frame_id, cam_id)
        gt_key = ('groudtruth', frame_id, cam_id)
        if pred_key not in splating_data or gt_key not in splating_data:
            continue

        gt = splating_data[gt_key][0]
        pred = splating_data[pred_key][0]
        gt_np = tensor_to_uint8_image(apply_vis_mask(gt, splating_data, frame_id, cam_id))
        pred_np = tensor_to_uint8_image(apply_vis_mask(pred, splating_data, frame_id, cam_id))
        rows.append((cam_id, gt_np, pred_np))

    if not rows:
        return False

    image_h, image_w = rows[0][1].shape[:2]
    canvas_w = label_width + image_w * 2 + separator
    canvas_h = header_height + len(rows) * image_h
    canvas = Image.new('RGB', (canvas_w, canvas_h), color=(255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    draw.text((label_width + image_w // 2 - 12, 8), 'GT', fill=(0, 0, 0))
    draw.text((label_width + image_w + separator + image_w // 2 - 28, 8), 'Render', fill=(0, 0, 0))

    for row_idx, (cam_id, gt_np, pred_np) in enumerate(rows):
        y = header_height + row_idx * image_h
        cam_name = camera_names[cam_id] if camera_names and cam_id < len(camera_names) else f'CAM_{cam_id}'
        draw.text((8, y + max(0, image_h // 2 - 8)), cam_name, fill=(0, 0, 0))
        canvas.paste(Image.fromarray(gt_np), (label_width, y))
        canvas.paste(Image.fromarray(pred_np), (label_width + image_w + separator, y))

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    canvas.save(save_path)
    return True


def build_scene_sample_dataset(scene_batch, max_samples=None):
    indices = scene_batch['sample_indices']
    if max_samples is not None:
        indices = indices[:max_samples]
    return SceneSampleDataset(scene_batch['dataset'], indices), list(indices)


def sanitize_path_part(value):
    return str(value).replace('/', '_').replace('\\', '_')


def export_stage1_3dgs(model, scene_dataloader, device, output_dir, max_samples=None,
                       opacity_thresh=0.0, max_points=None, seed=0):
    os.makedirs(output_dir, exist_ok=True)
    total_files = 0
    total_points = 0

    with torch.no_grad():
        for scene_idx, scene_batch in enumerate(scene_dataloader):
            scene_name = sanitize_path_part(scene_batch['scene_name'])
            print(f"\n[Scene {scene_idx + 1}/{len(scene_dataloader)}] {scene_name}")
            sample_ds, sample_ids = build_scene_sample_dataset(scene_batch, max_samples=max_samples)
            loader = DataLoader(
                sample_ds,
                batch_size=1,
                shuffle=False,
                num_workers=0,
                collate_fn=custom_collate_fn,
            )

            for local_idx, batch in enumerate(loader):
                sample_id = sample_ids[local_idx] if local_idx < len(sample_ids) else local_idx
                batch['scene_idx'] = scene_idx
                batch = to_device(batch, device)

                recontrast_data, render_data, splating_data = model.predict_step(batch, local_idx)
                splating_data = model._apply_occ_render_mask(splating_data, batch)
                gaussians = extract_gaussians(
                    recontrast_data,
                    batch_idx=0,
                    opacity_thresh=opacity_thresh,
                    max_points=max_points,
                    seed=seed + scene_idx * 100000 + local_idx,
                )

                sample_dir = os.path.join(output_dir, scene_name, f'sample_{int(sample_id):04d}')
                save_path = os.path.join(sample_dir, 'gaussians.ply')
                write_gaussians_ply(save_path, gaussians)
                vis_path = os.path.join(sample_dir, 'gt_render_compare.png')
                saved_vis = save_gt_render_comparison(
                    splating_data,
                    vis_path,
                    num_cams=getattr(model, 'num_cams', len(CAMERA_NAMES)),
                    frame_id=0,
                    camera_names=CAMERA_NAMES,
                )

                count = gaussians['xyz'].shape[0]
                total_files += 1
                total_points += count
                print(
                    f"  sample {int(sample_id):04d}: saved {count:,}/{gaussians['num_before_filter']:,} "
                    f"Gaussians -> {save_path}"
                )
                if saved_vis:
                    print(f"    GT/Render comparison -> {vis_path}")

    print(f"\nExport complete: {total_files} PLY files, {total_points:,} total Gaussians")
    print(f"Output directory: {output_dir}")


def load_scene_dataloader(config, max_scenes=None):
    config['data_cfg']['batch_size'] = 1
    config['model_cfg']['batch_size'] = 1
    if 'context_span' in config['data_cfg']:
        config['model_cfg']['context_span'] = config['data_cfg']['context_span']
    if 'nuscenes_version' in config['data_cfg']:
        config['model_cfg']['nuscenes_version'] = config['data_cfg']['nuscenes_version']

    # Match scripts/train.sh -> scripts.trainer: this datamodule honors
    # data_cfg.dataset_type='3d' and passes OCC kwargs to NuScenesdataset3D.
    data_module = VGGT4DGS_LITDataModule(cfg=config['data_cfg'])
    data_module.setup(stage='test')
    scene_dataloader = SceneDataLoader(data_module.test_dataset)

    if max_scenes is None:
        return scene_dataloader

    limited = []
    for i, scene_batch in enumerate(scene_dataloader):
        if i >= max_scenes:
            break
        limited.append(scene_batch)
    return limited


def parse_args():
    parser = argparse.ArgumentParser(description='Export Stage1-generated 3DGS data to ASCII PLY files.')
    parser.add_argument('--cfg_path', type=str, default='./configs/nuscenes/recondrive.yaml')
    parser.add_argument('--ckpt', type=str, default='./checkpoints/recondrive_stage1.ckpt')
    parser.add_argument('--output_dir', type=str, default='./work_dirs/stage1_3dgs_ply')
    parser.add_argument('--device', type=str, default='0', help='Device id or full device string, e.g. 0 or cuda:0')
    parser.add_argument('--max_scenes', type=int, default=None, help='Maximum number of scenes to export')
    parser.add_argument('--max_samples', type=int, default=None, help='Maximum number of samples per scene')
    parser.add_argument('--opacity_thresh', type=float, default=0.0, help='Drop Gaussians with opacity below this value')
    parser.add_argument('--max_points', type=int, default=None, help='Optional random downsample limit per PLY')
    parser.add_argument('--seed', type=int, default=0, help='Random seed used when --max_points downsamples')
    return parser.parse_args()


def main():
    args = parse_args()
    device = args.device if args.device.startswith(('cuda', 'cpu')) else f'cuda:{args.device}'
    if device.startswith('cuda') and not torch.cuda.is_available():
        print('CUDA was requested but is not available; falling back to CPU')
        device = 'cpu'

    print(f'Loading config: {args.cfg_path}')
    with open(args.cfg_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    print('Loading dataset...')
    scene_dataloader = load_scene_dataloader(config, max_scenes=args.max_scenes)

    print(f'Loading Stage1 model: {args.ckpt}')
    model = ReconDriveStage1_LITModelModule(cfg=config['model_cfg'], save_dir='./temp_log', logger=None)
    model.load_pretrained_checkpoint(args.ckpt)
    model.to(device)
    model.eval()

    export_stage1_3dgs(
        model=model,
        scene_dataloader=scene_dataloader,
        device=device,
        output_dir=args.output_dir,
        max_samples=args.max_samples,
        opacity_thresh=args.opacity_thresh,
        max_points=args.max_points,
        seed=args.seed,
    )


if __name__ == '__main__':
    main()
