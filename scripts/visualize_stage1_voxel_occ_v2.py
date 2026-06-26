#!/usr/bin/env python3
"""
Create visualization videos for Stage1 voxel-OCC inference.

The script is tailored for configs/nuscenes/recondrive_stage1_voxel_occ_v2.yaml:
- uses ReconDriveStage1_LITModelModule, not the generic Stage2 module
- renders frame-0 reconstruction with ae_global_points=True
- applies the OCC render mask used during Stage1 voxel-OCC validation
- writes per-scene MP4 videos and optional PNG frames
"""

import argparse
import os
import sys
from pathlib import Path

# Keep matplotlib caches inside writable workspace/tmp paths on locked-down hosts.
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "models"))

from dataset.vggt4dgs_data_module import VGGT4DGS_LITDataModule
from dataset.vggt4dgs_dataset import custom_collate_fn
from models.recondrive_stage1_model import ReconDriveStage1_LITModelModule

CAMERA_NAMES = [
    "CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT",
    "CAM_BACK_LEFT", "CAM_BACK_RIGHT", "CAM_BACK",
]


class SceneSampleDataset(Dataset):
    """Lazy scene sample wrapper matching scripts/inference.py without importing it."""

    def __init__(self, samples_or_indices, dataset=None, scene_idx=None):
        self.lazy_mode = dataset is not None and scene_idx is not None
        self.samples_or_indices = samples_or_indices
        self.dataset = dataset
        self.scene_idx = scene_idx

    def __len__(self):
        return len(self.samples_or_indices)

    def __getitem__(self, idx):
        item = self.samples_or_indices[idx]
        if self.lazy_mode:
            return self.dataset.__getitem__(item, self.scene_idx)
        return item


class SimpleSceneDataLoader:
    """Scene iterator for NuScenesdataset3D, using global dataset sample indices."""

    def __init__(self, dataset):
        self.dataset = dataset
        self.scene_offsets = []
        offset = 0
        for scene_tokens in dataset.scenes_data:
            self.scene_offsets.append(offset)
            offset += len(scene_tokens)

    def __len__(self):
        return self.dataset.get_num_scenes()

    def __iter__(self):
        for scene_idx, offset in enumerate(self.scene_offsets):
            scene_length = self.dataset.get_scene_length(scene_idx)
            yield {
                "scene_idx": scene_idx,
                "scene_name": self.dataset.get_scene_name(scene_idx),
                "scene_token": self.dataset.get_scene_token(scene_idx),
                "scene_length": scene_length,
                "sample_indices": list(range(offset, offset + scene_length)),
                "dataset": self.dataset,
            }


def to_device(data, device):
    if isinstance(data, dict):
        return {k: (v if k == "vehicle_annotations" else to_device(v, device)) for k, v in data.items()}
    if isinstance(data, (list, tuple)):
        return type(data)(to_device(x, device) for x in data)
    if torch.is_tensor(data):
        return data.to(device, non_blocking=True)
    return data


def tensor_to_uint8(tensor):
    """Convert [C,H,W] or [1,C,H,W] tensor in [0,1] to RGB uint8 [H,W,3]."""
    if tensor.dim() == 4:
        tensor = tensor[0]
    arr = tensor.detach().cpu().float().clamp(0, 1).numpy()
    if arr.shape[0] == 1:
        arr = np.repeat(arr, 3, axis=0)
    return (arr.transpose(1, 2, 0) * 255.0).astype(np.uint8)


def resize_image(img, height, width):
    if img.shape[:2] == (height, width):
        return img
    return cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)


def add_text_bar(img, text, height=24):
    bar = np.full((height, img.shape[1], 3), 18, dtype=np.uint8)
    pil = Image.fromarray(bar)
    draw = ImageDraw.Draw(pil)
    draw.text((8, 4), text, fill=(245, 245, 245), font=ImageFont.load_default())
    return np.concatenate([np.asarray(pil), img], axis=0)


def apply_visual_mask(img_tensor, splating_data, frame_id, cam_id):
    mask_key = ("warped_mask", frame_id, cam_id)
    if mask_key not in splating_data:
        return img_tensor.clamp(0, 1)

    mask = splating_data[mask_key][0]
    if mask.dim() == 2:
        mask = mask.unsqueeze(0)
    if mask.dim() == 3 and mask.shape[0] == 1:
        mask = mask.expand_as(img_tensor)
    elif mask.dim() == 3 and mask.shape[0] != img_tensor.shape[0]:
        mask = mask[:1].expand_as(img_tensor)
    return (img_tensor * mask.to(dtype=img_tensor.dtype, device=img_tensor.device)).clamp(0, 1)


def save_gaussian_scene_image(recontrast_data, save_path, batch_idx=0,
                              opacity_thresh=0.1, elev=25, azim=-60):
    """Save predicted 3D Gaussians as a PNG scatter view."""
    xyz = recontrast_data["xyz"][batch_idx].detach().cpu().float().numpy()
    opacity = recontrast_data["opacity_maps"][batch_idx].detach().cpu().float().numpy().squeeze(-1)
    sh = recontrast_data["sh_maps"][batch_idx].detach().cpu().float().numpy()

    if sh.shape[-1] != 3 and sh.shape[-2] == 3:
        sh = sh.transpose(0, 2, 1)

    mask = opacity > opacity_thresh
    if mask.sum() == 0:
        mask = np.ones_like(opacity, dtype=bool)
    xyz, opacity, sh = xyz[mask], opacity[mask], sh[mask]

    C0 = 0.28209479177387814
    rgb = np.clip(sh[:, 0, :] / C0 * 0.5 + 0.5, 0.0, 1.0)

    if len(xyz) > 100:
        for axis in range(3):
            lo, hi = np.percentile(xyz[:, axis], [1, 99])
            keep = (xyz[:, axis] >= lo) & (xyz[:, axis] <= hi)
            xyz, rgb, opacity = xyz[keep], rgb[keep], opacity[keep]

    alpha = np.clip(opacity, 0.05, 1.0)
    rgba = np.concatenate([rgb, alpha[:, None]], axis=1)

    fig = plt.figure(figsize=(6, 6), dpi=120)
    ax = fig.add_subplot(111, projection="3d")
    if len(xyz) > 0:
        ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=rgba, s=0.3, linewidths=0, depthshade=True)
        ranges = np.array([[xyz[:, i].min(), xyz[:, i].max()] for i in range(3)])
        max_range = (ranges[:, 1] - ranges[:, 0]).max() / 2 or 1.0
        mid = ranges.mean(axis=1)
        ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
        ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
        ax.set_zlim(mid[2] - max_range, mid[2] + max_range)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.view_init(elev=elev, azim=azim)
    ax.set_title(f"3D Gaussian Scene ({len(xyz):,} points)")
    fig.tight_layout()

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def make_camera_grid(splating_data, num_cams, frame_id=0, mask_visuals=True):
    tiles = []
    psnrs = []

    for cam_id in range(num_cams):
        pred_key = ("gaussian_color", frame_id, cam_id)
        gt_key = ("groudtruth", frame_id, cam_id)
        if pred_key not in splating_data or gt_key not in splating_data:
            continue

        pred = splating_data[pred_key][0].clamp(0, 1)
        gt = splating_data[gt_key][0].clamp(0, 1)
        if mask_visuals:
            pred_vis = apply_visual_mask(pred, splating_data, frame_id, cam_id)
            gt_vis = apply_visual_mask(gt, splating_data, frame_id, cam_id)
        else:
            pred_vis, gt_vis = pred, gt

        mse = torch.mean((pred - gt) ** 2).item()
        psnr = -10.0 * np.log10(mse + 1e-8)
        psnrs.append(psnr)

        gt_img = tensor_to_uint8(gt_vis)
        pred_img = tensor_to_uint8(pred_vis)
        tile = np.concatenate([gt_img, pred_img], axis=1)

        cam_name = CAMERA_NAMES[cam_id] if cam_id < len(CAMERA_NAMES) else f"CAM_{cam_id}"
        tiles.append(add_text_bar(tile, f"{cam_name} | GT / Render"))

    if not tiles:
        return None, None

    # Preserve the six-camera order in a compact 3x2 board.
    tile_h, tile_w = tiles[0].shape[:2]
    while len(tiles) < 6:
        tiles.append(np.zeros((tile_h, tile_w, 3), dtype=np.uint8))
    rows = [np.concatenate(tiles[i:i + 2], axis=1) for i in range(0, 6, 2)]
    return np.concatenate(rows, axis=0), float(np.mean(psnrs)) if psnrs else None


def make_video_frame(board, scene_name, sample_idx, mean_psnr, output_height=None):
    title = f"{scene_name} | sample {sample_idx:04d}"
    frame = add_text_bar(board, title, height=30)
    if output_height and frame.shape[0] != output_height:
        scale = output_height / frame.shape[0]
        frame = cv2.resize(frame, (int(frame.shape[1] * scale), output_height), interpolation=cv2.INTER_AREA)
    # MP4 encoders require even dimensions.
    h, w = frame.shape[:2]
    if h % 2 or w % 2:
        frame = cv2.copyMakeBorder(frame, 0, h % 2, 0, w % 2, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    return frame


def build_sample_loader(scene_batch, max_samples=None, frame_skip=1):
    if "samples" in scene_batch:
        all_items = list(scene_batch["samples"])
        indices = list(range(len(all_items)))[::max(1, frame_skip)]
        if max_samples is not None:
            indices = indices[:max_samples]
        dataset = SceneSampleDataset([all_items[i] for i in indices])
        return dataset, indices

    all_indices = list(scene_batch["sample_indices"])[::max(1, frame_skip)]
    if max_samples is not None:
        all_indices = all_indices[:max_samples]
    dataset = SceneSampleDataset(
        all_indices,
        dataset=scene_batch["dataset"],
        scene_idx=scene_batch["scene_idx"],
    )
    return dataset, all_indices


def render_batch(model, batch, batch_idx):
    model._set_stage1_frame_ids()
    recontrast_data = model.get_recontrast_data(batch, batch_idx)
    render_data = model.get_render_data(batch)
    splating_data = model.render_splating_imgs({**recontrast_data, "ae_global_points": True}, render_data)
    splating_data = model._apply_occ_render_mask(splating_data, batch)
    return recontrast_data, render_data, splating_data


def run_visualization(model, scene_dataloader, device, output_dir, max_scenes=None,
                      max_samples=None, frame_skip=1, fps=4, save_frames=False,
                      video_height=None, mask_visuals=True, save_gaussian=True,
                      gaussian_opacity_thresh=0.1, gaussian_elev=25,
                      gaussian_azim=-60):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    num_cams = getattr(model, "num_cams", 6)
    all_scene_psnr = []

    with torch.no_grad():
        for scene_count, scene_batch in enumerate(scene_dataloader):
            if max_scenes is not None and scene_count >= max_scenes:
                break

            scene_name = scene_batch["scene_name"]
            sample_dataset, original_indices = build_sample_loader(scene_batch, max_samples, frame_skip)
            sample_loader = DataLoader(
                sample_dataset,
                batch_size=1,
                shuffle=False,
                num_workers=0,
                pin_memory=False,
                collate_fn=custom_collate_fn,
            )

            video_path = output_dir / f"{scene_name}_stage1_voxel_occ_v2.mp4"
            writer = None
            scene_psnrs = []
            print(f"Processing {scene_name}: {len(sample_dataset)} samples -> {video_path}")

            try:
                for local_idx, batch in enumerate(sample_loader):
                    sample_idx = original_indices[local_idx] if local_idx < len(original_indices) else local_idx
                    batch = to_device(batch, device)
                    recontrast_data, _, splating_data = render_batch(model, batch, local_idx)

                    board, mean_psnr = make_camera_grid(
                        splating_data,
                        num_cams=num_cams,
                        frame_id=0,
                        mask_visuals=mask_visuals,
                    )
                    if board is None:
                        print(f"  sample {sample_idx:04d}: missing render outputs, skipped")
                        continue

                    frame = make_video_frame(board, scene_name, sample_idx, mean_psnr, video_height)
                    if writer is None:
                        h, w = frame.shape[:2]
                        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                        writer = cv2.VideoWriter(str(video_path), fourcc, fps, (w, h))
                        if not writer.isOpened():
                            raise RuntimeError(f"Failed to open video writer: {video_path}")

                    writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                    if save_frames:
                        frame_dir = output_dir / scene_name / "frames"
                        frame_dir.mkdir(parents=True, exist_ok=True)
                        Image.fromarray(frame).save(frame_dir / f"sample_{sample_idx:04d}.png")

                    if save_gaussian:
                        gaussian_dir = output_dir / scene_name / "gaussian"
                        save_gaussian_scene_image(
                            recontrast_data,
                            gaussian_dir / f"sample_{sample_idx:04d}.png",
                            batch_idx=0,
                            opacity_thresh=gaussian_opacity_thresh,
                            elev=gaussian_elev,
                            azim=gaussian_azim,
                        )

                    if mean_psnr is not None:
                        scene_psnrs.append(mean_psnr)
                    print(f"  sample {sample_idx:04d}: PSNR={mean_psnr:.2f} dB" if mean_psnr else f"  sample {sample_idx:04d}")
            finally:
                if writer is not None:
                    writer.release()

            if scene_psnrs:
                all_scene_psnr.extend(scene_psnrs)
                print(f"Finished {scene_name}: mean PSNR={np.mean(scene_psnrs):.2f} dB")
            elif video_path.exists() and video_path.stat().st_size == 0:
                video_path.unlink()

    if all_scene_psnr:
        print(f"Overall mean PSNR: {np.mean(all_scene_psnr):.2f} dB over {len(all_scene_psnr)} samples")
    print(f"Videos saved under: {output_dir}")


def load_config(cfg_path):
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    cfg["data_cfg"]["batch_size"] = 1
    cfg["model_cfg"]["batch_size"] = 1
    cfg["model_cfg"]["use_stage1"] = True
    cfg["model_cfg"]["model_variant"] = "voxel"
    if "context_span" in cfg["data_cfg"]:
        cfg["model_cfg"]["context_span"] = cfg["data_cfg"]["context_span"]
    if "nuscenes_version" in cfg["data_cfg"]:
        cfg["model_cfg"]["nuscenes_version"] = cfg["data_cfg"]["nuscenes_version"]
    return cfg


def main():
    parser = argparse.ArgumentParser(description="Stage1 voxel-OCC visualization video generator")
    parser.add_argument("--cfg_path", type=str, default="configs/nuscenes/recondrive_stage1_voxel_occ_v2.yaml")
    parser.add_argument("--ckpt", type=str, default="checkpoints/recondrive_stage1.ckpt")
    parser.add_argument("--output_dir", type=str, default="work_dirs/stage1_voxel_occ_v2_vis")
    parser.add_argument("--device", type=str, default="0", help="CUDA index, cuda:N, or cpu")
    parser.add_argument("--max_scenes", type=int, default=None)
    parser.add_argument("--max_samples", type=int, default=None, help="Max samples per scene after frame_skip")
    parser.add_argument("--frame_skip", type=int, default=1, help="Use every Nth sample when building videos")
    parser.add_argument("--fps", type=int, default=4)
    parser.add_argument("--video_height", type=int, default=None, help="Optional output video height")
    parser.add_argument("--save_frames", action="store_true", help="Also save PNG frames")
    parser.add_argument("--no_save_gaussian", action="store_true", help="Do not save 3D Gaussian PNG views")
    parser.add_argument("--gaussian_opacity_thresh", type=float, default=0.1)
    parser.add_argument("--gaussian_elev", type=float, default=25)
    parser.add_argument("--gaussian_azim", type=float, default=-60)
    parser.add_argument("--no_mask_visuals", action="store_true", help="Show unmasked GT/render images")
    args = parser.parse_args()

    if args.device == "cpu" or not torch.cuda.is_available():
        device = torch.device("cpu")
    elif args.device.startswith("cuda"):
        device = torch.device(args.device)
    else:
        device = torch.device(f"cuda:{args.device}")

    cfg = load_config(args.cfg_path)

    print("Loading dataset...")
    data_module = VGGT4DGS_LITDataModule(cfg=cfg["data_cfg"])
    data_module.setup(stage="test")
    scene_dataloader = SimpleSceneDataLoader(data_module.test_dataset)

    print(f"Loading Stage1 voxel model from: {args.ckpt}")
    model = ReconDriveStage1_LITModelModule(cfg=cfg["model_cfg"], save_dir=args.output_dir, logger=None)
    model.load_pretrained_checkpoint(args.ckpt, strict=False, verbose=True)
    model.to(device)
    model.eval()

    run_visualization(
        model=model,
        scene_dataloader=scene_dataloader,
        device=device,
        output_dir=args.output_dir,
        max_scenes=args.max_scenes,
        max_samples=args.max_samples,
        frame_skip=args.frame_skip,
        fps=args.fps,
        save_frames=args.save_frames,
        video_height=args.video_height,
        mask_visuals=not args.no_mask_visuals,
        save_gaussian=not args.no_save_gaussian,
        gaussian_opacity_thresh=args.gaussian_opacity_thresh,
        gaussian_elev=args.gaussian_elev,
        gaussian_azim=args.gaussian_azim,
    )


if __name__ == "__main__":
    main()
