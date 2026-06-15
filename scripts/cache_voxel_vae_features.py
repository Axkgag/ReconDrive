#!/usr/bin/env python3
"""Cache a small subset of sparse voxel features and estimate full cache size."""

import argparse
import json
import shutil
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "models"))

torch.multiprocessing.set_sharing_strategy("file_system")

from dataset.vggt4dgs_data_module import VGGT4DGS_LITDataModule
from dataset.vggt4dgs_dataset import custom_collate_fn
from models.recondrive_stage1_model import ReconDriveStage1_LITModelModule
from scripts.train_voxel_vae import (
    extract_stage1_inputs,
    get_sparse_voxel_feats,
    load_stage1_checkpoint,
    set_requires_grad,
    to_device,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Cache sparse voxel features for VAE size estimation.")
    parser.add_argument("--cfg_path", type=str, default="configs/nuscenes/voxel_vae.yaml")
    parser.add_argument("--cache_dir", type=str, default=None, help="Defaults to <save_dir>/voxel_feature_cache_probe.")
    parser.add_argument("--pretrained_ckpt", type=str, default=None, help="Stage1/voxel backbone checkpoint.")
    parser.add_argument("--split", choices=["train", "val", "both"], default="train")
    parser.add_argument("--max_samples", type=int, default=32, help="Number of samples to cache per split.")
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="fp16")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def build_loader(dataset, batch_size, shuffle, num_workers):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=custom_collate_fn,
    )


def feature_dtype(name):
    if name == "fp16":
        return torch.float16
    if name == "bf16":
        return torch.bfloat16
    return torch.float32


def format_bytes(num_bytes):
    value = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024.0 or unit == "TB":
            return f"{value:.2f} {unit}"
        value /= 1024.0


def safe_token(token, fallback):
    if token is None:
        return fallback
    return str(token).replace("/", "_")


def save_sample_cache(path, voxel_feats, coords_xyz, grid_size, token, dtype):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "voxel_feats": voxel_feats.detach().cpu().to(dtype),
        "coords": coords_xyz.detach().cpu().to(torch.int16),
        "grid_size": torch.tensor(grid_size, dtype=torch.int16),
        "token": token,
    }
    torch.save(payload, path)


@torch.no_grad()
def cache_split(split, loader, dataset, stage1, cfg, device, cache_root, max_samples, dtype, overwrite):
    out_dir = cache_root / split
    out_dir.mkdir(parents=True, exist_ok=True)
    grid_size = stage1.model.gs_head._voxel_grid_size()
    total_samples = len(dataset)
    saved = []
    num_voxels = []
    processed = 0
    start = time.time()

    pbar = tqdm(loader, desc=f"cache {split}", dynamic_ncols=True)
    for batch_idx, batch in enumerate(pbar):
        if processed >= max_samples:
            break
        batch = to_device(batch, device)
        images, intrinsics, extrinsics, depth_override = extract_stage1_inputs(stage1, batch)
        voxel_feats, unique_coords, _, _, _ = get_sparse_voxel_feats(
            stage1.model, images, intrinsics, extrinsics, depth_maps_override=depth_override
        )
        coords = unique_coords.long()
        batch_size = images.shape[0]
        for local_idx in range(batch_size):
            if processed >= max_samples:
                break
            dataset_idx = batch_idx * batch_size + local_idx
            token = None
            if hasattr(dataset, "sample_tokens") and dataset_idx < len(dataset.sample_tokens):
                token = dataset.sample_tokens[dataset_idx]
            stem = f"{dataset_idx:08d}_{safe_token(token, 'sample')}"
            out_path = out_dir / f"{stem}.pt"
            if out_path.exists() and not overwrite:
                file_size = out_path.stat().st_size
                cached = torch.load(out_path, map_location="cpu")
                n_voxels = int(cached["voxel_feats"].shape[0])
            else:
                mask = coords[:, 0] == local_idx
                sample_feats = voxel_feats[mask]
                sample_coords = coords[mask][:, 1:4]
                save_sample_cache(out_path, sample_feats, sample_coords, grid_size, token, dtype)
                file_size = out_path.stat().st_size
                n_voxels = int(sample_feats.shape[0])
            saved.append(file_size)
            num_voxels.append(n_voxels)
            processed += 1
            pbar.set_postfix(samples=processed, avg_size=format_bytes(np.mean(saved)))

    avg_size = float(np.mean(saved)) if saved else 0.0
    summary = {
        "split": split,
        "cache_dir": str(out_dir),
        "dtype": str(dtype).replace("torch.", ""),
        "grid_size": list(map(int, grid_size)),
        "cached_samples": int(len(saved)),
        "dataset_samples": int(total_samples),
        "cached_total_bytes": int(np.sum(saved)) if saved else 0,
        "avg_bytes_per_sample": avg_size,
        "median_bytes_per_sample": float(np.median(saved)) if saved else 0.0,
        "min_bytes_per_sample": int(np.min(saved)) if saved else 0,
        "max_bytes_per_sample": int(np.max(saved)) if saved else 0,
        "avg_voxels_per_sample": float(np.mean(num_voxels)) if num_voxels else 0.0,
        "median_voxels_per_sample": float(np.median(num_voxels)) if num_voxels else 0.0,
        "min_voxels_per_sample": int(np.min(num_voxels)) if num_voxels else 0,
        "max_voxels_per_sample": int(np.max(num_voxels)) if num_voxels else 0,
        "estimated_full_bytes": int(avg_size * total_samples),
        "elapsed_sec": time.time() - start,
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(
        f"[{split}] cached {summary['cached_samples']}/{summary['dataset_samples']} samples, "
        f"avg={format_bytes(summary['avg_bytes_per_sample'])}, "
        f"estimated full={format_bytes(summary['estimated_full_bytes'])}, "
        f"avg_voxels={summary['avg_voxels_per_sample']:.0f}"
    )
    return summary


def main():
    args = parse_args()
    with open(args.cfg_path) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    seed = int(cfg.get("seed", 42))
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device(args.device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    cache_root = Path(args.cache_dir or Path(cfg.get("save_dir", "./work_dirs/voxel_vae")) / "voxel_feature_cache_probe")
    cache_root.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.cfg_path, cache_root / "cfg.yaml")

    cfg["model_cfg"]["batch_size"] = cfg["data_cfg"]["batch_size"]
    if args.batch_size is not None:
        cfg["data_cfg"]["batch_size"] = args.batch_size
        cfg["model_cfg"]["batch_size"] = args.batch_size
    cfg["data_cfg"]["data_shuffle"] = False
    cfg["data_cfg"]["drop_last"] = False

    data_module = VGGT4DGS_LITDataModule(cfg=cfg["data_cfg"])
    data_module.setup(stage="fit")

    stage1 = ReconDriveStage1_LITModelModule(cfg=cfg["model_cfg"], save_dir=str(cache_root), logger=None).to(device)
    load_stage1_checkpoint(stage1, args.pretrained_ckpt or cfg["model_cfg"].get("recondrive_ckpt"))
    stage1.eval()
    set_requires_grad(stage1, False)
    stage1._set_stage1_frame_ids()

    splits = ["train", "val"] if args.split == "both" else [args.split]
    summaries = {}
    dtype = feature_dtype(args.dtype)
    for split in splits:
        if split == "train":
            summaries[split] = cache_split(
                split,
                build_loader(data_module.train_dataset, cfg["data_cfg"]["batch_size"], False, args.num_workers),
                data_module.train_dataset,
                stage1,
                cfg,
                device,
                cache_root,
                args.max_samples,
                dtype,
                args.overwrite,
            )
        else:
            summaries[split] = cache_split(
                split,
                build_loader(data_module.val_dataset, cfg["data_cfg"]["batch_size"], False, args.num_workers),
                data_module.val_dataset,
                stage1,
                cfg,
                device,
                cache_root,
                args.max_samples,
                dtype,
                args.overwrite,
            )

    total_estimate = sum(item["estimated_full_bytes"] for item in summaries.values())
    with open(cache_root / "summary.json", "w") as f:
        json.dump({"splits": summaries, "estimated_total_bytes": total_estimate}, f, indent=2)
    print(f"Estimated selected split(s) total: {format_bytes(total_estimate)}")
    print(f"Summary saved to: {cache_root / 'summary.json'}")


if __name__ == "__main__":
    main()
