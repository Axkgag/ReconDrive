#!/usr/bin/env python3
"""Train VoxelFeatureVAE on ReconDrive voxel features."""

import argparse
import os
import shutil
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "models"))

torch.set_float32_matmul_precision("highest")
torch.multiprocessing.set_sharing_strategy("file_system")

from dataset.vggt4dgs_data_module import VGGT4DGS_LITDataModule
from models.recondrive_stage1_model import ReconDriveStage1_LITModelModule
from models.voxel_vae import VoxelFeatureVAE
from utils.snapshot import PIPELINE_DEPLOYMENT, save_pipeline_snapshot


def parse_args():
    parser = argparse.ArgumentParser(description="Train voxel VAE from ReconDrive voxel features.")
    parser.add_argument("--cfg_path", type=str, default="configs/nuscenes/voxel_vae.yaml")
    parser.add_argument("--work_dir", type=str, default=None)
    parser.add_argument("--pretrained_ckpt", type=str, default=None, help="Stage1/voxel backbone checkpoint.")
    parser.add_argument("--resume_from", type=str, default=None, help="Full VAE training checkpoint.")
    parser.add_argument("--load_vae_from", type=str, default=None, help="Load only VAE weights.")
    parser.add_argument("--devices", type=int, default=None, help="Reserved for compatibility; this script uses one process/device.")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--vis_interval", type=int, default=None)
    parser.add_argument("--max_train_steps", type=int, default=None, help="Optional debug limit per run.")
    parser.add_argument("--max_val_steps", type=int, default=None, help="Optional validation limit per epoch.")
    return parser.parse_args()


def to_device(data, device):
    if isinstance(data, dict):
        return {k: to_device(v, device) for k, v in data.items()}
    if isinstance(data, (list, tuple)):
        return type(data)(to_device(x, device) for x in data)
    if torch.is_tensor(data):
        return data.to(device, non_blocking=True)
    return data


def set_requires_grad(module, requires_grad):
    for param in module.parameters():
        param.requires_grad = requires_grad


def extract_stage1_inputs(stage1, batch):
    inputs = batch["context_frames"]
    images = inputs[("color_aug", 0)]
    intrinsics = inputs["K"]
    extrinsics = inputs["c2e_extr"]
    depth_override = None
    if getattr(stage1, "voxel_depth_source", "pred") == "gt":
        depth_override = stage1._prepare_voxel_depth_override(
            inputs.get("gt_depth", None),
            images.shape[-2:],
            images.device,
            images.dtype,
        )
    return images, intrinsics, extrinsics, depth_override



def get_sparse_voxel_feats(stage1_model, images, intrinsics, extrinsics, depth_maps_override=None):
    """Extract sparse voxel features while honoring optional GT depth override."""
    with torch.amp.autocast("cuda", enabled=images.is_cuda, dtype=torch.bfloat16):
        aggregated_tokens_list, patch_start_idx = stage1_model.aggregator(images.to(torch.bfloat16))

    with torch.amp.autocast("cuda", enabled=False):
        depth_maps, _ = stage1_model.depth_head(
            aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx
        )
        depth_maps = torch.nn.functional.sigmoid(torch.log(depth_maps))
        depth_range = stage1_model.max_depth - stage1_model.min_depth
        depth_maps = stage1_model.min_depth + depth_range * depth_maps

        depth_maps_for_voxel = depth_maps_override if depth_maps_override is not None else depth_maps
        if depth_maps_for_voxel.dim() == 5 and depth_maps_for_voxel.shape[-1] == 1:
            depth_maps_for_voxel = depth_maps_for_voxel[..., 0]

        voxel_feats, unique_coords, _, _, depth, shape, num_voxels = stage1_model.gs_head.extract_voxel_features(
            aggregated_tokens_list=aggregated_tokens_list,
            images=images,
            patch_start_idx=patch_start_idx,
            depth_maps=depth_maps_for_voxel,
            intrinsics=intrinsics,
            extrinsics=extrinsics,
        )
    return voxel_feats, unique_coords, depth, shape, num_voxels

def sparse_to_dense(voxel_feats, unique_coords, grid_size, batch_size):
    if voxel_feats is None or unique_coords is None or unique_coords.numel() == 0:
        raise RuntimeError("No valid voxels were produced for this batch.")
    nx, ny, nz = grid_size
    feat_dim = voxel_feats.shape[-1]
    dense = voxel_feats.new_zeros(batch_size, nx, ny, nz, feat_dim)
    mask = torch.zeros(batch_size, nx, ny, nz, dtype=torch.bool, device=voxel_feats.device)
    coords = unique_coords.long()
    dense[coords[:, 0], coords[:, 1], coords[:, 2], coords[:, 3]] = voxel_feats
    mask[coords[:, 0], coords[:, 1], coords[:, 2], coords[:, 3]] = True
    return dense, mask


def dense_to_sparse(dense, unique_coords):
    coords = unique_coords.long()
    return dense[coords[:, 0], coords[:, 1], coords[:, 2], coords[:, 3]]


def build_voxel_out(stage1_model, sparse_feats, unique_coords):
    sparse_feats = stage1_model.gs_head.refiner(sparse_feats, unique_coords)
    voxel_params = stage1_model.gs_head.decoder(sparse_feats)
    voxel_params = voxel_params.view(
        sparse_feats.shape[0],
        stage1_model.gs_head.gaussians_per_voxel,
        stage1_model.gs_head.raw_gs_dim,
    )
    return {
        "voxel_params": voxel_params,
        "unique_coords": unique_coords,
        "voxel_centers": stage1_model.gs_head._coords_to_world_pos(unique_coords),
    }


def render_from_sparse(stage1, sparse_feats, unique_coords, images, depth):
    voxel_out = build_voxel_out(stage1.model, sparse_feats, unique_coords)
    recontrast = stage1.model.recontrast_voxel_out(voxel_out, images, depth)
    recontrast["ae_global_points"] = True
    return recontrast


def tensor_to_uint8(tensor):
    arr = tensor.detach().cpu().float().clamp(0, 1).numpy()
    return (arr.transpose(1, 2, 0) * 255).astype(np.uint8)


def save_render_comparison(stage1, batch_idx, global_step, save_dir, original_splating, recon_splating, split, epoch):
    frame_id = 0
    out_dir = Path(save_dir) / f"{split}_visualizations" / f"epoch_{epoch:04d}"
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(
        nrows=stage1.num_cams,
        ncols=2,
        figsize=(8, stage1.num_cams * 2.3),
        dpi=120,
    )
    titles = ["Original voxel render", "VAE voxel render"]
    for cam_id in range(stage1.num_cams):
        key = ("gaussian_color", frame_id, cam_id)
        if key not in original_splating or key not in recon_splating:
            plt.close(fig)
            return
        cam_name = stage1.camera_names[cam_id] if cam_id < len(stage1.camera_names) else f"CAM_{cam_id}"
        axes[cam_id, 0].imshow(tensor_to_uint8(original_splating[key][0]))
        axes[cam_id, 1].imshow(tensor_to_uint8(recon_splating[key][0]))
        axes[cam_id, 0].set_ylabel(cam_name, fontsize=9)
        for col in range(2):
            axes[cam_id, col].set_xticks([])
            axes[cam_id, col].set_yticks([])
            if cam_id == 0:
                axes[cam_id, col].set_title(titles[col], fontsize=10)
    fig.tight_layout()
    fig.savefig(out_dir / f"step_{global_step:08d}_batch_{batch_idx:05d}.png", bbox_inches="tight")
    plt.close(fig)


@torch.no_grad()
def visualize_batch(stage1, vae, batch, batch_idx, global_step, save_dir, split, grid_size, epoch):
    stage1.eval()
    vae.eval()
    stage1._set_stage1_frame_ids()
    images, intrinsics, extrinsics, depth_override = extract_stage1_inputs(stage1, batch)
    voxel_feats, unique_coords, depth, _, _ = get_sparse_voxel_feats(
        stage1.model, images, intrinsics, extrinsics, depth_maps_override=depth_override
    )
    dense, _ = sparse_to_dense(voxel_feats, unique_coords, grid_size, images.shape[0])
    recon_dense, _, _ = vae(dense)
    recon_sparse = dense_to_sparse(recon_dense, unique_coords)

    original_recontrast = render_from_sparse(stage1, voxel_feats, unique_coords, images, depth)
    recon_recontrast = render_from_sparse(stage1, recon_sparse, unique_coords, images, depth)
    render_data = stage1.get_render_data(batch)
    original_splating = stage1.render_splating_imgs(original_recontrast, render_data)
    recon_splating = stage1.render_splating_imgs(recon_recontrast, render_data)
    save_render_comparison(stage1, batch_idx, global_step, save_dir, original_splating, recon_splating, split, epoch)
    vae.train(split == "train")


def load_stage1_checkpoint(stage1, ckpt_path):
    if not ckpt_path:
        return
    ckpt = Path(ckpt_path)
    if not ckpt.exists():
        print(f"[WARN] Stage1 checkpoint not found: {ckpt_path}")
        return
    print(f"Loading Stage1 checkpoint: {ckpt_path}")
    stage1.load_pretrained_checkpoint(str(ckpt), strict=False, verbose=True)


def load_vae_only(vae, ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt.get("vae_state_dict", ckpt.get("state_dict", ckpt))
    vae.load_state_dict(state, strict=False)
    print(f"Loaded VAE weights from: {ckpt_path}")


def save_checkpoint(path, epoch, global_step, vae, optimizer, scheduler, cfg):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "global_step": global_step,
            "vae_state_dict": vae.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "cfg": cfg,
        },
        path,
    )


def train_one_epoch(stage1, vae, loader, optimizer, scheduler, cfg, device, epoch, global_step, writer, save_dir, grid_size, args):
    vae.train()
    stage1.eval()
    vis_interval = args.vis_interval
    if vis_interval is None:
        vis_interval = int(cfg["model_cfg"].get("train_vis_interval", cfg.get("train_cfg", {}).get("vis_interval", 0)))
    log_every = int(cfg.get("train_cfg", {}).get("log_every_n_steps", 50))
    empty_weight = float(cfg.get("vae_cfg", {}).get("empty_weight", 0.1))
    max_steps = args.max_train_steps
    running = []

    pbar = tqdm(loader, desc=f"train epoch {epoch}", dynamic_ncols=True)
    for batch_idx, batch in enumerate(pbar):
        if max_steps is not None and batch_idx >= max_steps:
            break
        step_start = time.time()
        batch = to_device(batch, device)
        images, intrinsics, extrinsics, depth_override = extract_stage1_inputs(stage1, batch)

        with torch.no_grad():
            voxel_feats, unique_coords, _, _, _ = get_sparse_voxel_feats(
                stage1.model, images, intrinsics, extrinsics, depth_maps_override=depth_override
            )
            dense, mask = sparse_to_dense(voxel_feats, unique_coords, grid_size, images.shape[0])

        recon, z_mu, z_logvar = vae(dense)
        loss, recon_loss, kl_loss = vae.loss(
            recon, dense, z_mu, z_logvar, mask=mask, empty_weight=empty_weight
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(vae.parameters(), 1.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        global_step += 1
        loss_val = float(loss.detach().cpu())
        running.append(loss_val)
        lr = optimizer.param_groups[0]["lr"]
        writer.add_scalar("train/loss", loss_val, global_step)
        writer.add_scalar("train/recon", float(recon_loss.detach().cpu()), global_step)
        writer.add_scalar("train/kl", float(kl_loss.detach().cpu()), global_step)
        writer.add_scalar("train/lr", lr, global_step)
        writer.add_scalar("train/grad_norm", float(grad_norm.detach().cpu()), global_step)

        if batch_idx % max(log_every, 1) == 0:
            pbar.set_postfix(
                loss=f"{loss_val:.4f}",
                recon=f"{float(recon_loss.detach().cpu()):.4f}",
                kl=f"{float(kl_loss.detach().cpu()):.4f}",
                lr=f"{lr:.2e}",
                sec=f"{time.time() - step_start:.2f}",
            )

        if vis_interval and vis_interval > 0 and global_step % vis_interval == 0:
            visualize_batch(stage1, vae, batch, batch_idx, global_step, save_dir, "train", grid_size, epoch)
            vae.train()

    return global_step, float(np.mean(running)) if running else 0.0


@torch.no_grad()
def validate(stage1, vae, loader, cfg, device, epoch, global_step, writer, save_dir, grid_size, args):
    vae.eval()
    stage1.eval()
    empty_weight = float(cfg.get("vae_cfg", {}).get("empty_weight", 0.1))
    max_steps = args.max_val_steps
    val_vis_interval = int(cfg["model_cfg"].get("val_vis_interval", 200))
    max_vis = int(cfg["model_cfg"].get("val_vis_max_per_epoch", 4))
    saved_vis = 0
    losses, recons, kls = [], [], []

    for batch_idx, batch in enumerate(tqdm(loader, desc=f"val epoch {epoch}", dynamic_ncols=True)):
        if max_steps is not None and batch_idx >= max_steps:
            break
        batch = to_device(batch, device)
        images, intrinsics, extrinsics, depth_override = extract_stage1_inputs(stage1, batch)
        voxel_feats, unique_coords, _, _, _ = get_sparse_voxel_feats(
            stage1.model, images, intrinsics, extrinsics, depth_maps_override=depth_override
        )
        dense, mask = sparse_to_dense(voxel_feats, unique_coords, grid_size, images.shape[0])
        recon, z_mu, z_logvar = vae(dense)
        loss, recon_loss, kl_loss = vae.loss(
            recon, dense, z_mu, z_logvar, mask=mask, empty_weight=empty_weight
        )
        losses.append(float(loss.cpu()))
        recons.append(float(recon_loss.cpu()))
        kls.append(float(kl_loss.cpu()))

        if saved_vis < max_vis and val_vis_interval > 0 and batch_idx % val_vis_interval == 0:
            visualize_batch(stage1, vae, batch, batch_idx, global_step, save_dir, "val", grid_size, epoch)
            saved_vis += 1

    metrics = {
        "loss": float(np.mean(losses)) if losses else 0.0,
        "recon": float(np.mean(recons)) if recons else 0.0,
        "kl": float(np.mean(kls)) if kls else 0.0,
    }
    writer.add_scalar("val/loss", metrics["loss"], epoch)
    writer.add_scalar("val/recon", metrics["recon"], epoch)
    writer.add_scalar("val/kl", metrics["kl"], epoch)
    return metrics


def main():
    args = parse_args()
    with open(args.cfg_path) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    seed = args.seed if args.seed is not None else int(cfg.get("seed", 42))
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device(args.device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    save_dir = Path(args.work_dir or cfg.get("save_dir", "./work_dirs/voxel_vae"))
    log_dir = save_dir / "log"
    ckpt_dir = save_dir / "ckpt"
    code_dir = save_dir / "code"
    for directory in (log_dir, ckpt_dir, code_dir):
        directory.mkdir(parents=True, exist_ok=True)
    try:
        save_pipeline_snapshot(PIPELINE_DEPLOYMENT, str(code_dir))
    except Exception as exc:
        print(f"[WARN] Could not save pipeline snapshot: {exc}")
    shutil.copy2(args.cfg_path, save_dir / "cfg.yaml")

    cfg["model_cfg"]["batch_size"] = cfg["data_cfg"]["batch_size"]
    if cfg["model_cfg"].get("model_variant") != "voxel":
        raise ValueError("configs/nuscenes/voxel_vae.yaml must use model_cfg.model_variant='voxel'.")

    writer = SummaryWriter(str(log_dir / "tb_log"))
    data_module = VGGT4DGS_LITDataModule(cfg=cfg["data_cfg"])
    data_module.setup(stage="fit")
    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()

    stage1 = ReconDriveStage1_LITModelModule(cfg=cfg["model_cfg"], save_dir=str(log_dir), logger=None).to(device)
    load_stage1_checkpoint(stage1, args.pretrained_ckpt or cfg["model_cfg"].get("recondrive_ckpt"))
    stage1.eval()
    set_requires_grad(stage1, False)
    stage1._set_stage1_frame_ids()

    grid_size = stage1.model.gs_head._voxel_grid_size()
    nx, ny, nz = grid_size
    if nx != ny:
        raise ValueError(f"VoxelFeatureVAE expects square X/Y resolution, got {grid_size}.")

    vae_cfg = cfg.get("vae_cfg", {})
    vae = VoxelFeatureVAE(
        feature_dim=int(cfg["model_cfg"].get("voxel_feature_dim", 256)),
        voxel_depth=nz,
        resolution=nx,
        base_channel=int(vae_cfg.get("base_channel", 64)),
        latent_channels=int(vae_cfg.get("latent_channels", 64)),
        expansion=int(vae_cfg.get("expansion", 8)),
        ch_mult=tuple(vae_cfg.get("ch_mult", [1, 2, 4, 8])),
        num_res_blocks=int(vae_cfg.get("num_res_blocks", 2)),
        attn_resolutions=tuple(vae_cfg.get("attn_resolutions", [50])),
        dropout=float(vae_cfg.get("dropout", 0.0)),
        kl_weight=float(vae_cfg.get("kl_weight", 5e-5)),
    ).to(device)

    train_cfg = cfg.get("train_cfg", {})
    optimizer = torch.optim.AdamW(
        vae.parameters(),
        lr=float(train_cfg.get("lr", 1e-4)),
        betas=(0.9, 0.98),
        eps=1e-7,
        weight_decay=float(train_cfg.get("weight_decay", 0.01)),
    )
    steps_per_epoch = max(1, len(train_loader))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=max(1, int(cfg["model_cfg"].get("lr_restart_epoch", 5)) * steps_per_epoch),
        T_mult=int(cfg["model_cfg"].get("lr_restart_mult", 2)),
        eta_min=float(train_cfg.get("lr", 1e-4)) * float(cfg["model_cfg"].get("lr_min_factor", 0.01)) * 0.1,
    )

    start_epoch = 0
    global_step = 0
    if args.resume_from:
        ckpt = torch.load(args.resume_from, map_location=device)
        vae.load_state_dict(ckpt["vae_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer"])
        if ckpt.get("scheduler") is not None:
            scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = int(ckpt.get("epoch", -1)) + 1
        global_step = int(ckpt.get("global_step", 0))
        print(f"Resumed from {args.resume_from}: epoch={start_epoch}, global_step={global_step}")
    elif args.load_vae_from:
        load_vae_only(vae, args.load_vae_from, device)

    trainable = sum(p.numel() for p in vae.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in stage1.parameters() if p.requires_grad)
    print(f"Voxel grid: {grid_size}, VAE trainable params: {trainable:,}, Stage1 trainable params: {frozen:,}")

    max_epochs = int(train_cfg.get("max_epochs", cfg.get("train_epoch", 50)))
    save_every = int(train_cfg.get("save_every_epochs", 1))
    best_val = float("inf")

    for epoch in range(start_epoch, max_epochs):
        global_step, train_loss = train_one_epoch(
            stage1, vae, train_loader, optimizer, scheduler, cfg, device, epoch, global_step, writer, str(log_dir), grid_size, args
        )
        val_metrics = validate(stage1, vae, val_loader, cfg, device, epoch, global_step, writer, str(log_dir), grid_size, args)
        print(
            f"[Epoch {epoch}] train_loss={train_loss:.6f} "
            f"val_loss={val_metrics['loss']:.6f} val_recon={val_metrics['recon']:.6f} val_kl={val_metrics['kl']:.6f}"
        )

        latest_path = ckpt_dir / "latest.pth"
        save_checkpoint(latest_path, epoch, global_step, vae, optimizer, scheduler, cfg)
        if (epoch + 1) % max(save_every, 1) == 0:
            save_checkpoint(ckpt_dir / f"epoch_{epoch:02d}.pth", epoch, global_step, vae, optimizer, scheduler, cfg)
        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            save_checkpoint(ckpt_dir / "best.pth", epoch, global_step, vae, optimizer, scheduler, cfg)

    writer.close()


if __name__ == "__main__":
    main()
