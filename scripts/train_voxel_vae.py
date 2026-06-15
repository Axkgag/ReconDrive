#!/usr/bin/env python3
"""Train VoxelFeatureVAE on ReconDrive voxel features."""

import argparse
import os
import shutil
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributed as dist
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "models"))

torch.set_float32_matmul_precision("highest")
torch.multiprocessing.set_sharing_strategy("file_system")

from dataset.vggt4dgs_data_module import VGGT4DGS_LITDataModule
from dataset.vggt4dgs_dataset import custom_collate_fn
from models.recondrive_stage1_model import ReconDriveStage1_LITModelModule
from models.voxel_vae import VoxelFeatureVAE
from utils.snapshot import PIPELINE_DEPLOYMENT, save_pipeline_snapshot


def parse_args():
    parser = argparse.ArgumentParser(description="Train voxel VAE from ReconDrive voxel features.")
    parser.add_argument("--cfg_path", type=str, default="configs/nuscenes/voxel_vae.yaml")
    parser.add_argument("--work_dir", type=str, default=None)
    parser.add_argument("--tb_dir", type=str, default=None, help="TensorBoard log directory. Defaults to <work_dir>/log/tb_log.")
    parser.add_argument("--pretrained_ckpt", type=str, default=None, help="Stage1/voxel backbone checkpoint.")
    parser.add_argument("--load_vae_from", type=str, default=None, help="Load only VAE weights.")
    parser.add_argument("--no_auto_resume", action="store_true", help="Do not auto-resume from <work_dir>/ckpt/last.ckpt.")
    parser.add_argument("--devices", type=int, default=None, help="Reserved for compatibility. Use torchrun --nproc_per_node for DDP.")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--vis_interval", type=int, default=None)
    parser.add_argument("--auto_scale_lr", action=argparse.BooleanOptionalAction, default=None, help="Scale VAE LR by DDP world size. Defaults to model_cfg.auto_scale_lr.")
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


def is_dist_avail_and_initialized():
    return dist.is_available() and dist.is_initialized()


def setup_distributed():
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        return 0, 0, 1, False
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ["WORLD_SIZE"])
    if not torch.cuda.is_available():
        raise RuntimeError("DDP mode requires CUDA. Launch single-process training for CPU.")
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", init_method="env://")
    return rank, local_rank, world_size, True


def cleanup_distributed():
    if is_dist_avail_and_initialized():
        dist.destroy_process_group()


def is_main_process():
    return not is_dist_avail_and_initialized() or dist.get_rank() == 0


def rank_zero_print(*args, **kwargs):
    if is_main_process():
        print(*args, **kwargs)


def unwrap_model(module):
    return module.module if isinstance(module, (torch.nn.DataParallel, DDP)) else module


def reduce_scalar(value, average=True):
    if not is_dist_avail_and_initialized():
        return float(value)
    tensor = torch.tensor(float(value), device=torch.cuda.current_device())
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    if average:
        tensor /= dist.get_world_size()
    return float(tensor.item())


def reduce_metrics(metrics):
    return {key: reduce_scalar(value, average=True) for key, value in metrics.items()}


def build_distributed_loader(dataset, batch_size, shuffle, drop_last, num_workers, distributed):
    sampler = None
    if distributed:
        sampler = DistributedSampler(dataset, shuffle=shuffle, drop_last=drop_last)
        shuffle = False
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        drop_last=drop_last,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=custom_collate_fn,
    )
    return loader, sampler


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



def extract_occ_voxel_feats(stage1_model, aggregated_tokens_list, images, patch_start_idx, intrinsics, extrinsics, occ_inputs):
    """Extract OCC voxel query features before Gaussian decoding."""
    gs_head = stage1_model.gs_head
    multi_level_feats = gs_head.build_multilevel_features(
        aggregated_tokens_list,
        images=images,
        patch_start_idx=patch_start_idx,
    )
    voxel_meta = gs_head.build_occ_queries(
        occ_inputs=occ_inputs,
        device=multi_level_feats[0].device,
        dtype=multi_level_feats[0].dtype,
    )
    voxel_centers = voxel_meta["voxel_centers"]
    batch_idx = voxel_meta["batch_idx"]
    voxel_coords = voxel_meta["voxel_coords"]
    coords_with_batch = torch.cat([batch_idx.unsqueeze(-1), voxel_coords], dim=-1)
    if voxel_centers.numel() == 0:
        return None, coords_with_batch, voxel_meta

    query = gs_head.occ_query_embed(gs_head.normalize_voxel_centers(voxel_centers))
    feature_h, feature_w = multi_level_feats[0].shape[-2:]
    reference_points, camera_mask = gs_head.project_voxels_to_cameras(
        voxel_centers=voxel_centers,
        batch_idx=batch_idx,
        intrinsics=intrinsics,
        extrinsics=extrinsics,
        feature_h=feature_h,
        feature_w=feature_w,
    )

    voxel_feats = query
    for layer in gs_head.occ_transformer:
        voxel_feats = layer(
            query=voxel_feats,
            coords_with_batch=coords_with_batch,
            multi_level_feats=multi_level_feats,
            reference_points=reference_points,
            batch_idx=batch_idx,
            camera_mask=camera_mask,
        )
    return voxel_feats, coords_with_batch, voxel_meta


def get_sparse_voxel_feats(stage1_model, images, intrinsics, extrinsics, depth_maps_override=None, occ_inputs=None):
    """Extract sparse voxel features while honoring the configured voxel source."""
    with torch.amp.autocast("cuda", enabled=images.is_cuda, dtype=torch.bfloat16):
        aggregated_tokens_list, patch_start_idx = stage1_model.aggregator(images.to(torch.bfloat16))

    with torch.amp.autocast("cuda", enabled=False):
        depth_maps, _ = stage1_model.depth_head(
            aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx
        )
        depth_maps = torch.nn.functional.sigmoid(torch.log(depth_maps))
        depth_range = stage1_model.max_depth - stage1_model.min_depth
        depth_maps = stage1_model.min_depth + depth_range * depth_maps

        if getattr(stage1_model, "voxel_feature_source", "depth_lift") == "occ_gt":
            voxel_feats, unique_coords, _ = extract_occ_voxel_feats(
                stage1_model,
                aggregated_tokens_list,
                images,
                patch_start_idx,
                intrinsics,
                extrinsics,
                occ_inputs,
            )
            shape = tuple(depth_maps.shape[:4])
            num_voxels = None if voxel_feats is None else voxel_feats.shape[0]
            return voxel_feats, unique_coords, depth_maps, shape, num_voxels

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


def build_sparse_recontrast(stage1_model, sparse_feats, unique_coords, batch_size, depth):
    gs_head = stage1_model.gs_head
    sparse_feats = gs_head.refiner(sparse_feats, unique_coords)
    raw = gs_head.decoder(sparse_feats).view(sparse_feats.shape[0], gs_head.gaussians_per_voxel, gs_head.raw_gs_dim)
    offset, rot, scale, opacity, sh = raw.split((3, 4, 3, 1, 3 * stage1_model.d_sh), dim=-1)

    offset = torch.tanh(offset) * (stage1_model.voxel_size * 0.5)
    rot = rot / (rot.norm(dim=-1, keepdim=True) + 1e-8)
    scale = torch.nn.functional.softplus(scale, beta=1) * 0.01
    opacity = torch.sigmoid(opacity)
    sh = sh.view(sh.shape[0], sh.shape[1], 3, stage1_model.d_sh) * stage1_model.sh_mask
    sh = sh.transpose(-1, -2).contiguous()

    coords = unique_coords.long()
    voxel_centers = gs_head.voxel_coords_to_world(coords[:, 1:], dtype=sparse_feats.dtype)
    xyz_sparse = voxel_centers.unsqueeze(1) + offset

    outputs = {
        "xyz": [],
        "rot_maps": [],
        "scale_maps": [],
        "opacity_maps": [],
        "sh_maps": [],
        "forward_flow": [],
        "sparse_gaussians": True,
        "ae_global_points": True,
    }
    for batch_id in range(batch_size):
        batch_mask = coords[:, 0] == batch_id
        xyz_b = xyz_sparse[batch_mask].reshape(-1, 3)
        outputs["xyz"].append(xyz_b)
        outputs["rot_maps"].append(rot[batch_mask].reshape(-1, 4))
        outputs["scale_maps"].append(scale[batch_mask].reshape(-1, 3))
        outputs["opacity_maps"].append(opacity[batch_mask].reshape(-1, 1))
        outputs["sh_maps"].append(sh[batch_mask].reshape(-1, sh.shape[-2], sh.shape[-1]))
        outputs["forward_flow"].append(xyz_b.new_zeros((xyz_b.shape[0], 3)))
    outputs["xyz_transformed"] = outputs["xyz"]
    outputs["rot_maps_transformed"] = outputs["rot_maps"]
    outputs["sh_maps_transformed"] = outputs["sh_maps"]
    if depth is not None:
        outputs["pred_depth_maps"] = depth.squeeze(-1) if depth.dim() == 5 else depth
    return outputs


def render_from_sparse(stage1, sparse_feats, unique_coords, images, depth):
    return build_sparse_recontrast(stage1.model, sparse_feats, unique_coords, images.shape[0], depth)


def tensor_to_uint8(tensor):
    arr = tensor.detach().cpu().float().clamp(0, 1).numpy()
    return (arr.transpose(1, 2, 0) * 255).astype(np.uint8)


def apply_occ_render_mask_for_visualization(stage1, splating_data, batch):
    splating_data = stage1._apply_occ_render_mask(splating_data, batch)
    return splating_data


def apply_mask_to_image(image, mask, invalid_value=0.0):
    if mask is None:
        return image
    if mask.dim() == 2:
        mask = mask.unsqueeze(0)
    if mask.dim() == 3 and mask.shape[0] == 1:
        mask = mask.expand_as(image)
    elif mask.dim() == 3 and mask.shape[0] != image.shape[0]:
        mask = mask[:1].expand_as(image)
    mask = mask.to(device=image.device, dtype=image.dtype)
    return (image * mask + invalid_value * (1.0 - mask)).clamp(0, 1)


def save_render_comparison(stage1, batch_idx, global_step, save_dir, original_splating, recon_splating, split, epoch, apply_occ_mask=True):
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
        mask_key = ("warped_mask", frame_id, cam_id)
        vis_mask = original_splating.get(mask_key, None) if apply_occ_mask else None
        if vis_mask is not None:
            vis_mask = vis_mask[0]
        original_img = apply_mask_to_image(original_splating[key][0], vis_mask)
        recon_img = apply_mask_to_image(recon_splating[key][0], vis_mask)
        axes[cam_id, 0].imshow(tensor_to_uint8(original_img))
        axes[cam_id, 1].imshow(tensor_to_uint8(recon_img))
        axes[cam_id, 0].set_ylabel(cam_name, fontsize=9)
        for col in range(2):
            axes[cam_id, col].set_xticks([])
            axes[cam_id, col].set_yticks([])
            if cam_id == 0:
                axes[cam_id, col].set_title(titles[col], fontsize=10)
    fig.tight_layout()
    fig.savefig(out_dir / f"epoch_{epoch:04d}_step_{global_step:08d}_batch_{batch_idx:05d}.png", bbox_inches="tight")
    plt.close(fig)


@torch.no_grad()
def visualize_batch(stage1, vae, batch, batch_idx, global_step, save_dir, split, grid_size, epoch):
    stage1.eval()
    vae.eval()
    stage1._set_stage1_frame_ids()
    images, intrinsics, extrinsics, depth_override = extract_stage1_inputs(stage1, batch)
    voxel_feats, unique_coords, depth, _, _ = get_sparse_voxel_feats(
        stage1.model,
        images,
        intrinsics,
        extrinsics,
        depth_maps_override=depth_override,
        occ_inputs=batch["context_frames"],
    )
    dense, _ = sparse_to_dense(voxel_feats, unique_coords, grid_size, images.shape[0])
    recon_dense, _, _ = unwrap_model(vae)(dense)
    recon_sparse = dense_to_sparse(recon_dense, unique_coords)

    original_recontrast = render_from_sparse(stage1, voxel_feats, unique_coords, images, depth)
    recon_recontrast = render_from_sparse(stage1, recon_sparse, unique_coords, images, depth)
    render_data = stage1.get_render_data(batch)
    original_splating = stage1.render_splating_imgs(original_recontrast, render_data)
    recon_splating = stage1.render_splating_imgs(recon_recontrast, render_data)
    original_splating = apply_occ_render_mask_for_visualization(stage1, original_splating, batch)
    recon_splating = apply_occ_render_mask_for_visualization(stage1, recon_splating, batch)
    save_render_comparison(stage1, batch_idx, global_step, save_dir, original_splating, recon_splating, split, epoch)
    vae.train(split == "train")


def load_stage1_checkpoint(stage1, ckpt_path):
    if not ckpt_path:
        return
    ckpt = Path(ckpt_path)
    if not ckpt.exists():
        rank_zero_print(f"[WARN] Stage1 checkpoint not found: {ckpt_path}")
        return
    rank_zero_print(f"Loading Stage1 checkpoint: {ckpt_path}")
    checkpoint = torch.load(str(ckpt), map_location=stage1.device)
    ckpt_cfg = checkpoint.get("hyper_parameters", {}).get("cfg", {})
    ckpt_model_cfg = ckpt_cfg.get("model_cfg", ckpt_cfg)
    ckpt_k = ckpt_model_cfg.get("voxel_gaussians_per_voxel")
    cur_k = getattr(stage1.model.gs_head, "gaussians_per_voxel", None)
    if ckpt_k is None and "state_dict" in checkpoint:
        decoder_bias = checkpoint["state_dict"].get("model.gs_head.decoder.2.bias")
        raw_gs_dim = getattr(stage1.model.gs_head, "raw_gs_dim", None)
        if decoder_bias is not None and raw_gs_dim:
            ckpt_k = decoder_bias.numel() // int(raw_gs_dim)
    if ckpt_k is not None and cur_k is not None and int(ckpt_k) != int(cur_k):
        raise ValueError(
            f"Checkpoint was trained with voxel_gaussians_per_voxel={ckpt_k}, "
            f"but current config uses {cur_k}. Update configs/nuscenes/voxel_vae.yaml "
            "or pass a matching --pretrained_ckpt."
        )
    stage1.load_pretrained_checkpoint(str(ckpt), strict=False, verbose=True)


def load_vae_only(vae, ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt.get("vae_state_dict", ckpt.get("state_dict", ckpt))
    unwrap_model(vae).load_state_dict(state, strict=False)
    rank_zero_print(f"Loaded VAE weights from: {ckpt_path}")


def load_vae_checkpoint(vae, optimizer, scheduler, ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    unwrap_model(vae).load_state_dict(ckpt["vae_state_dict"])
    if ckpt.get("optimizer") is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None and ckpt.get("scheduler") is not None:
        scheduler.load_state_dict(ckpt["scheduler"])
    start_epoch = int(ckpt.get("epoch", -1)) + 1
    global_step = int(ckpt.get("global_step", 0))
    best_val = float(ckpt.get("best_val", float("inf")))
    rank_zero_print(f"Loaded VAE checkpoint from: {ckpt_path} (epoch={start_epoch}, global_step={global_step})")
    return start_epoch, global_step, best_val


def save_vae_checkpoint(path, vae, optimizer, scheduler, cfg, epoch, global_step, best_val=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "epoch": epoch,
        "global_step": global_step,
        "vae_state_dict": unwrap_model(vae).state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "cfg": cfg,
    }
    if best_val is not None:
        payload["best_val"] = best_val
    torch.save(payload, path)


def train_one_epoch(stage1, vae, loader, optimizer, scheduler, cfg, device, epoch, global_step, writer, save_dir, grid_size, args):
    vae.train()
    stage1.eval()
    main_process = is_main_process()
    vis_interval = args.vis_interval
    if vis_interval is None:
        vis_interval = int(cfg["model_cfg"].get("train_vis_interval", cfg.get("train_cfg", {}).get("vis_interval", 0)))
    train_vis_first_batch = bool(cfg["model_cfg"].get("train_vis_first_batch_per_epoch", True))
    log_every = int(cfg.get("train_cfg", {}).get("log_every_n_steps", 50))
    empty_weight = float(cfg.get("vae_cfg", {}).get("empty_weight", 0.1))
    max_steps = args.max_train_steps
    running = []

    pbar = tqdm(loader, desc=f"train epoch {epoch}", dynamic_ncols=True, disable=not main_process)
    for batch_idx, batch in enumerate(pbar):
        if max_steps is not None and batch_idx >= max_steps:
            break
        step_start = time.time()
        batch = to_device(batch, device)
        images, intrinsics, extrinsics, depth_override = extract_stage1_inputs(stage1, batch)

        with torch.no_grad():
            voxel_feats, unique_coords, _, _, _ = get_sparse_voxel_feats(
                stage1.model,
                images,
                intrinsics,
                extrinsics,
                depth_maps_override=depth_override,
                occ_inputs=batch["context_frames"],
            )
            dense, mask = sparse_to_dense(voxel_feats, unique_coords, grid_size, images.shape[0])

        recon, z_mu, z_logvar = vae(dense)
        loss, recon_loss, kl_loss, loss_details = unwrap_model(vae).loss(
            recon, dense, z_mu, z_logvar, mask=mask, empty_weight=empty_weight, return_details=True
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
        if writer is not None:
            writer.add_scalar("train/loss", loss_val, global_step)
            writer.add_scalar("train/recon", float(recon_loss.detach().cpu()), global_step)
            writer.add_scalar("train/recon_occ", float(loss_details["recon_occ"].detach().cpu()), global_step)
            writer.add_scalar("train/recon_empty", float(loss_details["recon_empty"].detach().cpu()), global_step)
            writer.add_scalar("train/recon_occ_contrib", float(loss_details["recon_occ_contrib"].detach().cpu()), global_step)
            writer.add_scalar("train/recon_empty_contrib", float(loss_details["recon_empty_contrib"].detach().cpu()), global_step)
            writer.add_scalar("train/occ_ratio", float(loss_details["occ_ratio"].detach().cpu()), global_step)
            writer.add_scalar("train/kl", float(kl_loss.detach().cpu()), global_step)
            writer.add_scalar("train/lr", lr, global_step)
            writer.add_scalar("train/grad_norm", float(grad_norm.detach().cpu()), global_step)

        if main_process and batch_idx % max(log_every, 1) == 0:
            pbar.set_postfix(
                loss=f"{loss_val:.4f}",
                recon=f"{float(recon_loss.detach().cpu()):.4f}",
                occ=f"{float(loss_details['recon_occ'].detach().cpu()):.4f}",
                empty=f"{float(loss_details['recon_empty'].detach().cpu()):.4f}",
                occ_pct=f"{float(loss_details['occ_ratio'].detach().cpu()) * 100:.2f}",
                kl=f"{float(kl_loss.detach().cpu()):.4f}",
                lr=f"{lr:.2e}",
                sec=f"{time.time() - step_start:.2f}",
            )

        should_visualize = main_process and (
            (train_vis_first_batch and batch_idx == 0)
            or (vis_interval and vis_interval > 0 and global_step % vis_interval == 0)
        )
        if should_visualize:
            visualize_batch(stage1, vae, batch, batch_idx, global_step, save_dir, "train", grid_size, epoch)
            vae.train()

    train_loss = float(np.mean(running)) if running else 0.0
    return global_step, reduce_scalar(train_loss)


@torch.no_grad()
def validate(stage1, vae, loader, cfg, device, epoch, global_step, writer, save_dir, grid_size, args):
    vae.eval()
    stage1.eval()
    main_process = is_main_process()
    empty_weight = float(cfg.get("vae_cfg", {}).get("empty_weight", 0.1))
    max_steps = args.max_val_steps
    val_vis_interval = int(cfg["model_cfg"].get("val_vis_interval", 200))
    max_vis = int(cfg["model_cfg"].get("val_vis_max_per_epoch", 4))
    saved_vis = 0
    losses, recons, kls = [], [], []
    occ_recons, empty_recons, occ_contribs, empty_contribs, occ_ratios = [], [], [], [], []

    for batch_idx, batch in enumerate(tqdm(loader, desc=f"val epoch {epoch}", dynamic_ncols=True, disable=not main_process)):
        if max_steps is not None and batch_idx >= max_steps:
            break
        batch = to_device(batch, device)
        images, intrinsics, extrinsics, depth_override = extract_stage1_inputs(stage1, batch)
        voxel_feats, unique_coords, _, _, _ = get_sparse_voxel_feats(
            stage1.model,
            images,
            intrinsics,
            extrinsics,
            depth_maps_override=depth_override,
            occ_inputs=batch["context_frames"],
        )
        dense, mask = sparse_to_dense(voxel_feats, unique_coords, grid_size, images.shape[0])
        recon, z_mu, z_logvar = vae(dense)
        loss, recon_loss, kl_loss, loss_details = unwrap_model(vae).loss(
            recon, dense, z_mu, z_logvar, mask=mask, empty_weight=empty_weight, return_details=True
        )
        losses.append(float(loss.cpu()))
        recons.append(float(recon_loss.cpu()))
        kls.append(float(kl_loss.cpu()))
        occ_recons.append(float(loss_details["recon_occ"].cpu()))
        empty_recons.append(float(loss_details["recon_empty"].cpu()))
        occ_contribs.append(float(loss_details["recon_occ_contrib"].cpu()))
        empty_contribs.append(float(loss_details["recon_empty_contrib"].cpu()))
        occ_ratios.append(float(loss_details["occ_ratio"].cpu()))

        if main_process and saved_vis < max_vis and val_vis_interval > 0 and batch_idx % val_vis_interval == 0:
            visualize_batch(stage1, vae, batch, batch_idx, global_step, save_dir, "val", grid_size, epoch)
            saved_vis += 1

    metrics = {
        "loss": float(np.mean(losses)) if losses else 0.0,
        "recon": float(np.mean(recons)) if recons else 0.0,
        "kl": float(np.mean(kls)) if kls else 0.0,
        "recon_occ": float(np.mean(occ_recons)) if occ_recons else 0.0,
        "recon_empty": float(np.mean(empty_recons)) if empty_recons else 0.0,
        "recon_occ_contrib": float(np.mean(occ_contribs)) if occ_contribs else 0.0,
        "recon_empty_contrib": float(np.mean(empty_contribs)) if empty_contribs else 0.0,
        "occ_ratio": float(np.mean(occ_ratios)) if occ_ratios else 0.0,
    }
    metrics = reduce_metrics(metrics)
    if writer is not None:
        writer.add_scalar("val/loss", metrics["loss"], epoch)
        writer.add_scalar("val/recon", metrics["recon"], epoch)
        writer.add_scalar("val/recon_occ", metrics["recon_occ"], epoch)
        writer.add_scalar("val/recon_empty", metrics["recon_empty"], epoch)
        writer.add_scalar("val/recon_occ_contrib", metrics["recon_occ_contrib"], epoch)
        writer.add_scalar("val/recon_empty_contrib", metrics["recon_empty_contrib"], epoch)
        writer.add_scalar("val/occ_ratio", metrics["occ_ratio"], epoch)
        writer.add_scalar("val/kl", metrics["kl"], epoch)
    return metrics


def main():
    args = parse_args()
    rank, local_rank, world_size, distributed = setup_distributed()
    main_process = is_main_process()
    with open(args.cfg_path) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    seed = args.seed if args.seed is not None else int(cfg.get("seed", 42))
    seed = seed + rank
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if distributed:
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device(args.device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    save_dir = Path(args.work_dir or cfg.get("save_dir", "./work_dirs/voxel_vae"))
    log_dir = save_dir / "log"
    ckpt_dir = save_dir / "ckpt"
    code_dir = save_dir / "code"
    for directory in (log_dir, ckpt_dir, code_dir):
        directory.mkdir(parents=True, exist_ok=True)
    if main_process:
        try:
            save_pipeline_snapshot(PIPELINE_DEPLOYMENT, str(code_dir))
        except Exception as exc:
            print(f"[WARN] Could not save pipeline snapshot: {exc}")
        shutil.copy2(args.cfg_path, save_dir / "cfg.yaml")
    if distributed:
        dist.barrier()

    cfg["model_cfg"]["batch_size"] = cfg["data_cfg"]["batch_size"]
    if cfg["model_cfg"].get("model_variant") != "voxel":
        raise ValueError("configs/nuscenes/voxel_vae.yaml must use model_cfg.model_variant='voxel'.")

    tb_dir = Path(args.tb_dir) if args.tb_dir else log_dir / "tb_log"
    writer = None
    if main_process:
        tb_dir.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(str(tb_dir))
        print(f"TensorBoard log dir: {tb_dir}")
        print(f"DDP world_size={world_size}, distributed={distributed}")
    data_module = VGGT4DGS_LITDataModule(cfg=cfg["data_cfg"])
    data_module.setup(stage="fit")
    train_loader, train_sampler = build_distributed_loader(
        data_module.train_dataset,
        batch_size=int(cfg["data_cfg"]["batch_size"]),
        shuffle=bool(cfg["data_cfg"].get("data_shuffle", True)),
        drop_last=bool(cfg["data_cfg"].get("drop_last", False)),
        num_workers=int(cfg["data_cfg"].get("num_workers", 4)),
        distributed=distributed,
    )
    val_loader, val_sampler = build_distributed_loader(
        data_module.val_dataset,
        batch_size=int(cfg["data_cfg"]["batch_size"]),
        shuffle=False,
        drop_last=False,
        num_workers=int(cfg["data_cfg"].get("num_workers", 4)),
        distributed=distributed,
    )

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
    if distributed:
        vae = DDP(vae, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

    train_cfg = cfg.get("train_cfg", {})
    base_lr = float(train_cfg.get("lr", 1e-4))
    auto_scale_lr = args.auto_scale_lr
    if auto_scale_lr is None:
        auto_scale_lr = bool(cfg["model_cfg"].get("auto_scale_lr", False))
    lr = base_lr * world_size if distributed and auto_scale_lr else base_lr
    rank_zero_print(
        f"VAE lr: {lr:.6g} (base_lr={base_lr:.6g}, world_size={world_size}, auto_scale_lr={auto_scale_lr})"
    )
    optimizer = torch.optim.AdamW(
        vae.parameters(),
        lr=lr,
        betas=(0.9, 0.98),
        eps=1e-7,
        weight_decay=float(train_cfg.get("weight_decay", 0.01)),
    )
    steps_per_epoch = max(1, len(train_loader))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=max(1, int(cfg["model_cfg"].get("lr_restart_epoch", 5)) * steps_per_epoch),
        T_mult=int(cfg["model_cfg"].get("lr_restart_mult", 2)),
        eta_min=lr * float(cfg["model_cfg"].get("lr_min_factor", 0.01)) * 0.1,
    )
    rank_zero_print(f"Scheduler steps_per_epoch={steps_per_epoch}, T_0={scheduler.T_0}, eta_min={scheduler.eta_min:.6g}")

    start_epoch = 0
    global_step = 0
    best_val = float("inf")
    auto_resume_ckpt = ckpt_dir / "last.ckpt"
    if args.load_vae_from:
        load_vae_only(vae, args.load_vae_from, device)
        rank_zero_print("--load_vae_from is set; skip auto-resume optimizer/scheduler state.")
    elif not args.no_auto_resume and auto_resume_ckpt.exists():
        start_epoch, global_step, best_val = load_vae_checkpoint(
            vae, optimizer, scheduler, auto_resume_ckpt, device
        )
    elif not args.no_auto_resume:
        rank_zero_print(f"No auto-resume checkpoint found at: {auto_resume_ckpt}")

    trainable = sum(p.numel() for p in vae.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in stage1.parameters() if p.requires_grad)
    rank_zero_print(f"Voxel grid: {grid_size}, VAE trainable params: {trainable:,}, Stage1 trainable params: {frozen:,}")

    max_epochs = int(train_cfg.get("max_epochs", cfg.get("train_epoch", 50)))
    save_every = int(train_cfg.get("save_every_epochs", 1))

    for epoch in range(start_epoch, max_epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        if val_sampler is not None:
            val_sampler.set_epoch(epoch)
        global_step, train_loss = train_one_epoch(
            stage1, vae, train_loader, optimizer, scheduler, cfg, device, epoch, global_step, writer, str(log_dir), grid_size, args
        )
        val_metrics = validate(stage1, vae, val_loader, cfg, device, epoch, global_step, writer, str(log_dir), grid_size, args)
        if main_process:
            print(
                f"[Epoch {epoch}] train_loss={train_loss:.6f} "
                f"val_loss={val_metrics['loss']:.6f} val_recon={val_metrics['recon']:.6f} "
                f"val_occ={val_metrics['recon_occ']:.6f} val_empty={val_metrics['recon_empty']:.6f} "
                f"val_occ_contrib={val_metrics['recon_occ_contrib']:.6f} "
                f"val_empty_contrib={val_metrics['recon_empty_contrib']:.6f} "
                f"val_occ_ratio={val_metrics['occ_ratio'] * 100:.2f}% val_kl={val_metrics['kl']:.6f}"
            )

            is_best = val_metrics["loss"] < best_val
            if is_best:
                best_val = val_metrics["loss"]

            last_path = ckpt_dir / "last.ckpt"
            save_vae_checkpoint(last_path, vae, optimizer, scheduler, cfg, epoch, global_step, best_val=best_val)
            if (epoch + 1) % max(save_every, 1) == 0:
                save_vae_checkpoint(
                    ckpt_dir / f"epoch_{epoch:02d}.ckpt", vae, optimizer, scheduler, cfg, epoch, global_step, best_val=best_val
                )
            if is_best:
                save_vae_checkpoint(ckpt_dir / "best.ckpt", vae, optimizer, scheduler, cfg, epoch, global_step, best_val=best_val)
        if distributed:
            dist.barrier()

    if writer is not None:
        writer.close()
    cleanup_distributed()


if __name__ == "__main__":
    main()
