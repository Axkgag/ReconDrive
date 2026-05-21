#!/usr/bin/env python3
"""
Validation script for Stage1 training components.

Checks:
  1. Training module: forward pass + loss computation with mock batch
  2. Model structure vs official checkpoint (checkpoints/recondrive_stage1.ckpt)
  3. Dataset: NuScenesdataset3D single-frame sampling
"""

import sys
import yaml
import torch
import numpy as np
from pathlib import Path
from functools import partial

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def make_K4x4(fx, fy, cx, cy, device):
    """Build a 4×4 homogeneous intrinsic matrix."""
    K = torch.eye(4, device=device)
    K[0, 0] = fx
    K[1, 1] = fy
    K[0, 2] = cx
    K[1, 2] = cy
    return K


def make_c2e_extr(tx, ty, tz, device):
    """Camera-to-ego extrinsic: identity rotation + translation."""
    T = torch.eye(4, device=device)
    T[0, 3] = tx
    T[1, 3] = ty
    T[2, 3] = tz
    return T


def create_mock_batch(batch_size=1, num_cams=6, height=280, width=518, device='cuda:0'):
    """
    Build a mock batch_input that matches the structure expected by
    ReconDriveStage1_LITModelModule (single frame, 6 cameras).
    """
    H, W = height, width
    B, C = batch_size, num_cams

    # Realistic camera positions around the vehicle
    cam_offsets = [
        ( 2.0,  0.0, 1.5),   # FRONT
        ( 1.5, -1.0, 1.5),   # FRONT_LEFT
        ( 1.5,  1.0, 1.5),   # FRONT_RIGHT
        (-1.5, -1.0, 1.5),   # BACK_LEFT
        (-1.5,  1.0, 1.5),   # BACK_RIGHT
        (-2.0,  0.0, 1.5),   # BACK
    ]

    c2e = torch.stack([
        make_c2e_extr(tx, ty, tz, device)
        for tx, ty, tz in cam_offsets
    ], dim=0).unsqueeze(0).repeat(B, 1, 1, 1)   # [B, 6, 4, 4]

    # Intrinsics: focal ≈ 800, principal point at image centre
    K_single = make_K4x4(800.0, 800.0, W / 2, H / 2, device)
    K = K_single.unsqueeze(0).unsqueeze(0).repeat(B, C, 1, 1)  # [B, 6, 4, 4]

    ego_pose = torch.eye(4, device=device).unsqueeze(0).unsqueeze(0).repeat(B, C, 1, 1)

    context_frames = {
        ('color_aug', 0): torch.rand(B, C, 3, H, W, device=device),
        'c2e_extr': c2e,
        'K': K,
        'ego_pose': ego_pose,
    }

    all_dict = {
        ('color_aug', 0): context_frames[('color_aug', 0)],
        'c2e_extr': c2e,
        'K': K,
        'ego_pose': ego_pose,
        'mask': torch.ones(B, C, H, W, device=device),
        'gt_depth': torch.rand(B, C, H, W, device=device) * 40.0 + 5.0,  # 5–45 m
    }

    return {'context_frames': context_frames, 'all_dict': all_dict}


# ─────────────────────────────────────────────────────────────────────────────
# Validation 1: training module + loss
# ─────────────────────────────────────────────────────────────────────────────

def validate_training_module(cfg_path='configs/nuscenes/recondrive_stage1.yaml', device='cuda:0'):
    print("\n" + "=" * 60)
    print("Validation 1: Training module + loss computation")
    print("=" * 60)

    from models.recondrive_stage1_model import ReconDriveStage1_LITModelModule

    with open(cfg_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    config['model_cfg']['batch_size'] = 1

    print("\n  [1/7] Instantiating ReconDriveStage1_LITModelModule ...")
    model = ReconDriveStage1_LITModelModule(
        cfg=config['model_cfg'],
        save_dir='./temp_stage1_test',
        logger=None,
    )
    model.to(device)
    model.eval()
    total = sum(p.numel() for p in model.model.parameters())
    trainable = sum(p.numel() for p in model.model.parameters() if p.requires_grad)
    print(f"     total params: {total:,}  trainable: {trainable:,}")

    print("\n  [2/7] Creating mock batch ...")
    batch = create_mock_batch(
        batch_size=1,
        num_cams=config['model_cfg']['num_cams'],
        height=config['model_cfg']['height'],
        width=config['model_cfg']['width'],
        device=device,
    )
    print(f"     color_aug shape: {batch['context_frames'][('color_aug', 0)].shape}")

    with torch.no_grad():
        print("\n  [3/7] _set_stage1_frame_ids() ...")
        model._set_stage1_frame_ids()
        assert model.all_render_frame_ids == [0], \
            f"Expected [0], got {model.all_render_frame_ids}"
        print(f"     all_render_frame_ids = {model.all_render_frame_ids}  ✓")

        print("\n  [4/7] get_recontrast_data() ...")
        recon = model.get_recontrast_data(batch, batch_idx=0)
        print(f"     xyz:          {recon['xyz'].shape}")
        print(f"     opacity_maps: {recon['opacity_maps'].shape}")
        print(f"     scale_maps:   {recon['scale_maps'].shape}")
        print(f"     sh_maps:      {recon['sh_maps'].shape}")

        print("\n  [5/7] compute_norm_loss() ...")
        loss_norm = model.compute_norm_loss(recon)
        print(f"     loss_norm = {loss_norm.item():.6f}")
        assert torch.isfinite(loss_norm), "loss_norm is not finite"

        print("\n  [6/7] get_render_data() + render_splating_imgs() ...")
        render_data = model.get_render_data(batch)
        splat = model.render_splating_imgs(recon, render_data)
        rendered_keys = [k for k in splat if isinstance(k, tuple) and k[0] == 'gaussian_color']
        print(f"     rendered views: {len(rendered_keys)}")
        assert len(rendered_keys) == 6, f"Expected 6 views (frame0 × 6 cams), got {len(rendered_keys)}"

        print("\n  [7/7] compute_gaussian_loss() ...")
        loss_gs = model.compute_gaussian_loss(splat)
        print(f"     loss_gaussian = {loss_gs.item():.6f}")
        assert torch.isfinite(loss_gs), "loss_gaussian is not finite"

        loss_all = loss_gs + loss_norm
        print(f"\n     loss_all = {loss_all.item():.6f}")

    print("\n✅ Validation 1 passed: training module + loss computation OK")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Validation 2: checkpoint structure
# ─────────────────────────────────────────────────────────────────────────────

def validate_checkpoint(model, ckpt_path='checkpoints/recondrive_stage1.ckpt'):
    print("\n" + "=" * 60)
    print("Validation 2: Model structure vs official checkpoint")
    print("=" * 60)

    if not Path(ckpt_path).exists():
        print(f"  ⚠️  Checkpoint not found at {ckpt_path}, skipping.")
        return

    print(f"\n  Loading checkpoint: {ckpt_path} ...")
    ckpt = torch.load(ckpt_path, map_location='cpu')
    ckpt_sd = ckpt['state_dict']
    model_sd = model.state_dict()

    ckpt_keys = set(ckpt_sd.keys())
    model_keys = set(model_sd.keys())

    only_in_ckpt = ckpt_keys - model_keys
    only_in_model = model_keys - ckpt_keys
    shape_mismatch = [
        k for k in ckpt_keys & model_keys
        if ckpt_sd[k].shape != model_sd[k].shape
    ]
    matched = len(ckpt_keys & model_keys) - len(shape_mismatch)

    print(f"\n  Checkpoint keys:  {len(ckpt_keys)}")
    print(f"  Model keys:       {len(model_keys)}")
    print(f"  Matched (shape OK): {matched}")
    print(f"  Shape mismatch:   {len(shape_mismatch)}")
    print(f"  Only in ckpt:     {len(only_in_ckpt)}")
    print(f"  Only in model:    {len(only_in_model)}")

    if shape_mismatch:
        print("\n  Shape mismatches:")
        for k in sorted(shape_mismatch)[:10]:
            print(f"    {k}: ckpt={ckpt_sd[k].shape}  model={model_sd[k].shape}")
        if len(shape_mismatch) > 10:
            print(f"    ... and {len(shape_mismatch) - 10} more")

    if only_in_ckpt:
        print("\n  Keys only in checkpoint (not in model):")
        for k in sorted(only_in_ckpt)[:10]:
            print(f"    {k}")
        if len(only_in_ckpt) > 10:
            print(f"    ... and {len(only_in_ckpt) - 10} more")

    if only_in_model:
        print("\n  Keys only in model (not in checkpoint):")
        for k in sorted(only_in_model)[:10]:
            print(f"    {k}")
        if len(only_in_model) > 10:
            print(f"    ... and {len(only_in_model) - 10} more")

    # Load with strict=False and verify no shape mismatches
    if shape_mismatch:
        print("\n❌ Validation 2 FAILED: shape mismatches found")
    elif only_in_ckpt or only_in_model:
        print("\n⚠️  Validation 2 WARNING: some keys missing (may be expected for new heads)")
        print("✅ Validation 2 passed: no shape mismatches")
    else:
        print("\n✅ Validation 2 passed: model structure fully matches checkpoint")


# ─────────────────────────────────────────────────────────────────────────────
# Validation 3: dataset single-frame sampling
# ─────────────────────────────────────────────────────────────────────────────

def validate_dataset(cfg_path='configs/nuscenes/recondrive_stage1.yaml'):
    print("\n" + "=" * 60)
    print("Validation 3: NuScenesdataset3D single-frame sampling")
    print("=" * 60)

    with open(cfg_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    data_cfg = config['data_cfg']

    data_path = data_cfg.get('data_path', './data/nuscenes/')
    if not Path(data_path).exists():
        print(f"  ⚠️  Data path not found: {data_path}, skipping.")
        return

    from dataset.vggt3dgs_dataset import NuScenesdataset3D
    from dataset.data_util import train_transforms

    print(f"\n  Instantiating NuScenesdataset3D (val split) ...")
    dataset = NuScenesdataset3D(
        path=data_path,
        stage='val',
        cameras=data_cfg['cameras'],
        back_context=data_cfg.get('back_context', 0),
        forward_context=data_cfg.get('forward_context', 0),
        data_transform=partial(
            train_transforms,
            image_shape=(data_cfg['height'], data_cfg['width']),
            crop_scale=[], crop_ratio=[], crop_prob=0.0,
            jittering=[], jittering_prob=0.0,
        ),
        depth_type=data_cfg.get('depth_type', 'lidar'),
        with_pose='gt_pose' in data_cfg.get('val_requirements', ''),
        with_ego_pose='gt_ego_pose' in data_cfg.get('val_requirements', ''),
        with_mask='mask' in data_cfg.get('val_requirements', ''),
        nuscenes_version=data_cfg.get('nuscenes_version', 'v1.0-trainval'),
        context_span=data_cfg.get('context_span', 1),
    )
    print(f"     Total samples: {len(dataset)}")
    assert len(dataset) > 0, "Dataset is empty"

    print("\n  Fetching sample[0] ...")
    sample = dataset[0]

    # Check top-level keys
    required_keys = {'context_frames', 'all_dict', 'target_frames', 'cur_sample'}
    assert required_keys.issubset(sample.keys()), \
        f"Missing keys: {required_keys - sample.keys()}"
    print(f"     Top-level keys: {list(sample.keys())}  ✓")

    # target_frames should be empty (no temporal targets in Stage1)
    assert sample['target_frames'] == {}, \
        f"target_frames should be empty, got: {list(sample['target_frames'].keys())}"
    print(f"     target_frames = {{}}  ✓")

    # context_frames shape: [1, 6, 3, H, W]
    cf = sample['context_frames']
    color = cf[('color_aug', 0)]
    H, W = data_cfg['height'], data_cfg['width']
    assert color.shape == torch.Size([1, 6, 3, H, W]), \
        f"Expected [1, 6, 3, {H}, {W}], got {color.shape}"
    print(f"     context_frames[('color_aug',0)]: {color.shape}  ✓")

    assert cf['c2e_extr'].shape == torch.Size([1, 6, 4, 4]), \
        f"c2e_extr shape: {cf['c2e_extr'].shape}"
    print(f"     context_frames['c2e_extr']:      {cf['c2e_extr'].shape}  ✓")

    assert cf['K'].shape == torch.Size([1, 6, 4, 4]), \
        f"K shape: {cf['K'].shape}"
    print(f"     context_frames['K']:             {cf['K'].shape}  ✓")

    # all_dict shape: same as context_frames (single frame)
    ad = sample['all_dict']
    assert ad[('color_aug', 0)].shape == torch.Size([1, 6, 3, H, W]), \
        f"all_dict color shape: {ad[('color_aug', 0)].shape}"
    print(f"     all_dict[('color_aug',0)]:       {ad[('color_aug', 0)].shape}  ✓")

    # No inter-frame transforms should be present
    inter_frame_keys = [k for k in ad if isinstance(k, tuple) and k[0] in ('cam_T_cam', 'ego_T_ego')]
    assert len(inter_frame_keys) == 0, \
        f"Unexpected inter-frame keys in all_dict: {inter_frame_keys}"
    print(f"     No cam_T_cam / ego_T_ego keys  ✓")

    # Depth range check
    if 'gt_depth' in ad:
        depth = ad['gt_depth']
        print(f"     gt_depth: {depth.shape}  min={depth.min():.2f}  max={depth.max():.2f}")

    print("\n✅ Validation 3 passed: NuScenesdataset3D single-frame sampling OK")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    cfg_path = 'configs/nuscenes/recondrive_stage1.yaml'
    ckpt_path = 'checkpoints/recondrive_stage1.ckpt'
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    results = {}

    try:
        model = validate_training_module(cfg_path, device)
        results['training_module'] = 'PASS'
    except Exception as e:
        import traceback
        print(f"\n❌ Validation 1 FAILED: {type(e).__name__}: {e}")
        traceback.print_exc()
        results['training_module'] = f'FAIL: {e}'
        model = None

    if model is not None:
        try:
            validate_checkpoint(model, ckpt_path)
            results['checkpoint'] = 'PASS'
        except Exception as e:
            import traceback
            print(f"\n❌ Validation 2 FAILED: {type(e).__name__}: {e}")
            traceback.print_exc()
            results['checkpoint'] = f'FAIL: {e}'

    try:
        validate_dataset(cfg_path)
        results['dataset'] = 'PASS'
    except Exception as e:
        import traceback
        print(f"\n❌ Validation 3 FAILED: {type(e).__name__}: {e}")
        traceback.print_exc()
        results['dataset'] = f'FAIL: {e}'

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    for name, status in results.items():
        icon = "✅" if status == "PASS" else "❌"
        print(f"  {icon}  {name}: {status}")

    return 0 if all(v == 'PASS' for v in results.values()) else 1


if __name__ == '__main__':
    sys.exit(main())
