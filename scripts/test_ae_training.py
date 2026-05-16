#!/usr/bin/env python3
"""
Test script for ReconDriveAE training step.
Simulates a batch and runs through the full forward pass + loss computation.
"""

import torch
import yaml
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from models.recondrive_ae_model import ReconDriveAE_LITModelModule


def create_mock_batch(batch_size=1, num_cams=6, height=280, width=518, device='cuda:0'):
    """Create a mock batch_input that mimics the real data structure."""
    # context_frames: the input images and camera parameters
    context_frames = {
        ('color_aug', 0): torch.rand(batch_size, num_cams, 3, height, width, device=device),
        'c2e_extr': torch.eye(4, device=device).unsqueeze(0).unsqueeze(0).repeat(batch_size, num_cams, 1, 1),
        'K': torch.eye(3, device=device).unsqueeze(0).unsqueeze(0).repeat(batch_size, num_cams, 1, 1) * 300,  # focal length
        'ego_pose': torch.eye(4, device=device).unsqueeze(0).unsqueeze(0).repeat(batch_size, num_cams, 1, 1),
    }

    # Add realistic camera extrinsics (6 cameras around the vehicle)
    # Front, Front-Left, Front-Right, Back-Left, Back-Right, Back
    cam_positions = [
        [2.0, 0.0, 1.5],      # Front
        [1.5, -1.0, 1.5],     # Front-Left
        [1.5, 1.0, 1.5],      # Front-Right
        [-1.5, -1.0, 1.5],    # Back-Left
        [-1.5, 1.0, 1.5],     # Back-Right
        [-2.0, 0.0, 1.5],     # Back
    ]
    for i in range(num_cams):
        context_frames['c2e_extr'][0, i, 0, 3] = cam_positions[i][0]
        context_frames['c2e_extr'][0, i, 1, 3] = cam_positions[i][1]
        context_frames['c2e_extr'][0, i, 2, 3] = cam_positions[i][2]

    # all_dict: concatenated data for all frames
    all_dict = {
        ('color_aug', 0): context_frames[('color_aug', 0)],
        'c2e_extr': context_frames['c2e_extr'],
        'K': context_frames['K'],
        'ego_pose': context_frames['ego_pose'],
        'mask': torch.ones(batch_size, num_cams, height, width, device=device),
        'gt_depth': torch.rand(batch_size, num_cams, height, width, device=device) * 50 + 5.0,  # 5-55m depth
    }

    batch_input = {
        'context_frames': context_frames,
        'all_dict': all_dict,
    }

    return batch_input


def main():
    print("="*60)
    print("Testing ReconDriveAE training step")
    print("="*60)

    # Load config
    cfg_path = 'configs/nuscenes/recondrive_ae.yaml'
    print(f"\n1. Loading config from: {cfg_path}")
    with open(cfg_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    config['model_cfg']['batch_size'] = 1
    config['data_cfg']['batch_size'] = 1

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f"   Device: {device}")

    # Create model
    print("\n2. Creating ReconDriveAE model...")
    model = ReconDriveAE_LITModelModule(
        cfg=config['model_cfg'],
        save_dir='./temp_test',
        logger=None
    )
    model.to(device)
    model.eval()

    print(f"   ✓ Model created")
    print(f"   ✓ GaussianAE parameters: {sum(p.numel() for p in model.gaussian_ae.parameters()):,}")

    # Create mock batch
    print("\n3. Creating mock batch...")
    batch_input = create_mock_batch(
        batch_size=1,
        num_cams=config['model_cfg']['num_cams'],
        height=config['model_cfg']['height'],
        width=config['model_cfg']['width'],
        device=device
    )
    print(f"   ✓ Batch created")
    print(f"   ✓ Input images: {batch_input['context_frames'][('color_aug', 0)].shape}")

    # Run training step
    print("\n4. Running training_step (forward + loss)...")
    try:
        with torch.no_grad():
            # Initialize model state (like training_step does)
            print("   → set_normal_params()...")
            model.set_normal_params(batch_input)

            print("   → get_recontrast_data()...")
            batch_recontrast_data = model.get_recontrast_data(batch_input, batch_idx=0)
            print(f"     ✓ xyz: {batch_recontrast_data['xyz'].shape}")
            print(f"     ✓ opacity: {batch_recontrast_data['opacity_maps'].shape}")

            print("   → gaussian_ae.forward()...")
            batch_recontrast_recon, latent, gt_target = model.gaussian_ae.forward_with_targets(batch_recontrast_data)
            batch_recontrast_recon = model._attach_render_fields(batch_recontrast_recon)
            print(f"     ✓ latent features: {latent.F.shape}")
            print(f"     ✓ recon xyz: {batch_recontrast_recon['xyz'].shape}")

            print("   → gaussian_ae.loss_fn()...")
            loss_dict = model.gaussian_ae.loss_fn(batch_recontrast_recon, gt_target)
            print(f"     ✓ loss_attr: {loss_dict['total'].item():.6f}")
            print(f"       - xyz:     {loss_dict['xyz'].item():.6f}")
            print(f"       - rot:     {loss_dict['rot'].item():.6f}")
            print(f"       - scale:   {loss_dict['scale'].item():.6f}")
            print(f"       - opacity: {loss_dict['opacity'].item():.6f}")
            print(f"       - sh:      {loss_dict['sh'].item():.6f}")

            print("   → render_splating_imgs()...")
            batch_render_data = model.get_render_data(batch_input)
            batch_splating_data = model.render_splating_imgs(batch_recontrast_recon, batch_render_data)
            print(f"     ✓ rendered images: {len([k for k in batch_splating_data.keys() if 'gaussian_color' in str(k)])} views")

            print("   → compute_gaussian_loss()...")
            loss_render = model.compute_gaussian_loss(batch_splating_data)
            print(f"     ✓ loss_render: {loss_render.item():.6f}")

            loss_total = loss_dict['total'] + model.lambda_render * loss_render
            print(f"\n   ✓ TOTAL LOSS: {loss_total.item():.6f}")

        print("\n" + "="*60)
        print("✅ All checks passed! Training step works correctly.")
        print("="*60)

    except Exception as e:
        print(f"\n❌ Error during training_step:")
        print(f"   {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == '__main__':
    sys.exit(main())
