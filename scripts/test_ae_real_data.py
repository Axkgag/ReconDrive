#!/usr/bin/env python3
"""
Test ReconDriveAE with real data (1 sample from validation set).
"""

import torch
import yaml
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from models.recondrive_ae_model import ReconDriveAE_LITModelModule
from dataset.vggt3dgs_scene_data_module import VGGT3DGS_SceneDataModule


def main():
    print("="*60)
    print("Testing ReconDriveAE with real data")
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

    # Load data
    print("\n2. Loading validation dataset (1 sample)...")
    data_module = VGGT3DGS_SceneDataModule(cfg=config['data_cfg'])
    data_module.setup(stage='test')

    # Get first scene
    scene_dataloader = data_module.test_scene_dataloader()
    scene_batch = next(iter(scene_dataloader))
    scene_name = scene_batch['scene_name']
    print(f"   Scene: {scene_name}")

    # Get first sample from scene
    if 'samples' in scene_batch:
        from scripts.inference import SceneSampleDataset
        samples = scene_batch['samples'][:1]  # Only first sample
        sample_ds = SceneSampleDataset(samples)
    else:
        from scripts.inference import SceneSampleDataset
        indices = scene_batch['sample_indices'][:1]
        sample_ds = SceneSampleDataset(
            indices,
            dataset=scene_batch['dataset'],
            scene_idx=scene_batch['scene_idx'],
        )

    from torch.utils.data import DataLoader
    from dataset.vggt4dgs_scene_dataset import custom_collate_fn
    loader = DataLoader(sample_ds, batch_size=1, shuffle=False,
                       num_workers=0, collate_fn=custom_collate_fn)
    batch_input = next(iter(loader))

    # Move to device
    def to_device(data, device):
        if isinstance(data, dict):
            return {k: (v if k == 'vehicle_annotations' else to_device(v, device))
                    for k, v in data.items()}
        elif isinstance(data, (list, tuple)):
            return type(data)(to_device(x, device) for x in data)
        elif torch.is_tensor(data):
            return data.to(device)
        return data

    batch_input = to_device(batch_input, device)
    print(f"   ✓ Loaded 1 sample")

    # Create model
    print("\n3. Creating ReconDriveAE model...")
    model = ReconDriveAE_LITModelModule(
        cfg=config['model_cfg'],
        save_dir='./temp_test',
        logger=None
    )
    model.to(device)
    model.eval()
    print(f"   ✓ Model created")
    print(f"   ✓ GaussianAE parameters: {sum(p.numel() for p in model.gaussian_ae.parameters()):,}")

    # Run forward pass
    print("\n4. Running forward pass + loss computation...")
    try:
        with torch.no_grad():
            print("   → set_normal_params()...")
            model.set_normal_params(batch_input)

            print("   → get_recontrast_data()...")
            batch_recontrast_data = model.get_recontrast_data(batch_input, batch_idx=0)
            xyz = batch_recontrast_data['xyz']
            print(f"     ✓ xyz: {xyz.shape}")
            print(f"     ✓ xyz range X: [{xyz[..., 0].min():.2f}, {xyz[..., 0].max():.2f}]m")
            print(f"     ✓ xyz range Y: [{xyz[..., 1].min():.2f}, {xyz[..., 1].max():.2f}]m")
            print(f"     ✓ xyz range Z: [{xyz[..., 2].min():.2f}, {xyz[..., 2].max():.2f}]m")

            # Check how many points are in voxel range (COME-aligned)
            in_x = (xyz[..., 0] >= -40) & (xyz[..., 0] <= 40)
            in_y = (xyz[..., 1] >= -40) & (xyz[..., 1] <= 40)
            in_z = (xyz[..., 2] >= -1) & (xyz[..., 2] <= 5.4)
            in_range = in_x & in_y & in_z
            print(f"     ✓ Points in voxel range: {in_range.sum().item():,} / {xyz.shape[1]:,} ({100*in_range.sum().item()/xyz.shape[1]:.1f}%)")

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
        print(f"\n❌ Error:")
        print(f"   {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == '__main__':
    sys.exit(main())
