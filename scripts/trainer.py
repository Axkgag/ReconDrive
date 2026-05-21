#----------------------------------------------------------------#
# ReconDrive                                                     #
# Source code: https://github.com/TuojingAI/ReconDrive           #
# Copyright (c) TuojingAI. All rights reserved.                  #
#----------------------------------------------------------------#

import yaml
import argparse
import os
import sys
import subprocess
import torch
from pathlib import Path
from pytorch_lightning.loggers import TensorBoardLogger

# Add models directory to path for vggt imports
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root / "models"))

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

from utils.train_callback import ExportBestModelCallback, ExportMetricCallback
from dataset.vggt4dgs_data_module import VGGT4DGS_LITDataModule
from models.recondrive_model import ReconDrive_LITModelModule


from utils.snapshot import save_pipeline_snapshot, PIPELINE_DEPLOYMENT

torch.set_float32_matmul_precision('highest')

# DataLoader workers ship batches to the main process via shared memory.
# Default 'file_descriptor' strategy goes through /dev/shm which is small in K8s
# pods; 'file_system' uses /tmp instead and avoids "No space left on device".
torch.multiprocessing.set_sharing_strategy('file_system')


def load_and_merge_configs(main_cfg_path):
    """Load and merge main config with sub-configs"""
    with open(main_cfg_path) as f:
        main_cfg = yaml.load(f, Loader=yaml.FullLoader)

    return main_cfg


def main():
    parser = argparse.ArgumentParser(description='eval argparse')
    parser.add_argument('--cfg_path', type=str, required=True, help='Main config file path')
    parser.add_argument('--pretrained_ckpt', type=str, default='')
    parser.add_argument('--train_4d', action='store_true', help='4dgs')
    parser.add_argument('--devices', type=int, default=None, help='Number of GPUs to use (overrides config)')
    parser.add_argument('--use_ae', action='store_true', help='Use ReconDriveAE_LITModelModule')
    parser.add_argument('--use_stage1', action='store_true', help='Use ReconDriveStage1_LITModelModule')
    parser.add_argument('--work_dir', type=str, default=None, help='Working directory for checkpoints and logs')
    parser.add_argument('--tensorboard_dir', type=str, default=None, help='TensorBoard log directory')
    parser.add_argument('--resume', action='store_true', help='Auto-resume from latest checkpoint in work_dir')
    args = parser.parse_args()

    with open(args.cfg_path) as f:
        main_cfg = yaml.load(f, Loader=yaml.FullLoader)

    main_cfg['model_cfg']['batch_size'] = main_cfg['data_cfg']['batch_size']

    # Override devices if specified via command line
    if args.devices is not None:
        main_cfg['devices'] = args.devices
        print(f"Using {args.devices} GPU(s) from command line (overriding config)")
    else:
        print(f"Using {main_cfg['devices']} GPU(s) from config file")

    # Setup directories
    if args.work_dir:
        save_dir = args.work_dir
        print(f"Using work_dir from command line: {save_dir}")
    else:
        save_dir = main_cfg['save_dir']
        print(f"Using work_dir from config: {save_dir}")

    log_dir = os.path.join(save_dir, 'log')
    ckpt_dir = os.path.join(save_dir, 'ckpt')
    code_dir = os.path.join(save_dir, 'code')
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(code_dir, exist_ok=True)
    save_pipeline_snapshot(PIPELINE_DEPLOYMENT, code_dir)
    with open(os.path.join(save_dir,'cfg.yaml'),'w') as fw:
        yaml.dump(main_cfg, fw)

    pl.seed_everything(main_cfg['seed'], workers=True)

    # Setup TensorBoard logger
    if args.tensorboard_dir:
        tensorboard_save_dir = args.tensorboard_dir
        print(f"Using TensorBoard directory from command line: {tensorboard_save_dir}")
    else:
        tensorboard_save_dir = log_dir
        print(f"Using TensorBoard directory: {tensorboard_save_dir}")

    logger = TensorBoardLogger(
        save_dir=tensorboard_save_dir,
        name='logs'
    )

    if args.train_4d:
        data_module = VGGT4DGS_LITDataModule(
            cfg=main_cfg['data_cfg'],
        )

    use_stage1 = args.use_stage1 or main_cfg['model_cfg'].get('use_stage1', False)
    use_ae = args.use_ae or main_cfg['model_cfg'].get('use_gaussian_ae', False)
    if use_stage1:
        from models.recondrive_stage1_model import ReconDriveStage1_LITModelModule
        model_cls = ReconDriveStage1_LITModelModule
    elif use_ae:
        from models.recondrive_ae_model import ReconDriveAE_LITModelModule
        model_cls = ReconDriveAE_LITModelModule
    else:
        model_cls = ReconDrive_LITModelModule
    print(f"Model class: {model_cls.__name__}")

    # Auto-detect latest checkpoint for resume
    resume_ckpt = None
    if args.resume:
        # Look for latest checkpoint in ckpt_dir
        ckpt_files = []
        if os.path.exists(ckpt_dir):
            # Check for last.ckpt first (PyTorch Lightning's default)
            last_ckpt = os.path.join(ckpt_dir, 'last.ckpt')
            if os.path.exists(last_ckpt):
                resume_ckpt = last_ckpt
                print(f"Found last.ckpt for resume: {resume_ckpt}")
            else:
                # Look for epoch_XX.ckpt files
                for f in os.listdir(ckpt_dir):
                    if f.startswith('epoch_') and f.endswith('.ckpt'):
                        ckpt_path = os.path.join(ckpt_dir, f)
                        ckpt_files.append((os.path.getmtime(ckpt_path), ckpt_path))

                if ckpt_files:
                    # Sort by modification time, get the latest
                    ckpt_files.sort(reverse=True)
                    resume_ckpt = ckpt_files[0][1]
                    print(f"Found latest checkpoint for resume: {resume_ckpt}")
                else:
                    print(f"No checkpoint found in {ckpt_dir}, starting from scratch")
        else:
            print(f"Checkpoint directory {ckpt_dir} does not exist, starting from scratch")

    litmodel = model_cls(
        cfg=main_cfg['model_cfg'],
        save_dir=log_dir,
        logger=logger
    )

    # Load checkpoint with priority: resume_ckpt > pretrained_ckpt
    if resume_ckpt:
        print(f"Resuming training from: {resume_ckpt}")
        # Use PyTorch Lightning's load_from_checkpoint for full resume (optimizer state, epoch, etc.)
        # But we need to load it differently - we'll pass it to trainer.fit() via ckpt_path
    elif args.pretrained_ckpt:
        print(f"Loading pretrained weights from: {args.pretrained_ckpt}")
        litmodel.load_pretrained_checkpoint(args.pretrained_ckpt, strict=False, verbose=True)

    checkpoint_callback = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename='best_module',
        save_top_k=1,
        monitor="val/psnr",
        mode="max",
        save_last=True,
        every_n_epochs=1
    )

    # Save checkpoint every epoch with epoch number in filename
    periodic_checkpoint_callback = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename='epoch_{epoch:02d}',
        save_top_k=-1,  # Save all epochs
        every_n_epochs=1,
        save_last=False,
    )

    export_metric_callback = ExportMetricCallback(
        export_dir=log_dir,
        monitor='all',
        best_metric_name='val/psnr',
        best_mode='max',
        start_after_epoch=1,
    )

    trainer = pl.Trainer(
        max_epochs=main_cfg.get('train_epoch', 50),
        accelerator="gpu",
        devices=main_cfg['devices'],
        precision="32-true",
        gradient_clip_algorithm="norm",
        accumulate_grad_batches=8,
        gradient_clip_val=1.0,
        callbacks=[checkpoint_callback, periodic_checkpoint_callback, LearningRateMonitor(), export_metric_callback],
        deterministic=True,
        log_every_n_steps=10,
        enable_progress_bar=True,
        enable_model_summary=True,
        strategy='ddp_find_unused_parameters_true',
        profiler="simple",
        logger=logger
    )

    torch.use_deterministic_algorithms(mode=True,warn_only=True)

    # Resume training from checkpoint if available, otherwise start fresh
    if resume_ckpt:
        print(f"\n{'='*60}")
        print(f"RESUMING TRAINING FROM CHECKPOINT")
        print(f"Checkpoint: {resume_ckpt}")
        print(f"{'='*60}\n")
        trainer.fit(litmodel, data_module, ckpt_path=resume_ckpt)
    else:
        if args.pretrained_ckpt:
            print(f"\n{'='*60}")
            print(f"STARTING TRAINING WITH PRETRAINED WEIGHTS")
            print(f"Pretrained checkpoint: {args.pretrained_ckpt}")
            print(f"{'='*60}\n")
        else:
            print(f"\n{'='*60}")
            print(f"STARTING TRAINING FROM SCRATCH")
            print(f"{'='*60}\n")
        trainer.fit(litmodel, data_module)

    data_module.setup(stage='test')

    print(f"\nTesting best model...{checkpoint_callback.best_model_path}")
    best_model = model_cls.load_from_checkpoint(checkpoint_callback.best_model_path)
    trainer.test(best_model, data_module)

if __name__ == "__main__":
    main()
