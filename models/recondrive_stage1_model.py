#----------------------------------------------------------------#
# ReconDrive                                                     #
# Source code: https://github.com/TuojingAI/ReconDrive           #
# Copyright (c) TuojingAI. All rights reserved.                  #
#----------------------------------------------------------------#

from models.recondrive_model import ReconDrive_LITModelModule


class ReconDriveStage1_LITModelModule(ReconDrive_LITModelModule):
    """Stage1 single-frame 3D Gaussian training module."""

    def __init__(self, cfg, save_dir='.', logger=None):
        super().__init__(cfg, save_dir, logger)
        self._configure_stage1_trainable()

    def _configure_stage1_trainable(self):
        # Freeze all parameters first.
        for param in self.model.parameters():
            param.requires_grad = False

        # Unfreeze depth_head and gs_head.
        for param in self.model.depth_head.parameters():
            param.requires_grad = True
        for param in self.model.gs_head.parameters():
            param.requires_grad = True

        # Unfreeze LoRA parameters in aggregator.
        for name, param in self.model.aggregator.named_parameters():
            if "lora_" in name:
                param.requires_grad = True

        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        lora_params = sum(
            p.numel()
            for n, p in self.model.aggregator.named_parameters()
            if p.requires_grad and "lora_" in n
        )
        depth_params = sum(p.numel() for p in self.model.depth_head.parameters() if p.requires_grad)
        gs_params = sum(p.numel() for p in self.model.gs_head.parameters() if p.requires_grad)
        print(
            "Stage1 trainable parameters:",
            f"total={total_params:,}, trainable={trainable_params:,},",
            f"lora={lora_params:,}, depth_head={depth_params:,}, gs_head={gs_params:,}",
        )

    def _set_stage1_frame_ids(self):
        self.all_render_frame_ids = [0]

    def training_step(self, batch_input, batch_idx):
        self.stage = stage = 'train'

        self._set_stage1_frame_ids()
        self._log_weights_and_grads(batch_input)

        batch_recontrast_data = self.get_recontrast_data(batch_input, batch_idx)
        loss_norm = self.compute_norm_loss(batch_recontrast_data)

        batch_render_data = self.get_render_data(batch_input)
        batch_splating_data = self.render_splating_imgs(batch_recontrast_data, batch_render_data)

        loss_gaussian = self.compute_gaussian_loss(batch_splating_data)

        loss_depth = self.compute_depth_loss(batch_splating_data)

        self.log(f'{stage}/gs', loss_gaussian.item(), on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log(f'{stage}/norm', loss_norm.item(), on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log(f'{stage}/depth', loss_depth.item(), on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)

        loss_all = loss_gaussian + loss_norm + loss_depth
        psnr, ssim, lpips = self.compute_reconstruction_metrics(batch_splating_data, stage)

        del batch_input, batch_recontrast_data, batch_render_data, batch_splating_data
        del psnr, ssim, lpips

        return loss_all

    def validation_step(self, batch_input, batch_idx):
        self.stage = stage = 'val'

        self._set_stage1_frame_ids()
        batch_recontrast_data = self.get_recontrast_data(batch_input)

        batch_render_data = self.get_render_data(batch_input)
        loss_norm = self.compute_norm_loss(batch_recontrast_data)

        batch_splating_data = self.render_splating_imgs(batch_recontrast_data, batch_render_data)
        loss_gaussian = self.compute_gaussian_loss(batch_splating_data)

        self.log(f'{stage}/gs', loss_gaussian.item(), on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log(f'{stage}/norm', loss_norm.item(), on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)

        loss_all = loss_gaussian + loss_norm
        psnr, ssim, lpips = self.compute_reconstruction_metrics(batch_splating_data, stage)

        del batch_input, batch_recontrast_data, batch_render_data, batch_splating_data
        del psnr, ssim, lpips

        return loss_all

    def test_step(self, batch_input, batch_idx):
        self.stage = stage = 'test'

        self._set_stage1_frame_ids()
        batch_recontrast_data = self.get_recontrast_data(batch_input)

        batch_render_data = self.get_render_data(batch_input)
        loss_norm = self.compute_norm_loss(batch_recontrast_data)

        batch_splating_data = self.render_splating_imgs(batch_recontrast_data, batch_render_data)
        loss_gaussian = self.compute_gaussian_loss(batch_splating_data)

        self.log(f'{stage}/gs', loss_gaussian.item(), on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log(f'{stage}/norm', loss_norm.item(), on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)

        loss_all = loss_gaussian + loss_norm
        psnr, ssim, lpips = self.compute_reconstruction_metrics(batch_splating_data, stage)

        del batch_input, batch_recontrast_data, batch_render_data, batch_splating_data
        del psnr, ssim, lpips

        return loss_all
