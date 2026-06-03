from typing import Sequence, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


def nonlinearity(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)


def Normalize(in_channels: int) -> nn.GroupNorm:
    num_groups = in_channels // 4 if in_channels <= 32 else 32
    return nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True)


class Upsample(nn.Module):
    def __init__(self, in_channels: int, with_conv: bool) -> None:
        super().__init__()
        self.with_conv = with_conv
        if with_conv:
            self.conv = nn.Conv2d(in_channels, in_channels, 3, 1, 1)

    def forward(self, x: torch.Tensor, shape: Tuple[int, int]) -> torch.Tensor:
        do_rearrange = False
        if x.dim() == 5:
            b, c, t, h, w = x.size()
            do_rearrange = True
            x = rearrange(x, "b c t h w -> (b t) c h w")
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        diff_y = shape[0] - x.size(2)
        diff_x = shape[1] - x.size(3)
        x = F.pad(x, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2])
        if self.with_conv:
            x = self.conv(x)
        if do_rearrange:
            x = rearrange(x, "(b t) c h w -> b c t h w", b=b, t=t)
        return x


class Downsample(nn.Module):
    def __init__(self, in_channels: int, with_conv: bool) -> None:
        super().__init__()
        self.with_conv = with_conv
        if with_conv:
            self.conv = nn.Conv2d(in_channels, in_channels, 3, 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.with_conv:
            return self.conv(x)
        return F.avg_pool2d(x, kernel_size=2, stride=2)


class ResnetBlock(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: Optional[int] = None,
        conv_shortcut: bool = False,
        dropout: float,
        temb_channels: int = 0,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        if temb_channels > 0:
            self.temb_proj = nn.Linear(temb_channels, out_channels)
        self.norm2 = Normalize(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
            else:
                self.nin_shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x: torch.Tensor, temb: Optional[torch.Tensor] = None) -> torch.Tensor:
        h = self.norm1(x)
        h = nonlinearity(h)
        h = self.conv1(h)
        if temb is not None:
            h = h + self.temb_proj(nonlinearity(temb))[:, :, None, None]
        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)
        return x + h


class ResnetBlock3D(nn.Module):
    def __init__(self, *, in_channels: int, out_channels: Optional[int] = None, conv_shortcut: bool = False, dropout: float) -> None:
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels)
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = Normalize(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
            else:
                self.nin_shortcut = nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        h = nonlinearity(h)
        h = self.conv1(h)
        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)
        return x + h


class AttnBlock(nn.Module):
    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.norm = Normalize(in_channels)
        self.q = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.k = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.v = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.proj_out = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        q = self.q(h)
        k = self.k(h)
        v = self.v(h)
        b, c, h_size, w_size = q.shape
        q = q.reshape(b, c, h_size * w_size).permute(0, 2, 1)
        k = k.reshape(b, c, h_size * w_size)
        w_ = torch.bmm(q, k) * (int(c) ** (-0.5))
        w_ = torch.softmax(w_, dim=2)
        v = v.reshape(b, c, h_size * w_size)
        w_ = w_.permute(0, 2, 1)
        h = torch.bmm(v, w_).reshape(b, c, h_size, w_size)
        h = self.proj_out(h)
        return x + h


class AttnBlock3D(nn.Module):
    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.norm = Normalize(in_channels)
        self.q = nn.Conv3d(in_channels, in_channels, kernel_size=1, stride=1)
        self.k = nn.Conv3d(in_channels, in_channels, kernel_size=1, stride=1)
        self.v = nn.Conv3d(in_channels, in_channels, kernel_size=1, stride=1)
        self.proj_out = nn.Conv3d(in_channels, in_channels, kernel_size=1, stride=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        q = self.q(h)
        k = self.k(h)
        v = self.v(h)
        b, c, t, h_size, w_size = q.shape
        q = q.reshape(b * t, c, h_size * w_size).permute(0, 2, 1)
        k = k.reshape(b * t, c, h_size * w_size)
        w_ = torch.bmm(q, k) * (int(c) ** (-0.5))
        w_ = torch.softmax(w_, dim=2)
        v = v.reshape(b * t, c, h_size * w_size)
        w_ = w_.permute(0, 2, 1)
        h = torch.bmm(v, w_).reshape(b, c, t, h_size, w_size)
        h = self.proj_out(h)
        return x + h


class Encoder2D(nn.Module):
    def __init__(
        self,
        *,
        ch: int,
        out_ch: int,
        ch_mult: Sequence[int] = (1, 2, 4, 8),
        num_res_blocks: int = 2,
        attn_resolutions: Sequence[int] = (50,),
        dropout: float = 0.0,
        resamp_with_conv: bool = True,
        in_channels: int,
        resolution: int,
        z_channels: int,
        double_z: bool = True,
    ) -> None:
        super().__init__()
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels

        self.conv_in = nn.Conv2d(in_channels, self.ch, kernel_size=3, stride=1, padding=1)

        curr_res = resolution
        in_ch_mult = (1,) + tuple(ch_mult)
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for _ in range(self.num_res_blocks):
                block.append(
                    ResnetBlock(
                        in_channels=block_in,
                        out_channels=block_out,
                        temb_channels=self.temb_ch,
                        dropout=dropout,
                    )
                )
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample(block_in, resamp_with_conv)
                curr_res = curr_res // 2
            self.down.append(down)

        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
        )
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
        )

        self.norm_out = Normalize(block_in)
        self.conv_out = nn.Conv2d(
            block_in, 2 * z_channels if double_z else z_channels, kernel_size=3, stride=1, padding=1
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, list]:
        shapes = []
        h = self.conv_in(x)
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](h)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
            if i_level != self.num_resolutions - 1:
                shapes.append(h.shape[-2:])
                h = self.down[i_level].downsample(h)
        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h, shapes


class Decoder3D(nn.Module):
    def __init__(
        self,
        *,
        ch: int,
        out_ch: int,
        ch_mult: Sequence[int] = (1, 2, 4, 8),
        num_res_blocks: int = 2,
        attn_resolutions: Sequence[int] = (50,),
        dropout: float = 0.0,
        resamp_with_conv: bool = True,
        in_channels: int,
        resolution: int,
        z_channels: int,
        give_pre_end: bool = False,
    ) -> None:
        super().__init__()
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.give_pre_end = give_pre_end

        in_ch_mult = (1,) + tuple(ch_mult)
        block_in = ch * ch_mult[self.num_resolutions - 1]
        curr_res = resolution // 2 ** (self.num_resolutions - 1)

        self.conv_in = nn.Conv3d(z_channels, block_in, kernel_size=3, stride=1, padding=1)

        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock3D(in_channels=block_in, out_channels=block_in, dropout=dropout)
        self.mid.attn_1 = AttnBlock3D(block_in)
        self.mid.block_2 = ResnetBlock3D(in_channels=block_in, out_channels=block_in, dropout=dropout)

        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            for _ in range(self.num_res_blocks):
                block.append(ResnetBlock3D(in_channels=block_in, out_channels=block_out, dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock3D(block_in))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
                curr_res = curr_res * 2
            self.up.insert(0, up)

        self.norm_out = Normalize(block_in)
        self.conv_out = nn.Conv3d(block_in, out_ch, kernel_size=3, stride=1, padding=1)

    def forward(self, z: torch.Tensor, shapes: list) -> torch.Tensor:
        shapes = shapes.copy()
        h = self.conv_in(z)
        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks):
                h = self.up[i_level].block[i_block](h)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h, shapes.pop())
        if self.give_pre_end:
            return h
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h


class VoxelFeatureVAE(nn.Module):
    def __init__(
        self,
        *,
        feature_dim: int,
        voxel_depth: int,
        resolution: int,
        base_channel: int = 64,
        latent_channels: int = 64,
        expansion: int = 8,
        ch_mult: Sequence[int] = (1, 2, 4, 8),
        num_res_blocks: int = 2,
        attn_resolutions: Sequence[int] = (50,),
        dropout: float = 0.0,
        kl_weight: float = 5e-5,
    ) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.voxel_depth = voxel_depth
        self.expansion = expansion
        self.kl_weight = float(kl_weight)

        self.feature_embed = nn.Linear(feature_dim, expansion)
        self.feature_unembed = nn.Linear(expansion, feature_dim)

        self.encoder = Encoder2D(
            ch=base_channel,
            out_ch=base_channel,
            ch_mult=ch_mult,
            num_res_blocks=num_res_blocks,
            attn_resolutions=attn_resolutions,
            dropout=dropout,
            resamp_with_conv=True,
            in_channels=voxel_depth * expansion,
            resolution=resolution,
            z_channels=latent_channels * 2,
            double_z=False,
        )

        self.decoder = Decoder3D(
            ch=base_channel,
            out_ch=voxel_depth * expansion,
            ch_mult=ch_mult,
            num_res_blocks=num_res_blocks,
            attn_resolutions=attn_resolutions,
            dropout=dropout,
            resamp_with_conv=True,
            in_channels=voxel_depth * expansion,
            resolution=resolution,
            z_channels=latent_channels,
            give_pre_end=False,
        )

    def sample_z(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dim = z.shape[1] // 2
        mu = z[:, :dim]
        logvar = z[:, dim:]
        sigma = torch.exp(logvar / 2)
        eps = torch.randn_like(mu)
        return mu + sigma * eps, mu, logvar

    def forward_encoder(self, x: torch.Tensor) -> Tuple[torch.Tensor, list]:
        # x: [B, X, Y, Z, C]
        x = self.feature_embed(x)
        x = rearrange(x, "b x y z c -> (b) (z c) x y")
        z, shapes = self.encoder(x)
        z = rearrange(z, "b c x y -> b c 1 x y")
        return z, shapes

    def forward_decoder(self, z: torch.Tensor, shapes: list, input_shape: Tuple[int, int, int, int, int]) -> torch.Tensor:
        # z: [B, C, 1, X, Y]
        decoded = self.decoder(z, shapes)
        if decoded.dim() != 5:
            raise RuntimeError(f"Expected 5D decoder output, got shape {tuple(decoded.shape)}")
        b, _, d, x, y = decoded.shape
        if d != 1:
            raise RuntimeError(f"Expected decoder depth dimension 1, got {d}")
        z_dim = input_shape[3]
        decoded = decoded.squeeze(2).permute(0, 2, 3, 1).reshape(b, x, y, z_dim, self.expansion)
        decoded = self.feature_unembed(decoded)
        return decoded

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z, shapes = self.forward_encoder(x)
        z_sampled, z_mu, z_logvar = self.sample_z(z)
        recon = self.forward_decoder(z_sampled, shapes, x.shape)
        return recon, z_mu, z_logvar

    def loss(self, recon: torch.Tensor, target: torch.Tensor, z_mu: torch.Tensor, z_logvar: torch.Tensor, mask: Optional[torch.Tensor] = None, empty_weight: float = 0.1) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if mask is not None:
            weight = mask.float().unsqueeze(-1)
            weight = weight + (1.0 - weight) * empty_weight
        else:
            weight = 1.0
        recon_loss = ((recon - target) ** 2 * weight).mean()
        kl_loss = -0.5 * torch.mean(1 + z_logvar - z_mu.pow(2) - z_logvar.exp())
        total = recon_loss + float(self.kl_weight) * kl_loss
        return total, recon_loss, kl_loss
