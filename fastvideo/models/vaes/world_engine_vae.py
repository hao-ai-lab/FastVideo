# SPDX-License-Identifier: Apache-2.0
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.utils.parametrize import remove_parametrizations
from torch.nn.utils.parametrizations import weight_norm

from fastvideo.configs.models.vaes.world_engine_vae import WorldEngineVAEConfig


def WeightNormConv2d(*args, **kwargs) -> nn.Module:
    return weight_norm(nn.Conv2d(*args, **kwargs))


def bake_weight_norm(model: nn.Module) -> nn.Module:
    for module in model.modules():
        if "weight" in getattr(module, "parametrizations", {}):
            remove_parametrizations(module, "weight", leave_parametrized=True)
    return model


class ResBlock(nn.Module):

    def __init__(self, ch: int):
        super().__init__()
        hidden = 2 * ch
        n_grps = max(1, hidden // 16)
        self.conv1 = WeightNormConv2d(ch, hidden, 1, 1, 0)
        self.conv2 = WeightNormConv2d(hidden, hidden, 3, 1, 1, groups=n_grps)
        self.conv3 = WeightNormConv2d(hidden, ch, 1, 1, 0, bias=False)
        self.act1 = nn.LeakyReLU(inplace=False)
        self.act2 = nn.LeakyReLU(inplace=False)

    def forward(self, x: Tensor) -> Tensor:
        h = self.conv1(x)
        h = self.act1(h)
        h = self.conv2(h)
        h = self.act2(h)
        h = self.conv3(h)
        return x + h


class LandscapeToSquare(nn.Module):

    def __init__(self, ch_in: int, ch_out: int):
        super().__init__()
        self.proj = WeightNormConv2d(ch_in, ch_out, 3, 1, 1)

    def forward(self, x: Tensor) -> Tensor:
        x = F.interpolate(x, (512, 512), mode="bicubic")
        return self.proj(x)


class Downsample(nn.Module):

    def __init__(self, ch_in: int, ch_out: int):
        super().__init__()
        self.proj = WeightNormConv2d(ch_in, ch_out, 1, 1, 0, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        x = F.interpolate(x, scale_factor=0.5, mode="bicubic")
        return self.proj(x)


class DownBlock(nn.Module):

    def __init__(self, ch_in: int, ch_out: int, num_res: int = 1):
        super().__init__()
        self.down = Downsample(ch_in, ch_out)
        self.blocks = nn.ModuleList([ResBlock(ch_in) for _ in range(num_res)])

    def forward(self, x: Tensor) -> Tensor:
        for block in self.blocks:
            x = block(x)
        return self.down(x)


class SpaceToChannel(nn.Module):

    def __init__(self, ch_in: int, ch_out: int):
        super().__init__()
        self.proj = WeightNormConv2d(ch_in, ch_out // 4, 3, 1, 1)

    def forward(self, x: Tensor) -> Tensor:
        x = self.proj(x)
        return F.pixel_unshuffle(x, 2).contiguous()


class ChannelAverage(nn.Module):

    def __init__(self, ch_in: int, ch_out: int):
        super().__init__()
        self.proj = WeightNormConv2d(ch_in, ch_out, 3, 1, 1)
        self.grps = ch_in // ch_out
        self.scale = self.grps**0.5

    def forward(self, x: Tensor) -> Tensor:
        res = x
        x = self.proj(x.contiguous())
        res = res.view(res.shape[0], self.grps, res.shape[1] // self.grps,
                       res.shape[2], res.shape[3]).contiguous()
        res = res.mean(dim=1) * self.scale
        return res + x


class SquareToLandscape(nn.Module):

    def __init__(self, ch_in: int, ch_out: int):
        super().__init__()
        self.proj = WeightNormConv2d(ch_in, ch_out, 3, 1, 1)

    def forward(self, x: Tensor) -> Tensor:
        x = self.proj(x)
        return F.interpolate(x, (360, 640), mode="bicubic")


class Upsample(nn.Module):

    def __init__(self, ch_in: int, ch_out: int):
        super().__init__()
        self.proj = nn.Identity() if ch_in == ch_out else WeightNormConv2d(
            ch_in, ch_out, 1, 1, 0, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        x = self.proj(x)
        return F.interpolate(x, scale_factor=2.0, mode="bicubic")


class UpBlock(nn.Module):

    def __init__(self, ch_in: int, ch_out: int, num_res: int = 1):
        super().__init__()
        self.up = Upsample(ch_in, ch_out)
        self.blocks = nn.ModuleList([ResBlock(ch_out) for _ in range(num_res)])

    def forward(self, x: Tensor) -> Tensor:
        x = self.up(x)
        for block in self.blocks:
            x = block(x)
        return x


class ChannelToSpace(nn.Module):

    def __init__(self, ch_in: int, ch_out: int):
        super().__init__()
        self.proj = WeightNormConv2d(ch_in, ch_out * 4, 3, 1, 1)

    def forward(self, x: Tensor) -> Tensor:
        x = self.proj(x)
        return F.pixel_shuffle(x, 2).contiguous()


class ChannelDuplication(nn.Module):

    def __init__(self, ch_in: int, ch_out: int):
        super().__init__()
        self.proj = WeightNormConv2d(ch_in, ch_out, 3, 1, 1)
        self.reps = ch_out // ch_in
        self.scale = self.reps**-0.5

    def forward(self, x: Tensor) -> Tensor:
        res = x
        x = self.proj(x.contiguous())
        b, c, h, w = res.shape
        res = res.unsqueeze(2).expand(b, c, self.reps, h, w)
        res = res.reshape(b, c * self.reps, h, w).contiguous() * self.scale
        return res + x


class Encoder(nn.Module):

    def __init__(self, channels: int, latent_channels: int, ch_0: int,
                 ch_max: int, blocks_per_stage: tuple[int, ...],
                 skip_logvar: bool):
        super().__init__()
        self.conv_in = LandscapeToSquare(channels, ch_0)
        blocks, residuals = [], []
        ch = ch_0
        for block_count in blocks_per_stage:
            next_ch = min(ch * 2, ch_max)
            blocks.append(DownBlock(ch, next_ch, block_count))
            residuals.append(SpaceToChannel(ch, next_ch))
            ch = next_ch
        self.blocks = nn.ModuleList(blocks)
        self.residuals = nn.ModuleList(residuals)
        self.conv_out = ChannelAverage(ch, latent_channels)
        self.skip_logvar = skip_logvar
        if not skip_logvar:
            self.conv_out_logvar = WeightNormConv2d(ch, 1, 3, 1, 1)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv_in(x)
        for block, residual in zip(self.blocks, self.residuals, strict=True):
            x = block(x) + residual(x)
        return self.conv_out(x)


class Decoder(nn.Module):

    def __init__(self, channels: int, latent_channels: int, ch_0: int,
                 ch_max: int, blocks_per_stage: tuple[int, ...]):
        super().__init__()
        self.conv_in = ChannelDuplication(latent_channels, ch_max)
        blocks, residuals = [], []
        ch = ch_0
        for block_count in reversed(blocks_per_stage):
            next_ch = min(ch * 2, ch_max)
            blocks.append(UpBlock(next_ch, ch, block_count))
            residuals.append(ChannelToSpace(next_ch, ch))
            ch = next_ch
        self.blocks = nn.ModuleList(reversed(blocks))
        self.residuals = nn.ModuleList(reversed(residuals))
        self.act_out = nn.SiLU()
        self.conv_out = SquareToLandscape(ch_0, channels)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv_in(x)
        for block, residual in zip(self.blocks, self.residuals, strict=True):
            x = block(x) + residual(x)
        x = self.act_out(x)
        return self.conv_out(x)


class WorldEngineVAE(nn.Module):

    def __init__(self, config: WorldEngineVAEConfig):
        super().__init__()
        self.config = config
        arch = config.arch_config
        self.encoder = Encoder(
            arch.channels,
            arch.latent_channels,
            arch.encoder_ch_0,
            arch.encoder_ch_max,
            tuple(arch.encoder_blocks_per_stage),
            arch.skip_logvar,
        )
        self.decoder = Decoder(
            arch.channels,
            arch.latent_channels,
            arch.decoder_ch_0,
            arch.decoder_ch_max,
            tuple(arch.decoder_blocks_per_stage),
        )

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

    def encode(self, img: Tensor) -> Tensor:
        assert img.dim() == 3, "Expected [H, W, C] image tensor"
        img = img.unsqueeze(0).to(device=self.device, dtype=self.dtype)
        rgb = img.permute(0, 3, 1, 2).contiguous().div(255).mul(2).sub(1)
        return self.encoder(rgb)

    def decode(self, latent: Tensor) -> Tensor:
        decoded = self.decoder(latent)
        decoded = (decoded / 2 + 0.5).clamp(0, 1)
        decoded = (decoded * 255).round().to(torch.uint8)
        return decoded.squeeze(0).permute(1, 2, 0)[..., :3]

    def forward(self, x: Tensor, encode: bool = True) -> Tensor:
        return self.encode(x) if encode else self.decode(x)

    def bake_weight_norm(self) -> "WorldEngineVAE":
        bake_weight_norm(self)
        return self
