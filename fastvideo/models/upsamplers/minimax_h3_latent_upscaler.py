# SPDX-License-Identifier: Apache-2.0
"""MiniMax-H3 3D-convolution latent upscaler (LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler, MIT).

Port of the upstream node's ``LatentResizer3D`` and its ``target dimensions`` size logic, without the ComfyUI shims,
attention blocks (the node forces them off) and temporal chunking (Open-H3-Regenerate-2K runs without it). Module names
match the upstream ``latent_upsampler/*.safetensors`` checkpoint, so it loads without remapping.

The network works on a latent that was normalized with the statistics below, not the VAE-normalized latent the DiT
sees: ``upscale_latent`` therefore re-normalizes its input and de-normalizes the result with the same statistics.
"""
from __future__ import annotations

import re

import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from torch import nn

VAE_DOWNSAMPLE = 16
PIXEL_ALIGNMENT = 32

LATENTS_MEAN = (
    0.858090341091156, -0.9606591463088989, 1.0661640167236328, -0.5090325474739075, -0.2727581858634949,
    -1.3675414323806763, -0.2553254961967468, -0.26907554268836975, -0.5376840829849243, -0.0464097298681736,
    0.6657370328903198, 0.19690127670764923, -0.5460608005523682, -0.4035342037677765, -0.23683024942874908,
    0.25928452610969543, -0.30133944749832153, 0.211341992020607, -1.1206848621368408, 0.3581933379173279,
    -0.04225143790245056, 0.2604829967021942, 0.22864092886447906, 0.7056031823158264)
LATENTS_STD = (
    1.2223774194717407, 1.2767263650894165, 1.6831774711608887, 1.7549455165863037, 1.5636216402053833,
    2.194143533706665, 0.9653137922286987, 1.0569885969161987, 0.841948926448822, 0.7729952931404114,
    1.8955937623977661, 0.946841835975647, 0.7996809482574463, 0.44988900423049927, 0.7197399735450745,
    0.6936293244361877, 2.961095094680786, 2.7694199085235596, 3.0496184825897217, 2.1088054180145264,
    3.276226282119751, 3.1627357006073, 2.2816812992095947, 2.6127843856811523)


def _norm(channels: int) -> nn.GroupNorm:
    return nn.GroupNorm(32, channels)


class ResBlockEmb3D(nn.Module):

    def __init__(self, channels: int, emb_channels: int) -> None:
        super().__init__()
        self.in_layers = nn.Sequential(_norm(channels), nn.SiLU(), nn.Conv3d(channels, channels, 3, padding=1))
        self.emb_layers = nn.Sequential(nn.SiLU(), nn.Linear(emb_channels, 2 * channels))
        self.out_norm = _norm(channels)
        self.out_layers = nn.Sequential(nn.SiLU(), nn.Dropout(p=0.0), nn.Conv3d(channels, channels, 3, padding=1))

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        h = self.in_layers(x)
        scale, shift = self.emb_layers(emb).type(h.dtype)[..., None, None, None].chunk(2, dim=1)
        h = self.out_norm(h) * (1 + scale) + shift
        return x + self.out_layers(h)


class TemporalConv(nn.Module):

    def __init__(self, channels: int, kernel_size: int = 5) -> None:
        super().__init__()
        self.norm = _norm(channels)
        self.dwconv = nn.Conv3d(channels, channels, (kernel_size, 1, 1), padding=(kernel_size // 2, 0, 0),
                                groups=channels)
        self.pwconv = nn.Conv3d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pwconv(self.dwconv(F.silu(self.norm(x))))


def _blocks(count: int, channels: int, emb_dim: int, temporal_every: int, kernel: int) -> nn.ModuleList:
    blocks = nn.ModuleList()
    for index in range(count):
        blocks.append(ResBlockEmb3D(channels, emb_dim))
        if temporal_every > 0 and index % temporal_every == 0:
            blocks.append(TemporalConv(channels, kernel))
    return blocks


class MiniMaxH3LatentUpscaler(nn.Module):

    def __init__(self,
                 in_channels: int = 24,
                 in_blocks: int = 12,
                 out_blocks: int = 12,
                 channels: int = 512,
                 temporal_every: int = 2,
                 temporal_kernel: int = 5) -> None:
        super().__init__()
        emb_dim = 64
        self.conv_in = nn.Conv3d(in_channels, channels, 3, padding=1)
        self.embed = nn.Sequential(nn.Linear(1, emb_dim), nn.SiLU(), nn.Linear(emb_dim, emb_dim))
        self.in_blocks = _blocks(in_blocks, channels, emb_dim, temporal_every, temporal_kernel)
        self.out_blocks = _blocks(out_blocks, channels, emb_dim, temporal_every, temporal_kernel)
        self.norm_out = _norm(channels)
        self.conv_out = nn.Conv3d(channels, in_channels, 3, padding=1)

    def _run(self, blocks: nn.ModuleList, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        for block in blocks:
            x = block(x, emb.expand(x.shape[0], -1)) if isinstance(block, ResBlockEmb3D) else block(x)
        return x

    def forward(self, x: torch.Tensor, scale: float, target_size: tuple[int, int, int]) -> torch.Tensor:
        """``x``: [B, C, T, h, w] normalized latent; returns [B, C, T, *target_size[1:]]."""
        emb = self.embed(torch.tensor([[scale - 1.0]], dtype=x.dtype, device=x.device))
        x = self._run(self.in_blocks, self.conv_in(x), emb)
        x = F.interpolate(x, size=target_size, mode="trilinear", align_corners=False)
        x = self._run(self.out_blocks, x, emb)
        return self.conv_out(F.silu(self.norm_out(x)))

    @classmethod
    def from_checkpoint(cls, path: str) -> MiniMaxH3LatentUpscaler:
        state = load_file(path, device="cpu")
        model = cls(in_channels=state["conv_in.weight"].shape[1],
                    channels=state["conv_in.weight"].shape[0],
                    in_blocks=_count_blocks(state, "in_blocks"),
                    out_blocks=_count_blocks(state, "out_blocks"),
                    temporal_kernel=_temporal_kernel(state))
        model.load_state_dict(state, strict=True)
        return model.eval().requires_grad_(False)


def _count_blocks(state: dict[str, torch.Tensor], prefix: str) -> int:
    pattern = re.compile(rf"{prefix}\.(\d+)\.in_layers\.0\.weight")
    return len({int(m.group(1)) for key in state if (m := pattern.match(key))})


def _temporal_kernel(state: dict[str, torch.Tensor]) -> int:
    return next(tensor.shape[2] for key, tensor in state.items() if key.endswith("dwconv.weight"))


def resolve_upscale_geometry(input_size: tuple[int, int], target_pixels: tuple[int, int]) -> tuple[tuple[int, int], float]:
    """Latent (h, w) after aligning the target pixel size to 32, and the scale the network is conditioned on.

    ``input_size`` is the latent (h, w); ``target_pixels`` the requested (height, width).
    """
    h_in, w_in = input_size
    h_target, w_target = target_pixels
    scale = (w_target / (w_in * VAE_DOWNSAMPLE) + h_target / (h_in * VAE_DOWNSAMPLE)) / 2.0
    aligned = (round(round(h_target / PIXEL_ALIGNMENT) * PIXEL_ALIGNMENT / VAE_DOWNSAMPLE),
               round(round(w_target / PIXEL_ALIGNMENT) * PIXEL_ALIGNMENT / VAE_DOWNSAMPLE))
    return (max(1, aligned[0]), max(1, aligned[1])), scale


@torch.inference_mode()
def upscale_latent(model: MiniMaxH3LatentUpscaler, latent: torch.Tensor, target_pixels: tuple[int, int],
                   dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
    """Upscale a [C, T, h, w] latent to ``target_pixels`` (height, width); FP32 [C, T, H, W] out.

    The input is returned unchanged when it already has the target latent size.
    """
    channels, frames, h_in, w_in = latent.shape
    (h_out, w_out), scale = resolve_upscale_geometry((h_in, w_in), target_pixels)
    if (h_out, w_out) == (h_in, w_in):
        return latent.float()
    if scale < 1.0:
        raise ValueError("The MiniMax-H3 latent upscaler only upscales (effective scale >= 1.0).")
    mean = torch.tensor(LATENTS_MEAN, dtype=dtype, device=latent.device).view(1, -1, 1, 1, 1)
    std = torch.tensor(LATENTS_STD, dtype=dtype, device=latent.device).view(1, -1, 1, 1, 1)
    source = (latent[None].to(dtype) - mean) / std
    result = model(source, scale, (frames, h_out, w_out))
    return (result * std + mean).float()[0]
