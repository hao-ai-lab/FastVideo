from __future__ import annotations

import torch
import torch.nn.functional as F

from fastvideo.eval.metrics.base import BaseMetric
from fastvideo.eval.registry import register
from fastvideo.eval.types import MetricResult


def _ssim_per_frame(
    x: torch.Tensor,
    y: torch.Tensor,
    window_size: int = 11,
    C1: float = 0.01**2,
    C2: float = 0.03**2,
) -> torch.Tensor:
    """Compute SSIM for each frame.  Returns ``(N,)`` tensor where N = number of frames."""
    channels = x.shape[1]
    kernel = _gaussian_kernel(window_size, 1.5, channels, x.device, x.dtype)

    mu_x = F.conv2d(x, kernel, groups=channels, padding=window_size // 2)
    mu_y = F.conv2d(y, kernel, groups=channels, padding=window_size // 2)

    mu_x2 = mu_x * mu_x
    mu_y2 = mu_y * mu_y
    mu_xy = mu_x * mu_y

    sigma_x2 = F.conv2d(x * x, kernel, groups=channels, padding=window_size // 2) - mu_x2
    sigma_y2 = F.conv2d(y * y, kernel, groups=channels, padding=window_size // 2) - mu_y2
    sigma_xy = F.conv2d(x * y, kernel, groups=channels, padding=window_size // 2) - mu_xy

    num = (2 * mu_xy + C1) * (2 * sigma_xy + C2)
    den = (mu_x2 + mu_y2 + C1) * (sigma_x2 + sigma_y2 + C2)
    ssim_map = num / den

    return ssim_map.mean(dim=(1, 2, 3))


def _gaussian_kernel(size: int, sigma: float, channels: int, device, dtype):
    coords = torch.arange(size, device=device, dtype=dtype) - size // 2
    g = torch.exp(-coords**2 / (2 * sigma**2))
    g = g / g.sum()
    kernel_2d = g.unsqueeze(1) * g.unsqueeze(0)
    kernel = kernel_2d.unsqueeze(0).unsqueeze(0).repeat(channels, 1, 1, 1)
    return kernel


@register("common.ssim")
class SSIMMetric(BaseMetric):
    name = "common.ssim"
    requires_reference = True
    higher_is_better = True
    needs_gpu = False
    batch_unit = "frame"

    def __init__(self, window_size: int = 11) -> None:
        super().__init__()
        self.window_size = window_size

    def trial_forward(self, batch_size, *, height, width, num_frames):
        dummy = torch.randn(batch_size, 3, height, width, device=self.device)
        _ssim_per_frame(dummy, dummy, self.window_size)

    def compute(self, sample: dict) -> list[MetricResult]:
        gen = sample["video"].float()       # (B, T, C, H, W)
        ref = sample["reference"].float()   # (B, T, C, H, W)
        B, T = gen.shape[:2]
        n = min(gen.shape[1], ref.shape[1])
        gen, ref = gen[:, :n], ref[:, :n]

        gen_flat = gen.reshape(B * n, *gen.shape[2:])
        ref_flat = ref.reshape(B * n, *ref.shape[2:])

        # Chunked forward
        chunk = self._chunk_size or len(gen_flat)
        parts = []
        for i in range(0, len(gen_flat), chunk):
            parts.append(_ssim_per_frame(gen_flat[i:i+chunk], ref_flat[i:i+chunk], self.window_size))
        per_frame = torch.cat(parts).reshape(B, n)

        return [
            MetricResult(
                name=self.name,
                score=per_frame[b].mean().item(),
                details={"per_frame": per_frame[b].tolist()},
            )
            for b in range(B)
        ]
