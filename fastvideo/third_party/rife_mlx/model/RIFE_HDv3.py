"""Model wrapper — MLX port of train_log/RIFE_HDv3.py.

Wraps IFNet and exposes inference(img0, img1, timestep, scale) -> middle frame,
matching upstream: scale_list = [16,8,4,2,1] / scale; H,W padded to a multiple of
pad_to (64; the 5-block downsample factor) and cropped back after.
"""

from __future__ import annotations

import mlx.core as mx

from .IFNet_HDv3 import IFNet


class Model:
    def __init__(self, scale_list=(16, 8, 4, 2, 1), pad_to: int = 64) -> None:
        self.flownet = IFNet()
        self.scale_list = tuple(scale_list)
        self.pad_to = pad_to

    def inference(self, img0: mx.array, img1: mx.array,
                  timestep: float = 0.5, scale: float = 1.0) -> mx.array:
        """img0,img1: [N,H,W,3] in [0,1]. Returns the frame at `timestep`."""
        N, H, W, _ = img0.shape
        # pad must scale with 1/scale so the coarsest-scale interpolation
        # round-trips exactly (else the recovered flow size != padded size).
        pad = max(self.pad_to, int(round(self.pad_to / scale)))
        ph = ((H - 1) // pad + 1) * pad
        pw = ((W - 1) // pad + 1) * pad
        pad = [(0, 0), (0, ph - H), (0, pw - W), (0, 0)]
        i0 = mx.pad(img0, pad)
        i1 = mx.pad(img1, pad)

        scale_list = tuple(s / scale for s in self.scale_list)
        x = mx.concatenate([i0, i1], axis=-1)
        _flow, _mask, merged = self.flownet(x, timestep, scale_list)
        return merged[:, :H, :W, :]
