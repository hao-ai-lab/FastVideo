"""IFNet (RIFE 4.25) — MLX port, isomorphic to the bundled train_log/IFNet_HDv3.py.

NHWC throughout. Translated line-by-line from the pinned v4.25 source; only
PyTorch ops -> MLX ops and NCHW -> NHWC change. Parity-locked in P3 (Gate B).

Arch (pinned): 5 IFBlocks c=[192,128,96,64,32], in_planes=[15,28,28,28,28];
conv = Conv2d + LeakyReLU(0.2); ResConv = conv*beta + x -> LeakyReLU; lastconv =
ConvTranspose2d(c,52,4,2,1) + PixelShuffle(2) -> 13ch (flow4+mask1+feat8); Head
encoder -> 4ch; F.interpolate align_corners=False; warp align_corners=True.

MLX param keys (post-conversion) match torch except: conv0.{j}.0.* -> conv0.{j}.conv.*
and lastconv.0.* -> lastconv.* (see utils/convert.py).
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from ..ops.interpolate import interpolate_bilinear
from ..ops.spatial import pixel_shuffle_nhwc
from .warplayer import warp

_SLOPE = 0.2


def _lrelu(x):
    return nn.leaky_relu(x, _SLOPE)


class Conv(nn.Module):
    """conv = Conv2d + LeakyReLU(0.2). key: <p>.conv.weight"""

    def __init__(self, i, o, k=3, s=1, p=1):
        super().__init__()
        self.conv = nn.Conv2d(i, o, kernel_size=k, stride=s, padding=p)

    def __call__(self, x):
        return _lrelu(self.conv(x))


class ResConv(nn.Module):
    """conv*beta + x -> LeakyReLU. keys: .conv.weight/bias, .beta"""

    def __init__(self, c):
        super().__init__()
        self.conv = nn.Conv2d(c, c, kernel_size=3, stride=1, padding=1)
        self.beta = mx.ones((1, 1, 1, c))  # NHWC per-channel

    def __call__(self, x):
        return _lrelu(self.conv(x) * self.beta + x)


class Head(nn.Module):
    """Feature encoder. keys: cnn0,cnn1,cnn2 (Conv2d), cnn3 (ConvTranspose2d)."""

    def __init__(self):
        super().__init__()
        self.cnn0 = nn.Conv2d(3, 16, 3, stride=2, padding=1)
        self.cnn1 = nn.Conv2d(16, 16, 3, stride=1, padding=1)
        self.cnn2 = nn.Conv2d(16, 16, 3, stride=1, padding=1)
        self.cnn3 = nn.ConvTranspose2d(16, 4, 4, stride=2, padding=1)

    def __call__(self, x):
        x = _lrelu(self.cnn0(x))
        x = _lrelu(self.cnn1(x))
        x = _lrelu(self.cnn2(x))
        return self.cnn3(x)


class IFBlock(nn.Module):
    def __init__(self, in_planes, c=64):
        super().__init__()
        self.conv0 = [Conv(in_planes, c // 2, 3, 2, 1), Conv(c // 2, c, 3, 2, 1)]
        self.convblock = [ResConv(c) for _ in range(8)]
        self.lastconv = nn.ConvTranspose2d(c, 4 * 13, 4, stride=2, padding=1)

    def __call__(self, x, flow=None, scale=1):
        x = interpolate_bilinear(x, scale_factor=1.0 / scale, align_corners=False)
        if flow is not None:
            flow = interpolate_bilinear(flow, scale_factor=1.0 / scale,
                                        align_corners=False) * (1.0 / scale)
            x = mx.concatenate([x, flow], axis=-1)
        feat = self.conv0[0](x)
        feat = self.conv0[1](feat)
        for blk in self.convblock:
            feat = blk(feat)
        tmp = pixel_shuffle_nhwc(self.lastconv(feat), 2)
        tmp = interpolate_bilinear(tmp, scale_factor=scale, align_corners=False)
        flow = tmp[..., :4] * scale
        mask = tmp[..., 4:5]
        feat = tmp[..., 5:]
        return flow, mask, feat


class IFNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.block0 = IFBlock(7 + 8, c=192)
        self.block1 = IFBlock(8 + 4 + 8 + 8, c=128)
        self.block2 = IFBlock(8 + 4 + 8 + 8, c=96)
        self.block3 = IFBlock(8 + 4 + 8 + 8, c=64)
        self.block4 = IFBlock(8 + 4 + 8 + 8, c=32)
        self.encode = Head()

    def __call__(self, x, timestep=0.5, scale_list=(16, 8, 4, 2, 1)):
        """x: [N,H,W,6] (img0|img1 on channel axis). Returns (flow_list, mask, merged)."""
        channel = x.shape[-1] // 2
        img0 = x[..., :channel]
        img1 = x[..., channel:]
        N, H, W, _ = img0.shape
        timestep = mx.full((N, H, W, 1), float(timestep))

        f0 = self.encode(img0[..., :3])
        f1 = self.encode(img1[..., :3])
        flow_list, merged, mask_list = [], [], []
        warped_img0, warped_img1 = img0, img1
        flow = None
        mask = None
        blocks = [self.block0, self.block1, self.block2, self.block3, self.block4]
        for i in range(5):
            if flow is None:
                flow, mask, feat = blocks[i](
                    mx.concatenate([img0[..., :3], img1[..., :3], f0, f1, timestep], axis=-1),
                    None, scale=scale_list[i])
            else:
                wf0 = warp(f0, flow[..., :2])
                wf1 = warp(f1, flow[..., 2:4])
                fd, m0, feat = blocks[i](
                    mx.concatenate([warped_img0[..., :3], warped_img1[..., :3],
                                    wf0, wf1, timestep, mask, feat], axis=-1),
                    flow, scale=scale_list[i])
                mask = m0
                flow = flow + fd
            mask_list.append(mask)
            flow_list.append(flow)
            warped_img0 = warp(img0, flow[..., :2])
            warped_img1 = warp(img1, flow[..., 2:4])
            merged.append((warped_img0, warped_img1))
        mask = mx.sigmoid(mask)
        merged_final = warped_img0 * mask + warped_img1 * (1 - mask)
        return flow_list, mask, merged_final
