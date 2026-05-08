"""VBench Motion Smoothness — AMT-S frame interpolation quality.

Takes every-other frame, uses AMT-S to interpolate the missing middle
frames, then compares interpolated vs actual frames.
Score = (255 - mean_pixel_diff) / 255.  Higher = smoother motion.
"""

from __future__ import annotations

from typing import Any

import os

import cv2
import numpy as np
import torch

from fastvideo.eval.metrics.base import BaseMetric
from fastvideo.eval.registry import register
from fastvideo.eval.types import MetricResult


@register("vbench.motion_smoothness")
class MotionSmoothnessMetric(BaseMetric):

    name = "vbench.motion_smoothness"
    requires_reference = False
    higher_is_better = True
    needs_gpu = True
    dependencies = ["omegaconf"]

    def __init__(self) -> None:
        super().__init__()
        self._model: Any = None
        self._embt: Any = None
        self._chunk_size = 8

    def to(self, device):
        super().to(device)
        if self._model is not None:
            self._model = self._model.to(self.device)
        if self._embt is not None:
            self._embt = self._embt.to(self.device)
        return self

    def setup(self) -> None:
        if self._model is not None:
            return
        from omegaconf import OmegaConf
        import vbench.third_party.amt as _amt_pkg
        from vbench.third_party.amt.utils.build_utils import build_from_cfg
        from fastvideo.eval.models import ensure_checkpoint

        amt_dir = os.path.dirname(_amt_pkg.__file__)
        cfg_path = os.path.join(amt_dir, "cfgs", "AMT-S.yaml")

        ckpt_path = ensure_checkpoint(
            "amt-s.pth",
            source="https://huggingface.co/lalala125/AMT/resolve/main/amt-s.pth",
        )

        network_cfg = OmegaConf.load(cfg_path).network
        self._model = build_from_cfg(network_cfg)
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        self._model.load_state_dict(ckpt["state_dict"])
        self._model.to(self.device)
        self._model.eval()

        self._embt = torch.tensor(1 / 2).float().view(1, 1, 1, 1).to(self.device)

    def _get_scale(self, h: int, w: int) -> float:
        """Pick a downscale factor that keeps AMT's correlation volume
        within free GPU memory.

        Re-queries free memory on every call (rather than caching at setup
        time) so the scale adapts to whatever's actually available — other
        metric replicas already loaded, residual generator allocations,
        another process sharing the GPU, etc. The upstream version cached
        ``total_memory`` at setup, which on a shared/loaded GPU lets AMT
        attempt a 30+ GB correlation volume reshape and OOM.
        """
        if self.device.type != "cuda":
            return 1.0
        # Free memory that won't be claimed by other allocations during this
        # forward pass. min(free, total) is conservative against transient
        # spikes; mem_get_info returns (free, total) in bytes.
        free_bytes, _ = torch.cuda.mem_get_info(self.device)
        anchor_resolution = 1024 * 512
        anchor_memory = 1500 * 1024**2
        anchor_memory_bias = 2500 * 1024**2
        if free_bytes <= anchor_memory_bias:
            # Less than the model + scratch overhead is free; force the
            # most aggressive downscale we support.
            return 1 / 16
        scale = anchor_resolution / (h * w) * np.sqrt((free_bytes - anchor_memory_bias) / anchor_memory)
        if scale >= 1.0:
            return 1.0
        scale = 1 / np.floor(1 / np.sqrt(scale) * 16) * 16
        return float(scale)

    _MIN_AMT_SCALE = 1 / 16

    def _safe_amt_forward(self, in0: Any, in1: Any, embt: Any, scale: Any) -> Any:
        """Run one chunk through AMT with OOM-retry on two axes.

        Recovery strategy on ``CUDA out of memory``:

        1. **Halve the batch** until batch=1. AMT's per-pair memory
           dominates; splitting helps until each pair is on its own.
        2. **Halve the scale_factor** passed to AMT (which controls its
           internal feature-map resolution and therefore the correlation
           volume size). Bottoms out at ``_MIN_AMT_SCALE`` — beyond that
           the feature maps are too coarse to produce meaningful
           interpolation and we re-raise.

        Self-tunes under memory pressure: the upstream autoscale formula
        in :meth:`_get_scale` mis-extrapolates at large resolutions
        (treats memory as linear in pixel count, but AMT's correlation
        volume grows quadratically). This retry path makes the metric
        robust to that without rewriting the formula.
        """
        try:
            return self._model(in0, in1, embt, scale_factor=scale, eval=True)["imgt_pred"]
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            bs = in0.shape[0]
            if bs > 1:
                half = bs // 2
                a = self._safe_amt_forward(in0[:half], in1[:half], embt[:half], scale)
                b = self._safe_amt_forward(in0[half:], in1[half:], embt[half:], scale)
                return torch.cat([a, b], dim=0)
            if scale > self._MIN_AMT_SCALE:
                return self._safe_amt_forward(in0, in1, embt, scale / 2)
            raise

    @torch.no_grad()
    def compute(self, sample: dict) -> MetricResult:
        from vbench.third_party.amt.utils.utils import (
            img2tensor,
            tensor2img,
            check_dim_and_resize,
            InputPadder,
        )

        video = sample["video"]  # (T, C, H, W) [0, 1]
        chunk = self._chunk_size or 8

        frames_np = (video * 255).to(torch.uint8).cpu().numpy()
        frames_np = [f.transpose(1, 2, 0) for f in frames_np]  # list of (H,W,C)

        even_indices = list(range(0, len(frames_np), 2))
        if len(even_indices) <= 1:
            return MetricResult(name=self.name, score=1.0, details={})

        even_frames = [frames_np[i] for i in even_indices]
        inputs = [img2tensor(f).to(self.device) for f in even_frames]
        inputs = check_dim_and_resize(inputs)

        h, w = inputs[0].shape[-2:]
        scale = self._get_scale(h, w)
        padding = int(16 / scale)
        padder = InputPadder(inputs[0].shape, padding)
        inputs = padder.pad(*inputs)

        n_pairs = len(inputs) - 1
        all_in0 = [inputs[i] for i in range(n_pairs)]
        all_in1 = [inputs[i + 1] for i in range(n_pairs)]
        all_gt = [
            frames_np[even_indices[i] + 1] if even_indices[i] + 1 < len(frames_np) else frames_np[-1]
            for i in range(n_pairs)
        ]

        all_preds = []
        for start in range(0, len(all_in0), chunk):
            end = min(start + chunk, len(all_in0))
            in0_batch = torch.cat(all_in0[start:end], dim=0).to(self.device)
            in1_batch = torch.cat(all_in1[start:end], dim=0).to(self.device)
            embt = self._embt.expand(in0_batch.shape[0], -1, -1, -1)
            pred = self._safe_amt_forward(in0_batch, in1_batch, embt, scale)
            all_preds.append(pred.cpu())
        all_preds = torch.cat(all_preds, dim=0)

        diffs: list[float] = []
        for i in range(n_pairs):
            pred = all_preds[i:i + 1]
            pred_unpadded = padder.unpad(pred)[0]
            pred_np = tensor2img(pred_unpadded)
            gt_np = all_gt[i]
            # gt comes from frames_np; pred goes through check_dim_and_resize +
            # AMT pad/unpad, which can reshape. Match shapes before absdiff.
            if gt_np.shape[:2] != pred_np.shape[:2]:
                gt_np = cv2.resize(gt_np, (pred_np.shape[1], pred_np.shape[0]), interpolation=cv2.INTER_AREA)
            diffs.append(float(np.mean(cv2.absdiff(gt_np, pred_np))))

        vfi_score = float(np.mean(diffs)) if diffs else 0.0
        return MetricResult(
            name=self.name,
            score=(255.0 - vfi_score) / 255.0,
            details={"vfi_score": vfi_score},
        )
