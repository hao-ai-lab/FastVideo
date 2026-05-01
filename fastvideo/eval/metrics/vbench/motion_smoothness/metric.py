"""VBench Motion Smoothness — AMT-S frame interpolation quality.

Takes every-other frame, uses AMT-S to interpolate the missing middle
frames, then compares interpolated vs actual frames.
Score = (255 - mean_pixel_diff) / 255.  Higher = smoother motion.
"""

from __future__ import annotations

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
    batch_unit = "frame_pair"
    dependencies = ["omegaconf"]

    def __init__(self) -> None:
        super().__init__()
        self._model = None
        self._embt = None
        self._vram_avail = None
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

        if self.device.type == "cuda":
            self._vram_avail = torch.cuda.get_device_properties(self.device).total_memory
        else:
            self._vram_avail = None

    def _get_scale(self, h: int, w: int) -> float:
        if self._vram_avail is None:
            return 1.0
        anchor_resolution = 1024 * 512
        anchor_memory = 1500 * 1024**2
        anchor_memory_bias = 2500 * 1024**2
        scale = anchor_resolution / (h * w) * np.sqrt(
            (self._vram_avail - anchor_memory_bias) / anchor_memory
        )
        if scale >= 1.0:
            return 1.0
        scale = 1 / np.floor(1 / np.sqrt(scale) * 16) * 16
        return float(scale)

    def trial_forward(self, batch_size, *, height, width, num_frames):
        h, w = min(height, 512), min(width, 512)
        img0 = torch.randn(batch_size, 3, h, w, device=self.device)
        img1 = torch.randn(batch_size, 3, h, w, device=self.device)
        embt = self._embt.expand(batch_size, -1, -1, -1)
        with torch.no_grad():
            self._model(img0, img1, embt, scale_factor=1.0, eval=True)

    @torch.no_grad()
    def compute(self, sample: dict) -> list[MetricResult]:
        from vbench.third_party.amt.utils.utils import (
            img2tensor, tensor2img, check_dim_and_resize, InputPadder,
        )

        video = sample["video"]  # (B, T, C, H, W) float [0, 1]
        B, T = video.shape[:2]
        chunk = self._chunk_size or 8

        # Prepare all frame pairs across all videos
        # Even frames [0, 2, 4, ...] → pairs (0,2), (2,4), (4,6), ...
        all_in0, all_in1, all_gt = [], [], []
        pairs_per_video = []

        for b in range(B):
            frames_np = (video[b] * 255).to(torch.uint8).cpu().numpy()
            frames_np = [f.transpose(1, 2, 0) for f in frames_np]  # list of (H,W,C)

            even_indices = list(range(0, len(frames_np), 2))
            if len(even_indices) <= 1:
                pairs_per_video.append(0)
                continue

            even_frames = [frames_np[i] for i in even_indices]
            inputs = [img2tensor(f).to(self.device) for f in even_frames]
            inputs = check_dim_and_resize(inputs)

            h, w = inputs[0].shape[-2:]
            scale = self._get_scale(h, w)
            padding = int(16 / scale)
            padder = InputPadder(inputs[0].shape, padding)
            inputs = padder.pad(*inputs)

            n_pairs = len(inputs) - 1
            pairs_per_video.append(n_pairs)

            for i in range(n_pairs):
                all_in0.append(inputs[i])
                all_in1.append(inputs[i + 1])
                # Ground truth odd frame (the one between even[i] and even[i+1])
                odd_idx = even_indices[i] + 1
                if odd_idx < len(frames_np):
                    all_gt.append(frames_np[odd_idx])
                else:
                    all_gt.append(frames_np[-1])

        if not all_in0:
            return [MetricResult(name=self.name, score=1.0, details={}) for _ in range(B)]

        # Batched AMT forward — all pairs across all videos
        h, w = all_in0[0].shape[-2:]
        scale = self._get_scale(h, w)
        all_preds = []
        for start in range(0, len(all_in0), chunk):
            end = min(start + chunk, len(all_in0))
            in0_batch = torch.cat(all_in0[start:end], dim=0).to(self.device)
            in1_batch = torch.cat(all_in1[start:end], dim=0).to(self.device)
            embt = self._embt.expand(in0_batch.shape[0], -1, -1, -1)
            pred = self._model(in0_batch, in1_batch, embt,
                              scale_factor=scale, eval=True)["imgt_pred"]
            all_preds.append(pred.cpu())

        all_preds = torch.cat(all_preds, dim=0)

        # Unpad and compute per-video scores
        padding_val = int(16 / scale)
        padder = InputPadder(all_in0[0].shape, padding_val)

        results = []
        offset = 0
        for b in range(B):
            n_pairs = pairs_per_video[b]
            if n_pairs == 0:
                results.append(MetricResult(name=self.name, score=1.0, details={}))
                continue

            diffs = []
            for i in range(n_pairs):
                pred = all_preds[offset + i:offset + i + 1]
                pred_unpadded = padder.unpad(pred)[0]
                pred_np = tensor2img(pred_unpadded)
                gt_np = all_gt[offset + i]
                # gt is from the original frames_np; pred comes through
                # check_dim_and_resize + AMT pad/unpad, which can reshape.
                # Match shapes before absdiff.
                if gt_np.shape[:2] != pred_np.shape[:2]:
                    gt_np = cv2.resize(gt_np, (pred_np.shape[1], pred_np.shape[0]),
                                       interpolation=cv2.INTER_AREA)
                diff = np.mean(cv2.absdiff(gt_np, pred_np))
                diffs.append(diff)
            offset += n_pairs

            vfi_score = np.mean(diffs) if diffs else 0.0
            score = (255.0 - vfi_score) / 255.0

            results.append(MetricResult(
                name=self.name,
                score=float(score),
                details={"vfi_score": float(vfi_score)},
            ))

        return results
