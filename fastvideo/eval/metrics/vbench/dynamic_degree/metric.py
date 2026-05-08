"""VBench Dynamic Degree — RAFT optical flow motion detection.

For each consecutive frame pair, computes optical flow via RAFT and takes
the mean of the top 5% flow magnitudes.  If enough pairs exceed an
adaptive threshold, the video is classified as dynamic (1.0) vs static (0.0).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from easydict import EasyDict

from fastvideo.eval.metrics.base import BaseMetric
from fastvideo.eval.registry import register
from fastvideo.eval.types import MetricResult


@register("vbench.dynamic_degree")
class DynamicDegreeMetric(BaseMetric):

    name = "vbench.dynamic_degree"
    requires_reference = False
    higher_is_better = True
    needs_gpu = True
    dependencies = ["easydict"]

    def __init__(self) -> None:
        super().__init__()
        self._model: Any = None
        self._chunk_size = 16

    def to(self, device):
        super().to(device)
        if self._model is not None:
            self._model = self._model.to(self.device)
        return self

    def setup(self) -> None:
        if self._model is not None:
            return
        from vbench.third_party.RAFT.core.raft import RAFT

        args = EasyDict(small=False, mixed_precision=False, alternate_corr=False, dropout=0.0)
        model = torch.nn.DataParallel(RAFT(args))

        from fastvideo.eval.models import ensure_checkpoint
        ckpt_path = ensure_checkpoint(
            "raft-things.pth",
            source="sbalani/raft-things",
            filename="raft-things.pth",
        )
        model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
        model = model.module
        model.to(self.device)
        model.eval()
        self._model = model

    def _get_score(self, flow: torch.Tensor) -> float:
        """Top-5% mean flow magnitude (matching VBench dynamic_degree.get_score)."""
        flo = flow.permute(1, 2, 0).cpu().numpy()
        rad = np.sqrt(flo[..., 0]**2 + flo[..., 1]**2)
        h, w = rad.shape
        cut = max(1, int(h * w * 0.05))
        rad_flat = rad.flatten()
        return float(np.mean(np.sort(rad_flat)[-cut:]))

    @torch.no_grad()
    def compute(self, sample: dict) -> MetricResult:
        from vbench.third_party.RAFT.core.utils_core.utils import InputPadder

        video = sample["video"]  # (T, C, H, W) [0, 1]
        T, _, H, W = video.shape

        # fps controls the temporal sampling stride for optical flow.
        # vbench computes flow at 8fps (interval = round(fps/8)). The metric
        # cannot auto-derive fps from a tensor, so a missing fps would silently
        # use a wrong stride and produce a wrong score. Skip explicitly.
        if "fps" not in sample:
            return self._skip(sample, "missing 'fps' (required to set the "
                              "8fps optical-flow sampling stride)")
        fps = float(sample["fps"])
        interval = max(1, round(fps / 8.0))

        video_255 = video * 255.0
        chunk = self._chunk_size or 16

        # Cap chunk so the RAFT correlation volume doesn't overflow int32.
        # RAFT downsamples 8x in the feature encoder; CorrBlock's tensor is
        # shape (B*H1*W1, 1, H2, W2) with H1=H2=H/8, W1=W2=W/8. Its element
        # count is B*(H/8)^2*(W/8)^2 — F.avg_pool2d's index space starts to
        # overflow int32 around 2^31. Safety factor 2x.
        h_red = max(1, H // 8)
        w_red = max(1, W // 8)
        max_chunk = max(1, (1 << 30) // (h_red * h_red * w_red * w_red))
        chunk = min(chunk, max_chunk)

        indices = list(range(0, T, interval))
        n = len(indices)
        all_img1 = [video_255[indices[i]] for i in range(n - 1)]
        all_img2 = [video_255[indices[i + 1]] for i in range(n - 1)]

        scores: list[float] = []
        for start in range(0, len(all_img1), chunk):
            end = min(start + chunk, len(all_img1))
            img1_batch = torch.stack(all_img1[start:end]).to(self.device)
            img2_batch = torch.stack(all_img2[start:end]).to(self.device)
            padder = InputPadder(img1_batch.shape)
            img1p, img2p = padder.pad(img1_batch, img2_batch)
            _, flow = self._model(img1p, img2p, iters=20, test_mode=True)
            for i in range(flow.shape[0]):
                scores.append(self._get_score(flow[i]))

        scale = min(H, W)
        thres = 6.0 * (scale / 256.0)
        count_needed = round(4 * (n / 16.0))
        count_above = sum(1 for s in scores if s > thres)
        is_dynamic = 1.0 if count_above >= count_needed else 0.0

        return MetricResult(
            name=self.name,
            score=is_dynamic,
            details={
                "per_pair_magnitude": scores,
                "threshold": thres,
                "count_above": count_above,
                "count_needed": count_needed,
                "fps": fps,
                "interval": interval,
                "n_frames_used": n
            },
        )
