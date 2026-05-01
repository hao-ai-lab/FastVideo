"""VBench Dynamic Degree — RAFT optical flow motion detection.

For each consecutive frame pair, computes optical flow via RAFT and takes
the mean of the top 5% flow magnitudes.  If enough pairs exceed an
adaptive threshold, the video is classified as dynamic (1.0) vs static (0.0).
"""

from __future__ import annotations


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
    batch_unit = "frame_pair"

    def __init__(self) -> None:
        super().__init__()
        self._model = None
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

        args = EasyDict(small=False, mixed_precision=False,
                        alternate_corr=False, dropout=0.0)
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

    def trial_forward(self, batch_size, *, height, width, num_frames):
        from vbench.third_party.RAFT.core.utils_core.utils import InputPadder
        # RAFT's CorrBlock creates a (B*H*W, 1, H, W) correlation volume.
        # Cap resolution to avoid integer overflow at large batch sizes.
        h, w = min(height, 256), min(width, 256)
        img1 = torch.randn(batch_size, 3, h, w, device=self.device)
        img2 = torch.randn(batch_size, 3, h, w, device=self.device)
        padder = InputPadder(img1.shape)
        img1, img2 = padder.pad(img1, img2)
        with torch.no_grad():
            self._model(img1, img2, iters=20, test_mode=True)

    def _get_score(self, flow: torch.Tensor) -> float:
        """Top-5% mean flow magnitude (matching VBench dynamic_degree.get_score)."""
        flo = flow.permute(1, 2, 0).cpu().numpy()
        rad = np.sqrt(flo[..., 0] ** 2 + flo[..., 1] ** 2)
        h, w = rad.shape
        cut = max(1, int(h * w * 0.05))
        rad_flat = rad.flatten()
        return float(np.mean(np.sort(rad_flat)[-cut:]))

    @torch.no_grad()
    def compute(self, sample: dict) -> list[MetricResult]:
        from vbench.third_party.RAFT.core.utils_core.utils import InputPadder

        video = sample["video"]  # (B, T, C, H, W) float [0, 1]
        B, T, _, H, W = video.shape

        # fps controls the temporal sampling stride for optical flow.
        # vbench computes flow at 8fps (interval = round(fps/8)). The metric
        # cannot auto-derive fps from a tensor, so a missing fps would silently
        # use a wrong stride and produce a wrong score. Skip explicitly.
        if "fps" not in sample:
            return self._skip(sample, "missing 'fps' (required to set the "
                              "8fps optical-flow sampling stride)")
        fps = float(sample["fps"])
        interval = max(1, round(fps / 8.0))

        # Convert to uint8-scale float (RAFT expects ~[0,255])
        video_255 = video * 255.0
        chunk = self._chunk_size or 16

        # Cap chunk so the RAFT correlation volume doesn't overflow int32.
        # RAFT downsamples 8x in the feature encoder; CorrBlock's tensor is
        # shape (B*H1*W1, 1, H2, W2) with H1=H2=H/8, W1=W2=W/8. Its element
        # count is B*(H/8)^2*(W/8)^2 — F.avg_pool2d's index space starts to
        # overflow int32 around 2^31. Safety factor 2x.
        # At 384x640 → max_chunk=73 (batching helps).
        # At 1088x1920 → max_chunk=1 (matches upstream's pair-by-pair loop).
        h_red = max(1, H // 8)
        w_red = max(1, W // 8)
        max_chunk = max(1, (1 << 30) // (h_red * h_red * w_red * w_red))
        chunk = min(chunk, max_chunk)

        # Collect all frame pairs across all B videos
        all_img1, all_img2 = [], []
        pairs_per_video = []
        for b in range(B):
            indices = list(range(0, T, interval))
            n = len(indices)
            pairs_per_video.append(n - 1)
            for i in range(n - 1):
                all_img1.append(video_255[b, indices[i]])
                all_img2.append(video_255[b, indices[i + 1]])

        total_pairs = len(all_img1)

        # Batched RAFT forward with chunking
        all_scores = []
        for start in range(0, total_pairs, chunk):
            end = min(start + chunk, total_pairs)
            img1_batch = torch.stack(all_img1[start:end]).to(self.device)
            img2_batch = torch.stack(all_img2[start:end]).to(self.device)
            padder = InputPadder(img1_batch.shape)
            img1p, img2p = padder.pad(img1_batch, img2_batch)
            _, flow = self._model(img1p, img2p, iters=20, test_mode=True)
            for i in range(flow.shape[0]):
                all_scores.append(self._get_score(flow[i]))

        # Distribute back to per-video results
        results = []
        offset = 0
        for b in range(B):
            n_pairs = pairs_per_video[b]
            scores = all_scores[offset:offset + n_pairs]
            offset += n_pairs

            n = n_pairs + 1  # number of frames used
            scale = min(H, W)
            thres = 6.0 * (scale / 256.0)
            count_needed = round(4 * (n / 16.0))

            count_above = sum(1 for s in scores if s > thres)
            is_dynamic = 1.0 if count_above >= count_needed else 0.0

            results.append(MetricResult(
                name=self.name,
                score=is_dynamic,
                details={"per_pair_magnitude": scores,
                         "threshold": thres,
                         "count_above": count_above,
                         "count_needed": count_needed,
                         "fps": fps,
                         "interval": interval,
                         "n_frames_used": n},
            ))

        return results
