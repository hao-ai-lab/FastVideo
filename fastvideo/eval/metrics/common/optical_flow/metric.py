from __future__ import annotations

import numpy as np
import torch

from fastvideo.eval.metrics.base import BaseMetric
from fastvideo.eval.registry import register
from fastvideo.eval.types import MetricResult


def _compute_pixel_epe(
    flow_gt: np.ndarray,
    flow_gen: np.ndarray,
    min_mag: float = 0.5,
    max_mag_pct: float = 80.0,
) -> float:
    """Mean end-point error between two (H, W, 2) flow fields."""
    gt_mag = np.linalg.norm(flow_gt, axis=-1)
    gen_mag = np.linalg.norm(flow_gen, axis=-1)
    max_mag = np.maximum(gt_mag, gen_mag)

    mag_hi = np.percentile(max_mag, max_mag_pct)
    mask = (max_mag >= min_mag) & (max_mag <= mag_hi)

    epe_map = np.linalg.norm(flow_gt - flow_gen, axis=-1)
    if mask.sum() > 0:
        return float(epe_map[mask].mean())
    return float(epe_map.mean())


@register("common.optical_flow")
class OpticalFlowMetric(BaseMetric):
    """Compare optical flow between generated and reference videos.

    All frame pairs from all B videos (both gen and ref) are pooled
    and batched through the model together for GPU efficiency.
    """

    name = "common.optical_flow"
    requires_reference = True
    higher_is_better = False
    needs_gpu = True
    backbone = "optical_flow"
    dependencies = ["ptlflow"]
    batch_unit = "frame_pair"

    def __init__(
        self,
        model_name: str = "dpflow",
        ckpt: str = "things",
        min_mag: float = 0.5,
        max_mag_pct: float = 80.0,
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.ckpt = ckpt
        self.min_mag = min_mag
        self.max_mag_pct = max_mag_pct
        self._model = None
        self._chunk_size = 16  # default until calibrated

    def to(self, device: str | torch.device) -> OpticalFlowMetric:
        super().to(device)
        if self._model is not None:
            self._model = self._model.to(self.device)
        return self

    def setup(self) -> None:
        if self._model is not None:
            return
        import ptlflow
        self._model = ptlflow.get_model(self.model_name, ckpt_path=self.ckpt)
        self._model.eval()
        self._model = self._model.to(self.device)

    def trial_forward(self, batch_size, *, height, width, num_frames):
        from ptlflow.utils.io_adapter import IOAdapter
        io_adapter = IOAdapter(
            output_stride=self._model.output_stride,
            input_size=(height, width),
            cuda=(self.device.type == "cuda"),
        )
        # Create batch_size frame pairs
        dummy_bgr = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
        pair_tensors = []
        for _ in range(batch_size):
            inputs = io_adapter.prepare_inputs([dummy_bgr, dummy_bgr])
            pair_tensors.append(inputs["images"])
        batched = torch.cat(pair_tensors, dim=0)
        with torch.no_grad():
            self._model({"images": batched})

    def _compute_flows_batched(
        self, pair_frames: list[tuple[np.ndarray, np.ndarray]], io_adapter,
    ) -> list[np.ndarray]:
        """Compute flow for a list of (frame1, frame2) pairs, batched."""
        chunk = self._chunk_size or 16
        all_flows = []

        for batch_start in range(0, len(pair_frames), chunk):
            batch_end = min(batch_start + chunk, len(pair_frames))

            pair_tensors = []
            for f1, f2 in pair_frames[batch_start:batch_end]:
                inputs = io_adapter.prepare_inputs([f1, f2])
                pair_tensors.append(inputs["images"])

            batched_images = torch.cat(pair_tensors, dim=0)

            with torch.no_grad():
                preds = self._model({"images": batched_images})

            preds["images"] = batched_images
            preds = io_adapter.unscale(preds)

            flows_tensor = preds["flows"]
            if flows_tensor.dim() == 5:
                flows_tensor = flows_tensor.squeeze(1)
            for i in range(flows_tensor.shape[0]):
                flow = flows_tensor[i].detach().cpu().permute(1, 2, 0).numpy()
                all_flows.append(flow)

        return all_flows

    def compute(self, sample: dict) -> list[MetricResult]:
        if self._model is None:
            self.setup()

        from ptlflow.utils.io_adapter import IOAdapter

        gen_video = sample["video"].float()       # (B, T, C, H, W)
        ref_video = sample["reference"].float()   # (B, T, C, H, W)
        B = gen_video.shape[0]
        n = min(gen_video.shape[1], ref_video.shape[1])
        gen_video = gen_video[:, :n]
        ref_video = ref_video[:, :n]
        n_pairs = n - 1

        if n < 2:
            raise ValueError("Need at least 2 frames to compute optical flow")

        h, w = gen_video.shape[3], gen_video.shape[4]
        io_adapter = IOAdapter(
            output_stride=self._model.output_stride,
            input_size=(h, w),
            cuda=(self.device.type == "cuda"),
        )

        gen_pairs = []
        ref_pairs = []
        for b in range(B):
            gen_frames = _tensor_to_bgr_list(gen_video[b])
            ref_frames = _tensor_to_bgr_list(ref_video[b])
            for i in range(n_pairs):
                gen_pairs.append((gen_frames[i], gen_frames[i + 1]))
                ref_pairs.append((ref_frames[i], ref_frames[i + 1]))

        all_gen_flows = self._compute_flows_batched(gen_pairs, io_adapter)
        all_ref_flows = self._compute_flows_batched(ref_pairs, io_adapter)

        results = []
        for b in range(B):
            start = b * n_pairs
            end = start + n_pairs
            per_frame_epe = [
                _compute_pixel_epe(rf, gf, self.min_mag, self.max_mag_pct)
                for rf, gf in zip(all_ref_flows[start:end], all_gen_flows[start:end])
            ]
            results.append(MetricResult(
                name=self.name,
                score=float(np.mean(per_frame_epe)),
                details={"per_frame_epe": per_frame_epe},
            ))
        return results


def _tensor_to_bgr_list(video: torch.Tensor) -> list[np.ndarray]:
    frames = []
    for t in range(video.shape[0]):
        rgb = (video[t].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        bgr = rgb[:, :, ::-1].copy()
        frames.append(bgr)
    return frames
