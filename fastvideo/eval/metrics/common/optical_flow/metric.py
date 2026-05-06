from __future__ import annotations

from typing import Tuple

import numpy as np
import torch

from fastvideo.eval.metrics.base import BaseMetric
from fastvideo.eval.registry import register
from fastvideo.eval.types import MetricResult

_PER_FRAME_AGG_KEYS: tuple[str, ...] = (
    "mf_epe",
    "mf_angle_err",
    "mf_cosine",
    "mf_mag_ratio",
    "pixel_epe_mean",
    "pixel_epe_max",
    "px_angle_rmse",
    "grid_epe_mean",
    "grid_epe_max",
    "fl_all",
    "foe_dist",
    "flow_kl_2d",
)


def _trapezoid(vals: np.ndarray) -> float:
    fn = getattr(np, "trapezoid", None) or getattr(np, "trapz")
    return float(fn(vals))


def _estimate_foe(
    flow: np.ndarray,
    step: int = 8,
    min_mag: float = 0.5,
) -> Tuple[float, float]:
    """Least-squares Focus of Expansion. Returns (fx, fy)."""
    H, W = flow.shape[:2]
    ys = np.arange(step // 2, H, step)
    xs = np.arange(step // 2, W, step)
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    yy = yy.ravel()
    xx = xx.ravel()
    uu = flow[yy, xx, 0]
    vv = flow[yy, xx, 1]

    mag = np.sqrt(uu ** 2 + vv ** 2)
    valid = mag > min_mag
    if valid.sum() < 10:
        return W / 2.0, H / 2.0

    xx = xx[valid].astype(np.float64)
    yy = yy[valid].astype(np.float64)
    uu = uu[valid].astype(np.float64)
    vv = vv[valid].astype(np.float64)

    # v * fx - u * fy = v * x - u * y
    A = np.column_stack([vv, -uu])
    b = vv * xx - uu * yy
    result, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    return float(result[0]), float(result[1])


def _flow_kl_2d(
    flow_a: np.ndarray,
    flow_b: np.ndarray,
    n_angle_bins: int = 36,
    n_mag_bins: int = 20,
    min_mag: float = 0.5,
) -> float:
    """KL(P_a || P_b) over a joint (angle, log-magnitude) histogram."""
    def _hist(flow: np.ndarray) -> np.ndarray | None:
        u, v = flow[:, :, 0].ravel(), flow[:, :, 1].ravel()
        mag = np.sqrt(u ** 2 + v ** 2)
        angle = np.degrees(np.arctan2(v, u)) % 360
        valid = mag >= min_mag
        if valid.sum() < 10:
            return None
        mag = mag[valid]
        angle = angle[valid]
        mag_max = max(mag.max(), min_mag + 1.0)
        mag_edges = np.logspace(np.log10(min_mag), np.log10(mag_max), n_mag_bins + 1)
        angle_edges = np.linspace(0, 360, n_angle_bins + 1)
        h, _, _ = np.histogram2d(angle, mag, bins=[angle_edges, mag_edges])
        return h

    ha, hb = _hist(flow_a), _hist(flow_b)
    if ha is None or hb is None:
        return 0.0
    eps = 1.0
    p = (ha + eps) / (ha + eps).sum()
    q = (hb + eps) / (hb + eps).sum()
    return float((p * np.log(p / q)).sum())


def _compute_frame_metrics(
    flow_gt: np.ndarray,
    flow_gen: np.ndarray,
    grid_size: int = 8,
    min_mag: float = 0.5,
    max_mag_pct: float = 80.0,
) -> dict[str, float]:
    """Port of mhuo's compute_frame_metrics — see ptlflow_validation.py."""
    metrics: dict[str, float] = {}

    gt_mag_map = np.linalg.norm(flow_gt, axis=2)
    gen_mag_map = np.linalg.norm(flow_gen, axis=2)
    max_mag_map = np.maximum(gt_mag_map, gen_mag_map)
    mag_hi = np.percentile(max_mag_map, max_mag_pct)
    mag_mask = (max_mag_map >= min_mag) & (max_mag_map <= mag_hi)
    n_valid = int(mag_mask.sum())

    if n_valid > 0:
        mean_gt = flow_gt[mag_mask].mean(axis=0)
        mean_gen = flow_gen[mag_mask].mean(axis=0)
    else:
        mean_gt = flow_gt.reshape(-1, 2).mean(axis=0)
        mean_gen = flow_gen.reshape(-1, 2).mean(axis=0)

    metrics["mf_epe"] = float(np.linalg.norm(mean_gt - mean_gen))

    mf_min_mag = 0.1
    mag_gt = float(np.linalg.norm(mean_gt))
    mag_gen = float(np.linalg.norm(mean_gen))
    if mag_gt < mf_min_mag and mag_gen < mf_min_mag:
        metrics["mf_angle_err"] = 0.0
        metrics["mf_cosine"] = 1.0
    elif mag_gt < mf_min_mag or mag_gen < mf_min_mag:
        metrics["mf_angle_err"] = 90.0
        metrics["mf_cosine"] = 0.0
    elif mag_gt > 1e-6 and mag_gen > 1e-6:
        cos_sim = float(np.dot(mean_gt, mean_gen) / (mag_gt * mag_gen))
        cos_sim = float(np.clip(cos_sim, -1.0, 1.0))
        metrics["mf_angle_err"] = float(np.degrees(np.arccos(cos_sim)))
        metrics["mf_cosine"] = cos_sim
    else:
        metrics["mf_angle_err"] = 0.0
        metrics["mf_cosine"] = 1.0

    metrics["mf_mag_ratio"] = float(mag_gen / mag_gt) if mag_gt > 1e-6 else 1.0

    epe_map = np.linalg.norm(flow_gt - flow_gen, axis=2)
    if n_valid > 0:
        metrics["pixel_epe_mean"] = float(epe_map[mag_mask].mean())
        metrics["pixel_epe_max"] = float(epe_map[mag_mask].max())
    else:
        metrics["pixel_epe_mean"] = float(epe_map.mean())
        metrics["pixel_epe_max"] = float(epe_map.max())

    valid = mag_mask & (gt_mag_map > 0.5) & (gen_mag_map > 0.5)
    if valid.sum() > 0:
        dot = (flow_gt[:, :, 0] * flow_gen[:, :, 0]
               + flow_gt[:, :, 1] * flow_gen[:, :, 1])
        cos_map = np.clip(dot / (gt_mag_map * gen_mag_map + 1e-8), -1.0, 1.0)
        angle_map = np.degrees(np.arccos(cos_map))
        metrics["px_angle_rmse"] = float(np.sqrt((angle_map[valid] ** 2).mean()))
    else:
        metrics["px_angle_rmse"] = 0.0

    H, W = epe_map.shape
    gh, gw = H // grid_size, W // grid_size
    grid_vals = []
    for gi in range(grid_size):
        for gj in range(grid_size):
            cell_mask = mag_mask[gi * gh:(gi + 1) * gh, gj * gw:(gj + 1) * gw]
            cell_epe = epe_map[gi * gh:(gi + 1) * gh, gj * gw:(gj + 1) * gw]
            if cell_mask.sum() > 0:
                grid_vals.append(float(cell_epe[cell_mask].mean()))
            else:
                grid_vals.append(float(cell_epe.mean()))
    metrics["grid_epe_mean"] = float(np.mean(grid_vals))
    metrics["grid_epe_max"] = float(np.max(grid_vals))

    if n_valid > 0:
        outlier = (epe_map > 3.0) & (epe_map > 0.05 * gt_mag_map) & mag_mask
        metrics["fl_all"] = float(outlier.sum() / n_valid)
    else:
        outlier = (epe_map > 3.0) & (epe_map > 0.05 * gt_mag_map)
        metrics["fl_all"] = float(outlier.mean())

    foe_gt_x, foe_gt_y = _estimate_foe(flow_gt)
    foe_gen_x, foe_gen_y = _estimate_foe(flow_gen)
    metrics["foe_dist"] = float(
        np.sqrt((foe_gt_x - foe_gen_x) ** 2 + (foe_gt_y - foe_gen_y) ** 2)
    )

    metrics["flow_kl_2d"] = _flow_kl_2d(flow_gt, flow_gen)
    return metrics


def _aggregate_temporal(
    per_frame: list[dict[str, float]],
) -> dict[str, float | int | None]:
    """Port of mhuo's compute_temporal_metrics."""
    n = len(per_frame)
    if n == 0:
        return {"n_frames": 0}

    summary: dict[str, float | int | None] = {"n_frames": n}
    series: dict[str, np.ndarray] = {
        k: np.array([m[k] for m in per_frame]) for k in _PER_FRAME_AGG_KEYS
    }
    for name, vals in series.items():
        summary[f"{name}_mean"] = float(vals.mean())
        summary[f"{name}_std"] = float(vals.std())
        summary[f"{name}_max"] = float(vals.max())
        summary[f"{name}_auc"] = _trapezoid(vals) / max(n - 1, 1)

    epe_series = series["pixel_epe_mean"]
    window = min(5, n)
    if n >= window:
        baseline = float(np.median(epe_series[:window]))
        threshold = max(baseline * 2.0, 1.0)
        kernel = np.ones(window) / window
        smoothed = np.convolve(epe_series, kernel, mode="valid")
        divergence_frame: int | None = None
        for i, val in enumerate(smoothed):
            if val > threshold:
                divergence_frame = int(i)
                break
        summary["divergence_onset_frame"] = divergence_frame
        summary["divergence_threshold"] = float(threshold)
    else:
        summary["divergence_onset_frame"] = None
        summary["divergence_threshold"] = None
    return summary


@register("common.optical_flow")
class OpticalFlowMetric(BaseMetric):
    """Compare optical flow between generated and reference videos.

    Computes the metric set used by mhuo's ptlflow validation
    (see fastvideo/training/ptlflow_validation.py in mhuo's tree):
    mf_epe, mf_angle_err, mf_cosine, mf_mag_ratio, pixel_epe_mean/max,
    px_angle_rmse, grid_epe_*, fl_all, foe_dist, flow_kl_2d.
    Aggregated over time as _mean/_std/_max/_auc.

    The headline ``score`` is ``pixel_epe_mean_mean`` (lower is better);
    every other scalar lives in ``details`` so downstream consumers can
    pick whichever one they care about.

    All frame pairs from all B videos (both gen and ref) are pooled and
    batched through the model together for GPU efficiency.
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
        grid_size: int = 8,
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.ckpt = ckpt
        self.min_mag = min_mag
        self.max_mag_pct = max_mag_pct
        self.grid_size = grid_size
        self._model = None
        self._chunk_size = 16

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
        chunk = self._chunk_size or 16
        all_flows: list[np.ndarray] = []
        for start in range(0, len(pair_frames), chunk):
            end = min(start + chunk, len(pair_frames))
            pair_tensors = []
            for f1, f2 in pair_frames[start:end]:
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
        ref_video = sample["reference"].float()
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

        gen_pairs: list[tuple[np.ndarray, np.ndarray]] = []
        ref_pairs: list[tuple[np.ndarray, np.ndarray]] = []
        for b in range(B):
            gf = _tensor_to_bgr_list(gen_video[b])
            rf = _tensor_to_bgr_list(ref_video[b])
            for i in range(n_pairs):
                gen_pairs.append((gf[i], gf[i + 1]))
                ref_pairs.append((rf[i], rf[i + 1]))

        all_gen_flows = self._compute_flows_batched(gen_pairs, io_adapter)
        all_ref_flows = self._compute_flows_batched(ref_pairs, io_adapter)

        results: list[MetricResult] = []
        for b in range(B):
            start = b * n_pairs
            end = start + n_pairs
            per_frame = [
                _compute_frame_metrics(
                    rf, gf,
                    grid_size=self.grid_size,
                    min_mag=self.min_mag,
                    max_mag_pct=self.max_mag_pct,
                )
                for rf, gf in zip(all_ref_flows[start:end], all_gen_flows[start:end])
            ]
            summary = _aggregate_temporal(per_frame)
            score = summary.get("pixel_epe_mean_mean")
            details = dict(summary)
            details["per_frame_metrics"] = per_frame
            results.append(MetricResult(
                name=self.name,
                score=float(score) if score is not None else None,
                details=details,
            ))
        return results


def _tensor_to_bgr_list(video: torch.Tensor) -> list[np.ndarray]:
    frames = []
    for t in range(video.shape[0]):
        rgb = (video[t].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        bgr = rgb[:, :, ::-1].copy()
        frames.append(bgr)
    return frames
