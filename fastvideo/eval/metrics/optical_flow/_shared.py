"""Shared helpers for optical-flow metrics.

Both ``optical_flow.gt_optical_flow`` and
``optical_flow.synthetic_optical_flow`` extract per-frame flow with
``ptlflow`` and reduce it through the same per-pixel / per-frame /
temporal aggregation pipeline. The pipeline lives here so the two
metrics stay byte-identical on the comparison side and only differ in
how they construct the *reference* flow field.
"""

from __future__ import annotations

import numpy as np
import torch

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
    fn = getattr(np, "trapezoid", None) or np.trapz
    return float(fn(vals))


def _estimate_foe(
    flow: np.ndarray,
    step: int = 8,
    min_mag: float = 0.5,
) -> tuple[float, float]:
    """Least-squares Focus of Expansion. Returns (fx, fy)."""
    H, W = flow.shape[:2]
    ys = np.arange(step // 2, H, step)
    xs = np.arange(step // 2, W, step)
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    yy = yy.ravel()
    xx = xx.ravel()
    uu = flow[yy, xx, 0]
    vv = flow[yy, xx, 1]

    mag = np.sqrt(uu**2 + vv**2)
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
        mag = np.sqrt(u**2 + v**2)
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


def compute_frame_metrics(
    flow_gt: np.ndarray,
    flow_gen: np.ndarray,
    grid_size: int = 8,
    min_mag: float = 0.5,
    max_mag_pct: float = 80.0,
) -> dict[str, float]:
    """Per-frame comparison metrics between two HxWx2 flow fields.

    Port of mhuo's compute_frame_metrics — see ptlflow_validation.py.
    """
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
        dot = (flow_gt[:, :, 0] * flow_gen[:, :, 0] + flow_gt[:, :, 1] * flow_gen[:, :, 1])
        cos_map = np.clip(dot / (gt_mag_map * gen_mag_map + 1e-8), -1.0, 1.0)
        angle_map = np.degrees(np.arccos(cos_map))
        metrics["px_angle_rmse"] = float(np.sqrt((angle_map[valid]**2).mean()))
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
    metrics["foe_dist"] = float(np.sqrt((foe_gt_x - foe_gen_x)**2 + (foe_gt_y - foe_gen_y)**2))

    metrics["flow_kl_2d"] = _flow_kl_2d(flow_gt, flow_gen)
    return metrics


def aggregate_temporal(per_frame: list[dict[str, float]], ) -> dict[str, float | int | None]:
    """Aggregate per-frame metric dicts into mean/std/max/auc/onset summaries.

    Port of mhuo's compute_temporal_metrics.
    """
    n = len(per_frame)
    if n == 0:
        return {"n_frames": 0}

    summary: dict[str, float | int | None] = {"n_frames": n}
    series: dict[str, np.ndarray] = {k: np.array([m[k] for m in per_frame]) for k in _PER_FRAME_AGG_KEYS}
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


def tensor_to_bgr_list(video: torch.Tensor) -> list[np.ndarray]:
    """Convert ``(T, C, H, W)`` float [0,1] to a list of HWC BGR uint8 frames.

    Casts + permutes + BGR-swaps on-device and transfers once, instead
    of T per-frame ``.cpu()`` round-trips.
    """
    bgr_u8 = (video.float() * 255.0).clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1).flip(-1).contiguous()
    arr = bgr_u8.cpu().numpy()
    return [arr[t] for t in range(arr.shape[0])]


def load_ptlflow_model(model_name: str, ckpt: str, device: torch.device):
    """Load a ``ptlflow`` model on *device* in eval mode."""
    import ptlflow
    model = ptlflow.get_model(model_name, ckpt_path=ckpt)
    model.eval()
    return model.to(device)


def extract_video_flows(
    model,
    video: torch.Tensor,  # (T, C, H, W) float [0, 1]
    *,
    chunk: int,
    device: torch.device,
) -> list[np.ndarray]:
    """Run *model* on every consecutive frame pair in *video*.

    Returns a list of HxWx2 flow arrays of length ``T - 1``.
    """
    from ptlflow.utils.io_adapter import IOAdapter

    h, w = video.shape[2], video.shape[3]
    io_adapter = IOAdapter(
        output_stride=model.output_stride,
        input_size=(h, w),
        cuda=(device.type == "cuda"),
    )
    bgr_frames = tensor_to_bgr_list(video)
    pairs = [(bgr_frames[i], bgr_frames[i + 1]) for i in range(len(bgr_frames) - 1)]

    flows: list[np.ndarray] = []
    for start in range(0, len(pairs), chunk):
        end = min(start + chunk, len(pairs))
        pair_tensors = []
        for f1, f2 in pairs[start:end]:
            inputs = io_adapter.prepare_inputs([f1, f2])
            pair_tensors.append(inputs["images"])
        batched_images = torch.cat(pair_tensors, dim=0)
        with torch.no_grad():
            preds = model({"images": batched_images})
        preds["images"] = batched_images
        preds = io_adapter.unscale(preds)
        flows_tensor = preds["flows"]
        if flows_tensor.dim() == 5:
            flows_tensor = flows_tensor.squeeze(1)
        for i in range(flows_tensor.shape[0]):
            flow = flows_tensor[i].detach().cpu().permute(1, 2, 0).numpy()
            flows.append(flow)
    return flows
