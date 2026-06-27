# SPDX-License-Identifier: Apache-2.0
"""CoTracker End-Point-Error (EPE): does the generated motion follow the control?

This is MotionStream's motion-fidelity metric: the L2 distance between the
**input control tracks** and the tracks **re-extracted from the generated video**
(via CoTracker3) at the same query points. Low EPE = the model followed the
trajectories you asked for; high EPE = it ignored them.

Unlike a reference-video metric, this needs the *control tracks* that were fed to
the model, so it reads them from the sample:

    sample["video"]               (T, C, H, W) float in [0, 1]   (generated)
    sample["input_tracks"]        (T, N, 2)   control tracks
    sample["input_visibility"]    (T, N)      control visibility (1 = active)
    sample["input_tracks_normalized"]  bool   if True (default), coords are in
                                              [0,1] and get scaled by (W, H)

A module-level :func:`compute_epe` is exposed for standalone use (the
controllability test + Gradio app call it directly).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from fastvideo.eval.metrics.base import BaseMetric
from fastvideo.eval.registry import register
from fastvideo.eval.types import MetricResult

_HUB_REPO = "facebookresearch/co-tracker"
_HUB_MODEL = "cotracker3_offline"


def load_cotracker(device: str | torch.device):
    """Load CoTracker3 (prefer the offline hub cache; mirrors extract_tracks.py)."""
    hub_dir = Path(torch.hub.get_dir())
    local = hub_dir / (_HUB_REPO.replace("/", "_") + "_main")
    if local.exists():
        model = torch.hub.load(str(local), _HUB_MODEL, source="local", trust_repo=True)
    else:
        model = torch.hub.load(_HUB_REPO, _HUB_MODEL, trust_repo=True)
    return model.to(device).eval()


@torch.no_grad()
def _retrack(model, frames_thwc: np.ndarray, queries_txy: np.ndarray, device):
    """Track ``queries_txy`` (M,3 = t,x,y pixels) through ``frames_thwc`` (T,H,W,3 uint8).

    Returns (tracks [T,M,2] pixels, vis [T,M]).
    """
    video = torch.from_numpy(frames_thwc).float().permute(0, 3, 1, 2)[None]  # [1,T,3,H,W]
    q = torch.from_numpy(queries_txy.astype(np.float32))[None]               # [1,M,3]
    tracks, vis = model(video.to(device), queries=q.to(device))
    return tracks[0].cpu().numpy(), vis[0].cpu().numpy()


def compute_epe(frames_thwc: np.ndarray, input_tracks_px: np.ndarray,
                input_vis: np.ndarray, model, device, *, vis_thresh: float = 0.5,
                query_frame: int = 0, max_points: int = 600, seed: int = 0) -> dict:
    """EPE between input control tracks and tracks re-extracted from generated frames.

    ``frames_thwc``: generated video (T,H,W,3) uint8.
    ``input_tracks_px``: (T,N,2) control tracks in PIXEL coords of that video.
    ``input_vis``: (T,N) control visibility.
    ``max_points``: cap re-tracked points for speed (random subsample of active).
    Returns dict(epe, per_frame, n_points, coverage).
    """
    T = min(frames_thwc.shape[0], input_tracks_px.shape[0])
    frames_thwc = frames_thwc[:T]
    tracks_in = input_tracks_px[:T]
    vis_in = input_vis[:T]

    # Query the points that are active at the query frame, at their position there.
    active = vis_in[query_frame] > vis_thresh
    idx = np.nonzero(active)[0]
    if idx.size == 0:
        return {"epe": None, "per_frame": [], "n_points": 0, "coverage": 0.0}
    if max_points and idx.size > max_points:
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(idx, size=max_points, replace=False))
    q_xy = tracks_in[query_frame, idx]                                   # [M,2]
    queries = np.concatenate([np.full((idx.size, 1), query_frame, np.float32), q_xy], axis=1)

    rt_tracks, _rt_vis = _retrack(model, frames_thwc, queries, device)   # [T,M,2]
    Tr = min(T, rt_tracks.shape[0])

    tgt = tracks_in[:Tr][:, idx]                                         # [Tr,M,2]
    pred = rt_tracks[:Tr]                                                # [Tr,M,2]
    mask = vis_in[:Tr][:, idx] > vis_thresh                             # [Tr,M] follow input visibility
    d = np.sqrt(((pred - tgt) ** 2).sum(-1))                            # [Tr,M]

    per_frame = [float(d[t][mask[t]].mean()) if mask[t].any() else float("nan") for t in range(Tr)]
    valid = d[mask]
    epe = float(valid.mean()) if valid.size else None
    return {"epe": epe, "per_frame": per_frame, "n_points": int(idx.size),
            "coverage": float(mask.mean())}


@register("motion.cotracker_epe")
class CoTrackerEPEMetric(BaseMetric):
    """End-Point-Error of generated motion vs the input control tracks."""

    name = "motion.cotracker_epe"
    requires_reference = False  # needs control tracks in the sample, not a ref video
    higher_is_better = False    # EPE is a distance
    needs_gpu = True
    dependencies: list[str] = []

    def __init__(self, vis_thresh: float = 0.5, query_frame: int = 0) -> None:
        super().__init__()
        self.vis_thresh = float(vis_thresh)
        self.query_frame = int(query_frame)
        self._model = None

    def setup(self) -> None:
        if self._model is None:
            self._model = load_cotracker(self.device)

    def compute(self, sample: dict) -> MetricResult:
        if self._model is None:
            self.setup()
        video = sample.get("video")
        tracks = sample.get("input_tracks")
        vis = sample.get("input_visibility")
        if video is None or tracks is None or vis is None:
            return self._skip(sample, "missing video/input_tracks/input_visibility")

        # video (T,C,H,W) float [0,1] -> (T,H,W,3) uint8
        v = video.detach().cpu().numpy() if torch.is_tensor(video) else np.asarray(video)
        frames = np.transpose((np.clip(v, 0, 1) * 255).astype(np.uint8), (0, 2, 3, 1))
        T, H, W, _ = frames.shape

        tr = tracks.detach().cpu().numpy() if torch.is_tensor(tracks) else np.asarray(tracks)
        vs = vis.detach().cpu().numpy() if torch.is_tensor(vis) else np.asarray(vis)
        tr = tr.astype(np.float32).copy()
        if bool(sample.get("input_tracks_normalized", True)):
            tr[..., 0] *= W
            tr[..., 1] *= H

        res = compute_epe(frames, tr, vs, self._model, self.device,
                          vis_thresh=self.vis_thresh, query_frame=self.query_frame)
        if res["epe"] is None:
            return self._skip(sample, "no visible control points at query frame")
        return MetricResult(name=self.name, score=res["epe"],
                            details={"per_frame": res["per_frame"],
                                     "n_points": res["n_points"],
                                     "coverage": res["coverage"]})
