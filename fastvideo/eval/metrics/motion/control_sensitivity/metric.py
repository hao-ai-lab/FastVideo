# SPDX-License-Identifier: Apache-2.0
"""Counterfactual control sensitivity: does changing the control change the output?

The CoTracker-EPE metric asks "did the generated motion match the input tracks".
That can be fooled: on a static-ish clip with a static background, averaging EPE
over all points scores well even if the model ignored the *moved* points.

This metric instead does an **intervention diff**. Generate twice from the SAME
conditioning (first frame, text, seed, noise) but DIFFERENT control tracks:
``frames_a`` (reference, e.g. GT tracks) vs ``frames_b`` (a modified/foreign
control). The pixel diff ``|A - B|`` is the model's response to the control.

Scored only where the control *asks* for motion (the ROI = where ``frames_b``'s
tracks move), so a static background can't dilute it:

  - ``sensitivity_roi``    mean |A-B| inside the ROI      (high = it responded)
  - ``bg_leakage``         mean |A-B| outside the ROI     (low  = changes stayed local)
  - ``localization_iou``   IoU(where-control-moves, where-output-changes)
                                                          (high = changed where asked)

A model that ignores the tracks gives sensitivity_roi ~ 0 and iou ~ 0 regardless
of how its weights look. Diffs are in [0,1] (mean abs RGB diff / 255).

``compute_control_sensitivity`` and ``render_diff_heatmap`` are numpy-only so the
viewer / offline tooling can call them without a GPU.
"""
from __future__ import annotations

from typing import Any

import numpy as np


def _stamp(mask: np.ndarray, pts_xy: np.ndarray, radius: int) -> None:
    h, w = mask.shape
    for x, y in pts_xy:
        xi, yi = int(round(x)), int(round(y))
        x0, x1 = max(0, xi - radius), min(w, xi + radius + 1)
        y0, y1 = max(0, yi - radius), min(h, yi + radius + 1)
        if x1 > x0 and y1 > y0:
            mask[y0:y1, x0:x1] = True


def _moving_roi(tracks_px: Any, vis: Any, T: int, h: int, w: int, down: float, move_thresh: float,
                roi_radius: float) -> np.ndarray:
    """Mask [T,h,w] of where the control tracks move (displacement from frame 0)."""
    ta = tracks_px[:T]
    vv = (vis[:T] > 0.5) if vis is not None else np.ones(ta.shape[:2], bool)
    disp = np.sqrt(((ta - ta[0:1])**2).sum(-1))  # [T,N] px from frame-0 pos
    rr = max(1, int(round(roi_radius / down)))
    roi = np.zeros((T, h, w), bool)
    for t in range(T):
        moved = vv[t] & (disp[t] > move_thresh)
        if moved.any():
            _stamp(roi[t], ta[t][moved] / down, rr)
    return roi


def compute_control_sensitivity(frames_a: np.ndarray,
                                frames_b: np.ndarray,
                                tracks_px: np.ndarray | None = None,
                                vis: np.ndarray | None = None,
                                *,
                                down: int = 2,
                                move_thresh: float = 6.0,
                                diff_thresh: float = 0.06,
                                roi_radius: int = 12) -> dict:
    """Diff two generations (same conditioning, different control). See module docstring.

    ``frames_a``/``frames_b``: (T,H,W,3) uint8.  ``tracks_px``: control-B tracks in
    pixel coords (defines the ROI); if None the ROI is the whole frame (global diff).
    """
    T = min(len(frames_a), len(frames_b))
    a = frames_a[:T, ::down, ::down].astype(np.float32) / 255.0
    b = frames_b[:T, ::down, ::down].astype(np.float32) / 255.0
    diff = np.abs(a - b).mean(-1)  # [T,h,w] in [0,1]
    h, w = diff.shape[1:]

    if tracks_px is not None:
        roi = _moving_roi(tracks_px, vis, T, h, w, down, move_thresh, roi_radius)
        if not roi.any():  # control has no motion -> fall back to global
            roi = np.ones_like(diff, bool)
    else:
        roi = np.ones_like(diff, bool)

    did = diff > diff_thresh
    inter = float((roi & did).sum())
    union = float((roi | did).sum())
    out = ~roi
    return {
        "sensitivity_roi": float(diff[roi].mean()) if roi.any() else float("nan"),
        "bg_leakage": float(diff[out].mean()) if out.any() else 0.0,
        "localization_iou": (inter / union) if union > 0 else 0.0,
        "mean_diff": float(diff.mean()),
        "roi_frac": float(roi.mean()),
    }


def _hot(e: np.ndarray) -> np.ndarray:
    r = np.clip(e * 3, 0, 1)
    g = np.clip(e * 3 - 1, 0, 1)
    b = np.clip(e * 3 - 2, 0, 1)
    return (np.stack([r, g, b], -1) * 255).astype(np.uint8)


def render_diff_heatmap(frames_a: np.ndarray, frames_b: np.ndarray, *, scale: float = 0.35) -> np.ndarray:
    """(T,H,W,3) uint8 'hot' heatmap of |A-B| (black=identical -> white=max diff)."""
    T = min(len(frames_a), len(frames_b))
    d = np.abs(frames_a[:T].astype(np.float32) / 255.0 - frames_b[:T].astype(np.float32) / 255.0).mean(-1)
    return _hot(np.clip(d / scale, 0, 1))
