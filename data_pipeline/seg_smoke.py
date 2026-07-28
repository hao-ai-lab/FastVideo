# SPDX-License-Identifier: Apache-2.0
"""Smoke-test: run several ultralytics segmentation backends in no-prompt
'everything' mode on one real openvid frame. Report #masks + latency so we can
pick which variants are worth a full render. aarch64-safe (PyAV decode, no cv2)."""
import json
import sys
import time
from pathlib import Path

import numpy as np

DATA = Path("/mnt/lustre/vlm-s4duan/openvid_1m")
WEIGHTS = Path("/mnt/lustre/vlm-s4duan/models/seg")
WEIGHTS.mkdir(parents=True, exist_ok=True)

# (display name, weight file, loader class)
MODELS = [
    ("fastsam-s", "FastSAM-s.pt", "FastSAM"),
    ("fastsam-x", "FastSAM-x.pt", "FastSAM"),
    ("mobilesam", "mobile_sam.pt", "SAM"),
    ("sam-b", "sam_b.pt", "SAM"),
    ("sam2-t", "sam2_t.pt", "SAM"),
    ("sam2-b", "sam2_b.pt", "SAM"),
    ("sam2.1-l", "sam2.1_l.pt", "SAM"),
]


def read_frame0(path: str) -> np.ndarray:
    import av
    c = av.open(path)
    for f in c.decode(video=0):
        return f.to_ndarray(format="rgb24")
    raise RuntimeError(f"no frames in {path}")


def main() -> None:
    from ultralytics import FastSAM, SAM  # noqa: F401
    import torch

    items = json.loads((DATA / "videos2caption.json").read_text())
    item = items[0]
    vpath = str(DATA / "clips" / item["path"]) if (DATA / "clips" / item["path"]).exists() \
        else str(DATA / "videos" / item["path"])
    frame0 = read_frame0(vpath)
    H, W = frame0.shape[:2]
    print(f"frame: {vpath}  {W}x{H}", flush=True)

    loaders = {"FastSAM": FastSAM, "SAM": SAM}
    rows = []
    for name, wf, cls in MODELS:
        wp = WEIGHTS / wf
        try:
            t0 = time.time()
            model = loaders[cls](str(wp) if wp.exists() else wf)
            # everything-mode: no prompts. imgsz 1024, retina full-res masks.
            res = model(frame0, device="cuda", retina_masks=True, imgsz=1024,
                        conf=0.4, iou=0.9, verbose=False)
            dt = time.time() - t0
            m = res[0].masks
            nm = 0 if m is None else int(m.data.shape[0])
            mh, mw = (0, 0) if m is None else tuple(m.data.shape[1:])
            print(f"[OK] {name:10s} weights={wf:14s} masks={nm:4d} maskres={mw}x{mh} "
                  f"time={dt:6.2f}s", flush=True)
            rows.append((name, nm, dt))
            # cache downloaded weight into WEIGHTS dir
            if not wp.exists() and Path(wf).exists():
                Path(wf).replace(wp)
            del model
            torch.cuda.empty_cache()
        except Exception as e:  # noqa: BLE001
            print(f"[FAIL] {name:10s} weights={wf:14s} -> {type(e).__name__}: {e}", flush=True)
    print("\nsummary (name, n_masks, sec):", flush=True)
    for r in rows:
        print(f"  {r[0]:10s} {r[1]:4d}  {r[2]:6.2f}", flush=True)


if __name__ == "__main__":
    main()
