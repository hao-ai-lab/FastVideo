#!/usr/bin/env python
"""Reference-free acceptance report for TDM validation videos.

Scans a run directory's validation samples (``*/samples/step-*/``) and
reports the reference-free signals that TDM acceptance uses - frame
brightness/contrast and gradient sharpness - plus the optional forensic
paired MS-SSIM between student and teacher videos of the same sample.
Paired MS-SSIM is recorded, never asserted: it rewards blur and penalizes
the paired drift that distribution matching produces by design.

Example (on a GPU pod, from the repo root):

    PYTHONPATH=. python tests/local_tests/tdm/tools/tdm_video_report.py \
        --run-dir /workspace/run/tdm-standalone/wan-ladder \
        --out /tmp/video_report.json
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
from fastvideo.train.utils.tdm_metrics import frame_statistics, paired_ms_ssim

VIDEO_PATTERN = re.compile(r"prompt-(\d+)-(student|teacher)-(\d+)step\.mp4$")


def _load_video(path: Path) -> np.ndarray:
    import av

    with av.open(str(path)) as container:
        frames = [frame.to_ndarray(format="rgb24") for frame in container.decode(video=0)]
    if not frames:
        raise ValueError(f"no decodable frames in {path}")
    return np.stack(frames)


def _iter_videos(run_dir: Path) -> list[tuple[int, int, str, Path]]:
    found: list[tuple[int, int, str, Path]] = []
    for path in sorted(run_dir.glob("**/samples/step-*/*.mp4")):
        match = VIDEO_PATTERN.search(path.name)
        if match is None:
            continue
        prompt, role, steps = match.group(1), match.group(2), int(match.group(3))
        step_dir = int(path.parent.name.split("-")[1])
        found.append((step_dir, int(prompt), role, path))
    return found


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--out", default="")
    parser.add_argument("--paired-ms-ssim", action="store_true", help="also compute forensic paired MS-SSIM")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.is_dir():
        raise FileNotFoundError(run_dir)

    videos = _iter_videos(run_dir)
    if not videos:
        raise ValueError(f"no validation videos matched under {run_dir}")

    report: dict[str, Any] = {"run_dir": str(run_dir), "videos": {}, "paired_ms_ssim": {}}
    for step_dir, prompt, role, path in videos:
        stats = frame_statistics(_load_video(path))
        key = f"{path.parent.relative_to(run_dir)}/{path.name}"
        report["videos"][key] = {
            "step": step_dir,
            "prompt": prompt,
            "role": role,
            "stats": stats,
        }
        print(f"step={step_dir:06d} prompt={prompt} {role:8s} "
              f"std={stats['std']:7.2f} sharpness={stats['sharpness']:8.2f} frames={int(stats['frames'])}")

    if args.paired_ms_ssim:
        by_key = {(step, prompt, role): path for step, prompt, role, path in videos}
        for (step, prompt, role), path in sorted(by_key.items()):
            if role != "student":
                continue
            teacher = by_key.get((step, prompt, "teacher"))
            if teacher is None:
                continue
            try:
                paired = paired_ms_ssim(_load_video(path), _load_video(teacher))
            except RuntimeError as error:  # optional dependency missing
                print(f"paired MS-SSIM skipped: {error}")
                break
            key = f"step-{step:06d}/prompt-{prompt}"
            report["paired_ms_ssim"][key] = paired
            print(f"step={step:06d} prompt={prompt} paired_ms_ssim={paired['ms_ssim']:.4f} (forensic only)")

    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(f"VIDEO_REPORT {args.out}")


if __name__ == "__main__":
    main()
