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
VALIDATION_PATTERN = re.compile(r"validation_step_(\d+)_inference_steps_(\d+)_rank_(\d+)_video_(\d+)\.mp4$")


def _load_video(path: Path) -> np.ndarray:
    import av

    with av.open(str(path)) as container:
        frames = [frame.to_ndarray(format="rgb24") for frame in container.decode(video=0)]
    if not frames:
        raise ValueError(f"no decodable frames in {path}")
    return np.stack(frames)


def _iter_videos(run_dir: Path) -> list[dict[str, Any]]:
    """Match both validation layouts: the standalone ``samples/step-*``
    tree and the modular trainer's flat ``validation_step_*`` files."""
    found: list[dict[str, Any]] = []
    for path in sorted(run_dir.glob("**/*.mp4")):
        name = path.name
        match = VIDEO_PATTERN.search(name)
        if match is not None:
            found.append({
                "step": int(path.parent.name.split("-")[1]),
                "prompt": int(match.group(1)),
                "role": match.group(2),
                "path": path,
                "layout": "samples",
                "sampling_steps": int(match.group(3)),
            })
            continue
        match = VALIDATION_PATTERN.search(name)
        if match is not None:
            found.append({
                "step": int(match.group(1)),
                "prompt": int(match.group(4)),
                "role": "student",
                "path": path,
                "layout": "validation",
                "rank": int(match.group(3)),
                "sampling_steps": int(match.group(2)),
            })
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
    for entry in videos:
        path = entry["path"]
        stats = frame_statistics(_load_video(path))
        key = f"{path.parent.relative_to(run_dir)}/{path.name}"
        record: dict[str, Any] = {
            "step": entry["step"],
            "prompt": entry["prompt"],
            "role": entry["role"],
            "sampling_steps": entry["sampling_steps"],
            "layout": entry["layout"],
            "stats": stats,
        }
        if "rank" in entry:
            record["rank"] = entry["rank"]
        report["videos"][key] = record
        print(f"step={entry['step']:06d} prompt={entry['prompt']} {entry['role']:8s} "
              f"std={stats['std']:7.2f} sharpness={stats['sharpness']:8.2f} frames={int(stats['frames'])}")

    if args.paired_ms_ssim:
        by_key = {(entry["step"], entry["prompt"], entry["role"]): entry["path"] for entry in videos}
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
