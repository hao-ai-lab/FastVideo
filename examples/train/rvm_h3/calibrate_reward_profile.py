#!/usr/bin/env python3
"""Build fixed robust reward calibration from released FastH3 videos."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
from typing import Any

import av
import numpy as np
import torch

from fastvideo.train.methods.rl.common import (
    visual_text_from_h3_prompt,
)
from fastvideo.train.methods.rl.rewards import (
    build_multi_reward_scorer,
)
from fastvideo.train.methods.rl.rewards.mj_video import (
    MJ_VIDEO_BASE_MODEL_REVISION,
    MJ_VIDEO_MODEL_REVISION,
    MJ_VIDEO_SOURCE_REVISION,
)


_COMPONENTS = (
    "videoalign_ta",
    "mjvideo_cc",
    "mjvideo_fineness",
    "dynamic_tracking",
)
_PROMPT_PATTERN = re.compile(r"prompt-(\d+)\.mp4$")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(block)
    return digest.hexdigest()


def _git_head() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _read_video(path: Path) -> torch.Tensor:
    frames: list[np.ndarray] = []
    with av.open(str(path)) as container:
        for frame in container.decode(video=0):
            frames.append(
                frame.to_ndarray(format="rgb24")
            )
    if not frames:
        raise RuntimeError(
            f"No video frames decoded from {path}"
        )
    array = np.stack(frames)
    return (
        torch.from_numpy(array)
        .permute(3, 0, 1, 2)
        .contiguous()[None]
    )


def _read_prompts(path: Path) -> list[str]:
    prompts = [
        line.strip()
        for line in path.read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    if not prompts or any(not prompt for prompt in prompts):
        raise ValueError(
            f"Prompt file contains empty or no prompts: {path}"
        )
    return prompts


def _discover_inputs(
    video_dir: Path,
    prompts: list[str],
    *,
    max_videos: int,
) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for path in sorted(video_dir.glob("prompt-*.mp4")):
        match = _PROMPT_PATTERN.search(path.name)
        if match is None:
            continue
        index = int(match.group(1))
        if index >= len(prompts):
            raise IndexError(
                f"Video index {index} exceeds prompt file "
                f"length {len(prompts)}"
            )
        entries.append(
            {
                "index": index,
                "path": path,
                "prompt": prompts[index],
            }
        )
        if len(entries) >= max_videos:
            break
    if not entries:
        raise FileNotFoundError(
            f"No prompt-*.mp4 videos found under {video_dir}"
        )
    return entries


def _robust_component_stats(
    values: list[float],
    *,
    eps: float,
    constant_scale_fallback: float | None,
) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    if not np.isfinite(array).all():
        raise ValueError(
            "Calibration values contain NaN or Inf"
        )
    median = float(np.median(array))
    mad = float(np.median(np.abs(array - median)))
    std = float(np.std(array))
    robust_scale = 1.4826 * mad
    method = "median_mad"
    scale = robust_scale
    if scale <= eps:
        scale = std
        method = "std_fallback"
    if scale <= eps:
        if (
            constant_scale_fallback is None
            or not math.isfinite(constant_scale_fallback)
            or constant_scale_fallback <= eps
        ):
            raise ValueError(
                "Reward component is constant on the calibration bank. "
                "Increase bank diversity or pass an explicit positive "
                "--constant-scale-fallback after auditing the component."
            )
        scale = float(constant_scale_fallback)
        method = "constant_override"
    quantiles = np.quantile(
        array,
        [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99],
    )
    return {
        "center": median,
        "scale": scale,
        "count": int(array.size),
        "method": method,
        "mean": float(array.mean()),
        "std": std,
        "median": median,
        "mad": mad,
        "robust_scale": robust_scale,
        "min": float(array.min()),
        "max": float(array.max()),
        "q01": float(quantiles[0]),
        "q05": float(quantiles[1]),
        "q25": float(quantiles[2]),
        "q50": float(quantiles[3]),
        "q75": float(quantiles[4]),
        "q95": float(quantiles[5]),
        "q99": float(quantiles[6]),
    }


def _build_raw_scorer(
    *,
    device: str,
    batch_size: int,
):
    return build_multi_reward_scorer(
        {
            "rewards": {
                "videoalign_ta": 1.0,
                "mjvideo_cc": 1.0,
                "mjvideo_fineness": 1.0,
                "dynamic_tracking": 1.0,
            },
            "options": {
                "mjvideo_cc": {
                    "batch_size": batch_size,
                },
                "mjvideo_fineness": {
                    "batch_size": batch_size,
                },
                "dynamic_tracking": {
                    "frame_pairs": 4,
                    "top_fraction": 0.05,
                    "resize_short_edge": 256,
                    "pretrained": True,
                },
            },
        },
        device=device,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
    )
    parser.add_argument(
        "--video-dir",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--prompt-file",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--scores-output",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--device",
        default="cuda",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--max-videos",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=1e-6,
    )
    parser.add_argument(
        "--constant-scale-fallback",
        type=float,
        default=None,
    )
    args = parser.parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.max_videos <= 1:
        raise ValueError("--max-videos must be greater than one")
    if args.eps <= 0.0:
        raise ValueError("--eps must be positive")

    prompts = _read_prompts(args.prompt_file)
    entries = _discover_inputs(
        args.video_dir,
        prompts,
        max_videos=args.max_videos,
    )
    scorer = _build_raw_scorer(
        device=args.device,
        batch_size=args.batch_size,
    )

    component_values: dict[str, list[float]] = {
        name: []
        for name in _COMPONENTS
    }
    records: list[dict[str, Any]] = []
    input_digest = hashlib.sha256()
    for start in range(
        0,
        len(entries),
        args.batch_size,
    ):
        chunk = entries[
            start : start + args.batch_size
        ]
        media = torch.cat(
            [
                _read_video(item["path"])
                for item in chunk
            ],
            dim=0,
        )
        visual_prompts = [
            visual_text_from_h3_prompt(
                item["prompt"]
            )
            for item in chunk
        ]
        scores = scorer(
            media,
            visual_prompts,
        )
        for offset, item in enumerate(chunk):
            video_sha = _sha256_file(
                item["path"]
            )
            input_digest.update(
                (
                    f"{item['index']}\t"
                    f"{item['prompt']}\t"
                    f"{video_sha}\n"
                ).encode("utf-8")
            )
            record_scores = {
                name: float(
                    scores[name][offset]
                    .detach()
                    .float()
                    .cpu()
                )
                for name in _COMPONENTS
            }
            for name, value in record_scores.items():
                component_values[name].append(value)
            records.append(
                {
                    "index": int(item["index"]),
                    "path": str(item["path"]),
                    "video_sha256": video_sha,
                    "prompt": item["prompt"],
                    "visual_prompt": visual_prompts[offset],
                    "scores": record_scores,
                }
            )

    calibration = {
        "schema_version": 1,
        "profile": "physion_mj_v1",
        "created_at": datetime.now(
            timezone.utc
        ).isoformat(),
        "components": {
            name: _robust_component_stats(
                values,
                eps=args.eps,
                constant_scale_fallback=(
                    args.constant_scale_fallback
                ),
            )
            for name, values in component_values.items()
        },
        "provenance": {
            "fastvideo_git_head": _git_head(),
            "video_dir": str(
                args.video_dir.resolve()
            ),
            "prompt_file": str(
                args.prompt_file.resolve()
            ),
            "prompt_file_sha256": _sha256_file(
                args.prompt_file
            ),
            "input_digest_sha256": (
                input_digest.hexdigest()
            ),
            "num_videos": len(records),
            "mj_video_source_revision": (
                MJ_VIDEO_SOURCE_REVISION
            ),
            "mj_video_model_revision": (
                MJ_VIDEO_MODEL_REVISION
            ),
            "mj_video_base_model_revision": (
                MJ_VIDEO_BASE_MODEL_REVISION
            ),
            "mj_video_num_segments": 8,
            "mj_video_input_size": 448,
            "dynamic_tracking": {
                "frame_pairs": 4,
                "top_fraction": 0.05,
                "resize_short_edge": 256,
                "pretrained": True,
            },
        },
    }

    scores_output = (
        args.scores_output
        or args.output.with_suffix(".scores.jsonl")
    )
    scores_output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    scores_output.write_text(
        "".join(
            json.dumps(
                record,
                sort_keys=True,
            )
            + "\n"
            for record in records
        ),
        encoding="utf-8",
    )
    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    args.output.write_text(
        json.dumps(
            calibration,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "calibration": str(args.output),
                "scores": str(scores_output),
                "num_videos": len(records),
                "components": calibration["components"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
