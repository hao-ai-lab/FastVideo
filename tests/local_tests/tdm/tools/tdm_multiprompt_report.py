#!/usr/bin/env python
"""Reference-free multi-prompt diversity report for TDM validation videos.

Phase 3 acceptance only had one prompt, so it could show coherence and
sharpness but not whether the student recovers sample diversity. This tool
covers the missing axis on the modular validation layout.

The validation callback shards the caption list across sequence-parallel
groups and pads the list to a multiple of that degree (see
``fastvideo/dataset/validation_dataset.py``); rank ``r`` therefore saves the
``video_i`` sample of the padded row ``r * rows_per_group + i``. The tool
rebuilds that mapping from the same caption file, then reports, per
validation step and per caption:

- frame statistics (brightness/contrast/gradient sharpness), and
- a within-caption spread over the repeated seeds, plus the cross-caption
  spread and their ratio.

Spread is the mean pairwise L2 distance between per-video descriptors: the
temporal-mean frame pooled to ``--descriptor-size`` squared and flattened.
It is reference-free, matching the Phase 2 measurement policy that never
asserts paired metrics. A regression-only (conditional-mean) student
tightens within-caption spread toward zero, while a distribution-matched
student keeps the seeds apart; the cross-caption spread is the control for
"the samples merely differ by prompt".

Example (on a GPU pod, from the repo root):

    PYTHONPATH=. python tests/local_tests/tdm/tools/tdm_multiprompt_report.py \
        --run-dir /workspace/run/tdm-port/wan-tdm-multiprompt/treatment/output \
        --captions-json examples/train/configs/distribution_matching/wan/tdm_multiprompt_validation.json \
        --out /tmp/multiprompt-treatment.json
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from fastvideo.train.utils.tdm_metrics import frame_statistics

VALIDATION_PATTERN = re.compile(r"validation_step_(\d+)_inference_steps_(\d+)_rank_(\d+)_video_(\d+)\.mp4$")


def _load_video(path: Path) -> np.ndarray:
    import av

    with av.open(str(path)) as container:
        frames = [frame.to_ndarray(format="rgb24") for frame in container.decode(video=0)]
    if not frames:
        raise ValueError(f"no decodable frames in {path}")
    return np.stack(frames)


def _descriptor(frames: np.ndarray, size: int) -> torch.Tensor:
    """Temporal-mean frame pooled to ``size`` squared, flattened and L2-normalized."""
    tensor = torch.as_tensor(frames).float()
    if tensor.ndim == 4:
        tensor = tensor.mean(dim=0)
    chw = tensor.permute(2, 0, 1).unsqueeze(0)
    pooled = torch.nn.functional.adaptive_avg_pool2d(chw, size).flatten()
    return pooled / pooled.norm().clamp_min(1e-6)


def _mean_pairwise(vectors: list[torch.Tensor]) -> float:
    if len(vectors) < 2:
        return float("nan")
    stacked = torch.stack(vectors)
    distances = torch.cdist(stacked, stacked)
    upper = torch.triu_indices(len(vectors), len(vectors), offset=1)
    return float(distances[upper[0], upper[1]].mean())


def _load_caption_rows(path: Path) -> list[str]:
    payload = json.loads(path.read_text())
    rows = [entry["caption"] for entry in payload["data"]]
    if not rows:
        raise ValueError(f"no captions in {path}")
    return rows


def _padded_rows(rows: list[str], num_groups: int) -> list[str]:
    remainder = len(rows) % num_groups
    if remainder == 0:
        return rows
    return rows + rows[: num_groups - remainder]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--captions-json", required=True)
    parser.add_argument("--num-sp-groups", type=int, default=4)
    parser.add_argument("--descriptor-size", type=int, default=32)
    parser.add_argument("--label", default="")
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.is_dir():
        raise FileNotFoundError(run_dir)

    rows = _padded_rows(_load_caption_rows(Path(args.captions_json)), args.num_sp_groups)
    rows_per_group = len(rows) // args.num_sp_groups

    per_step: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for path in sorted(run_dir.glob("**/*.mp4")):
        match = VALIDATION_PATTERN.search(path.name)
        if match is None:
            continue
        step = int(match.group(1))
        rank = int(match.group(3))
        video_index = int(match.group(4))
        row = rank * rows_per_group + video_index
        if row >= len(rows):
            raise ValueError(f"{path.name}: row {row} outside {len(rows)} padded rows")
        per_step[step].append({
            "caption": rows[row],
            "rank": rank,
            "video_index": video_index,
            "path": path,
        })

    if not per_step:
        raise ValueError(f"no validation videos matched under {run_dir}")

    report: dict[str, Any] = {
        "run_dir": str(run_dir),
        "label": args.label,
        "captions_json": str(args.captions_json),
        "num_sp_groups": args.num_sp_groups,
        "rows_per_group": rows_per_group,
        "steps": {},
    }
    train_captions = set(rows[:4])
    train_within: list[float] = []
    heldout_within: list[float] = []
    train_sharp: list[float] = []
    heldout_sharp: list[float] = []

    for step in sorted(per_step):
        entries = per_step[step]
        by_caption: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for entry in entries:
            by_caption[entry["caption"]].append(entry)

        caption_records: dict[str, dict[str, float]] = {}
        descriptors: dict[str, list[torch.Tensor]] = {}
        step_train_sharp: list[float] = []
        step_heldout_sharp: list[float] = []
        for caption, group in by_caption.items():
            vectors = []
            sharpness = []
            std = []
            for entry in group:
                frames = _load_video(entry["path"])
                stats = frame_statistics(frames)
                sharpness.append(stats["sharpness"])
                std.append(stats["std"])
                vectors.append(_descriptor(frames, args.descriptor_size))
            descriptors[caption] = vectors
            within = _mean_pairwise(vectors)
            caption_records[caption] = {
                "seeds": float(len(group)),
                "sharpness": float(np.mean(sharpness)),
                "std": float(np.mean(std)),
                "within_spread": within,
            }
            if caption in train_captions:
                train_within.append(within)
                train_sharp.append(float(np.mean(sharpness)))
                step_train_sharp.append(float(np.mean(sharpness)))
            else:
                heldout_within.append(within)
                heldout_sharp.append(float(np.mean(sharpness)))
                step_heldout_sharp.append(float(np.mean(sharpness)))

        cross_vectors: list[tuple[str, torch.Tensor]] = []
        for caption, vectors in descriptors.items():
            for vector in vectors:
                cross_vectors.append((caption, vector))
        cross_pairs = []
        for i in range(len(cross_vectors)):
            for j in range(i + 1, len(cross_vectors)):
                if cross_vectors[i][0] != cross_vectors[j][0]:
                    cross_pairs.append(float((cross_vectors[i][1] - cross_vectors[j][1]).norm()))
        cross_spread = float(np.mean(cross_pairs)) if cross_pairs else float("nan")
        within_all = [record["within_spread"] for record in caption_records.values()]
        mean_within = float(np.mean(within_all))
        report["steps"][str(step)] = {
            "captions": caption_records,
            "mean_within_spread": mean_within,
            "cross_spread": cross_spread,
            "spread_ratio": (mean_within / cross_spread) if cross_spread else float("nan"),
        }
        print(f"step={step:06d} mean_within={mean_within:.5f} cross={cross_spread:.5f} "
              f"ratio={(mean_within / cross_spread):.4f} "
              f"train_sharp={np.mean(step_train_sharp):.2f} "
              f"heldout_sharp={np.mean(step_heldout_sharp):.2f}")

    report["summary"] = {
        "train_mean_within_spread": float(np.mean(train_within)),
        "heldout_mean_within_spread": float(np.mean(heldout_within)),
        "train_mean_sharpness": float(np.mean(train_sharp)),
        "heldout_mean_sharpness": float(np.mean(heldout_sharp)),
    }
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(f"MULTIPROMPT_REPORT {args.out}")


if __name__ == "__main__":
    main()
