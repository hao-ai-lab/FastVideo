#!/usr/bin/env python3
"""Load the pinned MJ-VIDEO model and score deterministic videos."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from fastvideo.train.methods.rl.rewards import build_multi_reward_scorer
from fastvideo.train.methods.rl.rewards.mj_video import (
    MJ_VIDEO_ASPECT_INDICES,
    MJ_VIDEO_BASE_MODEL_REVISION,
    MJ_VIDEO_MODEL_REVISION,
    MJ_VIDEO_SOURCE_REVISION,
)


def make_media(
    *,
    frames: int = 17,
    height: int = 256,
    width: int = 448,
) -> torch.Tensor:
    x = torch.linspace(
        0,
        255,
        width,
        dtype=torch.float32,
    )[None].expand(height, -1)
    y = torch.linspace(
        0,
        255,
        height,
        dtype=torch.float32,
    )[:, None].expand(-1, width)
    base = torch.stack(
        (x, y, (x + y) / 2),
        dim=0,
    ).to(torch.uint8)
    coherent_frames: list[torch.Tensor] = []
    inconsistent_frames: list[torch.Tensor] = []
    for frame_index in range(frames):
        left = int(
            (width - 64)
            * frame_index
            / max(1, frames - 1)
        )
        coherent = base.clone()
        coherent[
            :,
            height // 3 : height // 3 + 64,
            left : left + 64,
        ] = torch.tensor(
            [255, 32, 32],
            dtype=torch.uint8,
        )[:, None, None]
        coherent_frames.append(coherent)

        inconsistent = coherent.clone()
        if frame_index % 2:
            inconsistent = torch.flip(
                inconsistent,
                dims=(-1,),
            )
            inconsistent[0] = 255 - inconsistent[0]
        inconsistent_frames.append(inconsistent)
    return torch.stack(
        (
            torch.stack(coherent_frames, dim=1),
            torch.stack(inconsistent_frames, dim=1),
        ),
        dim=0,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    scorer = build_multi_reward_scorer(
        {
            "rewards": {
                "mjvideo_cc": 0.5,
                "mjvideo_fineness": 0.5,
            },
            "options": {
                "mjvideo_cc": {
                    "batch_size": 1,
                },
                "mjvideo_fineness": {
                    "batch_size": 1,
                },
            },
        },
        device=args.device,
    )
    media = make_media()
    prompts = [
        "A red square moves smoothly from left to right over a gradient.",
        "A red square moves from left to right while the scene flickers and changes inconsistently.",
    ]
    rewards = scorer(media, prompts)

    cc_scorer = scorer.scorers["mjvideo_cc"]
    fineness_scorer = scorer.scorers["mjvideo_fineness"]
    if cc_scorer.runtime is not fineness_scorer.runtime:
        raise RuntimeError(
            "MJ-VIDEO C&C and Fineness did not share one runtime"
        )
    expected_forwards = len(prompts)
    observed_forwards = int(cc_scorer.runtime.forward_calls)
    if observed_forwards != expected_forwards:
        raise RuntimeError(
            "Expected one MJ-VIDEO forward per batch-size-one video, "
            f"not one per aspect: expected={expected_forwards}, "
            f"observed={observed_forwards}"
        )

    serializable = {
        key: value.detach().float().cpu().tolist()
        for key, value in rewards.items()
    }
    for key, values in serializable.items():
        if not bool(
            torch.isfinite(
                torch.tensor(values)
            ).all()
        ):
            raise RuntimeError(
                f"Reward {key} returned non-finite values: {values}"
            )
    cc = torch.tensor(serializable["mjvideo_cc"])
    fineness = torch.tensor(serializable["mjvideo_fineness"])
    if torch.equal(cc, fineness):
        raise RuntimeError(
            "MJ-VIDEO C&C and Fineness returned identical tensors; "
            "verify aspect indices and checkpoint loading"
        )

    payload = {
        "source_revision": MJ_VIDEO_SOURCE_REVISION,
        "model_revision": MJ_VIDEO_MODEL_REVISION,
        "base_model_revision": MJ_VIDEO_BASE_MODEL_REVISION,
        "aspect_indices": {
            "fineness": MJ_VIDEO_ASPECT_INDICES["fineness"],
            "cc": MJ_VIDEO_ASPECT_INDICES["cc"],
        },
        "forward_calls": observed_forwards,
        "rewards": serializable,
    }
    text = json.dumps(
        payload,
        indent=2,
        sort_keys=True,
    ) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        args.output.write_text(
            text,
            encoding="utf-8",
        )
    print(text, end="")


if __name__ == "__main__":
    main()
