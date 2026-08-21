# SPDX-License-Identifier: Apache-2.0
"""One-GPU live VideoScore2 inference and parsing smoke."""

from __future__ import annotations

import argparse

import torch

from fastvideo.train.methods.rl.promptrl.rewards.service import (
    VideoScore2Judge,
)
from fastvideo.train.methods.rl.promptrl.video_io import encode_video_bytes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="TIGER-Lab/VideoScore2")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--adversarial",
        action="store_true",
        help="Use a flat yellow video mismatched to the prompt.",
    )
    args = parser.parse_args()

    # [B, C, T, H, W], with real temporal variation so the decoder and judge
    # both exercise a proper short video rather than a still-image edge case.
    if args.adversarial:
        video = torch.zeros((1, 3, 5, 64, 64))
        video[:, 0] = 0.8
        video[:, 1] = 0.8
        video[:, 2] = 0.1
        prompt = "a red cube rolling on a table"
    else:
        frames = torch.linspace(0.0, 1.0, 5).view(1, 1, 5, 1, 1)
        video = frames.expand(1, 3, 5, 64, 64).contiguous()
        prompt = "a gray screen gradually becoming white"
    encoded = encode_video_bytes(video, fps=4)
    judge = VideoScore2Judge(args.model_id, device=args.device)
    scores = judge.score_batch(
        [encoded],
        [prompt],
        fps=4,
    )
    if len(scores) != 1 or not all(
        key in scores[0]
        for key in (
            "composite",
            "visual_quality",
            "text_alignment",
            "physical_consistency",
        )
    ):
        raise RuntimeError(f"unexpected VideoScore2 result: {scores!r}")
    print(f"PROMPTRL_VIDEOSCORE2_SMOKE_OK scores={scores}", flush=True)


if __name__ == "__main__":
    main()
