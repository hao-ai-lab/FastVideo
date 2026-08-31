#!/usr/bin/env python3
"""Load every production RVM reward and score deterministic synthetic videos."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from fastvideo.train.methods.rl.rewards import build_multi_reward_scorer


def make_media(frames: int = 17, height: int = 256, width: int = 448) -> torch.Tensor:
    x = torch.linspace(0, 255, width, dtype=torch.float32)[None, :].expand(height, -1)
    y = torch.linspace(0, 255, height, dtype=torch.float32)[:, None].expand(-1, width)
    base = torch.stack((x, y, (x + y) / 2), dim=0).to(torch.uint8)
    static = base[:, None].repeat(1, frames, 1, 1)
    moving_frames = []
    for frame_index in range(frames):
        frame = base.clone()
        left = int((width - 48) * frame_index / max(1, frames - 1))
        frame[:, height // 3 : height // 3 + 48, left : left + 48] = torch.tensor(
            [255, 32, 32], dtype=torch.uint8
        )[:, None, None]
        moving_frames.append(frame)
    moving = torch.stack(moving_frames, dim=1)
    return torch.stack((static, moving), dim=0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    scorer = build_multi_reward_scorer(
        {
            "backend": "genrl",
            "rewards": {
                "videoalign_ta": 1.5,
                "videoalign_mq": 1.0,
                "hpsv3_general": 0.1,
                "hpsv3_percentile": 0.1,
                "dynamic_tracking": 0.7,
            },
            "options": {
                "dynamic_tracking": {
                    "frame_pairs": 4,
                    "top_fraction": 0.05,
                    "resize_short_edge": 256,
                    "pretrained": True,
                }
            },
        },
        device=args.device,
    )
    media = make_media()
    prompts = [
        "A static abstract gradient poster.",
        "A red square moves smoothly from left to right over an abstract gradient.",
    ]
    rewards = scorer(media, prompts)
    serializable = {key: value.detach().float().cpu().tolist() for key, value in rewards.items()}
    for key, values in serializable.items():
        if not bool(torch.isfinite(torch.tensor(values)).all()):
            raise RuntimeError(f"Reward {key} returned a non-finite value: {values}")
    if serializable["dynamic_tracking"][1] <= serializable["dynamic_tracking"][0]:
        raise RuntimeError(
            "Dynamic tracking did not score the moving synthetic clip above the static clip; "
            f"scores={serializable['dynamic_tracking']}"
        )
    text = json.dumps(serializable, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
