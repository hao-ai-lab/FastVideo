#!/usr/bin/env python3
"""01 — Flow-match finetuning (Wan2.1-1.3B).

The simplest method: regress the student's velocity prediction toward the flow-match target on noised
latents. It drives no rollout (it trains on given/sampled latents), but it goes through the same
`TrainingMethod` seam every other method does: `train_step(batch, iteration) -> (loss_map, metrics)`.

Run:  python3 v2_examples/training/01_finetune.py
"""
from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from v2.models.wan21 import build_wan21_card
from v2.training.methods import build_finetune


def main() -> None:
    method = build_finetune(build_wan21_card(), lr=0.05)
    batch = {"prompts": ["a red sports car", "a sailboat at dawn"], "seeds": [1, 2]}

    print("method:", method.name, "| manages_optimization:", method.manages_optimization())
    first = last = None
    for it in range(20):
        loss, metrics = method.train_step(batch, it)
        first = first or metrics
        last = metrics
    print(f"loss        : {first['loss']:.4f} -> {last['loss']:.4f}  (decreasing = the student fits)")
    print(f"grad_norm   : {last['grad_norm/student']:.4f}  (the #1396-style per-method regression signal)")


if __name__ == "__main__":
    main()
