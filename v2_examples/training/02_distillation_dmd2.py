#!/usr/bin/env python3
"""02 — DMD2 distillation (Wan2.1-1.3B).

Distribution-matching distillation: three roles on the same card — a **student** (the few-step model
being trained), a trainable **fake-score critic**, and a frozen **teacher**. The generator loss is the
DMD score difference (teacher − critic) on the student's own rollout latents; the critic is trained to
score the student. Two grad streams (student + critic) ⇒ two grad-norm signals.

Run:  python3 v2_examples/training/02_distillation_dmd2.py
"""
from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from v2.models.wan21 import build_wan21_card
from v2.training.methods import build_dmd2


def main() -> None:
    method = build_dmd2(build_wan21_card(), rollout_steps=3)
    batch = {"prompts": ["a waterfall in a canyon", "a lantern festival"], "seeds": [1, 2]}

    print("method:", method.name, "| roles: student + fake-score critic + frozen teacher")
    last = None
    for it in range(15):
        loss, last = method.train_step(batch, it)
    print(f"dmd_loss      : {last['dmd_loss']:.4f}      (generator: teacher − critic score difference)")
    print(f"critic_loss   : {last['critic_loss']:.4f}   (critic learns to score the student's samples)")
    print(f"grad_norm     : student={last.get('grad_norm/student', 0.0):.4f}  "
          f"critic={last['grad_norm/critic']:.4f}")


if __name__ == "__main__":
    main()
