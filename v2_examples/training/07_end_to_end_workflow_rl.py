#!/usr/bin/env python3
"""07 — End-to-end RL over a cross-model workflow (T2I → I2V).

Train BOTH stages of the T2I→I2V workflow (`flux-t2i` and `wan-i2v` — two *separate* models) from ONE
final-video reward. The rollout spans both instances with SDE capture; the same final-video advantage
drives a FlowGRPO PPO update on each stage's transformer, with two WeightSyncPlans on two instances.

The point: the **earlier model (T2I) is trained by a reward on the final video** — end-to-end credit
across a model boundary. So "rollout == serve + capture" holds for a *workflow*, not just one card.

Run:  python3 v2_examples/training/07_end_to_end_workflow_rl.py
"""
from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import numpy as np

from v2._vendor.models import build_flux_t2i_card, build_wan_i2v_card
from v2.training.methods import build_workflow_rl


def main() -> None:
    m = build_workflow_rl(build_flux_t2i_card(), build_wan_i2v_card(),
                          num_samples_per_prompt=4, rollout_steps=4, t2i_lr=0.03, i2v_lr=0.03)
    t2i_w0 = m.t2i.component("transformer").w_x.copy()
    i2v_w0 = m.i2v.component("transformer").w_x.copy()

    batch = {"prompts": ["a fox", "blue sky", "a city"], "seeds": [1, 2, 3]}
    last = None
    for it in range(20):
        _, last = m.train_step(batch, it)

    moved_t2i = not np.array_equal(t2i_w0, m.t2i.component("transformer").w_x)
    moved_i2v = not np.array_equal(i2v_w0, m.i2v.component("transformer").w_x)
    print("method:", m.name, "| two separate models trained from one final-video reward")
    print(f"  reward_mean        : {last['reward_mean']:.4f}")
    print(f"  T2I generator moved: {moved_t2i}   <- trained by a reward on the FINAL video (end-to-end credit)")
    print(f"  I2V generator moved: {moved_i2v}")
    print(f"  grad_norm          : t2i={last['grad_norm/t2i']:.3f}  i2v={last['grad_norm/i2v']:.3f}  "
          f"ppo_ratio={last['ppo_ratio_mean']:.4f}")
    print(f"  independent versions: t2i={last['t2i_weights_version']} i2v={last['i2v_weights_version']}")


if __name__ == "__main__":
    main()
