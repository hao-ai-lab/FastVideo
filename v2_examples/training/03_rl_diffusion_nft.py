#!/usr/bin/env python3
"""03 — DiffusionNFT: likelihood-free RL (Wan2.1-1.3B).

The landed RL method, and the reason for the C2 *split*: it is **likelihood-free** — no log-probs to
match — so consistency is the C2 *behavioral* rung (seeded sample + prediction-space identity). Key
properties this shows:

  * **Rollout == serve** — K samples of a prompt are produced by the SAME denoise loop the engine
    serves (in the ROLLOUT profile), captured as behavior.
  * **Samples from the *old* (decay-blended) policy**, not the student (the WeightSyncPlan carries that).
  * **Group-relative advantages** drive the update; the shared prompt encodes ONCE per K-sample group
    via the feature cache (the "24×" text-encode reduction).

Run:  python3 v2_examples/training/03_rl_diffusion_nft.py
"""
from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from v2._vendor.models.wan21 import build_wan21_card
from v2.training.methods import build_diffusion_nft


def main() -> None:
    nft = build_diffusion_nft(build_wan21_card(), num_video_per_prompt=4, num_inner_timesteps=2)
    batch = {"prompts": ["a red car", "a blue boat"], "seeds": [1, 2]}

    print("method:", nft.name, "| consistency:", nft.consistency_level().value, "(likelihood-free)")
    print("| samples from role:", nft.old_sync.role.value, "(NOT the student)")
    loss, m = nft.train_step(batch, 0)
    print(f"policy_loss   : {loss['policy_loss']:.3f}   kl_div_loss : {loss['kl_div_loss']:.5f}")
    print(f"reward_mean   : {m['reward_mean']:.3f}   advantage_std : {m['advantage_std']:.3f}")
    print(f"old_decay     : {m['old_decay']:.3f}   old_weights_version : {m['old_weights_version']}")

    fc = nft.old.caches.stats()["feature"]
    print(f"feature cache : {fc['hits']} hits / {fc['misses']} misses "
          f"(each K-sample group encodes its prompt once)")


if __name__ == "__main__":
    main()
