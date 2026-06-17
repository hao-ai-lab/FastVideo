#!/usr/bin/env python3
"""04 — Self-forcing: causal video distillation (Wan-causal).

"Train against your own KV-cached causal rollout." Self-forcing extends DMD2, but the student rollout
is the SHARED `chunk_rollout` loop (with the slab-KV cache) — exactly the causal loop the engine
*streams* at serve time. So the causal student is distilled under the runtime it will be served on
(the (recipe, runtime) pair). The loss is DMD2's distribution-matching, applied to the rollout latents.

Run:  python3 v2_examples/training/04_self_forcing.py
"""
from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from v2.models.wan_causal import build_wan_causal_card
from v2.training.methods import build_self_forcing


def main() -> None:
    method = build_self_forcing(build_wan_causal_card())
    batch = {"prompts": ["a drone flight over dunes", "a river through a forest"], "seeds": [1, 2]}

    print("method:", method.name, "| student_loop_id:", method.student_loop_id, "(the causal, KV-cached loop)")
    last = None
    for it in range(12):
        loss, last = method.train_step(batch, it)
    print(f"dmd_loss      : {last['dmd_loss']:.4f}   critic_loss : {last['critic_loss']:.4f}")
    print(f"grad_norm     : student={last.get('grad_norm/student', 0.0):.4f}  critic={last['grad_norm/critic']:.4f}")
    print("\nThe student trains on its OWN causal rollout (chunk_rollout + slab-KV) — same loop the "
          "engine streams.")


if __name__ == "__main__":
    main()
