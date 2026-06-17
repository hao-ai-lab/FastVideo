#!/usr/bin/env python3
"""05 — Joint LM + generator RL (UniRL/PromptRL).

One reward, two trainable experts: an `llm` prompt-refiner and a `transformer` flow generator. The
reward → a group-relative advantage → BOTH a **token policy gradient** on the LM and a **FlowGRPO PPO**
on the DiT, with two independent WeightSyncPlans. Likelihood-**based** C2: the per-step log-prob
identity (recomputed logp == rollout logp ⇒ PPO ratio ≈ 1). `joint=False` is PromptRL's prompt-only
mode (generator frozen; only the LM learns).

Run:  python3 v2_examples/training/05_joint_lm_generator_rl.py
"""
from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from v2.models.unified import build_unified_card
from v2.training.methods import build_unified_rl

TARGET = 3


def _run(joint: bool, iters: int = 40):
    m = build_unified_rl(build_unified_card(), joint=joint, target_action=TARGET,
                         num_samples_per_prompt=4, num_skip_refinement=1)
    batch = {"prompts": ["a fox", "blue sky", "a city at night"], "seeds": [1, 2, 3]}
    p0 = float(m.llm._probs()[TARGET])
    last = None
    for it in range(iters):
        _, last = m.train_step(batch, it)
    return m, p0, last


def main() -> None:
    print("== joint (LM + generator both trained) ==")
    m, p0, last = _run(joint=True)
    print(f"  LM P(target action): {p0:.3f} -> {last['refine_target_prob']:.3f}  (the LM learns to refine)")
    print(f"  lm_pg_loss={last['lm_pg_loss']:.4f}  dit_pg_loss={last['dit_pg_loss']:.4f}  "
          f"ppo_ratio={last['ppo_ratio_mean']:.4f}")
    print(f"  grad_norm: llm={last['grad_norm/llm']:.4f}  transformer={last['grad_norm/transformer']:.4f}")
    print(f"  independent versions: llm={last['llm_weights_version']} dit={last['transformer_weights_version']}")

    print("\n== prompt-only (generator frozen; PromptRL ablation) ==")
    _, p0b, lastb = _run(joint=False)
    print(f"  LM P(target action): {p0b:.3f} -> {lastb['refine_target_prob']:.3f}")
    print(f"  generator version  : {lastb['transformer_weights_version']}  (frozen; no dit_pg_loss key: "
          f"{'dit_pg_loss' not in lastb})")


if __name__ == "__main__":
    main()
