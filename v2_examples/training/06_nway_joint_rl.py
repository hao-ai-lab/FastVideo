#!/usr/bin/env python3
"""06 — N-way joint RL over arbitrary experts.

Generalizes joint RL from 2 experts to **N refiner LMs + a generator**, from one reward. The substrate
was already N-ready (a card holds N components/loops; WeightSyncPlan is per-component; grad-targets are
a dict) — only the method body loops over a list. Two credit modes:

  * `per_expert` — each refiner gets an advantage from its own reward term ⇒ all N learn cleanly.
  * `shared` — one reward to everyone (faithful to UniRL) ⇒ works but noisier (multi-agent variance).

Run:  python3 v2_examples/training/06_nway_joint_rl.py
"""
from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from v2._vendor.models.multi_expert import build_multi_expert_card, refiner_ids
from v2.training.methods import build_joint_multi_rl


def _run(credit: str, n: int = 3, iters: int = 60):
    rids = refiner_ids(n)
    targets = {rid: (i * 2 + 1) % 6 for i, rid in enumerate(rids)}
    m = build_joint_multi_rl(build_multi_expert_card(n_refiners=n), refiner_ids=rids,
                             target_actions=targets, joint_generator=True, credit=credit)
    p0 = {r: float(m.refiner(r)._probs()[targets[r]]) for r in rids}
    batch = {"prompts": ["a fox", "blue sky", "a city at night"], "seeds": [1, 2, 3]}
    last = None
    for it in range(iters):
        _, last = m.train_step(batch, it)
    return m, rids, targets, p0, last


def main() -> None:
    for credit in ("per_expert", "shared"):
        m, rids, targets, p0, last = _run(credit)
        print(f"== {int(last['n_experts'])} experts ({len(rids)} refiners + generator), credit={credit} ==")
        for r in rids:
            print(f"  {r} P(target): {p0[r]:.3f} -> {float(m.refiner(r)._probs()[targets[r]]):.3f}")
        vers = [last[f"weights_version/{r}"] for r in rids] + [last["weights_version/transformer"]]
        print(f"  generator grad_norm={last['grad_norm/transformer']:.3f}  "
              f"distinct expert versions: {len(set(vers))}/{len(vers)}\n")


if __name__ == "__main__":
    main()
