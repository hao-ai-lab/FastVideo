"""The (recipe, runtime) flywheel — RL → distill → a faster card (design_v3 §16; §9.18).

design_v3's central product claim: *preference/reward data → RL → a faster distilled card → a better
product*, with the recipe carrying provenance so the chain is auditable. This wires it end-to-end on the
shared loops:

  1. **RL** the base model (DiffusionNFT) → improved policy weights.
  2. **Distill** from the RL'd model: a DMD2 student with the **RL'd model as its teacher** learns to
     match it in fewer steps.
  3. The distilled card's ``RecipeSpec.parents`` records the chain ``base → rl → distilled`` — the
     (recipe, runtime) provenance that makes "this fast model came from that RL run" a typed fact.

Both stages drive the SAME ``diffusion_denoise`` loop the engine serves (no second sampler) — so the
flywheel is real, not aspirational.
"""
from __future__ import annotations

import dataclasses
from typing import Any

from .methods import build_diffusion_nft, build_dmd2


def derive_card(base_card: Any, model_id: str, *, method: str, parents: list[str]) -> Any:
    """A child (recipe, runtime) card: same components/loops, new recipe with provenance ``parents``."""
    recipe = dataclasses.replace(base_card.recipe, method=method, parents=list(parents))
    return dataclasses.replace(base_card, model_id=model_id, recipe=recipe).validate()


def run_flywheel(base_card: Any, *, rl_iters: int = 8, distill_iters: int = 15,
                 batch: dict | None = None) -> dict:
    """RL the base, then distill from the RL'd model. Returns the instances + the provenance cards."""
    batch = batch or {"prompts": ["a red car", "a blue boat"], "seeds": [1, 2]}

    # 1) RL — DiffusionNFT improves the base policy on the shared denoise loop
    nft = build_diffusion_nft(base_card, num_video_per_prompt=4, num_inner_timesteps=2)
    for it in range(rl_iters):
        nft.train_step(batch, it)
    rl_card = derive_card(base_card, base_card.model_id + "-rl",
                          method="diffusion_nft", parents=[base_card.model_id])

    # 2) Distill FROM the RL'd model: the DMD2 teacher IS the RL'd policy
    dmd = build_dmd2(base_card, rollout_steps=4)
    dmd.teacher.component("transformer").copy_from(nft.student.component("transformer"))
    for it in range(distill_iters):
        dmd.train_step(batch, it)
    distilled_card = derive_card(base_card, base_card.model_id + "-rl-distilled",
                                 method="dmd2", parents=[rl_card.model_id])

    return {"rl": nft.student, "rl_card": rl_card,
            "distilled": dmd.student, "distilled_card": distilled_card,
            "teacher": dmd.teacher, "base_card": base_card}
