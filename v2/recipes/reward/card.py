"""Reward-model ModelCard — a served scorer/verifier for RL (design_v3 §10).

A reward model is just another card: a `scorer` component + a `score` loop emitting `REWARD_BATCH`
work units. So a learned reward (PickScore / CLIP / a VLM judge) is loaded, placed, priced, and
scheduled like any model — enabling RLHF/RLAIF where the reward is a model, not a heuristic, and can
run on its own role-pool (reward disaggregation).
"""
from __future__ import annotations

from v2._enums import Capability, ConsistencyLevel, LoopKind, WorkUnitKind
from v2.card import (
    CapabilityMatrix,
    ComponentSpec,
    CostModel,
    LoopSpec,
    ModelCard,
    ParallelismContract,
    ParitySpec,
    PrecisionContract,
    RecipeSpec,
)
from v2.parallel import ParallelPlan
from v2.platform.backends.toy import ToyRewardModel, _seed_from
from v2.recipes.reward.loop import RewardLoop


def build_reward_card(model_id: str = "pickscore-reward", *, batch: int = 4) -> ModelCard:
    seed = _seed_from(model_id)
    cost = CostModel(kind=WorkUnitKind.REWARD_BATCH, base_seconds=3e-5, per_unit_seconds=1e-7)

    def score_factory():
        return RewardLoop(loop_id="score", scorer_id="scorer", cost=cost, media_slot="media", batch=batch)

    components = {
        "scorer": ComponentSpec("scorer", kind="reward_model",
                                load_id="fastvideo.rl.rewards:PickScore",
                                factory=lambda inst: ToyRewardModel(seed=seed),
                                resident_for=["score"], required_for={"reason"}),
    }
    loops = {
        "score": LoopSpec("score", kind=LoopKind.ENCODER, work_unit_kind=WorkUnitKind.REWARD_BATCH,
                          step_cost_model=cost, shared_weight_components=["scorer"], cache_policy=[],
                          loop_factory=score_factory),
    }
    return ModelCard(
        model_id=model_id, family="reward", components=components, loops=loops,
        capabilities=CapabilityMatrix.of(Capability.POLICY_ROLLOUT),
        recipe=RecipeSpec(method="base", assumes_loop="score",
                          assumes_precision="float32", consistency_required=ConsistencyLevel.C1),
        parity=ParitySpec(consistency_levels=[ConsistencyLevel.C1], interleave_required=True),
        caches={},
        precision=PrecisionContract(default_dtype="float32", training_precision="float32"),
        parallelism=ParallelismContract(valid_plans=[ParallelPlan.single()],
                                        default_plan=ParallelPlan.single()),
    ).validate()
