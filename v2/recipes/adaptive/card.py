"""Adaptive-compute card — a Wan-like T2V model whose denoise loop owns content-adaptive control flow
(cache-dit skip + early-exit). Reuses the Wan T2V program (loop_id ``diffusion_denoise``); only the
loop differs."""
from __future__ import annotations

from v2._enums import Capability, ConsistencyLevel, LoopKind, WorkUnitKind
from v2.card import (
    CacheContract,
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
from v2.loop.policies import ClassicCFG, FlowShiftPolicy, NoRouting, PrecisionPolicy
from v2.parallel import ParallelPlan
from v2.platform.backends.toy import ToyDiT, ToyTextEncoder, ToyVAE, _seed_from
from v2.recipes.adaptive.loop import CacheDiTDenoiseLoop


def build_adaptive_card(model_id: str = "wan-adaptive",
                        *,
                        cache_threshold: float = 0.02,
                        exit_threshold: float = 0.0) -> ModelCard:
    seed = _seed_from(model_id)
    cost = CostModel(kind=WorkUnitKind.DIFFUSION_STEP, base_seconds=1e-4, per_unit_seconds=1e-7)
    cfg, flow = ClassicCFG(), FlowShiftPolicy(shift=3.0)
    precision, expert = PrecisionPolicy(), NoRouting("transformer")

    def loop_factory():
        return CacheDiTDenoiseLoop(loop_id="diffusion_denoise",
                                   cfg=cfg,
                                   flow_shift=flow,
                                   precision=precision,
                                   expert=expert,
                                   cost=cost,
                                   cache_threshold=cache_threshold,
                                   exit_threshold=exit_threshold)

    components = {
        "text_encoder":
        ComponentSpec("text_encoder", kind="text_encoder", factory=lambda inst: ToyTextEncoder(), required_for={"t2v"}),
        "transformer":
        ComponentSpec("transformer",
                      kind="dit",
                      factory=lambda inst: ToyDiT(seed=seed),
                      resident_for=["diffusion_denoise"],
                      required_for={"t2v"}),
        "vae":
        ComponentSpec("vae", kind="vae", factory=lambda inst: ToyVAE(), required_for={"t2v"}),
    }
    loops = {
        "diffusion_denoise":
        LoopSpec("diffusion_denoise",
                 kind=LoopKind.DIFFUSION_DENOISE,
                 work_unit_kind=WorkUnitKind.DIFFUSION_STEP,
                 step_cost_model=cost,
                 shared_weight_components=["transformer"],
                 cache_policy=["feature"],
                 loop_factory=loop_factory),
    }
    return ModelCard(
        model_id=model_id,
        family="wan",
        components=components,
        loops=loops,
        capabilities=CapabilityMatrix.of(Capability.TEXT_TO_VIDEO, Capability.VAE_DECODE),
        recipe=RecipeSpec(method="base",
                          assumes_loop="diffusion_denoise",
                          assumes_precision="float32",
                          consistency_required=ConsistencyLevel.C1),
        parity=ParitySpec(consistency_levels=[ConsistencyLevel.C1], interleave_required=True),
        caches={
            "feature": CacheContract("feature", max_bytes=1 << 24, reuse_across_requests=True)
        },
        precision=PrecisionContract(default_dtype="float32", training_precision="float32"),
        parallelism=ParallelismContract(valid_plans=[ParallelPlan.single()], default_plan=ParallelPlan.single()),
    ).validate()
