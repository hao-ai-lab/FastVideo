"""Multi-expert ModelCard — N refiner LMs + one generator, for N-way joint RL (design_v3 §4, §10).

The generalization test of the UniRL stress test: where ``unified`` has *two* trainable experts
(``llm`` + ``transformer``), this card has ``n_refiners`` prompt-refiner LMs **plus** a generator —
N+1 disjoint experts on N+1 loops (``ar_decode_0…n-1`` + ``diffusion_denoise``). It exists to show
that "joint RL over more than two experts" needs **no new card primitive**: a card already holds an
arbitrary number of components and loops (Qwen-Omni proved three). The only question is whether the
*method* can drive N updates from one reward — answered by ``JointMultiExpertRL`` (no, it needs no
rewrite; it loops over an expert list).

The N refiners' refinements *compose* into the diffusion conditioning (each adds its embedding
offset), so the generator's output depends on every refiner — and one reward trains them all.
"""
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
from v2.platform.backends.toy import ToyDiT, ToyPromptRefiner, ToyTextEncoder, ToyVAE, _seed_from
from v2.recipes.omni import ARDecodeLoop
from v2.recipes.wan21.loop import WanDenoiseLoop

N_REFINE_ACTIONS = 6


def refiner_ids(n_refiners: int) -> list[str]:
    return [f"refiner_{i}" for i in range(n_refiners)]


def build_multi_expert_card(model_id: str = "multi-expert-rl", *, n_refiners: int = 2) -> ModelCard:
    seed = _seed_from(model_id)
    ar_cost = CostModel(kind=WorkUnitKind.AR_TOKEN, base_seconds=5e-5, per_unit_seconds=1e-7)
    dn_cost = CostModel(kind=WorkUnitKind.DIFFUSION_STEP, base_seconds=1e-4, per_unit_seconds=1e-7)
    cfg, flow = ClassicCFG(), FlowShiftPolicy(shift=3.0)
    precision, expert = PrecisionPolicy(), NoRouting("transformer")
    rids = refiner_ids(n_refiners)

    def denoise_factory():
        return WanDenoiseLoop(loop_id="diffusion_denoise", cfg=cfg, flow_shift=flow,
                              precision=precision, expert=expert, cost=dn_cost)

    def ar_factory(rid):
        return lambda: ARDecodeLoop(loop_id=f"ar_decode_{rid}", transformer_id=rid, cost=ar_cost,
                                    max_tokens=1, prompt_slot="prompt_tokens")

    components = {
        "text_encoder": ComponentSpec("text_encoder", kind="text_encoder",
                                      factory=lambda inst: ToyTextEncoder(), required_for={"t2v"}),
        "transformer": ComponentSpec("transformer", kind="dit",
                                     factory=lambda inst: ToyDiT(seed=seed),
                                     resident_for=["diffusion_denoise"], required_for={"t2v"}),
        "vae": ComponentSpec("vae", kind="vae", factory=lambda inst: ToyVAE(), required_for={"t2v"}),
    }
    loops = {
        "diffusion_denoise": LoopSpec("diffusion_denoise", kind=LoopKind.DIFFUSION_DENOISE,
                                      work_unit_kind=WorkUnitKind.DIFFUSION_STEP, step_cost_model=dn_cost,
                                      shared_weight_components=["transformer"], cache_policy=["feature"],
                                      loop_factory=denoise_factory),
    }
    for i, rid in enumerate(rids):                               # N refiner experts + N ar loops
        components[rid] = ComponentSpec(
            rid, kind="dit",
            factory=(lambda s: (lambda inst: ToyPromptRefiner(n_actions=N_REFINE_ACTIONS, seed=s)))(seed + 11 + i),
            resident_for=[f"ar_decode_{rid}"], required_for={"t2v"})
        loops[f"ar_decode_{rid}"] = LoopSpec(
            f"ar_decode_{rid}", kind=LoopKind.AR_DECODE, work_unit_kind=WorkUnitKind.AR_TOKEN,
            step_cost_model=ar_cost, shared_weight_components=[rid], cache_policy=["paged_kv"],
            loop_factory=ar_factory(rid))

    card = ModelCard(
        model_id=model_id, family="multi_expert", components=components, loops=loops,
        capabilities=CapabilityMatrix.of(Capability.TEXT_TO_VIDEO, Capability.REASONING_TEXT,
                                         Capability.VAE_DECODE, Capability.POLICY_ROLLOUT),
        recipe=RecipeSpec(method="base", assumes_loop="diffusion_denoise",
                          assumes_precision="float32", consistency_required=ConsistencyLevel.C2),
        parity=ParitySpec(consistency_levels=[ConsistencyLevel.C1, ConsistencyLevel.C2],
                          interleave_required=True),
        caches={
            "feature": CacheContract("feature", max_bytes=1 << 24, reuse_across_requests=True),
            "paged_kv": CacheContract("paged_kv", max_bytes=1 << 24, block_bytes=1 << 12,
                                      reuse_across_requests=False),
        },
        precision=PrecisionContract(default_dtype="float32", training_precision="float32"),
        parallelism=ParallelismContract(valid_plans=[ParallelPlan.single()],
                                        default_plan=ParallelPlan.single()),
    )
    return card.validate()
