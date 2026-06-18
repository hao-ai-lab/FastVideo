"""Unified LM+generator ModelCard — the UniRL/PromptRL stress test (design_v3 §4, §10).

This is the topological *opposite* of the Cosmos3 omni card, expressed in the SAME vocabulary:

  * Cosmos3 MoT: ONE ``transformer`` bound to BOTH ``ar_decode`` and ``diffusion_denoise``
    (shared weights — splitting them severs shared state).
  * Unified UniRL: TWO separate experts — an ``llm`` prompt-refiner (bound to ``ar_decode``) and a
    ``transformer`` flow generator (bound to ``diffusion_denoise``). The LM refines the prompt; its
    refinement conditions the generator. They do NOT share weights (UniRL keeps Qwen and FLUX as
    distinct experts), yet BOTH are trainable policies updated under one RL reward.

The stress-test claim: the Card/Loop/Program split that expresses MoT weight-sharing *also* expresses
two-separate-experts-joint-RL with no new primitive — only a second ``shared_weight_components`` target
and a method (``unified_rl``) that emits two WeightSyncPlans. The loops, caches, parity gate, rollout
capture, and serve path are unchanged.
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
from v2.models.backend import ToyDiT, ToyPromptRefiner, ToyTextEncoder, ToyVAE, _seed_from
from v2.models.omni import ARDecodeLoop
from v2.models.wan21.loop import WanDenoiseLoop

N_REFINE_ACTIONS = 8


def build_unified_card(model_id: str = "unirl-qwenflux") -> ModelCard:
    seed = _seed_from(model_id)
    ar_cost = CostModel(kind=WorkUnitKind.AR_TOKEN, base_seconds=5e-5, per_unit_seconds=1e-7)
    dn_cost = CostModel(kind=WorkUnitKind.DIFFUSION_STEP, base_seconds=1e-4, per_unit_seconds=1e-7)
    cfg, flow = ClassicCFG(), FlowShiftPolicy(shift=3.0)
    precision, expert = PrecisionPolicy(), NoRouting("transformer")

    def refine_factory():
        # the LM expert is the ar loop's "transformer" — a *different* component than the generator
        return ARDecodeLoop(loop_id="ar_decode", transformer_id="llm", cost=ar_cost, max_tokens=1)

    def denoise_factory():
        return WanDenoiseLoop(loop_id="diffusion_denoise", cfg=cfg, flow_shift=flow,
                              precision=precision, expert=expert, cost=dn_cost)

    components = {
        "text_encoder": ComponentSpec("text_encoder", kind="text_encoder",
                                      factory=lambda inst: ToyTextEncoder(),
                                      required_for={"reason", "t2v"}),
        "llm": ComponentSpec(                                 # the prompt-refiner expert (Qwen role)
            "llm", kind="dit",
            load_id="fastvideo.models.llm:QwenPromptRefiner",
            factory=lambda inst: ToyPromptRefiner(n_actions=N_REFINE_ACTIONS, seed=seed + 1),
            resident_for=["ar_decode"], required_for={"reason", "t2v"}),
        "transformer": ComponentSpec(                         # the flow generator expert (FLUX role)
            "transformer", kind="dit",
            load_id="fastvideo.models.dits.flux:FluxTransformer",
            factory=lambda inst: ToyDiT(seed=seed),
            resident_for=["diffusion_denoise"], required_for={"t2v"}),
        "vae": ComponentSpec("vae", kind="vae", factory=lambda inst: ToyVAE(),
                             required_for={"t2v"}),
    }
    loops = {
        "ar_decode": LoopSpec("ar_decode", kind=LoopKind.AR_DECODE, work_unit_kind=WorkUnitKind.AR_TOKEN,
                              step_cost_model=ar_cost, shared_weight_components=["llm"],
                              cache_policy=["paged_kv"], loop_factory=refine_factory),
        "diffusion_denoise": LoopSpec("diffusion_denoise", kind=LoopKind.DIFFUSION_DENOISE,
                                      work_unit_kind=WorkUnitKind.DIFFUSION_STEP, step_cost_model=dn_cost,
                                      shared_weight_components=["transformer"], cache_policy=["feature"],
                                      loop_factory=denoise_factory),
    }
    card = ModelCard(
        model_id=model_id, family="unified", components=components, loops=loops,
        capabilities=CapabilityMatrix.of(
            Capability.TEXT_TO_VIDEO, Capability.REASONING_TEXT,
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
