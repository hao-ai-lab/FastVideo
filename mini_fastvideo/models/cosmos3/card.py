"""Cosmos3 omni ModelCard (design_v3 §4.1, §15b) — the forcing function for omni-native serving.

ONE resident MoT ``transformer`` is bound to BOTH the ``ar_decode`` loop (the reasoner, und pathway)
and the ``diffusion_denoise`` loop (the joint generation pathway) — ``shared_weight_components`` on
both. A request runs the reasoner (prompt upsampling), packs its tokens into conditioning, then the
joint denoise — all on the same resident weights, with every AR token and denoise step a
runtime-visible WorkUnit (the differentiation vs vllm-omni's opaque DIFFUSION stage). The diffusion
loop is literally ``WanDenoiseLoop`` — the same step body the engine serves for Wan — bound to the
MoT module. ``sound_vae`` is declared ``optional_for`` non-t2vs tasks (the lazy-component fix, P8).
"""
from __future__ import annotations

from ..._enums import Capability, ConsistencyLevel, LoopKind, WorkUnitKind
from ...card import (
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
from ...loop.policies import ClassicCFG, FlowShiftPolicy, NoRouting, PrecisionPolicy
from ...parallel import ParallelPlan
from ..backend import ToyMoTDiT, ToyTokenizer, ToyVAE, _seed_from
from ..omni import ARDecodeLoop
from ..wan21.loop import WanDenoiseLoop


def build_cosmos3_card(model_id: str = "cosmos3-vfm") -> ModelCard:
    seed = _seed_from(model_id)
    ar_cost = CostModel(kind=WorkUnitKind.AR_TOKEN, base_seconds=5e-5, per_unit_seconds=1e-7)
    dn_cost = CostModel(kind=WorkUnitKind.DIFFUSION_STEP, base_seconds=1e-4, per_unit_seconds=1e-7)
    cfg, flow = ClassicCFG(), FlowShiftPolicy(shift=5.0)
    precision, expert = PrecisionPolicy(), NoRouting("transformer")

    def ar_factory():
        return ARDecodeLoop(loop_id="ar_decode", transformer_id="transformer", cost=ar_cost, max_tokens=6)

    def dn_factory():
        return WanDenoiseLoop(loop_id="diffusion_denoise", cfg=cfg, flow_shift=flow,
                              precision=precision, expert=expert, cost=dn_cost)

    components = {
        "tokenizer": ComponentSpec("tokenizer", kind="tokenizer",
                                   factory=lambda inst: ToyTokenizer(),
                                   required_for={"reason", "t2v", "t2vs"}),
        "transformer": ComponentSpec(
            "transformer", kind="dit",
            load_id="fastvideo.models.dits.cosmos3:Cosmos3VFMTransformer",
            factory=lambda inst: ToyMoTDiT(seed=seed),
            resident_for=["ar_decode", "diffusion_denoise"],     # never unload mid-request
            required_for={"reason", "t2v", "t2vs"}),
        "vae": ComponentSpec("vae", kind="vae", factory=lambda inst: ToyVAE(),
                             required_for={"t2v", "t2vs"}),
        "sound_vae": ComponentSpec("sound_vae", kind="audio_vae",
                                   load_id="fastvideo.models.audio:Cosmos3SoundVAE",
                                   optional_for={"reason", "t2v"}, required_for={"t2vs"}),
    }
    loops = {
        "ar_decode": LoopSpec("ar_decode", kind=LoopKind.AR_DECODE, work_unit_kind=WorkUnitKind.AR_TOKEN,
                              step_cost_model=ar_cost, shared_weight_components=["transformer"],
                              cache_policy=["paged_kv"], loop_factory=ar_factory),
        "diffusion_denoise": LoopSpec("diffusion_denoise", kind=LoopKind.DIFFUSION_DENOISE,
                                      work_unit_kind=WorkUnitKind.DIFFUSION_STEP, step_cost_model=dn_cost,
                                      shared_weight_components=["transformer"], cache_policy=["feature"],
                                      loop_factory=dn_factory),
    }
    card = ModelCard(
        model_id=model_id, family="cosmos3", components=components, loops=loops,
        capabilities=CapabilityMatrix.of(
            Capability.TEXT_TO_VIDEO, Capability.REASONING_TEXT, Capability.TEXT_TO_VIDEO_SOUND,
            Capability.VAE_DECODE, Capability.POLICY_ROLLOUT),
        recipe=RecipeSpec(method="base", assumes_loop="diffusion_denoise",
                          assumes_precision="float32", consistency_required=ConsistencyLevel.C1),
        parity=ParitySpec(consistency_levels=[ConsistencyLevel.C1], interleave_required=True),
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
