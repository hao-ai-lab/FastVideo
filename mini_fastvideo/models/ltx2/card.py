"""LTX2.3 ModelCard — the two-stage distilled (recipe, runtime) pair (design_v3 §4, §15).

Two loops — ``ltx2_base`` (8-step distilled) and ``ltx2_refine`` (3-step distilled) — BOTH bind
the same ``transformer`` component (shared by reference). The recipe is ``distilled`` and
``assumes_loop="ltx2_base"`` with ``assumes_precision`` baked in: serving this under a 50-step
sampler would be a typed mismatch (the teeth of §2.1).

Real wiring (repo): DiT ``ltx2:LTX2Transformer3DModel`` (48 layers), causal VAE
``ltx2vae``, Gemma text encoder, distilled sigma schedules + spatial upsampler between stages.
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
from ...parallel import ParallelPlan
from ..backend import ToyDiT, ToyTextEncoder, ToyVAE, _seed_from
from .loop import BASE_SIGMAS, REFINE_SIGMAS, LTX2DenoiseLoop


def build_ltx2_card(model_id: str = "ltx2.3-distilled") -> ModelCard:
    seed = _seed_from(model_id)
    cost = CostModel(kind=WorkUnitKind.DIFFUSION_STEP, base_seconds=1.5e-4, per_unit_seconds=1.2e-7)

    def base_factory():
        return LTX2DenoiseLoop(loop_id="ltx2_base", stage="base", sigmas=BASE_SIGMAS,
                               cfg_scale=3.0, stg_scale=0.1, cost=cost, input_slot=None, seed_offset=0)

    def refine_factory():
        return LTX2DenoiseLoop(loop_id="ltx2_refine", stage="refine", sigmas=REFINE_SIGMAS,
                               cfg_scale=1.0, stg_scale=0.0, cost=cost,
                               input_slot="ltx_upsampled", seed_offset=1000)

    components = {
        "text_encoder": ComponentSpec(component_id="text_encoder", kind="text_encoder",
                                      load_id="transformers:AutoModel",  # Gemma
                                      factory=lambda inst: ToyTextEncoder(), required_for={"t2v"}),
        "vae": ComponentSpec(component_id="vae", kind="vae",
                             load_id="fastvideo.models.vaes.ltx2vae:VideoDecoder",
                             factory=lambda inst: ToyVAE(), required_for={"t2v"}),
        "transformer": ComponentSpec(component_id="transformer", kind="dit",
                                     load_id="fastvideo.models.dits.ltx2:LTX2Transformer3DModel",
                                     factory=lambda inst: ToyDiT(seed=seed),
                                     resident_for=["ltx2_base", "ltx2_refine"], required_for={"t2v"}),
        # audio_vae/vocoder are optional_for T2V (declared, not implemented in the mini)
        "audio_vae": ComponentSpec(component_id="audio_vae", kind="audio_vae",
                                   load_id="fastvideo.models.audio.ltx2_audio_vae:AudioVAE",
                                   optional_for={"t2v"}, required_for={"t2vs"}),
    }
    loops = {
        "ltx2_base": LoopSpec(loop_id="ltx2_base", kind=LoopKind.DIFFUSION_DENOISE,
                              work_unit_kind=WorkUnitKind.DIFFUSION_STEP, step_cost_model=cost,
                              shared_weight_components=["transformer"], cache_policy=["feature"],
                              loop_factory=base_factory),
        "ltx2_refine": LoopSpec(loop_id="ltx2_refine", kind=LoopKind.DIFFUSION_DENOISE,
                                work_unit_kind=WorkUnitKind.DIFFUSION_STEP, step_cost_model=cost,
                                shared_weight_components=["transformer"], cache_policy=["feature"],
                                loop_factory=refine_factory),
    }
    card = ModelCard(
        model_id=model_id, family="ltx2",
        components=components, loops=loops,
        capabilities=CapabilityMatrix.of(Capability.TEXT_TO_VIDEO, Capability.VAE_DECODE),
        recipe=RecipeSpec(method="distilled", parents=["ltx2.3-base"], assumes_loop="ltx2_base",
                          assumes_precision="float32", consistency_required=ConsistencyLevel.C1),
        parity=ParitySpec(consistency_levels=[ConsistencyLevel.C1], interleave_required=True),
        caches={"feature": CacheContract(cache_class="feature", max_bytes=1 << 24,
                                         reuse_across_requests=True)},
        precision=PrecisionContract(default_dtype="float32", training_precision="float32"),
        parallelism=ParallelismContract(valid_plans=[ParallelPlan.single()],
                                        default_plan=ParallelPlan.single()),
    )
    return card.validate()
