"""LTX-2 ModelCards — split by ARCHITECTURE, not by version label (design_v3 §4, §15).

* ``build_ltx2_card`` — the **two-stage distilled** pipeline: ``ltx2_base`` (8-step) → learned spatial
  upsampler → ``ltx2_refine`` (3-step), both binding the same ``transformer``. Serves the distilled
  checkpoint that ships a ``spatial_upsampler`` (e.g. ``FastVideo/LTX2-Distilled-Diffusers``).
* ``build_ltx2_base_card`` — the **single-stage** pipeline: one request-driven flow-match loop at full
  latent res, no upsampler. Serves both the non-distilled base (``Davids048/LTX2-Base-Diffusers``,
  many-step) and the single-stage distilled (``FastVideo/LTX-2.3-Distilled-Diffusers``, few-step).

The "ltx2" vs "ltx2.3" version names do NOT map cleanly onto these — the real axis is
two-stage-with-upsampler vs single-stage, and dispatch picks by ``has_spatial_upsampler``.

Real wiring: DiT ``ltx2:LTX2Transformer3DModel``, causal VAE ``ltx2vae``, Gemma text encoder.
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
from ..backend import ToyAudioVAE, ToyDiT, ToyTextEncoder, ToyUpsampler, ToyVAE, _seed_from
from .loop import BASE_SIGMAS, REFINE_SIGMAS, LTX2DenoiseLoop


def build_ltx2_card(model_id: str = "ltx2-2stage-distilled") -> ModelCard:
    seed = _seed_from(model_id)
    cost = CostModel(kind=WorkUnitKind.DIFFUSION_STEP, base_seconds=1.5e-4, per_unit_seconds=1.2e-7)

    def base_factory():
        return LTX2DenoiseLoop(loop_id="ltx2_base", stage="base", sigmas=BASE_SIGMAS,
                               cfg_scale=3.0, stg_scale=0.1, cost=cost, input_slot=None, seed_offset=0)

    def refine_factory():
        return LTX2DenoiseLoop(loop_id="ltx2_refine", stage="refine", sigmas=REFINE_SIGMAS,
                               cfg_scale=1.0, stg_scale=0.0, cost=cost,
                               input_slot="ltx_upsampled", seed_offset=1000,
                               audio_input_slot="ltx_audio")   # read threaded audio in T2VS (else unused)

    components = {
        "text_encoder": ComponentSpec(component_id="text_encoder", kind="text_encoder",
                                      load_id="transformers:AutoModel",  # Gemma
                                      factory=lambda inst: ToyTextEncoder(), required_for={"t2v"}),
        "vae": ComponentSpec(component_id="vae", kind="vae",
                             load_id="fastvideo.models.vaes.ltx2vae:VideoDecoder",
                             factory=lambda inst: ToyVAE(), required_for={"t2v"}),
        # learned 2x spatial latent upsampler between the base and refine stages (real:
        # LTX2LatentUpsampler; CPU toy: nearest-neighbor). Applied in program.py:_upsample.
        "spatial_upsampler": ComponentSpec(
            component_id="spatial_upsampler", kind="upsampler",
            load_id="fastvideo.models.upsamplers.ltx2_upsampler:LTX2LatentUpsampler",
            factory=lambda inst: ToyUpsampler(), required_for={"t2v"}),
        "transformer": ComponentSpec(component_id="transformer", kind="dit",
                                     load_id="fastvideo.models.dits.ltx2:LTX2Transformer3DModel",
                                     factory=lambda inst: ToyDiT(seed=seed),
                                     resident_for=["ltx2_base", "ltx2_refine"], required_for={"t2v"}),
        # audio branch: lazy for T2V (not loaded), required for T2VS (joint audio+video, §9.11)
        "audio_vae": ComponentSpec(component_id="audio_vae", kind="audio_vae",
                                   load_id="fastvideo.models.audio.ltx2_audio_vae:AudioVAE",
                                   factory=lambda inst: ToyAudioVAE(),
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
        capabilities=CapabilityMatrix.of(Capability.TEXT_TO_VIDEO, Capability.TEXT_TO_VIDEO_SOUND,
                                         Capability.VAE_DECODE),
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


def build_ltx2_base_card(model_id: str = "ltx2-single-stage") -> ModelCard:
    """Single-stage LTX-2 card — one request-driven flow-match loop at FULL latent res (no base/refine
    split, no spatial upsampler). Serves BOTH the non-distilled base (`Davids048/LTX2-Base-Diffusers`,
    many-step) AND the single-stage distilled `FastVideo/LTX-2.3-Distilled-Diffusers` (few-step — pass a
    small num_inference_steps). Reuses the LTX-2 torch adapters (DiT/VAE/Gemma). NOTE: the schedule is a
    plain linspace; the distilled checkpoint would ideally use its own tuned few-step sigmas (follow-up)."""
    seed = _seed_from(model_id)
    cost = CostModel(kind=WorkUnitKind.DIFFUSION_STEP, base_seconds=1.5e-4, per_unit_seconds=1.2e-7)

    def single_factory():
        return LTX2DenoiseLoop(loop_id="ltx2_single", stage="single", sigmas=[1.0, 0.0],
                               cfg_scale=3.0, stg_scale=0.0, cost=cost, input_slot=None,
                               seed_offset=0, full_res=True, request_steps=True, shift=1.0)

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
                                     resident_for=["ltx2_single"], required_for={"t2v"}),
    }
    loops = {
        "ltx2_single": LoopSpec(loop_id="ltx2_single", kind=LoopKind.DIFFUSION_DENOISE,
                                work_unit_kind=WorkUnitKind.DIFFUSION_STEP, step_cost_model=cost,
                                shared_weight_components=["transformer"], cache_policy=["feature"],
                                loop_factory=single_factory),
    }
    card = ModelCard(
        model_id=model_id, family="ltx2",
        components=components, loops=loops,
        capabilities=CapabilityMatrix.of(Capability.TEXT_TO_VIDEO, Capability.VAE_DECODE),
        recipe=RecipeSpec(method="base", assumes_loop="ltx2_single",
                          assumes_precision="float32", consistency_required=ConsistencyLevel.C1),
        parity=ParitySpec(consistency_levels=[ConsistencyLevel.C1], interleave_required=True),
        caches={"feature": CacheContract(cache_class="feature", max_bytes=1 << 24,
                                         reuse_across_requests=True)},
        precision=PrecisionContract(default_dtype="float32", training_precision="float32"),
        parallelism=ParallelismContract(valid_plans=[ParallelPlan.single()],
                                        default_plan=ParallelPlan.single()),
    )
    return card.validate()
