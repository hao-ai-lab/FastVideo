"""LTX-2 ModelCards — split by ARCHITECTURE, not by version label.

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

from v2._enums import Capability, ConsistencyLevel, LoopKind, WorkUnitKind
from v2.card import (
    CacheContract,
    CapabilityMatrix,
    ComponentSpec,
    LoopSpec,
    ModelCard,
    ParallelismContract,
    ParitySpec,
    PrecisionContract,
    RecipeSpec,
    SamplingDefaults,
)
from v2.parallel import ParallelPlan
from v2.platform.backends.toy import ToyAudioVAE, ToyDiT, ToyTextEncoder, ToyUpsampler, ToyVAE, ToyVocoder, _seed_from
from v2.recipes._prompts import LTX2_NEG
from v2.recipes.ltx2.loop import BASE_SIGMAS, REFINE_SIGMAS, LTX2DenoiseLoop, LTX23DenoiseLoop


def build_ltx2_card(model_id: str = "ltx2-2stage-distilled") -> ModelCard:
    seed = _seed_from(model_id)

    def base_factory():
        # The frame-jump/blockiness was the OOM-forced 57-frame reduction (only 8 latent temporal frames);
        # at the model's intended 121 (16 latent frames, VAE tiling enabled) the shot is temporally coherent.
        # CFG drives brightness/contrast here (cfg=3 -> mean~86, cfg=1 -> washed out ~66), so keep cfg=3.
        # STG=0.0: the v2 "drop text" perturbation is NOT real skip-layer STG, so leave it off.
        return LTX2DenoiseLoop(loop_id="ltx2_base",
                               stage="base",
                               sigmas=BASE_SIGMAS,
                               cfg_scale=3.0,
                               stg_scale=0.0,
                               input_slot=None,
                               seed_offset=0)

    def refine_factory():
        return LTX2DenoiseLoop(loop_id="ltx2_refine",
                               stage="refine",
                               sigmas=REFINE_SIGMAS,
                               cfg_scale=1.0,
                               stg_scale=0.0,
                               input_slot="ltx_upsampled",
                               seed_offset=1000,
                               audio_input_slot="ltx_audio")  # read threaded audio in T2VS (else unused)

    components = {
        "text_encoder":
        ComponentSpec(
            component_id="text_encoder",
            kind="text_encoder",
            load_id="transformers:AutoModel",  # Gemma
            factory=lambda inst: ToyTextEncoder(),
            required_for={"t2v"}),
        "vae":
        ComponentSpec(component_id="vae",
                      kind="vae",
                      load_id="fastvideo.models.vaes.ltx2vae:VideoDecoder",
                      factory=lambda inst: ToyVAE(),
                      required_for={"t2v"}),
        # learned 2x spatial latent upsampler between the base and refine stages (real:
        # LTX2LatentUpsampler; CPU toy: nearest-neighbor). Applied in program.py:_upsample.
        "spatial_upsampler":
        ComponentSpec(component_id="spatial_upsampler",
                      kind="upsampler",
                      load_id="fastvideo.models.upsamplers.ltx2_upsampler:LTX2LatentUpsampler",
                      factory=lambda inst: ToyUpsampler(),
                      required_for={"t2v"}),
        "transformer":
        ComponentSpec(component_id="transformer",
                      kind="dit",
                      load_id="fastvideo.models.dits.ltx2:LTX2Transformer3DModel",
                      factory=lambda inst: ToyDiT(seed=seed),
                      resident_for=["ltx2_base", "ltx2_refine"],
                      required_for={"t2v"}),
        # audio branch: lazy for T2V (not loaded), required for T2VS (joint audio+video)
        "audio_vae":
        ComponentSpec(component_id="audio_vae",
                      kind="audio_vae",
                      load_id="fastvideo.models.audio.ltx2_audio_vae:AudioVAE",
                      factory=lambda inst: ToyAudioVAE(),
                      optional_for={"t2v"},
                      required_for={"t2vs"}),
    }
    loops = {
        "ltx2_base":
        LoopSpec(loop_id="ltx2_base",
                 kind=LoopKind.DIFFUSION_DENOISE,
                 work_unit_kind=WorkUnitKind.DIFFUSION_STEP,
                 shared_weight_components=["transformer"],
                 cache_policy=["feature"],
                 loop_factory=base_factory),
        "ltx2_refine":
        LoopSpec(loop_id="ltx2_refine",
                 kind=LoopKind.DIFFUSION_DENOISE,
                 work_unit_kind=WorkUnitKind.DIFFUSION_STEP,
                 shared_weight_components=["transformer"],
                 cache_policy=["feature"],
                 loop_factory=refine_factory),
    }
    card = ModelCard(
        model_id=model_id,
        family="ltx2",
        components=components,
        loops=loops,
        capabilities=CapabilityMatrix.of(Capability.TEXT_TO_VIDEO, Capability.TEXT_TO_VIDEO_SOUND,
                                         Capability.VAE_DECODE),
        recipe=RecipeSpec(method="distilled",
                          parents=["ltx2.3-base"],
                          assumes_loop="ltx2_base",
                          assumes_precision="float32",
                          consistency_required=ConsistencyLevel.C1),
        sampling_defaults=SamplingDefaults(  # two-stage distilled: 8-step base (+ 2-step refine in loop)
            num_steps=8,
            guidance_scale=1.0,
            height=1024,
            width=1536,
            num_frames=121,
            fps=24,
            negative_prompt=""),
        parity=ParitySpec(consistency_levels=[ConsistencyLevel.C1], interleave_required=True),
        caches={"feature": CacheContract(cache_class="feature", max_bytes=1 << 24, reuse_across_requests=True)},
        precision=PrecisionContract(default_dtype="float32", training_precision="float32"),
        parallelism=ParallelismContract(valid_plans=[ParallelPlan.single()], default_plan=ParallelPlan.single()),
    )
    return card.validate()


def build_ltx2_base_card(model_id: str = "ltx2-single-stage") -> ModelCard:
    """Single-stage LTX-2 card — one request-driven flow-match loop at FULL latent res (no base/refine
    split, no spatial upsampler). Serves BOTH the non-distilled base (`Davids048/LTX2-Base-Diffusers`,
    many-step) AND the single-stage distilled `FastVideo/LTX-2.3-Distilled-Diffusers` (few-step — pass a
    small num_inference_steps). Reuses the LTX-2 torch adapters (DiT/VAE/Gemma). NOTE: the schedule is a
    plain linspace; the distilled checkpoint would ideally use its own tuned few-step sigmas (follow-up)."""
    seed = _seed_from(model_id)

    def single_factory():
        return LTX2DenoiseLoop(loop_id="ltx2_single",
                               stage="single",
                               sigmas=[1.0, 0.0],
                               cfg_scale=3.0,
                               stg_scale=0.0,
                               input_slot=None,
                               seed_offset=0,
                               full_res=True,
                               request_steps=True,
                               shift=1.0)

    components = {
        "text_encoder":
        ComponentSpec(
            component_id="text_encoder",
            kind="text_encoder",
            load_id="transformers:AutoModel",  # Gemma
            factory=lambda inst: ToyTextEncoder(),
            required_for={"t2v"}),
        "vae":
        ComponentSpec(component_id="vae",
                      kind="vae",
                      load_id="fastvideo.models.vaes.ltx2vae:VideoDecoder",
                      factory=lambda inst: ToyVAE(),
                      required_for={"t2v"}),
        "transformer":
        ComponentSpec(component_id="transformer",
                      kind="dit",
                      load_id="fastvideo.models.dits.ltx2:LTX2Transformer3DModel",
                      factory=lambda inst: ToyDiT(seed=seed),
                      resident_for=["ltx2_single"],
                      required_for={"t2v"}),
    }
    loops = {
        "ltx2_single":
        LoopSpec(loop_id="ltx2_single",
                 kind=LoopKind.DIFFUSION_DENOISE,
                 work_unit_kind=WorkUnitKind.DIFFUSION_STEP,
                 shared_weight_components=["transformer"],
                 cache_policy=["feature"],
                 loop_factory=single_factory),
    }
    card = ModelCard(
        model_id=model_id,
        family="ltx2",
        components=components,
        loops=loops,
        capabilities=CapabilityMatrix.of(Capability.TEXT_TO_VIDEO, Capability.VAE_DECODE),
        recipe=RecipeSpec(method="base",
                          assumes_loop="ltx2_single",
                          assumes_precision="float32",
                          consistency_required=ConsistencyLevel.C1),
        sampling_defaults=SamplingDefaults(num_steps=40,
                                           guidance_scale=3.0,
                                           height=512,
                                           width=768,
                                           num_frames=121,
                                           fps=24,
                                           negative_prompt=LTX2_NEG),
        parity=ParitySpec(consistency_levels=[ConsistencyLevel.C1], interleave_required=True),
        caches={"feature": CacheContract(cache_class="feature", max_bytes=1 << 24, reuse_across_requests=True)},
        precision=PrecisionContract(default_dtype="float32", training_precision="float32"),
        parallelism=ParallelismContract(valid_plans=[ParallelPlan.single()], default_plan=ParallelPlan.single()),
    )
    return card.validate()


def build_ltx2_3_card(model_id: str = "ltx2.3-distilled") -> ModelCard:
    """LTX-2.3 single-stage JOINT text->video+audio (T2VS). Distinct from the single-stage *base*: the
    2.3 text encoder's connector emits SEPARATE video/audio projections and the DiT carries
    gated-attention params (both auto-built by the loaders from config.json), and ONE DiT forward
    cross-attends video<->audio (``LTX23DenoiseLoop``). Distilled few-step schedule (BASE_SIGMAS).
    Adds ``audio_vae`` (AudioDecoder) + ``vocoder`` for the audio branch; a plain T2V request runs
    video-only (audio components stay unbuilt)."""
    seed = _seed_from(model_id)

    def loop_factory():
        return LTX23DenoiseLoop(loop_id="ltx2_3", sigmas=BASE_SIGMAS, cfg_scale=1.0, stg_scale=0.0)

    components = {
        "text_encoder":
        ComponentSpec(
            component_id="text_encoder",
            kind="text_encoder",
            load_id="transformers:AutoModel",  # LTX2GemmaTextEncoderModel
            factory=lambda inst: ToyTextEncoder(),
            required_for={"t2v", "t2vs"}),
        "vae":
        ComponentSpec(component_id="vae",
                      kind="vae",
                      load_id="fastvideo.models.vaes.ltx2vae:VideoDecoder",
                      factory=lambda inst: ToyVAE(),
                      required_for={"t2v", "t2vs"}),
        "transformer":
        ComponentSpec(component_id="transformer",
                      kind="dit",
                      load_id="fastvideo.models.dits.ltx2:LTX2Transformer3DModel",
                      factory=lambda inst: ToyDiT(seed=seed),
                      resident_for=["ltx2_3"],
                      required_for={"t2v", "t2vs"}),
        # audio branch (T2VS only): AudioDecoder (latent->mel) + Vocoder (mel->waveform). Not built for t2v.
        "audio_vae":
        ComponentSpec(component_id="audio_vae",
                      kind="audio_vae",
                      load_id="fastvideo.models.audio.ltx2_audio_vae:AudioDecoder",
                      factory=lambda inst: ToyAudioVAE(),
                      required_for={"t2vs"},
                      optional_for={"t2v"}),
        "vocoder":
        ComponentSpec(component_id="vocoder",
                      kind="vocoder",
                      load_id="fastvideo.models.audio.ltx2_audio_vae:Vocoder",
                      factory=lambda inst: ToyVocoder(),
                      required_for={"t2vs"},
                      optional_for={"t2v"}),
    }
    loops = {
        "ltx2_3":
        LoopSpec(loop_id="ltx2_3",
                 kind=LoopKind.DIFFUSION_DENOISE,
                 work_unit_kind=WorkUnitKind.DIFFUSION_STEP,
                 shared_weight_components=["transformer"],
                 cache_policy=["feature"],
                 loop_factory=loop_factory),
    }
    card = ModelCard(
        model_id=model_id,
        family="ltx2",
        components=components,
        loops=loops,
        capabilities=CapabilityMatrix.of(Capability.TEXT_TO_VIDEO, Capability.TEXT_TO_VIDEO_SOUND,
                                         Capability.VAE_DECODE),
        recipe=RecipeSpec(method="distilled",
                          parents=["ltx2.3-base"],
                          assumes_loop="ltx2_3",
                          assumes_precision="float32",
                          consistency_required=ConsistencyLevel.C1),
        sampling_defaults=SamplingDefaults(  # joint A/V: per-modality cfg video 3 / audio 7
            num_steps=30,
            guidance_scale=3.0,
            height=512,
            width=768,
            num_frames=121,
            fps=24,
            negative_prompt=LTX2_NEG,
            guidance_per_modality={
                "video": 3.0,
                "audio": 7.0
            }),
        parity=ParitySpec(consistency_levels=[ConsistencyLevel.C1], interleave_required=True),
        caches={"feature": CacheContract(cache_class="feature", max_bytes=1 << 24, reuse_across_requests=True)},
        precision=PrecisionContract(default_dtype="float32", training_precision="float32"),
        parallelism=ParallelismContract(valid_plans=[ParallelPlan.single()], default_plan=ParallelPlan.single()),
    )
    return card.validate()
