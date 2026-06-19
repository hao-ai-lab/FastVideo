"""FastWan (DMD few-step Wan) ModelCard — self-contained recipe package.

FastWan = Distribution-Matching-Distillation few-step Wan. Same architecture as base Wan
(``WanTransformer3DModel`` / ``AutoencoderKLWan`` / UMT5), so the Wan torch adapters are reused via
``load_id`` (no ``adapter=`` needed; the torch backend's ``_make_dit`` already builds the transformer)
along with the Wan VAE / T5 adapters. The only new piece is the DMD few-step schedule in
``FastWanDMDLoop`` (predict-x0-then-renoise over a fixed ``dmd_denoising_steps`` list).

Loadable target (this card's primary id):
  * ``FastVideo/FastWan2.2-TI2V-5B-FullAttn-Diffusers`` — full attention (no VSA kernel needed),
    TI2V-5B geometry (z_dim=48 VAE, 16x spatial / 4x temporal), DMD steps [1000, 757, 522] (3-step),
    flow_shift 5.0 / DMD-scheduler shift 8.0. From ``FAST_WAN_2_2_TI2V_5B`` preset (704x1280, 121
    frames, fps 24, guidance 5.0).

BRINGUP variants (registered here but NOT loadable in this environment):
  * ``FastVideo/FastWan2.2-TI2V-5B-Diffusers`` (non-FullAttn TI2V-5B) — trained for the VSA (Video
    Sparse Attention) kernel, not built here (no nvcc). Same card/geometry; load is BRINGUP until VSA
    is built + selected.
  * ``FastVideo/FastWan2.1-T2V-1.3B-Diffusers`` and ``FastVideo/FastWan2.1-T2V-14B-480P-Diffusers``
    (Wan2.1 16/8/4 geometry, DMD steps [1000, 757, 522], flow_shift 8.0, 448x832x61, fps 16, guidance
    3.0) — also VSA-trained, AND the DMD checkpoints carry gated-attention params (``to_gate_compress``)
    the generic Wan loader cannot map, so they need a v2-side non-strict loader option. Both BRINGUP.

``stamp_wan21_checkpoints`` applies (diffusers transformer/vae/text_encoder subfolder layout).
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
    ParityTestSpec,
    PrecisionContract,
    RecipeSpec,
    SamplingDefaults,
)
from v2.loop.policies import EmbeddedGuidance, NoRouting, PrecisionPolicy
from v2.parallel import ParallelPlan
from v2.platform.backends.toy import ToyDiT, ToyTextEncoder, ToyVAE, _seed_from
from v2.recipes._prompts import WAN_NEG_CN, WAN_NEG_EN
from v2.recipes.fastwan.loop import (
    FASTWAN_TI2V_LATENT_CHANNELS,
    FASTWAN_TI2V_SPATIAL_RATIO,
    FASTWAN_TI2V_TEMPORAL_RATIO,
    FastWanDMDLoop,
)
from v2.recipes.wan21.card import stamp_wan21_checkpoints

# DMD few-step schedule for both FastWan configs (FastWan2_1_T2V_480P_Config /
# FastWan2_2_TI2V_5B_Config both use this exact 3-step list).
_DMD_DENOISING_STEPS = [1000, 757, 522]


def _build_fastwan_card(model_id: str,
                        *,
                        latent_channels: int,
                        spatial_ratio: int,
                        temporal_ratio: int,
                        sampling_defaults: SamplingDefaults,
                        checkpoint_root: str | None = None) -> ModelCard:
    """Shared FastWan card builder. The DMD few-step loop is the only delta vs base Wan; the Wan torch
    adapters are reused via ``load_id`` (no ``adapter=``). ``EmbeddedGuidance`` (single-branch) matches
    DMD's distilled, no-CFG forward.

    No ``FlowShiftPolicy``: DMD does not integrate a continuous flow-shift sigma schedule — the schedule
    is the fixed ``dmd_denoising_steps`` list, and the re-noise scheduler's shift is fixed at 8.0 inside
    ``FastWanDMDLoop`` (the pipeline ``flow_shift`` 5.0/8.0 is doc metadata on the per-variant builders,
    not a loop input). The negative prompt rides in ``sampling_defaults``."""
    seed = _seed_from(model_id)
    cost = CostModel(kind=WorkUnitKind.DIFFUSION_STEP, base_seconds=1e-4, per_unit_seconds=1e-7)
    cfg = EmbeddedGuidance()  # DMD is distilled single-branch (guidance embedded, NOT CFG)
    precision = PrecisionPolicy(compute_dtype="float32", scheduler_step_in_fp32=True)
    expert = NoRouting("transformer")

    def loop_factory():
        return FastWanDMDLoop(loop_id="diffusion_denoise",
                              cfg=cfg,
                              precision=precision,
                              expert=expert,
                              cost=cost,
                              denoising_steps=_DMD_DENOISING_STEPS,
                              latent_channels=latent_channels,
                              spatial_ratio=spatial_ratio,
                              temporal_ratio=temporal_ratio)

    components = {
        "text_encoder":
        ComponentSpec(component_id="text_encoder",
                      kind="text_encoder",
                      load_id="fastvideo.models.encoders.t5:T5EncoderModel",
                      factory=lambda inst: ToyTextEncoder(),
                      required_for={"t2v"}),
        "vae":
        ComponentSpec(component_id="vae",
                      kind="vae",
                      load_id="fastvideo.models.vaes.wanvae:AutoencoderKLWan",
                      factory=lambda inst: ToyVAE(),
                      required_for={"t2v"}),
        "transformer":
        ComponentSpec(component_id="transformer",
                      kind="dit",
                      load_id="fastvideo.models.dits.wanvideo:WanTransformer3DModel",
                      factory=lambda inst: ToyDiT(seed=seed),
                      resident_for=["diffusion_denoise"],
                      required_for={"t2v"}),
    }
    loops = {
        "diffusion_denoise":
        LoopSpec(loop_id="diffusion_denoise",
                 kind=LoopKind.DIFFUSION_DENOISE,
                 work_unit_kind=WorkUnitKind.DIFFUSION_STEP,
                 step_cost_model=cost,
                 shared_weight_components=["transformer"],
                 cache_policy=["feature"],
                 graph_capture="breakable_cudagraph",
                 loop_factory=loop_factory),
    }
    card = ModelCard(
        model_id=model_id,
        family="wan",
        components=components,
        loops=loops,
        capabilities=CapabilityMatrix.of(Capability.TEXT_TO_VIDEO, Capability.VAE_DECODE, Capability.POLICY_ROLLOUT),
        recipe=RecipeSpec(method="dmd",
                          assumes_loop="diffusion_denoise",
                          assumes_precision="float32",
                          consistency_required=ConsistencyLevel.C1),
        parity=ParitySpec(consistency_levels=[ConsistencyLevel.C1],
                          interleave_required=True,
                          tests=[ParityTestSpec(name="denoise_trajectory", level=ConsistencyLevel.C1, tap="latents")]),
        caches={"feature": CacheContract(cache_class="feature", max_bytes=1 << 24, reuse_across_requests=True)},
        precision=PrecisionContract(default_dtype="float32", training_precision="float32"),
        parallelism=ParallelismContract(valid_plans=[ParallelPlan.single()], default_plan=ParallelPlan.single()),
        sampling_defaults=sampling_defaults,
    )
    card.validate()
    if checkpoint_root:
        stamp_wan21_checkpoints(card, checkpoint_root)
    return card


def build_fastwan_ti2v_5b_card(model_id: str = "fastwan2.2-ti2v-5b-fullattn",
                               *,
                               checkpoint_root: str | None = None) -> ModelCard:
    """FastWan2.2-TI2V-5B-FullAttn (the loadable target; FullAttn = no VSA kernel needed).

    Geometry: z_dim=48 VAE, 16x spatial / 4x temporal (Wan2.2-TI2V-5B). DMD steps [1000, 757, 522]
    (3-step), pipeline flow_shift 5.0 (the DMD re-noise scheduler internally uses shift 8.0). Defaults
    from ``FAST_WAN_2_2_TI2V_5B``: 704x1280, 121 frames, fps 24, guidance 5.0. ``num_steps=3`` is the
    DMD schedule length (the preset's 50 is ignored by the DMD stage).

    Serves both ``FastWan2.2-TI2V-5B-FullAttn-Diffusers`` (loadable) and ``FastWan2.2-TI2V-5B-Diffusers``
    (non-FullAttn variant, VSA-trained, so load is BRINGUP; same card/geometry)."""
    return _build_fastwan_card(model_id,
                               latent_channels=FASTWAN_TI2V_LATENT_CHANNELS,
                               spatial_ratio=FASTWAN_TI2V_SPATIAL_RATIO,
                               temporal_ratio=FASTWAN_TI2V_TEMPORAL_RATIO,
                               sampling_defaults=SamplingDefaults(num_steps=3,
                                                                  guidance_scale=5.0,
                                                                  height=704,
                                                                  width=1280,
                                                                  num_frames=121,
                                                                  fps=24,
                                                                  negative_prompt=WAN_NEG_CN),
                               checkpoint_root=checkpoint_root)


def build_fastwan_t2v_1_3b_card(model_id: str = "fastwan2.1-t2v-1.3b",
                                *,
                                checkpoint_root: str | None = None) -> ModelCard:
    """FastWan2.1-T2V-1.3B / -14B-480P (BRINGUP: VSA kernel + non-strict ``to_gate_compress`` load).

    Geometry: Wan2.1 defaults (z_dim=16 VAE, 8x spatial / 4x temporal). DMD steps [1000, 757, 522],
    flow_shift 8.0. Defaults from ``FAST_WAN_T2V_480P``: 448x832, 61 frames, fps 16, guidance 3.0,
    3-step. Same card serves both the 1.3B and 14B-480P ids (size resolved from the checkpoint).

    BRINGUP: these DMD checkpoints are VSA-trained (kernel NOT built here) AND carry gated-attention
    params (``to_gate_compress``) the generic Wan loader cannot map -> a v2-side non-strict loader
    option is required. The recipe (DMD loop + geometry) is correct; the LOAD is the bringup gap."""
    return _build_fastwan_card(model_id,
                               latent_channels=16,
                               spatial_ratio=8,
                               temporal_ratio=4,
                               sampling_defaults=SamplingDefaults(num_steps=3,
                                                                  guidance_scale=3.0,
                                                                  height=448,
                                                                  width=832,
                                                                  num_frames=61,
                                                                  fps=16,
                                                                  negative_prompt=WAN_NEG_EN),
                               checkpoint_root=checkpoint_root)
