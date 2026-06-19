"""HY-WorldPlay-Bidirectional ModelCard: interactive world model on the v2 substrate.

HY-WorldPlay is a chunk-rollout world model: it generates video in temporal chunks where each chunk
after the first retrieves camera-aligned history "memory" frames as frozen context, and conditions on
three text/image streams (Qwen2.5-VL mllm + ByT5 glyph + SigLIP image) plus per-frame camera
(viewmats/Ks, injected via ProPE attention) and per-frame action ids. The registered preset is the
bidirectional t2v / degenerate (no-action, single straight-trajectory pose) path that CPU-verifies;
the full action/camera/memory conditioning is BRINGUP (needs a request-API extension to carry the pose
string -> viewmats/Ks/action expansion + point-cloud memory retrieval). See ``loop.py`` and
``torch_hyworld.py`` for the BRINGUP markers.

Architecture deltas vs Wan/Cosmos (all declared on the card so the recipe is self-contained):
  * DiT  ``fastvideo.models.dits.hyworld.hyworld:HYWorldTransformer3DModel`` (subclasses
    HunyuanVideo15Transformer3DModel) — a rectified-flow velocity predictor whose forward takes a
    65ch pre-concatenated latent ``[noise(32) | cond(32) + mask(1)]``, a list of text embeds
    ``[qwen, byt5]``, a list image embed ``[siglip]``, a per-latent-frame LongTensor ``timestep`` (+ a
    scalar ``timestep_txt``), and ``viewmats/Ks/action`` per-frame camera/action conditioning. The
    adapter (``HYWorldDiT``) marshals all of this internally so the loop only hands it
    ``dit(latent, text_embed, sigma)`` (the cosmos2/ToyDiT-compatible signature).
  * VAE  ``fastvideo.models.vaes.hyworldvae:AutoencoderKLHYWorld`` (z=32, 16x spatial, 4x temporal,
    scaling_factor=1.03682) — encode -> ``.mode() * scaling_factor`` (no shift/mean), decode inverts.
  * Encoders: Qwen2.5-VL (mllm primary text) + ByT5/T5 (glyph) + SigLIP (image). For the t2v preset the
    glyph + image streams are zeroed (the DiT detects an all-zero image and masks it).
  * Sampler: flow-match (``FlowMatchEulerDiscreteScheduler``), but the preset overrides the schedule
    with explicit sigmas ``linspace(1.0, 0.0, 51)[:-1]`` (50 steps), flow_shift 5.0, guidance 6.0.
``stamp_wan21_checkpoints`` applies (diffusers transformer/vae/text_encoder/text_encoder_2/image_encoder
subfolder layout); the superset stamp helper already covers ``text_encoder_2`` + ``image_encoder``.
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
from v2.loop.policies import ClassicCFG, FlowShiftPolicy, NoRouting, PrecisionPolicy
from v2.parallel import ParallelPlan
from v2.platform.backends.toy import ToyDiT, ToyImageEncoder, ToyTextEncoder, ToyVAE, _seed_from
from v2.recipes.hyworld.loop import (
    HYWORLD_CHUNK_LATENT_FRAMES,
    HYWORLD_LATENT_CHANNELS,
    HYWORLD_SPATIAL_RATIO,
    HYWORLD_STABILIZATION_LEVEL,
    HYWORLD_TEMPORAL_RATIO,
    HYWorldDenoiseLoop,
)
from v2.recipes.wan21.card import stamp_wan21_checkpoints

# Adapter refs declared on the card -> built lazily on a GPU box by ``_explicit_adapter`` (so the port
# never edits the shared ``_make_dit``/``_make_vae``/``_make_text_encoder``/``_make_image_encoder``).
_HYWORLD_DIT = "v2.platform.backends.torch_hyworld:HYWorldDiT"
_HYWORLD_VAE = "v2.platform.backends.torch_hyworld:HYWorldVAE"
_HYWORLD_QWEN = "v2.platform.backends.torch_hyworld:HYWorldQwenEncoder"
_HYWORLD_BYT5 = "v2.platform.backends.torch_hyworld:HYWorldByT5Encoder"
_HYWORLD_SIGLIP = "v2.platform.backends.torch_hyworld:HYWorldSiglipEncoder"

# The preset's negative prompt is empty (kept as a LOCAL module constant, not in the shared _prompts.py).
HYWORLD_NEG = ""

# The preset's explicit sigma schedule: linspace(1.0, 0.0, 51)[:-1] (50 values). The FlowShiftPolicy /
# the loop consume ``sigmas`` directly when present (the flow_shift below is the schedule fallback).
_HYWORLD_SIGMAS = tuple(1.0 - i / 50.0 for i in range(50))


def build_hyworld_card(model_id: str = "hy-worldplay-bidirectional",
                       *,
                       checkpoint_root: str | None = None,
                       sampling_defaults: SamplingDefaults | None = None) -> ModelCard:
    seed = _seed_from(model_id)
    cost = CostModel(kind=WorkUnitKind.DIFFUSION_STEP, base_seconds=1e-4, per_unit_seconds=1e-7)
    cfg = ClassicCFG()  # uncond + s·(cond - uncond); the pos/neg branches swap the encoder_attention_mask
    # flow_shift=5.0 is the scheduler shift; the preset overrides the schedule with explicit sigmas, so
    # FlowShiftPolicy.build_schedule(sigmas=...) returns those verbatim (shift becomes a no-op there).
    flow = FlowShiftPolicy(shift=5.0)
    precision = PrecisionPolicy(compute_dtype="float32", scheduler_step_in_fp32=True)
    expert = NoRouting("transformer")  # single DiT (the bidirectional model is not MoE)

    def loop_factory() -> HYWorldDenoiseLoop:
        return HYWorldDenoiseLoop(loop_id="hyworld_denoise",
                                  cfg=cfg,
                                  flow_shift=flow,
                                  precision=precision,
                                  expert=expert,
                                  cost=cost,
                                  latent_channels=HYWORLD_LATENT_CHANNELS,
                                  spatial_ratio=HYWORLD_SPATIAL_RATIO,
                                  temporal_ratio=HYWORLD_TEMPORAL_RATIO,
                                  chunk_latent_frames=HYWORLD_CHUNK_LATENT_FRAMES,
                                  stabilization_level=HYWORLD_STABILIZATION_LEVEL)

    components = {
        # Primary text stream — Qwen2.5-VL mllm (hidden_states[-3], crops 108 template tokens on GPU).
        "text_encoder":
        ComponentSpec(component_id="text_encoder",
                      kind="text_encoder",
                      load_id="fastvideo.models.encoders.qwen2_5:Qwen2_5_VLTextModel",
                      adapter=_HYWORLD_QWEN,
                      factory=lambda inst: ToyTextEncoder(),
                      required_for={"t2v"}),
        # Glyph text stream — ByT5/T5 (rendered-text encoder, text_states_dim_2=1472). Zeroed for plain t2v.
        "text_encoder_2":
        ComponentSpec(component_id="text_encoder_2",
                      kind="text_encoder",
                      load_id="fastvideo.models.encoders.t5_hf:T5EncoderModel",
                      adapter=_HYWORLD_BYT5,
                      factory=lambda inst: ToyTextEncoder(),
                      optional_for={"t2v"}),
        # Image stream — SigLIP (729 tokens x 1152 dim). Zeroed for t2v -> the DiT masks the image stream.
        "image_encoder":
        ComponentSpec(component_id="image_encoder",
                      kind="image_encoder",
                      load_id="fastvideo.models.encoders.siglip:SiglipVisionModel",
                      adapter=_HYWORLD_SIGLIP,
                      factory=lambda inst: ToyImageEncoder(),
                      optional_for={"t2v"}),
        "vae":
        ComponentSpec(component_id="vae",
                      kind="vae",
                      load_id="fastvideo.models.vaes.hyworldvae:AutoencoderKLHYWorld",
                      adapter=_HYWORLD_VAE,
                      factory=lambda inst: ToyVAE(),
                      required_for={"t2v"}),
        "transformer":
        ComponentSpec(component_id="transformer",
                      kind="dit",
                      load_id="fastvideo.models.dits.hyworld.hyworld:HYWorldTransformer3DModel",
                      adapter=_HYWORLD_DIT,
                      factory=lambda inst: ToyDiT(seed=seed),
                      resident_for=["hyworld_denoise"],
                      required_for={"t2v"}),
    }
    loops = {
        "hyworld_denoise":
        LoopSpec(
            loop_id="hyworld_denoise",
            kind=LoopKind.DIFFUSION_DENOISE,
            work_unit_kind=WorkUnitKind.DIFFUSION_STEP,
            step_cost_model=cost,
            shared_weight_components=["transformer"],
            cache_policy=["feature"],
            graph_capture="eager",  # chunk rollout + camera-aligned memory retrieval -> eager path
            loop_factory=loop_factory),
    }
    card = ModelCard(
        model_id=model_id,
        family="hyworld",
        components=components,
        loops=loops,
        capabilities=CapabilityMatrix.of(
            Capability.TEXT_TO_VIDEO,  # the registered (CPU-verified) degenerate path
            Capability.IMAGE_TO_VIDEO,  # first-frame SigLIP + VAE cond latent (BRINGUP)
            Capability.ACTION_CONDITIONING,  # per-frame action ids (BRINGUP)
            Capability.STREAMING_VIDEO_CONTINUATION,  # chunk rollout = interactive continuation (BRINGUP)
            Capability.VAE_DECODE,
            Capability.POLICY_ROLLOUT),
        recipe=RecipeSpec(method="base",
                          assumes_loop="hyworld_denoise",
                          assumes_precision="float32",
                          consistency_required=ConsistencyLevel.C1),
        parity=ParitySpec(consistency_levels=[ConsistencyLevel.C1],
                          interleave_required=True,
                          tests=[ParityTestSpec(name="denoise_trajectory", level=ConsistencyLevel.C1, tap="latents")]),
        caches={"feature": CacheContract(cache_class="feature", max_bytes=1 << 24, reuse_across_requests=True)},
        precision=PrecisionContract(default_dtype="float32", training_precision="float32"),
        parallelism=ParallelismContract(valid_plans=[ParallelPlan.single()], default_plan=ParallelPlan.single()),
        sampling_defaults=sampling_defaults or SamplingDefaults(num_steps=50,
                                                                guidance_scale=6.0,
                                                                height=480,
                                                                width=832,
                                                                num_frames=125,
                                                                fps=24,
                                                                negative_prompt=HYWORLD_NEG,
                                                                shift=5.0,
                                                                sigmas=_HYWORLD_SIGMAS),
    )
    card.validate()
    if checkpoint_root:
        # Diffusers layout: transformer/ vae/ text_encoder/ text_encoder_2/ image_encoder/ scheduler/.
        # The Wan stamp superset already covers text_encoder_2 + image_encoder (BRINGUP risk A).
        stamp_wan21_checkpoints(card, checkpoint_root)
    return card
