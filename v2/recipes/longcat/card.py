"""LongCat-Video-T2V ModelCard — text->video, flow-match with CFG-zero guidance.

Architecture deltas vs Wan (all declared on the card so the recipe is self-contained):
  * DiT  ``fastvideo.models.dits.longcat:LongCatTransformer3DModel`` (depth 48, hidden 4096, 32 heads,
    in/out 16ch, patch [1,2,2], caption_channels 4096). The adapter (``LongCatDiT`` in
    ``v2/recipes/longcat/adapter.py``) returns the **negated** velocity (folding the fastvideo
    stage's ``noise_pred = -noise_pred`` before the scheduler step) so the loop's flow-match Euler matches.
  * VAE  ``fastvideo.models.vaes.wanvae:AutoencoderKLWan`` — Wan-style (z=16, 8x/4x) in normalized latent
    space (mean/std). Reuses the v2 ``WanVAE`` torch adapter (no adapter override).
  * Text ``fastvideo.models.encoders.t5:T5EncoderModel`` (UMT5, 4096-dim) via the v2 ``T5Encoder`` (zero-pads
    to max_length=512 for the CFG-concat uniform-seq contract — no adapter override needed).
  * Sampler: ``FlowMatchEulerDiscreteScheduler`` with an explicit ``linspace(1.0, 0.001, num_steps)`` sigma
    schedule (NOT flow-shift) + CFG-zero optimized-scale guidance — both live in ``LongCatDenoiseLoop``.

``stamp_wan21_checkpoints`` applies (the diffusers transformer/vae/text_encoder/tokenizer subfolder layout
is identical to Wan). i2v/VC (first-frame latent replacement + per-frame timestep masking + num_cond_latents),
the refine stages (t_thresh crop, the latent-prep branch that SKIPS the init-noise re-scale), KV-cache, and
BSA are deferred — the registered preset is base T2V.
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
from v2.loop.policies import NoRouting, PrecisionPolicy
from v2.parallel import ParallelPlan
from v2.platform.backends.toy import ToyDiT, ToyTextEncoder, ToyVAE, _seed_from
from v2.recipes.longcat.loop import CFGZeroPolicy, LongCatDenoiseLoop
from v2.recipes.wan21.card import stamp_wan21_checkpoints

# LongCat's default negative prompt (the inference yaml / get_sampling_param default — identical to the Wan
# English negative). Kept as a LOCAL module constant so the recipe touches no shared prompts file.
LONGCAT_NEG = ("Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, "
               "images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, "
               "incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, "
               "misshapen limbs, fused fingers, still picture, messy background, three legs, many people in "
               "the background, walking backwards")

_LONGCAT_DIT = "v2.recipes.longcat.adapter:LongCatDiT"


def build_longcat_card(model_id: str = "longcat-video-t2v",
                       *,
                       checkpoint_root: str | None = None,
                       sampling_defaults: SamplingDefaults | None = None) -> ModelCard:
    seed = _seed_from(model_id)
    cost = CostModel(kind=WorkUnitKind.DIFFUSION_STEP, base_seconds=1e-4, per_unit_seconds=1e-7)
    cfg = CFGZeroPolicy()
    # Keep the numpy loop math in fp32 (lowest-risk bring-up). The torch DiT runs at its native bf16 dtype
    # and casts its output to fp32 (spec blocker #6); the loop never forces fp32 on the module.
    precision = PrecisionPolicy(compute_dtype="float32", scheduler_step_in_fp32=True)
    expert = NoRouting("transformer")

    def loop_factory():
        return LongCatDenoiseLoop(loop_id="diffusion_denoise", cfg=cfg, precision=precision, expert=expert, cost=cost)

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
                      load_id="fastvideo.models.dits.longcat:LongCatTransformer3DModel",
                      adapter=_LONGCAT_DIT,
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
        family="longcat",
        components=components,
        loops=loops,
        capabilities=CapabilityMatrix.of(Capability.TEXT_TO_VIDEO, Capability.VAE_DECODE, Capability.POLICY_ROLLOUT),
        recipe=RecipeSpec(method="base",
                          assumes_loop="diffusion_denoise",
                          assumes_precision="float32",
                          consistency_required=ConsistencyLevel.C1),
        parity=ParitySpec(consistency_levels=[ConsistencyLevel.C1],
                          interleave_required=True,
                          tests=[ParityTestSpec(name="denoise_trajectory", level=ConsistencyLevel.C1, tap="latents")]),
        caches={"feature": CacheContract(cache_class="feature", max_bytes=1 << 24, reuse_across_requests=True)},
        precision=PrecisionContract(default_dtype="float32", training_precision="float32"),
        parallelism=ParallelismContract(valid_plans=[ParallelPlan.single()], default_plan=ParallelPlan.single()),
        # Spec defaults: 50 steps, guidance 1.0, 480x848, 121 frames @ 24fps. flow_shift=None (the loop builds
        # the linspace schedule); the negative prompt is the LongCat default above.
        sampling_defaults=sampling_defaults or SamplingDefaults(num_steps=50,
                                                                guidance_scale=1.0,
                                                                height=480,
                                                                width=848,
                                                                num_frames=121,
                                                                fps=24,
                                                                negative_prompt=LONGCAT_NEG),
    )
    card.validate()
    if checkpoint_root:
        stamp_wan21_checkpoints(card, checkpoint_root)
    return card
