"""GEN3C-Cosmos-7B ModelCard — camera-controlled video diffusion (EDM denoiser).

GEN3C extends Cosmos-Predict2's EDM denoiser with camera/3D-cache conditioning. Architecture deltas vs
Cosmos (all declared on the card so the recipe is self-contained):
  * DiT  ``fastvideo.models.dits.gen3c:Gen3CTransformer3DModel`` — an EDM denoiser whose forward concats
    ``[latent(16) | input_mask(1) | pose_buffer(frame_buffer_max·32) | padding_mask(1)] = 82`` channels
    INTERNALLY from separate kwargs. The adapter (``ComponentSpec.adapter`` -> ``Gen3CDiT``) returns the
    raw network output and ``Gen3CDenoiseLoop`` does the EDM ``c_in/c_skip/c_out`` → x0 reconstruction +
    x0-space CFG (pose-zeroed uncond) + frame-replace conditioning + the forced-EDMEulerScheduler σ
    schedule (Karras ρ=7, σ_max=80 → σ_min=0.0002, terminal 0.0). Model timestep is ``c_noise = 0.25·log
    σ`` (NOT σ·1000, NOT raw σ).
  * VAE  ``fastvideo.models.vaes.gen3c_tokenizer_vae:AutoencoderKLGen3CTokenizer`` — a JIT-backed
    tokenizer that handles its OWN latent normalization (passthrough/internal mean_std buffers); the
    ``Gen3CVAE`` adapter does NOT apply an external (z-mean)/std. 8× temporal (121→16 latent frames), 8×
    spatial.
  * Text ``fastvideo.models.encoders.t5:T5EncoderModel`` (T5-Large, 1024-dim) via ``Gen3CT5Encoder``
    (raw last_hidden_state, no Wan zero-pad — same as Cosmos).
``stamp_wan21_checkpoints`` applies (diffusers transformer/vae/text_encoder subfolder layout).

BRINGUP (see ``program.py`` / ``torch_gen3c.py``): the camera-trajectory / MoGe-depth / 3D-cache-render
conditioning that fills ``condition_video_pose`` is a separate pre-loop stage needing a CUDA point-cloud
renderer + request-API extension (image_path + trajectory fields). The registered card is the degenerate
**t2v** path (zero conditioning) — it CPU-verifies and the GPU loop runs end-to-end without the camera
stack; with conditioning it becomes image+camera → video. The JIT tokenizer loader seam also needs
GPU-box verification (``from_jit_tokenizer`` expects encoder.jit/decoder.jit/mean_std.pt).
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
from v2.loop.policies import ClassicCFG, NoRouting, PrecisionPolicy
from v2.parallel import ParallelPlan
from v2.platform.backends.toy import ToyDiT, ToyTextEncoder, ToyVAE, _seed_from
from v2.recipes.gen3c.loop import Gen3CDenoiseLoop
from v2.recipes.wan21.card import stamp_wan21_checkpoints

_GEN3C_DIT = "v2.recipes.gen3c.adapter:Gen3CDiT"
_GEN3C_VAE = "v2.recipes.gen3c.adapter:Gen3CVAE"
_GEN3C_T5 = "v2.recipes.gen3c.adapter:Gen3CT5Encoder"

# The GEN3C default negative prompt (kept LOCAL to this recipe — the v2/recipes/_prompts.py shared bank
# is off-limits for a self-contained port). Used at guidance_scale > 1 (the official_uncond_at_unity
# policy); the default guidance_scale=1.0 + cfg_behavior='legacy' runs NO uncond branch.
GEN3C_NEG = ("The video captures a series of frames showing ugly scenes, static with no motion, motion blur, "
             "over-saturation, shaky footage, low resolution, grainy texture, pixelated images, poorly lit areas, "
             "underexposed and overexposed scenes, poor color balance, washed out colors, choppy sequences, jerky "
             "movements, low frame rate, artifacting, color banding, unnatural transitions, outdated special "
             "effects, fake elements, unconvincing visuals, poorly edited content, jump cuts, visual noise, and "
             "flickering. Overall, the video is of poor quality.")


def build_gen3c_card(model_id: str = "gen3c-cosmos-7b",
                     *,
                     checkpoint_root: str | None = None,
                     sampling_defaults: SamplingDefaults | None = None) -> ModelCard:
    seed = _seed_from(model_id)
    cost = CostModel(kind=WorkUnitKind.DIFFUSION_STEP, base_seconds=1e-4, per_unit_seconds=1e-7)
    cfg = ClassicCFG()
    precision = PrecisionPolicy(compute_dtype="float32", scheduler_step_in_fp32=True)
    expert = NoRouting("transformer")

    def loop_factory():
        return Gen3CDenoiseLoop(loop_id="diffusion_denoise",
                                cfg=cfg,
                                precision=precision,
                                expert=expert,
                                cost=cost,
                                sigma_max=80.0,
                                sigma_min=0.0002,
                                sigma_data=0.5,
                                sigma_conditional=0.001)

    components = {
        "text_encoder":
        ComponentSpec(component_id="text_encoder",
                      kind="text_encoder",
                      load_id="fastvideo.models.encoders.t5:T5EncoderModel",
                      adapter=_GEN3C_T5,
                      factory=lambda inst: ToyTextEncoder(),
                      required_for={"t2v"}),
        "vae":
        ComponentSpec(component_id="vae",
                      kind="vae",
                      load_id="fastvideo.models.vaes.gen3c_tokenizer_vae:AutoencoderKLGen3CTokenizer",
                      adapter=_GEN3C_VAE,
                      factory=lambda inst: ToyVAE(),
                      required_for={"t2v"}),
        "transformer":
        ComponentSpec(component_id="transformer",
                      kind="dit",
                      load_id="fastvideo.models.dits.gen3c:Gen3CTransformer3DModel",
                      adapter=_GEN3C_DIT,
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
        family="gen3c",
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
        # GEN3C defaults (the FORCED EDMEulerScheduler config + the registered preset): 35 steps,
        # guidance 1.0 (legacy cfg -> no uncond branch), 704x1280, 121 frames @ 24fps.
        sampling_defaults=sampling_defaults or SamplingDefaults(
            num_steps=35, guidance_scale=1.0, height=704, width=1280, num_frames=121, fps=24,
            negative_prompt=GEN3C_NEG),
    )
    card.validate()
    if checkpoint_root:
        stamp_wan21_checkpoints(card, checkpoint_root)
    return card
