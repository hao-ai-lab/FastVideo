"""FLUX.2 ModelCard(s) — text→image (BFL FLUX.2-dev primary; klein 4B/9B distilled variants).

Architecture deltas vs Wan (all declared on the card so the recipe is self-contained — the bucket-C
pattern: torch adapters via ``ComponentSpec.adapter``, a NEW ``Flux2DenoiseLoop``, ``stamp_wan21_checkpoints``
for the diffusers transformer/vae/text_encoder subfolder layout):

  * DiT  ``fastvideo.models.dits.flux_2:Flux2Transformer2DModel`` — dual-stream MMDiT (19 double + 38
    single blocks, inner_dim 3072, 24 heads, N-D RoPE axes (32,32,32,32) θ=2000). The adapter
    (``Flux2DiT``) builds img_ids/txt_ids RoPE position tensors from the latent shape, passes σ DIRECTLY
    as the timestep and the embedded guidance RAW (the DiT multiplies both by 1000 internally), wraps the
    text embed in a 1-element list, and returns the bare velocity. NO classic CFG (embedded guidance).
  * VAE  ``fastvideo.models.vaes.flux2vae:AutoencoderKLFlux2`` — packed 2×2 latents + BatchNorm whitening
    (NOT the Wan (z−mean)/std). ``Flux2VAE`` decode does BN-denorm (``z·bn_std + bn_mean``) → 2×2 unpatchify
    (64→16ch, full spatial) → ``module.decode`` → image. (Image-only: T==1, squeezed.)
  * Text dev=``mistral3:Mistral3ForConditionalGeneration`` (layers 10/20/30, FLUX2 system-message chat
    template); klein=``qwen3:Qwen3ForCausalLM`` (layers 9/18/27). ``Flux2TextEncoder`` stacks the chosen
    hidden-state layers into a [seq, 3·hidden=4096] embedding — NOT T5 last_hidden_state.

GATED weights (``black-forest-labs/FLUX.2-*``) → GPU is BRINGUP; the toy factories CPU-verify the recipe.
"""
from __future__ import annotations

from v2.core.enums import Capability, ConsistencyLevel, LoopKind, WorkUnitKind
from v2.core.card import (
    CacheContract,
    CapabilityMatrix,
    ComponentSpec,
    LoopSpec,
    ModelCard,
    ParallelismContract,
    ParitySpec,
    ParityTestSpec,
    PrecisionContract,
    RecipeSpec,
    SamplingDefaults,
)
from v2.core.loop.policies import EmbeddedGuidance, NoRouting, PrecisionPolicy
from v2.core.parallel import ParallelPlan
from v2.platform.backends.toy import ToyDiT, ToyTextEncoder, ToyVAE, _seed_from
from v2.recipes.flux2.loop import Flux2DenoiseLoop
from v2.recipes.wan21.card import stamp_wan21_checkpoints

# FLUX.2 conditions on a single embedded-guidance forward (no CFG), so there is no negative prompt — kept
# as a LOCAL module constant (the recipe is self-contained; nothing is added to the shared _prompts.py).
FLUX2_NEG: str | None = None

_FLUX2_DIT = "v2.recipes.flux2.adapter:Flux2DiT"
_FLUX2_VAE = "v2.recipes.flux2.adapter:Flux2VAE"
_FLUX2_MISTRAL3 = "v2.recipes.flux2.adapter:Flux2Mistral3Encoder"
_FLUX2_QWEN3 = "v2.recipes.flux2.adapter:Flux2Qwen3Encoder"


def _build_flux2_card(model_id: str, *, text_encoder_load_id: str, text_encoder_adapter: str,
                      checkpoint_root: str | None, sampling_defaults: SamplingDefaults) -> ModelCard:
    seed = _seed_from(model_id)
    cfg = EmbeddedGuidance()  # single conditioned forward; no uncond branch
    precision = PrecisionPolicy(compute_dtype="float32", scheduler_step_in_fp32=True)
    expert = NoRouting("transformer")

    def loop_factory():
        return Flux2DenoiseLoop(loop_id="diffusion_denoise", cfg=cfg, precision=precision, expert=expert)

    components = {
        "text_encoder":
        ComponentSpec(component_id="text_encoder",
                      kind="text_encoder",
                      load_id=text_encoder_load_id,
                      adapter=text_encoder_adapter,
                      factory=lambda inst: ToyTextEncoder(),
                      required_for={"t2i"}),
        "vae":
        ComponentSpec(component_id="vae",
                      kind="vae",
                      load_id="fastvideo.models.vaes.flux2vae:AutoencoderKLFlux2",
                      adapter=_FLUX2_VAE,
                      factory=lambda inst: ToyVAE(),
                      required_for={"t2i"}),
        "transformer":
        ComponentSpec(component_id="transformer",
                      kind="dit",
                      load_id="fastvideo.models.dits.flux_2:Flux2Transformer2DModel",
                      adapter=_FLUX2_DIT,
                      factory=lambda inst: ToyDiT(seed=seed),
                      resident_for=["diffusion_denoise"],
                      required_for={"t2i"}),
    }
    loops = {
        "diffusion_denoise":
        LoopSpec(loop_id="diffusion_denoise",
                 kind=LoopKind.DIFFUSION_DENOISE,
                 work_unit_kind=WorkUnitKind.DIFFUSION_STEP,
                 shared_weight_components=["transformer"],
                 cache_policy=["feature"],
                 graph_capture="breakable_cudagraph",
                 loop_factory=loop_factory),
    }
    card = ModelCard(
        model_id=model_id,
        family="flux2",
        components=components,
        loops=loops,
        capabilities=CapabilityMatrix.of(Capability.TEXT_TO_IMAGE, Capability.VAE_DECODE),
        recipe=RecipeSpec(method="base",
                          assumes_loop="diffusion_denoise",
                          assumes_precision="float32",
                          consistency_required=ConsistencyLevel.C1),
        parity=ParitySpec(consistency_levels=[ConsistencyLevel.C1],
                          tests=[ParityTestSpec(name="denoise_trajectory", level=ConsistencyLevel.C1, tap="latents")]),
        caches={"feature": CacheContract(cache_class="feature", max_bytes=1 << 24, reuse_across_requests=True)},
        precision=PrecisionContract(default_dtype="float32"),
        parallelism=ParallelismContract(valid_plans=[ParallelPlan.single()], default_plan=ParallelPlan.single()),
        sampling_defaults=sampling_defaults,
    )
    card.validate()
    if checkpoint_root:
        stamp_wan21_checkpoints(card, checkpoint_root)
    return card


def build_flux2_card(model_id: str = "flux2-dev", *, checkpoint_root: str | None = None) -> ModelCard:
    """FLUX.2-dev (the registered primary): Mistral3 text encoder (layers 10/20/30 via FLUX2 chat
    template), embedded guidance scale 4.0, 50 steps, 1024×1024. ``embedded_cfg_scale`` rides in
    ``guidance_scale``; ``do_classifier_free_guidance=False`` (no negative prompt)."""
    return _build_flux2_card(model_id,
                             text_encoder_load_id="fastvideo.models.encoders.mistral3:Mistral3ForConditionalGeneration",
                             text_encoder_adapter=_FLUX2_MISTRAL3,
                             checkpoint_root=checkpoint_root,
                             sampling_defaults=SamplingDefaults(num_steps=50,
                                                                guidance_scale=4.0,
                                                                height=1024,
                                                                width=1024,
                                                                num_frames=1,
                                                                fps=1,
                                                                negative_prompt=FLUX2_NEG))


def build_flux2_klein_card(model_id: str = "flux2-klein-4b", *, checkpoint_root: str | None = None) -> ModelCard:
    """FLUX.2-klein (4B/9B, distilled): Qwen3 text encoder (layers 9/18/27), NO guidance embedding
    (guidance_scale 1.0 -> single forward, no uncond), 4 steps, 1024×1024. The schedule still keys on the
    same empirical-mu grid; the 4-step bf16 parity is the GPU BRINGUP detail."""
    return _build_flux2_card(model_id,
                             text_encoder_load_id="fastvideo.models.encoders.qwen3:Qwen3ForCausalLM",
                             text_encoder_adapter=_FLUX2_QWEN3,
                             checkpoint_root=checkpoint_root,
                             sampling_defaults=SamplingDefaults(num_steps=4,
                                                                guidance_scale=1.0,
                                                                height=1024,
                                                                width=1024,
                                                                num_frames=1,
                                                                fps=1,
                                                                negative_prompt=FLUX2_NEG))
