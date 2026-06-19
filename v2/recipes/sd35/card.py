"""Stable Diffusion 3.5 Medium ModelCard — text→image (the registered preset). Image is treated as a
1-frame video (num_frames=1, 4D latents) throughout the v2 substrate.

Architecture deltas vs Wan (all declared on the card so the recipe is self-contained):
  * DiT  ``fastvideo.models.dits.sd3:SD3Transformer2DModel`` — a FLOW-MATCH MMDiT, but the forward needs
    TWO text conditioners: the assembled triple-encoder joint embed (``encoder_hidden_states``) AND a
    separate ``pooled_projections`` vector. The ``SD3DiT`` adapter takes both (joint embed as
    ``text_embed``, pooled as the positional ``context`` arg) and ``SD3DenoiseLoop`` threads them per CFG branch.
  * VAE  ``fastvideo.models.vaes.autoencoder_kl:AutoencoderKL`` — SD3.5 shift/scale normalization
    (``scaling_factor=1.5305``, ``shift_factor=0.0609`` read from the checkpoint vae config); the
    ``SD3VAE`` adapter normalizes/denormalizes around the raw AutoencoderKL space (4D image, not 5D video).
  * Text x3: ``CLIPTextModelWithProjection`` (clip_l + clip_g, penultimate-hidden + pooled) +
    ``T5EncoderModel`` (last_hidden_state). The program assembles the joint embed + dual-CLIP pooled.

Schedule: plain flow-match linspace with the diffusers ``FlowMatchEulerDiscreteScheduler`` config
``shift`` (3.0 for SD3.5-medium). The scheduler's optional ``use_dynamic_shifting`` (resolution-dependent
``mu``) is NOT applied here — if a checkpoint's scheduler config enables it, the static shift will mismatch
the reference schedule (BRINGUP blocker 6). CFG is the classic ``uncond + s·(cond − uncond)``.
"""
from __future__ import annotations

import os

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
from v2.platform.backends.toy import ToyTextEncoder, ToyVAE, _seed_from
from v2.recipes.sd35.loop import SD3DenoiseLoop

# SD3.5's default negative prompt is empty (LOCAL module constant — the spec's defaults.negative_prompt).
SD35_NEG = ""

# SD3.5-medium diffusers FlowMatchEulerDiscreteScheduler static flow shift (BRINGUP blocker 8: read the
# real scheduler config at deploy time; SD35Config.flow_shift is None, the shift lives in the scheduler).
SD35_FLOW_SHIFT = 3.0

_SD3_DIT = "v2.recipes.sd35.adapter:SD3DiT"
_SD3_VAE = "v2.recipes.sd35.adapter:SD3VAE"
_SD3_CLIP = "v2.recipes.sd35.adapter:SD3ClipEncoder"
_SD3_T5 = "v2.recipes.sd35.adapter:SD3T5Encoder"


def build_sd35_card(model_id: str = "stable-diffusion-3.5-medium",
                    *,
                    checkpoint_root: str | None = None,
                    flow_shift: float = SD35_FLOW_SHIFT,
                    sampling_defaults: SamplingDefaults | None = None) -> ModelCard:
    seed = _seed_from(model_id)
    cost = CostModel(kind=WorkUnitKind.DIFFUSION_STEP, base_seconds=1e-4, per_unit_seconds=1e-7)
    cfg = ClassicCFG()
    flow = FlowShiftPolicy(shift=flow_shift)  # static shift (no resolution-bucket lookup for SD3.5)
    precision = PrecisionPolicy(compute_dtype="float32", scheduler_step_in_fp32=True)
    expert = NoRouting("transformer")

    def loop_factory():
        return SD3DenoiseLoop(loop_id="diffusion_denoise",
                              cfg=cfg,
                              flow_shift=flow,
                              precision=precision,
                              expert=expert,
                              cost=cost)

    components = {
        # The two CLIP encoders (clip_l, clip_g) — penultimate-hidden + pooled. The third is T5.
        "text_encoder":
        ComponentSpec(component_id="text_encoder",
                      kind="text_encoder",
                      load_id="fastvideo.models.encoders.clip:CLIPTextModelWithProjection",
                      adapter=_SD3_CLIP,
                      factory=lambda inst: ToyTextEncoder(),
                      required_for={"t2i"}),
        "text_encoder_2":
        ComponentSpec(component_id="text_encoder_2",
                      kind="text_encoder",
                      load_id="fastvideo.models.encoders.clip:CLIPTextModelWithProjection",
                      adapter=_SD3_CLIP,
                      factory=lambda inst: ToyTextEncoder(),
                      required_for={"t2i"}),
        "text_encoder_3":
        ComponentSpec(component_id="text_encoder_3",
                      kind="text_encoder",
                      load_id="fastvideo.models.encoders.t5:T5EncoderModel",
                      adapter=_SD3_T5,
                      factory=lambda inst: ToyTextEncoder(),
                      required_for={"t2i"}),
        "vae":
        ComponentSpec(component_id="vae",
                      kind="vae",
                      load_id="fastvideo.models.vaes.autoencoder_kl:AutoencoderKL",
                      adapter=_SD3_VAE,
                      factory=lambda inst: ToyVAE(),
                      required_for={"t2i"}),
        "transformer":
        ComponentSpec(component_id="transformer",
                      kind="dit",
                      load_id="fastvideo.models.dits.sd3:SD3Transformer2DModel",
                      adapter=_SD3_DIT,
                      factory=lambda inst: _toy_dit(seed),
                      resident_for=["diffusion_denoise"],
                      required_for={"t2i"}),
    }
    loops = {
        "diffusion_denoise":
        LoopSpec(loop_id="diffusion_denoise",
                 kind=LoopKind.DIFFUSION_DENOISE,
                 work_unit_kind=WorkUnitKind.DIFFUSION_STEP,
                 step_cost_model=cost,
                 shared_weight_components=["transformer"],
                 cache_policy=["feature"],
                 graph_capture="eager",
                 loop_factory=loop_factory),
    }
    card = ModelCard(
        model_id=model_id,
        family="sd3",
        components=components,
        loops=loops,
        capabilities=CapabilityMatrix.of(Capability.TEXT_TO_IMAGE, Capability.VAE_DECODE, Capability.POLICY_ROLLOUT),
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
        sampling_defaults=sampling_defaults or SamplingDefaults(
            num_steps=28, guidance_scale=6.0, height=512, width=512, num_frames=1, fps=1, negative_prompt=SD35_NEG),
    )
    card.validate()
    if checkpoint_root:
        stamp_sd35_checkpoints(card, checkpoint_root)
    return card


def _toy_dit(seed: int):
    """CPU toy DiT for the transformer component. ToyDiT.__call__ accepts the positional ``context`` arg
    (it mean-pools it harmlessly), so the dual-conditioning ``dit(latent, joint_embed, sigma, context=pooled)``
    call CPU-verifies without any SD3-specific stand-in."""
    from v2.platform.backends.toy import ToyDiT
    return ToyDiT(seed=seed)


# Diffusers SD3.5 checkpoint layout: transformer/ + vae/ + text_encoder{,_2,_3}/ + tokenizer{,_2,_3}/.
# Distinct from ``_WAN21_SUBFOLDERS`` (one text_encoder) — SD3 carries THREE text encoders. NOTE: the
# torch backend's ``_make_text_encoder`` loads the tokenizer from a FIXED ``<root>/tokenizer`` sibling, so
# text_encoder_2/_3 would load the wrong tokenizer (BRINGUP blocker: needs the per-encoder tokenizer
# subfolder threaded into the maker, or a small adapter-side override at GPU bring-up).
_SD35_SUBFOLDERS = {
    "transformer": "transformer",
    "vae": "vae",
    "text_encoder": "text_encoder",
    "text_encoder_2": "text_encoder_2",
    "text_encoder_3": "text_encoder_3",
}


def stamp_sd35_checkpoints(card: ModelCard, model_root: str) -> ModelCard:
    """GPU deploy-time: point each component's ``ComponentSpec.checkpoint`` at its weights subfolder under
    ``model_root`` (a local diffusers dir, or an HF id resolved via ``snapshot_download``). Mirrors
    ``stamp_wan21_checkpoints`` but covers the THREE SD3.5 text encoders. Mutates and returns the card."""
    if not os.path.isdir(model_root):
        from huggingface_hub import snapshot_download
        model_root = snapshot_download(model_root)
    for component_id, subfolder in _SD35_SUBFOLDERS.items():
        if component_id in card.components:
            card.components[component_id].checkpoint = os.path.join(model_root, subfolder)
    return card
