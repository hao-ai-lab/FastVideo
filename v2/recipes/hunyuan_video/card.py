"""HunyuanVideo ModelCard — text->video (the registered preset). Flow-match rectified-flow video DiT.

Architecture deltas vs Wan (all declared on the card so the recipe is self-contained):
  * DiT  ``fastvideo.models.dits.hunyuanvideo:HunyuanVideoTransformer3DModel`` — velocity flow-match (like
    Wan), but ``encoder_hidden_states`` is a 2-element list ``[llama_hidden[B,L,4096], clip_pooled[B,768]]``.
    The adapter (``ComponentSpec.adapter`` -> ``HunyuanVideoDiT``) assembles that pair internally; the loop
    reuses ``WanDenoiseLoop`` math verbatim (timestep ``sigma*1000``, scalar per-batch, bare velocity out).
  * VAE  ``fastvideo.models.vaes.hunyuanvae:AutoencoderKLHunyuanVideo`` — scalar ``scaling_factor`` (0.476986)
    normalization, NOT Wan's per-channel mean/std -> the ``HunyuanVideoVAE`` adapter.
  * Text DUAL encoder: ``fastvideo.models.encoders.llama:LlamaModel`` (per-token sequence; prompt template +
    hidden-state-skip + crop) via ``HunyuanVideoLlamaEncoder``, and
    ``fastvideo.models.encoders.clip:CLIPTextModel`` (768-dim pooled) via ``HunyuanVideoCLIPEncoder``.

``stamp_wan21_checkpoints`` applies (diffusers transformer/vae/text_encoder subfolder layout); HunyuanVideo
additionally ships ``text_encoder_2/`` (CLIP) + ``tokenizer_2/`` — the stamp helper points each declared
component (incl. ``text_encoder_2``) at its subfolder (see the BRINGUP note below and the CLIP adapter for
the ``tokenizer_2`` wiring caveat).

Defaults (fastvideo HUNYUAN_T2V preset): 50 steps, guidance_scale 1.0 (CFG effectively off), 720x1280,
125 frames @ 24fps, flow_shift 7.0. A FastHunyuan variant ships shift 17, 6 steps.
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
from v2.platform.backends.toy import ToyDiT, ToyTextEncoder, ToyVAE, _seed_from
from v2.recipes.hunyuan_video.loop import HunyuanDenoiseLoop
from v2.recipes.wan21.card import stamp_wan21_checkpoints

_HUNYUAN_DIT = "v2.recipes.hunyuan_video.adapter:HunyuanVideoDiT"
_HUNYUAN_VAE = "v2.recipes.hunyuan_video.adapter:HunyuanVideoVAE"
_HUNYUAN_LLAMA = "v2.recipes.hunyuan_video.adapter:HunyuanVideoLlamaEncoder"
_HUNYUAN_CLIP = "v2.recipes.hunyuan_video.adapter:HunyuanVideoCLIPEncoder"

# Base HunyuanVideo runs guidance_scale=1.0 (CFG effectively off), so it has no published negative prompt;
# a LOCAL module constant so the recipe owns its own data (empty string = no negative conditioning).
HUNYUAN_NEG = ""


def build_hunyuan_video_card(model_id: str = "hunyuanvideo",
                             *,
                             flow_shift: float = 7.0,
                             num_steps: int = 50,
                             checkpoint_root: str | None = None,
                             sampling_defaults: SamplingDefaults | None = None) -> ModelCard:
    seed = _seed_from(model_id)
    cost = CostModel(kind=WorkUnitKind.DIFFUSION_STEP, base_seconds=1e-4, per_unit_seconds=1e-7)
    cfg = ClassicCFG()
    # No resolution-bucket lookup: HunyuanVideo uses a single flow_shift (7.0 base / 17.0 FastHunyuan).
    flow = FlowShiftPolicy(shift=flow_shift)
    precision = PrecisionPolicy(compute_dtype="float32", scheduler_step_in_fp32=True)
    expert = NoRouting("transformer")

    def loop_factory() -> HunyuanDenoiseLoop:
        return HunyuanDenoiseLoop(loop_id="diffusion_denoise",
                                  cfg=cfg,
                                  flow_shift=flow,
                                  precision=precision,
                                  expert=expert,
                                  cost=cost)  # 16/8/4 default geometry

    components = {
        "text_encoder":
        ComponentSpec(component_id="text_encoder",
                      kind="text_encoder",
                      load_id="fastvideo.models.encoders.llama:LlamaModel",
                      adapter=_HUNYUAN_LLAMA,
                      factory=lambda inst: ToyTextEncoder(),
                      required_for={"t2v"}),
        "text_encoder_2":
        ComponentSpec(component_id="text_encoder_2",
                      kind="text_encoder",
                      load_id="fastvideo.models.encoders.clip:CLIPTextModel",
                      adapter=_HUNYUAN_CLIP,
                      factory=lambda inst: ToyTextEncoder(),
                      required_for={"t2v"}),
        "vae":
        ComponentSpec(component_id="vae",
                      kind="vae",
                      load_id="fastvideo.models.vaes.hunyuanvae:AutoencoderKLHunyuanVideo",
                      adapter=_HUNYUAN_VAE,
                      factory=lambda inst: ToyVAE(),
                      required_for={"t2v"}),
        "transformer":
        ComponentSpec(component_id="transformer",
                      kind="dit",
                      load_id="fastvideo.models.dits.hunyuanvideo:HunyuanVideoTransformer3DModel",
                      adapter=_HUNYUAN_DIT,
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
        family="hunyuan",
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
        sampling_defaults=sampling_defaults or SamplingDefaults(num_steps=num_steps,
                                                                guidance_scale=1.0,
                                                                height=720,
                                                                width=1280,
                                                                num_frames=125,
                                                                fps=24,
                                                                negative_prompt=HUNYUAN_NEG,
                                                                shift=flow_shift),
    )
    card.validate()
    if checkpoint_root:
        stamp_wan21_checkpoints(card, checkpoint_root)
        # HunyuanVideo's CLIP secondary encoder lives in text_encoder_2/ (the Wan stamp covers it because
        # text_encoder_2 is in its subfolder superset); its tokenizer is tokenizer_2/ (BRINGUP: see the CLIP
        # adapter — the shared _make_text_encoder loads tokenizer/, so GPU-verify must wire tokenizer_2).
        if "text_encoder_2" in card.components and not card.components["text_encoder_2"].checkpoint:
            root = os.path.dirname(os.path.normpath(card.components["transformer"].checkpoint))
            card.components["text_encoder_2"].checkpoint = os.path.join(root, "text_encoder_2")
    return card


def build_fast_hunyuan_video_card(model_id: str = "fasthunyuan", *, checkpoint_root: str | None = None) -> ModelCard:
    """FastHunyuan (FastVideo/FastHunyuan-diffusers): same arch, distilled to 6 steps with flow_shift 17.0.
    Reuses the base card / adapters unchanged — only the schedule (shift + step count) differs."""
    return build_hunyuan_video_card(model_id=model_id,
                                    flow_shift=17.0,
                                    num_steps=6,
                                    checkpoint_root=checkpoint_root,
                                    sampling_defaults=SamplingDefaults(num_steps=6,
                                                                       guidance_scale=1.0,
                                                                       height=720,
                                                                       width=1280,
                                                                       num_frames=125,
                                                                       fps=24,
                                                                       negative_prompt=HUNYUAN_NEG,
                                                                       shift=17.0))
