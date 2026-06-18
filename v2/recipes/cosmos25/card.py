"""Cosmos-Predict2.5 ModelCard — text->video (the registered preset; video2world/image2world
conditioning is a later capability the loop already threads). Two sizes — 2B
(``KyleShao/Cosmos-Predict2.5-2B-Diffusers``) and 14B (``nvidia/Cosmos-Predict2.5-14B``) — share ONE
card: the transformer's ``num_attention_heads``/``num_layers`` are resolved from the checkpoint config by
the loader, so the card is size-agnostic and only the sampling defaults differ (we use the 2B preset).

Architecture deltas vs Wan (all GPU-path, declared on the card so the recipe is self-contained):
  * DiT  ``fastvideo.models.dits.cosmos2_5:Cosmos25Transformer3DModel`` — a flow-match velocity DiT whose
    adapter (``ComponentSpec.adapter`` -> ``Cosmos25DiT``) feeds a PER-FRAME plain-sigma timestep ``[B,T]``
    (NOT Wan's scalar ``sigma*1000``) + a mandatory zero ``condition_mask`` / ones ``padding_mask`` + an
    ``fps`` scalar. The model concats the masks itself (16 -> 18ch), so we feed the RAW 16ch latent.
  * VAE  ``fastvideo.models.vaes.cosmos25wanvae:Cosmos25WanVAE`` — Wan-style causal 3D VAE (z=16, 8x/4x)
    whose encode/decode normalize INTERNALLY (the Cosmos latent contract), so a dedicated
    ``Cosmos25WanVAE`` adapter marshals WITHOUT re-normalizing (reusing the v2 WanVAE adapter would
    double-normalize — see ``torch_cosmos25.py``).
  * Text ``fastvideo.models.encoders.reason1:Reason1TextEncoder`` — Qwen2.5-VL multimodal LM via
    ``Cosmos25Reason1Encoder`` (per-layer mean-normalized full-concat -> a 100352-dim sequence; the DiT's
    internal 100352->1024 GELU crossattn projection consumes it).
``stamp_wan21_checkpoints`` applies (diffusers transformer/vae/text_encoder subfolder layout + a sibling
tokenizer dir). Policies: ``ClassicCFG`` (embedded_cfg_scale=0 -> plain CFG), ``FlowShiftPolicy(shift=5.0)``,
fp32 scheduler step. Faithful to ``fastvideo/pipelines/stages/denoising.py:Cosmos25DenoisingStage``.
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
from v2.platform.backends.toy import ToyDiT, ToyTextEncoder, ToyVAE, _seed_from
from v2.recipes.cosmos25.loop import Cosmos25DenoiseLoop
from v2.recipes.wan21.card import stamp_wan21_checkpoints

# Cosmos-Predict2.5 default t2v negative prompt (a LOCAL recipe constant — the Cosmos2.5 preset's own
# quality-suppression prompt, encoded through the same Reason1 path for ClassicCFG).
_COSMOS25_NEG = ("The video captures a series of frames showing ugly scenes, static with no motion, motion blur, "
                 "over-saturation, shaky footage, low resolution, grainy texture, pixelated images, poorly lit areas, "
                 "underexposed and overexposed scenes, poor color balance, washed out colors, choppy sequences, jerky "
                 "movements, low frame rate, artifacting, color banding, unnatural transitions, outdated special "
                 "effects, fake elements, unconvincing visuals, poorly edited content, jump cuts, visual noise, and "
                 "flickering. Overall, the video is of poor quality.")

_COSMOS25_DIT = "v2.platform.backends.torch_cosmos25:Cosmos25DiT"
_COSMOS25_VAE = "v2.platform.backends.torch_cosmos25:Cosmos25WanVAE"
_COSMOS25_TEXT = "v2.platform.backends.torch_cosmos25:Cosmos25Reason1Encoder"


def build_cosmos25_card(model_id: str = "cosmos-predict2.5-2b",
                        *,
                        flow_shift: float = 5.0,
                        checkpoint_root: str | None = None,
                        sampling_defaults: SamplingDefaults | None = None) -> ModelCard:
    seed = _seed_from(model_id)
    cost = CostModel(kind=WorkUnitKind.DIFFUSION_STEP, base_seconds=1e-4, per_unit_seconds=1e-7)
    cfg = ClassicCFG()
    flow = FlowShiftPolicy(shift=flow_shift)
    precision = PrecisionPolicy(compute_dtype="float32", scheduler_step_in_fp32=True)
    expert = NoRouting("transformer")

    def loop_factory():
        return Cosmos25DenoiseLoop(loop_id="diffusion_denoise",
                                   cfg=cfg,
                                   flow_shift=flow,
                                   precision=precision,
                                   expert=expert,
                                   cost=cost)

    components = {
        "text_encoder":
        ComponentSpec(component_id="text_encoder",
                      kind="text_encoder",
                      load_id="fastvideo.models.encoders.reason1:Reason1TextEncoder",
                      adapter=_COSMOS25_TEXT,
                      factory=lambda inst: ToyTextEncoder(),
                      required_for={"t2v"}),
        "vae":
        ComponentSpec(component_id="vae",
                      kind="vae",
                      load_id="fastvideo.models.vaes.cosmos25wanvae:Cosmos25WanVAE",
                      adapter=_COSMOS25_VAE,
                      factory=lambda inst: ToyVAE(),
                      required_for={"t2v"}),
        "transformer":
        ComponentSpec(component_id="transformer",
                      kind="dit",
                      load_id="fastvideo.models.dits.cosmos2_5:Cosmos25Transformer3DModel",
                      adapter=_COSMOS25_DIT,
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
        family="cosmos25",
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
        sampling_defaults=sampling_defaults or SamplingDefaults(num_steps=35,
                                                                guidance_scale=7.0,
                                                                height=704,
                                                                width=1280,
                                                                num_frames=77,
                                                                fps=24,
                                                                negative_prompt=_COSMOS25_NEG),
    )
    card.validate()
    if checkpoint_root:
        stamp_wan21_checkpoints(card, checkpoint_root)
    return card
