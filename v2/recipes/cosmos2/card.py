"""Cosmos-Predict2-2B-Video2World ModelCard — text->video (the registered preset; video2world
conditioning is a later capability the loop already threads).

Architecture deltas vs Wan (declared on the card so the recipe is self-contained):
  * DiT  ``fastvideo.models.dits.cosmos:CosmosTransformer3DModel`` — an EDM denoiser (not flow-match); the
    ``CosmosDiT`` adapter returns the raw network output and ``CosmosDenoiseLoop`` does the EDM
    ``c_in/c_skip/c_out`` -> x0 reconstruction + x0-space CFG + the Karras (rho=7, sigma 80->0.002) schedule.
  * VAE  ``fastvideo.models.vaes.wanvae:AutoencoderKLWan`` — Wan-style (z=16, 8x/4x); reuses the v2
    ``WanVAE`` adapter unchanged (sigma_data=1.0 makes the Cosmos sigma_data factor a no-op).
  * Text ``fastvideo.models.encoders.t5:T5EncoderModel`` (T5-Large, 1024-dim) via ``CosmosT5Encoder``
    (raw last_hidden_state, no Wan zero-pad).
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
from v2.loop.policies import ClassicCFG, NoRouting, PrecisionPolicy
from v2.parallel import ParallelPlan
from v2.platform.backends.toy import ToyDiT, ToyTextEncoder, ToyVAE, _seed_from
from v2.recipes._prompts import COSMOS_NEG
from v2.recipes.cosmos2.loop import CosmosDenoiseLoop
from v2.recipes.wan21.card import stamp_wan21_checkpoints

_COSMOS_DIT = "v2.recipes.cosmos2.adapter:CosmosDiT"
_COSMOS_T5 = "v2.recipes.cosmos2.adapter:CosmosT5Encoder"


def build_cosmos2_card(model_id: str = "cosmos-predict2-2b-video2world",
                       *,
                       checkpoint_root: str | None = None,
                       sampling_defaults: SamplingDefaults | None = None) -> ModelCard:
    seed = _seed_from(model_id)
    cost = CostModel(kind=WorkUnitKind.DIFFUSION_STEP, base_seconds=1e-4, per_unit_seconds=1e-7)
    cfg = ClassicCFG()
    precision = PrecisionPolicy(compute_dtype="float32", scheduler_step_in_fp32=True)
    expert = NoRouting("transformer")

    def loop_factory():
        return CosmosDenoiseLoop(loop_id="diffusion_denoise",
                                 cfg=cfg,
                                 precision=precision,
                                 expert=expert,
                                 cost=cost,
                                 sigma_max=80.0,
                                 sigma_min=0.002,
                                 sigma_data=1.0,
                                 rho=7.0,
                                 augment_sigma=0.001)

    components = {
        "text_encoder":
        ComponentSpec(component_id="text_encoder",
                      kind="text_encoder",
                      load_id="fastvideo.models.encoders.t5:T5EncoderModel",
                      adapter=_COSMOS_T5,
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
                      load_id="fastvideo.models.dits.cosmos:CosmosTransformer3DModel",
                      adapter=_COSMOS_DIT,
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
        family="cosmos",
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
        sampling_defaults=sampling_defaults or SamplingDefaults(
            num_steps=35, guidance_scale=7.0, height=704, width=1280, num_frames=93, fps=16,
            negative_prompt=COSMOS_NEG),
    )
    card.validate()
    if checkpoint_root:
        stamp_wan21_checkpoints(card, checkpoint_root)
    return card
