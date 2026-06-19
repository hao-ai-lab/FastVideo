"""Matrix-Game 2.0 (Base, distilled) ModelCard — the INTERACTIVE mouse/keyboard world model ported as a
self-contained recipe (bucket-C). Registered preset is the i2v world-rollout entry; live action routing is
BRINGUP (needs a request-API extension — the loop already threads the action slots).

Architecture deltas vs Wan (all declared on the card so the recipe is self-contained):
  * DiT  ``fastvideo.models.dits.matrixgame2.causal_model:CausalMatrixGame2WanModel`` — a DISTILLED, CAUSAL,
    ACTION-CONDITIONED Wan-family transformer whose output is EPSILON (not velocity). The adapter
    (``ComponentSpec.adapter`` -> ``MatrixGame2CausalDiT``) returns the raw epsilon; ``MatrixGame2CausalDMDLoop``
    does the few-step DMD (eps->x0 via the scheduler sigma table + re-add_noise), the causal block-rollout,
    and the sliding-window KV-cache threading.
  * VAE  ``fastvideo.models.vaes.wanvae:AutoencoderKLWan`` — Wan 2.1 (z=16, 8x/4x); reuses the v2 ``WanVAE``
    adapter unchanged (mean/std normalization). The i2v cond_concat latent uses the SAME normalization.
  * Image  CLIP vision encoder (``fastvideo.models.encoders.clip:CLIPVisionModel``, 257x1280 tokens) is the
    SOLE cross-attention context (``MatrixGame2CLIPImageEncoder`` adapter + sibling ``image_processor``).
  * NO text encoder — Matrix-Game 2.0 ignores text; the condition_embedder returns no text tokens.
``stamp_wan21_checkpoints`` applies (diffusers transformer/vae/image_encoder subfolder layout).

Distilled: the 3-step DMD schedule ``dmd_denoising_steps=[1000,666,333]`` is intrinsic to the weights; the
preset surfaces ``num_inference_steps`` but the model is distilled for exactly these timesteps. CFG is OFF
(``guidance_scale=1.0``). The Base checkpoint is single-expert (``boundary_ratio=None``); GTA/TempleRun
distilled variants may carry a high/low-noise MoE boundary (the loop has the gated ``pred_noise_to_x_bound``
path for that).
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
from v2.platform.backends.toy import ToyDiT, ToyImageEncoder, ToyVAE, _seed_from
from v2.recipes.matrixgame2.loop import MatrixGame2CausalDMDLoop
from v2.recipes.matrixgame2.sampler import (
    MATRIXGAME2_CONTEXT_NOISE,
    MATRIXGAME2_DMD_STEPS,
    MATRIXGAME2_FLOW_SHIFT,
    MATRIXGAME2_NUM_FRAMES_PER_BLOCK,
)
from v2.recipes.wan21.card import stamp_wan21_checkpoints

# Matrix-Game 2.0 ignores text -> no negative prompt (kept as a LOCAL constant, not shared _prompts.py).
MATRIXGAME2_NEG = ""

_MG2_DIT = "v2.recipes.matrixgame2.adapter:MatrixGame2CausalDiT"
_MG2_CLIP = "v2.recipes.matrixgame2.adapter:MatrixGame2CLIPImageEncoder"


def build_matrixgame2_card(model_id: str = "matrix-game-2.0-base-distilled",
                           *,
                           flow_shift: float = MATRIXGAME2_FLOW_SHIFT,
                           boundary_ratio: float | None = None,
                           checkpoint_root: str | None = None,
                           sampling_defaults: SamplingDefaults | None = None) -> ModelCard:
    seed = _seed_from(model_id)
    cost = CostModel(kind=WorkUnitKind.DIFFUSION_STEP, base_seconds=1e-4, per_unit_seconds=1e-7)
    cfg = ClassicCFG()  # carried for the op-structure key; guidance_scale=1.0 makes it a no-op (CFG OFF)
    flow = FlowShiftPolicy(shift=flow_shift)  # seeds the FlowUniPC sigma table the eps->x0 conversion needs
    precision = PrecisionPolicy(compute_dtype="float32", scheduler_step_in_fp32=True)
    expert = NoRouting("transformer")  # single-expert Base; the loop gates the MoE bound path on boundary_ratio

    def loop_factory():
        return MatrixGame2CausalDMDLoop(loop_id="causal_dmd_denoise",
                                        cfg=cfg,
                                        flow_shift=flow,
                                        precision=precision,
                                        expert=expert,
                                        cost=cost,
                                        dmd_steps=MATRIXGAME2_DMD_STEPS,
                                        context_noise=MATRIXGAME2_CONTEXT_NOISE,
                                        num_frames_per_block=MATRIXGAME2_NUM_FRAMES_PER_BLOCK,
                                        boundary_ratio=boundary_ratio)

    components = {
        "image_encoder":
        ComponentSpec(component_id="image_encoder",
                      kind="image_encoder",
                      load_id="fastvideo.models.encoders.clip:CLIPVisionModel",
                      adapter=_MG2_CLIP,
                      factory=lambda inst: ToyImageEncoder(),
                      required_for={"i2v"}),
        "vae":
        ComponentSpec(component_id="vae",
                      kind="vae",
                      load_id="fastvideo.models.vaes.wanvae:AutoencoderKLWan",
                      factory=lambda inst: ToyVAE(),
                      required_for={"i2v"}),
        "transformer":
        ComponentSpec(component_id="transformer",
                      kind="dit",
                      load_id="fastvideo.models.dits.matrixgame2.causal_model:CausalMatrixGame2WanModel",
                      adapter=_MG2_DIT,
                      factory=lambda inst: ToyDiT(seed=seed),
                      resident_for=["causal_dmd_denoise"],
                      required_for={"i2v"}),
    }
    loops = {
        "causal_dmd_denoise":
        LoopSpec(
            loop_id="causal_dmd_denoise",
            kind=LoopKind.DIFFUSION_DENOISE,
            work_unit_kind=WorkUnitKind.DIFFUSION_STEP,
            step_cost_model=cost,
            shared_weight_components=["transformer"],
            cache_policy=["feature"],
            graph_capture="eager",  # causal KV-cache mutation + host-RNG re-noise -> not capturable
            loop_factory=loop_factory),
    }
    card = ModelCard(
        model_id=model_id,
        family="matrixgame",
        components=components,
        loops=loops,
        # World model: i2v (first-frame + action -> continuous frame extension) + VAE decode.
        capabilities=CapabilityMatrix.of(Capability.IMAGE_TO_VIDEO, Capability.VAE_DECODE),
        recipe=RecipeSpec(method="base",
                          assumes_loop="causal_dmd_denoise",
                          assumes_precision="float32",
                          consistency_required=ConsistencyLevel.C1),
        parity=ParitySpec(consistency_levels=[ConsistencyLevel.C1],
                          interleave_required=True,
                          tests=[ParityTestSpec(name="denoise_trajectory", level=ConsistencyLevel.C1, tap="latents")]),
        caches={"feature": CacheContract(cache_class="feature", max_bytes=1 << 24, reuse_across_requests=True)},
        precision=PrecisionContract(default_dtype="float32", training_precision="float32"),
        parallelism=ParallelismContract(valid_plans=[ParallelPlan.single()], default_plan=ParallelPlan.single()),
        sampling_defaults=sampling_defaults or SamplingDefaults(num_steps=3,
                                                                guidance_scale=1.0,
                                                                height=352,
                                                                width=640,
                                                                num_frames=57,
                                                                fps=25,
                                                                negative_prompt=MATRIXGAME2_NEG),
    )
    card.validate()
    if checkpoint_root:
        stamp_wan21_checkpoints(card, checkpoint_root)
    return card
