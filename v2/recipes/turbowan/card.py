"""TurboWan ModelCards — rCM few-step distilled Wan (self-contained recipe package).

TurboWan (Reparameterized Consistency Model distilled Wan, TurboDiffusion arXiv:2512.16093) reuses the Wan
architecture end-to-end: ``WanTransformer3DModel`` velocity DiT, ``AutoencoderKLWan`` VAE, UMT5 text encoder
(+ CLIP vision encoder for I2V). There is no new torch adapter — components carry the same ``load_id`` strings
as ``v2/recipes/wan21``, built by the existing ``WanDiT`` dispatch in ``torch_backend.py``. The only new work
is the rCM sampler/loop (``v2/recipes/turbowan/{sampler,loop}.py``).

Three HF ids, two builders:
  * ``build_turbowan_card`` (size-agnostic T2V) serves ``TurboWan2.1-T2V-1.3B`` (480p) and
    ``TurboWan2.1-T2V-14B`` (720p); rCM has no flow-shift schedule. Size is resolved from the checkpoint by
    the loader, so the per-id delta is just the default h/w.
  * ``build_turbowan_i2v_a14b_card`` serves ``TurboWan2.2-I2V-A14B`` — a boundary-routed MoE (two
    WanTransformer3DModel experts) + i2v conditioning (CLIP image encoder + first-frame [mask|cond]), reusing
    ``BoundaryTimestepRouting`` and the Wan i2v hooks. sigma_max=200 vs 80 for T2V.

SamplingDefaults match ``fastvideo/pipelines/basic/turbodiffusion/presets.py``: 4 steps, guidance_scale 1.0
(CFG off), and an empty negative prompt (a local constant — the shared WAN_NEG_* are not what the preset uses).
``stamp_wan21_checkpoints`` applies the diffusers subfolder layout.
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
from v2.loop.policies import BoundaryTimestepRouting, ClassicCFG, NoRouting, PrecisionPolicy
from v2.parallel import ParallelPlan
from v2.platform.backends.toy import ToyDiT, ToyImageEncoder, ToyTextEncoder, ToyVAE, _seed_from
from v2.recipes.turbowan.loop import TurboWanDenoiseLoop
from v2.recipes.wan21.card import stamp_wan21_checkpoints

# TurboWan presets use an empty negative prompt (guidance off at scale 1.0), so the shared Wan negatives
# do not apply. Local constant per the self-contained-package rule.
TURBOWAN_NEG = ""

_WAN_DIT = "fastvideo.models.dits.wanvideo:WanTransformer3DModel"
_WAN_VAE = "fastvideo.models.vaes.wanvae:AutoencoderKLWan"
_UMT5 = "fastvideo.models.encoders.t5:T5EncoderModel"
_CLIP_VISION = "fastvideo.models.encoders.clip:CLIPVisionModel"


def build_turbowan_card(model_id: str = "turbowan2.1-t2v-1.3b",
                        *,
                        sigma_max: float = 80.0,
                        height: int = 480,
                        width: int = 832,
                        num_frames: int = 81,
                        fps: int = 16,
                        num_steps: int = 4,
                        checkpoint_root: str | None = None) -> ModelCard:
    """TurboWan2.1 T2V (size-agnostic): serves the 1.3B (480p) and 14B (720p) ids. rCM 4-step, CFG off."""
    seed = _seed_from(model_id)
    cost = CostModel(kind=WorkUnitKind.DIFFUSION_STEP, base_seconds=1e-4, per_unit_seconds=1e-7)
    cfg = ClassicCFG()  # carried for the op-structure key (CFG off at scale 1)
    precision = PrecisionPolicy(compute_dtype="float32", scheduler_step_in_fp32=True)
    expert = NoRouting("transformer")

    def loop_factory() -> TurboWanDenoiseLoop:
        return TurboWanDenoiseLoop(loop_id="diffusion_denoise",
                                   cfg=cfg,
                                   precision=precision,
                                   expert=expert,
                                   cost=cost,
                                   sigma_max=sigma_max)

    components = {
        "text_encoder":
        ComponentSpec(component_id="text_encoder",
                      kind="text_encoder",
                      load_id=_UMT5,
                      factory=lambda inst: ToyTextEncoder(),
                      required_for={"t2v"}),
        "vae":
        ComponentSpec(component_id="vae",
                      kind="vae",
                      load_id=_WAN_VAE,
                      factory=lambda inst: ToyVAE(),
                      required_for={"t2v"}),
        "transformer":
        ComponentSpec(component_id="transformer",
                      kind="dit",
                      load_id=_WAN_DIT,
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
                 graph_capture="eager",
                 loop_factory=loop_factory),
    }
    card = ModelCard(
        model_id=model_id,
        family="wan",
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
        sampling_defaults=SamplingDefaults(num_steps=num_steps,
                                           guidance_scale=1.0,
                                           height=height,
                                           width=width,
                                           num_frames=num_frames,
                                           fps=fps,
                                           negative_prompt=TURBOWAN_NEG),
    )
    card.validate()
    if checkpoint_root:
        stamp_wan21_checkpoints(card, checkpoint_root)
    return card


def build_turbowan_i2v_a14b_card(model_id: str = "turbowan2.2-i2v-a14b",
                                 *,
                                 sigma_max: float = 200.0,
                                 boundary: float = 0.9,
                                 height: int = 720,
                                 width: int = 1280,
                                 num_frames: int = 81,
                                 fps: int = 16,
                                 num_steps: int = 4,
                                 checkpoint_root: str | None = None) -> ModelCard:
    """TurboWan2.2-I2V-A14B — rCM few-step MoE i2v. Two WanTransformer3DModel experts boundary-routed
    (``boundary_ratio`` 0.9, compared in raw-sigma space) + i2v conditioning (CLIP + first-frame
    [mask|cond]). sigma_max=200 (the I2V pipeline's value vs 80 for T2V). Reuses the Wan adapter, boundary
    policy, and i2v hooks; only the rCM sampler/loop is new."""
    seed = _seed_from(model_id)
    cost = CostModel(kind=WorkUnitKind.DIFFUSION_STEP, base_seconds=1e-4, per_unit_seconds=1e-7)
    cfg = ClassicCFG()
    precision = PrecisionPolicy(compute_dtype="float32", scheduler_step_in_fp32=True)
    expert = BoundaryTimestepRouting(high_noise="transformer", low_noise="transformer_2", boundary=boundary)

    def loop_factory() -> TurboWanDenoiseLoop:
        return TurboWanDenoiseLoop(loop_id="i2v_denoise",
                                   cfg=cfg,
                                   precision=precision,
                                   expert=expert,
                                   cost=cost,
                                   sigma_max=sigma_max)

    def _dit(cid: str) -> ComponentSpec:
        return ComponentSpec(component_id=cid,
                             kind="dit",
                             load_id=_WAN_DIT,
                             factory=lambda inst: ToyDiT(seed=seed),
                             resident_for=["i2v_denoise"],
                             required_for={"i2v"})

    components = {
        "text_encoder":
        ComponentSpec(component_id="text_encoder",
                      kind="text_encoder",
                      load_id=_UMT5,
                      factory=lambda inst: ToyTextEncoder(),
                      required_for={"i2v"}),
        "image_encoder":
        ComponentSpec(component_id="image_encoder",
                      kind="image_encoder",
                      load_id=_CLIP_VISION,
                      factory=lambda inst: ToyImageEncoder(),
                      required_for={"i2v"}),
        "vae":
        ComponentSpec(component_id="vae",
                      kind="vae",
                      load_id=_WAN_VAE,
                      factory=lambda inst: ToyVAE(),
                      required_for={"i2v"}),
        "transformer":
        _dit("transformer"),
        "transformer_2":
        _dit("transformer_2"),
    }
    loops = {
        "i2v_denoise":
        LoopSpec(loop_id="i2v_denoise",
                 kind=LoopKind.DIFFUSION_DENOISE,
                 work_unit_kind=WorkUnitKind.DIFFUSION_STEP,
                 step_cost_model=cost,
                 shared_weight_components=["transformer", "transformer_2"],
                 cache_policy=["feature"],
                 graph_capture="eager",
                 loop_factory=loop_factory),
    }
    card = ModelCard(
        model_id=model_id,
        family="wan",
        components=components,
        loops=loops,
        capabilities=CapabilityMatrix.of(Capability.IMAGE_TO_VIDEO, Capability.VAE_DECODE),
        recipe=RecipeSpec(method="base",
                          assumes_loop="i2v_denoise",
                          assumes_precision="float32",
                          consistency_required=ConsistencyLevel.C1),
        parity=ParitySpec(consistency_levels=[ConsistencyLevel.C1], interleave_required=True),
        caches={"feature": CacheContract(cache_class="feature", max_bytes=1 << 24, reuse_across_requests=True)},
        precision=PrecisionContract(default_dtype="float32", training_precision="float32"),
        parallelism=ParallelismContract(valid_plans=[ParallelPlan.single()], default_plan=ParallelPlan.single()),
        sampling_defaults=SamplingDefaults(num_steps=num_steps,
                                           guidance_scale=1.0,
                                           height=height,
                                           width=width,
                                           num_frames=num_frames,
                                           fps=fps,
                                           negative_prompt=TURBOWAN_NEG),
    )
    card.validate()
    if checkpoint_root:
        stamp_wan21_checkpoints(card, checkpoint_root)
    return card
