"""HunyuanVideo 1.5 ModelCard — text→video (the registered 480p t2v preset). i2v + SR are later
capabilities the loop/adapter thread but the t2v program does not yet exercise on CPU.

Architecture deltas vs Wan (all declared on the card so the recipe is self-contained):
  * DiT  ``HunyuanVideo15Transformer3DModel`` — a rectified-flow velocity network (54 dual-stream
    MMDoubleStream blocks, hidden 2048, in/out 65/32). Its forward takes a list of two text embeds
    (Qwen2.5-VL + ByT5) and a list image-embed; the ``HunyuanVideo15DiT`` adapter marshals those, so
    ``HunyuanVideo15DenoiseLoop`` (a thin ``WanDenoiseLoop`` subclass with z=32/16×/4× geometry) drives it
    with the unchanged flow-match math.
  * VAE  ``hunyuan15vae`` — causal-3D, z=32, 16× spatial / 4× temporal, normalized by a SCALAR
    ``scaling_factor=1.03682`` via the ``HunyuanVideo15VAE`` adapter (not Wan's mean/std).
  * Text encoders: ``text_encoder`` = Qwen2.5-VL (3584-d, ``hidden_states[-3]`` cropped of 108 template
    tokens) via ``HunyuanVideo15QwenEncoder``; ``text_encoder_2`` = ByT5/Glyph (1472-d, quoted-glyph text,
    may be empty) via ``HunyuanVideo15ByT5Encoder``.
``stamp_wan21_checkpoints`` applies (diffusers transformer/vae/text_encoder/text_encoder_2 subfolders).

Defaults (configs/pipelines + presets): 480p t2v -> 50 steps, guidance 6.0, 480×848, 121 frames, 24 fps,
flow_shift 5 (720p uses 9), empty negative prompt, explicit ``linspace(1,0,n+1)[:-1]`` sigma schedule.
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
from v2.recipes.hunyuan_video15.loop import (
    HUNYUAN15_LATENT_CHANNELS,
    HUNYUAN15_SPATIAL_RATIO,
    HUNYUAN15_TEMPORAL_RATIO,
    HunyuanVideo15DenoiseLoop,
)
from v2.recipes.wan21.card import stamp_wan21_checkpoints

# HunyuanVideo 1.5 ships with an EMPTY negative prompt (configs/.../presets.py). Kept as a LOCAL constant
# (the recipe owns its prompts; not added to the shared v2/recipes/_prompts.py).
HUNYUAN15_NEG = ""

_HY15_DIT = "v2.platform.backends.torch_hunyuan_video15:HunyuanVideo15DiT"
_HY15_VAE = "v2.platform.backends.torch_hunyuan_video15:HunyuanVideo15VAE"
_HY15_QWEN = "v2.platform.backends.torch_hunyuan_video15:HunyuanVideo15QwenEncoder"
_HY15_BYT5 = "v2.platform.backends.torch_hunyuan_video15:HunyuanVideo15ByT5Encoder"

# 480p t2v explicit sigma schedule = linspace(1, 0, num_steps+1)[:-1] (presets.py:_sigmas).
_DEFAULT_STEPS = 50


def _linspace_sigmas(num_steps: int) -> tuple[float, ...]:
    """Faithful to ``presets.py:_sigmas`` — ``np.linspace(1,0,n+1)[:-1]`` (drops the terminal 0). The
    loop's ``build_schedule`` consumes it verbatim when present (FlowShiftPolicy passes it through)."""
    return tuple((1.0 - i / num_steps) for i in range(num_steps))


def build_hunyuan_video15_card(model_id: str = "hunyuan-video-1.5-t2v-480p",
                               *,
                               flow_shift: float = 5.0,
                               height: int = 480,
                               width: int = 848,
                               num_steps: int = _DEFAULT_STEPS,
                               guidance_scale: float = 6.0,
                               checkpoint_root: str | None = None,
                               sampling_defaults: SamplingDefaults | None = None) -> ModelCard:
    seed = _seed_from(model_id)
    cost = CostModel(kind=WorkUnitKind.DIFFUSION_STEP, base_seconds=1e-4, per_unit_seconds=1e-7)
    cfg = ClassicCFG()
    # 480p flow_shift 5, 720p 9 (configs/pipelines/hunyuan15.py). The preset also hands an explicit sigma
    # schedule via SamplingDefaults.sigmas, which the loop prefers when present.
    flow = FlowShiftPolicy(shift=flow_shift, bucket_lookup={480 * 848: 5.0, 720 * 1280: 9.0})
    precision = PrecisionPolicy(compute_dtype="float32", scheduler_step_in_fp32=True)
    expert = NoRouting("transformer")

    def loop_factory():
        return HunyuanVideo15DenoiseLoop(loop_id="diffusion_denoise",
                                         cfg=cfg,
                                         flow_shift=flow,
                                         precision=precision,
                                         expert=expert,
                                         cost=cost,
                                         latent_channels=HUNYUAN15_LATENT_CHANNELS,
                                         spatial_ratio=HUNYUAN15_SPATIAL_RATIO,
                                         temporal_ratio=HUNYUAN15_TEMPORAL_RATIO)

    components = {
        "text_encoder":
        ComponentSpec(component_id="text_encoder",
                      kind="text_encoder",
                      load_id="fastvideo.models.encoders.qwen2_5:Qwen2_5_VLTextModel",
                      adapter=_HY15_QWEN,
                      factory=lambda inst: ToyTextEncoder(),
                      required_for={"t2v"}),
        "text_encoder_2":
        ComponentSpec(component_id="text_encoder_2",
                      kind="text_encoder",
                      load_id="fastvideo.models.encoders.t5:T5EncoderModel",
                      adapter=_HY15_BYT5,
                      factory=lambda inst: ToyTextEncoder(),
                      required_for={"t2v"}),
        "vae":
        ComponentSpec(component_id="vae",
                      kind="vae",
                      load_id="fastvideo.models.vaes.hunyuan15vae:AutoencoderKLHunyuanVideo15",
                      adapter=_HY15_VAE,
                      factory=lambda inst: ToyVAE(),
                      required_for={"t2v"}),
        "transformer":
        ComponentSpec(component_id="transformer",
                      kind="dit",
                      load_id="fastvideo.models.dits.hunyuanvideo15:HunyuanVideo15Transformer3DModel",
                      adapter=_HY15_DIT,
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
        # sequence_model_parallel_shard inside the DiT forward is a no-op at sp=1; default single() until
        # SP is GPU-validated (BRINGUP risk G).
        parallelism=ParallelismContract(valid_plans=[ParallelPlan.single()], default_plan=ParallelPlan.single()),
        sampling_defaults=sampling_defaults or SamplingDefaults(num_steps=num_steps,
                                                                guidance_scale=guidance_scale,
                                                                height=height,
                                                                width=width,
                                                                num_frames=121,
                                                                fps=24,
                                                                negative_prompt=HUNYUAN15_NEG,
                                                                shift=flow_shift,
                                                                sigmas=_linspace_sigmas(num_steps)),
    )
    card.validate()
    if checkpoint_root:
        stamp_wan21_checkpoints(card, checkpoint_root)
        # stamp_wan21_checkpoints does not cover HunyuanVideo 1.5's second text encoder (ByT5 in
        # ``text_encoder_2/``), so stamp it here to keep the recipe self-contained (no edit to the shared
        # stamp). The model root is the parent of any already-stamped subfolder (stamp resolves HF ids -> a
        # local snapshot).
        import os
        if "text_encoder_2" in card.components and not card.components["text_encoder_2"].checkpoint:
            model_root = os.path.dirname(os.path.normpath(card.components["transformer"].checkpoint))
            card.components["text_encoder_2"].checkpoint = os.path.join(model_root, "text_encoder_2")
    return card


def build_hunyuan_video15_720p_card(model_id: str = "hunyuan-video-1.5-t2v-720p",
                                    *,
                                    checkpoint_root: str | None = None) -> ModelCard:
    """HunyuanVideo 1.5 720p t2v — same components/adapters; only flow_shift (9), resolution, and the
    per-model defaults differ (configs/pipelines/hunyuan15.py:Hunyuan15T2V720PConfig)."""
    return build_hunyuan_video15_card(model_id=model_id,
                                      flow_shift=9.0,
                                      height=720,
                                      width=1280,
                                      checkpoint_root=checkpoint_root)
