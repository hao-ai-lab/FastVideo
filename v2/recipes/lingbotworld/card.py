"""LingBot-World-Base-Cam ModelCard — camera-conditioned, dual-guidance MoE image-to-video.

LingBot-World-Base-Cam (``FastVideo/LingBot-World-Base-Cam-Diffusers``) is Wan2.2-I2V-A14B plus a
camera/Plucker conditioning input. Everything is declared on the card so the recipe is self-contained:

  * DiT (x2)  ``fastvideo.models.dits.lingbotworld.model:LingBotWorldTransformer3DModel`` — a 2x14B
    boundary-routed MoE (``transformer`` high-noise, ``transformer_2`` low-noise; ``boundary_ratio=0.947``).
    The forward takes ``c2ws_plucker_emb`` (per-block FiLM camera injection) that a vanilla WanDiT would
    drop, so both experts use the ``LingBotWorldDiT`` adapter, which threads the Plucker tensor and keeps
    the Wan i2v 36ch ``[noise|mask+cond]`` concat.
  * VAE  ``fastvideo.models.vaes.wanvae:AutoencoderKLWan`` (z=16, 8x spatial / 4x temporal, mean/std
    normalized latent space) — reuses the v2 ``WanVAE`` adapter. ``spatial_scale=8`` for the Plucker
    downsample matches the 8x spatial compression.
  * Text  ``fastvideo.models.encoders.t5:T5EncoderModel`` (UMT5, text_len=512) — reuses ``T5Encoder``.
  * No CLIP image encoder: ``image_dim=null``, so first-frame conditioning is carried entirely by the
    36ch ``[noise|mask+cond]`` latent concat (see the ``cond_encode`` node).

Sampler: flow-match (``FlowUniPCMultistepScheduler``, ``prediction_type='flow_prediction'``) via the v2
``FlowShiftPolicy`` + ``FLOW_MATCH_STEP``, as in Wan. The defaults are unusual and must NOT inherit Wan's:
``flow_shift=10.0``, ``num_inference_steps=70``, dual ``guidance_scale``/``guidance_scale_2`` (both 5.0),
and a Chinese negative prompt.

``stamp_wan21_checkpoints`` applies (diffusers ``transformer``/``transformer_2``/``vae``/``text_encoder``
subfolder layout, all in ``_WAN21_SUBFOLDERS``).
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
    PrecisionContract,
    RecipeSpec,
    SamplingDefaults,
)
from v2.loop.policies import BoundaryTimestepRouting, ClassicCFG, FlowShiftPolicy, PrecisionPolicy
from v2.parallel import ParallelPlan
from v2.platform.backends.toy import ToyDiT, ToyTextEncoder, ToyVAE, _seed_from
from v2.recipes.lingbotworld.loop import LINGBOTWORLD_BOUNDARY, LingBotWorldDenoiseLoop
from v2.recipes.wan21.card import stamp_wan21_checkpoints

# The LingBot-World I2V negative prompt (presets.py LINGBOTWORLD_I2V) — kept LOCAL to this recipe so the
# port stays self-contained (no edit to the shared v2/recipes/_prompts.py).
LINGBOTWORLD_NEG_CN = ("画面突变，色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，"
                       "最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，"
                       "畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走，"
                       "镜头晃动，画面闪烁，模糊，噪点，水印，签名，文字，变形，扭曲，液化，不合逻辑的结构，卡顿，"
                       "PPT幻灯片感，过暗，欠曝，低对比度，霓虹灯光感，过度锐化，3D渲染感，人物，行人，游客，身体，"
                       "皮肤，肢体，面部特征，汽车，电线")

_LINGBOTWORLD_DIT = "v2.recipes.lingbotworld.adapter:LingBotWorldDiT"


def build_lingbotworld_card(model_id: str = "lingbot-world-base-cam",
                            *,
                            boundary: float = LINGBOTWORLD_BOUNDARY,
                            checkpoint_root: str | None = None,
                            sampling_defaults: SamplingDefaults | None = None) -> ModelCard:
    """LingBot-World-Base-Cam card — 2x14B camera-conditioned dual-guidance MoE i2v.

    Mirrors ``build_wan22_i2v_a14b_card`` but with ``flow_shift=10.0``, ``boundary_ratio=0.947``, both DiT
    experts on the ``LingBotWorldDiT`` adapter (threads ``c2ws_plucker_emb`` + Wan i2v cond), the
    ``LingBotWorldDenoiseLoop`` (dual guidance + camera), and the LingBot defaults (70 steps, gs 5.0,
    480x832, Chinese negative prompt). 2x14B bf16 won't co-reside on 80GB, so the adapter keeps the WanDiT
    CPU-offload swap (GPU-pending)."""
    seed = _seed_from(model_id)
    cost = CostModel(kind=WorkUnitKind.DIFFUSION_STEP, base_seconds=1e-4, per_unit_seconds=1e-7)
    cfg = ClassicCFG()
    flow = FlowShiftPolicy(shift=10.0)  # LingBot-World's unusually high flow shift (not the Wan 3.0/5.0)
    precision = PrecisionPolicy(compute_dtype="float32", scheduler_step_in_fp32=True)
    expert = BoundaryTimestepRouting(high_noise="transformer", low_noise="transformer_2", boundary=boundary)

    def loop_factory() -> LingBotWorldDenoiseLoop:
        return LingBotWorldDenoiseLoop(loop_id="i2v_denoise",
                                       cfg=cfg,
                                       flow_shift=flow,
                                       precision=precision,
                                       expert=expert,
                                       cost=cost,
                                       boundary=boundary)

    def _dit(cid: str) -> ComponentSpec:
        return ComponentSpec(cid,
                             kind="dit",
                             load_id="fastvideo.models.dits.lingbotworld.model:LingBotWorldTransformer3DModel",
                             adapter=_LINGBOTWORLD_DIT,
                             factory=lambda inst: ToyDiT(seed=seed),
                             resident_for=["i2v_denoise"],
                             required_for={"i2v"})

    components = {
        "text_encoder":
        ComponentSpec("text_encoder",
                      kind="text_encoder",
                      load_id="fastvideo.models.encoders.t5:T5EncoderModel",
                      factory=lambda inst: ToyTextEncoder(),
                      required_for={"i2v"}),
        # NOTE: LingBot-World-Base-Cam has NO CLIP image encoder — model_index.json lists
        # ``image_encoder`` / ``image_processor`` as ``[null, null]`` and the transformer config has
        # ``image_dim=null`` (the condition embedder is built without an image branch). Unlike
        # Wan2.2-I2V-A14B, the first-frame conditioning is carried ENTIRELY by the 36ch ``[noise|mask+cond]``
        # latent concat (no ``encoder_hidden_states_image``). So this card declares no image_encoder
        # component (declaring one would make ``stamp_wan21_checkpoints`` point at a non-existent subfolder
        # and fail the load).
        "vae":
        ComponentSpec("vae",
                      kind="vae",
                      load_id="fastvideo.models.vaes.wanvae:AutoencoderKLWan",
                      factory=lambda inst: ToyVAE(),
                      required_for={"i2v"}),
        "transformer":
        _dit("transformer"),
        "transformer_2":
        _dit("transformer_2"),
    }
    loops = {
        "i2v_denoise":
        LoopSpec("i2v_denoise",
                 kind=LoopKind.DIFFUSION_DENOISE,
                 work_unit_kind=WorkUnitKind.DIFFUSION_STEP,
                 step_cost_model=cost,
                 shared_weight_components=["transformer", "transformer_2"],
                 cache_policy=["feature"],
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
        caches={"feature": CacheContract("feature", max_bytes=1 << 24, reuse_across_requests=True)},
        precision=PrecisionContract(default_dtype="float32", training_precision="float32"),
        parallelism=ParallelismContract(valid_plans=[ParallelPlan.single()], default_plan=ParallelPlan.single()),
        sampling_defaults=sampling_defaults or SamplingDefaults(num_steps=70,
                                                                guidance_scale=5.0,
                                                                height=480,
                                                                width=832,
                                                                num_frames=81,
                                                                fps=16,
                                                                negative_prompt=LINGBOTWORLD_NEG_CN),
    )
    card.validate()
    if checkpoint_root:
        stamp_wan21_checkpoints(card, checkpoint_root)
    return card
