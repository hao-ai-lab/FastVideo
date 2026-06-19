"""Matrix-Game 3.0 ModelCard — image+action+camera -> video autoregressive world model.

Self-contained recipe package (bucket-C pattern, mirroring cosmos2): the card declares its torch adapter
via ``ComponentSpec.adapter`` (``MatrixGame3DiT`` in ``v2/platform/backends/torch_matrixgame3.py``) plus
``MatrixGame3DenoiseLoop`` (the autoregressive multi-clip loop), reusing the Wan VAE adapter + T5 +
``stamp_wan21_checkpoints``.

Architecture deltas vs Wan (all GPU-path, declared on the card):
  * DiT  ``fastvideo.models.dits.matrixgame3.model:MatrixGame3WanModel`` — a 5B autoregressive
    world-model DiT (30 layers, dim 24*128=3072, ffn 14336, patch (1,2,2), in/out_channels=48,
    ``use_memory=True``, ``sigma_theta=0.8``, action_config on the first 15 blocks). Predicts a velocity
    (``flow_prediction``). The adapter takes a PER-TOKEN timestep (cond-frame rows zeroed) + the
    action/camera/KV-memory bundle, built INTERNALLY so the loop's dit-call stays
    ``dit(latent, text_embed, sigma)``-shaped.
  * VAE  ``fastvideo.models.vaes.wanvae:AutoencoderKLWan`` — the ``light_vae`` variant (z_dim=48, 16x
    spatial / 4x temporal, Wan2.2-TI2V geometry); reuses the v2 ``WanVAE`` adapter unchanged (its
    ``_mean_invstd`` normalization is the MG3 normalized-latent convention).
  * Text ``fastvideo.models.encoders.t5:T5EncoderModel`` (T5, fp32, text_len 512) via the default
    ``T5Encoder`` adapter (Wan zero-pad-to-512 convention). NO image/CLIP encoder — the conditioning
    image goes only through the VAE (BRINGUP risk 6).

Sampler: flow-match via ``FlowUniPCMultistepScheduler`` (``flow_prediction``, flow_shift=5.0); distilled
3-step student, guidance_scale=1.0 (CFG effectively off unless ``use_base_model=True``).
``stamp_wan21_checkpoints`` applies (diffusers transformer/vae/text_encoder + tokenizer subfolders).
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
from v2.recipes.matrixgame3.loop import (
    MG3_LATENT_CHANNELS,
    MG3_SPATIAL_RATIO,
    MG3_TEMPORAL_RATIO,
    MatrixGame3DenoiseLoop,
)
from v2.recipes.wan21.card import stamp_wan21_checkpoints

# Matrix-Game 3.0 uses an EMPTY negative prompt by default (kept as a LOCAL module constant per the
# self-contained-recipe rule — not added to the shared v2/recipes/_prompts.py).
MG3_NEG = ""

_MG3_DIT = "v2.platform.backends.torch_matrixgame3:MatrixGame3DiT"


def build_matrixgame3_card(model_id: str = "matrixgame3-i2v",
                           *,
                           flow_shift: float = 5.0,
                           checkpoint_root: str | None = None,
                           sampling_defaults: SamplingDefaults | None = None) -> ModelCard:
    seed = _seed_from(model_id)
    cost = CostModel(kind=WorkUnitKind.DIFFUSION_STEP, base_seconds=1e-4, per_unit_seconds=1e-7)
    cfg = ClassicCFG()
    # MG3 720p uses flow_shift=5.0 (its 480P-derived pipeline config). The bucket lookup keeps the
    # 720p/480p shift table consistent with the Wan flow-shift convention.
    flow = FlowShiftPolicy(shift=flow_shift, bucket_lookup={480 * 832: 3.0, 720 * 1280: 5.0})
    precision = PrecisionPolicy(compute_dtype="float32", scheduler_step_in_fp32=True)
    expert = NoRouting("transformer")

    def loop_factory():
        return MatrixGame3DenoiseLoop(loop_id="mg3_denoise",
                                      cfg=cfg,
                                      flow_shift=flow,
                                      precision=precision,
                                      expert=expert,
                                      cost=cost,
                                      latent_channels=MG3_LATENT_CHANNELS,
                                      spatial_ratio=MG3_SPATIAL_RATIO,
                                      temporal_ratio=MG3_TEMPORAL_RATIO)

    components = {
        "text_encoder":
        ComponentSpec(component_id="text_encoder",
                      kind="text_encoder",
                      load_id="fastvideo.models.encoders.t5:T5EncoderModel",
                      factory=lambda inst: ToyTextEncoder(),
                      required_for={"i2v"}),
        "vae":
        ComponentSpec(
            component_id="vae",
            kind="vae",
            load_id="fastvideo.models.vaes.wanvae:AutoencoderKLWan",  # light_vae (z=48, 16x/4x)
            factory=lambda inst: ToyVAE(),
            required_for={"i2v"}),
        "transformer":
        ComponentSpec(component_id="transformer",
                      kind="dit",
                      load_id="fastvideo.models.dits.matrixgame3.model:MatrixGame3WanModel",
                      adapter=_MG3_DIT,
                      factory=lambda inst: ToyDiT(seed=seed),
                      resident_for=["mg3_denoise"],
                      required_for={"i2v"}),
    }
    loops = {
        "mg3_denoise":
        LoopSpec(loop_id="mg3_denoise",
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
        family="matrixgame",
        components=components,
        loops=loops,
        capabilities=CapabilityMatrix.of(Capability.IMAGE_TO_VIDEO, Capability.VAE_DECODE),
        recipe=RecipeSpec(method="base",
                          assumes_loop="mg3_denoise",
                          assumes_precision="float32",
                          consistency_required=ConsistencyLevel.C1),
        parity=ParitySpec(consistency_levels=[ConsistencyLevel.C1],
                          interleave_required=True,
                          tests=[ParityTestSpec(name="denoise_trajectory", level=ConsistencyLevel.C1, tap="latents")]),
        caches={"feature": CacheContract(cache_class="feature", max_bytes=1 << 24, reuse_across_requests=True)},
        precision=PrecisionContract(default_dtype="float32", training_precision="float32"),
        parallelism=ParallelismContract(valid_plans=[ParallelPlan.single()], default_plan=ParallelPlan.single()),
        sampling_defaults=sampling_defaults or SamplingDefaults(
            num_steps=3, guidance_scale=1.0, height=720, width=1280, num_frames=57, fps=25, negative_prompt=MG3_NEG),
    )
    card.validate()
    if checkpoint_root:
        stamp_wan21_checkpoints(card, checkpoint_root)
    return card
