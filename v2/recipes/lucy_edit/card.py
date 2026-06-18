"""Lucy-Edit ModelCard — Wan VIDEO-TO-VIDEO editor (decart-ai/Lucy-Edit-Dev).

Lucy-Edit is the Wan2.2-5B (TI2V) network repurposed as a prompt-driven video editor. It reuses the
Wan architecture wholesale — the only deltas vs Wan2.2-TI2V-5B are (a) the DiT's input channel count
(96 = 48 noise + 48 conditioning-video latent, vs 48 for pure t2v) and (b) the conditioning is the
VAE-encoded INPUT video, not a first-frame image. Discovered wiring (fastvideo source):
  * pipeline ``fastvideo.pipelines.basic.wan.lucy_edit_pipeline:LucyEditPipeline`` (subclass of
    ``WanVideoToVideoPipeline``) — text_encode -> conditioning -> timestep/latent prep ->
    ``VideoVAEEncodingStage`` -> denoise (concat ``[noise | video_latent]``) -> decode.
  * config ``LucyEditDevConfig`` (configs/pipelines/wan.py): DiT in_channels=96, out_channels=48,
    30 layers, 24 heads, ffn 14336; VAE z_dim=48, 16x spatial, 4x temporal (the Wan2.2 enhanced VAE),
    ``expand_timesteps=True``, ``lucy_edit_task=True`` (NO ``image_encoder`` in the repo).
  * sched ``FlowUniPCMultistepScheduler(shift=flow_shift)``, flow_shift 5.0 (inherits Wan2.2-TI2V-5B).
  * preset ``LUCY_EDIT_DEV`` (pipelines/basic/wan/presets.py): 480x832, 81 frames, fps 24, 50 steps,
    guidance 5.0, EMPTY negative prompt.

Because the arch IS Wan, this card declares the same component ``load_id``s as the Wan2.1 card and
needs NO custom adapter — the torch backend's ``_make_dit`` already builds ``WanTransformer3DModel``
(the 96ch in / 48ch out shape is resolved from ``LucyEditDevConfig`` on the GPU loader) and
``WanVAE``/``T5Encoder`` are reused as-is. The toy ``factory`` swaps in CPU stand-ins. The shared
``WanDenoiseLoop`` carries the conditioning video latent via its ``i2v_cond`` hook (channel-concat in
the Wan adapter); ``stamp_wan21_checkpoints`` applies (diffusers subfolder layout).

NOTES / BRINGUP:
  * Two ids served: ``decart-ai/Lucy-Edit-Dev`` and ``decart-ai/Lucy-Edit-1.1-Dev`` (same config /
    arch; 1.1 is a weights refresh). Same builder; the registry maps both ids to this card.
  * INPUT-VIDEO PLUMBING IS BRINGUP: v2's ``Request`` has a ``VideoPart`` modality but no wired
    ``input video -> VAE-encode -> conditioning latent`` path end to end. The program builds that
    node and the conditioning structure faithfully, and DEGRADES TO T2V when no input video is
    present (``video_latent`` stays None -> the loop runs the plain Wan t2v forward). GPU-verify the
    96ch concat + ``expand_timesteps`` per-token timestep on a real Lucy checkpoint.
  * The 5B DiT uses ``expand_timesteps`` (per-token timestep vector); that is internal to the Wan
    torch adapter / fastvideo denoising stage, not the loop's flow-match arithmetic.
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
from v2.recipes.lucy_edit.loop import (
    LUCY_LATENT_CHANNELS,
    LUCY_SPATIAL_RATIO,
    LUCY_TEMPORAL_RATIO,
    WanDenoiseLoop,
)
from v2.recipes.wan21.card import stamp_wan21_checkpoints

# Lucy-Edit ships an EMPTY negative prompt (preset LUCY_EDIT_DEV) — kept as a LOCAL constant per the
# self-contained recipe rule (no shared _prompts edit).
LUCY_NEG = ""

_LUCY_LOOP = "lucy_edit_denoise"


def build_lucy_edit_card(model_id: str = "lucy-edit-dev",
                         *,
                         flow_shift: float = 5.0,
                         height: int = 480,
                         width: int = 832,
                         num_frames: int = 81,
                         fps: int = 24,
                         num_steps: int = 50,
                         guidance_scale: float = 5.0,
                         checkpoint_root: str | None = None) -> ModelCard:
    """Lucy-Edit-Dev / Lucy-Edit-1.1-Dev card. Wan2.2-5B arch (WanTransformer3DModel in_ch=96 /
    AutoencoderKLWan z=48 16x/4x / UMT5) reused unchanged; v2v conditioning rides the WanDenoiseLoop
    ``i2v_cond`` hook (channel-concat of the VAE-encoded input video). The 96ch DiT input shape is
    resolved from ``LucyEditDevConfig`` by the GPU loader, so no adapter override is needed here."""
    seed = _seed_from(model_id)
    cost = CostModel(kind=WorkUnitKind.DIFFUSION_STEP, base_seconds=1e-4, per_unit_seconds=1e-7)
    cfg = ClassicCFG()
    # Lucy inherits Wan2.2-TI2V-5B's flow shift; the 480p/720p bucket lookup mirrors the Wan recipe.
    flow = FlowShiftPolicy(shift=flow_shift, bucket_lookup={480 * 832: 5.0, 720 * 1280: 5.0})
    precision = PrecisionPolicy(compute_dtype="float32", scheduler_step_in_fp32=True)
    expert = NoRouting("transformer")

    def loop_factory():
        # The Wan2.2 enhanced-VAE geometry (z=48, 16x spatial, 4x temporal). Reuses the shared Wan
        # flow-match loop; the input-video conditioning latent threads through ``i2v_cond`` (see loop.py).
        return WanDenoiseLoop(loop_id=_LUCY_LOOP,
                              cfg=cfg,
                              flow_shift=flow,
                              precision=precision,
                              expert=expert,
                              cost=cost,
                              latent_channels=LUCY_LATENT_CHANNELS,
                              spatial_ratio=LUCY_SPATIAL_RATIO,
                              temporal_ratio=LUCY_TEMPORAL_RATIO)

    components = {
        "text_encoder":
        ComponentSpec(component_id="text_encoder",
                      kind="text_encoder",
                      load_id="fastvideo.models.encoders.t5:T5EncoderModel",
                      factory=lambda inst: ToyTextEncoder(),
                      required_for={"v2v"}),
        "vae":
        ComponentSpec(component_id="vae",
                      kind="vae",
                      load_id="fastvideo.models.vaes.wanvae:AutoencoderKLWan",
                      factory=lambda inst: ToyVAE(),
                      required_for={"v2v"}),
        # WanTransformer3DModel with in_channels=96 (48 noise + 48 video-cond latent), out_channels=48.
        # The shape comes from LucyEditDevConfig on the GPU loader; no adapter= (the Wan dispatch builds it).
        "transformer":
        ComponentSpec(component_id="transformer",
                      kind="dit",
                      load_id="fastvideo.models.dits.wanvideo:WanTransformer3DModel",
                      factory=lambda inst: ToyDiT(seed=seed),
                      resident_for=[_LUCY_LOOP],
                      required_for={"v2v"}),
    }
    loops = {
        _LUCY_LOOP:
        LoopSpec(loop_id=_LUCY_LOOP,
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
        family="wan",
        components=components,
        loops=loops,
        # Lucy is a video editor; it also degrades to t2v when no input video is supplied (BRINGUP).
        capabilities=CapabilityMatrix.of(Capability.VIDEO_TO_VIDEO, Capability.TEXT_TO_VIDEO, Capability.VAE_DECODE),
        recipe=RecipeSpec(method="base",
                          assumes_loop=_LUCY_LOOP,
                          assumes_precision="float32",
                          consistency_required=ConsistencyLevel.C1),
        parity=ParitySpec(consistency_levels=[ConsistencyLevel.C1],
                          interleave_required=True,
                          tests=[ParityTestSpec(name="denoise_trajectory", level=ConsistencyLevel.C1, tap="latents")]),
        caches={"feature": CacheContract(cache_class="feature", max_bytes=1 << 24, reuse_across_requests=True)},
        precision=PrecisionContract(default_dtype="float32", training_precision="float32"),
        parallelism=ParallelismContract(valid_plans=[ParallelPlan.single()], default_plan=ParallelPlan.single()),
        sampling_defaults=SamplingDefaults(num_steps=num_steps,
                                           guidance_scale=guidance_scale,
                                           height=height,
                                           width=width,
                                           num_frames=num_frames,
                                           fps=fps,
                                           negative_prompt=LUCY_NEG),
    )
    card.validate()
    if checkpoint_root:
        stamp_wan21_checkpoints(card, checkpoint_root)
    return card
