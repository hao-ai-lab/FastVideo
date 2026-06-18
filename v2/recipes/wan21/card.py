"""Wan2.1-1.3B ModelCard (design_v3 §4) — the (recipe, runtime) pair for T2V.

Mirrors the real Wan2.1-1.3B wiring discovered in the repo:
  * DiT  ``fastvideo.models.dits.wanvideo:WanTransformer3DModel`` (40 layers, dim 5120)
  * VAE  ``fastvideo.models.vaes.wanvae:AutoencoderKLWan`` (z=16, 4×/8× compression)
  * T5   ``fastvideo.models.encoders.t5:T5EncoderModel``
  * sched ``FlowUniPCMultistepScheduler`` (flow_prediction; 480p shift 3.0, 720p 5.0)
``load_id`` records those for the GPU torch adapter; ``factory`` returns the CPU toy here.
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
from v2.loop.policies import (
    BoundaryTimestepRouting,
    ClassicCFG,
    FlowShiftPolicy,
    NoRouting,
    PrecisionPolicy,
)
from v2.parallel import ParallelPlan
from v2.platform.backends.toy import ToyDiT, ToyTextEncoder, ToyVAE, _seed_from
from v2.recipes._prompts import WAN_NEG_CN, WAN_NEG_EN
from v2.recipes.wan21.loop import WanDenoiseLoop


def build_wan21_card(model_id: str = "wan2.1-1.3b", *, cfg_policy=None, flow_shift: float = 3.0,
                     checkpoint_root: str | None = None, latent_channels: int = 16,
                     spatial_ratio: int = 8, temporal_ratio: int = 4,
                     sampling_defaults: SamplingDefaults | None = None) -> ModelCard:
    seed = _seed_from(model_id)
    cost = CostModel(kind=WorkUnitKind.DIFFUSION_STEP, base_seconds=1e-4, per_unit_seconds=1e-7)
    cfg = cfg_policy or ClassicCFG()
    flow = FlowShiftPolicy(shift=flow_shift, bucket_lookup={480 * 832: 3.0, 720 * 1280: 5.0})
    precision = PrecisionPolicy(compute_dtype="float32", scheduler_step_in_fp32=True)
    expert = NoRouting("transformer")

    def loop_factory():
        return WanDenoiseLoop(loop_id="diffusion_denoise", cfg=cfg, flow_shift=flow,
                              precision=precision, expert=expert, cost=cost,
                              latent_channels=latent_channels, spatial_ratio=spatial_ratio,
                              temporal_ratio=temporal_ratio)

    components = {
        "text_encoder": ComponentSpec(
            component_id="text_encoder", kind="text_encoder",
            load_id="fastvideo.models.encoders.t5:T5EncoderModel",
            factory=lambda inst: ToyTextEncoder(), required_for={"t2v"}),
        "vae": ComponentSpec(
            component_id="vae", kind="vae",
            load_id="fastvideo.models.vaes.wanvae:AutoencoderKLWan",
            factory=lambda inst: ToyVAE(), required_for={"t2v"}),
        "transformer": ComponentSpec(
            component_id="transformer", kind="dit",
            load_id="fastvideo.models.dits.wanvideo:WanTransformer3DModel",
            factory=lambda inst: ToyDiT(seed=seed),
            resident_for=["diffusion_denoise"], required_for={"t2v"}),
    }
    loops = {
        "diffusion_denoise": LoopSpec(
            loop_id="diffusion_denoise", kind=LoopKind.DIFFUSION_DENOISE,
            work_unit_kind=WorkUnitKind.DIFFUSION_STEP, step_cost_model=cost,
            shared_weight_components=["transformer"], cache_policy=["feature"],
            graph_capture="breakable_cudagraph", loop_factory=loop_factory),
    }
    card = ModelCard(
        model_id=model_id, family="wan",
        components=components, loops=loops,
        capabilities=CapabilityMatrix.of(
            Capability.TEXT_TO_VIDEO,   # base Wan2.1 is T2V-only; i2v is the separate InP variant
            Capability.VAE_DECODE, Capability.POLICY_ROLLOUT),
        recipe=RecipeSpec(method="base", assumes_loop="diffusion_denoise",
                          assumes_precision="float32", consistency_required=ConsistencyLevel.C1),
        parity=ParitySpec(consistency_levels=[ConsistencyLevel.C1], interleave_required=True,
                          tests=[ParityTestSpec(name="denoise_trajectory", level=ConsistencyLevel.C1,
                                                tap="latents")]),
        caches={"feature": CacheContract(cache_class="feature", max_bytes=1 << 24,
                                         reuse_across_requests=True)},
        precision=PrecisionContract(default_dtype="float32", training_precision="float32"),
        parallelism=ParallelismContract(
            valid_plans=[ParallelPlan.single(),
                         ParallelPlan(axes={"sp": 2, "cfgp": 2}, mesh_order=["cfgp", "sp"])],
            default_plan=ParallelPlan.single()),
        sampling_defaults=sampling_defaults or SamplingDefaults(
            num_steps=50, guidance_scale=3.0, height=480, width=832, num_frames=81, fps=16,
            negative_prompt=WAN_NEG_EN),
    )
    card.validate()
    if checkpoint_root:
        stamp_wan21_checkpoints(card, checkpoint_root)
    return card


def build_wan22_ti2v_card(model_id: str = "wan2.2-ti2v-5b", *,
                          checkpoint_root: str | None = None) -> ModelCard:
    """Wan2.2-TI2V-5B card. Same component classes as Wan2.1 (WanTransformer3DModel / AutoencoderKLWan /
    UMT5), so the torch adapters are reused as-is — the deltas are the higher-compression VAE geometry
    (z_dim=48, 16x spatial, 4x temporal) and the 480p flow shift 5.0. The DiT forward accepts a scalar
    timestep (its 1D path), so no per-frame ``expand_timesteps`` is needed for pure t2v. Uses the same
    ``transformer/vae/text_encoder`` subfolder layout, so ``stamp_wan21_checkpoints`` applies."""
    return build_wan21_card(model_id=model_id, flow_shift=5.0, checkpoint_root=checkpoint_root,
                            latent_channels=48, spatial_ratio=16, temporal_ratio=4,
                            sampling_defaults=SamplingDefaults(
                                num_steps=50, guidance_scale=5.0, height=704, width=1280,
                                num_frames=121, fps=24, negative_prompt=WAN_NEG_CN))


def build_wan_t2v_14b_card(model_id: str = "wan2.1-t2v-14b", *, checkpoint_root: str | None = None) -> ModelCard:
    """Wan2.1-T2V-14B (720p). Same WanTransformer3DModel / AutoencoderKLWan / UMT5 + 16/8/4 VAE geometry
    as the 1.3B — reuses the Wan recipe + torch adapter unchanged; only the size (resolved from the
    checkpoint by the loader), the 720p flow shift 5.0, and the per-model defaults differ."""
    return build_wan21_card(model_id=model_id, flow_shift=5.0, checkpoint_root=checkpoint_root,
                            sampling_defaults=SamplingDefaults(
                                num_steps=50, guidance_scale=5.0, height=720, width=1280,
                                num_frames=81, fps=16, negative_prompt=WAN_NEG_EN))


def build_wan22_a14b_card(model_id: str = "wan2.2-t2v-a14b", *, checkpoint_root: str | None = None,
                          boundary: float = 0.875) -> ModelCard:
    """Wan2.2-T2V-A14B card — MoE: two WanTransformer3DModel experts (in_ch=16, Wan2.1 16/8/4 geometry)
    with a boundary-timestep switch (``boundary_ratio`` 0.875). ``transformer`` is the high-noise expert
    (early, high-sigma steps), ``transformer_2`` the low-noise expert. ``BoundaryTimestepRouting`` picks
    per step and ``WanDenoiseLoop`` dispatches via ``model.component(expert_id)`` (both experts reuse the
    Wan torch adapter as-is). The boundary is in (shifted) sigma space: ctx.sigma>=0.875 -> high-noise,
    matching diffusers' ``t >= boundary_ratio*num_train_timesteps`` on the shifted timesteps. NOTE: 2x14B
    bf16 (~56GB) + UMT5 resident is near an 80GB GPU's limit — use modest res/frames."""
    seed = _seed_from(model_id)
    cost = CostModel(kind=WorkUnitKind.DIFFUSION_STEP, base_seconds=1e-4, per_unit_seconds=1e-7)
    cfg = ClassicCFG()
    flow = FlowShiftPolicy(shift=5.0, bucket_lookup={480 * 832: 3.0, 720 * 1280: 5.0})
    precision = PrecisionPolicy(compute_dtype="float32", scheduler_step_in_fp32=True)
    expert = BoundaryTimestepRouting(high_noise="transformer", low_noise="transformer_2", boundary=boundary)

    def loop_factory():
        return WanDenoiseLoop(loop_id="diffusion_denoise", cfg=cfg, flow_shift=flow,
                              precision=precision, expert=expert, cost=cost)  # 16/8/4 default geometry

    def _dit_spec(cid):
        return ComponentSpec(
            component_id=cid, kind="dit",
            load_id="fastvideo.models.dits.wanvideo:WanTransformer3DModel",
            factory=lambda inst: ToyDiT(seed=seed),
            resident_for=["diffusion_denoise"], required_for={"t2v"})

    components = {
        "text_encoder": ComponentSpec(
            component_id="text_encoder", kind="text_encoder",
            load_id="fastvideo.models.encoders.t5:T5EncoderModel",
            factory=lambda inst: ToyTextEncoder(), required_for={"t2v"}),
        "vae": ComponentSpec(
            component_id="vae", kind="vae",
            load_id="fastvideo.models.vaes.wanvae:AutoencoderKLWan",
            factory=lambda inst: ToyVAE(), required_for={"t2v"}),
        "transformer": _dit_spec("transformer"),
        "transformer_2": _dit_spec("transformer_2"),
    }
    loops = {
        "diffusion_denoise": LoopSpec(
            loop_id="diffusion_denoise", kind=LoopKind.DIFFUSION_DENOISE,
            work_unit_kind=WorkUnitKind.DIFFUSION_STEP, step_cost_model=cost,
            shared_weight_components=["transformer", "transformer_2"], cache_policy=["feature"],
            graph_capture="breakable_cudagraph", loop_factory=loop_factory),
    }
    card = ModelCard(
        model_id=model_id, family="wan",
        components=components, loops=loops,
        capabilities=CapabilityMatrix.of(
            Capability.TEXT_TO_VIDEO, Capability.VAE_DECODE, Capability.POLICY_ROLLOUT),
        recipe=RecipeSpec(method="base", assumes_loop="diffusion_denoise",
                          assumes_precision="float32", consistency_required=ConsistencyLevel.C1),
        parity=ParitySpec(consistency_levels=[ConsistencyLevel.C1], interleave_required=True,
                          tests=[ParityTestSpec(name="denoise_trajectory", level=ConsistencyLevel.C1,
                                                tap="latents")]),
        caches={"feature": CacheContract(cache_class="feature", max_bytes=1 << 24,
                                         reuse_across_requests=True)},
        precision=PrecisionContract(default_dtype="float32", training_precision="float32"),
        parallelism=ParallelismContract(valid_plans=[ParallelPlan.single()],
                                        default_plan=ParallelPlan.single()),
        sampling_defaults=SamplingDefaults(
            num_steps=40, guidance_scale=4.0, height=480, width=832, num_frames=81, fps=16,
            negative_prompt=WAN_NEG_CN),
    )
    card.validate()
    if checkpoint_root:
        stamp_wan21_checkpoints(card, checkpoint_root)
    return card


# Diffusers checkpoint layout: each component's weights live in a subfolder of the model root. The stamp
# loop skips any subfolder a given card doesn't declare, so this superset covers Wan2.1, Wan2.2 MoE
# (``transformer_2``), and LTX-2 (``spatial_upsampler``) alike.
_WAN21_SUBFOLDERS = {"transformer": "transformer", "transformer_2": "transformer_2",
                     "vae": "vae", "text_encoder": "text_encoder",
                     "image_encoder": "image_encoder",                 # Wan i2v CLIP vision
                     "spatial_upsampler": "spatial_upsampler",
                     "audio_vae": "audio_vae", "vocoder": "vocoder"}   # LTX-2.3 T2VS audio branch


def stamp_wan21_checkpoints(card: ModelCard, model_root: str) -> ModelCard:
    """GPU deploy-time: point each component's ``ComponentSpec.checkpoint`` at its weights subfolder
    under ``model_root`` (a local diffusers dir, or an HF id resolved via ``snapshot_download``). The
    toy cards leave ``checkpoint`` empty; the real torch adapters require it (BRINGUP risk A). The
    adapter derives the pipeline-config model root from ``dirname(checkpoint)`` and loads the sibling
    ``tokenizer/`` directory too. Mutates and returns the card."""
    import os
    if not os.path.isdir(model_root):
        from huggingface_hub import snapshot_download
        model_root = snapshot_download(model_root)
    for component_id, subfolder in _WAN21_SUBFOLDERS.items():
        if component_id in card.components:
            card.components[component_id].checkpoint = os.path.join(model_root, subfolder)
    return card
