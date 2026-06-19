"""Kandinsky-5.0-T2V-Lite ModelCard — text→video, ported into the v2 substrate.

Architecture deltas vs Wan (all declared on the card so the recipe is self-contained — the
bucket-C pattern: torch adapters via ``ComponentSpec.adapter``, a new ``Kandinsky5DenoiseLoop``):

  * DiT  ``fastvideo.models.dits.kandinsky5:Kandinsky5Transformer3DModel`` — a flow-match velocity
    predictor, BUT with **channels-LAST** latent geometry ``[B, T, H, W, C]`` (C=4, NOT Wan's
    channels-first ``[B, C, T, H, W]``) and TWO conditioning streams: Qwen2.5-VL token embeds
    (``encoder_hidden_states``) + a CLIP **pooled** vector (``pooled_projections``, folded into the
    time embedding — mandatory, the forward raises if it is None). The forward also requires per-request
    RoPE positions (``visual_rope_pos`` over the patched grid, ``text_rope_pos`` over the Qwen seq len)
    and a resolution-dependent ``scale_factor``. The adapter (-> ``Kandinsky5DiT``) builds all of these
    internally so the loop's dit-call stays compatible with the CPU ``ToyDiT`` signature.
  * VAE  ``fastvideo.models.vaes.hunyuanvae:AutoencoderKLHunyuanVideo`` — HunyuanVideo VAE (z=4, 8×/4×)
    with a SCALAR ``scaling_factor`` (latent = raw·sf; decode raw = latent/sf), NOT Wan's per-channel
    ``(z-mean)/std`` — reusing the Wan VAE adapter would corrupt latents (-> ``Kandinsky5VAE``).
  * Text ``fastvideo.models.encoders.qwen2_5:Qwen2_5_VLTextModel`` (Qwen embeds, in_text_dim=3584) +
    ``fastvideo.models.encoders.clip:CLIPTextModelWithProjection`` (pooled, in_text_dim2=768). Two
    text_encoder components (``text_encoder`` / ``text_encoder_2``) with distinct adapters.
  * sched ``FlowMatchEulerDiscreteScheduler`` — plain flow-match Euler (velocity prediction, σ·1000
    timestep convention); CFG combines ``uncond + s·(cond − uncond)`` == ``ClassicCFG``.

``stamp_kandinsky5_checkpoints`` (a local superset of ``stamp_wan21_checkpoints``) points each
component at its diffusers subfolder, adding ``text_encoder_2``/``tokenizer_2`` which the Wan stamp
lacks. ``flow_shift`` is read from the checkpoint's ``scheduler/scheduler_config.json`` at stamp time
(it is NOT hardcoded in the upstream code — the scheduler code default is 1.0).
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
from v2.recipes.kandinsky5.loop import (
    KANDINSKY5_LATENT_CHANNELS,
    KANDINSKY5_SPATIAL_RATIO,
    KANDINSKY5_TEMPORAL_RATIO,
    Kandinsky5DenoiseLoop,
)

# Kandinsky-5 default negative prompt (kept as a LOCAL module constant, not in the shared _prompts.py,
# so the recipe is self-contained). From the diffusers pipeline / repo defaults.
KANDINSKY5_NEG = ("Static, 2D cartoon, cartoon, 2d animation, paintings, images, worst quality, "
                  "low quality, ugly, deformed, walking backwards")

_KANDINSKY5_DIT = "v2.recipes.kandinsky5.adapter:Kandinsky5DiT"
_KANDINSKY5_QWEN = "v2.recipes.kandinsky5.adapter:Kandinsky5QwenEncoder"
_KANDINSKY5_CLIP = "v2.recipes.kandinsky5.adapter:Kandinsky5ClipEncoder"
_KANDINSKY5_VAE = "v2.recipes.kandinsky5.adapter:Kandinsky5VAE"


def build_kandinsky5_card(model_id: str = "kandinsky-5.0-t2v-lite-5s",
                          *,
                          checkpoint_root: str | None = None,
                          flow_shift: float = 1.0,
                          sampling_defaults: SamplingDefaults | None = None) -> ModelCard:
    seed = _seed_from(model_id)
    cost = CostModel(kind=WorkUnitKind.DIFFUSION_STEP, base_seconds=1e-4, per_unit_seconds=1e-7)
    cfg = ClassicCFG()
    # flow_shift default is 1.0 (the scheduler code default); the real value is read from the
    # checkpoint's scheduler_config.json by stamp_kandinsky5_checkpoints at GPU deploy time.
    flow = FlowShiftPolicy(shift=flow_shift)
    precision = PrecisionPolicy(compute_dtype="float32", scheduler_step_in_fp32=True)
    expert = NoRouting("transformer")

    def loop_factory():
        return Kandinsky5DenoiseLoop(loop_id="diffusion_denoise",
                                     cfg=cfg,
                                     flow_shift=flow,
                                     precision=precision,
                                     expert=expert,
                                     cost=cost,
                                     latent_channels=KANDINSKY5_LATENT_CHANNELS,
                                     spatial_ratio=KANDINSKY5_SPATIAL_RATIO,
                                     temporal_ratio=KANDINSKY5_TEMPORAL_RATIO)

    components = {
        # Qwen2.5-VL: token embeds [seq, 3584] -> encoder_hidden_states (the cross-attended text stream).
        "text_encoder":
        ComponentSpec(component_id="text_encoder",
                      kind="text_encoder",
                      load_id="fastvideo.models.encoders.qwen2_5:Qwen2_5_VLTextModel",
                      adapter=_KANDINSKY5_QWEN,
                      factory=lambda inst: ToyTextEncoder(),
                      required_for={"t2v"}),
        # CLIP-with-projection: pooled vector [768] -> pooled_projections (folded into the time embed).
        "text_encoder_2":
        ComponentSpec(component_id="text_encoder_2",
                      kind="text_encoder",
                      load_id="fastvideo.models.encoders.clip:CLIPTextModelWithProjection",
                      adapter=_KANDINSKY5_CLIP,
                      factory=lambda inst: ToyTextEncoder(),
                      required_for={"t2v"}),
        "vae":
        ComponentSpec(
            component_id="vae",
            kind="vae",
            load_id="fastvideo.models.vaes.hunyuanvae:AutoencoderKLHunyuanVideo",
            adapter=_KANDINSKY5_VAE,
            factory=lambda inst: ToyVAE(),  # toy stand-in: the toy LATENT_CHANNELS, not the
            required_for={"t2v"}),  # real z=16 (the CPU latent_shape uses the toy count)
        "transformer":
        ComponentSpec(
            component_id="transformer",
            kind="dit",
            load_id="fastvideo.models.dits.kandinsky5:Kandinsky5Transformer3DModel",
            adapter=_KANDINSKY5_DIT,
            factory=lambda inst: ToyDiT(seed=seed),  # toy stand-in uses the toy LATENT_CHANNELS
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
        family="kandinsky5",
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
        sampling_defaults=sampling_defaults or SamplingDefaults(num_steps=50,
                                                                guidance_scale=5.0,
                                                                height=512,
                                                                width=768,
                                                                num_frames=121,
                                                                fps=24,
                                                                negative_prompt=KANDINSKY5_NEG),
    )
    card.validate()
    if checkpoint_root:
        stamp_kandinsky5_checkpoints(card, checkpoint_root)
    return card


# Diffusers checkpoint layout for Kandinsky-5: a superset of the Wan layout adding the SECOND text
# encoder + its tokenizer (the shared _WAN21_SUBFOLDERS only has one text_encoder/tokenizer).
_KANDINSKY5_SUBFOLDERS = {
    "transformer": "transformer",
    "vae": "vae",
    "text_encoder": "text_encoder",
    "text_encoder_2": "text_encoder_2"
}


def stamp_kandinsky5_checkpoints(card: ModelCard, model_root: str) -> ModelCard:
    """GPU deploy-time: point each component's ``ComponentSpec.checkpoint`` at its weights subfolder under
    ``model_root`` (a local diffusers dir or an HF id resolved via ``snapshot_download``), and read the
    real ``flow_shift`` from ``scheduler/scheduler_config.json`` into the loop's FlowShiftPolicy (it is NOT
    hardcoded upstream — code default 1.0; BRINGUP risk J). The CLIP encoder's tokenizer lives in
    ``tokenizer_2`` (NOT the ``tokenizer`` the built-in text-encoder maker probes) — the Kandinsky5ClipEncoder
    adapter re-resolves it from the model root (BRINGUP). Mutates and returns the card."""
    import json
    import os
    if not os.path.isdir(model_root):
        from huggingface_hub import snapshot_download
        model_root = snapshot_download(model_root)
    for component_id, subfolder in _KANDINSKY5_SUBFOLDERS.items():
        if component_id in card.components:
            card.components[component_id].checkpoint = os.path.join(model_root, subfolder)
    # BRINGUP J: lift flow_shift from the scheduler config so >1.0 schedules are honored on GPU.
    sched_cfg = os.path.join(model_root, "scheduler", "scheduler_config.json")
    if os.path.exists(sched_cfg):
        with open(sched_cfg) as f:
            shift = json.load(f).get("shift")
        if shift is not None:
            loop_spec = card.loops.get("diffusion_denoise")
            base_factory = loop_spec.loop_factory if loop_spec is not None else None
            if base_factory is not None:

                def _shifted_factory(_base=base_factory, _shift=float(shift)):
                    loop = _base()
                    loop.flow_shift.shift = _shift
                    return loop

                loop_spec.loop_factory = _shifted_factory
    return card
