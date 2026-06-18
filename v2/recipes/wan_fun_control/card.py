"""Wan2.1-Fun-Control ModelCard — control-video-conditioned Wan2.1 (V2V/Control).

Primary HF id: ``IRMChen/Wan2.1-Fun-1.3B-Control-Diffusers`` (registry: ``WANV2VConfig`` +
``wan_fun_1_3b_control`` preset).

Mechanism (faithful to ``fastvideo/pipelines/basic/wan/wan_v2v_pipeline.py``): Wan2.1-Fun-Control is a
plain Wan2.1 flow-match DiT whose ONLY architectural delta from T2V is the input channel count. A CONTROL
VIDEO is VAE-encoded to a latent and concatenated onto the noise latent at every denoise step:
``cat([noise(16ch), control_latent(16ch), zero_pad(16ch)])`` → the 48-channel Fun-Control DiT input
(``WanVideoArchConfig`` in_channels=48; resolved from the checkpoint config by the loader). An optional
CLIP-vision REFERENCE image (``WAN2_1ControlCLIPVisionConfig``) feeds the DiT as
``encoder_hidden_states_image`` — analogous to the i2v ``[mask|cond]`` conditioning, but the primary
signal is the control video, not a first-frame image.

Reuse: the architecture IS Wan, so NO new DiT adapter (the torch backend's ``_make_dit`` already builds
``WanTransformer3DModel`` and the ``WanDiT.__call__`` ``cond=`` path already concats the control half),
NO new VAE/T5 adapter, NO new sampler, and the shared ``WanDenoiseLoop`` unchanged. The Fun-Control
program (``program.py``) builds the ``[control_latent | zero_pad]`` (32ch) tensor and writes it to the
``i2v_cond`` slot the loop already threads to the adapter. ``stamp_wan21_checkpoints`` applies (diffusers
transformer/vae/text_encoder/image_encoder subfolder layout).

BRINGUP (written-not-run): (a) Wan2.1-Fun-1.3B-Control loads via the generic Wan loader with
in_channels=48; (b) the control-video request input — v2's ``Request`` has no ``video()`` accessor, so
the program scans ``inputs`` for a ``VideoPart`` and DEGRADES to T2V (or ref-image i2v) when absent;
(c) the ``WAN2_1ControlCLIPVisionConfig`` reference-image encoder subfolder + bf16 dtype.
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
from v2.loop.policies import ClassicCFG, FlowShiftPolicy, NoRouting, PrecisionPolicy
from v2.parallel import ParallelPlan
from v2.platform.backends.toy import ToyDiT, ToyImageEncoder, ToyTextEncoder, ToyVAE, _seed_from
from v2.recipes.wan21.card import stamp_wan21_checkpoints
from v2.recipes.wan_fun_control.loop import WanDenoiseLoop

# Local constant (copied verbatim from the ``wan_fun_1_3b_control`` preset: _NEGATIVE_PROMPT_EN). Recipe
# DATA, not shared code — owned here so the package is self-contained.
_FUN_CONTROL_NEG = ("Bright tones, overexposed, static, blurred details, subtitles, style, works,"
                    " paintings, images, static, overall gray, worst quality, low quality, JPEG"
                    " compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly"
                    " drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture,"
                    " messy background, three legs, many people in the background, walking backwards")


def _ensure_fastvideo_control_detector() -> None:
    """BRINGUP: fastvideo's registry resolves the V2V/Control PipelineConfig (``WANV2VConfig`` — DiT/VAE/T5
    precisions + configs the loaders need) by model_path. The id ``IRMChen/Wan2.1-Fun-1.3B-Control-Diffusers``
    IS registered (``WANV2VConfig``), but ONLY as an exact HF-id / short-name map. At GPU load time the
    torch backend hands the loader the *local snapshot dir* (``.../snapshots/<hash>``) — whose short name is
    the opaque hash — and (unlike the sibling Wan2.1-Fun-InP, whose ``WanImageToVideoPipeline`` class name
    hits a registered detector) the Control entry has NO detector for its ``WanVideoToVideoPipeline`` class
    name, so the snapshot-dir path fails to resolve (``ValueError: No match found for pipeline ...``).

    fastvideo source is off-limits for this port (HARD RULE), so we add the missing detector here, at
    recipe-import time, via fastvideo's PUBLIC registry state — attaching it to the *existing* Control
    ``model_id`` (no duplicate ConfigInfo, no HF-path remap warning). The detector matches the
    ``WanVideoToVideoPipeline`` class name (from ``model_index.json``) and the ``wan2.1-fun-1.3b-control``
    path substring, so the snapshot-dir path now resolves to ``WANV2VConfig``. Idempotent. NOTE: the clean
    upstream fix is a ``model_detectors=[...]`` arg on the Control ``register_configs`` call in
    ``fastvideo/registry.py`` — reported in the bring-up notes."""
    try:
        from fastvideo import registry as _reg
    except Exception:
        return
    hf_id = "IRMChen/Wan2.1-Fun-1.3B-Control-Diffusers"
    model_id = _reg._MODEL_HF_PATH_TO_NAME.get(hf_id)
    if model_id is None:  # not registered (older fastvideo) -> nothing to do
        return

    def _is_wan_fun_control(path: str) -> bool:
        p = (path or "").lower()
        return "wanvideotovideopipeline" in p or "wan2.1-fun-1.3b-control" in p

    # Already attached? (idempotent across re-imports / repeated card builds.)
    if any(mid == model_id and getattr(det, "_wan_fun_control", False) for mid, det in _reg._MODEL_NAME_DETECTORS):
        return
    _is_wan_fun_control._wan_fun_control = True  # type: ignore[attr-defined]
    _reg._MODEL_NAME_DETECTORS.append((model_id, _is_wan_fun_control))


_ensure_fastvideo_control_detector()


def build_wan_fun_control_card(model_id: str = "wan2.1-fun-1.3b-control",
                               *,
                               flow_shift: float = 3.0,
                               height: int = 832,
                               width: int = 480,
                               num_frames: int = 49,
                               checkpoint_root: str | None = None) -> ModelCard:
    """Wan2.1-Fun-Control card. Same ``WanTransformer3DModel`` (in_ch=48, resolved from the checkpoint) /
    ``AutoencoderKLWan`` / UMT5 as Wan2.1 T2V + a CLIP reference-image encoder; reuses the ``WanDenoiseLoop``
    (its i2v ``cond=`` thread carries the control concat). Sampling defaults come from the
    ``wan_fun_1_3b_control`` preset (832x480, 49 frames, guidance 6.0, flow_shift 3.0, 50 steps)."""
    seed = _seed_from(model_id)
    cost = CostModel(kind=WorkUnitKind.DIFFUSION_STEP, base_seconds=1e-4, per_unit_seconds=1e-7)
    cfg, flow = ClassicCFG(), FlowShiftPolicy(shift=flow_shift)
    precision = PrecisionPolicy(compute_dtype="float32", scheduler_step_in_fp32=True)
    expert = NoRouting("transformer")

    def loop_factory() -> WanDenoiseLoop:
        return WanDenoiseLoop(loop_id="control_denoise",
                              cfg=cfg,
                              flow_shift=flow,
                              precision=precision,
                              expert=expert,
                              cost=cost)

    components = {
        "text_encoder":
        ComponentSpec("text_encoder",
                      kind="text_encoder",
                      load_id="fastvideo.models.encoders.t5:T5EncoderModel",
                      factory=lambda inst: ToyTextEncoder(),
                      required_for={"v2v"}),
        # Wan2.1-Fun-Control reference-image CLIP vision (WAN2_1ControlCLIPVisionConfig, bf16). Optional at
        # request time (the control video is the primary signal) but declared so the ref-image path works.
        "image_encoder":
        ComponentSpec(
            "image_encoder",
            kind="image_encoder",
            load_id="fastvideo.models.encoders.clip:CLIPVisionModel",  # BRINGUP: subfolder/dtype
            factory=lambda inst: ToyImageEncoder()),
        "vae":
        ComponentSpec("vae",
                      kind="vae",
                      load_id="fastvideo.models.vaes.wanvae:AutoencoderKLWan",
                      factory=lambda inst: ToyVAE(),
                      required_for={"v2v"}),
        "transformer":
        ComponentSpec("transformer",
                      kind="dit",
                      load_id="fastvideo.models.dits.wanvideo:WanTransformer3DModel",
                      factory=lambda inst: ToyDiT(seed=seed),
                      resident_for=["control_denoise"],
                      required_for={"v2v"}),
    }
    loops = {
        "control_denoise":
        LoopSpec("control_denoise",
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
        # Fun-Control is video-conditioned; map it to VIDEO_TO_VIDEO (+ decode). It degrades to T2V when no
        # control video is supplied (program writes i2v_cond=None -> WanDiT plain 16ch forward).
        capabilities=CapabilityMatrix.of(Capability.VIDEO_TO_VIDEO, Capability.VAE_DECODE),
        recipe=RecipeSpec(method="base",
                          assumes_loop="control_denoise",
                          assumes_precision="float32",
                          consistency_required=ConsistencyLevel.C1),
        parity=ParitySpec(consistency_levels=[ConsistencyLevel.C1], interleave_required=True),
        caches={"feature": CacheContract("feature", max_bytes=1 << 24, reuse_across_requests=True)},
        precision=PrecisionContract(default_dtype="float32", training_precision="float32"),
        parallelism=ParallelismContract(valid_plans=[ParallelPlan.single()], default_plan=ParallelPlan.single()),
        sampling_defaults=SamplingDefaults(num_steps=50,
                                           guidance_scale=6.0,
                                           height=height,
                                           width=width,
                                           num_frames=num_frames,
                                           fps=16,
                                           negative_prompt=_FUN_CONTROL_NEG),
    )
    card.validate()
    if checkpoint_root:
        stamp_wan21_checkpoints(card, checkpoint_root)
    return card
