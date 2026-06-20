"""Wan2.1 image-to-video (Fun-InP / I2V). Real i2v conditioning, mirroring fastvideo's
ImageEncodingStage + ImageVAEEncodingStage:
  * CLIP vision encodes the conditioning image -> ``i2v_img_embeds`` (the DiT's encoder_hidden_states_image);
  * the image is VAE-encoded as the first frame (rest zeros) -> ``cond_latent``; a 4-channel mask (first
    latent frame = 1) is prepended -> the 20-channel ``[mask|cond]`` written to ``i2v_cond``.
The shared WanDenoiseLoop picks both up from slots (its i2v hooks) and the Wan torch adapter concatenates
``[noise (16ch) | mask+cond (20ch)]`` -> the 36ch i2v DiT input. Reuses the Wan recipe/adapter/loop
unchanged otherwise.

BRINGUP (written-not-run): GPU-verify (a) Wan2.1-Fun-1.3B-InP loads via the generic Wan loader, (b) the
image_encoder subfolder/CLIP class + dtype, (c) that the generated video actually follows the conditioning
image (the mask/cond construction matches fastvideo; visual confirmation is human-in-the-loop)."""
from __future__ import annotations

import numpy as np

from v2._enums import Capability, ConsistencyLevel, LoopKind, WorkUnitKind
from v2.card import (
    CacheContract,
    CapabilityMatrix,
    ComponentSpec,
    LoopSpec,
    ModelCard,
    ParallelismContract,
    ParitySpec,
    PrecisionContract,
    RecipeSpec,
    SamplingDefaults,
)
from v2.loop.policies import BoundaryTimestepRouting, ClassicCFG, FlowShiftPolicy, NoRouting, PrecisionPolicy
from v2.parallel import ParallelPlan
from v2.platform.backends.toy import ToyDiT, ToyImageEncoder, ToyTextEncoder, ToyVAE, _seed_from
from v2.program import ComponentNode, ModelLoopNode, Program, ProgramKind
from v2.recipes._prompts import WAN_NEG_CN, WAN_NEG_EN
from v2.recipes.common import text_encode_node_fn as _text_encode
from v2.recipes.wan21.card import stamp_wan21_checkpoints
from v2.recipes.wan21.loop import WanDenoiseLoop

_WAN_TEMPORAL_RATIO = 4  # VAE temporal compression -> the i2v mask has this many channels


def build_wan21_i2v_card(model_id: str = "wan2.1-i2v-1.3b",
                         *,
                         flow_shift: float = 3.0,
                         height: int = 480,
                         width: int = 832,
                         num_frames: int = 81,
                         checkpoint_root: str | None = None) -> ModelCard:
    """Wan2.1 i2v (e.g. Wan2.1-Fun-1.3B-InP, WanI2V480PConfig). Same WanTransformer3DModel (in_ch=36) /
    AutoencoderKLWan / UMT5 as T2V + a CLIP image encoder; reuses the WanDenoiseLoop (i2v hooks)."""
    seed = _seed_from(model_id)
    cfg, flow = ClassicCFG(), FlowShiftPolicy(shift=flow_shift)
    precision, expert = PrecisionPolicy(compute_dtype="float32", scheduler_step_in_fp32=True), NoRouting("transformer")

    def loop_factory():
        return WanDenoiseLoop(loop_id="i2v_denoise",
                              cfg=cfg,
                              flow_shift=flow,
                              precision=precision,
                              expert=expert,
                              )

    components = {
        "text_encoder":
        ComponentSpec("text_encoder",
                      kind="text_encoder",
                      load_id="fastvideo.models.encoders.t5:T5EncoderModel",
                      factory=lambda inst: ToyTextEncoder(),
                      required_for={"i2v"}),
        "image_encoder":
        ComponentSpec(
            "image_encoder",
            kind="image_encoder",
            load_id="fastvideo.models.encoders.clip:CLIPVisionModel",  # BRINGUP: class/subfolder
            factory=lambda inst: ToyImageEncoder(),
            required_for={"i2v"}),
        "vae":
        ComponentSpec("vae",
                      kind="vae",
                      load_id="fastvideo.models.vaes.wanvae:AutoencoderKLWan",
                      factory=lambda inst: ToyVAE(),
                      required_for={"i2v"}),
        "transformer":
        ComponentSpec("transformer",
                      kind="dit",
                      load_id="fastvideo.models.dits.wanvideo:WanTransformer3DModel",
                      factory=lambda inst: ToyDiT(seed=seed),
                      resident_for=["i2v_denoise"],
                      required_for={"i2v"}),
    }
    loops = {
        "i2v_denoise":
        LoopSpec("i2v_denoise",
                 kind=LoopKind.DIFFUSION_DENOISE,
                 work_unit_kind=WorkUnitKind.DIFFUSION_STEP,
                 shared_weight_components=["transformer"],
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
        sampling_defaults=SamplingDefaults(num_steps=40,
                                           guidance_scale=5.0,
                                           height=height,
                                           width=width,
                                           num_frames=num_frames,
                                           fps=16,
                                           negative_prompt=WAN_NEG_EN),
    )
    card.validate()
    if checkpoint_root:
        stamp_wan21_checkpoints(card, checkpoint_root)
    return card


def build_wan22_i2v_a14b_card(model_id: str = "wan2.2-i2v-a14b",
                              *,
                              boundary: float = 0.9,
                              checkpoint_root: str | None = None) -> ModelCard:
    """Wan2.2-I2V-A14B — MoE (two WanTransformer3DModel experts + boundary routing) + i2v conditioning
    (CLIP + first-frame [mask|cond]). Reuses the Wan adapter (cond concat + MoE CPU offload), the shared
    WanDenoiseLoop (i2v hooks + the boundary expert), and the i2v program. Structural (GPU-pending: 2x14B)."""
    seed = _seed_from(model_id)
    cfg, flow = ClassicCFG(), FlowShiftPolicy(shift=5.0)
    precision = PrecisionPolicy(compute_dtype="float32", scheduler_step_in_fp32=True)
    expert = BoundaryTimestepRouting(high_noise="transformer", low_noise="transformer_2", boundary=boundary)

    def loop_factory():
        return WanDenoiseLoop(loop_id="i2v_denoise",
                              cfg=cfg,
                              flow_shift=flow,
                              precision=precision,
                              expert=expert,
                              )

    def _dit(cid: str) -> ComponentSpec:
        return ComponentSpec(cid,
                             kind="dit",
                             load_id="fastvideo.models.dits.wanvideo:WanTransformer3DModel",
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
        "image_encoder":
        ComponentSpec("image_encoder",
                      kind="image_encoder",
                      load_id="fastvideo.models.encoders.clip:CLIPVisionModel",
                      factory=lambda inst: ToyImageEncoder(),
                      required_for={"i2v"}),
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
        sampling_defaults=SamplingDefaults(num_steps=40,
                                           guidance_scale=3.5,
                                           height=480,
                                           width=832,
                                           num_frames=81,
                                           fps=16,
                                           negative_prompt=WAN_NEG_CN),
    )
    card.validate()
    if checkpoint_root:
        stamp_wan21_checkpoints(card, checkpoint_root)
    return card


# --- program: CLIP encode + first-frame VAE conditioning + i2v denoise + decode ------------------ #
def _i2v_image_encode(instance, slots, request, ctx) -> None:
    img = request.image()
    slots["i2v_img_embeds"] = (instance.component("image_encoder").encode_image(img.pixels)
                               if img is not None and getattr(img, "pixels", None) is not None else None)


def _i2v_cond_encode(instance, slots, request, ctx) -> None:
    """First-frame VAE conditioning + mask, mirroring fastvideo's ImageVAEEncodingStage: encode the
    conditioning image as frame 0 (rest zeros), prepend a 4-channel mask (first latent frame = 1)."""
    img = request.image()
    if img is None or getattr(img, "pixels", None) is None:
        slots["i2v_cond"] = None
        return
    px = np.asarray(img.pixels, dtype="float32")  # [3, H, W] in [-1, 1]
    nf = int(request.diffusion.num_frames)
    cond_video = np.zeros((px.shape[0], nf) + px.shape[1:], dtype="float32")
    cond_video[:, 0] = px  # frame 0 = conditioning image, rest zeros
    cond_latent = np.asarray(instance.component("vae").encode(cond_video), dtype="float32")  # [C, T, h, w]
    t, h, w = cond_latent.shape[1:]
    mask = np.zeros((_WAN_TEMPORAL_RATIO, t, h, w), dtype="float32")
    mask[:, 0] = 1.0  # first latent frame is the known (conditional) frame
    slots["i2v_cond"] = np.concatenate([mask, cond_latent], axis=0)  # [mask | cond] -> WanDiT concats with noise


def _i2v_vae_decode(instance, slots, request, ctx) -> None:
    slots["video"] = instance.component("vae").decode(slots["i2v_out"]["latents"])


def build_wan21_i2v_program() -> Program:
    return Program(
        program_id="wan21.i2v",
        kind=ProgramKind.INLINE,
        nodes=[
            ComponentNode("image_encode", fn=_i2v_image_encode, writes=("i2v_img_embeds", )),
            ComponentNode("cond_encode", fn=_i2v_cond_encode, writes=("i2v_cond", )),
            ComponentNode("text_encode", fn=_text_encode, writes=("text_embeds", "neg_text_embeds")),
            ModelLoopNode("denoise",
                          loop_id="i2v_denoise",
                          output_slot="i2v_out",
                          reads=("text_embeds", "i2v_cond", "i2v_img_embeds"),
                          writes=("i2v_out", )),
            ComponentNode("vae_decode", fn=_i2v_vae_decode, reads=("i2v_out", ), writes=("video", )),
        ],
        output_artifacts={
            "video": "video",
            "latents": "i2v_out"
        },
    ).validate()
