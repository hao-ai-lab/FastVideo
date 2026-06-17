"""T2I and I2V programs + the cross-model workflow that chains them (design_v3 §13, §15).

    Workflow "t2i_then_i2v":
        stage 1 (flux-t2i):  text_encode → t2i_denoise → vae_decode        → image
        stage 2 (wan-i2v):   encode_cond_image(image→cond) + text_encode
                             → condition_on_image(fold cond into text_embeds)
                             → i2v_denoise → vae_decode                     → video

The two stages run on two *different* registered instances; the ``Workflow`` threads the stage-1
``image`` artifact into the stage-2 request as an ``ImagePart``. The image-conditioning is a program
node (``condition_on_image``) that folds the image latent into the I2V conditioning — so the
unchanged ``WanDenoiseLoop`` produces an image-dependent video. Different image ⇒ different video.
"""
from __future__ import annotations

import numpy as np

from ...program import ComponentNode, ModelLoopNode, Program, ProgramKind, Workflow, WorkflowStage
from ...request import DiffusionParams, TaskType, make_request
from ...request.modalpart import ImagePart, TextPart
from ..common import text_encode_node_fn as _text_encode


# --- stage 1: text → image -------------------------------------------------------- #
def _t2i_vae_decode(instance, slots, request, ctx) -> None:
    slots["image"] = instance.component("vae").decode(slots["t2i_out"]["latents"])


def build_flux_t2i_program() -> Program:
    return Program(
        program_id="flux.t2i", kind=ProgramKind.INLINE,
        nodes=[
            ComponentNode("text_encode", fn=_text_encode, writes=("text_embeds", "neg_text_embeds")),
            ModelLoopNode("denoise", loop_id="t2i_denoise", output_slot="t2i_out",
                          reads=("text_embeds",), writes=("t2i_out",)),
            ComponentNode("vae_decode", fn=_t2i_vae_decode, reads=("t2i_out",), writes=("image",)),
        ],
        output_artifacts={"image": "image", "latents": "t2i_out"},
    ).validate()


# --- stage 2: text + image → video ------------------------------------------------ #
def _encode_cond_image(instance, slots, request, ctx) -> None:
    """VAE-encode the conditioning image to a latent (the I2V conditioning signal)."""
    img = request.image()
    if img is not None and img.pixels is not None:
        slots["cond_latent"] = instance.component("vae").encode(np.asarray(img.pixels, dtype="float32"))
    else:
        slots["cond_latent"] = None


def _condition_on_image(instance, slots, request, ctx) -> None:
    """Fold the conditioning-image latent into the text conditioning, so the denoise output depends
    on the image (the I2V hand-off — keeps ``WanDenoiseLoop`` unchanged)."""
    cond = slots.get("cond_latent")
    if cond is None:
        return
    shift = float(np.tanh(np.mean(np.asarray(cond, dtype="float64"))))   # image-derived conditioning
    te = np.asarray(slots["text_embeds"], dtype="float32")
    slots["text_embeds"] = (te + shift).astype("float32")
    slots["image_shift"] = shift


def _i2v_vae_decode(instance, slots, request, ctx) -> None:
    slots["video"] = instance.component("vae").decode(slots["i2v_out"]["latents"])


def build_wan_i2v_program() -> Program:
    return Program(
        program_id="wan.i2v", kind=ProgramKind.INLINE,
        nodes=[
            ComponentNode("encode_cond_image", fn=_encode_cond_image, writes=("cond_latent",)),
            ComponentNode("text_encode", fn=_text_encode, writes=("text_embeds", "neg_text_embeds")),
            ComponentNode("condition_on_image", fn=_condition_on_image,
                          reads=("cond_latent", "text_embeds"), writes=("text_embeds", "image_shift")),
            ModelLoopNode("denoise", loop_id="i2v_denoise", output_slot="i2v_out",
                          reads=("text_embeds",), writes=("i2v_out",)),
            ComponentNode("vae_decode", fn=_i2v_vae_decode, reads=("i2v_out",), writes=("video",)),
        ],
        output_artifacts={"video": "video", "latents": "i2v_out"},
    ).validate()


# --- the cross-model workflow ----------------------------------------------------- #
def build_t2i_then_i2v_workflow(t2i_id: str = "flux-t2i", i2v_id: str = "wan-i2v") -> Workflow:
    def t2i_stage(state):
        return make_request(TaskType.T2I, t2i_id, state["prompt"],
                            diffusion=DiffusionParams(num_steps=4, num_frames=1,
                                                      seed=state.get("seed", 0)))

    def i2v_stage(state):
        image = state["t2i:image"].tensor            # the decoded image from stage 1
        return make_request(TaskType.I2V, i2v_id, state["prompt"],
                            inputs=(TextPart(state["prompt"]), ImagePart(pixels=image)),
                            diffusion=DiffusionParams(num_steps=4, num_frames=81,   # → multi-frame video
                                                      seed=state.get("seed", 0)))

    return Workflow("image_video.t2i_i2v", [          # namespaced id: <package>.<pipeline> (a servable)
        WorkflowStage(t2i_id, t2i_stage, label="t2i"),
        WorkflowStage(i2v_id, i2v_stage, label="i2v"),
    ])
