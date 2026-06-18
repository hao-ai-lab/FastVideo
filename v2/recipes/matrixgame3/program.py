"""Matrix-Game 3.0 program: text_encode -> image VAE-encode (first frame) -> mg3_denoise -> vae_decode.

Mirrors the cosmos2 / Wan i2v node graph. The MG3 specifics (per-token timestep, first-frame pinning,
autoregressive clip loop) live entirely in ``MatrixGame3DenoiseLoop`` + the ``MatrixGame3DiT`` adapter, so
the node graph is the standard i2v shape:

  * ``text_encode``   — T5 prompt + (empty) negative prompt -> ``text_embeds`` / ``neg_text_embeds``.
  * ``image_encode``  — VAE-encode the conditioning image as the first frame -> ``image_latent`` (the
    ``MatrixGame3ImageVAEEncodingStage`` normalization: ``(z - latents_mean)/latents_std``, which the
    reused ``WanVAE`` adapter applies inside ``encode``). The loop pins the first ``cond_frames`` latent
    frames to this and re-pastes after every scheduler step. Absent (no image) -> ``image_latent=None``
    -> the loop runs the degenerate flow-match denoise (the CPU-toy path).
  * ``denoise``       — ``MatrixGame3DenoiseLoop`` reads ``text_embeds`` + ``image_latent`` from slots.
  * ``vae_decode``    — light_vae decode -> video.

BRINGUP: the action (mouse/keyboard) + camera (Plücker) conditioning would default to
``build_matrixgame3_action_preset(total_video_frames, seed)`` + a derived camera trajectory and be
written into slots here; that needs a request-API extension to carry the action streams + camera path
(see the loop's ``_build_clip_cond`` hook). The single-clip no-action path is what the registered preset
+ CPU verification exercise.
"""
from __future__ import annotations

import numpy as np

from v2.program import ComponentNode, ModelLoopNode, Program, ProgramKind
from v2.recipes.common import text_encode_node_fn as _text_encode


def _mg3_image_encode(instance, slots, request, ctx) -> None:
    """VAE-encode the conditioning image as the first frame (rest zeros) -> ``image_latent`` (normalized
    by the WanVAE adapter's ``_mean_invstd`` path, matching ``MatrixGame3ImageVAEEncodingStage``). No
    image -> ``image_latent`` stays None and the loop runs the degenerate flow-match denoise."""
    img = request.image()
    if img is None or getattr(img, "pixels", None) is None:
        slots["image_latent"] = None
        return
    px = np.asarray(img.pixels, dtype="float32")  # [3, H, W] in [-1, 1]
    # A single-frame "video" tensor [3, 1, H, W] -> the VAE encodes one latent frame the loop pins.
    cond_video = px[:, None]
    slots["image_latent"] = np.asarray(instance.component("vae").encode(cond_video), dtype="float32")


def _mg3_vae_decode(instance, slots, request, ctx) -> None:
    slots["video"] = instance.component("vae").decode(slots["denoise_out"]["latents"])


def build_matrixgame3_program() -> Program:
    return Program(
        program_id="matrixgame3.i2v.inline",
        kind=ProgramKind.INLINE,
        nodes=[
            ComponentNode("text_encode", fn=_text_encode, writes=("text_embeds", "neg_text_embeds")),
            ComponentNode("image_encode", fn=_mg3_image_encode, writes=("image_latent", )),
            ModelLoopNode("denoise",
                          loop_id="mg3_denoise",
                          output_slot="denoise_out",
                          reads=("text_embeds", "neg_text_embeds", "image_latent"),
                          writes=("denoise_out", )),
            ComponentNode("vae_decode", fn=_mg3_vae_decode, reads=("denoise_out", ), writes=("video", )),
        ],
        output_artifacts={
            "video": "video",
            "latents": "denoise_out"
        },
    ).validate()
