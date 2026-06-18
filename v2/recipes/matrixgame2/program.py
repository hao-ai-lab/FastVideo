"""Matrix-Game 2.0 i2v world-rollout program: image_encode(CLIP) -> cond_encode(first-frame VAE) ->
action_prepare(BRINGUP hook) -> causal DMD denoise -> vae_decode.

Deltas vs the Wan2.1 i2v program (faithful to ``matrixgame2_causal_dmd_pipeline.py`` +
``MatrixGame2ImageVAEEncodingStage``):
  * NO text_encode node — Matrix-Game 2.0 ignores text; the CLIP image embeds are the sole cross-attn
    context (written to ``i2v_img_embeds``).
  * cond_encode writes the RAW first-frame VAE latent to ``i2v_cond`` with NO 4-channel mask (unlike Wan2.1
    i2v's ``[mask|cond]``); the DiT channel-concats this raw cond onto the noise latent internally (the
    patch_embedding is the 32-in checkpoint).
  * action_prepare is a HOOK that routes per-frame mouse/keyboard arrays into ``mouse_cond``/``keyboard_cond``
    when the request carries them. BRINGUP: the request API does not yet surface game-action arrays, so this
    is a no-op on the registered path (the world model degrades to a first-frame-conditioned rollout). When
    the request-API extension lands, this node slices the action arrays per block (vae_time_compression=4).
"""
from __future__ import annotations

import numpy as np

from v2.program import ComponentNode, ModelLoopNode, Program, ProgramKind


def _image_encode(instance, slots, request, ctx) -> None:
    """CLIP-encode the first-frame conditioning image -> 257x1280 cross-attn context (sole conditioning)."""
    img = request.image()
    slots["i2v_img_embeds"] = (instance.component("image_encoder").encode_image(img.pixels)
                               if img is not None and getattr(img, "pixels", None) is not None else None)


def _cond_encode(instance, slots, request, ctx) -> None:
    """First-frame VAE conditioning, mirroring ``MatrixGame2ImageVAEEncodingStage``: encode the conditioning
    image as frame 0 (rest zeros) -> the RAW cond_latent (NO mask channel). The DiT channel-concats it onto
    the noise latent (dim=1). Same VAE normalization as the noise latents (the WanVAE adapter handles it)."""
    img = request.image()
    if img is None or getattr(img, "pixels", None) is None:
        slots["i2v_cond"] = None
        return
    px = np.asarray(img.pixels, dtype="float32")  # [3, H, W] in [-1, 1]
    nf = int(request.diffusion.num_frames)
    cond_video = np.zeros((px.shape[0], nf) + px.shape[1:], dtype="float32")
    cond_video[:, 0] = px  # frame 0 = conditioning image, rest zeros
    slots["i2v_cond"] = np.asarray(instance.component("vae").encode(cond_video), dtype="float32")  # raw [C,T,h,w]


def _action_prepare(instance, slots, request, ctx) -> None:
    """Route per-frame game actions (mouse[F,2] / keyboard[F,4]) into the loop's conditioning slots.
    BRINGUP: the request API does not yet expose action arrays, so this is a no-op (the world model runs the
    degenerate first-frame-conditioned rollout). When the action request-surface lands, read it here and the
    loop/adapter feed the ActionModule KV caches. Slots stay ``None`` until then."""
    slots["mouse_cond"] = getattr(getattr(request, "diffusion", None), "mouse_cond", None)
    slots["keyboard_cond"] = getattr(getattr(request, "diffusion", None), "keyboard_cond", None)


def _vae_decode(instance, slots, request, ctx) -> None:
    slots["video"] = instance.component("vae").decode(slots["denoise_out"]["latents"])


def build_matrixgame2_program() -> Program:
    return Program(
        program_id="matrixgame2.i2v.causal_dmd",
        kind=ProgramKind.INLINE,
        nodes=[
            ComponentNode("image_encode", fn=_image_encode, writes=("i2v_img_embeds", )),
            ComponentNode("cond_encode", fn=_cond_encode, writes=("i2v_cond", )),
            ComponentNode("action_prepare", fn=_action_prepare, writes=("mouse_cond", "keyboard_cond")),
            ModelLoopNode("denoise",
                          loop_id="causal_dmd_denoise",
                          output_slot="denoise_out",
                          reads=("i2v_img_embeds", "i2v_cond", "mouse_cond", "keyboard_cond"),
                          writes=("denoise_out", )),
            ComponentNode("vae_decode", fn=_vae_decode, reads=("denoise_out", ), writes=("video", )),
        ],
        output_artifacts={
            "video": "video",
            "latents": "denoise_out"
        },
    ).validate()
