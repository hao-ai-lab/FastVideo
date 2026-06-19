"""Matrix-Game 2.0 i2v world-rollout program: image_encode(CLIP) -> cond_encode(first-frame VAE) ->
action_prepare(BRINGUP hook) -> causal DMD denoise -> vae_decode.

Deltas vs the Wan2.1 i2v program (faithful to ``matrixgame2_causal_dmd_pipeline.py`` +
``MatrixGame2ImageVAEEncodingStage``):
  * NO text_encode node — Matrix-Game 2.0 ignores text; the CLIP image embeds are the sole cross-attn
    context (written to ``i2v_img_embeds``).
  * cond_encode writes a 20-channel ``cond_concat`` (4 first-frame mask channels + 16 VAE-latent channels)
    to ``i2v_cond``; the DiT channel-concats this onto the 16ch noise latent internally.
  * action_prepare is a HOOK that routes per-frame mouse/keyboard arrays into ``mouse_cond``/``keyboard_cond``
    when the request carries them. BRINGUP: the request API does not yet surface game-action arrays, so this
    is a no-op on the registered path (the world model degrades to a first-frame-conditioned rollout). When
    the request-API extension lands, this node slices the action arrays per block (vae_time_compression=4).
"""
from __future__ import annotations

import numpy as np

from v2.program import ComponentNode, ModelLoopNode, Program, ProgramKind

# Wan2.1 VAE compression (z=16, 8x spatial, 4x temporal). The DiT's i2v cond_concat is 20ch:
# 4 mask channels + 16 VAE-latent channels (faithful to MatrixGame2ImageVAEEncodingStage).
_MG2_LATENT_CHANNELS = 16
_MG2_MASK_CHANNELS = 4
_MG2_SPATIAL_RATIO = 8
_MG2_TEMPORAL_RATIO = 4


def _first_frame_pixels(request) -> np.ndarray:
    """Return the first-frame conditioning image as ``[3, H, W]`` float32 in [-1, 1]. A first frame is
    MANDATORY: it is the only spatial conditioning and drives the cond_concat for the DiT's 36 in-channels.
    When the request carries no image (degenerate world-rollout / plain text smoke request), synthesize a
    neutral mid-gray frame at the requested resolution and extend from there. Mirrors fastvideo's
    ``create_default_image`` fallback in ``MatrixGame2ImageEncodingStage`` (``batch.pil_image is None``)."""
    img = request.image()
    h = int(request.diffusion.height)
    w = int(request.diffusion.width)
    if img is not None and getattr(img, "pixels", None) is not None:
        px = np.asarray(img.pixels, dtype="float32")
        if px.ndim == 3 and px.shape[0] not in (1, 3) and px.shape[-1] in (1, 3):
            px = np.transpose(px, (2, 0, 1))  # HWC -> CHW
        if px.max() > 1.5:  # uint8-ish [0,255] -> [-1, 1]
            px = px / 127.5 - 1.0
        return px.astype("float32")
    return np.zeros((3, h, w), dtype="float32")  # neutral (mid-gray in [-1,1]) blank first frame


def _image_encode(instance, slots, request, ctx) -> None:
    """CLIP-encode the first-frame conditioning image -> 257x1280 cross-attn context (the SOLE conditioning;
    Matrix-Game 2.0 has no text encoder, so this cross-attn context must always be present)."""
    px = _first_frame_pixels(request)
    slots["i2v_img_embeds"] = instance.component("image_encoder").encode_image(px)


def _cond_encode(instance, slots, request, ctx) -> None:
    """First-frame VAE conditioning, mirroring ``MatrixGame2ImageVAEEncodingStage``: encode the conditioning
    image as frame 0 (rest zeros) -> the normalized 16ch ``img_cond`` latent (the WanVAE adapter applies the
    (z-mean)/std normalization + deterministic ``mode()``), then PREPEND the 4-channel first-frame mask to
    form the 20-channel ``cond_concat`` (``cat([mask[:4], img_cond], dim=1)``). The DiT channel-concats this
    onto the 16ch noise latent internally (dim=1) -> 36 in-channels (the checkpoint's ``in_channels=36``)."""
    px = _first_frame_pixels(request)  # [3, H, W] in [-1, 1]
    nf = int(request.diffusion.num_frames)
    cond_video = np.zeros((px.shape[0], nf) + px.shape[1:], dtype="float32")
    cond_video[:, 0] = px  # frame 0 = conditioning image, rest zeros
    img_cond = np.asarray(instance.component("vae").encode(cond_video), dtype="float32")  # [16, t, h, w]
    # First-frame latent mask: ones on latent-frame 0, zeros elsewhere; keep the first 4 channels.
    mask = np.zeros_like(img_cond)
    mask[:, 0] = 1.0
    cond_concat = np.concatenate([mask[:_MG2_MASK_CHANNELS], img_cond], axis=0)  # [20, t, h, w]
    slots["i2v_cond"] = cond_concat.astype("float32")


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
