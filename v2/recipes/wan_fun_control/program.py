"""Wan2.1-Fun-Control program: (ref-image CLIP encode) + control-video VAE encode → denoise → decode.

Faithful port of ``fastvideo/pipelines/basic/wan/wan_v2v_pipeline.py`` + its stages:

  * ``RefImageEncodingStage`` (image_encoding.py): a CLIP-vision reference image → ``image_embeds`` passed
    to the DiT as ``encoder_hidden_states_image``. Optional — defaults to a zero/None embed when absent.
  * ``VideoVAEEncodingStage`` (image_encoding.py): the CONTROL VIDEO is VAE-encoded to a latent
    ``control_latent`` (scaled into the DiT's normalized latent space by the ``WanVAE`` adapter, exactly
    like the noise latent).
  * ``DenoisingStage`` (denoising.py L380-390): for the V2V/Control path the DiT input is
    ``cat([noise_latent, control_latent, v2v_zero_pad], dim=channels)`` — i.e. noise(16) + control(16) +
    a 16-channel zero pad = the 48-channel Fun-Control DiT input (``WanVideoArchConfig`` in_channels=48).

We carry the ``[control_latent | zero_pad]`` (32ch) concat through the *existing* ``i2v_cond`` slot the
shared ``WanDenoiseLoop`` already threads to the ``WanDiT`` adapter's ``cond=`` kwarg. The adapter then
does ``cat([noise(16ch), cond(32ch)], dim=channels)`` → the 48ch DiT input, byte-for-byte the V2V concat
order ``[latent, video_latent, zero_pad]``. So the Wan recipe/adapter/loop are reused unchanged.

BRINGUP (written-not-run): the control-video request input. v2's ``Request`` has no ``video()`` accessor
yet, so we scan ``request.inputs`` for a ``VideoPart`` ourselves. When NO control video is present this
DEGRADES gracefully: ``i2v_cond`` is ``None`` → ``WanDiT`` runs the plain 16ch forward → pure T2V (and a
ref image, if any, still flows as i2v-style CLIP context). GPU-verify: (a) Wan2.1-Fun-1.3B-Control loads
via the generic Wan loader with in_channels=48, (b) the control video drives the motion, (c) the
``WAN2_1ControlCLIPVisionConfig`` reference-image encoder subfolder/dtype.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from v2.program import ComponentNode, ModelLoopNode, Program, ProgramKind
from v2.recipes.common import text_encode_node_fn as _text_encode


def _control_video_part(request: Any) -> Any:
    """Find the control-video input (a ``VideoPart``) on the request, or ``None`` for t2v degrade.

    ``Request`` exposes ``image()`` but not ``video()``, and we must not edit shared files, so we scan
    ``request.inputs`` for the VIDEO modality directly (duck-typed on ``.frames``)."""
    for part in getattr(request, "inputs", ()) or ():
        if getattr(part, "modality", None) == "video" or getattr(part, "frames", None) is not None:
            return part
    return None


def _ref_image_encode(instance: Any, slots: dict, request: Any, ctx: Any) -> None:
    """Optional reference image → CLIP-vision context (``RefImageEncodingStage``). Carried through the
    i2v ``i2v_img_embeds`` slot the WanDiT adapter passes as ``encoder_hidden_states_image``. Absent ->
    None (the adapter passes no image context; Fun-Control's primary signal is the control video)."""
    img = request.image()
    slots["i2v_img_embeds"] = (instance.component("image_encoder").encode_image(img.pixels)
                               if img is not None and getattr(img, "pixels", None) is not None else None)


def _control_video_encode(instance: Any, slots: dict, request: Any, ctx: Any) -> None:
    """Control video → VAE latent, then the Fun-Control ``[control_latent | zero_pad]`` concat.

    Mirrors ``VideoVAEEncodingStage`` (VAE-encode the control video) + the V2V concat in ``DenoisingStage``
    (``cat([latent, video_latent, zero_pad])``). We pre-build the ``[video_latent | zero_pad]`` (32ch) half
    so the loop/adapter only has to ``cat([noise(16ch), this(32ch)])`` → the 48ch Fun-Control DiT input.

    BRINGUP: the Fun-Control DiT's ``patch_embedding`` is a Conv3d with ``in_channels=48`` — it ALWAYS
    requires the 48ch input (no pure-16ch T2V path). The real pipeline reflects this: when no control
    video is supplied, ``RefImageEncodingStage`` still sets ``video_latent = zeros`` and
    ``VideoVAEEncodingStage`` VAE-encodes it (an asserted, non-optional stage). So we NEVER emit
    ``i2v_cond=None``; with no control video we VAE-encode a ZERO control video (faithful to the real
    zeros-``video_latent`` path) and concat the zero pad — the 48ch input is always well-formed."""
    nf = int(request.diffusion.num_frames)
    part = _control_video_part(request)
    if part is None or getattr(part, "frames", None) is None:
        # No control video: encode a zero control video (the real pipeline's zeros-``video_latent`` path).
        # Pixel-space zeros -> VAE -> the model's "empty control" latent; concat with the zero pad → 32ch.
        h, w = int(request.diffusion.height), int(request.diffusion.width)
        frames = np.zeros((3, nf, h, w), dtype="float32")
    else:
        frames = np.asarray(part.frames, dtype="float32")  # [T, H, W, 3] or [3, T, H, W]
        if frames.ndim == 4 and frames.shape[-1] == 3:  # [T, H, W, 3] -> [3, T, H, W]
            frames = np.transpose(frames, (3, 0, 1, 2))
        # Match the requested frame count: truncate, or zero-pad the tail (``_prepare_control_video_tensor``).
        t = frames.shape[1]
        if t > nf:
            frames = frames[:, :nf]
        elif t < nf:
            pad = np.zeros((frames.shape[0], nf - t) + frames.shape[2:], dtype="float32")
            frames = np.concatenate([frames, pad], axis=1)
    control_latent = np.asarray(instance.component("vae").encode(frames), dtype="float32")  # [16, T, h, w]
    zero_pad = np.zeros_like(control_latent)  # the V2V zero pad (fixed shape, never written)
    # [control_latent(16) | zero_pad(16)] -> WanDiT concats this AFTER the 16ch noise -> 48ch Fun-Control input.
    slots["i2v_cond"] = np.concatenate([control_latent, zero_pad], axis=0)


def _vae_decode(instance: Any, slots: dict, request: Any, ctx: Any) -> None:
    slots["video"] = instance.component("vae").decode(slots["control_out"]["latents"])


def build_wan_fun_control_program() -> Program:
    return Program(
        program_id="wan_fun_control.v2v.inline",
        kind=ProgramKind.INLINE,
        nodes=[
            ComponentNode("ref_image_encode", fn=_ref_image_encode, writes=("i2v_img_embeds", )),
            ComponentNode("control_encode", fn=_control_video_encode, writes=("i2v_cond", )),
            ComponentNode("text_encode", fn=_text_encode, writes=("text_embeds", "neg_text_embeds")),
            ModelLoopNode("denoise",
                          loop_id="control_denoise",
                          output_slot="control_out",
                          reads=("text_embeds", "neg_text_embeds", "i2v_cond", "i2v_img_embeds"),
                          writes=("control_out", )),
            ComponentNode("vae_decode", fn=_vae_decode, reads=("control_out", ), writes=("video", )),
        ],
        output_artifacts={
            "video": "video",
            "latents": "control_out"
        },
    ).validate()
