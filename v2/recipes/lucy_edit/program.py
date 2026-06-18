"""Lucy-Edit v2v program: video_vae_encode -> text_encode -> denoise -> vae_decode.

The one NEW node vs the Wan t2v program is ``video_vae_encode`` — the v2 analogue of fastvideo's
``VideoVAEEncodingStage``: VAE-encode the INPUT video (the clip being edited) into a conditioning
latent that the denoise loop channel-concatenates with the noise latent each step
(``[noise(48ch) | video_latent(48ch)]`` -> the 96ch Lucy DiT input). Faithful to
``fastvideo/pipelines/stages/image_encoding.py:VideoVAEEncodingStage`` + the ``is_lucy_edit`` concat
in ``denoising.py``:
  * encode the input video frames as a full latent (NOT a first-frame-only image like i2v);
  * Lucy uses the VAE distribution **mode** (``sample_mode="argmax"``), not a sampled latent;
  * NO 4-channel binary mask and NO CLIP image embedding (``i2v_img_embeds`` stays None) — those are
    the i2v conditioning; Lucy conditions on the raw video latent only.

The conditioning latent is written to the ``i2v_cond`` slot, which the shared ``WanDenoiseLoop``
reads and passes to the Wan adapter as ``cond=`` (its channel-concat path). Reuses
``WanDenoiseLoop`` + the Wan torch adapter unchanged.

BRINGUP: v2's ``Request`` carries a ``VideoPart`` (modality VIDEO) but the input-video plumbing is
not wired end to end yet. The node reads the first ``VideoPart`` from ``request.inputs`` if present
and DEGRADES TO T2V otherwise (``i2v_cond`` = None -> the loop runs the plain Wan t2v forward, i.e.
text-only generation at the requested resolution). On the CPU toy backend the conditioning path is
exercised whenever a ``VideoPart`` with frames is supplied.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from v2.program import ComponentNode, ModelLoopNode, Program, ProgramKind
from v2.recipes.common import text_encode_node_fn as _text_encode
from v2.request.modalpart import Modality


def _input_video(request: Any) -> Any:
    """Return the first VideoPart on the request (v2 has no ``Request.video()`` accessor yet), or None.
    BRINGUP seam: this is the single point where the (not-yet-wired) input-video plumbing attaches."""
    for part in getattr(request, "inputs", ()):
        if getattr(part, "modality", None) == Modality.VIDEO and getattr(part, "frames", None) is not None:
            return part
    return None


def _video_vae_encode(instance, slots, request, ctx) -> None:
    """VAE-encode the input video -> the conditioning latent (mirrors VideoVAEEncodingStage). Degrades
    to t2v (i2v_cond = None) when no input video is supplied (BRINGUP: input-video plumbing)."""
    part = _input_video(request)
    if part is None:
        slots["i2v_cond"] = None
        slots["i2v_img_embeds"] = None  # Lucy has NO CLIP image encoder; kept None for the loop hook
        return
    # Frames arrive [T, H, W, C] (VideoPart.frames) in [-1, 1]; the VAE encodes a [C, T, H, W] video. The
    # toy VAE is shape-tolerant; the GPU WanVAE consumes the real layout (its preprocess handles dtype/range).
    frames = np.asarray(part.frames, dtype="float32")
    video = np.moveaxis(frames, -1, 0) if frames.ndim == 4 else frames  # [T,H,W,C] -> [C,T,H,W]
    # Lucy uses the VAE distribution MODE (sample_mode="argmax"); the toy VAE.encode is deterministic, so
    # this is the mode by construction. The full video latent (no mask) is the conditioning.
    video_latent = np.asarray(instance.component("vae").encode(video), dtype="float32")  # [C, T, h, w]
    slots["i2v_cond"] = video_latent
    slots["i2v_img_embeds"] = None


def _vae_decode(instance, slots, request, ctx) -> None:
    slots["video"] = instance.component("vae").decode(slots["lucy_out"]["latents"])


def build_lucy_edit_program() -> Program:
    return Program(
        program_id="lucy_edit.v2v",
        kind=ProgramKind.INLINE,
        nodes=[
            ComponentNode("video_vae_encode", fn=_video_vae_encode, writes=("i2v_cond", "i2v_img_embeds")),
            ComponentNode("text_encode", fn=_text_encode, writes=("text_embeds", "neg_text_embeds")),
            ModelLoopNode("denoise",
                          loop_id="lucy_edit_denoise",
                          output_slot="lucy_out",
                          reads=("text_embeds", "neg_text_embeds", "i2v_cond", "i2v_img_embeds"),
                          writes=("lucy_out", )),
            ComponentNode("vae_decode", fn=_vae_decode, reads=("lucy_out", ), writes=("video", )),
        ],
        output_artifacts={
            "video": "video",
            "latents": "lucy_out"
        },
    ).validate()
