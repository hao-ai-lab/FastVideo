# SPDX-License-Identifier: Apache-2.0
"""LTX-2 image / continuation conditioning helpers.

This is a *minimal* port of ``FastVideo-internal/.../ltx2_i2v_conditioning.py``
that captures just the surface needed for the spatial-refine (SR)
pipeline's stage-2 path:

* the four ``ForwardBatch.extra`` keys (clean latent, denoise mask,
  stage-1/stage-2 last-latent for continuation),
* :func:`apply_ltx2_gaussian_noiser` — used by both denoising and refine
  to mix noise into the conditioning latent,
* :class:`LTX2ImageConditioningState` and a T2V-only
  :func:`build_ltx2_image_conditioning` that returns ``None`` when the
  request carries no image inputs and no continuation latents.

The full image-loading path (CRF re-encode, decode, conditioning latent
extraction, multi-image insertion) lives in the internal module and
will be ported separately when the public surface needs i2v / mid-roll
continuation. SR text-to-video does not exercise it, so keeping this
module narrow keeps the public stage graph honest.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from fastvideo.pipelines.pipeline_batch_info import ForwardBatch

LTX2_VIDEO_CLEAN_LATENT_KEY = "ltx2_video_clean_latent"
LTX2_VIDEO_DENOISE_MASK_KEY = "ltx2_video_denoise_mask"
LTX2_CONTINUATION_STAGE1_LAST_LATENT_KEY = "ltx2_continuation_stage1_last_latent"
LTX2_CONTINUATION_STAGE2_LAST_LATENT_KEY = "ltx2_continuation_stage2_last_latent"

LTX2_CONTINUATION_TARGET_FRAME_IDX = 0
LTX2_CONTINUATION_STRENGTH = 1.0
DEFAULT_LTX2_IMAGE_CRF = 33.0


@dataclass
class LTX2ImageConditioningState:
    """Result of building image / continuation conditioning.

    Mirrors the internal shape so the refine stage can read the same
    fields. The T2V port only ever instantiates this in the conditioning
    path; the refine code's ``image_conditioning is None`` branch is
    what fires for plain text-to-video SR.
    """

    clean_latent: torch.Tensor
    denoise_mask: torch.Tensor
    images: list[tuple[str, int, float]]
    latent_conditioned: bool


def resolve_ltx2_images(batch: ForwardBatch) -> list[tuple[str, int, float]]:
    """Collect any LTX-2 image conditioning inputs declared on the batch.

    The full internal version pulls from a few legacy attribute names;
    on the public batch we only support ``batch.ltx2_images`` (a list
    of ``(path, frame_index, strength)`` triples). When the field is
    missing or empty the SR path treats the request as T2V.
    """
    images = getattr(batch, "ltx2_images", None) or []
    if not isinstance(images, list):
        return []
    out: list[tuple[str, int, float]] = []
    for entry in images:
        if not isinstance(entry, list | tuple) or len(entry) < 3:
            continue
        path, frame_idx, strength = entry[0], entry[1], entry[2]
        out.append((str(path), int(frame_idx), float(strength)))
    return out


def build_ltx2_image_conditioning(
    *,
    batch: ForwardBatch,
    latents: torch.Tensor,
    vae: torch.nn.Module,
    height: int,
    width: int,
    image_crf: float | None = None,
    base_clean_latent: torch.Tensor | None = None,
) -> LTX2ImageConditioningState | None:
    """Return image-conditioning state, or ``None`` for plain T2V.

    The full internal builder handles three sources: image inputs,
    continuation latents, and ``ltx2_video_conditions``. The SR
    text-to-video path takes none of those, so the early-return
    ``None`` branch is what fires there. We retain the parameter shape
    so future i2v / continuation upstreaming can drop in without
    changing the refine stage call site.
    """
    del vae, height, width, image_crf, base_clean_latent  # unused in T2V
    images = resolve_ltx2_images(batch)
    has_latent_conditioning = (getattr(batch, "ltx2_conditioning_latent_stage1", None) is not None
                               or getattr(batch, "ltx2_conditioning_latent_stage2", None) is not None)
    has_video_conditions = bool(getattr(batch, "ltx2_video_conditions", None))
    if not images and not has_latent_conditioning and not has_video_conditions:
        return None

    raise NotImplementedError("LTX-2 image / continuation conditioning is not yet ported to the "
                              "public package. Open the corresponding upstream PR before "
                              "exercising this code path; SR text-to-video does not need it.")


def apply_ltx2_gaussian_noiser(
    *,
    noise: torch.Tensor,
    clean_latent: torch.Tensor,
    denoise_mask: torch.Tensor,
    noise_scale: float = 1.0,
) -> torch.Tensor:
    """Mix ``noise`` into ``clean_latent`` along ``denoise_mask`` * scale.

    Identical to the internal helper. The mask is expected to be
    broadcastable across channels and time; values close to 1 produce
    near-pure noise (used in a fresh stage-2 latent), values near 0
    leave the clean latent untouched (used in conditioning regions).
    """
    scaled_mask = denoise_mask * float(noise_scale)
    return (noise * scaled_mask + clean_latent * (1.0 - scaled_mask)).to(noise.dtype)


def post_process_ltx2_denoised(
    *,
    denoised: torch.Tensor,
    denoise_mask: torch.Tensor,
    clean_latent: torch.Tensor,
) -> torch.Tensor:
    """Inverse of :func:`apply_ltx2_gaussian_noiser` — restore the
    conditioning regions of ``clean_latent`` outside the denoise mask
    after the model has filled in the masked area."""
    return (denoised * denoise_mask + clean_latent.float() * (1.0 - denoise_mask)).to(denoised.dtype)


__all__ = [
    "DEFAULT_LTX2_IMAGE_CRF",
    "LTX2_CONTINUATION_STAGE1_LAST_LATENT_KEY",
    "LTX2_CONTINUATION_STAGE2_LAST_LATENT_KEY",
    "LTX2_CONTINUATION_STRENGTH",
    "LTX2_CONTINUATION_TARGET_FRAME_IDX",
    "LTX2_VIDEO_CLEAN_LATENT_KEY",
    "LTX2_VIDEO_DENOISE_MASK_KEY",
    "LTX2ImageConditioningState",
    "apply_ltx2_gaussian_noiser",
    "build_ltx2_image_conditioning",
    "post_process_ltx2_denoised",
    "resolve_ltx2_images",
]
