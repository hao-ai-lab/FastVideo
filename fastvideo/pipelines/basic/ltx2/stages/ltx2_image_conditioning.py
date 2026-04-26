# SPDX-License-Identifier: Apache-2.0
"""FastVideo-native LTX-2 image-to-video conditioning helpers.

Public-side port of ``FastVideo-internal/.../ltx2_i2v_conditioning.py``.
The module composes a ``clean_latent`` + ``denoise_mask`` pair that the
LTX-2 latent-prep + denoising stages mix into the noise tensor, so a
generated segment can be anchored to:

* one or more conditioning images at specific latent frame indices
  (``ltx2_images``),
* a multi-frame conditioning video clip jointly VAE-encoded
  (``ltx2_video_conditions``),
* a continuation latent carried over from the previous segment
  (``ltx2_conditioning_latent_stage1`` / ``_stage2``).

The streaming server's session controller populates the continuation
latents between segments; the legacy from_pretrained path passes
``ltx2_images`` / ``ltx2_image_crf`` through compat translation.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from io import BytesIO
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn.functional as F

from fastvideo.logger import init_logger
from fastvideo.models.vision_utils import load_image

if TYPE_CHECKING:
    from fastvideo.pipelines.pipeline_batch_info import ForwardBatch

try:
    import av
except ImportError:  # pragma: no cover - optional dependency
    av = None

logger = init_logger(__name__)

LTX2_VIDEO_CLEAN_LATENT_KEY = "ltx2_video_clean_latent"
LTX2_VIDEO_DENOISE_MASK_KEY = "ltx2_video_denoise_mask"
LTX2_CONTINUATION_STAGE1_LAST_LATENT_KEY = ("ltx2_continuation_stage1_last_latent")
LTX2_CONTINUATION_STAGE2_LAST_LATENT_KEY = ("ltx2_continuation_stage2_last_latent")
# NOTE: hard-coded for continuation quality experiments. The first
# latent frame of the next clip is anchored to the previous clip's
# last latent at full strength.
LTX2_CONTINUATION_TARGET_FRAME_IDX = 0
LTX2_CONTINUATION_STRENGTH = 1.0
DEFAULT_LTX2_IMAGE_CRF = 33.0


@dataclass
class LTX2ImageConditioningState:
    """Result of building image / continuation conditioning."""
    clean_latent: torch.Tensor
    denoise_mask: torch.Tensor
    images: list[tuple[str, int, float]]
    latent_conditioned: bool = False


def resolve_ltx2_images(batch: ForwardBatch) -> list[tuple[str, int, float]]:
    """Collect any LTX-2 image conditioning inputs from the batch.

    Falls back to ``batch.image_path`` for the simple single-image i2v
    case (anchors the first latent frame at full strength).
    """
    images = batch.ltx2_images
    if images is None and batch.image_path:
        images = [(batch.image_path, 0, 1.0)]
    if not images:
        return []

    resolved: list[tuple[str, int, float]] = []
    for item in images:
        if not isinstance(item, tuple | list) or len(item) != 3:
            raise ValueError("Each ltx2_images item must be a tuple/list of "
                             "(path, frame_idx, strength).")
        image_path, frame_idx, strength = item
        frame_idx_int = int(frame_idx)
        strength_float = float(strength)
        if frame_idx_int < 0:
            raise ValueError(f"LTX-2 frame_idx must be >= 0, got {frame_idx_int}")
        if strength_float < 0.0 or strength_float > 1.0:
            raise ValueError(f"LTX-2 image conditioning strength must be in [0, 1], "
                             f"got {strength_float}")
        resolved.append((str(image_path), frame_idx_int, strength_float))
    return resolved


def _resize_and_center_crop(
    tensor: torch.Tensor,
    height: int,
    width: int,
) -> torch.Tensor:
    if tensor.ndim != 3:
        raise ValueError(f"Expected image tensor [H, W, C], got shape {tuple(tensor.shape)}")

    tensor = tensor.permute(2, 0, 1).unsqueeze(0)
    _, _, src_h, src_w = tensor.shape
    scale = max(height / src_h, width / src_w)
    new_h = math.ceil(src_h * scale)
    new_w = math.ceil(src_w * scale)
    tensor = F.interpolate(
        tensor,
        size=(new_h, new_w),
        mode="bilinear",
        align_corners=False,
    )

    crop_top = (new_h - height) // 2
    crop_left = (new_w - width) // 2
    tensor = tensor[:, :, crop_top:crop_top + height, crop_left:crop_left + width]
    return tensor.unsqueeze(2)


def _encode_single_frame(output_file: BytesIO, image_array: np.ndarray, crf: float) -> None:
    container = av.open(output_file, "w", format="mp4")
    try:
        stream = container.add_stream(
            "libx264",
            rate=1,
            options={
                "crf": str(crf),
                "preset": "veryfast",
            },
        )
        height = image_array.shape[0] // 2 * 2
        width = image_array.shape[1] // 2 * 2
        image_array = image_array[:height, :width]
        stream.height = height
        stream.width = width
        frame = av.VideoFrame.from_ndarray(image_array, format="rgb24").reformat(format="yuv420p")
        container.mux(stream.encode(frame))
        container.mux(stream.encode())
    finally:
        container.close()


def _decode_single_frame(video_file: BytesIO) -> np.ndarray:
    container = av.open(video_file)
    try:
        stream = next(s for s in container.streams if s.type == "video")
        frame = next(container.decode(stream))
    finally:
        container.close()
    return frame.to_ndarray(format="rgb24")


def _preprocess_conditioning_image(
    image: np.ndarray,
    image_crf: float,
) -> np.ndarray:
    """H.264 CRF re-encode the conditioning image to match the
    quantization the model was trained on. ``image_crf <= 0.0`` skips
    the re-encode (used by the streaming server which conditions on
    already-decoded VAE-quality frames)."""
    if image_crf <= 0.0:
        return image
    if av is None:
        logger.warning("[LTX2] PyAV is unavailable; skipping CRF "
                       "conditioning preprocessing.")
        return image

    with BytesIO() as output_file:
        _encode_single_frame(output_file, image, image_crf)
        encoded = output_file.getvalue()
    with BytesIO(encoded) as video_file:
        return _decode_single_frame(video_file)


def load_ltx2_conditioning_image(
    image_path: str,
    *,
    height: int,
    width: int,
    dtype: torch.dtype,
    device: torch.device,
    image_crf: float,
) -> torch.Tensor:
    image = load_image(image_path)
    image_np = np.array(image)[..., :3]
    image_np = _preprocess_conditioning_image(image_np, image_crf=image_crf)
    image_tensor = torch.tensor(image_np, dtype=torch.float32, device=device)
    image_tensor = _resize_and_center_crop(image_tensor, height, width)
    image_tensor = (image_tensor / 127.5 - 1.0).to(device=device, dtype=dtype)
    return image_tensor


def load_ltx2_conditioning_video_clip(
    frame_paths: list[str],
    *,
    height: int,
    width: int,
    dtype: torch.dtype,
    device: torch.device,
    image_crf: float,
) -> torch.Tensor:
    """Load multiple frames and stack as ``[1, C, T, H, W]`` for joint
    VAE encoding so the resulting latent captures temporal/motion info.
    """
    frame_tensors: list[torch.Tensor] = []
    for path in frame_paths:
        image = load_image(path)
        image_np = np.array(image)[..., :3]
        image_np = _preprocess_conditioning_image(image_np, image_crf=image_crf)
        t = torch.tensor(image_np, dtype=torch.float32, device=device)
        # _resize_and_center_crop returns [1, C, 1, H, W]
        t = _resize_and_center_crop(t, height, width)
        frame_tensors.append(t)
    # Concat along T dimension -> [1, C, T, H, W]
    video = torch.cat(frame_tensors, dim=2)
    return (video / 127.5 - 1.0).to(device=device, dtype=dtype)


def _extract_video_latent(vae: torch.nn.Module, image: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        encoded = vae.encode(image)

    latent_dist = getattr(encoded, "latent_dist", None)
    if latent_dist is not None:
        encoded = latent_dist

    latent: torch.Tensor
    if torch.is_tensor(encoded):
        latent = encoded
    elif hasattr(encoded, "mode"):
        latent = encoded.mode()
    elif hasattr(encoded, "sample"):
        latent = encoded.sample()
    elif (isinstance(encoded, tuple | list) and encoded and torch.is_tensor(encoded[0])):
        latent = encoded[0]
    else:
        raise TypeError(f"Unsupported VAE encode output type: {type(encoded)}")

    if latent.ndim == 4:
        latent = latent.unsqueeze(2)
    if latent.ndim != 5:
        raise ValueError(f"Expected video latent with 5 dims [B,C,T,H,W], got "
                         f"{tuple(latent.shape)}")
    return latent


def _insert_conditioning_latent(
    *,
    conditioning_latent: torch.Tensor,
    clean_latent: torch.Tensor,
    denoise_mask: torch.Tensor,
    frame_idx: int,
    strength: float,
    source_name: str,
) -> None:
    if conditioning_latent.ndim == 4:
        conditioning_latent = conditioning_latent.unsqueeze(2)
    if conditioning_latent.ndim != 5:
        raise ValueError(f"LTX-2 {source_name} latent must have 5 dims [B,C,T,H,W], "
                         f"got {tuple(conditioning_latent.shape)}")

    if conditioning_latent.shape[0] == 1 and clean_latent.shape[0] > 1:
        conditioned_latent = conditioning_latent.expand(
            clean_latent.shape[0],
            -1,
            -1,
            -1,
            -1,
        )
    elif conditioning_latent.shape[0] == clean_latent.shape[0]:
        conditioned_latent = conditioning_latent
    else:
        raise ValueError(f"LTX-2 {source_name} latent batch mismatch: "
                         f"{conditioning_latent.shape[0]} vs {clean_latent.shape[0]}")

    if conditioned_latent.shape[1] != clean_latent.shape[1]:
        raise ValueError(f"LTX-2 {source_name} latent channels mismatch: "
                         f"{conditioned_latent.shape[1]} vs {clean_latent.shape[1]}")
    if conditioned_latent.shape[-2:] != clean_latent.shape[-2:]:
        raise ValueError(f"LTX-2 {source_name} latent spatial shape "
                         f"mismatch: {tuple(conditioned_latent.shape[-2:])} "
                         f"vs {tuple(clean_latent.shape[-2:])}")

    frame_count = conditioned_latent.shape[2]
    end_idx = frame_idx + frame_count
    if frame_idx < 0 or end_idx > clean_latent.shape[2]:
        raise ValueError(f"LTX-2 {source_name} latent frame range out of bounds: "
                         f"frame_idx={frame_idx}, frame_count={frame_count}, "
                         f"latent_frames={clean_latent.shape[2]}")

    clean_latent[:, :, frame_idx:end_idx] = conditioned_latent.to(
        device=clean_latent.device,
        dtype=clean_latent.dtype,
    )
    denoise_mask[:, :, frame_idx:end_idx] = 1.0 - float(strength)


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
    """Build the (clean_latent, denoise_mask) state for the next segment.

    Returns ``None`` for plain T2V (no images, no continuation, no
    video conditions). The denoise mask is 1 where the model should
    sample fresh, 0 where it should preserve the conditioning latent
    exactly. ``base_clean_latent is None`` corresponds to stage 1
    (fresh half-res latent); ``base_clean_latent`` set means stage 2
    (already-upsampled latent from the upsampler stage).
    """
    images = resolve_ltx2_images(batch)
    conditioning_latent_stage1 = getattr(batch, "ltx2_conditioning_latent_stage1", None)
    conditioning_latent_stage2 = getattr(batch, "ltx2_conditioning_latent_stage2", None)
    is_stage1_conditioning = base_clean_latent is None
    is_stage2_conditioning = not is_stage1_conditioning
    has_latent_conditioning = False
    continuation_latent_to_insert: torch.Tensor | None = None
    if (conditioning_latent_stage1 is not None and not torch.is_tensor(conditioning_latent_stage1)):
        raise TypeError("LTX-2 stage1 continuation latent conditioning "
                        "expects a torch.Tensor.")
    if (conditioning_latent_stage2 is not None and not torch.is_tensor(conditioning_latent_stage2)):
        raise TypeError("LTX-2 stage2 continuation latent conditioning "
                        "expects a torch.Tensor.")

    if (conditioning_latent_stage1 is None) != (conditioning_latent_stage2 is None):
        raise ValueError("LTX-2 continuation expects both stage1 and stage2 "
                         "conditioning latents (or neither for first round).")
    if is_stage1_conditioning and conditioning_latent_stage1 is not None:
        has_latent_conditioning = True
        continuation_latent_to_insert = conditioning_latent_stage1.to(
            device=latents.device,
            dtype=latents.dtype,
        )
    elif is_stage2_conditioning and conditioning_latent_stage2 is not None:
        has_latent_conditioning = True
        continuation_latent_to_insert = conditioning_latent_stage2.to(
            device=latents.device,
            dtype=latents.dtype,
        )

    video_conditions = getattr(batch, "ltx2_video_conditions", None) or []

    if not images and not has_latent_conditioning and not video_conditions:
        return None

    clean_latent = (torch.zeros_like(latents) if base_clean_latent is None else base_clean_latent.clone())

    denoise_mask = torch.ones(
        (
            latents.shape[0],
            1,
            latents.shape[2],
            latents.shape[3],
            latents.shape[4],
        ),
        dtype=torch.float32,
        device=latents.device,
    )

    if image_crf is None:
        image_crf = getattr(batch, "ltx2_image_crf", DEFAULT_LTX2_IMAGE_CRF)

    vae_param = next(vae.parameters(), None)
    encoder_dtype = (vae_param.dtype if vae_param is not None else latents.dtype)
    encoder_device = (vae_param.device if vae_param is not None else latents.device)
    cache: dict[tuple[str, int, int, float], torch.Tensor] = {}
    latent_conditioned = False

    if has_latent_conditioning:
        if continuation_latent_to_insert is None:
            raise RuntimeError("LTX-2 continuation latent conditioning state is invalid.")
        # NOTE: frame index and strength are intentionally hard-coded.
        # We always anchor the first frame of the next clip at full
        # strength to the previous clip's last latent.
        _insert_conditioning_latent(
            conditioning_latent=continuation_latent_to_insert,
            clean_latent=clean_latent,
            denoise_mask=denoise_mask,
            frame_idx=LTX2_CONTINUATION_TARGET_FRAME_IDX,
            strength=LTX2_CONTINUATION_STRENGTH,
            source_name="continuation",
        )
        latent_conditioned = True

    for image_path, frame_idx, strength in images:
        cache_key = (image_path, height, width, float(image_crf))
        image_latent = cache.get(cache_key)
        if image_latent is None:
            image_tensor = load_ltx2_conditioning_image(
                image_path=image_path,
                height=height,
                width=width,
                dtype=encoder_dtype,
                device=encoder_device,
                image_crf=float(image_crf),
            )
            image_latent = _extract_video_latent(vae, image_tensor).to(
                device=latents.device,
                dtype=latents.dtype,
            )
            cache[cache_key] = image_latent

        _insert_conditioning_latent(
            conditioning_latent=image_latent,
            clean_latent=clean_latent,
            denoise_mask=denoise_mask,
            frame_idx=frame_idx,
            strength=strength,
            source_name="image",
        )

    for frame_paths, frame_idx, strength in video_conditions:
        video_tensor = load_ltx2_conditioning_video_clip(
            frame_paths,
            height=height,
            width=width,
            dtype=encoder_dtype,
            device=encoder_device,
            image_crf=float(image_crf),
        )
        video_latent = _extract_video_latent(vae, video_tensor).to(
            device=latents.device,
            dtype=latents.dtype,
        )
        logger.info(
            "[LTX2] Video-clip condition: %d frames -> "
            "latent T=%d at frame_idx=%d strength=%.2f",
            len(frame_paths),
            video_latent.shape[2],
            frame_idx,
            strength,
        )
        _insert_conditioning_latent(
            conditioning_latent=video_latent,
            clean_latent=clean_latent,
            denoise_mask=denoise_mask,
            frame_idx=frame_idx,
            strength=strength,
            source_name="video_clip",
        )

    return LTX2ImageConditioningState(
        clean_latent=clean_latent,
        denoise_mask=denoise_mask,
        images=images,
        latent_conditioned=latent_conditioned,
    )


def apply_ltx2_gaussian_noiser(
    *,
    noise: torch.Tensor,
    clean_latent: torch.Tensor,
    denoise_mask: torch.Tensor,
    noise_scale: float = 1.0,
) -> torch.Tensor:
    """Mix ``noise`` into ``clean_latent`` along ``denoise_mask`` * scale.

    Values close to 1 in the mask produce near-pure noise (used in a
    fresh stage-2 latent), values near 0 leave the clean latent
    untouched (used in conditioning regions).
    """
    scaled_mask = denoise_mask * float(noise_scale)
    return (noise * scaled_mask + clean_latent * (1.0 - scaled_mask)).to(noise.dtype)


def post_process_ltx2_denoised(
    *,
    denoised: torch.Tensor,
    denoise_mask: torch.Tensor,
    clean_latent: torch.Tensor,
) -> torch.Tensor:
    """Restore the conditioning regions of ``clean_latent`` outside the
    denoise mask after the model has filled in the masked area."""
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
    "load_ltx2_conditioning_image",
    "load_ltx2_conditioning_video_clip",
    "post_process_ltx2_denoised",
    "resolve_ltx2_images",
]
