# SPDX-License-Identifier: Apache-2.0
"""Ref2VA conditioning for the MLX H3 runtime: references into model inputs.

Three things have to happen before a reference-conditioned request can denoise,
and they are grouped here because they share the same prepared-reference list
and the same failure modes:

1. the ordered references are presented to Qwen3-VL (labels, vision blocks,
   then the prompt) and encoded to conditioning hidden states;
2. each visual reference is encoded by the video VAE and noised to the
   conditioning timestep, and each soundtrack by the audio VAE;
3. the conditioning rows are packed ahead of the target rows.

The CUDA pipeline spreads the same work across its conditioning and latent
preparation stages; this module is the MLX equivalent, and mirrors the numbers
they produce (0.999 augmentation on video condition rows, clean audio condition
rows) because the distilled student was trained against them.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from fastvideo.logger import init_logger
from fastvideo.mlx_runtime.minimax_h3 import (
    MINIMAX_H3_KEYFRAME_NOISE_AUG,
    MiniMaxH3SchedulerState,
    patchify_video_latents,
)
from fastvideo.mlx_runtime.minimax_h3_reference import (
    MiniMaxH3PreparedReference,
    build_ref2va_presentation,
    sample_reference_video_frames,
    trim_reference_num_frames,
)

logger = init_logger(__name__)

# Salt mixed into the request seed for the reference draws, so they never share
# a stream with the target latents (which key off the bare request seed). The
# CUDA path makes the same draws with a torch generator; deriving them from the
# request seed keeps them reproducible per seed, not bit-identical to torch.
MINIMAX_H3_REFERENCE_ENCODE_SEED = 42
_POSTERIOR_STREAM = 0
_CONDITION_NOISE_STREAM = 1
_SEED_MODULUS = 2**32

# Qwen3-VL processor defaults, used when the checkpoint ships no
# preprocessor_config.json. The released H3 snapshot does ship one, so these
# only apply to a stripped-down model directory.
DEFAULT_VISION_MIN_PIXELS = 65536
DEFAULT_VISION_MAX_PIXELS = 16777216
# The released H3 video processor config, used when only the image config ships.
DEFAULT_VIDEO_MIN_PIXELS = 4096
DEFAULT_VIDEO_MAX_PIXELS = 25165824


def reference_draw_key(seed: int, stream: int, index: int):
    """MLX key for one reference draw: a function of request seed, stream and reference index."""
    import mlx.core as mx

    mixed = (int(seed) * 1_000_003 + MINIMAX_H3_REFERENCE_ENCODE_SEED + stream * 1_009 + index) % _SEED_MODULUS
    return mx.random.key(mixed)



def load_vision_limits(processor_dir: str | Path, *, video: bool = False) -> tuple[int, int]:
    """``(min_pixels, max_pixels)`` the Qwen3-VL image or video processor would use.

    Images read ``preprocessor_config.json``; videos read the separate
    ``video_preprocessor_config.json`` the HF video processor is built from, so a
    reference is resized onto the same grid the CUDA path produces. Both accept
    the HF ``size`` block (``shortest_edge``/``longest_edge``) or explicit
    ``min_pixels``/``max_pixels`` keys.
    """
    name = "video_preprocessor_config.json" if video else "preprocessor_config.json"
    default_min, default_max = ((DEFAULT_VIDEO_MIN_PIXELS, DEFAULT_VIDEO_MAX_PIXELS) if video else
                                (DEFAULT_VISION_MIN_PIXELS, DEFAULT_VISION_MAX_PIXELS))
    config_path = Path(processor_dir) / name
    if not config_path.is_file():
        return default_min, default_max
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    size = raw.get("size") or {}
    return (
        int(raw.get("min_pixels", size.get("shortest_edge", default_min))),
        int(raw.get("max_pixels", size.get("longest_edge", default_max))),
    )



def encode_reference_prompt(
    conditioner: Any,
    prompt: str,
    references: list[MiniMaxH3PreparedReference],
    *,
    vision_config: Any,
    min_pixels: int,
    max_pixels: int,
    video_min_pixels: int | None = None,
    video_max_pixels: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Present and encode the ordered references plus the prompt.

    Video references are sampled to the Qwen 2-fps presentation first, which is
    also what assigns their block timestamps, so it must run before the
    presentation is built.
    """
    from fastvideo.mlx_runtime.minimax_h3_vision import prepare_vision_inputs

    images = [np.asarray(reference.image) for reference in references if reference.media_type == "image"]
    if any(image is None for image in images):
        raise ValueError("A reference image must be prepared before conditioning.")
    videos = []
    for reference in references:
        if reference.media_type != "video":
            continue
        if reference.frames is None:
            raise ValueError("A reference video must be prepared before conditioning.")
        sampled, timestamps = sample_reference_video_frames(reference.frames)
        reference.block_timestamps = timestamps
        videos.append(np.stack(sampled))

    pixel_values, image_grid_thw, pixel_values_videos, video_grid_thw = prepare_vision_inputs(
        images,
        videos,
        config=vision_config,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
        video_min_pixels=video_min_pixels,
        video_max_pixels=video_max_pixels,
    )

    merge_area = vision_config.spatial_merge_size**2
    image_token_counts = [] if image_grid_thw is None else [
        int(np.prod(grid_thw)) // merge_area for grid_thw in image_grid_thw
    ]
    video_block_token_counts = [] if video_grid_thw is None else [
        int(grid_thw[1]) * int(grid_thw[2]) // merge_area for grid_thw in video_grid_thw
    ]
    if video_grid_thw is not None:
        for reference, grid_thw in zip([item for item in references if item.media_type == "video"],
                                       video_grid_thw,
                                       strict=True):
            if int(grid_thw[0]) != len(reference.block_timestamps):
                raise ValueError(f"Qwen3-VL produced {int(grid_thw[0])} blocks for a reference video, but "
                                 f"MiniMax-H3 labels {len(reference.block_timestamps)}.")

    token_ids, token_tags = build_ref2va_presentation(
        conditioner.tokenizer,
        prompt,
        references,
        image_token_counts,
        video_block_token_counts,
    )
    hidden, tags = conditioner.encode_tokens_with_vision(
        token_ids,
        np.asarray(token_tags, dtype=np.int64),
        pixel_values=pixel_values,
        image_grid_thw=image_grid_thw,
        pixel_values_videos=pixel_values_videos,
        video_grid_thw=video_grid_thw,
    )
    if hidden.shape[0] != len(token_ids):
        raise ValueError(f"The H3 conditioner returned {hidden.shape[0]} rows for {len(token_ids)} tokens.")
    return hidden, tags



def reference_condition_noise(
    shapes: tuple[tuple[int, int, int], ...],
    patch_size: tuple[int, int, int],
    latent_channels: int,
    *,
    seed: int,
) -> np.ndarray:
    """Packed noise for the noised video condition rows (one draw per reference)."""
    import mlx.core as mx

    rows = []
    for index, (num_latent_frames, latent_height, latent_width) in enumerate(shapes):
        noise = mx.random.normal(
            (1, latent_channels, num_latent_frames, latent_height, latent_width),
            key=reference_draw_key(seed, _CONDITION_NOISE_STREAM, index),
        )
        rows.append(patchify_video_latents(np.asarray(noise, dtype=np.float32), patch_size))
    return np.concatenate(rows)


def _visual_pixels(pixels: np.ndarray, pixels_are_uint8: bool) -> Any:
    """(H, W, 3) or (T, H, W, 3) uint8/float -> (1, 3, T, H, W) float in [0, 1]."""
    import mlx.core as mx

    array = pixels.astype(np.float32)
    if pixels_are_uint8:
        array = array / 255.0
    if array.ndim == 3:
        array = array[None]
    return mx.array(np.ascontiguousarray(array.transpose(3, 0, 1, 2))[None])


def _sample_latents(vae: Any, mean: Any, logvar: Any, *, seed: int, index: int) -> np.ndarray:
    """One seeded posterior draw with the reference fp16 round-trip."""
    import mlx.core as mx

    noise = mx.random.normal(mean.shape, key=reference_draw_key(seed, _POSTERIOR_STREAM, index))
    sampled = vae.sample_posterior(mean, logvar, noise)
    return np.asarray(vae.normalize_latents(sampled.astype(mx.float16).astype(mx.float32)), dtype=np.float32)


def encode_reference_conditions(
    *,
    video_vae: Any,
    audio_vae: Any | None,
    references: list[MiniMaxH3PreparedReference],
    patch_size: tuple[int, int, int],
    scheduler: MiniMaxH3SchedulerState,
    seed: int,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Encode every reference into packed condition rows.

    Video condition rows are augmented to ``MINIMAX_H3_KEYFRAME_NOISE_AUG`` so
    the student sees the reference the way it was trained; audio condition rows
    are handed over clean. Each reference's latent geometry is written back onto
    it, because the packed layout is built from it afterwards. ``seed`` is the
    request seed: the posterior draws and the condition noise derive from it.
    """
    import mlx.core as mx

    visual_rows: list[np.ndarray] = []
    for index, reference in enumerate(references):
        if reference.media_type == "audio":
            continue
        if reference.media_type == "image":
            if reference.image is None:
                raise ValueError("A reference image must be prepared before latent encoding.")
            pixels = _visual_pixels(np.asarray(reference.image), pixels_are_uint8=True)
            mean, logvar = video_vae.encode_keyframe(video_vae.normalize_pixels(pixels))
        else:
            if reference.frames is None:
                raise ValueError("A reference video must be prepared before latent encoding.")
            frames = reference.frames[:trim_reference_num_frames(reference.frames.shape[0])]
            pixels = _visual_pixels(frames, pixels_are_uint8=frames.dtype == np.uint8)
            mean, logvar = video_vae.encode(video_vae.normalize_pixels(pixels))
        latents = _sample_latents(video_vae, mean, logvar, seed=seed, index=index)
        reference.num_latent_frames = int(latents.shape[2])
        reference.latent_height = int(latents.shape[3])
        reference.latent_width = int(latents.shape[4])
        visual_rows.append(patchify_video_latents(latents, patch_size))
        mx.eval()

    if not visual_rows:
        raise ValueError("MiniMax-H3 Ref2VA requires at least one visual reference.")

    audio_rows: list[np.ndarray] = []
    for reference in references:
        if not reference.has_audio:
            continue
        if audio_vae is None:
            raise ValueError("A reference with a soundtrack requires the audio VAE encoder.")
        if reference.waveform is None:
            raise ValueError("A reference soundtrack must be prepared before latent encoding.")
        waveform = np.asarray(reference.waveform, dtype=np.float32)[:, None]
        mean, _ = audio_vae.encode(mx.array(waveform))
        # The audio posterior is taken at its mode, as on the CUDA path.
        latents = np.asarray(audio_vae.normalize_latents(mean), dtype=np.float32).transpose(0, 2, 1)
        reference.num_audio_latents = int(latents.shape[1])
        audio_rows.append(latents.reshape(-1, audio_vae.latent_channels))

    shapes = tuple((reference.num_latent_frames, reference.latent_height, reference.latent_width)
                   for reference in references if reference.media_type != "audio")
    noise = reference_condition_noise(shapes, patch_size, video_vae.latent_channels, seed=seed)
    clean = np.concatenate(visual_rows)
    if clean.shape != noise.shape:
        raise ValueError(f"Reference condition rows {clean.shape} do not match their noise {noise.shape}.")
    video_conditions = np.asarray(scheduler.scale_noise(mx.array(clean), MINIMAX_H3_KEYFRAME_NOISE_AUG,
                                                        mx.array(noise)),
                                  dtype=np.float32)
    audio_conditions = np.concatenate(audio_rows) if audio_rows else None
    if audio_conditions is not None and audio_conditions.shape[1] != audio_vae.latent_channels:
        raise ValueError("Audio condition rows disagree with the audio VAE latent width.")

    logger.info("Encoded %d reference(s): %d video condition rows, %s audio condition rows", len(references),
                video_conditions.shape[0], "no" if audio_conditions is None else int(audio_conditions.shape[0]))
    return video_conditions, audio_conditions


__all__ = [
    "MINIMAX_H3_REFERENCE_ENCODE_SEED",
    "encode_reference_conditions",
    "encode_reference_prompt",
    "load_vision_limits",
    "reference_condition_noise",
    "reference_draw_key",
]
