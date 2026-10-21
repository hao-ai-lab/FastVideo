# SPDX-License-Identifier: Apache-2.0
# mypy: disable-error-code=no-untyped-call
"""Qwen3-VL vision tower + deepstack + MRoPE index builder for MiniMax-H3 on MLX.

This is the Apple-Silicon counterpart of the visual half of
``fastvideo/models/encoders/minimax_h3_qwen3_vl.py``: the reference-conditioning
(Ref2VA) path needs to turn images and sampled reference-video frames into the
merged visual tokens the text conditioner splices into its embedding rows, plus
the DeEPSTACK feature blocks it adds back after the first few language layers.

Everything here is a faithful port of that torch module, in the same layout the
HF checkpoint ships (``model.visual.*`` keys, ``MiniMaxH3Qwen3VLVisionModel``
submodule names). Weights stream from the safetensors shards with the same
per-key reader the text conditioner uses, so the two paths share one on-disk
header cache and one BF16 handling path.

Boundary contract: numpy fp32 in, numpy fp32 out; MLX is only an internal
implementation detail. Production code here never imports PyTorch (torch appears
only in the parity tests).

Deliberate simplification vs. the HF ``image_processor``/``video_processor``:
resize is a numpy bicubic that reproduces ``torchvision``'s antialiased
``interpolate(..., mode="bicubic", antialias=True, align_corners=False)``
(including the width-first uint8 intermediate) to within one grey level,
rather than shelling out to torchvision; the geometry (smart-resize), the
mean/std, and the patch layout agree with HF exactly. Callers already own frame
sampling, so ``do_sample_frames`` is folded out of the video path.
"""

from __future__ import annotations

import json
import math
import struct
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import mlx.core as mx

from fastvideo.logger import init_logger

logger = init_logger(__name__)

# Qwen3-VL image normalisation (the released processor's image_mean/image_std).
VISION_IMAGE_MEAN = 0.5
VISION_IMAGE_STD = 0.5

# SMARTRESIZE guards matching the HF processors.
_MAX_ASPECT_RATIO = 200.0

# The four-tap bilinear gather in ``_interpolate_position_embeddings`` can
# otherwise materialise a multi-GiB temporary for packed reference videos.
_INTERPOLATION_TAP = 4


@dataclass(frozen=True)
class Qwen3VLVisionConfig:
    """Architecture constants mirroring ``vision_config`` in a Qwen3-VL json."""

    hidden_size: int = 1152
    depth: int = 27
    num_heads: int = 16
    intermediate_size: int = 4304
    in_channels: int = 3
    patch_size: int = 16
    temporal_patch_size: int = 2
    spatial_merge_size: int = 2
    out_hidden_size: int = 5120
    num_position_embeddings: int = 2304
    deepstack_visual_indexes: tuple[int, ...] = (8, 16, 24)
    hidden_act: str = "gelu_pytorch_tanh"

    @classmethod
    def from_config_json(cls, path: str | Path) -> Qwen3VLVisionConfig:
        raw = json.loads(Path(path).read_text())
        vision = raw.get("vision_config", raw)
        return cls(
            hidden_size=int(vision["hidden_size"]),
            depth=int(vision["depth"]),
            num_heads=int(vision["num_heads"]),
            intermediate_size=int(vision["intermediate_size"]),
            in_channels=int(vision.get("in_channels", 3)),
            patch_size=int(vision["patch_size"]),
            temporal_patch_size=int(vision.get("temporal_patch_size", 2)),
            spatial_merge_size=int(vision.get("spatial_merge_size", 2)),
            out_hidden_size=int(vision["out_hidden_size"]),
            num_position_embeddings=int(vision["num_position_embeddings"]),
            deepstack_visual_indexes=tuple(int(v) for v in vision.get("deepstack_visual_indexes", (8, 16, 24))),
            hidden_act=str(vision.get("hidden_act", "gelu_pytorch_tanh")),
        )


def _smart_resize_image(height: int, width: int, factor: int, min_pixels: int, max_pixels: int) -> tuple[int, int]:
    """HF image-processor smart resize: round to ``factor`` then clamp the area."""
    if height < factor or width < factor:
        raise ValueError(f"Image height:{height} or width:{width} is smaller than factor:{factor}")
    if max(height, width) / min(height, width) > _MAX_ASPECT_RATIO:
        raise ValueError(f"Absolute aspect ratio must be smaller than {_MAX_ASPECT_RATIO}, "
                         f"got {max(height, width) / min(height, width)}")
    height_bar = max(factor, round(height / factor) * factor)
    width_bar = max(factor, round(width / factor) * factor)
    if height_bar * width_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        height_bar = max(factor, math.floor(height / beta / factor) * factor)
        width_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif height_bar * width_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        height_bar = math.ceil(height * beta / factor) * factor
        width_bar = math.ceil(width * beta / factor) * factor
    return height_bar, width_bar


def _smart_resize_video(
    num_frames: int,
    height: int,
    width: int,
    temporal_factor: int,
    factor: int,
    min_pixels: int,
    max_pixels: int,
) -> tuple[int, int]:
    """HF video-processor smart resize, counting the padded temporal extent."""
    if height < factor or width < factor:
        raise ValueError(f"Video height:{height} or width:{width} is smaller than factor:{factor}")
    if max(height, width) / min(height, width) > _MAX_ASPECT_RATIO:
        raise ValueError(f"Absolute aspect ratio must be smaller than {_MAX_ASPECT_RATIO}, "
                         f"got {max(height, width) / min(height, width)}")
    height_bar = round(height / factor) * factor
    width_bar = round(width / factor) * factor
    frames_bar = math.ceil(num_frames / temporal_factor) * temporal_factor
    if frames_bar * height_bar * width_bar > max_pixels:
        beta = math.sqrt((num_frames * height * width) / max_pixels)
        height_bar = max(factor, math.floor(height / beta / factor) * factor)
        width_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif frames_bar * height_bar * width_bar < min_pixels:
        beta = math.sqrt(min_pixels / (num_frames * height * width))
        height_bar = math.ceil(height * beta / factor) * factor
        width_bar = math.ceil(width * beta / factor) * factor
    return height_bar, width_bar


def _cubic_kernel(distances: np.ndarray) -> np.ndarray:
    """Torchvision's antialiased bicubic kernel (a = -0.5, support 2)."""
    a = -0.5
    absolute = np.abs(distances)
    weights = np.zeros_like(absolute)
    inner = absolute <= 1.0
    weights[inner] = (a + 2) * absolute[inner]**3 - (a + 3) * absolute[inner]**2 + 1.0
    outer = (absolute > 1.0) & (absolute < 2.0)
    weights[outer] = (a * absolute[outer]**3 - 5 * a * absolute[outer]**2 + 8 * a * absolute[outer] - 4 * a)
    return weights / 6.0


def _resize_axis_weights(in_size: int, out_size: int) -> np.ndarray:
    """Per-output weights along one axis, matching torchvision's bicubic-AA op.

    ``F.interpolate(..., mode="bicubic", antialias=True, align_corners=False)``
    widens the kernel by the downscale ratio and normalises it over the whole
    finite axis (weights past the kernel support are exactly zero); reproducing
    that here keeps pixel values aligned without importing torchvision.
    """
    if in_size == out_size:
        return np.eye(in_size, dtype=np.float64)
    scale = in_size / out_size
    denominator = scale if scale >= 1.0 else 1.0
    sources = np.arange(in_size, dtype=np.float64)
    output_centers = scale * (np.arange(out_size, dtype=np.float64) + 0.5) - 0.5
    weights = np.empty((out_size, in_size), dtype=np.float64)
    for row, center in enumerate(output_centers):
        weight = _cubic_kernel((sources - center) / denominator)
        weights[row] = weight / weight.sum()
    return weights


def _bicubic_resize_uint8(image: np.ndarray, height: int, width: int) -> np.ndarray:
    """Match ``tvF.resize(interpolation=BICUBIC, antialias=True)`` on a uint8 HWC image.

    Two notes on fidelity: torchvision runs the innermost (width) axis first and
    materialises the intermediate as uint8 — clamped and rounded — before the
    height pass, and that clipping is observable (the uint8 resize is not one
    linear map). Residual disagreement is at most one grey level, from float
    accumulation order; callers see ``<= 1/127.5`` in normalised units.
    """
    in_height, in_width = image.shape[:2]
    if (in_height, in_width) == (height, width):
        return np.ascontiguousarray(image)
    row_weight = _resize_axis_weights(in_height, height)
    col_weight = _resize_axis_weights(in_width, width)
    source = image.astype(np.float32)
    horizontal = np.einsum("ijc,wj->iwc", source, col_weight)
    horizontal = np.clip(np.round(horizontal), 0, 255)
    vertical = np.einsum("hi,iwc->hwc", row_weight, horizontal)
    return np.clip(np.round(vertical), 0, 255).astype(np.uint8)


def _patchify(frames: np.ndarray, grid_t: int, grid_h: int, grid_w: int, *, patch_size: int, temporal_patch_size: int,
              merge_size: int, in_channels: int) -> np.ndarray:
    """Flatten normalised frames into the temporal-patch-major conv3d layout.

    Mirrors the HF processor's ``view``/``permute`` so the flattened patch vector
    is ``channel, temporal, patch, patch`` major — exactly what the Conv3d patch
    embed's ``view(-1, in_channels, temporal_patch_size, patch_size, patch_size)``
    reads back.
    """
    # Frames arrive HWC (the sampler's native layout); the conv3d patch vector is
    # channel-major, so move channels ahead of the spatial axes like HF does.
    frames = frames.transpose(0, 3, 1, 2)
    merged_h, merged_w = grid_h // merge_size, grid_w // merge_size
    layout = frames.reshape(
        grid_t,
        temporal_patch_size,
        in_channels,
        merged_h,
        merge_size,
        patch_size,
        merged_w,
        merge_size,
        patch_size,
    )
    layout = layout.transpose(0, 3, 6, 4, 7, 2, 1, 5, 8)
    return np.ascontiguousarray(layout).reshape(grid_t * grid_h * grid_w,
                                                in_channels * temporal_patch_size * patch_size * patch_size)


def _prepare_block(frames: np.ndarray, *, num_frames: int, source_height: int, source_width: int, grid: tuple[int, int],
                   config: Qwen3VLVisionConfig) -> np.ndarray:
    """Resize, normalise and patchify one image (T=1) or sampled video block."""
    height, width = grid
    resized = np.stack([_bicubic_resize_uint8(frames[index], height, width) for index in range(num_frames)])
    resized = resized.astype(np.float32) / 255.0
    resized = (resized - VISION_IMAGE_MEAN) / VISION_IMAGE_STD
    pad = -num_frames % config.temporal_patch_size
    if pad:
        resized = np.concatenate([resized, np.repeat(resized[-1:], pad, axis=0)], axis=0)
    grid_t = resized.shape[0] // config.temporal_patch_size
    return _patchify(resized,
                     grid_t,
                     height // config.patch_size,
                     width // config.patch_size,
                     patch_size=config.patch_size,
                     temporal_patch_size=config.temporal_patch_size,
                     merge_size=config.spatial_merge_size,
                     in_channels=config.in_channels)


def prepare_vision_inputs(
    images: list[np.ndarray] | None,
    videos: list[np.ndarray] | None,
    *,
    config: Qwen3VLVisionConfig,
    min_pixels: int,
    max_pixels: int,
    video_min_pixels: int | None = None,
    video_max_pixels: int | None = None,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    """HF image/video-processor equivalent for Ref2VA references.

    ``images`` is a list of ``(H, W, 3)`` uint8 arrays; ``videos`` is a list of
    ``(frames, H, W, 3)`` uint8 frame blocks already sampled at the presentation
    fps. Returns ``(pixel_values, image_grid_thw, pixel_values_videos,
    video_grid_thw)`` with the missing modality's pair set to ``None`` — slots a
    absent modality's pair stays ``None``. Videos resize against their own pixel
    limits (``video_*_pixels``, from the HF video processor config), defaulting to
    the image limits.
    """
    video_min_pixels = min_pixels if video_min_pixels is None else video_min_pixels
    video_max_pixels = max_pixels if video_max_pixels is None else video_max_pixels
    factor = config.patch_size * config.spatial_merge_size
    if factor <= 0:
        raise ValueError(f"Invalid patch/merge factor {factor}")
    pixel_values = image_grid_thw = None
    if images:
        patches: list[np.ndarray] = []
        grids: list[tuple[int, int, int]] = []
        for image in images:
            if image.ndim != 3 or image.shape[-1] != config.in_channels:
                raise ValueError(f"A reference image must be (H, W, {config.in_channels}), got {image.shape}")
            height, width = _smart_resize_image(image.shape[0], image.shape[1], factor, min_pixels, max_pixels)
            patches.append(
                _prepare_block(image[None],
                               num_frames=1,
                               source_height=image.shape[0],
                               source_width=image.shape[1],
                               grid=(height, width),
                               config=config))
            grids.append((1, height // config.patch_size, width // config.patch_size))
        pixel_values = np.concatenate(patches, axis=0)
        image_grid_thw = np.asarray(grids, dtype=np.int64)
    pixel_values_videos = video_grid_thw = None
    if videos:
        video_patches: list[np.ndarray] = []
        video_grids: list[tuple[int, int, int]] = []
        for frames in videos:
            if frames.ndim != 4 or frames.shape[-1] != config.in_channels:
                raise ValueError(f"A reference video block must be (frames, H, W, {config.in_channels}), "
                                 f"got {frames.shape}")
            if frames.shape[0] == 0:
                raise ValueError("A reference video block must contain frames")
            num_frames = frames.shape[0]
            height, width = _smart_resize_video(num_frames, frames.shape[1], frames.shape[2],
                                                config.temporal_patch_size, factor, video_min_pixels, video_max_pixels)
            video_patches.append(
                _prepare_block(frames,
                               num_frames=num_frames,
                               source_height=frames.shape[1],
                               source_width=frames.shape[2],
                               grid=(height, width),
                               config=config))
            # ``_prepare_block`` pads a block up to a whole temporal patch, so
            # the grid must count the padded frames: an 11-frame block is 6
            # patches, not 5, and the placeholder count is derived from this.
            padded_frames = num_frames + (-num_frames % config.temporal_patch_size)
            video_grids.append(
                (padded_frames // config.temporal_patch_size, height // config.patch_size, width // config.patch_size))
        pixel_values_videos = np.concatenate(video_patches, axis=0)
        video_grid_thw = np.asarray(video_grids, dtype=np.int64)
    return pixel_values, image_grid_thw, pixel_values_videos, video_grid_thw


def _rotate_half(x):
    first, second = mx.split(x, 2, axis=-1)
    return mx.concatenate([-second, first], axis=-1)


def _layer_norm(x, weight, bias, eps: float = 1e-6):
    mean = mx.mean(x, axis=-1, keepdims=True)
    variance = mx.mean((x - mean)**2, axis=-1, keepdims=True)
    return (x - mean) / mx.sqrt(variance + eps) * weight + bias


def _linear(x, weight, bias):
    return x @ weight.T + bias


def _vision_interpolation_indices_and_weights(
    grid_thw: np.ndarray,
    num_grid_per_side: int,
    spatial_merge_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Bilinear position-table gather in merge-block order (numpy port)."""
    side = num_grid_per_side
    merge = spatial_merge_size
    counts = grid_thw[:, 0] * grid_thw[:, 1] * grid_thw[:, 2]
    heights = np.repeat(grid_thw[:, 1], counts)
    widths = np.repeat(grid_thw[:, 2], counts)
    starts = np.repeat(np.concatenate([[0], np.cumsum(counts)[:-1]]), counts)
    within = (np.arange(int(counts.sum())) - starts) % (heights * widths)

    blocks_wide = widths // merge
    within_col = within % merge
    within_row = (within // merge) % merge
    block_col = (within // (merge * merge)) % blocks_wide
    block_row = within // (merge * merge * blocks_wide)
    rows = block_row * merge + within_row
    cols = block_col * merge + within_col

    def axis_taps_and_weights(indices: np.ndarray, sizes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        positions = indices.astype(np.float32) * (side - 1) / np.clip(sizes - 1, 1, None)
        floors = np.floor(positions)
        offsets = np.arange(2)
        taps = (floors.astype(np.int64)[:, None] + offsets).clip(0, side - 1)
        distances = np.abs(positions[:, None] - floors[:, None] - offsets)
        return taps, np.clip(1 - distances, 0, None)

    row_taps, row_weights = axis_taps_and_weights(rows, heights)
    col_taps, col_weights = axis_taps_and_weights(cols, widths)
    indices = (row_taps[:, :, None] * side + col_taps[:, None, :]).reshape(-1, _INTERPOLATION_TAP)
    weights = (row_weights[:, :, None] * col_weights[:, None, :]).reshape(-1, _INTERPOLATION_TAP)
    return indices.astype(np.int64), weights.astype(np.float32)


def _interpolate_vision_position_embeddings(
    position_embedding: np.ndarray,
    grid_thw: np.ndarray,
    num_grid_per_side: int,
    spatial_merge_size: int,
) -> np.ndarray:
    """Bilinear interpolation of the position table, fp32 throughout."""
    indices, weights = _vision_interpolation_indices_and_weights(grid_thw, num_grid_per_side, spatial_merge_size)
    gathered = position_embedding[indices]  # (N, taps, hidden)
    return (gathered * weights[:, :, None]).sum(axis=1).astype(np.float32)


class _ArrayIndex:
    """The slice of ``_ShardIndex`` the tower loader uses, over loaded arrays."""

    def __init__(self, arrays: Mapping[str, mx.array]) -> None:
        self.key_to_shard = arrays
        self._arrays = arrays

    def get_mlx(self, key: str) -> mx.array:
        return self._arrays[key]


class MLXQwen3VLVisionModel:
    """MLX Qwen3-VL vision tower producing merged tokens and deepstack blocks."""

    def __init__(self, config: Qwen3VLVisionConfig) -> None:
        self.config = config
        if config.hidden_size % config.num_heads:
            raise ValueError(f"vision hidden_size {config.hidden_size} is not divisible by num_heads "
                             f"{config.num_heads}")
        self.spatial_merge_size = config.spatial_merge_size
        self.num_grid_per_side = int(config.num_position_embeddings**0.5)
        self.head_dim = config.hidden_size // config.num_heads
        # The vision rotary table covers head_dim/4 frequencies, gathered for the
        # (row, col) pair and duplicated to head_dim, like the reference.
        dimension = self.head_dim // 2
        self.rotary_dimension = dimension
        self.inv_freq = (1.0 / (10000.0**(np.arange(0, dimension, 2, dtype=np.float32) / dimension))).astype(np.float32)
        self.patch_weight: mx.array | None = None
        self.patch_bias: mx.array | None = None
        self.pos_embed: mx.array | None = None
        self.blocks: list[dict[str, mx.array]] = []
        self.merger: dict[str, mx.array] = {}
        self.deepstack_mergers: list[dict[str, mx.array]] = []
        self.dtype = mx.float32

    @classmethod
    def from_safetensors_dir(cls, component_dir: str | Path, *, dtype: str = "fp32") -> MLXQwen3VLVisionModel:
        """Stream ``model.visual.*`` weights from the checkpoint shards."""
        from fastvideo.mlx_runtime.minimax_h3_conditioner import _ShardIndex

        index = _ShardIndex(Path(component_dir))
        try:
            return cls._from_index(Path(component_dir) / "config.json", index, dtype)
        finally:
            index.close()

    @classmethod
    def from_arrays(cls, config_path: str | Path, arrays: Mapping[str, mx.array], *, dtype: str = "fp32"):
        """Build the tower from already-loaded ``model.visual.*`` arrays (e.g. an encoder cache)."""
        return cls._from_index(config_path, _ArrayIndex(arrays), dtype)

    @classmethod
    def _from_index(cls, config_path: str | Path, index: Any, dtype: str) -> MLXQwen3VLVisionModel:
        config = Qwen3VLVisionConfig.from_config_json(Path(config_path))
        model = cls(config)
        model.dtype = {"fp32": mx.float32, "bf16": mx.bfloat16, "fp16": mx.float16}.get(dtype, mx.float32)
        prefix = "model.visual."
        keys = {key[len(prefix):] for key in index.key_to_shard if key.startswith(prefix)}
        missing = model._expected_weight_keys() - keys
        if missing:
            raise ValueError(f"Checkpoint is missing Qwen3-VL vision weights: {sorted(missing)}")
        cast = lambda key: index.get_mlx(prefix + key).astype(model.dtype)  # noqa: E731
        model.patch_weight = cast("patch_embed.proj.weight").reshape(config.hidden_size, config.in_channels,
                                                                     config.temporal_patch_size, config.patch_size,
                                                                     config.patch_size)
        model.patch_weight = model.patch_weight.reshape(config.hidden_size, -1)
        model.patch_bias = cast("patch_embed.proj.bias")
        model.pos_embed = cast("pos_embed.weight")
        model.blocks = [{
            "norm1.weight": cast(f"blocks.{layer}.norm1.weight"),
            "norm1.bias": cast(f"blocks.{layer}.norm1.bias"),
            "norm2.weight": cast(f"blocks.{layer}.norm2.weight"),
            "norm2.bias": cast(f"blocks.{layer}.norm2.bias"),
            "attn.qkv.weight": cast(f"blocks.{layer}.attn.qkv.weight"),
            "attn.qkv.bias": cast(f"blocks.{layer}.attn.qkv.bias"),
            "attn.proj.weight": cast(f"blocks.{layer}.attn.proj.weight"),
            "attn.proj.bias": cast(f"blocks.{layer}.attn.proj.bias"),
            "mlp.linear_fc1.weight": cast(f"blocks.{layer}.mlp.linear_fc1.weight"),
            "mlp.linear_fc1.bias": cast(f"blocks.{layer}.mlp.linear_fc1.bias"),
            "mlp.linear_fc2.weight": cast(f"blocks.{layer}.mlp.linear_fc2.weight"),
            "mlp.linear_fc2.bias": cast(f"blocks.{layer}.mlp.linear_fc2.bias"),
        } for layer in range(config.depth)]
        model.merger = model._merger_weights(cast, "merger", config.hidden_size)
        model.deepstack_mergers = [
            model._merger_weights(cast, f"deepstack_merger_list.{position}", config.hidden_size)
            for position in range(len(config.deepstack_visual_indexes))
        ]
        mx.eval(*model._arrays())
        logger.info("Loaded Qwen3-VL vision tower (depth=%d, dtype=%s)", config.depth, dtype)
        return model

    def _expected_weight_keys(self) -> set[str]:
        keys = {"patch_embed.proj.weight", "patch_embed.proj.bias", "pos_embed.weight"}
        for layer in range(self.config.depth):
            keys |= {
                f"blocks.{layer}.norm1.weight",
                f"blocks.{layer}.norm1.bias",
                f"blocks.{layer}.norm2.weight",
                f"blocks.{layer}.norm2.bias",
                f"blocks.{layer}.attn.qkv.weight",
                f"blocks.{layer}.attn.qkv.bias",
                f"blocks.{layer}.attn.proj.weight",
                f"blocks.{layer}.attn.proj.bias",
                f"blocks.{layer}.mlp.linear_fc1.weight",
                f"blocks.{layer}.mlp.linear_fc1.bias",
                f"blocks.{layer}.mlp.linear_fc2.weight",
                f"blocks.{layer}.mlp.linear_fc2.bias",
            }
        for name in ["merger"
                     ] + [f"deepstack_merger_list.{i}" for i in range(len(self.config.deepstack_visual_indexes))]:
            keys |= {
                f"{name}.norm.weight",
                f"{name}.norm.bias",
                f"{name}.linear_fc1.weight",
                f"{name}.linear_fc1.bias",
                f"{name}.linear_fc2.weight",
                f"{name}.linear_fc2.bias",
            }
        return keys

    def _merger_weights(self, cast, prefix: str, hidden_size: int) -> dict[str, mx.array]:
        return {
            "norm.weight": cast(f"{prefix}.norm.weight"),
            "norm.bias": cast(f"{prefix}.norm.bias"),
            "linear_fc1.weight": cast(f"{prefix}.linear_fc1.weight"),
            "linear_fc1.bias": cast(f"{prefix}.linear_fc1.bias"),
            "linear_fc2.weight": cast(f"{prefix}.linear_fc2.weight"),
            "linear_fc2.bias": cast(f"{prefix}.linear_fc2.bias"),
        }

    def _arrays(self):
        arrays = [self.patch_weight, self.patch_bias, self.pos_embed]
        for block in self.blocks:
            arrays.extend(block.values())
        arrays.extend(self.merger.values())
        for merger in self.deepstack_mergers:
            arrays.extend(merger.values())
        return arrays

    def _rotary_frequency_table(self, count: int) -> mx.array:
        positions = np.arange(count, dtype=np.float32)
        return mx.array(np.outer(positions, self.inv_freq))

    def _rotary_positions(self, grid_thw: np.ndarray) -> mx.array:
        """Per-token (row, col) frequencies, flattened to head_dim."""
        max_height_width = int(grid_thw[:, 1:].max())
        frequency_table = self._rotary_frequency_table(max_height_width)
        total_tokens = int(np.prod(grid_thw, axis=1).sum())
        position_rows = np.empty((total_tokens, 2), dtype=np.int64)
        offset = 0
        merge = self.spatial_merge_size
        for frames, height, width in grid_thw:
            frames, height, width = int(frames), int(height), int(width)
            merged_height, merged_width = height // merge, width // merge
            block_rows = np.arange(merged_height)
            block_cols = np.arange(merged_width)
            intra_rows = np.arange(merge)
            intra_cols = np.arange(merge)
            rows = block_rows[:, None, None, None] * merge + intra_rows[None, None, :, None]
            cols = block_cols[None, :, None, None] * merge + intra_cols[None, None, None, :]
            rows = np.broadcast_to(rows, (merged_height, merged_width, merge, merge)).reshape(-1)
            cols = np.broadcast_to(cols, (merged_height, merged_width, merge, merge)).reshape(-1)
            # ``tile``, not ``repeat``: the whole spatial grid is laid down once
            # per frame, matching the frame-major patch order the patchify
            # produces. ``repeat`` would instead run each coordinate's frames
            # together, which only coincides for a single frame.
            coordinates = np.tile(np.stack((rows, cols), axis=-1), (frames, 1))
            position_rows[offset:offset + coordinates.shape[0]] = coordinates
            offset += coordinates.shape[0]
        return frequency_table[mx.array(position_rows)].reshape(total_tokens, -1)

    def _interpolate_position_embeddings(self, grid_thw: np.ndarray) -> mx.array:
        embeddings = _interpolate_vision_position_embeddings(
            np.asarray(self.pos_embed, dtype=np.float32),
            grid_thw,
            self.num_grid_per_side,
            self.spatial_merge_size,
        )
        return mx.array(embeddings).astype(self.dtype)

    def _attention(self, hidden, sequence_lengths: list[int], cos, sin, weights: dict[str, mx.array]):
        sequence_length = hidden.shape[0]
        qkv = _linear(hidden, weights["attn.qkv.weight"], weights["attn.qkv.bias"])
        query, key, value = mx.split(qkv.reshape(sequence_length, 3, self.config.num_heads, self.head_dim), 3, axis=1)
        query, key, value = query[:, 0], key[:, 0], value[:, 0]
        cosine = cos[:, None, :]
        sine = sin[:, None, :]
        query = query * cosine + _rotate_half(query) * sine
        key = key * cosine + _rotate_half(key) * sine
        scaling = self.head_dim**-0.5
        outputs = []
        start = 0
        for length in sequence_lengths:
            query_chunk = query[start:start + length].transpose(1, 0, 2)[None]
            key_chunk = key[start:start + length].transpose(1, 0, 2)[None]
            value_chunk = value[start:start + length].transpose(1, 0, 2)[None]
            # Fused attention never materializes the (heads, n, n) scores: a
            # 2048-short-edge reference is ~28k patches, ~50 GB of fp32 scores.
            attended = mx.fast.scaled_dot_product_attention(query_chunk, key_chunk, value_chunk, scale=scaling)
            outputs.append(attended.transpose(0, 2, 1, 3))
            start += length
        merged = mx.concatenate(outputs, axis=1).reshape(sequence_length, -1)
        return _linear(merged, weights["attn.proj.weight"], weights["attn.proj.bias"])

    def _block(self, hidden, sequence_lengths, cos, sin, weights: dict[str, mx.array]):
        normed = _layer_norm(hidden, weights["norm1.weight"], weights["norm1.bias"])
        hidden = hidden + self._attention(normed, sequence_lengths, cos, sin, weights)
        normed = _layer_norm(hidden, weights["norm2.weight"], weights["norm2.bias"])
        return hidden + _mlp(normed, weights, self.config.hidden_act)

    def _merger_forward(self, hidden, weights: dict[str, mx.array], *, postshuffle: bool):
        merger_hidden = self.config.hidden_size * self.spatial_merge_size**2
        if postshuffle:
            hidden = _layer_norm(hidden.reshape(-1, merger_hidden), weights["norm.weight"], weights["norm.bias"])
        else:
            hidden = _layer_norm(hidden, weights["norm.weight"], weights["norm.bias"])
        hidden = hidden.reshape(-1, merger_hidden)
        hidden = _gelu_exact(_linear(hidden, weights["linear_fc1.weight"], weights["linear_fc1.bias"]))
        return _linear(hidden, weights["linear_fc2.weight"], weights["linear_fc2.bias"])

    def encode(self, pixel_values: np.ndarray, grid_thw: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
        """pixel_values/grid_thw -> (merged features, deepstack features), numpy fp32."""
        if self.patch_weight is None:
            raise RuntimeError("MLXQwen3VLVisionModel has no weights; use from_safetensors_dir().")
        grid_thw = np.asarray(grid_thw, dtype=np.int64)
        if grid_thw.ndim != 2 or grid_thw.shape[1] != 3:
            raise ValueError(f"grid_thw must be (n, 3), got {grid_thw.shape}")
        divisor = self.config.temporal_patch_size * self.config.patch_size**2 * self.config.in_channels
        if pixel_values.shape[0] != int(np.prod(grid_thw, axis=1).sum()):
            raise ValueError(f"pixel_values rows {pixel_values.shape[0]} do not match grid patch count "
                             f"{int(np.prod(grid_thw, axis=1).sum())}")
        if pixel_values.shape[1] != divisor:
            raise ValueError(f"pixel_values width {pixel_values.shape[1]} != {divisor}")

        pixels = mx.array(np.ascontiguousarray(pixel_values)).astype(self.dtype)
        hidden = pixels @ self.patch_weight.T + self.patch_bias
        hidden = hidden + self._interpolate_position_embeddings(grid_thw)

        rotary = self._rotary_positions(grid_thw).reshape(hidden.shape[0], -1)
        embedding = mx.concatenate([rotary, rotary], axis=-1)
        cos, sin = mx.cos(embedding), mx.sin(embedding)
        sequence_lengths = [int(value) for value in np.repeat(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0])]

        deepstack_features: list[mx.array] = []
        indexes = tuple(self.config.deepstack_visual_indexes)
        for layer_index, weights in enumerate(self.blocks):
            hidden = self._block(hidden, sequence_lengths, cos, sin, weights)
            if layer_index in indexes:
                merger = self.deepstack_mergers[indexes.index(layer_index)]
                deepstack_features.append(self._merger_forward(hidden, merger, postshuffle=True))
            # Bound the lazy graph to one block so its intermediates free as it goes.
            mx.eval(hidden, *deepstack_features)
        merged = self._merger_forward(hidden, self.merger, postshuffle=False)
        mx.eval(merged, *deepstack_features)
        return (np.asarray(merged,
                           dtype=np.float32), [np.asarray(feature, dtype=np.float32) for feature in deepstack_features])


def _gelu_exact(x):
    """Exact (erf) GELU, as the vision merger uses ``nn.GELU()``."""
    return 0.5 * x * (1.0 + mx.erf(x * (2.0**-0.5)))


def _gelu_tanh(x):
    """Tanh-approximate GELU (``gelu_pytorch_tanh``), the vision MLP activation."""
    inner = 0.7978845608028654 * (x + 0.044715 * x**3)
    return 0.5 * x * (1.0 + mx.tanh(inner))


def _mlp(hidden, weights: dict[str, mx.array], hidden_act: str):
    activated = _linear(hidden, weights["mlp.linear_fc1.weight"], weights["mlp.linear_fc1.bias"])
    activated = _gelu_tanh(activated) if "tanh" in hidden_act else _gelu_exact(activated)
    return _linear(activated, weights["mlp.linear_fc2.weight"], weights["mlp.linear_fc2.bias"])


def qwen3_vl_rope_index(
    input_ids: np.ndarray,
    image_grid_thw: np.ndarray | None,
    video_grid_thw: np.ndarray | None,
    *,
    vision_start_token_id: int,
    image_token_id: int,
    video_token_id: int,
    spatial_merge_size: int,
    attention_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Faithful numpy port of ``MiniMaxH3Qwen3VLConditioner._get_rope_index``.

    Returns ``(3, S)`` int64 for 1-D input and ``(3, B, S)`` for a ``(B, S)``
    batch. Positions restart per text/vision piece so the three MRoPE axes carry
    the temporal/height/width structure the language stack expects.
    """
    single = input_ids.ndim == 1
    input_ids = input_ids[None] if single else input_ids
    if image_grid_thw is None and video_grid_thw is None:
        if attention_mask is None:
            positions = np.arange(input_ids.shape[1], dtype=np.int64)[None, None, :].repeat(3, axis=0)
            positions = np.repeat(positions, input_ids.shape[0], axis=1)
        else:
            positions = np.cumsum(attention_mask.astype(np.int64), axis=-1) - 1
            positions[attention_mask == 0] = 1
            positions = positions[None].repeat(3, axis=0)
        return positions[:, 0] if single else positions
    if video_grid_thw is not None:
        video_grid_thw = np.repeat(video_grid_thw, video_grid_thw[:, 0], axis=0).copy()
        video_grid_thw[:, 0] = 1
    mask = np.ones_like(input_ids) if attention_mask is None else attention_mask
    position_ids = np.ones((3, input_ids.shape[0], input_ids.shape[1]), dtype=np.int64)
    image_index = 0
    video_index = 0
    merge = spatial_merge_size
    for batch_index, sequence in enumerate(input_ids):
        sequence = sequence[mask[batch_index] == 1]
        starts = np.argwhere(sequence == vision_start_token_id).reshape(-1)
        vision_tokens = sequence[starts + 1]
        image_count = int((vision_tokens == image_token_id).sum())
        video_count = int((vision_tokens == video_token_id).sum())
        tokens = sequence.tolist()
        pieces: list[np.ndarray] = []
        start = 0
        remaining_images, remaining_videos = image_count, video_count
        for _ in range(image_count + video_count):
            image_end = tokens.index(image_token_id, start) if remaining_images else len(tokens) + 1
            video_end = tokens.index(video_token_id, start) if remaining_videos else len(tokens) + 1
            if image_end < video_end:
                if image_grid_thw is None:
                    raise ValueError("Qwen3-VL image tokens require image_grid_thw")
                frames, height, width = (int(value) for value in image_grid_thw[image_index])
                image_index += 1
                remaining_images -= 1
                end = image_end
            else:
                if video_grid_thw is None:
                    raise ValueError("Qwen3-VL video tokens require video_grid_thw")
                frames, height, width = (int(value) for value in video_grid_thw[video_index])
                video_index += 1
                remaining_videos -= 1
                end = video_end
            grid_frames, grid_height, grid_width = frames, height // merge, width // merge
            text_length = end - start
            offset = int(pieces[-1].max()) + 1 if pieces else 0
            pieces.append(np.arange(text_length, dtype=np.int64)[None, :].repeat(3, axis=0) + offset)
            temporal = np.repeat(np.arange(grid_frames)[:, None], grid_height * grid_width, axis=1).reshape(-1)
            rows = np.broadcast_to(np.arange(grid_height)[None, :, None],
                                   (grid_frames, grid_height, grid_width)).reshape(-1)
            columns = np.broadcast_to(np.arange(grid_width)[None, None, :],
                                      (grid_frames, grid_height, grid_width)).reshape(-1)
            pieces.append(np.stack((temporal, rows, columns)) + text_length + offset)
            start = end + grid_frames * grid_height * grid_width
        if start < len(tokens):
            offset = int(pieces[-1].max()) + 1 if pieces else 0
            pieces.append(np.arange(len(tokens) - start, dtype=np.int64)[None, :].repeat(3, axis=0) + offset)
        positions = np.concatenate(pieces, axis=1)
        position_ids[:, batch_index, mask[batch_index] == 1] = positions
    return position_ids[:, 0] if single else position_ids


def read_vision_index_header(component_dir: str | Path) -> dict[str, Any]:
    """Expose the vision-only shard keys for callers building their own readers."""
    from fastvideo.mlx_runtime.minimax_h3_conditioner import _ShardIndex

    index = _ShardIndex(Path(component_dir))
    try:
        return {key: shard for key, shard in index.key_to_shard.items() if key.startswith("model.visual.")}
    finally:
        index.close()


def safetensors_bf16_available(path: str | Path) -> bool:
    """True when ``path`` is a readable safetensors shard (header probe)."""
    try:
        with open(path, "rb") as handle:
            (header_len, ) = struct.unpack("<Q", handle.read(8))
            json.loads(handle.read(header_len))
        return True
    except (OSError, ValueError, struct.error):
        return False


__all__ = [
    "MLXQwen3VLVisionModel",
    "Qwen3VLVisionConfig",
    "prepare_vision_inputs",
    "qwen3_vl_rope_index",
]
