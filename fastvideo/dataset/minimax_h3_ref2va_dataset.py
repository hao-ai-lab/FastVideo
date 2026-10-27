# SPDX-License-Identifier: Apache-2.0
"""Dataset contract for precomputed MiniMax H3 Ref2VA training samples."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pyarrow as pa
import torch
from torchdata.stateful_dataloader import StatefulDataLoader

from fastvideo.dataset.parquet_dataset_map_style import (
    LatentsParquetMapStyleDataset,
    passthrough,
    read_row_from_parquet_file,
)

MINIMAX_H3_REF2VA_SCHEMA_VERSION = "minimax_h3_ref2va_precomputed_v1"
MINIMAX_H3_REF2VA_VISUAL_ROW_WIDTH = 96
MINIMAX_H3_REF2VA_AUDIO_ROW_WIDTH = 32


def _tensor_triplet(name: str) -> list[pa.Field]:
    return [
        pa.field(f"{name}_bytes", pa.binary()),
        pa.field(f"{name}_shape", pa.list_(pa.int64())),
        pa.field(f"{name}_dtype", pa.string()),
    ]


pyarrow_schema_minimax_h3_ref2va = pa.schema([
    pa.field("id", pa.string()),
    pa.field("schema_version", pa.string()),

    # Target video/audio supervision.
    *_tensor_triplet("vae_latent"),
    *_tensor_triplet("audio_latent"),

    # Exact-length Qwen output. This field is not padded or cropped.
    *_tensor_triplet("text_embedding"),
    pa.field("text_token_tags", pa.list_(pa.int64())),

    # Cached Ref2VA conditions.
    #
    # ref_visual_anchor already contains the fixed:
    #   0.999 * clean_rows + 0.001 * fixed_noise
    #
    # ref_audio_anchor contains clean audio-VAE rows.
    *_tensor_triplet("ref_visual_anchor"),
    *_tensor_triplet("ref_audio_anchor"),

    # Ordered, canonical prepared-reference metadata.
    pa.field(
        "references",
        pa.list_(
            pa.struct([
                pa.field("media_type", pa.string()),
                pa.field("has_audio", pa.bool_()),
                pa.field("num_latent_frames", pa.int64()),
                pa.field("latent_height", pa.int64()),
                pa.field("latent_width", pa.int64()),
                pa.field("num_audio_latents", pa.int64()),
            ])),
    ),

    # Target metadata.
    pa.field("file_name", pa.string()),
    pa.field("caption", pa.string()),
    pa.field("media_type", pa.string()),
    pa.field("width", pa.int64()),
    pa.field("height", pa.int64()),
    pa.field("num_frames", pa.int64()),
    pa.field("duration_sec", pa.float64()),
    pa.field("fps", pa.float64()),
    pa.field("audio_sample_rate", pa.int64()),
])

_TENSOR_FIELDS = (
    "vae_latent",
    "audio_latent",
    "text_embedding",
    "ref_visual_anchor",
    "ref_audio_anchor",
)

_INFO_FIELDS = tuple(
    name
    for name in pyarrow_schema_minimax_h3_ref2va.names
    if not name.endswith(("_bytes", "_shape", "_dtype"))
    and name != "text_token_tags"
)


def _decode_float32_tensor(
    row: dict[str, Any],
    name: str,
) -> torch.Tensor:
    """Decode one tensor while preserving zero-length dimensions."""
    shape_value = row.get(f"{name}_shape")
    payload = row.get(f"{name}_bytes")
    dtype_name = row.get(f"{name}_dtype")

    if not isinstance(shape_value, list) or not shape_value:
        raise ValueError(
            f"{name}_shape must be a non-empty list, got {shape_value!r}"
        )
    if any(
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 0
        for value in shape_value
    ):
        raise ValueError(
            f"{name}_shape must contain non-negative integers, "
            f"got {shape_value!r}"
        )
    if not isinstance(payload, bytes | bytearray | memoryview):
        raise TypeError(f"{name}_bytes must be a bytes-like object")
    if dtype_name != "float32":
        raise ValueError(
            f"{name}_dtype must be 'float32', got {dtype_name!r}"
        )

    shape = tuple(shape_value)
    array = np.frombuffer(payload, dtype=np.float32)
    expected_elements = int(np.prod(shape, dtype=np.int64))

    if array.size != expected_elements:
        raise ValueError(
            f"{name} payload contains {array.size} float32 values, "
            f"but shape {shape} requires {expected_elements}"
        )

    return torch.from_numpy(array.reshape(shape).copy())


def _validate_reference_contract(
    row: dict[str, Any],
    visual_anchor: torch.Tensor,
    audio_anchor: torch.Tensor,
) -> None:
    """Validate ordered reference geometry against cached condition rows."""
    references = row.get("references")
    if not isinstance(references, list):
        raise TypeError("references must be an ordered list")
    if not all(isinstance(reference, dict) for reference in references):
        raise TypeError("Every prepared reference must be a mapping")

    count = len(references)
    if count > 12:
        raise ValueError("MiniMax H3 supports at most 12 references")

    media_types = [
        reference.get("media_type")
        for reference in references
    ]
    if any(
        media_type not in {"image", "video", "audio"}
        for media_type in media_types
    ):
        raise ValueError(
            f"Unsupported prepared reference media types: {media_types!r}"
        )
    if media_types.count("image") > 9:
        raise ValueError(
            "MiniMax H3 supports at most 9 image references"
        )
    if media_types.count("video") > 3:
        raise ValueError(
            "MiniMax H3 supports at most 3 video references"
        )
    if media_types.count("audio") > 3:
        raise ValueError(
            "MiniMax H3 supports at most 3 standalone audio references"
        )
    if count and all(
        media_type == "audio"
        for media_type in media_types
    ):
        raise ValueError(
            "A non-empty Ref2VA sample requires at least one "
            "image or video reference"
        )

    required_fields = {
        "media_type",
        "has_audio",
        "num_latent_frames",
        "latent_height",
        "latent_width",
        "num_audio_latents",
    }

    expected_visual_rows = 0
    expected_audio_rows = 0

    for index, reference in enumerate(references):
        if set(reference) != required_fields:
            raise ValueError(
                f"Prepared reference {index} must contain exactly "
                f"{sorted(required_fields)}, got {sorted(reference)}"
            )

        media_type = reference["media_type"]
        has_audio = reference["has_audio"]
        num_frames = reference["num_latent_frames"]
        height = reference["latent_height"]
        width = reference["latent_width"]
        num_audio_latents = reference["num_audio_latents"]

        geometry_values = (
            num_frames,
            height,
            width,
            num_audio_latents,
        )
        if any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < 0
            for value in geometry_values
        ):
            raise ValueError(
                f"Reference {index} geometry must contain "
                f"non-negative integers, got {geometry_values}"
            )

        if not isinstance(has_audio, bool):
            raise TypeError(
                f"Reference {index} has_audio must be bool"
            )
        if media_type == "image" and has_audio:
            raise ValueError(
                f"Image reference {index} cannot carry audio"
            )
        if media_type == "audio" and not has_audio:
            raise ValueError(
                f"Audio reference {index} must carry audio"
            )

        if media_type == "audio":
            if (num_frames, height, width) != (0, 0, 0):
                raise ValueError(
                    f"Audio reference {index} must not carry "
                    "visual geometry"
                )
        else:
            visual_geometry = (num_frames, height, width)
            if any(value <= 0 for value in visual_geometry):
                raise ValueError(
                    f"Visual reference {index} has incomplete "
                    f"geometry {visual_geometry}"
                )
            if height % 2 or width % 2:
                raise ValueError(
                    f"Visual reference {index} height and width "
                    "must be divisible by 2"
                )

            expected_visual_rows += (
                num_frames
                * (height // 2)
                * (width // 2)
            )

        if has_audio:
            if num_audio_latents <= 0:
                raise ValueError(
                    f"Audio-bearing reference {index} requires "
                    "positive audio latent length"
                )
            expected_audio_rows += 2 * num_audio_latents
        elif num_audio_latents != 0:
            raise ValueError(
                f"Silent reference {index} must have zero "
                "audio latent length"
            )

    expected_visual_shape = (
        expected_visual_rows,
        MINIMAX_H3_REF2VA_VISUAL_ROW_WIDTH,
    )
    expected_audio_shape = (
        expected_audio_rows,
        MINIMAX_H3_REF2VA_AUDIO_ROW_WIDTH,
    )

    if tuple(visual_anchor.shape) != expected_visual_shape:
        raise ValueError(
            f"ref_visual_anchor must have shape "
            f"{expected_visual_shape}, "
            f"got {tuple(visual_anchor.shape)}"
        )
    if tuple(audio_anchor.shape) != expected_audio_shape:
        raise ValueError(
            f"ref_audio_anchor must have shape "
            f"{expected_audio_shape}, "
            f"got {tuple(audio_anchor.shape)}"
        )


def collate_minimax_h3_ref2va_rows(
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Collate one row without padding or truncating Qwen tokens."""
    if len(rows) != 1:
        raise ValueError(
            "MiniMax H3 Ref2VA requires exactly one row per batch, "
            f"got {len(rows)}"
        )

    row = rows[0]
    if (
        row.get("schema_version")
        != MINIMAX_H3_REF2VA_SCHEMA_VERSION
    ):
        raise ValueError(
            "Unsupported Ref2VA schema version: "
            f"{row.get('schema_version')!r}; expected "
            f"{MINIMAX_H3_REF2VA_SCHEMA_VERSION!r}"
        )

    tensors = {
        name: _decode_float32_tensor(row, name)
        for name in _TENSOR_FIELDS
    }

    text_embedding = tensors["text_embedding"]
    if (
        text_embedding.ndim != 2
        or text_embedding.shape[0] == 0
        or text_embedding.shape[1] != 5120
    ):
        raise ValueError(
            "text_embedding must have shape [length, 5120], "
            f"got {tuple(text_embedding.shape)}"
        )

    raw_tags = row.get("text_token_tags")
    if not isinstance(raw_tags, list):
        raise TypeError("text_token_tags must be a list")
    if any(
        isinstance(tag, bool) or not isinstance(tag, int)
        for tag in raw_tags
    ):
        raise TypeError("text_token_tags must contain integers")

    text_token_tags = torch.tensor(
        raw_tags,
        dtype=torch.long,
    )
    if text_token_tags.shape != text_embedding.shape[:1]:
        raise ValueError(
            "text_token_tags must align one-to-one with "
            "text_embedding"
        )
    if not bool(
        (
            (text_token_tags == 0)
            | (text_token_tags == 1)
        ).all()
    ):
        raise ValueError(
            "text_token_tags may contain only MiniMax H3 "
            "vision=0 and text=1 tags"
        )

    _validate_reference_contract(
        row,
        tensors["ref_visual_anchor"],
        tensors["ref_audio_anchor"],
    )

    info = {
        field: row.get(field)
        for field in _INFO_FIELDS
    }
    info["prompt"] = info.get("caption", "")

    return {
        **{
            name: tensor.unsqueeze(0)
            for name, tensor in tensors.items()
        },
        "text_attention_mask": torch.ones(
            (1, text_embedding.shape[0]),
            dtype=torch.float32,
        ),
        "text_token_tags": text_token_tags.unsqueeze(0),
        "info_list": [info],
        "caption_text": [info.get("caption", "")],
    }


class MiniMaxH3Ref2VAParquetDataset(
    LatentsParquetMapStyleDataset
):
    """Map-style dataset for variable-length Ref2VA conditions."""

    def __init__(
        self,
        path: str | Sequence[str] | dict[str, int],
        batch_size: int,
        *,
        drop_last: bool = True,
        seed: int = 42,
    ) -> None:
        if batch_size != 1:
            raise ValueError(
                "MiniMax H3 Ref2VA requires batch_size=1"
            )

        super().__init__(
            path=path,
            batch_size=batch_size,
            parquet_schema=pyarrow_schema_minimax_h3_ref2va,
            cfg_rate=0.0,
            seed=seed,
            drop_last=drop_last,
            # __getitems__ below bypasses the generic text-padding
            # collator. The parent constructor still requires a value.
            text_padding_length=1,
        )

    def get_validation_negative_prompt(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, str]:
        raise RuntimeError(
            "MiniMax H3 Ref2VA does not use a dataset-level "
            "CFG negative prompt"
        )

    def __getitems__(
        self,
        indices: list[int],
    ) -> dict[str, Any]:
        rows = [
            read_row_from_parquet_file(
                self.parquet_files,
                index,
                self.lengths,
            )
            for index in indices
        ]
        return collate_minimax_h3_ref2va_rows(rows)


def build_minimax_h3_ref2va_dataloader(
    path: str | Sequence[str] | dict[str, int],
    batch_size: int,
    num_data_workers: int,
    *,
    drop_last: bool = True,
    seed: int = 42,
) -> tuple[
    MiniMaxH3Ref2VAParquetDataset,
    StatefulDataLoader,
]:
    """Build the stateful loader using FastVideo's DP/SP sampler."""
    dataset = MiniMaxH3Ref2VAParquetDataset(
        path,
        batch_size,
        drop_last=drop_last,
        seed=seed,
    )

    loader = StatefulDataLoader(
        dataset,
        batch_sampler=dataset.sampler,
        collate_fn=passthrough,
        num_workers=num_data_workers,
        pin_memory=True,
        persistent_workers=num_data_workers > 0,
    )
    return dataset, loader


__all__ = [
    "MINIMAX_H3_REF2VA_AUDIO_ROW_WIDTH",
    "MINIMAX_H3_REF2VA_SCHEMA_VERSION",
    "MINIMAX_H3_REF2VA_VISUAL_ROW_WIDTH",
    "MiniMaxH3Ref2VAParquetDataset",
    "build_minimax_h3_ref2va_dataloader",
    "collate_minimax_h3_ref2va_rows",
    "pyarrow_schema_minimax_h3_ref2va",
]