# SPDX-License-Identifier: Apache-2.0
"""Raw dataset manifest support for MiniMax H3 Ref2VA training and validation."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
from numbers import Integral, Real
from pathlib import Path
from typing import Any, Literal

from fastvideo.pipelines.basic.minimax_h3.reference import (
    MiniMaxH3Reference,
    decode_reference_video,
    validate_references,
)

MINIMAX_H3_REF2VA_RAW_SCHEMA_VERSION = "minimax_h3_ref2va_raw_v1"

RawReferenceType = Literal["image", "video", "audio", "video_audio"]


@dataclass(frozen=True)
class MiniMaxH3RawReference:
    """One validated, ordered reference entry from a raw dataset manifest."""

    media_type: RawReferenceType
    image_path: Path | None = None
    video_path: Path | None = None
    audio_path: Path | None = None
    fps: float | None = None
    sample_rate: int | None = None


@dataclass(frozen=True)
class MiniMaxH3Ref2VARawSample:
    """One validated raw Ref2VA sample with paths resolved against its manifest."""

    sample_id: str
    target_file: str
    target_video_path: Path
    caption: str
    references: tuple[MiniMaxH3RawReference, ...]


def _non_null_keys(value: dict[str, Any]) -> set[str]:
    """Ignore null union fields introduced by Hugging Face JSON loading."""
    return {key for key, item in value.items() if item is not None}


def _resolve_media_path(value: Any, *, manifest_path: Path, field: str, context: str) -> tuple[str, Path]:
    if not isinstance(value, str) or not value.strip():
        raise TypeError(f"{context} field {field!r} must be a non-empty path string")
    raw_path = value.strip()
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = manifest_path.parent / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{context} field {field!r} does not exist at {path}")
    return raw_path, path


def _optional_positive_fps(value: Any, *, context: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, Real) or float(value) <= 0:
        raise ValueError(f"{context} fps must be a positive number, got {value!r}")
    return float(value)


def _optional_positive_sample_rate(value: Any, *, context: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, Integral) or int(value) <= 0:
        raise ValueError(f"{context} sample_rate must be a positive integer, got {value!r}")
    return int(value)


def _parse_reference(
    entry: Any,
    *,
    manifest_path: Path,
    sample_id: str,
    index: int,
) -> MiniMaxH3RawReference:
    context = f"Sample {sample_id!r} reference {index}"
    if not isinstance(entry, dict):
        raise TypeError(f"{context} must be a JSON object")

    raw_type = entry.get("type")
    allowed_keys = {
        "image": ({"type", "image"}, set()),
        "video": ({"type", "video"}, {"fps"}),
        "audio": ({"type", "audio"}, {"sample_rate"}),
        "video_audio": ({"type", "video", "audio"}, {"fps", "sample_rate"}),
    }
    if raw_type not in allowed_keys:
        raise ValueError(f"{context} type must be image/video/audio/video_audio, got {raw_type!r}")

    required, optional = allowed_keys[raw_type]
    entry_keys = _non_null_keys(entry)
    if not required <= entry_keys or not entry_keys <= required | optional:
        raise ValueError(f"{context} type {raw_type!r} requires {sorted(required)} and permits "
                         f"{sorted(optional)}, got non-null fields {sorted(entry_keys)}")

    image_path = None
    video_path = None
    audio_path = None
    if raw_type == "image":
        _, image_path = _resolve_media_path(
            entry["image"],
            manifest_path=manifest_path,
            field="image",
            context=context,
        )
    if raw_type in {"video", "video_audio"}:
        _, video_path = _resolve_media_path(
            entry["video"],
            manifest_path=manifest_path,
            field="video",
            context=context,
        )
    if raw_type in {"audio", "video_audio"}:
        _, audio_path = _resolve_media_path(
            entry["audio"],
            manifest_path=manifest_path,
            field="audio",
            context=context,
        )

    return MiniMaxH3RawReference(
        media_type=raw_type,
        image_path=image_path,
        video_path=video_path,
        audio_path=audio_path,
        fps=_optional_positive_fps(entry.get("fps"), context=context),
        sample_rate=_optional_positive_sample_rate(entry.get("sample_rate"), context=context),
    )


def parse_minimax_h3_ref2va_raw_sample(
    record: Any,
    *,
    manifest_path: Path,
    allow_extra_fields: bool = False,
) -> MiniMaxH3Ref2VARawSample:
    """Validate one raw record and resolve every media path."""
    manifest_path = manifest_path.expanduser().resolve()
    if not isinstance(record, dict):
        raise TypeError(f"A record in {manifest_path} must be a JSON object")

    required_fields = {"schema_version", "id", "target", "caption", "references"}
    record_keys = _non_null_keys(record)
    missing = required_fields - record_keys
    extra = record_keys - required_fields
    if missing or (extra and not allow_extra_fields):
        raise ValueError(f"A record in {manifest_path} requires exactly {sorted(required_fields)}; "
                         f"missing={sorted(missing)}, extra={sorted(extra)}")
    if record["schema_version"] != MINIMAX_H3_REF2VA_RAW_SCHEMA_VERSION:
        raise ValueError(f"Unsupported raw schema {record['schema_version']!r}; "
                         f"expected {MINIMAX_H3_REF2VA_RAW_SCHEMA_VERSION!r}")

    sample_id = record["id"]
    if not isinstance(sample_id, str) or not sample_id.strip():
        raise TypeError("Raw Ref2VA field 'id' must be a non-empty string")
    sample_id = sample_id.strip()

    target = record["target"]
    if not isinstance(target, dict) or _non_null_keys(target) != {"video"}:
        raise ValueError(f"Sample {sample_id!r} target must contain exactly one non-null 'video' field")
    target_file, target_video_path = _resolve_media_path(
        target["video"],
        manifest_path=manifest_path,
        field="video",
        context=f"Sample {sample_id!r} target",
    )

    caption = record["caption"]
    if not isinstance(caption, str) or not caption.strip():
        raise TypeError(f"Sample {sample_id!r} caption must be a non-empty string")
    caption = caption.strip()

    raw_references = record["references"]
    if not isinstance(raw_references, list):
        raise TypeError(f"Sample {sample_id!r} references must be an ordered list")
    references = tuple(
        _parse_reference(
            entry,
            manifest_path=manifest_path,
            sample_id=sample_id,
            index=index,
        ) for index, entry in enumerate(raw_references))

    return MiniMaxH3Ref2VARawSample(
        sample_id=sample_id,
        target_file=target_file,
        target_video_path=target_video_path,
        caption=caption,
        references=references,
    )


def _records_from_json_document(document: Any, manifest_path: Path) -> list[Any]:
    if isinstance(document, list):
        return document
    if isinstance(document, dict) and set(document) == {"data"}:
        records = document["data"]
        if not isinstance(records, list):
            raise TypeError(f"Manifest {manifest_path} field 'data' must be a list")
        return records
    if isinstance(document, dict):
        return [document]
    raise TypeError(f"Manifest {manifest_path} must contain an object, an array, or an object with a 'data' array")


def load_minimax_h3_ref2va_raw_samples(manifest_path: Path) -> list[MiniMaxH3Ref2VARawSample]:
    """Load pretty JSON, JSON arrays/wrappers, or standard one-object-per-line JSONL."""
    manifest_path = manifest_path.expanduser().resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"MiniMax H3 raw manifest is missing at {manifest_path}")
    text = manifest_path.read_text(encoding="utf-8")
    if not text.strip():
        raise ValueError(f"MiniMax H3 raw manifest is empty at {manifest_path}")

    try:
        records = _records_from_json_document(json.loads(text), manifest_path)
    except json.JSONDecodeError:
        records = []
        for line_number, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"Manifest {manifest_path} is neither valid JSON nor one-object-per-line JSONL; "
                                 f"line {line_number} is invalid") from error

    if not records:
        raise ValueError(f"MiniMax H3 raw manifest contains no samples at {manifest_path}")
    samples = [parse_minimax_h3_ref2va_raw_sample(record, manifest_path=manifest_path) for record in records]
    sample_ids = [sample.sample_id for sample in samples]
    duplicate_ids = sorted(sample_id for sample_id, count in Counter(sample_ids).items() if count > 1)
    if duplicate_ids:
        raise ValueError(f"Manifest {manifest_path} contains duplicate sample ids: {duplicate_ids}")
    return samples


def build_minimax_h3_references(
    references: tuple[MiniMaxH3RawReference, ...] | list[MiniMaxH3RawReference], ) -> list[MiniMaxH3Reference]:
    """Convert raw reference specs while preserving their declared order and audio semantics."""
    converted: list[MiniMaxH3Reference] = []
    for index, reference in enumerate(references):
        if reference.media_type == "image":
            if reference.image_path is None:
                raise RuntimeError(f"Validated image reference {index} has no image path")
            converted.append(MiniMaxH3Reference(source=reference.image_path, media_type="image"))
            continue
        if reference.media_type == "audio":
            if reference.audio_path is None:
                raise RuntimeError(f"Validated audio reference {index} has no audio path")
            converted.append(
                MiniMaxH3Reference(
                    source=reference.audio_path,
                    media_type="audio",
                    sample_rate=reference.sample_rate,
                ))
            continue
        if reference.video_path is None:
            raise RuntimeError(f"Validated video reference {index} has no video path")
        if reference.media_type == "video":
            # A raw type=video is explicitly silent even if its container has
            # a soundtrack. Passing pixels avoids prepare_reference adopting it.
            frames, decoded_fps, decoded_soundtrack = decode_reference_video(reference.video_path)
            del decoded_soundtrack
            converted.append(
                MiniMaxH3Reference(
                    source=frames,
                    media_type="video",
                    fps=reference.fps if reference.fps is not None else decoded_fps,
                ))
            continue
        if reference.audio_path is None:
            raise RuntimeError(f"Validated video_audio reference {index} has no audio path")
        converted.append(
            MiniMaxH3Reference(
                source=reference.video_path,
                media_type="video",
                soundtrack=None if reference.video_path == reference.audio_path else reference.audio_path,
                fps=reference.fps,
                sample_rate=reference.sample_rate,
            ))

    return validate_references(converted)


__all__ = [
    "MINIMAX_H3_REF2VA_RAW_SCHEMA_VERSION",
    "MiniMaxH3RawReference",
    "MiniMaxH3Ref2VARawSample",
    "build_minimax_h3_references",
    "load_minimax_h3_ref2va_raw_samples",
    "parse_minimax_h3_ref2va_raw_sample",
]
