# SPDX-License-Identifier: Apache-2.0
"""Dataset normalization utilities for InterleaveThinker training files."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
import os
from pathlib import Path
from typing import Any, Literal

InterleaveDatasetKind = Literal["planner_sft", "planner_rl", "critic_sft", "critic_rl"]

DEFAULT_FILENAMES: dict[InterleaveDatasetKind, str] = {
    "planner_sft": "planner_sft.json",
    "planner_rl": "planner_rl.jsonl",
    "critic_sft": "critic_sft.json",
    "critic_rl": "critic_rl.jsonl",
}

IMAGE_PATH_KEYS = (
    "origin_image_path",
    "previous_image_path",
    "edited_image_path",
    "generated_image_path",
    "input_image_path",
    "output_image_path",
    "image_path",
    "target_img",
)
IMAGE_LIST_KEYS = (
    "images",
    "input_image_paths",
    "image_paths",
)
IMAGE_EXTENSIONS = frozenset({
    ".bmp",
    ".gif",
    ".jpeg",
    ".jpg",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
})


def load_interleave_dataset(
    data_path: str | os.PathLike[str],
    *,
    kind: InterleaveDatasetKind,
    image_dir: str | os.PathLike[str] = "",
    validate_image_files: bool = False,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for file_path in _resolve_dataset_files(data_path, kind):
        for raw in _load_json_records(file_path):
            records.append(
                normalize_interleave_dataset_record(
                    raw,
                    kind=kind,
                    image_dir=image_dir,
                    validate_image_files=validate_image_files,
                ))
    if not records:
        raise ValueError(f"No {kind} records found at {data_path!s}")
    return records


def load_planner_sft_records(
    data_path: str | os.PathLike[str],
    *,
    image_dir: str | os.PathLike[str] = "",
    validate_image_files: bool = False,
) -> list[dict[str, Any]]:
    return load_interleave_dataset(
        data_path,
        kind="planner_sft",
        image_dir=image_dir,
        validate_image_files=validate_image_files,
    )


def load_planner_rl_records(
    data_path: str | os.PathLike[str],
    *,
    image_dir: str | os.PathLike[str] = "",
    validate_image_files: bool = False,
) -> list[dict[str, Any]]:
    return load_interleave_dataset(
        data_path,
        kind="planner_rl",
        image_dir=image_dir,
        validate_image_files=validate_image_files,
    )


def load_critic_sft_records(
    data_path: str | os.PathLike[str],
    *,
    image_dir: str | os.PathLike[str] = "",
    validate_image_files: bool = False,
) -> list[dict[str, Any]]:
    return load_interleave_dataset(
        data_path,
        kind="critic_sft",
        image_dir=image_dir,
        validate_image_files=validate_image_files,
    )


def load_critic_rl_records(
    data_path: str | os.PathLike[str],
    *,
    image_dir: str | os.PathLike[str] = "",
    validate_image_files: bool = False,
) -> list[dict[str, Any]]:
    return load_interleave_dataset(
        data_path,
        kind="critic_rl",
        image_dir=image_dir,
        validate_image_files=validate_image_files,
    )


def normalize_interleave_dataset_record(
    record: Mapping[str, Any],
    *,
    kind: InterleaveDatasetKind,
    image_dir: str | os.PathLike[str] = "",
    validate_image_files: bool = False,
) -> dict[str, Any]:
    normalized = _resolve_record_image_paths(
        record,
        image_dir=image_dir,
        validate_image_files=validate_image_files,
    )
    if kind == "planner_sft":
        return normalize_planner_sft_record(normalized)
    if kind == "planner_rl":
        return normalize_planner_rl_record(normalized)
    if kind == "critic_sft":
        return normalize_critic_sft_record(normalized)
    if kind == "critic_rl":
        return normalize_critic_rl_record(normalized)
    raise ValueError(f"Unsupported InterleaveThinker dataset kind: {kind!r}")


def normalize_planner_sft_record(record: Mapping[str, Any]) -> dict[str, Any]:
    normalized = dict(record)
    messages = _require_messages(normalized, where="planner_sft")
    user_message = _first_message_content(messages, "user")
    assistant_message = _first_message_content(messages, "assistant")
    if not user_message:
        raise ValueError("planner_sft record requires a user message")
    if not assistant_message:
        raise ValueError("planner_sft record requires an assistant message")
    normalized.setdefault("instruction", user_message)
    normalized.setdefault("response", assistant_message)
    images = _string_sequence(normalized.get("images"))
    if images:
        normalized.setdefault("input_image_paths", images)
    return normalized


def normalize_planner_rl_record(record: Mapping[str, Any]) -> dict[str, Any]:
    normalized = dict(record)
    if "messages" in normalized:
        messages = _require_messages(normalized, where="planner_rl")
        normalized.setdefault("instruction", _first_message_content(messages, "user"))
    instruction = _first_text(
        normalized.get("instruction"),
        normalized.get("text_input"),
        normalized.get("origin_prompt"),
        normalized.get("prompt"),
    )
    if not instruction:
        raise ValueError("planner_rl record requires instruction, text_input, origin_prompt, or prompt")
    normalized["instruction"] = instruction
    images = _string_sequence(normalized.get("images"))
    if images:
        normalized.setdefault("input_image_paths", images)
    return normalized


def normalize_critic_sft_record(record: Mapping[str, Any]) -> dict[str, Any]:
    normalized = dict(record)
    if "messages" in normalized:
        messages = _require_messages(normalized, where="critic_sft")
        normalized.setdefault("instruction", _first_message_content(messages, "user"))
        normalized.setdefault("response", _first_message_content(messages, "assistant"))

    images = _string_sequence(normalized.get("images"))
    if images:
        if len(images) < 2:
            raise ValueError("critic_sft ShareGPT record requires at least two images")
        normalized.setdefault("origin_image_path", images[0])
        normalized.setdefault("edited_image_path", images[1])

    _require_text(normalized, "origin_image_path", "critic_sft")
    _require_text(normalized, "edited_image_path", "critic_sft")
    normalized.setdefault("previous_image_path", normalized["origin_image_path"])
    normalized.setdefault("generated_image_path", normalized["edited_image_path"])
    if "previous_prompt" not in normalized and "rewritten_prompt" in normalized:
        normalized["previous_prompt"] = normalized["rewritten_prompt"]
    if "rewritten_prompt" not in normalized and "previous_prompt" in normalized:
        normalized["rewritten_prompt"] = normalized["previous_prompt"]
    return normalized


def normalize_critic_rl_record(record: Mapping[str, Any]) -> dict[str, Any]:
    normalized = dict(record)
    _require_text(normalized, "origin_prompt", "critic_rl")
    _require_text(normalized, "origin_image_path", "critic_rl")
    _require_text(normalized, "edited_image_path", "critic_rl")

    previous_prompt = _first_text(
        normalized.get("previous_prompt"),
        normalized.get("rewritten_prompt"),
        normalized.get("refine_prompt"),
    )
    if not previous_prompt:
        raise ValueError("critic_rl record requires previous_prompt or rewritten_prompt")
    normalized["previous_prompt"] = previous_prompt
    normalized.setdefault("rewritten_prompt", previous_prompt)
    normalized.setdefault("previous_image_path", normalized["origin_image_path"])
    normalized.setdefault("generated_image_path", normalized["edited_image_path"])
    normalized["ground_truth"] = normalize_ground_truth(normalized)
    return normalized


def normalize_ground_truth(record: Mapping[str, Any]) -> dict[str, Any]:
    raw = record.get("ground_truth", record.get("evaluation", {}))
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"critic_rl ground_truth is not valid JSON: {exc}") from exc
    if not isinstance(raw, Mapping):
        raw = {}

    success = _optional_bool(raw.get("success", raw.get("previous_step_success", record.get("previous_step_success"))))
    if success is None:
        raise ValueError("critic_rl ground_truth requires boolean success or previous_step_success")
    return {
        "success": success,
        "semantics": _optional_float(raw.get("semantics", raw.get("semantic_score")), default=0.0),
        "quality": _optional_float(raw.get("quality", raw.get("quality_score")), default=0.0),
    }


def resolve_interleave_image_path(
    value: str,
    *,
    image_dir: str | os.PathLike[str] = "",
    validate_image_files: bool = False,
) -> str:
    expanded = os.path.expanduser(value)
    if not image_dir or os.path.isabs(expanded) or looks_like_uri(expanded):
        resolved = expanded
    else:
        resolved = str(Path(os.path.expanduser(str(image_dir))) / expanded)
    validate_image_path(resolved, validate_exists=validate_image_files)
    return resolved


def validate_image_path(
    value: str,
    *,
    validate_exists: bool = False,
) -> None:
    if looks_like_uri(value):
        return
    suffix = Path(value).suffix.lower()
    if suffix not in IMAGE_EXTENSIONS:
        raise ValueError(f"Unsupported image file extension for InterleaveThinker path {value!r}")
    if validate_exists and not Path(value).is_file():
        raise FileNotFoundError(f"InterleaveThinker image path does not exist: {value}")


def looks_like_uri(value: str) -> bool:
    return "://" in value or value.startswith("data:")


def _resolve_dataset_files(
    data_path: str | os.PathLike[str],
    kind: InterleaveDatasetKind,
) -> list[Path]:
    path = Path(os.path.expanduser(str(data_path)))
    if path.is_dir():
        path = path / DEFAULT_FILENAMES[kind]
    if not path.exists():
        raise FileNotFoundError(f"InterleaveThinker {kind} data file not found: {path}")
    if path.suffix not in {".json", ".jsonl"}:
        raise ValueError(f"Unsupported InterleaveThinker data file: {path}")
    return [path]


def _load_json_records(file_path: Path) -> list[dict[str, Any]]:
    try:
        if file_path.suffix == ".jsonl":
            records: list[dict[str, Any]] = []
            for line_number, line in enumerate(file_path.read_text(encoding="utf-8").splitlines(), start=1):
                if not line.strip():
                    continue
                raw_line = json.loads(line)
                if not isinstance(raw_line, Mapping):
                    raise ValueError(f"{file_path}:{line_number} must contain a JSON object")
                records.append(dict(raw_line))
            return records

        raw = json.loads(file_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Malformed JSON in InterleaveThinker data file {file_path}: {exc}") from exc

    if isinstance(raw, list):
        if not all(isinstance(item, Mapping) for item in raw):
            raise ValueError(f"{file_path} must contain only JSON objects")
        return [dict(item) for item in raw]
    if isinstance(raw, Mapping):
        for key in ("data", "records"):
            value = raw.get(key)
            if isinstance(value, list):
                if not all(isinstance(item, Mapping) for item in value):
                    raise ValueError(f"{file_path}.{key} must contain only JSON objects")
                return [dict(item) for item in value]
        return [dict(raw)]
    raise ValueError(f"{file_path} must contain a JSON object or list of objects")


def _resolve_record_image_paths(
    record: Mapping[str, Any],
    *,
    image_dir: str | os.PathLike[str],
    validate_image_files: bool,
) -> dict[str, Any]:
    normalized = dict(record)
    for key in IMAGE_PATH_KEYS:
        value = normalized.get(key)
        if isinstance(value, str) and value:
            normalized[key] = resolve_interleave_image_path(
                value,
                image_dir=image_dir,
                validate_image_files=validate_image_files,
            )
    for key in IMAGE_LIST_KEYS:
        value = normalized.get(key)
        if isinstance(value, Sequence) and not isinstance(value, str | bytes):
            normalized[key] = [
                resolve_interleave_image_path(
                    item,
                    image_dir=image_dir,
                    validate_image_files=validate_image_files,
                ) if isinstance(item, str) and item else item for item in value
            ]
    return normalized


def _require_messages(record: Mapping[str, Any], *, where: str) -> list[dict[str, Any]]:
    raw = record.get("messages")
    if not isinstance(raw, Sequence) or isinstance(raw, str | bytes):
        raise ValueError(f"{where} record requires ShareGPT-style messages")
    messages = [dict(item) for item in raw if isinstance(item, Mapping)]
    if len(messages) != len(raw):
        raise ValueError(f"{where} messages must be JSON objects")
    return messages


def _first_message_content(messages: Sequence[Mapping[str, Any]], role: str) -> str:
    for message in messages:
        if message.get("role") == role:
            content = message.get("content")
            if isinstance(content, str):
                return content
    return ""


def _require_text(record: Mapping[str, Any], key: str, where: str) -> str:
    value = record.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{where} record requires {key}")
    return value


def _string_sequence(value: Any) -> list[str]:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes):
        return []
    return [item for item in value if isinstance(item, str)]


def _first_text(*values: Any) -> str:
    for value in values:
        if isinstance(value, str) and value:
            return value
    return ""


def _optional_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    return None


def _optional_float(value: Any, *, default: float) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


__all__ = [
    "DEFAULT_FILENAMES",
    "IMAGE_EXTENSIONS",
    "IMAGE_LIST_KEYS",
    "IMAGE_PATH_KEYS",
    "InterleaveDatasetKind",
    "load_critic_rl_records",
    "load_critic_sft_records",
    "load_interleave_dataset",
    "load_planner_rl_records",
    "load_planner_sft_records",
    "looks_like_uri",
    "normalize_critic_rl_record",
    "normalize_critic_sft_record",
    "normalize_ground_truth",
    "normalize_interleave_dataset_record",
    "normalize_planner_rl_record",
    "normalize_planner_sft_record",
    "resolve_interleave_image_path",
    "validate_image_path",
]
