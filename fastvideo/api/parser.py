# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import dataclasses
import json
import types
from pathlib import Path
from typing import Any, Literal, Mapping, TypeVar, Union, get_args, get_origin, get_type_hints

import yaml

from fastvideo.api.documents import RunConfig, ServeConfig
from fastvideo.api.errors import DocumentValidationError
from fastvideo.api.overrides import apply_overrides, parse_cli_overrides

T = TypeVar("T")


def parse_document(document_type: type[T], raw: Mapping[str, Any] | T) -> T:
    """Parse a nested mapping into a typed inference document."""
    if isinstance(raw, document_type):
        return raw
    return _parse_dataclass(document_type, raw, "")


def document_to_dict(document: Any) -> Any:
    """Serialize a typed document into plain Python containers."""
    if dataclasses.is_dataclass(document) and not isinstance(document, type):
        return {
            field.name: document_to_dict(getattr(document, field.name))
            for field in dataclasses.fields(document)
        }
    if isinstance(document, list):
        return [document_to_dict(item) for item in document]
    if isinstance(document, dict):
        return {key: document_to_dict(value) for key, value in document.items()}
    return document


def load_document(
    document_type: type[T],
    path: str | Path,
    overrides: list[str] | Mapping[str, Any] | None = None,
) -> T:
    """Load a typed document from YAML or JSON."""
    raw = load_raw_document(path)
    if overrides:
        parsed_overrides = (
            parse_cli_overrides(overrides)
            if isinstance(overrides, list)
            else dict(overrides)
        )
        raw = apply_overrides(raw, parsed_overrides)
    return parse_document(document_type, raw)


def load_run_config(path: str | Path, overrides: list[str] | Mapping[str, Any] | None = None) -> RunConfig:
    return load_document(RunConfig, path, overrides)


def load_serve_config(path: str | Path, overrides: list[str] | Mapping[str, Any] | None = None) -> ServeConfig:
    return load_document(ServeConfig, path, overrides)


def load_raw_document(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    suffix = config_path.suffix.lower()
    with config_path.open(encoding="utf-8") as handle:
        if suffix in {".yaml", ".yml"}:
            raw = yaml.safe_load(handle)
        elif suffix == ".json":
            raw = json.load(handle)
        else:
            raise ValueError(f"Unsupported config file format: {config_path}")

    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise DocumentValidationError("", f"{config_path} must contain a top-level mapping")
    return dict(raw)


def _parse_dataclass(document_type: type[T], raw: Mapping[str, Any], path: str) -> T:
    if not isinstance(raw, Mapping):
        raise DocumentValidationError(path, f"expected mapping for {document_type.__name__}")

    type_hints = get_type_hints(document_type)
    fields_by_name = {field.name: field for field in dataclasses.fields(document_type)}

    for key in raw:
        if not isinstance(key, str):
            raise DocumentValidationError(path, "expected mapping keys to be strings")
        if key not in fields_by_name:
            raise DocumentValidationError(_join_path(path, key), "unknown field")

    values: dict[str, Any] = {}
    for name, field in fields_by_name.items():
        field_path = _join_path(path, name)
        if name in raw:
            values[name] = _parse_value(type_hints[name], raw[name], field_path)
            continue
        if field.default is not dataclasses.MISSING:
            continue
        if field.default_factory is not dataclasses.MISSING:
            continue
        raise DocumentValidationError(field_path, "missing required field")

    return document_type(**values)


def _parse_value(annotation: Any, value: Any, path: str) -> Any:
    if annotation is Any:
        return value

    origin = get_origin(annotation)
    if origin in {types.UnionType, Union}:
        return _parse_union(annotation, value, path)
    if origin is list:
        return _parse_list(annotation, value, path)
    if origin is dict:
        return _parse_dict(annotation, value, path)
    if origin is tuple:
        return _parse_tuple(annotation, value, path)
    if origin is Literal:
        return _parse_literal(annotation, value, path)

    if dataclasses.is_dataclass(annotation):
        return _parse_dataclass(annotation, value, path)

    if annotation is bool:
        if type(value) is not bool:
            raise DocumentValidationError(path, "expected bool")
        return value
    if annotation is int:
        if not isinstance(value, int) or isinstance(value, bool):
            raise DocumentValidationError(path, "expected int")
        return value
    if annotation is float:
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise DocumentValidationError(path, "expected float")
        return float(value)
    if annotation is str:
        if not isinstance(value, str):
            raise DocumentValidationError(path, "expected str")
        return value

    if isinstance(annotation, type):
        if not isinstance(value, annotation):
            raise DocumentValidationError(path, f"expected {annotation.__name__}")
        return value

    return value


def _parse_union(annotation: Any, value: Any, path: str) -> Any:
    union_args = get_args(annotation)
    if value is None and type(None) in union_args:
        return None

    candidates = [candidate for candidate in union_args if candidate is not type(None)]
    if len(candidates) == 1:
        return _parse_value(candidates[0], value, path)

    errors: list[str] = []
    for candidate in candidates:
        try:
            return _parse_value(candidate, value, path)
        except DocumentValidationError as exc:
            errors.append(exc.message)

    expected = ", ".join(_type_name(candidate) for candidate in candidates)
    detail = errors[0] if errors else f"expected one of ({expected})"
    raise DocumentValidationError(path, detail)


def _parse_list(annotation: Any, value: Any, path: str) -> list[Any]:
    if not isinstance(value, list):
        raise DocumentValidationError(path, "expected list")
    item_type = get_args(annotation)[0] if get_args(annotation) else Any
    return [
        _parse_value(item_type, item, f"{path}[{index}]")
        for index, item in enumerate(value)
    ]


def _parse_dict(annotation: Any, value: Any, path: str) -> dict[Any, Any]:
    if not isinstance(value, Mapping):
        raise DocumentValidationError(path, "expected mapping")

    key_type, value_type = (get_args(annotation) + (Any, Any))[:2]
    parsed: dict[Any, Any] = {}
    for key, item in value.items():
        key_path = _join_path(path, str(key))
        parsed_key = _parse_dict_key(key_type, key, path)
        parsed[parsed_key] = _parse_value(value_type, item, key_path)
    return parsed


def _parse_tuple(annotation: Any, value: Any, path: str) -> tuple[Any, ...]:
    if not isinstance(value, (list, tuple)):
        raise DocumentValidationError(path, "expected tuple")
    item_types = get_args(annotation)
    if len(item_types) == 2 and item_types[1] is Ellipsis:
        return tuple(
            _parse_value(item_types[0], item, f"{path}[{index}]")
            for index, item in enumerate(value)
        )
    if len(value) != len(item_types):
        raise DocumentValidationError(path, f"expected tuple of length {len(item_types)}")
    return tuple(
        _parse_value(item_type, item, f"{path}[{index}]")
        for index, (item_type, item) in enumerate(zip(item_types, value, strict=True))
    )


def _parse_literal(annotation: Any, value: Any, path: str) -> Any:
    allowed = get_args(annotation)
    if value not in allowed:
        raise DocumentValidationError(path, f"expected one of {sorted(allowed)!r}")
    return value


def _parse_dict_key(annotation: Any, value: Any, path: str) -> Any:
    if annotation is Any:
        return value
    if annotation is str:
        if not isinstance(value, str):
            raise DocumentValidationError(path, "expected string dictionary keys")
        return value
    if annotation is int:
        if not isinstance(value, int) or isinstance(value, bool):
            raise DocumentValidationError(path, "expected integer dictionary keys")
        return value
    return value


def _join_path(prefix: str, suffix: str) -> str:
    if not prefix:
        return suffix
    return f"{prefix}.{suffix}"


def _type_name(annotation: Any) -> str:
    origin = get_origin(annotation)
    if origin is not None:
        return str(annotation)
    if hasattr(annotation, "__name__"):
        return annotation.__name__
    return str(annotation)


__all__ = [
    "document_to_dict",
    "load_document",
    "load_raw_document",
    "load_run_config",
    "load_serve_config",
    "parse_document",
]
