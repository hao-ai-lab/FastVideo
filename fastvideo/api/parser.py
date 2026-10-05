# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import dataclasses
import enum
import json
import types
from pathlib import Path
from collections.abc import Mapping
from typing import Any, Literal, TypeVar, Union, get_args, get_origin, get_type_hints

import yaml

from fastvideo.api.errors import ConfigValidationError
from fastvideo.api.overrides import apply_overrides, normalize_overrides
from fastvideo.api.request_metadata import (
    bind_generation_request_raw,
    bind_run_config_raw,
    bind_serve_config_raw,
)
from fastvideo.api.schema import GenerationRequest, RunConfig, ServeConfig

T = TypeVar("T")
_UNION_ORIGINS = {types.UnionType, Union}


@dataclasses.dataclass(frozen=True)
class _DataclassSpec:
    cls: type[Any]
    type_hints: dict[str, Any]
    fields_by_name: dict[str, dataclasses.Field[Any]]


def parse_config(config_type: type[T], raw: Mapping[str, Any] | T) -> T:
    """Parse a nested mapping into a typed inference config object."""
    if isinstance(raw, config_type):
        return raw
    if not isinstance(raw, Mapping):
        raise ConfigValidationError("", f"expected mapping for {config_type.__name__}")
    parsed = _SchemaParser().parse_dataclass(config_type, raw, "")
    if config_type is GenerationRequest:
        return bind_generation_request_raw(parsed, raw)
    if config_type is RunConfig:
        return bind_run_config_raw(parsed, raw)
    if config_type is ServeConfig:
        return bind_serve_config_raw(parsed, raw)
    return parsed


def config_to_dict(config: Any) -> Any:
    """Serialize a typed config object into plain Python containers.

    A tagged-union member is written in its input form, ``{tag: fields}``, without the ``family`` tag field, so the
    result parses back with :func:`parse_config`.
    """
    if dataclasses.is_dataclass(config) and not isinstance(config, type):
        tag = union_tag(config)
        fields = {
            field.name: config_to_dict(getattr(config, field.name))
            for field in dataclasses.fields(config) if tag is None or field.name != TAG_FIELD
        }
        return fields if tag is None else {tag: fields}
    if isinstance(config, list):
        return [config_to_dict(item) for item in config]
    if isinstance(config, dict):
        return {key: config_to_dict(value) for key, value in config.items()}
    return config


def load_config(
    config_type: type[T],
    path: str | Path,
    overrides: list[str] | Mapping[str, Any] | None = None,
) -> T:
    """Load a typed config object from YAML or JSON."""
    raw = load_raw_config(path)
    normalized_overrides = normalize_overrides(overrides)
    if normalized_overrides:
        raw = apply_overrides(raw, normalized_overrides)
    return parse_config(config_type, raw)


def load_run_config(
    path: str | Path,
    overrides: list[str] | Mapping[str, Any] | None = None,
) -> RunConfig:
    return load_config(RunConfig, path, overrides)


def load_serve_config(
    path: str | Path,
    overrides: list[str] | Mapping[str, Any] | None = None,
) -> ServeConfig:
    return load_config(ServeConfig, path, overrides)


def load_raw_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open(encoding="utf-8") as handle:
        raw = _load_raw_mapping(handle, config_path)

    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise ConfigValidationError("", f"{config_path} must contain a top-level mapping")
    return dict(raw)


def _load_raw_mapping(handle: Any, config_path: Path) -> Any:
    suffix = config_path.suffix.lower()
    if suffix in {".yaml", ".yml"}:
        return yaml.safe_load(handle)
    if suffix == ".json":
        return json.load(handle)
    raise ValueError(f"Unsupported config file format: {config_path}")


# A tagged union is a dataclass that defines ``TAGS`` in its own namespace: a mapping from tag to the dataclass that
# the tag selects. Its input is a mapping with exactly one key, the tag, whose value is parsed as the selected class.
# Every selected class carries the tag in the ``TAG_FIELD`` field, whose default is the tag; the input never writes it.
TAG_FIELD = "family"


def tagged_union_tags(config_type: Any) -> Mapping[str, type[Any]] | None:
    """The ``TAGS`` table of a tagged-union dataclass, or ``None`` for any other annotation.

    Only the class that defines ``TAGS`` is the union; the classes that it selects inherit the attribute and parse as
    plain dataclasses.
    """
    if not isinstance(config_type, type) or not dataclasses.is_dataclass(config_type):
        return None
    return vars(config_type).get("TAGS")


def union_tag(config: Any) -> str | None:
    """The tag under which a tagged-union member instance is written, or ``None`` for any other value."""
    for base in type(config).__mro__:
        tags = tagged_union_tags(base)
        if tags is not None:
            for tag, member_type in tags.items():
                if member_type is type(config):
                    return tag
            raise ConfigValidationError("", f"{type(config).__name__} is not in the TAGS of {base.__name__}")
    return None


def strip_union_tags(config_type: type[Any], paths: set[str]) -> set[str]:
    """Rewrite dotted input paths through a tagged-union field into the field paths of the parsed config.

    An input path ``pipeline.model.ltx2.refine.enabled`` names the member block by its tag; the parsed config holds
    the block at ``pipeline.model``, so the path becomes ``pipeline.model.refine.enabled``. Paths that do not go
    through a tagged-union field of ``config_type`` are unchanged.
    """
    union_paths = _tagged_union_field_paths(config_type, "")
    if not union_paths:
        return paths
    stripped: set[str] = set()
    for path in paths:
        for union_path, tags in union_paths.items():
            if path.startswith(union_path + "."):
                tag, _, rest = path[len(union_path) + 1:].partition(".")
                if tag in tags:
                    path = _join_path(union_path, rest) if rest else union_path
                break
        stripped.add(path)
    return stripped


def _tagged_union_field_paths(config_type: type[Any], prefix: str) -> dict[str, Mapping[str, type[Any]]]:
    """Dotted path -> ``TAGS`` for every tagged-union field reachable through the nested dataclass fields."""
    found: dict[str, Mapping[str, type[Any]]] = {}
    spec = _get_dataclass_spec(config_type)
    for name in spec.fields_by_name:
        field_path = _join_path(prefix, name)
        for candidate in _annotation_classes(spec.type_hints[name]):
            tags = tagged_union_tags(candidate)
            if tags is not None:
                found[field_path] = tags
            elif dataclasses.is_dataclass(candidate):
                found.update(_tagged_union_field_paths(candidate, field_path))
    return found


def _annotation_classes(annotation: Any) -> list[type[Any]]:
    """The classes that an annotation names directly or as members of an optional or union annotation."""
    if get_origin(annotation) in _UNION_ORIGINS:
        return [candidate for candidate in get_args(annotation) if isinstance(candidate, type)]
    return [annotation] if isinstance(annotation, type) else []


class _SchemaParser:

    def parse_dataclass(
        self,
        config_type: type[T],
        raw: Mapping[str, Any],
        path: str,
    ) -> T:
        if not isinstance(raw, Mapping):
            raise ConfigValidationError(path, f"expected mapping for {config_type.__name__}")

        spec = _get_dataclass_spec(config_type)
        self._validate_keys(raw, spec, path)

        values: dict[str, Any] = {}
        for name, field in spec.fields_by_name.items():
            field_path = _join_path(path, name)
            if name in raw:
                values[name] = self.parse_value(spec.type_hints[name], raw[name], field_path)
                continue
            if _field_is_required(field):
                raise ConfigValidationError(field_path, "missing required field")

        return config_type(**values)

    def parse_value(self, annotation: Any, value: Any, path: str) -> Any:
        if annotation is Any:
            return value

        origin = get_origin(annotation)
        if origin in _UNION_ORIGINS:
            return self._parse_union(annotation, value, path)
        if origin is Literal:
            return self._parse_literal(annotation, value, path)
        if origin is list:
            return self._parse_list(annotation, value, path)
        if origin is dict:
            return self._parse_dict(annotation, value, path)
        if origin is tuple:
            return self._parse_tuple(annotation, value, path)
        if isinstance(annotation, type) and dataclasses.is_dataclass(annotation):
            tags = tagged_union_tags(annotation)
            if tags is not None:
                return self._parse_tagged_union(annotation, tags, value, path)
            return self.parse_dataclass(annotation, value, path)
        if isinstance(annotation, type) and issubclass(annotation, enum.Enum):
            return self._parse_enum(annotation, value, path)

        scalar_parser = _SCALAR_PARSERS.get(annotation)
        if scalar_parser is not None:
            return scalar_parser(value, path)

        return self._parse_instance(annotation, value, path)

    def _validate_keys(
        self,
        raw: Mapping[str, Any],
        spec: _DataclassSpec,
        path: str,
    ) -> None:
        for key in raw:
            if not isinstance(key, str):
                raise ConfigValidationError(path, "expected mapping keys to be strings")
            if key not in spec.fields_by_name:
                raise ConfigValidationError(_join_path(path, key), "unknown field")

    def _parse_tagged_union(self, union_type: type[Any], tags: Mapping[str, type[Any]], value: Any, path: str) -> Any:
        """Parse ``{tag: fields}`` into the member class that ``tags[tag]`` names, with ``TAG_FIELD`` set to the tag."""
        if not isinstance(value, Mapping):
            raise ConfigValidationError(path, f"expected mapping with one key for {union_type.__name__}")
        choices = ", ".join(sorted(tags))
        if len(value) != 1:
            raise ConfigValidationError(path, f"expected exactly one key, one of ({choices}); got {sorted(value)!r}")
        tag, fields = next(iter(value.items()))
        if tag not in tags:
            raise ConfigValidationError(_join_path(path, str(tag)), f"unknown key; expected one of ({choices})")
        member_path = _join_path(path, tag)
        if isinstance(fields, Mapping) and TAG_FIELD in fields:
            raise ConfigValidationError(_join_path(member_path, TAG_FIELD), f"set by the key {tag!r}; remove it")
        return self.parse_dataclass(tags[tag], fields, member_path)

    def _parse_union(self, annotation: Any, value: Any, path: str) -> Any:
        candidates = [candidate for candidate in get_args(annotation) if candidate is not type(None)]
        if value is None and len(candidates) != len(get_args(annotation)):
            return None
        if len(candidates) == 1:
            return self.parse_value(candidates[0], value, path)

        errors: list[str] = []
        for candidate in candidates:
            try:
                return self.parse_value(candidate, value, path)
            except ConfigValidationError as exc:
                errors.append(exc.message)

        expected = ", ".join(_type_name(candidate) for candidate in candidates)
        detail = errors[0] if errors else f"expected one of ({expected})"
        raise ConfigValidationError(path, detail)

    def _parse_literal(self, annotation: Any, value: Any, path: str) -> Any:
        allowed = get_args(annotation)
        if value not in allowed:
            raise ConfigValidationError(path, f"expected one of {sorted(allowed)!r}")
        return value

    def _parse_enum(self, annotation: type[enum.Enum], value: Any, path: str) -> enum.Enum:
        """An enum member passes through; any other value must be the value of a member."""
        if isinstance(value, annotation):
            return value
        try:
            return annotation(value)
        except (TypeError, ValueError):
            allowed = [member.value for member in annotation]
            raise ConfigValidationError(path, f"expected one of {allowed!r}") from None

    def _parse_list(self, annotation: Any, value: Any, path: str) -> list[Any]:
        if not isinstance(value, list):
            raise ConfigValidationError(path, "expected list")
        item_type = get_args(annotation)[0] if get_args(annotation) else Any
        return [self.parse_value(item_type, item, f"{path}[{index}]") for index, item in enumerate(value)]

    def _parse_dict(self, annotation: Any, value: Any, path: str) -> dict[Any, Any]:
        if not isinstance(value, Mapping):
            raise ConfigValidationError(path, "expected mapping")

        key_type, value_type = (get_args(annotation) + (Any, Any))[:2]
        parsed: dict[Any, Any] = {}
        for key, item in value.items():
            parsed_key = self._parse_dict_key(key_type, key, path)
            item_path = _join_path(path, str(key))
            parsed[parsed_key] = self.parse_value(value_type, item, item_path)
        return parsed

    def _parse_tuple(self, annotation: Any, value: Any, path: str) -> tuple[Any, ...]:
        if not isinstance(value, list | tuple):
            raise ConfigValidationError(path, "expected tuple")

        item_types = get_args(annotation)
        if len(item_types) == 2 and item_types[1] is Ellipsis:
            return tuple(self.parse_value(item_types[0], item, f"{path}[{index}]") for index, item in enumerate(value))

        if len(value) != len(item_types):
            raise ConfigValidationError(path, f"expected tuple of length {len(item_types)}")

        return tuple(
            self.parse_value(item_type, item, f"{path}[{index}]")
            for index, (item_type, item) in enumerate(zip(item_types, value, strict=True)))

    def _parse_dict_key(self, annotation: Any, value: Any, path: str) -> Any:
        if annotation is Any:
            return value
        if annotation is str:
            if not isinstance(value, str):
                raise ConfigValidationError(path, "expected string dictionary keys")
            return value
        if annotation is int:
            if not isinstance(value, int) or isinstance(value, bool):
                raise ConfigValidationError(path, "expected integer dictionary keys")
            return value
        return value

    def _parse_instance(self, annotation: Any, value: Any, path: str) -> Any:
        if isinstance(annotation, type) and not isinstance(value, annotation):
            raise ConfigValidationError(path, f"expected {annotation.__name__}")
        return value


def _parse_bool(value: Any, path: str) -> bool:
    if type(value) is not bool:
        raise ConfigValidationError(path, "expected bool")
    return value


def _parse_int(value: Any, path: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ConfigValidationError(path, "expected int")
    return value


def _parse_float(value: Any, path: str) -> float:
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise ConfigValidationError(path, "expected float")
    return float(value)


def _parse_str(value: Any, path: str) -> str:
    if not isinstance(value, str):
        raise ConfigValidationError(path, "expected str")
    return value


_SCALAR_PARSERS: dict[Any, Any] = {
    bool: _parse_bool,
    int: _parse_int,
    float: _parse_float,
    str: _parse_str,
}


def _field_is_required(field: dataclasses.Field[Any]) -> bool:
    return (field.default is dataclasses.MISSING and field.default_factory is dataclasses.MISSING)


def _get_dataclass_spec(config_type: type[Any]) -> _DataclassSpec:
    spec = _DATACLASS_SPEC_CACHE.get(config_type)
    if spec is not None:
        return spec

    spec = _DataclassSpec(
        cls=config_type,
        type_hints=get_type_hints(config_type),
        fields_by_name={field.name: field
                        for field in dataclasses.fields(config_type)},
    )
    _DATACLASS_SPEC_CACHE[config_type] = spec
    return spec


_DATACLASS_SPEC_CACHE: dict[type[Any], _DataclassSpec] = {}


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
    "TAG_FIELD",
    "config_to_dict",
    "load_config",
    "load_raw_config",
    "load_run_config",
    "load_serve_config",
    "parse_config",
    "strip_union_tags",
    "tagged_union_tags",
    "union_tag",
]
