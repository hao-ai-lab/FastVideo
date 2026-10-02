# SPDX-License-Identifier: Apache-2.0
"""Ordered, traceable resolution of a typed config into a frozen result.

Resolution turns the user's raw typed config (``GeneratorConfig``) into the
final values that runtime code reads. It follows three rules:

* The raw input is kept as it was given. Explicit paths, the dotted leaf
  paths that the user wrote, are recorded from the raw mapping when
  resolution starts, so "the user set this" is never inferred from defaults.
* Resolution runs an ordered list of steps. A step is a plain function that
  receives a read-only :class:`ResolutionView` and returns
  ``{dotted_path: value}``. A step never assigns fields. The driver records
  each returned mapping as a decision ``(source, values)``, where ``source``
  is the qualified name of the step; the last decision for a path wins.
* The result is frozen. :class:`ResolvedGeneratorConfig` mirrors the nested
  structure of ``GeneratorConfig`` (``resolved.engine.offload.dit``), rejects
  attribute assignment at every level, and reports the provenance of every
  path. The only way to change a resolved config is
  :meth:`ResolvedGeneratorConfig.with_override`, which validates the paths,
  returns a separate frozen result, and appends ``(source, values)`` to an
  override log.

Paths address dataclass fields with dots (``engine.parallelism.sp_size``).
Dict-valued fields such as ``pipeline.experimental`` accept keys that the
input does not have (``pipeline.experimental.attention_backend``). A path
through a plain value, or through an optional nested config that is
``None``, is rejected.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
import dataclasses
from types import MappingProxyType
from typing import Any

from fastvideo.api.parser import parse_config
from fastvideo.api.request_metadata import _record_value_paths
from fastvideo.api.schema import GeneratorConfig

# Source name reported for a path that no step or override decided.
INPUT_SOURCE = "input"

ResolutionStep = Callable[["ResolutionView"], Mapping[str, Any]]

# One recorded change: (step or override source, {dotted_path: value}).
Decision = tuple[str, dict[str, Any]]


class ResolutionError(ValueError):
    """A step or override named a path that the config does not have."""


@dataclasses.dataclass(frozen=True)
class PathProvenance:
    """Where the final value of one path came from.

    ``source`` is the qualified name of the step, or the source name of the
    override, that last set the path or an enclosing path; it is
    :data:`INPUT_SOURCE` when nothing changed the input value. ``raw_value``
    is the parsed input value including schema defaults, and ``raw_present``
    is false for dict keys that only resolution added.
    """

    path: str
    value: Any
    source: str
    raw_value: Any
    raw_present: bool
    explicit: bool


class _Struct:
    """A nested config dataclass, held as its class and a dict of field values."""

    __slots__ = ("config_class", "fields")

    def __init__(self, config_class: type[Any], fields: dict[str, Any]):
        self.config_class = config_class
        self.fields = fields

    def __getstate__(self) -> tuple[type[Any], dict[str, Any]]:
        return self.config_class, self.fields

    def __setstate__(self, state: tuple[type[Any], dict[str, Any]]) -> None:
        self.config_class, self.fields = state


def _is_dataclass_instance(value: Any) -> bool:
    return dataclasses.is_dataclass(value) and not isinstance(value, type)


def _to_tree(value: Any) -> Any:
    """Copy a value into the resolution tree.

    Dataclasses become ``_Struct`` nodes, and read-only mappings that a step
    took from :meth:`ResolutionView.get` become dicts again, so the tree only
    holds plain, picklable containers.
    """
    if _is_dataclass_instance(value):
        return _Struct(type(value),
                       {field.name: _to_tree(getattr(value, field.name))
                        for field in dataclasses.fields(value)})
    if isinstance(value, dict | MappingProxyType):
        return {key: _to_tree(child) for key, child in value.items()}
    if isinstance(value, list):
        return [_to_tree(child) for child in value]
    if isinstance(value, tuple):
        return tuple(_to_tree(child) for child in value)
    return deepcopy(value)


def _to_plain(node: Any) -> Any:
    """Copy a resolution tree into plain nested dicts, lists, and values."""
    if isinstance(node, _Struct):
        return {name: _to_plain(child) for name, child in node.fields.items()}
    if isinstance(node, dict):
        return {key: _to_plain(child) for key, child in node.items()}
    if isinstance(node, list):
        return [_to_plain(child) for child in node]
    return deepcopy(node)


def _to_config(node: Any) -> Any:
    """Rebuild typed config dataclasses from a resolution tree."""
    if isinstance(node, _Struct):
        return node.config_class(**{name: _to_config(child) for name, child in node.fields.items()})
    return deepcopy(node)


def _freeze(value: Any) -> Any:
    """Read-only copy of a leaf value: dicts become mapping proxies, lists and sets become tuples and frozensets."""
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze(child) for key, child in value.items()})
    if isinstance(value, list | tuple):
        return tuple(_freeze(child) for child in value)
    if isinstance(value, set):
        return frozenset(_freeze(child) for child in value)
    return deepcopy(value)


def _split(path: str) -> list[str] | None:
    """Dotted path parts, or ``None`` when ``path`` is not a non-empty dotted string."""
    if not isinstance(path, str) or not path or any(not part for part in path.split(".")):
        return None
    return path.split(".")


def _lookup(tree: _Struct, path: str) -> tuple[bool, Any]:
    """Return ``(present, value)`` for a dotted path, walking structure fields and dict keys."""
    parts = _split(path)
    if parts is None:
        return False, None
    node: Any = tree
    for part in parts:
        if isinstance(node, _Struct) and part in node.fields:
            node = node.fields[part]
        elif isinstance(node, dict) and part in node:
            node = node[part]
        else:
            return False, None
    return True, node


def _assign(tree: _Struct, path: str, value: Any, source: str) -> None:
    """Set one dotted path in the tree, rejecting paths that the config does not have.

    Intermediate parts must name existing structure fields or existing dict
    keys. The last part must name an existing structure field, or any key of
    a dict-valued field. A field that holds a nested structure accepts only a
    dataclass instance; an optional nested config that is ``None`` accepts a
    dataclass instance to install it.
    """
    parts = _split(path)
    if parts is None:
        raise ResolutionError(f"{source}: {path!r} is not a dotted config path")
    node: Any = tree
    for depth, part in enumerate(parts[:-1]):
        if isinstance(node, _Struct) and part in node.fields:
            node = node.fields[part]
        elif isinstance(node, dict) and part in node:
            node = node[part]
        else:
            walked = ".".join(parts[:depth + 1])
            raise ResolutionError(f"{source}: {path!r} goes through {walked!r}, which is not a config path")
    last = parts[-1]
    if isinstance(node, dict):
        node[last] = _to_tree(value)
        return
    if not isinstance(node, _Struct):
        parent = ".".join(parts[:-1])
        raise ResolutionError(f"{source}: {path!r} goes through {parent!r}, which holds a plain value")
    if last not in node.fields:
        raise ResolutionError(f"{source}: {path!r} is not a field of {node.config_class.__name__}")
    if isinstance(node.fields[last], _Struct) and not _is_dataclass_instance(value):
        raise ResolutionError(f"{source}: {path!r} is a nested config; set its fields or pass a dataclass instance")
    node.fields[last] = _to_tree(value)


def _apply(tree: _Struct, source: str, values: Mapping[str, Any]) -> Decision:
    """Validate one decision on a copy of the tree, then apply it, so a rejected path leaves ``tree`` unchanged."""
    if not isinstance(values, Mapping):
        raise ResolutionError(f"{source}: expected a mapping of config paths, got {type(values).__name__}")
    trial = deepcopy(tree)
    for path, value in values.items():
        _assign(trial, path, value, source)
    for path, value in values.items():
        _assign(tree, path, value, source)
    return source, {path: _to_plain(_to_tree(value)) for path, value in values.items()}


def _is_explicit(explicit_paths: frozenset[str], path: str) -> bool:
    """A path is explicit when the user wrote it or wrote a path inside it."""
    return path in explicit_paths or any(written.startswith(path + ".") for written in explicit_paths)


def _last_source(log: Sequence[Decision], path: str) -> str:
    """Source of the latest log entry that set ``path`` or an enclosing path."""
    for source, values in reversed(log):
        for decided in values:
            if path == decided or path.startswith(decided + "."):
                return source
    return INPUT_SOURCE


class ResolutionView:
    """Read-only view that resolution steps receive.

    Reads return the latest decision for a path, else the input value.
    Nested structures come back as read-only nodes and container values as
    read-only copies, so a step can change the config only by returning a
    decision.
    """

    __slots__ = ("_tree", "_explicit_paths", "_decisions")

    def __init__(self, tree: _Struct, explicit_paths: frozenset[str], decisions: list[Decision]):
        self._tree = tree
        self._explicit_paths = explicit_paths
        self._decisions = decisions

    def get(self, path: str) -> Any:
        present, value = _lookup(self._tree, path)
        if not present:
            raise ResolutionError(f"{path!r} is not a path of this config")
        return _FrozenNode(deepcopy(value)) if isinstance(value, _Struct) else _freeze(value)

    def is_explicit(self, path: str) -> bool:
        return _is_explicit(self._explicit_paths, path)

    def decided_by(self, path: str) -> str:
        return _last_source(self._decisions, path)


class _FrozenNode:
    """Read-only attribute access to one nested structure of a resolved config."""

    __slots__ = ("_struct", )

    def __init__(self, struct: _Struct):
        object.__setattr__(self, "_struct", struct)

    def __getattr__(self, name: str) -> Any:
        struct = object.__getattribute__(self, "_struct")
        if name not in struct.fields:
            raise AttributeError(f"{struct.config_class.__name__} has no field {name!r}")
        child = struct.fields[name]
        return _FrozenNode(child) if isinstance(child, _Struct) else _freeze(child)

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError(f"resolved config is read-only; cannot set {name!r}")

    def __delattr__(self, name: str) -> None:
        raise AttributeError(f"resolved config is read-only; cannot delete {name!r}")

    def __repr__(self) -> str:
        struct = object.__getattribute__(self, "_struct")
        return f"{struct.config_class.__name__}({_to_plain(struct)!r})"


class _ResolvedTree(_FrozenNode):
    """Frozen resolution result of any typed config, with per-path provenance and an override log.

    Fields are read by attribute with the same nesting as the typed config.
    The methods below are reserved names and take precedence over top-level
    fields of the same name.
    """

    __slots__ = ("_raw_tree", "_explicit_paths", "_decisions", "_override_log")

    def __init__(
        self,
        tree: _Struct,
        raw_tree: _Struct,
        explicit_paths: frozenset[str],
        decisions: Sequence[Decision],
        override_log: Sequence[Decision],
    ):
        super().__init__(tree)
        object.__setattr__(self, "_raw_tree", raw_tree)
        object.__setattr__(self, "_explicit_paths", explicit_paths)
        object.__setattr__(self, "_decisions", tuple(decisions))
        object.__setattr__(self, "_override_log", tuple(override_log))

    def __reduce__(self) -> tuple[Any, ...]:
        state = object.__getattribute__
        return type(self), (state(self, "_struct"), state(self, "_raw_tree"), state(self, "_explicit_paths"),
                            state(self, "_decisions"), state(self, "_override_log"))

    @property
    def decisions(self) -> tuple[Decision, ...]:
        """Resolution decisions in the order that the steps made them."""
        return deepcopy(object.__getattribute__(self, "_decisions"))

    @property
    def override_log(self) -> tuple[Decision, ...]:
        """Changes applied after resolution by :meth:`with_override`, in order."""
        return deepcopy(object.__getattribute__(self, "_override_log"))

    def is_explicit(self, path: str) -> bool:
        return _is_explicit(object.__getattribute__(self, "_explicit_paths"), path)

    def provenance(self, path: str) -> PathProvenance:
        """Final value, deciding source, input value, and explicit flag of one path."""
        present, value = _lookup(object.__getattribute__(self, "_struct"), path)
        if not present:
            raise ResolutionError(f"{path!r} is not a path of this config")
        raw_present, raw_value = _lookup(object.__getattribute__(self, "_raw_tree"), path)
        log = object.__getattribute__(self, "_decisions") + object.__getattribute__(self, "_override_log")
        return PathProvenance(
            path=path,
            value=_to_plain(value),
            source=_last_source(log, path),
            raw_value=_to_plain(raw_value),
            raw_present=raw_present,
            explicit=self.is_explicit(path),
        )

    def provenance_table(self) -> list[PathProvenance]:
        """Provenance of every leaf field and of every path that a step or override set, sorted by path."""
        paths: set[str] = set()

        def walk(node: _Struct, prefix: str) -> None:
            for name, child in node.fields.items():
                path = f"{prefix}.{name}" if prefix else name
                if isinstance(child, _Struct):
                    walk(child, path)
                else:
                    paths.add(path)

        walk(object.__getattribute__(self, "_struct"), "")
        for _, values in object.__getattribute__(self, "_decisions") + object.__getattribute__(self, "_override_log"):
            paths.update(values)
        return [self.provenance(path) for path in sorted(paths)]

    def with_override(self, source: str, values: Mapping[str, Any]) -> Any:
        """Return a separate frozen result with ``values`` applied and ``(source, values)`` added to the override log.

        Use this for decisions that can only be made after resolution, such
        as ones that depend on the device that a worker binds. Paths are
        validated exactly as for resolution steps; this object is unchanged.
        """
        if not isinstance(source, str) or not source:
            raise ResolutionError("an override needs a non-empty source name")
        tree = deepcopy(object.__getattribute__(self, "_struct"))
        override = _apply(tree, source, values)
        return type(self)(
            tree,
            object.__getattribute__(self, "_raw_tree"),
            object.__getattribute__(self, "_explicit_paths"),
            object.__getattribute__(self, "_decisions"),
            object.__getattribute__(self, "_override_log") + (override, ),
        )

    def to_dict(self) -> dict[str, Any]:
        """Plain nested dict copy of the resolved values."""
        return _to_plain(object.__getattribute__(self, "_struct"))


class ResolvedGeneratorConfig(_ResolvedTree):
    """Frozen result of resolving a ``GeneratorConfig``, with the provenance of every path."""

    __slots__ = ()

    def with_override(self, source: str, values: Mapping[str, Any]) -> ResolvedGeneratorConfig:
        return super().with_override(source, values)

    def to_config(self) -> GeneratorConfig:
        """Typed ``GeneratorConfig`` copy of the resolved values, for code that consumes the typed config."""
        return _to_config(object.__getattribute__(self, "_struct"))


def _run_steps(typed_config: Any, explicit_paths: frozenset[str],
               steps: Sequence[ResolutionStep]) -> tuple[_Struct, _Struct, list[Decision]]:
    """Run ``steps`` in order over a parsed typed config; return the resolved tree, the input tree, and decisions."""
    raw_tree = _to_tree(typed_config)
    tree = deepcopy(raw_tree)
    decisions: list[Decision] = []
    view = ResolutionView(tree, explicit_paths, decisions)
    for step in steps:
        source = getattr(step, "__qualname__", repr(step))
        values = step(view)
        if values:
            decisions.append(_apply(tree, source, values))
    return tree, raw_tree, decisions


def resolve_generator_config(raw: Mapping[str, Any], steps: Sequence[ResolutionStep]) -> ResolvedGeneratorConfig:
    """Parse a raw generator config mapping, run ``steps`` in order, and freeze the result.

    ``raw`` is the merged YAML, CLI-override, and keyword input in the nested
    shape of ``GeneratorConfig``. Every leaf written in ``raw`` is an
    explicit path. Each step sees the decisions of the steps before it.
    """
    if not isinstance(raw, Mapping):
        raise TypeError(f"expected a raw config mapping, got {type(raw).__name__}")
    explicit_paths: set[str] = set()
    _record_value_paths(raw, "", explicit_paths)
    tree, raw_tree, decisions = _run_steps(parse_config(GeneratorConfig, raw), frozenset(explicit_paths), steps)
    return ResolvedGeneratorConfig(tree, raw_tree, frozenset(explicit_paths), decisions, ())


__all__ = [
    "INPUT_SOURCE",
    "PathProvenance",
    "ResolutionError",
    "ResolutionStep",
    "ResolutionView",
    "ResolvedGeneratorConfig",
    "resolve_generator_config",
]
