# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol

from fastvideo.distillation.bundle import ModelBundle
from fastvideo.distillation.methods.base import DistillMethod
from fastvideo.distillation.runtime import FamilyArtifacts
from fastvideo.distillation.yaml_config import DistillRunConfig


class FamilyBuilder(Protocol):
    def __call__(self, *, cfg: DistillRunConfig) -> FamilyArtifacts:
        ...


class MethodBuilder(Protocol):
    def __call__(
        self,
        *,
        cfg: DistillRunConfig,
        bundle: ModelBundle,
        adapter: Any,
        validator: Any | None,
    ) -> DistillMethod:
        ...


_FAMILIES: dict[str, FamilyBuilder] = {}
_METHODS: dict[str, MethodBuilder] = {}
_BUILTINS_REGISTERED = False


def register_family(name: str) -> Callable[[FamilyBuilder], FamilyBuilder]:
    name = str(name).strip()
    if not name:
        raise ValueError("family name cannot be empty")

    def decorator(builder: FamilyBuilder) -> FamilyBuilder:
        if name in _FAMILIES:
            raise KeyError(f"Family already registered: {name!r}")
        _FAMILIES[name] = builder
        return builder

    return decorator


def register_method(name: str) -> Callable[[MethodBuilder], MethodBuilder]:
    name = str(name).strip()
    if not name:
        raise ValueError("method name cannot be empty")

    def decorator(builder: MethodBuilder) -> MethodBuilder:
        if name in _METHODS:
            raise KeyError(f"Method already registered: {name!r}")
        _METHODS[name] = builder
        return builder

    return decorator


def ensure_builtin_registrations() -> None:
    global _BUILTINS_REGISTERED
    if _BUILTINS_REGISTERED:
        return

    # NOTE: keep these imports explicit (no wildcard scanning) so registration
    # order is stable and failures are debuggable.
    from fastvideo.distillation.families import wan as _wan  # noqa: F401
    from fastvideo.distillation.methods.distribution_matching import dmd2 as _dmd2  # noqa: F401

    _BUILTINS_REGISTERED = True


def available_families() -> list[str]:
    return sorted(_FAMILIES.keys())


def available_methods() -> list[str]:
    return sorted(_METHODS.keys())


def get_family(name: str) -> FamilyBuilder:
    ensure_builtin_registrations()
    if name not in _FAMILIES:
        raise KeyError(f"Unknown family {name!r}. Available: {available_families()}")
    return _FAMILIES[name]


def get_method(name: str) -> MethodBuilder:
    ensure_builtin_registrations()
    if name not in _METHODS:
        raise KeyError(f"Unknown method {name!r}. Available: {available_methods()}")
    return _METHODS[name]
