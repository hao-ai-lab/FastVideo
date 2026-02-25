# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol

from fastvideo.distillation.roles import ModelBundle
from fastvideo.distillation.methods.base import DistillMethod
from fastvideo.distillation.utils.config import FamilyComponents
from fastvideo.distillation.utils.config import DistillRunConfig


class ModelBuilder(Protocol):
    def __call__(self, *, cfg: DistillRunConfig) -> FamilyComponents:
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


_MODELS: dict[str, ModelBuilder] = {}
_METHODS: dict[str, MethodBuilder] = {}
_BUILTINS_REGISTERED = False


def register_model(name: str) -> Callable[[ModelBuilder], ModelBuilder]:
    name = str(name).strip()
    if not name:
        raise ValueError("model name cannot be empty")

    def decorator(builder: ModelBuilder) -> ModelBuilder:
        if name in _MODELS:
            raise KeyError(f"Model already registered: {name!r}")
        _MODELS[name] = builder
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
    from fastvideo.distillation.models import wan as _wan  # noqa: F401
    from fastvideo.distillation.methods.distribution_matching import dmd2 as _dmd2  # noqa: F401
    from fastvideo.distillation.methods.fine_tuning import finetune as _finetune  # noqa: F401

    _BUILTINS_REGISTERED = True


def available_models() -> list[str]:
    return sorted(_MODELS.keys())


def available_methods() -> list[str]:
    return sorted(_METHODS.keys())


def get_model(name: str) -> ModelBuilder:
    ensure_builtin_registrations()
    if name not in _MODELS:
        raise KeyError(f"Unknown model {name!r}. Available: {available_models()}")
    return _MODELS[name]


def get_method(name: str) -> MethodBuilder:
    ensure_builtin_registrations()
    if name not in _METHODS:
        raise KeyError(f"Unknown method {name!r}. Available: {available_methods()}")
    return _METHODS[name]
