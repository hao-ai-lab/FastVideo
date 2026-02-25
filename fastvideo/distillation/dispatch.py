# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol

from fastvideo.distillation.methods.base import DistillMethod
from fastvideo.distillation.models.components import ModelComponents
from fastvideo.distillation.roles import RoleManager
from fastvideo.distillation.utils.config import DistillRunConfig, DistillRuntime


class ModelBuilder(Protocol):
    def __call__(self, *, cfg: DistillRunConfig) -> ModelComponents:
        ...


class MethodBuilder(Protocol):
    def __call__(
        self,
        *,
        cfg: DistillRunConfig,
        bundle: RoleManager,
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


def build_runtime_from_config(cfg: DistillRunConfig) -> DistillRuntime:
    """Build a distillation runtime from a YAML config.

    Assembles:
    - model components (bundle/adapter/dataloader/tracker/validator)
    - method implementation (algorithm) on top of those components
    """

    model_builder = get_model(str(cfg.recipe.family))
    components = model_builder(cfg=cfg)

    method_builder = get_method(str(cfg.recipe.method))
    method = method_builder(
        cfg=cfg,
        bundle=components.bundle,
        adapter=components.adapter,
        validator=components.validator,
    )

    return DistillRuntime(
        training_args=components.training_args,
        method=method,
        dataloader=components.dataloader,
        tracker=components.tracker,
        start_step=int(getattr(components, "start_step", 0) or 0),
    )

