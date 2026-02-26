# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, TYPE_CHECKING
from typing import Protocol

from fastvideo.distillation.methods.base import DistillMethod
from fastvideo.distillation.models.components import ModelComponents
from fastvideo.distillation.utils.config import DistillRunConfig

if TYPE_CHECKING:
    from fastvideo.fastvideo_args import TrainingArgs


@dataclass(slots=True)
class DistillRuntime:
    """Fully assembled runtime for `DistillTrainer.run()`."""

    training_args: TrainingArgs
    method: DistillMethod
    dataloader: Any
    start_step: int = 0


class ModelBuilder(Protocol):
    def __call__(self, *, cfg: DistillRunConfig) -> ModelComponents:
        ...


_MODELS: dict[str, ModelBuilder] = {}
_METHODS: dict[str, type[DistillMethod]] = {}
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


def register_method(
    name: str,
) -> Callable[[type[DistillMethod]], type[DistillMethod]]:
    name = str(name).strip()
    if not name:
        raise ValueError("method name cannot be empty")

    def decorator(method_cls: type[DistillMethod]) -> type[DistillMethod]:
        if name in _METHODS:
            raise KeyError(f"Method already registered: {name!r}")
        if not issubclass(method_cls, DistillMethod):
            raise TypeError(f"Registered method must subclass DistillMethod: {method_cls}")
        _METHODS[name] = method_cls
        return method_cls

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


def get_method(name: str) -> type[DistillMethod]:
    ensure_builtin_registrations()
    if name not in _METHODS:
        raise KeyError(f"Unknown method {name!r}. Available: {available_methods()}")
    return _METHODS[name]


def build_runtime_from_config(cfg: DistillRunConfig) -> DistillRuntime:
    """Build a distillation runtime from a YAML config.

    Assembles:
    - model components (bundle/adapter/dataloader/validator)
    - method implementation (algorithm) on top of those components
    """

    model_builder = get_model(str(cfg.recipe.family))
    components = model_builder(cfg=cfg)

    method_cls = get_method(str(cfg.recipe.method))
    method = method_cls.build(
        cfg=cfg,
        bundle=components.bundle,
        adapter=components.adapter,
        validator=components.validator,
    )

    return DistillRuntime(
        training_args=components.training_args,
        method=method,
        dataloader=components.dataloader,
        start_step=int(getattr(components, "start_step", 0) or 0),
    )
