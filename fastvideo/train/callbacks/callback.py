# SPDX-License-Identifier: Apache-2.0
"""Callback base class and CallbackDict manager.

Adapted from FastGen's callback pattern to FastVideo's types.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TYPE_CHECKING

from fastvideo.logger import init_logger
from fastvideo.train.utils.instantiate import instantiate

if TYPE_CHECKING:
    from fastvideo.train.methods.base import TrainingMethod
    from fastvideo.train.utils.training_config import (
        TrainingConfig, )

logger = init_logger(__name__)

# Well-known callback names that don't need ``_target_`` in YAML.
_BUILTIN_CALLBACKS: dict[str, str] = {
    "grad_clip": "fastvideo.train.callbacks.grad_clip.GradNormClipCallback",
    "validation": "fastvideo.train.callbacks.validation.ValidationCallback",
    "ema": "fastvideo.train.callbacks.ema.EMACallback",
}


class Callback:
    """Base callback with no-op hooks.

    Subclasses override whichever hooks they need.  The
    ``training_config`` and ``method`` attributes are set by
    ``CallbackDict`` after instantiation.
    """

    training_config: TrainingConfig
    method: TrainingMethod

    def on_train_start(
        self,
        method: TrainingMethod,
        iteration: int = 0,
    ) -> None:
        pass

    def on_training_step_end(
        self,
        method: TrainingMethod,
        loss_dict: dict[str, Any],
        iteration: int = 0,
    ) -> None:
        pass

    def on_before_optimizer_step(
        self,
        method: TrainingMethod,
        iteration: int = 0,
    ) -> None:
        pass

    def on_validation_begin(
        self,
        method: TrainingMethod,
        iteration: int = 0,
    ) -> None:
        pass

    def on_validation_end(
        self,
        method: TrainingMethod,
        iteration: int = 0,
    ) -> None:
        pass

    def on_train_end(
        self,
        method: TrainingMethod,
        iteration: int = 0,
    ) -> None:
        pass

    def state_dict(self) -> dict[str, Any]:
        return {}

    def load_state_dict(
        self, state_dict: dict[str, Any],
    ) -> None:
        pass


class CallbackDict:
    """Manages a collection of named callbacks.

    Instantiates each callback from its ``_target_`` config and
    dispatches hook calls to all registered callbacks.
    """

    def __init__(
        self,
        callback_configs: dict[str, dict[str, Any]],
        training_config: TrainingConfig,
    ) -> None:
        self._callbacks: dict[str, Callback] = {}
        if not callback_configs:
            return
        for name, cb_cfg in callback_configs.items():
            cb_cfg = dict(cb_cfg)
            if "_target_" not in cb_cfg:
                if name in _BUILTIN_CALLBACKS:
                    cb_cfg["_target_"] = (
                        _BUILTIN_CALLBACKS[name]
                    )
                else:
                    logger.warning(
                        "Callback %r is missing "
                        "'_target_', skipping: %s",
                        name,
                        cb_cfg,
                    )
                    continue
            logger.info(
                "Instantiating callback %r: %s",
                name,
                cb_cfg,
            )
            cb = instantiate(cb_cfg)
            if not isinstance(cb, Callback):
                raise TypeError(
                    f"Callback {name!r} resolved to "
                    f"{type(cb).__name__}, expected a "
                    f"Callback subclass."
                )
            cb.training_config = training_config
            self._callbacks[name] = cb

    def __getattr__(
        self, method_name: str,
    ) -> Callable[..., Any]:
        if method_name.startswith("_"):
            raise AttributeError(method_name)

        if method_name == "state_dict":

            def _state_dict() -> dict[str, Any]:
                return {
                    n: cb.state_dict()
                    for n, cb in self._callbacks.items()
                }

            return _state_dict

        if method_name == "load_state_dict":

            def _load_state_dict(
                state_dict: dict[str, Any],
            ) -> None:
                for n, cb in self._callbacks.items():
                    if n in state_dict:
                        cb.load_state_dict(state_dict[n])
                    else:
                        logger.warning(
                            "Callback %r not found in "
                            "checkpoint.",
                            n,
                        )

            return _load_state_dict

        def _dispatch(*args: Any, **kwargs: Any) -> None:
            for cb in self._callbacks.values():
                fn = getattr(cb, method_name, None)
                if fn is None:
                    continue
                if not callable(fn):
                    continue
                fn(*args, **kwargs)

        return _dispatch
