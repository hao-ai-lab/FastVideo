# SPDX-License-Identifier: Apache-2.0
"""Deferred loading and release of heavy pipeline modules.

A pipeline normally materializes every component before the first stage runs,
so peak memory is the sum of all components even though no two of them are
needed at the same moment. On a unified-memory device that sum is charged
against the same pool the activations come from, and a model whose components
individually fit can still fail to load.

``LazyModule`` turns that sum into a maximum. It stands in for a component,
loads it on first use, and drops it once the last stage holding it has run.
The pipeline decides when to release; this module only owns the proxying and
the load/free mechanics.
"""

from __future__ import annotations

import functools
import gc
import inspect
from collections.abc import Callable
from typing import Any, TypeGuard

import torch

from fastvideo.logger import init_logger

logger = init_logger(__name__)


def _cuda_allocated_gib() -> float | None:
    if not torch.cuda.is_available():
        return None
    return torch.cuda.memory_allocated() / 1024**3


class LazyModule:
    """A stand-in for a pipeline component that loads on first use.

    Every attribute access, call, and ``isinstance`` check forwards to the real
    component, materializing it if needed. ``release`` drops the reference and
    frees the allocator cache; a later access re-runs the loader, so releasing
    early is a latency cost, never a correctness one.
    """

    __slots__ = ("_lazy_name", "_lazy_loader", "_lazy_materialize_transform", "_lazy_module", "_lazy_release_callbacks")

    def __init__(self, name: str, loader: Callable[[], Any]) -> None:
        object.__setattr__(self, "_lazy_name", name)
        object.__setattr__(self, "_lazy_loader", loader)
        object.__setattr__(self, "_lazy_materialize_transform", None)
        object.__setattr__(self, "_lazy_module", None)
        object.__setattr__(self, "_lazy_release_callbacks", [])

    @property
    def lazy_name(self) -> str:
        return object.__getattribute__(self, "_lazy_name")

    @property
    def is_materialized(self) -> bool:
        return object.__getattribute__(self, "_lazy_module") is not None

    def materialize(self) -> Any:
        """Return the real component, loading it if this is the first use."""
        module = object.__getattribute__(self, "_lazy_module")
        if module is not None:
            return module

        name = object.__getattribute__(self, "_lazy_name")
        loader = object.__getattribute__(self, "_lazy_loader")
        logger.info("Loading deferred module %s", name)
        module = loader()
        if module is None:
            raise ValueError(f"Deferred loader for module {name} returned None")

        transform = object.__getattribute__(self, "_lazy_materialize_transform")
        if transform is not None:
            module = transform(module)
            if module is None:
                raise ValueError(f"Materialize transform for module {name} returned None")
        object.__setattr__(self, "_lazy_module", module)

        allocated = _cuda_allocated_gib()
        if allocated is not None:
            logger.info("Loaded deferred module %s, cuda allocated now %.2f GiB", name, allocated)
        return module

    def set_materialize_transform(self, transform: Callable[[Any], Any]) -> None:
        """Apply ``transform`` to this and every future loaded instance.

        Registering a transform does not itself load a deferred component. If
        something has already materialized the component, transform that
        instance immediately so current and future instances have the same
        setup. A transform may return a wrapper, as ``torch.compile`` does.
        """
        current_transform = object.__getattribute__(self, "_lazy_materialize_transform")
        module = object.__getattribute__(self, "_lazy_module")
        if current_transform is not None:
            inner = current_transform

            def chained(loaded: Any) -> Any:
                return transform(inner(loaded))

            stored: Callable[[Any], Any] = chained
            # The resident instance already ran ``inner``; only apply the new outer.
            immediate = transform
        else:
            stored = transform
            immediate = transform

        transformed = immediate(module) if module is not None else None
        if module is not None and transformed is None:
            raise ValueError(f"Materialize transform for module {self.lazy_name} returned None")

        object.__setattr__(self, "_lazy_materialize_transform", stored)
        if module is not None:
            object.__setattr__(self, "_lazy_module", transformed)

    def add_release_callback(self, callback: Callable[[], None]) -> None:
        """Run ``callback`` each time the real component is dropped."""
        object.__getattribute__(self, "_lazy_release_callbacks").append(callback)

    def release(self) -> bool:
        """Drop the real component. Returns True if something was released."""
        module = object.__getattribute__(self, "_lazy_module")
        if module is None:
            return False

        name = object.__getattribute__(self, "_lazy_name")
        before = _cuda_allocated_gib()
        object.__setattr__(self, "_lazy_module", None)
        del module
        for callback in list(object.__getattribute__(self, "_lazy_release_callbacks")):
            callback()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        after = _cuda_allocated_gib()
        if before is not None and after is not None:
            logger.info("Released deferred module %s, cuda allocated %.2f -> %.2f GiB, freed %.2f GiB", name, before,
                        after, before - after)
        else:
            logger.info("Released deferred module %s", name)
        return True

    # ------------------------------------------------------------------
    # Proxying
    # ------------------------------------------------------------------

    def __getattr__(self, item: str) -> Any:
        # __slots__ and the methods above are found by normal lookup, so
        # reaching here means the attribute belongs to the real component.
        attr = getattr(self.materialize(), item)
        if inspect.ismethod(attr) or inspect.isbuiltin(attr):
            return self._preserve_identity(attr)
        return attr

    def _preserve_identity(self, method: Any) -> Any:
        """Return the proxy, not the component, from self-returning methods.

        ``nn.Module.to`` and its relatives return ``self``, and callers write
        ``self.vae = self.vae.to(device)`` all over the stages. Handing back
        the real component there would quietly replace the proxy with a strong
        reference the pipeline cannot release, and the run would look normal
        while freeing nothing.
        """

        @functools.wraps(method)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            result = method(*args, **kwargs)
            if result is object.__getattribute__(self, "_lazy_module"):
                return self
            return result

        return wrapper

    def __setattr__(self, item: str, value: Any) -> None:
        setattr(self.materialize(), item, value)

    def __delattr__(self, item: str) -> None:
        delattr(self.materialize(), item)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.materialize()(*args, **kwargs)

    @property  # type: ignore[misc]
    def __class__(self) -> type:  # type: ignore[override]
        # isinstance() consults __class__ when the exact type does not match,
        # so forwarding it keeps `isinstance(module, FSDPModule)` and friends
        # honest. The cost is that an isinstance check materializes; a wrong
        # answer would be worse, because callers branch on it silently.
        return type(self.materialize())

    def __repr__(self) -> str:
        # Deliberately does not materialize: logging a pipeline must not
        # trigger a multi-gigabyte load.
        name = object.__getattribute__(self, "_lazy_name")
        state = "materialized" if object.__getattribute__(self, "_lazy_module") is not None else "deferred"
        return f"<LazyModule {name} ({state})>"


def is_lazy_module(obj: Any) -> TypeGuard[LazyModule]:
    """Type test that does not materialize, unlike ``isinstance``."""
    return type(obj) is LazyModule
