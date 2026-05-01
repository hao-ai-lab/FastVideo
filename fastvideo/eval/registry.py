from __future__ import annotations

import importlib.util
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from fastvideo.eval.metrics.base import BaseMetric

_REGISTRY: dict[str, type[BaseMetric]] = {}


def register(name: str):
    """Decorator to register a metric class.

    Usage::

        @register("ssim")
        class SSIMMetric(BaseMetric):
            ...
    """

    def wrapper(cls):
        _REGISTRY[name] = cls
        return cls

    return wrapper


def get_metric(name: str, **kwargs: Any) -> BaseMetric:
    """Instantiate a registered metric by name.

    Checks that optional dependencies are installed before instantiation
    and gives a clear install hint if not.
    """
    cls = _REGISTRY.get(name)
    if cls is None:
        available = ", ".join(sorted(_REGISTRY.keys()))
        raise KeyError(f"Unknown metric '{name}'. Available: {available}")

    for dep in getattr(cls, "dependencies", []):
        if not importlib.util.find_spec(dep):
            raise ImportError(
                f"{cls.__name__} requires '{dep}'. "
                f"Install with: pip install fastvideo[eval]"
            )

    return cls(**kwargs)


def list_metrics() -> list[str]:
    """Return sorted list of all registered metric names."""
    return sorted(_REGISTRY.keys())


def resolve_group(name: str) -> list[str] | None:
    """If *name* is a group prefix (e.g. ``"vbench"``), return all matching
    metric names.  Returns ``None`` if *name* is not a group."""
    prefix = name + "."
    matches = sorted(k for k in _REGISTRY if k.startswith(prefix))
    return matches if matches else None
