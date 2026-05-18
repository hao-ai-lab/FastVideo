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
    and gives a clear install hint pointing at the right extra group.
    """
    cls = _REGISTRY.get(name)
    if cls is None:
        available = ", ".join(sorted(_REGISTRY.keys()))
        raise KeyError(f"Unknown metric '{name}'. Available: {available}")

    for dep in getattr(cls, "dependencies", []):
        if not importlib.util.find_spec(dep):
            raise ImportError(f"{cls.__name__} requires '{dep}'. "
                              f"Install with: {_install_hint(name, dep)}")

    return cls(**kwargs)


def missing_dependencies(metric_name: str) -> list[str]:
    """Importable module names declared by *metric_name* that are not
    actually importable in this environment. Returns ``[]`` if all deps
    are satisfied or the metric is unknown.

    Used by group-style resolution to decide which metrics to silently
    skip (vs. naming a metric explicitly, where the missing dep should
    surface as :class:`ImportError`).
    """
    cls = _REGISTRY.get(metric_name)
    if cls is None:
        return []
    return [d for d in getattr(cls, "dependencies", []) if not importlib.util.find_spec(d)]


def _install_hint(metric_name: str, dep: str) -> str:
    """Copy-pastable install command that actually satisfies *dep*.

    Most metrics resolve to a single ``uv pip install -e .[<extra>]``
    (``imagebind`` is git-installed via ``[tool.uv.sources]`` and rides
    that recipe). ``detectron2`` stays a manual two-step: it builds
    C++ kernels against the user's torch and isn't on PyPI cleanly, so
    its install requires the extra *plus* a separate
    ``--no-build-isolation`` git+ install.
    """
    if dep == "detectron2":
        return ("uv pip install 'fastvideo[eval-vbench]' && "
                "uv pip install --no-build-isolation "
                "'git+https://github.com/facebookresearch/detectron2.git'")
    return f"uv pip install -e '.[{_extra_for(metric_name)}]'"


def _extra_for(metric_name: str) -> str:
    """Map a metric name to the smallest extra that satisfies its deps."""
    if metric_name.startswith("vbench."):
        return "eval-vbench"
    if metric_name.startswith("physics_iq"):
        return "eval-physics-iq"
    if metric_name.startswith("audio."):
        return "eval-audio"
    return "eval"


def list_metrics() -> list[str]:
    """Return sorted list of all registered metric names."""
    return sorted(_REGISTRY.keys())


def resolve_group(name: str) -> list[str] | None:
    """If *name* is a group prefix (e.g. ``"vbench"``), return all matching
    metric names.  Returns ``None`` if *name* is not a group."""
    prefix = name + "."
    matches = sorted(k for k in _REGISTRY if k.startswith(prefix))
    return matches if matches else None
