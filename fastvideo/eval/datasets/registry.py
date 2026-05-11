"""Registry for prompt-corpus datasets, mirroring :mod:`fastvideo.eval.registry`."""
from __future__ import annotations

from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from fastvideo.eval.datasets.base import BasePromptDataset

_REGISTRY: dict[str, type[BasePromptDataset]] = {}


def register_dataset(name: str):
    """Decorator to register a prompt-dataset class.

    Usage::

        @register_dataset("vbench")
        class VBenchPromptDataset(BasePromptDataset):
            ...
    """

    def wrapper(cls):
        cls.name = name
        _REGISTRY[name] = cls
        return cls

    return wrapper


def get_dataset(name: str, **kwargs: Any) -> BasePromptDataset:
    """Instantiate a registered dataset by name."""
    cls = _REGISTRY.get(name)
    if cls is None:
        available = ", ".join(sorted(_REGISTRY.keys()))
        raise KeyError(f"Unknown dataset '{name}'. Available: {available}")
    return cls(**kwargs)


def list_datasets() -> list[str]:
    """Return sorted list of all registered dataset names."""
    return sorted(_REGISTRY.keys())
