"""Prompt-corpus datasets for end-to-end benchmark evaluation.

Public API mirrors :mod:`fastvideo.eval` (metrics side):

    from fastvideo.eval.datasets import (
        PromptDataset, Sample,
        register_dataset, get_dataset, list_datasets,
    )

A dataset is an iterable of plain dicts (one per sample). Built-in
datasets self-register at import time. To add one, drop a module into
this package that subclasses :class:`PromptDataset` and decorates with
``@register_dataset("name")`` — auto-discovery picks it up.
"""
from fastvideo.eval.datasets.base import (BasePromptDataset, PromptDataset, Sample)
from fastvideo.eval.datasets.registry import (get_dataset, list_datasets, register_dataset)


def _autodiscover() -> None:
    """Import every non-underscore .py module / subpackage in this package
    so the ``@register_dataset`` decorators fire."""
    import importlib
    import os

    for entry in os.listdir(os.path.dirname(__file__)):
        if entry.startswith("_") or entry.startswith("."):
            continue
        if entry in {"base.py", "registry.py"}:
            continue
        if entry.endswith(".py"):
            importlib.import_module(f"{__name__}.{entry[:-3]}")
        elif os.path.isdir(os.path.join(os.path.dirname(__file__), entry)) \
                and os.path.exists(os.path.join(
                    os.path.dirname(__file__), entry, "__init__.py")):
            importlib.import_module(f"{__name__}.{entry}")


_autodiscover()

# Re-export the canonical class for typed imports.
from fastvideo.eval.datasets.vbench import VBenchPromptDataset  # noqa: E402

__all__ = [
    "PromptDataset",
    "BasePromptDataset",
    "Sample",
    "register_dataset",
    "get_dataset",
    "list_datasets",
    "VBenchPromptDataset",
]
