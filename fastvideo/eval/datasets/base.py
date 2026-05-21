"""Prompt-corpus datasets.

A :class:`PromptDataset` is an iterable of *sample dicts* describing the
prompts and conditions for a benchmark. Each sample is a plain dict —
no dataclass, no schema enforcement — that flows directly into both
generation (``VideoGenerator.generate_video(**sample)``) and scoring
(``Evaluator.evaluate(**eval_kwargs)``). The runner picks well-known
keys (``prompt``, ``n_samples``, ``dimensions``, ``auxiliary_info``,
...) and passes the rest through.

This matches the surrounding FastVideo style:

* :class:`fastvideo.dataset.validation_dataset.ValidationDataset` yields dicts.
* :meth:`fastvideo.VideoGenerator.generate_video` consumes ``**kwargs``.
* :meth:`fastvideo.eval.Evaluator.evaluate` consumes ``**kwargs``.

To add a new benchmark:

1. Subclass :class:`PromptDataset`, populate ``self._rows`` with dicts in
   ``__init__``.
2. Decorate with ``@register_dataset("my_bench")``.

Convention for ``auxiliary_info``: a *flat* dict of metric-keyed values
(e.g. ``{"color": "red"}``). Benchmarks with nested aux schemas (VBench's
``{dim: {key: val}}``) flatten at load time so every consumer sees the
same shape.
"""
from __future__ import annotations

from typing import TypedDict
from collections.abc import Iterator


class Sample(TypedDict, total=False):
    """Documented schema for a row yielded by :class:`PromptDataset`.

    Only ``prompt`` is required. Extra keys beyond these are forwarded to
    the runner's eval-kwargs builder verbatim, so action-conditioned or
    audio-bearing benchmarks can add their own fields without changing
    the base class.
    """
    prompt: str
    n_samples: int
    dimensions: list[str]
    auxiliary_info: dict
    image_path: str
    reference_video: str


class PromptDataset:
    """Iterable corpus of sample dicts. Subclasses populate ``self._rows``."""

    name: str = ""
    description: str = ""
    supports_dimensions: bool = False
    requires_reference_image: bool = False
    requires_reference_video: bool = False

    def __init__(self) -> None:
        self._rows: list[dict] = []

    def __iter__(self) -> Iterator[dict]:
        return iter(self._rows)

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, i: int) -> dict:
        return self._rows[i]

    def by_dimension(self) -> dict[str, list[dict]]:
        """Group samples by dimension. A multi-dim sample appears under each."""
        out: dict[str, list[dict]] = {}
        for s in self._rows:
            for d in s.get("dimensions", ()):
                out.setdefault(d, []).append(s)
        return out


# Back-compat alias for callers still importing the old class name.
BasePromptDataset = PromptDataset
