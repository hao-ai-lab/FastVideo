from __future__ import annotations

from pathlib import Path

import torch

from fastvideo.eval.evaluator import create_evaluator
from fastvideo.eval.types import MetricResult


def evaluate(
    generated: torch.Tensor | str | Path,
    reference: torch.Tensor | str | Path | None = None,
    metrics: list[str] | str = "all",
    device: str = "cuda",
    **kwargs,
) -> dict[str, MetricResult] | list[dict[str, MetricResult]]:
    """One-shot evaluation. For repeated use, prefer :func:`create_evaluator`.

    Parameters
    ----------
    generated : Tensor | str | Path
        Generated video. Either a pre-loaded ``(T, C, H, W)`` tensor or a
        path to an mp4/avi/etc. — paths are decoded by the worker.
    reference : Tensor | str | Path | None
        Reference video (same accepted shapes as *generated*).
    metrics : list[str] | str
        Metric names, or ``"all"``.
    device : str
        PyTorch device string.
    """
    ev = create_evaluator(metrics=metrics, device=device)
    kw: dict = {"video": generated, **kwargs}
    if reference is not None:
        kw["reference"] = reference
    return ev.evaluate(**kw)
