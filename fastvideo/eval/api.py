from __future__ import annotations

import torch

from fastvideo.eval.evaluator import create_evaluator
from fastvideo.eval.types import MetricResult


def evaluate(
    generated: torch.Tensor,
    reference: torch.Tensor | None = None,
    metrics: list[str] | str = "all",
    device: str = "cuda",
    **kwargs,
) -> dict[str, MetricResult] | list[dict[str, MetricResult]]:
    """One-shot evaluation. For repeated use, prefer :func:`create_evaluator`.

    Parameters
    ----------
    generated : Tensor
        Generated video ``(T, C, H, W)`` or batch ``(B, T, C, H, W)``.
    reference : Tensor | None
        Reference video (same shapes as *generated*).
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
