# SPDX-License-Identifier: Apache-2.0
"""Context managers for training-time FastVideo LoRA adapters."""

from __future__ import annotations

from contextlib import contextmanager
from collections.abc import Iterator

import torch

from fastvideo.layers.lora.linear import BaseLayerWithLoRA


@contextmanager
def temporarily_disable_lora(module: torch.nn.Module) -> Iterator[None]:
    """Disable every unmerged LoRA layer and restore its previous state.

    RVM uses this to obtain the frozen FastH3 reference prediction from the
    same materialized 35B model, avoiding a second model copy.
    """
    layers: list[tuple[BaseLayerWithLoRA, bool]] = []
    for submodule in module.modules():
        if not isinstance(submodule, BaseLayerWithLoRA):
            continue
        if submodule.merged:
            raise RuntimeError("Cannot disable a LoRA layer after it has been merged into the base weight")
        layers.append((submodule, bool(submodule.disable_lora)))
        submodule.disable_lora = True
    if not layers:
        raise RuntimeError("No training LoRA layers were found on the module")
    try:
        yield
    finally:
        for layer, old_value in layers:
            layer.disable_lora = old_value
