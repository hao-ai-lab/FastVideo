# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch


def apply_trainable(module: torch.nn.Module, *, trainable: bool) -> torch.nn.Module:
    """Apply train/eval mode + requires_grad based on a role's trainable flag."""

    module.requires_grad_(bool(trainable))
    if trainable:
        module.train()
    else:
        module.eval()
    return module

