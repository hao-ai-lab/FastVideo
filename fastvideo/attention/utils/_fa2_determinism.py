# SPDX-License-Identifier: Apache-2.0
"""Propagate PyTorch determinism without changing ordinary FA2 defaults."""

import inspect
from typing import Any

import torch


def _resolve_fa2_deterministic(requested: bool = False) -> bool:
    return requested or torch.are_deterministic_algorithms_enabled()


def _wrap_fa2_autograd(base: Any) -> Any:
    """Keep FA2's saved RNG/padding behavior, and recheck policy in backward."""

    def backward(ctx: Any, *gradients: Any) -> Any:
        ctx.deterministic = _resolve_fa2_deterministic(ctx.deterministic)
        return base.backward(ctx, *gradients)

    return type(f"_TorchDeterministic{base.__name__}", (base, ), {"backward": staticmethod(backward)})


def _call_fa2_autograd(function: Any, **values: Any) -> Any:
    # FA2 releases differ in whether forward takes is_grad_enabled explicitly.
    # Bind by its declared names, retaining the original positional ABI.
    names = tuple(inspect.signature(function.forward).parameters)[1:]
    return function.apply(*(values[name] for name in names))
