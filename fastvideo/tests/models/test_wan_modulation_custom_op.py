# SPDX-License-Identifier: Apache-2.0
"""Compiled Wan modulation forward/backward regression tests."""

from __future__ import annotations

import pytest
import torch

# Import registers the model-local custom operators.
import fastvideo.models.dits.wanvideo  # noqa: F401


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("temb_shape", [(2, 6, 32), (2, 3, 6, 32)])
def test_wan_modulation_fullgraph_backward_matches_eager(temb_shape: tuple[int, ...]) -> None:
    torch.manual_seed(0)
    scale_base = torch.randn(1, 6, 32, device="cuda", dtype=torch.bfloat16)
    temb_base = torch.randn(temb_shape, device="cuda", dtype=torch.bfloat16)
    chunk_shape = (*temb_shape[:2], 1, temb_shape[-1]) if len(temb_shape) == 4 else (temb_shape[0], 1,
                                                                                   temb_shape[-1])
    weights = [torch.randn(chunk_shape, device="cuda") for _ in range(6)]

    scale_eager = scale_base.clone().requires_grad_(True)
    temb_eager = temb_base.clone().requires_grad_(True)
    chunk_dim = 2 if len(temb_shape) == 4 else 1
    modulation = (scale_eager.unsqueeze(0).float() if len(temb_shape) == 4 else
                  scale_eager.float()) + temb_eager.float()
    eager_outputs = modulation.chunk(6, dim=chunk_dim)
    eager_loss = sum((output * weight).sum() for output, weight in zip(eager_outputs, weights))
    eager_grads = torch.autograd.grad(eager_loss, (scale_eager, temb_eager))

    scale_compiled = scale_base.clone().requires_grad_(True)
    temb_compiled = temb_base.clone().requires_grad_(True)

    def loss(scale_shift_table: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        outputs = torch.ops.fastvideo._wan_modulation_forward(scale_shift_table, temb)
        return sum((output * weight).sum() for output, weight in zip(outputs, weights))

    compiled_loss = torch.compile(loss, fullgraph=True)(scale_compiled, temb_compiled)
    compiled_grads = torch.autograd.grad(compiled_loss, (scale_compiled, temb_compiled))

    for compiled_grad, eager_grad in zip(compiled_grads, eager_grads):
        torch.testing.assert_close(compiled_grad, eager_grad, rtol=0, atol=0)
