# SPDX-License-Identifier: Apache-2.0
"""CPU checks of the emulator's host allocation decisions; kernel launches are replaced.

Load the two host definitions from the kernel file so these tests do not need
Triton or a CUDA driver. Numerical forward/backward coverage lives in
fastvideo-kernel/tests/test_attn_qat_vsa_train.py and requires CUDA.
"""
import ast
import math
from pathlib import Path

import pytest
import torch


def _host_api(calls):
    path = Path(__file__).resolve().parents[3] / "fastvideo-kernel/python/fastvideo_kernel/triton_kernels/attn_qat_vsa_train.py"
    tree = ast.parse(path.read_text())
    tree.body = [node for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef))
                 and node.name in {"_FP4VSAAttnQAT", "fp4_vsa_attn_qat"}]

    class Launch:
        def __getitem__(self, grid):
            def run(*args, **kwargs):
                calls.append(kwargs["SAVE_MRUN"])
                args[3].copy_(args[0])
                args[4].copy_(args[0])
                args[5].zero_()
            return run

    class BackwardLaunch:
        def __init__(self, outputs):
            self.outputs = outputs

        def __getitem__(self, grid):
            def run(*args, **kwargs):
                for index in self.outputs:
                    args[index].fill_(1)
            return run

    namespace = dict(torch=torch, math=math, BLOCK=128, LOG2E=math.log2(math.e), LOG2_INV_TWO_LEVEL=0.0,
                     _ROWS=64, _WARPS=8, _fwd_kernel=Launch(),
                     fp8_per_token_fake_quant=lambda value: value, mxfp8_fake_quant=lambda value: value,
                     _bwd_dq_kernel=BackwardLaunch((6,)), _bwd_dkdv_kernel=BackwardLaunch((7, 8)),
                     invert_block_lists=lambda idx, count, n: (idx, idx, count),
                     fake_quant_qkv_triton=lambda q, k, v, **kw: (q, k, v))
    exec(compile(tree, str(path), "exec"), namespace)
    return namespace["fp4_vsa_attn_qat"]


@pytest.mark.parametrize("grad_enabled,requires_grad,quantize_pv,prefix,expected", [
    (False, False, True, None, [False]),
    (False, True, True, None, [False]),
    (True, False, True, None, [False]),
    (True, True, True, None, [True]),
    (True, True, False, None, [False]),
    (False, True, True, "bf16", [False, False]),
    (True, True, True, "bf16", [True, False]),
])
def test_running_maxima_only_allocated_for_quantized_p_backward(
        monkeypatch, grad_enabled, requires_grad, quantize_pv, prefix, expected):
    launches = []
    allocations = []
    attention = _host_api(launches)
    q, k, v = [torch.ones(1, 2, 384, 128, dtype=torch.bfloat16, requires_grad=requires_grad) for _ in range(3)]
    indices = torch.zeros(1, 2, 3, 3, dtype=torch.int32)
    counts = torch.ones(1, 2, 3, dtype=torch.int32)
    original = torch.empty

    def allocate(shape, **kwargs):
        allocations.append(tuple(shape))
        return original(shape, **kwargs)

    monkeypatch.setattr(torch, "empty", allocate)
    with torch.set_grad_enabled(grad_enabled):
        out = attention(q, k, v, indices, counts, quantize_pv=quantize_pv,
                        prefix_precision=prefix, num_prefix_tiles=1 if prefix else None)
    assert launches == expected
    assert allocations.count((1, 2, 3, 3, 128)) == sum(expected)
    assert out.requires_grad is (grad_enabled and requires_grad)
    if out.requires_grad:
        out.sum().backward()
        assert all(t.grad is not None and torch.isfinite(t.grad).all() for t in (q, k, v))
