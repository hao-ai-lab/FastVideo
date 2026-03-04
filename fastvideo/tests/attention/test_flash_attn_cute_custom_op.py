# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch


_MODULE_NAME = "_fastvideo_flash_attn_cute_for_tests"


def _load_local_flash_attn_cute_module():
    if _MODULE_NAME in sys.modules:
        return sys.modules[_MODULE_NAME]

    module_path = (
        Path(__file__).resolve().parents[2]
        / "attention"
        / "utils"
        / "flash_attn_cute.py"
    )
    spec = importlib.util.spec_from_file_location(_MODULE_NAME, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module spec from {module_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[_MODULE_NAME] = module
    spec.loader.exec_module(module)
    return module


def _assert_close(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    dtype: torch.dtype,
    is_grad: bool = False,
) -> None:
    if dtype == torch.float16:
        atol = 6e-3 if is_grad else 4e-3
        rtol = 6e-3 if is_grad else 4e-3
    else:
        atol = 2e-2 if is_grad else 1e-2
        rtol = 2e-2 if is_grad else 1e-2
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)


def _extract_out(x):
    if isinstance(x, tuple):
        return x[0]
    return x


@pytest.fixture(scope="module")
def flash_attn_impls():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for flash_attn.cute parity tests")

    major, _ = torch.cuda.get_device_capability()
    if major not in (9, 10, 11):
        pytest.skip(
            f"flash_attn.cute supports SM 9/10/11, got compute capability {major}"
        )

    try:
        from flash_attn.cute import (
            flash_attn_func as ref_flash_attn_func,
            flash_attn_varlen_func as ref_flash_attn_varlen_func,
        )
    except ImportError as exc:
        pytest.skip(f"flash_attn.cute is not available: {exc}")

    mod = _load_local_flash_attn_cute_module()
    return (
        mod.flash_attn_func,
        mod.flash_attn_varlen_func,
        ref_flash_attn_func,
        ref_flash_attn_varlen_func,
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("causal", [False, True])
def test_flash_attn_func_parity_forward_backward(
    flash_attn_impls, dtype: torch.dtype, causal: bool
):
    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        pytest.skip("bfloat16 is not supported on this GPU")

    custom_flash_attn_func, _, ref_flash_attn_func, _ = flash_attn_impls

    torch.manual_seed(0)
    device = torch.device("cuda")
    batch, seqlen, nheads, headdim = 2, 64, 4, 64

    q_ref = torch.randn(
        batch,
        seqlen,
        nheads,
        headdim,
        device=device,
        dtype=dtype,
        requires_grad=True,
    )
    k_ref = torch.randn_like(q_ref, requires_grad=True)
    v_ref = torch.randn_like(q_ref, requires_grad=True)

    q_test = q_ref.detach().clone().requires_grad_(True)
    k_test = k_ref.detach().clone().requires_grad_(True)
    v_test = v_ref.detach().clone().requires_grad_(True)

    out_ref = _extract_out(ref_flash_attn_func(
        q_ref,
        k_ref,
        v_ref,
        softmax_scale=None,
        causal=causal,
        deterministic=False,
    ))
    out_test = _extract_out(custom_flash_attn_func(
        q_test,
        k_test,
        v_test,
        dropout_p=0.0,
        softmax_scale=None,
        causal=causal,
        deterministic=False,
    ))

    _assert_close(out_test, out_ref, dtype=dtype)

    dout = torch.randn_like(out_ref)
    dq_ref, dk_ref, dv_ref = torch.autograd.grad(
        (out_ref * dout).sum(), (q_ref, k_ref, v_ref)
    )
    dq_test, dk_test, dv_test = torch.autograd.grad(
        (out_test * dout).sum(), (q_test, k_test, v_test)
    )

    _assert_close(dq_test, dq_ref, dtype=dtype, is_grad=True)
    _assert_close(dk_test, dk_ref, dtype=dtype, is_grad=True)
    _assert_close(dv_test, dv_ref, dtype=dtype, is_grad=True)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("causal", [False, True])
def test_flash_attn_varlen_func_parity_forward_backward(
    flash_attn_impls, dtype: torch.dtype, causal: bool
):
    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        pytest.skip("bfloat16 is not supported on this GPU")

    (
        _,
        custom_flash_attn_varlen_func,
        _,
        ref_flash_attn_varlen_func,
    ) = flash_attn_impls

    torch.manual_seed(1)
    device = torch.device("cuda")
    nheads, headdim = 4, 64
    seqlens = [40, 24, 16]
    cu_vals = [0]
    for seqlen in seqlens:
        cu_vals.append(cu_vals[-1] + seqlen)
    cu = torch.tensor(cu_vals, device=device, dtype=torch.int32)
    total = int(cu[-1].item())
    max_seqlen = max(seqlens)

    q_ref = torch.randn(
        total, nheads, headdim, device=device, dtype=dtype, requires_grad=True
    )
    k_ref = torch.randn_like(q_ref, requires_grad=True)
    v_ref = torch.randn_like(q_ref, requires_grad=True)

    q_test = q_ref.detach().clone().requires_grad_(True)
    k_test = k_ref.detach().clone().requires_grad_(True)
    v_test = v_ref.detach().clone().requires_grad_(True)

    out_ref = _extract_out(ref_flash_attn_varlen_func(
        q_ref,
        k_ref,
        v_ref,
        cu_seqlens_q=cu,
        cu_seqlens_k=cu,
        max_seqlen_q=max_seqlen,
        max_seqlen_k=max_seqlen,
        softmax_scale=None,
        causal=causal,
        deterministic=False,
    ))
    out_test = _extract_out(custom_flash_attn_varlen_func(
        q_test,
        k_test,
        v_test,
        cu,
        cu,
        max_seqlen,
        max_seqlen,
        dropout_p=0.0,
        softmax_scale=None,
        causal=causal,
        deterministic=False,
    ))

    _assert_close(out_test, out_ref, dtype=dtype)

    dout = torch.randn_like(out_ref)
    dq_ref, dk_ref, dv_ref = torch.autograd.grad(
        (out_ref * dout).sum(), (q_ref, k_ref, v_ref)
    )
    dq_test, dk_test, dv_test = torch.autograd.grad(
        (out_test * dout).sum(), (q_test, k_test, v_test)
    )

    _assert_close(dq_test, dq_ref, dtype=dtype, is_grad=True)
    _assert_close(dk_test, dk_ref, dtype=dtype, is_grad=True)
    _assert_close(dv_test, dv_ref, dtype=dtype, is_grad=True)
