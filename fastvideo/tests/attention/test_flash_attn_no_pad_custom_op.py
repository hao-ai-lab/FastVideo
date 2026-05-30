# SPDX-License-Identifier: Apache-2.0
"""Regression guard for the masked/varlen custom ops.

`flash_attn_no_pad_compilable` / `flash_attn_varlen_qk_no_pad_compilable` wrap
the whole masked-attention functions in `torch.library.custom_op`s so dynamo
sees one traceable node (the internal unpad/pad bookkeeping runs eager inside).

On FA2 these ops register a real backward — `softmax_lse` is padded back to a
statically-shaped `[batch, nheads, seqlen]` form on the way out and re-unpadded
in backward, which calls FA2's `_flash_attn_varlen_backward`. So training also
backprops through the op (no graph break on the training path either).

On FA3/FA4 these ops are forward+fake only and the `*_compilable` dispatchers
carve out to the autograd.Function for grad-enabled calls (PR #1373 pattern).
Tests gating on FA2 are skipped on FA3/FA4.

These tests pin:
  - inference (no grad): output through the custom op is bit-identical to the
    original function;
  - training (requires_grad): gradients through the registered op match the
    original autograd.Function;
  - schema/fake-kernel consistency via torch.library.opcheck, both with and
    without grad-requiring inputs (the latter exercises
    `test_autograd_registration` — would have caught a missing/broken backward
    at unit-test time).

GPU assumptions: requires CUDA and the FA2 `flash_attn` varlen package.
Skips on CPU and when flash_attn is unavailable.
"""

from __future__ import annotations

import pytest
import torch


@pytest.fixture(scope="module")
def no_pad_impls():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the masked/varlen custom-op tests")
    try:
        from fastvideo.attention.utils import flash_attn_no_pad as mod
    except ImportError as exc:
        pytest.skip(f"flash_attn_no_pad not importable (flash_attn missing?): {exc}")
    return mod


def _dtype_skip(dtype):
    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        pytest.skip("bfloat16 is not supported on this GPU")


def _fa2_only(mod):
    if mod._FA_VARLEN_VERSION != "2":
        pytest.skip(
            f"register_autograd is only wired for FA2; got "
            f"_FA_VARLEN_VERSION={mod._FA_VARLEN_VERSION!r}"
        )


def _padding_mask(batch, seqlen, valid_lens, device):
    mask = torch.zeros(batch, seqlen, dtype=torch.bool, device=device)
    for i, n in enumerate(valid_lens):
        mask[i, :n] = True
    return mask


# --------------------------------------------------------------------------- #
# flash_attn_no_pad (masked self-attention)                                    #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_no_pad_inference_matches_original(no_pad_impls, dtype):
    """No-grad path through the custom op is bit-identical to the original."""
    _dtype_skip(dtype)
    mod = no_pad_impls
    torch.manual_seed(0)
    device = torch.device("cuda")
    b, s, h, d = 2, 64, 4, 64
    qkv = torch.randn(b, s, 3, h, d, device=device, dtype=dtype)
    mask = _padding_mask(b, s, [64, 48], device)

    with torch.inference_mode():
        out_ref = mod.flash_attn_no_pad(qkv, mask, causal=False, dropout_p=0.0)
        out_test = mod.flash_attn_no_pad_compilable(qkv, mask, causal=False, dropout_p=0.0)
    torch.testing.assert_close(out_test, out_ref, atol=0, rtol=0)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_no_pad_training_backward_through_registered_autograd(no_pad_impls, dtype):
    """FA2: grads flow through the registered op and match the original."""
    _dtype_skip(dtype)
    mod = no_pad_impls
    _fa2_only(mod)
    torch.manual_seed(0)
    device = torch.device("cuda")
    b, s, h, d = 2, 64, 4, 64
    mask = _padding_mask(b, s, [64, 48], device)

    qkv_ref = torch.randn(b, s, 3, h, d, device=device, dtype=dtype, requires_grad=True)
    qkv_test = qkv_ref.detach().clone().requires_grad_(True)

    out_ref = mod.flash_attn_no_pad(qkv_ref, mask, causal=False, dropout_p=0.0)
    # Go through the compilable wrapper (which on FA2 unconditionally routes
    # to the registered op — no carve-out).
    out_test = mod.flash_attn_no_pad_compilable(qkv_test, mask, causal=False, dropout_p=0.0)

    dout = torch.randn_like(out_ref)
    (dqkv_ref,) = torch.autograd.grad((out_ref * dout).sum(), (qkv_ref,))
    (dqkv_test,) = torch.autograd.grad((out_test * dout).sum(), (qkv_test,))

    atol = rtol = 6e-3 if dtype == torch.float16 else 2e-2
    torch.testing.assert_close(dqkv_test, dqkv_ref, atol=atol, rtol=rtol)


def test_no_pad_forward_opcheck(no_pad_impls):
    """Schema/fake-kernel consistency for the custom op (forward only)."""
    torch.manual_seed(0)
    device = torch.device("cuda")
    b, s, h, d = 2, 64, 4, 64
    qkv = torch.randn(b, s, 3, h, d, device=device, dtype=torch.float16)
    mask = _padding_mask(b, s, [64, 48], device)
    torch.library.opcheck(
        torch.ops.fastvideo._flash_attn_no_pad_forward,
        (qkv, mask, False, 0.0, None, False),
    )


def test_no_pad_opcheck_with_grad_inputs(no_pad_impls):
    """FA2: full opcheck including ``test_autograd_registration`` — catches a
    missing/inconsistent backward at unit-test time."""
    mod = no_pad_impls
    _fa2_only(mod)
    torch.manual_seed(0)
    device = torch.device("cuda")
    b, s, h, d = 2, 64, 4, 64
    qkv = torch.randn(b, s, 3, h, d, device=device, dtype=torch.float16, requires_grad=True)
    mask = _padding_mask(b, s, [64, 48], device)
    torch.library.opcheck(
        torch.ops.fastvideo._flash_attn_no_pad_forward,
        (qkv, mask, False, 0.0, None, False),
    )


# --------------------------------------------------------------------------- #
# flash_attn_varlen_qk_no_pad (cross-attn / unequal q-k seqlen)                #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_varlen_qk_inference_matches_original(no_pad_impls, dtype):
    _dtype_skip(dtype)
    mod = no_pad_impls
    torch.manual_seed(1)
    device = torch.device("cuda")
    b, sq, sk, h, d = 2, 48, 64, 4, 64
    q = torch.randn(b, sq, h, d, device=device, dtype=dtype)
    k = torch.randn(b, sk, h, d, device=device, dtype=dtype)
    v = torch.randn(b, sk, h, d, device=device, dtype=dtype)
    qmask = _padding_mask(b, sq, [48, 40], device)
    kmask = _padding_mask(b, sk, [64, 56], device)

    with torch.inference_mode():
        out_ref = mod.flash_attn_varlen_qk_no_pad(q, k, v, qmask, kmask, causal=False, dropout_p=0.0)
        out_test = mod.flash_attn_varlen_qk_no_pad_compilable(q, k, v, qmask, kmask, causal=False, dropout_p=0.0)
    torch.testing.assert_close(out_test, out_ref, atol=0, rtol=0)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_varlen_qk_training_backward_through_registered_autograd(no_pad_impls, dtype):
    """FA2: grads through q, k, v all flow via the registered op and match."""
    _dtype_skip(dtype)
    mod = no_pad_impls
    _fa2_only(mod)
    torch.manual_seed(1)
    device = torch.device("cuda")
    b, sq, sk, h, d = 2, 48, 64, 4, 64
    qmask = _padding_mask(b, sq, [48, 40], device)
    kmask = _padding_mask(b, sk, [64, 56], device)

    q_ref = torch.randn(b, sq, h, d, device=device, dtype=dtype, requires_grad=True)
    k_ref = torch.randn(b, sk, h, d, device=device, dtype=dtype, requires_grad=True)
    v_ref = torch.randn(b, sk, h, d, device=device, dtype=dtype, requires_grad=True)
    q_test = q_ref.detach().clone().requires_grad_(True)
    k_test = k_ref.detach().clone().requires_grad_(True)
    v_test = v_ref.detach().clone().requires_grad_(True)

    out_ref = mod.flash_attn_varlen_qk_no_pad(q_ref, k_ref, v_ref, qmask, kmask, causal=False, dropout_p=0.0)
    out_test = mod.flash_attn_varlen_qk_no_pad_compilable(q_test, k_test, v_test, qmask, kmask,
                                                          causal=False, dropout_p=0.0)

    dout = torch.randn_like(out_ref)
    dq_ref, dk_ref, dv_ref = torch.autograd.grad((out_ref * dout).sum(), (q_ref, k_ref, v_ref))
    dq_test, dk_test, dv_test = torch.autograd.grad((out_test * dout).sum(), (q_test, k_test, v_test))

    atol = rtol = 6e-3 if dtype == torch.float16 else 2e-2
    torch.testing.assert_close(dq_test, dq_ref, atol=atol, rtol=rtol)
    torch.testing.assert_close(dk_test, dk_ref, atol=atol, rtol=rtol)
    torch.testing.assert_close(dv_test, dv_ref, atol=atol, rtol=rtol)


def test_varlen_qk_forward_opcheck(no_pad_impls):
    torch.manual_seed(1)
    device = torch.device("cuda")
    b, sq, sk, h, d = 2, 48, 64, 4, 64
    q = torch.randn(b, sq, h, d, device=device, dtype=torch.float16)
    k = torch.randn(b, sk, h, d, device=device, dtype=torch.float16)
    v = torch.randn(b, sk, h, d, device=device, dtype=torch.float16)
    qmask = _padding_mask(b, sq, [48, 40], device)
    kmask = _padding_mask(b, sk, [64, 56], device)
    torch.library.opcheck(
        torch.ops.fastvideo._flash_attn_varlen_qk_no_pad_forward,
        (q, k, v, qmask, kmask, False, 0.0, None, False),
    )


def test_varlen_qk_opcheck_with_grad_inputs(no_pad_impls):
    """FA2: full opcheck with requires_grad inputs (autograd-registration check)."""
    mod = no_pad_impls
    _fa2_only(mod)
    torch.manual_seed(1)
    device = torch.device("cuda")
    b, sq, sk, h, d = 2, 48, 64, 4, 64
    q = torch.randn(b, sq, h, d, device=device, dtype=torch.float16, requires_grad=True)
    k = torch.randn(b, sk, h, d, device=device, dtype=torch.float16, requires_grad=True)
    v = torch.randn(b, sk, h, d, device=device, dtype=torch.float16, requires_grad=True)
    qmask = _padding_mask(b, sq, [48, 40], device)
    kmask = _padding_mask(b, sk, [64, 56], device)
    torch.library.opcheck(
        torch.ops.fastvideo._flash_attn_varlen_qk_no_pad_forward,
        (q, k, v, qmask, kmask, False, 0.0, None, False),
    )
