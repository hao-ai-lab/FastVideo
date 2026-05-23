# SPDX-License-Identifier: Apache-2.0
"""Regression guard for the FA2/FA3 default-path custom op (PR #1373).

`flash_attn_func_compilable` wraps the default FA2/FA3 `flash_attn_func` in a
`torch.library.custom_op` so dynamo sees a traceable node (no graph break)
during inference. That custom op registers a forward + fake kernel but *no*
backward, so it is opaque to autograd. To keep training working, the
dispatcher routes grad-enabled calls to the original `flash_attn_func` (an
autograd.Function) and only sends grad-free (inference) calls through the
custom op.

These tests pin both halves of that contract:
  - inference (no grad): output goes through the custom op and matches the
    original kernel;
  - training (requires_grad): gradients flow and match the original kernel;
  - the custom op itself is schema/fake-consistent under torch.library.opcheck.

GPU assumptions: requires CUDA and an active FA2/FA3 backend
(fa_version in {"2","3"}). Skips on CPU and on FA4/cute boxes (which take a
different, already-traceable path).
"""

from __future__ import annotations

import pytest
import torch


@pytest.fixture(scope="module")
def fa_default_impls():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the FA2/FA3 default custom-op tests")

    try:
        from fastvideo.attention.backends import flash_attn as fa_backend
    except ImportError as exc:
        pytest.skip(f"flash_attn backend not importable: {exc}")

    if fa_backend.fa_version not in ("2", "3"):
        pytest.skip(
            f"FA2/FA3 default custom op only exists for fa_version in (2, 3); "
            f"got {fa_backend.fa_version!r}"
        )

    # compilable dispatcher, the original FA wrapper it falls back to, the
    # raw custom op for opcheck, and the fa_version (FA2 has full register_
    # autograd; FA3 keeps the carve-out so some tests gate on this).
    return (
        fa_backend.flash_attn_func_compilable,
        fa_backend._fa_default,
        torch.ops.fastvideo._flash_attn_default_forward,
        fa_backend.fa_version,
    )


def _qkv(dtype, requires_grad):
    device = torch.device("cuda")
    batch, seqlen, nheads, headdim = 2, 64, 4, 64
    q = torch.randn(batch, seqlen, nheads, headdim, device=device,
                    dtype=dtype, requires_grad=requires_grad)
    k = torch.randn_like(q, requires_grad=requires_grad)
    v = torch.randn_like(q, requires_grad=requires_grad)
    return q, k, v


def _clone(*tensors):
    return tuple(t.detach().clone().requires_grad_(t.requires_grad) for t in tensors)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("causal", [False, True])
def test_default_compilable_inference_matches_original(fa_default_impls, dtype, causal):
    """No-grad path routes through the custom op and is numerically identical."""
    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        pytest.skip("bfloat16 is not supported on this GPU")
    compilable, original, _, _ = fa_default_impls

    torch.manual_seed(0)
    q, k, v = _qkv(dtype, requires_grad=False)
    with torch.inference_mode():
        out_ref = original(q, k, v, softmax_scale=None, causal=causal)
        out_test = compilable(q, k, v, softmax_scale=None, causal=causal)
    torch.testing.assert_close(out_test, out_ref, atol=0, rtol=0)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("causal", [False, True])
def test_default_compilable_training_backward_flows(fa_default_impls, dtype, causal):
    """Grad-enabled path falls back to the autograd.Function: grads flow + match.

    This is the exact regression PR #1373 introduced and this fix closes — a
    bare custom op with no registered backward breaks loss.backward().
    """
    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        pytest.skip("bfloat16 is not supported on this GPU")
    compilable, original, _, _ = fa_default_impls

    torch.manual_seed(0)
    q_ref, k_ref, v_ref = _qkv(dtype, requires_grad=True)
    q_test, k_test, v_test = _clone(q_ref, k_ref, v_ref)

    out_ref = original(q_ref, k_ref, v_ref, softmax_scale=None, causal=causal)
    out_test = compilable(q_test, k_test, v_test, softmax_scale=None, causal=causal)

    dout = torch.randn_like(out_ref)
    dq_ref, dk_ref, dv_ref = torch.autograd.grad(
        (out_ref * dout).sum(), (q_ref, k_ref, v_ref))
    dq_test, dk_test, dv_test = torch.autograd.grad(
        (out_test * dout).sum(), (q_test, k_test, v_test))

    atol = rtol = 6e-3 if dtype == torch.float16 else 2e-2
    torch.testing.assert_close(dq_test, dq_ref, atol=atol, rtol=rtol)
    torch.testing.assert_close(dk_test, dk_ref, atol=atol, rtol=rtol)
    torch.testing.assert_close(dv_test, dv_ref, atol=atol, rtol=rtol)


@pytest.mark.parametrize("causal", [False, True])
def test_default_forward_opcheck(fa_default_impls, causal):
    """Schema / fake-kernel consistency for the custom op (forward only)."""
    _, _, op, _ = fa_default_impls
    torch.manual_seed(0)
    q, k, v = _qkv(torch.float16, requires_grad=False)
    torch.library.opcheck(op, (q, k, v, None, causal))


# --------------------------------------------------------------------------- #
# FA2-only: backward is registered on the custom op itself. Exercises the     #
# `register_autograd` wiring directly (not via the dispatcher's carve-out).   #
# Skipped on FA3 until Kuan-Hao's Modal FA3 setup PR lands and we mirror the  #
# pattern there.                                                              #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("causal", [False, True])
def test_default_op_backward_through_registered_autograd(fa_default_impls, dtype, causal):
    """FA2: gradients flow through ``torch.ops.fastvideo._flash_attn_default_forward``
    itself (no dispatcher carve-out involved) and match the original
    ``flash_attn_func``'s gradients."""
    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        pytest.skip("bfloat16 is not supported on this GPU")
    _, original, op, fa_version = fa_default_impls
    if fa_version != "2":
        pytest.skip(f"register_autograd is only wired for FA2 right now; got {fa_version!r}")

    torch.manual_seed(0)
    q_ref, k_ref, v_ref = _qkv(dtype, requires_grad=True)
    q_test, k_test, v_test = _clone(q_ref, k_ref, v_ref)

    # Reference grads via the original autograd.Function.
    out_ref = original(q_ref, k_ref, v_ref, softmax_scale=None, causal=causal)
    # Custom-op grads via the registered backward — unpack (out, lse), discard lse.
    out_test, _ = op(q_test, k_test, v_test, None, causal)

    torch.testing.assert_close(out_test, out_ref,
                               atol=0 if dtype == torch.float16 else 1e-3,
                               rtol=0 if dtype == torch.float16 else 1e-3)

    dout = torch.randn_like(out_ref)
    dq_ref, dk_ref, dv_ref = torch.autograd.grad(
        (out_ref * dout).sum(), (q_ref, k_ref, v_ref))
    dq_test, dk_test, dv_test = torch.autograd.grad(
        (out_test * dout).sum(), (q_test, k_test, v_test))

    atol = rtol = 6e-3 if dtype == torch.float16 else 2e-2
    torch.testing.assert_close(dq_test, dq_ref, atol=atol, rtol=rtol)
    torch.testing.assert_close(dk_test, dk_ref, atol=atol, rtol=rtol)
    torch.testing.assert_close(dv_test, dv_ref, atol=atol, rtol=rtol)


@pytest.mark.parametrize("causal", [False, True])
def test_default_op_opcheck_with_grad_inputs(fa_default_impls, causal):
    """FA2: full ``opcheck`` including ``test_autograd_registration`` —
    catches a missing/inconsistent backward at unit-test time, which was
    exactly the gap that #1373's first revision shipped."""
    _, _, op, fa_version = fa_default_impls
    if fa_version != "2":
        pytest.skip(f"autograd registration only wired for FA2; got {fa_version!r}")
    torch.manual_seed(0)
    q, k, v = _qkv(torch.float16, requires_grad=True)
    torch.library.opcheck(op, (q, k, v, None, causal))
