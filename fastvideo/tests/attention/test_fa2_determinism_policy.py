# SPDX-License-Identifier: Apache-2.0
"""CPU-only FA2 policy tests: no backend, CUDA driver or weights are imported."""
from contextlib import contextmanager
import importlib.util
from pathlib import Path

import pytest
import torch

spec = importlib.util.spec_from_file_location(
    "fa2_policy_test", Path(__file__).resolve().parents[2] / "attention/utils/_fa2_determinism.py")
policy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(policy)


@contextmanager
def mode(enabled, warn_only=False):
    saved = (torch.are_deterministic_algorithms_enabled(), torch.is_deterministic_algorithms_warn_only_enabled())
    torch.use_deterministic_algorithms(enabled, warn_only=warn_only)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(saved[0], warn_only=saved[1])


@pytest.mark.parametrize("enabled,warn_only", [(False, False), (True, False), (True, True)])
@pytest.mark.parametrize("requested", [False, True])
def test_effective_policy(enabled, warn_only, requested):
    with mode(enabled, warn_only):
        assert policy._resolve_fa2_deterministic(requested) is (enabled or requested)


def test_cudnn_setting_does_not_enable_fa2():
    previous = torch.backends.cudnn.deterministic
    try:
        with mode(False):
            torch.backends.cudnn.deterministic = True
            assert policy._resolve_fa2_deterministic() is False
    finally:
        torch.backends.cudnn.deterministic = previous


def test_error_restores_global_mode():
    previous = (torch.are_deterministic_algorithms_enabled(), torch.is_deterministic_algorithms_warn_only_enabled())
    with pytest.raises(RuntimeError, match="sentinel"):
        with mode(True):
            raise RuntimeError("sentinel")
    assert previous == (torch.are_deterministic_algorithms_enabled(), torch.is_deterministic_algorithms_warn_only_enabled())


@pytest.mark.parametrize("forward_mode,backward_mode,requested", [
    (False, False, False), (False, True, False), (True, False, False),
    (True, True, False), (False, False, True),
])
@pytest.mark.parametrize("new_signature", [False, True])
def test_native_adapter_preserves_saved_context_and_merges_backward(forward_mode, backward_mode, requested, new_signature):
    observed = []

    class Old(torch.autograd.Function):
        @staticmethod
        def forward(ctx, value, deterministic):
            ctx.deterministic = deterministic
            ctx.save_for_backward(value)
            return value * 2

        @staticmethod
        def backward(ctx, gradient):
            observed.append((ctx.deterministic, ctx.saved_tensors[0].item()))
            return gradient * 2, None

    class New(Old):
        @staticmethod
        def forward(ctx, value, deterministic, is_grad_enabled):
            assert is_grad_enabled
            return Old.forward(ctx, value, deterministic)

        @staticmethod
        def backward(ctx, gradient):
            return (*Old.backward(ctx, gradient), None)

    adapted = policy._wrap_fa2_autograd(New if new_signature else Old)
    value = torch.tensor(3., requires_grad=True)
    with mode(forward_mode):
        out = policy._call_fa2_autograd(adapted, value=value,
                                      deterministic=policy._resolve_fa2_deterministic(requested),
                                      is_grad_enabled=torch.is_grad_enabled())
    with mode(backward_mode):
        out.backward()
    assert observed == [(requested or forward_mode or backward_mode, 3.)]
    assert value.grad.item() == 2.
