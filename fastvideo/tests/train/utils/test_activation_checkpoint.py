# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import copy
import importlib

import pytest
import torch
from torch import nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import CheckpointWrapper

import fastvideo.train.utils.activation_checkpoint as activation_checkpoint
from fastvideo.train.utils.activation_checkpoint import apply_activation_checkpointing

_TEST_OP_NAME = "fastvideo_activation_checkpoint_test::expensive_op"
_TEST_OP_CALLS = 0


@torch.library.custom_op(_TEST_OP_NAME, mutates_args=())
def _expensive_op(value: torch.Tensor) -> torch.Tensor:
    global _TEST_OP_CALLS
    _TEST_OP_CALLS += 1
    return torch.sin(value)


@_expensive_op.register_fake
def _expensive_op_fake(value: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(value)


def _setup_expensive_op_context(ctx, inputs, output) -> None:
    del output
    ctx.save_for_backward(inputs[0])


def _backward_expensive_op(ctx, grad_output: torch.Tensor) -> torch.Tensor:
    (value,) = ctx.saved_tensors
    return grad_output * torch.cos(value)


_expensive_op.register_autograd(_backward_expensive_op, setup_context=_setup_expensive_op_context)


class _ToyBlock(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Linear(4, 4)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value + _expensive_op(self.projection(value))


class _ToyTransformer(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([_ToyBlock() for _ in range(4)])

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            value = block(value)
        return value


def _run_toy_model(
    state_dict: dict[str, torch.Tensor],
    checkpointing_type: str | None,
) -> tuple[int, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    global _TEST_OP_CALLS
    _TEST_OP_CALLS = 0

    model = _ToyTransformer()
    model.load_state_dict(copy.deepcopy(state_dict))
    if checkpointing_type is not None:
        apply_activation_checkpointing(model, checkpointing_type)

    value = torch.linspace(-0.5, 0.5, 12, dtype=torch.float64).reshape(3, 4).requires_grad_()
    model.to(dtype=torch.float64)
    output = model(value)
    loss = output.square().mean()
    loss.backward()
    parameter_grads = {
        name.replace("._checkpoint_wrapped_module", ""): parameter.grad.detach().clone()
        for name, parameter in model.named_parameters()
    }
    return _TEST_OP_CALLS, output.detach(), loss.detach(), value.grad.detach().clone(), parameter_grads


def _toy_state_dict() -> dict[str, torch.Tensor]:
    torch.manual_seed(17)
    return _ToyTransformer().state_dict()


@pytest.mark.parametrize(
    ("checkpointing_type", "retain_test_op", "expected_calls"),
    [
        (None, True, 4),
        ("full", True, 8),
        ("ops", True, 4),
        ("ops", False, 8),
    ],
)
def test_checkpoint_policy_controls_expensive_op_recomputation(
    monkeypatch: pytest.MonkeyPatch,
    checkpointing_type: str | None,
    retain_test_op: bool,
    expected_calls: int,
) -> None:
    op_names = activation_checkpoint._SELECTIVE_ACTIVATION_CHECKPOINTING_OP_NAMES
    if retain_test_op:
        op_names = op_names | {_TEST_OP_NAME}
    monkeypatch.setattr(activation_checkpoint, "_SELECTIVE_ACTIVATION_CHECKPOINTING_OP_NAMES", op_names)

    calls, *_ = _run_toy_model(_toy_state_dict(), checkpointing_type)

    assert calls == expected_calls


def test_checkpoint_modes_preserve_outputs_and_gradients(monkeypatch: pytest.MonkeyPatch) -> None:
    op_names = activation_checkpoint._SELECTIVE_ACTIVATION_CHECKPOINTING_OP_NAMES | {_TEST_OP_NAME}
    monkeypatch.setattr(activation_checkpoint, "_SELECTIVE_ACTIVATION_CHECKPOINTING_OP_NAMES", op_names)
    state_dict = _toy_state_dict()

    baseline = _run_toy_model(state_dict, None)
    for checkpointing_type in ("full", "ops"):
        result = _run_toy_model(state_dict, checkpointing_type)
        torch.testing.assert_close(result[1], baseline[1], rtol=0, atol=0)
        torch.testing.assert_close(result[2], baseline[2], rtol=0, atol=0)
        torch.testing.assert_close(result[3], baseline[3], rtol=0, atol=0)
        assert result[4].keys() == baseline[4].keys()
        for name, baseline_grad in baseline[4].items():
            torch.testing.assert_close(result[4][name], baseline_grad, rtol=0, atol=0)


@pytest.mark.parametrize("checkpointing_type", ["full", "ops"])
def test_checkpointing_wraps_each_block_not_transformer_root(checkpointing_type: str) -> None:
    model = _ToyTransformer()

    result = apply_activation_checkpointing(model, checkpointing_type)

    assert result is model
    assert not isinstance(model, CheckpointWrapper)
    assert all(isinstance(block, CheckpointWrapper) for block in model.blocks)


@pytest.mark.parametrize("checkpointing_type", ["full", "ops"])
def test_checkpointing_rejects_transformers_without_known_block_lists(checkpointing_type: str) -> None:
    with pytest.raises(ValueError, match="Activation checkpointing is not applied successfully"):
        apply_activation_checkpointing(nn.Linear(4, 4), checkpointing_type)


def test_all_training_block_sparse_attention_ops_are_retained() -> None:
    block_sparse_attention = importlib.import_module("fastvideo_kernel.block_sparse_attn")
    training_op_names = {
        candidate._qualname
        for candidate in vars(block_sparse_attention).values()
        if getattr(candidate, "_qualname", "").startswith("fastvideo_kernel::block_sparse_attn_")
        and getattr(candidate, "_backward_fn", None) is not None
    }

    assert training_op_names
    missing_op_names = training_op_names - activation_checkpoint._SELECTIVE_ACTIVATION_CHECKPOINTING_OP_NAMES
    assert not missing_op_names, f"Training block-sparse attention ops missing from checkpoint save policy: {missing_op_names}"
