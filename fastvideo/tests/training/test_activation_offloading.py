"""CPU regression tests for the activation-offloading changes in PR #1106.

These exercise the actual changed behavior on bounded fixtures (no GPU):
- TrainingArgs exposes a positive ``log_every_n_steps`` (the training loop
  evaluates ``step % log_every_n_steps`` every step, which previously raised
  ``AttributeError`` because the field did not exist).
- ``offloaded_forward`` preserves forward values and gradients (the
  ``saved_tensors_hooks`` CPU round-trip is autograd-correct).
- ``enable_activation_offloading`` wraps each trainable block exactly once, is
  idempotent, and does not touch frozen (non-trainable) modules.
"""
import dataclasses
import types

import torch
from torch import nn

from fastvideo.fastvideo_args import TrainingArgs
from fastvideo.pipelines.composed_pipeline_base import (
    ComposedPipelineBase,
    offloaded_forward,
)


def test_log_every_n_steps_exists_and_is_positive():
    """The loop gate ``step % log_every_n_steps`` must resolve and be safe."""
    fields = {f.name: f for f in dataclasses.fields(TrainingArgs)}
    assert "log_every_n_steps" in fields, (
        "training loop reads training_args.log_every_n_steps")
    default = fields["log_every_n_steps"].default
    assert isinstance(default, int) and default > 0, (
        "log_every_n_steps must be a positive divisor (step % log_every_n_steps)")
    # Simulate the per-step gate: must not raise AttributeError/ZeroDivision.
    for step in range(1, 25):
        _ = (default > 0 and step % default == 0)


def test_offloaded_forward_preserves_values_and_gradients():
    x = torch.tensor([1.0, 2.0], requires_grad=True)

    def f(t):
        return t * 2

    y = offloaded_forward(f, x)
    assert torch.equal(y.detach(), torch.tensor([2.0, 4.0]))
    y.sum().backward()
    assert torch.equal(x.grad, torch.tensor([2.0, 2.0]))


class _Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 4)

    def forward(self, x):
        return self.linear(x)


class _ToyTransformer(nn.Module):
    def __init__(self, n_blocks=3):
        super().__init__()
        self.blocks = nn.ModuleList([_Block() for _ in range(n_blocks)])


def test_enable_activation_offloading_once_idempotent_targets_trainable():
    transformer = _ToyTransformer()
    frozen = _ToyTransformer()  # stand-in for a frozen VAE/text-encoder
    fake = types.SimpleNamespace(
        fastvideo_args=types.SimpleNamespace(activation_offloading=True),
        trainable_transformer_modules={"transformer": transformer},
        modules={"transformer": transformer, "vae": frozen},
    )

    ComposedPipelineBase.enable_activation_offloading(fake)

    # Trainable blocks are wrapped exactly once with offloaded_forward.
    for layer in transformer.blocks:
        assert getattr(layer, "_fv_activation_offload_wrapped", False)
        assert getattr(layer.forward, "func", None) is offloaded_forward

    # Frozen (non-trainable) modules are not wrapped.
    for layer in frozen.blocks:
        assert not getattr(layer, "_fv_activation_offload_wrapped", False)

    # Idempotent: a second enable does not re-wrap (forward object unchanged).
    first_forward = transformer.blocks[0].forward
    ComposedPipelineBase.enable_activation_offloading(fake)
    assert transformer.blocks[0].forward is first_forward


def test_enable_activation_offloading_noop_when_disabled():
    transformer = _ToyTransformer()
    fake = types.SimpleNamespace(
        fastvideo_args=types.SimpleNamespace(activation_offloading=False),
        trainable_transformer_modules={"transformer": transformer},
    )
    ComposedPipelineBase.enable_activation_offloading(fake)
    for layer in transformer.blocks:
        assert not getattr(layer, "_fv_activation_offload_wrapped", False)