# SPDX-License-Identifier: Apache-2.0
"""Trainable H3 gates must remain live during no-grad rollout forwards."""

from types import SimpleNamespace

import pytest
import torch

from fastvideo.models.dits.minimax_h3 import MiniMaxH3Attention
from fastvideo.train.models.minimax_h3.minimax_h3 import MiniMaxH3Model


@pytest.mark.parametrize("trainable", [False, True])
def test_loaded_role_keeps_gate_live_after_update_without_changing_inference_cache(monkeypatch, trainable):
    attention = MiniMaxH3Attention.__new__(MiniMaxH3Attention)
    torch.nn.Module.__init__(attention)
    attention.to_gate_compress = torch.nn.Linear(2, 2, bias=False)
    attention._gate_compress_active = None
    attention._gate_compress_training = False
    with torch.no_grad():
        attention.to_gate_compress.weight.zero_()
        assert attention._gate_active() is False
    transformer = torch.nn.Module()
    transformer.add_module("attention", attention)
    monkeypatch.setattr("fastvideo.train.models.minimax_h3.minimax_h3.load_module_from_path", lambda **_: transformer)

    model = MiniMaxH3Model.__new__(MiniMaxH3Model)
    model.training_config = SimpleNamespace()
    model._init_from = "unused"
    model._lora_config = None
    model.attention_backend = None
    loaded = model._load_transformer(
        trainable=trainable,
        disable_custom_init_weights=False,
        enable_gradient_checkpointing_type=None,
        transformer_override_safetensor=None,
    )
    assert loaded is transformer
    assert attention._gate_compress_training is trainable
    with torch.no_grad():
        assert attention._gate_active() is trainable
        if trainable:
            attention.to_gate_compress.weight.add_(1)
            assert attention._gate_active() is True
            assert torch.equal(attention.to_gate_compress(torch.ones(1, 2)), torch.full((1, 2), 2.0))
        # The inference cache is never rewritten in a compiled/rollout forward.
        assert attention._gate_compress_active is False
