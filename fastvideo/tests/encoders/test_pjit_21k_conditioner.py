# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from torch import nn

from fastvideo.models.encoders.pjit_21k import PJiTTextConditioner


class _Encoder(nn.Module):

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1))
        self.config = SimpleNamespace(layer_types=["full_attention"] * 28)

    def forward(self, **kwargs):
        assert kwargs["output_hidden_states"] is True
        states = tuple(torch.full((2, 4, 2), float(index)) for index in range(29))
        return SimpleNamespace(hidden_states=states)


def test_pjit_conditioner_uses_trained_template_layers_and_padding():
    tokenizer = Mock()
    tokenizer.apply_chat_template.side_effect = ["formatted short", "formatted long"]
    tokenizer.return_value = {
        "input_ids": torch.tensor([[1, 2, 0, 0], [1, 2, 3, 4]]),
        "attention_mask": torch.tensor([[1, 1, 0, 0], [1, 1, 1, 1]]),
    }
    conditioner = PJiTTextConditioner(_Encoder(), tokenizer)
    actual = conditioner(["short", "long"])

    for caption, call in zip(("short", "long"), tokenizer.apply_chat_template.call_args_list, strict=True):
        assert call.args == ([{"role": "user", "content": caption}], )
        assert call.kwargs == {"tokenize": False, "add_generation_prompt": True, "enable_thinking": False}
    tokenizer.assert_called_once_with(
        ["formatted short", "formatted long"], return_tensors="pt", padding=True, truncation=True, max_length=256)
    assert [tuple(row.shape) for row in actual] == [(2, 6), (4, 6)]
    for row in actual:
        assert row.dtype == torch.bfloat16
        torch.testing.assert_close(row, torch.tensor([8, 8, 16, 16, 28, 28], dtype=torch.bfloat16).expand_as(row))
    assert conditioner([]) == []


@pytest.mark.parametrize("layers", [(-1, ), (28, ), (0, )])
def test_pjit_conditioner_rejects_invalid_or_linear_attention_layers(layers):
    encoder = _Encoder()
    encoder.config.layer_types[0] = "linear_attention"
    with pytest.raises(ValueError, match="not a full-attention layer"):
        PJiTTextConditioner(encoder, Mock(), extraction_layers=layers)
