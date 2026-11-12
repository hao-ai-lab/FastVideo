# SPDX-License-Identifier: Apache-2.0
"""The negative-prompt text encoder must be freed as soon as encoding is done."""
import contextlib
import gc
import types
import weakref
from unittest import mock

import torch

from fastvideo.train.utils import negative_prompt
from fastvideo.train.utils.negative_prompt import _release_encoder_memory


class _Cyclic(torch.nn.Module):
    """A layer whose parameter holds the layer's bound weight loader, as ``set_weight_attrs`` does."""

    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(4))
        self.weight.weight_loader = self.weight_loader

    def weight_loader(self, param: torch.nn.Parameter, loaded_weight: torch.Tensor) -> None:
        param.data.copy_(loaded_weight)


def test_release_frees_cyclic_encoder_immediately() -> None:
    gc.collect()
    gc.disable()
    try:
        module = _Cyclic()
        ref = weakref.ref(module.weight)
        del module
        assert ref() is not None  # without a collection the cycle keeps the weights alive
        _release_encoder_memory()
        assert ref() is None
    finally:
        gc.enable()


def test_encode_negative_prompt_releases_the_encoder(monkeypatch) -> None:
    """The release must run from the call site, not only when the helper is invoked by hand."""
    released: list[bool] = []
    monkeypatch.setattr(negative_prompt, "_release_encoder_memory", lambda: released.append(True))
    monkeypatch.setattr(negative_prompt, "maybe_download_model", lambda model_path: "/tmp/model")
    monkeypatch.setattr(negative_prompt, "make_inference_args", lambda tc, model_path: types.SimpleNamespace())
    monkeypatch.setattr(negative_prompt, "set_forward_context", lambda **kwargs: contextlib.nullcontext())

    encoder = mock.MagicMock()
    loader = mock.MagicMock()
    loader.load.return_value.to.return_value.eval.return_value = encoder
    monkeypatch.setattr(negative_prompt, "TextEncoderLoader", mock.MagicMock(return_value=loader))

    tokenizer = mock.MagicMock()
    tokenizer_cls = mock.MagicMock()
    tokenizer_cls.from_pretrained.return_value = tokenizer
    monkeypatch.setattr(negative_prompt, "AutoTokenizer", tokenizer_cls)

    pipeline_config = types.SimpleNamespace(
        text_encoder_configs=[types.SimpleNamespace(tokenizer_kwargs={})],
        postprocess_text_funcs=[lambda outputs: outputs.hidden_states],
    )
    training_config = types.SimpleNamespace(pipeline_config=pipeline_config, model_path="/tmp/model")

    negative_prompt.encode_negative_prompt(
        training_config,
        prompt="",
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    assert released == [True]
