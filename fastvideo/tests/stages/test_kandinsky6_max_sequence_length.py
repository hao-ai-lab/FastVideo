# SPDX-License-Identifier: Apache-2.0
"""Regression tests for ``max_sequence_length`` semantics against the diffusers reference
(``pipeline_kandinsky6_ti2va.py``): a request's ``max_sequence_length`` maps to the Qwen encoder only
(``max_length = max_sequence_length + 129``, the prompt-template prefix crop) -- not, as the generic
``TextEncodingStage`` does by default, to every encoder uniformly. Applying it to the fixed-77-position
CLIP tokenizer too (the old behavior) makes ``padding="max_length"`` on a length bigger than CLIP's own
77 positions crash.

Also proves the shared hook this needed in ``TextEncodingStage`` (``_resolve_max_length`` +
``encode_text``'s per-encoder ``max_length`` sequence) leaves every *other* model's behavior identical:
a plain int or ``None`` (everyone else's call shape) still applies uniformly, unchanged.

Pure CPU harness: fake tokenizers/encoders record their kwargs instead of really tokenizing/encoding,
so no weights are needed.
"""
from __future__ import annotations

import types

import pytest
import torch

from fastvideo.configs.pipelines.kandinsky6 import KANDINSKY6_PROMPT_TEMPLATE_ENCODE_START_IDX
from fastvideo.pipelines.stages import text_encoding as text_encoding_module
from fastvideo.pipelines.stages.kandinsky6 import Kandinsky6TextEncodingStage
from fastvideo.pipelines.stages.text_encoding import TextEncodingStage


@pytest.fixture(autouse=True)
def _cpu_device(monkeypatch):
    # get_local_torch_device() falls back to "mps" without CUDA (see
    # fastvideo/distributed/parallel_state.py); force plain CPU tensors so this test does not depend
    # on MPS being usable in the sandbox that runs it.
    monkeypatch.setattr(text_encoding_module, "get_local_torch_device", lambda: torch.device("cpu"))


class _TensorDict(dict):

    def to(self, device):
        return self


class _FakeTokenizer:

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def __call__(self, texts, **kwargs):
        self.calls.append(dict(kwargs))
        seq_len = int(kwargs.get("max_length", 8))
        n = len(texts)
        return _TensorDict(input_ids=torch.zeros(n, seq_len, dtype=torch.long),
                           attention_mask=torch.ones(n, seq_len, dtype=torch.long))


class _FakeTextEncoder(torch.nn.Module):

    def __init__(self, hidden_size: int = 4) -> None:
        super().__init__()
        self.hidden_size = hidden_size

    def forward(self, input_ids, attention_mask, output_hidden_states=False):
        b, s = input_ids.shape
        return types.SimpleNamespace(last_hidden_state=torch.zeros(b, s, self.hidden_size), hidden_states=None,
                                     pooler_output=torch.zeros(b, self.hidden_size))


def _encoder_config(default_max_length: int) -> types.SimpleNamespace:
    return types.SimpleNamespace(
        tokenizer_kwargs={"max_length": default_max_length},
        is_chat_model=False,
        hidden_size=4,
        arch_config=types.SimpleNamespace(output_hidden_states=False),
    )


def _fastvideo_args(qwen_default: int, clip_default: int) -> types.SimpleNamespace:
    return types.SimpleNamespace(
        pipeline_config=types.SimpleNamespace(
            text_encoder_configs=[_encoder_config(qwen_default), _encoder_config(clip_default)],
            preprocess_text_funcs=[lambda s: s, lambda s: s],
            postprocess_text_funcs=[lambda o: o.last_hidden_state, lambda o: o.pooler_output],
            dit_config=types.SimpleNamespace(prefix="kandinsky6"),
            text_encoder_max_lengths=(qwen_default, clip_default),
        ),
        text_encoder_cpu_offload=False,
    )


def _make_stage(cls) -> tuple[TextEncodingStage, list[_FakeTokenizer], list[_FakeTextEncoder]]:
    tokenizers = [_FakeTokenizer(), _FakeTokenizer()]
    encoders = [_FakeTextEncoder(), _FakeTextEncoder()]
    stage = cls(text_encoders=encoders, tokenizers=tokenizers)
    return stage, tokenizers, encoders


# --------------------------------------------------------------------------------------------------
# Kandinsky6TextEncodingStage._resolve_max_length
# --------------------------------------------------------------------------------------------------


def test_resolve_max_length_is_none_when_the_request_does_not_set_one():
    stage, _, _ = _make_stage(Kandinsky6TextEncodingStage)
    batch = types.SimpleNamespace(max_sequence_length=None)

    assert stage._resolve_max_length(batch, fastvideo_args=None) is None


def test_resolve_max_length_maps_to_qwen_only_leaving_clip_at_its_own_default():
    stage, _, _ = _make_stage(Kandinsky6TextEncodingStage)
    batch = types.SimpleNamespace(max_sequence_length=1024)

    resolved = stage._resolve_max_length(batch, fastvideo_args=None)

    assert resolved == [KANDINSKY6_PROMPT_TEMPLATE_ENCODE_START_IDX + 1024, None]
    assert resolved[0] == 1153


def test_default_text_encoding_stage_resolve_max_length_is_unchanged_passthrough():
    # Every other model family uses the base TextEncodingStage: the hook's default implementation
    # must be exactly the old inlined behavior (`batch.max_sequence_length`, applied uniformly by
    # encode_text below) -- proof that adding the hook did not change anyone else's semantics.
    stage, _, _ = _make_stage(TextEncodingStage)
    for value in (None, 77, 1024):
        batch = types.SimpleNamespace(max_sequence_length=value)
        assert stage._resolve_max_length(batch, fastvideo_args=None) == value


# --------------------------------------------------------------------------------------------------
# encode_text: per-encoder override vs. the old uniform-int/None behavior
# --------------------------------------------------------------------------------------------------


def test_qwen_gets_the_overridden_length_and_clip_keeps_its_own_fixed_77():
    stage, tokenizers, _ = _make_stage(TextEncodingStage)
    fastvideo_args = _fastvideo_args(qwen_default=641, clip_default=77)

    stage.encode_text("a cat", fastvideo_args, encoder_index=[0, 1], max_length=[1153, None],
                      device=torch.device("cpu"))

    assert tokenizers[0].calls[-1]["max_length"] == 1153
    # GAP: the old code applied the same override to every encoder, which would have set CLIP's
    # max_length to 1153 too -- padding="max_length" with only 77 positions crashes for real.
    assert tokenizers[1].calls[-1]["max_length"] == 77


def test_no_override_falls_back_to_each_encoders_configured_default():
    stage, tokenizers, _ = _make_stage(TextEncodingStage)
    fastvideo_args = _fastvideo_args(qwen_default=641, clip_default=77)

    stage.encode_text("a cat", fastvideo_args, encoder_index=[0, 1], max_length=None, device=torch.device("cpu"))

    assert tokenizers[0].calls[-1]["max_length"] == 641
    assert tokenizers[1].calls[-1]["max_length"] == 77


def test_a_plain_int_max_length_still_applies_uniformly_to_every_encoder():
    # Regression: every non-Kandinsky6 model passes a single int (or None) as `max_length`, never a
    # sequence -- that call shape must keep behaving exactly as before.
    stage, tokenizers, _ = _make_stage(TextEncodingStage)
    fastvideo_args = _fastvideo_args(qwen_default=641, clip_default=77)

    stage.encode_text("a cat", fastvideo_args, encoder_index=[0, 1], max_length=999, device=torch.device("cpu"))

    assert tokenizers[0].calls[-1]["max_length"] == 999
    assert tokenizers[1].calls[-1]["max_length"] == 999


def test_kandinsky6_text_encoding_stage_forward_end_to_end_respects_the_split():
    stage, tokenizers, _ = _make_stage(Kandinsky6TextEncodingStage)
    fastvideo_args = _fastvideo_args(qwen_default=641, clip_default=77)
    batch = types.SimpleNamespace(
        prompt="a cat",
        max_sequence_length=1024,
        prompt_embeds=[],
        prompt_attention_mask=[],
        do_classifier_free_guidance=False,
        negative_prompt_embeds=None,
        negative_attention_mask=None,
        extra={},
    )

    out = stage.forward(batch, fastvideo_args)

    assert len(out.prompt_embeds) == 2
    assert tokenizers[0].calls[-1]["max_length"] == 1153
    assert tokenizers[1].calls[-1]["max_length"] == 77
