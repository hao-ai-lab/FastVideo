# SPDX-License-Identifier: Apache-2.0
"""Weight-free regression tests for reason1's chat-template id normalization.

Guards the fix for the Cosmos-Predict2.5 text-encoder crash: `apply_chat_template`
returns a `BatchEncoding` (a `collections.UserDict`, i.e. a `Mapping` but NOT a
`dict`), so an `isinstance(_, dict)` check missed it and raised. These cover every
shape `apply_chat_template` can return, so a future transformers change can't
silently reintroduce the crash.
"""

from collections import UserDict

import pytest

from fastvideo.models.encoders.reason1 import _normalize_chat_template_ids

IDS = [10, 11, 12]


class _FakeTensor:
    """Stand-in for a torch tensor: exposes ``.tolist()`` like real output does."""

    def __init__(self, value):
        self._value = value

    def tolist(self):
        return self._value


class _BatchEncodingLike(UserDict):
    """Mirrors ``transformers.BatchEncoding`` (``class BatchEncoding(UserDict)``):
    a Mapping that is deliberately NOT a ``dict`` subclass."""


@pytest.mark.parametrize(
    "tokenizer_output",
    [
        IDS,                                  # list[int]  (legacy list path)
        [IDS],                                # nested list[list[int]] (batch dim)
        _FakeTensor(IDS),                     # tensor
        _FakeTensor([IDS]),                   # batched tensor
        {"input_ids": IDS},                   # plain dict, flat
        {"input_ids": [IDS]},                 # plain dict, batched
        _BatchEncodingLike({"input_ids": IDS}),            # BatchEncoding, flat
        _BatchEncodingLike({"input_ids": [IDS]}),          # BatchEncoding, batched
        _BatchEncodingLike({"input_ids": _FakeTensor([IDS])}),  # the crash case
    ],
)
def test_normalizes_every_shape_to_flat_ids(tokenizer_output):
    assert _normalize_chat_template_ids(tokenizer_output) == IDS


def test_batch_encoding_is_not_a_dict_but_is_handled():
    # Documents the root cause: the old `isinstance(_, dict)` guard was False here.
    be = _BatchEncodingLike({"input_ids": IDS})
    assert not isinstance(be, dict)
    assert _normalize_chat_template_ids(be) == IDS


def test_unexpected_type_raises():
    with pytest.raises(RuntimeError, match="Unexpected chat_template output type"):
        _normalize_chat_template_ids(object())


def test_real_batch_encoding_if_available():
    # Belt-and-suspenders: exercise the actual transformers BatchEncoding when present.
    pytest.importorskip("transformers")
    from transformers.tokenization_utils_base import BatchEncoding
    assert not isinstance(BatchEncoding(), dict)  # the property the fix relies on
    assert _normalize_chat_template_ids(BatchEncoding({"input_ids": IDS})) == IDS
