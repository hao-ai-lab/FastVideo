"""Two-layer MLP as an ``nn.Sequential`` subclass — checkpoint-key compatible.

Extracted from the vendored official Wan2.1 model (@ 9737cba): the official
model builds its FFN / text / time embeddings as anonymous Sequentials, so
their checkpoint keys are integer-indexed (``ffn.0.weight``, ``ffn.2.weight``,
``time_embedding.0.weight`` ...). Subclassing Sequential with the same three
slots (0 = in-proj, 1 = activation, 2 = out-proj) keeps every official
checkpoint loading with no key mapping. Do not convert to named attributes.
"""
from __future__ import annotations

import torch.nn as nn

_ACTS = {
    "gelu_tanh": lambda: nn.GELU(approximate="tanh"),
    "silu": nn.SiLU,
}


class MLP(nn.Sequential):

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int | None = None,
                 act: str = "gelu_tanh"):
        super().__init__(nn.Linear(in_dim, hidden_dim), _ACTS[act](),
                         nn.Linear(hidden_dim, out_dim if out_dim is not None else in_dim))
