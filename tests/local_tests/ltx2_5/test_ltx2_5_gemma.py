# SPDX-License-Identifier: Apache-2.0
"""Focused compatibility tests for the LTX-2.5 packed Gemma 4 path."""

from types import SimpleNamespace

import torch

from fastvideo.models.dits.ltx2 import LTXRopeType
from fastvideo.models.encoders.gemma import (
    Embeddings1DConnector,
    GemmaConnectorConfig,
    _ensure_leading_bos,
    _get_bos_token_id,
)


def test_gemma4_bos_token_id_comes_from_nested_text_config() -> None:
    config = SimpleNamespace(text_config=SimpleNamespace(bos_token_id=2))
    assert _get_bos_token_id(config) == 2


def test_ensure_leading_bos_preserves_gemma3_and_fixes_gemma4() -> None:
    input_ids = torch.tensor([
        [0, 0, 2, 10, 11],
        [0, 0, 0, 10, 11],
        [10, 11, 12, 13, 14],
    ])
    attention_mask = torch.tensor([
        [0, 0, 1, 1, 1],
        [0, 0, 0, 1, 1],
        [1, 1, 1, 1, 1],
    ])

    actual_ids, actual_mask = _ensure_leading_bos(
        input_ids,
        attention_mask,
        bos_token_id=2,
    )

    torch.testing.assert_close(actual_ids[0], input_ids[0])
    torch.testing.assert_close(actual_mask[0], attention_mask[0])
    torch.testing.assert_close(actual_ids[1], torch.tensor([0, 0, 2, 10, 11]))
    torch.testing.assert_close(actual_mask[1], torch.tensor([0, 0, 1, 1, 1]))
    torch.testing.assert_close(actual_ids[2], torch.tensor([2, 10, 11, 12, 13]))
    torch.testing.assert_close(actual_mask[2], torch.ones(5, dtype=torch.long))


def test_ltx2_5_connector_ffn_can_omit_bias() -> None:
    connector = Embeddings1DConnector(
        GemmaConnectorConfig(
            num_attention_heads=2,
            attention_head_dim=4,
            num_layers=1,
            positional_embedding_theta=10000.0,
            positional_embedding_max_pos=[4096],
            rope_type=LTXRopeType.SPLIT,
            double_precision_rope=False,
            num_learnable_registers=None,
            apply_gated_attention=False,
            ff_bias=False,
        ))

    feed_forward = connector.transformer_1d_blocks[0].ff
    assert feed_forward.net[0].proj.bias is None
    assert feed_forward.net[2].bias is None
