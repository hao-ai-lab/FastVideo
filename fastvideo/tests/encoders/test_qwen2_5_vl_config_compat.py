# SPDX-License-Identifier: Apache-2.0
"""Config-compat regression tests for the custom Qwen2.5-VL encoder.

transformers 5.x stopped forwarding text-model attributes (``vocab_size``,
``pad_token_id``, ``hidden_size``, ...) from the composite ``Qwen2_5_VLConfig``
down from ``text_config``, turned ``rope_scaling`` into a property backed by
``rope_parameters``, dropped the ``"default"`` entry from ``ROPE_INIT_FUNCTIONS``
and moved ``rope_theta`` inside ``rope_parameters``. See issue #1576.

``qwen2_5_vl_custom`` absorbs all of that. These tests pin both the config
normalization and the resulting RoPE numerics, so the compat layer cannot
silently regress as transformers evolves. They are CPU-only and weight-free:
the config is built inline rather than downloaded.
"""
import torch
from transformers import Qwen2_5_VLConfig

from fastvideo.models.encoders.qwen2_5_vl_custom import (
    Qwen2_5_VLRotaryEmbedding,
    _compute_default_rope_parameters,
    _flatten_text_config,
)

# Shape of Qwen/Qwen2.5-VL-7B-Instruct, the checkpoint Cosmos 2.5's Reason1 text encoder is built on.
HIDDEN_SIZE = 3584
NUM_ATTENTION_HEADS = 28
VOCAB_SIZE = 152064
ROPE_THETA = 1000000.0
MROPE_SECTION = [16, 24, 24]
HEAD_DIM = HIDDEN_SIZE // NUM_ATTENTION_HEADS  # 128


def _build_config() -> Qwen2_5_VLConfig:
    # Flat, 4.x-style kwargs: this is what a real checkpoint's config.json carries.
    return Qwen2_5_VLConfig(
        hidden_size=HIDDEN_SIZE,
        intermediate_size=18944,
        num_attention_heads=NUM_ATTENTION_HEADS,
        num_hidden_layers=28,
        num_key_value_heads=4,
        vocab_size=VOCAB_SIZE,
        max_position_embeddings=128000,
        rope_theta=ROPE_THETA,
        rope_scaling={"type": "mrope", "mrope_section": MROPE_SECTION},
        tie_word_embeddings=False,
    )


def _expected_inv_freq() -> torch.Tensor:
    return 1.0 / (ROPE_THETA**(torch.arange(0, HEAD_DIM, 2, dtype=torch.int64).float() / HEAD_DIM))


def test_flatten_text_config_exposes_text_attrs_on_composite_config() -> None:
    """Attributes the module reads off the top-level config must resolve after flattening."""
    config = _flatten_text_config(_build_config())

    assert config.vocab_size == VOCAB_SIZE
    assert config.hidden_size == HIDDEN_SIZE
    assert config.num_attention_heads == NUM_ATTENTION_HEADS
    assert config.max_position_embeddings == 128000
    # Present but legitimately None for this checkpoint; the module reads it via getattr.
    assert hasattr(config, "pad_token_id")


def test_flatten_text_config_preserves_mrope_settings() -> None:
    """``rope_scaling`` must survive the 4.x -> 5.x ``rope_parameters`` rename, mrope section intact."""
    config = _flatten_text_config(_build_config())

    rope_scaling = config.rope_scaling
    assert rope_scaling is not None
    assert rope_scaling["mrope_section"] == MROPE_SECTION

    # The "rope_type" vs legacy "type" lookup performed by Qwen2_5_VLRotaryEmbedding.
    # Both transformers lines normalize an "mrope" request to "default" before we see it:
    # 4.x rewrites rope_scaling["type"] in place, 5.x sets rope_parameters["rope_type"].
    # Asserting "mrope" here would fail on every supported version.
    rope_type = rope_scaling.get("rope_type", rope_scaling.get("type"))
    assert rope_type == "default"


def test_compute_default_rope_parameters_reads_theta_on_any_transformers() -> None:
    """``rope_theta`` moved inside ``rope_parameters`` in 5.x; the helper must find it either way."""
    config = _flatten_text_config(_build_config())

    inv_freq, attention_scaling = _compute_default_rope_parameters(config)

    assert attention_scaling == 1.0
    assert inv_freq.shape == (HEAD_DIM // 2, )
    torch.testing.assert_close(inv_freq, _expected_inv_freq())


def test_rotary_embedding_handles_config_without_rope_scaling() -> None:
    """``reason1.py`` falls back to a config carrying no ``rope_scaling``; it must still build.

    This exercises a different branch on each line: on 4.x ``rope_scaling`` is ``None`` and
    ``Qwen2_5_VLRotaryEmbedding`` takes its ``else``; on 5.x ``rope_parameters`` is populated
    with a bare ``rope_theta``/``rope_type`` pair. Both must land on the default init fn.
    """
    config = _flatten_text_config(
        Qwen2_5_VLConfig(
            hidden_size=HIDDEN_SIZE,
            intermediate_size=18944,
            max_window_layers=28,
            num_attention_heads=NUM_ATTENTION_HEADS,
            num_hidden_layers=28,
            num_key_value_heads=4,
            tie_word_embeddings=False,
            vocab_size=VOCAB_SIZE,
        ))

    rotary_embedding = Qwen2_5_VLRotaryEmbedding(config)

    assert rotary_embedding.rope_type == "default"
    assert rotary_embedding.inv_freq.shape == (HEAD_DIM // 2, )


def test_rotary_embedding_builds_from_flattened_config() -> None:
    """End-to-end: the rotary embedding must pick a valid init fn and produce 4.x-identical numerics."""
    config = _flatten_text_config(_build_config())

    rotary_embedding = Qwen2_5_VLRotaryEmbedding(config)

    assert rotary_embedding.rope_type == "default"
    assert rotary_embedding.attention_scaling == 1.0
    torch.testing.assert_close(rotary_embedding.inv_freq, _expected_inv_freq())
