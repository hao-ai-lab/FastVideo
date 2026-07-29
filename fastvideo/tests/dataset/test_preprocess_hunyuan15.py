# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch

import fastvideo.pipelines.preprocess.preprocess_hunyuan15_overfit as preprocess


QWEN_DIM = 3584
BYT5_DIM = 1472


def test_tensor_to_record_preserves_values_and_shape() -> None:
    tensor = torch.arange(
        3 * 4,
        dtype=torch.float32,
    ).reshape(3, 4)

    record = preprocess.tensor_to_record(
        tensor,
        "text_embedding",
    )

    assert record["text_embedding_shape"] == [3, 4]
    assert record["text_embedding_dtype"] == "float32"

    decoded = np.frombuffer(
        record["text_embedding_bytes"],
        dtype=np.float32,
    ).reshape(record["text_embedding_shape"])

    np.testing.assert_array_equal(
        decoded,
        tensor.numpy(),
    )


def test_tensor_to_record_converts_input_to_float32() -> None:
    tensor = torch.ones(
        (2, 4),
        dtype=torch.bfloat16,
    )

    record = preprocess.tensor_to_record(
        tensor,
        "text_embedding",
    )

    assert record["text_embedding_dtype"] == "float32"

    decoded = np.frombuffer(
        record["text_embedding_bytes"],
        dtype=np.float32,
    )

    assert decoded.dtype == np.float32
    assert decoded.size == 8


def test_empty_byt5_tensor_serializes_as_empty_bytes() -> None:
    tensor = torch.empty(
        (0, BYT5_DIM),
        dtype=torch.float32,
    )

    record = preprocess.tensor_to_record(
        tensor,
        "text_embedding_2",
    )

    assert record["text_embedding_2_shape"] == [0, BYT5_DIM]
    assert record["text_embedding_2_dtype"] == "float32"
    assert record["text_embedding_2_bytes"] == b""

    decoded = np.frombuffer(
        record["text_embedding_2_bytes"],
        dtype=np.float32,
    ).reshape(record["text_embedding_2_shape"])

    assert decoded.shape == (0, BYT5_DIM)


def test_encode_byt5_returns_empty_embedding_without_glyph_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Captions without quoted text must not invoke the ByT5 model."""

    monkeypatch.setattr(
        preprocess,
        "byt5_preprocess_text",
        lambda caption: None,
    )

    tokenizer = Mock()
    encoder = Mock()
    encoder.config = SimpleNamespace(
        d_model=BYT5_DIM,
    )

    result = preprocess.encode_byt5_caption(
        caption="A dog runs across a grassy field.",
        tokenizer=tokenizer,
        encoder=encoder,
        device=torch.device("cpu"),
        max_length=256,
    )

    assert result.shape == (0, BYT5_DIM)
    assert result.dtype == torch.float32
    assert result.device.type == "cpu"

    tokenizer.assert_not_called()
    encoder.assert_not_called()


def test_encode_byt5_encodes_extracted_glyph_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only the extracted glyph prompt should be sent to ByT5."""

    glyph_text = "OPEN"

    monkeypatch.setattr(
        preprocess,
        "byt5_preprocess_text",
        lambda caption: glyph_text,
    )

    monkeypatch.setattr(
        preprocess,
        "byt5_postprocess_text",
        lambda outputs: outputs.last_hidden_state,
    )

    tokenizer = Mock(
        return_value={
            "input_ids": torch.tensor(
                [[10, 11, 12]],
                dtype=torch.long,
            ),
            "attention_mask": torch.tensor(
                [[1, 1, 1]],
                dtype=torch.long,
            ),
        }
    )

    encoder = Mock()
    encoder.config = SimpleNamespace(
        d_model=BYT5_DIM,
    )
    encoder.return_value = SimpleNamespace(
        last_hidden_state=torch.full(
            (1, 3, BYT5_DIM),
            2.0,
            dtype=torch.float32,
        )
    )

    result = preprocess.encode_byt5_caption(
        caption='A storefront sign reads "OPEN".',
        tokenizer=tokenizer,
        encoder=encoder,
        device=torch.device("cpu"),
        max_length=256,
    )

    tokenizer.assert_called_once_with(
        glyph_text,
        add_special_tokens=True,
        padding=False,
        truncation=True,
        max_length=256,
        return_tensors="pt",
    )

    encoder.assert_called_once()

    assert result.shape == (3, BYT5_DIM)
    assert result.dtype == torch.float32
    assert torch.all(result == 2.0)


def test_encode_byt5_trims_padding_tokens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        preprocess,
        "byt5_preprocess_text",
        lambda caption: "OPEN",
    )

    monkeypatch.setattr(
        preprocess,
        "byt5_postprocess_text",
        lambda outputs: outputs.last_hidden_state,
    )

    tokenizer = Mock(
        return_value={
            "input_ids": torch.tensor(
                [[10, 11, 12, 0, 0]],
                dtype=torch.long,
            ),
            "attention_mask": torch.tensor(
                [[1, 1, 1, 0, 0]],
                dtype=torch.long,
            ),
        }
    )

    encoder = Mock()
    encoder.config = SimpleNamespace(
        d_model=BYT5_DIM,
    )
    encoder.return_value = SimpleNamespace(
        last_hidden_state=torch.randn(
            1,
            5,
            BYT5_DIM,
        )
    )

    result = preprocess.encode_byt5_caption(
        caption='The sign reads "OPEN".',
        tokenizer=tokenizer,
        encoder=encoder,
        device=torch.device("cpu"),
        max_length=256,
    )

    assert result.shape == (3, BYT5_DIM)


class _FakeDistributionWithMode:

    def __init__(self, latent: torch.Tensor) -> None:
        self._latent = latent

    def mode(self) -> torch.Tensor:
        return self._latent


class _FakeDistributionWithMean:

    def __init__(self, latent: torch.Tensor) -> None:
        self.mean = latent


class _FakeVAE:

    def __init__(self, encoded_output: object) -> None:
        self.encoded_output = encoded_output

    def encode(self, video: torch.Tensor) -> object:
        del video
        return self.encoded_output


def test_encode_video_latent_accepts_direct_distribution() -> None:
    latent = torch.randn(
        1,
        32,
        2,
        4,
        4,
        dtype=torch.float16,
    )

    vae = _FakeVAE(
        _FakeDistributionWithMode(latent)
    )

    result = preprocess.encode_video_latent(
        vae=vae,
        video=torch.empty(1),
    )

    assert result.shape == (32, 2, 4, 4)
    assert result.dtype == torch.float32
    assert result.device.type == "cpu"

    torch.testing.assert_close(
        result,
        latent.squeeze(0).float(),
    )


def test_encode_video_latent_accepts_latent_dist_wrapper() -> None:
    latent = torch.randn(
        1,
        32,
        2,
        4,
        4,
    )

    encoded = SimpleNamespace(
        latent_dist=_FakeDistributionWithMode(latent),
    )

    vae = _FakeVAE(encoded)

    result = preprocess.encode_video_latent(
        vae=vae,
        video=torch.empty(1),
    )

    torch.testing.assert_close(
        result,
        latent.squeeze(0).float(),
    )


def test_encode_video_latent_falls_back_to_mean() -> None:
    latent = torch.randn(
        1,
        32,
        2,
        4,
        4,
    )

    vae = _FakeVAE(
        _FakeDistributionWithMean(latent)
    )

    result = preprocess.encode_video_latent(
        vae=vae,
        video=torch.empty(1),
    )

    torch.testing.assert_close(
        result,
        latent.squeeze(0).float(),
    )


def test_encode_video_latent_rejects_unknown_output() -> None:
    vae = _FakeVAE(object())

    with pytest.raises(
        TypeError,
        match="Unsupported VAE encode output type",
    ):
        preprocess.encode_video_latent(
            vae=vae,
            video=torch.empty(1),
        )


def test_encode_video_latent_does_not_apply_scaling_factor() -> None:
    """Scaling is applied by the training side, not preprocessing."""

    latent = torch.full(
        (1, 32, 2, 4, 4),
        2.0,
    )

    vae = _FakeVAE(
        _FakeDistributionWithMode(latent)
    )

    result = preprocess.encode_video_latent(
        vae=vae,
        video=torch.empty(1),
    )

    assert torch.all(result == 2.0)