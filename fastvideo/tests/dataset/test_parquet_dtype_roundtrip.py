# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np
import pytest
import torch

from fastvideo.dataset.dataloader.schema import pyarrow_schema_i2v
from fastvideo.dataset.utils import (
    _decode_tensor_bytes,
    collate_rows_from_parquet_schema,
    get_torch_tensors_from_row_dict,
)


def _serialized_fields(name: str, value: np.ndarray) -> dict[str, object]:
    return {
        f"{name}_bytes": value.tobytes(),
        f"{name}_shape": list(value.shape),
        f"{name}_dtype": str(value.dtype),
    }


def test_i2v_collate_respects_each_serialized_tensor_dtype() -> None:
    vae_latent = np.arange(16 * 2 * 2 * 2, dtype=np.float32).reshape(16, 2, 2, 2)
    text_embedding = np.arange(3 * 4, dtype=np.float32).reshape(3, 4)
    clip_feature = np.arange(2 * 4, dtype=np.float16).reshape(2, 4)
    first_frame_latent = np.arange(16 * 2 * 2 * 2, dtype=np.float32).reshape(16, 2, 2, 2)
    pil_image = np.arange(3 * 2 * 4, dtype=np.uint8).reshape(3, 2, 4)

    row: dict[str, object] = {
        "id": "sample",
        "file_name": "sample.mp4",
        "caption": "sample caption",
        "media_type": "video",
        "width": 4,
        "height": 2,
        "num_frames": 5,
        "duration_sec": 1.0,
        "fps": 5.0,
    }
    for name, value in (
        ("vae_latent", vae_latent),
        ("text_embedding", text_embedding),
        ("clip_feature", clip_feature),
        ("first_frame_latent", first_frame_latent),
        ("pil_image", pil_image),
    ):
        row.update(_serialized_fields(name, value))

    batch = collate_rows_from_parquet_schema(
        [row],
        pyarrow_schema_i2v,
        text_padding_length=4,
    )

    assert batch["vae_latent"].dtype == torch.float32
    assert batch["clip_feature"].dtype == torch.float16
    assert batch["first_frame_latent"].dtype == torch.float32
    assert batch["pil_image"].dtype == torch.uint8
    torch.testing.assert_close(batch["clip_feature"][0], torch.from_numpy(clip_feature))
    torch.testing.assert_close(batch["pil_image"][0], torch.from_numpy(pil_image))


def test_decode_tensor_bytes_accepts_torch_bfloat16_dtype() -> None:
    expected = torch.tensor([[1.0, -2.5], [3.25, 0.0]], dtype=torch.bfloat16)
    storage = expected.view(torch.uint16).numpy()

    actual = _decode_tensor_bytes(storage.tobytes(), list(expected.shape), "torch.bfloat16")

    assert actual.dtype == torch.bfloat16
    torch.testing.assert_close(actual, expected)


def test_decode_tensor_bytes_zero_returns_typed_zeros() -> None:
    expected = torch.tensor([[1.0, -2.5], [3.25, 0.0]], dtype=torch.bfloat16)
    storage = expected.view(torch.uint16).numpy()

    actual = _decode_tensor_bytes(storage.tobytes(), list(expected.shape), "torch.bfloat16", zero=True)

    assert actual.dtype == torch.bfloat16
    assert actual.shape == expected.shape
    assert actual.count_nonzero() == 0


def test_collate_cfg_drop_zeros_text_embedding_with_serialized_dtype() -> None:
    text_embedding = np.arange(3 * 4, dtype=np.float16).reshape(3, 4)
    row = _serialized_fields("text_embedding", text_embedding)
    row["caption"] = "sample caption"

    batch = collate_rows_from_parquet_schema(
        [row],
        pyarrow_schema_i2v,
        text_padding_length=4,
        cfg_rate=1.0,
    )

    assert batch["text_embedding"].dtype == torch.float16
    assert batch["text_embedding"].count_nonzero() == 0


def test_decode_tensor_bytes_rejects_mismatched_byte_length() -> None:
    with pytest.raises(ValueError, match="byte length"):
        _decode_tensor_bytes(b"\x00\x01", [2], "float32")


def test_legacy_collate_helper_respects_dtype_for_alias_keys() -> None:
    latent = np.arange(4, dtype=np.float16).reshape(2, 2)
    row = _serialized_fields("latent", latent)

    tensors = get_torch_tensors_from_row_dict(
        row,
        [("vae_latent", "latent")],
        cfg_rate=0.0,
    )

    assert tensors["vae_latent"].dtype == torch.float16
    torch.testing.assert_close(tensors["vae_latent"], torch.from_numpy(latent))
