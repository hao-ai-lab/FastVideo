# SPDX-License-Identifier: Apache-2.0
"""The HY1.5 parquet schema must survive collation into a training batch.

``Hunyuan15Model.prepare_batch`` reads ``text_embedding_2`` /
``text_attention_mask_2`` off ``raw_batch``, but ``collate_rows_from_parquet_schema``
only emits the tensor fields its schema declares. A schema without the ByT5
triplet therefore drops the glyph stream silently, so pin the contract against
a real parquet row instead of a hand-built dict.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from fastvideo.dataset.dataloader.schema import (
    pyarrow_schema_t2v,
    pyarrow_schema_t2v_dual_text,
)
from fastvideo.dataset.utils import collate_rows_from_parquet_schema

_QWEN_DIM = 3584
_BYT5_DIM = 1472


def _write_row(path: Path, byt5_tokens: int, schema: pa.Schema) -> dict:
    """Write one parquet row and read it back the way the dataset does."""
    qwen = np.zeros((4, _QWEN_DIM), dtype=np.float32)
    latent = np.zeros((32, 2, 4, 4), dtype=np.float32)
    record = {
        "id": "clip",
        "vae_latent_bytes": latent.tobytes(),
        "vae_latent_shape": list(latent.shape),
        "vae_latent_dtype": "float32",
        "text_embedding_bytes": qwen.tobytes(),
        "text_embedding_shape": list(qwen.shape),
        "text_embedding_dtype": "float32",
        "file_name": "clip.mp4",
        "caption": "a caption",
        "media_type": "video",
        "width": 480,
        "height": 832,
        "num_frames": 81,
        "duration_sec": 5.0,
        "fps": 16.0,
    }
    if "text_embedding_2_bytes" in schema.names:
        byt5 = np.zeros((byt5_tokens, _BYT5_DIM), dtype=np.float32)
        record["text_embedding_2_bytes"] = byt5.tobytes()
        record["text_embedding_2_shape"] = list(byt5.shape)
        record["text_embedding_2_dtype"] = "float32"

    assert set(record) == set(schema.names)
    pq.write_table(pa.table({k: [v] for k, v in record.items()}, schema=schema), path)
    return pq.read_table(path).to_pylist()[0]


def test_collate_keeps_second_text_stream(tmp_path: Path) -> None:
    row = _write_row(tmp_path / "data_00000.parquet", byt5_tokens=3, schema=pyarrow_schema_t2v_dual_text)

    batch = collate_rows_from_parquet_schema([row], pyarrow_schema_t2v_dual_text, text_padding_length=8)

    assert batch["text_embedding_2"].shape == (1, 8, _BYT5_DIM)
    torch.testing.assert_close(
        batch["text_attention_mask_2"],
        torch.tensor([[1.0] * 3 + [0.0] * 5]),
    )


def test_collate_keeps_zero_token_stream_width(tmp_path: Path) -> None:
    """A caption with no glyph text keeps its width, not a 768-wide stub."""
    row = _write_row(tmp_path / "data_00000.parquet", byt5_tokens=0, schema=pyarrow_schema_t2v_dual_text)

    batch = collate_rows_from_parquet_schema([row], pyarrow_schema_t2v_dual_text, text_padding_length=8)

    assert batch["text_embedding_2"].shape == (1, 8, _BYT5_DIM)
    assert int(batch["text_attention_mask_2"].sum().item()) == 0


def test_collate_omits_second_stream_for_legacy_parquet(tmp_path: Path) -> None:
    """A t2v parquet predating the ByT5 columns must not fabricate a stream.

    Leaving the key out is what lets ``prepare_batch`` apply its own
    zero-token fallback (and warn) instead of forwarding a mis-shaped tensor.
    """
    row = _write_row(tmp_path / "data_00000.parquet", byt5_tokens=0, schema=pyarrow_schema_t2v)

    batch = collate_rows_from_parquet_schema([row], pyarrow_schema_t2v_dual_text, text_padding_length=8)

    assert "text_embedding_2" not in batch
    assert "text_attention_mask_2" not in batch
    assert batch["text_embedding"].shape == (1, 8, _QWEN_DIM)


def test_collate_mixed_legacy_and_dual_rows(tmp_path: Path) -> None:
    """A legacy shard mixed into a dual-text batch must not break stacking.

    The all-legacy skip above cannot fire once one row carries the bytes;
    the rows without them become zero-token streams of the batch's real
    width (their masks say "no glyph text"), not a 768-wide stub that
    cannot stack against the real rows.
    """
    dual = _write_row(tmp_path / "data_00000.parquet",
                      byt5_tokens=3,
                      schema=pyarrow_schema_t2v_dual_text)
    legacy = _write_row(tmp_path / "data_00001.parquet",
                        byt5_tokens=0,
                        schema=pyarrow_schema_t2v)

    batch = collate_rows_from_parquet_schema([dual, legacy],
                                             pyarrow_schema_t2v_dual_text,
                                             text_padding_length=8)

    assert batch["text_embedding_2"].shape == (2, 8, _BYT5_DIM)
    torch.testing.assert_close(
        batch["text_attention_mask_2"],
        torch.tensor([[1.0] * 3 + [0.0] * 5, [0.0] * 8]),
    )
    assert batch["text_embedding"].shape == (2, 8, _QWEN_DIM)
