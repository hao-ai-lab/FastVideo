# SPDX-License-Identifier: Apache-2.0
"""Dual-text (Qwen + ByT5) collation across the rows of a batch.

test_hunyuan15_dual_text_schema.py covers the per-stream contract: the ByT5
stream is kept, a zero-token stream keeps its width, a legacy parquet emits no
stream, and legacy rows mix with dual-text ones. This file covers the rest:
the two streams padding independently, zero-token and glyph rows sharing a
batch, one CFG dropout decision per sample for both streams, and the primary
stream being mandatory.
"""

from __future__ import annotations

import random
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import pytest
import torch

from fastvideo.dataset.dataloader.parquet_io import records_to_table
from fastvideo.dataset.dataloader.schema import pyarrow_schema_t2v_dual_text
from fastvideo.dataset.utils import collate_rows_from_parquet_schema

QWEN_DIM = 3584
BYT5_DIM = 1472
TEXT_PADDING_LENGTH = 8


def _tensor_fields(
    tensor: np.ndarray,
    prefix: str,
) -> dict[str, Any]:
    """Serialize a NumPy tensor using the FastVideo parquet convention."""

    tensor = np.ascontiguousarray(tensor)

    return {
        f"{prefix}_bytes": tensor.tobytes(),
        f"{prefix}_shape": list(tensor.shape),
        f"{prefix}_dtype": str(tensor.dtype),
    }


def _make_row(
    *,
    sample_index: int | None,
    qwen_tokens: int = 3,
    byt5_tokens: int = 2,
    qwen_value: float = 1.0,
    byt5_value: float = 2.0,
) -> dict[str, Any]:
    """Build one HunyuanVideo 1.5 parquet row.

    byt5_tokens=0 is a caption with no quoted glyph text. sample_index=None
    leaves ``_sample_index`` out, as callers other than the map-style loader
    do.
    """

    row: dict[str, Any] = {
        "id": f"sample-{sample_index}",
        "file_name": f"sample-{sample_index}.mp4",
        "caption": f"caption {sample_index}",
        "media_type": "video",
        "width": 64,
        "height": 64,
        "num_frames": 5,
        "duration_sec": 1.0,
        "fps": 5.0,
    }
    if sample_index is not None:
        row["_sample_index"] = sample_index

    row.update(_tensor_fields(np.full((32, 2, 4, 4), 0.5, dtype=np.float32), "vae_latent"))
    row.update(_tensor_fields(np.full((qwen_tokens, QWEN_DIM), qwen_value, dtype=np.float32), "text_embedding"))
    row.update(_tensor_fields(np.full((byt5_tokens, BYT5_DIM), byt5_value, dtype=np.float32), "text_embedding_2"))
    return row


def _collate(
    rows: list[dict[str, Any]],
    *,
    cfg_rate: float = 0.0,
    seed: int = 42,
    rng: random.Random | None = None,
) -> dict[str, Any]:
    return collate_rows_from_parquet_schema(
        rows=rows,
        parquet_schema=pyarrow_schema_t2v_dual_text,
        text_padding_length=TEXT_PADDING_LENGTH,
        cfg_rate=cfg_rate,
        rng=rng,
        seed=seed,
    )


def _write_and_read_parquet(
    tmp_path,
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    parquet_path = tmp_path / "data_00000.parquet"
    pq.write_table(records_to_table(rows, pyarrow_schema_t2v_dual_text), parquet_path)
    return pq.read_table(parquet_path).to_pylist()


def _dropped(batch: dict[str, Any], key: str, row: int) -> bool:
    return torch.count_nonzero(batch[key][row]).item() == 0


def test_streams_pad_independently_through_parquet(tmp_path) -> None:
    rows = [
        _make_row(sample_index=0, qwen_tokens=3, byt5_tokens=2, qwen_value=1.0, byt5_value=2.0),
        _make_row(sample_index=1, qwen_tokens=5, byt5_tokens=4, qwen_value=3.0, byt5_value=4.0),
    ]

    batch = _collate(_write_and_read_parquet(tmp_path, rows))

    assert batch["text_embedding"].shape == (2, TEXT_PADDING_LENGTH, QWEN_DIM)
    assert batch["text_embedding_2"].shape == (2, TEXT_PADDING_LENGTH, BYT5_DIM)
    assert batch["text_attention_mask"].sum(dim=1).tolist() == [3, 5]
    assert batch["text_attention_mask_2"].sum(dim=1).tolist() == [2, 4]

    torch.testing.assert_close(batch["text_embedding"][0, :3], torch.ones((3, QWEN_DIM)))
    torch.testing.assert_close(batch["text_embedding_2"][0, :2], torch.full((2, BYT5_DIM), 2.0))
    torch.testing.assert_close(batch["text_embedding_2"][1, :4], torch.full((4, BYT5_DIM), 4.0))
    assert torch.count_nonzero(batch["text_embedding"][0, 3:]).item() == 0
    assert torch.count_nonzero(batch["text_embedding_2"][0, 2:]).item() == 0


def test_zero_token_and_glyph_rows_share_a_batch(tmp_path) -> None:
    rows = [
        _make_row(sample_index=0, qwen_tokens=3, byt5_tokens=0),
        _make_row(sample_index=1, qwen_tokens=4, byt5_tokens=2),
    ]

    batch = _collate(_write_and_read_parquet(tmp_path, rows))

    assert batch["text_embedding_2"].shape == (2, TEXT_PADDING_LENGTH, BYT5_DIM)
    assert batch["text_attention_mask_2"].sum(dim=1).tolist() == [0, 2]
    assert torch.count_nonzero(batch["text_embedding_2"][0]).item() == 0
    torch.testing.assert_close(batch["text_embedding_2"][1, :2], torch.full((2, BYT5_DIM), 2.0))


@pytest.mark.parametrize("with_sample_index", [True, False], ids=["sample_index", "rng_fallback"])
def test_cfg_dropout_drops_both_streams_together(with_sample_index: bool) -> None:
    """One decision per sample: a half-dropped condition is neither cond nor uncond."""
    num_rows, seed, cfg_rate = 64, 42, 0.5
    rows = [_make_row(sample_index=i if with_sample_index else None) for i in range(num_rows)]

    batch = _collate(rows, cfg_rate=cfg_rate, seed=seed, rng=random.Random(0))

    qwen = [_dropped(batch, "text_embedding", i) for i in range(num_rows)]
    byt5 = [_dropped(batch, "text_embedding_2", i) for i in range(num_rows)]
    assert qwen == byt5
    assert 0 < sum(qwen) < num_rows
    if with_sample_index:
        # Resume-safe: a pure function of seed ^ index, whatever the batch.
        assert qwen == [random.Random(seed ^ i).random() < cfg_rate for i in range(num_rows)]


def test_cfg_dropout_fallback_takes_one_draw_per_sample() -> None:
    """Without ``_sample_index`` the rng is drawn once per row, in row order.

    That is what the single-stream collator always drew, so its callers see
    the same sequence; the ByT5 stream must not draw a second value.
    """
    rng, reference = random.Random(7), random.Random(7)
    rows = [_make_row(sample_index=None) for _ in range(6)]

    batch = _collate(rows, cfg_rate=0.5, rng=rng)

    assert [_dropped(batch, "text_embedding", i) for i in range(6)] == [reference.random() < 0.5 for _ in rows]
    assert rng.random() == reference.random()


def test_cfg_rate_one_drops_both_streams() -> None:
    batch = _collate([_make_row(sample_index=0)], cfg_rate=1.0)

    assert torch.count_nonzero(batch["text_embedding"]).item() == 0
    assert torch.count_nonzero(batch["text_embedding_2"]).item() == 0


def test_missing_primary_text_embedding_names_the_rows() -> None:
    """Unlike ByT5, the Qwen stream has no zero-token fallback."""
    rows = [_make_row(sample_index=i) for i in range(3)]
    for key in ("text_embedding_bytes", "text_embedding_shape", "text_embedding_dtype"):
        del rows[1][key]
    rows[2]["text_embedding_bytes"] = None  # a null cell read back from parquet

    with pytest.raises(ValueError, match=r"text_embedding is missing from rows \[1, 2\] of 3"):
        _collate(rows)


def test_missing_primary_text_embedding_everywhere_is_an_error() -> None:
    rows = [_make_row(sample_index=i) for i in range(2)]
    for row in rows:
        row["text_embedding_bytes"] = None

    with pytest.raises(ValueError, match="text_embedding is missing from every row"):
        _collate(rows)
