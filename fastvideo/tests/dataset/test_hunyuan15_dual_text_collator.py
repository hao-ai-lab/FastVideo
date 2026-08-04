# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import torch
import pyarrow.parquet as pq

from fastvideo.dataset.dataloader.parquet_io import records_to_table
from fastvideo.dataset.dataloader.schema import pyarrow_schema_t2v
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
    sample_index: int,
    qwen_tokens: int = 3,
    byt5_tokens: int | None = 2,
    qwen_value: float = 1.0,
    byt5_value: float = 2.0,
) -> dict[str, Any]:
    """Build one minimal T2V parquet row for collator tests.

    byt5_tokens=None represents a legacy row that has no text_embedding_2
    fields at all.

    byt5_tokens=0 represents a valid HunyuanVideo 1.5 sample with no
    quoted glyph text.
    """

    qwen = np.full(
        (qwen_tokens, QWEN_DIM),
        qwen_value,
        dtype=np.float32,
    )

    latent = np.full(
        (32, 2, 4, 4),
        0.5,
        dtype=np.float32,
    )

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
        "_sample_index": sample_index,
    }

    row.update(_tensor_fields(latent, "vae_latent"))
    row.update(_tensor_fields(qwen, "text_embedding"))

    if byt5_tokens is not None:
        byt5 = np.full(
            (byt5_tokens, BYT5_DIM),
            byt5_value,
            dtype=np.float32,
        )
        row.update(_tensor_fields(byt5, "text_embedding_2"))

    return row


def _collate(
    rows: list[dict[str, Any]],
    *,
    cfg_rate: float = 0.0,
    seed: int = 42,
) -> dict[str, Any]:
    return collate_rows_from_parquet_schema(
        rows=rows,
        parquet_schema=pyarrow_schema_t2v,
        text_padding_length=TEXT_PADDING_LENGTH,
        cfg_rate=cfg_rate,
        seed=seed,
    )

def _write_and_read_parquet(
    tmp_path,
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    parquet_path = tmp_path / "data_00000.parquet"

    table = records_to_table(
        rows,
        pyarrow_schema_t2v,
    )
    pq.write_table(table, parquet_path)

    loaded_table = pq.read_table(parquet_path)
    return loaded_table.to_pylist()

def test_dual_text_embeddings_are_padded_independently() -> None:
    rows = [
        _make_row(
            sample_index=0,
            qwen_tokens=3,
            byt5_tokens=2,
        ),
        _make_row(
            sample_index=1,
            qwen_tokens=5,
            byt5_tokens=4,
        ),
    ]

    batch = _collate(rows)

    assert batch["text_embedding"].shape == (
        2,
        TEXT_PADDING_LENGTH,
        QWEN_DIM,
    )
    assert batch["text_embedding_2"].shape == (
        2,
        TEXT_PADDING_LENGTH,
        BYT5_DIM,
    )

    assert batch["text_attention_mask"].shape == (
        2,
        TEXT_PADDING_LENGTH,
    )
    assert batch["text_attention_mask_2"].shape == (
        2,
        TEXT_PADDING_LENGTH,
    )

    assert batch["text_attention_mask"][0].sum().item() == 3
    assert batch["text_attention_mask"][1].sum().item() == 5

    assert batch["text_attention_mask_2"][0].sum().item() == 2
    assert batch["text_attention_mask_2"][1].sum().item() == 4

    assert torch.count_nonzero(
        batch["text_embedding"][0, 3:]
    ).item() == 0

    assert torch.count_nonzero(
        batch["text_embedding_2"][0, 2:]
    ).item() == 0


def test_empty_byt5_embedding_preserves_hidden_dimension() -> None:
    """A [0, 1472] embedding must become a padded [L, 1472] tensor."""

    rows = [
        _make_row(
            sample_index=0,
            qwen_tokens=3,
            byt5_tokens=0,
        ),
    ]

    batch = _collate(rows)

    assert batch["text_embedding_2"].shape == (
        1,
        TEXT_PADDING_LENGTH,
        BYT5_DIM,
    )

    assert batch["text_attention_mask_2"].shape == (
        1,
        TEXT_PADDING_LENGTH,
    )

    assert batch["text_attention_mask_2"].sum().item() == 0
    assert torch.count_nonzero(
        batch["text_embedding_2"]
    ).item() == 0


def test_mixed_empty_and_nonempty_byt5_embeddings() -> None:
    """A batch may contain samples with and without glyph text."""

    rows = [
        _make_row(
            sample_index=0,
            byt5_tokens=0,
        ),
        _make_row(
            sample_index=1,
            byt5_tokens=4,
        ),
    ]

    batch = _collate(rows)

    assert batch["text_embedding_2"].shape == (
        2,
        TEXT_PADDING_LENGTH,
        BYT5_DIM,
    )

    assert batch["text_attention_mask_2"][0].sum().item() == 0
    assert batch["text_attention_mask_2"][1].sum().item() == 4

    assert torch.count_nonzero(
        batch["text_embedding_2"][0]
    ).item() == 0

    assert torch.count_nonzero(
        batch["text_embedding_2"][1, :4]
    ).item() > 0


def test_legacy_rows_without_byt5_skip_secondary_batch_fields() -> None:
    """A batch containing only old-format rows remains supported."""

    rows = [
        _make_row(
            sample_index=0,
            byt5_tokens=None,
        ),
        _make_row(
            sample_index=1,
            byt5_tokens=None,
        ),
    ]

    batch = _collate(rows)

    assert "text_embedding" in batch
    assert "text_attention_mask" in batch

    assert "text_embedding_2" not in batch
    assert "text_attention_mask_2" not in batch


def test_legacy_row_can_mix_with_hunyuan_row() -> None:
    """A missing ByT5 field is treated as zero tokens in a Hunyuan batch."""

    rows = [
        _make_row(
            sample_index=0,
            byt5_tokens=None,
        ),
        _make_row(
            sample_index=1,
            byt5_tokens=3,
        ),
    ]

    batch = _collate(rows)

    assert batch["text_embedding_2"].shape == (
        2,
        TEXT_PADDING_LENGTH,
        BYT5_DIM,
    )

    assert batch["text_attention_mask_2"][0].sum().item() == 0
    assert batch["text_attention_mask_2"][1].sum().item() == 3


def test_cfg_rate_zero_preserves_both_text_conditions() -> None:
    rows = [
        _make_row(
            sample_index=0,
            qwen_tokens=3,
            byt5_tokens=2,
        ),
    ]

    batch = _collate(
        rows,
        cfg_rate=0.0,
    )

    assert torch.count_nonzero(
        batch["text_embedding"][0, :3]
    ).item() > 0

    assert torch.count_nonzero(
        batch["text_embedding_2"][0, :2]
    ).item() > 0


def test_cfg_rate_one_drops_both_text_conditions() -> None:
    """Qwen and ByT5 must use the same CFG dropout decision."""

    rows = [
        _make_row(
            sample_index=0,
            qwen_tokens=3,
            byt5_tokens=2,
        ),
    ]

    batch = _collate(
        rows,
        cfg_rate=1.0,
    )

    assert torch.count_nonzero(
        batch["text_embedding"]
    ).item() == 0

    assert torch.count_nonzero(
        batch["text_embedding_2"]
    ).item() == 0

def test_dual_text_embeddings_parquet_round_trip(tmp_path) -> None:
    rows = [
        _make_row(
            sample_index=0,
            qwen_tokens=3,
            byt5_tokens=2,
            qwen_value=1.0,
            byt5_value=2.0,
        ),
        _make_row(
            sample_index=1,
            qwen_tokens=5,
            byt5_tokens=4,
            qwen_value=3.0,
            byt5_value=4.0,
        ),
    ]

    loaded_rows = _write_and_read_parquet(tmp_path, rows)
    batch = _collate(loaded_rows)

    assert batch["text_embedding"].shape == (
        2,
        TEXT_PADDING_LENGTH,
        QWEN_DIM,
    )
    assert batch["text_embedding_2"].shape == (
        2,
        TEXT_PADDING_LENGTH,
        BYT5_DIM,
    )

    assert batch["text_attention_mask"][0].sum().item() == 3
    assert batch["text_attention_mask"][1].sum().item() == 5
    assert batch["text_attention_mask_2"][0].sum().item() == 2
    assert batch["text_attention_mask_2"][1].sum().item() == 4

    torch.testing.assert_close(
        batch["text_embedding"][0, :3],
        torch.ones(
            (3, QWEN_DIM),
            dtype=torch.float32,
        ),
    )

    torch.testing.assert_close(
        batch["text_embedding_2"][0, :2],
        torch.full(
            (2, BYT5_DIM),
            2.0,
            dtype=torch.float32,
        ),
    )

def test_empty_byt5_embedding_parquet_round_trip(tmp_path) -> None:
    rows = [
        _make_row(
            sample_index=0,
            qwen_tokens=3,
            byt5_tokens=0,
        ),
        _make_row(
            sample_index=1,
            qwen_tokens=4,
            byt5_tokens=2,
        ),
    ]

    loaded_rows = _write_and_read_parquet(tmp_path, rows)
    batch = _collate(loaded_rows)

    assert batch["text_embedding_2"].shape == (
        2,
        TEXT_PADDING_LENGTH,
        BYT5_DIM,
    )

    assert batch["text_attention_mask_2"][0].sum().item() == 0
    assert batch["text_attention_mask_2"][1].sum().item() == 2

    assert torch.count_nonzero(
        batch["text_embedding_2"][0]
    ).item() == 0