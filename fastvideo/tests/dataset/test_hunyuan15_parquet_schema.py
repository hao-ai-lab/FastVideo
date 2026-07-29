# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pyarrow as pa

from fastvideo.dataset.dataloader.schema import pyarrow_schema_t2v


def test_hunyuan15_secondary_text_embedding_fields_exist() -> None:
    """The T2V schema must expose the secondary ByT5 embedding fields."""

    expected_fields = {
        "text_embedding_2_bytes",
        "text_embedding_2_shape",
        "text_embedding_2_dtype",
    }

    assert expected_fields.issubset(set(pyarrow_schema_t2v.names))


def test_hunyuan15_secondary_text_embedding_field_types() -> None:
    """The ByT5 fields must use the expected Arrow storage types."""

    bytes_field = pyarrow_schema_t2v.field("text_embedding_2_bytes")
    shape_field = pyarrow_schema_t2v.field("text_embedding_2_shape")
    dtype_field = pyarrow_schema_t2v.field("text_embedding_2_dtype")

    assert bytes_field.type == pa.binary()
    assert shape_field.type == pa.list_(pa.int64())
    assert dtype_field.type == pa.string()


def test_hunyuan15_secondary_text_embedding_fields_are_nullable() -> None:
    """Old parquet datasets may not contain the secondary text encoder."""

    assert pyarrow_schema_t2v.field(
        "text_embedding_2_bytes"
    ).nullable

    assert pyarrow_schema_t2v.field(
        "text_embedding_2_shape"
    ).nullable

    assert pyarrow_schema_t2v.field(
        "text_embedding_2_dtype"
    ).nullable