# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pyarrow as pa
import pytest

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


def test_writer_records_without_byt5_still_build_a_table() -> None:
    """A writer that predates the secondary encoder must still be able to write.

    ``nullable=True`` is pyarrow's default and asserting it proves nothing --
    it holds for every field in every schema. What matters is that
    ``pa.table(mapping, schema=...)`` demands a key for *every* schema field,
    so a record built before these columns existed raises KeyError, while
    ``records_to_table`` (from_pylist) fills them with null.
    """
    import pyarrow as pa

    from fastvideo.dataset.dataloader.parquet_io import records_to_table

    secondary = {
        "text_embedding_2_bytes",
        "text_embedding_2_shape",
        "text_embedding_2_dtype",
    }
    record: dict = {}
    for field in pyarrow_schema_t2v:
        if field.name in secondary:
            continue
        if pa.types.is_binary(field.type):
            record[field.name] = b""
        elif pa.types.is_string(field.type):
            record[field.name] = "x"
        elif pa.types.is_list(field.type):
            record[field.name] = [1]
        elif pa.types.is_integer(field.type):
            record[field.name] = 1
        else:
            record[field.name] = 1.0

    with pytest.raises(KeyError):
        pa.table({k: [record[k]] for k in record}, schema=pyarrow_schema_t2v)

    table = records_to_table([record], pyarrow_schema_t2v)
    assert table.schema.equals(pyarrow_schema_t2v)
    assert table.column("text_embedding_2_bytes").to_pylist() == [None]