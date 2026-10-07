# SPDX-License-Identifier: Apache-2.0
"""Contract for where the HunyuanVideo 1.5 ByT5 columns live.

They belong to ``pyarrow_schema_t2v_dual_text`` only. The shared
``pyarrow_schema_t2v`` stays as it was, so every other t2v writer keeps
building its table with ``pa.table(mapping, schema=pyarrow_schema_t2v)``.
Collating the dual-text rows is covered in test_hunyuan15_dual_text_schema.py
and test_hunyuan15_dual_text_collator.py.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

from fastvideo.dataset.dataloader import schema as schema_module
from fastvideo.dataset.dataloader.schema import (
    pyarrow_schema_t2v,
    pyarrow_schema_t2v_dual_text,
)

_BYT5_FIELDS = {
    "text_embedding_2_bytes": pa.binary(),
    "text_embedding_2_shape": pa.list_(pa.int64()),
    "text_embedding_2_dtype": pa.string(),
}


def test_dual_text_schema_is_t2v_plus_the_byt5_triplet() -> None:
    assert pyarrow_schema_t2v_dual_text.names == pyarrow_schema_t2v.names + list(_BYT5_FIELDS)
    for name, arrow_type in _BYT5_FIELDS.items():
        assert pyarrow_schema_t2v_dual_text.field(name).type == arrow_type


def test_t2v_writers_without_byt5_still_build_a_table() -> None:
    """``pa.table(mapping, schema=...)`` demands a key for every schema field.

    The existing overfit preprocess scripts build that mapping from their own
    record keys, so declaring the ByT5 columns in the shared schema would make
    each of them raise KeyError at the parquet-write step.
    """
    assert not set(_BYT5_FIELDS) & set(pyarrow_schema_t2v.names)

    record: dict = {}
    for field in pyarrow_schema_t2v:
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

    table = pa.table({k: [v] for k, v in record.items()}, schema=pyarrow_schema_t2v)
    assert table.schema.equals(pyarrow_schema_t2v)


@pytest.mark.parametrize(
    "name",
    sorted(name for name, value in vars(schema_module).items() if isinstance(value, pa.Schema)),
)
def test_schema_declares_each_field_once(name: str) -> None:
    """pyarrow accepts a repeated field name, then fails every lookup of it.

    The collator would also see the repeated tensor field twice.
    """
    names = getattr(schema_module, name).names
    assert len(names) == len(set(names)), sorted({n for n in names if names.count(n) > 1})
