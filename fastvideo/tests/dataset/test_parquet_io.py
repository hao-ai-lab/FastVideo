import os
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from fastvideo.dataset.dataloader import parquet_io
from fastvideo.dataset.dataloader.parquet_io import (
    ParquetDatasetWriter,
    records_to_table,
)


def test_records_to_table_types():
    schema = pa.schema([
        pa.field("id", pa.string()),
        pa.field("vae_latent_bytes", pa.binary()),
        pa.field("vae_latent_shape", pa.list_(pa.int64())),
        pa.field("duration_sec", pa.float64()),
        pa.field("width", pa.int64()),
    ])
    records = [{
        "id": "a",
        "vae_latent_bytes": b"\x00\x01",
        "vae_latent_shape": [1, 2, 3],
        "duration_sec": 1.5,
        "width": 640,
    }]

    table = records_to_table(records, schema)
    assert table.schema == schema
    assert table.num_rows == 1
    cols = {name: table.column(name).to_pylist()[0] for name in schema.names}
    assert cols["id"] == "a"
    assert isinstance(cols["vae_latent_bytes"], (bytes, bytearray))
    assert cols["vae_latent_shape"] == [1, 2, 3]
    assert abs(cols["duration_sec"] - 1.5) < 1e-6
    assert cols["width"] == 640


def test_writer_flush_and_remainder(tmp_path: Path):
    schema = pa.schema([pa.field("id", pa.string())])
    records = [{"id": str(i)} for i in range(25)]
    table = records_to_table(records, schema)

    out_dir = tmp_path / "out"
    writer = ParquetDatasetWriter(str(out_dir), samples_per_file=10)
    writer.append_table(table)
    written = writer.flush(num_workers=1)
    assert written == 20

    files = sorted(out_dir.rglob("*.parquet"))
    assert len(files) == 2
    total_rows = sum(pq.read_table(str(f)).num_rows for f in files)
    assert total_rows == 20

    # Append remainder to complete another chunk
    extra = records_to_table([{"id": str(i)} for i in range(5)], schema)
    writer.append_table(extra)
    written2 = writer.flush(num_workers=1)
    assert written2 == 10
    files2 = sorted(out_dir.rglob("*.parquet"))
    assert len(files2) == 3
    total_rows2 = sum(pq.read_table(str(f)).num_rows for f in files2)
    assert total_rows2 == 30


def test_writer_flush_write_remainder(tmp_path: Path):
    schema = pa.schema([pa.field("id", pa.string())])
    # 25 rows, 10 per file => 2 full files + 1 remainder(5)
    records = [{"id": str(i)} for i in range(25)]
    table = records_to_table(records, schema)

    out_dir = tmp_path / "out_last"
    writer = ParquetDatasetWriter(str(out_dir), samples_per_file=10)
    writer.append_table(table)
    # First flush writes 20
    written1 = writer.flush(num_workers=1)
    assert written1 == 20
    # Final flush with remainder
    written2 = writer.flush(num_workers=1, write_remainder=True)
    assert written2 == 5
    files = sorted(out_dir.rglob("*.parquet"))
    assert len(files) == 3
    total_rows = sum(pq.read_table(str(f)).num_rows for f in files)
    assert total_rows == 25


def test_writer_parallel_workers(tmp_path: Path):
    schema = pa.schema([pa.field("id", pa.string())])
    # 40 rows, 10 per file => 4 files
    records = [{"id": str(i)} for i in range(40)]
    table = records_to_table(records, schema)

    out_dir = tmp_path / "out_parallel"
    writer = ParquetDatasetWriter(str(out_dir), samples_per_file=10)
    writer.append_table(table)
    written = writer.flush(num_workers=2)
    assert written == 40

    # Ensure files exist under worker subdirs
    worker_dirs = [p for p in out_dir.iterdir() if p.is_dir() and p.name.startswith("worker_")]
    assert len(worker_dirs) >= 1
    files = sorted(out_dir.rglob("*.parquet"))
    assert len(files) == 4
    total_rows = sum(pq.read_table(str(f)).num_rows for f in files)
    assert total_rows == 40


def _write_ids(writer: ParquetDatasetWriter, ids: list[int], **flush_kwargs) -> int:
    writer.append_table(pa.table({"id": ids}))
    return writer.flush(**flush_kwargs)


def _read_ids(out_dir: Path) -> list[int]:
    ids = [i for f in out_dir.rglob("*.parquet") for i in pq.read_table(str(f)).column("id").to_pylist()]
    assert len(ids) == len(set(ids)), f"duplicate ids on disk: {sorted(ids)}"
    return sorted(ids)


@pytest.mark.parametrize(
    "flushes",
    [
        # The second flush has fewer chunks, so each worker's chunk range shrinks.
        [([0, 1, 2], 2), ([3, 4], 2)],
        # The second flush has more workers, so each worker's chunk range shrinks.
        [([0, 1, 2, 3], 2), ([4, 5, 6, 7], 4)],
    ],
    ids=["fewer_chunks", "more_workers"],
)
def test_writer_successive_flushes_preserve_samples(tmp_path: Path, flushes):
    writer = ParquetDatasetWriter(str(tmp_path), samples_per_file=1)
    expected: list[int] = []
    for ids, num_workers in flushes:
        assert _write_ids(writer, ids, num_workers=num_workers) == len(ids)
        expected.extend(ids)
    assert _read_ids(tmp_path) == expected


def test_writer_default_workers_preserve_samples(tmp_path: Path, monkeypatch):
    # The default worker count is min(cpu_count, chunks), so a final flush with
    # fewer chunks than earlier ones repartitions chunks across workers.
    monkeypatch.setattr(parquet_io.multiprocessing, "cpu_count", lambda: 2)
    writer = ParquetDatasetWriter(str(tmp_path), samples_per_file=2)
    assert _write_ids(writer, list(range(6))) == 6
    assert _write_ids(writer, list(range(6, 11)), write_remainder=True) == 5
    assert _read_ids(tmp_path) == list(range(11))


@pytest.mark.parametrize(
    "samples_per_file, flush_kwargs",
    [(1, {"num_workers": 1}), (10, {"write_remainder": True})],
    ids=["full_chunk", "remainder"],
)
def test_writer_does_not_overwrite_existing_shard(tmp_path: Path, samples_per_file, flush_kwargs):
    # Existing numbering with a gap (no data_chunk_0) must not cause the next
    # shard to reuse data_chunk_1.
    (tmp_path / "worker_0").mkdir()
    pq.write_table(pa.table({"id": [-1]}), str(tmp_path / "worker_0" / "data_chunk_1.parquet"))

    writer = ParquetDatasetWriter(str(tmp_path), samples_per_file=samples_per_file)
    assert _write_ids(writer, [0], **flush_kwargs) == 1
    assert _read_ids(tmp_path) == [-1, 0]


def test_writer_write_remainder_counts_all_rows(tmp_path: Path):
    writer = ParquetDatasetWriter(str(tmp_path), samples_per_file=10)
    assert _write_ids(writer, list(range(25)), num_workers=1, write_remainder=True) == 25
    assert _read_ids(tmp_path) == list(range(25))


def test_write_chunk_refuses_to_overwrite(tmp_path: Path):
    chunk_path = tmp_path / "data_chunk_0.parquet"
    pq.write_table(pa.table({"id": [-1]}), str(chunk_path))

    with pytest.raises(FileExistsError):
        parquet_io._write_chunk(pa.table({"id": [0]}), str(chunk_path), "zstd")

    assert pq.read_table(str(chunk_path)).column("id").to_pylist() == [-1]


def test_write_chunk_removes_temp_file_on_failure(tmp_path: Path, monkeypatch):

    def fail_mid_write(table, where, **kwargs):
        Path(where).write_bytes(b"partial")
        raise OSError("disk full")

    monkeypatch.setattr(parquet_io.pq, "write_table", fail_mid_write)

    with pytest.raises(OSError, match="disk full"):
        parquet_io._write_chunk(pa.table({"id": [0]}), str(tmp_path / "data_chunk_0.parquet"), "zstd")

    assert list(tmp_path.iterdir()) == []
