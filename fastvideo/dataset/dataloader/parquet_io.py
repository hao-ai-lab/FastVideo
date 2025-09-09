"""
Utilities for converting preprocessing records (dicts) into Arrow tables and
writing Parquet datasets in fixed-size chunks.

This module centralizes table construction and Parquet file writing so
pipelines only need to define their PyArrow schema and produce per-sample
record dictionaries.

Key APIs:
- records_to_table(records, schema): Safely convert a list of dictionaries into
  a pa.Table, casting to the provided schema.
- ParquetDatasetWriter: Buffer tables and flush to a directory as multiple
  Parquet files with a fixed number of rows per file. Uses temporary files and
  atomic rename to avoid partially written outputs.
"""

from __future__ import annotations

import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq


def records_to_table(records: list[dict[str, Any]], schema: pa.Schema) -> pa.Table:
    """Build a PyArrow table from Python record dicts using an explicit schema.

    Arrow will cast values to the target schema when possible (e.g., promoting
    Python ints/floats to pa.int64/pa.float64), eliminating hand-written per-
    field array construction.

    Args:
        records: List of dictionaries, each representing one row. Keys must
            match schema field names.
        schema: Target PyArrow schema. Controls field names and types.

    Returns:
        pa.Table: In-memory table matching the provided schema. If ``records``
        is empty, returns an empty table with the given schema.
    """
    if not records:
        return pa.table({}, schema=schema)
    return pa.Table.from_pylist(records, schema=schema)


class ParquetDatasetWriter:
    """Accumulate tables and flush them to a Parquet directory in fixed-size chunks.

    Behavior:
    - Writes files under worker-specific subdirectories for parallelism.
    - Uses temporary files and atomic rename to avoid partial files being left
      behind on failure.
    - Only full chunks of ``samples_per_file`` rows are written on each flush;
      any remainder rows are re-buffered for the next flush.

    Note:
    - Instances are not meant to be shared across processes. Create one writer
      per process if using multiprocessing.
    """

    def __init__(self, out_dir: str, samples_per_file: int, compression: str = "zstd") -> None:
        """Initialize the dataset writer.

        Args:
            out_dir: Output directory where Parquet files will be written.
            samples_per_file: Fixed number of rows per Parquet file.
            compression: Compression codec passed to ``pyarrow.parquet.write_table``
                (e.g., ``"zstd"``, ``"snappy"``, ``"gzip"``).
        """
        self.out_dir = out_dir
        self.samples_per_file = max(int(samples_per_file), 1)
        self.compression = compression
        os.makedirs(self.out_dir, exist_ok=True)
        self._tables: list[pa.Table] = []

    def append_table(self, table: pa.Table) -> None:
        """Append a non-empty table to the internal buffer.

        Args:
            table: A ``pa.Table`` to buffer. Empty or ``None`` tables are ignored.
        """
        if table is None or len(table) == 0:
            return
        self._tables.append(table)

    def _combine(self) -> pa.Table | None:
        """Combine all buffered tables into a single table, if any.

        Returns:
            A concatenated table, a single table if only one was buffered, or
            ``None`` if no tables are buffered.
        """
        if not self._tables:
            return None
        if len(self._tables) == 1:
            return self._tables[0]
        return pa.concat_tables(self._tables, promote_options='none')

    def flush(self, num_workers: int | None = None, write_remainder: bool = False) -> int:
        """Write accumulated tables to disk and clear the written portion.

        Only complete chunks of size ``samples_per_file`` are written. Any
        remainder rows are kept buffered for the next flush.

        Args:
            num_workers: Optional override for the number of parallel workers
                used to write chunks. Defaults to ``min(cpu_count, chunks)``.
            write_remainder: If True, also write any leftover rows (< samples_per_file)
                as a final small Parquet file (useful for the last flush at the
                end of preprocessing).

        Returns:
            int: Number of rows successfully written in this flush call.
        """
        combined = self._combine()
        self._tables = []
        if combined is None or len(combined) == 0:
            return 0

        num_samples = len(combined)
        total_chunks = num_samples // self.samples_per_file
        if total_chunks == 0:
            if not write_remainder:
                # Not enough to form a full chunk; keep buffered for next round
                # Re-buffer and return 0 written
                self._tables = [combined]
                return 0
            # Last flush: write the small remainder as a final file in worker_0
            worker_dir = os.path.join(self.out_dir, "worker_0")
            os.makedirs(worker_dir, exist_ok=True)
            # Determine next index
            num_parquets = 0
            for _, _, files in os.walk(worker_dir):
                for file in files:
                    if file.endswith('.parquet'):
                        num_parquets += 1
            chunk_path = os.path.join(worker_dir, f"data_chunk_{num_parquets}.parquet")
            temp_path = chunk_path + '.tmp'
            pq.write_table(combined, temp_path, compression=self.compression)
            if os.path.exists(chunk_path):
                os.remove(chunk_path)
            os.rename(temp_path, chunk_path)
            return num_samples

        # Only write full chunks; keep remainder for next flush
        written_rows = total_chunks * self.samples_per_file
        remainder = num_samples - written_rows

        table_to_write = combined.slice(0, written_rows)
        remainder_table = combined.slice(written_rows, remainder) if remainder > 0 else None
        if remainder_table is not None and len(remainder_table) > 0:
            if write_remainder:
                # Write the remainder as a final small file (worker_0)
                worker_dir = os.path.join(self.out_dir, "worker_0")
                os.makedirs(worker_dir, exist_ok=True)
                num_parquets = 0
                for _, _, files in os.walk(worker_dir):
                    for file in files:
                        if file.endswith('.parquet'):
                            num_parquets += 1
                remainder_path = os.path.join(worker_dir,
                                              f"data_chunk_{num_parquets}.parquet")
                temp_path = remainder_path + '.tmp'
                pq.write_table(remainder_table,
                               temp_path,
                               compression=self.compression)
                if os.path.exists(remainder_path):
                    os.remove(remainder_path)
                os.rename(temp_path, remainder_path)
            else:
                self._tables = [remainder_table]

        # Parallel write by chunk ranges
        if num_workers is None:
            num_workers = min(multiprocessing.cpu_count(), max(total_chunks, 1))
        num_workers = max(int(num_workers), 1)
        chunks_per_worker = (total_chunks + num_workers - 1) // num_workers

        work_ranges: list[tuple[int, int, pa.Table, int, str, int, str]] = []
        for worker_id in range(num_workers):
            start_chunk = worker_id * chunks_per_worker
            end_chunk = min((worker_id + 1) * chunks_per_worker, total_chunks)
            if start_chunk < end_chunk:
                work_ranges.append(
                    (
                        start_chunk,
                        end_chunk,
                        table_to_write,
                        worker_id,
                        self.out_dir,
                        self.samples_per_file,
                        self.compression,
                    )
                )

        written_total = 0
        if len(work_ranges) == 1:
            written_total += _process_chunk_range(work_ranges[0])
            return written_total

        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(_process_chunk_range, args) for args in work_ranges]
            for f in futures:
                written_total += f.result()
        return written_total + (len(remainder_table) if write_remainder and remainder_table is not None else 0)


def _process_chunk_range(args: Any) -> int:
    """Worker function to write a contiguous range of chunk files.

    Args:
        args: Tuple containing
            - start_chunk (int): inclusive start chunk index
            - end_chunk (int): exclusive end chunk index
            - table (pa.Table): concatenated table containing all rows to write
            - worker_id (int): numeric worker identifier
            - output_dir (str): base output directory
            - samples_per_file (int): rows per chunk file
            - compression (str): compression codec for Parquet

    Returns:
        int: Total number of rows written by this worker.
    """
    start_chunk, end_chunk, table, worker_id, output_dir, samples_per_file, compression = args
    total_written = 0
    num_samples = len(table)

    worker_dir = os.path.join(output_dir, f"worker_{worker_id}")
    os.makedirs(worker_dir, exist_ok=True)

    # Offset to continue numbering if files exist
    num_parquets = 0
    for root, _, files in os.walk(worker_dir):
        for file in files:
            if file.endswith('.parquet'):
                num_parquets += 1

    for i in range(start_chunk, end_chunk):
        start_sample = i * samples_per_file
        end_sample = min((i + 1) * samples_per_file, num_samples)
        if end_sample <= start_sample:
            continue
        chunk = table.slice(start_sample, end_sample - start_sample)

        chunk_path = os.path.join(worker_dir, f"data_chunk_{i + num_parquets}.parquet")
        temp_path = chunk_path + '.tmp'
        try:
            pq.write_table(chunk, temp_path, compression=compression)
            if os.path.exists(chunk_path):
                os.remove(chunk_path)
            os.rename(temp_path, chunk_path)
            total_written += len(chunk)
        except Exception:
            if os.path.exists(temp_path):
                os.remove(temp_path)
            raise

    return total_written



