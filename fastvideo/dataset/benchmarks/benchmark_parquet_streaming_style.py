# SPDX-License-Identifier: Apache-2.0
"""Small benchmark for projected sequential row-group reads.

Example:
  python -m fastvideo.dataset.benchmarks.benchmark_parquet_streaming_style \
    --data-path /path/to/parquet --batches 4
"""

from __future__ import annotations

import argparse
import os
import tempfile
import time

from fastvideo.dataset.dataloader.schema import pyarrow_schema_t2v
from fastvideo.dataset.parquet_dataset_streaming_style import (
    LatentsParquetStreamingDataset, )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--manifest-path", default="")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--read-batch-size", type=int, default=4)
    parser.add_argument("--batches", type=int, default=4)
    args = parser.parse_args()
    manifest_path = args.manifest_path or os.path.join(
        tempfile.gettempdir(), "fastvideo-streaming-benchmark.json")
    dataset = LatentsParquetStreamingDataset(
        args.data_path,
        args.batch_size,
        pyarrow_schema_t2v,
        manifest_path,
        num_workers=0,
        read_batch_size=args.read_batch_size,
        shuffle_row_groups=False,
    )
    started = time.perf_counter()
    rows = 0
    for index, batch in enumerate(dataset):
        rows += len(batch["info_list"])
        if index + 1 >= args.batches:
            break
    elapsed = time.perf_counter() - started
    print({
        "batches": index + 1,
        "rows": rows,
        "seconds": elapsed,
        "rows_per_second": rows / elapsed,
        "projected_columns": len(pyarrow_schema_t2v.names),
    })


if __name__ == "__main__":
    main()
