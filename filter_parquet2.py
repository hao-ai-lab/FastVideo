#!/usr/bin/env python3
"""
scan_parquet_mt.py

Recursively scans all Parquet files under the given root directory.
If any row in a file contains a black frame (all pixels below threshold),
writes a new parquet file with "filtered_" prefix and deletes the original.

Features
--------
• ThreadPoolExecutor for parallel I/O  
• tqdm progress bar with per-file updates  
• --dry-run flag for a safe preview  
• --workers flag to control thread count
• Writes filtered files in same location with "filtered_" prefix
"""

import argparse
import hashlib
import os
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from PIL import Image
from tqdm.auto import tqdm


def process_file(path: Path,
                 black_threshold: float = 5.0,
                 dry_run: bool = False,
                 images_output_dir: Path | None = None) -> int:
    """Process a parquet file and write a filtered version with prefix. Returns number of rows removed."""

    # Skip if already filtered
    if path.stem.startswith("filtered_"):
        # tqdm.write(f"[SKIP] Already filtered: {path}")
        # return 0
        truncate_prefix = True
    else:
        truncate_prefix = False

    # Read the entire table
    table = pq.read_table(path)
    total_rows = len(table)

    # Track which rows to keep
    rows_to_keep = []
    rows_removed = 0

    # Check each row
    for row_idx in range(total_rows):
        row = table.slice(row_idx, 1).to_pylist()[0]

        # Skip if any field is None
        if row["pil_image_bytes"] is None or row[
                "pil_image_shape"] is None or row["pil_image_dtype"] is None:
            tqdm.write(
                f"[WARN] Row {row_idx} in {path} has None values, keeping it")
            rows_to_keep.append(row_idx)
            continue

        # Convert bytes to numpy array
        image_bytes = row["pil_image_bytes"]
        shape = row["pil_image_shape"]
        dtype = row["pil_image_dtype"]

        # Convert bytes to numpy array with proper shape and dtype
        image_array = np.frombuffer(image_bytes,
                                    dtype=np.float32).reshape(shape)
        image_array = image_array.squeeze(
            0)  # Remove single-dimensional entries if any

        # Convert to uint8 for checking black frames
        if image_array.dtype != np.uint8:
            # Normalize to 0-255 range
            img_min = image_array.min()
            img_max = image_array.max()
            if img_max > img_min:
                image_uint8 = ((image_array - img_min) / (img_max - img_min) *
                               255).astype(np.uint8)
            else:
                image_uint8 = np.zeros_like(image_array, dtype=np.uint8)
        else:
            image_uint8 = image_array

        mean_value = np.mean(image_uint8)

        # Check if the frame is black
        if mean_value < black_threshold:
            tqdm.write(
                f"[INFO] Found black frame in {path} row {row_idx} (mean={mean_value:.2f})"
            )
            rows_removed += 1

            # Save black frame for inspection with unique ID
            if images_output_dir:
                # Create unique filename using file path, row index, and UUID
                # Hash the full path to handle duplicate filenames from different directories
                path_hash = hashlib.md5(str(
                    path.absolute()).encode()).hexdigest()[:8]
                unique_id = f"{path.stem}_{path_hash}_row{row_idx}_{uuid.uuid4().hex[:8]}"

                image_uint8 = np.transpose(image_uint8, (1, 2, 0))
                img = Image.fromarray(image_uint8)
                output_path = images_output_dir / f"{unique_id}.png"
                img.save(output_path)
                tqdm.write(f"[INFO] Saved black frame to {output_path}")
        else:
            # Keep this row
            rows_to_keep.append(row_idx)

    # Process based on what we found
    if rows_removed > 0:
        if dry_run:
            tqdm.write(
                f"[DRY-RUN] Would remove {rows_removed} rows from {path}")
            tqdm.write(
                f"[DRY-RUN] Would create filtered_{path.name} and delete original"
            )
        else:
            # Create new table with only the rows to keep
            if rows_to_keep:
                # Filter the table to keep only non-black frames
                new_table = table.take(rows_to_keep)

                if truncate_prefix:
                    name = path.name.replace("filtered_", "")
                    print(f"[INFO] Truncating prefix for {path.name} to {name}")
                else:
                    name = path.name

                # Create output path with "filtered_" prefix in same directory
                output_path = path.parent / f"filtered_{name}"

                # Write the filtered table
                pq.write_table(new_table, output_path)
                tqdm.write(
                    f"[INFO] Wrote filtered parquet ({rows_removed} rows removed) to {output_path}"
                )

                # Delete original file
                path.unlink()

                tqdm.write(f"[INFO] Deleted original file: {path}")
            else:
                # All rows were black, just delete the file
                path.unlink()
                tqdm.write(
                    f"[INFO] Deleted {path} (all {rows_removed} rows were black frames)"
                )
    else:
        # No black frames found
        if dry_run:
            tqdm.write(f"[DRY-RUN] No black frames in {path}")
        else:
            # Create output path with "filtered_" prefix
            output_path = path.parent / f"filtered_{path.name}"

            # Just copy the table as-is
            pq.write_table(table, output_path)
            tqdm.write(
                f"[INFO] No black frames in {path}, created {output_path}")

            # Delete original file
            path.unlink()
            tqdm.write(f"[INFO] Deleted original file: {path}")

    return rows_removed


def handle_file(path: Path, dry_run: bool, black_threshold: float,
                images_output_dir: Path | None) -> None:
    """Process a single parquet file."""
    try:
        process_file(path,
                     black_threshold=black_threshold,
                     dry_run=dry_run,
                     images_output_dir=images_output_dir)
    except Exception as exc:
        tqdm.write(f"[ERROR] Failed to process {path}: {exc}")


def main(root: Path, dry_run: bool, workers: int,
         black_threshold: float) -> None:
    # Create output directory for black frame images
    images_output_dir = Path.cwd() / f"filtered_{int(black_threshold)}"
    images_output_dir.mkdir(exist_ok=True)
    tqdm.write(f"[INFO] Saving black frames to {images_output_dir}")

    parquet_files = list(root.rglob("*.parquet"))
    if not parquet_files:
        print(f"[INFO] No Parquet files found in {root}")
        return

    workers = max(1, workers)
    with tqdm(total=len(parquet_files), desc="Scanning", unit="file") as bar:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [
                pool.submit(handle_file, fp, dry_run, black_threshold,
                            images_output_dir) for fp in parquet_files
            ]
            for _ in as_completed(futures):
                bar.update()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=
        "Filter black frames from Parquet files, write with 'filtered_' prefix, and delete originals."
    )
    parser.add_argument("--folder", type=Path, help="Root directory to scan")
    parser.add_argument("--dry-run",
                        action="store_true",
                        help="Preview changes only")
    parser.add_argument(
        "--workers",
        type=int,
        default=os.cpu_count() or 32,
        help="Number of worker threads (default: CPU count)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=5.0,
        help="Black frame threshold (default: 5.0)",
    )
    args = parser.parse_args()

    main(args.folder, args.dry_run, args.workers, args.threshold)
