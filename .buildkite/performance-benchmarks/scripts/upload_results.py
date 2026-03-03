"""Persist performance results to the Modal Volume.

Copies results JSON files from the local results/ directory to the
Modal Volume for historical tracking. Updates an index.json aggregate.

Usage:
    python .buildkite/performance-benchmarks/scripts/upload_results.py
"""
import glob
import json
import os
import shutil

RESULTS_DIR = os.path.join("fastvideo", "tests", "performance", "results")
VOLUME_DIR = os.environ.get("PERF_RESULTS_VOLUME", "")
MAX_ENTRIES_PER_BENCHMARK = 200


def main():
    if not VOLUME_DIR:
        print("PERF_RESULTS_VOLUME not set, skipping upload.")
        return

    result_files = glob.glob(os.path.join(RESULTS_DIR, "*.json"))
    if not result_files:
        print("No result files found, skipping upload.")
        return

    os.makedirs(VOLUME_DIR, exist_ok=True)

    # Copy new result files to volume
    for f in result_files:
        dest = os.path.join(VOLUME_DIR, os.path.basename(f))
        shutil.copy2(f, dest)
        print(f"Copied {f} -> {dest}")

    # Update index.json
    index_path = os.path.join(VOLUME_DIR, "index.json")
    existing = []
    if os.path.exists(index_path):
        with open(index_path) as fh:
            existing = json.load(fh)

    # Add new results
    for f in result_files:
        with open(f) as fh:
            existing.append(json.load(fh))

    # Cap per benchmark_id
    by_id = {}
    for entry in existing:
        bid = entry.get("benchmark_id", "unknown")
        by_id.setdefault(bid, []).append(entry)

    trimmed = []
    for bid, entries in by_id.items():
        entries.sort(key=lambda e: e.get("timestamp", ""))
        trimmed.extend(entries[-MAX_ENTRIES_PER_BENCHMARK:])

    trimmed.sort(key=lambda e: e.get("timestamp", ""))

    with open(index_path, "w") as fh:
        json.dump(trimmed, fh, indent=2)

    print(f"Updated {index_path} with {len(trimmed)} total entries")


if __name__ == "__main__":
    main()
