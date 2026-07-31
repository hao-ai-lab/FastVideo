#!/bin/bash
# Phase 0, step 0 — carve a SMALL overfit dataset out of the 720p openvid parquets.
# Symlinks a few data_chunk parquets into a dedicated dir; the map-style loader walks it for
# *.parquet, so a couple of chunks (~32 clips each) is a good overfit set. Non-destructive
# (symlinks only). Re-run to rebuild; it clears the stale map_style_cache.
set -uo pipefail

SRC_ROOT=${SRC_ROOT:-/home/hal-shared/motionstream/data/openvid-wantrack-parquets}
SRC_SHARD=${SRC_SHARD:-shard000}
N_CHUNKS=${N_CHUNKS:-2}                       # ~32 clips/chunk -> ~64 clips
OUT=${OUT:-/home/hal-kevin/data/motion-stream-test/overfit_subset_720p/combined_parquet_dataset}

src_worker="$SRC_ROOT/$SRC_SHARD/combined_parquet_dataset/worker_0"
[ -d "$src_worker" ] || { echo "[subset] source not found: $src_worker" >&2; exit 1; }

dst_worker="$OUT/worker_0"
rm -rf "$OUT"                                 # drop old subset + its map_style_cache
mkdir -p "$dst_worker"

n=0
for f in $(ls "$src_worker"/data_chunk_*.parquet | sort -V | head -n "$N_CHUNKS"); do
  ln -s "$(readlink -f "$f")" "$dst_worker/$(basename "$f")"
  n=$((n + 1))
done

echo "[subset] linked $n parquet chunk(s) from $SRC_SHARD -> $OUT"
python - "$OUT" <<'PY'
import glob, sys, pyarrow.parquet as pq
fs = glob.glob(f"{sys.argv[1]}/**/*.parquet", recursive=True)
rows = sum(pq.ParquetFile(f).metadata.num_rows for f in fs)
print(f"[subset] {len(fs)} file(s), {rows} clips total")
PY
echo "[subset] point the overfit config data_path at: $OUT"
