#!/bin/bash
# Benchmark Stage 3 (extract_tracks) + Stage 4 (segment_tracks) end-to-end on the
# motion-stream-test source videos, writing ALL outputs to a separate bench dir so
# the real dataset is never touched.
#
# Usage:
#   bash data_pipeline/benchmark_tracks.sh                 # 50 videos, 4 GPUs
#   GPUS=0 bash data_pipeline/benchmark_tracks.sh          # single GPU
#   LIMIT=5 bash data_pipeline/benchmark_tracks.sh         # smoke run
#   VIZ=1 bash data_pipeline/benchmark_tracks.sh           # include viz mp4 rendering
#   FUSED=1 bash data_pipeline/benchmark_tracks.sh         # fused stage 3+4 (extract --segment)
#
# Results append to $OUT_DIR/benchmark_results.txt (tagged with git commit); each
# entry records gpus/videos/viz/fused so runs stay comparable.
set -euo pipefail

SRC_VIDEOS=${SRC_VIDEOS:-/home/hal-kevin/data/motion-stream-test/videos}
SRC_MANIFEST=${SRC_MANIFEST:-/home/hal-kevin/data/motion-stream-test/videos2caption.json}
OUT_DIR=${OUT_DIR:-/home/hal-kevin/data/motion-stream-qtest}
GPUS=${GPUS:-0,1,2,3}
LIMIT=${LIMIT:-}
VIZ=${VIZ:-0}    # 0 = lean run (production-like for large-scale), 1 = render overlay mp4s
FUSED=${FUSED:-0}  # 1 = single fused pass (extract_tracks --segment); stage 4 not run

cd "$(dirname "$0")/.."
IFS=',' read -ra GPU_ARR <<< "$GPUS"
WORLD_SIZE=${#GPU_ARR[@]}
MANIFEST=bench_manifest.json
LOG="$OUT_DIR/benchmark.log"
RESULTS="$OUT_DIR/benchmark_results.txt"
LIMIT_ARGS=()
[[ -n "$LIMIT" ]] && LIMIT_ARGS=(--limit "$LIMIT")
VIZ_ARGS=()
[[ "$VIZ" == "1" ]] && VIZ_ARGS=(--viz --viz-dir "$OUT_DIR/bench_viz")
FUSED_ARGS=()
if [[ "$FUSED" == "1" ]]; then
    FUSED_ARGS=(--segment --vis-override-every 3)
    [[ "$VIZ" == "1" ]] && FUSED_ARGS+=("${VIZ_ARGS[@]}")
fi

# --- setup: symlink source videos, copy manifest without points_path -------------
mkdir -p "$OUT_DIR"
ln -sfn "$SRC_VIDEOS" "$OUT_DIR/bench_videos"
python - "$SRC_MANIFEST" "$OUT_DIR/$MANIFEST" <<'PY'
import json, sys
items = json.load(open(sys.argv[1]))
for it in items:
    it.pop("points_path", None)  # stage 3 re-patches these to the bench tracks dir
json.dump(items, open(sys.argv[2], "w"), indent=2)
print(f"[bench] manifest: {len(items)} items -> {sys.argv[2]}")
PY
rm -rf "$OUT_DIR/bench_tracks" "$OUT_DIR/bench_viz"
: > "$LOG"

N_VIDEOS=$(ls "$OUT_DIR"/bench_videos/*.mp4 | wc -l)
[[ -n "$LIMIT" ]] && N_VIDEOS=$LIMIT
COMMIT=$(git rev-parse --short HEAD 2>/dev/null || echo unknown)$(git diff --quiet 2>/dev/null || echo -dirty)
echo "[bench] commit=$COMMIT  gpus=$GPUS  videos=$N_VIDEOS  log=$LOG"

wait_workers() {  # wait_workers <stage-name> <pid...>
    local stage=$1 fail=0 pid
    shift
    for pid in "$@"; do
        wait "$pid" || fail=$((fail + 1))
    done
    [[ $fail -gt 0 ]] && echo "[bench] WARNING: $fail $stage worker(s) failed — check $LOG"
    return 0
}

# --- stage 3: extract tracks ------------------------------------------------------
t0=$(date +%s)
PIDS=()
for i in "${!GPU_ARR[@]}"; do
    CUDA_VISIBLE_DEVICES=${GPU_ARR[$i]} python -u data_pipeline/extract_tracks.py \
        --data-dir "$OUT_DIR" \
        --videos-subdir bench_videos \
        --out-subdir bench_tracks \
        --manifest "$MANIFEST" \
        --grid-size 50 \
        --device cuda \
        --detect-entries \
        --sam-conf 0.75 --sam-iou 0.9 --sam-imgsz 1024 \
        --entry-sample-every 2 --entry-min-area 0.001 --entry-new-area 0.5 \
        "${FUSED_ARGS[@]}" \
        --force \
        --rank "$i" --world-size "$WORLD_SIZE" \
        "${LIMIT_ARGS[@]}" \
        >> "$LOG" 2>&1 &
    PIDS+=($!)
done
wait_workers "stage-3" "${PIDS[@]}"
t1=$(date +%s)
S3=$((t1 - t0)); [[ $S3 -eq 0 ]] && S3=1
N_NPZ=$(ls "$OUT_DIR"/bench_tracks/*.npz 2>/dev/null | wc -l || true)
echo "[bench] stage 3: ${S3}s for $N_NPZ npz"
[[ "$N_NPZ" -eq "$N_VIDEOS" ]] || echo "[bench] WARNING: expected $N_VIDEOS npz — check $LOG"

# --- stage 4: segment tracks (skipped when FUSED=1: stage 3 already segmented) -----
S4=0
if [[ "$FUSED" != "1" ]]; then
t2=$(date +%s)
PIDS=()
for i in "${!GPU_ARR[@]}"; do
    CUDA_VISIBLE_DEVICES=${GPU_ARR[$i]} python -u data_pipeline/segment_tracks.py \
        --data-dir "$OUT_DIR" \
        --videos-subdir bench_videos \
        --manifest "$MANIFEST" \
        --conf 0.75 --iou 0.9 --imgsz 1024 \
        --vis-override-every 3 \
        "${VIZ_ARGS[@]}" \
        --force \
        --rank "$i" --world-size "$WORLD_SIZE" \
        "${LIMIT_ARGS[@]}" \
        >> "$LOG" 2>&1 &
    PIDS+=($!)
done
wait_workers "stage-4" "${PIDS[@]}"
t3=$(date +%s)
S4=$((t3 - t2)); [[ $S4 -eq 0 ]] && S4=1
echo "[bench] stage 4: ${S4}s"
fi

# --- summary ----------------------------------------------------------------------
{
    echo "=== $(date -u '+%Y-%m-%d %H:%M:%S') UTC  commit=$COMMIT  gpus=$GPUS  videos=$N_VIDEOS  viz=$VIZ  fused=$FUSED ==="
    awk -v s3="$S3" -v s4="$S4" -v w="$WORLD_SIZE" -v n="$N_VIDEOS" -v fused="$FUSED" 'BEGIN {
        label = (fused == "1") ? "stage 3+4 (fused):" : "stage 3 (extract):"
        printf "%s %5ds total  %6.1fs/video/worker  %5.1f videos/min\n", label, s3, s3*w/n, 60*n/s3
        if (fused != "1")
            printf "stage 4 (segment): %5ds total  %6.1fs/video/worker  %5.1f videos/min\n", s4, s4*w/n, 60*n/s4
        printf "end-to-end:        %5ds total\n", s3+s4
    }'
} | tee -a "$RESULTS"
echo "[bench] results appended to $RESULTS"
