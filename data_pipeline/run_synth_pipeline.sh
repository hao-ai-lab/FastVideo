#!/bin/bash
# Synth toy post-processing: tracks (+SAM) and parquet on the SINGLE synth video dir (not sharded).
# Mirrors run_openvid_shard.sh's PHASE interface for the 720p/24fps synth set produced by
# gen_synth_i2v_worker.py, reusing the same extract_tracks.py --segment and v1_preprocess calls.
#
#   PHASE=tracks  COMPILE=0 bash data_pipeline/run_synth_pipeline.sh   # CoTracker + SAM -> tracks/*.npz
#   PHASE=parquet           bash data_pipeline/run_synth_pipeline.sh   # captions + points_path + Stage 5 -> parquet
#   PHASE=both              bash data_pipeline/run_synth_pipeline.sh
#
# Resumable: extract_tracks skips clips whose npz exists (unless FORCE_TRACKS=1); v1_preprocess
# skips clip ids already written. Run tracks first (across all GPUs), then parquet (single-GPU).
set -uo pipefail
cd "$(dirname "$0")/.."

DATA_ROOT=${DATA_ROOT:-/home/hal-kevin/data/motion-stream-synth}
HEIGHT=${HEIGHT:-720}; WIDTH=${WIDTH:-1280}
NUM_FRAMES=${NUM_FRAMES:-121}; FPS=${FPS:-24}         # MUST match the generated videos (24fps)
GRID=${GRID:-50}
GPUS=${GPUS:-0,1,2,3}
AMP=${AMP:-1}; COMPILE=${COMPILE:-0}; FORCE_TRACKS=${FORCE_TRACKS:-0}
# v1_preprocess only uses the VAE / T5 / CLIP from MODEL_PATH (same Wan2.1 VAE across sizes), so
# any Wan2.1 model dir with those encoders works; override to a lighter one if you have it.
MODEL_PATH=${MODEL_PATH:-/home/hal-kevin/models/Wan2.1-I2V-14B-720P-Diffusers}
PARQUET_WORKERS=${PARQUET_WORKERS:-2}
VID_SUBDIR=videos
PHASE=${PHASE:-both}
case "$PHASE" in
    tracks)  TRACKS=1; PARQUET=0 ;;
    parquet) TRACKS=0; PARQUET=1 ;;
    both)    TRACKS=1; PARQUET=1 ;;
    *) echo "[synth] ERROR: PHASE must be tracks|parquet|both (got '$PHASE')" >&2; exit 1 ;;
esac

IFS=',' read -ra GPU_ARR <<< "$GPUS"; WORLD_SIZE=${#GPU_ARR[@]}
LOG_DIR="$DATA_ROOT/logs"; mkdir -p "$LOG_DIR" "$DATA_ROOT/tracks"
N_VID=$(ls "$DATA_ROOT/$VID_SUBDIR"/*.mp4 2>/dev/null | wc -l || true)
[ "$N_VID" -gt 0 ] || { echo "[synth] no videos under $DATA_ROOT/$VID_SUBDIR -- run gen_synth_i2v_worker first" >&2; exit 1; }
echo "[synth] DATA_ROOT=$DATA_ROOT videos=$N_VID PHASE=$PHASE ${WIDTH}x${HEIGHT}@${FPS}fps x${NUM_FRAMES}f GPUS=$GPUS"

# --- tracks (+ SAM object_ids / track_weights, fused) across all GPUs -------------------
if [[ "$TRACKS" == "1" ]]; then
    echo "[synth] extracting tracks (+segment) across $WORLD_SIZE GPU(s) ..."
    SPEED=(); [[ "$AMP" == "1" ]] && SPEED+=(--amp); [[ "$COMPILE" == "1" ]] && SPEED+=(--compile)
    FORCE_ARGS=(); [[ "$FORCE_TRACKS" == "1" ]] && FORCE_ARGS=(--force)
    if [[ "$COMPILE" == "1" ]]; then
        export TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR:-$HOME/.cache/torchinductor}
        export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-$HOME/.cache/triton}
    fi
    pids=()
    for i in "${!GPU_ARR[@]}"; do
        CUDA_VISIBLE_DEVICES=${GPU_ARR[$i]} python -u data_pipeline/extract_tracks.py \
            --data-dir "$DATA_ROOT" --videos-subdir "$VID_SUBDIR" --out-subdir tracks \
            --grid-size "$GRID" --device cuda \
            --detect-entries --sam-conf 0.75 --sam-iou 0.9 --sam-imgsz 1024 \
            --entry-sample-every 2 --entry-min-area 0.001 --entry-new-area 0.5 \
            --segment --vis-override-every 3 \
            "${SPEED[@]}" "${FORCE_ARGS[@]}" \
            --rank "$i" --world-size "$WORLD_SIZE" \
            >> "$LOG_DIR/tracks.log" 2>&1 &
        pids+=($!)
    done
    tfail=0; for p in "${pids[@]}"; do wait "$p" || tfail=$((tfail + 1)); done
    N_NPZ=$(ls "$DATA_ROOT"/tracks/*.npz 2>/dev/null | wc -l || true)
    echo "[synth] tracks: $N_NPZ/$N_VID npz (failed workers: $tfail) -- log: $LOG_DIR/tracks.log"
    [[ "$tfail" -gt 0 ]] && echo "[synth] WARNING: some track workers failed; inspect the log and re-run (resumable)"
fi

# --- parquet (Stage 5): captions from gen + points_path patch + v1_preprocess -----------
if [[ "$PARQUET" == "1" ]]; then
    N_NPZ=$(ls "$DATA_ROOT"/tracks/*.npz 2>/dev/null | wc -l || true)
    if [[ "$N_NPZ" -lt "$N_VID" ]]; then
        echo "[synth] SKIPPING parquet: tracks incomplete ($N_NPZ/$N_VID) -- run PHASE=tracks first" >&2
        exit 1
    fi
    # 1. compile the gen manifest shards -> videos2caption.json (real captions) + merge.txt
    python data_pipeline/merge_synth_manifests.py --output-dir "$DATA_ROOT"
    # 2. patch points_path (tracks/<stem>.npz) into each entry, preserving the captions
    python - "$DATA_ROOT" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1]); j = root / "videos2caption.json"; td = root / "tracks"
items = json.loads(j.read_text())
n = 0
for it in items:
    # preprocess validation computes num_frames = ceil(fps*duration); the gen manifest omits
    # 'duration', which makes it 0 and rejects every clip -- derive it from num_frames/fps.
    if not it.get("duration") and it.get("fps"):
        it["duration"] = it["num_frames"] / float(it["fps"])
    npz = td / (Path(it["path"]).stem + ".npz")
    if npz.exists():
        it["points_path"] = str(npz.resolve()); n += 1
j.write_text(json.dumps(items, indent=2))
print(f"[synth] manifest: {len(items)} entries, {n} with points_path (duration patched)")
PY
    # 3. Stage 5. --train_fps MUST equal the generated fps (24) or FrameSamplingStage resamples
    #    and the latents stop aligning with the tracks. num_latent_t = (num_frames-1)/4 + 1.
    NLT=$(( (NUM_FRAMES - 1) / 4 + 1 ))
    PQ_OUT="${PARQUET_ROOT:-$DATA_ROOT/preprocessed_i2v_track}"
    echo "[synth] Stage 5: v1_preprocess (${HEIGHT}x${WIDTH}, ${NUM_FRAMES}f, num_latent_t=$NLT, train_fps=$FPS) -> $PQ_OUT"
    CUDA_VISIBLE_DEVICES="${PARQUET_GPU:-${GPU_ARR[0]}}" \
    torchrun --nproc_per_node=1 -m fastvideo.pipelines.preprocess.v1_preprocess \
        --model_path "$MODEL_PATH" \
        --data_merge_path "$DATA_ROOT/merge.txt" \
        --output_dir "$PQ_OUT" \
        --preprocess_task i2v_track \
        --num_frames "$NUM_FRAMES" \
        --num_latent_t "$NLT" \
        --train_fps "$FPS" \
        --max_height "$HEIGHT" \
        --max_width "$WIDTH" \
        --preprocess_video_batch_size 1 \
        --dataloader_num_workers "$PARQUET_WORKERS" \
        --samples_per_file "${PARQUET_SAMPLES:-64}" \
        --flush_frequency "${PARQUET_FLUSH:-8}" \
        2>&1 | tee -a "$LOG_DIR/parquet.log"
    N_PQ=$(find "$PQ_OUT" -name '*.parquet' 2>/dev/null | wc -l || true)
    echo "[synth] parquet: $N_PQ file(s) -> $PQ_OUT/combined_parquet_dataset"
    echo "[synth] point the overfit config data_path at: $PQ_OUT/combined_parquet_dataset"
fi
