#!/bin/bash
# End-to-end preprocessing of one OpenVid-WanTrack shard, WITHOUT the VAE round-trip:
#
#   download shard -> extract -> crop+resize to 720p (CPU, parallel) -> fused tracks (GPU)
#
# NOTE (deliberate, per request): this skips Stage 2's VAE encode/decode and tracks the
# resized source frames directly. The 50-clip A/B (ab_vae_roundtrip.sh) found tracks then
# differ from round-trip tracks by ~5.5px on shared grid points, plus different entry-object
# sets (n_objects differed on 23/50 clips). Fine for a throughput measurement or a training
# A/B; see notes before adopting for a production set.
#
# NOTE (geometry): --height/--width define the coordinate frame the tracks live in. They
# must match what Stage 5 crops/resizes to, or tracks won't align with the latents. 720p
# here is NOT the current training geometry (480x832) -- set HEIGHT/WIDTH accordingly if
# these tracks are meant to feed the existing training config.
#
# Usage:
#   bash data_pipeline/run_openvid_shard.sh                    # shard 0, 720p, 4 GPUs
#   LIMIT=50 bash data_pipeline/run_openvid_shard.sh           # quick smoke run
#   SHARD=3 HEIGHT=480 WIDTH=832 bash data_pipeline/run_openvid_shard.sh
#   SKIP_DOWNLOAD=1 bash data_pipeline/run_openvid_shard.sh    # shard already on disk
set -euo pipefail

REPO_ID=${REPO_ID:-noctuashap/openvid-wantrack-clips}
SHARD=${SHARD:-0}
SHARD_NAME=$(printf "clips-%05d.tar" "$SHARD")
DATA_ROOT=${DATA_ROOT:-$(printf "/home/hal-shared/motionstream/data/openvid-wantrack/shard%03d" "$SHARD")}
HEIGHT=${HEIGHT:-720}
WIDTH=${WIDTH:-1280}
NUM_FRAMES=${NUM_FRAMES:-121}
FPS=${FPS:-24}
GPUS=${GPUS:-0,1,2,3}
CPU_WORKERS=${CPU_WORKERS:-$(( $(nproc) > 16 ? 16 : $(nproc) ))}
LIMIT=${LIMIT:-}
AMP=${AMP:-1}          # bf16 CoTracker (validated: ~1.1px delta, ~1.5x faster)
COMPILE=${COMPILE:-0}  # torch.compile main pass (~10-15%; pays a per-worker warmup)
VIZ=${VIZ:-0}
SKIP_DOWNLOAD=${SKIP_DOWNLOAD:-0}
TRACKS=${TRACKS:-1}              # 0 = skip tracking (parquet-only pass over existing npz)
PARQUET=${PARQUET:-0}            # 1 = also run Stage 5 (v1_preprocess) -> training parquets
# Stage 5's dataloader defaults to 1 worker, so video decode blocks the GPU between clips
# (the same bubble --prefetch removes in tracking). More workers overlap decode with encode,
# but each holds a decoded 720p clip (~334MB of raw frames), so this also drives host RAM.
PARQUET_WORKERS=${PARQUET_WORKERS:-2}
# How many samples the parquet writer buffers before flushing to disk. At 720p each sample's
# latents are ~2.4x the 480p reference, so the default (256) can OOM a 127GB node. Flushing
# every samples_per_file keeps the in-RAM buffer small.
PARQUET_FLUSH=${PARQUET_FLUSH:-64}
MODEL_PATH=${MODEL_PATH:-/home/hal-kevin/models/trackwan_1.3b_i2v_control_init}
RESUME=${RESUME:-0}              # 1 = continue an interrupted run: skip finished phases AND
                                 #     already-tracked clips (implies FORCE_TRACKS=0)
FORCE_TRACKS=${FORCE_TRACKS:-$([[ "$RESUME" == "1" ]] && echo 0 || echo 1)}
DRY_RUN=${DRY_RUN:-0}            # 1 = print the worker command lines and exit (no work done)

cd "$(dirname "$0")/.."
IFS=',' read -ra GPU_ARR <<< "$GPUS"
WORLD_SIZE=${#GPU_ARR[@]}
DL_DIR="$DATA_ROOT/download"
RAW_DIR="$DATA_ROOT/raw_videos"
LOG_DIR="$DATA_ROOT/logs"
mkdir -p "$DATA_ROOT" "$LOG_DIR"

# Two concurrent runs would interleave resize writes with the tracker's video glob, so the
# tracker would see a partially-populated dir and silently process a subset. Refuse to overlap.
LOCK="$DATA_ROOT/.run.lock"
if ! mkdir "$LOCK" 2>/dev/null; then
    echo "[openvid] ERROR: another run holds $LOCK (remove it if stale)" >&2; exit 1
fi
trap 'rmdir "$LOCK" 2>/dev/null || true' EXIT

LIMIT_ARGS=(); [[ -n "$LIMIT" ]] && LIMIT_ARGS=(--limit "$LIMIT")
secs() { date +%s; }
hms() { awk -v s="$1" 'BEGIN{printf "%dm%02ds", s/60, s%60}'; }

# --- progress tracking: survives SIGKILL (cluster reapers), enables RESUME=1 --------
PROGRESS="$DATA_ROOT/progress.json"
prog_set() {  # prog_set <phase> <json-object>
    python - "$PROGRESS" "$1" "$2" "$SHARD_NAME" "${HEIGHT}x${WIDTH}@${NUM_FRAMES}f" <<'PY'
import datetime, json, sys
from pathlib import Path
p, phase, payload, shard, target = Path(sys.argv[1]), sys.argv[2], json.loads(sys.argv[3]), sys.argv[4], sys.argv[5]
d = json.loads(p.read_text()) if p.exists() else {}
d.update(shard=shard, target=target)   # informational: the most recent run's target
now = datetime.datetime.now().isoformat(timespec="seconds")
# `target` is recorded PER PHASE: a later run at a different geometry must not be able to
# reuse filter/track outputs produced for the old one.
d.setdefault("phases", {})[phase] = {**payload, "target": target, "ts": now}
d["updated"] = now
tmp = p.with_suffix(".json.tmp")            # atomic: a kill mid-write must not corrupt state
tmp.write_text(json.dumps(d, indent=2))
tmp.replace(p)
PY
}
prog_done() {  # prog_done <phase> -> 0 if that phase completed for this target
    python - "$PROGRESS" "$1" "${HEIGHT}x${WIDTH}@${NUM_FRAMES}f" <<'PY'
import json, sys
from pathlib import Path
p = Path(sys.argv[1])
if not p.exists():
    sys.exit(1)
ph = json.loads(p.read_text()).get("phases", {}).get(sys.argv[2], {})
# a different target geometry invalidates that phase's outputs
sys.exit(0 if (ph.get("done") and ph.get("target") == sys.argv[3]) else 1)
PY
}
[[ "$RESUME" == "1" ]] && echo "[openvid] RESUME=1 -- finished phases and already-tracked clips will be skipped"

echo "[openvid] shard=$SHARD_NAME  target=${HEIGHT}x${WIDTH}  gpus=$GPUS  cpu_workers=$CPU_WORKERS"
echo "[openvid] data root: $DATA_ROOT"

# --- 1. download -------------------------------------------------------------------
if [[ "$SKIP_DOWNLOAD" != "1" && ! -f "$DL_DIR/$SHARD_NAME" ]]; then
    echo "[openvid] downloading $SHARD_NAME (~3.3 GB) ..."
    t=$(secs)
    if command -v hf >/dev/null 2>&1; then
        hf download "$REPO_ID" "$SHARD_NAME" --repo-type dataset --local-dir "$DL_DIR"
    elif command -v huggingface-cli >/dev/null 2>&1; then
        huggingface-cli download "$REPO_ID" "$SHARD_NAME" --repo-type dataset --local-dir "$DL_DIR"
    else
        echo "[openvid] ERROR: neither 'hf' nor 'huggingface-cli' found (pip install -U huggingface_hub)" >&2
        exit 1
    fi
    echo "[openvid] download: $(hms $(( $(secs) - t )))"
    prog_set download "{\"done\": true, \"secs\": $(( $(secs) - t ))}"
else
    echo "[openvid] download: skipped (have $DL_DIR/$SHARD_NAME)"
    prog_set download '{"done": true, "note": "pre-existing"}'
fi

# --- 2. extract --------------------------------------------------------------------
if [[ -z "$(ls -A "$RAW_DIR" 2>/dev/null)" ]]; then
    echo "[openvid] extracting ..."
    t=$(secs)
    mkdir -p "$RAW_DIR"
    tar -xf "$DL_DIR/$SHARD_NAME" -C "$RAW_DIR"
    # flatten any nested layout so *.mp4 all sit directly in RAW_DIR
    find "$RAW_DIR" -mindepth 2 -name '*.mp4' -exec mv -t "$RAW_DIR" {} + 2>/dev/null || true
    find "$RAW_DIR" -mindepth 1 -type d -empty -delete 2>/dev/null || true
    echo "[openvid] extract: $(hms $(( $(secs) - t )))"
else
    echo "[openvid] extract: skipped (raw_videos/ non-empty)"
fi
N_RAW=$(ls "$RAW_DIR"/*.mp4 2>/dev/null | wc -l || true)
echo "[openvid] raw clips: $N_RAW"
[[ "$N_RAW" -gt 0 ]] || { echo "[openvid] ERROR: no mp4s extracted" >&2; exit 1; }
prog_set extract "{\"done\": true, \"raw_clips\": $N_RAW}"

# --- 3. crop+resize to target geometry (CPU, parallel) ------------------------------
# If the clips are already at the target geometry, resizing is a pure lossy re-encode
# (measured 40 dB / 1.9-per-255 on this shard -- the same order as the VAE round-trip's
# distortion, for zero benefit) plus a duplicate copy on disk. Track the raw clips instead.
VID_SUBDIR=videos
if [[ "$RESUME" == "1" ]] && prog_done filter && [[ -n "$(ls -A "$DATA_ROOT/videos" 2>/dev/null)" ]]; then
    T_RESIZE=0
    N_VID=$(ls "$DATA_ROOT"/videos/*.mp4 2>/dev/null | wc -l || true)
    echo "[openvid] filter: skipped (progress.json says done; $N_VID clips staged)"
elif [[ "${FORCE_RESIZE:-0}" != "1" ]]; then
    # Scan every clip's metadata (no decode) and symlink through the conforming ones.
    # Clips at other resolutions/lengths are skipped and listed in skipped_clips.json.
    t=$(secs)
    python -u data_pipeline/filter_clips.py \
        --src-dir "$RAW_DIR" --out-dir "$DATA_ROOT/videos" \
        --height "$HEIGHT" --width "$WIDTH" --num-frames "$NUM_FRAMES" --clean \
        2>&1 | tee -a "$LOG_DIR/filter.log"
    # Rescue pass: clips that are readable but off-spec get re-encoded to the target
    # (only these -- the conforming majority stays symlinked, never re-encoded).
    NEEDS="$DATA_ROOT/needs_resize.txt"
    N_RESCUE=$(wc -l < "$NEEDS" 2>/dev/null || echo 0)
    if [[ "$N_RESCUE" -gt 0 ]]; then
        echo "[openvid] rescuing $N_RESCUE off-spec clip(s) by resize -> ${HEIGHT}x${WIDTH} ..."
        pids=()
        for i in $(seq 0 $((CPU_WORKERS - 1))); do
            python -u data_pipeline/resize_videos.py \
                --data-dir "$DATA_ROOT" \
                --video-subdir raw_videos \
                --out-subdir videos \
                --include-list "$NEEDS" \
                --height "$HEIGHT" --width "$WIDTH" \
                --num-frames "$NUM_FRAMES" --fps "$FPS" \
                --rank "$i" --world-size "$CPU_WORKERS" \
                >> "$LOG_DIR/resize.log" 2>&1 &
            pids+=($!)
        done
        rfail=0; for p in "${pids[@]}"; do wait "$p" || rfail=$((rfail + 1)); done
        [[ $rfail -gt 0 ]] && echo "[openvid] WARNING: $rfail rescue worker(s) failed -- see $LOG_DIR/resize.log"
    fi
    T_RESIZE=$(( $(secs) - t ))
    N_VID=$(ls "$DATA_ROOT"/videos/*.mp4 2>/dev/null | wc -l || true)
    N_SKIP=$(( N_RAW - N_VID ))
    echo "[openvid] filter: $(hms $T_RESIZE)   conforming: $N_VID / $N_RAW   skipped: $N_SKIP"
    # only record success if something was actually staged
    [[ "$N_VID" -gt 0 ]] && prog_set filter "{\"done\": true, \"kept\": $N_VID, \"skipped\": $N_SKIP, \"secs\": $T_RESIZE}"
    if [[ "$N_VID" -eq 0 ]]; then
        echo "[openvid] ERROR: no clips match ${HEIGHT}x${WIDTH}@${NUM_FRAMES}f." >&2
        echo "[openvid]        Run with FORCE_RESIZE=1 to re-encode them to the target instead." >&2
        exit 1
    fi
else
echo "[openvid] resizing $NATIVE -> ${HEIGHT}x${WIDTH} across $CPU_WORKERS CPU workers ..."
t=$(secs)
pids=()
for i in $(seq 0 $((CPU_WORKERS - 1))); do
    python -u data_pipeline/resize_videos.py \
        --data-dir "$DATA_ROOT" \
        --video-subdir raw_videos \
        --out-subdir videos \
        --height "$HEIGHT" --width "$WIDTH" \
        --num-frames "$NUM_FRAMES" --fps "$FPS" \
        --rank "$i" --world-size "$CPU_WORKERS" \
        "${LIMIT_ARGS[@]}" \
        >> "$LOG_DIR/resize.log" 2>&1 &
    pids+=($!)
done
rfail=0; for p in "${pids[@]}"; do wait "$p" || rfail=$((rfail + 1)); done
[[ $rfail -gt 0 ]] && echo "[openvid] WARNING: $rfail resize worker(s) failed -- see $LOG_DIR/resize.log"
T_RESIZE=$(( $(secs) - t ))
find "$DATA_ROOT/$VID_SUBDIR" -name '.*.tmp.mp4' -delete 2>/dev/null || true   # drop any stale temps
N_VID=$(ls "$DATA_ROOT"/$VID_SUBDIR/*.mp4 2>/dev/null | wc -l || true)
echo "[openvid] resize: $(hms $T_RESIZE)   usable clips: $N_VID / $N_RAW"
fi
[[ "$N_VID" -gt 0 ]] || { echo "[openvid] ERROR: no clips available to track" >&2; exit 1; }

# --- 4. manifest (so tracks get points_path patched + Stage 5 has an entry point) ---
python - "$DATA_ROOT" "$FPS" "$NUM_FRAMES" "$HEIGHT" "$WIDTH" "$VID_SUBDIR" <<'PY'
import json, sys
from pathlib import Path
root, fps, nf, h, w = Path(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
tracks_dir = root / "tracks"
items = []
for i, p in enumerate(sorted((root / sys.argv[6]).glob("*.mp4"))):
    it = {"idx": i, "path": p.name, "cap": [""], "fps": fps, "num_frames": nf,
          "duration": nf / fps, "resolution": {"width": w, "height": h}}
    # This manifest gets rewritten every run, so re-attach points_path here whenever the
    # npz exists. In PHASE=parquet, tracking is skipped and never patches it back, so Stage 5
    # would otherwise see no track sidecar (PreprocessPipeline_I2V_Track then errors).
    npz = tracks_dir / f"{p.stem}.npz"
    if npz.exists():
        it["points_path"] = str(npz.resolve())
    items.append(it)
(root / "videos2caption.json").write_text(json.dumps(items, indent=2))
n_pts = sum(1 for it in items if "points_path" in it)
print(f"[openvid] manifest: {len(items)} entries ({n_pts} with points_path) -> {root/'videos2caption.json'}")
PY

# Real captions from OpenVid-1M (joins 1:1 on clip filename). Without this every clip
# carries an empty prompt and Stage 5 bakes identical null T5 embeddings into the parquets.
if [[ "${CAPTIONS:-1}" == "1" ]]; then
    python -u data_pipeline/add_captions.py \
        --manifest "$DATA_ROOT/videos2caption.json" \
        --min-coverage "${MIN_CAPTION_COVERAGE:-0.9}" \
        2>&1 | tee -a "$LOG_DIR/captions.log" | tail -3
    cap_rc=${PIPESTATUS[0]}
    if [[ "$cap_rc" -ne 0 ]]; then
        echo "[openvid] ERROR: caption join failed -- see $LOG_DIR/captions.log" >&2
        echo "[openvid]        set CAPTIONS=0 to proceed with empty captions (tracks are still valid;" >&2
        echo "[openvid]        parquets built from them would have dead text conditioning)." >&2
        exit 1
    fi
fi

# --- 5. fused tracks (stages 3+4 in one pass, no VAE round-trip) --------------------
if [[ "$TRACKS" != "1" ]]; then
    T_TRACK=0; tfail=0
    N_NPZ=0
    N_NPZ_ALL=$(ls "$DATA_ROOT"/tracks/*.npz 2>/dev/null | wc -l || true)
    echo "[openvid] tracks: SKIPPED (TRACKS=0)   existing npz: $N_NPZ_ALL / $N_VID"
else
echo "[openvid] extracting tracks across $WORLD_SIZE GPUs ..."
SPEED=(); [[ "$AMP" == "1" ]] && SPEED+=(--amp); [[ "$COMPILE" == "1" ]] && SPEED+=(--compile)
VIZ_ARGS=(); [[ "$VIZ" == "1" ]] && VIZ_ARGS=(--viz --viz-dir "$DATA_ROOT/viz")
if [[ "$COMPILE" == "1" ]]; then
    export TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR:-$HOME/.cache/torchinductor}
    export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-$HOME/.cache/triton}
fi
FORCE_ARGS=(); [[ "$FORCE_TRACKS" == "1" ]] && FORCE_ARGS=(--force)
if [[ "$DRY_RUN" == "1" ]]; then
    echo "[openvid] DRY RUN -- rank 0 track worker would be:"
    echo "  CUDA_VISIBLE_DEVICES=${GPU_ARR[0]} python -u data_pipeline/extract_tracks.py" \
         "--data-dir $DATA_ROOT --videos-subdir $VID_SUBDIR --out-subdir tracks" \
         "--grid-size 50 --device cuda --detect-entries --sam-conf 0.75 --sam-iou 0.9 --sam-imgsz 1024" \
         "--entry-sample-every 2 --entry-min-area 0.001 --entry-new-area 0.5" \
         "--segment --vis-override-every 3 ${SPEED[*]} ${VIZ_ARGS[*]} ${FORCE_ARGS[*]} ${LIMIT_ARGS[*]}" \
         "--rank 0 --world-size $WORLD_SIZE"
    exit 0
fi
t=$(secs)
pids=()
for i in "${!GPU_ARR[@]}"; do
    CUDA_VISIBLE_DEVICES=${GPU_ARR[$i]} python -u data_pipeline/extract_tracks.py \
        --data-dir "$DATA_ROOT" \
        --videos-subdir "$VID_SUBDIR" \
        --out-subdir tracks \
        --grid-size 50 --device cuda \
        --detect-entries --sam-conf 0.75 --sam-iou 0.9 --sam-imgsz 1024 \
        --entry-sample-every 2 --entry-min-area 0.001 --entry-new-area 0.5 \
        --segment --vis-override-every 3 \
        "${SPEED[@]}" "${VIZ_ARGS[@]}" "${FORCE_ARGS[@]}" "${LIMIT_ARGS[@]}" \
        --rank "$i" --world-size "$WORLD_SIZE" \
        >> "$LOG_DIR/tracks.log" 2>&1 &
    pids+=($!)
done
tfail=0; for p in "${pids[@]}"; do wait "$p" || tfail=$((tfail + 1)); done
[[ $tfail -gt 0 ]] && echo "[openvid] WARNING: $tfail track worker(s) failed -- see $LOG_DIR/tracks.log"
T_TRACK=$(( $(secs) - t ))
# Count only npz written by THIS run -- counting the whole dir would fold in earlier runs
# and (with FORCE_TRACKS=0) report a rate for work that was skipped.
N_NPZ=$(find "$DATA_ROOT/tracks" -name '*.npz' -newermt "@$t" 2>/dev/null | wc -l || true)
N_NPZ_ALL=$(ls "$DATA_ROOT"/tracks/*.npz 2>/dev/null | wc -l || true)
echo "[openvid] tracks: $(hms $T_TRACK)   npz this run: $N_NPZ   total in dir: $N_NPZ_ALL / $N_VID"
if [[ "$N_NPZ_ALL" -ge "$N_VID" && "$tfail" -eq 0 && -z "$LIMIT" ]]; then
    prog_set tracks "{\"done\": true, \"npz\": $N_NPZ_ALL, \"clips\": $N_VID, \"secs\": $T_TRACK}"
    echo "[openvid] shard COMPLETE"
else
    prog_set tracks "{\"done\": false, \"npz\": $N_NPZ_ALL, \"clips\": $N_VID, \"secs\": $T_TRACK}"
    [[ "$N_NPZ_ALL" -lt "$N_VID" ]] && \
        echo "[openvid] INCOMPLETE: $(( N_VID - N_NPZ_ALL )) clips remain -- resume with: RESUME=1 SKIP_DOWNLOAD=1 bash $0"
fi
fi

# --- 6. Stage 5: parquets (opt-in; the tracks above are already usable without this) --
T_PARQUET=0
if [[ "$PARQUET" == "1" ]]; then
    if [[ "$N_NPZ_ALL" -lt "$N_VID" ]]; then
        echo "[openvid] SKIPPING parquets: tracks incomplete ($N_NPZ_ALL/$N_VID)" >&2
    else
        # num_latent_t = (num_frames - 1)/4 + 1 for WanVAE's 4x temporal compression.
        NLT=$(( (NUM_FRAMES - 1) / 4 + 1 ))
        echo "$DATA_ROOT/$VID_SUBDIR,$DATA_ROOT/videos2caption.json" > "$DATA_ROOT/data_merge.txt"
        # Parquet output location. Default keeps it beside the shard's other data; set
        # PARQUET_ROOT to collect all shards' parquets under one tree (their own dir),
        # each in a per-shard subdir so shards stay independent (wipe/verify are per-shard).
        if [[ -n "${PARQUET_ROOT:-}" ]]; then
            PQ_OUT="$PARQUET_ROOT/$(printf 'shard%03d' "$SHARD")"
        else
            PQ_OUT="$DATA_ROOT/preprocessed_i2v_track"
        fi
        # Stage 5 now resumes by clip id (preprocess_pipeline_base.py): a re-run skips clips
        # already written and appends only the rest, so a reaper kill costs minutes (the
        # unflushed buffer), not the whole shard. Do NOT wipe -- that would discard progress.
        echo "[openvid] Stage 5: parquets (${HEIGHT}x${WIDTH}, ${NUM_FRAMES}f, num_latent_t=$NLT, train_fps=$FPS) ..."
        t=$(secs)
        # --train_fps MUST equal the source fps: a mismatch makes FrameSamplingStage resample
        # (duplicating/dropping frames) so latents no longer align with the tracks in the same row.
        # v1_preprocess asserts num_gpus == 1 (fastvideo/pipelines/preprocess/v1_preprocess.py:27),
        # so Stage 5 is single-GPU per shard. Scale it by running shards concurrently
        # (one GPU each) rather than by raising nproc_per_node.
        CUDA_VISIBLE_DEVICES="${PARQUET_GPU:-${GPU_ARR[0]}}" \
        torchrun --nproc_per_node=1 -m fastvideo.pipelines.preprocess.v1_preprocess \
            --model_path "$MODEL_PATH" \
            --data_merge_path "$DATA_ROOT/data_merge.txt" \
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
            --flush_frequency "$PARQUET_FLUSH" \
            >> "$LOG_DIR/parquet.log" 2>&1
        prc=$?
        T_PARQUET=$(( $(secs) - t ))
        N_PQ=$(find "$PQ_OUT" -name '*.parquet' 2>/dev/null | wc -l || true)
        # Verify integrity: total rows must equal clip count AND be free of duplicate ids.
        # A reaper kill mid-run leaves a partial (rows < N_VID) which the next attempt wipes+redoes.
        read -r N_ROWS N_UNIQ < <(python - "$PQ_OUT" <<'PY'
import glob, sys
import pyarrow.parquet as pq
ids = []
for f in glob.glob(f"{sys.argv[1]}/**/*.parquet", recursive=True):
    ids += pq.read_table(f, columns=["id"]).column("id").to_pylist()
print(len(ids), len(set(ids)))
PY
)
        if [[ $prc -eq 0 && "$N_ROWS" == "$N_VID" && "$N_UNIQ" == "$N_VID" ]]; then
            echo "[openvid] parquets: $(hms $T_PARQUET)   rows=$N_ROWS unique=$N_UNIQ files=$N_PQ"
            prog_set parquet "{\"done\": true, \"rows\": $N_ROWS, \"files\": $N_PQ, \"secs\": $T_PARQUET}"
        else
            echo "[openvid] Stage 5 INCOMPLETE (rc=$prc, rows=$N_ROWS unique=$N_UNIQ, want $N_VID) -- see $LOG_DIR/parquet.log" >&2
            prog_set parquet "{\"done\": false, \"rows\": $N_ROWS, \"unique\": $N_UNIQ, \"secs\": $T_PARQUET}"
        fi
    fi
fi

# --- 7. summary --------------------------------------------------------------------
RESULTS="$DATA_ROOT/shard_results.txt"
{
    echo "=== $(date -u '+%Y-%m-%d %H:%M:%S') UTC  shard=$SHARD_NAME  ${HEIGHT}x${WIDTH}  gpus=$GPUS  cpu=$CPU_WORKERS  amp=$AMP compile=$COMPILE viz=$VIZ  (no VAE round-trip) ==="
    awk -v r="$T_RESIZE" -v tk="$T_TRACK" -v n="$N_NPZ" -v w="$WORLD_SIZE" -v c="$CPU_WORKERS" 'BEGIN {
        if (n == 0) { print "no npz produced"; exit }
        printf "resize (CPU): %6ds  %5.2fs/clip/worker\n", r, r*c/n
        printf "tracks (GPU): %6ds  %5.2fs/clip/worker  %5.1f clips/min\n", tk, tk*w/n, 60*n/tk
        printf "total:        %6ds for %d clips\n", r+tk, n
        printf "  -> 259k clips on %d GPUs (tracks only): %.1f h\n", w, 259000*(tk*w/n)/w/3600
        printf "  -> full shard (1000 clips) at this rate: %.1f min\n", (tk/n)*1000/60
    }'
    # `|| true`: a false [[ ]] would make this block (and so the piped tee) exit non-zero
    # under `set -e -o pipefail`, failing the whole shard after the work already succeeded.
    { [[ "$VIZ" == "1" ]] && echo "  NOTE: viz=1 -- overlay rendering dominates the GPU phase (~4x); projections are pessimistic, not a throughput measurement."; } || true
    { [[ -n "$LIMIT" ]] && echo "  NOTE: limit=$LIMIT -- per-worker startup is a large share at this size; totals understate steady-state throughput."; } || true
} | tee -a "$RESULTS"
echo "[openvid] appended to $RESULTS"
echo "[openvid] outputs: $DATA_ROOT/{videos,tracks}$([[ "$VIZ" == "1" ]] && echo ",viz")   logs: $LOG_DIR"
