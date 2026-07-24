#!/bin/bash
# Drive run_openvid_shard.sh over a range of shards, with resume and per-shard accounting.
#
# Every shard runs with RESUME=1, so re-invoking after an interruption (cluster reaper,
# node loss, Ctrl-C) picks up exactly where it stopped: finished shards are skipped via
# their progress.json, and a half-finished shard resumes at the first untracked clip.
#
# Usage:
#   SHARDS=0-130 bash data_pipeline/run_openvid_shards.sh
#   SHARDS=0-9,20,30-35 bash data_pipeline/run_openvid_shards.sh
#   SHARDS=0-130 CLEANUP=1 bash data_pipeline/run_openvid_shards.sh   # drop tar+raw after each
#   SHARDS=0-3 LIMIT=20 bash data_pipeline/run_openvid_shards.sh      # smoke run
#
# Passes through the per-shard knobs (GPUS, HEIGHT/WIDTH, AMP, COMPILE, VIZ, CPU_WORKERS,
# LIMIT, DATA_ROOT_BASE); see run_openvid_shard.sh for their meanings.
set -uo pipefail          # NOT -e: one bad shard must not kill a 130-shard run

SHARDS=${SHARDS:-0}
CLEANUP=${CLEANUP:-0}       # 1 = delete download/ and raw_videos/ once a shard completes
STOP_ON_FAIL=${STOP_ON_FAIL:-0}
# PARALLEL=1 runs one shard pipeline per GPU concurrently instead of one shard at a time
# across all GPUs. Stage 5 (v1_preprocess) asserts a single GPU, so sequential mode leaves
# 3 of 4 GPUs idle for the whole parquet phase; shard-level parallelism keeps them all busy.
PARALLEL=${PARALLEL:-0}
GPUS=${GPUS:-0,1,2,3}
# PHASE picks what this invocation produces:
#   tracks  -- tracks only, one shard at a time across all GPUs (fastest per-shard: ~13 min)
#   parquet -- Stage 5 only, over shards that already have tracks; forces PARALLEL=1 because
#              v1_preprocess is single-GPU, so concurrency has to come from running shards
#   both    -- everything per shard (honours PARALLEL as set)
PHASE=${PHASE:-both}
case "$PHASE" in
    tracks)  export TRACKS=1 PARQUET=0 ;;
    parquet) export TRACKS=0 PARQUET=1; PARALLEL=1 ;;
    both)    export TRACKS=1 ;;
    *) echo "[shards] ERROR: PHASE must be tracks|parquet|both (got '$PHASE')" >&2; exit 1 ;;
esac
DATA_ROOT_BASE=${DATA_ROOT_BASE:-/home/hal-shared/motionstream/data/openvid-wantrack/shard}
shard_root() { printf "%s%03d" "$DATA_ROOT_BASE" "$1"; }   # zero-padded: shard000 .. shard259

cd "$(dirname "$0")/.."
SHARD_SCRIPT=data_pipeline/run_openvid_shard.sh

# --- expand "0-9,20,30-35" into a list ---------------------------------------------
expand() {
    local spec=$1 tok lo hi out=()
    IFS=',' read -ra toks <<< "$spec"
    for tok in "${toks[@]}"; do
        if [[ "$tok" =~ ^([0-9]+)-([0-9]+)$ ]]; then
            lo=${BASH_REMATCH[1]}; hi=${BASH_REMATCH[2]}
            (( lo <= hi )) || { echo "[shards] ERROR: bad range '$tok'" >&2; exit 1; }
            for ((i = lo; i <= hi; i++)); do out+=("$i"); done
        elif [[ "$tok" =~ ^[0-9]+$ ]]; then
            out+=("$tok")
        else
            echo "[shards] ERROR: cannot parse '$tok' (want N or A-B, comma-separated)" >&2; exit 1
        fi
    done
    printf '%s\n' "${out[@]}"
}
mapfile -t SHARD_LIST < <(expand "$SHARDS")
N_SHARDS=${#SHARD_LIST[@]}

# 260 shards exist: clips-00000.tar .. clips-00259.tar
for s in "${SHARD_LIST[@]}"; do
    (( s <= 259 )) || { echo "[shards] ERROR: shard $s out of range (max 259)" >&2; exit 1; }
done

# A shard counts as complete only once every phase this run is producing has finished --
# with PARQUET=1 that includes Stage 5, so a shard whose tracks landed but whose parquets
# failed is retried rather than skipped.
case "$PHASE" in
    tracks)  REQUIRED_PHASES="tracks" ;;
    parquet) REQUIRED_PHASES="parquet" ;;          # tracks were the previous pass's job
    both)    REQUIRED_PHASES="tracks"; [[ "${PARQUET:-0}" == "1" ]] && REQUIRED_PHASES="tracks parquet" ;;
esac
shard_complete() {  # shard_complete <n> -> 0 if all required phases are done
    python - "$(shard_root "$1")/progress.json" $REQUIRED_PHASES <<'PY'
import json, sys
from pathlib import Path
p = Path(sys.argv[1])
if not p.exists():
    sys.exit(1)
phases = json.loads(p.read_text()).get("phases", {})
sys.exit(0 if all(phases.get(ph, {}).get("done") for ph in sys.argv[2:]) else 1)
PY
}

# Roll-up across all shards, so overall progress is one file rather than 131.
OVERALL=${OVERALL:-$(dirname "$DATA_ROOT_BASE")/progress.json}
overall_set() {  # overall_set <done> <skipped> <failed> <total> <clips> <elapsed> <failed-list>
    python - "$OVERALL" "$@" "$DATA_ROOT_BASE" "$SHARDS" <<'PY'
import datetime, json, os, sys
from pathlib import Path
p = Path(sys.argv[1])
done, skipped, failed, total, clips, elapsed = (int(x) for x in sys.argv[2:8])
failed_list, base, spec = sys.argv[8], sys.argv[9], sys.argv[10]
processed = done + skipped
d = {
    "spec": spec, "data_root_base": base,
    "shards_total": total, "complete": processed, "processed_this_session": done,
    "skipped_already_done": skipped, "failed": failed,
    "failed_shards": [int(x) for x in failed_list.split() if x],
    "npz_this_session": clips,
    "elapsed_min": round(elapsed / 60, 1),
    "avg_min_per_shard": round(elapsed / done / 60, 1) if done else None,
    "eta_hours": round((elapsed / done) * (total - processed - failed) / 3600, 1) if done else None,
    "updated": datetime.datetime.now().isoformat(timespec="seconds"),
}
tmp = p.with_suffix(f".json.tmp{os.getpid()}"); tmp.write_text(json.dumps(d, indent=2)); tmp.replace(p)
PY
}

echo "[shards] $N_SHARDS shard(s): ${SHARD_LIST[0]}..${SHARD_LIST[-1]}   phase=$PHASE  parallel=$PARALLEL  cleanup=$CLEANUP"
echo "[shards] overall progress: $OVERALL   per-shard: ${DATA_ROOT_BASE}<N>/progress.json"

# ---- parallel mode: one shard pipeline per GPU -------------------------------------
if [[ "$PARALLEL" == "1" ]]; then
    IFS=',' read -ra GPU_ARR <<< "$GPUS"
    NG=${#GPU_ARR[@]}
    echo "[shards] parallel: $NG concurrent pipelines, one GPU each (${GPUS})"
    T0=$(date +%s)
    wpids=()
    for gi in "${!GPU_ARR[@]}"; do
        (
            gpu=${GPU_ARR[$gi]}
            mine=()
            for ((j = gi; j < N_SHARDS; j += NG)); do mine+=("${SHARD_LIST[$j]}"); done
            echo "[gpu$gpu] ${#mine[@]} shard(s): ${mine[*]:0:6}$([[ ${#mine[@]} -gt 6 ]] && echo ' ...')"
            for s in "${mine[@]}"; do
                root="$(shard_root "$s")"
                if shard_complete "$s"; then echo "[gpu$gpu] shard $s already complete"; continue; fi
                t=$(date +%s)
                if SHARD="$s" RESUME=1 DATA_ROOT="$root" GPUS="$gpu" PARQUET_GPU="$gpu" \
                       bash "$SHARD_SCRIPT" >> "$root.log" 2>&1; then
                    echo "[gpu$gpu] shard $s OK in $(( ($(date +%s) - t) / 60 ))m"
                    [[ "$CLEANUP" == "1" ]] && shard_complete "$s" && rm -rf "$root/download" "$root/raw_videos"
                else
                    echo "[gpu$gpu] shard $s FAILED -- see $root.log" >&2
                fi
            done
        ) &
        wpids+=($!)
    done
    for p in "${wpids[@]}"; do wait "$p"; done

    # tally from the per-shard progress files (authoritative, survives restarts)
    done_n=0; fail_n=0; failed_list=()
    for s in "${SHARD_LIST[@]}"; do
        if shard_complete "$s"; then done_n=$((done_n + 1)); else fail_n=$((fail_n + 1)); failed_list+=("$s"); fi
    done
    npz_n=$(find "$(dirname "$DATA_ROOT_BASE")" -name '*.npz' -path '*/tracks/*' 2>/dev/null | wc -l || echo 0)
    overall_set "$done_n" 0 "$fail_n" "$N_SHARDS" "$npz_n" "$(( $(date +%s) - T0 ))" "${failed_list[*]:-}"
    echo "=============================================================="
    echo "[shards] done in $(( ($(date +%s) - T0) / 60 ))m: $done_n complete, $fail_n incomplete"
    echo "[shards] per-shard console logs: ${DATA_ROOT_BASE}<N>.log"
    (( fail_n > 0 )) && { echo "[shards] incomplete: ${failed_list[*]}"; echo "[shards] re-run to retry"; exit 1; }
    exit 0
fi
T0=$(date +%s)
n_done=0 n_skip=0 n_fail=0 clips_total=0
FAILED=()

for s in "${SHARD_LIST[@]}"; do
    root="$(shard_root "$s")"
    if shard_complete "$s"; then
        n_skip=$((n_skip + 1))
        echo "[shards] shard $s: already complete, skipping"
        continue
    fi

    echo "=============================================================="
    echo "[shards] shard $s  ($((n_done + n_skip + n_fail + 1))/$N_SHARDS)  elapsed $(( ($(date +%s) - T0) / 60 ))m"
    t=$(date +%s)
    rc=0
    SHARD="$s" RESUME=1 DATA_ROOT="$root" bash "$SHARD_SCRIPT" || rc=$?
    # count npz regardless of outcome: a shard can produce tracks and still fail a later phase
    c=$(ls "$root"/tracks/*.npz 2>/dev/null | wc -l || echo 0)
    clips_total=$((clips_total + c))
    if [[ $rc -eq 0 ]]; then
        n_done=$((n_done + 1))
        echo "[shards] shard $s OK in $(( ($(date +%s) - t) / 60 ))m  ($c npz)"
        if [[ "$CLEANUP" == "1" ]] && shard_complete "$s"; then
            # only after progress.json confirms completion -- never delete inputs for a
            # shard that would need re-processing
            rm -rf "$root/download" "$root/raw_videos"
            echo "[shards] shard $s: removed download/ and raw_videos/ (tracks kept)"
        fi
    else
        n_fail=$((n_fail + 1)); FAILED+=("$s")
        echo "[shards] shard $s FAILED -- see $root/logs/" >&2
        [[ "$STOP_ON_FAIL" == "1" ]] && { echo "[shards] stopping (STOP_ON_FAIL=1)" >&2; break; }
    fi

    overall_set "$n_done" "$n_skip" "$n_fail" "$N_SHARDS" "$clips_total" \
                "$(( $(date +%s) - T0 ))" "${FAILED[*]:-}"

    # rolling ETA from shards actually processed this session
    if (( n_done > 0 )); then
        avg=$(( ($(date +%s) - T0) / n_done ))
        left=$(( N_SHARDS - n_done - n_skip - n_fail ))
        echo "[shards] avg $(( avg / 60 ))m/shard, $left left, ETA $(( avg * left / 3600 ))h"
    fi
done

overall_set "$n_done" "$n_skip" "$n_fail" "$N_SHARDS" "$clips_total" \
            "$(( $(date +%s) - T0 ))" "${FAILED[*]:-}"
echo "=============================================================="
echo "[shards] done in $(( ($(date +%s) - T0) / 60 ))m: $n_done processed, $n_skip skipped, $n_fail failed"
echo "[shards] npz produced this session: $clips_total"
if (( n_fail > 0 )); then
    echo "[shards] failed shards: ${FAILED[*]}"
    echo "[shards] re-run the same command to retry them (completed shards are skipped)"
    exit 1
fi
