#!/bin/bash
# Package each shard's tracks/ into a tar and upload to a HuggingFace dataset repo,
# mirroring noctuashap/openvid-wantrack-tracks layout: one tars-NNNNN.tar per shard,
# each holding ~1000 .npz (flat, no directory prefix).
#
# Only shards whose progress.json marks tracks done are packaged. Resumable: a shard
# already present in the repo (checked via the HF API) is skipped.
#
# Usage:
#   REPO=<user-or-org>/<name> SHARDS=0-170 bash data_pipeline/upload_tracks.sh
#   REPO=FastVideo/openvid-wantrack-tracks-v2 SHARDS=0-170 PRIVATE=1 bash data_pipeline/upload_tracks.sh
#   ... DRY_RUN=1 ...    # build tars + report, do NOT create repo or upload
set -uo pipefail

REPO=${REPO:?set REPO=<owner>/<name>}
SHARDS=${SHARDS:-0-259}
DATA_ROOT_BASE=${DATA_ROOT_BASE:-/home/hal-shared/motionstream/data/openvid-wantrack/shard}
STAGING=${STAGING:-/home/hal-shared/motionstream/data/openvid-wantrack/_upload_tars}
PRIVATE=${PRIVATE:-1}          # create the repo private by default; you flip it public in the UI
KEEP_TARS=${KEEP_TARS:-0}      # 1 = keep local tar after upload (default: delete to save disk)
DRY_RUN=${DRY_RUN:-0}
PREFIX=${PREFIX:-tracks}       # tar basename: ${PREFIX}-00042.tar
README=${README:-data_pipeline/notes/tracks_dataset_README.md}  # uploaded as README.md if present

cd "$(dirname "$0")/.."
shard_root() { printf "%s%03d" "$DATA_ROOT_BASE" "$1"; }
mkdir -p "$STAGING"

# expand "0-9,20,30-35"
expand() {
    local tok lo hi out=(); IFS=',' read -ra toks <<< "$1"
    for tok in "${toks[@]}"; do
        if [[ "$tok" =~ ^([0-9]+)-([0-9]+)$ ]]; then
            for ((i=${BASH_REMATCH[1]}; i<=${BASH_REMATCH[2]}; i++)); do out+=("$i"); done
        elif [[ "$tok" =~ ^[0-9]+$ ]]; then out+=("$tok")
        else echo "[upload] ERROR: bad SHARDS token '$tok'" >&2; exit 1; fi
    done
    printf '%s\n' "${out[@]}"
}
mapfile -t LIST < <(expand "$SHARDS")

tracks_done() {  # tracks_done <n>
    python - "$(shard_root "$1")/progress.json" <<'PY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1])
sys.exit(0 if p.exists() and json.loads(p.read_text()).get("phases",{}).get("tracks",{}).get("done") else 1)
PY
}

# --- ensure repo exists (unless dry run) -------------------------------------------
if [[ "$DRY_RUN" != "1" ]]; then
    vis=(); [[ "$PRIVATE" == "1" ]] && vis=(--private)
    hf repo create "$REPO" --repo-type dataset "${vis[@]}" 2>/dev/null \
        && echo "[upload] created dataset repo $REPO" \
        || echo "[upload] repo $REPO already exists (ok)"
    # names of files already in the repo (via the hub API), to skip re-upload on resume
    mapfile -t REMOTE < <(python - "$REPO" <<'PY'
import sys
from huggingface_hub import HfApi
try:
    print("\n".join(HfApi().list_repo_files(sys.argv[1], repo_type="dataset")))
except Exception:
    pass
PY
)
    remote_has() { printf '%s\n' "${REMOTE[@]:-}" | grep -qx "$1"; }

    # upload README once (if present and not already there)
    if [[ -f "$README" ]] && ! remote_has "README.md"; then
        hf upload "$REPO" "$README" "README.md" --repo-type dataset >/dev/null 2>&1 \
            && echo "[upload] README.md uploaded" || echo "[upload] WARN: README upload failed" >&2
    fi
else
    remote_has() { return 1; }
fi

n_up=0 n_skip=0 n_todo=0
T0=$(date +%s)
for s in "${LIST[@]}"; do
    root="$(shard_root "$s")"
    tar_name=$(printf "%s-%05d.tar" "$PREFIX" "$s")
    if ! tracks_done "$s"; then continue; fi
    n_todo=$((n_todo+1))
    if remote_has "$tar_name"; then n_skip=$((n_skip+1)); echo "[upload] $tar_name already in repo, skip"; continue; fi

    ntracks=$(ls "$root"/tracks/*.npz 2>/dev/null | wc -l || echo 0)
    [[ "$ntracks" -gt 0 ]] || { echo "[upload] WARN shard $s: no npz despite progress=done, skip" >&2; continue; }

    tar_path="$STAGING/$tar_name"
    # Clip names start with '---', so a glob/ls would feed tar filenames it reads as options.
    # find -print0 | tar --null -T - is dash-safe; paths come out './name' (matches the
    # reference repo's layout). Exclude any leftover *.tmp.npz from an interrupted write.
    echo "[upload] packing shard $s: $ntracks npz -> $tar_name"
    ( cd "$root/tracks" && find . -maxdepth 1 -type f -name '*.npz' ! -name '*.tmp.npz' -print0 ) \
        | tar -cf "$tar_path" --null -C "$root/tracks" -T -
    sz=$(du -h "$tar_path" 2>/dev/null | cut -f1)

    if [[ "$DRY_RUN" == "1" ]]; then
        echo "[upload] DRY_RUN: built $tar_name ($sz), not uploading"
        [[ "$KEEP_TARS" == "1" ]] || rm -f "$tar_path"
        continue
    fi

    if hf upload "$REPO" "$tar_path" "$tar_name" --repo-type dataset >/dev/null 2>&1; then
        n_up=$((n_up+1)); echo "[upload] shard $s -> $tar_name ($sz) uploaded"
        [[ "$KEEP_TARS" == "1" ]] || rm -f "$tar_path"
    else
        echo "[upload] ERROR uploading $tar_name (kept at $tar_path)" >&2
    fi
done

echo "[upload] done in $(( ($(date +%s)-T0)/60 ))m: $n_up uploaded, $n_skip already present, $n_todo eligible"
[[ "$DRY_RUN" == "1" ]] && echo "[upload] (dry run: repo not created, nothing uploaded)"
echo "[upload] repo: https://huggingface.co/datasets/$REPO"
