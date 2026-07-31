#!/bin/bash
# Publish the processed (parquet) dataset to a HuggingFace dataset repo as a directory tree,
# mirroring noctuashap/openvid-wantrack-processed's layout (raw parquet files, not tarred).
# Uses `hf upload-large-folder` -- resumable and built for multi-TB uploads: re-running skips
# files already on the Hub, so a killed upload just continues.
#
# Usage:
#   REPO=FastVideo/openvid-wantrack-processed-v2 bash data_pipeline/upload_parquets.sh
#   ... DRY_RUN=1 ...      # verify completeness + print the command, upload nothing
set -uo pipefail

REPO=${REPO:?set REPO=<owner>/<name>}
PARQUET_ROOT=${PARQUET_ROOT:-/home/hal-shared/motionstream/data/openvid-wantrack-parquets}
SRC_ROOT=${SRC_ROOT:-/home/hal-shared/motionstream/data/openvid-wantrack}
PRIVATE=${PRIVATE:-1}
NUM_WORKERS=${NUM_WORKERS:-8}
README=${README:-data_pipeline/notes/processed_dataset_README.md}
DRY_RUN=${DRY_RUN:-0}

cd "$(dirname "$0")/.."
HF=$(command -v hf || command -v huggingface-cli) || { echo "[pub] ERROR: hf CLI not found" >&2; exit 1; }

# --- safety: verify the shards PRESENT under PARQUET_ROOT are duplicate-free before publishing.
# Validates whatever is present (no hardcoded 260, no source-video cross-check), so partial /
# derivative sets like the bf16 copy work; still catches real corruption via duplicate ids.
# Set VERIFY=0 to skip verification entirely.
if [[ "${VERIFY:-1}" == "1" ]]; then
echo "[pub] verifying present shards are duplicate-free before upload ..."
python - "$PARQUET_ROOT" <<'PY'
import glob, os, sys
import pyarrow.parquet as pq
pbase = sys.argv[1]
shards = sorted(d for d in os.listdir(pbase) if d.startswith("shard"))
bad=[]; total=0; nsh=0
for s in shards:
    fs=glob.glob(f"{pbase}/{s}/**/*.parquet", recursive=True)
    if not fs:
        continue
    ids=[i for f in fs for i in pq.read_table(f, columns=["id"]).column("id").to_pylist()]
    if len(ids)>0 and len(ids)==len(set(ids)):
        total+=len(ids); nsh+=1
    else:
        bad.append((s, len(ids), len(set(ids))))
if bad:
    print(f"[pub] REFUSING: {len(bad)} shard(s) empty/duplicate-id: {bad[:8]}")
    sys.exit(1)
print(f"[pub] OK: {nsh} shard(s) present, {total:,} clips (no duplicate ids)")
PY
[[ $? -eq 0 ]] || { echo "[pub] aborted -- fix the shards above, then re-run" >&2; exit 1; }
else
  echo "[pub] VERIFY=0 -> skipping shard verification"
fi

n_files=$(find "$PARQUET_ROOT" -name '*.parquet' | wc -l)
size=$(du -sh "$PARQUET_ROOT" 2>/dev/null | cut -f1)
echo "[pub] repo=$REPO  files=$n_files  size=$size  private=$PRIVATE"

if [[ "$DRY_RUN" == "1" ]]; then
    echo "[pub] DRY RUN -- would run:"
    echo "  $HF upload-large-folder $REPO $PARQUET_ROOT --repo-type dataset --include '*.parquet' --num-workers $NUM_WORKERS"
    echo "  (+ README.md upload)"
    exit 0
fi

# --- create repo + upload README once -------------------------------------------------
vis=(); [[ "$PRIVATE" == "1" ]] && vis=(--private)
"$HF" repo create "$REPO" --repo-type dataset "${vis[@]}" 2>/dev/null \
    && echo "[pub] created $REPO" || echo "[pub] repo exists (ok)"
[[ -f "$README" ]] && "$HF" upload "$REPO" "$README" README.md --repo-type dataset >/dev/null 2>&1 \
    && echo "[pub] README uploaded"

# --- upload the parquet tree (resumable) ----------------------------------------------
# --include '*.parquet' skips any stray files; the shard*/combined_parquet_dataset/... tree
# is preserved in the repo. Re-run this exact command to resume after any interruption.
echo "[pub] uploading parquet tree (resumable; re-run to continue if interrupted) ..."
"$HF" upload-large-folder "$REPO" "$PARQUET_ROOT" \
    --repo-type dataset --include '*.parquet' --num-workers "$NUM_WORKERS"

echo "[pub] done -> https://huggingface.co/datasets/$REPO"
