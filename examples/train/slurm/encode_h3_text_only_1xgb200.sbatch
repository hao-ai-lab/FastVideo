#!/bin/bash
# Encode the VidProM-H3 prompt list into text-only conditioning parquet on one
# GB200 tray: 4 GPUs = 4 independent shard processes of
# preprocess_minimax_h3_text_only.py (each loads the full Qwen3-VL encoder).
#
#   sbatch examples/train/slurm/encode_h3_text_only_1xgb200.sbatch
#
# Shards are resume-safe: resubmitting after a timeout/preemption continues
# after the rows already on disk. Paths assume the nvl-gb200 cluster layout
# (login /home is NOT on compute nodes — everything lives on /mnt/lustre);
# override via the environment, e.g. PROMPTS=... MODEL=... sbatch ...
#
# On this autoscaled cluster a PENDing job is what creates node demand — leave
# the job queued and the slurm-operator scales a tray up.

#SBATCH --job-name=h3-textenc
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --exclusive
#SBATCH --partition=all
#SBATCH --time=08:00:00
#SBATCH --output=/mnt/lustre/vlm-wlsaidhi/fastvideo/logs/h3-textenc-%j.out

set -uo pipefail

# Compute pods have no /home/<user>: triton's JIT cache (touched at fastvideo
# import), HF caches, and W&B netrc all resolve through $HOME — point it at
# lustre or every fastvideo import dies with PermissionError.
export HOME="${LUSTRE_HOME:-/mnt/lustre/vlm-wlsaidhi}"

REPO="${REPO:-/mnt/lustre/vlm-wlsaidhi/fastvideo/FastVideo}"
VENV="${VENV:-${REPO}/.venv}"
PROMPTS="${PROMPTS:-/mnt/lustre/vlm-wlsaidhi/fastvideo/data/vidprom_h3_v1_63136_20260811.txt}"
MODEL="${MODEL:-/mnt/lustre/vlm-k1kong/models/MiniMax-H3}"
OUT="${OUT:-/mnt/lustre/vlm-wlsaidhi/fastvideo/data/vidprom_h3_text_only}"
NUM_SHARDS="${NUM_SHARDS:-4}"
LIMIT="${LIMIT:-}"   # e.g. LIMIT=8 for a smoke run

mkdir -p "$OUT"

echo "node=$(hostname) repo=$REPO prompts=$PROMPTS model=$MODEL out=$OUT shards=$NUM_SHARDS limit=${LIMIT:-none}"
cd "$REPO"

pids=()
for k in $(seq 0 $((NUM_SHARDS - 1))); do
  CUDA_VISIBLE_DEVICES=$k MASTER_PORT=$((29531 + k)) \
    "$VENV/bin/python" fastvideo/pipelines/preprocess/preprocess_minimax_h3_text_only.py \
      --prompts-file "$PROMPTS" \
      --model-path "$MODEL" \
      --output-dir "$OUT" \
      --shard-index "$k" \
      --num-shards "$NUM_SHARDS" \
      ${LIMIT:+--limit "$LIMIT"} \
      > "$OUT/shard_${k}.log" 2>&1 &
  pids+=($!)
done

fail=0
for k in "${!pids[@]}"; do
  if ! wait "${pids[$k]}"; then
    echo "shard $k FAILED (see $OUT/shard_${k}.log)"
    fail=1
  fi
done

# The training dataloader caches the parquet file list under the dataset root;
# drop any stale cache so the next training start rescans fresh shards.
rm -rf "$OUT/map_style_cache"

"$VENV/bin/python" - <<EOF
import pyarrow.parquet as pq
from pathlib import Path
files = sorted(Path("$OUT").rglob("*.parquet"))
rows = sum(pq.ParquetFile(f).metadata.num_rows for f in files)
print(f"TOTAL: {rows} rows across {len(files)} parquet files under $OUT")
EOF

exit $fail
