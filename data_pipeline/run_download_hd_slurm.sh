#!/usr/bin/env bash
# Multi-node OpenVidHD download+extract (>=5s clips only), split-shard aware.
# Parts are sharded across tasks (SLURM_PROCID/NTASKS) so the ~8.8TB download runs
# in parallel across nodes. CPU/network only — no GPU. Idempotent (resumes).
#
# Usage:
#   NODES=8 TASKS_PER_NODE=3 bash data_pipeline/run_download_hd_slurm.sh
set -euo pipefail
WORK=/mnt/lustre/vlm-s4duan
NODES=${NODES:-8}; TASKS_PER_NODE=${TASKS_PER_NODE:-3}   # concurrent HF downloads per node
OUT=${OUT:-$WORK/openvid_1m}                             # goal folder
FILTER=${FILTER:-$WORK/openvid/OpenVidHD_filtered.txt}
mkdir -p "$OUT/videos" "$OUT/_zips" "$WORK/logs"
echo "download OpenVidHD -> $OUT/videos  ($NODES nodes x $TASKS_PER_NODE = $((NODES*TASKS_PER_NODE)) parallel downloads)"

sbatch -N "$NODES" --ntasks-per-node=$TASKS_PER_NODE --cpus-per-task=16 --mem=0 --exclusive \
  -t 48:00:00 -J openvid_dl --chdir="$WORK/FastVideo" \
  -o "$WORK/logs/openvid_dl_%j.out" -e "$WORK/logs/openvid_dl_%j.out" \
  --wrap "srun --chdir=$WORK/FastVideo bash -lc 'source .venv/bin/activate && export HF_HOME=$WORK/.hf && \
    python data_pipeline/openvid_download_hd.py \
      --videos-dir $OUT/videos --zip-dir $OUT/_zips --only-list $FILTER'"
