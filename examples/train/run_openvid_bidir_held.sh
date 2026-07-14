#!/usr/bin/env bash
# Launch a bidir teacher training run on a HELD node allocation, so it survives this
# cluster's ~80-min node-cordon recycling (a held allocation keeps its nodes because a
# drain can't complete while a job holds them — same reason the shao_wm2 sleep-infinity
# node lived a day). We hold N nodes with sleep infinity, then run torchrun training
# INSIDE the allocation via `srun --overlap`, auto-restarting on any exit (resume from
# the latest checkpoint). Verified: one srun --overlap fans out across all held nodes.
#
# Usage:
#   WANDB_API_KEY=<key> CFG=<config.yaml> NODES=<n> JOB=<name> \
#     bash examples/train/run_openvid_bidir_held.sh
set -uo pipefail
WORK=/mnt/lustre/vlm-s4duan
REPO=$WORK/FastVideo
CFG="${CFG:-examples/train/scenario/worldmodel/finetune_wantrack_openvid_sparse_1p3b.yaml}"
NODES="${NODES:-4}"; GPUS=4
JOB="${JOB:-openvid_bidir_1p3b}"
PORT="${PORT:-29500}"
# Stable, per-model W&B run id so crash-relaunches RESUME a single run instead of minting a
# fresh run every attempt (that's what fragmented the project into dozens of runs). wandb.init
# honors WANDB_RUN_ID + WANDB_RESUME from the env, so no code change is needed. Distinct per
# JOB (1p3b vs 14b); override with WANDB_RUN_ID=... to point restarts at an existing run.
WANDB_RUN_ID="${WANDB_RUN_ID:-${JOB}}"
: "${WANDB_API_KEY:?export WANDB_API_KEY before launching}"
TOTAL_GPUS=$(( NODES * GPUS ))
mkdir -p "$WORK/logs"
# output_dir (for checkpoint cleanup) — read from the config
OUTPUT_DIR=$(grep -oE 'output_dir:[^#]*' "$REPO/$CFG" | head -1 | sed 's/.*output_dir:[[:space:]]*//; s/"//g' | xargs)

# Drop a half-written LATEST checkpoint (a crash mid-save leaves checkpoint-N/dcp with no
# .metadata) so resume_from_checkpoint=latest falls back to the previous VALID checkpoint (or
# scratch) instead of crash-looping on "metadata is None". Keeps only complete checkpoints.
clean_bad_ckpt() {
  [ -n "$OUTPUT_DIR" ] || return 0
  local latest
  latest=$(ls -d "$OUTPUT_DIR"/checkpoint-* 2>/dev/null | sed 's/.*checkpoint-//' | sort -n | tail -1)
  [ -n "$latest" ] || return 0
  local d="$OUTPUT_DIR/checkpoint-$latest"
  if [ ! -f "$d/dcp/.metadata" ]; then
    echo "[held] checkpoint-$latest is incomplete (no dcp/.metadata) — removing so resume uses the last good one"
    rm -rf "$d"
  fi
}

# --- 1) hold the nodes (survives cordons; sleep infinity never releases them) --------
# Reuse an already-held allocation by passing ALLOC=<jobid> in the env — lets us swap
# env vars / config without losing the nodes to the cold pool.
if [ -z "${ALLOC:-}" ]; then
  echo "[held] requesting $NODES nodes ($TOTAL_GPUS GPUs) ..."
  ALLOC=$(sbatch -N"$NODES" --gres=gpu:$GPUS --ntasks-per-node=1 --exclusive -t 120:00:00 \
    -p all -J "${JOB}_hold" --chdir="$WORK" -o "$WORK/logs/${JOB}_hold_%j.out" \
    --wrap='srun sleep infinity' | grep -oE '[0-9]+' | head -1)
  [ -z "$ALLOC" ] && { echo "[held] sbatch failed (cold pool? submit a warmup + retry)"; exit 1; }
else
  echo "[held] reusing existing allocation JobID=$ALLOC"
fi
echo "[held] allocation JobID=$ALLOC ; waiting for it to start ..."
for i in $(seq 1 60); do
  [ "$(squeue -h -j "$ALLOC" -o '%t' 2>/dev/null)" = R ] && break; sleep 5
done
[ "$(squeue -h -j "$ALLOC" -o '%t' 2>/dev/null)" = R ] || { echo "[held] $ALLOC not running"; exit 1; }
NODELIST=$(squeue -h -j "$ALLOC" -o '%N')
MASTER=$(scontrol show hostnames "$NODELIST" | head -1)
echo "[held] nodes=$NODELIST  master=$MASTER  (cancel with: scancel $ALLOC)"

# --- 2) run training inside the held allocation, auto-restart on exit -----------------
attempt=0
while :; do
  # bail out if the held allocation itself is gone (should be rare)
  [ -n "$(squeue -h -j "$ALLOC" -o '%t' 2>/dev/null)" ] || { echo "[held] allocation $ALLOC vanished — re-run this script"; exit 1; }
  attempt=$((attempt + 1))
  clean_bad_ckpt   # remove any half-written checkpoint before (re)starting
  echo "=== [train] attempt $attempt on alloc $ALLOC (resume from latest valid checkpoint) ==="
  srun --overlap --jobid="$ALLOC" --nodes="$NODES" --ntasks="$NODES" --ntasks-per-node=1 \
    --chdir="$REPO" bash -lc "
      source .venv/bin/activate
      export HOME=$WORK HF_HOME=$WORK/.hf TORCH_HOME=$WORK/.torch MPLCONFIGDIR=$WORK/.mpl \
        TRITON_CACHE_DIR=$WORK/.cache/triton_${JOB} TORCHINDUCTOR_CACHE_DIR=$WORK/.cache/inductor_${JOB} \
        TOKENIZERS_PARALLELISM=false NCCL_CUMEM_ENABLE=0 PYTHONPATH=$REPO \
        WANDB_API_KEY=$WANDB_API_KEY WANDB_MODE=online \
        WANDB_RUN_ID=$WANDB_RUN_ID WANDB_RESUME=allow \
        WANTRACK_AUG=1 WANTRACK_SPARSE=1 WANTRACK_EXTRA_RANDOM=20 WANTRACK_EXTRA_MODE=random \
        WANTRACK_PMASK=0 WANTRACK_FIXED_SAMPLE=0 WANTRACK_MOTION_DROP=0 WANTRACK_TEXT_DROP=0 \
        WANTRACK_DEBUG=1 TRACKWAN_TRACK_BIAS=${TRACKWAN_TRACK_BIAS:-0} \
        TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800 TORCH_NCCL_TRACE_BUFFER_SIZE=2000 \
        NCCL_SOCKET_NTHREADS=4 NCCL_NSOCKS_PERTHREAD=8
      torchrun --nnodes=$NODES --nproc-per-node=$GPUS --node-rank=\$SLURM_PROCID \
        --rdzv-backend=c10d --rdzv-endpoint=$MASTER:$PORT \
        fastvideo/train/entrypoint/train.py --config $CFG \
        --training.distributed.num_gpus $TOTAL_GPUS
    "
  rc=$?
  if [ $rc -eq 0 ]; then echo "[train] finished cleanly (rc=0)"; break; fi
  echo "[train] exited rc=$rc — nodes still held, relaunching from checkpoint in 20s ..."
  sleep 20
done
echo "[held] training complete. Freeing allocation $ALLOC."
scancel "$ALLOC" 2>/dev/null || true
