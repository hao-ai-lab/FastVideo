#!/usr/bin/env bash
# Run a Wan2.1-14B WanTrack stage inside an ALREADY-HELD Slurm allocation.
#
# Same held-alloc + auto-restart pattern as run_openvid_bidir_held.sh, but every WANTRACK_*
# knob is parameterized instead of hardcoded, because the four stages of the 14B teacher
# pipeline need DIFFERENT sampling/masking settings:
#
#   A (fixed overfit)  : WANTRACK_FIXED_SAMPLE=1 FREEZE_HEAD=0
#   B (random overfit) : WANTRACK_FIXED_SAMPLE=0 FREEZE_HEAD=0
#   D (stage-1 openvid): FREEZE_HEAD=1, no dropping
#   E (stage-2 synth)  : FREEZE_HEAD=1 TRACK_DROP=0.5 MOTION_DROP=0.3 PMASK=0.2 MASK_CHUNK=8
#
# FREEZE_HEAD is the important one: the overfit stages MUST train track_encoder (that is the
# whole point — co-adapting it with the patch_embedding track slot), while stages D/E freeze it.
#
# Usage:
#   ALLOC=728 NODES=4 JOB=wan14b_stepA PORT=30910 \
#   CFG=examples/train/scenario/worldmodel/finetune_wantrack_synth_sparse_fixed_14b.yaml \
#   WANTRACK_FIXED_SAMPLE=1 WANTRACK_FREEZE_HEAD=0 \
#   WANDB_API_KEY=... bash examples/train/run_wan14b_held.sh
set -uo pipefail
WORK=/mnt/lustre/vlm-s4duan
REPO=$WORK/FastVideo
CFG="${CFG:?set CFG}"
NODES="${NODES:-4}"; GPUS=4
JOB="${JOB:-wan14b}"
PORT="${PORT:-30900}"
ALLOC="${ALLOC:?set ALLOC to a running held allocation jobid}"
WANDB_RUN_ID="${WANDB_RUN_ID:-${JOB}}"
: "${WANDB_API_KEY:?export WANDB_API_KEY}"
TOTAL_GPUS=$(( NODES * GPUS ))
mkdir -p "$WORK/logs"

OUTPUT_DIR=$(grep -oE 'output_dir:[^#]*' "$REPO/$CFG" | head -1 | sed 's/.*output_dir:[[:space:]]*//; s/"//g' | xargs)

# Per-stage WANTRACK defaults (overridable from the environment).
W_AUG="${WANTRACK_AUG:-1}"
W_SPARSE="${WANTRACK_SPARSE:-1}"
W_EXTRA_RANDOM="${WANTRACK_EXTRA_RANDOM:-20}"
W_EXTRA_MODE="${WANTRACK_EXTRA_MODE:-random}"
W_PMASK="${WANTRACK_PMASK:-0}"
W_MASK_CHUNK="${WANTRACK_MASK_CHUNK:-0}"
W_TRACK_DROP="${WANTRACK_TRACK_DROP:-0}"
W_MOTION_DROP="${WANTRACK_MOTION_DROP:-0}"
W_TEXT_DROP="${WANTRACK_TEXT_DROP:-0}"
W_FIXED="${WANTRACK_FIXED_SAMPLE:-0}"
W_FREEZE="${WANTRACK_FREEZE_HEAD:-0}"
W_BIAS="${TRACKWAN_TRACK_BIAS:-1}"

# A crash mid-save leaves checkpoint-N/dcp without .metadata; drop it so resume falls back to
# the last COMPLETE checkpoint instead of crash-looping on "metadata is None".
clean_bad_ckpt() {
  [ -n "$OUTPUT_DIR" ] || return 0
  local latest
  latest=$(ls -d "$OUTPUT_DIR"/checkpoint-* 2>/dev/null | sed 's/.*checkpoint-//' | sort -n | tail -1)
  [ -n "$latest" ] || return 0
  local d="$OUTPUT_DIR/checkpoint-$latest"
  if [ ! -f "$d/dcp/.metadata" ]; then
    echo "[held] checkpoint-$latest incomplete (no dcp/.metadata) — removing"
    rm -rf "$d"
  fi
}

[ "$(squeue -h -j "$ALLOC" -o '%t' 2>/dev/null)" = R ] || { echo "[held] alloc $ALLOC not running"; exit 1; }
NODELIST=$(squeue -h -j "$ALLOC" -o '%N')
# Pin the exact subset of the held allocation this stage runs on. Without --nodelist, srun is
# free to pick ANY $NODES of the allocation's nodes, so the rdzv endpoint (computed here) can
# land on a node that isn't in the subset — rank 0 then never listens there and every worker
# dies with "client socket has timed out while trying to connect". Choosing the subset up front
# makes MASTER provably the node hosting rank 0, and lets stages share one allocation safely.
# By default take the first $NODES nodes of the allocation. Pass NODELIST_OVERRIDE=a,b,c to pin
# an explicit subset — required when several stages share one allocation, otherwise every run
# grabs the same leading nodes and they fight over the same GPUs.
SUBSET="${NODELIST_OVERRIDE:-$(scontrol show hostnames "$NODELIST" | head -n "$NODES" | paste -sd,)}"
MASTER=$(echo "$SUBSET" | cut -d, -f1)
echo "[held] alloc=$ALLOC nodes=$NODELIST"
echo "[held] subset=$SUBSET master=$MASTER using $NODES node(s) / $TOTAL_GPUS GPU(s)"
echo "[held] cfg=$CFG out=$OUTPUT_DIR"
echo "[held] WANTRACK: fixed=$W_FIXED freeze=$W_FREEZE pmask=$W_PMASK chunk=$W_MASK_CHUNK track_drop=$W_TRACK_DROP motion_drop=$W_MOTION_DROP bias=$W_BIAS"

# Killing this script (or its srun) does NOT kill the torchrun/python ranks out on the compute
# nodes — they survive, keep ~107GB of GPU memory each, and silently keep writing checkpoints.
# That poisons the next run: it contends for GPUs and resurrects deleted output dirs. So sweep
# the subset on any exit. Matches on the config path so we only kill THIS stage's ranks.
# Liveness marker for anything chaining off this stage. Do NOT make chainers use
# `pgrep -f run_wan14b_held.sh`: that also matches any monitoring/shell command whose command
# line merely CONTAINS the string, so a watcher looking for this launcher keeps "seeing" it long
# after it exited (that stall cost ~1h of idle nodes between stages A and B).
RUNFILE="$WORK/logs/${JOB}.running"
echo "$$" > "$RUNFILE"

cleanup_ranks() {
  echo "[held] sweeping stray ranks for $CFG on $SUBSET ..."
  timeout 120 srun --overlap --jobid="$ALLOC" --nodelist="$SUBSET" --nodes="$NODES" \
    --ntasks="$NODES" --ntasks-per-node=1 \
    bash -c "pkill -9 -f 'entrypoint/train.py --config $CFG'; exit 0" >/dev/null 2>&1 || true
}
trap 'echo "[held] interrupted — cleaning up"; cleanup_ranks; rm -f "$RUNFILE"; exit 130' INT TERM
trap 'rm -f "$RUNFILE"' EXIT

attempt=0
while :; do
  [ -n "$(squeue -h -j "$ALLOC" -o '%t' 2>/dev/null)" ] || { echo "[held] alloc $ALLOC vanished"; exit 1; }
  attempt=$((attempt + 1))
  clean_bad_ckpt
  echo "=== [train] attempt $attempt on alloc $ALLOC ($(date)) ==="
  srun --overlap --jobid="$ALLOC" --nodelist="$SUBSET" --nodes="$NODES" --ntasks="$NODES" \
    --ntasks-per-node=1 --chdir="$REPO" bash -lc "
      source .venv/bin/activate
      export HOME=$WORK HF_HOME=$WORK/.hf TORCH_HOME=$WORK/.torch MPLCONFIGDIR=$WORK/.mpl \
        TRITON_CACHE_DIR=$WORK/.cache/triton_${JOB} TORCHINDUCTOR_CACHE_DIR=$WORK/.cache/inductor_${JOB} \
        TOKENIZERS_PARALLELISM=false NCCL_CUMEM_ENABLE=0 PYTHONPATH=$REPO \
        WANDB_API_KEY=$WANDB_API_KEY WANDB_MODE=online \
        WANDB_RUN_ID=$WANDB_RUN_ID WANDB_RESUME=allow \
        WANTRACK_AUG=$W_AUG WANTRACK_SPARSE=$W_SPARSE \
        WANTRACK_EXTRA_RANDOM=$W_EXTRA_RANDOM WANTRACK_EXTRA_MODE=$W_EXTRA_MODE \
        WANTRACK_PMASK=$W_PMASK WANTRACK_MASK_CHUNK=$W_MASK_CHUNK \
        WANTRACK_TRACK_DROP=$W_TRACK_DROP WANTRACK_MOTION_DROP=$W_MOTION_DROP \
        WANTRACK_TEXT_DROP=$W_TEXT_DROP WANTRACK_FIXED_SAMPLE=$W_FIXED \
        WANTRACK_FREEZE_HEAD=$W_FREEZE TRACKWAN_TRACK_BIAS=$W_BIAS \
        WANTRACK_DEBUG=1 \
        TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800 TORCH_NCCL_TRACE_BUFFER_SIZE=2000 \
        NCCL_SOCKET_NTHREADS=4 NCCL_NSOCKS_PERTHREAD=8
      torchrun --nnodes=$NODES --nproc-per-node=$GPUS --node-rank=\$SLURM_PROCID \
        --rdzv-backend=c10d --rdzv-endpoint=$MASTER:$PORT \
        fastvideo/train/entrypoint/train.py --config $CFG \
        --training.distributed.num_gpus $TOTAL_GPUS
    "
  rc=$?
  if [ $rc -eq 0 ]; then echo "[train] finished cleanly (rc=0)"; break; fi
  echo "[train] exited rc=$rc — nodes still held, relaunching in 20s ..."
  # A crashed rank can leave its siblings alive and holding GPU memory; clear them before retry
  # or the relaunch will OOM or hang waiting on a rendezvous the stale ranks are still in.
  cleanup_ranks
  sleep 20
done
echo "[held] stage complete; allocation $ALLOC left RUNNING for the next stage."
