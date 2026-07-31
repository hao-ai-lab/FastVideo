#!/usr/bin/env bash
# One-shot smoke test of the 720p stage-1 geometry inside held alloc 881.
# Unlike run_wan14b_held.sh this does NOT auto-restart on failure — a crash (e.g. OOM) is the
# RESULT we are looking for, not something to retry through.
set -uo pipefail
WORK=/mnt/lustre/vlm-s4duan
REPO=$WORK/FastVideo
ALLOC="${ALLOC:-881}"
NODES="${NODES:-16}"; GPUS=4
PORT="${PORT:-31777}"
CFG="${CFG:-examples/train/scenario/worldmodel/smoke_720p_14b_stage1.yaml}"
JOB="${JOB:-smoke720}"
TOTAL_GPUS=$(( NODES * GPUS ))
LOG=$WORK/logs/${JOB}.log

[ "$(squeue -h -j "$ALLOC" -o '%t' 2>/dev/null)" = R ] || { echo "alloc $ALLOC not running"; exit 1; }
NODELIST=$(squeue -h -j "$ALLOC" -o '%N')
# Pin the subset so the rdzv master provably hosts rank 0 (see run_wan14b_held.sh).
SUBSET=$(scontrol show hostnames "$NODELIST" | head -n "$NODES" | paste -sd,)
MASTER=$(echo "$SUBSET" | cut -d, -f1)
echo "[smoke] alloc=$ALLOC nodes=$NODES gpus=$TOTAL_GPUS master=$MASTER"
echo "[smoke] cfg=$CFG log=$LOG"

# Stage-1 env. WANTRACK_FREEZE_HEAD=0 — the track_encoder is TRAINABLE in stage-1.
# The 14B stage-1 config header claimed FREEZE_HEAD=1, but that contradicts the 1.3B run it
# says it copies exactly: run_openvid_bidir_held.sh never sets the var (code default "0") and
# the 1.3B logs have zero "FROZE track_encoder" lines. Every real freeze in this repo is
# stage-2/3 (run_openvid_stage2*/stage3_slurm.sh, the latter commented "head is converged" —
# converged BY stage-1 training it). wantrack.py itself calls it a "stage-2 knob ... after
# initial training", and notes the head only plateaus by step ~4700 of a 4800-step stage-1.
srun --overlap --jobid="$ALLOC" --nodelist="$SUBSET" --nodes="$NODES" --ntasks="$NODES" \
  --ntasks-per-node=1 --chdir="$REPO" bash -lc "
    source .venv/bin/activate
    export HOME=$WORK HF_HOME=$WORK/.hf TORCH_HOME=$WORK/.torch MPLCONFIGDIR=$WORK/.mpl \
      TRITON_CACHE_DIR=$WORK/.cache/triton_${JOB} TORCHINDUCTOR_CACHE_DIR=$WORK/.cache/inductor_${JOB} \
      TOKENIZERS_PARALLELISM=false NCCL_CUMEM_ENABLE=0 PYTHONPATH=$REPO \
      WANDB_MODE=disabled \
      WANTRACK_AUG=1 WANTRACK_SPARSE=1 WANTRACK_EXTRA_RANDOM=20 WANTRACK_EXTRA_MODE=random \
      WANTRACK_PMASK=0 WANTRACK_MASK_CHUNK=0 WANTRACK_TRACK_DROP=0 WANTRACK_MOTION_DROP=0 \
      WANTRACK_TEXT_DROP=0 WANTRACK_FIXED_SAMPLE=0 WANTRACK_FREEZE_HEAD=0 TRACKWAN_TRACK_BIAS=1 \
      WANTRACK_DEBUG=1 \
      TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800 NCCL_SOCKET_NTHREADS=4 NCCL_NSOCKS_PERTHREAD=8
    torchrun --nnodes=$NODES --nproc-per-node=$GPUS --node-rank=\$SLURM_PROCID \
      --rdzv-backend=c10d --rdzv-endpoint=$MASTER:$PORT \
      fastvideo/train/entrypoint/train.py --config $CFG \
      --training.distributed.num_gpus $TOTAL_GPUS
  " 2>&1 | tee "$LOG"
rc=${PIPESTATUS[0]}
echo "[smoke] srun exit rc=$rc"

# Always sweep stray ranks: a dead rank keeps ~100GB of GPU memory and would poison the real run.
echo "[smoke] sweeping stray ranks ..."
timeout 120 srun --overlap --jobid="$ALLOC" --nodelist="$SUBSET" --nodes="$NODES" \
  --ntasks="$NODES" --ntasks-per-node=1 \
  bash -c "pkill -9 -f 'entrypoint/train.py --config $CFG'; exit 0" >/dev/null 2>&1 || true
echo "[smoke] done (rc=$rc)"
exit $rc
