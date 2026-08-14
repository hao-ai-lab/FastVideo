#!/bin/bash
# 8-tray (8 x 4x GB200 = 32 GPU) MiniMax-H3 DMD2 launch from a Slurm login node.
#
#   sbatch examples/train/slurm/dmd2_32xgb200.sbatch
#   sbatch --nodes=2 examples/train/slurm/dmd2_32xgb200.sbatch          # smaller sweep
#   CONFIG=path/to/other.yaml sbatch examples/train/slurm/dmd2_32xgb200.sbatch
#
# The paired config (dmd2_simulate_fa4_32gpu.yaml) assumes 8 nodes:
# hsdp_replicate_dim must equal the node count and num_gpus the world size —
# this script forwards overrides derived from the actual allocation, so
# shrinking --nodes "just works" (global batch shrinks with it).
#
# Cluster-specific knobs (partition/account/container) are marked CLUSTER.

#SBATCH --job-name=h3-dmd2-32xgb200
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --time=48:00:00
#SBATCH --output=slurm-%x-%j.out
##SBATCH --partition=<CLUSTER: partition>
##SBATCH --account=<CLUSTER: account>

set -euo pipefail

# ── paths (shared FS visible on every tray) ──────────────────────────────
REPO="${REPO:-/mnt/FastVideo}"
VENV="${VENV:-${REPO}/.venv}"
CONFIG="${CONFIG:-${REPO}/examples/train/configs/distribution_matching/minimax_h3/dmd2_simulate_fa4_32gpu.yaml}"

# ── topology derived from the allocation ─────────────────────────────────
NNODES="${SLURM_JOB_NUM_NODES}"
GPUS_PER_NODE="${SLURM_GPUS_PER_NODE:-4}"
GPUS_PER_NODE="${GPUS_PER_NODE##*:}"   # "4" or "gb200:4" -> 4
WORLD_SIZE=$((NNODES * GPUS_PER_NODE))
# Topology overrides (default: sp across a tray, HSDP replicate=nodes/shard=4).
# e.g. SP_SIZE=1 HSDP_REPLICATE=1 HSDP_SHARD=$WORLD_SIZE for pure full-shard FSDP.
SP_SIZE="${SP_SIZE:-${GPUS_PER_NODE}}"
HSDP_REPLICATE="${HSDP_REPLICATE:-${NNODES}}"
HSDP_SHARD="${HSDP_SHARD:-${GPUS_PER_NODE}}"
MASTER_ADDR="$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)"
MASTER_PORT="${MASTER_PORT:-29501}"

echo "=== H3 DMD2 multi-node launch ==="
echo "Nodes:        ${NNODES} (${SLURM_JOB_NODELIST})"
echo "World size:   ${WORLD_SIZE}"
echo "Topology:     sp=${SP_SIZE} hsdp=(replicate=${HSDP_REPLICATE}, shard=${HSDP_SHARD})"
echo "Rendezvous:   ${MASTER_ADDR}:${MASTER_PORT}"
echo "Config:       ${CONFIG}"

# ── per-rank environment (srun propagates exported vars) ─────────────────
# CLUSTER: if compute nodes lack the login /home (k8s worker pods), submit
# with LUSTRE_HOME=<shared writable dir> — triton's JIT cache (touched at
# fastvideo import), HF caches, and ~/.netrc all resolve through $HOME.
export HOME="${LUSTRE_HOME:-${HOME}}"
export FASTVIDEO_FA4=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
# Node-local weight cache: relaunches rebuild all three 33B roles from tmpfs
# in seconds. PER_NODE=1 makes every tray write its own /dev/shm copy
# (single-writer semantics would leave 7 of 8 trays permanently cold).
export FASTVIDEO_WEIGHT_SHARD_CACHE=/dev/shm/fastvideo-wcache
export FASTVIDEO_WEIGHT_SHARD_CACHE_PER_NODE=1
# W&B: credentials come from ~/.netrc on the shared home; flip to offline if
# compute nodes have no egress.
export WANDB_MODE="${WANDB_MODE:-online}"
# CLUSTER: NCCL fabric tuning. These three come from the cluster onboarding
# guide's multi-node NCCL-test recipe (GB200 NVL72 + CX7 IB cross-rack):
# MNNVL is multi-node NVLink within a rack's NVL72 domain.
export NCCL_CUMEM_ENABLE="${NCCL_CUMEM_ENABLE:-1}"
export NCCL_MNNVL_ENABLE="${NCCL_MNNVL_ENABLE:-2}"
export UCX_NET_DEVICES="${UCX_NET_DEVICES:-eth0}"
# WARN costs nothing and turns silent transport hangs at communicator init
# into named errors in the log.
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

export NNODES MASTER_ADDR MASTER_PORT
export NUM_GPUS="${GPUS_PER_NODE}"

# Boot watchdog: NCCL communicator init can wedge silently (all ranks parked
# in ib_uverbs_event_read/poll, no WARN output, GPUs at 0%). If no weight
# loading has been logged within BOOT_TIMEOUT_MIN minutes, abort the job so
# the allocation isn't squatted on. 0 disables.
BOOT_TIMEOUT_MIN="${BOOT_TIMEOUT_MIN:-20}"
if [ "${BOOT_TIMEOUT_MIN}" -gt 0 ]; then
  (
    sleep $((BOOT_TIMEOUT_MIN * 60))
    if ! grep -qs "Loading weights took" "${REPO}/examples/train/logs/${SLURM_JOB_ID}-node0"/*.log; then
      echo "BOOT WATCHDOG: no weight loading after ${BOOT_TIMEOUT_MIN} min — aborting job ${SLURM_JOB_ID}"
      scancel "${SLURM_JOB_ID}"
    fi
  ) &
  WATCHDOG_PID=$!
fi

# CLUSTER: if trays require a container, wrap the srun payload accordingly,
# e.g. add `--container-image=... --container-mounts=/mnt:/mnt` (pyxis) so
# /mnt (repo, models, data) and /dev/shm are available inside.
srun --kill-on-bad-exit=1 --ntasks-per-node=1 bash -c "
  set -euo pipefail
  export NODE_RANK=\${SLURM_NODEID}
  # One log dir per node — parallel tee writers on one NFS file garble it.
  export LOG_DIR=examples/train/logs/\${SLURM_JOB_ID}-node\${SLURM_NODEID}
  source '${VENV}/bin/activate'
  cd '${REPO}'
  bash examples/train/run.sh '${CONFIG}' \
    --training.distributed.num_gpus ${WORLD_SIZE} \
    --training.distributed.hsdp_replicate_dim ${HSDP_REPLICATE} \
    --training.distributed.hsdp_shard_dim ${HSDP_SHARD} \
    --training.distributed.sp_size ${SP_SIZE}
"
SRUN_EXIT=$?
[ -n "${WATCHDOG_PID:-}" ] && kill "${WATCHDOG_PID}" 2>/dev/null
exit "${SRUN_EXIT}"
