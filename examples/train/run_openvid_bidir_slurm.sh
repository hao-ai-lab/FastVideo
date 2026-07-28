#!/usr/bin/env bash
# Launch the OpenVid bidir teacher run (Wan2.1 1.3B Fun-I2V) with the SPARSE point
# conditioning recipe + MotionStream stage-1 hparams. Wraps examples/train/run_slurm.sh
# with this cluster's settings (partition=all, 4 GPU/node) and the WANTRACK_* env that
# selects sparse mode with NO track masking and NO fixed/overfit sampling.
#
# WANDB_API_KEY must be exported in the environment before calling (never hard-coded).
#
# Usage:  WANDB_API_KEY=<key> bash examples/train/run_openvid_bidir_slurm.sh [num_nodes] [extra --dotted overrides]
set -euo pipefail
# CFG overridable for the 14B run: CFG=.../finetune_wantrack_openvid_sparse_14b.yaml
CFG="${CFG:-examples/train/scenario/worldmodel/finetune_wantrack_openvid_sparse_1p3b.yaml}"
NODES="${1:-8}"; shift || true
: "${WANDB_API_KEY:?export WANDB_API_KEY before launching}"

# ---- sparse point conditioning (1-per-SAM-object + EXTRA_RANDOM extras), no masking ----
export WANTRACK_AUG=1
export WANTRACK_SPARSE=1
export WANTRACK_EXTRA_RANDOM=20
export WANTRACK_EXTRA_MODE=random
export WANTRACK_PMASK=0          # no stochastic track masking (initial stage)
export WANTRACK_FIXED_SAMPLE=0   # no deterministic/overfit sampling
export WANTRACK_MOTION_DROP=0
export WANTRACK_TEXT_DROP=0
export WANTRACK_DEBUG="${WANTRACK_DEBUG:-1}"   # log sampling/coverage stats (throttled)

export WANDB_MODE=online

# ---- compute nodes: /home is NOT mounted; point every ~-based cache at writable Lustre/tmp ----
export HOME=/mnt/lustre/vlm-s4duan
export HF_HOME=/mnt/lustre/vlm-s4duan/.hf
export TORCH_HOME=/mnt/lustre/vlm-s4duan/.torch
export MPLCONFIGDIR=/mnt/lustre/vlm-s4duan/.mpl
export TORCHINDUCTOR_CACHE_DIR=/tmp/inductor_cache   # node-local; triton cache set in run_slurm.sh

# ---- this cluster ----
export PARTITION=all
export NUM_GPUS=4          # GPUs per node
export MEM=0               # all node memory
export CPUS_PER_TASK=128
export JOB_NAME="${JOB_NAME:-openvid_bidir_1p3b}"
export OUTPUT_DIR=/mnt/lustre/vlm-s4duan/logs/slurm

bash examples/train/run_slurm.sh "$CFG" "$NODES" "$@"
