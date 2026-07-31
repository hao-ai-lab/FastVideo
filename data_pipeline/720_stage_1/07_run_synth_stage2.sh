#!/bin/bash
# Step 07 — Stage 2 (robustness, ~1 day). 14B/720p, inits from exported Stage-1 teacher, HEAD FROZEN,
# sparse + heavy masking/dropout. Produces the FINAL bidir teacher. Run on BOTH racks (same command,
# NODE_RANK 0/1, MASTER_ADDR = rack0).
#   rack0: MASTER_ADDR=hpc-rack-1-6 NODE_RANK=0 bash data_pipeline/720_stage_1/07_run_synth_stage2.sh
#   rack1: MASTER_ADDR=hpc-rack-1-6 NODE_RANK=1 bash data_pipeline/720_stage_1/07_run_synth_stage2.sh
set -uo pipefail
cd ~/FastVideo

: "${MASTER_ADDR:?set MASTER_ADDR to rack-0 hostname (reachable from both racks)}"
: "${NODE_RANK:?set NODE_RANK: 0 on the master rack, 1 on the other}"
MASTER_PORT=${MASTER_PORT:-29504}
NNODES=${NNODES:-8}                 # 8 nodes x 4 GB200 = 32 GPUs (matches num_gpus:32 / replicate 8 in the config)
GPUS_PER_NODE=${GPUS_PER_NODE:-4}
CFG=data_pipeline/720_stage_1/finetune_wantrack_synth_stage2_14b_720p_d64_bias.yaml

export NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_3,mlx5_4
# export NCCL_DEBUG=INFO   # first launch only.

# Stage 2: sparse + robustness masking/dropout, HEAD still FROZEN.
FASTVIDEO_FA4=1 \
  TRACKWAN_TRACK_BIAS=1 WANTRACK_FREEZE_HEAD=1 \
  WANTRACK_AUG=1 WANTRACK_SPARSE=1 WANTRACK_EXTRA_RANDOM=20 \
  WANTRACK_TRACK_DROP=0.5 WANTRACK_MOTION_DROP=0.3 WANTRACK_PMASK=0.2 WANTRACK_MASK_CHUNK=8 \
  torchrun \
  --nnodes=${NNODES} --nproc_per_node=${GPUS_PER_NODE} --node_rank=${NODE_RANK} \
  --rdzv_id=synth_stage2_14b_720p --rdzv_backend=c10d --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
  --log-dir ~/FastVideo/torchrun_logs \
  -m fastvideo.train.entrypoint.train \
  --config ${CFG} \
  --training.distributed.num_gpus $((NNODES * GPUS_PER_NODE)) \
  --training.checkpoint.resume_from_checkpoint latest \
  "$@" \
  2>&1 | tee -a data_pipeline/720_stage_1/synth_stage2_node${NODE_RANK}.log
