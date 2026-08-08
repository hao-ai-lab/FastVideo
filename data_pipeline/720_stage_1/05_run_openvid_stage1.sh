#!/bin/bash
# Step 05 — OpenVid Stage 1 (the big run, ~10 days). 14B/720p, merged init, HEAD FROZEN, sparse.
# Run the SAME command on BOTH racks. MASTER_ADDR = rack0 host on both; NODE_RANK 0 on rack0, 1 on rack1.
#   rack0: MASTER_ADDR=hpc-rack-1-6 NODE_RANK=0 bash data_pipeline/720_stage_1/05_run_openvid_stage1.sh
#   rack1: MASTER_ADDR=hpc-rack-1-6 NODE_RANK=1 bash data_pipeline/720_stage_1/05_run_openvid_stage1.sh
# resume_from_checkpoint: latest -> relaunch both to continue after any interruption.
set -uo pipefail
cd ~/FastVideo

: "${MASTER_ADDR:?set MASTER_ADDR to rack-0 hostname (reachable from both racks)}"
: "${NODE_RANK:?set NODE_RANK: 0 on the master rack, 1 on the other}"
MASTER_PORT=${MASTER_PORT:-29503}
NNODES=${NNODES:-8}                 # 8 nodes x 4 GB200 = 32 GPUs (matches num_gpus:32 / replicate 8 in the config)
GPUS_PER_NODE=${GPUS_PER_NODE:-4}
CFG=data_pipeline/720_stage_1/finetune_wantrack_openvid_stage1_14b_720p_d64_bias.yaml

export NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_3,mlx5_4
# export NCCL_DEBUG=INFO   # uncomment on the FIRST launch to confirm NET/IB, then re-comment.

# Stage 1: sparse conditioning, HEAD FROZEN (train the DiT to use the merged track pathway), no masking.
# Every WANTRACK_/TRACKWAN_ knob is set EXPLICITLY (no reliance on code defaults) to match upstream
# stage-1 (D) and to avoid inheriting the overfit launcher's opposite settings (IMAGE_COND=0, FIXED_SAMPLE=1).
FASTVIDEO_FA4=1 \
  TRACKWAN_TRACK_BIAS=1 \
  WANTRACK_FREEZE_HEAD=1 \
  WANTRACK_IMAGE_COND=1 \
  WANTRACK_AUG=1 \
  WANTRACK_SPARSE=1 WANTRACK_EXTRA_RANDOM=20 WANTRACK_EXTRA_MODE=random \
  WANTRACK_FIXED_SAMPLE=0 \
  WANTRACK_PMASK=0 WANTRACK_MASK_CHUNK=0 \
  WANTRACK_TRACK_DROP=0 WANTRACK_MOTION_DROP=0 WANTRACK_TEXT_DROP=0 \
  torchrun \
  --nnodes=${NNODES} --nproc_per_node=${GPUS_PER_NODE} --node_rank=${NODE_RANK} \
  --rdzv_id=openvid_stage1_14b_720p --rdzv_backend=c10d --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
  --log-dir ~/FastVideo/torchrun_logs \
  -m fastvideo.train.entrypoint.train \
  --config ${CFG} \
  --training.distributed.num_gpus $((NNODES * GPUS_PER_NODE)) \
  --training.checkpoint.resume_from_checkpoint latest \
  "$@" \
  2>&1 | tee -a data_pipeline/720_stage_1/openvid_stage1_node${NODE_RANK}.log
