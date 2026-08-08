#!/bin/bash
# Step 05 (2-RACK variant) — OpenVid Stage 1 on 2 racks (2x4 = 8 GB200) instead of 8 nodes.
#
# SAME effective global batch as the 8-node 05: grad_accum is bumped 4x (4 -> 16) to compensate for
# 4x fewer GPUs, so the optimization is equivalent (see note below) — only the wall-clock is ~4x
# (~48 days vs ~12). Run the SAME command on BOTH racks; MASTER_ADDR = rack0 host on both,
# NODE_RANK 0 on rack0, 1 on rack1.
#   rack0: MASTER_ADDR=hpc-rack-1-6 NODE_RANK=0 bash data_pipeline/720_stage_1/05b_run_openvid_stage1_2rack.sh
#   rack1: MASTER_ADDR=hpc-rack-1-6 NODE_RANK=1 bash data_pipeline/720_stage_1/05b_run_openvid_stage1_2rack.sh
# resume_from_checkpoint: latest -> relaunch both to continue after any interruption.
set -uo pipefail
cd ~/FastVideo

: "${MASTER_ADDR:?set MASTER_ADDR to rack-0 hostname (reachable from both racks)}"
: "${NODE_RANK:?set NODE_RANK: 0 on the master rack, 1 on the other}"
MASTER_PORT=${MASTER_PORT:-29503}
NNODES=${NNODES:-2}                 # 2 racks x 4 GB200 = 8 GPUs
GPUS_PER_NODE=${GPUS_PER_NODE:-4}
CFG=data_pipeline/720_stage_1/finetune_wantrack_openvid_stage1_14b_720p_d64_bias.yaml
# bf16 parquets (complete 259k set, ~4.7TB). Loader honors the per-field _dtype; same clips/order
# as fp32 (so val_sample_indices are unchanged). Override to the fp32 set if ever needed:
#   DATA_PATH=/home/hal-shared/motionstream/data/openvid-wantrack-parquets
DATA_PATH=${DATA_PATH:-/home/hal-shared/motionstream/data/openvid-wantrack-parquets-bf16}

# --- batch math: hold the effective global batch EQUAL to the 8-node 05 ------------------
# 8-node config: replicate 8 x shard 4 = 32 GPUs, grad_accum 4.
# 2 racks:       replicate 2 x shard 4 =  8 GPUs, grad_accum 16.
# 4x fewer GPUs x 4x grad_accum = same effective batch under EITHER counting convention
# (num_gpus- or replicate_dim-based both scale by 4). Model still shards across 4 GPUs
# (shard_dim 4) as in the 8-node run, so per-GPU memory is unchanged (no OOM risk from this).
HSDP_REPLICATE=${HSDP_REPLICATE:-2}
HSDP_SHARD=${HSDP_SHARD:-4}
GRAD_ACCUM=${GRAD_ACCUM:-16}

export NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_3,mlx5_4
# export NCCL_DEBUG=INFO   # uncomment on the FIRST launch to confirm NET/IB, then re-comment.

# Stage 1 env — identical to 05 (every knob explicit): sparse, HEAD FROZEN, CLIP on, no masking.
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
  --rdzv_id=openvid_stage1_14b_720p_2rack --rdzv_backend=c10d --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
  --log-dir ~/FastVideo/torchrun_logs \
  -m fastvideo.train.entrypoint.train \
  --config ${CFG} \
  --training.data.data_path ${DATA_PATH} \
  --training.distributed.num_gpus $((NNODES * GPUS_PER_NODE)) \
  --training.distributed.hsdp_replicate_dim ${HSDP_REPLICATE} \
  --training.distributed.hsdp_shard_dim ${HSDP_SHARD} \
  --training.loop.gradient_accumulation_steps ${GRAD_ACCUM} \
  --training.checkpoint.resume_from_checkpoint latest \
  --training.checkpoint.training_state_checkpointing_steps 20 \
  --training.checkpoint.checkpoints_total_limit 50 \
  --callbacks.track_validation.validate_at_start false \
  --callbacks.track_validation.every_steps 100 \
  --callbacks.track_validation.val_sample_indices "[1660, 1888]" \
  "$@" \
  2>&1 | tee -a data_pipeline/720_stage_1/openvid_stage1_2rack_node${NODE_RANK}.log
