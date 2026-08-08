#!/bin/bash
# Phase 0, step 2 — overfit the track pathway. 14B/720p, d64+bias, sparse, HEAD TRAINABLE.
# Run the SAME command on BOTH racks (2x4 GB200). MASTER_ADDR = rack0 host on both; NODE_RANK
# differs (0 on rack0, 1 on rack1). STAGE=A uses fixed track IDs (lock onto the pattern), STAGE=B
# uses random IDs; run A first, then B (B resumes A's checkpoint via resume_from_checkpoint: latest).
#
#   rack0: MASTER_ADDR=hpc-rack-1-6 NODE_RANK=0 STAGE=A bash data_pipeline/720_stage_1/02_run_overfit.sh
#   rack1: MASTER_ADDR=hpc-rack-1-6 NODE_RANK=1 STAGE=A bash data_pipeline/720_stage_1/02_run_overfit.sh
# Extra args pass through to the trainer, e.g. ... STAGE=B bash 02_run_overfit.sh --training.loop.max_train_steps 1600
set -uo pipefail
cd ~/FastVideo

: "${MASTER_ADDR:?set MASTER_ADDR to rack-0 hostname (reachable from both racks)}"
: "${NODE_RANK:?set NODE_RANK: 0 on the master rack, 1 on the other}"
STAGE=${STAGE:-A}
MASTER_PORT=${MASTER_PORT:-29502}
CFG=data_pipeline/720_stage_1/finetune_wantrack_overfit_14b_720p_d64_bias.yaml

# Stage A = deterministic track IDs (fixed sampling); Stage B = random IDs.
# B writes to its OWN output dir (keeps A's dir pristine, A/B checkpoints separated) — mirrors
# upstream chain_stepAB.sh + run_stepB_seed.sh. Seed B's dir once with A's final checkpoint via
# 02b_seed_stageB.sh BEFORE launching B, so resume_from_checkpoint: latest picks up A's weights.
OUT_A=/home/hal-kevin/data/motion-stream-test/overfit_14b_720p_d64_bias_out
OUT_B=/home/hal-kevin/data/motion-stream-test/overfit_14b_720p_d64_bias_stageB_out
case "$STAGE" in
  A) FIXED=1; OUT_DIR=$OUT_A ;;
  B) FIXED=0; OUT_DIR=$OUT_B ;;
  *) echo "STAGE must be A or B (got '$STAGE')" >&2; exit 1 ;;
esac
if [ "$STAGE" = B ] && [ ! -d "$OUT_B" ]; then
  echo "[overfit] ERROR: STAGE=B but $OUT_B does not exist." >&2
  echo "          Seed it first (run ONCE): bash data_pipeline/720_stage_1/02b_seed_stageB.sh" >&2
  exit 1
fi

# Pin NCCL to the four active 400G InfiniBand HCAs (keep off the 200G Ethernet port).
export NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_3,mlx5_4
# export NCCL_DEBUG=INFO   # uncomment on the FIRST launch to confirm NET/IB, then re-comment.

FASTVIDEO_FA4=1 \
  TRACKWAN_TRACK_BIAS=1 WANTRACK_FREEZE_HEAD=0 \
  WANTRACK_AUG=1 WANTRACK_SPARSE=1 WANTRACK_EXTRA_RANDOM=20 \
  WANTRACK_PMASK=0 WANTRACK_MASK_CHUNK=0 \
  WANTRACK_IMAGE_COND=0 \
  WANTRACK_FIXED_SAMPLE=${FIXED} \
  torchrun \
  --nnodes=2 --nproc_per_node=4 --node_rank=${NODE_RANK} \
  --rdzv_id=overfit_14b_720p --rdzv_backend=c10d --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
  --log-dir ~/FastVideo/torchrun_logs \
  -m fastvideo.train.entrypoint.train \
  --config ${CFG} \
  --training.checkpoint.output_dir ${OUT_DIR} \
  --training.tracker.run_name overfit-14b-720p-d64-bias-stage${STAGE} \
  --training.checkpoint.resume_from_checkpoint latest \
  "$@" \
  2>&1 | tee -a data_pipeline/720_stage_1/overfit_node${NODE_RANK}_stage${STAGE}.log
