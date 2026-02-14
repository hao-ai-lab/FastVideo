#!/bin/bash

GPUS_PER_NODE=8
NUM_MACHINES=4
NUM_PROCESSES=$((NUM_MACHINES * GPUS_PER_NODE))
MASTER_PORT=19001
MASTER_ADDR=10.82.139.22

RANK=$1

# Launch command using torchrun (native PyTorch distributed)
torchrun \
    --nproc_per_node=${GPUS_PER_NODE} \
    --nnodes=${NUM_MACHINES} \
    --node_rank=${RANK} \
    --master_addr=${MASTER_ADDR} \
    --master_port=${MASTER_PORT} \
    scripts/train_qwenimage.py \
    --config config/grpo.py:pickscore_qwenimage