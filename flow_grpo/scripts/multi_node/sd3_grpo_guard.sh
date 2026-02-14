#!/bin/bash
# Common part for all nodes
export NCCL_IB_DISABLE=0
export NCCL_IB_HCA=mlx5
export NCCL_DEBUG=WARN
export NCCL_IB_GID_INDEX=3


MASTER_PORT=19001
RANK=$1
MASTER_ADDR=10.82.139.22
# Launch command (parameters automatically read from accelerate_multi_node.yaml)
accelerate launch --config_file scripts/accelerate_configs/multi_node.yaml \
    --num_machines 2 --num_processes 16 \
    --machine_rank ${RANK} --main_process_ip ${MASTER_ADDR} --main_process_port ${MASTER_PORT} \
    scripts/train_sd3_GRPO_Guard.py \
    --config config/grpo_guard.py:general_ocr_sd3_rationorm
