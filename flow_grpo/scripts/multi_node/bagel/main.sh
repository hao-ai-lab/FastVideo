#!/bin/bash

GPUS_PER_NODE=8
NUM_MACHINES=4
NUM_PROCESSES=$((NUM_MACHINES * GPUS_PER_NODE))
MASTER_PORT=19001
MASTER_ADDR=10.82.139.22

RANK=$1

accelerate launch --config_file scripts/accelerate_configs/fsdp.yaml \
    --num_machines ${NUM_MACHINES} --num_processes ${NUM_PROCESSES} \
    --machine_rank ${RANK} --main_process_ip ${MASTER_ADDR} --main_process_port ${MASTER_PORT} \
    scripts/train_bagel.py \
    --config config/grpo.py:pickscore_bagel
