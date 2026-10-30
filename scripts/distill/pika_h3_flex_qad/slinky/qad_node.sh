#!/bin/bash
# usage: qad_node.sh <node_rank> <master_ip> <out_dir> <config> [overrides]  (4 nodes x 4 GPUs; runs under env.sh)
NR=$1; MASTER=$2; OUT=$3; CFG=$4; shift 4
mkdir -p $OUT
FASTVIDEO_PG_TIMEOUT_MIN=120 $PY -m torch.distributed.run --nnodes=4 --nproc_per_node=4 --node_rank=$NR --master_addr=$MASTER \
  --master_port=29650 --log-dir $OUT/torchlogs-n$NR --redirects 3 --tee 3 -m fastvideo.train.entrypoint.train --config $CFG "$@"
