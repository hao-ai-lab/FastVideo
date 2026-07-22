#!/usr/bin/env bash
set -euo pipefail

script=/mnt/pr1630_realloc1629519_native_health.py
expected_script_hash=693591c1fe44886b3cebc88c85a00ad2473e499c21a594af51549654bc57871c
test "$(sha256sum "$script" | awk '{print $1}')" = "$expected_script_hash"

export GLOO_SOCKET_IFNAME=enP5p9s0
export NCCL_SOCKET_IFNAME=enP5p9s0
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,NET
export NCCL_DEBUG_FILE="/mnt/pr1630_realloc1629519_nccl_%h_%p.log"

log=/mnt/pr1630_realloc1629519_native_health_node${NODE_RANK}.log
/mnt/FastVideo/.venv/bin/torchrun \
    --nnodes "$NNODES" \
    --node-rank "$NODE_RANK" \
    --nproc-per-node 4 \
    --master-addr "$MASTER_ADDR" \
    --master-port "$MASTER_PORT" \
    "$script" > "$log" 2>&1

grep '^HEALTH_GATE ' "$log"
sha256sum "$log"
