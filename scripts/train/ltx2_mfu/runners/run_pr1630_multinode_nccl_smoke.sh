#!/usr/bin/env bash
set -euo pipefail

checkout=/mnt/fv-pr1630-fsdp-group-source
script=/mnt/pr1630_multinode_nccl_smoke.py
expected_head=20c36acefc97e8b743f79a5c52883561853a7d85
expected_script_hash=99c74569ba7807e2786b42d80b474ff77099e14aeecbffac81250555fc92e7ef

test "$(git -C "$checkout" rev-parse HEAD)" = "$expected_head"
test "$(sha256sum "$script" | awk '{print $1}')" = "$expected_script_hash"
cd "$checkout"

export PYTHONPATH="$checkout"
export GLOO_SOCKET_IFNAME=enP5p9s0
export NCCL_SOCKET_IFNAME=enP5p9s0
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,NET
export NCCL_DEBUG_FILE="/mnt/pr1630_nccl_smoke_%h_%p.log"

log=/mnt/pr1630_multinode_nccl_smoke_node${NODE_RANK}.log
/mnt/FastVideo/.venv/bin/torchrun \
    --nnodes "$NNODES" \
    --node-rank "$NODE_RANK" \
    --nproc-per-node 4 \
    --master-addr "$MASTER_ADDR" \
    --master-port "$MASTER_PORT" \
    "$script" > "$log" 2>&1

grep '^NCCL_SMOKE ' "$log"
sha256sum "$log"
