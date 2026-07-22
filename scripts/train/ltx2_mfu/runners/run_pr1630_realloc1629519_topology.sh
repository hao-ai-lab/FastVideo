#!/usr/bin/env bash
set -euo pipefail

log=/mnt/pr1630_realloc1629519_topology_node${NODE_RANK}.log
{
    date -u '+UTC %Y-%m-%dT%H:%M:%SZ'
    hostname
    printf 'NODE_RANK=%s NNODES=%s MASTER_ADDR=%s\n' "$NODE_RANK" "$NNODES" "$MASTER_ADDR"
    printf 'NCCL_NVLS_ENABLE=%s NCCL_MNNVL_ENABLE=%s\n' \
        "${NCCL_NVLS_ENABLE:-<unset>}" "${NCCL_MNNVL_ENABLE:-<unset>}"
    nvidia-smi --query-gpu=index,name,uuid,pci.bus_id,driver_version,memory.total --format=csv
    nvidia-smi topo -m
    nvidia-smi -q
    ls -la /dev/nvidia-caps-imex-channels
} > "$log" 2>&1

sha256sum "$log"
