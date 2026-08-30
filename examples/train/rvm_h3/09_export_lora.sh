#!/usr/bin/env bash
set -euo pipefail
# shellcheck source=common.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
activate_rvm_env

CONFIG="${1:?Usage: $0 <config.yaml> <checkpoint-dir> [adapter.safetensors]}"
CHECKPOINT="${2:?Usage: $0 <config.yaml> <checkpoint-dir> [adapter.safetensors]}"
OUTPUT="${3:-${CHECKPOINT%/}/fasth3_rvm_lora.safetensors}"
NUM_GPUS="${NUM_GPUS:-8}"
RVM_SP_SIZE="${RVM_SP_SIZE:-4}"
if (( NUM_GPUS % RVM_SP_SIZE != 0 )); then
    echo "NUM_GPUS=${NUM_GPUS} must be divisible by RVM_SP_SIZE=${RVM_SP_SIZE}" >&2
    exit 1
fi
REPLICAS=$((NUM_GPUS / RVM_SP_SIZE))

python -m torch.distributed.run \
    --nproc_per_node "${NUM_GPUS}" \
    --master_addr "${MASTER_ADDR:-127.0.0.1}" \
    --master_port "${MASTER_PORT:-29521}" \
    -m fastvideo.train.entrypoint.export_rvm_lora \
    --config "${CONFIG}" \
    --checkpoint "${CHECKPOINT}" \
    --output "${OUTPUT}" \
    --num-gpus "${NUM_GPUS}" \
    --sp-size "${RVM_SP_SIZE}" \
    --hsdp-replicate-dim "${REPLICAS}" \
    --hsdp-shard-dim "${RVM_SP_SIZE}"

echo "Exported inference adapter: ${OUTPUT}"
