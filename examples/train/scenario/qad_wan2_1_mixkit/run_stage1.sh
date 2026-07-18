#!/usr/bin/env bash
# Run the modular Attn-QAT finetune recipe.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
cd "${REPO_ROOT}"

DATA_DIR=${1:-data/HD-Mixkit-Finetune-Wan/combined_parquet_dataset}
NUM_GPUS=${NUM_GPUS:-4}
export NUM_GPUS
export FASTVIDEO_ATTN_QAT_FWD_EXACT_M=${FASTVIDEO_ATTN_QAT_FWD_EXACT_M:-0}

bash "${REPO_ROOT}/examples/train/run.sh" \
    "${SCRIPT_DIR}/stage1_attn_qat_finetune.yaml" \
    --training.data.data_path "${DATA_DIR}" \
    --training.distributed.num_gpus "${NUM_GPUS}" \
    --training.distributed.sp_size "${NUM_GPUS}" \
    --training.distributed.hsdp_replicate_dim 1 \
    --training.distributed.hsdp_shard_dim "${NUM_GPUS}"
