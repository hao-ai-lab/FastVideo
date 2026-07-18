#!/usr/bin/env bash
# Run modular DMD2 with Attn-QAT on the student only.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
cd "${REPO_ROOT}"

DATA_DIR=${1:-data/HD-Mixkit-Finetune-Wan/combined_parquet_dataset}
INIT_WEIGHTS=${2:-checkpoints/wan_t2v_qat_finetune/diffusers/transformer/model.safetensors}
NUM_GPUS=${NUM_GPUS:-4}
export NUM_GPUS

if [[ ! -f "${INIT_WEIGHTS}" ]]; then
    echo "Missing exported stage-1 weights: ${INIT_WEIGHTS}" >&2
    echo "Run export_stage1.sh before stage 2." >&2
    exit 1
fi

bash "${REPO_ROOT}/examples/train/run.sh" \
    "${SCRIPT_DIR}/stage2_attn_qat_dmd.yaml" \
    --models.student.transformer_override_safetensor "${INIT_WEIGHTS}" \
    --training.data.data_path "${DATA_DIR}" \
    --training.distributed.num_gpus "${NUM_GPUS}" \
    --training.distributed.sp_size 1 \
    --training.distributed.hsdp_replicate_dim "${NUM_GPUS}" \
    --training.distributed.hsdp_shard_dim 1
