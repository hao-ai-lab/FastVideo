#!/usr/bin/env bash
# Run the modular Attn-QAT finetune recipe.
set -euo pipefail

DATA_DIR=${1:-data/HD-Mixkit-Finetune-Wan/combined_parquet_dataset}
NUM_GPUS=${NUM_GPUS:-4}
export NUM_GPUS

bash examples/train/run.sh \
    examples/train/scenario/qad_wan2_1_mixkit/stage1_attn_qat_finetune.yaml \
    --training.data.data_path "${DATA_DIR}" \
    --training.distributed.num_gpus "${NUM_GPUS}" \
    --training.distributed.sp_size "${NUM_GPUS}" \
    --training.distributed.hsdp_replicate_dim 1 \
    --training.distributed.hsdp_shard_dim "${NUM_GPUS}"
