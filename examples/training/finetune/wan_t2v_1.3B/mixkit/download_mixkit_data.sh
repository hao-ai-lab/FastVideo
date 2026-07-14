#!/bin/bash
# Download the preprocessed MixKit finetune dataset used for the QAD 5090 recipe.
#
# This is the MixKit subset already VAE-encoded (Wan2.1-T2V-1.3B) and text-embedded
# into Parquet shards, so it can be fed straight to training with no further
# preprocessing. To build the Parquet from raw videos yourself, see README.md.
#
# Usage (run from the repo root):
#   bash examples/training/finetune/wan_t2v_1.3B/mixkit/download_mixkit_data.sh [DATA_ROOT]
set -euo pipefail

DATA_ROOT=${1:-data/HD-Mixkit-Finetune-Wan}

python scripts/huggingface/download_hf.py \
    --repo_id "weizhou03/HD-Mixkit-Finetune-Wan" \
    --local_dir "${DATA_ROOT}" \
    --repo_type "dataset"

echo "Done."
echo "  Train data:      ${DATA_ROOT}/combined_parquet_dataset"
echo "  Validation data: ${DATA_ROOT}/validation_parquet_dataset"
echo "Point your training script's data path at the combined_parquet_dataset directory."
