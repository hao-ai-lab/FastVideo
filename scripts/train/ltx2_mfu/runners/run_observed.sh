#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
checkout=${CHECKOUT:-$(cd -- "$script_dir/../../../.." && pwd)}
config=${CONFIG:-$checkout/examples/train/configs/overfit_ltx2_t2v.yaml}
num_gpus=${NUM_GPUS:-4}
local_batch_size=${LOCAL_BATCH_SIZE:-2}
max_steps=${MAX_STEPS:-300}
validation_every=${VALIDATION_EVERY:-50}
data_path=${DATA_PATH:-$checkout/data/ltx2_overfit_preprocessed:600}
validation_file=${VALIDATION_FILE:-$checkout/data/ltx2_overfit_preprocessed/validation_prompts.json}
run_name=${RUN_NAME:-ltx2-mfu-observed-$(date -u +%Y%m%dT%H%M%SZ)}
output_dir=${OUTPUT_DIR:-/tmp/$run_name}

if [[ "${NNODES:-1}" != 1 ]]; then
    echo "run_observed.sh is single-node; unset NNODES or set it to 1" >&2
    exit 2
fi
if [[ "$num_gpus" != 4 ]]; then
    echo "run_observed.sh currently supports the validated single-node 4-GPU path; got $num_gpus" >&2
    exit 2
fi

export NUM_GPUS=$num_gpus
export LOG_DIR=${LOG_DIR:-$output_dir/logs}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export FASTVIDEO_FA4=${FASTVIDEO_FA4:-1}
export FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED=${FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED:-1}
export FLASH_ATTENTION_CUTE_DSL_CACHE_DIR=${FLASH_ATTENTION_CUTE_DSL_CACHE_DIR:-/tmp/fastvideo-fa4-cache}
export NCCL_CTA_POLICY=${NCCL_CTA_POLICY:-2}
export WANDB_MODE=${WANDB_MODE:-online}

cd "$checkout"
bash examples/train/run.sh \
    "$config" \
    --models.student.attention_backend FLASH_ATTN \
    --models.student.enable_gradient_checkpointing_type null \
    --training.model.enable_gradient_checkpointing_type null \
    --training.model.enable_torch_compile true \
    --training.dit_precision fp32 \
    --training.optimizer.fused true \
    --training.distributed.num_gpus "$num_gpus" \
    --training.distributed.tp_size 1 \
    --training.distributed.sp_size 1 \
    --training.distributed.hsdp_replicate_dim 1 \
    --training.distributed.hsdp_shard_dim "$num_gpus" \
    --training.distributed.reshard_after_forward false \
    --training.distributed.fsdp_symmetric_memory true \
    --training.distributed.fsdp_modules_per_group 2 \
    --training.distributed.reduce_dtype bf16 \
    --training.data.train_batch_size "$local_batch_size" \
    --training.data.data_path "$data_path" \
    --training.data.dataloader_num_workers 1 \
    --training.loop.gradient_accumulation_steps 1 \
    --training.loop.max_train_steps "$max_steps" \
    --training.checkpoint.output_dir "$output_dir" \
    --training.tracker.run_name "$run_name" \
    --pipeline.dit_config.pack_attention_projections true \
    --callbacks.validation.dataset_file "$validation_file" \
    --callbacks.validation.every_steps "$validation_every" \
    --callbacks.validation.unload_pipeline_after_validation true \
    --callbacks.validation.offload_training_state false
