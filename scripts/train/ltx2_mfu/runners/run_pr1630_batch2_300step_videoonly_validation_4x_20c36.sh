#!/usr/bin/env bash
set -euo pipefail

checkout=/mnt/fv-pr1630-videoonly-validation-20c36
expected_head=20c36acefc97e8b743f79a5c52883561853a7d85
expected_denoising_hash=ae64970dfef610fa68ff9157e070fb3083fafe91018f4512cf42b7748f582674
denoising=fastvideo/pipelines/basic/ltx2/stages/ltx2_denoising.py
run_tag=${RUN_TAG:-pr1630_batch2_videoonly_validation_20c36}
max_steps=${MAX_STEPS:-300}
data_repeat=${DATA_REPEAT:-600}
validation_every=${VALIDATION_EVERY:-50}
unload_pipeline=${UNLOAD_PIPELINE_AFTER_VALIDATION:-false}
offload_training_state=${OFFLOAD_TRAINING_STATE:-false}

test "$(git -C "$checkout" rev-parse HEAD)" = "$expected_head"
test "$(git -C "$checkout" diff --name-only)" = "$denoising"
test "$(sha256sum "$checkout/$denoising" | awk '{print $1}')" = "$expected_denoising_hash"
git -C "$checkout" diff --check
cd "$checkout"

export PYTHONPATH="$checkout"
export OMP_NUM_THREADS=1
export FASTVIDEO_FA4=1
export FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED=1
export FLASH_ATTENTION_CUTE_DSL_CACHE_DIR=/mnt/fa4-cache
export GLOO_SOCKET_IFNAME=enP5p9s0
export NCCL_CTA_POLICY=2
export TORCHINDUCTOR_CACHE_DIR="/mnt/${run_tag}_cache"
if [[ "${DISABLE_WANDB:-false}" == true ]]; then
    export WANDB_MODE=disabled
else
    unset WANDB_MODE || true
fi
unset FASTVIDEO_FSDP2_AUTOWRAP || true
unset FASTVIDEO_TE_FP32_MASTER || true
unset TORCHINDUCTOR_MAX_AUTOTUNE_GEMM || true
unset TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_BACKENDS || true
unset TORCHINDUCTOR_CUTLASS_DIR || true
unset TORCHINDUCTOR_CUTLASS_ALLOWLIST || true

export NUM_GPUS=4
export LOG_DIR="/mnt/${run_tag}_logs"

bash examples/train/run.sh \
    "$checkout/examples/train/configs/overfit_ltx2_t2v.yaml" \
    --models.student.attention_backend FLASH_ATTN \
    --models.student.enable_gradient_checkpointing_type null \
    --training.model.enable_gradient_checkpointing_type null \
    --training.model.enable_torch_compile true \
    --training.dit_precision fp32 \
    --training.optimizer.fused true \
    --training.distributed.num_gpus 4 \
    --training.distributed.tp_size 1 \
    --training.distributed.sp_size 1 \
    --training.distributed.hsdp_replicate_dim 1 \
    --training.distributed.hsdp_shard_dim 4 \
    --training.distributed.reshard_after_forward false \
    --training.distributed.fsdp_symmetric_memory true \
    --training.distributed.fsdp_modules_per_group 2 \
    --training.distributed.reduce_dtype bf16 \
    --training.data.train_batch_size 2 \
    --training.data.data_path "/mnt/FastVideo/data/ltx2_overfit_preprocessed:${data_repeat}" \
    --training.data.dataloader_num_workers 1 \
    --training.loop.gradient_accumulation_steps 1 \
    --training.loop.max_train_steps "$max_steps" \
    --training.checkpoint.output_dir "/mnt/${run_tag}_output" \
    --training.tracker.run_name "$run_tag" \
    --pipeline.dit_config.pack_attention_projections true \
    --callbacks.validation.dataset_file /mnt/FastVideo/data/ltx2_overfit_preprocessed/validation_prompts.json \
    --callbacks.validation.every_steps "$validation_every" \
    --callbacks.validation.unload_pipeline_after_validation "$unload_pipeline" \
    --callbacks.validation.offload_training_state "$offload_training_state"
