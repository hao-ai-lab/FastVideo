#!/usr/bin/env bash
set -euo pipefail

checkout=/mnt/fv-pr1630-fsdp-group-source
expected_head=20c36acefc97e8b743f79a5c52883561853a7d85
expected_harness_hash=bf0861ff481c499fd65350d2f0f16c86487db88cb03c332e4be5c76540fcc7fd
harness=/mnt/benchmark_fastvideo_train_pack_d016.py

test "$(git -C "$checkout" rev-parse HEAD)" = "$expected_head"
test -z "$(git -C "$checkout" status --porcelain --untracked-files=no)"
test "$(sha256sum "$harness" | awk '{print $1}')" = "$expected_harness_hash"
cd "$checkout"

export PYTHONPATH="/mnt:$checkout"
export OMP_NUM_THREADS=1
export FASTVIDEO_FA4=1
export FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED=1
export FLASH_ATTENTION_CUTE_DSL_CACHE_DIR=/mnt/fa4-cache
export GLOO_SOCKET_IFNAME=enP5p9s0
export NCCL_CTA_POLICY=2
export WANDB_MODE=disabled
unset FASTVIDEO_FSDP2_AUTOWRAP || true
unset FASTVIDEO_TE_FP32_MASTER || true
unset TORCHINDUCTOR_MAX_AUTOTUNE_GEMM || true
unset TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_BACKENDS || true
unset TORCHINDUCTOR_CUTLASS_DIR || true

common_args=(
    --singleton-timestep
    --config "$checkout/examples/train/configs/overfit_ltx2_t2v.yaml"
    --models.student.attention_backend FLASH_ATTN
    --models.student.enable_gradient_checkpointing_type null
    --training.model.enable_gradient_checkpointing_type null
    --training.model.enable_torch_compile true
    --training.dit_precision fp32
    --training.optimizer.fused true
    --training.distributed.num_gpus 4
    --training.distributed.tp_size 1
    --training.distributed.sp_size 1
    --training.distributed.hsdp_replicate_dim 1
    --training.distributed.hsdp_shard_dim 4
    --training.distributed.reshard_after_forward false
    --training.distributed.fsdp_symmetric_memory true
    --training.distributed.fsdp_modules_per_group 2
    --training.distributed.reduce_dtype bf16
    --training.data.data_path /mnt/FastVideo/data/ltx2_overfit_preprocessed:256
    --training.data.dataloader_num_workers 1
    --training.loop.gradient_accumulation_steps 1
    --training.loop.max_train_steps 30
    --pipeline.dit_config.pack_attention_projections true
    --callbacks.grad_clip.log_grad_norms true
    --callbacks.validation.every_steps 0
)

run_variant() {
    local label=$1
    local batch_size=$2
    local log=/mnt/pr1630_batch2_group2_${label}_20c36.log
    /mnt/FastVideo/.venv/bin/torchrun \
        --standalone \
        --nproc-per-node 4 \
        "$harness" \
        "${common_args[@]}" \
        --training.data.train_batch_size "$batch_size" \
        --training.checkpoint.output_dir "/mnt/pr1630_batch2_group2_${label}_20c36" \
        > "$log" 2>&1
    grep -E '^(BF16_RESULT|BF16_SEMANTICS) ' "$log"
    sha256sum "$log"
}

run_variant control_a_b1 1
run_variant candidate_b2 2
run_variant control_b_b1 1
