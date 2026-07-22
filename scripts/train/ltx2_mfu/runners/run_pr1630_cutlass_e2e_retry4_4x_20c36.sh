#!/usr/bin/env bash
set -euo pipefail

checkout=/mnt/fv-pr1630-fsdp-group-source
expected_head=20c36acefc97e8b743f79a5c52883561853a7d85
expected_harness_hash=bf0861ff481c499fd65350d2f0f16c86487db88cb03c332e4be5c76540fcc7fd
expected_shim_hash=c3699cb611a3b242040cab57c7e0b785ec28003896fff98c235ec4474d3e9752
expected_cutlass_head=e67e63c331d6e4b729047c95cf6b92c8454cba89
harness=/mnt/benchmark_fastvideo_train_pack_d016.py
shim=/mnt/torchrun_rank_local_inductor_cache.py
cutlass=/mnt/FastVideo/fastvideo-kernel/include/cutlass

test "$(git -C "$checkout" rev-parse HEAD)" = "$expected_head"
test -z "$(git -C "$checkout" status --porcelain --untracked-files=no)"
test "$(sha256sum "$harness" | awk '{print $1}')" = "$expected_harness_hash"
test "$(sha256sum "$shim" | awk '{print $1}')" = "$expected_shim_hash"
test "$(git -C "$cutlass" rev-parse HEAD)" = "$expected_cutlass_head"
test -d "$cutlass/python/cutlass_library"
test -d "$cutlass/python/cutlass_cppgen"
cd "$checkout"

export PYTHONPATH="/mnt:$checkout"
export OMP_NUM_THREADS=1
export FASTVIDEO_FA4=1
export FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED=1
export FLASH_ATTENTION_CUTE_DSL_CACHE_DIR=/mnt/fa4-cache
export GLOO_SOCKET_IFNAME=enP5p9s0
export NCCL_CTA_POLICY=2
export WANDB_MODE=disabled
export TORCHINDUCTOR_CUTLASS_DIR="$cutlass"
export TORCHINDUCTOR_CUTLASS_INSTANTIATION_LEVEL=0
export CUTLASS_EPILOGUE_FUSION=0
unset FASTVIDEO_FSDP2_AUTOWRAP || true
unset FASTVIDEO_TE_FP32_MASTER || true

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
    --training.data.train_batch_size 1
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
    local autotune=$2
    local backends=$3
    local compile_threads=$4
    local cache=/mnt/pr1630_cutlass_e2e_cache_${label}_retry4_20c36
    local log=/mnt/pr1630_cutlass_e2e_${label}_retry4_20c36.log
    test ! -e "$cache"
    export TORCHINDUCTOR_CACHE_DIR_BASE="$cache"
    echo "CUTLASS_E2E starting=$label autotune=$autotune backends=$backends rank_local_cache=true compile_threads=$compile_threads"
    TORCHINDUCTOR_COMPILE_THREADS="$compile_threads" \
    TORCHINDUCTOR_MAX_AUTOTUNE_GEMM="$autotune" \
    TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_BACKENDS="$backends" \
        /mnt/FastVideo/.venv/bin/torchrun \
            --standalone \
            --nproc-per-node 4 \
            "$shim" \
            "$harness" \
            "${common_args[@]}" \
            --training.checkpoint.output_dir "/mnt/pr1630_cutlass_e2e_${label}_retry4_20c36" \
            > "$log" 2>&1
    echo "CUTLASS_E2E completed=$label"
    grep -E '^(BF16_RESULT|BF16_SEMANTICS) ' "$log"
    sha256sum "$log"
}

# Four ranks x the default 32 workers exhausted the 956-GiB host plus swap.
# Four workers per rank keeps compilation parallel without changing generated kernels.
run_variant cutlass 1 ATEN,CUTLASS 4
run_variant current_b 0 ATEN,TRITON,CPP 32
