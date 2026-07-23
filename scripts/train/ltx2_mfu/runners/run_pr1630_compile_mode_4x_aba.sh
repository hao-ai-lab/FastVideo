#!/usr/bin/env bash
set -euo pipefail

checkout=/mnt/fv-pr1630-pack-final
expected_head=fa47ce1ab570d33bb245a49f4cd63267282b2a54
expected_harness_hash=d7e571ac7043e4252f44260617e0de8f3467fd97ca3393393a04c64d1af482f3

actual_head=$(git -C "$checkout" rev-parse HEAD)
if [[ "$actual_head" != "$expected_head" ]]; then
    echo "unexpected checkout head: $actual_head" >&2
    exit 1
fi
if [[ -n $(git -C "$checkout" status --porcelain) ]]; then
    echo "benchmark checkout is dirty" >&2
    exit 1
fi
actual_harness_hash=$(sha256sum /mnt/benchmark_fastvideo_train_compile_mode.py | awk '{print $1}')
if [[ "$actual_harness_hash" != "$expected_harness_hash" ]]; then
    echo "unexpected benchmark harness hash: $actual_harness_hash" >&2
    exit 1
fi

export PYTHONPATH="$checkout"
export FASTVIDEO_FA4=1
export FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED=1
export FLASH_ATTENTION_CUTE_DSL_CACHE_DIR=/mnt/fa4-cache
export GLOO_SOCKET_IFNAME=enP5p9s0

run_variant() {
    local label=$1
    local compile_mode=$2
    local mode_args=()
    if [[ -n "$compile_mode" ]]; then
        mode_args=(--compile-mode "$compile_mode")
    fi
    echo "COMPILE_MODE_GATE starting=$label compile_mode=${compile_mode:-default}" >&2
    /mnt/FastVideo/.venv/bin/torchrun \
        --standalone \
        --nproc-per-node 4 \
        /mnt/benchmark_fastvideo_train_compile_mode.py \
        --singleton-timestep \
        "${mode_args[@]}" \
        --config "$checkout/examples/train/configs/overfit_ltx2_t2v.yaml" \
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
        --training.distributed.reduce_dtype bf16 \
        --training.data.train_batch_size 1 \
        --training.data.data_path /mnt/FastVideo/data/ltx2_overfit_preprocessed:256 \
        --training.data.dataloader_num_workers 1 \
        --training.loop.gradient_accumulation_steps 1 \
        --training.loop.max_train_steps 30 \
        --training.checkpoint.output_dir "/mnt/pr1630_compile_mode_4x_${label}_fa47" \
        --pipeline.dit_config.pack_attention_projections true \
        --callbacks.grad_clip.log_grad_norms true \
        --callbacks.validation.every_steps 0 \
        > "/mnt/pr1630_compile_mode_4x_${label}_fa47.log" 2>&1
    echo "COMPILE_MODE_GATE completed=$label" >&2
}

run_variant control_a ""
run_variant candidate max-autotune-no-cudagraphs
run_variant control_b ""
