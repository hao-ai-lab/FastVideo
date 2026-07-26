#!/usr/bin/env bash
set -euo pipefail

checkout=/mnt/fv-pr1630-opt-495
expected_head=49508050b
actual_head=$(git -C "$checkout" rev-parse --short=9 HEAD)
if [[ "$actual_head" != "$expected_head" ]]; then
    echo "unexpected checkout head: $actual_head" >&2
    exit 1
fi
if [[ -n "$(git -C "$checkout" status --porcelain)" ]]; then
    echo "benchmark checkout is dirty" >&2
    exit 1
fi

export PYTHONPATH="$checkout"
export FASTVIDEO_FA4=1
export FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED=1
export FLASH_ATTENTION_CUTE_DSL_CACHE_DIR=/mnt/fa4-cache
export GLOO_SOCKET_IFNAME=enP5p9s0

run_variant() {
    local label=$1
    local batch_size=$2
    local reshard_after_forward=$3
    /mnt/FastVideo/.venv/bin/torchrun \
        --nnodes "$NNODES" \
        --node-rank "$NODE_RANK" \
        --nproc-per-node 4 \
        --master-addr "$MASTER_ADDR" \
        --master-port "$MASTER_PORT" \
        /mnt/benchmark_fastvideo_train_ltx2_singleton_timestep.py \
        --singleton-timestep \
        --config "$checkout/examples/train/configs/overfit_ltx2_t2v.yaml" \
        --models.student.attention_backend FLASH_ATTN \
        --models.student.enable_gradient_checkpointing_type null \
        --training.model.enable_gradient_checkpointing_type null \
        --training.model.enable_torch_compile true \
        --training.dit_precision fp32 \
        --training.optimizer.fused true \
        --training.distributed.num_gpus 8 \
        --training.distributed.tp_size 1 \
        --training.distributed.sp_size 1 \
        --training.distributed.hsdp_replicate_dim 1 \
        --training.distributed.hsdp_shard_dim 8 \
        --training.distributed.reshard_after_forward "$reshard_after_forward" \
        --training.distributed.fsdp_symmetric_memory true \
        --training.distributed.reduce_dtype bf16 \
        --training.data.train_batch_size "$batch_size" \
        --training.data.data_path /mnt/FastVideo/data/ltx2_overfit_preprocessed:256 \
        --training.data.dataloader_num_workers 1 \
        --training.loop.gradient_accumulation_steps 1 \
        --training.loop.max_train_steps 30 \
        --training.checkpoint.output_dir "/mnt/pr1630_batch4_full_shard_8x_${label}" \
        --callbacks.grad_clip.log_grad_norms true \
        --callbacks.validation.every_steps 0 \
        > "/mnt/pr1630_batch4_full_shard_8x_${label}_node${NODE_RANK}.log" 2>&1
}

run_variant control_a 3 false
run_variant candidate 4 true
run_variant control_b 3 false
