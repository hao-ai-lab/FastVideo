#!/usr/bin/env bash
set -euo pipefail

checkout=/mnt/fv-pr1630-pack-d016237
expected_head=d01623709636d0d49b310caec3c86d196fc22499
expected_harness_hash=534f201aadfc3cb02df2ad130a1193c01ebfa97ddcfd38fd4ee05a5a1e4965a5

actual_head=$(git -C "$checkout" rev-parse HEAD)
if [[ "$actual_head" != "$expected_head" ]]; then
    echo "unexpected checkout head: $actual_head" >&2
    exit 1
fi
if [[ -n $(git -C "$checkout" status --porcelain) ]]; then
    echo "benchmark checkout is dirty" >&2
    exit 1
fi
actual_harness_hash=$(sha256sum /mnt/benchmark_fastvideo_train_pack_b3.py | awk '{print $1}')
if [[ "$actual_harness_hash" != "$expected_harness_hash" ]]; then
    echo "unexpected benchmark harness hash: $actual_harness_hash" >&2
    exit 1
fi

export PYTHONPATH="$checkout"
export FASTVIDEO_FA4=1
export FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED=1
export FLASH_ATTENTION_CUTE_DSL_CACHE_DIR=/mnt/fa4-cache
export FASTVIDEO_BENCH_LOCAL_BATCH_SIZE=3
export GLOO_SOCKET_IFNAME=enP5p9s0

run_variant() {
    local label=$1
    local packed=$2
    echo "PACK_B3_GATE starting=$label node_rank=$NODE_RANK packed=$packed" >&2
    /mnt/FastVideo/.venv/bin/torchrun \
        --nnodes "$NNODES" \
        --node-rank "$NODE_RANK" \
        --nproc-per-node 4 \
        --master-addr "$MASTER_ADDR" \
        --master-port "$MASTER_PORT" \
        /mnt/benchmark_fastvideo_train_pack_b3.py \
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
        --training.distributed.reshard_after_forward false \
        --training.distributed.fsdp_symmetric_memory true \
        --training.distributed.reduce_dtype bf16 \
        --training.data.train_batch_size 3 \
        --training.data.data_path /mnt/FastVideo/data/ltx2_overfit_preprocessed:256 \
        --training.data.dataloader_num_workers 1 \
        --training.loop.gradient_accumulation_steps 1 \
        --training.loop.max_train_steps 30 \
        --training.checkpoint.output_dir "/mnt/pr1630_pack_8x_b3_${label}_d016" \
        --pipeline.dit_config.pack_attention_projections "$packed" \
        --callbacks.grad_clip.log_grad_norms true \
        --callbacks.validation.every_steps 0 \
        > "/mnt/pr1630_pack_8x_b3_${label}_d016_node${NODE_RANK}.log" 2>&1
    echo "PACK_B3_GATE completed=$label node_rank=$NODE_RANK" >&2
}

run_variant control_a false
run_variant candidate true
run_variant control_b false
