#!/usr/bin/env bash
set -euo pipefail

checkout=/mnt/fv-pr1630-fsdp-group-source
expected_head=20c36acefc97e8b743f79a5c52883561853a7d85
expected_harness_hash=534f201aadfc3cb02df2ad130a1193c01ebfa97ddcfd38fd4ee05a5a1e4965a5
harness=/mnt/benchmark_fastvideo_train_pack_b3.py
topology=fallback_nvls0_mnnvl0_nosymm

test "$(git -C "$checkout" rev-parse HEAD)" = "$expected_head"
test -z "$(git -C "$checkout" status --porcelain --untracked-files=no)"
test "$(sha256sum "$harness" | awk '{print $1}')" = "$expected_harness_hash"
cd "$checkout"

export PYTHONPATH="/mnt:$checkout"
export OMP_NUM_THREADS=1
export FASTVIDEO_FA4=1
export FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED=1
export FLASH_ATTENTION_CUTE_DSL_CACHE_DIR=/mnt/fa4-cache
export FASTVIDEO_BENCH_LOCAL_BATCH_SIZE=3
export GLOO_SOCKET_IFNAME=enP5p9s0
export NCCL_SOCKET_IFNAME=enP5p9s0
export NCCL_NVLS_ENABLE=0
export NCCL_MNNVL_ENABLE=0
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
    --training.distributed.num_gpus 8
    --training.distributed.tp_size 1
    --training.distributed.sp_size 1
    --training.distributed.hsdp_replicate_dim 1
    --training.distributed.hsdp_shard_dim 8
    --training.distributed.reshard_after_forward false
    --training.distributed.fsdp_symmetric_memory false
    --training.distributed.reduce_dtype bf16
    --training.data.train_batch_size 3
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
    local modules_per_group=$2
    local stem=pr1630_fsdp_group_8x_b3_${topology}_${label}_20c36
    local log=/mnt/${stem}_node${NODE_RANK}.log
    echo "FSDP_GROUP_8X_B3 topology=$topology starting=$label node_rank=$NODE_RANK modules_per_group=$modules_per_group"
    timeout 1200s /mnt/FastVideo/.venv/bin/torchrun \
        --nnodes "$NNODES" \
        --node-rank "$NODE_RANK" \
        --nproc-per-node 4 \
        --master-addr "$MASTER_ADDR" \
        --master-port "$MASTER_PORT" \
        "$harness" \
        "${common_args[@]}" \
        --training.distributed.fsdp_modules_per_group "$modules_per_group" \
        --training.checkpoint.output_dir "/mnt/${stem}" \
        > "$log" 2>&1
    echo "FSDP_GROUP_8X_B3 topology=$topology completed=$label node_rank=$NODE_RANK"
    if [[ "$NODE_RANK" == 0 ]]; then
        grep -E '^(BF16_RESULT|BF16_SEMANTICS) ' "$log"
    fi
    sha256sum "$log"
}

run_variant control_a 1
run_variant group2 2
run_variant control_b 1
