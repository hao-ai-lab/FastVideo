#!/usr/bin/env bash
set -euo pipefail

checkout=/mnt/fv-pr1630-pack-final
expected_head=fa47ce1ab570d33bb245a49f4cd63267282b2a54
expected_control_hash=bf0861ff481c499fd65350d2f0f16c86487db88cb03c332e4be5c76540fcc7fd
expected_base_hash=0333e44a947af0cc53850ac068c425d848f48c88d576ebebc0b289eca7be051f
expected_candidate_hash=ecdbc0a69a29b0b5f37e6f77a0a2503316168bbb4476630a8dbb5fe2bd8421cb

actual_head=$(git -C "$checkout" rev-parse HEAD)
if [[ "$actual_head" != "$expected_head" ]]; then
    echo "unexpected checkout head: $actual_head" >&2
    exit 1
fi
if [[ -n $(git -C "$checkout" status --porcelain) ]]; then
    echo "benchmark checkout is dirty" >&2
    exit 1
fi
for spec in \
    "$expected_control_hash:/mnt/benchmark_fastvideo_train_pack_d016.py" \
    "$expected_base_hash:/mnt/zero2_ltx2_input_probe.py" \
    "$expected_candidate_hash:/mnt/pr1630_fa47_fixed_arena.py"; do
    expected=${spec%%:*}
    path=${spec#*:}
    actual=$(sha256sum "$path" | awk '{print $1}')
    if [[ "$actual" != "$expected" ]]; then
        echo "unexpected harness hash for $path: $actual" >&2
        exit 1
    fi
done

export PYTHONPATH="/mnt:$checkout"
export OMP_NUM_THREADS=1
export FASTVIDEO_FA4=1
export FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED=1
export FLASH_ATTENTION_CUTE_DSL_CACHE_DIR=/mnt/fa4-cache
export GLOO_SOCKET_IFNAME=enP5p9s0
export NCCL_CTA_POLICY=2
export WANDB_MODE=disabled
unset FASTVIDEO_TE_FP32_MASTER || true
cd "$checkout"

run_control() {
    local label=$1
    echo "FIXED_ARENA_GATE starting=$label" >&2
    /mnt/FastVideo/.venv/bin/torchrun \
        --standalone \
        --nproc-per-node 4 \
        /mnt/benchmark_fastvideo_train_pack_d016.py \
        --singleton-timestep \
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
        --training.checkpoint.output_dir "/mnt/pr1630_fa47_fixed_arena_${label}" \
        --pipeline.dit_config.pack_attention_projections true \
        --callbacks.grad_clip.log_grad_norms true \
        --callbacks.validation.every_steps 0 \
        > "/mnt/pr1630_fa47_fixed_arena_${label}.log" 2>&1
    echo "FIXED_ARENA_GATE completed=$label" >&2
}

run_candidate() {
    echo "FIXED_ARENA_GATE starting=candidate" >&2
    /mnt/FastVideo/.venv/bin/torchrun \
        --standalone \
        --nproc-per-node 4 \
        /mnt/pr1630_fa47_fixed_arena.py \
        --config "$checkout/examples/train/configs/overfit_ltx2_t2v.yaml" \
        --warmup 10 \
        --steps 20 \
        --gate-ms 410.314070 \
        --models.student.attention_backend FLASH_ATTN \
        --training.distributed.num_gpus 4 \
        --training.distributed.tp_size 1 \
        --training.distributed.sp_size 1 \
        --training.data.train_batch_size 1 \
        --training.data.data_path /mnt/FastVideo/data/ltx2_overfit_preprocessed:256 \
        --training.data.dataloader_num_workers 1 \
        --pipeline.dit_config.pack_attention_projections true \
        --callbacks.validation.every_steps 0 \
        > /mnt/pr1630_fa47_fixed_arena_candidate.log 2>&1
    echo "FIXED_ARENA_GATE completed=candidate" >&2
}

if [[ ${FIXED_ARENA_RESUME_AFTER_CONTROL_A:-0} != 1 ]]; then
    run_control control_a
fi
run_candidate
run_control control_b

grep -Eh '^(BF16_RESULT|FIXED_ARENA_LAYOUT|ZERO2_RESULT) ' \
    /mnt/pr1630_fa47_fixed_arena_control_a.log \
    /mnt/pr1630_fa47_fixed_arena_candidate.log \
    /mnt/pr1630_fa47_fixed_arena_control_b.log

sha256sum \
    /mnt/pr1630_fa47_fixed_arena_control_a.log \
    /mnt/pr1630_fa47_fixed_arena_candidate.log \
    /mnt/pr1630_fa47_fixed_arena_control_b.log
