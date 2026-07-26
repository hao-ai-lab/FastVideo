#!/usr/bin/env bash
set -euo pipefail

checkout=/mnt/fv-pr1630-pack-final
expected_head=fa47ce1ab570d33bb245a49f4cd63267282b2a54
expected_base_hash=bf0861ff481c499fd65350d2f0f16c86487db88cb03c332e4be5c76540fcc7fd
expected_bucket_hash=0392370dbc0aec3bedc4b600184cd36cdd5e7c60043c19433f262e085a2d08ad

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
    "$expected_base_hash:/mnt/benchmark_fastvideo_train_pack_d016.py" \
    "$expected_bucket_hash:/mnt/benchmark_fastvideo_train_fsdp_buckets_fa47ce1.py"; do
    expected=${spec%%:*}
    path=${spec#*:}
    actual=$(sha256sum "$path" | awk '{print $1}')
    if [[ "$actual" != "$expected" ]]; then
        echo "unexpected benchmark harness hash for $path: $actual" >&2
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
    local blocks_per_group=$2
    local log=/mnt/pr1630_fsdp_buckets_${label}_fa47ce1.log
    echo "FSDP_BUCKET_GATE starting=$label blocks_per_group=$blocks_per_group" >&2
    set +e
    /mnt/FastVideo/.venv/bin/torchrun \
        --standalone \
        --nproc-per-node 4 \
        /mnt/benchmark_fastvideo_train_fsdp_buckets_fa47ce1.py \
        --fsdp-blocks-per-group "$blocks_per_group" \
        "${common_args[@]}" \
        --training.checkpoint.output_dir "/mnt/pr1630_fsdp_buckets_${label}_fa47ce1" \
        > "$log" 2>&1
    local status=$?
    set -e
    echo "FSDP_BUCKET_GATE completed=$label status=$status" >&2
    return "$status"
}

# Bracket both practical candidates with the unchanged source topology.
run_variant control_a 1
run_variant group2 2
run_variant group4 4
run_variant control_b 1

# Root-only is a useful boundary but may OOM from a full-model gather/grad
# staging buffer. Run it after the bracket so failure cannot invalidate it.
root_status=skipped
if [[ ${RUN_ROOT_ONLY:-1} == 1 ]]; then
    if run_variant root_only root; then
        root_status=0
    else
        root_status=$?
    fi
fi
echo "FSDP_BUCKET_GATE root_status=$root_status" >&2

for label in control_a group2 group4 control_b root_only; do
    log=/mnt/pr1630_fsdp_buckets_${label}_fa47ce1.log
    if [[ -f "$log" ]]; then
        grep -Eh '^(BF16_BUCKET_PLAN|BF16_RESULT) ' "$log" || true
    fi
done

for label in control_a group2 group4 control_b root_only; do
    log=/mnt/pr1630_fsdp_buckets_${label}_fa47ce1.log
    if [[ -f "$log" ]]; then
        sha256sum "$log"
    fi
done
