#!/usr/bin/env bash
set -euo pipefail

checkout=/mnt/fv-pr1630-pack-final
expected_head=fa47ce1ab570d33bb245a49f4cd63267282b2a54
expected_control_hash=bf0861ff481c499fd65350d2f0f16c86487db88cb03c332e4be5c76540fcc7fd
expected_candidate_hash=12ccb47c662016184e9959dc9f20d7163bc06f4e3382b9f4d3d32acacc7930b0

node_rank=${NODE_RANK:-${SLURM_NODEID:-}}
if [[ ! "$node_rank" =~ ^[01]$ ]]; then
    echo "expected NODE_RANK or SLURM_NODEID to be 0 or 1, got: ${node_rank:-unset}" >&2
    exit 1
fi
if [[ -n ${NODE_RANK:-} && -n ${SLURM_NODEID:-} && "$NODE_RANK" != "$SLURM_NODEID" ]]; then
    echo "NODE_RANK=$NODE_RANK differs from SLURM_NODEID=$SLURM_NODEID" >&2
    exit 1
fi
suffix=node${node_rank}

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
    "$expected_candidate_hash:/mnt/benchmark_fastvideo_train_fused_clip_fa47ce1.py"; do
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

parity_log="/mnt/pr1630_fused_clip_parity_fa47ce1_${suffix}.log"
/mnt/FastVideo/.venv/bin/python \
    /mnt/benchmark_fastvideo_train_fused_clip_fa47ce1.py \
    --parity-only \
    > "$parity_log" 2>&1

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
    local harness=$2
    local log="/mnt/pr1630_fused_clip_${label}_fa47ce1_${suffix}.log"
    echo "FUSED_CLIP_DUAL_GATE node=$node_rank starting=$label harness=$harness" >&2
    /mnt/FastVideo/.venv/bin/torchrun \
        --standalone \
        --nproc-per-node 4 \
        "$harness" \
        "${common_args[@]}" \
        --training.checkpoint.output_dir "/mnt/pr1630_fused_clip_${label}_fa47ce1_${suffix}" \
        > "$log" 2>&1
    echo "FUSED_CLIP_DUAL_GATE node=$node_rank completed=$label" >&2
}

run_variant control_a /mnt/benchmark_fastvideo_train_pack_d016.py
run_variant candidate /mnt/benchmark_fastvideo_train_fused_clip_fa47ce1.py
run_variant control_b /mnt/benchmark_fastvideo_train_pack_d016.py

logs=(
    "$parity_log"
    "/mnt/pr1630_fused_clip_control_a_fa47ce1_${suffix}.log"
    "/mnt/pr1630_fused_clip_candidate_fa47ce1_${suffix}.log"
    "/mnt/pr1630_fused_clip_control_b_fa47ce1_${suffix}.log"
)
grep -HE '^(FUSED_CLIP_PARITY|FUSED_CLIP_COUNTS|BF16_RESULT|BF16_SEMANTICS) ' "${logs[@]}"
sha256sum "${logs[@]}"
