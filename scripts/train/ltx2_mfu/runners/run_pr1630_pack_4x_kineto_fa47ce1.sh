#!/usr/bin/env bash
set -euo pipefail

checkout=/mnt/fv-pr1630-fsdp-group-source
expected_head=20c36acefc97e8b743f79a5c52883561853a7d85
expected_base_hash=bf0861ff481c499fd65350d2f0f16c86487db88cb03c332e4be5c76540fcc7fd
expected_profiler_hash=0e5785a4c3b6bc8490facea0313c07e751830ee93a9130939595efd5dd690ab1

actual_head=$(git -C "$checkout" rev-parse HEAD)
if [[ "$actual_head" != "$expected_head" ]]; then
    echo "unexpected checkout head: $actual_head" >&2
    exit 1
fi
if [[ -n $(git -C "$checkout" status --porcelain --untracked-files=no) ]]; then
    echo "profiler checkout is dirty" >&2
    exit 1
fi
actual_base_hash=$(sha256sum /mnt/benchmark_fastvideo_train_pack_d016.py | awk '{print $1}')
actual_profiler_hash=$(sha256sum /mnt/profile_fastvideo_train_pack_fa47ce1.py | awk '{print $1}')
if [[ "$actual_base_hash" != "$expected_base_hash" ]]; then
    echo "unexpected base harness hash: $actual_base_hash" >&2
    exit 1
fi
if [[ "$actual_profiler_hash" != "$expected_profiler_hash" ]]; then
    echo "unexpected profiler harness hash: $actual_profiler_hash" >&2
    exit 1
fi

export PYTHONPATH="/mnt:$checkout"
export OMP_NUM_THREADS=1
export FASTVIDEO_FA4=1
export FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED=1
export FLASH_ATTENTION_CUTE_DSL_CACHE_DIR=/mnt/fa4-cache
export GLOO_SOCKET_IFNAME=enP5p9s0
export NCCL_CTA_POLICY=2
export WANDB_MODE=disabled
export FASTVIDEO_KINETO_PREFIX=/mnt/pr1630_group2_b2_4x_20c36_rank0
unset FASTVIDEO_TE_FP32_MASTER || true
unset TORCHINDUCTOR_MAX_AUTOTUNE_GEMM || true
unset TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_BACKENDS || true
unset TORCHINDUCTOR_CUTLASS_DIR || true

/mnt/FastVideo/.venv/bin/torchrun \
    --standalone \
    --nproc-per-node 4 \
    /mnt/profile_fastvideo_train_pack_fa47ce1.py \
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
    --training.distributed.fsdp_modules_per_group 2 \
    --training.distributed.reduce_dtype bf16 \
    --training.data.train_batch_size 2 \
    --training.data.data_path /mnt/FastVideo/data/ltx2_overfit_preprocessed:256 \
    --training.data.dataloader_num_workers 1 \
    --training.loop.gradient_accumulation_steps 1 \
    --training.loop.max_train_steps 14 \
    --training.checkpoint.output_dir /mnt/pr1630_group2_b2_4x_kineto_20c36 \
    --pipeline.dit_config.pack_attention_projections true \
    --callbacks.grad_clip.log_grad_norms true \
    --callbacks.validation.every_steps 0 \
    > /mnt/pr1630_group2_b2_4x_kineto_20c36.log 2>&1

sha256sum \
    /mnt/pr1630_group2_b2_4x_20c36_rank0.trace.json.gz \
    /mnt/pr1630_group2_b2_4x_20c36_rank0.summary.json \
    /mnt/pr1630_group2_b2_4x_kineto_20c36.log
