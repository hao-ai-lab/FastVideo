#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
checkout=${CHECKOUT:-$(cd -- "$script_dir/../../../.." && pwd)}
config=${CONFIG:-$checkout/examples/train/configs/overfit_ltx2_t2v.yaml}
harness=$script_dir/../harness/benchmark_fastvideo_train_pack_d016.py
torchrun=${TORCHRUN:-$checkout/.venv/bin/torchrun}

nnodes=${NNODES:-1}
node_rank=${NODE_RANK:-0}
nproc_per_node=${NPROC_PER_NODE:-4}
world_size=$((nnodes * nproc_per_node))
local_batch_size=${LOCAL_BATCH_SIZE:-2}
grad_accum_steps=${GRAD_ACCUM_STEPS:-1}
data_path=${DATA_PATH:-$checkout/data/ltx2_overfit_preprocessed:256}
output_dir=${OUTPUT_DIR:-/tmp/fastvideo-ltx2-mfu-${world_size}x-b${local_batch_size}}

source_head=$(git -C "$checkout" rev-parse HEAD)
tracked_status=$(git -C "$checkout" status --porcelain --untracked-files=no)
tracked_dirty=false
if [[ -n "$tracked_status" ]]; then
    tracked_dirty=true
    if [[ "${ALLOW_DIRTY:-0}" != 1 ]]; then
        echo "refusing to benchmark a tracked-dirty checkout; set ALLOW_DIRTY=1 to override" >&2
        exit 2
    fi
fi
printf 'BF16_SOURCE {"git_sha":"%s","tracked_dirty":%s}\n' "$source_head" "$tracked_dirty"

if [[ "$world_size" != 4 && "$world_size" != 8 ]]; then
    echo "run_current.sh is calibrated for 4 or 8 GPUs; got $world_size" >&2
    exit 2
fi
if [[ "$nnodes" -gt 1 && (-z "${MASTER_ADDR:-}" || -z "${MASTER_PORT:-}") ]]; then
    echo "MASTER_ADDR and MASTER_PORT are required when NNODES > 1" >&2
    exit 2
fi

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export FASTVIDEO_FA4=${FASTVIDEO_FA4:-1}
export FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED=${FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED:-1}
export FLASH_ATTENTION_CUTE_DSL_CACHE_DIR=${FLASH_ATTENTION_CUTE_DSL_CACHE_DIR:-/tmp/fastvideo-fa4-cache}
export FASTVIDEO_BENCH_LOCAL_BATCH_SIZE=$local_batch_size
export FASTVIDEO_BENCH_GRAD_ACCUM_STEPS=$grad_accum_steps
export NCCL_CTA_POLICY=${NCCL_CTA_POLICY:-2}
export WANDB_MODE=disabled

launcher=("$torchrun")
if [[ "$nnodes" == 1 ]]; then
    launcher+=(--standalone --nproc-per-node "$nproc_per_node")
else
    launcher+=(
        --nnodes "$nnodes"
        --node-rank "$node_rank"
        --nproc-per-node "$nproc_per_node"
        --master-addr "$MASTER_ADDR"
        --master-port "$MASTER_PORT"
    )
fi

"${launcher[@]}" "$harness" \
    --singleton-timestep \
    --config "$config" \
    --models.student.attention_backend FLASH_ATTN \
    --models.student.enable_gradient_checkpointing_type null \
    --training.model.enable_gradient_checkpointing_type null \
    --training.model.enable_torch_compile true \
    --training.dit_precision fp32 \
    --training.optimizer.fused true \
    --training.distributed.num_gpus "$world_size" \
    --training.distributed.tp_size 1 \
    --training.distributed.sp_size 1 \
    --training.distributed.hsdp_replicate_dim 1 \
    --training.distributed.hsdp_shard_dim "$world_size" \
    --training.distributed.reshard_after_forward false \
    --training.distributed.fsdp_symmetric_memory true \
    --training.distributed.fsdp_modules_per_group 2 \
    --training.distributed.reduce_dtype bf16 \
    --training.data.train_batch_size "$local_batch_size" \
    --training.data.data_path "$data_path" \
    --training.data.dataloader_num_workers 1 \
    --training.loop.gradient_accumulation_steps "$grad_accum_steps" \
    --training.loop.max_train_steps 30 \
    --training.checkpoint.output_dir "$output_dir" \
    --pipeline.dit_config.pack_attention_projections true \
    --callbacks.grad_clip.log_grad_norms true \
    --callbacks.validation.every_steps 0
