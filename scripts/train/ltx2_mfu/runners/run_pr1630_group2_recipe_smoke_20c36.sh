#!/usr/bin/env bash
set -euo pipefail

checkout=/mnt/fv-pr1630-fsdp-group-source
expected_head=20c36acefc97e8b743f79a5c52883561853a7d85
log=/mnt/pr1630_group2_recipe_smoke_20c36.log

test "$(git -C "$checkout" rev-parse HEAD)" = "$expected_head"
test -z "$(git -C "$checkout" status --porcelain)"
cd "$checkout"

export PYTHONPATH="$checkout"
export OMP_NUM_THREADS=1
export FASTVIDEO_FA4=1
export FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED=1
export FLASH_ATTENTION_CUTE_DSL_CACHE_DIR=/mnt/fa4-cache
export GLOO_SOCKET_IFNAME=enP5p9s0
export NCCL_CTA_POLICY=2
export WANDB_MODE=disabled
unset FASTVIDEO_FSDP2_AUTOWRAP || true

/mnt/FastVideo/.venv/bin/python -c \
    'from fastvideo.train.utils.config import load_run_config; c=load_run_config("/mnt/fv-pr1630-fsdp-group-source/examples/train/configs/overfit_ltx2_t2v.yaml"); assert c.training.distributed.fsdp_modules_per_group == 2; assert c.models["student"]["enable_gradient_checkpointing_type"] == "full"; print("GROUP2_RECIPE_CONFIG_OK")'

/mnt/FastVideo/.venv/bin/torchrun \
    --standalone \
    --nproc-per-node 4 \
    -m fastvideo.train.entrypoint.train \
    --config "$checkout/examples/train/configs/overfit_ltx2_t2v.yaml" \
    --models.student.attention_backend FLASH_ATTN \
    --training.loop.max_train_steps 2 \
    --training.data.data_path /mnt/FastVideo/data/ltx2_overfit_preprocessed:8 \
    --training.checkpoint.output_dir /mnt/pr1630_group2_recipe_smoke_20c36 \
    --training.checkpoint.training_state_checkpointing_steps 0 \
    --callbacks.validation.every_steps 0 \
    2>&1 | tee "$log"

sha256sum "$log"
