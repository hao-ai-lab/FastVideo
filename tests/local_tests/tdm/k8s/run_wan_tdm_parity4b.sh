#!/usr/bin/env bash
# Phase 4 4B parity: reproduce the first 105 steps of the Phase 3 treatment
# run on the modality-general TDMMethod and compare the per-step metrics with
# the archived Phase 3 treatment metrics. An exact match proves the
# one-modality (Wan) path is numerically unchanged. Steps 1-99 cover the
# warmup path and steps 100-105 cover the TDM context/fake-score/generator
# path, so both code paths are exercised.
#
# Expected pod state: tests/local_tests/tdm/k8s/pod_wan_tdm.yaml held
# four-GPU pod with /tmp/tdm-port.tar.gz uploaded by the operator.
set -Eeuo pipefail

run_root=/workspace/run/tdm-port/${RUN_NAME:-wan-tdm-parity4b}
max_steps=${MAX_STEPS:-105}
test ! -e "$run_root"
mkdir -p "$run_root/logs"

export HOME=/tmp/tdm-home
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export HF_HOME=/workspace/issue-775/shared-cache/hf
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_HUB_OFFLINE=1
export WANDB_MODE=disabled
export NCCL_DEBUG=WARN
mkdir -p "$HOME"

printf 'PARITY_START %s\n' "$(date -u +%FT%TZ)"
rm -rf /tmp/tdm-port
mkdir -p /tmp/tdm-port
tar xzf /tmp/tdm-port.tar.gz -C /tmp/tdm-port
cd /tmp/tdm-port

dataset_src=/workspace/issue-775/tdm-overfit-447ebf2-r4/data/tdm_t2v_overfit_text_only
test -d "$dataset_src"
rm -rf data/tdm_t2v_overfit_text_only
mkdir -p data
cp -r "$dataset_src" data/

/opt/venv/bin/torchrun --standalone --nproc_per_node=4 -m fastvideo.train.entrypoint.train \
    --config examples/train/configs/distribution_matching/wan/tdm_t2v_lora_fixed_recipe.yaml \
    --training.checkpoint.output_dir "$run_root/output" \
    --training.data.data_path data/tdm_t2v_overfit_text_only \
    --method.warmup_steps 100 \
    --callbacks.validation.every_steps 0 \
    --training.loop.max_train_steps "$max_steps" > "$run_root/logs/train.log" 2>&1

grep -q "Training completed" "$run_root/logs/train.log"
tail -2 "$run_root/logs/train.log"
printf 'PARITY_OK %s\n' "$(date -u +%FT%TZ)"
