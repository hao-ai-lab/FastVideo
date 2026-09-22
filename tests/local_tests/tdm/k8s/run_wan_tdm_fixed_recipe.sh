#!/usr/bin/env bash
# Phase 3: Wan 1.3B fixed-recipe overfit in the modular TDM stack.
#
# Two runs on the same held four-GPU node, identical except that the
# control overrides method.warmup_steps to cover the whole run:
#
#   treatment  warmup 100 (8-step ladder stage) then TDM 100 (4-step)
#   control    warmup 200 (pure regression across the same ladder)
#
# Acceptance is the reference-free video report
# (tests/local_tests/tdm/tools/tdm_video_report.py): frame sharpness and
# contrast of the validation samples. Paired metrics are forensic only
# and are not asserted.
#
# Expected pod state: tests/local_tests/tdm/k8s/pod_wan_tdm.yaml held
# four-GPU pod with /tmp/tdm-port.tar.gz uploaded by the operator.
set -Eeuo pipefail

run_name=${RUN_NAME:-wan-tdm-fixed-recipe}
run_root=/workspace/run/tdm-port/$run_name
test ! -e "$run_root"
mkdir -p "$run_root/logs"

record_status() {
    rc=$?
    printf 'completed_at=%s\nrc=%s\n' "$(date -u +%FT%TZ)" "$rc" > "$run_root/status.tmp"
    mv "$run_root/status.tmp" "$run_root/status"
}
trap record_status EXIT
exec > >(tee -a "$run_root/logs/stage.log") 2>&1

export HOME=/tmp/tdm-home
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export HF_HOME=/workspace/issue-775/shared-cache/hf
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_HUB_OFFLINE=1
export WANDB_MODE=disabled
export NCCL_DEBUG=WARN
mkdir -p "$HOME"

printf 'START %s %s\n' "$run_name" "$(date -u +%FT%TZ)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv
printf 'CODE_SHA256 %s\n' "$(sha256sum /tmp/tdm-port.tar.gz | cut -d' ' -f1)"

rm -rf /tmp/tdm-port
tar xzf /tmp/tdm-port.tar.gz -C /tmp
cd /tmp/tdm-port

dataset_src=/workspace/issue-775/tdm-overfit-447ebf2-r4/data/tdm_t2v_overfit_text_only
test -d "$dataset_src"
rm -rf data/tdm_t2v_overfit_text_only
mkdir -p data
cp -r "$dataset_src" data/
for shard in 0 1 2 3; do test -d "data/tdm_t2v_overfit_text_only/shard-$shard"; done
printf 'DATASET_OK %s\n' "$(date -u +%FT%TZ)"

config=examples/train/configs/distribution_matching/wan/tdm_t2v_lora_fixed_recipe.yaml

run_one() {
    local label=$1 warmup=$2
    local out="$run_root/$label/output"
    printf 'TRAIN %s %s warmup=%s\n' "$label" "$(date -u +%FT%TZ)" "$warmup"
    /opt/venv/bin/torchrun --standalone --nproc_per_node=4 -m fastvideo.train.entrypoint.train \
        --config "$config" \
        --training.checkpoint.output_dir "$out" \
        --method.warmup_steps "$warmup" > "$run_root/logs/train-$label.log" 2>&1
    grep -q "Training completed" "$run_root/logs/train-$label.log"
    tail -3 "$run_root/logs/train-$label.log"
}

run_one treatment 100
run_one control 200

printf 'REPORT %s\n' "$(date -u +%FT%TZ)"
for label in treatment control; do
    PYTHONPATH=/tmp/tdm-port /opt/venv/bin/python tests/local_tests/tdm/tools/tdm_video_report.py \
        --run-dir "$run_root/$label/output" \
        --out "$run_root/$label/video_report.json" > "$run_root/logs/video-report-$label.log" 2>&1
    grep -E "step=|VIDEO_REPORT" "$run_root/logs/video-report-$label.log" | tail -8
done

printf 'STAGE_OK %s\n' "$(date -u +%FT%TZ)"
