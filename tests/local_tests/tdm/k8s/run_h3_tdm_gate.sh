#!/usr/bin/env bash
# Phase 4 4D: MiniMax H3 joint video+audio TDM gate on one four-GPU node.
#
# Runs the shipped `overfit_minimax_h3_t2va_tdm.yaml` config (200 steps,
# warmup 50, 8->4 ladder, validation every 25) and produces the reference-free
# video report. The dataset is the 4-row replica of the single H3 T2VA asset
# at /workspace/issue-775/h3_overfit_t2va_x4 (one row per data-parallel rank).
#
# Expected pod state: tests/local_tests/tdm/k8s/pod_wan_tdm.yaml held
# four-GPU pod with /tmp/tdm-port.tar.gz uploaded by the operator.
set -Eeuo pipefail

run_name=${RUN_NAME:-h3-tdm-gate}
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
mkdir -p /tmp/tdm-port
tar xzf /tmp/tdm-port.tar.gz -C /tmp/tdm-port
cd /tmp/tdm-port

test -f /workspace/issue-775/h3_overfit_t2va_x4/data_00000.parquet

config=examples/train/configs/distribution_matching/overfit_minimax_h3_t2va_tdm.yaml
out="$run_root/output"

printf 'TRAIN %s\n' "$(date -u +%FT%TZ)"
/opt/venv/bin/torchrun --standalone --nproc_per_node=4 -m fastvideo.train.entrypoint.train \
    --config "$config" \
    --training.checkpoint.output_dir "$out" > "$run_root/logs/train.log" 2>&1

grep -q "Training completed" "$run_root/logs/train.log"
tail -3 "$run_root/logs/train.log"

printf 'REPORT %s\n' "$(date -u +%FT%TZ)"
PYTHONPATH=/tmp/tdm-port /opt/venv/bin/python tests/local_tests/tdm/tools/tdm_video_report.py \
    --run-dir "$out" \
    --out "$run_root/video_report.json" > "$run_root/logs/video-report.log" 2>&1
grep -E "step=" "$run_root/logs/video-report.log" | tail -6

printf 'GATE_OK %s\n' "$(date -u +%FT%TZ)"
