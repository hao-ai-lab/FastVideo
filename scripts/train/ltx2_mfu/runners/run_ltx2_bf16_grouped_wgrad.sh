#!/usr/bin/env bash
set -euo pipefail

if [[ "${NODE_RANK:-0}" != 0 ]]; then
  echo "Skipping NODE_RANK=${NODE_RANK}; benchmark owns one GPU on NODE_RANK=0."
  exit 0
fi

output="/mnt/bench_ltx2_bf16_grouped_wgrad_job1623561.jsonl"
CUDA_VISIBLE_DEVICES=0 /mnt/FastVideo/.venv/bin/python \
  /mnt/bench_ltx2_bf16_grouped_wgrad.py \
  --groups 2 4 --warmup 5 --samples 15 --inner 3 | tee "${output}"
sha256sum /mnt/bench_ltx2_bf16_grouped_wgrad.py "${output}"
