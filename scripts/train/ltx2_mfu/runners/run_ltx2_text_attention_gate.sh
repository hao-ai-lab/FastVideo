#!/usr/bin/env bash
set -euo pipefail
if [[ "${NODE_RANK:-0}" != 0 ]]; then
  exit 0
fi
cd /mnt/fv-pr1630-pack-final
source /mnt/FastVideo/.venv/bin/activate
export FASTVIDEO_FA4=1
export CUDA_VISIBLE_DEVICES=0
python /mnt/benchmark_ltx2_text_attention_backends.py --batch 1 --warmup 8 --repeats 31
python /mnt/benchmark_ltx2_text_attention_backends.py --batch 3 --warmup 8 --repeats 31
