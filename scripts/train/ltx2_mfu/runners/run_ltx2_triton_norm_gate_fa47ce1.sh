#!/usr/bin/env bash
set -euo pipefail

CUDA_VISIBLE_DEVICES=0 /mnt/FastVideo/.venv/bin/python \
    /mnt/benchmark_ltx2_triton_norm_gate_fa47ce1.py \
    2>&1 | tee /mnt/pr1630_ltx2_triton_norm_gate.log
sha256sum /mnt/pr1630_ltx2_triton_norm_gate.log
