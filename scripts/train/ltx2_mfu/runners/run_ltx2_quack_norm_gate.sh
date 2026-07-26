#!/usr/bin/env bash
set -euo pipefail

node_rank=${SLURM_NODEID:-${NODE_RANK:-0}}
CUDA_VISIBLE_DEVICES=0 /mnt/FastVideo/.venv/bin/python \
    /mnt/benchmark_ltx2_quack_norm_gate_fa47ce1.py \
    > "/mnt/pr1630_ltx2_quack_norm_gate_node${node_rank}.log" 2>&1

sha256sum "/mnt/pr1630_ltx2_quack_norm_gate_node${node_rank}.log"
