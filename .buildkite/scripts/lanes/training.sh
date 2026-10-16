#!/usr/bin/env bash
# Canonical Slurm CI selection for the legacy vanilla-training lane.
set -euo pipefail

export WANDB_MODE=offline
# Ring/USP parity needs four GPUs and runs in a separate process before training.
pytest ./fastvideo/tests/distributed/test_ring_attention.py -srP
exec pytest ./fastvideo/tests/training/Vanilla -srP
