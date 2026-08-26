#!/usr/bin/env bash
# Canonical Slurm CI selection for the legacy LoRA-training lane.
set -euo pipefail

export WANDB_MODE=offline
exec pytest ./fastvideo/tests/training/lora/test_lora_training.py -srP
