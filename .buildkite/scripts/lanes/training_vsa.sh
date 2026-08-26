#!/usr/bin/env bash
# Canonical Slurm CI selection for the legacy VSA-training lane.
set -euo pipefail

export WANDB_MODE=offline
exec pytest ./fastvideo/tests/training/VSA -srP
