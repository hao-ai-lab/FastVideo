#!/usr/bin/env bash
# Canonical Slurm CI selection for the self-forcing lane.
set -euo pipefail

export WANDB_MODE=offline
exec pytest ./fastvideo/tests/training/self-forcing/test_self_forcing.py -vs
