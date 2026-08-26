#!/usr/bin/env bash
# Canonical Slurm CI selection for the custom-kernel lane.
set -euo pipefail

exec pytest fastvideo-kernel/tests/ -vs
