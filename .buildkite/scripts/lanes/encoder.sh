#!/usr/bin/env bash
# Canonical Slurm CI selection for the encoder lane.
set -euo pipefail

exec pytest ./fastvideo/tests/encoders -vs
