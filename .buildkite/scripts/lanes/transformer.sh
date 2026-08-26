#!/usr/bin/env bash
# Canonical Slurm CI selection for the transformer lane.
set -euo pipefail

exec pytest ./fastvideo/tests/transformers ./fastvideo/tests/distributed/test_minimax_h3_packed_sp.py -vs
