#!/usr/bin/env bash
# Canonical Slurm CI selection for the VMoBA-inference lane.
set -euo pipefail

exec python fastvideo/tests/inference/vmoba/test_vmoba_inference.py
