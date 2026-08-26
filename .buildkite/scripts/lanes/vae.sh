#!/usr/bin/env bash
# Canonical Slurm CI selection for the VAE lane.
set -euo pipefail

exec pytest ./fastvideo/tests/vaes -vs
