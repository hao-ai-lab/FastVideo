#!/usr/bin/env bash
# Shared vae-lane test selection: executed by BOTH the Modal launcher
# (fastvideo/tests/modal/pr_test.py::run_vae_tests) and the Slurm CI
# runner, so the selections cannot drift.
set -euo pipefail

exec pytest ./fastvideo/tests/vaes -vs
