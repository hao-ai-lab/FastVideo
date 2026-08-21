#!/usr/bin/env bash
# Shared kernel-lane test selection: executed by BOTH the Modal launcher
# (fastvideo/tests/modal/pr_test.py::run_kernel_tests) and the Slurm CI
# runner, so the selections cannot drift.
set -euo pipefail

exec pytest fastvideo-kernel/tests/ -vs
