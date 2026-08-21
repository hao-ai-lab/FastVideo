#!/usr/bin/env bash
# Shared train_framework-lane test selection: executed by BOTH the Modal launcher
# (fastvideo/tests/modal/pr_test.py::run_train_framework_tests) and the Slurm CI
# runner, so the selections cannot drift.
set -euo pipefail

exec pytest ./fastvideo/tests/train/models ./fastvideo/tests/train/methods -vs
