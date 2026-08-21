#!/usr/bin/env bash
# Shared distillation_dmd-lane test selection: executed by BOTH the Modal launcher
# (fastvideo/tests/modal/pr_test.py::run_distill_dmd_tests) and the self-hosted
# CI runner, so the selections cannot drift.
set -euo pipefail

exec pytest ./fastvideo/tests/training/distill/test_distill_dmd.py -vs
