#!/usr/bin/env bash
# Shared golden-gate selection: executed by BOTH the Modal launcher
# (fastvideo/tests/modal/pr_test.py::run_golden_gate_tests) and the Slurm CI
# runner, so the selections cannot drift. Environment (HF_HOME, auth) is the
# backend's responsibility, not this script's.
set -euo pipefail

exec pytest ./fastvideo/tests/golden_gate -vs
