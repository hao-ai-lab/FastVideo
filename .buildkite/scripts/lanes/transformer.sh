#!/usr/bin/env bash
# Shared transformer-lane test selection: executed by BOTH the Modal launcher
# (fastvideo/tests/modal/pr_test.py::run_transformer_tests) and the self-hosted
# CI runner, so the selections cannot drift.
set -euo pipefail

exec pytest ./fastvideo/tests/transformers -vs
