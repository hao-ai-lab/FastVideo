#!/usr/bin/env bash
# Shared encoder-lane test selection: executed by BOTH the Modal launcher
# (fastvideo/tests/modal/pr_test.py::run_encoder_tests) and the self-hosted
# CI runner, so the selections cannot drift.
set -euo pipefail

exec pytest ./fastvideo/tests/encoders -vs
