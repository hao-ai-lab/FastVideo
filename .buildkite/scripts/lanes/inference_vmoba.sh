#!/usr/bin/env bash
# Shared VMoBA inference-lane selection: executed by BOTH the Modal launcher
# (fastvideo/tests/modal/pr_test.py::run_inference_tests_vmoba) and the
# self-hosted CI runner, so the selections cannot drift.
set -euo pipefail

exec python fastvideo/tests/inference/vmoba/test_vmoba_inference.py
