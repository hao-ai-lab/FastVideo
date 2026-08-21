#!/usr/bin/env bash
# Shared inference_lora-lane test selection: executed by BOTH the Modal launcher
# (fastvideo/tests/modal/pr_test.py::run_inference_lora_tests) and the self-hosted
# CI runner, so the selections cannot drift.
set -euo pipefail

exec pytest ./fastvideo/tests/inference/lora/test_lora_inference_similarity.py -vs
