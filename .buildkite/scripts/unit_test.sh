#!/usr/bin/env bash
set -euo pipefail

exec pytest \
  ./fastvideo/tests/api/ \
  ./fastvideo/tests/contract/ \
  ./fastvideo/tests/dataset/ \
  ./fastvideo/tests/workflow/ \
  ./fastvideo/tests/entrypoints/ \
  ./fastvideo/tests/loader/ \
  ./fastvideo/tests/pipelines/ \
  ./fastvideo/tests/platforms/ \
  ./fastvideo/tests/train/ \
  ./fastvideo/tests/stages/ \
  ./fastvideo/tests/ops/ \
  ./fastvideo/tests/worker/ \
  ./fastvideo/tests/training/test_trackers.py \
  ./fastvideo/tests/attention/test_sdpa_metadata_mask_contract.py \
  ./fastvideo/tests/modal/test_kernel_build_cache.py \
  ./fastvideo/tests/modal/test_pr_test.py \
  ./fastvideo/tests/modal/test_ssim_test.py \
  --ignore=./fastvideo/tests/entrypoints/test_openai_api_integration.py \
  --ignore=./fastvideo/tests/train/models \
  --ignore=./fastvideo/tests/train/methods \
  -vs
