#!/usr/bin/env bash
set -euo pipefail

# Collect the whole attention directory so new files cannot land uncovered.
# Its FA2/FA3 regression files skip when FA4 is selected (the Modal image
# enables FA4 by default), so pin FA4 off for the directory to be real
# coverage on every runner rather than a nominal collection.
export FASTVIDEO_FA4=0

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
  ./fastvideo/tests/inference/test_basic_fasth3_omniref_pdd.py \
  ./fastvideo/tests/attention/ \
  ./fastvideo/tests/layers/test_pdd_linear.py \
  ./fastvideo/tests/modal/test_kernel_build_cache.py \
  ./fastvideo/tests/modal/test_pr_test.py \
  ./fastvideo/tests/modal/test_ssim_test.py \
  --ignore=./fastvideo/tests/entrypoints/test_openai_api_integration.py \
  --ignore=./fastvideo/tests/train/models \
  --ignore=./fastvideo/tests/train/methods \
  -vs
