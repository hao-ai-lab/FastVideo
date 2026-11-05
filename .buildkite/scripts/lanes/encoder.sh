#!/usr/bin/env bash
# Canonical Slurm CI selection for the encoder lane.
set -euo pipefail

# The unit lane excludes GPU-marked stage tests; run the tiny CUDA fixture here.
pytest ./fastvideo/tests/stages/test_minimax_h3_text_conditioning_cache.py::test_native_qwen_conditioner_cuda_reuse_and_reload -vs
exec pytest ./fastvideo/tests/encoders -vs
