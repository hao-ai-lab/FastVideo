#!/usr/bin/env bash
# Canonical Slurm CI selection for the LoRA-inference lane.
set -euo pipefail

exec pytest ./fastvideo/tests/inference/lora/test_lora_inference_similarity.py -vs
