#!/usr/bin/env bash
# Canonical Slurm CI selection for the LoRA-extraction lane.
set -euo pipefail

exec pytest ./fastvideo/tests/lora_extraction/test_lora_extraction.py -vs
