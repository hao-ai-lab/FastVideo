#!/usr/bin/env bash
# Canonical Slurm CI selection for the OpenAI-compatible API lane.
set -euo pipefail

exec pytest ./fastvideo/tests/entrypoints/test_openai_api_integration.py -vs
