#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

prompts=${1:-prompts.jsonl}
output_dir=${2:-veo_training_data}

# generate_veo_training_data.py processes records serially.
exec python3 "$(dirname -- "${BASH_SOURCE[0]}")/generate_veo_training_data.py" \
    "$prompts" \
    --output-dir "$output_dir" \
    --limit 1000
