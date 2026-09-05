# SPDX-License-Identifier: Apache-2.0
"""Inference script for Cosmos Predict video generation.

Example usage:
    python examples/inference/basic/basic_cosmos_predict.py \
        --model_name nvidia/Cosmos-1.0-Prompt2World-7B-Video \
        --prompt "A cute dog walking."
"""
from fastvideo.utils.cli import inference_entry

if __name__ == "__main__":
    inference_entry()
