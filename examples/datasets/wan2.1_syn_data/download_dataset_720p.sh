#!/bin/bash

# 720P dataset
python scripts/huggingface/download_hf.py --repo_id "FastVideo/Wan-Syn_77x768x1280_250k" --local_dir "FastVideo/Wan-Syn_77x768x1280_250k" --repo_type "dataset"
