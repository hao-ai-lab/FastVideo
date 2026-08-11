#!/bin/bash
# Wan-Syn 720P dataset (77x768x1280, 250k clips).
python scripts/huggingface/download_hf.py --repo_id "FastVideo/Wan-Syn_77x768x1280_250k" --local_dir "data/Wan-Syn_77x768x1280_250k" --repo_type "dataset"
