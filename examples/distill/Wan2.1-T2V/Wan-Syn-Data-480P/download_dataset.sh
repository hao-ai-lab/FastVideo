#!/bin/bash
mkdir -p data
python scripts/huggingface/download_hf.py --repo_id "FastVideo/Wan-Syn_77x448x832_600k" --local_dir "data/Wan-Syn_77x448x832_600k" --repo_type "dataset"
