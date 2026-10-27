#!/bin/bash
# Wan-Syn 480P dataset (77x448x832, 600k clips).
python scripts/huggingface/download_hf.py --repo_id "FastVideo/Wan-Syn_77x448x832_600k" --local_dir "data/Wan-Syn_77x448x832_600k" --repo_type "dataset"
