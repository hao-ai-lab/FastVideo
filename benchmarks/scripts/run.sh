#!/bin/bash

# 1. Install missing dependency
pip install -q opencv-python-headless transformers huggingface_hub

# 2. Run FVD script
python benchmarks/fvd/run_fvd.py
