#!/bin/bash

# 1. Install missing dependency
pip install -q opencv-python-headless

# 2. Run FVD script
python /workspace/FastVideo/benchmarks/fvd/run_fvd.py