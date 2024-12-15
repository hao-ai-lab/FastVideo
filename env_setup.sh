#!/bin/bash
set -e

CONDA_ENV=${1:-""}
if [ -n "$CONDA_ENV" ]; then
    # This is required to activate conda environment
    eval "$(conda shell.bash hook)"

    conda create -n $CONDA_ENV python=3.10.0 -y
    conda activate $CONDA_ENV
else
    echo "Skipping conda environment creation. Make sure you have the correct environment activated."
fi

# install torch
pip install torch==2.5.0 torchvision --index-url https://download.pytorch.org/whl/cu121

# install FA2 and diffusers
pip install packaging ninja && pip install flash-attn==2.7.0.post2 --no-build-isolation 
pip install "git+https://github.com/huggingface/diffusers.git@bf64b32652a63a1865a0528a73a13652b201698b"

# install fastvideo
pip install -e .
