#!/usr/bin/env bash
set -e

CONDA_ENV=${1:-""}
if [ -n "$CONDA_ENV" ]; then
    # This is required to activate conda environment
    eval "$(conda shell.bash hook)"

    conda create -n $CONDA_ENV python=3.10.0 -y
    conda activate $CONDA_ENV
    # This is optional if you prefer to use built-in nvcc
    conda install -c nvidia cuda-toolkit -y
else
    echo "Skipping conda environment creation. Make sure you have the correct environment activated."
fi

# init a raw torch to avoid installation errors.
pip3 install torch==2.5.0 torchvision  --index-url https://download.pytorch.org/whl/cu121

# clone latest diffusers code
pip install git+https://github.com/huggingface/diffusers.git@76b7d86a9a5c0c2186efa09c4a67b5f5666ac9e3

# build flash attention
pip install packaging ninja && pip install flash-attn==2.7.0.post2 --no-build-isolation 

# install fastmochi
pip install -e .

# install train package
pip install -e ".[train]"
