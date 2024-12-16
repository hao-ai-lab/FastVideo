#!/bin/bash

# install torch
pip install torch==2.5.0 torchvision --index-url https://download.pytorch.org/whl/cu121

# install FA2 and diffusers
pip install packaging ninja && pip install flash-attn==2.7.0.post2 --no-build-isolation 
pip install "git+https://github.com/huggingface/diffusers.git@bf64b32652a63a1865a0528a73a13652b201698b"

# install fastvideo
pip install -e .
