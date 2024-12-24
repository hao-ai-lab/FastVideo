#!/bin/bash

# install torch
pip install torch==2.5.0 torchvision --index-url https://download.pytorch.org/whl/cu121

# install FA2 and diffusers
pip install packaging ninja && pip install flash-attn==2.7.0.post2 --no-build-isolation 

# install fastvideo
pip install -e .
