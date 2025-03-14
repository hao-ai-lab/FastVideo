import os

import torch

from fastvideo.models.stepvideo.config import parse_args

try:
    args = parse_args()
    torch.ops.load_library(
        os.path.join(args.model_dir, 'lib/liboptimus_ths-torch2.5-cu124.cpython-310-x86_64-linux-gnu.so'))
except Exception as err:
    print(err)
