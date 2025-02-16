import torch
from stepvideo.config import parse_args
import os

try:
    args = parse_args()
    torch.ops.load_library(os.path.join(args.model_dir, 'lib/liboptimus_ths-torch2.3-cu121.cpython-310-x86_64-linux-gnu.so'))
except Exception as err:
    print(err)
