from contextlib import contextmanager

import torch


try:
    from flash_attn import flash_attn_varlen_func as flash_varlen_attn
except ImportError:
    flash_varlen_attn = None

try:
    from sageattention import sageattn as sage_attn
except ImportError:
    sage_attn = None

from torch.nn.attention import SDPBackend, sdpa_kernel

training_backends = [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]
eval_backends = list(training_backends)
if torch.cuda.get_device_properties(0).major >= 9.0:
    # Enable fast CuDNN attention on Hopper.
    # This gives NaN on the backward pass for some reason,
    # so only use it for evaluation.
    eval_backends.append(SDPBackend.CUDNN_ATTENTION)

@contextmanager
def sdpa_attn_ctx(training: bool = False):
    with sdpa_kernel(training_backends if training else eval_backends):
        yield
