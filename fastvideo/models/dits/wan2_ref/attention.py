# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Description:
# Single point of entry for all generic attention ops (self and cross attention), that tries to
# deliver the best performance possible given any use case (GPU and environment).
#
# On Hopper GPUs (i.e. H100, H20, H200), Flash Attention 3 is the best-performing choice, but it
# needs to be installed. When it is not available, the second best choice is cuDNN attention, which
# we get using PyTorch's SDPA API.
#
# For all other use cases, we will just use PyTorch's SDPA, but we need to specify backends and
# priorities.
# Flash Attention 2, which is one of the backends, is the best choice for Ampere GPUs (both RTX and
# datacenter-class).
#
# For anything pre-Ampere, the only choice is "memory-efficient" (xformers) FMHA.
#
# For Ada and Blackwell RTX, it is unclear at the moment, so we defer to Flash Attention 2, and
# fallbacks are cuDNN and xformers.
#
# For Blackwell datacenter-class (B200, GB200), cuDNN is the best choice.
#
#
# Dispatching to the desired backends/paths are done by checking the compute capability (really SM
# number, which is just compute capability * 10) of the GPU device the input tensors are on.
#
# Here's a breakdown of relevant compute capabilities:
#
# | GPU / category | Arch  |
# |================|=======|
# | A100           | SM80  |
# | A40            | SM80  |
# | Ampere RTX     | SM86  |
# |----------------|-------|
# | Ada Lovelace   | SM89  |
# |----------------|-------|
# | H20            | SM90  |
# | H100           | SM90  |
# | H200           | SM90  |
# |----------------|-------|
# | B200           | SM100 |
# | Blackwell RTX  | SM103 |
# |----------------|-------|
#

from functools import partial

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

try:
    from flash_attn_3.flash_attn_interface import flash_attn_func

    FLASH_ATTN_3_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_3_AVAILABLE = False


def get_device_cc(device) -> int:
    """
    Returns the compute capability of a given torch device if it's a CUDA device, otherwise returns 0.

    Args:
        device: torch device.

    Returns:
        device_cc (int): compute capability in the SmXXX format (i.e. 90 for Hopper).
    """
    if torch.cuda.is_available() and torch.version.cuda and device.type == "cuda":
        major, minor = torch.cuda.get_device_capability(device)
        return major * 10 + minor
    return 0


def attention(
    q,
    k,
    v,
    dropout_p=0.0,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    deterministic=False,
):
    assert q.dtype == k.dtype and k.dtype == v.dtype
    dtype = q.dtype
    supported_dtypes = [torch.bfloat16, torch.float16, torch.float32]
    is_half = dtype in [torch.bfloat16, torch.float16]
    compute_cap = get_device_cc(q.device)

    if dtype not in supported_dtypes:
        raise NotImplementedError(f"{dtype=} is not supported.")

    if q_scale is not None:
        q = q * q_scale

    # If Flash Attention 3 is installed, and the user's running on a Hopper GPU (compute capability
    # 9.0, or SM90), use Flash Attention 3.
    if compute_cap == 90 and FLASH_ATTN_3_AVAILABLE and is_half:
        return flash_attn_func(
            q=q,
            k=k,
            v=v,
            softmax_scale=softmax_scale,
            causal=causal,
            deterministic=deterministic,
        )[0]
    else:
        # If Blackwell or Hopper (SM100 or SM90), cuDNN has native FMHA kernels. The Hopper one is
        # not always as fast as Flash Attention 3, but when Flash Attention is unavailable, it's
        # still a far better choice than Flash Attention 2 (Ampere).
        if compute_cap in [90, 100] and is_half:
            SDPA_BACKENDS = [
                SDPBackend.CUDNN_ATTENTION,
                SDPBackend.FLASH_ATTENTION,
                SDPBackend.EFFICIENT_ATTENTION,
            ]
            BEST_SDPA_BACKEND = SDPBackend.CUDNN_ATTENTION
        elif is_half:
            SDPA_BACKENDS = [
                SDPBackend.FLASH_ATTENTION,
                SDPBackend.CUDNN_ATTENTION,
                SDPBackend.EFFICIENT_ATTENTION,
            ]
            BEST_SDPA_BACKEND = SDPBackend.FLASH_ATTENTION if compute_cap >= 80 else SDPBackend.EFFICIENT_ATTENTION
        else:
            assert dtype == torch.float32, f"Unrecognized {dtype=}."
            SDPA_BACKENDS = [SDPBackend.EFFICIENT_ATTENTION]
            BEST_SDPA_BACKEND = SDPBackend.EFFICIENT_ATTENTION

        if deterministic:
            raise NotImplementedError("Deterministic mode in attention is only supported when Flash Attention 3 is available.")

        # Torch 2.6 and later allows priorities for backends, but for older versions
        # we can only run with a specific backend. As long as we pick ones we're certain
        # will work on that device, it should be fine.
        try:
            sdpa_kernel(backends=SDPA_BACKENDS, set_priority_order=True)
            sdpa_kernel_ = partial(sdpa_kernel, set_priority_order=True)
        except TypeError:
            sdpa_kernel_ = sdpa_kernel
            SDPA_BACKENDS = [BEST_SDPA_BACKEND]

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        with sdpa_kernel_(backends=SDPA_BACKENDS):
            out = torch.nn.functional.scaled_dot_product_attention(
                q,
                k,
                v,
                is_causal=causal,
                dropout_p=dropout_p,
                scale=softmax_scale,
            )

        out = out.transpose(1, 2).contiguous()
        return out
