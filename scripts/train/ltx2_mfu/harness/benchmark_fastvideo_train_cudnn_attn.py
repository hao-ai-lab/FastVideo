#!/usr/bin/env python3
"""Scratch gate: route LTX-2 video self-attention to cuDNN SDPA.

Wraps the FLASH_ATTN backend's compilable entrypoint so equal-length
query/key calls (video self-attention) run ``F.scaled_dot_product_attention``
under the cuDNN backend while unequal-length calls (text cross-attention)
stay on FA4, then runs the frozen packed benchmark harness unchanged. The
microbench precedent is a 2.3-3.4% per-call win for cuDNN at (B, 4290, 32,
128) fwd+bwd; this measures the end-to-end step.
"""

from __future__ import annotations

import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

import benchmark_fastvideo_train_pack_d016 as benchmark
import fastvideo.attention.backends.flash_attn as fa_module

_original = fa_module.flash_attn_func_compilable


def _hybrid_attention(q, k, v, softmax_scale=None, causal=False):
    if q.shape[1] == k.shape[1]:
        with sdpa_kernel(SDPBackend.CUDNN_ATTENTION):
            return F.scaled_dot_product_attention(
                q.transpose(1, 2),
                k.transpose(1, 2),
                v.transpose(1, 2),
                scale=softmax_scale,
                is_causal=causal,
            ).transpose(1, 2)
    return _original(q, k, v, softmax_scale=softmax_scale, causal=causal)


fa_module.flash_attn_func_compilable = _hybrid_attention

if __name__ == "__main__":
    benchmark.main()
