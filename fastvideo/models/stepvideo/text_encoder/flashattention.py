# Copyright 2025 StepFun Inc. All Rights Reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# ==============================================================================
import torch


def flash_attn_func(q,
                    k,
                    v,
                    dropout_p=0.0,
                    softmax_scale=None,
                    causal=True,
                    return_attn_probs=False,
                    tp_group_rank=0,
                    tp_group_size=1):
    softmax_scale = q.size(-1)**(-0.5) if softmax_scale is None else softmax_scale
    return torch.ops.Optimus.fwd(q, k, v, None, dropout_p, softmax_scale, causal, return_attn_probs, None,
                                 tp_group_rank, tp_group_size)[0]


class FlashSelfAttention(torch.nn.Module):

    def __init__(
        self,
        attention_dropout=0.0,
    ):
        super().__init__()
        self.dropout_p = attention_dropout

    def forward(self, q, k, v, cu_seqlens=None, max_seq_len=None):
        if cu_seqlens is None:
            output = flash_attn_func(q, k, v, dropout_p=self.dropout_p)
        else:
            raise ValueError('cu_seqlens is not supported!')

        return output
