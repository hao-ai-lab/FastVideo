#pragma once
#include <ATen/ATen.h>
// One head of self-attention over the H*W positions of every latent frame of
// one decoder chunk: qkv is [T,1,HW,3C] bf16 (q, k, v concatenated on the last
// dim, as binding.cpp's attention() builds it); the result is [T,1,HW,C] bf16.
at::Tensor attn_forward(const at::Tensor& qkv);
