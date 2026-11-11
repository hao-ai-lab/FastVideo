#pragma once
#include <ATen/ATen.h>
using at::Tensor;
Tensor nhwc_empty(int64_t c,int64_t t,int64_t h,int64_t w,const Tensor& like);
Tensor norm_pad(const Tensor& x, const Tensor& gamma, const Tensor& history, int pt, int ph, bool silu);
Tensor last_frames(const Tensor& padded, int n, int ph);
void finish(Tensor& x, const Tensor& bias, const Tensor& residual);
void write_video(const Tensor& x, Tensor& out, int frame_offset, int row_offset);

Tensor winograd_input(const Tensor& x,int ph);
Tensor winograd_input_fast(const Tensor& x,int ph);
// winograd_output / winograd_output_kernel are deleted: the fused inverse in wino_epilogue.cu
// replaced their only caller, and they had no live user left (PR172 round-1 review).

// Fused inverse transform + bias/residual + RMS norm/SiLU.  y = A^T M A + bias + residual;
// n (when gamma is given) = silu(rmsnorm(y)*sqrt(C)*gain) in the (D+pt, H+2ph, W, C) padded
// layout the next convolution consumes, with its first pt frames taken from `hist`.
struct WinoOut { Tensor y; Tensor n; };
WinoOut wino_epilogue(const std::vector<Tensor>& m,int H,int W,const Tensor& bias,
                      const Tensor& residual,const Tensor& gamma,const Tensor& hist,int pt,int ph,
                      bool write_y=true);
