#pragma once
#include <ATen/ATen.h>

// The output head, epilogue half: conv_out's 3x3 spatial stencil, its bias and the clamp to
// [-1,1], from the 112-channel plane the 3x1x1 convolution writes (conv.cu's `conv_n128`) to the
// 12-channel plane write_video consumes.
//   g:   [1,112,D,rows+2*g_off,W] channels-last-3d bf16, n = (i*3+j)*12 + c for the tap (i,j);
//        channels 108..111 are the zero-weight lanes and are never read.
//   out: [1,12,D,rows,W] channels-last-3d bf16, written whole.
// H_global is the height of the plane this stencil filters (352 = 44 latent rows x 8; the pixel
// plane is 704 tall and run()'s video row is twice this one). lo is the first row of this rank's
// band IN THAT PLANE, 8*(44*rank/world), and it must stay that: run() passes lo*16 to write_video,
// the same row on the 704-tall video grid, and the two derivations have to agree.
void head_stencil(const at::Tensor& g,const at::Tensor& bias,const at::Tensor& out,
                  int64_t H_global,int64_t g_off,int64_t lo);

// conv.cu: the head's 3x1x1 convolution, at the tile its 112 output channels want. Declared here
// so the two-sided declaration of a CUTLASS tile cannot drift from its definition in conv.cu.
at::Tensor conv_n128(const at::Tensor& x,const at::Tensor& w);
