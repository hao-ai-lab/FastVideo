// Resampler stage: the four-phase spatial upsamplers of up_blocks 0,1,2 of the Wan2.2 VAE decoder.
// See resample.h for why the phase convolutions stay direct and what this pass replaces.
#include "resample.h"
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
using B = __nv_bfloat16;
// conv.cu: a 2x2 low-resolution convolution of one output parity, padded per phase; `halo` is set when
// the input carries a one-row halo from the neighbour rank, so the input is narrowed instead of padded.
at::Tensor phase_conv(const at::Tensor&, const at::Tensor&, int, int, bool);

static cudaStream_t stream() { return at::cuda::getCurrentCUDAStream(); }
static const B* pc(const Tensor& t) { return (const B*)t.data_ptr<at::BFloat16>(); }
static B* pm(Tensor& t) { return (B*)t.data_ptr<at::BFloat16>(); }

union P8 { uint4 v; B b[8]; __nv_bfloat162 h[4]; };

// One thread: eight channels of one 2x2 low-resolution tile, all four phases.  The grid carries the
// coordinates (tx, t, ty) so there is no integer division on the hot path, and every access is a
// 16-byte vector: four phase outputs and the DupUp3D source in, four interleaved positions out.
__global__ void upsampler_kernel(const B* __restrict__ p0, const B* __restrict__ p1,
                                 const B* __restrict__ p2, const B* __restrict__ p3,
                                 B* __restrict__ y, const B* __restrict__ bias,
                                 const B* __restrict__ skip, int C, int HL, int WL, int TC,
                                 bool temporal, bool first) {
  const int cg = threadIdx.x;                       // 8-channel group
  const int tx = blockIdx.x, t = blockIdx.y, ty = blockIdx.z;
  const int CP = C / 8, SP = TC / 8;
  const P8 b = {((const uint4*)bias)[cg]};
  const int st = temporal ? (first ? (t + 1) / 2 : t / 2) : t;
  const int64_t wC = 2 * (int64_t)WL * C;            // one row of the interleaved output
  const int64_t tC = 2 * (int64_t)HL * wC;           // one frame of it
#pragma unroll
  for (int i = 0; i < 2; ++i) {
    const int yy = 2 * ty + i;
    if (yy >= HL) continue;
#pragma unroll
    for (int j = 0; j < 2; ++j) {
      const int xx = 2 * tx + j;
      if (xx >= WL) continue;
      const int64_t q0 = ((int64_t)t * HL + yy) * WL + xx;
      P8 ph[4];
      ph[0].v = ((const uint4*)p0)[q0 * CP + cg];
      ph[1].v = ((const uint4*)p1)[q0 * CP + cg];
      ph[2].v = ((const uint4*)p2)[q0 * CP + cg];
      ph[3].v = ((const uint4*)p3)[q0 * CP + cg];
      const int64_t s = ((int64_t)st * HL + yy) * WL + xx;
      P8 s0, s1;
      if (temporal) {
        s0.v = ((const uint4*)skip)[s * SP + cg];
        s1.v = s0.v;                                 // nearest x2 in time: both row parities alike
      } else {
        // up block 2: output channel J takes source channel 2*J + (row parity), so read both parities.
        s0.v = ((const uint4*)skip)[s * SP + 2 * cg];
        s1.v = ((const uint4*)skip)[s * SP + 2 * cg + 1];
      }
      B* out = y + (int64_t)t * tC + (int64_t)(2 * yy) * wC + (int64_t)(2 * xx) * C + 8 * cg;
      if (!temporal) {
        // channel J of the output takes source channel 2J + a: even channels from s0, odd from s1.
        P8 e, o;
#pragma unroll
        for (int k = 0; k < 4; ++k) {
          e.b[k] = s0.b[2 * k];
          e.b[k + 4] = s1.b[2 * k];
          o.b[k] = s0.b[2 * k + 1];
          o.b[k + 4] = s1.b[2 * k + 1];
        }
        s0.v = e.v;
        s1.v = o.v;
      }
      P8 r;
#pragma unroll
      for (int p = 0; p < 4; ++p) {
        const P8& sv = (p & 2) ? s1 : s0;            // row parity of this phase
#pragma unroll
        for (int k = 0; k < 4; ++k) r.h[k] = __hadd2(__hadd2(ph[p].h[k], b.h[k]), sv.h[k]);
        const int a = p >> 1, bcol = p & 1;
        ((uint4*)(out + (int64_t)a * wC + (int64_t)bcol * C))[0] = r.v;
      }
    }
  }
}

Tensor resample_upsample(const Tensor& x, const std::vector<Tensor>& pw, const Tensor& bias,
                         const Tensor& skip, bool halo, bool temporal, bool first) {
  TORCH_CHECK(pw.size() == 4, "resample_upsample needs the four phase kernels");
  std::vector<Tensor> q(4);
  for (int a = 0; a < 2; ++a) for (int b = 0; b < 2; ++b) q[2 * a + b] = phase_conv(x, pw[2 * a + b], a, b, halo);
  const int C = q[0].size(1), D = q[0].size(2), HL = q[0].size(3), WL = q[0].size(4);
  Tensor y = at::empty({1, C, D, 2 * HL, 2 * WL}, x.options().memory_format(at::MemoryFormat::ChannelsLast3d));
  dim3 block(C / 8, 1), grid((WL + 1) / 2, D, (HL + 1) / 2);
  upsampler_kernel<<<grid, block, 0, stream()>>>(pc(q[0]), pc(q[1]), pc(q[2]), pc(q[3]), pm(y), pc(bias),
                                                 pc(skip), C, HL, WL, (int)skip.size(1), temporal, first);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return y;
}
