// F(4x4,3x3) spatial input transform of the Winograd path of the Wan2.2 VAE decoder.
//
// ONE PASS: `winograd_input_fast` replaces `kernels.cu`'s `winograd_input` for the 18 Winograd
// convolutions (1024/512/256 channels; any other channel count keeps the general kernel), and
// its output V is BIT-IDENTICAL to it -- same patch origin, same block-uniform fast-path
// condition, same B^T-then-B order with the in-place column update, same 6a+b plane index, same
// single bf16 rounding at the store, same plane stride.  Nothing else in the tree moves but the
// one call site in binding.cpp and one declaration in kernels.h.
//
// Why it is faster, from the profile of the kernel it replaces (w01's J549): 581.8 M
// warp-instructions in 1.031 ms at 59.5% issue-active, of which 43.5% are ALU/integer against
// 47.6% FMA and 4.9% LSU; DRAM 23.6%, FMA pipe 32.4%.  It sits at 82% of its own instruction
// issue bound and the integer address math is the only large removable block:
//   1. a 3-D grid, blockIdx = (tx, ty, t) and threadIdx = the channel pair, so nothing divides;
//   2. one block-uniform fast path for the interior tiles, where the 36 loads carry no
//      predicate at all -- only the first and last tile of a row and the first and last tile
//      row can touch the time pad or the frame edge -- and the patch row pointer advances by
//      the row stride, so the 36 addresses are 36 LSU instructions with folded offsets;
//   3. the 36 plane stores walk one pointer by V's plane stride instead of doing an int64
//      multiply per store.
//
// MORE BLOCKS PER TILE INSTEAD OF ONE: SPLIT = 4 AT C = 1024, 2 AT C = 512 AND 256.
// A tile's C/2 channel pairs used to be covered by one block of C/2 threads, so `blockDim` and
// `C` were the same variable and the grid held exactly one block per tile.  A block is now the
// tile's width divided by SPLIT, and SPLIT blocks of the grid cover one tile.  At C = 1024 that
// moves the level from one block of 16 warps (32.08 ms) to five blocks of 4 warps (22.42 ms,
// 20 warps resident); at C = 512 and C = 256 the same move is worth 1.05 % and -0.4 % -- i.e.
// nothing -- so those two stay at SPLIT = 2.  See the table below for why.
//
// `__launch_bounds__(CP, 512 / CP)` IS LOAD-BEARING AND IS HALF THE MEASUREMENT.  It pins the
// register budget at 65536/512 = 128 at EVERY width -- exactly what main's
// `__launch_bounds__(512, 1)` grants at its one wide instantiation.  Without it ptxas may spend
// up to the architectural cap of 255 registers per thread (65536/CP is itself 512 at CP = 128,
// so it is the 255 that binds there), floor(65536/(CP * REG)) falls, the resident warps follow,
// and the same arm runs 2.25x SLOWER.  The counterfactual is measured at ONE instantiation:
// J3705 ran `wino_in_kernel_st<256,2>` both ways -- `__launch_bounds__(256, 1)` at 59.11 ms
// against `(256, 2)` at 26.26 ms, one integer and one body.  At the shipped CP = 128 the no-pin
// case gives floor(65536/(128*255)) = 2 blocks of 4 warps against the pinned 5, i.e. 20 resident
// warps down to 8 -- a 2.5x cut, not a halving, and the reason C = 1024 needs the pin most.
// The declaration and the launch are one change; neither is the arm without the other.
//
// Measured (J3868, one build, one process, A B B B B A over SPLIT = 1 2 4 8 2 1, f121x1
// --kernels, 1132 MHz on every leg, exit 0).  Both control legs are shown so the drift is visible:
//                     SPLIT=1      SPLIT=2          SPLIT=4        SPLIT=8      2 -> 4
//   C = 1024 (shipped)  32.08/32.08  26.26/26.53      22.42          22.27       1.1775
//   C =  512 (2 kept)   64.06/63.92  54.60/55.44      54.45          55.45       1.0105
//   C =  256 (2 kept)   60.47/60.66  59.56/60.05      60.04        (n/a)         0.9962
//   CALL               864.7        847.6            844.2          845.2
//   A/A 0.3 %, B/B 1.0-1.5 %; max |d| 0.1543 of 0.4984, rms 0.002093 of 0.005847, PSNR 59.6 dB,
//   IDENTICAL to every printed digit on all six legs.
//   The C = 256 SPLIT = 8 cell is `n/a`, not a decline: J3868's probe only instantiated the CP
//   values it needed and SPLIT = 8 at C = 256 would ask for CP = 16, which that build did not
//   contain.  The head's launcher is an exhaustive switch with no fall-back path of any kind.
//
// ONLY THE 2 -> 4 MOVE IS A C = 1024-ONLY GAIN, and that is why the other two keep SPLIT = 2:
//   the shipped SPLIT = 2 already takes the 1 -> 2 step on BOTH of them -- 64.06 -> 54.60 at
//   C = 512 is 1.1733x and 60.47 -> 59.56 at C = 256 is 1.0183x, the +15.5 % and +1.5 % below --
//   so neither is a rung that gains nothing.  What is declined on them is the FOURTH block:
//   C = 512's 1.0105 is inside the B/B drift (its own legs read 54.60 and 55.44, a 1.5 % spread);
//   C = 256's 0.9962 REVERSES -- its legs read 59.56, 60.05 and 60.66, a 1.8 % spread, and 60.04
//   sits inside it, so the option is declined rather than merely unpriced;
//   SPLIT = 8 at C = 1024 is 22.27 against 22.42 -- 0.7 % at a 1.0 % drift, a tie, not a win.
//
// WHY C = 1024 AND NOT THE OTHERS.  With REG 96 the realized resident warps are
// floor(65536/(96*blockDim)) * blockDim/32, i.e. 16 at blockDim 512 or 256, 20 at 128 or 64, and
// 21 at 32.  Main's C = 1024 case used blockDim 512, which rounds to ONE block and never reaches
// past 16 warps although 2048/96 = 21.3 is its ceiling; `__launch_bounds__` cannot fix that,
// because 2 * 512 * 96 = 98,304 > 65,536.  Only a narrower block can, and narrowing it is SPLIT.
// C = 512 and C = 256 already reach 20 warps at SPLIT = 2, so there is nothing there to buy.
// The three levels' rates were 2.609 / 3.340 / 3.760 TB/s (J3195's counted bytes).  DERIVED, not
// re-measured: at fixed byte counts rate_after/rate_before = t_before/t_after, so from J3868's own
// legs the after-rates are 2.609*32.08/22.42 = 3.733, 3.340*64.06/55.44 = 3.859 and
// 3.760*60.47/59.56 = 3.817 TB/s -- a 44.1 % spread closed to 3.4 %.  The largest move is the
// slow level's own (+43 %), and the fast two barely move (+15.5 % and +1.5 %), so what happened is
// that the slow level was walked up to where the other two already sat, not that a ceiling moved.
// An earlier draft of this comment stated 3.952 for C = 256, which needs 60.47 -> 57.5 ms and no
// leg of J3868 is below 59.56; it was a number carried over from a withdrawn row and it is wrong.
//
// `J3761`'s rebuild of the round-1 source: BUILD_RC=0, REG:96 STACK:0 SHARED:0 LOCAL:0.
// `J3889`'s build of this source: the same, with `wino_in_kernel<128,4>` at REG:95 (one register
// lower, which changes no block count: 65536/(128*95) = 5.39 -> 5, the same as at 96).
// Correctness at worlds 1 and 4 on all three short clips: J3889 f1x1/f5x1/f17x1 ok at
// x0.41/x0.41/x0.65 and J3908 f1x4/f5x4/f17x4 ok at x0.45/x0.45/x0.66, every leg inside main's
// own margins.
//
// The share of the call is flat in clip length (17.6 % at f121x1, 17.9 % at f481x1) and this
// function takes no rank, no world and no process group, so a world-1 cell is the geomean cell.
//
// Measured by w04 on main's own tree (kernel#1, this change alone, f121x1 at 1849 MHz): the
// three instantiations total 114.4 ms against 221.8 ms of `winograd_input_kernel`, and the call
// 969.9 -> 862.5 ms (-11.1%), with V bit-identical -- f1/f5/f17 keep the seed's max |d| and rms
// to the digit -- and f481x1 ok at x0.38 of the max and x0.41 of the rms allowance.  w21 wrote
// the kernel; w08 owns this file.
#include "kernels.h"
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <algorithm>

using B = __nv_bfloat16;
static cudaStream_t stream() { return at::cuda::getCurrentCUDAStream(); }

__device__ __forceinline__ void wbt6(float v0, float v1, float v2, float v3, float v4, float v5, float* y) {
  float a = v4 - 4 * v2, b = v3 - 4 * v1, c = v4 - v2, d = 2 * (v3 - v1);
  y[0] = 4 * v0 - 5 * v2 + v4; y[1] = a + b; y[2] = a - b; y[3] = c + d; y[4] = c - d; y[5] = 4 * v1 - 5 * v3 + v5;
}

// One block per (part-tile, frame); threadIdx.x = the channel pair within this block's share.
// CP = C/2/SPLIT is the block size.  The shipped dispatch is SPLIT = 4 at C = 1024 and SPLIT = 2
// at C = 512 and 256, so blockDim is 128 / 128 / 64 at C = 1024 / 512 / 256 -- but SPLIT is NOT
// uniform and no comment here may claim it is.  The invariant that actually guarantees coverage is
//     CP * SPLIT == C / 2                                    (128*4 = 128*2 = 64*2 = C/2)
// enforced by the `static_assert` in `winograd_input_fast` below, together with `grid.x == SPLIT*TW`
// from the same `constexpr`.  `(blockIdx.x % SPLIT) * CP + threadIdx.x` then walks [0, SPLIT*CP) =
// [0, C/2) exactly once -- every channel pair of a tile is written once and no pair twice -- and
// `blockIdx.x / SPLIT` over a grid of SPLIT*TW hits each tile exactly SPLIT times, for any SPLIT:
// the remainder cycles 0..SPLIT-1 within each group of SPLIT consecutive blockIdx, so TW need not be
// a multiple of SPLIT.
//
// SPLIT = 1 with CP = C/2 gives back main's one-block-per-tile BODY, but not its attribute (this
// one is `__launch_bounds__(CP, 512/CP)`, which equals main's `(CP, 1)` only at CP = 512) and not
// its grid shape, so do not use it as an exact revert; revert by checking out the parent commit.
//
// The launched blockDim must equal CP.  It does at all three call sites below, from the same
// template value, so the two cannot drift apart.
template <int CP, int SPLIT>
__global__ void __launch_bounds__(CP, 512 / CP)
wino_in_kernel(const B* __restrict__ x, B* __restrict__ v, int C, int D, int H, int W, int PH, int TH, int TW) {
  const int cp = (blockIdx.x % SPLIT) * CP + threadIdx.x;
  const int tx = blockIdx.x / SPLIT;
  const int ty = blockIdx.y, t = blockIdx.z;
  const int HP = H + 2 * PH;
  const int y0 = 4 * ty - 1 + PH, x0 = 4 * tx - 1;
  const int64_t rowStep = (int64_t)W * C;
  const B* base = x + (int64_t)t * HP * rowStep + (int64_t)y0 * rowStep + (int64_t)x0 * C + 2 * cp;
  float d[2][6][6];
  // Uniform across the block, so it is one branch and not 36 predicates.
  if ((y0 >= 0) && (y0 + 5 < HP) && (x0 >= 0) && (x0 + 5 < W)) {
    const B* prow = base;
#pragma unroll
    for (int a = 0; a < 6; a++) {
#pragma unroll
      for (int b = 0; b < 6; b++) {
        float2 f = __bfloat1622float2(*(const __nv_bfloat162*)(prow + b * C));
        d[0][a][b] = f.x; d[1][a][b] = f.y;
      }
      prow += rowStep;
    }
  } else {
    const B* prow = base;
#pragma unroll
    for (int a = 0; a < 6; a++) {
      const int y = y0 + a;
      const bool ok = (y >= 0) && (y < HP);
#pragma unroll
      for (int b = 0; b < 6; b++) {
        const int xx = x0 + b;
        if (ok && xx >= 0 && xx < W) {
          float2 f = __bfloat1622float2(*(const __nv_bfloat162*)(prow + b * C));
          d[0][a][b] = f.x; d[1][a][b] = f.y;
        } else {
          d[0][a][b] = 0.f; d[1][a][b] = 0.f;
        }
      }
      prow += rowStep;
    }
  }
  // B^T d down the patch's 6 columns, then B along its 6 rows, one bf16 rounding at the end.
#pragma unroll
  for (int k = 0; k < 2; k++)
#pragma unroll
    for (int b = 0; b < 6; b++) {
      float q[6];
      wbt6(d[k][0][b], d[k][1][b], d[k][2][b], d[k][3][b], d[k][4][b], d[k][5][b], q);
#pragma unroll
      for (int a = 0; a < 6; a++) d[k][a][b] = q[a];
    }
  B* dst = v + (int64_t)(((t * TH + ty) * TW + tx) * C + 2 * cp);
  const int64_t plane = (int64_t)D * TH * TW * C;
#pragma unroll
  for (int a = 0; a < 6; a++) {
    float q[2][6];
#pragma unroll
    for (int k = 0; k < 2; k++)
      wbt6(d[k][a][0], d[k][a][1], d[k][a][2], d[k][a][3], d[k][a][4], d[k][a][5], q[k]);
#pragma unroll
    for (int b = 0; b < 6; b++, dst += plane) *(__nv_bfloat162*)dst = __floats2bfloat162_rn(q[0][b], q[1][b]);
  }
}

Tensor winograd_input_fast(const Tensor& x, int ph) {
  const int C = x.size(1), D = x.size(2), H = x.size(3) - 2 * ph, W = x.size(4);
  const int TH = (H + 3) / 4, TW = (W + 3) / 4;
  // The 18 Winograd convolutions are the 1024-, 512- and 256-channel ones; anything else
  // keeps the general kernel.
  if (C != 1024 && C != 512 && C != 256) return winograd_input(x, ph);
  auto v = at::empty({36, D, TH, TW, C}, x.options());
  // SPLIT = 4 at C = 1024, 2 at C = 512 and 256.  CP * SPLIT = C/2 at every case, so a tile's
  // channel pairs are covered exactly once; grid.x is SPLIT * TW, a block count and not a tile
  // count.  blockDim is CP, taken from the same template value, so the two cannot drift.
  //
  // Why 4 only at C = 1024 (J3868: 2 -> 4 on the rung at each C, one build, one process):
  //   C=1024  <256,2> 26.26/26.53 -> <128,4> 22.42   1.1775   warps 16 -> 20
  //   C= 512  <128,2> 54.60/55.44 -> < 64,4> 54.45   1.0105   warps 20 -> 20   inside the 1.5 % drift
  //   C= 256  < 64,2> 59.56/60.05 -> < 32,4> 60.04   0.9962   warps 20 -> 21   it REVERSES
  // So the resolved gain is C = 1024 alone, and it is the only rung whose split buys warps:
  // with REG 96 the realized resident warps are floor(65536/(96*blockDim))*blockDim/32, i.e.
  // 16 at blockDim 512 or 256, 20 at 128 or 64, 21 at 32.  The step off 16 is the whole of it.
  // SPLIT = 8 at C = 1024 measured 22.27 against 22.42 -- 0.7 % at a 1.0 % B/B drift, not
  // resolvable, so it is not taken.  A tie is not a win.
  //
  // ONE `constexpr` PER CASE BINDS THE TEMPLATE ARGUMENT, THE BLOCK WIDTH, THE GRID MULTIPLIER AND
  // THE COVERAGE INVARIANT.  Three literals in three places is a desync waiting to become a bug:
  // raise SPLIT without raising grid.x and `tx = blockIdx.x / SPLIT` stays inside TW, the kernel
  // silently computes one block's share of every tile and the rest of V is never written; raise
  // grid.x without SPLIT and `tx` runs past TW, computing x0 = 4*tx-1 for tiles that do not exist,
  // which is an out-of-bounds read AND write rather than a slowdown.  The `static_assert` makes the
  // first a compile error and the single `constexpr` makes the second unrepresentable.
  const B* xp = (const B*)x.data_ptr<at::BFloat16>();
  B* vp = (B*)v.data_ptr<at::BFloat16>();
  auto st = stream();
  switch (C) {
    case 1024: {
      constexpr int SPLIT = 4, CP = 1024 / 2 / SPLIT;
      static_assert(CP * SPLIT == 1024 / 2, "CP * SPLIT must equal C/2 or a tile is not covered");
      const dim3 g((unsigned)(SPLIT * TW), (unsigned)TH, (unsigned)D);
      wino_in_kernel<CP, SPLIT><<<g, CP, 0, st>>>(xp, vp, C, D, H, W, ph, TH, TW);
      break;
    }
    case 512: {
      constexpr int SPLIT = 2, CP = 512 / 2 / SPLIT;
      static_assert(CP * SPLIT == 512 / 2, "CP * SPLIT must equal C/2 or a tile is not covered");
      const dim3 g((unsigned)(SPLIT * TW), (unsigned)TH, (unsigned)D);
      wino_in_kernel<CP, SPLIT><<<g, CP, 0, st>>>(xp, vp, C, D, H, W, ph, TH, TW);
      break;
    }
    case 256: {
      constexpr int SPLIT = 2, CP = 256 / 2 / SPLIT;
      static_assert(CP * SPLIT == 256 / 2, "CP * SPLIT must equal C/2 or a tile is not covered");
      const dim3 g((unsigned)(SPLIT * TW), (unsigned)TH, (unsigned)D);
      wino_in_kernel<CP, SPLIT><<<g, CP, 0, st>>>(xp, vp, C, D, H, W, ph, TH, TW);
      break;
    }
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return v;
}
