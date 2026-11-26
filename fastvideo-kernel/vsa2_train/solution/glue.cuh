// glue.cuh -- the backward's row-wise passes: what is neither the fine branch (bwd.cuh) nor the coarse branch
// (coarse_attn.cuh), the two passes that touch every row of the sequence once.
//
//   bwd_rows_kernel: per (block, head): D = rowsum(dout . (out - gate * Oc)) (the fine output from the bf16 `out`,
//     as the task's calibration emulation does), the row's lse in log2 units (+inf on padding rows, so its P is 0),
//     dgate = Oc * dout, the coarse branch's output gradient dco (the block's sum of gate * dout, as the coarse
//     backward's three bf16 component planes and the row's lift, coarse_tc.cuh) with its Dc = rowsum(dco * Oc), the
//     row's dqaccum zeroed, and the block's rows of dqc, dkc and dvc zeroed (`coarse_zero`, the three arrays in one,
//     which the coarse backward's splits add to).
//   bwd_post_kernel: dq = scale * dqaccum + the coarse share, zero on padding rows; it undoes the drain-native
//     layout dqaccum is written in (bwd.cuh: [slice of 32 columns][v4][row][4] within a 128-row block).
//
// The two tensors that pass over HBM here are read once each: `gate` and `dout` feed both dco and dgate in one pass
// (they were read twice while dco lived in the prep pass), and padding rows -- whose q, k, v, gate and dout are zero
// -- are written but not read (12.7% of the rows at s20).
//
// The two kernels are *defined* in glue.cu, one translation unit away from the `vsa2t.cu` that `#include`s every
// stage, because the register allocator's spill set for a kernel turns out to depend on the contents of its whole
// translation unit: `fwd_fine` moved from STACK:72 / 21 local accesses to STACK:80 / 25 when this header was first
// added to `vsa2t.cu`, with `fwd.cuh` byte-identical and `bwd_fine` unmoved as the control. A stage that can move
// another stage's frame by being inlined into the same file belongs in its own file.
#pragma once
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

#include "common.cuh"

namespace vsa2t {
namespace glue {

constexpr int kBlock = 128;    // rows of a block
constexpr int kHeadDim = 128;  // head dimension
// The fine backward's dQ drain slices (bwd.cuh: DQ_COLS) and its per-block element count.
constexpr int kDqCols = 32;
constexpr int kDqBlockElems = kBlock * kHeadDim;
static_assert(kDqCols * (kHeadDim / kDqCols) * kBlock == kDqBlockElems, "drain-native dqaccum block");
// `bwd_post_kernel` undoes the layout the drain writes, so `kDqCols` and `kBlock` must be the drain's own constants:
// if `bwd.cuh`'s slice width or block size changes, this reads the wrong elements rather than failing. That pin
// (`static_assert(kDqCols == bwd::DQ_COLS && kBlock == bwd::BLOCK)`) cannot live here, because `bwd.cuh` defines the
// backward's own kernels and this header is now included by two translation units; it sits in vsa2t.cu, the one file
// that includes both headers.

// The row pass: 512 threads, thread = (row group 0..31, 8 dims of the head), 4 rows per group; the block's tokens are
// covered by 32 groups x 4 = 128 rows.
constexpr int kRowsThreads = 512;
constexpr int kRowGroups = kRowsThreads / 16;         // 32
constexpr int kDimsPerThread = kHeadDim / 16;         // 8
constexpr int kRowsPerGroup = kBlock / kRowGroups;    // 4

// Per row: D, lse2, dgate, the row's dqaccum zeroed; per block: dco and Dc = rowsum(dco * Oc) ([heads][npad], npad =
// N rounded up to 32). Grid (blocks, heads). Defined in glue.cu.
__global__ void __launch_bounds__(kRowsThreads) bwd_rows_kernel(
    Geo g, const __nv_bfloat16* __restrict__ out, const __nv_bfloat16* __restrict__ dout,
    const __nv_bfloat16* __restrict__ gate, const float* __restrict__ lse, const float* __restrict__ oc,
    const uint32_t* __restrict__ rowmask, float* __restrict__ delta, float* __restrict__ lse2,
    __nv_bfloat16* __restrict__ dgate, __nv_bfloat16* __restrict__ dco_planes, int* __restrict__ dco_lift, int rpad,
    float* __restrict__ dc, float* __restrict__ dqaccum, float* __restrict__ coarse_zero);

// dq = dqaccum (drain-native, times sm_scale) + the coarse share, zero on padding rows; one CTA per (half block,
// head). kPostCtas is main's REGISTER BUDGET, not a residency cap -- see the register paragraph below before reading it
// as "3 CTAs": the shipped kernel compiles to REG:120 and runs FOUR resident CTAs (16 warps/SM). The tile, the pipeline
// and the row split are the only changes to this kernel. Defined in glue.cu.
constexpr int kPostThreads = 128;
constexpr int kPostCtas = 3;
// ROW SPLIT: rows of the 128-row block that one CTA owns, so the launch's x dimension is N * (kBlock/kPostRows). See
// the kernel's comment for the hypothesis this tests (w26: the stream rate is a function of the CTA COUNT, not of
// per-CTA work). Neither the launch nor the kernel spells a literal: both derive the tiling from this constant, so a
// non-divisor of kBlock is impossible by construction (a static_assert says so: kPostRows = 48 would truncate
// kTilesPerBlock to 2 and leave the block's last 32 rows unwritten into an empty_like buffer, with the build green).
// At the shipped kBlock = 128 / kPostThreads = 128 the values that build are 16, 32, 64 and 128; the smaller divisors
// are stopped by an assert in the kernel, which is where the reason lives (it depends on kPostThreads too, so a copy
// here would be a second place to go stale). 64 -- one CTA per half block -- is the SHAPE THIS KERNEL SHIPS, which is
// a policy and not the only correct value: 32 partitions four ways correctly and is the direction J6967's CTA-count
// curve favours, if anyone wants to measure it.
constexpr int kPostRows = 64;
// REGISTERS AND RESIDENCY: __launch_bounds__'s second argument is a REGISTER BUDGET for ptxas, not an occupancy cap,
// so `3` means "make 3 fit", and the resident count is whatever the compiled width allows. At the 128-row tile the
// kernel was REG:162 (fp32 stage, J1964) or REG:168 (bf16 tile, J6889) and a fourth CTA was excluded by the register
// file either way -- 162 x 128 x 4 = 82,944 and 168 x 128 x 4 = 86,016, both > 65,536 -- so a fourth CTA needed a
// narrower kernel, not a different launch bound. THE ROW SPLIT DELIVERED IT: at 64 rows it compiles to REG:120
// (128 x 120 = 15,360 per CTA), so 4 x 15,360 = 61,440 fits 65,536 and 4 x 17,408 B = 69,632 B fits the 228 KB, and
// the fourth CTA is resident for the shipped kernel with no change to kPostCtas. The split's -12.0 % at s8 is
// therefore grid + halved tile + 3 -> 4 CTAs / 12 -> 16 warps per SM together, and nothing here separates them.
__global__ void __launch_bounds__(kPostThreads, kPostCtas)
    bwd_post_kernel(const float* __restrict__ dqaccum, const float* __restrict__ dq_coarse,
                    const uint32_t* __restrict__ rowmask, __nv_bfloat16* __restrict__ dq, int num_blocks,
                    long long seqlen, float sm_scale);

}  // namespace glue
}  // namespace vsa2t
