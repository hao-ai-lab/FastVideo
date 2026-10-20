// SPDX-License-Identifier: Apache-2.0
// block_sparse_kernel_blk128_sm121.cuh -- the VSA block-sparse attention forward for 128-token blocks on the
// NVIDIA GB10 (DGX Spark, sm_121a), where the data-center Blackwell kernel (tcgen05, tensor memory) cannot run.
//
// What it computes is block_sparse_ptx_blackwell_blk128_fwd's output with need_lse=False: BHSD bf16, head_dim
// 128; for each query tile i that selects q2k_num[i] key tiles q2k_idx[i, 0..count), every valid row r <
// variable_block_sizes[i] of the tile is softmax(q_r K^T / sqrt(128)) V over the first variable_block_sizes[j]
// keys of each selected tile j, and every other row (padding rows, tiles that select nothing) is written zero.
// Entries of q2k_idx past a tile's count are never read. No LSE: inference forwards only.
//
// How, on mma.sync (the GB10 has no wgmma, tcgen05 or tensor memory): one CTA of 512 threads per (query tile,
// batch * head), one CTA per SM (99 KB of shared memory).
//   - Two groups of eight warps, 16 query rows each: the QK group computes S = Q K^T and the online softmax and
//     publishes P (bf16) and the rescale factors through shared memory; the PV group accumulates O += P V. Each
//     PV warp waits for its QK partner half-way through the partner's next QK burst, so the two groups' mma
//     bursts overlap instead of finishing together.
//   - K and V stream in 64-key chunks through TMA (two 64x64 boxes, 128-byte swizzle) into separate rings;
//     the last warp to finish reading a slot refills it.
//   - Each CTA walks its selection starting (launch index mod SMs) / SMs of the way through, so the CTAs that
//     run at once read the key tiles they share at the same time while those are in the L2; a query tile that
//     selects more than half of the tiles walks from the front.
// The order of the walk changes no result beyond the rounding of the softmax's sums, and depends only on the
// launch index, the counts and the SM count, so every call is deterministic.
#pragma once

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <climits>
#include <cmath>
#include <cstdint>

namespace vsa_sm121 {

// What one call reads and writes. All pointers are device pointers to contiguous tensors; BH = batch * heads,
// T = tiles, S = T * 128 tokens.
struct Vsa128Args {
	const void* q;              // [BH, S, 128] bf16; a tile's padding rows may hold anything, they are not read
	const void* k;              // [BH, S, 128] bf16
	const void* v;              // [BH, S, 128] bf16
	void* out;                  // [BH, S, 128] bf16; every element is written
	const int32_t* q2k_idx;     // [BH, T, max_kv]: each query tile's selected key tiles; entries past its count are not read
	const int32_t* q2k_num;     // [BH, T]: how many key tiles each query tile selects
	const int32_t* block_sizes; // [T]: valid tokens of each tile, which are its first ones
	int batch_heads;            // BH
	int tiles;                  // T
	int max_kv;                 // the last dimension of q2k_idx
};


namespace ptx {

// Copies 16 bytes from global to shared memory without going through L1 (.cg). Only the first `src_bytes`
// are read; the rest of the 16 are zero-filled, so src_bytes == 0 writes zeros and reads nothing.
__device__ __forceinline__ void cp_async_16(uint32_t dst, const void* src, uint32_t src_bytes) {
	asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(dst), "l"(src), "r"(src_bytes) : "memory");
}

// Closes the group of the cp.async copies this thread issued since the last commit (an empty group if none).
// UNUSED by the current kernel, deliberately. The main loop publishes a stage's copies through that stage's
// mbarrier (`cp.async.mbarrier.arrive.noinc`), which waits for every thread's copies into the stage and makes
// them visible -- something a commit group cannot do, because a group is private to the thread that issued it
// and `cp.async.wait_group` below retires only this thread's own. Both wrappers stay because they are the right
// tool for a thread-private wait; re-introducing cp_async_wait() beside the loop's per-stage waits would not
// restore anything, it would only add a wait that cannot see the other warps' copies.
__device__ __forceinline__ void cp_async_commit() {
	asm volatile("cp.async.commit_group;\n" ::: "memory");
}

// Waits until at most `kPending` of this thread's most recent groups are still in flight. The copies are
// visible to the other threads of the CTA only after a barrier that follows the wait.
template <int kPending>
__device__ __forceinline__ void cp_async_wait() {
	asm volatile("cp.async.wait_group %0;\n" ::"n"(kPending) : "memory");
}

// Initialises the shared-memory mbarrier at `addr` to flip its phase every `count` arrivals.
__device__ __forceinline__ void mbar_init(uint32_t addr, uint32_t count) {
	asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" ::"r"(addr), "r"(count) : "memory");
}

// Makes the initialisations of this thread visible to the whole CTA before any thread arrives.
__device__ __forceinline__ void mbar_init_fence() {
	asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
}

// One arrival at the mbarrier `addr`.
__device__ __forceinline__ void mbar_arrive(uint32_t addr) {
	uint64_t state;
	asm volatile("mbarrier.arrive.shared::cta.b64 %0, [%1];\n" : "=l"(state) : "r"(addr) : "memory");
}

// The mbarrier `addr` is signalled when this thread's prior cp.async copies have completed and are visible to
// the threads that observe the barrier's phase flip. One arrival per thread per pass.
__device__ __forceinline__ void cp_async_mbar_arrive(uint32_t addr) {
	asm volatile("cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];\n" ::"r"(addr) : "memory");
}

// True once phase `phase` (0 or 1, alternating) of the mbarrier `addr` has completed; does not block.
__device__ __forceinline__ bool mbar_try_wait(uint32_t addr, uint32_t phase) {
	uint32_t done;
	asm volatile("{\n"
				 "\t.reg .pred p;\n"
				 "\tmbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n"
				 "\tselp.b32 %0, 1, 0, p;\n"
				 "}\n"
				 : "=r"(done)
				 : "r"(addr), "r"(phase)
				 : "memory");
	return done != 0;
}

// Waits until phase `phase` of the mbarrier `addr` has completed.
__device__ __forceinline__ void mbar_wait(uint32_t addr, uint32_t phase) {
	while (!mbar_try_wait(addr, phase)) {
	}
}

// Brings the 128-byte line holding `p` into L2.
__device__ __forceinline__ void prefetch_l2(const void* p) {
	asm volatile("prefetch.global.L2 [%0];\n" ::"l"(p));
}

// Loads four 8x8 matrices of 16-bit elements: lanes 8i..8i+7 give the addresses of the 8 rows (16 bytes each)
// of matrix i, and r[i] receives, in every lane, elements (lane / 4, 2 * (lane % 4) + {0, 1}) of matrix i.
__device__ __forceinline__ void ldmatrix_x4(uint32_t (&r)[4], uint32_t addr) {
	asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
				 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
				 : "r"(addr));
}

// As ldmatrix_x4, with each matrix transposed: r[i] receives elements (2 * (lane % 4) + {0, 1}, lane / 4).
__device__ __forceinline__ void ldmatrix_x4_trans(uint32_t (&r)[4], uint32_t addr) {
	asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
				 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
				 : "r"(addr));
}

// d += a * b for a 16x16 bf16 A (row-major), a 16x8 bf16 B (column-major) and a 16x8 fp32 D, held as the
// warp-wide fragments of mma.m16n8k16. With g = lane / 4 and c = 2 * (lane % 4):
//   a: {A[g][c..c+1], A[g+8][c..c+1], A[g][c+8..c+9], A[g+8][c+8..c+9]}, two bf16 per register, lower k first
//   b: {B[c..c+1][g], B[c+8..c+9][g]}
//   d: {D[g][c], D[g][c+1], D[g+8][c], D[g+8][c+1]}
__device__ __forceinline__ void mma_bf16_16816(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
	asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
		"{%0, %1, %2, %3};\n"
		: "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
		: "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// Two floats rounded to nearest-even bf16 and packed, `lo` in the lower half: the element with the lower index.
__device__ __forceinline__ uint32_t pack_bf16x2(float lo, float hi) {
	uint32_t r;
	asm("cvt.rn.bf16x2.f32 %0, %1, %2;\n" : "=r"(r) : "f"(hi), "f"(lo));
	return r;
}

// 2^x by the special function unit (relative error about 2^-22; exp2(-inf) = +0).
__device__ __forceinline__ float exp2_approx(float x) {
	float y;
	asm("ex2.approx.ftz.f32 %0, %1;\n" : "=f"(y) : "f"(x));
	return y;
}

}  // namespace ptx

namespace split128 {
using bf16 = __nv_bfloat16;
constexpr int Tile = 128, Dim = 128, Chunk = 64;
constexpr int PhaseLead = 8;
constexpr int GroupThreads = 256, Threads = 512;
constexpr int OperandBytes = 16384, BoxBytes = 8192;
constexpr int KOffset = 0, VOffset = 32768, POffset = 81920;
constexpr int AlphaOffset = 98304;
constexpr int SumOffset = AlphaOffset + 512;
constexpr int BarOffset = SumOffset + 512;
constexpr int CounterOffset = BarOffset + 104; // 5 operand-full + 8 P-full
constexpr int SharedBytes = CounterOffset + 24; // 5 counters, 8-byte rounding
static_assert(SharedBytes <= 99 * 1024, "GB10 shared-memory budget");
constexpr float ScaleLog2 = 1.4426950408889634f * 0.088388347648318441f;

__device__ __forceinline__ int clamp_int(int a, int lo, int hi) { return min(max(a,lo),hi); }
// Q and output retain the original 256-byte-row swizzle. K/V alone use TMA boxes.
__device__ __forceinline__ uint32_t swizzle(int row, int unit) {
    return row * 256 + ((unit ^ (row & 7)) << 4);
}
__device__ __forceinline__ uint32_t p_swizzle(int row, int unit) {
    return row * 128 + ((unit ^ (row & 7)) << 4);
}
__device__ __forceinline__ uint32_t lane_offset(int key, int u0, int j) {
    return 128 * key + 16 * ((2 * j + u0) ^ (key & 7));
}

// The walk of one CTA's selection. It reads entries start..count-1, then
// 0..start-1: every entry once, in a rotated order (see `walk_start`).
struct Chunks {
    const int* selected;
    const int* sizes;
    // Low eight bits: tile valid count; bit eight: second half; bits nine
    // and up: the entry the walk starts at. Keep the replicated walker one
    // register smaller alongside the PV accumulator.
    int count, tiles, pos, tile, state;
    __device__ Chunks(const int* s, const int* b, int n, int t, int start)
        : selected(s), sizes(b), count(n), tiles(t), pos(0), tile(0), state(256 | (start << 9)) {}
    __device__ __forceinline__ bool next() {
        const int now = state & 511;
        if (now > Chunk && now < 256) { state |= 256; return true; }
        while (pos < count) {
            int at = pos++ + (state >> 9);
            if (at >= count) at -= count;
            tile = __ldg(selected + at);
            const int valid = static_cast<unsigned>(tile) < static_cast<unsigned>(tiles)
                ? clamp_int(__ldg(sizes + tile), 0, Tile) : 0;
            state = (state & ~511) | valid;
            if (valid) return true;
        }
        return false;
    }
    __device__ __forceinline__ int keys() const { return min((state & 255) - (((state >> 8) & 1) * Chunk), Chunk); }
    __device__ __forceinline__ int first_row() const { return tile * Tile + (((state >> 8) & 1) * Chunk); }
};

// Where a CTA's walk starts. CTAs start in launch order, `slots` at a time
// (one per SM), and each walks its selection over its whole life. Started
// at entry 0, the CTAs that run together are spread over every phase of
// their walks; the n-th of each `slots` consecutive CTAs instead starts
// n/slots of the way through its own selection, so that the CTAs running
// together are at about the same phase of their walks and read the key
// tiles they share at about the same time, while those are in the L2.
// A CTA that selects more than half of the tiles walks from entry 0: those
// start together and share one stream in order (phasing them measured
// -13 % at the long rows, PR41). The order changes no result beyond the
// rounding of the online softmax's sums, and depends only on the launch
// index, the count and the SM count, so a call is deterministic.
__device__ __forceinline__ int walk_start(int64_t launch, int count, int tiles, int slots) {
    return 2 * count <= tiles ? static_cast<int>((launch % slots) * count / slots) : 0;
}

__device__ __forceinline__ void tma_load_2d(uint32_t dst, const void* map,
                                          int col, int row, uint32_t full) {
    asm volatile("cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes "
                 "[%0], [%1, {%2, %3}], [%4];"
                 :: "r"(dst), "l"(map), "r"(col), "r"(row), "r"(full) : "memory");
}
// Called by one lane only. The full barrier has arrival count one and 16 KB
// expected transactions (two 64x64 boxes); masks still use the valid key count.
__device__ __forceinline__ void load_operand(uint32_t dst, const void* map,
                                            const Chunks& c, int head_row, uint32_t full) {
    // Host checks BH*S <= INT32_MAX; tensor coordinates are rows, never elements.
    const int row = head_row + c.first_row();
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    asm volatile("{ .reg .b64 st; mbarrier.arrive.expect_tx.shared::cta.b64 st, [%0], %1; }"
                 :: "r"(full), "r"(OperandBytes) : "memory");
    tma_load_2d(dst, map, 0, row, full);
    tma_load_2d(dst + BoxBytes, map, 64, row, full);
}
// Every warp finishes all of its reads before its one counter arrival. The
// CTA-scoped acq_rel RMW chains those reads to the elected warp's TMA issue.
// All lanes receive the result so the last-warp branch is uniform.
__device__ __forceinline__ bool last_reader(uint32_t counter, int lane) {
    __syncwarp();
    uint32_t old = 0;
    if (lane == 0) {
        asm volatile("atom.acq_rel.cta.shared::cta.add.u32 %0, [%1], 1;"
                     : "=r"(old) : "r"(counter) : "memory");
    }
    old = __shfl_sync(0xffffffffu, old, 0);
    return old == 7;
}
__device__ __forceinline__ void reset_counter(uint32_t counter) {
    asm volatile("st.shared.u32 [%0], 0;" :: "r"(counter) : "memory");
}
__device__ __forceinline__ void load_q(uint32_t dst, const bf16* q, int rows, int tid) {
#pragma unroll
    for (int i=0; i<4; ++i) {
        const int u = tid + i * Threads;
        const int row = u/16, unit = u%16;
        const bool valid = row < rows;
        ptx::cp_async_16(dst+swizzle(row,unit), q+(valid?row:0)*Dim+unit*8, valid?16:0);
    }
}
__device__ __forceinline__ void q_fragments(uint32_t (&qf)[8][4], uint32_t base, int warp, int lane) {
    const int row = warp*16+(lane&7)+((lane>>3)&1)*8;
    const int unit = lane>>4;
#pragma unroll
    for (int kk=0; kk<8; ++kk) ptx::ldmatrix_x4(qf[kk],base+swizzle(row,2*kk+unit));
}
template<int Begin,int End>
__device__ __forceinline__ void scores(float (&s)[8][4], const uint32_t (&qf)[8][4], const uint32_t (&a)[4]) {
    if constexpr (Begin==0) {
#pragma unroll
        for (int j=0; j<8; ++j) s[j][0]=s[j][1]=s[j][2]=s[j][3]=0.f;
    }
#pragma unroll
    for (int kk=Begin; kk<End; ++kk) {
#pragma unroll
        for (int nn=0; nn<4; ++nn) {
            uint32_t b[4];
            ptx::ldmatrix_x4(b,a[kk&3]+2048*nn+BoxBytes*(kk>>2));
            ptx::mma_bf16_16816(s[2*nn],qf[kk],b[0],b[1]);
            ptx::mma_bf16_16816(s[2*nn+1],qf[kk],b[2],b[3]);
        }
    }
}
__device__ __forceinline__ void mask(float (&s)[8][4], int keys, int lane) {
    const int col = 2*(lane&3);
#pragma unroll
    for (int j=0; j<8; ++j) {
#pragma unroll
        for (int e=0; e<2; ++e) {
            if (8*j+col+e>=keys) s[j][e]=s[j][2+e]=-INFINITY;
        }
    }
}
__device__ __forceinline__ void softmax(float (&s)[8][4], float (&m)[2], float (&l)[2], float (&alpha)[2]) {
    float mx[2]={m[0],m[1]};
#pragma unroll
    for (int j=0; j<8; ++j) {
        mx[0]=fmaxf(mx[0],fmaxf(s[j][0],s[j][1]));
        mx[1]=fmaxf(mx[1],fmaxf(s[j][2],s[j][3]));
    }
#pragma unroll
    for (int r=0; r<2; ++r) {
        mx[r]=fmaxf(mx[r],__shfl_xor_sync(0xffffffffu,mx[r],1));
        mx[r]=fmaxf(mx[r],__shfl_xor_sync(0xffffffffu,mx[r],2));
    }
    float bias[2];
#pragma unroll
    for (int r=0; r<2; ++r) {
        alpha[r]=ptx::exp2_approx((m[r]-mx[r])*ScaleLog2);
        bias[r]=-mx[r]*ScaleLog2;
        m[r]=mx[r];
    }
    float sum[2]={0.f,0.f};
#pragma unroll
    for (int j=0; j<8; ++j) {
        s[j][0]=ptx::exp2_approx(fmaf(s[j][0],ScaleLog2,bias[0]));
        s[j][1]=ptx::exp2_approx(fmaf(s[j][1],ScaleLog2,bias[0]));
        s[j][2]=ptx::exp2_approx(fmaf(s[j][2],ScaleLog2,bias[1]));
        s[j][3]=ptx::exp2_approx(fmaf(s[j][3],ScaleLog2,bias[1]));
        sum[0]+=s[j][0]+s[j][1];
        sum[1]+=s[j][2]+s[j][3];
    }
#pragma unroll
    for (int r=0; r<2; ++r) l[r]=l[r]*alpha[r]+sum[r];
}
__device__ __forceinline__ void store_p(uint32_t dst, const float (&p)[8][4], int warp, int lane) {
    const int row=warp*16+(lane&7)+((lane>>3)&1)*8;
    const int unit=lane>>4;
#pragma unroll
    for (int kk=0; kk<4; ++kk) {
        const uint32_t a0=ptx::pack_bf16x2(p[2*kk][0],p[2*kk][1]);
        const uint32_t a1=ptx::pack_bf16x2(p[2*kk][2],p[2*kk][3]);
        const uint32_t a2=ptx::pack_bf16x2(p[2*kk+1][0],p[2*kk+1][1]);
        const uint32_t a3=ptx::pack_bf16x2(p[2*kk+1][2],p[2*kk+1][3]);
        const uint32_t addr=dst+p_swizzle(row,2*kk+unit);
        asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};"
                     :: "r"(addr), "r"(a0), "r"(a1), "r"(a2), "r"(a3) : "memory");
    }
}
__device__ __forceinline__ void pv(float (&o)[16][4], const uint32_t (&p)[4][4], const uint32_t (&v_addr)[4]) {
#pragma unroll
    for (int kk=0; kk<4; ++kk) {
#pragma unroll
        for (int dd=0; dd<8; ++dd) {
            uint32_t b[4];
            ptx::ldmatrix_x4_trans(b,v_addr[dd&3]+2048*kk+BoxBytes*(dd>>2));
            ptx::mma_bf16_16816(o[2*dd],p[kk],b[0],b[1]);
            ptx::mma_bf16_16816(o[2*dd+1],p[kk],b[2],b[3]);
        }
    }
}
__device__ __forceinline__ void output(bf16* out, uint8_t* staging, const float (&o)[16][4],
                                      const float* sums, int rows, int warp, int lane) {
    const int row0=warp*16+(lane>>2);
    bool keep[2]; float inv[2];
#pragma unroll
    for (int r=0; r<2; ++r) {
        const float l=sums[row0+8*r];
        keep[r]=row0+8*r<rows && l>0.f;
        inv[r]=keep[r]?1.f/l:0.f;
    }
#pragma unroll
    for (int d=0; d<16; ++d) {
#pragma unroll
        for (int r=0; r<2; ++r) {
            const uint32_t val=keep[r]?ptx::pack_bf16x2(o[d][2*r]*inv[r],o[d][2*r+1]*inv[r]):0u;
            *reinterpret_cast<uint32_t*>(staging+swizzle(row0+8*r,d)+4*(lane&3))=val;
        }
    }
    __syncwarp();
#pragma unroll
    for (int i=0; i<8; ++i) {
        const int row=warp*16+2*i+(lane>>4), unit=lane&15;
        const uint4 val=*reinterpret_cast<const uint4*>(staging+swizzle(row,unit));
        *reinterpret_cast<uint4*>(out+row*Dim+unit*8)=val;
    }
}

__device__ __forceinline__ void pair_sync(int warp) {
    asm volatile("bar.sync %0, 64;" :: "r"(warp+1) : "memory");
}
// slots: the CTAs that run at once, one per SM; it sets the walks' phases.
__global__ __launch_bounds__(Threads,1) void forward(const Vsa128Args args,
        const __grid_constant__ CUtensorMap kmap, const __grid_constant__ CUtensorMap vmap, const int slots) {
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ == 1210
    // 128B TMA swizzle repeats over 1024B; all slot starts are multiples of 16KB.
    extern __shared__ __align__(1024) uint8_t smem[];
    const uint32_t base=static_cast<uint32_t>(__cvta_generic_to_shared(smem));
    const int tid=threadIdx.x, lane=tid&31, local_tid=tid&255, warp=local_tid>>5;
    const int tile=blockIdx.x, bh=blockIdx.y, tiles=args.tiles;
    const int head_row=bh*tiles*Tile; // host-checked TMA row coordinate range
    const int64_t qo=(int64_t(head_row)+int64_t(tile)*Tile)*Dim;
    const int64_t query=int64_t(bh)*tiles+tile;
    const int rows=clamp_int(__ldg(args.block_sizes+tile),0,Tile);
    const int count=clamp_int(__ldg(args.q2k_num+query),0,args.max_kv);
    bf16* out=static_cast<bf16*>(args.out)+qo;
    if (rows==0 || count==0) {
#pragma unroll
        for (int i=0;i<4;++i) reinterpret_cast<uint4*>(out)[tid+i*Threads]=make_uint4(0,0,0,0);
        return;
    }
    const int* selected=args.q2k_idx+query*args.max_kv;
    const uint32_t bar=base+BarOffset, counters=base+CounterOffset;
    const auto operand_full=[bar](int op,int stage) { return bar+(op*2+stage)*8; };
    const auto operand_counter=[counters](int op,int stage) { return counters+(op*2+stage)*4; };
    const uint32_t p_full=bar+40+warp*8;
    if (tid==0) {
#pragma unroll
        for (int i=0;i<5;++i) {
            ptx::mbar_init(bar+i*8,1);
            reset_counter(counters+i*4);
        }
#pragma unroll
        for (int i=0;i<8;++i) ptx::mbar_init(bar+40+i*8,32);
        ptx::mbar_init_fence();
    }
    // Q aliases both K slots only in the prologue. Its original cp.async and
    // swizzle are unchanged; all QK warps rendezvous after capturing Q.
    load_q(base+KOffset,static_cast<const bf16*>(args.q)+qo,rows,tid);
    ptx::cp_async_commit();
    ptx::cp_async_wait<0>();
    __syncthreads();
    const bool active=warp*16<rows;
    // The grid is (tiles, heads) with tiles fastest, so `query` is the launch index.
    Chunks fill(selected,args.block_sizes,count,tiles,walk_start(query,count,tiles,slots));

    if (tid<GroupThreads) {
        uint32_t qf[8][4];
        if (active) q_fragments(qf,base+KOffset,warp,lane);
        asm volatile("bar.sync 9, 256;" ::: "memory");
        int pending=0, keys0=0, keys1=0;
        if (fill.next()) {
            keys0=fill.keys(); ++pending;
            if (local_tid==0) load_operand(base+KOffset,&kmap,fill,head_row,operand_full(0,0));
        }
        if (fill.next()) {
            keys1=fill.keys(); ++pending;
            if (local_tid==0) load_operand(base+KOffset+OperandBytes,&kmap,fill,head_row,operand_full(0,1));
        }
        float m[2]={-INFINITY,-INFINITY}, l[2]={0.f,0.f};
        const int k_key=(lane&7)+((lane>>4)&1)*8, k_u0=(lane>>3)&1;
        uint32_t klane[4];
#pragma unroll
        for (int j=0;j<4;++j) klane[j]=lane_offset(k_key,k_u0,j);
        int iter=0;
        while (pending) {
            const int stage=iter&1, phase=(iter>>1)&1;
            ptx::mbar_wait(operand_full(0,stage),phase);
            float s[8][4];
            uint32_t kaddr[4];
#pragma unroll
            for (int j=0;j<4;++j) kaddr[j]=base+KOffset+stage*OperandBytes+klane[j];
            if (active) scores<0,PhaseLead>(s,qf,kaddr);
            if (iter) pair_sync(warp);
            if (active) scores<PhaseLead,8>(s,qf,kaddr);
            float alpha[2];
            if (active) {
                const int keys=stage?keys1:keys0;
                if (keys<Chunk) mask(s,keys,lane);
                softmax(s,m,l,alpha);
                store_p(base+POffset,s,warp,lane);
                if ((lane&3)==0) {
                    float* a=reinterpret_cast<float*>(smem+AlphaOffset);
                    a[warp*16+(lane>>2)]=alpha[0];
                    a[warp*16+(lane>>2)+8]=alpha[1];
                }
            }
            ptx::mbar_arrive(p_full);
            // The fill walkers advance identically in every lane and warp.
            // On iteration i the just-consumed K slot takes chunk i+2.
            if (fill.next()) {
                const int keys=fill.keys();
                if (stage) keys1=keys; else keys0=keys;
                const uint32_t counter=operand_counter(0,stage);
                if (last_reader(counter,lane)) {
                    if (lane==0) {
                        reset_counter(counter);
                        load_operand(base+KOffset+stage*OperandBytes,&kmap,fill,head_row,operand_full(0,stage));
                    }
                }
            } else --pending;
            ++iter;
        }
        if (iter) pair_sync(warp);
        float* sums=reinterpret_cast<float*>(smem+SumOffset);
#pragma unroll
        for (int r=0;r<2;++r) {
            float sum=l[r];
            sum+=__shfl_xor_sync(0xffffffffu,sum,1);
            sum+=__shfl_xor_sync(0xffffffffu,sum,2);
            if ((lane&3)==0) sums[warp*16+(lane>>2)+8*r]=sum;
        }
        // Every issued operand has been consumed; this publishes sums and
        // prevents PV output staging from overwriting QK's K alias too early.
        asm volatile("bar.sync 0, 512;" ::: "memory");
    } else {
        int pending=0;
#pragma unroll
        for (int s=0;s<3;++s) {
            if (fill.next()) {
                ++pending;
                if (local_tid==0) load_operand(base+VOffset+s*OperandBytes,&vmap,fill,head_row,operand_full(1,s));
            }
        }
        float o[16][4];
#pragma unroll
        for (int d=0;d<16;++d) o[d][0]=o[d][1]=o[d][2]=o[d][3]=0.f;
        const int v_key=(lane&7)+((lane>>3)&1)*8, v_u0=lane>>4;
        uint32_t vlane[4];
#pragma unroll
        for (int j=0;j<4;++j) vlane[j]=lane_offset(v_key,v_u0,j);
        const uint32_t plane=p_swizzle(warp*16+v_key,v_u0);
        int iter=0, stage=0;
        uint32_t phase=0;
        while (pending) {
            ptx::mbar_wait(operand_full(1,stage),phase);
            ptx::mbar_wait(p_full,iter&1);
            uint32_t pf[4][4];
            float a0=0.f,a1=0.f;
            if (active) {
                const float* a=reinterpret_cast<const float*>(smem+AlphaOffset);
                a0=a[warp*16+(lane>>2)]; a1=a[warp*16+(lane>>2)+8];
#pragma unroll
                for (int kk=0;kk<4;++kk) ptx::ldmatrix_x4(pf[kk],base+POffset+(plane^(32*kk)));
            }
            pair_sync(warp);
            if (active) {
#pragma unroll
                for (int d=0;d<16;++d) {
                    o[d][0]*=a0; o[d][1]*=a0;
                    o[d][2]*=a1; o[d][3]*=a1;
                }
                uint32_t vaddr[4];
#pragma unroll
                for (int j=0;j<4;++j) vaddr[j]=base+VOffset+stage*OperandBytes+vlane[j];
                pv(o,pf,vaddr);
            }
            // All three V slots were prefetched. Refill exactly the slot just
            // read with i+3, not A3's i+2 slot (stage+2)%3.
            if (fill.next()) {
                const uint32_t counter=operand_counter(1,stage);
                if (last_reader(counter,lane)) {
                    if (lane==0) {
                        reset_counter(counter);
                        load_operand(base+VOffset+stage*OperandBytes,&vmap,fill,head_row,operand_full(1,stage));
                    }
                }
            } else --pending;
            if (++stage==3) { stage=0; phase^=1; }
            ++iter;
        }
        asm volatile("bar.sync 0, 512;" ::: "memory");
        output(out,smem,o,reinterpret_cast<const float*>(smem+SumOffset),rows,warp,lane);
    }
#endif  // sm_121: every other device pass compiles an empty kernel
}

CUresult encode_map(CUtensorMap* map, const void* ptr, int64_t rows) {
    const cuuint64_t dims[2]={Dim,static_cast<cuuint64_t>(rows)};
    const cuuint64_t strides[1]={Dim*2};
    const cuuint32_t box[2]={64,64}, elem_strides[2]={1,1};
    return cuTensorMapEncodeTiled(map,CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,2,const_cast<void*>(ptr),
        dims,strides,box,elem_strides,CU_TENSOR_MAP_INTERLEAVE_NONE,CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_L2_256B,CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
}
} // namespace split128

inline cudaError_t launch(const Vsa128Args& args,cudaStream_t stream) {
    if (args.batch_heads<=0 || args.tiles<=0) return cudaSuccess;
    const int64_t rows=int64_t(args.batch_heads)*args.tiles*split128::Tile;
    if (rows>INT32_MAX) return cudaErrorInvalidValue;
    // The walker keeps its start entry (< count <= tiles) in bits 9..30 of an int.
    if (args.tiles>=(1<<22) || args.max_kv>=(1<<22)) return cudaErrorInvalidValue;
    CUtensorMap kmap,vmap;
    if (split128::encode_map(&kmap,args.k,rows)!=CUDA_SUCCESS ||
        split128::encode_map(&vmap,args.v,rows)!=CUDA_SUCCESS) return cudaErrorInvalidValue;
    cudaError_t err=cudaFuncSetAttribute(split128::forward,cudaFuncAttributeMaxDynamicSharedMemorySize,split128::SharedBytes);
    if (err!=cudaSuccess) return err;
    // One CTA per SM fits (99 KB of shared memory each): the SMs are the
    // CTAs that run at once. A device property, read once per device.
    int device=0;
    err=cudaGetDevice(&device);
    if (err!=cudaSuccess) return err;
    static int slots_device=-1, slots=1;
    if (device!=slots_device) {
        int sms=0;
        err=cudaDeviceGetAttribute(&sms,cudaDevAttrMultiProcessorCount,device);
        if (err!=cudaSuccess) return err;
        slots=sms>0 ? sms : 1;
        slots_device=device;
    }
    split128::forward<<<dim3(args.tiles,args.batch_heads),split128::Threads,split128::SharedBytes,stream>>>(args,kmap,vmap,slots);
    return cudaGetLastError();
}
}  // namespace vsa_sm121
