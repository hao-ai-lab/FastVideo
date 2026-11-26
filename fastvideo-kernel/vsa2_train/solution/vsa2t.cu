// vsa2t.cu -- the host side of the VSA2 training step: the geometry, the scratch layouts, the TMA tensor maps and the
// launches of each half's kernels (vsa2t.h), in stream order:
//
// forward:  fwd_prep (block and window means, row and window masks) -> fwd_coarse_tc (Oc, one flash-style launch)
//           -> router_kernel<2> (pooled scores, one sweep), <0>, <1> (two passes, flagged tiles only)
//           -> top-K (router_hist x3, router_tie_pin, router_select: selection)
//           -> fwd_lists, fwd_order (q2k lists; tiles by count, heaviest first)
//           -> fwd_fine (out and lse of the video rows, partials of the text chunks)
//           -> fwd_merge (out and lse of the text rows)
// backward: bwd_prep (block means, row masks) -> bwd_coarse_tc (Oc and its rows' lse2 again)
//           -> bwd_rows (D, lse2, dgate, dco, Dc, dqaccum zeroed) -> bwd_coarse_grad_tc (dqc, dkc, dvc, one launch)
// The coarse kernels (coarse_tc.cuh) take the block means as three bf16 components each (a row whose largest entry is
// below 2^-20 lifted by a power of 2 first), and every product as the eight component products i + j <= 3 accumulated
// in fp32.
//           -> bwd_lists, bwd_order (k2q lists from `selection`) -> bwd_fine (dk, dv; dq partials)
//           -> bwd_post (dq)
// `bwd_rows` and `bwd_post` live in glue.cu; they keep `bwd_` in their names so the stage breakdown still reads them
// as the backward's.
#include <cuda.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

#include "bwd.cuh"
#include "coarse_tc.cuh"
#include "common.cuh"
#include "fwd.cuh"
#include "glue.cuh"
#include "prep.cuh"
#include "router.cuh"
#include "topk.cuh"
#include "vsa2t.h"

// The pin `glue.cuh` cannot carry: its `bwd_post_kernel` undoes the drain-native layout `bwd.cuh` writes
// (`[slice of DQ_COLS columns][v4][row][4]` within a BLOCK-row block), so the two must agree. It lives here because
// this is the one translation unit that includes both headers.
static_assert(vsa2t::glue::kDqCols == vsa2t::bwd::DQ_COLS && vsa2t::glue::kBlock == vsa2t::bwd::BLOCK,
              "the drain-native layout is bwd.cuh's");
// The head dimension is the same kind of copy and was the one left unpinned: `bwd_post` indexes dqaccum and dq by
// glue::kHeadDim while the kernel that wrote that memory used bwd::HEAD_DIM, so a disagreement is a silent reshape
// rather than a compile error.
static_assert(vsa2t::glue::kHeadDim == vsa2t::bwd::HEAD_DIM, "the head dimension is bwd.cuh's");

namespace vsa2t {

namespace {

void check(cudaError_t e, const char* what) {
  if (e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}

size_t align256(size_t x) { return (x + 255) & ~size_t(255); }

// A 4-D tiled map of 2-byte elements, 128-byte swizzle, out-of-bounds elements read as zeros.
CUtensorMap make_map(const void* base, CUtensorMapDataType dtype, const uint64_t (&dims)[4], const uint64_t (&strides)[3],
                     const uint32_t (&box)[4]) {
  alignas(64) CUtensorMap m;
  const uint32_t es[4] = {1, 1, 1, 1};
  const CUresult r = cuTensorMapEncodeTiled(&m, dtype, 4, const_cast<void*>(base), dims, strides, box, es,
                                            CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
                                            CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  if (r != CUDA_SUCCESS) throw std::runtime_error("cuTensorMapEncodeTiled failed: " + std::to_string((int)r));
  return m;
}

// [heads][rows][128] as (dim0, row, subtile, head), boxes of 128 rows and both 64-column subtiles (one after the
// other in shared memory).
// ** THE ELEMENT TYPE IS A PARAMETER OF THE BUFFER, NOT OF THIS LAYOUT. **  This builder is shared by buffers
// whose types differ and must not change with them: the fine forward's K and V are __nv_bfloat16 (`k16`/`v16`)
// and the backward uses subtile_map.  So this form stays BFLOAT16 and the fp16 window
// means get their own builder below -- a name that says which format it makes, so that the NEXT caller has to
// choose instead of inheriting.  A silent 2-byte mismatch here is INERT in TMA (no conversion, same element
// size) and therefore produces NO symptom until someone trusts the declared type -- which is why the two
// builders are separate rather than one function with a flag.
CUtensorMap rows_map(const void* base, int heads, long long rows) {
  const uint64_t dims[4] = {64, (uint64_t)rows, 2, (uint64_t)heads};
  const uint64_t strides[3] = {256, 128, (uint64_t)rows * 256};
  const uint32_t box[4] = {64, 128, 2, 1};
  return make_map(base, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, dims, strides, box);
}

// The router's window means qw/kw, AND ONLY THOSE (the two rows_map_f16 call sites below): identical
// dims/strides/box to rows_map -- every 2-byte format has the same byte strides and the same 128B swizzle --
// with the element type that matches the idesc at router.cuh:131.  fp16 and not bf16 because bf16's 8
// mantissa bits reorder the top-K (issue I23).
//
// THE PRICE OF FP16 IS RANGE, and it is paid HERE: bf16's ceiling was ~2^128 and fp16's is 2^15.99, so this
// buffer is the overflow site on the forward path (the coarse branch stages its block means as three bf16
// components, which keep fp32's range).  Any fix that bounds the operands must cover qw/kw.
CUtensorMap rows_map_f16(const void* base, int heads, long long rows) {
  const uint64_t dims[4] = {64, (uint64_t)rows, 2, (uint64_t)heads};
  const uint64_t strides[3] = {256, 128, (uint64_t)rows * 256};
  const uint32_t box[4] = {64, 128, 2, 1};
  return make_map(base, CU_TENSOR_MAP_DATA_TYPE_FLOAT16, dims, strides, box);
}

// [heads][rows][128] as (dim0, head, row, subtile): the fine forward's Q and output tiles.
CUtensorMap head_token_map(const void* base, int heads, long long rows) {
  const uint64_t dims[4] = {64, (uint64_t)heads, (uint64_t)rows, 2};
  const uint64_t strides[3] = {(uint64_t)rows * 256, 256, 128};
  const uint32_t box[4] = {64, 1, 128, 2};
  return make_map(base, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, dims, strides, box);
}

// [heads][rows][128] as (dim0, row, subtile, head), one 64-column subtile of 128 rows per box: the fine backward's.
CUtensorMap subtile_map(const void* base, int heads, long long rows) {
  const uint64_t dims[4] = {64, (uint64_t)rows, 2, (uint64_t)heads};
  const uint64_t strides[3] = {256, 128, (uint64_t)rows * 256};
  const uint32_t box[4] = {64, 128, 1, 1};
  return make_map(base, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, dims, strides, box);
}

// A bump allocator over one scratch allocation (256-byte aligned pieces).
struct Carver {
  uint8_t* at;
  size_t used = 0;
  template <class T>
  T* take(size_t count) {
    T* p = reinterpret_cast<T*>(at ? at + used : nullptr);
    used += align256(count * sizeof(T));
    return p;
  }
};

constexpr int kChunks = 64;  // CTAs per head of the top-K's passes over the scores

// The row stride of the coarse rows' per-row statistics (lse2c, Dc): N rounded up to the coarse backward's 32-column
// chunks (glue::bwd_rows_kernel spells the same expression).
long long coarse_npad(const Geo& g) { return ((long long)g.N + 31) / 32 * 32; }

// A split operand's planes (three bf16 components, as the coarse kernels' tile planes) and rows' lifts (prep.cuh
// block_means, glue.cu bwd_rows).
void carve_one_split(Carver& c, const Geo& g, prep::SplitOut& o) {
  o.npad = coarse_tc::plane_rows(g.N);
  o.planes = c.take<__nv_bfloat16>((size_t)g.heads * coarse_tc::kPlanes * o.npad * 64);
  o.e = c.take<int>((size_t)g.heads * g.N);
}
// ... for the block means of q, k and v.
void carve_split(Carver& c, const Geo& g, prep::SplitOut& q, prep::SplitOut& k, prep::SplitOut& v) {
  carve_one_split(c, g, q);
  carve_one_split(c, g, k);
  carve_one_split(c, g, v);
}

// The forward's scratch, in carve order.
struct FwdScratch {
  uint32_t* rowmask;  // [N][4]
  uint16_t* winmask;  // [Nv]
  prep::SplitOut sq, sk, sv;  // the block means as bf16 components and row lifts (prep.cuh, coarse_tc.cuh)
  float* cpart;               // the coarse kernel's split partials and their row statistics (coarse_tc.cuh)
  float* cpstat;
  __half* qw;  // [heads][Nv * 16][128] window means
  __half* kw;
  float* oc;          // [heads][N][128] the coarse output
  float* lse2c;       // [heads][npad] the coarse rows' log2-sum-exp2 (written, unused by the forward)
  float* qnorm;       // [heads][Nv * 16] norms of the window means
  float* knorm;
  float* u;           // [heads][rtiles * 8][upitch] the router's one-pass sums
  int* flags;         // [heads][rtiles] the router's query tiles left to its two passes
  float* rstat;       // [heads][Nv * 16] router lse per query window
  float* scores;      // [heads][Nv][Nv]
  float* lpart;       // [ksplit][heads][Nv * kWindows] the router's row sums, one slice of the key range each
  uint32_t* tiecnt;   // [heads][kChunks]
  uint32_t* zeroed;   // what fwd_prep zeroes: hist, done, then ohist
  uint32_t* hist;     // [heads][3][2048] radix histograms
  uint32_t* done;     // [heads][3] the CTAs that added to each histogram
  uint32_t* ohist;    // [heads][N + 1] the q2k counts' histogram, then their positions
  uint32_t* state;    // [heads][8]
  int* q2k_idx;       // [heads * N][N]
  int* q2k_num;       // [heads * N]
  int* vt_cnt;        // [heads][vtpad] the fine forward's tiles' counts
  int* vorder;        // [heads][vtpad] the tiles in processing order
  float* part_o;      // [heads][P * tchunks][128][128] the text chunks' partial outputs
  float* part_lse;    // [heads][P * tchunks][128]
  size_t zero_words;
};

long long upitch(const Geo& g) { return ((long long)g.Nv * kWindows + 31) / 32 * 32; }

// The SM count, for the router's fill model: router::ksplit_for splits the sweep's key range only when the extra
// CTAs buy more waves than the prologue they pay for, and whether they do is a property of the machine, not of a
// workload.  Cached: this is one attribute read for the life of the module.
int sms() {
  static int n = 0;
  if (n == 0) {
    int v = 0;
    if (cudaDeviceGetAttribute(&v, cudaDevAttrMultiProcessorCount, 0) != cudaSuccess || v <= 0) v = 132;
    n = v;
  }
  return n;
}

FwdScratch carve_fwd(const Geo& g, void* base, size_t* bytes) {
  Carver c{static_cast<uint8_t*>(base)};
  FwdScratch s;
  const size_t hn = (size_t)g.heads * g.N;
  s.rowmask = c.take<uint32_t>((size_t)g.N * 4);
  s.winmask = c.take<uint16_t>((size_t)g.Nv);
  carve_split(c, g, s.sq, s.sk, s.sv);
  s.cpart = c.take<float>(coarse_tc::part_floats(g.N, g.heads));
  s.cpstat = c.take<float>(coarse_tc::pstat_floats(g.N, g.heads));
  s.qw = c.take<__half>((size_t)g.heads * g.Nv * kWindows * kDim);
  s.kw = c.take<__half>((size_t)g.heads * g.Nv * kWindows * kDim);
  s.oc = c.take<float>(hn * kDim);
  s.lse2c = c.take<float>((size_t)g.heads * coarse_npad(g));
  s.qnorm = c.take<float>((size_t)g.heads * g.Nv * kWindows);
  s.knorm = c.take<float>((size_t)g.heads * g.Nv * kWindows);
  s.u = c.take<float>((size_t)g.heads * g.rtiles * 8 * upitch(g));
  s.flags = c.take<int>((size_t)g.heads * g.rtiles);
  s.rstat = c.take<float>((size_t)g.heads * g.Nv * kWindows);
  s.scores = c.take<float>((size_t)g.heads * g.Nv * g.Nv);
  s.lpart = c.take<float>((size_t)router::ksplit_for(g.rtiles, g.heads, sms()) * g.heads * g.Nv * kWindows);
  s.tiecnt = c.take<uint32_t>((size_t)g.heads * kChunks);
  const size_t hist_words = (size_t)g.heads * 3 * topk::kBins, done_words = (size_t)g.heads * 3,
               ohist_words = (size_t)g.heads * (g.N + 1);
  s.zero_words = hist_words + done_words + ohist_words;
  s.zeroed = c.take<uint32_t>(s.zero_words);
  s.hist = s.zeroed;
  s.done = s.zeroed ? s.zeroed + hist_words : nullptr;
  s.ohist = s.zeroed ? s.zeroed + hist_words + done_words : nullptr;
  s.state = c.take<uint32_t>((size_t)g.heads * topk::ST_WORDS);
  s.q2k_idx = c.take<int>(hn * g.N);
  s.q2k_num = c.take<int>(hn);
  s.vt_cnt = c.take<int>((size_t)g.heads * g.vtpad);
  s.vorder = c.take<int>((size_t)g.heads * g.vtpad);
  const size_t slots = (size_t)g.heads * g.P * g.tchunks;
  s.part_o = c.take<float>(slots * kRows * kDim);
  s.part_lse = c.take<float>(slots * kRows);
  if (bytes) *bytes = c.used;
  return s;
}

// The backward's scratch, in carve order.
struct BwdScratch {
  uint32_t* rowmask;  // [N][4]
  float* rcnt;        // [N] 1 / the block's tokens
  prep::SplitOut sq, sk, sv;  // as bf16 components and row lifts (prep.cuh, coarse_tc.cuh)
  float* cpart;               // the coarse kernel's split partials and their row statistics (coarse_tc.cuh)
  float* cpstat;
  prep::SplitOut so;  // the coarse output's gradient dco, as bf16 components and row lifts (glue.cu bwd_rows)
  float* lse2c;       // [heads][npad] the coarse rows' log2-sum-exp2 (the recompute's)
  float* dc;          // [heads][npad] Dc = rowsum(dco * Oc)
  float* oc;          // [heads][N][128]
  float* dqc;         // [heads][N][128] the coarse shares of dq, dk, dv per token: one array of three, zeroed by
  float* dkc;         // bwd_rows, which the coarse backward's splits add to
  float* dvc;
  float* delta;       // [heads][S]
  float* lse2;        // [heads][S]
  float* dqaccum;     // [heads][S][128]
  int* k2q_idx;       // [heads * N][N]
  int* k2q_num;       // [heads * N]
  uint32_t* khist;    // [heads][N + 1] (zeroed by bwd_prep)
  int* korder;        // [heads][N]
  size_t zero_words;
};

BwdScratch carve_bwd(const Geo& g, void* base, size_t* bytes) {
  Carver c{static_cast<uint8_t*>(base)};
  BwdScratch s;
  const size_t hn = (size_t)g.heads * g.N;
  s.rowmask = c.take<uint32_t>((size_t)g.N * 4);
  s.rcnt = c.take<float>((size_t)g.N);
  carve_split(c, g, s.sq, s.sk, s.sv);
  s.cpart = c.take<float>(coarse_tc::part_floats(g.N, g.heads));
  s.cpstat = c.take<float>(coarse_tc::pstat_floats(g.N, g.heads));
  carve_one_split(c, g, s.so);
  s.lse2c = c.take<float>((size_t)g.heads * coarse_npad(g));
  s.dc = c.take<float>((size_t)g.heads * coarse_npad(g));
  s.oc = c.take<float>(hn * kDim);
  s.dqc = c.take<float>(3 * hn * kDim);
  s.dkc = s.dqc + hn * kDim;
  s.dvc = s.dkc + hn * kDim;
  s.delta = c.take<float>((size_t)g.heads * g.S);
  s.lse2 = c.take<float>((size_t)g.heads * g.S);
  s.dqaccum = c.take<float>((size_t)g.heads * g.S * kDim);
  s.k2q_idx = c.take<int>(hn * g.N);
  s.k2q_num = c.take<int>(hn);
  s.zero_words = (size_t)g.heads * (g.N + 1);
  s.khist = c.take<uint32_t>(s.zero_words);
  s.korder = c.take<int>(hn);
  if (bytes) *bytes = c.used;
  return s;
}

// Every kernel is launched with programmatic stream serialization: it may be scheduled while the previous kernel
// drains, and its first statement (griddepcontrol.wait) waits for that kernel's completion and memory. Since every
// kernel waits before anything else, early exits included, completion stays ordered along the stream. `cluster`: the
// fine kernels' cluster of one CTA, which cluster launch control needs.
template <typename... KArgs, typename... Args>
void launch(const char* what, void (*kernel)(KArgs...), dim3 grid, dim3 block, size_t smem, cudaStream_t stream,
            bool cluster, Args... args) {
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = grid;
  cfg.blockDim = block;
  cfg.dynamicSmemBytes = smem;
  cfg.stream = stream;
  cudaLaunchAttribute attr[2];
  attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr[0].val.programmaticStreamSerializationAllowed = 1;
  attr[1].id = cudaLaunchAttributeClusterDimension;
  attr[1].val.clusterDim.x = 1;
  attr[1].val.clusterDim.y = 1;
  attr[1].val.clusterDim.z = 1;
  cfg.attrs = attr;
  cfg.numAttrs = cluster ? 2 : 1;
  check(cudaLaunchKernelEx(&cfg, kernel, args...), what);
}

unsigned blocks_for(long long threads, int per_block) {
  return (unsigned)std::max<long long>(1, std::min<long long>((threads + per_block - 1) / per_block, 1 << 20));
}

// The coarse branch on tcgen05 with three bf16 components per operand (coarse_tc.cuh): Oc and the rows' lse2 (the
// backward's recompute uses them; the forward writes lse2 unused). A cluster of `ksplit` CTAs per 128-row tile.
void coarse_tc_attention(bool bwd, const Geo& g, const prep::SplitOut& q, const prep::SplitOut& k,
                         const prep::SplitOut& v, float* part, float* pstat, float* oc, float* lse2, float scale_log2,
                         cudaStream_t stream) {
  void (*kernel)(coarse_tc::FwdArgs) = bwd ? coarse_tc::bwd_coarse_tc_kernel : coarse_tc::fwd_coarse_tc_kernel;
  const char* what = bwd ? "bwd_coarse_tc_kernel" : "fwd_coarse_tc_kernel";
  check(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)coarse_tc::kFwdSmemBytes),
        what);
  // How many clusters of each size fit at once: a property of the device and the kernel, measured once per device.
  static int clusters[16][coarse_tc::kMaxSplit + 1] = {};
  int dev = 0;
  check(cudaGetDevice(&dev), "cudaGetDevice");
  int(&fit)[coarse_tc::kMaxSplit + 1] = clusters[dev & 15];
  if (fit[1] == 0) {
    for (int ks = 1; ks <= coarse_tc::kMaxSplit; ++ks) {
      cudaLaunchConfig_t cfg = {};
      cfg.gridDim = dim3((unsigned)ks, 1);
      cfg.blockDim = dim3(coarse_tc::kThreads);
      cfg.dynamicSmemBytes = coarse_tc::kFwdSmemBytes;
      cudaLaunchAttribute attr[1];
      attr[0].id = cudaLaunchAttributeClusterDimension;
      attr[0].val.clusterDim.x = (unsigned)ks;
      attr[0].val.clusterDim.y = 1;
      attr[0].val.clusterDim.z = 1;
      cfg.attrs = attr;
      cfg.numAttrs = 1;
      int n = 0;
      check(cudaOccupancyMaxActiveClusters(&n, (const void*)kernel, &cfg), "cudaOccupancyMaxActiveClusters");
      fit[ks] = n > 0 ? n : 1;
    }
  }
  coarse_tc::FwdArgs a;
  a.qp = q.planes;
  a.qe = q.e;
  a.kp = k.planes;
  a.ke = k.e;
  a.vp = v.planes;
  a.ve = v.e;
  a.rpad = q.npad;
  a.oc = oc;
  a.lse2 = lse2;
  a.part = part;
  a.pstat = pstat;
  a.N = g.N;
  a.npad = (int)coarse_npad(g);
  const int tiles = (g.N + coarse_tc::kTile - 1) / coarse_tc::kTile;
  a.tiles = tiles;
  a.ksplit = coarse_tc::fwd_split(g.N, g.heads, fit);
  a.scale_log2 = scale_log2;
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3((unsigned)(tiles * a.ksplit), (unsigned)g.heads);
  cfg.blockDim = dim3(coarse_tc::kThreads);
  cfg.dynamicSmemBytes = coarse_tc::kFwdSmemBytes;
  cfg.stream = stream;
  cudaLaunchAttribute attr[2];
  attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr[0].val.programmaticStreamSerializationAllowed = 1;
  attr[1].id = cudaLaunchAttributeClusterDimension;
  attr[1].val.clusterDim.x = (unsigned)a.ksplit;
  attr[1].val.clusterDim.y = 1;
  attr[1].val.clusterDim.z = 1;
  cfg.attrs = attr;
  cfg.numAttrs = 2;
  check(cudaLaunchKernelEx(&cfg, kernel, a), what);
}

// The coarse backward as one launch of three roles (coarse_tc.cuh): dqc, dkc, dvc from the split operands, dco's
// planes, Dc and the recompute's lse2c. `ksplit` CTAs per tile and role, which add their shares in L2.
void coarse_gradient_tc(const Geo& g, const BwdScratch& s, float scale_log2, float scale, cudaStream_t stream) {
  void (*kernel)(coarse_tc::GradArgs) = coarse_tc::bwd_coarse_grad_tc_kernel;
  check(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)coarse_tc::kGradSmemBytes),
        "bwd_coarse_grad_tc_kernel");
  static int resident[16] = {};  // co-resident CTAs, per device
  int dev = 0;
  check(cudaGetDevice(&dev), "cudaGetDevice");
  int& fit = resident[dev & 15];
  if (fit == 0) {
    int per_sm = 0, sms = 0;
    check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, kernel, coarse_tc::kThreads,
                                                        coarse_tc::kGradSmemBytes),
          "cudaOccupancyMaxActiveBlocksPerMultiprocessor");
    check(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev), "cudaDeviceGetAttribute");
    fit = per_sm * sms > 0 ? per_sm * sms : 1;
  }
  coarse_tc::GradArgs a;
  a.qp = s.sq.planes;
  a.qe = s.sq.e;
  a.kp = s.sk.planes;
  a.ke = s.sk.e;
  a.vp = s.sv.planes;
  a.ve = s.sv.e;
  a.op = s.so.planes;
  a.oe = s.so.e;
  a.lse2 = s.lse2c;
  a.dc = s.dc;
  a.rcnt = s.rcnt;
  a.dqc = s.dqc;
  a.dkc = s.dkc;
  a.dvc = s.dvc;
  a.N = g.N;
  a.npad = (int)coarse_npad(g);
  a.rpad = s.sq.npad;
  const int tiles = (g.N + coarse_tc::kTile - 1) / coarse_tc::kTile;
  a.tiles = tiles;
  a.ksplit = coarse_tc::grad_split(g.N, 3 * g.heads, fit);  // three roles per tile and head
  a.scale_log2 = scale_log2;
  a.scale = scale;
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3((unsigned)(3 * tiles * a.ksplit), (unsigned)g.heads);
  cfg.blockDim = dim3(coarse_tc::kThreads);
  cfg.dynamicSmemBytes = coarse_tc::kGradSmemBytes;
  cfg.stream = stream;
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr[0].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = attr;
  cfg.numAttrs = 1;
  check(cudaLaunchKernelEx(&cfg, kernel, a), "bwd_coarse_grad_tc_kernel");
}

}  // namespace

Geo make_geo(int heads, int text, int t, int h, int w, int topk) {
  Geo g;
  g.heads = heads;
  g.text = text;
  g.P = (text + kRows - 1) / kRows;
  g.t = t;
  g.h = h;
  g.w = w;
  g.nt = (t + 3) / 4;
  g.nh = (h + 3) / 4;
  g.nw = (w + 7) / 8;
  g.Nv = g.nt * g.nh * g.nw;
  g.N = g.P + g.Nv;
  g.S = (long long)g.N * kRows;
  g.topk = topk;
  const long long nn = (long long)g.Nv * g.Nv;
  g.K = std::min<long long>((long long)std::max(topk, 0) * g.Nv, nn);
  g.rtiles = (int)(((long long)g.Nv * kWindows + kRows - 1) / kRows);
  // Text chunks of about a video tile's count (P text blocks, topk selected tiles and the diagonal), at least 32.
  const int target = std::max(32, g.P + (int)std::min<long long>(std::max(topk, 0), g.Nv) + 1);
  g.tchunks = (g.N + target - 1) / target;
  g.tlen = (g.N + g.tchunks - 1) / g.tchunks;
  g.tchunks = (g.N + g.tlen - 1) / g.tlen;  // no empty chunk
  g.vt = g.Nv + g.P * g.tchunks;
  g.vtpad = g.vt + (g.vt & 1);
  return g;
}

size_t forward_scratch_bytes(const Geo& g) {
  size_t bytes = 0;
  carve_fwd(g, nullptr, &bytes);
  return bytes;
}

size_t backward_scratch_bytes(const Geo& g) {
  size_t bytes = 0;
  carve_bwd(g, nullptr, &bytes);
  return bytes;
}

void forward(const Geo& g, const void* q, const void* k, const void* v, const void* gate, void* out, float* selection,
             float* lse, void* scratch, cudaStream_t stream) {
  const FwdScratch s = carve_fwd(g, scratch, nullptr);
  const float scale = 1.f / std::sqrt((float)kDim);
  const float scale_log2 = (float)(1.4426950408889634 / std::sqrt((double)kDim));
  const auto* q16 = static_cast<const __nv_bfloat16*>(q);
  const auto* k16 = static_cast<const __nv_bfloat16*>(k);
  const auto* v16 = static_cast<const __nv_bfloat16*>(v);

  // Means, the masks, and the histograms zeroed.
  {
    prep::FwdArgs a;
    a.g = g;
    a.q = q16;
    a.k = k16;
    a.v = v16;
    a.sq = s.sq;
    a.sk = s.sk;
    a.sv = s.sv;
    a.qw = s.qw;
    a.kw = s.kw;
    a.qnorm = s.qnorm;
    a.knorm = s.knorm;
    a.rowmask = s.rowmask;
    a.winmask = s.winmask;
    a.zero = s.zeroed;
    a.zero_words = (long long)s.zero_words;
    launch("fwd_prep_kernel", prep::fwd_prep_kernel, dim3(g.N, g.heads), prep::kThreads, 0, stream, false, a);
  }

  // The coarse branch.
  coarse_tc_attention(false, g, s.sq, s.sk, s.sv, s.cpart, s.cpstat, s.oc, s.lse2c, scale_log2, stream);

  // The router: one sweep, then the two passes for the query tiles it flagged (none, unless a bound was loose).
  {
    check(cudaFuncSetAttribute(router::router_kernel<0>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                               (int)router::kSmemBytes), "router smem");
    check(cudaFuncSetAttribute(router::router_kernel<1>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                               (int)router::kSmemBytes), "router smem");
    check(cudaFuncSetAttribute(router::router_kernel<2>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                               (int)router::kSmemBytes), "router smem");
    const long long windows = (long long)g.Nv * kWindows;
    const CUtensorMap tq = rows_map_f16(s.qw, g.heads, windows);
    const CUtensorMap tk = rows_map_f16(s.kw, g.heads, windows);
    router::Args a;
    a.g = g;
    a.winmask = s.winmask;
    a.qnorm = s.qnorm;
    a.knorm = s.knorm;
    a.u = s.u;
    a.upitch = upitch(g);
    a.flags = s.flags;
    a.rstat = s.rstat;
    a.scores = s.scores;
    a.lpart = s.lpart;
    a.ksplit = router::ksplit_for(g.rtiles, g.heads, sms());
    a.scale_log2 = scale_log2;
    router::Args a1 = a;
    a1.ksplit = 1;  // a two-pass CTA owns the whole key range, so it needs no merge and writes its own score
    const dim3 grid(g.rtiles * a.ksplit, g.heads), grid1(g.rtiles, g.heads), block(router::kWarps * 32);
    const dim3 grid2(g.rtiles, g.heads, router::kPoolSlices);
    launch("router_kernel<2>", router::router_kernel<2>, grid, block, router::kSmemBytes, stream, false, tq, tk, a);
    launch("router_pool_kernel", router::router_pool_kernel, grid2, dim3(router::kPoolThreads), 0, stream, false, a);
    launch("router_kernel<0>", router::router_kernel<0>, grid1, block, router::kSmemBytes, stream, false, tq, tk, a1);
    launch("router_kernel<1>", router::router_kernel<1>, grid1, block, router::kSmemBytes, stream, false, tq, tk, a1);
  }

  // The top-K and the selection.
  {
    const long long per_head = (long long)g.Nv * g.Nv;
    const dim3 hgrid(kChunks, g.heads);
    launch("router_hist_kernel<0>", topk::router_hist_kernel<0>, hgrid, topk::kHistThreads, 0, stream, false, s.scores,
           s.hist, s.state, s.done, per_head, g.K);
    launch("router_hist_kernel<1>", topk::router_hist_kernel<1>, hgrid, topk::kHistThreads, 0, stream, false, s.scores,
           s.hist, s.state, s.done, per_head, g.K);
    launch("router_hist_kernel<2>", topk::router_hist_kernel<2>, hgrid, topk::kHistThreads, 0, stream, false, s.scores,
           s.hist, s.state, s.done, per_head, g.K);
    launch("router_tie_pin_kernel", topk::router_tie_pin_kernel, hgrid, topk::kHistThreads, 0, stream, false, s.scores,
           s.state, s.tiecnt, g.Nv);
    launch("router_select_kernel", topk::router_select_kernel, hgrid, 256, 0, stream, false, s.scores, selection,
           s.state, g.Nv);
  }

  // The q2k lists, the tiles' counts and their order.
  {
    const long long warps = (long long)g.heads * (g.N + g.P * g.tchunks + 1);
    launch("fwd_lists_kernel", topk::fwd_lists_kernel, blocks_for(warps * 32, 256), 256, 0, stream, false, g, selection,
           s.q2k_idx, s.q2k_num, s.vt_cnt, s.ohist);
    launch("fwd_order_scan_kernel", topk::fwd_order_scan_kernel, g.heads, 1024, 0, stream, false, s.ohist, g.N);
    launch("fwd_order_scatter_kernel", topk::fwd_order_scatter_kernel, blocks_for((long long)g.heads * g.vtpad, 256),
           256, 0, stream, false, s.vt_cnt, s.ohist, s.vorder, g.heads, g.vtpad, g.N);
  }

  // The fine branch, merged with the coarse one.
  {
    check(cudaFuncSetAttribute(fwd::fwd_fine_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)fwd::kSmemBytes),
          "fwd_fine smem");
    const CUtensorMap tq = head_token_map(q, g.heads, g.S);
    const CUtensorMap tk = rows_map(k, g.heads, g.S);
    const CUtensorMap tv = rows_map(v, g.heads, g.S);
    const CUtensorMap to = head_token_map(out, g.heads, g.S);
    fwd::Args a;
    a.heads = g.heads;
    a.num_blocks = g.N;
    a.P = g.P;
    a.Nv = g.Nv;
    a.tchunks = g.tchunks;
    a.tlen = g.tlen;
    a.vtpad = g.vtpad;
    a.seqlen = g.S;
    a.scale_log2 = scale_log2;
    a.q2k_idx = s.q2k_idx;
    a.q2k_num = s.q2k_num;
    a.vorder = s.vorder;
    a.part_o = s.part_o;
    a.part_lse = s.part_lse;
    a.rowmask = s.rowmask;
    a.gate = static_cast<const __nv_bfloat16*>(gate);
    a.coarse = s.oc;
    a.lse = lse;
    launch("fwd_fine_kernel", fwd::fwd_fine_kernel, dim3((unsigned)(g.heads * (g.vtpad / 2))), dim3(fwd::N_WARPS * 32),
           fwd::kSmemBytes, stream, true, tq, tk, tv, to, a);
    if (g.P > 0) {
      launch("fwd_merge_kernel", fwd::fwd_merge_kernel, blocks_for((long long)g.heads * g.P * kRows * 32, 256), 256, 0,
             stream, false, a, static_cast<__nv_bfloat16*>(out));
    }
  }
}

void backward(const Geo& g, const void* q, const void* k, const void* v, const void* gate, const void* out,
              const float* lse, const float* selection, const void* dout, void* dq, void* dk, void* dv, void* dgate,
              void* scratch, cudaStream_t stream) {
  const BwdScratch s = carve_bwd(g, scratch, nullptr);
  const float scale = 1.f / std::sqrt((float)kDim);
  const float scale_log2 = (float)(1.4426950408889634 / std::sqrt((double)kDim));

  {
    prep::BwdArgs a;
    a.g = g;
    a.q = static_cast<const __nv_bfloat16*>(q);
    a.k = static_cast<const __nv_bfloat16*>(k);
    a.v = static_cast<const __nv_bfloat16*>(v);
    a.sq = s.sq;
    a.sk = s.sk;
    a.sv = s.sv;
    a.rowmask = s.rowmask;
    a.rcnt = s.rcnt;
    a.zero = s.khist;
    a.zero_words = (long long)s.zero_words;
    launch("bwd_prep_kernel", prep::bwd_prep_kernel, dim3(g.N, g.heads), prep::kThreads, 0, stream, false, a);
  }

  // The coarse branch again: Oc and the rows' lse2c (the same kernel body as the forward's, so the same Oc).
  coarse_tc_attention(true, g, s.sq, s.sk, s.sv, s.cpart, s.cpstat, s.oc, s.lse2c, scale_log2, stream);

  // Rows: D, lse2, dgate, the coarse output's gradient, dqaccum zeroed. It must follow the coarse recompute (D needs
  // Oc) and precede the coarse backward (which needs dco), so dco is formed where gate and dout are already read.
  {
    launch("bwd_rows_kernel", glue::bwd_rows_kernel, dim3(g.N, g.heads), glue::kRowsThreads, 0, stream, false, g,
           static_cast<const __nv_bfloat16*>(out), static_cast<const __nv_bfloat16*>(dout),
           static_cast<const __nv_bfloat16*>(gate), lse, s.oc, s.rowmask, s.delta, s.lse2,
           static_cast<__nv_bfloat16*>(dgate), s.so.planes, s.so.e, s.so.npad, s.dc, s.dqaccum, s.dqc);
  }

  // The coarse backward, each row divided by its block's token count: one launch.
  coarse_gradient_tc(g, s, scale_log2, scale, stream);

  // The k2q lists from the selection, and their order.
  {
    const long long items = (long long)g.heads * g.N;
    launch("bwd_lists_kernel", topk::bwd_lists_kernel, blocks_for(items * 32, 256), 256, 0, stream, false, g, selection,
           s.k2q_idx, s.k2q_num);
    launch("bwd_order_hist_kernel", topk::bwd_order_hist_kernel, blocks_for(items, 256), 256, 0, stream, false,
           s.k2q_num, s.khist, g.heads, g.N, g.N);
    launch("bwd_order_scan_kernel", topk::bwd_order_scan_kernel, g.heads, 1024, 0, stream, false, s.khist, g.N);
    launch("bwd_order_scatter_kernel", topk::bwd_order_scatter_kernel, blocks_for(items, 256), 256, 0, stream, false,
           s.k2q_num, s.khist, s.korder, g.heads, g.N, g.N);
  }

  // The fine backward, then dq.
  {
    check(cudaFuncSetAttribute(bwd::bwd_fine_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, bwd::SMEM_TOTAL),
          "bwd_fine smem");
    const CUtensorMap tq = subtile_map(q, g.heads, g.S);
    const CUtensorMap tk = subtile_map(k, g.heads, g.S);
    const CUtensorMap tv = subtile_map(v, g.heads, g.S);
    const CUtensorMap tdo = subtile_map(dout, g.heads, g.S);
    const CUtensorMap tdk = subtile_map(dk, g.heads, g.S);
    const CUtensorMap tdv = subtile_map(dv, g.heads, g.S);
    bwd::Args a;
    a.heads = g.heads;
    a.num_blocks = g.N;
    a.P = g.P;
    a.seqlen = g.S;
    a.scale_log2 = scale_log2;
    a.sm_scale = scale;
    a.dqaccum = s.dqaccum;
    a.lse2 = s.lse2;
    a.delta = s.delta;
    a.k2q_idx = s.k2q_idx;
    a.k2q_num = s.k2q_num;
    a.korder = s.korder;
    a.rowmask = s.rowmask;
    a.dk_coarse = s.dkc;
    a.dv_coarse = s.dvc;
    launch("bwd_fine_kernel", bwd::bwd_fine_kernel, dim3((unsigned)(g.heads * g.N)), dim3(bwd::N_WARPS * 32),
           (size_t)bwd::SMEM_TOTAL, stream, true, tq, tk, tv, tdo, tdk, tdv, a);
    launch("bwd_post_kernel", glue::bwd_post_kernel, dim3(g.N * (glue::kBlock / glue::kPostRows), g.heads),
           glue::kPostThreads, 0, stream, false,
           s.dqaccum, s.dqc, s.rowmask, static_cast<__nv_bfloat16*>(dq), g.N, g.S, scale);
  }
}

}  // namespace vsa2t
