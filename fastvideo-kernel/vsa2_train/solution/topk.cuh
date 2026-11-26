// topk.cuh -- the router's top-K (eq. 8-9) and the block lists the fine branch runs on.
//
// Top-K: per head, the K = min(topk * Nv, Nv^2) largest of the Nv^2 tile-pair scores, plus every diagonal pair. The
// scores are non-negative floats, so their bit patterns sort as integers: a radix select over 11 + 11 + 10 bits finds
// the K-th largest score T, and the top-K then takes the tied pairs at T in ascending row-major pair index, so the
// selection is a function of the scores alone. The count it takes is `ties`, the non-diagonal pairs at T that the
// budget still wants; the kept count is in [K, K + Nv] and not [K, K + ties], because a diagonal exactly at T is kept
// by the diagonal rule and consumes budget without consuming the cut. Kernels: hist (a level's histogram,
// CTA-local then global atomics; the head's last CTA then picks the level's bin), tie_pin (the cut between the tied
// pairs the top-K takes and those it leaves), select (writes `selection`).
//
// Lists: the forward runs each query block over its kept key blocks (q2k: a text block keeps all N; a video tile the P
// text blocks, then its selected tiles in ascending order); the backward runs each key block over the query blocks
// that kept it (k2q: a text block is kept by all N; a video tile by the P text blocks, then the tiles that selected
// it, ascending). The work order sorts items by count, heaviest first (a counting sort: histogram, scan, scatter), so
// that the long items start first and the forward pairs items of similar counts.
#pragma once
#include <cuda_runtime.h>

#include "common.cuh"

namespace vsa2t {
namespace topk {

constexpr int kBins = 2048;
constexpr int kHistThreads = 512;
// State per head, u32[ST_WORDS]: the bits found so far, the count still to take at the current prefix, the mode, the
// number of ties at T to take, the tie pin's handshake, whether the winning bin holds more than one score (a launch
// trigger only: a bin's count includes the tied diagonals), and the exclusive row-major cut of the tied pairs the
// top-K takes (a tied pair is kept iff its index is below it).
enum { ST_PREFIX = 0, ST_KREM = 1, ST_MODE = 2, ST_TIES = 3, ST_TIEDONE = 4, ST_NEEDTIE = 5, ST_TIECUT = 6,
       ST_WORDS = 7 };
enum { MODE_SELECT = 0, MODE_DIAGONAL = 1, MODE_ALL = 2 };

template <int LEVEL>
__device__ __forceinline__ bool level_bin(uint32_t u, uint32_t prefix, int& bin) {
  if (LEVEL == 0) {
    bin = (int)(u >> 21);
    return true;
  } else if (LEVEL == 1) {
    bin = (int)((u >> 10) & 2047u);
    return (u >> 21) == prefix;
  } else {
    bin = (int)(u & 1023u);
    return (u >> 10) == prefix;
  }
}

// The whole CTA (kHistThreads threads): head hd's bin of this level holding the K-th largest score among those that
// match the prefix, and the state. Thread t owns bins [nbins - (t + 1) per, nbins - t per) (thread 0 the largest
// scores); the histogram is read through L2 (.cg): the other CTAs' atomics made it.
template <int LEVEL>
__device__ __forceinline__ void scan_level(const uint32_t* __restrict__ hist, uint32_t* state, long long K, long long all,
                                           int hd) {
  __shared__ uint32_t wsum[kHistThreads / 32];
  __shared__ int owner_warp;
  uint32_t* st = state + hd * ST_WORDS;
  const int tid = threadIdx.x, lane = tid & 31, wid = tid >> 5;
  if (LEVEL == 0) {
    if (K <= 0 || K >= all) {
      if (tid == 0) {
        st[ST_MODE] = K <= 0 ? MODE_DIAGONAL : MODE_ALL;
        st[ST_PREFIX] = 0;
        st[ST_KREM] = 0;
        st[ST_TIES] = 0;
        st[ST_NEEDTIE] = 0;
        st[ST_TIECUT] = 0;
        st[ST_TIEDONE] = 0;
      }
      return;
    }
  } else if (st[ST_MODE] != MODE_SELECT) {
    return;
  }
  const uint32_t krem = LEVEL == 0 ? (uint32_t)K : st[ST_KREM];
  const uint32_t prefix = LEVEL == 0 ? 0u : st[ST_PREFIX];
  constexpr int nbins = LEVEL == 2 ? 1024 : 2048, per = nbins / kHistThreads, bits = LEVEL == 2 ? 10 : 11;
  const uint32_t* h = hist + ((long long)hd * 3 + LEVEL) * kBins;
  const int top = nbins - tid * per;
  uint32_t mine[per], sum = 0;
#pragma unroll
  for (int b = 0; b < per; ++b) {
    mine[b] = __ldcg(h + top - 1 - b);  // mine[0] is the thread's largest bin
    sum += mine[b];
  }
  // Inclusive scan over the threads, in thread order (descending bins).
  uint32_t incl = sum;
#pragma unroll
  for (int o = 1; o < 32; o <<= 1) {
    const uint32_t y = __shfl_up_sync(0xffffffffu, incl, o);
    if (lane >= o) incl += y;
  }
  if (lane == 31) wsum[wid] = incl;
  if (tid == 0) owner_warp = kHistThreads / 32;
  __syncthreads();
  uint32_t before = 0;
  for (int w = 0; w < wid; ++w) before += wsum[w];
  incl += before;
  const unsigned hit = __ballot_sync(0xffffffffu, incl >= krem);
  if (lane == 0 && hit) atomicMin(&owner_warp, wid);
  __syncthreads();
  // The first thread whose range reaches the K-th score (there is one: krem <= the total).
  if (wid == owner_warp && lane == __ffs(hit) - 1) {
    uint32_t above = incl - sum;
    int bin = top - per;
    uint32_t bincount = 0;  // the winning bin's scores, diagonals included
#pragma unroll
    for (int b = 0; b < per; ++b) {
      if (above + mine[b] >= krem) {
        bin = top - 1 - b;
        bincount = mine[b];
        break;
      }
      above += mine[b];
    }
    st[ST_PREFIX] = (prefix << bits) | (uint32_t)bin;
    st[ST_KREM] = krem - above;
    if (LEVEL == 0) {
      st[ST_MODE] = MODE_SELECT;
      st[ST_TIEDONE] = 0;
      // No cut until the tie pin writes one: a bin that holds a single score at T leaves nothing to choose, and its
      // tied pair is kept by the pair's own test (the tie pin is not even launched for it).
      st[ST_TIECUT] = 0xffffffffu;
    }
    if (LEVEL == 2) {
      st[ST_TIES] = krem - above;  // T's bin the top-K still wants: diagonals at T are counted here too
      // More than one score in T's bin: some of them may tie at T, so the top-K's choice among them needs the cut.
      st[ST_NEEDTIE] = bincount >= 2u ? 1u : 0u;
    }
  }
}

// hist [heads][3][kBins] and done [heads][3] (zeroed before the call); grid (chunks, heads). The last CTA of a head to
// add its histogram scans it (scan_level), which the next level's kernel and the select read.
template <int LEVEL>
__global__ void __launch_bounds__(kHistThreads) router_hist_kernel(const float* __restrict__ scores, uint32_t* hist,
                                                                   uint32_t* state, uint32_t* done, long long per_head,
                                                                   long long K) {
  // Programmatic dependent launch: nothing of the previous kernel is read before it completes (vsa2t.cu).
  griddepcontrol_wait();
  __shared__ uint32_t sh[kBins];
  const int hd = blockIdx.y;
  uint32_t prefix = 0;
  if (LEVEL > 0) {
    // Every CTA of the head returns here together (the mode is the head's), so no last CTA is awaited.
    if (state[hd * ST_WORDS + ST_MODE] != MODE_SELECT) return;
    prefix = state[hd * ST_WORDS + ST_PREFIX];
  }
  for (int i = threadIdx.x; i < kBins; i += blockDim.x) sh[i] = 0;
  __syncthreads();
  const float* s = scores + hd * per_head;
  const long long chunk = (per_head + gridDim.x - 1) / gridDim.x;
  const long long lo = blockIdx.x * chunk, hi = min(per_head, lo + chunk);
  for (long long i = lo + threadIdx.x; i < hi; i += blockDim.x) {
    int bin;
    if (level_bin<LEVEL>(__float_as_uint(s[i]), prefix, bin)) atomicAdd(&sh[bin], 1u);
  }
  __syncthreads();
  uint32_t* dst = hist + ((long long)hd * 3 + LEVEL) * kBins;
  for (int i = threadIdx.x; i < kBins; i += blockDim.x)
    if (sh[i]) atomicAdd(&dst[i], sh[i]);
  // The last CTA of the head: every other CTA's atomics are visible once its own count has landed after them.
  __shared__ bool last;
  __threadfence();
  __syncthreads();
  if (threadIdx.x == 0) last = atomicAdd(&done[hd * 3 + LEVEL], 1u) == gridDim.x - 1;
  __syncthreads();
  if (last) {
    __threadfence();
    scan_level<LEVEL>(hist, state, K, per_head, hd);
  }
}

// The tie budget's cut, so that the select can stay a pure function of the scores: among the non-diagonal pairs whose
// score is bit-equal to the threshold T, the top-K takes the `ties` of them with the lowest row-major index p = i*nv + j,
// and this kernel finds the exclusive cut (`ST_TIECUT`: a tied pair is kept iff p < tiecut). A pair's rank among the
// ties is not local to it, hence a second pass over the scores; it returns at once unless the winning bin held a tie,
// which on typical inputs it does not. grid (chunks, heads), the top-K's own chunking; tiecnt [heads][chunks] is its
// scratch.
__global__ void __launch_bounds__(1024) router_tie_pin_kernel(const float* __restrict__ scores, uint32_t* state,
                                                             uint32_t* tiecnt, int nv) {
  // Programmatic dependent launch: nothing of the previous kernel is read before it completes (vsa2t.cu).
  griddepcontrol_wait();
  __shared__ uint32_t wcnt[32];
  __shared__ uint32_t tilesum, seen, found, rank;
  __shared__ int at;
  __shared__ bool last;
  const int hd = blockIdx.y;
  uint32_t* st = state + hd * ST_WORDS;
  const uint32_t ties = st[ST_TIES];
  const uint32_t mode = st[ST_MODE];
  // The tie path is the select mode's alone: every other mode keeps all of its pairs or just the diagonal, and the
  // state's cut says so (scan_level). A winning bin that holds a single score at T leaves nothing to choose either.
  if (mode != MODE_SELECT || st[ST_NEEDTIE] == 0u) return;
  if (ties == 0u) {
    // Unreachable in the select mode (the bin that reaches krem always leaves at least one score to take): kept so that
    // a broken invariant cannot turn into a selection that keeps every tied pair.
    if (blockIdx.x == 0 && threadIdx.x == 0) st[ST_TIECUT] = 0u;
    return;
  }
  const uint32_t thr = st[ST_PREFIX];
  const long long per_head = (long long)nv * nv;
  const long long chunk = (per_head + gridDim.x - 1) / gridDim.x;
  const long long lo = blockIdx.x * chunk, hi = min(per_head, lo + chunk);
  const float* s = scores + (long long)hd * per_head;
  const int tid = threadIdx.x, lane = tid & 31, wid = tid >> 5, nw = blockDim.x >> 5;
  const unsigned below_me = (1u << lane) - 1u;

  // Phase A (every CTA): this chunk's non-diagonal ties at T. The chunk is rows i0 .. i1-1 of the pair grid, its first
  // and last rows cut by the chunk's ends; the divisions are the CTA's, not the element's (a 64-bit divide is a call,
  // and a call spills the counters the loop carries).
  const long long i0 = lo / nv, i1 = (hi + nv - 1) / nv;
  uint32_t count = 0;
  for (long long i = i0; i < i1; ++i) {
    const long long b = i * nv, jlo = max(lo, b) - b, jhi = min(hi, b + nv) - b;
    for (long long j = jlo + tid; j < jhi; j += blockDim.x) {
      const long long p = b + j;
      if (i != j && __float_as_uint(s[p]) == thr) ++count;
    }
  }
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) {
    count += __shfl_down_sync(0xffffffffu, count, o);
  }
  if (lane == 0) wcnt[wid] = count;
  __syncthreads();
  if (tid == 0) {
    uint32_t tot = 0;
    for (int w = 0; w < nw; ++w) tot += wcnt[w];
    tiecnt[(long long)hd * gridDim.x + blockIdx.x] = tot;
  }
  // The last CTA of the head: every other CTA's count is visible once its own has landed after them.
  __threadfence();
  __syncthreads();
  if (tid == 0) last = atomicAdd(&st[ST_TIEDONE], 1u) == gridDim.x - 1;
  __syncthreads();
  if (!last) return;
  __threadfence();

  // Phase B (the head's last CTA): the chunk holding the `ties`-th non-diagonal tie, and the rank wanted within it.
  if (tid == 0) {
    uint32_t before = 0;
    int a = (int)gridDim.x;
    uint32_t r = 0;
    for (int c = 0; c < (int)gridDim.x; ++c) {
      const uint32_t n = __ldcg(tiecnt + (long long)hd * gridDim.x + c);
      if (before + n >= ties) {
        a = c;
        r = ties - before;
        break;
      }
      before += n;
    }
    at = a;
    rank = r;
    seen = 0;
    found = 0;
  }
  __syncthreads();
  if (at >= (int)gridDim.x) {
    // Fewer non-diagonal ties than the budget (tied diagonals take the rest of it): keep every one of them.
    if (tid == 0) st[ST_TIECUT] = 0xffffffffu;
    return;
  }
  // The crossing chunk in tiles of blockDim.x, the ties numbered as the select numbers them: the hits of the tiles
  // before this one, then those of the warps before this lane's, then those of its lanes below it. One rank matches one
  // pair, so the cut does not depend on which thread reaches it first.
  const long long clo = (long long)at * chunk, chi = min(per_head, clo + chunk);
  for (long long base = clo; base < chi; base += blockDim.x) {
    const long long p = base + tid;
    bool hit = false;
    if (p < chi) {
      const long long i = p / nv;
      hit = i != p - i * nv && __float_as_uint(s[p]) == thr;
    }
    const unsigned m = __ballot_sync(0xffffffffu, hit);
    if (lane == 0) wcnt[wid] = __popc(m);
    __syncthreads();
    if (tid == 0) {
      uint32_t acc = seen;
      for (int w = 0; w < nw; ++w) {
        const uint32_t c = wcnt[w];
        wcnt[w] = acc;  // the hits before this warp, from the start of the chunk
        acc += c;
      }
      tilesum = acc;
    }
    __syncthreads();
    if (hit && wcnt[wid] + __popc(m & below_me) == rank - 1u) found = (uint32_t)p + 1u;
    __syncthreads();
    if (found) break;  // every thread reads the same `found`, so no ballot of a later tile is reached divergently
    if (tid == 0) seen = tilesum;
    __syncthreads();
  }
  if (tid == 0) st[ST_TIECUT] = found ? found : 0xffffffffu;
}

// selection[hd][i][j] = 1 for the top-K pairs and the diagonal, else 0; grid (chunks, heads).
__global__ void __launch_bounds__(256) router_select_kernel(const float* __restrict__ scores, float* __restrict__ selection,
                                                            uint32_t* state, int nv) {
  // Programmatic dependent launch: nothing of the previous kernel is read before it completes (vsa2t.cu).
  griddepcontrol_wait();
  const int hd = blockIdx.y;
  uint32_t* st = state + hd * ST_WORDS;
  const uint32_t mode = st[ST_MODE], thr = st[ST_PREFIX], tiecut = st[ST_TIECUT];
  const long long per_head = (long long)nv * nv;
  const long long chunk = (per_head + gridDim.x - 1) / gridDim.x;
  const long long lo = blockIdx.x * chunk, hi = min(per_head, lo + chunk);
  const float* s = scores + hd * per_head;
  float* out = selection + hd * per_head;
  for (long long p = lo + threadIdx.x; p < hi; p += blockDim.x) {
    const long long i = p / nv, j = p - i * nv;
    bool keep = i == j || mode == MODE_ALL;
    if (mode == MODE_SELECT) {
      const uint32_t u = __float_as_uint(s[p]);
      if (u > thr) keep = true;
      else if (u == thr && i != j && (uint32_t)p < tiecut) keep = true;
    }
    out[p] = keep ? 1.f : 0.f;
  }
}

// ---- Work order: items of one head sorted by count, heaviest first ----

// hist [heads][max_count + 1] of the counts (zeroed before), turned into each count's first position in place.
__device__ __forceinline__ void order_scan(uint32_t* hist, int max_count) {
  __shared__ uint32_t warp_sums[32];
  __shared__ uint32_t carry;
  uint32_t* h = hist + (long long)blockIdx.x * (max_count + 1);
  const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5, nw = blockDim.x >> 5;
  if (threadIdx.x == 0) carry = 0;
  __syncthreads();
  // Descending counts: position of count c = the items of every count above c.
  for (int base = max_count; base >= 0; base -= blockDim.x) {
    const int c = base - (int)threadIdx.x;
    const uint32_t x = c >= 0 ? h[c] : 0u;
    uint32_t incl = x;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
      const uint32_t y = __shfl_up_sync(0xffffffffu, incl, o);
      if (lane >= o) incl += y;
    }
    if (lane == 31) warp_sums[wid] = incl;
    __syncthreads();
    uint32_t before = carry;
    for (int w2 = 0; w2 < wid; ++w2) before += warp_sums[w2];
    if (c >= 0) h[c] = before + incl - x;
    __syncthreads();
    if (threadIdx.x == 0) {
      uint32_t tot = 0;
      for (int w2 = 0; w2 < nw; ++w2) tot += warp_sums[w2];
      carry += tot;
    }
    __syncthreads();
  }
}

// order[hd][pos] = item, for items 0..items-1 of each head with counts num[hd][item]; grid over heads * items.
__device__ __forceinline__ void order_scatter(const int* __restrict__ num, uint32_t* hist, int* order, int heads,
                                              int items, int max_count) {
  const long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= (long long)heads * items) return;
  const int hd = (int)(tid / items), it = (int)(tid - (long long)hd * items);
  const int c = num[tid];
  const uint32_t pos = atomicAdd(&hist[(long long)hd * (max_count + 1) + c], 1u);
  order[(long long)hd * items + pos] = it;
}

// ---- Forward: q2k lists ----

// One warp per (head, i), i < N + P * tchunks + 1:
//   i < N: query block i's list q2k_idx[hd][i] (a text block: all N; a video tile: the P text blocks, then its
//          selected tiles in ascending order) and its count q2k_num[hd][i]; a video tile is fine-forward tile i - P;
//   i = N + u, u < P * tchunks: text chunk u, tile Nv + u, of its chunk's length;
//   i = N + P * tchunks: the phantom tile when the tiles are odd in number (count 0).
// Every tile's count goes to vt_cnt[hd][tile] and the histogram.
__global__ void fwd_lists_kernel(Geo g, const float* __restrict__ selection, int* q2k_idx, int* q2k_num, int* vt_cnt,
                                 uint32_t* ohist) {
  // Programmatic dependent launch: nothing of the previous kernel is read before it completes (vsa2t.cu).
  griddepcontrol_wait();
  const long long wid = ((long long)blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  const int lane = threadIdx.x & 31;
  const int per_head = g.N + g.P * g.tchunks + 1;
  if (wid >= (long long)g.heads * per_head) return;
  const int hd = (int)(wid / per_head), i = (int)(wid - (long long)hd * per_head);
  int tile = -1, count = 0;
  if (i < g.N) {
    int* list = q2k_idx + ((long long)hd * g.N + i) * g.N;
    if (i < g.P) {
      for (int j = lane; j < g.N; j += 32) list[j] = j;
      count = g.N;
    } else {
      for (int j = lane; j < g.P; j += 32) list[j] = j;
      count = g.P;
      const float* row = selection + ((long long)hd * g.Nv + (i - g.P)) * g.Nv;
      for (int j0 = 0; j0 < g.Nv; j0 += 32) {
        const int j = j0 + lane;
        const bool f = j < g.Nv && row[j] > 0.5f;
        const unsigned m = __ballot_sync(0xffffffffu, f);
        if (f) list[count + __popc(m & ((1u << lane) - 1u))] = g.P + j;
        count += __popc(m);
      }
      tile = i - g.P;
    }
    if (lane == 0) q2k_num[(long long)hd * g.N + i] = count;
  } else if (i < g.N + g.P * g.tchunks) {
    const int u = i - g.N, start = (u % g.tchunks) * g.tlen;
    tile = g.Nv + u;
    count = min(g.tlen, g.N - start);
  } else if (g.vt & 1) {
    tile = g.vt;
    count = 0;
  }
  if (lane == 0 && tile >= 0) {
    vt_cnt[(long long)hd * g.vtpad + tile] = count;
    atomicAdd(&ohist[(long long)hd * (g.N + 1) + count], 1u);
  }
}

__global__ void __launch_bounds__(1024) fwd_order_scan_kernel(uint32_t* hist, int max_count) {
  // Programmatic dependent launch: nothing of the previous kernel is read before it completes (vsa2t.cu).
  griddepcontrol_wait();
  order_scan(hist, max_count);
}
__global__ void fwd_order_scatter_kernel(const int* num, uint32_t* hist, int* order, int heads, int items, int max_count) {
  // Programmatic dependent launch: nothing of the previous kernel is read before it completes (vsa2t.cu).
  griddepcontrol_wait();
  order_scatter(num, hist, order, heads, items, max_count);
}

// ---- Backward: k2q lists ----

// One warp per (head, key block b): the query blocks that kept b, ascending. A text key block is kept by every query
// block; a video tile j by the P text blocks, then the video tiles i that selected it (column j of `selection`, read
// 32 rows at a time). Ascending lists make the backward deterministic per key block and walk the query blocks in the
// order neighbouring key blocks walk them.
__global__ void bwd_lists_kernel(Geo g, const float* __restrict__ selection, int* k2q_idx, int* k2q_num) {
  // Programmatic dependent launch: nothing of the previous kernel is read before it completes (vsa2t.cu).
  griddepcontrol_wait();
  const long long wid = ((long long)blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  const int lane = threadIdx.x & 31;
  if (wid >= (long long)g.heads * g.N) return;
  const int hd = (int)(wid / g.N), b = (int)(wid - (long long)hd * g.N);
  int* list = k2q_idx + wid * g.N;
  if (b < g.P) {
    for (int i = lane; i < g.N; i += 32) list[i] = i;
    if (lane == 0) k2q_num[wid] = g.N;
    return;
  }
  for (int i = lane; i < g.P; i += 32) list[i] = i;
  int count = g.P;
  const float* col = selection + (long long)hd * g.Nv * g.Nv + (b - g.P);
  for (int i0 = 0; i0 < g.Nv; i0 += 32) {
    const int i = i0 + lane;
    const bool f = i < g.Nv && col[(long long)i * g.Nv] > 0.5f;
    const unsigned m = __ballot_sync(0xffffffffu, f);
    if (f) list[count + __popc(m & ((1u << lane) - 1u))] = g.P + i;
    count += __popc(m);
  }
  if (lane == 0) k2q_num[wid] = count;
}

// The histogram of the k2q counts, for the order.
__global__ void bwd_order_hist_kernel(const int* __restrict__ num, uint32_t* hist, int heads, int items, int max_count) {
  // Programmatic dependent launch: nothing of the previous kernel is read before it completes (vsa2t.cu).
  griddepcontrol_wait();
  const long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= (long long)heads * items) return;
  const int hd = (int)(tid / items);
  atomicAdd(&hist[(long long)hd * (max_count + 1) + num[tid]], 1u);
}
__global__ void __launch_bounds__(1024) bwd_order_scan_kernel(uint32_t* hist, int max_count) {
  // Programmatic dependent launch: nothing of the previous kernel is read before it completes (vsa2t.cu).
  griddepcontrol_wait();
  order_scan(hist, max_count);
}
__global__ void bwd_order_scatter_kernel(const int* num, uint32_t* hist, int* order, int heads, int items, int max_count) {
  // Programmatic dependent launch: nothing of the previous kernel is read before it completes (vsa2t.cu).
  griddepcontrol_wait();
  order_scatter(num, hist, order, heads, items, max_count);
}

}  // namespace topk
}  // namespace vsa2t
