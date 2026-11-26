// common.cuh -- the geometry on the device: which rows of a block are tokens, which router windows hold one, and the
// lookup of the tables of both that the stage kernels read (rowmask: a bit per row of each block; winmask: a bit per
// window of each video tile; the prep kernels write them).
#pragma once
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

#include "vsa2t.h"

namespace vsa2t {

// The tile of video block v (0 <= v < Nv) in the tile grid.
struct TileCoord {
  int ti, hi, wi;
};
__device__ __forceinline__ TileCoord tile_coord(const Geo& g, int v) {
  const int per_t = g.nh * g.nw;
  TileCoord c;
  c.ti = v / per_t;
  const int rem = v - c.ti * per_t;
  c.hi = rem / g.nw;
  c.wi = rem - c.hi * g.nw;
  return c;
}

// Whether row r (0..127) of block b is a token: text rows up to the text's length, video slots inside the latent.
// A tile's slot r is (r / 32, r / 8 % 4, r % 8) in the tile.
__device__ __forceinline__ bool row_valid(const Geo& g, int b, int r) {
  if (b < g.P) return (long long)b * kRows + r < g.text;
  const TileCoord c = tile_coord(g, b - g.P);
  return c.ti * 4 + (r >> 5) < g.t && c.hi * 4 + ((r >> 3) & 3) < g.h && c.wi * 8 + (r & 7) < g.w;
}

// Tokens in block b.
__device__ __forceinline__ int block_count(const Geo& g, int b) {
  if (b < g.P) return (int)min((long long)kRows, (long long)g.text - (long long)b * kRows);
  const TileCoord c = tile_coord(g, b - g.P);
  return min(4, g.t - c.ti * 4) * min(4, g.h - c.hi * 4) * min(8, g.w - c.wi * 8);
}

// Router window gw of window index (gw / 16: video tile, gw % 16: window (wt, wh, ww) = (g / 8, g / 4 % 2, g % 4), its
// slots 2 x 2 x 2 from (2 wt, 2 wh, 2 ww)). A window holds a token iff its first slot is one.
__device__ __forceinline__ bool window_valid(const Geo& g, long long gw) {
  if (gw >= (long long)g.Nv * kWindows) return false;
  const int v = (int)(gw >> 4), wg = (int)(gw & 15);
  const TileCoord c = tile_coord(g, v);
  return c.ti * 4 + 2 * (wg >> 3) < g.t && c.hi * 4 + 2 * ((wg >> 2) & 1) < g.h && c.wi * 8 + 2 * (wg & 3) < g.w;
}

__device__ __forceinline__ bool mask_bit(const uint32_t* mask, long long tile, int r) {
  return (__ldg(mask + tile * 4 + (r >> 5)) >> (r & 31)) & 1u;
}

// The window bits of video tiles v and v + 1 (winmask: 16 bits per tile, bit g for window g), 0 past the last tile.
__device__ __forceinline__ uint32_t win_word(const uint16_t* winmask, int nv, long long v) {
  const uint32_t lo = v < nv ? __ldg(winmask + v) : 0u, hi = v + 1 < nv ? __ldg(winmask + v + 1) : 0u;
  return lo | (hi << 16);
}

}  // namespace vsa2t
