// vsa2t.h -- the host interface of the VSA2 training step: the geometry of a call, and the two halves (forward and
// backward), each of which launches its kernels on one stream and carves its scratch from one allocation.
//
// Layout (reference.py): the sequence is N = P + N_v blocks of 128 rows, P text blocks first (the text in order, the
// last block padded at its end), then the N_v video tiles of 4 x 4 x 8 tokens in raster order of the tile grid (a
// tile's tokens in raster order of the tile, padding where the tile leaves the latent). Router windows are 2 x 2 x 2
// tokens, 16 per tile. Every tensor is [heads][S = N * 128][128]; the batch and head axes are flattened into `heads`.
#pragma once
#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>

namespace vsa2t {

constexpr int kRows = 128;    // rows of a block
constexpr int kDim = 128;     // head dimension
constexpr int kWindows = 16;  // router windows per video tile

// The geometry of a call; plain data, passed by value to the kernels.
struct Geo {
  int heads;          // batch * heads
  int text, P;        // text tokens and text blocks
  int t, h, w;        // the latent's extent in tokens
  int nt, nh, nw;     // the tile grid
  int Nv, N;          // video tiles, blocks
  long long S;        // rows: N * 128
  int topk;           // tiles kept per video query tile, on average
  long long K;        // pairs the top-K keeps per head, min(topk * Nv, Nv * Nv)
  int rtiles;         // router tiles: windows (Nv * 16) in tiles of 128
  // The fine forward's tiles: the Nv video tiles, then each text block's key range cut into tchunks chunks of tlen
  // key blocks (a text query block keeps all N, ten times a video tile's count: whole, it would set the kernel's
  // length), then a phantom when their number is odd (the forward runs tiles in pairs).
  int tchunks, tlen;  // chunks per text block, key blocks per chunk (the last may be shorter)
  int vt, vtpad;      // tiles: Nv + P * tchunks, rounded up to even
};

Geo make_geo(int heads, int text, int t, int h, int w, int topk);

// q, k, v, gate: [heads][S][128] bf16. Writes out (bf16, like q), selection ([heads][Nv][Nv] fp32) and lse
// ([heads][S] fp32).
size_t forward_scratch_bytes(const Geo& g);
void forward(const Geo& g, const void* q, const void* k, const void* v, const void* gate, void* out, float* selection,
             float* lse, void* scratch, cudaStream_t stream);

// Reads only its arguments (the forward's out, lse and selection among them). Writes dq, dk, dv and dgate (bf16).
size_t backward_scratch_bytes(const Geo& g);
void backward(const Geo& g, const void* q, const void* k, const void* v, const void* gate, const void* out,
              const float* lse, const float* selection, const void* dout, void* dq, void* dk, void* dv, void* dgate,
              void* scratch, cudaStream_t stream);

}  // namespace vsa2t
