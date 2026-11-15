// Halo edge transport for the row-band split: the two boundary rows a 3x3 convolution needs from its
// neighbours, moved with the cheapest mechanism available. Representation-agnostic by design: it carries
// whatever `halo_pack` packs (today the normalized NDHWC bf16 edge rows; the fusion work may change that),
// and it leaves the packed layout and `halo_unpack` untouched, so the decoded video is bit-identical to the
// all-gather it replaces.
//
// Two ways in:
//   halo_exchange(...)                 one call: allocates the recv buffer, transports and closes. What
//                                      Decoder::conv uses today.
//   post_edges(...) / wait_edges(...)   the split seam for overlapping the transport with interior compute:
//                                      post enqueues the exchange and returns a handle without a host block,
//                                      wait closes it. `halo_pack` may run on one stream and the transport on
//                                      another: post_edges takes the stream the transport is ordered on.
#pragma once
#include <ATen/ATen.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>
#include <torch/csrc/distributed/c10d/Work.hpp>
#include <c10/util/intrusive_ptr.h>
#include <cuda_runtime.h>
#include <vector>

using PG = c10::intrusive_ptr<c10d::ProcessGroup>;

// Which transport the halo exchange uses, from WAN_HALO once per process. With nothing set the world-size rule
// in comm.cu decides; the variable is for forcing an arm in a measurement (both must be the same on every rank
// of a world: one rank in a collective and one in a p2p group would never meet).
//   unset: the rule -- the ring all-gather only at world 2, where both arms have been measured equal, and the
//          neighbour path at world 4 and at every world above it. (World 1 has no exchange at all: ph is 0
//          there, so nothing calls this.)
//   "ag" or "allgather": the ring collective, whatever the world size.
//   "p2p" or "neighbour": the neighbour path, whatever the world size. This is what the rule selects above
//          world 4, and the neighbour path across trays has never been measured: nothing at that world size
//          has, on either arm.
enum HaloTransport { kPolicy = 0, kAllGather = 1, kP2P = 2 };
int halo_transport();

// The exchange sites and the bytes they moved. Debug instrumentation only: nothing on the run path reads them,
// and with WAN_HALO_STATS unset they cost one increment per exchange.
struct HaloStats {
  int64_t exchanges = 0, bytes_sent = 0, bytes_received = 0;
};
HaloStats& halo_stats();

// Gather the two edges a rank must send: the first and the last real row of the padded NDHWC tensor
// [1,C,T,H,W] (H includes exactly one pad row per side), as [2,T,W,C] contiguous, edge-major: edge 0 is the
// first real row (H index 1), edge 1 the last (H index H-2). Same layout as the seed's pack_edges, eight
// channels per thread.
at::Tensor halo_pack(const at::Tensor& padded);

// Scatter the received edges into the pad rows: row 0 takes rank-1's edge 1, row H-1 takes rank+1's edge 0,
// and a rank with no neighbour on that side gets zeros. `edges` is the transport's `recv` buffer, and rank+1
// is the band *below* (higher rows), since the split gives rank 0 the rows [0, H/world).
void halo_unpack(at::Tensor& padded, const at::Tensor& edges, int rank, int world);

// The transport, split in two. `send` is halo_pack's [2,T,W,C]; `recv` is the caller's buffer of
// world*2*T*W*C elements in the all-gather's layout (slot r is rank r's two edges), which is what
// halo_unpack reads. The handle owns the works and keeps every send and recv view alive until it is waited.
struct HaloHandle {
  std::vector<c10::intrusive_ptr<c10d::Work>> works;
  std::vector<at::Tensor> keep;
  at::Tensor send, recv;  // flat views the p2p slices index
  bool started = false;
  HaloHandle() = default;
  HaloHandle(HaloHandle&&) = default;
  HaloHandle& operator=(HaloHandle&&) = default;
  HaloHandle(const HaloHandle&) = delete;
  HaloHandle& operator=(const HaloHandle&) = delete;
};

// Enqueue only: no host block, and nothing added to the compute stream. `comm` is the stream the transport is
// ordered on; it must already carry whatever ordering the caller needs (a wait on the pack, say), and the
// unpack must not run on a different stream before wait_edges has been called on it.
HaloHandle post_edges(const PG& pg, const at::Tensor& send, at::Tensor& recv, int rank, int world,
                      cudaStream_t comm);

// Close it: this orders the CALLER'S CURRENT STREAM after the transfer and returns at once. WorkNCCL::wait is
// an event wait on the stream that is current at the call, not a CPU block. So the next kernel enqueued on
// that stream sees the edges -- and a kernel enqueued on any other stream does not.
void wait_edges(HaloHandle& h);

// Allocate the recv buffer, post, close and return the edges in halo_unpack's layout.
at::Tensor halo_exchange(const at::Tensor& send, const PG& pg, int rank, int world);
