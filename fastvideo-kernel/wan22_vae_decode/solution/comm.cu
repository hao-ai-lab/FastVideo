// Halo edge transport: neighbour-only point to point and the seed's ring all-gather, over an
// eight-channels-per-thread edge pack/scatter. Which arm runs is use_allgather's world rule below --
// the ring only at world 2, neighbour-only at world 4 and at every world above it -- which WAN_HALO
// can force either way for a measurement.
// THIS POLICY IS STATED IN THREE PLACES AND THEY MUST AGREE: this file header, the block comment on
// use_allgather, and the WAN_HALO contract in comm.h. An earlier revision of this change corrected one
// of the three and was rightly blocked on the other two, so the cross-reference is the fix for the class
// and not for the instance.
//
// Why the point to point ops go in one coalesced group: ProcessGroupNCCL launches an unbatched p2p op as a
// group of its own on the group's stream, so a rank's p2p ops are stream-ordered against each other, and two
// ranks each waiting to send into the other's not-yet-issued receive is a cycle. (NCCL documents that
// serialization; w24's M1387 reaches the same deadlock from a direct all-to-all of unbatched p2p ops.)
// ProcessGroup::startCoalescing/endCoalescing puts all of this rank's neighbour ops into one NCCL group, which
// removes that ordering: the exchange is one hop, posted and closed with no host block.
#include "comm.h"
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cstdlib>
#include <cstring>

namespace {

cudaStream_t stream() { return at::cuda::getCurrentCUDAStream(); }

// One uint4 is eight bf16 channels; C is a multiple of 8 for every layer that sharded space (256/512/1024).
__global__ void pack_edge8(const uint4* __restrict__ x, uint4* __restrict__ y, int64_t n8, int CV, int T, int H,
                           int W) {
  for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; i < n8; i += (int64_t)gridDim.x * blockDim.x) {
    int64_t row = i / (W * CV), rv = i % (W * CV);
    int t = (int)(row % T), edge = (int)(row / T);
    y[i] = x[((int64_t)t * H + (edge ? H - 2 : 1)) * W * CV + rv];
  }
}

// Scatters the two received edges into the pad rows, eight channels per thread.
__global__ void unpack_edge8(uint4* __restrict__ x, const uint4* __restrict__ e, int64_t n8, int CV, int T, int H,
                             int W, int rank, int world) {
  for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; i < n8; i += (int64_t)gridDim.x * blockDim.x) {
    int64_t row = i / (W * CV), rv = i % (W * CV);
    int t = (int)(row % T), edge = (int)(row / T);
    int peer = rank + (edge ? 1 : -1);  // pad row 0 comes from rank-1, pad row H-1 from rank+1
    uint4 v = make_uint4(0u, 0u, 0u, 0u);
    if (peer >= 0 && peer < world) v = e[((int64_t)peer * 2 * T + (int64_t)(1 - edge) * T + t) * W * CV + rv];
    x[((int64_t)t * H + (edge ? H - 1 : 0)) * W * CV + rv] = v;
  }
}

int grid_for(int64_t work) {
  int64_t blocks = (work + 255) / 256;
  return (int)(blocks > 2048 ? 2048 : (blocks < 1 ? 1 : blocks));
}

// Where each transport is used, when WAN_HALO does not force one. With P the bytes of one edge (half the packed
// buffer): the ring sends (world-1)*2P and receives world*2P, of which this exchange reads 2P (one edge from each
// neighbour); the neighbour path sends and receives one edge per neighbour -- 2P in and out for an interior rank,
// P at the ends. So it is 6P -> 2P each way at world 4 and 2P/4P -> P/P at world 2, i.e. the received bytes fall
// by 3x and 4x. So the policy is the ring only where it has been measured to cost nothing, and the
// neighbour path everywhere else:
//   world 2   the two arms measured the same, and the copy counts are equal there too. J817, three
//             alternating rounds in one job at f121x2, minima 698.6 ms (ring) against 696.5 ms
//             (neighbour), 0.3% apart, J570's 2.3% the other way sitting inside its own run-to-run
//             spread; and J3057 ran none / ag / p2p at f481x2 in one job, 2305.6 / 2309.7 / 2293.2.
//   world 4   neighbour. J2211's own `for A in ag p2p ag p2p` at f481x4: ag 2462.9 / 2244.5 against
//             p2p 1010.6 / 1017.5, the p2p arm's A/A 0.68 %. The ring moves (world-1) copies each
//             way, the neighbour path one.
//   above 4   neighbour, and this rung is chosen on COPY COUNTS rather than on a measurement: the
//             ring moves (world-1) copies each way against the neighbour path's 1, so the ratio grows
//             with the world size, and no world above 4 has been measured on either arm here -- a
//             cross-tray one has never run. The world-4 gap (2.32x) is the only scaled evidence and
//             it is the smallest world where the two differ. Whoever first gets eight GPUs should
//             treat this rung as the thing to measure, not as settled.
// What that leaves: every world the harness runs is unchanged -- 1 by the early return, 2 true, 4 false
// -- and only world 3 and world 5 and up change.
// WAN_HALO forces either arm for a measurement.
bool use_allgather(int world) {
  if (world < 2) return true;  // one rank has no neighbour to talk to
  const int forced = halo_transport();
  if (forced == kAllGather) return true;
  if (forced == kP2P) return false;
  return world == 2;
}

}  // namespace

HaloStats& halo_stats() {
  static HaloStats stats;
  return stats;
}

int halo_transport() {
  static const int mode = [] {
    const char* s = std::getenv("WAN_HALO");
    if (!s) return (int)kPolicy;
    if (std::strcmp(s, "ag") == 0 || std::strcmp(s, "allgather") == 0) return (int)kAllGather;
    if (std::strcmp(s, "p2p") == 0 || std::strcmp(s, "neighbour") == 0) return (int)kP2P;
    return (int)kPolicy;
  }();
  return mode;
}

at::Tensor halo_pack(const at::Tensor& padded) {
  const int C = (int)padded.size(1), T = (int)padded.size(2), H = (int)padded.size(3), W = (int)padded.size(4);
  TORCH_CHECK(C % 8 == 0, "halo_pack needs C a multiple of 8");
  at::Tensor e = at::empty({2, T, W, C}, padded.options());
  const int CV = C / 8;
  int64_t n8 = (int64_t)2 * T * W * CV;
  pack_edge8<<<grid_for(n8), 256, 0, stream()>>>((const uint4*)padded.data_ptr(), (uint4*)e.data_ptr(), n8, CV, T, H,
                                                 W);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return e;
}

void halo_unpack(at::Tensor& padded, const at::Tensor& edges, int rank, int world) {
  const int C = (int)padded.size(1), T = (int)padded.size(2), H = (int)padded.size(3), W = (int)padded.size(4);
  TORCH_CHECK(C % 8 == 0, "halo_unpack needs C a multiple of 8");
  const int CV = C / 8;
  int64_t n8 = (int64_t)2 * T * W * CV;
  unpack_edge8<<<grid_for(n8), 256, 0, stream()>>>((uint4*)padded.data_ptr(), (const uint4*)edges.data_ptr(), n8, CV,
                                                   T, H, W, rank, world);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

HaloHandle post_edges(const PG& pg, const at::Tensor& send, at::Tensor& recv, int rank, int world,
                      cudaStream_t comm) {
  HaloHandle h;
  h.started = true;
  // Both buffers are flat here: the p2p slices are element offsets (narrowing the packed [2,T,W,C] on dim 0 would
  // index a dimension of size 2, which is what round 1 of this PR did).
  TORCH_CHECK(send.is_contiguous(), "halo edges must be contiguous");
  TORCH_CHECK(recv.is_contiguous(), "the halo receive buffer must be contiguous");
  h.send = send.reshape({-1});
  h.recv = recv.reshape({-1});
  HaloStats& s = halo_stats();
  ++s.exchanges;
  const int64_t bytes = send.numel() * (int64_t)send.element_size();
  // The transport is ordered on `comm`: ProcessGroupNCCL makes its own stream wait for the current stream and
  // the current stream wait for its end event, so this guard is what puts the collective after the pack.
  c10::cuda::CUDAStreamGuard guard(c10::cuda::getStreamFromExternal(comm, c10::cuda::current_device()));
  if (use_allgather(world)) {
    // A ring all-gather: each rank's buffer goes round the ring, (world-1) copies each way.
    at::Tensor in = send;
    h.keep.push_back(in);
    h.keep.push_back(recv);
    h.works.push_back(pg->_allgather_base(recv, in));
    s.bytes_sent += bytes * (world - 1);
    s.bytes_received += bytes * (world - 1);
    return h;
  }
  // The packed edges are [2,T,W,C]; the p2p slices below are element offsets, so they are taken flat (above).
  h.keep.push_back(send);
  h.keep.push_back(h.send);
  h.keep.push_back(recv);
  // Split on rank 0 = rows [0, H/world), so rank+1 is the band BELOW me and rank-1 the band above. Each rank
  // sends its first real row up to rank-1 and its last real row down to rank+1 -- one edge per neighbour, not
  // both to each -- and reads back only the two neighbour slots of `recv`; its other world-2 slots are never
  // written and stay uninitialised, which halo_unpack is careful not to read (it touches slot rank-1's edge 1
  // and slot rank+1's edge 0 and nothing else). On the all-gather path every slot is filled. Both leave the
  // same two edges in the same places.
  const int64_t half = send.numel() / 2;  // one edge: [T,W,C]; edge 0 first, edge 1 second
  const int neighbours = (rank > 0 ? 1 : 0) + (rank + 1 < world ? 1 : 0);
  s.bytes_sent += (int64_t)neighbours * half * send.element_size();
  s.bytes_received += (int64_t)neighbours * half * send.element_size();
  // Every tensor operation happens before the group opens: a throw between startCoalescing and endCoalescing
  // would leave the NCCL group open and hang every rank instead of failing this one.
  const bool hasLower = rank + 1 < world, hasUpper = rank > 0;
  std::vector<at::Tensor> recvLower, sendUpper, recvUpper, sendLower;
  if (hasLower) {
    at::Tensor v = h.recv.narrow(0, (int64_t)(rank + 1) * 2 * half, half);  // below: its edge 0
    h.keep.push_back(v);
    recvLower.push_back(v);
  }
  if (hasUpper) {
    at::Tensor v = h.send.narrow(0, 0, half);  // above: my first real row
    h.keep.push_back(v);
    sendUpper.push_back(v);
  }
  if (hasUpper) {
    at::Tensor v = h.recv.narrow(0, (int64_t)(rank - 1) * 2 * half + half, half);  // above: its edge 1
    h.keep.push_back(v);
    recvUpper.push_back(v);
  }
  if (hasLower) {
    at::Tensor v = h.send.narrow(0, half, half);  // below: my last real row
    h.keep.push_back(v);
    sendLower.push_back(v);
  }
  pg->startCoalescing(c10::DeviceType::CUDA);
  if (hasLower) pg->recv(recvLower, rank + 1, 0);
  if (hasUpper) pg->send(sendUpper, rank - 1, 0);
  if (hasUpper) pg->recv(recvUpper, rank - 1, 0);
  if (hasLower) pg->send(sendLower, rank + 1, 0);
  // The ops inside a coalesced group carry no work of their own: the group's work is the one to wait on.
  h.works.push_back(pg->endCoalescing(c10::DeviceType::CUDA));
  return h;
}

void wait_edges(HaloHandle& h) {
  if (!h.started) return;
  // Each wait installs the collective's completion as an event wait on the current stream and does not block
  // the host, so the caller's next kernel on that stream is ordered after the transfer.
  for (auto& w : h.works) w->wait();
  h.works.clear();
  h.started = false;
}

at::Tensor halo_exchange(const at::Tensor& send, const PG& pg, int rank, int world) {
  at::Tensor recv = at::empty({world * send.numel()}, send.options());
  HaloHandle h = post_edges(pg, send, recv, rank, world, at::cuda::getCurrentCUDAStream());
  wait_edges(h);
  return recv;
}
