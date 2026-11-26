# How this kernel was optimized: what was measured

The record of the optimization that produced `solution/` (2026-10-23 to 10-25, on one GB200 tray): where the time
is, what was merged, and what was measured and ruled out, so that later work starts from it. Numbers marked "dev"
were taken at the dev GPUs' locked 1132 MHz, in one job with their base. PR numbers are those of the optimization's
own repository.

## Where the time is

The breakdown of this main (one call, flushed L2), in ms and share of the call:

| | s8 | s20 |
|---|---|---|
| call | 1.962 | 8.613 |
| `bwd_fine` | 1.118 (57 %) | 5.458 (63 %) |
| `fwd_fine` | 0.519 (26 %) | 2.268 (26 %) |
| router stage (9 kernels) | 0.108 (6 %) | 0.419 (5 %) |
| everything else (15 kernels) | ~0.22 (11 %) | ~0.47 (5 %) |

Each stage against its floor at 2062 MHz: router x3.5 (s8) to x2.4 (s20), forward x2.4 to x1.9, backward x2.1 to x1.9.
The two fine kernels are 83-90 % of the call: that is where a large gain has to come from.

## What was merged, from the first solution (4.20 ms) to this one

- **`bwd_fine`**: persistent CTAs that claim work items by cluster launch control through a two-slot ring (PR270,
  carried by PR288); the softmax warps' register grant raised 136 -> 160 (PR221, 2.1 %); the per-row delta
  double-buffered (PR44); 32-bit descriptor offsets in the MMA warp, which avoids a ptxas 13.0 miscompile that drops a
  descriptor's high word (PR177); an `egress_ack` barrier that bounds `egress_go`'s lead to one phase and closes a
  lost-phase hang reachable under a delayed warp (PR296, PR300).
- **`bwd_post`**: 3 CTAs per SM (PR131), a bf16 staging tile and a two-half pipeline (PR355), half a block per CTA
  (PR371).
- **The coarse branch**: from nine fp16 `mma.sync` launches through N x N scratch to three flash-style kernels with a
  DSMEM merge (PR288, 1.2 %); then `tcgen05` with every value held as three bf16 components with independent exponents,
  8 products, fp32 accumulation (PR348, PR406). It is exact on the numeric domain (finite bf16 values with nonzero
  magnitudes in [2^-60, 2^16]).
- **The router**: fp16 window means (PR253); the sweep's pool cut out and its key range split (PR382, 1.1 %); a race
  in its slot drain repaired (PR395).
- **The top-K**: ties at the threshold broken by the lowest row-major index, not by atomic arrival order (PR157).

## The backward fine kernel, measured

- Per kept pair the five products take 2,560 cycles at peak; the loop runs at ~3,750 cycles per pair, ~68 % of the
  tensor ideal, on every workload (the pair loop is ~100 % of `bwd_fine`).
- The TMEM plan is full: S^T | dV | dP^T (aliased by dQ) | dK, 512 of 512 columns. dQ cannot be an operand from
  TMEM in a key-stationary kernel: tcgen05 writes a thread's registers only into its own lane, and the lanes hold keys.
- Timing probes on one tree (results wrong by design), dev, same job:

  | probe | `bwd_fine` s8 | s20 |
  |---|---|---|
  | half the MMA work on S^T, dP^T, dV, dK | -5.1 % | -4.5 % |
  | the softmax does half its columns | +0.4 % | +1.6 % |
  | dQ owns its TMEM (no alias wait) | -0.06 % | -0.04 % |
  | both of the last two | +1.3 % | +1.9 % |
  | no per-column LDS of lse/delta | +0.3 % | +0.5 % |
  | no dQ bulk reduce, staging stores kept | -1.7 % | -1.1 % |
  | no dQ staging stores and no reduce | **-6.7 %** | **-6.4 %** |

  The one term that moves the kernel is the dQ drain's 64 KB of shared-memory staging per pair, which slows the other
  warps' shared-memory traffic. Every other single term, removed alone, gives nothing.
- The dQ reduce port, measured by a probe of CTAs that only push: `cp.reduce.async.bulk.add.f32` reaches ~23 B/clk per SM with all
  152 SMs pushing (6.5 TB/s); depth beyond 1 buys nothing; pushes under 8 KB halve the rate; `red.global.v4` from
  registers reaches ~14 B/clk in the row-per-thread layout.
- Ruled out for this drain (4 warps, 2 x 16 KB staging, bulk reduce), each measured slower: smaller pushes or fewer
  bytes per push (+3 to +4 %), pushing only valid rows (+3.4 %), any share of the drain moved to `red.global` (+3.8 %
  for a quarter, +44 % for all of it), a dQ phase-shift schedule (+15 to +17 % of the call), 64-column dQ pushes with
  one slot (+15 to +17 %), dynamic partial-key M64 classes (+45 to +50 %; with the branch hoisted, +16 to +18 %).
- Closed by analysis: the 2-CTA FA4-style backward (FA4's own sm_100 backward rules out block sparsity in 2-CTA mode,
  and pre-reducing two key blocks' dQ pays only on the text key blocks' shared lists: a few percent of `bwd_fine`).
- What could still pay: a dQ egress that does not go through shared memory and is not slower than the bulk engine
  (none is known on sm_100), or a different decomposition of the backward that produces fewer fp32 dQ partials per
  pair or takes the staging off the other warps' path.
- Scheduling: a text key block is kept by every query block, so it is one long work item (394 pairs at s8), and the
  makespan is max(the longest item, the mean load per SM). The mean load binds today. A `bwd_fine` much faster per pair
  would have to split the text key blocks' query lists over several CTAs and reduce their dK/dV partials, as the
  forward already splits text query blocks.
- The tail kernels (`bwd_post`, `bwd_rows`, `bwd_prep`, `fwd_prep`) take ~190 us at s8 against ~96 us of floors; they
  are bound by mixed read and write DRAM traffic, not by bytes in flight (a bulk-copy `bwd_rows` was 20 % slower).

## The forward fine kernel, measured

- Its steady state is softmax-bound. In an instrumented timeline (+10 % overhead) the softmax warps are busy ~84 % of the
  period (P for one tile ~2,000 cycles, its row sum ~350), the MMA waits for P ~20 % of it, and the loads ~4 %. The
  rescale is off the critical path: the correction finishes 330-500 cycles before the softmax needs it.
- About a quarter of `fwd_fine` may lie outside the steady state (item transitions, start, tail): a model, not a
  measurement. The epilogue loads each chunk's gate and coarse output inside its 8-chunk loop, about 8 dependent HBM
  round trips per tile; prefetching them is an open lead.
- Removing the exponentials altogether measured -10 % of the forward, not -40 %: the SFU is not its wall.
- Rewrites that were correct on all five and the edges but slower (fine forward at s8, base ~790 us): one query tile
  per CTA with two resident CTAs (+23 %), its full-128-key sibling (+35 %), a 12-warp column-split softmax (+22 %),
  two query tiles per CTA (+55 %; braided, +24 %), a shared S with a separate correction warpgroup (+16 to +20 %),
  shared P with independent S (+25 %), M128/N256 tiles (+27 %). Moving 25-50 % of the exponentials to the FMA pipe did
  not rescue the column split. Removing every local-memory access gave no gain by itself.

## The router and the rest

- The router stage is 5-6 % of the call at 2.4-3.5 x its floor. A 256-key router variant measured no gain.
- `bwd_lists` and the `bwd_order_*` kernels take ~19 us at s8 and read only `selection`: they could run under the
  latency-bound coarse kernels with programmatic dependent launch (~0.4 %).

## How to measure

- Compare arms in one job, alternated, in separate processes: two builds of one module in one process share kernel
  names, and only the first loaded runs. Ratios across jobs differ by ~0.4 % between GPUs.
- Changes that move data change the clock under the power cap: judge them with sustained `tools/bench.py` runs, not
  only with call-by-call A/B timers.
