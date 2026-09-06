# Ulysses performance on GB200

Enable the fused transport with `FASTVIDEO_ULYSSES_A2A=auto` and rebuild
`fastvideo-kernel` from the same checkout. The tuned path applies to contiguous
BF16 operands with 56 global heads, head dimension 128, and sequence parallelism
of four on GB200. It uses 144 CTAs and exchanges one batch plane at a time.
Older kernel builds retain the original 36-CTA path.

Each rank keeps at most 1 GiB of registered window storage. A 250,000-token
sequence needs 896,000,000 bytes per plane with this geometry, so packed QKV or
QKVG can use the fused path without a window large enough for the entire pack.
The full output still needs its own GPU allocation. Chunking reduces registered
storage; it does not remove activation or optimizer memory costs.

Every call collectively agrees on geometry, window capacity, chunking and CTA
count before entering the kernel. Outputs own their storage, and backward uses
the plan saved by its forward. Contiguity alone does not replace these checks.
Existing unsupported-layout, capture, topology and lifecycle fallbacks remain.

## Long training sequences

By default, grad-tracked operands whose per-plane size exceeds 512 MiB keep
the original transport policy, including NCCL fallback for oversized packs.
This avoids enabling a measured allocation-pressure regression in a 250k-token
FSDP4 training recipe. No-grad inference can use tuned chunks directly.

After configuring sufficient activation memory headroom, enable long training
chunks explicitly:

```bash
export FASTVIDEO_ULYSSES_A2A=auto
export FASTVIDEO_ULYSSES_A2A_LONG_TRAINING=chunked
```

An activation offload policy can provide that headroom. Measure the complete
forward, backward, clipping and optimizer step with resident optimizer state;
transport microbenchmarks alone do not predict long-sequence training speed.
Compare allocator retries alongside step latency. Keep the allocator, FSDP
mesh, attention backend, checkpointing and precision identical across routes.

`FASTVIDEO_ULYSSES_A2A_LONG_TRAINING=auto` restores the conservative policy.
The 512 MiB boundary describes a transport operand; it is not a model-wide
activation-memory estimate. Revalidate other training recipes and hardware.

## Validation

CPU policy regressions live in
`fastvideo/tests/distributed/test_ulysses_h3_policy.py`. On an exclusive group
of four GB200 GPUs, run the native transport gate after rebuilding the kernel:

```bash
FASTVIDEO_ULYSSES_A2A=auto torchrun --standalone --nproc_per_node=4 \
  fastvideo/tests/distributed/check_ulysses_h3_native.py
```

The gate covers exact transport gradients at 32k, 128k and 250k, the long
training opt-in, retained-output ownership, and recovery after rank capability
disagreement. Full-model throughput and absolute model FLOP utilization (MFU)
are separate measurements; this transport gate reports neither.
