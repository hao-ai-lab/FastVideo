# FastVideo FA4 source fork

This directory vendors the `flash_attn.cute` Python package from
[Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention) at
commit [`82d6441eec5d4dfec120153db2c0145ae855a083`](https://github.com/Dao-AILab/flash-attention/commit/82d6441eec5d4dfec120153db2c0145ae855a083).
The upstream source path is `flash_attn/cute/`, and its BSD-3-Clause license is
preserved in `LICENSE`.

FastVideo keeps this as a first-class source fork because its video sparse
attention kernels need FA4 scheduling and tile-shape changes that are developed
and benchmarked together with `fastvideo-kernel`. The distribution intentionally
retains the upstream name `flash-attn-4` and import namespace `flash_attn.cute`.

The forward-kernel delta currently contains FastVideo's single-Q-block,
two-KV-stream Q128/KV64 schedule, four-slot score/probability double buffering,
and FP32 online-softmax merge. The Q64/KV64 primitives were adapted from
upstream's guarded smaller-tile commit
[`526c18d25bcbc7fc7d6740ab3c7c84ed2d42cb0b`](https://github.com/Dao-AILab/flash-attention/commit/526c18d25bcbc7fc7d6740ab3c7c84ed2d42cb0b),
then extended with the same two-stream sparse traversal and merge. These are
forward-only specializations; the existing upstream paths remain the fallback
outside their explicit SM100/SM110 dtype, shape, and metadata gates.

Install the working tree without resolving dependencies:

```bash
uv pip install --no-deps --editable fastvideo-kernel/fa4
```

The initial performance comparison came from FlashInfer issue
[#4554](https://github.com/flashinfer-ai/flashinfer/issues/4554) and its
[benchmark harness](https://gist.github.com/SolitaryThinker/90a1d1447929fc38dc509c1852e76532)
at gist revision `e15ac9066f23ef3690e33e1cc1fdac45b4b9099f`.

When refreshing from upstream, copy only `flash_attn/cute/` from an explicitly
checked-out commit, retain this file and FastVideo-specific changes, update the
commit and package version above, and rerun the FA4 correctness and GB200
performance gates before accepting the refresh.
