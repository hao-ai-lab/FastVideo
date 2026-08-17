# FA4 VSA block-64 handoff

## Status

This branch vendors FlashAttention-4's CuTe source as a first-class FastVideo
fork, adds the issue-#4554 benchmark, and implements forward-only SM100 sparse
attention paths for Q128/KV64 and Q64/KV64.

The requested single-Q-block/two-KV-workstream design is implemented. The two
workstreams split the canonical sparse traversal by ordinal parity, maintain
independent online-softmax state, and merge their FP32 output accumulators in
the correction epilogue. Q128/KV64 additionally uses four score/probability
slots to overlap successive blocks.

The measured result is not fully zero-loss:

- Native Q128/KV64 reaches about 90% of raw FA4 Q256/KV256 throughput.
- Native Q64/KV64 reaches about 55% of raw FA4 Q256/KV256 throughput.
- A newly added production-only VSA256 adapter reaches 80-86% of the public
  Q256 wrapper. It is selected with
  `FASTVIDEO_VSA_FA4_BLOCK_SHAPE=64x64`, but physically runs Q128/KV64 by
  pairing adjacent Q64 children whose sparse lists are identical at the
  original Q256 metadata boundary. It must not be presented as native
  Q64/KV64 performance.

## Source and provenance

- Fork: `fastvideo-kernel/fa4/`
- Upstream base: Dao-AILab/flash-attention commit
  `82d6441eec5d4dfec120153db2c0145ae855a083`
- Q64 primitives adapted from upstream commit
  `526c18d25bcbc7fc7d6740ab3c7c84ed2d42cb0b`
- Full refresh instructions: `fastvideo-kernel/fa4/UPSTREAM.md`
- Root `pyproject.toml` resolves `flash-attn-4` to the editable local fork.

The fork retains the upstream `flash-attn-4` distribution name and
`flash_attn.cute` import namespace.

## Main implementation

- `fastvideo-kernel/fa4/interface.py`
  - Strict compile-time gates for the dual-stream and double-buffer paths.
  - `FASTVIDEO_FA4_VSA_DUAL_STREAM=0` retains the upstream-style single-stream
    fallback.
  - `FASTVIDEO_FA4_VSA_SP_DOUBLE_BUFFER=0` retains the earlier dual-stream
    schedule for rollback and comparison.
- `fastvideo-kernel/fa4/block_sparse_utils.py`
  - Canonical ordinal-parity sparse traversal shared by load, MMA, and
    softmax code.
  - Correct handling for masked-only, full-only, mixed, odd-length, and empty
    streams.
- `fastvideo-kernel/fa4/flash_fwd_sm100.py`
  - One Q block, two KV workstreams, four S/P slots for Q128/KV64, independent
    phase tracking, and stable FP32 merge.
  - Native M64 score/output layouts and row-pair softmax support.
- `fastvideo-kernel/python/fastvideo_kernel/block_sparse_attn_256.py`
  - Public VSA256 shape selection via `FASTVIDEO_VSA_FA4_BLOCK_SHAPE`.
  - The `64x64` option is the newly added coalescing adapter described above.
- `fastvideo-kernel/python/fastvideo_kernel/block_sparse_attn_cute_fwd.py`
  - Direct Q128/KV64 and native Q64/KV64 BSHD entrypoints.

## Benchmark

Harness:

```bash
fastvideo-kernel/benchmarks/bench_vsa_blackwell.py
```

It derives from FlashInfer issue #4554 and immutable gist revision
`e15ac9066f23ef3690e33e1cc1fdac45b4b9099f`. The default `exact256` mask mode
uses identical selected token pairs across block shapes, checks every output
against FP32 masked SDPA, and reports sparse-aware algorithmic TFLOP/s plus MFU.

Four-GPU command:

```bash
PYTHONPATH="$PWD/fastvideo-kernel/fa4:$PWD/fastvideo-kernel/python" \
CUTE_DSL_ENABLE_TVM_FFI=1 \
FASTVIDEO_VSA_CUTEDSL=1 \
FASTVIDEO_FA4_VSA_DUAL_STREAM=1 \
FASTVIDEO_FA4_VSA_SP_DOUBLE_BUFFER=1 \
uv run --no-sync torchrun --standalone --nproc-per-node=4 \
  fastvideo-kernel/benchmarks/bench_vsa_blackwell.py \
  --seq_lens 32768 --sparsities dense 90 --mask_mode exact256 \
  --arms cutedsl256 fa4_wrapper fa4 \
  --block_shapes 256x256 128x64 64x64 \
  --warmup 5 --rep 20 --out /tmp/fa4_final_tray.json
```

Environment used for the final run:

- 4x NVIDIA GB200, SM100
- PyTorch `2.12.0+cu130`
- CUDA 13.0, driver 580.159.04
- BF16, B=1, H=12, D=128, S=32768
- MFU denominator: 2.5 PFLOP/s dense BF16 per GPU, 10 PFLOP/s per tray

### Four-GPU aggregate results

| Raw schedule | Dense PFLOP/s | Dense MFU | 90% sparse PFLOP/s | 90% sparse MFU | Relative to raw Q256 |
|---|---:|---:|---:|---:|---:|
| Q256/KV256 | 6.353 | 63.53% | 5.852 | 58.52% | 100% |
| Q128/KV64 | 5.764 | 57.64% | 5.234 | 52.34% | 90.7% dense / 89.5% sparse |
| Native Q64/KV64 | 3.408 | 34.08% | 3.215 | 32.15% | 53.6% dense / 54.9% sparse |

| Public VSA256 route | Dense PFLOP/s | Dense MFU | 90% sparse PFLOP/s | 90% sparse MFU | Relative to public Q256 |
|---|---:|---:|---:|---:|---:|
| Default Q256 | 6.185 | 61.85% | 4.494 | 44.94% | 100% |
| Added `64x64` coalescing adapter | 5.317 | 53.17% | 3.584 | 35.84% | 86.0% dense / 79.7% sparse |

Fixed-base NCU showed that score/probability double buffering improves the
Q128/KV64 end-to-end wrapper by 8.74% dense and 4.35% at 90% sparsity over
the retained legacy dual-stream schedule.

## Validation

Final GB200 runs:

```bash
CUDA_VISIBLE_DEVICES=3 \
PYTHONPATH="$PWD/fastvideo-kernel/fa4:$PWD/fastvideo-kernel/python" \
CUTE_DSL_ENABLE_TVM_FFI=1 \
FASTVIDEO_VSA_CUTEDSL=1 \
FASTVIDEO_FA4_VSA_DUAL_STREAM=1 \
FASTVIDEO_FA4_VSA_SP_DOUBLE_BUFFER=1 \
uv run --no-sync pytest -vs \
  fastvideo-kernel/tests/test_fa4_vsa_block_shapes.py
```

Result: 14 passed. Coverage includes selected counts 0-5, odd/even and
one-stream-empty cases, persistent phase transitions, masked/full/mixed
metadata, variable block sizes, native Q64, Q128, public VSA256 routes, and
the legacy double-buffer opt-out.

Existing compatibility tests:

```bash
uv run --no-sync pytest -vs \
  fastvideo-kernel/tests/test_vsa256_forward.py \
  fastvideo-kernel/tests/test_vsa256_forward_cross.py \
  fastvideo-kernel/tests/test_vsa256_forward_vbs.py
```

Result: 4 passed.

Other completed gates:

- `pre-commit run --files ...`: passed for all scoped files.
- `python -m py_compile`: passed for the fork, wrappers, tests, and benchmark.
- FA4 sdist and wheel builds: passed.
- `twine check`: passed for both distributions.
- Isolated wheel import: resolved `flash_attn.cute` and the expected local
  version.
- `git diff --check`: passed.

## Known limitations and follow-up

- These optimized specializations are forward-only.
- Native Q128/KV64 and Q64/KV64 do not meet the original zero-loss/30%-loss
  targets; keep the native and coalesced-adapter numbers separate.
- The existing kernel CI lane uses H100. The SM100/SM110 tests skip there, so
  the four-GB200 local run is currently the only hardware regression gate.
- The workspace's untracked `uv.lock` predates the local source selection and
  was deliberately not overwritten. `uv lock --dry-run --offline` resolves
  only the expected `flash-attn-4` source/version update.
- Rejected performance experiments include a second UMMA issuer, paired wait
  batching, cooperative 2-CTA execution, cluster multicast, CTA-wide stats
  barriers, and correction wait reordering. All either regressed or failed to
  improve the fixed-base profiles and were not retained.
