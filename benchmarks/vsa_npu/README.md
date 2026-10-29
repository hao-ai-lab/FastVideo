# Opt-in Ascend Wan VSA Forward

Complete existing Wan VSA forward on Ascend 910B: tiling, compression,
compressed attention, threshold selection, sparse attention, gates and
restoration. This is not a new attention algorithm or a demonstrated full-video
speedup. Dense attention has different semantics.

## Supported Domain

BF16 eager inference; B=1, D=64/128, equal positive Q/K/V head counts,
noncausal/default scale, 64-token logical tiles, fixed physical 64x64 schedule,
at most 624 logical blocks. SP1/SP4 use existing sequence/head redistribution;
SP4 requires heads divisible by four. Gate is absent or matches Q's complete
shape/dtype/device. Unsupported calls raise; dense is not a fallback. Default
NPU SDPA dispatch is unchanged. FP16, backward and torch.compile are unsupported.
No opaque custom op is registered.

Compression/32-step selection are ported from upstream `fused_compress_topk.py`
at `6743f4b4b76634f7854942eddf49a68fe65e66f0`; sparse forward derives from
`block_sparse_attn_triton.py` there, retaining FlashAttention/OpenAI attribution.
The adapter, bounded NPU schedule, workspace lifetime checks, lazy dispatch,
SP regression and standalone harness are this port's engineering contributions.
Threshold ties, FP32 accumulation and the BF16 probability operand cast remain.

## Environment

Validated: Python 3.11.14, torch 2.9.0+cpu,
torch_npu 2.9.0.post1+gitdc51c2d, triton-ascend 3.2.0.dev20260322,
Ascend SDK 8.5.1, driver 25.0.rc1.1, 910B2C with 64 GB HBM per card.
Provide a compatible NPU PyTorch/Triton environment independently. Do not install
drivers or the full CUDA dependency bundle for this standalone experiment.
`requirements.txt` records the additional standalone import/test packages.

From this checkout with that environment activated:

```bash
source /usr/local/Ascend/cann-8.5.1/set_env.sh
export PYTHONPATH=$PWD
export FASTVIDEO_ATTENTION_BACKEND=VIDEO_SPARSE_ATTN
BENCH=benchmarks/vsa_npu
RUN=$PWD/vsa-results-$(date -u +%Y%m%dT%H%M%SZ)
mkdir "$RUN"
```

Inspect allocation, physical IDs and `npu-smi info -t topo` first. Use only
allocated healthy idle cards. `lease.py` takes host-local locks, checks live
health/processes immediately before imports, binds physical IDs, isolates caches
and logs, and bounds owned process groups. Locks only coordinate cooperating
processes: monitor timings and reject foreign overlap. Every output must be new.
Replace example IDs with your allocation; the frozen primary rotation currently
requires physical IDs 0,4,5,6.

## Correctness

```bash
python "$BENCH/lease.py" --devices 4 --run-dir "$RUN/sp1-lease" -- \
  python "$BENCH/validate.py" --suite contract --output "$RUN/sp1.json"
python "$BENCH/lease.py" --devices 0,1,2,3 --run-dir "$RUN/sp4-lease" -- \
  python "$BENCH/launch_group.py" --suite contract --output "$RUN/sp4" --timeout 600
python "$BENCH/lease.py" --devices 4 --run-dir "$RUN/pytest-sp1" -- \
  python -m pytest -q fastvideo/tests/attention/test_video_sparse_attn_npu.py
FASTVIDEO_TEST_NPU_VSA_SP4=1 python "$BENCH/lease.py" --devices 0,1,2,3 --run-dir "$RUN/pytest-sp4" -- \
  python -m pytest -q fastvideo/tests/attention/test_video_sparse_attn_npu_sp.py
```

Suites: `smoke`, `tiny`, `contract`, `frames`, `magnitudes`. Contracts cover
partial tiles, gates, strides, changed-input reuse, output lifetime and global
SP4 selection/restoration. SP4 pytest includes owned-rank failure cleanup.
`phase1_contract.py` replays 21 frozen adversarial cases; diagnostic 32-row
schedules do not expand production support. Tolerances stay atol=rtol=0.02,
with exact masks and valid lengths. Synthetic allclose is not model quality.

Generate independent local primary references, then check them and a CPU FP64
tiny oracle (no model weights or other machines' tensors):

```bash
python "$BENCH/lease.py" --devices 4 --run-dir "$RUN/reference-lease" --timeout 1200 -- \
  python "$BENCH/generate_references.py" --output "$RUN/references"
python "$BENCH/lease.py" --devices 4 --run-dir "$RUN/numerics-lease" --timeout 1200 -- \
  python "$BENCH/numerics.py" --reference-dir "$RUN/references" --output "$RUN/numerics.json"
```

## Paired Measurements

```bash
python "$BENCH/run.py" --mode primary --devices 0,4,5,6 --output "$RUN/primary"
python "$BENCH/lease.py" --devices 0,1,2,3 --run-dir "$RUN/sp4-timing-lease" --timeout 1800 -- \
  python "$BENCH/launch_group.py" --program timing --output "$RUN/sp4-timing" --timeout 1680
python "$BENCH/run.py" --mode replicas --devices 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 \
  --reference-dir "$RUN/references" --output "$RUN/replicas"
python "$BENCH/analyze.py" --primary "$RUN/primary" --sp4 "$RUN/sp4-timing" \
  --replicas "$RUN/replicas" --output "$RUN/paired_summary.json"
```

Run sequentially. Four disjoint SP1 workers precompile at a global barrier;
definitive SP4 latency uses one group with other groups idle. Aggregate
throughput uses four independent SP4 replicas, not SP16. Primary uses three
frozen seeds, four devices, balanced AB/BA, 30 calls per role, 12 pair units.
`analyze.py` bootstraps pairs (2,000 resamples, seed 1024), not individual calls.
Correctness, stable warmup and unchanged warmed JIT counts are required.

Boundary-synchronized wall time includes selection, allocations, casts, gates,
layout/restoration; SP4 includes collectives and identical timed input-shard
preparation. Device events, max-rank latency, coordinator intervals and throughput
are separate measurements. Peak memory includes live harness inputs, prepacked
and reference tensors plus workspace; it is not isolated scratch usage.

`timing.py --mode curve` and `launch_group.py --program timing --timing-mode
sp4curve` screen frames 17/33/81, D=64/128, sparsities 0.5/0.6/0.7/0.8/0.9.
One seed/pair per bucket gives sampled comparisons, not interpolated crossings.
Run `startup.py --reference-dir DIR --output FILE` under a new lease/cache for
cold startup, then a new process with `lease.py --cache-dir SAME_CACHE` for disk
startup. Warmed execution is reported separately.

See `RESULTS.md` for revision-specific results and negative comparisons.
No learned checkpoint, full-video quality or full-project CUDA test is claimed.
