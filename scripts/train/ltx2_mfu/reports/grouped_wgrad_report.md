# LTX-2 BF16 grouped-weight-gradient GB200 gate

Result: reject. Grouping weight-gradient GEMMs across adjacent blocks cannot
provide the requested 10 ms/step win. Even the prepacked arithmetic ceiling is
only 2.512 ms/step for groups of 2 and 2.073 ms/step for groups of 4. Including
the copies needed by stock `torch.bmm` changes those to regressions.

## Scope and method

- GPU: one NVIDIA GB200, CUDA 13.0, PyTorch 2.12.0+cu130, job 1623561.
- Exact PR #1630 packed video-path roles across 48 blocks:
  - self QKV: `4290 x 4096 -> 12288` (one role/block)
  - self output, cross Q, cross output: `4290 x 4096 -> 4096` (three roles/block)
  - text KV: `1024 x 4096 -> 8192` (one role/block)
  - FF up: `4290 x 4096 -> 16384` (one role/block)
  - FF down: `4290 x 16384 -> 4096` (one role/block)
- Serial baseline is the autograd weight-gradient formula, one GEMM per
  parameter: `dW_i = dY_i.T @ X_i`.
- Grouped candidate is the equivalent strided batch GEMM:
  `bmm(stack(dY).transpose(1, 2), stack(X))` for 2 or 4 adjacent blocks.
- `prepacked` assumes the stacked inputs already exist and is therefore an
  optimistic arithmetic ceiling. `staged` also copies separate activations and
  output gradients into contiguous batch buffers. `staged_scatter` additionally
  copies each result to a separate parameter-gradient buffer.
- Forward, full-checkpoint recompute, and dgrad exact-shape GEMMs are measured
  for context but are identical between paths. Only wgrad can be deferred.
- Working tensors and returned gradients are BF16. The candidate does not touch
  the optimizer contract: resident master weights and moments remain FP32.

For each case and mode, the projection is:

`median(((serial_A + serial_B) / 2) - candidate) * (48 / group) * roles_per_block`

The aggregate sums all seven roles. This excludes the likely loss of FSDP
reduce-scatter overlap, which makes the prepacked result more optimistic.

## Results

| Group | Prepacked ceiling | Staged | Staged + scatter | >=10 ms |
|---:|---:|---:|---:|:---:|
| 2 | +2.512 ms | -11.199 ms | -19.859 ms | no |
| 4 | +2.073 ms | -11.436 ms | -20.566 ms | no |

Every exact-shape grouped result was bit-identical to its serial result
(`max_abs=0`, `relative_l2=0`), passing the declared BF16 tolerance
`rtol=0.016`, `atol=0.01`. Serial A/B control drift was at most 2.505%; the
paired per-sample projection avoids attributing that drift to the candidate.

## Reproduce

Local scratch files:

- `/private/tmp/bench_ltx2_bf16_grouped_wgrad.py`
- `/private/tmp/run_ltx2_bf16_grouped_wgrad.sh`
- `/private/tmp/bench_ltx2_bf16_grouped_wgrad_job1623561.jsonl`

Single-GPU command inside a GB200 container:

```bash
CUDA_VISIBLE_DEVICES=0 /mnt/FastVideo/.venv/bin/python \
  /mnt/bench_ltx2_bf16_grouped_wgrad.py \
  --groups 2 4 --warmup 5 --samples 15 --inner 3
```

The runner exits on `NODE_RANK != 0`, so the recorded two-tray launch was:

```bash
FV_JOBID=1623561 .agents/skills/run-fastvideo-dlcluster/dlrun \
  --profile gb200x8 'bash /mnt/run_ltx2_bf16_grouped_wgrad.sh'
```

SHA-256:

```text
1eea3f150c18b8eddc9019edd957cd1e522af0d41c8678a537d1959a625289a2  bench_ltx2_bf16_grouped_wgrad.py
874c17cdfd805ca1f2e1c2e3f9df80c00aa433a215269135fb4d7814d0abeed7  run_ltx2_bf16_grouped_wgrad.sh
731a0f72b031d8768b199a6cd9990192d7e7dd5ae07a5238b8be5e3da21ea631  bench_ltx2_bf16_grouped_wgrad_job1623561.jsonl
```

No FastVideo source was edited. Do not integrate deferred BF16 wgrad from this
gate; it is below threshold before accounting for autograd hooks, buffer
lifetime, FSDP communication, or optimizer traversal.
