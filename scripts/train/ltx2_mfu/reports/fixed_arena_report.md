# LTX-2 BF16 input/fixed-arena correction (4x GB200)

Date: 2026-07-21

FastVideo source: `7f139e2b28610063d2f30526ba8f0ccae5d88944` (PR #1630)

Allocation: Slurm `1619868`, 4x NVIDIA GB200

Workload: `FastVideo/LTX2-Distilled-Diffusers`, standard mixed precision (FP32 original/master weights and optimizer state, BF16 working compute and reductions), local batch 1, 81x480x832, dense FA4, regional compile, no activation checkpointing, fused AdamW, no forward reshard.

## Measurement contract

- Source rows: 30 steps; result is the median over steps 11-30 of each step's slowest-rank `Trainer.step_time_sec`.
- Fixed-arena rows: 10 warmup + 20 measured steps; result is the median of each step's slowest-rank wall time.
- Both wall timers begin before `next(data_stream)` / `next(data_iter)`.
- Validation, checkpoint writes, W&B, model load, and compile warmup are excluded.
- Model MFU uses the existing PR #1630 formula: `14.444115 / step_seconds`.

## Exact results

| Runtime | Input | Median step | Samples/s | MFU | Input phase |
|---|---|---:|---:|---:|---:|
| Exact PR source, FSDP2 `SHARD_GRAD_OP` | original 4 rows, worker 0 | 493.569924 ms | 8.104222 | 29.264577% | included |
| Exact PR source, FSDP2 `SHARD_GRAD_OP` | existing path repeat `:32`, worker 1 | 466.474781 ms | 8.574954 | 30.964407% | included |
| Fixed-arena ZeRO-2 prototype | original 4 rows, worker 0 | 448.771899 ms | 8.913214 | 32.185872% | 42.909133 ms |
| Fixed-arena ZeRO-2 prototype | existing path repeat `:32`, worker 1 | 421.850613 ms | 9.482030 | 34.239882% | 0.266017 ms |

Paired deltas:

- Existing path-repeat + worker: source saves **27.095143 ms / 5.489626%** and raises MFU by 1.699830 points.
- Same input change in the fixed-arena runtime: saves **26.921287 ms / 5.998880%**.
- Fixed arenas against the same repeated/prefetched input: saves **44.624168 ms / 9.566255%** and raises MFU by 3.275474 points.
- Combined input + fixed arenas against the matched original source control: saves **71.719311 ms / 14.530730%** and raises MFU from 29.264577% to 34.239882%.
- The 50% target remains 288.8823 ms, so the corrected best measured runtime still needs **132.968313 ms** (a further 1.4603x speedup).

## What changed in the interpretation

The earlier fixed-arena result, 403.724612 ms / 35.777147% MFU, started its wall timer after batch loading. On the original four-row fixture, batch loading costs 42.909133 ms. That CPU stall also lets the previous step's asynchronous parameter all-gathers finish outside the timer: the fixed-arena GPU span is 404.939789 ms with the slow input but 421.526962 ms when input is prefetched. Removing 42.643116 ms of input therefore exposes 16.587173 ms of all-gather and nets 26.921287 ms, not the full input duration.

Consequences:

- Do not cite 403.724612 ms as inclusive training step time or 75.585301 ms as the independent fixed-arena gain.
- Keep 403.724612 ms only as the historical post-fetch timing result.
- Use 421.850613 ms / 34.239882% as the current inclusive fixed-arena result.
- Use 44.624168 ms as the matched independent fixed-arena saving and 71.719311 ms as the matched combined saving.

## Minimal keep/reject decision

Keep the existing data-path repeat feature; no loader cache or new data format is needed. For the full 300-step overfit config, repeat the four-row source 300 times and use one persistent worker:

```yaml
training:
  data:
    data_path:
      data/ltx2_overfit_preprocessed: 300
    dataloader_num_workers: 1
```

The tested `:32` form gives each DP rank 32 batches, enough for the 30-step timing window. `300` extends the same existing virtual repeat mechanism across the full overfit run without duplicating the 12 MB Parquet file.

Reject these alternatives:

- Worker 1 with the original one-batch/rank epoch: 503.078686 ms on rank 0, 10.933746 ms slower than its matched worker-0 rank-0 control.
- Physical one-row-group repack: 461.077923 ms on rank 0, only 2.946550 ms faster than virtual repeat while expanding the fixture to 1.58 GB.
- A new row-group cache abstraction: unnecessary while the existing virtual repeat + persistent worker hides the debug Parquet reader.

## Artifacts

- Exact source control log: `/tmp/bf16_input_control_maxrank2_7f139e2b.log`, SHA256 `5e1ffa7fbbeeff10f20814f2315e4175df79d76f2a97665ca62553af235ec8e0`
- Exact source repeat log: `/tmp/bf16_input_repeat32_maxrank_7f139e2b.log`, SHA256 `05f753036f10f6475c21ac7ffb2deb6c8374e0028ddc2f9c3413f85c63368405`
- Fixed-arena control log: `/tmp/zero2_ltx2_input_control.log`, SHA256 `b4beac801f1ef7800220bf9ac492979d5079b2ecf7fbbb5489fe0f768f3b1049`
- Fixed-arena repeat log: `/tmp/zero2_ltx2_input_repeat32.log`, SHA256 `cde13d5588a2c266a5ff248e5baa09282fd03465724d033bf0e27bdabf23589d`
- Final all-rank source harness: `/tmp/benchmark_fastvideo_train.py`, SHA256 `b0334528ad59cbf6c51758f2de1040316ac4f0ff370c526330f8e5851d7a1ef1`
- Inclusive-timer fixed-arena remote probe: `/mnt/zero2_ltx2_input_probe.py`, SHA256 `2418237ac36b1e065b4887f933d0ab1ab61310ffa5129b35cedf014cd91f44bb`
- Historical fixed-arena probe remains restored locally at `/tmp/zero2_ltx2_probe.py`, SHA256 `7ff05aafe045a53e754a88c4842fe7be33606440e2b1a8f18718f2793c24fff6`
