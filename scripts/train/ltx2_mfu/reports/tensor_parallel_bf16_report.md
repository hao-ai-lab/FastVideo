# LTX-2 BF16 DP4 vs Megatron TP gate on 4x GB200

Verdict: **reject TP2xDP2/B2 and TP4/B4 as routes to 50% MFU**. Tensor
parallelism keeps the same 300.096 TF of packed-linear work per rank because
local batch grows with TP. Measured compute is therefore unchanged, while the
required hidden-state all-reduces add 57.3 ms (TP2) or 118.9 ms (TP4).

## Reproduction

```bash
FV_JOBID=1619868 .agents/skills/run-fastvideo-dlcluster/dlrun --profile gb200 \
  'unset TORCH_LOGS; OMP_NUM_THREADS=1 torchrun --standalone --nproc-per-node=4 \
   /mnt/bench_ltx2_tp_bf16.py | tee /mnt/bench_ltx2_tp_bf16.jsonl'
```

- GPU: NVIDIA GB200, SM100; 4 ranks on one tray
- PyTorch `2.12.0+cu130`, CUDA 13.0, NCCL 2.29.7
- BF16 `torch.mm`, TF32 disabled
- CUDA events; 3 warmups; median of 7 samples; 3 launches/sample
- Reported case latency is the maximum rank median, so both concurrent TP2
  groups are represented.
- TP sanity checks compare row-parallel fprop and column-parallel dgrad against
  full BF16 GEMMs. Max absolute difference was 0.03125 (BF16 reduction order).

## Exact Megatron mapping

Each of 48 blocks has six video-hidden-state and one text-context all-reduce.

| pack | split | collective | TP2 local GEMM MxKxN | TP4 local GEMM MxKxN |
|---|---|---|---:|---:|
| self QKV | column N | dX video | 8580x4096x6144 | 17160x4096x3072 |
| self out | row K | fprop video | 8580x2048x4096 | 17160x1024x4096 |
| cross Q | column N | dX video | 8580x4096x2048 | 17160x4096x1024 |
| cross KV | column N | dX text | 2048x4096x4096 | 4096x4096x2048 |
| cross out | row K | fprop video | 8580x2048x4096 | 17160x1024x4096 |
| FFN up | column N | dX video | 8580x4096x8192 | 17160x4096x4096 |
| FFN down | row K | fprop video | 8580x8192x4096 | 17160x4096x4096 |

All fprop, dgrad, and wgrad GEMMs are counted. Attention heads are sharded; no
full QKV or FFN-up activation is replicated.

## Measured gate

| topology | packed compute | effective BF16 | all-reduce only | sequential compute+AR | AR payload | ring-wire floor |
|---|---:|---:|---:|---:|---:|---:|
| DP4 / B1 | 204.120 ms | 1470.2 TF/s | - | 204.120 ms | - | - |
| TP2 x DP2 / B2 | 205.043 ms | 1463.6 TF/s | 57.269 ms | 262.855 ms | 19.603 GiB | 19.603 GiB/rank |
| TP4 / B4 | 200.536 ms | 1496.5 TF/s | 118.852 ms | 319.518 ms | 39.205 GiB | 58.808 GiB/rank |

TP2 is 0.45% slower than DP4 before communication. TP4 saves only 3.585 ms
(1.79%) before communication. Logical all-reduce payload throughput was 342.3
GiB/s for TP2 and 329.9 GiB/s for TP4.

The fixed-arena step is 403.725 ms and the 50% target is 288.882 ms, requiring
114.843 ms saved. With the DP4 projection baseline removed, the replacement
projection segment must be at most 89.277 ms.

| topology | projected step with free collectives | projected step with measured sequential segment | result |
|---|---:|---:|---|
| TP2 x DP2 / B2 | 404.648 ms | 462.460 ms | fail |
| TP4 / B4 | 400.140 ms | 519.123 ms | fail |

Even granting perfect communication overlap, TP2 leaves only 83.839 ms and TP4
88.346 ms for the current 199.605 ms non-projection portion: an unsupported
2.38x/2.26x speedup. TP does not reduce its per-rank attention arithmetic here
(`B x sharded_heads` stays constant), and replicated residual/elementwise work
grows with B. The measured sequential TP4 projection segment alone exceeds the
entire 288.882 ms target.

## Memory bound

The current fixed-arena run measured 149.792064 GiB peak and 97.234253 GiB
steady per rank on a 185.030 GiB GB200. Assuming BF16 parameters+grads shard by
TP and FP32 master/moments remain partitioned over the four total ranks:

| topology | training-state floor | peak if non-state memory is unchanged | conservative peak if all non-state peak scales with B | scratch peak |
|---|---:|---:|---:|---:|
| DP4 / B1 | 85.022 GiB | 149.792 GiB | 149.792 GiB | 0.609 GiB |
| TP2 x DP2 / B2 | 60.730 GiB | 125.500 GiB | 190.270 GiB | 0.617 GiB |
| TP4 / B4 | 48.584 GiB | 113.354 GiB | 307.664 GiB | 0.748 GiB |

The scratch benchmark is safe. A full TP run is not proven memory-safe because
the true activation split lies between those peak bounds; a full-model smoke
would be required only if timing passed, which it does not.

## Artifacts

- `/mnt/bench_ltx2_tp_bf16.py`:
  `2e2d51d354b309fd8caffae27ed8824f3e5fd698e8542d05f8698f997c43b8de`
- `/mnt/bench_ltx2_tp_bf16.jsonl`:
  `488cb5aa394df060b53a89ffb223dcda6b8d705cfb6d162206e9dc72099ca821`
