# VSA2 attention kernels (sm_100a / sm_103a)

The forward and the backward of VSA2 attention for the joint [text; video] sequence of VSA-H3 at 128-token tiles, on
one B200 / GB200 or B300 GPU. FastVideo's VSA-H3 backend runs the forward for FastPAN2 sampling (its `vsa2` variant,
selected by the checkpoint's `fastvideo_inference.json`); the backward serves training. VSA2 is "Improving Video
Sparse Attention with Fine-grained Router and Sparse Rebasing" (Zhang et al., ICML 2026). `reference.py` defines the result exactly, in fp32 with
autograd.

## What it computes

- **Blocks.** N blocks of 128 rows: P = ceil(text / 128) text blocks first (the last one padded at its end), then
  N_v video tiles of 4 x 4 x 8 tokens in raster order of the tile grid, the latent padded up to whole tiles.
- **Coarse branch.** Mean-pooled Q, K and V per block, dense attention between all N blocks, broadcast back to each
  block's rows.
- **Router**, over the video alone and without gradients. Q and K are mean-pooled over 2 x 2 x 2 windows (16 per
  tile), each query window takes a softmax over every key window, and the probabilities are pooled to tile pairs.
  Per head it keeps the top N_v x topk pairs over all N_v^2, plus every diagonal pair.
- **Fine branch.** Exact attention of each query row over the valid rows of the blocks it keeps. Text queries attend
  densely, and every video tile keeps every text block (the text keys are exempt from the top-K).
- **Gated merge.** `out = fine + gate * coarse`, with the gate per row, head and channel and zero on the text rows.
- **Backward** of `sum(out * dout)`, through the fine branch, the coarse branch and the gate. The selection takes no
  gradient.

## Use

```python
import sys
sys.path.insert(0, "fastvideo-kernel/vsa2_train")
from vsa2_train import vsa2_attention

# q, k, v, gate: [B, H, S, 128] bf16, contiguous, in the block layout above, zero on padding rows; gate zero on text
# rows. t, h, w: the latent's extent in tokens; text: the text tokens; topk: ceil(0.1 * N_v) at sparsity 0.9.
out, selection = vsa2_attention(q, k, v, gate, t, h, w, text, topk)
out.backward(dout)  # q.grad, k.grad, v.grad, gate.grad
```

The first call builds the extension from `solution/` (about 20 s) into `$VSA2_TRAIN_BUILD_DIR` (default
`~/.cache/fastvideo/vsa2_train`, one directory per `LOCAL_RANK`), and later processes reuse the build. The raw entry
points are `forward`, `backward` and `run` in `solution/binding.cpp`, all destination-passing and on the current
stream. The backward reads only what the forward returned (`out`, `lse`, `selection`), so the two halves can run at
different times, as autograd runs them.

## Performance

One call (forward and backward) at 4 heads, as at sequence parallelism 8 of 32 heads. Timed by `tools/bench.py` on
one GB200 GPU, at a mean SM clock of 1.79 GHz under the power cap:

| Workload | Latent (T x H x W) | Text | Tokens | topk | This kernel | Dense SDPA fwd + bwd | VSA-128 today | Floor |
|---|---|---|---|---|---|---|---|---|
| `s8` | 49 x 22 x 39 | 512 | 42,554 | 39 | **1.94 ms** | 8.75 ms (x4.5) | 4.56 ms (x2.4) | 0.89 ms |
| `s8t1536` | 49 x 22 x 39 | 1,536 | 43,578 | 39 | **2.44 ms** | 9.01 ms (x3.7) | 4.74 ms (x1.9) | 1.18 ms |
| `s12` | 73 x 22 x 39 | 512 | 63,146 | 57 | **3.71 ms** | 18.85 ms (x5.1) | 6.17 ms (x1.7) | 1.79 ms |
| `s16` | 97 x 22 x 39 | 512 | 83,738 | 75 | **6.30 ms** | 40.06 ms (x6.4) | 8.94 ms (x1.4) | 3.00 ms |
| `s20` | 121 x 22 x 39 | 512 | 104,330 | 93 | **9.85 ms** | 61.90 ms (x6.3) | 12.21 ms (x1.2) | 4.52 ms |
| geomean | | | | | **4.05 ms** | 20.6 ms (x5.1) | 6.80 ms (x1.68) | 1.91 ms |

- **Dense SDPA:** torch's flash SDPA forward and backward over the valid tokens.
- **VSA-128 today:** VSA-H3 at 128-token tiles with the same exempt text and top-k, on FastVideo's sm_100a
  block-sparse kernels with the coarse branch in torch, with no fine-grained router. Its kernels alone take
  0.94 + 1.56 ms at `s8` and 3.30 + 6.99 ms at `s20`, so this whole step is faster than its fine-branch kernels alone.
- **Floor:** the sum over the router, the forward and the backward of each stage's slowest unit at peak (tensor cores
  at 2.06 GHz, 16 exponentials per clock per SM, 8 TB/s). The kernel is at x2.1-2.2 of it. The two fine kernels take
  83-90 % of the call: `bwd_fine` 57-63 % and `fwd_fine` 26 % (`notes/optimization.md`).

## Correctness

- **The five workloads**, against the fp32 reference. Every element must satisfy `|x - ref| <= atol + rtol |ref|`:
  `out` and `dgate` at 2e-3 and 2e-2, `dq`, `dk` and `dv` at 1e-2 and 2e-2, `lse` at 1e-3 and 1e-5. The selection
  is judged on every pair more than 1 % away from a head's threshold, where ties between correct implementations can
  fall either way.
- **Other inputs**: `tools/edge.py` judges each output against three times the error of a bf16 flash-style
  implementation of the same input (`tools/emulate.py`). The inputs in scope are edge.py's random ones, at any
  latent, text length, top-K, batch and head count, with q and k scaled by at most 16. The numeric domain is finite
  bf16 values with nonzero magnitudes in [2^-60, 2^16].
- Checked on this tree: the five workloads, plus edge.py at 9x10x21 with no text, 13x22x39 with 200 text tokens,
  3 heads, batch 2 and q, k x8, 5x8x16 with a partial text block and topk 0, a single 4x4x8 tile with one text token,
  and 21x22x39 with 512 text tokens and q, k x16.

```bash
cd fastvideo-kernel/vsa2_train
GUILD_SM_CLOCK_MHZ=2062 python tools/bench.py --solution solution --reference reference.py   # the five workloads
python tools/edge.py --solution solution --reference reference.py --thw 9,10,21 --text 0 --topk 3
GUILD_SM_CLOCK_MHZ=2062 python tools/breakdown.py --solution solution --out /tmp/vsa2-breakdown   # by kernel
cd .. && python -m pytest tests/test_vsa2_train.py    # the autograd wrapper
```

## Limits

- sm_100a and sm_103a only (B200 / GB200, B300): `tcgen05` with tensor memory, TMA, clusters and cluster launch
  control. The loader builds for the GPU's own architecture; in a build for any other the kernels trap.
- Head dimension 128, tiles of 4 x 4 x 8 and router windows of 2 x 2 x 2.
- One stream, the current one.
- Wired into FastVideo through `fastvideo/attention/backends/video_sparse_attn_h3.py` (`variant="vsa2"`), which
  loads this directory (`FASTVIDEO_VSA2_TRAIN_DIR`, by default this checkout's) and raises on other GPUs. The
  directory is not part of the `fastvideo-kernel` wheel: like `wan22_vae_decode/`, it builds from source on first use.
