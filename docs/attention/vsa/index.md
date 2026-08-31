# Video Sparse Attention (VSA)

Sparse attention mechanism selecting top-k blocks.

## Installation

VSA is included in the `fastvideo-kernel` package. See the [main Attention page](../index.md) for build instructions.

## Apple Silicon (MiniMax H3 / FastH3)

The native MLX runtime has an inference-only H3 VSA path that is separate from
the CUDA `fastvideo-kernel` package:

- **INT8 / INT6 / INT4** are **weight-only**. They cover linear matrices,
  including `attn.to_gate_compress` when you convert with `--include-vsa`.
  Attention Q/K/V stay BF16 (or the selected activation dtype). There is no
  INT6 Q/K/V attention kernel.
- **Dense-only checkpoints** (the default converter) drop the 50 gate
  matrices and keep fused SDPA. They remain valid for dense inference.
- **VSA-capable checkpoints** retain those gates, quantize them on the same
  affine grid, and record `vsa.capable` in `mlx_h3_dit.json`. Runtime VSA is
  still off until you pass `--vsa`.
- **Tile sizes** 64 `(4, 4, 4)` and 256 `(4, 8, 8)`. Prefix keys can be
  `exempt` or `compete`. `--vsa-dense-first-n-steps` and `--vsa-dense-layers`
  force dense SDPA on the selected steps or blocks.
- **`--vsa-impl auto`** uses the chunked gather+SDPA **reference** path.
  `--vsa-impl simd` runs the SIMD-group Metal kernel (tile 64, head dim 128)
  and falls back to reference on unsupported shapes or kernel failure. The
  runtime executes a small kernel probe before use and remembers failures
  for the process, so later blocks do not retry a broken backend.
  It is not the default: 480p four-step generation is faster than reference
  but does not yet match reference video. `--vsa-impl reference` is the same
  as `auto`.

See the [Apple Silicon guide](../../getting_started/installation/mps.md) for
conversion and `mlx_fasth3.py` flags. Do not enable VSA on a dense-only
checkpoint; reconvert with `--include-vsa` first.

H3 uses fused MLX RMSNorm by default, including dense inference. This can
change BF16 rounding relative to the older explicit normalization path.
`FASTVIDEO_MLX_FAST_NORM` controls Wan normalization only.

The generation report aggregates VSA statistics across all blocks and steps.
`impl_counts` and `fallback_reasons` show mixed execution and fallback.
`video_keep` is the mean number of selected video-key tiles per video query
tile and head, including dense overrides. `achieved_sparsity` is the matching
mean video-tile sparsity; in `compete` mode it measures actual selections,
not the requested top-k budget. These are tile counts, not token-level FLOPs.

## Usage

```python
from fastvideo_kernel import video_sparse_attn

# q, k, v: [batch_size, num_heads, seq_len, head_dim]
# variable_block_sizes: Number of valid tokens per block
# q_variable_block_sizes: Number of valid tokens per q block (can differ from KV for q/k of different lengths)
# topk: Number of blocks to attend

output = video_sparse_attn(
    q, k, v, 
    block_sizes,
    block_sizes,
    topk=32
)
```

## Citation

If you use Video Sparse Attention in your research, please cite:

```bibtex
@article{zhang2025vsa,
  title={Vsa: Faster video diffusion with trainable sparse attention},
  author={Zhang, Peiyuan and Chen, Yongqi and Huang, Haofeng and Lin, Will and Liu, Zhengzhong and Stoica, Ion and Xing, Eric and Zhang, Hao},
  journal={arXiv preprint arXiv:2505.13389},
  year={2025}
}
```
