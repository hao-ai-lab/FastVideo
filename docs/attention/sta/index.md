# Sliding Tile Attention (STA)

STA inference integration is archived from `main`.

The full STA pipeline code (including mask search and STA inference wiring in
`fastvideo/`) is preserved in:

- https://github.com/hao-ai-lab/FastVideo/tree/sta_do_not_delete

In this branch, STA kernels in `fastvideo-kernel` are still kept.

## Why STA is not in `main`

We do not keep STA pipeline integration in `main` because we believe Video
Sparse Attention (VSA) is strictly better than STA for the actively maintained
FastVideo inference path.

## What to checkout for STA workflows

To run the full STA workflow, switch to the archived branch:

```bash
git fetch origin
git checkout sta_do_not_delete
```

## Mask Search (archive branch)

The reference script is:

- `examples/inference/sta_mask_search/inference_wan_sta.sh`

It runs two stages:

1. `STA_searching` (full search), output at
   `inference_results/sta/mask_search_full`
2. `STA_tuning` (sparse tuning), output at
   `inference_results/sta/mask_search_sparse`

Run:

```bash
export FASTVIDEO_ATTENTION_BACKEND=SLIDING_TILE_ATTN
export FASTVIDEO_ATTENTION_CONFIG=assets/mask_strategy_wan.json

bash examples/inference/sta_mask_search/inference_wan_sta.sh
```

## STA Inference (archive branch)

With a selected mask strategy, run inference with:

```bash
export FASTVIDEO_ATTENTION_BACKEND=SLIDING_TILE_ATTN
export FASTVIDEO_ATTENTION_CONFIG=assets/mask_strategy_wan.json

fastvideo generate \
  --model-path Wan-AI/Wan2.1-T2V-14B-Diffusers \
  --num-gpus 2 \
  --tp-size 2 \
  --sp-size 2 \
  --height 768 \
  --width 1280 \
  --num-frames 69 \
  --num-inference-steps 50 \
  --prompt "A cinematic wildlife shot of a lion walking in golden grasslands." \
  --output-path outputs_video/STA/
```

Python usage on the archive branch can also set `STA_mode` in
`VideoGenerator.from_pretrained(...)`:

- `STA_searching`
- `STA_tuning`
- `STA_inference`

## Kernel-level API (current branch)

STA kernels remain available from `fastvideo-kernel`. See
[Attention overview](../index.md) for build instructions.

## Citation

If you use Sliding Tile Attention in your research, please cite:

```bibtex
@article{zhang2025fast,
  title={Fast video generation with sliding tile attention},
  author={Zhang, Peiyuan and Chen, Yongqi and Su, Runlong and Ding, Hangliang and Stoica, Ion and Liu, Zhengzhong and Zhang, Hao},
  journal={arXiv preprint arXiv:2502.04507},
  year={2025}
}
```
