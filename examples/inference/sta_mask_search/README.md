# STA Mask Search (Archived Workflow)

The full STA integration is preserved in:

- https://github.com/hao-ai-lab/FastVideo/tree/sta_do_not_delete

Switch to the branch before running mask search:

```bash
git fetch origin
git checkout sta_do_not_delete
```

Run mask search + tuning:

```bash
export FASTVIDEO_ATTENTION_BACKEND=SLIDING_TILE_ATTN
export FASTVIDEO_ATTENTION_CONFIG=assets/mask_strategy_wan.json

bash examples/inference/sta_mask_search/inference_wan_sta.sh
```

This script runs:

- `STA_searching` and writes to `inference_results/sta/mask_search_full`
- `STA_tuning` and writes to `inference_results/sta/mask_search_sparse`

Example STA inference (same archived branch):

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
  --prompt "A cinematic wildlife shot of a lion walking in golden grasslands."
```
