# Sliding Tile Attention (STA) Branch

This branch is a stash/testing branch for people who want to try
Sliding Tile Attention (STA). The top-level README is intentionally
STA-only.

## What is STA

Sliding Tile Attention is an optimized attention backend for
window-based video generation.

- Blog: https://hao-ai-lab.github.io/blogs/sta/
- Paper: https://arxiv.org/abs/2502.04507
- In-repo STA docs: `docs/attention/sta/index.md`

## Setup

Install FastVideo from source:

```bash
uv pip install -e .
```

Build the STA kernel package:

```bash
cd fastvideo-kernel
./build.sh
cd ..
```

## Run STA Inference

STA backend:

```bash
export FASTVIDEO_ATTENTION_BACKEND=SLIDING_TILE_ATTN
```

Ready-to-run examples:

- HunyuanVideo: `scripts/inference/v1_inference_hunyuan_STA.sh`
- Wan2.1-T2V-14B: `scripts/inference/v1_inference_wan_STA.sh`

Run:

```bash
bash scripts/inference/v1_inference_hunyuan_STA.sh
# or
bash scripts/inference/v1_inference_wan_STA.sh
```

Both scripts already set STA-related env vars:

- `FASTVIDEO_ATTENTION_BACKEND=SLIDING_TILE_ATTN`
- `FASTVIDEO_ATTENTION_CONFIG` to an STA mask strategy JSON

## STA Mask Strategy Files

- HunyuanVideo config: `assets/mask_strategy_hunyuan.json`
- Wan config: `assets/mask_strategy_wan.json`

## STA Mask Search (Wan2.1-T2V-14B)

Run mask search + tuning from repo root:

```bash
bash examples/inference/sta_mask_search/inference_wan_sta.sh
```

What this script does:

- Runs `STA_searching` first.
- Runs `STA_tuning` next (`skip_time_steps=12` by default).
- Uses prompt shards from `assets/prompt_0.txt` to `assets/prompt_7.txt`.

Important notes:

- Default script is set to 8 GPUs (`num_gpu=8`). If needed, edit
  `examples/inference/sta_mask_search/inference_wan_sta.sh`.
- STA searching/tuning currently supports `69x768x1280` (Wan setting).

Generated files:

- Search results: `output/mask_search_result_pos_1280x768/`
- Tuned strategy: `output/mask_search_strategy_1280x768/mask_strategy_s12.json`

Use the tuned mask for STA inference:

```bash
export FASTVIDEO_ATTENTION_BACKEND=SLIDING_TILE_ATTN
export FASTVIDEO_ATTENTION_CONFIG=output/mask_search_strategy_1280x768/mask_strategy_s12.json
python examples/inference/sta_mask_search/wan_example.py --STA_mode STA_inference --num_gpus 1
```

## Citation

```bibtex
@article{zhang2025fast,
  title={Fast video generation with sliding tile attention},
  author={Zhang, Peiyuan and Chen, Yongqi and Su, Runlong and Ding, Hangliang and Stoica, Ion and Liu, Zhengzhong and Zhang, Hao},
  journal={arXiv preprint arXiv:2502.04507},
  year={2025}
}
```
