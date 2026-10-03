# Install FastVideo with MLX

Install FastVideo on a Mac, then generate from the cookbook. Local video uses
the native MLX runtime, not CUDA, and not the old PyTorch MPS demo at
`examples/inference/basic/basic_mps.py`.

## Requirements

- macOS 14 or newer
- Python 3.12
- `ffmpeg` (`brew install ffmpeg`)

## Install

Cookbook commands run from a clone.

```bash
git clone https://github.com/hao-ai-lab/FastVideo.git && cd FastVideo
uv venv --python 3.12 --seed
source .venv/bin/activate
brew install ffmpeg
uv pip install -e ".[mlx]"
```

Conda is optional. After you activate a Conda env, still install with
`uv pip` as above.

`uv pip install "fastvideo[mlx]"` from PyPI installs the extra only. It does
not ship the example scripts the cookbook copies.

## Generate a video

Open the cookbook. Select Apple Silicon as the runtime. Each recipe has a
Python command. FastH3 also has a server path for the playground and the
OpenAI Python client.

- [Wan recipes](../../cookbook/wan.md) for FastMetal 1.3B, 5B, and 14B
- [MiniMax H3 recipes](../../cookbook/minimax-h3.md) for FastH3 V1 and FastH3 V2
- [H3 server guide](../../cookbook/openai-api.md) for the playground, cURL, and SDKs

FastH3 is two distilled MiniMax-H3 checkpoints. V1 is the four-step launch.
Some Hub repo names still say Preview. That name is historical. V1 is a full
model, not a demo. V2 is the eight-step checkpoint. More forwards is why V2
is the higher-quality FastH3.

Recorded shapes and evidence live in the
[support matrix](../../inference/support_matrix.md#apple-silicon-native-runtime).

## Pruned eight-forward checkpoint

The pruned FastH3 checkpoint has 42 transformer blocks and rank-16 AdaLN.
Its `fastvideo_inference.json` fixes eight denoising forwards, video/audio
shifts of 10/3, and VSA sparsity 0.8. Keep that file beside the transformer
when converting. The converter reads its schedule to build the AdaLN cache.

```bash
hf download FastVideo/FastH3-Pruned-8Step-BF16-ckpt300 \
  --local-dir ./FastH3-Pruned-8Step-BF16-ckpt300 \
  --exclude 'text_encoder/*'

# The pruned repo's packed NVFP4 encoder is for Blackwell. MLX reads the
# official BF16 encoder through layer 50. The last three shards are unused.
hf download MiniMaxAI/MiniMax-H3 \
  --local-dir ./FastH3-Pruned-8Step-BF16-ckpt300 \
  --include 'text_encoder/model-0000[1-9]-of-00014.safetensors' \
  --include 'text_encoder/model-0001[0-1]-of-00014.safetensors' \
  --include 'text_encoder/model.safetensors.index.json' \
  --include 'text_encoder/config.json'

python scripts/checkpoint_conversion/convert_minimax_h3_mlx.py \
  --model-root ./FastH3-Pruned-8Step-BF16-ckpt300/transformer \
  --out ./FastH3-Pruned-MLX-vsa \
  --formats "int8 int6" --include-vsa

python examples/inference/basic/mlx_fasth3.py \
  --model-root ./FastH3-Pruned-8Step-BF16-ckpt300 \
  --mlx-checkpoint ./FastH3-Pruned-MLX-vsa/int8 \
  --prompt "(S1) A potter asks <d>[English] Is the rim ready?</d>" \
  --height 480 --width 832 --num-frames 243 --steps 8 \
  --vsa --vsa-sparsity 0.8 --vsa-tile-size 64 \
  --output-path ./outputs/fasth3_pruned_int8_480p.mp4
```

At 24 fps, 124 frames is the legal H3 count for a roughly five-second clip.
Use `--num-frames 124` and a separate output path for that run. The `--fast`
and `--fast-spatial` options change the workload and are not part of the
native-resolution benchmark. A 36 GB Mac may need INT6 and phased loading;
measure memory before claiming all-resident operation.

## Hardware

- FastMetal 1.3B and 5B: 16 GB unified memory and up
- FastMetal 14B: 36 GB unified memory and up
- FastH3 V1 and V2: validated on an M4 Max with 36 GB unified memory

## Troubleshooting

- **`basic_mps.py` is the wrong path.** That script is PyTorch MPS. Use an
  Apple Silicon recipe in the cookbook.
- **Muxing fails.** Install `ffmpeg` with Homebrew.
- **A cookbook command cannot find a script.** Run it from the FastVideo
  clone after `uv pip install -e ".[mlx]"`.

If that does not match what you see, open an issue on the
[GitHub repository](https://github.com/hao-ai-lab/FastVideo) or ask in the
[Slack community](https://join.slack.com/t/fastvideo/shared_invite/zt-3f4lao1uq-u~Ipx6Lt4J27AlD2y~IdLQ).
