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

## FastH3 V2 and Trim with INT6

The released sources are `FastVideo/FastVideo-FastH3-8-Step-V2` and
`FastVideo/FastVideo-FastH3-Trim-8-Step`. Trim has 42 transformer blocks and
rank-16 AdaLN. Both use eight denoising forwards, video/audio shifts of 10/3,
VSA sparsity 0.8, a native NVFP4 text encoder, and the 26-layer light video VAE.
Keep `fastvideo_inference.json` beside `transformer/`; conversion reads its
schedule to build the AdaLN cache.

Convert the BF16 transformer to affine INT6 with its VSA gates:

```bash
hf download FastVideo/FastVideo-FastH3-Trim-8-Step \
  --local-dir ./FastH3-Trim

python scripts/checkpoint_conversion/convert_minimax_h3_mlx.py \
  --model-root ./FastH3-Trim/transformer \
  --out ./FastH3-Trim-MLX \
  --formats "int6" --include-vsa
```

For V2, use `FastVideo/FastVideo-FastH3-8-Step-V2` and separate source/output
directories. Preconverted release snapshots use the same names with the
`-MLX-INT6` suffix. Each snapshot includes the encoder in MLX layout, both VAEs,
and the trained schedule, so it does not require a second encoder download.

The 36 GiB M4 Max release recipe uses phased placement. It loads the encoder,
DiT, and decoders in turn. Use reference attention and native output geometry:

```python
from pathlib import Path
from fastvideo.mlx_runtime.minimax_h3_pipeline import MiniMaxH3MLXPipeline

root = Path("./FastH3-Trim")
pipeline = MiniMaxH3MLXPipeline(
    model_root=root,
    mlx_dit_checkpoint="./FastH3-Trim-MLX/int6",
    conditioner_mode="nvfp4",
    resident=False,
    vae_dtype="fp16",
    metal_wired_limit_gib=27,
)
try:
    pipeline.generate(
        "A corgi news anchor sits behind a desk and gives a cheerful bark.",
        output_path="./outputs/trim-int6-corgi.mp4",
        width=832, height=480, num_frames=124, seed=1234,
        num_steps=8, vsa=True, vsa_sparsity=0.8, vsa_tile_size=64,
        vsa_impl="reference", vae_tile_height=256, vae_tile_width=256,
    )
finally:
    pipeline.close()
```

Set `FASTVIDEO_MLX_DQ_GEMM=1` before running this Python command. It selects the
validated affine dequantization followed by dense matrix multiplication.
124 frames at 24 fps is roughly five seconds. The recipe preserves all frames
and the requested resolution.

### Native NVFP4 encoder and cache

The MLX conditioner reads the released packed NVFP4 weights without
requantization. It retains the layers H3 reads and can cache the packed weights
in MLX layout. The cache is written in a staging directory and published by a
single rename. A cache hit changes storage layout, not encoder arithmetic.

MLX uses BF16 embeddings and FP32 activations; CUDA uses quantized activations.
Generated video and audio must be reviewed before claiming cross-runtime
quality parity. MLX 0.32.2 supports the required operator on Apple Silicon.

### Metal wired memory

MLX's allocation limit and wired-memory limit are separate. The optional
`metal_wired_limit_gib` calls `mx.set_wired_limit` to keep selected Metal
allocations in physical memory. It does not add RAM. An explicit request fails
if the installed MLX build cannot apply it. `close()` restores the previous
wired limit. The phased pipeline also sets a 30 GiB maximum allocator guideline
and restores its previous value on close. Resident placement keeps the existing
allocator limit so larger Macs can hold all components.

The tested 36 GiB M4 Max recipe uses 27 GiB and phased placement. Resident
placement also requires capacity for all components and peak activations;
wiring cannot make an oversized stack fit. Inspect `mx.device_info()` before
choosing a limit on another Mac.

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
