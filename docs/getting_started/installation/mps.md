# Apple Silicon FastWan

FastWan-QAD-INT8-1.3B is the Apple-native, text-to-video release candidate.
Its DiT denoising loop runs in MLX; prompt encoding and TAEHV decode use
PyTorch MPS. It does not provide image-to-video support.

## Validated configuration

The recorded release result is from an Apple M4 Max with 36 GB-class unified
memory (MLX reports 38.65 GB), macOS 14+, Python 3.12, and MLX 0.31.2:
480x832, 81 frames, three-step DMD, INT8 DiT + TAEHV decode in 123.7 seconds
end to end (117.6 seconds denoise; 5.63 GiB MLX peak).

This is the only launch-supported hardware configuration. Allocator-cap tests
are useful engineering evidence, not a claim that a physical 16 GB Mac works.

## Install from source

```console
brew install ffmpeg
git clone https://github.com/hao-ai-lab/FastVideo.git
cd FastVideo
uv venv --python 3.12 --seed
source .venv/bin/activate
uv pip install -e '.[mlx]'
```

The `mlx` extra is pinned to the MLX 0.31.2 compatibility range and only
resolves on Apple Silicon.

## Generate a video

After publication, download the release model and run the one supported
source-tree entrypoint. The script's defaults are the release configuration
(480x832, 81 frames, 3-step DMD, INT8 DiT, compiled forward, bf16 prompt
encode, TAEHV decode), so the minimal command is:

```console
huggingface-cli download FastVideo/FastWan-QAD-INT8-1.3B-Diffusers \
  --local-dir ~/models/FastWan-QAD-INT8-1.3B

python examples/inference/basic/mlx_wan_prompt_to_video.py \
  --model-root ~/models/FastWan-QAD-INT8-1.3B \
  --mlx-checkpoint ~/models/FastWan-QAD-INT8-1.3B/mlx_dit \
  --prompt "A fox runs through a misty pine forest, leaves kicking up behind it." \
  --output-path video_samples/fox.mp4
```

The release repository must include the hybrid Diffusers components and the
verified `mlx_dit/` directory. TAEHV's checkpoint is SHA-256 verified before
use; its vendored source is MIT-licensed, while FastVideo is Apache-2.0.

### Faster repeat runs

- `--mlx-checkpoint` (above) loads pre-quantized INT8 weights and skips
  requantization: warm reloads take well under a second. Without the release
  `mlx_dit/` directory you can create one locally once with
  `--save-mlx-checkpoint ~/models/fastwan_mlx_int8`.
- `--prompt-embeds-cache <path>.npy` reuses the UMT5 encode for a repeated
  prompt, skipping the text-encoder load entirely.
- The DiT forward is compiled with `mx.compile` by default (bit-identical to
  eager, ~1.4x faster denoise in the A/B). Pass `--no-mlx-compile` to force
  eager execution when debugging.

### Quality mode

TAEHV is a tiny distilled decoder: it is the right default for speed and
memory, but the full Wan VAE preserves more fine texture. On Macs with 32 GB+
of unified memory you can trade decode time for fidelity:

```console
  --decode-backend wan-vae   # bf16 decode; add --vae-decode-dtype fp16 on
                             # macOS/torch builds without bf16 MPS support
```

## Troubleshooting

- VSA is unsupported on MPS: unset `FASTVIDEO_ATTENTION_BACKEND` or set it
  to `TORCH_SDPA`.
- If your macOS/torch build rejects bf16 on MPS, pass
  `--text-encoder-dtype fp16` (and `--vae-decode-dtype fp16` with the
  `wan-vae` backend).
- Do not infer physical-16-GB support from allocator-cap experiments.
- Keep `mlx_dit/` beside the model's `transformer/`, `text_encoder/`,
  `tokenizer/`, VAE, and scheduler files.
