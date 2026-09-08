# Basic Video Generation Tutorial
The `VideoGenerator` class provides the primary Python interface for doing offline video generation, which is interacting with a diffusion pipeline without using a separate inference api server.

## Requirements
- At least a single NVIDIA GPU with CUDA 12.4.
- Python 3.10-3.12

## Installation
If you have not installed FastVideo, please following these [instructions](https://hao-ai-lab.github.io/FastVideo/getting_started/installation) first.

## Usage
The first script in this example shows the most basic usage of FastVideo. If you are new to Python and FastVideo, you should start here.

```bash
# if you have not cloned the directory:
git clone https://github.com/hao-ai-lab/FastVideo.git && cd FastVideo

python examples/inference/basic/basic.py
```

### Apple Silicon (FastMetal-QAD)

Use the MLX runtime with FastMetal-QAD. See the
[Apple Silicon guide](https://hao-ai-lab.github.io/FastVideo/getting_started/installation/mps/).

```bash
hf download FastVideo/FastMetal-1.3B-QAD --local-dir ./FastMetal-1.3B-QAD

python examples/inference/basic/mlx_wan_prompt_to_video.py \
  --model-root ./FastMetal-1.3B-QAD \
  --mlx-checkpoint ./FastMetal-1.3B-QAD \
  --prompt "A bird's-eye view of a misty forest valley at dawn."
```

5B uses
[`mlx_wan22_generate.py`](https://github.com/hao-ai-lab/FastVideo/blob/main/examples/inference/basic/mlx_wan22_generate.py)
with
`FastVideo/FastMetal-5B-QAD`.

`examples/inference/basic/basic_mps.py` is the older PyTorch MPS demo.

FastH3 Preview T2VA also runs through the native MLX runtime. Convert the DiT
to INT8, INT6, or INT4 first, then run:

```bash
python examples/inference/basic/mlx_fasth3.py \
  --model-root ./FastH3-Preview-v0.2 \
  --mlx-checkpoint ./FastH3-MLX/int6 \
  --prompt "(S1) A presenter says <d>[English] Fast H3 is amazing.</d>" \
  --height 480 --width 832 --num-frames 124 \
  --output-path ./outputs/fasth3_int6.mp4
```

Pass `--fast` for temporal RIFE fast mode and `--fast-spatial` for spatial
fast mode (reduced-canvas denoise + pixel-space upsample); the two compose.
VSA is opt-in: convert with `--include-vsa` and pass `--vsa` (see the
[Apple Silicon guide](https://hao-ai-lab.github.io/FastVideo/getting_started/installation/mps/)).
This MLX entrypoint currently supports T2VA only; FL2VA, Ref2VA, and
two-pass refinement remain follow-up work. INT6/INT8/INT4 are
weight-only; VSA attention activations stay BF16. Dense-only checkpoints keep
working for dense inference.

The complete setup and conversion commands are in the
[Apple Silicon guide](https://hao-ai-lab.github.io/FastVideo/getting_started/installation/mps/).

For an example running DMD+VSA inference:
```
python examples/inference/basic/basic_dmd.py
```

For the typed config/request path added during the inference API refactor:
```
python examples/inference/basic/basic_dmd_new_api.py
```

### FastH3 Preview

The verified [basic FastH3 example](https://github.com/hao-ai-lab/FastVideo/blob/main/examples/inference/basic/basic_fasth3.py)
runs the few-step (4-forward, DMD2-distilled) MiniMax-H3 preview, generating
synchronized video and audio with its trained block-sparse VSA attention:

```bash
UV_TORCH_BACKEND=cu130 uv pip install -e ".[fasth3]"
```

This installs the pinned FA4 CuTe package and FastVideo kernel release used by
the measured GB200 profile. Then run:

```
python examples/inference/basic/basic_fasth3.py --prompt "your prompt"
```
The default checkpoint, [FastH3 Preview v0.2](https://huggingface.co/FastVideo/FastVideo-Minimax-FastH3-Preview-v0.2), is public on the Hub under the MiniMax H3 Community License. Review its model card and license before use or redistribution.

The default `all` profile is the fastest measured four-GPU Preview recipe on GB200. It selects VSA sparsity 0.9 with 64-token tiles and the sm_100a sparse kernel, enables FA4 for eligible non-VSA paths, regionally compiles and replicates the sparse DiT, compiles and temporally parallelizes the video VAE with the `gather` strategy, and pins CPU-offloaded component memory. It also pins the benchmark protocol: five sigma-grid points (exactly four DiT forwards), one excluded seed-999 warmup, then three timed seed-1000 requests with distinct output paths.

The equivalent explicit command is:

```bash
python examples/inference/basic/basic_fasth3.py \
  --prompt "your prompt" \
  --profile all \
  --num-gpus 4 \
  --steps 5 \
  --vsa-sparsity 0.9 \
  --vsa-tile-size 64 \
  --vsa-kernel sm100a \
  --compile-vae \
  --parallel-vae \
  --replicated-dit \
  --pin-cpu-memory \
  --fa4 \
  --no-torch-compile \
  --inference-torch-compile \
  --ulysses-a2a off \
  --warmup \
  --repeats 3 \
  --seed 1000 \
  --warmup-seed 999
```

`all` enables the inference-only H3 fusions and regional compile. Both can change floating-point operation order, so this is a report-only performance profile rather than an exact-parity route. Use `--profile strict` to disable the H3 fusions while preserving regional compile, or `--profile strict --no-inference-torch-compile` for the eager strict route. Individual `--no-*` switches are available for portability and attribution; in particular, use `--vsa-kernel triton --no-fa4` if the Blackwell kernels are unavailable. `--h3-sequential-load` / `--no-h3-sequential-load` override the auto split that releases Qwen3-VL before DiT/VAE load (on by default on GB10, off on discrete GPUs). The script preserves the warmup and each measured video under distinct paths, then prints per-request wall time plus a warmup-excluded median.

One script covers each validated duration; regional compile is the fastest
measured DiT route for all three:

```bash
# 5 s
python examples/inference/basic/basic_fasth3.py \
  --prompt "your prompt" --output outputs/fasth3_5s
# 10 s
python examples/inference/basic/basic_fasth3.py \
  --prompt "your prompt" --num-frames 243 --output outputs/fasth3_10s
# 15 s
python examples/inference/basic/basic_fasth3.py \
  --prompt "your prompt" --num-frames 345 --output outputs/fasth3_15s
```

Pass `--no-inference-torch-compile` to recover the eager sparse-DiT route.

#### Hopper (H100 80 GB)

Hopper has no FA4 and no sm_100a VSA kernel, so the GB200 profile above does
not apply there. The measured Hopper route keeps the same checkpoint, four
forwards and 90% VSA sparsity and changes three things: the 64-token VSA path
runs on the ThunderKittens sm_90a kernel (`--vsa-kernel tk`, built with
`cd fastvideo-kernel && CMAKE_ARGS='-DFASTVIDEO_KERNEL_BUILD_TK=ON' ./build.sh`),
the DiT is FSDP-sharded across the GPUs instead of replicated
(`--no-replicated-dit`; a replicated 66 GB BF16 DiT does not fit 80 GB), and
the Qwen3-VL text encoder stays resident as block-scaled FP8 (35.5 GB instead
of 66.7 GB) rather than being offloaded on every request:

```bash
python scripts/checkpoint_conversion/quantize_minimax_h3_text_encoder_fp8.py \
  --source ./FastH3-Preview-v1/text_encoder --output ./FastH3-TextEncoder-FP8

python examples/inference/basic/basic_fasth3.py \
  --prompt "your prompt" \
  --num-gpus 4 --no-replicated-dit \
  --vsa-kernel tk --no-fa4 \
  --text-encoder-weights ./FastH3-TextEncoder-FP8 \
  --no-offload-text-encoder --no-offload-vae \
  --height 544 --width 960 --num-frames 345
```

`examples/inference/basic/basic_fasth3_h100.yaml` is the same profile for
`fastvideo generate --config`. Measured points on this route (345 frames,
24 FPS, stereo audio, warm, loading and compilation excluded):

| GPUs | Resolution | Playback | Generation | Source |
|---|---|---|---|---|
| 8x H100 80 GB | 1344x768 | 14.375 s | 13.506 s (13.503 / 13.506 / 13.554) | [hlander-ai/minimax-h3](https://github.com/hlander-ai/minimax-h3) on FastVideo `b2db0c0` with these patches, Triton VSA |
| 4x H100 80 GB, NVLink | 960x544 | 14.375 s | 12.98-14.02 s over 8 consecutive clips | Windflow streaming worker on FastVideo `b2db0c0` with these patches, TK VSA |

The 4x H100 row was measured through a downstream streaming worker rather than
this script; `basic_fasth3.py` numbers for that shape will follow in a later
PR. The FP8 text encoder is a precision change relative to the stock BF16
encoder; the other two changes are lossless.

### FastH3 Preview LoRAs

The LoRA release runs on top of `MiniMaxAI/MiniMax-H3` with the same default
compile, fusion, FA4, VSA, and parallel-VAE profile as the full FastH3 example:

```bash
bash examples/inference/basic/run_fasth3_lora_preview_vsa_datafree.sh \
  --prompt "your prompt"
```

The four release launchers are:

- `run_fasth3_lora_preview_vsa_datafree.sh`
- `run_fasth3_lora_preview_vsa_synthetic_step1300.sh`
- `run_fasth3_lora_preview_vsa_synthetic_step1900.sh`
- `run_fasth3_lora_preview_dense_datafree.sh`

Each downloads its exact private adapter file from
`FastVideo/FastVideo-FastH3-4-step-Preview-v1-LoRA`; authenticate with `hf auth
login` first. Pass `--lora-strength 0.5` to interpolate every adapter payload at
half strength. Strength `1` applies the published rank-64 adapter at its trained
scale and approximates the full student; `0` removes its weight deltas. VSA
launchers still use sparse attention at strength `0` and require FastVideo's
tile-64 VSA kernel; the dense launcher selects FA4. Each launcher writes to its
own variant directory by default so comparison outputs do not collide.

## Basic Walkthrough

All you need to generate videos using multi-gpus from state-of-the-art diffusion pipelines is the following few lines!

```python
from fastvideo import VideoGenerator

def main():
    generator = VideoGenerator.from_pretrained(
        "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        num_gpus=1,
    )

    prompt = ("A curious raccoon peers through a vibrant field of yellow sunflowers, its eyes "
             "wide with interest. The playful yet serene atmosphere is complemented by soft "
             "natural light filtering through the petals. Mid-shot, warm and cheerful tones.")
    video = generator.generate_video(prompt)

if __name__ == "__main__":
    main()
```
