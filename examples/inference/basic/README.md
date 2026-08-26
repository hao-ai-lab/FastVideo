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

`all` enables the inference-only H3 fusions and regional compile. Both can change floating-point operation order, so this is a report-only performance profile rather than an exact-parity route. Use `--profile strict` to disable the H3 fusions while preserving regional compile, or `--profile strict --no-inference-torch-compile` for the eager strict route. Individual `--no-*` switches are available for portability and attribution; in particular, use `--vsa-kernel triton --no-fa4` if the Blackwell kernels are unavailable. The script preserves the warmup and each measured video under distinct paths, then prints per-request wall time plus a warmup-excluded median.

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
