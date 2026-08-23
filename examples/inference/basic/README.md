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

For an example on Apple silicon: 
```
python examples/inference/basic/basic_mps.py
```

For an example running DMD+VSA inference:
```
python examples/inference/basic/basic_dmd.py
```

For the typed config/request path added during the inference API refactor:
```
python examples/inference/basic/basic_dmd_new_api.py
```

For the few-step (4-forward, DMD2-distilled) MiniMax-H3 preview, generating synchronized video and audio with its trained block-sparse VSA attention:

```bash
UV_TORCH_BACKEND=cu130 uv pip install -e ".[fasth3]"
```

This installs the pinned FA4 CuTe package and FastVideo kernel release used by
the measured GB200 profile. Then run:

```
python examples/inference/basic/basic_fasth3.py --prompt "your prompt"
```
The default checkpoint `FastVideo/FastVideo-Minimax-FastH3-Preview-v0.1` is private on the Hub while its license review completes; until it flips public, pass `--model-path` with a local snapshot of the release.

The default `all` profile is the fastest measured four-GPU Preview recipe on GB200. It selects VSA sparsity 0.9 with 64-token tiles and the sm_100a sparse kernel, enables FA4 for eligible non-VSA paths, keeps the sparse DiT eager and replicated, compiles and temporally parallelizes the video VAE with the `gather` strategy, and pins CPU-offloaded component memory. It also pins the benchmark protocol: five sigma-grid points (exactly four DiT forwards), one excluded seed-999 warmup, then three timed seed-1000 requests with distinct output paths.

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
  --no-inference-torch-compile \
  --ulysses-a2a off \
  --warmup \
  --repeats 3 \
  --seed 1000 \
  --warmup-seed 999
```

`all` enables the inference-only H3 fusions. They change floating-point operation order, so this is a report-only performance profile rather than an exact-parity route. Use `--profile strict` to disable only those fusions while preserving every other setting. Individual `--no-*` switches are available for portability and attribution; in particular, use `--vsa-kernel triton --no-fa4` if the Blackwell kernels are unavailable. The script preserves the warmup and each measured video under distinct paths, then prints per-request wall time plus a warmup-excluded median.

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
