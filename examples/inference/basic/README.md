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

For the few-step (4-step, DMD2-distilled) MiniMax-H3 preview, generating synchronized video and audio, optionally with block-sparse VSA attention:
```
python examples/inference/basic/basic_fasth3.py --prompt "your prompt" [--vsa-sparsity 0.9]
```
The default checkpoint `FastVideo/FastVideo-Minimax-FastH3-Preview-v0.1` is private on the Hub while its license review completes; until it flips public, pass `--model-path` with a local snapshot of the release.

On Blackwell (sm_100) GPUs with a `fastvideo-kernel` build that carries the sm_100a block-sparse extension, `--vsa-kernel sm100a` routes the tile-64 attention forwards through the CUDA kernel instead of Triton (it sets `FASTVIDEO_VSA_SM100A=1` before the pipeline boots); if the extension or the arch is missing, the run warns once and falls back to Triton.

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
