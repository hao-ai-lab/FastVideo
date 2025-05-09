# Quick Start

This page contains instructions and code to get you quickly started with video generation using FastVideo.

## Requirements
- **OS: Linux**
- **Python: 3.10-3.12**
- **CUDA 12.4**
- **At least 1 NVIDIA GPU**

## Installation

We recommend using a environment manager such as `Conda`.

```bash
pip install fastvideo
```

Also see the [Installation Guide](installation.md).

## Generating Your First Video
Here's a minimal example to generate a video using the default settings. All of the following code snippets can be directly copied into a Python file and executed with

```bash
python example.py
```

```python
from fastvideo import VideoGenerator

def main():
    # Create a video generator with a pre-trained model
    generator = VideoGenerator.from_pretrained(
        "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        num_gpus=1,  # Adjust based on your hardware
    )

    # Define a prompt for your video
    prompt = "A curious raccoon peers through a vibrant field of yellow sunflowers, its eyes wide with interest."

    # Generate the video
    video = generator.generate_video(prompt)
if __name__ == '__main__':
    main()
```

The generated video will be saved in the current directory under `outputs/` by default.

## Customizing Generation

You can customize various parameters when generating videos:

```python
from fastvideo import VideoGenerator, SamplingParam

def main():
    # Create the generator
    generator = VideoGenerator.from_pretrained(
        "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        num_gpus=1,
    )

    # Create and customize sampling parameters
    sampling_param = SamplingParam.from_pretrained("Wan-AI/Wan2.1-T2V-1.3B-Diffusers")
    # How many frames to generate
    sampling_param.num_frames = 45
    # How many steps we denoise the video
    sampling_param.num_inference_steps = 30
    # How strongly does the video to conform to the prompt
    sampling_param.guidance_scale = 7.5

    # Optional: provide an initial image for image-to-video generation
    sampling_param.image_path = "path/to/your/image.jpg"  # Optional

    # Generate video with custom parameters
    prompt = "A beautiful sunset over a calm ocean, with gentle waves."
    video = generator.generate_video(
        prompt, 
        sampling_param=sampling_param, 
        output_path="my_videos/", # controls where videos are saved
        return_frames=True # also return frames from this call (defaults to False)
    )

    # `video` now contains frames
if __name__ == '__main__':
    main()
```

## Available Models

FastVideo supports various models for text-to-video generation:

- `Wan-AI/Wan2.1-T2V-1.3B-Diffusers` - A popular text-to-video model
- `FastVideo/FastHunyuan-Diffusers` - A high-performance model for video generation

## Advanced Configuration

You can use PipelineConfig for more advanced customization:

```python
from fastvideo import VideoGenerator, PipelineConfig

# Load the default configuration for a model
config = PipelineConfig.from_pretrained("Wan-AI/Wan2.1-T2V-1.3B-Diffusers")

# Modify configuration settings
config.vae_config.scale_factor = 16
config.vae_config.precision = "fp16"

# Create generator with custom config
generator = VideoGenerator.from_pretrained(
    "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
    num_gpus=4,
    pipeline_config=config,
)

# Generate video
prompt = "A futuristic cityscape with flying cars and neon signs."
video = generator.generate_video(prompt)
```

## Multi-GPU Setup

FastVideo automatically handles distributed setup when multiple GPUs are specified:

```python
generator = VideoGenerator.from_pretrained(
    "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
    num_gpus=4,  # Will use 4 GPUs in parallel
)
```

## Hardware Requirements

- **Basic Inference**: Minimum 20GB VRAM for quantized models (e.g., single RTX 4090)
- **LoRA Finetuning**: 40GB GPU memory each for 2 GPUs with LoRA
- **Full Finetuning/Distillation**: Multiple high-memory GPUs recommended (e.g., H100)

## Next Steps

- Explore the [API Reference](../api/index.md) for more details
- Learn about [Advanced Inference Options](../inference/overview_back.md)
- See [Examples](../examples/index.md) for more usage scenarios
