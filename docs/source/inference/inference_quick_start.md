# Inference Quick Start

This page contains step-by-step instructions to get you quickly started with video generation using FastVideo.

## Table of Contents
- [Software Requirements](#software-requirements)
- [Installation](#installation)
- [Generating Your First Video](#generating-your-first-video)
- [Customizing Generation](#customizing-generation)
- [Available Models](#available-models)
- [Multi-GPU Setup](#multi-gpu-setup)
- [Image-to-Video Generation](#image-to-video-generation)
- [Output Formats and Options](#output-formats-and-options)
- [Performance Optimization](#performance-optimization)
- [Troubleshooting](#troubleshooting)
- [Advanced Configuration](#advanced-configuration)
- [Next Steps](#next-steps)

## Software Requirements
- **OS**: Linux (Tested on Ubuntu 22.04+)
- **Python**: 3.10-3.12
- **CUDA**: 12.4

## Installation

We recommend using an environment manager such as `Conda` to create a clean environment:

```bash
# Create and activate a new conda environment
conda create -n fastvideo python=3.10
conda activate fastvideo

# Install FastVideo
pip install fastvideo
```

For advanced installation options, see the [Installation Guide](installation.md).

## Generating Your First Video
Here's a minimal example to generate a video using the default settings. Create a file called `example.py` with the following code:

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
    video = generator.generate_video(
        prompt,
        return_frames=True,  # Also return frames from this call (defaults to False)
        output_path="my_videos/",  # Controls where videos are saved
        save_video=True
    )

if __name__ == '__main__':
    main()
```

Run the script with:

```bash
python example.py
```

The generated video will be saved in the current directory under `my_videos/`.

## Customizing Generation

You can customize various parameters when generating videos using the `SamplingParam` class:

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
    
    # Video resolution (width, height)
    sampling_param.width = 1024
    sampling_param.height = 576
    
    # How many steps we denoise the video (higher = better quality, slower generation)
    sampling_param.num_inference_steps = 30
    
    # How strongly the video conforms to the prompt (higher = more faithful to prompt)
    sampling_param.guidance_scale = 7.5
    
    # Random seed for reproducibility
    sampling_param.seed = 42  # Optional, leave unset for random results

    # Generate video with custom parameters
    prompt = "A beautiful sunset over a calm ocean, with gentle waves."
    video = generator.generate_video(
        prompt, 
        sampling_param=sampling_param, 
        output_path="my_videos/",  # Controls where videos are saved
        return_frames=True,  # Also return frames from this call (defaults to False)
        save_video=True
    )

    # If return_frames=True, video contains the generated frames as a NumPy array
    print(f"Generated {len(video)} frames")

if __name__ == '__main__':
    main()
```

## Available Models

Please see the [support matrix](#support-matrix) for the list of supported models and their available optimizations.

## Multi-GPU Setup

FastVideo automatically distributes the generation process when multiple GPUs are specified:

```python
# Will use 4 GPUs in parallel for faster generation
generator = VideoGenerator.from_pretrained(
    "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
    num_gpus=4,
)
```

## Image-to-Video Generation

You can generate a video starting from an initial image:

```python
from fastvideo import VideoGenerator, SamplingParam

# Create the generator
generator = VideoGenerator.from_pretrained("Wan-AI/Wan2.1-T2V-1.3B-Diffusers")

# Set up parameters with an initial image
sampling_param = SamplingParam.from_pretrained("Wan-AI/Wan2.1-T2V-1.3B-Diffusers")
sampling_param.image_path = "path/to/your/image.jpg"
sampling_param.num_frames = 24
sampling_param.image_strength = 0.8  # How much to preserve the original image (0-1)

# Generate video based on the image
prompt = "A photograph coming to life with gentle movement"
video = generator.generate_video(prompt, sampling_param=sampling_param)
```

## Output Formats and Options

FastVideo provides several options for saving and manipulating the generated videos:

```python
from fastvideo import VideoGenerator, SamplingParam, VideoFormat

# Create the generator
generator = VideoGenerator.from_pretrained("Wan-AI/Wan2.1-T2V-1.3B-Diffusers")

# Generate with custom output options
video = generator.generate_video(
    "Drone footage of a tropical rainforest",
    output_path="videos/rainforest/",  # Custom save location
    filename="rainforest_flyover",  # Custom filename (no extension needed)
    format=VideoFormat.MP4,  # Output format: MP4, GIF, or WEBM
    fps=30,  # Frames per second for the output video
    loop=False,  # Whether to loop the video (for GIF)
    save_frames=True,  # Also save individual frames as images
)
```

## Performance Optimization

Optimize FastVideo for your hardware:

```python
from fastvideo import VideoGenerator, PipelineConfig

# Create an optimized configuration
config = PipelineConfig.from_pretrained("Wan-AI/Wan2.1-T2V-1.3B-Diffusers")

# Memory optimization
config.enable_vae_slicing = True  # Process VAE in slices to reduce memory
config.enable_model_cpu_offload = True  # Offload models to CPU when not in use

# Speed optimization
config.vae_config.precision = "fp16"  # Use half precision for VAE
config.enable_xformers = True  # Use xformers for attention computation

# Create generator with optimized config
generator = VideoGenerator.from_pretrained(
    "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
    pipeline_config=config,
)
```

## Troubleshooting

Common issues and their solutions:

### Out of Memory Errors
If you encounter CUDA out of memory errors:
- Reduce `num_frames` or video resolution
- Enable memory optimization with `enable_model_cpu_offload`
- Try a smaller model or use quantized versions
- Use `num_gpus` > 1 if multiple GPUs are available

### Slow Generation
To speed up generation:
- Reduce `num_inference_steps` (20-30 is usually sufficient)
- Use half precision (`fp16`) for the VAE
- Use multiple GPUs if available

### Unexpected Results
If the generated video doesn't match your prompt:
- Try increasing `guidance_scale` (7.0-9.0 works well)
- Make your prompt more detailed and specific
- Experiment with different random seeds
- Try a different model

## Advanced Configuration

For advanced customization, use the `PipelineConfig` class:

```python
from fastvideo import VideoGenerator, PipelineConfig

# Load the default configuration for a model
config = PipelineConfig.from_pretrained("Wan-AI/Wan2.1-T2V-1.3B-Diffusers")

# Modify configuration settings
config.vae_config.scale_factor = 16
config.vae_config.precision = "fp16"
config.scheduler_config.beta_schedule = "linear"
config.attention_config.attention_type = "xformers"

# Create generator with custom config
generator = VideoGenerator.from_pretrained(
    "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
    pipeline_config=config,
)

# Generate video
prompt = "A futuristic cityscape with flying cars and neon signs."
video = generator.generate_video(prompt)
```

## Next Steps

- Explore the [API Reference](../api/index.md) for detailed documentation
- Learn about [Advanced Inference Options](../inference/overview_back.md)
- See [Examples](../examples/index.md) for more usage scenarios
- Check out the [Model Training](../training/overview.md) guide to fine-tune models
- Join our [Community Discord](https://discord.gg/fastvideo) for support and sharing
