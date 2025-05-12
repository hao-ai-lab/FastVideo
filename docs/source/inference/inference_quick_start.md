# Inference Quick Start

This page contains step-by-step instructions to get you quickly started with video generation using FastVideo.

## Table of Contents
- [Generating Your First Video](#generating-your-first-video)
- [Customizing Generation](#customizing-generation)
- [Available Models](#available-models)
- [Image-to-Video Generation](#image-to-video-generation)
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
conda create -n fastvideo python=3.12
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

## Available Models

Please see the [support matrix](#support-matrix) for the list of supported models and their available optimizations.

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

## Optimizations

## Next Steps

- Explore the [API Reference](../api/index.md) for detailed documentation
- Learn about [Advanced Inference Options](../inference/overview_back.md)
- See [Examples](../examples/index.md) for more usage scenarios
- Check out the [Model Training](../training/overview.md) guide to fine-tune models
- Join our [Community Discord](https://discord.gg/fastvideo) for support and sharing
