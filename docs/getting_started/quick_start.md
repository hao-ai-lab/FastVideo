# 🚀 Quick Start

Get up and running with FastVideo in minutes!

## Installation

First, install FastVideo:

```bash
# Create and activate a new conda environment
conda create -n fastvideo python=3.12
conda activate fastvideo

# Install FastVideo
pip install fastvideo
```

## Basic Usage

### Text-to-Video Generation

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

### Image-to-Video Generation

```python
from fastvideo import VideoGenerator, SamplingParam

def main():
    # Create the generator
    model_name = "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers"
    generator = VideoGenerator.from_pretrained(model_name, num_gpus=1)

    # Set up parameters with an initial image
    sampling_param = SamplingParam.from_pretrained(model_name)
    sampling_param.image_path = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/astronaut.jpg"
    sampling_param.num_frames = 107

    # Generate video based on the image
    prompt = "A photograph coming to life with gentle movement"
    generator.generate_video(prompt, sampling_param=sampling_param,
                             output_path="my_videos/",
                             save_video=True)

if __name__ == '__main__':
    main()
```

## Next Steps

- [Installation Guide](installation.md) - Detailed installation instructions
- [Configuration](../inference/configuration.md) - Learn about configuration options
- [Examples](../inference/examples/) - Explore more examples
- [Optimizations](../inference/optimizations.md) - Performance optimization tips
- [Low VRAM Inference](../inference/low_vram_inference.md) - Memory-saving settings (CPU offload, sharded loading, etc.)
