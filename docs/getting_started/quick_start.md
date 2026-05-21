# 🚀 Quick Start

Already installed FastVideo? Generate your first video below. If not, see the [Installation Guide](installation.md) first.

## ⚡ Generate Your Command

Select your task and hardware to get a ready-to-run command:

<div class="quick-start-guide-wrap">
  <iframe
    class="quick-start-guide-frame"
    src="/config-generator/"
    title="FastVideo Guided Config Generator"
  ></iframe>
</div>

<script>
window.addEventListener("message", (e) => {
  if (e.data && e.data.type === "quick-start-guide-height") {
    const frame = document.querySelector(".quick-start-guide-frame");
    if (frame) frame.style.height = e.data.height + "px";
  }
});
</script>

!!! tip "Need more control?"
    Use the [Advanced Tuning Guide](advanced_tuning_guide.md) to tune all parameters — resolution, attention backend, memory offloading, and more.

## Python API

Prefer Python over the CLI? Here are equivalent examples:

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

- [Advanced Tuning Guide](advanced_tuning_guide.md) - Fine-grained parameter tuning
- [Installation Guide](installation.md) - Detailed installation instructions
- [Configuration](../inference/configuration.md) - Learn about configuration options
- [Examples](../inference/examples/examples_inference_index.md) - Explore more
  examples
- [Optimizations](../inference/optimizations.md) - Performance optimization tips
