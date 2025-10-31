# 🚀 Quick Start

Get up and running with FastVideo in minutes!

## Installation

First, install FastVideo:

```bash
pip install fastvideo
```

## Basic Usage

### Text-to-Video Generation

```python
from fastvideo import FastVideoPipeline

# Initialize the pipeline
pipe = FastVideoPipeline.from_pretrained("wan2.1-t2v-1.3B")

# Generate a video
prompt = "A cat playing with a ball of yarn"
video = pipe(prompt, num_frames=16, height=512, width=512)

# Save the video
video.save("output.mp4")
```

### Image-to-Video Generation

```python
from fastvideo import FastVideoPipeline
from PIL import Image

# Load an image
image = Image.open("input.jpg")

# Initialize the pipeline
pipe = FastVideoPipeline.from_pretrained("wan2.1-i2v-14B-480p")

# Generate a video from the image
video = pipe(image, num_frames=16, height=480, width=480)

# Save the video
video.save("output.mp4")
```

## Next Steps

- [Installation Guide](installation.md) - Detailed installation instructions
- [Configuration](../inference/configuration.md) - Learn about configuration options
- [Examples](../inference/examples/) - Explore more examples
- [Optimizations](../inference/optimizations.md) - Performance optimization tips
