# Basic Video Generation Tutorial
The `VideoGenerator` class provides the primary Python interface for doing offline video generation, which is interacting with a diffusion pipeline without using a separate inference api server.


## Usage
The first script in this example shows the most basic usage of FastVideo. If you are new to Python and FastVideo, you should start here.

```bash
python fastvideo/v1/examples/inference/basic/basic.py
```

# Basic Walkthrough

All you need to generate videos using multi-gpus from state-of-the-art diffusion pipelines is the following few lines! 

```python
from fastvideo import VideoGenerator

generator = VideoGenerator.from_pretrained(
    "FastVideo/FastHunyuan-Diffusers",
    num_gpus=2,
)

prompt = "A beautiful woman in a red dress walking down a street"
video = generator.generate_video(prompt)
```

More to come! These examples and APIs are still under construction!