# FastVideo Gradio Demo

This is a Gradio-based web interface for generating videos using the FastVideo framework. The demo allows users to create videos from text prompts with various customization options.

## Overview

The demo uses the FastVideo framework to generate videos based on text prompts. It provides a simple web interface built with Gradio that allows users to:

- Enter text prompts to generate videos
- Customize video parameters (dimensions, number of frames, etc.)
- Use negative prompts to guide the generation process
- Set or randomize seeds for reproducibility

---

## Usage

Run the demo with:

```bash
python examples/inference/gradio/gradio_demo.py
```

This will start a web server at `http://0.0.0.0:7860` where you can access the interface.

---

## Model Initialization

This demo initializes a `VideoGenerator` with the minimum required arguments for inference. Users can seamlessly adjust inference options between generations, including prompts, resolution, video length, or even the number of inference steps, *without ever needing to reload the model*.

## Video Generation

The core functionality is in the `generate_video` function, which:
1. Processes user inputs
2. Uses the FastVideo VideoGenerator from earlier to run inference (`generator.generate_video()`)
3. Returns an output path that Gradio uses to display the generated video

## Gradio Interface

The interface is built with several components:
- A text input for the prompt
- A video display for the result
- Inference options in a collapsible accordion:
  - Height and width sliders
  - Number of frames slider
  - Guidance scale slider
  - Inference steps slider
  - Negative prompt options
  - Seed controls

### Inference Options

- **Height/Width**: Control the resolution of the generated video
- **Number of Frames**: Set how many frames to generate
- **Guidance Scale**: Control how closely the generation follows the prompt
- **Inference Steps**: More steps can improve quality but take longer
- **Negative Prompt**: Specify what you don't want to see in the video
- **Seed**: Control randomness for reproducible results
