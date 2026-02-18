# Configuration

## Multi-GPU Setup

FastVideo automatically distributes the generation process when multiple GPUs are specified:

```python
# Will use 4 GPUs in parallel for faster generation
generator = VideoGenerator.from_pretrained(
    "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
    num_gpus=4,
)
```

## Customizing Generation

- `PipelineConfig`: Initialization time parameters
- `SamplingParam`: Generation time parameters

You can customize generation behavior using `PipelineConfig` and
`SamplingParam`:

```python
from fastvideo import VideoGenerator, SamplingParam, PipelineConfig

def main():
    model_name = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
    config = PipelineConfig.from_pretrained(model_name)
    config.vae_precision = "fp16"

    # Create the generator
    generator = VideoGenerator.from_pretrained(
        model_name,
        num_gpus=1,
        dit_layerwise_offload=True,  # FastVideoArgs option
        pipeline_config=config
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

## JSON/YAML Config Files (CLI)

The CLI supports `--config` with JSON or YAML. Command-line arguments override
config file values.

```bash
fastvideo generate --config config.yaml
```

Use CLI argument names as keys (underscore or hyphen is accepted). Example:

```yaml
model_path: "FastVideo/FastHunyuan-diffusers"
prompt: "A capybara relaxing in a hammock"
num_gpus: 2
sp_size: 2
num_frames: 45
height: 720
width: 1280
num_inference_steps: 6
seed: 1024
dit_precision: "bf16"
vae_precision: "fp16"
vae_tiling: true
vae_sp: true
enable_torch_compile: false
```

## Performance Optimization

For configuring optimizations, please see our [optimizations guide](optimizations.md)
