# Configuration

## Multi-GPU Setup

FastVideo automatically distributes the generation process when multiple GPUs are specified:

```python
# Will use 4 GPUs in parallel for faster generation
generator = VideoGenerator.from_pretrained(
    "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
    {"engine": {"num_gpus": 4}},
)
```

One node uses the multiprocessing executor (`execution_backend: mp`, the
default). Two machines — for example two DGX Sparks, one GPU each — need Ray:

```yaml
generator:
  engine:
    num_gpus: 2
    execution_backend: ray
    parallelism:
      sp_size: 2
```

Set `RAY_ADDRESS` and `FASTVIDEO_HOST_IP` to the interconnect IPs, not Wi-Fi.
The FastH3 example selects Ray automatically when `RAY_ADDRESS` is set. Full
bring-up: [Pair two NVIDIA DGX Sparks](../getting_started/installation/spark_pair.md).

## Customizing Generation

`VideoGenerator.from_pretrained(model_path, config)` takes the startup
settings as a nested mapping at their typed config paths, such as
`{"engine": {"num_gpus": 2, "offload": {"dit": False}}}`; it is
`VideoGenerator.from_config` with `model_path` added to the mapping. Pass
generation settings to `VideoGenerator.generate` as a request:

```python
from fastvideo import VideoGenerator

def main():
    model_name = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"

    # Create the generator
    generator = VideoGenerator.from_config({
        "model_path": model_name,
        "engine": {
            "num_gpus": 1,
            "offload": {"dit_layerwise": True},
            "precision": {"vae": "fp16"},
        },
    })

    # Generate video with custom parameters
    prompt = "A beautiful sunset over a calm ocean, with gentle waves."
    video = generator.generate({
        "prompt": prompt,
        "sampling": {
            # How many frames to generate
            "num_frames": 45,
            # Video resolution (width, height)
            "width": 1024,
            "height": 576,
            # How many steps we denoise the video (higher = better quality, slower generation)
            "num_inference_steps": 30,
            # How strongly the video conforms to the prompt (higher = more faithful to prompt)
            "guidance_scale": 7.5,
            # Random seed for reproducibility
            "seed": 42,  # Optional, leave unset for random results
        },
        "output": {
            "output_path": "my_videos/",  # Controls where videos are saved
            "save_video": True,
        },
    })

    # If return_frames=True, frames are available in video.frames
    print(f"Generated {len(video.frames)} frames")

if __name__ == '__main__':
    main()
```

## JSON/YAML Config Files (CLI)

The inference CLI is config-first. Use an explicit subcommand with `--config`,
then apply optional dotted overrides on top, matching the training CLI style.
By default, CLI generation uses `return_frames=false` unless you set
`request.output.return_frames: true` in config or via a dotted override.

```bash
fastvideo generate --config config.yaml
```

Example nested config:

```yaml
generator:
  model_path: FastVideo/FastHunyuan-diffusers
  engine:
    num_gpus: 2
    parallelism:
      sp_size: 2
request:
  prompt: A capybara relaxing in a hammock
  sampling:
    num_frames: 45
    height: 720
    width: 1280
    num_inference_steps: 6
    seed: 1024
  output:
    output_path: outputs/
```

Override individual values from the CLI with dotted paths:

```bash
fastvideo generate --config config.yaml --request.sampling.seed 42
```

## Where a Value Came From

FastVideo resolves the generator config once at startup and records the source of every value: the input config
(`input`; `explicit` tells whether you wrote the value or it is the schema default), a `FASTVIDEO_*` environment
variable, the model's defaults, or a derived value. A worker's device policy and values read from checkpoint files
are recorded too.

```python
generator = VideoGenerator.from_config(config)
generator.resolved_config.provenance("engine.parallelism.sp_size")
# PathProvenance(path='engine.parallelism.sp_size', value=2, source='derive_parallel_sizes', ...)

result = generator.generate(request)
result.resolved_request.provenance("sampling.num_frames")
# PathProvenance(..., value=81, source='fill_sampling_defaults[preset wan_t2v_1_3b]', explicit=False)
```

`resolved_config.provenance_table()` lists every path. Every value is decided before resolution ends, including the
device offload policy and the checkpoint defaults; after that, `resolved_config` is read-only.

## Performance Optimization

For configuring optimizations, please see our [optimizations guide](optimizations.md)
