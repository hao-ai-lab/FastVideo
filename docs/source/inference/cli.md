# FastVideo CLI Inference

The FastVideo CLI provides a quick way to access the FastVideo inference pipeline for video generation. For more advanced usage,
see the Python interface [here](https://hao-ai-lab.github.io/FastVideo/inference/examples/basic.html).

## Basic Usage

The basic command to generate a video is:

```bash
fastvideo generate --model-path {MODEL_PATH} --prompt {PROMPT}
```

### Required Parameters

- `--model-path {MODEL_PATH}`: Path to the model or model ID
- `--prompt {PROMPT}`: Text description for the video you want to generate

## Common Arguments

To see all the options, you can use the `--help` flag:

```bash
fastvideo generate --help
```

### Hardware Configuration

- `--num-gpus {NUM_GPUS}`: Number of GPUs to use
- `--tp-size {TP_SIZE}`: Tensor parallelism size (only for the encoder, should not be larger than 1 if text encoder offload is enabled, as layerwise offload + prefetch is faster)
- `--sp-size {SP_SIZE}`: Sequence parallelism size (Typically should match the number of GPUs)

#### Video Configuration

- `--height {HEIGHT}`: Height of the generated video
- `--width {WIDTH}`: Width of the generated video
- `--num-frames {NUM_FRAMES}`: Number of frames to generate
- `--fps {FPS}`: Frames per second for the saved video

#### Generation Parameters

- `--num-inference-steps {STEPS}`: Number of denoising steps
- `--negative-prompt {PROMPT}`: Negative prompt to guide generation away from certain concepts
- `--seed {SEED}`: Random seed for reproducible generation

#### Output Options

- `--output-path {PATH}`: Directory to save the generated video
- `--save-video`: Whether to save the video to disk
- `--return-frames`: Whether to return the raw frames

## Using Configuration Files

Instead of specifying all parameters on the command line, you can use a configuration file:

```bash
fastvideo generate --config {CONFIG_FILE_PATH}
```

The config file should be in JSON or YAML format with the same parameter names as the CLI options. Command-line arguments will take precedence over settings in the configuration file, allowing you to override specific values while keeping the rest from the config file.

Example configuration file (config.json):

```json
{
    "model_path": "FastVideo/FastHunyuan-diffusers",
    "prompt": "A beautiful woman in a red dress walking down a street",
    "output_path": "outputs/",
    "num_gpus": 2,
    "sp_size": 2,
    "tp_size": 1,
    "num_frames": 45,
    "height": 720,
    "width": 1280,
    "num_inference_steps": 6,
    "seed": 1024,
    "fps": 24,
    "precision": "bf16",
    "vae_precision": "fp16",
    "vae_tiling": true,
    "vae_sp": true,
    "vae_config": {
        "load_encoder": false,
        "load_decoder": true,
        "tile_sample_min_height": 256,
        "tile_sample_min_width": 256
    },
    "text_encoder_precisions": [
        "fp16",
        "fp16"
    ],
    "mask_strategy_file_path": null,
    "enable_torch_compile": false
}
```

Or using YAML format (config.yaml):

```yaml
model_path: "FastVideo/FastHunyuan-diffusers"
prompt: "A beautiful woman in a red dress walking down a street"
output_path: "outputs/"
num_gpus: 2
sp_size: 2
tp_size: 1
num_frames: 45
height: 720
width: 1280
num_inference_steps: 6
seed: 1024
fps: 24
precision: "bf16"
vae_precision: "fp16"
vae_tiling: true
vae_sp: true
vae_config:
  load_encoder: false
  load_decoder: true
  tile_sample_min_height: 256
  tile_sample_min_width: 256
text_encoder_precisions:
  - "fp16"
  - "fp16"
mask_strategy_file_path: null
enable_torch_compile: false
```

## Examples

Generating a simple video:

```bash
fastvideo generate --model-path FastVideo/FastHunyuan-diffusers --prompt "A cat playing with a ball of yarn" --num-frames 45 --height 720 --width 1280 --num-inference-steps 6 --seed 1024 --output-path outputs/
```

Using a negative prompt to avoid certain elements:

```bash
fastvideo generate --model-path FastVideo/FastHunyuan-diffusers --prompt "A beautiful forest landscape" --negative-prompt "people, buildings, roads"
```

Combining command line arguments and a configuration file:

```bash
fastvideo generate --config config.json --prompt "A capybara lounging in a hammock"
```

## Troubleshooting

- If you encounter CUDA out-of-memory errors, try reducing the video dimensions or number of frames, or the number of inference steps.
- For reproducible results, set the same seed value between runs.
