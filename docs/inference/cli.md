# FastVideo CLI Inference

The FastVideo CLI exposes the same core inference controls as the Python API.

## Basic Usage

Use either:

1. `--model-path` + `--prompt`
2. `--model-path` + `--prompt-txt` (batch prompts, one line per prompt)
3. `--config` (JSON/YAML)

```bash
fastvideo generate --model-path Wan-AI/Wan2.1-T2V-1.3B-Diffusers \
  --prompt "A cat playing with a ball of yarn"
```

```bash
fastvideo generate --model-path Wan-AI/Wan2.1-T2V-1.3B-Diffusers \
  --prompt-txt prompts.txt
```

You cannot provide both `--prompt` and `--prompt-txt` in the same run.

## View All Arguments

```bash
fastvideo generate --help
```

Arguments come from:

- FastVideo runtime args (`FastVideoArgs`)
- Sampling args (`SamplingParam`)
- Pipeline config args (`PipelineConfig`)

## Common Arguments

### Parallelism

- `--num-gpus`
- `--sp-size`
- `--tp-size`

### Sampling

- `--num-frames`
- `--height` / `--width`
- `--num-inference-steps`
- `--guidance-scale`
- `--seed`
- `--negative-prompt`

### Output

- `--output-path`
- `--save-video` / `--no-save-video`
- `--return-frames`

### Offloading and Performance

- `--dit-layerwise-offload`
- `--use-fsdp-inference`
- `--text-encoder-cpu-offload`
- `--image-encoder-cpu-offload`
- `--vae-cpu-offload`
- `--enable-torch-compile`
- `--torch-compile-kwargs`

## Using Config Files

```bash
fastvideo generate --config config.yaml
```

Config files can be JSON or YAML. CLI flags override config-file values.

Example `config.yaml`:

```yaml
model_path: "FastVideo/FastHunyuan-diffusers"
prompt: "A capybara lounging in a hammock"
output_path: "outputs/"
num_gpus: 2
sp_size: 2
tp_size: 1
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

Notes:

- Use `dit_precision` / `vae_precision` (not `precision`).
- Nested config objects are supported, for example `vae_config` and
  `dit_config`.

## Examples

Simple generation:

```bash
fastvideo generate \
  --model-path FastVideo/FastHunyuan-diffusers \
  --prompt "A cat playing with a ball of yarn" \
  --num-frames 45 --height 720 --width 1280 \
  --num-inference-steps 6 --seed 1024 \
  --output-path outputs/
```

Config + CLI override:

```bash
fastvideo generate --config config.yaml --prompt "A panda skiing at sunset"
```
