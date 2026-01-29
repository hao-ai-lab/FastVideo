# FastVideo Architecture Overview

This document summarizes how FastVideo is structured and how a Diffusers-style
model repo maps into a runnable pipeline. It is intended for contributors who
need the high-level layout and key entrypoints, not every internal detail.

## FastVideo structure at a glance

FastVideo maps a Diffusers-style repo into a pipeline like this:

- `fastvideo/models/*`: model implementations (DiT, VAE, encoders, upsamplers).
- `fastvideo/configs/models/*`: arch configs and `param_names_mapping` for
  weight name translation.
- `fastvideo/configs/pipelines/*`: pipeline wiring (component classes + names).
- `fastvideo/configs/sample/*`: default runtime sampling parameters.
- `fastvideo/pipelines/basic/*`: end-to-end pipelines.
- `fastvideo/pipelines/stages/*`: reusable pipeline stages.
- `fastvideo/models/loader/*`: component loaders for Diffusers-style repos.
- `model_index.json`: HF repo entrypoint mapping component names to classes.

Flow:
`model_index.json` -> component loaders -> model modules -> pipeline stages ->
sampling params.

Minimal usage (from `examples/inference/basic/basic.py`):

```python
from fastvideo import VideoGenerator
from fastvideo.configs.sample import SamplingParam

model_id = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"  # or official_weights/<model_name>/
generator = VideoGenerator.from_pretrained(model_id, num_gpus=1)

sampling = SamplingParam.from_pretrained(model_id)
sampling.num_frames = 45
video = generator.generate_video(
    "A vibrant city street at sunset.",
    sampling_param=sampling,
    output_path="video_samples",
    save_video=True,
)
```

## Configuration system

FastVideo uses typed configs to keep model definitions, pipeline wiring, and
runtime parameters consistent:

- `fastvideo/configs/models/`: architecture definitions, layer shapes, and
  `param_names_mapping` rules for key renaming.
- `fastvideo/configs/pipelines/`: pipeline wiring and required components.
- `fastvideo/configs/sample/`: default sampling parameters (steps, frames,
  guidance scale, resolution, fps).
- `fastvideo/configs/registry.py`: pipeline registry.
- `fastvideo/configs/sample/registry.py`: sampling param registry.

`FastVideoArgs` (in `fastvideo/fastvideo_args.py`) provides runtime settings and
is passed into pipeline construction and stages.

## Weights and Diffusers format

FastVideo follows the HuggingFace Diffusers repo layout. This keeps loaders
compatible with HF repos and makes it easy to add new components.

Typical Diffusers repo:

```
<model-repo>/
  model_index.json
  scheduler/
    scheduler_config.json
  transformer/               # or unet/ for image models
    config.json
    diffusion_pytorch_model.safetensors
  vae/
    config.json
    diffusion_pytorch_model.safetensors
  text_encoder/
    config.json
    model.safetensors
  tokenizer/
    tokenizer_config.json
    tokenizer.json
```

Key points:

- `model_index.json` is the root map that tells FastVideo which components to
  load and which classes implement them.
- Each component lives in its own folder with a `config.json` and weights.
- Weights are usually in `diffusion_pytorch_model.safetensors`.

Note on tensor names:

Official checkpoints often use different `state_dict` names than FastVideo's
module layout. We translate tensor names via the DiT arch config mapping
(`param_names_mapping` under `fastvideo/configs/models/dits/`). This is similar
in spirit to name-translation layers used in systems like vLLM and SGLang.

Example HF repo (Wan 2.1 T2V 1.3B Diffusers):

```
https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B-Diffusers/tree/main
```

Example `model_index.json` from that repo:

```json
{
  "_class_name": "WanPipeline",
  "_diffusers_version": "0.33.0.dev0",
  "scheduler": [
    "diffusers",
    "UniPCMultistepScheduler"
  ],
  "text_encoder": [
    "transformers",
    "UMT5EncoderModel"
  ],
  "tokenizer": [
    "transformers",
    "T5TokenizerFast"
  ],
  "transformer": [
    "diffusers",
    "WanTransformer3DModel"
  ],
  "vae": [
    "diffusers",
    "AutoencoderKLWan"
  ]
}
```

How this maps to FastVideo:

- `WanPipeline` -> `fastvideo/pipelines/basic/wan/wan_pipeline.py`
- `WanTransformer3DModel` -> `fastvideo/models/dits/wanvideo.py`
- `AutoencoderKLWan` -> `fastvideo/models/vaes/wanvae.py`
- `UMT5EncoderModel` -> `fastvideo/models/encoders/t5.py`
- `T5TokenizerFast` -> loaded via HF in `fastvideo/models/loader/`
- `UniPCMultistepScheduler` -> loaded via Diffusers scheduler utilities
- Pipeline defaults -> `fastvideo/configs/pipelines/wan.py`
- Sampling defaults -> `fastvideo/configs/sample/wan.py`

## Pipeline system

- `fastvideo/pipelines/basic/*` contains end-to-end pipelines for each model
  family.
- `fastvideo/pipelines/stages/*` contains reusable, testable stages.
- Pipelines subclass `ComposedPipelineBase` and declare required components via
  `_required_config_modules`.
- `ForwardBatch` (in `fastvideo/pipelines/pipeline_batch_info.py`) carries
  prompts, latents, timesteps, and intermediate state across stages.

## Model components

- DiT models: `fastvideo/models/dits/`
- VAEs: `fastvideo/models/vaes/`
- Text/image encoders: `fastvideo/models/encoders/`
- Schedulers: `fastvideo/models/schedulers/`
- Upsamplers: `fastvideo/models/upsamplers/`
- Optional audio models: `fastvideo/models/audio/`

## Attention and distributed execution

- Attention backends live in `fastvideo/attention/` and can be selected via
  `FASTVIDEO_ATTENTION_BACKEND`.
- `LocalAttention` is used for cross-attention and most attention layers.
- `DistributedAttention` is used for full-sequence self-attention in the DiT.
- Tensor-parallel layers live in `fastvideo/layers/`.
- Sequence/tensor parallel utilities live in `fastvideo/distributed/`.

## Related docs

- [Contributing overview](../contributing/overview.md)
- [Coding agents workflow](../contributing/coding_agents.md)
- [Testing guide](../contributing/testing.md)
