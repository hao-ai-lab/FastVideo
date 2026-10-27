# FastVideo Architecture Overview

This document summarizes how FastVideo is structured and how a Diffusers-style
model repo maps into a runnable pipeline. It is intended for contributors who
need the high-level layout and key entrypoints, not every internal detail.

## FastVideo structure at a glance

FastVideo maps a Diffusers-style repo into a pipeline like this:

- `fastvideo/models/*`: model implementations (DiT, VAE, encoders, upsamplers).
- `fastvideo/configs/models/*`: arch configs and `param_names_mapping` for
  weight name translation.
- `fastvideo/models/wan/`: Wan's transformers, VAE, configs, and variant definitions
  together; the old Wan component/config modules remain compatibility imports.
- `fastvideo/configs/pipelines/*`: pipeline wiring (component classes + names).
- `fastvideo/api/sampling_param.py`: runtime sampling parameters.
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
from fastvideo.api.sampling_param import SamplingParam

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
- `fastvideo/models/wan/config.py`: Wan's transformer architecture and mapping rules,
  co-located with `transformer.py`.
- `fastvideo/models/wan/vae_config.py`: Wan's VAE architecture and runtime
  settings, co-located with `vae.py`.
- `fastvideo/models/wan/pipeline_config.py`: Wan component and pipeline defaults;
  `configs/pipelines/wan.py` remains a compatibility import.
- `fastvideo/models/wan/definition.py`: data-only Wan variants linking HF aliases,
  config classes, presets, workload metadata, and default sampling algorithms.
- `fastvideo/configs/pipelines/`: pipeline wiring and required components.
- `fastvideo/api/sampling_param.py`: sampling parameters (steps, frames,
  guidance scale, resolution, fps). Defaults come from presets in
  `fastvideo/pipelines/basic/<family>/presets.py`.
- `fastvideo/registry.py`: unified registry for pipeline config + sampling
  defaults and model metadata resolution. Wan registrations consume its
  family-local definitions; other families use `register_configs(...)` blocks.

Wan definitions reference existing defaults rather than copying them. Dense
UniPC, dense DMD, and causal DMD remain separate sampling algorithms. Dense
DMD uses a full training-noise scheduler with shift 8.0, separate from the
configurable scheduler mutated during timestep preparation. The catalog does
not override checkpoint manifests, user pipeline overrides, or component
precision settings. HF IDs, local checkpoints, and old config imports retain
their existing resolution behavior, including first-match detector ordering.

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
(`param_names_mapping` under `fastvideo/configs/models/dits/`, or
`fastvideo/models/wan/config.py` for Wan). This is similar
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
- `WanTransformer3DModel` -> `fastvideo/models/wan/transformer.py`
- `WanVideoConfig` -> `fastvideo/models/wan/config.py`
- `AutoencoderKLWan` -> `fastvideo/models/wan/vae.py`
- `WanVAEConfig` -> `fastvideo/models/wan/vae_config.py`
- `UMT5EncoderModel` -> `fastvideo/models/encoders/t5.py`
- `T5TokenizerFast` -> loaded via HF in `fastvideo/models/loader/`
- `UniPCMultistepScheduler` -> loaded via Diffusers scheduler utilities
- Variant definitions -> `fastvideo/models/wan/definition.py`
- Pipeline defaults -> `fastvideo/models/wan/pipeline_config.py`
- Sampling defaults -> `fastvideo/pipelines/basic/wan/presets.py`

## Pipeline system

- `fastvideo/pipelines/basic/*` contains end-to-end pipelines for each model
  family.
- `fastvideo/pipelines/stages/*` contains reusable, testable stages.
- Pipelines subclass `ComposedPipelineBase` and declare required components via
  `_required_config_modules`.
- `ForwardBatch` (in `fastvideo/pipelines/pipeline_batch_info.py`) carries
  prompts, latents, timesteps, and intermediate state across stages.

## Model components

- DiT models: `fastvideo/models/dits/`; dense Wan: `fastvideo/models/wan/`
- VAEs: `fastvideo/models/vaes/`; Wan: `fastvideo/models/wan/vae.py`
- Text/image encoders: `fastvideo/models/encoders/`
- Schedulers: `fastvideo/models/schedulers/`
- Upsamplers: `fastvideo/models/upsamplers/`
- Optional audio models: `fastvideo/models/audio/`

## Attention and distributed execution

- Attention backends live in `fastvideo/attention/` and can be selected via
  `FASTVIDEO_ATTENTION_BACKEND`.
- SageAttention3 is split into two selectable backends:
  `SAGE_ATTN_THREE` for the regular upstream package and
  `ATTN_QAT_INFER` for the FastVideoKernel-backed inference variant.
- `ATTN_QAT_TRAIN` is a separate FastVideoKernel Triton backend for the QAT attention
  path.
- `LocalAttention` is used for cross-attention and most attention layers.
- `DistributedAttention` is used for full-sequence self-attention in the DiT.
- Tensor-parallel layers live in `fastvideo/layers/`.
- Sequence/tensor parallel utilities live in `fastvideo/distributed/`.

## Related docs

- [Contributing overview](../contributing/overview.md)
- [Coding agents workflow](../contributing/coding_agents.md)
- [Testing guide](../contributing/testing.md)
