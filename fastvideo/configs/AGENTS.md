# `fastvideo/configs/` — Config-Driven Model Registry

Two layers of dataclass configs feed every pipeline: **arch configs** (what the model is) and **pipeline configs** (how to run it).

## Layout

```
configs/
├── configs.py                  # Dataset / loader enums (DatasetType, VideoLoaderType)
├── utils.py                    # update_config_from_args, shallow_asdict helpers
├── backend/                    # Attention backend defaults
├── models/
│   ├── base.py                 # ModelConfig ABC
│   ├── dits/                   # DiTConfig per model (wanvideo, ltx2, hunyuan, ...)
│   ├── vaes/                   # VAEConfig per model
│   ├── encoders/               # EncoderConfig (t5, clip, llama, qwen2_5, gemma, siglip, ...)
│   ├── upsamplers/             # UpsamplerConfig (hunyuan15)
│   └── audio/                  # Audio-model configs (ltx2_audio_vae, ...)
├── pipelines/
│   ├── base.py                 # PipelineConfig ABC + (de)serialization
│   └── <model>.py              # Concrete configs (HunyuanConfig, WanT2V480PConfig, ...)
└── *.json                      # Frozen reference configs for shipped models
```

## How Configs Hook Into the Registry

`fastvideo/registry.py` imports every concrete `PipelineConfig` and exposes
`get_pipeline_config_cls_from_name(...)`. Adding a new pipeline config requires:

1. Subclass `PipelineConfig` in `pipelines/<model>.py`.
2. Reference its component arch configs (DiT / VAE / encoder / upsampler).
3. Add the import + name mapping in `fastvideo/registry.py`.

Configs that do not appear in `registry.py` are unreachable from `VideoGenerator`.

## Arch vs Pipeline — Where Does This Field Go?

| Field type | Lives on |
|-----------|----------|
| Architecture constants (hidden dim, num heads, layer count) | `configs/models/<role>/<model>.py` |
| Default sampling params (steps, cfg, shift, fps) | `configs/pipelines/<model>.py` |
| Runtime overrides (precision, sp_size, tp_size, attention backend) | `configs/pipelines/base.py` defaults + CLI flags via `fastvideo_args.py` |
| `param_names_mapping` for HF → FastVideo state-dict | Arch config (lives with the model definition) |

If a knob is tunable per inference call → `SamplingParam`, not `PipelineConfig`.

## Anti-Patterns

- Hard-coding architecture constants inside model classes — always read from the arch config.
- Using `argparse` directly here. Configs deserialize from dicts via `update_config_from_args`.
- Importing from `fastvideo.pipelines` here. Configs are the lower layer; the dependency is one-way.
