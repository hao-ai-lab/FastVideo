# `fastvideo/models/` — Model Implementations

**Generated:** 2026-05-02

DiT / VAE / encoder / scheduler / upsampler / audio model classes. **Pre-commit excludes this directory**, except Wan's configs, definition, and `__init__.py`. Match neighboring file style manually for excluded files.

## Layout

```
models/
├── wan/                       # Family-local dense/causal Wan transformers, VAE, and configs
├── dits/
│   ├── <model>.py              # Single-file DiT (ltx2, hunyuanvideo, cosmos, ...); wanvideo is a shim
│   ├── hyworld/                # Multi-file DiT family
│   ├── lingbotworld/           # ditto
│   └── matrixgame2/            # ditto
├── vaes/                       # AutoencoderKL variants + shared utilities; wanvae is a shim
├── encoders/                   # T5, CLIP, Llama, Qwen2.5, Gemma, SigLIP, Reason1, audio conditioner
├── schedulers/                 # FlowMatch / EulerDiscrete / DPM custom schedulers
├── upsamplers/                 # Hunyuan15 super-resolution
├── audio/                      # Audio-VAE/decoder modules (LTX-2 audio, Stable Audio)
├── camera/                     # Camera-conditioning modules (Gen3C)
└── loader/                     # component_loader.py, fsdp_load.py, weight_utils.py
```

`loader/component_loader.py` is the central entry point that the pipeline uses
to instantiate model components from a HF directory. New components plug in
through `register_*` calls or by extending the `ComponentLoader` mappings.

Wan is the first family-local package: edit `wan/transformer.py` with
`wan/config.py`, `wan/causal_transformer.py`, or `wan/vae.py` with
`wan/vae_config.py`. The old `dits/wanvideo.py`, `dits/causal_wanvideo.py`,
`vaes/wanvae.py`, and matching `configs/models/` paths remain
compatibility re-exports. Shared encoders and VAE utilities stay in their current
locations. Wan pipeline defaults and variant metadata live in
`wan/pipeline_config.py` and `wan/definition.py`; old pipeline config imports
remain aliases. See `wan/AGENTS.md`; do not migrate other components as part of
a Wan edit.

## Adding a Model Component (DiT / VAE / Encoder)

1. Read `fastvideo/layers/AGENTS.md` first — it defines which tensor-parallel
   linear / attention layer to use. Do not freelance.
2. Define the arch in `models/<role>/<model>.py`. Mirror the official reference's
   constructor args; do not "improve" the layer choices.
3. Add the matching arch config in `configs/models/<role>/<model>.py`.
4. Expose `param_names_mapping` on the config — it is the **source of truth** for
   the converter under `scripts/checkpoint_conversion/`.
5. Use `init_logger(__name__)`, not stdlib logging.

## State-Dict Discipline

- The native component's `state_dict()` defines target keys + shapes.
- Conversion scripts (`scripts/checkpoint_conversion/`) bend to the model, not
  the other way around.
- Fused QKV / packed MLP layouts must be documented in the config or the model
  module — converters need to split/fuse accordingly.

## Anti-Patterns

- Importing `transformers` / `diffusers` model classes at runtime inside the
  forward path — these belong in the loader, not the architecture file.
- Adding training-only state (EMA buffers, optimizer state) to the inference
  state-dict surface.
- Calling `torch.distributed` directly. Go through `fastvideo.distributed`.
- Treating this directory as lint-clean. It isn't (see pre-commit excludes).
