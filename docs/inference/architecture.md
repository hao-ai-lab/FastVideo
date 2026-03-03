# Inference Architecture

This section documents the FastVideo inference pipeline: how models are
discovered, configs resolved, components loaded, and stages composed to
generate video. Training-specific code paths (FSDP, gradient checkpointing)
are out of scope.

## Registries

FastVideo uses three registries that work together to resolve a
user-provided `model_path` into a runnable pipeline.

### Model Registry (`fastvideo/models/registry.py`)

Maps HuggingFace architecture class names (e.g. `"WanTransformer3DModel"`)
to FastVideo model classes. Two discovery mechanisms:

1. **Hardcoded dicts** — `_TEXT_TO_VIDEO_DIT_MODELS`, `_VAE_MODELS`,
   `_SCHEDULERS`, `_TEXT_ENCODER_MODELS`, `_IMAGE_ENCODER_MODELS`,
   `_UPSAMPLERS`, `_AUDIO_MODELS`. Each entry is
   `{hf_class_name: (component_name, module_name, class_name)}`.
2. **AST-based discovery** — `_discover_and_register_models()` walks
   `fastvideo/models/` and parses each `.py` file's AST looking for an
   `EntryClass` variable assignment. Discovered models take priority over
   hardcoded entries. For example,
   `fastvideo/models/dits/wanvideo.py` exports
   `EntryClass = WanTransformer3DModel`.

Both feed into a unified `_FAST_VIDEO_MODELS` dict, which populates the
singleton `ModelRegistry` — an instance of `_ModelRegistry`. Components are
wrapped in `_LazyRegisteredModel` for deferred import.

**Key API:** `ModelRegistry.resolve_model_cls(architectures)` iterates
candidate architecture strings and returns the first matching
`(model_cls, arch)` tuple. Called by `TransformerLoader`, `VAELoader`, etc.

### Config Registry (`fastvideo/registry.py`)

Maps model paths and names to `(PipelineConfig, SamplingParam)` class
pairs. Registration happens at module load via `_register_configs()`, which
calls `register_configs()` for each model family:

```python
register_configs(
    sampling_param_cls=WanT2V_1_3B_SamplingParam,
    pipeline_config_cls=WanT2V480PConfig,
    hf_model_paths=["Wan-AI/Wan2.1-T2V-1.3B-Diffusers"],
    model_detectors=[lambda path: "wanpipeline" in path.lower()],
)
```

Each call populates three data structures:
- `_CONFIG_REGISTRY: dict[str, ConfigInfo]` — auto-incrementing ID to
  `ConfigInfo(sampling_param_cls, pipeline_config_cls)`.
- `_MODEL_HF_PATH_TO_NAME: dict[str, str]` — HF path to registry ID.
- `_MODEL_NAME_DETECTORS: list[tuple[str, Callable]]` — lambda detectors.

**Resolution priority** (`_get_config_info()`):
1. Exact HF path match in `_MODEL_HF_PATH_TO_NAME`.
2. Partial match on short model name (last path segment, case-insensitive).
3. Detector-based match — runs each detector against the lowercased path
   and the `_class_name` from `model_index.json`.
4. `RuntimeError` if no match.

**Top-level resolver:** `get_model_info(model_path, pipeline_type,
workload_type)` combines config resolution with pipeline resolution to
return a `ModelInfo(pipeline_cls, sampling_param_cls, pipeline_config_cls)`.

### Pipeline Registry (`fastvideo/pipelines/pipeline_registry.py`)

Discovers pipeline classes by scanning Python packages under
`fastvideo/pipelines/{basic,preprocess,training}/`.

`import_pipeline_classes()` iterates architecture subdirectories
(e.g. `wan/`, `hunyuan/`), imports each module, and collects those
exporting an `EntryClass` attribute. Supports single class or list.
Returns `{pipeline_type_str: {pipeline_class_name: pipeline_cls}}`.

`_PipelineRegistry.resolve_pipeline_cls(pipeline_name, pipeline_type,
workload_type)` looks up the pipeline class by the `_class_name` field
from `model_index.json`.

## Config Mechanism

### Config Hierarchy

```
PipelineConfig                    (fastvideo/configs/pipelines/base.py)
├── WanT2V480PConfig              (fastvideo/configs/pipelines/wan.py)
│   ├── WanT2V720PConfig
│   └── WanI2V480PConfig
├── HunyuanConfig                 (fastvideo/configs/pipelines/hunyuan.py)
├── LTX2T2VConfig                 (fastvideo/configs/pipelines/ltx2.py)
├── CosmosConfig                  (fastvideo/configs/pipelines/cosmos.py)
└── ... (15+ model families)
```

`PipelineConfig` holds:
- Video generation params: `embedded_cfg_scale`, `flow_shift`,
  `disable_autocast`, `is_causal`.
- Nested model configs: `dit_config: DiTConfig`, `vae_config: VAEConfig`,
  `text_encoder_configs: tuple[EncoderConfig, ...]`.
- Precision settings: `dit_precision`, `vae_precision`,
  `text_encoder_precisions`.

Model-specific subclasses override defaults. For example,
`WanT2V480PConfig` sets `flow_shift=3.0` and uses `WanVideoConfig` as
its DiT config.

### ModelConfig / ArchConfig (`fastvideo/configs/models/base.py`)

`ModelConfig` wraps an `ArchConfig` using `__getattr__` proxy — attribute
access falls through to `arch_config` transparently. `ArchConfig` holds
architecture fields from `config.json` (hidden_size, num_attention_heads,
etc.) and is immutable after initialization. `update_model_arch()` writes
to `ArchConfig`; `update_model_config()` writes to `ModelConfig` fields.

Concrete hierarchy: `DiTConfig` → `DiTArchConfig`, `VAEConfig` →
`VAEArchConfig`, `EncoderConfig` → `EncoderArchConfig`.

### Config Construction

- `PipelineConfig.from_pretrained(model_path)` — resolves config class
  via `get_pipeline_config_cls_from_name()`, instantiates with defaults.
- `PipelineConfig.from_kwargs(kwargs)` — resolves class, optionally loads
  JSON via `load_from_json()`, then applies CLI overrides via
  `update_config_from_dict()`.
- `dump_to_json()` / `load_from_json()` — JSON persistence. Callable
  fields and `arch_config` are excluded from dumps.

### SamplingParam (`fastvideo/configs/sample/`)

Generation parameters separate from pipeline config. Each model family
provides defaults:

```python
@dataclass
class WanT2V_1_3B_SamplingParam(SamplingParam):
    height: int = 480
    width: int = 832
    num_frames: int = 81
    guidance_scale: float = 3.0
    num_inference_steps: int = 50
```

## Component Loading

### ComponentLoader (`fastvideo/models/loader/component_loader.py`)

Abstract base with a `load(model_path, fastvideo_args)` method.
`ComponentLoader.for_module_type(module_type, library)` is a factory
that dispatches to specialized loaders via a `module_loaders` dict:

| Module type | Loader class | Library |
|---|---|---|
| `scheduler` | `SchedulerLoader` | diffusers |
| `transformer`, `transformer_2`, `transformer_3` | `TransformerLoader` | diffusers |
| `vae` | `VAELoader` | diffusers |
| `text_encoder`, `text_encoder_2`, `text_encoder_3` | `TextEncoderLoader` | transformers |
| `tokenizer`, `tokenizer_2`, `tokenizer_3` | `TokenizerLoader` | transformers |
| `image_encoder` | `ImageEncoderLoader` | transformers |
| `image_processor`, `feature_extractor` | `ImageProcessorLoader` | transformers |
| `audio_vae`, `audio_decoder` | `AudioDecoderLoader` | diffusers |
| `vocoder` | `VocoderLoader` | diffusers |
| `upsampler`, `upsampler_2` | `UpsamplerLoader` | diffusers |

`TransformerLoader` reads `config.json` from the component directory,
resolves the class via `ModelRegistry.resolve_model_cls()`, instantiates
the model, and loads safetensors weights. CPU offload and layerwise
offload are applied based on `FastVideoArgs`.

Unknown module types fall back to `GenericComponentLoader`.

### model_index.json

Diffusers-format JSON at the model root. Keys are module names; values
are `[library, class_name]` tuples:

```json
{
    "_class_name": "WanPipeline",
    "_diffusers_version": "0.24.0",
    "transformer": ["diffusers", "WanTransformer3DModel"],
    "vae": ["diffusers", "AutoencoderKLWan"],
    "text_encoder": ["transformers", "UMT5EncoderModel"],
    "tokenizer": ["transformers", "AutoTokenizer"],
    "scheduler": ["diffusers", "FlowMatchEulerDiscreteScheduler"]
}
```

`ComposedPipelineBase.load_modules()` reads this file via
`_load_config()`, strips metadata keys (`_class_name`,
`_diffusers_version`, `_name_or_path`), detects MoE pipelines via
`boundary_ratio`, then loads each module listed in the pipeline's
`required_config_modules`. Modules not in `required_config_modules` are
skipped. `_extra_config_module_map` allows aliasing (e.g. mapping
`"transformer_2"` to an alternate directory name).

`PipelineComponentLoader.load_module()` orchestrates per-component
loading by calling `ComponentLoader.for_module_type()` then `.load()`.

## Stage Design

### PipelineStage (`fastvideo/pipelines/stages/base.py`)

Abstract base class using the Template Method pattern:

- `__call__(batch, fastvideo_args)` — orchestrates verification, timing,
  and error handling. Not overridden by subclasses.
- `forward(batch, fastvideo_args) -> ForwardBatch` — abstract, contains
  the stage logic.
- `verify_input()` / `verify_output()` — optional hooks returning
  `VerificationResult`. Default: no checks.

When `fastvideo_args.enable_stage_verification` is `True`, `__call__`
runs input verification before `forward()` and output verification after.
When `envs.FASTVIDEO_STAGE_LOGGING` is set, execution time is measured
with `torch.cuda.synchronize()` and logged.

### ForwardBatch (`fastvideo/pipelines/pipeline_batch_info.py`)

Dataclass carrying all pipeline state between stages. Key field groups:

- **Inputs**: `prompt`, `negative_prompt`, `image_path`, `pil_image`,
  `video_path`.
- **Embeddings**: `prompt_embeds: list[Tensor]`,
  `negative_prompt_embeds`, `prompt_attention_mask`, `image_embeds`.
- **Latents**: `latents`, `image_latent`, `noise_pred`,
  `lq_latents`.
- **Dimensions**: `height`, `width`, `num_frames`, `height_latents`,
  `width_latents`.
- **Scheduler**: `timesteps`, `num_inference_steps`, `guidance_scale`,
  `sigmas`.
- **Task-specific**: `mouse_cond`/`keyboard_cond` (MatrixGame), `pose`
  (HYWorld), `camera_states` (GameCraft), `c2ws_plucker_emb`
  (LingBotWorld).
- **Output**: `output: Tensor | None`.
- **Logging**: `logging_info: PipelineLoggingInfo`.

`__post_init__` enables CFG when `guidance_scale > 1.0` or LTX2 text
CFG scales differ from 1.0.

### Stage Catalog

Standard stages (typical execution order):

| Stage | File | Purpose |
|---|---|---|
| `InputValidationStage` | `stages/input_validation.py` | Validates input dimensions and types |
| `TextEncodingStage` | `stages/text_encoding.py` | Encodes prompts via text encoders |
| `ImageEncodingStage` | `stages/image_encoding.py` | Encodes input images (I2V pipelines) |
| `ConditioningStage` | `stages/conditioning.py` | Prepares conditioning embeddings |
| `TimestepPreparationStage` | `stages/timestep_preparation.py` | Sets up scheduler timesteps |
| `LatentPreparationStage` | `stages/latent_preparation.py` | Initializes noise latents |
| `DenoisingStage` | `stages/denoising.py` | Main diffusion denoising loop |
| `DecodingStage` | `stages/decoding.py` | Decodes latents to video via VAE |

Specialized variants: `CausalDenoisingStage`, `LTX2DenoisingStage`,
`LongCatDenoisingStage`, `GameCraftDenoisingStage`,
`HYWorldDenoisingStage`, `MatrixGameDenoisingStage`,
`SRDenoisingStage`, `LTX2AudioDecodingStage`, `SD35ConditioningStage`,
`LTX2TextEncodingStage`, `LTX2LatentPreparationStage`.

### Verification System (`fastvideo/pipelines/stages/validators.py`)

`StageValidators` (aliased as `V`) provides static validators:
`not_none`, `positive_int`, `is_tensor`, `tensor_with_dims`,
`positive_int_divisible(divisor)`, etc.

`VerificationResult` collects check results:
```python
result = VerificationResult()
result.add_check("height", batch.height, V.positive_int_divisible(8))
result.add_check("width", batch.width, V.positive_int_divisible(8))
```

`is_valid()` returns whether all checks passed. `get_failure_summary()`
provides detailed error messages. Failed verification raises
`StageVerificationError`.

## Pipeline Architecture

### ComposedPipelineBase (`fastvideo/pipelines/composed_pipeline_base.py`)

Abstract base for all inference pipelines. Lifecycle:

1. **`__init__(model_path, fastvideo_args)`** — initializes distributed
   environment via `maybe_init_distributed_environment_and_model_parallel
   (tp_size, sp_size)`, then calls `load_modules()` to populate
   `self.modules`.
2. **`post_init()`** — calls `initialize_pipeline()` (model-specific
   setup), `create_pipeline_stages()` (abstract — subclasses wire stages),
   optionally applies `torch.compile` to transformers, and calls
   `warmup_sequence_parallel_communication()`.
3. **`forward(batch, fastvideo_args)`** — iterates `self.stages` calling
   each stage in order. Decorated with `@torch.no_grad()`.

Key class attributes:
- `_required_config_modules: list[str]` — module names to load from
  `model_index.json`.
- `_extra_config_module_map: dict[str, str]` — aliases for module dirs.
- `is_video_pipeline: bool` — whether this produces video output.

Key methods:
- `add_stage(name, stage)` — appends to `_stages` list and
  `_stage_name_mapping` dict, also sets attribute on `self`.
- `get_module(name, default)` — retrieves a loaded module.
- `from_pretrained(model_path, **kwargs)` — class method constructing
  `FastVideoArgs` and calling `cls(...)` then `post_init()`.

### LoRAPipeline (`fastvideo/pipelines/lora_pipeline.py`)

Extends `ComposedPipelineBase` with LoRA adapter support. Sits in the
MRO between the concrete pipeline and `ComposedPipelineBase`:

```python
class WanPipeline(LoRAPipeline, ComposedPipelineBase):
    ...
```

Key functionality:
- `convert_to_lora_layers()` — scans transformer blocks, replaces target
  linear layers (default: q/k/v/o projections) with LoRA equivalents via
  `get_lora_layer()`.
- `set_lora_adapter(path)` — loads safetensors containing `lora_A`,
  `lora_B`, `lora_alpha` and maps weights to internal layers.
- `merge_lora_weights()` / `unmerge_lora_weights()` — activates or
  deactivates LoRA in the forward pass.
- `LoRAModelLayers` — groups LoRA layers by transformer block for
  efficient layerwise offload.

### Distributed Inference

`maybe_init_distributed_environment_and_model_parallel(tp_size, sp_size)`
in `fastvideo/distributed/` initializes `torch.distributed` and creates
tensor-parallel (TP) and sequence-parallel (SP) process groups.

Key APIs: `get_tp_rank()`, `get_tp_world_size()`, `get_sp_rank()`,
`get_sp_world_size()`, `get_world_rank()`, `get_world_size()`.

`warmup_sequence_parallel_communication()` pre-warms NCCL communicators
to avoid slow first forward passes.

Usage: `torchrun --nproc-per-node=N -m fastvideo.entrypoints.cli.main
generate --model-path ... --tp-size N --sp-size M`.

### torch.compile Integration

When `fastvideo_args.enable_torch_compile` is `True`,
`_maybe_compile_pipeline_module()` checks for a `_compile_conditions`
attribute on the module. If present, only matching submodules are
compiled. Otherwise, the entire module is compiled. FSDP-wrapped
modules are skipped.

### Entry Points

**Python API** (`fastvideo/entrypoints/video_generator.py`):

```python
generator = VideoGenerator.from_pretrained(
    model_path="Wan-AI/Wan2.1-T2V-14B-Diffusers",
    num_gpus=1, tp_size=1, sp_size=1,
)
result = generator.generate_video(
    prompt="A cat dancing",
    height=720, width=1280, num_frames=81,
)
```

**CLI** (`fastvideo/entrypoints/cli/`):

```bash
fastvideo generate \
    --model-path "Wan-AI/Wan2.1-T2V-14B-Diffusers" \
    --prompt "A cat dancing" \
    --num-gpus 1
```

**FastVideoArgs** (`fastvideo/fastvideo_args.py`): Central args dataclass.
Key fields: `model_path`, `mode` (`ExecutionMode`), `workload_type`
(`WorkloadType`), `pipeline_config` (`PipelineConfig`), `num_gpus`,
`tp_size`, `sp_size`, `lora_path`, `dit_cpu_offload`,
`dit_layerwise_offload`, `enable_torch_compile`,
`enable_stage_verification`.

Constructed via `FastVideoArgs.from_kwargs(**kwargs)` which resolves the
`PipelineConfig` from the registry, applies JSON config if provided, and
merges CLI overrides.

## End-to-End Inference Flow

```
User: VideoGenerator.from_pretrained(model_path, **kwargs)
  │
  ├─ FastVideoArgs.from_kwargs() → PipelineConfig resolved via registry
  ├─ get_model_info() → ModelInfo(pipeline_cls, sampling_param_cls, ...)
  │   ├─ model_index.json read → _class_name extracted
  │   ├─ pipeline_registry resolves pipeline_cls from _class_name
  │   └─ config_registry resolves config classes from model_path
  │
  ├─ pipeline_cls.__init__(model_path, fastvideo_args)
  │   ├─ maybe_init_distributed(tp_size, sp_size)
  │   └─ load_modules() → reads model_index.json, loads each component
  │       ├─ ComponentLoader.for_module_type() → specialized loader
  │       └─ loader.load() → model class resolved, weights loaded
  │
  └─ pipeline.post_init()
      ├─ initialize_pipeline() → model-specific setup
      ├─ create_pipeline_stages() → stages wired with modules
      ├─ torch.compile (if enabled)
      └─ warmup_sequence_parallel_communication()

User: generator.generate_video(prompt, ...)
  │
  ├─ ForwardBatch constructed from SamplingParam + user args
  └─ pipeline.forward(batch, fastvideo_args)
      ├─ InputValidationStage → validates dims
      ├─ TextEncodingStage → prompt → embeddings
      ├─ ConditioningStage → prepares conditioning
      ├─ TimestepPreparationStage → scheduler timesteps
      ├─ LatentPreparationStage → random noise
      ├─ DenoisingStage → iterative denoising loop
      └─ DecodingStage → latents → video frames
```

## Adding a New Model Family — Checklist

1. **Pipeline config** — Create a `PipelineConfig` subclass in
   `fastvideo/configs/pipelines/<model>.py`. Set DiT/VAE/encoder configs,
   flow_shift, precision defaults.

2. **Sampling param** — Create a `SamplingParam` subclass in
   `fastvideo/configs/sample/<model>.py`. Set default height, width,
   num_frames, guidance_scale, num_inference_steps.

3. **Register configs** — In `fastvideo/registry.py`, add a
   `register_configs()` call inside `_register_configs()` with
   `hf_model_paths` and/or `model_detectors`.

4. **Pipeline class** — Create a subclass of `ComposedPipelineBase` (or
   `LoRAPipeline` + `ComposedPipelineBase`) in
   `fastvideo/pipelines/basic/<model>/<model>_pipeline.py`.
   - Set `_required_config_modules` listing needed components.
   - Implement `create_pipeline_stages()` wiring stages via `add_stage()`.
   - Optionally override `initialize_pipeline()` for custom setup.
   - Export `EntryClass = YourPipeline` at module level.

5. **Model classes** (if custom) — Add DiT/VAE implementations in
   `fastvideo/models/dits/` or `fastvideo/models/vaes/` with
   `EntryClass = YourModel`. The AST discovery will register them
   automatically.

6. **Custom stages** (if needed) — Subclass `PipelineStage` in
   `fastvideo/pipelines/stages/`, implement `forward()`, optionally
   implement `verify_input()`/`verify_output()`.

7. **Verify** — Run `fastvideo generate --model-path <path> --prompt
   "test" --num-inference-steps 2` to confirm the pipeline loads and
   generates output.
