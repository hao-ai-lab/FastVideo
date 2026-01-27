# 🔍 FastVideo Overview

This document outlines FastVideo's architecture for developers interested in framework internals or contributions. It serves as an onboarding guide for new contributors by providing an overview of the most important directories and files within the `fastvideo/` codebase.

## Table of Contents - Directory Structure and Files

- [`fastvideo/pipelines/`](#pipeline-system) - Core diffusion pipeline components
- [`fastvideo/models/`](#model-components) - Model implementations
  - [`dits/`](#transformer-models) - Transformer-based diffusion models
  - [`vaes/`](#vae-variational-auto-encoder) - Variational autoencoders
  - [`encoders/`](#text-and-image-encoders) - Text and image encoders
  - [`schedulers/`](#schedulers) - Diffusion schedulers
- [`fastvideo/configs/`](#configuration-system) - Pipeline, model, and sampling configs
- [`fastvideo/attention/`](#optimized-attention) - Optimized attention implementations
- [`fastvideo/distributed/`](#distributed-processing) - Distributed computing utilities
- [`fastvideo/layers/`](#tensor-parallelism) - Custom neural network layers
- [`fastvideo/platforms/`](#platforms) - Hardware platform abstractions
- [`fastvideo/worker/`](#executor-and-worker-system) - Multi-GPU process management
- [`fastvideo/fastvideo_args.py`](#fastvideoargs) - Argument handling
- [`fastvideo/forward_context.py`](#forward-context-management) - Forward pass context management
- `fastvideo/utils.py` - Utility functions
- [`fastvideo/logger.py`](#logger) - Logging infrastructure

## Core Architecture

FastVideo separates model components from execution logic with these principles:

- **Component Isolation**: Models (encoders, VAEs, transformers) are isolated from execution (pipelines, stages, distributed processing)
- **Modular Design**: Components can be independently replaced
- **Distributed Execution**: Supports various parallelism strategies (Tensor, Sequence)
- **Custom Attention Backends**: Components can support and use different Attention implementations
- **Pipeline Abstraction**: Consistent interface across diffusion models

## FastVideo structure at a glance

This section summarizes how FastVideo maps to a diffusion pipeline and where
to look when adding or extending a model.

Key pieces and how they fit:

- **Model definitions** (`fastvideo/models/…`): the actual PyTorch modules for
  transformers (DiT), VAEs, encoders, upsamplers, etc.
- **Arch config** (`fastvideo/configs/models/**/`): describes layer shapes and
  naming; where key‑rename maps live so checkpoints align with FastVideo.
- **Sampling params** (`fastvideo/configs/sample/`): runtime defaults like
  steps, guidance scale, and resolution.
- **Pipeline config** (`fastvideo/configs/pipelines/`): wiring that tells
  FastVideo which components to load and how stages are assembled.
- **Component loading** (`fastvideo/models/loader/`): reads HuggingFace /
  Diffusers‑style folders and instantiates modules based on config.
- **Diffusers structure** (`model_index.json`): the root index that maps
  component names (`transformer`, `vae`, `text_encoder`, etc.) to their classes.

Think of the flow as:
`model_index.json` → component loaders → model modules → pipeline stages →
sampling params.

Example mapping (based on `examples/inference/basic/basic.py`):

- `VideoGenerator.from_pretrained("Wan-AI/Wan2.1-T2V-1.3B-Diffusers")`
  resolves a pipeline config and sampling defaults via the registries.
- The HuggingFace repo’s `model_index.json` tells FastVideo which components to
  load (transformer, VAE, text encoder, tokenizer, etc.).
- The component loaders read those folders and instantiate the modules.
- `SamplingParam` (or defaults from `fastvideo/configs/sample/`) defines
  steps, frames, resolution, and guidance scale used during generation.

```python
from fastvideo import VideoGenerator

generator = VideoGenerator.from_pretrained(
    "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
    num_gpus=1,
)

prompt = "A curious raccoon peers through a vibrant field of yellow sunflowers..."
generator.generate_video(prompt, output_path="video_samples", save_video=True)
```

## FastVideoArgs

The `FastVideoArgs` class in `fastvideo/fastvideo_args.py` serves as the central configuration system for FastVideo. It contains all parameters needed to control model loading, inference configuration, performance optimization settings, and more.

Key features include:

- **Command-line Interface**: Automatic conversion between CLI arguments and dataclass fields
- **Configuration Groups**: Organized by functional areas (model loading, video params, optimization settings)
- **Context Management**: Global access to current settings via `get_current_fastvideo_args()`
- **Parameter Validation**: Ensures valid combinations of settings

Common configuration areas:

- **Model paths and loading options**: `model_path`, `trust_remote_code`, `revision`
- **Distributed execution settings**: `num_gpus`, `tp_size`, `sp_size`
- **Video generation parameters**: `height`, `width`, `num_frames`, `num_inference_steps`
- **Precision settings**: Control computation precision for different components

Example usage:

```python
# Load arguments from command line
fastvideo_args = prepare_fastvideo_args(sys.argv[1:])

# Access parameters
model = load_model(fastvideo_args.model_path)

# Set as global context
with set_current_fastvideo_args(fastvideo_args):
    # Code that requires access to these arguments
    result = generate_video()
```

## Configuration System

FastVideo’s configuration layer lives under `fastvideo/configs/` and provides
typed defaults and registries for models, pipelines, and runtime parameters.

- `fastvideo/configs/models/`: architecture definitions (model shapes, layer
  naming, and checkpoint key‑mapping rules).
  - `dits/`, `vaes/`, `encoders/`, `audio/`: component‑specific configs.
  - ArchConfig and its subfields are a superset of the model's config.json file.
- `fastvideo/configs/pipelines/`: pipeline wiring and which components are
  required for a given model family. Initialization parameters are passed to the pipeline via `fastvideo_args`.
- `fastvideo/configs/sample/`: generation defaults, can be overridden by the user and different across generations (frames, steps, guidance scale,
  resolution, fps).
- `fastvideo/configs/registry.py`: pipeline registry and routing rules.
- `fastvideo/configs/sample/registry.py`: sampling param registry.

## Weights and Diffusers format

FastVideo follows the standard HuggingFace Diffusers layout for model weights.
This keeps our loaders compatible with HF repos and makes it easy to add new
components.

Typical Diffusers repo structure:

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
- Weights are usually in `diffusion_pytorch_model.safetensors`, but FastVideo
  also accepts `model.safetensors` for custom components (e.g., upsamplers).
- Some pipelines include extra components (audio VAE, vocoder, image encoder).

Note on tensor names:

Official checkpoints often use different `state_dict` naming than FastVideo’s
module layout. We translate tensor names via the DiT arch config mapping
(under `fastvideo/configs/models/dits/`) so weights load cleanly into FastVideo
modules. This is the same kind of name‑translation layer used in systems like
vLLM and SGLang when their internal module structure differs from upstream
checkpoints.

Let’s go through a concrete example.

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

- `WanPipeline` → `fastvideo/pipelines/basic/wan/wan_pipeline.py`
- `WanTransformer3DModel` → `fastvideo/models/dits/wanvideo.py`
- `AutoencoderKLWan` → `fastvideo/models/vaes/wanvae.py`
- `UMT5EncoderModel` → `fastvideo/models/encoders/t5.py`
- `T5TokenizerFast` → loaded via HuggingFace in `fastvideo/models/loader/`
- `UniPCMultistepScheduler` → loaded via diffusers scheduler utilities
- Pipeline defaults → `fastvideo/configs/pipelines/wan.py`
- Sampling defaults → `fastvideo/configs/sample/wan.py`

!!! note
    The model loader logic is implemented in `fastvideo/models/loader/component_loader.py` and will parse the `model_index.json` file and load the appropriate components.

## Pipeline System

### `ComposedPipelineBase`

This foundational class provides:

- **Model Loading**: Automatically loads components from HuggingFace-Diffusers-compatible model directories
- **Stage Management**: Creates and orchestrates processing stages
- **Data Flow Coordination**: Ensures proper state flow between stages

```python
class MyCustomPipeline(ComposedPipelineBase):
    _required_config_modules = [
        "text_encoder", "tokenizer", "vae", "transformer", "scheduler"
    ]
    
    def initialize_pipeline(self, fastvideo_args: FastVideoArgs):
        # Pipeline-specific initialization
        pass
        
    def create_pipeline_stages(self, fastvideo_args: FastVideoArgs):
        self.add_stage("input_validation_stage", InputValidationStage())
        self.add_stage("text_encoding_stage", CLIPTextEncodingStage(
            text_encoder=self.get_module("text_encoder"),
            tokenizer=self.get_module("tokenizer")
        ))
        # Additional stages...
```

### Pipeline Stages

Each stage handles a specific diffusion process component:

- **Input Validation**: Parameter verification
- **Text Encoding**: CLIP, LLaMA, or T5-based encoding
- **Image Encoding**: Image input processing
- **Timestep & Latent Preparation**: Setup for diffusion
- **Denoising**: Core diffusion loop
- **Decoding**: Latent-to-pixel conversion

Each stage implements a standard interface:

```python
def forward(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> ForwardBatch:
    # Process batch and update state
    return batch
```

![Pipeline execution and data flow](../assets/images/pipeline.png)

### ForwardBatch

Defined in `fastvideo/pipelines/pipeline_batch_info.py`, `ForwardBatch` encapsulates the data payload passed between pipeline stages. It typically holds:

- **Input Data**: Prompts, images, generation parameters
- **Intermediate State**: Embeddings, latents, timesteps, accumulated during stage execution
- **Output Storage**: Generated results and metadata
- **Configuration**: Sampling parameters, precision settings

This structure facilitates clear state transitions between stages.

## Model Components

The `fastvideo/models/` directory contains implementations of the core neural network models used in video diffusion:

### Transformer Models

Transformer networks perform the actual denoising during diffusion:

- **Location**: `fastvideo/models/dits/`
- **Examples**:
  - `WanTransformer3DModel`
  - `HunyuanVideoTransformer3DModel`

Features include:

- Text/image conditioning
- Standardized interface for model-specific optimizations

```python
def forward(
    self, 
    latents,                    # [B, T, C, H, W]
    encoder_hidden_states,      # Text embeddings
    timestep,                   # Current diffusion timestep
    encoder_hidden_states_image=None,  # Optional image embeddings
    **kwargs
):
    # Perform denoising computation
    return noise_pred  # Predicted noise residual
```

### VAE (Variational Auto-Encoder)

VAEs handle conversion between pixel space and latent space:

- **Location**: `fastvideo/models/vaes/`
- **Examples**:
  - `AutoencoderKLWan`
  - `AutoencoderKLHunyuanVideo`

These models compress image/video data to a more efficient latent representation (typically 4x-8x smaller in each dimension).

FastVideo's VAE implementations include:

- Efficient video batch processing
- Memory optimization
- Optional tiling for large frames
- Distributed weight support

### Text and Image Encoders

Encoders process conditioning inputs into embeddings:

- **Location**: `fastvideo/models/encoders/`
- **Text Encoders**:
  - `CLIPTextModel`
  - `LlamaModel`
  - `UMT5EncoderModel`
- **Image Encoders**:
  - `CLIPVisionModel`

Encoders are used to encode text and image conditioning inputs into embeddings. They are used in the text encoding and image encoding stages of the pipeline and are added as conditionings to the DiT model.

### Schedulers

Schedulers manage the diffusion sampling process:

- **Location**: `fastvideo/models/schedulers/`
- **Examples**:
  - `UniPCMultistepScheduler`
  - `FlowMatchEulerDiscreteScheduler`

These components control:

- Diffusion timestep sequences
- Noise prediction to latent update conversions
- Quality/speed trade-offs

```python
def step(
    self, 
    model_output: torch.Tensor,
    timestep: torch.LongTensor,
    sample: torch.Tensor,
    **kwargs
) -> torch.Tensor:
    # Process model output and update latents
    # Return updated latents
    return prev_sample
```

This diagram shows how models are discovered, validated, and loaded across entrypoints, executors, pipelines, and model loaders.

![Model loading flow](../assets/images/load_models.png)

## Optimized Attention

The `fastvideo/attention/` directory contains optimized attention implementations crucial for efficient video diffusion:

### Attention Backends

Multiple implementations with automatic selection:

- **FLASH_ATTN**: Optimized for supporting hardware
- **TORCH_SDPA**: Built-in PyTorch scaled dot-product attention
- **SLIDING_TILE_ATTN**: For very long sequences

```python
# Configure available attention backends for this layer
self.attn = LocalAttention(
    num_heads=num_heads,
    head_size=head_dim,
    causal=False,
    supported_attention_backends=(_Backend.FLASH_ATTN, _Backend.TORCH_SDPA)
)

# Override via environment variable
# export FASTVIDEO_ATTENTION_BACKEND=FLASH_ATTN
```

![Attention backend selector design](../assets/images/attention_backend.png)

### Attention Patterns

Supports various patterns with memory optimization techniques:

- **Cross/Self/Temporal/Global-Local Attention**
- Chunking, progressive computation, optimized masking

## Distributed Processing

The `fastvideo/distributed/` directory contains implementations for distributed model execution:

### Tensor Parallelism

Tensor parallelism splits model weights across devices:

- **Implementation**: Through `RowParallelLinear` and `ColumnParallelLinear` layers
- **Use cases**: Will be used by encoder models as their sequence lengths are shorter and enables efficient sharding.

```python
# Tensor-parallel layers in a transformer block
from fastvideo.layers.linear import ColumnParallelLinear, RowParallelLinear

# Split along output dimension
self.qkv_proj = ColumnParallelLinear(
    input_size=hidden_size,
    output_size=3 * hidden_size,
    bias=True,
    gather_output=False
)

# Split along input dimension
self.out_proj = RowParallelLinear(
    input_size=hidden_size,
    output_size=hidden_size,
    bias=True,
    input_is_parallel=True
)
```

### Sequence Parallelism

Sequence parallelism splits sequences across devices:

- **Implementation**: Through `DistributedAttention` and sequence splitting
- **Use cases**: Long video sequences or high-resolution processing. Used by DiT models.

```python
# Distributed attention for long sequences
from fastvideo.attention import DistributedAttention

self.attn = DistributedAttention(
    num_heads=num_heads,
    head_size=head_dim,
    causal=False,
    supported_attention_backends=(_Backend.SLIDING_TILE_ATTN, _Backend.FLASH_ATTN)
)
```

### Communication Primitives

Efficient distributed operations via AllGather, AllReduce, and synchronization mechanisms.

Efficient communication primitives minimize distributed overhead:

- **Sequence-Parallel AllGather**: Collects sequence chunks
- **Tensor-Parallel AllReduce**: Combines partial results
- **Distributed Synchronization**: Coordinates execution

## Forward Context Management

### ForwardContext

Defined in `fastvideo/forward_context.py`, `ForwardContext` manages execution-specific state *within* a forward pass, particularly for low-level optimizations. It is accessed via `get_forward_context()`.

- **Attention Metadata**: Configuration for optimized attention kernels (`attn_metadata`)
- **Profiling Data**: Potential hooks for performance metrics collection

This context-based approach enables:

- Dynamic optimization based on execution state (e.g., attention backend selection)
- Step-specific customizations within model components

Usage example:

```python
with set_forward_context(current_timestep, attn_metadata, fastvideo_args):
    # During this forward pass, components can access context
    # through get_forward_context()
    output = model(inputs)
```

## Executor and Worker System

The `fastvideo/worker/` directory contains the distributed execution framework:

### Executor Abstraction

FastVideo implements a flexible execution model for distributed processing:

- **Executor Base Class**: An abstract base class defining the interface for all executors
- **MultiProcExecutor**: Primary implementation that spawns and manages worker processes
- **GPU Workers**: Handle actual model execution on individual GPUs

The MultiProcExecutor implementation:

1. Spawns worker processes for each GPU
2. Establishes communication channels via pipes
3. Coordinates distributed operations across workers
4. Handles graceful startup and shutdown of the process group

Each GPU worker:

1. Initializes the distributed environment
2. Builds the pipeline for the specified model
3. Executes requested operations on its assigned GPU
4. Manages local resources and communicates results back to the executor

This design allows FastVideo to efficiently utilize multiple GPUs while providing a simple, unified interface for model execution.

## Platforms

The `fastvideo/platforms/` directory provides hardware platform abstractions that enable FastVideo to run efficiently on different hardware configurations:

### Platform Abstraction

FastVideo's platform abstraction layer enables:

- **Hardware Detection**: Automatic detection of available hardware
- **Backend Selection**: Appropriate selection of compute kernels
- **Memory Management**: Efficient utilization of hardware-specific memory features

The primary components include:

- **Platform Interface**: Defines the common API for all platform implementations
- **CUDA Platform**: Optimized implementation for NVIDIA GPUs
- **Backend Enum**: Used throughout the codebase for feature selection

Usage example:

```python
from fastvideo.platforms import current_platform, _Backend

# Check hardware capabilities
if current_platform.supports_backend(_Backend.FLASH_ATTN):
    # Use FlashAttention implementation
else:
    # Fall back to standard implementation
```

The platform system is designed to be extensible for future hardware targets.

## Logger

See [PR](https://github.com/hao-ai-lab/FastVideo/pull/356)

*TODO*: (help wanted) Add an environment variable that disables process-aware logging.

## Contributing to FastVideo

If you're a new contributor, here are some common areas to explore:

1. **Adding a new model**: Implement new model types in the appropriate subdirectory of `fastvideo/models/`
2. **Optimizing performance**: Look at attention implementations or memory management
3. **Adding a new pipeline**: Create a new pipeline subclass in `fastvideo/pipelines/`
4. **Hardware support**: Extend the `platforms` module for new hardware targets

When adding code, follow these practices:

- Use type hints for better code readability
- Add appropriate docstrings
- Maintain the separation between model components and execution logic
- Follow existing patterns for distributed processing
