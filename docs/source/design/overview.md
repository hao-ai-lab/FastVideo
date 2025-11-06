# 🔍 FastVideo Overview

This document outlines FastVideo's architecture for developers interested in framework internals or contributions. It serves as an onboarding guide for new contributors by providing an overview of the most important directories and files within the `fastvideo/` codebase.

## Table of Contents - Directory Structure and Files

- [`fastvideo/pipelines/`](#design-pipeline-system) - Core diffusion pipeline components
- [`fastvideo/models/`](#design-model-components) - Model implementations
  - [`dits/`](#design-transformer-models) - Transformer-based diffusion models
  - [`vaes/`](#design-vae-variational-auto-encoder) - Variational autoencoders
  - [`encoders/`](#design-text-and-image-encoders) - Text and image encoders
  - [`schedulers/`](#design-schedulers) - Diffusion schedulers
- [`fastvideo/attention/`](#design-optimized-attention) - Optimized attention implementations
- [`fastvideo/distributed/`](#design-distributed-processing) - Distributed computing utilities
- [`fastvideo/layers/`](#design-tensor-parallelism) - Custom neural network layers
- [`fastvideo/platforms/`](#design-platforms) - Hardware platform abstractions
- [`fastvideo/worker/`](#design-executor-and-worker-abstractions) - Multi-GPU process management
- [`fastvideo/fastvideo_args.py`](#design-fastvideo-args) - Argument handling
- [`fastvideo/forward_context.py`](#design-forwardcontext) - Forward pass context management
- `fastvideo/utils.py` - Utility functions
- [`fastvideo/logger.py`](#design-logger) - Logging infrastructure

## Core Architecture

FastVideo separates model components from execution logic with these principles:
- **Component Isolation**: Models (encoders, VAEs, transformers) are isolated from execution (pipelines, stages, distributed processing)
- **Modular Design**: Components can be independently replaced
- **Distributed Execution**: Supports various parallelism strategies (Tensor, Sequence)
- **Custom Attention Backends**: Components can support and use different Attention implementations
- **Pipeline Abstraction**: Consistent interface across diffusion models

(design-fastvideo-args)=
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

(design-pipeline-system)=
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

(design-forwardbatch)=
### ForwardBatch

Defined in `fastvideo/pipelines/pipeline_batch_info.py`, `ForwardBatch` encapsulates the data payload passed between pipeline stages. It typically holds:

- **Input Data**: Prompts, images, generation parameters
- **Intermediate State**: Embeddings, latents, timesteps, accumulated during stage execution
- **Output Storage**: Generated results and metadata
- **Configuration**: Sampling parameters, precision settings

This structure facilitates clear state transitions between stages.

(design-model-components)=
## Model Components

The `fastvideo/models/` directory contains implementations of the core neural network models used in video diffusion:

(design-transformer-models)=
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

(design-vae-variational-auto-encoder)=
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

(design-text-and-image-encoders)=
### Text and Image Encoders

Encoders process conditioning inputs into embeddings:

- **Location**: `fastvideo/models/encoders/`
- **Text Encoders**:
  - `CLIPTextModel`
  - `LlamaModel`
  - `UMT5EncoderModel`
- **Image Encoders**:
  - `CLIPVisionModel`

FastVideo implements optimizations such as:
- Vocab parallelism for distributed processing
- Caching for common prompts
- Precision-tuned computation

(design-schedulers)=
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

(design-optimized-attention)=
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

### Attention Patterns
Supports various patterns with memory optimization techniques:
- **Cross/Self/Temporal/Global-Local Attention**
- Chunking, progressive computation, optimized masking

(design-distributed-processing)=
## Distributed Processing

The `fastvideo/distributed/` directory contains implementations for distributed model execution:

(design-tensor-parallelism)=
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

(design-forwardcontext)=
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

(design-executor-and-worker-abstractions)=
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

(design-platforms)=
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

(design-logger)=
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
