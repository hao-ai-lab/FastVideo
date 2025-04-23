# FastVideo Architecture

This document outlines FastVideo's architecture for developers interested in framework internals or contributions.

## Table of Contents - V1 Architecture

- [`fastvideo/v1/pipelines/`](#design-pipeline-system)
  - [`pipeline_batch_info.py`](#design-forwardbatch)
- [`fastvideo/v1/models/`](#design-model-components)
  - [`dits/`](#design-transformer-models) (Transformer Models)
  - [`vaes/`](#design-vae-variational-auto-encoder)
  - [`encoders/`](#design-text-and-image-encoders)
  - [`schedulers/`](#design-schedulers)
- [`fastvideo/v1/attention/`](#design-optimized-attention)
- [`fastvideo/v1/distributed/`](#design-distributed-processing)
- [`fastvideo/v1/layers/`](#design-tensor-parallelism)
- [`fastvideo/v1/forward_context.py`](#design-forwardcontext)
- [`fastvideo/v1/worker/`](#design-executor-and-worker-abstractions)

## Core Architecture

FastVideo separates model components from execution logic with these principles:
- **Component Isolation**: Models (encoders, VAEs, transformers) are isolated from execution (pipelines, stages, distributed processing)
- **Modular Design**: Components can be independently replaced
- **Distributed Execution**: Supports various parallelism strategies (Tensor, Sequence)
- **Custom Attention Backends**: Components can support and use different Attention implementations
- **Pipeline Abstraction**: Consistent interface across diffusion models

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

Defined in `fastvideo/v1/pipelines/pipeline_batch_info.py`, `ForwardBatch` encapsulates the data payload passed between pipeline stages. It typically holds:

- **Input Data**: Prompts, images, generation parameters
- **Intermediate State**: Embeddings, latents, timesteps, accumulated during stage execution
- **Output Storage**: Generated results and metadata
- **Configuration**: Sampling parameters, precision settings

This structure facilitates clear state transitions between stages.

(design-model-components)=
## Model Components

(design-transformer-models)=
### Transformer Models

Transformer networks perform the actual denoising during diffusion:

- **Location**: `fastvideo/v1/models/dits/`
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

- **Location**: `fastvideo/v1/models/vaes/`
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

- **Location**: `fastvideo/v1/models/encoders/`
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

- **Location**: `fastvideo/v1/models/schedulers/`
- **Examples**:
  - `UniPCMultistepScheduler`
  - `FlowMatchEulerDiscreteScheduler`

These components control:
- Diffusion timestep sequences
- Noise prediction to latent update conversions
- Quality/speed trade-offs

```python
def forward(
    self, 
    hidden_states: torch.Tensor,
    encoder_hidden_states: Union[torch.Tensor, List[torch.Tensor]],  # Text embeddings
    timestep: torch.LongTensor,                   # Current diffusion timestep
    encoder_hidden_states_image: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,  # Optional image embeddings
    guidance=None,
    **kwargs
) -> torch.Tensor:
    # Perform denoising computation
    return noise_pred  # Predicted noise residual
```

## Optimized Attention

(design-optimized-attention)=
## Optimized Attention

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

## Distributed Processing

(design-distributed-processing)=
## Distributed Processing

(design-tensor-parallelism)=
### Tensor Parallelism

Tensor parallelism splits model weights across devices:

- **Implementation**: Through `RowParallelLinear` and `ColumnParallelLinear` layers
- **Use cases**: Large models that exceed single-GPU memory

```python
# Tensor-parallel layers in a transformer block
from fastvideo.v1.layers.linear import ColumnParallelLinear, RowParallelLinear

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
- **Use cases**: Long video sequences or high-resolution processing

```python
# Distributed attention for long sequences
from fastvideo.v1.attention import DistributedAttention

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

## ForwardBatch and State Management

### ForwardBatch

Defined in `fastvideo/v1/pipelines/pipeline_batch_info.py`, `ForwardBatch` encapsulates the data payload passed between pipeline stages. It typically holds:

- **Input Data**: Prompts, images, generation parameters
- **Intermediate State**: Embeddings, latents, timesteps, accumulated during stage execution
- **Output Storage**: Generated results and metadata
- **Configuration**: Sampling parameters, precision settings

This structure facilitates clear state transitions between stages.

(design-forwardcontext)=
### ForwardContext

Defined in `fastvideo/v1/forward_context.py`, `ForwardContext` manages execution-specific state *within* a forward pass, particularly for low-level optimizations. It is accessed via `get_forward_context()`.

- **Attention Metadata**: Configuration for optimized attention kernels (`attn_metadata`)
- **Profiling Data**: Potential hooks for performance metrics collection

This context-based approach enables:
- Dynamic optimization based on execution state (e.g., attention backend selection)
- Step-specific customizations within model components

(design-executor-and-worker-abstractions)=
## Executor and Worker Abstractions

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

## Implementation Best Practices

### Design Benefits and Framework Comparisons

FastVideo's architecture offers several advantages through its design decisions:

#### Component Separation
Unlike frameworks that tightly couple model components with execution logic, FastVideo explicitly separates:
- Model definitions (encoders, VAEs, transformers)
- Execution orchestration (pipelines, stages, distributed strategies)
- Attention implementations (backend selection, optimization)

This separation provides concrete benefits:
- Components can be independently optimized or replaced
- Execution strategies can be changed without modifying models
- Development can progress in parallel across different system areas

#### Pipeline Stage Abstraction
The stage-based pipeline design differs from monolithic approaches seen in frameworks like HuggingFace Diffusers:

- **FastVideo**: Explicit stage boundaries with clear interfaces (`InputValidationStage`, `CLIPTextEncodingStage`, etc.)
- **HF Diffusers**: Typically implements pipeline logic within a single forward method

This staging approach enables:
- Targeted performance profiling and optimization per stage
- Better memory management through stage-specific tensor lifecycle
- Easier addition of new capabilities by inserting or modifying specific stages

#### Execution Framework
FastVideo's worker abstraction and distributed processing approach allows for:

- Explicit control over device placement and synchronization
- Multiple parallelism strategies (tensor, sequence) depending on workload characteristics
- Hardware-specific optimizations without changing core model code

In contrast, frameworks like xDIT and HF Diffusers often rely on PyTorch's built-in distributed capabilities, which may not be optimized for video-specific workloads with high memory requirements.

#### Attention Implementation
The flexible attention backend system with auto-selection logic provides:

- Automatic optimization based on available hardware capabilities
- Support for specialized kernels like FlashAttention when beneficial
- Fallback mechanisms for broader hardware compatibility

This approach balances the speed benefits of specialized implementations with the flexibility to run across different environments.

#### State Management
The separation between `ForwardBatch` (inter-stage communication) and `ForwardContext` (intra-stage optimization) helps manage complexity:

- Cleanly separates the "what" (data being processed) from the "how" (execution details)
- Provides explicit hooks for stage-specific optimizations
- Facilitates debugging by making state transitions visible

This is particularly beneficial for video diffusion, where managing large tensors efficiently is critical for performance.

### Adding New Pipelines

Follow the pattern outlined in our [pipeline contribution guide](../contributing/add_pipeline.md):

### Performance Optimization
- **Distributed Strategies**: Choose parallelism based on video characteristics
- **Memory Management**: Use activation checkpointing, lifecycle management, mixed precision
- **Attention Optimization**: Select backends and techniques based on workload

### Extension Points
FastVideo supports extensibility through:

1. **Custom Models**: Implement and register following framework patterns

2. **Custom Stages**:

```python
class MyCustomStage(PipelineStage):
    def __init__(self, custom_module):
        super().__init__()
        self.custom_module = custom_module
        
    def forward(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> ForwardBatch:
        result = self.custom_module(batch.input_data)
        batch.output_data = result
        return batch
```

1. **Custom Attention**: Subclass from base classes, implement required methods, specify backends

## Conclusion
FastVideo's architecture provides a powerful, extensible framework for video diffusion that balances performance and usability. Its modular design enables continuous improvement while maintaining a consistent interface.
