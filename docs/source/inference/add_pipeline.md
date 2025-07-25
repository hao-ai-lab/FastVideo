(add-pipeline)=

# 🏗️ Adding a New Pipeline

This guide explains how to implement a custom diffusion pipeline in FastVideo, leveraging the framework's modular architecture for high-performance video generation.

## Implementation Process Overview

1. **Port Required Modules** - Identify and implement necessary model components
2. **Create Directory Structure** - Set up pipeline files and folders
3. **Implement Pipeline Class** - Build the pipeline using existing or custom stages
4. **Register Your Pipeline** - Make it discoverable by the framework
5. **Configure Your Pipeline** - (Coming soon)

Need help? Join our [Slack community](https://join.slack.com/t/fastvideo/shared_invite/zt-38u6p1jqe-yDI1QJOCEnbtkLoaI5bjZQ).

## Step 1: Pipeline Modules

### Identifying Required Modules

FastVideo uses the Hugging Face Diffusers format for model organization:

1. Examine the `model_index.json` in the HF model repository:

```json
{
    "_class_name": "WanImageToVideoPipeline",
    "_diffusers_version": "0.33.0.dev0",
    "image_encoder": ["transformers", "CLIPVisionModelWithProjection"],
    "image_processor": ["transformers", "CLIPImageProcessor"],
    "scheduler": ["diffusers", "UniPCMultistepScheduler"],
    "text_encoder": ["transformers", "UMT5EncoderModel"],
    "tokenizer": ["transformers", "T5TokenizerFast"],
    "transformer": ["diffusers", "WanTransformer3DModel"],
    "vae": ["diffusers", "AutoencoderKLWan"]
}
```

1. For each component:
   - Note the originating library (`transformers` or `diffusers`)
   - Identify the class name
   - Check if it's already available in FastVideo

2. Review config files in each component's directory for architecture details

### Implementing Modules

Place new modules in the appropriate directories:
- Encoders: `fastvideo/models/encoders/`
- VAEs: `fastvideo/models/vaes/`
- Transformer models: `fastvideo/models/dits/`
- Schedulers: `fastvideo/models/schedulers/`

### Adapting Model Layers

#### Layer Replacements
Replace standard PyTorch layers with FastVideo optimized versions:
- nn.LayerNorm → fastvideo.layers.layernorm.RMSNorm
- Embedding layers → fastvideo.layers.vocab_parallel_embedding modules
- Activation functions → versions from fastvideo.layers.activation

#### Distributed Linear Layers
Use appropriate parallel layers for distribution:

```python
# Output dimension parallelism
from fastvideo.layers.linear import ColumnParallelLinear
self.q_proj = ColumnParallelLinear(
    input_size=hidden_size,
    output_size=head_size * num_heads,
    bias=bias,
    gather_output=False
)

# Fused QKV projection
from fastvideo.layers.linear import QKVParallelLinear
self.qkv_proj = QKVParallelLinear(
    hidden_size=hidden_size,
    head_size=attention_head_dim,
    total_num_heads=num_attention_heads,
    bias=True
)

# Input dimension parallelism
from fastvideo.layers.linear import RowParallelLinear
self.out_proj = RowParallelLinear(
    input_size=head_size * num_heads,
    output_size=hidden_size,
    bias=bias,
    input_is_parallel=True
)
```

### Attention Layers
Replace standard attention with FastVideo's optimized attention:

```python
# Local attention patterns
from fastvideo.attention import LocalAttention
from fastvideo.attention.backends.abstract import _Backend
self.attn = LocalAttention(
    num_heads=num_heads,
    head_size=head_dim,
    dropout_rate=0.0,
    softmax_scale=None,
    causal=False,
    supported_attention_backends=(_Backend.FLASH_ATTN, _Backend.TORCH_SDPA)
)

# Distributed attention for long sequences
from fastvideo.attention import DistributedAttention
self.attn = DistributedAttention(
    num_heads=num_heads,
    head_size=head_dim,
    dropout_rate=0.0,
    softmax_scale=None,
    causal=False,
    supported_attention_backends=(_Backend.SLIDING_TILE_ATTN, _Backend.FLASH_ATTN, _Backend.TORCH_SDPA)
)
```

#### Define supported backend selection

```python
   _supported_attention_backends = (_Backend.FLASH_ATTN, _Backend.TORCH_SDPA)
```

### Registering Models

Register implemented modules in the model registry:

```python
# In fastvideo/models/registry.py
_TEXT_TO_VIDEO_DIT_MODELS = {
    "YourTransformerModel": ("dits", "yourmodule", "YourTransformerClass"),
}

_VAE_MODELS = {
    "YourVAEModel": ("vaes", "yourvae", "YourVAEClass"),
}
```

## Step 2: Directory Structure

Create a new directory for your pipeline:

```
fastvideo/pipelines/
├── your_pipeline/
│   ├── __init__.py
│   └── your_pipeline.py
```

## Step 3: Implement Pipeline Class

Pipelines are composed of stages, each handling a specific part of the diffusion process:

- **InputValidationStage**: Validates input parameters
- **Text Encoding Stages**: Handle text encoding (CLIP/Llama/T5)
- **CLIPImageEncodingStage**: Processes image inputs
- **TimestepPreparationStage**: Prepares diffusion timesteps
- **LatentPreparationStage**: Manages latent representations
- **ConditioningStage**: Processes conditioning inputs
- **DenoisingStage**: Performs denoising diffusion
- **DecodingStage**: Converts latents to pixels

### Creating Your Pipeline

```python
from fastvideo.pipelines.composed_pipeline_base import ComposedPipelineBase
from fastvideo.pipelines.stages import (
    InputValidationStage, CLIPTextEncodingStage, TimestepPreparationStage,
    LatentPreparationStage, DenoisingStage, DecodingStage
)
from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
import torch

class MyCustomPipeline(ComposedPipelineBase):
    """Custom diffusion pipeline implementation."""
    
    # Define required model components from model_index.json
    _required_config_modules = [
        "text_encoder", "tokenizer", "vae", "transformer", "scheduler"
    ]
    
    @property
    def required_config_modules(self) -> List[str]:
        return self._required_config_modules
        
    def initialize_pipeline(self, fastvideo_args: FastVideoArgs):
        """Initialize pipeline-specific components."""
        pass
        
    def create_pipeline_stages(self, fastvideo_args: FastVideoArgs):
        """Set up pipeline stages with proper dependency injection."""
        self.add_stage(
            stage_name="input_validation_stage",
            stage=InputValidationStage()
        )
        
        self.add_stage(
            stage_name="prompt_encoding_stage",
            stage=CLIPTextEncodingStage(
                text_encoder=self.get_module("text_encoder"),
                tokenizer=self.get_module("tokenizer")
            )
        )
        
        self.add_stage(
            stage_name="timestep_preparation_stage",
            stage=TimestepPreparationStage(
                scheduler=self.get_module("scheduler")
            )
        )
        
        self.add_stage(
            stage_name="latent_preparation_stage",
            stage=LatentPreparationStage(
                scheduler=self.get_module("scheduler"),
                vae=self.get_module("vae")
            )
        )
        
        self.add_stage(
            stage_name="denoising_stage",
            stage=DenoisingStage(
                transformer=self.get_module("transformer"),
                scheduler=self.get_module("scheduler")
            )
        )
        
        self.add_stage(
            stage_name="decoding_stage",
            stage=DecodingStage(
                vae=self.get_module("vae")
            )
        )
    
# Register the pipeline class
EntryClass = MyCustomPipeline
```

### Creating Custom Stages (Optional)

If existing stages don't meet your needs, create custom ones:

```python
from fastvideo.pipelines.stages.base import PipelineStage

class MyCustomStage(PipelineStage):
    """Custom processing stage for the pipeline."""
    
    def __init__(self, custom_module, other_param=None):
        super().__init__()
        self.custom_module = custom_module
        self.other_param = other_param
        
    def forward(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> ForwardBatch:
        # Access input data
        input_data = batch.some_attribute
        
        # Validate inputs
        if input_data is None:
            raise ValueError("Required input is missing")
            
        # Process with your module
        result = self.custom_module(input_data)
        
        # Update batch with results
        batch.some_output = result
        
        return batch
```

Add your custom stage to the pipeline:

```python
self.add_stage(
    stage_name="my_custom_stage",
    stage=MyCustomStage(
        custom_module=self.get_module("custom_module"),
        other_param="some_value"
    )
)
```

#### Stage Design Principles

1. **Single Responsibility**: Focus on one specific task
2. **Functional Pattern**: Receive and return a `ForwardBatch` object
3. **Dependency Injection**: Pass dependencies through constructor
4. **Input Validation**: Validate inputs for clear error messages

## Step 4: Register Your Pipeline

Define `EntryClass` at the end of your pipeline file:

```python
# Single pipeline class
EntryClass = MyCustomPipeline

# Or multiple pipeline classes
EntryClass = [MyCustomPipeline, MyOtherPipeline]
```

The registry will automatically:
1. Scan all packages under `fastvideo/pipelines/`
2. Look for `EntryClass` variables
3. Register pipelines using their class names as identifiers

## Best Practices

- **Reuse Existing Components**: Leverage built-in stages and modules
- **Follow Module Organization**: Place new modules in appropriate directories
- **Match Model Patterns**: Follow existing code patterns and conventions
