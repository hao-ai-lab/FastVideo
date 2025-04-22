(add-pipeline)=

# Adding a New Diffusion Pipeline


Welcome to FastVideo's pipeline contribution guide! Whether you're a first-time contributor or an experienced developer, this comprehensive walkthrough will help you implement your own diffusion pipeline in the FastVideo framework. With FastVideo's modular architecture, you can leverage existing components while adding your own innovations to create high-performance video generation pipelines.

## Overview

FastVideo provides a modular pipeline architecture that makes it easy to create custom diffusion pipelines while reusing common components. The pipeline system is built around a composition of stages, where each stage handles a specific part of the diffusion process.

This guide explains how to create and register a new pipeline within the FastVideo framework. It assumes that you've read through the design docs under (#inference-overview).

In general, adding a new pipeline will involve the following steps:

1. **Port Required Pipeline Modules**:
   - Identify which modules your pipeline needs (VAEs, encoders, DiTs, schedulers, etc.)
   - When a new diffusion pipeline is released on HuggingFace, you can
   examine the `model_index.json` file on HuggingFace Hub to see what modules it
   uses.
   - If they don't exist in FastVideo yet, implement them under the appropriate subdirectories in `fastvideo/v1/models/`
   - Often times it is much faster and easier to start with an implementation from vLLM, SGLang, or HF Transformers and replace the `nn.Module` operators with modules from `fastvideo.v1.layers`.
   - Register the new modules in the appropriate registry (in `fastvideo/v1/models/registry.py`)
   - As more pipeline modules are added to FastVideo, we hope that new pipelines will be able to reuse more modules.

2. **Create Pipeline Directory Structure**:
   - Create a new directory under `fastvideo/v1/pipelines/your_pipeline/`
   - Add `__init__.py` and `your_pipeline.py` files to this directory

3. **Implement Pipeline Class**:
   - Reuse existing stages where possible
   - Create custom stage classes inheriting from `PipelineStage` if needed
   - Inherit from `ComposedPipelineBase` for your pipeline class
   - Define `_required_config_modules` matching the `model_index.json` file
   - Implement abstract methods: `initialize_pipeline` and `create_pipeline_stages`
   - Order stages correctly in `create_pipeline_stages`

4. **Register Your Pipeline**:
   - Add `EntryClass = YourCustomPipeline` at the end of your pipeline file
   - This allows the pipeline registry to automatically discover and register your pipeline

5. **Configuring Your Pipeline**:
   - Coming soon! :)


In the following sections we will do a very detailed walkthrough of how to add a new diffusion pipeline to FastVideo! If you have any questions or get stuck, please reach out for help at our [Slack community](https://join.slack.com/t/fastvideo/shared_invite/zt-2zf6ru791-sRwI9lPIUJQq1mIeB_yjJg).


## Step 1: Pipeline Modules

This section will walk you through how to port modules needed by your new diffusion pipeline to FastVideo, allowing it to be automatically parallelized across different GPUs and use optimization features such as Sliding Tile Attention.

### Porting Required Modules 

When porting modules from other libraries to FastVideo, you'll need to understand how to adapt them to FastVideo's architecture. This typically involves replacing key components with their FastVideo counterparts.

#### Identifying which modules are needed

FastVideo expects model weight checkpoints and config files to be organized in the Hugging Face Diffusers format. This allows FastVideo to seamlessly load and optimize Diffusers-compatible models while adding distributed processing capabilities. 

:::{note}
Sometimes new model's weights uploaded to HuggingFace do not conform 
:::

One key config file at the root of all HF Diffusers weights is the `model_index.json` 

(See [Wan-2.1](https://huggingface.co/Wan-AI/Wan2.1-I2V-14B-720P-Diffusers/tree/main) as an example)

Sample `model_index.json` file:
```json
{
    "_class_name": "WanImageToVideoPipeline",
    "_diffusers_version": "0.33.0.dev0",
    "image_encoder": [
        "transformers",
        "CLIPVisionModelWithProjection"
        ],
    "image_processor": [
        "transformers",
        "CLIPImageProcessor"
        ],
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

Each subdirectory typically contains model weights, configuration files, and sometimes additional metadata specific to that component.

**How to Map Components to FastVideo:**

1. For each component in the `model_index.json`:
   - Identify the originating library (`transformers` or `diffusers`)
   - Note the class name (e.g., `CLIPVisionModelWithProjection`, `WanTransformer3DModel`)
   - This information helps determine which FastVideo modules you'll need to implement or if they are already available in FastVideo :)

2. Examine config files in each directory:
   - Most components have a `config.json` that details the model architecture
   - These configs provide essential parameters like hidden sizes, number of layers, etc.

3. Reference the component directories in the HF model to understand the specific implementation details and weight formats

Once you've identified all required components, you can start implementing them in the appropriate FastVideo directories.

**Module Organization**
- Place encoders in `fastvideo/v1/models/encoders/`
- Place VAEs in `fastvideo/v1/models/vaes/`
- Place transformer models in `fastvideo/v1/models/dits/`
- Place schedulers in `fastvideo/v1/models/schedulers/`

#### Layer Replacements (All modules)
   When adapting models from HuggingFace, vLLM, or SGLang, you'll need to replace standard layers with FastVideo optimized versions:
  
- `nn.LayerNorm` or other implementations → `fastvideo.v1.layers.layernorm.RMSNorm`
- Embedding layers → `fastvideo.v1.layers.vocab_parallel_embedding` modules
- Activation functions → versions from `fastvideo.v1.layers.activation`

#### Distributed Linear Layers (Encoder models):
   Use the appropriate distributed linear layers based on the parallelism strategy:
  
   ```python
   # For tensor parallelism along output dimension 
   from fastvideo.v1.layers.linear import ColumnParallelLinear, QKVParallelLinear, MergedColumnParallelLinear
   
   # Instead of nn.Linear for projection layers
   self.q_proj = ColumnParallelLinear(
       input_size=hidden_size,
       output_size=head_size * num_heads,
       bias=bias,
       gather_output=False
   )
   
   # For fused QKV projection (as seen in hunyuanvideo.py)
   self.qkv_proj = QKVParallelLinear(
       hidden_size=hidden_size,
       head_size=attention_head_dim,
       total_num_heads=num_attention_heads,
       bias=True
   )
   
   # For tensor parallelism along input dimension
   from fastvideo.v1.layers.linear import RowParallelLinear
   
   # Instead of nn.Linear for output projection
   self.out_proj = RowParallelLinear(
       input_size=head_size * num_heads,
       output_size=hidden_size,
       bias=bias,
       input_is_parallel=True
   )
   ```

#### Attention Layers (Encoder and DiT models):
   Replace standard attention implementations with FastVideo's attention system using either `DistributedAttention` or `LocalAttention` based on the use case:
  
   ```python
   # Original attention in most models
   attention_output = F.scaled_dot_product_attention(query, key, value)
   
   # For standard local attention patterns (as used in transformer blocks)
   from fastvideo.v1.attention import LocalAttention
   from fastvideo.v1.attention.backends.abstract import _Backend
   
   self.attn = LocalAttention(
       num_heads=num_heads,
       head_size=head_dim,
       dropout_rate=0.0,
       softmax_scale=None,
       causal=False,  # Set to True for causal attention
       supported_attention_backends=(_Backend.FLASH_ATTN, _Backend.TORCH_SDPA)
   )
   
   # For handling long 3D attention with sequence parallelism (as in hunyuanvideo.py)
   from fastvideo.v1.attention import DistributedAttention
   
   self.attn = DistributedAttention(
       num_heads=num_heads,
       head_size=head_dim,
       dropout_rate=0.0,
       softmax_scale=None,
       causal=False,
       supported_attention_backends=(_Backend.SLIDING_TILE_ATTN, _Backend.FLASH_ATTN, _Backend.TORCH_SDPA)
   )
   ```

##### Define supported backend selection (as in hunyuanvideo.py)
```python
   _supported_attention_backends = (_Backend.FLASH_ATTN, _Backend.TORCH_SDPA)
```
  
<!-- # Define FSDP shard conditions for efficient distributed training
   _fsdp_shard_conditions = [
       lambda n, m: "double" in n and str.isdigit(n.split(".")[-1]),
       lambda n, m: "single" in n and str.isdigit(n.split(".")[-1]),
       lambda n, m: "refiner" in n and str.isdigit(n.split(".")[-1]),
   ]

   ``` -->

#### Registering Models

Finally, after implementation, register your modules in the model registry:

```python
# In fastvideo/v1/models/registry.py

_TEXT_TO_VIDEO_DIT_MODELS = {
    "YourTransformerModel": ("dits", "yourmodule", "YourTransformerClass"),
}

_VAE_MODELS = {
    "YourVAEModel": ("vaes", "yourvae", "YourVAEClass"),
}
```

This registration maps model class names from Hugging Face to their FastVideo implementations, allowing the component loader to find and instantiate the appropriate class.

## Step 2: Directory Structure

Create a new directory for your new pipeline class under `fastvideo/v1/pipelines/`:

```
fastvideo/v1/pipelines/
├── your_pipeline/
│   ├── __init__.py
│   └── your_pipeline.py
```

## Step 3: Diffusion Pipeline Class

Pipeline classes are composed of Pipeline `Stages`. `Stages` are the building blocks that make up a pipeline. Each stage is responsible for a specific part of the diffusion process:

- **InputValidationStage**: Validates input parameters
- **CLIPTextEncodingStage/LlamaEncodingStage/T5EncodingStage**: Handles text encoding with different models
- **CLIPImageEncodingStage**: Processes image inputs
- **TimestepPreparationStage**: Prepares timesteps for diffusion
- **LatentPreparationStage**: Manages the latent space representation
- **ConditioningStage**: Processes conditioning inputs
- **DenoisingStage**: Performs the core denoising diffusion process
- **DecodingStage**: Converts latents back to pixel space

Pipeline stages follow a functional programming pattern, receiving a `ForwardBatch` object and returning an updated version. This approach makes stages composable and easier to test in isolation.

### Putting it all together

#### 1. Create a Pipeline Class

Your pipeline class should inherit from `ComposedPipelineBase` and implement the required abstract methods:

```python
from fastvideo.v1.pipelines.composed_pipeline_base import ComposedPipelineBase
from fastvideo.v1.pipelines.stages import (
    InputValidationStage,
    CLIPTextEncodingStage,
    TimestepPreparationStage,
    LatentPreparationStage,
    DenoisingStage,
    DecodingStage
)
from fastvideo.v1.fastvideo_args import FastVideoArgs
from fastvideo.v1.pipelines.pipeline_batch_info import ForwardBatch
import torch

class MyCustomPipeline(ComposedPipelineBase):
    """
    Custom diffusion pipeline implementation.
    """
    
    # Define required model components
    # These modules are going to be loaded by FastVideo and made available to the pipeline below in create_pipeline_stages
    _required_config_modules = [
        "text_encoder",
        "tokenizer", 
        "vae",
        "transformer",
        "scheduler"
    ]
    
    @property
    def required_config_modules(self) -> List[str]:
        return self._required_config_modules
        
    def initialize_pipeline(self, fastvideo_args: FastVideoArgs):
        """Initialize pipeline-specific components."""
        # Add any pipeline-specific initialization here
        pass
        
    def create_pipeline_stages(self, fastvideo_args: FastVideoArgs):
        """Set up pipeline stages with proper dependency injection."""
        # Add and configure stages
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
    
    @torch.no_grad()
    def forward(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> ForwardBatch:
        """
        Optional custom forward implementation if needed.
        By default, ComposedPipelineBase provides a forward method that runs all stages.
        Override only if you need custom logic between stages.
        """
        # You can add custom logic here before or after calling stages
        # The default implementation from the base class is usually sufficient
        return super().forward(batch, fastvideo_args)

# Register the pipeline class
EntryClass = MyCustomPipeline
```

#### 2. [Optional] Implementing Custom Pipeline Stages

If the existing stages don't meet your needs, you can create custom stages by inheriting from `PipelineStage`. Each stage should be focused on a specific part of the diffusion process and follow the functional pattern.

```python
from fastvideo.v1.pipelines.stages.base import PipelineStage
from fastvideo.v1.fastvideo_args import FastVideoArgs
from fastvideo.v1.pipelines.pipeline_batch_info import ForwardBatch
import torch

class MyCustomStage(PipelineStage):
    """
    Custom processing stage for the pipeline.
    
    This stage handles [describe what your stage does].
    """
    
    def __init__(self, custom_module, other_param=None):
        """
        Initialize the custom stage.
        
        Args:
            custom_module: The module to use for processing
            other_param: Additional configuration parameter
        """
        super().__init__()
        self.custom_module = custom_module
        self.other_param = other_param
        
    def forward(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> ForwardBatch:
        """
        Process the batch and update it with your stage's results.
        
        Args:
            batch: The current batch information
            fastvideo_args: The inference arguments
            
        Returns:
            The updated batch with this stage's processing applied
        """
        # Access input data from the batch
        input_data = batch.some_attribute
        
        # Validate inputs
        if input_data is None:
            raise ValueError("Required input is missing")
            
        # Process with your module
        result = self.custom_module(input_data)
        
        # Update the batch with the results
        batch.some_output = result
        
        # Return the updated batch
        return batch
```

#### Common Stage Patterns

FastVideo's pipeline stages follow several common patterns that you should adopt:

1. **Single Responsibility**: Each stage should focus on one specific aspect of the pipeline.

2. **Input/Output**: Stages receive a `ForwardBatch` object containing all pipeline state and return the same object after modification.

3. **Module Injection**: Dependencies should be injected through the constructor for better flexibility.

4. **Input Validation**: Always validate inputs to provide clear error messages.

Then add your custom stage to the pipeline:

```python
self.add_stage(
    stage_name="my_custom_stage",
    stage=MyCustomStage(
        custom_module=self.get_module("custom_module"),
        other_param="some_value"
    )
)
```

## 4. Register Your Pipeline

The pipeline registry automatically detects and loads your pipeline through the following mechanism:

1. It scans all packages under `fastvideo/v1/pipelines/`
2. For each package, it looks for an `EntryClass` variable that defines the pipeline class(es)
3. The pipeline is registered using the class name as its identifier

Simply define `EntryClass` at the end of your pipeline file:

```python
# Single pipeline class
EntryClass = MyCustomPipeline

# Or multiple pipeline classes
EntryClass = [MyCustomPipeline, MyOtherPipeline]
```

## 5. Configuring your Pipeline
Coming soon!

## Recommendations

- **Follow Module Organization**: Place new modules in appropriate directories (`encoders/`, `dits/`, `vaes/`, etc.)
- **Reuse Existing Stages**: Leverage built-in pipeline stages whenever possible to ensure consistent behavior
