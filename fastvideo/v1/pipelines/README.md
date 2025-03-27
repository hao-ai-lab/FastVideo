# Adding a New Custom Pipeline

This guide explains how to add a new custom pipeline to the FastVideo framework. The pipeline system is designed to be modular and extensible, allowing you to implement custom video generation pipelines while reusing common components.

## Directory Structure

Create a new directory for your pipeline under `fastvideo/v1/pipelines/`:

```
fastvideo/v1/pipelines/
├── your_pipeline/
│   ├── __init__.py
│   └── your_pipeline.py
```

## Implementation Steps

1. **Create Pipeline Class**
   - Your pipeline class should inherit from `ComposedPipelineBase`
   - Implement required methods and define pipeline stages

2. **Define EntryClass**
   - At the end of your pipeline file, define `EntryClass` to expose your pipeline
   - This is how the pipeline registry detects and loads your implementation

3. **Required Methods**
   - `required_config_modules()`: List required model components
   - `create_pipeline_stages()`: Define and configure pipeline stages
   - `initialize_pipeline()`: Set up any pipeline-specific initialization
   - `forward()`: Implement the main pipeline execution flow

## Example Implementation

Here's a basic template for implementing a new pipeline:

```python
from fastvideo.v1.pipelines.composed_pipeline_base import ComposedPipelineBase
from fastvideo.v1.pipelines.stages import (
    InputValidationStage,
    ConditioningStage,
    # Import other required stages
)
from fastvideo.v1.inference_args import InferenceArgs
from fastvideo.v1.pipelines.pipeline_batch_info import ForwardBatch

class YourCustomPipeline(ComposedPipelineBase):
    def required_config_modules(self):
        return [
            "text_encoder",
            "vae",
            "transformer",
            "scheduler"
            # Add other required modules
        ]

    def create_pipeline_stages(self, inference_args: InferenceArgs):
        # Add and configure pipeline stages
        self.add_stage(
            stage_name="input_validation_stage",
            stage=InputValidationStage()
        )
        # Add more stages as needed

    def initialize_pipeline(self, inference_args: InferenceArgs):
        # Initialize pipeline-specific components
        pass

    @torch.no_grad()
    def forward(self, batch: ForwardBatch, inference_args: InferenceArgs) -> ForwardBatch:
        # Implement your pipeline's forward pass
        batch = self.input_validation_stage(batch, inference_args)
        # Add more stage executions
        return batch

# This is required for pipeline registry detection
EntryClass = YourCustomPipeline
```

## Pipeline Registry

The pipeline registry automatically detects and loads your pipeline through the following mechanism:

1. It scans all packages under `fastvideo/v1/pipelines/`
2. For each package, it looks for an `EntryClass` variable
3. The `EntryClass` can be either:
   - A single pipeline class
   - A list of pipeline classes (for multiple implementations in one module)
4. The registry uses the class name as the pipeline architecture identifier

## Available Stages

You can use these pre-built stages in your pipeline:

- `InputValidationStage`: Validates input parameters
- `TimestepPreparationStage`: Prepares timesteps for diffusion
- `LatentPreparationStage`: Prepares latent space
- `ConditioningStage`: Handles conditioning inputs
- `DenoisingStage`: Performs the denoising process
- `DecodingStage`: Decodes the final output
- `LlamaEncodingStage`: Text encoding with LLaMA
- `CLIPTextEncodingStage`: Text encoding with CLIP

## Best Practices

1. **Stage Organization**
   - Organize stages in a logical order
   - Use clear, descriptive stage names
   - Document any custom stage logic

2. **Error Handling**
   - Implement proper error handling in each stage
   - Use the logger for debugging and monitoring

3. **Configuration**
   - Clearly specify required modules in `required_config_modules()`
   - Document any pipeline-specific configuration parameters

4. **Testing**
   - Add unit tests for your pipeline
   - Test with different input configurations
   - Verify pipeline outputs

## Example Usage

After implementing your pipeline, you can use it like this:

```python
from fastvideo.v1.pipelines import PipelineRegistry

# Get your pipeline class
pipeline_cls, _ = PipelineRegistry.resolve_pipeline_cls("YourCustomPipeline")

# Initialize and use the pipeline
pipeline = pipeline_cls(...)
result = pipeline(...)
```