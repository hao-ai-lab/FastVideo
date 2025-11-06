# Pipeline Execution Walkthrough

This document explains how FastVideo pipelines are executed, from CLI entry points to the actual stage-by-stage execution.

## Table of Contents
1. [Execution Flow Overview](#execution-flow-overview)
2. [Entry Points](#entry-points)
3. [VideoGenerator: The Main Interface](#videogenerator-the-main-interface)
4. [Worker Architecture](#worker-architecture)
5. [Pipeline Composition](#pipeline-composition)
6. [Stage Execution](#stage-execution)
7. [How Adding a Stage Changes Functionality](#how-adding-a-stage-changes-functionality)

---

## Execution Flow Overview

The execution flow follows this path:

```
CLI Command (fastvideo generate)
    ↓
GenerateSubcommand.cmd()
    ↓
VideoGenerator.from_pretrained()
    ↓
VideoGenerator.generate_video()
    ↓
Executor.execute_forward() [multiproc or ray]
    ↓
Worker.execute_forward() [on each GPU]
    ↓
Pipeline.forward()
    ↓
Stage1 → Stage2 → Stage3 → ... → StageN
```

---

## Entry Points

### 1. CLI Entry Point

**File**: `fastvideo/entrypoints/cli/generate.py`

When you run:
```bash
fastvideo generate --model-path "path/to/model" --prompt "A cat playing"
```

The `GenerateSubcommand.cmd()` method is invoked:

```python
def cmd(self, args: argparse.Namespace) -> None:
    # Parse arguments
    init_args = {...}  # num_gpus, tp_size, sp_size, model_path
    generation_args = {...}  # prompt, num_frames, height, width, etc.
    
    # Create VideoGenerator
    generator = VideoGenerator.from_pretrained(
        model_path=model_path,
        **init_args
    )
    
    # Generate video
    generator.generate_video(prompt=prompt, **generation_args)
```

### 2. Python API Entry Point

You can also use it directly in Python:

```python
from fastvideo import VideoGenerator

generator = VideoGenerator.from_pretrained(
    model_path="FastVideo/Wan2.1-T2V-14B-Diffusers",
    num_gpus=2,
    tp_size=2,
)

output = generator.generate_video(
    prompt="A cat playing with a ball",
    num_frames=69,
    height=768,
    width=1280,
)
```

---

## VideoGenerator: The Main Interface

**File**: `fastvideo/entrypoints/video_generator.py`

The `VideoGenerator` is the user-facing API. It handles:
- Model loading and configuration
- Batch preparation
- Coordinating execution across multiple GPUs
- Video saving and output formatting

### Key Methods

#### `from_pretrained()` - Initialization

```python
@classmethod
def from_pretrained(cls, model_path: str, **kwargs) -> "VideoGenerator":
    # 1. Create FastVideoArgs from kwargs
    fastvideo_args = FastVideoArgs.from_kwargs(model_path=model_path, **kwargs)
    
    # 2. Determine executor type (multiproc or ray)
    executor_class = Executor.get_class(fastvideo_args)
    
    # 3. Create VideoGenerator with executor
    return cls(
        fastvideo_args=fastvideo_args,
        executor_class=executor_class,
        log_stats=False,
    )
```

#### `generate_video()` - Main Generation Method

```python
def generate_video(self, prompt: str, **kwargs) -> dict:
    # 1. Create sampling parameters
    sampling_param = SamplingParam.from_pretrained(model_path)
    sampling_param.update(kwargs)
    
    # 2. Validate and adjust dimensions
    # (e.g., ensure num_frames is divisible by num_gpus)
    
    # 3. Prepare ForwardBatch
    batch = ForwardBatch(
        prompt=prompt,
        height=height,
        width=width,
        num_frames=num_frames,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        seed=seed,
        # ... many other parameters
    )
    
    # 4. Execute on workers
    output_batch = self.executor.execute_forward(batch, fastvideo_args)
    
    # 5. Process and save output
    samples = output_batch.output
    frames = self._convert_to_frames(samples)
    
    if save_video:
        self._save_video(frames, output_path)
    
    return {
        "samples": samples,
        "frames": frames,
        "generation_time": gen_time,
        ...
    }
```

---

## Worker Architecture

FastVideo uses a distributed worker architecture to support multi-GPU execution.

### Executor Classes

**File**: `fastvideo/worker/executor.py`

The `Executor` is an abstract base class with two implementations:

1. **MultiprocExecutor** - Uses Python multiprocessing for local multi-GPU
2. **RayDistributedExecutor** - Uses Ray for distributed execution

```python
class Executor(ABC):
    def execute_forward(self, forward_batch: ForwardBatch, fastvideo_args: FastVideoArgs):
        # Sends RPC call to all workers
        outputs = self.collective_rpc("execute_forward", kwargs={
            "forward_batch": forward_batch,
            "fastvideo_args": fastvideo_args
        })
        return outputs[0]["output_batch"]
```

### Worker Class

**File**: `fastvideo/worker/gpu_worker.py`

Each GPU has a `Worker` instance that:
1. Initializes the device and distributed environment
2. Builds the pipeline from model config
3. Executes forward passes

```python
class Worker:
    def __init__(self, fastvideo_args, local_rank, rank, distributed_init_method):
        self.fastvideo_args = fastvideo_args
        self.local_rank = local_rank
        self.rank = rank
        
    def init_device(self) -> None:
        # 1. Set up CUDA device
        self.device = get_local_torch_device()
        
        # 2. Initialize distributed environment (NCCL, etc.)
        maybe_init_distributed_environment_and_model_parallel(
            self.fastvideo_args.tp_size,
            self.fastvideo_args.sp_size,
            self.distributed_init_method
        )
        
        # 3. Build the pipeline
        self.pipeline = build_pipeline(self.fastvideo_args)
    
    def execute_forward(self, forward_batch: ForwardBatch, fastvideo_args: FastVideoArgs):
        # Execute the pipeline
        output_batch = self.pipeline.forward(forward_batch, self.fastvideo_args)
        return output_batch
```

### Pipeline Building

**File**: `fastvideo/pipelines/__init__.py`

The `build_pipeline()` function:
1. Downloads model if needed
2. Reads `model_index.json` to get `_class_name`
3. Looks up pipeline class in registry
4. Instantiates and initializes the pipeline

```python
def build_pipeline(fastvideo_args: FastVideoArgs) -> PipelineWithLoRA:
    # 1. Get model path
    model_path = maybe_download_model(fastvideo_args.model_path)
    
    # 2. Read model_index.json
    config = verify_model_config_and_directory(model_path)
    pipeline_name = config.get("_class_name")  # e.g., "WanVideoPipeline"
    
    # 3. Get pipeline class from registry
    pipeline_registry = get_pipeline_registry(pipeline_type)
    pipeline_cls = pipeline_registry.resolve_pipeline_cls(
        pipeline_name,
        pipeline_type,
        fastvideo_args.workload_type
    )
    
    # 4. Instantiate pipeline
    pipeline = pipeline_cls(model_path, fastvideo_args)
    pipeline.post_init()
    
    return pipeline
```

---

## Pipeline Composition

Pipelines in FastVideo are **composed of stages**. Each pipeline inherits from `ComposedPipelineBase`.

### ComposedPipelineBase

**File**: `fastvideo/pipelines/composed_pipeline_base.py`

This is the foundation for all pipelines:

```python
class ComposedPipelineBase(ABC):
    def __init__(self, model_path, fastvideo_args, required_config_modules=None):
        self.model_path = model_path
        self.fastvideo_args = fastvideo_args
        self._stages: list[PipelineStage] = []
        self._stage_name_mapping: dict[str, PipelineStage] = {}
        
        # Load modules (vae, transformer, text_encoder, etc.)
        self.modules = self.load_modules(fastvideo_args)
    
    def post_init(self) -> None:
        # Create pipeline stages
        self.create_pipeline_stages(self.fastvideo_args)
    
    @abstractmethod
    def create_pipeline_stages(self, fastvideo_args: FastVideoArgs):
        """Subclasses implement this to define their stage sequence"""
        raise NotImplementedError
    
    def add_stage(self, stage_name: str, stage: PipelineStage):
        """Add a stage to the pipeline"""
        self._stages.append(stage)
        self._stage_name_mapping[stage_name] = stage
        setattr(self, stage_name, stage)
    
    @torch.no_grad()
    def forward(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs):
        """Execute all stages sequentially"""
        for stage in self.stages:
            batch = stage(batch, fastvideo_args)
        return batch
```

### Example: WanPipeline

**File**: `fastvideo/pipelines/basic/wan/wan_pipeline.py`

Here's how a concrete pipeline defines its stages:

```python
class WanPipeline(LoRAPipeline, ComposedPipelineBase):
    _required_config_modules = [
        "text_encoder", "tokenizer", "vae", "transformer", "scheduler"
    ]
    
    def create_pipeline_stages(self, fastvideo_args: FastVideoArgs) -> None:
        """Define the sequence of stages for Wan video generation"""
        
        # Stage 1: Validate inputs
        self.add_stage(
            stage_name="input_validation_stage",
            stage=InputValidationStage()
        )
        
        # Stage 2: Encode text prompts
        self.add_stage(
            stage_name="prompt_encoding_stage",
            stage=TextEncodingStage(
                text_encoders=[self.get_module("text_encoder")],
                tokenizers=[self.get_module("tokenizer")],
            )
        )
        
        # Stage 3: Prepare conditioning
        self.add_stage(
            stage_name="conditioning_stage",
            stage=ConditioningStage()
        )
        
        # Stage 4: Prepare timesteps
        self.add_stage(
            stage_name="timestep_preparation_stage",
            stage=TimestepPreparationStage(
                scheduler=self.get_module("scheduler")
            )
        )
        
        # Stage 5: Prepare latent noise
        self.add_stage(
            stage_name="latent_preparation_stage",
            stage=LatentPreparationStage(
                scheduler=self.get_module("scheduler"),
                transformer=self.get_module("transformer", None)
            )
        )
        
        # Stage 6: Denoising loop
        self.add_stage(
            stage_name="denoising_stage",
            stage=DenoisingStage(
                transformer=self.get_module("transformer"),
                scheduler=self.get_module("scheduler"),
                vae=self.get_module("vae"),
                pipeline=self
            )
        )
        
        # Stage 7: Decode to pixels
        self.add_stage(
            stage_name="decoding_stage",
            stage=DecodingStage(
                vae=self.get_module("vae"),
                pipeline=self
            )
        )
```

---

## Stage Execution

Each stage is a self-contained processing unit that transforms the `ForwardBatch`.

### PipelineStage Base Class

**File**: `fastvideo/pipelines/stages/base.py`

```python
class PipelineStage(ABC):
    def __call__(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs):
        """Execute the stage with verification and logging"""
        
        stage_name = self.__class__.__name__
        
        # Optional: Pre-execution verification
        if fastvideo_args.enable_stage_verification:
            input_result = self.verify_input(batch, fastvideo_args)
            if not input_result.is_valid():
                raise StageVerificationError(...)
        
        # Execute the stage
        if envs.FASTVIDEO_STAGE_LOGGING:
            logger.info("[%s] Starting execution", stage_name)
            start_time = time.perf_counter()
            result = self.forward(batch, fastvideo_args)
            execution_time = time.perf_counter() - start_time
            logger.info("[%s] Completed in %s ms", stage_name, execution_time * 1000)
        else:
            result = self.forward(batch, fastvideo_args)
        
        # Optional: Post-execution verification
        if fastvideo_args.enable_stage_verification:
            output_result = self.verify_output(result, fastvideo_args)
            if not output_result.is_valid():
                raise StageVerificationError(...)
        
        return result
    
    @abstractmethod
    def forward(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs):
        """Implement stage-specific logic"""
        raise NotImplementedError
```

### Example Stage: TextEncodingStage

**File**: `fastvideo/pipelines/stages/text_encoding.py`

```python
class TextEncodingStage(PipelineStage):
    def __init__(self, text_encoders, tokenizers):
        self.text_encoders = text_encoders
        self.tokenizers = tokenizers
    
    def forward(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs):
        """Encode text prompts into embeddings"""
        
        # Encode positive prompt
        prompt_embeds_list, prompt_masks_list = self.encode_text(
            batch.prompt,
            fastvideo_args,
            encoder_index=list(range(len(self.text_encoders))),
            return_attention_mask=True,
        )
        
        for pe in prompt_embeds_list:
            batch.prompt_embeds.append(pe)
        
        # Encode negative prompt if CFG is enabled
        if batch.do_classifier_free_guidance:
            neg_embeds_list, _ = self.encode_text(
                batch.negative_prompt,
                fastvideo_args,
                encoder_index=list(range(len(self.text_encoders))),
            )
            for ne in neg_embeds_list:
                batch.negative_prompt_embeds.append(ne)
        
        return batch
    
    def encode_text(self, text, fastvideo_args, encoder_index, ...):
        """Helper to encode text with multiple encoders"""
        embeds_list = []
        for i in encoder_index:
            tokenizer = self.tokenizers[i]
            text_encoder = self.text_encoders[i]
            
            # Tokenize
            text_inputs = tokenizer(text, ...).to(device)
            
            # Encode
            outputs = text_encoder(
                input_ids=text_inputs["input_ids"],
                attention_mask=text_inputs["attention_mask"],
            )
            
            embeds_list.append(outputs.hidden_states)
        
        return embeds_list
```

### Example Stage: DenoisingStage

**File**: `fastvideo/pipelines/stages/denoising.py`

This is the core denoising loop:

```python
class DenoisingStage(PipelineStage):
    def __init__(self, transformer, scheduler, vae=None, pipeline=None):
        self.transformer = transformer
        self.scheduler = scheduler
        self.vae = vae
        self.pipeline = weakref.ref(pipeline) if pipeline else None
    
    def forward(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs):
        """Run the iterative denoising process"""
        
        latents = batch.latents
        timesteps = batch.timesteps
        prompt_embeds = batch.prompt_embeds
        
        # Denoising loop
        for i, t in enumerate(timesteps):
            # Prepare input
            latent_model_input = self.scheduler.scale_model_input(latents, t)
            
            # Predict noise with transformer
            noise_pred = self.transformer(
                latent_model_input,
                prompt_embeds,
                timestep=t,
                guidance=guidance_scale,
            )
            
            # Apply classifier-free guidance if enabled
            if batch.do_classifier_free_guidance:
                noise_pred_uncond = self.transformer(
                    latent_model_input,
                    batch.negative_prompt_embeds,
                    timestep=t,
                )
                noise_pred = noise_pred_uncond + guidance_scale * (
                    noise_pred - noise_pred_uncond
                )
            
            # Compute previous sample
            latents = self.scheduler.step(
                noise_pred, t, latents
            )[0]
        
        # Update batch with denoised latents
        batch.latents = latents
        return batch
```

---

## How Adding a Stage Changes Functionality

The stage-based architecture makes it easy to extend pipelines with new functionality.

### Example 1: Adding an Image Conditioning Stage

To add image-to-video (I2V) capability to a text-to-video pipeline:

```python
class WanI2VPipeline(WanPipeline):
    """Wan pipeline with image conditioning"""
    
    _required_config_modules = [
        "text_encoder", "tokenizer", "vae", "transformer", "scheduler",
        "image_encoder"  # Additional module
    ]
    
    def create_pipeline_stages(self, fastvideo_args: FastVideoArgs) -> None:
        # Stage 1-2: Same as WanPipeline
        self.add_stage("input_validation_stage", InputValidationStage())
        self.add_stage("prompt_encoding_stage", TextEncodingStage(...))
        
        # NEW STAGE: Encode input image
        self.add_stage(
            stage_name="image_encoding_stage",
            stage=ImageEncodingStage(
                image_encoder=self.get_module("image_encoder"),
                vae=self.get_module("vae"),
            )
        )
        
        # Stages 3-7: Continue as normal
        self.add_stage("conditioning_stage", ConditioningStage())
        self.add_stage("timestep_preparation_stage", TimestepPreparationStage(...))
        self.add_stage("latent_preparation_stage", LatentPreparationStage(...))
        self.add_stage("denoising_stage", DenoisingStage(...))
        self.add_stage("decoding_stage", DecodingStage(...))
```

The new `ImageEncodingStage` would:
1. Load image from `batch.image_path`
2. Encode it with the VAE
3. Store result in `batch.image_latent`
4. The denoising stage automatically uses `batch.image_latent` if present

### Example 2: Adding Distillation/DMD Support

To add distribution matching distillation (single-step generation):

```python
class WanDMDPipeline(WanPipeline):
    """Wan pipeline with DMD for fast single-step generation"""
    
    def create_pipeline_stages(self, fastvideo_args: FastVideoArgs) -> None:
        # Stages 1-5: Same as WanPipeline
        self.add_stage("input_validation_stage", InputValidationStage())
        self.add_stage("prompt_encoding_stage", TextEncodingStage(...))
        self.add_stage("conditioning_stage", ConditioningStage())
        self.add_stage("timestep_preparation_stage", TimestepPreparationStage(...))
        self.add_stage("latent_preparation_stage", LatentPreparationStage(...))
        
        # REPLACE denoising stage with DMD version
        self.add_stage(
            stage_name="denoising_stage",
            stage=DmdDenoisingStage(  # Different implementation
                transformer=self.get_module("transformer"),
                scheduler=FlowMatchEulerDiscreteScheduler(shift=8.0),
            )
        )
        
        # Stage 7: Same as WanPipeline
        self.add_stage("decoding_stage", DecodingStage(...))
```

The `DmdDenoisingStage`:
- Uses a different scheduler
- Performs fewer denoising steps (typically 1-4 vs 50)
- Applies noise differently between steps

### Example 3: Adding Preprocessing for Training

For training pipelines, you might add preprocessing stages:

```python
class WanTrainingPipeline(WanPipeline):
    """Wan pipeline for training with data preprocessing"""
    
    def create_training_stages(self, training_args: TrainingArgs):
        # NEW STAGES: Data preprocessing
        self.add_stage(
            stage_name="video_loading_stage",
            stage=VideoLoadingStage()
        )
        
        self.add_stage(
            stage_name="video_augmentation_stage",
            stage=VideoAugmentationStage(
                augmentations=training_args.augmentations
            )
        )
        
        self.add_stage(
            stage_name="video_encoding_stage",
            stage=VideoEncodingStage(vae=self.get_module("vae"))
        )
        
        # Continue with inference stages
        self.add_stage("prompt_encoding_stage", TextEncodingStage(...))
        # ...
```

### Key Benefits of Stage-Based Architecture

1. **Modularity**: Each stage is self-contained and testable
2. **Reusability**: Stages can be shared across pipelines
3. **Extensibility**: Easy to add new functionality by inserting stages
4. **Debugging**: Can enable per-stage logging and verification
5. **Flexibility**: Can swap implementations (e.g., DMD vs standard denoising)

### What Each Stage Does

Here's what happens at each stage in a typical T2V pipeline:

| Stage | Input | Output | Purpose |
|-------|-------|--------|---------|
| **InputValidationStage** | `prompt`, params | Validated inputs | Check dimensions, prompt format |
| **TextEncodingStage** | `prompt`, `negative_prompt` | `prompt_embeds`, `negative_prompt_embeds` | Encode text to embeddings |
| **ConditioningStage** | Embeddings | `clip_embedding_pos`, `clip_embedding_neg` | Prepare conditioning signals |
| **TimestepPreparationStage** | `num_inference_steps` | `timesteps` | Generate denoising schedule |
| **LatentPreparationStage** | `num_frames`, `height`, `width`, `seed` | `latents` (noise) | Initialize random noise |
| **DenoisingStage** | `latents`, `timesteps`, embeddings | `latents` (denoised) | Iterative denoising with transformer |
| **DecodingStage** | `latents` | `output` (pixels) | Decode latents to video frames |

---

## Summary

The execution flow is:

1. **Entry**: CLI or Python API creates `VideoGenerator`
2. **Setup**: VideoGenerator creates an `Executor` (multiproc/ray)
3. **Workers**: Executor spawns `Worker` instances on each GPU
4. **Pipeline**: Each worker builds a `Pipeline` from model config
5. **Stages**: Pipeline executes stages sequentially
6. **Output**: Results are gathered and returned to user

**Adding a stage** changes functionality by:
- Inserting new processing between existing stages
- Modifying the `ForwardBatch` with new data (e.g., `image_latent`)
- Downstream stages automatically use the new data if available
- No changes needed to other stages (loose coupling)

This architecture makes FastVideo highly extensible while keeping the codebase maintainable!

