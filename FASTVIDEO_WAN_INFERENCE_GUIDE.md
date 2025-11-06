# FastVideo WAN Model Inference: Complete Guide

This document provides a comprehensive walkthrough of how FastVideo performs video generation inference using WAN (Wan-AI) models, with concrete code examples and detailed explanations.

---

## Table of Contents
1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Inference Flow: Step-by-Step](#inference-flow-step-by-step)
4. [Concrete Examples](#concrete-examples)
5. [Key Components](#key-components)
6. [Configuration & Parameters](#configuration--parameters)
7. [Model Variants](#model-variants)

---

## Overview

FastVideo is a unified post-training and inference framework for accelerated video generation. It supports multiple video diffusion models, including the WAN (Wan-AI) family of models. The framework uses a modular pipeline architecture with distinct stages for text encoding, denoising, and VAE decoding.

### Key Features for WAN Models:
- **Flow-matching scheduler** (FlowUniPCMultistepScheduler)
- **Temporal and spatial VAE compression** (4x temporal, 8x spatial)
- **Classifier-free guidance (CFG)** support
- **Flexible parallelism** (tensor parallelism, sequence parallelism)
- **Memory optimizations** (CPU offload, FSDP inference)
- **Attention backends** (Sliding Tile Attention, Video Sparse Attention, VMOBA)

---

## Architecture

### High-Level Pipeline Structure

```
User Input (Prompt) 
    ↓
[CLI/Python API] → VideoGenerator.from_pretrained()
    ↓
[FastVideoArgs] → Configuration parsing & validation
    ↓
[Executor] → Multiprocess/Ray worker initialization
    ↓
[Worker] → Pipeline construction
    ↓
[WanPipeline] → Load components (VAE, DiT, Text Encoder, Scheduler)
    ↓
[Pipeline Stages] → Sequential execution:
    1. Input Validation
    2. Text Encoding (T5)
    3. Conditioning (CFG setup)
    4. Timestep Preparation
    5. Latent Preparation
    6. Denoising Loop (DiT + Scheduler)
    7. VAE Decoding
    ↓
[Output] → Video frames saved to disk
```

### Component Layout

**Core Modules:**
- **VideoGenerator** (`fastvideo/entrypoints/video_generator.py`): Main user-facing API
- **WanPipeline** (`fastvideo/pipelines/basic/wan/wan_pipeline.py`): Pipeline orchestrator
- **WanVideo** (`fastvideo/models/dits/wanvideo.py`): Diffusion Transformer model
- **WanVAE** (`fastvideo/models/vaes/wanvae.py`): Video VAE encoder/decoder
- **FlowUniPCMultistepScheduler** (`fastvideo/models/schedulers/scheduling_flow_unipc_multistep.py`): Scheduler

---

## Inference Flow: Step-by-Step

### 1. Entry Point & Command Parsing

**Starting Point:** CLI command `fastvideo generate` or Python API

```bash
fastvideo generate \
    --model-path Wan-AI/Wan2.1-T2V-1.3B-Diffusers \
    --prompt "A curious raccoon in sunflowers" \
    --num-gpus 1 \
    --height 480 --width 832 --num-frames 77 \
    --num-inference-steps 50 \
    --guidance-scale 6.0 \
    --flow-shift 8.0 \
    --seed 1024
```

**Code Flow:**
1. **Entry:** `pyproject.toml` defines console script → `fastvideo.entrypoints.cli.main:main`
2. **CLI Parser:** `fastvideo/entrypoints/cli/main.py` creates `FlexibleArgumentParser`
3. **Subcommand:** `GenerateSubcommand` in `fastvideo/entrypoints/cli/generate.py`
4. **Argument Splitting:**
   - **Init args** (for VideoGenerator): `model_path`, `num_gpus`, `tp_size`, `sp_size`
   - **Generation args** (for generate_video): sampling parameters (height, width, steps, etc.)

**Relevant Code:**
```python
# fastvideo/entrypoints/cli/generate.py
class GenerateSubcommand(CLISubcommand):
    def cmd(self, args: argparse.Namespace) -> None:
        init_args = {k: v for k, v in merged_args.items() 
                     if k not in self.generation_arg_names}
        generation_args = {k: v for k, v in merged_args.items() 
                          if k in self.generation_arg_names}
        
        model_path = init_args.pop('model_path')
        prompt = generation_args.pop('prompt', None)
        
        generator = VideoGenerator.from_pretrained(model_path=model_path, **init_args)
        generator.generate_video(prompt=prompt, **generation_args)
```

---

### 2. VideoGenerator Initialization

**Purpose:** Create the generator instance and load configuration.

**Code Flow:**
```python
# fastvideo/entrypoints/video_generator.py
class VideoGenerator:
    @classmethod
    def from_pretrained(cls, model_path: str, **kwargs) -> "VideoGenerator":
        # Step 1: Build FastVideoArgs from model_path + user kwargs
        kwargs['model_path'] = model_path
        fastvideo_args = FastVideoArgs.from_kwargs(**kwargs)
        
        return cls.from_fastvideo_args(fastvideo_args)
    
    @classmethod
    def from_fastvideo_args(cls, fastvideo_args: FastVideoArgs) -> "VideoGenerator":
        # Step 2: Select executor (multiprocess or Ray)
        executor_class = Executor.get_class(fastvideo_args)
        
        # Step 3: Return VideoGenerator instance
        return cls(fastvideo_args=fastvideo_args, 
                  executor_class=executor_class,
                  log_stats=False)
```

**Configuration Resolution:**

The `FastVideoArgs` includes nested `PipelineConfig`:
- Model ID `Wan-AI/Wan2.1-T2V-1.3B-Diffusers` → Matched to `WanT2V480PConfig`
- User kwargs override default config values (e.g., `--flow-shift 8.0` overrides default 3.0)

```python
# fastvideo/configs/pipelines/wan.py
@dataclass
class WanT2V480PConfig(PipelineConfig):
    dit_config: DiTConfig = field(default_factory=WanVideoConfig)
    vae_config: VAEConfig = field(default_factory=WanVAEConfig)
    vae_tiling: bool = False
    flow_shift: float | None = 3.0  # Can be overridden
    text_encoder_configs: tuple[EncoderConfig, ...] = field(
        default_factory=lambda: (T5Config(),))
    precision: str = "bf16"
    vae_precision: str = "fp32"
```

---

### 3. Executor & Worker Initialization

**Purpose:** Set up multiprocess workers for distributed inference.

**Executor Selection:**
- `distributed_executor_backend == "mp"` → `MultiprocExecutor` (default)
- `distributed_executor_backend == "ray"` → `RayExecutor` (for Ray clusters)

**Worker Initialization:**
```python
# fastvideo/worker/multiproc_executor.py
class MultiprocExecutor(Executor):
    def _init_executor(self):
        world_size = self.fastvideo_args.num_gpus
        
        # Launch worker processes (one per GPU)
        for rank in range(world_size):
            worker_proc = WorkerMultiprocProc(rank=rank, ...)
            worker_proc.start()
        
        # Initialize pipeline on each worker
        self._run_workers("init_device")
```

**Worker Pipeline Construction:**
```python
# fastvideo/worker/gpu_worker.py
class Worker:
    def init_device(self):
        # 1. Set CUDA device
        torch.cuda.set_device(self.local_rank)
        
        # 2. Initialize distributed (if multi-GPU)
        if self.world_size > 1:
            torch.distributed.init_process_group(...)
        
        # 3. Build pipeline
        self.pipeline = build_pipeline(self.fastvideo_args)
```

**Pipeline Building:**
```python
# fastvideo/pipelines/__init__.py
def build_pipeline(fastvideo_args: FastVideoArgs, 
                   pipeline_type: PipelineType = PipelineType.BASIC):
    # 1. Download model if needed
    maybe_download_model(model_path)
    
    # 2. Read model_index.json
    verify_model_config_and_directory(model_path)
    
    # 3. Get pipeline class (e.g., WanPipeline)
    pipeline_cls = get_pipeline_class_from_config(model_path, workload_type)
    
    # 4. Instantiate pipeline
    return pipeline_cls(model_path, fastvideo_args)
```

---

### 4. WanPipeline Component Loading

**Purpose:** Load all model components (VAE, DiT, text encoder, tokenizer, scheduler).

```python
# fastvideo/pipelines/basic/wan/wan_pipeline.py
class WanPipeline(LoRAPipeline, ComposedPipelineBase):
    _required_config_modules = [
        "text_encoder",  # T5-XXL
        "tokenizer",     # T5 tokenizer
        "vae",           # WanVAE
        "transformer",   # WanVideo (DiT)
        "scheduler"      # FlowUniPCMultistepScheduler
    ]
    
    def initialize_pipeline(self, fastvideo_args: FastVideoArgs):
        # Replace scheduler with custom flow-matching version
        self.modules["scheduler"] = FlowUniPCMultistepScheduler(
            shift=fastvideo_args.pipeline_config.flow_shift)  # e.g., 8.0
    
    def create_pipeline_stages(self, fastvideo_args: FastVideoArgs) -> None:
        # Stage 1: Input validation
        self.add_stage("input_validation_stage", 
                      InputValidationStage())
        
        # Stage 2: Text encoding (T5)
        self.add_stage("prompt_encoding_stage", 
                      TextEncodingStage(
                          text_encoders=[self.get_module("text_encoder")],
                          tokenizers=[self.get_module("tokenizer")]))
        
        # Stage 3: Conditioning (CFG setup)
        self.add_stage("conditioning_stage", 
                      ConditioningStage())
        
        # Stage 4: Timestep preparation
        self.add_stage("timestep_preparation_stage", 
                      TimestepPreparationStage(
                          scheduler=self.get_module("scheduler")))
        
        # Stage 5: Latent preparation (noise initialization)
        self.add_stage("latent_preparation_stage", 
                      LatentPreparationStage(
                          scheduler=self.get_module("scheduler"),
                          transformer=self.get_module("transformer")))
        
        # Stage 6: Denoising loop
        self.add_stage("denoising_stage", 
                      DenoisingStage(
                          transformer=self.get_module("transformer"),
                          scheduler=self.get_module("scheduler"),
                          vae=self.get_module("vae"),
                          pipeline=self))
        
        # Stage 7: VAE decoding
        self.add_stage("decoding_stage", 
                      DecodingStage(vae=self.get_module("vae"),
                                   pipeline=self))
```

**Component Loading Details:**
```python
# fastvideo/pipelines/composed_pipeline_base.py
class ComposedPipelineBase:
    def load_modules(self, model_path: str, fastvideo_args: FastVideoArgs):
        # Read model_index.json
        config = json.load(open(f"{model_path}/model_index.json"))
        
        for module_name in self._required_config_modules:
            loader = PipelineComponentLoader.get_loader(module_name)
            
            if module_name == "transformer":
                # Load DiT with precision (bf16) and optional FSDP
                self.modules[module_name] = TransformerLoader.load(
                    path=f"{model_path}/transformer",
                    fastvideo_args=fastvideo_args)
            
            elif module_name == "vae":
                # Load VAE with precision (fp32 for WAN)
                self.modules[module_name] = VAELoader.load(
                    path=f"{model_path}/vae",
                    fastvideo_args=fastvideo_args)
            
            elif module_name == "text_encoder":
                # Load T5-XXL encoder
                self.modules[module_name] = TextEncoderLoader.load(
                    path=f"{model_path}/text_encoder",
                    fastvideo_args=fastvideo_args)
```

---

### 5. Generate Video: Batch Creation

**Purpose:** Parse prompts and create `ForwardBatch` objects for inference.

```python
# fastvideo/entrypoints/video_generator.py
class VideoGenerator:
    def generate_video(self, prompt: str | None = None, 
                      sampling_param: SamplingParam | None = None,
                      **kwargs) -> dict[str, Any]:
        # Handle batch processing from text file
        if self.fastvideo_args.prompt_txt is not None:
            with open(self.fastvideo_args.prompt_txt) as f:
                prompts = [line.strip() for line in f if line.strip()]
            
            results = []
            for batch_prompt in prompts:
                result = self._generate_single_video(
                    prompt=batch_prompt,
                    sampling_param=sampling_param,
                    **kwargs)
                results.append(result)
            return results
        
        # Single prompt generation
        return self._generate_single_video(
            prompt=prompt,
            sampling_param=sampling_param,
            **kwargs)
    
    def _generate_single_video(self, prompt: str, 
                               sampling_param: SamplingParam,
                               **kwargs) -> dict:
        # Align dimensions
        if sampling_param.use_temporal_scaling_frames:
            orig_latent_num_frames = (sampling_param.num_frames - 1) // 4 + 1
        
        sampling_param.height = align_to(sampling_param.height, 16)
        sampling_param.width = align_to(sampling_param.width, 16)
        
        # Create ForwardBatch
        batch = ForwardBatch(
            prompt=prompt,
            negative_prompt=sampling_param.negative_prompt,
            num_frames=sampling_param.num_frames,
            height=sampling_param.height,
            width=sampling_param.width,
            guidance_scale=sampling_param.guidance_scale,
            num_inference_steps=sampling_param.num_inference_steps,
            seed=sampling_param.seed,
            fps=sampling_param.fps,
            # ... more fields
        )
        
        # Execute on worker
        result = self.executor.execute_forward(batch, self.fastvideo_args)
        
        # Save video
        frames = self._process_output(result)
        self._save_video(frames, output_path, fps=sampling_param.fps)
        
        return {"frames": frames, "prompt": prompt, ...}
```

---

### 6. Pipeline Execution: Stage-by-Stage

The `ComposedPipelineBase.forward()` method executes stages sequentially:

```python
# fastvideo/pipelines/composed_pipeline_base.py
class ComposedPipelineBase:
    def forward(self, batch: ForwardBatch, 
                fastvideo_args: FastVideoArgs) -> ForwardBatch:
        for stage_name, stage in self.stages.items():
            batch = stage.forward(batch, fastvideo_args)
        return batch
```

#### **Stage 1: Input Validation**

```python
# fastvideo/pipelines/stages/input_validation.py
class InputValidationStage(PipelineStage):
    def forward(self, batch: ForwardBatch, 
                fastvideo_args: FastVideoArgs) -> ForwardBatch:
        # Check prompt is not empty
        assert batch.prompt is not None and len(batch.prompt) > 0
        
        # Validate dimensions
        assert 256 <= batch.height <= 2048
        assert 256 <= batch.width <= 2048
        assert 1 <= batch.num_frames <= 256
        
        # Validate guidance scale
        batch.do_classifier_free_guidance = batch.guidance_scale > 1.0
        
        return batch
```

#### **Stage 2: Text Encoding**

```python
# fastvideo/pipelines/stages/text_encoding.py
class TextEncodingStage(PipelineStage):
    def __init__(self, text_encoders, tokenizers):
        self.text_encoders = text_encoders  # [T5-XXL]
        self.tokenizers = tokenizers        # [T5 tokenizer]
    
    def forward(self, batch: ForwardBatch, 
                fastvideo_args: FastVideoArgs) -> ForwardBatch:
        # Encode positive prompt
        prompt_embeds_list, prompt_masks_list = self.encode_text(
            batch.prompt,
            fastvideo_args,
            encoder_index=[0])  # Use first (only) encoder
        
        batch.prompt_embeds = prompt_embeds_list  # [B, 512, 4096]
        batch.prompt_attention_mask = prompt_masks_list
        
        # Encode negative prompt if CFG enabled
        if batch.do_classifier_free_guidance:
            neg_embeds_list, neg_masks_list = self.encode_text(
                batch.negative_prompt,
                fastvideo_args,
                encoder_index=[0])
            
            batch.negative_prompt_embeds = neg_embeds_list
            batch.negative_attention_mask = neg_masks_list
        
        return batch
    
    def encode_text(self, text: str, fastvideo_args, encoder_index):
        tokenizer = self.tokenizers[encoder_index[0]]
        text_encoder = self.text_encoders[encoder_index[0]]
        
        # Tokenize
        text_inputs = tokenizer(
            text,
            padding="max_length",
            max_length=512,
            truncation=True,
            return_tensors="pt")
        
        # Encode
        with torch.no_grad():
            outputs = text_encoder(
                input_ids=text_inputs.input_ids,
                attention_mask=text_inputs.attention_mask)
        
        # Postprocess (apply T5 specific processing)
        embeds = fastvideo_args.pipeline_config.postprocess_text_funcs[0](outputs)
        
        return [embeds], [text_inputs.attention_mask]
```

#### **Stage 3: Conditioning**

```python
# fastvideo/pipelines/stages/conditioning.py
class ConditioningStage(PipelineStage):
    def forward(self, batch: ForwardBatch, 
                fastvideo_args: FastVideoArgs) -> ForwardBatch:
        # This stage handles additional conditioning like CFG concatenation
        # For WAN, it's mostly a pass-through
        return batch
```

#### **Stage 4: Timestep Preparation**

```python
# fastvideo/pipelines/stages/timestep_preparation.py
class TimestepPreparationStage(PipelineStage):
    def __init__(self, scheduler):
        self.scheduler = scheduler
    
    def forward(self, batch: ForwardBatch, 
                fastvideo_args: FastVideoArgs) -> ForwardBatch:
        # Set timesteps on scheduler
        self.scheduler.set_timesteps(
            batch.num_inference_steps,  # e.g., 50
            device=get_local_torch_device())
        
        # Store timesteps in batch
        batch.timesteps = self.scheduler.timesteps  # [50] tensor
        
        return batch
```

**FlowUniPCMultistepScheduler Timesteps:**
```python
# fastvideo/models/schedulers/scheduling_flow_unipc_multistep.py
class FlowUniPCMultistepScheduler(SchedulerMixin, ConfigMixin):
    def __init__(self, shift: float = 1.0, num_train_timesteps: int = 1000):
        self.shift = shift  # e.g., 8.0
        self.num_train_timesteps = num_train_timesteps
    
    def set_timesteps(self, num_inference_steps: int, device):
        # Generate timesteps with shifting
        timesteps = np.linspace(1.0, 0.0, num_inference_steps + 1)
        
        # Apply shift: t' = shift / (1 + (shift - 1) * (1 - t))
        timesteps = self.shift / (1 + (self.shift - 1) * (1 - timesteps))
        
        self.timesteps = torch.from_numpy(timesteps).to(device)
```

#### **Stage 5: Latent Preparation**

```python
# fastvideo/pipelines/stages/latent_preparation.py
class LatentPreparationStage(PipelineStage):
    def __init__(self, scheduler, transformer):
        self.scheduler = scheduler
        self.transformer = transformer
    
    def forward(self, batch: ForwardBatch, 
                fastvideo_args: FastVideoArgs) -> ForwardBatch:
        # Calculate latent shape
        vae_config = fastvideo_args.pipeline_config.vae_config
        spatial_compression = vae_config.arch_config.spatial_compression_ratio  # 8
        temporal_compression = vae_config.arch_config.temporal_compression_ratio  # 4
        
        if vae_config.use_temporal_scaling_frames:
            latent_num_frames = (batch.num_frames - 1) // temporal_compression + 1
        else:
            latent_num_frames = batch.num_frames
        
        shape = (
            1,  # batch_size
            self.transformer.num_channels_latents,  # 16 for WAN
            latent_num_frames,  # e.g., (77-1)//4 + 1 = 20
            batch.height // spatial_compression,  # 480//8 = 60
            batch.width // spatial_compression,   # 832//8 = 104
        )
        
        # Generate initial noise
        generator = torch.Generator(device=get_local_torch_device())
        if batch.seed is not None:
            generator.manual_seed(batch.seed)
        
        latents = torch.randn(
            shape,
            generator=generator,
            device=get_local_torch_device(),
            dtype=torch.bfloat16)
        
        # Scale by initial noise sigma
        if hasattr(self.scheduler, "init_noise_sigma"):
            latents = latents * self.scheduler.init_noise_sigma
        
        batch.latents = latents  # [1, 16, 20, 60, 104]
        batch.raw_latent_shape = shape
        
        return batch
```

#### **Stage 6: Denoising Loop**

This is the core stage where the diffusion model iteratively denoises the latents.

```python
# fastvideo/pipelines/stages/denoising.py
class DenoisingStage(PipelineStage):
    def __init__(self, transformer, scheduler, vae=None, pipeline=None):
        self.transformer = transformer  # WanVideo DiT
        self.scheduler = scheduler      # FlowUniPCMultistepScheduler
        self.vae = vae
        self.pipeline = weakref.ref(pipeline) if pipeline else None
        
        # Select attention backend
        self.attn_backend = get_attn_backend(
            head_size=transformer.hidden_size // transformer.num_attention_heads,
            dtype=torch.float16,
            supported_attention_backends=(
                AttentionBackendEnum.SLIDING_TILE_ATTN,
                AttentionBackendEnum.VIDEO_SPARSE_ATTN,
                AttentionBackendEnum.VMOBA_ATTN,
                AttentionBackendEnum.FLASH_ATTN,
                AttentionBackendEnum.TORCH_SDPA))
    
    def forward(self, batch: ForwardBatch, 
                fastvideo_args: FastVideoArgs) -> ForwardBatch:
        # Setup
        latents = batch.latents  # [1, 16, 20, 60, 104]
        timesteps = batch.timesteps  # [50]
        prompt_embeds = batch.prompt_embeds[0]  # [1, 512, 4096]
        negative_prompt_embeds = batch.negative_prompt_embeds[0] if batch.do_classifier_free_guidance else None
        
        # Prepare extra kwargs for transformer
        extra_kwargs = {
            "encoder_hidden_states": prompt_embeds,
            "encoder_attention_mask": batch.prompt_attention_mask[0] if batch.prompt_attention_mask else None,
        }
        
        # Denoising loop
        for i, t in enumerate(tqdm(timesteps, desc="Denoising")):
            # Handle classifier-free guidance
            if batch.do_classifier_free_guidance:
                # Duplicate latents for conditional and unconditional
                latent_model_input = torch.cat([latents, latents], dim=0)
                
                # Combine embeddings
                combined_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
                extra_kwargs["encoder_hidden_states"] = combined_embeds
            else:
                latent_model_input = latents
            
            # Scale model input
            latent_model_input = self.scheduler.scale_model_input(
                latent_model_input, t)
            
            # Predict noise with DiT
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                noise_pred = self.transformer(
                    latent_model_input,
                    timestep=t,
                    **extra_kwargs)
            
            # Apply CFG
            if batch.do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + batch.guidance_scale * (
                    noise_pred_text - noise_pred_uncond)
            
            # Scheduler step
            latents = self.scheduler.step(
                noise_pred, t, latents,
                return_dict=False)[0]
        
        batch.latents = latents  # Denoised latents
        return batch
```

**WanVideo (DiT) Forward Pass:**
```python
# fastvideo/models/dits/wanvideo.py (simplified)
class WanVideo(nn.Module):
    def __init__(self, hidden_size=3072, num_attention_heads=24, 
                 depth=32, num_channels_latents=16):
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.depth = depth
        self.num_channels_latents = num_channels_latents
        
        # Patch embedding
        self.patch_embed = PatchEmbed(
            in_channels=num_channels_latents,
            out_channels=hidden_size,
            patch_size=(1, 2, 2))
        
        # Time/text embedding
        self.time_text_embed = WanTimeTextImageEmbedding(
            dim=hidden_size,
            time_freq_dim=256,
            text_embed_dim=4096)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            WanTransformerBlock(dim=hidden_size, num_heads=num_attention_heads)
            for _ in range(depth)
        ])
        
        # Output projection
        self.norm_out = RMSNorm(hidden_size)
        self.proj_out = nn.Linear(hidden_size, num_channels_latents * 4)
    
    def forward(self, x, timestep, encoder_hidden_states, 
                encoder_attention_mask=None):
        # x: [B, C, T, H, W] = [2, 16, 20, 60, 104] (with CFG)
        # timestep: scalar or [B]
        # encoder_hidden_states: [B, 512, 4096]
        
        # 1. Patch embedding
        x = self.patch_embed(x)  # [B, T*H*W/4, hidden_size]
        
        # 2. Time/text embedding
        time_emb, time_proj, text_emb, _ = self.time_text_embed(
            timestep, encoder_hidden_states)
        
        # 3. Transformer blocks with cross-attention
        for block in self.blocks:
            x = block(x, time_proj, text_emb, encoder_attention_mask)
        
        # 4. Output projection
        x = self.norm_out(x)
        x = self.proj_out(x)  # [B, T*H*W/4, 16*4]
        
        # 5. Unpatchify
        x = self.unpatchify(x)  # [B, 16, 20, 60, 104]
        
        return x
```

#### **Stage 7: VAE Decoding**

```python
# fastvideo/pipelines/stages/decoding.py
class DecodingStage(PipelineStage):
    def __init__(self, vae, pipeline=None):
        self.vae = vae  # WanVAE
        self.pipeline = weakref.ref(pipeline) if pipeline else None
    
    def forward(self, batch: ForwardBatch, 
                fastvideo_args: FastVideoArgs) -> ForwardBatch:
        latents = batch.latents  # [1, 16, 20, 60, 104]
        
        # Decode with VAE
        video = self.decode(latents, fastvideo_args)
        
        batch.output = video  # [1, 3, 77, 480, 832]
        return batch
    
    def decode(self, latents: torch.Tensor, 
               fastvideo_args: FastVideoArgs) -> torch.Tensor:
        # Setup VAE precision
        vae_dtype = PRECISION_TO_TYPE[
            fastvideo_args.pipeline_config.vae_precision]  # fp32
        
        # Unscale latents
        if hasattr(self.vae, 'scaling_factor'):
            latents = latents / self.vae.scaling_factor
        
        # Apply shift if needed
        if hasattr(self.vae, 'shift_factor') and self.vae.shift_factor:
            latents = latents + self.vae.shift_factor
        
        # Decode
        with torch.autocast(device_type="cuda", dtype=vae_dtype, 
                           enabled=(vae_dtype != torch.float32)):
            if fastvideo_args.pipeline_config.vae_tiling:
                self.vae.enable_tiling()
            
            video = self.vae.decode(latents)  # [1, 3, 77, 480, 832]
        
        # Normalize to [0, 1]
        video = (video / 2 + 0.5).clamp(0, 1)
        
        # Move to CPU
        video = video.cpu().float()
        
        return video
```

---

### 7. Output Processing & Saving

```python
# fastvideo/entrypoints/video_generator.py
class VideoGenerator:
    def _generate_single_video(self, ...):
        # ... (batch creation and execution as shown above)
        
        # Process output
        video = result.output  # [1, 3, 77, 480, 832]
        
        # Rearrange to frame list
        frames = rearrange(video, "b c t h w -> t h w c")  # [77, 480, 832, 3]
        frames = (frames * 255).numpy().astype(np.uint8)
        
        # Save video
        if sampling_param.save_video:
            imageio.mimsave(
                output_path,
                frames,
                fps=sampling_param.fps,  # 16
                codec='libx264',
                quality=8)
        
        return {
            "frames": frames if sampling_param.return_frames else None,
            "prompt": prompt,
            "size": (sampling_param.height, sampling_param.width),
            "num_frames": sampling_param.num_frames,
            "generation_time": generation_time,
        }
```

---

## Concrete Examples

### Example 1: Basic Text-to-Video (T2V)

**Python API:**
```python
from fastvideo import VideoGenerator

# Initialize generator
generator = VideoGenerator.from_pretrained(
    "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
    num_gpus=1,
    dit_cpu_offload=False,
    vae_cpu_offload=False,
    text_encoder_cpu_offload=True,
    pin_cpu_memory=False)

# Generate video
prompt = "A curious raccoon peers through a vibrant field of yellow sunflowers"
video = generator.generate_video(
    prompt,
    output_path="outputs_video/",
    save_video=True,
    height=480,
    width=832,
    num_frames=77,
    num_inference_steps=50,
    fps=16,
    guidance_scale=6.0,
    seed=1024)
```

**CLI:**
```bash
fastvideo generate \
    --model-path Wan-AI/Wan2.1-T2V-1.3B-Diffusers \
    --num-gpus 1 \
    --height 480 --width 832 --num-frames 77 \
    --num-inference-steps 50 --fps 16 \
    --guidance-scale 6.0 --flow-shift 8.0 \
    --prompt "A curious raccoon peers through a vibrant field of yellow sunflowers" \
    --negative-prompt "Bright tones, overexposed, static, blurred details" \
    --seed 1024 \
    --output-path outputs_video/
```

### Example 2: Image-to-Video (I2V)

**Python API:**
```python
from fastvideo import VideoGenerator

# Initialize generator
generator = VideoGenerator.from_pretrained(
    "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers",
    num_gpus=2,
    tp_size=2,  # Tensor parallelism across 2 GPUs
    sp_size=2,  # Sequence parallelism across 2 GPUs
    use_fsdp_inference=True,
    dit_cpu_offload=False,
    vae_cpu_offload=False,
    text_encoder_cpu_offload=True)

# Generate video from image
prompt = "An astronaut hatching from an egg, on the surface of the moon"
image_path = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/astronaut.jpg"

video = generator.generate_video(
    prompt,
    image_path=image_path,
    output_path="outputs_i2v/",
    save_video=True,
    height=480,
    width=832,
    num_frames=77,
    num_inference_steps=40,
    fps=16,
    guidance_scale=5.0,
    seed=1024)
```

**CLI:**
```bash
fastvideo generate \
    --model-path Wan-AI/Wan2.1-I2V-14B-480P-Diffusers \
    --num-gpus 2 --tp-size 2 --sp-size 2 \
    --height 480 --width 832 --num-frames 77 \
    --num-inference-steps 40 --fps 16 \
    --flow-shift 3.0 --guidance-scale 5.0 \
    --image-path "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/astronaut.jpg" \
    --prompt "An astronaut hatching from an egg, on the surface of the moon" \
    --seed 1024 \
    --output-path outputs_i2v/
```

### Example 3: Batch Processing with Prompt File

**Prompt file** (`prompts.txt`):
```
A curious raccoon peers through a vibrant field of yellow sunflowers
A majestic lion strides across the golden savanna
A colorful parrot flies through a tropical rainforest
```

**CLI:**
```bash
fastvideo generate \
    --model-path Wan-AI/Wan2.1-T2V-1.3B-Diffusers \
    --num-gpus 1 \
    --height 480 --width 832 --num-frames 77 \
    --num-inference-steps 50 --fps 16 \
    --guidance-scale 6.0 --flow-shift 8.0 \
    --prompt-txt prompts.txt \
    --seed 1024 \
    --output-path outputs_batch/
```

This will generate 3 videos, one for each prompt.

### Example 4: Using FastWan (Distilled Model)

**Python API:**
```python
import os
from fastvideo import VideoGenerator

# Set attention backend to Video Sparse Attention
os.environ["FASTVIDEO_ATTENTION_BACKEND"] = "VIDEO_SPARSE_ATTN"

# Initialize with FastWan distilled model
generator = VideoGenerator.from_pretrained(
    "FastVideo/FastWan2.1-T2V-1.3B-Diffusers",
    num_gpus=1)

# Generate with fewer steps (distilled model supports 4-8 steps)
prompt = "A curious raccoon peers through a vibrant field of yellow sunflowers"
video = generator.generate_video(
    prompt,
    output_path="my_videos/",
    save_video=True,
    num_inference_steps=8,  # Much fewer steps!
    height=480,
    width=832,
    num_frames=77)
```

### Example 5: Using Wan2.2 (MoE Model)

**Python API:**
```python
from fastvideo import VideoGenerator

# Initialize Wan2.2 (requires MoE support)
generator = VideoGenerator.from_pretrained(
    "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
    num_gpus=1,
    use_fsdp_inference=True,  # Required for MoE
    dit_cpu_offload=True,     # Offload DiT to save memory
    vae_cpu_offload=False,
    text_encoder_cpu_offload=True,
    pin_cpu_memory=True)

# Generate 720p video
prompt = "A curious raccoon peers through a vibrant field of yellow sunflowers"
video = generator.generate_video(
    prompt,
    output_path="outputs_720p/",
    save_video=True,
    height=720,
    width=1280,
    num_frames=81,
    num_inference_steps=50)
```

---

## Key Components

### 1. WanVideo (DiT Model)

**File:** `fastvideo/models/dits/wanvideo.py`

**Architecture:**
- **Type:** Diffusion Transformer (DiT)
- **Hidden size:** 3072 (1.3B) or larger (14B)
- **Attention heads:** 24
- **Depth:** 32 transformer blocks
- **Latent channels:** 16
- **Patch size:** (1, 2, 2) - temporal×height×width

**Key Components:**
- `WanTimeTextImageEmbedding`: Combines timestep, text, and optional image embeddings
- `WanTransformerBlock`: Self-attention + cross-attention + MLP
- `WanSelfAttention`: Temporal-spatial self-attention
- `WanT2VCrossAttention`: Cross-attention with text embeddings

**Attention Mechanisms:**
- Full 3D attention (temporal + spatial)
- Rotary position embeddings (RoPE)
- QK normalization
- Support for various backends (Flash Attention, STA, VSA, VMOBA)

### 2. WanVAE

**File:** `fastvideo/models/vaes/wanvae.py`

**Properties:**
- **Spatial compression:** 8x (480×832 → 60×104)
- **Temporal compression:** 4x (77 frames → 20 latent frames)
- **Latent channels:** 16
- **Precision:** FP32 (for stability)

**Compression Formula:**
```python
latent_height = height // 8
latent_width = width // 8
latent_frames = (num_frames - 1) // 4 + 1
```

### 3. Text Encoder

**Model:** T5-XXL (4.7B parameters)

**Properties:**
- **Max sequence length:** 512 tokens
- **Hidden size:** 4096
- **Output:** Text embeddings [batch, 512, 4096]

**Postprocessing:**
```python
def t5_postprocess_text(outputs: BaseEncoderOutput) -> torch.Tensor:
    mask = outputs.attention_mask
    hidden_state = outputs.last_hidden_state
    seq_lens = mask.gt(0).sum(dim=1).long()
    
    # Trim to actual length, pad to 512
    prompt_embeds = [u[:v] for u, v in zip(hidden_state, seq_lens)]
    prompt_embeds_tensor = torch.stack([
        torch.cat([u, u.new_zeros(512 - u.size(0), u.size(1))])
        for u in prompt_embeds
    ], dim=0)
    
    return prompt_embeds_tensor
```

### 4. FlowUniPCMultistepScheduler

**File:** `fastvideo/models/schedulers/scheduling_flow_unipc_multistep.py`

**Type:** Flow-matching scheduler (not standard diffusion)

**Key Parameters:**
- **shift:** Controls timestep distribution (3.0-12.0 depending on model)
- **num_train_timesteps:** 1000
- **solver_order:** 2 (UniPC order)
- **prediction_type:** "flow_prediction"

**Timestep Scheduling:**
```python
# Linear schedule: [1.0, ..., 0.0]
t = np.linspace(1.0, 0.0, num_inference_steps + 1)

# Apply shift: t' = shift / (1 + (shift - 1) * (1 - t))
t_shifted = shift / (1 + (shift - 1) * (1 - t))
```

**Flow Matching:**
- Predicts the flow (velocity) instead of noise
- Interpolates between noise and data: `x_t = (1-t)*noise + t*data`
- Flow prediction: `v_t = data - noise`

---

## Configuration & Parameters

### FastVideoArgs

**File:** `fastvideo/fastvideo_args.py`

**Key Parameters:**
```python
@dataclass
class FastVideoArgs:
    # Model
    model_path: str  # HF model ID or local path
    
    # Distributed
    num_gpus: int = 1
    tp_size: int = 1     # Tensor parallelism size
    sp_size: int = 1     # Sequence parallelism size
    
    # Memory management
    dit_cpu_offload: bool = False
    vae_cpu_offload: bool = False
    text_encoder_cpu_offload: bool = True
    pin_cpu_memory: bool = False
    use_fsdp_inference: bool = False
    
    # Compilation
    enable_torch_compile: bool = False
    torch_compile_backend: str = "inductor"
    
    # Attention
    attention_backend: str | None = None  # or env var FASTVIDEO_ATTENTION_BACKEND
    
    # Pipeline config (nested)
    pipeline_config: PipelineConfig = None
```

### PipelineConfig (WAN)

**File:** `fastvideo/configs/pipelines/wan.py`

**WanT2V480PConfig:**
```python
@dataclass
class WanT2V480PConfig(PipelineConfig):
    # DiT
    dit_config: DiTConfig = WanVideoConfig()
    
    # VAE
    vae_config: VAEConfig = WanVAEConfig()
    vae_tiling: bool = False
    vae_sp: bool = False
    
    # Scheduler
    flow_shift: float = 3.0
    
    # Text encoder
    text_encoder_configs: tuple[EncoderConfig, ...] = (T5Config(),)
    
    # Precision
    precision: str = "bf16"          # DiT precision
    vae_precision: str = "fp32"      # VAE precision
    text_encoder_precisions: tuple[str, ...] = ("fp32",)
```

**Model-specific configs:**
- `WanT2V480PConfig`: 1.3B T2V, 480P, shift=3.0
- `WanT2V720PConfig`: 14B T2V, 720P, shift=5.0
- `WanI2V480PConfig`: 14B I2V, 480P, shift=3.0
- `WanI2V720PConfig`: 14B I2V, 720P, shift=5.0
- `Wan2_2_T2V_A14B_Config`: 14B MoE T2V, shift=12.0
- `Wan2_2_I2V_A14B_Config`: 14B MoE I2V, shift=5.0

### SamplingParam

**File:** `fastvideo/configs/sample/base.py`

**Parameters:**
```python
@dataclass
class SamplingParam:
    # Prompt
    prompt: str | None = None
    negative_prompt: str = ""
    prompt_path: str | None = None
    
    # Dimensions
    height: int = 480
    width: int = 832
    num_frames: int = 77
    fps: int = 16
    
    # Sampling
    num_inference_steps: int = 50
    guidance_scale: float = 6.0
    eta: float = 0.0
    
    # Randomness
    seed: int | None = None
    generator: torch.Generator | None = None
    
    # Output
    output_path: str = "outputs/"
    save_video: bool = True
    return_frames: bool = False
    
    # I2V specific
    image_path: str | None = None
    
    # Advanced
    latents: torch.Tensor | None = None
    timesteps: list[int] | None = None
    num_videos_per_prompt: int = 1
```

---

## Model Variants

### WAN 2.1 Series

| Model | Size | Resolution | Type | HF Model ID |
|-------|------|------------|------|-------------|
| Wan2.1-T2V-1.3B | 1.3B | 480P (480×832) | Text-to-Video | `Wan-AI/Wan2.1-T2V-1.3B-Diffusers` |
| Wan2.1-T2V-14B | 14B | 720P (720×1280) | Text-to-Video | `Wan-AI/Wan2.1-T2V-14B-Diffusers` |
| Wan2.1-I2V-14B | 14B | 480P (480×832) | Image-to-Video | `Wan-AI/Wan2.1-I2V-14B-480P-Diffusers` |
| Wan2.1-I2V-14B | 14B | 720P (720×1280) | Image-to-Video | `Wan-AI/Wan2.1-I2V-14B-720P-Diffusers` |

**Recommended Settings:**
```python
# Wan2.1-T2V-1.3B
{
    "num_inference_steps": 50,
    "guidance_scale": 6.0,
    "flow_shift": 8.0,  # Override default 3.0
    "height": 480,
    "width": 832,
    "num_frames": 77
}

# Wan2.1-I2V-14B
{
    "num_inference_steps": 40,
    "guidance_scale": 5.0,
    "flow_shift": 3.0,
    "height": 480,
    "width": 832,
    "num_frames": 77
}
```

### WAN 2.2 Series (MoE)

| Model | Size | Resolution | Type | HF Model ID |
|-------|------|------------|------|-------------|
| Wan2.2-T2V-A14B | 14B (MoE) | 720P (720×1280) | Text-to-Video | `Wan-AI/Wan2.2-T2V-A14B-Diffusers` |
| Wan2.2-I2V-A14B | 14B (MoE) | 720P (720×1280) | Image-to-Video | `Wan-AI/Wan2.2-I2V-A14B-Diffusers` |
| Wan2.2-TI2V-5B | 5B | 720P (704×1280) | Text/Image-to-Video | `Wan-AI/Wan2.2-TI2V-5B-Diffusers` |

**Recommended Settings:**
```python
# Wan2.2-T2V-A14B (requires FSDP)
{
    "num_inference_steps": 50,
    "guidance_scale": 6.0,
    "flow_shift": 12.0,  # Higher shift for MoE
    "height": 720,
    "width": 1280,
    "num_frames": 81,
    "use_fsdp_inference": True,
    "dit_cpu_offload": True
}

# Wan2.2-TI2V-5B
{
    "num_inference_steps": 50,
    "guidance_scale": 6.0,
    "flow_shift": 5.0,
    "height": 704,
    "width": 1280,
    "num_frames": 121  # Extended length support
}
```

### FastWAN Series (Distilled)

| Model | Size | Resolution | Steps | HF Model ID |
|-------|------|------------|-------|-------------|
| FastWan2.1-T2V-1.3B | 1.3B | 480P | 4-8 | `FastVideo/FastWan2.1-T2V-1.3B-Diffusers` |
| FastWan2.1-T2V-14B | 14B | 720P | 4-8 | `FastVideo/FastWan2.1-T2V-14B-Diffusers` |
| FastWan2.2-TI2V-5B | 5B | 720P | 4-8 | `FastVideo/FastWan2.2-TI2V-5B-Diffusers` |

**Recommended Settings:**
```python
# FastWan (Distilled models support 4-8 steps)
{
    "num_inference_steps": 8,  # Much fewer steps!
    "guidance_scale": 6.0,
    "flow_shift": 8.0,
    "height": 480,
    "width": 832,
    "num_frames": 77
}
```

**Note:** FastWAN models use sparse distillation and DMD (Distribution Matching Distillation) to achieve >50x speedup compared to the original models.

---

## Performance Optimizations

### Attention Backends

FastVideo supports multiple attention backends for different hardware and use cases:

**1. Flash Attention (Default)**
```python
# Automatically selected for standard inference
generator = VideoGenerator.from_pretrained(model_path, num_gpus=1)
```

**2. Video Sparse Attention (VSA)**
```python
import os
os.environ["FASTVIDEO_ATTENTION_BACKEND"] = "VIDEO_SPARSE_ATTN"

generator = VideoGenerator.from_pretrained(model_path, num_gpus=1)
```

**3. Sliding Tile Attention (STA)**
```python
import os
os.environ["FASTVIDEO_ATTENTION_BACKEND"] = "SLIDING_TILE_ATTN"
os.environ["FASTVIDEO_ATTENTION_CONFIG"] = "path/to/mask_strategy_wan.json"

generator = VideoGenerator.from_pretrained(model_path, num_gpus=1)
```

**4. VMOBA Attention**
```python
import os
os.environ["FASTVIDEO_ATTENTION_BACKEND"] = "VMOBA_ATTN"

generator = VideoGenerator.from_pretrained(model_path, num_gpus=1)
```

### Memory Optimizations

**CPU Offloading:**
```python
generator = VideoGenerator.from_pretrained(
    model_path,
    num_gpus=1,
    dit_cpu_offload=True,      # Offload DiT to CPU between steps
    vae_cpu_offload=True,       # Offload VAE to CPU
    text_encoder_cpu_offload=True,  # Offload T5 to CPU
    pin_cpu_memory=True)        # Pin CPU memory for faster transfers
```

**FSDP Inference (for large models):**
```python
generator = VideoGenerator.from_pretrained(
    "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
    num_gpus=1,
    use_fsdp_inference=True)  # Shard model across GPUs
```

### Parallelism

**Tensor Parallelism (split model across GPUs):**
```python
generator = VideoGenerator.from_pretrained(
    model_path,
    num_gpus=4,
    tp_size=4)  # Split model across 4 GPUs
```

**Sequence Parallelism (split sequence across GPUs):**
```python
generator = VideoGenerator.from_pretrained(
    model_path,
    num_gpus=4,
    sp_size=4)  # Split sequence (temporal) across 4 GPUs
```

**Combined:**
```python
generator = VideoGenerator.from_pretrained(
    model_path,
    num_gpus=8,
    tp_size=2,  # 2-way tensor parallelism
    sp_size=4)  # 4-way sequence parallelism
```

---

## Summary

FastVideo provides a comprehensive and modular framework for video generation with WAN models. The inference pipeline follows a clear flow:

1. **Input** → CLI or Python API
2. **Configuration** → FastVideoArgs + PipelineConfig
3. **Initialization** → VideoGenerator + Executor + Workers
4. **Pipeline Construction** → Load components (VAE, DiT, T5, Scheduler)
5. **Stage Execution:**
   - Input Validation
   - Text Encoding (T5)
   - Conditioning (CFG)
   - Timestep Preparation (FlowUniPC)
   - Latent Preparation (noise initialization)
   - Denoising Loop (DiT + Scheduler)
   - VAE Decoding
6. **Output** → Save video to disk

Key advantages:
- **Modular architecture** enables easy customization
- **Multiple attention backends** for different hardware
- **Flexible parallelism** for scaling to large models
- **Memory optimizations** for limited VRAM
- **Unified API** for all WAN model variants

For more information, see the [FastVideo documentation](https://hao-ai-lab.github.io/FastVideo/).







