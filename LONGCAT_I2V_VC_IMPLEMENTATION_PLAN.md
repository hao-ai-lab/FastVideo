# LongCat I2V and Video Continuation Implementation Plan

**Status**: Implementation Plan  
**Created**: 2024  
**Target**: FastVideo Integration with Original LongCat Approach

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture Analysis](#architecture-analysis)
3. [Phase 1: I2V Implementation](#phase-1-i2v-implementation)
4. [Phase 2: Video Continuation Implementation](#phase-2-video-continuation-implementation)
5. [Testing Strategy](#testing-strategy)
6. [API Design](#api-design)
7. [Implementation Checklist](#implementation-checklist)

---

## Overview

### Goals

Implement two video generation capabilities for LongCat in FastVideo:
1. **I2V (Image-to-Video)**: Generate video from a single input image
2. **VC (Video Continuation)**: Extend video from multiple conditioning frames

### Key Principles

- **Use Original LongCat Approach**: Follow the proven implementation from `/mnt/fast-disks/hao_lab/shao/LongCat-Video`
- **Modular Design**: Implement as separate pipeline stages that compose cleanly
- **Incremental Development**: I2V first (simpler), then VC (adds KV cache)
- **Maintain Compatibility**: Work within existing FastVideo architecture

### Success Criteria

- [ ] I2V matches original LongCat quality and behavior
- [ ] VC achieves 2-3x speedup with KV cache vs. without
- [ ] Code is maintainable and well-documented
- [ ] Integrates seamlessly with existing FastVideo APIs

---

## Architecture Analysis

### Original LongCat I2V/VC Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                       INPUT PROCESSING                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  I2V: Single Image  ──────┐                                   │
│  VC:  Video Frames (13+)  │                                   │
│                           │                                    │
│                           ▼                                    │
│                   [VAE Encoding]                              │
│                           │                                    │
│                           ▼                                    │
│              Image/Video Latents                              │
│              (1 or N frames)                                  │
│                           │                                    │
└───────────────────────────┼────────────────────────────────────┘
                            │
┌───────────────────────────┼────────────────────────────────────┐
│                    LATENT PREPARATION                          │
├───────────────────────────┼────────────────────────────────────┤
│                           │                                    │
│  1. Generate random noise for all frames                      │
│  2. Replace first N frames with conditioned latents           │
│  3. Normalize latents (LongCat-specific)                      │
│  4. Calculate num_cond_latents                                │
│                           │                                    │
│     num_cond_latents = 1 + (num_cond_frames - 1) // 4        │
│     - I2V: 1 frame → num_cond_latents = 1                    │
│     - VC:  13 frames → num_cond_latents = 4                  │
│                           │                                    │
└───────────────────────────┼────────────────────────────────────┘
                            │
┌───────────────────────────┼────────────────────────────────────┐
│              KV CACHE INIT (VC ONLY, OPTIONAL)                 │
├───────────────────────────┼────────────────────────────────────┤
│                           │                                    │
│  IF use_kv_cache == True:                                     │
│    1. Extract conditioning latents                            │
│    2. Run transformer with:                                   │
│       - timestep = 0 (conditioning frames)                    │
│       - skip_crs_attn = True                                  │
│       - return_kv = True                                      │
│    3. Store kv_cache_dict                                     │
│    4. Remove conditioning latents from main latents           │
│                           │                                    │
└───────────────────────────┼────────────────────────────────────┘
                            │
┌───────────────────────────┼────────────────────────────────────┐
│                    DENOISING LOOP                              │
├───────────────────────────┼────────────────────────────────────┤
│                           │                                    │
│  FOR each timestep t in [T, T-1, ..., 1]:                    │
│    1. Expand latents for CFG                                  │
│    2. Prepare timestep tensor                                 │
│                                                               │
│    3. **CRITICAL**: Modify timestep for conditioning:         │
│       timestep = timestep.unsqueeze(-1).repeat(1, num_frames)│
│       IF not use_kv_cache:                                    │
│         timestep[:, :num_cond_latents] = 0  # Mark as fixed  │
│                                                               │
│    4. Run transformer:                                        │
│       noise_pred = transformer(                              │
│         latents,                                             │
│         prompt_embeds,                                       │
│         timestep,                                            │
│         num_cond_latents=num_cond_latents,  # NEW           │
│         kv_cache_dict=kv_cache_dict,         # NEW (VC only)│
│       )                                                       │
│                                                               │
│    5. Apply CFG-zero optimized guidance                      │
│    6. Negate noise_pred (flow matching convention)           │
│                                                               │
│    7. **CRITICAL**: Selective scheduler step:                │
│       IF use_kv_cache:                                        │
│         latents = scheduler.step(noise_pred, t, latents)    │
│       ELSE:                                                   │
│         # Skip conditioning frames                           │
│         latents[:,:,num_cond_latents:] =                     │
│           scheduler.step(                                    │
│             noise_pred[:,:,num_cond_latents:],              │
│             t,                                               │
│             latents[:,:,num_cond_latents:]                  │
│           )                                                  │
│                                                               │
│  IF use_kv_cache:                                             │
│    # Concatenate conditioning latents back                   │
│    latents = cat([cond_latents, latents], dim=2)            │
│                           │                                    │
└───────────────────────────┼────────────────────────────────────┘
                            │
┌───────────────────────────┼────────────────────────────────────┐
│                       VAE DECODING                             │
├───────────────────────────┼────────────────────────────────────┤
│                           │                                    │
│  1. Denormalize latents (reverse normalization)               │
│  2. VAE decode to pixel space                                 │
│  3. Post-process and return video                             │
│                           │                                    │
└───────────────────────────┴────────────────────────────────────┘
```

### Key Differences: I2V vs VC

| Aspect | I2V | VC |
|--------|-----|-----|
| **Input** | 1 image | 13+ video frames |
| **num_cond_frames** | 1 | 13 (or configurable) |
| **num_cond_latents** | 1 | 4 (13 frames ÷ 4 temporal compression + 1) |
| **KV Cache** | Not needed (negligible speedup) | Essential (2-3x speedup) |
| **Timestep masking** | `timestep[:, :1] = 0` | `timestep[:, :4] = 0` (if not using cache) |
| **Denoising** | Skip first 1 frame | Skip first 4 latent frames |

---

## Phase 1: I2V Implementation

### Overview

I2V is the foundation for VC. It establishes:
- VAE encoding pipeline
- `num_cond_latents` parameter threading
- Timestep masking logic
- Selective denoising

### 1.1 Create LongCat I2V Pipeline

**File**: `fastvideo/pipelines/basic/longcat/longcat_i2v_pipeline.py`

```python
class LongCatImageToVideoPipeline(LoRAPipeline, ComposedPipelineBase):
    """
    LongCat Image-to-Video pipeline.
    
    Generates video from a single input image using conditioning.
    """
    
    _required_config_modules = [
        "text_encoder", "tokenizer", "vae", "transformer", "scheduler"
    ]
    
    def initialize_pipeline(self, fastvideo_args: FastVideoArgs):
        """Initialize LongCat-specific settings."""
        # Same as base LongCat T2V pipeline
        pass
    
    def create_pipeline_stages(self, fastvideo_args: FastVideoArgs):
        """Set up I2V-specific pipeline stages."""
        
        # 1. Input validation
        self.add_stage(
            stage_name="input_validation_stage",
            stage=InputValidationStage()
        )
        
        # 2. Text encoding (same as T2V)
        self.add_stage(
            stage_name="prompt_encoding_stage",
            stage=TextEncodingStage(
                text_encoders=[self.get_module("text_encoder")],
                tokenizers=[self.get_module("tokenizer")],
            )
        )
        
        # 3. Image VAE encoding (NEW)
        self.add_stage(
            stage_name="image_vae_encoding_stage",
            stage=LongCatImageVAEEncodingStage(
                vae=self.get_module("vae")
            )
        )
        
        # 4. Timestep preparation
        self.add_stage(
            stage_name="timestep_preparation_stage",
            stage=TimestepPreparationStage(
                scheduler=self.get_module("scheduler")
            )
        )
        
        # 5. Latent preparation with I2V conditioning
        self.add_stage(
            stage_name="latent_preparation_stage",
            stage=LongCatI2VLatentPreparationStage(
                scheduler=self.get_module("scheduler"),
                transformer=self.get_module("transformer")
            )
        )
        
        # 6. Denoising with I2V support
        self.add_stage(
            stage_name="denoising_stage",
            stage=LongCatI2VDenoisingStage(
                transformer=self.get_module("transformer"),
                transformer_2=self.get_module("transformer_2", None),
                scheduler=self.get_module("scheduler"),
                vae=self.get_module("vae"),
                pipeline=self
            )
        )
        
        # 7. Decoding
        self.add_stage(
            stage_name="decoding_stage",
            stage=DecodingStage(
                vae=self.get_module("vae"),
                pipeline=self
            )
        )


EntryClass = LongCatImageToVideoPipeline
```

**Registry Entry**: Add to model registry for auto-detection

### 1.2 Image VAE Encoding Stage

**File**: `fastvideo/pipelines/stages/longcat_image_vae_encoding.py`

```python
class LongCatImageVAEEncodingStage(PipelineStage):
    """
    Encode input image to latent space for I2V conditioning.
    
    This stage:
    1. Preprocesses image to match target dimensions
    2. Encodes via VAE to latent space
    3. Applies LongCat-specific normalization
    4. Stores latent for later conditioning
    """
    
    def __init__(self, vae):
        super().__init__()
        self.vae = vae
    
    def forward(
        self,
        batch: ForwardBatch,
        fastvideo_args: FastVideoArgs,
    ) -> ForwardBatch:
        """Encode image to latent."""
        
        # 1. Get image from batch
        image = batch.pil_image  # PIL.Image
        if image is None:
            raise ValueError("pil_image must be provided for I2V")
        
        # 2. Preprocess image
        # Resize to match height/width
        from fastvideo.models.vision_utils import (
            resize, pil_to_numpy, numpy_to_pt, normalize
        )
        
        height = batch.height
        width = batch.width
        
        image = resize(image, height, width, resize_mode="default")
        image = pil_to_numpy(image)
        image = numpy_to_pt(image)
        image = normalize(image)  # to [-1, 1]
        
        # 3. Add batch and temporal dimensions
        # [C, H, W] -> [1, C, 1, H, W]
        image = image.unsqueeze(0).unsqueeze(2)
        image = image.to(get_local_torch_device(), dtype=torch.float32)
        
        # 4. Encode via VAE
        self.vae = self.vae.to(get_local_torch_device())
        
        with torch.no_grad():
            encoder_output = self.vae.encode(image)
            latent = self.retrieve_latents(encoder_output, batch.generator)
        
        # 5. Apply LongCat normalization
        latent = self.normalize_latents(latent)
        
        # 6. Store in batch
        batch.image_latent = latent
        batch.num_cond_frames = 1
        
        # Calculate num_cond_latents
        vae_temporal_scale = self.vae.temporal_compression_ratio
        batch.num_cond_latents = 1  # For single image, always 1
        
        logger.info(
            f"I2V: Encoded image to latent shape {latent.shape}, "
            f"num_cond_latents={batch.num_cond_latents}"
        )
        
        return batch
    
    def retrieve_latents(self, encoder_output, generator):
        """Sample from VAE posterior."""
        if hasattr(encoder_output, 'latent_dist'):
            return encoder_output.latent_dist.sample(generator)
        elif hasattr(encoder_output, 'latents'):
            return encoder_output.latents
        else:
            raise AttributeError("Could not access latents")
    
    def normalize_latents(self, latents):
        """
        Apply LongCat-specific latent normalization.
        
        Formula: (latents - mean) * (1/std)
        
        This matches the original LongCat implementation.
        """
        latents_mean = torch.tensor(
            self.vae.config.latents_mean
        ).view(1, self.vae.config.z_dim, 1, 1, 1).to(
            latents.device, latents.dtype
        )
        
        latents_std = torch.tensor(
            self.vae.config.latents_std
        ).view(1, self.vae.config.z_dim, 1, 1, 1).to(
            latents.device, latents.dtype
        )
        
        return (latents - latents_mean) / latents_std
```

### 1.3 I2V Latent Preparation Stage

**File**: `fastvideo/pipelines/stages/longcat_i2v_latent_preparation.py`

```python
class LongCatI2VLatentPreparationStage(LatentPreparationStage):
    """
    Prepare latents with image conditioning for first frame.
    
    This stage:
    1. Generates random noise for all frames
    2. Replaces first frame with encoded image latent
    3. Marks conditioning information in batch
    """
    
    def forward(
        self,
        batch: ForwardBatch,
        fastvideo_args: FastVideoArgs,
    ) -> ForwardBatch:
        """Prepare latents with I2V conditioning."""
        
        # 1. Calculate dimensions
        num_frames = batch.num_frames
        height = batch.height
        width = batch.width
        
        vae_temporal_scale = self.transformer.config.arch_config.patch_size[0]
        vae_spatial_scale = 8  # Standard for Wan VAE
        
        num_latent_frames = (num_frames - 1) // vae_temporal_scale + 1
        latent_height = height // vae_spatial_scale
        latent_width = width // vae_spatial_scale
        
        num_channels = self.transformer.config.in_channels
        
        # 2. Generate random noise for all frames
        shape = (
            batch.batch_size,
            num_channels,
            num_latent_frames,
            latent_height,
            latent_width
        )
        
        latents = torch.randn(
            shape,
            generator=batch.generator,
            device=get_local_torch_device(),
            dtype=torch.float32
        )
        
        # 3. Replace first frame with conditioned image latent
        if batch.image_latent is not None:
            num_cond_latents = batch.num_cond_latents
            latents[:, :, :num_cond_latents] = batch.image_latent
            
            logger.info(
                f"I2V: Replaced first {num_cond_latents} latent frames "
                f"with image conditioning"
            )
        
        # 4. Store in batch
        batch.latents = latents
        
        return batch
```

### 1.4 I2V Denoising Stage

**File**: `fastvideo/pipelines/stages/longcat_i2v_denoising.py`

```python
class LongCatI2VDenoisingStage(LongCatDenoisingStage):
    """
    LongCat denoising with I2V conditioning support.
    
    Key modifications from base denoising:
    1. Sets timestep=0 for conditioning frames
    2. Passes num_cond_latents to transformer
    3. Only applies scheduler step to non-conditioned frames
    """
    
    def forward(
        self,
        batch: ForwardBatch,
        fastvideo_args: FastVideoArgs,
    ) -> ForwardBatch:
        """Run denoising loop with I2V conditioning."""
        
        # Load transformer if needed
        if not fastvideo_args.model_loaded["transformer"]:
            from fastvideo.models.model_loader import TransformerLoader
            loader = TransformerLoader()
            self.transformer = loader.load(
                fastvideo_args.model_paths["transformer"],
                fastvideo_args
            )
            fastvideo_args.model_loaded["transformer"] = True
        
        # Setup
        target_dtype = torch.bfloat16
        autocast_enabled = (
            target_dtype != torch.float32
        ) and not fastvideo_args.disable_autocast
        
        latents = batch.latents
        timesteps = batch.timesteps
        prompt_embeds = batch.prompt_embeds[0]
        prompt_attention_mask = batch.prompt_attention_mask[0] if batch.prompt_attention_mask else None
        guidance_scale = batch.guidance_scale
        do_classifier_free_guidance = batch.do_classifier_free_guidance
        
        # Get num_cond_latents from batch
        num_cond_latents = getattr(batch, 'num_cond_latents', 0)
        
        # Prepare negative prompts for CFG
        if do_classifier_free_guidance:
            negative_prompt_embeds = batch.negative_prompt_embeds[0]
            negative_prompt_attention_mask = (
                batch.negative_attention_mask[0]
                if batch.negative_attention_mask else None
            )
            
            prompt_embeds_combined = torch.cat(
                [negative_prompt_embeds, prompt_embeds], dim=0
            )
            if prompt_attention_mask is not None:
                prompt_attention_mask_combined = torch.cat(
                    [negative_prompt_attention_mask, prompt_attention_mask],
                    dim=0
                )
            else:
                prompt_attention_mask_combined = None
        else:
            prompt_embeds_combined = prompt_embeds
            prompt_attention_mask_combined = prompt_attention_mask
        
        # Denoising loop
        num_inference_steps = len(timesteps)
        with tqdm(total=num_inference_steps, desc="I2V Denoising") as progress_bar:
            for i, t in enumerate(timesteps):
                
                # 1. Expand latents for CFG
                if do_classifier_free_guidance:
                    latent_model_input = torch.cat([latents] * 2)
                else:
                    latent_model_input = latents
                
                latent_model_input = latent_model_input.to(target_dtype)
                
                # 2. Expand timestep to match batch size
                timestep = t.expand(latent_model_input.shape[0]).to(target_dtype)
                
                # 3. CRITICAL: Expand timestep to temporal dimension
                # and set conditioning frames to timestep=0
                timestep = timestep.unsqueeze(-1).repeat(
                    1, latent_model_input.shape[2]
                )
                timestep[:, :num_cond_latents] = 0  # Mark conditioning frames
                
                # 4. Run transformer with num_cond_latents
                batch.is_cfg_negative = False
                with set_forward_context(
                    current_timestep=i,
                    attn_metadata=None,
                    forward_batch=batch,
                ), torch.autocast(
                    device_type='cuda',
                    dtype=target_dtype,
                    enabled=autocast_enabled
                ):
                    noise_pred = self.transformer(
                        hidden_states=latent_model_input,
                        encoder_hidden_states=prompt_embeds_combined,
                        timestep=timestep,
                        encoder_attention_mask=prompt_attention_mask_combined,
                        num_cond_latents=num_cond_latents,  # NEW
                    )
                
                # 5. Apply CFG with optimized scale
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                    
                    B = noise_pred_cond.shape[0]
                    positive = noise_pred_cond.reshape(B, -1)
                    negative = noise_pred_uncond.reshape(B, -1)
                    
                    # CFG-zero optimized scale
                    st_star = self.optimized_scale(positive, negative)
                    st_star = st_star.view(B, 1, 1, 1, 1)
                    
                    noise_pred = (
                        noise_pred_uncond * st_star +
                        guidance_scale * (noise_pred_cond - noise_pred_uncond * st_star)
                    )
                
                # 6. CRITICAL: Negate for flow matching scheduler
                noise_pred = -noise_pred
                
                # 7. CRITICAL: Only update non-conditioned frames
                latents[:, :, num_cond_latents:] = self.scheduler.step(
                    noise_pred[:, :, num_cond_latents:],
                    t,
                    latents[:, :, num_cond_latents:],
                    return_dict=False
                )[0]
                
                progress_bar.update()
        
        # Update batch with denoised latents
        batch.latents = latents
        return batch
```

### 1.5 Update Transformer Forward Method

**File**: `fastvideo/models/dits/longcat.py`

**Modification**: Add `num_cond_latents` parameter to forward signature

```python
def forward(
    self,
    hidden_states: torch.Tensor,           # [B, C, T, H, W]
    encoder_hidden_states: torch.Tensor,   # [B, N_text, C_text]
    timestep: torch.LongTensor,            # [B, T]
    encoder_attention_mask: torch.Tensor | None = None,
    num_cond_latents: int | None = None,   # NEW: Number of conditioning latents
    **kwargs
) -> torch.Tensor:
    """
    Forward pass with I2V/VC conditioning support.
    
    Args:
        num_cond_latents: Number of conditioning latent frames.
            - For I2V: 1
            - For VC: 4+ (depends on num_cond_frames)
            - For T2V: None or 0
    """
    # ... existing patch embedding, timestep embedding, caption embedding ...
    
    # Pass num_cond_latents through transformer blocks
    for i, block in enumerate(self.blocks):
        x = block(
            x, context, t,
            latent_shape=(N_t, N_h, N_w),
            num_cond_latents=num_cond_latents  # NEW parameter
        )
    
    # ... final layer ...
    return output
```

### 1.6 Update Attention Blocks

**File**: `fastvideo/models/dits/longcat.py`

**Modification**: Handle conditioning frames in attention

```python
class LongCatAttention:
    """Self-attention with optional conditioning frame support."""
    
    def forward(
        self,
        x: torch.Tensor,
        shape: tuple[int, int, int],
        num_cond_latents: int | None = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Apply self-attention with optional RoPE skipping for conditioning.
        
        Args:
            num_cond_latents: If provided, skip RoPE for first N latent frames
        """
        # Standard attention projection
        q, k, v = self.to_qkv(x)
        
        # If we have conditioning frames, handle them specially
        if num_cond_latents is not None and num_cond_latents > 0:
            T, H, W = shape
            N = x.shape[1]  # Total tokens
            num_cond_tokens = num_cond_latents * (N // T)
            
            # Split into conditioning and noise tokens
            q_cond = q[:, :, :num_cond_tokens]
            k_cond = k[:, :, :num_cond_tokens]
            v_cond = v[:, :, :num_cond_tokens]
            
            q_noise = q[:, :, num_cond_tokens:]
            k_noise = k[:, :, num_cond_tokens:]
            v_noise = v[:, :, num_cond_tokens:]
            
            # Apply RoPE only to noise tokens
            q_noise, k_noise = self.rope_3d(
                q_noise, k_noise,
                shape=(T - num_cond_latents, H, W)
            )
            
            # Recombine
            q = torch.cat([q_cond, q_noise], dim=2)
            k = torch.cat([k_cond, k_noise], dim=2)
            # v doesn't need RoPE, so no modification
        else:
            # Standard path: apply RoPE to all tokens
            q, k = self.rope_3d(q, k, shape=shape)
        
        # Run attention
        out = self.attention_impl(q, k, v)
        
        return out
```

### 1.7 Entry Point Registration

**File**: Create new config or update registry

```python
# In model registry or auto-detection logic
MODEL_PIPELINE_MAPPING = {
    # ... existing mappings ...
    "longcat-i2v": "fastvideo.pipelines.basic.longcat.longcat_i2v_pipeline",
}
```

---

## Phase 2: Video Continuation Implementation

### Overview

VC builds on I2V by:
1. Encoding multiple video frames instead of one image
2. Implementing KV cache for performance
3. Managing cache lifecycle

### 2.1 Create LongCat VC Pipeline

**File**: `fastvideo/pipelines/basic/longcat/longcat_vc_pipeline.py`

```python
class LongCatVideoContinuationPipeline(LoRAPipeline, ComposedPipelineBase):
    """
    LongCat Video Continuation pipeline.
    
    Extends video from multiple conditioning frames using KV cache optimization.
    """
    
    _required_config_modules = [
        "text_encoder", "tokenizer", "vae", "transformer", "scheduler"
    ]
    
    def create_pipeline_stages(self, fastvideo_args: FastVideoArgs):
        """Set up VC-specific pipeline stages."""
        
        # 1. Input validation
        self.add_stage(
            stage_name="input_validation_stage",
            stage=InputValidationStage()
        )
        
        # 2. Text encoding
        self.add_stage(
            stage_name="prompt_encoding_stage",
            stage=TextEncodingStage(
                text_encoders=[self.get_module("text_encoder")],
                tokenizers=[self.get_module("tokenizer")],
            )
        )
        
        # 3. Video VAE encoding (NEW - different from I2V)
        self.add_stage(
            stage_name="video_vae_encoding_stage",
            stage=LongCatVideoVAEEncodingStage(
                vae=self.get_module("vae")
            )
        )
        
        # 4. Timestep preparation
        self.add_stage(
            stage_name="timestep_preparation_stage",
            stage=TimestepPreparationStage(
                scheduler=self.get_module("scheduler")
            )
        )
        
        # 5. Latent preparation with VC conditioning
        self.add_stage(
            stage_name="latent_preparation_stage",
            stage=LongCatVCLatentPreparationStage(
                scheduler=self.get_module("scheduler"),
                transformer=self.get_module("transformer")
            )
        )
        
        # 6. KV Cache initialization (OPTIONAL, NEW)
        if fastvideo_args.pipeline_config.enable_kv_cache:
            self.add_stage(
                stage_name="kv_cache_init_stage",
                stage=LongCatKVCacheInitStage(
                    transformer=self.get_module("transformer"),
                    text_encoder=self.get_module("text_encoder")
                )
            )
        
        # 7. Denoising with VC and KV cache support
        self.add_stage(
            stage_name="denoising_stage",
            stage=LongCatVCDenoisingStage(
                transformer=self.get_module("transformer"),
                transformer_2=self.get_module("transformer_2", None),
                scheduler=self.get_module("scheduler"),
                vae=self.get_module("vae"),
                pipeline=self
            )
        )
        
        # 8. Decoding
        self.add_stage(
            stage_name="decoding_stage",
            stage=DecodingStage(
                vae=self.get_module("vae"),
                pipeline=self
            )
        )


EntryClass = LongCatVideoContinuationPipeline
```

### 2.2 Video VAE Encoding Stage

**File**: `fastvideo/pipelines/stages/longcat_video_vae_encoding.py`

```python
class LongCatVideoVAEEncodingStage(PipelineStage):
    """
    Encode video frames to latent space for VC conditioning.
    
    Differences from I2V:
    - Encodes multiple frames (e.g., 13)
    - Takes last N frames from input video
    - Calculates larger num_cond_latents
    """
    
    def __init__(self, vae):
        super().__init__()
        self.vae = vae
    
    def forward(
        self,
        batch: ForwardBatch,
        fastvideo_args: FastVideoArgs,
    ) -> ForwardBatch:
        """Encode video frames to latent."""
        
        # 1. Get video frames from batch
        video_frames = batch.video_latent  # List[PIL.Image] or video path
        num_cond_frames = batch.num_cond_frames  # e.g., 13
        
        if video_frames is None:
            raise ValueError("video_latent must be provided for VC")
        
        # 2. Load and preprocess video
        if isinstance(video_frames, str):
            # Load from video path
            from diffusers.utils import load_video
            video_frames = load_video(video_frames)
        
        # Take last num_cond_frames frames
        video_frames = video_frames[-num_cond_frames:]
        
        # 3. Preprocess video frames
        from fastvideo.models.vision_utils import (
            resize, pil_to_numpy, numpy_to_pt, normalize
        )
        
        height = batch.height
        width = batch.width
        
        processed_frames = []
        for frame in video_frames:
            frame = resize(frame, height, width, resize_mode="default")
            frame = pil_to_numpy(frame)
            frame = numpy_to_pt(frame)
            frame = normalize(frame)
            processed_frames.append(frame.unsqueeze(0))
        
        # Stack frames: [num_frames, C, H, W]
        video_tensor = torch.stack(processed_frames, dim=2)  # [1, C, T, H, W]
        video_tensor = video_tensor.to(get_local_torch_device(), dtype=torch.float32)
        
        # 4. Encode via VAE
        self.vae = self.vae.to(get_local_torch_device())
        
        with torch.no_grad():
            encoder_output = self.vae.encode(video_tensor)
            latent = self.retrieve_latents(encoder_output, batch.generator)
        
        # 5. Apply LongCat normalization
        latent = self.normalize_latents(latent)
        
        # 6. Calculate num_cond_latents
        vae_temporal_scale = self.vae.temporal_compression_ratio
        num_cond_latents = 1 + (num_cond_frames - 1) // vae_temporal_scale
        
        # 7. Store in batch
        batch.video_latent = latent
        batch.num_cond_latents = num_cond_latents
        
        logger.info(
            f"VC: Encoded {num_cond_frames} frames to latent shape {latent.shape}, "
            f"num_cond_latents={num_cond_latents}"
        )
        
        return batch
    
    def retrieve_latents(self, encoder_output, generator):
        """Sample from VAE posterior."""
        if hasattr(encoder_output, 'latent_dist'):
            return encoder_output.latent_dist.sample(generator)
        elif hasattr(encoder_output, 'latents'):
            return encoder_output.latents
        else:
            raise AttributeError("Could not access latents")
    
    def normalize_latents(self, latents):
        """Apply LongCat-specific latent normalization."""
        latents_mean = torch.tensor(
            self.vae.config.latents_mean
        ).view(1, self.vae.config.z_dim, 1, 1, 1).to(
            latents.device, latents.dtype
        )
        
        latents_std = torch.tensor(
            self.vae.config.latents_std
        ).view(1, self.vae.config.z_dim, 1, 1, 1).to(
            latents.device, latents.dtype
        )
        
        return (latents - latents_mean) / latents_std
```

### 2.3 VC Latent Preparation Stage

**File**: `fastvideo/pipelines/stages/longcat_vc_latent_preparation.py`

```python
class LongCatVCLatentPreparationStage(LatentPreparationStage):
    """
    Prepare latents with video conditioning for first N frames.
    
    Similar to I2V but handles multiple conditioning frames.
    """
    
    def forward(
        self,
        batch: ForwardBatch,
        fastvideo_args: FastVideoArgs,
    ) -> ForwardBatch:
        """Prepare latents with VC conditioning."""
        
        # 1. Calculate dimensions
        num_frames = batch.num_frames
        height = batch.height
        width = batch.width
        
        vae_temporal_scale = 4  # Standard for Wan VAE
        vae_spatial_scale = 8
        
        num_latent_frames = (num_frames - 1) // vae_temporal_scale + 1
        latent_height = height // vae_spatial_scale
        latent_width = width // vae_spatial_scale
        
        num_channels = self.transformer.config.in_channels
        
        # 2. Generate random noise for all frames
        shape = (
            batch.batch_size,
            num_channels,
            num_latent_frames,
            latent_height,
            latent_width
        )
        
        latents = torch.randn(
            shape,
            generator=batch.generator,
            device=get_local_torch_device(),
            dtype=torch.float32
        )
        
        # 3. Replace first N frames with conditioned video latents
        if batch.video_latent is not None:
            num_cond_latents = batch.num_cond_latents
            latents[:, :, :num_cond_latents] = batch.video_latent
            
            logger.info(
                f"VC: Replaced first {num_cond_latents} latent frames "
                f"with video conditioning (from {batch.num_cond_frames} frames)"
            )
        
        # 4. Store in batch
        batch.latents = latents
        
        return batch
```

### 2.4 KV Cache Initialization Stage

**File**: `fastvideo/pipelines/stages/longcat_kv_cache_init.py`

```python
class LongCatKVCacheInitStage(PipelineStage):
    """
    Pre-compute KV cache for conditioning frames.
    
    This is the KEY optimization for VC:
    - Runs transformer once on conditioning frames
    - Caches K/V tensors for all blocks
    - Reuses cache at every denoising step
    - Achieves 2-3x speedup
    """
    
    def __init__(self, transformer, text_encoder):
        super().__init__()
        self.transformer = transformer
        self.text_encoder = text_encoder
    
    def forward(
        self,
        batch: ForwardBatch,
        fastvideo_args: FastVideoArgs,
    ) -> ForwardBatch:
        """Initialize KV cache from conditioning latents."""
        
        # Check if KV cache is enabled
        if not fastvideo_args.pipeline_config.enable_kv_cache:
            batch.kv_cache_dict = {}
            batch.use_kv_cache = False
            logger.info("KV cache disabled, skipping initialization")
            return batch
        
        batch.use_kv_cache = True
        offload_kv_cache = fastvideo_args.pipeline_config.offload_kv_cache
        
        # 1. Extract conditioning latents
        num_cond_latents = batch.num_cond_latents
        cond_latents = batch.latents[:, :, :num_cond_latents]
        
        logger.info(
            f"Initializing KV cache for {num_cond_latents} conditioning latents"
        )
        
        # 2. Prepare timestep tensor (all zeros for conditioning)
        timestep = torch.zeros(
            cond_latents.shape[0],
            cond_latents.shape[2],
            device=cond_latents.device,
            dtype=torch.float32
        )
        
        # 3. Create dummy prompt embeddings (will be skipped anyway)
        max_sequence_length = 512  # Standard for T5
        empty_embeds = torch.zeros(
            [cond_latents.shape[0], 1, max_sequence_length,
             self.text_encoder.config.d_model],
            device=cond_latents.device,
            dtype=self.transformer.dtype
        )
        
        # 4. Run transformer with special flags
        # - return_kv=True: Return K/V cache
        # - skip_crs_attn=True: Skip cross-attention (not needed for cache)
        # - offload_kv_cache: Whether to offload to CPU
        
        with torch.no_grad():
            _, kv_cache_dict = self.transformer(
                hidden_states=cond_latents,
                timestep=timestep,
                encoder_hidden_states=empty_embeds,
                return_kv=True,
                skip_crs_attn=True,
                offload_kv_cache=offload_kv_cache
            )
        
        # 5. Store cache in batch
        batch.kv_cache_dict = kv_cache_dict
        batch.cond_latents = cond_latents  # Save for later concatenation
        
        # 6. Remove conditioning latents from main latents
        # (they're already in the cache)
        batch.latents = batch.latents[:, :, num_cond_latents:]
        
        logger.info(
            f"KV cache initialized: {len(kv_cache_dict)} blocks, "
            f"offload={offload_kv_cache}"
        )
        
        return batch
```

### 2.5 VC Denoising Stage

**File**: `fastvideo/pipelines/stages/longcat_vc_denoising.py`

```python
class LongCatVCDenoisingStage(LongCatI2VDenoisingStage):
    """
    LongCat denoising with VC and KV cache support.
    
    Key differences from I2V:
    - Supports KV cache (reuses cached K/V)
    - Handles larger num_cond_latents
    - Concatenates conditioning latents back after denoising
    """
    
    def forward(
        self,
        batch: ForwardBatch,
        fastvideo_args: FastVideoArgs,
    ) -> ForwardBatch:
        """Run denoising loop with VC conditioning and KV cache."""
        
        # Load transformer if needed
        if not fastvideo_args.model_loaded["transformer"]:
            from fastvideo.models.model_loader import TransformerLoader
            loader = TransformerLoader()
            self.transformer = loader.load(
                fastvideo_args.model_paths["transformer"],
                fastvideo_args
            )
            fastvideo_args.model_loaded["transformer"] = True
        
        # Setup
        target_dtype = torch.bfloat16
        autocast_enabled = (
            target_dtype != torch.float32
        ) and not fastvideo_args.disable_autocast
        
        latents = batch.latents
        timesteps = batch.timesteps
        prompt_embeds = batch.prompt_embeds[0]
        prompt_attention_mask = batch.prompt_attention_mask[0] if batch.prompt_attention_mask else None
        guidance_scale = batch.guidance_scale
        do_classifier_free_guidance = batch.do_classifier_free_guidance
        
        # Get VC-specific parameters
        num_cond_latents = getattr(batch, 'num_cond_latents', 0)
        use_kv_cache = getattr(batch, 'use_kv_cache', False)
        kv_cache_dict = getattr(batch, 'kv_cache_dict', {})
        
        logger.info(
            f"VC Denoising: num_cond_latents={num_cond_latents}, "
            f"use_kv_cache={use_kv_cache}"
        )
        
        # Prepare negative prompts for CFG
        if do_classifier_free_guidance:
            negative_prompt_embeds = batch.negative_prompt_embeds[0]
            negative_prompt_attention_mask = (
                batch.negative_attention_mask[0]
                if batch.negative_attention_mask else None
            )
            
            prompt_embeds_combined = torch.cat(
                [negative_prompt_embeds, prompt_embeds], dim=0
            )
            if prompt_attention_mask is not None:
                prompt_attention_mask_combined = torch.cat(
                    [negative_prompt_attention_mask, prompt_attention_mask],
                    dim=0
                )
            else:
                prompt_attention_mask_combined = None
        else:
            prompt_embeds_combined = prompt_embeds
            prompt_attention_mask_combined = prompt_attention_mask
        
        # Denoising loop
        num_inference_steps = len(timesteps)
        with tqdm(total=num_inference_steps, desc="VC Denoising") as progress_bar:
            for i, t in enumerate(timesteps):
                
                # 1. Expand latents for CFG
                if do_classifier_free_guidance:
                    latent_model_input = torch.cat([latents] * 2)
                else:
                    latent_model_input = latents
                
                latent_model_input = latent_model_input.to(target_dtype)
                
                # 2. Expand timestep to match batch size
                timestep = t.expand(latent_model_input.shape[0]).to(target_dtype)
                
                # 3. Expand timestep to temporal dimension
                timestep = timestep.unsqueeze(-1).repeat(
                    1, latent_model_input.shape[2]
                )
                
                # 4. CRITICAL: Handle timestep masking
                if not use_kv_cache:
                    # If not using cache, mask conditioning frame timesteps
                    timestep[:, :num_cond_latents] = 0
                # If using cache, all frames in latents are noise frames
                # (conditioning is in cache), so no masking needed
                
                # 5. Prepare transformer kwargs
                transformer_kwargs = {
                    'num_cond_latents': num_cond_latents if not use_kv_cache else None,
                }
                
                if use_kv_cache:
                    transformer_kwargs['kv_cache_dict'] = kv_cache_dict
                
                # 6. Run transformer
                batch.is_cfg_negative = False
                with set_forward_context(
                    current_timestep=i,
                    attn_metadata=None,
                    forward_batch=batch,
                ), torch.autocast(
                    device_type='cuda',
                    dtype=target_dtype,
                    enabled=autocast_enabled
                ):
                    noise_pred = self.transformer(
                        hidden_states=latent_model_input,
                        encoder_hidden_states=prompt_embeds_combined,
                        timestep=timestep,
                        encoder_attention_mask=prompt_attention_mask_combined,
                        **transformer_kwargs
                    )
                
                # 7. Apply CFG with optimized scale
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                    
                    B = noise_pred_cond.shape[0]
                    positive = noise_pred_cond.reshape(B, -1)
                    negative = noise_pred_uncond.reshape(B, -1)
                    
                    st_star = self.optimized_scale(positive, negative)
                    st_star = st_star.view(B, 1, 1, 1, 1)
                    
                    noise_pred = (
                        noise_pred_uncond * st_star +
                        guidance_scale * (noise_pred_cond - noise_pred_uncond * st_star)
                    )
                
                # 8. Negate for flow matching scheduler
                noise_pred = -noise_pred
                
                # 9. Scheduler step
                if use_kv_cache:
                    # All frames in latents are noise frames
                    latents = self.scheduler.step(
                        noise_pred, t, latents, return_dict=False
                    )[0]
                else:
                    # Skip conditioning frames
                    latents[:, :, num_cond_latents:] = self.scheduler.step(
                        noise_pred[:, :, num_cond_latents:],
                        t,
                        latents[:, :, num_cond_latents:],
                        return_dict=False
                    )[0]
                
                progress_bar.update()
        
        # 10. If using KV cache, concatenate conditioning latents back
        if use_kv_cache:
            latents = torch.cat([batch.cond_latents, latents], dim=2)
            logger.info("Concatenated conditioning latents back to output")
        
        # Update batch with denoised latents
        batch.latents = latents
        return batch
```

### 2.6 Update Transformer for KV Cache

**File**: `fastvideo/models/dits/longcat.py`

**Modifications**: Add KV cache support to forward method

```python
def forward(
    self,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    timestep: torch.LongTensor,
    encoder_attention_mask: torch.Tensor | None = None,
    num_cond_latents: int | None = None,
    # NEW parameters for KV cache
    return_kv: bool = False,
    kv_cache_dict: dict | None = None,
    skip_crs_attn: bool = False,
    offload_kv_cache: bool = False,
    **kwargs
) -> torch.Tensor | tuple[torch.Tensor, dict]:
    """
    Forward pass with KV cache support.
    
    Args:
        return_kv: If True, return (output, kv_cache_dict)
        kv_cache_dict: Pre-computed K/V cache {block_idx: (k, v)}
        skip_crs_attn: If True, skip cross-attention (for cache init)
        offload_kv_cache: If True, offload cache to CPU
    """
    B, _, T, H, W = hidden_states.shape
    
    N_t = T // self.patch_size[0]
    N_h = H // self.patch_size[1]
    N_w = W // self.patch_size[2]
    
    # ... existing patch embedding, timestep embedding, caption embedding ...
    
    # Pass through blocks with KV cache
    kv_cache_dict_ret = {}
    
    for i, block in enumerate(self.blocks):
        # Get cache for this block if available
        block_kv_cache = kv_cache_dict.get(i, None) if kv_cache_dict else None
        
        # Forward through block
        block_outputs = block(
            x, context, t,
            latent_shape=(N_t, N_h, N_w),
            num_cond_latents=num_cond_latents,
            return_kv=return_kv,
            kv_cache=block_kv_cache,
            skip_crs_attn=skip_crs_attn
        )
        
        # Handle return value
        if return_kv:
            x, kv_cache = block_outputs
            # Store cache
            if offload_kv_cache:
                kv_cache_dict_ret[i] = (
                    kv_cache[0].cpu(),
                    kv_cache[1].cpu()
                )
            else:
                kv_cache_dict_ret[i] = kv_cache
        else:
            x = block_outputs
    
    # ... final layer ...
    
    if return_kv:
        return output, kv_cache_dict_ret
    else:
        return output
```

### 2.7 Update Attention Block for KV Cache

**File**: `fastvideo/models/dits/longcat.py`

**Modifications**: Add KV cache handling to attention

```python
class LongCatTransformerBlock:
    """Transformer block with KV cache support."""
    
    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,  # context
        t: torch.Tensor,  # timestep embedding
        latent_shape: tuple[int, int, int],
        num_cond_latents: int | None = None,
        return_kv: bool = False,
        kv_cache: tuple[torch.Tensor, torch.Tensor] | None = None,
        skip_crs_attn: bool = False
    ):
        """
        Forward with optional KV cache.
        
        Args:
            kv_cache: (k_cache, v_cache) from previous computation
            return_kv: If True, return (x, (k, v))
            skip_crs_attn: If True, skip cross-attention
        """
        # ... modulation ...
        
        # Self-attention with KV cache
        if kv_cache is not None:
            # Move cache to device if offloaded
            kv_cache = (
                kv_cache[0].to(x.device),
                kv_cache[1].to(x.device)
            )
            attn_outputs = self.self_attn.forward_with_kv_cache(
                x_m,
                shape=latent_shape,
                kv_cache=kv_cache
            )
        else:
            attn_outputs = self.self_attn(
                x_m,
                shape=latent_shape,
                num_cond_latents=num_cond_latents,
                return_kv=return_kv
            )
        
        if return_kv:
            x_s, kv_cache_new = attn_outputs
        else:
            x_s = attn_outputs
        
        x = x + x_s
        
        # Cross-attention (skip if requested)
        if not skip_crs_attn:
            x = x + self.cross_attn(x, y, ...)
        
        # FFN
        x = x + self.ffn(x)
        
        if return_kv:
            return x, kv_cache_new
        else:
            return x


class LongCatAttention:
    """Self-attention with KV cache support."""
    
    def forward_with_kv_cache(
        self,
        x: torch.Tensor,
        shape: tuple[int, int, int],
        kv_cache: tuple[torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        """
        Attention forward using pre-computed KV cache.
        
        Cache contains K/V for conditioning frames.
        New K/V is computed for noise frames.
        """
        k_cache, v_cache = kv_cache
        
        # Compute Q/K/V for noise frames
        q, k, v = self.to_qkv(x)
        
        # Apply RoPE to new K/V (for noise frames only)
        # Note: Cache already has RoPE applied (or skipped for conditioning)
        T, H, W = shape
        T_cache = k_cache.shape[2] // (H * W)  # Infer from cache shape
        T_new = T - T_cache
        
        q, k = self.rope_3d(q, k, shape=(T_new, H, W))
        
        # Concatenate cached K/V with new K/V
        k_full = torch.cat([k_cache, k], dim=2)
        v_full = torch.cat([v_cache, v], dim=2)
        
        # Run attention with full K/V
        out = self.attention_impl(q, k_full, v_full)
        
        return out
```

---

## Testing Strategy

### Unit Tests

#### 1. VAE Encoding Tests

```python
def test_image_vae_encoding():
    """Test single image encoding for I2V."""
    # Create test image
    # Run encoding stage
    # Verify output shape and normalization
    pass

def test_video_vae_encoding():
    """Test multi-frame encoding for VC."""
    # Create test video
    # Run encoding stage
    # Verify num_cond_latents calculation
    pass
```

#### 2. Latent Preparation Tests

```python
def test_i2v_latent_preparation():
    """Test latent preparation with image conditioning."""
    # Create mock image latent
    # Run preparation stage
    # Verify first frame is replaced
    pass

def test_vc_latent_preparation():
    """Test latent preparation with video conditioning."""
    # Create mock video latent
    # Run preparation stage
    # Verify first N frames are replaced
    pass
```

#### 3. KV Cache Tests

```python
def test_kv_cache_initialization():
    """Test KV cache pre-computation."""
    # Create conditioning latents
    # Run cache init stage
    # Verify cache structure and contents
    pass

def test_kv_cache_usage():
    """Test attention with KV cache."""
    # Create cache
    # Run attention forward_with_kv_cache
    # Verify output matches non-cached version
    pass
```

### Integration Tests

#### 1. End-to-End I2V

```python
def test_i2v_pipeline():
    """Test complete I2V pipeline."""
    model_path = "weights/longcat-native"
    image_path = "test_image.jpg"
    
    generator = VideoGenerator.from_pretrained(model_path, task="i2v")
    
    video = generator.generate_video(
        prompt="A dog running",
        image_path=image_path,
        num_frames=93,
        height=480,
        width=832,
    )
    
    assert video.shape == (93, 480, 832, 3)
```

#### 2. End-to-End VC

```python
def test_vc_pipeline_without_cache():
    """Test VC without KV cache."""
    model_path = "weights/longcat-native"
    video_path = "test_video.mp4"
    
    generator = VideoGenerator.from_pretrained(model_path, task="vc")
    
    video = generator.generate_video_continuation(
        video_path=video_path,
        prompt="The video continues...",
        num_cond_frames=13,
        num_frames=93,
        use_kv_cache=False,
    )
    
    assert video.shape == (93, 480, 832, 3)

def test_vc_pipeline_with_cache():
    """Test VC with KV cache."""
    # Same as above but with use_kv_cache=True
    # Should produce same output but faster
    pass
```

### Comparison Tests

#### 1. I2V Quality Comparison

```python
def test_i2v_vs_original():
    """Compare I2V output with original LongCat implementation."""
    # Run FastVideo I2V
    # Run original LongCat-Video I2V
    # Compare PSNR/SSIM
    # Verify close match (allowing for numerical differences)
    pass
```

#### 2. VC Cache Performance

```python
def test_vc_cache_speedup():
    """Verify KV cache provides speedup."""
    import time
    
    # Run without cache
    start = time.time()
    video_no_cache = run_vc(use_kv_cache=False)
    time_no_cache = time.time() - start
    
    # Run with cache
    start = time.time()
    video_with_cache = run_vc(use_kv_cache=True)
    time_with_cache = time.time() - start
    
    # Verify speedup
    speedup = time_no_cache / time_with_cache
    assert speedup >= 2.0, f"Expected 2x+ speedup, got {speedup:.2f}x"
    
    # Verify outputs match
    assert torch.allclose(video_no_cache, video_with_cache, atol=1e-3)
```

### Performance Benchmarks

```python
def benchmark_i2v():
    """Benchmark I2V performance."""
    # Measure: throughput, memory, latency
    pass

def benchmark_vc():
    """Benchmark VC performance with/without cache."""
    # Compare cache vs no-cache
    # Measure memory overhead of cache
    pass
```

---

## API Design

### High-Level API

```python
from fastvideo import VideoGenerator

# Load model
generator = VideoGenerator.from_pretrained(
    "weights/longcat-native",
    num_gpus=2,
)

# =========================
# I2V: Image-to-Video
# =========================

video = generator.generate_video(
    prompt="A dog running through a field",
    image_path="dog.jpg",  # Triggers I2V mode
    negative_prompt="blurry, low quality",
    num_frames=93,
    height=480,
    width=832,
    num_inference_steps=50,
    guidance_scale=4.0,
    seed=42,
)

# =========================
# VC: Video Continuation
# =========================

video = generator.generate_video(
    prompt="The motorcycle continues down the road",
    video_path="motorcycle.mp4",  # Triggers VC mode
    num_cond_frames=13,  # Number of frames to condition on
    num_frames=93,  # Total frames to generate
    use_kv_cache=True,  # Enable KV cache (recommended)
    offload_kv_cache=False,  # Keep cache on GPU
    num_inference_steps=50,
    guidance_scale=4.0,
)

# =========================
# Long Video Generation
# (Autoregressive VC)
# =========================

segments = []
current_video = initial_video

for i in range(10):  # Generate 10 segments
    segment = generator.generate_video(
        prompt=prompt,
        video_path=current_video,
        num_cond_frames=13,
        num_frames=93,
        use_kv_cache=True,
    )
    
    # Keep last 13 frames for next segment
    segments.append(segment[13:])
    current_video = segment

# Concatenate all segments
long_video = concatenate_segments(segments)
```

### Low-Level API

```python
from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.pipelines.basic.longcat import (
    LongCatImageToVideoPipeline,
    LongCatVideoContinuationPipeline
)

# I2V with custom args
args = FastVideoArgs(
    model_path="weights/longcat-native",
    pipeline_config={
        "enable_bsa": True,
        "enable_kv_cache": False,  # N/A for I2V
    }
)

pipeline = LongCatImageToVideoPipeline.from_pretrained(args)
output = pipeline.forward(batch)

# VC with custom args
args = FastVideoArgs(
    model_path="weights/longcat-native",
    pipeline_config={
        "enable_kv_cache": True,
        "offload_kv_cache": False,
    }
)

pipeline = LongCatVideoContinuationPipeline.from_pretrained(args)
output = pipeline.forward(batch)
```

### CLI Interface

```bash
# I2V
fastvideo generate \
    --model-path weights/longcat-native \
    --task i2v \
    --image-path dog.jpg \
    --prompt "A dog running" \
    --num-frames 93 \
    --height 480 \
    --width 832 \
    --output-path outputs/i2v_dog.mp4

# VC
fastvideo generate \
    --model-path weights/longcat-native \
    --task vc \
    --video-path motorcycle.mp4 \
    --num-cond-frames 13 \
    --prompt "Continuing down the road" \
    --num-frames 93 \
    --use-kv-cache \
    --output-path outputs/vc_motorcycle.mp4
```

---

## Implementation Checklist

### Phase 1: I2V Implementation

#### Core Components
- [ ] Create `LongCatImageToVideoPipeline`
- [ ] Implement `LongCatImageVAEEncodingStage`
- [ ] Implement `LongCatI2VLatentPreparationStage`
- [ ] Implement `LongCatI2VDenoisingStage`
- [ ] Update `LongCatTransformer3DModel.forward()` with `num_cond_latents` parameter
- [ ] Update `LongCatAttention` to skip RoPE for conditioning frames
- [ ] Update `LongCatTransformerBlock` to pass `num_cond_latents`

#### Integration
- [ ] Add I2V pipeline to model registry
- [ ] Add I2V task detection logic
- [ ] Implement high-level API (`generate_video` with `image_path`)
- [ ] Add CLI support for I2V

#### Testing
- [ ] Unit test: Image VAE encoding
- [ ] Unit test: I2V latent preparation
- [ ] Unit test: Attention with num_cond_latents
- [ ] Integration test: End-to-end I2V pipeline
- [ ] Comparison test: I2V vs original LongCat

#### Documentation
- [ ] API documentation for I2V
- [ ] Example scripts for I2V
- [ ] Update README with I2V usage

### Phase 2: VC Implementation

#### Core Components
- [ ] Create `LongCatVideoContinuationPipeline`
- [ ] Implement `LongCatVideoVAEEncodingStage`
- [ ] Implement `LongCatVCLatentPreparationStage`
- [ ] Implement `LongCatKVCacheInitStage`
- [ ] Implement `LongCatVCDenoisingStage`
- [ ] Update transformer forward with KV cache parameters
- [ ] Implement `LongCatAttention.forward_with_kv_cache()`
- [ ] Update blocks to support `return_kv`, `kv_cache`, `skip_crs_attn`

#### Integration
- [ ] Add VC pipeline to model registry
- [ ] Add VC task detection logic
- [ ] Implement high-level API (`generate_video_continuation`)
- [ ] Add CLI support for VC

#### Testing
- [ ] Unit test: Video VAE encoding
- [ ] Unit test: VC latent preparation
- [ ] Unit test: KV cache initialization
- [ ] Unit test: Attention with KV cache
- [ ] Integration test: VC without cache
- [ ] Integration test: VC with cache
- [ ] Performance test: Cache speedup (2x+ expected)
- [ ] Comparison test: VC vs original LongCat

#### Documentation
- [ ] API documentation for VC
- [ ] Example scripts for VC
- [ ] Example: Long video generation (autoregressive)
- [ ] Update README with VC usage

### Phase 3: Optimization & Polish

#### Performance
- [ ] Profile KV cache memory usage
- [ ] Optimize cache offloading if needed
- [ ] Benchmark I2V vs T2V performance
- [ ] Benchmark VC with different num_cond_frames

#### Quality
- [ ] Test I2V with various images
- [ ] Test VC with different video types
- [ ] Verify quality matches original
- [ ] Add quality comparison metrics (PSNR, SSIM, LPIPS)

#### Robustness
- [ ] Handle edge cases (very short videos, very long videos)
- [ ] Add input validation
- [ ] Improve error messages
- [ ] Add recovery for OOM errors

#### Documentation
- [ ] Architecture documentation
- [ ] Implementation notes
- [ ] Troubleshooting guide
- [ ] Performance tuning guide

---

## Timeline Estimate

| Phase | Tasks | Estimated Time |
|-------|-------|----------------|
| **Phase 1: I2V** | Core implementation | 3-4 days |
| | Testing & debugging | 2-3 days |
| | Documentation | 1 day |
| **Phase 2: VC** | Core implementation | 4-5 days |
| | KV cache implementation | 2-3 days |
| | Testing & debugging | 2-3 days |
| | Documentation | 1 day |
| **Phase 3: Polish** | Optimization | 2-3 days |
| | Quality testing | 2 days |
| | Final documentation | 1 day |
| **Total** | | **20-28 days** |

---

## Success Metrics

### Correctness
- [ ] I2V output visually matches original LongCat-Video
- [ ] VC output visually matches original LongCat-Video
- [ ] PSNR > 40dB compared to original (accounting for numerical precision)
- [ ] All unit tests pass
- [ ] All integration tests pass

### Performance
- [ ] I2V completes in reasonable time (similar to T2V)
- [ ] VC with cache achieves 2-3x speedup vs without cache
- [ ] Memory usage is acceptable (no OOM on target hardware)
- [ ] KV cache overhead < 20% of total memory

### Usability
- [ ] Simple API for common use cases
- [ ] Clear documentation with examples
- [ ] Good error messages
- [ ] Works with existing FastVideo infrastructure

### Maintainability
- [ ] Clean, modular code
- [ ] Well-documented implementation
- [ ] Follows FastVideo conventions
- [ ] Easy to extend or modify

---

## Notes & Considerations

### Implementation Priorities

1. **I2V First**: Establishes foundation, simpler to implement and test
2. **VC Without Cache**: Proves correctness before optimization
3. **KV Cache**: Add performance optimization last

### Critical Details

1. **Normalization**: Must use LongCat-specific normalization (not standard scaling)
2. **Timestep Masking**: Conditioning frames must have timestep=0
3. **Selective Denoising**: Only apply scheduler to non-conditioned frames
4. **RoPE Handling**: Skip RoPE for conditioning frames
5. **Flow Matching**: Must negate noise_pred before scheduler step

### Potential Issues

1. **Memory**: VC with many conditioning frames uses more memory
2. **Cache Overhead**: KV cache adds ~15-20% memory overhead
3. **Numerical Precision**: May see small differences vs original due to precision
4. **VAE Encoding**: Must handle video loading and preprocessing correctly

### Future Extensions

1. **Enhanced HF Schedule**: Better timestep schedule for VC
2. **Autoregressive Long Video**: Chain multiple VC calls
3. **Refinement Support**: Integrate with LongCat refinement stage
4. **Multi-GPU VC**: Distribute cache across GPUs

---

## References

- Original LongCat-Video: `/mnt/fast-disks/hao_lab/shao/LongCat-Video`
- FastVideo Causal WAN: `fastvideo/models/dits/causal_wanvideo.py`
- FastVideo WAN I2V: `fastvideo/pipelines/basic/wan/wan_i2v_pipeline.py`
- LongCat Config: `fastvideo/configs/pipelines/longcat.py`

---

**End of Implementation Plan**

