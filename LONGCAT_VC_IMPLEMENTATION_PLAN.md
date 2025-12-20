# LongCat Video Continuation (VC) Implementation Plan

## Overview

Video Continuation (VC) extends an input video by conditioning on its last N frames. It is a generalization of I2V:
- **I2V**: 1 conditioning frame → generates video
- **VC**: 13+ conditioning frames → generates video continuation

The key optimization is the **KV cache** which pre-computes attention K/V for conditioning frames once and reuses them at every denoising step, providing 2-3x speedup.

---

## Reference Implementation

Original LongCat VC: `/mnt/fast-disks/hao_lab/shao/LongCat-Video/run_demo_video_continuation.py`

Key method: `LongCatVideoPipeline.generate_vc()` in `longcat_video/pipeline_longcat_video.py`

---

## Existing Components to Reuse

| Component | Location | Notes |
|-----------|----------|-------|
| `LongCatI2VDenoisingStage` | `stages/longcat_i2v_denoising.py` | Has timestep masking, selective denoising |
| `LongCatI2VLatentPreparationStage` | `stages/longcat_i2v_latent_preparation.py` | Adapt for multiple frames |
| `LongCatImageVAEEncodingStage` | `stages/longcat_image_vae_encoding.py` | Extend for video |
| `LongCatTransformer3DModel` | `models/dits/longcat.py` | Add KV cache params |
| `LongCatSelfAttention` | `models/dits/longcat.py` | Add `forward_with_kv_cache` |

---

## Architecture

### VC Pipeline Flow

```
Input Video (13+ frames)
       │
       ▼
┌──────────────────────────────────────────────────────────────┐
│ 1. Video VAE Encoding                                        │
│    - Encode num_cond_frames to latent space                  │
│    - Apply LongCat normalization: (latents - mean) / std     │
│    - Calculate num_cond_latents = 1 + (num_cond_frames-1)//4 │
└──────────────────────────────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────────────────────────┐
│ 2. Latent Preparation                                        │
│    - Generate noise for all target frames                    │
│    - Replace first num_cond_latents with encoded video       │
└──────────────────────────────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────────────────────────┐
│ 3. KV Cache Initialization (if use_kv_cache=True)            │
│    - Extract cond_latents = latents[:, :, :num_cond_latents] │
│    - Run transformer with return_kv=True, skip_crs_attn=True │
│    - Store kv_cache_dict = {block_idx: (k, v)}               │
│    - Remove cond from latents: latents = latents[:,:,N:]     │
└──────────────────────────────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────────────────────────┐
│ 4. Denoising Loop                                            │
│                                                              │
│    FOR each timestep t:                                      │
│      - Expand latents for CFG                                │
│      - Prepare timestep tensor [B, T]                        │
│                                                              │
│      IF use_kv_cache:                                        │
│        - All frames in latents are noise (no masking)        │
│        - Pass kv_cache_dict to transformer                   │
│      ELSE:                                                   │
│        - timestep[:, :num_cond_latents] = 0                  │
│        - Pass num_cond_latents to transformer                │
│                                                              │
│      - Run transformer → noise_pred                          │
│      - Apply CFG-zero optimized guidance                     │
│      - Negate: noise_pred = -noise_pred                      │
│                                                              │
│      IF use_kv_cache:                                        │
│        - latents = scheduler.step(noise_pred, t, latents)    │
│      ELSE:                                                   │
│        - latents[:,:,N:] = scheduler.step(pred[:,:,N:],...)  │
│                                                              │
│    IF use_kv_cache:                                          │
│      - Concatenate: latents = cat([cond_latents, latents])   │
└──────────────────────────────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────────────────────────┐
│ 5. VAE Decoding                                              │
│    - Denormalize latents                                     │
│    - Decode to pixel space                                   │
└──────────────────────────────────────────────────────────────┘
       │
       ▼
   Output Video
```

### KV Cache Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                    KV Cache Initialization                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  cond_latents [B, C, num_cond_latents, H, W]                   │
│        │                                                        │
│        ▼                                                        │
│  ┌───────────────────────────────────────┐                     │
│  │ Transformer Forward                    │                     │
│  │   - timestep = zeros [B, num_cond_lat] │                     │
│  │   - return_kv = True                   │                     │
│  │   - skip_crs_attn = True               │                     │
│  └───────────────────────────────────────┘                     │
│        │                                                        │
│        ▼                                                        │
│  kv_cache_dict = {                                             │
│    0: (k_block0, v_block0),  # [B, heads, N_cond, head_dim]   │
│    1: (k_block1, v_block1),                                    │
│    ...                                                         │
│    47: (k_block47, v_block47)                                  │
│  }                                                              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘


┌─────────────────────────────────────────────────────────────────┐
│                    KV Cache Usage (Each Step)                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  noise_latents [B, C, T_noise, H, W]                           │
│        │                                                        │
│        ▼                                                        │
│  ┌───────────────────────────────────────┐                     │
│  │ Self-Attention (forward_with_kv_cache)│                     │
│  │                                        │                     │
│  │  1. Compute Q, K, V for noise tokens   │                     │
│  │  2. Apply RoPE to K_noise              │                     │
│  │  3. K_full = [K_cache, K_noise]        │                     │
│  │     V_full = [V_cache, V_noise]        │                     │
│  │  4. Attention: Q_noise @ K_full^T      │                     │
│  │                                        │                     │
│  └───────────────────────────────────────┘                     │
│        │                                                        │
│        ▼                                                        │
│  output [B, N_noise, C]                                        │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Implementation Details

### 1. num_cond_latents Calculation

```python
# VAE temporal compression is 4x
vae_temporal_scale = 4
num_cond_latents = 1 + (num_cond_frames - 1) // vae_temporal_scale

# Examples:
#   1 frame  → 1 latent  (I2V)
#   5 frames → 2 latents
#   9 frames → 3 latents  
#  13 frames → 4 latents (standard VC)
#  25 frames → 7 latents
```

### 2. Video VAE Encoding

**File**: `fastvideo/pipelines/stages/longcat_video_vae_encoding.py`

```python
class LongCatVideoVAEEncodingStage(PipelineStage):
    """Encode video frames to latent space for VC conditioning."""
    
    def __init__(self, vae):
        super().__init__()
        self.vae = vae
    
    def forward(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> ForwardBatch:
        # 1. Get video frames
        video_frames = batch.video_frames  # List[PIL.Image] or video path
        num_cond_frames = batch.num_cond_frames  # e.g., 13
        
        # 2. Load video if path
        if isinstance(video_frames, str):
            from diffusers.utils import load_video
            video_frames = load_video(video_frames)
        
        # 3. Take last num_cond_frames
        video_frames = video_frames[-num_cond_frames:]
        
        # 4. Preprocess and stack frames
        # Resize, normalize to [-1, 1], stack to [1, C, T, H, W]
        video_tensor = self.preprocess_video(video_frames, batch.height, batch.width)
        
        # 5. Encode via VAE
        with torch.no_grad():
            latent = self.vae.encode(video_tensor).latent_dist.sample(batch.generator)
        
        # 6. Apply LongCat normalization
        latent = self.normalize_latents(latent)
        
        # 7. Calculate num_cond_latents
        vae_temporal_scale = 4
        num_cond_latents = 1 + (num_cond_frames - 1) // vae_temporal_scale
        
        # 8. Store in batch
        batch.video_latent = latent
        batch.num_cond_latents = num_cond_latents
        
        return batch
    
    def normalize_latents(self, latents):
        """LongCat normalization: (latents - mean) / std"""
        latents_mean = torch.tensor(self.vae.config.latents_mean).view(1, -1, 1, 1, 1)
        latents_std = torch.tensor(self.vae.config.latents_std).view(1, -1, 1, 1, 1)
        return (latents - latents_mean.to(latents)) / latents_std.to(latents)
```

### 3. KV Cache Parameters in Transformer

**File**: `fastvideo/models/dits/longcat.py`

Add to `LongCatTransformer3DModel.forward()`:

```python
def forward(
    self,
    hidden_states: torch.Tensor,           # [B, C, T, H, W]
    encoder_hidden_states: torch.Tensor,   # [B, N_text, C_text]
    timestep: torch.LongTensor,            # [B] or [B, T]
    encoder_attention_mask: torch.Tensor | None = None,
    num_cond_latents: int = 0,
    # === NEW KV CACHE PARAMS ===
    return_kv: bool = False,
    kv_cache_dict: dict | None = None,
    skip_crs_attn: bool = False,
    offload_kv_cache: bool = False,
    **kwargs
) -> torch.Tensor | tuple[torch.Tensor, dict]:
    
    B, _, T, H, W = hidden_states.shape
    N_t = T // self.patch_size[0]
    N_h = H // self.patch_size[1]
    N_w = W // self.patch_size[2]
    
    # ... existing embedding code ...
    
    # Transform through blocks with KV cache
    kv_cache_dict_ret = {}
    
    for i, block in enumerate(self.blocks):
        block_kv_cache = kv_cache_dict.get(i, None) if kv_cache_dict else None
        
        block_out = block(
            x, context, t,
            latent_shape=(N_t, N_h, N_w),
            num_cond_latents=num_cond_latents,
            return_kv=return_kv,
            kv_cache=block_kv_cache,
            skip_crs_attn=skip_crs_attn,
        )
        
        if return_kv:
            x, kv_cache = block_out
            if offload_kv_cache:
                kv_cache_dict_ret[i] = (kv_cache[0].cpu(), kv_cache[1].cpu())
            else:
                kv_cache_dict_ret[i] = kv_cache
        else:
            x = block_out
    
    # ... existing output code ...
    
    if return_kv:
        return output, kv_cache_dict_ret
    return output
```

### 4. KV Cache in Transformer Block

**File**: `fastvideo/models/dits/longcat.py`

Modify `LongCatTransformerBlock.forward()`:

```python
def forward(
    self,
    x: torch.Tensor,
    context: torch.Tensor,
    t: torch.Tensor,
    latent_shape: tuple,
    num_cond_latents: int = 0,
    return_kv: bool = False,
    kv_cache: tuple | None = None,
    skip_crs_attn: bool = False,
    **kwargs
) -> torch.Tensor | tuple:
    
    # ... existing modulation code ...
    
    # Self-attention with KV cache
    if kv_cache is not None:
        # Move cache to device if offloaded
        kv_cache = (kv_cache[0].to(x.device), kv_cache[1].to(x.device))
        attn_out = self.self_attn.forward_with_kv_cache(
            x_norm,
            latent_shape=latent_shape,
            num_cond_latents=num_cond_latents,
            kv_cache=kv_cache,
        )
    else:
        attn_out = self.self_attn(
            x_norm,
            latent_shape=latent_shape,
            num_cond_latents=num_cond_latents,
            return_kv=return_kv,
        )
    
    if return_kv:
        attn_out, kv_cache_new = attn_out
    
    # ... existing residual code ...
    
    # Cross-attention (skip if requested for cache init)
    if not skip_crs_attn:
        # If using KV cache, no need for num_cond_latents in cross-attn
        cross_num_cond = None if kv_cache is not None else num_cond_latents
        x = x + self.cross_attn(x_norm_cross, context, 
                                 latent_shape=latent_shape,
                                 num_cond_latents=cross_num_cond)
    
    # ... existing FFN code ...
    
    if return_kv:
        return x, kv_cache_new
    return x
```

### 5. Self-Attention with KV Cache

**File**: `fastvideo/models/dits/longcat.py`

Add to `LongCatSelfAttention`:

```python
def forward(
    self,
    x: torch.Tensor,
    latent_shape: tuple,
    num_cond_latents: int = 0,
    return_kv: bool = False,
    **kwargs
) -> torch.Tensor | tuple:
    """Standard forward, optionally returning KV for caching."""
    B, N, C = x.shape
    T, H, W = latent_shape
    
    # Project to Q/K/V
    q, _ = self.to_q(x)
    k, _ = self.to_k(x)
    v, _ = self.to_v(x)
    
    q = q.view(B, N, self.num_heads, self.head_dim)
    k = k.view(B, N, self.num_heads, self.head_dim)
    v = v.view(B, N, self.num_heads, self.head_dim)
    
    q = self.q_norm(q)
    k = self.k_norm(k)
    
    # Save pre-RoPE K/V for cache if requested
    if return_kv:
        k_cache = k.transpose(1, 2).clone()  # [B, heads, N, head_dim]
        v_cache = v.transpose(1, 2).clone()
    
    # Apply RoPE
    q_rope = q.transpose(1, 2)
    k_rope = k.transpose(1, 2)
    q_rope, k_rope = self.rope_3d(q_rope, k_rope, grid_size=latent_shape)
    q = q_rope.transpose(1, 2)
    k = k_rope.transpose(1, 2)
    
    # Handle conditioning mode (split attention)
    if num_cond_latents > 0:
        num_cond_tokens = num_cond_latents * (N // T)
        
        # Cond tokens only attend to themselves
        q_cond = q[:, :num_cond_tokens]
        k_cond = k[:, :num_cond_tokens]
        v_cond = v[:, :num_cond_tokens]
        out_cond = self.attn(q_cond, k_cond, v_cond)
        
        # Noise tokens attend to ALL tokens
        q_noise = q[:, num_cond_tokens:]
        out_noise = self.attn(q_noise, k, v)
        
        out = torch.cat([out_cond, out_noise], dim=1)
    else:
        out = self.attn(q, k, v)
    
    out = out.reshape(B, N, C)
    out, _ = self.to_out(out)
    
    if return_kv:
        return out, (k_cache, v_cache)
    return out


def forward_with_kv_cache(
    self,
    x: torch.Tensor,              # [B, N_noise, C]
    latent_shape: tuple,          # (T_noise, H, W)
    num_cond_latents: int,
    kv_cache: tuple,              # (k_cond, v_cond) - [B, heads, N_cond, head_dim]
) -> torch.Tensor:
    """
    Forward using cached K/V from conditioning frames.
    
    x contains only NOISE tokens.
    kv_cache contains pre-computed K/V for CONDITIONING tokens.
    """
    B, N, C = x.shape
    T, H, W = latent_shape
    
    k_cache, v_cache = kv_cache
    
    # Handle batch size mismatch (cache might be batch_size=1)
    if k_cache.shape[0] == 1 and B > 1:
        k_cache = k_cache.expand(B, -1, -1, -1)
        v_cache = v_cache.expand(B, -1, -1, -1)
    
    # Project to Q/K/V for noise tokens
    q, _ = self.to_q(x)
    k, _ = self.to_k(x)
    v, _ = self.to_v(x)
    
    q = q.view(B, N, self.num_heads, self.head_dim)
    k = k.view(B, N, self.num_heads, self.head_dim)
    v = v.view(B, N, self.num_heads, self.head_dim)
    
    q = self.q_norm(q)
    k = self.k_norm(k)
    
    # Apply RoPE to noise tokens
    q_rope = q.transpose(1, 2)
    k_rope = k.transpose(1, 2)
    q_rope, k_rope = self.rope_3d(q_rope, k_rope, grid_size=latent_shape)
    q = q_rope.transpose(1, 2)
    k = k_rope.transpose(1, 2)
    v = v.transpose(1, 2)
    
    # Concatenate cached K/V with noise K/V
    # Cache: [B, heads, N_cond, head_dim]
    # Noise: [B, heads, N_noise, head_dim]
    k_full = torch.cat([k_cache, k], dim=2)
    v_full = torch.cat([v_cache, v], dim=2)
    
    # Noise queries attend to full K/V (cond + noise)
    # q: [B, N_noise, heads, head_dim] → [B, heads, N_noise, head_dim]
    q = q.transpose(1, 2)
    
    # Run attention
    out = self._scaled_dot_product_attention(q, k_full, v_full)
    
    # Reshape output
    out = out.transpose(1, 2)  # [B, N_noise, heads, head_dim]
    out = out.reshape(B, N, C)
    out, _ = self.to_out(out)
    
    return out


def _scaled_dot_product_attention(self, q, k, v):
    """Scaled dot-product attention."""
    # q: [B, heads, N_q, head_dim]
    # k, v: [B, heads, N_kv, head_dim]
    scale = self.head_dim ** -0.5
    attn = torch.matmul(q, k.transpose(-2, -1)) * scale
    attn = F.softmax(attn, dim=-1)
    out = torch.matmul(attn, v)
    return out
```

### 6. KV Cache Initialization Stage

**File**: `fastvideo/pipelines/stages/longcat_kv_cache_init.py`

```python
class LongCatKVCacheInitStage(PipelineStage):
    """Pre-compute KV cache for conditioning frames."""
    
    def __init__(self, transformer):
        super().__init__()
        self.transformer = transformer
    
    def forward(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> ForwardBatch:
        if not getattr(fastvideo_args.pipeline_config, 'use_kv_cache', False):
            batch.kv_cache_dict = {}
            batch.use_kv_cache = False
            return batch
        
        batch.use_kv_cache = True
        offload_kv_cache = getattr(fastvideo_args.pipeline_config, 'offload_kv_cache', False)
        
        num_cond_latents = batch.num_cond_latents
        cond_latents = batch.latents[:, :, :num_cond_latents]
        
        # Timestep = 0 for conditioning (they are "clean")
        timestep = torch.zeros(
            cond_latents.shape[0], cond_latents.shape[2],
            device=cond_latents.device, dtype=cond_latents.dtype
        )
        
        # Empty prompt embeddings (cross-attn will be skipped)
        max_seq_len = 512
        caption_dim = self.transformer.config.caption_channels
        empty_embeds = torch.zeros(
            cond_latents.shape[0], max_seq_len, caption_dim,
            device=cond_latents.device, dtype=cond_latents.dtype
        )
        
        # Run transformer with return_kv=True, skip_crs_attn=True
        with torch.no_grad():
            _, kv_cache_dict = self.transformer(
                hidden_states=cond_latents,
                encoder_hidden_states=empty_embeds,
                timestep=timestep,
                return_kv=True,
                skip_crs_attn=True,
                offload_kv_cache=offload_kv_cache,
            )
        
        # Store cache and save cond_latents for later concatenation
        batch.kv_cache_dict = kv_cache_dict
        batch.cond_latents = cond_latents
        
        # Remove conditioning latents from main latents
        batch.latents = batch.latents[:, :, num_cond_latents:]
        
        return batch
```

### 7. VC Denoising Stage

**File**: `fastvideo/pipelines/stages/longcat_vc_denoising.py`

```python
class LongCatVCDenoisingStage(LongCatI2VDenoisingStage):
    """VC denoising with KV cache support."""
    
    def forward(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> ForwardBatch:
        # ... load transformer if needed ...
        
        latents = batch.latents
        timesteps = batch.timesteps
        num_cond_latents = batch.num_cond_latents
        use_kv_cache = getattr(batch, 'use_kv_cache', False)
        kv_cache_dict = getattr(batch, 'kv_cache_dict', {})
        
        # ... prepare CFG prompts ...
        
        for i, t in enumerate(timesteps):
            # Expand for CFG
            latent_model_input = torch.cat([latents] * 2) if do_cfg else latents
            
            # Prepare timestep
            timestep = t.expand(latent_model_input.shape[0])
            timestep = timestep.unsqueeze(-1).repeat(1, latent_model_input.shape[2])
            
            # Timestep masking only needed when NOT using cache
            if not use_kv_cache:
                timestep[:, :num_cond_latents] = 0
            
            # Transformer kwargs
            transformer_kwargs = {}
            if use_kv_cache:
                transformer_kwargs['kv_cache_dict'] = kv_cache_dict
            else:
                transformer_kwargs['num_cond_latents'] = num_cond_latents
            
            # Run transformer
            noise_pred = self.transformer(
                hidden_states=latent_model_input,
                encoder_hidden_states=prompt_embeds,
                timestep=timestep,
                **transformer_kwargs,
            )
            
            # Apply CFG-zero
            if do_cfg:
                noise_pred = self.apply_cfg_zero(noise_pred, guidance_scale)
            
            # Negate for flow matching
            noise_pred = -noise_pred
            
            # Scheduler step
            if use_kv_cache:
                # All latents are noise frames
                latents = self.scheduler.step(noise_pred, t, latents)[0]
            else:
                # Only update noise frames
                latents[:, :, num_cond_latents:] = self.scheduler.step(
                    noise_pred[:, :, num_cond_latents:],
                    t,
                    latents[:, :, num_cond_latents:],
                )[0]
        
        # Concatenate conditioning latents back if using cache
        if use_kv_cache:
            latents = torch.cat([batch.cond_latents, latents], dim=2)
        
        batch.latents = latents
        return batch
```

### 8. VC Pipeline

**File**: `fastvideo/pipelines/basic/longcat/longcat_vc_pipeline.py`

```python
class LongCatVideoContinuationPipeline(LoRAPipeline, ComposedPipelineBase):
    """LongCat Video Continuation pipeline."""
    
    _required_config_modules = [
        "text_encoder", "tokenizer", "vae", "transformer", "scheduler"
    ]
    
    def create_pipeline_stages(self, fastvideo_args: FastVideoArgs):
        # 1. Input validation
        self.add_stage("input_validation", InputValidationStage())
        
        # 2. Text encoding
        self.add_stage("text_encoding", TextEncodingStage(
            text_encoders=[self.get_module("text_encoder")],
            tokenizers=[self.get_module("tokenizer")],
        ))
        
        # 3. Video VAE encoding (NEW)
        self.add_stage("video_vae_encoding", LongCatVideoVAEEncodingStage(
            vae=self.get_module("vae")
        ))
        
        # 4. Timestep preparation
        self.add_stage("timestep_preparation", TimestepPreparationStage(
            scheduler=self.get_module("scheduler")
        ))
        
        # 5. Latent preparation (reuse/adapt I2V stage)
        self.add_stage("latent_preparation", LongCatVCLatentPreparationStage(
            scheduler=self.get_module("scheduler"),
            transformer=self.get_module("transformer"),
        ))
        
        # 6. KV cache initialization (optional)
        if fastvideo_args.pipeline_config.use_kv_cache:
            self.add_stage("kv_cache_init", LongCatKVCacheInitStage(
                transformer=self.get_module("transformer"),
            ))
        
        # 7. Denoising
        self.add_stage("denoising", LongCatVCDenoisingStage(
            transformer=self.get_module("transformer"),
            scheduler=self.get_module("scheduler"),
            vae=self.get_module("vae"),
            pipeline=self,
        ))
        
        # 8. Decoding
        self.add_stage("decoding", DecodingStage(
            vae=self.get_module("vae"),
            pipeline=self,
        ))
```

---

## Enhanced HF Schedule (Optional)

The original VC uses an "enhanced high-frequency" timestep schedule for better temporal coherence:

```python
def get_enhanced_hf_timesteps(timesteps, device):
    """
    Modify timestep schedule for better high-frequency detail.
    
    Instead of ending at the scheduler's default, add uniform steps
    from 500 to 0 with 10 steps at the end.
    """
    tail_uniform_start = 500
    tail_uniform_end = 0
    num_tail_uniform_steps = 10
    
    # Create uniform tail
    timesteps_uniform_tail = torch.linspace(
        tail_uniform_start, tail_uniform_end, 
        num_tail_uniform_steps, device=device
    )
    
    # Filter original timesteps to keep only those > 500
    filtered = timesteps[timesteps > tail_uniform_start]
    
    # Concatenate
    return torch.cat([filtered, timesteps_uniform_tail])
```

---

## Files to Create

1. `fastvideo/pipelines/stages/longcat_video_vae_encoding.py`
2. `fastvideo/pipelines/stages/longcat_kv_cache_init.py`
3. `fastvideo/pipelines/stages/longcat_vc_denoising.py`
4. `fastvideo/pipelines/basic/longcat/longcat_vc_pipeline.py`

## Files to Modify

1. `fastvideo/models/dits/longcat.py`
   - Add KV cache params to `LongCatTransformer3DModel.forward()`
   - Add KV cache handling to `LongCatTransformerBlock.forward()`
   - Add `forward_with_kv_cache()` to `LongCatSelfAttention`
   - Modify `forward()` in `LongCatSelfAttention` to optionally return KV

2. `fastvideo/pipelines/stages/__init__.py`
   - Export new stages

3. `fastvideo/pipelines/stages/longcat_i2v_latent_preparation.py`
   - Generalize to handle both image_latent and video_latent

---

## API Usage

### Python

```python
from fastvideo import VideoGenerator

generator = VideoGenerator.from_pretrained("weights/longcat-native")

# Video Continuation
video = generator.generate_video(
    prompt="The motorcycle continues down the road",
    video_path="motorcycle.mp4",
    num_cond_frames=13,
    num_frames=93,
    use_kv_cache=True,
    num_inference_steps=50,
    guidance_scale=4.0,
)

# With Distillation LoRA
generator.load_lora("weights/longcat-native/lora/cfg_step_lora.safetensors")
video = generator.generate_video(
    ...,
    num_inference_steps=16,
    use_distill=True,
    guidance_scale=1.0,
)

# Autoregressive Long Video
segments = []
current_video = initial_video
for _ in range(10):
    segment = generator.generate_video(
        video_path=current_video,
        prompt=prompt,
        num_cond_frames=13,
        num_frames=93,
        use_kv_cache=True,
    )
    segments.append(segment[13:])
    current_video = segment
```

### CLI

```bash
fastvideo generate \
    --model-path weights/longcat-native \
    --video-path motorcycle.mp4 \
    --num-cond-frames 13 \
    --prompt "Continuing the journey" \
    --num-frames 93 \
    --use-kv-cache \
    --output-path output_vc.mp4
```

---

## Testing Checklist

- [ ] Video VAE encoding produces correct latent shape
- [ ] num_cond_latents calculation is correct for various frame counts
- [ ] KV cache initialization produces correct cache structure
- [ ] forward_with_kv_cache produces same output as full attention (for same input)
- [ ] VC without cache produces coherent video continuation
- [ ] VC with cache produces identical output to without cache
- [ ] VC with cache is 2-3x faster than without cache
- [ ] VC works with distill LoRA (16 steps)
- [ ] VC output can be passed to refine stage
- [ ] Autoregressive generation (chaining VC calls) works
- [ ] Multi-GPU with SP works correctly

---

## Key Implementation Notes

1. **RoPE in Cache**: Cache stores K/V **before** RoPE. When using cache, RoPE is applied only to new noise tokens with positions starting after conditioning frames.

2. **Cache Memory**: 48 blocks × 32 heads × num_cond_tokens × head_dim × 2 (K+V) × 2 (bf16 bytes). For 13 cond frames at 480p, ~2-3GB.

3. **skip_crs_attn**: During cache init, cross-attention is skipped because:
   - Conditioning frames don't need text guidance (they're already "clean")
   - Reduces computation during cache initialization

4. **CFG with Cache**: Cache is computed once without CFG. During denoising, the same cache is used for both conditional and unconditional branches (latents are doubled, cache is broadcasted).

5. **Post-Denoising Concatenation**: After denoising with cache, conditioning latents must be concatenated back before VAE decoding: `latents = cat([cond_latents, denoised_latents], dim=2)`.

















