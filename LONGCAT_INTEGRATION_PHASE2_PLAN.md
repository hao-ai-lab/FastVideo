# LongCat Integration Phase 2: Native FastVideo Implementation Plan

**Date**: November 6, 2025  
**Status**: Planning Phase  
**Goal**: Reimplement LongCat using FastVideo's native layers and conventions

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Architecture Overview](#architecture-overview)
3. [Implementation Strategy](#implementation-strategy)
4. [Detailed Component Design](#detailed-component-design)
5. [Weight Conversion Plan](#weight-conversion-plan)
6. [Implementation Phases](#implementation-phases)
7. [Testing Strategy](#testing-strategy)
8. [Success Criteria](#success-criteria)
9. [Risk Mitigation](#risk-mitigation)

---

## Executive Summary

### Current State (Phase 1)
- ✅ Working wrapper using LongCat's original modules from `third_party/`
- ✅ Three critical issues identified and fixed:
  1. Parameter order mismatch
  2. CFG-zero optimized guidance
  3. Noise prediction negation for flow matching
- ✅ Successfully generates videos via FastVideo API

### Phase 2 Goals
- 🎯 Reimplement LongCat DiT using FastVideo's native layers
- 🎯 Replace `nn.Linear` with `ReplicatedLinear` for tensor parallelism
- 🎯 Use `DistributedAttention` instead of custom attention backends
- 🎯 Integrate CFG-zero at the model level
- 🎯 Support FSDP, compilation, and all FastVideo optimizations
- 🎯 Maintain or improve performance vs. wrapper

### Key Challenges
1. **Fused to separate projections**: Split QKV and KV weight matrices
2. **Custom attention backends**: Adapt BSA to FastVideo's attention system
3. **3D RoPE**: Implement temporal-spatial rotary embeddings
4. **CFG-zero**: Build into forward pass, not just denoising stage
5. **FP32 operations**: Preserve numerical stability for normalization

---

## Architecture Overview

### Current Wrapper Structure

```
fastvideo/third_party/longcat_video/
├── modules/
│   ├── longcat_video_dit.py         # Wrapper DiT (imports from original)
│   ├── attention.py                  # Custom attention backends
│   ├── blocks.py                     # Embedders, FFN, LayerNorm
│   └── lora_utils.py                 # Custom LoRA implementation
└── pipeline_longcat_video.py         # Reference pipeline

fastvideo/pipelines/basic/longcat/
└── longcat_pipeline.py               # FastVideo pipeline wrapper

fastvideo/pipelines/stages/
└── longcat_denoising.py              # CFG-zero denoising stage
```

### Target Native Structure

```
fastvideo/models/dits/
├── longcat.py                        # NEW: Native LongCat DiT
│   ├── LongCatTransformer3DModel     # Main model class
│   ├── LongCatBlock                  # Single-stream transformer block
│   ├── LongCatSelfAttention          # Self-attention with 3D RoPE
│   ├── LongCatCrossAttention         # Cross-attention
│   └── LongCatSwiGLUFFN              # SwiGLU feed-forward network

fastvideo/attention/
└── block_sparse_attention.py        # NEW: BSA backend for all models

fastvideo/pipelines/basic/longcat/
└── longcat_pipeline.py               # MODIFIED: Use native DiT

fastvideo/pipelines/stages/
└── denoising.py                      # MODIFIED: Add CFG-zero support
                                      # OR: Keep longcat_denoising.py

fastvideo/configs/pipelines/
└── longcat.py                        # MODIFIED: Update for native model
```

---

## Implementation Strategy

### Guiding Principles

1. **Incremental Development**: Build and test each component independently
2. **Numerical Equivalence**: Validate outputs match wrapper at each step
3. **FastVideo Conventions**: Follow patterns from WanVideo/HunyuanVideo
4. **Backward Compatibility**: Keep wrapper functional until native is validated
5. **Reusability**: Make BSA and other components available to all models

### Development Approach

```
Phase 2.1: Core Layers (Week 1)
├── Embeddings (patch, time, caption)
├── Normalization (RMSNorm FP32, LayerNorm FP32)
├── SwiGLU FFN
└── 3D RoPE utilities

Phase 2.2: Attention (Week 2)
├── Self-attention with separate Q/K/V projections
├── Cross-attention with separate K/V projections
├── 3D RoPE integration
└── Standard attention backend (FlashAttention2)

Phase 2.3: Block & Model (Week 2-3)
├── LongCatBlock with AdaLN modulation
├── LongCatTransformer3DModel
├── CFG-zero integration
└── Weight loading and mapping

Phase 2.4: BSA Integration (Week 3)
├── Extract BSA kernel from original LongCat
├── Create AttentionBackendEnum.BSA
├── Integrate with DistributedAttention dispatcher
└── Test on LongCat and potentially other models

Phase 2.5: Advanced Features (Week 4)
├── LoRA support
├── KV caching for video continuation
├── Context parallelism
└── Distilled inference support

Phase 2.6: Testing & Optimization (Week 4-5)
├── Numerical validation
├── Performance benchmarking
├── FSDP & compilation testing
└── Memory profiling
```

---

## Detailed Component Design

### 1. Model Class: `LongCatTransformer3DModel`

**File**: `fastvideo/models/dits/longcat.py`

**Inherits**: `CachableDiT` (for TeaCache support)

**Class Attributes**:
```python
class LongCatTransformer3DModel(CachableDiT):
    # FSDP sharding: shard at each transformer block
    _fsdp_shard_conditions = [
        lambda n, m: "blocks" in n and n.split(".")[-1].isdigit(),
    ]
    
    # Parameter name mapping for weight conversion
    param_names_mapping = {
        # Defined in LongCatVideoConfig
        # Maps original LongCat names to FastVideo names
    }
    reverse_param_names_mapping = {
        # Reverse mapping for checkpoint saving
    }
    lora_param_names_mapping = {
        # LoRA parameter mapping
    }
    
    # Supported attention backends
    _supported_attention_backends = (
        AttentionBackendEnum.FLASH_ATTN,
        AttentionBackendEnum.TORCH_SDPA,
        AttentionBackendEnum.BSA,  # NEW: Block-sparse attention
    )
    
    # Required attributes
    hidden_size: int = 4096
    num_attention_heads: int = 32
    num_channels_latents: int = 16
```

**Constructor Parameters**:
```python
def __init__(
    self,
    config: LongCatVideoConfig,  # FastVideo config object
    hf_config: dict[str, Any],   # From config.json
    **kwargs
) -> None:
    super().__init__(config=config, hf_config=hf_config)
    
    # Extract architecture parameters
    self.hidden_size = config.hidden_size  # 4096
    self.num_heads = config.num_attention_heads  # 32
    self.depth = config.depth  # 48
    self.mlp_ratio = config.mlp_ratio  # 4
    self.in_channels = config.in_channels  # 16
    self.out_channels = config.out_channels  # 16
    self.patch_size = config.patch_size  # [1, 2, 2]
    
    # Embeddings
    self.patch_embed = PatchEmbed3D(...)
    self.time_embedder = TimestepEmbedder(...)
    self.caption_embedder = CaptionEmbedder(...)
    
    # Transformer blocks (48 blocks)
    self.blocks = nn.ModuleList([
        LongCatBlock(
            hidden_size=self.hidden_size,
            num_heads=self.num_heads,
            mlp_ratio=self.mlp_ratio,
            adaln_tembed_dim=config.adaln_tembed_dim,
            config=config,
        )
        for _ in range(self.depth)
    ])
    
    # Output projection
    self.final_layer = FinalLayer(...)
```

**Forward Method** (CRITICAL: Parameter order):
```python
def forward(
    self,
    hidden_states: torch.Tensor,           # [B, C, T, H, W]
    encoder_hidden_states: torch.Tensor,   # [B, 1, N_text, C_text] or [B, N_text, C_text]
    timestep: torch.LongTensor,            # [B] or [B, T]
    encoder_attention_mask: torch.Tensor | None = None,
    guidance: float | None = None,         # Unused, for API compatibility
    **kwargs
) -> torch.Tensor:
    """
    Forward pass with FastVideo parameter ordering.
    
    NOTE: This is different from original LongCat which has
          (hidden_states, timestep, encoder_hidden_states)
    """
    # 1. Patch embedding
    x = self.patch_embed(hidden_states)  # [B, N, C]
    
    # 2. Timestep embedding
    t = self.time_embedder(timestep, ...)  # [B, T, C_t]
    
    # 3. Caption embedding
    y = self.caption_embedder(encoder_hidden_states, encoder_attention_mask)
    
    # 4. Transformer blocks
    for block in self.blocks:
        x = block(x, y, t, ...)
    
    # 5. Output projection
    output = self.final_layer(x, t)
    
    # Reshape to [B, C_out, T, H, W]
    output = rearrange(output, 'b (t h w) c -> b c t h w', t=T, h=H, w=W)
    
    return output
```

### 2. Transformer Block: `LongCatBlock`

**Design**: Single-stream block with self-attention → cross-attention → FFN

```python
class LongCatBlock(nn.Module):
    """
    Single-stream transformer block with:
    - AdaLN modulation (FP32)
    - Self-attention with 3D RoPE
    - Cross-attention
    - SwiGLU FFN
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: int,
        adaln_tembed_dim: int,
        config: LongCatVideoConfig,
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        head_dim = hidden_size // num_heads
        
        # AdaLN modulation (6 parameters: scale/shift for attn & ffn, gate for residual)
        # Use FP32 for stability
        self.adaln_modulation = nn.Sequential(
            nn.SiLU(),
            ReplicatedLinear(adaln_tembed_dim, 6 * hidden_size, bias=True)
        )
        
        # Normalization layers (FP32)
        self.norm_attn = RMSNorm(hidden_size, eps=1e-6, elementwise_affine=False)
        self.norm_ffn = RMSNorm(hidden_size, eps=1e-6, elementwise_affine=False)
        self.norm_cross = RMSNorm(hidden_size, eps=1e-6, elementwise_affine=True)
        
        # Self-attention with 3D RoPE
        self.self_attn = LongCatSelfAttention(
            dim=hidden_size,
            num_heads=num_heads,
            config=config,
        )
        
        # Cross-attention
        self.cross_attn = LongCatCrossAttention(
            dim=hidden_size,
            num_heads=num_heads,
            config=config,
        )
        
        # SwiGLU FFN
        ffn_hidden_dim = int(hidden_size * mlp_ratio * 2 / 3)
        # Round up to nearest multiple of 256
        ffn_hidden_dim = 256 * ((ffn_hidden_dim + 255) // 256)
        
        self.ffn = LongCatSwiGLUFFN(
            dim=hidden_size,
            hidden_dim=ffn_hidden_dim,
        )
    
    def forward(
        self,
        x: torch.Tensor,              # [B, N, C]
        y: torch.Tensor,              # [1, N_text, C] (compacted)
        t: torch.Tensor,              # [B, T, C_t]
        y_seqlen: list[int],          # [B] - valid text tokens per sample
        latent_shape: tuple,          # (T, H, W)
        **kwargs
    ) -> torch.Tensor:
        """
        Forward pass with AdaLN modulation.
        """
        B, N, C = x.shape
        T, H, W = latent_shape
        
        # === AdaLN Modulation (CRITICAL: FP32) ===
        with torch.amp.autocast(device_type='cuda', dtype=torch.float32):
            shift_msa, scale_msa, gate_msa, \
            shift_mlp, scale_mlp, gate_mlp = \
                self.adaln_modulation(t).unsqueeze(2).chunk(6, dim=-1)  # [B, T, 1, C]
        
        # === Self-Attention ===
        # Modulate in FP32, then convert back
        x_norm = modulate_fp32(self.norm_attn, x.view(B, T, -1, C), shift_msa, scale_msa)
        x_norm = x_norm.view(B, N, C)
        
        attn_out = self.self_attn(x_norm, latent_shape=latent_shape)
        
        # Residual with gating (FP32)
        with torch.amp.autocast(device_type='cuda', dtype=torch.float32):
            x = x + gate_msa.view(B, -1, C) * attn_out
        
        # === Cross-Attention ===
        x_norm_cross = self.norm_cross(x)
        cross_out = self.cross_attn(x_norm_cross, y, y_seqlen, latent_shape)
        x = x + cross_out
        
        # === FFN ===
        x_norm_ffn = modulate_fp32(self.norm_ffn, x.view(B, T, -1, C), shift_mlp, scale_mlp)
        x_norm_ffn = x_norm_ffn.view(B, N, C)
        
        ffn_out = self.ffn(x_norm_ffn)
        
        # Residual with gating (FP32)
        with torch.amp.autocast(device_type='cuda', dtype=torch.float32):
            x = x + gate_mlp.view(B, -1, C) * ffn_out
        
        return x
```

### 3. Self-Attention: `LongCatSelfAttention`

**Key Features**:
- Separate Q/K/V projections (not fused)
- 3D RoPE (temporal + spatial)
- Per-head RMS normalization
- Uses FastVideo's `DistributedAttention`

```python
class LongCatSelfAttention(nn.Module):
    """
    Self-attention with 3D RoPE and per-head normalization.
    """
    
    def __init__(
        self,
        dim: int,
        num_heads: int,
        config: LongCatVideoConfig,
    ):
        super().__init__()
        
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        
        # Separate Q/K/V projections (NOT fused like original)
        self.to_q = ReplicatedLinear(dim, dim, bias=True)
        self.to_k = ReplicatedLinear(dim, dim, bias=True)
        self.to_v = ReplicatedLinear(dim, dim, bias=True)
        
        # Per-head RMS normalization (FP32)
        self.q_norm = RMSNorm(self.head_dim, eps=1e-6)
        self.k_norm = RMSNorm(self.head_dim, eps=1e-6)
        
        # Output projection
        self.to_out = ReplicatedLinear(dim, dim, bias=True)
        
        # 3D RoPE (temporal + spatial)
        self.use_rope = True
        # RoPE parameters will be cached in forward
        self._rope_cache = {}
        
        # FastVideo attention backend
        self.attn_backend = DistributedAttention(
            num_heads=num_heads,
            head_size=self.head_dim,
            supported_attention_backends=config._supported_attention_backends,
        )
    
    def forward(
        self,
        x: torch.Tensor,              # [B, N, C]
        latent_shape: tuple,          # (T, H, W)
        **kwargs
    ) -> torch.Tensor:
        """
        Forward pass with 3D RoPE.
        """
        B, N, C = x.shape
        T, H, W = latent_shape
        
        # Project to Q/K/V
        q = self.to_q(x)  # [B, N, C]
        k = self.to_k(x)
        v = self.to_v(x)
        
        # Reshape to heads
        q = q.view(B, N, self.num_heads, self.head_dim)
        k = k.view(B, N, self.num_heads, self.head_dim)
        v = v.view(B, N, self.num_heads, self.head_dim)
        
        # Per-head RMS normalization (FP32)
        with torch.amp.autocast(device_type='cuda', dtype=torch.float32):
            q = self.q_norm(q)
            k = self.k_norm(k)
        
        # Apply 3D RoPE
        if self.use_rope:
            cos, sin = self._get_3d_rope(T, H, W, q.device, q.dtype)
            q = apply_3d_rotary_emb(q, cos, sin, latent_shape)
            k = apply_3d_rotary_emb(k, cos, sin, latent_shape)
        
        # Reshape for attention [B, num_heads, N, head_dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Run attention through FastVideo backend
        out = self.attn_backend(q, k, v)  # [B, num_heads, N, head_dim]
        
        # Reshape and project out
        out = out.transpose(1, 2).reshape(B, N, C)
        out = self.to_out(out)
        
        return out
    
    def _get_3d_rope(
        self,
        T: int,
        H: int,
        W: int,
        device: torch.device,
        dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get or compute 3D RoPE (cached).
        
        Returns:
            cos, sin: [N, head_dim] tensors for rotary embedding
        """
        key = (T, H, W, device, dtype)
        if key not in self._rope_cache:
            # Compute 3D positional embeddings
            # Implementation similar to original LongCat
            cos, sin = compute_3d_rope(T, H, W, self.head_dim, device, dtype)
            self._rope_cache[key] = (cos, sin)
        return self._rope_cache[key]
```

### 4. Cross-Attention: `LongCatCrossAttention`

**Key Features**:
- Separate Q projection, fused K/V projection (split during forward)
- No RoPE
- Per-head RMS normalization
- Variable-length text handling with compacted representation

```python
class LongCatCrossAttention(nn.Module):
    """
    Cross-attention for text conditioning.
    """
    
    def __init__(
        self,
        dim: int,
        num_heads: int,
        config: LongCatVideoConfig,
    ):
        super().__init__()
        
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        
        # Q from image, K/V from text
        self.to_q = ReplicatedLinear(dim, dim, bias=True)
        self.to_k = ReplicatedLinear(dim, dim, bias=True)
        self.to_v = ReplicatedLinear(dim, dim, bias=True)
        
        # Per-head RMS normalization (FP32)
        self.q_norm = RMSNorm(self.head_dim, eps=1e-6)
        self.k_norm = RMSNorm(self.head_dim, eps=1e-6)
        
        # Output projection
        self.to_out = ReplicatedLinear(dim, dim, bias=True)
        
        # Attention backend
        self.attn_backend = DistributedAttention(
            num_heads=num_heads,
            head_size=self.head_dim,
            supported_attention_backends=config._supported_attention_backends,
        )
    
    def forward(
        self,
        x: torch.Tensor,              # [B, N_img, C]
        y: torch.Tensor,              # [1, N_text_total, C]
        y_seqlen: list[int],          # [B] - valid tokens per sample
        latent_shape: tuple,
        **kwargs
    ) -> torch.Tensor:
        """
        Forward pass with variable-length text.
        
        Note: y is compacted (all batches concatenated) for efficiency.
              Use y_seqlen to split back to individual samples.
        """
        B, N_img, C = x.shape
        
        # Project queries
        q = self.to_q(x)  # [B, N_img, C]
        
        # Project keys and values (from text)
        k = self.to_k(y)  # [1, N_text_total, C]
        v = self.to_v(y)
        
        # Reshape to heads
        q = q.view(B, N_img, self.num_heads, self.head_dim)
        k = k.view(1, -1, self.num_heads, self.head_dim)
        v = v.view(1, -1, self.num_heads, self.head_dim)
        
        # Per-head RMS normalization (FP32)
        with torch.amp.autocast(device_type='cuda', dtype=torch.float32):
            q = self.q_norm(q)
            k = self.k_norm(k)
        
        # Expand k, v for each sample in batch
        # Handle variable-length sequences with y_seqlen
        k_list = []
        v_list = []
        offset = 0
        for seq_len in y_seqlen:
            k_list.append(k[0, offset:offset+seq_len])
            v_list.append(v[0, offset:offset+seq_len])
            offset += seq_len
        
        # Pad to max length in batch
        max_seq_len = max(y_seqlen)
        k_padded = torch.stack([
            torch.cat([k_i, k_i.new_zeros(max_seq_len - k_i.shape[0], self.num_heads, self.head_dim)])
            for k_i in k_list
        ])  # [B, max_seq_len, num_heads, head_dim]
        
        v_padded = torch.stack([
            torch.cat([v_i, v_i.new_zeros(max_seq_len - v_i.shape[0], self.num_heads, self.head_dim)])
            for v_i in v_list
        ])
        
        # Transpose for attention [B, num_heads, seq_len, head_dim]
        q = q.transpose(1, 2)
        k_padded = k_padded.transpose(1, 2)
        v_padded = v_padded.transpose(1, 2)
        
        # Create attention mask for padded tokens
        attn_mask = torch.zeros(B, max_seq_len, device=x.device, dtype=torch.bool)
        for i, seq_len in enumerate(y_seqlen):
            attn_mask[i, seq_len:] = True  # Mask padding
        
        # Run attention
        out = self.attn_backend(q, k_padded, v_padded, attn_mask=attn_mask)
        
        # Reshape and project out
        out = out.transpose(1, 2).reshape(B, N_img, C)
        out = self.to_out(out)
        
        return out
```

### 5. SwiGLU FFN: `LongCatSwiGLUFFN`

**Key Features**:
- Three separate projections (gate, up, down)
- SwiGLU activation: `gate(x) * SiLU(up(x))`
- Hidden dimension: `256 * ceil(2 * dim * mlp_ratio / 3 / 256)`

```python
class LongCatSwiGLUFFN(nn.Module):
    """
    SwiGLU feed-forward network.
    
    FFN(x) = down(gate(x) * SiLU(up(x)))
    """
    
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
    ):
        super().__init__()
        
        # Three projections for SwiGLU
        self.gate = ReplicatedLinear(dim, hidden_dim, bias=False)
        self.up = ReplicatedLinear(dim, hidden_dim, bias=False)
        self.down = ReplicatedLinear(hidden_dim, dim, bias=False)
        
        self.act = nn.SiLU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: gate(x) * SiLU(up(x)) -> down
        """
        gate_out = self.gate(x)
        up_out = self.up(x)
        return self.down(gate_out * self.act(up_out))
```

### 6. 3D RoPE Implementation

**File**: `fastvideo/layers/rotary_embedding_3d.py` (NEW)

```python
def compute_3d_rope(
    T: int,
    H: int,
    W: int,
    head_dim: int,
    device: torch.device,
    dtype: torch.dtype,
    base: float = 10000.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute 3D rotary position embeddings.
    
    Args:
        T: Temporal dimension
        H: Height dimension (after patch embedding)
        W: Width dimension (after patch embedding)
        head_dim: Attention head dimension
        device, dtype: Tensor properties
        base: RoPE base frequency
    
    Returns:
        cos, sin: [T*H*W, head_dim] tensors
    """
    # Split head_dim across 3 dimensions (T, H, W)
    # Each gets head_dim // 3 (with adjustment for divisibility)
    dim_t = head_dim // 3
    dim_h = head_dim // 3
    dim_w = head_dim - dim_t - dim_h  # Remaining
    
    # Compute frequency bands for each dimension
    inv_freq_t = 1.0 / (base ** (torch.arange(0, dim_t, 2, dtype=torch.float32, device=device) / dim_t))
    inv_freq_h = 1.0 / (base ** (torch.arange(0, dim_h, 2, dtype=torch.float32, device=device) / dim_h))
    inv_freq_w = 1.0 / (base ** (torch.arange(0, dim_w, 2, dtype=torch.float32, device=device) / dim_w))
    
    # Create position indices
    t_pos = torch.arange(T, dtype=torch.float32, device=device)
    h_pos = torch.arange(H, dtype=torch.float32, device=device)
    w_pos = torch.arange(W, dtype=torch.float32, device=device)
    
    # Compute position embeddings for each dimension
    freqs_t = torch.outer(t_pos, inv_freq_t)  # [T, dim_t//2]
    freqs_h = torch.outer(h_pos, inv_freq_h)  # [H, dim_h//2]
    freqs_w = torch.outer(w_pos, inv_freq_w)  # [W, dim_w//2]
    
    # Combine: for each (t, h, w) position, concatenate all frequencies
    # Shape: [T, H, W, head_dim//2]
    freqs = torch.cat([
        freqs_t[:, None, None, :].expand(T, H, W, -1),
        freqs_h[None, :, None, :].expand(T, H, W, -1),
        freqs_w[None, None, :, :].expand(T, H, W, -1),
    ], dim=-1)
    
    # Flatten spatial dimensions: [T*H*W, head_dim//2]
    freqs = freqs.reshape(-1, head_dim // 2)
    
    # Compute cos and sin (duplicate for complex pairs)
    cos = torch.cos(freqs).to(dtype)
    sin = torch.sin(freqs).to(dtype)
    
    # Interleave for complex number format: [T*H*W, head_dim]
    cos = torch.stack([cos, cos], dim=-1).flatten(-2)
    sin = torch.stack([sin, sin], dim=-1).flatten(-2)
    
    return cos, sin


def apply_3d_rotary_emb(
    x: torch.Tensor,              # [B, N, num_heads, head_dim]
    cos: torch.Tensor,            # [N, head_dim]
    sin: torch.Tensor,            # [N, head_dim]
    latent_shape: tuple,          # (T, H, W)
) -> torch.Tensor:
    """
    Apply 3D rotary embeddings to query or key tensor.
    """
    # Reshape for complex number rotation
    # Complex representation: (x[..., 0::2], x[..., 1::2])
    x_even = x[..., 0::2]  # Real part
    x_odd = x[..., 1::2]   # Imaginary part
    
    # Expand cos/sin for broadcasting
    cos = cos.unsqueeze(0).unsqueeze(2)  # [1, N, 1, head_dim]
    sin = sin.unsqueeze(0).unsqueeze(2)
    
    cos_even = cos[..., 0::2]
    cos_odd = cos[..., 1::2]
    sin_even = sin[..., 0::2]
    sin_odd = sin[..., 1::2]
    
    # Rotation: (x_real * cos - x_imag * sin, x_real * sin + x_imag * cos)
    out_even = x_even * cos_even - x_odd * sin_even
    out_odd = x_even * sin_odd + x_odd * cos_odd
    
    # Interleave back
    out = torch.stack([out_even, out_odd], dim=-1).flatten(-2)
    
    return out
```

### 7. Embeddings

**Patch Embedding** (3D):
```python
class PatchEmbed3D(nn.Module):
    """
    3D patch embedding with Conv3d.
    """
    def __init__(
        self,
        in_channels: int = 16,
        hidden_size: int = 4096,
        patch_size: list[int] = [1, 2, 2],
    ):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv3d(
            in_channels,
            hidden_size,
            kernel_size=patch_size,
            stride=patch_size,
            bias=True,
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C_in, T, H, W]
        Returns:
            [B, N, C] where N = (T/pt) * (H/ph) * (W/pw)
        """
        x = self.proj(x)  # [B, C, T', H', W']
        B, C, T, H, W = x.shape
        x = rearrange(x, 'b c t h w -> b (t h w) c')
        return x
```

**Timestep Embedder**:
```python
class TimestepEmbedder(nn.Module):
    """
    Sinusoidal timestep embedding + MLP.
    """
    def __init__(
        self,
        frequency_embedding_size: int = 256,
        adaln_tembed_dim: int = 512,
    ):
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        
        self.mlp = nn.Sequential(
            ReplicatedLinear(frequency_embedding_size, adaln_tembed_dim, bias=True),
            nn.SiLU(),
            ReplicatedLinear(adaln_tembed_dim, adaln_tembed_dim, bias=True),
        )
    
    def forward(
        self,
        timestep: torch.Tensor,  # [B] or [B, T]
        latent_shape: tuple,     # (T, H, W)
    ) -> torch.Tensor:
        """
        Returns: [B, T, adaln_tembed_dim]
        """
        B = timestep.shape[0]
        T, H, W = latent_shape
        N_t = T  # Assuming patch_size[0] = 1
        
        # Expand to per-frame timesteps
        if len(timestep.shape) == 1:
            timestep = timestep.unsqueeze(1).expand(-1, N_t)  # [B, T]
        
        # Sinusoidal embedding (FP32)
        with torch.amp.autocast(device_type='cuda', dtype=torch.float32):
            t_freq = timestep_embedding(timestep.flatten(), self.frequency_embedding_size)
            t_emb = self.mlp(t_freq)  # [B*T, C_t]
        
        # Reshape to [B, T, C_t]
        t_emb = t_emb.reshape(B, N_t, -1)
        
        return t_emb


def timestep_embedding(
    timesteps: torch.Tensor,
    dim: int,
    max_period: int = 10000,
) -> torch.Tensor:
    """
    Create sinusoidal timestep embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(half, dtype=torch.float32, device=timesteps.device) / half
    )
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding
```

**Caption Embedder**:
```python
class CaptionEmbedder(nn.Module):
    """
    Project text embeddings and handle attention masking.
    """
    def __init__(
        self,
        caption_channels: int = 4096,
        hidden_size: int = 4096,
        text_tokens_zero_pad: bool = True,
    ):
        super().__init__()
        self.text_tokens_zero_pad = text_tokens_zero_pad
        
        # Two-layer MLP projection
        self.proj = nn.Sequential(
            ReplicatedLinear(caption_channels, hidden_size, bias=True),
            nn.SiLU(),
            ReplicatedLinear(hidden_size, hidden_size, bias=True),
        )
    
    def forward(
        self,
        encoder_hidden_states: torch.Tensor,        # [B, N_text, C_text] or [B, 1, N_text, C_text]
        encoder_attention_mask: torch.Tensor | None = None,  # [B, N_text]
    ) -> tuple[torch.Tensor, list[int]]:
        """
        Returns:
            y: [1, N_text_total, C] - compacted representation
            y_seqlen: [B] - valid tokens per sample
        """
        # Handle extra dimension
        if len(encoder_hidden_states.shape) == 4:
            encoder_hidden_states = encoder_hidden_states.squeeze(1)  # [B, N_text, C_text]
        
        # Project
        y = self.proj(encoder_hidden_states)  # [B, N_text, C]
        
        # Handle attention masking
        if self.text_tokens_zero_pad and encoder_attention_mask is not None:
            # Zero out padded tokens
            y = y * encoder_attention_mask.unsqueeze(-1)
            # Create all-ones mask (padded tokens already zeroed)
            encoder_attention_mask = torch.ones_like(encoder_attention_mask)
        
        # Compact representation: remove padding
        if encoder_attention_mask is not None:
            y_seqlen = encoder_attention_mask.sum(dim=1).long().tolist()
            
            # Extract valid tokens
            y_list = [y[i, :seq_len] for i, seq_len in enumerate(y_seqlen)]
            
            # Concatenate all batches
            y = torch.cat(y_list, dim=0).unsqueeze(0)  # [1, N_total, C]
        else:
            B, N, C = y.shape
            y_seqlen = [N] * B
            y = y.reshape(1, -1, C)
        
        return y, y_seqlen
```

### 8. Final Layer

```python
class FinalLayer(nn.Module):
    """
    Final output projection with AdaLN modulation.
    """
    def __init__(
        self,
        hidden_size: int,
        out_channels: int,
        adaln_tembed_dim: int,
    ):
        super().__init__()
        
        # AdaLN for final layer (2 parameters: scale and shift)
        self.adaln_modulation = nn.Sequential(
            nn.SiLU(),
            ReplicatedLinear(adaln_tembed_dim, 2 * hidden_size, bias=True),
        )
        
        self.norm = RMSNorm(hidden_size, eps=1e-6, elementwise_affine=False)
        
        # Output projection (C -> patch_size^3 * out_channels)
        patch_dim = out_channels  # Simplified for patch_size=[1,2,2]
        self.proj = ReplicatedLinear(hidden_size, patch_dim, bias=True)
    
    def forward(
        self,
        x: torch.Tensor,  # [B, N, C]
        t: torch.Tensor,  # [B, T, C_t]
    ) -> torch.Tensor:
        """
        Returns: [B, N, out_channels * patch_size^3]
        """
        B, N, C = x.shape
        _, T_emb, _ = t.shape
        
        # AdaLN modulation (FP32)
        with torch.amp.autocast(device_type='cuda', dtype=torch.float32):
            shift, scale = self.adaln_modulation(t).unsqueeze(2).chunk(2, dim=-1)
        
        # Modulate
        x = modulate_fp32(self.norm, x.view(B, T_emb, -1, C), shift, scale)
        x = x.reshape(B, N, C)
        
        # Project
        x = self.proj(x)
        
        return x


def modulate_fp32(
    norm: nn.Module,
    x: torch.Tensor,
    shift: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    """
    Apply modulation in FP32 for numerical stability.
    """
    orig_dtype = x.dtype
    
    with torch.amp.autocast(device_type='cuda', dtype=torch.float32):
        x = x.to(torch.float32)
        shift = shift.to(torch.float32)
        scale = scale.to(torch.float32)
        
        # Normalize and modulate
        x_norm = norm(x)
        x_mod = x_norm * (1 + scale) + shift
    
    return x_mod.to(orig_dtype)
```

---

## Weight Conversion Plan

### Mapping Strategy

**Original LongCat → Native FastVideo**

| Original Parameter | Native FastVideo Parameter | Conversion |
|-------------------|---------------------------|-----------|
| `x_embedder.proj.weight` | `patch_embed.proj.weight` | Direct copy |
| `t_embedder.mlp.0.weight` | `time_embedder.mlp.0.weight` | Direct copy |
| `y_embedder.y_proj.0.weight` | `caption_embedder.proj.0.weight` | Direct copy |
| `blocks.{i}.attn.qkv.weight` | `blocks.{i}.self_attn.to_q.weight`<br>`blocks.{i}.self_attn.to_k.weight`<br>`blocks.{i}.self_attn.to_v.weight` | **Split** into 3 parts |
| `blocks.{i}.attn.proj.weight` | `blocks.{i}.self_attn.to_out.weight` | Direct copy |
| `blocks.{i}.cross_attn.q_linear.weight` | `blocks.{i}.cross_attn.to_q.weight` | Direct copy |
| `blocks.{i}.cross_attn.kv_linear.weight` | `blocks.{i}.cross_attn.to_k.weight`<br>`blocks.{i}.cross_attn.to_v.weight` | **Split** into 2 parts |
| `blocks.{i}.cross_attn.proj.weight` | `blocks.{i}.cross_attn.to_out.weight` | Direct copy |
| `blocks.{i}.ffn.w1.weight` | `blocks.{i}.ffn.gate.weight` | Direct copy |
| `blocks.{i}.ffn.w2.weight` | `blocks.{i}.ffn.down.weight` | Direct copy |
| `blocks.{i}.ffn.w3.weight` | `blocks.{i}.ffn.up.weight` | Direct copy |
| `final_layer.linear.weight` | `final_layer.proj.weight` | Direct copy |

### Conversion Script

**File**: `scripts/checkpoint_conversion/longcat_native_weights_converter.py`

```python
#!/usr/bin/env python3
"""
Convert LongCat wrapper weights to native FastVideo format.

Key transformations:
1. Split fused QKV projections (self-attention)
2. Split fused KV projections (cross-attention)
3. Rename parameters to match native implementation
"""

import argparse
import glob
from collections import OrderedDict
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from tqdm import tqdm


def split_qkv(qkv_weight: torch.Tensor, qkv_bias: torch.Tensor | None = None):
    """
    Split fused QKV projection into separate Q, K, V.
    
    Args:
        qkv_weight: [3*dim, dim]
        qkv_bias: [3*dim]
    
    Returns:
        (q_weight, k_weight, v_weight), (q_bias, k_bias, v_bias)
    """
    dim = qkv_weight.shape[0] // 3
    q, k, v = torch.chunk(qkv_weight, 3, dim=0)
    
    if qkv_bias is not None:
        q_bias, k_bias, v_bias = torch.chunk(qkv_bias, 3, dim=0)
    else:
        q_bias = k_bias = v_bias = None
    
    return (q, k, v), (q_bias, k_bias, v_bias)


def split_kv(kv_weight: torch.Tensor, kv_bias: torch.Tensor | None = None):
    """
    Split fused KV projection into separate K, V.
    
    Args:
        kv_weight: [2*dim, dim]
        kv_bias: [2*dim]
    
    Returns:
        (k_weight, v_weight), (k_bias, v_bias)
    """
    dim = kv_weight.shape[0] // 2
    k, v = torch.chunk(kv_weight, 2, dim=0)
    
    if kv_bias is not None:
        k_bias, v_bias = torch.chunk(kv_bias, 2, dim=0)
    else:
        k_bias = v_bias = None
    
    return (k, v), (k_bias, v_bias)


def convert_weights(source_weights: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """
    Convert LongCat wrapper weights to native FastVideo format.
    """
    converted = OrderedDict()
    
    # Track processed keys
    processed_keys = set()
    
    print("Converting weights...")
    
    for key, value in tqdm(source_weights.items()):
        if key in processed_keys:
            continue
        
        # === Embedders ===
        if key.startswith("x_embedder."):
            new_key = key.replace("x_embedder.", "patch_embed.")
            converted[new_key] = value
        
        elif key.startswith("t_embedder.mlp.0."):
            new_key = key.replace("t_embedder.mlp.0.", "time_embedder.mlp.0.")
            converted[new_key] = value
        
        elif key.startswith("t_embedder.mlp.2."):
            new_key = key.replace("t_embedder.mlp.2.", "time_embedder.mlp.2.")
            converted[new_key] = value
        
        elif key.startswith("y_embedder.y_proj.0."):
            new_key = key.replace("y_embedder.y_proj.0.", "caption_embedder.proj.0.")
            converted[new_key] = value
        
        elif key.startswith("y_embedder.y_proj.2."):
            new_key = key.replace("y_embedder.y_proj.2.", "caption_embedder.proj.2.")
            converted[new_key] = value
        
        # === Transformer Blocks ===
        elif ".attn.qkv." in key:
            # Split QKV for self-attention
            block_idx = key.split(".")[1]
            
            # Get weight and bias
            qkv_weight = value
            qkv_bias_key = key.replace(".weight", ".bias")
            qkv_bias = source_weights.get(qkv_bias_key)
            
            # Split
            (q, k, v), (q_bias, k_bias, v_bias) = split_qkv(qkv_weight, qkv_bias)
            
            # Store
            converted[f"blocks.{block_idx}.self_attn.to_q.weight"] = q
            converted[f"blocks.{block_idx}.self_attn.to_k.weight"] = k
            converted[f"blocks.{block_idx}.self_attn.to_v.weight"] = v
            
            if q_bias is not None:
                converted[f"blocks.{block_idx}.self_attn.to_q.bias"] = q_bias
                converted[f"blocks.{block_idx}.self_attn.to_k.bias"] = k_bias
                converted[f"blocks.{block_idx}.self_attn.to_v.bias"] = v_bias
            
            # Mark both weight and bias as processed
            processed_keys.add(key)
            if qkv_bias is not None:
                processed_keys.add(qkv_bias_key)
        
        elif ".attn.proj." in key:
            new_key = key.replace(".attn.proj.", ".self_attn.to_out.")
            converted[new_key] = value
        
        elif ".attn.q_norm." in key or ".attn.k_norm." in key:
            # Norms stay the same
            new_key = key.replace(".attn.", ".self_attn.")
            converted[new_key] = value
        
        elif ".cross_attn.q_linear." in key:
            new_key = key.replace(".cross_attn.q_linear.", ".cross_attn.to_q.")
            converted[new_key] = value
        
        elif ".cross_attn.kv_linear." in key:
            # Split KV for cross-attention
            block_idx = key.split(".")[1]
            
            # Get weight and bias
            kv_weight = value
            kv_bias_key = key.replace(".weight", ".bias")
            kv_bias = source_weights.get(kv_bias_key)
            
            # Split
            (k, v), (k_bias, v_bias) = split_kv(kv_weight, kv_bias)
            
            # Store
            converted[f"blocks.{block_idx}.cross_attn.to_k.weight"] = k
            converted[f"blocks.{block_idx}.cross_attn.to_v.weight"] = v
            
            if k_bias is not None:
                converted[f"blocks.{block_idx}.cross_attn.to_k.bias"] = k_bias
                converted[f"blocks.{block_idx}.cross_attn.to_v.bias"] = v_bias
            
            # Mark both weight and bias as processed
            processed_keys.add(key)
            if kv_bias is not None:
                processed_keys.add(kv_bias_key)
        
        elif ".cross_attn.proj." in key:
            new_key = key.replace(".cross_attn.proj.", ".cross_attn.to_out.")
            converted[new_key] = value
        
        elif ".cross_attn.q_norm." in key or ".cross_attn.k_norm." in key:
            # Norms stay the same
            converted[key] = value
        
        elif ".ffn.w1." in key:
            new_key = key.replace(".ffn.w1.", ".ffn.gate.")
            converted[new_key] = value
        
        elif ".ffn.w2." in key:
            new_key = key.replace(".ffn.w2.", ".ffn.down.")
            converted[new_key] = value
        
        elif ".ffn.w3." in key:
            new_key = key.replace(".ffn.w3.", ".ffn.up.")
            converted[new_key] = value
        
        elif ".adaLN_modulation." in key or ".mod_norm_" in key or ".pre_crs_attn_norm." in key:
            # Norms and modulations stay the same
            converted[key] = value
        
        # === Final Layer ===
        elif key.startswith("final_layer.linear."):
            new_key = key.replace("final_layer.linear.", "final_layer.proj.")
            converted[new_key] = value
        
        elif key.startswith("final_layer.adaLN_modulation."):
            # Keep same name
            converted[key] = value
        
        else:
            # Unknown key, keep as-is and warn
            print(f"Warning: Unknown key '{key}', keeping as-is")
            converted[key] = value
    
    return converted


def validate_conversion(original: dict, converted: dict):
    """
    Validate that conversion preserved all parameters correctly.
    """
    print("\nValidating conversion...")
    
    # Count parameters
    orig_count = sum(p.numel() for p in original.values())
    conv_count = sum(p.numel() for p in converted.values())
    
    print(f"  Original parameters: {orig_count:,}")
    print(f"  Converted parameters: {conv_count:,}")
    
    if orig_count != conv_count:
        print(f"  ❌ Parameter count mismatch!")
        return False
    
    print(f"  ✓ Parameter count matches")
    
    # Verify QKV splits
    print("\n  Verifying QKV splits...")
    for i in range(48):  # 48 blocks
        # Check self-attention QKV
        orig_qkv = original.get(f"blocks.{i}.attn.qkv.weight")
        if orig_qkv is not None:
            conv_q = converted[f"blocks.{i}.self_attn.to_q.weight"]
            conv_k = converted[f"blocks.{i}.self_attn.to_k.weight"]
            conv_v = converted[f"blocks.{i}.self_attn.to_v.weight"]
            
            reconstructed = torch.cat([conv_q, conv_k, conv_v], dim=0)
            
            if not torch.allclose(orig_qkv, reconstructed):
                print(f"    ❌ QKV mismatch in block {i}")
                return False
        
        # Check cross-attention KV
        orig_kv = original.get(f"blocks.{i}.cross_attn.kv_linear.weight")
        if orig_kv is not None:
            conv_k = converted[f"blocks.{i}.cross_attn.to_k.weight"]
            conv_v = converted[f"blocks.{i}.cross_attn.to_v.weight"]
            
            reconstructed = torch.cat([conv_k, conv_v], dim=0)
            
            if not torch.allclose(orig_kv, reconstructed):
                print(f"    ❌ KV mismatch in block {i}")
                return False
    
    print(f"  ✓ All splits verified")
    
    return True


def main():
    parser = argparse.ArgumentParser(description="Convert LongCat weights to native FastVideo format")
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Path to source weights (longcat-for-fastvideo/transformer/)"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to output converted weights"
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Run validation after conversion"
    )
    
    args = parser.parse_args()
    
    # Load source weights
    print(f"Loading weights from {args.source}...")
    source_path = Path(args.source)
    
    # Load all shards
    shard_files = sorted(glob.glob(str(source_path / "*.safetensors")))
    if not shard_files:
        print(f"Error: No safetensors files found in {source_path}")
        return
    
    print(f"Found {len(shard_files)} shard(s)")
    
    source_weights = {}
    for shard_file in shard_files:
        print(f"  Loading {Path(shard_file).name}...")
        source_weights.update(load_file(shard_file))
    
    print(f"Loaded {len(source_weights)} parameters")
    
    # Convert
    converted_weights = convert_weights(source_weights)
    
    print(f"\nConverted to {len(converted_weights)} parameters")
    
    # Validate if requested
    if args.validate:
        if not validate_conversion(source_weights, converted_weights):
            print("\n❌ Validation failed!")
            return
        print("\n✓ Validation passed!")
    
    # Save
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    
    output_file = output_path / "model.safetensors"
    print(f"\nSaving to {output_file}...")
    save_file(converted_weights, str(output_file))
    
    print("\n✅ Conversion complete!")
    
    # Print size info
    size_gb = output_file.stat().st_size / (1024**3)
    print(f"Output size: {size_gb:.2f} GB")


if __name__ == "__main__":
    main()
```

### Usage

```bash
# Convert weights
python scripts/checkpoint_conversion/longcat_native_weights_converter.py \
    --source weights/longcat-for-fastvideo/transformer \
    --output weights/longcat-native/transformer \
    --validate

# Update model_index.json to point to native implementation
# transformer: ["fastvideo.models.dits", "LongCatTransformer3DModel"]
```

---

## Implementation Phases

### Phase 2.1: Foundation (Week 1)

**Goal**: Implement basic layers without attention

**Tasks**:
- [ ] Create `fastvideo/models/dits/longcat.py` skeleton
- [ ] Implement embeddings:
  - [ ] `PatchEmbed3D`
  - [ ] `TimestepEmbedder`
  - [ ] `CaptionEmbedder`
- [ ] Implement `LongCatSwiGLUFFN`
- [ ] Implement `FinalLayer`
- [ ] Create normalization helpers:
  - [ ] `RMSNorm` with FP32 support (if not exists)
  - [ ] `modulate_fp32` utility
- [ ] Unit tests for each component

**Validation**:
- [ ] Each component loads converted weights correctly
- [ ] Forward passes produce expected output shapes
- [ ] FP32 operations maintain numerical precision

### Phase 2.2: 3D RoPE (Week 1-2)

**Goal**: Implement 3D rotary position embeddings

**Tasks**:
- [ ] Create `fastvideo/layers/rotary_embedding_3d.py`
- [ ] Implement `compute_3d_rope()`
- [ ] Implement `apply_3d_rotary_emb()`
- [ ] Unit tests with known position patterns
- [ ] Compare with original LongCat RoPE output

**Validation**:
- [ ] RoPE values match original implementation
- [ ] Correct frequency bands for T/H/W dimensions
- [ ] Caching works correctly

### Phase 2.3: Attention Modules (Week 2)

**Goal**: Implement self and cross attention

**Tasks**:
- [ ] Implement `LongCatSelfAttention`:
  - [ ] Separate Q/K/V projections with `ReplicatedLinear`
  - [ ] 3D RoPE integration
  - [ ] Use `DistributedAttention` backend
  - [ ] Per-head RMS normalization
- [ ] Implement `LongCatCrossAttention`:
  - [ ] Separate projections
  - [ ] Variable-length text handling
  - [ ] Attention masking for padding
- [ ] Unit tests for both attention types

**Validation**:
- [ ] Load split weights correctly
- [ ] Attention outputs match wrapper (within FP precision)
- [ ] RoPE applied correctly
- [ ] Masking works for variable-length sequences

### Phase 2.4: Complete Model (Week 2-3)

**Goal**: Assemble full transformer model

**Tasks**:
- [ ] Implement `LongCatBlock`:
  - [ ] AdaLN modulation
  - [ ] Self-attention integration
  - [ ] Cross-attention integration
  - [ ] FFN integration
  - [ ] Residual connections with gating
- [ ] Implement `LongCatTransformer3DModel`:
  - [ ] Complete forward pass
  - [ ] Parameter ordering
  - [ ] Shape handling
- [ ] Add config class `LongCatVideoConfig`
- [ ] Add parameter name mappings
- [ ] Implement weight loader

**Validation**:
- [ ] Full model instantiates correctly
- [ ] Loads converted weights
- [ ] Forward pass completes
- [ ] Output shape matches wrapper

### Phase 2.5: CFG-zero Integration (Week 3)

**Goal**: Integrate CFG-zero at model or stage level

**Decision Point**: Implement in model or denoising stage?

**Option A: Model-level (Recommended)**
- Add `do_cfg_zero` flag to model config
- Override `forward()` to handle batched CFG internally
- Pros: Cleaner separation, reusable
- Cons: More changes to model class

**Option B: Stage-level (Current approach)**
- Keep `LongCatDenoisingStage` with CFG-zero
- Model remains agnostic to guidance method
- Pros: Less model complexity, matches Phase 1
- Cons: CFG logic outside model

**Tasks**:
- [ ] Decide on implementation approach
- [ ] If model-level: Add CFG-zero to forward pass
- [ ] If stage-level: Update `LongCatDenoisingStage` to use native model
- [ ] Test both CFG and non-CFG paths
- [ ] Validate guidance scale behavior

**Validation**:
- [ ] CFG-zero formula produces same results as wrapper
- [ ] Guidance scale sweep shows expected behavior
- [ ] Performance comparable to wrapper

### Phase 2.6: Pipeline Integration (Week 3)

**Goal**: Connect native model to FastVideo pipeline

**Tasks**:
- [ ] Update `longcat_pipeline.py` to use native model
- [ ] Update model loader to recognize native class
- [ ] Test CPU offloading
- [ ] Test FSDP sharding (if using multiple GPUs)
- [ ] Add FSDP shard conditions
- [ ] Test end-to-end inference

**Validation**:
- [ ] Pipeline loads native model
- [ ] Generates videos successfully
- [ ] Output quality matches wrapper
- [ ] Memory usage acceptable

### Phase 2.7: BSA Integration (Week 3-4)

**Goal**: Add block-sparse attention as FastVideo backend

**Tasks**:
- [ ] Locate BSA CUDA kernels in original LongCat
- [ ] Extract and adapt for FastVideo
- [ ] Create `fastvideo/attention/block_sparse_attention.py`
- [ ] Add `AttentionBackendEnum.BSA`
- [ ] Integrate with `DistributedAttention` dispatcher
- [ ] Test on LongCat
- [ ] (Optional) Test on other models for 720p+

**Validation**:
- [ ] BSA backend selectable via config
- [ ] Numerically equivalent to dense attention
- [ ] Memory savings at high resolution
- [ ] Performance improvement

### Phase 2.8: Advanced Features (Week 4)

**Goal**: Implement remaining LongCat features

**Tasks**:
- [ ] LoRA support:
  - [ ] Test FastVideo's native LoRA system
  - [ ] Adapt LongCat LoRAs if needed
- [ ] KV caching for video continuation
- [ ] Context parallelism (if needed)
- [ ] Distilled inference support
- [ ] Documentation

**Validation**:
- [ ] LoRA loading and inference
- [ ] Video continuation works
- [ ] Context parallelism scales correctly

### Phase 2.9: Optimization & Testing (Week 4-5)

**Goal**: Validate, benchmark, and optimize

**Tasks**:
- [ ] Numerical equivalence tests:
  - [ ] Compare outputs with wrapper at every block
  - [ ] Test multiple resolutions
  - [ ] Test different inference steps
- [ ] Performance benchmarks:
  - [ ] Inference speed vs wrapper
  - [ ] Memory usage
  - [ ] FSDP scaling
- [ ] Compilation testing:
  - [ ] Test `torch.compile()` on model
  - [ ] Measure speedup
- [ ] Integration tests:
  - [ ] Multiple prompts
  - [ ] Different guidance scales
  - [ ] Different resolutions

**Success Criteria**:
- [ ] Output identical to wrapper (within 1e-5 absolute error)
- [ ] Performance ≥ wrapper
- [ ] Memory usage ≤ wrapper + 10%
- [ ] All FastVideo features work (FSDP, offloading, etc.)

---

## Testing Strategy

### Unit Tests

**File**: `tests/models/dits/test_longcat.py`

```python
import pytest
import torch
from fastvideo.models.dits.longcat import (
    LongCatSelfAttention,
    LongCatCrossAttention,
    LongCatSwiGLUFFN,
    LongCatBlock,
    LongCatTransformer3DModel,
)
from fastvideo.configs.models.dits import LongCatVideoConfig


class TestLongCatComponents:
    @pytest.fixture
    def config(self):
        return LongCatVideoConfig()
    
    def test_self_attention_shape(self, config):
        """Test self-attention output shape."""
        attn = LongCatSelfAttention(
            dim=4096,
            num_heads=32,
            config=config,
        )
        
        B, N, C = 2, 1000, 4096
        x = torch.randn(B, N, C)
        latent_shape = (20, 30, 50)
        
        out = attn(x, latent_shape)
        
        assert out.shape == (B, N, C)
    
    def test_cross_attention_shape(self, config):
        """Test cross-attention with variable-length text."""
        attn = LongCatCrossAttention(
            dim=4096,
            num_heads=32,
            config=config,
        )
        
        B, N_img, C = 2, 1000, 4096
        N_text_total = 450  # 200 + 250 for two samples
        
        x = torch.randn(B, N_img, C)
        y = torch.randn(1, N_text_total, C)
        y_seqlen = [200, 250]
        
        out = attn(x, y, y_seqlen, latent_shape=(20, 30, 50))
        
        assert out.shape == (B, N_img, C)
    
    def test_swiglu_ffn_shape(self):
        """Test SwiGLU FFN output shape."""
        ffn = LongCatSwiGLUFFN(dim=4096, hidden_dim=11008)
        
        B, N, C = 2, 1000, 4096
        x = torch.randn(B, N, C)
        
        out = ffn(x)
        
        assert out.shape == (B, N, C)
    
    def test_block_forward(self, config):
        """Test full block forward pass."""
        block = LongCatBlock(
            hidden_size=4096,
            num_heads=32,
            mlp_ratio=4,
            adaln_tembed_dim=512,
            config=config,
        )
        
        B, N, C = 2, 1000, 4096
        T, H, W = 20, 30, 50
        
        x = torch.randn(B, N, C)
        y = torch.randn(1, 450, C)
        t = torch.randn(B, T, 512)
        y_seqlen = [200, 250]
        
        out = block(x, y, t, y_seqlen, latent_shape=(T, H, W))
        
        assert out.shape == (B, N, C)
    
    def test_model_forward(self, config):
        """Test full model forward pass."""
        model = LongCatTransformer3DModel(
            config=config,
            hf_config={},
        )
        
        B, C_in, T, H, W = 2, 16, 21, 60, 104
        N_text = 512
        C_text = 4096
        
        hidden_states = torch.randn(B, C_in, T, H, W)
        encoder_hidden_states = torch.randn(B, N_text, C_text)
        timestep = torch.randint(0, 1000, (B,))
        
        out = model(hidden_states, encoder_hidden_states, timestep)
        
        C_out = config.out_channels
        assert out.shape == (B, C_out, T, H, W)
```

### Integration Tests

**File**: `tests/pipelines/test_longcat_pipeline.py`

```python
import pytest
import torch
from fastvideo import VideoGenerator
from fastvideo.fastvideo_args import FastVideoArgs


class TestLongCatPipeline:
    @pytest.fixture
    def generator(self):
        """Create video generator with native LongCat."""
        return VideoGenerator.from_pretrained(
            "weights/longcat-native",
            num_gpus=1,
            use_fsdp_inference=False,
            dit_cpu_offload=True,
        )
    
    def test_basic_inference(self, generator):
        """Test basic text-to-video generation."""
        video = generator.generate_video(
            prompt="A cat playing piano",
            num_inference_steps=2,  # Fast test
            height=480,
            width=832,
            num_frames=25,  # Reduced for testing
            seed=42,
        )
        
        assert video is not None
        assert video.shape == (25, 480, 832, 3)
    
    def test_cfg_guidance(self, generator):
        """Test different guidance scales."""
        for guidance_scale in [1.0, 4.0, 7.0]:
            video = generator.generate_video(
                prompt="A cat playing piano",
                guidance_scale=guidance_scale,
                num_inference_steps=2,
                height=480,
                width=832,
                num_frames=25,
                seed=42,
            )
            assert video is not None
```

### Numerical Equivalence Tests

**File**: `tests/models/test_longcat_equivalence.py`

```python
import pytest
import torch
from fastvideo.models.dits.longcat import LongCatTransformer3DModel
from fastvideo.third_party.longcat_video.modules.longcat_video_dit import (
    LongCatVideoTransformer3DModel as WrapperModel
)


class TestNumericalEquivalence:
    """
    Test that native implementation produces identical outputs to wrapper.
    """
    
    @pytest.fixture
    def inputs(self):
        """Create test inputs."""
        torch.manual_seed(42)
        
        B, C_in, T, H, W = 1, 16, 21, 60, 104
        N_text = 512
        C_text = 4096
        
        return {
            "hidden_states": torch.randn(B, C_in, T, H, W),
            "encoder_hidden_states": torch.randn(B, N_text, C_text),
            "timestep": torch.randint(0, 1000, (B,)),
        }
    
    def test_embeddings_equivalence(self, inputs):
        """Test that embeddings match."""
        # Load both models
        wrapper = WrapperModel.from_pretrained("weights/longcat-for-fastvideo/transformer")
        native = LongCatTransformer3DModel.from_pretrained("weights/longcat-native/transformer")
        
        wrapper.eval()
        native.eval()
        
        # Extract embeddings
        with torch.no_grad():
            # Wrapper embeddings
            x_wrapper = wrapper.x_embedder(inputs["hidden_states"])
            t_wrapper = wrapper.t_embedder(inputs["timestep"], ...)
            y_wrapper = wrapper.y_embedder(inputs["encoder_hidden_states"])
            
            # Native embeddings
            x_native = native.patch_embed(inputs["hidden_states"])
            t_native = native.time_embedder(inputs["timestep"], ...)
            y_native = native.caption_embedder(inputs["encoder_hidden_states"])
        
        # Compare
        assert torch.allclose(x_wrapper, x_native, atol=1e-5)
        assert torch.allclose(t_wrapper, t_native, atol=1e-5)
        # y may differ due to compaction, compare element-wise
    
    def test_full_model_equivalence(self, inputs):
        """Test that full model outputs match."""
        wrapper = WrapperModel.from_pretrained("weights/longcat-for-fastvideo/transformer")
        native = LongCatTransformer3DModel.from_pretrained("weights/longcat-native/transformer")
        
        wrapper.eval()
        native.eval()
        
        with torch.no_grad():
            out_wrapper = wrapper(**inputs)
            out_native = native(**inputs)
        
        # Should be nearly identical
        max_diff = (out_wrapper - out_native).abs().max().item()
        print(f"Max difference: {max_diff}")
        
        assert torch.allclose(out_wrapper, out_native, atol=1e-4, rtol=1e-3), \
            f"Outputs differ by {max_diff}"
```

### Performance Benchmarks

**File**: `benchmarks/longcat_native_vs_wrapper.py`

```python
import time
import torch
from fastvideo import VideoGenerator


def benchmark_inference(model_path, num_runs=5):
    """Benchmark inference time and memory."""
    generator = VideoGenerator.from_pretrained(
        model_path,
        num_gpus=1,
        use_fsdp_inference=False,
        dit_cpu_offload=True,
    )
    
    times = []
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    for i in range(num_runs):
        start = time.time()
        
        video = generator.generate_video(
            prompt="A cat playing piano",
            num_inference_steps=20,
            height=480,
            width=832,
            num_frames=65,
            seed=42 + i,
        )
        
        elapsed = time.time() - start
        times.append(elapsed)
        print(f"  Run {i+1}: {elapsed:.2f}s")
    
    peak_memory = torch.cuda.max_memory_allocated() / (1024**3)
    
    return {
        "mean_time": sum(times) / len(times),
        "std_time": torch.tensor(times).std().item(),
        "peak_memory_gb": peak_memory,
    }


if __name__ == "__main__":
    print("Benchmarking wrapper...")
    wrapper_results = benchmark_inference("weights/longcat-for-fastvideo")
    
    print("\nBenchmarking native...")
    native_results = benchmark_inference("weights/longcat-native")
    
    print("\n=== Results ===")
    print(f"Wrapper: {wrapper_results['mean_time']:.2f}s ± {wrapper_results['std_time']:.2f}s, "
          f"{wrapper_results['peak_memory_gb']:.2f} GB")
    print(f"Native:  {native_results['mean_time']:.2f}s ± {native_results['std_time']:.2f}s, "
          f"{native_results['peak_memory_gb']:.2f} GB")
    
    speedup = wrapper_results['mean_time'] / native_results['mean_time']
    memory_ratio = native_results['peak_memory_gb'] / wrapper_results['peak_memory_gb']
    
    print(f"\nSpeedup: {speedup:.2f}x")
    print(f"Memory ratio: {memory_ratio:.2f}x")
```

---

## Success Criteria

### Functional Requirements

- [ ] **Model loads successfully** from converted weights
- [ ] **Generates videos** with quality matching wrapper
- [ ] **Numerical equivalence** to wrapper (< 1e-4 max absolute error)
- [ ] **All resolutions work**: 480p baseline, 720p with BSA
- [ ] **CFG-zero** guidance produces correct behavior
- [ ] **Parameter order** follows FastVideo conventions
- [ ] **Noise negation** handled correctly for flow matching

### Performance Requirements

- [ ] **Inference speed** ≥ wrapper (ideally 10-20% faster)
- [ ] **Memory usage** ≤ wrapper + 10%
- [ ] **FSDP sharding** reduces memory with multiple GPUs
- [ ] **Compilation** works with `torch.compile()`
- [ ] **CPU offloading** works correctly

### Integration Requirements

- [ ] **FastVideo API** (`VideoGenerator.from_pretrained()`) works
- [ ] **All pipeline stages** compatible
- [ ] **LoRA loading** works with FastVideo's system
- [ ] **Model registry** recognizes native implementation
- [ ] **Config system** properly configured

### Code Quality Requirements

- [ ] **Type hints** throughout
- [ ] **Docstrings** for all public methods
- [ ] **Unit tests** for all components (>80% coverage)
- [ ] **Integration tests** for end-to-end workflows
- [ ] **No regression** in other FastVideo models

---

## Risk Mitigation

### Risk 1: Numerical Differences

**Risk**: Native implementation produces different outputs than wrapper

**Mitigation**:
1. Test each component independently before integration
2. Use FP32 for all normalization and modulation operations
3. Compare intermediate activations block-by-block
4. Use identical random seeds for testing
5. Check for dtype mismatches

### Risk 2: Performance Degradation

**Risk**: Native implementation is slower than wrapper

**Mitigation**:
1. Profile hotspots early
2. Use `ReplicatedLinear` for proper tensor parallelism
3. Enable compilation where possible
4. Optimize attention backend selection
5. Consider BSA for high-resolution inference

### Risk 3: Weight Conversion Issues

**Risk**: Converted weights don't load or produce wrong results

**Mitigation**:
1. Validate conversion with numerical checks
2. Test reconstruction of fused weights
3. Compare parameter counts before/after
4. Load and test each component individually
5. Use automated validation script

### Risk 4: BSA Integration Complexity

**Risk**: Block-sparse attention is difficult to extract/adapt

**Mitigation**:
1. Start without BSA, add later
2. Use original LongCat BSA as reference
3. Test on simple attention patterns first
4. Fall back to dense attention if BSA fails
5. Make BSA optional via config flag

### Risk 5: Breaking Other Models

**Risk**: Changes to shared code affect other FastVideo models

**Mitigation**:
1. Minimize changes to core FastVideo code
2. Add new features as optional flags
3. Run regression tests on all models
4. Keep LongCat-specific code isolated
5. Code review before merging

### Risk 6: Timeline Overruns

**Risk**: Implementation takes longer than 4-5 weeks

**Mitigation**:
1. Prioritize core functionality over advanced features
2. Implement incrementally with working checkpoints
3. Skip non-critical features (BSA, context parallelism) initially
4. Use wrapper as fallback if needed
5. Adjust scope based on progress

---

## Appendix: File Checklist

### New Files to Create

- [ ] `fastvideo/models/dits/longcat.py` - Native DiT implementation
- [ ] `fastvideo/layers/rotary_embedding_3d.py` - 3D RoPE utilities
- [ ] `fastvideo/attention/block_sparse_attention.py` - BSA backend
- [ ] `fastvideo/configs/models/dits/longcat.py` - Config class
- [ ] `scripts/checkpoint_conversion/longcat_native_weights_converter.py` - Weight converter
- [ ] `tests/models/dits/test_longcat.py` - Unit tests
- [ ] `tests/pipelines/test_longcat_pipeline.py` - Integration tests
- [ ] `tests/models/test_longcat_equivalence.py` - Numerical tests
- [ ] `benchmarks/longcat_native_vs_wrapper.py` - Performance tests
- [ ] `docs/models/longcat.md` - Documentation

### Files to Modify

- [ ] `fastvideo/pipelines/basic/longcat/longcat_pipeline.py` - Use native model
- [ ] `fastvideo/models/registry.py` - Register native model
- [ ] `fastvideo/configs/pipelines/longcat.py` - Update config
- [ ] (Optional) `fastvideo/pipelines/stages/denoising.py` - Add CFG-zero option
- [ ] (Optional) `fastvideo/attention/__init__.py` - Export BSA backend

### Files to Keep (Phase 1 Wrapper)

- Keep `fastvideo/third_party/longcat_video/` until validation complete
- Use as reference and fallback
- Remove after Phase 2 validation passes

---

## Next Steps

1. **Review this plan** with team
2. **Set up development branch** (`phase2-native-longcat`)
3. **Begin Phase 2.1** - Foundation layers
4. **Track progress** using GitHub issues/milestones
5. **Weekly check-ins** to adjust plan as needed

---

**Document Version**: 1.0  
**Last Updated**: November 6, 2025  
**Estimated Completion**: December 2025  
**Owner**: Integration Team



