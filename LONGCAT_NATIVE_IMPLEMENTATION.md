# LongCat Native FastVideo Implementation - Phase 2

**Date**: November 6, 2025  
**Status**: Foundation Complete ✅  
**Next**: Attention Implementation & Weight Conversion

---

## What Was Implemented

### ✅ Core Components Created

#### 1. **Native Model** - `fastvideo/models/dits/longcat.py`

A complete native FastVideo implementation of LongCat that **eliminates all dependencies on third_party code**:

- **Embeddings** (using FastVideo layers):
  - `PatchEmbed3D`: 3D patch embedding with Conv3d
  - `TimestepEmbedder`: Sinusoidal timestep embedding + MLP (using `ReplicatedLinear`)
  - `CaptionEmbedder`: Text embedding projection (using `ReplicatedLinear`)

- **Attention Modules** (placeholders for now, as requested):
  - `LongCatSelfAttention`: Self-attention with separate Q/K/V projections
    - Uses `ReplicatedLinear` instead of `nn.Linear`
    - Uses FastVideo's `DistributedAttention` backend
    - Per-head RMS normalization with FastVideo's `RMSNorm`
    - **TODO**: Implement 3D RoPE in future phase
  
  - `LongCatCrossAttention`: Cross-attention for text conditioning
    - Uses `ReplicatedLinear` for all projections
    - Uses FastVideo's `DistributedAttention` backend
    - **TODO**: Add variable-length text handling

- **FFN** (fully implemented):
  - `LongCatSwiGLUFFN`: SwiGLU feed-forward network
    - Uses `ReplicatedLinear` for all projections (w1/gate, w2/down, w3/up)
    - No bias (as per original LongCat)

- **Transformer Block** (fully implemented):
  - `LongCatTransformerBlock`: Single-stream block with:
    - AdaLN modulation in FP32 (using `ReplicatedLinear`)
    - Self-attention → Cross-attention → FFN sequence
    - Residual connections with gating
    - RMS normalization (using FastVideo's `RMSNorm`)

- **Output Layer** (fully implemented):
  - `FinalLayer`: Final projection with AdaLN modulation

- **Main Model** (fully implemented):
  - `LongCatTransformer3DModel`: Inherits from `CachableDiT`
    - FSDP shard conditions defined
    - FastVideo parameter ordering: `(hidden_states, encoder_hidden_states, timestep)`
    - Supported attention backends: FlashAttention, TorchSDPA
    - Complete forward pass implementation

#### 2. **Model Configuration** - `fastvideo/configs/models/dits/longcat.py`

- `LongCatVideoArchConfig`: Architecture configuration
  - All model hyperparameters (hidden_size=4096, depth=48, etc.)
  - FSDP shard conditions
  - Parameter name mapping (for weight conversion)
  - Attention backend configuration

- `LongCatVideoConfig`: Main model configuration
  - Inherits from `DiTConfig`
  - Prefix: "longcat"

#### 3. **Registry Updates**

- **Model Registry** (`fastvideo/models/registry.py`):
  - Added `LongCatTransformer3DModel` pointing to native implementation
  - Kept `LongCatVideoTransformer3DModel` for Phase 1 wrapper compatibility

- **Config Registry** (`fastvideo/configs/models/dits/__init__.py`):
  - Exported `LongCatVideoConfig`

#### 4. **Pipeline Configuration Updates**

- **`fastvideo/configs/pipelines/longcat.py`**:
  - Imported native `LongCatVideoConfig`
  - Added documentation explaining Phase 1 vs Phase 2 options
  - Maintains backward compatibility with wrapper

#### 5. **Weight Conversion Script** - `scripts/checkpoint_conversion/longcat_native_weights_converter.py`

- Placeholder script structure
- Helper functions for QKV/KV splitting
- **TODO**: Implement full conversion logic

---

## Key Design Decisions

### 1. **No Third-Party Dependencies** ✅

All components reimplemented using FastVideo's native layers:
- `nn.Linear` → `ReplicatedLinear` (for tensor parallelism)
- `nn.LayerNorm` → `RMSNorm` (from FastVideo)
- Custom attention → `DistributedAttention` (from FastVideo)

### 2. **FastVideo Conventions Followed** ✅

- Inherits from `CachableDiT` (supports TeaCache optimization)
- Parameter ordering: `(hidden_states, encoder_hidden_states, timestep)`
- FSDP shard conditions defined
- Config system integration
- Model registry integration

### 3. **Attention as Placeholder** ✅ (As Requested)

- Attention modules use FastVideo's `DistributedAttention`
- **No 3D RoPE yet** - marked with TODO comments
- **No variable-length text handling yet** - marked with TODO
- Basic functionality in place, ready for enhancement

### 4. **Maintains Compatibility** ✅

- Phase 1 wrapper (`LongCatVideoTransformer3DModel`) still available
- Phase 2 native (`LongCatTransformer3DModel`) registered separately
- Users can choose which to use

---

## What Can Be Reused From LongCat

Based on the check of existing FastVideo code and the Phase 1/2 documentation:

### ✅ **Can Use FastVideo Equivalents** (Implemented)

| LongCat Component | FastVideo Equivalent | Used? |
|-------------------|---------------------|-------|
| `nn.Linear` | `ReplicatedLinear` | ✅ Yes |
| `LayerNorm_FP32` | `RMSNorm` with FP32 support | ✅ Yes |
| `FeedForwardSwiGLU` | Custom implementation with `ReplicatedLinear` | ✅ Yes |
| Attention | `DistributedAttention` | ✅ Yes (placeholder) |
| `PatchEmbed3D` | Custom (Conv3d) | ✅ Reimplemented |
| `TimestepEmbedder` | Custom with `ReplicatedLinear` | ✅ Reimplemented |
| `CaptionEmbedder` | Custom with `ReplicatedLinear` | ✅ Reimplemented |

### ⚠️ **Need Custom Implementation** (TODO)

| Component | Status | Notes |
|-----------|--------|-------|
| 3D RoPE | ❌ TODO | Need to implement in future phase |
| Variable-length text | ❌ TODO | Cross-attention needs enhancement |
| CFG-zero | ❌ TODO | Keep in denoising stage or add to model |
| Noise negation | ❌ TODO | For flow matching scheduler |
| BSA (Block-Sparse Attention) | ❌ TODO | Optional, for 720p |

---

## What Still Needs to Be Done

### High Priority

1. **3D RoPE Implementation**
   - Create `fastvideo/layers/rotary_embedding_3d.py`
   - Implement `compute_3d_rope()` and `apply_3d_rotary_emb()`
   - Integrate into `LongCatSelfAttention`

2. **Weight Conversion**
   - Complete `longcat_native_weights_converter.py`
   - Implement QKV splitting logic
   - Implement KV splitting logic
   - Implement parameter renaming according to mapping
   - Validate numerical equivalence

3. **Variable-Length Text Handling**
   - Update `LongCatCrossAttention` to handle compacted text
   - Implement `y_seqlen` processing
   - Add attention masking for padded tokens

4. **CFG-Zero Integration**
   - Decide: model-level or stage-level?
   - Implement optimized guidance scale formula
   - Test with different guidance scales

5. **Flow Matching Integration**
   - Ensure noise negation in correct place
   - Test with `FlowMatchEulerDiscreteScheduler`
   - Validate sigma schedules

### Medium Priority

6. **Testing & Validation**
   - Create unit tests for each component
   - Create integration tests for full model
   - Numerical equivalence tests vs wrapper
   - Performance benchmarks

7. **LoRA Support**
   - Test with FastVideo's native LoRA system
   - Adapt LongCat LoRA format if needed

8. **Advanced Features**
   - KV caching for video continuation
   - BSA for 720p
   - Context parallelism
   - Distilled inference

### Low Priority

9. **Optimization**
   - Enable `torch.compile()`
   - Profile and optimize bottlenecks
   - Test FSDP scaling

10. **Documentation**
    - API documentation
    - Usage examples
    - Migration guide from Phase 1

---

## How to Use (Once Weight Conversion is Done)

### Using Native Model

```python
# Update model_index.json to use native model:
{
    "_class_name": "LongCatPipeline",
    "transformer": ["diffusers", "LongCatTransformer3DModel"]  # Native
}

# Or keep wrapper:
{
    "_class_name": "LongCatPipeline",
    "transformer": ["diffusers", "LongCatVideoTransformer3DModel"]  # Wrapper
}

# Then use normally:
from fastvideo import VideoGenerator

generator = VideoGenerator.from_pretrained(
    "weights/longcat-native",  # After weight conversion
    num_gpus=1,
)

video = generator.generate_video(
    prompt="A cat playing piano",
    num_inference_steps=50,
    guidance_scale=4.0,
)
```

---

## File Structure

```
fastvideo/
├── models/
│   └── dits/
│       ├── longcat.py                    # ✅ NEW: Native implementation
│       └── longcat_video_dit.py          # Phase 1 wrapper (keep for now)
│
├── configs/
│   ├── models/
│   │   └── dits/
│   │       ├── longcat.py                # ✅ NEW: Native config
│   │       └── __init__.py               # ✅ UPDATED: Export LongCatVideoConfig
│   └── pipelines/
│       └── longcat.py                    # ✅ UPDATED: Added native config import
│
├── models/
│   └── registry.py                       # ✅ UPDATED: Added native model
│
└── third_party/
    └── longcat_video/                    # Phase 1 wrapper (can remove after Phase 2)

scripts/
└── checkpoint_conversion/
    └── longcat_native_weights_converter.py  # ✅ NEW: Weight converter (TODO)
```

---

## Next Steps

### Immediate (This Week)

1. **Implement 3D RoPE**
   - Reference: `fastvideo/third_party/longcat_video/modules/rope_3d.py`
   - Create FastVideo-native version
   - Test numerical equivalence

2. **Complete Weight Converter**
   - Implement QKV/KV splitting
   - Implement parameter renaming
   - Test with actual weights

3. **Test Basic Forward Pass**
   - Create test with random inputs
   - Verify shapes and no errors
   - Compare with wrapper output

### Short Term (Next 2 Weeks)

4. **Add Variable-Length Text**
5. **Integrate CFG-Zero**
6. **Test Full Pipeline**

### Medium Term (Next Month)

7. **Add Advanced Features**
8. **Optimize Performance**
9. **Complete Documentation**

---

## Success Criteria

- [x] Native model loads successfully
- [x] All components use FastVideo layers
- [x] No dependencies on third_party code (except for reference)
- [ ] Forward pass completes without errors
- [ ] Weights convert successfully
- [ ] Output numerically equivalent to wrapper
- [ ] Performance >= wrapper
- [ ] All FastVideo features work (FSDP, offloading, etc.)

---

## References

- **Phase 1 Findings**: `LONGCAT_INTEGRATION_PHASE1_FINDINGS.md`
- **Phase 1 Summary**: `LONGCAT_INTEGRATION_PHASE1.md`
- **Phase 2 Plan**: `LONGCAT_INTEGRATION_PHASE2_PLAN.md`
- **Original LongCat**: `/mnt/fast-disks/hao_lab/shao/LongCat-Video/`
- **FastVideo Docs**: https://hao-ai-lab.github.io/FastVideo/

---

**Document Version**: 1.0  
**Last Updated**: November 6, 2025  
**Status**: Foundation Complete, Ready for Enhancement

