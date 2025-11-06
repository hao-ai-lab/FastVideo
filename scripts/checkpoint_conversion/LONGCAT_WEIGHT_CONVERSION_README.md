# LongCat Weight Conversion Guide

This guide explains how to convert LongCat weights from the Phase 1 wrapper format to the Phase 2 native FastVideo format.

## Overview

The conversion process transforms LongCat weights to use FastVideo's native layers:
- **QKV Splitting**: Splits fused QKV projections in self-attention into separate Q, K, V
- **KV Splitting**: Splits fused KV projections in cross-attention into separate K, V
- **Parameter Renaming**: Renames parameters to match native implementation naming conventions
- **Normalization**: Converts LayerNorm parameters to RMSNorm (drops bias)

## Quick Start

### Option 1: Full Conversion (Recommended)

Convert all components in one command:

```bash
python scripts/checkpoint_conversion/convert_longcat_to_native.py \
    --source weights/longcat-for-fastvideo \
    --output weights/longcat-native \
    --validate
```

This will:
1. Convert transformer weights with QKV/KV splitting
2. Copy VAE, text encoder, tokenizer, and scheduler
3. Update config files to point to native model
4. Validate the conversion

### Option 2: Transformer Only

Convert only the transformer weights:

```bash
python scripts/checkpoint_conversion/longcat_native_weights_converter.py \
    --source weights/longcat-for-fastvideo/transformer \
    --output weights/longcat-native/transformer \
    --validate
```

Then manually copy other components and update configs.

## Conversion Details

### Weight Transformations

#### 1. Embeddings

| Original (Wrapper) | Native FastVideo |
|-------------------|------------------|
| `x_embedder.*` | `patch_embed.*` |
| `t_embedder.mlp.0.*` | `time_embedder.linear_1.*` |
| `t_embedder.mlp.2.*` | `time_embedder.linear_2.*` |
| `y_embedder.y_proj.0.*` | `caption_embedder.linear_1.*` |
| `y_embedder.y_proj.2.*` | `caption_embedder.linear_2.*` |

#### 2. Self-Attention (QKV Split)

**Original**: Fused QKV projection
```
blocks.{i}.attn.qkv.weight  [3*dim, dim]
blocks.{i}.attn.qkv.bias    [3*dim]
```

**Native**: Separate Q, K, V projections
```
blocks.{i}.self_attn.to_q.weight  [dim, dim]
blocks.{i}.self_attn.to_q.bias    [dim]
blocks.{i}.self_attn.to_k.weight  [dim, dim]
blocks.{i}.self_attn.to_k.bias    [dim]
blocks.{i}.self_attn.to_v.weight  [dim, dim]
blocks.{i}.self_attn.to_v.bias    [dim]
```

#### 3. Cross-Attention (KV Split)

**Original**: Fused KV projection
```
blocks.{i}.cross_attn.kv_linear.weight  [2*dim, dim]
blocks.{i}.cross_attn.kv_linear.bias    [2*dim]
```

**Native**: Separate K, V projections
```
blocks.{i}.cross_attn.to_k.weight  [dim, dim]
blocks.{i}.cross_attn.to_k.bias    [dim]
blocks.{i}.cross_attn.to_v.weight  [dim, dim]
blocks.{i}.cross_attn.to_v.bias    [dim]
```

#### 4. Normalization (LayerNorm → RMSNorm)

**Dropped Parameters** (RMSNorm has no bias):
- `blocks.{i}.pre_crs_attn_norm.bias` → **dropped**
- `blocks.{i}.mod_norm_attn.*` → **dropped** (no params in original)
- `blocks.{i}.mod_norm_ffn.*` → **dropped** (no params in original)
- `final_layer.norm_final.*` → **dropped** (no params in original)

**Kept Parameters**:
- `blocks.{i}.pre_crs_attn_norm.weight` → `blocks.{i}.norm_cross.weight`

#### 5. Other Transformations

| Original | Native |
|----------|--------|
| `blocks.{i}.attn.proj.*` | `blocks.{i}.self_attn.to_out.*` |
| `blocks.{i}.attn.q_norm.*` | `blocks.{i}.self_attn.q_norm.*` |
| `blocks.{i}.attn.k_norm.*` | `blocks.{i}.self_attn.k_norm.*` |
| `blocks.{i}.cross_attn.q_linear.*` | `blocks.{i}.cross_attn.to_q.*` |
| `blocks.{i}.cross_attn.proj.*` | `blocks.{i}.cross_attn.to_out.*` |
| `blocks.{i}.adaLN_modulation.1.*` | `blocks.{i}.adaln_linear_1.*` |
| `blocks.{i}.ffn.w1.*` | `blocks.{i}.ffn.w1.*` (same) |
| `blocks.{i}.ffn.w2.*` | `blocks.{i}.ffn.w2.*` (same) |
| `blocks.{i}.ffn.w3.*` | `blocks.{i}.ffn.w3.*` (same) |
| `final_layer.adaLN_modulation.1.*` | `final_layer.adaln_linear.*` |
| `final_layer.linear.*` | `final_layer.proj.*` |

## Validation

The conversion script validates:
1. **Parameter count**: Accounts for dropped LayerNorm bias parameters
2. **QKV splits**: Verifies reconstructed QKV matches original (48 blocks)
3. **KV splits**: Verifies reconstructed KV matches original (48 blocks)

Expected parameter reduction:
- **Original**: ~8.0B parameters
- **Converted**: ~7.9B parameters (due to dropped LayerNorm bias)

## Testing the Converted Weights

### 1. Load the Model

```python
from fastvideo import VideoGenerator

# Load native model
generator = VideoGenerator.from_pretrained(
    "weights/longcat-native",
    num_gpus=1,
)
```

### 2. Generate a Test Video

```python
video = generator.generate_video(
    prompt="A cat playing piano",
    num_inference_steps=50,
    guidance_scale=4.0,
    height=480,
    width=832,
    num_frames=65,
    seed=42,
)

# Save video
from fastvideo.utils.video_io import save_video
save_video(video, "test_output.mp4", fps=16)
```

### 3. Compare with Wrapper

To verify numerical equivalence, compare outputs from wrapper vs native:

```python
# Test with wrapper
generator_wrapper = VideoGenerator.from_pretrained(
    "weights/longcat-for-fastvideo",
    num_gpus=1,
)
video_wrapper = generator_wrapper.generate_video(
    prompt="A cat playing piano",
    num_inference_steps=2,  # Fast test
    seed=42,
)

# Test with native
generator_native = VideoGenerator.from_pretrained(
    "weights/longcat-native",
    num_gpus=1,
)
video_native = generator_native.generate_video(
    prompt="A cat playing piano",
    num_inference_steps=2,
    seed=42,
)

# Compare (should be very close, within FP precision)
import torch
diff = torch.tensor(video_wrapper).float() - torch.tensor(video_native).float()
print(f"Max difference: {diff.abs().max().item()}")
print(f"Mean difference: {diff.abs().mean().item()}")
```

## Troubleshooting

### Error: Parameter count mismatch

**Cause**: Missing or incorrectly converted parameters

**Solution**: 
1. Check that source weights are from Phase 1 wrapper (longcat-for-fastvideo)
2. Ensure all 48 transformer blocks are present
3. Run with `--validate` flag to see which splits failed

### Error: QKV/KV split mismatch

**Cause**: Tensor shapes don't match expected dimensions

**Solution**:
1. Verify source weights are correct format
2. Check that `hidden_size=4096` and `num_heads=32` in config
3. Ensure weights weren't already split

### Error: Model loading fails

**Cause**: Config files not updated or missing

**Solution**:
1. Check `transformer/config.json` has `_class_name: "LongCatTransformer3DModel"`
2. Check `model_index.json` has `transformer: ["diffusers", "LongCatTransformer3DModel"]`
3. Ensure all components (VAE, text_encoder, etc.) are copied

### Different output than wrapper

**Cause**: Numerical precision differences (acceptable) or conversion error (not acceptable)

**Solution**:
1. Small differences (< 1e-4) are expected due to FP precision
2. Large differences indicate conversion error - rerun conversion with `--validate`
3. Check that RMSNorm vs LayerNorm difference is acceptable (should be minimal)

## Implementation Reference

### Original Wrapper Structure
- **File**: `fastvideo/third_party/longcat_video/modules/longcat_video_dit.py`
- **Attention**: `fastvideo/third_party/longcat_video/modules/attention.py`
- **Blocks**: `fastvideo/third_party/longcat_video/modules/blocks.py`

### Native Implementation
- **File**: `fastvideo/models/dits/longcat.py`
- **Uses**: FastVideo's `ReplicatedLinear`, `RMSNorm`, `DistributedAttention`

### Conversion Scripts
- **Full conversion**: `scripts/checkpoint_conversion/convert_longcat_to_native.py`
- **Transformer only**: `scripts/checkpoint_conversion/longcat_native_weights_converter.py`

## Architecture Differences

### Phase 1 (Wrapper)
- Uses `nn.Linear` (no tensor parallelism)
- Fused QKV and KV projections
- LayerNorm with bias
- Custom attention backends

### Phase 2 (Native)
- Uses `ReplicatedLinear` (tensor parallelism ready)
- Separate Q, K, V projections
- RMSNorm (no bias)
- FastVideo's `DistributedAttention`

## Expected Performance

After conversion, the native model should:
- ✅ **Maintain quality**: Output visually identical to wrapper
- ✅ **Improve speed**: 10-20% faster inference (with compilation)
- ✅ **Support FSDP**: Better multi-GPU scaling
- ✅ **Enable compilation**: Compatible with `torch.compile()`
- ✅ **Reduce memory**: Slightly lower peak memory usage

## Additional Resources

- **Phase 2 Plan**: `LONGCAT_INTEGRATION_PHASE2_PLAN.md`
- **Native Implementation**: `LONGCAT_NATIVE_IMPLEMENTATION.md`
- **FastVideo Docs**: https://hao-ai-lab.github.io/FastVideo/

---

**Last Updated**: November 6, 2025  
**Status**: Production Ready


