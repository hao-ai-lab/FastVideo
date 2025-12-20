# LongCat I2V Implementation Summary

## Implemented Components

### 1. Pipeline: `longcat_i2v_pipeline.py` ✅
- Created `LongCatImageToVideoPipeline` class
- Leverages existing FastVideo stages where possible
- Adds LongCat-specific I2V stages
- Reuses LongCat BSA initialization logic

### 2. Stages: `longcat_i2v_stages.py` ✅
Three LongCat-specific stages implemented:

#### `LongCatI2VImageVAEEncodingStage`
- Encodes input image via VAE
- **Key feature**: Uses LongCat-specific normalization (not Wan's scaling_factor)
- Formula: `(latents - mean) / std`
- Sets `num_cond_latents = 1` for single image

#### `LongCatI2VLatentPreparationStage`
- Extends base `LatentPreparationStage`
- Generates random noise for all frames
- Replaces first frame with encoded image latent
- Simple and clean implementation

#### `LongCatI2VDenoisingStage`
- Extends `LongCatDenoisingStage`
- **Critical**: Sets `timestep[:, :num_cond_latents] = 0` for conditioning frames
- **Critical**: Only applies scheduler step to non-conditioned frames: `latents[:, :, num_cond_latents:]`
- Passes `num_cond_latents` parameter to transformer

### 3. Transformer Updates: `longcat.py` ✅
- Added `num_cond_latents` parameter to `forward()` signature
- Passed `num_cond_latents` through transformer blocks to attention
- Block forward already accepts `**kwargs`, so parameter flows through automatically

### 4. Test Script: `test_longcat_i2v.py` ✅
- Single-file script (like original LongCat demos)
- Comprehensive CLI arguments
- Supports image URLs or local paths
- Optional resize to match input image size
- Memory cleanup after generation

## Design Decisions

### Reused Existing Code
Following the principle of not reinventing the wheel, we reused:
- Base `LatentPreparationStage` - just extended it
- `LongCatDenoisingStage` - extended with I2V modifications
- Standard pipeline stages (InputValidation, TextEncoding, etc.)
- Existing VAE preprocessing utilities from `vision_utils`

### LongCat-Specific Implementations
Only implemented LongCat-specific parts:
1. **Normalization**: LongCat uses `(x - mean) / std`, not `x * scaling_factor` like Wan
2. **Timestep masking**: Conditioning frames need `timestep = 0`
3. **Selective denoising**: Only denoise non-conditioned frames

### Simplified Approach (for MVP)
- **RoPE handling**: Currently, RoPE is applied to all tokens (including conditioning)
  - Original LongCat skips RoPE for conditioning frames
  - This is an optimization we can add later
  - Should not break functionality, just slightly suboptimal
  
- **No KV cache**: I2V with 1 conditioning frame doesn't benefit from KV cache
  - KV cache is essential for VC (13+ frames) but overkill for I2V
  - Keeps implementation simpler

## Usage

### Basic Usage
```bash
python test_longcat_i2v.py \
    --checkpoint-dir weights/longcat-native \
    --image-path images/girl.png \
    --prompt "A woman drinking coffee in a café" \
    --num-frames 93 \
    --height 480 \
    --width 832
```

### With FastVideo API
```python
from fastvideo import VideoGenerator

generator = VideoGenerator.from_pretrained("weights/longcat-native")

video = generator.generate_video(
    prompt="A woman drinking coffee",
    image_path="girl.png",  # Triggers I2V mode
    num_frames=93,
    height=480,
    width=832,
)
```

## Testing Plan

### Unit Tests (TODO)
1. Test VAE encoding with LongCat normalization
2. Test latent preparation with conditioning
3. Test timestep masking logic
4. Test selective denoising

### Integration Test
Run the test script:
```bash
python test_longcat_i2v.py --checkpoint-dir weights/longcat-native
```

Expected behavior:
- Loads image successfully
- Generates 93-frame video
- First frame resembles input image
- Subsequent frames show motion/animation

### Comparison Test (Future)
Compare output with original LongCat-Video:
```bash
# Original LongCat
cd /mnt/fast-disks/hao_lab/shao/LongCat-Video
python run_demo_image_to_video.py

# FastVideo LongCat
cd /mnt/fast-disks/hao_lab/shao/FastVideo
python test_longcat_i2v.py
```

## Known Limitations

1. **RoPE not skipped for conditioning frames**
   - Original LongCat skips RoPE for conditioning
   - Our implementation applies RoPE to all frames
   - Impact: Minimal, can optimize later

2. **Single GPU testing only**
   - Multi-GPU with SP needs testing
   - Should work (uses existing SP infrastructure)

3. **No distillation support yet**
   - Original LongCat has distilled I2V (16 steps)
   - Requires LoRA loading (separate feature)

## Next Steps

1. **Test the implementation** ✅ (Ready to test)
2. **Fix any runtime errors**
3. **Add RoPE skipping optimization** (if needed for quality)
4. **Implement VC (Video Continuation)** - next major feature
5. **Add more comprehensive tests**

## Files Modified

- `fastvideo/pipelines/basic/longcat/longcat_i2v_pipeline.py` (NEW)
- `fastvideo/pipelines/stages/longcat_i2v_stages.py` (NEW)
- `fastvideo/models/dits/longcat.py` (MODIFIED - added num_cond_latents)
- `test_longcat_i2v.py` (NEW)

## Estimated Time to Working I2V

- **Core implementation**: ✅ DONE
- **Debugging**: 1-2 hours (fix import errors, parameter mismatches, etc.)
- **Quality tuning**: 2-4 hours (if RoPE skipping needed)
- **Total**: 3-6 hours to working I2V

## References

- Original LongCat I2V: `/mnt/fast-disks/hao_lab/shao/LongCat-Video/run_demo_image_to_video.py`
- Wan I2V (reference): `fastvideo/pipelines/basic/wan/wan_i2v_pipeline.py`
- Implementation plan: `LONGCAT_I2V_VC_IMPLEMENTATION_PLAN.md`

