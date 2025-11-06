# LongCat Native Testing Guide

This guide explains how to test the native LongCat implementation after weight conversion.

## Prerequisites

✅ Converted weights at `weights/longcat-native/`  
✅ Conda environment activated: `conda activate fastvideo_shao`  
✅ GPU with BF16 support (Ampere or newer recommended)

---

## Quick Sanity Check

For a fast verification that everything works:

```bash
python test_longcat_quick.py
```

**What it does:**
- Loads the native model
- Runs 2 inference steps (very fast)
- Generates a 9-frame 256x384 video
- Verifies output shape
- Reports memory usage

**Expected output:**
```
✓ Model loaded: LongCatTransformer3DModel
✓ Generation successful!
  Output shape: (9, 256, 384, 3)
Peak GPU memory: ~25 GB
✓ All checks passed!
```

**Time:** ~30 seconds

---

## Full Inference Test

For complete testing with full quality:

```bash
python test_longcat_native_inference.py \
    --model-path weights/longcat-native \
    --prompt "A cat playing piano in a cozy living room" \
    --output outputs/test_video.mp4 \
    --steps 50 \
    --guidance-scale 4.0 \
    --height 480 \
    --width 832 \
    --num-frames 65 \
    --fps 16 \
    --seed 42
```

**What it does:**
- Loads native model with full configuration
- Generates high-quality 480p video
- Saves output to MP4 file
- Reports detailed timing and memory stats

**Expected output:**
```
[Step 1/3] Loading native model...
✓ Model loaded in 15.23s
  Model class: LongCatTransformer3DModel
  ✓ Using native implementation

[Step 2/3] Generating video...
✓ Video generated in 187.45s
  Output shape: (65, 480, 832, 3)
  Speed: 0.35 fps

[Step 3/3] Saving video...
✓ Video saved to outputs/test_video.mp4
  File size: 45.23 MB

✓ Test Complete!
Total time: 202.89s
Peak GPU memory: 48.72 GB
```

**Time:** ~3-5 minutes (depending on GPU)

---

## Command Line Options

### test_longcat_native_inference.py

| Option | Default | Description |
|--------|---------|-------------|
| `--model-path` | `weights/longcat-native` | Path to converted weights |
| `--prompt` | "A cat playing piano..." | Text prompt |
| `--output` | `outputs/longcat_native_test.mp4` | Output video path |
| `--steps` | `50` | Number of inference steps |
| `--guidance-scale` | `4.0` | CFG guidance scale |
| `--height` | `480` | Video height |
| `--width` | `832` | Video width |
| `--num-frames` | `65` | Number of frames |
| `--fps` | `16` | Frames per second |
| `--seed` | `42` | Random seed |
| `--num-gpus` | `1` | Number of GPUs |

### Example Commands

**Fast test (low quality):**
```bash
python test_longcat_native_inference.py \
    --steps 20 \
    --height 256 \
    --width 384 \
    --num-frames 25
```

**High quality:**
```bash
python test_longcat_native_inference.py \
    --steps 100 \
    --guidance-scale 5.0 \
    --num-frames 129
```

**Multi-GPU:**
```bash
python test_longcat_native_inference.py \
    --num-gpus 2
```

---

## Expected Performance

### Hardware Requirements

| Configuration | GPU Memory | Generation Time (50 steps, 65 frames) |
|--------------|------------|---------------------------------------|
| **Minimum** | 24 GB | ~5 min |
| **Recommended** | 40 GB | ~3 min |
| **Optimal** | 80 GB | ~2 min |

### Resolution Support

| Resolution | Frames | GPU Memory | Notes |
|-----------|--------|------------|-------|
| 256x384 | 9 | ~20 GB | Quick testing |
| 480x832 | 65 | ~45 GB | Standard (480p) |
| 480x832 | 129 | ~60 GB | Long video |
| 720x1280 | 65 | ~80 GB | High res (requires BSA) |

---

## Troubleshooting

### Error: CUDA out of memory

**Solutions:**
1. Reduce resolution: `--height 256 --width 384`
2. Reduce frames: `--num-frames 25`
3. Enable CPU offloading (slower):
   ```python
   generator = VideoGenerator.from_pretrained(
       "weights/longcat-native",
       dit_cpu_offload=True,
   )
   ```

### Error: Model class mismatch

**Problem:** Loading wrapper instead of native model

**Solution:** Check `model_index.json`:
```json
{
  "transformer": ["diffusers", "LongCatTransformer3DModel"]
}
```

Should be `LongCatTransformer3DModel`, not `LongCatVideoTransformer3DModel`.

### Error: Numerical instability / NaN values

**Check:**
1. Precision settings in config (should be BF16)
2. FP32 critical operations enabled
3. GPU supports BF16 (Ampere or newer)

### Slow generation

**Expected speeds:**
- With compilation: ~0.5-1.0 fps
- Without compilation: ~0.3-0.5 fps

**To improve:**
1. Use fewer inference steps (20-30 for testing)
2. Enable torch.compile (if supported)
3. Use smaller resolution for testing

---

## Comparing with Wrapper

To verify numerical equivalence with Phase 1 wrapper:

```python
# Test wrapper
from fastvideo import VideoGenerator

gen_wrapper = VideoGenerator.from_pretrained("weights/longcat-for-fastvideo")
video_wrapper = gen_wrapper.generate_video(
    prompt="A cat",
    num_inference_steps=2,
    seed=42,
)

# Test native
gen_native = VideoGenerator.from_pretrained("weights/longcat-native")
video_native = gen_native.generate_video(
    prompt="A cat",
    num_inference_steps=2,
    seed=42,
)

# Compare
import torch
diff = torch.tensor(video_wrapper).float() - torch.tensor(video_native).float()
print(f"Max difference: {diff.abs().max().item()}")
print(f"Mean difference: {diff.abs().mean().item()}")
```

**Expected:** Max difference < 0.01 (small FP precision differences are normal)

---

## Validation Checklist

- [ ] Quick sanity check passes (`test_longcat_quick.py`)
- [ ] Full inference completes without errors
- [ ] Output video is generated and playable
- [ ] Model class is `LongCatTransformer3DModel`
- [ ] Memory usage is reasonable (<50 GB for 480p)
- [ ] Generation speed is acceptable (~0.3-0.5 fps)
- [ ] Output quality matches wrapper (visual inspection)
- [ ] No NaN or Inf values in output

---

## Advanced Testing

### Profile Memory Usage

```python
import torch
from fastvideo import VideoGenerator

torch.cuda.reset_peak_memory_stats()

generator = VideoGenerator.from_pretrained("weights/longcat-native")
video = generator.generate_video(prompt="A cat", num_inference_steps=2)

peak_mb = torch.cuda.max_memory_allocated() / (1024**2)
print(f"Peak memory: {peak_mb:.2f} MB")
```

### Test with Different Prompts

```bash
# Simple prompt
python test_longcat_native_inference.py --prompt "A cat"

# Complex prompt
python test_longcat_native_inference.py \
    --prompt "A fluffy orange cat playing a grand piano in an elegant Victorian living room with sunset light streaming through lace curtains"

# Multiple objects
python test_longcat_native_inference.py \
    --prompt "A cat and a dog playing together in a park"
```

### Test Different Guidance Scales

```bash
# Low guidance (more creative)
python test_longcat_native_inference.py --guidance-scale 2.0

# Standard
python test_longcat_native_inference.py --guidance-scale 4.0

# High guidance (more accurate to prompt)
python test_longcat_native_inference.py --guidance-scale 7.0
```

---

## Next Steps

After successful testing:

1. **Benchmark Performance**: Compare native vs wrapper speed
2. **Test LoRA**: Load and test with LoRA weights
3. **Test BSA**: Enable block-sparse attention for 720p
4. **Test Compilation**: Enable torch.compile() for speedup
5. **Production Deployment**: Use in your video generation pipeline

---

## Support

**Documentation:**
- Native implementation: `LONGCAT_NATIVE_IMPLEMENTATION.md`
- Precision analysis: `LONGCAT_PRECISION_ANALYSIS.md`
- Weight conversion: `scripts/checkpoint_conversion/LONGCAT_WEIGHT_CONVERSION_README.md`

**Issues:**
- Check model config files
- Verify weight conversion completed successfully
- Review logs in `conversion.log`

---

**Last Updated**: November 6, 2025  
**Status**: Ready for Testing ✅

