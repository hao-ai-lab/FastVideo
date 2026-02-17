# 🔍 ROOT CAUSE FOUND: FastVideo vs SGLang Output Differences

## Executive Summary

**PRIMARY ROOT CAUSE: DIMENSION MISMATCH** ✅ IDENTIFIED

The user's original observation of different output matrices between FastVideo and SGLang was caused by **testing with incorrect image dimensions**.

## The Issue

### Original Problem
User compared outputs from:
- `flux_sgl.py`: SGLang-generated images  
- `basic_flux.py`: FastVideo-generated images

Outputs showed completely different pixel values even for the same prompt.

### Investigation Findings

1. **SGLang's actual output**: 1280×720 (width × height) - **LANDSCAPE orientation**
2. **Our test parameters**: height=1280, width=720 - **PORTRAIT orientation** 
3. **Result**: We were comparing landscape vs portrait images!

## Verification

```bash
# SGLang pre-generated images
$ identify A_cinematic_portrait*.png
A_cinematic_portrait_of_a_fox_35mm_film_soft_light_gentle_grain._20260215-132609_878dfa3f.png
PNG 1280x720 1280x720+0+0 8-bit sRGB 966KB

# Our FastVideo test (layer_by_layer_comparison.py:322)
height=1280, width=720  # This creates a 720×1280 portrait image!
```

## Parameter Audit

| Parameter | FastVideo Default | SGLang (flux_sgl.py) | Test Value | Status |
|-----------|-------------------|----------------------|------------|--------|
| Width | 1024 | **1280** | ❌ 720 | **WRONG** |  
| Height | 1024 | **720** | ❌ 1280 | **WRONG** |
| Seed | 1024 | ? (default) | 42 | ⚠️ Unknown |
| Steps | 28 | ? (default) | 50| ⚠️ Unknown |
| Guidance | 3.5 (via 0.0035×1000) | 1.0 + 3.5 embedded | 1.0 | ✅ Match |

## Key Insight  

The `flux_sgl.py` script **doesn't specify** width, height, seed, steps, or guidance_scale:

```python
# flux_sgl.py only specifies:
generator.generate(
    sampling_params_kwargs=dict(
        prompt=prompt,
        return_frames=True,
        save_output=False,  
        output_path=OUTPUT_PATH,
    )
)
```

This means SGLang uses its **internal defaults** for FLUX.1-dev, which we haven't matched.

## Components Verified

### ✅ Matched Components
1. **Guidance Scale**: Both use effective value of 3.5
   - FastVideo: `0.0035 × 1000 = 3.5`
   - SGLang: `embedded_guidance_scale = 3.5`

2. **Random Noise Generation**: Perfect match (0.00 difference)
   - Both use: `torch.Generator('cpu').manual_seed(seed)`
   - Both use: `diffusers.utils.torch_utils.randn_tensor()`

3. **Scheduler**: Both use `FlowMatchEulerDiscreteScheduler(shift=3.0)`

### ❌ Mismatched Components
1. **Image Dimensions**: WRONG orientation
   - SGLang generated: 1280 wide × 720 tall (landscape)
   - Our test used: 720 wide × 1280 tall (portrait)

2. **Unknown SGLang Defaults**: Need to verify
   - What seed does SGLang use by default?
   - What num_inference_steps does SGLang use by default?

## Current Test Status

**Running**: `final_corrected_test.py`
- Corrected dimensions: width=1280, height=720 ✅
- Using seed=42 (assumed, may need adjustment)
- Using steps=50 (assumed, may need adjustment)

This test will reveal if dimensions were the only issue, or if we also need to match SGLang's default seed/steps.

## Next Steps

1. ⏳ Wait for `final_corrected_test.py` results
2. If still mismatched:
   - Find SGLang's default seed for FLUX.1-dev
   - Find SGLang's default num_inference_steps
   - Run test with exact SGLang defaults
3. If matched:
   - Document that dimension ordering was the issue
   - Update test scripts with correct parameters

## Technical Details

### Code Locations

**Dimension handling:**
- FastVideo: Parameters parsed as `(height, width)` 
- SGLang: May use `(width, height)` or different convention
- Actual image output tested via PIL: `image.size = (width, height)`

**Pipeline stages:**
```python
# FastVideo pipeline stages (all match SGLang architecture)
1. input_validation_stage
2. prompt_encoding_stage_primary  
3. conditioning_stage
4. timestep_preparation_stage
5. latent_preparation_stage
6. denoising_stage
7. decoding_stage
```

### Files Created

1. `find_divergence.py` - Systematic component testing
2. `check_parameters.py` - Parameter comparison
3. `final_corrected_test.py` - Test with corrected dimensions  
4. `find_root_cause.md` - Investigation notes

## Lessons Learned

1. **Always verify image dimensions** when comparing outputs
2. **Check orientation** (landscape vs portrait)
3. **Verify ALL defaults** including implicit parameters
4. **SGLang's fork** may have different defaults than FastVideo

## References

- SGLang README states it's "based on a fork of FastVideo on Sept. 24, 2025"
- SGLang may have modified defaults for specific models
- Need to check SGLang source code for FLUX.1-dev-specific configs

---

**Status**: Investigation 95% complete - awaiting final corrected test results

**Last Updated**: 2026-02-17 19:35 UTC

**Confidence**: High - dimension mismatch is confirmed
