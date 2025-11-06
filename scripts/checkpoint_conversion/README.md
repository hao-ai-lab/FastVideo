# Weight Conversion Scripts

This directory contains scripts for converting model weights to FastVideo format.

## LongCat Conversion

### Phase 1: Original LongCat → Wrapper Format
Use this to prepare original LongCat weights for FastVideo:

```bash
python scripts/checkpoint_conversion/longcat_to_fastvideo.py \
    --source /path/to/LongCat-Video/weights/LongCat-Video \
    --output weights/longcat-for-fastvideo
```

This creates a FastVideo-compatible structure using the wrapper model (`LongCatVideoTransformer3DModel`).

### Phase 2: Wrapper → Native FastVideo Format
Convert from wrapper to native implementation for better performance:

#### Option 1: Full Conversion (Recommended)
```bash
python scripts/checkpoint_conversion/convert_longcat_to_native.py \
    --source weights/longcat-for-fastvideo \
    --output weights/longcat-native \
    --validate
```

This converts all components and updates configs.

#### Option 2: Transformer Only
```bash
python scripts/checkpoint_conversion/longcat_native_weights_converter.py \
    --source weights/longcat-for-fastvideo/transformer \
    --output weights/longcat-native/transformer \
    --validate
```

Then manually copy other components.

### Validation
Check converted weights:

```bash
python scripts/checkpoint_conversion/validate_longcat_weights.py \
    --model-path weights/longcat-native
```

### Detailed Documentation
See [LONGCAT_WEIGHT_CONVERSION_README.md](LONGCAT_WEIGHT_CONVERSION_README.md) for:
- Detailed conversion process
- Parameter transformations
- Troubleshooting guide
- Testing procedures

## WanVideo Conversion

Convert WanVideo weights:

```bash
python scripts/checkpoint_conversion/wan_to_diffusers.py
```

(Edit the script to set source and output paths)

## File Summary

| File | Purpose | When to Use |
|------|---------|-------------|
| `longcat_to_fastvideo.py` | Original LongCat → Wrapper | First-time setup |
| `convert_longcat_to_native.py` | Wrapper → Native (full) | Complete conversion |
| `longcat_native_weights_converter.py` | Wrapper → Native (transformer only) | Manual conversion |
| `validate_longcat_weights.py` | Validate converted weights | After conversion |
| `wan_to_diffusers.py` | WanVideo conversion | WanVideo models |
| `LONGCAT_WEIGHT_CONVERSION_README.md` | Detailed LongCat guide | Reference |


