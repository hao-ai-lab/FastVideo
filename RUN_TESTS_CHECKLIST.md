# GameCraft Testing Checklist

## ✅ Before Running Tests - Setup Required

### Step 1: Basic Tests (No Setup Needed)
These tests work WITHOUT weights or official repo - run these FIRST:

```bash
# Test 1: Shape validation (architecture correctness)
pytest fastvideo/tests/transformers/test_hunyuangamecraft.py::test_gamecraft_output_shape -v -s

# Test 2: Camera conditioning (feature validation)
pytest fastvideo/tests/transformers/test_hunyuangamecraft.py::test_gamecraft_camera_conditioning -v -s
```

**If these fail:** Your model architecture has issues, fix before proceeding.

---

### Step 2: Setup for Numerical Tests

#### Required Downloads:

**A) Official GameCraft Repo** (~500MB)
```bash
# Clone to correct location
cd fastvideo/models
git clone https://github.com/Tencent-Hunyuan/Hunyuan-GameCraft-1.0 Hunyuan-GameCraft-1.0-main
cd ../..

# Verify
ls fastvideo/models/Hunyuan-GameCraft-1.0-main/
```

**B) Model Weights** (~12GB)
```bash
# Install HuggingFace CLI
pip install -U "huggingface_hub[cli]"

# Create directory
mkdir -p weights/gamecraft_models

# Download main checkpoint
cd weights/gamecraft_models
huggingface-cli download tencent/Hunyuan-GameCraft-1.0 \
  --include "weights/mp_rank_00_model_states.pt" \
  --local-dir .
cd ../..

# Verify (should be ~12GB)
ls -lh weights/gamecraft_models/mp_rank_00_model_states.pt
```

**C) Set Environment Variable**
```bash
export GAMECRAFT_MODEL_PATH="weights/gamecraft_models"

# Add to ~/.bashrc for persistence
echo 'export GAMECRAFT_MODEL_PATH="$HOME/path/to/weights/gamecraft_models"' >> ~/.bashrc
```

---

### Step 3: Numerical Alignment Test (THE CRITICAL ONE)

**This is what your PhD advisor means by "align transformers numerical values"**

```bash
# Set env var
export GAMECRAFT_MODEL_PATH="weights/gamecraft_models"

# Run the numerical comparison
pytest fastvideo/tests/transformers/test_hunyuangamecraft.py::test_gamecraft_vs_original -v -s
```

**What to expect:**
```
Loading ORIGINAL GameCraft implementation...
✓ Original model loaded
Loading FASTVIDEO GameCraft implementation...
✓ FastVideo model loaded
Creating test inputs...
Running ORIGINAL forward pass...
✓ Original output shape: torch.Size([1, 16, 9, 88, 152])
Running FASTVIDEO forward pass...
✓ FastVideo output shape: torch.Size([1, 16, 9, 88, 152])
Comparing outputs...
============================================================
NUMERICAL COMPARISON RESULTS
============================================================
Max absolute diff:  1.23e-06  ← MUST BE < 1e-5 ✓
Mean absolute diff: 3.45e-07
============================================================
✓✓✓ PASSED: Max diff 1.23e-06 < 1e-5
✓ FastVideo GameCraft matches original implementation!
```

**Success = Max diff < 1e-5** (0.00001)

---

### Step 4: Generate Reference Latent

```bash
pytest fastvideo/tests/transformers/test_hunyuangamecraft.py::test_gamecraft_transformer_distributed -v -s
```

**First run will output:**
```
No reference latent set. Save this value for future tests:
REFERENCE_LATENT = 123.456789012345
```

**Action:** Copy this value to line 62 of `test_hunyuangamecraft.py`:
```python
REFERENCE_LATENT = 123.456789012345  # Replace None with this value
```

**Run again to verify:**
```bash
pytest fastvideo/tests/transformers/test_hunyuangamecraft.py::test_gamecraft_transformer_distributed -v -s
```

Should now pass with: `✓ Numerical diff test PASSED`

---

## 🚀 Complete Test Sequence

### Run All Tests in Order:

```bash
#!/bin/bash
# Save as run_all_tests.sh

echo "=========================================="
echo "GameCraft Validation Test Suite"
echo "=========================================="
echo ""

# Test 1: Architecture
echo "Test 1/4: Output Shape Validation..."
pytest fastvideo/tests/transformers/test_hunyuangamecraft.py::test_gamecraft_output_shape -v
if [ $? -ne 0 ]; then echo "❌ Shape test failed!"; exit 1; fi
echo "✓ Shape test passed"
echo ""

# Test 2: Camera
echo "Test 2/4: Camera Conditioning..."
pytest fastvideo/tests/transformers/test_hunyuangamecraft.py::test_gamecraft_camera_conditioning -v
if [ $? -ne 0 ]; then echo "❌ Camera test failed!"; exit 1; fi
echo "✓ Camera test passed"
echo ""

# Test 3: Numerical alignment (THE KEY TEST)
echo "Test 3/4: Numerical Alignment (Critical!)..."
echo "Comparing FastVideo vs Original GameCraft..."
pytest fastvideo/tests/transformers/test_hunyuangamecraft.py::test_gamecraft_vs_original -v -s
if [ $? -ne 0 ]; then echo "❌ Numerical alignment failed!"; exit 1; fi
echo "✓ Numerical alignment test passed"
echo ""

# Test 4: Reference latent
echo "Test 4/4: Reference Latent Validation..."
pytest fastvideo/tests/transformers/test_hunyuangamecraft.py::test_gamecraft_transformer_distributed -v
if [ $? -ne 0 ]; then echo "⚠ Set REFERENCE_LATENT in test file"; fi
echo ""

echo "=========================================="
echo "✅ All tests completed!"
echo "=========================================="
```

---

## ⚠️ Common Issues

### Issue 1: "Official repo not found"
```bash
Error: Could not import original implementation

Solution:
cd fastvideo/models
git clone https://github.com/Tencent-Hunyuan/Hunyuan-GameCraft-1.0 Hunyuan-GameCraft-1.0-main
```

### Issue 2: "Weights not found"
```bash
Error: [Errno 2] No such file or directory: 'weights/gamecraft_models/...'

Solution:
export GAMECRAFT_MODEL_PATH="weights/gamecraft_models"
# Or check path exists:
ls -la weights/gamecraft_models/mp_rank_00_model_states.pt
```

### Issue 3: "CUDA out of memory"
```bash
Error: RuntimeError: CUDA out of memory

Solution:
# Use smaller test size or CPU offloading
export CPU_OFFLOAD=1
# Or modify test to use smaller batch
```

### Issue 4: "Max diff too large" (> 1e-5)
```bash
Max diff: 2.45e-04 exceeds tolerance 1e-5

Possible causes:
1. Weights not loaded correctly
2. Text embedding format mismatch
3. RoPE embedding differences
4. Camera preprocessing differences

Debug:
# Check layer-by-layer outputs
# Verify checkpoint loads without errors
# Compare intermediate activations
```

---

## 📊 Success Criteria Summary

| Test | Purpose | Success |
|------|---------|---------|
| `test_gamecraft_output_shape` | Architecture correct | All shapes match |
| `test_gamecraft_camera_conditioning` | Camera works | Outputs differ with different camera |
| `test_gamecraft_vs_original` | **Numerical alignment** | **Max diff < 1e-5** ⭐ |
| `test_gamecraft_transformer_distributed` | Checkpoint consistency | REFERENCE_LATENT matches |

---

## 🎯 What Your PhD Advisor Needs

**"Align transformers numerical values"** = Test 3 must pass:

```
Max absolute difference: < 1e-5 (0.00001)
```

**This proves:**
- ✅ Your port is mathematically correct
- ✅ Weights load properly
- ✅ All layers work identically
- ✅ Ready for production use

---

## 📝 Quick Start

```bash
# 1. Run basic tests first (no setup)
pytest fastvideo/tests/transformers/test_hunyuangamecraft.py::test_gamecraft_output_shape -v

# 2. If pass, download everything (weights + repo)
# ... (see Step 2 above)

# 3. Run numerical alignment
export GAMECRAFT_MODEL_PATH="weights/gamecraft_models"
pytest fastvideo/tests/transformers/test_hunyuangamecraft.py::test_gamecraft_vs_original -v -s

# 4. Check result
# ✅ Max diff < 1e-5 = SUCCESS
# ❌ Max diff > 1e-5 = Debug needed
```

---

## 🔍 Verify Before Testing

Run this quick check:

```bash
# Check files
ls fastvideo/models/dits/hunyuangamecraft.py
ls fastvideo/tests/transformers/test_hunyuangamecraft.py

# Check imports
python -c "from fastvideo.models.dits.hunyuangamecraft import HunyuanGameCraftTransformer3DModel; print('✓ Import OK')"

# Check GPU
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPUs: {torch.cuda.device_count()}')"

# Check official repo (for numerical test)
ls fastvideo/models/Hunyuan-GameCraft-1.0-main/ || echo "⚠ Official repo not cloned"

# Check weights (for numerical test)
ls weights/gamecraft_models/mp_rank_00_model_states.pt || echo "⚠ Weights not downloaded"
```

All checks pass? → Ready to test! 🚀
