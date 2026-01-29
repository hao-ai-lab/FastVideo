# Transfer GameCraft Port to GPU Cluster

## 🎯 Quick Transfer Commands

### Method 1: Git Push/Pull (Recommended)
```bash
# On your local machine
git add -A
git commit -m "Add HunyuanGameCraft port with tests"
git push origin hunyuan-gamecraft-porting

# On GPU cluster
git clone <your-fork-url>
cd FastVideo
git checkout hunyuan-gamecraft-porting
```

### Method 2: rsync (For uncommitted changes)
```bash
# From your local machine, sync to GPU cluster
rsync -avz --progress \
  --exclude='*.pyc' \
  --exclude='__pycache__/' \
  --exclude='.git/' \
  --exclude='*.pt' \
  --exclude='*.pth' \
  --exclude='*.safetensors' \
  --exclude='weights/' \
  --exclude='data/' \
  --exclude='outputs/' \
  --exclude='results/' \
  --exclude='fastvideo/models/Hunyuan-GameCraft-1.0-main/' \
  --exclude='.venv/' \
  --exclude='venv/' \
  --exclude='.DS_Store' \
  /Users/mihirjagtap/Documents/GitHub/FastVideo/ \
  username@gpu-cluster:/path/to/FastVideo/
```

---

## 📦 Essential Files Checklist

### ✅ MUST TRANSFER - GameCraft Implementation

#### Core Model Files
- [ ] `fastvideo/models/dits/hunyuangamecraft.py` (1057 lines - main model)
- [ ] `fastvideo/configs/models/dits/hunyuangamecraft.py` (config)
- [ ] `fastvideo/models/registry.py` (modified - has GameCraft registration)
- [ ] `fastvideo/configs/models/dits/__init__.py` (modified - exports config)

#### Test Files
- [ ] `fastvideo/tests/transformers/test_hunyuangamecraft.py` (565 lines)

#### Documentation
- [ ] `GAMECRAFT_TODO.md`
- [ ] `GAMECRAFT_VALIDATION_PLAN.md`
- [ ] `README_GAMECRAFT.md`
- [ ] `PR_CHECKLIST.md`
- [ ] `TESTING_WITH_OFFICIAL_REPO.md`
- [ ] `validate_gamecraft.sh`

#### Project Infrastructure (already in repo)
- [ ] `pyproject.toml`
- [ ] `requirements-mkdocs.txt` (if testing docs)
- [ ] `.gitignore`
- [ ] `.pre-commit-config.yaml`

---

## ❌ DO NOT TRANSFER (Excluded by .gitignore)

### Large Files (Download on cluster instead)
```bash
# These should NOT be transferred:
❌ weights/                              # Model weights (12GB+)
❌ data/                                 # Training data
❌ fastvideo/models/Hunyuan-GameCraft-1.0-main/  # Official repo (git ignored)
❌ *.pt, *.pth, *.safetensors           # Checkpoints
❌ *.mp4, *.png, *.jpg, *.gif           # Media files
❌ outputs/, results/, samples/         # Generated outputs
❌ cache_dir/, wandb/, runs/            # Caching/logging
```

### Build Artifacts
```bash
❌ __pycache__/
❌ *.pyc
❌ build/
❌ dist/
❌ *.egg-info/
❌ .venv/, venv/, env/
```

### System Files
```bash
❌ .DS_Store
❌ .vscode/
❌ *.swp, *.swo
```

---

## 🚀 Setup on GPU Cluster

### Step 1: Clone/Sync Code
```bash
# On GPU cluster
cd /path/to/your/workspace
git clone <your-fork> FastVideo
cd FastVideo
git checkout hunyuan-gamecraft-porting
```

### Step 2: Create Environment
```bash
# Create conda environment
conda create -n fastvideo_gamecraft python=3.12 -y
conda activate fastvideo_gamecraft

# Install PyTorch (adjust CUDA version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Install FastVideo dependencies
pip install -e .

# Install flash attention (recommended)
pip install flash-attn==2.7.4.post1 --no-build-isolation

# Optional: Install VSA for optimized attention
git submodule update --init --recursive
python setup_vsa.py install
```

### Step 3: Download Official GameCraft Repo (For Comparison Test)
```bash
# Clone official repo to models directory
cd fastvideo/models
git clone https://github.com/Tencent-Hunyuan/Hunyuan-GameCraft-1.0 Hunyuan-GameCraft-1.0-main
cd ../..

# This directory is in .gitignore - won't be committed
```

### Step 4: Download Model Weights
```bash
# Install HuggingFace CLI
pip install -U "huggingface_hub[cli]"

# Create weights directory
mkdir -p weights/gamecraft_models

# Download main model weights
cd weights/gamecraft_models
huggingface-cli download tencent/Hunyuan-GameCraft-1.0 \
  --include "weights/mp_rank_00_model_states.pt" \
  --local-dir .

# Optional: Download distilled model
huggingface-cli download tencent/Hunyuan-GameCraft-1.0 \
  --include "weights/mp_rank_00_model_states_distill.pt" \
  --local-dir .

cd ../..

# Set environment variable
export GAMECRAFT_MODEL_PATH="weights/gamecraft_models"
```

### Step 5: Verify Installation
```bash
# Check Python can import modules
python -c "from fastvideo.models.dits.hunyuangamecraft import HunyuanGameCraftTransformer3DModel; print('✓ Model import successful')"
python -c "from fastvideo.configs.models.dits import HunyuanGameCraftConfig; print('✓ Config import successful')"

# Check weights exist
ls -lh weights/gamecraft_models/mp_rank_00_model_states.pt
```

---

## 🧪 Run Tests on GPU Cluster

### Test Order (Progressive Validation)

Run tests in this order - each validates a different aspect:

#### Test 1: Basic Shape Test (No weights needed)
**Purpose:** Verify model architecture is correct

```bash
pytest fastvideo/tests/transformers/test_hunyuangamecraft.py::test_gamecraft_output_shape -v -s
```

**Success Criteria:** All shapes match (B, C, T, H, W)

---

#### Test 2: Camera Conditioning (No weights needed)
**Purpose:** Verify camera inputs affect outputs

```bash
pytest fastvideo/tests/transformers/test_hunyuangamecraft.py::test_gamecraft_camera_conditioning -v -s
```

**Success Criteria:** Different camera inputs produce different outputs

---

#### Test 3: 🎯 **NUMERICAL ALIGNMENT** (Requires weights + official repo) ⭐
**Purpose:** Verify your port matches original implementation exactly
**THIS IS THE CRITICAL TEST YOUR ADVISOR MENTIONED**

```bash
# Ensure everything is downloaded
export GAMECRAFT_MODEL_PATH="weights/gamecraft_models"

# Run the numerical comparison
pytest fastvideo/tests/transformers/test_hunyuangamecraft.py::test_gamecraft_vs_original -v -s
```

**Success Criteria:** 
- ✅ Max absolute difference < **1e-5** (0.00001)
- ✅ Mean absolute difference < 1e-6

**What this test does:**
1. Loads original GameCraft model from `fastvideo/models/Hunyuan-GameCraft-1.0-main/`
2. Loads your FastVideo GameCraft model
3. Creates identical inputs (same seed: 42)
4. Runs forward pass on both
5. Compares outputs element-wise
6. Reports numerical differences

**Expected output:**
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
Original output range: [-2.345678, 3.456789]
FastVideo output range: [-2.345679, 3.456790]

Absolute Difference:
  Max:  1.23e-06  ← MUST BE < 1e-5
  Mean: 3.45e-07

Relative Error:
  Max:  2.34e-07
  Mean: 1.12e-08
============================================================
✓✓✓ PASSED: Max diff 1.23e-06 < 1e-5
✓ FastVideo GameCraft matches original implementation!
```

**If test fails (diff > 1e-5):**
- Check checkpoint loaded correctly
- Verify same input shapes
- Check RoPE embeddings match
- Verify text embedding format (pooled vs unpooled)
- Check camera latent preprocessing

---

#### Test 4: Distributed Test + Reference Latent (Requires weights)
**Purpose:** Verify checkpoint loading consistency

```bash
pytest fastvideo/tests/transformers/test_hunyuangamecraft.py::test_gamecraft_transformer_distributed -v -s
```

**What to do:**
1. First run will output: `REFERENCE_LATENT = 123.456789...`
2. Copy this value to line 62 of `test_hunyuangamecraft.py`
3. Run again to verify consistency

**Success Criteria:** Same latent value across runs

---

### Run All Tests (Automated)
```bash
# Run validation script
chmod +x validate_gamecraft.sh
./validate_gamecraft.sh
```

---

## 🎯 Numerical Alignment Deep Dive

### Understanding the Thresholds

| Diff Level | Quality | Meaning |
|------------|---------|---------|
| < 1e-6 | Perfect | Essentially identical |
| < 1e-5 | ✅ Excellent | **Target for DiT models** |
| < 1e-4 | Good | Acceptable for VAE |
| < 1e-3 | Fair | May have visible artifacts |
| > 1e-3 | ❌ Poor | Implementation likely wrong |

### Common Issues and Fixes

**Issue 1: Max diff ~ 1e-3 (Too large)**
```python
# Likely causes:
- Checkpoint not loaded correctly
- Different weight initialization
- Missing some layers in forward pass
```
**Fix:** Check model architecture matches exactly

**Issue 2: Max diff ~ 1e-4 (Close but not quite)**
```python
# Likely causes:
- Text embedding format mismatch
- RoPE embeddings calculated differently
- Modulation layers have slight differences
```
**Fix:** Compare layer-by-layer outputs

**Issue 3: Max diff ~ 1e-6 or better (Perfect!)**
```python
# Success! Your port is numerically aligned ✓
```
**Next:** Set REFERENCE_LATENT and move to pipeline

---

## 📁 Minimal File Transfer List

If bandwidth is limited, transfer ONLY these files:

```bash
# Core implementation (3 files)
fastvideo/models/dits/hunyuangamecraft.py
fastvideo/configs/models/dits/hunyuangamecraft.py
fastvideo/models/registry.py

# Config exports (1 file)
fastvideo/configs/models/dits/__init__.py

# Tests (1 file)
fastvideo/tests/transformers/test_hunyuangamecraft.py

# Documentation (6 files)
GAMECRAFT_TODO.md
GAMECRAFT_VALIDATION_PLAN.md
README_GAMECRAFT.md
PR_CHECKLIST.md
TESTING_WITH_OFFICIAL_REPO.md
validate_gamecraft.sh

# Total: 12 files (~150KB)
```

Then download everything else on the cluster (weights, official repo, etc.)

---

## 🔍 Verify All Files Transferred

### Quick Checklist Script
```bash
#!/bin/bash
# Save as check_gamecraft_files.sh on GPU cluster

echo "Checking GameCraft files..."

files=(
  "fastvideo/models/dits/hunyuangamecraft.py"
  "fastvideo/configs/models/dits/hunyuangamecraft.py"
  "fastvideo/tests/transformers/test_hunyuangamecraft.py"
  "GAMECRAFT_TODO.md"
  "GAMECRAFT_VALIDATION_PLAN.md"
  "validate_gamecraft.sh"
)

missing=0
for file in "${files[@]}"; do
  if [ -f "$file" ]; then
    echo "✓ $file"
  else
    echo "✗ MISSING: $file"
    missing=$((missing + 1))
  fi
done

if [ $missing -eq 0 ]; then
  echo ""
  echo "✅ All GameCraft files present!"
else
  echo ""
  echo "❌ $missing files missing"
  exit 1
fi
```

---

## 🎯 Recommended Transfer Workflow

### For Clean Transfer (Recommended)
```bash
# 1. On local machine - commit everything
git add fastvideo/models/dits/hunyuangamecraft.py
git add fastvideo/configs/models/dits/hunyuangamecraft.py
git add fastvideo/models/registry.py
git add fastvideo/configs/models/dits/__init__.py
git add fastvideo/tests/transformers/test_hunyuangamecraft.py
git add *.md validate_gamecraft.sh
git commit -m "Add HunyuanGameCraft port with validation tests"
git push origin hunyuan-gamecraft-porting

# 2. On GPU cluster - pull changes
git clone <your-fork> FastVideo
cd FastVideo
git checkout hunyuan-gamecraft-porting
```

### For Quick Iteration (With uncommitted changes)
```bash
# Use rsync to sync only source code (excludes large files automatically)
rsync -avz --progress \
  --include='*/' \
  --include='*.py' \
  --include='*.md' \
  --include='*.sh' \
  --include='*.yaml' \
  --include='*.json' \
  --include='*.toml' \
  --exclude='*' \
  ./ username@gpu-cluster:/path/to/FastVideo/
```

---

## 💡 Pro Tips

### 1. Use Screen/Tmux
```bash
# On GPU cluster - tests can take hours
screen -S gamecraft_test
# or
tmux new -s gamecraft_test

# Run tests
pytest fastvideo/tests/transformers/test_hunyuangamecraft.py -v -s

# Detach: Ctrl+A, D (screen) or Ctrl+B, D (tmux)
```

### 2. Set Environment Variables Permanently
```bash
# Add to ~/.bashrc or ~/.zshrc on GPU cluster
echo 'export GAMECRAFT_MODEL_PATH="/path/to/weights/gamecraft_models"' >> ~/.bashrc
echo 'export FASTVIDEO_ATTENTION_BACKEND="VIDEO_SPARSE_ATTN"' >> ~/.bashrc
source ~/.bashrc
```

### 3. Download Weights Before Starting Tests
```bash
# Download in background while setting up
nohup huggingface-cli download tencent/Hunyuan-GameCraft-1.0 \
  --include "weights/*" \
  --local-dir weights/gamecraft_models > download.log 2>&1 &
```

### 4. Check GPU Availability
```bash
# On GPU cluster
nvidia-smi
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}')"
```

---

## 🆘 Troubleshooting

### Issue: Import Error
```bash
# Solution: Install in editable mode
pip install -e .
```

### Issue: CUDA Out of Memory
```bash
# Solution: Use CPU offloading
export CPU_OFFLOAD=1
# Or modify test to use smaller batch size
```

### Issue: Can't Find Weights
```bash
# Solution: Check path and set env var
ls -la weights/gamecraft_models/
export GAMECRAFT_MODEL_PATH="$(pwd)/weights/gamecraft_models"
```

### Issue: Official Repo Not Found
```bash
# Solution: Clone it to correct location
cd fastvideo/models
git clone https://github.com/Tencent-Hunyuan/Hunyuan-GameCraft-1.0 Hunyuan-GameCraft-1.0-main
cd ../..
```

---

## ✅ Success Criteria

You've successfully transferred when:
- [ ] All GameCraft files exist on cluster
- [ ] Python can import `HunyuanGameCraftTransformer3DModel`
- [ ] Shape test passes: `test_gamecraft_output_shape`
- [ ] Weights are downloaded
- [ ] Official repo is cloned (for comparison)
- [ ] GPU is accessible: `torch.cuda.is_available() == True`

---

## 📊 Expected File Sizes

```
Local machine transfer (code only):     ~150 KB
Model weights download on cluster:      ~12 GB
Official repo clone on cluster:         ~500 MB
Total disk space needed on cluster:     ~15 GB
```

---

## 🎉 Next Steps After Transfer

1. ✅ Verify all files with `check_gamecraft_files.sh`
2. ✅ Run basic tests (shape, camera)
3. ⏳ Download weights (12GB - takes time)
4. ⏳ Clone official repo
5. 🔬 Run numerical comparison
6. 📊 Generate reference latent
7. 🚀 Start pipeline integration

Good luck! 🚀
