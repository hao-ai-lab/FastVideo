#!/bin/bash
# Verify all GameCraft files are present on GPU cluster

echo "=========================================="
echo "GameCraft File Verification"
echo "=========================================="
echo ""

# Core implementation files
echo "📦 Core Implementation Files:"
files_core=(
  "fastvideo/models/dits/hunyuangamecraft.py"
  "fastvideo/configs/models/dits/hunyuangamecraft.py"
  "fastvideo/models/registry.py"
  "fastvideo/configs/models/dits/__init__.py"
)

missing_core=0
for file in "${files_core[@]}"; do
  if [ -f "$file" ]; then
    lines=$(wc -l < "$file")
    echo "  ✓ $file ($lines lines)"
  else
    echo "  ✗ MISSING: $file"
    missing_core=$((missing_core + 1))
  fi
done
echo ""

# Test files
echo "🧪 Test Files:"
files_test=(
  "fastvideo/tests/transformers/test_hunyuangamecraft.py"
)

missing_test=0
for file in "${files_test[@]}"; do
  if [ -f "$file" ]; then
    lines=$(wc -l < "$file")
    echo "  ✓ $file ($lines lines)"
  else
    echo "  ✗ MISSING: $file"
    missing_test=$((missing_test + 1))
  fi
done
echo ""

# Documentation files
echo "📚 Documentation Files:"
files_docs=(
  "GAMECRAFT_TODO.md"
  "GAMECRAFT_VALIDATION_PLAN.md"
  "README_GAMECRAFT.md"
  "PR_CHECKLIST.md"
  "TESTING_WITH_OFFICIAL_REPO.md"
  "validate_gamecraft.sh"
  "TRANSFER_TO_GPU_CLUSTER.md"
)

missing_docs=0
for file in "${files_docs[@]}"; do
  if [ -f "$file" ]; then
    echo "  ✓ $file"
  else
    echo "  ✗ MISSING: $file"
    missing_docs=$((missing_docs + 1))
  fi
done
echo ""

# Check Python imports
echo "🐍 Python Import Tests:"
python_imports=(
  "from fastvideo.models.dits.hunyuangamecraft import HunyuanGameCraftTransformer3DModel"
  "from fastvideo.configs.models.dits import HunyuanGameCraftConfig"
)

import_errors=0
for import_cmd in "${python_imports[@]}"; do
  if python -c "$import_cmd" 2>/dev/null; then
    echo "  ✓ $import_cmd"
  else
    echo "  ✗ FAILED: $import_cmd"
    import_errors=$((import_errors + 1))
  fi
done
echo ""

# Check external dependencies
echo "📥 External Dependencies:"

# Check official repo
if [ -d "fastvideo/models/Hunyuan-GameCraft-1.0-main" ]; then
  echo "  ✓ Official GameCraft repo cloned"
else
  echo "  ⚠ Official repo not found (needed for test_gamecraft_vs_original)"
  echo "    Clone with: cd fastvideo/models && git clone https://github.com/Tencent-Hunyuan/Hunyuan-GameCraft-1.0 Hunyuan-GameCraft-1.0-main"
fi

# Check weights
if [ -d "weights/gamecraft_models" ]; then
  echo "  ✓ Weights directory exists"
  if [ -f "weights/gamecraft_models/mp_rank_00_model_states.pt" ]; then
    size=$(du -h "weights/gamecraft_models/mp_rank_00_model_states.pt" | cut -f1)
    echo "    ✓ Main checkpoint found ($size)"
  else
    echo "    ⚠ Main checkpoint not found"
    echo "      Download with: huggingface-cli download tencent/Hunyuan-GameCraft-1.0 --include 'weights/mp_rank_00_model_states.pt' --local-dir weights/gamecraft_models"
  fi
  
  if [ -f "weights/gamecraft_models/mp_rank_00_model_states_distill.pt" ]; then
    size=$(du -h "weights/gamecraft_models/mp_rank_00_model_states_distill.pt" | cut -f1)
    echo "    ✓ Distilled checkpoint found ($size)"
  else
    echo "    ⚠ Distilled checkpoint not found (optional)"
  fi
else
  echo "  ⚠ Weights directory not found"
  echo "    Create with: mkdir -p weights/gamecraft_models"
fi
echo ""

# Check GPU
echo "🎮 GPU Status:"
if command -v nvidia-smi &> /dev/null; then
  gpu_count=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
  echo "  ✓ nvidia-smi available"
  echo "  ✓ GPUs detected: $gpu_count"
  nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader | while IFS=, read -r idx name memory; do
    echo "    - GPU $idx: $name ($memory)"
  done
else
  echo "  ✗ nvidia-smi not found"
fi

if python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
  gpu_count=$(python -c "import torch; print(torch.cuda.device_count())")
  echo "  ✓ PyTorch CUDA available ($gpu_count GPUs)"
else
  echo "  ✗ PyTorch CUDA not available"
fi
echo ""

# Summary
echo "=========================================="
echo "Summary"
echo "=========================================="
total_missing=$((missing_core + missing_test + missing_docs))

if [ $total_missing -eq 0 ] && [ $import_errors -eq 0 ]; then
  echo "✅ All essential files present and imports working!"
  echo ""
  echo "🚀 Ready for testing! Run:"
  echo "   pytest fastvideo/tests/transformers/test_hunyuangamecraft.py::test_gamecraft_output_shape -v"
  exit 0
else
  echo "❌ Issues found:"
  [ $missing_core -gt 0 ] && echo "   - $missing_core core files missing"
  [ $missing_test -gt 0 ] && echo "   - $missing_test test files missing"
  [ $missing_docs -gt 0 ] && echo "   - $missing_docs doc files missing"
  [ $import_errors -gt 0 ] && echo "   - $import_errors import errors"
  echo ""
  echo "See TRANSFER_TO_GPU_CLUSTER.md for setup instructions"
  exit 1
fi
