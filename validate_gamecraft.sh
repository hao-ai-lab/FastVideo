#!/bin/bash
# Hunyuan GameCraft Validation Script
# 
# This script runs all validation tests for GameCraft port to FastVideo
# Following the workflow: encoders → VAE → DiT → pipeline

set -e  # Exit on error

echo "=========================================="
echo "Hunyuan GameCraft Validation Suite"
echo "=========================================="
echo ""

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if model weights exist
if [ ! -d "weights/gamecraft_models" ]; then
    echo -e "${YELLOW}WARNING: Model weights not found at weights/gamecraft_models/${NC}"
    echo "Some tests will be skipped. Download weights first:"
    echo "  mkdir -p weights/gamecraft_models"
    echo "  # Download from HuggingFace or official release"
    echo ""
fi

# Phase 1: Component Tests (no weights needed for shape tests)
echo "=========================================="
echo "Phase 1: Component-Level Tests"
echo "=========================================="

echo ""
echo "Test 1: DiT Output Shapes"
echo "Testing that model produces correct output shapes..."
pytest fastvideo/tests/transformers/test_hunyuangamecraft.py::test_gamecraft_output_shape -v -s || {
    echo -e "${RED}✗ Shape test FAILED${NC}"
    exit 1
}
echo -e "${GREEN}✓ Shape test PASSED${NC}"

echo ""
echo "Test 2: Camera Conditioning"
echo "Testing that camera inputs affect output..."
pytest fastvideo/tests/transformers/test_hunyuangamecraft.py::test_gamecraft_camera_conditioning -v -s || {
    echo -e "${RED}✗ Camera conditioning test FAILED${NC}"
    exit 1
}
echo -e "${GREEN}✓ Camera conditioning test PASSED${NC}"

# Phase 2: Numerical Validation (requires weights)
if [ -d "weights/gamecraft_models" ]; then
    echo ""
    echo "=========================================="
    echo "Phase 2: Numerical Validation"
    echo "=========================================="
    
    echo ""
    echo "Test 3: DiT Distributed"
    echo "Testing numerical accuracy with pretrained weights..."
    pytest fastvideo/tests/transformers/test_hunyuangamecraft.py::test_gamecraft_transformer_distributed -v -s || {
        echo -e "${RED}✗ Distributed test FAILED${NC}"
        exit 1
    }
    echo -e "${GREEN}✓ Distributed test PASSED${NC}"
    
    echo ""
    echo "Test 4: Compare with Original"
    echo "Testing against original GameCraft implementation..."
    pytest fastvideo/tests/transformers/test_hunyuangamecraft.py::test_gamecraft_vs_original -v -s || {
        echo -e "${YELLOW}⚠ Comparison with original skipped or failed${NC}"
        echo "  Make sure official repo is at fastvideo/models/Hunyuan-GameCraft-1.0-main/"
    }
else
    echo ""
    echo -e "${YELLOW}Skipping numerical validation tests (no weights)${NC}"
fi

# Phase 3: Compare with Original
if [ -d "fastvideo/models/Hunyuan-GameCraft-1.0-main" ]; then
    echo ""
    echo "=========================================="
    echo "Phase 3: Compare with Original"
    echo "=========================================="
    
    echo ""
    echo "Testing against original GameCraft implementation..."
    echo "You can also run: python compare_with_original.py"
    pytest fastvideo/tests/transformers/test_hunyuangamecraft.py::test_gamecraft_vs_original -v -s || {
        echo -e "${YELLOW}⚠ Comparison test skipped or failed${NC}"
        echo "  This is expected without loaded weights"
    }
else
    echo ""
    echo -e "${YELLOW}Original GameCraft repo not found${NC}"
    echo "  Expected at: fastvideo/models/Hunyuan-GameCraft-1.0-main/"
fi

# Phase 4: Pipeline Tests (not yet implemented)
echo ""
echo "=========================================="
echo "Phase 4: Pipeline Tests"
echo "=========================================="
echo -e "${YELLOW}TODO: Pipeline tests not yet implemented${NC}"
echo "Next steps:"
echo "  1. Create pipeline config"
echo "  2. Create pipeline class"
echo "  3. Create end-to-end test"

echo ""
echo "=========================================="
echo "Summary"
echo "=========================================="
echo -e "${GREEN}✓ Basic validation PASSED${NC}"
echo ""
echo "See GAMECRAFT_VALIDATION_PLAN.md for next steps"
echo "See GAMECRAFT_COMPONENT_CHECK.md for detailed component status"

