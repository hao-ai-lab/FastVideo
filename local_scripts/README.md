# Model Weight Comparison Tools

This directory contains tools for loading and comparing model weights between different formats.

## Files

- `weight_compare.py` - Comprehensive model comparison tool
- `simple_model_loader.py` - Simple model inspection tool
- `requirements.txt` - Required dependencies
- `README.md` - This file

## Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### 1. Simple Model Inspection

To quickly inspect the structure of both models:

```bash
python simple_model_loader.py
```

This will:
- Load both models and show their basic structure
- Display the number of parameters
- Show sample parameter keys and shapes
- Calculate total parameter count

### 2. Comprehensive Model Comparison

To perform a detailed comparison between the models:

```bash
python weight_compare.py
```

This will:
- Load both models
- Compare parameter names and structures
- Identify common and unique parameters
- Calculate differences between matching parameters
- Generate a detailed report saved to `model_comparison_results.txt`

## Model Paths

The scripts are configured to load:

1. **DMD PyTorch Model**: `/mnt/sharefs/users/hao.zhang/DMD/wan_bidirectional_dmd_from_scratch/2025-06-20-08-17-06.607828_seed1024/checkpoint_model_004800/model.pt`

2. **Wan2.1 Safetensors Model**: `../Wan-AI/Wan2.1-T2V-1.3B-Diffusers/transformer/diffusion_pytorch_model.safetensors`

## Output

The comparison tool generates:
- Console output with summary statistics
- Detailed report file (`model_comparison_results.txt`) containing:
  - Parameter counts for each model
  - List of common and unique parameters
  - Top parameters with largest differences
  - Shape mismatches
  - Statistical analysis of differences

## Features

- **Flexible Loading**: Handles different PyTorch checkpoint formats
- **Memory Efficient**: Uses CPU loading to avoid GPU memory issues
- **Detailed Analysis**: Provides comprehensive comparison metrics
- **Error Handling**: Graceful handling of missing files and loading errors
- **Progress Reporting**: Shows loading progress and intermediate results 