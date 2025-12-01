"""Test LoRA extraction, merging, and verification pipeline."""
import subprocess
import sys
from pathlib import Path

def test_lora_extraction_pipeline():
    """Test end-to-end LoRA extraction workflow."""
    repo_root = Path(__file__).parents[3]
    lora_scripts = repo_root / "scripts" / "lora_extraction"
    
    # 1. Extract rank-16 adapter
    print("\nExtracting rank-16 adapter")
    result = subprocess.run([
        sys.executable, "extract_lora.py",
        "--base", "Wan-AI/Wan2.2-TI2V-5B-Diffusers",
        "--ft", "FastVideo/FastWan2.2-TI2V-5B-FullAttn-Diffusers",
        "--out", "adapter_r16.safetensors",
        "--rank", "16"
    ], cwd=lora_scripts, capture_output=True, text=True)
    
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr, file=sys.stderr)
        raise AssertionError(f"Extract failed with code {result.returncode}")
    
    # 2. Merge adapter
    print("\nMerging adapter")
    result = subprocess.run([
        sys.executable, "merge_lora.py",
        "--base", "Wan-AI/Wan2.2-TI2V-5B-Diffusers",
        "--adapter", "adapter_r16.safetensors",
        "--ft", "FastVideo/FastWan2.2-TI2V-5B-FullAttn-Diffusers",
        "--output", "merged_r16"
    ], cwd=lora_scripts, capture_output=True, text=True)
    
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr, file=sys.stderr)
        raise AssertionError(f"Merge failed with code {result.returncode}")
    
    # 3. Verify numerical accuracy
    print("\nVerifying merged model")
    result = subprocess.run([
        sys.executable, "verify_lora.py",
        "--merged", "merged_r16",
        "--ft", "FastVideo/FastWan2.2-TI2V-5B-FullAttn-Diffusers"
    ], cwd=lora_scripts, capture_output=True, text=True)
    
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr, file=sys.stderr)
        raise AssertionError(f"Verify failed with code {result.returncode}")
    
    # 4. Test inference quality (minimal params for CI speed)
    print("\nTesting inference quality")
    result = subprocess.run([
        sys.executable, "lora_inference_comparison.py",
        "--base", "merged_r16",
        "--ft", "FastVideo/FastWan2.2-TI2V-5B-FullAttn-Diffusers",
        "--adapter", "NONE",
        "--output-dir", "inference_test",
        "--prompt", "A cat sitting on a windowsill",
        "--seed", "42",
        "--height", "480",
        "--width", "480",
        "--num-frames", "25",
        "--num-inference-steps", "16",
        "--compute-ssim",
        "--compute-lpips"
    ], cwd=lora_scripts, capture_output=True, text=True)
    
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr, file=sys.stderr)
        raise AssertionError(f"Inference comparison failed with code {result.returncode}")
    
    # Check that SSIM/LPIPS metrics were computed
    import json
    metrics_file = lora_scripts / "inference_test" / "steps16_A cat sitting on a windowsill_ssim.json"
    if not metrics_file.exists():
        raise AssertionError("SSIM metrics file not found")
    
    with open(metrics_file) as f:
        metrics = json.load(f)
    
    ssim = metrics.get("ssim")
    lpips = metrics.get("lpips")
    
    if ssim is None or lpips is None:
        raise AssertionError("SSIM or LPIPS metrics missing")
    
    print(f"\nMetrics: SSIM={ssim:.4f}, LPIPS={lpips:.4f}")