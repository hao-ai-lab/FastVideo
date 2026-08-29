"""Test LoRA extraction, merging, and verification pipeline."""
import sys
from pathlib import Path

import pytest
import torch

# Add scripts/lora_extraction to path for imports
repo_root = Path(__file__).parents[3]
lora_scripts = repo_root / "scripts" / "lora_extraction"
sys.path.insert(0, str(lora_scripts))

# Import the core functions
from extract_lora import extract_lora_adapter
from merge_lora import merge_lora
from verify_lora import main as verify_lora_main


@pytest.mark.parametrize(
    "extraction_device",
    [
        pytest.param("cpu", id="cpu"),
        pytest.param(
            "cuda:0",
            id="gpu",
            marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable"),
        ),
    ],
)
def test_lora_extraction_pipeline(extraction_device: str):
    """Test the existing Wan2.2 extraction workflow on CPU and GPU."""
    import tempfile

    # Use temp directory for outputs to avoid polluting repo
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        device_name = extraction_device.replace(":", "-")
        adapter_path = tmpdir_path / f"adapter_r16_{device_name}.safetensors"
        merged_dir = tmpdir_path / f"merged_r16_{device_name}"

        # 1. Extract rank-16 adapter
        print(f"\nExtracting rank-16 adapter on {extraction_device}")
        extract_lora_adapter(
            base="Wan-AI/Wan2.2-TI2V-5B-Diffusers",
            ft="FastVideo/FastWan2.2-TI2V-5B-FullAttn-Diffusers",
            out=str(adapter_path),
            rank=16,
            load_mode="indexed",
            device=extraction_device,
            svd_method="exact",
        )
        assert adapter_path.exists(), "Adapter file was not created"

        # 2. Merge adapter
        print("\nMerging adapter")
        merge_lora(
            base="Wan-AI/Wan2.2-TI2V-5B-Diffusers",
            adapter=str(adapter_path),
            ft="FastVideo/FastWan2.2-TI2V-5B-FullAttn-Diffusers",
            output=str(merged_dir),
        )
        assert merged_dir.exists(), "Merged model directory was not created"

        # 3. Verify numerical accuracy
        print("\nVerifying merged model")
        # verify_lora uses sys.argv, so we need to mock it
        old_argv = sys.argv
        try:
            sys.argv = [
                "verify_lora.py",
                "--merged",
                str(merged_dir),
                "--ft",
                "FastVideo/FastWan2.2-TI2V-5B-FullAttn-Diffusers",
            ]
            verify_lora_main()
        finally:
            sys.argv = old_argv

        print("\nLoRA extraction pipeline test PASSED")
