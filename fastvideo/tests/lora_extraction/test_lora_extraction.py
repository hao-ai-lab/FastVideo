"""Test LoRA extraction, merging, and verification pipeline."""
import sys
from pathlib import Path

# Add scripts/lora_extraction to path for imports
repo_root = Path(__file__).parents[3]
lora_scripts = repo_root / "scripts" / "lora_extraction"
sys.path.insert(0, str(lora_scripts))

# Import the core functions
from extract_lora import extract_lora_adapter
from merge_lora import merge_lora
from verify_lora import main as verify_lora_main


def test_lora_extraction_pipeline():
    """Test end-to-end LoRA extraction workflow."""
    import tempfile
    
    # Use temp directory for outputs to avoid polluting repo
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        adapter_path = tmpdir_path / "adapter_r16.safetensors"
        merged_dir = tmpdir_path / "merged_r16"
        
        # 1. Extract rank-16 adapter
        print("\nExtracting rank-16 adapter")
        extract_lora_adapter(
            base="Wan-AI/Wan2.2-TI2V-5B-Diffusers",
            ft="FastVideo/FastWan2.2-TI2V-5B-FullAttn-Diffusers",
            out=str(adapter_path),
            rank=16,
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
                "--merged", str(merged_dir),
                "--ft", "FastVideo/FastWan2.2-TI2V-5B-FullAttn-Diffusers",
            ]
            verify_lora_main()
        finally:
            sys.argv = old_argv
        
        print("\nLoRA extraction pipeline test PASSED")
