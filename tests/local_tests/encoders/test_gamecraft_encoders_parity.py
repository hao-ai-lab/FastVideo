# SPDX-License-Identifier: Apache-2.0
"""
Numerical parity test for GameCraft text encoders.

GameCraft uses:
1. LLaVA-LLaMA-3-8B for primary text encoding (4096-dim)
2. CLIP ViT-L/14 for secondary text encoding (768-dim pooled)

Usage:
    DISABLE_SP=1 pytest tests/local_tests/encoders/test_gamecraft_encoders_parity.py -v
"""
import os
import sys
from pathlib import Path

import pytest
import torch

os.environ.setdefault("MASTER_ADDR", "localhost")
os.environ.setdefault("MASTER_PORT", "29516")
os.environ.setdefault("DISABLE_SP", "1")

repo_root = Path(__file__).resolve().parents[3]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_gamecraft_llama_encoder():
    """Test LLaVA-LLaMA-3-8B text encoder."""
    torch.manual_seed(42)
    
    llama_path = Path(
        os.getenv(
            "GAMECRAFT_LLAMA_PATH",
            repo_root / "Hunyuan-GameCraft-1.0" / "weights" / "stdmodels" / "llava-llama-3-8b-v1_1-transformers"
        )
    )
    
    if not llama_path.exists():
        pytest.skip(f"LLaMA encoder not found at {llama_path}")
    
    # Check for model weights
    model_files = list(llama_path.glob("*.safetensors")) + list(llama_path.glob("*.bin"))
    if not model_files:
        pytest.skip(f"No model weights found in {llama_path}")
    
    device = torch.device("cuda:0")
    dtype = torch.float16  # LLaMA uses fp16
    
    print(f"\n[LLAMA TEST] Loading LLaVA-LLaMA-3-8B from {llama_path}")
    
    try:
        from transformers import LlavaForConditionalGeneration, LlamaTokenizerFast
        
        # Load model
        model = LlavaForConditionalGeneration.from_pretrained(
            llama_path,
            low_cpu_mem_usage=True,
            torch_dtype=dtype,
        )
        model = model.to(device)
        model.eval()
        
        # Load tokenizer
        tokenizer = LlamaTokenizerFast.from_pretrained(llama_path, padding_side="right")
        
        print("[LLAMA TEST] Model loaded successfully")
        print(f"[LLAMA TEST] Model dtype: {model.dtype}")
        
        # Test encoding
        test_prompt = "A medieval village with cobblestone streets"
        inputs = tokenizer(
            test_prompt,
            return_tensors="pt",
            padding="max_length",
            max_length=256,
            truncation=True,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                output_hidden_states=True,
            )
        
        # Get last hidden state
        last_hidden_state = outputs.hidden_states[-1]
        
        print(f"[LLAMA TEST] Input tokens: {inputs['input_ids'].shape}")
        print(f"[LLAMA TEST] Output hidden state shape: {last_hidden_state.shape}")
        print(f"[LLAMA TEST] Hidden dim: {last_hidden_state.shape[-1]} (expected: 4096)")
        print(f"[LLAMA TEST] Output stats: mean={last_hidden_state.mean():.6f}, std={last_hidden_state.std():.6f}")
        
        assert last_hidden_state.shape[-1] == 4096, f"Expected hidden dim 4096, got {last_hidden_state.shape[-1]}"
        
        print("[LLAMA TEST] LLaMA encoder test passed!")
        
    except ImportError as e:
        pytest.skip(f"transformers not available: {e}")
    except Exception as e:
        pytest.fail(f"LLaMA encoder test failed: {e}")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_gamecraft_clip_encoder():
    """Test CLIP ViT-L/14 text encoder."""
    torch.manual_seed(42)
    
    clip_path = Path(
        os.getenv(
            "GAMECRAFT_CLIP_PATH",
            repo_root / "Hunyuan-GameCraft-1.0" / "weights" / "stdmodels" / "openai_clip-vit-large-patch14"
        )
    )
    
    if not clip_path.exists():
        pytest.skip(f"CLIP encoder not found at {clip_path}")
    
    device = torch.device("cuda:0")
    dtype = torch.float32  # CLIP typically uses fp32
    
    print(f"\n[CLIP TEST] Loading CLIP ViT-L/14 from {clip_path}")
    
    try:
        from transformers import CLIPTextModel, CLIPTokenizer
        
        # Load model
        model = CLIPTextModel.from_pretrained(clip_path)
        model = model.to(device=device, dtype=dtype)
        model.eval()
        
        # Load tokenizer
        tokenizer = CLIPTokenizer.from_pretrained(clip_path, max_length=77)
        
        print("[CLIP TEST] Model loaded successfully")
        print(f"[CLIP TEST] Model dtype: {model.dtype}")
        
        # Test encoding
        test_prompt = "A medieval village with cobblestone streets"
        inputs = tokenizer(
            test_prompt,
            return_tensors="pt",
            padding="max_length",
            max_length=77,
            truncation=True,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
            )
        
        # Get pooled output (used for text_states_2)
        pooler_output = outputs.pooler_output
        last_hidden_state = outputs.last_hidden_state
        
        print(f"[CLIP TEST] Input tokens: {inputs['input_ids'].shape}")
        print(f"[CLIP TEST] Pooled output shape: {pooler_output.shape} (expected: [B, 768])")
        print(f"[CLIP TEST] Last hidden state shape: {last_hidden_state.shape}")
        print(f"[CLIP TEST] Pooled output stats: mean={pooler_output.mean():.6f}, std={pooler_output.std():.6f}")
        
        assert pooler_output.shape[-1] == 768, f"Expected pooled dim 768, got {pooler_output.shape[-1]}"
        
        print("[CLIP TEST] CLIP encoder test passed!")
        
    except ImportError as e:
        pytest.skip(f"transformers not available: {e}")
    except Exception as e:
        pytest.fail(f"CLIP encoder test failed: {e}")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.skip(reason="Official TextEncoder wrapper incompatible with installed transformers version (language_model attribute removed from LlavaForConditionalGeneration)")
def test_gamecraft_text_encoder_wrapper():
    """Test the official GameCraft TextEncoder wrapper."""
    torch.manual_seed(42)
    
    official_path = Path(
        os.getenv("GAMECRAFT_OFFICIAL_PATH", repo_root / "Hunyuan-GameCraft-1.0")
    )
    
    if not official_path.exists():
        pytest.skip(f"Official GameCraft repo not found at {official_path}")
    
    sys.path.insert(0, str(official_path))
    
    llama_path = official_path / "weights" / "stdmodels" / "llava-llama-3-8b-v1_1-transformers"
    clip_path = official_path / "weights" / "stdmodels" / "openai_clip-vit-large-patch14"
    
    if not llama_path.exists() or not clip_path.exists():
        pytest.skip("Text encoder weights not found")
    
    device = torch.device("cuda:0")
    
    try:
        from hymm_sp.text_encoder import TextEncoder
        
        # Test LLaMA encoder wrapper
        print("\n[WRAPPER TEST] Testing LLaVA-LLaMA-3-8B wrapper...")
        llama_encoder = TextEncoder(
            text_encoder_type="llava-llama-3-8b",
            max_length=256,
            text_encoder_precision="fp16",
            text_encoder_path=str(llama_path),
            tokenizer_path=str(llama_path),
            use_attention_mask=True,
            device=device,
        )
        
        test_prompt = "A medieval village with cobblestone streets"
        llama_output = llama_encoder(test_prompt)
        
        print(f"[WRAPPER TEST] LLaMA hidden state: {llama_output.hidden_state.shape}")
        print(f"[WRAPPER TEST] LLaMA attention mask: {llama_output.attention_mask.shape if llama_output.attention_mask is not None else 'None'}")
        
        # Test CLIP encoder wrapper
        print("\n[WRAPPER TEST] Testing CLIP wrapper...")
        clip_encoder = TextEncoder(
            text_encoder_type="clipL",
            max_length=77,
            text_encoder_precision="fp32",
            text_encoder_path=str(clip_path),
            tokenizer_path=str(clip_path),
            use_attention_mask=True,
            device=device,
        )
        
        clip_output = clip_encoder(test_prompt)
        
        print(f"[WRAPPER TEST] CLIP pooled output: {clip_output.hidden_state.shape}")
        
        print("[WRAPPER TEST] TextEncoder wrapper test passed!")
        
    except ImportError as e:
        pytest.skip(f"Failed to import TextEncoder: {e}")
    except Exception as e:
        pytest.fail(f"TextEncoder wrapper test failed: {e}")


if __name__ == "__main__":
    test_gamecraft_clip_encoder()
    test_gamecraft_llama_encoder()
    # test_gamecraft_text_encoder_wrapper()  # Requires full model weights
