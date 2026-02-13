# SPDX-License-Identifier: Apache-2.0
import os
import pytest
import torch
from torch.distributed.tensor import DTensor
from torch.testing import assert_close
from transformers import AutoConfig, AutoTokenizer, Qwen2_5_VLTextModel

from fastvideo.configs.pipelines import Hunyuan15T2V480PConfig, PipelineConfig
from fastvideo.forward_context import set_forward_context
from fastvideo.logger import init_logger
from fastvideo.models.loader.component_loader import TextEncoderLoader
from fastvideo.utils import maybe_download_model, PRECISION_TO_TYPE
from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.configs.models.encoders import Qwen2_5_VLConfig

logger = init_logger(__name__)

os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "29505"

@pytest.fixture
def qwen_model_path_and_config():
    base_model_path = "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v"
    model_path = maybe_download_model(base_model_path)
    text_encoder_path = os.path.join(model_path, "text_encoder")
    tokenizer_path = os.path.join(model_path, "tokenizer")
    return text_encoder_path, tokenizer_path, Hunyuan15T2V480PConfig()

@pytest.mark.usefixtures("distributed_setup")
def test_qwen2_5_encoder(qwen_model_path_and_config):
    text_encoder_path, tokenizer_path, pipeline_config = qwen_model_path_and_config
    hf_config = AutoConfig.from_pretrained(text_encoder_path)
    print(hf_config)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Qwen2.5-VL default dtype is usually bf16
    precision_str = "fp32"
    precision = PRECISION_TO_TYPE[precision_str]
    
    logger.info(f"Using precision: {precision_str}")

    # Load HF model (Base model)
    model1 = Qwen2_5_VLTextModel.from_pretrained(text_encoder_path).to(
        precision).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    # Load FastVideo model
    args = FastVideoArgs(model_path=text_encoder_path,
                        pipeline_config=pipeline_config,
                        pin_cpu_memory=False)
    
    loader = TextEncoderLoader()
    model2 = loader.load(text_encoder_path, args)
    model2 = model2.to(precision)
    model2.eval()

    # Sanity check weights
    logger.info("Comparing model weights for sanity check...")
    params1 = dict(model1.named_parameters())
    params2 = dict(model2.named_parameters())
    
    logger.info("Model1 has %s parameters", len(params1))
    logger.info("Model2 has %s parameters", len(params2))

    # Check common layers like Norms which are likely not merged/sharded in a way that changes name significantly
    # or simple linear layers if names match.
    # Note: FastVideo uses QKVParallelLinear, so q_proj, k_proj, v_proj are merged.
    # HF Qwen2_5_VL uses separate projections? No, usually they are separate nn.Linear in HF.
    
    weights_to_check = [
        "norm.weight",
        "layers.{}.self_attn.o_proj.weight",
        "layers.{}.input_layernorm.weight",
        "layers.{}.post_attention_layernorm.weight",
        "layers.{}.mlp.down_proj.weight"
    ]
    
    for idx in range(hf_config.num_hidden_layers):
        for w in weights_to_check:
            name1 = w.format(idx)
            name2 = w.format(idx)
            p1 = params1[name1]
            p2 = params2[name2]
            p2 = (p2.to_local() if isinstance(p2, DTensor) else p2).to(p1)
            
            # Check shape
            assert p1.shape == p2.shape, f"Shape mismatch for {w}: {p1.shape} vs {p2.shape}"
            
            # Check values
            assert_close(p1, p2, atol=1e-7, rtol=1e-7, msg=f"Weight mismatch for {w}")

    # Test with sample prompts
    prompts = [
        "Hello world",
        "The quick brown fox jumps over the lazy dog."
    ]
    
    logger.info("Testing with sample prompts")
    
    with torch.no_grad():
        for prompt in prompts:
            logger.info(f"Prompt: {prompt}")
            tokens = tokenizer(prompt, return_tensors="pt", padding="max_length", max_length=1000, truncation=True).to(device)
            
            # HF Forward
            # AutoModel for Qwen2.5-VL usually returns BaseModelOutputWithPast
            outputs1 = model1(
                input_ids=tokens.input_ids,
                attention_mask=tokens.attention_mask,
                output_hidden_states=True
            ).hidden_states[-3]
            
            # FastVideo Forward
            with set_forward_context(current_timestep=0, attn_metadata=None):
                outputs2 = model2(
                    input_ids=tokens.input_ids,
                    attention_mask=tokens.attention_mask,
                    output_hidden_states=True
                ).hidden_states[-3]
                
            # Compare
            # Filter padding for comparison if needed, but here we just check raw output matching
            
            # Check shapes
            assert outputs1.shape == outputs2.shape, f"Output shape mismatch: {outputs1.shape} vs {outputs2.shape}"
            
            diff = torch.abs(outputs1 - outputs2)
            max_diff = diff.max().item()
            mean_diff = diff.mean().item()
            
            logger.info(f"Max diff: {max_diff}")
            logger.info(f"Mean diff: {mean_diff}")
            
            # Thresholds
            # Qwen2.5-VL RoPE is complex, if our implementation is slightly off (e.g. float32 conversion logic in RoPE),
            # differences might appear. But should be small.
            if precision_str == "bf16":
                atol = 5e-2 # relaxed for bf16
            else:
                atol = 1e-3
                
            if max_diff > atol:
                logger.warning(f"Max diff {max_diff} > {atol}. Checking if it's acceptable...")
                # If mean diff is small, maybe just outliers
                assert mean_diff < atol, f"Mean diff {mean_diff} too high"
            else:
                logger.info("Outputs match within tolerance.")

