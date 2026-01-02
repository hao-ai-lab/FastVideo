# SPDX-License-Identifier: Apache-2.0
import os

import numpy as np
import pytest
import torch
from diffusers import WanTransformer3DModel
from torch.testing import assert_close

from fastvideo.configs.pipelines import PipelineConfig
from fastvideo.forward_context import set_forward_context
from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.logger import init_logger
# from fastvideo.models.loader.component_loader import TransformerLoader
from fastvideo.utils import maybe_download_model
from fastvideo.configs.models.dits import WanVideoConfig
from fastvideo.models.dits.wan2_ref.wan2pt1 import WanModel
import re
import glob
from safetensors.torch import load_file as load_safetensors
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch


logger = init_logger(__name__)

os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "29503"

BASE_MODEL_PATH = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
MODEL_PATH = maybe_download_model(BASE_MODEL_PATH,
                                  local_dir=os.path.join("data", BASE_MODEL_PATH) # store in the large /workspace disk on Runpod
                                  )
TRANSFORMER_PATH = os.path.join(MODEL_PATH, "transformer")
print(TRANSFORMER_PATH)

@pytest.mark.usefixtures("distributed_setup")
def test_wan_transformer():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    precision = torch.bfloat16
    precision_str = "bf16"
    args = FastVideoArgs(model_path=TRANSFORMER_PATH,
                         dit_cpu_offload=True,
                         pipeline_config=PipelineConfig(dit_config=WanVideoConfig(), dit_precision=precision_str))
    args.device = device




    # 1.3B model args from TurboDiffusion/turbodiffusion/rcm/configs/defaults/net.py
    wan2pt1_1pt3B_net_args = dict(
        dim=1536,
        eps=1e-06,
        ffn_dim=8960,
        freq_dim=256,
        in_dim=16,
        num_heads=12,
        num_layers=30,
        out_dim=16,
        text_len=512,
    )

    model2 = WanModel(
        **wan2pt1_1pt3B_net_args,
        model_type="t2v",
        patch_size=(1, 2, 2), # Default in WanModel, explicit here
        qk_norm=True, # Default
        cross_attn_norm=True, # Config says True in WanVideoConfig default? No, WanVideoConfig default is True? 
        # WanVideoConfig (step 85) says cross_attn_norm=True.
        # WanModel (step 8) defaults cross_attn_norm=False? No, let's check step 8.
        # Step 8: check WanModel init. Warning: WanModel init defaults: cross_attn_norm=True.
        # Wait, step 8 snippet of WanModel init:
        # cross_attn_norm=True (line 560).
        # So defaults match.
        # However, checking qk_norm. WanModel default True.
    ).to(device, dtype=precision).eval()

    # Load Diffusers weights from disk
    diffusers_state_dict = {}
    safetensor_files = glob.glob(os.path.join(TRANSFORMER_PATH, "*.safetensors"))
    for sf in safetensor_files:
        diffusers_state_dict.update(load_safetensors(sf))
        
    # Prepare state dict for Reference Model (model2)
    ref_state_dict = {}
    
    # Manual Mapping Ref -> Diffusers (FastVideo)
    manual_mapping = {
        r"^patch_embedding\.(.*)": r"patch_embedding.\1", 
        r"^text_embedding\.0\.(.*)": r"condition_embedder.text_embedder.linear_1.\1",
        r"^text_embedding\.2\.(.*)": r"condition_embedder.text_embedder.linear_2.\1",
        r"^time_embedding\.0\.(.*)": r"condition_embedder.time_embedder.linear_1.\1",
        r"^time_embedding\.2\.(.*)": r"condition_embedder.time_embedder.linear_2.\1",
        r"^time_projection\.1\.(.*)": r"condition_embedder.time_proj.\1",
        r"^blocks\.(\d+)\.modulation": r"blocks.\1.scale_shift_table",
        r"^blocks\.(\d+)\.self_attn\.q\.(.*)": r"blocks.\1.attn1.to_q.\2",
        r"^blocks\.(\d+)\.self_attn\.k\.(.*)": r"blocks.\1.attn1.to_k.\2",
        r"^blocks\.(\d+)\.self_attn\.v\.(.*)": r"blocks.\1.attn1.to_v.\2",
        r"^blocks\.(\d+)\.self_attn\.o\.(.*)": r"blocks.\1.attn1.to_out.0.\2",
        r"^blocks\.(\d+)\.self_attn\.norm_q\.(.*)": r"blocks.\1.attn1.norm_q.\2",
        r"^blocks\.(\d+)\.self_attn\.norm_k\.(.*)": r"blocks.\1.attn1.norm_k.\2",
        r"^blocks\.(\d+)\.cross_attn\.q\.(.*)": r"blocks.\1.attn2.to_q.\2",
        r"^blocks\.(\d+)\.cross_attn\.k\.(.*)": r"blocks.\1.attn2.to_k.\2",
        r"^blocks\.(\d+)\.cross_attn\.v\.(.*)": r"blocks.\1.attn2.to_v.\2",
        r"^blocks\.(\d+)\.cross_attn\.o\.(.*)": r"blocks.\1.attn2.to_out.0.\2",
        r"^blocks\.(\d+)\.cross_attn\.norm_q\.(.*)": r"blocks.\1.attn2.norm_q.\2",
        r"^blocks\.(\d+)\.cross_attn\.norm_k\.(.*)": r"blocks.\1.attn2.norm_k.\2",
        r"^blocks\.(\d+)\.norm1\.(.*)": r"blocks.\1.norm1.\2",
        # Ref norm3 is Pre-CrossAttn -> Diffusers norm2
        r"^blocks\.(\d+)\.norm3\.(.*)": r"blocks.\1.norm2.\2",
        # Ref norm2 is Pre-FFN -> Diffusers norm3 (assuming standard DiT)
        r"^blocks\.(\d+)\.norm2\.(.*)": r"blocks.\1.norm3.\2",
        
        r"^blocks\.(\d+)\.ffn\.0\.(.*)": r"blocks.\1.ffn.net.0.proj.\2",
        r"^blocks\.(\d+)\.ffn\.2\.(.*)": r"blocks.\1.ffn.net.2.\2",
        # Head
        r"^head\.head\.(.*)": r"proj_out.\1",
        r"^head\.norm\.(.*)": r"norm_out.\1",
        r"^head\.modulation": r"scale_shift_table", # Assumed mapping
    }
    
    loaded_count = 0
    total_count = 0
    used_diffusers_keys = set()  # Track which Diffusers keys are used
    
    # logger.info("Diffusers 'head' or 'norm_out' keys:")
    # for k in diffusers_state_dict.keys():
    #     if "head" in k or "norm_out" in k:
    #         logger.info(f"{k}: {diffusers_state_dict[k].shape}")

    missing_keys = []
    # Iterate over Reference Model keys and find corresponding Diffusers key
    for name, param in model2.named_parameters():
        total_count += 1
        # Strip checkpoint wrapper prefix if present
        clean_name = name.replace("_checkpoint_wrapped_module.", "")
        
        mapped_name = clean_name
        matched = False
        for pattern, repl in manual_mapping.items():
            if re.match(pattern, clean_name):
                mapped_name = re.sub(pattern, repl, clean_name)
                matched = True
                break
        
        if mapped_name in diffusers_state_dict:
             dw = diffusers_state_dict[mapped_name]
             # Handle Patch Embedding Reshape: [1536, 16, 1, 2, 2] -> [1536, 64]
             if "patch_embedding" in name and dw.ndim == 5:
                 logger.info(f"Reshaping patch embedding {mapped_name} from {dw.shape} to {param.shape}")
                 dw = dw.flatten(1)
                 
             # Check shape
             if dw.shape != param.shape:
                 logger.warning(f"Shape mismatch for {name} (mapped to {mapped_name}): Ref {param.shape} vs Diffusers {dw.shape}. Skipping.")
                 missing_keys.append(name)
                 continue
             ref_state_dict[clean_name] = dw  # Use clean_name to match state_dict() keys
             used_diffusers_keys.add(mapped_name)  # Track used key
             loaded_count += 1
        else:
             missing_keys.append(f"{name} (mapped to {mapped_name})")

    logger.info(f"Loaded {loaded_count}/{total_count} parameters into Reference Model.")
    if missing_keys:
        logger.warning(f"Missing Keys ({len(missing_keys)}):")
        for k in missing_keys:
            logger.warning(k)
            
    assert loaded_count > 0, "No parameters loaded!"
    
    # Debug: Check if keys in ref_state_dict match model2's state_dict
    model2_state_dict_keys = set(model2.state_dict().keys())
    ref_keys = set(ref_state_dict.keys())
    
    keys_in_ref_not_in_model = ref_keys - model2_state_dict_keys
    keys_in_model_not_in_ref = model2_state_dict_keys - ref_keys
    
    if keys_in_ref_not_in_model:
        logger.warning(f"Keys in ref_state_dict NOT in model2.state_dict ({len(keys_in_ref_not_in_model)}):")
        for k in sorted(list(keys_in_ref_not_in_model))[:10]:
            logger.warning(f"  {k}")
    if keys_in_model_not_in_ref:
        logger.warning(f"Keys in model2.state_dict NOT in ref_state_dict ({len(keys_in_model_not_in_ref)}):")
        for k in sorted(list(keys_in_model_not_in_ref))[:10]:
            logger.warning(f"  {k}")
    
    model2.load_state_dict(ref_state_dict, strict=False)
    
    # Print unused Diffusers keys
    print("Unused Diffusers Keys:", set(diffusers_state_dict.keys()) - used_diffusers_keys)
    unused_diffusers_keys = set(diffusers_state_dict.keys()) - used_diffusers_keys
    if unused_diffusers_keys:
        logger.info(f"Unused Diffusers Keys ({len(unused_diffusers_keys)}):")
        for k in sorted(unused_diffusers_keys):
            logger.info(f"  {k}: {diffusers_state_dict[k].shape}")
    
    model1 = WanTransformer3DModel.from_pretrained(
        TRANSFORMER_PATH, device=device,
        torch_dtype=precision).to(device, dtype=precision).requires_grad_(False)
        
    # Count the number of named parameters (tensors, not elements)
    count1 = len(list(model1.named_parameters()))
    count2 = len(list(model2.named_parameters()))

    print(f"Model 1 (Source) Parameter Count: {count1}")
    print(f"Model 2 (Target) Parameter Count: {count2}")

    if count1 == count2:
        print("✅ Great! Parameter counts match. Structural transfer is possible.")
    else:
        print(f"❌ Mismatch! You have a difference of {abs(count1 - count2)} layers.")
        print("Do NOT use the 'zip' method. You must use Regex mapping.")
    
    # Compare model sizes
    params1 = sum(p.numel() for p in model1.parameters())
    params2 = sum(p.numel() for p in model2.parameters())
    logger.info(f"Model 1 (Diffusers) Total Params: {params1}")
    logger.info(f"Model 2 (Reference) Total Params: {params2}")

    # Specific check for patch_embedding
    pe1 = model1.patch_embedding.weight.flatten()
    pe2 = model2.patch_embedding.weight.flatten()
    logger.info(f"Patch Embedding 1 Sum: {pe1.sum().item()}")
    logger.info(f"Patch Embedding 2 Sum: {pe2.sum().item()}")
    assert_close(pe1.float(), pe2.float(), atol=1e-3, rtol=1e-3, msg="Patch embeddings do not match!")
        
    # copy_weights(model1, model2, mapping) # Replaced by loading from disk

    total_params = sum(p.numel() for p in model1.parameters())
    # Calculate weight sum for model1 (converting to float64 to avoid overflow)
    weight_sum_model1 = sum(
        p.to(torch.float64).sum().item() for p in model1.parameters())
    # Also calculate mean for more stable comparison
    weight_mean_model1 = weight_sum_model1 / total_params
    logger.info("Model 1 weight sum: %s", weight_sum_model1)
    logger.info("Model 1 weight mean: %s", weight_mean_model1)

    # Calculate weight sum for model2 (converting to float64 to avoid overflow)
    total_params_model2 = sum(p.numel() for p in model2.parameters())
    weight_sum_model2 = sum(
        p.to(torch.float64).sum().item() for p in model2.parameters())
    # Also calculate mean for more stable comparison
    weight_mean_model2 = weight_sum_model2 / total_params_model2
    logger.info("Model 2 weight sum: %s", weight_sum_model2)
    logger.info("Model 2 weight mean: %s", weight_mean_model2)

    weight_sum_diff = abs(weight_sum_model1 - weight_sum_model2)
    logger.info("Weight sum difference: %s", weight_sum_diff)
    weight_mean_diff = abs(weight_mean_model1 - weight_mean_model2)
    logger.info("Weight mean difference: %s", weight_mean_diff)

    # Per-parameter comparison to identify which parameters differ
    logger.info("=== Per-Parameter Comparison ===")
    model1_params = {name: p for name, p in model1.named_parameters()}
    model2_params = {name: p for name, p in model2.named_parameters()}
    
    # Build reverse mapping from Diffusers names to Reference names
    differing_params = []
    for ref_name, ref_param in model2_params.items():
        # Find the corresponding Diffusers name using the mapping
        clean_name = ref_name.replace("_checkpoint_wrapped_module.", "")
        mapped_name = clean_name
        for pattern, repl in manual_mapping.items():
            if re.match(pattern, clean_name):
                mapped_name = re.sub(pattern, repl, clean_name)
                break
        
        if mapped_name in model1_params:
            diff1_param = model1_params[mapped_name]
            ref_sum = ref_param.to(torch.float64).sum().item()
            diff1_sum = diff1_param.to(torch.float64).sum().item()
            if abs(ref_sum - diff1_sum) > 1e-3:
                differing_params.append((ref_name, mapped_name, ref_sum, diff1_sum, abs(ref_sum - diff1_sum)))
    
    if differing_params:
        logger.info(f"Found {len(differing_params)} differing parameters:")
        for ref_name, diff_name, ref_sum, diff_sum, diff in sorted(differing_params, key=lambda x: -x[4])[:20]:
            logger.info(f"  {ref_name} -> {diff_name}: ref={ref_sum:.4f}, diff={diff_sum:.4f}, delta={diff:.4f}")
    else:
        logger.info("All parameters match!")

    # Set both models to eval mode
    model1 = model1.eval()
    model2 = model2.eval()

    # Create identical inputs for both models
    batch_size = 1
    seq_len = 30

    # Video latents [B, C, T, H, W]
    hidden_states = torch.randn(batch_size,
                                16,
                                21,
                                160,
                                90,
                                device=device,
                                dtype=precision)

    # Text embeddings [B, L, D] (including global token)
    encoder_hidden_states = torch.randn(batch_size,
                                        seq_len + 1,
                                        4096,
                                        device=device,
                                        dtype=precision)

    # Timestep
    timestep = torch.tensor([500], device=device, dtype=precision)

    forward_batch = ForwardBatch(
        data_type="dummy",
    )

    with torch.amp.autocast('cuda', dtype=precision):
        output1 = model1(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            timestep=timestep,
            return_dict=False,
        )[0]
        with set_forward_context(
                current_timestep=0,
                attn_metadata=None,
                forward_batch=forward_batch,
        ):
            # Adapt input for WanModel (Reference)
            t_input = timestep.unsqueeze(1) if timestep.dim() == 1 else timestep
            output2 = model2(
                x_B_C_T_H_W=hidden_states,
                timesteps_B_T=t_input,
                crossattn_emb=encoder_hidden_states,
            )

    # Check if outputs have the same shape
    assert output1.shape == output2.shape, f"Output shapes don't match: {output1.shape} vs {output2.shape}"
    assert output1.dtype == output2.dtype, f"Output dtype don't match: {output1.dtype} vs {output2.dtype}"

    def compare_tensors(t1, t2, atol=1e-5, rtol=1e-3):
        """
        Compares two tensors for shape, exact equality, and closeness.
        
        Args:
            t1, t2: The tensors to compare.
            atol: Absolute tolerance (default 1e-5).
            rtol: Relative tolerance (default 1e-3).
        """
        print(f"--- Comparing Tensors ---")
        
        # 1. Check Shapes
        if t1.shape != t2.shape:
            print(f"❌ SHAPE MISMATCH: {t1.shape} vs {t2.shape}")
            return
        print(f"✅ Shapes match: {t1.shape}")

        # 2. Check Exact Equality (Strict)
        if torch.equal(t1, t2):
            print(f"✅ Tensors are EXACTLY equal.")
        else:
            print(f"⚠️  Tensors are NOT exactly equal.")

        # 3. Check Closeness (approximate equality for floats)
        # This is critical for float32/float16 where precision errors occur
        is_close = torch.allclose(t1, t2, atol=atol, rtol=rtol)
        
        if is_close:
            print(f"✅ Tensors are NUMERICALLY CLOSE (tol={atol})")
        else:
            print(f"❌ Tensors are DIFFERENT.")
            
            # 4. Debug Statistics (If different, tell me by how much)
            diff = torch.abs(t1 - t2)
            print(f"   - Max diff:  {diff.max().item():.6f}")
            print(f"   - Mean diff: {diff.mean().item():.6f}")
            print(f"   - Mismatched elements: {torch.sum(diff > atol).item()} / {t1.numel()}")
    compare_tensors(output1, output2)

    # Check if outputs are similar (allowing for small numerical differences)
    assert_close(output1, output2, atol=1e-1, rtol=1e-2)
    assert_close(output1, output2, atol=1e-5, rtol=1e-6)

test_wan_transformer()


# The large weight sum difference persists because the two model architectures are not identical, even though they have the same total parameter elements.

# Here's the key insight:

# Diffusers model (
# wanvideo.py
# ) has:

# self_attn_residual_norm with elementwise_affine=True (line 285-286) → contains learnable LayerNorm weight/bias params
# These are post-self-attention norms that exist in the FastVideo implementation
# Reference model (
# wan2pt1.py
# ) has:

# norm3 with elementwise_affine=True → contains learnable LayerNorm weight/bias params
# But this is a pre-cross-attention norm
# The mapping norm3 -> norm2 correctly maps the Reference's pre-cross-attn norm to Diffusers. However, there's no corresponding mapping for Diffusers' self_attn_residual_norm.norm.weight/bias because the Reference model doesn't have an equivalent layer.

# The result:

# All 825 Reference model parameters get loaded from Diffusers weights ✓
# But Diffusers model (Model1) has additional parameters (like self_attn_residual_norm.norm.weight/bias × 30 blocks) that contribute to its higher weight sum
# These extra parameters exist in Model1 but have no equivalent in Model2
# To confirm this, you could add debug code to print:

# len(list(model1.named_parameters())) vs len(list(model2.named_parameters())) — The count of parameter tensors (not elements)
# Which keys in diffusers_state_dict were not used during loading