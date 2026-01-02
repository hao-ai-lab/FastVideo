# SPDX-License-Identifier: Apache-2.0
"""
Comparison test between FastVideo WanVideo+SLA and TurboDiffusion WanVideo+SLA.

This test loads TurboWan2.1-T2V-1.3B-480P weights into both implementations
and verifies that forward passes produce similar outputs.
"""

import os
import sys

# Set SLA attention backend BEFORE importing FastVideo modules
os.environ["FASTVIDEO_ATTENTION_BACKEND"] = "SLA_ATTN"
os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "29505"

import pytest
import torch
from torch.testing import assert_close

# Add TurboDiffusion to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FASTVIDEO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
TURBODIFFUSION_PATH = os.path.abspath(os.path.join(FASTVIDEO_ROOT, "TurboDiffusion", "turbodiffusion"))
if TURBODIFFUSION_PATH not in sys.path:
    sys.path.insert(0, TURBODIFFUSION_PATH)

from fastvideo.configs.pipelines import PipelineConfig
from fastvideo.forward_context import set_forward_context
from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.logger import init_logger
from fastvideo.models.loader.component_loader import TransformerLoader
from fastvideo.utils import maybe_download_model
from fastvideo.configs.models.dits import WanVideoConfig
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch

logger = init_logger(__name__)

# TurboDiffusion checkpoint
TURBO_MODEL_ID = "TurboDiffusion/TurboWan2.1-T2V-1.3B-480P"
TURBO_LOCAL_DIR = os.path.join("data", TURBO_MODEL_ID)
CHECKPOINT_FILENAME = "TurboWan2.1-T2V-1.3B-480P.pth"

# Base WanVideo model for config reference
BASE_MODEL_PATH = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"


def load_turbodiffusion_model(checkpoint_path: str, device: torch.device, dtype: torch.dtype):
    """
    Load TurboDiffusion WanModel with SLA attention.
    """
    from rcm.networks.wan2pt1 import WanModel as TurboWanModel
    from SLA import SparseLinearAttention as SLA
    from rcm.networks.wan2pt1 import WanSelfAttention
    
    # 1.3B model args
    wan_net_args = dict(
        dim=1536,
        eps=1e-06,
        ffn_dim=8960,
        freq_dim=256,
        in_dim=16,
        num_heads=12,
        num_layers=30,
        out_dim=16,
        text_len=512,
        model_type="t2v",
    )
    
    with torch.device("meta"):
        model = TurboWanModel(**wan_net_args)
    
    # Replace attention with SLA BEFORE loading weights
    sla_count = 0
    for module in model.modules():
        if type(module) is WanSelfAttention:
            module.attn_op.local_attn = SLA(
                head_dim=module.dim // module.num_heads,
                topk=0.1,
                BLKQ=128,
                BLKK=64,
            )
            sla_count += 1
    
    logger.info(f"TurboDiffusion: Created {sla_count} SLA modules")
    
    # Load weights
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    sla_weights = [k for k in state_dict.keys() if 'local_attn.proj_l' in k]
    logger.info(f"Checkpoint contains {len(sla_weights)} SLA weights")
    
    model.load_state_dict(state_dict, assign=True)
    model = model.to(device, dtype=dtype).eval()
    
    return model


def compare_tensors(t1, t2, atol=1e-5, rtol=1e-3):
    """
    Compares two tensors for shape, exact equality, and closeness.
    """
    print(f"--- Comparing Tensors ---")
    
    if t1.shape != t2.shape:
        print(f"❌ SHAPE MISMATCH: {t1.shape} vs {t2.shape}")
        return
    print(f"✅ Shapes match: {t1.shape}")

    if torch.equal(t1, t2):
        print(f"✅ Tensors are EXACTLY equal.")
    else:
        print(f"⚠️  Tensors are NOT exactly equal.")

    is_close = torch.allclose(t1, t2, atol=atol, rtol=rtol)
    
    if is_close:
        print(f"✅ Tensors are NUMERICALLY CLOSE (tol={atol})")
    else:
        print(f"❌ Tensors are DIFFERENT.")
        diff = torch.abs(t1 - t2)
        print(f"   - Max diff:  {diff.max().item():.6f}")
        print(f"   - Mean diff: {diff.mean().item():.6f}")
        print(f"   - Mismatched elements: {torch.sum(diff > atol).item()} / {t1.numel()}")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.usefixtures("distributed_setup")
def test_wanvideo_sla_comparison():
    """
    Test that FastVideo WanTransformer3DModel with SLA_ATTN produces similar 
    outputs to TurboDiffusion WanModel with SLA.
    """
    device = torch.device("cuda:0")
    dtype = torch.bfloat16
    precision_str = "bf16"
    
    # Download TurboDiffusion checkpoint
    turbo_checkpoint_dir = maybe_download_model(TURBO_MODEL_ID, local_dir=TURBO_LOCAL_DIR)
    turbo_checkpoint_path = os.path.join(turbo_checkpoint_dir, CHECKPOINT_FILENAME)
    
    if not os.path.exists(turbo_checkpoint_path):
        pytest.skip(f"Checkpoint not found at {turbo_checkpoint_path}")
    
    # Download base WanVideo model for FastVideo config
    base_model_path = maybe_download_model(
        BASE_MODEL_PATH, 
        local_dir=os.path.join("data", BASE_MODEL_PATH)
    )
    transformer_path = os.path.join(base_model_path, "transformer")
    
    # Load TurboDiffusion model
    logger.info("="*60)
    logger.info("Loading TurboDiffusion model with SLA...")
    turbo_model = load_turbodiffusion_model(turbo_checkpoint_path, device, dtype)

    
    # Load FastVideo model using TransformerLoader (same as test_wanvideo_orig.py)
    logger.info("="*60)
    logger.info("Loading FastVideo WanTransformer3DModel with SLA_ATTN...")
    args = FastVideoArgs(
        model_path=transformer_path,
        dit_cpu_offload=True,
        pipeline_config=PipelineConfig(dit_config=WanVideoConfig(), dit_precision=precision_str)
    )
    args.device = device
    
    loader = TransformerLoader()
    fv_model = loader.load(transformer_path, args).to(dtype=dtype)
    
    # Load TurboDiffusion weights into FastVideo model
    logger.info("Loading TurboDiffusion weights into FastVideo model...")
    turbo_state_dict = torch.load(turbo_checkpoint_path, map_location="cpu", weights_only=True)
    
    # Define weight mapping from TurboDiffusion -> FastVideo
    # TurboDiffusion: blocks.X.self_attn.attn_op.local_attn.proj_l.{weight,bias}
    # FastVideo:      blocks.X.attn1.attn_impl.proj_l.{weight,bias}
    weight_mapping = {
        r"^blocks\.(\d+)\.self_attn\.q\.(.*)$": r"blocks.\1.to_q.\2",
        r"^blocks\.(\d+)\.self_attn\.k\.(.*)$": r"blocks.\1.to_k.\2",
        r"^blocks\.(\d+)\.self_attn\.v\.(.*)$": r"blocks.\1.to_v.\2",
        r"^blocks\.(\d+)\.self_attn\.o\.(.*)$": r"blocks.\1.to_out.\2",
        r"^blocks\.(\d+)\.self_attn\.norm_q\.(.*)$": r"blocks.\1.norm_q.\2",
        r"^blocks\.(\d+)\.self_attn\.norm_k\.(.*)$": r"blocks.\1.norm_k.\2",
        # SLA proj_l weights
        r"^blocks\.(\d+)\.self_attn\.attn_op\.local_attn\.proj_l\.(.*)$": r"blocks.\1.attn1.attn_impl.proj_l.\2",
        # Cross attention
        r"^blocks\.(\d+)\.cross_attn\.q\.(.*)$": r"blocks.\1.attn2.to_q.\2",
        r"^blocks\.(\d+)\.cross_attn\.k\.(.*)$": r"blocks.\1.attn2.to_k.\2",
        r"^blocks\.(\d+)\.cross_attn\.v\.(.*)$": r"blocks.\1.attn2.to_v.\2",
        r"^blocks\.(\d+)\.cross_attn\.o\.(.*)$": r"blocks.\1.attn2.to_out.\2",
        r"^blocks\.(\d+)\.cross_attn\.norm_q\.(.*)$": r"blocks.\1.attn2.norm_q.\2",
        r"^blocks\.(\d+)\.cross_attn\.norm_k\.(.*)$": r"blocks.\1.attn2.norm_k.\2",
        # Norms and FFN
        r"^blocks\.(\d+)\.norm1\.(.*)$": r"blocks.\1.norm1.\2",
        r"^blocks\.(\d+)\.norm3\.(.*)$": r"blocks.\1.self_attn_residual_norm.norm.\2",  # TurboDiff norm3 -> FV self_attn_residual_norm.norm
        r"^blocks\.(\d+)\.norm2\.(.*)$": r"blocks.\1.norm3.\2",  # TurboDiff norm2 -> FV norm3
        r"^blocks\.(\d+)\.ffn\.0\.(.*)$": r"blocks.\1.ffn.fc_in.\2",  # TurboDiff ffn.0 -> FV ffn.fc_in
        r"^blocks\.(\d+)\.ffn\.2\.(.*)$": r"blocks.\1.ffn.fc_out.\2",  # TurboDiff ffn.2 -> FV ffn.fc_out
        r"^blocks\.(\d+)\.modulation$": r"blocks.\1.scale_shift_table",
        r"^patch_embedding\.(.*)$": r"patch_embedding.proj.\1",  # TurboDiff patch_embedding -> FV patch_embedding.proj
        r"^text_embedding\.0\.(.*)$": r"condition_embedder.text_embedder.fc_in.\1",  # TurboDiff text_embedding.0 -> FV fc_in
        r"^text_embedding\.2\.(.*)$": r"condition_embedder.text_embedder.fc_out.\1",  # TurboDiff text_embedding.2 -> FV fc_out
        r"^time_embedding\.0\.(.*)$": r"condition_embedder.time_embedder.mlp.fc_in.\1",  # TurboDiff time_embedding.0 -> FV mlp.fc_in
        r"^time_embedding\.2\.(.*)$": r"condition_embedder.time_embedder.mlp.fc_out.\1",  # TurboDiff time_embedding.2 -> FV mlp.fc_out
        r"^time_projection\.1\.(.*)$": r"condition_embedder.time_modulation.linear.\1",  # TurboDiff time_projection.1 -> FV time_modulation.linear
        # Head
        r"^head\.head\.(.*)$": r"proj_out.\1",
        r"^head\.norm\.(.*)$": r"norm_out.\1",
        r"^head\.modulation$": r"scale_shift_table",
    }

    import re
    fv_model_state = fv_model.state_dict()
    loaded_weights = {}
    sla_loaded = 0
    
    for turbo_key, turbo_tensor in turbo_state_dict.items():
        fv_key = turbo_key
        
        # Apply mapping
        for pattern, replacement in weight_mapping.items():
            if re.match(pattern, turbo_key):
                # print("Matched pattern:", pattern, replacement, turbo_key)
                fv_key = re.sub(pattern, replacement, turbo_key)
                break
        
        # Handle patch embedding reshape: TurboDiff [1536, 64] -> FV [1536, 16, 1, 2, 2]
        if "patch_embedding" in fv_key and fv_key in fv_model_state:
            target_shape = fv_model_state[fv_key].shape
            if turbo_tensor.shape != target_shape:
                # TurboDiff stores as flattened [out_channels, in_channels*patch_size]
                # FV stores as [out_channels, in_channels, patch_t, patch_h, patch_w]
                turbo_tensor = turbo_tensor.view(target_shape)
        
        # Check if key exists in FastVideo model
        if fv_key in fv_model_state:
            if turbo_tensor.shape == fv_model_state[fv_key].shape:
                # print("Successfully loaded:", fv_key, turbo_key)
                loaded_weights[fv_key] = turbo_tensor
                if 'proj_l' in fv_key:
                    sla_loaded += 1
            else:
                print("Shape mismatch for key:", fv_key, turbo_key)
                print("Turbo shape:", turbo_tensor.shape)
                print("FV shape:", fv_model_state[fv_key].shape)

        else:
            print("Key not found in FastVideo model:", fv_key, turbo_key)
    
    logger.info(f"Mapped {len(loaded_weights)} weights (including {sla_loaded} SLA proj_l weights)")

    # Load weights manually to handle DTensor compatibility
    # For DTensors, we need to access the underlying local tensor
    loaded_count = 0
    with torch.no_grad():
        for name, param in fv_model.named_parameters():
            if name in loaded_weights:
                new_weight = loaded_weights[name].to(param.device, dtype=param.dtype)
                # Handle DTensor by accessing the local tensor
                if hasattr(param, '_local_tensor'):
                    param._local_tensor.copy_(new_weight)
                else:
                    param.copy_(new_weight)
                loaded_count += 1
    
    logger.info(f"Successfully loaded {loaded_count} weights into FastVideo model")
    
    # Verify SLA weights were loaded
    sla_count = 0
    for name, param in fv_model.named_parameters():
        if 'proj_l' in name:
            sla_count += 1
    logger.info(f"FastVideo: Found {sla_count} SLA proj_l layers")
    
    # Set both models to eval mode
    turbo_model = turbo_model.eval()
    fv_model = fv_model.eval()
    
    # Create test inputs
    batch_size = 1
    seq_len = 30
    
    # Video latents [B, C, T, H, W]
    hidden_states = torch.randn(
        batch_size, 16, 21, 60, 80,
        device=device, dtype=dtype
    )
    
    # Text embeddings [B, L, D]
    encoder_hidden_states = torch.randn(
        batch_size, seq_len + 1, 4096,
        device=device, dtype=dtype
    )
    
    # Timestep
    timestep = torch.tensor([500], device=device, dtype=dtype)
    
    forward_batch = ForwardBatch(data_type="dummy")
    
    logger.info("="*60)
    logger.info("Running forward passes...")
    with torch.no_grad(), torch.amp.autocast('cuda', dtype=dtype):
        # TurboDiffusion forward
        t_input = timestep.unsqueeze(1)
        turbo_output = turbo_model(
            x_B_C_T_H_W=hidden_states,
            timesteps_B_T=t_input,
            crossattn_emb=encoder_hidden_states,
        )
        
        # FastVideo forward
        with set_forward_context(current_timestep=0, attn_metadata=None, forward_batch=forward_batch):
            fv_output = fv_model(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                timestep=timestep,
            )
    
    # Check outputs
    logger.info("="*60)
    logger.info(f"TurboDiffusion output shape: {turbo_output.shape}")
    logger.info(f"FastVideo output shape: {fv_output.shape}")
    
    assert turbo_output.shape == fv_output.shape, f"Output shapes don't match: {turbo_output.shape} vs {fv_output.shape}"
    logger.info(f"TurboDiffusion dtype: {turbo_output.dtype}, FastVideo dtype: {fv_output.dtype}")
    
    # Convert to same dtype for comparison
    turbo_float = turbo_output.float()
    fv_float = fv_output.float()
    
    compare_tensors(turbo_float, fv_float)

     # Final assertions
    print("RUNNING FIRST ASSERTION:")
    assert_close(turbo_output.float(), fv_output.float(), atol=1e-1, rtol=1e-2)
    print("RUNNING SECOND ASSERTION:")
    assert_close(turbo_output.float(), fv_output.float(), atol=1e-3, rtol=1e-4)
    print("RUNNING THIRD ASSERTION:")
    assert_close(turbo_output.float(), fv_output.float(), atol=1e-6, rtol=1e-7)
    
    # Log success stats
    diff = torch.abs(turbo_float - fv_float)
    logger.info(f"Max diff: {diff.max().item():.4f}, Mean diff: {diff.mean().item():.4f}")
    
    logger.info("="*60)
    logger.info("✅ Test passed: Outputs match!")

if __name__ == "__main__":
    # Initialize distributed environment for standalone execution
    import numpy as np
    from fastvideo.distributed import (
        maybe_init_distributed_environment_and_model_parallel,
        cleanup_dist_env_and_memory
    )
    
    torch.manual_seed(42)
    np.random.seed(42)
    maybe_init_distributed_environment_and_model_parallel(1, 1)
    
    try:
        test_wanvideo_sla_comparison()
    finally:
        cleanup_dist_env_and_memory()
