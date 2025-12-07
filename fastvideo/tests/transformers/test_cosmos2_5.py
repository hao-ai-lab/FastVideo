# SPDX-License-Identifier: Apache-2.0
"""
Test COSMOS 2.5 DiT implementation against reference.
Compares FastVideo's Cosmos25Transformer3DModel with the official MinimalV1LVGDiT from cosmos-predict2.5.
"""
import os
import sys

import pytest
import torch

# Add cosmos-predict2.5 to Python path for loading reference model
TEST_DIR = os.path.dirname(os.path.abspath(__file__))

COSMOS_PREDICT2_5_PATH = os.path.join(TEST_DIR, '..', '..', '..', '..', 'cosmos-predict2.5')
COSMOS_PREDICT2_5_PATH = os.path.normpath(COSMOS_PREDICT2_5_PATH)
if os.path.exists(COSMOS_PREDICT2_5_PATH) and COSMOS_PREDICT2_5_PATH not in sys.path:
    sys.path.insert(0, COSMOS_PREDICT2_5_PATH)

from fastvideo.forward_context import set_forward_context
from fastvideo.logger import init_logger
from fastvideo.utils import maybe_download_model
# Use Cosmos 2.5 specific config
from fastvideo.configs.models.dits.cosmos2_5 import Cosmos25VideoConfig
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch


logger = init_logger(__name__)

# Log the cosmos-predict2.5 path after logger is initialized
if os.path.exists(COSMOS_PREDICT2_5_PATH):
    logger.info(f"cosmos-predict2.5 found at: {COSMOS_PREDICT2_5_PATH}")
else:
    logger.warning(f"cosmos-predict2.5 not found at: {COSMOS_PREDICT2_5_PATH}")

os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "29505"

# COSMOS 2.5 model path - update this based on the actual HuggingFace model ID
# The model has subdirectories: base/pre-trained, base/post-trained, auto/multiview, robot/action-cond
BASE_MODEL_PATH = "nvidia/Cosmos-Predict2.5-2B"
CHECKPOINT_SUBDIR = "base/post-trained"
CHECKPOINT_FILENAME = "81edfebe-bd6a-4039-8c1d-737df1a790bf_ema_bf16.pt"


MODEL_PATH = maybe_download_model(BASE_MODEL_PATH, local_dir=None)
TRANSFORMER_PATH = os.path.join(MODEL_PATH, CHECKPOINT_SUBDIR, "transformer")
if not os.path.exists(TRANSFORMER_PATH):
    # Try without subdirectory
    TRANSFORMER_PATH = os.path.join(MODEL_PATH, "transformer")


def load_reference_cosmos25_model(checkpoint_path: str, device, dtype):
    """
    Load the reference COSMOS 2.5 model from cosmos-predict2.5 repo.
    This assumes the cosmos-predict2.5 repo is available in the Python path.
    """
    try:
        # Try to import from cosmos-predict2.5 repo
        from cosmos_predict2._src.predict2.networks.minimal_v1_lvg_dit import MinimalV1LVGDiT
        
        # COSMOS 2.5 2B model configuration
        model_config = {
            'max_img_h': 240,
            'max_img_w': 240,
            'max_frames': 128,
            'in_channels': 16, 
            'out_channels': 16,
            'patch_spatial': 2,
            'patch_temporal': 1,
            'model_channels': 2048,  # 2B model
            'num_blocks': 28,
            'num_heads': 16,
            'mlp_ratio': 4.0,
            'crossattn_emb_channels': 1024,
            'pos_emb_cls': 'rope3d',
            'pos_emb_learnable': True,
            'pos_emb_interpolation': 'crop',
            'use_adaln_lora': True,
            'adaln_lora_dim': 256,
            'rope_h_extrapolation_ratio': 3.0,
            'rope_w_extrapolation_ratio': 3.0,
            'rope_t_extrapolation_ratio': 1.0,
            'extra_per_block_abs_pos_emb': False,
            'rope_enable_fps_modulation': False,
            'use_crossattn_projection': True,  
            'crossattn_proj_in_channels': 100352, 
            'concat_padding_mask': True,
            'atten_backend': 'torch',
        }
        
        model = MinimalV1LVGDiT(**model_config)
        
        # Load checkpoint if path exists
        if os.path.exists(checkpoint_path):
            logger.info(f"Loading reference model from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            
            # Extract state dict
            if 'state_dict' in checkpoint:
                checkpoint_state = checkpoint['state_dict']
            elif 'model' in checkpoint:
                checkpoint_state = checkpoint['model']
            else:
                checkpoint_state = checkpoint
            
            # Filter to only model parameters (remove training metadata)
            model_state = {k: v for k, v in checkpoint_state.items() 
                          if k.startswith('net.') and 'accum_' not in k}
            
            # Transform checkpoint keys to match model's expected format
            # 1. Strip 'net.' prefix (e.g., 'net.blocks.0.self_attn.*' -> 'blocks.0.self_attn.*')
            # 2. Add '_checkpoint_wrapped_module' after 'blocks.N.' if model expects it
            transformed_state = {}
            
            # First, check what the model expects
            model_state_dict = model.state_dict()
            needs_checkpoint_wrapper = any('_checkpoint_wrapped_module' in k for k in model_state_dict.keys())
            
            for key, value in model_state.items():
                # Strip 'net.' prefix
                if key.startswith('net.'):
                    new_key = key[4:]  # Remove 'net.' prefix
                else:
                    new_key = key
                
                # Add '_checkpoint_wrapped_module' if needed
                if needs_checkpoint_wrapper and new_key.startswith('blocks.'):
                    # Pattern: 'blocks.N.something' -> 'blocks.N._checkpoint_wrapped_module.something'
                    parts = new_key.split('.', 2)
                    if len(parts) >= 3 and parts[0] == 'blocks' and parts[1].isdigit():
                        new_key = f"{parts[0]}.{parts[1]}._checkpoint_wrapped_module.{parts[2]}"
                
                transformed_state[new_key] = value
            
            # Load with strict=False to handle any remaining mismatches
            missing_keys, unexpected_keys = model.load_state_dict(transformed_state, strict=False)
            
            if missing_keys:
                logger.warning(f"Missing keys when loading reference model: {len(missing_keys)} keys")
                # Show all missing keys for debugging
                logger.warning("All missing keys:")
                for k in missing_keys:
                    logger.warning(f"  - {k}")
                # Filter out _extra_state and pos_embedder keys as they're optional
                missing_important = [k for k in missing_keys 
                                    if '_extra_state' not in k and 'pos_embedder' not in k and 'accum_' not in k]
                if missing_important:
                    logger.warning(f"Missing important keys ({len(missing_important)} total):")
                    for k in missing_important[:10]:  # Show first 10
                        logger.warning(f"  - {k}")
                    if len(missing_important) > 10:
                        logger.warning(f"  ... and {len(missing_important) - 10} more")
            
            if unexpected_keys:
                logger.warning(f"Unexpected keys when loading reference model: {len(unexpected_keys)} keys")
                logger.warning("All unexpected keys:")
                for k in unexpected_keys:
                    logger.warning(f"  - {k}")
            
            logger.info(f"Successfully loaded {len(transformed_state)} parameters into reference model")
        else:
            logger.warning(f"Checkpoint path {checkpoint_path} not found, using random weights")
        
        model = model.to(device, dtype=dtype)
        model.eval()
        
        return model
        
    except ImportError as e:
        logger.error(f"Failed to import cosmos-predict2.5: {e}")
        logger.info("Make sure cosmos-predict2.5 is in your Python path")
        return None


@pytest.mark.usefixtures("distributed_setup")
def test_cosmos25_transformer():
    """Test COSMOS 2.5 transformer against reference implementation."""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    precision = torch.bfloat16
    precision_str = "bfloat16"
    
    # Create COSMOS 2.5 specific config
    from fastvideo.configs.models.dits.cosmos2_5 import Cosmos25ArchConfig
    
    arch_config = Cosmos25ArchConfig(
        num_attention_heads=16,
        attention_head_dim=128,  # 2048 / 16
        in_channels=16,
        out_channels=16,
        num_layers=28,
        patch_size=(1, 2, 2),
        max_size=(128, 240, 240),
        rope_scale=(1.0, 3.0, 3.0),  # T, H, W
        text_embed_dim=1024,
        mlp_ratio=4.0,
        adaln_lora_dim=256,
        use_adaln_lora=True,
        concat_padding_mask=True,
        extra_pos_embed_type=None,  
        use_crossattn_projection=True, 
        rope_enable_fps_modulation=False,
        qk_norm="rms_norm",
    )
    
    cosmos25_config = Cosmos25VideoConfig(arch_config=arch_config)
    
    # Create FastVideo model directly (Cosmos 2.5 is not in diffusers format)
    logger.info("Creating FastVideo COSMOS 2.5 model...")
    from fastvideo.models.dits.cosmos2_5 import Cosmos25Transformer3DModel
    
    # Get hf_config from the arch_config for model initialization
    hf_config = {
        'in_channels': arch_config.in_channels,
        'out_channels': arch_config.out_channels,
        'num_attention_heads': arch_config.num_attention_heads,
        'attention_head_dim': arch_config.attention_head_dim,
        'num_layers': arch_config.num_layers,
        'patch_size': arch_config.patch_size,
        'max_size': arch_config.max_size,
        'rope_scale': arch_config.rope_scale,
        'text_embed_dim': arch_config.text_embed_dim,
        'mlp_ratio': arch_config.mlp_ratio,
        'adaln_lora_dim': arch_config.adaln_lora_dim,
        'use_adaln_lora': arch_config.use_adaln_lora,
        'concat_padding_mask': arch_config.concat_padding_mask,
        'extra_pos_embed_type': arch_config.extra_pos_embed_type,
        'use_crossattn_projection': arch_config.use_crossattn_projection,
        'rope_enable_fps_modulation': arch_config.rope_enable_fps_modulation,
        'qk_norm': arch_config.qk_norm,
    }
    
    fastvideo_model = Cosmos25Transformer3DModel(config=cosmos25_config, hf_config=hf_config)
    fastvideo_model = fastvideo_model.to(device, dtype=precision)
    fastvideo_model.eval()
    
    # Construct checkpoint path using relative paths
    checkpoint_file = os.path.join(MODEL_PATH, CHECKPOINT_SUBDIR, CHECKPOINT_FILENAME)
    
    if not os.path.exists(checkpoint_file):
        logger.warning(f"Checkpoint file not found at {checkpoint_file}")
        logger.info("Will test architecture without loading checkpoint weights")
        checkpoint_file = None
    
    # Load checkpoint into FastVideo model using param_names_mapping
    if checkpoint_file:
        logger.info(f"Loading checkpoint into FastVideo model from {checkpoint_file}")
        from fastvideo.models.loader.utils import hf_to_custom_state_dict, get_param_names_mapping
        
        checkpoint = torch.load(checkpoint_file, map_location='cpu')
        
        # Extract state dict (checkpoint might have 'state_dict', 'model', or be the dict itself)
        if 'state_dict' in checkpoint:
            checkpoint_state = checkpoint['state_dict']
        elif 'model' in checkpoint:
            checkpoint_state = checkpoint['model']
        else:
            checkpoint_state = checkpoint
        
        # Filter to only model parameters (remove training metadata like accum_*)
        model_state = {k: v for k, v in checkpoint_state.items() 
                      if k.startswith('net.') and 'accum_' not in k}
        
        # Convert checkpoint keys to FastVideo format using param_names_mapping
        param_names_mapping_fn = get_param_names_mapping(
            cosmos25_config.arch_config.param_names_mapping
        )
        custom_state_dict, reverse_mapping = hf_to_custom_state_dict(
            model_state, param_names_mapping_fn
        )
        
        # Only load keys that exist in the model
        model_param_names = set(fastvideo_model.state_dict().keys())
        filtered_state_dict = {
            k: v.to(device=device, dtype=precision) 
            for k, v in custom_state_dict.items() 
            if k in model_param_names
        }
        
        # Load into FastVideo model
        missing_keys, unexpected_keys = fastvideo_model.load_state_dict(
            filtered_state_dict, strict=False
        )
        
        if missing_keys:
            logger.warning(f"Missing keys when loading checkpoint: {len(missing_keys)} keys")
            # Filter out _extra_state keys as they're optional
            missing_non_extra = [k for k in missing_keys if '_extra_state' not in k]
            if missing_non_extra:
                logger.warning(f"Missing non-extra keys (first 10): {missing_non_extra[:10]}")
        
        if unexpected_keys:
            logger.warning(f"Unexpected keys when loading checkpoint: {len(unexpected_keys)} keys")
        
        logger.info(f"Successfully loaded {len(filtered_state_dict)} parameters into FastVideo model")
    
    # Try to load reference model from the raw checkpoint
    logger.info("Loading reference COSMOS 2.5 model...")
    reference_model = load_reference_cosmos25_model(checkpoint_file, device, precision) if checkpoint_file else None
    
    # Set models to eval mode
    fastvideo_model = fastvideo_model.eval()
    if reference_model is not None:
        reference_model = reference_model.eval()
    
    # Create test inputs
    batch_size = 1
    seq_len = 77  # Typical T5 sequence length
    
    # Video latents [B, C, T, H, W]
    # COSMOS 2.5: 16 channels (VAE latent), no condition mask in input
    hidden_states = torch.randn(
        batch_size,
        16,  # VAE channels only (condition mask added internally)
        1,   # Single frame for image generation (or 16 for video)
        64,  # Height (720p / 8 / 2 patch = 45, use 64 for testing)
        64,  # Width
        device=device,
        dtype=precision
    )
    
    # Condition mask [B, 1, T, H, W] - for video2world conditioning
    condition_mask = torch.zeros(
        batch_size,
        1,
        1,
        64,
        64,
        device=device,
        dtype=precision
    )
    
    # Text embeddings [B, L, D] - Qwen 7B embeddings (100,352 dims)
    # Using 100,352 dimensions to match the crossattn_projection layer
    encoder_hidden_states = torch.randn(
        batch_size,
        seq_len,
        100352,  
        device=device,
        dtype=precision
    )
    
    # Timestep [B, T] - official model expects [B, T] shape with dtype matching model precision
    # For single frame, use [B, 1]
    timestep = torch.full((batch_size, 1), 500.0, device=device, dtype=precision)
    
    # Padding mask [B, H, W] - official model expects NO channel dimension
    # It's added internally via unsqueeze(1) if needed
    padding_mask = torch.ones(
        batch_size,
        64,
        64,
        device=device,
        dtype=precision
    )
    
    # FPS for temporal scaling
    fps = 16
    
    forward_batch = ForwardBatch(
        data_type="dummy",
    )
    
    logger.info("Running inference...")
    
    with torch.no_grad():
        with torch.autocast('cuda', dtype=precision):
            # FastVideo model
            with set_forward_context(
                current_timestep=500,
                attn_metadata=None,
                forward_batch=forward_batch,
            ):
                # FastVideo expects padding_mask in [B, 1, H, W] format
                padding_mask_fv = padding_mask.unsqueeze(1)  # Add channel dimension for FastVideo
                # FastVideo supports both [B] and [B, T] formats - use [B, T] to match official model
                # This ensures each frame gets its own timestep embedding (even if values are the same)
                timestep_fv = timestep  # Already in [B, T] format
                
                output_fv = fastvideo_model(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    timestep=timestep_fv,
                    condition_mask=condition_mask,
                    padding_mask=padding_mask_fv,
                    fps=fps,
                )
            
            # Reference model (if available)
            if reference_model is not None:
                # Prepare input for reference model
                # MinimalV1LVGDiT adds condition mask internally, so pass them separately
                from cosmos_predict2._src.predict2.conditioner import DataType
                
                # Determine data_type based on temporal dimension
                num_frames = hidden_states.shape[2]
                ref_data_type = DataType.VIDEO if num_frames > 1 else DataType.IMAGE
                
                # Reference model expects different input format
                # Pass hidden_states without condition_mask (model concatenates it internally)
                # timestep is already in [B, T] format with correct dtype
                # padding_mask is already in [B, H, W] format (no channel dimension)
                # FPS should be a tensor [B] or scalar
                fps_tensor = torch.tensor([fps], device=device, dtype=precision)
                
                output_ref = reference_model(
                    x_B_C_T_H_W=hidden_states,  # [B, 16, T, H, W] - model will add condition mask
                    timesteps_B_T=timestep,  # Already in [B, T] format
                    crossattn_emb=encoder_hidden_states,
                    condition_video_input_mask_B_C_T_H_W=condition_mask if ref_data_type == DataType.VIDEO else None,
                    fps=fps_tensor,
                    padding_mask=padding_mask,  # [B, H, W] format
                    data_type=ref_data_type,
                )
    
    # Check FastVideo output shape and dtype
    logger.info(f"FastVideo output shape: {output_fv.shape}")
    logger.info(f"FastVideo output dtype: {output_fv.dtype}")
    assert output_fv.shape[0] == batch_size, "Batch size mismatch"
    assert output_fv.shape[1] == 16, "Output channels should be 16"
    assert output_fv.dtype == precision, f"Output dtype mismatch: {output_fv.dtype} vs {precision}"
    
    # Compare with reference if available
    if reference_model is not None:
        logger.info(f"Reference output shape: {output_ref.shape}")
        
        # Check if outputs have the same shape
        assert output_fv.shape == output_ref.shape, \
            f"Output shapes don't match: {output_fv.shape} vs {output_ref.shape}"
        assert output_fv.dtype == output_ref.dtype, \
            f"Output dtype don't match: {output_fv.dtype} vs {output_ref.dtype}"
        
        # Check if outputs are similar
        max_diff = torch.max(torch.abs(output_fv - output_ref))
        mean_diff = torch.mean(torch.abs(output_fv - output_ref))
        relative_diff = mean_diff / (torch.mean(torch.abs(output_ref)) + 1e-8)
        
        logger.info(f"Max difference: {max_diff.item():.6f}")
        logger.info(f"Mean difference: {mean_diff.item():.6f}")
        logger.info(f"Relative difference: {relative_diff.item():.6f}")
        
        
        # Allow for some numerical differences due to implementation details
        assert max_diff < 1e-1, f"Maximum difference too large: {max_diff.item()}"
        assert mean_diff < 1e-2, f"Mean difference too large: {mean_diff.item()}"
        
        logger.info("✓ COSMOS 2.5 FastVideo implementation matches reference!")
    else:
        logger.warning("Reference model not available, skipping comparison")
        logger.info("✓ COSMOS 2.5 FastVideo model runs successfully!")


@pytest.mark.usefixtures("distributed_setup")
def test_cosmos25_transformer_video():
    """Test COSMOS 2.5 transformer with video input (multiple frames)."""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    precision = torch.bfloat16
    precision_str = "bfloat16"
    
    # Create COSMOS 2.5 specific config
    from fastvideo.configs.models.dits.cosmos2_5 import Cosmos25ArchConfig
    
    arch_config = Cosmos25ArchConfig(
        num_attention_heads=16,
        attention_head_dim=128,
        in_channels=16,
        out_channels=16,
        num_layers=28,
        patch_size=(1, 2, 2),
        max_size=(128, 240, 240),
        rope_scale=(1.0, 3.0, 3.0),
        text_embed_dim=1024,
        mlp_ratio=4.0,
        adaln_lora_dim=256,
        use_adaln_lora=True,
        concat_padding_mask=True,
        extra_pos_embed_type=None,
        use_crossattn_projection=True,  # Enable to match official model
        rope_enable_fps_modulation=False,
        qk_norm="rms_norm",
    )
    
    cosmos25_config = Cosmos25VideoConfig(arch_config=arch_config)
    
    # Create FastVideo model directly (Cosmos 2.5 is not in diffusers format)
    logger.info("Creating FastVideo COSMOS 2.5 model for video test...")
    from fastvideo.models.dits.cosmos2_5 import Cosmos25Transformer3DModel
    
    # Get hf_config from the arch_config for model initialization
    hf_config = {
        'in_channels': arch_config.in_channels,
        'out_channels': arch_config.out_channels,
        'num_attention_heads': arch_config.num_attention_heads,
        'attention_head_dim': arch_config.attention_head_dim,
        'num_layers': arch_config.num_layers,
        'patch_size': arch_config.patch_size,
        'max_size': arch_config.max_size,
        'rope_scale': arch_config.rope_scale,
        'text_embed_dim': arch_config.text_embed_dim,
        'mlp_ratio': arch_config.mlp_ratio,
        'adaln_lora_dim': arch_config.adaln_lora_dim,
        'use_adaln_lora': arch_config.use_adaln_lora,
        'concat_padding_mask': arch_config.concat_padding_mask,
        'extra_pos_embed_type': arch_config.extra_pos_embed_type,
        'use_crossattn_projection': arch_config.use_crossattn_projection,
        'rope_enable_fps_modulation': arch_config.rope_enable_fps_modulation,
        'qk_norm': arch_config.qk_norm,
    }
    
    model = Cosmos25Transformer3DModel(config=cosmos25_config, hf_config=hf_config)
    model = model.to(device, dtype=precision)
    model.eval()
    
    # Create video input with multiple frames
    batch_size = 1
    num_frames = 16  # Video with 16 frames
    seq_len = 77
    
    hidden_states = torch.randn(
        batch_size,
        16,
        num_frames,  # Multiple frames
        64,
        64,
        device=device,
        dtype=precision
    )
    
    condition_mask = torch.zeros(
        batch_size,
        1,
        num_frames,
        64,
        64,
        device=device,
        dtype=precision
    )
    # Set first 2 frames as conditioning
    condition_mask[:, :, :2, :, :] = 1.0
    
    encoder_hidden_states = torch.randn(
        batch_size,
        seq_len,
        100352,  # Qwen 7B embedding dimension (matches crossattn_proj input)
        device=device,
        dtype=precision
    )
    
    timestep = torch.tensor([500], device=device, dtype=torch.long)
    
    padding_mask = torch.ones(
        batch_size,
        1,
        64,
        64,
        device=device,
        dtype=precision
    )
    
    fps = 16
    
    forward_batch = ForwardBatch(
        data_type="dummy",
    )
    
    logger.info("Running video inference...")
    
    with torch.no_grad():
        with torch.autocast('cuda', dtype=precision):
            with set_forward_context(
                current_timestep=500,
                attn_metadata=None,
                forward_batch=forward_batch,
            ):
                output = model(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    timestep=timestep,
                    condition_mask=condition_mask,
                    padding_mask=padding_mask,
                    fps=fps,
                )
    
    logger.info(f"Video output shape: {output.shape}")
    logger.info(f"Video output dtype: {output.dtype}")
    
    # Check output shape
    assert output.shape[0] == batch_size, "Batch size mismatch"
    assert output.shape[1] == 16, "Output channels should be 16"
    assert output.shape[2] == num_frames, "Number of frames mismatch"
    assert output.dtype == precision, f"Output dtype mismatch"
    
    logger.info("✓ COSMOS 2.5 video inference successful!")


if __name__ == "__main__":
    # Run tests directly
    test_cosmos25_transformer()
    test_cosmos25_transformer_video()

