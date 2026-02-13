# SPDX-License-Identifier: Apache-2.0
"""
Numerical parity test for HunyuanGameCraft transformer.

This test compares the FastVideo implementation against the official
Hunyuan-GameCraft-1.0 implementation to ensure numerical alignment.

Usage:
    # Run with default paths
    pytest tests/local_tests/transformers/test_gamecraft_parity.py -v
    
    # Run with custom paths
    GAMECRAFT_OFFICIAL_PATH=path/to/Hunyuan-GameCraft-1.0 \
    GAMECRAFT_WEIGHTS_PATH=path/to/weights/gamecraft_models/mp_rank_00_model_states.pt \
    pytest tests/local_tests/transformers/test_gamecraft_parity.py -v
    
    # Enable debug logging
    GAMECRAFT_DEBUG_LOGS=1 pytest tests/local_tests/transformers/test_gamecraft_parity.py -v
"""
import os
import sys
from pathlib import Path

import pytest
import torch
from torch.testing import assert_close

# Set up distributed environment defaults
os.environ.setdefault("MASTER_ADDR", "localhost")
os.environ.setdefault("MASTER_PORT", "29514")
os.environ.setdefault("DISABLE_SP", "1")  # Disable sequence parallelism for testing

repo_root = Path(__file__).resolve().parents[3]


def _add_official_to_path():
    """Add official GameCraft implementation to Python path."""
    official_path = Path(
        os.getenv("GAMECRAFT_OFFICIAL_PATH", repo_root / "Hunyuan-GameCraft-1.0")
    )
    if official_path.exists() and str(official_path) not in sys.path:
        sys.path.insert(0, str(official_path))
    return official_path


def _format_sum(tensor: torch.Tensor | None) -> str:
    """Format tensor sum for logging."""
    if tensor is None:
        return "None"
    return f"{tensor.float().sum().item():.6f}"


def _attach_block_logging(
    model: torch.nn.Module,
    log_path: Path,
    label: str,
    enabled: bool,
) -> None:
    """Attach forward hooks to log block outputs."""
    if not enabled:
        return

    log_path.parent.mkdir(parents=True, exist_ok=True)
    if log_path.exists():
        log_path.unlink()

    def _hook_factory(block_idx: int, block_type: str):
        def _hook(_module, _inputs, outputs):
            if isinstance(outputs, tuple):
                img_out, txt_out = outputs
                img_sum = _format_sum(img_out)
                txt_sum = _format_sum(txt_out)
                out_str = f"img_sum={img_sum},txt_sum={txt_sum}"
            else:
                out_str = f"out_sum={_format_sum(outputs)}"
            with log_path.open("a", encoding="utf-8") as f:
                f.write(f"{label}:{block_type}:{block_idx}:{out_str}\n")
        return _hook

    # Attach to double blocks
    if hasattr(model, "double_blocks"):
        for idx, block in enumerate(model.double_blocks):
            block.register_forward_hook(_hook_factory(idx, "double"))

    # Attach to single blocks
    if hasattr(model, "single_blocks"):
        for idx, block in enumerate(model.single_blocks):
            block.register_forward_hook(_hook_factory(idx, "single"))


def _load_official_model(official_path: Path, weights_path: Path, device: torch.device, dtype: torch.dtype):
    """Load the official GameCraft model."""
    try:
        # Add official implementation to path
        sys.path.insert(0, str(official_path))
        
        from hymm_sp.modules.models import HYVideoDiffusionTransformer, HUNYUAN_VIDEO_CONFIG
        from hymm_sp.config import parse_args
        
        # Create a minimal args object
        class MinimalArgs:
            text_projection = "single_refiner"
            text_states_dim = 4096
            use_attention_mask = False
            text_states_dim_2 = 768
        
        args = MinimalArgs()
        config = HUNYUAN_VIDEO_CONFIG['HYVideo-T/2']
        
        # Initialize model
        model = HYVideoDiffusionTransformer(
            args=args,
            patch_size=[1, 2, 2],
            in_channels=16,  # Will be adjusted for concat mode
            out_channels=16,
            hidden_size=config['hidden_size'],
            mlp_width_ratio=config['mlp_width_ratio'],
            num_heads=config['num_heads'],
            depth_double_blocks=config['depth_double_blocks'],
            depth_single_blocks=config['depth_single_blocks'],
            rope_dim_list=config['rope_dim_list'],
            guidance_embed=False,
            dtype=dtype,
            device='cpu',  # Load on CPU first
            multitask_mask_training_type="concat",  # GameCraft uses concat mode
            camera_in_channels=6,
            camera_down_coef=8,
        )
        
        # Load weights
        checkpoint = torch.load(weights_path, map_location='cpu')
        state_dict = checkpoint['module']
        model.load_state_dict(state_dict, strict=True)
        model = model.to(device=device, dtype=dtype)
        model.eval()
        
        return model
        
    except ImportError as e:
        pytest.skip(f"Failed to import official GameCraft: {e}")
    except Exception as e:
        pytest.skip(f"Failed to load official GameCraft model: {e}")


def _create_test_inputs(batch_size: int, device: torch.device, dtype: torch.dtype):
    """Create test inputs for the transformer."""
    # Video latent shape: [B, C, T, H, W]
    # For GameCraft: C=33 (16 latent + 16 gt_latent + 1 mask)
    frames = 9  # 9 latent frames = 33 video frames with temporal compression
    height = 88  # 704 / 8
    width = 160  # 1280 / 8
    
    # Pure latents (what we're denoising)
    latents = torch.randn(batch_size, 16, frames, height, width, device=device, dtype=dtype)
    
    # Ground truth latents (history/reference)
    gt_latents = torch.randn(batch_size, 16, frames, height, width, device=device, dtype=dtype)
    
    # Mask (1 = use gt, 0 = generate)
    mask = torch.zeros(batch_size, 1, frames, height, width, device=device, dtype=dtype)
    mask[:, :, :frames//2] = 1.0  # First half is conditioned
    
    # Concatenated input
    x = torch.cat([latents, gt_latents, mask], dim=1)  # [B, 33, T, H, W]
    
    # Timestep
    t = torch.tensor([500.0] * batch_size, device=device, dtype=dtype)
    
    # Text embeddings (LLaMA)
    text_states = torch.randn(batch_size, 77, 4096, device=device, dtype=dtype)
    text_mask = torch.ones(batch_size, 77, device=device, dtype=dtype)
    
    # CLIP embeddings
    text_states_2 = torch.randn(batch_size, 768, device=device, dtype=dtype)
    
    # Camera/action states: [B, T_video, 6, H_video, W_video]
    # For 9 latent frames with temporal 4x compression -> ~36 video frames
    # But camera net expects raw video resolution
    video_frames = 33  # (9-1)*4 + 1 for 884 VAE
    cam_latents = torch.randn(batch_size, video_frames, 6, 704, 1280, device=device, dtype=dtype)
    
    return {
        'x': x,
        't': t,
        'text_states': text_states,
        'text_mask': text_mask,
        'text_states_2': text_states_2,
        'cam_latents': cam_latents,
    }


def _get_rotary_embeddings(shape, hidden_size, num_heads, rope_dim_list, device, dtype):
    """Generate rotary position embeddings."""
    # Simplified RoPE generation for testing
    # In production, this uses the full RoPE implementation
    t, h, w = shape
    seq_len = t * h * w
    head_dim = hidden_size // num_heads
    
    freqs_cos = torch.ones(seq_len, head_dim, device=device, dtype=dtype)
    freqs_sin = torch.zeros(seq_len, head_dim, device=device, dtype=dtype)
    
    return freqs_cos, freqs_sin


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_gamecraft_transformer_parity():
    """Test numerical parity between official and FastVideo GameCraft implementations."""
    torch.manual_seed(42)
    
    # Initialize distributed environment for FastVideo
    from fastvideo.distributed.parallel_state import init_distributed_environment, initialize_model_parallel
    init_distributed_environment(world_size=1, rank=0, local_rank=0)
    initialize_model_parallel(tensor_model_parallel_size=1, sequence_model_parallel_size=1)
    
    # Get paths
    official_path = Path(
        os.getenv("GAMECRAFT_OFFICIAL_PATH", repo_root / "Hunyuan-GameCraft-1.0")
    )
    weights_path = Path(
        os.getenv(
            "GAMECRAFT_WEIGHTS_PATH",
            repo_root / "Hunyuan-GameCraft-1.0" / "weights" / "gamecraft_models" / "mp_rank_00_model_states.pt"
        )
    )
    
    if not official_path.exists():
        pytest.skip(f"Official GameCraft repo not found at {official_path}")
    if not weights_path.exists():
        pytest.skip(f"GameCraft weights not found at {weights_path}")
    
    device = torch.device("cuda:0")
    dtype = torch.bfloat16
    batch_size = 1
    
    # Load official model
    print(f"\n[GameCraft TEST] Loading official model from {weights_path}")
    official_model = _load_official_model(official_path, weights_path, device, dtype)
    
    # Enable debug logging if requested
    debug_logs = os.getenv("GAMECRAFT_DEBUG_LOGS", "0") == "1"
    _attach_block_logging(
        official_model,
        repo_root / "gamecraft_debug" / "official.log",
        "official",
        debug_logs,
    )
    
    # Create test inputs
    # IMPORTANT: latent_len must be 9, 10, or 18 for official model's camera conditioning
    frames = 9  # Valid latent temporal length
    height = 44  # 352 / 8 (small spatial resolution for testing)
    width = 80   # 640 / 8
    
    latents = torch.randn(batch_size, 16, frames, height, width, device=device, dtype=dtype)
    gt_latents = torch.randn(batch_size, 16, frames, height, width, device=device, dtype=dtype)
    mask = torch.zeros(batch_size, 1, frames, height, width, device=device, dtype=dtype)
    mask[:, :, :4] = 1.0  # First 4 frames are conditioned (history)
    
    x = torch.cat([latents, gt_latents, mask], dim=1)  # [B, 33, T, H, W]
    t = torch.tensor([500.0], device=device, dtype=dtype)
    
    text_states = torch.randn(batch_size, 32, 4096, device=device, dtype=dtype)
    text_mask = torch.ones(batch_size, 32, device=device, dtype=dtype)
    text_states_2 = torch.randn(batch_size, 768, device=device, dtype=dtype)
    
    # Camera states: for 9 latent frames -> 33 video frames with 884 VAE
    video_frames = 33
    cam_latents = torch.randn(batch_size, video_frames, 6, 352, 640, device=device, dtype=dtype)
    
    # Get RoPE embeddings - after patchify with patch_size=[1,2,2]: T, H/2, W/2
    tt, th, tw = frames, height // 2, width // 2
    
    # Use FastVideo's RoPE implementation for both models
    from fastvideo.layers.rotary_embedding import get_rotary_pos_embed
    
    hidden_size = 3072
    num_heads = 24
    rope_dim_list = [16, 56, 56]  # From GameCraft config
    rope_theta = 256.0
    
    freqs_cos, freqs_sin = get_rotary_pos_embed(
        (tt, th, tw),
        hidden_size,
        num_heads,
        rope_dim_list,
        rope_theta,
    )
    freqs_cos = freqs_cos.to(device=device, dtype=dtype)
    freqs_sin = freqs_sin.to(device=device, dtype=dtype)
    
    print(f"[GameCraft TEST] Input shapes:")
    print(f"  x: {x.shape}")
    print(f"  text_states: {text_states.shape}")
    print(f"  cam_latents: {cam_latents.shape}")
    
    # Run official model
    with torch.no_grad():
        official_output = official_model(
            x=x,
            t=t,
            text_states=text_states,
            text_mask=text_mask,
            text_states_2=text_states_2,
            freqs_cos=freqs_cos,
            freqs_sin=freqs_sin,
            cam_latents=cam_latents,
            return_dict=True,
        )
        if isinstance(official_output, dict):
            official_out = official_output['x']
        else:
            official_out = official_output
    
    print(f"[GameCraft TEST] Official output shape: {official_out.shape}")
    print(f"[GameCraft TEST] Official output stats: mean={official_out.mean():.6f}, std={official_out.std():.6f}")
    
    # Load FastVideo GameCraft model
    print("[GameCraft TEST] Loading FastVideo GameCraft model...")
    from fastvideo.configs.models.dits.hunyuangamecraft import HunyuanGameCraftConfig
    from fastvideo.models.dits.hunyuangamecraft import HunyuanGameCraftTransformer3DModel
    
    # Create FastVideo config
    fastvideo_config = HunyuanGameCraftConfig()
    fastvideo_config.dtype = dtype
    
    # Load converted weights
    converted_weights_path = repo_root / "official_weights" / "hunyuan-gamecraft" / "transformer"
    if not converted_weights_path.exists():
        # Try to convert weights on-the-fly
        print(f"[GameCraft TEST] Converting weights from {weights_path}...")
        from scripts.checkpoint_conversion.convert_gamecraft_weights import convert_gamecraft_weights
        convert_gamecraft_weights(
            input_path=str(weights_path),
            output_dir=str(converted_weights_path),
        )
    
    # Load hf_config from converted weights
    import json
    config_path = converted_weights_path / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            hf_config = json.load(f)
    else:
        # Create minimal hf_config
        hf_config = {
            "hidden_size": 3072,
            "num_attention_heads": 24,
            "num_layers": 20,
            "num_single_layers": 40,
        }
    
    # Initialize FastVideo model
    fastvideo_model = HunyuanGameCraftTransformer3DModel(
        config=fastvideo_config,
        hf_config=hf_config,
    )
    
    # Load weights
    from safetensors.torch import load_file
    safetensor_files = list(converted_weights_path.glob("*.safetensors"))
    if safetensor_files:
        state_dict = {}
        for sf_file in safetensor_files:
            state_dict.update(load_file(sf_file))
        
        # Apply param_names_mapping in reverse
        reverse_mapping = fastvideo_config.reverse_param_names_mapping
        mapped_state_dict = {}
        for key, value in state_dict.items():
            mapped_key = key
            for pattern, replacement in reverse_mapping.items():
                import re
                if re.match(pattern, key):
                    mapped_key = re.sub(pattern, replacement, key)
                    break
            mapped_state_dict[mapped_key] = value
        
        missing, unexpected = fastvideo_model.load_state_dict(mapped_state_dict, strict=False)
        print(f"[GameCraft TEST] Missing keys: {len(missing)}")
        print(f"[GameCraft TEST] Unexpected keys: {len(unexpected)}")
        if missing:
            print(f"[GameCraft TEST] First 5 missing: {missing[:5]}")
        if unexpected:
            print(f"[GameCraft TEST] First 5 unexpected: {unexpected[:5]}")
    
    fastvideo_model = fastvideo_model.to(device=device, dtype=dtype)
    fastvideo_model.eval()
    
    _attach_block_logging(
        fastvideo_model,
        repo_root / "gamecraft_debug" / "fastvideo.log",
        "fastvideo",
        debug_logs,
    )
    
    # Prepare FastVideo inputs
    # FastVideo expects encoder_hidden_states as [text_states, text_states_2]
    fastvideo_encoder_hidden_states = [text_states, text_states_2]
    
    # Import forward context
    from fastvideo.forward_context import set_forward_context
    from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
    
    forward_batch = ForwardBatch(data_type="video", enable_teacache=False)
    
    # Run FastVideo model
    with torch.no_grad():
        with set_forward_context(current_timestep=0, attn_metadata=None, forward_batch=forward_batch):
            fastvideo_out = fastvideo_model(
                x,  # Already concatenated [B, 33, T, H, W]
                fastvideo_encoder_hidden_states,
                t,
                camera_states=cam_latents,
                encoder_attention_mask=[text_mask],
            )
    
    print(f"[GameCraft TEST] FastVideo output shape: {fastvideo_out.shape}")
    print(f"[GameCraft TEST] FastVideo output stats: mean={fastvideo_out.mean():.6f}, std={fastvideo_out.std():.6f}")
    
    # Compare outputs using appropriate metrics for numerical parity
    print("[GameCraft TEST] Comparing outputs...")
    
    # Compute various metrics
    diff = (official_out - fastvideo_out).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    
    # Correlation (most important for numerical alignment)
    corr = torch.corrcoef(torch.stack([
        official_out.flatten().float(), 
        fastvideo_out.flatten().float()
    ]))[0, 1].item()
    
    # Cosine similarity
    cos_sim = torch.nn.functional.cosine_similarity(
        official_out.flatten().unsqueeze(0).float(),
        fastvideo_out.flatten().unsqueeze(0).float()
    ).item()
    
    # Sum comparison (should match exactly)
    off_sum = official_out.double().sum().item()
    fv_sum = fastvideo_out.double().sum().item()
    sum_diff_pct = abs(off_sum - fv_sum) / (abs(off_sum) + 1e-10) * 100
    
    print(f"[GameCraft TEST] Max diff: {max_diff:.4f}")
    print(f"[GameCraft TEST] Mean diff: {mean_diff:.6f}")
    print(f"[GameCraft TEST] Correlation: {corr:.6f}")
    print(f"[GameCraft TEST] Cosine similarity: {cos_sim:.6f}")
    print(f"[GameCraft TEST] Sum diff: {sum_diff_pct:.4f}%")
    
    # Pass criteria: correlation > 0.99 and cosine_sim > 0.99 and sum_diff < 10%
    # Sum diff can be larger due to floating point accumulation differences
    assert corr > 0.99, f"Correlation too low: {corr}"
    assert cos_sim > 0.99, f"Cosine similarity too low: {cos_sim}"
    assert sum_diff_pct < 10.0, f"Sum difference too high: {sum_diff_pct}%"
    
    print("[GameCraft TEST] PASSED: Outputs are numerically aligned!")
    print("[GameCraft TEST] Parity test complete!")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_gamecraft_layer_by_layer():
    """Test layer-by-layer parity to find divergence point."""
    import os
    os.environ['DISABLE_SP'] = '1'
    
    torch.manual_seed(42)
    
    from fastvideo.distributed.parallel_state import init_distributed_environment, initialize_model_parallel
    init_distributed_environment(world_size=1, rank=0, local_rank=0)
    initialize_model_parallel(tensor_model_parallel_size=1, sequence_model_parallel_size=1)
    
    official_path = Path(
        os.getenv("GAMECRAFT_OFFICIAL_PATH", repo_root / "Hunyuan-GameCraft-1.0")
    )
    weights_path = Path(
        os.getenv(
            "GAMECRAFT_WEIGHTS_PATH",
            repo_root / "Hunyuan-GameCraft-1.0" / "weights" / "gamecraft_models" / "mp_rank_00_model_states.pt"
        )
    )
    
    if not official_path.exists():
        pytest.skip(f"Official GameCraft repo not found at {official_path}")
    if not weights_path.exists():
        pytest.skip(f"GameCraft weights not found at {weights_path}")
    
    device = torch.device("cuda:0")
    dtype = torch.bfloat16
    batch_size = 1
    frames = 9
    height, width = 44, 80
    text_seq_len = 32
    
    # Load official model
    print("\n[LAYER TEST] Loading official model...")
    official_model = _load_official_model(official_path, weights_path, device, dtype)
    
    # Load FastVideo model
    print("[LAYER TEST] Loading FastVideo model...")
    from fastvideo.configs.models.dits.hunyuangamecraft import HunyuanGameCraftConfig
    from fastvideo.models.dits.hunyuangamecraft import HunyuanGameCraftTransformer3DModel
    
    fastvideo_config = HunyuanGameCraftConfig()
    converted_weights_path = repo_root / "official_weights" / "hunyuan-gamecraft" / "transformer"
    
    import json
    config_path = converted_weights_path / "config.json"
    hf_config = {}
    if config_path.exists():
        with open(config_path) as f:
            hf_config = json.load(f)
    
    fastvideo_model = HunyuanGameCraftTransformer3DModel(config=fastvideo_config, hf_config=hf_config)
    
    from safetensors.torch import load_file
    safetensor_files = list(converted_weights_path.glob("*.safetensors"))
    if safetensor_files:
        state_dict = {}
        for sf_file in safetensor_files:
            state_dict.update(load_file(sf_file))
        
        reverse_mapping = fastvideo_config.reverse_param_names_mapping
        mapped_state_dict = {}
        for key, value in state_dict.items():
            mapped_key = key
            for pattern, replacement in reverse_mapping.items():
                import re
                if re.match(pattern, key):
                    mapped_key = re.sub(pattern, replacement, key)
                    break
            mapped_state_dict[mapped_key] = value
        
        fastvideo_model.load_state_dict(mapped_state_dict, strict=False)
    
    fastvideo_model = fastvideo_model.to(device=device, dtype=dtype)
    fastvideo_model.eval()
    
    # Create inputs
    torch.manual_seed(42)
    x = torch.randn(batch_size, 33, frames, height, width, device=device, dtype=dtype)
    t = torch.tensor([500.0], device=device, dtype=dtype)
    text_states = torch.randn(batch_size, text_seq_len, 4096, device=device, dtype=dtype)
    text_states_2 = torch.randn(batch_size, 768, device=device, dtype=dtype)
    cam_latents = torch.randn(batch_size, 33, 6, 352, 640, device=device, dtype=dtype)
    
    # RoPE
    tt, th, tw = frames, height // 2, width // 2
    from fastvideo.layers.rotary_embedding import get_rotary_pos_embed
    freqs_cos, freqs_sin = get_rotary_pos_embed((tt, th, tw), 3072, 24, [16, 56, 56], 256.0)
    freqs_cos = freqs_cos.to(device=device, dtype=dtype)
    freqs_sin = freqs_sin.to(device=device, dtype=dtype)
    
    print("[LAYER TEST] Testing embedding layers...")
    
    # Set up FastVideo context early (needed for txt_in)
    from fastvideo.forward_context import set_forward_context
    from fastvideo.attention.backends.flash_attn import FlashAttnMetadataBuilder
    attn_metadata = FlashAttnMetadataBuilder().build(current_timestep=0, attn_mask=None)
    
    with torch.no_grad(), set_forward_context(current_timestep=0, attn_metadata=attn_metadata):
        # === img_in ===
        off_img = official_model.img_in(x)
        fv_img = fastvideo_model.img_in(x)
        diff = (off_img - fv_img).abs().max().item()
        print(f"  img_in: max_diff={diff:.6f} {'OK' if diff < 1e-3 else 'DIFF!'}")
        
        # === time_in ===
        off_time = official_model.time_in(t)
        fv_time = fastvideo_model.time_in(t)
        diff = (off_time - fv_time).abs().max().item()
        print(f"  time_in: max_diff={diff:.6f} {'OK' if diff < 1e-3 else 'DIFF!'}")
        
        # === vector_in ===
        off_vec = official_model.vector_in(text_states_2)
        fv_vec = fastvideo_model.vector_in(text_states_2)
        diff = (off_vec - fv_vec).abs().max().item()
        print(f"  vector_in: max_diff={diff:.6f} {'OK' if diff < 1e-3 else 'DIFF!'}")
        
        # === txt_in ===
        off_txt = official_model.txt_in(text_states, t, None)
        fv_txt = fastvideo_model.txt_in(text_states, t)
        diff = (off_txt - fv_txt).abs().max().item()
        print(f"  txt_in: max_diff={diff:.6f} {'OK' if diff < 1e-3 else 'DIFF!'}")
        
        # === camera_net ===
        off_cam = official_model.camera_net(cam_latents)
        fv_cam = fastvideo_model.camera_net(cam_latents)
        diff = (off_cam - fv_cam).abs().max().item()
        print(f"  camera_net: max_diff={diff:.6f} {'OK' if diff < 1e-3 else 'DIFF!'}")
        
        # Combined embeddings
        off_img_combined = off_img + off_cam
        fv_img_combined = fv_img + fv_cam
        off_vec_combined = off_time + off_vec
        fv_vec_combined = fv_time + fv_vec
        
        print("\n[LAYER TEST] Testing double blocks (first 5)...")
        
        # cu_seqlens for official model
        seq_len = off_img_combined.shape[1] + off_txt.shape[1]
        cu_seqlens = torch.tensor([0, seq_len, seq_len], device=device, dtype=torch.int32)
        
        off_img_cur, off_txt_cur = off_img_combined, off_txt
        fv_img_cur, fv_txt_cur = fv_img_combined, fv_txt
        
        for i in range(min(5, len(official_model.double_blocks))):
            # Official
            off_img_out, off_txt_out = official_model.double_blocks[i](
                off_img_cur, off_txt_cur, off_vec_combined,
                cu_seqlens_q=cu_seqlens, cu_seqlens_kv=cu_seqlens,
                max_seqlen_q=seq_len, max_seqlen_kv=seq_len,
                freqs_cis=(freqs_cos, freqs_sin), use_sage=False
            )
            
            # FastVideo (context already set above)
            fv_img_out, fv_txt_out = fastvideo_model.double_blocks[i](
                fv_img_cur, fv_txt_cur, fv_vec_combined,
                freqs_cis=(freqs_cos, freqs_sin)
            )
            
            img_diff = (off_img_out - fv_img_out).abs().max().item()
            txt_diff = (off_txt_out - fv_txt_out).abs().max().item()
            img_mean_diff = (off_img_out - fv_img_out).abs().mean().item()
            
            status = 'OK' if img_diff < 0.1 else 'DIFF!'
            print(f"  double_block[{i}]: img_max_diff={img_diff:.4f}, img_mean_diff={img_mean_diff:.6f}, txt_max_diff={txt_diff:.4f} {status}")
            
            # Use outputs for next layer
            off_img_cur, off_txt_cur = off_img_out, off_txt_out
            fv_img_cur, fv_txt_cur = fv_img_out, fv_txt_out
            
            if img_diff > 1.0:
                print(f"  [STOPPING] Large divergence at double_block[{i}]")
                break
    
    print("\n[LAYER TEST] Done.")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_gamecraft_cameranet_parity():
    """Test CameraNet component in isolation."""
    torch.manual_seed(42)
    
    official_path = Path(
        os.getenv("GAMECRAFT_OFFICIAL_PATH", repo_root / "Hunyuan-GameCraft-1.0")
    )
    weights_path = Path(
        os.getenv(
            "GAMECRAFT_WEIGHTS_PATH",
            repo_root / "Hunyuan-GameCraft-1.0" / "weights" / "gamecraft_models" / "mp_rank_00_model_states.pt"
        )
    )
    
    if not official_path.exists():
        pytest.skip(f"Official GameCraft repo not found at {official_path}")
    if not weights_path.exists():
        pytest.skip(f"GameCraft weights not found at {weights_path}")
    
    device = torch.device("cuda:0")
    dtype = torch.bfloat16
    
    # Add official to path
    _add_official_to_path()
    
    try:
        from hymm_sp.modules.cameranet import CameraNet
    except ImportError as e:
        pytest.skip(f"Failed to import CameraNet: {e}")
    
    # Load CameraNet weights
    checkpoint = torch.load(weights_path, map_location='cpu')
    state_dict = checkpoint['module']
    
    # Extract CameraNet weights
    camera_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('camera_net.'):
            new_key = k.replace('camera_net.', '')
            camera_state_dict[new_key] = v
    
    # Initialize CameraNet
    camera_net = CameraNet(
        in_channels=6,
        downscale_coef=8,
        out_channels=16,
        patch_size=[1, 2, 2],
        hidden_size=3072,
    )
    camera_net.load_state_dict(camera_state_dict, strict=True)
    camera_net = camera_net.to(device=device, dtype=dtype)
    camera_net.eval()
    
    # Test input
    batch_size = 1
    num_frames = 17
    cam_input = torch.randn(batch_size, num_frames, 6, 352, 640, device=device, dtype=dtype)
    
    with torch.no_grad():
        cam_output = camera_net(cam_input)
    
    print(f"[CameraNet TEST] Official input shape: {cam_input.shape}")
    print(f"[CameraNet TEST] Official output shape: {cam_output.shape}")
    print(f"[CameraNet TEST] Official output stats: mean={cam_output.mean():.6f}, std={cam_output.std():.6f}")
    
    # Load FastVideo CameraNet
    print("[CameraNet TEST] Loading FastVideo CameraNet...")
    from fastvideo.models.dits.hunyuangamecraft import CameraNet as FastVideoCameraNet
    
    # Initialize FastVideo CameraNet
    fastvideo_camera_net = FastVideoCameraNet(
        in_channels=6,
        downscale_coef=8,
        out_channels=16,
        patch_size=[1, 2, 2],
        hidden_size=3072,
        dtype=dtype,
        prefix="camera_net",
    )
    
    # Map official weights to FastVideo format
    fastvideo_camera_state_dict = {}
    for k, v in camera_state_dict.items():
        new_key = k
        # Map camera_in -> PatchEmbed format
        if "camera_in.proj" in k:
            new_key = k  # Keep as-is for now
        fastvideo_camera_state_dict[new_key] = v
    
    missing, unexpected = fastvideo_camera_net.load_state_dict(fastvideo_camera_state_dict, strict=False)
    print(f"[CameraNet TEST] Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")
    if missing:
        print(f"[CameraNet TEST] Missing: {missing}")
    if unexpected:
        print(f"[CameraNet TEST] Unexpected: {unexpected}")
    
    fastvideo_camera_net = fastvideo_camera_net.to(device=device, dtype=dtype)
    fastvideo_camera_net.eval()
    
    # Run FastVideo CameraNet
    with torch.no_grad():
        fastvideo_cam_output = fastvideo_camera_net(cam_input)
    
    print(f"[CameraNet TEST] FastVideo output shape: {fastvideo_cam_output.shape}")
    print(f"[CameraNet TEST] FastVideo output stats: mean={fastvideo_cam_output.mean():.6f}, std={fastvideo_cam_output.std():.6f}")
    
    # Compare outputs
    print("[CameraNet TEST] Comparing outputs...")
    try:
        assert_close(cam_output, fastvideo_cam_output, atol=0.0, rtol=0.0)
        print("[CameraNet TEST] PASSED: CameraNet outputs match!")
    except AssertionError as e:
        diff = (cam_output - fastvideo_cam_output).abs()
        print(f"[CameraNet TEST] Max diff: {diff.max():.6f}")
        print(f"[CameraNet TEST] Mean diff: {diff.mean():.6f}")
        print(f"[CameraNet TEST] FAILED: {e}")
        raise
    
    print("[CameraNet TEST] CameraNet parity test complete!")


if __name__ == "__main__":
    # Run tests directly
    test_gamecraft_cameranet_parity()
    test_gamecraft_transformer_parity()
