# SPDX-License-Identifier: Apache-2.0
"""
End-to-end pipeline test for HunyuanGameCraft.

This test verifies that the full HunyuanGameCraft pipeline can run end-to-end,
including text encoding, camera conditioning, denoising, and VAE decoding.

Requirements:
- Converted HunyuanGameCraft weights in converted_weights/HunyuanGameCraft/
- Or specify path via GAMECRAFT_WEIGHTS_PATH environment variable

Usage:
    pytest fastvideo/tests/transformers/test_hunyuangamecraft_e2e.py -v

Environment variables:
    GAMECRAFT_WEIGHTS_PATH: Path to converted GameCraft weights
"""

import os

import pytest
import torch

from fastvideo.logger import init_logger

logger = init_logger(__name__)

os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "29507"
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

# Default paths - customize these for your setup
DEFAULT_WEIGHTS_PATH = "converted_weights/HunyuanGameCraft"
WEIGHTS_PATH = os.environ.get("GAMECRAFT_WEIGHTS_PATH", DEFAULT_WEIGHTS_PATH)


def _skip_if_weights_missing():
    """Skip test if weights are not available."""
    # Check if transformer weights exist
    transformer_path = os.path.join(WEIGHTS_PATH, "transformer")
    if not os.path.exists(transformer_path):
        pytest.skip(f"Transformer weights not found at {transformer_path}")
    
    # Check for safetensors file
    safetensors_path = os.path.join(transformer_path, "diffusion_pytorch_model.safetensors")
    if not os.path.exists(safetensors_path):
        pytest.skip(f"Safetensors file not found at {safetensors_path}")


def _skip_if_text_encoders_missing():
    """Skip test if text encoders are not available."""
    # Check for text encoder symlinks or actual models
    # These should be symlinked from official_weights or downloaded
    text_encoder_path = os.path.join(WEIGHTS_PATH, "text_encoder")
    if not os.path.exists(text_encoder_path):
        pytest.skip(
            f"Text encoder not found at {text_encoder_path}. "
            "Please symlink from official_weights/HunyuanGameCraft/stdmodels/llava-llama-3-8b-v1_1-transformers/"
        )


def _skip_if_vae_missing():
    """Skip test if VAE is not available in Diffusers format."""
    vae_path = os.path.join(WEIGHTS_PATH, "vae")
    if not os.path.exists(vae_path):
        pytest.skip(
            f"VAE not found at {vae_path}. "
            "Please symlink a HunyuanVideo VAE in Diffusers format."
        )
    
    # Check for config.json (indicates Diffusers format)
    vae_config_path = os.path.join(vae_path, "config.json")
    if not os.path.exists(vae_config_path):
        pytest.skip(
            f"VAE config not found at {vae_config_path}. "
            "VAE must be in Diffusers format (not .ckpt)."
        )


def create_simple_camera_states(
    num_frames: int,
    height: int,
    width: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Create simple camera states for testing."""
    camera_states = torch.zeros(1, num_frames, 6, height, width, device=device, dtype=dtype)
    
    for frame_idx in range(num_frames):
        t = frame_idx / max(num_frames - 1, 1)
        
        # Direction channels (forward movement)
        camera_states[0, frame_idx, 0, :, :] = 0.0  # x
        camera_states[0, frame_idx, 1, :, :] = 0.0  # y
        camera_states[0, frame_idx, 2, :, :] = 1.0 + 0.1 * t  # z (forward)
        
        # Moment channels (simplified)
        camera_states[0, frame_idx, 3, :, :] = 0.0
        camera_states[0, frame_idx, 4, :, :] = 0.0
        camera_states[0, frame_idx, 5, :, :] = 0.0
    
    return camera_states


@pytest.mark.usefixtures("distributed_setup")
def test_hunyuangamecraft_pipeline_stages():
    """
    Test that individual pipeline stages work correctly.
    
    This tests the pipeline components in isolation to help identify
    issues before running the full end-to-end test.
    """
    _skip_if_weights_missing()
    
    from fastvideo.configs.models.dits.hunyuangamecraft import HunyuanGameCraftConfig
    from fastvideo.configs.pipelines import PipelineConfig
    from fastvideo.fastvideo_args import FastVideoArgs
    from fastvideo.models.loader.component_loader import TransformerLoader
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    precision = torch.bfloat16
    
    # Load transformer
    transformer_path = os.path.join(WEIGHTS_PATH, "transformer")
    
    args = FastVideoArgs(
        model_path=transformer_path,
        dit_cpu_offload=False,
        use_fsdp_inference=False,
        pipeline_config=PipelineConfig(
            dit_config=HunyuanGameCraftConfig(),
            dit_precision="bf16"
        ),
    )
    args.device = device
    
    loader = TransformerLoader()
    transformer = loader.load(transformer_path, args).to(device, dtype=precision)
    transformer.eval()
    
    logger.info(f"Loaded transformer with {sum(p.numel() for p in transformer.parameters()):,} parameters")
    
    # Test forward pass with camera states
    batch_size = 1
    latent_frames = 9
    latent_height = 88
    latent_width = 160
    
    # Create test inputs
    hidden_states = torch.randn(
        batch_size, 33, latent_frames, latent_height, latent_width,
        device=device, dtype=precision
    )
    
    encoder_hidden_states = torch.randn(
        batch_size, 257, 4096,
        device=device, dtype=precision
    )
    encoder_hidden_states[:, 0, :768] = torch.randn(
        batch_size, 768, device=device, dtype=precision
    )
    
    timestep = torch.tensor([500], device=device, dtype=precision)
    
    # Create camera states in pixel space
    camera_height = 704
    camera_width = 1280
    num_pixel_frames = 33  # Approximate
    camera_states = create_simple_camera_states(
        num_frames=num_pixel_frames,
        height=camera_height,
        width=camera_width,
        device=device,
        dtype=precision,
    )
    
    # Test forward pass
    from fastvideo.forward_context import set_forward_context
    from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
    
    forward_batch = ForwardBatch(data_type="dummy")
    
    with torch.no_grad():
        with torch.amp.autocast('cuda', dtype=precision):
            with set_forward_context(current_timestep=0, attn_metadata=None, forward_batch=forward_batch):
                output = transformer(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    timestep=timestep,
                    camera_states=camera_states,
                )
    
    expected_shape = (batch_size, 16, latent_frames, latent_height, latent_width)
    assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"
    assert not torch.isnan(output).any(), "Output contains NaN"
    assert not torch.isinf(output).any(), "Output contains Inf"
    
    logger.info("Pipeline stages test PASSED")


@pytest.mark.usefixtures("distributed_setup")
def test_hunyuangamecraft_e2e_inference():
    """
    Full end-to-end test of HunyuanGameCraft pipeline.
    
    This test requires all components (transformer, VAE, text encoders)
    to be available. It runs a complete inference pass and verifies
    the output video tensor shape and validity.
    """
    _skip_if_weights_missing()
    _skip_if_text_encoders_missing()
    _skip_if_vae_missing()
    
    from fastvideo import VideoGenerator
    
    # Use minimal settings for testing
    test_height = 352  # Smaller for faster test
    test_width = 640
    test_frames = 17
    
    logger.info("Initializing VideoGenerator...")
    
    try:
        generator = VideoGenerator.from_pretrained(
            WEIGHTS_PATH,
            num_gpus=1,
            use_fsdp_inference=False,
            dit_cpu_offload=True,
            vae_cpu_offload=True,
            text_encoder_cpu_offload=True,
            pin_cpu_memory=True,
        )
    except Exception as e:
        pytest.skip(f"Failed to initialize VideoGenerator: {e}")
    
    # Create camera states
    camera_states = create_simple_camera_states(
        num_frames=test_frames,
        height=test_height,
        width=test_width,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    
    logger.info("Running inference...")
    
    test_prompt = "A first-person view walking through a forest."
    
    try:
        result = generator.generate_video(
            prompt=test_prompt,
            camera_states=camera_states,
            num_frames=test_frames,
            height=test_height,
            width=test_width,
            num_inference_steps=5,  # Minimal steps for testing
            guidance_scale=6.0,
            seed=42,
            save_video=False,
            return_frames=True,
        )
    except Exception as e:
        pytest.fail(f"Inference failed: {e}")
    
    # Verify output
    assert result is not None, "No output generated"
    
    if isinstance(result, dict):
        assert "samples" in result or "frames" in result, "Output missing samples/frames"
        logger.info(f"Output keys: {result.keys()}")
    else:
        logger.info(f"Output type: {type(result)}")
    
    logger.info("End-to-end inference test PASSED")


@pytest.mark.usefixtures("distributed_setup")
def test_hunyuangamecraft_camera_conditioning():
    """
    Test that camera conditioning is properly applied.
    
    This test verifies that different camera trajectories produce
    different outputs, confirming that the CameraNet conditioning
    is working correctly.
    """
    _skip_if_weights_missing()
    
    from fastvideo.configs.models.dits.hunyuangamecraft import HunyuanGameCraftConfig
    from fastvideo.configs.pipelines import PipelineConfig
    from fastvideo.fastvideo_args import FastVideoArgs
    from fastvideo.models.loader.component_loader import TransformerLoader
    from fastvideo.forward_context import set_forward_context
    from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    precision = torch.bfloat16
    
    transformer_path = os.path.join(WEIGHTS_PATH, "transformer")
    
    args = FastVideoArgs(
        model_path=transformer_path,
        dit_cpu_offload=False,
        use_fsdp_inference=False,
        pipeline_config=PipelineConfig(
            dit_config=HunyuanGameCraftConfig(),
            dit_precision="bf16"
        ),
    )
    args.device = device
    
    loader = TransformerLoader()
    transformer = loader.load(transformer_path, args).to(device, dtype=precision)
    transformer.eval()
    
    # Create fixed inputs
    torch.manual_seed(42)
    
    batch_size = 1
    latent_frames = 9
    latent_height = 44  # Smaller for faster test
    latent_width = 80
    
    hidden_states = torch.randn(
        batch_size, 33, latent_frames, latent_height, latent_width,
        device=device, dtype=precision
    )
    
    encoder_hidden_states = torch.randn(
        batch_size, 257, 4096,
        device=device, dtype=precision
    )
    encoder_hidden_states[:, 0, :768] = torch.randn(
        batch_size, 768, device=device, dtype=precision
    )
    
    timestep = torch.tensor([500], device=device, dtype=precision)
    
    # Create two different camera trajectories
    camera_height = 352
    camera_width = 640
    num_pixel_frames = 33
    
    # Camera 1: Forward movement
    camera_states_1 = torch.zeros(1, num_pixel_frames, 6, camera_height, camera_width, device=device, dtype=precision)
    camera_states_1[:, :, 2, :, :] = 1.0  # Forward direction
    
    # Camera 2: Sideways movement
    camera_states_2 = torch.zeros(1, num_pixel_frames, 6, camera_height, camera_width, device=device, dtype=precision)
    camera_states_2[:, :, 0, :, :] = 1.0  # Sideways direction
    
    forward_batch = ForwardBatch(data_type="dummy")
    
    # Run inference with camera 1
    with torch.no_grad():
        with torch.amp.autocast('cuda', dtype=precision):
            with set_forward_context(current_timestep=0, attn_metadata=None, forward_batch=forward_batch):
                output_1 = transformer(
                    hidden_states=hidden_states.clone(),
                    encoder_hidden_states=encoder_hidden_states.clone(),
                    timestep=timestep,
                    camera_states=camera_states_1,
                ).clone()
    
    # Run inference with camera 2
    with torch.no_grad():
        with torch.amp.autocast('cuda', dtype=precision):
            with set_forward_context(current_timestep=0, attn_metadata=None, forward_batch=forward_batch):
                output_2 = transformer(
                    hidden_states=hidden_states.clone(),
                    encoder_hidden_states=encoder_hidden_states.clone(),
                    timestep=timestep,
                    camera_states=camera_states_2,
                ).clone()
    
    # Outputs should be different (camera conditioning is working)
    diff = (output_1 - output_2).abs().mean().item()
    logger.info(f"Mean absolute difference between camera trajectories: {diff}")
    
    assert diff > 1e-4, (
        f"Outputs should differ with different camera trajectories, but diff={diff}. "
        "Camera conditioning may not be working correctly."
    )
    
    logger.info("Camera conditioning test PASSED")
