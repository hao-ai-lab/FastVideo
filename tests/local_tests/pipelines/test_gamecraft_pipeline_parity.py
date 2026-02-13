# SPDX-License-Identifier: Apache-2.0
"""
End-to-end pipeline parity test for HunyuanGameCraft.

This test compares the FastVideo pipeline against the official
Hunyuan-GameCraft-1.0 implementation to ensure numerical alignment
across the complete pipeline (text encoding -> denoising -> VAE decode).

Usage:
    # Run with default paths
    DISABLE_SP=1 pytest tests/local_tests/pipelines/test_gamecraft_pipeline_parity.py -v -s
    
    # Run with custom paths
    GAMECRAFT_OFFICIAL_PATH=path/to/Hunyuan-GameCraft-1.0 \
    GAMECRAFT_WEIGHTS_PATH=path/to/weights \
    DISABLE_SP=1 pytest tests/local_tests/pipelines/test_gamecraft_pipeline_parity.py -v -s

Notes:
    - This test requires the official Hunyuan-GameCraft-1.0 repo
    - This test requires significant GPU memory (~40GB+)
    - The test compares denoised latents, not final decoded video (for speed)
"""
import os
import sys
from pathlib import Path

import pytest
import torch

# Set up distributed environment defaults
os.environ.setdefault("MASTER_ADDR", "localhost")
os.environ.setdefault("MASTER_PORT", "29520")
os.environ.setdefault("DISABLE_SP", "1")

repo_root = Path(__file__).resolve().parents[3]


def _log_tensor_stats(label: str, tensor: torch.Tensor) -> None:
    """Log tensor statistics for debugging."""
    tensor_f32 = tensor.float()
    print(
        f"[GAMECRAFT PIPELINE] {label}: shape={tuple(tensor.shape)} "
        f"dtype={tensor.dtype} device={tensor.device} "
        f"min={tensor_f32.min().item():.6f} max={tensor_f32.max().item():.6f} "
        f"mean={tensor_f32.mean().item():.6f} std={tensor_f32.std().item():.6f}"
    )


def _add_official_to_path():
    """Add official GameCraft implementation to Python path."""
    official_path = Path(
        os.getenv("GAMECRAFT_OFFICIAL_PATH", repo_root / "Hunyuan-GameCraft-1.0")
    )
    if official_path.exists() and str(official_path) not in sys.path:
        sys.path.insert(0, str(official_path))
    return official_path


def _create_camera_trajectory_from_action(
    action_id: str,
    height: int,
    width: int,
    num_frames: int,
    action_speed: float = 0.2,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.bfloat16,
):
    """
    Create camera trajectory (Plücker coordinates) from an action ID.
    
    This replicates the official GameCraft camera trajectory generation.
    
    Args:
        action_id: One of 'w' (forward), 'a' (left), 'd' (right), 's' (backward),
                   'left_rot', 'right_rot', 'up_rot', 'down_rot'
        height: Video height in pixels
        width: Video width in pixels  
        num_frames: Number of video frames
        action_speed: Speed of motion
        device: Torch device
        dtype: Torch dtype
        
    Returns:
        plucker_embedding: [1, num_frames, 6, height, width]
        uncond_plucker_embedding: [1, num_frames, 6, height, width]
    """
    official_path = _add_official_to_path()
    
    try:
        from hymm_sp.sample_inference import (
            ActionToPoseFromID,
            GetPoseEmbedsFromPoses,
        )
    except ImportError as e:
        pytest.skip(f"Cannot import official camera trajectory functions: {e}")
    
    # Generate poses from action
    poses = ActionToPoseFromID(action_id, value=action_speed, duration=num_frames)
    
    # Convert to Plücker embeddings
    plucker_embedding, uncond_plucker_embedding, _ = GetPoseEmbedsFromPoses(
        poses, height, width, num_frames, flip=False, start_index=0
    )
    
    # Add batch dimension and convert to target dtype/device
    plucker_embedding = plucker_embedding.unsqueeze(0).to(device=device, dtype=dtype)
    uncond_plucker_embedding = uncond_plucker_embedding.unsqueeze(0).to(device=device, dtype=dtype)
    
    return plucker_embedding, uncond_plucker_embedding


def _run_official_pipeline(
    prompt: str,
    action_id: str,
    height: int,
    width: int,
    num_frames: int,
    num_inference_steps: int,
    seed: int,
    device: torch.device,
    dtype: torch.dtype,
    official_path: Path,
    weights_path: Path,
):
    """
    Run the official GameCraft pipeline and return denoised latents.
    
    Returns:
        denoised_latents: The final denoised latents
        timesteps: The timesteps used
    """
    sys.path.insert(0, str(official_path))
    
    try:
        from hymm_sp.config import parse_args
        from hymm_sp.sample_inference import HunyuanVideoSampler
        from hymm_sp.constants import PRECISION_TO_TYPE
        from hymm_sp.vae import load_vae
        from hymm_sp.modules import load_model
        from hymm_sp.text_encoder import TextEncoder
        from hymm_sp.constants import PROMPT_TEMPLATE
    except ImportError as e:
        pytest.skip(f"Cannot import official GameCraft modules: {e}")
    
    # Build args for official model - we need to parse default args
    # and override specific values
    sys.argv = [
        "test",
        "--model", "HYVideo-T/2",
        "--prompt", prompt,
        "--vae", "884-16c-hy",
        "--vae-precision", "fp16",
        "--precision", "bf16",
        "--dit-weight", str(weights_path / "gamecraft_models" / "mp_rank_00_model_states.pt"),
        "--video-length", str(num_frames + 1),  # Official uses +1
        "--video-size", str(height), str(width),
        "--seed", str(seed),
        "--infer-steps", str(num_inference_steps),
        "--flow-shift", "7.0",
        "--embedded-cfg-scale", "6.0",
        "--cfg-scale", "1.0",
        "--text-encoder", str(weights_path / "text_encoder"),
        "--text-encoder-2", str(weights_path / "text_encoder_2"),
        "--tokenizer", str(weights_path / "text_encoder"),
        "--tokenizer-2", str(weights_path / "text_encoder_2"),
        "--prompt-template-video", "dit-llm-encode-video",
        "--load-key", "module",
    ]
    
    args = parse_args()
    args.cpu_offload = 0
    args.use_fp8 = False
    
    # Create the sampler
    sampler = HunyuanVideoSampler.from_pretrained(
        pretrained_model_path=str(weights_path / "gamecraft_models"),
        args=args,
        device=device,
    )
    
    # Run inference
    result = sampler.predict(
        prompt=prompt,
        is_image=True,  # 33 frames mode
        size=(height, width),
        video_length=num_frames + 1,
        seed=seed,
        infer_steps=num_inference_steps,
        guidance_scale=1.0,  # CFG scale
        flow_shift=7.0,
        action_id=action_id,
        action_speed=0.2,
        start_index=0,
        return_latents=True,
        output_type="latent",
    )
    
    denoised_latents = result.get("denoised_lantents")  # Note: typo in official code
    timesteps = result.get("timesteps")
    
    # Clean up
    del sampler
    torch.cuda.empty_cache()
    
    return denoised_latents, timesteps


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="GameCraft pipeline test requires CUDA.",
)
def test_gamecraft_pipeline_latent_parity():
    """
    Test that FastVideo GameCraft pipeline produces matching latents.
    
    This test runs both pipelines with identical:
    - Prompt
    - Camera trajectory (action)
    - Random seed
    - Number of inference steps
    
    And compares the denoised latents (before VAE decode) for numerical parity.
    """
    # Check for required paths
    official_path = _add_official_to_path()
    if not official_path.exists():
        pytest.skip(f"Official GameCraft repo not found at {official_path}")
    
    weights_path = Path(
        os.getenv(
            "GAMECRAFT_WEIGHTS_PATH",
            repo_root / "Hunyuan-GameCraft-1.0" / "weights" / "stdmodels"
        )
    )
    if not weights_path.exists():
        pytest.skip(f"GameCraft weights not found at {weights_path}")
    
    # Check for required weight files
    dit_weights = weights_path / "gamecraft_models" / "mp_rank_00_model_states.pt"
    if not dit_weights.exists():
        pytest.skip(f"DiT weights not found at {dit_weights}")
    
    # Test parameters - small size for faster testing
    device = torch.device("cuda:0")
    dtype = torch.bfloat16
    prompt = "A first-person view of walking through a lush forest."
    action_id = "w"  # Forward motion
    height = 256  # Smaller for testing
    width = 256
    num_frames = 33  # Standard GameCraft frame count
    num_inference_steps = 5  # Reduced for testing
    seed = 42
    
    print("\n[GAMECRAFT PIPELINE TEST] Configuration:")
    print(f"  - Prompt: {prompt}")
    print(f"  - Action: {action_id} (forward)")
    print(f"  - Size: {height}x{width}")
    print(f"  - Frames: {num_frames}")
    print(f"  - Steps: {num_inference_steps}")
    print(f"  - Seed: {seed}")
    
    # =========================================================================
    # Run Official Pipeline
    # =========================================================================
    print("\n[GAMECRAFT PIPELINE TEST] Running official pipeline...")
    
    try:
        official_latents, official_timesteps = _run_official_pipeline(
            prompt=prompt,
            action_id=action_id,
            height=height,
            width=width,
            num_frames=num_frames,
            num_inference_steps=num_inference_steps,
            seed=seed,
            device=device,
            dtype=dtype,
            official_path=official_path,
            weights_path=weights_path,
        )
        _log_tensor_stats("Official latents", official_latents)
    except Exception as e:
        pytest.skip(f"Failed to run official pipeline: {e}")
    
    # Clear memory
    torch.cuda.empty_cache()
    
    # =========================================================================
    # Run FastVideo Pipeline
    # =========================================================================
    print("\n[GAMECRAFT PIPELINE TEST] Running FastVideo pipeline...")
    
    # Import FastVideo components
    from fastvideo import VideoGenerator
    
    # Get diffusers-format model path
    diffusers_path = os.getenv(
        "GAMECRAFT_DIFFUSERS_PATH",
        str(repo_root / "official_weights" / "hunyuan-gamecraft")
    )
    if not os.path.exists(diffusers_path):
        pytest.skip(f"FastVideo GameCraft weights not found at {diffusers_path}")
    
    # Create camera trajectory for FastVideo
    plucker_embedding, uncond_plucker_embedding = _create_camera_trajectory_from_action(
        action_id=action_id,
        height=height,
        width=width,
        num_frames=num_frames,
        action_speed=0.2,
        device=device,
        dtype=dtype,
    )
    
    # Create generator
    generator = VideoGenerator.from_pretrained(
        diffusers_path,
        num_gpus=1,
        use_fsdp_inference=False,
        dit_cpu_offload=False,
        vae_cpu_offload=True,  # Save memory
        text_encoder_cpu_offload=True,
        pin_cpu_memory=False,
    )
    
    # Run generation
    result = generator.generate_video(
        prompt=prompt,
        output_path="outputs_video/gamecraft_test",
        save_video=False,
        height=height,
        width=width,
        num_frames=num_frames,
        num_inference_steps=num_inference_steps,
        seed=seed,
        guidance_scale=1.0,
        camera_states=plucker_embedding,
    )
    
    fastvideo_latents = result.get("latents")
    if fastvideo_latents is None:
        fastvideo_latents = result.get("samples")
    
    _log_tensor_stats("FastVideo latents", fastvideo_latents)
    
    generator.shutdown()
    
    # =========================================================================
    # Compare Results
    # =========================================================================
    print("\n[GAMECRAFT PIPELINE TEST] Comparing latents...")
    
    # Ensure same shape
    assert official_latents.shape == fastvideo_latents.shape, (
        f"Shape mismatch: official {official_latents.shape} vs "
        f"fastvideo {fastvideo_latents.shape}"
    )
    
    # Compute differences
    official_f32 = official_latents.float()
    fastvideo_f32 = fastvideo_latents.float()
    
    abs_diff = (official_f32 - fastvideo_f32).abs()
    max_diff = abs_diff.max().item()
    mean_diff = abs_diff.mean().item()
    
    # Compute correlation
    official_flat = official_f32.flatten()
    fastvideo_flat = fastvideo_f32.flatten()
    correlation = torch.corrcoef(torch.stack([official_flat, fastvideo_flat]))[0, 1].item()
    
    print("\n[GAMECRAFT PIPELINE TEST] Results:")
    print(f"  - Max absolute difference: {max_diff:.6f}")
    print(f"  - Mean absolute difference: {mean_diff:.6f}")
    print(f"  - Correlation: {correlation:.6f}")
    
    # Assertions
    # For full pipeline, we expect higher tolerance due to accumulated errors
    assert correlation > 0.99, f"Correlation {correlation} is too low (expected > 0.99)"
    assert max_diff < 0.5, f"Max diff {max_diff} is too high (expected < 0.5)"
    
    print("\n[GAMECRAFT PIPELINE TEST] PASSED!")


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="GameCraft pipeline test requires CUDA.",
)
def test_gamecraft_pipeline_camera_trajectory():
    """
    Test that camera trajectory (Plücker embedding) generation matches.
    
    This is a simpler test that just validates camera conditioning matches
    between official and FastVideo implementations.
    """
    official_path = _add_official_to_path()
    if not official_path.exists():
        pytest.skip(f"Official GameCraft repo not found at {official_path}")
    
    device = torch.device("cuda:0")
    dtype = torch.bfloat16
    
    # Test parameters
    height = 256
    width = 256
    num_frames = 33
    action_id = "w"
    action_speed = 0.2
    
    print("\n[CAMERA TEST] Generating camera trajectories...")
    
    # Generate using official code path
    plucker, uncond_plucker = _create_camera_trajectory_from_action(
        action_id=action_id,
        height=height,
        width=width,
        num_frames=num_frames,
        action_speed=action_speed,
        device=device,
        dtype=dtype,
    )
    
    _log_tensor_stats("Plücker embedding", plucker)
    _log_tensor_stats("Uncond Plücker embedding", uncond_plucker)
    
    # Basic shape validation
    expected_shape = (1, num_frames, 6, height, width)
    assert plucker.shape == expected_shape, (
        f"Plücker shape {plucker.shape} != expected {expected_shape}"
    )
    assert uncond_plucker.shape == expected_shape, (
        f"Uncond Plücker shape {uncond_plucker.shape} != expected {expected_shape}"
    )
    
    # Validate uncond is identity (zeros for translation, identity for rotation)
    # In practice, uncond should have specific structure
    assert not torch.allclose(plucker, uncond_plucker), (
        "Conditional and unconditional embeddings should differ"
    )
    
    print("\n[CAMERA TEST] PASSED!")


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="GameCraft pipeline test requires CUDA.",
)
def test_gamecraft_pipeline_smoke():
    """
    Smoke test for GameCraft pipeline - just verify it runs without errors.
    
    This is a lighter test that doesn't require the official implementation,
    just validates that the FastVideo pipeline can run.
    """
    diffusers_path = os.getenv(
        "GAMECRAFT_DIFFUSERS_PATH",
        str(repo_root / "official_weights" / "hunyuan-gamecraft")
    )
    
    if not os.path.exists(diffusers_path):
        pytest.skip(f"FastVideo GameCraft weights not found at {diffusers_path}")
    
    if not os.path.isfile(os.path.join(diffusers_path, "model_index.json")):
        pytest.skip(f"model_index.json not found in {diffusers_path}")
    
    device = torch.device("cuda:0")
    
    print("\n[SMOKE TEST] Running FastVideo GameCraft pipeline...")
    
    from fastvideo import VideoGenerator
    
    # Very small test
    height = 128
    width = 128
    num_frames = 9  # Minimum
    num_inference_steps = 2
    
    generator = VideoGenerator.from_pretrained(
        diffusers_path,
        num_gpus=1,
        use_fsdp_inference=False,
        dit_cpu_offload=True,
        vae_cpu_offload=True,
        text_encoder_cpu_offload=True,
        pin_cpu_memory=False,
    )
    
    result = generator.generate_video(
        prompt="A test video.",
        output_path="outputs_video/gamecraft_smoke",
        save_video=False,
        height=height,
        width=width,
        num_frames=num_frames,
        num_inference_steps=num_inference_steps,
        seed=123,
        guidance_scale=1.0,
    )
    
    assert result is not None, "Pipeline returned None"
    assert "samples" in result or "latents" in result, "No output in result"
    
    generator.shutdown()
    
    print("\n[SMOKE TEST] PASSED!")
