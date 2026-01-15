# SPDX-License-Identifier: Apache-2.0
"""Tests for HyWorld Transformer model."""

import json
import os

import pytest
import torch

from fastvideo.configs.pipelines.base import PipelineConfig
from fastvideo.distributed.parallel_state import (
    get_sp_parallel_rank,
    get_sp_world_size)
from fastvideo.logger import init_logger
from fastvideo.models.loader.component_loader import TransformerLoader
from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.forward_context import set_forward_context
from fastvideo.configs.models.dits import HyWorldConfig
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch

logger = init_logger(__name__)

os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "29506"

# HyWorld model path
MODEL_PATH = "/mnt/weka/home/hao.zhang/mhuo/data/hyworld"
TRANSFORMER_PATH = os.path.join(MODEL_PATH, "transformer")
CONFIG_PATH = os.path.join(TRANSFORMER_PATH, "config.json")

LOCAL_RANK = 0
RANK = 0
WORLD_SIZE = 1

# Reference latent sum for regression testing (to be updated after first successful run)
REFERENCE_LATENT = None  # Will be set after initial run


@pytest.mark.usefixtures("distributed_setup")
def test_hyworld_transformer():
    """Test HyWorld transformer forward pass."""
    logger.info(
        f"Initializing process: rank={RANK}, local_rank={LOCAL_RANK}, world_size={WORLD_SIZE}"
    )

    torch.cuda.set_device(f"cuda:{LOCAL_RANK}")

    # Get tensor parallel info
    sp_rank = get_sp_parallel_rank()
    sp_world_size = get_sp_world_size()

    logger.info(
        f"Process rank {RANK} initialized with SP rank {sp_rank} in SP world size {sp_world_size}"
    )

    # Load config
    with open(CONFIG_PATH) as f:
        config = json.load(f)
    logger.info("Loaded HyWorld config: %s", config)

    device = torch.device(f"cuda:{LOCAL_RANK}")
    precision = torch.bfloat16
    precision_str = "bf16"

    args = FastVideoArgs(
        model_path=TRANSFORMER_PATH,
        dit_cpu_offload=True,
        pipeline_config=PipelineConfig(
            dit_config=HyWorldConfig(),
            dit_precision=precision_str
        )
    )
    args.device = device

    loader = TransformerLoader()
    model = loader.load(TRANSFORMER_PATH, args)
    model.eval()

    # Log model info
    total_params = sum(p.numel() for p in model.parameters())
    logger.info("Model loaded with %d parameters", total_params)

    # Create random inputs for testing
    batch_size = 1
    num_frames = 5  # HyWorld typically processes video frames
    height = 40  # Latent height (post-VAE)
    width = 72  # Latent width (post-VAE)
    latent_channels = 65  # HyWorld uses 65 channels (32*2 + 1 for concat condition)

    # Video latents [B, C, T, H, W]
    hidden_states = torch.randn(
        batch_size,
        latent_channels,
        num_frames,
        height,
        width,
        device=device,
        dtype=precision
    )

    # Apply sequence parallel sharding if needed
    chunk_per_rank = hidden_states.shape[2] // sp_world_size
    hidden_states = hidden_states[:, :, sp_rank * chunk_per_rank:(sp_rank + 1) * chunk_per_rank]

    # Text embeddings - HyWorld uses two text encoders
    text_seq_len = 256
    text_embed_dim = 3584  # Qwen2.5-VL dim
    text_embed_2_dim = 1472  # T5 dim (byT5)

    encoder_hidden_states = torch.randn(
        batch_size,
        text_seq_len,
        text_embed_dim,
        device=device,
        dtype=precision
    )
    encoder_hidden_states_2 = torch.randn(
        batch_size,
        text_seq_len,
        text_embed_2_dim,
        device=device,
        dtype=precision
    )

    # Attention masks
    encoder_attention_mask = torch.ones(
        batch_size, text_seq_len,
        device=device,
        dtype=precision
    )
    encoder_attention_mask_2 = torch.ones(
        batch_size, text_seq_len,
        device=device,
        dtype=precision
    )

    # Image embeddings for I2V - SigLIP outputs
    image_seq_len = 729  # 27x27 patches for 384x384 image
    image_embed_dim = 1152  # SigLIP hidden size
    encoder_hidden_states_image = torch.randn(
        batch_size,
        image_seq_len,
        image_embed_dim,
        device=device,
        dtype=precision
    )

    # Timesteps
    timestep = torch.tensor([500], device=device, dtype=torch.long)
    timestep_txt = torch.tensor([500], device=device, dtype=torch.long)

    # Action conditioning (HyWorld-specific, required)
    action = torch.randn(batch_size * num_frames, device=device, dtype=precision)

    # Camera parameters for ProPE (required)
    viewmats = torch.eye(4, device=device, dtype=precision).unsqueeze(0).unsqueeze(0)
    viewmats = viewmats.expand(batch_size, num_frames, -1, -1).contiguous()

    Ks = torch.eye(3, device=device, dtype=precision).unsqueeze(0).unsqueeze(0)
    Ks = Ks.expand(batch_size, num_frames, -1, -1).contiguous()

    forward_batch = ForwardBatch(
        data_type="dummy",
    )
    data = torch.load("/mnt/weka/home/hao.zhang/mhuo/HY-WorldPlay/kwargs.pt")
    # Take first batch only (batch_size=1) for tensors with batch dimension
    viewmats = data["viewmats"][0:1]
    Ks = data["Ks"][0:1]
    hidden_states = data["hidden_states"][0:1]
    encoder_hidden_states = data["text_states"][0:1]
    encoder_hidden_states_2 = data["extra_kwargs"]['byt5_text_states'][0:1]
    encoder_hidden_states_image = data["vision_states"][0:1]
    encoder_attention_mask = data["encoder_attention_mask"][0:1]
    encoder_attention_mask_2 = data["extra_kwargs"]['byt5_text_mask'][0:1]
    # Keep the first half for timestep, timestep_txt, action
    timestep = data["timestep"][:len(data["timestep"])//2]
    timestep_txt = data["timestep_txt"][:len(data["timestep_txt"])//2]
    action = data["action"][:len(data["action"])//2]
    print("encoder_hidden_states.shape", encoder_hidden_states.shape)
    print("encoder_hidden_states_2.shape", encoder_hidden_states_2.shape)
    print("encoder_hidden_states_image.shape", encoder_hidden_states_image.shape)
    print("encoder_attention_mask.shape", encoder_attention_mask.shape)
    print("encoder_attention_mask_2.shape", encoder_attention_mask_2.shape)
    print("timestep.shape", timestep.shape)
    print("timestep_txt.shape", timestep_txt.shape)
    print("action.shape", action.shape)
    print("viewmats.shape", viewmats.shape)
    print("Ks.shape", Ks.shape)

    # Disable gradients for inference
    with torch.no_grad():
        with torch.amp.autocast(device_type="cuda", dtype=precision):
            with set_forward_context(
                current_timestep=0,
                attn_metadata=None,
                forward_batch=forward_batch
            ):
                output = model(
                    hidden_states=hidden_states,
                    encoder_hidden_states=[encoder_hidden_states, encoder_hidden_states_2],
                    encoder_hidden_states_image=[encoder_hidden_states_image],
                    encoder_attention_mask=[encoder_attention_mask, encoder_attention_mask_2],
                    timestep=timestep,
                    timestep_txt=timestep_txt,
                    action=action,
                    viewmats=viewmats,
                    Ks=Ks,
                )

    print("output.shape", output.shape)
    output_gt = torch.load("/mnt/weka/home/hao.zhang/mhuo/HY-WorldPlay/noise_pred.pt")
    print("output_gt[0:1].shape", output_gt[0:1].shape)
    print("output and output_gt[0:1] are close: ", torch.allclose(output, output_gt[0:1]))

    # Compute output statistics
    latent_sum = output.double().sum().item()
    latent_mean = output.double().mean().item()
    latent_std = output.double().std().item()

    logger.info("Output shape: %s", output.shape)
    logger.info("Output latent sum: %s", latent_sum)
    logger.info("Output latent mean: %s", latent_mean)
    logger.info("Output latent std: %s", latent_std)
    logger.info("Max memory allocated: %.2f GB", torch.cuda.max_memory_allocated() / 1024**3)

    # Basic sanity checks
    assert output is not None, "Output should not be None"
    assert not torch.isnan(output).any(), "Output contains NaN values"
    assert not torch.isinf(output).any(), "Output contains Inf values"

    # Check output shape matches input spatial dimensions
    # HyWorld outputs should have same spatial dims as input (after unpatchify)
    logger.info("HyWorld transformer test passed!")


@pytest.mark.usefixtures("distributed_setup")
def test_hyworld_transformer_with_prope():
    """Test HyWorld transformer forward pass with ProPE camera conditioning."""
    torch.cuda.set_device(f"cuda:{LOCAL_RANK}")

    sp_rank = get_sp_parallel_rank()
    sp_world_size = get_sp_world_size()

    device = torch.device(f"cuda:{LOCAL_RANK}")
    precision = torch.bfloat16
    precision_str = "bf16"

    args = FastVideoArgs(
        model_path=TRANSFORMER_PATH,
        dit_cpu_offload=True,
        pipeline_config=PipelineConfig(
            dit_config=HyWorldConfig(),
            dit_precision=precision_str
        )
    )
    args.device = device

    loader = TransformerLoader()
    model = loader.load(TRANSFORMER_PATH, args)
    model.eval()

    # Create inputs
    batch_size = 1
    num_frames = 5
    height = 40
    width = 72
    latent_channels = 65

    hidden_states = torch.randn(
        batch_size, latent_channels, num_frames, height, width,
        device=device, dtype=precision
    )
    chunk_per_rank = hidden_states.shape[2] // sp_world_size
    hidden_states = hidden_states[:, :, sp_rank * chunk_per_rank:(sp_rank + 1) * chunk_per_rank]

    text_seq_len = 256
    encoder_hidden_states = torch.randn(batch_size, text_seq_len, 3584, device=device, dtype=precision)
    encoder_hidden_states_2 = torch.randn(batch_size, text_seq_len, 1472, device=device, dtype=precision)
    encoder_attention_mask = torch.ones(batch_size, text_seq_len, device=device, dtype=precision)
    encoder_attention_mask_2 = torch.ones(batch_size, text_seq_len, device=device, dtype=precision)
    encoder_hidden_states_image = torch.randn(batch_size, 729, 1152, device=device, dtype=precision)

    timestep = torch.tensor([500], device=device, dtype=torch.long)
    timestep_txt = torch.tensor([500], device=device, dtype=torch.long)
    action = torch.randn(batch_size * num_frames, device=device, dtype=precision)

    # Camera parameters for ProPE
    viewmats = torch.eye(4, device=device, dtype=precision).unsqueeze(0).unsqueeze(0)
    viewmats = viewmats.expand(batch_size, num_frames, -1, -1).contiguous()

    Ks = torch.eye(3, device=device, dtype=precision).unsqueeze(0).unsqueeze(0)
    Ks = Ks.expand(batch_size, num_frames, -1, -1).contiguous()

    forward_batch = ForwardBatch(data_type="dummy")

    with torch.no_grad():
        with torch.amp.autocast(device_type="cuda", dtype=precision):
            with set_forward_context(current_timestep=0, attn_metadata=None, forward_batch=forward_batch):
                output = model(
                    hidden_states=hidden_states,
                    encoder_hidden_states=[encoder_hidden_states, encoder_hidden_states_2],
                    encoder_hidden_states_image=[encoder_hidden_states_image],
                    encoder_attention_mask=[encoder_attention_mask, encoder_attention_mask_2],
                    timestep=timestep,
                    timestep_txt=timestep_txt,
                    action=action,
                    viewmats=viewmats,
                    Ks=Ks,
                )

    assert output is not None, "Output should not be None"
    assert not torch.isnan(output).any(), "Output contains NaN values"
    logger.info("HyWorld transformer test (with ProPE) passed!")
