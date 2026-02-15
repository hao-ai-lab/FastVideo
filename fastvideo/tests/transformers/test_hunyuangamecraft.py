# SPDX-License-Identifier: Apache-2.0
"""
Regression test for HunyuanGameCraft transformer.

Tests that the FastVideo HunyuanGameCraft implementation produces consistent outputs.
"""
import os

import pytest
import torch

from fastvideo.configs.models.dits.hunyuangamecraft import HunyuanGameCraftConfig
from fastvideo.configs.pipelines import PipelineConfig
from fastvideo.distributed.parallel_state import (
    get_sp_parallel_rank,
    get_sp_world_size,
)
from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.forward_context import set_forward_context
from fastvideo.logger import init_logger
from fastvideo.models.loader.component_loader import TransformerLoader
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch

logger = init_logger(__name__)

os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "29507"
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("DISABLE_SP", "1")

# Path to converted weights (local path, not HuggingFace)
TRANSFORMER_PATH = "official_weights/hunyuan-gamecraft/transformer"

# Reference latent computed from FastVideo HunyuanGameCraft model with:
# - seed=42, batch=1, frames=9, H=44, W=80, text_seq=32
# - 33 camera frames at 352x640 resolution
# - timestep=500
REFERENCE_LATENT = 42351.12903189659


@pytest.mark.usefixtures("distributed_setup")
def test_hunyuangamecraft_transformer():
    """Test HunyuanGameCraft transformer regression."""
    
    if not os.path.exists(TRANSFORMER_PATH):
        pytest.skip(f"Weights not found at {TRANSFORMER_PATH}")
    
    sp_rank = get_sp_parallel_rank()
    sp_world_size = get_sp_world_size()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    precision = torch.bfloat16
    precision_str = "bf16"

    args = FastVideoArgs(
        model_path=TRANSFORMER_PATH,
        dit_cpu_offload=False,
        use_fsdp_inference=False,
        pipeline_config=PipelineConfig(
            dit_config=HunyuanGameCraftConfig(), dit_precision=precision_str
        ),
    )
    args.device = device

    loader = TransformerLoader()
    model = loader.load(TRANSFORMER_PATH, args).to(device, dtype=precision)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    weight_sum = sum(p.to(torch.float64).sum().item() for p in model.parameters())
    weight_mean = weight_sum / total_params
    logger.info("Total parameters: %s", total_params)
    logger.info("Weight sum: %s", weight_sum)
    logger.info("Weight mean: %s", weight_mean)

    torch.manual_seed(42)

    batch_size = 1
    latent_frames = 9  # GameCraft uses 9 latent frames
    latent_height = 44
    latent_width = 80
    text_seq_len = 32

    # Input latents [B, 33, T, H, W] - 16 latent + 16 gt_latent + 1 mask
    hidden_states = torch.randn(
        batch_size, 33, latent_frames, latent_height, latent_width,
        device=device, dtype=precision
    )

    if sp_world_size > 1:
        chunk_per_rank = hidden_states.shape[2] // sp_world_size
        hidden_states = hidden_states[:, :, sp_rank * chunk_per_rank:(sp_rank + 1) * chunk_per_rank]

    # Text embeddings (LLaMA)
    text_states = torch.randn(batch_size, text_seq_len, 4096, device=device, dtype=precision)
    
    # CLIP pooled embeddings
    text_states_2 = torch.randn(batch_size, 768, device=device, dtype=precision)
    
    # Text mask
    text_mask = torch.ones(batch_size, text_seq_len, device=device, dtype=torch.long)

    # Timestep
    timestep = torch.tensor([500], device=device, dtype=precision)

    # Camera states [B, num_frames, 6, H, W] - 33 video frames at full resolution
    # For 9 latent frames, we use 33 camera frames (matching official model)
    camera_states = torch.randn(
        batch_size, 33, 6, 352, 640,
        device=device, dtype=precision
    )

    encoder_hidden_states = [text_states, text_states_2]

    forward_batch = ForwardBatch(data_type="video", enable_teacache=False)

    with torch.no_grad():
        with torch.amp.autocast('cuda', dtype=precision):
            with set_forward_context(current_timestep=0, attn_metadata=None, forward_batch=forward_batch):
                output = model(
                    hidden_states,
                    encoder_hidden_states,
                    timestep,
                    camera_states=camera_states,
                    encoder_attention_mask=[text_mask],
                )

    latent = output.double().sum().item()
    logger.info(f"Output shape: {output.shape}")
    logger.info(f"Current latent: {latent}")

    if REFERENCE_LATENT is not None:
        diff = abs(REFERENCE_LATENT - latent)
        relative_diff = diff / abs(REFERENCE_LATENT)
        logger.info(f"Reference latent: {REFERENCE_LATENT}")
        logger.info(f"Absolute diff: {diff}, Relative diff: {relative_diff * 100:.4f}%")

        assert relative_diff < 0.005, \
            f"Output latents differ significantly: relative diff = {relative_diff * 100:.4f}% (max allowed: 0.5%)"
    else:
        logger.info(f"No reference latent set. To enable regression testing, set REFERENCE_LATENT = {latent}")
