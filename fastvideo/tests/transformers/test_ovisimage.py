# SPDX-License-Identifier: Apache-2.0
"""
Distributed forward-pass test for OvisImageTransformer2DModel.

Mirrors the pattern of test_hunyuanvideo.py:
  - Uses FastVideo's TransformerLoader to load real weights
  - Runs a forward pass with fixed inputs under a distributed environment
  - Checks output shape, finiteness, and (if REFERENCE_LATENT is set)
    the double-precision sum against a committed reference value

Set OVIS_WEIGHTS env var to the local model root, e.g.
    OVIS_WEIGHTS=official_weights/ovis_image \
        pytest fastvideo/tests/transformers/test_ovisimage.py -vs

REFERENCE_LATENT is the double-precision sum of the output latent,
computed with seed=42 on bf16. Verified on RTX 5090.
"""

import os

import pytest
import torch

from fastvideo.configs.models.dits import OvisImageTransformer2DModelConfig
from fastvideo.configs.pipelines.base import PipelineConfig
from fastvideo.distributed.parallel_state import (get_sp_parallel_rank,
                                                   get_sp_world_size)
from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.forward_context import set_forward_context
from fastvideo.logger import init_logger
from fastvideo.models.loader.component_loader import TransformerLoader

logger = init_logger(__name__)

os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "29507"

LOCAL_WEIGHTS = os.getenv("OVIS_WEIGHTS", "official_weights/ovis_image")
TRANSFORMER_PATH = os.path.join(LOCAL_WEIGHTS, "transformer")

LOCAL_RANK = 0
RANK = 0
WORLD_SIZE = 1

# Reference latent: output.double().sum() with seed=42 on L40S GPU.
# Set to None to skip numerical comparison (use this on the first run to
# discover the value, then commit it here).
REFERENCE_LATENT = 292.9996643066406  # bf16, seed=42 (tolerance 1e-2)


@pytest.mark.skipif(
    not os.path.exists(TRANSFORMER_PATH),
    reason=(f"Ovis-Image transformer weights not found at {TRANSFORMER_PATH}. "
            f"Set OVIS_WEIGHTS env var or download from AIDC-AI/Ovis-Image-7B."))
@pytest.mark.usefixtures("distributed_setup")
def test_ovisimage_transformer():
    """
    Load OvisImageTransformer2DModel via TransformerLoader, run a forward pass
    with fixed random inputs (seed=42), check shape + finiteness, and
    optionally compare the output sum against REFERENCE_LATENT.
    """
    torch.cuda.set_device(f"cuda:{LOCAL_RANK}")
    torch.manual_seed(42)

    sp_rank = get_sp_parallel_rank()
    sp_world_size = get_sp_world_size()
    logger.info(f"rank={RANK}, sp_rank={sp_rank}, sp_world_size={sp_world_size}")

    device = torch.device(f"cuda:{LOCAL_RANK}")
    precision_str = "bf16"

    args = FastVideoArgs(
        model_path=TRANSFORMER_PATH,
        dit_cpu_offload=False,
        pipeline_config=PipelineConfig(
            dit_config=OvisImageTransformer2DModelConfig(),
            dit_precision=precision_str,
        ),
    )
    args.device = device

    loader = TransformerLoader()
    model = loader.load(TRANSFORMER_PATH, args)
    model.eval()

    # Fixed small inputs (32×32 latents for speed; real inference uses 128×128)
    B = 1
    C_vae = 16   # in_channels=64, packed factor=4  →  VAE channels = 64//4 = 16
    H, W = 32, 32
    txt_seq = 32
    joint_dim = 2048  # joint_attention_dim

    hidden_states = torch.randn(B, C_vae, H, W,
                                device=device, dtype=torch.bfloat16)
    encoder_hidden_states = torch.randn(B, txt_seq, joint_dim,
                                        device=device, dtype=torch.bfloat16)
    # Ovis-Image FastVideo convention: timestep in [0, 1000]
    timestep = torch.tensor([500.0], device=device, dtype=torch.bfloat16)

    with torch.no_grad():
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            with set_forward_context(current_timestep=0,
                                     attn_metadata=None,
                                     forward_batch=None):
                output = model(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    timestep=timestep,
                )

    # Shape and finiteness
    assert output.shape == (B, C_vae, H, W), \
        f"Unexpected output shape: {output.shape}"
    assert torch.isfinite(output).all(), \
        "Output contains NaN or Inf — check weight loading or forward pass"

    latent = output.double().sum().item()
    logger.info(f"Output latent sum = {latent:.8f}")

    if REFERENCE_LATENT is not None:
        diff = abs(REFERENCE_LATENT - latent)
        logger.info(f"Reference={REFERENCE_LATENT:.8f}, diff={diff:.2e}")
        assert diff < 1e-2, \
            f"Latent sum differs from reference by {diff:.2e} (threshold 1e-2)"
    else:
        logger.info(
            "REFERENCE_LATENT is None — skipping numerical comparison.\n"
            "To pin it, set in this file:\n"
            f"    REFERENCE_LATENT = {latent}")
