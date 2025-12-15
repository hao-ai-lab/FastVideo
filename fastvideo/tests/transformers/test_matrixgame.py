# SPDX-License-Identifier: Apache-2.0
import os

import pytest
import torch

from fastvideo.configs.models.dits.matrixgame import MatrixGameWanVideoConfig
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
from fastvideo.utils import maybe_download_model

logger = init_logger(__name__)

os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "29505"
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

BASE_MODEL_PATH = os.environ.get(
    "MATRIX_GAME_BASE_MODEL", "/workspace/Matrix-Game-2.0-Diffusers"
)
MODEL_VARIANT = os.environ.get("MATRIX_GAME_VARIANT", "base_distilled_model")
REFERENCE_LATENT = -112914.0136833489


def _resolve_transformer_path() -> str:
    if os.path.exists(BASE_MODEL_PATH):
        model_root = BASE_MODEL_PATH
    else:
        model_root = maybe_download_model(
            BASE_MODEL_PATH, local_dir=os.path.join("data", BASE_MODEL_PATH)
        )

    candidate = os.path.join(model_root, MODEL_VARIANT, "transformer")
    if not os.path.isdir(candidate):
        fallback = os.path.join(model_root, "transformer")
        if os.path.isdir(fallback):
            candidate = fallback
        else:
            raise FileNotFoundError(
                f"Could not locate transformer weights under {candidate} or {fallback}"
            )
    return candidate


@pytest.mark.usefixtures("distributed_setup")
def test_matrixgame_transformer():
    transformer_path = _resolve_transformer_path()

    sp_rank = get_sp_parallel_rank()
    sp_world_size = get_sp_world_size()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    precision = torch.bfloat16
    precision_str = "bf16"

    args = FastVideoArgs(
        model_path=transformer_path,
        dit_cpu_offload=False,
        use_fsdp_inference=False,
        pipeline_config=PipelineConfig(
            dit_config=MatrixGameWanVideoConfig(), dit_precision=precision_str
        ),
    )
    args.device = device

    loader = TransformerLoader()
    model = loader.load(transformer_path, args).to(device, dtype=precision)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    weight_sum = sum(p.to(torch.float64).sum().item() for p in model.parameters())
    weight_mean = weight_sum / total_params
    logger.info("Total parameters: %s", total_params)
    logger.info("Weight sum: %s", weight_sum)
    logger.info("Weight mean: %s", weight_mean)

    torch.manual_seed(42)

    batch_size = 1
    latent_frames = 15
    latent_height = 44
    latent_width = 80
    image_dim = 1280
    image_seq_len = 257

    # Set num_frame_per_block to match actual latent_frames for action_model
    model.num_frame_per_block = latent_frames
    # Force full attention mode
    model.local_attn_size = -1
    # Clear cached block masks so they regenerate with correct settings
    model.block_mask = None
    model.block_mask_mouse = None
    model.block_mask_keyboard = None

    # Video latents [B, C, T, H, W]
    x = torch.randn(batch_size, 16, latent_frames, latent_height, latent_width,
                    device=device, dtype=precision)
    cond_concat = torch.randn(batch_size, 20, latent_frames, latent_height, latent_width,
                              device=device, dtype=precision)
    hidden_states = torch.cat([x, cond_concat], dim=1)

    if sp_world_size > 1:
        chunk_per_rank = hidden_states.shape[2] // sp_world_size
        hidden_states = hidden_states[:, :, sp_rank * chunk_per_rank:(sp_rank + 1) * chunk_per_rank]

    # Image embeddings
    encoder_hidden_states_image = [
        torch.randn(batch_size, image_seq_len, image_dim, device=device, dtype=precision)
    ]

    # Timestep
    timestep = torch.full((batch_size, latent_frames), 500, device=device, dtype=precision)

    # Action conditions
    num_pixel_frames = (latent_frames - 1) * 4 + 1
    mouse_cond = torch.randn(batch_size, num_pixel_frames, 2, device=device, dtype=precision)
    keyboard_cond = torch.randn(batch_size, num_pixel_frames, 4, device=device, dtype=precision)

    # Text embeddings (empty for i2v)
    encoder_hidden_states = torch.zeros(batch_size, 0, model.config.hidden_size,
                                        device=device, dtype=precision)

    forward_batch = ForwardBatch(data_type="dummy")
    model.num_frame_per_block = latent_frames

    with torch.no_grad():
        with torch.amp.autocast('cuda', dtype=precision):
            with set_forward_context(current_timestep=0, attn_metadata=None, forward_batch=forward_batch):
                output = model(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_hidden_states_image=encoder_hidden_states_image,
                    timestep=timestep,
                    mouse_cond=mouse_cond,
                    keyboard_cond=keyboard_cond,
                )

    latent = output.double().sum().item()

    diff = abs(REFERENCE_LATENT - latent)
    relative_diff = diff / abs(REFERENCE_LATENT)
    logger.info(f"Reference latent: {REFERENCE_LATENT}, Current latent: {latent}")
    logger.info(f"Absolute diff: {diff}, Relative diff: {relative_diff * 100:.4f}%")

    # The 0.28% diff now
    assert relative_diff < 0.005, f"Output latents differ significantly: relative diff = {relative_diff * 100:.4f}% (max allowed: 0.5%)"
