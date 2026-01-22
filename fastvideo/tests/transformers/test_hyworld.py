# SPDX-License-Identifier: Apache-2.0
import os

import pytest
import torch

from fastvideo.configs.models.dits import HyWorldConfig
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
os.environ["MASTER_PORT"] = "29503"
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

MODEL_PATH = maybe_download_model("mignonjia/hyworld")
TRANSFORMER_PATH = os.path.join(MODEL_PATH, "transformer")
REFERENCE_LATENT = -201883.48880133405  # Pre-computed reference value


@pytest.mark.usefixtures("distributed_setup")
def test_hyworld_transformer():
    transformer_path = TRANSFORMER_PATH

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
            dit_config=HyWorldConfig(), dit_precision=precision_str
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

    # Create inputs for the model
    batch_size = 1
    seq_len = 30
    seq_len_2 = 70
    num_frames = 16
    latent_height = 120
    latent_width = 30

    # Video latents [B, C, T, H, W]
    hidden_states = torch.randn(
        batch_size, 65, num_frames, latent_height, latent_width,
        device=device, dtype=precision
    )

    if sp_world_size > 1:
        chunk_per_rank = hidden_states.shape[2] // sp_world_size
        hidden_states = hidden_states[:, :, sp_rank * chunk_per_rank:(sp_rank + 1) * chunk_per_rank]

    # Text embeddings [B, L, D] (including global token)
    encoder_hidden_states = torch.randn(
        batch_size, seq_len + 1, 3584, device=device, dtype=precision
    )
    # Create attention mask for encoder_hidden_states
    encoder_attention_mask = torch.ones(
        batch_size, seq_len + 1, device=device, dtype=torch.bool
    )
    encoder_hidden_states[:, 15:] = 0
    encoder_attention_mask[:, 15:] = False

    encoder_hidden_states_2 = torch.randn(
        batch_size, seq_len_2 + 1, 1472, device=device, dtype=precision
    )
    encoder_attention_mask_2 = torch.ones(
        batch_size, seq_len_2 + 1, device=device, dtype=torch.bool
    )
    encoder_hidden_states_2[:, 39:] = 0
    encoder_attention_mask_2[:, 39:] = False

    # Image embeddings
    image_embeds = torch.zeros(
        batch_size, 729, 1152, dtype=precision, device=device
    )

    # Action tensor [B*T] - discrete action per frame, first frame of each batch is 0
    action = torch.randint(0, 10, (batch_size * num_frames,), device=device)
    action[::num_frames] = 0  # First frame of each batch is 0
    action = action.to(dtype=precision)

    # Camera view matrices [B, T, 4, 4] - 4x4 extrinsic/view matrices
    viewmats = torch.eye(4, dtype=precision, device=device).unsqueeze(0).unsqueeze(0).expand(
        batch_size, num_frames, -1, -1
    ).contiguous()

    # Camera intrinsics [B, T, 3, 3] - 3x3 intrinsic matrices
    Ks = torch.eye(3, dtype=precision, device=device).unsqueeze(0).unsqueeze(0).expand(
        batch_size, num_frames, -1, -1
    ).contiguous()

    # Timestep [B*T] - one timestep value per frame
    timestep = torch.full((batch_size * num_frames,), 500, device=device, dtype=precision)
    # Timestep for text [B] - one timestep value per batch
    timestep_txt = torch.tensor([500], device=device, dtype=precision)

    forward_batch = ForwardBatch(data_type="dummy")

    with torch.no_grad():
        with torch.amp.autocast("cuda", dtype=precision):
            with set_forward_context(
                current_timestep=0,
                attn_metadata=None,
                forward_batch=forward_batch,
            ):
                output = model(
                    hidden_states=hidden_states,
                    encoder_hidden_states=[encoder_hidden_states, encoder_hidden_states_2],
                    encoder_attention_mask=[encoder_attention_mask, encoder_attention_mask_2],
                    encoder_hidden_states_image=[image_embeds],
                    timestep=timestep,
                    timestep_txt=timestep_txt,
                    action=action,
                    viewmats=viewmats,
                    Ks=Ks,
                )

    latent = output.double().sum().item()

    diff = abs(REFERENCE_LATENT - latent)
    relative_diff = diff / abs(REFERENCE_LATENT)
    logger.info(f"Reference latent: {REFERENCE_LATENT}, Current latent: {latent}")
    logger.info(f"Absolute diff: {diff}, Relative diff: {relative_diff * 100:.4f}%")

    # Allow 0.5% relative difference
    assert relative_diff < 0.005, f"Output latents differ significantly: relative diff = {relative_diff * 100:.4f}% (max allowed: 0.5%)"
