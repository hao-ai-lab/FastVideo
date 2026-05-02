# SPDX-License-Identifier: Apache-2.0
import os

import numpy as np
import pytest
import torch
from diffusers import WanTransformer3DModel
from torch.testing import assert_close

from fastvideo.configs.pipelines import PipelineConfig
from fastvideo.forward_context import set_forward_context
from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.logger import init_logger
from fastvideo.models.loader.component_loader import TransformerLoader
from fastvideo.utils import maybe_download_model
from fastvideo.configs.models.dits import WanVideoConfig
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch


logger = init_logger(__name__)

os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "29503"

BASE_MODEL_PATH = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
MODEL_PATH = maybe_download_model(BASE_MODEL_PATH, local_dir=os.path.join("data", BASE_MODEL_PATH))
TRANSFORMER_PATH = os.path.join(MODEL_PATH, "transformer")


@pytest.mark.usefixtures("distributed_setup")
def test_wan_transformer():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    precision = torch.bfloat16
    precision_str = "bf16"
    args = FastVideoArgs(
        model_path=TRANSFORMER_PATH,
        dit_cpu_offload=True,
        pipeline_config=PipelineConfig(dit_config=WanVideoConfig(), dit_precision=precision_str),
    )
    args.device = device

    loader = TransformerLoader()
    model2 = loader.load(TRANSFORMER_PATH, args).to(dtype=precision)

    model1 = (
        WanTransformer3DModel.from_pretrained(TRANSFORMER_PATH, device=device, torch_dtype=precision)
        .to(device, dtype=precision)
        .requires_grad_(False)
    )

    total_params = sum(p.numel() for p in model1.parameters())
    # Calculate weight sum for model1 (converting to float64 to avoid overflow)
    weight_sum_model1 = sum(p.to(torch.float64).sum().item() for p in model1.parameters())
    # Also calculate mean for more stable comparison
    weight_mean_model1 = weight_sum_model1 / total_params
    logger.info("Model 1 weight sum: %s", weight_sum_model1)
    logger.info("Model 1 weight mean: %s", weight_mean_model1)

    # Calculate weight sum for model2 (converting to float64 to avoid overflow)
    total_params_model2 = sum(p.numel() for p in model2.parameters())
    weight_sum_model2 = sum(p.to(torch.float64).sum().item() for p in model2.parameters())
    # Also calculate mean for more stable comparison
    weight_mean_model2 = weight_sum_model2 / total_params_model2
    logger.info("Model 2 weight sum: %s", weight_sum_model2)
    logger.info("Model 2 weight mean: %s", weight_mean_model2)

    weight_sum_diff = abs(weight_sum_model1 - weight_sum_model2)
    logger.info("Weight sum difference: %s", weight_sum_diff)
    weight_mean_diff = abs(weight_mean_model1 - weight_mean_model2)
    logger.info("Weight mean difference: %s", weight_mean_diff)

    # Set both models to eval mode
    model1 = model1.eval()
    model2 = model2.eval()

    # Create identical inputs for both models
    batch_size = 1
    seq_len = 30

    # Video latents [B, C, T, H, W]
    hidden_states = torch.randn(batch_size, 16, 21, 160, 90, device=device, dtype=precision)

    # Text embeddings [B, L, D] (including global token)
    encoder_hidden_states = torch.randn(batch_size, seq_len + 1, 4096, device=device, dtype=precision)

    # Timestep
    timestep = torch.tensor([500], device=device, dtype=precision)

    forward_batch = ForwardBatch(
        data_type="dummy",
    )

    with torch.amp.autocast("cuda", dtype=precision):
        output1 = model1(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            timestep=timestep,
            return_dict=False,
        )[0]
        with set_forward_context(
            current_timestep=0,
            attn_metadata=None,
            forward_batch=forward_batch,
        ):
            output2 = model2(
                hidden_states=hidden_states, encoder_hidden_states=encoder_hidden_states, timestep=timestep
            )

    # Check if outputs have the same shape
    assert output1.shape == output2.shape, f"Output shapes don't match: {output1.shape} vs {output2.shape}"
    assert output1.dtype == output2.dtype, f"Output dtype don't match: {output1.dtype} vs {output2.dtype}"

    # Check if outputs are similar (allowing for small numerical differences)
    assert_close(output1, output2, atol=1e-1, rtol=1e-2)


def _find_first_diverging_layer(
    model1,
    model2,
    hidden_states,
    encoder_hidden_states,
    timestep,
    forward_batch,
    atol=1e-2,
    rtol=1e-2,
):
    """
    Find the first block layer where model1 and model2 outputs diverge.
    Uses forward hooks to capture intermediate outputs.
    Returns (first_diverging_layer_idx, outputs1, outputs2) or (None, outputs1, outputs2) if no divergence.
    """
    outputs1 = []
    outputs2 = []

    def make_hook(store_list):
        def hook(module, input, output):
            out = output[0] if isinstance(output, tuple) else output
            store_list.append(out.detach().clone())
        return hook

    hooks1 = [
        block.register_forward_hook(make_hook(outputs1))
        for block in model1.blocks
    ]
    hooks2 = [
        block.register_forward_hook(make_hook(outputs2))
        for block in model2.blocks
    ]

    try:
        with torch.amp.autocast("cuda", dtype=hidden_states.dtype):
            _ = model1(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                timestep=timestep,
                return_dict=False,
            )[0]
        with set_forward_context(
            current_timestep=0,
            attn_metadata=None,
            forward_batch=forward_batch,
        ):
            with torch.amp.autocast("cuda", dtype=hidden_states.dtype):
                _ = model2(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    timestep=timestep,
                )
    finally:
        for h in hooks1 + hooks2:
            h.remove()

    if len(outputs1) != len(outputs2):
        logger.warning(
            "Block count mismatch: model1 has %d blocks, model2 has %d",
            len(outputs1), len(outputs2),
        )
        return 0, outputs1, outputs2  # Divergence at layer 0

    first_diverging = None
    for i, (o1, o2) in enumerate(zip(outputs1, outputs2)):
        # Handle potential shape mismatch (e.g. sequence parallel sharding)
        if o1.shape != o2.shape:
            first_diverging = i
            logger.info(
                "Layer %d: shape mismatch - model1 %s vs model2 %s",
                i, o1.shape, o2.shape,
            )
            break
        try:
            assert_close(o1, o2, atol=atol, rtol=rtol)
        except AssertionError as e:
            first_diverging = i
            max_diff = (o1.float() - o2.float()).abs().max().item()
            mean_diff = (o1.float() - o2.float()).abs().mean().item()
            logger.info(
                "Layer %d: outputs diverge (max_diff=%.6e, mean_diff=%.6e): %s",
                i, max_diff, mean_diff, e,
            )
            break

    return first_diverging, outputs1, outputs2


@pytest.mark.usefixtures("distributed_setup")
def test_wan_transformer_layer_divergence():
    """
    Find at which block layer output1 and output2 first diverge.
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    precision = torch.bfloat16
    precision_str = "bf16"
    args = FastVideoArgs(
        model_path=TRANSFORMER_PATH,
        dit_cpu_offload=True,
        pipeline_config=PipelineConfig(
            dit_config=WanVideoConfig(), dit_precision=precision_str
        ),
    )
    args.device = device

    loader = TransformerLoader()
    model2 = loader.load(TRANSFORMER_PATH, args).to(dtype=precision)
    model1 = WanTransformer3DModel.from_pretrained(
        TRANSFORMER_PATH, device=device, torch_dtype=precision
    ).to(device, dtype=precision).requires_grad_(False)

    model1 = model1.eval()
    model2 = model2.eval()

    batch_size = 1
    seq_len = 30
    hidden_states = torch.randn(
        batch_size, 16, 21, 160, 90, device=device, dtype=precision
    )
    encoder_hidden_states = torch.randn(
        batch_size, seq_len + 1, 4096, device=device, dtype=precision
    )
    timestep = torch.tensor([500], device=device, dtype=precision)
    forward_batch = ForwardBatch(data_type="dummy")

    # Use atol=0, rtol=0 for exact match (detect any numerical difference)
    first_div, outputs1, outputs2 = _find_first_diverging_layer(
        model1,
        model2,
        hidden_states,
        encoder_hidden_states,
        timestep,
        forward_batch,
        atol=0.0,
        rtol=0.0,
    )

    if first_div is not None:
        logger.info(
            "First diverging layer: %d (0-indexed, out of %d blocks)",
            first_div, len(outputs1),
        )
        print(f"\n>>> First diverging layer: {first_div} (0-indexed)")
    else:
        logger.info("All %d layers match within tolerance.", len(outputs1))
        print(f"\n>>> All {len(outputs1)} layers match within tolerance.")
