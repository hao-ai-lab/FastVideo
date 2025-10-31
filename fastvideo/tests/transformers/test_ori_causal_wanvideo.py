# SPDX-License-Identifier: Apache-2.0
import os
import math

import numpy as np
import pytest
import torch
from fastvideo.tests.third_party.wan.modules.causal_model import CausalWanModel

from fastvideo.configs.pipelines import PipelineConfig
from fastvideo.forward_context import set_forward_context
from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.logger import init_logger
from fastvideo.models.loader.component_loader import TransformerLoader
from fastvideo.models.dits.causal_wanvideo import CausalWanTransformer3DModel
from fastvideo.utils import maybe_download_model
from fastvideo.configs.models.dits import WanVideoConfig
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch


logger = init_logger(__name__)

os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "29503"

BASE_MODEL_PATH = "wlsaidhi/SFWan2.1-T2V-1.3B-Diffusers"
MODEL_PATH = maybe_download_model(BASE_MODEL_PATH,
                                  local_dir=os.path.join(
                                      'data', BASE_MODEL_PATH))
TRANSFORMER_PATH = os.path.join(MODEL_PATH, "transformer")


@pytest.mark.usefixtures("distributed_setup")
def test_ori_causal_wan_transformer():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    precision = torch.bfloat16
    precision_str = "bf16"
    args = FastVideoArgs(model_path=TRANSFORMER_PATH,
                         dit_cpu_offload=True,
                         pipeline_config=PipelineConfig(dit_config=WanVideoConfig(), dit_precision=precision_str))
    args.device = device

    loader = TransformerLoader()
    model2 = loader.load(TRANSFORMER_PATH, args).to(dtype=precision)

    model1 = CausalWanModel.from_pretrained(
        "/mnt/weka/home/hao.zhang/wei/Self-Forcing/wan_models/Wan2.1-T2V-1.3B", device=device,
        torch_dtype=precision).to(device, dtype=precision).requires_grad_(False)
    causal_state_dict = torch.load("/mnt/weka/home/hao.zhang/wei/Self-Forcing/checkpoints/self_forcing_dmd.pt")["generator_ema"]
    new_state_dict = {}
    for k, v in causal_state_dict.items():
        if k.startswith("model."):
            new_state_dict[k.replace("model.", "")] = v
    causal_state_dict = new_state_dict
    model1.load_state_dict(causal_state_dict)

    total_params = sum(p.numel() for p in model1.parameters())
    # Calculate weight sum for model1 (converting to float64 to avoid overflow)
    weight_sum_model1 = sum(
        p.to(torch.float64).sum().item() for p in model1.parameters())
    # Also calculate mean for more stable comparison
    weight_mean_model1 = weight_sum_model1 / total_params
    logger.info("Model 1 weight sum: %s", weight_sum_model1)
    logger.info("Model 1 weight mean: %s", weight_mean_model1)

    # Calculate weight sum for model2 (converting to float64 to avoid overflow)
    total_params_model2 = sum(p.numel() for p in model2.parameters())
    weight_sum_model2 = sum(
        p.to(torch.float64).sum().item() for p in model2.parameters())
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
    text_seq_len = 30

    # Video latents [B, C, T, H, W]
    hidden_states = torch.randn(batch_size,
                                16,
                                12,
                                160,
                                90,
                                device=device,
                                dtype=precision)
    block_sizes = [3 for _ in range(4)]
    timesteps = [1000, 750, 500, 250]

    # Text embeddings [B, L, D] (including global token)
    encoder_hidden_states = torch.randn(batch_size,
                                        text_seq_len + 1,
                                        4096,
                                        device=device,
                                        dtype=precision)

    output1 = _causal_inference(model1, hidden_states.clone(), encoder_hidden_states.clone(), block_sizes, timesteps, precision)
    logger.info("Finish inference for model1")
    output2 = _causal_inference(model2, hidden_states.clone(), encoder_hidden_states.clone(), block_sizes, timesteps, precision)

    # Check if outputs have the same shape
    assert output1.shape == output2.shape, f"Output shapes don't match: {output1.shape} vs {output2.shape}"
    assert output1.dtype == output2.dtype, f"Output dtype don't match: {output1.dtype} vs {output2.dtype}"

    logger.info("Output 1 Sum: %s", output1.float().sum().item())
    logger.info("Output 2 Sum: %s", output2.float().sum().item())

    # Check if outputs are similar (allowing for small numerical differences)
    max_diff = torch.max(torch.abs(output1 - output2))
    mean_diff = torch.mean(torch.abs(output1 - output2))
    logger.info("Max Diff: %s", max_diff.item())
    logger.info("Mean Diff: %s", mean_diff.item())
    assert max_diff < 1e-4, f"Maximum difference between outputs: {max_diff.item()}"
    # mean diff
    assert mean_diff < 1e-4, f"Mean difference between outputs: {mean_diff.item()}"

def _causal_inference(transformer, latents, prompt_embeds, block_sizes, timesteps, target_dtype):
    forward_batch = ForwardBatch(
        data_type="dummy",
    )
    start_index = 0
    pos_start_base = 0
    frame_seq_length = latents.shape[-1] * latents.shape[-2] // (WanVideoConfig().arch_config.patch_size[-1] * WanVideoConfig().arch_config.patch_size[-2])
    seq_len = frame_seq_length * latents.shape[2]
    kv_cache1 = _initialize_kv_cache(transformer, batch_size=latents.shape[0],
                                      kv_cache_size=frame_seq_length * latents.shape[2],
                                      dtype=target_dtype,
                                      device=latents.device)
    crossattn_cache = _initialize_crossattn_cache(
                transformer,
                batch_size=latents.shape[0],
                max_text_len=WanVideoConfig().arch_config.text_len,
                dtype=target_dtype,
                device=latents.device)
    for current_num_frames, t_cur in zip(block_sizes, timesteps):
        # logger.info(f"Current frame idx: {start_index}, Current timestep: {t_cur}")
        # logger.info(f"k cache sum: {sum(kv_cache['k'].float().sum().item() for kv_cache in kv_cache1)}, v cache sum: {sum(kv_cache['v'].float().sum().item() for kv_cache in kv_cache1)}")
        # logger.info(f"latents sum: {latents.float().sum().item()}, encoder_hidden_states sum: {prompt_embeds.float().sum().item()}")
        current_latents = latents[:, :, start_index:start_index +
                                    current_num_frames, :, :]

        attn_metadata = None

        with set_forward_context(current_timestep=0,
                                attn_metadata=attn_metadata,
                                forward_batch=forward_batch):
            # Run transformer; follow DMD stage pattern
            t_expanded_noise = t_cur * torch.ones(
                (current_latents.shape[0], 1),
                device=current_latents.device,
                dtype=torch.long)
            if isinstance(transformer, CausalWanModel):
                pred_noise_btchw = transformer(
                    x=current_latents,
                    context=prompt_embeds,
                    t=t_expanded_noise,
                    seq_len=seq_len,
                    kv_cache=kv_cache1,
                    crossattn_cache=crossattn_cache,
                    current_start=(pos_start_base + start_index) *
                    frame_seq_length
                )
            elif isinstance(transformer, CausalWanTransformer3DModel):
                pred_noise_btchw = transformer(
                    current_latents,
                    prompt_embeds,
                    t_expanded_noise,
                    kv_cache=kv_cache1,
                    crossattn_cache=crossattn_cache,
                    current_start=(pos_start_base + start_index) *
                    frame_seq_length,
                    start_frame=start_index
                )

        # Write back and advance
        latents[:, :, start_index:start_index +
                current_num_frames, :, :] = pred_noise_btchw.clone()

        # Re-run with context timestep to update KV cache using clean context
        context_noise = 0
        t_context = torch.ones([latents.shape[0]],
                                device=latents.device,
                                dtype=torch.long) * int(context_noise)
        context_bcthw = pred_noise_btchw.to(target_dtype)
        with set_forward_context(current_timestep=0,
                                attn_metadata=attn_metadata,
                                forward_batch=forward_batch):
            t_expanded_context = t_context.unsqueeze(1)
            if isinstance(transformer, CausalWanModel):
                _ = transformer(
                    x=context_bcthw,
                    context=prompt_embeds,
                    t=t_expanded_context,
                    seq_len=seq_len,
                    kv_cache=kv_cache1,
                    crossattn_cache=crossattn_cache,
                    current_start=(pos_start_base + start_index) *
                    frame_seq_length
                )
            elif isinstance(transformer, CausalWanTransformer3DModel):
                _ = transformer(
                    context_bcthw,
                    prompt_embeds,
                    t_expanded_context,
                    kv_cache=kv_cache1,
                    crossattn_cache=crossattn_cache,
                    current_start=(pos_start_base + start_index) *
                    frame_seq_length,
                    start_frame=start_index
                )
        start_index += current_num_frames

    return latents

def _initialize_kv_cache(transformer, batch_size, kv_cache_size, dtype, device) -> None:
    """
    Initialize a Per-GPU KV cache aligned with the Wan model assumptions.
    """
    kv_cache1 = []
    if isinstance(transformer, CausalWanModel):
        num_attention_heads = transformer.num_heads
        attention_head_dim = transformer.dim // transformer.num_heads
    elif isinstance(transformer, CausalWanTransformer3DModel):
        num_attention_heads = transformer.num_attention_heads
        attention_head_dim = transformer.attention_head_dim

    for _ in range(len(transformer.blocks)):
        kv_cache1.append({
            "k":
            torch.zeros([
                batch_size, kv_cache_size, num_attention_heads,
                attention_head_dim
            ],
                        dtype=dtype,
                        device=device),
            "v":
            torch.zeros([
                batch_size, kv_cache_size, num_attention_heads,
                attention_head_dim
            ],
                        dtype=dtype,
                        device=device),
            "global_end_index":
            torch.tensor([0], dtype=torch.long, device=device),
            "local_end_index":
            torch.tensor([0], dtype=torch.long, device=device),
        })

    return kv_cache1

def _initialize_crossattn_cache(transformer, batch_size, max_text_len, dtype,
                                    device) -> None:
    """
    Initialize a Per-GPU cross-attention cache aligned with the Wan model assumptions.
    """
    crossattn_cache = []
    if isinstance(transformer, CausalWanModel):
        num_attention_heads = transformer.num_heads
        attention_head_dim = transformer.dim // transformer.num_heads
    elif isinstance(transformer, CausalWanTransformer3DModel):
        num_attention_heads = transformer.num_attention_heads
        attention_head_dim = transformer.attention_head_dim

    for _ in range(len(transformer.blocks)):
        crossattn_cache.append({
            "k":
            torch.zeros([
                batch_size, max_text_len, num_attention_heads,
                attention_head_dim
            ],
                        dtype=dtype,
                        device=device),
            "v":
            torch.zeros([
                batch_size, max_text_len, num_attention_heads,
                attention_head_dim
            ],
                        dtype=dtype,
                        device=device),
            "is_init":
            False,
        })
    return crossattn_cache
