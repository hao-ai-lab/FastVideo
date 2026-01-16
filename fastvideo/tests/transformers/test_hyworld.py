# SPDX-License-Identifier: Apache-2.0
import os

import numpy as np
import pytest
import torch

import sys
# first need to download the HY-WorldPlay model from Github
sys.path.append("/mnt/weka/home/hao.zhang/mhuo/HY-WorldPlay")
from hyvideo.models.transformers.worldplay_1_5_transformer import (
    HunyuanVideo_1_5_DiffusionTransformer,
)

from fastvideo.configs.pipelines import PipelineConfig
from fastvideo.forward_context import set_forward_context
from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.logger import init_logger
from fastvideo.models.loader.component_loader import TransformerLoader
from fastvideo.utils import maybe_download_model
from fastvideo.configs.models.dits import HyWorldConfig
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch


logger = init_logger(__name__)

os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "29503"

BASE_MODEL_PATH = "/mnt/weka/home/hao.zhang/mhuo/data/hyworld"
MODEL_PATH = maybe_download_model(BASE_MODEL_PATH,
                                  local_dir=os.path.join(
                                      'data', BASE_MODEL_PATH))
TRANSFORMER_PATH = os.path.join(MODEL_PATH, "transformer")

@pytest.mark.usefixtures("distributed_setup")
def test_hyworld_transformer():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    precision = torch.bfloat16
    precision_str = "bf16"
    args = FastVideoArgs(model_path=TRANSFORMER_PATH,
                         dit_cpu_offload=True,
                         pipeline_config=PipelineConfig(dit_config=HyWorldConfig(), dit_precision=precision_str))
    args.device = device

    loader = TransformerLoader()
    model2 = loader.load(TRANSFORMER_PATH, args).to(dtype=precision)

    from safetensors.torch import load_file
    
    # Load base model first
    model1 = HunyuanVideo_1_5_DiffusionTransformer.from_pretrained(
        "/mnt/weka/home/hao.zhang/.cache/huggingface/hub/models--tencent--HunyuanVideo-1.5/snapshots/9b49404b3f5df2a8f0b31df27a0c7ab872e7b038/transformer/480p_i2v", device=device,
        torch_dtype=precision)
    model1.add_action_parameters()
    # Load full state dict including action/prope weights
    state_dict = load_file("/mnt/weka/home/hao.zhang/mhuo/data/HY-WorldPlay/bidirectional_model/diffusion_pytorch_model.safetensors")
    model1.load_state_dict(state_dict, strict=True)
    model1 = model1.to(device, dtype=precision).requires_grad_(False)

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
    seq_len = 30
    seq_len_2 = 70
    num_frames = 16

    # Video latents [B, C, T, H, W]
    hidden_states = torch.randn(batch_size,
                                65,
                                num_frames,
                                120,
                                30,
                                device=device,
                                dtype=precision)

    # Text embeddings [B, L, D] (including global token)
    encoder_hidden_states = torch.randn(batch_size,
                                        seq_len + 1,
                                        3584,
                                        device=device,
                                        dtype=precision)
    # Create attention mask for encoder_hidden_states
    encoder_attention_mask = torch.ones(batch_size,
                                        seq_len + 1,
                                        device=device,
                                        dtype=torch.bool)
    encoder_hidden_states[:, 15:] = 0
    encoder_attention_mask[:, 15:] = False

    encoder_hidden_states_2 = torch.randn(batch_size,
                                        seq_len_2 + 1,
                                        1472,
                                        device=device,
                                        dtype=precision)
    encoder_attention_mask_2 = torch.ones(batch_size,
                                        seq_len_2 + 1,
                                        device=device,
                                        dtype=torch.bool)
    encoder_hidden_states_2[:, 39:] = 0
    encoder_attention_mask_2[:, 39:] = False

    image_embeds = torch.zeros(batch_size, 729, 1152, dtype=precision, device=device)

    # Action tensor [B*T] - discrete action per frame, first frame of each batch is 0
    action = torch.randint(0, 10, (batch_size * num_frames,), device=device)
    action[::num_frames] = 0  # First frame of each batch is 0
    action = action.to(dtype=precision)

    # Camera view matrices [B, T, 4, 4] - 4x4 extrinsic/view matrices
    viewmats = torch.eye(4, dtype=precision, device=device).unsqueeze(0).unsqueeze(0).expand(batch_size, num_frames, -1, -1).contiguous()

    # Camera intrinsics [B, T, 3, 3] - 3x3 intrinsic matrices
    Ks = torch.eye(3, dtype=precision, device=device).unsqueeze(0).unsqueeze(0).expand(batch_size, num_frames, -1, -1).contiguous()

    # Timestep [B*T] - one timestep value per frame
    timestep = torch.full((batch_size * num_frames,), 500, device=device, dtype=precision)
    # Timestep for text [B] - one timestep value per batch
    timestep_txt = torch.tensor([500], device=device, dtype=precision)

    forward_batch = ForwardBatch(
        data_type="dummy",
    )

    extra_kwargs = {
        "byt5_text_states": encoder_hidden_states_2,
        "byt5_text_mask": encoder_attention_mask_2
    }

    # print all shapes
    # print(f"hidden_states.shape: {hidden_states.shape}")
    # print(f"encoder_hidden_states.shape: {encoder_hidden_states.shape}")
    # print(f"encoder_attention_mask.shape: {encoder_attention_mask.shape}")
    # print(f"encoder_hidden_states_2.shape: {encoder_hidden_states_2.shape}")
    # print(f"encoder_attention_mask_2.shape: {encoder_attention_mask_2.shape}")
    # print(f"image_embeds.shape: {image_embeds.shape}")
    # print(f"action.shape: {action.shape}")
    # print(f"viewmats.shape: {viewmats.shape}")
    # print(f"Ks.shape: {Ks.shape}")
    # print(f"timestep.shape: {timestep.shape}")
    # print(f"timestep_txt.shape: {timestep_txt.shape}")

    with torch.no_grad():
        with torch.amp.autocast(device_type="cuda", dtype=precision):
            output1 = model1(
                bi_inference=True,
                ar_txt_inference=False,
                ar_vision_inference=False,
                hidden_states=hidden_states,
                text_states=encoder_hidden_states,
                text_states_2=None,
                encoder_attention_mask=encoder_attention_mask,
                timestep=timestep,
                timestep_txt=timestep_txt,
                vision_states=image_embeds,
                return_dict=False,
                extra_kwargs=extra_kwargs,
                action=action,
                viewmats=viewmats,
                Ks=Ks,
            )[0]
            with set_forward_context(
                    current_timestep=0,
                    attn_metadata=None,
                    forward_batch=forward_batch,
            ):
                output2 = model2(hidden_states=hidden_states,
                                encoder_hidden_states=[encoder_hidden_states, encoder_hidden_states_2],
                                encoder_attention_mask=[encoder_attention_mask, encoder_attention_mask_2],
                                encoder_hidden_states_image=[image_embeds],
                                timestep=timestep,
                                timestep_txt=timestep_txt,
                                action=action,
                                viewmats=viewmats,
                                Ks=Ks)
            
            # # Compare outputs (both are dicts now)
            # print("\n=== Output Comparison ===")
            # for key in output1.keys():
            #     if key in output2:
            #         o1 = output1[key]
            #         o2 = output2[key]
            #         if o1.shape == o2.shape:
            #             match = torch.allclose(o1.float(), o2.float(), atol=1e-4, rtol=1e-4)
            #             max_diff = (o1.float() - o2.float()).abs().max().item()
            #             print(f"{key}: match={match}, max_diff={max_diff}")
            #         else:
            #             print(f"{key}: shape mismatch - {o1.shape} vs {o2.shape}")
    
    # Check final hidden_states (the main output)
    # hs1 = output1["hidden_states"]
    # hs2 = output2["hidden_states"]
    hs1 = output1
    hs2 = output2
    assert hs1.shape == hs2.shape, f"Output shapes don't match: {hs1.shape} vs {hs2.shape}"
    assert hs1.dtype == hs2.dtype, f"Output dtype don't match: {hs1.dtype} vs {hs2.dtype}"

    # Check if outputs are similar (allowing for small numerical differences)
    max_diff = torch.max(torch.abs(hs1.float() - hs2.float()))
    mean_diff = torch.mean(torch.abs(hs1.float() - hs2.float()))
    logger.info("Max Diff: %s", max_diff.item())
    logger.info("Mean Diff: %s", mean_diff.item())
    assert max_diff < 1e-4, f"Maximum difference between outputs: {max_diff.item()}"
    assert mean_diff < 1e-4, f"Mean difference between outputs: {mean_diff.item()}"


if __name__ == "__main__":
    import torch.distributed as dist
    from fastvideo.distributed import init_distributed_environment, initialize_model_parallel
    
    # Initialize distributed environment for single GPU
    if not dist.is_initialized():
        init_distributed_environment(world_size=1, rank=0, local_rank=0)
        initialize_model_parallel(tensor_model_parallel_size=1, sequence_model_parallel_size=1)
    
    test_hyworld_transformer()
    
    dist.destroy_process_group()