# SPDX-License-Identifier: Apache-2.0
"""
GRPO utilities for Wan model in FastVideo.

This module ports the SDE step and pipeline functions from FlowGRPO to work with
FastVideo's scheduler and pipeline interfaces.

Ported from:
- flow_grpo/flow_grpo/diffusers_patch/wan_pipeline_with_logprob.py

Key adaptations:
1. Uses FastVideo's FlowUniPCMultistepScheduler instead of diffusers' UniPCMultistepScheduler
2. Works with FastVideo's WanPipeline (ComposedPipelineBase) instead of diffusers' WanPipeline
3. Direct module access via pipeline.get_module() instead of pipeline attributes
4. Simplified prompt encoding (direct text encoder usage instead of pipeline stages)
"""

import math
import time
from typing import Any

import torch
from tqdm import tqdm
from diffusers.utils.torch_utils import randn_tensor

from fastvideo.forward_context import set_forward_context
from fastvideo.logger import init_logger
from fastvideo.models.schedulers.scheduling_flow_unipc_multistep import (
    FlowUniPCMultistepScheduler)
from fastvideo.pipelines.basic.wan.wan_pipeline import WanPipeline
from fastvideo.utils import get_compute_dtype

# for test_wan_transformer2
import os
from diffusers import WanTransformer3DModel
from fastvideo.utils import maybe_download_model
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch

from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.configs.pipelines import PipelineConfig
from fastvideo.configs.models.dits import WanVideoConfig
from fastvideo.models.loader.component_loader import TransformerLoader

logger = init_logger(__name__)

def test_wan_transformer():
    BASE_MODEL_PATH = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
    MODEL_PATH = maybe_download_model(BASE_MODEL_PATH,
                                    local_dir=os.path.join(
                                        'data', BASE_MODEL_PATH))
    TRANSFORMER_PATH = os.path.join(MODEL_PATH, "transformer")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    precision = torch.bfloat16
    precision_str = "bf16"
    args = FastVideoArgs(model_path=TRANSFORMER_PATH,
                         dit_cpu_offload=True,
                         pipeline_config=PipelineConfig(dit_config=WanVideoConfig(), dit_precision=precision_str))
    args.device = device

    loader = TransformerLoader()
    model2 = loader.load(TRANSFORMER_PATH, args).to(dtype=precision)

    model1 = WanTransformer3DModel.from_pretrained(
        TRANSFORMER_PATH, device=device,
        torch_dtype=precision).to(device, dtype=precision).requires_grad_(False)

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

    # Video latents [B, C, T, H, W]
    hidden_states = torch.randn(batch_size,
                                16,
                                21,
                                160,
                                90,
                                device=device,
                                dtype=precision)

    # Text embeddings [B, L, D] (including global token)
    encoder_hidden_states = torch.randn(batch_size,
                                        seq_len + 1,
                                        4096,
                                        device=device,
                                        dtype=precision)

    # Timestep
    timestep = torch.tensor([500], device=device, dtype=precision)

    forward_batch = ForwardBatch(
        data_type="dummy",
    )

    with torch.amp.autocast('cuda', dtype=precision):
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
            output2 = model2(hidden_states=hidden_states,
                             encoder_hidden_states=encoder_hidden_states,
                             timestep=timestep)

    # Check if outputs have the same shape
    assert output1.shape == output2.shape, f"Output shapes don't match: {output1.shape} vs {output2.shape}"
    assert output1.dtype == output2.dtype, f"Output dtype don't match: {output1.dtype} vs {output2.dtype}"

    # Check if outputs are similar (allowing for small numerical differences)
    max_diff = torch.max(torch.abs(output1 - output2))
    mean_diff = torch.mean(torch.abs(output1 - output2))
    logger.info("Max Diff: %s", max_diff.item())
    logger.info("Mean Diff: %s", mean_diff.item())
    assert max_diff < 1e-1, f"Maximum difference between outputs: {max_diff.item()}"
    # mean diff
    assert mean_diff < 1e-2, f"Mean difference between outputs: {mean_diff.item()}"
    '''
    INFO 01-19 22:53:46 [wan_grpo_utils.py:74] Model 1 weight sum: 395834.3506456231████                                                    | 1/2 [00:00<00:00,  7.84it/s]
    INFO 01-19 22:53:46 [wan_grpo_utils.py:75] Model 1 weight mean: 0.0002789536598289884
    INFO 01-19 22:53:47 [wan_grpo_utils.py:83] Model 2 weight sum: 395834.3506456231
    INFO 01-19 22:53:47 [wan_grpo_utils.py:84] Model 2 weight mean: 0.0002789536598289884
    INFO 01-19 22:53:47 [wan_grpo_utils.py:87] Weight sum difference: 0.0
    INFO 01-19 22:53:47 [wan_grpo_utils.py:89] Weight mean difference: 0.0
    INFO 01-19 22:53:54 [wan_grpo_utils.py:145] Max Diff: 0.08203125
    INFO 01-19 22:53:54 [wan_grpo_utils.py:146] Mean Diff: 0.01129150390625
    '''


def test_wan_transformer2(model2):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    precision = torch.bfloat16

    logger.info("loading model1 transformer weight")
    BASE_MODEL_PATH = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
    MODEL_PATH = maybe_download_model(BASE_MODEL_PATH,
                                    local_dir=os.path.join(
                                        'data', BASE_MODEL_PATH))
    TRANSFORMER_PATH = os.path.join(MODEL_PATH, "transformer")
    model1 = WanTransformer3DModel.from_pretrained(
        TRANSFORMER_PATH,
        device=device,
        torch_dtype=precision,
    ).to(device, dtype=precision).requires_grad_(False)

    total_params = sum(p.numel() for p in model1.parameters())
    # Calculate weight sum for model1 (converting to float64 to avoid overflow)
    weight_sum_model1 = sum(
        p.to(torch.float64).sum().item() for p in model1.parameters()
    )
    # Also calculate mean for more stable comparison
    weight_mean_model1 = weight_sum_model1 / total_params
    logger.info("Model 1 weight sum: %s", weight_sum_model1)
    logger.info("Model 1 weight mean: %s", weight_mean_model1)

    # Calculate weight sum for model2 (converting to float64 to avoid overflow)
    total_params_model2 = sum(p.numel() for p in model2.parameters())
    weight_sum_model2 = sum(
        p.to(torch.float64).sum().item() for p in model2.parameters()
    )
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
    hidden_states = torch.randn(
        batch_size,
        16,
        21,
        160,
        90,
        device=device,
        dtype=precision,
    )

    # Text embeddings [B, L, D] (including global token)
    encoder_hidden_states = torch.randn(
        batch_size,
        seq_len + 1,
        4096,
        device=device,
        dtype=precision,
    )

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
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                timestep=timestep,
            )

    # Print basic stats for debugging (cast to float32 for stability)
    out1 = output1.detach().float()
    out2 = output2.detach().float()
    logger.info(
        "output1 stats: min=%s max=%s mean=%s std=%s",
        out1.min().item(),
        out1.max().item(),
        out1.mean().item(),
        out1.std(unbiased=False).item(),
    )
    logger.info(
        "output2 stats: min=%s max=%s mean=%s std=%s",
        out2.min().item(),
        out2.max().item(),
        out2.mean().item(),
        out2.std(unbiased=False).item(),
    )

    # Check if outputs have the same shape
    assert (
        output1.shape == output2.shape
    ), f"Output shapes don't match: {output1.shape} vs {output2.shape}"
    assert (
        output1.dtype == output2.dtype
    ), f"Output dtype don't match: {output1.dtype} vs {output2.dtype}"

    # Check if outputs are similar (allowing for small numerical differences)
    max_diff = torch.max(torch.abs(output1 - output2))
    mean_diff = torch.mean(torch.abs(output1 - output2))
    logger.info("Max Diff: %s", max_diff.item())
    logger.info("Mean Diff: %s", mean_diff.item())
    assert max_diff < 1e-1, f"Maximum difference between outputs: {max_diff.item()}"
    # mean diff
    assert mean_diff < 1e-2, f"Mean difference between outputs: {mean_diff.item()}"

    '''
    when --dit_precision "bf16", use_fsdp hardcoded to False:
    INFO 01-19 22:01:24 [wan_grpo_utils.py:65] Model 1 weight sum: 395834.3506456231████████████████████████████████████████████████████████| 2/2 [00:00<00:00,  3.25it/s]
    INFO 01-19 22:01:24 [wan_grpo_utils.py:66] Model 1 weight mean: 0.0002789536598289884
    INFO 01-19 22:01:24 [wan_grpo_utils.py:75] Model 2 weight sum: 395125.463677882
    INFO 01-19 22:01:24 [wan_grpo_utils.py:76] Model 2 weight mean: 0.0002739000890162162
    INFO 01-19 22:01:24 [wan_grpo_utils.py:79] Weight sum difference: 708.8869677411276
    INFO 01-19 22:01:24 [wan_grpo_utils.py:81] Weight mean difference: 5.053570812772192e-06
    INFO 01-19 22:01:32 [wan_grpo_utils.py:139] output1 stats: min=-2.28125 max=1.921875 mean=-0.16638492047786713 std=0.458170622587204
    INFO 01-19 22:01:32 [wan_grpo_utils.py:146] output2 stats: min=-2.296875 max=1.90625 mean=-0.166452556848526 std=0.4579130709171295
    INFO 01-19 22:01:32 [wan_grpo_utils.py:165] Max Diff: 0.08984375
    INFO 01-19 22:01:32 [wan_grpo_utils.py:166] Mean Diff: 0.0120849609375

    when --dit_precision "fp32", use_fsdp not changed:

    '''


def sde_step_with_logprob(
    scheduler: FlowUniPCMultistepScheduler,
    model_output: torch.FloatTensor,
    timestep: float | torch.FloatTensor,
    sample: torch.FloatTensor,
    prev_sample: torch.FloatTensor | None = None,
    generator: torch.Generator | None = None,
    determistic: bool = False,
    return_pixel_log_prob: bool = False,
    return_dt_and_std_dev_t: bool = False
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, ...]:
    """
    Predict the sample from the previous timestep by reversing the SDE. 
    This function propagates the flow process from the learned model outputs 
    (most often the predicted velocity) and computes log probabilities.

    Ported from FlowGRPO's sde_step_with_logprob to work with FastVideo's
    FlowUniPCMultistepScheduler.

    Args:
        scheduler: FastVideo FlowUniPCMultistepScheduler instance
        model_output: The direct output from learned flow model
        timestep: The current discrete timestep in the diffusion chain
        sample: A current instance of a sample created by the diffusion process
        prev_sample: Optional previous sample (if provided, used instead of sampling)
        generator: Optional random number generator
        determistic: If True, no noise is added (deterministic sampling)
        return_pixel_log_prob: If True, return pixel-level log probabilities (not used)
        return_dt_and_std_dev_t: If True, return dt and std_dev_t separately

    Returns:
        If return_dt_and_std_dev_t=True:
            (prev_sample, log_prob, prev_sample_mean, std_dev_t, sqrt_dt)
        Otherwise:
            (prev_sample, log_prob, prev_sample_mean, std_dev_t * sqrt_dt)
    """
    

    # # Convert all variables to fp32 for numerical stability
    # model_output = model_output.float()
    # sample = sample.float()
    # if prev_sample is not None:
    #     prev_sample = prev_sample.float()
    
    # Get step indices for current and previous timesteps
    # Handle both single timestep and batch of timesteps
    if isinstance(timestep, torch.Tensor):
        if timestep.ndim == 0:
            timestep = timestep.unsqueeze(0)
        step_indices = [
            scheduler.index_for_timestep(t.item()) for t in timestep
        ]
    else:
        step_indices = [scheduler.index_for_timestep(timestep)]

    prev_step_indices = [step + 1 for step in step_indices]

    # Move sigmas to sample device
    sigmas = scheduler.sigmas.to(sample.device)
# myregion debug: hardcode sigmas to flow_grpo's
    sigmas = torch.Tensor([0.9997, 0.9824, 0.9639, 0.9441, 0.9227, 0.8996, 0.8746, 0.8475, 0.8178,
        0.7853, 0.7496, 0.7102, 0.6663, 0.6173, 0.5621, 0.4997, 0.4283, 0.3459,
        0.2498, 0.1362, 0.0000]).to(sample.device, sample.dtype)
# end region

    # Get sigma values for current and previous steps
    sigma = sigmas[step_indices].view(-1, 1, 1, 1, 1)
    sigma_prev = sigmas[prev_step_indices].view(-1, 1, 1, 1, 1)
    sigma_max = sigmas[0].item()  # First sigma (highest)
    sigma_min = sigmas[-1].item()  # Last sigma (lowest)

    dt = sigma_prev - sigma

# myregion debug
    print(f"[DEBUG]: sigma_max: {sigma_max}, sigma_min: {sigma_min}, dt: {dt}")
    print(f"[DEBUG]: in sde_step_with_logprob(), timestep: {timestep}")
    print(f"[DEBUG]: in sde_step_with_logprob(), sigmas: {sigmas}")
    print(f"[DEBUG]: in sde_step_with_logprob(), step_indices: {step_indices}")
    print(f"[DEBUG]: in sde_step_with_logprob(), prev_step_indices: {prev_step_indices}")
    '''
    [DEBUG]: in sde_step_with_logprob(), timestep: tensor([428, 428, 428, 428], device='cuda:0')
    [DEBUG]: in sde_step_with_logprob(), sigmas: tensor([0.9999, 0.9826, 0.9642, 0.9443, 0.9230, 0.8999, 0.8749, 0.8477, 0.8181,
            0.7856, 0.7499, 0.7104, 0.6665, 0.6175, 0.5624, 0.4999, 0.4285, 0.3461,
            0.2499, 0.1363, 0.0000], device='cuda:0')
    [DEBUG]: in sde_step_with_logprob(), step_indices: [16, 16, 16, 16]
    [DEBUG]: in sde_step_with_logprob(), prev_step_indices: [17, 17, 17, 17]

    DEBUG]: in sde_step_with_logprob(), timestep: tensor([249], device='cuda:0')███████████████▎       | 18/20 [00:04<00:00,  3.87step/s, step_time=0.26s, timestep=346.0]
    [DEBUG]: in sde_step_with_logprob(), sigmas: tensor([0.9999, 0.9826, 0.9642, 0.9443, 0.9230, 0.8999, 0.8749, 0.8477, 0.8181,
            0.7856, 0.7499, 0.7104, 0.6665, 0.6175, 0.5624, 0.4999, 0.4285, 0.3461,
            0.2499, 0.1363, 0.0000], device='cuda:0')
    [DEBUG]: in sde_step_with_logprob(), step_indices: [18]
    [DEBUG]: in sde_step_with_logprob(), prev_step_indices: [19]

    [DEBUG]: in sde_step_with_logprob(), timestep: tensor([617, 617, 617, 617], device='cuda:0')
    [DEBUG]: in sde_step_with_logprob(), sigmas: tensor([0.9999, 0.9826, 0.9642, 0.9443, 0.9230, 0.8999, 0.8749, 0.8477, 0.8181,
            0.7856, 0.7499, 0.7104, 0.6665, 0.6175, 0.5624, 0.4999, 0.4285, 0.3461,
            0.2499, 0.1363, 0.0000], device='cuda:0')
    [DEBUG]: in sde_step_with_logprob(), step_indices: [13, 13, 13, 13]
    [DEBUG]: in sde_step_with_logprob(), prev_step_indices: [14, 14, 14, 14]
    '''
# endregion

    # Compute std_dev_t and prev_sample_mean using SDE formulation
    std_dev_t = sigma_min + (sigma_max - sigma_min) * sigma
    prev_sample_mean = (sample * (1 + std_dev_t**2 / (2 * sigma) * dt) +
                        model_output * (1 + std_dev_t**2 * (1 - sigma) /
                                        (2 * sigma)) * dt)

    if prev_sample is not None and generator is not None:
        raise ValueError(
            "Cannot pass both generator and prev_sample. Please make sure that either `generator` or"
            " `prev_sample` stays `None`.")

    # Sample prev_sample if not provided
    if prev_sample is None:
        variance_noise = randn_tensor(
            model_output.shape,
            generator=generator,
            device=model_output.device,
            dtype=model_output.dtype,
        )
        sqrt_dt = torch.sqrt(-1 * dt)  # dt is negative (going backwards)
        prev_sample = prev_sample_mean + std_dev_t * sqrt_dt * variance_noise
    else:
        sqrt_dt = torch.sqrt(-1 * dt)

    # No noise is added during evaluation (deterministic)
    if determistic:
        prev_sample = sample + dt * model_output
        sqrt_dt = torch.sqrt(-1 * dt)

    # Compute log probability: log p(prev_sample | sample, model_output)
    # Assuming Gaussian distribution: N(prev_sample_mean, (std_dev_t * sqrt_dt)^2)
    std_dev_sqrt_dt = std_dev_t * sqrt_dt
    log_prob = (
        -((prev_sample.detach() - prev_sample_mean)**2) /
        (2 * (std_dev_sqrt_dt**2)) - torch.log(
            std_dev_sqrt_dt + 1e-8)  # Add small epsilon for numerical stability
        - torch.log(
            torch.sqrt(2 * torch.as_tensor(math.pi, device=sample.device))))

    # Mean along all but batch dimension
    log_prob = log_prob.mean(dim=tuple(range(1, log_prob.ndim)))

    if return_dt_and_std_dev_t:
        return prev_sample, log_prob, prev_sample_mean, std_dev_t, sqrt_dt
    return prev_sample, log_prob, prev_sample_mean, std_dev_t * sqrt_dt


def wan_pipeline_with_logprob(
    pipeline,
    prompt: str | list[str] = None,
    negative_prompt: str | list[str] = None,
    height: int = 480,
    width: int = 832,
    num_frames: int = 81,
    num_inference_steps: int = 50,
    guidance_scale: float = 5.0,
    num_videos_per_prompt: int | None = 1,
    generator: torch.Generator | list[torch.Generator] | None = None,
    latents: torch.Tensor | None = None,
    prompt_embeds: torch.Tensor | None = None,
    negative_prompt_embeds: torch.Tensor | None = None,
    output_type: str | None = "pt",
    return_dict: bool = False,
    attention_kwargs: dict[str, Any] | None = None,
    max_sequence_length: int = 512,
    determistic: bool = False,
    kl_reward: float = 0.0,
    return_pixel_log_prob: bool = False,
) -> tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor],
           list[torch.Tensor], torch.Tensor | None]:
    """
    Wan pipeline with log probability computation for GRPO training.

    Ported from FlowGRPO's wan_pipeline_with_logprob to work with FastVideo's WanPipeline.
    This function generates videos and computes log probabilities at each denoising step.

    Args:
        pipeline: FastVideo WanPipeline instance
        prompt: Text prompt(s) for generation
        negative_prompt: Negative prompt(s) for classifier-free guidance
        height: Height of generated video
        width: Width of generated video
        num_frames: Number of frames in generated video
        num_inference_steps: Number of denoising steps
        guidance_scale: Classifier-free guidance scale
        num_videos_per_prompt: Number of videos to generate per prompt
        generator: Random generator for reproducibility
        latents: Optional initial latents
        prompt_embeds: Optional pre-computed prompt embeddings
        negative_prompt_embeds: Optional pre-computed negative prompt embeddings
        output_type: Output type ("pt" for PyTorch tensor, "np" for numpy, "latent" for latents only)
        return_dict: Whether to return dict (not used, always returns tuple)
        attention_kwargs: Optional attention kwargs
        max_sequence_length: Maximum sequence length for text encoding
        determistic: If True, use deterministic sampling (no noise)
        kl_reward: KL reward coefficient (if > 0, computes KL divergence)
        return_pixel_log_prob: If True, return pixel-level log probabilities (not used)

    Returns:
        Tuple of:
            - video: Generated video tensor [B, C, T, H, W] or latents if output_type="latent"
            - all_latents: List of latents at each step [num_steps+1] of shape [B, C, T, H, W]
            - all_log_probs: List of log probabilities at each step [num_steps] of shape [B]
            - all_kl: List of KL divergences at each step [num_steps] of shape [B] (if kl_reward > 0)
            - prompt_ids: Tokenized prompt IDs [B, seq_len] (None if prompt_embeds were provided)
    """
    # Get device from transformer
    transformer = pipeline.get_module("transformer")

# myregion debug: test transformer output
    logger.info(f"testing transformer, running test_wan_transformer2")
    test_wan_transformer()
    # test_wan_transformer2(transformer)
# endregion

    # hardcode dtype for debug
    # transformer_dtype = torch.float32
    # use get_compute_dtype() to get dtype based on mixed precision
    transformer_dtype = get_compute_dtype()
    logger.info(f"[DEBUG]: transformer_dtype: {transformer_dtype}")

    # Get scheduler and other modules
    scheduler = pipeline.get_module("scheduler")
    vae = pipeline.get_module("vae")

    # Determine batch size
    if prompt is not None and isinstance(prompt, str):
        batch_size = 1
    elif prompt is not None and isinstance(prompt, list):
        batch_size = len(prompt)
    elif prompt_embeds is not None:
        batch_size = prompt_embeds.shape[0]
    else:
        raise ValueError("Either prompt or prompt_embeds must be provided")

    # Encode prompts if not provided
    prompt_ids = None
    if prompt_embeds is None:
        # Encode prompts directly using text encoder and tokenizer
        # This is a simplified encoding - for full pipeline encoding, use TextEncodingStage
        text_encoder = pipeline.get_module("text_encoder")
        tokenizer = pipeline.get_module("tokenizer")

        # Normalize to list
        if isinstance(prompt, str):
            prompts_list = [prompt]
        else:
            prompts_list = prompt

        # Tokenize prompts
        text_inputs = tokenizer(prompts_list,
                                padding="max_length",
                                max_length=max_sequence_length,
                                truncation=True,
                                return_tensors="pt").to(pipeline.device)

        # Store prompt_ids for return
        prompt_ids = text_inputs["input_ids"]

        # Encode with text encoder
        with torch.no_grad():
            outputs = text_encoder(
                text_inputs["input_ids"],
                attention_mask=text_inputs["attention_mask"],
                output_hidden_states=True,
            )
            # Get last hidden state (Wan typically uses last hidden state)
            prompt_embeds = outputs.last_hidden_state

        # Encode negative prompts if CFG is enabled
        if guidance_scale > 1.0:
            if negative_prompt is None:
                negative_prompt = [""] * len(prompts_list)
            elif isinstance(negative_prompt, str):
                negative_prompt = [negative_prompt]

            neg_text_inputs = tokenizer(negative_prompt,
                                        padding="max_length",
                                        max_length=max_sequence_length,
                                        truncation=True,
                                        return_tensors="pt").to(pipeline.device)

            with torch.no_grad():
                neg_outputs = text_encoder(
                    neg_text_inputs["input_ids"],
                    attention_mask=neg_text_inputs["attention_mask"],
                    output_hidden_states=True,
                )
                negative_prompt_embeds = neg_outputs.last_hidden_state
        else:
            negative_prompt_embeds = None

# myregion Debug: Print shapes of prompt embeddings
        logger.info(f"After encoding - prompt_embeds shape: {prompt_embeds.shape if prompt_embeds is not None else None}")
        logger.info(f"After encoding - negative_prompt_embeds shape: {negative_prompt_embeds.shape if negative_prompt_embeds is not None else None}")
        logger.info(f"After encoding - prompt_embeds dtype: {prompt_embeds.dtype if prompt_embeds is not None else None}")
        logger.info(f"After encoding - negative_prompt_embeds dtype: {negative_prompt_embeds.dtype if negative_prompt_embeds is not None else None}")
        '''
        INFO 01-17 05:31:13 [wan_grpo_utils.py:290] After encoding - prompt_embeds shape: torch.Size([4, 512, 4096])
        INFO 01-17 05:31:13 [wan_grpo_utils.py:291] After encoding - negative_prompt_embeds shape: None
        INFO 01-17 05:31:13 [wan_grpo_utils.py:292] After encoding - prompt_embeds dtype: torch.float32
        INFO 01-17 05:31:13 [wan_grpo_utils.py:293] After encoding - negative_prompt_embeds dtype: None
        '''
# endregion
    # logger.info("wan_pipeline_with_logprob's transformer class type: %s", type(transformer))
    # logger.info("Variables in transformer: %s", str(dir(transformer)))
    prompt_embeds = prompt_embeds.to(transformer_dtype)
    if negative_prompt_embeds is not None:
        negative_prompt_embeds = negative_prompt_embeds.to(transformer_dtype)


    # Prepare timesteps
    scheduler.set_timesteps(num_inference_steps, device=pipeline.device)
    timesteps = scheduler.timesteps

    # Prepare latent variables
    num_channels_latents = transformer.config.in_channels
    vae = pipeline.get_module("vae")
    # Get VAE scale factors
    vae_scale_factor_spatial = vae.spatial_compression_ratio
    vae_scale_factor_temporal = vae.temporal_compression_ratio

    if latents is None:
        # Generate random latents
        # Note: num_frames in latents accounts for temporal compression
        num_latent_frames = (num_frames - 1) // vae_scale_factor_temporal + 1
        latents_shape = (
            batch_size * num_videos_per_prompt,
            num_channels_latents,
            num_latent_frames,
            height // vae_scale_factor_spatial,
            width // vae_scale_factor_spatial,
        )
        if generator is not None:
            if isinstance(generator, list):
                latents = [
                    torch.randn(
                        latents_shape[1:],
                        generator=gen,
                        device=pipeline.device,
                        dtype=transformer_dtype,
                    ) for gen in generator
                ]
                latents = torch.stack(latents, dim=0)
            else:
                latents = torch.randn(
                    latents_shape,
                    generator=generator,
                    device=pipeline.device,
                    dtype=transformer_dtype,
                )
        else:
            latents = torch.randn(latents_shape,
                                  device=pipeline.device,
                                  dtype=transformer_dtype)
    else:
        latents = latents.to(device=pipeline.device, dtype=transformer_dtype)

# myregion Debug: Print latents shape, dtype, and value range
    logger.info("=" * 80)
    logger.info("Latents Debug Information:")
    logger.info(f"  Shape: {latents.shape}")
    logger.info(f"  Dtype: {latents.dtype}")
    logger.info(f"  Min value: {latents.min().item():.6f}")
    logger.info(f"  Max value: {latents.max().item():.6f}")
    logger.info(f"  Mean value: {latents.mean().item():.6f}")
    logger.info(f"  Std value: {latents.std().item():.6f}")
    logger.info(f"  Device: {latents.device}")
    logger.info("=" * 80)
    '''
    INFO 01-17 07:41:33 [wan_grpo_utils.py:355] ================================================================================
    INFO 01-17 07:41:33 [wan_grpo_utils.py:356] Latents Debug Information:
    INFO 01-17 07:41:33 [wan_grpo_utils.py:357]   Shape: torch.Size([4, 16, 9, 30, 52])
    INFO 01-17 07:41:33 [wan_grpo_utils.py:358]   Dtype: torch.bfloat16
    INFO 01-17 07:41:33 [wan_grpo_utils.py:359]   Min value: -4.500000
    INFO 01-17 07:41:33 [wan_grpo_utils.py:360]   Max value: 4.656250
    INFO 01-17 07:41:33 [wan_grpo_utils.py:361]   Mean value: 0.000111
    INFO 01-17 07:41:33 [wan_grpo_utils.py:362]   Std value: 1.000000
    INFO 01-17 07:41:33 [wan_grpo_utils.py:363]   Device: cuda:0
    INFO 01-17 07:41:33 [wan_grpo_utils.py:364] ================================================================================
    '''
# endregion
    

    all_latents = [latents]
    all_log_probs = []
    all_kl = []


# myregion Debug
    logger.info("Tensor type issue debugging:")
    logger.info(f"latents: {type(latents)}")
    logger.info(f"prompt_embeds: {type(prompt_embeds)}")
    logger.info(f"[DEBUG]: before denoising loop: type(timesteps): {type(timesteps)}")
    logger.info(f"[DEBUG]: before denoising loop: timesteps.shape: {timesteps.shape}")
# endregion

    # Progress bar for denoising loop
    progress_bar = tqdm(enumerate(timesteps),
                        total=len(timesteps),
                        desc="Denoising steps",
                        unit="step")

    for i, t in progress_bar:
        step_start_time = time.time()
        latents_ori = latents.clone()
        timestep = t.expand(latents.shape[0]) if isinstance(
            t, torch.Tensor) else torch.tensor([t] * latents.shape[0],
                                               device=pipeline.device)
        
        
        logger.info(f"[DEBUG]: before set_forward_context: current_timestep=i:{i}")
        # Predict noise with transformer
        with set_forward_context(
                current_timestep=t.item(),
                attn_metadata=None,
                forward_batch=None,
        ):
            noise_pred = transformer(
                hidden_states=latents,
                timestep=timestep,
                encoder_hidden_states=prompt_embeds,
                attention_kwargs=attention_kwargs,
                return_dict=False,
            )[0]
        noise_pred = noise_pred.to(prompt_embeds.dtype)

        # Classifier-free guidance
        if guidance_scale > 1.0:
            with set_forward_context(
                    current_timestep=i,
                    attn_metadata=None,
                    forward_batch=None,
            ):
                noise_uncond = transformer(
                    hidden_states=latents,
                    timestep=timestep,
                    encoder_hidden_states=negative_prompt_embeds,
                    attention_kwargs=attention_kwargs,
                    return_dict=False,
                )[0]
            noise_pred = noise_uncond + guidance_scale * (noise_pred -
                                                          noise_uncond)

        # SDE step with log probability
        latents, log_prob, prev_latents_mean, std_dev_t = sde_step_with_logprob(
            scheduler,
            noise_pred,#.float(),
            t.unsqueeze(0) if isinstance(t, torch.Tensor) else t,
            latents,#.float(),
            determistic=determistic,
            return_pixel_log_prob=return_pixel_log_prob)
        # sde_step_with_logprob returns fp32
        # latents = latents.to(transformer_dtype)
        prev_latents = latents.clone()

        all_latents.append(latents)
        all_log_probs.append(log_prob)

        # Compute KL divergence if kl_reward > 0 (for KL reward in sampling)
        if kl_reward > 0 and not determistic:
            # Use reference model (disable adapter if using LoRA)
            latent_model_input_ref = torch.cat(
                [latents_ori] *
                2) if guidance_scale > 1.0 else latents_ori
            with set_forward_context(
                    current_timestep=i,
                    attn_metadata=None,
                    forward_batch=None,
            ):
                with transformer.disable_adapter() if hasattr(
                        transformer, 'disable_adapter') else torch.no_grad():
                    noise_pred_ref = transformer(
                        hidden_states=latent_model_input_ref,
                        timestep=timestep,
                        encoder_hidden_states=prompt_embeds,
                        attention_kwargs=attention_kwargs,
                        return_dict=False,
                    )[0]
            noise_pred_ref = noise_pred_ref.to(prompt_embeds.dtype)

            # Perform guidance for reference model
            if guidance_scale > 1.0:
                noise_pred_uncond_ref, noise_pred_text_ref = noise_pred_ref.chunk(
                    2)
                noise_pred_ref = noise_pred_uncond_ref + guidance_scale * (
                    noise_pred_text_ref - noise_pred_uncond_ref)

            # Compute reference log prob
            _, ref_log_prob, ref_prev_latents_mean, ref_std_dev_t = sde_step_with_logprob(
                scheduler,
                noise_pred_ref.float(),
                t.unsqueeze(0) if isinstance(t, torch.Tensor) else t,
                latents_ori.float(),
                prev_sample=prev_latents.float(),
                determistic=determistic,
            )

            # Compute KL divergence: KL = (mean_diff)^2 / (2 * std^2)
            assert torch.allclose(
                std_dev_t, ref_std_dev_t
            ), "std_dev_t should match between current and reference"
            kl = (prev_latents_mean - ref_prev_latents_mean)**2 / (2 *
                                                                   std_dev_t**2)
            kl = kl.mean(dim=tuple(range(1, kl.ndim)))
            all_kl.append(kl)
        else:
            # No KL reward, set to zero
            all_kl.append(torch.zeros(len(latents), device=latents.device))

        # Update progress bar with timing information
        step_time = time.time() - step_start_time
        progress_bar.set_postfix({
            "step_time":
            f"{step_time:.2f}s",
            "timestep":
            f"{t.item() if isinstance(t, torch.Tensor) else t:.1f}"
        })

    # Decode latents to video if needed
    if output_type != "latent":
        latents = latents.to(vae.dtype)

        # Apply VAE normalization (Wan VAE specific)
        # Wan VAE requires denormalization before decoding
        if hasattr(vae, 'config') and hasattr(vae.config,
                                              'latents_mean') and hasattr(
                                                  vae.config, 'latents_std'):
            # Get z_dim from config or VAE
            z_dim = getattr(vae.config, 'z_dim', latents.shape[1])
            latents_mean = (torch.tensor(vae.config.latents_mean,
                                         device=latents.device,
                                         dtype=latents.dtype).view(
                                             1, z_dim, 1, 1, 1))
            latents_std = (
                1.0 / torch.tensor(vae.config.latents_std,
                                   device=latents.device,
                                   dtype=latents.dtype).view(1, z_dim, 1, 1, 1))
            latents = latents / latents_std + latents_mean
        elif hasattr(vae, 'latents_mean') and hasattr(vae, 'latents_std'):
            # Alternative: check if latents_mean/std are direct attributes
            z_dim = latents.shape[1]
            latents_mean = (torch.tensor(vae.latents_mean,
                                         device=latents.device,
                                         dtype=latents.dtype).view(
                                             1, z_dim, 1, 1, 1))
            latents_std = (1.0 / torch.tensor(
                vae.latents_std, device=latents.device,
                dtype=latents.dtype).view(1, z_dim, 1, 1, 1))
            latents = latents / latents_std + latents_mean

        # Decode using VAE
        with torch.no_grad():
            video = vae.decode(latents.float(), return_dict=False)[0]
            # VAE.decode returns tensor directly (not tuple)

            # Postprocess video: convert from [-1, 1] to [0, 1]
            # FastVideo VAE typically outputs in [-1, 1] range
            video = (video / 2 + 0.5).clamp(0, 1)
    else:
        video = latents

    return video, all_latents, all_log_probs, all_kl, prompt_ids
