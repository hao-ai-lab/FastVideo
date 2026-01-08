# SPDX-License-Identifier: Apache-2.0
"""
LTX-2 denoising stage using the native sigma schedule.
"""

from __future__ import annotations

import math
import os

import torch
from tqdm.auto import tqdm

from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.forward_context import set_forward_context
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.pipelines.stages.base import PipelineStage
from fastvideo.pipelines.stages.validators import StageValidators as V
from fastvideo.pipelines.stages.validators import VerificationResult
from fastvideo.logger import init_logger
from fastvideo.utils import PRECISION_TO_TYPE

BASE_SHIFT_ANCHOR = 1024
MAX_SHIFT_ANCHOR = 4096

logger = init_logger(__name__)


def _log_non_finite(
    name: str,
    tensor: torch.Tensor | None,
    step_index: int | None = None,
) -> bool:
    if tensor is None:
        return False
    if torch.isfinite(tensor).all():
        return False
    non_finite = ~torch.isfinite(tensor)
    count = int(non_finite.sum().item())
    step_info = f" step={step_index}" if step_index is not None else ""
    logger.error(
        "[LTX2] Non-finite detected in %s%s: count=%d shape=%s dtype=%s",
        name,
        step_info,
        count,
        tuple(tensor.shape),
        tensor.dtype,
    )
    finite_vals = tensor[~non_finite]
    if finite_vals.numel() > 0:
        logger.error(
            "[LTX2] %s%s finite range: min=%s max=%s",
            name,
            step_info,
            finite_vals.min().item(),
            finite_vals.max().item(),
        )
    return True


def _log_tensor_stats(
    name: str,
    tensor: torch.Tensor | None,
    step_index: int | None = None,
) -> None:
    if tensor is None:
        return
    step_info = f" step={step_index}" if step_index is not None else ""
    logger.info(
        "[LTX2] %s%s stats: shape=%s dtype=%s min=%s max=%s mean=%s",
        name,
        step_info,
        tuple(tensor.shape),
        tensor.dtype,
        tensor.min().item(),
        tensor.max().item(),
        tensor.mean().item(),
    )


def _ltx2_sigmas(
    steps: int,
    latent: torch.Tensor | None,
    device: torch.device,
    max_shift: float = 2.05,
    base_shift: float = 0.95,
    stretch: bool = True,
    terminal: float = 0.1,
) -> torch.Tensor:
    tokens = math.prod(
        latent.shape[2:]) if latent is not None else MAX_SHIFT_ANCHOR
    sigmas = torch.linspace(1.0,
                            0.0,
                            steps + 1,
                            device=device,
                            dtype=torch.float32)

    mm = (max_shift - base_shift) / (MAX_SHIFT_ANCHOR - BASE_SHIFT_ANCHOR)
    b = base_shift - mm * BASE_SHIFT_ANCHOR
    sigma_shift = tokens * mm + b

    numerator = math.exp(sigma_shift)
    sigmas = torch.where(
        sigmas != 0,
        numerator / (numerator + (1 / sigmas - 1)),
        torch.zeros_like(sigmas),
    )

    if stretch:
        non_zero_mask = sigmas != 0
        non_zero_sigmas = sigmas[non_zero_mask]
        one_minus_z = 1.0 - non_zero_sigmas
        scale_factor = one_minus_z[-1] / (1.0 - terminal)
        stretched = 1.0 - (one_minus_z / scale_factor)
        sigmas = sigmas.clone()
        sigmas[non_zero_mask] = stretched

    return sigmas


class LTX2DenoisingStage(PipelineStage):
    """Run the LTX-2 denoising loop over the sigma schedule."""

    def __init__(self, transformer) -> None:
        super().__init__()
        self.transformer = transformer

    def forward(
        self,
        batch: ForwardBatch,
        fastvideo_args: FastVideoArgs,
    ) -> ForwardBatch:
        debug_nans = os.getenv("LTX2_DEBUG_NANS", "0") == "1"
        abort_on_nans = os.getenv("LTX2_DEBUG_NANS_ABORT", "0") == "1"
        verbose_logs = os.getenv("LTX2_DEBUG_DENOISE_LOG", "0") == "1"
        step_log_interval = int(
            os.getenv("LTX2_DEBUG_DENOISE_LOG_INTERVAL", "1"))
        if batch.latents is None:
            raise ValueError("Latents must be provided before denoising.")

        latents = batch.latents
        prompt_embeds = batch.prompt_embeds[0]
        prompt_mask = (batch.prompt_attention_mask[0]
                       if batch.prompt_attention_mask else None)

        neg_prompt_embeds = None
        neg_prompt_mask = None
        if batch.do_classifier_free_guidance:
            assert batch.negative_prompt_embeds is not None
            neg_prompt_embeds = batch.negative_prompt_embeds[0]
            if batch.negative_attention_mask:
                neg_prompt_mask = batch.negative_attention_mask[0]

        # Ensure text conditioning is on the same device as latents.
        if prompt_embeds.device != latents.device:
            prompt_embeds = prompt_embeds.to(latents.device)
        if prompt_mask is not None and prompt_mask.device != latents.device:
            prompt_mask = prompt_mask.to(latents.device)
        if neg_prompt_embeds is not None and neg_prompt_embeds.device != latents.device:
            neg_prompt_embeds = neg_prompt_embeds.to(latents.device)
        if neg_prompt_mask is not None and neg_prompt_mask.device != latents.device:
            neg_prompt_mask = neg_prompt_mask.to(latents.device)

        target_dtype = PRECISION_TO_TYPE[
            fastvideo_args.pipeline_config.dit_precision]
        autocast_enabled = (target_dtype != torch.float32
                            ) and not fastvideo_args.disable_autocast

        sigmas = _ltx2_sigmas(
            steps=batch.num_inference_steps,
            latent=latents,
            device=latents.device,
        )
        logger.info(
            "[LTX2] Denoising start: steps=%d dtype=%s guidance=%s "
            "sigmas_shape=%s latents_shape=%s",
            batch.num_inference_steps,
            target_dtype,
            batch.guidance_scale,
            tuple(sigmas.shape),
            tuple(latents.shape),
        )

        if debug_nans:
            if _log_non_finite("latents_start", latents):
                if abort_on_nans:
                    raise RuntimeError("Non-finite latents before denoising.")
            if _log_non_finite("prompt_embeds", prompt_embeds):
                if abort_on_nans:
                    raise RuntimeError("Non-finite prompt embeddings.")
            if neg_prompt_embeds is not None and _log_non_finite(
                    "negative_prompt_embeds", neg_prompt_embeds):
                if abort_on_nans:
                    raise RuntimeError("Non-finite negative prompt embeddings.")
        if verbose_logs:
            _log_tensor_stats("latents_start", latents)
            _log_tensor_stats("prompt_embeds", prompt_embeds)
            if neg_prompt_embeds is not None:
                _log_tensor_stats("negative_prompt_embeds", neg_prompt_embeds)

        for step_index in tqdm(range(len(sigmas) - 1)):
            sigma = sigmas[step_index]
            sigma_next = sigmas[step_index + 1]
            if verbose_logs and step_log_interval > 0 and (
                    step_index % step_log_interval == 0):
                logger.info(
                    "[LTX2] Step %d/%d sigma=%s sigma_next=%s",
                    step_index + 1,
                    len(sigmas) - 1,
                    sigma.item(),
                    sigma_next.item(),
                )
            timestep = torch.full(
                (latents.shape[0], 1),
                sigma.item(),
                device=latents.device,
                dtype=target_dtype,
            )

            with torch.autocast(
                    device_type="cuda",
                    dtype=target_dtype,
                    enabled=autocast_enabled,
            ), set_forward_context(
                    current_timestep=sigma.item(),
                    attn_metadata=None,
                    forward_batch=batch,
            ):
                pos_denoised = self.transformer(
                    hidden_states=latents.to(target_dtype),
                    encoder_hidden_states=prompt_embeds,
                    encoder_attention_mask=prompt_mask,
                    timestep=timestep,
                )
                if neg_prompt_embeds is not None:
                    neg_denoised = self.transformer(
                        hidden_states=latents.to(target_dtype),
                        encoder_hidden_states=neg_prompt_embeds,
                        encoder_attention_mask=neg_prompt_mask,
                        timestep=timestep,
                    )
                    pos_denoised = pos_denoised + (batch.guidance_scale - 1) * (
                        pos_denoised - neg_denoised)

            if debug_nans:
                if _log_non_finite("pos_denoised", pos_denoised, step_index):
                    if abort_on_nans:
                        raise RuntimeError(
                            f"Non-finite pos_denoised at step {step_index}.")
                if neg_prompt_embeds is not None and _log_non_finite(
                        "neg_denoised", neg_denoised, step_index):
                    if abort_on_nans:
                        raise RuntimeError(
                            f"Non-finite neg_denoised at step {step_index}.")
            if verbose_logs and step_log_interval > 0 and (
                    step_index % step_log_interval == 0):
                _log_tensor_stats("pos_denoised", pos_denoised, step_index)
                if neg_prompt_embeds is not None:
                    _log_tensor_stats("neg_denoised", neg_denoised, step_index)

            velocity = (latents.float() - pos_denoised.float()) / sigma
            latents = (latents.float() + velocity * (sigma_next - sigma)).to(
                latents.dtype)
            if debug_nans and _log_non_finite("latents", latents, step_index):
                if abort_on_nans:
                    raise RuntimeError(
                        f"Non-finite latents at step {step_index}.")
            if verbose_logs and step_log_interval > 0 and (
                    step_index % step_log_interval == 0):
                _log_tensor_stats("latents", latents, step_index)

        batch.latents = latents
        logger.info("[LTX2] Denoising done.")
        return batch

    def verify_input(self, batch: ForwardBatch,
                     fastvideo_args: FastVideoArgs) -> VerificationResult:
        result = VerificationResult()
        result.add_check("latents", batch.latents,
                         [V.is_tensor, V.with_dims(5)])
        result.add_check("prompt_embeds", batch.prompt_embeds, V.list_not_empty)
        result.add_check("num_inference_steps", batch.num_inference_steps,
                         V.positive_int)
        return result
