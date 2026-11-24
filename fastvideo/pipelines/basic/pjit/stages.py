# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Linum Inc.
# Sampling arithmetic adapted from https://github.com/Linum-AI/pjit-21k/blob/main/pjit_21k/sampler.py
"""P-JiT caption conditioning and pixel-space Euler sampling."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.distributed.tensor import DTensor

from fastvideo.configs.pipelines.pjit import PJiTPipelineConfig
from fastvideo.distributed import get_local_torch_device
from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.forward_context import set_forward_context
from fastvideo.models.encoders.pjit_21k import PJiTTextConditioner
from fastvideo.pipelines.basic.pjit.noise import pixel_noise
from fastvideo.pipelines.lazy_module import is_lazy_module
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.pipelines.stages.base import PipelineStage
from fastvideo.pipelines.stages.validators import StageValidators as V
from fastvideo.pipelines.stages.validators import VerificationResult


class PJiTTextEncodingStage(PipelineStage):
    performance_component_metric = "text_encoder_time_s"

    def __init__(self, text_encoder, tokenizer):
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer

    def verify_input(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> VerificationResult:
        result = VerificationResult()
        if not batch.prompt_embeds:
            result.add_check("prompt", batch.prompt, V.string_or_list_strings)
        return result

    def verify_output(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> VerificationResult:
        result = VerificationResult()
        result.add_check("prompt_embeds", batch.prompt_embeds, V.list_of_tensors_min_dims(2))
        result.add_check("negative_prompt_embeds", batch.negative_prompt_embeds, V.list_of_tensors_min_dims(2))
        return result

    @torch.inference_mode()
    def forward(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> ForwardBatch:
        if batch.num_frames != 1 or batch.num_videos_per_prompt != 1:
            raise ValueError("P-JiT generates one image per request (num_frames=1, num_videos_per_prompt=1).")
        if batch.image_path or batch.video_path or batch.pil_image is not None:
            raise ValueError("P-JiT accepts text-to-image requests.")
        if not isinstance(batch.height, int) or not isinstance(batch.width, int) or min(batch.height, batch.width) < 1:
            raise ValueError("P-JiT requires positive image dimensions.")
        if batch.num_inference_steps < 1:
            raise ValueError("P-JiT requires at least one Euler step.")
        arch = fastvideo_args.pipeline_config.dit_config.arch_config
        device = get_local_torch_device()
        if batch.prompt_embeds and batch.negative_prompt_embeds:
            return batch
        param = next(self.text_encoder.parameters(), None)
        move = param is not None and not isinstance(param, DTensor) and param.device.type != device.type
        if move:
            self.text_encoder.to(device)
        try:
            encode = PJiTTextConditioner(self.text_encoder,
                                         self.tokenizer,
                                         arch.text_layers,
                                         arch.text_len,
                                         device=device)
            # Linum runs Qwen twice with no batch padding, including the empty negative caption.
            with set_forward_context(current_timestep=0, attn_metadata=None):
                if not batch.prompt_embeds:
                    batch.prompt_embeds = encode([_one_caption(batch.prompt)])
                if not batch.negative_prompt_embeds:
                    batch.negative_prompt_embeds = encode([_one_caption(batch.negative_prompt or "")])
        finally:
            if move and fastvideo_args.text_encoder_cpu_offload:
                self.text_encoder.to("cpu")
        return batch


def _one_caption(prompt: str | list[str] | None) -> str:
    if isinstance(prompt, list) and len(prompt) == 1:
        prompt = prompt[0]
    if not isinstance(prompt, str):
        raise ValueError("P-JiT requires one caption per request.")
    return prompt


def _pad_captions(embeddings: list[torch.Tensor], length: int,
                  device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    rows = []
    lengths = []
    for embedding in embeddings:
        if embedding.ndim == 3 and embedding.shape[0] == 1:
            embedding = embedding[0]  # Public precomputed prompt_embeds can include the batch dimension.
        if embedding.ndim != 2 or embedding.shape[0] > length:
            raise ValueError(f"Expected a P-JiT caption embedding [L, D] with L <= {length}, got {embedding.shape}.")
        lengths.append(embedding.shape[0])
        rows.append(F.pad(embedding, (0, 0, 0, length - embedding.shape[0])))
    return torch.stack(rows).to(device), torch.tensor(lengths, dtype=torch.int32, device=device)


def _apg_step(state: torch.Tensor, rounded: torch.Tensor, clean_cond: torch.Tensor, clean_uncond: torch.Tensor,
              t: torch.Tensor, t_next: torch.Tensor, previous_momentum: torch.Tensor, guidance: float,
              config: PJiTPipelineConfig) -> tuple[torch.Tensor, torch.Tensor]:
    """APG acts on velocities of the rounded model input; Euler updates the fp32 state."""
    cond = (rounded - clean_cond.float()) / t
    uncond = (rounded - clean_uncond.float()) / t
    momentum = (cond - uncond) + config.apg_momentum * previous_momentum.bfloat16().float()
    diff = momentum
    if config.apg_rescale > 0:
        norm = diff.norm(p=2, dim=(1, 3, 4), keepdim=True)
        diff = diff * torch.minimum(torch.ones_like(norm), config.apg_rescale / norm)

    # The reference projects in fp64 across the full image. Metal lacks fp64 tensors.
    projection_device = torch.device("cpu") if state.device.type == "mps" else state.device
    direction = F.normalize(cond.to(projection_device).double(), dim=(1, 2, 3, 4))
    diff64 = diff.to(projection_device).double()
    parallel = (diff64 * direction).sum(dim=(1, 2, 3, 4), keepdim=True) * direction
    orthogonal = (diff64 - parallel).float().to(state.device)
    guided = cond + guidance * (orthogonal + config.apg_eta * parallel.float().to(state.device))
    clean = rounded - t * guided
    return state + (t_next - t) * ((state - clean) / t), momentum


def _pixels(state: torch.Tensor) -> torch.Tensor:
    return ((state.clamp(-1, 1) + 1) / 2 * 255).round().to(torch.uint8)


class PJiTDenoisingStage(PipelineStage):
    performance_component_metric = "dit_time_s"

    def __init__(self, transformer):
        self.transformer = transformer

    def verify_input(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> VerificationResult:
        result = VerificationResult()
        result.add_check("prompt_embeds", batch.prompt_embeds, V.list_not_empty)
        result.add_check("negative_prompt_embeds", batch.negative_prompt_embeds, V.list_not_empty)
        return result

    def verify_output(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> VerificationResult:
        result = VerificationResult()
        result.add_check("output", batch.output, V.with_dims(5))
        return result

    @torch.inference_mode()
    def forward(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> ForwardBatch:
        config = fastvideo_args.pipeline_config
        device = get_local_torch_device()
        if len(batch.prompt_embeds) != 1 or not batch.negative_prompt_embeds or len(batch.negative_prompt_embeds) != 1:
            raise ValueError("P-JiT requires one conditional and one unconditional caption embedding.")
        text, lengths = _pad_captions([batch.prompt_embeds[0], batch.negative_prompt_embeds[0]],
                                      config.dit_config.arch_config.text_len, device)
        if not isinstance(batch.height, int) or not isinstance(batch.width, int):
            raise ValueError("P-JiT requires a height and width.")
        shape = (1, 3, 1, batch.height, batch.width)
        if batch.latents is None:
            if batch.seed is None:
                raise ValueError("P-JiT requires a seed to draw initial noise.")
            state = config.noise_scale * pixel_noise(shape, batch.seed, device, config.noise_sm_count)
        else:
            # A supplied x(t=1) already includes noise_scale.
            if not isinstance(batch.latents, torch.Tensor) or batch.latents.shape != shape:
                raise ValueError(f"P-JiT initial latents must be a floating-point tensor of shape {shape}.")
            if not batch.latents.is_floating_point():
                raise ValueError("P-JiT initial latents must be floating-point pixel values.")
            state = batch.latents.to(device=device, dtype=torch.float32)
        momentum = torch.zeros_like(state)
        times = torch.linspace(1., 0., batch.num_inference_steps + 1, device=device, dtype=torch.float32)
        model_times = (times[:-1] * 1000).long()
        trajectory = []
        decoded = []
        offload = fastvideo_args.dit_cpu_offload and not is_lazy_module(self.transformer)
        if offload:
            self.transformer.to(device)
        try:
            for i in range(batch.num_inference_steps):
                if getattr(self, "interrupt", False):
                    raise InterruptedError("P-JiT generation interrupted")
                rounded = state.bfloat16().float()
                # Both predictions share a forward, conditional first, even at guidance scale one.
                with set_forward_context(current_timestep=i, attn_metadata=None,
                                         forward_batch=batch), torch.autocast(device_type=device.type,
                                                                              dtype=torch.bfloat16):
                    cond, uncond = self.transformer(x_t=torch.cat((rounded, rounded)),
                                                    t=model_times[i].expand(2),
                                                    text=text,
                                                    text_lens=lengths).chunk(2)
                state, momentum = _apg_step(state, rounded, cond, uncond, times[i], times[i + 1], momentum,
                                            batch.guidance_scale, config)
                if batch.return_trajectory_latents:
                    trajectory.append(state.cpu())
                if batch.return_trajectory_decoded:
                    decoded.append(_pixels(state).cpu().float().div_(255))
        finally:
            if offload:
                self.transformer.to("cpu")
        if trajectory:
            batch.trajectory_latents = torch.stack(trajectory, dim=1)
            batch.trajectory_timesteps = model_times.cpu()
        if decoded:
            batch.trajectory_decoded = decoded
        batch.output = _pixels(state)
        return batch
