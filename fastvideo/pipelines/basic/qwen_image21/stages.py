# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Qwen Team and The HuggingFace Team. All rights reserved.
# Adapted from Diffusers' Apache-2.0 QwenImage21Pipeline.
"""Separate preparation, encoding, scheduling, denoising and decoding stages."""

from __future__ import annotations

import numpy as np
import torch
from torch.distributed.tensor import DTensor
from tqdm.auto import tqdm

from fastvideo.distributed import get_local_torch_device
from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.hooks.hooks import ModuleHookManager
from fastvideo.logger import init_logger
from fastvideo.models import pinned_offload
from fastvideo.models.dits.qwen_image21 import QwenImage21KVCache, QwenImage21KVCacheAllocationError
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.pipelines.stages.base import PipelineStage
from fastvideo.pipelines.stages.validators import V, VerificationResult

from .inputs import (
    STATE_KEY,
    SYSTEM_PROMPT,
    QwenImage21State,
    collect_references,
    extract_prompt,
    normalize_latents,
    pack_latents,
    prompt_template,
    reference_pixels,
    resize_reference,
    schedule_shift,
    unpack_latents,
    validate_request,
    vision_reference,
)

logger = init_logger(__name__)


def _state(batch: ForwardBatch) -> QwenImage21State:
    state = batch.extra.get(STATE_KEY)
    if not isinstance(state, QwenImage21State):
        raise ValueError("Qwen-Image-2.1 input preparation must precede this stage")
    return state


class QwenImage21InputStage(PipelineStage):

    def verify_input(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> VerificationResult:
        result = VerificationResult()
        result.add_check("height", batch.height, V.positive_int_divisible(32))
        result.add_check("width", batch.width, V.positive_int_divisible(32))
        result.add_check("num_frames", batch.num_frames, lambda value: value == 1)
        return result

    def verify_output(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> VerificationResult:
        return VerificationResult().add_check(STATE_KEY, batch.extra.get(STATE_KEY),
                                              lambda value: isinstance(value, QwenImage21State))

    def forward(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> ForwardBatch:
        validate_request(batch, fastvideo_args)
        references = collect_references(batch.references, batch.image_path, batch.pil_image)
        references = [resize_reference(image, batch.reference_resolution) for image in references]
        shapes = [(1, image.height // 16, image.width // 16) for image in references]
        shapes.append((1, batch.height // 16, batch.width // 16))
        batch.extra[STATE_KEY] = QwenImage21State(references=references, img_shapes=[shapes])
        batch.batch_size = 1
        batch.do_classifier_free_guidance = batch.true_cfg_scale > 1 and batch.negative_prompt is not None
        if batch.true_cfg_scale > 1 and not batch.do_classifier_free_guidance:
            logger.warning("true_cfg_scale > 1 requires an explicit negative_prompt; guidance is disabled")
        return batch


class QwenImage21TextEncodingStage(PipelineStage):
    performance_component_metric = "text_encoding"

    def __init__(self, text_encoder, processor):
        self.text_encoder = text_encoder
        self.processor = processor

    def verify_input(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> VerificationResult:
        return VerificationResult().add_check(STATE_KEY, batch.extra.get(STATE_KEY),
                                              lambda value: isinstance(value, QwenImage21State))

    def verify_output(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> VerificationResult:
        return VerificationResult().add_check("prompt_embeds", batch.prompt_embeds,
                                              lambda value: len(value) == 1 and value[0].ndim == 3)

    def _encode(self, prompt: str, state: QwenImage21State, device: torch.device):
        kwargs = dict(text=[prompt_template(prompt, len(state.references))],
                      padding=True,
                      padding_side="left",
                      return_tensors="pt")
        if state.references:
            kwargs["images"] = [vision_reference(image) for image in state.references]
        model_inputs = self.processor(**kwargs)
        inputs = {key: value.to(device) for key, value in model_inputs.items() if isinstance(value, torch.Tensor)}
        hidden = self.text_encoder(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            pixel_values=inputs.get("pixel_values"),
            image_grid_thw=inputs.get("image_grid_thw"),
            mm_token_type_ids=inputs.get("mm_token_type_ids"),
        )
        system = [{"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]}]
        system_ids = self.processor.apply_chat_template(system, tokenize=True, return_dict=False)
        drop_tokens = len(system_ids[0])
        image_token_id = self.processor.tokenizer.encode("<|image_pad|>")[0]
        embeds, mask, image_mask = extract_prompt(hidden, inputs["input_ids"], inputs["attention_mask"], drop_tokens,
                                                  image_token_id)
        return embeds, None if bool(mask.all()) else mask, image_mask

    @torch.no_grad()
    def forward(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> ForwardBatch:
        state = _state(batch)
        device = get_local_torch_device()
        first_param = next(self.text_encoder.parameters(), None)
        # The loader's FSDP2 CPU offload already streams individual text layers.
        move = (fastvideo_args.text_encoder_cpu_offload and first_param is not None
                and not isinstance(first_param, DTensor))
        if move:
            pinned_offload.load(self.text_encoder, device, pin=fastvideo_args.pin_cpu_memory)
        try:
            prompt = batch.prompt[0] if isinstance(batch.prompt, list) else batch.prompt
            if prompt is None:
                raise ValueError("A text prompt is required")
            embeds, mask, state.image_pad_mask = self._encode(prompt, state, device)
            batch.prompt_embeds = [embeds]
            batch.prompt_attention_mask = [mask]
            if batch.do_classifier_free_guidance:
                if batch.negative_prompt is None:
                    raise ValueError("True CFG requires an explicit negative prompt")
                embeds, mask, state.negative_image_pad_mask = self._encode(batch.negative_prompt, state, device)
                batch.negative_prompt_embeds = [embeds]
                batch.negative_attention_mask = [mask]
        finally:
            if move:
                pinned_offload.unload(self.text_encoder)
        return batch


class QwenImage21LatentStage(PipelineStage):
    performance_component_metric = "vae_encoding"

    def __init__(self, vae):
        self.vae = vae

    def verify_input(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> VerificationResult:
        return VerificationResult().add_check("prompt_embeds", batch.prompt_embeds, lambda value: len(value) == 1)

    def verify_output(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> VerificationResult:
        return VerificationResult().add_check("latents", batch.latents, V.with_dims(3))

    @torch.no_grad()
    def forward(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> ForwardBatch:
        state = _state(batch)
        device, dtype = get_local_torch_device(), batch.prompt_embeds[0].dtype
        channels = fastvideo_args.pipeline_config.dit_config.arch_config.in_channels
        shape = (1, channels, 1, batch.height // 16, batch.width // 16)
        if batch.latents is None:
            generator = batch.generator
            if isinstance(generator, list):
                if len(generator) != 1:
                    raise ValueError("Qwen-Image-2.1 requires one generator")
                generator = generator[0]
            if generator is None:
                generator = torch.Generator(device="cpu").manual_seed(batch.seed or 0)
            noise_device = generator.device
            batch.latents = pack_latents(
                torch.randn(shape, generator=generator, device=noise_device, dtype=dtype).to(device))
        else:
            if batch.latents.ndim == 5:
                batch.latents = pack_latents(batch.latents)
            if tuple(batch.latents.shape) != (1, shape[3] * shape[4], channels):
                raise ValueError("Supplied latents must match [1, target tokens, latent channels]")
            batch.latents = batch.latents.to(device=device, dtype=dtype)
        batch.raw_latent_shape = shape
        if state.references:
            pinned_offload.load(self.vae, device, pin=fastvideo_args.pin_cpu_memory)
            try:
                if fastvideo_args.pipeline_config.vae_tiling:
                    self.vae.enable_tiling()
                else:
                    self.vae.disable_tiling()
                vae_dtype = next(self.vae.parameters()).dtype
                rows = []
                for image in state.references:
                    pixels = reference_pixels(image).to(device=device, dtype=vae_dtype)
                    latents = self.vae.encode(pixels).latent_dist.mode()
                    latents = normalize_latents(latents, self.vae.config.latents_mean, self.vae.config.latents_std)
                    rows.append(pack_latents(latents).to(dtype))
                batch.image_latent = torch.cat(rows, dim=1)
                expected = int(state.image_pad_mask.sum().item()) * 4
                if expected != batch.image_latent.shape[1]:
                    raise ValueError("The processor's image slots do not match the VAE reference grid. "
                                     "Use the checkpoint's processor and a reference_resolution "
                                     "within its image budget.")
            finally:
                if fastvideo_args.vae_cpu_offload:
                    pinned_offload.unload(self.vae)
        # One target mask slot stands for four unpatched latent tokens.
        slots = state.image_pad_mask.new_ones(1, batch.latents.shape[1] // 4)
        state.image_pad_mask = torch.cat((state.image_pad_mask, slots), dim=1)
        if state.negative_image_pad_mask is not None:
            slots = state.negative_image_pad_mask.new_ones(1, batch.latents.shape[1] // 4)
            state.negative_image_pad_mask = torch.cat((state.negative_image_pad_mask, slots), dim=1)
        return batch


class QwenImage21ScheduleStage(PipelineStage):

    def __init__(self, scheduler):
        self.scheduler = scheduler

    def verify_input(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> VerificationResult:
        return VerificationResult().add_check("latents", batch.latents, V.with_dims(3))

    def verify_output(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> VerificationResult:
        return VerificationResult().add_check("timesteps", batch.timesteps, V.with_dims(1))

    def forward(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> ForwardBatch:
        sigmas = batch.sigmas.tolist() if isinstance(batch.sigmas, torch.Tensor) else batch.sigmas
        if sigmas is None:
            sigmas = np.linspace(1.0, 1 / batch.num_inference_steps, batch.num_inference_steps).tolist()
        if not sigmas or any(not np.isfinite(sigma) or sigma <= 0 or sigma > 1 for sigma in sigmas):
            raise ValueError("sigmas must be finite values in (0, 1]")
        if any(first <= second for first, second in zip(sigmas, sigmas[1:], strict=False)):
            raise ValueError("sigmas must be strictly decreasing")
        self.scheduler.set_timesteps(sigmas=sigmas,
                                     device=get_local_torch_device(),
                                     mu=schedule_shift(batch.latents.shape[1], self.scheduler.config))
        self.scheduler.set_begin_index(0)
        batch.timesteps = self.scheduler.timesteps
        batch.num_inference_steps = len(batch.timesteps)
        return batch


class QwenImage21DenoisingStage(PipelineStage):
    performance_component_metric = "denoising"

    def __init__(self, transformer, scheduler):
        self.transformer, self.scheduler = transformer, scheduler

    def verify_input(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> VerificationResult:
        return VerificationResult().add_check("latents", batch.latents,
                                              V.with_dims(3)).add_check("timesteps", batch.timesteps, V.with_dims(1))

    def verify_output(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> VerificationResult:
        return VerificationResult().add_check("latents", batch.latents, V.with_dims(3))

    @torch.no_grad()
    def forward(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> ForwardBatch:
        state, device = _state(batch), get_local_torch_device()
        cache_device = fastvideo_args.pipeline_config.kv_cache_device
        if cache_device not in ("cpu", "cuda"):
            raise ValueError("kv_cache_device must be cpu or cuda; use_kv_cache=False disables caching")
        cache_enabled = batch.use_kv_cache and self.transformer.config.causal_condition
        caches = []
        if cache_enabled:
            caches = [QwenImage21KVCache(len(self.transformer.transformer_blocks), storage_device=cache_device)]
            if batch.do_classifier_free_guidance:
                caches.append(QwenImage21KVCache(len(self.transformer.transformer_blocks), storage_device=cache_device))
        move = (fastvideo_args.dit_cpu_offload and not fastvideo_args.dit_layerwise_offload
                and not fastvideo_args.use_fsdp_inference)
        try:
            if move:
                pinned_offload.load(self.transformer, device, pin=fastvideo_args.pin_cpu_memory)
            for index, timestep in enumerate(tqdm(batch.timesteps, desc="Denoising", total=len(batch.timesteps))):
                latent_input = batch.latents
                if batch.image_latent is not None:
                    latent_input = torch.cat((batch.image_latent, latent_input), dim=1)
                mode = "extract" if cache_enabled and index == 0 else ("cached" if cache_enabled else None)

                def predict(embeds: torch.Tensor, mask: torch.Tensor | None, image_mask: torch.Tensor,
                            cache: QwenImage21KVCache | None, latent_input: torch.Tensor, timestep: torch.Tensor,
                            mode: str | None) -> torch.Tensor:
                    result = self.transformer(
                        hidden_states=latent_input,
                        timestep=timestep.expand(1).to(batch.latents.dtype) / 1000,
                        encoder_hidden_states=embeds,
                        encoder_hidden_states_mask=mask,
                        img_shapes=state.img_shapes,
                        img_mask=image_mask,
                        kv_cache=cache,
                        kv_cache_mode=mode,
                        return_dict=False,
                    )[0]
                    return result[:, -batch.latents.shape[1]:]

                while True:
                    try:
                        positive = predict(batch.prompt_embeds[0], batch.prompt_attention_mask[0], state.image_pad_mask,
                                           caches[0] if caches else None, latent_input, timestep, mode)
                        noise = positive
                        if batch.do_classifier_free_guidance:
                            negative = predict(batch.negative_prompt_embeds[0], batch.negative_attention_mask[0],
                                               state.negative_image_pad_mask, caches[1] if caches else None,
                                               latent_input, timestep, mode)
                            noise = negative + batch.true_cfg_scale * (positive - negative)
                        break
                    except QwenImage21KVCacheAllocationError:
                        if not cache_enabled:
                            raise
                        logger.warning("Host prefix-cache allocation failed; retrying this request without KV caching")
                        for cache in caches:
                            cache.clear()
                        caches = []
                        cache_enabled, mode = False, None
                        # An aborted layer may retain the next layer's prefetch.
                        self._release_layerwise()
                batch.latents = self.scheduler.step(noise, timestep, batch.latents, return_dict=False)[0]
                batch.step_index, batch.timestep = index, timestep
        finally:
            for cache in caches:
                cache.clear()
            if fastvideo_args.dit_layerwise_offload:
                self._release_layerwise()
            if move:
                pinned_offload.unload(self.transformer)
        return batch

    def _release_layerwise(self) -> None:
        # The shared offloader links the last block back to the first, leaving
        # its next-step prefetch resident. Also handle an interrupted block.
        for block in self.transformer.transformer_blocks:
            manager = ModuleHookManager.get_from(block)
            hook = manager.get_forward_hook("LayerwiseOffloadHook") if manager is not None else None
            if hook is not None and hook.state.gpu_named_parameters:
                torch.cuda.current_stream().wait_stream(hook.state.async_copy_stream)
                hook.state.release_gpu_params()


class QwenImage21DecodingStage(PipelineStage):
    performance_component_metric = "vae_decoding"

    def __init__(self, vae):
        self.vae = vae

    def verify_input(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> VerificationResult:
        return VerificationResult().add_check("latents", batch.latents, V.with_dims(3))

    def verify_output(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> VerificationResult:
        return VerificationResult().add_check("output", batch.output, V.is_tensor)

    @torch.no_grad()
    def forward(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> ForwardBatch:
        if fastvideo_args.output_type == "latent":
            batch.output = batch.latents
            return batch
        device = get_local_torch_device()
        pinned_offload.load(self.vae, device, pin=fastvideo_args.pin_cpu_memory)
        try:
            if fastvideo_args.pipeline_config.vae_tiling:
                self.vae.enable_tiling()
            else:
                self.vae.disable_tiling()
            dtype = next(self.vae.parameters()).dtype
            latents = unpack_latents(batch.latents, batch.height, batch.width).to(device=device, dtype=dtype)
            latents = normalize_latents(latents,
                                        self.vae.config.latents_mean,
                                        self.vae.config.latents_std,
                                        inverse=True)
            # Keep alpha as the fourth channel all the way to the PNG exporter.
            batch.output = self.vae.decode(latents, return_dict=False)[0].div(2).add(0.5).clamp(0, 1).float()
        finally:
            if fastvideo_args.vae_cpu_offload:
                pinned_offload.unload(self.vae)
        return batch
