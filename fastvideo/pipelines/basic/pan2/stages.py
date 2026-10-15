# SPDX-License-Identifier: Apache-2.0
"""PAN2-specific pipeline stages: prompt encoding, first-frame conditioning, sigma schedule and fp32 sampling."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import PIL.Image
import torch
import torchvision.transforms.functional as TF
from torch.distributed.tensor import DTensor

from fastvideo.distributed import get_local_torch_device
from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.forward_context import set_forward_context
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.pipelines.stages.base import PipelineStage
from fastvideo.pipelines.stages.denoising import DenoisingStage
from fastvideo.pipelines.stages.input_validation import InputValidationStage
from fastvideo.pipelines.stages.text_encoding import TextEncodingStage
from fastvideo.pipelines.stages.timestep_preparation import TimestepPreparationStage
from fastvideo.pipelines.stages.validators import StageValidators as V
from fastvideo.pipelines.stages.validators import VerificationResult
from fastvideo.utils import PRECISION_TO_TYPE

_USER_HEADER = "<|im_start|>user\n"


class PAN2InputValidationStage(InputValidationStage):
    """Rejects geometry the PAN2 VAE and patch embedding cannot represent."""

    def forward(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> ForwardBatch:
        batch = super().forward(batch, fastvideo_args)
        pipeline_config = fastvideo_args.pipeline_config
        vae_arch = pipeline_config.vae_config.arch_config
        spatial_multiple = vae_arch.spatial_compression_ratio * pipeline_config.dit_config.arch_config.patch_size[1]
        if batch.height % spatial_multiple != 0 or batch.width % spatial_multiple != 0:
            raise ValueError(f"PAN2 height and width must be divisible by {spatial_multiple}, "
                             f"got {batch.height} and {batch.width}.")
        if (batch.num_frames - 1) % vae_arch.temporal_compression_ratio != 0:
            raise ValueError(f"PAN2 num_frames - 1 must be divisible by {vae_arch.temporal_compression_ratio}, "
                             f"got {batch.num_frames}.")
        if batch.pil_image is not None and not isinstance(batch.pil_image, PIL.Image.Image):
            raise ValueError("PAN2 image-to-video expects the first frame as a PIL image or an image_path.")
        return batch


class PAN2TextEncodingStage(TextEncodingStage):
    """Encodes a prompt with PAN2's chat template.

    PAN2 conditions on the tokens after the ``<|im_start|>user\\n`` header (the prompt and the assistant header),
    read from the text encoder's hidden states with its last ``text_encoder_skip_layers`` layers skipped.
    """

    @staticmethod
    def tokenize(tokenizer: Any, prompt: str, system_prompt: str, max_length: int) -> tuple[torch.Tensor, int]:
        """Token ids truncated to ``max_length`` tokens after the user header, and where that header ends."""
        # An empty prompt is encoded as a single space, as in training.
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt or " "}]
        input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True,
                                                  return_dict=True)["input_ids"]
        if isinstance(input_ids, torch.Tensor):
            input_ids = input_ids.tolist()
        if input_ids and isinstance(input_ids[0], list):
            input_ids = input_ids[0]
        header_ids = tokenizer(_USER_HEADER, add_special_tokens=False)["input_ids"]
        crop_start = next((i + len(header_ids)
                           for i in range(len(input_ids)) if input_ids[i:i + len(header_ids)] == header_ids), None)
        if crop_start is None:
            raise ValueError(f"The PAN2 tokenizer's chat template has no {_USER_HEADER!r} header.")
        return torch.tensor([input_ids[:crop_start + max_length]], dtype=torch.long), crop_start

    @torch.no_grad()
    def encode_text(
        self,
        text: str | list[str],
        fastvideo_args: FastVideoArgs,
        encoder_index: int | list[int] | None = None,
        return_attention_mask: bool = False,
        return_type: str = "list",
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
        max_length: int | Sequence[int | None] | None = None,
        truncation: bool | None = None,
        padding: bool | str | None = None,
    ):
        pipeline_config = fastvideo_args.pipeline_config
        texts = [text] if isinstance(text, str) else list(text)
        if len(texts) != 1:
            raise ValueError(f"PAN2 encodes one prompt per request, got {len(texts)}.")
        if isinstance(max_length, Sequence):
            max_length = max_length[0]
        max_length = max_length or pipeline_config.text_max_sequence_length

        tokenizer = self.tokenizers[0]
        text_encoder = self.text_encoders[0]
        target_device = torch.device(device) if device is not None else get_local_torch_device()
        # A module parked on CPU by text_encoder_cpu_offload computes on the target device and goes back afterwards;
        # FSDP-wrapped (DTensor) encoders move their own shards.
        first_param = next(text_encoder.parameters(), None)
        input_device = first_param.device if first_param is not None else target_device
        moved_for_forward = False
        if (first_param is not None and not isinstance(first_param, DTensor)
                and input_device.type != target_device.type):
            text_encoder = text_encoder.to(target_device)
            input_device = target_device
            moved_for_forward = True

        input_ids, crop_start = self.tokenize(tokenizer, texts[0], pipeline_config.text_system_prompt, max_length)
        with set_forward_context(current_timestep=0, attn_metadata=None):
            outputs = text_encoder(input_ids=input_ids.to(input_device), output_hidden_states=True)
        prompt_embeds = outputs.hidden_states[-(pipeline_config.text_encoder_skip_layers + 1)][:, crop_start:]
        # fp32 is exact for the bf16 encoder output. The shared latent stage draws the initial noise in this dtype,
        # and PAN2 samples on fp32 latents; the denoiser casts the embeddings to the DiT dtype.
        prompt_embeds = prompt_embeds.to(device=target_device, dtype=dtype or torch.float32, copy=True)
        attention_mask = torch.ones(prompt_embeds.shape[:2], dtype=torch.long, device=target_device)
        if moved_for_forward and fastvideo_args.text_encoder_cpu_offload:
            text_encoder.to("cpu")
        return self.return_embeds([prompt_embeds], [attention_mask], return_type, return_attention_mask, [0])


class PAN2TimestepPreparationStage(TimestepPreparationStage):
    """PAN2 samples on uniformly spaced sigmas from 1 down to (excluding) 0; the scheduler applies the shift."""

    def forward(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> ForwardBatch:
        if batch.timesteps is None and batch.sigmas is None:
            batch.sigmas = np.linspace(1.0, 0.0, batch.num_inference_steps + 1)[:-1].tolist()
        return super().forward(batch, fastvideo_args)


class PAN2ConditioningStage(PipelineStage):
    """Builds the conditioning latents and mask that PAN2 stacks after the noisy latents along channels.

    Text-to-video conditions on zeros. Image-to-video fills the first latent frame with the normalized VAE latents
    of the first frame, scaled to cover the output and center cropped, and sets that frame's mask channel to one.
    """

    def __init__(self, vae) -> None:
        super().__init__()
        self.vae = vae

    @staticmethod
    def preprocess_image(image: PIL.Image.Image, height: int, width: int) -> torch.Tensor:
        """``[1, 3, 1, H, W]`` pixels in ``[-1, 1]``: scale to cover the canvas, round the size, center crop."""
        image = image.convert("RGB")
        scale = max(width / image.width, height / image.height)
        image = image.resize((round(image.width * scale), round(image.height * scale)), resample=PIL.Image.LANCZOS)
        left, top = round((image.width - width) / 2), round((image.height - height) / 2)
        image = image.crop((left, top, left + width, top + height))
        return (TF.to_tensor(image) * 2.0 - 1.0)[None, :, None]

    def _encode_first_frame(self, image: PIL.Image.Image, batch: ForwardBatch,
                            fastvideo_args: FastVideoArgs) -> torch.Tensor:
        device = get_local_torch_device()
        vae_dtype = PRECISION_TO_TYPE[fastvideo_args.pipeline_config.vae_precision]
        self.vae = self.vae.to(device)
        pixels = self.preprocess_image(image, batch.height, batch.width).to(device=device, dtype=vae_dtype)
        latents = self.vae.encode(pixels).mode().float()
        shift = self.vae.shift_factor.to(latents.device, latents.dtype)
        scale = self.vae.scaling_factor.to(latents.device, latents.dtype)
        if fastvideo_args.vae_cpu_offload:
            self.vae.to("cpu")
        return (latents - shift) * scale

    @torch.no_grad()
    def forward(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> ForwardBatch:
        latents = batch.latents
        batch_size, num_channels, num_frames, height, width = latents.shape
        condition = torch.zeros(batch_size,
                                num_channels + 1,
                                num_frames,
                                height,
                                width,
                                device=latents.device,
                                dtype=torch.float32)
        if batch.pil_image is not None:
            image_latents = self._encode_first_frame(batch.pil_image, batch, fastvideo_args)
            condition[:, :num_channels, :1] = image_latents.to(condition.device)
            condition[:, num_channels:, :1] = 1.0
        batch.image_latent = condition
        return batch

    def verify_input(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> VerificationResult:
        result = VerificationResult()
        result.add_check("latents", batch.latents, [V.is_tensor, V.with_dims(5)])
        result.add_check("height", batch.height, V.positive_int)
        result.add_check("width", batch.width, V.positive_int)
        return result

    def verify_output(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> VerificationResult:
        result = VerificationResult()
        result.add_check("image_latent", batch.image_latent, [V.is_tensor, V.with_dims(5)])
        return result


class PAN2DenoisingStage(DenoisingStage):
    """Keeps the latents in fp32 across the loop; they are cast to the DiT dtype for each forward."""

    def prepare_scheduler_model_output(self, noise_pred, state) -> torch.Tensor:
        return noise_pred.float()
