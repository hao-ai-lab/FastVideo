# SPDX-License-Identifier: Apache-2.0
"""Reference-image and Qwen3.5 conditioning stages for MAGI-2."""

from __future__ import annotations

import json
from typing import Any

import torch
from diffusers.utils import load_image
from diffusers.video_processor import VideoProcessor
from PIL import Image

from fastvideo.distributed import get_sp_group
from fastvideo.fastvideo_args import FastVideoArgs, WorkloadType
from fastvideo.models.dits.magi2_runtime import psm
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.pipelines.stages.base import PipelineStage
from fastvideo.pipelines.stages.validators import VerificationResult


def _resizepad(image: Image.Image, target_height: int, target_width: int) -> Image.Image:
    """Fit an RGB image inside a white canvas with the official letterboxing rule."""
    width, height = image.size
    if width <= 0 or height <= 0:
        raise ValueError(
            f"MAGI-2 received an invalid image size: width={width}, height={height}"
        )
    scale = min(target_width / width, target_height / height)
    resized_width = max(1, int(round(width * scale)))
    resized_height = max(1, int(round(height * scale)))
    resized = image.convert("RGB").resize(
        (resized_width, resized_height),
        resample=Image.Resampling.LANCZOS,
    )
    canvas = Image.new("RGB", (target_width, target_height), (255, 255, 255))
    canvas.paste(
        resized,
        (
            (target_width - resized_width) // 2,
            (target_height - resized_height) // 2,
        ),
    )
    return canvas


def _ensure_figure_reference(prompt: str) -> str:
    """Address the single I2V image as ``<Figure 1>`` in a plain or JSON prompt."""
    try:
        prompt_object = json.loads(prompt)
        if not isinstance(prompt_object, dict):
            raise ValueError("prompt JSON is not an object")
        prompt_object["reference_layer"] = [
            "The first frame refers to <Figure 1>"
        ]
        return json.dumps(prompt_object, ensure_ascii=False)
    except (json.JSONDecodeError, TypeError, ValueError):
        return prompt + "reference_layer:The first frame refers to <Figure 1>"


class Magi2ReferenceImageStage(PipelineStage):
    """Encode one I2V reference image on the context-parallel leader."""

    def __init__(
        self,
        image_encoder: Any | None,
        preview_height: int,
        preview_width: int,
    ) -> None:
        """Store the Wan encoder and the published preview resolution."""
        super().__init__()
        self.image_encoder = image_encoder
        self.preview_height = preview_height
        self.preview_width = preview_width
        self.video_processor = VideoProcessor(vae_scale_factor=32)

    def verify_input(
        self,
        batch: ForwardBatch,
        fastvideo_args: FastVideoArgs,
    ) -> VerificationResult:
        """Return the default validation record; ``forward`` gives precise errors."""
        del batch, fastvideo_args
        return VerificationResult()

    def verify_output(
        self,
        batch: ForwardBatch,
        fastvideo_args: FastVideoArgs,
    ) -> VerificationResult:
        """Return the default validation record for T2V and I2V outputs."""
        del batch, fastvideo_args
        return VerificationResult()

    @torch.inference_mode()
    def forward(
        self,
        batch: ForwardBatch,
        fastvideo_args: FastVideoArgs,
    ) -> ForwardBatch:
        """Encode and broadcast the official FP32 Wan posterior mean for I2V."""
        if fastvideo_args.workload_type != WorkloadType.I2V:
            batch.magi2_ref_image_feat = None
            batch.magi2_ref_image_feat_len = None
            batch.magi2_ref_image_special_tokens = None
            return batch

        is_leader = psm.is_group_first_rank("cp")
        payload: dict[str, torch.Tensor] | None = None
        if is_leader:
            if self.image_encoder is None:
                raise RuntimeError("MAGI-2 I2V requires the Wan image encoder")
            image = batch.pil_image
            if image is None and batch.image_path is not None:
                image = load_image(batch.image_path)
            if not isinstance(image, Image.Image):
                raise TypeError("MAGI-2 I2V requires one PIL image or image path")

            maximum_length = max(self.preview_height, self.preview_width)
            if image.width > image.height:
                target_width = maximum_length
                target_height = int(image.height * maximum_length / image.width)
            else:
                target_height = maximum_length
                target_width = int(image.width * maximum_length / image.height)
            image = _resizepad(image, target_height, target_width)
            device = torch.device("cuda", torch.cuda.current_device())
            image_tensor = self.video_processor.preprocess(
                image,
                height=target_height,
                width=target_width,
            )
            image_tensor = image_tensor.to(device=device, dtype=torch.bfloat16)
            image_tensor = image_tensor.unsqueeze(2)[:, :3]
            self.image_encoder.to(device=device, dtype=torch.float32)
            reference_latent = self.image_encoder.encode(image_tensor.float())
            latent_height = int(reference_latent.shape[-2])
            latent_width = int(reference_latent.shape[-1])
            payload = {
                "reference_latent": reference_latent.unsqueeze(1),
                "reference_length": torch.tensor(
                    [[[latent_height, latent_width]]],
                    device=device,
                    dtype=torch.long,
                ),
            }
            if fastvideo_args.image_encoder_cpu_offload:
                self.image_encoder.to("cpu")

        broadcast_payload = get_sp_group().broadcast_tensor_dict(payload, src=0)
        if broadcast_payload is None:
            raise RuntimeError("MAGI-2 reference-image broadcast returned no payload")
        batch.magi2_ref_image_feat = broadcast_payload["reference_latent"]
        batch.magi2_ref_image_feat_len = broadcast_payload["reference_length"]
        if not isinstance(batch.prompt, str):
            raise TypeError("MAGI-2 accepts one prompt string per request")
        batch.prompt = _ensure_figure_reference(batch.prompt)
        if is_leader and fastvideo_args.image_encoder_cpu_offload:
            torch.cuda.empty_cache()
        return batch


class Magi2TextEncodingStage(PipelineStage):
    """Encode positive and negative prompts on the context-parallel leader."""

    def __init__(self, text_encoder: Any | None) -> None:
        """Store the native Qwen3.5 encoder loaded from the release checkpoint."""
        super().__init__()
        self.text_encoder = text_encoder

    def verify_input(
        self,
        batch: ForwardBatch,
        fastvideo_args: FastVideoArgs,
    ) -> VerificationResult:
        """Return the default record; prompt validation occurs in ``forward``."""
        del batch, fastvideo_args
        return VerificationResult()

    def verify_output(
        self,
        batch: ForwardBatch,
        fastvideo_args: FastVideoArgs,
    ) -> VerificationResult:
        """Return the default record for the broadcast conditioning tensors."""
        del batch, fastvideo_args
        return VerificationResult()

    @torch.inference_mode()
    def forward(
        self,
        batch: ForwardBatch,
        fastvideo_args: FastVideoArgs,
    ) -> ForwardBatch:
        """Produce Qwen3.5 hidden states and the optional figure-token embedding."""
        if not isinstance(batch.prompt, str):
            raise TypeError("MAGI-2 accepts one prompt string per request")
        if not isinstance(batch.negative_prompt, str):
            raise TypeError("MAGI-2 requires one negative prompt string")

        is_leader = psm.is_group_first_rank("cp")
        payload: dict[str, torch.Tensor | None] | None = None
        if is_leader:
            if self.text_encoder is None:
                raise RuntimeError("MAGI-2 requires the Qwen3.5 text encoder")
            device = torch.device("cuda", torch.cuda.current_device())
            self.text_encoder.to(device=device)
            text_context = self.text_encoder.encode(batch.prompt).to(torch.bfloat16)
            negative_context = self.text_encoder.encode(batch.negative_prompt).to(
                torch.bfloat16
            )
            special_tokens = None
            if batch.magi2_ref_image_feat is not None:
                special_tokens = self.text_encoder.get_special_token(
                    batch.prompt,
                    ["<Figure 1>"],
                    text_context,
                ).unsqueeze(0)
            payload = {
                "text_context": text_context,
                "negative_context": negative_context,
                "special_tokens": special_tokens,
            }
            if fastvideo_args.text_encoder_cpu_offload:
                self.text_encoder.to("cpu")

        broadcast_payload = get_sp_group().broadcast_tensor_dict(payload, src=0)
        if broadcast_payload is None:
            raise RuntimeError("MAGI-2 text-conditioning broadcast returned no payload")
        batch.magi2_text_context = broadcast_payload["text_context"]
        batch.magi2_negative_context = broadcast_payload["negative_context"]
        batch.magi2_ref_image_special_tokens = broadcast_payload["special_tokens"]
        if is_leader and fastvideo_args.text_encoder_cpu_offload:
            torch.cuda.empty_cache()
        return batch


__all__ = ["Magi2ReferenceImageStage", "Magi2TextEncodingStage"]
