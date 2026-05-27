# SPDX-License-Identifier: Apache-2.0
"""Flux2 text encoding stages."""
from __future__ import annotations

from typing import Any

import torch

from fastvideo.distributed import get_local_torch_device
from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.forward_context import set_forward_context
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.pipelines.stages.text_encoding import TextEncodingStage

FLUX2_SYSTEM_MESSAGE = ("You are an AI that reasons about image descriptions. You give structured "
                        "responses focusing on object relationships, object\nattribution and actions "
                        "without speculation.")


def _format_flux2_full_input(prompts: list[str], system_message: str) -> list[list[dict[str, Any]]]:
    return [[
        {
            "role": "system",
            "content": [{
                "type": "text",
                "text": system_message
            }],
        },
        {
            "role": "user",
            "content": [{
                "type": "text",
                "text": prompt.replace("[IMG]", "")
            }],
        },
    ] for prompt in prompts]


def _prepare_flux2_text_ids(prompt_embeds: torch.Tensor) -> torch.Tensor:
    batch_size, seq_len, _ = prompt_embeds.shape
    text_ids = torch.cartesian_prod(
        torch.arange(1, device=prompt_embeds.device),
        torch.arange(1, device=prompt_embeds.device),
        torch.arange(1, device=prompt_embeds.device),
        torch.arange(seq_len, device=prompt_embeds.device),
    )
    return text_ids.unsqueeze(0).expand(batch_size, -1, -1)


class Flux2TextEncodingStage(TextEncodingStage):
    """Text encoding for Flux2 full and Klein variants."""

    def _uses_embedded_guidance(self, fastvideo_args: FastVideoArgs) -> bool:
        return getattr(fastvideo_args.pipeline_config, "embedded_cfg_scale", None) is not None

    @torch.no_grad()
    def forward(
        self,
        batch: ForwardBatch,
        fastvideo_args: FastVideoArgs,
    ) -> ForwardBatch:
        if self._uses_embedded_guidance(fastvideo_args):
            batch.do_classifier_free_guidance = False
            batch.negative_prompt_embeds = []

        if batch.prompt_embeds is not None and len(batch.prompt_embeds) > 0:
            if "flux2_txt_ids" not in batch.extra:
                batch.extra["flux2_txt_ids"] = _prepare_flux2_text_ids(batch.prompt_embeds[0])
            return batch

        if getattr(fastvideo_args.pipeline_config, "flux2_text_encoder_type", "") != "mistral3":
            return super().forward(batch, fastvideo_args)

        assert batch.prompt is not None
        prompt_embeds, attention_mask = self.encode_flux2_full_text(
            batch.prompt,
            fastvideo_args,
            max_length=batch.max_sequence_length,
        )
        batch.prompt_embeds.append(prompt_embeds)
        batch.extra["flux2_txt_ids"] = _prepare_flux2_text_ids(prompt_embeds)
        if batch.prompt_attention_mask is not None:
            batch.prompt_attention_mask.append(attention_mask)
        return batch

    @torch.no_grad()
    def encode_flux2_full_text(
        self,
        text: str | list[str],
        fastvideo_args: FastVideoArgs,
        max_length: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        tokenizer = self.tokenizers[0]
        text_encoder = self.text_encoders[0]
        encoder_config = fastvideo_args.pipeline_config.text_encoder_configs[0]
        arch_config = encoder_config.arch_config

        prompts = [text] if isinstance(text, str) else text
        max_sequence_length = max_length or getattr(arch_config, "text_len", 512) or 512
        hidden_state_layers = getattr(
            fastvideo_args.pipeline_config,
            "text_encoder_out_layers",
            (10, 20, 30),
        )
        system_message = getattr(
            fastvideo_args.pipeline_config,
            "flux2_system_message",
            FLUX2_SYSTEM_MESSAGE,
        )

        messages = _format_flux2_full_input(prompts, system_message)
        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=False,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=max_sequence_length,
        )

        try:
            encoder_device = next(text_encoder.parameters()).device
        except StopIteration:
            encoder_device = get_local_torch_device()
        encoder_dtype = getattr(text_encoder, "dtype", None)

        input_ids = inputs["input_ids"].to(encoder_device)
        attention_mask = inputs["attention_mask"].to(encoder_device)

        forward_kwargs: dict[str, Any] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "output_hidden_states": True,
            "use_cache": False,
        }
        if "pixel_values" in inputs:
            forward_kwargs["pixel_values"] = inputs["pixel_values"].to(
                device=encoder_device,
                dtype=encoder_dtype or torch.bfloat16,
            )
        if "image_sizes" in inputs:
            forward_kwargs["image_sizes"] = inputs["image_sizes"].to(encoder_device)

        with set_forward_context(current_timestep=0, attn_metadata=None):
            outputs = text_encoder(**forward_kwargs)

        if outputs.hidden_states is None:
            raise ValueError("Full Flux2 requires output_hidden_states=True from text encoder")

        stacked = torch.stack([outputs.hidden_states[k] for k in hidden_state_layers], dim=1)
        if encoder_dtype is not None:
            stacked = stacked.to(dtype=encoder_dtype)
        batch_size, num_layers, seq_len, hidden_dim = stacked.shape
        prompt_embeds = stacked.permute(0, 2, 1, 3).reshape(
            batch_size,
            seq_len,
            num_layers * hidden_dim,
        )
        return prompt_embeds, attention_mask
