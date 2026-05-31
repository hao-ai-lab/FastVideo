# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import re
from math import sqrt

import numpy as np
import torch

from fastvideo.distributed import get_local_torch_device
from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.logger import init_logger
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.pipelines.stages.base import PipelineStage
from fastvideo.pipelines.stages.validators import StageValidators as V
from fastvideo.pipelines.stages.validators import VerificationResult

logger = init_logger(__name__)


def calculate_shift(
    image_seq_len: int,
    base_seq_len: int = 256,
    base_shift: float = 0.25,
    max_shift: float = 0.75,
) -> float:
    return (image_seq_len / base_seq_len)**0.5 * max_shift + base_shift


def get_glyph_texts(prompt: str | list[str]) -> list[str] | list[list[str]]:
    if isinstance(prompt, str):
        prompts: list[str] = [prompt]
        is_batch = False
    else:
        prompts = prompt
        is_batch = True
    out: list[list[str]] = []
    for p in prompts:
        out.append(
            re.findall(r"'([^']*)'", p) + re.findall(r"“([^“”]*)”", p) + re.findall(r'"([^"]*)"', p) +
            re.findall(r"「([^「」]*)」", p))
    return out if is_batch else out[0]


def compute_glyph_embeds(
    prompts: list[str],
    tokenizer,
    text_encoder,
    device: torch.device,
    dtype: torch.dtype,
    max_sequence_length: int = 2048,
) -> torch.Tensor:
    all_glyph_texts = get_glyph_texts(prompts)
    all_glyph_embeds = []
    for glyph_texts in all_glyph_texts:
        if len(glyph_texts) == 0:
            glyph_texts = [""]
        input_ids = tokenizer(
            glyph_texts,
            max_length=max_sequence_length,
            truncation=True,
        ).input_ids
        input_ids = [[tokenizer.pad_token_id] * ((len(input_ids) + 1) % 2) + ids for ids in input_ids]
        max_length = max(len(ids) for ids in input_ids)
        attention_mask = torch.tensor(
            [[1] * len(ids) + [0] * (max_length - len(ids)) for ids in input_ids],
            device=device,
        )
        input_ids_t = torch.tensor(
            [ids + [tokenizer.pad_token_id] * (max_length - len(ids)) for ids in input_ids],
            device=device,
        )
        outputs = text_encoder(input_ids_t, attention_mask=attention_mask)
        glyph_embeds = outputs.last_hidden_state[attention_mask.bool()].unsqueeze(0)
        all_glyph_embeds.append(glyph_embeds)

    max_seq_len = max(emb.size(1) for emb in all_glyph_embeds)
    padded = []
    for emb in all_glyph_embeds:
        if emb.size(1) < max_seq_len:
            pad = torch.zeros(emb.size(0), max_seq_len - emb.size(1), emb.size(2), device=device, dtype=emb.dtype)
            emb = torch.cat([pad, emb], dim=1)
        padded.append(emb)
    return torch.cat(padded, dim=0).to(device=device, dtype=dtype)


def _grid_dims(height: int, width: int) -> tuple[int, int, int, int]:
    th, tw = height // 32, width // 32
    ratio = th / tw
    pth = int(sqrt(ratio) * 16)
    ptw = int(sqrt(1 / ratio) * 16)
    return th, tw, pth, ptw


def _upsample_d32_to_d16(tokens: torch.Tensor, th: int, tw: int) -> torch.Tensor:
    tokens = tokens.view(1, 1, th, tw).float()
    tokens = torch.nn.functional.interpolate(tokens, scale_factor=2, mode="nearest").long()
    return tokens.view(1, -1)


class GlmImageBeforeDenoisingStage(PipelineStage):

    def __init__(self,
                 vae,
                 text_encoder,
                 tokenizer,
                 processor,
                 transformer,
                 scheduler,
                 vision_language_encoder=None) -> None:
        super().__init__()
        self.vae = vae
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.processor = processor
        self.transformer = transformer
        self.scheduler = scheduler
        if isinstance(vision_language_encoder, tuple):
            self.vision_language_encoder, self.vl_processor = (vision_language_encoder[0], vision_language_encoder[1]
                                                               or processor)
        else:
            self.vision_language_encoder = vision_language_encoder
            self.vl_processor = processor

    @torch.no_grad()
    def forward(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> ForwardBatch:
        device = get_local_torch_device()
        dtype = torch.bfloat16
        th, tw, pth, ptw = _grid_dims(batch.height, batch.width)

        if batch.seed is not None:
            torch.manual_seed(batch.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(batch.seed)

        # 1-3. AR token generation. I2I prepends the condition image and uses a
        # single-scale target grid; T2I is multi-scale.
        is_t2i = batch.pil_image is None
        if self.vision_language_encoder is not None:
            content = [{"type": "text", "text": batch.prompt}]
            if not is_t2i:
                content.insert(0, {"type": "image", "image": batch.pil_image})
            messages = [{"role": "user", "content": content}]
            inputs = self.vl_processor.apply_chat_template(messages,
                                                           tokenize=True,
                                                           target_h=batch.height,
                                                           target_w=batch.width,
                                                           return_dict=True,
                                                           return_tensors="pt").to(device)

            if is_t2i:
                up_h, up_w = th, tw
                large_start, large_count = pth * ptw, th * tw
                max_new = large_count + (pth * ptw) + 1
            else:
                # Condition grid(s) first, target grid last.
                _, t_h, t_w = inputs["image_grid_thw"][-1].tolist()
                up_h, up_w = int(t_h), int(t_w)
                large_start, large_count = 0, up_h * up_w
                max_new = large_count + 1

            outputs = self.vision_language_encoder.generate(**inputs, max_new_tokens=max_new, do_sample=True)
            gen_tokens = outputs[0][inputs.input_ids.shape[-1]:]
            if gen_tokens.shape[0] >= large_start + large_count:
                large_tokens = gen_tokens[large_start:large_start + large_count]
            else:
                available = gen_tokens[large_start:]
                large_tokens = torch.zeros(large_count, dtype=gen_tokens.dtype, device=gen_tokens.device)
                if available.shape[0] > 0:
                    large_tokens[:min(available.shape[0], large_count)] = available[:large_count]
                logger.warning("AR generated %d tokens, expected %d. Padding with zeros.", gen_tokens.shape[0],
                               large_start + large_count)
            batch.prior_token_id = _upsample_d32_to_d16(large_tokens, up_h, up_w)
            batch.prior_token_drop = torch.zeros(batch.prior_token_id.shape, dtype=torch.bool, device=device)

            if not is_t2i:
                self._compute_source_prior_tokens(batch, inputs)
        else:
            num_prior_tokens = 4 * th * tw
            logger.warning("No vision_language_encoder provided; using random dropped priors.")
            batch.prior_token_id = torch.randint(0, 16384, (1, num_prior_tokens), device=device)
            batch.prior_token_drop = torch.ones(batch.prior_token_id.shape, dtype=torch.bool, device=device)

        # 4. Glyph T5 encoding.
        prompts = [batch.prompt] if isinstance(batch.prompt, str) else list(batch.prompt)
        prompt_embeds = compute_glyph_embeds(prompts, self.tokenizer, self.text_encoder, device, dtype)

        # 5. CFG-side negative encoding.
        if batch.do_classifier_free_guidance:
            neg_prompts = [batch.negative_prompt or ""] * len(prompts)
            neg_embeds = compute_glyph_embeds(neg_prompts, self.tokenizer, self.text_encoder, device, dtype)
            L_pos, L_neg = prompt_embeds.shape[1], neg_embeds.shape[1]
            max_L = max(L_pos, L_neg)
            if L_pos < max_L:
                pad = torch.zeros(prompt_embeds.shape[0],
                                  max_L - L_pos,
                                  prompt_embeds.shape[2],
                                  device=device,
                                  dtype=dtype)
                prompt_embeds = torch.cat([pad, prompt_embeds], dim=1)
            if L_neg < max_L:
                pad = torch.zeros(neg_embeds.shape[0], max_L - L_neg, neg_embeds.shape[2], device=device, dtype=dtype)
                neg_embeds = torch.cat([pad, neg_embeds], dim=1)
            # Row 0 conditional (positive), row 1 unconditional (negative).
            prompt_embeds = torch.cat([prompt_embeds, neg_embeds], dim=0)
            att_pos = torch.ones((1, max_L), device=device)
            att_neg = torch.ones((1, max_L), device=device)
            if L_pos < max_L:
                att_pos[:, :max_L - L_pos] = 0
            if L_neg < max_L:
                att_neg[:, :max_L - L_neg] = 0
            attention_mask = torch.cat([att_pos, att_neg], dim=0)
        else:
            attention_mask = torch.ones((1, prompt_embeds.shape[1]), device=device)

        batch.prompt_embeds = [prompt_embeds]
        batch.attention_mask = attention_mask

        # 6. Latents + dynamic flow shift.
        if batch.seed is not None:
            torch.manual_seed(batch.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(batch.seed)
        batch.latents = torch.randn((1, 16, 1, batch.height // 8, batch.width // 8), device=device, dtype=dtype)

        # Integer-cast linspace timesteps with resolution-dependent shift applied to
        # sigmas only; the DiT is conditioned on the unshifted integer timesteps.
        ntt = self.scheduler.config.num_train_timesteps
        patch_size = self.transformer.patch_size
        image_seq_len = ((batch.height // 8) * (batch.width // 8)) // (patch_size**2)
        sched_timesteps = np.linspace(ntt, 1.0, batch.num_inference_steps + 1)[:-1].astype(np.int64).astype(np.float32)
        sched_sigmas = sched_timesteps / ntt
        self.scheduler.set_shift(calculate_shift(image_seq_len))
        self.scheduler.set_timesteps(batch.num_inference_steps,
                                     device=device,
                                     sigmas=sched_sigmas.tolist(),
                                     timesteps=sched_timesteps.tolist())
        batch.timesteps = self.scheduler.timesteps
        return batch

    @torch.no_grad()
    def _compute_source_prior_tokens(self, batch: ForwardBatch, inputs) -> None:
        image_grid_thw = inputs["image_grid_thw"]
        num_condition_images = image_grid_thw.shape[0] - 1
        source_grids = image_grid_thw[:num_condition_images]
        embed = torch.cat(self.vision_language_encoder.get_image_features(inputs["pixel_values"], source_grids), dim=0)
        src_ids_d32 = self.vision_language_encoder.get_image_tokens(embed, source_grids)
        split_sizes = source_grids.prod(dim=-1).tolist()
        upsampled = [
            _upsample_d32_to_d16(ids, int(grid[1]), int(grid[2])).squeeze(0)
            for ids, grid in zip(torch.split(src_ids_d32, split_sizes), source_grids, strict=False)
        ]
        src_grids_up = source_grids.clone()
        src_grids_up[:, 1] *= 2
        src_grids_up[:, 2] *= 2
        batch.extra["glm_prior_token_image_ids"] = torch.cat(upsampled, dim=0)
        batch.extra["glm_source_image_grid_thw"] = src_grids_up

    def verify_input(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> VerificationResult:
        result = VerificationResult()
        result.add_check("prompt", batch.prompt, V.string_not_empty)
        return result

    def verify_output(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> VerificationResult:
        result = VerificationResult()
        result.add_check("prior_token_id", batch.prior_token_id, V.is_tensor)
        return result
