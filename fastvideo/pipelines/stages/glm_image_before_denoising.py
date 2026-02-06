# SPDX-License-Identifier: Apache-2.0
"""
GLM-Image before denoising stage.

This stage handles the preprocessing for GLM-Image generation, including:
- Prompt parsing and tokenization
- AR token generation via the vision-language encoder
- Latent preparation for the denoising process
"""

import re
from math import sqrt

import torch
from fastvideo.distributed import get_local_torch_device
from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.logger import init_logger
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.pipelines.stages.base import PipelineStage
from fastvideo.pipelines.stages.validators import StageValidators as V
from fastvideo.pipelines.stages.validators import VerificationResult

logger = init_logger(__name__)


def calculate_shift(image_seq_len,
                    base_seq_len: int = 256,
                    base_shift: float = 0.25,
                    max_shift: float = 0.75) -> float:
    m = (image_seq_len / base_seq_len)**0.5
    return m * max_shift + base_shift


class GlmImageBeforeDenoisingStage(PipelineStage):

    def __init__(self,
                 vae,
                 text_encoder,
                 tokenizer,
                 processor,
                 transformer,
                 scheduler,
                 vision_language_encoder=None):
        super().__init__()
        self.vae = vae
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.processor = processor
        self.transformer = transformer
        self.scheduler = scheduler

        if isinstance(vision_language_encoder, tuple):
            self.vision_language_encoder = vision_language_encoder[0]
            self.vl_processor = vision_language_encoder[1] or processor
        else:
            self.vision_language_encoder = vision_language_encoder
            self.vl_processor = processor

    def _format_prompt(self, prompt: str, h: int, w: int) -> str:
        """Add <sop>H W<eop> to prompt if missing. H, W are pixel sizes / 32."""
        th, tw = h // 32, w // 32
        tag = f"<sop>{th} {tw}<eop>"
        if tag not in prompt:
            # Check if there's an existing tag to replace
            prompt = re.sub(r"<sop>\d+\s+\d+<eop>", "", prompt)
            prompt = f"{prompt} {tag}"
        return prompt

    def _parse_shape(self, prompt: str) -> tuple[int, int, int, int]:
        match = re.search(r"<sop>(\d+)\s+(\d+)<eop>", prompt)
        if not match:
            raise ValueError(
                f"Prompt must contain <sop>H W<eop>, got: {prompt}")
        th, tw = int(match.group(1)), int(match.group(2))
        ratio = th / tw
        pth = int(sqrt(ratio) * 16)
        ptw = int(sqrt(1 / ratio) * 16)
        return th, tw, pth, ptw

    def _upsample_tokens(self, tokens: torch.Tensor, th: int,
                         tw: int) -> torch.Tensor:
        """2x nearest neighbor upsample from d32 to d16 tokens."""
        # tokens: [N]
        tokens = tokens.view(1, 1, th, tw)
        tokens = torch.nn.functional.interpolate(tokens.float(),
                                                 scale_factor=2,
                                                 mode="nearest").long()
        return tokens.view(1, -1)

    @torch.no_grad()
    def forward(self, batch: ForwardBatch,
                fastvideo_args: FastVideoArgs) -> ForwardBatch:
        device = get_local_torch_device()
        dtype = torch.bfloat16

        # 1. Format and parse prompt
        batch.prompt = self._format_prompt(batch.prompt, batch.height,
                                           batch.width)
        th, tw, pth, ptw = self._parse_shape(batch.prompt)

        if batch.seed is not None:
            torch.manual_seed(batch.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(batch.seed)

        # 2. Generate AR tokens
        if self.vision_language_encoder is not None:
            # Expand prompt for AR model (SGLang style)
            expanded_prompt = batch.prompt.replace(
                f"<sop>{th} {tw}<eop>",
                f"<sop>{th} {tw}<eop><sop>{pth} {ptw}<eop>")

            content = [{"type": "text", "text": expanded_prompt}]
            if hasattr(batch, 'image') and batch.image is not None:
                content.insert(0, {"type": "image", "image": batch.image})

            messages = [{"role": "user", "content": content}]
            inputs = self.vl_processor.apply_chat_template(
                messages,
                tokenize=True,
                target_h=batch.height,
                target_w=batch.width,
                return_dict=True,
                return_tensors="pt").to(device)

            # Total tokens: small_image + large_image
            max_new = (th * tw) + (pth * ptw) + 1
            
            # Use sampling to match SGLang/creative generation quality
            outputs = self.vision_language_encoder.generate(
                **inputs, 
                max_new_tokens=max_new, 
                do_sample=True,
                top_p=0.7,
                top_k=50,
                temperature=1.0
            )

            # Extract large image tokens
            gen_tokens = outputs[0][inputs.input_ids.shape[-1]:]
            expected_start = pth * ptw
            expected_count = th * tw

            # Handle variable-length output: take what we can, pad if needed
            if len(gen_tokens) >= expected_start + expected_count:
                large_tokens = gen_tokens[expected_start:expected_start +
                                          expected_count]
            else:
                # Fallback: pad with zeros if not enough tokens
                available = gen_tokens[expected_start:] if len(
                    gen_tokens) > expected_start else gen_tokens
                large_tokens = torch.zeros(expected_count,
                                           dtype=gen_tokens.dtype,
                                           device=gen_tokens.device)
                large_tokens[:min(len(available), expected_count
                                  )] = available[:expected_count]
                logger.warning(
                    "AR generated %d tokens, expected %d. Padding with zeros.",
                    len(gen_tokens), expected_start + expected_count)

            # Upsample to d16
            batch.prior_token_id = self._upsample_tokens(large_tokens, th, tw)
            batch.prior_token_drop = torch.zeros(batch.prior_token_id.shape,
                                                 dtype=torch.bool,
                                                 device=device)

        else:
            # No AR model - use random prior tokens (unconditional generation fallback)
            # Number of patches after upsampling: (2*th) * (2*tw)
            num_prior_tokens = 4 * th * tw  # Upsampled from th*tw
            logger.warning(
                "No vision_language_encoder provided. Using random prior tokens for unconditional generation."
            )
            batch.prior_token_id = torch.randint(0,
                                                 16384, (1, num_prior_tokens),
                                                 device=device)
            batch.prior_token_drop = torch.ones(
                batch.prior_token_id.shape, dtype=torch.bool,
                device=device)  # Drop all priors

        # 3. Encode prompts with T5 (Glyphs ONLY)
        # SGLang logic: Only encode text inside quotes for rendering.
        def get_glyph_texts(text):
            text = text[0] if isinstance(text, list) else text
            ocr_texts = (
                re.findall(r"'([^']*)'", text)
                + re.findall(r"“([^“”]*)”", text)
                + re.findall(r'"([^"]*)"', text)
                + re.findall(r"「([^「」]*)」", text)
            )
            return ocr_texts

        glyph_texts = get_glyph_texts(batch.prompt)
        # If no glyphs, use empty string (or pad token)
        texts_to_encode = glyph_texts if len(glyph_texts) > 0 else [""]
        
        # Helper to encode list of texts and get embeddings matching SGLang style
        # SGLang pads to max_length among the batch of glyphs, and masks rest.
        # But here 'batch.prompt' is 1 string (BS=1 usually).
        # We need to return [1, L, D] or [1, max_glyph_len, D]?
        # SGLang: glyph_embeds = outputs.last_hidden_state[attention_mask.bool()].unsqueeze(0)
        # It flattens valid tokens?
        
        # Let's inspect SGLang `_get_glyph_embeds` (Step 627, lines 434-473)
        # It creates a batch of glyphs. Encodes them.
        # Then `outputs.last_hidden_state[attention_mask.bool()]` -> Selects ONLY valid tokens from all glyphs.
        # .unsqueeze(0) -> [1, Total_Valid_Tokens, D]
        
        text_inputs = self.tokenizer(
            texts_to_encode,
            padding="longest", # Pad to longest in this micro-batch
            max_length=512,
            truncation=True,
            return_tensors="pt"
        ).to(device)
        
        # Add padding for processing if needed (SGLang adds pad token buffering?)
        # SGLang line 450: `[pad] * ... + input_ids` (Left/Right pad?)
        # It seems extensive padding logic.
        # For now, standard encoding + mask selection should approximate it.
        
        prompt_outputs = self.text_encoder(
            input_ids=text_inputs.input_ids,
            attention_mask=text_inputs.attention_mask
        )
        
        # Flatten valid tokens from all glyphs into one sequence
        # inputs.attention_mask identifies valid tokens (excluding pad)
        valid_mask = text_inputs.attention_mask.bool()
        prompt_embeds = prompt_outputs.last_hidden_state[valid_mask].unsqueeze(0).to(dtype)
        
        # SGLang returns [1, Seq, D].
        batch.attention_mask = None # Attention mask handled by packing? 
        # Actually SGLang `GlmImageAttention` uses `context_len` logic.
        # FastVideo `GlmImageTransformerBlock` expects `encoder_hidden_states` [B, L, D].
        # If L varies, it's fine.
        
        # Handle Negatives (CFG)
        if batch.do_classifier_free_guidance:
             neg_prompt = batch.negative_prompt or ""
             neg_glyph_texts = get_glyph_texts(neg_prompt)
             neg_texts_to_encode = neg_glyph_texts if len(neg_glyph_texts) > 0 else [""]
             
             neg_inputs = self.tokenizer(
                 neg_texts_to_encode,
                 padding="longest",
                 max_length=512,
                 truncation=True,
                 return_tensors="pt"
             ).to(device)
             
             neg_outputs = self.text_encoder(
                 input_ids=neg_inputs.input_ids,
                 attention_mask=neg_inputs.attention_mask
             )
             neg_valid_mask = neg_inputs.attention_mask.bool()
             neg_prompt_embeds = neg_outputs.last_hidden_state[neg_valid_mask].unsqueeze(0).to(dtype)
             
             # Pad to match length for concatenation?
             # Or use list? FastVideo handles lists?
             # `glm_image.py` (Model) handles `encoder_hidden_states` as Tensor. I need to PAD.
             
             L_pos = prompt_embeds.shape[1]
             L_neg = neg_prompt_embeds.shape[1]
             max_L = max(L_pos, L_neg)
             
             # Pad with zeros
             if L_pos < max_L:
                 pad = torch.zeros((1, max_L - L_pos, prompt_embeds.shape[2]), device=device, dtype=dtype)
                 prompt_embeds = torch.cat([prompt_embeds, pad], dim=1)
             if L_neg < max_L:
                 pad = torch.zeros((1, max_L - L_neg, neg_prompt_embeds.shape[2]), device=device, dtype=dtype)
                 neg_prompt_embeds = torch.cat([neg_prompt_embeds, pad], dim=1)
                 
             prompt_embeds = torch.cat([neg_prompt_embeds, prompt_embeds], dim=0)
             # Attention mask for transformer? 
             # We should provide explicit mask if we padded.
             # SGLang: `text_attn_mask`.
             # Here we construct mask: [1, max_L]
             att_mask_pos = torch.ones((1, max_L), device=device)
             if L_pos < max_L: att_mask_pos[:, L_pos:] = 0
             
             att_mask_neg = torch.ones((1, max_L), device=device)
             if L_neg < max_L: att_mask_neg[:, L_neg:] = 0
             
             attention_mask = torch.cat([att_mask_neg, att_mask_pos], dim=0)

        else:
             attention_mask = torch.ones((1, prompt_embeds.shape[1]), device=device)

        batch.prompt_embeds = [prompt_embeds]
        batch.attention_mask = attention_mask

        # 4. Prepare Latents
        # GLM-Image uses 16 channels, VAE compression factor is 8
        # Add extra dimension for video compatibility: [B, C, T, H, W] with T=1

        # Set seed for deterministic latent generation
        if batch.seed is not None:
            torch.manual_seed(batch.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(batch.seed)

        latents = torch.randn((1, 16, 1, batch.height // 8, batch.width // 8),
                              device=device,
                              dtype=dtype)
        # Flow matching doesn't use init_noise_sigma, just pure noise
        batch.latents = latents

        # 5. Dynamic Flow Shift
        num_patches = (batch.height // 16) * (batch.width // 16)
        shift = calculate_shift(num_patches)
        if hasattr(self.scheduler, 'set_shift'):
            self.scheduler.set_shift(shift)
        self.scheduler.set_timesteps(batch.num_inference_steps, device=device)
        batch.timesteps = self.scheduler.timesteps

        return batch

    def verify_input(self, batch: ForwardBatch,
                     fastvideo_args: FastVideoArgs) -> VerificationResult:
        result = VerificationResult()
        result.add_check("prompt", batch.prompt, V.string_not_empty)
        return result

    def verify_output(self, batch: ForwardBatch,
                      fastvideo_args: FastVideoArgs) -> VerificationResult:
        result = VerificationResult()
        result.add_check("prior_token_id", batch.prior_token_id, V.is_tensor)
        return result
