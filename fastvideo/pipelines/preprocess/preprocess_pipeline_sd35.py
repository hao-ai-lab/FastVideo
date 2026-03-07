# SPDX-License-Identifier: Apache-2.0
"""
SD3.5 Data Preprocessing pipeline.

Handles the three-encoder text conditioning for SD3.5:
  - CLIP-L  → penultimate hidden states + pooled output
  - CLIP-G  → penultimate hidden states + pooled output
  - T5-XXL  → last hidden state (optionally zero-masked by attention mask)

The combined text embedding saved to parquet is:
  text_embedding:      cat([clip_l, clip_g], dim=-1) zero-padded to T5 dim,
                       then cat with t5 along the sequence axis.
                       Shape: (clip_seq + t5_seq, 4096)
  pooled_projection:   cat([clip_l_pooled, clip_g_pooled], dim=-1)
                       Shape: (2048,)
"""

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from fastvideo.dataset.dataloader.schema import pyarrow_schema_sd35
from fastvideo.distributed import get_local_torch_device
from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.forward_context import set_forward_context
from fastvideo.logger import init_logger
from fastvideo.pipelines.preprocess.preprocess_pipeline_base import (
    BasePreprocessPipeline)

logger = init_logger(__name__)

# T5 max tokens consistent with SimpleTuner / FastVideo inference default
_T5_MAX_LENGTH = 154
_CLIP_MAX_LENGTH = 77


class PreprocessPipeline_SD35(BasePreprocessPipeline):
    """SD3.5 preprocessing pipeline: VAE + 3 text encoders."""

    _required_config_modules = [
        "text_encoder",
        "text_encoder_2",
        "text_encoder_3",
        "tokenizer",
        "tokenizer_2",
        "tokenizer_3",
        "vae",
    ]

    def get_pyarrow_schema(self):
        return pyarrow_schema_sd35

    def create_pipeline_stages(self, fastvideo_args: FastVideoArgs):
        # No standard TextEncodingStage — we handle encoding manually so that
        # we can capture both hidden states and pooled CLIP outputs in a single
        # forward pass per encoder.
        pass

    @torch.no_grad()
    def _encode_clip(
        self,
        text_encoder,
        tokenizer,
        prompts: list[str],
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Run a CLIP text encoder and return (penultimate_hidden, pooled).

        Returns:
            penultimate_hidden: (B, seq, hidden_dim)
            pooled:             (B, hidden_dim)  — pooler_output (projected CLS)
        """
        enc = tokenizer(
            prompts,
            padding="max_length",
            max_length=_CLIP_MAX_LENGTH,
            truncation=True,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)

        with set_forward_context(current_timestep=0, attn_metadata=None):
            out = text_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )

        # penultimate hidden state — same as _sd35_clip_text_postprocess
        hidden = out.hidden_states[-2].to(dtype=dtype)
        # projected pooled output — FastVideo's CLIPTextModelWithProjection
        # stores the projected CLS token in pooler_output (BaseEncoderOutput),
        # not in text_embeds (which is a HuggingFace-only attribute).
        pooled = out.pooler_output.to(dtype=dtype)
        return hidden, pooled

    @torch.no_grad()
    def _encode_t5(
        self,
        text_encoder,
        tokenizer,
        prompts: list[str],
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """
        Run the T5 text encoder and return last hidden state.
        Padding tokens are zeroed out (SimpleTuner zero_padding_tokens=True).

        Returns:
            hidden: (B, seq, 4096)
        """
        enc = tokenizer(
            prompts,
            padding="max_length",
            max_length=_T5_MAX_LENGTH,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)

        with set_forward_context(current_timestep=0, attn_metadata=None):
            out = text_encoder(input_ids=input_ids)

        hidden = out.last_hidden_state.to(dtype=dtype)
        # Zero out padding tokens to avoid bias from pad embeddings
        hidden = hidden * attention_mask.unsqueeze(-1).to(dtype=dtype)
        return hidden

    @torch.no_grad()
    def _encode_prompts(
        self,
        prompts: list[str],
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Encode a batch of prompts with all three text encoders and combine.

        Returns:
            encoder_hidden_states: (B, clip_seq+t5_seq, 4096)
            pooled_projections:    (B, 2048)
        """
        dtype = torch.float32  # encode in fp32, caller can cast

        text_encoder = self.get_module("text_encoder")
        text_encoder_2 = self.get_module("text_encoder_2")
        text_encoder_3 = self.get_module("text_encoder_3")
        tokenizer = self.get_module("tokenizer")
        tokenizer_2 = self.get_module("tokenizer_2")
        tokenizer_3 = self.get_module("tokenizer_3")

        clip_1_hidden, clip_1_pooled = self._encode_clip(
            text_encoder, tokenizer, prompts, device, dtype)
        clip_2_hidden, clip_2_pooled = self._encode_clip(
            text_encoder_2, tokenizer_2, prompts, device, dtype)
        t5_hidden = self._encode_t5(
            text_encoder_3, tokenizer_3, prompts, device, dtype)

        # Concatenate CLIP hidden states along feature dim: (B, 77, 2048)
        clip_hidden = torch.cat([clip_1_hidden, clip_2_hidden], dim=-1)

        # Zero-pad CLIP to T5 feature dim: (B, 77, 4096)
        t5_dim = t5_hidden.shape[-1]
        clip_hidden = F.pad(clip_hidden, (0, t5_dim - clip_hidden.shape[-1]))

        # Concatenate along sequence dim: (B, 77+154, 4096)
        encoder_hidden_states = torch.cat([clip_hidden, t5_hidden], dim=-2)

        # Pooled projections: (B, 2048)
        pooled_projections = torch.cat([clip_1_pooled, clip_2_pooled], dim=-1)

        return encoder_hidden_states, pooled_projections

    def get_extra_features(
        self,
        valid_data: dict[str, Any],
        fastvideo_args: FastVideoArgs,
    ) -> dict[str, Any]:
        """Encode text and return pooled_projection alongside text_embedding."""
        device = get_local_torch_device()
        prompts: list[str] = valid_data["text"]

        encoder_hidden_states, pooled_projections = self._encode_prompts(
            prompts, device)

        return {
            "text_embedding": encoder_hidden_states,
            "pooled_projection": pooled_projections,
        }

    def preprocess_video_and_text(self, fastvideo_args: FastVideoArgs, args):
        """
        Override to store VAE latents as (C, T, H, W) with T=1 for SD3.5,
        consistent with the training pipeline's temporal-dimension convention.
        """
        import os

        import pyarrow.parquet as pq
        from torch.utils.data import DataLoader
        from tqdm import tqdm

        from fastvideo.dataset import getdataset
        from fastvideo.dataset.dataloader.parquet_io import (
            ParquetDatasetWriter, records_to_table)

        os.makedirs(args.output_dir, exist_ok=True)
        combined_parquet_dir = os.path.join(args.output_dir,
                                            "combined_parquet_dataset")
        os.makedirs(combined_parquet_dir, exist_ok=True)
        local_rank = int(os.getenv("RANK", 0))

        start_idx = 0
        for root, _, files in os.walk(combined_parquet_dir):
            for file in files:
                if file.endswith(".parquet"):
                    table = pq.read_table(os.path.join(root, file))
                    start_idx += table.num_rows

        train_dataset = getdataset(args)
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=args.preprocess_video_batch_size,
            num_workers=args.dataloader_num_workers,
        )

        device = get_local_torch_device()
        vae = self.get_module("vae")
        num_processed = 0

        pbar = tqdm(train_dataloader,
                    desc="Processing images/videos",
                    unit="batch",
                    disable=local_rank != 0)

        for _, data in enumerate(pbar):
            if data is None:
                continue

            with torch.inference_mode():
                valid_indices = [
                    i for i, pv in enumerate(data["pixel_values"])
                    if not torch.all(pv == 0)
                ]
                if not valid_indices:
                    continue

                num_processed += len(valid_indices)
                valid_data = {
                    "pixel_values":
                    torch.stack(
                        [data["pixel_values"][i] for i in valid_indices]),
                    "text": [
                        data["text"][i] for i in valid_indices
                    ] if "text" in data else ["" for _ in valid_indices],
                    "path": [data["path"][i] for i in valid_indices],
                    "fps": [data["fps"][i] for i in valid_indices]
                    if "fps" in data else [1.0] * len(valid_indices),
                    "duration": [data["duration"][i] for i in valid_indices]
                    if "duration" in data else [0.0] * len(valid_indices),
                }

                # VAE encode — pixel_values from dataset are (B, C, 1, H, W)
                # for images; squeeze the T=1 dim since AutoencoderKL expects
                # (B, C, H, W).
                pv = valid_data["pixel_values"].to(device)
                if pv.ndim == 5:
                    pv = pv.squeeze(2)  # (B, C, 1, H, W) → (B, C, H, W)
                with torch.autocast("cuda", dtype=torch.float32):
                    raw_latents = vae.encode(pv).latent_dist.mean

                # Add T=1 dimension back: (B, C, H, W) → (B, C, 1, H, W)
                raw_latents = raw_latents.unsqueeze(2)

                # Text encoding (returns per-sample tensors)
                encoder_hidden_states, pooled_projections = (
                    self._encode_prompts(valid_data["text"], device))

            batch_data = []
            for idx, video_path in enumerate(valid_data["path"]):
                video_name = os.path.basename(video_path).split(".")[0]

                # Per-sample tensors (already CPU-safe after inference_mode)
                vae_latent = raw_latents[idx].cpu().numpy().astype(np.float32)
                text_emb = (encoder_hidden_states[idx].cpu().float().numpy())
                pooled = pooled_projections[idx].cpu().float().numpy()

                pixel = valid_data["pixel_values"][idx]
                record = {
                    "id":
                    video_name,
                    "vae_latent_bytes":
                    vae_latent.tobytes(),
                    "vae_latent_shape":
                    list(vae_latent.shape),
                    "vae_latent_dtype":
                    str(vae_latent.dtype),
                    "text_embedding_bytes":
                    text_emb.tobytes(),
                    "text_embedding_shape":
                    list(text_emb.shape),
                    "text_embedding_dtype":
                    str(text_emb.dtype),
                    "pooled_projection_bytes":
                    pooled.tobytes(),
                    "pooled_projection_shape":
                    list(pooled.shape),
                    "pooled_projection_dtype":
                    str(pooled.dtype),
                    "file_name":
                    video_name,
                    "caption":
                    valid_data["text"][idx],
                    "media_type":
                    "image",
                    "width":
                    int(pixel.shape[-2]),
                    "height":
                    int(pixel.shape[-1]),
                    "num_frames":
                    1,
                    "duration_sec":
                    float(valid_data["duration"][idx]),
                    "fps":
                    float(valid_data["fps"][idx]),
                }
                batch_data.append(record)

            if batch_data:
                table = records_to_table(batch_data, self.get_pyarrow_schema())
                if not hasattr(self, "dataset_writer"):
                    self.dataset_writer = ParquetDatasetWriter(
                        out_dir=combined_parquet_dir,
                        samples_per_file=args.samples_per_file,
                    )
                self.dataset_writer.append_table(table)
                logger.info("Collected batch with %s samples", len(table))

            if num_processed >= args.flush_frequency:
                written = self.dataset_writer.flush()
                logger.info("Flushed %s samples to parquet", written)
                num_processed = 0

        # Final flush for any remaining buffered samples
        if hasattr(self, "dataset_writer"):
            written = self.dataset_writer.flush(write_remainder=True)
            if written:
                logger.info("Final flush: wrote %s samples to parquet", written)


EntryClass = PreprocessPipeline_SD35
