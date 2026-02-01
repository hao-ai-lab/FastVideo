# SPDX-License-Identifier: Apache-2.0
"""
HyWorld preprocessing pipeline for FastVideo.

This pipeline precomputes latents and embeddings for HyWorld training:
- VAE-encoded video latents
- LLM text embeddings (Qwen 2.5 VL)
- ByT5 text embeddings  
- SigLIP vision embeddings
- First frame VAE latent (image condition)

Output format: .pt files compatible with HYWorldCameraDataset
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import imageio
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from fastvideo.distributed import get_local_torch_device
from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.forward_context import set_forward_context
from fastvideo.logger import init_logger
from fastvideo.models.dits.hyworld.resolution_utils import get_closest_resolution
from fastvideo.pipelines.composed_pipeline_base import ComposedPipelineBase
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.utils import PRECISION_TO_TYPE

logger = init_logger(__name__)


def _read_video_frames(video_path: str, num_frames: int) -> np.ndarray:
    """Read video frames as uint8 RGB: (F, H, W, 3)."""
    frames: list[np.ndarray] = []
    with imageio.get_reader(video_path) as reader:
        for frame in reader:
            frames.append(np.asarray(frame, dtype=np.uint8))
            if len(frames) >= num_frames:
                break
    if not frames:
        raise RuntimeError(f"Empty video: {video_path}")
    return np.stack(frames, axis=0)


def _to_vae_input(frames: np.ndarray, device: torch.device,
                  dtype: torch.dtype) -> torch.Tensor:
    """
    Convert frames to VAE input format.
    
    Args:
        frames: (F, H, W, 3) uint8
        
    Returns:
        (1, 3, F, H, W) float in [-1, 1]
    """
    x = torch.from_numpy(frames).to(device=device)
    x = x.permute(3, 0, 1, 2).unsqueeze(0).contiguous()  # (1, 3, F, H, W)
    x = x.to(dtype=torch.float32) / 255.0
    x = x * 2.0 - 1.0
    return x.to(dtype=dtype)


def _resize_and_center_crop(image: np.ndarray, target_width: int,
                            target_height: int) -> np.ndarray:
    """Resize and center crop an image to target dimensions."""
    from PIL import Image as PILImage
    
    img = PILImage.fromarray(image)
    orig_w, orig_h = img.size
    
    # Calculate scale to cover target dimensions
    scale = max(target_width / orig_w, target_height / orig_h)
    new_w = int(orig_w * scale)
    new_h = int(orig_h * scale)
    
    # Resize
    img = img.resize((new_w, new_h), PILImage.LANCZOS)
    
    # Center crop
    left = (new_w - target_width) // 2
    top = (new_h - target_height) // 2
    img = img.crop((left, top, left + target_width, top + target_height))
    
    return np.array(img)


class PreprocessPipeline_HYWorld(ComposedPipelineBase):
    """
    HyWorld preprocessing pipeline.
    
    Precomputes latents and embeddings for HyWorld training, outputting
    .pt files compatible with HYWorldCameraDataset.
    
    Required modules:
    - vae: HunyuanVideo VAE (with encoder enabled)
    - text_encoder: Qwen 2.5 VL for LLM embeddings
    - text_encoder_2: ByT5 for glyph text embeddings
    - tokenizer: Qwen tokenizer
    - tokenizer_2: ByT5 tokenizer
    - image_encoder: SigLIP vision encoder
    - feature_extractor: SigLIP image processor
    """

    _required_config_modules = [
        "vae", "text_encoder", "tokenizer", "text_encoder_2", "tokenizer_2",
        "image_encoder", "feature_extractor"
    ]

    def create_pipeline_stages(self, fastvideo_args: FastVideoArgs):
        """HyWorld preprocessing doesn't use staged pipeline execution."""
        pass

    @torch.no_grad()
    def forward(
        self,
        batch: ForwardBatch,
        fastvideo_args: FastVideoArgs,
        args,
    ):
        """
        Run the preprocessing pipeline.
        
        Reads raw manifest, processes each sample, and outputs .pt files.
        """
        if not self.post_init_called:
            self.post_init()

        device = get_local_torch_device()
        
        # Get precision settings
        vae_dtype = PRECISION_TO_TYPE.get(
            fastvideo_args.pipeline_config.vae_precision, torch.float16)
        text_encoder_precisions = getattr(
            fastvideo_args.pipeline_config, 'text_encoder_precisions',
            ('bf16', 'fp32'))
        llm_dtype = PRECISION_TO_TYPE.get(text_encoder_precisions[0],
                                          torch.bfloat16)
        byt5_dtype = PRECISION_TO_TYPE.get(text_encoder_precisions[1],
                                           torch.float32)

        # Load raw manifest
        manifest_path = Path(args.data_merge_path).expanduser().resolve()
        items = json.loads(manifest_path.read_text(encoding="utf-8"))

        # Output directories
        out_root = Path(args.output_dir).expanduser().resolve()
        latent_root = out_root / "latent_pt"
        latent_root.mkdir(parents=True, exist_ok=True)

        # Move models to device
        vae = self.get_module("vae").to(device)
        text_encoder = self.get_module("text_encoder").to(device)
        text_encoder_2 = self.get_module("text_encoder_2").to(device)
        tokenizer = self.get_module("tokenizer")
        tokenizer_2 = self.get_module("tokenizer_2")
        image_encoder = self.get_module("image_encoder").to(device)
        feature_extractor = self.get_module("feature_extractor")

        # Get preprocessing functions from config
        preprocess_text_funcs = getattr(
            fastvideo_args.pipeline_config, 'preprocess_text_funcs',
            (lambda x: x, lambda x: x))
        postprocess_text_funcs = getattr(
            fastvideo_args.pipeline_config, 'postprocess_text_funcs',
            (lambda x, m: (x.last_hidden_state, m), lambda x: x.last_hidden_state))
        text_encoder_max_lengths = getattr(
            fastvideo_args.pipeline_config, 'text_encoder_max_lengths',
            (1108, 256))
        
        # Get target resolution for bucket selection (default: 480p)
        target_resolution = getattr(
            fastvideo_args.pipeline_config, 'target_resolution', '480p')

        # Process each sample
        training_items = []
        
        for i, item in enumerate(tqdm(items, desc="Preprocessing samples")):
            sample_dir = latent_root / f"sample_{i:05d}"
            sample_dir.mkdir(parents=True, exist_ok=True)
            out_pt = sample_dir / "latent.pt"

            # Skip if already processed
            if out_pt.exists():
                logger.info(f"Skipping already processed sample {i}")
                # training_items.append(self._create_training_item(
                #     out_pt, item))
                # continue

            try:
                payload = self._process_sample(
                    item=item,
                    device=device,
                    vae=vae,
                    vae_dtype=vae_dtype,
                    text_encoder=text_encoder,
                    text_encoder_2=text_encoder_2,
                    tokenizer=tokenizer,
                    tokenizer_2=tokenizer_2,
                    image_encoder=image_encoder,
                    feature_extractor=feature_extractor,
                    llm_dtype=llm_dtype,
                    byt5_dtype=byt5_dtype,
                    preprocess_text_funcs=preprocess_text_funcs,
                    postprocess_text_funcs=postprocess_text_funcs,
                    text_encoder_max_lengths=text_encoder_max_lengths,
                    target_resolution=target_resolution,
                )

                torch.save(payload, out_pt)
                logger.info(f"[{i+1}/{len(items)}] Saved {out_pt}")
                
                training_items.append(self._create_training_item(out_pt, item))

            except Exception as e:
                logger.error(f"Error processing sample {i}: {e}")
                raise

        # Write training JSON
        training_json_path = out_root / "training_manifest.json"
        training_json_path.write_text(
            json.dumps(training_items, indent=2), encoding="utf-8")
        logger.info(f"Wrote {len(training_items)} items to {training_json_path}")

    def _create_training_item(self, latent_path: Path,
                              raw_item: dict) -> dict:
        """Create a training JSON item from raw manifest item."""
        return {
            "latent_path": str(latent_path),
            "pose_path": str(Path(raw_item["pose_path"]).expanduser().resolve()),
            "action_path": str(Path(raw_item["action_path"]).expanduser().resolve()),
        }

    def _process_sample(
        self,
        item: dict,
        device: torch.device,
        vae,
        vae_dtype: torch.dtype,
        text_encoder,
        text_encoder_2,
        tokenizer,
        tokenizer_2,
        image_encoder,
        feature_extractor,
        llm_dtype: torch.dtype,
        byt5_dtype: torch.dtype,
        preprocess_text_funcs: tuple,
        postprocess_text_funcs: tuple,
        text_encoder_max_lengths: tuple,
        target_resolution: str = "480p",
    ) -> dict:
        """
        Process a single sample and return the payload dict.
        
        Returns:
            dict with keys: latent, prompt_embeds, prompt_mask, image_cond,
                           vision_states, byt5_text_states, byt5_text_mask
        """
        video_path = str(item["video_path"])
        prompt = str(item.get("text") or item.get("caption") or
                     "a 3D scene with camera movement")
        num_frames = int(item.get("meta", {}).get("num_frames", 125))

        # 1. Read video frames
        frames = _read_video_frames(video_path, num_frames=num_frames)
        original_height, original_width = frames.shape[1], frames.shape[2]
        first_frame = Image.fromarray(frames[0])
        
        # Get closest bucket resolution (matching HY-WorldPlay behavior)
        height, width = get_closest_resolution(
            original_height, original_width, target_resolution)
        logger.debug(f"Resolution: {original_width}x{original_height} -> "
                     f"{width}x{height} (bucket for {target_resolution})")

        # 2. VAE encode full video
        video_input = _to_vae_input(frames, device=device, dtype=vae_dtype)
        with torch.autocast(device_type="cuda",
                           dtype=vae_dtype,
                           enabled=(device.type == "cuda")):
            latent_dist = vae.encode(video_input)
            # Handle different VAE output formats
            if hasattr(latent_dist, 'latent_dist'):
                latents = latent_dist.latent_dist.mode()
            elif hasattr(latent_dist, 'mean'):
                latents = latent_dist.mean
            else:
                latents = latent_dist
            
            # Apply scaling factor
            scaling_factor = getattr(vae.config, 'scaling_factor', 
                                    getattr(vae, 'scaling_factor', 0.476986))
            latents = latents * scaling_factor

        # 3. VAE encode first frame for image condition
        first_frame_input = _to_vae_input(
            frames[0:1], device=device, dtype=vae_dtype)
        with torch.autocast(device_type="cuda",
                           dtype=vae_dtype,
                           enabled=(device.type == "cuda")):
            first_frame_dist = vae.encode(first_frame_input)
            if hasattr(first_frame_dist, 'latent_dist'):
                image_cond = first_frame_dist.latent_dist.mode()
            elif hasattr(first_frame_dist, 'mean'):
                image_cond = first_frame_dist.mean
            else:
                image_cond = first_frame_dist
            image_cond = image_cond * scaling_factor

        # 4. LLM text encoding (Qwen 2.5 VL)
        llm_preprocess = preprocess_text_funcs[0]
        llm_postprocess = postprocess_text_funcs[0]
        
        processed_prompt = llm_preprocess(prompt)
        
        # Tokenize for LLM
        if isinstance(processed_prompt, list):
            # Chat format
            text_inputs = tokenizer.apply_chat_template(
                processed_prompt,
                tokenize=True,
                add_generation_prompt=False,
                return_tensors="pt",
                return_dict=True,
                padding="max_length",
                max_length=text_encoder_max_lengths[0],
                truncation=True,
            )
        else:
            text_inputs = tokenizer(
                processed_prompt,
                return_tensors="pt",
                padding="max_length",
                max_length=text_encoder_max_lengths[0],
                truncation=True,
            )

        input_ids = text_inputs["input_ids"].to(device)
        attention_mask = text_inputs["attention_mask"].to(device)

        with torch.autocast(device_type="cuda", dtype=llm_dtype):
            with set_forward_context(current_timestep=0, attn_metadata=None):
                llm_outputs = text_encoder(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                )

        prompt_embeds, prompt_mask = llm_postprocess(llm_outputs, attention_mask)

        # 5. ByT5 text encoding
        byt5_preprocess = preprocess_text_funcs[1]
        byt5_postprocess = postprocess_text_funcs[1]
        
        glyph_text = byt5_preprocess(prompt)
        
        # Always encode with ByT5, using empty string if no glyph text
        # This ensures consistent output shape (1, max_length, dim)
        byt5_input_text = glyph_text if glyph_text is not None else ""
        
        byt5_inputs = tokenizer_2(
            byt5_input_text,
            return_tensors="pt",
            padding="max_length",
            max_length=text_encoder_max_lengths[1],
            truncation=True,
        )
        byt5_input_ids = byt5_inputs["input_ids"].to(device)
        byt5_attention_mask = byt5_inputs["attention_mask"].to(device)

        with torch.autocast(device_type="cuda", dtype=byt5_dtype):
            with set_forward_context(current_timestep=0, attn_metadata=None):
                byt5_outputs = text_encoder_2(
                    input_ids=byt5_input_ids,
                    attention_mask=byt5_attention_mask,
                )
        byt5_text_states = byt5_postprocess(byt5_outputs)
        byt5_text_mask = byt5_attention_mask

        # 6. SigLIP vision encoding
        # Match inference exactly (fastvideo/pipelines/stages/image_encoding.py)
        from fastvideo.models.dits.hyworld.data_utils import resize_and_center_crop
        
        image_np = np.array(first_frame.convert("RGB"))
        
        # Resize to target resolution BEFORE SigLIP preprocessing
        image_np = resize_and_center_crop(image_np,
                                          target_width=width,
                                          target_height=height)
        
        # Get model dtype for proper precision matching (HY-WorldPlay uses fp16)
        model_dtype = next(image_encoder.parameters()).dtype
        
        image_inputs = feature_extractor.preprocess(
            images=image_np, return_tensors="pt").to(
                device=device, dtype=model_dtype)
        pixel_values = image_inputs['pixel_values']

        with set_forward_context(current_timestep=0, attn_metadata=None):
            vision_outputs = image_encoder(pixel_values=pixel_values)
        
        vision_states = vision_outputs.last_hidden_state

        # Build payload
        payload = {
            "latent": latents.detach().cpu(),
            "prompt_embeds": prompt_embeds.detach().cpu(),
            "prompt_mask": prompt_mask.detach().cpu(),
            "image_cond": image_cond.detach().cpu(),
            "vision_states": vision_states.detach().cpu(),
            "byt5_text_states": byt5_text_states.detach().cpu(),
            "byt5_text_mask": byt5_text_mask.detach().cpu(),
        }
        
        return payload


EntryClass = PreprocessPipeline_HYWorld
