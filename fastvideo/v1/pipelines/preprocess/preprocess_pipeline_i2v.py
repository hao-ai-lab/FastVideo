# SPDX-License-Identifier: Apache-2.0
"""
I2V Data Preprocessing pipeline implementation.

This module contains an implementation of the I2V Data Preprocessing pipeline
using the modular pipeline architecture.
"""
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import PIL
from PIL import Image
import os
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import gc
import pyarrow as pa
from fastvideo.v1.configs.sample import SamplingParam
from fastvideo.v1.pipelines.pipeline_batch_info import ForwardBatch

from fastvideo.v1.dataset.dataloader.schema import pyarrow_schema_i2v
from fastvideo.v1.distributed import get_torch_device
from fastvideo.v1.fastvideo_args import FastVideoArgs
from fastvideo.v1.forward_context import set_forward_context
from fastvideo.v1.pipelines.preprocess_pipeline_base import (
    BasePreprocessPipeline)
from fastvideo.v1.models.vision_utils import numpy_to_pt, pil_to_numpy, normalize
from fastvideo.v1.dataset.validation_dataset import ValidationDataset
from fastvideo.v1.pipelines.stages import TextEncodingStage, ImageEncodingStage

from fastvideo.v1.logger import init_logger

logger = init_logger(__name__)


class PreprocessPipeline_I2V(BasePreprocessPipeline):
    """I2V preprocessing pipeline implementation."""

    _required_config_modules = [
        "text_encoder", "tokenizer", "vae", "image_encoder", "image_processor"
    ]

    def create_pipeline_stages(self, fastvideo_args: FastVideoArgs):
        self.add_stage(stage_name="prompt_encoding_stage",
                       stage=TextEncodingStage(
                           text_encoders=[self.get_module("text_encoder")],
                           tokenizers=[self.get_module("tokenizer")],
                       ))

        self.add_stage(stage_name="image_encoding_stage",
                       stage=ImageEncodingStage(
                           image_encoder=self.get_module("image_encoder"),
                           image_processor=self.get_module("image_processor"),
                       ))


    def preprocess_validation_text(self, fastvideo_args: FastVideoArgs, args):
        """Process validation text prompts and save them to parquet files.
        
        This base implementation handles the common validation text processing logic.
        Subclasses can override this method to add pipeline-specific features.
        """
        # Create Parquet dataset directory for validation
        validation_parquet_dir = os.path.join(args.output_dir,
                                              "validation_parquet_dataset")
        os.makedirs(validation_parquet_dir, exist_ok=True)

        validation_dataset = ValidationDataset(args.validation_dataset_file)

        from itertools import chain


        # for data in validation_dataset:
            # print(data)

        # assert False

        # with open(args.validation_prompt_txt, encoding="utf-8") as file:
        #     lines = file.readlines()
        # prompts = [line.strip() for line in lines]

        # Prepare batch data for Parquet dataset
        batch_data = []
        sampling_param = SamplingParam.from_pretrained(
            fastvideo_args.model_path)
        if sampling_param.negative_prompt:
            negative_prompt = {
                'caption': sampling_param.negative_prompt,
                'image_path': None,
                'video_path': None,
            }
        else:
            negative_prompt = None

        validation_dataset = chain([negative_prompt], validation_dataset)

        



        # Add progress bar for validation text preprocessing
        pbar = tqdm(enumerate(validation_dataset),
                    desc="Processing validation dataset",
                    unit="sample")
        for idx, row in pbar:
            print(idx)
            # print(type(row))
            # print(row)
            prompt = row['caption']
            print(row)
            print(prompt)


            with torch.inference_mode():
                # Text Encoder
                batch = ForwardBatch(
                    data_type="video",
                    prompt=prompt,
                    prompt_embeds=[],
                    prompt_attention_mask=[],
                )
                assert hasattr(self, "prompt_encoding_stage")
                result_batch = self.prompt_encoding_stage(batch, fastvideo_args)

                if idx != 0:
                    image = None
                    if 'image' in row:
                        image = row['image']
                    else:
                        assert 'video' in row
                        image = row['video'][0]
                    
                    assert image is not None
                    result_batch.pil_image = image

                    # assert image is not None
                    assert hasattr(self, "image_encoding_stage")
                    result_batch = self.image_encoding_stage(result_batch, fastvideo_args)

            prompt_embeds = result_batch.prompt_embeds[0]
            prompt_attention_mask = result_batch.prompt_attention_mask[0]

            file_name = prompt.split(".")[0]

            # Get the sequence length from attention mask (number of 1s)
            seq_len = prompt_attention_mask.sum().item()

            text_embedding = prompt_embeds[0, :seq_len].cpu().numpy()
            text_attention_mask = prompt_attention_mask[
                0, :seq_len].cpu().numpy().astype(np.uint8)

            if idx != 0:
                image_embeds = result_batch.image_embeds[0]
                image_embeds = image_embeds.cpu().numpy()
            else:
                image_embeds = None

            # Log the shapes after removing padding
            logger.info(
                "Shape after removing padding - Embeddings: %s, Mask: %s",
                text_embedding.shape, text_attention_mask.shape)


            # image embedding
            if idx != 0:
                clip_feature = {
                    'clip_feature': image_embeds,
                }
            else:
                clip_feature = None

            # Create record for Parquet dataset
            record = self.create_record(video_name=file_name,
                                        vae_latent=np.array([],
                                                            dtype=np.float32),
                                        text_embedding=text_embedding,
                                        text_attention_mask=text_attention_mask,
                                        valid_data=None,
                                        idx=0,
                                        extra_features=clip_feature)
            batch_data.append(record)

            logger.info("Saved validation sample: %s", file_name)

        if batch_data:
            # Add progress bar for writing to Parquet dataset
            write_pbar = tqdm(total=1,
                              desc="Writing to Parquet dataset",
                              unit="batch")
            # Convert batch data to PyArrow arrays
            arrays = []
            for field in self.get_schema_fields():
                if field.endswith('_bytes'):
                    arrays.append(
                        pa.array([record[field] for record in batch_data],
                                 type=pa.binary()))
                elif field.endswith('_shape'):
                    arrays.append(
                        pa.array([record[field] for record in batch_data],
                                 type=pa.list_(pa.int32())))
                elif field in ['width', 'height', 'num_frames']:
                    arrays.append(
                        pa.array([record[field] for record in batch_data],
                                 type=pa.int32()))
                elif field in ['duration_sec', 'fps']:
                    arrays.append(
                        pa.array([record[field] for record in batch_data],
                                 type=pa.float32()))
                else:
                    arrays.append(
                        pa.array([record[field] for record in batch_data]))

            table = pa.Table.from_arrays(arrays, names=self.get_schema_fields())
            write_pbar.update(1)
            write_pbar.close()

            logger.info("Total validation samples: %s", len(table))

            work_range = (0, 1, table, 0, validation_parquet_dir, len(table))

            total_written = 0
            failed_ranges = []
            with ProcessPoolExecutor(max_workers=1) as executor:
                futures = {
                    executor.submit(self.process_chunk_range, work_range):
                    work_range
                }
                for future in tqdm(futures, desc="Processing chunks"):
                    try:
                        total_written += future.result()
                    except Exception as e:
                        work_range = futures[future]
                        failed_ranges.append(work_range)
                        logger.error("Failed to process range %s-%s: %s",
                                     work_range[0], work_range[1], str(e))

            if failed_ranges:
                logger.warning("Retrying %s failed ranges sequentially",
                               len(failed_ranges))
                for work_range in failed_ranges:
                    try:
                        total_written += self.process_chunk_range(work_range)
                    except Exception as e:
                        logger.error(
                            "Failed to process range %s-%s after retry: %s",
                            work_range[0], work_range[1], str(e))

            logger.info("Total validation samples written: %s", total_written)

            # Clear memory
            del table
            gc.collect()  # Force garbage collection

    def get_schema_fields(self) -> List[str]:
        """Get the schema fields for I2V pipeline."""
        return [f.name for f in pyarrow_schema_i2v]

    def get_extra_features(self, valid_data: Dict[str, Any],
                           fastvideo_args: FastVideoArgs) -> Dict[str, Any]:
        features = {}
        """Get CLIP features from the first frame of each video."""
        first_frame = valid_data["pixel_values"][:, :, 0, :, :].permute(
            0, 2, 3, 1)  # (B, C, T, H, W) -> (B, H, W, C)
        batch_size, _, num_frames, height, width = valid_data["pixel_values"].shape
        latent_height = height // self.get_module("vae").spatial_compression_ratio
        latent_width = width // self.get_module("vae").spatial_compression_ratio

        processed_images = []
        # Frame has values between -1 and 1
        for frame in first_frame:
            frame = (frame + 1) * 127.5
            frame_pil = Image.fromarray(frame.cpu().numpy().astype(np.uint8))
            processed_img = self.get_module("image_processor")(
                images=frame_pil, return_tensors="pt")
            processed_images.append(processed_img)

        # Get CLIP features
        pixel_values = torch.cat(
            [img['pixel_values'] for img in processed_images],
            dim=0).to(get_torch_device())
        with torch.no_grad():
            image_inputs = {'pixel_values': pixel_values}
            with set_forward_context(current_timestep=0, attn_metadata=None):
                clip_features = self.get_module("image_encoder")(**image_inputs)
            clip_features = clip_features.last_hidden_state
        
        features["clip_feature"] = clip_features

        """Get VAE features from the first frame of each video"""
        video_conditions = []
        for frame in first_frame:
            processed_img = frame.to(device="cpu", dtype=torch.float32)
            processed_img = processed_img.unsqueeze(0).permute(0, 3, 1, 2).unsqueeze(2)
            # (B, H, W, C) -> (B, C, 1, H, W)
            video_condition = torch.cat([
                processed_img,
                processed_img.new_zeros(processed_img.shape[0], processed_img.shape[1],
                                num_frames - 1, height, width)
            ],
                                        dim=2)
            video_condition = video_condition.to(device=get_torch_device(),
                                                dtype=torch.float32)
            video_conditions.append(video_condition)
        
        video_conditions = torch.cat(video_conditions, dim=0)

        with torch.autocast(device_type="cuda",
                            dtype=torch.float32,
                            enabled=True):
            encoder_outputs = self.get_module("vae").encode(video_conditions)

        latent_condition = encoder_outputs.mean
        if (hasattr(self.get_module("vae"), "shift_factor")
                and self.get_module("vae").shift_factor is not None):
            if isinstance(self.get_module("vae").shift_factor, torch.Tensor):
                latent_condition -= self.get_module("vae").shift_factor.to(
                    latent_condition.device, latent_condition.dtype)
            else:
                latent_condition -= self.get_module("vae").shift_factor

        if isinstance(self.get_module("vae").scaling_factor, torch.Tensor):
            latent_condition = latent_condition * self.get_module("vae").scaling_factor.to(
                latent_condition.device, latent_condition.dtype)
        else:
            latent_condition = latent_condition * self.get_module("vae").scaling_factor

        mask_lat_size = torch.ones(batch_size, 1, num_frames, latent_height,
                                   latent_width)
        mask_lat_size[:, :, list(range(1, num_frames))] = 0
        first_frame_mask = mask_lat_size[:, :, 0:1]
        first_frame_mask = torch.repeat_interleave(
            first_frame_mask,
            dim=2,
            repeats=self.get_module("vae").temporal_compression_ratio)
        mask_lat_size = torch.concat(
            [first_frame_mask, mask_lat_size[:, :, 1:, :]], dim=2)
        mask_lat_size = mask_lat_size.view(batch_size, -1,
                                           self.get_module("vae").temporal_compression_ratio,
                                           latent_height, latent_width)
        mask_lat_size = mask_lat_size.transpose(1, 2)
        mask_lat_size = mask_lat_size.to(latent_condition.device)

        image_latent = torch.concat([mask_lat_size, latent_condition],
                                          dim=1)

        features["encoded_first_frame"] = image_latent

        return features

    def create_record(
            self,
            video_name: str,
            vae_latent: np.ndarray,
            text_embedding: np.ndarray,
            text_attention_mask: np.ndarray,
            valid_data: Optional[Dict[str, Any]],
            idx: int,
            extra_features: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Create a record for the Parquet dataset with CLIP features."""
        record = super().create_record(video_name=video_name,
                                       vae_latent=vae_latent,
                                       text_embedding=text_embedding,
                                       text_attention_mask=text_attention_mask,
                                       valid_data=valid_data,
                                       idx=idx,
                                       extra_features=extra_features)

        if extra_features and "clip_feature" in extra_features:
            clip_feature = extra_features["clip_feature"]
            record.update({
                "clip_feature_bytes": clip_feature.tobytes(),
                "clip_feature_shape": list(clip_feature.shape),
                "clip_feature_dtype": str(clip_feature.dtype),
            })
        else:
            record.update({
                "clip_feature_bytes": b"",
                "clip_feature_shape": [],
                "clip_feature_dtype": "",
            })

        if extra_features and "encoded_first_frame" in extra_features:
            encoded_first_frame = extra_features["encoded_first_frame"]
            record.update({
                "encoded_first_frame_bytes": encoded_first_frame.tobytes(),
                "encoded_first_frame_shape": list(encoded_first_frame.shape),
                "encoded_first_frame_dtype": str(encoded_first_frame.dtype),
            })
        else:
            record.update({
                "encoded_first_frame_bytes": b"",
                "encoded_first_frame_shape": [],
                "encoded_first_frame_dtype": "",
            })

        return record


EntryClass = PreprocessPipeline_I2V
