# SPDX-License-Identifier: Apache-2.0
"""Bernini-R single-video editing, using FastVideo-owned components."""

from fastvideo.pipelines.composed_pipeline_base import ComposedPipelineBase
from fastvideo.pipelines.lora_pipeline import LoRAPipeline
from fastvideo.pipelines.stages import DecodingStage, TextEncodingStage
from fastvideo.pipelines.basic.bernini.stages import (BerniniDenoisingStage, BerniniInputStage,
                                                      BerniniVideoEncodingStage)


class BerniniRPipeline(LoRAPipeline, ComposedPipelineBase):
    _required_config_modules = ["text_encoder", "tokenizer", "transformer", "vae", "scheduler"]

    def load_modules(self, fastvideo_args, loaded_modules=None):
        if fastvideo_args.sp_size != 1 or fastvideo_args.tp_size != 1 or fastvideo_args.num_gpus != 1:
            raise ValueError("Bernini-R currently requires num_gpus=sp_size=tp_size=1")
        if fastvideo_args.pipeline_config.vae_precision != "fp32":
            raise ValueError("Bernini-R requires FP32 VAE encoding; set vae_precision='fp32'")
        return super().load_modules(fastvideo_args, loaded_modules)

    def create_pipeline_stages(self, fastvideo_args):
        self.add_stage("input_validation_stage", BerniniInputStage())
        self.add_stage(
            "prompt_encoding_stage",
            TextEncodingStage(text_encoders=[self.get_module("text_encoder")],
                              tokenizers=[self.get_module("tokenizer")]))
        self.add_stage("video_encoding_stage", BerniniVideoEncodingStage(self.get_module("vae")))
        self.add_stage("denoising_stage",
                       BerniniDenoisingStage(self.get_module("transformer"), self.get_module("scheduler")))
        self.add_stage("decoding_stage", DecodingStage(self.get_module("vae")))


EntryClass = BerniniRPipeline
