# SPDX-License-Identifier: Apache-2.0
"""Kandinsky6 video super-resolution (VSR) pipeline: source video -> x2 / x2.25 / x4 video with the source audio.

The model is text-free.  An official ``Kandinsky6SRPipeline`` bundle holds ``transformer`` (SR DiT), ``vae`` (KVAE),
``latent_upscaler`` (x2 / x4 latent upscalers) and ``scheduler``: ``FlowMatchEulerDiscreteScheduler`` for
``Kandinsky-6.0-VSR-5s-Diffusers`` and ``PiflowScheduler`` for the 2-step distilled
``Kandinsky-6.0-VSR-distilled2steps-5s-Diffusers``.  See ``docs/inference/kandinsky6_sr.md``.
"""
from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.pipelines.composed_pipeline_base import ComposedPipelineBase
from fastvideo.pipelines.stages.kandinsky6_sr import (Kandinsky6SRDecodingStage, Kandinsky6SRDenoisingStage,
                                                      Kandinsky6SRLatentPreparationStage,
                                                      Kandinsky6SRVideoEncodingStage)


class Kandinsky6SRPipeline(ComposedPipelineBase):

    _required_config_modules = ["transformer", "vae", "latent_upscaler", "scheduler"]

    def create_pipeline_stages(self, fastvideo_args: FastVideoArgs) -> None:
        transformer = self.get_module("transformer")
        vae = self.get_module("vae")
        self.add_stage(stage_name="video_encoding_stage", stage=Kandinsky6SRVideoEncodingStage(vae=vae))
        self.add_stage(stage_name="latent_preparation_stage",
                       stage=Kandinsky6SRLatentPreparationStage(latent_upscaler=self.get_module("latent_upscaler"),
                                                                vae=vae,
                                                                transformer=transformer))
        self.add_stage(stage_name="denoising_stage",
                       stage=Kandinsky6SRDenoisingStage(transformer=transformer,
                                                        scheduler=self.get_module("scheduler")))
        self.add_stage(stage_name="decoding_stage", stage=Kandinsky6SRDecodingStage(vae=vae, transformer=transformer))


EntryClass = Kandinsky6SRPipeline
