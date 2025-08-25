# SPDX-License-Identifier: Apache-2.0
"""
ODE Trajectory Data Preprocessing pipeline implementation.

This module contains an implementation of the ODE Trajectory Data Preprocessing pipeline
using the modular pipeline architecture.

Sec 4.3 of CausVid paper: https://arxiv.org/pdf/2412.07772
"""

from fastvideo.dataset.dataloader.schema import (
    pyarrow_schema_t2v_ode_trajectory)
from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.pipelines.preprocess.preprocess_pipeline_base import (
    BasePreprocessPipeline)
from fastvideo.pipelines.stages import DenoisingStage, TextEncodingStage


class PreprocessPipeline_ODE_Trajectory(BasePreprocessPipeline):
    """ODE Trajectory preprocessing pipeline implementation."""

    _required_config_modules = [
        "text_encoder", "tokenizer", "vae", "transformer"
    ]

    def get_schema_fields(self):
        """Get the schema fields for ODE Trajectory pipeline."""
        return [f.name for f in pyarrow_schema_t2v_ode_trajectory]

    def create_pipeline_stages(self, fastvideo_args: FastVideoArgs):
        """Set up pipeline stages with proper dependency injection."""
        self.add_stage(stage_name="prompt_encoding_stage",
                       stage=TextEncodingStage(
                           text_encoders=[self.get_module("text_encoder")],
                           tokenizers=[self.get_module("tokenizer")],
                       ))
        self.add_stage(
            stage_name="denoising_stage",
            stage=DenoisingStage(
                transformer=[self.get_module("transformer")],
                transformer_2=[self.get_module("transformer_2", None)],
                scheduler=[self.get_module("scheduler")],
                pipeline=self,
            ))

    def forward(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs, args):
        if not self.post_init_called:
            self.post_init()

        # Initialize class variables for data sharing
        self.video_data: dict[str, Any] = {}  # Store video metadata and paths
        self.latent_data: dict[str, Any] = {}  # Store latent tensors
        self.preprocess_video_and_text_ode_trajectory(fastvideo_args, args)

        sampling_params = {
            "num_inference_steps": 50,
            "guidance_scale": 7.5,
            "guidance_rescale": 0.0,
            "do_classifier_free_guidance": False,
        }
