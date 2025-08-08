from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.pipelines.composed_pipeline_base import ComposedPipelineBase
from fastvideo.pipelines.preprocess.preprocess_stages import VideoTransformStage
from fastvideo.pipelines.stages import (EncodingStage, ImageEncodingStage,
                                        TextEncodingStage)


class I2VPreprocessPipeline(ComposedPipelineBase):
    _required_config_modules = [
        "image_encoder", "image_processor", "text_encoder", "tokenizer", "vae"
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
        self.add_stage(stage_name="video_encoding_stage",
                       stage=EncodingStage(vae=self.get_module("vae"), ))


class T2VPreprocessPipeline(ComposedPipelineBase):
    _required_config_modules = ["text_encoder", "tokenizer", "vae"]

    def create_pipeline_stages(self, fastvideo_args: FastVideoArgs):
        assert fastvideo_args.preprocess_config is not None

        self.add_stage(stage_name="prompt_encoding_stage",
                       stage=TextEncodingStage(
                           text_encoders=[self.get_module("text_encoder")],
                           tokenizers=[self.get_module("tokenizer")],
                       ))
        self.add_stage(
            stage_name="video_transform_stage",
            stage=VideoTransformStage(
                train_fps=fastvideo_args.preprocess_config.train_fps,
                num_frames=fastvideo_args.preprocess_config.num_frames,
                max_height=fastvideo_args.preprocess_config.max_height,
                max_width=fastvideo_args.preprocess_config.max_width,
                do_temporal_sample=fastvideo_args.preprocess_config.
                do_temporal_sample,
            ))
        self.add_stage(stage_name="video_encoding_stage",
                       stage=EncodingStage(vae=self.get_module("vae"), ))


EntryClass = [I2VPreprocessPipeline, T2VPreprocessPipeline]
