from fastvideo.api.resolution import ResolvedGeneratorConfig
from fastvideo.pipelines.composed_pipeline_base import ComposedPipelineBase
from fastvideo.pipelines.preprocess.preprocess_stages import (TextTransformStage, VideoTransformStage)
from fastvideo.pipelines.stages import (EncodingStage, ImageEncodingStage, TextEncodingStage)
from fastvideo.pipelines.stages.image_encoding import ImageVAEEncodingStage


class PreprocessPipelineI2V(ComposedPipelineBase):
    _required_config_modules = ["image_encoder", "image_processor", "text_encoder", "tokenizer", "vae"]

    def create_pipeline_stages(self, resolved_config: ResolvedGeneratorConfig):
        self.add_stage(stage_name="text_transform_stage",
                       stage=TextTransformStage(
                           cfg_uncondition_drop_rate=resolved_config.preprocess.training_cfg_rate,
                           seed=resolved_config.preprocess.seed,
                       ))
        self.add_stage(stage_name="prompt_encoding_stage",
                       stage=TextEncodingStage(
                           text_encoders=[self.get_module("text_encoder")],
                           tokenizers=[self.get_module("tokenizer")],
                       ))
        self.add_stage(stage_name="video_transform_stage",
                       stage=VideoTransformStage(
                           train_fps=resolved_config.preprocess.train_fps,
                           num_frames=resolved_config.preprocess.num_frames,
                           max_height=resolved_config.preprocess.max_height,
                           max_width=resolved_config.preprocess.max_width,
                           do_temporal_sample=resolved_config.preprocess.do_temporal_sample,
                       ))
        if (self.get_module("image_encoder") is not None and self.get_module("image_processor") is not None):
            self.add_stage(stage_name="image_encoding_stage",
                           stage=ImageEncodingStage(
                               image_encoder=self.get_module("image_encoder"),
                               image_processor=self.get_module("image_processor"),
                           ))
        self.add_stage(stage_name="image_vae_encoding_stage", stage=ImageVAEEncodingStage(vae=self.get_module("vae"), ))
        self.add_stage(stage_name="video_encoding_stage", stage=EncodingStage(vae=self.get_module("vae"), ))


class PreprocessPipelineT2V(ComposedPipelineBase):
    _required_config_modules = ["text_encoder", "tokenizer", "vae"]

    def create_pipeline_stages(self, resolved_config: ResolvedGeneratorConfig):
        self.add_stage(stage_name="text_transform_stage",
                       stage=TextTransformStage(
                           cfg_uncondition_drop_rate=resolved_config.preprocess.training_cfg_rate,
                           seed=resolved_config.preprocess.seed,
                       ))
        self.add_stage(stage_name="prompt_encoding_stage",
                       stage=TextEncodingStage(
                           text_encoders=[self.get_module("text_encoder")],
                           tokenizers=[self.get_module("tokenizer")],
                       ))
        self.add_stage(stage_name="video_transform_stage",
                       stage=VideoTransformStage(
                           train_fps=resolved_config.preprocess.train_fps,
                           num_frames=resolved_config.preprocess.num_frames,
                           max_height=resolved_config.preprocess.max_height,
                           max_width=resolved_config.preprocess.max_width,
                           do_temporal_sample=resolved_config.preprocess.do_temporal_sample,
                       ))
        self.add_stage(stage_name="video_encoding_stage", stage=EncodingStage(vae=self.get_module("vae"), ))


EntryClass = [PreprocessPipelineI2V, PreprocessPipelineT2V]
