

from fastvideo.v1.dataset.preprocessing_datasets import TextEncodingStage
from fastvideo.v1.fastvideo_args import FastVideoArgs
from fastvideo.v1.pipelines.composed_pipeline_base import ComposedPipelineBase
from fastvideo.v1.pipelines.stages.image_encoding import ImageEncodingStage


class I2VPreprocessPipeline(ComposedPipelineBase):
    _required_config_modules = ["image_encoder", "image_processor", "text_encoder", "tokenizer"]

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


class T2VPreprocessPipeline(ComposedPipelineBase):
    _required_config_modules = ["text_encoder", "tokenizer"]

    def create_pipeline_stages(self, fastvideo_args: FastVideoArgs):
        self.add_stage(stage_name="prompt_encoding_stage",
                       stage=TextEncodingStage(
                           text_encoders=[self.get_module("text_encoder")],
                           tokenizers=[self.get_module("tokenizer")],
                       ))

EntryClass = [I2VPreprocessPipeline, T2VPreprocessPipeline]
