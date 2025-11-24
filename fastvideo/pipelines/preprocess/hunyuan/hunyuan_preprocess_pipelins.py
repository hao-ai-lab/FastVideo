from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.pipelines.composed_pipeline_base import ComposedPipelineBase
from fastvideo.pipelines.preprocess.preprocess_stages import (
    TextTransformStage, VideoTransformStage)
from fastvideo.pipelines.stages import (EncodingStage, TextEncodingStage)
        

class PreprocessPipelineT2V(ComposedPipelineBase):
    _required_config_modules = [
        "text_encoder", "tokenizer", "text_encoder_2", "tokenizer_2", "vae"
    ]

    def create_pipeline_stages(self, fastvideo_args: FastVideoArgs):
        assert fastvideo_args.preprocess_config is not None
        
        self.add_stage(
            stage_name="text_transform_stage",
            stage=TextTransformStage(
                cfg_uncondition_drop_rate=fastvideo_args.preprocess_config.training_cfg_rate,
                seed=fastvideo_args.preprocess_config.seed,
            )
        )
        
        text_encoders = [
            self.get_module("text_encoder"),
            self.get_module("text_encoder_2")
        ]
        tokenizers = [
            self.get_module("tokenizer"),
            self.get_module("tokenizer_2")
        ]
        
        self.add_stage(
            stage_name="prompt_encoding_stage",
            stage=TextEncodingStage(
                text_encoders=text_encoders,
                tokenizers=tokenizers,
            )
        )
        
        self.add_stage(
            stage_name="video_transform_stage",
            stage=VideoTransformStage(
                train_fps=fastvideo_args.preprocess_config.train_fps,
                num_frames=fastvideo_args.preprocess_config.num_frames,
                max_height=fastvideo_args.preprocess_config.max_height,
                max_width=fastvideo_args.preprocess_config.max_width,
                do_temporal_sample=fastvideo_args.preprocess_config.do_temporal_sample,
            )
        )
        
        self.add_stage(
            stage_name="video_encoding_stage",
            stage=EncodingStage(vae=self.get_module("vae"))
        )

EntryClass = [PreprocessPipelineT2V]



