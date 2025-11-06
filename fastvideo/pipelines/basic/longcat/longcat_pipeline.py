from fastvideo.logger import init_logger
from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.pipelines import ComposedPipelineBase, LoRAPipeline
from fastvideo.pipelines.stages import (
    InputValidationStage,
    TextEncodingStage,
    TimestepPreparationStage,
    LatentPreparationStage,
    DecodingStage
)
from .stages.longcat_denoising import LongCatDenoisingStage
from .stages.longcat_conditioning import LongCatConditioningStage
# TODO: KVCacheStage, RefineStage
try:
    from .stages.longcat_kvcache import LongCatKVCacheStage
except ImportError:
    LongCatKVCacheStage = None
try:
    from .stages.longcat_refine import LongCatRefineStage
except ImportError:
    LongCatRefineStage = None

logger = init_logger(__name__)

class LongCatPipeline(LoRAPipeline, ComposedPipelineBase):
    """
    LongCat Video Diffusion pipeline
    """
    
    _required_config_modules = [
        "text_encoder",
        "tokenizer",
        "vae",
        "transformer",
        "scheduler"
    ]
    
    def initialize_pipeline(self, fastvideo_args: FastVideoArgs):
        # TODO: this or just pass?
        return super().initialize_pipeline(fastvideo_args)
    
    def create_pipeline_stages(self, fastvideo_args: FastVideoArgs):
        
        self.add_stage(
            stage_name="input_validation_stage",
            stage=InputValidationStage()
        )
        
        self.add_stage(
            stage_name="prompt_encoding_stage",
            stage=TextEncodingStage(
                text_encoders=[self.get_module("text_encoder")],
                tokenizers=[self.get_module("tokenizer")],
            )
        )
        
        self.add_stage(
            stage_name="conditioning_stage",
            stage=LongCatConditioningStage(
                vae=self.get_module("vae"),
            )
        )
        
        self.add_stage(
            stage_name="timestep_preparation_stage",
            stage=TimestepPreparationStage(
                scheduler=self.get_module("scheduler"),
            )
        )
        
        self.add_stage(
            stage_name="latent_preparation_stage",
            stage=LatentPreparationStage(
                scheduler=self.get_module("scheduler"),
                transformer=self.get_module("transformer", None),
            )
        )
        
        if LongCatKVCacheStage is not None and fastvideo_args.pipeline_config.enable_kvcache:
            self.add_stage(
                stage_name="kv_cache_stage",
                stage=LongCatKVCacheStage(
                    transformer=self.get_module("transformer", None),
                )
            )
            
        self.add_stage(
            stage_name="denoising_stage",
            stage=LongCatDenoisingStage(
                transformer=self.get_module("transformer"),
                scheduler=self.get_module("scheduler"),
                vae=self.get_module("vae"),
                pipeline=self,
            )
        )
        
        self.add_stage(
            stage_name="decoding_stage",
            stage=DecodingStage(
                vae=self.get_module("vae"),
                pipeline=self,
            )
        )
        
EntryClass = LongCatPipeline

