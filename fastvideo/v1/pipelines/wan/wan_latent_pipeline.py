from fastvideo.v1.fastvideo_args import FastVideoArgs
from fastvideo.v1.logger import init_logger
from fastvideo.v1.pipelines.composed_pipeline_base import ComposedPipelineBase
from fastvideo.v1.pipelines.pipeline_batch_info import ForwardBatch

logger = init_logger(__name__)


class WanLatentPipeline(ComposedPipelineBase):
    _required_config_modules = ["text_encoder", "tokenizer", "vae"]

    # def initialize_pipeline(self, fastvideo_args: FastVideoArgs):

    def create_pipeline_stages(self, fastvideo_args: FastVideoArgs):
        pass

    def forward(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs):
        logger.info("WAN Latent Pipeline forward")
        pass
