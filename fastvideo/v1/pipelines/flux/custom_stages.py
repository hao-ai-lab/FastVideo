import torch

from fastvideo.v1.fastvideo_args import FastVideoArgs
from fastvideo.v1.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.v1.pipelines.stages.base import PipelineStage

class DenoisingPreprocessingStage(PipelineStage):
    def __init__(self):
        super().__init__()

    def forward(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> ForwardBatch:
        # [B, in_channels // 4, 1, H, W] -> [B, H // 2, W // 2, in_channels]
        b, c, _, h, w = batch.latents.shape
        latents = batch.latents.view(b, c, h // 2, 2, w // 2, 2)
        latents = latents.permute(0, 2, 4, 1, 3, 5)
        latents = latents.reshape(b, (h // 2) * (w // 2), c * 4)
        batch.latents = latents
        return batch

class DenoisingPostprocessingStage(PipelineStage):
    def __init__(self):
        super().__init__()

    def forward(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> ForwardBatch:
        latents = batch.latents
        # Skip decoding if output type is latent
        if fastvideo_args.output_type == "latent":
            latents = latents
        else:
            # [B, H // 2, W // 2, in_channels] -> [B, in_channels // 4, 1, H, W]
            # b, h, w, c = latents.shape
            # latents = latents.view(b, h, w, c // 4, 2, 2)
            # latents = latents.permute(0, 3, 1, 4, 2, 5)
            # latents = latents.reshape(b, c // 4, h * 2, w * 2)
            latents = latents.squeeze(2)

        batch.latents = latents
        return batch

class ImageOutputStage(PipelineStage):
    def __init__(self):
        super().__init__()

    def forward(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> ForwardBatch:
        output = batch.output
        output = output.unsqueeze(2)
        batch.output = output
        return batch