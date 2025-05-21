import numpy as np

from fastvideo.v1.fastvideo_args import FastVideoArgs
from fastvideo.v1.logger import init_logger
from fastvideo.v1.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.v1.pipelines.stages.base import PipelineStage

logger = init_logger(__name__)


class TimestepsPreparationPreStage(PipelineStage):

    def __init__(self, scheduler) -> None:
        super().__init__()
        self.scheduler = scheduler

    def calculate_shift(self,
                        image_seq_len,
                        base_seq_len: int = 256,
                        max_seq_len: int = 4096,
                        base_shift: float = 0.5,
                        max_shift: float = 1.15):
        m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
        b = base_shift - m * base_seq_len
        mu = image_seq_len * m + b
        return mu

    def forward(self, batch: ForwardBatch,
                fastvideo_args: FastVideoArgs) -> ForwardBatch:
        assert batch.height is not None
        assert batch.width is not None

        batch.sigmas = np.linspace(
            1.0, 1 / batch.num_inference_steps,
            batch.num_inference_steps) if batch.sigmas is None else batch.sigmas
        spatial_compression_ratio = fastvideo_args.vae_config.arch_config.spatial_compression_ratio
        batch.extra_set_timesteps_kwargs["mu"] = (
            batch.extra_set_timesteps_kwargs.get("mu", None)
            or self.calculate_shift(
                (batch.height // spatial_compression_ratio // 2) *
                (batch.width // spatial_compression_ratio // 2),
                self.scheduler.config.get("base_image_seq_len", 256),
                self.scheduler.config.get("max_image_seq_len", 4096),
                self.scheduler.config.get("base_shift", 0.5),
                self.scheduler.config.get("max_shift", 1.15),
            ))
        return batch


class DenoisingPreprocessingStage(PipelineStage):

    def __init__(self) -> None:
        super().__init__()

    def forward(self, batch: ForwardBatch,
                fastvideo_args: FastVideoArgs) -> ForwardBatch:
        # [B, in_channels // 4, 1, H, W] -> [B, H // 2, W // 2, in_channels]
        assert batch.latents is not None
        b, c, _, h, w = batch.latents.shape
        latents = batch.latents.view(b, c, h // 2, 2, w // 2, 2)
        latents = latents.permute(0, 2, 4, 1, 3, 5)
        latents = latents.reshape(b, (h // 2) * (w // 2), c * 4)
        batch.latents = latents
        return batch


class DenoisingPostprocessingStage(PipelineStage):

    def __init__(self) -> None:
        super().__init__()

    def forward(self, batch: ForwardBatch,
                fastvideo_args: FastVideoArgs) -> ForwardBatch:
        assert batch.latents is not None
        assert batch.height_latents is not None
        assert batch.width_latents is not None

        latents = batch.latents
        # Skip decoding if output type is latent
        if fastvideo_args.output_type == "latent":
            latents = latents
        else:
            # [B, (H // 2) * (W // 2), in_channels] -> [B, in_channels // 4, 1, H, W]
            b, _, c = latents.shape
            h, w = batch.height_latents, batch.width_latents
            latents = latents.view(b, h // 2, w // 2, c // 4, 2, 2)
            latents = latents.permute(0, 3, 1, 4, 2, 5)
            latents = latents.reshape(b, c // 4, h, w)
            # latents = latents.squeeze(2)

        batch.latents = latents
        return batch
