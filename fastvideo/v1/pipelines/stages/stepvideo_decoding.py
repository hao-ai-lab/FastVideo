import asyncio
from fastvideo.v1.fastvideo_args import FastVideoArgs
from fastvideo.v1.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.v1.pipelines.stages.base import PipelineStage
from fastvideo.models.stepvideo.utils import VideoProcessor

class StepVideoDecodingStage(PipelineStage):
    """
    After denoising, decode latents → video (via remote VAE)
    and then post-process exactly like v0.
    """
    def __init__(self, vae_client) -> None:
        super().__init__()
        self.vae_client = vae_client
        
    def forward(
        self,
        batch: ForwardBatch,
        fastvideo_args: FastVideoArgs,
    ) -> ForwardBatch:
        latents = batch.latents
        if latents is None:
            raise ValueError("Latents must be provided")

        # 1) decode or pass through
        if fastvideo_args.output_type != "latent":
            # move to CPU if needed, then reorder [B, F, C, H, W] → [B, C, F, H, W]
            latents_for_vae = latents.cpu().permute(0, 2, 1, 3, 4).contiguous()
            # print(f">>> decoding stage, {latents_for_vae.shape}")
            # print(latents_for_vae[0, 0, :5, 0, 0])
            video_bcftw = asyncio.run(self.vae_client(latents_for_vae))
            video_bfchw = video_bcftw.permute(0, 2, 1, 3, 4).contiguous()
            # 2) normalize from [-1,1] to [0,1]
            video = (video_bfchw / 2 + 0.5).clamp(0.0, 1.0)

            # 3) move to CPU float32
            video = video.cpu().float()
        else:
            video = latents

        # 2) offload all models/hooks
        if hasattr(self, 'maybe_free_model_hooks'):
            self.maybe_free_model_hooks()

        # 3) set the batch output
        batch.output = video
        return batch
