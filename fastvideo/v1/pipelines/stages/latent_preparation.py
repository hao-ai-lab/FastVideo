"""
Latent preparation stage for diffusion pipelines.
"""
from fastvideo.v1.pipelines.stages.base import PipelineStage
from fastvideo.v1.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.v1.inference_args import InferenceArgs
from diffusers.utils.torch_utils import randn_tensor
from fastvideo.v1.logger import init_logger

logger = init_logger(__name__)


class LatentPreparationStage(PipelineStage):
    """
    Stage for preparing initial latent variables for the diffusion process.
    
    This stage handles the preparation of the initial latent variables that will be
    denoised during the diffusion process.
    """

    def __init__(self, scheduler):
        super().__init__()
        self.scheduler = scheduler

    def forward(
        self,
        batch: ForwardBatch,
        inference_args: InferenceArgs,
    ) -> ForwardBatch:
        """
        Prepare initial latent variables for the diffusion process.
        
        Args:
            batch: The current batch information.
            inference_args: The inference arguments.
            
        Returns:
            The batch with prepared latent variables.
        """
        
        # Adjust video length based on VAE version if needed
        if hasattr(self, 'adjust_video_length'):
            batch = self.adjust_video_length(batch, inference_args)
        # Determine batch size
        if isinstance(batch.prompt, list):
            batch_size = len(batch.prompt)
        elif batch.prompt is not None:
            batch_size = 1
        else:
            batch_size = batch.prompt_embeds.shape[0]

        # Adjust batch size for number of videos per prompt
        batch_size *= batch.num_videos_per_prompt

        # Get required parameters
        dtype = batch.prompt_embeds[0].dtype
        device = batch.device
        generator = batch.generator
        latents = batch.latents
        num_frames = batch.num_frames
        height = batch.height
        width = batch.width

        # Calculate latent shape
        shape = (
            batch_size,
            inference_args.num_channels_latents,
            num_frames,
            int(height) // inference_args.vae_scale_factor,
            int(width) // inference_args.vae_scale_factor,
        )

        # Validate generator if it's a list
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        # Generate or use provided latents
        if latents is None:
            latents = randn_tensor(shape,
                                   generator=generator,
                                   device=device,
                                   dtype=dtype)
        else:
            latents = latents.to(device)

        # Scale the initial noise if needed
        if hasattr(self.scheduler, "init_noise_sigma"):
            latents = latents * self.scheduler.init_noise_sigma

        # Update batch with prepared latents
        batch.latents = latents


        return batch

    def adjust_video_length(self, batch: ForwardBatch,
                            inference_args: InferenceArgs) -> ForwardBatch:
        """
        Adjust video length based on VAE version.
        
        Args:
            batch: The current batch information.
            inference_args: The inference arguments.
            
        Returns:
            The batch with adjusted video length.
        """
        video_length = batch.num_frames
        # TODO
        batch.num_frames = (video_length - 1) // 4 + 1
        return batch
