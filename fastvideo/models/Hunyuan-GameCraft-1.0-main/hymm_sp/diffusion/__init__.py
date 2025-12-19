from .pipelines import HunyuanVideoGamePipeline
from .schedulers import FlowMatchDiscreteScheduler

def load_diffusion_pipeline(args, rank, vae, text_encoder, text_encoder_2, model, scheduler=None,
                            device=None, progress_bar_config=None):
    """ Load the denoising scheduler for inference. """
    if scheduler is None:
        scheduler = FlowMatchDiscreteScheduler(
            shift=args.flow_shift_eval_video,
            reverse=args.flow_reverse,
            solver=args.flow_solver,
        )
    # Only enable progress bar for rank 0
    progress_bar_config = progress_bar_config or {'leave': True, 'disable': rank != 0}

    pipeline = HunyuanVideoGamePipeline(vae=vae,
                                       text_encoder=text_encoder,
                                       text_encoder_2=text_encoder_2,
                                       transformer=model,
                                       scheduler=scheduler,
                                    #    safety_checker=None,
                                    #    feature_extractor=None,
                                    #    requires_safety_checker=False,
                                       progress_bar_config=progress_bar_config,
                                       args=args,
                                       )
    if not args.cpu_offload:
        pipeline = pipeline.to(device)

    return pipeline
