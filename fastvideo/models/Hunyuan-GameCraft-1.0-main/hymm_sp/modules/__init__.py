from .models import HYVideoDiffusionTransformer, HUNYUAN_VIDEO_CONFIG
"""
This module provides functionality to load the Hunyuan video diffusion transformer model,
which is used for video generation tasks with specific configurations and parameters.
"""

def load_model(args, in_channels, out_channels, factor_kwargs):
    """
    Load and initialize the HYVideoDiffusionTransformer model with specified parameters.

    Args:
        args: Command-line arguments or configuration object containing model settings.
            Must include 'model' attribute to select the appropriate configuration.
        in_channels (int): Number of input channels for the model.
        out_channels (int): Number of output channels the model should produce.
        factor_kwargs (dict): Additional keyword arguments for factor adjustments
            in the model architecture.

    Returns:
        HYVideoDiffusionTransformer: Initialized instance of the video diffusion transformer
            model with the specified configuration.

    Notes:
        - Uses the HUNYUAN_VIDEO_CONFIG dictionary to retrieve model-specific configurations
          based on the model name provided in args.
        - Sets multitask_mask_training_type to "concat" as a default for this loading setup.
    """
    # Initialize the Hunyuan video diffusion transformer with combined configurations
    # Merges base config from HUNYUAN_VIDEO_CONFIG and additional factor arguments
    model = HYVideoDiffusionTransformer(
        args,
        in_channels=in_channels,
        out_channels=out_channels,
        multitask_mask_training_type="concat",
        **HUNYUAN_VIDEO_CONFIG[args.model],  # Unpack model-specific configuration
        ** factor_kwargs,  # Unpack additional factor adjustments
    )
    return model
