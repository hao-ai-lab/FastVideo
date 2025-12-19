import torch
from pathlib import Path
from .autoencoder_kl_causal_3d import AutoencoderKLCausal3D
from ..constants import VAE_PATH, PRECISION_TO_TYPE

def load_vae(vae_type,
             vae_precision=None,
             sample_size=None,
             vae_path=None,
             logger=None,
             device=None
             ):
    """
    Load and configure a Variational Autoencoder (VAE) model.
    
    This function handles loading 3D causal VAE models, including configuration,
    weight loading, precision setting, and device placement. It ensures the model
    is properly initialized for inference.

    Parameters:
        vae_type (str): Type identifier for the VAE, must follow '???-*' format for 3D VAEs
        vae_precision (str, optional): Desired precision type (e.g., 'fp16', 'fp32'). 
                                     Uses model's default if not specified.
        sample_size (tuple, optional): Input sample dimensions to override config defaults
        vae_path (str, optional): Path to VAE model files. Uses predefined path from
                                VAE_PATH constant if not specified.
        logger (logging.Logger, optional): Logger instance for progress/debug messages
        device (torch.device, optional): Target device to place the model (e.g., 'cuda' or 'cpu')

    Returns:
        tuple: Contains:
            - vae (AutoencoderKLCausal3D): Loaded and configured VAE model
            - vae_path (str): Actual path used to load the VAE
            - spatial_compression_ratio (int): Spatial dimension compression factor
            - time_compression_ratio (int): Temporal dimension compression factor

    Raises:
        ValueError: If vae_type does not follow the required 3D VAE format '???-*'
    """
    if vae_path is None:
        vae_path = VAE_PATH[vae_type]
    vae_compress_spec, _, _ = vae_type.split("-")
    length = len(vae_compress_spec)
    # Process 3D VAE (valid format with 3-character compression spec)
    if length == 3:
        if logger is not None:
            logger.info(f"Loading 3D VAE model ({vae_type}) from: {vae_path}")
        config = AutoencoderKLCausal3D.load_config(vae_path)
        if sample_size:
            vae = AutoencoderKLCausal3D.from_config(config, sample_size=sample_size)
        else:
            vae = AutoencoderKLCausal3D.from_config(config)
        ckpt = torch.load(Path(vae_path) / "pytorch_model.pt", map_location=vae.device)
        if "state_dict" in ckpt:
            ckpt = ckpt["state_dict"]
        vae_ckpt = {k.replace("vae.", ""): v for k, v in ckpt.items() if k.startswith("vae.")}
        vae.load_state_dict(vae_ckpt)

        spatial_compression_ratio = vae.config.spatial_compression_ratio
        time_compression_ratio = vae.config.time_compression_ratio
    else:
        raise ValueError(f"Invalid VAE model: {vae_type}. Must be 3D VAE in the format of '???-*'.")

    if vae_precision is not None:
        vae = vae.to(dtype=PRECISION_TO_TYPE[vae_precision])

    vae.requires_grad_(False)

    if logger is not None:
        logger.info(f"VAE to dtype: {vae.dtype}")

    if device is not None:
        vae = vae.to(device)

    # Ensure model is in evaluation mode (disables dropout/batch norm training behavior)
    # Note: Even with dropout rate 0, eval mode is recommended for consistent inference
    vae.eval()

    return vae, vae_path, spatial_compression_ratio, time_compression_ratio
