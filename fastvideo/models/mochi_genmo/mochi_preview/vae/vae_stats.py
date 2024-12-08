import torch

# Channel-wise mean and standard deviation of VAE encoder latents
STATS = {
    "mean": torch.Tensor(
        [
            -0.06730895953510081,
            -0.038011381506090416,
            -0.07477820912866141,
            -0.05565264470995561,
            0.012767231469026969,
            -0.04703542746246419,
            0.043896967884726704,
            -0.09346305707025976,
            -0.09918314763016893,
            -0.008729793427399178,
            -0.011931556316503654,
            -0.0321993391887285,
        ]
    ),
    "std": torch.Tensor(
        [
            0.9263795028493863,
            0.9248894543193766,
            0.9393059390890617,
            0.959253732819592,
            0.8244560132752793,
            0.917259975397747,
            0.9294154431013696,
            1.3720942357788521,
            0.881393668867029,
            0.9168315692124348,
            0.9185249279345552,
            0.9274757570805041,
        ]
    ),
}


def dit_latents_to_vae_latents(dit_outputs: torch.Tensor) -> torch.Tensor:
    """Unnormalize latents output by Mochi's DiT to be compatible with VAE.
    Run this on sampled latents before calling the VAE decoder.

    Args:
        latents (torch.Tensor): [B, C_z, T_z, H_z, W_z], float

    Returns:
        torch.Tensor: [B, C_z, T_z, H_z, W_z], float
    """
    mean = STATS["mean"][:, None, None, None]
    std = STATS["std"][:, None, None, None]

    assert dit_outputs.ndim == 5
    assert dit_outputs.size(1) == mean.size(0) == std.size(0)
    return dit_outputs * std.to(dit_outputs) + mean.to(dit_outputs)


def vae_latents_to_dit_latents(vae_latents: torch.Tensor):
    """Normalize latents output by the VAE encoder to be compatible with Mochi's DiT.
    E.g, for fine-tuning or video-to-video.
    """
    mean = STATS["mean"][:, None, None, None]
    std = STATS["std"][:, None, None, None]

    assert vae_latents.ndim == 5
    assert vae_latents.size(1) == mean.size(0) == std.size(0)
    return (vae_latents - mean.to(vae_latents)) / std.to(vae_latents)
