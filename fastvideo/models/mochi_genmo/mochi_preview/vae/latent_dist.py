"""Container for latent space posterior."""

import torch


class LatentDistribution:
    def __init__(self, mean: torch.Tensor, logvar: torch.Tensor):
        """Initialize latent distribution.

        Args:
            mean: Mean of the distribution. Shape: [B, C, T, H, W].
            logvar: Logarithm of variance of the distribution. Shape: [B, C, T, H, W].
        """
        assert mean.shape == logvar.shape
        self.mean = mean
        self.logvar = logvar

    def sample(self, temperature=1.0, generator: torch.Generator = None, noise=None):
        if temperature == 0.0:
            return self.mean

        if noise is None:
            noise = torch.randn(self.mean.shape, device=self.mean.device, dtype=self.mean.dtype, generator=generator)
        else:
            assert noise.device == self.mean.device
            noise = noise.to(self.mean.dtype)

        if temperature != 1.0:
            raise NotImplementedError(f"Temperature {temperature} is not supported.")

        # Just Gaussian sample with no scaling of variance.
        return noise * torch.exp(self.logvar * 0.5) + self.mean

    def mode(self):
        return self.mean
