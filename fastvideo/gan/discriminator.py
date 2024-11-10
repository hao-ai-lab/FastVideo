
import torch
from torch import nn
import numpy as np
from torch.nn.utils.parametrizations import spectral_norm

class DummyDiscriminator(nn.Module):
    def __init__(self, dim_in, num_layers):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(nn.Linear(dim_in, 1))
        
    def forward(self, features):
        logits = []
        for layer, feature in zip(self.layers, features):
            mean = feature.mean(dim=1)
            logits.append(layer(mean))
        return torch.cat(logits, dim=1)
    

class ResidualBlock(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (self.fn(x) + x) / np.sqrt(2)
    
    
class SpectralConv1d():
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.conv = spectral_norm(nn.Conv1d(*args, **kwargs))
    def forward(self, x):
        return self.conv(x)
    
class BatchNormLocal(nn.Module):
    def __init__(self, num_features: int, affine: bool = True, virtual_bs: int = 8, eps: float = 1e-5):
        super().__init__()
        self.virtual_bs = virtual_bs
        self.eps = eps
        self.affine = affine

        if self.affine:
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.size()

        # Calculate stats.
        mean = x.mean([0, 2], keepdim=True)
        var = x.var([0, 2], keepdim=True, unbiased=False)
        x = (x - mean) / (torch.sqrt(var + self.eps))

        if self.affine:
            x = x * self.weight[None, :, None] + self.bias[None, :, None]

        return x.view(shape)
    
def make_block(channels: int, kernel_size: int) -> nn.Module:
    return nn.Sequential(
        SpectralConv1d(
            channels,
            channels,
            kernel_size = kernel_size,
            padding = kernel_size//2,
            padding_mode = 'circular',
        ),
        BatchNormLocal(channels),
        nn.LeakyReLU(0.2, True),
    )
    
class LADDDiscriminator(nn.Module):
    def __init__(self, channels: int, text_c_dim: int, cmap_dim: int = 64):
        super().__init__()
        self.channels = channels
        self.text_c_dim = text_c_dim
        self.cmap_dim = cmap_dim

        self.main = nn.Sequential(
            make_block(channels, kernel_size=1),
            ResidualBlock(make_block(channels, kernel_size=9))
        )


        self.cmapper = nn.Linear(self.text_c_dim, cmap_dim)
        self.cls = SpectralConv1d(channels, cmap_dim, kernel_size=1, padding=0)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        h = self.main(x)
        out = self.cls(h)

        cmap = self.cmapper(c).unsqueeze(-1)
        out = (out * cmap).sum(1, keepdim=True) * (1 / np.sqrt(self.cmap_dim))

        return out

