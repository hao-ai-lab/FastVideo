
import torch
from torch import nn

class DummyDiscriminatorHead(nn.Module):
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