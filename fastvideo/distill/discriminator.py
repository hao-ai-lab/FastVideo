import torch.nn as nn
from diffusers.utils import logging

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class DiscriminatorHead(nn.Module):

    def __init__(self, input_channel, output_channel=1):
        super().__init__()
        inner_channel = 1024
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channel, inner_channel, 1, 1, 0),
            nn.GroupNorm(32, inner_channel),
            nn.LeakyReLU(inplace=True),  # use LeakyReLu instead of GELU shown in the paper to save memory
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(inner_channel, inner_channel, 1, 1, 0),
            nn.GroupNorm(32, inner_channel),
            nn.LeakyReLU(inplace=True),  # use LeakyReLu instead of GELU shown in the paper to save memory
        )

        self.conv_out = nn.Conv2d(inner_channel, output_channel, 1, 1, 0)

    def forward(self, x):
        b, twh, c = x.shape
        t = twh // (30 * 53)
        x = x.view(-1, 30 * 53, c)
        x = x.permute(0, 2, 1)
        x = x.view(b * t, c, 30, 53)
        x = self.conv1(x)
        x = self.conv2(x) + x
        x = self.conv_out(x)
        return x


class Discriminator(nn.Module):

    def __init__(
        self,
        stride=8,
        num_h_per_head=1,
        adapter_channel_dims=[3072],
        total_layers=48,
    ):
        super().__init__()
        adapter_channel_dims = adapter_channel_dims * (total_layers // stride)
        self.stride = stride
        self.num_h_per_head = num_h_per_head
        self.head_num = len(adapter_channel_dims)
        self.heads = nn.ModuleList([
            nn.ModuleList([DiscriminatorHead(adapter_channel) for _ in range(self.num_h_per_head)])
            for adapter_channel in adapter_channel_dims
        ])

    def forward(self, features):
        outputs = []

        def create_custom_forward(module):

            def custom_forward(*inputs):
                return module(*inputs)

            return custom_forward

        assert len(features) == len(self.heads)
        for i in range(0, len(features)):
            for h in self.heads[i]:
                # out = torch.utils.checkpoint.checkpoint(
                #     create_custom_forward(h),
                #     features[i],
                #     use_reentrant=False
                # )
                out = h(features[i])
                outputs.append(out)
        return outputs
