from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders import FromOriginalModelMixin, PeftAdapterMixin
from diffusers.models.attention import JointTransformerBlock
from diffusers.models.attention_processor import Attention, AttentionProcessor
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.normalization import AdaLayerNormContinuous
from diffusers.utils import (
    USE_PEFT_BACKEND,
    is_torch_version,
    logging,
    scale_lora_layers,
    unscale_lora_layers,
)
from diffusers.models.embeddings import CombinedTimestepTextProjEmbeddings, PatchEmbed
from diffusers.models.transformers.transformer_2d import Transformer2DModelOutput
from diffusers.models.transformers.transformer_sd3 import SD3Transformer2DModel

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name



class DiscriminatorHead(nn.Module):
    def __init__(self, input_channel, output_channel=1):
        super().__init__()
        inner_channel = 1024
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channel, inner_channel, 1, 1, 0),
            nn.GroupNorm(32, inner_channel),
            nn.LeakyReLU(
                inplace=True
            ),  # use LeakyReLu instead of GELU shown in the paper to save memory
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(inner_channel, inner_channel, 1, 1, 0),
            nn.GroupNorm(32, inner_channel),
            nn.LeakyReLU(
                inplace=True
            ),  # use LeakyReLu instead of GELU shown in the paper to save memory
        )

        self.conv_out = nn.Conv2d(inner_channel, output_channel, 1, 1, 0)

    def forward(self, x):
        b, twh, c = x.shape
        t = twh // (30 * 53)
        x = x.view(-1, 30 *53, c)
        x = x.permute(0, 2, 1)
        x = x.view(b*t, c, 30, 53)
        x = self.conv1(x)
        x = self.conv2(x) + x
        x = self.conv_out(x)
        return x


class Discriminator(nn.Module):

    def __init__(
        self,
        num_h_per_head=1,
        adapter_channel_dims=[3072] * 12,
    ):
        super().__init__()
        self.num_h_per_head = num_h_per_head
        self.head_num = len(adapter_channel_dims)
        self.heads = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        DiscriminatorHead(adapter_channel)
                        for _ in range(self.num_h_per_head)
                    ]
                )
                for adapter_channel in adapter_channel_dims
            ]
        )

    def _forward_teacher(
        self, teacher_transformer, sample, timestep, encoder_hidden_states, encoder_attention_mask
    ):
        features = teacher_transformer(
            sample,
            encoder_hidden_states,
            timestep,
            encoder_attention_mask, # B, L
            output_attn=True,
            return_dict= False
        )[1]
            
        return features

    def _forward(self, features):
        outputs = []
        stride = 4
        assert len(features) // stride == len(self.heads)
        for i in range(0, len(features), stride):
            for h in self.heads[i//stride]:
                outputs.append(h(features[i]))
        for feature, head in zip(features, self.heads):
            for h in head:
                outputs.append(h(feature))
        return outputs

    def forward(self, flag, *args):
        if flag == "d_loss":
            return self.d_loss(*args)
        elif flag == "g_loss":
            return self.g_loss(*args)
        else:
            assert 0, "not supported"

    def d_loss(
        self,
        teacher_transformer,
        sample_fake,
        sample_real,
        timestep,
        encoder_hidden_states,
        encoder_attention_mask,
        weight,
    ):
        loss = 0.0
        # collate sample_fake and sample_real
        sample = torch.cat([sample_fake, sample_real], dim=0)
        with torch.inference_mode():
            features = self._forward_teacher(
                teacher_transformer,
                sample,
                timestep.repeat(sample.shape[0]),
                encoder_hidden_states.repeat(sample.shape[0], 1, 1),
                encoder_attention_mask.repeat(sample.shape[0], 1),
            )
        outputs = self._forward(
            features.clone().detach()
        )
        fake_outputs = [output[: sample_fake.shape[0]] for output in outputs]
        real_outputs = [output[sample_fake.shape[0] :] for output in outputs]
        for fake_output, real_output in zip(fake_outputs, real_outputs):
            loss += (
                torch.mean(weight * torch.relu(fake_output.float() + 1))
                + torch.mean(weight * torch.relu(1 - real_output.float()))
            ) / (self.head_num * self.num_h_per_head)
        return loss

    def g_loss(
        self,
        teacher_transformer,
        sample_fake,
        timestep,
        encoder_hidden_states,
        encoder_attention_mask,
        weight,
    ):
        loss = 0.0
        features = self._forward_teacher(
            teacher_transformer, sample_fake, timestep, encoder_hidden_states, encoder_attention_mask
        )
        fake_outputs = self._forward(
            features,
        )
        for fake_output in fake_outputs:
            loss += torch.mean(weight * torch.relu(1 - fake_output.float())) / (
                self.head_num * self.num_h_per_head
            )
        return loss
