# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass

from fastvideo.configs.models.vaes.cosmosvae import CosmosVAEConfig


@dataclass
class Gen3CVAEConfig(CosmosVAEConfig):
    """
    GEN3C VAE config placeholder.

    GEN3C uses tokenizer-backed VAE loading logic at runtime, but we keep a
    model-specific config class so pipeline/model configs stay model-scoped.
    """

